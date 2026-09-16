"""Locks for a backtest run's persistent record (phase 03.7, plan 09).

D-24: every run writes its own directory `output_dir/{class}_{timestamp}/`
holding config.json, weights.zarr, equity.zarr (value, returns),
liquidations.json, metrics.json, report.html and fingerprint.json. An existing
directory is never overwritten, and every JSON artifact is strict JSON (NaN and
inf persisted as null, timestamps as ISO strings).

D-27: the run records a data fingerprint for every dataset it read -- the price
dataset over the fill and valuation columns, each factor's dataset over the
columns it consumes, including the warm-up bars -- and a rebuild that supplies
`expected_fingerprint` warns (and still completes) when a digest, range or
count differs. Datasets get appended to, and Tiingo re-bases adjusted prices
after new dividends, so a re-run must detect that its data changed rather than
silently disagree with the stored run.

Why the fingerprint must be NaN-canonical: NaN has many bit patterns. Two reads
of one store, or two code paths producing "missing", can carry NaNs with
different payload bits that compare as the same missing value but hash
differently byte for byte. Without canonicalizing NaN (and -0.0 vs 0.0) before
hashing, an unchanged store could report a changed digest, and a warning that
fires on unchanged data is a warning people learn to ignore.

D-23 / D-21 / D-08: report.html is one self-contained page around a plotly
figure. It states its dates as TEXT -- the window and bar count, the training
window and the in-sample/out-of-sample ranges, every string byte-identical to
the same run's metrics.json -- carries the whole/in-sample/out-of-sample
metrics as an HTML table, and draws named traces on three rows of a shared time
axis: "equity" (plus "liquidation" markers when the run liquidated) on x,
"drawdown" on x2, and "monthly_return". The in-sample range is shaded when the
window overlaps training, there is no benchmark trace, and the note that
short-side returns are optimistic because no borrow cost is modelled still
appears. metrics.json carries the same note and no benchmark key.

Quick 260915-udx: a second note states that the top-level trade metrics are
vectorbt exit trades (lot level) while the positions-prefixed rows are the
position-level view, and the nested `whole.positions` block those rows are
rendered from reaches both artifacts. This file locks them at the artifact
level -- note verbatim in the page and in metrics.json, block strict-JSON
clean, rows on the page. The proof that the block really IS the positions view
(rather than a second copy of the exit-trades one) needs a run that trims
without closing, and lives in tests/test_backtest_engine.py.

The metric table is rendered from whatever keys the blocks carry at render
time, never from a list written into the report module: the metric set is
moving to vectorbt's own, and the report is written inside the staging
directory of a run, so a report that hardcoded metric names would raise on the
day that lands and take the entire run directory with it. That property is
proved at the leaf level in tests/test_backtest_report.py.

D-28: `use_wandb=False` never calls `wandb.init`; `use_wandb=True` logs the
flattened numeric metrics and the report to a separate `{class}_backtest` run
named after the run directory, then finishes it.

Everything is synthetic, CPU-only and offline; wandb is disabled and any wandb
call is asserted through a monkeypatched recorder.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig
from tests.backtest_fixtures import (
    ADJUSTED_COLUMNS,
    RAW_COLUMNS,
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)
TRAIN_END_BAR = 24
MARKET = USEquityCrossectionSelectStockVectorBt.MARKET

# The shared overlapping run: window bars 20..40 overlap the training window
# (bars 0..24 plus a 1-bar label horizon), and the symbol picked at the first
# rebalance delists at bar 23, so the run carries a real forced liquidation.
OVERLAP_START_BAR = 20
OVERLAP_END_BAR = 40
DELIST_BAR = OVERLAP_START_BAR + 3

D24_ARTIFACTS = [
    "config.json",
    "equity.zarr",
    "fingerprint.json",
    "liquidations.json",
    "metrics.json",
    "report.html",
    "weights.zarr",
]


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture
def warnings_sink():
    """Loguru WARNING-and-above messages emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _strict_json(path: Path):
    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token!r} in {path}")

    return json.loads(path.read_text(), parse_constant=_reject)


def _model_dates() -> dict:
    return dict(
        start_date=_day(BARS[0]),
        end_date=_day(BARS[29]),
        train_start=_day(BARS[0]),
        train_end=_day(BARS[TRAIN_END_BAR]),
        test_start=_day(BARS[TRAIN_END_BAR + 1]),
        test_end=_day(BARS[29]),
    )


def _trained_store(root: Path, **store_kwargs):
    """A seeded price store plus a checkpoint trained on bars 0..TRAIN_END_BAR."""
    dataset_config = write_price_store(root / "store", n_bars=N_BARS, **store_kwargs)
    checkpoint = train_checkpoint(
        make_model(root / "train", dataset_config, **_model_dates())
    )
    return dataset_config, checkpoint


def _backtester(
    root: Path,
    dataset_config,
    checkpoint: Path,
    *,
    tag: str,
    window_start_bar: int,
    window_end_bar: int,
    **overrides,
) -> USEquityCrossectionSelectStockVectorBt:
    """A fresh backtester (own datasets and model objects) in load mode."""
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / f"backtest_{tag}", dataset_config, **_model_dates()),
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(BARS[window_start_bar]),
        end_date=_day(BARS[window_end_bar]),
        output_dir=str(root / "runs"),
        rebalance_periods=5,
        direction="long_only",
        top_n=2,
    )
    kwargs.update(overrides)
    return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(**kwargs))


def _picked_at(root: Path, start_bar: int) -> str:
    """The symbol the fixture model ranks first at `start_bar`.

    The store is seeded, so a probe store with the same seed tells which symbol
    the model picks (the fixture factor scores by past return on adjClose, the
    factor's own input column). Same recipe as
    tests/test_backtest_engine.py::test_end_to_end_delisting_run_records_the_liquidation.
    """
    probe = xr.open_zarr(
        write_price_store(root / "probe", n_bars=N_BARS).zarr_file_path
    ).load()
    close = probe["adjClose"].transpose("timestamp", "symbol").values
    past_return = close[start_bar] / close[start_bar - 1] - 1.0
    return SYMBOLS[int(np.argmax(past_return))]


@pytest.fixture(scope="module")
def overlap_run(tmp_path_factory):
    """One overlapping, liquidating run shared by the read-only artifact locks."""
    root = tmp_path_factory.mktemp("overlap")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WANDB_MODE", "disabled")
        mp.setenv("WANDB_SILENT", "true")
        picked = _picked_at(root, OVERLAP_START_BAR)
        dataset_config, checkpoint = _trained_store(
            root, delist_at={picked: DELIST_BAR}
        )
        backtester = _backtester(
            root,
            dataset_config,
            checkpoint,
            tag="overlap",
            window_start_bar=OVERLAP_START_BAR,
            window_end_bar=OVERLAP_END_BAR,
        )
        config_before_run = backtester.get_config()
        result = backtester.run()
    return {
        "backtester": backtester,
        "result": result,
        "config_before_run": config_before_run,
        "dataset_config": dataset_config,
    }


# --------------------------------------------------------------------------
# D-24: the run directory
# --------------------------------------------------------------------------


def test_run_directory_holds_every_d24_artifact(overlap_run):
    run_dir = overlap_run["result"].run_dir
    assert sorted(p.name for p in run_dir.iterdir()) == D24_ARTIFACTS
    assert not list(run_dir.rglob("*.tmp"))


def test_weights_zarr_round_trips_identically(overlap_run):
    result = overlap_run["result"]
    persisted = xr.open_zarr(result.run_dir / "weights.zarr").load()
    xr.testing.assert_identical(persisted, result.weights)


def test_equity_zarr_has_value_and_returns_on_timestamp(overlap_run):
    result = overlap_run["result"]
    equity = xr.open_zarr(result.run_dir / "equity.zarr").load()
    assert set(equity.data_vars) == {"value", "returns"}
    for name in ("value", "returns"):
        assert equity[name].dims == ("timestamp",)
    np.testing.assert_array_equal(
        equity["value"].values, result.simulation.value.values
    )
    np.testing.assert_array_equal(
        equity["returns"].values, result.simulation.returns.values
    )
    np.testing.assert_array_equal(
        equity["timestamp"].values, result.simulation.value.timestamp.values
    )


def test_every_json_artifact_is_strict_json(overlap_run):
    result = overlap_run["result"]
    run_dir = result.run_dir
    for name in ("config.json", "metrics.json", "liquidations.json", "fingerprint.json"):
        _strict_json(run_dir / name)

    # Not vacuous: the run really liquidated, and each record carries the
    # pd.Timestamp fields that plain json.dump cannot serialize.
    liquidations = _strict_json(run_dir / "liquidations.json")
    assert result.simulation.liquidations, "the fixture run must liquidate"
    assert len(liquidations) == len(result.simulation.liquidations)
    first = liquidations[0]
    assert set(first) == {"symbol", "signal_timestamp", "fill_timestamp", "price"}
    assert first["symbol"] == result.simulation.liquidations[0]["symbol"]
    assert pd.Timestamp(first["fill_timestamp"]) == pd.Timestamp(
        result.simulation.liquidations[0]["fill_timestamp"]
    )


def test_existing_run_directory_is_never_overwritten(tmp_path, monkeypatch):
    dataset_config, checkpoint = _trained_store(tmp_path)
    backtester = _backtester(
        tmp_path,
        dataset_config,
        checkpoint,
        tag="fixed",
        window_start_bar=30,
        window_end_bar=50,
    )
    existing = tmp_path / "runs" / "FixedRunName"
    existing.mkdir(parents=True)
    (existing / "metrics.json").write_text('{"kept": true}')
    monkeypatch.setattr(backtester, "_run_dir_name", lambda: "FixedRunName")

    with pytest.raises(RuntimeError, match="FixedRunName"):
        backtester.run()

    assert sorted(p.name for p in existing.iterdir()) == ["metrics.json"]
    assert (existing / "metrics.json").read_text() == '{"kept": true}'


def test_an_interrupted_persist_leaves_no_run_directory(tmp_path, monkeypatch):
    """Code review WR-08: a run directory exists only once every D-24 artifact is written.

    The old `_report_and_persist` created `output_dir/{class}_{ts}/` and wrote
    config.json (and the zarr stores) before metrics, report and fingerprint.
    An interruption there (Ctrl-C is simulated while report.html is written)
    left a directory with a valid config.json and no metrics. It looked
    finished, and a rebuild would "reproduce" it. After the interruption
    `output_dir` must hold nothing, not even the staging directory. Once the
    report works again, the same backtester writes one complete directory. Red
    on the old code: the half-written directory remains.
    """
    import quantlab.base.backtest as backtest_module

    dataset_config, checkpoint = _trained_store(tmp_path)
    backtester = _backtester(
        tmp_path,
        dataset_config,
        checkpoint,
        tag="interrupted",
        window_start_bar=30,
        window_end_bar=50,
    )
    real_report = backtest_module.write_backtest_report

    def _interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(backtest_module, "write_backtest_report", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        backtester.run()

    runs = tmp_path / "runs"
    assert (sorted(p.name for p in runs.iterdir()) if runs.exists() else []) == []

    monkeypatch.setattr(backtest_module, "write_backtest_report", real_report)
    result = backtester.run()

    assert sorted(p.name for p in runs.iterdir()) == [result.run_dir.name]
    assert sorted(p.name for p in result.run_dir.iterdir()) == D24_ARTIFACTS


# --------------------------------------------------------------------------
# D-27: the data fingerprint
# --------------------------------------------------------------------------


def _panel(values: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"adjClose": (["timestamp", "symbol"], values)},
        coords={
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["AAA", "BBB"],
        },
    )


def test_fingerprint_is_stable_and_nan_canonical(tmp_path):
    from quantlab.utils.fingerprint import dataset_fingerprint

    zarr_path = write_price_store(tmp_path / "store", n_bars=20).zarr_file_path
    columns = [MARKET.fill_price_column, MARKET.valuation_price_column]
    first = dataset_fingerprint(xr.open_zarr(zarr_path), columns)
    second = dataset_fingerprint(xr.open_zarr(zarr_path), columns)
    assert first == second
    assert first["algorithm"] == "sha256"
    assert first["variables"] == sorted(columns)
    assert (first["n_timestamps"], first["n_symbols"]) == (20, len(SYMBOLS))

    payload_nan = np.frombuffer(
        np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64
    )[0]
    canonical = np.array([[1.0, np.nan], [0.0, 2.0]])
    other_bits = np.array([[1.0, payload_nan], [-0.0, 2.0]])
    # Control: the two arrays really differ byte for byte.
    assert canonical.tobytes() != other_bits.tobytes()
    assert (
        dataset_fingerprint(_panel(canonical), ["adjClose"])["digest"]
        == dataset_fingerprint(_panel(other_bits), ["adjClose"])["digest"]
    )

    changed = canonical.copy()
    changed[1, 1] = 2.5
    assert (
        dataset_fingerprint(_panel(changed), ["adjClose"])["digest"]
        != dataset_fingerprint(_panel(canonical), ["adjClose"])["digest"]
    )

    with pytest.raises(ValueError, match="notAColumn"):
        dataset_fingerprint(_panel(canonical), ["adjClose", "notAColumn"])


def test_fingerprint_json_covers_price_and_factor_datasets(overlap_run):
    result = overlap_run["result"]
    assert (result.run_dir / "fingerprint.json").is_file(), sorted(
        p.name for p in result.run_dir.iterdir()
    )
    from quantlab.utils.fingerprint import dataset_fingerprint

    fingerprints = _strict_json(result.run_dir / "fingerprint.json")
    assert set(fingerprints) == {"price_dataset", "factor[0]:PastReturnFactor"}

    entry_keys = {
        "algorithm",
        "digest",
        "variables",
        "start",
        "end",
        "n_timestamps",
        "n_symbols",
    }
    for entry in fingerprints.values():
        assert set(entry) == entry_keys
        assert entry["algorithm"] == "sha256"
        assert entry["n_symbols"] == len(SYMBOLS)

    price = fingerprints["price_dataset"]
    columns = [MARKET.fill_price_column, MARKET.valuation_price_column]
    assert price["variables"] == sorted(columns)
    assert price["n_timestamps"] == OVERLAP_END_BAR - OVERLAP_START_BAR + 1
    # The digest is over exactly the window's fill and valuation columns.
    store = xr.open_zarr(overlap_run["dataset_config"].zarr_file_path).sel(
        timestamp=slice(_day(BARS[OVERLAP_START_BAR]), _day(BARS[OVERLAP_END_BAR]))
    )
    assert price["digest"] == dataset_fingerprint(store, columns)["digest"]

    factor = fingerprints["factor[0]:PastReturnFactor"]
    # A Polars factor consumes the whole lazyframe: every data variable counts.
    assert factor["variables"] == sorted(ADJUSTED_COLUMNS + RAW_COLUMNS)
    # The factor range includes the warm-up bars before the window start (D-27).
    assert pd.Timestamp(factor["start"]) < pd.Timestamp(price["start"])
    assert pd.Timestamp(factor["end"]) == pd.Timestamp(price["end"])


def test_get_config_carries_the_data_fingerprint_after_a_run(overlap_run):
    assert "data_fingerprint" not in overlap_run["config_before_run"]
    result = overlap_run["result"]
    config = overlap_run["backtester"].get_config()
    fingerprints = _strict_json(result.run_dir / "fingerprint.json")
    assert json.loads(json.dumps(config["data_fingerprint"])) == fingerprints
    assert _strict_json(result.run_dir / "config.json")["data_fingerprint"] == fingerprints


def _fingerprint_warnings(messages: list[str]) -> list[str]:
    return [m for m in messages if "fingerprint" in m]


def test_matching_expected_fingerprint_logs_no_warning(tmp_path, warnings_sink):
    dataset_config, checkpoint = _trained_store(tmp_path)
    first = _backtester(
        tmp_path, dataset_config, checkpoint, tag="a", window_start_bar=30, window_end_bar=50
    ).run()
    expected = _strict_json(first.run_dir / "fingerprint.json")

    rebuild = _backtester(
        tmp_path, dataset_config, checkpoint, tag="b", window_start_bar=30, window_end_bar=50
    )
    rebuild.expected_fingerprint = expected
    warnings_sink.clear()
    rebuild.run()
    assert _fingerprint_warnings(warnings_sink) == []


def test_changed_store_logs_a_fingerprint_warning_and_completes(tmp_path, warnings_sink):
    dataset_config, checkpoint = _trained_store(tmp_path)
    first = _backtester(
        tmp_path, dataset_config, checkpoint, tag="a", window_start_bar=30, window_end_bar=50
    ).run()
    expected = _strict_json(first.run_dir / "fingerprint.json")

    # A retroactive re-base: one adjusted close inside the window changes.
    store = xr.open_zarr(dataset_config.zarr_file_path).load()
    for variable in store.variables.values():
        variable.encoding = {}
    column = MARKET.valuation_price_column
    store[column][35, 0] = float(store[column][35, 0]) * 1.01
    store.to_zarr(dataset_config.zarr_file_path, mode="w")

    rebuild = _backtester(
        tmp_path, dataset_config, checkpoint, tag="b", window_start_bar=30, window_end_bar=50
    )
    rebuild.expected_fingerprint = expected
    warnings_sink.clear()
    result = rebuild.run()

    assert result is not None and result.run_dir.exists()
    price_warnings = [
        m for m in _fingerprint_warnings(warnings_sink) if "price_dataset" in m
    ]
    assert len(price_warnings) == 1, warnings_sink
    assert "digest" in price_warnings[0]


# --------------------------------------------------------------------------
# D-27 / code review WR-05: fingerprint what predictions and training really read
# --------------------------------------------------------------------------


def _read_strategy_backtester(root, dataset_config, checkpoint, factor_store, *, tag):
    """Load mode whose model reads its features from a saved factor STORE."""
    from quantlab.base.config import MLConfig, PolarsFactorConfig
    from tests.backtest_fixtures import (
        FirstFeatureHead,
        ForwardReturnLabel,
        PastReturnFactor,
    )

    factor = PastReturnFactor(
        PolarsFactorConfig(
            window=5,
            dataset=make_stock_dataset(dataset_config),
            file_path=str(factor_store),
            kwargs={"n": 1},
        )
    )
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            window=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": 1},
        )
    )
    model = FirstFeatureHead(
        MLConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(root / f"backtest_{tag}" / "models"),
            factor_data_strategy="read",
            label_data_strategy="cal",
            val_size=0.0,
            **_model_dates(),
        )
    )
    return _backtester(
        root, dataset_config, checkpoint, tag=tag,
        window_start_bar=30, window_end_bar=50, model=model,
    )


def _shift_value(zarr_path, variable: str, timestamp, symbol: str, shift: float) -> None:
    """Rewrite one value of a Zarr store, as a factor recompute or a re-base would."""
    ds = xr.open_zarr(zarr_path).load()
    for item in ds.variables.values():
        item.encoding = {}
    point = dict(timestamp=timestamp, symbol=symbol)
    before = float(ds[variable].loc[point])
    assert np.isfinite(before), (variable, point)
    ds[variable].loc[point] = before + shift
    ds.to_zarr(zarr_path, mode="w")


def test_read_strategy_fingerprints_the_factor_store_predictions_came_from(
    tmp_path, warnings_sink
):
    """Code review WR-05: under `factor_data_strategy="read"` the factor STORE is fingerprinted.

    Read-strategy features come from the saved factor store, not from the raw
    dataset. The old code hashed only the raw dataset behind each factor, so a
    recomputed or edited factor store changed the predictions and weights
    while the stored fingerprint still matched, with no warning. D-27 exists to
    catch exactly that. The run must record `factor_store[0]:PastReturnFactor`
    and warn on it, and only on it, after one stored factor value changes. Red
    on the old code: no such key.
    """
    from quantlab.base.config import PolarsFactorConfig
    from tests.backtest_fixtures import PastReturnFactor

    dataset_config, checkpoint = _trained_store(tmp_path)
    factor_store = tmp_path / "factors" / "past_ret.zarr"
    PastReturnFactor(
        PolarsFactorConfig(
            window=5,
            dataset=make_stock_dataset(dataset_config),
            file_path=str(factor_store),
            kwargs={"n": 1},
        )
    ).cal().save(mode="w")

    first = _read_strategy_backtester(
        tmp_path, dataset_config, checkpoint, factor_store, tag="a"
    ).run()
    expected = _strict_json(first.run_dir / "fingerprint.json")
    assert [k for k in expected if k.startswith("factor_store[")] == [
        "factor_store[0]:PastReturnFactor"
    ], sorted(expected)

    _shift_value(factor_store, "past_ret_1", BARS[35], SYMBOLS[0], shift=0.01)
    rebuild = _read_strategy_backtester(
        tmp_path, dataset_config, checkpoint, factor_store, tag="b"
    )
    rebuild.expected_fingerprint = expected
    warnings_sink.clear()
    result = rebuild.run()

    assert result.run_dir.exists()
    changed = _fingerprint_warnings(warnings_sink)
    store_warnings = [m for m in changed if "factor_store[0]:PastReturnFactor" in m]
    assert len(store_warnings) == 1, warnings_sink
    assert "digest" in store_warnings[0]
    assert not any("'price_dataset'" in m or "'factor[0]" in m for m in changed), changed


def test_train_mode_fingerprints_the_data_the_model_trained_on(tmp_path, warnings_sink):
    """Code review WR-05: train mode fingerprints each factor and label dataset it trains on.

    The model trains on bars 0..29, and the backtest window 30..50 warms up
    from bar 25. A retroactive re-base at bar 10 lies inside training only.
    The old fingerprint covered just the price window and the factors'
    warm-up plus window, so the rebuild silently retrained a different model.
    The run must record `train_factor[0]:PastReturnFactor` and
    `train_label[0]:ForwardReturnLabel`, and the rebuild must warn on both
    and on neither backtest-window key. Red on the old code: no such keys.
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)

    def _train_mode(tag: str) -> USEquityCrossectionSelectStockVectorBt:
        return USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=make_stock_dataset(dataset_config),
                model=make_model(
                    tmp_path / f"backtest_{tag}", dataset_config, **_model_dates()
                ),
                model_mode="train",
                start_date=_day(BARS[30]),
                end_date=_day(BARS[50]),
                output_dir=str(tmp_path / "runs"),
                rebalance_periods=5,
                direction="long_only",
                top_n=2,
            )
        )

    first = _train_mode("a").run()
    expected = _strict_json(first.run_dir / "fingerprint.json")
    assert {
        "train_factor[0]:PastReturnFactor",
        "train_label[0]:ForwardReturnLabel",
    } <= set(expected), sorted(expected)

    _shift_value(
        dataset_config.zarr_file_path,
        MARKET.valuation_price_column,
        BARS[10],
        SYMBOLS[0],
        shift=0.5,
    )
    rebuild = _train_mode("b")
    rebuild.expected_fingerprint = expected
    warnings_sink.clear()
    rebuild.run()

    changed = _fingerprint_warnings(warnings_sink)
    assert any("train_factor[0]:PastReturnFactor" in m for m in changed), warnings_sink
    assert any("train_label[0]:ForwardReturnLabel" in m for m in changed), warnings_sink
    assert not any("'price_dataset'" in m or "'factor[0]" in m for m in changed), changed


# --------------------------------------------------------------------------
# D-23 / D-21 / D-08: the report and the short-side note
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def disjoint_run(tmp_path_factory):
    """One run whose window (bars 30..50) lies after the training window."""
    root = tmp_path_factory.mktemp("disjoint")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WANDB_MODE", "disabled")
        mp.setenv("WANDB_SILENT", "true")
        dataset_config, checkpoint = _trained_store(root)
        backtester = _backtester(
            root,
            dataset_config,
            checkpoint,
            tag="disjoint",
            window_start_bar=30,
            window_end_bar=50,
        )
        result = backtester.run()
    return {"backtester": backtester, "result": result}


def _report_html(run: dict) -> str:
    path = run["result"].run_dir / "report.html"
    assert path.is_file(), sorted(p.name for p in path.parent.iterdir())
    return path.read_text()


def _report_traces(html: str) -> dict[str, dict]:
    """The trace list plotly embeds as `Plotly.newPlot(id, [traces], layout, ...)`, by name.

    plotly 5.x serializes numpy arrays as plain JSON lists, so the persisted y
    values can be decoded straight from the page.
    """
    start = html.index("[", html.index("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    return {trace["name"]: trace for trace in traces}


def test_report_has_equity_and_drawdown_and_shades_the_in_sample_range(overlap_run):
    html = _report_html(overlap_run)
    metrics = _strict_json(overlap_run["result"].run_dir / "metrics.json")
    in_sample_range = metrics["in_sample_range"]
    assert in_sample_range is not None, "the fixture window must overlap training"

    assert '"name":"equity"' in html
    assert '"name":"drawdown"' in html
    # The curves are the persisted run's values: drawdown = value / running max - 1.
    traces = _report_traces(html)
    # Quick 260915-sxx grew the page from two panels to three. The fixture run
    # liquidates, so it carries the markers too. Everything else this test
    # proves is unchanged: the equity y values are still the persisted ones,
    # drawdown is still value over running max minus one, the axes are still
    # x/x2, the band is still the persisted in-sample range, and the notes
    # still appear.
    # Quick 260915-v6i added the deepest-drawdown triangles. This fixture run
    # draws down (asserted below), so both markers are always present here.
    assert set(traces) == {
        "equity",
        "drawdown",
        "monthly_return",
        "liquidation",
        "deepest_drawdown_start",
        "deepest_drawdown_end",
    }
    value = xr.open_zarr(overlap_run["result"].run_dir / "equity.zarr")["value"].values
    np.testing.assert_allclose(traces["equity"]["y"], value, rtol=1e-12)
    expected_drawdown = value / np.maximum.accumulate(value) - 1.0
    assert expected_drawdown.min() < 0, "the fixture run must draw down"
    np.testing.assert_allclose(traces["drawdown"]["y"], expected_drawdown, atol=1e-12)
    assert traces["drawdown"]["xaxis"] == "x2" and traces["equity"]["xaxis"] == "x"
    assert '"type":"rect"' in html
    # The shaded band is the persisted in-sample range, not an assumed one.
    assert f'"x0":"{in_sample_range[0]}"' in html
    assert f'"x1":"{in_sample_range[1]}"' in html
    notes = overlap_run["backtester"]._report_notes()
    assert notes
    for note in notes:
        assert note in html


def test_report_carries_the_metric_table_and_the_axis_toggle(overlap_run):
    """Quick 260915-sxx: the page carries the numbers, not just the picture.

    The metric table is rendered from whatever the metrics mapping carries, so
    this asserts the persisted numbers reached the page rather than asserting
    a particular metric list. The log button is what makes a curve that
    compounded by orders of magnitude readable.
    """
    html = _report_html(overlap_run)
    metrics = _strict_json(overlap_run["result"].run_dir / "metrics.json")

    assert "<h2>Metrics</h2>" in html
    for block in ("whole", "in_sample", "out_of_sample"):
        assert f"<th>{block}</th>" in html

    # Every finite number in the whole block reached the page, compared by
    # value after parsing the cell back -- not by re-formatting it here.
    rendered = dict(re.findall(r"<tr><th>([^<]+)</th><td>([^<]*)</td>", html))
    checked = 0
    for key, value in metrics["whole"].items():
        if not isinstance(value, float) or not np.isfinite(value):
            continue
        assert key in rendered, (key, sorted(rendered))
        assert float(rendered[key]) == pytest.approx(value, rel=1e-5)
        checked += 1
    assert checked >= 5, "the whole block must carry several finite numbers"

    # The nested turnover group is flattened to dotted paths by walking it.
    assert "turnover.sum" in rendered

    assert '"yaxis.type":"log"' in html and '"yaxis.type":"linear"' in html


def test_report_states_the_window_and_split_dates_as_text(overlap_run):
    """Quick 260915-sxx: the page states its dates in words, not only as a band.

    Before this task the report carried no date anywhere: the reader saw a
    shaded region and had to open metrics.json separately to learn which
    window it covered, where training ended, and which bars were in-sample.
    Every date on the page is the string the run's OWN metrics.json carries --
    the summary reads the persisted bar labels instead of reformatting
    timestamps, so the page and the metrics cannot drift apart.
    """
    html = _report_html(overlap_run)
    metrics = _strict_json(overlap_run["result"].run_dir / "metrics.json")
    value = xr.open_zarr(overlap_run["result"].run_dir / "equity.zarr")["value"]

    n_bars = value.sizes["timestamp"]
    first = _day(value.timestamp.values[0])
    last = _day(value.timestamp.values[-1])
    assert f"{first} .. {last} ({n_bars} bars)" in html

    training_window = metrics["training_window"]
    assert training_window is not None, "the fixture model must record train dates"
    assert f"{training_window[0]} .. {training_window[1]}" in html

    in_sample_range = metrics["in_sample_range"]
    assert in_sample_range is not None, "the fixture window must overlap training"
    assert f"{in_sample_range[0]} .. {in_sample_range[1]}" in html

    out_of_sample_ranges = metrics["out_of_sample_ranges"]
    assert out_of_sample_ranges, "the fixture window must have out-of-sample bars"
    for start, end in out_of_sample_ranges:
        assert f"{start} .. {end}" in html

    # The setup those numbers were produced under is on the page too.
    config = overlap_run["backtester"].config
    assert f"<td>{config.model_mode}</td>" in html
    assert f"<td>{config.direction}</td>" in html


def test_report_without_in_sample_overlap_has_no_shaded_range(disjoint_run):
    html = _report_html(disjoint_run)
    assert disjoint_run["result"].metrics["in_sample_range"] is None
    # Control: the page really carries the curves.
    assert '"name":"equity"' in html
    assert '"type":"rect"' not in html


def test_report_marks_the_deepest_drawdown_on_the_persisted_equity_curve(overlap_run):
    """Quick 260915-v6i: the triangles land on the curve the run persisted.

    The leaf tests prove the markers are drawn where they are told; this
    proves a REAL run tells them the right place -- both endpoints are values
    of `equity.zarr`, not of some re-derived curve.
    """
    html = _report_html(overlap_run)
    traces = _report_traces(html)
    value = xr.open_zarr(overlap_run["result"].run_dir / "equity.zarr")["value"].values

    start = traces["deepest_drawdown_start"]
    end = traces["deepest_drawdown_end"]
    assert start["marker"]["symbol"] == "triangle-up"
    assert end["marker"]["symbol"] == "triangle-down"
    for marker in (start, end):
        assert len(marker["y"]) == 1
        assert np.isclose(marker["y"][0], value, rtol=1e-12).any(), marker["y"]

    # The span is stated in trading days, never as a calendar timedelta: a
    # `Timedelta` repr (`7 days 00:00:00`) would add a `days` that is not
    # part of `trading days`.
    hover = end["hovertemplate"]
    assert "trading days" in hover
    assert hover.count("days") == hover.count("trading days"), hover


def test_the_page_states_the_deepest_drawdown_span_in_words(overlap_run):
    """The picture and the text agree: same bar labels, same units."""
    html = _report_html(overlap_run)
    row = re.search(r"<tr><th>Deepest drawdown span</th><td>([^<]*)</td></tr>", html)
    assert row is not None, "the dates-and-setup block must state the span"

    text = row.group(1)
    assert "trading days" in text
    assert text.count("days") == text.count("trading days"), text

    # The same two bars the triangles sit on.
    traces = _report_traces(html)
    for name in ("deepest_drawdown_start", "deepest_drawdown_end"):
        label = str(traces[name]["x"][0])[:10]
        assert label in text, (name, label, text)


def test_report_and_metrics_carry_no_benchmark(overlap_run):
    html = _report_html(overlap_run)
    assert '"name":"equity"' in html
    assert '"name":"benchmark"' not in html
    assert "benchmark" not in _strict_json(overlap_run["result"].run_dir / "metrics.json")
    assert "benchmark" not in overlap_run["result"].metrics


def test_metrics_json_carries_the_short_side_note(overlap_run):
    metrics = _strict_json(overlap_run["result"].run_dir / "metrics.json")
    assert "notes" in metrics, sorted(metrics)
    notes = metrics["notes"]
    assert notes == overlap_run["backtester"]._report_notes()
    text = " ".join(notes).lower()
    assert "short" in text and "borrow" in text and "optimistic" in text


# --------------------------------------------------------------------------
# quick 260915-udx: the lot-level vs position-level note and the nested block
# --------------------------------------------------------------------------

#: The two position-level metrics vectorbt reports as a duration; `to_jsonable`
#: renders a Timedelta as a string and NaT as null, so these two are the one
#: pair in the block that is legitimately not a number.
POSITION_DURATION_KEYS = {"Avg Winning Trade Duration", "Avg Losing Trade Duration"}


def test_report_and_metrics_carry_the_lot_versus_position_note(overlap_run):
    """Both notes reach both artifacts, and the new one survives HTML escaping.

    Every note is rendered through `html.escape`, so a note containing any of
    the five characters escaping rewrites would appear in the page in escaped
    form and NOT verbatim. That is asserted as a property of the note text
    rather than left to whoever edits it next.
    """
    html = _report_html(overlap_run)
    metrics = _strict_json(overlap_run["result"].run_dir / "metrics.json")
    notes = overlap_run["backtester"]._report_notes()

    assert len(notes) >= 2, notes
    assert metrics["notes"] == notes
    for note in notes:
        assert note in html, note
        assert not set(note) & set("<>&\"'"), note

    text = " ".join(notes).lower()
    # The base short-side note survives alongside the new one.
    assert "short" in text and "borrow" in text and "optimistic" in text
    # And the new note says which set is which.
    for mark in ("exit trades", "lot level", "position level"):
        assert mark in text, mark


def test_metrics_json_carries_the_nested_positions_block(overlap_run):
    """The nested block strict-parses: no NaN or Infinity token, no stray type."""
    metrics = _strict_json(overlap_run["result"].run_dir / "metrics.json")
    positions = metrics["whole"]["positions"]

    assert isinstance(positions, dict) and positions
    for key, value in positions.items():
        if key in POSITION_DURATION_KEYS:
            assert value is None or isinstance(value, str), (key, value)
            continue
        assert value is None or (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ), (key, value)
        if isinstance(value, float):
            assert np.isfinite(value), key


def test_report_shows_the_positions_rows_beside_the_lot_level_rows(overlap_run):
    """The page carries the position-level numbers under a distinguishable name.

    This is what makes the distinction legible without opening the source: the
    reader sees `Win Rate [%]` and `positions.Win Rate [%]` as separate rows.
    The rows come from the generic dotted-path flattening, so no report edit
    was needed -- and that is exactly why it is worth asserting here.
    """
    html = _report_html(overlap_run)
    metrics = _strict_json(overlap_run["result"].run_dir / "metrics.json")
    rendered = dict(re.findall(r"<tr><th>([^<]+)</th><td>([^<]*)</td>", html))

    checked = 0
    for key, value in metrics["whole"]["positions"].items():
        row = f"positions.{key}"
        assert row in rendered, (row, sorted(rendered))
        # The lot-level twin is on the page too, under its bare name.
        assert key in rendered, key
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert float(rendered[row]) == pytest.approx(value, rel=1e-5)
            checked += 1
    assert checked >= 3, "the positions block must carry several finite numbers"


# --------------------------------------------------------------------------
# D-28: optional wandb
# --------------------------------------------------------------------------


class _Summary:
    def __init__(self):
        self.data: dict = {}

    def update(self, values: dict):
        self.data.update(values)


class _RecordingRun:
    def __init__(self):
        self.summary = _Summary()
        self.logged: list[dict] = []
        self.finished = 0

    def log(self, values: dict):
        self.logged.append(values)

    def finish(self):
        self.finished += 1


class _RecordingHtml:
    def __init__(self, data):
        self.data = data


def test_wandb_is_never_initialized_when_disabled(tmp_path, monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("wandb.init must not be called when use_wandb is False")

    # Patched only after training: the model layer's own train() legitimately
    # opens a wandb run; this lock is about the backtester alone.
    dataset_config, checkpoint = _trained_store(tmp_path)
    monkeypatch.setattr("wandb.init", _refuse)
    backtester = _backtester(
        tmp_path, dataset_config, checkpoint, tag="off", window_start_bar=30, window_end_bar=50
    )
    assert backtester.config.use_wandb is False
    result = backtester.run()
    assert result.run_dir.exists()


def test_wandb_logs_metrics_and_report_to_a_separate_backtest_run(tmp_path, monkeypatch):
    init_calls: list[dict] = []
    runs: list[_RecordingRun] = []

    def _init(**kwargs):
        init_calls.append(kwargs)
        run = _RecordingRun()
        runs.append(run)
        return run

    # Patched only after training: the model layer's own train() opens its own
    # wandb run, which is not the backtest run under test.
    dataset_config, checkpoint = _trained_store(tmp_path)
    monkeypatch.setattr("wandb.init", _init)
    monkeypatch.setattr("wandb.Html", _RecordingHtml)
    backtester = _backtester(
        tmp_path,
        dataset_config,
        checkpoint,
        tag="on",
        window_start_bar=OVERLAP_START_BAR,
        window_end_bar=OVERLAP_END_BAR,
        use_wandb=True,
    )
    result = backtester.run()

    assert len(init_calls) == 1
    call = init_calls[0]
    assert call["project"] == "USEquityCrossectionSelectStockVectorBt_backtest"
    assert call["name"] == result.run_dir.name
    json.dumps(call["config"], allow_nan=False)  # the run config is strict JSON

    (run,) = runs
    summary = run.summary.data
    assert summary, "the summary must receive metrics"
    for key, value in summary.items():
        assert key.split("/")[0] in {"whole", "in_sample", "out_of_sample"}, key
        assert isinstance(value, (int, float)) and not isinstance(value, bool), key
        assert np.isfinite(value), key
    assert any(key.startswith("in_sample/") for key in summary)
    assert any(key.startswith("out_of_sample/") for key in summary)
    assert "whole/turnover/sum" in summary
    assert summary["whole/Total Return [%]"] == pytest.approx(
        result.metrics["whole"]["Total Return [%]"]
    )

    reports = [entry["report"] for entry in run.logged if "report" in entry]
    assert len(reports) == 1
    assert reports[0].data == (result.run_dir / "report.html").read_text()
    assert run.finished == 1
