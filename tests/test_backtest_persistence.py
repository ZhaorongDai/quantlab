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
axis: "equity" on x, "drawdown" on x2, and "monthly_return". Forced
liquidations are NOT drawn on the chart (quick 260916-hro) even for a run that
liquidated, while liquidations.json still records them in full; that pairing
is asserted in one place below. The in-sample range is shaded when the
window overlaps training, there is no benchmark trace, and the note that
short-side returns are optimistic because no borrow cost is modelled still
appears. metrics.json carries the same note and no benchmark key.

Phase 03.8 (D-02): a second note states that the trade metrics are the
position-level view -- one entry-to-flat round trip per symbol, so a partial
trim is not counted as its own closed trade -- and that `whole` carries
`order_count`, the number of fills. This file locks that at the artifact
level: the note verbatim in the page and in metrics.json, the top-level trade
metrics strict-JSON clean, and no nested `whole.positions` block left behind.
The proof that the reported view really IS the positions view (rather than a
copy of the exit-trades one) needs a run that trims without closing, and lives
in tests/test_backtest_engine.py.

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
    # 03.11-09: the record carries BOTH halves of a security's name -- the
    # human `symbol` (the period-correct ticker when a `.crsp_tickers.json`
    # sits beside the price store) and `axis_symbol`, the panel label itself.
    # This fixture store has NO sidecar, which is the control arm: the two are
    # equal and the file reads exactly as it did before the sidecar existed.
    assert set(first) == {
        "symbol",
        "axis_symbol",
        "signal_timestamp",
        "fill_timestamp",
        "price",
    }
    assert first["symbol"] == result.simulation.liquidations[0]["symbol"]
    assert first["axis_symbol"] == first["symbol"]
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
    # Quick 260915-sxx grew the page from two panels to three. Everything this
    # test proves is unchanged: the equity y values are still the persisted
    # ones, drawdown is still value over running max minus one, the axes are
    # still x/x2, the band is still the persisted in-sample range, and the
    # notes still appear.
    # Quick 260915-v6i added the deepest-drawdown triangles. This fixture run
    # draws down (asserted below), so both markers are always present here.
    # Quick 260916-hro removed the "liquidation" markers from the CHART. This
    # fixture run really does liquidate, so their absence here is the point;
    # the surviving JSON half of that split is asserted in its own test below.
    assert set(traces) == {
        "equity",
        "drawdown",
        "monthly_return",
        "deepest_drawdown_valley",
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


def _second_figure_traces(html: str) -> dict[str, dict]:
    """Traces of the SECOND `Plotly.newPlot(` on the page, by name.

    `_report_traces` reads the FIRST one by design -- that is what keeps the
    three-row figure's exact-trace-set lock above meaningful. The monthly
    heatmap (03.8 D-04) lives in its own div after it, so without this parser
    it would be invisible to every persisted-report lock. Mirrors the parser
    in tests/test_backtest_report.py; the two test files carry their own
    parsers by convention rather than importing across test modules.
    """
    first = html.index("Plotly.newPlot(")
    second = html.index("Plotly.newPlot(", first + 1)
    start = html.index("[", second)
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    return {trace["name"]: trace for trace in traces}


def test_report_carries_the_monthly_heatmap_in_a_second_div(overlap_run):
    """03.8 D-04: a real run's page carries the year-by-month heatmap.

    It is a SECOND plotly div, not a trace of the main figure (whose exact
    set is locked above), with 12 month columns and one row per year the run
    spans. It lives INSIDE report.html: the run directory still holds exactly
    the D-24 artifacts, so no sibling file appeared for it.
    """
    run_dir = overlap_run["result"].run_dir
    html = _report_html(overlap_run)

    assert html.count("Plotly.newPlot(") == 2
    heatmap = _second_figure_traces(html)["monthly_return_heatmap"]
    assert heatmap["type"] == "heatmap"
    assert heatmap["x"] == [f"{month:02d}" for month in range(1, 13)]

    timestamps = pd.DatetimeIndex(xr.open_zarr(run_dir / "equity.zarr")["timestamp"].values)
    years = sorted({str(year) for year in timestamps.year})
    assert heatmap["y"] == years
    assert len(heatmap["z"]) == len(years)
    assert all(len(row) == 12 for row in heatmap["z"])
    # The months the bar row shows are exactly the heatmap's non-null cells.
    bars = _report_traces(html)["monthly_return"]
    bar_months = {(label[:4], label[5:7]) for label in bars["x"]}
    cell_months = {
        (heatmap["y"][row], heatmap["x"][col])
        for row, cells in enumerate(heatmap["z"])
        for col, value in enumerate(cells)
        if value is not None
    }
    assert cell_months == bar_months

    assert sorted(p.name for p in run_dir.iterdir()) == D24_ARTIFACTS


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


#: The rows of a real overlapping run whose `out_of_sample - in_sample` cell is
#: a number (03.8 D-01). Written out, never derived from `_delta` or from the
#: metrics mapping: a census that asks the implementation what it should
#: contain agrees with any answer.
DELTA_CARRYING_ROWS = {
    # the 13 returns-accessor floats -- ratios included, per D-01
    "Total Return [%]",
    "Annualized Return [%]",
    "Annualized Volatility [%]",
    "Max Drawdown [%]",
    "Sharpe Ratio",
    "Calmar Ratio",
    "Omega Ratio",
    "Sortino Ratio",
    "Skew",
    "Kurtosis",
    "Tail Ratio",
    "Common Sense Ratio",
    "Value at Risk",
    # the per-slice activity counts and sums
    "order_count",
    "fees_paid",
    "traded_notional",
    "closed_trade_count",
    "open_trade_count",
    "turnover.mean_per_rebalance",
    "turnover.sum",
    "turnover.annualized",
}

#: The slice rows whose delta is a dash by TYPE: timestamps and durations.
DELTA_DASHED_SLICE_ROWS = {"Start", "End", "Period", "Max Drawdown Duration"}


def _flat_keys(block: dict, prefix: str = "") -> set[str]:
    """A metrics block's keys as the report's dotted row names."""
    keys: set[str] = set()
    for key, value in block.items():
        if isinstance(value, dict):
            keys |= _flat_keys(value, f"{prefix}{key}.")
        else:
            keys.add(f"{prefix}{key}")
    return keys


def test_report_delta_census_pins_which_rows_carry_a_number(overlap_run):
    """03.8 D-01 on a real run: exactly which rows get a delta, both halves.

    "Every delta cell is a number or a dash" passes when the predicate returns
    None for everything, and a pure count survives a predicate that differences
    the wrong rows -- so both the carrying set and the dashed set are pinned.

    If the carrying set ever shrinks, do not shrink the literal to match. Two
    causes are legitimate and must be stated here rather than absorbed:
    `turnover.mean_per_rebalance` and `turnover.annualized` are nan for a slice
    with no fill bar (this fixture fills in both slices, so they are finite),
    or the metrics key set itself changed. Anything else is a predicate defect.

    Rows present only in `whole` (the engine's STATS_METRICS names) must dash:
    that is the `None - 3.0` case, which would raise inside the staging
    directory and delete the run. The page having been written proves it did
    not.
    """
    html = _report_html(overlap_run)
    metrics = _strict_json(overlap_run["result"].run_dir / "metrics.json")
    assert html.startswith("<!DOCTYPE html>")
    assert "<th>out_of_sample - in_sample</th>" in html

    # All four cells per row; the first-<td> parser above cannot see column 4.
    # Scoped to the metric table: the dates-and-setup table has the same shape.
    table = html.split('<table class="metrics">', 1)[1].split("</table>", 1)[0]
    rows = re.findall(r"<tr><th>([^<]+)</th>((?:<td>[^<]*</td>)+)</tr>", table)
    cells = {name: re.findall(r"<td>([^<]*)</td>", run) for name, run in rows}
    assert {len(row) for row in cells.values()} == {4}, cells

    carrying = {name for name, row in cells.items() if row[3] != "—"}
    dashed = {name for name, row in cells.items() if row[3] == "—"}
    in_both_slices = _flat_keys(metrics["in_sample"]) & _flat_keys(
        metrics["out_of_sample"]
    )
    whole_only = _flat_keys(metrics["whole"]) - (
        _flat_keys(metrics["in_sample"]) | _flat_keys(metrics["out_of_sample"])
    )

    assert carrying == DELTA_CARRYING_ROWS
    assert dashed & in_both_slices == DELTA_DASHED_SLICE_ROWS
    assert not carrying & DELTA_DASHED_SLICE_ROWS
    assert carrying | DELTA_DASHED_SLICE_ROWS == in_both_slices
    for name in carrying:
        float(cells[name][3])  # a real number, never the token nan

    assert whole_only, "the whole block must carry rows the slices do not"
    for name in whole_only:
        assert cells[name][3] == "—", name

    # D-05: the delta is report-only; metrics.json gained nothing.
    text = (overlap_run["result"].run_dir / "metrics.json").read_text()
    assert "out_of_sample - in_sample" not in text
    for block in ("whole", "in_sample", "out_of_sample"):
        assert not [k for k in _flat_keys(metrics[block]) if "delta" in k.lower()]


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
    """Quick 260916-hro: the triangles land on the curve the run persisted.

    The leaf tests prove the markers are drawn where they are told; this
    proves a REAL run tells them the right place -- both endpoints are values
    of `equity.zarr`, not of some re-derived curve.

    The up triangle is additionally checked against an INDEPENDENTLY derived
    valley: the drawdown curve is recomputed here straight from `equity.zarr`
    and its argmin taken, without asking the engine anything. That is what
    goes red if the marker ever slides back to the bar the drawdown started.
    """
    html = _report_html(overlap_run)
    traces = _report_traces(html)
    equity = xr.open_zarr(overlap_run["result"].run_dir / "equity.zarr")
    value = equity["value"].values

    valley = traces["deepest_drawdown_valley"]
    end = traces["deepest_drawdown_end"]
    assert valley["marker"]["symbol"] == "triangle-up"
    assert end["marker"]["symbol"] == "triangle-down"
    for marker in (valley, end):
        assert len(marker["y"]) == 1
        assert np.isclose(marker["y"][0], value, rtol=1e-12).any(), marker["y"]

    # The deepest bar of the deepest drawdown is the global argmin of
    # `value / running max - 1`, so it can be recovered from the persisted
    # equity alone.
    drawdown = value / np.maximum.accumulate(value) - 1.0
    assert drawdown.min() < 0, "the fixture run must draw down"
    valley_bar = int(np.argmin(drawdown))
    assert str(valley["x"][0])[:10] == _day(equity["timestamp"].values[valley_bar])
    # Non-vacuity: the valley is not the first bar of the window, so this is a
    # real localisation rather than a marker parked at the start by accident.
    assert valley_bar > 0

    # The span is stated in trading days, never as a calendar timedelta: a
    # `Timedelta` repr (`7 days 00:00:00`) would add a `days` that is not
    # part of `trading days`.
    hover = end["hovertemplate"]
    assert "trading days" in hover
    assert hover.count("days") == hover.count("trading days"), hover


def test_the_page_states_the_deepest_drawdown_span_in_words(overlap_run):
    """The picture and the text agree: same bar labels, same units."""
    html = _report_html(overlap_run)
    row = re.search(
        r"<tr><th>Deepest drawdown \(valley to recovery\)</th><td>([^<]*)</td></tr>",
        html,
    )
    assert row is not None, "the dates-and-setup block must state the span"

    text = row.group(1)
    assert "trading days" in text
    assert text.count("days") == text.count("trading days"), text

    # The same two bars the triangles sit on.
    traces = _report_traces(html)
    for name in ("deepest_drawdown_valley", "deepest_drawdown_end"):
        label = str(traces[name]["x"][0])[:10]
        assert label in text, (name, label, text)


def test_the_liquidating_run_drops_the_chart_markers_and_keeps_the_json(overlap_run):
    """D-02 in ONE test: the picture loses the markers, the data survives.

    Split across two tests, either half could be deleted while the other kept
    passing -- and that is precisely the regression worth guarding, because
    the chart trace and liquidations.json were fed from the SAME
    `simulation.liquidations` and only the chart was meant to lose it.
    """
    result = overlap_run["result"]
    traces = _report_traces(_report_html(overlap_run))

    # Non-vacuity: this run really did force a liquidation, so an absent
    # marker cannot be explained away by there being nothing to draw.
    assert result.simulation.liquidations, "the fixture run must liquidate"
    assert "liquidation" not in traces

    # ...and the artifact still carries every record, field for field.
    persisted = _strict_json(result.run_dir / "liquidations.json")
    assert len(persisted) == len(result.simulation.liquidations)
    for stored, live in zip(persisted, result.simulation.liquidations):
        assert set(stored) == {
            "symbol",
            "axis_symbol",
            "signal_timestamp",
            "fill_timestamp",
            "price",
        }
        assert stored["symbol"] == live["symbol"]
        for field in ("fill_timestamp", "signal_timestamp"):
            assert pd.Timestamp(stored[field]) == pd.Timestamp(live[field]), field
        assert stored["price"] == pytest.approx(live["price"], rel=1e-12)

    # The run directory is untouched: still exactly the seven D-24 entries.
    assert sorted(p.name for p in result.run_dir.iterdir()) == D24_ARTIFACTS


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
# Phase 03.8 (D-02): the one-trade-view note and the absent nested block
# --------------------------------------------------------------------------

#: The two TOP-LEVEL trade metrics vectorbt reports as a duration;
#: `to_jsonable` renders a Timedelta as a string and NaT as null, so these two
#: are the one legitimately non-numeric pair among the trade metrics.
POSITION_DURATION_KEYS = {"Avg Winning Trade Duration", "Avg Losing Trade Duration"}

#: vectorbt's display names for the trade-derived metrics, written out here
#: rather than read off the engine: a test that asked the implementation what
#: it should contain would agree with any answer. This mirrors the same literal
#: in tests/test_backtest_engine.py deliberately -- each file states its own
#: expectation instead of importing the other's.
TRADE_METRIC_NAMES = (
    "Total Trades",
    "Total Closed Trades",
    "Total Open Trades",
    "Open Trade PnL",
    "Win Rate [%]",
    "Best Trade [%]",
    "Worst Trade [%]",
    "Avg Winning Trade [%]",
    "Avg Losing Trade [%]",
    "Avg Winning Trade Duration",
    "Avg Losing Trade Duration",
    "Profit Factor",
    "Expectancy",
)


def test_report_and_metrics_carry_the_one_trade_view_note(overlap_run):
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
    # And the new note says which view the trade metrics are, and names the
    # execution-activity count that replaced the lot-level set.
    for mark in ("position level", "round trip", "partial trim", "order_count"):
        assert mark in text, mark


def test_metrics_json_carries_one_trade_view_and_no_nested_positions_block(
    overlap_run,
):
    """The top-level trade metrics strict-parse, and the nested block is gone.

    `test_report_shows_the_positions_rows_beside_the_lot_level_rows` was
    deleted alongside the nested block: there are no `positions.`-prefixed rows
    on the page any more, and the surviving metric-table test already covers
    the top-level rows these numbers are now rendered as.
    """
    metrics = _strict_json(overlap_run["result"].run_dir / "metrics.json")
    whole = metrics["whole"]

    assert "positions" not in whole, sorted(whole)
    # The companion change (CONTEXT item 3): dropping the lot-level set would
    # otherwise leave the whole-window block with no execution-activity count.
    assert isinstance(whole["order_count"], int) and not isinstance(
        whole["order_count"], bool
    )
    assert whole["order_count"] > 0

    for key in TRADE_METRIC_NAMES:
        assert key in whole, key
        value = whole[key]
        if key in POSITION_DURATION_KEYS:
            assert value is None or isinstance(value, str), (key, value)
            continue
        assert value is None or (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ), (key, value)
        if isinstance(value, float):
            assert np.isfinite(value), key


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
