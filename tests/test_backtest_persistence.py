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

Everything is synthetic, CPU-only and offline; wandb is disabled and any wandb
call is asserted through a monkeypatched recorder.
"""

import json
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


@pytest.mark.xfail(strict=True, reason="report.html is written by 03.7-09 Task 2")
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
