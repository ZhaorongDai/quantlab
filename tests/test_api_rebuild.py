"""A `quantlab.api.backtest` run kept on disk rebuilds and replays from its directory.

A run given `output_dir` (or saved with `BacktestReport.save`) holds its input panels
under `inputs/`, and its recipe names them relative to the run directory, so
`BacktestRun.open(run_dir).rebuild_backtester()` rebuilds the backtester and replaying
the run's weights through `run_weights` writes the same metrics, weights, equity and
data fingerprints, without a warning, even after the directory has moved. A run on a
Zarr-backed dataset writes no `inputs/` and keeps its store paths.

Results are read through `BacktestRun`. The recipe's `config.json` and the `inputs/`
stores are named only where they are the subject: the relative store paths the recipe
records, and the refusal of a relative path without `run_dir` (`_config`).

Everything is synthetic, CPU-only and offline.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

import quantlab.api as qa
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.stock import StockDataset
from quantlab.core.component import rebuild
from quantlab.base.backtest import BaseBacktester
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import write_price_store

N_BARS = 30
SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    """Fail the test if anything starts a W&B run."""
    import wandb

    def _refuse(*args, **kwargs):
        raise AssertionError("a rebuilt API run must never start a W&B run")

    monkeypatch.setattr(wandb, "init", _refuse)


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    yield messages
    logger.remove(handler_id)


def _bars() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=N_BARS)


def _prices(symbols=SYMBOLS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(rng.normal(0, 0.02, (N_BARS, len(symbols))).cumsum(axis=0))
    return pd.DataFrame(
        {
            "timestamp": np.repeat(_bars(), len(symbols)),
            "symbol": list(symbols) * N_BARS,
            "open": (close * rng.uniform(0.99, 1.01, close.shape)).ravel(),
            "close": close.ravel(),
            "volume": rng.uniform(1e5, 1e6, close.size),
        }
    )


def _scores() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "timestamp": np.repeat(_bars(), len(SYMBOLS)),
            "symbol": SYMBOLS * N_BARS,
            "score": rng.normal(size=N_BARS * len(SYMBOLS)),
        }
    )


def _report(output_dir=None, *, benchmark: bool = True):
    """Top-2 of daily-rebalanced scores: the average trade durations are not whole days."""
    return qa.backtest(
        _prices(),
        scores=_scores(),
        top_n=2,
        rebalance_periods=1,
        benchmark=_prices(["QQQ"], seed=7) if benchmark else None,
        output_dir=output_dir,
    )


def _config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _replay(run_dir: Path):
    """Rebuild the run in ``run_dir`` and replay its saved weights."""
    run = BacktestRun.open(run_dir)
    rebuilt = run.rebuild_backtester()
    return rebuilt, rebuilt.run_weights(run.weights())


def _assert_same_run(first_dir: Path, second_dir: Path) -> None:
    """The two run directories hold the same metrics, weights, equity and fingerprints."""
    assert first_dir != second_dir
    first, second = BacktestRun.open(first_dir), BacktestRun.open(second_dir)
    assert second.metrics() == first.metrics()
    assert second.data_fingerprint == first.data_fingerprint
    xr.testing.assert_identical(first.weights(), second.weights())
    xr.testing.assert_identical(first.equity(), second.equity())


# --------------------------------------------------------------------------- inputs


def test_a_run_with_output_dir_holds_its_input_panels(tmp_path):
    report = _report(tmp_path / "runs")
    run_dir = report.raw.run_dir

    assert sorted(p.name for p in (run_dir / "inputs").iterdir()) == [
        "benchmark_dataset.zarr",
        "price_dataset.zarr",
    ]
    config = _config(run_dir)
    assert config["price_dataset"]["zarr_file_path"] == "inputs/price_dataset.zarr"
    assert config["benchmark_dataset"]["zarr_file_path"] == "inputs/benchmark_dataset.zarr"
    assert BacktestRun.open(run_dir).benchmark_source == "the FrameDataset held in memory"
    stored = xr.open_zarr(run_dir / "inputs" / "price_dataset.zarr").load()
    held = FrameDataset(_prices()).panel(_bars()[0], _bars()[-1])
    for name in ("open", "close", "volume"):
        np.testing.assert_array_equal(stored[name].values, held[name].values)


def test_report_save_writes_the_inputs_too(tmp_path):
    run_dir = _report().save(tmp_path / "saved")

    assert (run_dir / "inputs" / "price_dataset.zarr").is_dir()
    assert _config(run_dir)["price_dataset"]["zarr_file_path"] == "inputs/price_dataset.zarr"


def test_without_output_dir_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert _report().raw.run_dir is None
    assert list(tmp_path.iterdir()) == []


def test_a_run_on_a_zarr_backed_dataset_writes_no_inputs(tmp_path):
    prices = StockDataset(write_price_store(tmp_path / "store", n_bars=N_BARS))
    stored = xr.open_zarr(prices.config.zarr_file_path).load()
    weights = xr.zeros_like(stored["adjClose"]).rename("weight")
    config = CrossSectionBacktestConfig(
        price_dataset=prices,
        start_date="2024-01-01",
        end_date=_bars()[-1].strftime("%Y-%m-%d"),
        output_dir=str(tmp_path / "runs"),
        rebalance_periods=1,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
    )

    run_dir = USEquityCrossectionSelectStockVectorBt(config).run_weights(weights).run_dir

    assert not (run_dir / "inputs").exists()
    assert BacktestRun.open(run_dir).rebuild("price_dataset") == prices


# --------------------------------------------------------------------------- rebuild


@pytest.mark.parametrize("benchmark", [True, False], ids=["benchmark", "no_benchmark"])
def test_the_rebuilt_run_replays_the_weights_with_the_same_result(
    tmp_path, warning_messages, benchmark
):
    report = _report(tmp_path / "runs", benchmark=benchmark)
    run_dir = report.raw.run_dir
    duration = pd.Timedelta(report.metrics["whole"]["Avg Winning Trade Duration"])
    assert duration % pd.Timedelta("1D") != pd.Timedelta(0)
    warning_messages.clear()

    rebuilt, again = _replay(run_dir)

    _assert_same_run(run_dir, again.run_dir)
    assert rebuilt.expected_fingerprint == BacktestRun.open(run_dir).data_fingerprint
    assert warning_messages == []


def test_a_moved_run_directory_still_rebuilds(tmp_path, warning_messages):
    first = _report(tmp_path / "runs").raw.run_dir
    moved = Path(shutil.move(first, tmp_path / "elsewhere"))
    warning_messages.clear()

    _, again = _replay(moved)

    _assert_same_run(moved, again.run_dir)
    assert warning_messages == []


def test_a_rebuilt_run_writes_a_self_contained_directory_again(tmp_path):
    first = _report(tmp_path / "runs").raw.run_dir
    second = _replay(first)[1].run_dir
    shutil.rmtree(first)

    third = _replay(second)[1].run_dir

    assert _config(second)["price_dataset"]["zarr_file_path"] == "inputs/price_dataset.zarr"
    _assert_same_run(second, third)


def test_the_rebuilt_config_round_trips_through_the_loader(tmp_path):
    run_dir = _report(tmp_path / "runs").raw.run_dir
    rebuilt = BacktestRun.open(run_dir).rebuild_backtester()

    again = rebuild(rebuilt.get_config(), expected=BaseBacktester)

    assert again.get_config() == rebuilt.get_config()
    assert again.config.price_dataset == rebuilt.config.price_dataset
    assert rebuilt.config.price_dataset.config.zarr_file_path == str(
        run_dir / "inputs" / "price_dataset.zarr"
    )


def test_a_relative_input_path_needs_the_run_directory(tmp_path):
    run_dir = _report(tmp_path / "runs").raw.run_dir

    with pytest.raises(ValueError, match="relative to the run directory.*run_dir="):
        rebuild(_config(run_dir), expected=BaseBacktester)
