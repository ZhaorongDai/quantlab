"""A `quantlab.api.backtest` run kept on disk rebuilds and replays from its directory.

A run given `output_dir` (or saved with `BacktestReport.save`) holds its input panels
under `inputs/`, and its `config.json` names them relative to the run directory, so
`load_backtester_from_config(config, run_dir=run_dir)` rebuilds the backtester and
`run_weights(run_dir / "weights.zarr")` replays the saved weights with the same equity,
metrics and data fingerprints, even after the directory has moved.

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
from quantlab.dataset.memory import FrameDataset
from quantlab.utils.module import load_backtester_from_config

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
    return qa.backtest(
        _prices(),
        scores=_scores(),
        top_n=2,
        rebalance_periods=5,
        benchmark=_prices(["QQQ"], seed=7) if benchmark else None,
        output_dir=output_dir,
    )


def _config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _fingerprint(run_dir: Path) -> dict:
    return json.loads((run_dir / "fingerprint.json").read_text(encoding="utf-8"))


def _fingerprint_warnings(messages: list[str]) -> list[str]:
    return [m for m in messages if "data fingerprint mismatch" in m]


def _assert_same_run(first, second) -> None:
    np.testing.assert_array_equal(
        first.simulation.value.values, second.simulation.value.values
    )
    np.testing.assert_array_equal(
        first.simulation.returns.values, second.simulation.returns.values
    )
    assert json.dumps(first.metrics, sort_keys=True, default=str) == json.dumps(
        second.metrics, sort_keys=True, default=str
    )


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


# --------------------------------------------------------------------------- rebuild


@pytest.mark.parametrize("benchmark", [True, False], ids=["benchmark", "no_benchmark"])
def test_the_rebuilt_run_replays_the_weights_with_the_same_result(
    tmp_path, warning_messages, benchmark
):
    report = _report(tmp_path / "runs", benchmark=benchmark)
    run_dir = report.raw.run_dir

    rebuilt = load_backtester_from_config(_config(run_dir), run_dir=run_dir)
    again = rebuilt.run_weights(run_dir / "weights.zarr")

    _assert_same_run(report.raw, again)
    xr.testing.assert_identical(
        xr.open_zarr(run_dir / "weights.zarr").load(),
        xr.open_zarr(again.run_dir / "weights.zarr").load(),
    )
    assert rebuilt.expected_fingerprint == _fingerprint(run_dir)
    assert _fingerprint(again.run_dir) == _fingerprint(run_dir)
    assert _fingerprint_warnings(warning_messages) == []
    if benchmark:
        np.testing.assert_array_equal(
            report.raw.benchmark.value.values, again.benchmark.value.values
        )


def test_a_moved_run_directory_still_rebuilds(tmp_path, warning_messages):
    report = _report(tmp_path / "runs")
    moved = Path(shutil.move(report.raw.run_dir, tmp_path / "elsewhere"))

    again = load_backtester_from_config(_config(moved), run_dir=moved).run_weights(
        moved / "weights.zarr"
    )

    _assert_same_run(report.raw, again)
    assert _fingerprint_warnings(warning_messages) == []


def test_a_rebuilt_run_writes_a_self_contained_directory_again(tmp_path):
    first = _report(tmp_path / "runs").raw.run_dir
    second = (
        load_backtester_from_config(_config(first), run_dir=first)
        .run_weights(first / "weights.zarr")
        .run_dir
    )
    shutil.rmtree(first)

    third = load_backtester_from_config(_config(second), run_dir=second).run_weights(
        second / "weights.zarr"
    )

    assert _config(second)["price_dataset"]["zarr_file_path"] == "inputs/price_dataset.zarr"
    assert _fingerprint(third.run_dir) == _fingerprint(second)


def test_the_rebuilt_config_round_trips_through_the_loader(tmp_path):
    run_dir = _report(tmp_path / "runs").raw.run_dir
    rebuilt = load_backtester_from_config(_config(run_dir), run_dir=run_dir)

    again = load_backtester_from_config(rebuilt.get_config())

    assert again.get_config() == rebuilt.get_config()
    assert again.config.price_dataset == rebuilt.config.price_dataset
    assert rebuilt.config.price_dataset.config.zarr_file_path == str(
        run_dir / "inputs" / "price_dataset.zarr"
    )


def test_a_relative_input_path_needs_the_run_directory(tmp_path):
    run_dir = _report(tmp_path / "runs").raw.run_dir

    with pytest.raises(ValueError, match="relative to the run directory.*run_dir="):
        load_backtester_from_config(_config(run_dir))


def test_run_weights_refuses_a_missing_weight_store(tmp_path):
    run_dir = _report(tmp_path / "runs").raw.run_dir
    rebuilt = load_backtester_from_config(_config(run_dir), run_dir=run_dir)

    with pytest.raises(FileNotFoundError, match="nowhere.zarr"):
        rebuilt.run_weights(tmp_path / "nowhere.zarr")
