"""Byte lock on a backtest run's ``report.html`` and ``metrics.json`` (#115).

The inputs of the report became public functions in
``quantlab.utils.backtest_report`` (``report_summary``, ``report_windows``,
``report_chart_inputs``, ``report_portfolio_inputs``) so another executor can
write a report in quantlab's format. The backtester builds its own report
through them, and its pages and metrics must not change by a byte. What is
locked here: ``run()`` (long-only with a benchmark, long-short with a
delisting and no benchmark), ``run_weights()`` with and without a benchmark,
and ``run_cv()`` with and without a benchmark each write a ``report.html``
and a ``metrics.json`` whose SHA-256, after replacing the three things that
differ between two identical runs (the temporary directory, the run
directory's timestamped name and plotly's random div ids), is the one
recorded below, captured on the code before #115.

The bytes depend on the installed plotly, numpy and vectorbt and were
captured on macOS; a version bump that changes them shows up here first.
Beside the hashes, the recipe ``report_windows`` documents for building
``run_cv()`` fold rows from ``metrics.json`` is checked against the bars each
fold traded.

A change that is meant to alter the page or the metrics updates the hashes;
the failure message names the scenario and file, and the normalized text is
written next to the run so it can be diffed.

Everything is synthetic, CPU-only and offline.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.utils.backtest_stats import bar_label
from tests.backtest_fixtures import (
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
WINDOW_START = 30
WINDOW_END = 55

CV_BARS = 80
CV_TRAIN_PERIODS = 30
CV_FIRST_TEST_BAR = 30
CV_LAST_TEST_BAR = 77

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_RUN_NAME = re.compile(r"\b(\w+?)_\d{8}_\d{6}_\d{6}\b")


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _model_dates(bars, last: int) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[last]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[last]),
    )


def _single_run_config(root: Path, *, benchmark: bool, **overrides) -> CrossSectionBacktestConfig:
    """Load-mode config over a checkpoint trained on bars 0-24, window bars 30-55."""
    bars = pd.bdate_range("2024-01-01", periods=N_BARS)
    delist_at = overrides.pop("delist_at", None)
    dataset_config = write_price_store(root / "store", n_bars=N_BARS, delist_at=delist_at)
    dates = _model_dates(bars, 29)
    checkpoint = train_checkpoint(make_model(root / "train", dataset_config, **dates))
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / "backtest", dataset_config, **dates),
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(bars[WINDOW_START]),
        end_date=_day(bars[WINDOW_END]),
        output_dir=str(root / "runs"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        benchmark_dataset=(
            make_stock_dataset(
                write_price_store(root / "benchmark", symbols=["QQQ"], seed=7, n_bars=N_BARS)
            )
            if benchmark
            else None
        ),
    )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


def _run_long_only_benchmark(root: Path) -> Path:
    return USEquityCrossectionSelectStockVectorBt(
        _single_run_config(root, benchmark=True)
    ).run().run_dir


def _run_long_short_delisting(root: Path) -> Path:
    config = _single_run_config(
        root,
        benchmark=False,
        constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=2)),
        rebalance_periods=3,
        fees=0.002,
        delist_at={"CCC": 42},
    )
    return USEquityCrossectionSelectStockVectorBt(config).run().run_dir


def _weights(bars: np.ndarray, symbols: list[str]) -> xr.DataArray:
    """Hold rows except every fifth bar, which rotates a long-short book."""
    rows = np.full((bars.size, len(symbols)), np.nan)
    for i in range(0, bars.size, 5):
        row = np.zeros(len(symbols))
        row[(i // 5) % len(symbols)] = 0.5
        row[(i // 5 + 2) % len(symbols)] = 0.3
        row[(i // 5 + 4) % len(symbols)] = -0.2
        rows[i] = row
    return xr.DataArray(
        rows, dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": symbols}
    )


def _run_weights(root: Path, *, benchmark: bool) -> Path:
    config = _single_run_config(
        root, benchmark=benchmark, model=None, model_mode=None, checkpoint=None
    )
    backtester = USEquityCrossectionSelectStockVectorBt(config)
    prices = xr.open_zarr(config.price_dataset.config.zarr_file_path)
    bars = prices.timestamp.values[WINDOW_START : WINDOW_END + 1]
    return backtester.run_weights(_weights(bars, list(prices.symbol.values))).run_dir


@pytest.fixture(scope="module")
def cv_project(tmp_path_factory):
    """One real ``train_cv`` run: 8 folds of 6 test bars over bars 30..77."""
    root = tmp_path_factory.mktemp("lock_cv_project")
    dataset_config = write_price_store(root, n_bars=CV_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    model = make_model(root / "train", dataset_config, n_forward_periods=2, **_cv_dates(bars))
    model.collect()
    model.train_cv(train_periods=CV_TRAIN_PERIODS)
    (manifest,) = sorted((root / "train" / "models").rglob("cv_folds.json"))
    return root, dataset_config, bars, manifest.parent


def _cv_dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[CV_BARS - 1]),
        train_start=_day(bars[0]),
        train_end=_day(bars[CV_TRAIN_PERIODS - 1]),
        test_start=_day(bars[CV_TRAIN_PERIODS]),
        test_end=_day(bars[CV_BARS - 1]),
    )


def _run_cv(root: Path, cv_project, *, benchmark: bool) -> Path:
    project_root, dataset_config, bars, project_dir = cv_project
    config = CrossSectionBacktestConfig(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / "backtest", dataset_config, n_forward_periods=2, **_cv_dates(bars)),
        model_mode="load",
        cv_project_dir=str(project_dir),
        start_date=_day(bars[CV_FIRST_TEST_BAR]),
        end_date=_day(bars[CV_LAST_TEST_BAR]),
        output_dir=str(root / "runs"),
        rebalance_periods=2,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        benchmark_dataset=(
            make_stock_dataset(
                write_price_store(root / "benchmark", symbols=["QQQ"], seed=7, n_bars=CV_BARS)
            )
            if benchmark
            else None
        ),
    )
    return USEquityCrossectionSelectStockVectorBt(config).run_cv().run_dir


def normalized(text: str, roots: list[Path]) -> str:
    """``text`` with temporary directories, run names and plotly div ids replaced."""
    for root in roots:
        text = text.replace(str(root), "<ROOT>")
    text = _UUID.sub("<UUID>", text)
    return _RUN_NAME.sub(r"\1_<STAMP>", text)


def run_files(run_dir: Path, roots: list[Path]) -> dict[str, str]:
    """The normalized ``report.html`` and ``metrics.json`` of ``run_dir``."""
    return {
        name: normalized((run_dir / name).read_text(encoding="utf-8"), roots)
        for name in ("report.html", "metrics.json")
    }


def scenarios(tmp_path: Path, cv_project) -> dict[str, tuple[Path, list[Path]]]:
    """Run every scenario under ``tmp_path``: name -> (run dir, roots to hide)."""
    cv_root = cv_project[0]
    out = {}
    for name, run in (
        ("run_long_only_benchmark", _run_long_only_benchmark),
        ("run_long_short_delisting", _run_long_short_delisting),
        ("run_weights_benchmark", lambda root: _run_weights(root, benchmark=True)),
        ("run_weights", lambda root: _run_weights(root, benchmark=False)),
        ("run_cv_benchmark", lambda root: _run_cv(root, cv_project, benchmark=True)),
        ("run_cv", lambda root: _run_cv(root, cv_project, benchmark=False)),
    ):
        root = tmp_path / name
        root.mkdir()
        out[name] = (run(root), [root, cv_root])
    return out


#: SHA-256 of each normalized file, captured on the code before #115.
EXPECTED: dict[str, str] = {
    "run_long_only_benchmark/report.html": "49b9eeac6b1e48fc2d868febb095da30c2c7d056cd0c5dc7937abe384619aa9b",
    "run_long_only_benchmark/metrics.json": "5f58a352a7ad34937257e292565347f450f0f5c67796fd766cf7e6dcd8e0d24e",
    "run_long_short_delisting/report.html": "231ecc93fbeea3142aa90e5e11c3c804b632525d7e9325914d1fb390e233fc1b",
    "run_long_short_delisting/metrics.json": "547ac168229cfd2ece147b773478f4c024726edd88bd8ad9f06de8fc6a37febc",
    "run_weights_benchmark/report.html": "20a39e3a6aee280fdaa44b777b2f6d5e74bd21e59c2507bf20856617fdbd3333",
    "run_weights_benchmark/metrics.json": "c36c3017470634549e6caa5b325c60c327db0c070ff08b275b6e218c273d404b",
    "run_weights/report.html": "84d467db277f582b77cad390c0dfb783ccc0a38f9456ca1aaf070348ca4bdf7c",
    "run_weights/metrics.json": "69176432c0ee292a60ad339b16ee2968f0603283eb6827cc8221228cd78436e1",
    "run_cv_benchmark/report.html": "dc48544040c406ad98cc55f137dd367308efd4f951b9285e9b020bc074db0395",
    "run_cv_benchmark/metrics.json": "411cbd7c72992a3b4052afddcf1620b10385b467085e2db8e75dd121ca914d2f",
    "run_cv/report.html": "1f2ac8b188dd35d0b8823b397f73415a2ab542ec13cd145262cf93bab3e212d6",
    "run_cv/metrics.json": "c907ee5e678440b9415c82a0aca1f9aba834d1666b7ed98bf78c3686d5909cd5",
}


def test_report_and_metrics_are_byte_identical_to_before_the_public_builders(
    tmp_path, cv_project
):
    got = {}
    for name, (run_dir, roots) in scenarios(tmp_path, cv_project).items():
        for file, text in run_files(run_dir, roots).items():
            key = f"{name}/{file}"
            got[key] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            (tmp_path / f"{name}.normalized.{file}").write_text(text, encoding="utf-8")
    changed = sorted(key for key in EXPECTED if got.get(key) != EXPECTED[key])
    assert not changed, (
        f"changed against the recorded bytes: {changed}; the normalized files are "
        f"under {tmp_path}. Recorded hashes now: {json.dumps(got, indent=1)}"
    )
    assert sorted(got) == sorted(EXPECTED)


def test_fold_rows_built_from_metrics_json_are_the_bars_each_fold_traded(tmp_path, cv_project):
    run_dir = _run_cv(tmp_path, cv_project, benchmark=False)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    for fold in metrics["folds"]:
        whole = fold["metrics"]["whole"]
        traded = xr.open_zarr(run_dir / "folds" / f"fold_{fold['fold']}" / "equity.zarr").timestamp.values
        assert (bar_label(whole["Start"]), bar_label(whole["End"])) == (
            bar_label(traded[0]), bar_label(traded[-1])
        )
