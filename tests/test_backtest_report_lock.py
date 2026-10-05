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
directory's timestamped name and plotly's random div ids) and rounding every
decimal number to 10 significant digits, is the one recorded below, captured
on the code before #115.

The rounding is there because the last digits of a float differ between
CPUs: numpy sums in another order on another SIMD width, so a Sharpe ratio
printed on macOS as ``-3.9547411421850533`` prints as ``-3.9547411421850547``
on a Linux runner, from the 13th significant digit on. Ten digits keep every
change a reader of the report could see. The bytes still depend on the
installed plotly, numpy and vectorbt; a version bump that changes them shows
up here first.
Beside the hashes, the recipe ``report_windows`` documents for building
``run_cv()`` fold rows from ``metrics.json`` is checked against the bars each
fold traded.

The hashes lock bytes on disk, so they read ``report.html`` and ``metrics.json``
by name; the fold-row check reads the run through ``BacktestRun`` (#134).

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
from quantlab.runs.backtest_run import BacktestRun
from quantlab.utils.date_range import bar_label
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
#: A decimal number on its own: not part of a word, a version or a longer number.
_DECIMAL = re.compile(r"(?<![\w.])-?\d+(?:\.\d+(?:[eE][-+]?\d+)?|[eE][-+]?\d+)(?![\w.])")


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
    cv = model.train_cv(train_periods=CV_TRAIN_PERIODS)
    return root, dataset_config, bars, cv.path


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
    """``text`` with temporary directories, run names and plotly div ids replaced
    and decimal numbers rounded to 10 significant digits."""
    for root in roots:
        text = text.replace(str(root), "<ROOT>")
    text = _UUID.sub("<UUID>", text)
    text = _RUN_NAME.sub(r"\1_<STAMP>", text)
    return _DECIMAL.sub(lambda match: format(float(match.group()), ".10g"), text)


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


#: SHA-256 of each normalized file, captured on the code before #115. #123
#: moved fold checkpoints from `{Cls}_cv_fold_{i}/` to `fold_{i}/`, which the
#: run_cv metrics.json record; its two hashes were re-captured then, after
#: checking that undoing just that path change gives the old hashes back.
#: All twelve were re-captured when decimals began to be rounded, after
#: checking that the same code at full precision gives the old hashes back
#: and that a Linux runner's output, rounded, gives the new ones.
EXPECTED: dict[str, str] = {
    "run_long_only_benchmark/report.html": "8c82484b36edf093ee91cfe049a3bdf8554878d12c73a808c10b7163b767f4e5",
    "run_long_only_benchmark/metrics.json": "67e754641b89a5a99007ca69e64093e4f555a87a70b7539b27902da99a05f965",
    "run_long_short_delisting/report.html": "1464d6a0d110496680a0c11d3fbd35ed23113df22f5c5185d2f0d8da15634ac7",
    "run_long_short_delisting/metrics.json": "d1aecc507332b6616dd96e31bc716c648a56e4158a0988d144c1fa8378b1625b",
    "run_weights_benchmark/report.html": "82e171cf20ce3dda4dc452b975b59c783faeaff9219539d32b81bbf982717d16",
    "run_weights_benchmark/metrics.json": "aaf9c98eca07ec66a8f01999e3a6de56221133d6cb07b0a9dd8ea175170c84b2",
    "run_weights/report.html": "7bdd8d285ebca72ef91b02867c22750a8ea4f7e7d668826bcff316c7d5d8894f",
    "run_weights/metrics.json": "fbfd860207fe6f7410dba4f91f905b38e9a45739d4c22f7c5d48de5e0b209b73",
    "run_cv_benchmark/report.html": "6fffabe43720bb37f6617c8499b9732c0b6410da58ea5c480c97dfd5e1bf74b9",
    "run_cv_benchmark/metrics.json": "d6889ddaf7373f430bdcbcae882ca4764d86c17baf457ee15dc34737972375b1",
    "run_cv/report.html": "020d6aeca7f2cd2b83d91a07aba3568e56a6c47be24c26c47a2898772cc7ac71",
    "run_cv/metrics.json": "52bea0786c6385d0230d6bd5e9fdce1679805645de9c67620de53537cdea2721",
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
    run = BacktestRun.open(run_dir)
    folds = {fold.index: fold for fold in run.folds}
    for fold in run.metrics()["folds"]:
        whole = fold["metrics"]["whole"]
        traded = folds[fold["fold"]].equity().timestamp.values
        assert (bar_label(whole["Start"]), bar_label(whole["End"])) == (
            bar_label(traded[0]), bar_label(traded[-1])
        )
