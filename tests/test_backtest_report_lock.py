"""Byte lock on a backtest run's ``report.html`` and ``metrics.json`` (#115).

The inputs of the report became public functions in
``quantlab.runs.backtest_report`` (``report_summary``, ``report_windows``,
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
The same runs with the holdings left out of ``write_backtest_report`` (the
quantlab-trader call path) must give the reports' bytes from before the
Holdings tab (#225).
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
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
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
#: Re-recorded when model runs gained the ``attribution`` block and the
#: Attribution tab; their metrics with that block removed hash as before.
#: The six reports were re-recorded when the Portfolio construction setup
#: row became a scrolling box (d0e4a0cd); removing that box's CSS rule and
#: wrapper gives the old hashes back. The metrics did not change.
#: The six reports were re-recorded for the dashboard page (#215): header,
#: sidebar sections, cards, chart styling, hover explanations and the plain
#: metric definitions. The metrics did not change.
#: The six reports were re-recorded for the Holdings tab (#225); the same
#: runs written without holdings give the #215 hashes back
#: (`WITHOUT_HOLDINGS`). The metrics did not change.
#: Re-recorded when Top-10 holding became the ten largest holdings by size
#: (shorts included) and the tab's explanation gained that and the cash
#: definition. The metrics did not change.
#: Re-recorded when the Holdings tab dropped its total return, annualised
#: return and max drawdown tiles, which repeated the headline cards.
#: Re-recorded when the Holdings table gained its path-in-period column.
#: Re-recorded when a short's target bar became light red beside its dark red holding bar.
#: Re-recorded when the target / holding header gained a key of the two bars,
#: long and short colours of each (which is which on hover only).
#: Re-recorded when each Holdings day embedded its summary (holdings count,
#: top-10 holding, new names since the previous rebalance) for the page to read
#: (#228). The metrics did not change.
EXPECTED: dict[str, str] = {
    "run_long_only_benchmark/report.html": "f7d745caa15194c295e35b00c709723e0603fe3bc2c8069cf24e9ebdd9a56e5a",
    "run_long_only_benchmark/metrics.json": "b4901c705f203ce3a769b40c357ce8a444d16df4488492ab2abf4f03e2697be6",
    "run_long_short_delisting/report.html": "a3575aa25295e894df006d7ac381498ba96085503f6d19f5f1c2f738f2a18bfe",
    "run_long_short_delisting/metrics.json": "61e9a1bee817c10179cc309e8722ad9a7a9a56c896183d0a3fe5d79ee90c79ed",
    "run_weights_benchmark/report.html": "f3a09c7e860de234281a3f581acbd3bf2f52b0e206c870e32c92ad7254908806",
    "run_weights_benchmark/metrics.json": "aaf9c98eca07ec66a8f01999e3a6de56221133d6cb07b0a9dd8ea175170c84b2",
    "run_weights/report.html": "717b5b66dd26f3904f643f6c304bfcf921b1d8d058241be60a340ef81f67d38e",
    "run_weights/metrics.json": "fbfd860207fe6f7410dba4f91f905b38e9a45739d4c22f7c5d48de5e0b209b73",
    "run_cv_benchmark/report.html": "e7be944dda490bc91cf1314c5fdce7a6d288502ed96c0b26dc5beea862464e0c",
    "run_cv_benchmark/metrics.json": "2df35a73119a4f7d4ed32983c76662dbe9b07b552c19169a870ddb79b3a8b3a3",
    "run_cv/report.html": "ba104362504000bdf29ef01bf53eac281a98dbe957d21537d36649c73bda34d8",
    "run_cv/metrics.json": "966f1e174eb2b1058c99048b48599fddff839a163a1fb847113bbac7542d6ced",
}


#: The six reports as they were before the Holdings tab (#225), the hashes
#: recorded for #215. A report written without holdings, as quantlab-trader
#: writes one, must still be these bytes.
WITHOUT_HOLDINGS: dict[str, str] = {
    "run_long_only_benchmark/report.html": "8caaa6219ea786273f1c8a00032708572e0fc572c5f57cd198702279ba330d15",
    "run_long_short_delisting/report.html": "54b3cea2423d40c569b90d5a32a141c8fdcf218c2fe83b3b34ee45a06ad5907f",
    "run_weights_benchmark/report.html": "dd91414c66fc9013ceb3b93b5ae80465ea2404684ac6e0427fe83859977888fe",
    "run_weights/report.html": "dd2b7cc3db804b68b16afbf9020ac0aa963a053e18c1bcca717a80c6476b056d",
    "run_cv_benchmark/report.html": "100bcf7d9935a5e8f090e7d6700214ffc1f9d88e4062db923b8b286d8122203f",
    "run_cv/report.html": "cbd7443db5f3d1384a370e01f2d85dc4b3443a9f961c9afb7fe82e502f67d89a",
}


def _assert_hashes(tmp_path: Path, cv_project, expected: dict[str, str]) -> None:
    """Run every scenario and compare the normalized files named in ``expected``."""
    got = {}
    for name, (run_dir, roots) in scenarios(tmp_path, cv_project).items():
        for file, text in run_files(run_dir, roots).items():
            key = f"{name}/{file}"
            if key not in expected:
                continue
            got[key] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            (tmp_path / f"{name}.normalized.{file}").write_text(text, encoding="utf-8")
    changed = sorted(key for key in expected if got.get(key) != expected[key])
    assert not changed, (
        f"changed against the recorded bytes: {changed}; the normalized files are "
        f"under {tmp_path}. Recorded hashes now: {json.dumps(got, indent=1)}"
    )
    assert sorted(got) == sorted(expected)


def test_report_and_metrics_are_byte_identical_to_before_the_public_builders(
    tmp_path, cv_project
):
    _assert_hashes(tmp_path, cv_project, EXPECTED)


def test_without_holdings_the_report_is_byte_identical_to_before_the_holdings_tab(
    tmp_path, cv_project, monkeypatch
):
    """The quantlab-trader call path: ``write_backtest_report`` given no holdings."""
    import quantlab.backtest.base as base

    write = base.write_backtest_report

    def without_holdings(*args, holdings=None, holding_names=None, **kwargs):
        write(*args, **kwargs)

    monkeypatch.setattr(base, "write_backtest_report", without_holdings)
    _assert_hashes(tmp_path, cv_project, WITHOUT_HOLDINGS)


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
