"""`BaseBacktester.run_weights` and the optional `BacktestConfig.output_dir` (#64).

`run_weights(weights)` backtests a precomputed target-weight panel without a
model: it loads the configured prices, simulates (a weight at bar t fills at
bar t+1), handles the benchmark like `run()`, and reports whole-window metrics
only, since there is no training window to split against. What is locked here:

- **Parity.** The weights a model-driven `run()` produced, fed to
  `run_weights` on a config with no model, give the same equity curve, the
  same whole-window metrics and the same benchmark comparison.
- **No model needed.** `model` and `model_mode` are both set or both `None`
  (a half-set pair is refused at construction); `run()` and `run_cv()` refuse
  a config without a model with a message pointing at `run_weights`.
- **Contract.** Weights that break the D-03 contract (a row mixing NaN and
  finite values, gross exposure above one, axes other than the price bars)
  raise, naming the offending bar.
- **In memory.** `output_dir=None` writes nothing anywhere, for `run()` and
  `run_weights`, and the result's `run_dir` is `None`.
- **Persistence.** With an output directory, `run_weights` writes the usual
  run-directory artifacts and its metrics carry no in/out-of-sample blocks.

Everything is synthetic, CPU-only and offline.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
from quantlab.runs.backtest_run import BacktestRun, Market
from quantlab.utils.module import load_backtester_from_config
from quantlab.portfolio.predefined.top_n import TopNConstructor
from tests.backtest_fixtures import (
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
WINDOW_START = 30
WINDOW_END = 55

RUN_DIR_ARTIFACTS = [
    "config.json",
    "equity.zarr",
    "metrics.json",
    "report.html",
    "run.json",
    "settlements.json",
    "weights.zarr",
]


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _bars() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=N_BARS)


def _files_under(root: Path) -> list[str]:
    return sorted(
        str(Path(dirpath, name).relative_to(root))
        for dirpath, _, names in os.walk(root)
        for name in names
    )


@pytest.fixture
def stores(tmp_path):
    """The price store, a one-symbol benchmark store and a trained checkpoint."""
    bars = _bars()
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    benchmark_config = write_price_store(
        tmp_path / "benchmark", symbols=["QQQ"], seed=7, n_bars=N_BARS
    )
    model_dates = dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **model_dates)
    )
    return dict(
        root=tmp_path,
        dataset=dataset_config,
        benchmark=benchmark_config,
        model_dates=model_dates,
        checkpoint=checkpoint,
    )


def _config(stores, *, with_model: bool, **overrides) -> CrossSectionBacktestConfig:
    bars = _bars()
    kwargs = dict(
        price_dataset=make_stock_dataset(stores["dataset"]),
        start_date=_day(bars[WINDOW_START]),
        end_date=_day(bars[WINDOW_END]),
        output_dir=str(stores["root"] / "runs"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        benchmark_dataset=make_stock_dataset(stores["benchmark"]),
    )
    if with_model:
        kwargs.update(
            model=make_model(
                stores["root"] / "backtest", stores["dataset"], **stores["model_dates"]
            ),
            model_mode="load",
            checkpoint=str(stores["checkpoint"]),
        )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


def _run_weights(stores, weights, **overrides):
    backtester = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=False, **overrides)
    )
    return backtester.run_weights(weights)


def _assert_same_block(got: dict, want: dict) -> None:
    assert sorted(got) == sorted(want)
    for key, value in want.items():
        if isinstance(value, float) and np.isnan(value):
            assert np.isnan(got[key]), key
        else:
            assert got[key] == value, key


# --------------------------------------------------------------------------
# Parity with run()
# --------------------------------------------------------------------------


def test_run_weights_reproduces_the_run_that_produced_the_weights(stores):
    run_result = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=True)
    ).run()

    result = _run_weights(
        stores, run_result.weights, output_dir=str(stores["root"] / "weights_runs")
    )

    np.testing.assert_array_equal(
        result.simulation.value.values, run_result.simulation.value.values
    )
    np.testing.assert_array_equal(
        result.simulation.value.timestamp.values,
        run_result.simulation.value.timestamp.values,
    )
    np.testing.assert_array_equal(
        result.simulation.returns.values, run_result.simulation.returns.values
    )
    _assert_same_block(result.metrics["whole"], run_result.metrics["whole"])
    _assert_same_block(
        result.metrics["benchmark"]["whole"], run_result.metrics["benchmark"]["whole"]
    )
    _assert_same_block(
        result.metrics["relative"]["whole"], run_result.metrics["relative"]["whole"]
    )
    np.testing.assert_array_equal(
        result.benchmark.value.values, run_result.benchmark.value.values
    )
    xr.testing.assert_identical(result.weights, run_result.weights)
    assert result.predictions is None


def test_the_report_timeline_of_given_weights_is_one_row_of_traded_bars(stores):
    """No model, so no training window: one row, the window the weights traded."""
    run_result = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=True)
    ).run()
    result = _run_weights(
        stores, run_result.weights, output_dir=str(stores["root"] / "weights_runs")
    )

    page = (result.run_dir / "report.html").read_text(encoding="utf-8")
    timeline = page[page.index("<h2>Windows</h2>") : page.index("</svg>")]
    timestamps = result.simulation.value.timestamp.values
    assert re.findall(r"<title>([^<]+)</title>", timeline) == [
        f"traded {_day(timestamps[0])} .. {_day(timestamps[-1])}"
    ]


def test_run_weights_accepts_a_data_array_in_any_axis_order(stores):
    reference = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=True, output_dir=None)
    ).run()
    shuffled = (
        reference.weights["weight"]
        .transpose("symbol", "timestamp")
        .isel(symbol=[3, 1, 0, 5, 2, 4])
        .rename("my_weights")
    )

    result = _run_weights(stores, shuffled, output_dir=None)

    np.testing.assert_array_equal(
        result.simulation.value.values, reference.simulation.value.values
    )
    assert result.weights["weight"].dims == ("timestamp", "symbol")
    assert result.weights.symbol.values.tolist() == SYMBOLS


# --------------------------------------------------------------------------
# A config without a model
# --------------------------------------------------------------------------


def test_run_refuses_a_config_without_a_model(stores):
    backtester = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=False)
    )
    with pytest.raises(ValueError, match=r"run\(\) requires config\.model.*run_weights"):
        backtester.run()


def test_run_cv_refuses_a_config_without_a_model(stores):
    backtester = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=False, cv_project_dir=str(stores["root"]))
    )
    with pytest.raises(ValueError, match=r"run_cv\(\) requires config\.model"):
        backtester.run_cv()


def test_a_model_without_a_model_mode_is_refused_at_construction(stores):
    config = _config(stores, with_model=True, model_mode=None)
    with pytest.raises(
        ValueError,
        match=r"model and model_mode must be both set or both None, got "
        r"model=FirstFeatureHead and model_mode=None",
    ):
        USEquityCrossectionSelectStockVectorBt(config)


def test_a_model_mode_without_a_model_is_refused_at_construction(stores):
    config = _config(stores, with_model=False, model_mode="load")
    with pytest.raises(
        ValueError, match=r"got model=None and model_mode='load'"
    ):
        USEquityCrossectionSelectStockVectorBt(config)


def test_config_without_a_model_serializes_and_rebuilds(stores):
    backtester = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=False)
    )
    config = backtester.get_config()
    assert config["model"] is None
    assert config["model_mode"] is None

    rebuilt = load_backtester_from_config(json.loads(json.dumps(config)))
    assert rebuilt.config.model is None


# --------------------------------------------------------------------------
# The weight contract
# --------------------------------------------------------------------------


def _hold_everywhere(stores) -> xr.Dataset:
    """An all-NaN (hold) weight panel on the window's price bars."""
    bars = _bars()[WINDOW_START : WINDOW_END + 1]
    return xr.Dataset(
        {
            "weight": (
                ("timestamp", "symbol"),
                np.full((len(bars), len(SYMBOLS)), np.nan),
            )
        },
        coords={"timestamp": bars, "symbol": SYMBOLS},
    )


def test_a_row_mixing_nan_and_finite_values_trades_only_the_finite_targets(stores):
    weights = _hold_everywhere(stores)
    weights["weight"][3, 0] = 0.5

    result = _run_weights(stores, weights, output_dir=None)

    orders = result.simulation.orders
    fill_bar = weights.timestamp.values[4]
    traded = {str(s) for s, t in zip(orders["symbol"].values, orders["timestamp"].values) if t == fill_bar}
    assert traded == {str(weights.symbol.values[0])}


def test_gross_exposure_above_one_is_refused_naming_the_bar(stores):
    weights = _hold_everywhere(stores)
    weights["weight"][5, :] = 0.3
    bar = _day(weights.timestamp.values[5])
    with pytest.raises(ValueError, match=rf"weight row at {bar} has gross exposure"):
        _run_weights(stores, weights, output_dir=None)


def test_weights_off_the_price_bars_are_refused_naming_the_bar(stores):
    weights = _hold_everywhere(stores).isel(timestamp=slice(0, -1))
    last_bar = _day(_bars()[WINDOW_END])
    with pytest.raises(ValueError, match=rf"missing .*{last_bar}"):
        _run_weights(stores, weights, output_dir=None)


def test_weights_with_an_unknown_symbol_are_refused(stores):
    weights = _hold_everywhere(stores).assign_coords(symbol=SYMBOLS[:-1] + ["ZZZ"])
    with pytest.raises(ValueError, match=r"ZZZ"):
        _run_weights(stores, weights, output_dir=None)


# --------------------------------------------------------------------------
# output_dir=None and persistence
# --------------------------------------------------------------------------


def test_output_dir_none_writes_nothing_for_run_and_run_weights(stores, monkeypatch):
    workdir = stores["root"] / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    before = _files_under(stores["root"])

    run_result = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=True, output_dir=None)
    ).run()
    weights_result = _run_weights(stores, run_result.weights, output_dir=None)

    assert run_result.run_dir is None
    assert weights_result.run_dir is None
    assert _files_under(stores["root"]) == before


def test_run_weights_with_an_output_dir_writes_a_whole_window_run_directory(stores):
    run_result = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=True, output_dir=None)
    ).run()

    result = _run_weights(stores, run_result.weights)

    assert result.run_dir.parent == stores["root"] / "runs"
    assert sorted(p.name for p in result.run_dir.iterdir()) == RUN_DIR_ARTIFACTS
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert sorted(metrics) == ["benchmark", "execution", "notes", "relative", "whole"]
    assert sorted(metrics["benchmark"]) == ["axis_symbol", "symbol", "whole"]
    assert sorted(metrics["relative"]) == ["whole"]
    assert sorted(result.metrics) == sorted(metrics)
    run = BacktestRun.open(result.run_dir)
    assert run.kind == "run_weights" and run.trained_run() is None
    assert run.rebuild("model") is None
    assert sorted(run.data_fingerprint) == ["benchmark_dataset", "price_dataset"]
    assert run.market == Market(fill_price_column="adjOpen", valuation_price_column="adjClose")
    np.testing.assert_array_equal(
        run.weights()["weight"].values, run_result.weights["weight"].values
    )
    report = (result.run_dir / "report.html").read_text()
    assert "In-sample" not in report
    assert "precomputed weights" in report


# --------------------------------------------------------------------------
# report_figure
# --------------------------------------------------------------------------


def _weights_result(stores, **overrides):
    backtester = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=False, output_dir=None, **overrides)
    )
    return backtester, backtester.run_weights(_hold_everywhere(stores))


def test_report_figure_draws_a_result_of_its_own_backtester(stores):
    backtester, result = _weights_result(stores)

    figure = backtester.report_figure(result)

    names = [trace.name for trace in figure.data]
    assert names[0] == "equity" and "benchmark_equity" in names


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(start_date=_day(_bars()[WINDOW_START + 1])), r"window"),
        (dict(init_cash=5.0), r"init_cash"),
        (dict(benchmark_dataset=None), r"benchmark"),
    ],
)
def test_report_figure_refuses_a_result_of_another_config(stores, overrides, message):
    _, result = _weights_result(stores)
    other = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=False, output_dir=None, **overrides)
    )

    with pytest.raises(ValueError, match=message):
        other.report_figure(result)


def test_report_figure_refuses_a_cv_result(stores):
    from quantlab.base.backtest import CVBacktestResult

    backtester, result = _weights_result(stores)
    cv_result = CVBacktestResult(
        run_dir=None,
        folds=[],
        weights=result.weights,
        simulation=result.simulation,
        metrics={},
    )

    with pytest.raises(TypeError, match=r"run_cv\(\) result"):
        backtester.report_figure(cv_result)


# --------------------------------------------------------------------------
# WeightsVectorBt config checks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("top_n", [True, 2.0, 0])
def test_weights_backtester_refuses_a_top_n_that_is_not_a_positive_integer(stores, top_n):
    from quantlab.backtest.predefined.weights import WeightsVectorBt
    from quantlab.base.config import WeightsBacktestConfig

    bars = _bars()
    config = WeightsBacktestConfig(
        price_dataset=make_stock_dataset(stores["dataset"]),
        start_date=_day(bars[WINDOW_START]),
        end_date=_day(bars[WINDOW_END]),
        output_dir=None,
        rebalance_periods=1,
        fill_price_column="adjOpen",
        valuation_price_column="adjClose",
        trading_days_per_year=252,
        session_minutes_per_day=390,
        top_n=top_n,
    )
    with pytest.raises(ValueError, match=r"top_n must be an integer >= 1 or None"):
        WeightsVectorBt(config)
