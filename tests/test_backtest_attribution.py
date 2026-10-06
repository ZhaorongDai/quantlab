"""Backtest attribution: universe and score-group curves, and the excess decomposition."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_attribution import (
    annualized_log_growth,
    excess_decomposition,
    rebalanced_group_values,
)
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import make_model, make_stock_dataset, write_price_store
from tests.test_backtest_benchmark import _config as benchmark_config

NAN = np.nan

#: Six bars, two symbols, rebalancing on bars 0 and 3 (fills on bars 1 and 4).
FILL = np.array([[10.0, 20.0], [10.0, 20.0], [11.0, 18.0], [12.0, 16.0], [12.0, 16.0], [13.0, 15.0]])
VALUATION = np.array([[10.0, 20.0], [11.0, 22.0], [12.0, 18.0], [12.0, 16.0], [13.0, 18.0], [14.0, 15.0]])
REBALANCE = np.array([True, False, False, True, False, False])
#: B scores higher on bar 0, A on bar 3.
SCORES = np.array([[1.0, 2.0], [NAN, NAN], [NAN, NAN], [2.0, 1.0], [NAN, NAN], [NAN, NAN]])


def test_one_group_is_the_equal_weighted_universe_rebalanced_at_each_fill():
    """Worked by hand: bar 1 buys A at 10 and B at 20, held to bar 4's open
    (A 12, B 16: value 1.0), then rebought equally and valued at the closes."""
    (universe,) = rebalanced_group_values(FILL, VALUATION, SCORES, REBALANCE, groups=1)
    np.testing.assert_allclose(universe, [1.0, 1.1, 1.05, 1.0, 1.1041666666666667, 1.0520833333333333])


def test_groups_run_from_the_lowest_scores_to_the_highest():
    """With two groups the top group holds B after bar 0 and A after bar 3;
    the bottom group the other one."""
    bottom, top = rebalanced_group_values(FILL, VALUATION, SCORES, REBALANCE, groups=2)
    np.testing.assert_allclose(top, [1.0, 1.1, 0.9, 0.8, 0.8 * 13 / 12, 0.8 * 14 / 12])
    np.testing.assert_allclose(bottom, [1.0, 1.1, 1.2, 1.2, 1.2 * 18 / 16, 1.2 * 15 / 16])


def test_a_delisted_holding_is_settled_at_its_last_valuation_like_the_engine_does():
    """B's last bar is 2 (marked delisted there) with an open of 18 and a close
    of 17; the engine settles it at 17. The universe holds 0.5 of each from bar
    1 (A 10, B 20), so bar 4's open values it at mean(12/10, 17/20) = 1.025."""
    fill = FILL.copy()
    valuation = VALUATION.copy()
    valuation[2, 1] = 17.0
    fill[3:, 1] = NAN
    valuation[3:, 1] = NAN
    delisted = np.zeros_like(FILL, dtype=bool)
    delisted[2, 1] = True
    scores = SCORES.copy()
    scores[3, 1] = NAN
    (universe,) = rebalanced_group_values(fill, valuation, scores, REBALANCE, groups=1, delisted=delisted)
    np.testing.assert_allclose(universe[:4], [1.0, 1.1, 1.025, 1.025])
    np.testing.assert_allclose(universe[4:], [1.025 * 13 / 12, 1.025 * 14 / 12])


def test_a_rebalance_with_fewer_symbols_than_groups_leaves_every_group_in_cash_until_the_next():
    """Two symbols cannot fill three groups: the groups stay flat over that
    holding period, while the universe still holds both."""
    groups = rebalanced_group_values(FILL, VALUATION, SCORES, REBALANCE, groups=3)
    np.testing.assert_allclose(groups, np.ones((3, 6)))


def test_the_excess_splits_into_universe_selection_and_costs_that_add_up():
    """Over two years: the strategy grew 1.21x, 1.331x before costs, the universe
    1.1x and the benchmark not at all. Annualised log growth: universe ln(1.1)/2,
    selection ln(1.21)/2, costs ln(1/1.1)/2, together ln(1.21)/2 = ln(1.1)."""
    parts = excess_decomposition(
        strategy=[100.0, 121.0], gross=[100.0, 133.1], universe=[1.0, 1.1], benchmark=[50.0, 50.0], years=2.0
    )
    ln = np.log(1.1)
    assert parts["universe"] == pytest.approx(ln / 2)
    assert parts["selection"] == pytest.approx(ln)
    assert parts["costs"] == pytest.approx(-ln / 2)
    assert parts["total"] == pytest.approx(ln)


def test_without_a_curve_before_costs_selection_carries_the_costs_and_no_cost_part_is_given():
    """An engine that cannot simulate without costs: selection is the strategy
    over the universe, ln(1.21 / 1.1) / 2, and there is no ``costs`` key."""
    parts = excess_decomposition(
        strategy=[100.0, 121.0], gross=None, universe=[1.0, 1.1], benchmark=[50.0, 50.0], years=2.0
    )
    assert "costs" not in parts
    assert parts["selection"] == pytest.approx(np.log(1.1) / 2)
    assert parts["total"] == pytest.approx(np.log(1.21) / 2)


def test_annualized_log_growth_is_the_log_of_the_last_over_the_first_value_per_year():
    assert annualized_log_growth([2.0, 3.0, 8.0], years=2.0) == pytest.approx(np.log(4.0) / 2)


def test_without_a_benchmark_the_excess_is_measured_over_the_universe():
    parts = excess_decomposition(
        strategy=[100.0, 121.0], gross=[100.0, 133.1], universe=[1.0, 1.1], benchmark=None, years=2.0
    )
    assert "universe" not in parts
    assert parts["total"] == pytest.approx(np.log(1.1) / 2)


# --------------------------------------------------------------------------
# A model backtest records its attribution
# --------------------------------------------------------------------------


def _years(result) -> float:
    """The window's length in years: bars x bar interval over the market's year (252 days)."""
    return result.simulation.value.sizes["timestamp"] * result.simulation.bar_interval / pd.Timedelta(days=252)


def _log_growth(curve) -> float:
    values = np.asarray(curve.values, dtype=np.float64)
    return float(np.log(values[-1] / values[0]))


def test_a_run_splits_its_excess_over_the_benchmark_into_parts_that_add_up(tmp_path):
    result = USEquityCrossectionSelectStockVectorBt(
        benchmark_config(tmp_path, fees=0.001, slippage=0.001)
    ).run()
    attribution = result.metrics["attribution"]
    parts = attribution["decomposition"]

    expected = (_log_growth(result.simulation.value) - _log_growth(result.benchmark.value)) / _years(result)
    assert parts["total"] == pytest.approx(expected, rel=1e-9)
    assert parts["universe"] + parts["selection"] + parts["costs"] == pytest.approx(expected, rel=1e-9)
    assert parts["costs"] < 0
    assert attribution["groups"] == 10
    assert len(attribution["group_annualized_log_return"]) == 10


def test_without_costs_the_cost_part_is_zero(tmp_path):
    result = USEquityCrossectionSelectStockVectorBt(benchmark_config(tmp_path)).run()
    assert result.metrics["attribution"]["decomposition"]["costs"] == 0.0


def test_run_cv_attributes_the_stitched_curve(tmp_path):
    n_bars, train_periods = 80, 30
    dataset_config = write_price_store(tmp_path / "store", n_bars=n_bars)
    bars = pd.bdate_range("2024-01-01", periods=n_bars).strftime("%Y-%m-%d")
    dates = dict(
        start_date=bars[0], end_date=bars[-1], train_start=bars[0], train_end=bars[train_periods - 1],
        test_start=bars[train_periods], test_end=bars[-1],
    )
    model = make_model(tmp_path / "train", dataset_config, **dates)
    model.collect()
    cv_unit = model.train_cv(train_periods=train_periods, expanding=True)
    cv = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(tmp_path / "backtest", dataset_config, **dates),
        model_mode="load", cv_project_dir=str(cv_unit.path),
        start_date=bars[train_periods], end_date=bars[-1], output_dir=None, rebalance_periods=2,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        fees=0.001, slippage=0.001,
        benchmark_dataset=make_stock_dataset(
            write_price_store(tmp_path / "benchmark", symbols=["QQQ"], seed=7, n_bars=n_bars)
        ),
    )).run_cv()

    parts = cv.metrics["stitched"]["attribution"]["decomposition"]
    years = cv.simulation.value.sizes["timestamp"] * cv.simulation.bar_interval / pd.Timedelta(days=252)
    expected = (_log_growth(cv.simulation.value) - _log_growth(cv.benchmark.value)) / years
    assert parts["universe"] + parts["selection"] + parts["costs"] == pytest.approx(expected, rel=1e-9)
    assert parts["costs"] < 0
    # Only the stitched curve is attributed; a fold's own simulation is not.
    assert all("attribution" not in fold["metrics"] for fold in cv.metrics["folds"])


# --------------------------------------------------------------------------
# The run directory and the report
# --------------------------------------------------------------------------


def test_the_run_directory_records_the_attribution_curves_and_the_report_draws_them(tmp_path):
    result = USEquityCrossectionSelectStockVectorBt(
        benchmark_config(tmp_path, fees=0.001, slippage=0.001)
    ).run()
    run = BacktestRun.open(result.run_dir)
    equity = run.equity()
    groups = result.metrics["attribution"]["groups"]
    assert equity["universe_value"].sizes == {"timestamp": result.simulation.value.sizes["timestamp"]}
    assert equity["gross_value"].values[-1] > equity["value"].values[-1]
    assert equity["group_value"].sizes["group"] == groups

    page = run.report()
    assert 'data-tab' in page and ">Attribution</button>" in page
    for name in ("attribution_strategy", "attribution_gross", "attribution_universe",
                 "attribution_benchmark", "group_annualized_log_return",
                 *(f"attribution_group_{g}" for g in range(1, groups + 1))):
        assert f'"name":"{name}"' in page, name
    for row in (">Universe vs benchmark</th>", ">Selection</th>", ">Costs</th>", ">Total</th>"):
        assert row in page, row
