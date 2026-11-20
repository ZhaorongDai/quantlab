"""The approximate backtest's TopkDropout against a backtest run.

Each case runs `run()` with a `TopkDropoutConfig` on the synthetic price
store of `tests/test_backtest_approximate.py`, then the approximate backtest
with the same rule on the run's recorded predictions:

- the decided bars are the run's, and without fees the weights at them equal
  the run's (TopkDropout keeps holdings at their drifted weight, so the
  bars are decided forward from the drifted book, as the run decides them);
- with fees the value stays within 1e-3 of the run's (the run caps buys at
  the cash the fees leave; the approximation does not);
- long-only rebalancing every bar and every fifth bar, and long-short every
  bar: its drifted book reaches a gross of 1.0044 at 2024-03-27, which the
  rule scales back to 1 (before that fix, `run()` refused the row).

What turns this file red: the approximation decides from the undrifted book,
a bar from a book other than the previous decision, or picks other names
than the per-bar rule.
"""

import numpy as np
import pytest

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import USEquityVectorBt
from quantlab.portfolio.config import TopkDropoutConfig
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import make_model, make_stock_dataset
from tests.test_backtest_approximate import FEES, VALUE_TOLERANCE, WINDOW, _approximate, _day, store  # noqa: F401

CASES = {
    "long_only_r1": (TopkDropoutConfig(direction="long_only", topk=4, n_drop=1), 1),
    "long_only_r5": (TopkDropoutConfig(direction="long_only", topk=4, n_drop=2), 5),
    "long_short_r1": (TopkDropoutConfig(direction="long_short", topk=3, n_drop=1), 1),
}


def _run(store, config: TopkDropoutConfig, rebalance_periods: int, fees: float, tag: str):  # noqa: F811
    root, dataset_config, bars = store
    model = make_model(
        root / f"tkd_model_{tag}", dataset_config,
        start_date=_day(bars[0]), end_date=_day(bars[59]), train_start=_day(bars[0]),
        train_end=_day(bars[49]), test_start=_day(bars[50]), test_end=_day(bars[59]),
    )
    return USEquityVectorBt(CrossSectionBacktestConfig(
        price_dataset=make_stock_dataset(dataset_config), model=model, model_mode="train",
        start_date=_day(bars[WINDOW[0]]), end_date=_day(bars[WINDOW[1]]),
        output_dir=str(root / f"tkd_runs_{tag}"), rebalance_periods=rebalance_periods,
        constructor=TopkDropoutConstructor(config), fees=fees, slippage=0.0,
    )).run()


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, store):  # noqa: F811
    config, rebalance_periods = CASES[request.param]
    out = {}
    for fees in (0.0, FEES):
        result = _run(store, config, rebalance_periods, fees, f"{request.param}_{fees}")
        predictions = BacktestRun.open(result.run_dir).predictions()
        account = _approximate(store, rebalance_periods, cost=fees).run(predictions, TopkDropoutConstructor(config))
        out[fees] = (result, account)
    return out


def test_without_fees_the_weights_at_the_decided_bars_equal_the_runs(case):
    result, account = case[0.0]
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values
    np.testing.assert_array_equal(account.decided_weights.timestamp.values, weights.timestamp.values[decided])
    np.testing.assert_allclose(account.decided_weights.values, weights["weight"].values[decided], atol=1e-9)


def test_with_fees_the_value_stays_close_to_the_runs(case):
    result, account = case[FEES]
    run_value = result.simulation.value.values / result.simulation.value.values[0]
    np.testing.assert_array_equal(account.value.timestamp.values, result.simulation.value.timestamp.values)
    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)


def _three_symbol_account(price: np.ndarray, score: np.ndarray):
    """The approximate TopkDropout (topk 1, n_drop 1, daily, no cost) on three symbols A, B, C over eight bars."""
    import pandas as pd
    import xarray as xr

    from quantlab.backtest.config import ApproximateBacktestConfig
    from quantlab.backtest.predefined.us_equity import USEquityApproximateBacktest
    from quantlab.dataset.memory import FrameDataset
    from quantlab.runs.prediction_panel import LabelSpec, PredictionPanel

    bars = pd.bdate_range("2024-01-02", periods=8)
    coords = {"timestamp": bars, "symbol": np.array(["A", "B", "C"], dtype=object)}
    dims = ("timestamp", "symbol")
    prices = FrameDataset(xr.Dataset({"adjOpen": (dims, price), "adjClose": (dims, price)}, coords=coords))
    panel = PredictionPanel(
        xr.Dataset({"ret_1": (dims, score)}, coords=coords), [LabelSpec("ret_1", "standardized", 1, 1)]
    )
    backtest = USEquityApproximateBacktest(ApproximateBacktestConfig(
        price_dataset=prices, start_date=str(bars[0].date()), end_date=str(bars[-1].date()),
        rebalance_periods=1, cost=0.0,
    ))
    return backtest.run(panel, TopkDropoutConstructor(TopkDropoutConfig("long_only", topk=1, n_drop=1)))


def test_a_buy_without_a_fill_price_is_rejected_and_the_slot_is_not_locked():
    """A has the best score on bar 3, its last priced bar, and the rule switches into it.

    As in a backtest run, the buy at bar 4 has no fill price and is rejected
    while the sell of B fills, so the book is cash for a bar; the decision at
    bar 4 sees A settled and buys B back, which fills at bar 5. Before the
    fix A was bought at bar 3's valuation and held for good, the value frozen
    from bar 4 on.
    """
    up = 100.0 * 1.01 ** np.arange(8)
    price = np.stack([up, up, up], axis=1)
    price[4:, 0] = np.nan
    score = np.tile([0.0, 1.0, 0.5], (8, 1))
    score[3] = [2.0, 1.0, 0.5]
    score[4:, 0] = np.nan
    account = _three_symbol_account(price, score)
    np.testing.assert_array_equal(account.weights["weight"].sel(symbol="B").values[:-1], [1, 1, 1, 0, 1, 1, 1])
    np.testing.assert_allclose(
        account.value.values, [1.0, 1.0, 1.01, 1.01**2, 1.01**3, 1.01**3, 1.01**4, 1.01**5], rtol=1e-12
    )


def test_a_delisted_holding_is_settled_and_frees_its_slot():
    """A is held, delists after bar 3 and is settled into cash at its last valuation on bar 4.

    The decision at bar 4 sees the slot free and buys B, which fills at bar
    5; before the fix A stayed locked and the value never moved again.
    """
    price = np.stack([100.0 * 1.02 ** np.arange(8), 100.0 * 1.01 ** np.arange(8), np.full(8, 100.0)], axis=1)
    price[4:, 0] = np.nan
    score = np.tile([2.0, 1.0, 0.5], (8, 1))
    score[4:, 0] = np.nan
    account = _three_symbol_account(price, score)
    weights = account.weights["weight"].transpose("timestamp", "symbol").values
    np.testing.assert_array_equal(weights[:4], [[1.0, 0.0, 0.0]] * 4)
    np.testing.assert_array_equal(weights[4:7], [[0.0, 1.0, 0.0]] * 3)
    settled = 1.02 ** 2
    np.testing.assert_allclose(
        account.value.values, [1.0, 1.0, 1.02, settled, settled, settled, settled * 1.01, settled * 1.01**2],
        rtol=1e-12,
    )
