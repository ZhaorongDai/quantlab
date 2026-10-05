"""The holdings ``DecisionInputs.weights`` hands a rule are the engine's, costs included (#118).

``DecisionInputs.weights`` replays each rebalance's weights through the Execution
module with the settings it is given (``execution=``), so the current weights
a rule receives at a rebalance bar equal what the vectorbt engine holds at
that bar's close when it simulates the same weights with the same settings.
What is locked here, on a hand-built market through the public weights
backtest:

- the equality, with fees and slippage, on both sizing bases, long-only (fees
  push the fully invested buys past the cash, so they are cut) and
  long-short, across rejected orders, locked positions and settlements;
- without ``execution`` the replay is fill-price sizing without costs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.weights import WeightsVectorBt
from quantlab.base.config import WeightsBacktestConfig
from quantlab.base.portfolio import PortfolioConstructor
from quantlab.dataset.memory import FrameDataset
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.execution.rules import ExecutionSettings

NAN = np.nan
N_BARS, N_SYMBOLS, REBALANCE = 40, 6, 3
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)
SYMBOLS = [f"SYM{j}" for j in range(N_SYMBOLS)]

#: Contexts seen by the rule, in call order.
SEEN: list = []


@dataclass(frozen=True)
class ProportionalConfig:
    direction: str = "long_only"


class Proportional(PortfolioConstructor):
    """Keeps locked positions and spreads the rest of the book over the
    tradable symbols in proportion to their scores; records its contexts."""

    config_cls = ProportionalConfig

    def construct(self, context):
        SEEN.append(context)
        locked = context.locked
        row = context.current_weights.where(locked, 0.0)
        score = context.predictions["score"]
        free = context.tradable & ~locked & np.isfinite(score)
        if self.config.direction == "long_only":
            score = np.abs(score)
        score = score.where(free, 0.0)
        budget = 1.0 - float(np.abs(row).sum())
        return row + budget * score / float(np.abs(score).sum())


@pytest.fixture(autouse=True)
def _clear_seen():
    SEEN.clear()


def _market():
    """Open and close prices with every execution case in them.

    The open moves away from the close, so the sizing bases differ. SYM2 has
    no prices on bars 10 to 12: bar 9's order for it is rejected on bar 10,
    and at the rebalance bar 12 it is held and not tradable, so locked. SYM4
    delists on bar 17 (settled on bar 18) and SYM5 on bar 26 at a close of 0.
    """
    rng = np.random.default_rng(3)
    close = 20.0 * np.cumprod(1 + rng.normal(0.0, 0.03, (N_BARS, N_SYMBOLS)), axis=0)
    open_ = close * (1 + rng.normal(0.0, 0.01, (N_BARS, N_SYMBOLS)))
    open_[[10, 11, 12], 2] = NAN
    close[[10, 11, 12], 2] = NAN
    open_[18:, 4] = close[18:, 4] = NAN
    close[26, 5] = 0.0
    open_[27:, 5] = close[27:, 5] = NAN
    scores = rng.normal(0.0, 1.0, (N_BARS, N_SYMBOLS))
    return open_, close, scores


def _prices(open_, close) -> FrameDataset:
    return FrameDataset(pd.DataFrame({
        "timestamp": np.repeat(BARS, N_SYMBOLS),
        "symbol": SYMBOLS * N_BARS,
        "open": open_.ravel(),
        "close": close.ravel(),
    }))


def _panel(values, name=None):
    return xr.DataArray(values, dims=("timestamp", "symbol"), coords={"timestamp": BARS, "symbol": SYMBOLS}, name=name)


def _construct(direction, execution=None):
    open_, close, scores = _market()
    dataset = _prices(open_, close)
    weights = DecisionInputs(
        dataset,
        Proportional(ProportionalConfig(direction)),
        fill_column="open",
        valuation_column="close",
        rebalance_periods=REBALANCE,
        anchor=BARS[0],
        execution=execution,
    ).weights(_panel(scores).to_dataset(name="score"))
    return dataset, weights["weight"], close


def _engine_weights(dataset, weights, close, settings):
    """The weights the engine holds at each bar's close for ``weights``."""
    config = WeightsBacktestConfig(
        price_dataset=dataset, start_date=str(BARS[0].date()), end_date=str(BARS[-1].date()),
        output_dir=None, rebalance_periods=1, fees=settings.fees, slippage=settings.slippage,
        init_cash=1000.0, fill_price_column="open", valuation_price_column="close",
        sizing_basis=settings.sizing_basis, trading_days_per_year=252, session_minutes_per_day=390,
    )
    simulation = WeightsVectorBt(config).run_weights(weights).simulation
    shares = np.zeros((N_BARS, N_SYMBOLS))
    orders = simulation.orders
    for timestamp, symbol, side, size in zip(
        orders["timestamp"].values, orders["symbol"].values, orders["side"].values, orders["size"].values
    ):
        shares[BARS.get_loc(pd.Timestamp(timestamp)), SYMBOLS.index(str(symbol))] += size if side == "Buy" else -size
    shares = np.cumsum(shares, axis=0)
    valuation = pd.DataFrame(close).ffill().to_numpy()
    worth = np.where(shares != 0, shares * np.nan_to_num(valuation), 0.0)
    return worth / simulation.value.values[:, None], simulation


@pytest.mark.parametrize("direction", ["long_only", "long_short"])
@pytest.mark.parametrize("sizing_basis", ["fill", "valuation"])
def test_a_rule_sees_the_weights_the_engine_holds_with_costs(sizing_basis, direction):
    settings = ExecutionSettings(sizing_basis=sizing_basis, fees=0.002, slippage=0.001)
    dataset, weights, close = _construct(direction, settings)

    held, simulation = _engine_weights(dataset, weights, close, settings)

    # Every case is exercised: a rejection, a lock, both settlements.
    assert simulation.rejected_orders
    assert len(simulation.settlements) == 2
    assert any(bool(context.locked.any()) for context in SEEN)
    assert (SEEN[0].current_weights.values == 0.0).all()
    for context in SEEN[1:]:
        bar = BARS.get_loc(context.timestamp)
        np.testing.assert_allclose(context.current_weights.values, held[bar], rtol=1e-9, atol=1e-12)


def test_without_execution_the_replay_sizes_at_the_fill_price_without_costs():
    dataset, default, close = _construct("long_only")
    seen = list(SEEN)
    SEEN.clear()
    _, explicit, _ = _construct("long_only", ExecutionSettings())

    np.testing.assert_array_equal(default.values, explicit.values)
    held, _ = _engine_weights(dataset, default, close, ExecutionSettings())
    for context in seen[1:]:
        bar = BARS.get_loc(context.timestamp)
        np.testing.assert_allclose(context.current_weights.values, held[bar], rtol=1e-9, atol=1e-12)
