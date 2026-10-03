"""The return window and staleness a rule reads (#88, ADR 0014).

What is locked here, and what turns it red (hand-built panels, no vectorbt):

- One-bar returns come from the last known valuation price: a halt shows as
  zero returns, then the whole gap on the bar the symbol trades again.
- Staleness counts the bars since a symbol's last real valuation price,
  resets when prices return, and is NaN before the first price (and, see
  test_portfolio_history_window.py, beyond the rule's ``history_bars``).
- The window and staleness at a bar are the same whether the panel ends at
  that bar or later (nothing after the bar is read).
- Ledoit-Wolf leaves out a symbol staler than `max_stale_bars` and keeps one
  at exactly `max_stale_bars`; a halt inside the window keeps it covered.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import LedoitWolfConfig
from quantlab.base.portfolio import PortfolioConstructor, PortfolioContext
from quantlab.dataset.memory import FrameDataset
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel

SYMBOLS = ["AAA", "BBB"]


class _Watch(PortfolioConstructor):
    """Holds nothing and records every context."""

    config_cls = LedoitWolfConfig
    seen: list = []

    @property
    def lookback_bars(self):
        return self.config.lookback_bars

    def construct(self, context):
        type(self).seen.append(context)
        return xr.zeros_like(context.current_weights)


def _prices(values):
    ts = pd.bdate_range("2024-01-01", periods=len(values))
    return xr.DataArray(
        np.asarray(values, float), dims=("timestamp", "symbol"), coords={"timestamp": ts, "symbol": SYMBOLS}
    )


def _run(prices, lookback, rebalance_at):
    """The contexts ``DecisionInputs.context`` builds at ``rebalance_at``, holding nothing."""
    inputs = DecisionInputs(
        FrameDataset(xr.Dataset({"open": prices, "close": prices})),
        _Watch(LedoitWolfConfig(lookback_bars=lookback)),
        fill_column="open",
        valuation_column="close",
        rebalance_periods=1,
        anchor=prices.timestamp.values[0],
    )
    flat = xr.DataArray(np.zeros(len(SYMBOLS)), dims="symbol", coords={"symbol": SYMBOLS})
    predictions = xr.Dataset({"ret": ("symbol", np.ones(len(SYMBOLS)))}, coords={"symbol": SYMBOLS})
    return [inputs.context(prices.timestamp.values[t], predictions, flat) for t in rebalance_at]


PRICES = [[10, 20], [11, 21], [12, np.nan], [13, np.nan], [14, 25], [15, 26]]


def test_a_halt_shows_as_zero_returns_then_the_whole_gap():
    context = _run(_prices(PRICES), lookback=5, rebalance_at=[5])[0]

    bbb = context.returns.sel(symbol="BBB").values
    np.testing.assert_allclose(bbb, [21 / 20 - 1, 0.0, 0.0, 25 / 21 - 1, 26 / 25 - 1])


def test_staleness_counts_bars_since_the_last_real_price_and_resets():
    values = [[np.nan, 20], [10, np.nan], [11, np.nan], [12, 22]]
    seen = _run(_prices(values), lookback=2, rebalance_at=[0, 1, 2, 3])  # history_bars 3

    staleness = [context.staleness.values.tolist() for context in seen]
    assert np.isnan(staleness[0][0]) and staleness[0][1] == 0
    assert staleness[1:] == [[0, 1], [0, 2], [0, 0]]


def test_the_window_and_staleness_read_nothing_after_the_bar():
    full = _run(_prices(PRICES), lookback=3, rebalance_at=[3])[0]
    cut = _run(_prices(PRICES[:4]), lookback=3, rebalance_at=[3])[0]

    xr.testing.assert_identical(full.returns, cut.returns)
    xr.testing.assert_identical(full.staleness, cut.staleness)


def _context(returns, staleness):
    n = returns.shape[1]
    symbols = [f"S{i}" for i in range(n)]
    coords = {"symbol": symbols}
    return PortfolioContext(
        timestamp=pd.Timestamp("2024-03-01"),
        predictions=xr.Dataset(coords=coords),
        tradable=xr.DataArray(np.ones(n, bool), dims="symbol", coords=coords),
        current_weights=xr.DataArray(np.zeros(n), dims="symbol", coords=coords),
        returns=xr.DataArray(
            returns, dims=("timestamp", "symbol"),
            coords={"timestamp": pd.bdate_range("2024-01-01", periods=returns.shape[0]), **coords},
        ),
        staleness=xr.DataArray(np.asarray(staleness, float), dims="symbol", coords=coords),
    )


def test_ledoit_wolf_leaves_out_symbols_staler_than_max_stale_bars():
    returns = np.random.default_rng(0).normal(0, 0.02, size=(30, 3))
    risk = LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=30, max_stale_bars=5))

    estimate = risk.estimate(_context(returns, [0, 5, 6]))

    assert estimate.symbols.tolist() == ["S0", "S1"]


def test_a_halt_inside_the_window_keeps_a_symbol_covered():
    values = [[10.0 + t, 20.0 + 2 * t + (t % 3)] for t in range(12)]
    values[4][1] = values[5][1] = np.nan  # BBB halts on bars 4-5
    context = _run(_prices(values), lookback=8, rebalance_at=[11])[0]

    estimate = LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=8)).estimate(context)

    assert estimate.symbols.tolist() == ["AAA", "BBB"]


def test_max_stale_bars_defaults_to_five_and_round_trips():
    risk = LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=20, max_stale_bars=3))

    config = json.loads(json.dumps(risk.get_config()))

    assert LedoitWolfConfig(lookback_bars=20).max_stale_bars == 5
    assert config["max_stale_bars"] == 3
    assert LedoitWolfRiskModel.from_config(config) == risk


def test_a_negative_max_stale_bars_is_refused():
    with pytest.raises(ValueError, match="max_stale_bars"):
        LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=20, max_stale_bars=-1))
