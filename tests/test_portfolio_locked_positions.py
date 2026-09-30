"""Locked positions: held symbols that are not tradable at the bar (#87, ADR 0014).

What is locked here, and what turns it red (hand-built contexts and panels, no
vectorbt):

- The context says which symbols are locked: held and not tradable.
- Top-n keeps a locked position at its current weight, never picks it, and
  splits the remaining budget over its k picks: (1 - locked) / k long-only,
  0.5 minus the side's locked exposure per side long-short; a side with no
  budget left adds nothing. With nothing locked the book is unchanged.
- The driver refuses a rule that moves a locked position or gives weight to
  a symbol that is neither tradable nor held, naming the rule, bar and symbol.
- The drifted current weights handed to a rule model rejected orders: a
  symbol without a fill price at the fill bar keeps its pre-trade holding.
- Delisted symbols settle into cash on the bar after their last price.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import TopNConfig
from quantlab.base.portfolio import PortfolioConstructor, PortfolioContext
from quantlab.portfolio.predefined.top_n import TopNConstructor

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _context(scores, tradable, current):
    coords = {"symbol": SYMBOLS}
    return PortfolioContext(
        timestamp=pd.Timestamp("2024-01-02"),
        predictions=xr.Dataset({"ret": ("symbol", np.asarray(scores, float))}, coords=coords),
        tradable=xr.DataArray(np.asarray(tradable, bool), dims="symbol", coords=coords),
        current_weights=xr.DataArray(np.asarray(current, float), dims="symbol", coords=coords),
    )


def test_the_context_marks_held_untradable_symbols_as_locked():
    context = _context([1, 2, 3, 4, 5], [True, False, False, True, True], [0.2, 0.3, 0.0, 0.0, 0.5])

    assert context.locked.values.tolist() == [False, True, False, False, False]


def test_long_only_top_n_keeps_a_locked_position_and_splits_the_rest():
    # BBB is held at 0.3 and halted; it scores highest but is not picked again.
    context = _context([0.1, 9.0, 0.5, 0.4, 0.2], [True, False, True, True, True], [0.2, 0.3, 0.0, 0.0, 0.5])

    row = TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(context).values

    np.testing.assert_array_equal(row, [0.0, 0.3, 0.35, 0.35, 0.0])


def test_long_short_top_n_gives_each_side_its_remaining_budget():
    # AAA is a locked long of 0.1, EEE a locked short of -0.2.
    context = _context(
        [9.0, 0.5, 0.4, 0.3, -9.0], [False, True, True, True, False], [0.1, 0.0, 0.0, 0.0, -0.2]
    )

    row = TopNConstructor(TopNConfig(direction="long_short", top_n=1)).construct(context).values

    np.testing.assert_allclose(row, [0.1, 0.4, 0.0, -0.3, -0.2], rtol=0, atol=1e-15)


def test_a_side_without_budget_adds_nothing():
    context = _context([0.1, 0.2, 0.3, 0.4, 0.5], [False, True, True, True, True], [1.0, 0.0, 0.0, 0.0, 0.0])

    row = TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(context).values

    np.testing.assert_array_equal(row, [1.0, 0.0, 0.0, 0.0, 0.0])


def test_nothing_locked_is_the_plain_equal_weight_book():
    context = _context([0.1, 0.2, 0.3, 0.4, 0.5], [True] * 5, [0.2, 0.2, 0.2, 0.2, 0.2])

    row = TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(context).values

    np.testing.assert_array_equal(row, [0.0, 0.0, 0.0, 0.5, 0.5])


class _Mover(PortfolioConstructor):
    """Puts everything in AAA, ignoring locks."""

    config_cls = TopNConfig

    def construct(self, context):
        row = xr.zeros_like(context.current_weights)
        row.loc[{"symbol": "AAA"}] = 1.0
        return row


def _panel(values):
    ts = pd.bdate_range("2024-01-01", periods=len(values))
    return xr.DataArray(
        np.asarray(values, float), dims=("timestamp", "symbol"), coords={"timestamp": ts, "symbol": SYMBOLS[:2]}
    )


def test_the_driver_refuses_a_rule_that_moves_a_locked_position():
    # Bar 0 buys AAA and BBB half each; BBB is halted on bar 2, a rebalance bar.
    class _Half(PortfolioConstructor):
        config_cls = TopNConfig

        def construct(self, context):
            if context.current_weights.sum() == 0:
                return xr.full_like(context.current_weights, 0.5)
            row = xr.zeros_like(context.current_weights)
            row.loc[{"symbol": "AAA"}] = 1.0
            return row

    prices = _panel([[10, 20], [10, 20], [10, np.nan], [10, 20]])
    predictions = xr.Dataset({"ret": prices * 0 + 1.0})
    tradable = prices.notnull()

    with pytest.raises(ValueError, match=r"_Half.*BBB.*2024-01-03"):
        _Half(TopNConfig(direction="long_only", top_n=1)).construct_panel(
            predictions, tradable, np.array([True, False, True, False]),
            fill_price=prices, valuation_price=prices,
        )


def test_the_driver_refuses_weight_on_a_symbol_neither_tradable_nor_held():
    prices = _panel([[10, np.nan], [10, 20]])
    predictions = xr.Dataset({"ret": prices.fillna(0) * 0 + 1.0})

    class _Blind(PortfolioConstructor):
        config_cls = TopNConfig

        def construct(self, context):
            return xr.full_like(context.current_weights, 0.5)

    with pytest.raises(ValueError, match=r"_Blind.*BBB.*2024-01-01"):
        _Blind(TopNConfig(direction="long_only", top_n=1)).construct_panel(
            predictions, prices.notnull(), np.array([True, False]), fill_price=prices, valuation_price=prices
        )


class _Recorder(PortfolioConstructor):
    config_cls = TopNConfig
    seen: list = []

    def construct(self, context):
        type(self).seen.append(context)
        if len(type(self).seen) == 1:
            return xr.full_like(context.current_weights, 0.5)
        if len(type(self).seen) == 2:
            row = xr.zeros_like(context.current_weights)
            row.loc[{"symbol": "AAA"}] = 1.0
            return row.where(~context.locked, context.current_weights)
        return context.current_weights.copy()


def test_the_current_weights_model_a_rejected_order():
    """Bar 0 buys half-half at bar 1's price. Bar 2 asks to sell BBB, but BBB has
    no price at bar 3, so the order is rejected and BBB is still held at bar 4."""
    _Recorder.seen = []
    fill = _panel([[10, 20], [10, 20], [10, 20], [10, np.nan], [10, 40], [10, 40]])
    predictions = xr.Dataset({"ret": fill.fillna(0) * 0 + 1.0})

    _Recorder(TopNConfig(direction="long_only", top_n=1)).construct_panel(
        predictions, fill.notnull(), np.array([True, False, True, False, True, False]),
        fill_price=fill, valuation_price=fill,
    )

    at_bar_4 = _Recorder.seen[2].current_weights.values
    # AAA: 0.5 of 1.0 stays 0.5 (flat price); BBB doubled from 20 to 40.
    np.testing.assert_allclose(at_bar_4, [0.5 / 1.5, 1.0 / 1.5], rtol=1e-12)


def test_a_delisted_holding_is_cash_after_its_settlement():
    """BBB's last price is bar 1; it settles at bar 2, so at bar 3 it is not
    held (and not locked) although it has no price."""
    _Recorder.seen = []
    fill = _panel([[10, 20], [10, 20], [10, np.nan], [10, np.nan], [10, np.nan]])
    predictions = xr.Dataset({"ret": fill.fillna(0) * 0 + 1.0})
    delisted = xr.zeros_like(fill, dtype=bool)
    delisted[1, 1] = True

    _Recorder(TopNConfig(direction="long_only", top_n=1)).construct_panel(
        predictions, fill.notnull(), np.array([True, False, False, True, False]),
        fill_price=fill, valuation_price=fill, delisted=delisted,
    )

    later = _Recorder.seen[1]
    np.testing.assert_allclose(later.current_weights.values, [0.5, 0.0])
    assert not later.locked.values.any()
