"""Locks for `quantlab/backtest/selection.py`: when the backtest rebalances and what it may fill.

When to rebalance is the backtester's decision (the rule of what to hold is
the portfolio layer's, locked in `test_portfolio_top_n.py`):

- **D-18 schedule.** `rebalance_mask` anchors at the first bar and steps by
  `rebalance_periods`; the last bar never rebalances, since its signal has no
  next bar inside the window to fill on.
- **Tradability (ADR 0014).** `MarketDataset.tradable_bars` marks a symbol tradable at bar t
  exactly when its fill price at t + 1 is finite.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.selection import rebalance_mask


def test_rebalance_mask_anchors_at_first_bar_and_steps_by_period():
    """n_bars=21, p=5: True exactly at 0, 5, 10, 15. Index 20 is a multiple of
    5 but it is the window's last bar, whose signal has no t+1 fill bar inside
    the window, so it is False (D-18 anchor; RESEARCH Pitfall 13). Goes red if
    the anchor moves off bar 0 or the last-bar exclusion is dropped."""
    mask = rebalance_mask(21, 5)

    assert mask.dtype == bool and mask.shape == (21,)
    assert np.flatnonzero(mask).tolist() == [0, 5, 10, 15]
    assert not mask[20]


def test_rebalance_mask_period_one_is_every_bar_but_the_last():
    """p=1 rebalances every bar except the last one (Pitfall 13)."""
    assert rebalance_mask(4, 1).tolist() == [True, True, True, False]


def test_rebalance_mask_period_longer_than_window_rebalances_once():
    """A period longer than the window still rebalances on the anchor bar, so
    the book is built once and held (D-18)."""
    assert rebalance_mask(4, 10).tolist() == [True, False, False, False]


def test_rebalance_mask_rejects_non_positive_period():
    """p=0 would make `mask[::0]` raise an opaque slicing error, and p=-1 would
    step backwards from the anchor. Both must be a ValueError naming the
    parameter."""
    for period in (0, -1):
        with pytest.raises(ValueError, match="rebalance_periods"):
            rebalance_mask(10, period)


def test_a_symbol_is_tradable_where_it_has_a_fill_price_at_the_bar(tmp_path):
    """Tradability reads only the bar itself (ADR 0014): OLD, delisted after
    bar 1, is tradable on bars 0-1; NEW, listing at bar 2, from bar 2; the
    last bar is tradable like any other."""
    from tests.backtest_fixtures import make_stock_dataset, write_price_store

    dataset = make_stock_dataset(write_price_store(tmp_path, n_bars=4))
    fill = xr.Dataset(
        {"adjOpen": (("timestamp", "symbol"), [[10.0, np.nan], [11.0, np.nan], [np.nan, 20.0], [np.nan, 21.0]])},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=4), "symbol": ["OLD", "NEW"]},
    )

    tradable = dataset.tradable_bars(fill, "adjOpen")

    assert tradable.dims == ("timestamp", "symbol")
    assert tradable.values.tolist() == [
        [True, False],
        [True, False],
        [False, True],
        [False, True],
    ]