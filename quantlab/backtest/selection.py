"""When a cross-sectional backtest rebalances, and which symbols it may hold.

When to rebalance is the backtester's decision, so it stays in the backtest
layer: ``rebalance_mask`` marks the rebalance bars, and ``next_bar_eligible``
the symbols the vectorised engine can fill at the next bar. What to hold on
a rebalance bar is the portfolio layer's decision (``quantlab.portfolio``),
handed both.
"""

import numpy as np
import xarray as xr


def rebalance_mask(n_bars: int, rebalance_periods: int) -> np.ndarray:
    """Return a boolean mask marking the bars on which the portfolio rebalances.

    The first bar rebalances and so does every ``rebalance_periods``-th bar
    after it. The last bar never rebalances: a signal formed there has no
    following bar inside the window to fill on.

    Parameters
    ----------
    n_bars : int
        Number of bars in the backtest window.
    rebalance_periods : int
        Rebalance every this many bars.

    Returns
    -------
    np.ndarray
        A boolean array of length ``n_bars``.

    Raises
    ------
    ValueError
        If ``rebalance_periods`` is smaller than 1.

    Examples
    --------
    >>> rebalance_mask(7, 3)
    array([ True, False, False,  True, False, False, False])
    """
    if rebalance_periods < 1:
        raise ValueError(
            f"rebalance_periods must be >= 1, got {rebalance_periods}"
        )
    mask = np.zeros(n_bars, dtype=bool)
    mask[::rebalance_periods] = True
    if n_bars > 0:
        mask[-1] = False
    return mask


def next_bar_eligible(fill_price: xr.DataArray) -> xr.DataArray:
    """Return which symbols can be filled at the bar after each bar.

    A signal at bar t fills at bar t + 1's fill price, so a symbol is
    eligible at t when that price is finite. Pass the raw, not
    forward-filled, fill-price panel, so a symbol with no price at the next
    bar (delisted) is never eligible. The last bar has no next bar and is
    never eligible.

    Parameters
    ----------
    fill_price : xr.DataArray
        Fill prices on ``(timestamp, symbol)``.

    Returns
    -------
    xr.DataArray
        Booleans on the same labels.

    Examples
    --------
    >>> import pandas as pd
    >>> prices = xr.DataArray(
    ...     [[10.0, 20.0], [11.0, np.nan], [12.0, 21.0]],
    ...     dims=("timestamp", "symbol"),
    ...     coords={"timestamp": pd.bdate_range("2024-01-01", periods=3), "symbol": ["AAA", "BBB"]},
    ... )
    >>> next_bar_eligible(prices).values
    array([[ True, False],
           [ True,  True],
           [False, False]])
    """
    return np.isfinite(fill_price.shift(timestamp=-1))
