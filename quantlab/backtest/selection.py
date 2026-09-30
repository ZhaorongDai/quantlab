"""When a cross-sectional backtest rebalances.

When to rebalance is the backtester's decision, so it stays in the backtest
layer: ``rebalance_mask`` marks the rebalance bars. What to hold on a
rebalance bar is the portfolio layer's decision (``quantlab.portfolio``);
which symbols can be traded there is the price dataset's
(``MarketDataset.tradable_bars``).
"""

import numpy as np


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
