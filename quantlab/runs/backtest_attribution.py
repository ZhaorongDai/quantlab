"""Backtest attribution: where a strategy's excess over its benchmark comes from.

A model backtest picks a few names from a universe and pays to trade them,
so its excess over the benchmark has three parts:

- *universe*: the universe held equally weighted against the benchmark,
  which the strategy earns or loses whatever it picks;
- *selection*: the strategy before costs against that equal-weighted
  universe, what the scores and the portfolio rule add;
- *costs*: the strategy after costs against before them.

The parts are differences of log growth, so they add up to the strategy's
log growth over the benchmark exactly. ``rebalanced_group_values`` gives the
universe curve, and the curves of the universe cut into groups by score,
which show where in the ranking the scores carry information.
"""

import numpy as np


def rebalanced_group_values(fill, valuation, scores, rebalance, groups: int, delisted=None) -> np.ndarray:
    """Return the value curves of the score groups, each held equally weighted between rebalances.

    On every rebalance bar t the universe is each symbol with a finite score
    on t and a fill price on t + 1. It is ranked by score (ties by symbol
    order) and cut into ``groups`` groups of equal count, the first holding
    the lowest scores; when it has fewer symbols than ``groups``, every group
    holds cash until the next rebalance. Each group is bought equally
    weighted at the fill prices of t + 1 and held, valued at the valuation
    prices, until the fill bar of the next rebalance, where it is sold at the
    fill prices and the next group bought: no costs, and the same
    fill-a-bar-later timing as the simulation. A holding marked in
    ``delisted`` is settled as the engine settles it, at its last valuation
    price; any other holding without a later price keeps its last one. A
    group with no symbol holds its value flat. Each curve starts at 1.0 and
    stays there until the first fill.

    Parameters
    ----------
    fill, valuation : array-like
        Fill and valuation prices on ``[T, S]``, NaN where a bar has none.
    scores : array-like
        Scores on ``[T, S]``; only the rebalance bars' rows are read, and NaN
        leaves a symbol out of the universe.
    rebalance : array-like
        Booleans on ``[T]``: the bars whose scores are traded.
    groups : int
        How many groups to cut the universe into; 1 gives the universe.
    delisted : array-like, optional
        Booleans on ``[T, S]``: a symbol's last bar before it delists, as the
        price dataset's ``delisting_bars`` marks it.

    Returns
    -------
    np.ndarray
        ``[groups, T]``, the value of 1.0 invested in each group.

    Raises
    ------
    ValueError
        If ``groups`` is below 1 or the panels' shapes differ.

    Examples
    --------
    >>> fill = np.array([[10.0, 20.0], [10.0, 20.0], [11.0, 22.0]])
    >>> valuation = np.array([[10.0, 20.0], [12.0, 20.0], [12.0, 24.0]])
    >>> scores = np.array([[1.0, 2.0], [np.nan, np.nan], [np.nan, np.nan]])
    >>> rebalanced_group_values(fill, valuation, scores, [True, False, False], groups=2)
    array([[1. , 1.2, 1.2],
           [1. , 1. , 1.2]])
    """
    if groups < 1:
        raise ValueError(f"groups must be at least 1, got {groups}")
    raw_fill = np.asarray(fill, dtype=np.float64)
    shape = raw_fill.shape
    if np.shape(valuation) != shape or np.shape(scores) != shape:
        raise ValueError(
            f"fill, valuation and scores must share one [T, S] shape, got {shape}, "
            f"{np.shape(valuation)} and {np.shape(scores)}"
        )
    held_valuation = _forward_fill(np.asarray(valuation, dtype=np.float64))
    # A delisted holding is worth its last valuation from the bar after its
    # mark on, at any later fill too: the engine settles it into cash there.
    settled = np.zeros(shape, dtype=bool)
    if delisted is not None:
        settled[1:] = np.logical_or.accumulate(np.asarray(delisted, dtype=bool), axis=0)[:-1]
    exit_price = np.where(settled, held_valuation, _forward_fill(raw_fill))
    scores = np.asarray(scores, dtype=np.float64)
    n_bars = shape[0]
    entries = {t + 1 for t in np.flatnonzero(np.asarray(rebalance, dtype=bool)) if t + 1 < n_bars}

    values = np.ones((groups, n_bars))
    level = np.ones(groups)
    members: list[np.ndarray] = [np.array([], dtype=int)] * groups
    entry = None
    for t in range(n_bars):
        if t in entries:
            if entry is not None:
                level = _group_levels(level, members, exit_price[t], raw_fill[entry])
            entry = t
            members = _cut(scores[t - 1], raw_fill[t], groups)
        if entry is not None:
            values[:, t] = _group_levels(level, members, held_valuation[t], raw_fill[entry])
    return values


def annualized_log_growth(curve, years: float) -> float:
    """Return a value curve's log growth from its first to its last value, per year.

    Parameters
    ----------
    curve : array-like
        Values on the window's bars.
    years : float
        The window's length in years.

    Returns
    -------
    float
        ``log(curve[-1] / curve[0]) / years``.

    Examples
    --------
    >>> round(annualized_log_growth([1.0, 1.05, 1.21], years=2.0), 6)
    0.09531
    """
    values = np.asarray(curve, dtype=np.float64)
    return float(np.log(values[-1] / values[0]) / years)


def excess_decomposition(strategy, gross, universe, benchmark, years: float) -> dict[str, float]:
    """Split the strategy's annualised log growth over the benchmark into its three parts.

    ``strategy`` is the value curve after costs, ``gross`` the same weights
    simulated without costs (``None`` when the engine cannot), ``universe``
    the equal-weighted universe (``rebalanced_group_values`` with one group)
    and ``benchmark`` the benchmark's (``None`` without one). Each part is
    the difference of two curves' ``annualized_log_growth``: ``universe`` is
    universe over benchmark, ``selection`` gross over universe, ``costs``
    strategy over gross, and ``total`` their sum, the strategy over the
    benchmark. Without a benchmark ``universe`` is left out and ``total`` is
    measured over the universe; without ``gross`` ``costs`` is left out and
    ``selection`` is the strategy over the universe.

    Parameters
    ----------
    strategy, universe : array-like
        Value curves on the window's bars.
    gross, benchmark : array-like or None
        Value curves on the window's bars, or ``None``.
    years : float
        The window's length in years.

    Returns
    -------
    dict of str to float
        ``universe`` (with a benchmark), ``selection``, ``costs`` (with
        ``gross``) and ``total``.

    Examples
    --------
    >>> parts = excess_decomposition([1.0, 1.21], [1.0, 1.331], [1.0, 1.1], [1.0, 1.0], years=1.0)
    >>> {k: round(v, 4) for k, v in parts.items()}
    {'universe': 0.0953, 'selection': 0.1906, 'costs': -0.0953, 'total': 0.1906}
    """
    def growth(curve) -> float:
        return annualized_log_growth(curve, years)

    parts = {}
    if benchmark is not None:
        parts["universe"] = growth(universe) - growth(benchmark)
    before_costs = strategy if gross is None else gross
    parts["selection"] = growth(before_costs) - growth(universe)
    if gross is not None:
        parts["costs"] = growth(strategy) - growth(gross)
    parts["total"] = sum(parts.values())
    return parts


def _cut(scores: np.ndarray, fill: np.ndarray, groups: int) -> list[np.ndarray]:
    """Return each group's symbol indices, lowest scores first."""
    eligible = np.flatnonzero(np.isfinite(scores) & np.isfinite(fill))
    if eligible.size < groups:
        return [np.array([], dtype=int)] * groups
    ranked = eligible[np.argsort(scores[eligible], kind="stable")]
    bucket = np.arange(ranked.size) * groups // max(ranked.size, 1)
    return [ranked[bucket == g] for g in range(groups)]


def _group_levels(level: np.ndarray, members: list[np.ndarray], price: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Return each group's value at ``price`` given its level at ``cost``."""
    out = level.copy()
    for g, held in enumerate(members):
        if held.size:
            out[g] = level[g] * np.mean(price[held] / cost[held])
    return out


def _forward_fill(panel: np.ndarray) -> np.ndarray:
    """Carry each column's last finite value forward over NaN."""
    index = np.where(np.isfinite(panel), np.arange(panel.shape[0])[:, None], 0)
    np.maximum.accumulate(index, axis=0, out=index)
    filled = panel[index, np.arange(panel.shape[1])]
    return filled
