"""The approximate backtest's execution, compiled: a book of weights filled at prices and marked at closes.

The approximate backtest (``quantlab.backtest.approximate``) gives up most of
the Execution rules (``quantlab.execution.rules``) for speed. What it keeps is
here, once, for the backtest's simulation and for the mean-variance solve
that drifts its own holdings forward in time with a turnover penalty
(``quantlab.portfolio.predefined.approximate._support.mean_variance_kernels.solve_forward``):

- a weight decided at bar t fills at bar t + 1's fill price; an order on a
  symbol with none there is rejected and the symbol keeps its holding,
  marked at the previous bar's valuation price (``entry_prices``,
  ``simulate``), as a backtest run rejects it;
- between fills a book is its weights at their fill prices, marked at each
  bar's valuation price, the rest cash (``mark_book``); a price missing on
  either side counts as no change (``price_relative``);
- each fill costs one rate times its traded weight, against the previous book
  marked at the fill's prices (``simulate``);
- a holding delisted on bar b is settled into cash at its last valuation on
  b + 1, before that bar's fill, and a fill on b + 1 does not buy it back
  (``simulate``), as a backtest run settles it.

Everything is compiled with numba (``cache=True``: the compiled code is kept
in ``__pycache__`` beside this file, or under ``NUMBA_CACHE_DIR``; an install
that cannot write there compiles again in each process). Import this module
only where it is used: importing it imports numba.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def price_relative(price: float, entry: float) -> float:
    """``price / entry``, 1.0 where it is not finite (a price missing on either side)."""
    relative = price / entry
    return relative if np.isfinite(relative) else 1.0


@njit(cache=True)
def mark_book(weights, entry, prices, marked) -> float:
    """Mark a book at ``prices``: ``marked`` its positions' values, returning the book's value, cash included.

    ``weights`` are the book's weights when it was filled at ``entry``;
    ``1 - sum(weights)`` was cash. The value is per unit of the book's value
    at the fill.
    """
    value = 1.0
    for i in range(weights.shape[0]):
        value -= weights[i]
        marked[i] = weights[i] * price_relative(prices[i], entry[i])
        value += marked[i]
    return value


@njit(cache=True)
def ffill(values):
    """Forward-fill NaN down the rows of a ``[T, S]`` array."""
    out = values.copy()
    for t in range(1, out.shape[0]):
        for i in range(out.shape[1]):
            if not np.isfinite(out[t, i]):
                out[t, i] = out[t - 1, i]
    return out


@njit(cache=True)
def entry_prices(fill, close):
    """``[T, S]``: the price a weight decided at the previous bar fills at: the fill price, else the previous close.

    ``close`` is forward-filled; the first bar has no previous close.
    """
    out = fill.copy()
    for t in range(out.shape[0]):
        for i in range(out.shape[1]):
            if not np.isfinite(out[t, i]):
                out[t, i] = close[t - 1, i] if t > 0 else np.nan
    return out


@njit(cache=True)
def settle_book(weights, entry, close, settle, marked) -> float:
    """Settle a book's holdings marked in ``settle`` into cash at ``close``; return the book's growth.

    The book is re-based at ``close``: ``weights`` become the positions'
    weights of the book's value there, ``entry`` becomes ``close``, and a
    settled position's weight becomes cash. Returns the book's value at
    ``close`` per unit of its value at its last base, 1.0 when nothing held
    is settled (the book is then left as it was).
    """
    held = False
    for i in range(weights.shape[0]):
        if settle[i] and weights[i] != 0.0:
            held = True
    if not held:
        return 1.0
    grown = mark_book(weights, entry, close, marked)
    for i in range(weights.shape[0]):
        weights[i] = 0.0 if settle[i] else marked[i] / grown
        if np.isfinite(close[i]):
            entry[i] = close[i]
    return grown


@njit(cache=True)
def simulate(targets, decisions, entry, unfilled, close, settle, cost):
    """Simulate the targets decided at ``decisions`` in one pass over the bars; ``(value, traded, fill_value, filled)``.

    ``targets[k]`` (``[K, S]``; an all-NaN row holds) is decided at row
    ``decisions[k]`` (in time order, each before the last row) and fills at
    the next row's ``entry`` price; ``unfilled`` (``[T, S]`` booleans) marks
    where the raw fill price is missing, where an order is rejected and the
    symbol keeps its holding. ``close`` is the forward-filled valuation
    price. ``settle`` (``[T, S]`` booleans) marks the bar after each
    delisting: a holding there turns into cash at ``close`` before the bar's
    fill, which leaves the symbol out (``settle_book``). A fill costs
    ``cost`` times its traded weight against the previous book marked at the
    fill's prices. Returns the value per bar (1.0 before
    the first fill); for each fill, in order, the signed weight traded in
    each symbol (``[F, S]``, the target less the previous book marked at the
    fill's prices) and the value it was traded against (the book before the
    fill's costs); and which decisions filled.
    """
    T, S = close.shape
    K = targets.shape[0]
    filled = np.ones(K, dtype=np.bool_)
    for k in range(K):
        for i in range(S):
            if not np.isfinite(targets[k, i]):
                filled[k] = False
                break
    F = filled.sum()
    value = np.ones(T)
    traded = np.zeros((F, S))
    fill_value = np.empty(F)
    weights = np.zeros(S)
    book_entry = np.zeros(S)
    marked = np.zeros(S)
    at_fill = 1.0  # the value at the last fill, after its costs
    live = False
    fills = 0
    due = -1  # a decision whose fill is due at this bar
    decision = 0
    target = np.zeros(S)
    for t in range(T):
        if live:
            at_fill *= settle_book(weights, book_entry, close[t], settle[t], marked)
        if due >= 0:
            grown = mark_book(weights, book_entry, entry[t], marked) if live else 1.0
            total = 0.0
            for i in range(S):
                held = marked[i] / grown if live else 0.0
                if settle[t, i]:
                    target[i] = 0.0
                elif unfilled[t, i]:
                    target[i] = held
                else:
                    target[i] = targets[due, i]
                traded[fills, i] = target[i] - held
                total += abs(traded[fills, i])
            at_fill *= grown
            fill_value[fills] = at_fill
            at_fill *= 1.0 - cost * total
            weights[:] = target
            book_entry[:] = entry[t]
            live = True
            fills += 1
            due = -1
        if live:
            value[t] = at_fill * mark_book(weights, book_entry, close[t], marked)
        while decision < K and decisions[decision] == t:
            if filled[decision]:
                due = decision
            decision += 1
    return value, traded, fill_value, filled
