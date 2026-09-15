"""Locks for `quantlab/backtest/selection.py`: the cross-sectional TopN selector.

Target weights are the Phase 5 optimizer interface (03.7 D-03), so every rule
that decides what the backtest actually trades is locked here with pure tests
on hand-built `(timestamp, symbol)` panels. No store, no model, no vectorbt.

What this file locks, and why each rule matters:

- **D-03 weights contract.** A non-rebalance row is all-NaN ("hold"), and a
  rebalance row is fully finite with every unselected symbol at exactly 0.0.
  A NaN on a rebalance bar is not "no position": vectorbt reads it as "keep
  the old position". The stale position then holds cash the new buys need, so
  the whole rebalance is silently blocked with no error (03.7-RESEARCH.md
  Pitfall 3).
- **D-09 book construction.** `long_only` puts 1/k on each of the top k names.
  `long_short` puts +0.5/k on the top k and -0.5/k on the bottom k, and the two
  books never share a symbol. An overlapping symbol would net to zero while
  still being counted twice in gross exposure, so the book would be both
  under-invested and mis-reported.
- **D-10** fixed `top_n`, and **D-12 eligibility**: a symbol needs a finite
  score AND a finite next-bar fill price. A symbol delisted at t+1 cannot be
  filled, so selecting it would leave its book weight uninvested. When fewer
  than `top_n` symbols are eligible, the book splits equally among the ones
  that are, and a warning names the bar.
- **Deterministic ties.** Equal scores resolve by symbol-axis order through a
  stable sort, so one panel always yields one set of weights (D-25
  reproducibility).
- **D-18 schedule** and **D-11 score-label resolution** (Task 2 of plan
  03.7-03, below the book tests), plus the D-03 invariant on randomized panels.
"""

import contextlib
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.backtest.selection import (
    CrossSectionTopNSelector,
    rebalance_mask,
    resolve_score_label,
)
from quantlab.base.config import CrossSectionBacktestConfig

NAN = np.nan
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _panel(values, symbols=SYMBOLS) -> xr.DataArray:
    """A `(timestamp, symbol)` DataArray on a business-day axis from 2024-01-01."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    timestamps = pd.bdate_range("2024-01-01", periods=values.shape[0])
    return xr.DataArray(
        values,
        dims=("timestamp", "symbol"),
        coords={"timestamp": timestamps, "symbol": list(symbols)},
    )


def _finite_fill(scores: xr.DataArray) -> xr.DataArray:
    """A next-bar fill price panel that is finite everywhere, on the score axes."""
    return xr.full_like(scores, 100.0)


def _select(direction, top_n, scores, fill=None, rebalance=None) -> np.ndarray:
    """Run the selector and return the `[T, S]` weight array."""
    if fill is None:
        fill = _finite_fill(scores)
    if rebalance is None:
        rebalance = np.ones(scores.sizes["timestamp"], dtype=bool)
    selector = CrossSectionTopNSelector(direction=direction, top_n=top_n)
    out = selector.select(scores, fill, np.asarray(rebalance, dtype=bool))
    assert out["weight"].dims == ("timestamp", "symbol")
    return out["weight"].values


@contextlib.contextmanager
def _warnings():
    """Capture loguru WARNING messages into a list; the sink is always removed."""
    messages: list = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


# --------------------------------------------------------------------------
# Task 1: book construction, eligibility, short books, ties (D-03/09/10/12)
# --------------------------------------------------------------------------

# Distinct scores: descending order is BBB(0.9) DDD(0.8) FFF(0.4) CCC(0.3)
# EEE(0.2) AAA(0.1).
DISTINCT = [0.1, 0.9, 0.3, 0.8, 0.2, 0.4]


def test_long_only_top_n_equal_weights_sum_to_one():
    """top_n=2 long_only: BBB and DDD get 0.5 each, the other four get exactly
    0.0 (not NaN), and the row sums to 1.0. Goes red if unselected symbols are
    left NaN (Pitfall 3), if the ranking is ascending, or if the book is not
    1/k equal weight."""
    w = _select("long_only", 2, _panel(DISTINCT))[0]

    assert w[1] == 0.5 and w[3] == 0.5
    others = w[[0, 2, 4, 5]]
    assert not np.isnan(others).any()
    assert (others == 0.0).all()
    assert w.sum() == 1.0


def test_long_short_books_are_half_each_and_disjoint():
    """top_n=2 long_short: the top two (BBB, DDD) get +0.25, the bottom two
    (AAA, EEE) get -0.25, the middle two get 0.0. Gross is 1.0, net is 0.0 and
    no symbol is in both books (D-09). Goes red if a book is sized 1/k instead
    of 0.5/k, or if the short book is taken from the top of the ranking."""
    w = _select("long_short", 2, _panel(DISTINCT))[0]

    assert w[1] == 0.25 and w[3] == 0.25
    assert w[0] == -0.25 and w[4] == -0.25
    assert w[2] == 0.0 and w[5] == 0.0
    assert np.abs(w).sum() == pytest.approx(1.0, abs=1e-12)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    longs, shorts = set(np.flatnonzero(w > 0)), set(np.flatnonzero(w < 0))
    assert longs.isdisjoint(shorts)


def test_nan_score_is_never_selected():
    """BBB would be the top name but its score is NaN, so it is ineligible and
    the next two finite scores (DDD, FFF) are chosen (D-12). Goes red if NaN
    survives into the ranking, where `-nan` sorts to an arbitrary position."""
    scores = list(DISTINCT)
    scores[1] = NAN
    w = _select("long_only", 2, _panel(scores))[0]

    assert w[1] == 0.0
    assert w[3] == 0.5 and w[5] == 0.5
    assert w.sum() == 1.0


def test_nan_next_fill_price_is_never_selected():
    """BBB has the top score but no fill price on t+1 (delisted), so it gets
    0.0 and DDD/FFF are chosen (D-12). Goes red if eligibility looks at the
    score alone: the order would target a symbol that cannot be filled."""
    scores = _panel(DISTINCT)
    fill = _finite_fill(scores)
    fill[0, 1] = NAN
    w = _select("long_only", 2, scores, fill)[0]

    assert w[1] == 0.0
    assert w[3] == 0.5 and w[5] == 0.5


def test_short_long_only_book_splits_among_available_and_warns():
    """Bar 0 has every symbol eligible; on bar 1 only CCC is. With top_n=2,
    bar 1 puts the whole book (1.0) on CCC, and exactly one WARNING fires,
    naming bar 1's timestamp and not bar 0's. Goes red if a short book keeps
    1/top_n (leaving cash idle), or if the warning is silent or names the
    wrong bar."""
    scores = _panel([DISTINCT, [NAN, NAN, 0.5, NAN, NAN, NAN]])
    timestamps = scores.timestamp.values

    with _warnings() as messages:
        w = _select("long_only", 2, scores)

    assert w[1, 2] == 1.0
    assert (w[1, [0, 1, 3, 4, 5]] == 0.0).all()
    assert len(messages) == 1
    assert str(pd.Timestamp(timestamps[1]).date()) in messages[0]
    assert str(pd.Timestamp(timestamps[0]).date()) not in messages[0]


def test_long_short_with_fewer_than_twice_top_n_eligible_stays_disjoint():
    """Three eligible symbols (BBB 0.9, DDD 0.5, FFF 0.1) with top_n=2: each
    book gets k = 3 // 2 = 1, so BBB is +0.5, FFF is -0.5, DDD is 0.0, and a
    warning fires. Goes red if k is computed as min(top_n, n_eligible): both
    books would then take two names and DDD would sit in both."""
    scores = _panel([NAN, 0.9, NAN, 0.5, NAN, 0.1])

    with _warnings() as messages:
        w = _select("long_short", 2, scores)[0]

    assert w[1] == 0.5
    assert w[5] == -0.5
    assert w[3] == 0.0
    assert (w[[0, 2, 4]] == 0.0).all()
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert len(messages) == 1


def test_no_eligible_symbol_gives_a_flat_row_not_nan():
    """Every score NaN on a rebalance bar: the row is all 0.0, never NaN, and a
    warning fires, in both directions. A flat row liquidates the book; a NaN
    row would silently keep yesterday's positions (Pitfall 3)."""
    scores = _panel([NAN] * len(SYMBOLS))

    for direction in ("long_only", "long_short"):
        with _warnings() as messages:
            w = _select(direction, 2, scores)[0]
        assert not np.isnan(w).any(), direction
        assert (w == 0.0).all(), direction
        assert len(messages) == 1, direction


def test_equal_scores_resolve_by_symbol_axis_order():
    """All scores equal, top_n=2, long_only: the first two symbols on the axis
    (AAA, BBB) are chosen, and two calls return identical arrays. Goes red if
    the tie-break depends on anything other than axis order (D-25).

    The six-symbol case alone cannot catch a non-stable sort: numpy's
    quicksort/heapsort fall back to a stable insertion sort below 16 elements,
    so mutating the sort kind stayed green on it (plan 03.7-03 mutation M3).
    A 20-symbol axis is wide enough for an unstable sort to reorder ties, so
    it carries the lock, in both directions and with ties at the top of a
    mixed-score row."""
    scores = _panel([1.0] * len(SYMBOLS))

    first = _select("long_only", 2, scores)
    second = _select("long_only", 2, scores)

    assert first[0, 0] == 0.5 and first[0, 1] == 0.5
    assert (first[0, 2:] == 0.0).all()
    assert np.array_equal(first, second)

    wide_symbols = [f"S{i:02d}" for i in range(20)]
    wide = _panel([1.0] * 20, symbols=wide_symbols)

    w = _select("long_only", 2, wide)[0]
    assert np.flatnonzero(w).tolist() == [0, 1]

    w = _select("long_short", 2, wide)[0]
    assert np.flatnonzero(w > 0).tolist() == [0, 1]
    assert np.flatnonzero(w < 0).tolist() == [18, 19]

    # Ties at the top of a mixed row: the five symbols scoring 2.0 sit at
    # axis positions 3, 7, 11, 15 and 19; the first two of them win.
    mixed = np.linspace(-1.0, 1.0, 20)
    mixed[[3, 7, 11, 15, 19]] = 2.0
    w = _select("long_only", 2, _panel(mixed, symbols=wide_symbols))[0]
    assert np.flatnonzero(w).tolist() == [3, 7]


def test_non_rebalance_rows_are_all_nan():
    """A 4-bar panel with mask [True, False, True, False]: rows 1 and 3 are
    all NaN ("hold"), rows 0 and 2 are fully finite (D-03). Goes red if the
    selector writes 0.0 off-rebalance, which would liquidate the book every
    bar, or leaves NaN on a rebalance row."""
    rng = np.random.default_rng(7)
    scores = _panel(rng.normal(size=(4, len(SYMBOLS))))

    w = _select("long_only", 2, scores, rebalance=[True, False, True, False])

    assert np.isnan(w[1]).all() and np.isnan(w[3]).all()
    assert np.isfinite(w[0]).all() and np.isfinite(w[2]).all()
