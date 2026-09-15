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


def test_fill_prices_pair_with_scores_by_label_not_position():
    """Code review WR-09: `select` pairs each score with ITS symbol's fill price.

    BBB has the top score but no fill price. The fill panel has the same
    shape as the scores and its symbols in reverse order. Pairing by position
    would put the NaN on EEE, select BBB and target a symbol that cannot be
    filled. The old `select` compared shapes only and went red here. The
    fixed selector aligns by coordinate label, so BBB is ineligible and DDD and
    FFF are chosen, exactly as with an identically ordered fill panel.
    """
    scores = _panel(DISTINCT)
    fill = _finite_fill(scores)
    fill.loc[dict(symbol="BBB")] = NAN
    reversed_fill = fill.isel(symbol=slice(None, None, -1))
    assert reversed_fill.shape == scores.shape
    assert reversed_fill.symbol.values.tolist() != scores.symbol.values.tolist()

    w = _select("long_only", 2, scores, reversed_fill)[0]

    assert w[1] == 0.0
    assert w[3] == 0.5 and w[5] == 0.5
    np.testing.assert_array_equal(w, _select("long_only", 2, scores, fill)[0])


def test_fill_prices_on_other_labels_are_refused():
    """Code review WR-09: a fill panel on other symbols or bars is a caller error.

    Same shape, different labels: one symbol swapped for `ZZZ`, or every
    timestamp moved by a day (a mis-applied shift). The old shape-only check
    accepted both and paired values by position, so both cases go red. The
    error must name the differing axis or label.
    """
    scores = _panel(DISTINCT)
    other_symbols = _panel([100.0] * len(SYMBOLS), symbols=SYMBOLS[:-1] + ["ZZZ"])
    with pytest.raises(ValueError, match="ZZZ"):
        _select("long_only", 2, scores, other_symbols)

    shifted = _finite_fill(scores).assign_coords(
        timestamp=scores.timestamp.values + np.timedelta64(1, "D")
    )
    with pytest.raises(ValueError, match="timestamp"):
        _select("long_only", 2, scores, shifted)


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


# --------------------------------------------------------------------------
# Task 2: schedule (D-18), score label (D-11), parameters (D-10), and the
# D-03 invariant on randomized panels
# --------------------------------------------------------------------------


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


def test_score_label_defaults_to_first_label():
    """None resolves to the model's FIRST declared label, in declared order
    rather than alphabetical order: "ret_5" before "ret_1" (D-11). An explicit
    known label is returned unchanged."""
    assert resolve_score_label(None, ["ret_5", "ret_1"]) == "ret_5"
    assert resolve_score_label("ret_1", ["ret_5", "ret_1"]) == "ret_1"


def test_unknown_score_label_raises_listing_known_labels():
    """An unknown label raises a ValueError that names it and lists every
    known label, so a typo is fixable from the message alone (D-11). A model
    with no labels at all also raises instead of indexing into an empty
    list."""
    with pytest.raises(ValueError) as excinfo:
        resolve_score_label("ret_20", ["ret_5", "ret_1"])
    message = str(excinfo.value)
    assert "ret_20" in message
    assert "ret_5" in message and "ret_1" in message

    with pytest.raises(ValueError):
        resolve_score_label(None, [])


def test_selector_rejects_bad_parameters():
    """top_n=0 and an unknown direction raise at construction, before any
    panel is touched. top_n=0 would otherwise warn on every bar and liquidate
    the book, and "short_only" would silently fall into the long_short
    branch."""
    with pytest.raises(ValueError, match="top_n"):
        CrossSectionTopNSelector(direction="long_only", top_n=0)
    with pytest.raises(ValueError, match="direction"):
        CrossSectionTopNSelector(direction="short_only", top_n=2)


def test_cross_section_config_has_no_quantile_mode():
    """D-10 fixes the pick count as `top_n`; there is no quantile mode. Goes
    red if a quantile field is added to the config or `top_n` is removed."""
    names = [f.name for f in fields(CrossSectionBacktestConfig)]

    assert [n for n in names if "quantile" in n.lower()] == []
    assert "top_n" in names


def test_weights_contract_holds_on_random_panels():
    """50 seeded panels (T=12, S=8) with random NaN holes in scores and next
    fill prices, a random period in 1..4, a random top_n in 1..5, and the two
    directions alternating by seed. On every panel:

    - every non-rebalance row is all NaN;
    - every rebalance row is fully finite, with sum(|w|) <= 1 + 1e-12;
    - a long_short rebalance row with any nonzero weight nets to 0 within
      1e-12.

    Each failure message carries the seed, direction, top_n and period, so the
    breaking panel can be rebuilt. The trailing non-vacuity checks make sure
    the loop really exercised short books and nonzero long_short rows."""
    n_bars, symbols = 12, [f"S{i}" for i in range(8)]
    long_short_nonzero_rows = 0
    short_book_rows = 0

    for seed in range(50):
        rng = np.random.default_rng(seed)
        direction = ("long_only", "long_short")[seed % 2]
        top_n = int(rng.integers(1, 6))
        period = int(rng.integers(1, 5))
        context = f"seed={seed} direction={direction} top_n={top_n} p={period}"

        scores = rng.normal(size=(n_bars, len(symbols)))
        scores[rng.random(scores.shape) < 0.3] = NAN
        fill = rng.uniform(10.0, 100.0, size=scores.shape)
        fill[rng.random(fill.shape) < 0.2] = NAN
        mask = rebalance_mask(n_bars, period)

        weights = _select(
            direction,
            top_n,
            _panel(scores, symbols=symbols),
            _panel(fill, symbols=symbols),
            mask,
        )

        for t in range(n_bars):
            row = weights[t]
            where = f"{context} t={t}"
            if not mask[t]:
                assert np.isnan(row).all(), where
                continue
            assert np.isfinite(row).all(), where
            assert np.abs(row).sum() <= 1.0 + 1e-12, where
            eligible = int((np.isfinite(scores[t]) & np.isfinite(fill[t])).sum())
            if direction == "long_short":
                if eligible // 2 < top_n:
                    short_book_rows += 1
                if (row != 0.0).any():
                    long_short_nonzero_rows += 1
                    assert abs(row.sum()) <= 1e-12, where
            elif eligible < top_n:
                short_book_rows += 1

    assert long_short_nonzero_rows > 0
    assert short_book_rows > 0
