"""`member_correlation` measures how much ensemble members agree, bar by bar.

What is locked here, and what turns it red:

- On each bar, over the symbols where every member is finite, the Pearson
  correlation of each pair of members is averaged over the pairs; the bar
  values are averaged over bars, ignoring NaN.
- Identical members give 1, an anti-correlated pair gives -1 and independent
  random members give about 0.
- A NaN cell drops that symbol from the bar for every member.
- A bar with fewer than two common symbols, or where every pair involves a
  constant member, is NaN and skipped by the mean.
- One member gives NaN (no pair); no member or mismatched shapes raise
  `ValueError`.
- No `RuntimeWarning` on skipped bars.

Everything is synthetic and hand-built.
"""

import warnings

import numpy as np
import pytest

from quantlab.utils.ensemble import member_correlation


def test_identical_members_correlate_perfectly():
    rng = np.random.default_rng(0)
    panel = rng.normal(size=(5, 20))

    mean, per_bar = member_correlation([panel, panel.copy(), panel * 3.0 + 1.0])

    assert mean == pytest.approx(1.0)
    np.testing.assert_allclose(per_bar, 1.0)
    assert per_bar.shape == (5,)


def test_an_anti_correlated_pair_gives_minus_one():
    rng = np.random.default_rng(1)
    panel = rng.normal(size=(4, 10))

    mean, per_bar = member_correlation([panel, -2.0 * panel])

    assert mean == pytest.approx(-1.0)
    np.testing.assert_allclose(per_bar, -1.0)


def test_independent_members_are_about_uncorrelated():
    rng = np.random.default_rng(2)
    panels = [rng.normal(size=(200, 500)) for _ in range(4)]

    mean, _ = member_correlation(panels)

    assert abs(mean) < 0.01


def test_the_bar_value_is_the_mean_over_pairs():
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    b = np.array([[1.0, 3.0, 2.0, 4.0]])
    c = np.array([[4.0, 3.0, 2.0, 1.0]])
    pairs = [
        np.corrcoef(x[0], y[0])[0, 1] for x, y in ((a, b), (a, c), (b, c))
    ]

    mean, per_bar = member_correlation([a, b, c])

    assert per_bar[0] == pytest.approx(np.mean(pairs))
    assert mean == pytest.approx(np.mean(pairs))


def test_a_nan_cell_drops_the_symbol_for_every_member():
    a = np.array([[1.0, 2.0, 3.0, 4.0, np.nan]])
    b = np.array([[2.0, 1.0, 4.0, 3.0, 100.0]])
    expected = np.corrcoef(a[0, :4], b[0, :4])[0, 1]

    mean, per_bar = member_correlation([a, b])

    assert per_bar[0] == pytest.approx(expected)
    assert mean == pytest.approx(expected)


def test_bars_with_fewer_than_two_common_symbols_are_skipped():
    a = np.array([[1.0, 2.0, 3.0], [1.0, np.nan, np.nan], [np.nan] * 3])
    b = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0], [np.nan] * 3])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean, per_bar = member_correlation([a, b])

    assert per_bar[0] == pytest.approx(1.0)
    assert np.isnan(per_bar[1]) and np.isnan(per_bar[2])
    assert mean == pytest.approx(1.0)


def test_a_constant_member_leaves_its_pairs_out():
    a = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    b = np.array([[5.0, 5.0, 5.0], [3.0, 2.0, 1.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean, per_bar = member_correlation([a, b])

    assert np.isnan(per_bar[0])
    assert per_bar[1] == pytest.approx(-1.0)
    assert mean == pytest.approx(-1.0)


def test_one_member_or_no_usable_bar_is_nan():
    panel = np.arange(6.0).reshape(2, 3)

    mean, per_bar = member_correlation([panel])
    assert np.isnan(mean) and np.isnan(per_bar).all() and per_bar.shape == (2,)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean, _ = member_correlation([np.full((2, 3), np.nan)] * 2)
    assert np.isnan(mean)


def test_values_stay_within_minus_one_and_one():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(50, 30))
    panels = [base + 1e-9 * rng.normal(size=base.shape) for _ in range(3)]

    _, per_bar = member_correlation(panels)

    assert np.all((per_bar >= -1.0) & (per_bar <= 1.0))


def test_invalid_input_raises():
    with pytest.raises(ValueError, match="at least one"):
        member_correlation([])
    with pytest.raises(ValueError, match="shape"):
        member_correlation([np.zeros((2, 3)), np.zeros((2, 4))])
    with pytest.raises(ValueError, match="2-D"):
        member_correlation([np.zeros(3), np.zeros(3)])
