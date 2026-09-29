"""`average_predictions` z-scores each panel per bar and averages over panels.

What is locked here, and what turns it red:

- Members on very different scales contribute equally: the average is the
  mean of per-bar cross-sectional z-scores (`ddof=1`), not of raw values.
- A member whose bar is degenerate (constant, or fewer than two finite
  values) is left out on that bar; the other members still count.
- A cell only some members predict is the mean of those members; a cell no
  member predicts is NaN.
- Different variable sets raise `ValueError`.
- Coordinates are outer-joined over the panels.
- No `RuntimeWarning` on all-NaN cells or degenerate bars.

Everything is synthetic and hand-built.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.utils.ensemble import average_predictions

STAMPS = pd.date_range("2024-01-01", periods=3, freq="D")
SYMBOLS = ["A", "B", "C", "D"]


def _panel(values, *, stamps=STAMPS, symbols=SYMBOLS, name="ret") -> xr.Dataset:
    return xr.Dataset(
        {name: (("timestamp", "symbol"), np.asarray(values, dtype=np.float64))},
        coords={"timestamp": stamps, "symbol": symbols},
    )


def _zscore(row) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    finite = np.isfinite(row)
    out = np.full(row.shape, np.nan)
    out[finite] = (row[finite] - row[finite].mean()) / row[finite].std(ddof=1)
    return out


BASE = np.array(
    [
        [1.0, 2.0, 4.0, 8.0],
        [3.0, 1.0, 2.0, 0.0],
        [0.5, 0.25, 1.0, 2.0],
    ]
)
OTHER = np.array(
    [
        [2.0, 1.0, 3.0, 5.0],
        [1.0, 2.0, 0.0, 4.0],
        [1.0, 3.0, 2.0, 0.0],
    ]
)


def test_members_on_different_scales_contribute_equally():
    small = _panel(BASE)
    large = _panel(OTHER * 1e6 + 7.0)

    out = average_predictions([small, large])

    expected = np.vstack(
        [(_zscore(a) + _zscore(b)) / 2 for a, b in zip(BASE, OTHER)]
    )
    np.testing.assert_allclose(out["ret"].values, expected)
    # The same scaled member twice gives the plain z-score of that member.
    np.testing.assert_allclose(
        average_predictions([small, _panel(BASE * 1e9)])["ret"].values,
        np.vstack([_zscore(row) for row in BASE]),
    )


def test_a_degenerate_bar_of_one_member_is_ignored():
    degenerate = OTHER.copy()
    degenerate[0] = 5.0  # constant cross-section: zero std
    degenerate[1] = [np.nan, 3.0, np.nan, np.nan]  # one finite value

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = average_predictions([_panel(BASE), _panel(degenerate)])

    np.testing.assert_allclose(out["ret"].values[0], _zscore(BASE[0]))
    np.testing.assert_allclose(out["ret"].values[1], _zscore(BASE[1]))
    np.testing.assert_allclose(
        out["ret"].values[2], (_zscore(BASE[2]) + _zscore(OTHER[2])) / 2
    )


def test_partial_coverage_averages_the_members_present():
    partial = OTHER.copy()
    partial[:, 3] = np.nan  # this member never predicts D
    gap = BASE.copy()
    gap[2, 0] = np.nan  # and this one misses A on the last bar

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = average_predictions([_panel(gap), _panel(partial)])

    for t in range(3):
        za, zb = _zscore(gap[t]), _zscore(partial[t])
        expected = np.nanmean(np.vstack([za, zb]), axis=0)
        np.testing.assert_allclose(out["ret"].values[t], expected)
    # D comes from the first member alone.
    np.testing.assert_allclose(out["ret"].values[:, 3], [_zscore(r)[3] for r in gap])


def test_a_cell_no_member_predicts_is_nan():
    a = BASE.copy()
    b = OTHER.copy()
    a[1, 2] = np.nan
    b[1, 2] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = average_predictions([_panel(a), _panel(b)])

    assert np.isnan(out["ret"].values[1, 2])
    assert np.isfinite(np.delete(out["ret"].values[1], 2)).all()


def test_mismatched_variable_sets_raise():
    with pytest.raises(ValueError, match="variable"):
        average_predictions([_panel(BASE), _panel(OTHER, name="other")])
    two = _panel(BASE).assign(extra=_panel(OTHER)["ret"])
    with pytest.raises(ValueError, match="variable"):
        average_predictions([_panel(BASE), two])


def test_no_panel_raises():
    with pytest.raises(ValueError):
        average_predictions([])


def test_coordinates_are_outer_joined():
    first = _panel(BASE, symbols=["A", "B", "C", "D"])
    later = STAMPS + pd.Timedelta(days=1)
    second = _panel(OTHER, stamps=later, symbols=["B", "C", "D", "E"])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = average_predictions([first, second])

    assert list(out.symbol.values) == ["A", "B", "C", "D", "E"]
    np.testing.assert_array_equal(out.timestamp.values, STAMPS.append(later[-1:]).values)
    assert out["ret"].dims == ("timestamp", "symbol")
    # Day 1: only the first member has the bar.
    np.testing.assert_allclose(out["ret"].values[0, :4], _zscore(BASE[0]))
    assert np.isnan(out["ret"].values[0, 4])
    # Last day: only the second member.
    np.testing.assert_allclose(out["ret"].values[3, 1:], _zscore(OTHER[2]))
    # Day 2: A from the first member alone, E from the second alone.
    z1, z2 = _zscore(BASE[1]), _zscore(OTHER[0])
    np.testing.assert_allclose(out["ret"].values[1, 0], z1[0])
    np.testing.assert_allclose(out["ret"].values[1, 4], z2[3])
    np.testing.assert_allclose(out["ret"].values[1, 1:4], (z1[1:] + z2[:3]) / 2)


def test_every_variable_is_averaged_and_order_kept():
    a = _panel(BASE).assign(second=_panel(OTHER)["ret"])
    b = _panel(OTHER).assign(second=_panel(BASE)["ret"])

    out = average_predictions([a, b])

    assert list(out.data_vars) == ["ret", "second"]
    np.testing.assert_allclose(out["ret"].values, out["second"].values)
