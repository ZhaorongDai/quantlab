"""Average prediction panels after a per-bar cross-sectional z-score.

Several models, or one model under several seeds, predict the same target on
different scales. ``average_predictions`` puts every panel on one scale
before averaging: on each bar each panel is z-scored over its symbols with
finite values, ``(x - mean) / std`` with the sample standard deviation
(``ddof=1``, as ``CrossSectionalZScore`` does), and the result is the mean of
those z-scores over panels, ignoring NaN. The output is in z-score units, not
in the units of the target.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr

__all__ = ["average_predictions"]

_DIMS = ("timestamp", "symbol")


def _cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score ``[..., T, S]`` values over the last axis, NaN on a degenerate bar.

    A bar is degenerate when it holds fewer than two finite values or all its
    finite values are equal (zero standard deviation). Non-finite inputs stay
    NaN. No ``RuntimeWarning`` is raised on empty or degenerate bars.
    """
    finite = np.isfinite(values)
    count = finite.sum(axis=-1, keepdims=True)
    filled = np.where(finite, values, 0.0)
    mean = filled.sum(axis=-1, keepdims=True) / np.maximum(count, 1)
    deviation = np.where(finite, values - mean, 0.0)
    variance = (deviation**2).sum(axis=-1, keepdims=True) / np.maximum(count - 1, 1)
    std = np.sqrt(variance)
    # Constancy is tested on the values themselves: a constant bar can leave a
    # rounding residue of one ulp in the mean, and so a tiny non-zero std.
    high = np.where(finite, values, -np.inf).max(axis=-1, keepdims=True)
    low = np.where(finite, values, np.inf).min(axis=-1, keepdims=True)
    valid = (count >= 2) & (high > low)
    return np.where(
        finite & valid, deviation / np.where(valid, std, 1.0), np.nan
    )


def average_predictions(panels: Sequence[xr.Dataset]) -> xr.Dataset:
    """Return the mean of the panels' per-bar cross-sectional z-scores.

    Every panel is an ``xr.Dataset`` on ``(timestamp, symbol)`` with one
    variable per label, and all panels carry the same variables. The panels
    are outer-joined on both coordinates. Then, per panel, per variable and
    per bar, the finite values are z-scored over symbols with ``ddof=1``; a
    bar with fewer than two finite values or a constant cross-section makes
    that panel NaN on that bar. Each cell of the result is the mean of the
    z-scores the panels have there, with equal weights; a cell no panel has
    a z-score for is NaN.

    Parameters
    ----------
    panels : sequence of xr.Dataset
        The prediction panels, for example one per ensemble member.

    Returns
    -------
    xr.Dataset
        One float64 variable per label, in the first panel's variable order,
        on ``(timestamp, symbol)`` over the union of the coordinates.

    Raises
    ------
    ValueError
        If ``panels`` is empty or two panels carry different variable sets.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr
    >>> def panel(values):
    ...     return xr.Dataset(
    ...         {"ret": (("timestamp", "symbol"), np.array(values))},
    ...         coords={"timestamp": pd.date_range("2024-01-01", periods=2),
    ...                 "symbol": ["A", "B", "C"]},
    ...     )
    >>> small = panel([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    >>> large = panel([[300.0, 100.0, 200.0], [10.0, 30.0, 20.0]])
    >>> average_predictions([small, large])["ret"].values
    array([[ 0. , -0.5,  0.5],
           [-1. ,  1. ,  0. ]])
    """
    panels = list(panels)
    if not panels:
        raise ValueError("average_predictions needs at least one panel")
    names = list(panels[0].data_vars)
    for i, panel in enumerate(panels[1:], start=1):
        if set(panel.data_vars) != set(names):
            raise ValueError(
                f"average_predictions: panel {i} has variables "
                f"{sorted(map(str, panel.data_vars))}, panel 0 has "
                f"{sorted(map(str, names))}; every panel must carry the same "
                f"variable set"
            )

    aligned = xr.align(*(panel[names] for panel in panels), join="outer")
    first = aligned[0]
    averaged = {}
    for name in names:
        stacked = np.stack(
            [
                np.asarray(panel[name].transpose(*_DIMS).values, dtype=np.float64)
                for panel in aligned
            ]
        )
        scores = _cross_sectional_zscore(stacked)
        present = np.isfinite(scores)
        count = present.sum(axis=0)
        total = np.where(present, scores, 0.0).sum(axis=0)
        averaged[name] = (
            _DIMS,
            np.where(count > 0, total / np.maximum(count, 1), np.nan),
        )
    return xr.Dataset(
        averaged,
        coords={dim: first[dim].values for dim in _DIMS},
    )
