"""Cross-sectional transforms of ``[..., T, S]`` arrays: one bar at a time, over its symbols.

``cross_sectional_zscore`` standardises each bar over its finite symbols with the
sample standard deviation (``ddof=1``, as the KunQuant ``CrossSectionalZScore``
operator does). The ensemble average and the mean-variance optimiser's score
standardisation both use it.
"""

import numpy as np


def cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score ``[..., T, S]`` values over the last axis, NaN on a degenerate bar.

    A bar is degenerate when it holds fewer than two finite values or all its
    finite values are equal (zero standard deviation). Non-finite inputs stay
    NaN. No ``RuntimeWarning`` is raised on empty or degenerate bars.

    Examples
    --------
    >>> cross_sectional_zscore(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, np.nan]]))
    array([[-1.,  0.,  1.],
           [nan, nan, nan]])
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
