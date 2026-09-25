"""Regression metrics for prediction panels: MSE, RMSE, MAE, R2, IC and RankIC.

The return models score their predictions with these functions. Predictions
and targets are ``[T, S]`` panels (time by symbol), and every function follows
the same conventions: inputs are cast to float64 and must share a shape; only
cells where both prediction and target are finite take part in any sum or
ranking; and an empty set of usable cells yields NaN without raising a
``RuntimeWarning``.

The IC (information coefficient) measures how well predictions order the
symbols. Cross-sectional IC is the Pearson correlation between prediction and
target across the symbols of one timestamp, averaged over time. RankIC ranks
each row first (average ranks on ties) and then computes IC on the ranks, so
it is a per-timestamp Spearman correlation and is not dominated by
outliers. Both are fully vectorised, with no per-row Python loop, since a
panel can hold tens of thousands of timestamps.
"""

import numpy as np
from scipy.stats import rankdata


def _joint(pred, target) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast both inputs to float64, check shapes, and return them with the joint mask.

    The joint mask marks cells where both inputs are finite.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(pred, target, mask)`` where ``mask`` is True where both are finite.

    Raises
    ------
    ValueError
        If the two inputs differ in shape.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {p.shape} vs {t.shape}"
        )
    return p, t, np.isfinite(p) & np.isfinite(t)


def mse(pred, target) -> float:
    """Return the mean squared error over the jointly finite cells, or NaN if none.

    Parameters
    ----------
    pred : array_like
        Predictions, any shape.
    target : array_like
        Realised values, the same shape as ``pred``.

    Examples
    --------
    >>> mse([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    1.0
    >>> mse([1.0, np.nan], [np.nan, 2.0])
    nan
    """
    p, t, mask = _joint(pred, target)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    diff = p[mask] - t[mask]
    return float(np.dot(diff, diff) / n)


def rmse(pred, target) -> float:
    """Return ``sqrt(mse(pred, target))``, or NaN when the MSE is undefined.

    Parameters
    ----------
    pred : array_like
        Predictions, any shape.
    target : array_like
        Realised values, the same shape as ``pred``.

    Examples
    --------
    >>> rmse([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    1.0
    """
    value = mse(pred, target)
    return float(np.sqrt(value)) if np.isfinite(value) else float("nan")


def mae(pred, target) -> float:
    """Return the mean absolute error over the jointly finite cells, or NaN if none.

    Parameters
    ----------
    pred : array_like
        Predictions, any shape.
    target : array_like
        Realised values, the same shape as ``pred``.

    Examples
    --------
    >>> mae([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    0.5
    """
    p, t, mask = _joint(pred, target)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.abs(p[mask] - t[mask]).sum() / n)


def r2(pred, target) -> float:
    """Return the coefficient of determination ``1 - SS_res / SS_tot``.

    NaN is returned when fewer than two cells are usable or when the target is
    constant over the usable cells (``SS_tot == 0``), because R2 is undefined
    in both cases.

    Parameters
    ----------
    pred : array_like
        Predictions, any shape.
    target : array_like
        Realised values, the same shape as ``pred``.

    Examples
    --------
    >>> r2([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    0.7142857142857143
    >>> r2([1.0, 2.0], [3.0, 3.0])
    nan
    """
    p, t, mask = _joint(pred, target)
    n = int(mask.sum())
    if n < 2:
        return float("nan")
    tv = t[mask]
    ss_tot = float(np.sum((tv - tv.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((tv - p[mask]) ** 2))
    return 1.0 - ss_res / ss_tot


def cross_sectional_ic(pred, target) -> float:
    """Return the mean over time of the per-row Pearson correlation.

    A row (timestamp) is skipped when it has fewer than two usable symbols or
    when either the prediction or the target is constant over its usable
    symbols. Constancy is tested exactly (masked max equals masked min) rather
    than with a variance threshold, so genuinely small cross-sectional spreads
    are not misread as constant. NaN is returned when every row is skipped.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.

    Examples
    --------
    >>> pred = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    >>> target = np.array([[9.0, 1.0, 4.0], [1.0, 9.0, 4.0]])
    >>> cross_sectional_ic(pred, target)
    0.989743318610787
    """
    p, t, mask = _joint(pred, target)
    if p.ndim != 2:
        raise ValueError(f"cross_sectional_ic expects a 2-D [T, S] panel, got shape {p.shape}")

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        n = mask.sum(axis=1)
        safe_n = np.maximum(n, 1)
        p_mean = np.where(mask, p, 0.0).sum(axis=1) / safe_n
        t_mean = np.where(mask, t, 0.0).sum(axis=1) / safe_n
        p_dev = np.where(mask, p - p_mean[:, None], 0.0)
        t_dev = np.where(mask, t - t_mean[:, None], 0.0)
        cov = (p_dev * t_dev).sum(axis=1)
        var_p = (p_dev * p_dev).sum(axis=1)
        var_t = (t_dev * t_dev).sum(axis=1)

        p_varies = np.max(np.where(mask, p, -np.inf), axis=1, initial=-np.inf) > np.min(
            np.where(mask, p, np.inf), axis=1, initial=np.inf
        )
        t_varies = np.max(np.where(mask, t, -np.inf), axis=1, initial=-np.inf) > np.min(
            np.where(mask, t, np.inf), axis=1, initial=np.inf
        )
        valid = (n >= 2) & p_varies & t_varies & (var_p > 0) & (var_t > 0)

        n_valid = int(valid.sum())
        if n_valid == 0:
            return float("nan")
        per_row = cov[valid] / np.sqrt(var_p[valid] * var_t[valid])
        return float(per_row.sum() / n_valid)


def cross_sectional_rank_ic(pred, target) -> float:
    """Return the cross-sectional IC computed on per-row ranks.

    Cells outside the joint mask are set to NaN on both panels before ranking.
    The order matters: a symbol missing only on the target side would
    otherwise still occupy a rank on the prediction side and shift every other
    rank in that row. Ties receive their average rank.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.

    Examples
    --------
    The panel from ``cross_sectional_ic`` orders every row the same way
    on both sides, so its rank correlation is exactly 1:

    >>> cross_sectional_rank_ic(pred, target)
    1.0
    """
    p, t, mask = _joint(pred, target)
    if p.ndim != 2:
        raise ValueError(
            f"cross_sectional_rank_ic expects a 2-D [T, S] panel, got shape {p.shape}"
        )
    p_ranks = rankdata(np.where(mask, p, np.nan), axis=1, nan_policy="omit")
    t_ranks = rankdata(np.where(mask, t, np.nan), axis=1, nan_policy="omit")
    return cross_sectional_ic(p_ranks, t_ranks)


def regression_panel_metrics(pred, target) -> dict[str, float]:
    """Return all six panel metrics keyed ``mse, rmse, mae, r2, ic, rank_ic``.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.

    Examples
    --------
    >>> metrics = regression_panel_metrics(pred, target)
    >>> sorted(metrics)
    ['ic', 'mae', 'mse', 'r2', 'rank_ic', 'rmse']
    >>> metrics["rank_ic"]
    1.0
    """
    return {
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "r2": r2(pred, target),
        "ic": cross_sectional_ic(pred, target),
        "rank_ic": cross_sectional_rank_ic(pred, target),
    }
