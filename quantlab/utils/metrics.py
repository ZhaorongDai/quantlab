"""Regression metrics for prediction panels: MSE, RMSE, MAE, R2, IC and RankIC,
and the level metrics of a volatility prediction.

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

The per-timestamp values behind both means are available as series
(``cross_sectional_ic_series``, ``cross_sectional_rank_ic_series``), NaN on a
skipped row. The ICIR (information ratio of the IC) divides the mean of the
valid values of such a series by their sample standard deviation
(``information_ratio``), so it measures how stable a signal is, not only how
strong.

The IC of a volatility prediction measures only how well it ranks the
symbols' volatility. A consumer that uses the predicted level, such as a
covariance built from it, also needs the level right, which
``volatility_level_metrics`` scores: the QLIKE loss and the ratio of realised
to predicted variance.
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


def cross_sectional_ic_series(pred, target) -> np.ndarray:
    """Return the per-row Pearson correlation of a panel, NaN on a skipped row.

    A row (timestamp) is skipped when it has fewer than two usable symbols or
    when either the prediction or the target is constant over its usable
    symbols. Constancy is tested exactly (masked max equals masked min) rather
    than with a variance threshold, so genuinely small cross-sectional spreads
    are not misread as constant.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.

    Returns
    -------
    np.ndarray
        A float64 array of length ``T``.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.

    Examples
    --------
    >>> pred = np.array([[1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [1.0, 2.0, 3.0]])
    >>> target = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 4.0], [1.0, 3.0, 2.0]])
    >>> cross_sectional_ic_series(pred, target)
    array([1. , 1. , 0.5])
    """
    p, t, mask = _joint(pred, target)
    if p.ndim != 2:
        raise ValueError(
            f"cross_sectional_ic expects a 2-D [T, S] panel, got shape {p.shape}"
        )

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

        series = np.full(p.shape[0], np.nan)
        series[valid] = cov[valid] / np.sqrt(var_p[valid] * var_t[valid])
        return series


def cross_sectional_ic(pred, target) -> float:
    """Return the mean over time of the per-row Pearson correlation.

    The mean runs over the rows ``cross_sectional_ic_series`` does not skip
    (see there for when a row is skipped). NaN is returned when every row is
    skipped.

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
    return _finite_mean(cross_sectional_ic_series(pred, target))


def _finite_mean(series: np.ndarray) -> float:
    """Return the mean of the finite values of ``series``, NaN when there are none."""
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return float("nan")
    return float(finite.sum() / finite.size)


def information_ratio(series) -> float:
    """Return the mean of the finite values of ``series`` over their sample std.

    This is the ICIR when ``series`` is a per-bar IC series. NaN values (the
    skipped bars of an IC series) are left out rather than counted as 0. The
    standard deviation uses ``ddof=1``, so fewer than two finite values give
    NaN, and so do finite values that are all equal (tested exactly, as for
    the IC's constancy test), since the ratio is then undefined.

    Parameters
    ----------
    series : array_like
        A 1-D sequence of per-bar values.

    Examples
    --------
    >>> information_ratio([0.1, np.nan, 0.3])
    1.4142135623730951
    >>> information_ratio([0.1, np.nan])
    nan
    """
    values = np.asarray(series, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size < 2 or finite.max() == finite.min():
        return float("nan")
    return float(finite.mean() / finite.std(ddof=1))


def cross_sectional_rank_ic_series(pred, target) -> np.ndarray:
    """Return the per-row IC computed on per-row ranks, NaN on a skipped row.

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

    Returns
    -------
    np.ndarray
        A float64 array of length ``T``.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.

    Examples
    --------
    >>> cross_sectional_rank_ic_series([[1.0, 10.0, 100.0]], [[1.0, 3.0, 2.0]])
    array([0.5])
    """
    p, t, mask = _joint(pred, target)
    if p.ndim != 2:
        raise ValueError(
            f"cross_sectional_rank_ic expects a 2-D [T, S] panel, got shape {p.shape}"
        )
    p_ranks = rankdata(np.where(mask, p, np.nan), axis=1, nan_policy="omit")
    t_ranks = rankdata(np.where(mask, t, np.nan), axis=1, nan_policy="omit")
    return cross_sectional_ic_series(p_ranks, t_ranks)


def cross_sectional_rank_ic(pred, target) -> float:
    """Return the cross-sectional IC computed on per-row ranks.

    The mean of ``cross_sectional_rank_ic_series`` over the rows it does not
    skip, or NaN when every row is skipped.

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
    return _finite_mean(cross_sectional_rank_ic_series(pred, target))


def regression_panel_metrics(
    pred, target, *, return_series: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return the eight panel metrics keyed ``mse, rmse, mae, r2, ic, rank_ic, icir, rank_icir``.

    ``ic`` and ``icir`` are the mean and the ``information_ratio`` of one
    ``cross_sectional_ic_series``, and ``rank_ic`` and ``rank_icir`` of one
    ``cross_sectional_rank_ic_series``, so each series is computed once.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.
    return_series : bool, default False
        Also return the two per-row series the IC metrics were computed
        from, as ``{"ic": ..., "rank_ic": ...}`` (length ``T``, NaN on a
        skipped row).

    Returns
    -------
    dict or tuple
        The metrics, or ``(metrics, series)`` when ``return_series`` is True.

    Examples
    --------
    >>> metrics = regression_panel_metrics(pred, target)
    >>> sorted(metrics)
    ['ic', 'icir', 'mae', 'mse', 'r2', 'rank_ic', 'rank_icir', 'rmse']
    >>> metrics["rank_ic"]
    1.0
    >>> metrics, series = regression_panel_metrics(pred, target, return_series=True)
    >>> series["rank_ic"]
    array([1., 1.])
    """
    ic_metrics, series = ic_panel_metrics(pred, target, return_series=True)
    metrics = {
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "r2": r2(pred, target),
        **ic_metrics,
    }
    if return_series:
        return metrics, series
    return metrics


def ic_panel_metrics(
    pred, target, *, return_series: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return the four IC metrics keyed ``ic, rank_ic, icir, rank_icir``.

    The IC half of ``regression_panel_metrics``, which calls it: the mean and
    the ``information_ratio`` of one ``cross_sectional_ic_series`` and of one
    ``cross_sectional_rank_ic_series``. It suits predictions whose scale
    means nothing, such as an ensemble's average of z-scores, where an error
    metric would compare units that differ from the target's.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.
    return_series : bool, default False
        Also return the two per-row series, as ``{"ic": ..., "rank_ic":
        ...}`` (length ``T``, NaN on a skipped row).

    Returns
    -------
    dict or tuple
        The metrics, or ``(metrics, series)`` when ``return_series`` is True.

    Examples
    --------
    On the panel from ``cross_sectional_ic``, whose two rows have the same
    IC, so the ICIR is undefined:

    >>> ic_panel_metrics(pred, target)
    {'ic': 0.989743318610787, 'rank_ic': 1.0, 'icir': nan, 'rank_icir': nan}
    """
    ic = cross_sectional_ic_series(pred, target)
    rank_ic = cross_sectional_rank_ic_series(pred, target)
    metrics = {
        "ic": _finite_mean(ic),
        "rank_ic": _finite_mean(rank_ic),
        "icir": information_ratio(ic),
        "rank_icir": information_ratio(rank_ic),
    }
    if return_series:
        return metrics, {"ic": ic, "rank_ic": rank_ic}
    return metrics


def volatility_level_metrics(pred, target) -> dict[str, float]:
    """Return the level metrics of a volatility prediction, ``qlike`` and ``variance_ratio``.

    Both compare predicted and realised variance on the cells where the
    prediction and the target are finite and positive (a non-positive
    volatility has no variance ratio). ``qlike`` is the mean over those
    cells of ``q - log(q) - 1`` with ``q = target**2 / pred**2``: zero for a
    perfect prediction, and it penalises an under-prediction of variance
    more than an over-prediction of the same size; a prediction close to
    zero where the target is not makes it very large, and infinite (written
    as null in ``metrics.json``) once ``q`` overflows, which flags a model
    predicting next to no risk. ``variance_ratio`` is ``mean(target**2) /
    mean(pred**2)``, pooled over cells, so the most volatile symbols weigh
    most: 1 when the predicted variance is unbiased, above 1 when risk is
    under-predicted.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predicted volatilities.
    target : array_like
        A 2-D ``[T, S]`` panel of realised volatilities, on the same scale.

    Returns
    -------
    dict[str, float]
        ``{"qlike": ..., "variance_ratio": ...}``, NaN without a usable cell.

    Examples
    --------
    A prediction of half the realised volatility on every cell, so a
    quarter of its variance:

    >>> volatility_level_metrics([[0.1, 0.2]], [[0.2, 0.4]])
    {'qlike': 1.6137056388801092, 'variance_ratio': 4.0}
    """
    p, t, mask = _joint(pred, target)
    with np.errstate(invalid="ignore"):
        mask &= (p > 0) & (t > 0)
    if not mask.any():
        return {"qlike": float("nan"), "variance_ratio": float("nan")}
    predicted, realised = p[mask] ** 2, t[mask] ** 2
    q = realised / predicted
    return {
        "qlike": float(np.mean(q - np.log(q) - 1.0)),
        "variance_ratio": float(realised.mean() / predicted.mean()),
    }


def scores_volatility_level(label, scale: str | None) -> bool:
    """Return whether a label's predictions get ``volatility_level_metrics``.

    They do when the label's ``kind`` is ``"volatility"`` and the prediction
    is on the label's own scale (``scale`` is ``"raw"``, as a predictor's
    ``label_scales`` reports it): a level metric on a standardized
    prediction compares units that differ from the label's. An object
    without ``kind`` counts as a return label.

    Parameters
    ----------
    label : object
        The label object, such as a ``Forward``.
    scale : str or None
        The prediction's scale for that label, ``"raw"`` or
        ``"standardized"``.

    Returns
    -------
    bool

    Examples
    --------
    >>> from quantlab.label.predefined.fret import Return, Volatility
    >>> scores_volatility_level(Volatility, "raw"), scores_volatility_level(Volatility, "standardized")
    (True, False)
    >>> scores_volatility_level(Return, "raw")
    False
    """
    return getattr(label, "kind", "return") == "volatility" and scale == "raw"
