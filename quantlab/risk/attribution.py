"""Factor attribution: a book's per-bar return split over a factor risk model (ADR 0026).

Given the holdings at the start of each bar, as signed fractions of the NAV,
``factor_attribution`` splits each bar's NAV return into five terms that sum to
it exactly:

- ``factor``: per factor k, the holdings at the close of t-1 times the
  exposures of t-1 times the regression store's factor return of row t
  (``factor_contribution``); a NaN factor return (a thin industry) counts as 0;
- ``specific``: the covered holdings times their specific returns of row t;
- ``uncovered``: the held symbols without a full set of exposures at t-1,
  without a specific return at t or without a risk-free rate at t-1 (the
  three pieces a covered symbol's return is split into), times their own
  return of bar t;
- ``risk_free``: the covered holdings times the risk-free rate of t-1, read
  from the risk model's price inputs (the regression is on excess returns);
- ``trading``: the NAV return less the four terms above: the fills at t's
  open, fees, slippage and idle cash.

The terms are linked over time as log contributions, each bar's terms times
``ln(1+r)/r``, so the cumulative sums over bars and terms are log NAV at every
bar, and no bar's value depends on the bars after it. ``attribution_summary``
gives each term's and each factor's annualized log growth and the coverage of
a run of bars.

The module reads a ``FactorRiskModel``'s stores and inputs only; it builds
nothing (ADR 0024) and imports nothing of the portfolio, backtest or runs
layers.

Examples
--------
With ``holdings``, ``nav_returns`` and ``symbol_returns`` a book's panels and
``model`` a ``FactorRiskModel`` whose regression store covers them:

>>> attribution = factor_attribution(holdings, nav_returns, symbol_returns, model)
>>> bool(np.allclose(attribution["contribution"].sum("term"), nav_returns))
True
>>> sorted(attribution_summary(attribution, years=1.0)["annualized_log_return"])
['factor', 'risk_free', 'specific', 'total', 'trading', 'uncovered']
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.risk.base import FactorRiskModel

#: The terms a bar's NAV return is split into, in the order of the ``term`` axis.
TERMS = ("factor", "specific", "uncovered", "risk_free", "trading")

_FACTOR, _SPECIFIC, _UNCOVERED, _RISK_FREE, _TRADING = range(len(TERMS))

#: Below this mean covered weight, ``attribution_summary`` adds a coverage note.
LOW_COVERAGE = 0.9


def factor_attribution(
    holdings: xr.DataArray,
    nav_returns: xr.DataArray,
    symbol_returns: xr.DataArray,
    risk_model: FactorRiskModel,
) -> xr.Dataset:
    """Split each bar's NAV return over ``risk_model``'s factors (see the module docstring).

    Parameters
    ----------
    holdings : xr.DataArray
        On ``(timestamp, symbol)``: row t holds what the book held at the
        close of the bar before t, as signed fractions of that bar's NAV. The
        first row has no bar before it in the window, so its holdings are not
        attributed (its return is all ``trading``).
    nav_returns : xr.DataArray
        The book's return of each bar, on ``timestamp``.
    symbol_returns : xr.DataArray
        Each symbol's own return of each bar, on ``(timestamp, symbol)``, for
        the uncovered term; NaN counts as 0.
    risk_model : FactorRiskModel
        The model whose regression store, exposures and risk-free rate are
        read over the window; its stores must cover it.

    Returns
    -------
    xr.Dataset
        On the holdings' ``timestamp``: ``contribution`` on ``(timestamp,
        term)`` (``TERMS``) and ``log_contribution`` (each bar's terms times
        ``ln(1+r)/r``), ``factor_contribution`` and
        ``factor_log_contribution`` on ``(timestamp, factor)``, ``return``
        (the NAV return), ``covered_weight`` (the covered share of the gross
        held weight, NaN on a bar holding nothing) and ``gross_weight``.

    Raises
    ------
    ValueError
        If the regression store does not cover the window (build or extend
        it first), or symbols were held but none was ever covered (the
        book's and the risk model's symbol axes may differ).

    Examples
    --------
    >>> attribution = factor_attribution(holdings, nav_returns, symbol_returns, model)
    >>> attribution["factor_contribution"].dims
    ('timestamp', 'factor')
    """
    holdings = holdings.transpose("timestamp", "symbol")
    timestamps = holdings["timestamp"].values
    symbols = [str(s) for s in holdings["symbol"].values]
    factors = list(risk_model.factor_names)
    first, last = pd.Timestamp(timestamps[0]), pd.Timestamp(timestamps[-1])
    regression = risk_model.regression.read(first, last)

    weight = np.nan_to_num(np.asarray(holdings.values, dtype=np.float64))
    returns = np.nan_to_num(np.asarray(nav_returns.sel(timestamp=timestamps).values, dtype=np.float64))
    own = np.nan_to_num(_on_axes(symbol_returns, timestamps, symbols))
    factor_return = np.nan_to_num(
        regression["factor_return"]
        .reindex(timestamp=timestamps)
        .transpose("timestamp", "factor")
        .values.astype(np.float64)
    )
    specific = _on_axes(regression["specific_return"], timestamps, symbols)
    risk_free = _on_axes(
        risk_model.prices(first, last)[risk_model.config.risk_free_column], timestamps, symbols
    )
    exposures, has_exposures = _exposures(risk_model, timestamps, symbols)

    n_bars = timestamps.size
    contribution = np.zeros((n_bars, len(TERMS)))
    factor_part = np.zeros((n_bars, len(factors)))
    covered_weight = np.full(n_bars, np.nan)
    held = weight != 0.0
    for t in range(1, n_bars):
        covered = (
            held[t] & has_exposures[t - 1] & np.isfinite(specific[t]) & np.isfinite(risk_free[t - 1])
        )
        uncovered = held[t] & ~covered
        w = weight[t]
        factor_part[t] = (w[covered] @ exposures[t - 1][covered]) * factor_return[t]
        contribution[t, _SPECIFIC] = w[covered] @ specific[t][covered]
        contribution[t, _UNCOVERED] = w[uncovered] @ own[t][uncovered]
        contribution[t, _RISK_FREE] = w[covered] @ risk_free[t - 1][covered]
        gross = np.abs(w[held[t]]).sum()
        if gross > 0:
            covered_weight[t] = np.abs(w[covered]).sum() / gross
    contribution[:, _FACTOR] = factor_part.sum(axis=1)
    contribution[:, _TRADING] = returns - np.delete(contribution, _TRADING, axis=1).sum(axis=1)

    if held[1:].any() and not np.nan_to_num(covered_weight).any():
        raise ValueError(
            f"factor attribution: no held symbol is covered by {risk_model.class_name} on any "
            f"bar from {first} to {last} (exposures at the bar before and a specific return at "
            f"the bar). Check that the book and the risk model use the same symbol axis "
            f"(for example PERMNO against permaticker)."
        )

    link = _link(returns)[:, None]
    gross = np.abs(weight).sum(axis=1)
    gross[0] = 0.0
    return xr.Dataset(
        {
            "contribution": (("timestamp", "term"), contribution),
            "log_contribution": (("timestamp", "term"), contribution * link),
            "factor_contribution": (("timestamp", "factor"), factor_part),
            "factor_log_contribution": (("timestamp", "factor"), factor_part * link),
            "return": ("timestamp", returns),
            "covered_weight": ("timestamp", covered_weight),
            "gross_weight": ("timestamp", gross),
        },
        coords={"timestamp": timestamps, "term": list(TERMS), "factor": factors},
    )


def attribution_summary(attribution: xr.Dataset, years: float) -> dict:
    """Return the annualized log growth of each term and factor, and the coverage.

    Parameters
    ----------
    attribution : xr.Dataset
        ``factor_attribution``'s dataset, or a run of its bars.
    years : float
        The years the bars span, to annualize by.

    Returns
    -------
    dict
        ``annualized_log_return``: each term's, and ``total``, their sum
        (the NAV's annualized log growth); ``factor_annualized_log_return``:
        each factor's; ``coverage``: ``mean_covered_weight`` and
        ``min_covered_weight`` over the bars holding something, and a
        ``note`` when the mean is below ``LOW_COVERAGE`` (else None).

    Examples
    --------
    >>> summary = attribution_summary(attribution, years=1.0)
    >>> sorted(summary)
    ['annualized_log_return', 'coverage', 'factor_annualized_log_return']
    """
    terms = attribution["log_contribution"].sum("timestamp") / years
    factors = attribution["factor_log_contribution"].sum("timestamp") / years
    growth = {str(term): float(terms.sel(term=term)) for term in attribution["term"].values}
    growth["total"] = float(terms.sum())
    covered = attribution["covered_weight"].values
    covered = covered[np.isfinite(covered)]
    mean = float(covered.mean()) if covered.size else None
    note = None
    if mean is not None and mean < LOW_COVERAGE:
        note = (
            f"The risk model covers on average {mean:.1%} of the gross held weight "
            f"(minimum {covered.min():.1%}); the rest is in the uncovered term."
        )
    return {
        "annualized_log_return": growth,
        "factor_annualized_log_return": {
            str(name): float(factors.sel(factor=name)) for name in attribution["factor"].values
        },
        "coverage": {
            "mean_covered_weight": mean,
            "min_covered_weight": float(covered.min()) if covered.size else None,
            "note": note,
        },
    }


def _link(returns: np.ndarray) -> np.ndarray:
    """Return ``ln(1+r)/r`` per bar, 1 where ``r`` is 0."""
    returns = np.asarray(returns, dtype=np.float64)
    nonzero = returns != 0.0
    link = np.ones_like(returns)
    link[nonzero] = np.log1p(returns[nonzero]) / returns[nonzero]
    return link


def _on_axes(panel: xr.DataArray, timestamps: np.ndarray, symbols: list[str]) -> np.ndarray:
    """Return ``panel`` on ``(timestamps, symbols)``, symbols matched as text, NaN where absent."""
    panel = panel.assign_coords(symbol=[str(s) for s in panel["symbol"].values])
    return np.asarray(
        panel.reindex(timestamp=timestamps, symbol=symbols).transpose("timestamp", "symbol").values,
        dtype=np.float64,
    )


def _exposures(
    risk_model: FactorRiskModel, timestamps: np.ndarray, symbols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exposures ``[T, S, K]`` at each bar and whether each symbol has them all.

    Only the bars before the last are read: the last bar's exposures attribute
    nothing in the window.
    """
    n_bars, n_symbols, n_factors = timestamps.size, len(symbols), len(risk_model.factor_names)
    matrix = np.zeros((n_bars, n_symbols, n_factors))
    covered = np.zeros((n_bars, n_symbols), dtype=bool)
    if n_bars < 2:
        return matrix, covered
    panel = risk_model.exposures(pd.Timestamp(timestamps[0]), pd.Timestamp(timestamps[-2]))
    panel = panel.assign_coords(symbol=[str(s) for s in panel["symbol"].values])
    panel = panel.reindex(timestamp=timestamps[:-1], symbol=symbols).load()
    for t in range(n_bars - 1):
        rows, has = risk_model.exposure_matrix(panel.isel(timestamp=t, drop=True))
        has = np.asarray(has, dtype=bool) & np.isfinite(rows).all(axis=1)
        matrix[t] = np.where(has[:, None], rows, 0.0)
        covered[t] = has
    return matrix, covered
