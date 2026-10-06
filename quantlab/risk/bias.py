"""Bias statistics: whether a risk model's volatility forecasts are calibrated.

Follows Menchero, Orr and Wang, *The Barra US Equity Model (USE4),
Methodology Notes* (MSCI, 2011), Appendix A. For a portfolio ``n`` with
return ``R_nt`` over the bar after ``t`` and volatility forecast ``sigma_nt``
made at ``t``, the standardized outcome ``b_nt = R_nt / sigma_nt`` (A1) has
a standard deviation of 1 when the forecasts are right. The bias statistic is
the realized standard deviation of the outcomes over a window of ``T`` bars
(A2, about their mean, ``T - 1`` in the denominator); for normal returns and
right forecasts about 95 percent of bias statistics fall within ``1 +/-
sqrt(2/T)`` (A3). Above the band the model underpredicts risk, below it
overpredicts. The rolling bias statistic is the same over a window rolled one
bar at a time (A4); across portfolios, its mean (A5), 5th and 95th
percentiles and mean absolute deviation from 1 (MRAD, A6) summarise each
window.

``bias_statistics`` computes these from any realized returns and forecasts.
``risk_model_bias_statistics`` reads them from a factor risk model's stores:

- **factor**: each factor's return over the next bar, the return of its pure
  factor portfolio, against the square root of its forecast variance;
- **specific**: each symbol's specific return over the next bar against its
  forecast specific risk;
- **random**: random active portfolios as in the USE4 Empirical Notes (§5,
  Figure 5.4): ``portfolio_size`` randomly selected estimation-universe
  stocks, capitalization weighted, less the capitalization-weighted
  estimation universe.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.risk.base import Date, FactorRiskModel


def bias_statistics(
    realized: xr.DataArray,
    forecast: xr.DataArray,
    window: int | None = None,
    min_observations: int | None = None,
) -> xr.Dataset:
    """Return the bias statistics of volatility forecasts against realized returns.

    Parameters
    ----------
    realized : xr.DataArray
        On ``(timestamp, <portfolio dim>)``: each portfolio's return over the
        bar after ``timestamp``.
    forecast : xr.DataArray
        Same coordinates: the volatility forecast for that return, made at
        ``timestamp``. A forecast that is missing or not positive gives no
        outcome.
    window : int, optional
        Bars per rolling window; without it no rolling statistics are
        returned. USE4 rolls 12-month windows (252 daily bars).
    min_observations : int, optional
        Fewest outcomes a rolling window needs for a bias statistic; NaN with
        fewer. Defaults to half the window, rounded up (our choice: a
        portfolio may miss a few bars, a symbol on a halt).

    Returns
    -------
    xr.Dataset
        ``realized``, ``forecast`` and ``outcome`` (their ratio, A1) on the
        input's dimensions. Over the whole range, one window (A2, A3), per
        portfolio: ``bias``, ``count`` (outcomes) and ``band`` (the half-width
        of the confidence interval, ``sqrt(2 / count)``). With ``window``, on
        the input's dimensions, labelled by each window's last bar:
        ``rolling_bias`` (A4), ``rolling_count`` and ``rolling_band``; and on
        ``timestamp``, across portfolios, ``rolling_mean`` (A5),
        ``rolling_p5``, ``rolling_p95`` and ``rolling_mrad`` (A6).

    Raises
    ------
    ValueError
        If ``window`` is below 2, or ``min_observations`` is not between 2
        and ``window``.

    Examples
    --------
    With ``returns`` drawn with exactly the volatility ``sigma``:

    >>> stats = bias_statistics(returns, sigma, window=252)
    >>> float(stats["bias"].mean())  # doctest: +SKIP
    1.003
    >>> float(bias_statistics(returns, sigma / 2)["bias"].mean())  # doctest: +SKIP
    2.006
    """
    if realized.dims != forecast.dims or realized.dims[0] != "timestamp" or realized.ndim != 2:
        raise ValueError(
            f"bias_statistics(): realized and forecast must both be on (timestamp, "
            f"<portfolio dim>); got {realized.dims} and {forecast.dims}."
        )
    forecast = forecast.reindex_like(realized)
    realized_values = realized.values.astype(np.float64)
    forecast_values = forecast.values.astype(np.float64)
    usable = np.isfinite(realized_values) & np.isfinite(forecast_values) & (forecast_values > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        outcome = np.where(usable, realized_values / forecast_values, np.nan)

    dims = realized.dims
    stats = xr.Dataset(
        {
            "realized": (dims, realized_values),
            "forecast": (dims, forecast_values),
            "outcome": (dims, outcome),
        },
        coords=realized.coords,
    )
    count = np.isfinite(outcome).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # fewer than two outcomes
        bias = np.nanstd(outcome, axis=0, ddof=1)
    stats["bias"] = (dims[1:], np.where(count >= 2, bias, np.nan))
    stats["count"] = (dims[1:], count)
    with np.errstate(divide="ignore"):
        stats["band"] = np.sqrt(2.0 / stats["count"].astype(np.float64))
    if window is None:
        return stats

    if window < 2:
        raise ValueError(f"bias_statistics(): window must be at least 2, got {window}.")
    least = -(-window // 2) if min_observations is None else min_observations
    if not 2 <= least <= window:
        raise ValueError(
            f"bias_statistics(): min_observations must be between 2 and the window, "
            f"{window}; got {least}."
        )
    rolling, rolling_count = _rolling_std(outcome, window, least)
    stats["rolling_bias"] = (dims, rolling)
    stats["rolling_count"] = (dims, rolling_count)
    with np.errstate(divide="ignore"):
        stats["rolling_band"] = (dims, np.sqrt(2.0 / rolling_count))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # bars without a window
        stats["rolling_mean"] = ("timestamp", np.nanmean(rolling, axis=1))
        stats["rolling_p5"] = ("timestamp", np.nanpercentile(rolling, 5, axis=1))
        stats["rolling_p95"] = ("timestamp", np.nanpercentile(rolling, 95, axis=1))
        stats["rolling_mrad"] = ("timestamp", np.nanmean(np.abs(rolling - 1.0), axis=1))
    return stats


def _rolling_std(values: np.ndarray, window: int, least: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``ddof=1`` standard deviation and count of each column's trailing window.

    ``values`` is ``[T, N]`` with NaN for a missing value; row ``t`` covers
    rows ``t - window + 1`` to ``t``. A window with fewer than ``least``
    values is NaN. Uses pandas' online rolling variance.
    """
    frame = pd.DataFrame(values)
    rolling = frame.rolling(window, min_periods=least)
    count = frame.notna().astype(np.int64).rolling(window, min_periods=1).sum()
    return rolling.std(ddof=1).to_numpy(), count.to_numpy().astype(np.int64)


def risk_model_bias_statistics(
    model: FactorRiskModel,
    start: Date,
    end: Date,
    *,
    window: int | None = 252,
    min_observations: int | None = None,
    random_portfolios: int = 100,
    portfolio_size: int = 500,
    seed: int = 0,
) -> dict[str, xr.Dataset]:
    """Return the bias statistics of a factor risk model's forecasts from its stores.

    Every estimate row from ``start`` to ``end`` is a forecast made at its
    bar ``t``; its outcome is the regression row of the next bar of the
    regression store, so the last estimate bar of the store has none. The
    exposures are the model's (``FactorRiskModel.exposures``).

    A random active portfolio holds, at each bar, the ``portfolio_size``
    eligible stocks it ranks first, capitalization weighted, less every
    eligible stock capitalization weighted. Each portfolio ranks the symbols
    of the estimate store at random once, from ``seed`` (our choice: a stock
    is kept while it stays eligible, and models sharing a regression store
    get the same portfolios). A stock is eligible at ``t`` when it is in the
    estimation universe, has every exposure, a market cap, a specific risk
    and a specific return over the next bar (our choice: the stores hold no
    return for a stock that stops trading, so it leaves the portfolios the
    bar before), and every factor it is exposed to has a forecast variance.
    Its forecast variance is ``x' F x + sum h^2 s^2`` with ``x`` the
    portfolio's factor exposures, a correlation without enough bars counted
    as 0 (our choice); its realized return is ``sum h (X f + u)`` over the
    next bar, the stocks' excess returns, a factor without a return on that
    bar contributing nothing.

    Parameters
    ----------
    model : FactorRiskModel
        A model whose regression and estimate stores cover the range (the
        regression store to at least the bar after ``end`` for the last
        forecast to have an outcome).
    start, end : str, datetime.date or pd.Timestamp
        The forecast bars, both inclusive, inside the estimate store's range.
    window, min_observations : int, optional
        The rolling window and its fewest outcomes (``bias_statistics``);
        ``window=None`` for no rolling statistics.
    random_portfolios : int, default 100
        Random active portfolios (USE4 Empirical Notes: 100).
    portfolio_size : int, default 500
        Stocks per random portfolio (USE4 Empirical Notes: 500). With as
        many stocks as the eligible universe, a portfolio is the benchmark
        and has no active risk.
    seed : int, default 0
        Seed of the random ranking.

    Returns
    -------
    dict of str to xr.Dataset
        ``"factor"`` on ``(timestamp, factor)``, ``"specific"`` on
        ``(timestamp, symbol)`` and ``"random"`` on ``(timestamp,
        portfolio)``, each as returned by ``bias_statistics``, ``timestamp``
        the forecast bar.

    Raises
    ------
    ValueError
        If a store has no recorded range or does not cover the range
        (``RiskStore.read``).

    Examples
    --------
    >>> stats = risk_model_bias_statistics(model, "2010-01-04", "2025-12-31")
    >>> stats["factor"]["bias"].sel(factor="style_beta")  # doctest: +SKIP
    >>> stats["random"]["rolling_mean"].plot()  # doctest: +SKIP
    """
    config = model.config
    estimate = model.estimate.read(start, end).load()
    recorded = model.regression.store_range()
    if recorded is None:
        raise ValueError(
            f"risk_model_bias_statistics(): the regression store at "
            f"{config.regression_path} has no recorded range; build it first."
        )
    regression = model.regression.read(start, recorded[1]).load()
    bars = estimate["timestamp"].values
    regression_bars = regression["timestamp"].values
    position = np.searchsorted(regression_bars, bars)
    known = (position < len(regression_bars)) & (
        regression_bars[np.minimum(position, len(regression_bars) - 1)] == bars
    )
    if not known.all():
        missing = bars[~known][0]
        raise ValueError(
            f"risk_model_bias_statistics(): estimate bar {missing} is not a bar of the "
            f"regression store; rebuild the estimate store from it."
        )
    has_next = position + 1 < len(regression_bars)
    bars, following = bars[has_next], position[has_next] + 1
    if not len(bars):
        raise ValueError(
            f"risk_model_bias_statistics(): no estimate bar from {start} to {end} has a "
            f"next bar in the regression store, so no forecast has an outcome."
        )
    estimate = estimate.sel(timestamp=bars)

    factor_returns = regression["factor_return"].transpose("timestamp", "factor").values[following]
    covariance = estimate["factor_covariance"].transpose("timestamp", "factor_i", "factor_j").values
    variance = np.diagonal(covariance, axis1=1, axis2=2)
    with np.errstate(invalid="ignore"):
        factor_volatility = np.sqrt(variance)
    factor = _named(
        factor_returns, factor_volatility, bars, "factor", list(model.factor_names)
    )

    symbols = estimate["symbol"].values
    specific_risk = estimate["specific_risk"].transpose("timestamp", "symbol").values
    specific_returns = (
        regression["specific_return"]
        .reindex(symbol=symbols)
        .transpose("timestamp", "symbol")
        .values[following]
    )
    specific = _named(specific_returns, specific_risk, bars, "symbol", symbols)

    exposures = model.exposures(bars[0], bars[-1]).reindex(timestamp=bars, symbol=symbols).load()
    cap = (
        model.prices(bars[0], bars[-1])[config.market_cap_column]
        .reindex(timestamp=bars, symbol=symbols)
        .transpose("timestamp", "symbol")
        .values
    )
    ranking = np.random.default_rng(seed).random((random_portfolios, len(symbols)))
    realized = np.full((len(bars), random_portfolios), np.nan)
    forecast = np.full((len(bars), random_portfolios), np.nan)
    if config.estu_name is None:
        estu = np.ones(cap.shape, dtype=bool)
    else:
        estu = exposures[config.estu_name].transpose("timestamp", "symbol").values == 1.0
    for row in range(len(bars)):
        matrix, covered = model.exposure_matrix(exposures.isel(timestamp=row))
        realized[row], forecast[row] = _random_active(
            matrix,
            covered & estu[row],
            cap[row],
            covariance[row],
            specific_risk[row],
            factor_returns[row],
            specific_returns[row],
            ranking,
            portfolio_size,
        )
    random = _named(realized, forecast, bars, "portfolio", np.arange(random_portfolios))

    return {
        name: bias_statistics(realized, forecast, window, min_observations)
        for name, (realized, forecast) in
        {"factor": factor, "specific": specific, "random": random}.items()
    }


def _named(realized, forecast, bars, dim, items) -> tuple[xr.DataArray, xr.DataArray]:
    """Wrap ``[T, N]`` realized returns and forecasts as data arrays on ``(timestamp, dim)``."""
    coords = {"timestamp": bars, dim: items}
    return (
        xr.DataArray(realized, coords, ("timestamp", dim)),
        xr.DataArray(forecast, coords, ("timestamp", dim)),
    )


def _random_active(
    matrix: np.ndarray,
    candidates: np.ndarray,
    cap: np.ndarray,
    covariance: np.ndarray,
    specific_risk: np.ndarray,
    factor_returns: np.ndarray,
    specific_returns: np.ndarray,
    ranking: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each random active portfolio's realized return and forecast volatility at one bar.

    See ``risk_model_bias_statistics`` for the portfolios. ``matrix`` holds
    the symbols' exposures (``FactorRiskModel.exposure_matrix``),
    ``candidates`` whether a symbol has them all and is in the estimation
    universe. ``ranking`` is ``[portfolios, symbols]``; a portfolio holds its
    highest-ranked eligible stocks.
    """
    has_variance = np.isfinite(np.diagonal(covariance))
    eligible = (
        candidates & np.isfinite(cap) & (cap > 0)
        & np.isfinite(specific_risk) & np.isfinite(specific_returns)
        & ~((matrix != 0) & ~has_variance).any(axis=1)
    )
    n_portfolios = len(ranking)
    if not eligible.any():
        return np.full(n_portfolios, np.nan), np.full(n_portfolios, np.nan)
    members = np.flatnonzero(eligible)
    exposure, risk, member_cap = matrix[members], specific_risk[members], cap[members]
    stock_returns = exposure @ np.nan_to_num(factor_returns) + specific_returns[members]

    # Each portfolio's top ``size`` members by its ranking, cap weighted,
    # less the cap-weighted universe of eligible stocks.
    if size >= len(members):
        # Every portfolio is the benchmark.
        return np.zeros(n_portfolios), np.zeros(n_portfolios)
    held = np.zeros((n_portfolios, len(members)), dtype=bool)
    top = np.argpartition(-ranking[:, members], size - 1, axis=1)[:, :size]
    np.put_along_axis(held, top, True, axis=1)
    weights = np.where(held, member_cap, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    active = weights - member_cap / member_cap.sum()

    factor_covariance = np.nan_to_num(covariance)
    portfolio_exposure = active @ exposure
    variance = np.einsum(
        "pi,ij,pj->p", portfolio_exposure, factor_covariance, portfolio_exposure
    ) + (active**2) @ (risk**2)
    return active @ stock_returns, np.sqrt(np.clip(variance, 0.0, None))


__all__ = ["bias_statistics", "risk_model_bias_statistics"]
