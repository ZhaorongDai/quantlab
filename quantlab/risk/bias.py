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
``risk_model_bias_statistics`` reads them from a factor risk model's stores,
over one bar or a ``horizon`` of several (USE4 tests monthly), the outcome
then the sum of the bars' returns and the forecast ``sqrt(h)`` times the
one-bar one:

- **factor**: each factor's return over the next bar, the return of its pure
  factor portfolio, against the square root of its forecast variance;
- **eigenfactor**: the eigenfactors of each bar's forecast covariance (USE4
  Methodology Notes §4.2, Figure 4.1), numbered from the lowest variance:
  the next bar's return of each, the factor returns weighted by its unit
  eigenvector, against the square root of its eigenvalue;
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

from quantlab.risk.base import Date, FactorRiskForecast, FactorRiskModel, covered_factors


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
    Twenty portfolios over 1000 bars, their returns drawn with exactly the
    forecast volatility ``sigma``:

    >>> import numpy as np, pandas as pd, xarray as xr
    >>> rng = np.random.default_rng(0)
    >>> sigma = xr.DataArray(
    ...     np.full((1000, 20), 0.01), dims=("timestamp", "portfolio"),
    ...     coords={"timestamp": pd.bdate_range("2020-01-01", periods=1000)},
    ... )
    >>> returns = sigma * rng.standard_normal((1000, 20))
    >>> stats = bias_statistics(returns, sigma, window=252)
    >>> round(float(stats["bias"].mean()), 2), round(float(stats["band"][0]), 3)
    (1.0, 0.045)
    >>> round(float(stats["rolling_mean"].isel(timestamp=-1)), 2)
    0.99

    A forecast of half the volatility underpredicts risk by two:

    >>> round(float(bias_statistics(returns, sigma / 2)["bias"].mean()), 2)
    1.99
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
    horizon: int = 1,
    window: int | None = None,
    min_observations: int | None = None,
    random_portfolios: int = 100,
    portfolio_size: int = 500,
    seed: int = 0,
) -> dict[str, xr.Dataset]:
    """Return the bias statistics of a factor risk model's forecasts from its stores.

    Every estimate row is a forecast made at its bar ``t`` of the next bar.
    Over a ``horizon`` of ``h`` bars, a forecast is taken every ``h``
    estimate bars from ``start`` (the outcomes do not overlap), its outcome
    is the sum of the next ``h`` bars' returns in the regression store, and
    its forecast volatility ``sqrt(h)`` times the one-bar one: the one-bar
    forecasts are of the variance per bar of a multi-bar return, which is
    what Newey-West adjusts, so a horizon longer than one bar tests that
    adjustment. A forecast without ``h`` regression bars after it, or an
    outcome missing one of its bars, has none. The forecasts, outcomes,
    exposures, market caps and estimation universe are read aligned by
    ``FactorRiskModel.forecast_window``.

    The eigenfactors of a bar are those of the covariance block of the
    factors ``covered_factors`` keeps, numbered from 1, the lowest
    eigenvalue; with fewer factors kept, the last ones are missing that bar.
    An eigenfactor has no outcome on a bar missing the return of a kept
    factor, nor when its eigenvalue is not positive (a pairwise estimate
    need not be positive semi-definite).

    A random active portfolio holds, at each bar, the ``portfolio_size``
    eligible stocks it ranks first, capitalization weighted, less every
    eligible stock capitalization weighted, held over the horizon. Each portfolio ranks the symbols
    of the estimate store at random once, from ``seed`` (our choice: a stock
    is kept while it stays eligible, and models sharing a regression store
    get the same portfolios). A stock is eligible at ``t`` when the model's
    forecast at ``t`` covers it (``FactorRiskModel.forecast``: every
    exposure, a specific risk, no exposure to a factor without a
    covariance, the rule the portfolio's covariance estimator uses), and it
    is in the estimation universe, with a market cap and a return over
    every bar of the horizon (our choice: the stores hold no return for a
    stock that stops trading, so it leaves the portfolios before). Its
    forecast variance is that forecast's, ``h (x' F x + sum w^2 s^2)`` with
    ``w`` its active weights and ``x`` its factor exposures; its realized return is ``sum w (X
    f + u)`` summed over the horizon's bars, the stocks' excess returns (each
    bar's ``X`` the exposures of the bar before it), a factor without a
    return on a bar contributing nothing.

    Parameters
    ----------
    model : FactorRiskModel
        A model whose regression and estimate stores cover the range (the
        regression store to at least the bar after ``end`` for the last
        forecast to have an outcome).
    start, end : str, datetime.date or pd.Timestamp
        The forecast bars, both inclusive, inside the estimate store's range.
    horizon : int, default 1
        Bars per outcome. USE4 tests monthly (21 daily bars).
    window : int, optional
        Outcomes per rolling window (``bias_statistics``); by default about
        a year, ``round(252 / horizon)`` (USE4: 12 months).
    min_observations : int, optional
        Fewest outcomes in a rolling window (``bias_statistics``).
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
        ``"factor"`` on ``(timestamp, factor)``, ``"eigenfactor"`` on
        ``(timestamp, eigenfactor)``, ``"specific"`` on ``(timestamp,
        symbol)`` and ``"random"`` on ``(timestamp, portfolio)``, each as returned by ``bias_statistics``, ``timestamp``
        the forecast bar.

    Raises
    ------
    ValueError
        If ``horizon`` is below 1, or a store has no recorded range or does
        not cover the range (``RiskStore.read``).

    Examples
    --------
    With ``model`` the ``Use4RiskModel`` of
    ``examples/sharadar_us_equity/risk_model.py``, its stores built over the
    Sharadar history:

    >>> stats = risk_model_bias_statistics(model, "2007-07-13", "2026-10-02")
    >>> sorted(stats)
    ['eigenfactor', 'factor', 'random', 'specific']
    >>> round(float(stats["random"]["bias"].mean()), 3)
    1.003
    >>> round(float(stats["factor"]["bias"].sel(factor="style_beta")), 3)
    1.119
    """
    if horizon < 1:
        raise ValueError(f"risk_model_bias_statistics(): horizon must be at least 1, got {horizon}.")
    estimate = model.estimate.read(start, end)
    bars, symbols = estimate["timestamp"].values, estimate["symbol"].values
    recorded = model.regression.store_range()
    if recorded is None:
        raise ValueError(
            f"risk_model_bias_statistics(): {model.class_name}'s regression store has no "
            f"recorded range; build it first."
        )
    regression_bars = model.regression.read(start, recorded[1])["timestamp"].values
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
    # Non-overlapping outcomes: a forecast every ``horizon`` bars, each with
    # the ``horizon`` regression bars after it.
    complete = np.flatnonzero(position + horizon < len(regression_bars))
    picks = complete[::horizon]
    if not len(picks):
        raise ValueError(
            f"risk_model_bias_statistics(): no estimate bar from {start} to {end} has "
            f"{horizon} bar(s) after it in the regression store, so no forecast has an "
            f"outcome."
        )
    bars, origin = bars[picks], position[picks]
    # Row r of ``aligned`` is regression bar origin[0] + r: its outcome beside
    # the forecast inputs of the bar before. A pick's forecast is the row after it.
    aligned = model.forecast_window(
        regression_bars[origin[0] : origin[-1] + horizon + 1], forecasts_through=bars[-1]
    ).reindex(symbol=symbols)
    rows = origin - origin[0]
    scale = float(horizon)

    def summed(values: np.ndarray) -> np.ndarray:
        """Sum the rows after each pick over the horizon; NaN when one is missing."""
        total = np.zeros((len(rows), *values.shape[1:]))
        for step in range(1, horizon + 1):
            total += values[rows + step]
        return total

    factor_returns = aligned["factor_return"].transpose("timestamp", "factor").values
    covariance = (
        aligned["factor_covariance"].transpose("timestamp", "factor_i", "factor_j").values[rows + 1]
    )
    variance = np.diagonal(covariance, axis1=1, axis2=2)
    with np.errstate(invalid="ignore"):
        factor_volatility = np.sqrt(scale * variance)
    factor_outcomes = summed(factor_returns)
    factor = _named(factor_outcomes, factor_volatility, bars, "factor", list(model.factor_names))

    n_factors = len(model.factor_names)
    eigen_realized = np.full((len(bars), n_factors), np.nan)
    eigen_forecast = np.full((len(bars), n_factors), np.nan)
    for row in range(len(bars)):
        kept = covered_factors(covariance[row])
        if not kept.any():
            continue
        block = covariance[row][np.ix_(kept, kept)]
        eigenvalues, eigenvectors = np.linalg.eigh((block + block.T) / 2)
        eigen_realized[row, : kept.sum()] = factor_outcomes[row, kept] @ eigenvectors
        eigen_forecast[row, : kept.sum()] = np.sqrt(scale * np.clip(eigenvalues, 0.0, None))
    eigenfactor = _named(
        eigen_realized, eigen_forecast, bars, "eigenfactor", np.arange(1, n_factors + 1)
    )

    specific_risk = aligned["specific_risk"].transpose("timestamp", "symbol").values[rows + 1]
    specific_returns = aligned["specific_return"].transpose("timestamp", "symbol").values
    specific = _named(
        summed(specific_returns), np.sqrt(scale) * specific_risk, bars, "symbol", symbols
    )

    # Each stock's excess return over a bar is its exposures of the bar before
    # (the row's) times the bar's factor returns plus its specific return; a
    # factor without a return on the bar adds nothing.
    ranking = np.random.default_rng(seed).random((random_portfolios, len(symbols)))
    realized = np.full((len(bars), random_portfolios), np.nan)
    forecast = np.full((len(bars), random_portfolios), np.nan)
    symbol_position = pd.Index(symbols)
    for k, row in enumerate(rows):
        stock_returns = np.zeros(len(symbols))
        for step in range(1, horizon + 1):
            bar = aligned.isel(timestamp=row + step)
            stock_returns += (
                model.exposure_matrix(bar)[0] @ np.nan_to_num(factor_returns[row + step])
                + specific_returns[row + step]
            )
        inputs = aligned.isel(timestamp=row + 1)
        at_bar = model.forecast(inputs, inputs).scaled(scale)
        position = symbol_position.get_indexer(at_bar.symbols)
        realized[k], forecast[k] = _random_active(
            at_bar,
            np.asarray(inputs["estimation_universe"].values == True)[position],  # noqa: E712 - NaN is not
            np.asarray(inputs["market_cap"].values, dtype=np.float64)[position],
            stock_returns[position],
            ranking[:, position],
            portfolio_size,
        )
    random = _named(realized, forecast, bars, "portfolio", np.arange(random_portfolios))

    if window is None:
        window = max(2, round(252 / horizon))
    return {
        name: bias_statistics(realized, forecast, window, min_observations)
        for name, (realized, forecast) in
        {
            "factor": factor, "eigenfactor": eigenfactor, "specific": specific,
            "random": random,
        }.items()
    }


def _named(realized, forecast, bars, dim, items) -> tuple[xr.DataArray, xr.DataArray]:
    """Wrap ``[T, N]`` realized returns and forecasts as data arrays on ``(timestamp, dim)``."""
    coords = {"timestamp": bars, dim: items}
    return (
        xr.DataArray(realized, coords, ("timestamp", dim)),
        xr.DataArray(forecast, coords, ("timestamp", dim)),
    )


def _random_active(
    at_bar: FactorRiskForecast,
    estu: np.ndarray,
    cap: np.ndarray,
    stock_returns: np.ndarray,
    ranking: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each random active portfolio's realized return and forecast volatility at one bar.

    See ``risk_model_bias_statistics`` for the portfolios. ``at_bar`` is the
    model's forecast at the bar over the horizon; ``estu``, ``cap``,
    ``stock_returns`` (over the horizon) and ``ranking`` (``[portfolios,
    symbols]``; a portfolio holds its highest-ranked eligible stocks) are on
    its symbols. A stock is eligible in the estimation universe, with a
    market cap and a return.
    """
    eligible = estu & np.isfinite(cap) & (cap > 0) & np.isfinite(stock_returns)
    n_portfolios = len(ranking)
    if not eligible.any():
        return np.full(n_portfolios, np.nan), np.full(n_portfolios, np.nan)
    members = np.flatnonzero(eligible)
    member_cap, stock_returns = cap[members], stock_returns[members]

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

    factor, specific = at_bar.subset(members).portfolio_variance(active)
    return active @ stock_returns, np.sqrt(np.clip(factor + specific, 0.0, None))


__all__ = ["bias_statistics", "risk_model_bias_statistics"]
