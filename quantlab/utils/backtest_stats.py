"""Return, relative, win-rate and turnover statistics of a backtest, as public functions.

These are the returns-based rows and the turnover rows of a backtest run's
``metrics.json``: the ``in_sample``, ``out_of_sample`` and ``benchmark``
return blocks, the ``relative`` block, the win rates and the three turnover
rows. ``quantlab.base.backtest`` computes them through this module for every
engine, and a tool that simulates a run elsewhere
(an event-driven replay of a quantlab run) calls the same functions to report
comparable numbers.

The module imports only numpy, pandas and xarray: no quantlab model or
dataset layer, no vectorbt and no torch, so importing it is cheap and drags
in no simulation engine. ``return_stats`` reimplements vectorbt's
``ReturnsAccessor.stats`` (vectorbt 1.1) to the bit: every sum and product
is accumulated in index order, as vectorbt's compiled kernels do, rather than
with numpy's pairwise reductions, which round differently in the last bits.

Every function takes per-bar values on a ``timestamp`` axis (an
``xarray.DataArray``) and plain numbers; *ranges* are inclusive pairs of bar
labels (``"2024-01-02"`` or ``"2024-01-02T15:30:00"``), compared as exact
timestamps, in time order and never overlapping.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

#: Days in a calendar year, used to annualize bars longer than one day.
CALENDAR_DAYS_PER_YEAR = 365.25


def year_freq(bar_interval, trading_days_per_year: int, session_minutes_per_day: int) -> pd.Timedelta:
    """Return one year, in vectorbt's convention, for bars of ``bar_interval``.

    ``year_freq / bar_interval`` is the number of bars per year. Intraday
    bars use trading days times session minutes divided by the bar's
    minutes (one-minute bars: 252 x 390). A bar of exactly one day is one
    trading day, giving ``trading_days_per_year``. A longer bar spans
    calendar time (a weekly bar is one calendar week whatever the holidays),
    so bars per year is ``CALENDAR_DAYS_PER_YEAR`` divided by the bar's days,
    capped at ``trading_days_per_year``. The function is continuous at one
    day and non-increasing in the interval.

    Parameters
    ----------
    bar_interval
        Anything ``pd.Timedelta`` accepts, such as ``"1D"``, ``"5min"`` or a
        ``numpy.timedelta64``.
    trading_days_per_year : int
        Trading days in a year of the market.
    session_minutes_per_day : int
        Length of one trading session in minutes.

    Returns
    -------
    pd.Timedelta
        The year length.

    Raises
    ------
    ValueError
        If ``bar_interval`` is not positive.

    Examples
    --------
    >>> year_freq("1D", 252, 390) / pd.Timedelta("1D")
    252.0
    >>> year_freq("1min", 252, 390) / pd.Timedelta("1min")
    98280.0
    >>> round(year_freq("7D", 252, 390) / pd.Timedelta("7D"), 2)
    52.18
    """
    interval = pd.Timedelta(bar_interval)
    if interval <= pd.Timedelta(0):
        raise ValueError(f"bar_interval must be positive, got {interval}")
    one_day = pd.Timedelta(days=1)
    if interval >= one_day:
        bars_per_year = min(
            float(trading_days_per_year), CALENDAR_DAYS_PER_YEAR * (one_day / interval)
        )
    else:
        minutes = interval / pd.Timedelta(minutes=1)
        bars_per_year = trading_days_per_year * session_minutes_per_day / minutes
    return interval * bars_per_year


def label_ns(label) -> np.datetime64:
    """Return a bar label (or any timestamp) as a nanosecond ``datetime64``.

    Parameters
    ----------
    label
        A bar label as ``metrics.json`` writes it (an ISO date or timestamp),
        or anything ``pd.Timestamp`` accepts.

    Returns
    -------
    numpy.datetime64
        The exact instant, at nanosecond resolution; a date is midnight.

    Examples
    --------
    >>> label_ns("2024-01-02")
    np.datetime64('2024-01-02T00:00:00.000000000')
    """
    return np.datetime64(pd.Timestamp(str(label)).to_datetime64(), "ns")


def in_ranges(timestamps, ranges: Sequence[tuple[str, str]]) -> np.ndarray:
    """Return a boolean mask of the ``timestamps`` inside any of ``ranges``.

    Both ends of a range are included and compared as exact timestamps (a
    date is midnight), never by day.

    Parameters
    ----------
    timestamps : array_like of datetime64
        The bars to test.
    ranges : sequence of (str, str)
        Inclusive bar-label ranges.

    Returns
    -------
    numpy.ndarray
        A boolean mask, ``True`` where a bar falls inside any range.

    Examples
    --------
    >>> ts = pd.bdate_range("2024-01-01", periods=5).values
    >>> in_ranges(ts, [("2024-01-02", "2024-01-03")]).tolist()
    [False, True, True, False, False]
    """
    ts = np.asarray(timestamps).astype("datetime64[ns]")
    mask = np.zeros(ts.size, dtype=bool)
    for start, end in ranges:
        mask |= (ts >= label_ns(start)) & (ts <= label_ns(end))
    return mask


def _cut(returns: xr.DataArray, ranges: Sequence[tuple[str, str]] | None) -> pd.Series:
    """Return ``returns`` as a pandas Series, cut to ``ranges`` when given."""
    series = returns.to_pandas()
    if ranges is None:
        return series
    pieces = [
        series.loc[pd.Timestamp(str(start)) : pd.Timestamp(str(end))]
        for start, end in ranges
    ]
    return pd.concat(pieces) if len(pieces) > 1 else pieces[0]


def _seq_sum(values: np.ndarray) -> float:
    """Sum in index order, as a compiled loop does (numpy's ``sum`` is pairwise)."""
    return float(np.cumsum(values)[-1]) if values.size else 0.0


def _nanmean(values: np.ndarray) -> float:
    """The mean of the non-NaN values, accumulated in order; NaN when there are none."""
    kept = values[~np.isnan(values)]
    if kept.size == 0:
        return float("nan")
    return _seq_sum(kept) / kept.size


def _nanstd(values: np.ndarray, ddof: int) -> float:
    """The standard deviation of the non-NaN values, as vectorbt's ``nanstd_1d_nb``."""
    kept = values[~np.isnan(values)]
    count = kept.size
    rcount = max(count - ddof, 0)
    if rcount == 0:
        return float("nan")
    mean = _seq_sum(kept) / count
    deviation = kept - mean
    variance = _seq_sum(deviation * deviation) / count
    return float(np.sqrt(variance * count / rcount))


def _percentile(values: np.ndarray, q: float) -> float:
    """The ``q``-th percentile by linear interpolation, in numba's arithmetic.

    numpy interpolates the same two order statistics with a differently
    rounded formula; this is the one vectorbt's compiled kernels run.
    """
    if values.size == 1:
        return float(values[0])
    ordered = np.sort(values)
    rank = 1 + (values.size - 1) * (q / 100.0)
    floor = int(np.floor(rank))
    weight = rank - floor
    return float(ordered[floor - 1] * (1 - weight) + ordered[floor] * weight)


def _cumulative(values: np.ndarray) -> np.ndarray:
    """The compounded value of 1 after each bar; a NaN return keeps the value."""
    return np.cumprod(np.where(np.isnan(values), 1.0, values + 1.0))


def _drawdown_records(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each drawdown's depth and duration in bars, as vectorbt records them.

    A drawdown starts on the bar after a peak and ends on the bar that
    recovers it (duration ``end - start``) or, still running, on the last bar
    (duration ``end - start + 1``).
    """
    depths, durations = [], []
    running = False
    peak_idx, peak_val, valley_val = -1, value[0], value[0]
    last = value.size - 1
    for i, current in enumerate(value):
        if np.isnan(current):
            continue
        stored = None
        if np.isnan(peak_val) or current >= peak_val:
            if not running:
                peak_val, peak_idx = current, i
            else:
                running = False
                stored = "recovered"
        elif not running:
            running = True
            valley_val = current
        elif current < valley_val:
            valley_val = current
        if i == last and running:
            running = False
            stored = "active"
        if stored is not None:
            depths.append((valley_val - peak_val) / peak_val)
            start = peak_idx + 1
            durations.append(i - start + (1 if stored == "active" else 0))
            peak_idx, peak_val, valley_val = i, current, current
    return np.asarray(depths, dtype=np.float64), np.asarray(durations, dtype=np.int64)


def return_stats(
    returns: xr.DataArray,
    *,
    bar_interval,
    year_freq: pd.Timedelta,
    ranges: Sequence[tuple[str, str]] | None = None,
) -> dict:
    """Return the return statistics of a per-bar return series, keyed by row name.

    These are the rows of the ``in_sample``, ``out_of_sample`` and
    ``benchmark`` blocks of ``metrics.json``, equal to the bit to vectorbt's
    ``ReturnsAccessor.stats`` with ``freq=bar_interval`` and ``year_freq``.
    ``ann`` below is ``year_freq / bar_interval``, the bars per year, and a
    NaN return is skipped by every row except ``Period`` and the
    annualization exponent, which count every bar.

    - ``Start``, ``End``: the first and last bar label (``pd.Timestamp``);
      ``Period``: the number of bars times ``bar_interval``.
    - ``Total Return [%]``: compounded return; ``Annualized Return [%]``:
      ``(1 + total) ** (ann / bars) - 1``; ``Annualized Volatility [%]``:
      standard deviation (ddof 1) times ``sqrt(ann)``.
    - ``Max Drawdown [%]``: the deepest drawdown of the compounded curve, a
      positive percentage, NaN without any drawdown; ``Max Drawdown
      Duration``: the longest one in bars times ``bar_interval`` (``NaT``
      without any).
    - ``Sharpe Ratio`` (mean over standard deviation times ``sqrt(ann)``),
      ``Calmar Ratio``, ``Omega Ratio`` and ``Sortino Ratio`` with a zero
      risk-free rate and required return; ``inf`` when the denominator is 0.
    - ``Skew`` and ``Kurtosis`` (pandas, bias-corrected), ``Tail Ratio``
      (95th over 5th percentile, absolute), ``Common Sense Ratio`` (tail
      ratio times one plus the annualized return) and ``Value at Risk``
      (the 5th percentile return, a fraction).

    Parameters
    ----------
    returns : xarray.DataArray
        Per-bar returns on a sorted ``timestamp`` axis.
    bar_interval
        The bar spacing, anything ``pd.Timedelta`` accepts.
    year_freq : pd.Timedelta
        One year, as ``year_freq`` returns it.
    ranges : sequence of (str, str), optional
        Inclusive bar-label ranges to cut the series to, concatenated in
        order; the whole series when omitted.

    Returns
    -------
    dict
        Seventeen rows, in the order listed above.

    Raises
    ------
    ValueError
        If no return falls inside ``ranges``.

    Examples
    --------
    >>> returns = xr.DataArray(
    ...     [0.01, -0.02, 0.015, 0.005], dims=("timestamp",),
    ...     coords={"timestamp": pd.bdate_range("2024-01-01", periods=4)},
    ... )
    >>> stats = return_stats(
    ...     returns, bar_interval="1D", year_freq=year_freq("1D", 252, 390)
    ... )
    >>> round(stats["Total Return [%]"], 6), stats["Period"]
    (0.967023, Timedelta('4 days 00:00:00'))
    >>> round(stats["Max Drawdown [%]"], 6), stats["Max Drawdown Duration"]
    (2.0, Timedelta('3 days 00:00:00'))
    >>> sliced = return_stats(
    ...     returns, bar_interval="1D", year_freq=year_freq("1D", 252, 390),
    ...     ranges=[("2024-01-02", "2024-01-03")],
    ... )
    >>> round(sliced["Total Return [%]"], 6), sliced["Period"]
    (-0.53, Timedelta('2 days 00:00:00'))
    """
    series = _cut(returns, ranges)
    if series.empty:
        raise ValueError(f"no return inside {list(ranges or [])}")
    freq = pd.Timedelta(bar_interval)
    ann = pd.Timedelta(year_freq) / freq
    r = np.asarray(series.to_numpy(), dtype=np.float64)
    n = r.size
    nan, inf = float("nan"), float("inf")

    growth = _cumulative(r)
    total = float(growth[-1]) - 1.0
    with np.errstate(over="ignore"):
        annualized = float(np.power(growth[-1], ann / n)) - 1.0
    volatility = _nanstd(r, 1) * ann ** (1.0 / 2.0) if n >= 2 else nan

    depths, durations = _drawdown_records(growth)
    max_drawdown = float(depths.min()) if depths.size else nan
    max_duration = freq * int(durations.max()) if durations.size else pd.NaT

    if n < 2:
        sharpe = nan
    else:
        std = _nanstd(r, 1)
        sharpe = inf if std == 0.0 else _nanmean(r) / std * float(np.sqrt(ann))

    curve = growth * 100.0
    worst = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    calmar = nan if worst == 0.0 else annualized / abs(worst)

    if ann == 1:
        threshold = 0.0
    elif ann <= -1:
        threshold = nan
    else:
        threshold = (1 + 0.0) ** (1.0 / ann) - 1
    excess = r - 0.0 - threshold
    with np.errstate(invalid="ignore"):
        gains = _seq_sum(excess[excess > 0.0])
        losses = -1.0 * _seq_sum(excess[excess < 0.0])
    omega = nan if np.isnan(threshold) else (inf if losses == 0.0 else gains / losses)

    if n < 2:
        sortino = nan
    else:
        downside = np.where(r > 0, 0.0, r)
        downside_risk = float(np.sqrt(_nanmean(downside**2)) * np.sqrt(ann))
        sortino = inf if downside_risk == 0.0 else _nanmean(r) * ann / downside_risk

    kept = r[~np.isnan(r)]
    if kept.size:
        upper = abs(_percentile(kept, 95))
        lower = abs(_percentile(kept, 5))
        tail = inf if lower == 0.0 else upper / lower
        var = _percentile(kept, 5)
    else:
        tail = var = nan

    return {
        "Start": series.index[0],
        "End": series.index[-1],
        "Period": freq * n,
        "Total Return [%]": total * 100,
        "Annualized Return [%]": annualized * 100,
        "Annualized Volatility [%]": volatility * 100,
        "Max Drawdown [%]": -max_drawdown * 100,
        "Max Drawdown Duration": max_duration,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Omega Ratio": omega,
        "Sortino Ratio": sortino,
        "Skew": float(series.skew()),
        "Kurtosis": float(series.kurtosis()),
        "Tail Ratio": tail,
        "Common Sense Ratio": tail * (1 + annualized),
        "Value at Risk": var,
    }


def relative_stats(
    returns: xr.DataArray,
    benchmark_returns: xr.DataArray,
    *,
    bar_interval,
    year_freq: pd.Timedelta,
    ranges: Sequence[tuple[str, str]],
) -> dict:
    """Return a strategy's statistics relative to its benchmark over ``ranges``.

    These are the rows of the ``relative`` blocks of ``metrics.json`` (with
    ``win_rates`` against the benchmark). Both series are cut to ``ranges``
    and the bars where either is NaN are dropped. The *relative NAV*
    compounds ``(1 + r) / (1 + b)`` bar by bar (``r`` the strategy's return,
    ``b`` the benchmark's); over a whole window it equals the strategy's
    value divided by the benchmark's, the excess-return curve the report
    draws. ``ann`` is ``year_freq / bar_interval``, and every key ending in
    ``[%]`` is in percent:

    - ``Strategy Total Return [%]`` / ``Benchmark Total Return [%]``:
      compounded returns of each series;
    - ``Excess Return [%]``: relative NAV at the end minus 1, the geometric
      excess; ``Annualized Excess Return [%]``: the same compounded to one
      year;
    - ``Total Return Difference [%]``: the arithmetic difference of the two
      total returns;
    - ``Excess Max Drawdown [%]``: the deepest fall of the relative NAV from
      its running peak (starting at 1), negative or 0;
    - ``Tracking Error [%]``: standard deviation (ddof 1) of ``r - b`` times
      ``sqrt(ann)``; ``Information Ratio``: mean of ``r - b`` times ``ann``
      over the tracking error;
    - ``Beta`` and ``Correlation`` of ``r`` on ``b``, and ``CAPM Alpha
      [%]``, the annualized intercept ``mean(r) - beta * mean(b)``;
    - ``Win Rate vs Benchmark [%]``: share of bars with ``r > b``;
    - ``Bars``: number of bars used.

    A statistic that is undefined (fewer than two bars, a flat benchmark,
    zero tracking error) is NaN.

    Parameters
    ----------
    returns, benchmark_returns : xarray.DataArray
        Per-bar returns of the strategy and the benchmark, on the same
        ``timestamp`` axis.
    bar_interval
        The bar spacing, anything ``pd.Timedelta`` accepts.
    year_freq : pd.Timedelta
        One year, as ``year_freq`` returns it.
    ranges : sequence of (str, str)
        Inclusive bar-label ranges of the slice.

    Returns
    -------
    dict
        Thirteen rows, in the order listed above.

    Raises
    ------
    ValueError
        If the two series are not on the same bars.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=3)
    >>> strategy = xr.DataArray([0.02, -0.01, 0.03], dims="timestamp", coords={"timestamp": bars})
    >>> benchmark = xr.DataArray([0.01, 0.0, 0.01], dims="timestamp", coords={"timestamp": bars})
    >>> stats = relative_stats(
    ...     strategy, benchmark, bar_interval="1D",
    ...     year_freq=year_freq("1D", 252, 390), ranges=[("2024-01-01", "2024-01-03")],
    ... )
    >>> round(stats["Total Return Difference [%]"], 4), stats["Bars"]
    (1.9994, 3)
    >>> round(stats["Win Rate vs Benchmark [%]"], 2)
    66.67
    """
    ts = returns.timestamp.values.astype("datetime64[ns]")
    if not np.array_equal(ts, benchmark_returns.timestamp.values.astype("datetime64[ns]")):
        raise ValueError("the benchmark returns are not on the strategy's bars")
    mask = in_ranges(ts, ranges)
    r = np.asarray(returns.values, dtype=np.float64)[mask]
    b = np.asarray(benchmark_returns.values, dtype=np.float64)[mask]
    finite = np.isfinite(r) & np.isfinite(b)
    r, b = r[finite], b[finite]
    n = int(r.size)
    interval = pd.Timedelta(bar_interval)
    bars_per_year = float(pd.Timedelta(year_freq) / interval)
    nan = float("nan")

    stats = {
        "Strategy Total Return [%]": nan,
        "Benchmark Total Return [%]": nan,
        "Excess Return [%]": nan,
        "Annualized Excess Return [%]": nan,
        "Total Return Difference [%]": nan,
        "Excess Max Drawdown [%]": nan,
        "Tracking Error [%]": nan,
        "Information Ratio": nan,
        "Beta": nan,
        "Correlation": nan,
        "CAPM Alpha [%]": nan,
        "Win Rate vs Benchmark [%]": nan,
        "Bars": n,
    }
    if n == 0:
        return stats

    strategy_total = float(np.prod(1.0 + r) - 1.0)
    benchmark_total = float(np.prod(1.0 + b) - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.cumprod((1.0 + r) / (1.0 + b))
        peak = np.maximum.accumulate(np.concatenate(([1.0], relative)))[1:]
        relative_drawdown = relative / peak - 1.0
    final = float(relative[-1])
    excess_max_drawdown = (
        float(min(np.nanmin(relative_drawdown), 0.0))
        if np.isfinite(relative_drawdown).any()
        else nan
    )
    stats.update({
        "Strategy Total Return [%]": strategy_total * 100.0,
        "Benchmark Total Return [%]": benchmark_total * 100.0,
        "Excess Return [%]": (final - 1.0) * 100.0,
        "Annualized Excess Return [%]": (
            float(final ** (bars_per_year / n) - 1.0) * 100.0 if final > 0 else nan
        ),
        "Total Return Difference [%]": (strategy_total - benchmark_total) * 100.0,
        "Excess Max Drawdown [%]": excess_max_drawdown * 100.0,
        "Win Rate vs Benchmark [%]": float(np.mean(r > b)) * 100.0,
    })
    if n < 2:
        return stats

    active = r - b
    tracking = float(np.std(active, ddof=1) * np.sqrt(bars_per_year))
    variance = float(np.var(b, ddof=1))
    stats["Tracking Error [%]"] = tracking * 100.0
    if tracking > 0:
        stats["Information Ratio"] = float(np.mean(active) * bars_per_year / tracking)
    if variance > 0:
        beta = float(np.cov(r, b, ddof=1)[0, 1] / variance)
        stats["Beta"] = beta
        stats["CAPM Alpha [%]"] = float(
            (np.mean(r) - beta * np.mean(b)) * bars_per_year * 100.0
        )
        if np.std(r) > 0:
            stats["Correlation"] = float(np.corrcoef(r, b)[0, 1])
    return stats


def win_rates(
    returns: xr.DataArray,
    fill_timestamps,
    *,
    ranges: Sequence[tuple[str, str]],
    benchmark_returns: xr.DataArray | None = None,
) -> dict:
    """Return the share of holding periods and of months the strategy won, in percent.

    A *holding period* runs from a bar with fills up to the bar before the
    next one; the bars before the first fill hold nothing and are left out.
    A month is a calendar month of the bar labels. Each period's per-bar
    returns inside ``ranges`` are compounded; the strategy wins a period
    when its return beats the benchmark's, or, without a benchmark, when it
    is positive. A bar where either return is NaN, or at or below -100% (a
    value that reached zero, whose log return does not exist), is left out,
    and a slice with no period gives NaN.

    Parameters
    ----------
    returns : xarray.DataArray
        The strategy's per-bar returns on a ``timestamp`` axis.
    fill_timestamps : array_like of datetime64
        The bar of every fill (repeats allowed); they mark the holding
        periods.
    ranges : sequence of (str, str)
        Inclusive bar-label ranges of the slice; bars outside them are left
        out.
    benchmark_returns : xarray.DataArray, optional
        The benchmark's per-bar returns on the same bars. Without it a
        period is won when its return is positive.

    Returns
    -------
    dict
        ``Rebalance Win Rate [%]`` and ``Monthly Win Rate [%]``, with
        `` vs Benchmark`` before `` [%]`` when ``benchmark_returns`` is given.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=4)
    >>> returns = xr.DataArray([0.0, 0.01, -0.02, 0.01], dims="timestamp", coords={"timestamp": bars})
    >>> win_rates(returns, bars[[0, 2]].values, ranges=[("2024-01-01", "2024-01-04")])
    {'Rebalance Win Rate [%]': 50.0, 'Monthly Win Rate [%]': 0.0}
    """
    ts = returns.timestamp.values.astype("datetime64[ns]")
    r = np.asarray(returns.values, dtype=np.float64)
    b = (
        np.asarray(benchmark_returns.values, dtype=np.float64)
        if benchmark_returns is not None
        else np.zeros_like(r)
    )
    fills = np.unique(np.asarray(fill_timestamps).astype("datetime64[ns]"))
    period = np.searchsorted(fills, ts, side="right") - 1
    with np.errstate(invalid="ignore"):
        valid = (r > -1.0) & (b > -1.0)
    keep = in_ranges(ts, ranges) & (period >= 0) & valid
    months = pd.DatetimeIndex(ts).to_period("M").asi8

    def share(keys: np.ndarray) -> float:
        """Return the percentage of the kept bars' groups under ``keys`` won."""
        if not keep.any():
            return float("nan")
        frame = pd.DataFrame({"r": np.log1p(r[keep]), "b": np.log1p(b[keep]), "k": keys[keep]})
        sums = frame.groupby("k")[["r", "b"]].sum()
        return float((sums["r"] > sums["b"]).mean() * 100.0)

    suffix = " vs Benchmark" if benchmark_returns is not None else ""
    return {
        f"Rebalance Win Rate{suffix} [%]": share(period),
        f"Monthly Win Rate{suffix} [%]": share(months),
    }


def turnover(orders: xr.Dataset, value: xr.DataArray, init_cash: float) -> xr.DataArray:
    """Return the turnover of every bar that had fills, on a ``timestamp`` axis.

    Turnover is the one-sided traded notional of the bar (the sum of
    ``|size| x price`` over its orders) divided by the portfolio value of
    the previous bar, or ``init_cash`` for the first bar of the window. One
    sided means a full buy-in from cash is about 1 and replacing the whole
    book (sell then buy) about 2. Using the value before the fills keeps the
    bar's own profit or loss out of the ratio.

    Parameters
    ----------
    orders : xarray.Dataset
        One fill per entry of an ``order`` dimension, with the variables
        ``timestamp``, ``size`` (signed or not) and ``price``; a dataset
        without an ``order`` dimension means no fills.
    value : xarray.DataArray
        Portfolio value after each bar, on a ``timestamp`` axis that holds
        every order's bar.
    init_cash : float
        Value before the first bar.

    Returns
    -------
    xarray.DataArray
        One entry per bar with fills; empty when there are none.

    Raises
    ------
    ValueError
        If an order timestamp is not on the value axis.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=3)
    >>> value = xr.DataArray([1000.0, 1100.0, 1100.0], dims="timestamp", coords={"timestamp": bars})
    >>> orders = xr.Dataset({
    ...     "timestamp": ("order", bars[[1, 2, 2]].values),
    ...     "size": ("order", [100.0, -50.0, 40.0]),
    ...     "price": ("order", [10.0, 11.0, 11.0]),
    ... })
    >>> turnover(orders, value, init_cash=1000.0).values.tolist()
    [1.0, 0.9]
    """
    if orders.sizes.get("order", 0) == 0:
        return xr.DataArray(
            np.array([], dtype=np.float64),
            dims=("timestamp",),
            coords={"timestamp": np.array([], dtype="datetime64[ns]")},
        )

    order_ts = orders["timestamp"].values.astype("datetime64[ns]")
    notional = np.abs(orders["size"].values.astype(np.float64)) * orders[
        "price"
    ].values.astype(np.float64)
    fill_bars, inverse = np.unique(order_ts, return_inverse=True)
    traded = np.zeros(fill_bars.size, dtype=np.float64)
    np.add.at(traded, inverse, notional)

    value_ts = value.timestamp.values.astype("datetime64[ns]")
    idx = np.searchsorted(value_ts, fill_bars)
    if (idx >= value_ts.size).any() or not np.array_equal(
        value_ts[np.minimum(idx, value_ts.size - 1)], fill_bars
    ):
        raise ValueError("an order timestamp is not on the equity timestamp axis")
    values = np.asarray(value.values, dtype=np.float64)
    previous = np.where(idx > 0, values[np.maximum(idx - 1, 0)], float(init_cash))
    return xr.DataArray(
        traded / previous, dims=("timestamp",), coords={"timestamp": fill_bars}
    )


def turnover_stats(
    turnover: xr.DataArray,
    *,
    bar_interval,
    year_freq: pd.Timedelta,
    rebalance_periods: int,
) -> dict:
    """Summarize per-fill-bar turnover as the three percent-valued metric rows.

    ``Turnover per Rebalance [%]`` is the mean over the fill bars, ``Total
    Turnover [%]`` their sum and ``Annualized Turnover [%]`` the mean times
    bars per year (``year_freq / bar_interval``) divided by
    ``rebalance_periods``. With no fills the mean and the annualized value
    are NaN and the total is 0.

    Parameters
    ----------
    turnover : xarray.DataArray
        ``turnover``'s output, possibly cut to a slice.
    bar_interval
        The bar spacing, anything ``pd.Timedelta`` accepts.
    year_freq : pd.Timedelta
        One year, as ``year_freq`` returns it.
    rebalance_periods : int
        Bars between rebalances.

    Returns
    -------
    dict
        The three rows.

    Examples
    --------
    >>> flows = xr.DataArray([1.0, 0.5], dims="timestamp")
    >>> stats = turnover_stats(
    ...     flows, bar_interval="1D", year_freq=year_freq("1D", 252, 390), rebalance_periods=5
    ... )
    >>> {key: round(value, 6) for key, value in stats.items()}
    {'Turnover per Rebalance [%]': 75.0, 'Total Turnover [%]': 150.0, 'Annualized Turnover [%]': 3780.0}
    """
    values = np.asarray(turnover.values, dtype=np.float64)
    interval = pd.Timedelta(bar_interval)
    bars_per_year = pd.Timedelta(year_freq) / interval
    mean = float(values.mean()) if values.size else float("nan")
    return {
        "Turnover per Rebalance [%]": mean * 100.0,
        "Total Turnover [%]": float(values.sum()) * 100.0,
        "Annualized Turnover [%]": mean * bars_per_year / rebalance_periods * 100.0,
    }
