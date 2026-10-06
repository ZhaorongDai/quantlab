"""Benchmark beta: each symbol's rolling OLS beta of one-bar returns on a single-symbol benchmark.

The beta of a symbol at bar t is the slope of the ordinary least-squares
regression, with an intercept, of its one-bar returns on the benchmark's over
the ``lookback_bars`` returns ending at t. It is the exposure a portfolio rule
holds near 1 to keep a book's market risk at the benchmark's
(``MeanVarianceConfig.exposure_bounds``).
"""

import numpy as np
import xarray as xr

from quantlab.factor.base import Factor
from quantlab.factor.config import BenchmarkBetaConfig
from quantlab.utils.returns import one_bar_returns

#: The factor's one output.
OUTPUT = "beta"


class BenchmarkBeta(Factor):
    """Each symbol's rolling OLS beta of one-bar returns on ``config.benchmark``.

    A one-bar return is ``price[t] / price[t - 1] - 1`` on ``price_column``,
    NaN when either price is missing, for the symbols and the benchmark alike;
    the benchmark is read over the same bars and placed on the symbols'
    timestamps. The beta at t uses the ``lookback_bars`` returns ending at t
    where both are present, and is NaN when fewer than ``min_bars`` are. It
    reads nothing after t. The panel is float64, one variable, ``beta``.

    Parameters
    ----------
    config : BenchmarkBetaConfig
        The factor config: the symbols' ``dataset``, the ``benchmark`` and
        the window.

    Raises
    ------
    ValueError
        At construction, if ``min_bars`` is below 2 or above
        ``lookback_bars``, or ``warmup_bars`` is below ``lookback_bars``; on
        ``compute``, if the benchmark panel does not hold exactly one symbol.

    Examples
    --------
    With ``stocks`` a price dataset and ``vt`` the single-symbol VT store:

    >>> beta = BenchmarkBeta(BenchmarkBetaConfig(
    ...     dataset=stocks, benchmark=vt, file_path="factor/beta_vt.zarr",
    ... ))
    >>> beta.get_factor_names(), beta.warmup_bars
    (('beta',), 252)
    >>> list(beta.compute("2024-01-02", "2024-12-31").data_vars)
    ['beta']
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = BenchmarkBetaConfig

    # Narrower type annotation for readers and type checkers only.
    config: BenchmarkBetaConfig

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``("beta",)``."""
        return (OUTPUT,)

    def _validate_config(self) -> None:
        """Refuse ``min_bars`` outside ``[2, lookback_bars]`` and a warm-up shorter than the window.

        Raises
        ------
        ValueError
            If ``min_bars`` is below 2 or above ``lookback_bars``, or
            ``warmup_bars`` is below ``lookback_bars``.
        """
        if not 2 <= self.config.min_bars <= self.config.lookback_bars:
            raise ValueError(
                f"{self.class_name}: config.min_bars must be at least 2 and at most "
                f"lookback_bars={self.config.lookback_bars}, got {self.config.min_bars}."
            )
        if self.config.warmup_bars < self.config.lookback_bars:
            raise ValueError(
                f"{self.class_name}: config.warmup_bars={self.config.warmup_bars} is below "
                f"lookback_bars={self.config.lookback_bars}; the first bars would use "
                f"short windows."
            )

    def _input_variables(self) -> list[str]:
        """Read only the price column."""
        return self.config.dataset.own_names([self.config.price_column])

    def _compute_panel(self, inputs: xr.Dataset) -> xr.Dataset:
        """Return the betas over every bar of ``inputs``, warm-up included.

        Raises
        ------
        ValueError
            If the benchmark panel over those bars does not hold exactly one
            symbol.
        """
        column = self.config.price_column
        prices = self.config.dataset.to_shared_names(inputs)[column].transpose("timestamp", "symbol")
        timestamps = prices["timestamp"].values
        benchmark = self.config.benchmark
        panel = benchmark.to_shared_names(benchmark.panel(
            timestamps[0], timestamps[-1], variables=benchmark.own_names([column])
        ))
        if panel.sizes.get("symbol") != 1:
            raise ValueError(
                f"{self.class_name}: the benchmark must hold exactly one symbol, "
                f"got {panel.sizes.get('symbol')} ({list(panel['symbol'].values)[:5]})."
            )
        market = panel[column].isel(symbol=0).reindex(timestamp=timestamps).values.astype(np.float64)
        beta = rolling_beta(
            one_bar_returns(prices.values.astype(np.float64)),
            one_bar_returns(market),
            self.config.lookback_bars,
            self.config.min_bars,
        )
        return xr.Dataset(
            {OUTPUT: (("timestamp", "symbol"), beta)},
            coords={"timestamp": timestamps, "symbol": prices["symbol"].values},
        )


def rolling_beta(returns, market, lookback: int, min_bars: int) -> np.ndarray:
    """Return the rolling OLS slope of each column of ``returns`` on ``market``.

    Each window is the ``lookback`` rows ending at a row; only the rows where
    both the column's return and ``market`` are finite count, and a window
    with fewer than ``min_bars`` of them, or with a constant ``market``, is
    NaN.

    Parameters
    ----------
    returns : array-like
        Returns on ``[T, S]``.
    market : array-like
        The benchmark's returns on ``[T]``.
    lookback : int
        Rows per window.
    min_bars : int
        Fewest rows a window needs.

    Returns
    -------
    np.ndarray
        The slopes on ``[T, S]``.

    Examples
    --------
    >>> market = np.array([0.01, -0.02, 0.03, 0.0])
    >>> rolling_beta(np.column_stack([2 * market, -market]), market, lookback=3, min_bars=2).round(6)
    array([[nan, nan],
           [ 2., -1.],
           [ 2., -1.],
           [ 2., -1.]])
    """
    y = np.asarray(returns, dtype=np.float64)
    x = np.broadcast_to(np.asarray(market, dtype=np.float64)[:, None], y.shape)
    valid = np.isfinite(y) & np.isfinite(x)
    y0, x0 = np.where(valid, y, 0.0), np.where(valid, x, 0.0)

    def window_sum(values: np.ndarray) -> np.ndarray:
        total = np.cumsum(values, axis=0)
        total[lookback:] = total[lookback:] - total[:-lookback]
        return total

    n = window_sum(valid.astype(np.float64))
    sx, sy = window_sum(x0), window_sum(y0)
    sxx, sxy = window_sum(x0 * x0), window_sum(x0 * y0)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = sxx - sx * sx / n
        cov = sxy - sx * sy / n
        beta = cov / var
    # A constant benchmark leaves only the round-off of the running sums in
    # ``var``: below 1e-12 of the window's sum of squares it counts as 0. The
    # 1e-300 floor keeps an all-zero window (sum of squares 0) at 0 too.
    return np.where((n >= min_bars) & (var > 1e-12 * np.maximum(sxx, 1e-300)), beta, np.nan)
