"""``ledoit_wolf_covariance``: the Ledoit-Wolf estimate of a window of returns, with no context.

The function ``LedoitWolfEstimator`` calls on the covered symbols' window,
so the same estimate can be evaluated outside a backtest (bias statistics
of the optimiser's holdings, for example). Checked against the shrinkage
written out by hand, and against the estimator on the same window.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.base import PortfolioContext
from quantlab.portfolio.config import LedoitWolfEstimatorConfig
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.risk.predefined.ledoit_wolf import ledoit_wolf_covariance


def _window(bars: int = 40, symbols: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.01, size=(bars, 1))
    return common + rng.normal(0.0, 0.02, size=(bars, symbols)) * np.linspace(0.5, 1.5, symbols)


def _by_hand(window: np.ndarray) -> np.ndarray:
    """Ledoit and Wolf (2004): ``(1 - d) S + d mu I``, ``S`` the demeaned sample covariance (1/T)."""
    x = window - window.mean(axis=0)
    t, n = x.shape
    sample = x.T @ x / t
    mu = np.trace(sample) / n
    target = mu * np.eye(n)
    delta = ((sample - target) ** 2).sum() / n
    beta = sum(((np.outer(row, row) - sample) ** 2).sum() for row in x) / n / t**2
    shrinkage = min(beta, delta) / delta
    return (1 - shrinkage) * sample + shrinkage * target


def test_the_estimate_is_the_ledoit_wolf_shrunk_sample_covariance():
    window = _window()
    np.testing.assert_allclose(ledoit_wolf_covariance(window), _by_hand(window), rtol=1e-10)


def test_given_volatilities_replace_the_shrunk_ones_and_keep_the_correlations():
    window = _window()
    shrunk = _by_hand(window)
    sd = np.sqrt(np.diag(shrunk))
    given = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    covariance = ledoit_wolf_covariance(window, volatility=given)

    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), given, rtol=1e-12)
    np.testing.assert_allclose(
        covariance / np.outer(given, given), shrunk / np.outer(sd, sd), rtol=1e-10
    )


@pytest.mark.parametrize(
    "window, volatility",
    [
        (np.full((10, 2), np.nan), None),
        (_window()[:1], None),
        (_window(), np.ones(4)),
        (_window(), np.array([0.01, 0.0, 0.01, 0.01, 0.01])),
    ],
    ids=["not finite", "one bar", "volatility length", "volatility not positive"],
)
def test_a_window_it_cannot_estimate_from_is_refused(window, volatility):
    with pytest.raises(ValueError):
        ledoit_wolf_covariance(window, volatility=volatility)


def test_the_estimator_returns_the_function_of_its_covered_symbols_window():
    window = _window(bars=30, symbols=4)
    window[:3, 1] = np.nan  # listed three bars into the window: not covered
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    context = PortfolioContext(
        timestamp=pd.Timestamp("2024-03-25"),
        predictions=xr.Dataset(coords={"symbol": symbols}),
        tradable=xr.DataArray([True] * 4, dims="symbol", coords={"symbol": symbols}),
        current_weights=xr.DataArray(np.zeros(4), dims="symbol", coords={"symbol": symbols}),
        returns=xr.DataArray(window, dims=("timestamp", "symbol"), coords={
            "timestamp": pd.bdate_range("2024-01-01", periods=30), "symbol": symbols}),
    )

    estimate = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=30)).estimate(context)

    assert estimate.symbols.tolist() == ["AAA", "CCC", "DDD"]
    np.testing.assert_array_equal(
        estimate.covariance, ledoit_wolf_covariance(window[:, [0, 2, 3]])
    )
