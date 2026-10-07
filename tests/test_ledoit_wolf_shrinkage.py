"""``LedoitWolfEstimator``'s estimate is the Ledoit-Wolf shrunk sample covariance of its covered window.

Checked against the shrinkage written out by hand (Ledoit and Wolf, 2004),
through the estimator: the covered symbols' window in, their shrunk
covariance out, with given volatilities replacing the shrunk ones and the
correlations kept.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.portfolio.base import PortfolioContext
from quantlab.portfolio.config import LedoitWolfEstimatorConfig
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


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


def _context(window: np.ndarray) -> PortfolioContext:
    symbols = SYMBOLS[: window.shape[1]]
    on_symbol = {"dims": "symbol", "coords": {"symbol": symbols}}
    return PortfolioContext(
        timestamp=pd.Timestamp("2024-03-25"),
        predictions=xr.Dataset(coords={"symbol": symbols}),
        tradable=xr.DataArray([True] * len(symbols), **on_symbol),
        current_weights=xr.DataArray(np.zeros(len(symbols)), **on_symbol),
        returns=xr.DataArray(window, dims=("timestamp", "symbol"), coords={
            "timestamp": pd.bdate_range("2024-01-01", periods=len(window)), "symbol": symbols}),
    )


def _estimator(window: np.ndarray) -> LedoitWolfEstimator:
    return LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=len(window)))


def test_the_estimate_is_the_ledoit_wolf_shrunk_sample_covariance():
    window = _window()
    estimate = _estimator(window).estimate(_context(window))
    np.testing.assert_allclose(estimate.covariance, _by_hand(window), rtol=1e-10)


def test_given_volatilities_replace_the_shrunk_ones_and_keep_the_correlations():
    window = _window()
    shrunk = _by_hand(window)
    sd = np.sqrt(np.diag(shrunk))
    given = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    volatility = xr.DataArray(given, dims="symbol", coords={"symbol": SYMBOLS})

    covariance = _estimator(window).estimate(_context(window), volatility).covariance

    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), given, rtol=1e-12)
    np.testing.assert_allclose(
        covariance / np.outer(given, given), shrunk / np.outer(sd, sd), rtol=1e-10
    )


def test_the_estimate_is_the_shrinkage_of_the_covered_symbols_window_only():
    window = _window(bars=30, symbols=4)
    window[:3, 1] = np.nan  # listed three bars into the window: not covered

    estimate = _estimator(window).estimate(_context(window))

    assert estimate.symbols.tolist() == ["AAA", "CCC", "DDD"]
    np.testing.assert_allclose(estimate.covariance, _by_hand(window[:, [0, 2, 3]]), rtol=1e-10)
