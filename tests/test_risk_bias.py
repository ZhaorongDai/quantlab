"""Bias statistics of a factor risk model's forecasts (#196).

``bias_statistics`` is checked on synthetic forecasts: returns drawn with
exactly the forecast volatility give a bias near 1, forecasts at half the
true volatility a bias near 2, and the rolling statistics equal the standard
deviation of each window's standardized outcomes written here directly.

``risk_model_bias_statistics`` is checked end to end on the estimate store of
``tests/test_risk_estimate.py``: the factor and specific outcomes are the
next bar's factor and specific returns over the forecast volatility of the
bar, read back from the stores; random active portfolios are reproducible
from their seed and vanish when every portfolio is the benchmark.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.risk.bias import bias_statistics, risk_model_bias_statistics
from tests.test_risk_estimate import START, _risk_model
from tests.test_risk_regression import _T, _day

_BARS = pd.bdate_range("2000-01-03", periods=2000).values


def _forecasts(n_items: int = 40, seed: int = 196) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ``(realized, forecast)``: returns drawn with the forecast volatility."""
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.005, 0.03, size=(len(_BARS), n_items))
    realized = sigma * rng.standard_normal(sigma.shape)
    coords = {"timestamp": _BARS, "portfolio": np.arange(n_items)}
    dims = ("timestamp", "portfolio")
    return xr.DataArray(realized, coords, dims), xr.DataArray(sigma, coords, dims)


def test_exact_forecasts_give_a_bias_near_one():
    realized, forecast = _forecasts()
    stats = bias_statistics(realized, forecast)
    band = np.sqrt(2.0 / len(_BARS))
    np.testing.assert_allclose(stats["band"].values, band)
    assert (stats["count"] == len(_BARS)).all()
    assert abs(float(stats["bias"].mean()) - 1.0) < band / 2
    # Roughly 95 percent inside 1 +/- sqrt(2/T) for normal returns.
    inside = (abs(stats["bias"] - 1.0) <= stats["band"]).mean()
    assert float(inside) >= 0.85


def test_forecasts_at_half_the_volatility_give_a_bias_near_two():
    realized, forecast = _forecasts()
    stats = bias_statistics(realized, forecast / 2.0)
    assert abs(float(stats["bias"].mean()) - 2.0) < 2 * np.sqrt(2.0 / len(_BARS))


def test_rolling_bias_is_each_windows_standard_deviation():
    realized, forecast = _forecasts(n_items=5)
    realized[100:130, 2] = np.nan  # a gap: fewer outcomes in its windows
    stats = bias_statistics(realized, forecast, window=60, min_observations=40)
    outcome = (realized / forecast).values
    np.testing.assert_allclose(stats["outcome"].values, outcome)
    rolling = stats["rolling_bias"].values
    count = stats["rolling_count"].values
    for t in (38, 39, 59, 60, 120, 135, 159, 1999):
        window = outcome[max(t - 59, 0) : t + 1]
        for item in range(5):
            values = window[:, item][np.isfinite(window[:, item])]
            assert count[t, item] == len(values)
            if len(values) < 40:
                assert np.isnan(rolling[t, item])
            else:
                np.testing.assert_allclose(rolling[t, item], values.std(ddof=1), rtol=1e-9)
    np.testing.assert_allclose(stats["rolling_band"].values, np.sqrt(2.0 / count))


def test_rolling_summaries_across_portfolios():
    realized, forecast = _forecasts()
    stats = bias_statistics(realized, forecast, window=252)
    rolling = stats["rolling_bias"].values
    t = 500
    np.testing.assert_allclose(stats["rolling_mean"].values[t], rolling[t].mean())
    np.testing.assert_allclose(stats["rolling_mrad"].values[t], np.abs(rolling[t] - 1).mean())
    np.testing.assert_allclose(stats["rolling_p5"].values[t], np.percentile(rolling[t], 5))
    np.testing.assert_allclose(stats["rolling_p95"].values[t], np.percentile(rolling[t], 95))
    # Perfect normal forecasts: the mean rolling 12-month bias is about 1 and
    # the MRAD about 0.17 for monthly windows; here sqrt(2/252)-scale.
    assert abs(float(stats["rolling_mean"][300:].mean()) - 1.0) < 0.02
    assert np.isnan(stats["rolling_mean"].values[: 252 // 2 - 1]).all()


def test_a_zero_or_missing_forecast_has_no_outcome():
    realized, forecast = _forecasts(n_items=3)
    forecast[5, 0] = 0.0
    forecast[6, 1] = np.nan
    stats = bias_statistics(realized, forecast)
    assert np.isnan(stats["outcome"].values[5, 0]) and np.isnan(stats["outcome"].values[6, 1])
    assert stats["count"].values.tolist() == [len(_BARS) - 1, len(_BARS) - 1, len(_BARS)]


def test_bias_statistics_rejects_a_bad_window():
    realized, forecast = _forecasts(n_items=2)
    with pytest.raises(ValueError, match="window"):
        bias_statistics(realized, forecast, window=1)
    with pytest.raises(ValueError, match="min_observations"):
        bias_statistics(realized, forecast, window=10, min_observations=11)


# ----------------------------------------------------------------------
# On a factor risk model's stores
# ----------------------------------------------------------------------

FIRST = 10  # the first estimate bar the statistics read


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    model = _risk_model(tmp_path_factory.mktemp("bias"))
    model.regression.build(_day(START), _day(_T - 1))
    model.estimate.build(_day(FIRST), _day(_T - 1))
    return model


@pytest.fixture(scope="module")
def stats(model):
    return risk_model_bias_statistics(
        model, _day(FIRST), _day(_T - 1), window=8, random_portfolios=4, portfolio_size=10
    )


def test_factor_outcomes_are_next_bar_returns_over_the_forecast(model, stats):
    estimate = model.estimate.read(_day(FIRST), _day(_T - 1))
    regression = model.regression.read(_day(FIRST), _day(_T - 1))
    factor = stats["factor"]
    # The last estimate bar has no next bar.
    assert factor["timestamp"].values.tolist() == estimate["timestamp"].values[:-1].tolist()
    variance = np.diagonal(estimate["factor_covariance"].values, axis1=1, axis2=2)[:-1]
    expected = regression["factor_return"].values[1:] / np.sqrt(variance)
    np.testing.assert_allclose(factor["outcome"].values, expected, rtol=1e-12)
    assert factor["outcome"].dims == ("timestamp", "factor")
    assert factor["rolling_bias"].dims == ("timestamp", "factor")


def test_specific_outcomes_are_next_bar_specific_returns_over_the_risk(model, stats):
    estimate = model.estimate.read(_day(FIRST), _day(_T - 1))
    regression = model.regression.read(_day(FIRST), _day(_T - 1))
    specific = stats["specific"]
    symbols = specific["symbol"].values
    risk = estimate["specific_risk"].sel(symbol=symbols).values[:-1]
    returns = regression["specific_return"].sel(symbol=symbols).values[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.where(risk > 0, returns / risk, np.nan)
    np.testing.assert_allclose(specific["outcome"].values, expected, rtol=1e-12)
    # Symbols outside the universe carry specific returns, so some outcomes exist.
    assert np.isfinite(specific["outcome"].values).sum() > 0


def test_random_portfolios_are_reproducible_from_the_seed(model, stats):
    random = stats["random"]
    assert random["outcome"].dims == ("timestamp", "portfolio")
    assert random.sizes["portfolio"] == 4
    assert np.isfinite(random["outcome"].values).all()
    again = risk_model_bias_statistics(
        model, _day(FIRST), _day(_T - 1), window=8, random_portfolios=4, portfolio_size=10
    )["random"]
    xr.testing.assert_identical(random, again)
    other = risk_model_bias_statistics(
        model, _day(FIRST), _day(_T - 1), window=8, random_portfolios=4, portfolio_size=10,
        seed=1,
    )["random"]
    assert not np.allclose(random["outcome"].values, other["outcome"].values)


def test_a_portfolio_of_the_whole_universe_has_no_active_risk(model):
    random = risk_model_bias_statistics(
        model, _day(FIRST), _day(_T - 1), window=8, random_portfolios=2, portfolio_size=1000
    )["random"]
    assert np.isnan(random["outcome"].values).all()
    np.testing.assert_allclose(random["realized"].values, 0.0, atol=1e-15)


# ----------------------------------------------------------------------
# Over a horizon
# ----------------------------------------------------------------------

HORIZON = 3


@pytest.fixture(scope="module")
def horizon_stats(model):
    return risk_model_bias_statistics(
        model, _day(FIRST), _day(_T - 1), horizon=HORIZON, window=4, random_portfolios=4,
        portfolio_size=10,
    )


def test_a_horizon_sums_the_next_bars_against_sqrt_h_times_the_forecast(model, horizon_stats):
    estimate = model.estimate.read(_day(FIRST), _day(_T - 1))
    regression = model.regression.read(_day(FIRST), _day(_T - 1))
    bars = estimate["timestamp"].values
    # A forecast every HORIZON bars that has HORIZON bars after it.
    picks = np.arange(0, len(bars) - HORIZON, HORIZON)
    factor = horizon_stats["factor"]
    assert factor["timestamp"].values.tolist() == bars[picks].tolist()
    variance = np.diagonal(estimate["factor_covariance"].values, axis1=1, axis2=2)[picks]
    returns = regression["factor_return"].values
    summed = np.stack([returns[p + 1 : p + 1 + HORIZON].sum(axis=0) for p in picks])
    np.testing.assert_allclose(
        factor["outcome"].values, summed / np.sqrt(HORIZON * variance), rtol=1e-12
    )

    specific = horizon_stats["specific"]
    symbols = specific["symbol"].values
    risk = estimate["specific_risk"].sel(symbol=symbols).values[picks]
    u = regression["specific_return"].sel(symbol=symbols).values
    summed = np.stack([u[p + 1 : p + 1 + HORIZON].sum(axis=0) for p in picks])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.where(risk > 0, summed / (np.sqrt(HORIZON) * risk), np.nan)
    np.testing.assert_allclose(specific["outcome"].values, expected, rtol=1e-12)


def test_random_portfolios_over_a_horizon(model, horizon_stats):
    random = horizon_stats["random"]
    assert random.sizes["timestamp"] == horizon_stats["factor"].sizes["timestamp"]
    assert np.isfinite(random["outcome"].values).all()
    whole = risk_model_bias_statistics(
        model, _day(FIRST), _day(_T - 1), horizon=HORIZON, random_portfolios=2,
        portfolio_size=1000,
    )["random"]
    np.testing.assert_allclose(whole["realized"].values, 0.0, atol=1e-15)


def test_a_horizon_of_one_is_the_one_bar_test(model, stats):
    one = risk_model_bias_statistics(
        model, _day(FIRST), _day(_T - 1), horizon=1, window=8, random_portfolios=4,
        portfolio_size=10,
    )
    for group in ("factor", "specific", "random"):
        xr.testing.assert_allclose(one[group], stats[group], rtol=1e-12)


def test_the_horizon_must_be_positive(model):
    with pytest.raises(ValueError, match="horizon"):
        risk_model_bias_statistics(model, _day(FIRST), _day(_T - 1), horizon=0)


def test_eigenfactor_outcomes_are_next_bar_eigenfactor_returns_over_their_volatility(model, stats):
    estimate = model.estimate.read(_day(FIRST), _day(_T - 1))
    regression = model.regression.read(_day(FIRST), _day(_T - 1))
    eigen = stats["eigenfactor"]
    assert eigen["outcome"].dims == ("timestamp", "eigenfactor")
    assert eigen["eigenfactor"].values.tolist() == list(range(1, len(model.factor_names) + 1))
    covariance = estimate["factor_covariance"].values
    returns = regression["factor_return"].values
    for t in range(eigen.sizes["timestamp"]):
        kept = np.isfinite(np.diag(covariance[t]))  # no pair lacks a correlation here
        block = covariance[t][np.ix_(kept, kept)]
        values, vectors = np.linalg.eigh(block)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.where(values > 0, returns[t + 1, kept] @ vectors / np.sqrt(values), np.nan)
        np.testing.assert_allclose(
            eigen["outcome"].values[t, : kept.sum()], expected, rtol=1e-9, err_msg=f"bar {t}"
        )
        assert np.isnan(eigen["outcome"].values[t, kept.sum():]).all()
    assert np.isfinite(eigen["outcome"].values).sum() > 0
