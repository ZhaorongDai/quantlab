"""A factor risk model's forecast at a bar: ``FactorRiskForecast`` and ``FactorRiskModel.forecast``.

What is locked here, and what turns it red:

- A forecast is ``B F B' + diag(D)`` over the symbols it covers: its dense
  covariance and variances, a book's net exposures, its factor and specific
  variance and its x-sigma-rho contributions all agree with the dense
  matrix, for one book or several at once; ``scaled`` multiplies ``F`` and
  ``D``, ``subset`` keeps rows.
- ``FactorRiskModel.forecast`` covers a symbol only with every exposure
  (``exposure_matrix``), a specific risk at the bar, and no exposure to a
  factor the bar has no covariance for (``covered_factors``); such a factor
  leaves the forecast's ``factor_names``. It is the one coverage rule of the
  portfolio's estimator, factor attribution and bias statistics.

The model is the planted USE4 model of ``tests/test_risk_estimate.py``.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.risk.base import FactorRiskForecast
from tests.test_risk_estimate import BAR, built  # noqa: F401 - the fixture
from tests.test_risk_regression import _SYMBOLS, _STYLES, _day, _plant

SYMBOLS = np.array(["AAA", "BBB", "CCC"])
B = np.array([[1.0, 0.5], [1.0, -1.0], [1.0, 0.0]])
F = np.array([[0.04, 0.01], [0.01, 0.09]])
D = np.array([0.01, 0.02, 0.03])


def _forecast() -> FactorRiskForecast:
    return FactorRiskForecast(
        symbols=SYMBOLS, factor_names=("market", "value"), exposures=B,
        factor_covariance=F, specific_variance=D,
    )


def test_the_forecast_is_b_f_b_plus_d():
    forecast = _forecast()
    dense = B @ F @ B.T + np.diag(D)
    np.testing.assert_allclose(forecast.covariance, dense)
    np.testing.assert_allclose(forecast.variance, np.diag(dense))
    assert [part.shape for part in forecast.factor_form()] == [(3, 2), (2, 2), (3,)]


def test_a_books_risk_agrees_with_the_dense_matrix():
    forecast = _forecast()
    w = np.array([0.5, -0.2, 0.3])
    np.testing.assert_allclose(forecast.exposure(w), w @ B)
    factor, specific = forecast.portfolio_variance(w)
    np.testing.assert_allclose(factor + specific, w @ forecast.covariance @ w)
    np.testing.assert_allclose(specific, (w**2) @ D)
    by_factor, by_specific = forecast.risk_contributions(w)
    np.testing.assert_allclose(by_factor.sum() + by_specific, np.sqrt(factor + specific))
    x = w @ B
    np.testing.assert_allclose(by_factor, x * (F @ x) / np.sqrt(factor + specific))
    # Several books at once.
    books = np.stack([w, -w, np.zeros(3)])
    factors, specifics = forecast.portfolio_variance(books)
    np.testing.assert_allclose(factors + specifics, [w @ forecast.covariance @ w] * 2 + [0.0])
    # A book without risk has no contributions.
    assert np.isnan(np.concatenate([*forecast.risk_contributions(np.zeros(3))[:1], [np.nan]])).all()


def test_scaled_and_subset():
    forecast = _forecast()
    np.testing.assert_allclose(forecast.scaled(5).covariance, 5 * forecast.covariance)
    part = forecast.subset(np.array([2, 0]))
    assert part.symbols.tolist() == ["CCC", "AAA"]
    assert part.factor_names == ("market", "value")
    np.testing.assert_allclose(part.covariance, forecast.covariance[np.ix_([2, 0], [2, 0])])


def _at(bar: int):
    _, exposures, _, _ = _plant()
    return exposures.isel(timestamp=bar, drop=True).copy(deep=True)


def _row(model, bar: int):
    return model.estimate.read(_day(bar), _day(bar)).isel(timestamp=0).load()


def test_the_model_covers_symbols_with_exposures_and_a_specific_risk(built):  # noqa: F811
    exposures = _at(BAR)
    exposures[_STYLES[0]].loc[{"symbol": _SYMBOLS[3]}] = np.nan
    row = _row(built, BAR).copy(deep=True)
    row["specific_risk"].loc[{"symbol": _SYMBOLS[5]}] = np.nan

    forecast = built.forecast(row, exposures)

    # Symbol 3 lost a style, 5 its specific risk; the last has an unknown industry.
    assert forecast.symbols.tolist() == [
        s for s in _SYMBOLS[:-1] if s not in (_SYMBOLS[3], _SYMBOLS[5])
    ]
    assert forecast.factor_names == tuple(built.factor_names)
    np.testing.assert_array_equal(
        forecast.factor_covariance, row["factor_covariance"].values
    )
    np.testing.assert_array_equal(
        forecast.specific_variance, row["specific_risk"].sel(symbol=forecast.symbols).values ** 2
    )
    matrix, _ = built.exposure_matrix(exposures.sel(symbol=forecast.symbols))
    np.testing.assert_array_equal(forecast.exposures, matrix)


def test_a_factor_without_a_covariance_leaves_with_its_symbols(built):  # noqa: F811
    # Bar 28: industry 4 has too few returns in its volatility window.
    forecast = built.forecast(_row(built, 28), _at(28))
    assert "industry_4" not in forecast.factor_names
    assert len(forecast.factor_names) == len(built.factor_names) - 1
    assert np.isfinite(forecast.factor_covariance).all()
    assert not set(_SYMBOLS[54:60]) & set(forecast.symbols.tolist())
    assert len(forecast.symbols) == 54


def test_the_store_estimator_returns_the_models_forecast(built):  # noqa: F811
    from tests.test_risk_estimate import _context
    from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig
    from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator

    context = _context(built)
    estimate = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=built)).estimate(context)
    expected = built.forecast(_row(built, BAR), context.risk_exposures)
    assert isinstance(estimate, FactorRiskForecast)
    assert estimate.symbols.tolist() == expected.symbols.tolist()
    np.testing.assert_array_equal(estimate.covariance, expected.covariance)
    assert pd.Timestamp(context.timestamp) == pd.Timestamp(_day(BAR))


def test_a_forecast_over_no_symbol_is_empty(built):  # noqa: F811
    exposures = _at(BAR).isel(symbol=[])
    forecast = built.forecast(_row(built, BAR), exposures)
    assert forecast.symbols.size == 0 and forecast.exposures.shape == (0, len(built.factor_names))
    with pytest.raises(ValueError, match="symbols"):
        forecast.portfolio_variance(np.zeros(3))
