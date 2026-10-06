"""The Newey-West adjustment of a factor risk model's estimate store (#197).

The planted panel of ``tests/test_risk_regression.py`` with AR(1) country and
style factor returns gives a regression store with serially correlated
factor returns. The estimate store built from it is checked against
Newey-West moments written here directly from the definition: the
exponentially weighted covariance of lag 0 over each pair's common bars,
plus Bartlett-weighted lagged covariances ``(1 - l / (L + 1)) (G_l + G_l')``
with each column about its weighted mean over the window; the specific
variance times ``1 + 2 sum_l (1 - l / (L + 1)) rho_l``. Small windows make the
truncation and the missing industry returns matter.
"""

import dataclasses

import numpy as np
import pytest
import xarray as xr

from tests.test_risk_estimate import START, WINDOWS, _risk_model
from tests.test_risk_regression import _STYLES, _T, _day, _plant

LAGS = dict(
    volatility_lags=2,
    correlation_lags=1,
    specific_lags=2,
    specific_autocorrelation_half_life=6.0,
    specific_autocorrelation_window=10,
)
LEAST = WINDOWS["min_observations"]


def _weights(length, half_life):
    return 0.5 ** (np.arange(length - 1, -1, -1) / half_life)


def _centred(x, half_life):
    """``x`` about its weighted mean over its present bars, NaN kept."""
    w = _weights(len(x), half_life)
    present = np.isfinite(x)
    return x - (w[present] * x[present]).sum() / w[present].sum()


def _lag_moment(x, y, lag, half_life):
    """Weighted mean of ``x(t) y(t - lag)`` (each centred) over the bars both have."""
    w = _weights(len(x), half_life)[lag:]
    a, b = _centred(x, half_life)[lag:], _centred(y, half_life)[: len(y) - lag]
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < LEAST:
        return 0.0
    return (w[both] * a[both] * b[both]).sum() / w[both].sum()


def _moment(x, y, half_life):
    """Lag-0 weighted covariance over the bars both have, about their own means."""
    w = _weights(len(x), half_life)
    both = np.isfinite(x) & np.isfinite(y)
    if both.sum() < LEAST:
        return np.nan
    w, x, y = w[both], x[both], y[both]
    mx, my = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    return (w * (x - mx) * (y - my)).sum() / w.sum()


def _bartlett(lags):
    return [1.0 - lag / (lags + 1) for lag in range(1, lags + 1)]


def _expected_covariance(history):
    vol = history[-WINDOWS["volatility_window"]:]
    cor = history[-WINDOWS["correlation_window"]:]
    k = history.shape[1]
    vol_h, cor_h = WINDOWS["volatility_half_life"], WINDOWS["correlation_half_life"]
    sigma = np.empty(k)
    for i in range(k):
        variance = _moment(vol[:, i], vol[:, i], vol_h) + 2 * sum(
            b * _lag_moment(vol[:, i], vol[:, i], lag, vol_h)
            for lag, b in enumerate(_bartlett(LAGS["volatility_lags"]), start=1)
        )
        sigma[i] = np.sqrt(max(variance, 0.0))

    def own_lags(x):
        return 2 * sum(
            b * _lag_moment(x, x, lag, cor_h)
            for lag, b in enumerate(_bartlett(LAGS["correlation_lags"]), start=1)
        )

    expected = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if i == j:
                rho = 1.0
            else:
                pair = np.isfinite(cor[:, i]) & np.isfinite(cor[:, j])
                xi, xj = np.where(pair, cor[:, i], np.nan), np.where(pair, cor[:, j], np.nan)
                # The variances over the pair's bars, plus each column's own lags.
                vi = _moment(xi, xi, cor_h) + own_lags(cor[:, i])
                vj = _moment(xj, xj, cor_h) + own_lags(cor[:, j])
                # Lag 0 over the pair's bars; the lags over each column whole,
                # centred over its own bars.
                rho = (
                    _moment(xi, xj, cor_h)
                    + sum(
                        b * (
                            _lag_moment(cor[:, i], cor[:, j], lag, cor_h)
                            + _lag_moment(cor[:, j], cor[:, i], lag, cor_h)
                        )
                        for lag, b in enumerate(_bartlett(LAGS["correlation_lags"]), start=1)
                    )
                ) / np.sqrt(vi * vj)
            expected[i, j] = rho * sigma[i] * sigma[j]
    return expected


def _expected_specific(history):
    window = history[-WINDOWS["specific_window"]:]
    auto = history[-LAGS["specific_autocorrelation_window"]:]
    auto_h = LAGS["specific_autocorrelation_half_life"]
    out = np.empty(history.shape[1])
    for n in range(history.shape[1]):
        variance = _moment(window[:, n], window[:, n], WINDOWS["specific_half_life"])
        u = auto[:, n]
        present = np.isfinite(u)
        c = _centred(u, auto_h)
        w = _weights(len(u), auto_h)
        gamma0 = (w[present] * c[present] ** 2).sum() / w[present].sum() if present.any() else 0.0
        multiplier = 1.0
        for lag, b in enumerate(_bartlett(LAGS["specific_lags"]), start=1):
            a, d = c[lag:], c[:-lag]
            both = np.isfinite(a) & np.isfinite(d)
            if both.sum() >= LEAST and gamma0 > 0:
                wl = w[lag:][both]
                multiplier += 2 * b * ((wl * a[both] * d[both]).sum() / wl.sum()) / gamma0
        out[n] = np.sqrt(variance * max(multiplier, 0.0))
    return out


@pytest.fixture(scope="module")
def planted():
    return _plant(autocorrelation=0.6)


@pytest.fixture(scope="module")
def built(tmp_path_factory, planted):
    model = _risk_model(tmp_path_factory.mktemp("newey_west"), planted=planted, **LAGS)
    model.regression.build(_day(START), _day(_T - 1))
    model.estimate.build(_day(START), _day(_T - 1))
    return model


def test_the_factor_covariance_matches_newey_west(built):
    returns = built.regression.read(_day(START), _day(_T - 1))["factor_return"].values
    got = built.estimate.read(_day(START), _day(_T - 1))["factor_covariance"].values
    for t in range(len(returns)):
        np.testing.assert_allclose(
            got[t], _expected_covariance(returns[: t + 1]), rtol=1e-9, atol=1e-15,
            err_msg=f"bar {t}",
        )


def test_the_specific_risk_matches_newey_west(built):
    regression = built.regression.read(_day(START), _day(_T - 1))
    specific = regression["specific_return"]
    rows = built.estimate.read(_day(START), _day(_T - 1))
    got = rows["specific_risk"].sel(symbol=specific["symbol"]).values
    values = specific.values
    for t in range(LEAST - 1, len(values)):
        np.testing.assert_allclose(
            got[t], _expected_specific(values[: t + 1]), rtol=1e-9, atol=1e-15,
            err_msg=f"bar {t}",
        )


def test_positive_autocorrelation_raises_the_factor_volatilities(tmp_path):
    # AR(1) at 0.8 on the country and styles: averaged over the last bars,
    # Newey-West variances exceed the raw ones (single bars are noisy with
    # windows this short).
    planted = _plant(autocorrelation=0.8)
    variances = []
    for name, lags in (("adjusted", LAGS), ("raw", {})):
        model = _risk_model(tmp_path, name, planted=planted, **lags)
        model.regression.build(_day(START), _day(_T - 1))
        model.estimate.build(_day(START), _day(_T - 1))
        covariance = model.estimate.read(_day(20), _day(_T - 1))["factor_covariance"]
        factors = ["country", *_STYLES]
        variances.append(np.stack(
            [covariance.sel(factor_i=f, factor_j=f).values for f in factors]
        ))
    assert float(np.mean(variances[0] / variances[1])) > 1.2


def test_zero_lags_reproduce_the_raw_store(tmp_path, planted):
    raw = _risk_model(tmp_path, "raw", planted=planted)
    zero = _risk_model(
        tmp_path, "zero", planted=planted,
        **{**LAGS, "volatility_lags": 0, "correlation_lags": 0, "specific_lags": 0},
    )
    for model in (raw, zero):
        model.regression.build(_day(START), _day(_T - 1))
        model.estimate.build(_day(15), _day(_T - 1))
    a = raw.estimate.read(_day(15), _day(_T - 1)).load()
    b = zero.estimate.read(_day(15), _day(_T - 1)).load()
    for name in a.data_vars:
        np.testing.assert_array_equal(a[name].values, b[name].values)


def test_build_then_extend_equals_build(tmp_path, planted):
    once = _risk_model(tmp_path, "once", planted=planted, **LAGS)
    once.regression.build(_day(START), _day(_T - 1))
    once.estimate.build(_day(15), _day(_T - 1))
    twice = _risk_model(tmp_path, "twice", planted=planted, **LAGS)
    twice.regression.build(_day(START), _day(25))
    twice.estimate.build(_day(15), _day(22))
    twice.regression.extend(_day(_T - 1))
    twice.estimate.extend(_day(30)).extend(_day(_T - 1))
    a = once.estimate.read(_day(15), _day(_T - 1)).load()
    b = twice.estimate.read(_day(15), _day(_T - 1)).load()
    xr.testing.assert_identical(a, b)


def test_lags_must_fit_their_window(built):
    with pytest.raises(ValueError, match="volatility_lags"):
        type(built)(dataclasses.replace(built.config, volatility_lags=WINDOWS["volatility_window"]))
    with pytest.raises(ValueError, match="specific_lags"):
        type(built)(dataclasses.replace(built.config, specific_lags=-1))


def _multiplier(x, lags, half_life):
    """``1 + 2 sum_l b_l rho_l`` of one column, at least 0, from the definition."""
    present = np.isfinite(x)
    if not present.any():
        return 1.0
    w = _weights(len(x), half_life)
    c = _centred(x, half_life)
    gamma0 = (w[present] * c[present] ** 2).sum() / w[present].sum()
    multiplier = 1.0
    for lag, b in enumerate(_bartlett(lags), start=1):
        a, d = c[lag:], c[:-lag]
        both = np.isfinite(a) & np.isfinite(d)
        if both.sum() >= LEAST and gamma0 > 0:
            wl = w[lag:][both]
            multiplier += 2 * b * ((wl * a[both] * d[both]).sum() / wl.sum()) / gamma0
    return max(multiplier, 0.0)


def test_factor_autocorrelations_can_use_their_own_half_life_and_window(tmp_path, planted):
    own = dict(volatility_autocorrelation_half_life=8.0, volatility_autocorrelation_window=12)
    model = _risk_model(tmp_path, "own", planted=planted, **LAGS, **own)
    model.regression.build(_day(START), _day(_T - 1))
    model.estimate.build(_day(START), _day(_T - 1))
    returns = model.regression.read(_day(START), _day(_T - 1))["factor_return"].values
    covariance = model.estimate.read(_day(START), _day(_T - 1))["factor_covariance"].values
    got = np.diagonal(covariance, axis1=1, axis2=2)
    for t in range(len(returns)):
        history = returns[: t + 1]
        vol = history[-WINDOWS["volatility_window"]:]
        auto = history[-12:]
        for k in range(returns.shape[1]):
            expected = _moment(vol[:, k], vol[:, k], WINDOWS["volatility_half_life"]) * _multiplier(
                auto[:, k], LAGS["volatility_lags"], 8.0
            )
            np.testing.assert_allclose(got[t, k], expected, rtol=1e-9, atol=1e-15, err_msg=f"{t},{k}")
    # The window counts in the warm-up.
    assert model.estimate_warmup_bars == max(12, WINDOWS["correlation_window"]) - 1


def test_the_estimates_do_not_depend_on_njobs(tmp_path, planted):
    stores = []
    for njobs in (1, 3):
        model = _risk_model(tmp_path, f"jobs{njobs}", planted=planted, **LAGS, njobs=njobs)
        model.regression.build(_day(START), _day(_T - 1))
        model.estimate.build(_day(START), _day(_T - 1))
        stores.append(model.estimate.read(_day(START), _day(_T - 1)).load())
    xr.testing.assert_identical(stores[0], stores[1])
    with pytest.raises(ValueError, match="njobs"):
        type(model)(dataclasses.replace(model.config, njobs=0))
