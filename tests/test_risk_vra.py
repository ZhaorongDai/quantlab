"""The volatility regime adjustment of a factor risk model's forecasts (#200).

The planted panel of ``tests/test_risk_regression.py``, with a specific return
for every symbol, gives a regression store; the estimate store built from it
with the adjustment is checked against USE4 Methodology Notes §4.3 and §5.3
written here from the definitions, on the forecasts of the same store built
without it: each bar's factor cross-sectional bias statistic ``B_F =
sqrt(mean_k (f_k / sigma_k)^2)`` (eq. 4.3) and specific one ``B_S =
sqrt(sum_n w_n (u_n / sigma_n)^2)`` (eq. 5.10, cap weights of the previous
bar over its estimation universe), each forecast the previous bar's; the
multiplier ``lambda = sqrt(sum_t w_t B_t^2)`` over the window with weights
of the half-life normalised to 1 (eqs. 4.4, 5.11); the covariance times
``lambda_F^2``, so the correlations do not move, the specific risk times
``lambda_S`` (eqs. 4.5, 5.12). A panel whose returns triple midway shows the
adjusted forecast catching up first.
"""

import dataclasses

import numpy as np
import pytest

from tests.test_risk_estimate import START, WINDOWS, _risk_model
from tests.test_risk_regression import _T, _day, _plant

VRA = dict(vra_half_life=3.0, vra_window=6)
LEAST = WINDOWS["min_observations"]


def _pair(tmp_path, name, planted, **config):
    """The stores built without the adjustment and with it."""
    stores = []
    for suffix, extra in (("raw", dict(vra_half_life=None)), ("vra", VRA)):
        model = _risk_model(tmp_path, f"{name}_{suffix}", planted=planted, **{**extra, **config})
        model.regression.build(_day(START), _day(_T - 1))
        model.estimate.build(_day(START), _day(_T - 1))
        stores.append(model)
    return stores


@pytest.fixture(scope="module")
def planted():
    return _plant(specific_everywhere=0.02)


@pytest.fixture(scope="module")
def pair(tmp_path_factory, planted):
    return _pair(tmp_path_factory.mktemp("vra"), "pair", planted)


def _expected_multipliers(raw, planted):
    prices, exposures, _, _ = planted
    regression = raw.regression.read(_day(START), _day(_T - 1))
    rows = raw.estimate.read(_day(START), _day(_T - 1))
    bars = regression["timestamp"].values
    symbols = rows["symbol"].values
    covariance = rows["factor_covariance"].values
    sigma = rows["specific_risk"].values
    f = regression["factor_return"].values
    u = regression["specific_return"].reindex(symbol=symbols).values
    cap = prices["marketcap"].sel(timestamp=bars).reindex(symbol=symbols).values
    estu = exposures["estu"].sel(timestamp=bars).reindex(symbol=symbols).values == 1.0
    factor_bias = np.full(len(bars), np.nan)
    specific_bias = np.full(len(bars), np.nan)
    for s in range(1, len(bars)):
        vol = np.sqrt(np.diag(covariance[s - 1]))
        ok = np.isfinite(vol) & (vol > 0) & np.isfinite(f[s])
        if ok.any():
            factor_bias[s] = np.sqrt(np.mean((f[s][ok] / vol[ok]) ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = u[s] / sigma[s - 1]
        ok = np.isfinite(z) & (sigma[s - 1] > 0) & np.isfinite(cap[s - 1]) & (cap[s - 1] > 0)
        ok &= estu[s - 1]
        if ok.any():
            w = cap[s - 1][ok]
            specific_bias[s] = np.sqrt((w * z[ok] ** 2).sum() / w.sum())

    def multiplier(bias, j):
        s = np.arange(max(1, j - VRA["vra_window"] + 1), j + 1)
        b = bias[s]
        w = 0.5 ** ((j - s) / VRA["vra_half_life"])
        ok = np.isfinite(b)
        if ok.sum() < LEAST:
            return 1.0
        return np.sqrt((w[ok] * b[ok] ** 2).sum() / w[ok].sum())

    return (
        np.array([multiplier(factor_bias, j) for j in range(len(bars))]),
        np.array([multiplier(specific_bias, j) for j in range(len(bars))]),
    )


def test_the_adjustment_matches_eqs_4_3_to_5_12(pair, planted):
    raw, vra = pair
    factor, specific = _expected_multipliers(raw, planted)
    plain = raw.estimate.read(_day(START), _day(_T - 1))
    adjusted = vra.estimate.read(_day(START), _day(_T - 1))
    np.testing.assert_allclose(
        adjusted["factor_volatility_multiplier"].values, factor, rtol=1e-12
    )
    np.testing.assert_allclose(
        adjusted["specific_volatility_multiplier"].values, specific, rtol=1e-12
    )
    np.testing.assert_allclose(
        adjusted["factor_covariance"].values,
        plain["factor_covariance"].values * factor[:, None, None] ** 2,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        adjusted["specific_risk"].values,
        plain["specific_risk"].values * specific[:, None],
        rtol=1e-12,
    )
    # The multipliers do move away from 1.
    assert np.abs(factor[10:] - 1.0).max() > 0.05
    assert np.abs(specific[10:] - 1.0).max() > 0.05


def _correlation(covariance):
    vol = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    return covariance / (vol[:, :, None] * vol[:, None, :])


def test_the_correlations_are_unchanged(pair):
    raw, vra = pair
    plain = raw.estimate.read(_day(START), _day(_T - 1))["factor_covariance"].values
    adjusted = vra.estimate.read(_day(START), _day(_T - 1))["factor_covariance"].values
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(_correlation(adjusted), _correlation(plain), rtol=1e-12)


def test_the_adjusted_forecast_catches_up_with_a_volatility_jump(tmp_path):
    # Slow volatilities (a 40-bar half-life over a 12-bar window) lag a
    # regime change; the multipliers lift them. The first bars after the
    # jump overshoot (the jump bar's returns against the low forecasts before
    # it), so the forecasts are compared from the sixth bar on.
    jump = 25
    slow = dict(
        volatility_half_life=40.0, volatility_window=12, specific_half_life=40.0,
        specific_window=8,
    )
    raw, vra = _pair(tmp_path, "jump", _plant(specific_everywhere=0.02, jump=(jump, 3.0)), **slow)
    later = slice(jump - START + 6, None)
    plain = raw.estimate.read(_day(START), _day(_T - 1))
    adjusted = vra.estimate.read(_day(START), _day(_T - 1))

    def country(rows):
        return np.sqrt(rows["factor_covariance"].sel(factor_i="country", factor_j="country").values)

    # The country factor's volatility is planted at 0.01 a bar, 0.03 after the jump.
    gap_plain = np.abs(country(plain)[later] - 0.03).mean()
    gap_adjusted = np.abs(country(adjusted)[later] - 0.03).mean()
    assert gap_adjusted < gap_plain / 2
    assert (adjusted["factor_volatility_multiplier"].values[later] > 1.0).all()
    assert (adjusted["specific_volatility_multiplier"].values[later] > 1.0).all()


def test_build_then_extend_equals_build(tmp_path, pair):
    _, once = pair
    twice = _risk_model(tmp_path, "twice", planted=_plant(specific_everywhere=0.02), **VRA)
    twice.regression.build(_day(START), _day(_T - 1))
    twice.estimate.build(_day(START), _day(20))
    twice.estimate.extend(_day(_T - 1))
    a = once.estimate.read(_day(START), _day(_T - 1)).load()
    b = twice.estimate.read(_day(START), _day(_T - 1)).load()
    for name in a.data_vars:
        np.testing.assert_array_equal(a[name].values, b[name].values, err_msg=name)


def test_the_warm_up_counts_the_window(pair):
    raw, vra = pair
    assert vra.estimate_warmup_bars == raw.estimate_warmup_bars + VRA["vra_window"]


def test_the_default_half_life_is_use4s(pair):
    config = pair[1].config
    defaults = type(config)(
        exposures=config.exposures, dataset=config.dataset, exposure_data_strategy="cal"
    )
    assert defaults.vra_half_life == 42.0  # USE4S
    assert defaults.vra_window == 126  # our choice: three half-lives


@pytest.mark.parametrize("field, value", [("vra_half_life", 0.0), ("vra_window", 1)])
def test_invalid_vra_parameters_are_refused(pair, field, value):
    with pytest.raises(ValueError, match=field):
        type(pair[1])(dataclasses.replace(pair[1].config, **{field: value}))
