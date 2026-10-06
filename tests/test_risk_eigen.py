"""The eigenfactor risk adjustment of a factor risk model's covariance (#198).

The planted panel of ``tests/test_risk_regression.py`` gives a regression
store with a factor left out on some bars (industry 4). The estimate store
built from it with the adjustment on is checked against USE4 Methodology
Notes Appendix B written here from the definitions: each bar's covariance
``F0`` (the hand-written truncated estimator of ``tests/test_risk_estimate.py``)
over the factors it covers, its eigen-decomposition ``U0' F0 U0 = D0``,
simulated complete histories ``f_m = U0 b_m`` with ``b_m`` normal of variance
``D0`` drawn from the bar's own random stream, each history's covariance
``F_m`` from the same estimator, ``v(k) = sqrt(mean_m (U_m' F0 U_m)(k) /
D_m(k))`` (eq. B7), the scaled ``a (v_P(k) - 1) + 1`` of a parabolic fit
``v_P`` (eq. B8), and ``U0 v^2 D0 U0'`` (eqs. B9-B10). The covered block
and the parabola are written out again here, so the reference does not lean
on the code it checks.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.test_risk_estimate import START, WINDOWS, _expected_covariance, _risk_model
from tests.test_risk_newey_west import LAGS
from tests.test_risk_newey_west import _expected_covariance as _newey_west_covariance
from tests.test_risk_regression import _T, _day

SIMULATIONS = 15
EIGEN = dict(eigen_simulations=SIMULATIONS, eigen_seed=7)
LONGEST = max(WINDOWS["volatility_window"], WINDOWS["correlation_window"])
TINY = 1e-10  # eigenvalues below this times the largest are taken as 0


def _built(tmp_path, name, **config):
    model = _risk_model(tmp_path, name, **{**EIGEN, **config})
    model.regression.build(_day(START), _day(_T - 1))
    model.estimate.build(_day(START), _day(_T - 1))
    return model


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    return _built(tmp_path_factory.mktemp("eigen"), "simulated")


def _covered(covariance):
    """The factors of the block adjusted: a variance, then greedily every pair."""
    kept = np.isfinite(np.diag(covariance))
    while True:
        missing = ~np.isfinite(covariance) & kept[:, None] & kept[None, :]
        if not missing.any():
            return kept
        kept[np.argmax(missing.sum(axis=1))] = False


def _parabola(v, valid, skip):
    k = np.arange(1, len(v) + 1, dtype=float)
    fit = valid & (k > skip)
    if fit.sum() < 3:
        return v
    return np.polyval(np.polyfit(k[fit], v[fit], 2), k)


def _expected(
    history, timestamp, seed=7, simulations=SIMULATIONS, scale=None, skip=15,
    estimator=_expected_covariance,
):
    """USE4 eqs. B1-B10 at the last row of ``history``, ``estimator`` the sample covariance."""
    raw = estimator(history)
    kept = _covered(raw)
    out = raw.copy()
    if not kept.any():
        return out
    f0 = raw[np.ix_(kept, kept)]
    f0 = (f0 + f0.T) / 2
    d0, u0 = np.linalg.eigh(f0)
    d0 = np.clip(d0, 0.0, None)
    f0 = (u0 * d0) @ u0.T  # what the histories are drawn from
    positive = d0 > d0.max() * TINY
    length = min(len(history), LONGEST)
    rng = np.random.default_rng([seed, pd.Timestamp(timestamp).value])
    ratio = np.zeros(len(d0))
    count = np.zeros(len(d0))
    used = 0
    # With no more bars than factors the simulated covariances are singular.
    for _ in range(simulations if length > len(d0) else 0):
        b = rng.standard_normal((length, len(d0))) * np.sqrt(d0)
        fm = estimator(b @ u0.T)
        if not np.isfinite(fm).all():
            continue  # a negative Newey-West variance: left out
        used += 1
        dm, um = np.linalg.eigh(fm)
        true = np.array([um[:, k] @ f0 @ um[:, k] for k in range(len(d0))])
        good = dm > dm.max() * TINY
        ratio += np.where(good, true / np.where(good, dm, 1.0), 0.0)
        count += good
    valid = positive & (count == used) & (used > 0)
    v = np.where(valid, np.sqrt(ratio / np.maximum(count, 1)), 1.0)
    if scale is not None:
        v = np.where(valid, scale * (_parabola(v, valid, skip) - 1.0) + 1.0, 1.0)
    adjusted = (u0 * np.where(positive, v**2 * d0, 0.0)) @ u0.T
    out[np.ix_(kept, kept)] = (adjusted + adjusted.T) / 2
    return out


def _check(model, rtol=1e-9, **expected):
    regression = model.regression.read(_day(START), _day(_T - 1))
    factor_returns = regression["factor_return"].values
    bars = regression["timestamp"].values
    got = model.estimate.read(_day(START), _day(_T - 1))["factor_covariance"].values
    estimator = expected.get("estimator", _expected_covariance)
    adjusted = 0
    for t in range(len(factor_returns)):
        want = _expected(factor_returns[: t + 1], bars[t], **expected)
        np.testing.assert_allclose(got[t], want, rtol=rtol, atol=1e-15, err_msg=f"bar {t}")
        raw = estimator(factor_returns[: t + 1])
        adjusted += not np.allclose(want, raw, equal_nan=True)
    # The adjustment does change the covariance.
    assert adjusted > len(factor_returns) // 2


def test_the_simulated_adjustment_matches_eq_b7(simulated):
    _check(simulated)


def test_the_simulations_use_the_newey_west_estimator(tmp_path):
    model = _built(tmp_path, "newey_west", **LAGS)
    # A bar just past as many bars as factors has nearly singular simulated
    # covariances, whose smallest eigenvalues amplify round-off.
    _check(model, rtol=1e-7, estimator=_newey_west_covariance)


def test_the_scaled_adjustment_matches_eq_b8(tmp_path):
    model = _built(tmp_path, "scaled", eigen_scale=1.4, eigen_fit_skip=1)
    _check(model, scale=1.4, skip=1)


def _risk_model_rows(model, **config):
    other = type(model)(dataclasses.replace(
        model.config,
        estimate_path=model.config.estimate_path.replace(".zarr", "_other.zarr"),
        **config,
    ))
    other.estimate.build(_day(START), _day(_T - 1))
    return other.estimate.read(_day(START), _day(_T - 1))["factor_covariance"].values


def test_low_volatility_eigenfactors_are_scaled_up(simulated):
    # Sampling error underpredicts the smallest eigenvariance (Figure 4.3).
    raw = _risk_model_rows(simulated, eigen_simulations=0)
    adjusted = simulated.estimate.read(_day(START), _day(_T - 1))["factor_covariance"].values
    ratios = []
    for t in range(10, len(raw)):
        kept = _covered(raw[t])
        d0, u0 = np.linalg.eigh(raw[t][np.ix_(kept, kept)])
        smallest = u0[:, 0] @ adjusted[t][np.ix_(kept, kept)] @ u0[:, 0]
        ratios.append(smallest / d0[0])
    assert np.median(ratios) > 1.0


def test_the_adjusted_covariance_is_symmetric_positive_semi_definite(simulated):
    rows = simulated.estimate.read(_day(START), _day(_T - 1))["factor_covariance"].values
    for covariance in rows:
        kept = _covered(covariance)
        block = covariance[np.ix_(kept, kept)]
        np.testing.assert_array_equal(block, block.T)
        if kept.any():
            assert np.linalg.eigvalsh(block).min() >= -1e-12 * np.abs(block).max()


def test_the_same_config_gives_a_bit_identical_store_and_a_new_seed_another(tmp_path, simulated):
    again = _built(tmp_path, "again")
    other = _built(tmp_path, "other", eigen_seed=8)
    a = simulated.estimate.read(_day(START), _day(_T - 1)).load()
    b = again.estimate.read(_day(START), _day(_T - 1)).load()
    c = other.estimate.read(_day(START), _day(_T - 1)).load()
    np.testing.assert_array_equal(a["factor_covariance"].values, b["factor_covariance"].values)
    late = slice(10, None)
    assert not np.allclose(
        a["factor_covariance"].values[late], c["factor_covariance"].values[late], equal_nan=True
    )


def test_build_then_extend_and_njobs_give_the_same_store(tmp_path, simulated):
    twice = _risk_model(tmp_path, "twice", **EIGEN, njobs=3)
    twice.regression.build(_day(START), _day(_T - 1))
    twice.estimate.build(_day(START), _day(20))
    twice.estimate.extend(_day(_T - 1))
    a = simulated.estimate.read(_day(START), _day(_T - 1)).load()
    b = twice.estimate.read(_day(START), _day(_T - 1)).load()
    xr.testing.assert_identical(a, b)
    np.testing.assert_array_equal(a["factor_covariance"].values, b["factor_covariance"].values)


def test_the_defaults_are_use4s_simulated_adjustment(simulated):
    defaults = type(simulated.config)(
        exposures=simulated.config.exposures, dataset=simulated.config.dataset,
        exposure_data_strategy="cal",
    )
    assert (defaults.eigen_simulations, defaults.eigen_seed) == (1000, 0)
    assert (defaults.eigen_scale, defaults.eigen_fit_skip) == (None, 15)


@pytest.mark.parametrize(
    "field, value",
    [("eigen_simulations", -1), ("eigen_scale", 0.0), ("eigen_fit_skip", -1)],
)
def test_invalid_eigen_parameters_are_refused(simulated, field, value):
    with pytest.raises(ValueError, match=field):
        type(simulated)(dataclasses.replace(simulated.config, **{field: value}))
