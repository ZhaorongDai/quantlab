"""Exposure bounds of the mean-variance optimiser on hand-built contexts.

``MeanVarianceConfig.exposure_bounds`` holds the portfolio's exposure
``sum_i w_i * x_i`` to a factor output between two bounds, the locked
positions' exposure included; ``exposure_factors`` are the factors the rule
declares, whose values at the bar arrive in ``context.factors``. A candidate
without an exposure gets no weight, and a held one is closed
(``closed_without_exposure``).
"""

import dataclasses

import numpy as np
import pytest
import xarray as xr

from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import BenchmarkBetaConfig
from quantlab.factor.predefined.benchmark_beta import BenchmarkBeta
from quantlab.portfolio.base import PortfolioConstructionError
from quantlab.portfolio.config import LedoitWolfEstimatorConfig
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from tests.test_portfolio_mean_variance import LOOKBACK, SYMBOLS, _context, _optimizer

#: The best predictions sit on the highest exposures, so an unconstrained
#: book's exposure is well above 1.
PREDICTION = np.array([2.0, 1.5, 0.2, 0.1, -0.5, -1.0])
EXPOSURE = np.array([2.0, 1.8, 0.6, 0.5, 0.4, 0.3])


def _prices(path=None) -> FrameDataset:
    times = np.datetime64("2024-01-01") + np.arange(5).astype("timedelta64[D]")
    data = FrameDataset(xr.Dataset(
        {"adjClose": (("timestamp", "symbol"), np.ones((5, 1)))},
        coords={"timestamp": times, "symbol": ["VT"]},
    ))
    return data if path is None else data.to_zarr(path)


def _beta(stocks=None, benchmark=None) -> BenchmarkBeta:
    return BenchmarkBeta(BenchmarkBetaConfig(
        warmup_bars=10, dataset=stocks or _prices(), benchmark=benchmark or _prices(),
        lookback_bars=10, min_bars=5,
    ))


def _with_exposure(context, exposure):
    factors = xr.Dataset({"beta": ("symbol", np.asarray(exposure, float))}, coords={"symbol": SYMBOLS})
    return dataclasses.replace(context, factors=factors)


def _bounded(**overrides):
    return _optimizer(exposure_factors=(_beta(),), exposure_bounds={"beta": (0.9, 1.1)}, **overrides)


def _exposure(weights, exposure) -> float:
    return float(np.nansum(np.asarray(weights.values) * np.asarray(exposure)))


def test_the_book_exposure_lands_inside_the_bounds_when_unconstrained_it_would_not():
    context = _with_exposure(_context(prediction=PREDICTION), EXPOSURE)
    free = _optimizer(risk_aversion=1.0).construct(context)
    assert _exposure(free, EXPOSURE) > 1.2
    bounded = _bounded(risk_aversion=1.0).construct(context)
    assert 0.9 - 1e-6 <= _exposure(bounded, EXPOSURE) <= 1.1 + 1e-6
    assert bounded.sum() == pytest.approx(1.0)


def test_a_candidate_without_an_exposure_gets_no_weight_and_a_held_one_is_closed():
    exposure = EXPOSURE.copy()
    exposure[[0, 2]] = np.nan
    current = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
    # Without AAA and CCC, a cap of 0.5 still lets BBB lift the book to 0.9.
    weights = _bounded(weight_cap=0.5).construct(_with_exposure(_context(prediction=PREDICTION, current=current), exposure))
    assert weights.sel(symbol="AAA") == 0.0 and weights.sel(symbol="CCC") == 0.0
    assert weights.attrs["events"]["closed_without_exposure"] == ["CCC"]


def test_a_locked_positions_exposure_counts_toward_the_bound():
    """FFF is held at 0.3 and not tradable, with an exposure of 2: it alone
    brings 0.6, so the other 0.7 may add at most 0.5, below what AAA and BBB
    would add."""
    exposure = EXPOSURE.copy()
    exposure[5] = 2.0
    eligible = [True] * 5 + [False]
    current = [0.0] * 5 + [0.3]
    context = _with_exposure(_context(prediction=PREDICTION, eligible=eligible, current=current), exposure)
    weights = _bounded().construct(context)
    assert weights.sel(symbol="FFF") == pytest.approx(0.3)
    assert _exposure(weights, exposure) <= 1.1 + 1e-6


def test_a_locked_position_without_an_exposure_makes_the_bar_fail_rather_than_count_zero():
    exposure = EXPOSURE.copy()
    exposure[5] = np.nan
    eligible = [True] * 5 + [False]
    current = [0.0] * 5 + [0.3]
    context = _with_exposure(_context(prediction=PREDICTION, eligible=eligible, current=current), exposure)
    with pytest.raises(PortfolioConstructionError, match="FFF"):
        _bounded().construct(context)


def test_a_candidate_pool_of_high_exposures_only_makes_the_bar_infeasible():
    """The two best predictions carry exposures of 2.0 and 1.8; a pool of two
    cannot reach 1.1, while the whole universe can."""
    context = _with_exposure(_context(prediction=PREDICTION), EXPOSURE)
    assert 0.9 - 1e-6 <= _exposure(_bounded(weight_cap=0.5).construct(context), EXPOSURE) <= 1.1 + 1e-6
    with pytest.raises(PortfolioConstructionError):
        _bounded(weight_cap=0.5, candidate_top_k=2).construct(context)


def test_a_long_short_book_is_held_inside_its_exposure_bounds():
    """Dollar-neutral, and beta-neutral within 0.05."""
    context = _with_exposure(_context(prediction=PREDICTION), EXPOSURE)
    free = _optimizer(direction="long_short", risk_aversion=1.0).construct(context)
    assert abs(_exposure(free, EXPOSURE)) > 0.1
    weights = _optimizer(
        direction="long_short", risk_aversion=1.0, exposure_factors=(_beta(),),
        exposure_bounds={"beta": (-0.05, 0.05)},
    ).construct(context)
    assert abs(_exposure(weights, EXPOSURE)) <= 0.05 + 1e-6
    assert float(weights.sum()) == pytest.approx(0.0, abs=1e-9)


def test_bounds_the_candidates_cannot_reach_make_the_bar_infeasible():
    context = _with_exposure(_context(prediction=PREDICTION), np.full(len(SYMBOLS), 1.5))
    with pytest.raises(PortfolioConstructionError):
        _bounded().construct(context)


@pytest.mark.parametrize("overrides, match", [
    (dict(exposure_bounds={"size": (0.9, 1.1)}), "not an output"),
    (dict(exposure_bounds={"beta": (1.1, 0.9)}), "lower bound"),
    (dict(exposure_bounds={"beta": (0.9,)}), "a pair"),
])
def test_bad_exposure_bounds_are_refused_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        _optimizer(exposure_factors=(_beta(),), **overrides)


def test_two_different_declared_factors_with_one_output_name_are_refused():
    """Two betas over different windows both name their output ``beta``; an
    equal factor declared twice is one (test below)."""
    longer = BenchmarkBeta(BenchmarkBetaConfig(
        warmup_bars=20, dataset=_prices(), benchmark=_prices(), lookback_bars=20, min_bars=5,
    ))
    with pytest.raises(ValueError, match="more than once"):
        _optimizer(exposure_factors=(_beta(), longer), exposure_bounds={"beta": (0.9, 1.1)})


def test_the_rule_declares_its_exposure_factors():
    beta = _beta()
    assert _optimizer(exposure_factors=(beta,), exposure_bounds={"beta": (0.9, 1.1)}).required_factors() == [beta]


def test_the_optimizer_round_trips_through_its_config_with_its_exposure_factors(tmp_path):
    beta = _beta(_prices(tmp_path / "stocks.zarr"), _prices(tmp_path / "vt.zarr"))
    optimizer = _optimizer(exposure_factors=(beta,), exposure_bounds={"beta": (0.9, 1.1)})
    again = rebuild(optimizer.get_config())
    assert again.config.exposure_bounds == {"beta": (0.9, 1.1)}
    assert again.required_factors()[0].get_config() == beta.get_config()


class _RiskModel:
    """Stands in for a factor risk model whose one exposure is ``beta``."""

    exposure_names = ("beta",)


class _WithRiskModel(LedoitWolfEstimator):
    """A covariance estimator declaring a factor risk model, as ``FactorRiskStoreEstimator`` does."""

    def required_risk_model(self):
        return _RiskModel()


def test_an_exposure_of_the_risk_model_can_be_bounded_without_declaring_a_factor():
    """The bounded exposure arrives in context.risk_exposures (#230)."""
    optimizer = _optimizer(
        risk_aversion=1.0,
        covariance=_WithRiskModel(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
        exposure_bounds={"beta": (0.9, 1.1)},
    )
    assert optimizer.required_factors() == []
    context = _with_exposure(_context(prediction=PREDICTION), EXPOSURE)
    context = dataclasses.replace(context, factors=None, risk_exposures=context.factors)
    weights = optimizer.construct(context)
    assert 0.9 - 1e-6 <= _exposure(weights, EXPOSURE) <= 1.1 + 1e-6
    with pytest.raises(ValueError, match="holds no exposure 'beta'"):
        optimizer.construct(dataclasses.replace(context, risk_exposures=None))


def test_a_bound_on_a_name_both_a_factor_and_the_risk_model_produce_is_refused():
    covariance = _WithRiskModel(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK))
    with pytest.raises(ValueError, match="both an output"):
        _optimizer(covariance=covariance, exposure_factors=(_beta(),), exposure_bounds={"beta": (0.9, 1.1)})
    # Declaring the factor without bounding the shared name is fine.
    _optimizer(covariance=covariance, exposure_factors=(_beta(),))
