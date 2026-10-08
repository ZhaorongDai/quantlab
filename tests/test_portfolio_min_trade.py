"""``MeanVarianceConfig.min_trade``: a solved change smaller than it is not traded.

A candidate whose solved weight differs from its current one by less than
``min_trade`` keeps its current weight, so no order is placed for it: neither
a solver residue on an unheld symbol nor a tiny adjustment of a held one. When
the skipped sells leave the long-only book above its budget, the candidates
that do trade are scaled down to absorb the excess; when they cannot, the small
sells are made as solved. With no turnover penalty the solution does not
depend on the current weights, so each test perturbs a plain solution by hand.
"""

import numpy as np
import pytest

from quantlab.core.component import rebuild
from tests.test_portfolio_mean_variance import _context, _optimizer

PARAMS = dict(risk_aversion=50.0, weight_cap=0.5)


def _plain():
    """The plain solution on the default context: all six names held, five above 0.09."""
    weights = _optimizer(**PARAMS).construct(_context()).values
    assert (weights > 0.09).sum() >= 3
    return weights


def _with_min_trade(min_trade, current):
    return _optimizer(min_trade=min_trade, **PARAMS).construct(_context(current=current)).values


def test_a_solver_residue_on_an_unheld_symbol_is_not_bought():
    """At risk aversion 1 the solver leaves about 1.8e-10 on DDD."""
    plain = _optimizer(risk_aversion=1.0, weight_cap=0.5).construct(_context()).values
    residue = (plain > 0) & (plain < 1e-6)
    assert residue.any()
    weights = _optimizer(risk_aversion=1.0, weight_cap=0.5, min_trade=1e-4).construct(_context()).values
    assert (weights[residue] == 0.0).all()
    np.testing.assert_allclose(weights[~residue], plain[~residue], atol=1e-9)


def test_a_change_below_min_trade_keeps_the_current_weight():
    plain = _plain()
    big = np.flatnonzero(plain > 0.09)[:2]
    current = plain.copy()
    current[big] += [4e-4, -4e-4]
    weights = _with_min_trade(1e-3, current)
    np.testing.assert_array_equal(weights[big], current[big])
    np.testing.assert_allclose(weights, current, atol=1e-12)
    # Without it, the optimiser moves them back to the plain solution.
    moved = _optimizer(**PARAMS).construct(_context(current=current)).values
    np.testing.assert_allclose(moved[big], plain[big], atol=1e-6)


def test_a_new_position_below_min_trade_is_not_opened_and_its_budget_stays_cash():
    plain = _plain()
    small = int(np.argmin(np.where(plain > 0, plain, np.inf)))
    weights = _with_min_trade(plain[small] + 1e-3, np.zeros(len(plain)))
    assert weights[small] == 0.0
    others = np.flatnonzero((plain > plain[small] + 1e-3))
    np.testing.assert_allclose(weights[others], plain[others], atol=1e-9)
    assert weights.sum() == pytest.approx(1.0 - plain[small], abs=1e-9)


def test_skipped_sells_above_the_budget_are_absorbed_by_the_names_that_trade():
    """AAA..: one name 5e-4 above its solution (a skipped sell), another not
    held at all (a large buy): the buy is cut so the book sums to one."""
    plain = _plain()
    sell, buy = np.flatnonzero(plain > 0.09)[:2]
    current = plain.copy()
    current[sell] += 5e-4
    current[buy] = 0.0
    weights = _with_min_trade(1e-3, current)
    assert weights[sell] == current[sell]
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert plain[buy] - 1e-3 < weights[buy] < plain[buy]


def test_small_sells_are_made_when_nothing_else_trades_to_absorb_them():
    plain = _plain()
    held = np.flatnonzero(plain > 0.09)[:2]
    current = plain.copy()
    current[held] += 4e-4  # both small sells; the book would sum to 1.0008
    weights = _with_min_trade(1e-3, current)
    np.testing.assert_allclose(weights[held], plain[held], atol=1e-9)
    assert weights.sum() <= 1.0 + 1e-12


@pytest.mark.parametrize("overrides, match", [
    (dict(min_trade=-1e-4), "min_trade"),
    (dict(min_trade=1e-4, direction="long_short"), "long_only"),
])
def test_bad_min_trade_is_refused_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        _optimizer(**overrides)


def test_min_trade_round_trips_through_the_config():
    optimizer = _optimizer(min_trade=1e-4)
    assert rebuild(optimizer.get_config()).config.min_trade == 1e-4
