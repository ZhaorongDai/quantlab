"""The public one-bar decision: ``build_context`` and ``decide`` (#108, trader ADR 0005).

What is locked here, and what turns it red (hand-built panels, no vectorbt):

- At every rebalance bar, ``build_context`` on the valuation prices up to
  that bar gives the context ``construct_panel`` built, for the top-n rule
  and for mean-variance with a Ledoit-Wolf risk model (halts, a late listing
  and a delisting in the prices), and without prices.
- ``decide`` on that context returns the panel's row as its weights, and the
  row's events.
- ``decide`` holds a bar whose ``construct`` raises
  ``PortfolioConstructionError`` (all-NaN weights, the message as
  ``failure``); a broken contract still raises ``ValueError``.
- ``build_context`` refuses missing prices when the rule reads returns,
  prices that do not end at the bar, and missing factors when the rule
  declares them.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import LedoitWolfConfig, MeanVarianceConfig, TopNConfig
from quantlab.base.portfolio import (
    Decision,
    LabelSpec,
    PortfolioConstructionError,
    PortfolioConstructor,
    PortfolioContext,
)
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor

SYMBOLS = [f"S{i}" for i in range(8)]
WARMUP = 30
BARS = 60
SPECS = [LabelSpec(name="ret_5", scale="raw", delay=1, span=5)]


def _market():
    """Prices with a halt, a late listing and a delisting; predictions on the last BARS bars."""
    rng = np.random.default_rng(7)
    ts = pd.bdate_range("2024-01-01", periods=WARMUP + BARS)
    prices = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(len(ts), len(SYMBOLS))), axis=0))
    prices[WARMUP + 10 : WARMUP + 16, 1] = np.nan  # S1 halts
    prices[: WARMUP + 5, 2] = np.nan  # S2 lists late
    prices[WARMUP + 30 :, 3] = np.nan  # S3 delists
    delisted = np.zeros(prices.shape, dtype=bool)
    delisted[WARMUP + 29, 3] = True
    coords = {"timestamp": ts, "symbol": SYMBOLS}
    price = xr.DataArray(prices, dims=("timestamp", "symbol"), coords=coords, name="close")
    scores = rng.normal(size=(BARS, len(SYMBOLS)))
    scores[:, [1, 3]] += 3.0  # S1 and S3 are held into the halt and the delisting
    predictions = xr.Dataset(
        {"ret_5": (("timestamp", "symbol"), scores)},
        coords={"timestamp": ts[WARMUP:], "symbol": SYMBOLS},
    )
    tradable = price.isel(timestamp=slice(WARMUP, None)).notnull()
    marks = xr.DataArray(delisted, dims=("timestamp", "symbol"), coords=coords)
    rebalance = np.zeros(BARS, dtype=bool)
    rebalance[::3] = True
    return price, predictions, tradable, marks, rebalance


class _RecordingTopN(TopNConstructor):
    seen: list = []

    def construct(self, context):
        type(self).seen.append(context)
        return super().construct(context)


class _RecordingMeanVariance(MeanVarianceOptimizer):
    seen: list = []

    def construct(self, context):
        type(self).seen.append(context)
        return super().construct(context)


def _top_n():
    rule = _RecordingTopN(TopNConfig(direction="long_only", top_n=3))
    rule.bind(SPECS)
    return rule


def _mean_variance():
    rule = _RecordingMeanVariance(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=20)),
            ic=0.05,
            risk_aversion=5.0,
            turnover_penalty=0.002,
            weight_cap=0.4,
        )
    )
    rule.bind(SPECS)
    return rule


def _assert_same_context(built: PortfolioContext, looped: PortfolioContext):
    assert built.timestamp == looped.timestamp
    xr.testing.assert_identical(built.predictions, looped.predictions)
    xr.testing.assert_identical(built.tradable, looped.tradable)
    xr.testing.assert_identical(built.current_weights, looped.current_weights)
    xr.testing.assert_identical(built.returns, looped.returns)
    if looped.staleness is None:
        assert built.staleness is None
    else:
        xr.testing.assert_identical(built.staleness, looped.staleness)
    assert built.factors is None and looped.factors is None


def _replay(rule, *, with_prices=True):
    """Run the panel, then rebuild and decide every bar it saw through the public pair."""
    price, predictions, tradable, marks, rebalance = _market()
    type(rule).seen = []
    prices = dict(fill_price=price, valuation_price=price, delisted=marks) if with_prices else {}
    panel = rule.construct_panel(predictions, tradable, rebalance, **prices)
    seen = list(type(rule).seen)
    assert len(seen) == rebalance.sum()
    for looped in seen:
        t = looped.timestamp
        built = rule.build_context(
            t,
            predictions.sel(timestamp=t),
            tradable.sel(timestamp=t),
            looped.current_weights,
            valuation_price=price.sel(timestamp=slice(None, t)) if with_prices else None,
        )
        _assert_same_context(built, looped)
        decision = rule.decide(built)
        assert isinstance(decision, Decision)
        assert decision.failure is None
        assert decision.weights.dims == ("symbol",)
        assert decision.weights.symbol.values.tolist() == SYMBOLS
        np.testing.assert_array_equal(decision.weights.values, panel["weight"].sel(timestamp=t).values)
    return panel, seen


def test_top_n_build_context_and_decide_reproduce_the_panel_at_every_rebalance_bar():
    panel, seen = _replay(_top_n())

    held = [c for c in seen if (c.current_weights.values != 0).any()]
    assert held, "the replay must cover bars with holdings"
    assert any(bool(c.locked.any()) for c in seen), "the replay must cover a locked position"


def test_mean_variance_build_context_and_decide_reproduce_the_panel_at_every_rebalance_bar():
    panel, seen = _replay(_mean_variance())

    assert all(c.returns.sizes["timestamp"] == 20 for c in seen)
    assert np.isfinite(panel["weight"].values).any()


def test_without_prices_the_context_has_an_empty_window_and_no_staleness():
    _, seen = _replay(_top_n(), with_prices=False)

    assert seen[0].returns.sizes["timestamp"] == 0 and seen[0].staleness is None


class _Fails(PortfolioConstructor):
    config_cls = TopNConfig

    def construct(self, context):
        raise PortfolioConstructionError("solver failed: infeasible")


class _Returns(PortfolioConstructor):
    """Returns the row it is given, to probe the contract check."""

    config_cls = TopNConfig
    row: list = []

    def construct(self, context):
        out = xr.DataArray(np.asarray(self.row, float), dims="symbol", coords={"symbol": context.symbols})
        out.attrs["events"] = {"tie_at_cutoff": 2}
        return out


def _hand_context(*, current=(0.0, 0.0, 0.0), tradable=(True, True, True)):
    symbols = ["AAA", "BBB", "CCC"]
    rule = _Returns(TopNConfig(direction="long_only", top_n=1))
    return rule, rule.build_context(
        pd.Timestamp("2024-01-02"),
        xr.Dataset({"ret_5": ("symbol", [0.3, 0.1, 0.2])}, coords={"symbol": symbols}),
        xr.DataArray(list(tradable), dims="symbol", coords={"symbol": symbols}),
        xr.DataArray(list(current), dims="symbol", coords={"symbol": symbols}),
    )


def test_decide_holds_a_bar_the_rule_cannot_solve_with_its_message():
    _, context = _hand_context()
    decision = _Fails(TopNConfig(direction="long_only", top_n=1)).decide(context)

    assert np.isnan(decision.weights.values).all()
    assert decision.weights.symbol.values.tolist() == ["AAA", "BBB", "CCC"]
    assert decision.failure == "solver failed: infeasible"
    assert decision.events == {}


def test_decide_returns_the_rows_weights_and_events():
    rule, context = _hand_context()
    rule.row = [1.0, 0.0, 0.0]
    decision = rule.decide(context)

    assert decision.weights.values.tolist() == [1.0, 0.0, 0.0]
    assert decision.failure is None
    assert decision.events == {"tie_at_cutoff": 2}


@pytest.mark.parametrize(
    ("row", "current", "tradable", "message"),
    [
        ([1.0, np.nan, 0.0], (0, 0, 0), (True, True, True), "mixing finite"),
        ([0.0, 1.0, 0.0], (0.5, 0, 0), (False, True, True), "changed the locked position"),
        ([0.5, 0.5, 0.0], (0, 0, 0), (True, False, True), "untradable, unheld"),
    ],
)
def test_decide_raises_on_a_broken_contract(row, current, tradable, message):
    rule, context = _hand_context(current=current, tradable=tradable)
    rule.row = row

    with pytest.raises(ValueError, match=message):
        rule.decide(context)


def test_build_context_refuses_missing_prices_when_the_rule_reads_returns():
    price, predictions, tradable, _, _ = _market()
    t = predictions.timestamp.values[5]
    current = xr.zeros_like(tradable.sel(timestamp=t), dtype=float)

    with pytest.raises(ValueError, match="valuation_price"):
        _mean_variance().build_context(t, predictions.sel(timestamp=t), tradable.sel(timestamp=t), current)


def test_build_context_refuses_prices_that_do_not_end_at_the_bar():
    price, predictions, tradable, _, _ = _market()
    t = predictions.timestamp.values[5]
    current = xr.zeros_like(tradable.sel(timestamp=t), dtype=float)

    with pytest.raises(ValueError, match="end at"):
        _mean_variance().build_context(
            t, predictions.sel(timestamp=t), tradable.sel(timestamp=t), current, valuation_price=price
        )


class _Size:
    """Stands in for a ``Factor``: all the rule layer reads of one is its names."""

    def get_factor_names(self):
        return ["size"]


class _NeedsFactors(TopNConstructor):
    seen: list = []

    def required_factors(self):
        return [_Size()]

    def construct(self, context):
        type(self).seen.append(context)
        return super().construct(context)


def test_build_context_refuses_missing_factors_when_the_rule_declares_them():
    _, context = _hand_context()
    rule = _NeedsFactors(TopNConfig(direction="long_only", top_n=1))

    with pytest.raises(ValueError, match="factors"):
        rule.build_context(context.timestamp, context.predictions, context.tradable, context.current_weights)

    exposures = xr.Dataset({"size": ("symbol", [1.0, 2.0])}, coords={"symbol": ["CCC", "AAA"]})
    built = rule.build_context(
        context.timestamp, context.predictions, context.tradable, context.current_weights, factors=exposures
    )
    assert built.factors["size"].sel(symbol="AAA") == 2.0
    assert np.isnan(built.factors["size"].sel(symbol="BBB"))


def test_build_context_refuses_factor_values_lacking_a_declared_name():
    _, context = _hand_context()
    rule = _NeedsFactors(TopNConfig(direction="long_only", top_n=1))
    other = xr.Dataset({"beta": ("symbol", [1.0, 2.0, 3.0])}, coords={"symbol": ["AAA", "BBB", "CCC"]})

    with pytest.raises(ValueError, match=r"\['size'\]"):
        rule.build_context(context.timestamp, context.predictions, context.tradable, context.current_weights, factors=other)


def test_build_context_gives_the_factor_values_the_panel_loop_gave():
    price, predictions, tradable, _, rebalance = _market()
    exposures = xr.Dataset({"size": np.log(price).isel(timestamp=slice(WARMUP, None)).isel(symbol=slice(1, None))})
    rule = _NeedsFactors(TopNConfig(direction="long_only", top_n=3))
    _NeedsFactors.seen = []
    rule.construct_panel(predictions, tradable, rebalance, factors=exposures)

    for looped in _NeedsFactors.seen:
        t = looped.timestamp
        built = rule.build_context(
            t, predictions.sel(timestamp=t), tradable.sel(timestamp=t), looped.current_weights,
            factors=exposures.sel(timestamp=t),
        )
        xr.testing.assert_identical(built.factors, looped.factors)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p.isel(timestamp=[1, 0, 2, 3, 4, 5]), "increasing"),
        (lambda p: p.isel(symbol=[0, 0, 1, 2, 3, 4, 5, 6, 7]), "duplicate symbols"),
    ],
)
def test_build_context_refuses_a_malformed_price_history(change, message):
    price, predictions, tradable, _, _ = _market()
    t = predictions.timestamp.values[0]
    prices = change(price.isel(timestamp=slice(WARMUP - 5, WARMUP + 1)))
    current = xr.zeros_like(tradable.sel(timestamp=t), dtype=float)

    with pytest.raises(ValueError, match=message):
        _top_n().build_context(
            t, predictions.sel(timestamp=t), tradable.sel(timestamp=t), current, valuation_price=prices
        )


def test_build_context_refuses_tradability_that_is_not_boolean():
    _, context = _hand_context()
    rule = _Returns(TopNConfig(direction="long_only", top_n=1))

    with pytest.raises(ValueError, match="booleans"):
        rule.build_context(
            context.timestamp, context.predictions, context.tradable.astype(float).where(context.tradable == 0),
            context.current_weights,
        )
