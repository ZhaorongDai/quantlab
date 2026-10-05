"""The one-bar decision: ``PortfolioConstructor.decide`` (#108, trader ADR 0005).

What is locked here, and what turns it red (hand-built contexts, no
vectorbt; the contexts ``DecisionInputs`` assembles are locked in
``test_decision_inputs.py``):

- ``decide`` returns the row ``construct`` gave as its weights, and the
  row's events.
- ``decide`` holds a bar whose ``construct`` raises
  ``PortfolioConstructionError`` (all-NaN weights, the message as
  ``failure``); a broken contract still raises ``ValueError``.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.base import (
    PortfolioConstructionError,
    PortfolioConstructor,
    PortfolioContext,
)


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
    return rule, PortfolioContext(
        timestamp=pd.Timestamp("2024-01-02"),
        predictions=xr.Dataset({"ret_5": ("symbol", [0.3, 0.1, 0.2])}, coords={"symbol": symbols}),
        tradable=xr.DataArray(list(tradable), dims="symbol", coords={"symbol": symbols}),
        current_weights=xr.DataArray(np.asarray(current, float), dims="symbol", coords={"symbol": symbols}),
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
