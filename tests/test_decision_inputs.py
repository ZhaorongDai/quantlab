"""``DecisionInputs`` assembles a rule's decision inputs, for a panel and for one bar (#127).

What is locked here, and what turns it red (an in-memory price dataset with a
halt, a late listing and a delisting; no vectorbt):

- On every rebalance bar, the context ``weights()`` built equals ``context()``
  at that bar given the replayed holdings, value for value (top-n,
  mean-variance with Ledoit-Wolf, and a rule declaring a factor), and
  ``decide`` on it returns the panel's row.
- ``context()`` puts a held symbol without a prediction on the bar's symbols,
  locked where it cannot trade.
- ``rebalances(t)`` is the schedule ``weights()`` follows: every
  ``rebalance_periods`` bars from the anchor, never the last bar or ``end``,
  counted on the dataset's calendar when the panel starts after the anchor.
- ``weights()`` warns once when the dataset holds fewer bars than the first
  window needs; ``context()`` does not.
- One module assembles: no ``PortfolioContext`` is built outside it, the
  backtester, the backtest base and the API reference none of the assembly
  names (tradability, schedule, warm-up, factors, the replay book), and a
  rule has no ``build_context`` or ``construct_panel``.
- Refused: a panel before the anchor, a bar off the calendar, non-finite
  holdings, a panel handed as one bar's predictions, and factor panels
  lacking a declared name.
"""

import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import LedoitWolfConfig, MeanVarianceConfig, TopNConfig
from quantlab.base.portfolio import LabelSpec, PortfolioContext
from quantlab.dataset.memory import FrameDataset
from quantlab.portfolio.decision_inputs import DecisionInputs, rebalance_mask
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor

SYMBOLS = [f"S{i}" for i in range(8)]
WARMUP = 30
BARS = 60
PERIODS = 3
SPECS = [LabelSpec(name="ret_5", scale="raw", delay=1, span=5)]
TS = pd.bdate_range("2024-01-01", periods=WARMUP + BARS)


def _price():
    """Prices with a halt, a late listing and a delisting."""
    rng = np.random.default_rng(7)
    prices = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(len(TS), len(SYMBOLS))), axis=0))
    prices[WARMUP + 10 : WARMUP + 16, 1] = np.nan  # S1 halts
    prices[: WARMUP + 5, 2] = np.nan  # S2 lists late
    prices[WARMUP + 30 :, 3] = np.nan  # S3 delists
    return xr.DataArray(prices, dims=("timestamp", "symbol"), coords={"timestamp": TS, "symbol": SYMBOLS})


def _predictions():
    rng = np.random.default_rng(8)
    scores = rng.normal(size=(BARS, len(SYMBOLS)))
    scores[:, [1, 3]] += 3.0  # S1 and S3 are held into the halt and the delisting
    return xr.Dataset(
        {"ret_5": (("timestamp", "symbol"), scores)},
        coords={"timestamp": TS[WARMUP:], "symbol": SYMBOLS},
    )


def _dataset(price=None):
    price = _price() if price is None else price
    return FrameDataset(xr.Dataset({"open": price, "close": price}))


def _inputs(rule, dataset=None, *, anchor=None, end=None):
    return DecisionInputs(
        _dataset() if dataset is None else dataset,
        rule,
        fill_column="open",
        valuation_column="close",
        rebalance_periods=PERIODS,
        anchor=TS[WARMUP] if anchor is None else anchor,
        end=end,
    )


class _LogPrice:
    """Stands in for a ``Factor``: ``size`` is the log close, from the first bar asked for."""

    def get_factor_names(self):
        return ["size"]

    def compute(self, first, last):
        return xr.Dataset({"size": np.log(_price()).sel(timestamp=slice(first, last))})


class _Recording:
    """Records every context ``construct`` is handed."""

    seen: list

    def construct(self, context):
        self.seen.append(context)
        return super().construct(context)


class _TopN(_Recording, TopNConstructor):
    pass


class _MeanVariance(_Recording, MeanVarianceOptimizer):
    pass


class _WithFactor(_TopN):
    def required_factors(self):
        return [_LogPrice()]


def _rule(kind):
    if kind == "mean_variance":
        rule = _MeanVariance(
            MeanVarianceConfig(
                expected_return_label="ret_5",
                risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=20)),
                ic=0.05,
                risk_aversion=5.0,
                turnover_penalty=0.002,
                weight_cap=0.4,
            )
        )
    else:
        rule = (_WithFactor if kind == "factor" else _TopN)(TopNConfig(direction="long_only", top_n=3))
    rule.bind(SPECS)
    rule.seen = []
    return rule


def _assert_same_context(built: PortfolioContext, looped: PortfolioContext):
    assert built.timestamp == looped.timestamp
    for name in ("predictions", "tradable", "current_weights", "returns", "staleness", "factors"):
        a, b = getattr(built, name), getattr(looped, name)
        if b is None:
            assert a is None, name
        else:
            xr.testing.assert_identical(a, b)


@pytest.mark.parametrize("kind", ["top_n", "mean_variance", "factor"])
def test_context_reproduces_every_rebalance_bar_of_weights(kind):
    rule = _rule(kind)
    inputs = _inputs(rule)
    predictions = _predictions()

    panel = inputs.weights(predictions)

    looped = list(rule.seen)
    assert len(looped) == rebalance_mask(BARS, PERIODS).sum()
    assert any((c.current_weights.values != 0).any() for c in looped)
    assert any(bool(c.locked.any()) for c in looped), "the replay must cover a locked position"
    for context in looped:
        t = context.timestamp
        built = inputs.context(t, predictions.sel(timestamp=t), context.current_weights)
        _assert_same_context(built, context)
        decision = rule.decide(built)
        assert decision.failure is None
        np.testing.assert_array_equal(decision.weights.values, panel["weight"].sel(timestamp=t).values)
    if kind == "mean_variance":
        assert all(c.returns.sizes["timestamp"] == 20 for c in looped)
    if kind == "factor":
        assert all(c.factors is not None for c in looped)


def test_a_held_symbol_without_a_prediction_joins_the_bar_locked():
    rule = _rule("top_n")
    t = TS[WARMUP + 12]  # S1 halted
    predictions = _predictions().sel(timestamp=t).drop_sel(symbol="S1")
    held = xr.DataArray([0.4], dims="symbol", coords={"symbol": ["S1"]})

    context = _inputs(rule).context(t, predictions, held)

    assert context.symbols.tolist() == [s for s in SYMBOLS if s != "S1"] + ["S1"]
    assert np.isnan(context.predictions["ret_5"].sel(symbol="S1"))
    assert context.locked.sel(symbol="S1")
    assert rule.decide(context).weights.sel(symbol="S1") == 0.4


def test_rebalances_is_the_schedule_weights_follows():
    rule = _rule("top_n")
    inputs = _inputs(rule)
    predictions = _predictions()

    panel = inputs.weights(predictions)

    decided = np.isfinite(panel["weight"].values).all(axis=1)
    asked = np.array([inputs.rebalances(t) for t in TS[WARMUP:]])
    np.testing.assert_array_equal(asked, rebalance_mask(BARS, PERIODS))
    assert not (decided & ~asked).any()
    assert not inputs.rebalances(TS[WARMUP - 1])  # before the anchor
    assert not _inputs(rule, end=TS[WARMUP + 3]).rebalances(TS[WARMUP + 3])


def test_a_panel_starting_after_the_anchor_keeps_the_anchors_schedule():
    rule = _rule("top_n")
    inputs = _inputs(rule, anchor=TS[WARMUP - 2])
    predictions = _predictions()

    inputs.weights(predictions)

    decided = [c.timestamp for c in rule.seen]
    expected = [t for i, t in enumerate(TS[WARMUP - 2 :]) if i % PERIODS == 0 and t >= TS[WARMUP]]
    assert decided == [pd.Timestamp(t) for t in expected if t != TS[-1]]


def test_only_weights_warns_about_a_short_history():
    rule = _rule("mean_variance")  # history_bars 26: 25 bars before the first
    short = _dataset(_price().isel(timestamp=slice(WARMUP - 10, None)))
    inputs = _inputs(rule, short)
    predictions = _predictions()

    with pytest.warns(UserWarning, match=r"reads 25 bar\(s\) of prices .* holds only 10; .* short by 15"):
        inputs.weights(predictions)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = TS[WARMUP]
        inputs.context(t, predictions.sel(timestamp=t), xr.zeros_like(predictions["ret_5"].sel(timestamp=t)))


def test_refusals():
    rule = _rule("top_n")
    inputs = _inputs(rule)
    predictions = _predictions()
    t = TS[WARMUP + 1]
    bar = predictions.sel(timestamp=t)
    flat = xr.zeros_like(bar["ret_5"])

    with pytest.raises(ValueError, match="before the anchor"):
        _inputs(rule, anchor=TS[WARMUP + 1]).weights(predictions)
    with pytest.raises(ValueError, match="not a bar"):
        inputs.context(pd.Timestamp("2024-01-06"), bar, flat)  # a Saturday
    with pytest.raises(ValueError, match="finite"):
        inputs.context(t, bar, flat.where(flat.symbol != "S0"))
    with pytest.raises(ValueError, match="one bar"):
        inputs.context(t, predictions, flat)
    with pytest.raises(ValueError, match="rebalance_periods"):
        DecisionInputs(_dataset(), rule, fill_column="open", valuation_column="close", rebalance_periods=0, anchor=t)


def test_factor_panels_lacking_a_declared_name_are_refused():
    class _Other(_LogPrice):
        def get_factor_names(self):
            return ["beta"]

    class _Declares(_TopN):
        def required_factors(self):
            return [_Other()]

    rule = _Declares(TopNConfig(direction="long_only", top_n=1))
    rule.seen = []
    with pytest.raises(ValueError, match=r"\['beta'\]"):
        _inputs(rule).weights(_predictions())


# -- one module assembles the inputs --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = "quantlab/portfolio/decision_inputs.py"
CALLERS = ("quantlab/backtest/predefined/us_equity.py", "quantlab/base/backtest.py", "quantlab/api/_backtest.py")
#: Names only the module may use to assemble decision inputs.
ASSEMBLY = {"tradable_bars", "rebalance_mask", "bar_before", "required_factors", "history_bars", "lookback_bars", "ExecutionBook"}


def _names(path: Path) -> set[str]:
    """Every name and attribute the code of ``path`` refers to (docstrings are not code)."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
    return names


def test_no_context_is_built_outside_the_module():
    builders = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "quantlab").rglob("*.py")
        if any(
            isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "PortfolioContext"
            for node in ast.walk(ast.parse(path.read_text()))
        )
    )
    assert builders == [MODULE]


@pytest.mark.parametrize("caller", CALLERS)
def test_the_callers_hold_no_assembly_wiring(caller):
    assert _names(REPO_ROOT / caller) & ASSEMBLY == set()


def test_the_rules_hold_no_assembly_code():
    from quantlab.base.portfolio import PortfolioConstructor

    assert not hasattr(PortfolioConstructor, "build_context")
    assert not hasattr(PortfolioConstructor, "construct_panel")
