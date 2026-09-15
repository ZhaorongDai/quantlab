"""Structural contract of the backtest layer (phase 03.7, plan 14).

This file is the backtest layer's equivalent of `tests/test_model_hierarchy.py`:
it keeps the architecture plan 03.7-01 built from eroding silently. The layer is

- `BaseBacktester` (`quantlab/base/backtest.py`) -- the ONLY home of `run()`;
- `VectorBtBacktester` (`quantlab/backtest/engine_vectorbt.py`) -- the engine
  layer, still abstract;
- `USEquityCrossectionSelectStockVectorBt` (`quantlab/backtest/us_equity.py`)
  -- a named composition of a market spec, a selector and an engine.

What is locked here, the decision each lock enforces, and what turns it red:

- **Exact abstract-method sets (D-01).** Each layer's `__abstractmethods__` is
  compared with a literal set. A hook that silently gains a default, a new
  abstract hook, or an engine that stops implementing one of its three hooks
  changes a set and fails the test.
- **`run()` is the single template (D-02).** No class above `BaseBacktester` in
  the concrete class's MRO may define `run` or any public function, classmethod,
  staticmethod or property of its own. An override that "only calls super()"
  is still a second implementation of the public entry, so it is red.
- **Type check first (D-01).** A concrete backtester given a plain
  `BacktestConfig` raises `TypeError` naming `CrossSectionBacktestConfig`
  before any dataset or model attribute is read. The config carries bare
  `object()`s, so if another check runs first it raises a different error and
  the test fails.
- **Price column names live only in the market spec (D-04).** No string
  constant inside any function body of `quantlab/base/backtest.py` or
  `quantlab/backtest/*.py` equals the spec's fill or valuation column name. A
  method that hardcodes the column is red. The names are read from
  `US_EQUITY_MARKET`, so the lock follows the spec.
- **No config factories (D-32).** No import in the backtest layer,
  `tests/backtest_fixtures.py` or `tests/test_backtest_*.py` resolves to
  `quantlab.config`. Relative spellings are resolved against the file's
  package, because a substring scan cannot see them.
- **One-directional layering.** No quantlab module outside
  `quantlab/backtest/`, `quantlab/base/backtest.py` and
  `quantlab/utils/module.py` (the config loader) imports the backtest layer.
- **vectorbt stays inside the engine (D-31).** No quantlab module other than
  `quantlab/backtest/engine_vectorbt.py` imports vectorbt. `quantlab/vecbt/`
  is temporarily exempt; plan 03.7-12 retires that package and removes the
  exemption.

Everything is static or construction-only: offline and CPU-only, with no store
and no model.
"""

import ast
import functools
import types
from pathlib import Path

import pytest

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.backtest import BaseBacktester
from quantlab.base.config import BacktestConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Kinds of class-dict entry that make a name part of a class's callable
#: surface. A plain class attribute (`config_cls = SomeConfig`, `MARKET = ...`)
#: is configuration, not a method, and is deliberately not listed.
_METHOD_KINDS = (
    types.FunctionType,
    classmethod,
    staticmethod,
    property,
    functools.cached_property,
    functools.partialmethod,
)


# --------------------------------------------------------------------------
# Class surface (D-01, D-02)
# --------------------------------------------------------------------------


def test_abstract_method_sets_are_exact():
    """The engine-by-inheritance hierarchy, one exact set per layer (D-01)."""
    assert BaseBacktester.__abstractmethods__ == frozenset(
        {
            "config_cls",
            "_generate_signals",
            "_simulate",
            "_simulate_benchmark",
            "_engine_stats",
        }
    )
    assert VectorBtBacktester.__abstractmethods__ == frozenset(
        {"config_cls", "_generate_signals"}
    )
    assert USEquityCrossectionSelectStockVectorBt.__abstractmethods__ == frozenset()


def test_run_lives_only_on_base_backtester():
    """`run()` is the one template (D-02): nothing above the base redefines it
    or grows a public method of its own."""
    assert "run" in vars(BaseBacktester)

    mro = USEquityCrossectionSelectStockVectorBt.__mro__
    above = mro[: mro.index(BaseBacktester)]
    assert above, "the concrete backtester must inherit from BaseBacktester"

    offenders = {}
    for cls in above:
        names = sorted(
            name
            for name, value in vars(cls).items()
            if name == "run"
            or (not name.startswith("_") and isinstance(value, _METHOD_KINDS))
        )
        if names:
            offenders[cls.__qualname__] = names
    assert offenders == {}, offenders


def test_backtest_configs_are_constructed_with_their_own_classes():
    """The config type check is the setter's first statement (D-01).

    Both object fields are bare `object()`s. Reading any dataset or model
    attribute would raise `AttributeError`, and the model type check would raise
    a `TypeError` that names `BaseModel`. Only the class check that runs first
    names `CrossSectionBacktestConfig`.
    """
    config = BacktestConfig(
        price_dataset=object(),
        model=object(),
        model_mode="train",
        start_date="2024-01-01",
        end_date="2024-02-01",
        output_dir="unused",
        rebalance_periods=1,
    )
    with pytest.raises(
        TypeError,
        match=(
            r"^USEquityCrossectionSelectStockVectorBt requires a "
            r"CrossSectionBacktestConfig, got BacktestConfig$"
        ),
    ):
        USEquityCrossectionSelectStockVectorBt(config)
