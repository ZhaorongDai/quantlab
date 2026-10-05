"""The backtester depends on the `Predictor` protocol, not on `BaseModel`.

What is locked here, and what turns it red:

- `BaseModel` satisfies `Predictor` structurally: it has every protocol
  member and does not inherit the protocol.
- An object that implements the protocol by delegation, without inheriting
  `BaseModel`, is accepted by `run()` in load and train mode and produces the
  same predictions, weights, equity curve and data fingerprints as the model
  it wraps (the same data, recorded under the wrapper's component paths).
- An object missing a protocol member is refused at construction with a
  `TypeError` naming the protocol and the missing member.
- A run's rebuild (`BacktestRun.rebuild_backtester`) rebuilds the model
  through `from_config` of the class named in the saved recipe.
- Source rule: `quantlab/backtest/base.py` and `quantlab/backtest/*.py` read
  no attribute of the model other than a protocol member (so no `config` and
  no `_`-prefixed method), and never name `ModelConfig`.

Everything is synthetic, CPU-only and offline.
"""

import ast
from pathlib import Path
from typing import get_protocol_members

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.base import Predictor
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.model.base import BaseModel
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import (
    DelegatingPredictor,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
N_BARS = 60


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _model_dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )


def _backtester(tmp_path, dataset_config, model, bars, *, name, checkpoint=None):
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=model,
            model_mode="train" if checkpoint is None else "load",
            checkpoint=None if checkpoint is None else str(checkpoint),
            start_date=_day(bars[30]),
            end_date=_day(bars[50]),
            output_dir=str(tmp_path / name / "runs"),
            rebalance_periods=5,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
            fees=0.0,
            slippage=0.0,
            init_cash=1_000_000.0,
        )
    )


def _setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    return dataset_config, bars


def _saved_fingerprints(result) -> dict:
    return BacktestRun.open(result.run_dir).data_fingerprint


def _digests(result) -> list:
    """The recorded requests and digests, whatever component path keys them."""
    return sorted(
        (str(entry["request"]), entry["digest"])
        for entries in _saved_fingerprints(result).values()
        for entry in entries
    )


def _assert_same_result(a, b) -> None:
    xr.testing.assert_identical(a.predictions, b.predictions)
    xr.testing.assert_identical(a.weights, b.weights)
    np.testing.assert_array_equal(a.simulation.value.values, b.simulation.value.values)
    assert _digests(a) == _digests(b)


def test_base_model_satisfies_the_protocol_structurally():
    members = get_protocol_members(Predictor)
    assert members == {
        "labels",
        "train_bounds",
        "test_bounds",
        "fitted_train_bounds",
        "label_delays",
        "label_scales",
        "predict_window",
        "collect",
        "train",
        "load",
        "check_checkpoint",
        "get_config",
        "from_config",
    }
    assert all(hasattr(BaseModel, name) for name in members)
    assert Predictor not in BaseModel.__mro__
    assert not hasattr(BaseModel, "_assert_trained_variables")


def test_a_delegating_predictor_matches_the_model_in_load_mode(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    dates = _model_dates(bars)
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **dates))

    plain = _backtester(
        tmp_path, dataset_config,
        make_model(tmp_path / "plain", dataset_config, **dates),
        bars, name="plain", checkpoint=checkpoint,
    ).run()
    wrapped = _backtester(
        tmp_path, dataset_config,
        DelegatingPredictor(make_model(tmp_path / "wrapped", dataset_config, **dates)),
        bars, name="wrapped", checkpoint=checkpoint,
    ).run()

    _assert_same_result(plain, wrapped)
    assert plain.metrics["training_window"] == wrapped.metrics["training_window"]


def test_a_delegating_predictor_matches_the_model_in_train_mode(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    dates = _model_dates(bars)

    plain = _backtester(
        tmp_path, dataset_config,
        make_model(tmp_path / "plain", dataset_config, **dates),
        bars, name="plain",
    ).run()
    wrapped = _backtester(
        tmp_path, dataset_config,
        DelegatingPredictor(make_model(tmp_path / "wrapped", dataset_config, **dates)),
        bars, name="wrapped",
    ).run()

    _assert_same_result(plain, wrapped)
    assert Path(wrapped.metrics["trained_checkpoint"]).is_file()
    # Training reads are the trained unit's: only labels are read for
    # training alone, and no label read is in the backtest's record.
    assert not any(".labels." in key for key in _saved_fingerprints(wrapped))


def _without(member: str) -> type:
    """A copy of `DelegatingPredictor` lacking one protocol member."""
    namespace = {
        key: value
        for key, value in vars(DelegatingPredictor).items()
        if key != member and key not in ("__dict__", "__weakref__")
    }
    return type(f"No_{member}", (), namespace)


@pytest.mark.parametrize(
    "member",
    ["predict_window", "label_delays", "label_scales", "from_config", "check_checkpoint"],
)
def test_an_object_missing_a_member_is_refused(tmp_path, member):
    dataset_config, bars = _setup(tmp_path)
    incomplete = _without(member)(
        make_model(tmp_path / "m", dataset_config, **_model_dates(bars))
    )
    with pytest.raises(TypeError, match=rf"Predictor protocol.*No_{member}.*'{member}'"):
        _backtester(tmp_path, dataset_config, incomplete, bars, name="m")


def test_rebuild_goes_through_from_config_of_the_saved_class(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    dates = _model_dates(bars)
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **dates))
    original = _backtester(
        tmp_path, dataset_config,
        DelegatingPredictor(make_model(tmp_path / "wrapped", dataset_config, **dates)),
        bars, name="wrapped", checkpoint=checkpoint,
    )
    first = original.run()

    rebuilt = BacktestRun.open(first.run_dir).rebuild_backtester()

    assert type(rebuilt.config.model).__name__ == "DelegatingPredictor"
    again = rebuilt.run()
    _assert_same_result(first, again)


# --------------------------------------------------------------------------
# Source rule: the backtest layer uses the model only through the protocol
# --------------------------------------------------------------------------


def _backtest_layer_files() -> list[Path]:
    return [REPO_ROOT / "quantlab/backtest/base.py"] + sorted(
        p for p in (REPO_ROOT / "quantlab/backtest").glob("*.py")
    )


def _is_model_expression(node: ast.AST, aliases: set[str]) -> bool:
    """`<anything>.config.model`, `config.model` or a name bound to one."""
    if isinstance(node, ast.Name):
        return node.id in aliases
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "model"
        and (
            (isinstance(node.value, ast.Name) and node.value.id == "config")
            or (isinstance(node.value, ast.Attribute) and node.value.attr == "config")
        )
    )


def _model_attribute_reads(source: str) -> list[tuple[int, str]]:
    """Every `<model>.<attr>` in `source`, as `(line, attr)`."""
    found: list[tuple[int, str]] = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        aliases: set[str] = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and _is_model_expression(node.value, set()):
                aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and _is_model_expression(
                node.value, aliases
            ):
                found.append((node.lineno, node.attr))
    return found


def test_the_rule_sees_the_ways_the_model_is_reached():
    """Guard the guard: the scanner finds reads through each spelling."""
    source = (
        "def f(self, config):\n"
        "    self.config.model.config.labels\n"
        "    config.model._feature_start()\n"
        "    model = self.config.model\n"
        "    model.config.factors\n"
    )
    reads = [attr for _, attr in _model_attribute_reads(source)]
    assert reads.count("config") == 2 and "_feature_start" in reads


def test_backtest_layer_reads_only_protocol_members_of_the_model():
    members = get_protocol_members(Predictor)
    violations = [
        f"{path.name}:{line}: {attr}"
        for path in _backtest_layer_files()
        for line, attr in _model_attribute_reads(path.read_text(encoding="utf-8"))
        if attr not in members
    ]
    assert not violations, violations


def test_backtest_layer_never_names_model_config():
    for path in _backtest_layer_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {
            alias.name
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            for alias in n.names
        }
        assert "ModelConfig" not in names, path
