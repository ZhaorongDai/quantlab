"""Components are serialised and rebuilt from one declaration (#130).

A config dataclass marks which of its fields hold components with
`quantlab.base.component.component`. One generic `to_dict` writes each declared field
as the component's own config (its `"name"` the import path), and one generic
`from_config(d, run_dir=None)` rebuilds by recursing along those fields only, threading
`run_dir` to every level. What is locked here:

- every dataset, factor and label kind round-trips: `from_config(get_config(x))` (and
  the dispatching `rebuild`) rebuilds an equal object, through a JSON trip;
- a free-form dict holding a `"name"` key stays data;
- an unknown key in a saved config is refused;
- a `FrameDataset` recorded relative to a run directory, nested under a factor or a
  label, rebuilds given `run_dir` and is refused without it;
- a class without a config class is never rebuilt.
"""

import dataclasses
import json
from pathlib import Path

import pytest
import xarray as xr

from quantlab.base.component import Component, component, rebuild
from quantlab.base.config import (
    ConstituentDatasetConfig,
    DatasetConfig,
    FactorConfig,
    ForwardConfig,
    MarketFeatureConfig,
    PolarsFactorConfig,
)
from quantlab.dataset.constituent import SP500ConstituentDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.market import MarketFeatures
from quantlab.label.forward import Forward
from quantlab.label.predefined.fret import BinaryReturn, Return, Volatility
from tests.backtest_fixtures import ADJUSTED_COLUMNS, PastReturnFactor, write_price_store


def _json(config: dict) -> dict:
    return json.loads(json.dumps(config))


@pytest.fixture
def stock(tmp_path) -> StockDataset:
    return StockDataset(write_price_store(tmp_path / "stock"))


def _one_symbol(stock: StockDataset, symbol: str, path: Path) -> StockDataset:
    xr.open_zarr(stock.config.zarr_file_path).sel(symbol=[symbol]).load().to_zarr(path)
    return StockDataset(dataclasses.replace(stock.config, zarr_file_path=str(path)))


def _past_return(dataset, **kwargs) -> PastReturnFactor:
    return PastReturnFactor(
        PolarsFactorConfig(warmup_bars=3, dataset=dataset, kwargs={"n": 2, **kwargs})
    )


KINDS = {
    "stock dataset": lambda stock, tmp: stock,
    "constituent dataset": lambda stock, tmp: SP500ConstituentDataset(
        ConstituentDatasetConfig(
            zarr_file_path=str(tmp / "sp500.zarr"), cache_dir=str(tmp / "_cache")
        )
    ),
    "merged dataset": lambda stock, tmp: MergedDataset(
        [_one_symbol(stock, "AAA", tmp / "a.zarr"), _one_symbol(stock, "BBB", tmp / "b.zarr")]
    ),
    "frame dataset on a store": lambda stock, tmp: FrameDataset(
        xr.open_zarr(stock.config.zarr_file_path).load()
    ).to_zarr(tmp / "held.zarr"),
    "spot dataset": lambda stock, tmp: SpotKlineDataset(
        dataclasses.replace(stock.config, market="crypto_spot")
    ),
    "polars factor": lambda stock, tmp: _past_return(stock),
    "factor over several datasets": lambda stock, tmp: _past_return(
        (_one_symbol(stock, "AAA", tmp / "a.zarr"), _one_symbol(stock, "BBB", tmp / "b.zarr"))
    ),
    "alpha158 factor": lambda stock, tmp: Alpha158Stock(
        FactorConfig(
            warmup_bars=60, dataset=stock, mode="batch", data_columns=ADJUSTED_COLUMNS
        )
    ),
    "kunquant factor": lambda stock, tmp: Alpha101Stock(
        FactorConfig(
            warmup_bars=10,
            dataset=stock,
            mode="batch",
            data_columns=ADJUSTED_COLUMNS,
            factor_names=("alpha001",),
        )
    ),
    "market features": lambda stock, tmp: MarketFeatures(
        MarketFeatureConfig(
            dataset=stock, series={"spy": _one_symbol(stock, "AAA", tmp / "spy.zarr")}
        )
    ),
    "forward label": lambda stock, tmp: Forward(
        ForwardConfig(factor=_past_return(stock), span=2)
    ),
    **{
        f"{label.__name__} label": (
            lambda stock, tmp, label=label: label(
                FactorConfig(
                    warmup_bars=0,
                    dataset=stock,
                    mode="batch",
                    data_columns=("adjOpen",),
                    kwargs={"n_forward_periods": 5},
                )
            )
        )
        for label in (Return, BinaryReturn, Volatility)
    },
}


@pytest.mark.parametrize("kind", KINDS)
def test_every_dataset_factor_and_label_round_trips(kind, stock, tmp_path):
    original = KINDS[kind](stock, tmp_path)
    saved = _json(original.get_config())

    by_class = type(original).from_config(saved)
    by_name = rebuild(saved)

    assert type(by_class) is type(original) and by_class == original
    assert type(by_name) is type(original) and by_name == original
    assert _json(by_name.get_config()) == saved


def test_rebuild_leaves_the_saved_dict_unchanged(stock):
    saved = _json(Forward(ForwardConfig(factor=_past_return(stock), span=2)).get_config())
    before = json.dumps(saved, sort_keys=True)

    rebuild(saved)

    assert json.dumps(saved, sort_keys=True) == before


def test_a_free_form_dict_holding_name_stays_data(stock):
    spec = {"name": "quantlab.dataset.stock.StockDataset", "zarr_file_path": "x.zarr"}
    factor = _past_return(stock, spec=spec)

    rebuilt = rebuild(_json(factor.get_config()))

    assert rebuilt.config.kwargs["spec"] == spec
    assert rebuilt == factor


def test_an_unknown_key_is_refused(stock):
    saved = _json(_past_return(stock).get_config())

    with pytest.raises(ValueError, match="unknown key.*'colour'"):
        rebuild({**saved, "colour": "red"})
    with pytest.raises(ValueError, match="unknown key.*'colour'"):
        rebuild({**saved, "dataset": {**saved["dataset"], "colour": "red"}})


def _recorded_frame_factor(stock, run_dir: Path) -> dict:
    """A label over a factor over a `FrameDataset` recorded relative to `run_dir`."""
    held = FrameDataset(xr.open_zarr(stock.config.zarr_file_path).load())
    label = Forward(ForwardConfig(factor=_past_return(held), span=2))
    saved = _json(label.get_config())
    saved["factor"]["dataset"] = held.persist_with_run(run_dir, "factor_dataset")
    return saved


def test_a_nested_frame_dataset_rebuilds_against_run_dir(stock, tmp_path):
    run_dir = tmp_path / "run"
    saved = _recorded_frame_factor(stock, run_dir)
    assert saved["factor"]["dataset"]["zarr_file_path"] == "inputs/factor_dataset.zarr"

    moved = tmp_path / "moved"
    run_dir.rename(moved)
    rebuilt = rebuild(saved, run_dir=moved)

    dataset = rebuilt.config.factor.config.dataset
    assert type(dataset) is FrameDataset
    assert dataset.config.zarr_file_path == str(moved / "inputs/factor_dataset.zarr")
    xr.testing.assert_equal(
        dataset.panel("2024-01-01", "2024-03-29"),
        stock.panel("2024-01-01", "2024-03-29"),
    )


def test_a_nested_frame_dataset_without_run_dir_is_refused(stock, tmp_path):
    saved = _recorded_frame_factor(stock, tmp_path / "run")

    with pytest.raises(ValueError, match="run_dir"):
        rebuild(saved)


def test_a_class_without_a_config_class_is_never_rebuilt():
    with pytest.raises(TypeError, match="declares no config_cls"):
        rebuild({"name": "pathlib.Path"})


# -- the declaration itself, on a toy component ------------------------------------------


@dataclasses.dataclass(frozen=True)
class _LeafConfig:
    size: int


class _Leaf(Component):
    config_cls = _LeafConfig

    def __init__(self, config):
        self.config = config

    def __eq__(self, other):
        return type(other) is type(self) and other.config == self.config


@dataclasses.dataclass(frozen=True)
class _TreeConfig:
    leaf: _Leaf | None = component(default=None)
    leaves: list = component(many=True, default_factory=list)
    named: dict = component(many=True, default_factory=dict)
    options: dict = dataclasses.field(default_factory=dict)


class _Tree(_Leaf):
    config_cls = _TreeConfig


def test_declared_fields_serialise_as_component_configs_and_rebuild():
    tree = _Tree(
        _TreeConfig(
            leaf=_Leaf(_LeafConfig(1)),
            leaves=[_Leaf(_LeafConfig(2)), _Leaf(_LeafConfig(3))],
            named={"a": _Leaf(_LeafConfig(4))},
            options={"name": "not.a.Component", "x": 1},
        )
    )
    leaf_name = f"{__name__}._Leaf"

    saved = _json(tree.get_config())

    assert saved == {
        "leaf": {"size": 1, "name": leaf_name},
        "leaves": [{"size": 2, "name": leaf_name}, {"size": 3, "name": leaf_name}],
        "named": {"a": {"size": 4, "name": leaf_name}},
        "options": {"name": "not.a.Component", "x": 1},
        "name": f"{__name__}._Tree",
    }
    assert _Tree.from_config(saved) == tree


def test_an_empty_component_field_stays_empty():
    saved = _json(_Tree(_TreeConfig()).get_config())

    assert saved["leaf"] is None
    assert _Tree.from_config(saved) == _Tree(_TreeConfig())
