"""Persisted dataset/factor configs rebuild with the config class their class declares (D-26).

Before this lock, `quantlab/utils/module.py` hardcoded the config class:
`load_dataset_from_config` always built `DatasetConfig(**config)` and
`load_factor_from_config` always built `FactorConfig(**config)`. Two shipped
hierarchies could therefore not be rebuilt from their own saved config, and
03.7-CONTEXT.md D-26 records both TypeErrors verbatim:

- a `PolarsFactorConfig` dict ->
  `TypeError: FactorConfig.__init__() missing 2 required keyword-only
  arguments: 'mode' and 'data_columns'`;
- a `ConstituentDatasetConfig` dict ->
  `TypeError: DatasetConfig.__init__() got an unexpected keyword argument
  'cache_dir'`.

The fix mirrors `BaseModel.config_cls`: each intermediate class declares a
plain `config_cls` class attribute and the loaders build
`cls.config_cls(**config)`. It is deliberately NOT an abstract property,
because several tests subclass `BaseDataset` and `Factor` directly and an
abstract member would make those subclasses uninstantiable.

Two conventions every test here follows:

- Both sides are JSON-normalized with `json.loads(json.dumps(cfg,
  default=str))` before comparison. Tuples serialize to lists, so a config
  compared before and after a JSON trip would differ on container type alone
  and hide nothing of interest.
- Configs are constructed directly, never through the `quantlab/config`
  package's factory functions (D-32). The round trip must not route through
  factories, whose paths point at production data and whose known defects
  are out of scope.

The round-trip tests hand each loader a deep copy of the saved dict, so a
loader that mutates its input fails `test_loaders_do_not_mutate_their_input`
rather than a round-trip test for an unrelated reason (RESEARCH Pitfall 9).
"""

import copy
import json
from pathlib import Path
from typing import Callable

import pytest

import quantlab.utils.module as module_utils
from quantlab.base.config import (
    ConstituentDatasetConfig,
    DatasetConfig,
    FactorConfig,
    MLConfig,
    PolarsFactorConfig,
)
from quantlab.base.constituent import IndexConstituentDataset
from quantlab.base.data import BaseDataset, MarketDataset
from quantlab.base.factor import Factor, FactorKunQuant, FactorPolars
from quantlab.dataset.constituent import SP500ConstituentDataset
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.momentum import Momentum
from quantlab.ml_model.xgb import XGBoostRegressor
from tests.test_model_hierarchy import FakePanel, _kwargs

_ADJUSTED_COLUMNS = ("adjHigh", "adjLow", "adjClose", "adjOpen", "adjVolume")


def _normalized(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg, default=str))


def _momentum(spot_config: DatasetConfig, tmp_path: Path) -> Momentum:
    return Momentum(
        PolarsFactorConfig(
            window=5,
            dataset=SpotKlineDataset(spot_config),
            file_path=str(tmp_path / "factors" / "momentum.zarr"),
            kwargs={"n": 5},
        )
    )


def _alpha101(stock_config: DatasetConfig, tmp_path: Path) -> Alpha101Stock:
    return Alpha101Stock(
        FactorConfig(
            window=10,
            dataset=StockDataset(stock_config),
            file_path=str(tmp_path / "factors" / "alpha101.zarr"),
            mode="batch",
            data_columns=_ADJUSTED_COLUMNS,
        )
    )


def _sp500(tmp_path: Path) -> SP500ConstituentDataset:
    return SP500ConstituentDataset(
        ConstituentDatasetConfig(
            zarr_file_path=str(tmp_path / "us_equity" / "sp500_constituent.zarr"),
            cache_dir=str(tmp_path / "reference" / "_cache"),
        )
    )


def test_polars_factor_round_trips_with_its_own_config_class(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    saved = _normalized(_momentum(spot_kline_zarr(), tmp_path).get_config())

    rebuilt = module_utils.load_factor_from_config(copy.deepcopy(saved))

    assert type(rebuilt) is Momentum
    assert type(rebuilt.config) is PolarsFactorConfig
    assert _normalized(rebuilt.get_config()) == saved


def test_kunquant_factor_round_trips(
    stock_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    saved = _normalized(_alpha101(stock_zarr(), tmp_path).get_config())

    rebuilt = module_utils.load_factor_from_config(copy.deepcopy(saved))

    assert type(rebuilt) is Alpha101Stock
    assert type(rebuilt.config) is FactorConfig
    assert _normalized(rebuilt.get_config()) == saved


def test_market_dataset_round_trips(
    stock_zarr: Callable[..., DatasetConfig],
) -> None:
    saved = _normalized(StockDataset(stock_zarr()).get_config())

    rebuilt = module_utils.load_dataset_from_config(copy.deepcopy(saved))

    assert type(rebuilt) is StockDataset
    assert type(rebuilt.config) is DatasetConfig
    assert _normalized(rebuilt.get_config()) == saved


def test_market_dataset_loads_a_config_saved_with_catalog_path(
    stock_zarr: Callable[..., DatasetConfig],
) -> None:
    """A `config.json` written before `catalog_path` was removed still loads."""
    saved = _normalized(StockDataset(stock_zarr()).get_config())
    old = {**saved, "catalog_path": "/data/catalog"}

    rebuilt = module_utils.load_dataset_from_config(old)

    assert _normalized(rebuilt.get_config()) == saved
    assert "catalog_path" in old


def test_constituent_dataset_round_trips(tmp_path: Path) -> None:
    """Offline: constructing the panel dataset performs no fetch; only
    `from_raw_data()` would reach the network, and nothing here calls it."""
    saved = _normalized(_sp500(tmp_path).get_config())

    rebuilt = module_utils.load_dataset_from_config(copy.deepcopy(saved))

    assert type(rebuilt) is SP500ConstituentDataset
    assert type(rebuilt.config) is ConstituentDatasetConfig
    assert _normalized(rebuilt.get_config()) == saved


def test_loaders_do_not_mutate_their_input(
    stock_zarr: Callable[..., DatasetConfig],
    spot_kline_zarr: Callable[..., DatasetConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller reuses the dict it saved, e.g. to serialize it again beside a
    backtest run. A loader that writes rebuilt objects back into that dict
    leaves objects inside what should be JSON (RESEARCH Pitfall 9)."""
    dataset_saved = _normalized(StockDataset(stock_zarr()).get_config())
    before = copy.deepcopy(dataset_saved)
    module_utils.load_dataset_from_config(dataset_saved)
    assert dataset_saved == before

    factor_saved = _normalized(_momentum(spot_kline_zarr(), tmp_path).get_config())
    before = copy.deepcopy(factor_saved)
    module_utils.load_factor_from_config(factor_saved)
    assert factor_saved == before

    # Same stand-in factor loader as tests/test_model_hierarchy.py: only the
    # factor rebuild is faked, the model class lookup is real.
    monkeypatch.setattr(
        module_utils,
        "load_factor_from_config",
        lambda cfg: FakePanel(cfg["factor_names"]),
    )
    model_saved = XGBoostRegressor(MLConfig(**_kwargs(tmp_path))).get_config()
    model_saved["resolved_hyperparameters"] = {"eta": 0.3}
    before = copy.deepcopy(model_saved)
    module_utils.load_model_from_config(model_saved)
    assert model_saved == before


class _ClassWithoutConfigCls:
    """Accepts a config like a real dataset/factor would, but declares no
    `config_cls`, so the loaders have no way to know which config to build."""

    def __init__(self, config):
        self.config = config


def test_a_class_without_config_cls_is_refused_by_name(
    stock_zarr: Callable[..., DatasetConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    bare_path = "tests.test_config_roundtrip._ClassWithoutConfigCls"
    real_lookup = module_utils.get_cls_from_path
    monkeypatch.setattr(
        module_utils,
        "get_cls_from_path",
        lambda path: _ClassWithoutConfigCls if path == bare_path else real_lookup(path),
    )
    dataset_saved = _normalized(StockDataset(stock_zarr()).get_config())

    bare_dataset = dict(dataset_saved, name=bare_path)
    with pytest.raises(TypeError, match="_ClassWithoutConfigCls"):
        module_utils.load_dataset_from_config(bare_dataset)

    # The nested dataset resolves to the real StockDataset, so the refusal
    # below comes from the factor loader's own check, not the dataset's.
    bare_factor = {"name": bare_path, "window": 5, "dataset": dataset_saved}
    with pytest.raises(TypeError, match="_ClassWithoutConfigCls"):
        module_utils.load_factor_from_config(bare_factor)


def test_config_cls_is_a_plain_class_attribute() -> None:
    expected = {
        MarketDataset: DatasetConfig,
        IndexConstituentDataset: ConstituentDatasetConfig,
        FactorKunQuant: FactorConfig,
        FactorPolars: PolarsFactorConfig,
    }
    for cls, config_cls in expected.items():
        assert vars(cls).get("config_cls") is config_cls, cls.__qualname__

    assert "config_cls" not in BaseDataset.__abstractmethods__
    assert "config_cls" not in Factor.__abstractmethods__
