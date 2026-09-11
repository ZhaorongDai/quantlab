from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from quantlab.enums.data import Frequency, Market, Vendor

if TYPE_CHECKING:
    from .data import MarketDataset
    from .factor import Factor


@dataclass(kw_only=True)
class BaseDatasetConfig:
    zarr_file_path: str
    start_date: str | None = None
    end_date: str | None = None
    symbols: tuple | None = None
    kwargs: dict | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class DatasetConfig(BaseDatasetConfig):
    raw_data_dir_path: str
    catalog_path: str
    #: REINSTATED by 03.5 D-02, having been dropped as unused by `441e630`.
    #: Placed beside `frequency` to mirror `AcquisitionConfig`, where the two
    #: are also adjacent and `market` comes first.
    #:
    #: Its first reader is `quantlab/acquisition/registry.py:convert()`: the
    #: capability lookup key is `(market, frequency, data_type)`, which is
    #: exactly `Capability`'s own key. Without `market` here the key would be
    #: `(frequency, data_type)` and would resolve AMBIGUOUSLY the day one
    #: vendor serves two markets at the same frequency.
    #:
    #: Deliberately OFF `BaseDatasetConfig` and pinned there by
    #: `tests/test_dataset_hierarchy.py:MARKET_ONLY_CONFIG_FIELDS`: a
    #: constituent panel has no market, and handing it one would impose a
    #: boundary it must not have.
    market: Market
    frequency: Frequency
    vendor: Vendor | None = None


@dataclass(kw_only=True)
class ConstituentDatasetConfig(BaseDatasetConfig):
    cache_dir: str
    as_of: str | None = None


@dataclass
class AcquisitionConfig:
    market: Market
    frequency: Frequency
    vendor: Vendor
    raw_data_dir_path: str
    watermark_path: str
    symbols: tuple[str, ...]
    start_date: str | None = None
    end_date: str | None = None
    kwargs: dict | None = None
    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class UniverseConfig:
    output_path: str
    cache_dir: str
    kwargs: dict | None = None
    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class BaseFactorConfig:
    window: int
    dataset: "MarketDataset"
    file_path: str | None = None
    factor_names: tuple[str, ...] | None = None
    start_date: str | None = None
    end_date: str | None = None
    symbols: tuple[str, ...] | None = None
    kwargs: dict | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class FactorConfig(BaseFactorConfig):
    mode: Literal["stream", "batch"]
    data_columns: tuple[str, ...]
    njobs: int = 128


@dataclass(kw_only=True)
class PolarsFactorConfig(BaseFactorConfig):
    """Polars-backend factor configuration.

    Adds no fields to `BaseFactorConfig`. The Polars backend is batch-only
    so it deliberately has no `mode`
    """


@dataclass
class DLConfig:
    # 数据相关
    factors: list["Factor"]
    labels: list["Factor"]
    model_save_dir: str
    factor_data_strategy: Literal["read", "cal"]
    label_data_strategy: Literal["read", "cal"]
    start_date: str | None = None
    end_date: str | None = None
    num_workers: int = 4

    # 回测相关
    backtest_data = None

    # 训练相关
    hyperparameters: dict = field(default_factory=dict)
    lr: float = 1e-3
    lr_refit: float = 0.0
    epochs: int = 100
    early_stopping: bool = False
    early_stopping_patience: int = 5
    batch_size: int = 1024
    val_size: float = 0.2
    random_seed: int = 42
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class MLConfig:
    # 数据相关
    factors: list["Factor"]
    labels: list["Factor"]
    model_save_dir: str
    factor_data_strategy: Literal["read", "cal"]
    label_data_strategy: Literal["read", "cal"]
    start_date: str | None = None
    end_date: str | None = None

    # 回测相关
    backtest_data = None

    # 训练相关
    hyperparameters: dict = field(default_factory=dict)
    val_size: float = 0.2
    random_seed: int = 42
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)
