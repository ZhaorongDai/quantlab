from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from enums.data import Frequency, Market

if TYPE_CHECKING:
    from .data import Dataset
    from .factor import Factor


@dataclass
class DatasetConfig:
    raw_data_dir_path: str
    zarr_file_path: str
    catalog_path: str
    market: Market
    frequency: Frequency
    start_date: str | None = None
    end_date: str | None = None
    symbols: tuple | None = None
    kwargs: dict | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class AcquisitionConfig:
    market: Market
    frequency: Frequency
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
    """Config for the survivorship-bias-free, point-in-time US-equity
    universe reference table (02-08-PLAN.md / 02-CONTEXT.md D-12).

    This is reference/metadata, not xarray/Zarr pipeline data (Locked
    Decision A1, 02-08-PLAN.md) -- persisted via PlBackend/parquet, same
    footing as config/instruments.yaml. Deliberately carries no
    credential/API-key field: acquisition/universe.py makes zero
    authenticated requests.
    """

    output_path: str
    cache_dir: str
    kwargs: dict | None = None
    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class BaseFactorConfig:
    """Every field shared by both factor backends (KunQuant and Polars).

    Backend-agnostic: the shared `base/factor.py:Factor` base reads only these
    fields, so a config carrying nothing beyond them is enough to construct and
    drive any factor implementation (D-03).
    """

    window: int
    dataset: "Dataset"
    file_path: str | None = None
    factor_names: list | None = None
    start_date: str | None = None
    end_date: str | None = None
    symbols: list | None = None
    kwargs: dict | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class FactorConfig(BaseFactorConfig):
    """KunQuant-backend factor configuration.

    The three fields added here are KunQuant-specific: `mode` drives the
    batch/stream branch in the KunQuant factor class, `data_columns` names the
    inputs of the compiled KunQuant graph, and `njobs` sizes
    `kr.createMultiThreadExecutor`. None of them exist on the Polars sibling,
    which is why no method of the shared `Factor` base may read them.
    """

    mode: Literal["stream", "batch"]
    data_columns: list
    njobs: int = 128


@dataclass(kw_only=True)
class PolarsFactorConfig(BaseFactorConfig):
    """Polars-backend factor configuration.

    Adds no fields to `BaseFactorConfig`. The Polars backend is batch-only
    (D-07), so it deliberately has no `mode`; factor names are resolved
    dynamically from the lazyframe schema (D-05), so `factor_names` is expected
    to stay `None` until `cal()` runs.
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
