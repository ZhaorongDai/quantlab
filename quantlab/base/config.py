from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Literal

from quantlab.enums.data import Frequency, Market, Vendor

if TYPE_CHECKING:
    from .data import MarketDataset
    from .factor import Factor
    from .model import BaseModel


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

    # 训练相关
    hyperparameters: dict = field(default_factory=dict)
    # 对树模型头（如 XGBoostRegressor），patience 按 boosting 轮数计，由库的原生
    # 早停执行；刻意没有 `epochs` 字段——ML 头没有外层 epoch 循环。
    early_stopping: bool = False
    early_stopping_patience: int = 5
    val_size: float = 0.2
    random_seed: int = 42
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class BacktestConfig:
    """回测器的公共配置（03.7 D-13/D-18/D-19/D-08/D-28）。

    选股相关的参数不在这里，放在子类上：D-01 预留的时序兄弟类拿到的配置不应该
    带着截面字段。
    """

    # 数据与模型
    price_dataset: "MarketDataset"
    model: "BaseModel"
    model_mode: Literal["train", "load"]

    # 回测窗口与输出
    start_date: str
    end_date: str
    output_dir: str

    # 调仓
    rebalance_periods: int

    # 模型准备
    checkpoint: str | None = None
    cv_project_dir: str | None = None

    # 成交成本与资金
    fees: float = 0.0005
    slippage: float = 0.0005
    init_cash: float = 1_000_000.0

    # 基准（本阶段不实现，D-08；保留槽位）
    benchmark_dataset: "MarketDataset | None" = None

    # 记录
    use_wandb: bool = False

    name: str | None = None

    #: 持有对象的字段。`to_dict` 跳过它们，由回测器的 `get_config` 逐个嵌套。
    _OBJECT_FIELDS = ("price_dataset", "model", "benchmark_dataset")

    def to_dict(self):
        """只返回标量字段。

        刻意不用 `asdict(self)`：它会深拷贝已经读进内存的面板和训练好的模型，
        每调一次 `get_config` 就复制一遍（03.7-RESEARCH.md Pitfall 8）。
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in self._OBJECT_FIELDS
        }


@dataclass(kw_only=True)
class CrossSectionBacktestConfig(BacktestConfig):
    """截面选股回测的配置（D-09/D-10/D-11）。没有分位数字段（D-10）。"""

    # 选股
    direction: Literal["long_only", "long_short"]
    top_n: int
    score_label: str | None = None
