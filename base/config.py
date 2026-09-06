from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from enums.data import Frequency, Market

if TYPE_CHECKING:
    from .data import MarketDataset
    from .factor import Factor


@dataclass(kw_only=True)
class BaseDatasetConfig:
    """Every field the shared `base/data.py:BaseDataset` contract reads (D-03).

    Storage-medium-agnostic and market-agnostic: each field is here because a
    method on the shared base actually consumes it, and for no other reason.

    - `zarr_file_path` is the store `read()` opens and `save()` writes.
    - `start_date` / `end_date` / `symbols` are the window `_filter()` slices
      the backend down to; the config setter fills the two dates from
      `enums.constant.Date` when the caller leaves them `None`.
    - `kwargs` is the per-subclass escape hatch `_raw_data_to_xr()`
      implementations read.
    - `name` is written by the config setter itself, to the dataset's
      `import_path`, so a saved config can be reconstructed by
      `utils/module.py:get_cls_from_path`.

    A dataset carrying nothing beyond these fields is enough to construct and
    drive any `BaseDataset` subclass, which is what lets a non-OHLCV dataset
    (an index-membership panel, say) reuse the whole storage lifecycle.
    """

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
    """Market-dataset configuration -- the config `MarketDataset` takes.

    The four fields added here are market-dataset-specific. `raw_data_dir_path`
    is the local raw-ingestion directory a market dataset parses its CSV/parquet
    source out of, and `catalog_path` is the destination
    `MarketDataset._write_catalog()` hands to nautilus's `ParquetDataCatalog`.
    Neither has any meaning for a dataset whose source is not local raw market
    files and which is never written to a nautilus catalog, which is exactly why
    no method of the shared `BaseDataset` may read them. `market` and
    `frequency` are the locked `enums/data.py` tokens that select the
    `data/{market}/{frequency}/` layout; a dataset whose calendar axis is a
    property of its own densification rather than a configured bar size does not
    have them either.

    The class name stays `DatasetConfig`: every existing call site and
    `utils/module.py`'s `DatasetConfig(**config)` checkpoint reload reference it
    by that name.
    """

    raw_data_dir_path: str
    catalog_path: str
    market: Market
    frequency: Frequency


@dataclass(kw_only=True)
class ConstituentDatasetConfig(BaseDatasetConfig):
    """Index-membership-panel dataset configuration (DATA-06, D-03).

    Adds exactly one field: `cache_dir`, the directory an index membership
    fetcher keeps its cached source snapshot in.

    What it deliberately does NOT carry, and why:

    - no `catalog_path` -- a membership panel has no bar representation and is
      never written to a nautilus `ParquetDataCatalog`;
    - no `raw_data_dir_path` -- its raw source is a remote fetch, whose only
      local footprint is the snapshot under `cache_dir`; there is no directory
      of raw market files to ingest;
    - no `market` / `frequency` -- the panel's daily calendar axis is a
      property of the interval-to-grid densification, not a configured bar
      frequency, and its index scope is chosen by the concrete dataset class
      rather than by a market token.

    Forcing those four onto a membership panel would mean supplying meaningless
    values for all of them -- the "meaningless stub" anti-pattern D-03 exists to
    eliminate.
    """

    cache_dir: str


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
    dataset: "MarketDataset"
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
