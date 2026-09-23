"""Dataclass configurations threaded through every layer of the pipeline.

Each domain object (dataset, factor, model, backtester, acquisition, universe
catalog) is constructed from one of the config classes here and exposes it as
``self.config``. The object's config setter fills in derived values on
assignment, such as inherited dates and the ``name`` field, which records the
owning class's dotted import path so the object can be rebuilt from the
serialised dict (see ``quantlab.utils.module``). ``to_dict()`` on each config
produces that dict.

Fields are documented with ``#:`` comments so the meaning of each one sits
beside its definition.
"""

from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Literal

from quantlab.enums.data import BarInterval, Frequency, Market, Vendor

if TYPE_CHECKING:
    from .data import MarketDataset
    from .factor import Factor
    from .model import BaseModel


@dataclass(kw_only=True)
class BaseDatasetConfig:
    """Fields every dataset shares, whatever it holds.

    Both market panels and constituent (index membership) panels build on
    this class. Market-specific fields live on ``DatasetConfig``.
    """

    #: Path of the Zarr store the dataset reads from and writes to.
    zarr_file_path: str
    #: First date to keep, inclusive. ``None`` means no lower bound.
    start_date: str | None = None
    #: Last date to keep, inclusive. ``None`` means no upper bound.
    end_date: str | None = None
    #: Restrict the panel to these symbols. ``None`` means every symbol.
    symbols: tuple | None = None
    #: Free-form options a specific dataset class may read (for example
    #: ``data_type`` for tick data). ``None`` is treated as empty.
    kwargs: dict | None = None

    #: Dotted import path of the dataset class; filled by the config setter
    #: and used to rebuild the dataset from its serialised config.
    name: str | None = None

    def to_dict(self):
        """Return the config as a plain dict via ``dataclasses.asdict``."""
        return asdict(self)


@dataclass(kw_only=True)
class DatasetConfig(BaseDatasetConfig):
    """Config of a market data panel built from a raw download tree.

    ``market`` and ``frequency`` sit here, not on ``BaseDatasetConfig``,
    because a constituent panel has neither. Together with the ``data_type``
    entry of ``kwargs`` they form the key the vendor registry uses to pick the
    converter for this dataset.
    """

    #: Root of the raw download tree the panel is converted from.
    raw_data_dir_path: str
    #: Directory of the Nautilus Trader data catalog the dataset can export to.
    catalog_path: str
    #: The market this panel belongs to.
    market: Market
    #: The acquisition frequency of the raw data.
    frequency: Frequency
    #: The vendor the raw data was downloaded from, when it matters for
    #: conversion.
    vendor: Vendor | None = None


@dataclass(kw_only=True)
class NbboDatasetConfig(DatasetConfig):
    """Config of the WRDS TAQ NBBO bar panel.

    ``frequency`` stays ``"tick"`` (the raw tier holds one row per NBBO
    record); the size of the bars the panel is resampled to is the separate
    ``bar_interval``. The four ``drop_*`` and ``keep_*`` fields are the
    resampler's record filter and are config fields, not ``kwargs``, so a
    rebuild from ``config.json`` reproduces the panel exactly.
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: Always tick: the raw tier is one row per NBBO record.
    frequency: Frequency = "tick"
    #: Always WRDS for this dataset.
    vendor: Vendor | None = "wrds"
    #: Bar size the tick records are resampled to.
    bar_interval: BarInterval = "1m"
    #: Start of the session window, US/Eastern wall clock ``HH:MM``.
    session_start: str = "09:30"
    #: End of the session window, US/Eastern wall clock ``HH:MM``.
    session_end: str = "16:00"
    #: Drop records whose bid exceeds the ask (both sides present).
    drop_crossed: bool = True
    #: Drop records whose bid equals the ask.
    drop_locked: bool = False
    #: Drop records carrying a non-positive price.
    drop_nonpositive_price: bool = True
    #: Keep only these ``qu_cond`` quote conditions. ``None`` keeps every
    #: condition.
    keep_qu_cond: tuple[str, ...] | None = None


#: PERMNO of the QQQ ETF. A module constant because it is also the value a user
#: passes to ``permnos`` when they want the ETF in some other window.
QQQ_PERMNO: str = "86755"


@dataclass(kw_only=True)
class CrspDatasetConfig(DatasetConfig):
    """Config of the CRSP Stock v2 daily panel.

    The three market fields default rather than being asked for: CRSP Stock v2
    is US equity, daily, and reached through the ``wrds`` account. They remain
    fields (not constants on the dataset) because the vendor registry resolves
    the converter from ``(market, frequency, data_type)`` read off this object.

    The panel's ``symbol`` axis is the integer PERMNO, so the inherited
    ticker-side ``symbols`` field is refused by the dataset's config setter;
    use ``permnos`` instead. See ``docs/wrds_crsp.md``.
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: Always daily for this vendor.
    frequency: Frequency = "1d"
    #: Always WRDS for this dataset.
    vendor: Vendor | None = "wrds"

    #: Directory of the CRSP reference tables (``stksecurityinfohist`` and
    #: friends) the conversion reads its symbology from. Required and not
    #: derived from ``raw_data_dir_path``: the reference tier is a sibling of
    #: the raw root pulled by a separate step, and a conversion pointed at a
    #: raw tree whose sibling was never filled must fail saying so.
    reference_dir: str

    #: Restrict the conversion to these PERMNOs, as digit strings. ``None``
    #: means every PERMNO present in the raw tier; an empty tuple is refused at
    #: config assignment because it could mean either "none" or "all". A
    #: PERMNO listed here is an explicit roster: every one of its rows survives
    #: ``security_filter`` regardless of its share or security type, and the
    #: override is recorded under ``roster_overrides`` in the filter report
    #: written next to the store.
    permnos: tuple[str, ...] | None = None

    #: Which securities the panel holds: either the name of a preset
    #: (``"equity_common"``, ``"shrcd_10_11"``, ``"none"``) or an explicit
    #: ``{column: allowed values}`` mapping over the filterable CRSP type
    #: columns. The default keeps common stock, including REITs and non-US
    #: incorporated issuers, and drops ADRs, units, funds, ETFs and unknown
    #: types. The predicate is evaluated per date against the daily type
    #: columns, so a security that changed type keeps only the era in which it
    #: qualified. Filtering happens at conversion time, never in the raw tier,
    #: so a re-filter is a re-conversion rather than a re-download.
    security_filter: str | dict = "equity_common"

    #: Name of an index (one of ``CrspMembership.INDEXES``) whose point-in-time
    #: membership acts as an explicit roster for this conversion. During a
    #: PERMNO's membership spell it is exempt from ``security_filter``; outside
    #: its spells the filter applies normally. Overrides are recorded under
    #: ``roster_overrides`` in the filter report. ``None`` means no index roster.
    roster_universe: str | None = None

    @classmethod
    def qqq_benchmark(
        cls,
        *,
        zarr_file_path: str,
        raw_data_dir_path: str,
        catalog_path: str,
        reference_dir: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "CrspDatasetConfig":
        """Return the config of a store holding only the QQQ ETF.

        The benchmark gets its own store rather than one more symbol in the
        equity panel: everything in the panel enters cross-sectional ranking
        and model training, and an ETF ranked against its own constituents is
        the index competing with itself. Two settings are fixed here because
        getting either wrong is silent: ``permnos=(QQQ_PERMNO,)`` selects the
        ETF alone, and ``security_filter="none"`` is required because QQQ is a
        fund, which the default filter drops.

        This store is data only; ``BacktestConfig.benchmark_dataset`` does not
        yet consume it.

        Args:
            zarr_file_path: Path of the benchmark's own Zarr store.
            raw_data_dir_path: Root of the CRSP raw download tree.
            catalog_path: Nautilus catalog directory (unused by CRSP but
                required by ``DatasetConfig``).
            reference_dir: Directory of the CRSP reference tables.
            start_date: First date to keep, inclusive.
            end_date: Last date to keep, inclusive.
        """
        return cls(
            zarr_file_path=zarr_file_path,
            raw_data_dir_path=raw_data_dir_path,
            catalog_path=catalog_path,
            reference_dir=reference_dir,
            start_date=start_date,
            end_date=end_date,
            permnos=(QQQ_PERMNO,),
            security_filter="none",
        )


@dataclass(kw_only=True)
class ConstituentDatasetConfig(BaseDatasetConfig):
    """Config of an index-membership panel (a boolean mask over time and symbol)."""

    #: Directory holding the membership source the panel is built from (the
    #: universe catalog cache or the CRSP reference tables).
    cache_dir: str
    #: Date the membership is observed from. ``None`` means today, so a panel
    #: that must be reproducible later should pin it.
    as_of: str | None = None


@dataclass
class AcquisitionConfig:
    """Config of a raw-data download for one market, frequency and vendor."""

    #: The market being downloaded.
    market: Market
    #: The acquisition frequency (daily, minute or tick).
    frequency: Frequency
    #: The vendor the data is fetched from.
    vendor: Vendor
    #: Root of the raw tree the shards are written into.
    raw_data_dir_path: str
    #: Directory holding the per-symbol watermarks that let an interrupted
    #: download resume.
    watermark_path: str
    #: The symbols to download.
    symbols: tuple[str, ...]
    #: First date to fetch, inclusive. ``None`` means the vendor's earliest.
    start_date: str | None = None
    #: Last date to fetch, inclusive. ``None`` means the latest available.
    end_date: str | None = None
    #: Vendor-specific options (batch sizes, feeds, ``data_type``). ``None``
    #: is treated as empty.
    kwargs: dict | None = None
    #: Dotted import path of the acquisition class; filled by the config
    #: setter.
    name: str | None = None

    def to_dict(self):
        """Return the config as a plain dict via ``dataclasses.asdict``."""
        return asdict(self)


@dataclass
class UniverseConfig:
    """Config of the point-in-time universe catalog builder."""

    #: File the built catalog is written to.
    output_path: str
    #: Directory where downloaded reference files are cached.
    cache_dir: str
    #: Builder-specific options. ``None`` is treated as empty.
    kwargs: dict | None = None
    #: Dotted import path of the catalog class; filled by the config setter.
    name: str | None = None

    def to_dict(self):
        """Return the config as a plain dict via ``dataclasses.asdict``."""
        return asdict(self)


@dataclass(kw_only=True)
class BaseFactorConfig:
    """Fields every factor (and label) shares, whichever backend computes it.

    Dates and symbols left ``None`` are inherited from ``dataset`` by the
    factor's config setter. The factor reads a window of ``window`` days before
    ``start_date`` so rolling computations are warm at the first requested
    bar.
    """

    #: Lookback, in calendar days, read before ``start_date`` to warm up
    #: rolling computations.
    window: int
    #: The market dataset the factor is computed from.
    dataset: "MarketDataset"
    #: Path of the Zarr store the computed factor values are saved to and
    #: read back from.
    file_path: str | None = None
    #: Names of the factor variables this factor produces; filled from the
    #: factor definition when left ``None``.
    factor_names: tuple[str, ...] | None = None
    #: First date to compute, inclusive. ``None`` inherits the dataset's.
    start_date: str | None = None
    #: Last date to compute, inclusive. ``None`` inherits the dataset's.
    end_date: str | None = None
    #: Restrict computation to these symbols. ``None`` inherits the dataset's.
    symbols: tuple[str, ...] | None = None
    #: Free-form options a specific factor class may read.
    kwargs: dict | None = None

    #: Dotted import path of the factor class; filled by the config setter.
    name: str | None = None

    def to_dict(self):
        """Return the config as a plain dict via ``dataclasses.asdict``."""
        return asdict(self)


@dataclass(kw_only=True)
class FactorConfig(BaseFactorConfig):
    """Config of a KunQuant-computed factor."""

    #: ``"batch"`` compiles the graph for a full historical window;
    #: ``"stream"`` compiles it for incremental per-bar updates.
    mode: Literal["stream", "batch"]
    #: The dataset variables fed into the compiled graph as inputs.
    data_columns: tuple[str, ...]
    #: Threads used by the KunQuant executor.
    njobs: int = 128


@dataclass(kw_only=True)
class PolarsFactorConfig(BaseFactorConfig):
    """Config of a Polars-computed factor.

    Adds no fields to ``BaseFactorConfig``. The Polars backend is batch-only,
    so there is no ``mode``.
    """


@dataclass
class DLConfig:
    """Config of a torch model head trained through the epoch loop.

    ``train_start``, ``train_end``, ``test_start`` and ``test_end`` bound the
    training and test windows; rolling cross-validation overwrites them fold
    by fold. See ``docs/model.md``.
    """

    #: The factors whose values form the model's input features.
    factors: list["Factor"]
    #: The factors (labels) whose values form the prediction targets.
    labels: list["Factor"]
    #: Root directory checkpoints and their ``config.json`` are written under.
    model_save_dir: str
    #: ``"read"`` loads factor values from their stores; ``"cal"`` computes
    #: them first.
    factor_data_strategy: Literal["read", "cal"]
    #: ``"read"`` loads label values from their stores; ``"cal"`` computes
    #: them first.
    label_data_strategy: Literal["read", "cal"]
    #: First date of data to collect, inclusive.
    start_date: str | None = None
    #: Last date of data to collect, inclusive.
    end_date: str | None = None
    #: Worker processes for the torch ``DataLoader``.
    num_workers: int = 4

    #: Architecture-specific hyperparameters passed to the model head.
    hyperparameters: dict = field(default_factory=dict)
    #: Learning rate of the main training run.
    lr: float = 1e-3
    #: Learning rate for online refitting during prediction; ``0.0`` disables
    #: refitting.
    lr_refit: float = 0.0
    #: Maximum number of training epochs.
    epochs: int = 100
    #: Stop when the validation loss has not improved for
    #: ``early_stopping_patience`` epochs and roll back to the best epoch.
    early_stopping: bool = False
    #: Epochs without improvement tolerated before early stopping triggers.
    early_stopping_patience: int = 5
    #: Batch size of the ``DataLoader``.
    batch_size: int = 1024
    #: Fraction of the training window held out, at its end, for validation.
    val_size: float = 0.2
    #: Seed applied to Python, numpy and torch before training.
    random_seed: int = 42
    #: First date of the training window, inclusive.
    train_start: str | None = None
    #: Last date of the training window, inclusive.
    train_end: str | None = None
    #: First date of the test window, inclusive.
    test_start: str | None = None
    #: Last date of the test window, inclusive.
    test_end: str | None = None

    #: Dotted import path of the model class; filled by the config setter.
    name: str | None = None

    def to_dict(self):
        """Return the config as a plain dict via ``dataclasses.asdict``."""
        return asdict(self)


@dataclass
class MLConfig:
    """Config of a tree or other non-torch model head.

    There is no ``epochs`` field: an ML head has no outer epoch loop, and
    ``early_stopping_patience`` counts boosting rounds (or the library's own
    unit) through the library's native early stopping. See ``docs/model.md``.
    """

    #: The factors whose values form the model's input features.
    factors: list["Factor"]
    #: The factors (labels) whose values form the prediction targets.
    labels: list["Factor"]
    #: Root directory checkpoints and their ``config.json`` are written under.
    model_save_dir: str
    #: ``"read"`` loads factor values from their stores; ``"cal"`` computes
    #: them first.
    factor_data_strategy: Literal["read", "cal"]
    #: ``"read"`` loads label values from their stores; ``"cal"`` computes
    #: them first.
    label_data_strategy: Literal["read", "cal"]
    #: First date of data to collect, inclusive.
    start_date: str | None = None
    #: Last date of data to collect, inclusive.
    end_date: str | None = None

    #: Library hyperparameters passed to the model head.
    hyperparameters: dict = field(default_factory=dict)
    #: Enable the library's native early stopping on the validation split.
    early_stopping: bool = False
    #: Rounds without improvement tolerated before early stopping triggers.
    early_stopping_patience: int = 5
    #: Fraction of the training window held out, at its end, for validation.
    val_size: float = 0.2
    #: Seed applied before training.
    random_seed: int = 42
    #: First date of the training window, inclusive.
    train_start: str | None = None
    #: Last date of the training window, inclusive.
    train_end: str | None = None
    #: First date of the test window, inclusive.
    test_start: str | None = None
    #: Last date of the test window, inclusive.
    test_end: str | None = None

    #: Dotted import path of the model class; filled by the config setter.
    name: str | None = None

    def to_dict(self):
        """Return the config as a plain dict via ``dataclasses.asdict``."""
        return asdict(self)


@dataclass(kw_only=True)
class BacktestConfig:
    """Fields every backtester shares.

    Selection parameters are not here but on subclasses such as
    ``CrossSectionBacktestConfig``, so a time-series backtester does not carry
    cross-sectional fields. See ``docs/backtest.md``.
    """

    #: The dataset whose prices the simulation trades on.
    price_dataset: "MarketDataset"
    #: The model that produces the scores the target weights are built from.
    model: "BaseModel"
    #: ``"train"`` trains ``model`` on its own dates first; ``"load"`` restores
    #: a checkpoint (``checkpoint`` for ``run()``, ``cv_project_dir`` for
    #: ``run_cv()``).
    model_mode: Literal["train", "load"]

    #: First date of the backtest window, inclusive.
    start_date: str
    #: Last date of the backtest window, inclusive.
    end_date: str
    #: Directory each run writes its own run directory under.
    output_dir: str

    #: Rebalance every this many bars.
    rebalance_periods: int

    #: Checkpoint to restore in ``"load"`` mode for ``run()``.
    checkpoint: str | None = None
    #: Directory of a ``train_cv`` run, read by ``run_cv()`` to replay each
    #: fold with its own checkpoint.
    cv_project_dir: str | None = None

    #: Proportional fee per trade.
    fees: float = 0.0005
    #: Proportional slippage per trade.
    slippage: float = 0.0005
    #: Starting cash of the simulated portfolio.
    init_cash: float = 1_000_000.0

    #: Reserved for a benchmark price series; supplying one currently raises
    #: ``NotImplementedError`` in the backtester's config setter.
    benchmark_dataset: "MarketDataset | None" = None

    #: Log the run to Weights & Biases.
    use_wandb: bool = False

    #: Dotted import path of the backtester class; filled by the config setter.
    name: str | None = None

    #: Fields holding live objects. ``to_dict`` skips them; the backtester's
    #: ``get_config`` nests each one's own config instead.
    _OBJECT_FIELDS = ("price_dataset", "model", "benchmark_dataset")

    def to_dict(self):
        """Return only the scalar fields as a dict.

        ``dataclasses.asdict`` is avoided on purpose: it would deep-copy the
        panels already read into memory and the trained model on every call.
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in self._OBJECT_FIELDS
        }


@dataclass(kw_only=True)
class CrossSectionBacktestConfig(BacktestConfig):
    """Config of a cross-sectional top-N selection backtest."""

    #: ``"long_only"`` holds the top ``top_n`` names; ``"long_short"`` also
    #: shorts the bottom ``top_n``, with the two legs disjoint.
    direction: Literal["long_only", "long_short"]
    #: Number of names selected on each side at every rebalance.
    top_n: int
    #: The model label whose prediction ranks the symbols. ``None`` selects
    #: the model's first label.
    score_label: str | None = None
