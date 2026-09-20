from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Literal

from quantlab.enums.data import BarInterval, Frequency, Market, Vendor

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
class NbboDatasetConfig(DatasetConfig):
    """Config of the WRDS TAQ NBBO bar panel (`dataset/nbbo.py`, phase 03.9).

    `frequency` stays the ACQUISITION frequency (`"tick"`: the raw tier holds
    one row per NBBO record). The PANEL's bar size is the separate
    `bar_interval` (D-08 / D-26), so the locked `Frequency` literal is not
    extended.

    `session_start` / `session_end` are US/Eastern wall-clock `HH:MM` edges of
    the panel's session window (D-09); the default is regular trading hours.

    The four filter fields are the resampler's record filter (D-10), read by
    `dataset/nbbo_resample.py:NbboFilterPolicy.from_config`. They are config
    fields, not `kwargs`, so a rebuild from `config.json` reproduces the panel:
    `drop_crossed` (bid > ask, both sides present), `drop_locked` (bid == ask),
    `drop_nonpositive_price` (a present price <= 0) and `keep_qu_cond` (an
    optional `qu_cond` allow-list; None keeps every condition).
    """

    market: Market = "us_equity"
    frequency: Frequency = "tick"
    vendor: Vendor | None = "wrds"
    bar_interval: BarInterval = "1m"
    session_start: str = "09:30"
    session_end: str = "16:00"
    drop_crossed: bool = True
    drop_locked: bool = False
    drop_nonpositive_price: bool = True
    keep_qu_cond: tuple[str, ...] | None = None


#: QQQ's PERMNO (`03.10-LIVE-CHECK-NDX-QQQ.json` key `C1_qqq_names`). A module
#: constant rather than a literal inside the classmethod below, because the
#: same number is the one a user passes to `permnos` when they want the ETF in
#: some other window.
QQQ_PERMNO: str = "86755"


@dataclass(kw_only=True)
class CrspDatasetConfig(DatasetConfig):
    """Config of the CRSP Stock v2 daily panel (`dataset/crsp.py`, phase 03.10).

    The three market fields default rather than being asked for: CRSP Stock v2
    is US equity, daily, and reached through the `wrds` account. They stay
    FIELDS (not constants on the dataset) because `registry.convert()` resolves
    its capability from `(market, frequency, data_type)` read off this object,
    and a config that could not state them would not be resolvable.
    """

    market: Market = "us_equity"
    frequency: Frequency = "1d"
    vendor: Vendor | None = "wrds"

    #: The CRSP reference tier (`stksecurityinfohist` and friends) this
    #: conversion reads its symbology from. REQUIRED, and deliberately not
    #: derived from `raw_data_dir_path`: the reference tier is a SIBLING of
    #: the raw root, pulled by a different step, and a conversion pointed at a
    #: raw tree whose sibling was never filled must fail saying so rather than
    #: guessing a path.
    reference_dir: str

    #: Restrict the conversion to these PERMNOs (digit strings). `None` means
    #: every PERMNO present in the raw tier. This is the RAW-side filter; the
    #: inherited `symbols` is the TICKER-side one, applied after symbology.
    #:
    #: **Also an EXPLICIT ROSTER, which overrides `security_filter`** (GAP-C,
    #: the operator's decision of 2026-09-20). Setting this says "I named these
    #: securities", so every row of every PERMNO listed here survives the type
    #: filter regardless of its `sharetype` / `securitytype` / `securitysubtype`
    #: -- on every one of its dates, not only inside some window. The type filter
    #: screens an UNSPECIFIED population; it does not overrule a roster. The
    #: override is recorded under `roster_overrides` in
    #: `{zarr}.crsp_filter_report.json`, never applied silently.
    permnos: tuple[str, ...] | None = None

    #: `{PERMNO: symbol}`, applied BEFORE every symbology rule. The live case
    #: is QQQ (PERMNO 86755), whose ticker really was `QQQQ` from 2004-12-01
    #: to 2011-03-22 -- a rename an index panel does not want to see, because
    #: the instrument never changed.
    symbol_overrides: dict[str, str] | None = None

    #: WHICH SECURITIES the panel holds (D-06, D-17). Either the name of a
    #: preset in `quantlab/dataset/crsp.py:SECURITY_FILTER_PRESETS`
    #: (`"equity_common"`, `"shrcd_10_11"`, `"none"`) or an explicit
    #: `{column: allowed values}` mapping over
    #: `quantlab/dataset/crsp.py:FILTERABLE_COLUMNS`.
    #:
    #: The default is `"equity_common"`, D-17's common-stock panel: REITs
    #: (share type `SB` included) and non-US-incorporated issuers stay; ADRs,
    #: units, funds/ETFs and unknown types go. The predicate is evaluated PER
    #: DATE against `dsf_v2`'s own per-day type columns, so a security that
    #: changed what it is keeps only the era in which it qualified.
    #:
    #: The filter lives HERE and never in the SQL or the raw tier: raw stays
    #: CRSP-complete, and re-filtering a panel is a re-conversion rather than
    #: a re-download.
    security_filter: str | dict = "equity_common"

    #: Break the adjusted series where a ticker column changes COMPANY (D-18).
    #: When True (the default) the incoming PERMNO's first row in a symbol
    #: column gets NaN `adjOpen/adjHigh/adjLow/adjClose/adjVolume`, so no
    #: return and no rolling window spans two securities. Raw prices and
    #: `permno` are untouched, and the seam is reported either way.
    nan_adj_at_permno_seam: bool = True

    #: **THE INDEX THIS CONVERSION IS SCOPED TO**, one of
    #: `quantlab/dataset/crsp_membership.py:CrspMembership.INDEXES`. It has two
    #: uses, and the second is the larger one:
    #:
    #: 1. It breaks a same-day ticker collision (D-04). `None` means that
    #:    tie-break is unavailable, and a collision no other rule resolves
    #:    REFUSES the conversion rather than merging two securities into one
    #:    column.
    #: 2. It is an EXPLICIT ROSTER that overrides `security_filter` (GAP-C, the
    #:    operator's decision of 2026-09-20). The index provider already decided
    #:    membership, so a member is never dropped by the type filter during its
    #:    membership spell -- per DATE, from the same `permno_intervals` frame
    #:    the tie-break reads. Outside its spells a PERMNO is an unspecified
    #:    population again and the filter applies normally. The override is
    #:    recorded under `roster_overrides` in
    #:    `{zarr}.crsp_filter_report.json`, never applied silently.
    #:
    #: **The NAME is narrower than the responsibility, and was not changed.**
    #: Every store this phase has already written carries a serialized
    #: `config.json` that `quantlab/utils/module.py:load_backtester_from_config`
    #: and the dataset rebuild path read back BY KEY, so a rename breaks the
    #: round trip for data that exists on disk today. A SECOND field naming the
    #: same universe would be worse: two fields carrying one fact can disagree,
    #: and nothing at runtime would notice. A rename becomes forced the first
    #: time a conversion legitimately needs a roster universe DIFFERENT from its
    #: collision tie-break universe; at that point the field splits and both
    #: names become accurate.
    collision_universe: str | None = None

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
        """The QQQ benchmark store: its OWN store, filter off, ticker pinned.

        **Why a separate store rather than one more symbol** (D-15). Everything
        in the equity panel enters cross-sectional ranking and model training,
        and an ETF ranked against the very constituents it holds is not a stock
        pick -- it is the index competing with itself in the same cross
        section. Keeping QQQ in its own store means the benchmark can be read
        beside the panel without ever being read INSIDE it.

        The three settings are stated here rather than left to the caller
        because getting any one of them wrong is silent:

        - `permnos=(QQQ_PERMNO,)` -- the ETF alone;
        - `security_filter="none"` -- QQQ is `FUND`/`ETF`, which the equity
          panel's default filter drops by design (D-06/D-17), so a benchmark
          store built with that filter would come out EMPTY;
        - `symbol_overrides={QQQ_PERMNO: "QQQ"}` -- CRSP's period-correct
          ticker really was `QQQQ` from 2004-12-01 to 2011-03-22, so without
          the override one instrument's history would arrive as two columns
          with a hole in each.

        **This store is DATA ONLY in phase 03.10** (D-16). Nothing here, and
        nothing in the CRSP dataset or acquisition modules, touches
        `BacktestConfig.benchmark_dataset` -- that slot still raises
        `NotImplementedError` (phase 03.7 D-08), and wiring it is a separate
        task with its own decisions about alignment and warm-up.

        A CLASSMETHOD on the config, deliberately not a `quantlab/config`
        factory function: this project's config factories hardcode absolute
        per-machine paths, and every path here is an argument.
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
            symbol_overrides={QQQ_PERMNO: "QQQ"},
        )


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
