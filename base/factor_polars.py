"""The Polars factor backend (FACTOR-03, D-03..D-07).

Deliberately isolated in its own module: nothing here imports a compiled-graph
symbol, so the Polars path carries no dependency on the batch/stream graph
machinery that lives in `base/factor.py`.
"""

from abc import abstractmethod
from typing import Self

import polars as pl
import xarray as xr

from base.config import PolarsFactorConfig
from base.factor import Factor
from utils.timer import Timer

_INDEX_COLUMNS = ("timestamp", "symbol")

#: Rows fetched by the bounded probe read that resolves factor names. Only the
#: SCHEMA of the result is consulted, so this is not a row requirement -- it is
#: what makes the probe carry the store's real dtypes, which a zero-row stub
#: could only fake (03-VERIFICATION.md, Gap 1).
_SCHEMA_PROBE_ROWS = 8


class FactorPolars(Factor):
    """用 Polars 表达式书写因子逻辑的因子后端，只做批量计算，不提供流式接口。

    它与编译计算图后端是兄弟而不是父子：两者都从同一个因子基类派生，因此可以互相
    替换而不必改动模型层。子类只需要覆写一个方法，其余全部继承。

    Batch-only factor backend whose factor logic is written in Polars.

    A sibling -- not a descendant -- of the compiled-graph backend declared in
    `base/factor.py`: both derive from the shared `Factor` base (D-03), so a
    `FactorPolars` subclass drops into `DLConfig.factors` with zero edits to
    `base/model.py`. A subclass overrides exactly one hook,
    `_get_factor_lazyframe(lf)`, and inherits `cal()` from here.

    **Batch-only by decision (D-07).** There is no streaming counterpart --
    no `cal_stream`, no `init_stream`, no compiled-graph helper. The streaming
    requirement belongs to the other backend; adding a dormant streaming
    surface here would be an unused member, not an extension point.

    **What is lazy, precisely (D-04).** The DERIVED factor computation graph --
    the expression chain a subclass builds on top of `lf` -- is deferred until
    `cal()` calls `.collect()`. The RAW dataset load is NOT deferred:
    `Dataset.get_lazyframe()` materializes the whole in-memory dataset into a
    pandas frame before wrapping it with `.lazy()`. That is true of the other
    backend too (its `cal()` likewise reads the whole dataset up front), so it
    is a property of the data layer, not a leak in this contract.

    **Factor names are DERIVED from the computation graph** (D-05), at
    config-assignment time, via a bounded probe read: `_get_factor_names()`
    runs `_get_factor_lazyframe()` over a few rows read straight from the
    dataset's STORE via `BaseDataset.head()`, which opens the store by path,
    and reads the resulting schema. Names therefore come
    from what the graph PRODUCES, never from a declaration and never from the
    factor store on disk -- so a store written under one horizon, read back
    under a config asking for another, yields the config's name and fails
    loudly at lookup instead of silently reporting the stale one.

    An explicit `config.factor_names` still wins: the inherited
    `_maybe_resolve_factor_names()` hook derives only when nothing was pinned,
    and this class deliberately does NOT override it.

    **Construction touches disk, by decision** (03-VERIFICATION.md, "Gap
    Dispositions" -> "Gap 1"). Because names resolve in the config setter,
    merely constructing a factor performs a bounded read -- measured at
    ~25-50 ms and flat in store size, since it is dominated by metadata-open
    overhead rather than data volume. D-04's "computation starts at `cal()`"
    is not violated: a few rows plus metadata is not the computation. The
    alternative -- a zero-row schema stub -- was cheaper and was rejected,
    because hand-constructing the input dtypes makes any dtype-sensitive
    expression derive a different name or fail spuriously. Real rows carry
    real dtypes for free.

    Because the probe opens the store DIRECTLY -- rather than reaching it
    through a `read()` that would narrow the shared dataset in place (RV-01)
    -- a factor cannot be constructed before its dataset's store exists on
    disk. That narrowing of what is constructible is filed as RV-02 in
    `03-VERIFICATION.md` and is deliberately NOT closed here (D-3 of the
    RV-01 fix plan).
    """

    def __init__(self, config: PolarsFactorConfig):
        """构造一个 Polars 因子。

        这里没有额外动作，但要留意：构造过程会经由配置赋值触发一次对存储的有界
        探查，用来确定因子名——因此数据集的存储必须已经存在于磁盘上。原因与代价
        都记在类文档里。

        Args:
            config (PolarsFactorConfig): Polars 因子配置。
        """
        super().__init__(config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """直接问计算图会产出什么，以此确定因子名，而不是让人另外声明一遍。

        名字来自计算图真正产出的列，所以声明与实现不可能对不上——这类不一致往往
        要到很久之后取不到某一列时才暴露。

        Derive the factor names by asking the graph what it produces.

        Four constraints, each of which fails silently if broken:

        - It does not read `self.config.factor_names`. The base-class hook
          owns the explicit-pin channel; consulting it here would re-couple
          the two and make "pin wins, else derive" circular.
        - It does not touch `self.data_backend`. This runs from inside the
          `Factor.config` setter, BEFORE `Factor.__init__` has assigned the
          factor's own storage backend. The DATASET's backend, reached via
          `self.config.dataset`, is a different object and does exist.
        - It does not depend on the probe returning any ROWS. Only
          `collect_schema()` is consulted, so a date window yielding zero rows
          still yields the right names. `_SCHEMA_PROBE_ROWS` is not a row
          requirement -- it exists so real dtypes come along for free.
        - **The probe must NOT go through `read()`.** `BaseDataset.read()`
          runs `_filter()`, which narrows `data_backend.data` IN PLACE via
          `filter_by_date`, and `XrBackend.read()`'s cache early-return makes
          that narrowing survive into `cal()`. Because this runs BEFORE
          `_reset_dataset_config()` widens the dataset's window by the
          factor's `window` days, and `filter_by_date` can only narrow, a
          probe that read would silently drop the factor's entire lookback --
          a factor column that is quietly part-NaN with nothing raised
          anywhere (RV-01, `03-VERIFICATION.md`). `head()` takes the store
          path and opens the store itself, so it filters nothing. Do not put
          `.read()` back in front of it.

        Note that `_reset_dataset_config()` runs AFTER name resolution in the
        setter, so the probe sees the dataset's own date window rather than
        the factor's. Irrelevant to a schema -- do not "fix" it by reordering
        the setter, which would break the ordering `Factor.__init__` depends
        on.

        Returns:
            tuple[str, ...]: 计算图产出的因子名，已排除时间与标的这两列索引。
        """
        probe = self.config.dataset.head(_SCHEMA_PROBE_ROWS)
        factor_lf = self._get_factor_lazyframe(probe)
        return tuple(
            name
            for name in factor_lf.collect_schema().names()
            if name not in _INDEX_COLUMNS
        )

    @abstractmethod
    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """在这里写因子；这是一个子类唯一需要覆写的方法。

        Write the factor here. This is the one method a subclass overrides.

        Args:
            lf (pl.LazyFrame): 数据集已经读入的惰性帧，列名是底层存储中未经改名
                的原始列名，因此一个因子是照着某一个市场的原始列名写出来的。
                the dataset's already-read lazyframe, carrying the RAW,
                un-renamed column names of the underlying store. There is no
                per-market normalization step on this path, so a factor is
                written against one market's raw column names (see
                `factor/momentum.py` and 03-RESEARCH.md Open Question 2).

        Returns:
            pl.LazyFrame: 只含时间、标的以及算出来的因子值列的惰性帧；除这两列
                索引之外剩下的列有多少，就是这个类产出多少个因子。
            A `pl.LazyFrame` containing ONLY `timestamp`, `symbol` and the
            computed factor value column(s). No raw price or volume column may
            survive into the output -- whatever columns come back (minus the
            two index columns) ARE the factors, and are persisted as such.

        Contract:
            Nothing in this hook may materialize. Do not call `.collect()`,
            `.fetch()` or any other eager method: `cal()` is what triggers
            computation (D-04).

            Note that the lazyframe arrives as a PARAMETER, unlike the
            compiled-graph backend's `_get_factor_func()`, which builds its own
            input nodes and takes no arguments. That difference is deliberate:
            it makes a factor's logic unit-testable in isolation against a
            hand-built lazyframe, with no `Dataset` and no store on disk.
        """
        ...

    def cal(self) -> Self:
        """跑一遍因子计算图，把结果转成面板交出去。

        因子名在这里再确定一次，用的是本次真正跑出来的列：构造时那次探查读的是很
        少的几行，而这次是全量数据，以实际产出为准才不会出现名实不符。

        Polars 只是这个类内部的实现手段：结果在离开本类之前就被转成面板，层与层
        之间的交换格式始终不变。

        Returns:
            Self: 已持有计算结果的因子自身，可继续链式调用。
        """
        lf = self.config.dataset.read().get_lazyframe()
        factor_lf = self._get_factor_lazyframe(lf)

        # D-05: the factor names ARE the non-index columns of the returned
        # frame. collect_schema() inspects the schema only -- it does not
        # materialize -- so this stays inside D-04's laziness contract.
        self.config.factor_names = tuple(
            name
            for name in factor_lf.collect_schema().names()
            if name not in _INDEX_COLUMNS
        )

        with Timer(f"{self.__class__.__name__}: cal"):
            # D-06 / FACTOR-04: Polars is an internal implementation detail;
            # the result becomes an xr.Dataset before it leaves this class, so
            # the module boundary stays xarray-only. Same conversion idiom as
            # dataset/backend.py:PlBackend.get_xarray_dataset().
            frame = factor_lf.collect().to_pandas()
            frame = frame.set_index(list(_INDEX_COLUMNS))
            data = xr.Dataset.from_dataframe(frame)

        self.data_backend.to_internal(data)
        self._auto_filter()
        return self
