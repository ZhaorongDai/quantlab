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
    """Batch-only factor backend whose factor logic is written in Polars.

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
    runs `_get_factor_lazyframe()` over a few rows fetched through
    `BaseDataset.head()` and reads the resulting schema. Names therefore come
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
    """

    def __init__(self, config: PolarsFactorConfig):
        super().__init__(config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Derive the factor names by asking the graph what it produces.

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
        - The `.read()` call stays. `XrBackend.read` returns early when data
          is already in memory (the ordinary case, since `BaseDataset.
          __init__` has read it), so this is normally a cache hit; a dataset
          whose `_reset_symbols()` is a no-op has nothing loaded and needs it.

        Note that `_reset_dataset_config()` runs AFTER name resolution in the
        setter, so the probe sees the dataset's own date window rather than
        the factor's. Irrelevant to a schema -- do not "fix" it by reordering
        the setter, which would break the ordering `Factor.__init__` depends
        on.
        """
        probe = self.config.dataset.read().head(_SCHEMA_PROBE_ROWS)
        factor_lf = self._get_factor_lazyframe(probe)
        return tuple(
            name
            for name in factor_lf.collect_schema().names()
            if name not in _INDEX_COLUMNS
        )

    @abstractmethod
    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Write the factor here. This is the one method a subclass overrides.

        Args:
            lf: the dataset's already-read lazyframe, carrying the RAW,
                un-renamed column names of the underlying store. There is no
                per-market normalization step on this path, so a factor is
                written against one market's raw column names (see
                `factor/momentum.py` and 03-RESEARCH.md Open Question 2).

        Returns:
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
