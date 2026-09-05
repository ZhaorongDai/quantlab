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

    **Documented precondition on factor names.** Because names come from the
    computed frame's schema (D-05) rather than from a declaration,
    `_get_factor_names()` -- and therefore `num_factors` -- is only valid AFTER
    `cal()` or `read()` has populated `config.factor_names`. `base/model.py`
    always calls `.cal()`/`.read()` on every factor inside `collect()` before
    it asks for names, so no code path in the pipeline hits this. It is a
    narrow, intentional gap, not an oversight.
    """

    def __init__(self, config: PolarsFactorConfig):
        super().__init__(config)

    def _maybe_resolve_factor_names(self) -> None:
        # D-05: names come from the computed lazyframe's schema, so they
        # cannot be known at config-assignment time (i.e. inside __init__).
        # Keeping the inherited eager default would make merely CONSTRUCTING
        # a factor trigger a disk read, contradicting D-04's "computation
        # starts at cal()". `cal()` assigns config.factor_names instead.
        return None

    def _get_factor_names(self) -> tuple[str, ...]:
        if self.config.factor_names is None:
            raise RuntimeError(
                f"{self.class_name}: factor names are not resolved yet. "
                "Polars factor names are read from the computed frame's "
                "schema (D-05), so cal() or read() must run before "
                "_get_factor_names()/num_factors is valid."
            )
        return tuple(self.config.factor_names)

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
