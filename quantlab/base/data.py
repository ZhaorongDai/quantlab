import datetime
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Self

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from tqdm import tqdm

from quantlab.base.config import BaseDatasetConfig, DatasetConfig
from quantlab.dataset.backend import XrBackend
from quantlab.dataset.cleaning import clean_market_data
from quantlab.enums.constant import Date
from quantlab.utils.timer import Timer


class BaseDataset(ABC):
    """The shared, storage-medium-agnostic dataset contract (D-03, DATA-06).

    Everything the pipeline needs from a dataset lives here: the `config`
    lifecycle, the `XrBackend` storage round-trip (`read`/`save`), the
    `xr.Dataset` / `pl.LazyFrame` accessors, the `from_raw_data()` ingestion
    pipeline with its overridable `_clean()` hook, and the single abstract
    member `_raw_data_to_xr()` that every dataset kind implements for itself.

    The base is deliberately free of any nautilus or KunQuant concept -- no
    bar conversion, no `ParquetDataCatalog`, no compiled-graph input arrays,
    and no read of the market-only `raw_data_dir_path`/`catalog_path`/
    `market`/`frequency` config fields. That is what lets a dataset with no
    OHLCV shape at all -- an index-membership panel, say -- complete the whole
    persistence lifecycle by implementing exactly one abstract method, instead
    of carrying two meaningless `raise NotImplementedError` stubs.
    """

    #: What `from_raw_data_chunked()` does when the pinned whole-range symbol
    #: axis no longer matches the STORE's -- the routine consequence of a new
    #: listing between two periodic refreshes (260906-x2s).
    #:
    #: - `refuse`  the DEFAULT, byte-identical to the behaviour before this
    #:             knob existed: the `ChunkLedger` roster error raises and the
    #:             store is untouched.
    #: - `rebuild` direction 2 -- re-densify EVERY window from raw onto the new
    #:             union, recovering the new listing's REAL history. Correct
    #:             and expensive.
    #: - `widen`   direction 1 -- keep the store, widen its symbol axis in
    #:             place and NaN-backfill the new listing's whole historical
    #:             block. Cheap in wall-clock, but it does not re-read raw, so
    #:             history the vendor has is not recovered.
    #:
    #: `ingest_us_equity.py --on-new-listing` derives its `choices` from this
    #: tuple; it is never restated there.
    NEW_LISTING_STRATEGIES: tuple[str, ...] = ("refuse", "rebuild", "widen")

    #: ONE-SHOT handoff from the construction-time raw fallback to the FIRST
    #: `from_raw_data()` call after it, and to nothing else.
    #:
    #: When `_reset_symbols()` cannot resolve a symbol axis from the store it
    #: materialises the whole panel through `from_raw_data()`. The two ingest
    #: scripts then immediately call `from_raw_data()` themselves, converting
    #: the identical raw tree a second time microseconds later -- measured at
    #: 2 conversions for one ingest. These two attributes let that first call
    #: recognise the panel it is about to rebuild and return it instead.
    #:
    #: `_construction_raw_panel` holds the panel object the fallback produced;
    #: `_construction_raw_window` holds the `(start_date, end_date)` pair it
    #: was produced for.
    #:
    #: Deliberately CLASS attributes rather than `__init__` assignments: the
    #: relative order of `self.data_backend = ...` and `self.config = config`
    #: inside `__init__` is source-introspected by
    #: `tests/test_dataset_hierarchy.py`, and a subclass that builds its
    #: config outside `BaseDataset.__init__` must not hit a missing attribute.
    _construction_raw_panel: Optional[xr.Dataset] = None
    _construction_raw_window: Optional[tuple] = None

    def __init__(self, config: BaseDatasetConfig):
        # Ordering is load-bearing, and it is deliberately the OPPOSITE of
        # `base/factor.py:Factor.__init__`, which assigns its config first.
        # Here the config property setter below DOES reach the storage
        # backend -- through `_reset_symbols()` -> `read()` -- so the backend
        # must already exist by the time the setter fires. Do not "harmonize"
        # the two hierarchies: reversing these two lines raises
        # `AttributeError` on every dataset construction.
        self.data_backend = XrBackend()
        self.config = config

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def num_symbols(self) -> int:
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.size

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    @property
    def symbols(self) -> list[str]:
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.values.tolist()

    @property
    def time_interval(self) -> np.timedelta64:
        """相邻时间戳之差的**众数**（针对周末/停牌造成的缺口）。

        `get_xarray_dataset(["timestamp"])` 要的就是「只剩时间轴」的那份数据，
        差分只在这一根轴上做。以前 `XrBackend` 忽略 `indexes`，这里拿回的是整个
        面板，于是这个属性在该后端下**根本跑不通**——两个错误接连出现（都在
        `data/data/us_equity/1d/us_all.zarr` 上实测过）：

            TypeError: numpy boolean subtract, the `-` operator, is not
            supported ...                      # .diff 撞上布尔的 anomaly_flag
            AttributeError: 'Dataset' object has no attribute 'to_series'

        2026-09-07 一并修好：`indexes` 现在真的收窄维度，而 `.to_series()` 是
        `DataArray` 的方法不是 `Dataset` 的，所以这里显式取 `["timestamp"]` 这个
        坐标再差分。由 `tests/test_backend_indexes.py` 锁。

        唯一的调用点是 `dataset/spot.py:_xr_to_bars`（nautilus 那条路）。
        """
        timestamps = self.data_backend.get_xarray_dataset(["timestamp"])[
            "timestamp"
        ]
        return (
            timestamps.diff(dim="timestamp")
            .to_series()
            .mode()  # 取众数，针对周末数据缺失的情况
            .values[0]
        )

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def _filter(self):
        self.data_backend.filter_by_date(
            "timestamp", self.config.start_date, self.config.end_date
        )
        if self.config.symbols is not None:
            self.data_backend.filter_by_symbol("symbol", self.config.symbols)

    @property
    def config(self) -> BaseDatasetConfig:
        return self._config

    @config.setter
    def config(self, config: BaseDatasetConfig):
        self._config = config
        self._config.name = self.import_path

        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

        # Normalise both dates to zero-padded ISO ONCE, here at the boundary.
        # Every downstream date comparison in this codebase is LEXICOGRAPHIC on
        # these strings -- `_clamp_coverage_start`'s `requested >=
        # coverage_start`, `_densify`'s `max(start_date, coverage_start)`,
        # `get_symbols_as_of`'s `as_of_date < coverage_start` -- so a non-padded
        # or non-ISO value ("2007-2-1", "01/01/2005") does not fail to match, it
        # compares WRONG and silently skips the clamp or the coverage guard.
        # Normalising once keeps every comparison downstream a plain string
        # comparison, which is why they are written that way.
        self._config.start_date = self._normalize_date(
            self._config.start_date, "start_date"
        )
        self._config.end_date = self._normalize_date(
            self._config.end_date, "end_date"
        )

        # Normalise `symbols` to the declared `tuple | None` HERE rather than
        # inside `_reset_symbols()`. `_reset_symbols()` is an overridable seam
        # -- `IndexConstituentDataset` correctly makes it a no-op -- so
        # normalising there left the declared contract false for that whole
        # branch of the hierarchy: whatever the caller passed (a list, from the
        # constituent config factories) survived unchanged and reached
        # `filter_by_symbol(col, symbols: tuple[str, ...])`. It worked by
        # accident because `.sel` accepts both.
        if self._config.symbols is not None:
            self._config.symbols = tuple(self._config.symbols)
            self._reset_symbols()

    def _normalize_date(self, value: str, field_name: str) -> str:
        try:
            return datetime.date.fromisoformat(str(value)).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.class_name}: {field_name} must be an ISO YYYY-MM-DD "
                f"date string, got {value!r}. Dates are compared "
                f"lexicographically throughout this pipeline, so a non-ISO "
                f"value compares wrong rather than failing to match."
            ) from exc

    def _reset_symbols(self):
        """Resolve `config.symbols` eagerly from the store, at
        config-assignment time.

        The default is exactly the behaviour every dataset class has always
        had: when the caller pinned a symbol subset, resolve the symbol axis
        from the store -- falling back to `from_raw_data()` when the store
        cannot supply one -- and overwrite `config.symbols` with whatever was
        actually resolved.

        **Three cases, decided BEFORE `read()` is called.** The store is
        probed with `_stored_symbol_axis()`, which answers coordinate-only and
        lazily:

        - **absent** (`None`, no store on disk): fall back to
          `from_raw_data()`.
        - **present and POPULATED** (a non-empty list): `read()`, exactly as
          before.
        - **present but EMPTY** (`[]`, or a store with no `symbol`
          coordinate): fall back to `from_raw_data()` as well. This case is
          reachable from ANY run that wrote a zero-row panel -- an acquisition
          that fetched nothing, an aborted chunked ingest, a window that
          pruned to nothing -- and it is why the probe cannot be replaced by
          an `except` clause. An empty store does not raise
          `FileNotFoundError`: `read()` SUCCEEDS and then `_filter()` ->
          `filter_by_symbol` -> `.sel({"symbol": [...]})` raises
          `ValueError: could not convert string to float` from INSIDE
          `read()`, because zarr reads a zero-length symbol axis back as
          float64. No handler on `read()` can see that as "no data here".

        The inner `try/except FileNotFoundError` is KEPT rather than made
        dead: it is still reachable if the store is removed between the probe
        and the read.

        Overridable seam: a dataset whose symbol axis is derived from its own
        source rather than from a store -- or one whose `from_raw_data()`
        fallback would perform a remote fetch merely to construct the object
        -- overrides this to a no-op and resolves its symbols inside
        `_raw_data_to_xr()` instead. Only `FileNotFoundError` is caught around
        the read, so for such a dataset any network or parse error would
        otherwise escape `__init__`.
        """
        store_path = self.config.zarr_file_path
        stored_symbols = self._stored_symbol_axis(store_path)
        if not stored_symbols:
            if stored_symbols is None and not Path(store_path).exists():
                reason = "there is no store at that path"
            elif stored_symbols is None:
                reason = (
                    "the store at that path carries no 'symbol' coordinate"
                )
            else:
                reason = (
                    "the store at that path is present but its symbol axis is "
                    "EMPTY (a zero-row panel from an earlier run)"
                )
            logger.warning(
                f"{self.class_name} data not found, try to read from csv "
                f"({store_path}: {reason})"
            )
            self.from_raw_data()
            # Hand the panel just materialised to the NEXT `from_raw_data()`
            # call -- the caller's -- so one ingest converts raw once. Recorded
            # on this fallback path ONLY: on the `read()` path below the
            # caller's `from_raw_data()` is doing real, necessary work and must
            # not be skipped.
            self._construction_raw_panel = getattr(
                self.data_backend, "data", None
            )
            self._construction_raw_window = (
                self._config.start_date,
                self._config.end_date,
            )
        else:
            try:
                self.read()
            except FileNotFoundError:
                logger.warning(
                    f"{self.class_name} data not found, try to read from csv"
                )
                self.from_raw_data()
        symbols = tuple(self._get_symbols())
        self._config.symbols = symbols

    def _get_symbols(self) -> list[str]:
        return self.data_backend.get_xarray_dataset(
            ["symbol", "timestamp"]
        ).symbol.values.tolist()

    def read(self, **kwargs):
        self.data_backend.read(self.config.zarr_file_path, **kwargs)
        self._filter()
        return self

    def save(self, **kwargs):
        with Timer(f"{self.__class__.__name__}: save"):
            self._filter()
            self.data_backend.write(self.config.zarr_file_path, **kwargs)

    def get_config(self) -> dict:
        return self.config.to_dict()  # type: ignore

    def get_lazyframe(self) -> pl.LazyFrame:
        return self.data_backend.get_lazyframe()

    def head(self, n: int) -> pl.LazyFrame:
        """A BOUNDED read of at most `n` rows -- `get_lazyframe()`'s twin.

        Concrete, not abstract, and shaped exactly like the pass-through above
        it: the bound and now the STORE LOCATION are both properties of the
        STORAGE MEDIUM. The dataset supplies the path it already owns --
        exactly as `read()` does, one method up -- and adds no opinion of its
        own, so every dataset kind inherits whatever its backend implements.
        `tests/test_dataset_hierarchy.py` pins
        `BaseDataset.__abstractmethods__` to `{"_raw_data_to_xr"}` and that
        assertion is correct -- one abstract member is what lets a dataset
        with no OHLCV shape at all complete the whole lifecycle.

        The signature takes `n` only: the path is not the caller's to choose,
        and its one caller (`base/factor_polars.py`) has no business naming
        the store.

        Used by `base/factor_polars.py` to learn what its computation graph
        produces WITHOUT going through `read()`: names derive from the GRAPH,
        and the graph needs a schema, not data. Routing that probe through
        `read()` is what silently narrowed the shared dataset's date window
        and dropped the factor's lookback (RV-01) -- `read()` runs `_filter()`
        and the backend caches the result.
        """
        return self.data_backend.head(self.config.zarr_file_path, n)

    def get_xarray_dataset(self) -> xr.Dataset:
        return self.data_backend.get_xarray_dataset(["timestamp", "symbol"])

    def from_raw_data(self) -> Self:
        """Materialise the raw source into the backend: convert, `_clean()`,
        store.

        **The one-shot handoff.** When `_reset_symbols()` could not resolve a
        symbol axis from the store it already ran this exact pipeline at
        construction time; the ingest scripts then call this method again
        immediately, and the raw tree was being converted TWICE for one
        ingest. If the constructor left a panel behind, the backend still
        holds that very object, and the config's date window is still the one
        it was built for, this call returns it unchanged. The handoff is read
        and CLEARED on entry whether or not it is used, so it is valid for
        exactly one call -- a later `from_raw_data()` re-converts, as it
        always did. This is a duplicate removal, not a cache.

        What this deliberately does NOT change:

        - the signature is still `from_raw_data(self) -> Self`, and no caller
          passes or receives anything new;
        - name/symbol derivation still happens EAGERLY at construction, from a
          full raw materialisation -- D-05's construction-time-names contract
          is untouched, and RV-02 stays open;
        - with `symbols=None` (the chunked ingest path) `_reset_symbols()`
          never fires, no handoff is ever recorded, and behaviour is identical
          to before;
        - when the store is POPULATED, `_reset_symbols()` takes the `read()`
          branch and records no handoff, so this method converts exactly as it
          always did. The skip only ever applies to a panel the constructor
          itself just built.

        One named narrowing: a caller that mutates the raw SOURCE location
        (`config.raw_data_dir_path`, `vendor`, `frequency`) in place on a live
        instance between construction and the first `from_raw_data()` receives
        the constructor's panel rather than a re-read of the new location. The
        guard compares the date window, not the source path. Nothing in this
        repo does that; a changed date window IS detected.
        """
        pending_panel = getattr(self, "_construction_raw_panel", None)
        pending_window = getattr(self, "_construction_raw_window", None)
        # Cleared FIRST and unconditionally: the handoff must not survive this
        # call whether or not it is used.
        self._construction_raw_panel = None
        self._construction_raw_window = None

        if (
            pending_panel is not None
            # Identity, not equality: a backend whose panel was replaced no
            # longer holds what the constructor produced.
            and getattr(self.data_backend, "data", None) is pending_panel
            and pending_window
            == (self.config.start_date, self.config.end_date)
        ):
            logger.debug(
                f"{self.class_name}: reusing the panel the construction-time "
                f"raw fallback already materialised for "
                f"{pending_window} -- skipping one duplicate conversion. The "
                f"handoff is now spent; any later from_raw_data() reconverts."
            )
            return self

        data = self._raw_data_to_xr()
        data = self._clean(data)
        self.data_backend.to_internal(data)  # type: ignore
        return self

    def _raw_axes_in_range(self) -> tuple[list[str], "pd.DatetimeIndex"]:
        """Return `(pinned_symbols, observed_timestamps)` for the config's
        whole date range, from ONE scan of the raw source.

        Overridable seam. **This default is correct but NOT memory-bounded:**
        it derives both axes from `_raw_data_to_xr()`, so it materialises the
        entire dense whole-range panel -- exactly the allocation
        `from_raw_data_chunked()` exists to avoid. It is the historical
        behaviour rather than a `raise NotImplementedError` stub, following
        `_reset_symbols()`'s idiom, so every existing subclass keeps working
        unchanged.

        A subclass whose raw source can push a date/column filter DOWN before
        materialisation (a `pl.LazyFrame` over parquet, say) overrides this
        and gets the memory bound; one that cannot inherits a working, slower
        default and is warned about it at run time.
        """
        data = self._raw_data_to_xr()
        symbols = [str(symbol) for symbol in data["symbol"].values.tolist()]
        return symbols, pd.DatetimeIndex(data["timestamp"].values)

    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset:
        """Densify ONE time window, onto `symbols` when a pinned axis is given.

        Overridable seam, and the same caveat as `_raw_axes_in_range()`
        applies: **this default is correct but NOT memory-bounded**, because
        it densifies the whole range and slices afterwards. Overriding it is
        what turns chunking from a bounded WRITE into a bounded DENSIFY.

        When `symbols` is supplied the returned panel's `symbol` coordinate
        equals it exactly, including symbols with no row in this window --
        those become all-NaN columns, which is the same value the whole-range
        densification already produces for an untraded cell.
        """
        data = self._raw_data_to_xr()
        data = data.sel(timestamp=slice(start_date, end_date))
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def from_raw_data_chunked(
        self,
        granularity: str = "year",
        ledger_path: str | None = None,
        append_dim: str = "timestamp",
        on_new_listing: str = "refuse",
    ) -> Self:
        """Densify and append ONE time window at a time (D-01).

        Peak memory scales with the WINDOW rather than the range, which is
        what makes the full multi-decade, full-market panel materialisable on
        a machine that cannot hold it whole.

        The ordering is load-bearing:

        1. The symbol axis is resolved ONCE over the whole range, BEFORE any
           window exists (D-02) -- the same all-time-union rule
           `base/constituent.py:_densify` follows. If each window derived its
           own axis, the chunks would carry inconsistent coordinates and the
           append would silently misalign, so every window is materialised on
           this one pinned axis and checked against it element-for-element.
        2. Windows come from the OBSERVED timestamp axis, never from calendar
           arithmetic: trading days are not calendar days.
        3. Completed windows are recorded in a sidecar ledger, so an
           interrupted run resumes at the first unwritten window (D-04).

        `on_new_listing` decides what happens when step 1's pinned axis no
        longer matches the STORE's -- the routine consequence of a new listing
        between two periodic refreshes. See `NEW_LISTING_STRATEGIES`; the
        default `refuse` reproduces the pre-260906-x2s behaviour exactly.
        """
        from quantlab.base.chunking import ChunkLedger, TimeChunkPlanner

        if on_new_listing not in self.NEW_LISTING_STRATEGIES:
            raise ValueError(
                f"{self.class_name}: unknown on_new_listing strategy "
                f"{on_new_listing!r}; accepted values are "
                f"{list(self.NEW_LISTING_STRATEGIES)}."
            )

        if type(self)._raw_data_to_xr_window is BaseDataset._raw_data_to_xr_window:
            logger.warning(
                f"{self.class_name}: _raw_data_to_xr_window has not been "
                f"overridden, so each window is produced by densifying the "
                f"WHOLE range and slicing. Chunking still bounds the write "
                f"and still gives a resumable run, but the memory win is "
                f"absent -- override the seam for a source that can push the "
                f"date filter down before materialising."
            )

        symbols, timestamps = self._raw_axes_in_range()
        planner = TimeChunkPlanner(granularity)
        windows = planner.plan_from_timestamps(timestamps)
        ledger = ChunkLedger(
            ledger_path or ChunkLedger.default_path(self.config.zarr_file_path),
            append_dim=append_dim,
        )

        # Reconcile the pinned axis against the STORE's before the ledger
        # check, because two of the three strategies change what that check is
        # looking at: `widen` makes the store agree with the pinned axis, and
        # `rebuild` moves the store out of the way so the run becomes a first
        # run. `refuse` changes nothing and lets `assert_consistent` raise
        # exactly as it did before this knob existed.
        ledger, rebuild_asides = self._reconcile_new_listings(
            symbols, ledger, append_dim, on_new_listing
        )

        try:
            # Before the first irreversible append, not after: the ledger and
            # the store are two independent records of the same truth and a
            # resume trusts neither alone (D-04 / T-13w-02).
            ledger.assert_consistent(symbols, self.config.zarr_file_path)

            logger.info(
                f"{self.class_name}: chunked ingestion over {len(windows)} "
                f"{granularity} window(s), {len(symbols)} pinned symbol(s), "
                f"{len(timestamps)} observed timestamp(s)."
            )

            first_timestamp = timestamps.min() if len(timestamps) else None
            boundaries = 0
            for start, end in windows:
                if ledger.is_written(start, end):
                    logger.info(
                        f"{self.class_name}: window {start.date()}..{end.date()} "
                        f"already recorded in the ledger, skipping."
                    )
                    continue

                window = self._raw_data_to_xr_window(start, end, symbols)
                actual = [str(symbol) for symbol in window["symbol"].values.tolist()]
                if actual != symbols:
                    raise ValueError(
                        f"{self.class_name}: window {start.date()}..{end.date()} "
                        f"came back on a symbol axis of {len(actual)} label(s), "
                        f"but the pinned whole-range axis has {len(symbols)}. "
                        f"Every window must be materialised on the pinned axis "
                        f"(D-02); appending this one would silently misalign "
                        f"every column in the store."
                    )

                window = self._clean(window)
                window = self._pin_append_dtypes(window)
                if start != first_timestamp:
                    boundaries += 1

                self.data_backend.to_internal(window)
                self.data_backend.append(
                    self.config.zarr_file_path, append_dim=append_dim
                )
                ledger.record(start, end, int(window.sizes[append_dim]), symbols)
                logger.info(
                    f"{self.class_name}: appended window "
                    f"{start.date()}..{end.date()} "
                    f"({int(window.sizes[append_dim])} row(s))."
                )

            if boundaries:
                logger.warning(
                    f"{self.class_name}: cleaning ran per window, so at "
                    f"{boundaries} chunk-boundary timestamp(s) `flag_anomalies` "
                    f"had no prior sample to diff against and a single-step jump "
                    f"across that boundary is not flagged. A bounded, documented "
                    f"consequence of chunking -- finer --chunk granularity "
                    f"produces more such boundaries, not fewer."
                )
        except BaseException:
            # A rebuild that DELETED the store first would have no way back
            # from a failure halfway through a multi-hour re-densify. The
            # originals were renamed aside, so restoring them is a rename.
            if rebuild_asides is not None:
                self._restore_rebuild_asides(rebuild_asides)
            raise
        else:
            if rebuild_asides is not None:
                self._discard_rebuild_asides(rebuild_asides)
        return self

    #: Appended to the store and ledger paths while a `rebuild` is in flight.
    #: Deliberately the same suffix `XrBackend.widen_symbol_axis` uses: both are
    #: "the previous authoritative copy, kept until the replacement lands".
    SUPERSEDED_SUFFIX = ".superseded.tmp"

    @staticmethod
    def _stored_symbol_axis(store_path: str, dim: str = "symbol") -> Optional[list]:
        """The store's `dim` labels, or None when there is no store (or no such
        coordinate).

        Opened the way `ChunkLedger._store_tail` opens it -- lazily, coordinate
        only, closed in a `finally`. Reading the data variables to answer an
        axis question would defeat the whole point of chunking.
        """
        if not Path(store_path).exists():
            return None
        store = xr.open_zarr(store_path)
        try:
            if dim not in store.coords:
                return None
            return [str(label) for label in store[dim].values.tolist()]
        finally:
            store.close()

    def _widen_fill_values(self) -> dict:
        """Per-variable fill values for a `widen`'s reindex.

        `_pin_append_dtypes` promotes integer variables to float64 before an
        append but deliberately leaves `anomaly_flag` BOOL, so a real cleaned
        market panel has exactly one non-float variable -- and
        `XrBackend.widen_symbol_axis` refuses to NaN-backfill a non-float
        variable without an explicit fill. Without this the widen of a real
        store would ALWAYS refuse on `anomaly_flag`; it is required, not
        academic.

        `False` is the honest value for a symbol that was not trading: it was
        not flagged because there was nothing to flag.

        A seam rather than a constant because a future non-OHLCV `Dataset`
        subclass carries different variables -- the same reasoning `_clean()`
        is an overridable hook for.
        """
        return {"anomaly_flag": False}

    def _reconcile_new_listings(
        self,
        symbols: list,
        ledger,
        append_dim: str,
        on_new_listing: str,
    ) -> tuple:
        """Apply `on_new_listing` when the STORE's symbol axis has drifted from
        the pinned whole-range one. Returns `(ledger, rebuild_asides)`.

        Falls through completely unchanged -- no store read beyond the
        coordinate, no log line -- when the axes already agree, which is the
        overwhelmingly common case.
        """
        store_path = self.config.zarr_file_path
        stored = self._stored_symbol_axis(store_path)
        if stored is None or stored == list(symbols):
            return ledger, None

        added = [symbol for symbol in symbols if symbol not in set(stored)]
        removed = [symbol for symbol in stored if symbol not in set(symbols)]

        if on_new_listing == "refuse":
            # Fall through unchanged: `assert_consistent` raises next, its
            # message already naming the roster refresh. Logged first so the
            # operator learns the remedy exists rather than inferring that the
            # only way forward is deleting the store.
            logger.info(
                f"{self.class_name}: the store at {store_path} holds "
                f"{len(stored)} symbol(s) but the pinned whole-range axis has "
                f"{len(symbols)} ({len(added)} added, {len(removed)} removed). "
                f"on_new_listing='refuse' (the default), so this run will "
                f"halt. Pass --on-new-listing rebuild to re-densify every "
                f"window from raw, or --on-new-listing widen to keep the store "
                f"and backfill the new listing(s) with NaN."
            )
            return ledger, None

        if on_new_listing == "widen":
            logger.warning(
                f"{self.class_name}: widening {store_path} from {len(stored)} "
                f"to {len(symbols)} symbol(s); added={added}, "
                f"removed={removed}. The added symbol(s) will carry NaN for "
                f"the ENTIRE historical block -- a widen does not re-read raw, "
                f"so history the vendor already has is not recovered. Use "
                f"on_new_listing='rebuild' for that. The whole store is also "
                f"materialised in memory to rewrite it (no dask here), so "
                f"'rebuild' is the strategy for a store too large to hold."
            )
            self.data_backend.widen_symbol_axis(
                store_path,
                list(symbols),
                append_dim=append_dim,
                fill_values=self._widen_fill_values(),
            )
            # In the SAME operation, never later: the ledger fingerprints the
            # pinned symbol list order-sensitively, so a widened store whose
            # ledger still carries the old fingerprint is un-resumable.
            ledger.rebase(symbols)
            return ledger, None

        # rebuild
        from quantlab.base.chunking import ChunkLedger

        logger.warning(
            f"{self.class_name}: rebuilding {store_path} -- EVERY window will "
            f"be re-densified from raw onto the new {len(symbols)}-symbol "
            f"union (was {len(stored)}); added={added}, removed={removed}. "
            f"This recovers the added symbol(s) REAL history rather than "
            f"backfilling NaN, at the cost of a full re-densify."
        )
        asides = {
            "store": store_path,
            "store_aside": f"{store_path}{self.SUPERSEDED_SUFFIX}",
            "ledger": ledger.path,
            "ledger_aside": f"{ledger.path}{self.SUPERSEDED_SUFFIX}",
            "ledger_existed": Path(ledger.path).exists(),
        }
        # Renamed aside, NEVER deleted: a rebuild that deletes first has no way
        # back from a failure halfway through.
        os.replace(asides["store"], asides["store_aside"])
        if asides["ledger_existed"]:
            os.replace(asides["ledger"], asides["ledger_aside"])
        # A FRESH ledger at the original path, so the run below is an ordinary
        # first run: no store, no recorded windows, `assert_consistent` passes.
        return ChunkLedger(asides["ledger"], append_dim=append_dim), asides

    def _restore_rebuild_asides(self, asides: dict) -> None:
        """Put the pre-rebuild store and ledger back, discarding the partial."""
        if Path(asides["store"]).exists():
            shutil.rmtree(asides["store"], ignore_errors=True)
        os.replace(asides["store_aside"], asides["store"])
        Path(asides["ledger"]).unlink(missing_ok=True)
        if asides["ledger_existed"]:
            os.replace(asides["ledger_aside"], asides["ledger"])
        logger.warning(
            f"{self.class_name}: the rebuild of {asides['store']} failed; the "
            f"pre-rebuild store and ledger have been restored."
        )

    @staticmethod
    def _discard_rebuild_asides(asides: dict) -> None:
        """The rebuild landed -- drop the superseded copies."""
        shutil.rmtree(asides["store_aside"], ignore_errors=True)
        Path(asides["ledger_aside"]).unlink(missing_ok=True)

    @staticmethod
    def _pin_append_dtypes(data: xr.Dataset) -> xr.Dataset:
        """Promote integer data variables to float64 before an append.

        The dtype of a window is a function of its own DENSITY: a window in
        which every pinned symbol traded on every timestamp keeps pandas'
        int64 for `volume`, while any window with a gap upcasts to float64
        for the NaN. Left alone, the store's dtype would therefore be decided
        by whichever window happened to be written first, and a later
        float64 NaN appended into an int64 variable is silently cast to 0 --
        a fabricated observation where data was missing.
        `XrBackend.append` refuses that append; this makes the refusal
        unreachable by pinning the dtype to what the dense panel is anyway
        (`estimate_dense_panel` sizes it at 8 bytes per value).

        Booleans are left alone: `anomaly_flag` is a flag, not a measurement.
        """
        promoted = {
            name: variable.astype("float64")
            for name, variable in data.data_vars.items()
            if np.issubdtype(variable.dtype, np.integer)
        }
        return data.assign(**promoted) if promoted else data

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Cleaning hook run after raw-to-xarray conversion, before persistence.

        Defaults to the shared market-data cleaning pipeline (anomaly-flagging,
        schema validation — see dataset/cleaning.py). Overridable so a future
        Dataset subclass whose data isn't OHLCV-shaped tabular market data
        (e.g. unstructured sources like news) is not forced through
        market-specific validation it doesn't apply to.
        """
        return clean_market_data(data)

    @abstractmethod
    def _raw_data_to_xr(self) -> xr.Dataset: ...


class MarketDataset(BaseDataset):
    """The market-data dataset backend.

    Owns everything nautilus- and KunQuant-specific: the
    `ParquetDataCatalog` write (`_write_catalog`), the bar-conversion path
    (`to_nautilus` / `_to_nautilus`), and the compiled-graph input path
    (`to_kunquant` / `_to_kunquant`). Keeping all five off `BaseDataset` is
    what lets a dataset with no bar, no catalog and no KunQuant
    representation subclass the shared base directly instead of carrying two
    meaningless `raise NotImplementedError` stubs (D-03).
    """

    # Narrowed for readers and type checkers only -- this is a bare
    # annotation, so it does not shadow `BaseDataset.config`. It records that
    # the three methods below legitimately read the market-only
    # `catalog_path` field, which lives on `DatasetConfig` and not on the
    # shared `BaseDatasetConfig`.
    config: DatasetConfig

    def _write_catalog(self, data: list):
        catalog = ParquetDataCatalog(
            self.config.catalog_path, fs_protocol="file"
        )
        catalog.write_data(data)

    def to_nautilus(
        self, venue: str = "BINANCE", n_jobs: int = 16, write: bool = True
    ) -> tuple[list[list], list[Instrument]]:
        data = self.read().get_xarray_dataset()
        data, instruments = self._to_nautilus(data, venue=venue, n_jobs=n_jobs)
        if write:
            for d in tqdm(data, desc="Writing data"):
                self._write_catalog(d)
            for instrument in tqdm(instruments, desc="Writing instruments"):
                self._write_catalog([instrument])
        return data, instruments

    def to_kunquant(
        self, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        data = self.read().get_xarray_dataset()
        return self._to_kunquant(data, data_columns)

    @abstractmethod
    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]: ...

    @abstractmethod
    def _to_nautilus(
        self, data: xr.Dataset, venue: str, n_jobs: int
    ) -> tuple[list[list], list[Instrument]]: ...

