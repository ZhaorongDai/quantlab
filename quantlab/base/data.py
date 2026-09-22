import datetime
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from quantlab.base.progress import CancelToken, ProgressEvent, ProgressReporter
from quantlab.dataset.backend import XrBackend
from quantlab.dataset.cleaning import clean_market_data
from quantlab.enums.constant import Date
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer


@dataclass(frozen=True)
class ConversionResult:
    """What ONE raw->Zarr conversion did, as a value the caller can render.
    """

    #: The Zarr store that was written -- `config.zarr_file_path`, echoed so
    #: the caller does not have to keep the config alive to name the output.
    zarr_path: str
    #: The chunk-ledger sidecar. A resume reads it; an operator inspecting a
    #: half-finished backfill reads it too.
    ledger_path: str
    #: The window granularity the run planned on (`"year"`, `"quarter"`, ...).
    granularity: str
    #: How many symbols the pinned whole-range axis carried (D-02's axis,
    #: resolved once BEFORE any window existed).
    pinned_symbols: int
    #: Windows the planner produced over the observed timestamp axis.
    windows_planned: int
    #: Windows this run materialised and appended.
    windows_written: int
    #: Windows skipped because the ledger already recorded them. `resumed` is
    #: this being non-zero.
    windows_skipped: int
    #: Rows appended along the append dimension, summed over written windows.
    rows_written: int
    #: OBSERVED peak: the largest `window.nbytes` this run materialised.
    #: `None` when no window was written.
    peak_window_bytes: int | None
    #: The caller's pre-flight estimate, echoed back. `None` when the caller
    #: did not offer one -- which is the common case and is not a warning.
    predicted_peak_bytes: int | None = None
    #: Whether at least one ledger-recorded window was skipped, i.e. this run
    #: continued an earlier one rather than starting from nothing.
    resumed: bool = False
    #: Whether a caller's `CancelToken` was observed at a window boundary and
    #: the loop stopped early (03.5 D-05). LIVE as of plan 06 -- it was
    #: declared-and-always-`False` in plan 01 so the console never had to learn
    #: about a new field later. `cancelled=True` with `windows_written=2` and
    #: `windows_planned=4` is the honest description of a stopped run; without
    #: this flag it would be indistinguishable from a run that had only two
    #: windows of work to do.
    cancelled: bool = False
    #: Whether an `on_new_listing='rebuild'` was UNDONE before this result was
    #: published, i.e. the superseded store and ledger were renamed back and
    #: the partial rebuild was discarded. Only ever `True` together with
    #: `cancelled`: a rebuild stopped at a window boundary has not re-densified
    #: every window, so the superseded copies are still the authoritative ones
    #: and rolling back is what keeps them reachable.
    #:
    #: It is what makes `windows_written` / `rows_written` READABLE on that
    #: path: those counts describe what the loop appended, and a rollback threw
    #: exactly that away. A renderer that ignores this flag will report work
    #: that no longer exists on disk.
    rebuild_rolled_back: bool = False


class BaseDataset(ABC):
    """The shared, storage-medium-agnostic dataset contract (D-03, DATA-06).
    """

    NEW_LISTING_STRATEGIES: tuple[str, ...] = ("refuse", "rebuild", "widen")

    #: `{field name: why it is refused}` -- the fields of a FACTOR's config
    #: that a factor built over THIS dataset must not set. Empty for every
    #: vendor that has no such field; a subclass overrides it to declare one.
    #:
    #: **A DECLARATION, read by `quantlab/base/factor.py`.** The refusal it
    #: causes is raised in the factor's config setter, but the reason belongs
    #: to the vendor, so the vendor writes it and the base only reads. That
    #: direction is the point: `base/factor.py` must not learn the name of a
    #: concrete `Dataset` subclass, because this repository's layering runs
    #: `base -> dataset/factor/label -> model -> backtest` and an `isinstance`
    #: check against a vendor class in the factor base would run the other way.
    #:
    #: The live case is `CrspStockDataset` (03.11-08) and `symbols`: a factor
    #: over a CRSP panel hands `BaseFactorConfig.symbols` to
    #: `XrBackend.filter_by_symbol`, whose bare `.sel` would meet an int64
    #: PERMNO axis (D-01) and raise a mid-run `KeyError` blaming the data.
    #:
    #: The value is the SENTENCE appended to the refusal, so it must say why
    #: AND what to use instead. A reason that only says "not supported" leaves
    #: the caller exactly as stuck as the `KeyError` did.
    REJECTED_FACTOR_CONFIG_FIELDS: dict[str, str] = {}

    #: How `update()` asks `from_raw_data_chunked()` to resolve the strategy
    #: from raw-layer EVIDENCE instead of being told one.
    #:
    #: An `object()` and NOT a string, deliberately and structurally. A string
    #: sentinel would be a SECOND public way to ask for automatic resolution:
    #: reachable from `--on-new-listing`, from a config file, from a JSON
    #: round-trip. A non-string object is reachable only from code holding this
    #: private attribute, which is what makes `update()` the single public
    #: route by construction rather than by convention.
    #:
    #: Deliberately NOT a member of `NEW_LISTING_STRATEGIES`: that tuple is the
    #: CLI's `choices` source and the unknown-strategy message's contents, and
    #: an object no operator can type belongs in neither.
    _AUTOMATIC = object()

    #: How many qualifying symbols the rebuild report names before it
    #: truncates (and says that it did). A 500-symbol admission must not
    #: produce 500 log lines, but a report that named none of them would be
    #: the silent switch this whole path exists to replace.
    NEW_LISTING_REPORT_LIMIT: int = 20

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

    def __init__(self, config: BaseDatasetConfig):
        # FIRST, above the two load-bearing lines below and outside their
        # ordering rule: `from_raw_data_chunked()` publishes its outcome here
        # for `registry.convert()` to read, mirroring `Acquisition.last_result`
        # which `registry.run()` reads the same way one layer up. `None` until
        # a chunked run completes -- a dataset that was only `read()` has no
        # conversion to describe, and saying so with `None` is honest where a
        # zero-filled result would claim a run happened.
        self.last_chunk_result: "ConversionResult | None" = None

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
    def _progress_vendor(self) -> str:
        """What goes in `ProgressEvent.vendor` for a conversion event.

        `vendor` is a REQUIRED field with no default, and a conversion has no
        vendor in the acquisition sense -- it reads a raw tier that is already
        on disk and constructs no client. The field is REUSED rather than a new
        one added because `ProgressEvent` is frozen and consumed by an
        out-of-repo reporter: widening it is a contract change, reusing it is
        not. The config's own vendor token is the closest true answer (it names
        whose raw tier is being converted); the class name is the fallback for
        a config that carries none, and is never empty.
        """
        return getattr(self.config, "vendor", None) or self.class_name

    def _emit_progress(
        self, reporter: ProgressReporter | None, event: ProgressEvent
    ) -> None:
        """Deliver one event to `reporter`, never raising.

        The same contract `Acquisition._emit` holds one layer up (03.4-RESEARCH
        Pitfall 9), applied here because the console's callback now runs inside
        a SECOND loop this repository owns: unwrapped, a UI bug would propagate
        out of the window loop and end a multi-hour conversion. The exception is
        LOGGED at warning rather than swallowed, because a broken console that
        produces no signal at all is worse than a noisy one.

        **It does NOT scrub, and that asymmetry is deliberate rather than a
        missing guard.** `Acquisition._emit` runs the message through the
        vendor's `CREDENTIAL_ENV_VARS` because vendor error strings reach that
        layer and Tiingo's echo back a request URL carrying the API token. This
        loop constructs no vendor client and sees no vendor response text, so
        there is no credential-bearing string in scope -- its events carry
        window dates, counts and the dataset's class name.

        A `None` reporter is a no-op. No default reporter is instantiated here:
        `from_raw_data_chunked` already logs per-window progress through loguru,
        and a second bar beside those lines would be a behaviour change.
        """
        if reporter is None:
            return
        try:
            reporter.emit(event)
        except Exception as exc:  # noqa: BLE001 -- isolation is the point
            logger.warning(
                f"{self.class_name}: progress reporter "
                f"{type(reporter).__name__} raised on a {event.kind!r} event "
                f"and was ignored; the conversion is unaffected. "
                f"{type(exc).__name__}: {exc}"
            )

    @property
    def symbols(self) -> list[str]:
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.values.tolist()

    @property
    def time_interval(self) -> np.timedelta64:
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
        """
        return self.data_backend.head(self.config.zarr_file_path, n)

    def get_xarray_dataset(self) -> xr.Dataset:
        return self.data_backend.get_xarray_dataset(["timestamp", "symbol"])

    def from_raw_data(self) -> Self:
        """Materialise the raw source into the backend: convert, `_clean()`,
        store.
        """
        data = self._raw_data_to_xr()
        data = self._clean(data)
        self.data_backend.to_internal(data)  # type: ignore
        return self

    def _raw_axes_in_range(self) -> tuple[list, "pd.DatetimeIndex"]:
        """Return `(pinned_symbols, observed_timestamps)` for the config's
        whole date range, from ONE scan of the raw source.

        One of THREE places the pinned symbol axis is decided -- the others
        being `StockDataset._raw_axes_in_range` and
        `CrspDataset._raw_axes_in_range`. Its ORDER comes from
        `quantlab/utils/symbol_axis.py:sort_symbol_axis`, which is where that
        contract is stated and argued; do not restate it here. Its element
        TYPE is the raw panel's own (03.11-04): an unconditional `str()` here
        was handed straight back to `_raw_data_to_xr_window`'s `reindex`,
        which matches nothing against an int64 coordinate and densifies a
        whole window of NaN without raising.

        This site did not sort at all before -- it leaned on whatever
        `_raw_data_to_xr()` happened to produce. Sorting explicitly is what
        makes all three sites answer the same question the same way.
        """
        data = self._raw_data_to_xr()
        symbols = sort_symbol_axis(data["symbol"].values.tolist())
        return symbols, pd.DatetimeIndex(data["timestamp"].values)

    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset:
        """Densify ONE time window, onto `symbols` when a pinned axis is given.
        """
        data = self._raw_data_to_xr()
        data = data.sel(timestamp=slice(start_date, end_date))
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def update(
        self,
        granularity: str = "year",
        ledger_path: str | None = None,
        append_dim: str = "timestamp",
    ) -> Self:
        """Bring the store up to date -- the AUTOMATIC incremental path.
        """
        return self.from_raw_data_chunked(
            granularity=granularity,
            ledger_path=ledger_path,
            append_dim=append_dim,
            on_new_listing=self._AUTOMATIC,
        )

    def from_raw_data_chunked(
        self,
        granularity: str = "year",
        ledger_path: str | None = None,
        append_dim: str = "timestamp",
        on_new_listing: str | object = "refuse",
        *,
        reporter: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> Self:
        """Densify and append ONE time window at a time
        """
        from quantlab.base.chunking import ChunkLedger, TimeChunkPlanner

        # The sentinel is admitted by IDENTITY, beside the published tuple
        # rather than inside it. The raised message below is unchanged and
        # still lists only the three public values -- the sentinel is not
        # advertised, because nothing a user can type could reach it anyway.
        if (
            on_new_listing is not self._AUTOMATIC
            and on_new_listing not in self.NEW_LISTING_STRATEGIES
        ):
            raise ValueError(
                f"{self.class_name}: unknown on_new_listing strategy "
                f"{on_new_listing!r}; accepted values are "
                f"{list(self.NEW_LISTING_STRATEGIES)}."
            )

        symbols, timestamps = self._raw_axes_in_range()
        planner = TimeChunkPlanner(granularity)
        windows = planner.plan_from_timestamps(timestamps)
        resolved_ledger_path = ledger_path or ChunkLedger.default_path(
            self.config.zarr_file_path
        )
        ledger = ChunkLedger(resolved_ledger_path, append_dim=append_dim)

        # Reconcile the pinned axis against the STORE's before the ledger
        # check, because two of the three strategies change what that check is
        # looking at: `widen` makes the store agree with the pinned axis, and
        # `rebuild` moves the store out of the way so the run becomes a first
        # run. `refuse` changes nothing and lets `assert_consistent` raise
        # exactly as it did before this knob existed.
        #
        # `append_dim_size` goes to reconciliation SEPARATELY from the in-loop
        # `widen_and_append` call below, and both are needed. Reconciliation
        # runs BEFORE the loop and its `widen` branch rewrites the whole store
        # with `mode="w"`, re-pinning the on-disk chunk grid from whatever the
        # store holds at that moment -- which is less than the whole range on
        # exactly the paths that reach here, a crash-resume or a rolled-back
        # rebuild. The loop's copy cannot cover it, because the loop has not
        # started. `len(timestamps)` is the same D-02 once-resolved axis both
        # call sites read, and the two must not diverge.
        ledger, rebuild_asides = self._reconcile_new_listings(
            symbols,
            ledger,
            append_dim,
            on_new_listing,
            append_dim_size=len(timestamps),
        )
        # Set BEFORE the `try`, so the result construction below can read it on
        # every path out of the loop rather than only the one that assigns it.
        # ANNOTATED rather than bare, and deliberately: this is the SEED, not a
        # claim about a rollback. Since plan `03.6-09` the only place that
        # asserts a rollback happened is the cancelled arm below, which now
        # takes `_restore_rebuild_asides`'s own answer -- so the one remaining
        # literal in this function must be visibly the declaration of a
        # default and not a second, silent source of truth.
        rebuild_rolled_back: bool = False

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
            # Accounting for `last_chunk_result`. Counted INSIDE the existing
            # loop rather than reconstructed from the ledger afterwards: the
            # ledger records completed work across ALL runs, so reading it at
            # the end would report an earlier run's windows as this one's --
            # which is the exact distinction `resumed` exists to draw.
            windows_written = 0
            windows_skipped = 0
            rows_written = 0
            cancelled = False
            peak_window_bytes: int | None = None
            self._emit_progress(
                reporter,
                ProgressEvent(
                    kind="conversion_started",
                    vendor=self._progress_vendor,
                    total=len(windows),
                    message=(
                        f"{self.class_name} {granularity} conversion "
                        f"-> {self.config.zarr_file_path}"
                    ),
                    detail={
                        "pinned_symbols": len(symbols),
                        "granularity": granularity,
                        "zarr_path": self.config.zarr_file_path,
                    },
                ),
            )
            for start, end in windows:
                # The FIRST thing each iteration does, and the only place the
                # token is read (T-03.5-19). An operator can stop while the
                # pinned-axis scan is still running, so a loop that asked
                # after materialising would pay for a window nobody wanted.
                if cancel is not None and cancel.is_cancelled():
                    cancelled = True
                    # The resumability promise is TRUE of an ordinary run --
                    # the ledger records completed windows -- and FALSE of a
                    # rebuild, whose partial store is rolled back below so the
                    # superseded copies survive. Saying so here rather than
                    # letting the operator discover it from the rollback line.
                    resumability = (
                        "This run is a rebuild, so the windows written so far "
                        "will be DISCARDED and the pre-rebuild store restored; "
                        "a later rebuild starts over."
                        if rebuild_asides is not None
                        else "Every recorded window stays resumable."
                    )
                    logger.warning(
                        f"{self.class_name}: cancel observed at the "
                        f"{start.date()}..{end.date()} window boundary; "
                        f"stopping with {windows_written} window(s) written "
                        f"this run. {resumability}"
                    )
                    self._emit_progress(
                        reporter,
                        ProgressEvent(
                            kind="cancelled",
                            vendor=self._progress_vendor,
                            completed=windows_written + windows_skipped,
                            total=len(windows),
                            message=(
                                f"cancelled at {start.date()}..{end.date()}"
                            ),
                            detail={"granularity": granularity},
                        ),
                    )
                    break

                if ledger.is_written(start, end):
                    windows_skipped += 1
                    logger.info(
                        f"{self.class_name}: window {start.date()}..{end.date()} "
                        f"already recorded in the ledger, skipping."
                    )
                    self._emit_progress(
                        reporter,
                        ProgressEvent(
                            kind="window_skipped",
                            vendor=self._progress_vendor,
                            completed=windows_written + windows_skipped,
                            total=len(windows),
                            message=(
                                f"skipped {start.date()}..{end.date()} "
                                f"(already in the ledger)"
                            ),
                            detail={
                                "start": str(start.date()),
                                "end": str(end.date()),
                            },
                        ),
                    )
                    continue

                window = self._raw_data_to_xr_window(start, end, symbols)
                # Compared in the PINNED axis's own spelling, not coerced to
                # text. A `str()` here read every int64 PERMNO axis (D-01) as
                # `['14593']` against a pinned `[14593]` and refused every
                # window of a CRSP conversion; on a ticker axis `.tolist()`
                # already yields the same `str` the pinned list holds, so this
                # is byte-identical there.
                actual = list(window["symbol"].values.tolist())
                if actual != list(symbols):
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
                # Measured AFTER `_clean`/`_pin_append_dtypes`, because that
                # is the object the append actually holds -- an upcast during
                # dtype pinning is part of the peak, not an accounting detail.
                window_bytes = int(window.nbytes)
                if (
                    peak_window_bytes is None
                    or window_bytes > peak_window_bytes
                ):
                    peak_window_bytes = window_bytes
                if start != first_timestamp:
                    boundaries += 1

                self.data_backend.to_internal(window)
                # THREE axes, not one, through the single composed backend
                # entry point -- the reconciliation itself lives in
                # `XrBackend.widen_and_append` and is not reimplemented here.
                #
                # An argued behaviour change rather than an oversight, on
                # three grounds. (1) When the symbol axis AND the variable set
                # already agree -- every existing test and every existing
                # production call -- the composed method delegates to the
                # IDENTICAL `append()`, and an absent store delegates too, so
                # existing behaviour is preserved by construction; measured,
                # the suite stayed at its full pass count when this swap was
                # applied alone. (2) `on_new_listing` expresses intent about
                # the SYMBOL axis only, so no explicit caller's stated intent
                # is being overridden on the variable axis. (3) The dangerous
                # shape -- a vendor column RENAME -- presents as one variable
                # missing plus one new, and the MISSING half is still refused
                # unconditionally, so it still HALTS: no window written, no
                # history truncated.
                #
                # The closing `append()` INSIDE the composed method is what
                # keeps the overlap and dtype refusals live on this path. A
                # conditional writer keyed on the strategy would put two write
                # paths in one loop and give the guard ordering two places to
                # drift apart.
                #
                # ACCEPTED SIDE EFFECT, measured rather than reasoned about:
                # because the composed method commits its widens BEFORE that
                # closing append, a rename-shaped window leaves the store
                # carrying BOTH the stored name and the newly-introduced one
                # -- the new one materialised all-NaN across the store's
                # existing extent -- whereas the plain append this replaces
                # refused the identical shape without touching the store at
                # all. The append dimension stays where it was and every
                # stored value stays bit-identical either way, which is why
                # this is accepted rather than prevented. Locked by
                # `tests/test_chunked_ingest.py::
                # test_a_window_missing_a_stored_variable_is_still_refused`,
                # so this comment describes an asserted fact.
                #
                # `append_dim_size` states the WHOLE range's extent so the
                # store's on-disk chunk grid is a property of the store rather
                # than of whichever window created it -- `len(timestamps)` is
                # D-02's once-resolved axis, already in hand before this loop
                # began. Passed on EVERY iteration, never guarded by a window
                # index: `append()` consults it only when the store does not
                # yet exist, and the creating write is NOT iteration zero on
                # either of the two paths that matter -- a resume skips the
                # windows the ledger already records, and
                # `on_new_listing="rebuild"` moves the store aside so a later
                # call creates it. Store existence is the condition and
                # `append()` already owns it.
                self.data_backend.widen_and_append(
                    self.config.zarr_file_path,
                    append_dim=append_dim,
                    fill_values=self._widen_fill_values(),
                    append_dim_size=len(timestamps),
                )
                ledger.record(
                    start, end, int(window.sizes[append_dim]), symbols
                )
                windows_written += 1
                rows_written += int(window.sizes[append_dim])
                logger.info(
                    f"{self.class_name}: appended window "
                    f"{start.date()}..{end.date()} "
                    f"({int(window.sizes[append_dim])} row(s))."
                )
                self._emit_progress(
                    reporter,
                    ProgressEvent(
                        kind="window_written",
                        vendor=self._progress_vendor,
                        completed=windows_written + windows_skipped,
                        total=len(windows),
                        message=f"appended {start.date()}..{end.date()}",
                        detail={
                            "start": str(start.date()),
                            "end": str(end.date()),
                            "rows": int(window.sizes[append_dim]),
                            "window_bytes": window_bytes,
                        },
                    ),
                )

            # After the loop DRAINED, whether it ran to the end or stopped on a
            # cancel -- a stopped conversion still finished the call, and a
            # reporter that only ever saw a finish event on the happy path
            # would have no way to release what it opened on the other.
            self._emit_progress(
                reporter,
                ProgressEvent(
                    kind="conversion_finished",
                    vendor=self._progress_vendor,
                    completed=windows_written + windows_skipped,
                    total=len(windows),
                    message=(
                        f"{self.class_name}: {windows_written} written, "
                        f"{windows_skipped} skipped, "
                        f"{len(windows)} planned"
                    ),
                    detail={
                        "windows_written": windows_written,
                        "windows_skipped": windows_skipped,
                        "windows_planned": len(windows),
                        "rows_written": rows_written,
                        "cancelled": cancelled,
                    },
                ),
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
                if cancelled:
                    # A cancel is a HALFWAY EXIT, not a landing. The loop
                    # `break`s out of the middle of a re-densify, so the
                    # partial store at the authoritative path holds only the
                    # windows this run got to -- and the superseded copies are
                    # still the only complete record of everything the store
                    # held before. Discarding them here (which this arm did
                    # until now, because `break` lands in the SUCCESS arm)
                    # is silent, unrecoverable data loss: `rebuild` exists
                    # precisely for the case where raw can no longer
                    # reconstruct what the store had (symbols dropped from
                    # the roster have no raw rows in the current window).
                    #
                    # Treated like the exception arm rather than merely
                    # KEEPING the asides, because keeping them would leave
                    # the TRUNCATED partial store sitting at the path every
                    # reader resolves, with the complete copy hidden behind a
                    # `.superseded.tmp` suffix nothing looks at. A rollback
                    # costs the cancelled run's re-densify work; it does not
                    # cost data.
                    # The METHOD's answer, not an assumption: a restore
                    # that the filesystem refused reports `False` (WR-04),
                    # and `rebuild_rolled_back` must not claim a rollback
                    # that did not happen.
                    rebuild_rolled_back = self._restore_rebuild_asides(
                        rebuild_asides, reason="cancelled"
                    )
                else:
                    self._discard_rebuild_asides(rebuild_asides)

        # Published only on the SUCCESS path, and deliberately: the `except`
        # arm above re-raises, so a caller that sees an exception must not
        # also find a result object describing a partial run as if it had
        # finished. A stale `last_chunk_result` from an earlier successful run
        # would be worse still, which is why `__init__` seeds it to `None`.
        self.last_chunk_result = ConversionResult(
            zarr_path=self.config.zarr_file_path,
            ledger_path=resolved_ledger_path,
            granularity=granularity,
            pinned_symbols=len(symbols),
            windows_planned=len(windows),
            windows_written=windows_written,
            windows_skipped=windows_skipped,
            rows_written=rows_written,
            peak_window_bytes=peak_window_bytes,
            resumed=windows_skipped > 0,
            cancelled=cancelled,
            rebuild_rolled_back=rebuild_rolled_back,
        )
        return self

    #: Appended to the store and ledger paths while a `rebuild` is in flight.
    #:
    #: **That comment called the shared suffix deliberate and was right about
    #: WHY, but it did not anticipate what sharing it COSTS.** SUPERSEDED by
    #: phase 03.6's third gap-closure pass (plan `03.6-10`); the original
    #: wording is kept above rather than deleted so the correction is legible
    #: (D-18). It read: "Deliberately the same suffix
    #: `XrBackend.widen_symbol_axis` uses: both are
    #: 'the previous authoritative copy, kept until the replacement lands'."
    #: That claim is still TRUE and is not retracted -- the two artefacts
    #: really do mean the same thing.
    #:
    #: The consequence it missed: both producers write the SAME suffix after
    #: the SAME store path, so on disk a `rebuild` aside is INDISTINGUISHABLE
    #: from a crashed widen's aside. And the rebuild aside can outlive its
    #: run: `_restore_rebuild_asides` executes only on an exception or a
    #: cancel, and a SIGKILL reaches neither, so a killed
    #: `on_new_listing="rebuild"` leaves `<store>.superseded.tmp` on disk with
    #: nothing left to reclaim it (on the resume, `_reconcile_new_listings`
    #: sees the partial store's roster already on the new union, falls
    #: through, and `_discard_rebuild_asides` never runs).
    #:
    #: Before plan `03.6-10` that residue wedged the widen path: with a store
    #: AND a non-empty residue both present, `XrBackend.widen_symbol_axis` had
    #: no guard, so it materialised and wrote the ENTIRE sidecar and only then
    #: raised from the closing rename -- measured `OSError: [Errno 66]
    #: Directory not empty`, the whole rewrite thrown away and an orphaned
    #: `.widening.tmp` holding a complete widened store left behind. That
    #: method now decides the state BEFORE any write: a non-empty residue is
    #: refused with a message naming both producers and the manual remedy, an
    #: empty one is removed. Locked by `tests/test_widen_crash_residue.py`.
    #:
    #: The fix here is SINGLE DEFINITION rather than a rename: this is a
    #: reference to `XrBackend.SUPERSEDED_SUFFIX`, so the two cannot drift
    #: apart and the widen guard cannot stop recognising the aside it exists
    #: to notice. Renaming the dataset side (the review's other branch) was
    #: DECLINED: it would change an on-disk artefact name for no safety the
    #: guard does not already provide, and would make an existing
    #: `.superseded.tmp` residue on a real operator's disk invisible to the
    #: very code meant to notice it.
    SUPERSEDED_SUFFIX = XrBackend.SUPERSEDED_SUFFIX

    @staticmethod
    def _stored_symbol_axis(
        store_path: str, dim: str = "symbol"
    ) -> Optional[list]:
        """The store's `dim` labels, or None when there is no store (or no such
        coordinate).

        Opened the way `ChunkLedger._store_tail` opens it -- lazily, coordinate
        only, closed in a `finally`. Reading the data variables to answer an
        axis question would defeat the whole point of chunking.

        **The labels come back in the store's OWN spelling** (03.11-04). This
        method answers "what IS the store's axis", and rendering the answer as
        text is not a formatting choice -- it is a type decision taken on the
        caller's behalf, and the caller cannot see it was taken. The one
        caller, `_reconcile_new_listings`, subtracts this list from the pinned
        whole-range axis: against an int64 PERMNO store the stringified answer
        put the two sides in different alphabets, so `added` became the ENTIRE
        pinned axis and `removed` the ENTIRE stored one, at the same time.
        Downstream that is either an `on_new_listing="refuse"` halt for a
        reason that is not true, or a `"widen"` handed a target axis that is
        not a superset of the stored one -- the exact input
        `XrBackend.widen_symbol_axis` used to answer by silently emptying the
        store (fixed in 03.11-02, which now refuses it instead).

        On a ticker axis `.tolist()` already yields `str`, so this is
        byte-for-byte what the old spelling produced.
        """
        if not Path(store_path).exists():
            return None
        store = xr.open_zarr(store_path)
        try:
            if dim not in store.coords:
                return None
            return list(store[dim].values.tolist())
        finally:
            store.close()

    @staticmethod
    def _stored_append_extent(
        store_path: str, append_dim: str = "timestamp"
    ) -> Optional[tuple]:
        """The store's FIRST and LAST `append_dim` label, or None when there is
        no store (no such coordinate, or a zero-length axis).
        """
        if not Path(store_path).exists():
            return None
        store = xr.open_zarr(store_path)
        try:
            if append_dim not in store.coords:
                return None
            values = store[append_dim].values
            if not len(values):
                return None
            return (values[0], values[-1])
        finally:
            store.close()

    def _added_symbols_with_raw_history(
        self, added: list, start, end
    ) -> dict[str, int]:
        """Which of `added` already carry raw rows in the CLOSED window
        `[start, end]`, and how many -- the EVIDENCE `update()` resolves the
        widen-vs-rebuild choice from.

        **The `str()` below is DELIBERATE and must stay** (03.11-04). It looks
        like the sibling strong-cast removed from `_stored_symbol_axis`, and it
        is not: that one compared against the PINNED axis (int64 on a PERMNO
        panel), this one compares against the RAW tier's `symbol` column, and
        CRSP's raw `symbol` is the PERMNO in its STRING form -- see
        `quantlab/acquisition/wrds/crsp.py:317-319`, which writes it that way
        and which D-11 forbids this phase from touching. `str(10107)` is
        exactly `"10107"`, so the cast is what makes the two sides meet.
        Removing it would make the probe find zero raw rows for every added
        PERMNO, and the widen-vs-rebuild resolver would silently choose
        `widen` -- backfilling NaN over history the vendor already has.

        Written down because this is a place that LOOKS like it should change
        and must not; the next person through would otherwise fix it.
        """
        wanted = [str(symbol) for symbol in added]
        if not wanted:
            return {}

        if (
            type(self)._added_symbols_with_raw_history
            is BaseDataset._added_symbols_with_raw_history
        ):
            logger.warning(
                f"{self.class_name}: _added_symbols_with_raw_history has not "
                f"been overridden, so the raw-history probe is answered by "
                f"densifying the whole window and counting. The answer is "
                f"correct but the memory bound is absent -- override the seam "
                f"for a source that can push a symbol predicate down before "
                f"materialising."
            )

        window = self._raw_data_to_xr_window(start, end, symbols=None)
        if "symbol" not in window.coords:
            return {}
        present = {str(label) for label in window["symbol"].values.tolist()}

        counts: dict[str, int] = {}
        for symbol in wanted:
            if symbol not in present:
                continue
            column = window.sel(symbol=symbol)
            observed = None
            for variable in column.data_vars.values():
                notnull = variable.notnull()
                observed = notnull if observed is None else (observed | notnull)
            if observed is None:
                continue
            rows = int(np.count_nonzero(observed.values))
            if rows:
                counts[symbol] = rows
        return counts

    def _widen_fill_values(self) -> dict:
        """Per-variable fill values for a `widen`'s reindex.
        """
        return {"anomaly_flag": False}

    def _resolve_new_listing_strategy(
        self,
        added: list,
        removed: list,
        store_path: str,
        append_dim: str,
    ) -> str:
        """Read the widen-vs-rebuild choice off the raw tier and SAY it.
        """
        if removed:
            logger.warning(
                f"{self.class_name}: {len(removed)} symbol(s) present in the "
                f"store at {store_path} are absent from the pinned "
                f"whole-range axis ({sorted(removed)[: self.NEW_LISTING_REPORT_LIMIT]}"
                f"{', truncated' if len(removed) > self.NEW_LISTING_REPORT_LIMIT else ''}), "
                f"so this resolves to 'refuse'. NEITHER other strategy is safe "
                f"for a dropped label: 'widen' cannot express one at all "
                f"(widen_symbol_axis refuses a target axis that is not a "
                f"superset of the stored one), and 'rebuild' would silently "
                f"DISCARD that label's stored history. Choosing between losing "
                f"history and halting is an operator's call -- re-run "
                f"from_raw_data_chunked() with an explicit on_new_listing once "
                f"you know which you mean."
            )
            return "refuse"

        extent = self._stored_append_extent(store_path, append_dim)
        if extent is None:
            logger.info(
                f"{self.class_name}: the store at {store_path} has no "
                f"{append_dim} extent to lose (absent, or a zero-length axis), "
                f"so there is no history a NaN backfill could destroy; "
                f"resolving to 'widen' without asking raw anything."
            )
            return "widen"

        start, end = extent
        evidence = self._added_symbols_with_raw_history(added, start, end)
        if not evidence:
            logger.info(
                f"{self.class_name}: the raw tier was asked about all "
                f"{len(added)} added symbol(s) over the STORE's own extent "
                f"({start}..{end}) and reported no rows there for any of them. "
                f"They are genuine new listings, NaN is the correct value over "
                f"the store's history, and a rebuild would be pure cost; "
                f"resolving to 'widen'."
            )
            return "widen"

        ranked = sorted(evidence.items(), key=lambda item: (-item[1], item[0]))
        shown = ranked[: self.NEW_LISTING_REPORT_LIMIT]
        listing = ", ".join(f"{symbol}={rows}" for symbol, rows in shown)
        truncation = (
            f" (top {len(shown)} of {len(ranked)} by raw row count; the rest "
            f"are not listed)"
            if len(ranked) > len(shown)
            else ""
        )
        logger.warning(
            f"{self.class_name}: {len(ranked)} of {len(added)} added symbol(s) "
            f"ALREADY carry raw rows inside the store's own extent "
            f"({start}..{end}), so this resolves to 'rebuild' rather than "
            f"'widen' -- a widen does not re-read raw, so it would replace "
            f"that existing history with NaN and nothing would report it. "
            f"Qualifying symbol(s) by raw row count: {listing}{truncation}. "
            f"Rebuild is a WHOLE-STORE operation, not a per-symbol one, so "
            f"this run re-densifies EVERY window."
        )
        return "rebuild"

    def _reconcile_new_listings(
        self,
        symbols: list,
        ledger,
        append_dim: str,
        on_new_listing: str,
        *,
        append_dim_size: Optional[int] = None,
    ) -> tuple:
        """Apply `on_new_listing` when the STORE's symbol axis has drifted from
        the pinned whole-range one. Returns `(ledger, rebuild_asides)`.
        """
        store_path = self.config.zarr_file_path
        stored = self._stored_symbol_axis(store_path)
        if stored is None or stored == list(symbols):
            return ledger, None

        added = [symbol for symbol in symbols if symbol not in set(stored)]
        removed = [symbol for symbol in stored if symbol not in set(symbols)]

        # The ONE point the sentinel is exchanged for a published strategy.
        # Deliberately AFTER the early fall-through above, which already
        # guarantees the axes genuinely drifted -- so the resolver's raw probe
        # is never paid on the common case, and there is no second
        # drift-detection site to drift apart from this one.
        if on_new_listing is self._AUTOMATIC:
            on_new_listing = self._resolve_new_listing_strategy(
                added, removed, store_path, append_dim
            )

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
                f"on_new_listing='rebuild' for that, which is now the ONLY "
                f"reason to prefer it: the widen sizes itself and rewrites the "
                f"store block by block when it would not fit in memory "
                f"(XrBackend.MAX_WIDEN_BYTES), so 'rebuild' is no longer the "
                f"strategy a large store forces."
            )
            self.data_backend.widen_symbol_axis(
                store_path,
                list(symbols),
                append_dim=append_dim,
                fill_values=self._widen_fill_values(),
                append_dim_size=append_dim_size,
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

    def _restore_rebuild_asides(
        self, asides: dict, *, reason: str = "failed"
    ) -> bool:
        """Put the pre-rebuild store and ledger back, discarding the partial.
        """
        try:
            if Path(asides["store"]).exists():
                shutil.rmtree(asides["store"], ignore_errors=True)
            os.replace(asides["store_aside"], asides["store"])
            Path(asides["ledger"]).unlink(missing_ok=True)
            if asides["ledger_existed"]:
                os.replace(asides["ledger_aside"], asides["ledger"])
            logger.warning(
                f"{self.class_name}: the rebuild of {asides['store']} was "
                f"{reason}; the pre-rebuild store and ledger have been "
                f"restored. Any window this run re-densified has been "
                f"discarded with the partial store -- a resumed rebuild "
                f"starts over."
            )
            return True
        except OSError as exc:
            logger.error(
                f"{self.class_name}: the rebuild of {asides['store']} was "
                f"{reason}, but the pre-rebuild copy could NOT be put back: "
                f"{type(exc).__name__}: {exc}. NOTHING WAS DESTROYED -- the "
                f"complete pre-rebuild copy is STILL at "
                f"{asides['store_aside']}, and the partial rebuild is at "
                f"{asides['store']}. Recover by hand: once you have dealt "
                f"with whatever refused the rename, move "
                f"{asides['store_aside']} back over {asides['store']}. This "
                f"is reported rather than raised so a cleanup error cannot "
                f"replace the failure that stopped the rebuild."
            )
            return False

    @staticmethod
    def _discard_rebuild_asides(asides: dict) -> None:
        """The rebuild landed -- drop the superseded copies."""
        shutil.rmtree(asides["store_aside"], ignore_errors=True)
        Path(asides["ledger_aside"]).unlink(missing_ok=True)

    @staticmethod
    def _pin_append_dtypes(data: xr.Dataset) -> xr.Dataset:
        """Promote integer data variables to float64 before an append.
        """
        promoted = {
            name: variable.astype("float64")
            for name, variable in data.data_vars.items()
            if np.issubdtype(variable.dtype, np.integer)
        }
        return data.assign(**promoted) if promoted else data

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Cleaning hook run after raw-to-xarray conversion, before persistence.
        """
        return clean_market_data(data)

    @abstractmethod
    def _raw_data_to_xr(self) -> xr.Dataset: ...


class MarketDataset(BaseDataset):
    """The market-data dataset backend.
    """

    # Narrowed for readers and type checkers only
    config: DatasetConfig

    # D-26: the config class `quantlab/utils/module.py` rebuilds this dataset with.
    config_cls = DatasetConfig

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

    @abstractmethod
    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset: ...
