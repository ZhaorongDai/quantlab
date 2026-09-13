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

# `quantlab/base/progress.py` is a LEAF by contract -- its module docstring
# says so and `tests/test_acquisition_progress.py` asserts it structurally by
# walking that file's `ast`. It imports stdlib plus `tqdm` and nothing from
# quantlab, so importing it HERE introduces no cycle, which is what lets the
# chunk loop carry the same two handles the acquisition loop already carries
# (03.5 D-05).
from quantlab.base.progress import CancelToken, ProgressEvent, ProgressReporter
from quantlab.dataset.backend import XrBackend
from quantlab.dataset.cleaning import clean_market_data
from quantlab.enums.constant import Date
from quantlab.utils.timer import Timer


@dataclass(frozen=True)
class ConversionResult:
    """What ONE raw->Zarr conversion did, as a value the caller can render.

    Returned by `quantlab/acquisition/registry.py:convert()` (03.5 D-04), so
    an in-process caller -- the out-of-repo `quantlab-console` first -- learns
    the outcome without reading disk, and without this layer printing a line.

    **Defined HERE, in `base/`, for the direction stated at
    `quantlab/base/acquisition.py:113-118`** for `AcquisitionResult`: the base
    layer must not import the acquisition package, so a result type living
    beside the registry would have to be imported backwards or duplicated.
    `registry.py` imports it from here, exactly as it already imports
    `AcquisitionResult`. It lives in `base/data.py` rather than
    `base/acquisition.py` because it describes the DATASET layer's output --
    `from_raw_data_chunked()` is what fills it in.

    **`peak_window_bytes` and `predicted_peak_bytes` are two different
    measurements and are deliberately not merged.** `peak_window_bytes` is
    OBSERVED: the `nbytes` of the largest window this run actually
    materialised, and `None` when no window was written (a fully-resumed run
    materialises nothing). `predicted_peak_bytes` is the CALLER'S OWN
    pre-flight estimate, echoed back untouched so a report can put prediction
    beside outcome.

    **Why `convert()` computes nothing of its own (03.5 D-11).** SUPERSEDED by
    phase 03.6 (SC-3). The original sentence read: "`convert()` cannot compute
    it: 03.5 D-11 puts the RAM guard at every call site, so `convert()` has no
    roster category to size against and asks for no arithmetic of its own."
    The deleted guard is not named here: phase 03.6's SC-3 gate asserts those
    symbols appear nowhere in the tree as executable references, and the
    convention this repository adopted with that annotation is to describe the
    deleted capability rather than restate its name. Phase 03.6 DELETED the
    guard, so the premise is false while the conclusion happens to survive:
    `convert()` still asks for no arithmetic of its own, but the reason is now
    that NO call site runs a RAM guard at all -- not that the guard lives
    upstream. The recorded risk is UNMITIGATED rather than
    mitigated-at-the-call-site. `predicted_peak_bytes` survives as an
    out-of-repo caller's OWN estimate, echoed so a report can put prediction
    beside outcome; passing it buys no protection. See
    `quantlab/acquisition/registry.py` -- `convert()`'s own docstring carries
    the fuller treatment of this same D-11 supersession.

    **Carries paths, integer counts and booleans, and nothing else**
    (T-03.5-02). No vendor response body, no exception text, no environment
    value -- the credential rule the registry module docstring states in full
    applies to every egress from this repository, and a result object rendered
    into a console screenshot is an egress.
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

    NEW_LISTING_STRATEGIES: tuple[str, ...] = ("refuse", "rebuild", "widen")

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
        """
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
        default.

        No run-time warning accompanies that default any more, on either seam.
        For a market dataset the obligation is enforced at construction
        instead -- `MarketDataset` re-declares `_raw_data_to_xr_window`
        abstract (D-08) -- and for a non-market dataset the default is the
        intended behaviour rather than a degradation worth warning about.
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

        **Who still inherits this.** Non-market datasets -- an index-membership
        panel, say, which subclasses `BaseDataset` directly -- and for them the
        default is the intended behaviour, not a degradation. `MarketDataset`
        RE-DECLARES this method `@abstractmethod` (D-08), so a market dataset
        never reaches this body: the seam is the single entrance ticket to
        chunked conversion, and an obligation that can be met by accident is
        not an obligation. A market source with no windowed densify used to
        get a `logger.warning` and silent degradation to whole-range densify
        plus slice; it now fails to instantiate, with the missing method
        named. `StockDataset._raw_data_to_xr_window` is the reference
        implementation a new author copies the shape of;
        `SpotKlineDataset._raw_data_to_xr_window` is what an honest
        not-yet-bounded implementation looks like.

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

    def update(
        self,
        granularity: str = "year",
        ledger_path: str | None = None,
        append_dim: str = "timestamp",
    ) -> Self:
        """Bring the store up to date -- the AUTOMATIC incremental path.

        `Factor.update()`'s counterpart at the dataset layer, and the split
        means the same thing on both sides: `from_raw_data()` and
        `from_raw_data_chunked()` CONVERT with an explicit strategy and their
        behaviour is unchanged; `update()` EXTENDS and works out for itself
        what the strategy has to be. Which one a caller reaches for is how it
        says which it means.

        **There is no strategy parameter, and its absence is the feature.**
        The choice between widening the store's symbol axis and rebuilding it
        from raw is not a preference -- it is a FACT about the raw tier, and a
        caller who guesses it wrong loses data invisibly. So it is read rather
        than asked for. Three-way, from evidence:

        - ANY newly-added symbol already carrying raw rows INSIDE the store's
          own append-dim extent -> `rebuild`. Those rows are history a widen
          would replace with NaN, leaving the store indistinguishable from one
          where the data never existed. Whole-store, because rebuild is not a
          per-symbol operation.
        - NO added symbol carrying such rows -> `widen`. They are genuine new
          listings, NaN is the correct value over the store's history, and a
          rebuild would be pure cost.
        - ANY REMOVED symbol -> `refuse`. `widen` cannot express a dropped
          label at all (`XrBackend.widen_symbol_axis` refuses a target axis
          that is not a superset of the stored one) and `rebuild` would
          silently discard that label's stored history. Neither is safe to
          choose without an operator, so this path halts rather than inventing
          an answer.

        The decision is SPOKEN before a rebuild runs -- how many added symbols
        qualified and, for each, its raw row count inside the store's extent.
        A silent strategy switch is the same opacity as a wrong flag, in the
        other direction.

        The probe is only ever paid when the symbol axes actually DRIFTED, and
        it is asked about the STORE's extent rather than the config's range.
        Measured ordering (see `_added_symbols_with_raw_history`): the extent
        probe costs less than a whole-tier probe, which costs less than the
        `_raw_axes_in_range()` scan every chunked run already pays
        unconditionally.

        **No route to overwrite a range the store already holds.** There is no
        `mode` parameter, nothing named to loosen or bypass a guard, and the
        unconditional append-dim overlap refusal is inherited through the
        UNCHANGED backend append. That guard runs before any keyword is read,
        so no argument gets past it whatever it is named. Re-deriving a stored
        range is the wholesale interface's job.

        Everything else -- window planning, the ledger, resume, the per-window
        loop, rebuild-aside restoration -- is `from_raw_data_chunked()`'s,
        forwarded verbatim rather than reimplemented. This method differs from
        it in exactly one respect: where the strategy comes from.
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

        Its type admits a NON-`str` arm for exactly ONE private sentinel,
        `_AUTOMATIC`, which `update()` passes and nothing else may. The runtime
        check against it is an IDENTITY comparison against that one object, so
        the widened annotation documents what is REACHABLE rather than inviting
        arbitrary values. The annotation moved because leaving it claiming
        `str` after a non-`str` value became legal would be FALSE, and a false
        annotation misleads in the one direction that matters here: it implies
        some string is the automatic route, which is precisely what the
        sentinel exists to forbid. The DEFAULT VALUE is untouched, so every
        caller passing an explicit strategy gets byte-identical behaviour.

        **`reporter` and `cancel` are the console's two handles on a running
        conversion** (03.5 D-05), word for word the pair `registry.run()`
        carries for an acquisition and for the identical reason: neither can
        be added from outside, because the loop lives here. Both are
        KEYWORD-ONLY with `None` defaults, so every existing call site --
        including `update()`'s -- is unchanged and gets today's behaviour
        exactly: loguru progress lines, and no way to stop the run early.

        No default reporter is instantiated when none is supplied. This method
        already logs its own per-window progress through loguru, and opening a
        stderr bar beside those lines would be a behaviour change wearing a
        feature's clothes.

        `cancel` is observed at WINDOW BOUNDARIES only -- at the top of each
        iteration, before the window is materialised. Stopping mid-window
        would leave a densified panel unappended for no benefit, and a check
        between the append and `ledger.record` would leave the store and the
        ledger disagreeing about the same window (T-03.5-19). Every window
        recorded before the stop stays resumable, which is the
        completed-work-stays-resumable precondition `ChunkLedger` already
        supplies and the acquisition side had to retrofit with atomic
        sidecars.

        Neither object is ever assigned onto `self.config`.
        `BaseDatasetConfig.to_dict()` is `asdict(self)` and lands on disk
        beside model checkpoints, where a `threading.Event` cannot be
        serialised and a live reporter object is not reproducible
        configuration.
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
        rebuild_rolled_back = False

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
                actual = [
                    str(symbol) for symbol in window["symbol"].values.tolist()
                ]
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
                # Measured AFTER `_clean`/`_pin_append_dtypes`, because that
                # is the object the append actually holds -- an upcast during
                # dtype pinning is part of the peak, not an accounting detail.
                window_bytes = int(window.nbytes)
                if peak_window_bytes is None or window_bytes > peak_window_bytes:
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
                    self._restore_rebuild_asides(
                        rebuild_asides, reason="cancelled"
                    )
                    rebuild_rolled_back = True
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
    #: Deliberately the same suffix `XrBackend.widen_symbol_axis` uses: both are
    #: "the previous authoritative copy, kept until the replacement lands".
    SUPERSEDED_SUFFIX = ".superseded.tmp"

    @staticmethod
    def _stored_symbol_axis(
        store_path: str, dim: str = "symbol"
    ) -> Optional[list]:
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

    @staticmethod
    def _stored_append_extent(
        store_path: str, append_dim: str = "timestamp"
    ) -> Optional[tuple]:
        """The store's FIRST and LAST `append_dim` label, or None when there is
        no store (no such coordinate, or a zero-length axis).

        Opened exactly the way `_stored_symbol_axis` opens it -- lazily,
        coordinate only, closed in a `finally`. Answering an extent question by
        reading data variables would defeat chunking.

        **The endpoints are taken by INDEXING, never by reduction, and the
        length check that makes that safe is load-bearing rather than
        defensive.** A zero-length append axis is a real shape in this repo,
        not a hypothetical: `data/data/us_equity/1d/stock_alpaca.zarr` carries
        a `timestamp` coordinate of size 0, and `.values.min()` on it raises
        `ValueError: zero-size array to reduction operation minimum which has
        no identity`. Any run that wrote a zero-row panel -- an acquisition
        that fetched nothing, an aborted chunked ingest, a window that pruned
        to nothing -- reaches that shape. `ChunkLedger._store_tail` already
        solves it with `values[-1] if len(values) else None`; this is the same
        idiom over both ends.

        None is the honest answer for all three cases and the resolver reads
        it as one thing: there is no stored history to lose here.
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

        A symbol absent from the mapping carries no rows there; the mapping is
        never padded with zeros, so `if probe_result:` reads as "there is
        history to recover". An empty `added` returns `{}` without touching
        raw at all, which is what keeps the common case free.

        Overridable seam, and the same caveat `_raw_axes_in_range()` carries
        applies: **this default is correct but NOT memory-bounded**, because it
        densifies the whole window through `_raw_data_to_xr_window()` and
        counts afterwards. It exists rather than a `raise NotImplementedError`
        stub so a subclass that has not overridden the densify seams --
        `SpotKlineDataset` and `IndexConstituentDataset` today -- keeps
        working, and it warns at run time when it is the one running so an
        operator learns the probe took the slow route instead of just waiting.
        A subclass whose raw source can push a symbol predicate DOWN before
        materialisation should override it.

        **Cost is recorded as an ORDERING first and absolutes second**, because
        absolutes go stale on other hardware and a stale number in a docstring
        is a claim the next reader cannot check. The durable relation:
        probing the STORE's extent costs LESS than probing the whole raw tier,
        which costs less than `_raw_axes_in_range()` -- which every chunked run
        already pays UNCONDITIONALLY. So the probe never adds more than a step
        the path was already taking, and it is paid ONLY when the symbol axes
        actually drifted (`_reconcile_new_listings` falls through before
        reaching the resolver otherwise).

        The dated reading that ordering came from: measured 2026-09-08 on this
        repo's real raw tier (26,584 `.pqt` files across 13 `month=`
        partitions, 153.1 MB apparent / 208 MiB on disk), warm cache --
        1.32-1.75 s for the store-extent window, 2.24-2.38 s for the whole
        tier, 3.22-3.32 s for `_raw_axes_in_range()`. Those seconds are machine-
        and cache-dependent and only their ORDER is load-bearing; an earlier
        reading of the same three steps on a colder cache was about 1.7x higher
        and preserved the same order.
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

    def _resolve_new_listing_strategy(
        self,
        added: list,
        removed: list,
        store_path: str,
        append_dim: str,
    ) -> str:
        """Read the widen-vs-rebuild choice off the raw tier and SAY it.

        Returns one of the three published `NEW_LISTING_STRATEGIES` values, so
        the existing branches below run untouched. This method chooses WHICH
        branch runs; it reimplements none of them.

        Lives here, at the point `added`/`removed` are already known, rather
        than in `update()` -- for two measured reasons. Cost: resolving in
        `update()` would compute the pinned whole-range axis TWICE, and
        `_raw_axes_in_range()` is the MORE expensive of the two steps, not the
        cheaper. Correctness: `_reconcile_new_listings` already owns the ONLY
        drift-detection site, and a second one is the two-independent-guards
        shape that has already produced one silent contract drift in this
        repository.
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

        Falls through completely unchanged -- no store read beyond the
        coordinate, no log line -- when the axes already agree, which is the
        overwhelmingly common case.

        `append_dim_size` is the whole range's extent, forwarded to the
        `widen` branch's `mode="w"` rewrite so the store's on-disk chunk grid
        stays a property of the store rather than of how much of it happened
        to be written when the roster changed. The other two branches do not
        need it: `rebuild` moves the store aside so the next write is a
        CREATING write, which `append()` already sizes from the same value,
        and `refuse` writes nothing at all.
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
    ) -> None:
        """Put the pre-rebuild store and ledger back, discarding the partial.

        `reason` names WHY the rebuild is being undone, so the operator reads
        "failed" for an exception and "cancelled" for their own stop rather
        than one wording standing in for both. It is the only difference
        between the two callers: an exception and a cancel are the same
        halfway exit as far as the superseded copies are concerned.
        """
        if Path(asides["store"]).exists():
            shutil.rmtree(asides["store"], ignore_errors=True)
        os.replace(asides["store_aside"], asides["store"])
        Path(asides["ledger"]).unlink(missing_ok=True)
        if asides["ledger_existed"]:
            os.replace(asides["ledger_aside"], asides["ledger"])
        logger.warning(
            f"{self.class_name}: the rebuild of {asides['store']} was "
            f"{reason}; the pre-rebuild store and ledger have been restored. "
            f"Any window this run re-densified has been discarded with the "
            f"partial store -- a resumed rebuild starts over."
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
        (float64, 8 bytes per value).

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

    A SIXTH member is owned here for a different reason:
    `_raw_data_to_xr_window` is re-declared `@abstractmethod` (D-08). It is
    not market-specific -- `BaseDataset` keeps a working concrete default for
    the dataset kinds that cannot push a date filter down -- but for a market
    dataset it is the single entrance ticket to chunked conversion, which is
    the only conversion mode the registry entry point offers. Making it
    abstract HERE and nowhere else moves "this source has no windowed
    densify" from a `logger.warning` nobody reads during a multi-hour
    backfill to a `TypeError` at the new author's first construction, while
    leaving the shared base's default intact for everyone else.
    """

    # Narrowed for readers and type checkers only
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

    @abstractmethod
    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset: ...
