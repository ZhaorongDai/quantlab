"""Dataset base classes: from raw vendor files to the canonical panel.

This module is the first layer of the pipeline. ``BaseDataset`` owns the shared
lifecycle of every dataset: a config with normalised ISO dates, a Zarr-backed
storage backend, ``from_raw_data``/``save``/``read`` for whole-range conversion,
and ``from_raw_data_chunked``/``update`` for converting one time window at a
time with a resumable ledger. ``MarketDataset`` adds the two market-data exits,
``to_kunquant`` and ``to_nautilus``. A concrete dataset lives under
``quantlab/dataset/`` and implements ``_raw_data_to_xr``; the factor layer
consumes the panel through ``get_xarray_dataset`` or ``get_lazyframe``.
See ``docs/dataset.md`` and ``docs/chunking.md``.
"""

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
from quantlab.backend import XrBackend
from quantlab.dataset._support.cleaning import clean_market_data
from quantlab.enums.constant import Date
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer


@dataclass(frozen=True)
class ConversionResult:
    """Summary of one chunked raw-to-Zarr conversion.

    ``BaseDataset.from_raw_data_chunked`` publishes an instance on
    ``last_chunk_result`` when it completes, so a caller can render what the
    run did without keeping the config alive.

    Examples
    --------
    >>> ds = DemoDataset(config).from_raw_data_chunked(granularity="month")
    >>> result = ds.last_chunk_result
    >>> result.windows_written, result.rows_written, result.resumed
    (1, 6, False)
    """

    #: The Zarr store that was written (``config.zarr_file_path``).
    zarr_path: str
    #: Path of the chunk-ledger sidecar that records completed windows.
    ledger_path: str
    #: Window granularity the run planned on (``"year"``, ``"quarter"``, ...).
    granularity: str
    #: Number of symbols on the pinned whole-range axis.
    pinned_symbols: int
    #: Number of windows the planner produced over the observed timestamps.
    windows_planned: int
    #: Number of windows this run materialised and appended.
    windows_written: int
    #: Number of windows skipped because the ledger already recorded them.
    windows_skipped: int
    #: Rows appended along the append dimension, summed over written windows.
    rows_written: int
    #: Largest ``window.nbytes`` this run materialised; ``None`` when no
    #: window was written.
    peak_window_bytes: int | None
    #: The caller's pre-flight estimate, echoed back; ``None`` when none was
    #: offered, which is the common case.
    predicted_peak_bytes: int | None = None
    #: Whether at least one window was skipped, i.e. this run continued an
    #: earlier one.
    resumed: bool = False
    #: Whether a ``CancelToken`` stopped the loop at a window boundary. A
    #: cancelled run with fewer windows written than planned is otherwise
    #: indistinguishable from a run that simply had less work to do.
    cancelled: bool = False
    #: Whether an ``on_new_listing="rebuild"`` run was rolled back before this
    #: result was published, restoring the superseded store and ledger. Only
    #: ever ``True`` together with ``cancelled``. When set, ``windows_written``
    #: and ``rows_written`` describe work that no longer exists on disk.
    rebuild_rolled_back: bool = False


class BaseDataset(ABC):
    """Storage-agnostic dataset contract shared by every dataset in quantlab.

    A subclass implements ``_raw_data_to_xr`` (raw files to a dense
    ``(timestamp, symbol)`` panel) and inherits everything else: config
    normalisation, Zarr persistence through ``XrBackend``, the cleaning hook,
    and the chunked ingestion path with its ledger and new-listing strategies.
    Datasets with no bars, KunQuant input or Nautilus catalog (a boolean
    membership panel, say) subclass this directly; market data subclasses
    ``MarketDataset``.

    Examples
    --------
    A minimal subclass and the whole storage lifecycle::

        class MembershipDataset(BaseDataset):
            def _raw_data_to_xr(self) -> xr.Dataset:
                return load_membership_panel(self.config.kwargs["source"])

        ds = MembershipDataset(config)
        ds.from_raw_data().save()
        panel = MembershipDataset(config).read().get_xarray_dataset()

    The method examples below use ``DemoDataset``, a ``MarketDataset``
    subclass whose ``_raw_data_to_xr`` returns six business days of
    synthetic OHLCV bars for ``AAA``, ``BBB`` and ``CCC``, built with a
    ``DatasetConfig`` covering ``2024-01-02`` to ``2024-01-05``.
    """

    NEW_LISTING_STRATEGIES: tuple[str, ...] = ("refuse", "rebuild", "widen")

    #: ``{field name: reason}`` for the factor-config fields that a factor
    #: built over this dataset must not set. The factor base class reads this
    #: declaration and raises in its config setter, appending the reason, so
    #: the base never needs to know a concrete dataset class. Empty by default;
    #: a subclass overrides it, for example to refuse ``symbols`` on a panel
    #: whose symbol axis is not string-typed. Each reason should say why and
    #: what to use instead.
    REJECTED_FACTOR_CONFIG_FIELDS: dict[str, str] = {}

    #: Sentinel ``update()`` passes to ``from_raw_data_chunked()`` to ask for
    #: the new-listing strategy to be resolved from raw-tier evidence. It is an
    #: ``object()`` rather than a string so that no CLI flag, config file or
    #: JSON round-trip can reach it; it is deliberately not a member of
    #: ``NEW_LISTING_STRATEGIES``, which is what the CLI offers as choices.
    _AUTOMATIC = object()

    #: How many qualifying symbols the rebuild report names before it
    #: truncates.
    NEW_LISTING_REPORT_LIMIT: int = 20

    def __init__(self, config: BaseDatasetConfig):
        """Create the storage backend and assign the config.

        The backend is created before the config is assigned because the
        config setter may reach the backend; reversing the two lines raises
        ``AttributeError``.
        """
        # Set before the two lines below: `from_raw_data_chunked()` publishes
        # its outcome here and a dataset that was only `read()` reports `None`.
        self.last_chunk_result: "ConversionResult | None" = None

        self.data_backend = XrBackend()
        self.config = config

    def __repr__(self):
        """Return ``ClassName(config=...)``."""
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def num_symbols(self) -> int:
        """Return the number of symbols in the loaded panel.

        Examples
        --------
        >>> ds.num_symbols
        3
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.size

    @property
    def class_name(self) -> str:
        """Return the concrete class name, used in log and error messages.

        Examples
        --------
        >>> ds.class_name
        DemoDataset
        """
        return self.__class__.__name__

    @property
    def _progress_vendor(self) -> str:
        """Return the value placed in ``ProgressEvent.vendor`` by a conversion.

        A conversion constructs no vendor client, so the config's own vendor
        token is used when it has one and the class name otherwise; the field
        is required and must not be empty.
        """
        return getattr(self.config, "vendor", None) or self.class_name

    def _emit_progress(
        self, reporter: ProgressReporter | None, event: ProgressEvent
    ) -> None:
        """Deliver one progress event to ``reporter`` without ever raising.

        A ``None`` reporter is a no-op. An exception raised by the reporter is
        logged at warning level and otherwise ignored, so a broken console
        cannot end a multi-hour conversion. Events carry window dates, counts
        and the class name only, so no credential scrubbing is needed here.
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
        """Return the symbol labels of the loaded panel, in axis order.

        Examples
        --------
        >>> ds.symbols
        ['AAA', 'BBB', 'CCC']
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.values.tolist()

    @property
    def time_interval(self) -> np.timedelta64:
        """Return the most common spacing between consecutive timestamps.

        The mode is used rather than the minimum so that gaps such as
        weekends do not distort the answer.

        Examples
        --------
        >>> ds.time_interval  # daily bars read back from Zarr
        np.timedelta64(86400000000000,'ns')
        """
        timestamps = self.data_backend.get_xarray_dataset(["timestamp"])[
            "timestamp"
        ]
        return (
            timestamps.diff(dim="timestamp")
            .to_series()
            .mode()
            .values[0]
        )

    @property
    def import_path(self) -> str:
        """Return the dotted ``module.QualName`` path of the concrete class.

        Examples
        --------
        >>> ds.import_path  # for a class defined in a script
        __main__.DemoDataset
        """
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def _filter(self):
        """Narrow the backend to the config's date range and symbols in place."""
        self.data_backend.filter_by_date(
            "timestamp", self.config.start_date, self.config.end_date
        )
        if self.config.symbols is not None:
            self.data_backend.filter_by_symbol("symbol", self.config.symbols)

    @property
    def config(self) -> BaseDatasetConfig:
        """Return the dataset config.

        Examples
        --------
        >>> ds.config.start_date, ds.config.end_date
        ('2024-01-02', '2024-01-05')
        >>> ds.config.name
        __main__.DemoDataset
        """
        return self._config

    @config.setter
    def config(self, config: BaseDatasetConfig):
        """Assign the config, filling in defaults and checking its dates.

        ``name`` is set to the class's import path, a missing ``start_date``
        or ``end_date`` falls back to ``Date.START_DATE``/``Date.END_DATE``,
        and both dates must be ISO ``YYYY-MM-DD`` strings (a ``date`` object
        is accepted and stringified). Every date comparison downstream is a
        plain string comparison, so a value such as ``"2007-2-1"`` would
        compare wrong rather than fail to match, and is refused here instead.

        Examples
        --------
        >>> ds.config = dataclasses.replace(ds.config, start_date="2024-01-03")
        >>> ds.config.start_date
        '2024-01-03'
        >>> ds.config = dataclasses.replace(ds.config, start_date="01/02/2024")
        Traceback (most recent call last):
        ValueError: DemoDataset: start_date must be an ISO YYYY-MM-DD date ...
        """
        self._config = config
        self._config.name = self.import_path

        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

        self._config.start_date = self._normalize_date(
            self._config.start_date, "start_date"
        )
        self._config.end_date = self._normalize_date(
            self._config.end_date, "end_date"
        )

    def _normalize_date(self, value: str, field_name: str) -> str:
        """Return ``value`` as a zero-padded ISO date string.

        Raises
        ------
        ValueError
            If ``value`` is not an ISO ``YYYY-MM-DD`` date.
        """
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
        """Return the symbol labels of the loaded panel."""
        return self.data_backend.get_xarray_dataset(
            ["symbol", "timestamp"]
        ).symbol.values.tolist()

    def read(self, **kwargs):
        """Open the Zarr store and narrow it to the config's window.

        Cleaning and densification do not happen here; they belong to
        ``from_raw_data``. Keyword arguments are passed to
        ``XrBackend.read``.

        Returns
        -------
        BaseDataset
            ``self``, for chaining.

        Examples
        --------
        >>> panel = DemoDataset(config).read().get_xarray_dataset()
        >>> dict(panel.sizes)  # narrowed to the config's four days
        {'timestamp': 4, 'symbol': 3}
        """
        self.data_backend.read(self.config.zarr_file_path, **kwargs)
        self._filter()
        return self

    def save(self, **kwargs):
        """Narrow the loaded panel to the config's window and write it to Zarr.

        The write replaces the whole store directory. Keyword arguments are
        passed to ``XrBackend.write``.

        Examples
        --------
        >>> DemoDataset(config).from_raw_data().save()
        >>> Path(config.zarr_file_path).is_dir()
        True
        """
        with Timer(f"{self.__class__.__name__}: save"):
            self._filter()
            self.data_backend.write(self.config.zarr_file_path, **kwargs)

    def get_config(self) -> dict:
        """Return the config as a plain dictionary.

        Examples
        --------
        >>> ds.get_config()["start_date"]
        '2024-01-02'
        """
        return self.config.to_dict()  # type: ignore

    def get_lazyframe(self) -> pl.LazyFrame:
        """Return the loaded panel as a long-format polars ``LazyFrame``.

        Examples
        --------
        >>> ds.get_lazyframe().collect().shape  # 4 days x 3 symbols, 8 columns
        (12, 8)
        """
        return self.data_backend.get_lazyframe()

    def head(self, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows of the store as a ``LazyFrame``.

        The store is opened by path; the loaded panel and the config window
        are left untouched, which makes this safe for probing column names.

        Examples
        --------
        >>> ds.head(2).collect().shape
        (2, 8)
        """
        return self.data_backend.head(self.config.zarr_file_path, n)

    def get_xarray_dataset(self) -> xr.Dataset:
        """Return the loaded panel indexed by ``(timestamp, symbol)``.

        Examples
        --------
        >>> tuple(ds.get_xarray_dataset().dims)
        ('timestamp', 'symbol')
        """
        return self.data_backend.get_xarray_dataset(["timestamp", "symbol"])

    def from_raw_data(self) -> Self:
        """Convert the raw source for the whole configured range into memory.

        Runs ``_raw_data_to_xr``, then ``_clean``, and hands the result to the
        backend. Nothing is written to disk until ``save`` is called. Every
        call reconverts; there is no caching.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> ds = DemoDataset(config).from_raw_data()
        >>> list(ds.get_xarray_dataset().data_vars)  # cleaning adds the flag
        ['open', 'high', 'low', 'close', 'volume', 'anomaly_flag']
        """
        data = self._raw_data_to_xr()
        data = self._clean(data)
        self.data_backend.to_internal(data)  # type: ignore
        return self

    def _raw_axes_in_range(self) -> tuple[list, "pd.DatetimeIndex"]:
        """Return ``(pinned_symbols, observed_timestamps)`` for the whole range.

        The default converts the whole range through ``_raw_data_to_xr`` and
        reads both axes off the result. The symbol order comes from
        ``sort_symbol_axis`` and the label type is the raw panel's own, so a
        later window ``reindex`` matches it. A subclass whose raw source can
        project columns should override this to avoid the full materialisation.
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
        """Return the dense panel for one time window.

        The default converts the whole range and slices it, so it bounds the
        write but not the memory used to densify. When ``symbols`` is given
        the result is reindexed onto exactly that axis, with absent symbols as
        all-NaN columns.
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
        """Bring the store up to date, choosing the new-listing strategy itself.

        Equivalent to ``from_raw_data_chunked`` with the strategy resolved
        from evidence: when the store's symbol axis has drifted from the raw
        tier, the raw tier is asked whether the added symbols already carry
        rows inside the store's own time extent. If none do they are new
        listings and the store is widened; if some do the store is rebuilt so
        their history is recovered; a symbol that disappeared from the raw
        tier resolves to ``"refuse"``. The choice is logged before it runs.

        Parameters
        ----------
        granularity : str
            Window size handed to ``TimeChunkPlanner``.
        ledger_path : str | None
            Ledger sidecar path; defaults to one beside the store.
        append_dim : str
            Dimension windows are appended along.

        Returns
        -------
        Self
            ``self``, with ``last_chunk_result`` populated.

        Examples
        --------
        >>> ds = DemoDataset(config).update(granularity="month")
        >>> ds.last_chunk_result.windows_written
        1
        >>> again = DemoDataset(config).update(granularity="month")
        >>> again.last_chunk_result.windows_skipped, again.last_chunk_result.resumed
        (1, True)
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
        """Convert the raw source one time window at a time, appending to Zarr.

        The symbol axis for the whole range is resolved once, before any
        window, and every window is densified onto it. Completed windows are
        recorded in a ledger sidecar, so a crashed or cancelled run resumes by
        skipping them. Cleaning runs per window, which means a price jump
        straddling a window boundary is not flagged.

        When the store already exists with a different symbol axis,
        ``on_new_listing`` decides what happens: ``"refuse"`` halts,
        ``"widen"`` adds the new symbols as NaN over the store's history, and
        ``"rebuild"`` renames the store aside and re-densifies every window;
        a failed or cancelled rebuild restores the original store.

        Parameters
        ----------
        granularity : str
            Window size handed to ``TimeChunkPlanner``
            (``"year"``, ``"quarter"``, ``"month"``, ...).
        ledger_path : str | None
            Ledger sidecar path; defaults to one beside the store.
        append_dim : str
            Dimension windows are appended along.
        on_new_listing : str | object
            One of ``NEW_LISTING_STRATEGIES``.
        reporter : ProgressReporter | None
            Optional progress sink for per-window events.
        cancel : CancelToken | None
            Optional token checked at every window boundary.

        Returns
        -------
        Self
            ``self``, with ``last_chunk_result`` describing the run.

        Raises
        ------
        ValueError
            If ``on_new_listing`` is unknown, or a window comes
            back on a symbol axis other than the pinned one.

        Examples
        --------
        >>> ds = MyMarketDataset(config)
        >>> ds.from_raw_data_chunked(granularity="quarter")
        >>> ds.last_chunk_result.windows_written
        8
        """
        from quantlab.base.chunking import ChunkLedger, TimeChunkPlanner

        # The private sentinel is accepted by identity; the message still lists
        # only the public strategies.
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

        # Reconcile the pinned axis against the store before the ledger check:
        # `widen` and `rebuild` both change what that check looks at. The
        # whole-range extent is passed so a widen re-pins the store's chunk
        # grid from the full range rather than from a partial store.
        ledger, rebuild_asides = self._reconcile_new_listings(
            symbols,
            ledger,
            append_dim,
            on_new_listing,
            append_dim_size=len(timestamps),
        )
        # Seeded before the `try` so the result below can read it on every path.
        rebuild_rolled_back: bool = False

        try:
            # The ledger and the store are two records of the same truth; check
            # they agree before the first irreversible append.
            ledger.assert_consistent(symbols, self.config.zarr_file_path)

            logger.info(
                f"{self.class_name}: chunked ingestion over {len(windows)} "
                f"{granularity} window(s), {len(symbols)} pinned symbol(s), "
                f"{len(timestamps)} observed timestamp(s)."
            )

            first_timestamp = timestamps.min() if len(timestamps) else None
            boundaries = 0
            # Counted inside the loop rather than read back from the ledger,
            # which also records earlier runs' windows.
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
                # Checked first, before any window is materialised.
                if cancel is not None and cancel.is_cancelled():
                    cancelled = True
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
                # Compared in the pinned axis's own type, not as text: an
                # integer-typed axis would never match its stringified form.
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
                # Measured after cleaning and dtype pinning: that is the object
                # the append actually holds.
                window_bytes = int(window.nbytes)
                if (
                    peak_window_bytes is None
                    or window_bytes > peak_window_bytes
                ):
                    peak_window_bytes = window_bytes
                if start != first_timestamp:
                    boundaries += 1

                self.data_backend.to_internal(window)
                # `widen_and_append` reconciles the symbol and variable axes
                # and then appends; when both already agree it is a plain
                # append, and a window missing a stored variable is still
                # refused. `append_dim_size` is passed on every iteration
                # because the store may be created on any of them (a resume
                # skips windows, a rebuild moves the store aside), and the
                # backend consults it only when the store does not exist yet.
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

            # Emitted whether the loop ran to the end or stopped on a cancel,
            # so a reporter can always release what it opened.
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
            # The originals were renamed aside, not deleted, so a failed
            # rebuild is undone by renaming them back.
            if rebuild_asides is not None:
                self._restore_rebuild_asides(rebuild_asides)
            raise
        else:
            if rebuild_asides is not None:
                if cancelled:
                    # A cancelled rebuild has re-densified only some windows,
                    # so the superseded copies are still the only complete
                    # record; roll back rather than leave a truncated store at
                    # the path every reader resolves. The flag takes the
                    # method's answer, since the restore itself can fail.
                    rebuild_rolled_back = self._restore_rebuild_asides(
                        rebuild_asides, reason="cancelled"
                    )
                else:
                    self._discard_rebuild_asides(rebuild_asides)

        # Published on the success path only; the `except` arm re-raises and
        # must not leave a result describing a partial run.
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

    #: Suffix appended to the store and ledger paths while a ``rebuild`` is in
    #: flight. It is a reference to ``XrBackend.SUPERSEDED_SUFFIX`` rather than
    #: a second literal so the backend's widen guard, which refuses to run over
    #: a leftover aside, always recognises the aside a rebuild leaves behind
    #: (a killed rebuild can leave one with nothing left to reclaim it).
    SUPERSEDED_SUFFIX = XrBackend.SUPERSEDED_SUFFIX

    @staticmethod
    def _stored_symbol_axis(
        store_path: str, dim: str = "symbol"
    ) -> Optional[list]:
        """Return the store's ``dim`` labels, or ``None`` if there is no store.

        Only the coordinate is read, never the data variables. Labels come
        back in the store's own type so the caller can compare them against
        the pinned axis without a spurious mismatch.
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
        """Return the store's first and last ``append_dim`` label.

        Returns ``None`` when there is no store, no such coordinate, or the
        axis is empty.
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
        """Count raw rows each of ``added`` carries in the closed window.

        This is the evidence ``update`` resolves the widen-versus-rebuild
        choice from. The default densifies the whole window and counts, which
        is correct but not memory-bounded; a subclass whose raw source can
        push a symbol predicate down should override it.

        Symbols are compared as text because the raw tier stores its symbol
        column as strings even when the pinned axis is integer-typed. Without
        the cast the probe would find no rows for any added symbol and the
        resolver would silently choose ``widen``.

        Returns
        -------
        dict[str, int]
            ``{symbol: row count}`` for the symbols that have at least one row.
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
        """Return per-variable fill values used when widening the symbol axis.

        Non-float variables need an explicit fill; the default covers the
        boolean ``anomaly_flag`` added by cleaning.
        """
        return {"anomaly_flag": False}

    def _resolve_new_listing_strategy(
        self,
        added: list,
        removed: list,
        store_path: str,
        append_dim: str,
    ) -> str:
        """Choose a new-listing strategy from raw-tier evidence and log why.

        A removed symbol resolves to ``"refuse"``, since neither ``widen``
        nor ``rebuild`` can keep its history safely. Otherwise the added
        symbols are probed over the store's own time extent: no raw rows
        there means genuine new listings and ``"widen"``; any rows means a
        widen would replace real history with NaN, so ``"rebuild"``.
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
        """Apply ``on_new_listing`` when the store's symbol axis has drifted.

        Returns ``(ledger, rebuild_asides)``. ``rebuild_asides`` is ``None``
        unless a rebuild was started, in which case it names the renamed-aside
        store and ledger so they can be restored or discarded later.
        """
        store_path = self.config.zarr_file_path
        stored = self._stored_symbol_axis(store_path)
        if stored is None or stored == list(symbols):
            return ledger, None

        added = [symbol for symbol in symbols if symbol not in set(stored)]
        removed = [symbol for symbol in stored if symbol not in set(symbols)]

        # Resolved only after the early return above, so the raw probe is
        # never paid when the axes agree.
        if on_new_listing is self._AUTOMATIC:
            on_new_listing = self._resolve_new_listing_strategy(
                added, removed, store_path, append_dim
            )

        if on_new_listing == "refuse":
            # Fall through unchanged; `assert_consistent` raises next.
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
            # The ledger fingerprints the symbol list, so it must be rebased in
            # the same operation or the widened store is not resumable.
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
        # Renamed aside, never deleted, so a failed rebuild can be undone.
        os.replace(asides["store"], asides["store_aside"])
        if asides["ledger_existed"]:
            os.replace(asides["ledger"], asides["ledger_aside"])
        # A fresh ledger at the original path makes the run an ordinary first
        # run.
        return ChunkLedger(asides["ledger"], append_dim=append_dim), asides

    def _restore_rebuild_asides(
        self, asides: dict, *, reason: str = "failed"
    ) -> bool:
        """Put the pre-rebuild store and ledger back, discarding the partial.

        Returns
        -------
        bool
            ``True`` if the originals were restored, ``False`` if the
            filesystem refused; in that case nothing is destroyed and the
            error log names both copies so an operator can recover by hand.
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
        """Delete the superseded store and ledger after a rebuild landed."""
        shutil.rmtree(asides["store_aside"], ignore_errors=True)
        Path(asides["ledger_aside"]).unlink(missing_ok=True)

    @staticmethod
    def _pin_append_dtypes(data: xr.Dataset) -> xr.Dataset:
        """Promote integer data variables to float64 before an append.

        The first window written fixes each variable's dtype in the store; an
        integer column would then silently cast later NaN cells to zero.
        """
        promoted = {
            name: variable.astype("float64")
            for name, variable in data.data_vars.items()
            if np.issubdtype(variable.dtype, np.integer)
        }
        return data.assign(**promoted) if promoted else data

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Validate and flag the converted panel before it is stored.

        The default runs ``clean_market_data`` (OHLCV schema check plus
        ``anomaly_flag``). Override it for panels whose columns are not
        lowercase OHLCV, or that are not market data at all.
        """
        return clean_market_data(data)

    @abstractmethod
    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the raw source as a dense ``(timestamp, symbol)`` panel.

        The result must cover the config's whole date range and carry a
        unique ``(timestamp, symbol)`` index; deduplicate before calling
        ``to_xarray``. Cleaning is applied by the caller, not here.
        """


class MarketDataset(BaseDataset):
    """Dataset of market bars with KunQuant and Nautilus exits.

    Adds ``to_kunquant`` (contiguous ``[time, symbol]`` float32 arrays for the
    compiled factor graphs) and ``to_nautilus`` (bars and instruments written
    to a ``ParquetDataCatalog``) on top of ``BaseDataset``. A subclass
    implements ``_raw_data_to_xr``, ``_raw_data_to_xr_window``,
    ``_to_kunquant`` and ``_to_nautilus``; an exit it does not support may
    simply raise.

    Examples
    --------
    >>> ds = MyMarketDataset(config).read()
    >>> inputs, symbols, timestamps = ds.to_kunquant(("open", "close"))
    >>> inputs["close"].shape
    (2516, 503)
    """

    # Narrowed for readers and type checkers only
    config: DatasetConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = DatasetConfig

    def _write_catalog(self, data: list):
        """Write a list of Nautilus objects to the configured catalog."""
        catalog = ParquetDataCatalog(
            self.config.catalog_path, fs_protocol="file"
        )
        catalog.write_data(data)

    def to_nautilus(
        self, venue: str = "BINANCE", n_jobs: int = 16, write: bool = True
    ) -> tuple[list[list], list[Instrument]]:
        """Read the store and convert it to Nautilus bars and instruments.

        Parameters
        ----------
        venue : str
            Venue name used in the instrument identifiers.
        n_jobs : int
            Number of parallel workers for the per-symbol conversion.
        write : bool
            Whether to also write the result to the parquet catalog.

        Returns
        -------
        tuple[list[list], list[Instrument]]
            A tuple ``(bars, instruments)`` where ``bars`` holds one list of
            ``Bar`` objects per symbol.

        Examples
        --------
        Needs a subclass whose ``_to_nautilus`` builds the instruments:

        >>> bars, instruments = ds.to_nautilus(venue="NASDAQ", write=False)
        """
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
        """Read the store and convert it to KunQuant input arrays.

        Parameters
        ----------
        data_columns : tuple[str, ...]
            Column names, in KunQuant's vocabulary, to export.

        Returns
        -------
        tuple[dict, np.ndarray, np.ndarray]
            A tuple ``(inputs, symbols, timestamps)`` where ``inputs`` maps
            each column to a contiguous ``[time, symbol]`` float32 array.

        Examples
        --------
        >>> inputs, symbols, timestamps = ds.to_kunquant(("open", "close"))
        >>> inputs["close"].shape, inputs["close"].dtype
        ((4, 3), dtype('float32'))
        >>> symbols.tolist()
        ['AAA', 'BBB', 'CCC']
        """
        data = self.read().get_xarray_dataset()
        return self._to_kunquant(data, data_columns)

    @abstractmethod
    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Convert a panel to ``(inputs, symbols, timestamps)`` for KunQuant.

        This is where vendor column names are mapped onto KunQuant's
        (``open``/``high``/``low``/``close``/``volume``/``amount``).
        """

    @abstractmethod
    def _to_nautilus(
        self, data: xr.Dataset, venue: str, n_jobs: int
    ) -> tuple[list[list], list[Instrument]]:
        """Convert a panel to per-symbol Nautilus bars and their instruments."""

    @abstractmethod
    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset:
        """Return the dense panel for one time window, on ``symbols`` if given.

        Declared abstract again here so that every market dataset states
        explicitly how it joins the chunked conversion path.
        """
