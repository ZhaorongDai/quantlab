"""Dataset base classes: turning raw vendor files into a stored panel.

This module is the first layer of the pipeline (data, then factors, models
and backtests). Every layer exchanges data as a *panel*: an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, with one data
variable per field (``open``, ``close``, ``volume`` and so on). Panels are
stored on disk as Zarr directories (a chunked array format), called *stores*
below. The *raw tier* is the vendor's files as downloaded, before
conversion.

``BaseDataset`` holds the lifecycle shared by every dataset: a config with
normalised ISO dates, a Zarr storage backend, ``from_raw_data``, ``save`` and
``read`` for converting the whole date range at once, and
``from_raw_data_chunked`` and ``update`` for converting one time window at a
time so that an interrupted run can resume. ``MarketDataset`` adds the
``to_kunquant`` export for market data (arrays for the KunQuant factor
engine).

A concrete dataset lives under ``quantlab/dataset/`` and implements
``_raw_data_to_xr``. The factor layer reads the panel through
``get_xarray_dataset`` or ``get_lazyframe``.
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

from quantlab.base.config import BaseDatasetConfig, DatasetConfig
from quantlab.base.progress import CancelToken, ProgressEvent, ProgressReporter
from quantlab.backend import XrBackend
from quantlab.dataset._support.cleaning import clean_market_data
from quantlab.enums.constant import Date
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer


@dataclass(frozen=True)
class ConversionResult:
    """Summary of one chunked conversion from raw files to a Zarr store.

    ``BaseDataset.from_raw_data_chunked`` stores an instance on the
    dataset's ``last_chunk_result`` attribute when it finishes, so a caller
    can report what the run did. The attributes are documented inline below.

    Examples
    --------
    >>> ds = DemoDataset(config).from_raw_data_chunked(granularity="month")
    >>> result = ds.last_chunk_result
    >>> result.windows_written, result.rows_written, result.resumed
    (1, 6, False)
    """

    #: The Zarr store that was written (``config.zarr_file_path``).
    zarr_path: str
    #: Path of the ledger file (a JSON sidecar next to the store) that records
    #: completed windows.
    ledger_path: str
    #: Window granularity the run planned on (``"year"``, ``"quarter"``, ...).
    granularity: str
    #: Number of symbols on the pinned symbol axis, the fixed list of symbols
    #: for the whole date range that every window is written on.
    pinned_symbols: int
    #: Number of windows the planner produced over the observed timestamps.
    windows_planned: int
    #: Number of windows this run built and appended.
    windows_written: int
    #: Number of windows skipped because the ledger already recorded them.
    windows_skipped: int
    #: Rows appended along the append dimension, summed over written windows.
    rows_written: int
    #: Largest in-memory size (``window.nbytes``) of any window this run built;
    #: ``None`` when no window was written.
    peak_window_bytes: int | None
    #: The caller's own estimate of the peak, passed through unchanged;
    #: usually ``None``.
    predicted_peak_bytes: int | None = None
    #: Whether at least one window was skipped, meaning this run continued an
    #: earlier one.
    resumed: bool = False
    #: Whether a ``CancelToken`` stopped the loop between windows. Without this
    #: flag a cancelled run would look like a run that had less work to do.
    cancelled: bool = False
    #: Whether an ``on_new_listing="rebuild"`` run was cancelled and rolled
    #: back, restoring the previous store and ledger. Only ``True`` together
    #: with ``cancelled``. When set, ``windows_written`` and ``rows_written``
    #: describe work that no longer exists on disk.
    rebuild_rolled_back: bool = False


class BaseDataset(ABC):
    """Abstract base class shared by every dataset in quantlab.

    A subclass implements ``_raw_data_to_xr``, which reads the raw files and
    returns a *dense* panel (a full ``timestamp`` by ``symbol`` grid, with
    NaN where a symbol has no value). Everything else is inherited: config
    normalisation, Zarr storage through ``XrBackend``, the cleaning hook, and
    chunked ingestion with its resume ledger and its handling of *new
    listings* (symbols that appear in the raw tier but not yet in the store).
    Datasets that are not price bars, such as a boolean index-membership
    panel, derive from this class directly; market data derives from
    ``MarketDataset``.

    Parameters
    ----------
    config : BaseDatasetConfig
        The dataset config. It is modified in place on assignment; see the
        ``config`` property.

    Attributes
    ----------
    data_backend : XrBackend
        Holds the panel in memory and reads and writes the Zarr store.
    last_chunk_result : ConversionResult or None
        Summary of the last ``from_raw_data_chunked`` or ``update`` run, or
        ``None`` if neither has completed on this object.

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

    #: Accepted values of ``on_new_listing``: stop (``"refuse"``), rewrite the
    #: whole store (``"rebuild"``), or add NaN columns (``"widen"``).
    NEW_LISTING_STRATEGIES: tuple[str, ...] = ("refuse", "rebuild", "widen")

    #: ``{field name: reason}`` for factor-config fields that a factor built on
    #: this dataset must not set. The factor base class reads this and raises,
    #: quoting the reason, so it never needs to know concrete dataset classes.
    #: Empty by default. A subclass may override it, for example to refuse
    #: ``symbols`` on a panel whose symbol labels are not strings. Each reason
    #: should say why and what to use instead.
    REJECTED_FACTOR_CONFIG_FIELDS: dict[str, str] = {}

    #: Marker value that ``update()`` passes to ``from_raw_data_chunked()`` to
    #: have the new-listing strategy chosen from raw-tier evidence. It is an
    #: ``object()`` rather than a string so no command-line flag, config file
    #: or JSON value can produce it, and it is not in
    #: ``NEW_LISTING_STRATEGIES``, which is what the command line offers.
    _AUTOMATIC = object()

    #: Maximum number of symbols named in a new-listing log message.
    NEW_LISTING_REPORT_LIMIT: int = 20

    def __init__(self, config: BaseDatasetConfig):
        """Initialize the dataset; see the class docstring for parameters.

        The backend is created before the config is assigned because a
        config setter may use the backend; swapping the two steps raises
        ``AttributeError``.
        """
        # `from_raw_data_chunked()` stores its summary here; a dataset that was
        # only read reports None.
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
        'DemoDataset'
        """
        return self.__class__.__name__

    @property
    def _progress_vendor(self) -> str:
        """Return the value for ``ProgressEvent.vendor`` during a conversion.

        A conversion has no vendor client, so the config's ``vendor`` is used
        when it has one and the class name otherwise. The field must not be
        empty.
        """
        return getattr(self.config, "vendor", None) or self.class_name

    def _emit_progress(
        self, reporter: ProgressReporter | None, event: ProgressEvent
    ) -> None:
        """Deliver one progress event to ``reporter`` without ever raising.

        A ``None`` reporter does nothing. An exception raised by the reporter
        is logged as a warning and otherwise ignored, so a broken display
        cannot end a multi-hour conversion. Conversion events carry only
        window dates, counts and the class name, so there are no credentials
        to remove from them.
        """
        if reporter is None:
            return
        try:
            reporter.emit(event)
        except Exception as exc:  # noqa: BLE001 -- a reporter must never stop the run
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

        The most common spacing (the mode) is used rather than the minimum so
        that gaps such as weekends and holidays do not distort the answer.

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

        This is how a saved config names the class to rebuild.

        Examples
        --------
        >>> ds.import_path  # for a class defined in a script
        '__main__.DemoDataset'
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
        '__main__.DemoDataset'
        """
        return self._config

    @config.setter
    def config(self, config: BaseDatasetConfig):
        """Assign the config, filling in defaults and checking its dates.

        The config is modified in place. ``name`` is set to the class's
        import path. A missing ``start_date`` or ``end_date`` falls back to
        ``Date.START_DATE`` or ``Date.END_DATE``. Both dates must be ISO
        ``YYYY-MM-DD`` strings; a ``datetime.date`` is accepted and converted.
        Later code compares dates as plain strings, so a value such as
        ``"2007-2-1"`` would silently compare wrong; it is refused here
        instead.

        Parameters
        ----------
        config : BaseDatasetConfig
            The config to assign.

        Raises
        ------
        ValueError
            If a date is not an ISO ``YYYY-MM-DD`` date.

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

        ``field_name`` is used only in the error message.

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

        Nothing is cleaned or converted here; that is ``from_raw_data``'s job.

        Parameters
        ----------
        **kwargs
            Passed to ``XrBackend.read``.

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

        The write replaces the whole store directory.

        Parameters
        ----------
        **kwargs
            Passed to ``XrBackend.write``.

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

        Long format means one row per ``(timestamp, symbol)`` pair, with one
        column per variable.

        Examples
        --------
        >>> ds.get_lazyframe().collect().shape  # 4 days x 3 symbols, 8 columns
        (12, 8)
        """
        return self.data_backend.get_lazyframe()

    def head(self, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows of the store as a ``LazyFrame``.

        The store is opened from disk; the panel in memory and the config's
        date range are not used or changed, so this is a safe way to look at
        column names.

        Parameters
        ----------
        n : int
            Maximum number of rows.

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
        """Convert the raw files for the whole configured range, in memory.

        Runs ``_raw_data_to_xr``, then ``_clean``, and gives the result to
        the backend. Nothing is written to disk until ``save`` is called.
        Every call converts again; nothing is cached.

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

        The default converts the whole range with ``_raw_data_to_xr`` and
        reads both axes from the result. ``sort_symbol_axis`` sets the symbol
        order, and labels keep the raw panel's own type so that each window's
        ``reindex`` matches them. A subclass whose raw files can be read one
        column at a time should override this to avoid building the full
        panel.
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

        The default converts the whole range and slices it, so it limits the
        size of each write but not the memory used to build the panel. When
        ``symbols`` is given, the result is reindexed onto exactly those
        symbols, with missing ones as all-NaN columns.
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

        This is ``from_raw_data_chunked`` with ``on_new_listing`` chosen from
        the data. When the raw tier has symbols the store lacks, the raw tier
        is checked for rows of those symbols inside the store's existing date
        range. If there are none, the symbols are genuinely new listings and
        the store is widened with NaN history. If there are some, the store is
        rebuilt so that their real history is included. If a symbol in the
        store has disappeared from the raw tier, the run refuses. The choice
        is logged before it runs.

        Parameters
        ----------
        granularity : str, default "year"
            Window size, passed to ``TimeChunkPlanner``.
        ledger_path : str or None, default None
            Ledger file path; ``None`` uses ``<store>.chunks.json``.
        append_dim : str, default "timestamp"
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
        """Convert the raw files one time window at a time, appending to Zarr.

        The *pinned symbol axis*, the list of symbols for the whole date
        range, is worked out once before any window, and every window is
        built on exactly that axis so that all windows line up column by
        column. Completed windows are recorded in a ledger file, so a
        crashed or cancelled run resumes by skipping them. Cleaning runs per
        window, so a price jump that spans two windows is not flagged.

        When the store already exists with a different symbol axis,
        ``on_new_listing`` decides what happens. ``"refuse"`` stops.
        ``"widen"`` adds the new symbols with NaN over the store's existing
        history. ``"rebuild"`` moves the store aside and rebuilds every window
        from the raw files; if the rebuild fails or is cancelled, the
        original store is put back.

        Parameters
        ----------
        granularity : str, default "year"
            Window size, passed to ``TimeChunkPlanner`` (``"year"``,
            ``"quarter"``, ``"month"``, ``"day"`` or ``"hour"``).
        ledger_path : str or None, default None
            Ledger file path; ``None`` uses ``<store>.chunks.json``.
        append_dim : str, default "timestamp"
            Dimension windows are appended along.
        on_new_listing : str, default "refuse"
            One of ``NEW_LISTING_STRATEGIES``. ``update()`` passes a private
            marker here instead, to have the strategy chosen automatically.
        reporter : ProgressReporter or None, default None
            Receives an event for each window.
        cancel : CancelToken or None, default None
            Checked before each window; when set, the run stops there.

        Returns
        -------
        Self
            ``self``, with ``last_chunk_result`` describing the run.

        Raises
        ------
        ValueError
            If ``on_new_listing`` is unknown, if a window comes back on a
            symbol axis other than the pinned one, or if the ledger and the
            store disagree (see ``ChunkLedger.assert_consistent``).

        Examples
        --------
        >>> ds = DemoDataset(config).from_raw_data_chunked(granularity="month")
        >>> ds.last_chunk_result.windows_written
        1
        """
        from quantlab.base.chunking import ChunkLedger, TimeChunkPlanner

        # The private marker is accepted by identity; the error lists only the
        # public strategies.
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

        # Handle a changed symbol axis before the ledger check, because widen
        # and rebuild both change what that check looks at. The full timestamp
        # count is passed so a widen sizes the store's Zarr chunks for the
        # whole range, not just for what is stored so far.
        ledger, rebuild_asides = self._reconcile_new_listings(
            symbols,
            ledger,
            append_dim,
            on_new_listing,
            append_dim_size=len(timestamps),
        )
        # Set before the `try` so the result below can always read it.
        rebuild_rolled_back: bool = False

        try:
            # The ledger and the store record the same history; check they agree
            # before the first append, which cannot be undone.
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
                        f"to {self.config.zarr_file_path}"
                    ),
                    detail={
                        "pinned_symbols": len(symbols),
                        "granularity": granularity,
                        "zarr_path": self.config.zarr_file_path,
                    },
                ),
            )
            for start, end in windows:
                # Check for cancellation before building the next window.
                if cancel is not None and cancel.is_cancelled():
                    cancelled = True
                    resumability = (
                        "This run is a rebuild, so the windows written so far "
                        "will be discarded and the pre-rebuild store restored; "
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
                # Compare in the axis's own type, not as text: integer labels
                # would never equal their string forms.
                actual = list(window["symbol"].values.tolist())
                if actual != list(symbols):
                    raise ValueError(
                        f"{self.class_name}: window {start.date()}..{end.date()} "
                        f"came back on a symbol axis of {len(actual)} label(s), "
                        f"but the pinned whole-range axis has {len(symbols)}. "
                        f"Every window must be built on the pinned axis; "
                        f"appending this one would silently misalign every "
                        f"column in the store."
                    )

                window = self._clean(window)
                window = self._pin_append_dtypes(window)
                # Measure after cleaning and dtype promotion: that is what the
                # append actually holds in memory.
                window_bytes = int(window.nbytes)
                if (
                    peak_window_bytes is None
                    or window_bytes > peak_window_bytes
                ):
                    peak_window_bytes = window_bytes
                if start != first_timestamp:
                    boundaries += 1

                self.data_backend.to_internal(window)
                # widen_and_append adds any missing symbols or variables to the
                # store, then appends; if nothing is missing it is a plain
                # append. append_dim_size is passed every time because any
                # iteration may be the one that creates the store (a resume
                # skips windows, a rebuild moves the store aside); the backend
                # only uses it when creating the store.
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

            # Emitted whether the loop finished or was cancelled, so a reporter
            # can always release what it opened.
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
                    f"{boundaries} window-boundary timestamp(s) `flag_anomalies` "
                    f"had no previous value to compare with, and a one-step jump "
                    f"across that boundary is not flagged. This is an expected "
                    f"side effect of chunking; a finer --chunk granularity "
                    f"creates more such boundaries, not fewer."
                )
        except BaseException:
            # The originals were renamed, not deleted, so a failed rebuild is
            # undone by renaming them back.
            if rebuild_asides is not None:
                self._restore_rebuild_asides(rebuild_asides)
            raise
        else:
            if rebuild_asides is not None:
                if cancelled:
                    # A cancelled rebuild has rebuilt only some windows, so the
                    # set-aside copies are still the only complete data. Roll
                    # back rather than leave a truncated store where readers
                    # look. The flag records whether the restore succeeded.
                    rebuild_rolled_back = self._restore_rebuild_asides(
                        rebuild_asides, reason="cancelled"
                    )
                else:
                    self._discard_rebuild_asides(rebuild_asides)

        # Only reached on success; the `except` branch re-raises and must not
        # leave a result describing a partial run.
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

    #: Suffix added to the store and ledger paths while a rebuild is running
    #: and the originals are set aside. It reuses ``XrBackend.SUPERSEDED_SUFFIX``
    #: so the backend, which refuses to widen while a set-aside copy exists,
    #: always recognises one left behind by a rebuild that was killed.
    SUPERSEDED_SUFFIX = XrBackend.SUPERSEDED_SUFFIX

    @staticmethod
    def _stored_symbol_axis(
        store_path: str, dim: str = "symbol"
    ) -> Optional[list]:
        """Return the store's ``dim`` labels, or ``None`` if there is no store.

        Only the coordinate is read, never the data variables. Labels keep
        the store's own type so that comparing them with the pinned axis does
        not report a false mismatch.
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
        """Return the store's first and last ``append_dim`` labels as a pair.

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
        """Count the raw rows each symbol in ``added`` has between ``start`` and ``end``.

        ``update`` uses these counts to choose between widening and
        rebuilding. The default builds the whole window and counts, which is
        correct but uses memory for every symbol; a subclass whose raw files
        can be filtered by symbol while reading should override it.

        Symbols are compared as text because the raw tier stores symbols as
        strings even when the pinned axis uses integers. Without the
        conversion no added symbol would ever match, and the resolver would
        silently choose ``widen``.

        Parameters
        ----------
        added : list
            Symbols present in the raw tier but not in the store.
        start, end : timestamp-like
            The date range to check, both ends included.

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
                f"building the whole window and counting. The answer is "
                f"correct but memory is not bounded; override this method for "
                f"a source that can filter by symbol while reading."
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

        Non-float variables need an explicit fill value because NaN is not
        available for them. The default covers the boolean ``anomaly_flag``
        that cleaning adds.
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

        If any stored symbol was removed, the answer is ``"refuse"``, since
        neither widening nor rebuilding keeps its history safely. Otherwise
        the raw tier is checked for rows of the added symbols inside the
        store's existing date range. No rows means they are new listings, so
        ``"widen"``. Any rows means widening would replace real history with
        NaN, so ``"rebuild"``.

        Returns
        -------
        str
            ``"refuse"``, ``"widen"`` or ``"rebuild"``.
        """
        if removed:
            logger.warning(
                f"{self.class_name}: {len(removed)} symbol(s) present in the "
                f"store at {store_path} are absent from the pinned "
                f"whole-range axis ({sorted(removed)[: self.NEW_LISTING_REPORT_LIMIT]}"
                f"{', truncated' if len(removed) > self.NEW_LISTING_REPORT_LIMIT else ''}), "
                f"so this resolves to 'refuse'. Neither other strategy is safe "
                f"for a dropped symbol: 'widen' cannot remove one "
                f"(widen_symbol_axis refuses a target axis that is not a "
                f"superset of the stored one), and 'rebuild' would silently "
                f"discard that symbol's stored history. Choosing between losing "
                f"history and stopping is your call: re-run "
                f"from_raw_data_chunked() with an explicit on_new_listing once "
                f"you know which you want."
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
                f"{len(added)} added symbol(s) over the store's own date range "
                f"({start}..{end}) and reported no rows there for any of them. "
                f"They are genuine new listings, NaN is the correct value over "
                f"the store's history, and a rebuild would gain nothing; "
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
            f"already have raw rows inside the store's own date range "
            f"({start}..{end}), so this resolves to 'rebuild' rather than "
            f"'widen': a widen does not re-read the raw files, so it would "
            f"replace that history with NaN and nothing would report it. "
            f"Qualifying symbol(s) by raw row count: {listing}{truncation}. "
            f"A rebuild rewrites the whole store, not just these symbols, so "
            f"this run rebuilds every window."
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
        """Apply ``on_new_listing`` when the store's symbols differ from ``symbols``.

        Returns
        -------
        tuple
            ``(ledger, rebuild_asides)``. ``ledger`` is the ledger to use from
            now on. ``rebuild_asides`` is ``None`` unless a rebuild started,
            in which case it is a dict naming the original and set-aside
            paths of the store and ledger, so they can later be restored or
            deleted.
        """
        store_path = self.config.zarr_file_path
        stored = self._stored_symbol_axis(store_path)
        if stored is None or stored == list(symbols):
            return ledger, None

        added = [symbol for symbol in symbols if symbol not in set(stored)]
        removed = [symbol for symbol in stored if symbol not in set(symbols)]

        # Resolved only after the early return above, so the raw tier is not
        # queried when the axes already agree.
        if on_new_listing is self._AUTOMATIC:
            on_new_listing = self._resolve_new_listing_strategy(
                added, removed, store_path, append_dim
            )

        if on_new_listing == "refuse":
            # Leave everything unchanged; `assert_consistent` raises next.
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
                f"removed={removed}. The added symbol(s) will be NaN for all "
                f"existing history: a widen does not re-read the raw files, so "
                f"history the vendor already has is not recovered. Use "
                f"on_new_listing='rebuild' for that; it is the only reason to "
                f"prefer it, since a widen that would not fit in memory "
                f"rewrites the store block by block "
                f"(XrBackend.MAX_WIDEN_BYTES), so store size does not force a "
                f"rebuild."
            )
            self.data_backend.widen_symbol_axis(
                store_path,
                list(symbols),
                append_dim=append_dim,
                fill_values=self._widen_fill_values(),
                append_dim_size=append_dim_size,
            )
            # The ledger stores a fingerprint of the symbol list, so update it
            # now or the widened store cannot be resumed.
            ledger.rebase(symbols)
            return ledger, None

        # Remaining case: on_new_listing == "rebuild".
        from quantlab.base.chunking import ChunkLedger

        logger.warning(
            f"{self.class_name}: rebuilding {store_path}: every window will "
            f"be rebuilt from the raw files on the new {len(symbols)}-symbol "
            f"axis (was {len(stored)}); added={added}, removed={removed}. "
            f"This recovers the added symbol(s)' real history instead of "
            f"filling it with NaN, at the cost of a full rebuild."
        )
        asides = {
            "store": store_path,
            "store_aside": f"{store_path}{self.SUPERSEDED_SUFFIX}",
            "ledger": ledger.path,
            "ledger_aside": f"{ledger.path}{self.SUPERSEDED_SUFFIX}",
            "ledger_existed": Path(ledger.path).exists(),
        }
        # Rename rather than delete, so a failed rebuild can be undone.
        os.replace(asides["store"], asides["store_aside"])
        if asides["ledger_existed"]:
            os.replace(asides["ledger"], asides["ledger_aside"])
        # With a fresh ledger at the original path, the rebuild proceeds like
        # an ordinary first run.
        return ChunkLedger(asides["ledger"], append_dim=append_dim), asides

    def _restore_rebuild_asides(
        self, asides: dict, *, reason: str = "failed"
    ) -> bool:
        """Put the pre-rebuild store and ledger back, deleting the partial rebuild.

        Parameters
        ----------
        asides : dict
            The paths returned by ``_reconcile_new_listings``.
        reason : str, default "failed"
            Word used in the log message (``"failed"`` or ``"cancelled"``).

        Returns
        -------
        bool
            ``True`` if the originals were restored. ``False`` if the
            filesystem refused; then nothing is deleted, and the error log
            names both copies so the user can recover by hand. The error is
            logged rather than raised so it cannot hide the failure that
            stopped the rebuild.
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
                f"restored. Any window this run rebuilt has been discarded "
                f"with the partial store, so a later rebuild starts over."
            )
            return True
        except OSError as exc:
            logger.error(
                f"{self.class_name}: the rebuild of {asides['store']} was "
                f"{reason}, but the pre-rebuild copy could not be put back: "
                f"{type(exc).__name__}: {exc}. Nothing was deleted: the "
                f"complete pre-rebuild copy is still at "
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
        """Delete the set-aside store and ledger after a successful rebuild."""
        shutil.rmtree(asides["store_aside"], ignore_errors=True)
        Path(asides["ledger_aside"]).unlink(missing_ok=True)

    @staticmethod
    def _pin_append_dtypes(data: xr.Dataset) -> xr.Dataset:
        """Promote integer data variables to float64 before an append.

        The first window written fixes each variable's dtype in the store. An
        integer variable cannot hold NaN, so the missing cells of later
        windows would be silently cast to integers; storing float64 keeps
        them as NaN.
        """
        promoted = {
            name: variable.astype("float64")
            for name, variable in data.data_vars.items()
            if np.issubdtype(variable.dtype, np.integer)
        }
        return data.assign(**promoted) if promoted else data

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Validate and flag the converted panel before it is stored.

        The default runs ``clean_market_data``, which checks for the OHLCV
        columns (open, high, low, close, volume) and adds a boolean
        ``anomaly_flag`` variable marking suspicious jumps. Override it for
        panels whose columns are not lowercase OHLCV, or that are not market
        data at all.
        """
        return clean_market_data(data)

    @abstractmethod
    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the raw files as a dense ``(timestamp, symbol)`` panel.

        The result must cover the config's whole date range, and each
        ``(timestamp, symbol)`` pair must appear once; remove duplicates
        before converting a table with ``to_xarray``. The caller applies
        cleaning, so do not clean here.
        """


class MarketDataset(BaseDataset):
    """Dataset of market price bars, with a KunQuant export.

    A *bar* is one period's open, high, low, close and volume for a symbol.
    On top of ``BaseDataset`` this class adds ``to_kunquant``, which returns
    contiguous ``[time, symbol]`` float32 arrays for KunQuant (the compiled
    factor engine). A subclass implements ``_raw_data_to_xr``,
    ``_raw_data_to_xr_window`` and ``_to_kunquant``.

    Parameters
    ----------
    config : DatasetConfig
        The dataset config.

    Examples
    --------
    Using the ``DemoDataset`` described on ``BaseDataset``:

    >>> ds = DemoDataset(config).from_raw_data()
    >>> ds.save()
    >>> inputs, symbols, timestamps = ds.to_kunquant(("open", "close"))
    >>> inputs["close"].shape
    (4, 3)
    """

    # Narrower type annotation for readers and type checkers only.
    config: DatasetConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = DatasetConfig

    def to_kunquant(
        self, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Read the store and convert it to KunQuant input arrays.

        Parameters
        ----------
        data_columns : tuple[str, ...]
            Columns to export, named as KunQuant names them (``open``,
            ``high``, ``low``, ``close``, ``volume``, ``amount``).

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

        This is where vendor column names are mapped onto KunQuant's names
        (``open``, ``high``, ``low``, ``close``, ``volume``, ``amount``).
        """

    @abstractmethod
    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset:
        """Return the dense panel for one time window, on ``symbols`` if given.

        Declared abstract again here so that every market dataset must say
        explicitly how it builds one window for chunked conversion, rather
        than inheriting the slow default.
        """
