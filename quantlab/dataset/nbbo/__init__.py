"""NBBO quote-bar panel built from raw WRDS TAQ quote files.

The NBBO (National Best Bid and Offer) is the best bid and the best ask
price for a US stock across all exchanges at each moment. TAQ (Trade and
Quote) is NYSE's tick-level database of every trade and quote, sold through
WRDS (Wharton Research Data Services). A *panel* is an ``xarray.Dataset``
indexed by ``timestamp`` and ``symbol``, the format every quantlab layer
exchanges.

``NbboPanelDataset`` reads the tick-level raw files that the WRDS TAQ
download writes (``.../wrds/data_type=nbbo/date=.../symbol=.../*.pqt``),
resamples each trading session onto a regular bar grid with
``quantlab.dataset.nbbo.resample``, and returns the panel the rest of the
pipeline stores and uses. Bars are *right-closed*: a bar labelled 09:31
covers quotes after 09:30 up to and including 09:31. The bar size is
``NbboDatasetConfig.bar_interval``, while the raw data's ``frequency`` stays
``"tick"``. Session opens and closes come from the NYSE (XNYS) exchange
calendar, which handles half days, daylight-saving changes and non-trading
days.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.base.config import DatasetConfig, NbboDatasetConfig
from quantlab.base.data import BaseDataset
from quantlab.dataset._support.cleaning import NBBO_PANEL_VARIABLES, clean_nbbo_panel
from quantlab.dataset.nbbo.resample import (
    FILTER_STATS_COUNTS,
    NbboFilterPolicy,
    NbboResampler,
)
from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import BAR_INTERVAL_SECONDS
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.timer import Timer

#: Appended to the store path to name the filter-stats sidecar, a JSON file
#: next to the store directory rather than inside it.
FILTER_STATS_SUFFIX = ".nbbo_filter_stats.json"


class NbboPanelDataset(StockDataset):
    """Dense NBBO bar panel resampled from WRDS TAQ ``complete_nbbo`` records.

    It inherits from ``StockDataset`` the vendor-root check, the tick-data
    ``data_type=`` scan root, the hive schema and ``has_raw_data``. It
    overrides how the axes and each window's panel are built, and
    ``_clean``: the panel has no OHLCV (open, high, low, close, volume)
    columns, so ``clean_nbbo_panel`` validates it instead.

    The session window (``session_start``/``session_end``, US Eastern clock
    time) defaults to regular hours, 09:30 to 16:00, and may be set anywhere
    inside 04:00 to 20:00; an edge outside that range is refused when the
    dataset is created. On a half day only regular-hours edges are clipped
    to the early close, so an extended-hours window still runs to its clock
    end, and its bars after the close hold after-close quotes. A ``date=``
    directory that is not a trading session makes the conversion fail with
    ``ValueError``.

    Every window starts from the last valid quote at or before its start.
    The raw files are read by session date and hold the whole day, so the
    quote in force when the window opens is always available. Bar labels can
    fall outside the session date's UTC calendar day: an extended close
    lands on the next one. The config's filter fields (``drop_crossed``,
    ``drop_locked``, ``drop_nonpositive_price``, ``keep_qu_cond``) are passed
    to the resampler as an ``NbboFilterPolicy``, and the number of quotes
    dropped per session is written to a JSON sidecar next to the store (see
    ``filter_stats_path``).

    Convert bars shorter than a minute with ``granularity="day"``: one-second
    bars over a large universe are millions of rows per session, and one
    window is held in memory at a time.

    Parameters
    ----------
    dataset_config : NbboDatasetConfig
        Must have ``frequency="tick"`` and a ``bar_interval`` from
        ``BAR_INTERVAL_SECONDS``; also sets the session window, the quote
        filters and optionally ``symbols``.

    Attributes
    ----------
    last_filter_stats : dict or None
        The filter-stats sidecar content after the last window resampled.

    Examples
    --------
    Needs a raw NBBO tier on disk under the configured vendor root:

    >>> config = NbboDatasetConfig(
    ...     raw_data_dir_path="downloads/us_equity/tick/wrds_taq/wrds",
    ...     catalog_path="data/us_equity/catalog",
    ...     zarr_file_path="data/us_equity/tick/wrds_nbbo_1m.zarr",
    ...     start_date="2024-01-24",
    ...     end_date="2024-01-25",
    ...     bar_interval="1m",
    ... )
    >>> NbboPanelDataset(config).from_raw_data_chunked(granularity="day")
    >>> panel = NbboPanelDataset(config).read().get_xarray_dataset()
    >>> panel["bid"].dims
    ('timestamp', 'symbol')
    """

    #: The config class used to rebuild the dataset from a saved ``config.json``.
    config_cls = NbboDatasetConfig

    #: The ``data_type=`` hive-key value of the raw files this panel reads.
    DATA_TYPE = "nbbo"

    #: The merged filter-stats sidecar content after the last window this
    #: instance resampled; ``None`` until then. Windows skipped because the
    #: chunk ledger marks them done leave it, and the sidecar, unchanged.
    last_filter_stats: dict | None = None

    @BaseDataset.config.setter
    def config(self, config: DatasetConfig):
        """Assign the config after checking the NBBO-specific fields.

        After the base class processes the config, it must be an
        ``NbboDatasetConfig`` with ``frequency="tick"`` and a ``bar_interval``
        from ``BAR_INTERVAL_SECONDS``. The session calendar is created here
        so that a bad window fails when the dataset is created rather than
        on the first conversion.

        Parameters
        ----------
        config : NbboDatasetConfig
            The new configuration.

        Raises
        ------
        TypeError
            If ``config`` is not an ``NbboDatasetConfig``.
        ValueError
            If ``frequency`` is not ``"tick"``, ``bar_interval``
            is unknown, or the session window is malformed or outside
            04:00 to 20:00 ET.

        Examples
        --------
        >>> ds.config = NbboDatasetConfig(
        ...     raw_data_dir_path="downloads/us_equity/tick/wrds_taq/wrds",
        ...     catalog_path="data/us_equity/catalog",
        ...     zarr_file_path="data/us_equity/tick/wrds_nbbo_1h.zarr",
        ...     bar_interval="1h",
        ... )
        Traceback (most recent call last):
        ...
        ValueError: NbboPanelDataset: bar_interval '1h' is not one of ...
        """
        BaseDataset.config.fset(self, config)
        if not isinstance(config, NbboDatasetConfig):
            raise TypeError(
                f"{self.class_name} needs an NbboDatasetConfig, got "
                f"{type(config).__name__}."
            )
        if config.frequency != "tick":
            raise ValueError(
                f"{self.class_name}: frequency must be 'tick' (the raw data "
                f"holds one row per NBBO record); the panel's bar size is "
                f"bar_interval. Got frequency {config.frequency!r}."
            )
        if config.bar_interval not in BAR_INTERVAL_SECONDS:
            raise ValueError(
                f"{self.class_name}: bar_interval {config.bar_interval!r} is "
                f"not one of {list(BAR_INTERVAL_SECONDS)}."
            )
        # The exchange calendar itself loads on the first session_bounds
        # call; only the window is checked here.
        self._calendar = XnysSessionCalendar(config.session_start, config.session_end)

    @property
    def _tick_data_type(self) -> str:
        """Return the ``data_type=`` value used to build the scan root, ``"nbbo"``."""
        return self.DATA_TYPE

    @property
    def filter_stats_path(self) -> str:
        """Return the path of the filter-stats sidecar next to the store.

        Examples
        --------
        >>> ds.filter_stats_path
        'data/us_equity/tick/wrds_nbbo_1m.zarr.nbbo_filter_stats.json'
        """
        return f"{self.config.zarr_file_path}{FILTER_STATS_SUFFIX}"

    @property
    def _resampler(self) -> NbboResampler:
        """Return a new resampler for the configured bar size and quote filters."""
        return NbboResampler(
            self.config.bar_interval, NbboFilterPolicy.from_config(self.config)
        )

    # -- sessions and axes ------------------------------------------------------

    def _session_dates(self) -> list[date]:
        """Return every ``date=`` directory under the scan root, ascending."""
        root = self._scan_root()
        if not root.exists():
            return []
        dates = []
        for path in root.glob("date=*"):
            if path.is_dir():
                dates.append(date.fromisoformat(path.name.split("=", 1)[1]))
        return sorted(dates)

    def _session_bounds(self, dates) -> pl.DataFrame:
        """Return ``(date, open, close)`` for each session date, in naive UTC.

        This calls ``XnysSessionCalendar.session_bounds``: a date that is not
        a trading session raises ``ValueError``, and a session whose window is
        empty after clipping is left out.

        Parameters
        ----------
        dates : iterable of date
            Session dates.

        Returns
        -------
        pl.DataFrame
            Columns ``date``, ``open`` and ``close``.
        """
        return self._calendar.session_bounds(dates)

    def _dates_in_config_range(self) -> list[date]:
        """Return the raw session dates inside the configured date range."""
        start = date.fromisoformat(self.config.start_date)
        end = date.fromisoformat(self.config.end_date)
        return [day for day in self._session_dates() if start <= day <= end]

    def _raw_symbols(self, dates) -> list[str]:
        """Return the sorted ``symbol=`` directory names found under the given dates.

        Parameters
        ----------
        dates : iterable of date
            Session dates whose ``date=`` directories to list.

        Returns
        -------
        list of str
            The distinct symbols, sorted.
        """
        root = self._scan_root()
        symbols = set()
        for day in dates:
            for path in (root / f"date={day.isoformat()}").glob("symbol=*"):
                if path.is_dir():
                    symbols.add(path.name.split("=", 1)[1])
        return sorted(symbols)

    def _raw_axes_in_range(self) -> tuple[list[str], pd.DatetimeIndex]:
        """Return the symbols and bar labels for the configured range, without resampling.

        Symbols are ``config.symbols`` when set, otherwise the ``symbol=``
        directory names. The timestamps are the bar labels of the session
        grid for the session dates present in the raw data, never the
        timestamps of the quotes themselves.

        Returns
        -------
        symbols : list of str
            The symbol axis, sorted.
        timestamps : pd.DatetimeIndex
            The bar labels.

        Raises
        ------
        ValueError
            If no symbols are found; a store with an empty symbol axis would
            have no labels from which to set the axis dtype.
        """
        self._assert_vendor_root()
        dates = self._dates_in_config_range()
        if self.config.symbols is not None:
            symbols = sorted(str(symbol) for symbol in self.config.symbols)
        else:
            symbols = self._raw_symbols(dates)
        if not symbols:
            raise ValueError(
                f"{self.class_name}: no symbols to convert in "
                f"[{self.config.start_date}, {self.config.end_date}] "
                f"(config.symbols={self.config.symbols!r}); refusing to "
                f"write an empty panel."
            )
        labels = self._resampler.labels(self._session_bounds(dates))
        return symbols, pd.DatetimeIndex(labels["timestamp"].to_list())

    # -- filter-stats sidecar ------------------------------------------------------

    def _merge_filter_stats(
        self, dates, stats: pl.DataFrame | None, policy: NbboFilterPolicy
    ) -> dict:
        """Merge one window's per-(date, symbol) drop counts into the sidecar.

        Every session date the window resampled is replaced completely (a
        window resamples whole sessions for every symbol on the axis, so its
        counts for a date are complete), and other dates are kept.
        ``totals`` is recomputed over all merged sessions, so resampling a
        date again never counts it twice.

        Parameters
        ----------
        dates : list of date
            The session dates the window resampled.
        stats : pl.DataFrame or None
            Per-(date, symbol) drop counts from the resampler, or ``None``
            if the window had no records.
        policy : NbboFilterPolicy
            The quote filters used, recorded in the sidecar.

        Returns
        -------
        dict
            The sidecar content as written, also stored on
            ``last_filter_stats``.
        """
        path = Path(self.filter_stats_path)
        existing = json.loads(path.read_text()) if path.exists() else {}
        by_session: dict = dict(existing.get("by_session", {}))

        fresh: dict[str, dict] = {day.isoformat(): {} for day in dates}
        if stats is not None:
            for row in stats.iter_rows(named=True):
                fresh.setdefault(row["date"].isoformat(), {})[str(row["symbol"])] = {
                    name: int(row[name]) for name in FILTER_STATS_COUNTS
                }
        by_session.update(fresh)

        totals = {name: 0 for name in FILTER_STATS_COUNTS}
        for per_symbol in by_session.values():
            for counts in per_symbol.values():
                for name in FILTER_STATS_COUNTS:
                    totals[name] += int(counts.get(name, 0))

        payload = {
            "config": {
                "bar_interval": self.config.bar_interval,
                "session_start": self.config.session_start,
                "session_end": self.config.session_end,
                "drop_crossed": policy.drop_crossed,
                "drop_locked": policy.drop_locked,
                "drop_nonpositive_price": policy.drop_nonpositive_price,
                "keep_qu_cond": (
                    list(policy.keep_qu_cond)
                    if policy.keep_qu_cond is not None
                    else None
                ),
            },
            "by_session": by_session,
            "totals": totals,
        }
        write_json_atomically(path, payload, indent=2, sort_keys=True)
        # Round-trip through JSON so the value in memory equals the file.
        self.last_filter_stats = json.loads(json.dumps(payload, sort_keys=True))
        return self.last_filter_stats

    # -- densify ------------------------------------------------------------------

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[str] | None = None
    ) -> xr.Dataset:
        """Resample the sessions whose bar labels fall in the window onto a dense grid.

        The raw files are filtered on the ``date`` hive key, never on a
        timestamp window, so the last quote before the open (which seeds the
        first bar) is kept. A ``(label, symbol)`` cell with no bar is NaN in
        every variable.

        Parameters
        ----------
        start_date : date-like
            First bar label to include, inclusive.
        end_date : date-like
            Last bar label to include, inclusive.
        symbols : list of str, optional
            The symbol axis. When ``None``, ``config.symbols`` or the raw
            ``symbol=`` directories are used.

        Returns
        -------
        xr.Dataset
            A dataset with the ``NBBO_PANEL_VARIABLES`` as float64 variables
            on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            If there is no raw data under the scan root or the symbol axis
            is empty.
        """
        self._assert_vendor_root()
        if not self.has_raw_data():
            raise ValueError(
                f"{self.class_name}: no raw NBBO data under "
                f"{str(self._scan_root())!r}. Acquire it first."
            )
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        if symbols is None:
            symbols = (
                sorted(str(symbol) for symbol in self.config.symbols)
                if self.config.symbols is not None
                else self._raw_symbols(self._session_dates())
            )
        symbols = [str(symbol) for symbol in symbols]
        if not symbols:
            raise ValueError(
                f"{self.class_name}: the window {start}..{end} has no symbols; "
                f"refusing to write an empty panel."
            )

        resampler = self._resampler
        # Only the configured range, the same sessions _raw_axes_in_range
        # used, so a raw date outside the range is never looked up.
        sessions = self._session_bounds(self._dates_in_config_range())
        labels = resampler.labels(sessions).filter(
            pl.col("timestamp").is_between(start, end, closed="both")
        )
        dates = labels["date"].unique().sort().to_list()
        sessions = sessions.filter(pl.col("date").is_in(dates))

        bars = None
        stats = None
        if dates:
            scan = pl.scan_parquet(
                str(self._scan_root() / "**" / f"*{self.RAW_SHARD_SUFFIX}"),
                hive_partitioning=True,
                hive_schema=self._scanned_hive_schema(),
            ).filter(
                pl.col("date").is_in(dates),
                pl.col("symbol").is_in(symbols),
            )
            scan = self._assert_single_vendor_and_drop(scan)
            records = scan.collect()
            if records.height:
                bars, stats = resampler.resample_with_stats(records, sessions)
            self._merge_filter_stats(dates, stats, resampler.policy)

        label_values = labels["timestamp"].sort().to_list()
        grid = pl.DataFrame(
            {"timestamp": label_values}, schema={"timestamp": pl.Datetime("ns")}
        ).join(
            pl.DataFrame(
                {"symbol": symbols, "_position": list(range(len(symbols)))},
                schema={"symbol": pl.String, "_position": pl.Int64},
            ),
            how="cross",
        )
        if bars is not None:
            grid = grid.join(
                bars.drop("date"), on=["timestamp", "symbol"], how="left"
            )
        else:
            grid = grid.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(name)
                for name in NBBO_PANEL_VARIABLES
            )
        grid = grid.sort(["timestamp", "_position"])

        shape = (len(label_values), len(symbols))
        variables = {
            name: (
                ("timestamp", "symbol"),
                grid[name]
                .cast(pl.Float64)
                .fill_null(np.nan)
                .to_numpy()
                .astype("float64")
                .reshape(shape),
            )
            for name in NBBO_PANEL_VARIABLES
        }
        return xr.Dataset(
            variables,
            coords={
                "timestamp": pd.DatetimeIndex(label_values).values,
                "symbol": np.array([str(symbol) for symbol in symbols], dtype=object),
            },
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Resample the whole configured range as one window.

        Returns
        -------
        xr.Dataset
            The panel for the configured range.

        Raises
        ------
        ValueError
            If no session inside the range has raw data.
        """
        with Timer(f" {self.__class__.__name__}: from pqt"):
            symbols, timestamps = self._raw_axes_in_range()
            if len(timestamps) == 0:
                raise ValueError(
                    f"{self.class_name}: no session with raw NBBO data inside "
                    f"[{self.config.start_date}, {self.config.end_date}]."
                )
            return self._raw_data_to_xr_window(
                timestamps[0], timestamps[-1], symbols=symbols
            )

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Validate the NBBO panel with ``clean_nbbo_panel`` instead of the OHLCV cleaner."""
        return clean_nbbo_panel(data)

    def _widen_fill_values(self) -> dict:
        """Return no special fill values for symbols added to an existing store.

        When the symbol axis is widened, a new symbol's earlier history must
        be filled with something. Every panel variable is float64, and NaN
        already means "no quote existed", which is correct there. This holds
        for ``n_updates`` too: NaN differs from 0, which means "a quote was
        in force but did not change".

        Returns
        -------
        dict
            An empty dict, so every variable is filled with NaN.
        """
        return {}
