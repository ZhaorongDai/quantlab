"""The WRDS TAQ NBBO bar panel: raw `complete_nbbo` tick shards -> Zarr (03.9).

`NbboPanelDataset` reads the raw tier `WrdsTaqNbboAcquisition` writes
(`.../wrds/data_type=nbbo/date=/symbol=/`) and materialises a dense
`[timestamp, symbol]` panel of right-closed NBBO bars through
`dataset/nbbo/resample.py:NbboResampler`. The bar size is
`NbboDatasetConfig.bar_interval`; the raw tier's `frequency` stays `"tick"`.

Session edges come from `quantlab/dataset/_support/session_calendar.py:XnysSessionCalendar`
(D-23): the one place a session's open/close is decided, half days, DST and
non-sessions included. Nothing here localises a wall-clock time itself.

**Filter drop counts** land in a JSON sidecar beside the store,
`{zarr_file_path}{FILTER_STATS_SUFFIX}` (D-10): per `(session date, symbol)`
and as totals, merged across chunk windows and across runs. It is a sibling of
the store (never inside it: a `mode="w"` rewrite replaces the store directory)
and never under the raw root (a stray JSON there would sit in the parquet
scan's tree).

**Chunking advice.** Convert sub-minute bars with `granularity="day"`: a 1s
S&P 500 day is ~11.7M bar-rows (23,400 bars x 500 symbols), and one window is
materialised in memory at a time.
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

#: Appended to the store path to name the filter-stats sidecar, a SIBLING of
#: the store directory (like `ChunkLedger.SUFFIX`).
FILTER_STATS_SUFFIX = ".nbbo_filter_stats.json"


class NbboPanelDataset(StockDataset):
    """A dense NBBO bar panel resampled from WRDS TAQ `complete_nbbo` records.

    Reuses `StockDataset`'s vendor-root assertion, tick `data_type=` scan
    root, hive schema, single-vendor provenance check and `has_raw_data`.
    Overrides the axes, the window densifier and `_clean` (D-14: the OHLCV
    cleaner would raise on a panel that has no OHLCV).

    **Session edges come from the XNYS calendar (D-23).** The window
    (`session_start`/`session_end`, ET wall clock) defaults to regular hours
    09:30-16:00 (D-09) and may be set anywhere inside 04:00-20:00 ET (D-29);
    an edge outside that range is refused when the dataset is constructed.
    On a half day only regular-hours edges are clipped to the early close;
    extended-hours edges are unchanged, so a 04:00-20:00 panel on 2024-11-29
    still runs to 20:00 and its post-13:00 bars carry post-close state. A
    `date=` directory that is not an XNYS session fails the conversion with a
    `ValueError` naming it; a session whose clipped window is empty (e.g.
    13:30-16:00 on a half day) contributes no labels.

    **Every window is seeded** from the last valid NBBO at or before its
    start: the raw scan is by session DATE, and raw holds the whole
    04:00-20:00 day (D-04), so the record in force at any window start is in
    the scanned frame.

    **Labels are not confined to the session date's UTC day.** An extended
    close (20:00 ET) lands on the next UTC calendar day; nothing here maps a
    label back to a date by its UTC calendar day -- the `date` travels with
    each label from the resampler.

    **Filter knobs** (`drop_crossed`, `drop_locked`, `drop_nonpositive_price`,
    `keep_qu_cond`) reach the resampler as an `NbboFilterPolicy` built from
    the config (`_resampler`, D-10).
    """

    #: The config class `quantlab/utils/module.py` rebuilds this dataset with.
    config_cls = NbboDatasetConfig

    #: The raw tier's `data_type=` hive key this panel is built from.
    DATA_TYPE = "nbbo"

    #: The merged filter-stats sidecar content after the last window this
    #: instance resampled; `None` until one has been. Windows the chunk
    #: ledger skips leave it (and the sidecar) untouched.
    last_filter_stats: dict | None = None

    @BaseDataset.config.setter
    def config(self, config: DatasetConfig):
        BaseDataset.config.fset(self, config)
        if not isinstance(config, NbboDatasetConfig):
            raise TypeError(
                f"{self.class_name} needs an NbboDatasetConfig, got "
                f"{type(config).__name__}."
            )
        if config.frequency != "tick":
            raise ValueError(
                f"{self.class_name}: frequency must be 'tick' (the raw tier "
                f"holds one row per NBBO record); the panel's bar size is "
                f"bar_interval. Got frequency {config.frequency!r}."
            )
        if config.bar_interval not in BAR_INTERVAL_SECONDS:
            raise ValueError(
                f"{self.class_name}: bar_interval {config.bar_interval!r} is "
                f"not one of {list(BAR_INTERVAL_SECONDS)}."
            )
        # Constructed here so a bad window (outside 04:00-20:00 ET, malformed,
        # or start >= end) fails at dataset construction. The exchange
        # calendar itself loads lazily, on the first `session_bounds` call.
        self._calendar = XnysSessionCalendar(config.session_start, config.session_end)

    @property
    def _tick_data_type(self) -> str:
        return self.DATA_TYPE

    @property
    def filter_stats_path(self) -> str:
        """`{zarr_file_path}{FILTER_STATS_SUFFIX}`, a sibling of the store."""
        return f"{self.config.zarr_file_path}{FILTER_STATS_SUFFIX}"

    @property
    def _resampler(self) -> NbboResampler:
        return NbboResampler(
            self.config.bar_interval, NbboFilterPolicy.from_config(self.config)
        )

    # -- sessions and axes ------------------------------------------------------

    def _session_dates(self) -> list[date]:
        """Every `date=` directory under the scan root, ascending."""
        root = self._scan_root()
        if not root.exists():
            return []
        dates = []
        for path in root.glob("date=*"):
            if path.is_dir():
                dates.append(date.fromisoformat(path.name.split("=", 1)[1]))
        return sorted(dates)

    def _session_bounds(self, dates) -> pl.DataFrame:
        """`(date, open, close)` per session date, naive UTC (D-23).

        Delegates to `XnysSessionCalendar.session_bounds`: a non-session date
        raises `ValueError` naming it, a session whose clipped window is empty
        is omitted.
        """
        return self._calendar.session_bounds(dates)

    def _dates_in_config_range(self) -> list[date]:
        start = date.fromisoformat(self.config.start_date)
        end = date.fromisoformat(self.config.end_date)
        return [day for day in self._session_dates() if start <= day <= end]

    def _raw_symbols(self, dates) -> list[str]:
        root = self._scan_root()
        symbols = set()
        for day in dates:
            for path in (root / f"date={day.isoformat()}").glob("symbol=*"):
                if path.is_dir():
                    symbols.add(path.name.split("=", 1)[1])
        return sorted(symbols)

    def _raw_axes_in_range(self) -> tuple[list[str], pd.DatetimeIndex]:
        """`(symbols, bar labels)` for the config's range, without resampling.

        Symbols are `config.symbols` when set, else the `symbol=` directory
        names. The timestamps are the session-grid LABELS of the session dates
        present in raw -- never the observed record timestamps.
        """
        self._assert_vendor_root()
        dates = self._dates_in_config_range()
        if self.config.symbols is not None:
            symbols = sorted(str(symbol) for symbol in self.config.symbols)
        else:
            symbols = self._raw_symbols(dates)
        if not symbols:
            # Refused before any store exists: an empty pinned axis would
            # create a store whose symbol coordinate has no labels to type.
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

        Every session date the window resampled is REPLACED wholesale (a
        window resamples whole sessions over every pinned symbol, so its
        counts for a date are complete); other dates are kept. `totals` is
        recomputed over the merged sessions, so re-resampling a date -- an
        extended session split across two UTC-day windows, or a rerun --
        never double-counts.
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
        # Round-tripped so the in-process value is exactly the file content.
        self.last_filter_stats = json.loads(json.dumps(payload, sort_keys=True))
        return self.last_filter_stats

    # -- densify ------------------------------------------------------------------

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[str] | None = None
    ) -> xr.Dataset:
        """Resample the sessions whose labels intersect `[start, end]` and
        densify them onto labels x symbols.

        The raw tier is filtered on the `date` hive key, never on a timestamp
        window: the seed record before the open must survive.
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
        # The config's range only: the same session set `_raw_axes_in_range`
        # pinned, so a raw date outside the range is never asked about.
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
        """Route the panel to NBBO validation, not OHLCV cleaning (D-14)."""
        return clean_nbbo_panel(data)

    def _widen_fill_values(self) -> dict:
        """No `anomaly_flag` here, so nothing to fill on a widen.

        Every panel variable is float64 and NaN is its "no data" value, which
        is exactly what a newly widened symbol has over its pre-listing
        history -- `n_updates` and `n_ambiguous_ties` included: NaN there
        means "no record existed", distinct from 0 ("in force, not updated").
        """
        return {}
