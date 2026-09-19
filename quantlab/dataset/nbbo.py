"""The WRDS TAQ NBBO bar panel: raw `complete_nbbo` tick shards -> Zarr (03.9).

`NbboPanelDataset` reads the raw tier `WrdsTaqNbboAcquisition` writes
(`.../wrds/data_type=nbbo/date=/symbol=/`) and materialises a dense
`[timestamp, symbol]` panel of right-closed NBBO bars through
`dataset/nbbo_resample.py:NbboResampler`. The bar size is
`NbboDatasetConfig.bar_interval`; the raw tier's `frequency` stays `"tick"`.

This first version (plan 03.9-01) is the tracer's slice: the session window
is the config's ET wall-clock `session_start`/`session_end` localised per
date. Plan 06 replaces `_session_bounds` with the XNYS calendar (half days)
and adds the sidecar and multi-day hardening.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.base.config import DatasetConfig, NbboDatasetConfig
from quantlab.base.data import BaseDataset
from quantlab.dataset.cleaning import NBBO_PANEL_VARIABLES, clean_nbbo_panel
from quantlab.dataset.nbbo_resample import NbboResampler
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import BAR_INTERVAL_SECONDS
from quantlab.utils.timer import Timer


class NbboPanelDataset(StockDataset):
    """A dense NBBO bar panel resampled from WRDS TAQ `complete_nbbo` records.

    Reuses `StockDataset`'s vendor-root assertion, tick `data_type=` scan
    root, hive schema, single-vendor provenance check and `has_raw_data`.
    Overrides the axes, the window densifier and `_clean` (D-14: the OHLCV
    cleaner would raise on a panel that has no OHLCV).
    """

    #: The config class `quantlab/utils/module.py` rebuilds this dataset with.
    config_cls = NbboDatasetConfig

    #: The raw tier's `data_type=` hive key this panel is built from.
    DATA_TYPE = "nbbo"

    #: The time zone the session edges and the `date=` hive key are in.
    SESSION_TIME_ZONE = "America/New_York"

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

    @property
    def _tick_data_type(self) -> str:
        return self.DATA_TYPE

    @property
    def _resampler(self) -> NbboResampler:
        return NbboResampler(self.config.bar_interval)

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
        """`(date, open, close)` per session date, naive UTC.

        The config's ET wall-clock edges are localised per date, so DST is
        right on every day.
        """
        rows = []
        for day in dates:
            opens = pd.Timestamp(
                f"{day.isoformat()} {self.config.session_start}",
                tz=self.SESSION_TIME_ZONE,
            )
            closes = pd.Timestamp(
                f"{day.isoformat()} {self.config.session_end}",
                tz=self.SESSION_TIME_ZONE,
            )
            rows.append(
                {
                    "date": day,
                    "open": opens.tz_convert("UTC").tz_localize(None).to_pydatetime(),
                    "close": closes.tz_convert("UTC").tz_localize(None).to_pydatetime(),
                }
            )
        return pl.DataFrame(
            rows,
            schema={
                "date": pl.Date,
                "open": pl.Datetime("ns"),
                "close": pl.Datetime("ns"),
            },
        )

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
        labels = self._resampler.labels(self._session_bounds(dates))
        return symbols, pd.DatetimeIndex(labels["timestamp"].to_list())

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
        sessions = self._session_bounds(self._session_dates())
        labels = resampler.labels(sessions).filter(
            pl.col("timestamp").is_between(start, end, closed="both")
        )
        dates = labels["date"].unique().sort().to_list()
        sessions = sessions.filter(pl.col("date").is_in(dates))

        bars = None
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
                bars = resampler.resample(records, sessions)

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
        """No `anomaly_flag` here, so nothing to fill on a widen."""
        return {}
