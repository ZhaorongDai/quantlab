"""Sharadar daily price panel with permatickers as the symbol axis.

Sharadar is the primary US-equity vendor (ADR 0023). Its price tables (SEP for
stocks) are keyed by ticker, and when a delisted company's ticker is reused
Sharadar renames that company's whole history to a suffixed ticker. The
ticker is therefore not a stable axis. The *permaticker*, Sharadar's
unchanging integer id of one share class, is: it appears only in the TICKERS
table, which maps each ``(table, ticker)`` to its permaticker.

``SharadarStockDataset`` converts the raw tier (the vendor's own rows, pulled
by ``quantlab.acquisition.sharadar.client`` into parquet under
``<download-dir>/sharadar/``) into a dense ``(timestamp, symbol)`` Zarr panel
whose ``symbol`` coordinate is the int64 permaticker. Each raw row is mapped
through the TICKERS rows of its own table, so a ticker of a fund never maps a
stock. The conversion refuses a raw ticker TICKERS does not know, a ticker
TICKERS maps to two permatickers, and two rows of one permaticker on one
date, rather than guess which company a row belongs to.

The panel holds raw (unadjusted) prices, so a later dividend or split never
rewrites a stored row:

- ``close`` is SEP's ``closeunadj``;
- ``open``/``high``/``low`` are SEP's split-adjusted values times
  ``closeunadj / close``, and ``volume`` is SEP's split-adjusted volume
  divided by that ratio (Sharadar's own imputation of the raw values).

Sharadar's adjusted columns (``close`` and the other split-adjusted values,
``closeadj``) and ``lastupdated`` are not stored: the vendor rewrites them over
the whole history on every ex-date. quantlab builds its own adjusted prices
instead, from the raw prices and the dividend, spinoff-value and split events
of the ACTIONS table, with the CRSP panel's convention and names (``adjOpen``, ``adjHigh``,
``adjLow``, ``adjClose``, ``adjVolume``, ``divCash``, ``splitFactor``; see
``SharadarStockDataset._adjust``). The panel therefore carries the same
twelve daily variables as a CRSP or Tiingo panel, and a factor, label or
backtest reads it without knowing the vendor. ``tradable_bars`` and
``delisting_bars`` are the ``MarketDataset`` defaults: a bar without a fill
price is untradable, and a delisted security is settled at its last close,
with no delisting return imputed (Sharadar gives none).

Examples
--------
>>> config = SharadarDatasetConfig(
...     zarr_file_path="/data/zarrs/sharadar_sep_1d.zarr",
...     raw_data_dir_path="/data/downloads/sharadar",
... )
>>> SharadarStockDataset(config).from_raw_data().save()
>>> panel = SharadarStockDataset(config).panel("2024-01-02", "2024-01-05")
>>> sorted(panel.data_vars)
['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'anomaly_flag', 'close', 'divCash', 'high', 'low', 'open', 'splitFactor', 'volume']
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.dataset.base import MarketDataset
from quantlab.dataset.config import DatasetConfig, SharadarDatasetConfig
from quantlab.dataset.sharadar.membership import ROSTER_UNIVERSES, sp500_intervals
from quantlab.dataset.sharadar.tables import (
    map_permatickers,
    permaticker_mapping,
    scan_raw_table,
    table,
)
from quantlab.enums.data import TiingoColumns
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: The panel's variables: the twelve shared daily variables, in
#: ``TiingoColumns.EOD`` order, as the CRSP panel holds them.
PRICE_VARIABLES: tuple[str, ...] = tuple(TiingoColumns.EOD.split(","))

#: ACTIONS types that pay cash per share on their date, as ``divCash``:
#: ordinary dividends and the value of spun-off shares.
DISTRIBUTIONS: tuple[str, ...] = ("dividend", "spinoffdividend")

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


class SharadarStockDataset(MarketDataset):
    """Dense daily panel of raw OHLCV from a Sharadar price table, keyed by permaticker.

    Parameters
    ----------
    dataset_config : SharadarDatasetConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``; ``table`` names
        the price table (``"sep"``); ``permatickers`` optionally restricts the
        conversion. The ticker-based ``symbols`` field must be unset.

    Examples
    --------
    >>> ds = SharadarStockDataset(config)
    >>> ds.from_raw_data().save()
    >>> SharadarStockDataset(config).panel("2024-01-02", "2024-01-05").symbol.dtype
    dtype('int64')
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarDatasetConfig

    def _normalize_config(self, config: DatasetConfig) -> SharadarDatasetConfig:
        """Normalise as the base class does, then check the Sharadar fields.

        Refuses anything that is not a ``SharadarDatasetConfig``, a table
        that is not a price table, any ``symbols`` value and an empty
        ``permatickers``. ``permatickers`` is returned as a tuple of ints.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarDatasetConfig``.
        ValueError
            For the other invalid fields.
        KeyError
            If ``table`` is not a known Sharadar table.

        Examples
        --------
        >>> SharadarStockDataset(dataclasses.replace(config, symbols=("AAPL",)))
        Traceback (most recent call last):
        ValueError: SharadarStockDataset: config.symbols is not selectable ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarDatasetConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarDatasetConfig, got "
                f"{type(config).__name__}."
            )
        if "closeunadj" not in table(config.table).schema:
            raise ValueError(
                f"{self.class_name}: table {config.table!r} is not a price table."
            )
        if config.symbols is not None:
            raise ValueError(
                f"{self.class_name}: config.symbols is not selectable on a "
                f"Sharadar panel; got {config.symbols!r}. The symbol axis is "
                f"the permaticker, and a ticker can be renamed or reused. Use "
                f"config.permatickers instead."
            )
        if config.permatickers is not None:
            permatickers = tuple(int(value) for value in config.permatickers)
            if not permatickers:
                raise ValueError(
                    f"{self.class_name}: config.permatickers is empty. Pass "
                    f"None for every security, or name at least one."
                )
            config = dataclasses.replace(config, permatickers=permatickers)
        if config.roster_universe is not None and config.roster_universe not in ROSTER_UNIVERSES:
            raise ValueError(
                f"{self.class_name}: roster_universe {config.roster_universe!r} "
                f"is not a Sharadar universe; known: {ROSTER_UNIVERSES}."
            )
        if config.category_filter is not None:
            categories = tuple(str(value) for value in config.category_filter)
            if not categories:
                raise ValueError(
                    f"{self.class_name}: config.category_filter is empty. Pass "
                    f"None to keep every category, or name at least one."
                )
            config = dataclasses.replace(config, category_filter=categories)
        return config

    def _on_config_installed(self) -> None:
        """Drop the derivation cached for the previous config."""
        self._derivation_cache: pl.DataFrame | None = None

    def _derivation(self) -> pl.DataFrame:
        """Return the panel's rows for the whole configured window, cached.

        Returns
        -------
        pl.DataFrame
            Columns ``timestamp``, ``symbol`` (the permaticker) and
            ``PRICE_VARIABLES``, one row per pair.

        Raises
        ------
        ValueError
            If a raw ticker has no permaticker, a ticker has two, two rows
            share a permaticker and date, or the window holds no row.
        """
        if self._derivation_cache is not None:
            return self._derivation_cache
        root = self.config.raw_data_dir_path
        start = datetime.fromisoformat(self.config.start_date).date()
        end = datetime.fromisoformat(self.config.end_date).date()
        with Timer(f"{self.class_name}: derive"):
            prices = (
                scan_raw_table(root, self.config.table)
                .filter(pl.col("date").is_between(pl.lit(start), pl.lit(end)))
                .collect()
            )
            mapping = permaticker_mapping(root, self.config.table)
            frame = self._map_permatickers(prices, mapping)
            frame = frame.filter(pl.col("permaticker").is_in(self._universe()))
            if frame.height == 0:
                raise ValueError(
                    f"{self.class_name}: no {self.config.table!r} row in "
                    f"[{self.config.start_date}, {self.config.end_date}] for "
                    f"the configured universe."
                )
            self._assert_unique_keys(frame)
            ratio = (
                pl.when(pl.col("close") > 0)
                .then(pl.col("closeunadj") / pl.col("close"))
                .otherwise(None)
            )
            frame = frame.select(
                pl.col("date").cast(pl.Datetime("ns")).alias("timestamp"),
                pl.col("permaticker").alias("symbol"),
                (pl.col("open") * ratio).alias("open"),
                (pl.col("high") * ratio).alias("high"),
                (pl.col("low") * ratio).alias("low"),
                pl.col("closeunadj").alias("close"),
                (pl.col("volume") / ratio).alias("volume"),
                ratio.alias("_ratio"),
            )
            events = self._events(mapping, start, end)
            frame = self._adjust(frame, events).select(
                "timestamp",
                "symbol",
                *(pl.col(name).cast(pl.Float64) for name in PRICE_VARIABLES),
            )
        self._derivation_cache = frame
        return frame

    def _universe(self) -> list[int]:
        """Return the permatickers the conversion keeps.

        An explicit roster (``permatickers``, the members of
        ``roster_universe`` inside the window, or both) is kept whole.
        Without one, the market universe is every permaticker whose TICKERS
        ``category`` is in ``category_filter``, or every one when that is
        ``None``.
        """
        config = self.config
        root = config.raw_data_dir_path
        if config.permatickers is None and config.roster_universe is None:
            tickers = scan_raw_table(root, "tickers").filter(
                pl.col("table").is_in(table(config.table).tickers_labels)
            )
            if config.category_filter is not None:
                tickers = tickers.filter(
                    pl.col("category").is_in(list(config.category_filter))
                )
            return tickers.select("permaticker").unique().collect().to_series().to_list()
        roster = set(config.permatickers or ())
        if config.roster_universe is not None:
            start = date.fromisoformat(config.start_date)
            end = date.fromisoformat(config.end_date)
            spells = sp500_intervals(root).filter(
                (pl.col("start_date") <= end) & (pl.col("end_date") >= start)
            )
            roster.update(spells.get_column("permaticker").to_list())
        return sorted(roster)

    def _events(self, mapping: pl.DataFrame, start, end) -> pl.DataFrame:
        """Return the window's cash distributions and splits per permaticker and date.

        ACTIONS is keyed by the same current ticker as the price table, so its
        rows are mapped through the price table's TICKERS rows; a row of a
        ticker outside the price table is another security and is dropped.
        The cash distributions are the ``DISTRIBUTIONS`` actions: dividends
        and ``spinoffdividend``, the dollar value of the spun-off shares
        issued per parent share (adjusted for later splits like a dividend).
        The ``spinoff`` row of the same event gives the share ratio and is
        not counted again. Several distributions on one date are summed,
        several splits multiplied.

        Returns
        -------
        pl.DataFrame
            Columns ``timestamp``, ``symbol``, ``_dividend`` (Sharadar's
            split-adjusted value, null when none) and ``_split`` (new shares
            per old share, null when none).
        """
        actions = (
            scan_raw_table(self.config.raw_data_dir_path, "actions")
            .filter(
                pl.col("date").is_between(pl.lit(start), pl.lit(end))
                & pl.col("action").is_in([*DISTRIBUTIONS, "split"])
            )
            .collect()
            .join(mapping, on="ticker", how="inner")
        )
        return actions.group_by(
            pl.col("date").cast(pl.Datetime("ns")).alias("timestamp"),
            pl.col("permaticker").alias("symbol"),
        ).agg(
            pl.col("value").filter(pl.col("action").is_in(DISTRIBUTIONS)).sum().alias("_dividend"),
            pl.col("value").filter(pl.col("action") == "split").product().alias("_split"),
            pl.col("action").is_in(DISTRIBUTIONS).any().alias("_has_dividend"),
            (pl.col("action") == "split").any().alias("_has_split"),
        ).select(
            "timestamp",
            "symbol",
            pl.when("_has_dividend").then("_dividend").alias("_dividend"),
            pl.when("_has_split").then("_split").alias("_split"),
        )

    def _adjust(self, frame: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
        """Add the events and the adjusted prices to the raw rows.

        Follows the CRSP path's convention (``quantlab.dataset.crsp``):

        - ``divCash`` is the cash paid per share held on the ex-date, 0.0
          on other rows. ACTIONS gives a dividend adjusted for every later
          split, so it is multiplied back by the row's
          ``closeunadj / close`` (SEP's own later-split factor; ACTIONS and
          SEP are adjusted for the same splits when pulled together).
        - ``splitFactor`` is the split's new shares per old share on its
          effective date, 1.0 on other rows.
        - The day's total return is
          ``(close * splitFactor + divCash) / close_prev - 1``, with
          ``close_prev`` the last earlier positive raw close, so a halt is
          spanned. ``adjClose`` is each permaticker's *anchor* (its first
          row in the window with a positive close) grown by the product of
          those returns since, and NaN where ``close`` is missing.
        - ``adjOpen``/``adjHigh``/``adjLow`` are scaled by
          ``adjClose / close``; ``adjVolume`` is the raw volume in the
          anchor's shares, divided by the splits since the anchor.

        An event on a date without a positive close cannot enter the chain;
        it is logged and dropped. No delisting return is imputed.
        """
        frame = frame.join(events, on=["timestamp", "symbol"], how="left").sort(
            "symbol", "timestamp"
        )
        # Only the panel's own securities: the events of one the universe
        # filtered out are not dropped from anything.
        dropped = events.join(
            frame.select("symbol").unique(), on="symbol", how="semi"
        ).join(
            frame.filter(pl.col("close") > 0).select("timestamp", "symbol"),
            on=["timestamp", "symbol"],
            how="anti",
        )
        if dropped.height:
            sample = [
                f"{row['symbol']}@{str(row['timestamp'])[:10]}"
                for row in dropped.sort("symbol", "timestamp").head(_ERROR_SAMPLE).to_dicts()
            ]
            logger.warning(
                f"{self.class_name}: {dropped.height} dividend/split event(s) "
                f"fall on a date without a positive close and are left out "
                f"of the adjusted price, first {sample}."
            )
        frame = frame.with_columns(
            (pl.col("_dividend") * pl.col("_ratio")).fill_null(0.0).alias("divCash"),
            pl.col("_split").fill_null(1.0).alias("splitFactor"),
        )
        positive = pl.when(pl.col("close") > 0).then(pl.col("close"))
        close_prev = positive.shift(1).forward_fill().over("symbol")
        ret = (
            pl.col("close") * pl.col("splitFactor") + pl.col("divCash")
        ) / close_prev - 1.0
        frame = frame.with_columns(
            (1.0 + ret.fill_null(0.0)).cum_prod().over("symbol").alias("_G"),
            pl.col("splitFactor").cum_prod().over("symbol").alias("_S"),
        )
        anchor = (
            frame.filter(pl.col("close") > 0)
            .group_by("symbol")
            .agg(
                pl.col("close").first().alias("_close_anchor"),
                pl.col("_G").first().alias("_G_anchor"),
                pl.col("_S").first().alias("_S_anchor"),
            )
        )
        unanchored = frame.select("symbol").unique().join(anchor, on="symbol", how="anti")
        if unanchored.height:
            logger.warning(
                f"{self.class_name}: {unanchored.height} permaticker(s) have no "
                f"positive close in the window, so no adjusted price, first "
                f"{unanchored.sort('symbol').head(_ERROR_SAMPLE)['symbol'].to_list()}."
            )
        frame = frame.join(anchor, on="symbol", how="left").with_columns(
            pl.when(pl.col("close") > 0)
            .then(pl.col("_close_anchor") * pl.col("_G") / pl.col("_G_anchor"))
            .alias("adjClose")
        )
        factor = pl.col("adjClose") / pl.col("close")
        return frame.with_columns(
            (pl.col("open") * factor).alias("adjOpen"),
            (pl.col("high") * factor).alias("adjHigh"),
            (pl.col("low") * factor).alias("adjLow"),
            (pl.col("volume") * pl.col("_S_anchor") / pl.col("_S")).alias("adjVolume"),
        )

    def _map_permatickers(
        self, prices: pl.DataFrame, mapping: pl.DataFrame
    ) -> pl.DataFrame:
        """Add each raw row's permaticker, refusing a missing or ambiguous one."""
        return map_permatickers(
            prices, mapping, owner=self.class_name, code=self.config.table
        )

    def _assert_unique_keys(self, frame: pl.DataFrame) -> None:
        """Raise if two rows share a permaticker and a date."""
        duplicates = (
            frame.group_by("permaticker", "date")
            .agg(pl.col("ticker").sort())
            .filter(pl.col("ticker").list.len() > 1)
            .sort("permaticker", "date")
        )
        if duplicates.height:
            sample = [
                f"{row['permaticker']}@{row['date']}:{row['ticker']}"
                for row in duplicates.head(_ERROR_SAMPLE).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {duplicates.height} (permaticker, date) "
                f"pair(s) have several raw rows, first {sample}. Two tickers "
                f"of one permaticker priced on one date cannot both be the "
                f"security's price; refusing rather than choosing one."
            )

    # -- axes and windows ---------------------------------------------------

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the permatickers (sorted) and the dates of the derivation."""
        derivation = self._derivation()
        symbols = sort_symbol_axis(
            derivation.get_column("symbol").unique().to_list()
        )
        timestamps = derivation.get_column("timestamp").unique().sort().to_list()
        return symbols, pd.DatetimeIndex(timestamps)

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Return one window of the derivation as a dense panel.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list of int, optional
            Permatickers to reindex onto; ``None`` keeps those in the window.
        """
        window = self._derivation().filter(
            pl.col("timestamp").is_between(
                pl.lit(pd.Timestamp(start_date).to_datetime64()),
                pl.lit(pd.Timestamp(end_date).to_datetime64()),
            )
        )
        if symbols is not None:
            symbols = [int(s) for s in symbols]
            window = window.filter(pl.col("symbol").is_in(symbols))
        data = window.to_pandas().set_index(["timestamp", "symbol"]).to_xarray()
        if symbols is not None:
            data = data.reindex(symbol=symbols)
        else:
            data = data.reindex(symbol=sort_symbol_axis(data.symbol.values.tolist()))
        return data.sortby("timestamp")

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the dense panel for the whole configured window."""
        return self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=None
        )

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export ``data_columns`` as ``[time, symbol]`` float32 arrays."""
        with Timer(f"{self.class_name}: to kunquant"):
            return self._kunquant_arrays(self.to_shared_names(data), data_columns)
