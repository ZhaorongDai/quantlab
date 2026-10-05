"""Sharadar daily price panel with permatickers as the symbol axis.

Sharadar is the primary US-equity vendor (ADR 0023). Its price tables (SEP for
stocks) are keyed by ticker, and when a delisted company's ticker is reused
Sharadar renames that company's whole history to a suffixed ticker. The
ticker is therefore not a stable axis. The *permaticker*, Sharadar's
unchanging integer id of one share class, is: it appears only in the TICKERS
table, which maps each ``(table, ticker)`` to its permaticker.

``SharadarStockDataset`` converts the raw tier (the vendor's own rows, pulled
by ``quantlab.acquisition.sharadar`` into parquet under
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
the whole history on every ex-date.

Examples
--------
>>> config = SharadarDatasetConfig(
...     zarr_file_path="/data/zarrs/sharadar_sep_1d.zarr",
...     raw_data_dir_path="/data/downloads/sharadar",
... )
>>> SharadarStockDataset(config).from_raw_data().save()
>>> panel = SharadarStockDataset(config).panel("2024-01-02", "2024-01-05")
>>> sorted(panel.data_vars)
['anomaly_flag', 'close', 'high', 'low', 'open', 'volume']
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.base import MarketDataset
from quantlab.dataset.config import DatasetConfig, SharadarDatasetConfig
from quantlab.dataset.sharadar.tables import scan_raw_table, table
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: The panel's variables, in order.
PRICE_VARIABLES: tuple[str, ...] = ("open", "high", "low", "close", "volume")

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
            mapping = (
                scan_raw_table(root, "tickers")
                .filter(pl.col("table") == table(self.config.table).api_name)
                .select("ticker", "permaticker")
                .unique()
                .collect()
            )
            frame = self._map_permatickers(prices, mapping)
            if self.config.permatickers is not None:
                frame = frame.filter(
                    pl.col("permaticker").is_in(list(self.config.permatickers))
                )
            if frame.height == 0:
                raise ValueError(
                    f"{self.class_name}: no {self.config.table!r} row in "
                    f"[{self.config.start_date}, {self.config.end_date}] for "
                    f"the configured permatickers."
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
            ).with_columns(
                pl.col(name).cast(pl.Float64) for name in PRICE_VARIABLES
            )
        self._derivation_cache = frame
        return frame

    def _map_permatickers(
        self, prices: pl.DataFrame, mapping: pl.DataFrame
    ) -> pl.DataFrame:
        """Add each raw row's permaticker, refusing a missing or ambiguous one."""
        used = mapping.join(prices.select("ticker").unique(), on="ticker")
        ambiguous = (
            used.group_by("ticker")
            .agg(pl.col("permaticker").sort())
            .filter(pl.col("permaticker").list.len() > 1)
            .sort("ticker")
        )
        if ambiguous.height:
            sample = dict(ambiguous.head(_ERROR_SAMPLE).iter_rows())
            raise ValueError(
                f"{self.class_name}: {ambiguous.height} {self.config.table!r} "
                f"ticker(s) map to several permatickers in TICKERS, first "
                f"{sample}. Refusing rather than guessing which company a "
                f"row belongs to; re-pull TICKERS."
            )
        frame = prices.join(used, on="ticker", how="left")
        unmapped = (
            frame.filter(pl.col("permaticker").is_null())
            .get_column("ticker")
            .unique()
            .sort()
        )
        if unmapped.len():
            raise ValueError(
                f"{self.class_name}: {unmapped.len()} {self.config.table!r} "
                f"ticker(s) have no permaticker in TICKERS, first "
                f"{unmapped.head(_ERROR_SAMPLE).to_list()}. TICKERS is "
                f"probably older than the price table (a ticker changed "
                f"between the two pulls); re-pull TICKERS."
            )
        return frame

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
