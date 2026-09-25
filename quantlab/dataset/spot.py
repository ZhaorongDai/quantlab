"""Binance spot klines loaded as a ``(timestamp, symbol)`` panel.

A *kline* (candlestick) is one bar of open, high, low, close and volume
(OHLCV) for a trading pair over a fixed interval. A *panel* is an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, the format every
quantlab layer exchanges.

``SpotKlineDataset`` reads the monthly, header-less CSV files that Binance
publishes for its spot market and stacks them into one panel for the
configured date range. It can also export that panel to KunQuant (the
compiled factor engine) and to Nautilus Trader ``Bar`` objects. It is the
crypto counterpart of ``quantlab.dataset.stock``.
"""

from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr
from joblib import Parallel, delayed
from nautilus_trader.model.currencies import (
    BTC,
    USDT,
)
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import (
    InstrumentId,
    Symbol,
    Venue,
)
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler
from tqdm import tqdm

from quantlab.base.config import DatasetConfig
from quantlab.base.data import MarketDataset
from quantlab.dataset._support.cleaning import dedup_raw_frame, flag_anomalies, validate_schema
from quantlab.enums.data import BinanceCSVHeaders
from quantlab.utils.file import file_date_filter, get_csv_files
from quantlab.utils.nautilus import (
    generate_bar_type_str,
    get_crypto_currency,
    get_crypto_currency_pair,
    parse_symbol_currencies,
)
from quantlab.utils.timer import Timer


class SpotKlineDataset(MarketDataset):
    """Binance spot kline dataset built from monthly CSV files.

    The raw directory holds one header-less CSV per symbol and month, named
    ``<SYMBOL>-...csv``. The column names come from
    ``BinanceCSVHeaders.SPOT`` and the symbol comes from the file name.
    Columns keep Binance's Title-Case spelling (``Open``, ``High``, ...);
    ``_to_kunquant`` renames them to the lowercase names KunQuant expects.

    Parameters
    ----------
    dataset_config : DatasetConfig
        Paths (raw CSV directory, Zarr store, Nautilus catalog), market,
        frequency and date range of the dataset.

    Examples
    --------
    >>> config = DatasetConfig(
    ...     raw_data_dir_path="downloads/crypto_spot/1d/klines",
    ...     zarr_file_path="data/crypto_spot/1d/spot.zarr",
    ...     catalog_path="data/crypto_spot/catalog",
    ...     market="crypto_spot",
    ...     frequency="1d",
    ... )
    >>> SpotKlineDataset(config).from_raw_data().save()
    >>> panel = SpotKlineDataset(config).read().get_xarray_dataset()
    """

    # Binance columns are Title-Case, so the schema check cannot use the
    # shared lowercase default names; ``_clean`` passes these instead.
    _RAW_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    def __init__(self, dataset_config: DatasetConfig):
        """Initialize the dataset; see the class docstring for parameters."""
        super().__init__(dataset_config)

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Check the Title-Case OHLCV columns exist and add ``anomaly_flag``.

        ``flag_anomalies`` only inspects lowercase price columns, so on this
        panel the flag is always ``False``. It is still added so every
        dataset carries the same set of variables.

        Parameters
        ----------
        data : xr.Dataset
            The raw panel.

        Returns
        -------
        xr.Dataset
            The same panel with an ``anomaly_flag`` variable.
        """
        data = validate_schema(data, required_columns=self._RAW_REQUIRED_COLUMNS)
        data = flag_anomalies(data)
        return data

    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset:
        """Return the dense panel for one time window.

        This is a non-memory-bounded implementation. It converts the whole
        configured range with ``_raw_data_to_xr`` and slices afterwards, so
        a chunked build limits the size of each write but not of the
        conversion, and repeats the full conversion for every window. A
        bounded version would filter the monthly file list by date with
        ``file_date_filter`` before loading, the way ``StockDataset``
        filters inside its parquet scan.

        Parameters
        ----------
        start_date, end_date : date-like
            Inclusive bounds of the window.
        symbols : list of str, optional
            If given, the result has exactly these symbols, in this order. A
            symbol with no row in the window becomes an all-NaN column
            instead of being dropped.

        Returns
        -------
        xr.Dataset
            The panel restricted to the window.
        """
        data = self._raw_data_to_xr()
        data = data.sel(timestamp=slice(start_date, end_date))
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    @staticmethod
    def _spot_kline_to_df(csv_file: Path, before_2025: bool) -> pl.LazyFrame:
        """Scan one monthly kline CSV into a ``LazyFrame`` with a ``symbol`` column.

        Binance changed the unit of ``Open time`` from milliseconds to
        microseconds in 2025, so ``before_2025`` picks the right parse. Both
        paths return microsecond timestamps. The symbol is the part of the
        file name before the first dash.

        Parameters
        ----------
        csv_file : Path
            One monthly kline CSV file.
        before_2025 : bool
            True if the file covers a month before 2025 (millisecond epochs).

        Returns
        -------
        pl.LazyFrame
            The file's rows with a parsed ``Open time`` and a ``symbol`` column.
        """
        df = pl.scan_csv(
            str(csv_file), has_header=False, new_columns=BinanceCSVHeaders.SPOT
        )

        if before_2025:
            df = df.with_columns(
                pl.from_epoch("Open time", time_unit="ms")
                .dt.cast_time_unit("us")
                .alias("Open time")
            )
        else:
            df = df.with_columns(
                pl.from_epoch("Open time", time_unit="us").alias("Open time")
            )
        df = df.with_columns(
            pl.lit(str(csv_file.name).split("-")[0]).alias("symbol"),
        )
        return df

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Stack the CSV files in the configured date range into a dense panel.

        Files are selected by the date in their name, concatenated and sorted.
        Duplicate ``(timestamp, symbol)`` rows keep the last occurrence. The
        result is then turned into a dense panel with pandas ``to_xarray``,
        so missing ``(timestamp, symbol)`` cells become NaN.

        Returns
        -------
        xr.Dataset
            The dense panel for the configured date range.

        Raises
        ------
        ValueError
            If no CSV file in the raw directory matches the range.
        """
        with Timer(f" {self.__class__.__name__}: from csv"):
            csv_files = get_csv_files(self.config.raw_data_dir_path)
            csv_files = file_date_filter(
                csv_files,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            csv_before_2025 = file_date_filter(csv_files, end_date="2024-12-31")
            csv_after_2025 = file_date_filter(
                csv_files, start_date="2025-01-01"
            )

            dfs: list[pl.LazyFrame] = []
            if csv_after_2025:
                dfs.extend(
                    self._spot_kline_to_df(csv, before_2025=False)
                    for csv in csv_after_2025
                )
            if csv_before_2025:
                dfs.extend(
                    self._spot_kline_to_df(csv, before_2025=True)
                    for csv in csv_before_2025
                )
            if not dfs:
                raise ValueError(
                    f"No CSV file matching the configured date range was found "
                    f"under {self.config.raw_data_dir_path}"
                )

            res = pl.concat(dfs)
            res = res.rename({"Open time": "timestamp"})
            res = res.sort(by=["timestamp", "symbol"])
            res = dedup_raw_frame(res, keep="last")
            res = res.collect().to_pandas().set_index(["timestamp", "symbol"])
            return res.to_xarray()

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Rename Binance columns to KunQuant's names and export float32 arrays.

        ``Quote asset volume`` (traded value in the quote currency) becomes
        ``amount`` and the OHLCV columns are lowercased.

        Parameters
        ----------
        data : xr.Dataset
            The panel to export.
        data_columns : tuple of str
            KunQuant input names to export, e.g. ``("open", "close")``.

        Returns
        -------
        input_dict : dict of str to np.ndarray
            One C-contiguous float32 array of shape ``[time, symbol]`` per
            requested column.
        symbols : np.ndarray
            Symbol labels of the second axis.
        timestamp : np.ndarray
            Timestamps of the first axis.
        """
        with Timer(f"{self.__class__.__name__}: to kunquant"):
            data = data.rename(
                {
                    "Quote asset volume": "amount",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            data = data.sortby(["timestamp", "symbol"])
            timestamp = data["timestamp"].values
            symbols = data["symbol"].values
            input_dict = {}
            for col in data_columns:
                input_dict[col] = np.ascontiguousarray(
                    data[col].to_numpy().astype(np.float32)
                )  # [time, symbol]
            return input_dict, symbols, timestamp

    @staticmethod
    def _get_instrument(symbol: str, venue: str):
        """Build the Nautilus ``CurrencyPair`` instrument for a symbol such as ``BTCUSDT``.

        Parameters
        ----------
        symbol : str
            Binance pair name; base and quote currencies are parsed from it.
        venue : str
            Venue (exchange) name, e.g. ``"BINANCE"``.

        Returns
        -------
        CurrencyPair
            The Nautilus instrument.
        """
        base_symbol, quote_symbol = parse_symbol_currencies(symbol)
        base_currency = get_crypto_currency(symbol=base_symbol)
        quote_currency = get_crypto_currency(symbol=quote_symbol)

        currency_pair = get_crypto_currency_pair(
            symbol=symbol,
            base=base_currency,
            quote=quote_currency,
            venue=venue,
        )
        return currency_pair

    def _xr_to_bars(
        self, data: xr.Dataset, symbol: str, venue: str = "BINANCE"
    ):
        """Convert one symbol of the panel to Nautilus ``Bar`` objects.

        Rows with any NaN are dropped first. Any error is printed and an
        empty list is returned, so one bad symbol does not stop the parallel
        conversion of the others.

        Parameters
        ----------
        data : xr.Dataset
            The full ``(timestamp, symbol)`` panel.
        symbol : str
            Trading pair to convert, e.g. ``BTCUSDT``.
        venue : str, default "BINANCE"
            Venue name used in the bar type.

        Returns
        -------
        list
            The list of bars for ``symbol``, or ``[]`` on failure.
        """
        try:
            d = data.sel(symbol=symbol)

            currency_pair = self._get_instrument(symbol=symbol, venue=venue)
            bar_type_str = generate_bar_type_str(
                time_interval=self.time_interval, symbol=symbol, venue=venue
            )
            bar_type = BarType.from_str(bar_type_str)

            wrangler = BarDataWrangler(
                instrument=currency_pair, bar_type=bar_type
            )

            df = d.to_dataframe().dropna().reset_index()
            df = df.rename(
                {
                    "Close": "close",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                },
                axis=1,
            )

            required_columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            df = df[required_columns].set_index("timestamp")

            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            return wrangler.process(df)

        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
            return []

    def _to_nautilus(
        self, data: xr.Dataset, venue: str = "BINANCE", n_jobs: int = 16
    ) -> tuple[list[list[Bar]], list[InstrumentId]]:
        """Convert every symbol to Nautilus bars in parallel.

        Parameters
        ----------
        data : xr.Dataset
            The full ``(timestamp, symbol)`` panel.
        venue : str, default "BINANCE"
            Venue name used for the instruments and bar types.
        n_jobs : int, default 16
            Number of joblib worker processes.

        Returns
        -------
        tuple[list[list[Bar]], list[InstrumentId]]
            ``(bars, instruments)``: one list of bars per symbol, in the
            order of ``self.symbols``, and the matching currency pairs.
        """
        symbols = self.symbols
        instruments = [
            self._get_instrument(symbol=symbol, venue=venue)
            for symbol in symbols
        ]
        res: list = Parallel(n_jobs=n_jobs)(  # type: ignore
            delayed(self._xr_to_bars)(data, symbol, venue)
            for symbol in tqdm(symbols, desc="To nautilus bar")
        )
        return res, instruments
