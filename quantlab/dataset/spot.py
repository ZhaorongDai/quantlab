"""Binance spot klines as a ``(timestamp, symbol)`` panel.

``SpotKlineDataset`` reads the monthly, header-less CSV files that Binance
publishes for spot markets, stacks them into one panel per configured date
range, and exposes the KunQuant and Nautilus exits of ``MarketDataset``. It is
the crypto counterpart of ``quantlab/dataset/stock.py``.
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
    ``<SYMBOL>-...csv``; the columns come from ``BinanceCSVHeaders.SPOT`` and
    the symbol from the file name. Column names keep Binance's Title-Case
    (``Open``, ``High``, ...), and ``_to_kunquant`` maps them onto KunQuant's
    lowercase vocabulary.

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
    # shared lowercase default; `_clean()` passes these names instead.
    _RAW_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    def __init__(self, dataset_config: DatasetConfig):
        """Create the dataset from a ``DatasetConfig``."""
        super().__init__(dataset_config)

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Validate the Title-Case OHLCV schema and add ``anomaly_flag``.

        ``flag_anomalies`` only inspects lowercase price columns, so on this
        panel it adds an all-``False`` flag; it is kept for a consistent
        variable set across datasets.
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

        This is a non-memory-bounded implementation: it converts the whole
        configured range through ``_raw_data_to_xr`` and slices afterwards,
        so a chunked run bounds the write but not the densification, and pays
        one whole-range conversion per window. A bounded version would push
        the window down over the monthly file list with ``file_date_filter``
        before materialising, the way ``StockDataset`` pushes predicates into
        its parquet scan.

        When ``symbols`` is given the result is reindexed onto exactly that
        axis; a symbol with no row in the window comes back as an all-NaN
        column rather than being dropped.
        """
        data = self._raw_data_to_xr()
        data = data.sel(timestamp=slice(start_date, end_date))
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    @staticmethod
    def _spot_kline_to_df(csv_file: Path, before_2025: bool) -> pl.LazyFrame:
        """Scan one monthly kline CSV into a ``LazyFrame`` with a ``symbol`` column.

        Binance switched the ``Open time`` epoch unit from milliseconds to
        microseconds in 2025, so ``before_2025`` selects the parse; both
        paths yield microsecond timestamps. The symbol is the file name's
        first dash-separated token.
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
        """Stack the CSV files in the config's date range into a dense panel.

        Files are selected by the date in their name, concatenated, sorted and
        deduplicated on ``(timestamp, symbol)`` keeping the last row before the
        pandas ``to_xarray`` densification.

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
        """Rename Binance columns to KunQuant's and export float32 arrays.

        ``Quote asset volume`` becomes ``amount``; the OHLCV columns are
        lowercased. Each requested column is returned as a contiguous
        ``[time, symbol]`` float32 array.
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
        """Build the Nautilus ``CurrencyPair`` for a symbol such as ``BTCUSDT``."""
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
        """Convert one symbol's column of the panel to Nautilus ``Bar`` objects.

        Rows with any NaN are dropped first. Any failure is printed and an
        empty list is returned, so one bad symbol does not abort the parallel
        conversion.

        Parameters
        ----------
        data : xr.Dataset
            The full ``(timestamp, symbol)`` panel.
        symbol : str
            Trading pair to convert, e.g. ``BTCUSDT``.
        venue : str
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
