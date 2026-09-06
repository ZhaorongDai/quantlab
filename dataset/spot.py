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

from base.config import DatasetConfig
from base.data import MarketDataset
from dataset.cleaning import dedup_raw_frame, flag_anomalies, validate_schema
from enums.data import BinanceCSVHeaders
from utils.file import file_date_filter, get_csv_files
from utils.nautilus import (
    generate_bar_type_str,
    get_crypot_currency,
    get_crypto_currency_pair,
    parse_symbol_currencies,
)
from utils.timer import Timer


class SpotKlineDataset(MarketDataset):
    # Binance raw columns are Title-Case (Open/High/Low/Close/Volume), unlike
    # the shared dataset/cleaning.py module's lowercase convention (D-06..D-08
    # default, tuned for StockDataset's already-lowercase Tiingo columns).
    # `_clean()` below overrides validate_schema()'s required-column names to
    # match, rather than renaming columns pipeline-wide (out of this plan's
    # D-04 scope boundary -- see 02-05 deviation notes).
    _RAW_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Override base/data.py:Dataset._clean()'s default (lowercase-column)
        schema check with Binance's actual Title-Case OHLCV column names.
        flag_anomalies() still runs for schema/shape consistency with other
        Dataset subclasses; its price-like-column list is lowercase-only, so
        it is a documented no-op for spot data (no regression -- spot had no
        anomaly-flagging integration prior to 02-03/this plan)."""
        data = validate_schema(data, required_columns=self._RAW_REQUIRED_COLUMNS)
        data = flag_anomalies(data)
        return data

    @staticmethod
    def _spot_kline_to_df(csv_file: Path, before_2025: bool) -> pl.LazyFrame:
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
                    f"在文件夹 {self.config.raw_data_dir_path} 中未发现符合要求的 CSV 文件"
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
        base_symbol, quote_symbol = parse_symbol_currencies(symbol)
        base_currency = get_crypot_currency(symbol=base_symbol)
        quote_currency = get_crypot_currency(symbol=quote_symbol)

        # 创建货币对instrument
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
        """将xarray数据转换为Nautilus Trader的Bar对象

        Args:
            data: xarray数据集
            symbol: 交易对符号 (如 BTCUSDT)
            venue: 交易所名称

        Returns:
            list[Bar]: Bar对象列表
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

            # 验证数据完整性
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            return wrangler.process(df)

        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
            return []

    def _to_nautilus(
        self, data: xr.Dataset, venue: str = "BINANCE", n_jobs: int = 16
    ) -> tuple[list[list[Bar]], list[InstrumentId]]:
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
