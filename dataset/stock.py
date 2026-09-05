from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

from base.config import DatasetConfig
from base.data import Dataset
from dataset.cleaning import dedup_raw_frame
from enums.data import BinanceCSVHeaders
from utils.file import file_date_filter, get_pqt_files
from utils.timer import Timer


class StockDataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    def _raw_data_to_xr(self) -> xr.Dataset:
        with Timer(f" {self.__class__.__name__}: from pqt"):
            files = get_pqt_files(self.config.raw_data_dir_path)
            stock_dfs = []
            for file in tqdm(files):
                stock_dfs.append(pl.scan_parquet(file))
            data = pl.concat(stock_dfs)
            # data = data.rename({"date": "timestamp", "ticker": "symbol"})
            data = data.filter(
                pl.col("timestamp")
                >= pl.lit(self.config.start_date).str.to_datetime(),
                pl.col("timestamp")
                <= pl.lit(self.config.end_date).str.to_datetime(),
            )
            data = data.sort(by=["timestamp", "symbol"])
            data = dedup_raw_frame(data, keep="last")
            data = data.collect().to_pandas().set_index(["timestamp", "symbol"])
            return data.to_xarray()

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        with Timer(f"{self.__class__.__name__}: to kunquant"):
            data = data.drop_vars(["open", "high", "low", "close", "volume"])
            data = data.rename(
                {
                    "adjOpen": "open",
                    "adjHigh": "high",
                    "adjLow": "low",
                    "adjClose": "close",
                    "adjVolume": "volume",
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
        raise ValueError("Not finished")

    def _xr_to_bars(
        self, data: xr.Dataset, symbol: str, venue: str = "BINANCE"
    ):
        raise ValueError("Not finished")

    def _to_nautilus(
        self, data: xr.Dataset, venue: str = "BINANCE", n_jobs: int = 16
    ):
        raise ValueError("Not finished")
