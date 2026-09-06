from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

from base.config import DatasetConfig
from base.data import MarketDataset
from dataset.cleaning import dedup_raw_frame
from enums.data import BinanceCSVHeaders
from utils.file import file_date_filter, get_pqt_files
from utils.timer import Timer


class StockDataset(MarketDataset):
    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    def _scan_raw(self, start_date=None, end_date=None) -> pl.LazyFrame:
        """The shared LazyFrame pipeline: scan, concat, date-filter, sort,
        dedup -- returned UNCOLLECTED.

        Parameterised by the window so polars can push the date predicate
        DOWN into the parquet scan. That push-down is the whole reason the
        chunked path is memory-bounded for this class and not merely
        write-bounded: only the window's rows are ever materialised. `None`
        means "the config's own edge", which is what keeps
        `_raw_data_to_xr()` byte-identical to its pre-refactor self.
        """
        files = get_pqt_files(self.config.raw_data_dir_path)
        stock_dfs = []
        for file in tqdm(files):
            stock_dfs.append(pl.scan_parquet(file))
        # diagonal_relaxed: raw parquet files may come from different
        # acquisition sources/vendors with columns in a different order
        # (or a differing but compatible column set) -- pl.concat's
        # default ("vertical") requires exact column order across every
        # input and raises polars.exceptions.InvalidOperationError
        # otherwise, which would crash ingestion whenever raw files
        # under the same raw_data_dir_path don't share one exact writer.
        data = pl.concat(stock_dfs, how="diagonal_relaxed")
        # data = data.rename({"date": "timestamp", "ticker": "symbol"})
        start = self._as_datetime(
            self.config.start_date if start_date is None else start_date
        )
        end = self._as_datetime(
            self.config.end_date if end_date is None else end_date
        )
        data = data.filter(
            pl.col("timestamp") >= pl.lit(start),
            pl.col("timestamp") <= pl.lit(end),
        )
        data = data.sort(by=["timestamp", "symbol"])
        return dedup_raw_frame(data, keep="last")

    @staticmethod
    def _as_datetime(value) -> datetime:
        """Normalise a window edge to a naive `datetime`.

        An ISO date string resolves to that date at MIDNIGHT, exactly what
        the pre-refactor `pl.lit(...).str.to_datetime()` produced, so the
        inclusive `<=` end-edge semantics are unchanged. A `pd.Timestamp`
        coming from `TimeChunkPlanner.plan_from_timestamps()` is an OBSERVED
        timestamp and passes through with its time-of-day intact.
        """
        return pd.Timestamp(value).to_pydatetime()

    def _raw_axes_in_range(self) -> tuple[list[str], pd.DatetimeIndex]:
        """Both axes for the config's whole range, from one scan, WITHOUT
        densifying anything (D-02).

        Two single-column unique scans: polars projects each one on its own,
        so peak memory is one column of the raw frame rather than the dense
        `[timestamp, symbol]` grid. The symbol axis is `sorted()` for the
        same reason `base/constituent.py:_densify` sorts its all-time union
        -- it must match the coordinate order `to_xarray()` produces, which
        is the pandas MultiIndex level order.
        """
        scan = self._scan_raw()
        symbols = sorted(
            str(symbol)
            for symbol in scan.select("symbol").unique().collect()["symbol"].to_list()
        )
        timestamps = (
            scan.select("timestamp").unique().collect()["timestamp"].to_list()
        )
        return symbols, pd.DatetimeIndex(sorted(timestamps))

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: Optional[list[str]] = None
    ) -> xr.Dataset:
        """Densify ONE window, reindexed onto the pinned symbol axis.

        A pinned symbol with no row in this window becomes an all-NaN column
        -- and the integer columns it touches upcast to float. That is not a
        chunking artefact: the whole-range densification already produces
        exactly this for an untraded cell, because
        `set_index([...]).to_xarray()` emits the full `[timestamp, symbol]`
        cartesian product with NaN in the gaps. The two paths therefore
        agree, which is what
        `tests/test_chunked_ingest.py::test_chunked_store_matches_the_unchunked_store`
        pins.
        """
        data = self._scan_raw(start_date, end_date)
        data = data.collect().to_pandas().set_index(["timestamp", "symbol"])
        data = data.to_xarray()
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def _raw_data_to_xr(self) -> xr.Dataset:
        with Timer(f" {self.__class__.__name__}: from pqt"):
            # The whole range, on no pinned axis -- byte-identical to the
            # pre-refactor body, which is why every existing assertion in
            # tests/test_stock_dataset.py stands untouched.
            return self._raw_data_to_xr_window(
                self.config.start_date, self.config.end_date, symbols=None
            )

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
            # D-02: Tiingo supplies no native dollar-volume ("amount") column,
            # while KunQuant's Alpha101/Alpha158 AllData graphs derive vwap
            # from it. Synthesize the standard `volume * close` proxy here,
            # once and centrally, so every KunQuant factor class reading
            # US-equity data gets it without per-class duplication. The rename
            # above has already run, so `volume`/`close` are the ADJUSTED
            # series -- the proxy is adjusted dollar-volume, consistent with
            # the rest of the adjusted-price pipeline. Double-guarded: only
            # when the caller actually asks for `amount` and only when the
            # dataset does not already carry a real vendor column of that name.
            if "amount" in data_columns and "amount" not in data.data_vars:
                data = data.assign(amount=data["volume"] * data["close"])
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
