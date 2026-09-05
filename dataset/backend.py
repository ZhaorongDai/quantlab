from pathlib import Path
from typing import Optional, Self

import pandas as pd
import polars as pl
import xarray as xr

from base.backend import DataBackend


class XrBackend(DataBackend):
    def __init__(self) -> None:
        super().__init__()

    def read(self, path: str, overwrite: bool = False, **kwargs) -> Self:
        if not overwrite and hasattr(self, "data"):
            return self

        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = xr.open_dataset(path, **kwargs)
        return self

    def write(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("mode", "w")
        self.data.to_zarr(path, **kwargs)
        return self

    def to_internal(self, data: xr.Dataset) -> Self:
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        self.data = self.data.sel({col: slice(start_date, end_date)})
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        self.data = self.data.sel({col: list(symbols)})
        return self

    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        return self.data

    def get_lazyframe(self) -> pl.LazyFrame:
        data = self.data.to_dataframe().reset_index()
        return pl.from_pandas(data).lazy()


class PlBackend(DataBackend):
    def read(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = pl.scan_parquet(path)
        return self

    def write(self, path: str, **kwargs) -> Self:
        self.data.collect().write_parquet(path, **kwargs)
        return self

    def to_internal(self, data: pl.LazyFrame) -> Self:
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        self.data = self.data.filter(
            pl.col(col).is_between(
                pl.lit(pd.to_datetime(start_date)),
                pl.lit(pd.to_datetime(end_date)),
            )
        )
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        self.data = self.data.filter(pl.col(col).is_in(symbols))
        return self

    def get_lazyframe(self) -> pl.LazyFrame:
        return self.data

    def get_xarray_dataset(self, indexes: list[str]) -> xr.Dataset:
        data = self.data.collect().to_pandas()
        data = data.set_index(indexes)
        return xr.Dataset.from_dataframe(data)
