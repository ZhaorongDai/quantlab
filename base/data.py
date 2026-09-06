import datetime
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import polars as pl
import xarray as xr
from loguru import logger
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from tqdm import tqdm

from base.config import BaseDatasetConfig, DatasetConfig
from dataset.backend import XrBackend
from dataset.cleaning import clean_market_data
from enums.constant import Date
from utils.timer import Timer


class BaseDataset(ABC):
    """The shared, storage-medium-agnostic dataset contract (D-03, DATA-06).

    Everything the pipeline needs from a dataset lives here: the `config`
    lifecycle, the `XrBackend` storage round-trip (`read`/`save`), the
    `xr.Dataset` / `pl.LazyFrame` accessors, the `from_raw_data()` ingestion
    pipeline with its overridable `_clean()` hook, and the single abstract
    member `_raw_data_to_xr()` that every dataset kind implements for itself.

    The base is deliberately free of any nautilus or KunQuant concept -- no
    bar conversion, no `ParquetDataCatalog`, no compiled-graph input arrays,
    and no read of the market-only `raw_data_dir_path`/`catalog_path`/
    `market`/`frequency` config fields. That is what lets a dataset with no
    OHLCV shape at all -- an index-membership panel, say -- complete the whole
    persistence lifecycle by implementing exactly one abstract method, instead
    of carrying two meaningless `raise NotImplementedError` stubs.
    """

    def __init__(self, config: BaseDatasetConfig):
        # Ordering is load-bearing, and it is deliberately the OPPOSITE of
        # `base/factor.py:Factor.__init__`, which assigns its config first.
        # Here the config property setter below DOES reach the storage
        # backend -- through `_reset_symbols()` -> `read()` -- so the backend
        # must already exist by the time the setter fires. Do not "harmonize"
        # the two hierarchies: reversing these two lines raises
        # `AttributeError` on every dataset construction.
        self.data_backend = XrBackend()
        self.config = config

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def num_symbols(self) -> int:
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.size

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    @property
    def symbols(self) -> list[str]:
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.values.tolist()

    @property
    def time_interval(self) -> np.timedelta64:
        return (
            self.data_backend.get_xarray_dataset(["timestamp"])
            .diff(dim="timestamp")
            .to_series()
            .mode()  # 取众数，针对周末数据缺失的情况
            .values[0]
        )

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def _filter(self):
        self.data_backend.filter_by_date(
            "timestamp", self.config.start_date, self.config.end_date
        )
        if self.config.symbols is not None:
            self.data_backend.filter_by_symbol("symbol", self.config.symbols)

    @property
    def config(self) -> BaseDatasetConfig:
        return self._config

    @config.setter
    def config(self, config: BaseDatasetConfig):
        self._config = config
        self._config.name = self.import_path

        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

        # Normalise both dates to zero-padded ISO ONCE, here at the boundary.
        # Every downstream date comparison in this codebase is LEXICOGRAPHIC on
        # these strings -- `_clamp_coverage_start`'s `requested >=
        # coverage_start`, `_densify`'s `max(start_date, coverage_start)`,
        # `get_symbols_as_of`'s `as_of_date < coverage_start` -- so a non-padded
        # or non-ISO value ("2007-2-1", "01/01/2005") does not fail to match, it
        # compares WRONG and silently skips the clamp or the coverage guard.
        # Normalising once keeps every comparison downstream a plain string
        # comparison, which is why they are written that way.
        self._config.start_date = self._normalize_date(
            self._config.start_date, "start_date"
        )
        self._config.end_date = self._normalize_date(
            self._config.end_date, "end_date"
        )

        if self._config.symbols is not None:
            self._reset_symbols()

    def _normalize_date(self, value: str, field_name: str) -> str:
        try:
            return datetime.date.fromisoformat(str(value)).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.class_name}: {field_name} must be an ISO YYYY-MM-DD "
                f"date string, got {value!r}. Dates are compared "
                f"lexicographically throughout this pipeline, so a non-ISO "
                f"value compares wrong rather than failing to match."
            ) from exc

    def _reset_symbols(self):
        """Resolve `config.symbols` eagerly from the store, at
        config-assignment time.

        The default is exactly the behaviour every dataset class has always
        had: when the caller pinned a symbol subset, read the store (falling
        back to `from_raw_data()` if it does not exist yet) and overwrite
        `config.symbols` with whatever the store actually holds.

        Overridable seam: a dataset whose symbol axis is derived from its own
        source rather than from a store -- or one whose `from_raw_data()`
        fallback would perform a remote fetch merely to construct the object
        -- overrides this to a no-op and resolves its symbols inside
        `_raw_data_to_xr()` instead. Note the fallback only catches
        `FileNotFoundError`, so for such a dataset any network or parse error
        would otherwise escape `__init__`.
        """
        try:
            self.read()
        except FileNotFoundError:
            logger.warning(
                f"{self.class_name} data not found, try to read from csv"
            )
            self.from_raw_data()
        symbols = tuple(self._get_symbols())
        self._config.symbols = symbols

    def _get_symbols(self) -> list[str]:
        return self.data_backend.get_xarray_dataset(
            ["symbol", "timestamp"]
        ).symbol.values.tolist()

    def read(self, **kwargs):
        self.data_backend.read(self.config.zarr_file_path, **kwargs)
        self._filter()
        return self

    def save(self, **kwargs):
        with Timer(f"{self.__class__.__name__}: save"):
            self._filter()
            self.data_backend.write(self.config.zarr_file_path, **kwargs)

    def get_config(self) -> dict:
        return self.config.to_dict()  # type: ignore

    def get_lazyframe(self) -> pl.LazyFrame:
        return self.data_backend.get_lazyframe()

    def get_xarray_dataset(self) -> xr.Dataset:
        return self.data_backend.get_xarray_dataset(["timestamp", "symbol"])

    def from_raw_data(self) -> Self:
        data = self._raw_data_to_xr()
        data = self._clean(data)
        self.data_backend.to_internal(data)  # type: ignore
        return self

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Cleaning hook run after raw-to-xarray conversion, before persistence.

        Defaults to the shared market-data cleaning pipeline (anomaly-flagging,
        schema validation — see dataset/cleaning.py). Overridable so a future
        Dataset subclass whose data isn't OHLCV-shaped tabular market data
        (e.g. unstructured sources like news) is not forced through
        market-specific validation it doesn't apply to.
        """
        return clean_market_data(data)

    @abstractmethod
    def _raw_data_to_xr(self) -> xr.Dataset: ...


class MarketDataset(BaseDataset):
    """The market-data dataset backend.

    Owns everything nautilus- and KunQuant-specific: the
    `ParquetDataCatalog` write (`_write_catalog`), the bar-conversion path
    (`to_nautilus` / `_to_nautilus`), and the compiled-graph input path
    (`to_kunquant` / `_to_kunquant`). Keeping all five off `BaseDataset` is
    what lets a dataset with no bar, no catalog and no KunQuant
    representation subclass the shared base directly instead of carrying two
    meaningless `raise NotImplementedError` stubs (D-03).
    """

    # Narrowed for readers and type checkers only -- this is a bare
    # annotation, so it does not shadow `BaseDataset.config`. It records that
    # the three methods below legitimately read the market-only
    # `catalog_path` field, which lives on `DatasetConfig` and not on the
    # shared `BaseDatasetConfig`.
    config: DatasetConfig

    def _write_catalog(self, data: list):
        catalog = ParquetDataCatalog(
            self.config.catalog_path, fs_protocol="file"
        )
        catalog.write_data(data)

    def to_nautilus(
        self, venue: str = "BINANCE", n_jobs: int = 16, write: bool = True
    ) -> tuple[list[list], list[Instrument]]:
        data = self.read().get_xarray_dataset()
        data, instruments = self._to_nautilus(data, venue=venue, n_jobs=n_jobs)
        if write:
            for d in tqdm(data, desc="Writing data"):
                self._write_catalog(d)
            for instrument in tqdm(instruments, desc="Writing instruments"):
                self._write_catalog([instrument])
        return data, instruments

    def to_kunquant(
        self, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        data = self.read().get_xarray_dataset()
        return self._to_kunquant(data, data_columns)

    @abstractmethod
    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]: ...

    @abstractmethod
    def _to_nautilus(
        self, data: xr.Dataset, venue: str, n_jobs: int
    ) -> tuple[list[list], list[Instrument]]: ...

