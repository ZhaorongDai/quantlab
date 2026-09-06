import datetime
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
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

        # Normalise `symbols` to the declared `tuple | None` HERE rather than
        # inside `_reset_symbols()`. `_reset_symbols()` is an overridable seam
        # -- `IndexConstituentDataset` correctly makes it a no-op -- so
        # normalising there left the declared contract false for that whole
        # branch of the hierarchy: whatever the caller passed (a list, from the
        # constituent config factories) survived unchanged and reached
        # `filter_by_symbol(col, symbols: tuple[str, ...])`. It worked by
        # accident because `.sel` accepts both.
        if self._config.symbols is not None:
            self._config.symbols = tuple(self._config.symbols)
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

    def _raw_axes_in_range(self) -> tuple[list[str], "pd.DatetimeIndex"]:
        """Return `(pinned_symbols, observed_timestamps)` for the config's
        whole date range, from ONE scan of the raw source.

        Overridable seam. **This default is correct but NOT memory-bounded:**
        it derives both axes from `_raw_data_to_xr()`, so it materialises the
        entire dense whole-range panel -- exactly the allocation
        `from_raw_data_chunked()` exists to avoid. It is the historical
        behaviour rather than a `raise NotImplementedError` stub, following
        `_reset_symbols()`'s idiom, so every existing subclass keeps working
        unchanged.

        A subclass whose raw source can push a date/column filter DOWN before
        materialisation (a `pl.LazyFrame` over parquet, say) overrides this
        and gets the memory bound; one that cannot inherits a working, slower
        default and is warned about it at run time.
        """
        data = self._raw_data_to_xr()
        symbols = [str(symbol) for symbol in data["symbol"].values.tolist()]
        return symbols, pd.DatetimeIndex(data["timestamp"].values)

    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset:
        """Densify ONE time window, onto `symbols` when a pinned axis is given.

        Overridable seam, and the same caveat as `_raw_axes_in_range()`
        applies: **this default is correct but NOT memory-bounded**, because
        it densifies the whole range and slices afterwards. Overriding it is
        what turns chunking from a bounded WRITE into a bounded DENSIFY.

        When `symbols` is supplied the returned panel's `symbol` coordinate
        equals it exactly, including symbols with no row in this window --
        those become all-NaN columns, which is the same value the whole-range
        densification already produces for an untraded cell.
        """
        data = self._raw_data_to_xr()
        data = data.sel(timestamp=slice(start_date, end_date))
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def from_raw_data_chunked(
        self,
        granularity: str = "year",
        ledger_path: str | None = None,
        append_dim: str = "timestamp",
    ) -> Self:
        """Densify and append ONE time window at a time (D-01).

        Peak memory scales with the WINDOW rather than the range, which is
        what makes the full multi-decade, full-market panel materialisable on
        a machine that cannot hold it whole.

        The ordering is load-bearing:

        1. The symbol axis is resolved ONCE over the whole range, BEFORE any
           window exists (D-02) -- the same all-time-union rule
           `base/constituent.py:_densify` follows. If each window derived its
           own axis, the chunks would carry inconsistent coordinates and the
           append would silently misalign, so every window is materialised on
           this one pinned axis and checked against it element-for-element.
        2. Windows come from the OBSERVED timestamp axis, never from calendar
           arithmetic: trading days are not calendar days.
        3. Completed windows are recorded in a sidecar ledger, so an
           interrupted run resumes at the first unwritten window (D-04).
        """
        from base.chunking import ChunkLedger, TimeChunkPlanner

        if type(self)._raw_data_to_xr_window is BaseDataset._raw_data_to_xr_window:
            logger.warning(
                f"{self.class_name}: _raw_data_to_xr_window has not been "
                f"overridden, so each window is produced by densifying the "
                f"WHOLE range and slicing. Chunking still bounds the write "
                f"and still gives a resumable run, but the memory win is "
                f"absent -- override the seam for a source that can push the "
                f"date filter down before materialising."
            )

        symbols, timestamps = self._raw_axes_in_range()
        planner = TimeChunkPlanner(granularity)
        windows = planner.plan_from_timestamps(timestamps)
        ledger = ChunkLedger(
            ledger_path or ChunkLedger.default_path(self.config.zarr_file_path),
            append_dim=append_dim,
        )

        logger.info(
            f"{self.class_name}: chunked ingestion over {len(windows)} "
            f"{granularity} window(s), {len(symbols)} pinned symbol(s), "
            f"{len(timestamps)} observed timestamp(s)."
        )

        first_timestamp = timestamps.min() if len(timestamps) else None
        boundaries = 0
        for start, end in windows:
            if ledger.is_written(start, end):
                logger.info(
                    f"{self.class_name}: window {start.date()}..{end.date()} "
                    f"already recorded in the ledger, skipping."
                )
                continue

            window = self._raw_data_to_xr_window(start, end, symbols)
            actual = [str(symbol) for symbol in window["symbol"].values.tolist()]
            if actual != symbols:
                raise ValueError(
                    f"{self.class_name}: window {start.date()}..{end.date()} "
                    f"came back on a symbol axis of {len(actual)} label(s), "
                    f"but the pinned whole-range axis has {len(symbols)}. "
                    f"Every window must be materialised on the pinned axis "
                    f"(D-02); appending this one would silently misalign "
                    f"every column in the store."
                )

            window = self._clean(window)
            window = self._pin_append_dtypes(window)
            if start != first_timestamp:
                boundaries += 1

            self.data_backend.to_internal(window)
            self.data_backend.append(
                self.config.zarr_file_path, append_dim=append_dim
            )
            ledger.record(start, end, int(window.sizes[append_dim]), symbols)
            logger.info(
                f"{self.class_name}: appended window "
                f"{start.date()}..{end.date()} "
                f"({int(window.sizes[append_dim])} row(s))."
            )

        if boundaries:
            logger.warning(
                f"{self.class_name}: cleaning ran per window, so at "
                f"{boundaries} chunk-boundary timestamp(s) `flag_anomalies` "
                f"had no prior sample to diff against and a single-step jump "
                f"across that boundary is not flagged. A bounded, documented "
                f"consequence of chunking -- finer --chunk granularity "
                f"produces more such boundaries, not fewer."
            )
        return self

    @staticmethod
    def _pin_append_dtypes(data: xr.Dataset) -> xr.Dataset:
        """Promote integer data variables to float64 before an append.

        The dtype of a window is a function of its own DENSITY: a window in
        which every pinned symbol traded on every timestamp keeps pandas'
        int64 for `volume`, while any window with a gap upcasts to float64
        for the NaN. Left alone, the store's dtype would therefore be decided
        by whichever window happened to be written first, and a later
        float64 NaN appended into an int64 variable is silently cast to 0 --
        a fabricated observation where data was missing.
        `XrBackend.append` refuses that append; this makes the refusal
        unreachable by pinning the dtype to what the dense panel is anyway
        (`estimate_dense_panel` sizes it at 8 bytes per value).

        Booleans are left alone: `anomaly_flag` is a flag, not a measurement.
        """
        promoted = {
            name: variable.astype("float64")
            for name, variable in data.data_vars.items()
            if np.issubdtype(variable.dtype, np.integer)
        }
        return data.assign(**promoted) if promoted else data

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

