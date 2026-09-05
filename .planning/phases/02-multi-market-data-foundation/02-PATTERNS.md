# Phase 2: Multi-Market Data Foundation - Patterns

**Generated:** 2026-09-04
**Purpose:** Concrete, file-by-file implementation patterns for the planner. Extracted from direct reads of the actual files this phase touches (not RESEARCH.md's paraphrase). Every excerpt below is copy-pasted from the real source, not reconstructed from memory.

<scope>
## Files This Phase Creates or Modifies

| File | Action | Role |
|------|--------|------|
| `base/config.py` | MODIFY | Add `market`/`frequency` fields to `DatasetConfig`; add new `AcquisitionConfig` dataclass |
| `config/__init__.py` | MODIFY | Retrofit all 4 existing factories to derive paths from `market`/`frequency`; add new `stock_kline_config()` |
| `dataset/stock.py` | MODIFY | Add dedup-before-`to_xarray()` in `_raw_data_to_xr()` (Pitfall 2) |
| `dataset/spot.py` | MODIFY | Add dedup-before-`to_xarray()` in `_raw_data_to_xr()` (Pitfall 2) |
| `dataset/backend.py` | MODIFY | `XrBackend.write()` — default `mode="w"` (Pitfall 1) |
| `base/data.py` | MODIFY | `from_raw_data()` calls new centralized `clean_market_data()` hook |
| `dataset/cleaning.py` | NEW | Shared cleaning module: `dedup_raw_frame()` (tabular, pre-`to_xarray`) + `clean_market_data()` (xr.Dataset, post-conversion: anomaly-flag + schema-validate) |
| `base/acquisition.py` | NEW | `Acquisition(ABC)` — mirrors `Dataset`/`FactorKunQuant` ABC+concrete lifecycle pattern |
| `acquisition/__init__.py` | NEW | Empty, per existing package convention |
| `acquisition/tiingo.py` | NEW | `TiingoAcquisition(Acquisition)` — encapsulated, config-driven, multi-frequency, incremental-refresh Tiingo client |
| `enums/data.py` | MODIFY (optional) | Add `Market`/`Frequency` Literal aliases and/or a `TiingoColumns` constant for the explicit `columns=` list (Pitfall 3) |

</scope>

---

## 1. `base/config.py` — `DatasetConfig` + new `AcquisitionConfig`

**Current state (full file already read):**
```python
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

@dataclass
class DatasetConfig:
    raw_data_dir_path: str
    zarr_file_path: str
    catalog_path: str
    start_date: str | None = None
    end_date: str | None = None
    symbols: tuple | None = None
    kwargs: dict | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)
```

**Pattern to follow:** every config dataclass in this codebase (`DatasetConfig`, `FactorConfig`, `DLConfig`, `MLConfig`) is a flat `@dataclass` with:
- required fields first, optional fields (`= None` or `field(default_factory=...)`) after
- a trailing `name: str | None = None` field, auto-populated by the owning class's `config.setter` (see `base/data.py:Dataset.config` setter — `self._config.name = self.import_path`) — **do not set `name` yourself in factory functions**
- a `to_dict()` method that is just `asdict(self)` — used both for `get_config()` and for JSON checkpoint serialization (`base/model.py:_save_model`)

**Add to `DatasetConfig`** (D-01), placed alongside other identity fields near the top:
```python
market: Literal["us_equity", "crypto_spot"]
frequency: Literal["1d", "1m", "tick"]
```
These are **required, no-default** fields (matching `raw_data_dir_path`/`zarr_file_path`/`catalog_path` which are also required-no-default) — this forces every existing call site to be updated, which is intentional: it surfaces every place a `DatasetConfig` is constructed so none silently defaults to the wrong market. Known call sites to update: `config/__init__.py:spot_kline_config()`, `test.py` (hand-built `DatasetConfig` for `StockDataset` — flagged CLEAN-02/QUAL-02 cleanup, Phase 7, but this phase's field addition will break it if left untouched; add `market`/`frequency` there too since it's a required field, without otherwise refactoring `test.py`'s structure).

**New `AcquisitionConfig` dataclass** — same file, same style, for `TiingoAcquisition`:
```python
@dataclass
class AcquisitionConfig:
    market: Literal["us_equity", "crypto_spot"]
    frequency: Literal["1d", "1m", "tick"]
    raw_data_dir_path: str          # where fetched raw files land (feeds a Dataset's raw_data_dir_path)
    watermark_path: str             # per-symbol last-fetched-date sidecar file location
    symbols: tuple[str, ...]
    start_date: str | None = None   # full-backfill start; None => use watermark/refresh mode
    end_date: str | None = None
    kwargs: dict | None = None

    name: str | None = None

    def to_dict(self):
        return asdict(self)
```
**Do NOT add an `api_key`/`credential` field here** — `AcquisitionConfig.to_dict()` is exactly the kind of object that could get serialized into a checkpoint JSON via the same `get_config()`/`to_dict()` idiom used by `DatasetConfig`/`FactorConfig`. `TiingoAcquisition` must read `TIINGO_API_KEY` from `os.environ` directly inside `__init__`, never store it on this dataclass (this is the exact SEC-01/Phase-1 leak pattern repeating if violated).

---

## 2. `config/__init__.py` — factory function retrofit + new `stock_kline_config()`

**Current state (full file already read).** Every factory follows this shape:
```python
def _data_root() -> Path:
    env_value = os.environ.get("QUANTLAB_DATA_DIR")
    if env_value:
        return Path(env_value)
    return Path(__file__).resolve().parent.parent / "data"


def spot_kline_config(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list | None = None,
    kwargs: dict = None,  # type: ignore
):
    return DatasetConfig(
        raw_data_dir_path=str(
            _data_root() / "downloads" / "spot" / "monthly" / "klines"
        ),
        zarr_file_path=str(_data_root() / "data" / "spot" / "klines.zarr"),
        catalog_path=str(_data_root() / "data" / "catalog"),
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        kwargs=kwargs,
    )
```
Note the existing hardcoded, non-uniform segments: `"downloads"/"spot"/"monthly"/"klines"` vs `"data"/"spot"/"klines.zarr"` — no `market`/`frequency` concept exists yet, exactly the gap D-01/D-02 close.

**Retrofit pattern (D-02 path convention: `data/{market}/{frequency}/{name}.zarr`).** Recommend a small private helper next to `_data_root()`, matching the existing one-helper-per-concern style:
```python
def _market_data_root(market: str, frequency: str) -> Path:
    return _data_root() / "data" / market / frequency


def _market_downloads_root(market: str, frequency: str) -> Path:
    return _data_root() / "downloads" / market / frequency
```
Then `spot_kline_config()` becomes (D-04 — path/config change only, no ingestion logic change):
```python
def spot_kline_config(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list | None = None,
    kwargs: dict = None,  # type: ignore
    market: Literal["crypto_spot"] = "crypto_spot",
    frequency: Literal["1d", "1m", "tick"] = "1d",
):
    return DatasetConfig(
        market=market,
        frequency=frequency,
        raw_data_dir_path=str(_market_downloads_root(market, frequency) / "spot" / "monthly" / "klines"),
        zarr_file_path=str(_market_data_root(market, frequency) / "klines.zarr"),
        catalog_path=str(_data_root() / "data" / "catalog"),
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        kwargs=kwargs,
    )
```
`alpha101_config()`/`alpha158_config()`/`spot_label_config()` all embed a `SpotKlineDataset(spot_kline_config(symbols=symbols))` call — these need no change beyond passing through `market`/`frequency` params if the planner wants them overridable at that level (optional; `spot_kline_config()`'s own defaults already satisfy D-04 without touching these three).

**New `stock_kline_config()` (D-09) — same call shape, add analogous to `spot_kline_config()`:**
```python
def stock_kline_config(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list | None = None,
    kwargs: dict = None,  # type: ignore
    market: Literal["us_equity"] = "us_equity",
    frequency: Literal["1d", "1m", "tick"] = "1d",
):
    return DatasetConfig(
        market=market,
        frequency=frequency,
        raw_data_dir_path=str(_market_downloads_root(market, frequency) / "nasdaq_data"),
        zarr_file_path=str(_market_data_root(market, frequency) / "stock.zarr"),
        catalog_path=str(_data_root() / "data" / "catalog"),
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        kwargs=kwargs,
    )
```
This directly replaces `test.py`'s hand-built `DatasetConfig(raw_data_dir_path=..., zarr_file_path=..., catalog_path="none")` — do not modify `test.py`'s structure beyond what's needed to not break the `market`/`frequency` required-field addition (full `test.py` cleanup is Phase 7 scope per CONTEXT.md).

Consider a matching `stock_acquisition_config()` factory returning an `AcquisitionConfig` for `TiingoAcquisition`, in the same file, same style (no separate `acquisition/config.py` needed — keep all config factories centralized here, matching the existing "one config module" convention).

---

## 3. `dataset/stock.py` / `dataset/spot.py` — dedup-before-`to_xarray()` (Pitfall 2 / D-05)

**Current `StockDataset._raw_data_to_xr()` (full method, already read):**
```python
def _raw_data_to_xr(self) -> xr.Dataset:
    with Timer(f" {self.__class__.__name__}: from pqt"):
        files = get_pqt_files(self.config.raw_data_dir_path)
        stock_dfs = []
        for file in tqdm(files):
            stock_dfs.append(pl.scan_parquet(file))
        data = pl.concat(stock_dfs)
        data = data.filter(
            pl.col("timestamp") >= pl.lit(self.config.start_date).str.to_datetime(),
            pl.col("timestamp") <= pl.lit(self.config.end_date).str.to_datetime(),
        )
        data = data.sort(by=["timestamp", "symbol"])
        data = data.collect().to_pandas().set_index(["timestamp", "symbol"])
        return data.to_xarray()
```

**Current `SpotKlineDataset._raw_data_to_xr()` (full method, already read)** ends the same way:
```python
res = pl.concat(dfs)
res = res.rename({"Open time": "timestamp"})
res = res.sort(by=["timestamp", "symbol"])
res = res.collect().to_pandas().set_index(["timestamp", "symbol"])
return res.to_xarray()
```

Both already call `.sort(by=["timestamp", "symbol"])` on a `pl.LazyFrame` right before `.collect().to_pandas().set_index([...])`. The fix is a single inserted line in each, calling the new `dataset/cleaning.py:dedup_raw_frame()` on the **polars LazyFrame/DataFrame, before `set_index`**, matching RESEARCH.md Pattern 2 exactly:

```python
# StockDataset._raw_data_to_xr — insert after the existing .sort(...) line:
data = data.sort(by=["timestamp", "symbol"])
data = dedup_raw_frame(data, keep="last")          # NEW — D-05, must run before to_xarray()
data = data.collect().to_pandas().set_index(["timestamp", "symbol"])
return data.to_xarray()
```
```python
# SpotKlineDataset._raw_data_to_xr — same insertion point:
res = res.sort(by=["timestamp", "symbol"])
res = dedup_raw_frame(res, keep="last")            # NEW — D-05
res = res.collect().to_pandas().set_index(["timestamp", "symbol"])
return res.to_xarray()
```
Both files already `import polars as pl`; add `from dataset.cleaning import dedup_raw_frame` to each. **Do not** put dedup in `base/data.py:from_raw_data()` — by the time that hook runs, `_raw_data_to_xr()` has already returned an `xr.Dataset` and the crash (`ValueError: cannot convert a DataFrame with a non-unique MultiIndex into xarray`) has already happened inside the subclass method.

---

## 4. `dataset/backend.py:XrBackend.write()` — default `mode="w"` (Pitfall 1)

**Current (full method, already read):**
```python
def write(self, path: str, **kwargs) -> Self:
    if not Path(path).exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    self.data.to_zarr(path, **kwargs)
    return self
```
**Fix — minimal, one line, matches RESEARCH.md's Code Examples exactly:**
```python
def write(self, path: str, **kwargs) -> Self:
    if not Path(path).exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("mode", "w")   # NEW — unblock re-run/incremental-refresh (Pitfall 1)
    self.data.to_zarr(path, **kwargs)
    return self
```
Note `PlBackend.write()` in the same file has no such issue (`self.data.collect().write_parquet(path, **kwargs)` — parquet overwrite is not `FileExistsError`-prone the same way). Only `XrBackend` needs this change. This also means `Dataset.save()` (`base/data.py`) and any caller passing an explicit `mode=` kwarg still work unchanged (`kwargs.setdefault` only fills in when absent) — no call-site changes required elsewhere, including `FactorKunQuant.save()` in `base/factor.py` which already explicitly passes `mode: Literal["a", "w"] = "a"` and would be unaffected.

---

## 5. `base/data.py:Dataset.from_raw_data()` — centralized cleaning hook

**Current (full method, already read):**
```python
def from_raw_data(self) -> Self:
    data = self._raw_data_to_xr()
    self.data_backend.to_internal(data)  # type: ignore
    return self
```
**Modification (D-05..D-08, Pattern 1 — single insertion point covers every current and future `Dataset` subclass, satisfying Success Criterion 4 without touching `StockDataset`/`SpotKlineDataset`'s `to_internal` call sites):**
```python
def from_raw_data(self) -> Self:
    data = self._raw_data_to_xr()
    data = clean_market_data(data)       # NEW — D-06/D-07/D-08 (NaN-gap check, anomaly flag, schema validate)
    self.data_backend.to_internal(data)  # type: ignore
    return self
```
Add `from dataset.cleaning import clean_market_data` to `base/data.py`'s imports. Note this is a **cross-package import from `base` into `dataset`** — check existing import direction: `base/data.py` already imports `from dataset.backend import XrBackend`, so `base` → `dataset` imports are an established, non-circular direction in this codebase (confirmed: `dataset/backend.py`, `dataset/stock.py`, `dataset/spot.py` do not import anything from `base.data`, only `base.config`). This new import does not introduce a cycle.

---

## 6. `dataset/cleaning.py` (NEW) — shared cleaning module

**Closest analog for module style:** `my_ops/preprocess.py` (only existing "cleaning-adjacent" module) — but that module is KunQuant-graph-op style (`WindowedCompositiveOp` subclasses using `Builder`/`decompose()`), which is the wrong pattern here since raw-market-data cleaning runs on plain tabular/xarray data, not inside a compiled factor graph. The correct style reference is plain functions operating on `pl.LazyFrame`/`xr.Dataset`, matching how `utils/file.py` (`get_csv_files`, `file_date_filter`) is written: small, single-purpose, type-hinted, no classes.

**Two entry points, per RESEARCH.md Pattern 2/3 split:**

```python
"""Shared raw-market-data cleaning, reused by every Dataset subclass.

Two entry points, called at two different pipeline stages:
- dedup_raw_frame(): tabular (polars), called by each subclass's
  _raw_data_to_xr() BEFORE the final .to_xarray() conversion — required
  because a non-unique (timestamp, symbol) index crashes to_xarray().
- clean_market_data(): xr.Dataset, called once centrally in
  base/data.py:Dataset.from_raw_data() AFTER conversion — anomaly-flagging
  and schema validation only. Missing-timestamp NaN-gap behavior requires
  no code here: pandas.to_xarray() already produces the full
  (timestamp, symbol) cartesian product with NaN for absent combinations,
  given a unique index (verified empirically, see 02-RESEARCH.md Pattern 3).
  Do not add forward-fill/interpolate anywhere in this module (D-06).
"""
from typing import Literal

import numpy as np
import polars as pl
import xarray as xr
from loguru import logger

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")  # adapt per-source if needed


def dedup_raw_frame(
    data: pl.LazyFrame, keep: Literal["first", "last"] = "last"
) -> pl.LazyFrame:
    """Deterministic dedup on (timestamp, symbol) — D-05.

    keep="last": later-arriving rows (e.g. a corrected re-drop of a vendor's
    monthly file) win over earlier ones. Must run before .to_xarray();
    a non-unique MultiIndex raises ValueError there (see 02-RESEARCH.md
    Common Pitfalls, Pitfall 2).
    """
    return data.unique(subset=["timestamp", "symbol"], keep=keep)


def flag_anomalies(data: xr.Dataset) -> xr.Dataset:
    """Flag (not delete/correct) zero/negative price or extreme jumps — D-07.

    Adds a boolean data_var, e.g. `anomaly_flag`, alongside existing
    variables. Logs a warning summary via loguru, matching the existing
    Dataset/FactorKunQuant logging style (base/data.py already imports
    `from loguru import logger`).
    """
    ...


def validate_schema(data: xr.Dataset, required_columns: tuple[str, ...] = REQUIRED_COLUMNS) -> xr.Dataset:
    """Basic schema/type/non-null validation — D-08.

    Raises (not silently passes) if a required column is missing or has
    an unexpected dtype; logs a warning (does not raise) for unexpected
    nulls in non-key columns, consistent with D-07's flag-don't-delete
    philosophy.
    """
    ...


def clean_market_data(data: xr.Dataset) -> xr.Dataset:
    """Single entry point called from base/data.py:Dataset.from_raw_data().

    NaN-gap behavior (D-06) requires no logic here — already guaranteed by
    the dedup step + pandas.to_xarray()'s cartesian-product behavior
    upstream in each subclass's _raw_data_to_xr(). This function only
    layers anomaly-flagging (D-07) and schema validation (D-08) on top.
    """
    data = validate_schema(data)
    data = flag_anomalies(data)
    return data
```

**Do not** implement any reindex-to-full-calendar-grid or forward-fill logic here — per RESEARCH.md's "Don't Hand-Roll" table, `pandas.DataFrame.set_index([...]).to_xarray()` already does this for free once inputs are deduplicated.

---

## 7. `base/acquisition.py` (NEW) — `Acquisition(ABC)`

**Closest analog:** `base/factor.py:FactorKunQuant` — an ABC with a `config` property/setter that normalizes the config on assignment (auto-filling dates, resolving names), plus a small set of `abstractmethod`s that concrete subclasses implement, plus non-abstract orchestration methods (`cal()`/`save()`/`read()` in `FactorKunQuant`; here, `download()`/`refresh()`). Also draw the `config` setter idiom directly from `base/data.py:Dataset`:

```python
# base/data.py — the config setter idiom to mirror (already read in full above)
@property
def config(self) -> DatasetConfig:
    return self._config

@config.setter
def config(self, config: DatasetConfig):
    self._config = config
    self._config.name = self.import_path
    if self._config.start_date is None:
        self._config.start_date = Date.START_DATE
    if self._config.end_date is None:
        self._config.end_date = Date.END_DATE
    if self._config.symbols is not None:
        self._reset_symbols()
```

**New `base/acquisition.py` skeleton:**
```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from base.config import AcquisitionConfig
from enums.constant import Date


class Acquisition(ABC):
    """Remote-fetch layer, decoupled from Dataset (which only converts
    already-local raw files — see base/data.py:Dataset and
    dataset/stock.py, dataset/spot.py: no network calls anywhere in the
    Dataset hierarchy today; do not add download()/refresh() to Dataset
    itself, per 02-RESEARCH.md Anti-Patterns).
    """

    def __init__(self, config: AcquisitionConfig):
        self.config = config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> AcquisitionConfig:
        return self._config

    @config.setter
    def config(self, config: AcquisitionConfig):
        self._config = config
        self._config.name = self.import_path
        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def _watermark_path(self, symbol: str) -> Path:
        return Path(self.config.watermark_path) / f"{symbol}.json"

    def _read_watermark(self, symbol: str) -> str | None:
        """Returns the last-successfully-fetched date for `symbol`, or
        None if never fetched. Small per-symbol JSON sidecar file — kept
        decoupled from the Dataset/Zarr layer per 02-RESEARCH.md
        Anti-Patterns (do not derive from Zarr's max timestamp)."""
        ...

    def _write_watermark(self, symbol: str, last_date: str) -> None:
        ...

    def download(self, symbols: list[str] | None = None) -> Self:
        """Full-history backfill for the given symbols (or self.config.symbols)."""
        symbols = symbols or list(self.config.symbols)
        for symbol in symbols:
            self._fetch_and_write(symbol, start_date=self.config.start_date, end_date=self.config.end_date)
            self._write_watermark(symbol, self.config.end_date)
        return self

    def refresh(self, symbols: list[str] | None = None) -> Self:
        """Incremental refresh — fetch only new data since each symbol's watermark."""
        symbols = symbols or list(self.config.symbols)
        for symbol in symbols:
            start = self._read_watermark(symbol) or self.config.start_date
            self._fetch_and_write(symbol, start_date=start, end_date=self.config.end_date)
            self._write_watermark(symbol, self.config.end_date)
        return self

    @abstractmethod
    def _fetch_and_write(self, symbol: str, start_date: str, end_date: str) -> None:
        """Fetch remote data for one symbol/date-range and write raw local
        files under self.config.raw_data_dir_path, in the schema the
        matching Dataset subclass's _raw_data_to_xr() expects."""
        ...
```

Single-responsibility methods (`_fetch_and_write`, `_read_watermark`, `_write_watermark`) mirror the existing convention of small private helper methods prefixed `_` (see `Dataset._filter`, `Dataset._get_symbols`, `FactorKunQuant._auto_filter`, `FactorKunQuant._reset_dataset_config`).

---

## 8. `acquisition/tiingo.py` (NEW) — `TiingoAcquisition(Acquisition)`

**Reference for WHAT (per D-10 — API call shape only, not structure), from `scripts/download_stock_data_from_tiingo.py` (full script already read):**
```python
# Existing script's TiingoClient construction + call shape — reuse the
# call shape, not the flat/procedural structure:
config = {}
config["session"] = True
config["api_key"] = os.environ["TIINGO_API_KEY"]
client = TiingoClient(config)

data = pl.DataFrame(
    client.get_ticker_price(
        stock,
        fmt="json",
        startDate=start_date,
        endDate=end_date,
        frequency="daily",
    )
)
data = data.with_columns(pl.col("date").cast(pl.Datetime))
data = data.rename({"date": "timestamp"})
data = data.with_columns(pl.lit(stock).alias("symbol"))
```
Note the script's `frequency="daily"` is a literal hardcoded inside `download_stock()`'s body — D-10 explicitly requires this be a parameter on the new component instead (frequency comes from `self.config.frequency`, mapped e.g. `"1d" -> "daily"`).

**Pitfall 3 (from RESEARCH.md) — always pass explicit `columns=`, do not rely on the client's undocumented default:**
```python
columns="open,high,low,close,volume,adjOpen,adjHigh,adjLow,adjClose,adjVolume,divCash,splitFactor"
```
This is required because `dataset/stock.py:StockDataset._to_kunquant()` unconditionally does:
```python
data = data.drop_vars(["open", "high", "low", "close", "volume"])
data = data.rename({"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close", "adjVolume": "volume"})
```
— if `adjOpen`/etc. are absent from the fetched JSON, this raises a `KeyError`/rename failure downstream, not at acquisition time. Put the explicit `columns=` string as a module-level constant in `acquisition/tiingo.py` (or `enums/data.py` alongside `BinanceCSVHeaders`, matching that existing "headers/columns as a dataclass constant" convention) so it's visible and auditable in one place.

**Skeleton, following `Acquisition(ABC)` contract above:**
```python
import os
from pathlib import Path

import polars as pl
from tiingo import TiingoClient

from base.acquisition import Acquisition
from base.config import AcquisitionConfig

TIINGO_COLUMNS = (
    "open,high,low,close,volume,"
    "adjOpen,adjHigh,adjLow,adjClose,adjVolume,divCash,splitFactor"
)

_FREQUENCY_MAP = {"1d": "daily"}  # extend when intraday frequencies are added


class TiingoAcquisition(Acquisition):
    def __init__(self, config: AcquisitionConfig):
        super().__init__(config)
        if not os.environ.get("TIINGO_API_KEY"):
            raise RuntimeError(
                "TIINGO_API_KEY environment variable is not set. Export it "
                "before running acquisition (see Tiingo dashboard for your key)."
            )
        self._client = TiingoClient({
            "session": True,
            "api_key": os.environ["TIINGO_API_KEY"],
        })

    def _fetch_and_write(self, symbol: str, start_date: str, end_date: str) -> None:
        frequency = _FREQUENCY_MAP[self.config.frequency]
        raw = self._client.get_ticker_price(
            symbol,
            fmt="json",
            startDate=start_date,
            endDate=end_date,
            frequency=frequency,
            columns=TIINGO_COLUMNS,
        )
        data = pl.DataFrame(raw)
        if data.is_empty():
            return
        data = data.with_columns(pl.col("date").cast(pl.Datetime))
        data = data.rename({"date": "timestamp"})
        data = data.with_columns(pl.lit(symbol).alias("symbol"))

        out_dir = Path(self.config.raw_data_dir_path) / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        data.write_parquet(out_dir / "data.pqt")
```
This raw-parquet-file schema (`{raw_data_dir_path}/{symbol}/data.pqt`) intentionally matches what `dataset/stock.py:StockDataset._raw_data_to_xr()` already reads via `get_pqt_files(self.config.raw_data_dir_path)` (which does `Path(dir_path).rglob("*.pqt")` — recursive, so the per-symbol subdirectory layout works unchanged). **No change to `StockDataset._raw_data_to_xr()`'s file-discovery logic is needed** — only the dedup insertion from Section 3 above.

---

## 9. `enums/data.py` — optional additions

**Current (full file already read):**
```python
from dataclasses import dataclass

@dataclass
class BinanceCSVHeaders:
    SPOT = [...]
```
If the planner wants `Market`/`Frequency` as named type aliases rather than bare `Literal[...]` repeated in three files (`base/config.py`, `config/__init__.py` factory signatures, `base/acquisition.py`), add here matching the existing `@dataclass`-of-constants convention, or a plain `type` alias:
```python
from typing import Literal

Market = Literal["us_equity", "crypto_spot"]
Frequency = Literal["1d", "1m", "tick"]
```
This is optional (RESEARCH.md Assumptions Log A2 flags exact literal tokens as planner discretion) — bare `Literal[...]` repeated inline is also consistent with existing style (`FactorConfig.mode: Literal["stream", "batch"]` is inline, not aliased). Prefer the alias only if `market`/`frequency` end up referenced in 3+ files, to avoid drift.

---

<import_graph>
## Import Direction (confirmed, non-circular)

```
base/config.py        <- no project-internal imports (only TYPE_CHECKING stdlib)
base/backend.py        <- no project-internal imports
base/data.py           <- base.config, dataset.backend, enums.constant, utils.timer   (existing)
                        <- dataset.cleaning                                            (NEW, this phase)
base/acquisition.py    <- base.config, enums.constant                                  (NEW, this phase)
dataset/backend.py     <- base.backend                                                 (existing)
dataset/cleaning.py    <- (stdlib/polars/xarray/loguru only, no project imports)        (NEW, this phase)
dataset/stock.py       <- base.config, base.data, enums.data, utils.file, utils.timer  (existing)
                        <- dataset.cleaning                                             (NEW, this phase)
dataset/spot.py        <- base.config, base.data, enums.data, utils.file, utils.nautilus, utils.timer (existing)
                        <- dataset.cleaning                                             (NEW, this phase)
acquisition/tiingo.py  <- base.acquisition, base.config, tiingo (3rd-party)             (NEW, this phase)
config/__init__.py     <- base.config, dataset.backend, dataset.spot                    (existing)
                        <- dataset.stock, acquisition.tiingo (if adding config factories for both) (NEW, this phase)
```
No cycle introduced: `dataset/cleaning.py` has zero project-internal imports (leaf module), and `base/data.py` importing from `dataset.cleaning` mirrors the already-established `base` → `dataset` direction (`base/data.py` already imports `dataset.backend`).

</import_graph>

<naming_and_style_notes>
## Naming/Style Conventions to Follow

- **No `__init__.py` re-exports** — every package's `__init__.py` in this repo is empty; `acquisition/__init__.py` must also be empty. Callers use full dotted paths: `from acquisition.tiingo import TiingoAcquisition`, never `from acquisition import TiingoAcquisition`.
- **Private helper methods prefixed `_`** — e.g. `_fetch_and_write`, `_read_watermark`, `_data_root`, `_market_data_root`.
- **`Timer` context manager for expensive I/O** — every existing `_raw_data_to_xr()` wraps its body in `with Timer(f"{self.__class__.__name__}: <verb>"):` (`from utils.timer import Timer`). Apply the same to `TiingoAcquisition.download()`/`refresh()` for consistency, though not strictly required by any decision.
- **`loguru.logger` for warnings, not `print`** — `dataset/spot.py:_xr_to_bars` is the one place still using bare `print(...)` for an error; do not copy that — use `from loguru import logger` (already the dominant pattern in `base/data.py`, `base/factor.py`, `utils/timer.py`, `utils/nautilus.py`).
- **Type hints on every public method signature**, `Self` return type for chainable builder-style methods (`.download().save()` style), matching `Dataset.read()`/`Dataset.save()`/`FactorKunQuant.read()`/`FactorKunQuant.save()` all returning `Self`.
</naming_and_style_notes>

---

*Pattern extraction for: 02-multi-market-data-foundation*
*Based on direct reads of: `base/config.py`, `base/data.py`, `base/backend.py`, `base/factor.py`, `dataset/backend.py`, `dataset/stock.py`, `dataset/spot.py`, `config/__init__.py`, `my_ops/preprocess.py`, `scripts/download_stock_data_from_tiingo.py`, `enums/data.py`, `enums/constant.py`, `utils/file.py`, `utils/module.py`, `base/model.py` (lines 190-220), `test.py`, `pyproject.toml`*
