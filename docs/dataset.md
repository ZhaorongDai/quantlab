# Dataset

English | [简体中文](zh-CN/dataset.md)

A dataset converts the raw files a vendor delivers into the panel that every later stage of quantlab consumes: an `xarray.Dataset` indexed by `(timestamp, symbol)`, stored as Zarr. Each market or vendor has one subclass of `BaseDataset` or `MarketDataset`. Factors, models and backtests read the panel and never see the raw format.

## Prerequisites

The raw files come from the acquisition layer (see the acquisition guide). The sessions below run on a small synthetic raw tree in a temporary directory, so no network access or credentials are needed. The tree follows the layout `StockDataset` reads: one Parquet shard per month under a directory named after the vendor, with a `vendor` column in every row.

```python
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

logger.remove()  # quantlab logs through loguru at INFO level; silence it here

root = Path(tempfile.mkdtemp())
raw = root / "downloads/us_equity/1d/us_all/tiingo"
for month, days in {"2024-01": [2, 3, 4], "2024-02": [1, 2]}.items():
    rows = [
        dict(timestamp=datetime.fromisoformat(f"{month}-{d:02d}"), symbol=sym,
             open=px + d, high=px + d + 1, low=px + d - 1, close=px + d,
             volume=1_000.0)
        for d in days for sym, px in [("AAPL", 100.0), ("MSFT", 300.0)]
    ]
    part = raw / f"month={month}"
    part.mkdir(parents=True)
    pl.DataFrame(rows).with_columns(vendor=pl.lit("tiingo")).write_parquet(part / "part-0.pqt")
```

## The basics

### The panel

Every dataset produces the same shape of object: an `xarray.Dataset` whose data variables all lie on the two dimensions `timestamp` and `symbol`. The panel is dense. When a symbol has no bar at a timestamp, the cell holds NaN in every variable rather than the row being absent. Variables keep the names the vendor uses (`open`, `close`, `adjClose`, `Volume`, and so on), and cleaning adds one boolean variable, `anomaly_flag`.

### The config

A dataset is built from a config dataclass. `BaseDatasetConfig` carries what every dataset needs: `zarr_file_path`, `start_date`, `end_date`, `symbols` and a free-form `kwargs` dictionary. `DatasetConfig` adds the fields of a market-data panel: `raw_data_dir_path`, `market`, `frequency` and `vendor`. The dataset class never branches on `market` or `frequency`; they are labels used by the vendor registry to pick a converter.

```python
>>> import dataclasses
>>> from quantlab.base.config import DatasetConfig
>>> from quantlab.dataset.stock import StockDataset
>>> config = DatasetConfig(
...     raw_data_dir_path=str(raw),
...     zarr_file_path=str(root / "data/us_all.zarr"),
...     market="us_equity",
...     frequency="1d",
...     vendor="tiingo",
...     start_date="2024-01-01",
...     end_date="2024-02-29",
... )
>>> ds = StockDataset(config)
>>> ds.config.name
'quantlab.dataset.stock.StockDataset'
```

Assigning a config fills in `name` with the dotted import path of the class, which is how a saved config is turned back into an object. A missing `start_date` or `end_date` becomes `1900-01-01` or `2100-01-01`, so a date filter always has two ends. Both dates must be ISO `YYYY-MM-DD` strings, because every date comparison in the pipeline is a string comparison.

```python
>>> open_ended = dataclasses.replace(config, start_date=None, end_date=None)
>>> d = StockDataset(open_ended)
>>> d.config.start_date, d.config.end_date
('1900-01-01', '2100-01-01')
>>> bad = dataclasses.replace(config, end_date="2024-2-29")
>>> try:
...     StockDataset(bad)
... except ValueError as exc:
...     print(exc)
...
StockDataset: end_date must be an ISO YYYY-MM-DD date string, got '2024-2-29'. Dates are compared lexicographically throughout this pipeline, so a non-ISO value compares wrong rather than failing to match.
```

### Convert, save and read

Three methods cover the storage lifecycle. `from_raw_data()` reads the raw files for the configured range, runs the dataset's cleaning step and holds the result in memory. `save()` writes it to `zarr_file_path`. `read()` opens the Zarr store later and narrows it to the configured dates and symbols. Each returns the dataset, so the calls chain, and `get_xarray_dataset()` returns the panel.

```python
>>> ds = ds.from_raw_data()
>>> ds.get_xarray_dataset()
<xarray.Dataset> Size: 466B
Dimensions:       (timestamp: 5, symbol: 2)
Coordinates:
  * timestamp     (timestamp) datetime64[us] 40B 2024-01-02 ... 2024-02-02
  * symbol        (symbol) object 16B 'AAPL' 'MSFT'
Data variables:
    open          (timestamp, symbol) float64 80B 102.0 302.0 ... 102.0 302.0
    high          (timestamp, symbol) float64 80B 103.0 303.0 ... 103.0 303.0
    low           (timestamp, symbol) float64 80B 101.0 301.0 ... 101.0 301.0
    close         (timestamp, symbol) float64 80B 102.0 302.0 ... 102.0 302.0
    volume        (timestamp, symbol) float64 80B 1e+03 1e+03 ... 1e+03 1e+03
    anomaly_flag  (timestamp, symbol) bool 10B False False False ... False False
>>> ds.save()
>>> panel = StockDataset(config).read().get_xarray_dataset()
>>> panel["close"].to_pandas()
symbol       AAPL   MSFT
timestamp               
2024-01-02  102.0  302.0
2024-01-03  103.0  303.0
2024-01-04  104.0  304.0
2024-02-01  101.0  301.0
2024-02-02  102.0  302.0
```

Only weekdays that exist in the raw files appear on the time axis; the window `2024-01-01` to `2024-02-29` does not create rows for days without data.

### Looking at a stored panel

A dataset that has been read exposes a few cheap properties. `time_interval` is the most common gap between timestamps, so a weekend or a holiday does not change it. `get_lazyframe()` returns the same data as a long-format polars `LazyFrame`, and `head(n)` opens the store by path and returns at most `n` rows without touching the loaded panel.

```python
>>> ds = StockDataset(config).read()
>>> ds.symbols, ds.num_symbols
(['AAPL', 'MSFT'], 2)
>>> ds.time_interval
np.timedelta64(86400000000000,'ns')
>>> ds.get_lazyframe().collect().shape
(10, 8)
>>> ds.head(2).collect().columns
['timestamp', 'symbol', 'anomaly_flag', 'close', 'high', 'low', 'open', 'volume']
```

## Common tasks

### Restrict the dates or symbols on read

`read()` applies `start_date`, `end_date` and, when it is set, `symbols`. A different config over the same store gives a different view.

```python
>>> feb = dataclasses.replace(config, start_date="2024-02-01", symbols=("MSFT",))
>>> StockDataset(feb).read().get_xarray_dataset()["close"].to_pandas()
symbol       MSFT
timestamp        
2024-02-01  301.0
2024-02-02  302.0
```

### Read the anomaly flags

Cleaning runs inside `from_raw_data()`. It checks that the required columns exist, reports nulls, and adds `anomaly_flag`. A cell is flagged when a price is zero or negative, or when `close` moves by more than 50 percent from a positive previous close. Values are never changed, filled or dropped; the flag only marks them. The functions can be called on any panel.

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.dataset._support.cleaning import flag_anomalies, validate_schema
>>> close = np.array([[10.0, 5.0], [10.5, 0.0], [20.0, 5.2], [20.5, 5.3]])
>>> panel = xr.Dataset(
...     {name: (("timestamp", "symbol"), close) for name in ("open", "high", "low", "close")}
...     | {"volume": (("timestamp", "symbol"), np.full((4, 2), 1000.0))},
...     coords={"timestamp": pd.date_range("2024-01-02", periods=4), "symbol": ["AAA", "BBB"]},
... )
>>> flagged = flag_anomalies(panel)
>>> flagged["anomaly_flag"].to_pandas()
symbol        AAA    BBB
timestamp               
2024-01-02  False  False
2024-01-03  False   True
2024-01-04   True  False
2024-01-05  False  False
>>> bool(flagged["close"].equals(panel["close"]))
True
```

`BBB` prints a zero close on 2024-01-03, and `AAA` jumps from 10.5 to 20.0 on 2024-01-04. A missing required column is the one cleaning failure that raises:

```python
>>> validate_schema(panel.drop_vars("volume"))
Traceback (most recent call last):
    ...
ValueError: validate_schema: required column(s) missing from dataset: ['volume']
```

Duplicate `(timestamp, symbol)` rows must be removed before a frame is converted to xarray. `dedup_raw_frame(frame, keep="last")` does this on a polars `LazyFrame` and keeps the last row by default, because a later vendor file more often carries a correction; `keep="first"` keeps the earlier one.

### Export arrays for KunQuant

`MarketDataset.to_kunquant` reads the store and returns a dictionary of contiguous `[time, symbol]` float32 arrays, plus the symbol and timestamp axes. The factor layer calls it; it can also be called directly.

```python
>>> inputs, symbols, timestamps = ds.to_kunquant(("open", "close"))
>>> inputs["close"].shape, inputs["close"].dtype
((5, 2), dtype('float32'))
>>> symbols.tolist()
['AAPL', 'MSFT']
```

### Convert one window at a time

For a long history, `from_raw_data_chunked()` converts a month, quarter or year at a time and appends each to the store, and `update()` continues a store that already exists. The chunking guide covers both in detail.

```python
>>> monthly = dataclasses.replace(config, zarr_file_path=str(root / "data/monthly.zarr"))
>>> ds = StockDataset(monthly).from_raw_data_chunked(granularity="month")
>>> result = ds.last_chunk_result
>>> result.windows_planned, result.windows_written, result.rows_written
(2, 2, 5)
```

### Resample onto coarser bars

`resample(freq, how)` returns a copy of the dataset whose panel is aggregated onto coarser bars: minute bars into daily bars, for example. `freq` is one of `1s`, `5s`, `10s`, `15s`, `30s`, `1m`, `5m`, `10m`, `15m`, `30m`, `1h` and `1d`, and must be coarser than the store's own bars. `how` names one method per variable, from `first`, `last`, `max`, `min`, `sum`, `mean` and `count`, or one method as a string for every variable. NaN cells are skipped. The copy shares no memory with the source, and the source is not changed.

The session below writes a two-day minute store and reads it through `SpotKlineDataset`.

```python
>>> minutes = pd.DatetimeIndex(np.concatenate([
...     pd.date_range(f"2024-01-0{d} 00:00", periods=4, freq="min").values for d in (2, 3)
... ]))
>>> close = np.arange(1.0, 9.0)[:, None] * np.array([[1.0, 10.0]])
>>> xr.Dataset(
...     {"Open": (["timestamp", "symbol"], close - 0.5),
...      "Close": (["timestamp", "symbol"], close),
...      "Volume": (["timestamp", "symbol"], np.ones((8, 2)))},
...     coords={"timestamp": minutes, "symbol": ["AAAUSDT", "BBBUSDT"]},
... ).to_zarr("data/klines.zarr", mode="w")
>>> config = DatasetConfig(raw_data_dir_path="downloads/spot", zarr_file_path="data/klines.zarr",
...                        market="crypto_spot", frequency="1m")
>>> minute = SpotKlineDataset(config).read()
>>> daily = minute.resample("1d", {"Open": "first", "Close": "last", "Volume": "sum"})
>>> daily.get_xarray_dataset()["Close"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0     40.0
2024-01-03      8.0     80.0
>>> daily.get_xarray_dataset()["Volume"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0      4.0
2024-01-03      4.0      4.0
>>> daily.time_interval, minute.time_interval
(np.timedelta64(86400000000000,'ns'), np.timedelta64(60000000000,'ns'))
>>> minute.get_xarray_dataset().sizes["timestamp"], minute.config.resample_freq
(8, None)
```

The copy's config records the request in `resample_freq` and `resample_how`, so it round-trips through `get_config()` and `load_dataset_from_config`. A dataset built with those fields set resamples on `read()`. `save()` writes the resampled panel to `store_path`, a store beside the source with `_resample_<freq>` in its name, and a later `read()` with the same fields opens that store instead of resampling again.

```python
>>> daily.config.resample_freq, daily.config.resample_how
('1d', {'Open': 'first', 'Close': 'last', 'Volume': 'sum'})
>>> daily.store_path
'data/klines_resample_1d.zarr'
>>> daily.save()
>>> sorted(p.name for p in Path("data").iterdir())
['klines.zarr', 'klines_resample_1d.zarr']
>>> reader = SpotKlineDataset(dataclasses.replace(
...     config, resample_freq="1d", resample_how={"Open": "first", "Close": "last", "Volume": "sum"}))
>>> reader.read().get_xarray_dataset()["Close"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0     40.0
2024-01-03      8.0     80.0
```

Bars are cut on the UTC clock by default, labelled at their start, which suits bars stamped at their open time. A dataset whose bars follow trading sessions overrides `_resample_labels`; `NbboPanelDataset` cuts by NYSE session, so `"1d"` labels each session with its date at midnight and lines up with daily stores.

## Extending

### A new market source

A new source needs one subclass of `MarketDataset` and a config. Three methods are required. `_raw_data_to_xr` returns the panel for the whole configured range, deduplicated and unique on `(timestamp, symbol)`. `_raw_data_to_xr_window` returns one date window, reindexed onto `symbols` when they are given; the simplest form slices the whole-range result. `_to_kunquant` maps the panel onto arrays. The example reads one CSV per symbol and is saved as `csv_daily.py`.

```python
# csv_daily.py
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.base.data import MarketDataset


class CsvDailyDataset(MarketDataset):
    """One CSV file per symbol with columns date,open,high,low,close,volume."""

    def _raw_data_to_xr(self) -> xr.Dataset:
        frames = []
        for path in sorted(Path(self.config.raw_data_dir_path).glob("*.csv")):
            df = pd.read_csv(path, parse_dates=["date"])
            frames.append(df.rename(columns={"date": "timestamp"}).assign(symbol=path.stem))
        df = pd.concat(frames).drop_duplicates(["timestamp", "symbol"], keep="last")
        return df.set_index(["timestamp", "symbol"]).sort_index().to_xarray()

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None) -> xr.Dataset:
        data = self._raw_data_to_xr().sel(timestamp=slice(start_date, end_date))
        return data if symbols is None else data.reindex(symbol=list(symbols))

    def _to_kunquant(self, data, data_columns):
        data = data.sortby(["timestamp", "symbol"])
        inputs = {c: np.ascontiguousarray(data[c].to_numpy().astype(np.float32))
                  for c in data_columns}
        return inputs, data["symbol"].values, data["timestamp"].values
```

`to_xarray()` builds the dense grid, so `BBB` below gets NaN on the day it has no file row. The subclass needs no other change to work with the rest of the pipeline.

```python
>>> csv_raw = root / "csv"
>>> csv_raw.mkdir()
>>> for sym, base, skip in [("AAA", 10.0, 0), ("BBB", 20.0, 1)]:
...     dates = pd.bdate_range("2024-01-02", periods=5)[skip:]
...     px = base + np.arange(len(dates))
...     pd.DataFrame({"date": dates, "open": px, "high": px + 1, "low": px - 1,
...                   "close": px, "volume": 1000.0}).to_csv(csv_raw / f"{sym}.csv", index=False)
...
>>> from csv_daily import CsvDailyDataset
>>> csv_config = DatasetConfig(
...     raw_data_dir_path=str(csv_raw),
...     zarr_file_path=str(root / "csv_daily.zarr"),
...     market="us_equity",
...     frequency="1d",
... )
>>> CsvDailyDataset(csv_config).from_raw_data().save()
>>> ds = CsvDailyDataset(csv_config).read()
>>> ds.get_xarray_dataset()["close"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-02  10.0   NaN
2024-01-03  11.0  20.0
2024-01-04  12.0  21.0
2024-01-05  13.0  22.0
2024-01-08  14.0  23.0
>>> inputs, symbols, timestamps = ds.to_kunquant(("close",))
>>> inputs["close"].shape, symbols.tolist()
((5, 2), ['AAA', 'BBB'])
```

A subclass that leaves out a required method cannot be constructed:

```python
>>> from quantlab.base.data import BaseDataset
>>> class Incomplete(BaseDataset):
...     pass
...
>>> Incomplete(csv_config)
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Incomplete without an implementation for abstract method '_raw_data_to_xr'
```

For a raw source that can filter by date before it loads, implement `_raw_data_to_xr_window` to read only that window. `StockDataset` does so by pruning Parquet partitions, which bounds memory by the window. The slicing form above bounds only the write.

### A dataset that is not OHLCV

A panel without price columns subclasses `BaseDataset` directly, uses `BaseDatasetConfig`, and overrides `_clean`. The default `_clean` requires OHLCV columns, so a boolean membership panel is validated with its own function instead. `clean_membership_panel` checks the dtype, dimensions and time order and returns the panel unchanged.

```python
>>> from quantlab.base.config import BaseDatasetConfig
>>> from quantlab.base.data import BaseDataset
>>> from quantlab.dataset._support.cleaning import clean_membership_panel
>>> class InIndexDataset(BaseDataset):
...     def _raw_data_to_xr(self) -> xr.Dataset:
...         days = pd.date_range("2024-01-02", periods=3)
...         member = np.array([[True, False], [True, True], [True, True]])
...         return xr.Dataset({"is_member": (("timestamp", "symbol"), member)},
...                           coords={"timestamp": days, "symbol": ["AAA", "BBB"]})
...     def _clean(self, data: xr.Dataset) -> xr.Dataset:
...         return clean_membership_panel(data)
...
>>> member_config = BaseDatasetConfig(zarr_file_path=str(root / "member.zarr"))
>>> InIndexDataset(member_config).from_raw_data().save()
>>> InIndexDataset(member_config).read().get_xarray_dataset()["is_member"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-02  True  False
2024-01-03  True   True
2024-01-04  True   True
```

## Notes

Cleaning belongs to `from_raw_data()`. `read()` only opens the store and narrows it, so a store written earlier is returned as it was saved.

`save()` narrows the panel to the config window and replaces the whole store directory. Zarr may print a `ZarrUserWarning` about consolidated metadata on write; it is harmless.

`read()` narrows the loaded panel in place and does nothing if the dataset already holds data. Changing `dataset.config` to a wider window and calling `read()` again keeps the narrow panel. Pass `overwrite=True` to reload the store from disk.

```python
>>> ds = StockDataset(dataclasses.replace(config, end_date="2024-01-03")).read()
>>> ds.config = dataclasses.replace(config, end_date="2024-02-29")
>>> ds.read().get_xarray_dataset().sizes["timestamp"]
2
>>> ds.read(overwrite=True).get_xarray_dataset().sizes["timestamp"]
5
```

Cleaning never fills or repairs a value. Chunked conversion cleans one window at a time, so a price jump that straddles a window boundary is not flagged.

Reading a store that does not exist raises `FileNotFoundError: File .../missing.zarr does not exist.`

`StockDataset` reads one vendor's directory only. The raw root must end in the vendor name and `DatasetConfig.vendor` must be set; otherwise the scan refuses, for example with `StockDataset: DatasetConfig.vendor is not set, so there is no way to check that ... holds exactly one vendor's data.` or `StockDataset: raw_data_dir_path '...' has basename 'tiingo' but the configured vendor is 'alpaca'.` An empty or missing raw tree raises `StockDataset: no raw data for vendor 'tiingo' at frequency '1d' under '...'.` `SpotKlineDataset` raises `No CSV file matching the configured date range was found under ...` when no monthly file falls in the range.

A resampled dataset is a view of its source store. `from_raw_data()`, `from_raw_data_chunked()` and `update()` refuse with `SpotKlineDataset.from_raw_data(): a resampled dataset (resample_freq='1d') is a view of its source store and cannot be built from raw files. Build or update the source dataset, then resample it.` A `how` dict must name every variable: `SpotKlineDataset: resample_how does not name ['Open', 'Volume']; every variable of the panel needs a method (or pass one method as a str).` A target no coarser than the store's bars is refused: `SpotKlineDataset: resample_freq='1m' (60s) is not coarser than the panel's own bars (60s).` The saved resampled store is a cache like a factor store: rebuilding the source does not refresh it. Delete it, or `save()` again from a freshly resampled copy.

Intraday datasets use `XnysSessionCalendar` (`quantlab.dataset._support.session_calendar`) to turn an Eastern-time window into each date's real exchange open and close, half days included, as naive UTC timestamps.

Both built-in datasets implement `to_kunquant()`.

Two rows with the same `(timestamp, symbol)` reaching `to_xarray()` raise `ValueError: cannot convert a DataFrame with a non-unique MultiIndex into xarray`. Deduplicate first.

## See also

The chunking guide covers `from_raw_data_chunked()`, `update()` and resuming. The acquisition and registry guides describe how raw files are downloaded and how a converter is chosen from a config. The backend guide covers `XrBackend`, and the factor guide shows how a factor reads a dataset. Relevant modules: `quantlab.base.data`, `quantlab.base.config`, `quantlab.dataset.spot`, `quantlab.dataset.stock`, `quantlab.dataset._support.cleaning` and `quantlab.dataset._support.session_calendar`.
