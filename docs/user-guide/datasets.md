# Datasets and storage

A dataset object turns raw vendor files on disk into a *panel*, stores that
panel as a Zarr store, and hands it to the factor layer. This page explains
what a dataset does, how it is configured, which datasets ship with quantlab
and when to use each, how to read and filter a panel, how the two storage
backends work, how to grow a store as new data arrives, and what the cleaning
step does and does not do. Read it after
[Concepts](concepts.md) and before [Factors](factors.md). Downloading the raw
files is covered in [Data sources](data-sources.md) and, for CRSP and TAQ, in
[WRDS](wrds.md).

Every snippet below comes from the runnable example
[`examples/build_panel.py`](../../examples/build_panel.py), which builds
everything from synthetic data in a temporary directory:

```bash
uv run python examples/build_panel.py
```

## What a dataset does

A *panel* is an `xarray.Dataset` whose variables (`open`, `close`, `volume`,
...) are two-dimensional arrays indexed by `timestamp` and `symbol`. It is
dense: every symbol has a cell on every timestamp, and a cell for which the
vendor has no observation (a symbol that has not listed yet, a delisted name,
a trading halt) holds NaN. Every layer of quantlab exchanges data in this
shape.

Data moves through three tiers:

1. The *raw tier* is what a downloader wrote: vendor files in their original
   columns, for example hive-partitioned Parquet shards
   (`.../tiingo/month=2024-01/part-*.pqt`) or monthly Binance CSV files.
2. The dataset converts the raw tier into a dense panel in memory. The
   conversion deduplicates rows, densifies onto the `(timestamp, symbol)`
   grid and runs the cleaning step.
3. The panel is saved as a Zarr store, a directory of chunked arrays that
   xarray reads lazily. Factors, labels and the backtester read this store;
   they never touch the raw tier.

Every dataset class shares the same lifecycle, defined on
`quantlab.base.data.BaseDataset`:

| Method | What it does |
|---|---|
| `from_raw_data()` | Convert the whole configured range from the raw tier into memory. Nothing is written. |
| `save()` | Narrow the in-memory panel to the config's dates and symbols, then write it, replacing the store. |
| `read()` | Open the Zarr store and narrow it to the config's dates and symbols. |
| `get_xarray_dataset()` | Return the loaded panel with dimensions `(timestamp, symbol)`. |
| `from_raw_data_chunked()` | Convert and append one time window at a time, resumably. |
| `update()` | Bring an existing store up to date with the raw tier. |

`from_raw_data`, `read` and the chunked methods return the dataset itself, so
calls chain.

## Configuring a dataset

A market dataset is built from a `quantlab.base.config.DatasetConfig`:

```python
from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset

config = DatasetConfig(
    raw_data_dir_path="/data/downloads/us_equity/1d/demo/tiingo",
    zarr_file_path="/data/data/us_equity/1d/demo.zarr",
    market="us_equity",       # or "crypto_spot"
    frequency="1d",           # "1d", "1m" or "tick"
    vendor="tiingo",          # "tiingo", "alpaca" or "wrds"
    start_date="2024-01-01",  # optional, inclusive
    end_date="2024-12-31",    # optional, inclusive
    symbols=("AAPL", "MSFT"), # optional; None keeps every symbol
)
dataset = StockDataset(config)
```

`raw_data_dir_path` is the raw tier and `zarr_file_path` is the store.
`kwargs` holds dataset-specific options, such as
`{"data_type": "quotes"}` for tick data.

Assigning the config normalises it. `name` is set to the dataset class's
dotted import path, which lets a saved `config.json` rebuild the same class
later. A missing `start_date` or `end_date` becomes the open bounds
`1900-01-01` and `2100-01-01`. Dates must be zero-padded ISO strings
(`YYYY-MM-DD`): quantlab compares dates as strings throughout, so a value such
as `"2024-1-2"` is refused with a `ValueError` rather than allowed to compare
wrongly.

You rarely write paths by hand. The factories in `quantlab.config` derive
every path from one data root, which is `--data-dir` on the command-line
scripts, else the `QUANTLAB_DATA_DIR` environment variable, else a `data/`
directory beside the repository:

```python
from quantlab.config import set_data_root, stock_kline_config

set_data_root("/mnt/quant")
config = stock_kline_config(subdir="us_all", store_name="us_all.zarr", vendor="tiingo")
config.raw_data_dir_path   # '/mnt/quant/downloads/us_equity/1d/us_all/tiingo'
config.zarr_file_path      # '/mnt/quant/data/us_equity/1d/us_all.zarr'
```

`spot_kline_config` does the same for Binance klines. See the module
docstring of `quantlab.config` for the full layout.

## Available datasets

| Class | Market, frequency | `symbol` axis | Raw tier | Config |
|---|---|---|---|---|
| `quantlab.dataset.spot.SpotKlineDataset` | crypto spot, daily | trading pair (`BTCUSDT`) | Binance monthly kline CSVs | `DatasetConfig` |
| `quantlab.dataset.stock.StockDataset` | US equity, daily or minute | ticker (`AAPL`) | Tiingo or Alpaca Parquet shards | `DatasetConfig` |
| `quantlab.dataset.crsp.CrspStockDataset` | US equity, daily | integer PERMNO | CRSP daily stock file via WRDS | `CrspDatasetConfig` |
| `quantlab.dataset.nbbo.NbboPanelDataset` | US equity, bars from ticks | ticker | TAQ NBBO quotes via WRDS | `NbboDatasetConfig` |

The index-membership panels described in [Universes](universes.md) are
datasets too, built from `ConstituentDatasetConfig`.

### Binance spot klines

A *kline* (candlestick) is Binance's name for an OHLCV bar: open, high, low,
close and volume over a fixed interval. `SpotKlineDataset` stacks the
header-less monthly CSV files Binance publishes for bulk download into one
panel; the symbol is taken from each file name. The columns keep Binance's
Title-Case names (`Open`, `Close`, `Quote asset volume`, ...), and
`to_kunquant` maps them to the lowercase names the factor engine expects. The
CSVs are not downloaded by quantlab; `scripts/ingest_binance_spot.py` converts
CSVs you already have into the store. Use this dataset for crypto research.

### US stocks from Tiingo or Alpaca

`StockDataset` reads the raw tier the Tiingo and Alpaca downloaders write. The
daily panel carries raw prices (`open`, `high`, `low`, `close`, `volume`),
split- and dividend-adjusted prices (`adjOpen` ... `adjVolume`), `divCash` and
`splitFactor`. Raw and adjusted prices answer different questions: adjusted
prices give correct returns across splits and dividends, raw prices say what a
stock actually cost on the day, which matters for liquidity rules and fills.

The raw root must end in the vendor's own directory
(`.../us_all/tiingo`, never `.../us_all`), and every shard carries a `vendor`
column. Both are checked before any row is used, because two vendors' shards
under one root would otherwise merge silently into a blended price series.
Keep one vendor per store. Daily data is partitioned by `month`, minute data
by session `date`, and tick data (quotes or trades, selected with
`kwargs={"data_type": ...}`) has no dense-panel form and is not converted.

Tiingo is the natural choice for a long daily history across the whole
market, delisted names included; Alpaca adds minute bars. Tickers are the
symbol axis, so a company that changed its ticker appears under whichever
ticker the vendor files its history under.

### CRSP daily stocks

`CrspStockDataset` builds a panel from the CRSP daily stock file, the academic
reference for US equities, pulled through a WRDS (Wharton Research Data
Services) account. Its `symbol` axis is the *PERMNO*, CRSP's permanent integer
identifier for a security. A PERMNO never changes when a ticker does (FB and
META are the same PERMNO 13407), so a PERMNO panel is immune to ticker
renames. The panel carries the same twelve variables as a Tiingo panel, so
factors and backtests read it unchanged, plus CRSP extras such as `ret`,
`market_cap`, `shrout` and `is_delisting`.

Because the axis is integer PERMNOs, the ticker-side `symbols` field is
refused on its config; restrict a conversion with `permnos` instead, and pick
the security types to keep with `security_filter` (the default,
`"equity_common"`, drops ADRs, funds, ETFs and units). Period-correct tickers
are written to a sidecar file next to the store for display; see
`quantlab.dataset.crsp.tickers.CrspTickerLookup`. Use CRSP when survivorship
bias and corporate-action accuracy matter most, which is almost always for a
historical backtest. [WRDS](wrds.md) covers the download and every config
field.

### NBBO quote bars

The *NBBO* (national best bid and offer) is the highest bid and lowest ask
across all US exchanges at each moment. `NbboPanelDataset` reads TAQ NBBO
records pulled through WRDS and resamples each trading session onto regular
bars of `bar_interval` (`"1s"` to `"30m"`), producing variables such as
`bid`, `ask`, `mid`, `spread`, `spread_bps` and time-weighted sizes. Session
times come from the NYSE calendar, so half days are handled. Use it to study
spreads, liquidity or intraday microstructure. Convert it with
`from_raw_data_chunked(granularity="day")`: a second-level panel over many
symbols is large, and one window is held in memory at a time.

## Build and read a panel

With a raw tier on disk, converting and saving is two calls. The example
writes January and February 2024 for four symbols, where `DDD` stops trading
at the end of January and `BBB` has one bad zero close:

```python
dataset = StockDataset(config).from_raw_data()
dataset.save()
panel = dataset.get_xarray_dataset()
print(dict(panel.sizes))
print(list(panel.data_vars))
print(dataset.symbols, pd.Timedelta(dataset.time_interval))
```

```text
2. full panel: {'timestamp': 43, 'symbol': 4}
   variables: ['open', 'high', 'low', 'close', 'volume', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume', 'divCash', 'splitFactor', 'anomaly_flag']
   symbols: ['AAA', 'BBB', 'DDD', 'PNY']
   bar spacing: 1 days 00:00:00
```

`time_interval` is the most common gap between timestamps, so weekends do not
distort it. `num_symbols` and `symbols` describe the loaded axis.

Later sessions read the store rather than reconverting:

```python
panel = StockDataset(config).read().get_xarray_dataset()
```

## Filter by date and symbol

The config's `start_date`, `end_date` and `symbols` narrow what `read()` and
`save()` return. Both date bounds are inclusive.

```python
narrow = StockDataset(
    stock_config(start_date="2024-01-29", end_date="2024-02-02", symbols=("AAA", "DDD"))
).read()
print(narrow.get_xarray_dataset()["close"].to_pandas().round(2))
```

```text
3. narrowed panel: {'timestamp': 5, 'symbol': 2}
symbol        AAA    DDD
timestamp
2024-01-29  48.16  25.48
2024-01-30  48.10  25.24
2024-01-31  48.76  25.00
2024-02-01  48.44    NaN
2024-02-02  48.61    NaN
```

`DDD` stays on the axis after it stops trading; its cells simply become NaN.
Symbols never disappear from a panel because they have no data in a window.

Three behaviours are worth knowing:

- Filtering happens in place on the loaded panel, and `read()` does not
  reload a panel it already holds. Widening the dates of a dataset that has
  already been read therefore has no effect until you call
  `read(overwrite=True)`. Constructing a fresh dataset object is the simplest
  habit.
- `save()` narrows first and then replaces the whole store. Saving from a
  dataset configured for one month leaves a one-month store. To add data to
  an existing store, use the chunked path below.
- `head(n)` returns the first `n` rows of the store as a Polars `LazyFrame`
  without loading or narrowing anything, which is handy for checking column
  names.

`get_lazyframe()` returns the loaded panel in long format, one row per
`(timestamp, symbol)`, for code that prefers Polars.

## Storage backends

A dataset delegates persistence to a *backend*, the object that knows how
data is stored, as opposed to what it means. Both backends live in
`quantlab.backend` and implement the `quantlab.base.backend.DataBackend`
interface (`read`, `write`, `to_internal`, `filter_by_date`,
`filter_by_symbol`, `get_xarray_dataset`, `get_lazyframe`, `head`).

`XrBackend` holds an `xarray.Dataset` and stores it as Zarr. Every dataset,
factor and model owns one as `data_backend`. Beyond `read` and `write` it
provides `append`, which extends a store along the time axis after checking
that the new window cannot corrupt it (same symbol axis, same variables, no
overlap with stored timestamps, no NaN written into an integer variable), and
the `widen_*` methods used when a store gains symbols or variables.

`PlBackend` holds a Polars `LazyFrame` and stores it as a Parquet file. It is
used for long-format reference tables such as the universe catalog, where one
row per record is the natural shape.

```python
from quantlab.backend import PlBackend, XrBackend

XrBackend().to_internal(panel).write("copy.zarr")
backend = XrBackend().read("copy.zarr")

long_frame = backend.get_lazyframe().select("timestamp", "symbol", "close", "volume")
PlBackend().to_internal(long_frame).write("close.parquet")
table = PlBackend().read("close.parquet")
table.filter_by_date("timestamp", "2024-02-01", "2024-02-02")
table.filter_by_symbol("symbol", ("AAA", "BBB"))
as_panel = table.get_xarray_dataset(["timestamp", "symbol"])
```

### The `get_xarray_dataset(indexes)` shape contract

`get_xarray_dataset(indexes)` returns the data indexed by exactly the
dimensions named in `indexes`, in that order. Variables laid out on any other
dimension are dropped, dimensions no longer used are dropped with their
coordinates, and the result is transposed onto the requested order. Asking
for a dimension the data does not have raises `ValueError`. Passing `None`
returns the held object unchanged, with no shape request. For `PlBackend` the
argument is required and names the columns that become dimensions.

```python
backend.get_xarray_dataset(["timestamp", "symbol"])["close"].dims
backend.get_xarray_dataset(["symbol", "timestamp"])["close"].dims
backend.get_xarray_dataset(["timestamp"])
```

```text
4. XrBackend dims, (timestamp, symbol): ('timestamp', 'symbol')
   XrBackend dims, (symbol, timestamp): ('symbol', 'timestamp')
   indexes=['timestamp'] keeps {'timestamp': 43} and variables []
   PlBackend round trip: {'timestamp': 2, 'symbol': 2} ['close', 'volume']
```

Code that consumes a panel should always ask for the shape it needs. The
dataset's own `get_xarray_dataset()` does this for you and always returns
`(timestamp, symbol)`.

## Append new data and convert large histories

`from_raw_data()` materialises the whole date range in memory at once, which
is fine for a few years of daily data on a few thousand symbols and not fine
for a full-market minute history. `from_raw_data_chunked()` converts one time
window at a time and appends each to the store, so peak memory scales with
the window:

```python
dataset = StockDataset(config).from_raw_data_chunked(granularity="month")
result = dataset.last_chunk_result
```

`granularity` is one of `"year"`, `"quarter"`, `"month"`, `"day"` or
`"hour"`. Before the first window, the dataset scans the raw tier once to fix
the *pinned symbol axis*, the full ordered list of symbols over the whole
range, and densifies every window onto it, so all windows line up column by
column. Each completed window is recorded in a *ledger*, a JSON file next to
the store (`<store>.chunks.json`). A run that crashes or is cancelled resumes
by skipping the windows the ledger already lists. The run's outcome is
published as a `ConversionResult` on `last_chunk_result`:

```text
5. first chunked run: planned 2 written 2 skipped 0 rows 43
   second run (nothing new): written 0 skipped 2 resumed True
```

When new raw data lands, `update()` brings the store up to date. In the
example, March arrives together with a new listing, `EEE`:

```python
updated = StockDataset(config).update(granularity="month")
```

```text
   update after March: written 1 skipped 2
   store now: {'timestamp': 64, 'symbol': 5} symbols ['AAA', 'BBB', 'DDD', 'EEE', 'PNY']
   EEE closes observed before March: 0
```

Only the March window was converted. Because the symbol axis grew, the store
had to be reconciled first, and `from_raw_data_chunked` offers three
strategies through `on_new_listing`:

- `"refuse"` (the default) stops with an error that explains the choice.
- `"widen"` adds the new symbols to the existing store as NaN over its
  history. This is right for genuine new listings, which had no history.
- `"rebuild"` sets the store aside and reconverts every window from the raw
  tier, which recovers real history for symbols the raw tier already had. A
  failed or cancelled rebuild restores the original store.

`update()` chooses for you: it asks the raw tier whether the added symbols
have any rows inside the store's existing time range. None means new
listings, so it widens (as it did for `EEE` above); some means it rebuilds. A
symbol that disappeared from the raw tier resolves to `"refuse"`, since both
other strategies would lose data. The decision is logged before it runs.

The ingest scripts expose the same path: `--to-zarr` converts after
downloading, `--chunk` sets the granularity, and `--on-new-listing` picks the
strategy. See [Data sources](data-sources.md).

`SpotKlineDataset` supports the chunked path but reconverts the whole range
for every window, so it bounds the size of each write, not the memory used.
`StockDataset`, `CrspStockDataset` and `NbboPanelDataset` push the window into
their Parquet scan and are memory-bounded.

## Data cleaning

Cleaning in quantlab reports problems; it never repairs them. Nothing fills,
interpolates or corrects a value, so everything in a panel was observed by
the vendor. The rules live in `quantlab.dataset._support.cleaning`:

- Before densifying, `dedup_raw_frame` drops duplicate `(timestamp, symbol)`
  rows, keeping the last one. Overlapping downloads produce such duplicates,
  and a later file more often carries corrected data.
- After densifying, `clean_market_data` checks that the OHLCV columns exist
  (a missing column raises), warns about unexpected nulls, and adds a boolean
  `anomaly_flag` variable. A cell is flagged when a price is zero or negative,
  or when `close` moves by more than 50% in one bar. The check reads the
  lowercase price columns, so on a Binance panel (Title-Case columns) the
  flag is present but always `False`.

In the example, the bad zero close of `BBB` is flagged and left as it is:

```text
WARNING: flag_anomalies: flagged 1 anomalous (timestamp, symbol) data point(s) (zero/negative price or extreme jump); values left unmodified, see `anomaly_flag`.
   anomaly_flag cells: [(Timestamp('2024-01-16 00:00:00'), 'BBB')]
```

Decide downstream what a flagged cell means for your research, for example by
masking it before computing factors. Two details apply to chunked
conversions: cleaning runs per window, so a jump that straddles a window
boundary is not flagged (a warning says how many boundaries were involved),
and integer variables are promoted to float64 so that later windows can hold
NaN. Panels that are not OHLCV bars use their own validators:
`clean_membership_panel` for index membership and `clean_nbbo_panel` for
NBBO bars.

## Export to KunQuant

Besides the panel itself, market datasets have one exit.
`to_kunquant(data_columns)` returns contiguous float32 `[time, symbol]`
arrays, the input format of the KunQuant factor engine; the factor layer
calls it for you (see [Factors](factors.md)).

## See also

- [Universes](universes.md): restrict a panel to point-in-time index members
  or to liquid, tradeable symbols.
- [Data sources](data-sources.md) and [WRDS](wrds.md): produce the raw tier.
- [Extending quantlab](../developer-guide/extending.md): write a dataset for a
  new vendor by implementing `_raw_data_to_xr`.
- The docstrings of `quantlab.base.data.BaseDataset`,
  `quantlab.backend.XrBackend` and `quantlab.base.chunking.ChunkLedger` for
  every parameter.
