# Massive Trade bars

Massive (Stocks Developer) is quantlab's vendor of intraday US-equity trades ([ADR 0030](adr/0030-massive-trade-bars-are-built-from-every-trade-on-permaticker.md)). It serves every SIP trade, and its own minute and day aggregates, as one gzipped CSV per data type and trading day over a rolling ten-year window. quantlab downloads those files as they are, and builds its own **Trade bars** from the trades: one-minute bars (or any `BarInterval`) on the Sharadar **permaticker** axis, counted by the SIP's rules per trade condition.

## Prerequisites

Downloading needs the Massive credentials, read only from the environment:

```bash
export MASSIVE_API_KEY=<your-api-key>
export MASSIVE_S3_ACCESS_KEY_ID=<your-s3-access-key-id>
# optional; defaults to MASSIVE_API_KEY
export MASSIVE_S3_SECRET_ACCESS_KEY=<your-s3-secret>
```

A missing variable raises `MassiveCredentialError` when the client is made, before any transfer. Converting and reading a store need no credential and no network, but they need the Sharadar raw tier's TICKERS and ACTIONS (see [sharadar.md](sharadar.md)) to map tickers to permatickers.

The data is licensed for personal use: keep raw files and stores on your own machines, never in the repository or a tracker artifact.

## Pulling the raw tier

```python
from datetime import date

from quantlab.acquisition.massive.client import MassiveClient

client = MassiveClient()
client.condition_table("/data/quantlab/downloads")
client.download_day("trades", date(2016, 11, 25), "/data/quantlab/downloads")
client.download_day("minute_aggs", date(2016, 11, 25), "/data/quantlab/downloads")
```

The files land under `<download-dir>/massive/`:

```text
massive/trades/2016/2016-11-25.csv.gz
massive/minute_aggs/2016/2016-11-25.csv.gz
massive/conditions/conditions_<UTC stamp>.json
```

A file is written as `<name>.part`, checked against the vendor's size and decoded to the end of its gzip, then renamed into place; a file already there whole is not fetched again. A day without a file raises `MassiveNotPublishedError` (a holiday, or today before the vendor publishes); a refused key or a day outside the plan's window raises `MassiveEntitlementError`; a network failure raises `MassiveTransportError` after its retries.

The download script (a backfill from the oldest day, download and conversion pipelined) is not written yet; `registry.run()` refuses Massive, as it refuses Sharadar.

## Building and reading the panel

```python
from quantlab.dataset.config import MassiveTradeBarsDatasetConfig
from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

config = MassiveTradeBarsDatasetConfig(
    zarr_file_path="/data/quantlab/zarrs/massive_trade_bars_1m.zarr",
    raw_data_dir_path="/data/quantlab/downloads/massive",
    sharadar_dir="/data/quantlab/downloads/sharadar",
    start_date="2016-11-25",
    end_date="2016-11-25",
)
MassiveTradeBarDataset(config).from_raw_data_chunked(granularity="day")
panel = MassiveTradeBarDataset(config).panel("2016-11-25", "2016-11-26")
```

The panel holds `open`, `high`, `low`, `close`, `volume` and `n_trades` on `(timestamp, symbol)`:

- **Counting.** Each trade counts towards high/low, open/close and volume separately, by the consolidated update rules of all its conditions (Massive's condition table). Corrected and cancelled trades (every `correction` but 0, a regular trade, and 12, the corrected print that replaces a trade), and trades with a condition the table does not know, are dropped.
- **Time.** Bars are cut on `sip_timestamp`, right-closed and labelled at their end: the bar labelled 14:31 UTC covers trades after 14:30 up to and including 14:31.
- **Session.** The regular session of the XNYS calendar, half days included (`session_start`/`session_end` set another window).
- **Empty bars.** A bar without an eligible trade has NaN prices and zero `volume` and `n_trades`; a bar of volume-only trades (odd lots, for example) has volume and NaN prices.
- **Symbols.** Each raw `(date, ticker)` is mapped to its permaticker as traded that day, through Sharadar's SEP tickers and then SFP's. A ticker that maps to nothing is dropped, logged and recorded in the sidecar `<store>.massive_stats.json`, with the day's trade counts and what each rule dropped.

Convert one day per window: a day of the whole market is tens of millions of trades (a 2016 half day, 14 million trades, peaks at about 5 GB of memory).
