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

A file is fetched in byte ranges of `part_bytes` (32 MiB), several at once, each written at its offset of `<name>.part`; a range whose stream breaks is asked again for only the bytes it still lacks. The whole file is then checked against the vendor's size and decoded to the end of its gzip, and only then renamed into place; a file already there whole is not fetched again. A day without a file raises `MassiveNotPublishedError` (a holiday, or today before the vendor publishes); a refused key or a day outside the plan's window raises `MassiveEntitlementError`; a network failure or throttling raises `MassiveTransportError` after its retries, each after a back-off that doubles.

A range of days goes through `download_days`, oldest first, with several files in flight; it asks only for the XNYS sessions of the range (weekends and holidays are not asked for) and yields one `DayDownload` per session in date order, so a caller can convert day d while d+1 downloads:

```python
for done in client.download_days("minute_aggs", date(2016, 10, 11), date(2016, 12, 30),
                                 "/data/quantlab/downloads"):
    print(done.day, done.published, done.fetched_bytes, done.seconds)
```

A session Massive has no file for yet comes back with `published` false and is skipped. Each published day moves the data type's watermark (`massive/<data type>/_watermark.json`) to it while no earlier day of the run is missing, and a later run starts after the watermark, so an interrupted run resumes where it stopped and a day not published yet is asked for again. The watermark, not the files present, says what was downloaded (a trade file is deleted once converted); it does not say what was converted.

Concurrency is set on the client: one file is fetched at a time in `streams` byte ranges (default 16), while up to `files - 1` earlier ones are verified (`files` defaults to 4). Verifying decodes the whole gzip on one core and takes about twice as long as fetching: on the training server on 2026-10-09 a 2 GB trade file of 2025 fetched in 33 s at 16 streams and decoded in 81 s. Ranged reads from the server with the vendor reached directly: 1 stream 9 MB/s, 4 streams 35, 8 streams 54, 16 streams 40 to 61, 32 streams 72 MB/s.

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

The panel holds, on `(timestamp, symbol)`:

| variable | what it is |
|---|---|
| `open`, `high`, `low`, `close` | prices of the trades eligible for the matching update rule |
| `volume`, `n_trades` | size and count of the volume-eligible trades |
| `dollar_volume` | price times size of the volume-eligible trades; VWAP is `dollar_volume / volume` |
| `buy_volume`, `sell_volume` | volume signed by the tick rule (below) |
| `offexchange_volume` | volume reported through a TRF (`trf_id` non-zero) |
| `oddlot_volume` | volume of trades under 100 shares, fractional ones included |

Volumes are floats, so fractional shares are not truncated.

- **Counting.** Each trade counts towards high/low, open/close and volume separately, by the consolidated update rules of all its conditions (Massive's condition table). Corrected and cancelled trades (every `correction` but 0, a regular trade, and 12, the corrected print that replaces a trade), and trades with a condition the table does not know, are dropped.
- **Time.** Bars are cut on `sip_timestamp`, right-closed and labelled at their end: the bar labelled 14:31 UTC covers trades after 14:30 up to and including 14:31.
- **Session.** The regular session of the XNYS calendar, half days included (`session_start`/`session_end` set another window).
- **Tick rule.** Each volume-eligible trade is compared with the price of the volume-eligible trade before it in the same session: an up-tick is a buy, a down-tick a sell, a zero tick takes the side of the last non-zero tick. The previous price carries across bars but not across sessions; a session's first trade, and zero ticks before its first price change, count in neither split, so `buy_volume + sell_volume` may be less than `volume` (about 1% of the volume on 2016-11-25).
- **Empty bars.** A bar without an eligible trade has NaN prices and zero volumes and `n_trades`; a bar of volume-only trades (odd lots, for example) has volume and NaN prices.
- **Symbols.** Each raw `(date, ticker)` is mapped to its permaticker as traded that day, through Sharadar's SEP tickers and then SFP's. A ticker that maps to nothing is dropped, logged and recorded in the sidecar `<store>.massive_stats.json`, with the day's counts: `trades_in`, `dropped_correction`, `dropped_unknown_condition`, `outside_session`, `volume_ineligible` (kept, but no condition lets it count for volume), `unmapped_tickers` and `unmapped_trades`.

Convert one day per window: a day of the whole market is tens of millions of trades (a 2016 half day, 14 million trades, peaks at about 5 GB of memory).
