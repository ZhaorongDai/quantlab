# Sharadar daily stocks

Sharadar is quantlab's primary US-equity vendor ([ADR 0023](adr/0023-sharadar-is-the-primary-vendor-stores-hold-raw-prices-on-permaticker.md)). quantlab pulls Sharadar's tables from its own API (`api.sharadar.com/v1.0`), keeps them as raw parquet, and converts the stock price table (SEP) into a `(timestamp, symbol)` Zarr panel whose symbol axis is the **permaticker**.

## Prerequisites

Downloading needs a paid sharadar.com key, read only from the environment:

```bash
export SHARADAR_API_KEY=<your-sharadar-key>
```

A missing key, or an HTTP 401/403 from Sharadar, raises `SharadarEntitlementError` naming the table. Converting and reading a store need no key and no network.

The data is licensed for personal use: keep raw files and stores on your own machines, never in the repository or a tracker artifact.

## Pulling the raw tier

`SharadarClient.bulk_table(code, download_dir)` pulls one whole table as Sharadar's bulk zip and writes it as `<download_dir>/sharadar/<code>/<code>.parquet`, with the vendor's column names and order checked against the declared schema. The tables available so far are `sep` (stock prices), `sfp` (fund prices), `sf1` (fundamentals), `actions` (dividends, splits and other corporate actions), `tickers` (the ticker-to-permaticker mapping) and `indicators` (the data dictionary); TICKERS and INDICATORS stay parquet sidecar tables and never become Zarr stores.

```python
from quantlab.acquisition.sharadar.client import SharadarClient

client = SharadarClient()
for code in ("sep", "actions", "tickers", "indicators"):
    client.bulk_table(code, "/data/quantlab/downloads")
```

The zip is downloaded in parallel byte ranges (`SharadarClient(download_workers=8, part_bytes=64 << 20)` by default), or as one stream when the storage ignores `Range`. Rate-limit (429) and server-error responses, connections that fail to open and byte ranges whose stream breaks are retried after a back-off. A full-history SEP zip is about 1 GB.

## Building and reading the panel

```python
from quantlab.dataset.config import SharadarDatasetConfig
from quantlab.dataset.sharadar.stock import SharadarStockDataset

config = SharadarDatasetConfig(
    zarr_file_path="/data/quantlab/zarrs/sharadar_sep_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarStockDataset(config).from_raw_data().save()
panel = SharadarStockDataset(config).panel("2024-01-02", "2024-01-05")
```

Observed on a one-row synthetic raw tier:

```python
>>> sorted(panel.data_vars), panel.symbol.dtype
(['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'anomaly_flag', 'close', 'divCash', 'high', 'low', 'open', 'splitFactor', 'volume'], dtype('int64'))
```

What the panel means:

- **Symbol axis.** Each raw row is mapped to its permaticker through the TICKERS rows of its own table (labelled `SEP` in the bulk file, `stocks` over the REST API). A renamed company keeps one column; a delisted company whose ticker was reused keeps its own column. The conversion refuses a ticker TICKERS does not know, a ticker mapped to two permatickers, and two rows of one permaticker on one date.
- **Raw prices.** `close` is SEP's `closeunadj`. `open`, `high` and `low` are SEP's split-adjusted values times `closeunadj / close`, and `volume` is SEP's split-adjusted volume divided by that ratio. Sharadar's adjusted columns are not stored, because the vendor rewrites them over the whole history on every ex-date.
- **Adjusted prices.** quantlab chains them itself from the raw prices and the `dividend` and `split` rows of ACTIONS, with the CRSP panel's convention and names, so a factor or backtest written against CRSP reads a Sharadar store unchanged:
  - `divCash` is the cash per share on the ex-date: dividends plus `spinoffdividend`, the value of spun-off shares per parent share (the `spinoff` share-ratio row is not counted again). ACTIONS gives a dividend adjusted for later splits, so it is multiplied back by that day's `closeunadj / close`; pull SEP and ACTIONS together so both are adjusted for the same splits.
  - `splitFactor` is new shares per old share on the split's effective date, 1.0 otherwise.
  - The day's total return is `(close + divCash) * splitFactor / previous close - 1`. On an ex-date that is also a split date, the distribution is cash per share after the split (DD on 2019-06-03: a 1-for-3 reverse split and the Corteva spin-off), so the split scales it with the close. This matches Sharadar's own `closeadj` on 280 of the 283 such events priced on the 2026-10-05 pull; `adjClose` starts at each permaticker's first positive close in the window (the anchor) and grows by those returns. `adjOpen`/`adjHigh`/`adjLow` scale with `adjClose / close`; `adjVolume` is the raw volume in the anchor's shares.
  - An event on a date without a positive close is logged and left out. As on CRSP, moving `start_date` later moves the anchor and rescales the adjusted history; a store that only grows forward keeps every past value.
- **Trading contract.** A bar without a fill price is untradable. So is a bar Sharadar carries forward through a trading halt: volume 0 and the previous close repeated. For example, SIVB repeats its $106.04 close from 2023-03-13 to 2023-03-27 before its first OTC print of $0.40, so a backtest cannot sell it at $106.04 during the halt; the position stays locked until a real print. A delisted security settles at its last close. Sharadar has no delisting return and none is imputed, so backtests are slightly optimistic on names that went bankrupt (ADR 0023).
- **Selection.** See the universe below; the ticker-based `symbols` field is refused.

## Universe

Which permatickers a conversion keeps depends on whether it has an explicit roster.

- **Market universe (no roster).** Filtered by the TICKERS `category` through `category_filter`, whose default `"default"` is the table's own: for SEP, domestic common stock in every share class (`Domestic Common Stock`, `Domestic Common Stock Primary Class` and `Domestic Common Stock Secondary Class`; ADRs, Canadian filers, preferreds and every other category are dropped, which since 2024 keeps 5,902 of the 7,895 permatickers priced in SEP); for SFP, every fund category. Pass `category_filter=None` to keep everything, or a tuple of categories of your own.
- **Roster.** `permatickers=(...)` names securities explicitly, and `roster_universe="sp500"` takes every permaticker that was an S&P 500 member at some point in the conversion window, with all its bars. A roster is never filtered by category: a listed member is never silently dropped. Both together give the union.

The `sp500` raw table (pull it with `client.bulk_table("sp500", ...)`) also becomes a point-in-time membership panel on the permaticker axis:

```python
from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset

membership = SharadarSP500ConstituentDataset(ConstituentDatasetConfig(
    zarr_file_path="/data/quantlab/zarrs/sharadar_sp500_membership.zarr",
    cache_dir="/data/quantlab/downloads/sharadar",
))
membership.from_raw_data().save()
```

An `added` or `removed` row's date is the effective membership date, so a stock is a member from its `added` date and stops being one on its `removed` date. The changes go back to the index's launch in 1957, but Sharadar's prices, and so its permatickers, begin on 1997-12-31, so the panel answers membership from that date: the 24 members that left before then are dropped, and so is a former member Sharadar never priced (one, CBB1, which left on 1998-01-27), with a warning naming it. A current member without a permaticker is refused, because it means TICKERS is older than SP500. A ticker with no change at all was a member throughout. On the 2026-10-05 pull the panel holds 499 to 507 members a day. The panel ends on the table's last date, not today, so a rebuild from the same raw file gives the same panel.

## Funds and benchmarks

SFP (funds: ETFs, closed-end funds, ETNs and the like) converts through the same path as SEP, adjusted prices included: pull `sfp` with `client.bulk_table("sfp", ...)` and set `table="sfp"`. Its TICKERS rows are labelled `SFP`, its default `category_filter` keeps every fund category (`ETF`, `CEF`, `ETD`, `ETN`, `CEF Preferred`, `UNIT`, `ETMF`, `IDX`, `MF` on the 2026-10-05 pull); a whole-table store is `sharadar_sfp_1d.zarr`. A backtest benchmark is a store holding one fund:

```python
from quantlab.dataset.config import SPY_PERMATICKER, SharadarDatasetConfig
from quantlab.dataset.sharadar.stock import SharadarStockDataset

spy = SharadarDatasetConfig.etf_benchmark(
    permaticker=SPY_PERMATICKER,  # 118691, SPY's SFP permaticker in TICKERS
    zarr_file_path="/data/quantlab/zarrs/sharadar_spy_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarStockDataset(spy).from_raw_data().save()
# then BacktestConfig(benchmark_dataset=SharadarStockDataset(spy), ...)
```

## Fundamentals (SF1)

SF1 holds one row per company, dimension and filing, with 105 indicators (income statement, balance sheet, cash flow, per-share values and ratios). Its raw tier keeps every dimension. A store holds one *as-reported* dimension: `ARQ` (each fiscal quarter) or `ART` (trailing twelve months). For those, `date` is the SEC filing date (the release date), and a filing that restates a period is a new row. The most-recent dimensions (`MRQ`, `MRY`, `MRT`) are dated at the period end and rewritten on restatement, so they would leak values one to three months early. `SharadarFundamentalsConfig` refuses them, and so does `ARY`.

```python
from quantlab.dataset.config import SharadarFundamentalsConfig
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset

config = SharadarFundamentalsConfig(
    zarr_file_path="/data/quantlab/zarrs/sharadar_sf1_arq.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
    dimension="ARQ",          # or "ART"
    stale_after_days=365,     # the default; None never expires a row
)
SharadarFundamentalsDataset(config).update()
panel = SharadarFundamentalsDataset(config).panel("2024-01-02", "2024-12-31")
```

The panel follows the point-in-time rule of [ADR 0003](adr/0003-point-in-time-fundamentals-by-release-date.md):

- **Calendar and axis.** The timestamps are SEP's trading days, so the panel lines up with the price panels bar for bar, and the `symbol` axis is the permaticker. The universe fields (`permatickers`, `roster_universe`, `category_filter`) work as for a price panel; the default keeps domestic common stock.
- **Placement.** A row is available from the first trading day on or after its release date. A weekend filing is shown on Monday.
- **Selection.** On each day the panel shows, per security, the available row with the latest fiscal period, and within that period the latest filing. A restatement is shown from its own release date, never earlier. A later filing of an older period never replaces a newer period's row.
- **Staleness.** A row no newer row has replaced stops being shown `stale_after_days` after its release date.
- **Variables.** Each indicator is a float64 variable whose `unit` attribute is the vendor's unit type from INDICATORS: `currency` (the reporting currency), `USD`, `ratio`, `units` or a per-share unit. SF1's `marketcap` and `ev` are in USD. `release_date` and `reportperiod` (the fiscal period end) say which row each cell shows.

On the 2026-10-05 pull, SF1 has 3.2 million rows (680,000 ARQ and 690,000 ART). Each store holds 15,645 permatickers on 7,233 trading days from 1997-12-31, takes about 10 minutes to build with a 3.4 GB peak per year window, and is about 450 MB on disk. About 4,900 companies are shown on a 2024 trading day, and no cell shows a row before its release date.

**Updates by `lastupdated`.** SF1 is keyed by `(ticker, dimension, date, reportperiod)`, not by date, so it is refreshed differently from the price tables. `SharadarClient.updated_table("sf1", download_dir)` asks REST for every row whose `lastupdated` is on or after the table's watermark. Pages are cut at whole tickers (`ticker.gte`), so a ticker's rows are never split across two pages. The rows are written as `sf1/updated_<pulled at>_<since>.parquet`, and reading the table replaces each earlier row with the same key; the newest pull wins. The watermark moves to the day the pull started. A new bulk pull deletes the updated files. `update()` then appends the new trading days. A stored day is never rewritten, so a value the vendor changes afterwards reaches only the days after the update. A first filer is added to the store with no history; a security that gains rows inside the store's range rebuilds it, as for the price stores. On 2026-10-05, the rows changed in the week since 2026-09-28 (1,436) came back in one request. Sharadar stops a query after 15 seconds: a pull spanning a mass re-stamp of `lastupdated` (the vendor re-stamped most of SF1 on 2026-07-31) can answer HTTP 503, and `update.py` then pulls SF1 in bulk instead.

Known limits:
- A row the vendor re-files under a new `date` or `reportperiod` is a new key, so the old row stays beside it until the next bulk pull.
- A row placed by an update reaches only the days after that update, while a rebuild places it at its release date. Both are free of look-ahead, but a rebuilt store and an updated one can differ on the days between.
- Unlike the Compustat panel, there is no link end: a delisted company's last row is shown until it goes stale.

## Scripts

The download and the daily update are two scripts, run from the repository root. Both read `SHARADAR_API_KEY`, take `--download-dir` (raw tables under `<download-dir>/sharadar/<table>/`) and `--zarr-dir` (the stores), both defaulting to the current directory, and refuse either directory inside the repository, because the data is licensed for personal use.

```bash
export SHARADAR_API_KEY=<your-sharadar-key>
# once: every table as a bulk zip, then the price and membership stores
uv run python scripts/sharadar/download.py --download-dir /data/quantlab/downloads --zarr-dir /data/quantlab/zarrs
# every morning: TICKERS and SP500 whole, SEP/SFP/ACTIONS as trailing windows, SF1 by lastupdated, then append
uv run python scripts/sharadar/update.py --download-dir /data/quantlab/downloads --zarr-dir /data/quantlab/zarrs
```

`download.py` pulls `tickers`, `indicators`, `sep`, `sfp`, `actions`, `sp500` and `sf1` (never METRICS) and builds `sharadar_sep_1d.zarr`, `sharadar_sfp_1d.zarr`, `sharadar_sp500_1d.zarr` (the `roster_universe="sp500"` store: every permaticker ever a member, with all its bars), `sharadar_spy_1d.zarr` (SPY alone, `SPY_PERMATICKER`), `sharadar_sp500_membership.zarr`, and the fundamentals stores `sharadar_sf1_arq.zarr` and `sharadar_sf1_art.zarr`, all with `update()`, so each keeps the chunk ledger the daily update reads; `--start` narrows the stores, `--years` picks the history tier. `update.py` extends each store from the first day it holds and prints where vendor corrections were reported. Sharadar is registered as a source (`DataSourceRegistry.get("sharadar")`, one capability per table), but its raw tier is whole tables rather than a symbol-batched download, so `registry.run()` refuses it and points here; `registry.convert()` builds the SEP, SFP and SF1 stores.

## Daily update

A bulk pull is the first download; every morning after it, refresh the raw tier with trailing date windows and append the new bars to the store:

```python
from quantlab.acquisition.sharadar.client import SharadarClient
from quantlab.dataset.sharadar.stock import SharadarStockDataset

client = SharadarClient()
client.bulk_table("tickers", "/data/quantlab/downloads")  # new listings and ticker changes
for code in ("sep", "sfp", "actions"):
    client.window_table(code, "/data/quantlab/downloads")
SharadarStockDataset(config).update()
```

**The raw tier.** `window_table` asks REST for every calendar day from the 10th most recent raw trading day on or before the table's watermark through today (US/Eastern), one request per day and page of 10,000 rows in ticker order. The rows are written as one file, `<code>/window_<pulled at>_<from>_<to>.parquet`, which is a complete copy of the table over those dates: reading the table (`quantlab.dataset.sharadar.tables.scan_raw_table`) keeps a row only from the newest file covering its date, so a vendor correction inside the window replaces the bulk row and a ticker change inside it cannot duplicate a security. Only after the file is written is the table's watermark, `<code>/_watermark.json`, moved to the window's last day; a failed pull leaves both as they were, and the next pull starts from the last watermark that was written. A new bulk pull deletes the windows it supersedes and sets the watermark to the day of the pull.

**The store.** `SharadarStockDataset.update()` is `BaseDataset.update`: build the store with it the first time (it keeps the chunk ledger an update reads), and every later run appends only the bars after the ledger's last end, at any granularity. Sharadar adds what a vendor with rewritten history needs through two hooks, its derivation's end and `_continue_store`:

- The derivation ends at the watermark of every input table (the price table and ACTIONS), so a bar is never stored before its dividends and splits are known. An update interrupted between the pulls stops at the older watermark and the next one continues from the store's last bar.
- The last 10 stored bars are derived again from the raw tier. A raw price, `divCash` or `splitFactor` the vendor has since changed is listed in `<store>.corrections.json` (table, permaticker, date, variable, stored and vendor value) and logged; it is never written. Earlier rows of the store stay byte-identical.
- Each security's new adjusted prices continue from its last stored `adjClose` and `adjVolume`, so the chain is never re-anchored, even for a security halted for longer than the overlap. On the 2026-10-05 pull, a store built through 2026-09-25 and updated to the end of the raw tier equals a store built in one go to within 7e-16.
- A new security follows `BaseDataset.update`'s rule: a new listing is added with no history; a security new to the store that has bars inside its range (a roster change) rebuilds the store, the one case where earlier rows are rewritten.

## Periodic bulk diff

The daily update compares only the last 10 stored bars with the vendor. To find drift anywhere in a store, pull every input table in full into a separate download directory and diff the store against it:

```python
from quantlab.acquisition.sharadar.client import SharadarClient
from quantlab.dataset.sharadar.stock import SharadarStockDataset

client = SharadarClient()
for code in ("sep", "tickers", "actions"):
    client.bulk_table(code, "/data/quantlab/bulk_check")
differences = SharadarStockDataset(config).diff("/data/quantlab/bulk_check/sharadar")
```

Pulling into a separate directory keeps the store's own raw tier as it was built; `diff()` with no argument compares with the store's own raw tier instead (after a bulk pull has replaced it). Every stored security is compared over the store's dates, up to the compared raw tier's watermark, on the raw prices, `divCash` and `splitFactor`, one calendar year at a time. Each difference names the table, permaticker, date and variable, with the stored and vendor values (`None` where one side has no value, as for a bar the vendor dropped). The differences are returned and written to `<store>.diff.json`, which every run rewrites; the store, its chunk ledger and both raw tiers are left unchanged. Securities the vendor has but the store lacks are not compared. On the 2026-10-05 raw tier, a store from 2025 diffs against its own raw tier in about 4 seconds with no difference.
