# Data sources

This page explains how quantlab gets market data onto your disk. It covers the
three vendors quantlab can download from, how to give it credentials, how to
start a download from the command line or from Python, where the files land,
how an interrupted download resumes, how vendor rate limits and the pre-flight
volume guard behave, how to look at what is already downloaded without any
credentials, and how raw downloads become the `(timestamp, symbol)` panel the
rest of the pipeline reads. Read it before your first download. WRDS (CRSP daily
stock files and TAQ quotes) has enough of its own concepts that it has a
separate page, [WRDS: CRSP and TAQ](wrds.md).

The runnable companion to this page is `examples/inspect_data_sources.py`. It
needs no credentials or network access, and the output shown below comes from
it.

## Two tiers: raw files and the panel

Every download produces a *raw tier*: parquet files that hold the vendor's rows
more or less exactly as they arrived, plus a few small JSON bookkeeping files.
Nothing downstream reads the raw tier directly. A separate *conversion* step
turns it into a Zarr store holding an `xarray.Dataset` indexed by `timestamp`
and `symbol`. The project calls this a *panel*: one row per date, one column per
security, one variable per field (close, volume and so on). Factors, models and
backtests read panels, never raw files.

Keeping the two steps apart is deliberate. A full-market backfill can take
hours, and converting it is a separate long job; each step can be interrupted
and resumed on its own, and you can re-convert with different settings without
downloading anything again. See [Datasets](datasets.md) for the panel side.

## The vendors

Each vendor is described once in `quantlab.registry` by a *source descriptor*,
which lists the environment variables holding its credentials and the
*capabilities* it serves. A capability is one `(market, frequency, data_type)`
combination, together with the dataset class that can convert it to a panel.

| Vendor | Serves | Credentials | Converts to a panel |
|---|---|---|---|
| `tiingo` | US equity daily bars (end-of-day prices with adjusted columns) | `TIINGO_API_KEY` | yes, `StockDataset` |
| `alpaca` | US equity daily and minute bars; tick-level quotes and trades | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | bars yes; quotes and trades no |
| `wrds` | CRSP daily stock data; TAQ national best bid and offer quotes | `WRDS_USERNAME` (password in `~/.pgpass`) | yes, `CrspStockDataset` and `NbboPanelDataset` |

Alpaca's quotes and trades are individually timestamped events on an irregular
time axis. No dense panel can represent them without inventing data, so they
stay in the raw tier, where you can query them with polars.

Binance crypto klines are handled differently: `scripts/ingest_binance_spot.py`
converts monthly CSV files you have already downloaded from Binance's public
bulk-download site. It needs no credentials and is not a registry source.

### Listing sources in Python

`DataSourceRegistry.all()` returns every descriptor. Importing
`quantlab.registry` imports all vendor modules, so the list is always complete.

```python
from quantlab.registry import DataSourceRegistry, credential_status

for source in DataSourceRegistry.all():
    print(f"{source.vendor}: {source.display_name}")
    for cap in source.capabilities:
        converter = cap.dataset_cls.__name__ if cap.dataset_cls else "(raw only)"
        print(f"    market={cap.market} frequency={cap.frequency} "
              f"data_type={cap.data_type} -> {converter}")
    print(f"    credentials: {credential_status(source)}")
```

On a machine with no credentials set, this prints:

```text
alpaca: Alpaca Market Data
    market=us_equity frequency=1d data_type=bars -> StockDataset
    market=us_equity frequency=1m data_type=bars -> StockDataset
    market=us_equity frequency=tick data_type=quotes -> (raw only)
    market=us_equity frequency=tick data_type=trades -> (raw only)
    credentials: {'APCA_API_KEY_ID': False, 'APCA_API_SECRET_KEY': False}
tiingo: Tiingo EOD
    market=us_equity frequency=1d data_type=None -> StockDataset
    credentials: {'TIINGO_API_KEY': False}
wrds: WRDS (NYSE TAQ millisecond NBBO; CRSP Stock v2 daily)
    market=us_equity frequency=tick data_type=nbbo -> NbboPanelDataset
    market=us_equity frequency=1d data_type=crsp_daily -> CrspStockDataset
    credentials: {'WRDS_USERNAME': False}
```

`DataSourceRegistry.get("tiingo")` returns one descriptor, and
`source.supports("us_equity", "tick")` answers whether it serves a combination.
See the docstrings in `quantlab/registry.py` for the full interface.

## Credentials

Credentials are read from environment variables and nowhere else. No script
takes a key as an argument, because a key typed on the command line ends up in
your shell history, and no config field holds one, because configs are saved
as JSON beside model checkpoints. Keys are also removed from every error
message before it is logged or written to disk.

```bash
export TIINGO_API_KEY=...
export APCA_API_KEY_ID=...
export APCA_API_SECRET_KEY=...
export WRDS_USERNAME=...          # the WRDS password goes in ~/.pgpass, see wrds.md
```

`credential_status(source)` (shown above) and `is_configured(source)` tell you
whether the variables are set. They return booleans only and never read a value
into anything else. A variable set to the empty string counts as unset.

A credential is demanded only when a vendor client is constructed, which is the
moment a download starts. Everything else on this page (sizing a request,
inspecting files, converting raw files to a panel) works without one.

## Downloading from the command line

Each vendor has an ingest script under `scripts/`. They share one set of flags.

| Flag | Meaning |
|---|---|
| `--symbols AAPL,MSFT` | An explicit list of tickers. |
| `--universe sp500 --as-of-date 2015-06-01` | A roster from the point-in-time universe table instead (see [Universes](universes.md)). Choices: `sp500`, `nasdaq100`, `nasdaq_all`, `us_all`. |
| `--start-date`, `--end-date` | The window, inclusive. |
| `--refresh` | Fetch forward from each symbol's last downloaded date instead of backfilling the window. |
| `--to-zarr`, `--chunk`, `--on-new-listing` | Also convert the raw tier to a panel (see below). |
| `--force-volume`, `--rows-per-symbol-day` | Volume-guard controls (see below). |
| `--data-dir` | Storage root for this run. |

A *point-in-time* roster lists the securities that belonged to an index or
exchange on a given date, including those that have since been delisted.
Downloading only today's members would give you *survivorship bias*: a history
made only of companies that survived, which makes past returns look better
than they were. Build the universe table once with
`uv run python scripts/refresh_us_equity_universe.py` before using `--universe`.

The scripts, with the examples from their own help text:

```bash
# Tiingo daily bars
uv run python scripts/ingest_tiingo.py --symbols AAPL,MSFT \
    --start-date 2024-01-01 --end-date 2024-12-31
uv run python scripts/ingest_tiingo.py --universe sp500 --as-of-date 2015-06-01

# The whole US listed market from Tiingo (about 15,000 symbols, several hours)
uv run python scripts/ingest_us_equity.py --dry-run     # sizes the job, needs no key
uv run python scripts/ingest_us_equity.py

# Alpaca: daily or minute bars, or full-resolution quotes and trades
uv run python scripts/ingest_alpaca.py --symbols AAPL --frequency 1m \
    --start-date 2024-01-02 --end-date 2024-01-31
uv run python scripts/ingest_alpaca.py --symbols AAPL --frequency tick \
    --data-type quotes --rows-per-symbol-day 1000000 \
    --start-date 2024-01-02 --end-date 2024-01-02
```

`ingest_us_equity.py` resolves its roster by *interval overlap*: every symbol
that traded at any point in the window, not just the members on one day. It
adds `--category`, `--limit N` (the first N symbols, handy for a smoke test),
`--max-workers`, and the quota flags described under rate limits.
`scripts/backfill_us_equity_full.py` wraps the same full-market job as a Python
plan object with separately callable stages. The WRDS scripts are described in
[WRDS: CRSP and TAQ](wrds.md). Run any script with `--help` for its complete
flag list.

Without credentials a script still runs its pre-flight size estimate, then
stops when the vendor client is built:

```text
$ uv run python scripts/ingest_tiingo.py --symbols AAPL,MSFT \
      --start-date 2024-01-01 --end-date 2024-06-30
Pre-flight volume estimate (zero vendor requests issued):
  roster:            (explicit --symbols list)
  window:            2024-01-01 .. 2024-06-30
  symbols:           2
  trading days (~):  126
  rows (~):          252 (1/symbol-day, density 1.000)
  raw on disk (~):   0.00 GiB
  requests (~):      2 (batch_size=1, page_limit=10,000)
  wall clock (~):    0.0 h at 200 req/min
Acquiring symbols=('AAPL', 'MSFT') via Tiingo EOD (refresh=False)
...
RuntimeError: TIINGO_API_KEY environment variable is not set. Export it before running acquisition (see Tiingo dashboard for your key).
```

## Downloading from Python

`quantlab.registry.run(descriptor, config)` runs a download in the current
process and returns an `AcquisitionResult`. The descriptor's `config_factory`
builds an `AcquisitionConfig` with the standard paths for that vendor. This
needs `TIINGO_API_KEY` and network access, so no output is shown:

```python
from quantlab.registry import DataSourceRegistry, run

source = DataSourceRegistry.get("tiingo")
config = source.config_factory(
    symbols=("AAPL", "MSFT"),
    start_date="2024-01-01",
    end_date="2024-05-31",
    kwargs={"max_workers": 4},
)
result = run(source, config)
print(result.succeeded, result.failures, result.coverage)
```

`result.failures` maps each symbol that failed in this run to a message;
`result.cancelled` and `result.quota_aborted` say whether the run stopped
early. `run(..., refresh=True)` fetches each symbol forward from where it last
stopped instead of backfilling the whole window.

Per-run tuning lives in `config.kwargs`, so a run is fully described by its
config: `batch_size` (symbols per request), `max_workers` (concurrent
requests), `resume`, `progress`, the quota and rate-limit knobs below, and
vendor options such as Alpaca's `data_type`, `feed`, `adjustment` and
`page_limit`. Alpaca bars are requested unadjusted (`adjustment="raw"`) unless
you say otherwise. See the docstrings of `quantlab.base.acquisition.Acquisition`
and each vendor class for details.

### Progress events

By default a download draws a progress bar on stderr. To follow it from your
own code, pass a `reporter`. `CallbackProgressReporter` calls a function with
each `ProgressEvent`; `NullProgressReporter` discards them. The example script
collects the events of a three-symbol download:

```python
from quantlab.base.progress import CallbackProgressReporter

events = []
result = run(source, config, reporter=CallbackProgressReporter(events.append))
for event in events:
    print(f"    event {event.kind:<16} {event.completed}/{event.total}")
```

```text
    event coverage         3/3
    event run_started      0/3
    event batch_completed  1/3
    event batch_completed  2/3
    event batch_completed  3/3
    event run_finished     3/3
```

`coverage` carries counts of symbols already on disk in `event.detail`, and
`quota_exhausted` and `cancelled` mark early stops. The full list is
`quantlab.base.progress.EVENT_KINDS`. An exception raised inside a reporter is
logged and ignored, so a bug in your progress display cannot end a long
download.

### Cancelling

A reporter cannot stop a run; a `CancelToken` can. Pass one as `cancel=` and
call `token.cancel()` from any thread. The run stops at the next batch
boundary, keeps every completed batch on disk, and a later run resumes from
there. In the example, the token is set as soon as the first batch lands:

```python
from quantlab.base.progress import CallbackProgressReporter, CancelToken

token = CancelToken()

def stop_after_first_batch(event):
    if event.kind == "batch_completed":
        token.cancel()

result = run(source, wider_config,
             reporter=CallbackProgressReporter(stop_after_first_batch),
             cancel=token)
print("cancelled:", result.cancelled, "newly completed:", result.succeeded)
```

```text
cancelled: True newly completed: ('AMD',)
```

## Where files land

All paths derive from one *data root*. It is `--data-dir` if you pass it, else
the `QUANTLAB_DATA_DIR` environment variable, else a `data/` directory at the
top of the repository. In Python, call `quantlab.config.set_data_root(path)`
before building any config, because configs record their paths when they are
created.

Beneath the root, raw downloads live under `downloads/{market}/{frequency}/`
and panels under `data/{market}/{frequency}/`. After the example's Tiingo-layout
download the tree looks like this:

```text
<root>/downloads/us_equity/1d/nasdaq_data/
    tiingo/month=2024-01/part-6f28d5ecbc46ad74-00000.pqt   raw shards
    tiingo/month=2024-02/...
    _watermarks/tiingo/AAPL.json                           one sidecar per symbol
    _watermarks/tiingo/_failures.json                      failure manifest
    _watermarks/tiingo/_pages/6f28d5ecbc46ad74.pages.json  page ledgers
<root>/data/us_equity/1d/stock.zarr                        the converted panel
```

Each parquet file is a *shard*: the rows of one request page for one partition.
Directories named `key=value` form a *hive* layout, so a reader can skip whole
months (daily data), days (minute data) or data-type, day and symbol
directories (tick data) without opening them. The bookkeeping lives in a
sibling `_watermarks/` directory rather than inside the raw tree, because a
polars directory scan reads every file under the root it is given.

Each vendor has its own directory, so two vendors' rows never mix. The default
locations are:

| Download | Raw tier | Panel |
|---|---|---|
| `ingest_tiingo.py` | `downloads/us_equity/1d/nasdaq_data/tiingo` | `data/us_equity/1d/stock.zarr` |
| `ingest_us_equity.py` | `downloads/us_equity/1d/us_all/tiingo` | `data/us_equity/1d/us_all.zarr` |
| `ingest_alpaca.py` | `downloads/us_equity/{1d,1m,tick}/nasdaq_data/alpaca` | `data/us_equity/{1d,1m}/stock_alpaca.zarr` |
| `ingest_wrds_crsp.py` | `downloads/us_equity/1d/wrds_crsp/wrds` | `data/us_equity/1d/wrds_crsp_*_1d.zarr` |
| `ingest_wrds_taq.py` | `downloads/us_equity/tick/wrds_taq/wrds` | `data/us_equity/tick/wrds_nbbo_*.zarr` |

## Resuming an interrupted download

You resume a download by running the same command again. Nothing else is
needed, because the state that drives resumption lives on disk.

A *watermark* is a small JSON sidecar per symbol, for example
`{"last_date": "2024-03-29", "start_date": "2024-01-01"}`, written after the
symbol's batch completes. Before fetching anything, a run compares every
requested symbol's sidecar with the requested window and skips the symbols
already covered. Running the example's first download a second time fetches
nothing:

```text
coverage: {'requested': 3, 'pending': 0, 'skipped': 3, 'covered': 3, 'widened': 0, 'legacy': 0, 'no_data': 0}
```

The counts mean: `covered`, already downloaded over the window; `widened`,
downloaded but starting later than the new `--start-date`, so fetched again;
`legacy`, a sidecar from an older version that records no start date;
`no_data`, the vendor was asked about this window and returned nothing, which
is recorded so the same window is not asked again (a longer window still is);
`pending`, what this run will fetch.

Vendors that split one request into several pages (Alpaca, and WRDS, where a
page is one trading day or one calendar year) also keep a *page ledger* per
batch under `_pages/`. It records which pages have landed and the token for the
next one, so an interrupted batch resumes at its next page rather than starting
over. Shard names are deterministic, so a page fetched twice overwrites itself
instead of duplicating rows.

Symbols that fail are listed with the reason in `_failures.json`. A failure in
one batch never stops the others. A failed symbol has no watermark, so the next
run tries it again. The manifest accumulates across runs; read it with
`SourceInspector.failures` (below).

Two situations need a deliberate choice. `download()` (the default) fills in
the requested window, while `--refresh` / `refresh()` starts each symbol from
its own last date and never widens the start. And sidecars written by older
versions of quantlab may lack a start date; by default they are skipped with a
warning on every run. `ingest_us_equity.py --stamp-legacy-watermarks
2016-01-01` records the start you know they were fetched from, issuing no
requests, and `--legacy-watermarks refetch` treats them as not covered instead.

## Rate limits and quotas

Vendors limit how fast you can ask, and they do not all mean the same thing by
it, so each vendor class decides how to read a refusal.

Tiingo answers HTTP 429 when the account's hourly request allocation is spent.
That affects every remaining symbol, so the run stops sending requests at once
(the progress bar reads `QUOTA EXHAUSTED -- draining, not fetching`),
`result.quota_aborted` is true, and a later run resumes. To have the run wait
instead, set `kwargs["wait_for_quota"] = True` or pass `--wait-for-quota` to
`ingest_us_equity.py`: it then sleeps `quota_wait_seconds` (default 3600) and
retries, at most `quota_max_waits` times (default 3).

Alpaca answers 429 when a per-minute ceiling is hit, which clears in seconds.
The batch that hit it waits `rate_limit_backoff_seconds` (default 5) and
retries, up to `rate_limit_max_retries` times (default 6), before it is
recorded as failed for the next run. The free plan allows about 200 historical
requests per minute and the paid plan about 10,000; at 200 per minute a
full-market daily backfill takes about 8 minutes but a full-market minute-bar
backfill takes about 50 hours.

WRDS has no request quota. Its limits are disk space, which the SQL volume
guard protects, and Duo two-factor prompts, which is why a WRDS run uses one
connection and stops instead of reconnecting (see [WRDS](wrds.md)).

## The volume guard

Before any vendor client is created, every ingest script estimates how much
the request will download and refuses a request that is too large. For the
REST vendors (Tiingo, Alpaca) the estimate is arithmetic over the roster and a
trading calendar, and three independent ceilings apply: 20 GiB of raw files,
50,000 requests, and 4 hours of wall-clock time at the assumed request rate.
Any one alone would let a bad case through: a tick request can pass the request
count and still fill the disk.

Tick volume cannot be derived from a calendar, so an Alpaca tick request must
say how many rows one symbol produces in one day (`--rows-per-symbol-day`,
measured by sampling one symbol-day). A year of quotes for two very liquid
names is refused, and the message suggests a narrower request that fits:

```text
$ uv run python scripts/ingest_alpaca.py --symbols AAPL,MSFT --frequency tick \
      --data-type quotes --rows-per-symbol-day 1000000 \
      --start-date 2024-01-01 --end-date 2024-12-31
...
ValueError: Refusing to fetch (explicit --symbols list) tick over 2024-01-01..2024-12-31: 2 symbol(s) x 253 trading day(s) x 1,000,000 row(s)/symbol-day = 506,000,000 row(s) -> 50,600 request(s), 28.27 GiB, 4.2 h at 200 req/min (batch_size=100, page_limit=10,000). Over the raw-bytes ceiling (28.27 GiB > 20.00 GiB, MAX_RAW_BYTES); over the request ceiling (50,600 > 50,000, MAX_ACQUISITION_REQUESTS); over the wall-clock ceiling (4.2 h > 4.0 h, MAX_ACQUISITION_WALL_CLOCK_HOURS) -- 1.4x the tightest ceiling. A narrowing that fits: the same window at <= 1 symbol(s) (a smaller --universe, e.g. an index-constituent category), or this roster over <= 258 calendar day(s) (2024-01-01..2024-09-14), which is ~35,600 request(s), ~19.89 GiB, ~3.0 h. Or raise that ceiling deliberately via the max_raw_bytes / max_requests / max_wall_clock_hours keyword (readable from config.kwargs), or pass force=True (--force-volume) to proceed anyway.
```

`--force-volume` skips the refusal for one run; the estimate is still printed.
There is intentionally no environment variable or config setting that turns
the guard off everywhere. The guard itself is
`quantlab.universe.UniverseCatalog.assert_acquisition_volume_fits`.

WRDS requests are priced differently, from real row counts that the server
returns for `count(*)` queries, by `SqlVolumeGuard` (20 GiB and 700 million
rows by default). The example script shows its refusal on made-up counts:

```text
refused: Refusing to pull 500 symbol(s) over 2024-01-22..2024-01-25: 4 trading day(s), 1,000,000,000 row(s) (counted with count(*)) x 30 B/row = 27.94 GiB. over the raw-bytes ceiling (27.94 GiB > 20.00 GiB; raise MAX_RAW_BYTES or the 'max_raw_bytes' kwargs key); over the raw-rows ceiling (1,000,000,000 > 700,000,000 rows; raise MAX_RAW_ROWS or the 'max_raw_rows' kwargs key). A date segment that fits: --start-date 2024-01-22 --end-date 2024-01-23; run the rest as further guarded segments ...
```

## Inspecting what is on disk

`SourceInspector` answers questions about a source's local files. It builds no
vendor client, so it needs no credentials and makes no network request. Each
method takes the config that locates the data.

```python
from quantlab.acquisition._support.inspector import SourceInspector

inspector = SourceInspector()
inspector.coverage(acq_config)            # what a download would skip or fetch
inspector.failures(acq_config)            # {symbol: reason} across all runs
inspector.inventory(acq_config, ds_config)  # shard, sidecar and store figures
inspector.browse_raw(ds_config, ["AAPL"], "2024-01-01", "2024-01-04")  # polars LazyFrame
inspector.browse_zarr(ds_config, ["AAPL"], "2024-01-02", "2024-01-04") # xarray Dataset
```

After the example's cancelled run, `coverage` reports that two of the six
requested symbols are still to fetch, and `inventory` summarises the raw tier:

```text
{'requested': 6, 'pending': 2, 'skipped': 4, 'covered': 4, 'widened': 0, 'legacy': 0, 'no_data': 0}
    root: <root>/downloads/us_equity/1d/nasdaq_data/tiingo
    shards: 12
    symbols_with_watermark: 4
    coverage_start: 2024-01-01
    coverage_last_date: 2024-03-29
    failures: 0
```

`browse_raw` and `browse_zarr` require a symbol list and a date window, so
what they return is always narrow. `browse_zarr` raises for a symbol the store
does not hold rather than returning an empty column, because an empty column
would look like a security with no history. `ingest_us_equity.py --dry-run`
prints the same coverage report for the full-market job.

## Converting raw downloads into a panel

Add `--to-zarr` to an ingest script to convert after downloading, or call
`quantlab.registry.convert(descriptor, dataset_config)` in Python. Conversion
reads only local files, so it needs no credentials. In the example it converts
the raw tier written above:

```python
from quantlab.config import stock_kline_config
from quantlab.registry import DataSourceRegistry, convert

ds_config = stock_kline_config(start_date="2024-01-01", end_date="2024-03-29",
                               symbols=("AAPL", "AMD", "MSFT", "NVDA"))
result = convert(DataSourceRegistry.get("tiingo"), ds_config, granularity="month")
print(result.windows_written, result.windows_planned, result.rows_written)
```

```text
    windows written: 3/3, rows: 65, store: <root>/data/us_equity/1d/stock.zarr
    dims: {'timestamp': 65, 'symbol': 4}
```

The conversion runs one time window at a time (`granularity`, or `--chunk`:
`year` by default, down to `day` or `hour`) and records each finished window,
so an interrupted or cancelled conversion resumes at the first unwritten
window. It accepts `reporter=` and `cancel=` like `run()`. Nothing checks that a
window fits in memory, so choose a finer granularity for a large roster or for
minute data.

When the roster has grown since the store was built (a new listing between two
refreshes), `--on-new-listing` decides what happens: `refuse` (the default)
stops and leaves the store untouched; `widen` adds the new symbols with empty
history, which is right for a genuine new listing; `rebuild` rebuilds every
window from the raw tier, which is right when a symbol already had history.
[Datasets](datasets.md) covers stores, backends and chunked conversion in
detail.
