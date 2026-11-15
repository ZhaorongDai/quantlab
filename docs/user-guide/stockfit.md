# StockFit

[StockFit](https://api.stockfit.io) serves SEC-filing data parsed from EDGAR as JSON: normalized
statements with their restatement history, footnote tables, ownership filings, fund filings and
filing calendars. quantlab treats it as a vendor that stands alone: a `STOCKFIT_API_KEY` is all it
needs, and its data is keyed by the SEC's CIK, not by another vendor's symbols. quantlab takes no
prices from StockFit.

## Credentials

Set the key in the environment; nothing reads it from a file:

```bash
export STOCKFIT_API_KEY=...
```

It is sent only as an `Authorization: Bearer` header, never in a URL, and is kept out of every log
line and error message.

## The raw tier

StockFit has no bulk files, so its raw tier is built request by request: one pull of one endpoint
for one *entity* (a company's CIK, or a market-wide list) at a time. Each pull is a directory named
by its UTC pull stamp:

```text
<download-dir>/stockfit/symbols/operating/20261116T140000000000/000001.json
<download-dir>/stockfit/symbols/operating/20261116T140000000000/_pull.json
<download-dir>/stockfit/income_statement_quarter/1093691/20261116T140000000000/000001.json
<download-dir>/stockfit/income_statement_quarter/1093691/_watermark.json
```

A pull directory holds each page's response body as StockFit returned it and a manifest
(`_pull.json`) with the query, the page count and a content hash. Three rules keep the tier
trustworthy:

- **A pull appears whole or not at all.** It is written to the entity's `_pending` directory and
  published by one directory rename, so a killed process never leaves a half-written pull that a
  reader takes as whole.
- **Only changes are kept.** A pull whose content equals the entity's newest pull is dropped; only
  the entity's watermark (the UTC day it was last pulled) moves. Every changed pull is kept, so the
  pulls of an entity are the versions StockFit served over time.
- **One module owns the layout.** {py:mod}`quantlab.dataset.stockfit.raw` defines every path and
  reads the tier ({py:func}`~quantlab.dataset.stockfit.raw.read_pull`,
  {py:func}`~quantlab.dataset.stockfit.raw.read_pull_as_of`,
  {py:func}`~quantlab.dataset.stockfit.raw.pulls`,
  {py:func}`~quantlab.dataset.stockfit.raw.company_ciks`); the writer imports it.

```python
from quantlab.dataset.stockfit import raw

root = "/data/downloads/stockfit"
raw.company_ciks(root)[:3]
versions = raw.pulls(root, "income_statement_quarter", "1093691")
latest = raw.read_pull(root, "income_statement_quarter", "1093691")
```

## Refreshing

One call keeps the raw tier current, through the registry or directly:

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import refresh_raw_tier

result = refresh_raw_tier(DataSourceRegistry.get("stockfit"), "/data/downloads")
print(result.summary())
```

It pulls the company lists (operating and delisted companies) and then, for every company, the
income statement, balance sheet and cash-flow statement, annual, quarterly and TTM, with as many
periods as StockFit serves. Pass `endpoints=` to
{py:func}`quantlab.acquisition.stockfit.refresh.refresh_raw_tier` to pull some endpoints only, and
`ciks=` to pull some companies only.

StockFit limits requests per minute and per day, and the limits depend on the key. The client
reads them from the `X-RateLimit-*` headers of every response: it waits for the minute window to
have room, retries an HTTP 429 after the window rolls, and stops sending once the day's requests
are spent (`day_reserve=` keeps some unspent), so a refresh stops cleanly instead of failing on the
cap. A 404 means StockFit holds nothing for that entity; nothing is published.

A part that fails (an endpoint the plan does not include, the daily cap, an entity whose request
kept failing) is listed in `result.failed` and the other endpoints still run.

## Adding an endpoint

Endpoints are data: each is a row of `quantlab.acquisition.stockfit.endpoints.ENDPOINTS`, an
instance of the class for its *pull shape*:

- {py:class}`~quantlab.acquisition.stockfit.endpoints.MarketList`: a market-wide paged list, one
  entity per named variant of its query;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.PerCompany`: one request per company,
  selected by `cik`.

An endpoint of an existing shape is one row: a name (its directory), the path and the fixed query
parameters. A new shape is a subclass of
{py:class}`~quantlab.acquisition.stockfit.endpoints.Endpoint` that says which entities it is
pulled for and what query selects one.

## Available dates

Every StockFit panel places a row on the NYSE trading day it became public, by one rule
({py:mod}`quantlab.dataset.stockfit.available`):

- a row with the SEC's acceptance time (`acceptedAt`) is available on the day it was accepted when
  accepted before that session's close (16:00 New York time, 13:00 on an early close), and on the
  next trading day when accepted at or after the close, on a weekend or on a holiday;
- a row with only a filing date (`dateFiled`) is available on that day, or the next trading day
  when it is not one.

So a close-based backtest never trades on a filing made after the close. A panel's days are NYSE's
trading days through the day before the newest pull of the endpoint it reads: a refresh made on a
day holds every filing public by the previous close.

## The filing panel

The refresh pulls each company's filing list (`/api/filings`, every form, newest first) into
`filings/<cik>/`. {py:func}`quantlab.dataset.stockfit.filings.read_filings` returns the newest list
of every company as one table (CIK, accession number, form type, filing date, acceptance time,
period, amendment flag, 8-K events), the table accession lookups read. Item text stays raw.

{py:class}`~quantlab.dataset.stockfit.filings.StockFitFilingsDataset` turns it into a daily panel
by CIK, each filing on its available date only; a day without a filing holds 0 and `False`:

| Variable | Meaning |
|---|---|
| `form_10_k`, `form_10_q`, `form_8_k`, `form_def_14a`, `form_13f_hr`, ... | Filings of each form StockFit names (17 forms), amendments included |
| `form_other` | Filings of any other form (Form 144, S-1, ...) |
| `event_earnings`, `event_officer_change`, ... | An 8-K of the day lists the event (StockFit's 31 8-K events; `earnings` is Item 2.02) |
| `amendment` | A filing of the day is an amendment |

```python
from quantlab.dataset.config import StockFitFilingsConfig
from quantlab.dataset.stockfit.filings import StockFitFilingsDataset

config = StockFitFilingsConfig(
    zarr_file_path="/data/market/stockfit/stockfit_filings_1d/stockfit_filings_1d.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
    start_date="2000-01-01",
)
StockFitFilingsDataset(config).update()
panel = StockFitFilingsDataset(config).panel("2026-10-01", "2026-10-31")
panel["event_earnings"].sel(symbol=1652044)  # Alphabet's CIK
```

StockFit's filing lists have gaps (a company's early 10-Ks can be missing, and a delisted company
may list only a few filings), so a zero means StockFit lists no filing that day, not that none was
made.
