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

## 13F institutional ownership

The refresh pulls 13F data last, since it is by far the largest part:

- `institutional_holders`: for every company and calendar quarter since 2013, every 13F holder row
  of the company's securities (one row per manager and CUSIP, every share class), entity
  `<cik>_<quarter end>`, newest quarter first;
- `manager_13f_filings`: for every manager those rows name, its 13F filings (originals and
  amendments) with their filing date and acceptance time;
- `manager_portfolio`: for every manager and quarter its filings report, its whole portfolio. These
  stay raw files; no panel reads them.

{py:class}`~quantlab.dataset.stockfit.holdings.StockFitHoldingsDataset` sums the holder rows by
security and quarter into a daily panel on NYSE's trading days. Its symbol is the 9-character
CUSIP, kept as text, so the share classes of one company are separate symbols. Only share holdings
count (no principal amounts, no options). Variables: `holders` (managers holding the security),
`shares_held`, `value_held` (USD, as reported for the quarter end) and `quarter_end`.

Available dates:

- A quarter is shown from the first trading day on or after its end + 45 days until the next
  quarter's, the same quarter for every security. Before day 45 nothing of it is shown, and a
  security with no row in the shown quarter shows nothing.
- Within the shown quarter a row counts from the later of that day and its filing's available date.
  A filing accepted before that day's NYSE close (13:00 on a half day) is available that day, one
  accepted at or after the close on the next trading day; without an acceptance time, on its filing
  date. A row whose filing is in no pulled filing list counts from the day after its holder rows
  were pulled.

Look-ahead caveat: StockFit merges an amendment (13F-HR/A) into the quarter it amends and gives the
amendment's accession on the row. The original value is never shown, and a position first
disclosed in a late amendment (a confidential stake, say) is missing from its quarter until the
amendment's date, or entirely when that quarter is no longer shown by then.

```python
from quantlab.dataset.config import StockFitHoldingsConfig
from quantlab.dataset.stockfit.holdings import StockFitHoldingsDataset

config = StockFitHoldingsConfig(
    zarr_file_path="/data/market/stockfit/stockfit_holdings_1d/stockfit_holdings_1d.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
)
StockFitHoldingsDataset(config).update()
panel = StockFitHoldingsDataset(config).panel("2024-01-02", "2024-12-31")
panel["holders"].sel(symbol="02079K305")  # Alphabet class A
```

## Adding an endpoint

Endpoints are data: each is a row of `quantlab.acquisition.stockfit.endpoints.ENDPOINTS`, an
instance of the class for its *pull shape*:

- {py:class}`~quantlab.acquisition.stockfit.endpoints.MarketList`: a market-wide paged list, one
  entity per named variant of its query;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.PerCompany`: one request per company,
  selected by `cik`;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.PerCompanyQuarter`: one request per company
  and calendar quarter (`cik` and `reportDate`), entity `<cik>_<quarter end>`;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.PerManager`: one request per 13F manager
  named in the raw tier's holder rows;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.PerManagerQuarter`: one request per 13F
  manager and quarter its 13F filings report.

An endpoint of an existing shape is one row: a name (its directory), the path and the fixed query
parameters. A new shape is a subclass of
{py:class}`~quantlab.acquisition.stockfit.endpoints.Endpoint` that says which entities it is
pulled for and what query selects one.
