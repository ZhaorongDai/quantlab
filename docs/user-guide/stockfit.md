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

Every StockFit panel places a row at its *available date*, the first day a close-based backtest may
use it, by one rule ({py:func}`quantlab.dataset.stockfit.placement.available_dates`):

- a row with an EDGAR acceptance time (`acceptedAt`) is available on its New York day when it was
  accepted before that day's NYSE close (16:00, or 13:00 on a half day), and on the next NYSE trading
  day when it was accepted at or after the close, or on a day without a session;
- a row with only a filing date (`dateFiled`) is available on that day (the next trading day when
  it is not one), as in the Sharadar panels.

So a 10-Q accepted at 16:30 is not on that day's bar; one accepted at 15:59 is. The panels use
NYSE's trading calendar, through the day before the raw tier's newest pull.

## Statement panels

{py:class}`~quantlab.dataset.stockfit.statements.StockFitStatementsDataset` builds a daily panel of
the three statements' 178 facts on the CIK axis from the quarterly pulls
(`income_statement_quarter`, `balance_sheet_quarter`, `cash_flow_statement_quarter`). One store holds
one `period` and one `reported` set:

| `period` | Each day shows |
|---|---|
| `"quarter"` | the latest fiscal quarter available by that day |
| `"ttm"` | the four latest quarters, once they are four consecutive fiscal quarters: flows summed, weighted share counts averaged, balance-sheet items and `endCash` from the latest quarter, `beginningCash` from the earliest |

| `reported` | Each value is |
|---|---|
| `"as_known"` | as known that day: a restatement shows from its own available date, never earlier |
| `"original"` | as first reported; it never changes after its first filing |

StockFit keeps, per fact, which filing supplied each value and the value each later filing replaced
(`sources[accession].facts[fact].before`). The panel rebuilds every version of a fact from that chain
and places each at its filing's available date. Amendments are filings like any other; the value of
a pruned filing, and a reconstructed quarter (Q4 from the fiscal year less nine months, cash-flow Q2
and Q3 from year-to-date values), are dated at the period's original filing. A later filing of an
older quarter never replaces a newer one, and a quarter no newer one replaced stops being shown
`stale_after_days` (365) after its filing date.

Facts keep the vendor's names; the five the cash-flow statement shares with the income statement
are prefixed (`cashFlowNetIncome`, `cashFlowDepreciation`, `cashFlowAmortization`,
`cashFlowDepreciationAndAmortization`, `cashFlowImpairments`). `release_date` (the shown quarter's
original filing date) and `period_end` (its fiscal period end) say which quarter a cell shows.

```python
from quantlab.dataset.config import StockFitStatementsConfig
from quantlab.dataset.stockfit.statements import StockFitStatementsDataset

config = StockFitStatementsConfig(
    zarr_file_path="/data/market/stockfit/stockfit_statements_ttm/stockfit_statements_ttm.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
    period="ttm",
)
panel = StockFitStatementsDataset(config).update().panel("2024-01-02", "2024-12-31")
panel["revenue"].sel(symbol=1093691)  # Plug Power's trailing revenue, as known each day
```

Annual statements, and StockFit's own TTM response (its latest period only), stay in the raw tier.

Look-ahead caveats:

- A fact StockFit computes from others (listed in a period's `derived`, such as `ebitda` or
  `freeCashFlow`) has no restatement chain: the panel shows its latest value from the period's first
  filing, so a derived fact of a later-restated period may show the restated value early.
- The raw tier is taken as complete through the day before its newest statement pull: a company not
  re-pulled since is taken to be unchanged.
