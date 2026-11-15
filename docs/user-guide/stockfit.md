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
periods as StockFit serves, and the House and Senate trade feeds. Pass `endpoints=` to
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
- {py:class}`~quantlab.acquisition.stockfit.endpoints.SinceFeed`: a market-wide feed asked for
  everything filed since a day: the whole archive on the first pull, then from the last pull's day
  less `overlap_days` (30), so each pull holds only its window.

An endpoint of an existing shape is one row: a name (its directory), the path and the fixed query
parameters. A new shape is a subclass of
{py:class}`~quantlab.acquisition.stockfit.endpoints.Endpoint` that says which entities it is
pulled for and what query selects one.

## Congress trades

{py:class}`~quantlab.dataset.stockfit.congress.StockFitCongressTradesDataset` is a daily panel, on
the CIK axis and NYSE's trading days, of the stock trades members of Congress disclose under the
STOCK Act. It reads the feeds `congress_house_trades` (House Clerk, since 2014) and
`congress_senate_trades` (Senate eFD, since 2012), pulled with `since=` from the archive start and
then from the last pull's day.

| Variable | Meaning |
|---|---|
| `purchase_min`, `purchase_max` | Sums of the purchases' disclosed amount bands, lower and upper bounds (USD) |
| `sale_min`, `sale_max` | The same for sales and partial sales |
| `purchases`, `sales` | Number of transactions |
| `members_buying`, `members_selling` | Distinct members with a purchase or a sale |

- **Available date**: the report's filing date, on that trading day or the next one (reports carry
  no acceptance time). The transaction date, often weeks earlier, never places a trade.
- **Which trades count**: only those whose `linkConfidence` is `high` and whose `resolvedTicker`
  names exactly one company on the transaction date. Tickers are reused, so a ticker is resolved
  through the current and prior tickers of StockFit's company lists: a prior ticker's holder held it
  at least until its `lastSeen` and maybe up to 182 days after (StockFit's `lastSeen` is a lower
  bound), and a trade in that handover window, when another company holds the ticker later, is not
  placed. Unresolved, low-confidence and paper reports stay in the raw tier only.
- An open amount band (`Over $50,000,000`) adds its lower bound to both bounds. A trade an amendment
  repeats exactly is counted on its first report only. Every asset type StockFit resolved to a
  listing counts, options among them. A day without a counted trade is 0.
- **Look-ahead**: the company lists are read as of their newest pull, so the ticker history is
  today's view of it.

```python
from quantlab.dataset.config import StockFitCongressTradesConfig
from quantlab.dataset.stockfit.congress import StockFitCongressTradesDataset

config = StockFitCongressTradesConfig(
    zarr_file_path="/data/stores/stockfit_congress_trades_1d.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
)
StockFitCongressTradesDataset(config).update()
panel = StockFitCongressTradesDataset(config).panel("2024-01-02", "2024-03-28")
```
