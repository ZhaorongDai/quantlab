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
income statement, balance sheet and cash-flow statement, annual, quarterly and TTM, and the 17
footnote tables, annual and quarterly, with as many periods as StockFit serves. Pass `endpoints=` to
{py:func}`quantlab.acquisition.stockfit.refresh.refresh_raw_tier` to pull some endpoints only, and
`ciks=` to pull some companies only.

StockFit limits requests per minute and per day, and the limits depend on the key. The client
reads them from the `X-RateLimit-*` headers of every response: it waits for the minute window to
have room, retries an HTTP 429 after the window rolls, and stops sending once the day's requests
are spent (`day_reserve=` keeps some unspent), so a refresh stops cleanly instead of failing on the
cap. A 404 means StockFit holds nothing for that entity; nothing is published.

A part that fails (an endpoint the plan does not include, the daily cap, an entity whose request
kept failing) is listed in `result.failed` and the other endpoints still run.

## Footnote panels

StockFit parses 17 footnote tables out of 10-K, 10-Q and 20-F filings: revenue and business
segmentation, debt structure, credit facilities, leases, stock compensation, concentration,
retirement plans, restructuring, asset composition, buybacks, subsidiary financials, the
fair-value hierarchy, REIT properties, backlog, supplier finance and workforce cost.
{py:class}`~quantlab.dataset.stockfit.footnotes.StockFitFootnoteDataset` builds one daily panel
per footnote and period type on the CIK axis:

```python
from quantlab.dataset.config import StockFitFootnoteConfig
from quantlab.dataset.stockfit.footnotes import StockFitFootnoteDataset

config = StockFitFootnoteConfig(
    zarr_file_path="/data/factors/stockfit_leases_quarter.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
    footnote="leases",      # a key of FOOTNOTES
    period="quarter",       # or "annual"
)
dataset = StockFitFootnoteDataset(config)
dataset.update()
dataset.panel("2024-01-01", "2024-12-31")["operating_liability_total"]
```

A panel holds the footnote's fixed fields and StockFit's own rollups (the debt totals by debt type,
the fair-value `level3Share`, the concentration shares tagged without a named counterparty, the
whole-plan view of stock compensation), plus `period_end`, `filing_date`, `fiscal_year` and
`may_include_restatement`. Per-member lists (segments, tranches, facilities, counterparties,
programs, subsidiaries, properties) stay in the raw tier, and quantlab computes no summary of them
(member counts, largest shares, concentration indices): those are factors. Business segmentation,
credit facilities and REIT properties have no fixed field, so their panels say only which period is
the latest known each day.

What a panel shows is data: each footnote is a
{py:class}`~quantlab.dataset.stockfit.footnotes.FootnoteSpec` in
`quantlab.dataset.stockfit.footnotes.FOOTNOTES`, a path per variable into the response body. A
step is a key, a key with a selector (`measures[measure=assets]` picks the first element whose
fields match; `null` matches a missing field) or `[]`, the list of periods. Adding a field is one
line:

```python
"backlog": FootnoteSpec(
    "/api/footnotes/backlog",
    {
        "remaining_performance_obligation": "[].remainingPerformanceObligation",
        "expected_share_in_window": "[].expectedShareInWindow",
    },
),
```

**Point in time.** A period's values are shown from its `dateFiled` (the newest filing that
reported it, on that day; footnotes carry no acceptance time) until a newer period replaces them,
on NYSE's trading calendar; a restated older period never replaces a newer one, and a period
released more than `stale_after_days` (400 by default) before the day is not shown. StockFit keeps
no restatement history for footnotes: it serves each period's latest values. **Values backfilled
from a company's first pull may therefore include restatements made after the filing date they are
placed at**; the panel flags them with `may_include_restatement = 1`. From the first pull on, each
pull is compared with the one before: a period that is new or changed is shown from its filing date
when that filing came after the previous pull, and from the pull otherwise (a pull made at or after
the session's close counts from the next trading day), with the flag at 0. That part of the history
is truly point in time, and it grows with every refresh.

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
