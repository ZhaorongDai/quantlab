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
  selected by `cik`;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.PerFundSeries`: one request per fund series,
  selected by its ticker or Composite FIGI;
  {py:class}`~quantlab.acquisition.stockfit.endpoints.PerFundQuarter`: one per series and quarter,
  each asked once; {py:class}`~quantlab.acquisition.stockfit.endpoints.PerAdviser`: one per
  adviser CRD;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.SinceFeed`: a market-wide feed asked for
  everything filed since a day: the whole archive on the first pull, then from the last pull's day
  less `overlap_days` (30), so each pull holds only its window.

An endpoint of an existing shape is one row: a name (its directory), the path and the fixed query
parameters. A new shape is a subclass of
{py:class}`~quantlab.acquisition.stockfit.endpoints.Endpoint` that says which entities it is
pulled for and what query selects one.

## Fund series

StockFit's fund data (N-PORT, N-1A, N-CSR, N-CEN and the issuers' daily ETF holdings) is keyed by
the *fund series*: one series of a fund trust, named by the trust's CIK and the series id,
`<cik>-<series id>` (the Vanguard 500 Index Fund is `36405-2839`). A trust files for many series,
so two series of one trust are two symbols; an ETF's share class is a security, not a series.

The refresh first pulls the fund lists (`fund_symbols`: `/api/symbols` for each fund kind, `etf`,
`mf`, `cef` and `fund`, operating and delisted). StockFit's fund endpoints take a security selector,
not a series id, so each series is asked for by its first ticker, or by its Composite FIGI when it
has no ticker; a series with neither is not pulled. Then, per series:

| Endpoint (raw-tier name) | StockFit path | Kept as |
|---|---|---|
| `fund_profile` | `/api/fund/profile` (N-PORT summary) | panel |
| `fund_flows` | `/api/fund/flows` (N-PORT monthly flows) | panel |
| `fund_fees` | `/api/fund/fees` (N-1A prospectus fee table) | panel |
| `fund_health`, `fund_fee_analysis` | `/api/fund/fund-health`, `/api/fund/fee-analysis` (N-CEN) | panel of the per-period fields |
| `fund_performance` | `/api/fund/performance` (N-CSR, per share class) | raw only |
| `fund_holdings` | `/api/fund/holdings` (N-PORT holdings) | raw only, one entity per series and quarter (`<series>_<YYYY-MM-DD>`) |
| `fund_holdings_daily` | `/api/fund/holdings/daily` (issuer websites) | raw only, a snapshot per refresh |
| `fund_service_providers` | `/api/fund/service-providers` (N-CEN, up to 40 years) | raw only |
| `fund_by_adviser` | `/api/fund/by-adviser` | raw only, one entity per adviser CRD |

N-PORT holdings are asked for each quarter the series' N-PORT summary lists, once: a quarter
already asked for is not asked again. Daily holdings are pulled for the series with a ticker in
StockFit's supported list (`fund_daily_supported`); StockFit keeps only about 45 days of them, so
the history is the raw tier's own: an unchanged day publishes nothing, a changed one adds a pull.
The adviser lookup is pulled for the CRD numbers of the advisers in the series' service providers.
Pass `fund_series=` to the refresh to pull some series only.

```python
from quantlab.dataset.stockfit import raw

root = "/data/downloads/stockfit"
raw.fund_series(root)[:3]
for at in raw.pulls(root, "fund_holdings_daily", "36405-2839"):
    snapshot = raw.read_pull(root, "fund_holdings_daily", "36405-2839", at)
```

### Fund panels

Four datasets in {py:mod}`quantlab.dataset.stockfit.funds` turn the newest pull of every series
into a daily panel on NYSE's sessions, the fund series as symbols, each built from a
{py:class}`~quantlab.dataset.config.StockFitFundConfig` whose `raw_data_dir_path` is the StockFit
raw tier:

| Dataset | Variables | Available from |
|---|---|---|
| {py:class}`~quantlab.dataset.stockfit.funds.StockFitFundProfileDataset` | total assets, liabilities, net assets, three monthly returns, sales, redemptions, holding count | quarter end + 60 days |
| {py:class}`~quantlab.dataset.stockfit.funds.StockFitFundFlowsDataset` | sales, reinvestments and redemptions of each month of the quarter, monthly returns, net and total assets | quarter end + 60 days |
| {py:class}`~quantlab.dataset.stockfit.funds.StockFitFundFeesDataset` | management, 12b-1 and other fees, total and net expense ratio, waiver, sales charges, redemption fee | the filing date |
| {py:class}`~quantlab.dataset.stockfit.funds.StockFitFundNcenDataset` | index tracking, securities lending, red flags, expense limits, commissions, in-kind creation (yes/no as 1.0/0.0) | reporting period end + 75 days |

Each day shows the series' latest row available by then, and the row's own date (`report_date`
or `filed_date`). A date that falls on a non-session rolls to the next session. The days run to
the day before the panel's endpoints were last pulled.

```python
from quantlab.dataset.config import StockFitFundConfig
from quantlab.dataset.stockfit.funds import StockFitFundProfileDataset

config = StockFitFundConfig(
    zarr_file_path="/data/market/stockfit/stockfit_fund_profile_1d.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
    start_date="2019-01-02",
)
panel = StockFitFundProfileDataset(config).update().panel("2024-01-02", "2024-12-31")
panel["net_assets"].sel(symbol="36405-2839")
```

What the panels are safe for:

- **N-PORT and N-CEN are placed at their deadlines.** StockFit serves no filing date for them, so
  a quarter shows from quarter end + 60 days (N-PORT) or period end + 75 days (N-CEN), the SEC
  deadlines. Funds often file earlier, so a value can appear some days after it was public, never
  before.
- **Amendments replace the original.** Values are as StockFit served them on the pull day: a
  backfilled quarter shows an amended N-PORT or N-CEN from the original's deadline. From the first
  pull on, a changed value is a new pull, so the raw tier keeps every version seen.
- **Fees are one share class's.** The fee table is that of the share class whose ticker the series
  was asked for by.

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
