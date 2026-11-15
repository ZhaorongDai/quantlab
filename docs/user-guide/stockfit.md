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

The same call does the first backfill and every later update; run it once a day. Each run:

1. pulls the market-wide lists: the company and fund lists (operating and delisted), the funds
   with daily holdings, the feed of the last 30 days' filings (`/api/filings/latest`, one entity
   per form type: 10-K, 10-Q, 20-F, Form 4, 13F-HR, N-PORT, 13D, 13G, DEF 14A), a snapshot of the
   declared split and dividend calendars, and the House and Senate trade feeds, each from its last
   pull's day;
2. plans the per-entity pulls due, in this order, within the requests left today:
   - the entities a filing made since their last pull touches: a new 10-K, 10-Q or 20-F re-pulls
     that company's statements and its XBRL split history, a new filing of any form the feed
     follows re-pulls its filing list and its profile, and a trust's new N-PORT re-pulls the
     N-PORT summary and flows of each of its series;
   - the entities never pulled, in the table's order, which is the backfill order (for the
     statements: income statement, balance sheet and cash-flow statement, annual, quarterly and
     TTM, with as many periods as StockFit serves);
   - the periodic re-pulls, which catch a filing the feed missed or does not follow: every
     company's statements, filing list, profile and split history once a week; each fund series'
     N-PORT, fee and performance endpoints every 30 days, its N-CEN endpoints and the adviser
     lookup every 90 days, and its daily holdings once a day.

The pulls that do not fit in the day's requests are deferred, and the next run plans them again
from what is on disk, so a backfill that needs weeks of requests resumes where it stopped. Pass
`endpoints=` to {py:func}`quantlab.acquisition.stockfit.refresh.refresh_raw_tier` to refresh some
endpoints only, and `ciks=` to pull some companies only.

StockFit limits requests per minute and per day, and the limits depend on the key. The client
reads them from the `X-RateLimit-*` headers of every response: it waits for the minute window to
have room, waits out an HTTP 429 and sends again (a throttle never fails a run), and stops sending
once the day's requests are spent (`day_reserve=` keeps some unspent). Running out of the day's
requests is not a failure: each endpoint's part of the result says how many pulls it deferred to
the next run. A 404 means StockFit holds nothing for that entity; nothing is published.

A part that fails (an endpoint the plan does not include, an entity whose request kept failing) is
listed in `result.failed` and the other endpoints still run.

## Corporate event panels

Two panels in {py:mod}`quantlab.dataset.stockfit.events` turn StockFit's corporate events into
daily panels on NYSE's trading calendar, one column per company CIK. **None of these events is
used to adjust a price**, and quantlab takes no price from StockFit. A split ratio here is a
feature, not an adjustment factor. Price adjustment comes from the price vendor.

{py:class}`~quantlab.dataset.stockfit.events.StockFitCorporateEventsDataset` shows each event on
the day it became public and NaN on every other day:

| Variable | Source endpoint | Shown on |
|---|---|---|
| `split_ratio`, `split_date` | `/api/company/stock-splits` (XBRL split history) | the trading day of `firstDisclosedAt` |
| `predecessor_cik`, `successor_cik` | `/api/company/details` (`predecessors`, `successors`; Form 8-K12B) | the trading day of `effectiveDate` |

A historical split is placed at `firstDisclosedAt`, the acceptance day of the first 10-Q or 10-K
that tagged it in XBRL, never at its split `date`. Filers tag a split weeks to a year after it
happens, so a split dated 1 March and first tagged on 30 April shows on 30 April. Before that, a
backtest could not have read it from EDGAR. `split_date` carries the split's own date, as the
filer tagged it (usually the ex-date, sometimes the record or announcement date). Several splits
first tagged on one day are multiplied. The history covers XBRL filings only, so it starts around
2010. A succession shows the other company's CIK on both companies' columns, on its effective day.
StockFit lists few successions, so expect this part to be sparse.

{py:class}`~quantlab.dataset.stockfit.events.StockFitDeclaredEventsDataset` shows the splits and
dividends that companies have declared but not yet executed. It reads the forward calendars
`/api/company/upcoming-stock-splits` and `/api/earnings/upcoming-dividends`, which look 28 days
ahead. The refresh snapshots them every time it runs. On each trading day, a company shows the
soonest declared split (`declared_split_ratio`, `declared_split_date`) and the soonest declared
dividend (`declared_dividend` in USD per share, summed over the dividends going ex that day,
`declared_dividend_annual`, `declared_dividend_ex_date`). These come from the newest snapshot
available that day, and an event stays shown through its own date. A snapshot counts on the day
it was pulled if it was pulled before that session's close (13:00 New York time on a half day).
Otherwise it counts on the next trading day. Calendar rows name tickers, not CIKs, so each row is
matched to a CIK through the operating company list the raw tier held when the snapshot was
pulled. A row whose tickers name no company, or name more than one, is left out. StockFit keeps
no archive of past declarations, so this panel has a history only from the first snapshot on.

Both panels are built through the last day every endpoint they read was pulled:

```python
from quantlab.dataset.config import StockFitCorporateEventsConfig
from quantlab.dataset.stockfit.events import StockFitCorporateEventsDataset

config = StockFitCorporateEventsConfig(
    zarr_file_path="/data/stores/stockfit_corporate_events_1d.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
    start_date="2010-01-04",
)
panel = StockFitCorporateEventsDataset(config).update().panel("2020-01-02", "2024-12-31")
panel["split_ratio"].sel(symbol=1652044).dropna("timestamp")
```

## Adding an endpoint

Endpoints are data: each is a row of `quantlab.acquisition.stockfit.endpoints.ENDPOINTS`, an
instance of the class for its *pull shape*:

- {py:class}`~quantlab.acquisition.stockfit.endpoints.MarketList`: a market-wide paged list, one
  entity per named variant of its query;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.PerCompany`: one request per company,
  selected by `cik`;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.LatestFilings`: the recent-filings feed, one
  entity per form type;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.PerFundSeries`: one request per fund series,
  selected by its ticker or Composite FIGI;
  {py:class}`~quantlab.acquisition.stockfit.endpoints.PerFundQuarter`: one per series and quarter,
  each asked once; {py:class}`~quantlab.acquisition.stockfit.endpoints.PerAdviser`: one per
  adviser CRD;
- {py:class}`~quantlab.acquisition.stockfit.endpoints.SinceFeed`: a market-wide feed asked for
  everything filed since a day: the whole archive on the first pull, then from the last pull's day
  less `overlap_days` (30), so each pull holds only its window. It is a market-wide list, pulled on
  every run.

An endpoint of an existing shape is one row: a name (its directory), the path and the fixed query
parameters. Its place in the table is its place in the backfill. A market-wide list is pulled on
every run; any other endpoint is backfilled once per entity and then re-pulled when a filing of one
of its `triggers` form types is made for the entity, and, with `repull_days`, once its last pull is
that many days old. A new shape is a subclass of
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
| `fund_holdings_daily` | `/api/fund/holdings/daily` (issuer websites) | raw only, a snapshot a day |
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
