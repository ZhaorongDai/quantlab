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
<root>/downloads/stockfit/symbols/operating/20261116T140000000000/000001.json
<root>/downloads/stockfit/symbols/operating/20261116T140000000000/_pull.json
<root>/downloads/stockfit/income_statement_quarter/1093691/20261116T140000000000/000001.json
<root>/downloads/stockfit/income_statement_quarter/1093691/_watermark.json
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
     that company's statements, footnotes and XBRL split history, a new filing of any form the feed
     follows re-pulls its filing list and its profile, and a trust's new N-PORT re-pulls the
     N-PORT summary and flows of each of its series;
   - the entities never pulled, in the table's order, which is the backfill order (the
     statements: income statement, balance sheet and cash-flow statement, annual, quarterly and
     TTM; then the 17 footnote tables, annual and quarterly; each with as many periods as
     StockFit serves);
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
    zarr_file_path="/data/market/stockfit/stockfit_leases_quarter/stockfit_leases_quarter.zarr",
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
    zarr_file_path="/data/market/stockfit/stockfit_corporate_events_1d/stockfit_corporate_events_1d.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
    start_date="2010-01-04",
)
panel = StockFitCorporateEventsDataset(config).update().panel("2020-01-02", "2024-12-31")
panel["split_ratio"].sel(symbol=1652044).dropna("timestamp")
```

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
  manager and quarter its 13F filings report;
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
| `"ttm"` | the four latest quarters, once they are four consecutive fiscal quarters: flows summed (missing when one of the four lacks the flow, never a partial sum), weighted share counts averaged, balance-sheet items and `endCash` from the latest quarter, `beginningCash` from the earliest |

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

To read a company panel on Sharadar's permatickers, wrap it in the CIK map
({py:class}`~quantlab.dataset.sharadar.cik_map.CikMapDataset`; see the Sharadar guide).

Look-ahead caveats:

- A fact StockFit computes from others (listed in a period's `derived`, such as `ebitda` or
  `freeCashFlow`) has no restatement chain: the panel shows its latest value from the period's first
  filing, so a derived fact of a later-restated period may show the restated value early.
- The raw tier is taken as complete through the day before its newest statement pull: a company not
  re-pulled since is taken to be unchanged.

## As-reported facts, derived metrics, share classes, scores and trends

These endpoints are re-pulled when the company files a new 10-K, 10-Q or 20-F (scores and trends
also once a month). Unlike the statements, none of them keeps the values it replaced: each pull is
what StockFit serves today. Their panels share one rule
({py:mod}`quantlab.dataset.stockfit.period_panel`):

- Values from a company's **first pull** are placed at the filing date of their fiscal period, but
  they **may include restatements made after that date**. The panel flags them with
  `may_include_restatement = 1`.
- From the first pull on, each pull is compared with the one before. A value that a later pull adds
  or changes is shown from its filing date when that filing came after the previous pull, and
  otherwise from the day of the pull (the vendor changed it silently). Such values are not flagged
  (`may_include_restatement = 0`), so from a company's first pull on the history is point in time.
- Each day shows the latest fiscal period available by then. `release_date` and `period_end` say
  which period that is. A later filing of an older period never replaces a newer one.

| Dataset | Endpoints | What a store holds | Placed at |
|---|---|---|---|
| {py:class}`~quantlab.dataset.stockfit.as_reported.StockFitAsReportedDataset` | `as_reported_quarter`, `as_reported_annual` | the chosen us-gaap concepts (`concepts=`), each named without its prefix (`Revenues`) | the period's own filing |
| {py:class}`~quantlab.dataset.stockfit.derived.StockFitDerivedDataset` | `growth_quarter`, `key_metrics_quarter` (the `general` block), `eps_history_quarter`, `dividend_history_quarter` (per-share values as filed, not split-adjusted), `share_classes` | one `table=` | the filing of the same CIK and period in the statements |
| {py:class}`~quantlab.dataset.stockfit.scores_trends.StockFitScoresTrendsDataset` | `scores`, `trends` | one `table=` | the day of the pull |

The as-reported concept list is a config parameter, so a store's variables stay fixed. Regenerate the
usual list, the us-gaap concepts that at least 5% of companies use, from the raw tier with
{py:func}`~quantlab.dataset.stockfit.as_reported.as_reported_concepts`. Filer extensions stay in the
raw tier.

```python
from quantlab.dataset.config import StockFitAsReportedConfig
from quantlab.dataset.stockfit.as_reported import StockFitAsReportedDataset, as_reported_concepts

root = "/data/downloads/stockfit"
config = StockFitAsReportedConfig(
    zarr_file_path="/data/market/stockfit/stockfit_as_reported_quarter/stockfit_as_reported_quarter.zarr",
    raw_data_dir_path=root,
    concepts=as_reported_concepts(root, period="quarter", min_share=0.05),
)
panel = StockFitAsReportedDataset(config).update().panel("2024-01-02", "2024-12-31")
panel["NetIncomeLoss"].where(panel["may_include_restatement"] == 0)  # point-in-time values only
```

Derived metrics and share classes carry no filing date of their own. A row whose CIK and period end
match no statement period is dropped, and the build counts it in the dataset's `build_report`:

```python
from quantlab.dataset.config import StockFitDerivedConfig
from quantlab.dataset.stockfit.derived import StockFitDerivedDataset

dataset = StockFitDerivedDataset(StockFitDerivedConfig(
    zarr_file_path="/data/market/stockfit/stockfit_growth_quarter/stockfit_growth_quarter.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
    table="growth",
)).update()
dataset.build_report  # DerivedBuildReport(rows=..., unmatched=...)
```

StockFit serves share classes for the latest period only, so `share_classes` (`class_count`,
`shares_outstanding`, `shares_authorized`, summed over the classes) gains one period per pull; the
per-class detail stays in the raw tier. Scores and trends have no history at all: each pull's values
show from the day of the pull, and nothing shows before a company's first pull. Their text
categories are coded: the Altman zone as `distress` 0, `grey` 1, `safe` 2; a margin direction as
`compressing` -1, `stable` 0, `expanding` 1; each Piotroski criterion as 1 when met. The sector and
industry blocks of the key metrics vary by company and stay in the raw tier.

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
    zarr_file_path="/data/market/stockfit/stockfit_fund_profile_1d/stockfit_fund_profile_1d.zarr",
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
    zarr_file_path="/data/market/stockfit/stockfit_congress_trades_1d/stockfit_congress_trades_1d.zarr",
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

## Insider transactions

The refresh pulls each company's insider transactions (`/api/insider-transactions`, every page)
into `insider_transactions/<cik>/`, and re-pulls a company when it gets a new Form 4.
{py:class}`~quantlab.dataset.stockfit.insiders.StockFitInsidersDataset` has the variables of the
Sharadar insider panel, with the same names and signs, on the CIK axis: `net_shares` and
`net_value` (open-market purchases minus sales), `net_shares_<code>` and `net_value_<code>` per
transaction code (`P`, `S`, `A`, `M`, `F`, `G`, `D`, `other`), `net_shares_derivative`,
`insiders_buying` and `insiders_selling`. Each line is placed on its filing's available date
(never the transaction date), and a line a later filing repeats exactly is counted once.

StockFit cleans the rows before serving them, so the two vendors differ where its rules bite:

- only officers' and directors' transactions are served (a 10% holder who is neither is left out);
- a trade several co-owners report appears once, under the most senior filer;
- a line priced at $100,000 or more per share is dropped as a data-entry error;
- a Form 5 line is dropped when a Form 4 reports the same trade.

## 13D/13G stakes

Each company's Schedule 13D and 13G filings (`/api/ownership/beneficial-owners/history`) land in
`beneficial_owners_history/<cik>/`, re-pulled on a new 13D or 13G. A row's `reportDate` is a UTC
time of varying precision, sometimes hours after the SEC accepted the filing, so it is never used:
{py:class}`~quantlab.dataset.stockfit.beneficial_owners.StockFitBeneficialOwnersDataset` looks
each filing up by accession number in the company's filing list and places it by its acceptance
time. A filing the list does not hold yet is left out until the list is refreshed. On its
available day only:

| Variable | Meaning |
|---|---|
| `filings_13d`, `filings_13g` | Filings of the day, amendments included, one per accession number |
| `activist` | A filing of the day is flagged activist |
| `percent_of_class`, `shares_owned` | The largest stake a filing of the day reports; NaN without a filing, 0 when a holder reports it sold out |

StockFit serves the structured filings the SEC has required since December 2024, so the panel
starts in 2025.

## Executive pay and governance

The pay-versus-performance table of a proxy statement (`/api/executives/compensation`), its
company-selected measure (`/api/executives/performance-measures`) and the governance flags
(`/api/executives/governance`) land in `executives_*/<cik>/`, re-pulled on a new DEF 14A. None
carries a filing date: {py:class}`~quantlab.dataset.stockfit.executives.StockFitExecutivesDataset`
shows fiscal year Y's values from 120 days after its end (the proxy deadline), the end taken from
the company's annual income statement for Y (a year the statements do not date is left out). The
variables are `ceo_total_comp`, `ceo_actually_paid_comp`, `avg_officer_total_comp`,
`avg_officer_actually_paid_comp`, `company_tsr`, `peer_group_tsr`, `company_selected_measure` (its
name and unit stay raw, so compare it only within a company) and four flags as 1.0/0.0
(`insider_trading_policy`, `mnpi_considered_in_award_timing`, `award_timing_predetermined`,
`mnpi_disclosure_timed_for_comp`). The flags have no fiscal year of their own: a pull of them
belongs to the newest fiscal year of the compensation as of that pull, so they gain a history only
from the first pull on. Values are as served today, restated figures included.

```python
from quantlab.dataset.config import StockFitExecutivesConfig
from quantlab.dataset.stockfit.executives import StockFitExecutivesDataset

config = StockFitExecutivesConfig(
    zarr_file_path="/data/market/stockfit/stockfit_executives_1d/stockfit_executives_1d.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
    start_date="2022-01-01",
)
StockFitExecutivesDataset(config).update()
panel = StockFitExecutivesDataset(config).panel("2025-01-02", "2025-12-31")
panel["ceo_actually_paid_comp"].sel(symbol=1652044)  # Alphabet's CIK

## Snapshot panels

StockFit keeps no history of some of what it serves: the company lists give each company's status
and SFIC sector and industry today, the earnings calendar what is expected next, the IPO calendar
where each deal stands. The refresh pulls these on every run and the raw tier keeps every pull that
changed something, so these panels accumulate from the first refresh on. All are on the CIK axis
and NYSE's trading days, in {py:mod}`quantlab.dataset.stockfit.snapshots`.

- **Available date**: each snapshot is placed at the time it was pulled, by the rule every StockFit
  panel uses: pulled before a session's close (16:00 New York, 13:00 on an early close), it is
  available that trading day, otherwise on the next one. A cell shows the newest snapshot available
  that day, so a value shows only from the first snapshot that carried it and a change from the day
  of the snapshot that carried the change.
- **No backfill**: a panel starts on the day of the first refresh. The first snapshot of the company
  lists gives each company's *current* status and industry from that day on, not their history.
- A company a market-wide snapshot stops listing (an earnings date that passed, an offering older
  than a year) shows nothing from that snapshot on.

| Panel | Source endpoint | Variables |
|---|---|---|
| {py:class}`~quantlab.dataset.stockfit.snapshots.StockFitListingDataset` | `/api/symbols` (operating, delisted), `/api/company/delisted` | `operating` (1, or 0 once delisted), `delisting_day`, `sector`, `industry` (SFIC, coded; `names` maps codes to names) |
| {py:class}`~quantlab.dataset.stockfit.snapshots.StockFitEarningsConsensusDataset` | `/api/earnings/upcoming` (next 28 days) | `earnings_date`, `filing_date`, `eps_estimate`, `analyst_count`, `confirmed` |
| {py:class}`~quantlab.dataset.stockfit.snapshots.StockFitIpoCalendarDataset` | `/api/company/ipo-calendar` (each stage) | `stage` (1 filed, 2 upcoming, 3 priced, 4 withdrawn), `price_range_low`, `price_range_high`, `price`, `shares_offered`, `dollar_value`, `filed_date`, `expected_price_date`, `priced_date`, `withdrawn_date` |
| {py:class}`~quantlab.dataset.stockfit.snapshots.StockFitOfferingsDataset` | `/api/filings/recent-offerings` (last 365 days) | `filed_date`, `priced`, `shares_offered`, `price_range_low`, `price_range_high`, `proposed_max_offering` |
| {py:class}`~quantlab.dataset.stockfit.snapshots.StockFitFilingCalendarDataset` | `/api/filings/calendar` (per company) | `expected_10k`, `expected_10q`, `expected_def14a`, and `confidence_<form>` (1 low, 2 medium, 3 high) |
| {py:class}`~quantlab.dataset.stockfit.snapshots.StockFitNewsCountsDataset` | `/api/lookup/news` (per company, latest 50) | `news_count` |

- **SFIC codes**: a sector or industry is coded by the CRC-32 of its name, so a code never depends on
  the day the store was built.
- **Earnings consensus**: `confirmed` is 1 when StockFit gives an expected report time, which it does
  once exchange data confirms the date, and 0 while the date is its prediction from the company's
  filing cadence (no estimate then). StockFit keeps no consensus history, so an earnings surprise can
  only be computed from the estimates captured before each report.
- **Ticker-keyed rows**: the earnings and IPO calendars name a company by ticker. A ticker is resolved
  to the one company of the company lists that may have held it on the day of the pull, as for
  Congress trades. An IPO deal names its company only by the ticker it intends to list under, so most
  deals show on their company only once it is listed; unresolved rows stay in the raw tier only.
- **Per-company snapshots**: the filing calendar and the news are one request per operating company.
  They are pulled once with the backfill and then re-pulled once a day from what the day's requests
  leave, oldest first, so on a busy day some companies are a day or more old.
- **News**: a headline counts once, on the trading day it became known: the later of its publication
  time and the pull that first carried it, so the first pull's headlines all land on the first
  pull's day. `news_count` is 0 on a day without a headline and NaN before the company's first pull.
  More than 50 headlines between two pulls lose the oldest, so a count is a lower bound. Titles,
  summaries and links stay in the raw tier only.

```python
from quantlab.dataset.config import StockFitEarningsConsensusConfig
from quantlab.dataset.stockfit.snapshots import StockFitEarningsConsensusDataset

config = StockFitEarningsConsensusConfig(
    zarr_file_path="/data/market/stockfit/stockfit_earnings_consensus_1d/stockfit_earnings_consensus_1d.zarr",
    raw_data_dir_path="/data/downloads/stockfit",
)
StockFitEarningsConsensusDataset(config).update()
panel = StockFitEarningsConsensusDataset(config).panel("2026-11-16", "2026-11-20")
```
