# US-equity data sources for point-in-time fundamentals, Barra-style factors and live trading

Researched 2026-10-04. The question is which source can feed (a) Barra-style risk and style factors (size, beta, momentum, residual volatility, non-linear size, book-to-price, liquidity, earnings yield including analyst-forecast EP, growth, leverage, dividend yield, industry) and (b) strictly point-in-time (PIT) fundamentals, **for a strategy that will trade live**. That last requirement drives everything. The source must give a PIT history long enough to backtest, it must update daily or shortly after filings, and the history and the live feed must use the same field definitions and ids, so that the factor traded is the factor backtested. Its licence must also allow an individual to trade their own account.

Every external fact cites a primary source by tag (table below): vendor docs, schema pages, terms, pricing pages, or a live request made on 2026-10-04 from the macOS dev host (tagged `live`). Repo facts cite `path:line`. "Unverified" means only a secondary page exists or the primary page could not be read. Unverified claims are kept out of the recommendation logic. Requirement ids R1-R11 refer to the checklist in `free-data-sources.md` §1 (L35-45).

Already covered elsewhere and not repeated here: Massive/Polygon is **not PIT** (`massive.md` §2). Free price sources (yfinance, Tiingo free, Alpaca, Alpha Vantage, Stooq) are covered in `free-data-sources.md` §2. Which Compustat items and signals are worth building is covered in `fundamentals-for-return-model.md`.

## Sources

| Tag | Source |
|---|---|
| `sec_api` | `sec.gov/search-filings/edgar-application-programming-interfaces` |
| `sec_fsds` / `sec_fsds_doc` | `sec.gov/data-research/sec-markets-data/financial-statement-data-sets`; `sec.gov/files/financial-statement-data-sets.pdf` |
| `sec_xbrl_rule` | SEC Release 33-9002, "Interactive Data to Improve Financial Reporting" (2009), `sec.gov/rules/final/2009/33-9002.pdf` |
| `sec_reuse` | `sec.gov/about/privacy-information` (copyright and reuse section) |
| `sec_access` | `sec.gov/os/accessing-edgar-data` (as cited in `free-data-sources.md`) |
| `sh_fund` / `sh_faq` / `sh_stocks` / `sh_tickers` / `sh_daily` / `sh_actions` / `sh_sp500` | `sharadar.com/docs/{fundamentals,faqs,stocks,tickers,daily,actions,sp500}` |
| `sh_sub` / `sh_terms` / `sh_llms` | `sharadar.com/subscribe`; `sharadar.com/terms` (Personal Use License Terms); `sharadar.com/llms.txt` |
| `ndl_meta` | Nasdaq Data Link table metadata, `data.nasdaq.com/api/v3/datatables/SHARADAR/{SF1,SEP,TICKERS,DAILY,...}/metadata.json` (no key needed) |
| `tiingo_fund` / `tiingo_price` | `tiingo.com/documentation/fundamentals`; `tiingo.com/about/pricing` |
| `eod_llms` / `eod_fund` | `eodhd.com/llms.txt` (says "Last verified against the live API: 2026-09-28"); `eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds` |
| `fmp_docs` / `fmp_price` | `site.financialmodelingprep.com/api-docs.md`; `site.financialmodelingprep.com/pricing.md` |
| `fh_swagger` | Finnhub OpenAPI spec, `finnhub.io/static/swagger.json` |
| `intrinio_price` | `intrinio.com/pricing` |
| `qc_ms` | QuantConnect, "US Fundamental Data by Morningstar", `quantconnect.com/docs/v2/writing-algorithms/datasets/morningstar/us-fundamental-data` |
| `simfin_price` | `simfin.com/en/prices/` |
| `norgate` | `norgatedata.com/data-content-tables.php` |
| `databento` | `databento.com/pricing` |
| `figi` / `figi_api` | `openfigi.com/about/figi`; `openfigi.com/api/documentation` |
| `wrds_tou` | WRDS Terms of Use, `wrds-www.wharton.upenn.edu/users/tou/` |
| `wrds_spg` | WRDS, "S&P Global Market Intelligence" vendor page, `wrds-www.wharton.upenn.edu/pages/about/data-vendors/sp-global-market-intelligence` |
| `lseg_co` | LSEG company data page, `lseg.com/en/data-analytics/financial-data/company-data/ibes-estimates` |
| `msci_efm` | `msci.com/data-and-analytics/factor-investing/equity-factor-models` |
| `live` | Requests made 2026-10-04: SEC `companyfacts`/`submissions`/`frames`; Sharadar `test-api-key` queries (the docs publish this key; it returns about 5 years for a few large caps); EODHD `api_token=demo` (published demo key, AAPL only); read-only `information_schema`/`max(date)` queries on the user's WRDS account |

---

## 1. What "live-capable PIT" requires

Beyond R1-R11, which were written for research on CRSP, live trading adds these requirements:

- **L1 PIT values and dates.** Each fundamental row carries the date it became public. Ideally the value is the one first reported, and later revisions arrive as new rows dated by their own filing.
- **L2 Same pipeline, history and live.** The live update is the same table with the same ids and the same transformation as the history. A backtest on vendor A plus live trading on vendor B is a different factor.
- **L3 Latency.** New filings and prices arrive by the next session. EOD prices arrive by the evening.
- **L4 Stable security id without PERMNO.** It must survive ticker changes and reuse (R5), and map to the broker's symbol.
- **L5 Licence.** An individual may run automated trading of their own account on the data, and may keep running it.
- **L6 Barra inputs.** Prices and total returns (R2/R3), shares outstanding, book equity, earnings, sales, debt, dividends, an industry classification, and, for the forecast-EP descriptor, a PIT history of consensus forward EPS.

## 2. Comparison

"AR" means first-reported values kept, dated by filing. "Restated" means values are overwritten by later filings.

| Source | PIT values | Release date field | Delisted | History | Estimates history | Industry | Update latency (live) | Licence allows own trading | Price |
|---|---|---|---|---|---|---|---|---|---|
| **SEC EDGAR** companyfacts / FSDS | **Every vintage kept** (live) | `filed` per fact; `acceptanceDateTime` per filing | yes (live: SVB, Twitter) | XBRL phased in 2009-2011 (`sec_xbrl_rule`) | none | SIC (current) | **under 1 minute** (`sec_api`) | public information (`sec_reuse`) | free |
| **Sharadar** (bundle) | **AR dims = first-reported**; MR = restated (`sh_faq`, live) | `datekey` = 10-K/10-Q filing date | yes, ~18k cos. (`sh_fund`) | 1998 fundamentals, Dec 1997 prices | none | SIC, Morningstar-style sector/industry (current); SIC changes in `actions` | daily 17:30 and 23:30 ET, lag under 1 day (`sh_fund`) | **yes, explicitly** (`sh_faq`) | $499/yr full bundle (`sh_sub`) |
| **Tiingo** fundamentals add-on | As-Reported dimension exists (`tiingo_fund`) | `date` "released to the public" | permaTicker for delisted | "over 20 years" | none | unverified | within 12-24 h of the SEC filing | "own personal use" (`tiingo_price`) | add-on, price not published |
| **EODHD** Fundamentals | one row per period. AAPL shows first-reported values (live, 1 case); policy undocumented | `filing_date` (pre-1994 rows = period end, live) | 26k US delisted (`eod_llms`) | 1985 for AAPL (live) | current trend snapshot only; per-event `epsEstimate` | GICS (current) | unverified | personal plans; commercial needs B2B (`eod_llms`) | $59.99/mo; $99.99/mo all-in-one |
| **FMP** | one row per period; restatement policy undocumented | `filingDate`, `acceptedDate` | Premium+ | 30+ yr (Premium) | current consensus only | profile (current) | "real-time" on paid plans (`fmp_price`) | "Licence - Personal" | $69/mo Premium |
| **Finnhub** | `financials-reported` is per filing (vintages) (`fh_swagger`) | `filedDate`, `acceptedDate` | unverified | unverified | current only (no as-of field) | unverified | unverified | unverified | unverified |
| **Intrinio** | "standardized & as-reported" (`intrinio_price`); PIT unverified | unverified | unverified | 2006 | unverified | unverified | "real-time updates" | Individual: no redistribution | $150/mo individual |
| **QuantConnect / Morningstar** | "As Original Reported" (`qc_ms`) | filing date, or as-of + 45 d for older | yes | 1998 | n/a | Morningstar | ~6 am daily (`qc_ms`) | inside QC only (unverified) | QC subscription |
| **Massive** | **no**, restated (`massive.md`) | latest filing's date | yes | 2009 | Benzinga add-on | SIC | — | individual plans | $29/mo add-on |
| **WRDS CRSP** (`crsp_a_stock`) | n/a (prices) | — | yes, `dlret` | 1925 | — | SIC/NAICS history | **annual vintage, ends 2025-12-31** (live) | **academic, non-commercial only** (`wrds_tou`) | institutional |
| **WRDS Compustat `fundq`** | restated (ADR 0003) | `rdq` | yes | 1971 `rdq` | — | SIC, GICS | daily on WRDS (`wrds_spg`); max `rdq` 2026-10-02 (live) | academic only | institutional |
| **Compustat Snapshot** | PIT vintages | point dates | yes | — | — | — | monthly on WRDS (`wrds_spg`) | academic only | **not entitled** (live) |
| **I/B/E/S on WRDS** | monthly `statpers` snapshots | `statpers`, `anndats` | yes | **1976-01-15** (live) | **yes, PIT** | — | latest `statpers` 2026-08-20 (live) | academic only | **entitled** (live) |
| FactSet, LSEG (Worldscope/I/B/E/S), Bloomberg, S&P Capital IQ / Xpressfeed | PIT products exist (`lseg_co`) | yes | yes | 1980s US (`lseg_co`) | yes | GICS/TRBC | daily | institutional licences | not published |
| MSCI Barra, Axioma, Northfield | ready-made exposures | — | — | — | — | — | daily (unverified) | institutional | not published (`msci_efm`) |

## 3. Per-source findings

### SEC EDGAR (free, the primary record)

- **Latency.** "The APIs are updated in real-time as filings are disseminated. The submissions API is updated with a typical processing delay of less than a second; the xbrl APIs are updated with a typical processing delay of under a minute." Bulk ZIPs "are recompiled nightly". No key is needed, and CORS is not supported (`sec_api`). The limit is 10 requests/s with a contact `User-Agent` (`sec_access`). The nightly `companyfacts.zip` was 1.41 GB on 2026-10-04 (`live`, HTTP `content-length`).
- **All vintages are kept (live).** A companyfacts fact has `end`, `val`, `accn`, `fy`, `fp`, `form`, `filed`, and `start` for duration facts. For AAPL `NetIncomeLoss`, 112 of 123 periods have more than one `filed` value, and 5 have *different* values. FY2008 net income is 4,834M in the 10-K filed 2009-10-27 and 6,119M in the 10-K/A filed 2010-01-25 (Apple's revenue-recognition restatement). `Assets` at 2008-09-27 is 39,572M as first filed on 2009-07-22 and 36,171M from 2010-01-25. For MMM, Q1 2024 `Revenues` is 8,003M in the 10-Q filed 2024-04-30 and 6,016M (after the Solventum spin-off) in the 10-K filed 2025-02-05 (`live`). So first-reported values *and* revisions, each dated by its filing, can be rebuilt.
- **Frames are not PIT.** Frames return "one fact for each reporting entity that is last filed" (`sec_api`). Use companyfacts or FSDS.
- **Intraday timestamp.** The submissions API gives `acceptanceDateTime` per accession. The AAPL 10-Q for the quarter ended 2026-06-27 was accepted 2026-07-31T10:01:02Z (`live`). Join on `accn` to place a fact after the exact acceptance time.
- **Financial Statement Data Sets** are the bulk equivalent: "as filed", from 2009-04-15, updated **quarterly**. SUB carries `filed`, `accepted` (date-time) and `prevrpt`, which "indicates that the submission information was subsequently amended" (`sec_fsds`, `sec_fsds_doc` §2-3). Being quarterly, they suit history but not live use.
- **Coverage start.** XBRL began with fiscal periods ending on or after 2009-06-15 for large accelerated filers with float above $5 billion (about 500 companies), 2010-06-15 for other large accelerated filers, and 2011-06-15 for all remaining filers including smaller reporting companies (`sec_xbrl_rule`). A broad-universe EDGAR history therefore starts around 2011. Comparative periods in the first filings reach a few years further back: AAPL's FY2007 net income appears in its 10-K filed 2009-10-27 (`live`).
- **Delisted issuers stay.** SVB Financial (last `Assets` filed 2023-02-24) and Twitter (2022-07-26) are both still served (`live`).
- **Ids.** CIK is per registrant, not per share class. `company_tickers.json` maps only current tickers (10,440 rows, `live`). EDGAR therefore needs a CIK-to-security map from another source (§4).
- **Normalisation cost (the real cost).** Companies use different tags for one concept: AAPL revenue is `Revenues` in 11 facts and `RevenueFromContractWithCustomerExcludingAssessedTax` from 2019 (`live`). Cash-flow items are year-to-date. Fiscal calendars vary. Duration and instant facts mix. Every factor needs a concept map with fallbacks, YTD differencing (the same as `fundamentals-for-return-model.md` §4 step 3) and sanity checks. FSDS's `tag`/`pre` tables help. Values are as tagged by filers: the SEC "cannot guarantee the accuracy of the data sets" (`sec_fsds_doc` §1).
- **No analyst estimates and no prices.** It must be paired with a price vendor.
- **Reuse.** "Information presented on sec.gov is considered public information and may be copied or further distributed" (`sec_reuse`).

### Sharadar (sharadar.com, also on Nasdaq Data Link)

- **PIT design.** "Point-in-time/As-Reported financials available via AR dimension observations (ARQ/ARY/ART). Most-Recent-Reported financials available via MR dimension observations (MRQ/MRY/MRT)" (`sh_fund`). As-Reported "is a point-in-time view, time-indexed to the SEC form 10 filing date, and excludes restatements. That view is not rewritten when a later filing restates a prior period" (`sh_faq`). The AR dimension "presents data for the latest reporting period at that filing date" (`sh_fund`). So AR keeps the **first-reported value only**, and a restated value never enters AR.
- **Verified live against EDGAR.** MMM ARQ for 2024-03-31 has `date` 2024-04-30 and revenue 8,003M, while MRQ has `date` = `reportperiod` = 2024-03-31 and revenue 6,016M. This matches EDGAR's first filing and later restatement exactly. AAPL ARQ `date`s match its 10-Q filing dates, e.g. 2026-07-31 (`live`). Note that MR rows are dated by **report period**, so using MR in a backtest is look-ahead.
- **Caveats in its own docs.** Values come from form 10 filings, "the information may have been separately disclosed to the market days (or on rare occassion - weeks) earlier under separate form 8" (`sh_fund`). This makes the data late rather than early, which is safe. After a filing delay, catch-up filings give only the most recent period (`sh_fund`). `lastupdated` was 2026-07-31 on every AAPL row (`live`), so it is not a vintage marker.
- **Coverage and latency.** Fundamentals: about 18,000 active and delisted companies, primary common class, NYSE/Nasdaq/NYSEMKT, from January 1998. They are delivered "Daily at 17h30 and 23h30 US Eastern (ET)" with reporting lag "< 1 day" (`sh_fund`). Nasdaq Data Link metadata shows SF1, SEP, DAILY and ACTIONS as `update_frequency: CONTINUOUS`, refreshed 2026-10-04 (`ndl_meta`). SF1's primary key is `(ticker, dimension, datekey, reportperiod)` (`ndl_meta`).
- **Prices (`stocks`/SEP).** About 21,000 tickers including delisted, from December 1997. OHLCV are split-adjusted, `closeunadj` is raw, and `closeadj` is "split, cash dividend and spinoff adjusted" (`sh_stocks`, `sh_faq`). A total-return-adjusted open is `open * closeadj / close`, which is the same construction as the CRSP conversion's scaling (`free-data-sources.md` R2). **No delisting-return field** was found. `actions` holds delisting dates and reasons plus cash/stock acquisition consideration (`sh_actions`, `sh_faq`), so an R3-style delisting return must be built from those, and bankruptcies end at the last price.
- **Ids.** `permaticker` is "Sharadar's own unchanging and unique identifier for a security. Separate share classes of the same issuer receive separate permatickers" (`sh_faq`). Tickers are rewritten on change, and a reused ticker goes to the active company while "the delisted company gets a number appended" (`sh_faq`). The `tickers` row carries `cusips`, `figi`, `siccode`, `sector`/`industry`/`famaindustry` and a `secfilings` URL that contains the **CIK** (AAPL: permaticker 199059, FIGI BBG000B9XRY4, CIK 0000320193, `live`). **Data tables are keyed by ticker**, so joins must go through `tickers`.
- **Other tables.** `daily` (market cap, EV, P/E, P/B, P/S, "Point-in-time/As-Reported basis", `sh_daily`). `sp500` (current members, changes and quarterly snapshots from 1998, `sh_sp500`). `actions` (splits, dividends, spin-offs, delistings, ticker changes, and SIC changes as `sicchangefrom`/`sicchangeto`, `sh_faq`). Also insiders (SF2) and 13F holdings (SF3). **No analyst estimates.**
- **Industry is current, not PIT.** The `tickers` docs do not say the sector fields are historical (`sh_tickers`). SIC history can be rebuilt from the `actions` SIC changes.
- **Price and licence.** Personal Use plans: Fundamentals $399/yr, Prices $299/yr, Bundle $499/yr (full history, annual billing). Monthly: $39 / $39 / $69 (`sh_sub`). "Personal Use covers individuals using the data for their own purposes: research, backtesting, and automated trading of their own account with no external clients or money managed for others" (`sh_faq`). Restrictions (`sh_terms`): natural persons only, with no professional or entity use. No redistribution. On termination, data must be deleted within 30 days, but "research outputs, backtest results, models ... trade logs" may be kept. And a clause that matters for this **public** repo (R11): conclusions from "testing or evaluation of the Services or the Services Data ... shall not be published ... without prior written approval of Sharadar". This document is based only on public pages and the published test key. After subscribing, keep data-quality findings about Sharadar out of the public repo.

### Tiingo fundamentals (same vendor as the repo's `TiingoAcquisition`)

The Statements endpoint has a Most-Recent (`asReported=false`) and an As-Reported (`asReported=true`) dimension. "Prior period data is pulled from the latest report for the Most-Recent ... dimension, current period is taken only for As-Reported." `date` is "the date the statement data was released to the public". Data is updated "within 12-24 hours of being made available by the SEC", goes back "over 20 years", and `permaTicker` is a "Permanent Tiingo Ticker ... Can be used as a primary key" (`tiingo_fund`). It is "an add-on subscription" with the Dow 30 free for evaluation, and **its price is not published** (`tiingo_fund`, `tiingo_price`). Power (prices) costs $30/month for individuals, and internal use means "only ... your own personal use" (`tiingo_price`). The design matches Sharadar's AR, and Tiingo prices are already integrated. Whether it keeps revisions, its delisted coverage and its price need a sales enquiry.

### EODHD, FMP, Finnhub, Intrinio, SimFin, QuantConnect

- **EODHD.** One record per fiscal period with `date` and `filing_date` (`eod_fund`). For AAPL, the 2008-09-30 quarter shows `totalAssets` 39,572M with `filing_date` 2008-11-05, the original 10-K date and the pre-restatement value (`live`). 34 of 164 quarterly rows (all before 1994) have `filing_date` equal to the period end, i.e. a placeholder (`live`). One example does not establish a policy, and the docs state none, so it is **not PIT by documentation**. Estimates: `Earnings.Trend` is a current snapshot with 7/30/60/90-day-ago EPS, and `Earnings.History` keeps one `epsEstimate` per reported quarter with `beforeAfterMarket` (`live`). GICS sector, CIK and OpenFIGI are in `General` (`live`). Pricing: Fundamentals $59.99/month, All-In-One $99.99/month. "Commercial/display use requires B2B pricing", and there are 26,000+ delisted US tickers (`eod_llms`).
- **FMP.** Statements have one record per period with `filingDate` and `acceptedDate`. The as-reported endpoints return raw XBRL tags per fiscal year with no filing date in the example (`fmp_docs`). Analyst estimates are per future period with no as-of date, i.e. current only (`fmp_docs`). Personal plans: Starter $29, Premium $69 (30+ years), Ultimate $139 per month. Delisted companies need Premium, and bulk statements need Ultimate (`fmp_price`). **Not PIT.**
- **Finnhub.** `/stock/financials-reported` returns one `Report` per filing with `accessNumber`, `form`, `filedDate` and `acceptedDate`, i.e. an EDGAR mirror with vintages. `/stock/eps-estimate` has no as-of field (`fh_swagger`). Price, history and delisted coverage are unverified. It adds little over EDGAR directly.
- **Intrinio.** "15+ years of standardized & as-reported financial statement data", back to 2006. Individual plan $150/month with "No redistribution or display" (`intrinio_price`). PIT behaviour unverified.
- **SimFin.** Plans $15-$71/month with 5-20+ years of history, personal licence (`simfin_price`). PIT behaviour unverified.
- **QuantConnect / Morningstar.** "All the data is loaded using 'As Original Reported' figures". The file date is the filing date when present, else "approximated 45 days after the as of date". The data is delivered live at about 6 am, from 1998 (`qc_ms`). But Morningstar's feed was replaced: "From September 23, 2026, US fundamentals come from the new feeds for the whole history", and "about one third differ ... backtests run on it will not reproduce" (`qc_ms`). It is usable only inside QuantConnect (unverified), and the 2026 rebase shows the vendor-revision risk of a black-box PIT history.

### Prices and reference only

- **Norgate.** Delisted US stocks from 1950 and historical constituents for the S&P 500, Russell and Nasdaq-100 on Platinum/Diamond. Fundamentals are "current" only (`norgate`). Price and licence unverified.
- **Databento.** Standard $199/month includes live data and allows "Personal use" and "Commercial use", with 1-day OHLCV, "16+ years" of history, and separate corporate-actions and security-master products (`databento`). It suits live intraday prices, not fundamentals.

### WRDS (research and validation only)

- **Licence.** "The WRDS services are for academic and non-commercial research purposes only. Users may not use data downloaded from the WRDS database for any non-academic or commercial endeavor". Access ends when the user leaves the subscribing institution (`wrds_tou`). Trading one's own money is not academic research. On this reading WRDS data must not drive live trading, and an account that ends with the affiliation cannot be a durable live feed. (Whether personal trading counts as "commercial" is the user's legal call. This document does not rely on a "yes".)
- **Latency.** The account reads `crsp_a_stock`, the annual product (`quantlab/acquisition/wrds/crsp.py:26-31`). Its last day is **2025-12-31**, more than nine months stale on 2026-10-04 (`live`). `crsp_q_stock`/`crsp_m_stock` are not entitled (`live`). Compustat on WRDS is updated daily (`wrds_spg`): `comp.fundq` has `rdq` up to 2026-10-02 (`live`), but its values are restated (ADR 0003).
- **Entitlements found (`live`).** `ibes.statsum_epsus` is **readable**, with `statpers` from 1976-01-15 to 2026-08-20 and 35,205 rows in the latest month. `ibes.det_epsus` runs to 2026-08-20. This settles the open question in `fundamentals-for-return-model.md` (I/B/E/S entitlement). `comp_snapshot` (behind `compsnap.wrds_csq_pit`) and `zacks_all` (behind the `zacks.*` views) are **permission denied**, which confirms ADR 0003 for Snapshot. WRDS lists "Compustat Snapshot" as monthly and "Compustat Point in Time (Charter Oaks)" as quarterly (`wrds_spg`).
- **Role.** CRSP + Compustat + I/B/E/S remain the best *research* reference. Use them to validate a live-vendor panel on the overlapping years (prices, shares and returns vs CRSP; AR fundamentals vs `fundq` by `rdq`), and to test whether analyst-forecast EP adds anything before paying for a live estimates feed.

### Institutional vendors and ready-made risk models

LSEG offers "as-reported financials ... point-in-time" with history from "the early 1980s for US companies" through Workspace, Datastream, DataScope and APIs, with pricing on request (`lseg_co`). MSCI lists over 70 equity factor models, delivered through "Barra platforms, flat files, third-party platforms or Snowflake", pricing on request (`msci_efm`). FactSet, Bloomberg, S&P Capital IQ/Xpressfeed, Axioma and Northfield publish no prices, and their PIT products were not read. All are institutional sales, out of scope for an individual. Zacks fundamentals and estimates on Nasdaq Data Link could not be inspected: the metadata endpoint hit the anonymous limit of 50 calls per day (`live`), and the WRDS `zacks` views are permission denied (`live`). JKP and Chen-Zimmermann factor data (`fundamentals-for-return-model.md` [jkpdoc], [cz]) are monthly research files built from CRSP/Compustat, updated a few times a year (JKP "through December 2025"), so they cannot feed daily live signals. Use them as a benchmark for one's own factor definitions.

## 4. Pairings for live trading

### A. Sharadar bundle alone (history and live from one table set)

- **What it covers.** L1: AR dimensions. L2: the same tables update daily and history and live share keys. L3: evening updates. L4: `permaticker` plus FIGI/CUSIP/CIK. L5: explicit. L6: prices, shares (via `daily`/SF1), book, earnings, sales, debt, dividends (`actions`) and SIC/sector. Beta, momentum, residual volatility and liquidity come from SEP.
- **Gaps.** No consensus estimates (forecast EP), no delisting return field, industry not PIT except SIC via `actions`, and no revisions in AR (first-reported only). Full history from 1998, so 2010-2024 backtests are covered.
- **Cost.** $499/yr.

### B. SEC EDGAR fundamentals plus a price vendor

- **What it covers.** L1: the strongest PIT (every vintage, acceptance timestamps). L3: under a minute. L5: public. One code path builds history from `companyfacts.zip` and stays live from the APIs, so L2 holds *within your own pipeline*.
- **Costs.** The normalisation work in §3. A broad universe only from about 2011 (`sec_xbrl_rule`), which is shorter than R1's 2010 warm-up for small caps. A security map is still needed: CIK is issuer-level, so the price vendor's security id must map to CIK. Sharadar `tickers` (permaticker to CIK/FIGI/CUSIP) or EODHD `General` (CIK, OpenFIGI) provide it. FIGI "never changes ... retired and never reused" and is open data (`figi`), and OpenFIGI's API is free and rate-limited (`figi_api`).
- **Price-vendor choices.** Sharadar Prices $299/yr (delisted, permaticker, CIK in `tickers`). Tiingo Power $30/month (CRSP-method adjustment, permaTicker; the R5 ticker-reuse issue noted in `free-data-sources.md`). EODHD EOD $19.99/month (26k delisted, CIK/FIGI in fundamentals). Databento for live intraday.

### C. Recommended: A plus EDGAR as audit and revision layer (WRDS for validation)

Use Sharadar as the production source, and EDGAR to (1) audit AR values and dates on a sample and (2) optionally add revision rows Sharadar's AR omits. Keep WRDS for research-only checks. The id spine is `permaticker`, with CIK and FIGI stored as attributes for EDGAR and broker mapping.

## 5. Recommendation

Ranked for an individual who will trade live, and who holds WRDS CRSP + Compustat + I/B/E/S for research only:

1. **Buy the Sharadar Bundle, full history, $499/yr.** It is the only source found that documents first-reported PIT fundamentals dated by filing, daily updates of the same tables, delisted coverage, a stable share-class id with CIK/FIGI crosswalk, and a licence that explicitly allows automated trading of one's own account (`sh_fund`, `sh_faq`, `sh_sub`). The live MMM/AAPL checks agree with EDGAR.
2. **Add SEC EDGAR (free) as the second source.** First as an audit of Sharadar AR (values and filing dates). Later, if needed, as the source of revision-aware vintages and acceptance-time placement. EDGAR alone (option B) is viable for a lowest-cost build, but the normalisation work and the 2009-2011 phase-in make it a project in itself.
3. **Keep WRDS for validation only.** Compare the Sharadar-based panel with CRSP/Compustat over 2010-2024 before trading. Use I/B/E/S (entitled) to measure whether forecast EP adds anything beyond trailing EP in this universe.
4. **Analyst-forecast EP: defer.** No vendor checked offers an individual a licensable PIT consensus history *and* a live feed: FMP, Finnhub and EODHD expose only current consensus, and I/B/E/S on WRDS is academic-only with monthly snapshots. Options: (a) build the earnings-yield factor on trailing EP from Sharadar ART first; (b) if the I/B/E/S research shows value, price a live estimates feed (Zacks via Nasdaq Data Link, FactSet or LSEG, all by enquiry); (c) start snapshotting a cheap vendor's current consensus daily now. A self-collected history is PIT by construction but only begins today.
5. **Ask Tiingo for the fundamentals add-on price.** If it is close to Sharadar's and keeps an As-Reported dimension with delisted coverage, it would reuse the repo's existing Tiingo integration (`quantlab/acquisition/tiingo.py`). Unverified until priced.

**What to do first.**

1. Use the published Sharadar test key, or a one-month $69 bundle, to pull `tickers`, `actions`, SF1 ARQ/ART and SEP for the current S&P 500. Map permaticker to PERMNO through CUSIP and CIK (CCM/`stocknames` on WRDS) for the overlap.
2. Compare research panels side by side (shares, market cap, total return, book equity, earnings and their available dates) against CRSP/`fundq`, on WRDS, without publishing Sharadar-specific findings (`sh_terms`).
3. Define the delisting-return rule from `actions` (R3).
4. Then subscribe annually and write the acquisition as a library vendor next to Tiingo, keyed by permaticker.

## Unverified claims

- Tiingo fundamentals add-on price, whether its As-Reported dimension keeps revisions, its delisted depth and its industry fields.
- EODHD's restatement policy (only one AAPL case observed) and its update latency.
- Finnhub's prices, history depth, delisted coverage and licence.
- Intrinio's and SimFin's PIT behaviour, release-date fields and delisted coverage.
- That QuantConnect's Morningstar data cannot be exported for use outside QuantConnect.
- Norgate's prices and licence terms.
- Whether Sharadar sector/industry fields are ever historical (the docs are silent), and whether a CRSP-style delisting return can be rebuilt fully from Sharadar `actions`.
- Pricing and PIT details of FactSet, Bloomberg, S&P Capital IQ/Xpressfeed, Zacks, Axioma, Northfield and MSCI Barra (none published).
- Whether trading one's own account counts as "commercial" under the WRDS terms (the document does not rely on it either way).
- JKP and Chen-Zimmermann data licences.

[sec_api]: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
[sec_fsds]: https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets
[sec_fsds_doc]: https://www.sec.gov/files/financial-statement-data-sets.pdf
[sec_xbrl_rule]: https://www.sec.gov/rules/final/2009/33-9002.pdf
[sec_reuse]: https://www.sec.gov/about/privacy-information
[sec_access]: https://www.sec.gov/os/accessing-edgar-data
[sh_fund]: https://sharadar.com/docs/fundamentals
[sh_faq]: https://sharadar.com/docs/faqs
[sh_stocks]: https://sharadar.com/docs/stocks
[sh_tickers]: https://sharadar.com/docs/tickers
[sh_daily]: https://sharadar.com/docs/daily
[sh_actions]: https://sharadar.com/docs/actions
[sh_sp500]: https://sharadar.com/docs/sp500
[sh_sub]: https://sharadar.com/subscribe
[sh_terms]: https://sharadar.com/terms
[sh_llms]: https://sharadar.com/llms.txt
[ndl_meta]: https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1/metadata.json
[tiingo_fund]: https://www.tiingo.com/documentation/fundamentals
[tiingo_price]: https://www.tiingo.com/about/pricing
[eod_llms]: https://eodhd.com/llms.txt
[eod_fund]: https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds
[fmp_docs]: https://site.financialmodelingprep.com/api-docs.md
[fmp_price]: https://site.financialmodelingprep.com/pricing.md
[fh_swagger]: https://finnhub.io/static/swagger.json
[intrinio_price]: https://intrinio.com/pricing
[qc_ms]: https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/morningstar/us-fundamental-data
[simfin_price]: https://www.simfin.com/en/prices/
[norgate]: https://norgatedata.com/data-content-tables.php
[databento]: https://databento.com/pricing
[figi]: https://www.openfigi.com/about/figi
[figi_api]: https://www.openfigi.com/api/documentation
[wrds_tou]: https://wrds-www.wharton.upenn.edu/users/tou/
[wrds_spg]: https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/sp-global-market-intelligence
[lseg_co]: https://www.lseg.com/en/data-analytics/financial-data/company-data/ibes-estimates
[msci_efm]: https://www.msci.com/data-and-analytics/factor-investing/equity-factor-models
