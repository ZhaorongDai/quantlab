# Massive (formerly Polygon.io) for Barra-style and point-in-time fundamental factors

Researched 2026-10-04. The question is whether Massive can supply the inputs for Barra-style risk factors and point-in-time (PIT) fundamental factors on US equities, and which plan to buy if so. Every external fact cites Massive's own documentation or pricing page by tag (table below), read on 2026-10-04. Repo facts cite paths. "Unverified" marks a claim no primary source was read for.

## Sources

| Tag | Source |
|---|---|
| [pricing] | `massive.com/pricing` |
| [business] | `massive.com/business` |
| [is] | `massive.com/docs/rest/stocks/fundamentals/income-statements` |
| [bs] | `massive.com/docs/rest/stocks/fundamentals/balance-sheets` |
| [ratios] | `massive.com/docs/rest/stocks/fundamentals/ratios` |
| [float] | `massive.com/docs/rest/stocks/fundamentals/float` |
| [si] | `massive.com/docs/rest/stocks/fundamentals/short-interest` |
| [overview] | `massive.com/docs/rest/stocks/tickers/ticker-overview` |
| [tickers] | `massive.com/docs/rest/stocks/tickers/all-tickers` |
| [fidx] | `massive.com/docs/rest/stocks/filings/index` |
| [bz] | `massive.com/docs/rest/partners/benzinga/earnings` |

---

## 1. Verdict

**Massive does not meet the PIT requirement.** Its statements keep one, latest-restated record per fiscal period, and its historical share counts are keyed to the period end rather than to the filing date. For research, CRSP plus Compustat `fundq` placed by `rdq` (`docs/adr/0003-point-in-time-fundamentals-by-release-date.md`) is better on every input in the table in section 3. The project now targets live trading, however, and WRDS may not be usable there (update lag, licence). The source that serves both backtest and live is chosen in `us-equity-data-sources.md`. Massive's statement history cannot be that source, because a backtest on restated values is not the factor that would be traded.

## 2. Fundamentals

- **Endpoints and coverage.** `/stocks/financials/v1/income-statements`, `balance-sheets` and `cash-flow-statements`, sourced from SEC 10-K and 10-Q filings, with `timeframe` `quarterly`, `annual` or `trailing_twelve_months`. Records date back to 2009-03-29 ([is], [bs]).
- **Not point-in-time.** `filing_date` is "the date of the most recent SEC filing that included this period's data. This is not necessarily the date this period was originally filed. Because SEC filings restate comparative data for prior periods, multiple records can share the same filing_date" ([is], [bs]). So one record per period survives, carrying the latest restated values and the latest filing's date. Neither the first-reported values nor the first-release date is kept.
- **The documented workaround recovers dates, not values.** The docs point to the EDGAR filings index (`/stocks/filings/vX/index`) for "the original filing date" ([is]). That index returns accession number, CIK, form type, filing date and URL per filing ([fidx]). It gives the first release date, but the record's values are still the restated ones.
- **Compared with Compustat `fundq`.** `fundq` values are restated too, but it keeps `rdq`, the original report date that ADR 0003 places rows by. Massive's `filing_date` moves forward on every restatement, so it is strictly worse for PIT placement.
- **Ratios have no history.** The ratios endpoint gives TTM ratios "for the most recent trading day", and its plan history is "Not applicable to this endpoint" ([ratios]). It cannot be backtested.

## 3. Inputs for a Barra-style model

| Input | Massive | WRDS (already available) |
|---|---|---|
| Daily price/volume (beta, momentum, volatility, liquidity) | Daily aggregates. Delisted tickers are listed via `active=false` with a `delisted_utc` ([tickers], [overview]). History: 5 years on Starter, 10 on Developer, 20+ on Advanced ([pricing]). Keyed by ticker. No delisting return was found in the docs | CRSP: PERMNO-keyed, `dlret` |
| PIT shares outstanding (size) | Ticker Overview takes a `date`, but "we compare this date with the period of report date on the SEC filing": a filing submitted 2019-07-31 for the period ending 2019-06-29 is returned for a query dated 2019-06-29 ([overview]). That is about one month of look-ahead. It also returns one ticker per call | CRSP `shrout`, daily |
| Free float | `effective_date` per measurement; plan history "Not applicable" ([float]) | CRSP shares; no free float |
| Industry | SIC code and description only ([overview]); no GICS | CRSP SIC/NAICS history |
| Value, growth, leverage, profitability | Statements above: not PIT, from 2009 | Compustat `fundq` + `rdq` |
| Forecast earnings (Barra earnings yield) | Benzinga Earnings add-on: `estimated_eps` and `estimated_revenue` consensus per reported period, from 2010-04-30, $99/month individual ([bz]) | I/B/E/S, if the account is entitled (unverified) |
| Short interest | Settlement-date records from 2017-12-29, in every Stocks plan ([si]) | Compustat short interest (see `fundamentals-for-return-model.md`) |

## 4. Plans

Prices from [pricing] and [business]; annual billing is 20% off ([pricing]).

| Plan | Price | Relevant content |
|---|---|---|
| Financials & Ratios add-on, individual | $29/month | All statement history; usable without a Stocks plan |
| Stocks Advanced, individual ("non-pros only") | $199/month | 20+ years of prices, plus financials & ratios |
| Stocks Developer / Starter, individual | $79 / $29 per month | 10 / 5 years of prices; no financials |
| Benzinga Earnings, individual | $99/month | Consensus EPS/revenue events from 2010 |
| Stocks Business | $2,499/month | Business licence |
| Financials & Ratios, business | $699/month | Business licence |

If Massive is ever bought, the $29 add-on is the only relevant plan. It suits fetching the latest statements for live trading, not a PIT backtest.

## 5. Recommendation

1. Do not use Massive statements as the factor history. For research, the ADR 0003 path (CRSP prices and shares, Compustat `fundq` placed by `rdq`) remains the benchmark.
2. For live trading and for values that are immune to restatement, rebuild vintages from SEC EDGAR XBRL data, which Massive's statements are themselves derived from. See `us-equity-data-sources.md` for the sourced assessment.

## Unverified claims

- Whether Massive's daily aggregates carry delisting returns. None was found in the pages read.
- Whether the individual licence allows the repo's use. The pricing page labels individual plans "Individual only" and Advanced "Non-pros only", but the terms of service were not read.
- Whether the I/B/E/S entitlement exists on this WRDS account.

[pricing]: https://massive.com/pricing
[business]: https://massive.com/business
[is]: https://massive.com/docs/rest/stocks/fundamentals/income-statements
[bs]: https://massive.com/docs/rest/stocks/fundamentals/balance-sheets
[ratios]: https://massive.com/docs/rest/stocks/fundamentals/ratios
[float]: https://massive.com/docs/rest/stocks/fundamentals/float
[si]: https://massive.com/docs/rest/stocks/fundamentals/short-interest
[overview]: https://massive.com/docs/rest/stocks/tickers/ticker-overview
[tickers]: https://massive.com/docs/rest/stocks/tickers/all-tickers
[fidx]: https://massive.com/docs/rest/stocks/filings/index
[bz]: https://massive.com/docs/rest/partners/benzinga/earnings
