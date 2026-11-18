# StockFit fixtures

These are real StockFit API responses (`https://api.stockfit.io/v1`), recorded once on 2026-11-15 with a Professional-tier key. Tests replay them, so CI needs no StockFit key.

## Format

- Each `<name>.json` file holds the **response body** of one GET request, re-serialized with `indent=1`. No request or response headers are kept: no `Authorization` header, token or key appears in any file.
- `index.json` maps each fixture name to the request that produced it (`path`, query `params`, HTTP `status`). Where a long array was cut down, its entry has a `trimmed` note saying what was kept. Every other file is the full response to the query listed.
- File names are `<endpoint family>_<entity>[_<variant>]`.

## Entities

| Entity | CIK | Why it is here |
|---|---|---|
| Plug Power (PLUG) | 1093691 | Restatement chain: FY2018 was restated by the 2021 10-K and touched by the 2022 10-K/A |
| Alphabet (GOOGL/GOOG) | 1652044 | Multi-class issuer: one CIK, three share classes, several CUSIPs/FIGIs |
| TSMC (TSM) | 1046179 | ADR of a foreign filer: 20-F statements converted from TWD |
| Twitter (TWTR) | 1418091 | Delisted company |
| Google Inc. | 1288776 | Delisted predecessor (company details) |
| Realty Income (O) | 726728 | REIT properties footnote |
| Boeing (BA) | 12927 | Retirement-plan and supplier-finance footnotes (Alphabet has none) |
| Coca-Cola (KO) | 21344 | Declared-dividend lookup |
| NVIDIA (NVDA) | 1045810 | Congress trades by company |
| Vanguard 500 Index Fund (VOO) | 36405, series 2839 | Fund series (N-PORT, flows, fees, N-CSR performance, daily holdings) |
| Berkshire Hathaway | 1067983 | 13F manager; its 2023 confidential Chubb stake came in a 13F-HR/A |

## Files by endpoint family

- **Reference**: `lookup_symbol_*`, `lookup_cik_*`, `lookup_cusip_*`, `lookup_figi_*`, `symbols_operating_page1`, `symbols_delisted_page1`, `company_delisted_page1`, `company_details_*`, and the dictionaries `symbols_sectors`, `symbols_industries`, `symbols_exchanges`.
- **Statements**: `income_statement_*`, `balance_sheet_*`, `cash_flow_statement_*` for PLUG (annual 10 periods, quarter 12, TTM), TWTR (quarter) and TSM (annual), plus `income_statement_googl_quarter_source_details` (`includeSourceDetails=true`).
- **As-reported**: `as_reported_plug_annual` (trimmed to FY2025 and FY2018), `as_reported_plug_quarter` (trimmed to 2 periods).
- **Footnotes**: one `footnote_<table>_<entity>` per footnote endpoint (17, annual, `limit=2`), plus `footnote_buybacks_googl_quarter`. `footnote_reit_properties_o` keeps 3 properties per period.
- **Derived**: `growth_googl_quarter`, `key_metrics_googl_quarter`, `eps_history_googl_quarter`, `dividend_history_googl_quarter`, `scores_googl`, `trends_googl`.
- **Filings**: `filings_googl_page1`, `filings_googl_8k_earnings`, `filings_latest_page1`, `filings_latest_10q`, `filings_by_accession_googl_13g`, `filings_berkshire_13f` (trimmed to the 2023-Q3/Q4 originals and amendments), `filings_vanguard_index_nport` (trimmed), `filings_item_list_googl_10q`.
- **Forward snapshots**: `earnings_date_googl`, `earnings_calendar_googl_aapl`, `earnings_upcoming_page1`, `earnings_upcoming_dividends_page1`, `earnings_dividend_date_ko`, `company_ipo_calendar_upcoming`, `filings_calendar_googl`, `filings_recent_offerings`, `filings_offering_wela`, `filings_offering_googl` (empty).
- **Corporate events**: `stock_splits_googl`, `recent_stock_splits_page1`, `upcoming_stock_splits_page1`.
- **Insiders**: `insider_transactions_googl_page1`.
- **Executives**: `executives_officers_googl`, `executives_compensation_googl`, `executives_governance_googl`, `executives_performance_measures_googl`.
- **13F**: `institutional_holders_googl_class_a_page1` (by CUSIP), `institutional_holders_history_googl_class_a` (an issuer-level aggregate whatever the selector), `institutional_holders_cb_2023q3` (Berkshire's row comes from a 13F-HR/A filed 2024-05-15; trimmed), `ownership_portfolio_berkshire_page1`.
- **13D/13G**: `beneficial_owners_googl`, `beneficial_owners_history_plug`.
- **Share classes**: `share_classes_googl` (latest period only).
- **Congress**: `congress_{house,senate}_trades_nvda`, `congress_{house,senate}_trades_recent`.
- **News**: `news_googl`.
- **Funds**: `fund_profile_voo`, `fund_flows_voo`, `fund_fees_voo`, `fund_performance_voo`, `fund_holdings_voo_page1`, `fund_holdings_daily_voo_page1`.

## How they were recorded

A short script sent each GET with the key from the `STOCKFIT_API_KEY` environment variable as a bearer token, paced under the per-minute limit, and saved only the parsed JSON body. Recording these fixtures and checking the vendor's documented behaviour used 149 requests. Before the files were copied, they were scanned for the key value and for `Bearer`, `Authorization` and `apiKey`, and nothing was found. Values are as StockFit served them on the recording date. Forward-looking responses (earnings dates, upcoming splits and dividends, IPO calendar, news) describe that day.
