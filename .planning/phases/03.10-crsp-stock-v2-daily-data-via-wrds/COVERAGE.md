# API Coverage — WRDS CRSP Stock v2 (annual update), Compustat index constituents, CRSP/Compustat Merged links (WRDS PostgreSQL)

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
> Scope: the WRDS PostgreSQL surface this phase's providers (`quantlab/acquisition/wrds_crsp.py`,
> `quantlab/acquisition/wrds_crsp_reference.py`) could use, as seen in `03.10-LIVE-CHECK*.json`.

| capability | decision | reason |
|---|---|---|
| crsp_a_stock.dsf_v2 (daily security data, pre-joined with per-day type columns and cumulative factors) | INTEGRATE | |
| crsp_a_stock.stksecurityinfohist (security info / ticker, trading symbol, share class history) | INTEGRATE | |
| crsp_a_stock.stkdelists (delisting events) | INTEGRATE | |
| crsp_a_stock.stkdistributions (distribution events) | INTEGRATE | |
| crsp_a_indexes.dsp500list_v2 (CRSP point-in-time S&P 500 membership) | INTEGRATE | |
| comp.idxcst_his for gvkeyx 000208 (Nasdaq-100 constituent history) | INTEGRATE | |
| crsp_a_ccm.ccmxpf_lnkhist (dated gvkey/iid -> PERMNO links) | INTEGRATE | |
| pg_namespace has_schema_privilege (entitlement probe per schema) | INTEGRATE | |
| information_schema.columns (pinned-column check for dsf_v2) | INTEGRATE | |
| count(*) per (year, PERMNO batch) and per reference table (volume probe, completeness check) | INTEGRATE | |
| max(dlycaldt) over dsf_v2 (annual product-end probe) | INTEGRATE | |
| COPY (SELECT ... WHERE ...) TO STDOUT CSV (bulk pull) | INTEGRATE | |
| crsp_a_stock.stkdlysecuritydata | OPT-OUT | not needed — same rows as dsf_v2 without the per-day type columns and cumulative factors; kept only as the one-line `CrspQueries.DAILY_TABLE` fallback (D-19) |
| crsp_a_stock.wrds_dsfv2_query | OPT-OUT | explicitly rejected — duplicates (permno, dlycaldt) rows (360 in 2020, D-19) |
| crsp_a_stock.stocknames_v2 | OPT-OUT | not needed — lacks tradingsymbol; stksecurityinfohist carries everything symbology needs |
| crsp_a_stock.stkissuerinfohist | OPT-OUT | not needed — usincflg / issuertype arrive per day on dsf_v2 |
| CRSP monthly tables (msf_v2, stkmthsecuritydata) | OPT-OUT | explicitly out of scope — CRSP monthly is a deferred idea |
| crsp_m_* / crsp_q_* (monthly / quarterly update products) | OPT-OUT | not entitled (D-01) |
| crsp_a_indexes index levels (Nasdaq Composite, market returns) | OPT-OUT | not needed — CRSP index files hold no Nasdaq-100 constituents (D-14) |
| comp fundamentals (funda / fundq) and other CCM uses | OPT-OUT | explicitly out of scope (CONTEXT domain: other Compustat/CCM uses) |
| comp.idx_index | OPT-OUT | not needed — gvkeyx 000208 is pinned from the live check |
| crsp_a_ccm other tables (ccmxpf_linktable, lnkused, lnkrng, comphist, ...) | OPT-OUT | not needed — ccmxpf_lnkhist is the dated link table the join uses |
| crsp_q_mutualfunds.holdings (QQQ holdings) | OPT-OUT | not needed yet — a cross-check source only, not the universe (D-14) |
| legacy SIZ tables (crsp.dsf, dsenames, dsedelist) | OPT-OUT | deprecated — stopped updating after the December 2024 data |
| wrds Python package Connection / raw_sql | OPT-OUT | explicitly out of scope — D-03 (03.9 D-20): prompts interactively and breaks under pandas 3; psycopg2 via the shared WrdsSession |
