# API Coverage — WRDS NYSE TAQ (millisecond product) on WRDS PostgreSQL

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
> Scope: the WRDS PostgreSQL surface this phase's provider (`quantlab/acquisition/wrds_taq.py`) could use.

| capability | decision | reason |
|---|---|---|
| taqm_YYYY.complete_nbbo_YYYYMMDD (NBBO records, 2003-present) | INTEGRATE | |
| information_schema.tables (trading-day enumeration) | INTEGRATE | |
| information_schema.columns (per-table column introspection, pre/post 2018 eras) | INTEGRATE | |
| has_schema_privilege (per-year entitlement probe) | INTEGRATE | |
| count(*) per day table and symbol batch (volume probe) | INTEGRATE | |
| COPY (SELECT ... WHERE ...) TO STDOUT CSV (bulk pull) | INTEGRATE | |
| taqm_YYYY.nbbom_YYYYMMDD (SIP NBBO file) | OPT-OUT | explicitly out of scope — incomplete (misses single-venue NBBO states); D-01/D-18 chose complete_nbbo |
| taqmsec.* views | OPT-OUT | not needed — views over the same base tables; D-18 pins the taqm_YYYY base tables |
| taqm_YYYY.cqm_YYYYMMDD (per-exchange quotes) | OPT-OUT | explicitly out of scope — CONTEXT deferred idea |
| taqm_YYYY.ctm_YYYYMMDD (trades) | OPT-OUT | explicitly out of scope — CONTEXT deferred idea |
| TAQ-CRSP link tables (permno linking) | OPT-OUT | explicitly out of scope — CONTEXT deferred idea |
| luld_cqm / luld_ctm / mastm / wct daily tables | OPT-OUT | not needed — no decision in this phase consumes them |
| wrds Python package Connection / raw_sql | OPT-OUT | explicitly out of scope — D-20: prompts interactively and breaks under pandas 3; psycopg2 used directly |
