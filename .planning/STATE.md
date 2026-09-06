---
gsd_state_version: 1.0
milestone: v1.0
current_phase: 03.1
current_phase_name: Index Historical Constituents Data Layer (INSERTED)
status: verifying
stopped_at: Completed quick task 260906-0iy (us_all roster + Tiingo bulk ingest)
last_updated: "2026-09-06T04:46:38.487Z"
last_activity: 2026-09-05
last_activity_desc: Phase 03.1 execution started
state_head: 767b33f49fc11e4d6224b1bbecdee0f1afaa4be3
progress:
  total_phases: 8
  completed_phases: 0
  total_plans: 24
  completed_plans: 22
milestone_name: milestone
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-04)

**Core value:** 一条打通的、config 驱动可复现的量化流水线（数据→因子→收益模型→组合优化→目标持仓→回测→结果），模块间用清晰的输入输出契约组合，任何一环都能独立替换/扩展而不需要推倒重来。
**Current focus:** Phase 03.1 — Index Historical Constituents Data Layer (INSERTED)

## Current Position

Phase: 03.1 (Index Historical Constituents Data Layer (INSERTED)) — EXECUTING
Plan: 4 of 4
Status: Phase complete — ready for verification
Last activity: 2026-09-05 — Phase 03.1 execution started

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 03 P01 | 31 min | 2 tasks | 7 files |
| Phase 03 P02 | 6 min | 3 tasks | 3 files |
| Phase 03 P03 | 7 min | 3 tasks | 5 files |
| Phase 03 P04 | 8 min | 3 tasks | 4 files |
| Phase 03 P05 | 15 min | 3 tasks | 4 files |
| Phase 03.1 P01 | 6 min | 3 tasks | 8 files |
| Phase 03.1 P02 | 7 min | 2 tasks | 4 files |
| Phase 03.1 P03 | 15 min | 3 tasks | 7 files |
| Phase 03.1 P04 | 18 min | 3 tasks | 7 files |
| Phase quick-260906-0iy P01 | 15 min | 3 tasks | 10 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: Phases follow the pipeline's natural sequential dependency (data → factor → model → portfolio → backtest) per the project's own Key Decision to deliver the full chain with baseline implementations at every stage before deepening any one stage.
- Roadmap: Phase 1 (cleanup/security) is a mandatory precondition — later phases assume a working `uv sync`, no leaked credentials, and accurate docs.
- Roadmap: QUAL-01/QUAL-02 (tests, code quality) placed in a dedicated final Phase 7 since they apply across all core modules built in Phases 2-6.
- [Phase 03]: my_ops composite ops accept and ignore `options` rather than reading it — Neither WindowedZScore nor WindowedRobustStandardization needs decomposition options; accepting-and-ignoring is a strict superset of prior behaviour and matches the KunQuant-shipped ops in KunQuant/ops/CompOp.py that do the same. Reading options would have been unverifiable new behaviour on a bug-fix commit.
- [Phase 03]: Phase-3 test fixtures write Zarr stores directly, bypassing raw CSV/parquet ingestion — Factor windows need ~60 timestamps x 8 symbols; driving that through SpotKlineDataset._raw_data_to_xr() would make every factor test also a CSV-parsing test. Writing the synthetic panel straight to Zarr keeps factor tests testing factors, at ~0.5s each.
- [Phase 03]: Each Wave-0 scaffold test file ships one real infrastructure self-test, never a placeholder — A file with zero tests makes its per-file pytest command exit 5 ("no tests ran"), which reads as green. Each scaffold instead asserts the exact fixture contract the plan that fills it depends on, so it has value alone and fails loudly if that contract drifts.
- [Phase 03]: Factor mode-divergent members (_auto_filter/num_symbols/symbols) carry the batch-shaped body on the shared base, with FactorKunQuant restoring exact behaviour via super()-delegating overrides — A naive hoist would be invisible today and fatal for 03-04: config.mode lives only on FactorConfig, so PolarsFactorConfig would raise AttributeError at runtime, only on the path someone happens to call. Test-time source-introspection enforcement replaces that runtime surprise.
- [Phase 03]: Factor._maybe_resolve_factor_names() ships now with the pre-refactor eager body as its default, even though nothing overrides it yet — Introducing the seam and preserving behaviour are the same edit; 03-04 FactorPolars overrides it to a no-op for D-05 dynamic schema resolution. Adding it later would mean re-touching the config setter in a plan with other things to prove.
- [Phase 03]: No Factor.config_cls ClassVar was added - utils/module.py hardcoded FactorConfig(**config) checkpoint reload is deferred to Phase 4 — The path is only reachable from Phase 4 model-checkpoint reload and no Phase-3 success criterion touches it; an unused attribute violates QUAL-02 and would pre-decide a mechanism Phase 4 should choose for itself.
- [Phase 03]: amount = volume * close is synthesized once centrally in StockDataset._to_kunquant(), double-guarded, not per factor class — The boundary method is the one place every KunQuant consumer of US-equity data passes through; per-class fixes would need one edit per factor class forever and each is a place to get adjusted-vs-raw wrong. The double guard (requested AND absent) keeps it inert and stops it overwriting a real vendor column if one ever appears (T-03-03-01).
- [Phase 03]: Alpha158Stock replicates Alpha158SpotKline verbatim (double-AllData build, six unused Input nodes); the normalization wrapper is the single deliberate divergence — D-01 fixes the invocation shape across markets, so tidying one class alone would silently diverge them and tidying both would edit working code for cosmetics. A vars() method-set equality assertion is the mechanical guard.
- [Phase 03]: The D-09 four-class normalization matrix is locked by one equality assertion against a literal, with docstrings on all four classes — Raw vs normalized factor values are indistinguishable to a downstream consumer at runtime (T-03-03-02). A per-market strategy choice that looks like an inconsistency must be impossible to 'align' silently; the assertion message routes a would-be changer to D-09 in 03-CONTEXT.md before the test literal.
- [Phase 03]: FactorPolars resolves factor names inside cal() from collect_schema().names(), never at construction time — Keeping the inherited eager _maybe_resolve_factor_names() would make merely constructing a Polars factor trigger a disk read, contradicting D-04's computation-starts-at-cal() contract. The documented cost is that _get_factor_names()/num_factors raise until cal()/read() has run - a gap base/model.py never hits.
- [Phase 03]: Polars factors are not portable across markets: Dataset.get_lazyframe() applies no per-market column normalization (03-RESEARCH.md Open Question 2), so factor/momentum.py is written against crypto-spot Title-Case Close — D-08 requires only one example factor and adding a get_lazyframe()-side rename layer was explicitly out of Phase 3 scope. Recorded in both docstrings a new factor author reads; revisit when a second Polars factor must span both markets.
- [Phase 03.1]: BaseFactorConfig.dataset is annotated "MarketDataset", not "BaseDataset" — FactorKunQuant calls dataset.to_kunquant(), a MarketDataset-only member, so widening the annotation would be a lie that type-checks. A future constituent-masking consumer takes a BaseDataset of its own; do not pre-widen.
- [Phase 03.1]: The pre-split name base.data.Dataset was retired, not kept as a permanent alias — Two live names for one class is exactly the ambiguity a later reader 'fixes' wrongly, and an unused alias violates QUAL-02. A transitional alias existed for exactly one commit so the split could land with a green suite; test 7 makes reintroducing it a test failure.
- [Phase 03.1]: _reset_symbols() became a named overridable seam on BaseDataset rather than an isinstance branch or a config boolean flag — It is called from the shared config setter, so pushing it down to MarketDataset would force the setter to split too. Overriding it to a no-op is proven to suppress both the store read and the from_raw_data() network fallback, which is what lets 03.1-02 construct a constituent dataset offline.
- [Phase 03.1]: BaseDataset.__init__ keeps its data_backend-first order, the deliberate inverse of base/factor.py:Factor.__init__ — The dataset config setter DOES reach the storage backend via _reset_symbols() -> read(), whereas no method reachable from the factor setter may. Both invariants are now locked by their own inverse guard tests so neither hierarchy can be silently harmonized onto the other.
- [Phase 03.1]: Nasdaq-100 anchor comes from stockanalysis.com with slickcharts.com documented in the class docstring as the fallback, deliberately not implemented as a second code path — Wikipedia's Nasdaq-100 page renders components through a navbox template with no parseable constituents table, so a commercial scraped source was unavoidable. Implementing both sources doubles the untested surface for a failure mode that has not happened; naming the alternative in the docstring is what a maintainer actually needs the day the primary dies.
- [Phase 03.1]: The Nasdaq-100 anchor is asserted at 102 rows, not 100, and fetch_anchor() carries its own MIN_ANCHOR_ROWS=50 structural-drift guard — The index carries multiple share classes for some issuers (GOOGL/GOOG, FOX/FOXA), so an ==100 assertion fails against correct data. Unlike the S&P anchor's hosted CSV, this scraped page has no cache-fallback path, so a truncated parse would close every unmentioned membership and silently reintroduce the survivorship bias the layer exists to remove (T-03.1-02-02).
- [Phase 03.1]: Only two message strings were parameterised via INDEX_LABEL/CATEGORY when hoisting fetch_changes(); both ValueError guard messages stayed word-for-word — The fallback logger.error and _load_cache's RuntimeError are the only index-specific text. Keeping the missing-columns and row-count-monotonicity messages verbatim means the S&P path renders byte-identical output post-extraction, so the refactor is provably behaviour-preserving rather than merely test-passing.
- [Phase 03.1]: The Nasdaq-100 panel gets its own nasdaq100_constituent.zarr store rather than sharing the S&P panel's — The two coverage starts are ~31 years apart (1976-07-01 vs 2007-02-01), and in a boolean panel a fabricated pre-coverage region is indistinguishable at read time from a genuine 'nobody was a member'. A consumer wanting both opens both and joins on the intersection of their timestamp axes -- a deliberate, visible step rather than an implicit and wrong union.
- [Phase 03.1]: UniverseCatalog's category list and coverage guard are driven by a MEMBERSHIP_FETCHERS registry, with NasdaqUniverseFetcher deliberately outside it — The pre-existing guard hardcoded sp500_constituent, so the newly-added nasdaq100_constituent category would have answered pre-2007 queries with a silently incomplete roster -- the exact failure DATA-05 exists to prevent. Deriving each boundary from the fetcher's own PIT_COVERAGE_START makes a boundary-less category impossible to create by omission. nasdaq_all stays out because it is a full-exchange roster with no membership-interval semantics and no coverage start (D-02); test_nasdaq_all_has_no_coverage_boundary makes that absence a tested property.
- [Phase 03.1]: DATA-06 was proved with a baseline SHA pinned before any edit, not a working-tree check — GSD commits per task, so 'git status --porcelain -- base/' alone prints nothing in exactly the world where base/ was edited and committed -- a proof that cannot fail. Pinning BASE_SHA into the git dir before Task 1 and gating on 'git diff BASE_SHA..HEAD -- base/' PLUS the working tree keeps the claim red after each task commit. Final result for 8be0adb: both halves empty.

### Pending Todos

- Broader test-code/temporary-script cleanup (`test.py`, `test_nt.ipynb`, `read_mock_data_sink.py` as CLEAN-02 "test code / temporary scripts" candidates beyond the Phase 1 hardcoded-path fix applied to `test.py`) is intentionally deferred to Phase 7 (QUAL-02: "no leftover test/temp/redundant code remains outside the established `tests/` structure"), not silently dropped from Phase 1. Phase 1 only fixes `test.py`'s hardcoded per-developer paths (01-01 Task 3); it does not relocate, rename, or remove these files.

### Blockers/Concerns

- Tiingo API key currently leaked in `scripts/download_stock_data_from_tiingo.py` and pushed to `origin/main` — user should revoke/rotate the key in the Tiingo dashboard independent of the git-history reset planned in Phase 1.

### Roadmap Evolution

- Phase 03.1 inserted after Phase 3: Index Historical Constituents Data Layer — daily point-in-time S&P 500 / Nasdaq-100 membership panels as xarray (URGENT)

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | PLAT-01..06 (service API, multi-user, online factor/model editing, web frontend) | Deferred to v2 | Initial requirements definition |
| v2 | DATA-V2-01 (full tick-data production ingestion) | Deferred to v2 | Initial requirements definition |
| v2 | DATA-V2-02 (full minute-frequency historical backfill) | Deferred to v2 | Initial requirements definition |
| Phase 7 | `test.py`, `test_nt.ipynb`, `read_mock_data_sink.py` broader cleanup/relocation (CLEAN-02 test-code candidates) | Deferred to Phase 7 (QUAL-02) | Phase 1 planning revision, 2026-09-04 |

## Session Continuity

Last session: 2026-09-06T04:46:38.252Z
Stopped at: Completed quick task 260906-0iy (us_all roster + Tiingo bulk ingest)
Resume file: None
</content>
