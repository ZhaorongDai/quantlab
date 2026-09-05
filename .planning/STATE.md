---
gsd_state_version: 1.0
milestone: v1.0
current_phase: 03
current_phase_name: Factor Computation (KunQuant + Polars)
status: verifying
stopped_at: Completed 03-05-PLAN.md
last_updated: "2026-09-05T19:04:47.719Z"
last_activity: 2026-09-05
last_activity_desc: Phase 03 execution started
state_head: 76475682eb9ca0322abe270dc41c942aec667983
progress:
  total_phases: 7
  completed_phases: 0
  total_plans: 18
  completed_plans: 18
milestone_name: milestone
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-04)

**Core value:** 一条打通的、config 驱动可复现的量化流水线（数据→因子→收益模型→组合优化→目标持仓→回测→结果），模块间用清晰的输入输出契约组合，任何一环都能独立替换/扩展而不需要推倒重来。
**Current focus:** Phase 03 — Factor Computation (KunQuant + Polars)

## Current Position

Phase: 03 (Factor Computation (KunQuant + Polars)) — EXECUTING
Plan: 5 of 5
Status: Phase complete — ready for verification
Last activity: 2026-09-05 — Phase 03 execution started

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

### Pending Todos

- Broader test-code/temporary-script cleanup (`test.py`, `test_nt.ipynb`, `read_mock_data_sink.py` as CLEAN-02 "test code / temporary scripts" candidates beyond the Phase 1 hardcoded-path fix applied to `test.py`) is intentionally deferred to Phase 7 (QUAL-02: "no leftover test/temp/redundant code remains outside the established `tests/` structure"), not silently dropped from Phase 1. Phase 1 only fixes `test.py`'s hardcoded per-developer paths (01-01 Task 3); it does not relocate, rename, or remove these files.

### Blockers/Concerns

- Tiingo API key currently leaked in `scripts/download_stock_data_from_tiingo.py` and pushed to `origin/main` — user should revoke/rotate the key in the Tiingo dashboard independent of the git-history reset planned in Phase 1.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | PLAT-01..06 (service API, multi-user, online factor/model editing, web frontend) | Deferred to v2 | Initial requirements definition |
| v2 | DATA-V2-01 (full tick-data production ingestion) | Deferred to v2 | Initial requirements definition |
| v2 | DATA-V2-02 (full minute-frequency historical backfill) | Deferred to v2 | Initial requirements definition |
| Phase 7 | `test.py`, `test_nt.ipynb`, `read_mock_data_sink.py` broader cleanup/relocation (CLEAN-02 test-code candidates) | Deferred to Phase 7 (QUAL-02) | Phase 1 planning revision, 2026-09-04 |

## Session Continuity

Last session: 2026-09-05T19:04:47.693Z
Stopped at: Completed 03-05-PLAN.md
Resume file: None
</content>
