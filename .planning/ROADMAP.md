# Roadmap: quantlab

## Overview

quantlab starts as a ~6000-line personal prototype (layered `DataBackend → Dataset → Factor/Label → Model → Backtest` architecture, KunQuant factor engine, xarray/Zarr storage, torch models, vectorbt/nautilus_trader backtesting) that is functionally rich but operationally broken: it has a leaked API key in git history, an unresolvable dependency lockfile, hardcoded per-developer paths, and a stale README. The roadmap first makes the repository clean, secure, and runnable (Phase 1), then builds a full config-driven vertical MVP pipeline — data → factor → return model → portfolio optimization → target holdings → backtest — one pipeline stage at a time, each stage delivering a simple/baseline but fully working, independently swappable capability that the next stage builds on. The data layer is designed from the start to support multiple markets and frequencies. The pipeline closes with an end-to-end, config-reproducible run and a final pass that adds unit tests and a code-quality cleanup across all core modules.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Codebase Cleanup & Security Hardening** - Repo is clean, secure, dependency-resolvable, and documented accurately (completed 2026-09-05)
- [x] **Phase 2: Multi-Market Data Foundation** - Users can ingest US equities and Binance spot data into unified, extensible xarray/Zarr storage (completed 2026-09-05)
- [ ] **Phase 3: Factor Computation (KunQuant + Polars)** - Users can compute Alpha158 factors (batch + streaming) via KunQuant and new factors via Polars
- [x] **Phase 03.1: Index Historical Constituents Data Layer** - Point-in-time index membership panels (completed 2026-09-07)
- [x] **Phase 03.2: Multi-Source Data Acquisition Abstraction (Alpaca)** - Second vendor through the same batched, resumable, volume-guarded `Acquisition` abstraction (completed 2026-09-06)
- [ ] **Phase 03.3: Tick Data Storage (Non-Dense Event Axis)** - Raw tick shards reach a persisted store via a tick-specific Dataset with a non-dense event axis
- [ ] **Phase 03.4: Data Source Registry (Operator-Surface Foundation)** - One registered descriptor per data source, consumed by quantlab's own CLI and by the out-of-repo `quantlab-console` operator surface
- [ ] **Phase 03.5: Registry-level raw→Zarr Conversion Entry Point** - A conversion entry point beside `registry.run()`, with the mode chosen explicitly and its RAM guard answerable before the run
- [ ] **Phase 4: Baseline Return Prediction Model** - Users can train a baseline model that consumes factor xarray data and outputs return predictions
- [ ] **Phase 5: Portfolio Optimization & Target Holdings** - Users can turn predictions into long-short, unlevered target holdings
- [ ] **Phase 6: End-to-End Backtest & Reproducible Pipeline** - Full pipeline runs end-to-end from one config, verified via vectorbt backtest
- [ ] **Phase 7: Testing & Code Quality** - Core modules have unit tests and a Zen-of-Python-consistent code style

## Phase Details

### Phase 1: Codebase Cleanup & Security Hardening

**Goal**: The repository is a clean, secure, working foundation — no leaked credentials, a working dependency environment, accurate docs, and no dead/broken/duplicate code — ready for new development.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04, SEC-01
**Success Criteria** (what must be TRUE):

  1. `git log` on the repo shows a fresh history with no commit containing the Tiingo API key (verified by searching full history)
  2. Running `uv sync` from a clean clone produces an environment where all actually-imported third-party packages (torch, xarray, polars, KunQuant, vectorbt, nautilus_trader, etc.) import successfully
  3. No API keys or other secrets are hardcoded anywhere in the codebase; the Tiingo key is read from a `TIINGO_API_KEY` environment variable
  4. `README.md` accurately describes the current module layout and how to run the pipeline (no references to non-existent files)
  5. Known broken/duplicate code found during codebase review is fixed or removed (`vecbt/bt.py:backtest_from_signals` runs without error when called with valid arguments; duplicate Binance instrument-parsing logic is consolidated into one implementation)

**Plans:** 5/5 plans complete

Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Secrets & config hardening (Tiingo key -> env var, config/__init__.py paths -> QUANTLAB_DATA_DIR)
- [x] 01-02-PLAN.md — Dependency reconciliation (pyproject.toml/uv.lock -> working uv sync)
- [x] 01-03-PLAN.md — Dead/duplicate code cleanup (vecbt/bt.py fix, Binance parsing dedup)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-04-PLAN.md — README accuracy pass

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-05-PLAN.md — Git history reset (checkpoint-gated, runs last)

### Phase 2: Multi-Market Data Foundation

**Goal**: Users can ingest and store market data for multiple markets/frequencies through one extensible Dataset/DataBackend abstraction, with all data landing in canonical xarray/Zarr storage.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):

  1. User can run a documented command/script to pull US equities daily data from Tiingo (auth via env var) and see it persisted as a `[timestamp, symbol]` xarray.Dataset in Zarr
  2. User can run a documented command/script to pull/refresh Binance spot kline data through the same Dataset/DataBackend abstraction into the same Zarr storage format
  3. Both data sources pass through a shared cleaning/preprocessing module (`dataset/cleaning.py` — planner discretion per 02-CONTEXT.md, not `my_ops` which is KunQuant-graph-op style and the wrong fit for tabular/xarray cleaning) before being persisted, producing a valid `xarray.Dataset`
  4. A design/code review confirms adding a new market or frequency only requires a new `Dataset` subclass + config — no changes needed in factor/model/backtest code

**Plans:** 8/8 plans complete

Plans:
**Wave 1**

- [x] 02-01-PLAN.md — Test infrastructure (pytest + fixtures) + Zarr overwrite fix (Pitfall 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-02-PLAN.md — Config schema foundation: DatasetConfig market/frequency, AcquisitionConfig, path convention retrofit, stock_kline_config()
- [x] 02-03-PLAN.md — Shared cleaning module (dataset/cleaning.py) wired into Dataset.from_raw_data()

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 02-04-PLAN.md — Acquisition(ABC) + TiingoAcquisition (encapsulated fetch + incremental refresh)
- [x] 02-05-PLAN.md — Binance retrofit vertical slice (SpotKlineDataset dedup + ingest_binance_spot.py)
- [x] 02-06-PLAN.md — Extensibility contract proof (FakeDataset lifecycle + core-layer purity check)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 02-07-PLAN.md — StockDataset Tiingo integration + ingest_tiingo.py (completes US-equities slice end-to-end)

**Supplement** *(added after original 7 plans completed, per D-12/D-12a — independent, wave 1, no dependency on the above)*

- [x] 02-08-PLAN.md — Survivorship-bias-free, point-in-time US equity universe (NASDAQ-listed incl. delisted + point-in-time S&P 500 membership via acquisition/universe.py, wired into ingest_tiingo.py --universe/--as-of-date)

### Phase 3: Factor Computation (KunQuant + Polars)

**Goal**: Users can compute the existing Alpha158 factor set (batch and streaming) via KunQuant, and can add new factors via a new Polars batch backend, with xarray.Dataset as the sole exchange format.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: FACTOR-01, FACTOR-02, FACTOR-03, FACTOR-04
**Success Criteria** (what must be TRUE):

  1. User can compute the Alpha158 factor set in batch mode from stored market data and get back an `xarray.Dataset`
  2. User can invoke KunQuant's streaming (`cal_stream`) factor computation path and get incremental factor updates without error
  3. User can compute at least one new factor via the Polars batch backend and get output conforming to the same `xarray.Dataset` contract
  4. No factor-pipeline code path passes a plain DataFrame between modules — inputs/outputs are `xarray.Dataset` only

**Plans:** 7 plans (5 executed; 2 gap-closure plans pending after the `gaps_found` verdict)

Plans:
**Wave 1**

- [x] 03-01-PLAN.md — Unblock FACTOR-01: fix the `my_ops` decompose() signature drift that breaks every batch `cal()`, prove Alpha158/Alpha101 end-to-end, build the Wave-0 test fixtures

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03-02-PLAN.md — Extract the shared `Factor` ABC from `FactorKunQuant`, split `FactorConfig` into base/KunQuant/Polars siblings, widen `DLConfig`/`MLConfig` (D-03)

**Wave 3** *(blocked on Wave 2 completion — the two plans below are independent and run in parallel)*

- [x] 03-03-PLAN.md — US-equity KunQuant coverage: central `amount = volume * close` proxy, `Alpha101Stock` crash fix, new `Alpha158Stock`, config factories, NORM-01 normalization matrix (D-01/D-02)
- [x] 03-04-PLAN.md — `FactorPolars(Factor)` batch backend + the `Momentum` example factor (D-04..D-08)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03-05-PLAN.md — Streaming smoke test (fixes the arch-dependent SIMD-width crash), live two-backend interchangeability proof, README coverage

**Gap closure — Wave 1** *(added after 03-VERIFICATION.md returned `gaps_found`; the two plans below touch disjoint files and run in parallel)*

- [ ] 03-06-PLAN.md — Close CR-01: resolve Polars factor names on the `read()` path so `factor_data_strategy="read"` is backend-independent (D-03), plus the two-backend read-strategy lock
- [ ] 03-07-PLAN.md — Close CR-02: US-equity `amount` becomes typical-price dollar volume (GAP-D-01) so `vwap` is no longer identically `close`, plus the VWAP non-degeneracy lock

### Phase 03.1: Index Historical Constituents Data Layer (INSERTED)

**Goal**: Users can obtain daily point-in-time membership panels for the S&P 500 and the Nasdaq-100 as `xarray.Dataset`, so any downstream layer can mask its universe per day without survivorship bias.
**Mode:** mvp
**Requirements**: DATA-05, DATA-06
**Depends on:** Phase 3
**Success Criteria** (what must be TRUE):

  1. User can obtain a daily S&P 500 and a daily Nasdaq-100 membership panel as an `xarray.Dataset` with dims `timestamp`/`symbol` and a boolean `is_member` variable, persisted to and reloaded from Zarr
  2. Each panel is survivorship-bias-free within its own coverage window: its symbol axis is the all-time union so delisted former members are real mostly-False columns, its left edge is clamped to that index's point-in-time coverage start (1976-07-01 / 2007-02-01), and a removal's effective date is still a membership day — matching `UniverseCatalog.get_symbols_as_of()` exactly
  3. A membership query before an index's coverage start raises rather than returning a silently incomplete roster
  4. The constituent dataset classes and the market dataset classes share one `BaseDataset` abstraction, with no OHLCV-only member (`_to_kunquant`/`_to_nautilus`) reachable from the constituent side, and adding a further index requires zero edits under `base/`

**Plans:** 5/5 plans complete

Plans:
**Wave 1**

- [x] 03.1-01-PLAN.md — Split `base/data.py` into `BaseDataset`/`MarketDataset` and `DatasetConfig` into a three-way config split (wave 1)
- [x] 03.1-02-PLAN.md — Extract `IndexMembershipFetcher`, add `Nasdaq100MembershipFetcher` and the third universe category (wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03.1-03-PLAN.md — `IndexConstituentDataset` + `SP500ConstituentDataset`: interval-to-daily-panel densification with all six correctness locks (wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03.1-04-PLAN.md — `Nasdaq100ConstituentDataset`, registry-driven `UniverseCatalog` coverage guards, README (wave 3)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03.1-05-PLAN.md — Gap closure (VERIFICATION truth 3 / DATA-05): extend the registry-driven coverage guard to `get_symbols_in_range()` so a pre-coverage window raises instead of silently returning a censored roster (wave 4)

### Phase 03.2: Multi-Source Data Acquisition Abstraction (Alpaca) (INSERTED)

**Goal:** Users can acquire market data from a second vendor (Alpaca) through the same `Acquisition` abstraction as Tiingo, where that abstraction now natively supports batched multi-symbol requests, page-level resumable pagination, and a pre-flight volume guard — with both vendors' raw data coexisting under a vendor-namespaced, hive-partitioned layout.
**Requirements**: TBD (no existing REQ-ID covers vendor-level source extensibility; closest sibling is DATA-03's market/frequency extensibility)
**Depends on:** Phase 3
**Canonical refs:** `.planning/phases/03.2-multi-source-data-acquisition-abstraction-alpaca/03.2-CONTEXT.md`
**Plans:** 7/7 plans complete

**Success Criteria** (what must be TRUE):

  1. `Acquisition`'s fetch primitive is batched (multi-symbol per call); a single-symbol vendor is expressed as the degenerate case, with no `NotImplementedError` stub anywhere in the hierarchy
  2. Concurrency, resume, failure isolation and the global quota abort live once in the base class and drive both Tiingo and Alpaca; every behaviour locked by quick task 260906-26o still holds
  3. An interrupted paginated fetch resumes at the failed page rather than at the start of the batch
  4. A symbol absent from a batch response is recorded as "queried, no data" — distinguishable at read time from both a fetch failure and a never-fetched symbol
  5. User can fetch Alpaca daily bars, minute bars, and quotes/trades to raw storage, selecting the vendor by config
  6. A fetch whose estimated volume exceeds the guard threshold refuses before issuing requests, and can be overridden explicitly
  7. Tiingo and Alpaca raw data land under separate vendor path segments and cannot be silently merged into one xarray

**Scope fences:** raw → xarray/Zarr conversion for tick is deferred to a follow-up phase; Alpaca's trading/broker API and corporate actions are out of scope; this delivers acquisition capability, not a full-market minute/tick backfill (DATA-V2-01/02 remain v2).

Plans:
**Wave 1**

- [x] 03.2-01-PLAN.md — Wave-0 validation scaffolding: paginating/hive/config fixtures plus the five new test files, each with a real self-test

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03.2-02-PLAN.md — TRACER: one Alpaca daily batch end to end through the batched `_fetch_page` primitive to a vendor-namespaced hive shard, then page-level resume and structural cross-vendor isolation (SC-1, SC-3, SC-5, SC-7)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03.2-03-PLAN.md — D-02 orchestration lift: concurrency, resume, failure isolation and the global abort hoisted onto `Acquisition`; `ConcurrentTiingoAcquisition` retired; per-vendor `_classify_error` (SC-1, SC-2)
- [x] 03.2-04-PLAN.md — Pre-flight volume guard on `UniverseCatalog`: estimate then refuse on bytes, requests and wall clock, with an explicit override (SC-6)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03.2-05-PLAN.md — D-04 "queried, no data" third state, additive on disk and gated on batch completion (SC-4)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 03.2-06-PLAN.md — Alpaca minute bars and quotes/trades at full resolution, US/Eastern session-date hive key, tier limitation documented on the class (SC-5, SC-7)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 03.2-07-PLAN.md — `utils/cli.py` shared argument groups, `ingest_alpaca.py`, the guard wired into every entry point, and the three credential-dependent verifications (SC-5, SC-6)

**Cross-cutting constraints:**

- All 43 tests in `tests/test_tiingo_acquisition.py` + `tests/test_tiingo_quota.py` are green at every commit (D-02 hard constraint — this plan modifies `base/acquisition.py`)

### Phase 03.3: Tick Data Storage (Non-Dense Event Axis)

**Goal**: Raw tick shards (quotes/trades) reach a queryable, persisted store through a tick-specific
`Dataset` whose storage shape is a non-dense EVENT axis -- never the dense `[timestamp, symbol]` panel.
**Depends on**: Phase 03.2
**Requirements**: TBD (D-16, D-18 carried forward from 03.2)
**Success Criteria** (what must be TRUE):

  1. TBD -- run `/gsd-discuss-phase 03.3` to settle the storage shape, then `/gsd-plan-phase 03.3`

**Carried-forward constraints from Phase 03.2 (do NOT re-derive):**

- The dense `[timestamp, symbol]` panel CANNOT express an irregular event axis. Reaching for it is what
  produced the out-of-memory failure in quick task 260906-13w -- accurately: a ~7.2 GiB float64
  grid held together with a ~29.6M-row pandas frame and conversion scratch OOM'd a 16 GiB machine,
  on FULL-MARKET DAILY data, not on tick. (The 03.2-06-SUMMARY wording this constraint was copied
  from compresses that into "the 16 GiB OOM", which reads as though the panel were 16 GiB and as
  though tick caused it; `ingest_us_equity.py`'s module header is the accurate source.) Tick would
  be far worse, which is the point -- but the recorded failure is not tick's.
  `_scan_raw` already reads tick
  correctly -- root-scoped, symbol preserved, UNDEDUPED -- and stops there.
- Many genuine quotes/trades share one `(timestamp, symbol)`; that is what a tick stream IS.
  `dedup_raw_frame` exists only for `.to_xarray()`'s unique-index requirement and must never be applied
  to tick (D-16 resolution preservation).
- Trade/quote condition-code and exchange-code decoding (`stocks/meta/conditions`, `stocks/meta/exchanges`)
  belongs to this phase, not to acquisition (03.2 COVERAGE.md).

**Plans**: TBD

### Phase 03.4: Data Source Registry (Operator-Surface Foundation) (INSERTED)

**Goal**: Every data source quantlab can acquire is described by ONE registered descriptor, so both
quantlab's own code and an out-of-repo operator console enumerate the same sources from the same
definition -- and adding a vendor is registering a descriptor, not editing five call sites.
**Depends on**: Phase 03.2 (the `Acquisition` abstraction, watermark/coverage sidecars, vendor path
segments), Phase 2 (`DatasetConfig`/Zarr layout)
**Requirements**: Settled in discussion 2026-09-08 -- see
`.planning/phases/03.4-data-source-registry/03.4-CONTEXT.md` (D-01..D-20). Downstream agents MUST
read it; several decisions deliberately override wording elsewhere in this section.
**Success Criteria** (what must be TRUE):

  1. Every acquirable source is enumerable from ONE registry with no vendor named at the call site,
     and each descriptor carries its capabilities, credential env var NAMES, and acquisition class.
  2. Enumerating sources and reading their credential status never returns a credential VALUE and
     never requires one to be present.
  3. The read-side surface answers inventory, coverage, failure reasons and row-level browsing on a
     machine with NO credentials configured, and issues zero vendor requests.
  4. Coverage judgement is computed by the same code the real acquisition run uses -- not a second
     implementation.
  5. A caller can start an acquisition programmatically, observe structured progress, cancel it at a
     batch boundary leaving the store and watermarks resumable, and receive a result object naming
     successes and failures.
  6. The in-repo ingest entry points reach their source through the registry, so the registry has a
     consumer in this repository.

**Why this phase exists, and what it deliberately EXCLUDES.**

The TUI/web operator console this registry serves lives in a SEPARATE repository
(`quantlab-console`, developer decision 2026-09-07), which depends on quantlab rather than the
reverse. Only the quantlab-side contract belongs here. The console's service layer, Textual TUI,
scheduler process and future web backend are OUT of this phase and out of this roadmap.

**In scope (quantlab side):**

- **`DataSourceRegistry`.** ONE descriptor per VENDOR, carrying a list of capability dataclasses
  (market / frequency / data type -- Alpaca's `tick` splits into `quotes` and `trades`), plus the
  acquisition class and config factory as direct class references, plus the credential env var
  names. Descriptors are dataclass INSTANCES in a class-level registration tuple, populated by a
  registration decorator. Enumeration is complete from a cold import -- that intent is unchanged --
  but the MECHANISM was amended on 2026-09-08 (CONTEXT D-07 AMENDED, developer-confirmed): the
  vendor imports sit at the BOTTOM of `quantlab/acquisition/registry.py`, and
  `quantlab/acquisition/__init__.py` STAYS EMPTY. A non-empty package `__init__.py` would run the
  vendor imports on every `import quantlab.acquisition.universe` -- the one module whose whole
  guarantee is that no acquisition client can be constructed there -- and it would do so while
  `tests/test_volume_guard.py` stayed green, because that test's structural arm is an AST scan of
  `universe.py`'s own source and cannot see a transitive import. Bottom placement is also what lets
  each descriptor be defined beside its own vendor class (`tiingo.py` / `alpaca.py` import
  `registry.py`, never the reverse), so the repo's empty-`__init__.py` convention turns out not to
  be broken after all. Cold-import completeness is proved by an import test that runs in a CHILD
  PROCESS, because a pytest session that has already imported both vendor modules would measure the
  session rather than the import graph. This follows the spirit of the idiom
  `UniverseCatalog.MEMBERSHIP_FETCHERS` / `ROSTER_FETCHERS` establishes -- one class-level
  registration tuple -- and the instance-vs-class difference is a deliberate 2026-09-08 decision
  (CONTEXT D-05), not a second registry style.
- **A programmatic write entry point, and progress/cancellation to go with it.** The console starts
  an acquisition by calling quantlab IN-PROCESS (see the boundary contract below), through
  `registry.run(descriptor, config, *, refresh=False, reporter=None, cancel=None)`. That entry point
  reports progress as event objects through a pluggable reporter (tqdm-backed by default), accepts
  a cancellation token checked at batch boundaries so a cancelled multi-hour backfill stays
  resumable, and returns an `AcquisitionResult`. It is ACQUISITION-ONLY and stops at the raw parquet
  tier (CONTEXT D-14 AMENDED 2026-09-08, developer-confirmed): the raw->Zarr conversion stays with
  the entry points, because the three of them convert in three different modes behind three
  differently-sized RAM guards, and folding that into one call would pick one of the three for
  everybody. If the console ever needs conversion, it gets a SEPARATE registry-level call, never a
  flag on `run()`. `_failures.json` is still written on every exit path, but it is NOT resume input:
  resume is driven entirely by watermark-sidecar presence (CONTEXT D-18 FACTUAL CORRECTION
  2026-09-08). It has TWO in-repo readers -- `SourceInspector.failures()`, the credential-free
  operator view, and `Acquisition._merge_unattempted_failures`, the pre-write merge that folds back
  the entries a run had no news about -- and both reach the file through the single tolerant reader
  `CoverageLedger.read_failure_manifest`, because two tolerant readers would be two copies of the
  failure policy free to drift. It is kept because it is the crash-durable operator record: a
  process that dies returns no `AcquisitionResult`. (FACTUAL CORRECTION 2026-09-09, plan 03.4-09:
  the earlier no-reader clause here was true when D-18 was corrected on 2026-09-08 and plan 03.4-05
  falsified it by adding the second reader; the set above is what `read_failure_manifest()`'s call
  sites under `quantlab/` were enumerated to be when this was written, not a count carried forward.)
  The file is a DURABLE CROSS-RUN record, not a per-run artifact: plan 03.4-08 made the pre-write
  merge unconditional and lifted it out of the resume loop to sit immediately before the write, so
  all five exit paths are covered BY CONSTRUCTION rather than audited one at a time. `_run` merges
  into `manifest = dict(failures)`, a COPY, and hands the run's own untouched `failures` to the
  result object (REVIEW CR-01, `c0e392f`), so the result and the file on disk are two INDEPENDENT
  values. The three relationships a console may write assertions against are therefore
  containments, never an equality in either direction: `set(result.failures) <=
  set(json.load(_failures.json))`; the MESSAGES agree on every key the two share; and
  `set(result.failures) <= set(result.requested)`, which is what makes the result alignable with
  `coverage` (built over `requested`). Read an empty `{}` as "every symbol this run had news about
  came back clean AND the disk held no other carried-forward entry" -- and read it in one direction
  only, because a NON-empty manifest likewise does not imply the preceding run failed: on a busy
  store most entries are folded back rather than re-observed. The operator protection this phase
  pins is asserted by the two tests that can fail on CONTENTS:
  `tests/test_acquisition_progress.py::test_the_manifest_survives_a_quota_abort_on_the_default_path`
  (after a quota abort on the DEFAULT path the previous run's entries are still in the manifest,
  read back from disk) and `::test_a_disjoint_rerun_does_not_inherit_earlier_failures` (a run over a
  roster disjoint from an earlier failing one is not inflated by that other roster's entries).
  (FACTUAL CORRECTION 2026-09-09, quick 260909-174: the equality this bullet asserted here was
  RETIRED on purpose by REVIEW CR-01, not merely discounted -- keeping it would have meant keeping
  the single shared dict that leaked earlier runs' failures into `AcquisitionResult`; the reasoning
  and the replacement contract are the D-18 row and the `result_and_manifest_agree` note in
  `.planning/phases/03.4-data-source-registry/03.4-VALIDATION.md`.)
- **Retire the hardcoded vendor dispatch.** `ingest_tiingo.py` -> `TiingoAcquisition` and
  `ingest_alpaca.py` -> `AlpacaAcquisition` are hardcoded at each call site today; there is no
  enumerable list an operator surface could render. These scripts become THIN SHELLS over the
  programmatic entry point, resolving their source through the registry. They stop being a
  human-facing surface -- the TUI becomes that -- but they are KEPT, because they are the
  registry's consumer IN THIS REPO and stop it rotting into a console-only side table, they are
  live proof the programmatic entry point works, and they are the fallback on a machine with no
  TUI. The redundancy question this bullet used to ask is ANSWERED (plan 03.4-06):
  `ingest_us_equity.py` is NOT redundant with `ingest_tiingo.py --universe us_all`. The two differ
  in roster mode (`in_range` vs `as_of`), in storage subdirectory and therefore watermark tree
  (`us_all` vs `tiingo`), in conversion mode (chunked, opt-in behind `--to-zarr` vs unconditional
  whole-window), in dataset symbols (`None` vs the resolved roster), and in three flags that exist
  on only one of them (`--dry-run`, `--stamp-legacy-watermarks`, `--chunk`/`--on-new-listing`). All
  THREE shells are therefore KEPT, and all three are thin: `ingest_tiingo.py` was also rerouted
  through `SOURCE.config_factory` in plan 06, because calling `stock_acquisition_config` directly
  produced an identical config only while `"tiingo"` remained that factory's incumbent default --
  the vendor was never actually routed.
- **The read-side query surface the console consumes in-process.** A read-only inspector that needs
  NO credentials -- constructing an `Acquisition` is not an option, because `TiingoAcquisition`
  raises without `TIINGO_API_KEY`. `Acquisition.coverage_report()` already answers the coverage
  question with ZERO vendor requests and shares `_partition_by_coverage` with the real run; the
  inspector must SHARE that logic, never re-implement it. Reachable without re-implementing it out
  of repo: per-source inventory (symbol count, coverage span, disk footprint, last-updated),
  per-symbol coverage and `_failures.json` reasons, and a LAZY row-level read for data browsing --
  `pl.scan_parquet` + hive pruning + slice for the raw tier, `.sel()` slicing for Zarr. `us_all` is
  ~15.4k symbols x ~5.2k trading days (~30M rows), so the lazy read takes symbols AND a date range
  as REQUIRED arguments: pruning always happens, and asking for a whole tier takes deliberately
  passing the full roster.
- **Credentials are reported as configured / not configured, never as values.** The descriptor names
  the env vars; nothing reads or returns them. This repository has a real leaked-key incident in its
  history (Phase 1), and an operator dashboard that prints an env var is how the next one happens.

**Explicitly OUT of scope (belongs to `quantlab-console`):** the Textual TUI, the service layer, the
scheduler process and its job files, run history/logs, the download form and progress UI, the CSV
export, the quality-check report rendering, and the future HTTP API and web frontend. Also out, and
newly so as of 2026-09-08: keeping a multi-hour backfill from blocking the UI (a thread/process pool
or job queue), installing a log sink to render acquisition logs, and preventing two concurrent
acquisitions of the same source. All three now live entirely in the console repository.

**Boundary contract with the console (revised 2026-09-08 -- supersedes the 2026-09-07 lock):**

- The console depends on quantlab; quantlab NEVER depends on the console.
- READS (coverage, inventory, row-level browsing, quality checks) run IN-PROCESS via a direct import
  of quantlab.
- WRITES (an actual acquisition run) ALSO run IN-PROCESS, via the programmatic entry point above.
  **SETTLED 2026-09-08 (CONTEXT D-12), shipped 2026-09-09.** This is the contract; the 2026-09-07
  decision -- the subprocess-CLI rule it supersedes, under which the console launched writes by
  invoking quantlab's CLI in a child process -- is no longer in force and must not be implemented.
  Two things that rule bought are hereby given up. Process isolation: keeping a multi-hour backfill
  from taking the console down is now the console's OWN job, which is exactly why quantlab must
  provide a cancel token checked at batch boundaries and structured progress events -- the loop now
  runs inside the console's process. And the "export the equivalent CLI command" guarantee, which
  was true by construction only while the exported command WAS what the console ran; any such export
  is now a reconstruction, and has to be tested as one rather than trusted.
- Concurrency control is the console's responsibility, and quantlab still adds NO lock (CONTEXT
  D-20). The atomicity question this bullet used to leave open is ANSWERED (plan 03.4-03): the
  watermark sidecar and `_failures.json` were NOT atomic -- both were plain `open(..., "w")` +
  `json.dump`, while `ChunkLedger._flush` and `PageLedger._flush` in the same tree already used temp
  file + `fsync` + `os.replace`. They now are, through one shared writer,
  `quantlab/utils/atomic.py:write_json_atomically`, which is the only `NamedTemporaryFile` left
  under `quantlab/` and is what all four JSON sidecar writers delegate to. No lock was added and
  none is planned. The residual risk stays visible: the thin-shell scripts run outside any console
  task queue, so a shell and the console can still overlap on one source -- the atomic write BOUNDS
  what that overlap can do (no half-written sidecar, no surviving `.tmp`) rather than preventing
  it.

**Plans:** 11/11 plans executed (9/11 executed; 03.4-10 and 03.4-11 close the 4th recurrence of the D-18 documentation-consistency gap in `03.4-VERIFICATION.md`)

Plans:
**Wave 1**

- [x] 03.4-01-PLAN.md — Wave-0 validation scaffolding: `isolated_registry` / `no_credentials`
  fixtures plus the five new test files, each with a real infrastructure self-test

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03.4-02-PLAN.md — TRACER: `DataSourceRegistry` + both descriptors + `AcquisitionResult` +
  the programmatic `run()`, driven end to end by `ingest_tiingo.py` as a thin shell (SC-1, SC-2,
  D-01..D-07, D-12, D-14, D-18)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03.4-03-PLAN.md — Atomic sidecar writes through one shared `write_json_atomically`, adopted
  by both existing ledgers and both acquisition writers (D-20, SC-5)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03.4-04-PLAN.md — `CoverageLedger` extraction (`Acquisition` delegates) + the credential-free
  `SourceInspector` with lazy row-level browsing (SC-3, SC-4, D-08..D-11)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 03.4-05-PLAN.md — Progress events through a pluggable reporter + a cancel token at the batch
  boundary, with the result object and the failure manifest kept in agreement (SC-5, D-13, D-16..D-19)

**Wave 6** *(blocked on Wave 5 completion — 06's shells call the `run()` signature 05 widens, and
both plans end on a whole-suite pytest gate, so they are sequenced rather than run in parallel)*

- [x] 03.4-06-PLAN.md — `ingest_alpaca.py` and `ingest_us_equity.py` reduced to thin shells over
  the registry, with a credential-free `--dry-run` coverage report (SC-6, SC-1, D-15)

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 03.4-07-PLAN.md — Documentation debt: rewrite the three superseded ROADMAP statements, mark
  `03.4-RESEARCH.md`'s open questions resolved, add `example/registry.md`, update the stale sections
  of `example/acquisition.md` (D-12, D-13, D-15, D-19, D-20)

**Wave 8** *(gap closure — blocked on `03.4-VERIFICATION.md`, which scored 5/6 and found two gaps)*

- [x] 03.4-08-PLAN.md — GAP 1 (SC-5): relocate the unattempted-failure merge to run unconditionally
  before the manifest write so every exit of `_run`'s resume loop is covered by construction, pinned
  by a red-first regression asserting on the manifest's on-disk CONTENTS after a default-path quota
  abort — not on the result/manifest equality, which is built from one dict and cannot fail
  (D-17, D-18)

**Wave 9** *(blocked on Wave 8 — the doc correction must describe the call site 08 leaves behind)*

- [x] 03.4-09-PLAN.md — GAP 2 (D-18 FACTUAL CORRECTION): the ROADMAP, `example/acquisition.md` and
  `example/registry.md` all deny that any module under `quantlab/` reads `_failures.json` and then
  name a reader two lines later; restate the reader set from a source enumeration run in the plan,
  and file REVIEW CR-01 on the defect ledger (D-18)

**Wave 10** *(blocked on Wave 9 — the 4th recurrence of the same defect; the acceptance method
itself is what changes, so it must be built after the 3rd attempt's word list is on the record)*

- [x] 03.4-10-PLAN.md — GAP (D-18, 4th recurrence): five sites still state the retired "last run"
  semantics — the docstring SUMMARY LINES of both manifest readers
  (`CoverageLedger.read_failure_manifest`, `SourceInspector.failures`), the `example/registry.md`
  API table, and two lighter wordings in `quantlab/base/acquisition.py` / `tests/test_tiingo_quota.py`.
  Replaces the acceptance method: a SENTENCE-LEVEL semantic-intersection enumerator
  (`manifest_sentence_audit.py`) produces the candidate set and every candidate is adjudicated into
  a committed ledger — candidates first, word list last (D-18)

**Wave 11** *(blocked on Wave 10 — the extended literal word list may only be derived from the
retired-sentence record plan 10 produces)*

- [x] 03.4-11-PLAN.md — Mutation-prove all three arms of the sentence-level gate (including a
  same-tree "literal scan green / candidate enumeration red" comparison), extend
  `manifest_semantics_scan.py` from 11 to 16 patterns derived from `manifest-sentence-retired.tsv`,
  regenerate `scan-allowlist.txt`, and record the candidates-before-word-list ordering in
  `.planning/STATE.md` (D-18)

### Phase 03.5: Registry-level raw→Zarr Conversion Entry Point (INSERTED)

**Goal**: A caller that is not one of quantlab's own scripts — the out-of-repo `quantlab-console`
first, but the contract is not written for it — can convert an acquired raw tier into the Zarr
tier through the registry, choosing the conversion mode explicitly and being told which RAM guard
that choice implies and what peak it predicts BEFORE anything is allocated.
**Mode:** mvp
**Requirements**: DATA-07, DATA-08 (to be added in discuss-phase)
**Depends on:** Phase 03.4
**Success Criteria** (what must be TRUE):

  1. A conversion can be started through the registry without naming a vendor class, a `Dataset`
     subclass, or an `ingest_*.py` script at the call site — the same standard `run()` already
     meets for acquisition (03.4 SC-1).
  2. `registry.run()` remains acquisition-only. Conversion is a SEPARATE call, never a flag on it
     (03.4 D-14): the three modes carry three differently-sized RAM guards, and folding them into
     one call is how one of them silently gets the wrong guard.
  3. The mode is an explicit argument with NO default, and an omitted mode raises rather than
     picking one — the same reasoning `resolve_symbols(mode=...)` already encodes, that a wrong
     choice here is silent and its symptom invisible.
  4. The caller can ask which guard a given (mode, roster, window) selects and what peak it
     predicts, and get an answer WITHOUT starting the conversion — the estimate is a value the
     caller can render, not a line this layer prints.
  5. A guard refusal names the remedy that would fit (a finer `--chunk`, a narrower window)
     rather than only refusing, preserving `assert_chunked_panel_fits`'s current behaviour.
  6. All three ingest shells reach conversion through this entry point rather than each calling
     `StockDataset` directly — they stay the registry's in-repo consumers and live proof the
     programmatic path works (03.4 D-15 role b). The `--chunk` / `--on-new-listing` divergence
     between `ingest_us_equity.py` and `ingest_tiingo.py` / `ingest_alpaca.py` disappears as a
     consequence of that delegation, not as new surface grown on the shells.
  7. Tick stays refused, not silently converted: the dense `[timestamp, symbol]` panel cannot
     express an irregular event axis, and that conversion belongs to Phase 03.3 (03.4 D-18).

**Why this phase exists now**: it is the unmet upstream precondition `quantlab-console`'s Phase 8
(`CVT-01`/`CVT-02`/`CVT-03`) names by hand. That roadmap put its conversion phase last ONLY
because this entry point did not exist, and states that moving it earlier is safe once this
lands — nothing else over there depends on it. Its `QC-02` check meanwhile reports raw/Zarr
divergence it cannot close, with "run a thin shell by hand" as the written interim workaround.
CVT-02's requirement that the guard and predicted peak be visible in the UI is the same demand as
Success Criterion 4, and is why that criterion is about ANSWERING rather than printing.

**Plans:** 2/6 plans executed

Plans:

**Wave 1**

- [x] 03.5-01-PLAN.md — TRACER: one Tiingo capability reaches Zarr through `registry.convert()`
  end-to-end (`DatasetConfig.market` reinstated, `Capability.dataset_cls`, `ConversionResult`,
  `convert()`), then Alpaca's four rows and the three capability-resolution refusals (D-01..D-04)
- [x] 03.5-02-PLAN.md — Split `estimate_chunked_panel` out of `assert_chunked_panel_fits` so the
  loop completes and every over-budget window carries its remedy; hoist `print_chunk_report` into
  `quantlab/utils/cli.py`; collapse `--to-zarr` to one flag with one help text (D-06/D-07/D-10/D-12/D-13)
- [ ] 03.5-03-PLAN.md — Write DATA-07/DATA-08 into REQUIREMENTS.md and rewrite ROADMAP 03.5's Goal
  and the Success Criteria D-06 superseded, including SC-6's fourth-shell scope honesty (D-06/D-09)

**Wave 2** *(blocked on Wave 1)*

- [ ] 03.5-04-PLAN.md — `_raw_data_to_xr_window` becomes abstract on `MarketDataset` and the
  warn-and-degrade path is deleted; `SpotKlineDataset` gains the explicit, honestly-documented
  implementation (D-08/D-09)
- [ ] 03.5-05-PLAN.md — All three US-equity shells delegate conversion to `convert()`, adopt the
  chunked guard and both chunk flags, and the two AST test suites are re-expressed so they keep
  their teeth (D-07/D-11, SC-6)

**Wave 3** *(blocked on Wave 2 — shares `quantlab/base/data.py`)*

- [ ] 03.5-06-PLAN.md — Thread `reporter`/`cancel` through the chunk loop with cancellation at
  window boundaries, forward them from `convert()`, and de-stale `run()`'s three-modes docstring
  (D-05/D-06)

### Phase 03.6: Frequency-Keyed Chunking Policy (INSERTED)

**Goal**: The time-window granularity of a chunked conversion is decided by ONE per-frequency
constant table rather than resolved at runtime or guessed per invocation, and the RAM guard's
REFUSING half is deleted while its ESTIMATING half survives as a value the caller renders.
**Requirements**: TBD (to be settled in discuss-phase)
**Depends on:** Phase 03.5
**Success Criteria** (what must be TRUE):

  1. A conversion's window granularity comes from a single source -- `CHUNK_GRANULARITY_BY_FREQUENCY`
     -- whose values are asserted AT IMPORT TIME to be members of `TimeChunkPlanner.GRANULARITIES`.
     Reproducing a conversion then needs only the config, never knowledge of how large the roster
     happened to be on the day it ran (the CLAUDE.md reproducibility constraint).
  2. `--chunk` drops from a required choice to an OVERRIDE: absent, it takes the table's value for
     the frequency; present, it wins.
  3. `day` joins `GRANULARITIES` and `_period_key` supports it. Full-market `1m` at day granularity
     is ~157 MB per window, which is what makes it the right default for that frequency.
  4. NO code path raises on a RAM budget any more: `assert_dense_panel_fits` and
     `assert_chunked_panel_fits` are deleted; `estimate_dense_panel` and `estimate_chunked_panel`
     survive; `MAX_DENSE_PANEL_BYTES` is demoted from a refusal line to an estimate's reference value.
  5. The three US-equity shells RENDER the estimate rather than being stopped by it -- the predicted
     peak is still visible before any memory is allocated, which is what 03.5 SC-4 actually asked for.
  6. Every locked text this phase falsifies is rewritten rather than left standing: ROADMAP 03.5's
     SC-5 (a refusal names the fitting remedy) is removed, D-05 (`assert_dense_panel_fits` and
     `MAX_DENSE_PANEL_BYTES` are kept) is formally amended, and the refusal-bearing halves of 03.5's
     D-10/D-11 are marked superseded.
  7. `tick` is deliberately ABSENT from the constant table, and its absence IS the "not wired up yet"
     answer -- no `if frequency == "tick"` branch is introduced. Same shape as
     `BARS_PER_DAY_BY_FREQUENCY`'s deliberate tick omission and 03.5 D-01's `dataset_cls=None`.

**Notes carried into discuss-phase (do NOT re-derive):**

- The decision to DELETE the guard was the developer's, made 2026-09-11. Its known cost is accepted:
  the full-market tick case (~924 GB for one dense hour at 7,700 symbols) is no longer refused and
  will reach OOM instead. The repo's recorded precedent for that failure mode is quick task
  260906-13w (~7.2 GiB grid + ~29.6M-row frame OOM'd a 16 GiB box, on full-market DAILY, not tick).
- Whether `GRANULARITIES` drops a further level to `hour` is NOT decided here. Its precondition is
  that the raw tier partitions by hour too -- `RAW_HIVE_KEYS` is `1d=month`, `1m=date`, `tick=date`,
  so an hourly window would be finer than the shard it reads and would re-open the same files
  6.5-16x. Recommendation: stop at `day`.
- `plan_calendar` hardcodes `pd.date_range(freq="D")` and returns `YYYY-MM-DD` string pairs. Both are
  day-resolution assumptions sitting OUTSIDE the `_period_key` single-definition guarantee, so a
  future `hour` would silently drift `plan_calendar` from `plan_from_timestamps` -- the exact drift
  the class docstring calls "structurally impossible". Deriving `freq` from the granularity is a
  cheap compatibility hook worth taking now.
- `_period_key` returns `tuple[int, int]`. `day` fits as `(year, dayofyear)`; `hour` would not.

**Plans:** 0 plans

Plans:

- [ ] TBD (run /gsd-plan-phase 03.6 to break down)

### Phase 4: Baseline Return Prediction Model

**Goal**: Users can train a simple baseline model that consumes factor data directly as xarray and produces return/rank predictions.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: MODEL-01, MODEL-02
**Success Criteria** (what must be TRUE):

  1. User can train a baseline linear-regression-style model on stored factor + label xarray data
  2. Model training and inference read directly from `xarray.Dataset` with no DataFrame conversion step in between
  3. User can retrieve, per symbol/timestamp, a future-return or return-rank prediction from the trained model

**Plans**: TBD

### Phase 5: Portfolio Optimization & Target Holdings

**Goal**: Users can turn model predictions into a long-short, unlevered target holdings vector per symbol, persisted to disk.
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: PORT-01, PORT-02
**Success Criteria** (what must be TRUE):

  1. User can run a baseline portfolio optimizer (e.g. mean-variance or equal-weight long-short) that consumes return/rank predictions and outputs a target % holding per symbol
  2. For any generated allocation, net exposure and gross exposure are both ≤100%, verifiable by summing the output
  3. Target holdings for a given run are persisted to disk and reloadable

**Plans**: TBD

### Phase 6: End-to-End Backtest & Reproducible Pipeline

**Goal**: The full data→factor→model→portfolio→backtest pipeline runs end-to-end from a single config file with reproducible results, verified via vectorbt, with module contracts and dual-use architecture (time-series + cross-sectional) demonstrated, and the event-driven backtest path kept runnable.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: BT-01, BT-02, CFG-01, ARCH-01, ARCH-02
**Success Criteria** (what must be TRUE):

  1. User can feed target holdings into a vectorbt-based backtest and get back an equity curve plus key performance metrics
  2. User can run the entire data→factor→model→portfolio→backtest pipeline end-to-end driven by a single config file, and re-running with the same config reproduces the same result
  3. User can run `backtest/test_strategy.py`'s NautilusTrader strategy without runtime errors (functional, not production-hardened)
  4. At least one pipeline stage (e.g. the model) can be swapped for an alternate implementation without modifying the code of any other stage, demonstrating the input/output contract holds
  5. The same architecture is demonstrated to support both a single-symbol time-series use case and a multi-symbol cross-sectional multi-factor use case

**Plans**: TBD

### Phase 7: Testing & Code Quality

**Goal**: Core pipeline modules have unit tests in the project's style, and the codebase reads as clean, well-documented Python consistent with the Zen of Python.
**Mode:** mvp
**Depends on**: Phase 6
**Requirements**: QUAL-01, QUAL-02
**Success Criteria** (what must be TRUE):

  1. Each core module (data, factor, model, portfolio, backtest) has at least one passing unit test runnable via a single test command (e.g. `uv run pytest`)
  2. The test suite passes cleanly with no import/config errors
  3. A code review pass confirms public functions/classes across core modules have docstrings and parameter/return type annotations
  4. No leftover test/temp/redundant code remains outside the established `tests/` structure

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Codebase Cleanup & Security Hardening | 5/5 | Complete   | 2026-09-05 |
| 2. Multi-Market Data Foundation | 8/8 | Complete   | 2026-09-05 |
| 3. Factor Computation (KunQuant + Polars) | 5/5 | In Progress|  |
| 03.1 Index Historical Constituents Data Layer | 5/5 | Complete    | 2026-09-07 |
| 03.2 Multi-Source Data Acquisition Abstraction (Alpaca) | 7/7 | Complete   | 2026-09-06 |
| 03.4 Data Source Registry (Operator-Surface Foundation) | 11/11 | In Progress|  |
| 4. Baseline Return Prediction Model | 0/TBD | Not started | - |
| 5. Portfolio Optimization & Target Holdings | 0/TBD | Not started | - |
| 6. End-to-End Backtest & Reproducible Pipeline | 0/TBD | Not started | - |
| 7. Testing & Code Quality | 0/TBD | Not started | - |
</content>
