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
- [ ] **Phase 03.1: Index Historical Constituents Data Layer** - Point-in-time index membership panels (verification: gaps_found)
- [x] **Phase 03.2: Multi-Source Data Acquisition Abstraction (Alpaca)** - Second vendor through the same batched, resumable, volume-guarded `Acquisition` abstraction (completed 2026-09-06)
- [ ] **Phase 03.3: Tick Data Storage (Non-Dense Event Axis)** - Raw tick shards reach a persisted store via a tick-specific Dataset with a non-dense event axis
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

**Plans:** 5 plans (4/4 executed, 1 gap-closure pending)

Plans:

- [x] 03.1-01-PLAN.md — Split `base/data.py` into `BaseDataset`/`MarketDataset` and `DatasetConfig` into a three-way config split (wave 1)
- [x] 03.1-02-PLAN.md — Extract `IndexMembershipFetcher`, add `Nasdaq100MembershipFetcher` and the third universe category (wave 1)
- [x] 03.1-03-PLAN.md — `IndexConstituentDataset` + `SP500ConstituentDataset`: interval-to-daily-panel densification with all six correctness locks (wave 2)
- [x] 03.1-04-PLAN.md — `Nasdaq100ConstituentDataset`, registry-driven `UniverseCatalog` coverage guards, README (wave 3)
- [ ] 03.1-05-PLAN.md — Gap closure (VERIFICATION truth 3 / DATA-05): extend the registry-driven coverage guard to `get_symbols_in_range()` so a pre-coverage window raises instead of silently returning a censored roster (wave 4)

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
| 03.1 Index Historical Constituents Data Layer | 4/4 | Gaps Found |  |
| 03.2 Multi-Source Data Acquisition Abstraction (Alpaca) | 7/7 | Complete   | 2026-09-06 |
| 4. Baseline Return Prediction Model | 0/TBD | Not started | - |
| 5. Portfolio Optimization & Target Holdings | 0/TBD | Not started | - |
| 6. End-to-End Backtest & Reproducible Pipeline | 0/TBD | Not started | - |
| 7. Testing & Code Quality | 0/TBD | Not started | - |
</content>
