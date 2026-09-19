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
- [ ] **Phase 03.5: Registry-level raw→Zarr Conversion Entry Point** - A conversion entry point beside `registry.run()`, chunked-only, with its RAM guard answerable before the run
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

### Phase 03.9: WRDS TAQ Consolidated Quotes to NBBO Zarr Panel (INSERTED)

**Goal:** NYSE TAQ millisecond consolidated quotes (WRDS `taqm_*` `cqm_*` tables,
https://wrds-www.wharton.upenn.edu/pages/get-data/nyse-trade-and-quote/millisecond-trade-and-quote-daily-product-2003-present-updated-daily/consolidated-quotes/)
reach the canonical `[timestamp, symbol]` Zarr panel as fixed-frequency NBBO snapshots, through
the existing Acquisition → Dataset layering.
**Requirements**: TBD
**Depends on:** Phase 03.2 (Acquisition abstraction), Phase 03.1 (point-in-time universes)
**Plans:** 0 plans

**User decisions already made (2026-09-19, do NOT re-ask in discuss-phase):**

- Storage grain (refined in discuss-phase, see `03.9-CONTEXT.md`): WRDS **NBBO records are pulled
  raw** into parquet (no server-side aggregation), then **resampled locally** at a configurable
  frequency into the dense `[timestamp, symbol]` panel. Raw events on a non-dense Zarr axis stay
  Phase 03.3's scope.
- Universe: **reuse the existing point-in-time universes** (`sp500_constituent` /
  `nasdaq100_constituent`); date range via CLI arguments.
- Credentials: WRDS account with TAQ access; username from the `WRDS_USERNAME` environment variable,
  password in `~/.pgpass`. Never a config field, never in logs (same rules as `CREDENTIAL_ENV_VARS`).
- Layering: a new vendor provider under `quantlab/acquisition/` writes raw parquet shards only (never
  Zarr); a Dataset class converts them to Zarr — matching `example/acquisition.md`'s hard boundary.

