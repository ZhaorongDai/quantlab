---
phase: 03-factor-computation-kunquant-polars
plan: 01
subsystem: testing
tags: [kunquant, xarray, zarr, pytest, polars, alpha158, alpha101, factors]

# Dependency graph
requires:
  - phase: 02-data-acquisition-and-datasets
    provides: "SpotKlineDataset/StockDataset + XrBackend Zarr persistence, the tests/ suite and tests/conftest.py this plan extends"
provides:
  - "A working batch KunQuant factor path: Alpha158SpotKline.cal() and Alpha101SpotKline.cal() return real xr.Dataset factor values (was TypeError on every call)"
  - "my_ops/preprocess.py composite ops conforming to installed KunQuant 0.1.11's CompositiveOp.decompose(self, options: dict) contract"
  - "spot_kline_zarr / stock_zarr synthetic-Zarr factory fixtures in tests/conftest.py (seeded, strictly-positive, network-free)"
  - "stock_pqt_row / write_stock_pqt fixtures promoted out of tests/test_stock_dataset.py"
  - "Four Phase-3 test files (test_factor_kunquant, test_factor_stream, test_factor_polars, test_factor_hierarchy) existing, collecting cleanly and green"
affects: [03-02, 03-03, 03-04, 03-05, factor-hierarchy-refactor, polars-factor-backend, streaming-factors]

actuals:
  tokens: 6679
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Synthetic-Zarr factory fixtures: write the store to disk, THEN return the DatasetConfig (Dataset.config's setter reads the store when symbols is not None)"
    - "Factor tests pass an explicit 1-3 element factor_names list and njobs=4 to keep each compiled KunQuant graph tiny and the executor to 4 threads"
    - "Wave-0 scaffold test files carry one real infrastructure self-test rather than zero tests, so per-file pytest commands cannot silently exit 5"

key-files:
  created:
    - tests/test_factor_kunquant.py
    - tests/test_factor_stream.py
    - tests/test_factor_polars.py
    - tests/test_factor_hierarchy.py
  modified:
    - my_ops/preprocess.py
    - tests/conftest.py
    - tests/test_stock_dataset.py

key-decisions:
  - "my_ops composite ops accept and ignore `options` rather than reading it — a strict superset of prior behaviour, matching the KunQuant-shipped ops in KunQuant/ops/CompOp.py"
  - "Phase-3 test fixtures write Zarr stores directly, bypassing raw CSV/parquet ingestion, so factor tests test factors and not CSV parsing"
  - "Each Wave-0 scaffold test file ships one real infrastructure self-test, never a placeholder, so its per-file pytest command cannot exit 5 and read as green"
  - "A shared module-private `_positive_random_walk()` generator backs both Zarr fixtures, guaranteeing every price/volume entry is > 1.0 because KunQuant divides by these columns"

patterns-established:
  - "Write-before-config ordering: any fixture returning a DatasetConfig must persist its store first (base/data.py:Dataset.config setter calls _reset_symbols() -> read())"
  - "Boundary-contract naming: the KunQuant path sees lowercase open/close/volume (renamed by _to_kunquant()); the Polars path sees raw Binance Title-Case Close — locked by test_factor_polars.py"
  - "Grep-style core-layer purity checks (tests/test_extensibility_contract.py idiom) reused as permanent regression locks on architectural claims"

requirements-completed: [FACTOR-01]

coverage:
  - id: D1
    description: "Alpha158SpotKline.cal().get_features() returns an xr.Dataset with [timestamp, symbol] dims and the requested factor variables, carrying finite values — the batch KunQuant path no longer raises TypeError"
    requirement: FACTOR-01
    verification:
      - kind: integration
        ref: "tests/test_factor_kunquant.py#test_alpha158_spot_batch_cal_returns_xarray_dataset"
        status: pass
    human_judgment: false
  - id: D2
    description: "Alpha101SpotKline.cal().get_features() likewise returns an xr.Dataset — the shared WindowedZScore normalization op works for both alpha families"
    requirement: FACTOR-01
    verification:
      - kind: integration
        ref: "tests/test_factor_kunquant.py#test_alpha101_spot_batch_cal_returns_xarray_dataset"
        status: pass
    human_judgment: false
  - id: D3
    description: "my_ops/preprocess.py's two composite ops match installed KunQuant 0.1.11's decompose(self, options: dict) contract, with every Chinese comment and docstring preserved byte-identical"
    requirement: FACTOR-01
    verification:
      - kind: other
        ref: "grep -c 'def decompose(self, options: dict)' my_ops/preprocess.py == 2 && grep -c 'def decompose(self) ' == 0; git diff --stat shows 4 insertions / 2 deletions"
        status: pass
    human_judgment: false
  - id: D4
    description: "Synthetic Binance-shaped (spot_kline_zarr) and Tiingo-shaped (stock_zarr) Zarr fixtures are buildable from tests/conftest.py with zero network access and zero raw CSV/parquet ingestion"
    verification:
      - kind: integration
        ref: "tests/test_factor_kunquant.py + tests/test_factor_stream.py + tests/test_factor_polars.py consume spot_kline_zarr; stock_zarr smoke-tested through StockDataset.read()/to_kunquant() during execution"
        status: pass
    human_judgment: false
  - id: D5
    description: "tests/test_stock_dataset.py no longer defines private _COLUMNS/_row/_write_stock_pqt helpers; it consumes the shared conftest fixtures and its three existing tests pass unchanged"
    verification:
      - kind: unit
        ref: "uv run pytest tests/test_stock_dataset.py -q -> 3 passed; grep -c '_write_stock_pqt' tests/test_stock_dataset.py == 0"
        status: pass
    human_judgment: false
  - id: D6
    description: "All four Phase-3 test files exist, collect cleanly under pytest --collect-only, and each reports exactly 1+ passed when run alone"
    verification:
      - kind: unit
        ref: "uv run pytest tests/ --collect-only -q (exit 0); per-file runs -> 1 passed each; uv run pytest tests/ -q -> 44 passed"
        status: pass
    human_judgment: false

duration: 31 min
completed: 2026-09-05
status: complete
---

# Phase 3 Plan 01: Unblock Batch Factor Computation + Nyquist Wave 0 Summary

**A one-line-per-class `decompose(self, options: dict)` signature fix that resurrects every batch KunQuant `.cal()` in the repository, proven end-to-end by new Alpha158/Alpha101 tests over seeded synthetic Zarr fixtures, plus the four Phase-3 test files and shared fixtures the rest of the phase writes into.**

## Performance

- **Duration:** 31 min
- **Started:** 2026-09-05T13:42Z (approx.)
- **Completed:** 2026-09-05T14:13:06-04:00
- **Tasks:** 2
- **Files modified:** 7 (3 modified, 4 created)

## Accomplishments

- **Fixed the blocking defect that made ROADMAP Phase 3 Success Criterion 1 false.** `my_ops/preprocess.py` declared `decompose(self)` on both `WindowedZScore` and `WindowedRobustStandardization`, while installed KunQuant 0.1.11 declares the contract as `CompositiveOp.decompose(self, options: dict)` (`KunQuant/Op.py:292`) and invokes it positionally (`KunQuant/passes/Decompose.py:15`). Because **both** `Alpha101SpotKline` and `Alpha158SpotKline` wrap every `Output(...)` in `WindowedZScore(..., self.config.window)`, every batch `.cal()` in the repository died with `TypeError: WindowedZScore.decompose() takes 1 positional argument but 2 were given`. It now returns real factor values.
- **Proved the fix end-to-end, not just structurally.** `tests/test_factor_kunquant.py` runs the full tracer path — synthetic Zarr store → `SpotKlineDataset._to_kunquant()` → `FactorKunQuant._make()`/`kr.runGraph` → `xr.Dataset` — for both alpha families, asserting `sizes == {"timestamp": 60, "symbol": 8}`, the exact `data_vars`, and finite values.
- **Closed every Wave-0 gap in 03-VALIDATION.md.** `spot_kline_zarr` and `stock_zarr` synthetic-Zarr factories and the promoted `stock_pqt_row` / `write_stock_pqt` parquet helpers now live in `tests/conftest.py`; the `_write_stock_pqt` duplication is gone; and all four Phase-3 test files exist, collect cleanly and run green.
- **Made each scaffold immediately load-bearing.** The three files 03-02/03-04/03-05 will fill in each assert the specific fixture contract that plan depends on — 8 symbols for the streaming replay, raw Title-Case `Close` for the Polars backend, and a permanent grep lock proving `base/model.py` never dispatches on a concrete factor backend (D-03 interchangeability).

## Task Commits

Each task was committed atomically:

1. **Task 1 (tracer): Fix my_ops decompose() signature drift and prove Alpha158/Alpha101 batch cal() end-to-end** — `23c3ad1` (fix)
2. **Task 2: Complete the Nyquist Wave-0 scaffolding — stock fixture, promoted pqt helpers, three remaining test files** — `53ab3d9` (test)

**Plan metadata:** see the `docs(03-01)` commit that carries this SUMMARY.

## Files Created/Modified

- `my_ops/preprocess.py` — **modified.** Both composite ops now declare `decompose(self, options: dict)`, each with a one-line comment pointing at the `KunQuant/passes/Decompose.py:15` call site. Bodies untouched; every Chinese comment and docstring byte-identical (diff is exactly 4 insertions / 2 deletions).
- `tests/conftest.py` — **modified.** Added `_positive_random_walk()` (shared, strictly-`> 1.0` seeded generator), the `spot_kline_zarr` and `stock_zarr` Zarr factory fixtures, and the promoted `stock_pqt_row` / `write_stock_pqt` parquet fixtures with their `_STOCK_PQT_COLUMNS` constant.
- `tests/test_stock_dataset.py` — **modified.** Private `_COLUMNS` / `_row()` / `_write_stock_pqt()` deleted; the three existing tests now take `stock_pqt_row` / `write_stock_pqt` as fixture parameters and assert exactly what they asserted before.
- `tests/test_factor_kunquant.py` — **created.** The FACTOR-01 regression lock: two end-to-end batch `.cal()` tests plus a `_factor_config()` helper that builds `FactorConfig` with 100% keyword arguments and `njobs=4`.
- `tests/test_factor_stream.py` — **created.** FACTOR-02 scaffold (filled by 03-05); asserts the fixture panel is 8 symbols wide and ≥ 40 timestamps deep.
- `tests/test_factor_polars.py` — **created.** FACTOR-03 scaffold (filled by 03-04); locks the `get_lazyframe()` raw Title-Case column contract `factor/momentum.py` will be written against.
- `tests/test_factor_hierarchy.py` — **created.** FACTOR-04 + D-03 scaffold (filled by 03-02/03-05); grep-style purity check proving `base/model.py` names no concrete factor backend and never `isinstance`-dispatches on a factor type.

## Decisions Made

1. **`options` is accepted and ignored, not read.** Neither op needs decomposition options, and the KunQuant-shipped ops in `KunQuant/ops/CompOp.py` that ignore `options` use exactly this shape. Accepting-and-ignoring is a strict superset of the previous behaviour, which is the right risk profile for a bug-fix commit — reading `options` would have introduced unverifiable new behaviour alongside the fix.
2. **Fixtures write Zarr directly rather than driving raw ingestion.** Factor windows need ~60 timestamps × 8 symbols; routing that through `SpotKlineDataset._raw_data_to_xr()` would make every factor test also a CSV-parsing test. The direct-to-Zarr fixtures keep each factor test at ~0.5 s and keep failures attributable to the factor layer.
3. **Scaffolds ship a real self-test, never zero tests.** A test file with no tests makes its per-file pytest command exit 5 ("no tests ran"), which reads as green in a summary line. Each scaffold instead asserts the exact contract its future plan depends on.
4. **One shared `_positive_random_walk()` backs both fixtures.** KunQuant divides by price and volume columns, so a zero or negative entry would poison every downstream factor with inf/NaN. Starting at 100.0, cumulatively summing normal increments and taking `abs(...) + 1.0` guarantees positivity without destroying the walk's time-series structure. `high`/`low`/`open` are derived as fixed positive multiples of the same base, so OHLC ordering holds by construction.

## Deviations from Plan

None — plan executed exactly as written.

Two implementation notes worth recording, neither of which changes plan scope:

- **`tests/conftest.py` gained `polars` and `datetime` module-level imports** beyond the `numpy`/`pandas`/`xarray`/`base.config` set Task 1 Step B specified. This is a direct consequence of Task 2 Step A, which asks that `_row()` and `_write_stock_pqt()` be moved with the *same body* — those bodies call `datetime.fromisoformat` and `pl.DataFrame`. The constraint that actually matters (the standing collect-time import-safety rule) is fully honoured: `tests/conftest.py` still imports no `dataset.*` or `factor.*` module, and `uv run pytest tests/ --collect-only -q` exits 0.
- **`stock_zarr` was smoke-tested before being left unused.** No test in this plan consumes it (03-03 is its first consumer), so it was verified during execution via a throwaway test that ran it through `StockDataset(...).read().get_xarray_dataset()` and `to_kunquant(("open", "close", "volume"))` — confirming the store carries both the raw lowercase and adjusted groups (`_to_kunquant()` drops the former and renames the latter, so a missing group raises), that the result is `{'timestamp': 60, 'symbol': 2}`, and that every value is strictly positive. The throwaway file was deleted and never committed. An unverified unused fixture would have been a landmine for 03-03.

**Total deviations:** 0 auto-fixed.
**Impact on plan:** None. Both tasks landed as specified, and the tracer's automated `<verify>` passed on the first run after the fix.

## Verification Results

Plan-level `<verification>`, re-run after both commits:

| Command | Expected | Actual |
|---|---|---|
| `uv run pytest tests/ -q` | zero failures | **44 passed** |
| `uv run pytest tests/test_factor_kunquant.py -x -q` | 2 passed | **2 passed** |
| `uv run pytest tests/ --collect-only -q` | exit 0, no collection errors | **exit 0** |
| `uv run pytest tests/test_stock_dataset.py -q` | 3 passed | **3 passed** |
| `uv run pytest tests/test_factor_{stream,polars,hierarchy}.py -q` | 1 passed each | **1 passed each** |
| `grep -c "def decompose(self, options: dict)" my_ops/preprocess.py` | 2 | **2** |
| `grep -c "def decompose(self) " my_ops/preprocess.py` | 0 | **0** |
| `grep -c "_write_stock_pqt" tests/test_stock_dataset.py` | 0 | **0** |
| `grep -c "def write_stock_pqt" tests/conftest.py` | 1 | **1** |
| `git diff --stat my_ops/preprocess.py` | 2 signature + 2 comment lines | **4 insertions, 2 deletions** |

The suite grew from 41 to 44 tests (2 tracer + 3 scaffold self-tests, minus none removed — the three stock tests were rewired, not replaced).

## Tracer Feedback Gate

Task 1 was `type="tracer"` with no `gate` attribute (default `blocking`) and an automated-only `<verify>`. With auto mode inactive (`workflow._auto_chain_active: false`, `workflow.auto_advance: false`) and `workflow.human_verify_mode: end-of-phase`, the gate resolves to *re-run the tracer verify and continue without a checkpoint*. `uv run pytest tests/test_factor_kunquant.py -x -q` was re-run after the Task 1 commit and reported 2 passed, so expansion into Task 2 proceeded. No human checkpoint was owed or skipped.

## Known Stubs

Three files are **intentionally partial**, exactly as the plan specifies. None blocks this plan's goal; each carries a passing test today and names the plan that completes it.

| File | Status | Resolved by |
|---|---|---|
| `tests/test_factor_stream.py` | Scaffold — 1 fixture self-test; the FACTOR-02 `cal_stream()` replay smoke test and the aarch64 SIMD block-width fix are not here | **03-05** |
| `tests/test_factor_polars.py` | Scaffold — 1 boundary-contract test; the `FactorPolars` ABC, `Momentum` factor and D-04 laziness proof are not here | **03-04** |
| `tests/test_factor_hierarchy.py` | Scaffold — 1 permanent purity lock; the `Factor` ABC extraction assertions and the interchangeability integration test are not here | **03-02**, **03-05** |

`stock_zarr` in `tests/conftest.py` is likewise unused by any committed test until 03-03; it was smoke-tested during execution (see Deviations) rather than left unverified.

## Issues Encountered

None. The plan's `<interfaces>` block had already executed and verified every fact it asserted (the KunQuant contract, the valid factor names, the minimal `data_columns` per subset, the write-before-config ordering hazard), so no interface needed re-derivation and nothing behaved unexpectedly.

## User Setup Required

None — no external service configuration required. This plan installed zero new packages; `pytest`, `polars`, `xarray` and `KunQuant` were already declared in `pyproject.toml` and installed.

## Next Phase Readiness

**Ready for 03-02.** Wave 1 is complete and the phase's blocking defect is gone, so every later plan now has a green baseline to regression-test against:

- **03-02** (`Factor` ABC extraction) can refactor `base/factor.py` against `tests/test_factor_kunquant.py` as the behaviour-preservation lock and `tests/test_factor_hierarchy.py` as the D-03 purity lock.
- **03-03** (`Alpha158Stock`, D-02 `amount` proxy, D-09 normalization matrix) has `stock_zarr` and the promoted parquet helpers waiting.
- **03-04** (`FactorPolars`, `factor/momentum.py`) has the raw Title-Case column contract locked in `tests/test_factor_polars.py`.
- **03-05** (streaming) has the 8-symbol panel it needs locked in `tests/test_factor_stream.py`.

**Requirement status:** FACTOR-01 is satisfied *for the crypto-spot market* by this plan, but was **not** marked Complete in REQUIREMENTS.md — `requirements ready-ids` reports `0/1 ready` because sibling plans in this phase also declare FACTOR-01 (03-03 extends it to US equities). It will flip Complete when the last declaring plan produces its SUMMARY. This is the shared-ID gate working as intended, not a gap.

**Concerns:** none blocking. Two known Phase-3 defects remain open and are already assigned: `FactorKunQuant._make_stream()`'s hardcoded x86 SIMD block width (BUG-02, breaks streaming compilation on aarch64 — 03-05 Task 1) and `Alpha101Stock`'s missing `amount` input (03-03 Task 1). Neither is touched by this plan and neither affects the batch path proven here.

---
*Phase: 03-factor-computation-kunquant-polars*
*Completed: 2026-09-05*

## Self-Check: PASSED

All 7 key files verified present on disk; both task commits (`23c3ad1`, `53ab3d9`) verified present in `git log --all`; full suite re-run green (44 passed).
