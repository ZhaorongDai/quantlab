---
phase: quick-260906-w3t
plan: 01
subsystem: data
tags: [xarray, zarr, polars, factor, databackend, rv-01, regression-test]

requires:
  - phase: 03-factor-computation-kunquant-polars
    provides: FactorPolars, DataBackend.head, Momentum, the 260906-usg bounded-read contract
provides:
  - "DataBackend.head(path, n) — path-threaded bounded read; both backends open the store themselves and never touch self.data"
  - "A FactorPolars name probe that performs no filtering, so a factor's lookback window survives construction"
  - "The first test in this repo whose DatasetConfig carries dates — the regression guard RV-01 was hiding behind"
affects: [04-return-model, factor-computation, dataset-backend]

actuals:
  tokens: 21000
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "A bounded probe read takes the store PATH and opens the store itself, rather than depending on a caller's prior read() having populated shared state"

key-files:
  created: []
  modified:
    - base/backend.py
    - dataset/backend.py
    - base/data.py
    - base/factor_polars.py
    - tests/conftest.py
    - tests/test_backend_head.py
    - tests/test_factor_polars.py

key-decisions:
  - "D-1 shipped as planned: head(path, n) threads the store path, path first, mirroring read(path, **kwargs). read() itself is untouched — same signature, same meaning, no filter= parameter. This removed a CALLER, not a method."
  - "D-2 shipped: XrBackend.head opens with xr.open_dataset, the same opener read() uses. open_zarr stays confined to _assert_append_compatible's append-specific concern."
  - "D-3 shipped: RV-02 is NOT closed. head() carries read()'s explicit Path(path).exists() guard and its exact FileNotFoundError message, and that behaviour is now locked by test_head_raises_immediately_on_an_absent_store rather than left to prose."
  - "Task 3's expected timestamp count is asserted as the literal 49, never derived from config.dataset.config.start_date — that value is written by the code under test, so a derived expectation would pass under the bug."

patterns-established:
  - "Probe reads open their own store: a construction-time probe must not reach data through a method that filters shared state in place."
  - "A fixture default that makes a filter a no-op hides every filtering defect behind it — dated fixtures are the guard."

requirements-completed: [FACTOR-03, FACTOR-04]

coverage:
  - id: D1
    description: "DataBackend.head(path, n) opens the store itself — a freshly-constructed backend that has never had read() or to_internal() called answers head(path, 2) with real rows"
    requirement: FACTOR-03
    verification:
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_opens_the_store_without_a_prior_read"
        status: pass
    human_judgment: false
  - id: D2
    description: "head(path, n) preserves the get_lazyframe() schema, returns at most n rows, and does not mutate self.data — the 260906-usg contract, re-pointed at the new signature"
    requirement: FACTOR-03
    verification:
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_preserves_the_get_lazyframe_schema"
        status: pass
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_returns_at_most_n_rows"
        status: pass
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_does_not_mutate_backend_state"
        status: pass
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_is_an_interface_obligation_not_a_convenience"
        status: pass
    human_judgment: false
  - id: D3
    description: "RV-02's loud, correctly-named error is preserved on purpose — head(missing_path, n) raises FileNotFoundError immediately on both backends rather than at collect() time"
    verification:
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_raises_immediately_on_an_absent_store"
        status: pass
    human_judgment: false
  - id: D4
    description: "The FactorPolars construction-time probe performs no filtering — probe is self.config.dataset.head(_SCHEMA_PROBE_ROWS) with no .read() in its call chain"
    requirement: FACTOR-03
    verification:
      - kind: unit
        ref: "tests/test_factor_polars.py#test_an_explicit_factor_names_pin_is_not_overwritten_at_construction"
        status: pass
      - kind: unit
        ref: "tests/test_factor_polars.py#test_factor_names_resolve_dynamically_from_the_lazyframe_schema"
        status: pass
    human_judgment: false
  - id: D5
    description: "RV-01 closed: a factor whose dataset config carries dates computes over the widened lookback window (49 bars, not 29) and its factor column carries zero NaN over the requested window"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_factor_polars.py#test_a_dated_dataset_config_keeps_the_factor_lookback_window"
        status: pass
      - kind: other
        ref: "mutation: .read() restored on the probe -> AssertionError: assert 29 == 49"
        status: pass
      - kind: other
        ref: "mutation: full pre-fix world (probe read() + XrBackend.head on self.data) -> AssertionError: assert 29 == 49; measured 40/232 NaN = 17.2%"
        status: pass
    human_judgment: false

duration: 18 min
completed: 2026-09-06
status: complete
---

# Quick Task 260906-w3t: Fix RV-01 — make DataBackend.head read directly Summary

**`DataBackend.head(path, n)` now takes the store path and opens the store itself, so the `FactorPolars` name probe no longer routes through `BaseDataset.read()` → `_filter()` → in-place `filter_by_date` — closing RV-01, where a factor silently computed over 29 bars instead of 49 and returned a 17.2%-NaN column with nothing raised.**

## Performance

- **Duration:** 18 min
- **Tasks:** 3
- **Files modified:** 7
- **Full suite:** 387 passed (baseline 384; +3 new tests), pytest exit code 0

## Accomplishments

- **The probe stops filtering.** `head` is declared on the `DataBackend` ABC as `head(path, n)`, path first, mirroring `read(path, **kwargs)`. Both concrete backends open their own store; neither reads or assigns `self.data`. `BaseDataset.head(n)` keeps its signature and threads `self.config.zarr_file_path` through, exactly as `read()` does.
- **Both halves of the bug are bypassed.** `BaseDataset.read()`'s in-place narrowing never runs for the probe, so there is nothing left for `XrBackend.read()`'s cache early-return to make permanent. `cal()`'s `read()` is now the first read of the store, and `_filter()` runs once against the already-widened dates.
- **The regression is guarded by a test that goes red under mutation.** `test_a_dated_dataset_config_keeps_the_factor_lookback_window` is the first test in this repo whose `DatasetConfig` carries dates — the exact shape 384 green tests could not see RV-01 through.
- **The now-false docstring is gone.** `_get_factor_names()`'s "The `.read()` call stays" bullet would have told a future reader to reintroduce the defect; it is replaced by the constraint that actually holds.

## Task Commits

1. **Task 1 (RED): failing tests for path-threaded head()** — `6fd3622` (test)
2. **Task 1 (GREEN): thread the store path into head() on both backends** — `8961775` (feat)
3. **Task 2: drop read() from the FactorPolars name probe** — `a460247` (fix)
4. **Task 3: the dated-fixture RV-01 regression test** — `5668e93` (test)

## Files Created/Modified

- `base/backend.py` — `DataBackend.head(path, n)` as `@abstractmethod`; docstring gains the third obligation (raise `FileNotFoundError` at the call, not at `.collect()`) and records why `self.data` is off limits
- `dataset/backend.py` — `XrBackend.head` opens with `xr.open_dataset(path)` inside try/finally-close, bounding every dim from the *opened* dataset; `PlBackend.head` scans straight from `path` so the limit pushes into the reader; both carry `read()`'s `Path(path).exists()` guard
- `base/data.py` — `BaseDataset.head(n)` threads `zarr_file_path` through; stays concrete; docstring records that the store LOCATION is now a storage-medium property too
- `base/factor_polars.py` — probe is `self.config.dataset.head(_SCHEMA_PROBE_ROWS)`; three docstring blocks corrected
- `tests/conftest.py` — `spot_kline_zarr` accepts optional `start_date`/`end_date` (default `None`, so every existing caller is byte-identical)
- `tests/test_backend_head.py` — fixture writes a zarr store and yields `(backend, path)`; two new tests
- `tests/test_factor_polars.py` — `_momentum_config` gains optional `window`/`start_date`/`end_date`; the dated regression test

## Mutation Results

Every mutation the plan specified was run, and each outcome is recorded — including the one that was *supposed* to stay green.

| # | Mutation | Expected | Observed |
|---|----------|----------|----------|
| T1-a | `XrBackend.head` reverted to `self.data` body **and** signature `head(self, n)` | red | **RED** — `TypeError: XrBackend.head() takes 2 positional arguments but 3 were given` |
| T1-b | `XrBackend.head` keeps `path` param but body reads `self.data` (the "simplify it back" regression) | red | **RED** — `AttributeError: Please cal 'read' or 'to_internal' first.`, the plan's predicted signal exactly |
| T2 | `.read()` restored in front of `.head(...)` on the probe, Task 3's test not yet written | **green** | **GREEN** — 30 passed |
| T3-1 | `.read()` restored on the probe, with Task 3's test present | red | **RED** — `AssertionError: assert 29 == 49` |
| T3-2 | Full pre-fix world: T3-1 **plus** `XrBackend.head` reading `self.data` | red | **RED** — `AssertionError: assert 29 == 49` |

**On T1-a vs T1-b.** The plan predicted T1-a would fail with `AttributeError: Please cal 'read' or 'to_internal' first.` It failed with a `TypeError` instead: reverting the *signature* makes the arity error fire before the body is ever reached, shadowing the predicted signal. T1-b was run to isolate the mechanism — same `self.data` body, path parameter retained — and produced the predicted `AttributeError` verbatim. T1-b is also the more realistic regression: a future reader "simplifying" `head` back to `self.data` would keep the signature its callers depend on.

**On T2 — the deliberately-green mutation.** Restoring `.read()` on the probe left the whole suite green (30 passed). That is the finding, not a failure to reproduce: at that point every fixture in the repo left `DatasetConfig.start_date`/`end_date` unset, so `_filter()` was a no-op and the probe's narrowing was unobservable. This is precisely why Task 3 exists, and why the fix without the dated fixture would have been unprotected.

**Measured NaN half of T3-2.** The dated test's first assertion (29 != 49) trips before the NaN assertion is reached, so the NaN half was measured directly under the mutation: **29 timestamps, 40 NaN of 232 values = 17.2%** — matching the RV-01 measurement filed in `03-VERIFICATION.md` to the decimal.

## Decisions Made

- **D-1 shipped as planned.** `head(path, n)`, path first. `read()` is untouched — `git diff` over this task shows no edit to any `def read` line in `base/data.py` or `dataset/backend.py`. The rejected alternative (a `filter=` parameter on `read()` plus a clear-after) stays rejected; its correctness depended on one side effect cancelling another.
- **D-2 shipped.** `XrBackend.head` uses `xr.open_dataset`. `open_zarr` appears in this class only in `_assert_append_compatible`, where it is an append-specific concern, and in `head`'s docstring explaining the choice.
- **D-3 shipped — RV-02 stays filed, not closed.** `head()` carries `read()`'s `Path(path).exists()` guard and its exact message on both backends, so a missing store still fails loudly and correctly-named rather than as an xarray engine-guess error or a deferred `PlBackend` collect-time failure. The one behaviour that *did* narrow is now explicit: a dataset whose data exists only in memory (the `_reset_symbols()` → `from_raw_data()` fallback, store never written) was previously probeable via the cache early-return and now raises. Accepted deliberately, and unreachable from any shipped factory or fixture today. Closing RV-02 properly means deciding whether name derivation may be deferred, which re-opens D-05's "names are known from construction onward" contract — a design call for the user, not a side effect of this fix.
- **Task 3's `window` parameter.** `_momentum_config` previously set `window=n`, making the lookback and the horizon the same number. That is itself why no existing test could tell whether `_reset_dataset_config()`'s widening reached the dataset. An optional `window` (defaulting to `n`) separates them without touching any existing caller.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `_momentum_config` could not express `window=20, n=5`**
- **Found during:** Task 3
- **Issue:** The plan's Task 3 action specifies extending `_momentum_config` with `start_date`/`end_date` only, but the test it describes needs `window=20` with `kwargs={"n": 5}`. The existing helper hardcodes `window=n`, so the two could not be separated and the described test was unbuildable as written.
- **Fix:** Added an optional `window: int | None = None` parameter defaulting to `n`, alongside the specified `start_date`/`end_date`. Every pre-existing call site is byte-identical.
- **Files modified:** `tests/test_factor_polars.py`
- **Verification:** All 4 pre-existing tests in the module still pass unchanged; full suite 387 passed.
- **Committed in:** `5668e93`

**2. [Rule 3 - Blocking] Unused `Callable` import after the fixture rewrite**
- **Found during:** Task 1
- **Issue:** `tests/test_backend_head.py`'s new `stores`/`backends` fixtures are typed `dict[str, str]` and `list[tuple[DataBackend, str]]`, leaving the module's `from typing import Callable` import dead.
- **Fix:** Removed the import.
- **Files modified:** `tests/test_backend_head.py`
- **Verification:** Module imports and collects cleanly; 6 tests pass.
- **Committed in:** `6fd3622`

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both are mechanical enablers for the plan's own described tests. No scope creep — no behaviour was added beyond what the plan specified.

## TDD Gate Compliance

Task 1 ran a clean RED → GREEN cycle: `6fd3622` (test, 5 failed / 1 passed — the abstractness test is signature-independent) preceded `8961775` (feat, 13 passed).

Task 3 is marked `tdd="true"` but its test was **green on first run**, because the plan sequences the implementation (Tasks 1–2) ahead of the regression test. This is by design, not a skipped RED: the plan makes Task 3's two mutations the falsifiability proof in place of a temporal RED, and both were run and observed red (`assert 29 == 49`). The test is therefore demonstrably capable of failing for the reason it claims. No REFACTOR commit was needed for either task.

## Issues Encountered

- The plan's T1-a mutation predicted an `AttributeError` but produces a `TypeError` first, because reverting the signature shadows the body. Resolved by running the T1-b variant, which isolates the mechanism and reproduces the predicted signal exactly. Both are recorded above rather than the inconvenient one being dropped.
- The known `kun::StreamContext::~StreamContext()` full-suite flake did not occur; the full run completed in 16s with exit code 0.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- RV-01 is closed and guarded. Phase 4 consumes the factor panel this fix repairs; a `Momentum` + `Alpha158SpotKline` `DLConfig` no longer silently yields a part-NaN `momentum_5` column.
- **RV-02 remains open and filed** in `03-VERIFICATION.md`, deliberately (D-3). Its disposition is now locked by a test rather than by prose. Deciding it means deciding whether `FactorPolars` name derivation may be deferred to first `get_factor_names()`, which re-opens D-05's construction-time-names contract — a user design call.
- No dependency changes (`pyproject.toml` / `uv.lock` diff empty). `README.md` untouched.

---
*Quick task: 260906-w3t*
*Completed: 2026-09-06*
## Self-Check: PASSED

All 4 task commits verified present in `git log`. All 7 modified source/test files
verified on disk, plus this SUMMARY. No scratch artifacts left in `tests/`. Full suite
re-run after restoring every mutation: **387 passed**, pytest exit code 0.
