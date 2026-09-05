---
phase: 02-multi-market-data-foundation
plan: 06
subsystem: testing
tags: [pytest, xarray, dataset-abc, extensibility-contract]

# Dependency graph
requires:
  - phase: 02-02
    provides: Market/Frequency type aliases and DatasetConfig fields
  - phase: 02-03
    provides: dataset/cleaning.py (dedup_raw_frame, flag_anomalies, validate_schema, clean_market_data)
provides:
  - "tests/test_extensibility_contract.py: automated core-layer purity check + FakeDataset lifecycle proof"
affects: [phase-07-quality-cleanup]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Test-only ABC subclass (FakeDataset) proves an abstraction's extensibility contract by construction, without registering a fourth real market"
    - "Grep-style purity check (non-comment-line literal-substring scan) as a repeatable regression guard against architectural erosion in core layers"

key-files:
  created: [tests/test_extensibility_contract.py]
  modified: []

key-decisions:
  - "FakeDataset._raw_data_to_xr() supplies the full OHLCV column set (not just `close`) because dataset/cleaning.py:validate_schema() (landed in 02-03) requires all of open/high/low/close/volume to be present — discovered via a genuine RED-phase test failure, not assumed up front"
  - "Renamed test_no_market_specific_logic_in_core_layers to test_core_layer_purity_no_market_specific_logic so it matches the plan's `-k purity` selector"

patterns-established:
  - "Core-layer purity check pattern: any future core-layer file added to the pipeline should extend CORE_LAYER_FILES/FORBIDDEN_SUBSTRINGS in tests/test_extensibility_contract.py rather than adding a new ad hoc check"

requirements-completed: []  # DATA-03 not yet marked complete — Task 3 (human-verify design/code review) is still pending as of this summary

# Metrics
duration: 25min (Tasks 1-2 only; Task 3 checkpoint pending)
completed: 2026-09-05
---

# Phase 2 Plan 06: Extensibility Contract Proof (Tasks 1-2 of 3) Summary

**Automated grep-style purity check + FakeDataset from_raw_data()->save()->read() lifecycle proof for DATA-03, both passing (2/2); Task 3's human design/code review checkpoint is still pending.**

## Status: IN PROGRESS — paused at checkpoint:human-verify (Task 3)

Tasks 1 and 2 (both automated, `autonomous: false` at the plan level only because of Task 3) are complete and committed. Task 3 is a `checkpoint:human-verify` gate requiring a human to read `base/factor.py`, `base/model.py`, `base/backend.py` in full and confirm no structural coupling (isinstance/hasattr dispatch, market/frequency-value branching) exists beyond what the automated grep in Task 1 can catch. Per the plan's execution instructions, this was NOT resolved autonomously — it is being returned to the orchestrator for the user to review and confirm. This SUMMARY reflects progress through Task 2 only and will need to be finalized once Task 3's checkpoint is resolved.

## Performance

- **Duration:** ~25 min (Tasks 1-2)
- **Started:** 2026-09-05T14:00:00Z (approx.)
- **Completed (Tasks 1-2):** 2026-09-05T14:39:25Z
- **Tasks:** 2 of 3 completed (Task 3 pending human-verify checkpoint)
- **Files modified:** 1 (`tests/test_extensibility_contract.py`, created)

## Accomplishments
- Automated, non-comment-aware grep-style check proving `base/factor.py`, `base/model.py`, `base/backend.py` contain no reference to `SpotKlineDataset`, `StockDataset`, `crypto_spot`, or `us_equity`
- A genuinely novel, test-only `FakeDataset(Dataset)` subclass (market="fake_market", frequency="1d") runs the full `from_raw_data()` -> `save()` -> `read()` lifecycle successfully, proving DATA-03/ROADMAP Success Criterion 4 by construction
- Both checks live in the standard `uv run pytest tests/` suite going forward (18 tests total pass, including the 2 new ones), not as a one-time manual review

## Task Commits

Each task was committed atomically:

1. **Task 1: Automated core-layer purity check** - `6b4a8ac` (test)
2. **Task 2: FakeDataset full lifecycle proof (TDD)** - `9df1652` (test, RED) then `319f4a4` (feat, GREEN)

**Plan metadata:** pending (will be added once Task 3's checkpoint resolves and the plan is fully complete)

_Note: Task 2 used tdd="true" — RED (`9df1652`) then GREEN (`319f4a4`); no REFACTOR commit needed, changes were minimal._

## Files Created/Modified
- `tests/test_extensibility_contract.py` - Core-layer purity check (`test_core_layer_purity_no_market_specific_logic`) + `FakeDataset(Dataset)` test-only subclass and its lifecycle test (`test_fake_dataset_lifecycle`)

## Decisions Made
- Renamed the purity-check test function from `test_no_market_specific_logic_in_core_layers` to `test_core_layer_purity_no_market_specific_logic` so `uv run pytest tests/test_extensibility_contract.py -k purity -v` (the plan's literal acceptance-criteria command) actually selects it — the original name contained no substring matching `purity`.
- `FakeDataset._raw_data_to_xr()` supplies the full OHLCV column set rather than only `close` (as the plan's illustrative example suggested), because `dataset/cleaning.py:validate_schema()` (landed in phase 02-03, after this plan's interfaces section was written) raises `ValueError` if any of `open`/`high`/`low`/`close`/`volume` is missing. This was discovered as a genuine RED-phase test failure and fixed in the GREEN commit — no change to `dataset/cleaning.py` itself, since that validation is a correct, already-established behavior (D-08) that this test-only subclass simply needs to satisfy like any real Dataset subclass would.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test function name did not match plan's `-k purity` selector**
- **Found during:** Task 1
- **Issue:** Task 1's acceptance criteria specifies `uv run pytest tests/test_extensibility_contract.py -k purity -v` must show 1 passed, but the initially-implied name `test_no_market_specific_logic_in_core_layers` contains no `purity` substring, so `-k purity` selects 0 tests.
- **Fix:** Named the test `test_core_layer_purity_no_market_specific_logic` instead, satisfying both the descriptive-name intent and the `-k purity` selector.
- **Files modified:** `tests/test_extensibility_contract.py`
- **Verification:** `uv run pytest tests/test_extensibility_contract.py -k purity -v` shows `1 passed`.
- **Committed in:** `6b4a8ac` (Task 1 commit)

**2. [Rule 1 - Bug] FakeDataset's raw data initially missing required OHLCV columns**
- **Found during:** Task 2 (RED phase)
- **Issue:** The plan's interfaces/behavior description illustrates `_raw_data_to_xr()` with only a `close` variable; running the test against that literal implementation fails with `ValueError: validate_schema: required column(s) missing from dataset: ['open', 'high', 'low', 'volume']` because `Dataset.from_raw_data()` unconditionally runs `clean_market_data()` (landed in 02-03), which requires the full OHLCV set.
- **Fix:** Added `open`/`high`/`low`/`volume` alongside `close`, each with a distinct numeric offset so a round-trip mismatch on any one variable would be caught.
- **Files modified:** `tests/test_extensibility_contract.py`
- **Verification:** `uv run pytest tests/test_extensibility_contract.py -v` shows `2 passed`; full suite (`uv run pytest tests/ -q`) shows `18 passed`.
- **Committed in:** `319f4a4` (Task 2 GREEN commit); the failing state was itself committed first at `9df1652` per the TDD RED/GREEN protocol.

---

**Total deviations:** 2 auto-fixed (both Rule 1 — bugs in the initial literal plan wording, not architectural changes)
**Impact on plan:** Both fixes were necessary for the tests to actually run and pass as specified by the plan's own acceptance criteria. No scope creep — no file outside `tests/test_extensibility_contract.py` was touched.

## Issues Encountered
None beyond the two auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness

Tasks 1-2 are complete, committed, and verified (`uv run pytest tests/ -q` — 18 passed). Task 3 is a blocking `checkpoint:human-verify` gate (per ROADMAP Phase 2 Success Criterion 4's explicit requirement for a human-confirmed design/code review, which a literal-substring grep cannot fully substitute for). The plan is NOT complete until Task 3 is resolved — DATA-03 should not be marked complete in REQUIREMENTS.md, and this SUMMARY should be revised/finalized once the human review confirms no structural coupling was found (or after any coupling found is fixed).

---
*Phase: 02-multi-market-data-foundation*
*Status as of this summary: Tasks 1-2 complete, Task 3 checkpoint pending*
