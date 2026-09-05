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

requirements-completed: [DATA-03]  # DATA-03 fully proven: automated grep (Task 1) + FakeDataset lifecycle (Task 2) + human-confirmed design/code review (Task 3)

# Metrics
duration: 25min (Tasks 1-2) + human review turnaround (Task 3)
completed: 2026-09-05
---

# Phase 2 Plan 06: Extensibility Contract Proof Summary

**Automated grep-style purity check + FakeDataset from_raw_data()->save()->read() lifecycle proof for DATA-03, both passing (2/2), plus a human-confirmed design/code review finding no structural coupling in base/factor.py, base/model.py, base/backend.py.**

## Status: COMPLETE

All three tasks are complete. Tasks 1 and 2 (automated) were committed first. Task 3, a `checkpoint:human-verify` gate, required a human to read `base/factor.py`, `base/model.py`, `base/backend.py` in full and confirm no structural coupling (isinstance/hasattr dispatch, market/frequency-value branching) exists beyond what the automated grep in Task 1 can catch. The reviewer confirmed: no market-specific coupling beyond the automated grep's scope was found in `base/factor.py`, `base/model.py`, `base/backend.py`. All three files interact with `Dataset`-shaped objects only through the `Dataset`/`DataBackend` ABC's public interface (`get_xarray_dataset`, `to_kunquant`, `num_symbols`, `symbols`, `get_config`, etc.); the only `isinstance`/`hasattr` checks present (in `base/model.py`) are for `torch.nn.Module`/`DLConfig`/`MLConfig`/`torch.Tensor` types, unrelated to any concrete `Dataset` subclass or to `config.market`/`config.frequency` values. This resolves the checkpoint and completes DATA-03/ROADMAP Phase 2 Success Criterion 4.

## Performance

- **Duration:** ~25 min (Tasks 1-2, automated) + human review turnaround (Task 3)
- **Started:** 2026-09-05T14:00:00Z (approx.)
- **Completed (Tasks 1-2):** 2026-09-05T14:39:25Z
- **Completed (Task 3, human review confirmed):** 2026-09-05
- **Tasks:** 3 of 3 completed
- **Files modified:** 1 (`tests/test_extensibility_contract.py`, created)

## Accomplishments
- Automated, non-comment-aware grep-style check proving `base/factor.py`, `base/model.py`, `base/backend.py` contain no reference to `SpotKlineDataset`, `StockDataset`, `crypto_spot`, or `us_equity`
- A genuinely novel, test-only `FakeDataset(Dataset)` subclass (market="fake_market", frequency="1d") runs the full `from_raw_data()` -> `save()` -> `read()` lifecycle successfully, proving DATA-03/ROADMAP Success Criterion 4 by construction
- Both checks live in the standard `uv run pytest tests/` suite going forward (18 tests total pass, including the 2 new ones), not as a one-time manual review

## Task Commits

Each task was committed atomically:

1. **Task 1: Automated core-layer purity check** - `6b4a8ac` (test)
2. **Task 2: FakeDataset full lifecycle proof (TDD)** - `9df1652` (test, RED) then `319f4a4` (feat, GREEN)
3. **Task 3: Human design/code review checkpoint** - resolved via explicit human confirmation (no code change; this SUMMARY finalization is the closing commit)

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

## Human Review (Task 3)

**Checkpoint:** `checkpoint:human-verify`, gate="blocking"
**Scope reviewed:** `base/factor.py`, `base/model.py`, `base/backend.py` (full-file read, not just the grep's literal-substring scope)
**Resume signal received:** "confirmed"
**Reviewer's finding:** No market-specific coupling beyond the automated grep's scope was found in `base/factor.py`, `base/model.py`, `base/backend.py`. All three files interact with `Dataset`-shaped objects only through the `Dataset`/`DataBackend` ABC's public interface (`get_xarray_dataset`, `to_kunquant`, `num_symbols`, `symbols`, `get_config`, etc.); the only `isinstance`/`hasattr` checks present (in `base/model.py`) are for `torch.nn.Module`/`DLConfig`/`MLConfig`/`torch.Tensor` types, unrelated to any concrete `Dataset` subclass or to `config.market`/`config.frequency` values.

This satisfies ROADMAP Phase 2 Success Criterion 4's literal "design/code review" requirement in full, complementing (not replacing) Task 1's automated grep check.

## Next Phase Readiness

All three tasks are complete, committed, and verified (`uv run pytest tests/test_extensibility_contract.py -v` — 2 passed). DATA-03/ROADMAP Phase 2 Success Criterion 4 is now fully proven: automated grep purity check + FakeDataset lifecycle proof + human-confirmed design/code review. This plan is closed; DATA-03 completion will be reflected in REQUIREMENTS.md/STATE.md/ROADMAP.md by the orchestrator during the centralized merge, per this plan's parallel-executor scope boundary.

---
*Phase: 02-multi-market-data-foundation*
*Status as of this summary: COMPLETE — all 3 tasks done, Task 3 human review confirmed*
