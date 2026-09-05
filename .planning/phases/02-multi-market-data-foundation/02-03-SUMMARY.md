---
phase: 02-multi-market-data-foundation
plan: 03
subsystem: data
tags: [xarray, polars, data-cleaning, data-quality, dataset]

# Dependency graph
requires:
  - phase: 02-multi-market-data-foundation
    provides: "02-01 Wave 1 groundwork (test infra, XrBackend.write() mode=\"w\" fix)"
provides:
  - "dataset/cleaning.py: dedup_raw_frame(), flag_anomalies(), validate_schema(), clean_market_data()"
  - "base/data.py:Dataset.from_raw_data() centrally calls clean_market_data() for every current and future Dataset subclass"
affects: [02-04, 02-05, 02-06, 02-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Leaf cleaning module (dataset/cleaning.py) with zero project-internal imports, consumed by base/data.py — mirrors the base->dataset non-circular import direction already established by dataset.backend"
    - "Two-stage cleaning split: dedup_raw_frame() runs pre-to_xarray() inside each subclass's _raw_data_to_xr() (Waves 3-4's job); clean_market_data() runs post-conversion, centrally, once, in base/data.py"

key-files:
  created:
    - dataset/cleaning.py
    - tests/test_cleaning.py
  modified:
    - base/data.py

key-decisions:
  - "dedup_raw_frame() defaults to keep=\"last\" — later-arriving vendor files more often carry corrected/reprocessed data than earlier ones"
  - "flag_anomalies() extreme-jump check guards the prior (shifted) price against zero/negative/NaN before computing a percentage change, avoiding spurious inf/NaN-driven false-positive flags when the previous data point was itself an anomaly"
  - "No forward-fill/interpolate/fillna anywhere in dataset/cleaning.py (enforced by grep in acceptance criteria) — NaN gaps pass through unchanged by design (D-06)"

patterns-established:
  - "Shared cleaning hook wired once in base/data.py:Dataset.from_raw_data() — new Dataset subclasses inherit dedup-adjacent NaN-gap preservation, anomaly-flagging, and schema validation with zero subclass code changes"

requirements-completed: [DATA-04]

# Metrics
duration: ~20min
completed: 2026-09-05
---

# Phase 2 Plan 03: Shared Market-Data Cleaning Module Summary

Built `dataset/cleaning.py` (dedup, anomaly-flagging, schema validation) and wired its post-conversion half centrally into `base/data.py:Dataset.from_raw_data()`, so every current and future `Dataset` subclass gets NaN-gap-preserving, non-destructive data cleaning for free.

## Performance

- **Duration:** ~20 min
- **Started:** 2026-09-04T22:19:00-04:00 (first RED commit)
- **Completed:** 2026-09-04T22:24:00-04:00 (last GREEN commit)
- **Tasks:** 2/2 completed
- **Files modified:** 3 (1 new module, 1 new test file, 1 modified)

## Accomplishments
- `dataset/cleaning.py` created as a leaf module (zero `base`/`dataset` imports) providing `dedup_raw_frame()`, `flag_anomalies()`, `validate_schema()`, `clean_market_data()` — satisfying CONTEXT.md D-05 through D-08.
- `base/data.py:Dataset.from_raw_data()` now calls `clean_market_data()` centrally between raw-conversion and backend persistence — no changes needed in `StockDataset` or `SpotKlineDataset` to benefit from it (directly satisfies ROADMAP Success Criterion 4).
- Full unit test coverage (9 tests) for all four cleaning rules, following strict per-task TDD (RED commit, then GREEN commit, for both tasks).

## Task Commits

Each task was committed atomically, following per-task TDD (RED then GREEN):

1. **Task 1: Implement dedup_raw_frame() (D-05)**
   - `ac82ac7` (test) — 3 failing tests for dedup semantics and the to_xarray() crash it prevents
   - `4b34d17` (feat) — `dedup_raw_frame()` implementation, all 3 tests green
2. **Task 2: Implement flag_anomalies()/validate_schema()/clean_market_data() and wire into from_raw_data() (D-06, D-07, D-08)**
   - `f798d62` (test) — 6 failing tests (flag-without-mutate, schema raise/warn, NaN-gap preservation, from_raw_data wiring, required-columns constant)
   - `56b6eb1` (feat) — full implementation + `base/data.py` wiring, all 9 tests green

## Files Created/Modified
- `dataset/cleaning.py` - New leaf module: `dedup_raw_frame()` (tabular polars dedup), `flag_anomalies()` (xr.Dataset boolean `anomaly_flag` for zero/negative price or extreme jump), `validate_schema()` (raise on missing required column, warn on unexpected non-key nulls), `clean_market_data()` (composes the latter two)
- `base/data.py` - Added `from dataset.cleaning import clean_market_data` import; `Dataset.from_raw_data()` now calls `clean_market_data(data)` between `_raw_data_to_xr()` and `to_internal(data)`
- `tests/test_cleaning.py` - 9 unit tests covering dedup determinism, the to_xarray() crash it prevents, anomaly-flag correctness/non-mutation, schema validation raise/warn behavior, NaN-gap preservation, and the `from_raw_data()` wiring (via monkeypatch)

## Decisions Made
- **Extreme-jump denominator guard:** the initial `flag_anomalies()` implementation divided by the prior `close` value unconditionally, which produced `inf` (and a spurious jump flag) whenever the prior price was itself zero/negative — a case already flagged by the zero/negative-price check. Fixed by only evaluating the jump condition when the prior (shifted) price is strictly positive, so an already-anomalous prior point doesn't cascade into a second, unrelated flag on the next point. This surfaced during Task 2's GREEN run (test 4 failed once before the fix) and was resolved before committing GREEN — not a plan deviation, just implementation refinement within the TDD RED->GREEN loop for the same task.
- Confirmed no import cycle introduced: `dataset.stock`, `dataset.spot`, and `base.data` all import cleanly together after the change.

## Deviations from Plan

None - plan executed exactly as written. `dataset/cleaning.py`'s function signatures, `base/data.py`'s insertion point, and the threat-model mitigations (T-02-03-01/02/03) all match the plan's `<action>`/`<interfaces>` sections directly.

## Issues Encountered

**Worktree base drift (pre-execution, not part of the plan's tasks):** At startup, the worktree's HEAD was found on the repo's `Initial commit` (64200da) rather than the expected phase-tracking commit (8a9851c) that this plan was supposed to branch from — the worktree had somehow been created without the accumulated Phase 1/Phase 2-Wave-1 history. Working tree was clean (`git status --short` empty), so per the plan's `<worktree_branch_check>` protocol this was corrected with the one sanctioned `git reset --hard 8a9851c8af1b813a89223abf355e9f915618c394` before any plan work began. No plan-authored commits or files were affected by this correction.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

`dataset/cleaning.py` is ready for Wave 3/4 plans (02-04 onward) to call `dedup_raw_frame()` directly from `StockDataset._raw_data_to_xr()` and `SpotKlineDataset._raw_data_to_xr()` (pre-`to_xarray()` insertion point, per 02-PATTERNS.md Section 3 — not this plan's scope). `clean_market_data()` is already active for any `Dataset` subclass today with zero further wiring needed. No blockers identified for downstream plans in this phase.

---
*Phase: 02-multi-market-data-foundation*
*Completed: 2026-09-05*
