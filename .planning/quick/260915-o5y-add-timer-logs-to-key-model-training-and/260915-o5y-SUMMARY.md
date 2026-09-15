---
phase: quick-260915-o5y
plan: 01
subsystem: model-training, backtest
tags: [loguru, timer, logging, xgboost, vectorbt]

requires:
  - phase: 03.7
    provides: MLModel._fit, BaseModel.collect, BaseBacktester._backtest_window / _align_and_predict
provides:
  - "Start and consumed-time INFO logs for six key steps: collect merge, to_array, fit_model, evaluate, align_and_predict, simulate"
affects: [model-training, backtest, large-run observability]

actuals:
  tokens: 1234
  tasks: 2
  commits: 2
plan_head_before: 415485b3595554fa3eb1bd0283dc3e8898c1882b

tech-stack:
  added: []
  patterns:
    - "with Timer(f\"{self.class_name}: <step>\") around heavy pipeline steps (house style from base/factor.py)"

key-files:
  created: []
  modified:
    - quantlab/base/model.py
    - quantlab/base/backtest.py

key-decisions:
  - "align_and_predict timer lives inside _align_and_predict (not at the call site) so every run() window and every run_cv fold logs it"
  - "run_cv's stitched-curve _simulate call left untimed, per locked scope"

patterns-established:
  - "Key-step timing: one Timer per heavy step, never inside per-batch/per-round/per-symbol loops"

requirements-completed: [260915-o5y]

coverage:
  - id: D1
    description: "Six key-step timers emit start and consumed-time lines in a real train-then-backtest run"
    requirement: "260915-o5y"
    verification:
      - kind: integration
        ref: "tests/test_backtest_run.py#test_run_load_mode_end_to_end_long_only (pytest -s, grep for all six timer names)"
        status: pass
    human_judgment: false
  - id: D2
    description: "No behavior change: targeted suites unchanged and diff is addition-only"
    requirement: "260915-o5y"
    verification:
      - kind: unit
        ref: "uv run pytest tests/test_ml_models.py tests/test_xgb_model.py tests/test_backtest_engine.py tests/test_backtest_run.py tests/test_backtest_run_cv.py tests/test_backtest_persistence.py -q (107 passed)"
        status: pass
      - kind: other
        ref: "git diff -w -U0 38a9be1 -- quantlab/base/model.py quantlab/base/backtest.py (0 removed, 8 added lines exactly matching expected set)"
        status: pass
    human_judgment: false

duration: 4min
completed: 2026-09-15
status: complete
---

# Quick 260915-o5y Plan 01: Timer Logs for Key Training and Backtest Steps Summary

**loguru `Timer` start/consumed-time logs around BaseModel.collect's merge, MLModel._fit's to_array/fit_model/evaluate, and the backtester's align_and_predict and simulate steps, so large Alpha101 -> XGBoost -> vectorbt runs are no longer silent after `Return: cal`.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-09-15T21:36:03Z
- **Completed:** 2026-09-15T21:39:32Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `BaseModel.collect`: `{cls}: collect merge` wraps only `combine_by_coords` + `sortby`.
- `MLModel._fit`: `{cls}: to_array` wraps the four-way conversion. `{cls}: fit_model` wraps `_fit_model`. `{cls}: evaluate` wraps the train, val and test evaluations.
- `BaseBacktester._align_and_predict`: `{cls}: align_and_predict` wraps the feature recompute and `predict_panel`. It fires for every run() window and every run_cv fold.
- `BaseBacktester._backtest_window`: `{cls}: simulate` wraps the `_simulate` call.
- The only textual changes are 2 import lines and 6 `with Timer(...)` lines, plus re-indentation of the wrapped statements. Nothing about computation, ordering, return values or signatures changed.

## Task Commits

1. **Task 1: Tracer, Timer imports plus collect merge and simulate timers** - `ac9e258` (feat)
2. **Task 2: to_array, fit_model, evaluate and align_and_predict timers** - `35a4601` (feat)

## Files Created/Modified
- `quantlab/base/model.py` - Timer import; collect merge timer; to_array, fit_model and evaluate timers in MLModel._fit
- `quantlab/base/backtest.py` - Timer import; align_and_predict timer inside _align_and_predict; simulate timer in _backtest_window

## Verification Results
- Tracer gate: running `test_run_load_mode_end_to_end_long_only` with `-s` exited 0. Its output had the collect merge and simulate start and consumed-time lines. The whitespace-insensitive diff at that point was addition-only with the 4 expected lines. Auto mode was off and the check was fully automated, so this passing re-run was the gate and no checkpoint was needed.
- The six targeted suites reported 107 passed, the same as the baseline.
- A log probe on the same node showed a `Starting {cls}: {name}` line and a `{cls}: {name} consumed time: X.XXs` line for all six names.
- `git diff -w -U0 38a9be1` has 0 removed lines and 8 added lines, exactly the expected set.
- Only `quantlab/base/backtest.py` and `quantlab/base/model.py` changed under quantlab/, tests/ and test.py.
- Tests ran from the worktree against the main checkout's `.venv`. `quantlab.__file__` resolved to the worktree copy.

## Decisions Made
None beyond the plan. The locked scope was followed as written: run_cv's stitched `_simulate`, fingerprints, `wandb.init`, checkpoint saving, xgboost rounds, DLModel and test.py all remain untimed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- A large Alpha101Stock -> Return -> XGBoostRegressor -> USEquityCrossectionSelectStockVectorBt run will now log each heavy step's start and duration.

## Self-Check: PASSED
- FOUND: quantlab/base/model.py
- FOUND: quantlab/base/backtest.py
- FOUND: .planning/quick/260915-o5y-add-timer-logs-to-key-model-training-and/260915-o5y-SUMMARY.md
- FOUND: ac9e258
- FOUND: 35a4601
