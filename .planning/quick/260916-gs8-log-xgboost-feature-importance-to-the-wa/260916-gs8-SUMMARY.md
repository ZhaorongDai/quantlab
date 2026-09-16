---
phase: quick-260916-gs8
plan: 01
subsystem: model
tags: [xgboost, wandb, telemetry, feature-importance]
status: complete

requires:
  - "quantlab/ml_model/xgb.py:XGBoostRegressor._record_feature_importance (03.7-18)"
  - "quantlab/ml_model/xgb.py:_WandbEvalCallback (per-round curve steps)"
provides:
  - "feature_importance/{type} — a top-30 descending bar chart per importance type on the W&B Charts dashboard"
  - "feature_importance_table/{type} — a wandb.Table carrying every factor, descending"
  - "XGBoostRegressor._last_log_step — the epoch the per-round callback last recorded"
affects:
  - "quantlab/ml_model/xgb.py"
  - "tests/test_xgb_model.py"
  - "example/model.md"

tech-stack:
  added: []
  patterns:
    - "Best-effort telemetry: per-type try/except Exception with both-or-neither payload assignment, plus an independent try/except around the single log call"
    - "Chart payload logged at the CURRENT step so wandb merges it into the final round's row instead of opening a new one"

key-files:
  created: []
  modified:
    - "quantlab/ml_model/xgb.py"
    - "tests/test_xgb_model.py"
    - "example/model.md"

decisions:
  - "The step problem is solved with an explicit step=self._last_log_step, never commit= — commit= is not in this project's recorder protocol (11 call sites, both test doubles are log(data, step=None)) and commit=False would park a row waiting for a next log that never comes."
  - "import wandb at module scope, not deferred — wandb is already in sys.modules via quantlab/base/model.py, so the cost is a dict lookup, and a deferred import inside best-effort telemetry is exactly where an ImportError would surface mid-training."
  - "Chart keys live in the feature_importance* namespace, disjoint from the summary's importance_ prefix — this is what lets the pre-existing D-02 regression guard (test line 727, now 820) keep working verbatim."
  - "except Exception is deliberately broader than the file's existing except xgb.core.XGBoostError: get_score's failure mode is known and documented, chart rendering has no declared exception contract, and this method runs before _save_model."

metrics:
  duration: ~12 min
  completed: 2026-09-16

actuals:
  tokens: 7592
  tasks: 3
  commits: 3
plan_head_before: 50cff452e9fccf46b54b0d058e401d14b9837c49
---

# Quick Task 260916-gs8: Log XGBoost Feature Importance to the W&B Charts Dashboard Summary

XGBoost feature importance now reaches the W&B **Charts** dashboard as a sorted-descending top-30 bar chart plus a full `wandb.Table` per importance type, logged in one row merged into the final boosting round's step — while every existing `importance_{type}/{name}` summary scalar stays byte-identical.

## What Shipped

**`quantlab/ml_model/xgb.py`**

- Module-scope `import wandb`; two new constants beside `_IMPORTANCE_TYPES`: `_IMPORTANCE_CHART_TOP_N = 30` (D-04) and `_IMPORTANCE_CHART_PREFIX = "feature_importance"` (the deliberately disjoint namespace).
- `XGBoostRegressor._last_log_step`, initialised to `None` in `__init__`, reset to `None` as the first statement of `_fit_model` (so a deep-copied CV fold cannot inherit a stale step), and written by `_WandbEvalCallback.after_iteration` immediately after its existing `recorder.log(row, step=epoch)`.
- `_record_feature_importance` builds, per successful importance type, a full `wandb.Table` (all factors, descending by value) and a `wandb.plot.bar` over the top 30, then logs all of them in a **single** `log(charts, step=self._last_log_step)` call.
- Everything that computes importance is untouched (D-02): the `gblinear` early return, the `get_score` call, the `f{i}` index parsing and its `ValueError`, the 0.0 fill, the non-scalar skip, and the `summary.update` keys/values.

**Threat T-gs8-01 (the highest-severity item)** — this method runs *before* `_save_model`, so a drawing bug must never cost a trained checkpoint. Per-type chart construction sits in its own `try`/`except Exception` with both-or-neither payload assignment; the single `log` call has its own `try`/`except Exception` that warns and swallows. Task 2 proves the checkpoint and `finish()` survive a raising `wandb.plot.bar`.

**`tests/test_xgb_model.py`** — `CHART_PREFIX` plus `_curve_rows()` / `_chart_rows()` helpers that split `recorder.logs`; the five arithmetic-broken tests repaired through `_curve_rows(...)`; three new tests.

**`example/model.md`** — the XGB row of the W&B table and the feature-importance paragraph corrected; the now-false "只进 summary / 不进逐轮曲线" claims replaced, tagged 2026-09-16.

## Measured Baselines

Both baselines were re-measured in this worktree before any edit, and both **match `<planning_measurements>` exactly — no discrepancy to record**:

| Suite | Plan's figure | Measured baseline | Final |
|---|---|---|---|
| `tests/test_xgb_model.py` | 47 | **47 passed** | **50 passed** (+3) |
| Neighbour set (7 files) | 101 | **101 passed** | **101 passed** (unchanged) |

Final counts hit the plan's acceptance numbers exactly: baseline+3 = 50, and the neighbour set is unmoved. `tests/test_ml_models.py` was never edited and neither `FakeRecorder` gained a `commit` parameter.

The environment facts were verified rather than assumed: `quantlab.__file__` resolves into this worktree, wandb is 0.29.0, `wandb.plot.bar` returns a `wandb.plot.custom_chart.CustomChart` exposing `.table.data`, and `wandb.Table.data` is list-of-lists preserving row order.

## Tests Added

| Test | What turns it red |
|---|---|
| `test_importance_charts_reach_the_wandb_run_in_one_row_at_the_final_round_step` | Charts opening a step of their own (curve would stop being `range(12)`), a type charted without its table, the never-split `f_const` dropped, rows not descending, or a summary scalar moving |
| `test_importance_charts_are_sorted_descending_and_capped_at_the_top_30` | The 30-of-35 cut, the descending order with 0.0-filled factors last, the step being reinvented rather than passed through, or any of the 105 summary scalars changing |
| `test_a_chart_that_fails_to_build_is_skipped_and_the_others_still_chart` | A per-type chart failure costing the checkpoint, contaminating the other types, half-charting a type, or losing the `gain` summary entries |

The second test drives `_record_feature_importance` directly with a fake Booster and **no training at all** — 30-of-35 is only an unambiguous assertion when the correct answer is fixed by construction.

## Constraint Compliance

- **Task 1 kept indivisible** — the chart row and the five arithmetic repairs landed in one commit, so no commit is red.
- **No assertion weakened** — the step-contiguity check, `len(...) == 25`, `len(...) == 12` and the `{"train-rmse", "val-rmse"} <= set(row)` checks are all still made, now over the curve rows.
- **Lines 331, 373 and 727 left verbatim.** The D-02 guard (`not any(key.startswith("importance_") ...)`, now line 820) still works unchanged because chart keys start with `feature_importance`. Only that test's docstring was updated, to say what the guard now guards.
- **D-02 byte-exact** — no line of the importance computation was touched.

## Deviations from Plan

None — the plan executed exactly as written. No auto-fixes were required, no authentication gates were hit, and no architectural decision arose.

## Known Stubs

None. The changed files were scanned for stub markers (`TODO`, `FIXME`, `placeholder`, `coming soon`, `not available`) with zero hits, and for skipped/xfailed tests — every `skip` occurrence is prose inside a test name, docstring, or warning assertion, not `pytest.mark.skip` or `xfail`.

## Threat Flags

None. No new network endpoint, auth path, file-access pattern, or trust-boundary schema change was introduced. T-gs8-03 (factor names sent to W&B) remains `accept`: the identical names already go to the same run as summary keys, so there is no new recipient and no new field.

## Commits

| Task | Commit | Message |
|---|---|---|
| 1 | `ec3b7bf` | feat(quick-260916-gs8): chart xgboost feature importance in the W&B run |
| 2 | `7b160f6` | test(quick-260916-gs8): lock the importance chart order, cut and envelope |
| 3 | `6d38b43` | docs(quick-260916-gs8): correct the docs that said importance never reaches log |

Measured from the plan ledger: `git rev-list --count 50cff45..HEAD` = **3**. `git diff --stat` over the same range touches exactly `quantlab/ml_model/xgb.py`, `tests/test_xgb_model.py` and `example/model.md` (347 insertions, 28 deletions), matching the plan's verification clause. No commit deleted a tracked file; the working tree is clean with no untracked files.

## Self-Check: PASSED

- All three modified files exist on disk (`quantlab/ml_model/xgb.py`, `tests/test_xgb_model.py`, `example/model.md`).
- All three commits exist in the log (`ec3b7bf`, `7b160f6`, `6d38b43`), and `rev-list --count` from the recorded `plan_head_before` independently measures 3.
- Both verification suites re-run green at the final commit: 50 passed and 101 passed.
