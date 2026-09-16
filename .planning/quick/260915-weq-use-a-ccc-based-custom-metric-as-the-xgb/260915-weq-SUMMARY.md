---
phase: quick-260915-weq
plan: 01
subsystem: model
tags: [xgboost, early-stopping, metrics, ml-model]
status: complete
requires:
  - quantlab/ml_model/xgb.py:XGBoostRegressor
  - xgboost 3.4.1 custom_metric contract
provides:
  - quantlab/ml_model/xgb.py:pooled_ccc_loss
  - quantlab/ml_model/xgb.py:ccc_loss_metric
  - "ccc_loss as the xgboost early-stopping decision metric"
affects:
  - quantlab/ml_model/xgb.py
  - tests/test_xgb_model.py
  - example/model.md
tech_stack:
  added: []
  patterns:
    - "xgb.train(custom_metric=) -> last metric in each data set's list -> EarlyStopping(metric_name=None) resolves to it"
    - "degenerate numeric guards return the WORST value, never NaN, so early-stopping comparisons stay meaningful"
key_files:
  created: []
  modified:
    - quantlab/ml_model/xgb.py
    - tests/test_xgb_model.py
    - example/model.md
decisions:
  - "D-01..D-05 applied as written; no config field, no per-date variant (D-03), reference oracle lives only in the test file"
metrics:
  duration: ~35 min
  completed: 2026-09-15
commits: 3
plan_head_before: 78285f951019ad279c3f70b28f967c3e7602fcad
actuals:
  tasks: 3
  commits: 3
---

# Phase quick-260915-weq Plan 01: CCC-based early-stopping criterion Summary

A pooled CCC loss (`1 - ccc`) is now the xgboost early-stopping decision metric, registered
through `xgb.train(custom_metric=...)`, with RMSE demoted to an observation curve — and the
criterion swap is machine-proved to be non-vacuous by a pinned seed where the two argmins
disagree by 20 rounds.

## What shipped

| Task | Commit | What |
|---|---|---|
| 1 (tracer, TDD) | `0e133f9` | `pooled_ccc_loss` + `ccc_loss_metric`, wired via `custom_metric=`; oracle + criterion tests |
| 2 (TDD) | `fca0fd4` | Locks on curve keys, loss direction, four degenerate inputs, multi-label orientation |
| 3 | `b84a5e7` | Every RMSE-as-criterion statement corrected; pooled trade-off recorded as a user decision |

## Test counts

| Suite | Before | After |
|---|---|---|
| `tests/test_xgb_model.py` | **31 passed** (re-measured on this machine: 53.9s cold, 4.4s warm) | **47 passed** (+16) |
| Model-layer neighbours (7 files) | — | **101 passed** |

No LEARNING assertion went red. `test_learns_a_factor_driven_label`,
`test_unrelated_label_gives_no_ic`, `test_multi_label_predicts_every_label` and
`test_train_cv_sequential`'s `cv_mean_test_ic > 0.3` all passed unchanged — no threshold was
touched. Switching the criterion to CCC did not degrade what the model learns on these panels.

## Measured: the multi-output `get_label()` layout

**The planning fact was wrong.** The plan said `dtrain.get_label()` "returns one flat vector for
a multi-output DMatrix". Measured on xgboost 3.4.1 with a throwaway 3-row / 2-label DMatrix:

```
label in : [[10,20],[11,21],[12,22]]
get_label(): [[10. 20.] [11. 21.] [12. 22.]]   shape (3, 2)  <- already 2-D, row-major
```

It returns `(n_rows, n_labels)` **two-dimensional and row-major**; single-label returns a flat
`(n_rows,)`. The planned `reshape(num_row(), -1)` is still exactly right — it normalises both
shapes and is an identity op on the 2-D case — so the code is unchanged by this correction, but
the docstring now records what was actually observed rather than what was assumed.

## The pinned seed, and why that seed

`CRITERION_SEED = 17`, 300 rounds, patience 10.

Chosen by searching seeds 10..59 driving the same harness. The result that matters:

| seed | best_iteration | argmin val-ccc_loss | argmin val-rmse | diverges |
|---|---|---|---|---|
| 13 (the provisional pin) | 1 | 1 | 1 | **no** |
| **17 (pinned)** | **65** | **65** | **45** | **yes, by 20 rounds** |

The plan's illustrative seed 13 puts both argmins on round 1, so the divergence arm would have
been vacuous — exactly threat T-weq-01. Seed 17 was picked for the widest clean gap among the
short-running candidates, so the assertion cannot flip on a rounding accident. Roughly a third of
seeds do not diverge at all, which is why this constant is documented as a measured choice.

**Mutation-verified, not merely green.** Deleting `custom_metric=ccc_loss_metric` from
`xgb.train` turns exactly three tests red — the criterion test, the curves test, and
`test_hyperparameters_pass_through`'s key-set check. The feature cannot go inert behind a green
suite. (This follows the repo's own 03.2 lesson that a lock passing on arrival must be
mutation-verified.)

## The Task 3 gate: what I replaced, and why

The plan's gate was weaker than its own `<done>` claim:

```
! grep -rn "..." fileA fileB && grep -c "ccc_loss" quantlab/ml_model/xgb.py example/model.md && uv run pytest ...
```

`grep -c PATTERN fileA fileB` prints a per-file count and **exits 0 if any one file matched**, so
it would have passed with `example/model.md` containing no `ccc_loss` at all. Replaced with three
separate probes, each of which independently fails:

| Probe | Result |
|---|---|
| `grep -c "ccc_loss" quantlab/ml_model/xgb.py` | **11** |
| `grep -c "ccc_loss" example/model.md` | **7** |
| `grep -rn "默认 RMSE\|逐轮曲线与早停判据" quantlab/ml_model/xgb.py example/model.md` | empty |
| `uv run pytest tests/test_xgb_model.py -q` | 47 passed |

Run against the pre-Task-3 tree these correctly fail: `example/model.md` scored **0** and the
banned phrases matched 3 lines. The flawed gate would have reported success on that same tree —
confirming the defect was real, not theoretical.

Every surviving "RMSE" mention in both files was reviewed by eye: all four either state the
criterion is *not* RMSE, or flag that `best_score` used to be one.

## Deviations from plan

### [Rule 1 - Bug] Two test-authoring bugs, both caught by a real RED

**1. The orientation test's oracle ignored float32 label storage.**
Found during Task 2. `xgb.DMatrix` stores labels as **float32**, so a float64 array handed in
comes back rounded; the adapter therefore scores the round-tripped value. My first oracle used
the pristine float64 array and the two disagreed in the 8th significant digit
(`1.0604537996708554` vs `1.060453803647541`). Fixed by computing the expectation on
`labels.astype(np.float32).astype(np.float64)` and documenting why. **This is a property of
xgboost, not a defect in the shipped metric** — the Task 1 oracle tests call `pooled_ccc_loss`
directly on float64 arrays with no DMatrix involved, and match the reference to `rel=1e-12`, so
D-04's "the number must be identical" holds exactly where it is claimed.

**2. My "wrong reading" was actually the right one.** The same test asserted the value differed
from `labels.reshape(-1, order="F")[:n]`. Verified empirically that this Fortran-order slice *is*
column 0, so the assertion was asking the value to differ from itself. Replaced with the C-order
slice `reshape(-1)[:n]`, which is the genuine transposed misreading (it interleaves both labels).

Both were found by running the tests, not by reading them.

### Planning facts that turned out wrong

1. **`get_label()` layout** — see above; 2-D row-major, not a flat vector. No code impact.
2. **Baseline timing** — the plan recorded 4.37s; measured 53.9s on the first cold run in this
   worktree (venv build) and 3.8–5.2s warm. The 31-test count matched exactly.
3. **Seed 13 does not diverge** — the plan's CRITERION sketch would have been vacuous on it.

Everything else in `<planning_facts>` held, including the load-bearing one: `custom_metric`
becomes the early-stopping criterion with no extra configuration, and `_WandbEvalCallback`
needed **zero** edits (now asserted by a test rather than assumed).

## Locked decisions honoured

- **D-03 (no per-date variant)** — scanned both source files and the test file for
  `逐日` / `per-date` / `截面 ccc` / `cross-sectional ccc` / `TODO` / `FIXME`: clean. No variant,
  switch, comment or proposal anywhere in the diff.
- **D-05 (no config field)** — `custom_metric` is passed unconditionally; no new config key.
- `DEFAULT_PARAMS["eval_metric"]` is still `"rmse"` and rmse curves are still recorded
  (`test_default_params` unchanged and green).
- The user's reference implementation exists **only** in `tests/test_xgb_model.py` as
  `_reference_ccc_loss`, never in production code.

## Threat register outcomes

| Threat | Outcome |
|---|---|
| T-weq-01 (metric ordering silently reverts to RMSE) | Mitigated **and mutation-proved**: deleting the kwarg turns 3 tests red |
| T-weq-02 (`best_score` changes meaning silently) | Documented in both the class docstring and `example/model.md` |
| T-weq-03 (degenerate input aborts training) | Four degenerate inputs locked to a finite `1.0` under `np.errstate(invalid/divide/over="raise")` |

No new threat surface: no network, no dependency, no package install, no file-format change.

## Notes for the orchestrator

`STATE.md` was deliberately **not** modified and no docs commit was made — per the execution
constraints, the orchestrator owns the docs commit. The three commits above contain code and
tests only.

## Self-Check: PASSED

- `quantlab/ml_model/xgb.py` — FOUND (modified)
- `tests/test_xgb_model.py` — FOUND (modified)
- `example/model.md` — FOUND (modified)
- `.planning/quick/260915-weq-.../260915-weq-SUMMARY.md` — FOUND
- commits `0e133f9`, `fca0fd4`, `b84a5e7` — all FOUND in `git log`
- `git rev-list --count 78285f95..HEAD` = **3** (measured, matches `commits:`)
- `git diff --stat 78285f95..HEAD` touches exactly the 3 files in `files_modified`
- no file deletions in any commit; no untracked leftovers
