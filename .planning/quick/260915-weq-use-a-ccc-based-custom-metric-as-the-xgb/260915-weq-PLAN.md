---
phase: quick-260915-weq
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/ml_model/xgb.py
  - tests/test_xgb_model.py
  - example/model.md
autonomous: true
requirements: [QUICK-260915-weq]

must_haves:
  truths:
    - "An XGBoost run with early_stopping=True picks the round that minimises validation CCC loss, not validation RMSE."
    - "The shipped metric returns the same number as the user's reference implementation on non-degenerate input."
    - "The per-round W&B curves carry train-ccc_loss / val-ccc_loss alongside the existing rmse curves, with no change to _WandbEvalCallback."
    - "Constant predictions, a single row, and a degenerate denominator return a finite loss instead of raising."
    - "No reader of xgb.py or example/model.md is told that the criterion is RMSE, and the pooled-vs-cross-sectional trade-off is recorded as a deliberate choice."
  artifacts:
    - quantlab/ml_model/xgb.py
    - tests/test_xgb_model.py
    - example/model.md
  key_links:
    - "xgb.train(custom_metric=...) -> the LAST metric in each data set's list -> xgb.callback.EarlyStopping(metric_name=None) watches it"
    - "DEFAULT_PARAMS['eval_metric']='rmse' stays FIRST, so rmse remains an observation curve and ccc_loss remains the decision metric"
    - "dtrain.get_label() flat vector -> reshape to (n_rows, L) -> primary label column 0, matching MLModel._compute_metrics"
---

<objective>
Make a pooled CCC (concordance correlation coefficient) loss the xgboost early-stopping criterion for `XGBoostRegressor`, replacing RMSE as the decision metric while keeping RMSE visible as a per-round observation curve.

Purpose: RMSE rewards a prediction for being small; CCC rewards it for tracking the label in both location and scale. The user chose the pooled form deliberately.
Output: a custom metric wired into `xgb.train`, machine-checked against the user's own reference implementation, plus the docstring and doc corrections the change forces.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@quantlab/ml_model/xgb.py
@tests/test_xgb_model.py
@example/model.md
</context>

<planning_facts>
Measured at planning time against this tree (`ac27faa`, clean). Trust a file over this list if they disagree, and record the discrepancy in the SUMMARY.

- **Baseline: `uv run pytest tests/test_xgb_model.py -q` = 31 passed in 4.37s.** That file is the dedicated coverage for `XGBoostRegressor`; `tests/test_ml_models.py` covers `MLModel` orchestration with a stub head and explicitly defers the xgboost head to it. `test_model_hierarchy.py`, `test_model_predict_panel.py`, `test_config_roundtrip.py`, `test_ml_backend.py` touch the class only for config round-trip / persistence / panel shape.
- xgboost is **3.4.1**. `custom_metric` signature is `(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]`; labels come from `dtrain.get_label()`. The user's reference takes `(y_true, y_pred)`, so an adapter is unavoidable.
- **`custom_metric` becomes the early-stopping criterion with no extra configuration.** The built-in `eval_metric` comes FIRST in each data set's metric list and `custom_metric` LAST; `xgb.callback.EarlyStopping` with `metric_name=None` watches the LAST one (`xgboost/callback.py` line 482, already documented at `example/model.md:994-995`). Keeping `eval_metric: "rmse"` in `DEFAULT_PARAMS` is therefore safe AND desirable.
- The returned value is a **loss** (`1 - ccc`), lower is better, matching `EarlyStopping`'s default `maximize=False`. Do **not** pass `maximize=True`.
- `_WandbEvalCallback` (xgb.py:49-58) iterates `evals_log` generically, so `train-ccc_loss` / `val-ccc_loss` appear in W&B automatically. **Expect zero edits there** — Task 2 proves it rather than assuming it.
- `best_score` written to the W&B summary (xgb.py:286-292) silently changes meaning: after this change it is a CCC loss, not an RMSE, and is not comparable to any previously recorded run. No code change needed; the docstring must say so.
- `_to_rows` (xgb.py:219-226) flattens `[T,S,*]` to rows and has already dropped date membership, and drops any row whose labels are not all finite. The pooled form is the natural fit for that shape.
- **`timeout` and `gtimeout` do not exist on this machine.** Never wrap a verify command in one — it exits 0 having run nothing and reads as a false green.
- `DEFAULT_PARAMS` in this checkout is intact. Do not "restore" anything.
- **One existing test is expected to go red and must be updated, not worked around:** `test_hyperparameters_pass_through` (xgb_model.py:369) asserts `set(recorders[0].logs[0][0]) == {"train-mae", "val-mae"}` with an exact `==`. A custom metric adds two more keys to every row.
</planning_facts>

<locked_decisions>
From the user directly. Do not re-litigate.

- **D-01**: CCC replaces RMSE as the early-stopping *decision* metric.
- **D-02**: The reference implementation is the user's:
  `mu/var` of both vectors, `cov = mean((pred-mu_pred)*(true-mu_true))`, `ccc = 2*cov / (var_pred + var_true + (mu_pred-mu_true)**2)`, returning `1 - ccc`.
- **D-03**: **POOLED, not per-date.** A per-date cross-sectional variant was proposed and explicitly rejected ("不做逐日截面 ccc 就按我发的做"). Do NOT add a per-date variant, a config switch, or a TODO/comment proposing one anywhere in code, tests or docs.
- **D-04**: The code shape is free ("不一定要按我的写法 结果一样即可") but the returned VALUE must match the reference. The signature has to change anyway, and NaN/degenerate guards are authorised — but on non-degenerate all-finite input the number must be identical.
- **D-05**: No config field. The criterion change is unconditional, matching how `eval_metric` is set today.
</locked_decisions>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Wire a pooled CCC loss through xgb.train and prove it is what EarlyStopping watches</name>
  <files>quantlab/ml_model/xgb.py, tests/test_xgb_model.py</files>
  <read_first>quantlab/ml_model/xgb.py (`DEFAULT_PARAMS` at 145, `_to_rows` at 219, `_fit_model` at 228-292), tests/test_xgb_model.py (fixtures `ArrayPanel`/`_panels`/`_config`/`_train`/`FakeRecorder`/`recorders` at lines 67-213, and the appended-section banner style at lines 568, 637, 669)</read_first>
  <behavior>
    New tests, appended to tests/test_xgb_model.py under a dated section banner in the style already used at lines 568/637/669 (this file's house pattern for scope additions — do NOT create a new test file):
    - ORACLE: the user's reference function, pasted verbatim into the test file as `_reference_ccc_loss(y_true, y_pred)`, returns the same value as the shipped metric (`pytest.approx`, tight rtol) across several seeds and sizes, including (a) a case where prediction variance is tiny relative to the label's and (b) a case with a large mean offset. The oracle lives only in the test file, never in production code.
    - CRITERION: on a run with `early_stopping=True` and a noise label, `booster.best_iteration` equals the argmin of the per-round `val-ccc_loss` series recovered from the `recorders` fixture, and `booster.best_score` equals that minimum. To keep it non-vacuous, the same test asserts the argmin of the `val-rmse` series is a DIFFERENT round — pick and pin a seed where they diverge, and state in the test docstring that the divergence assertion is what stops the test passing vacuously.
  </behavior>
  <action>
Add to quantlab/ml_model/xgb.py, at module level beside `_PARAM_ALIASES`/`_IMPORTANCE_TYPES`:

1. A pure numeric helper computing the pooled concordance correlation coefficient loss from two 1-D arrays, per D-02. Use `np.var`/`np.mean` population moments exactly as the reference does (`np.var` default ddof=0 — do not switch to ddof=1, that changes the value). Return `float`.
   Guards (authorised by D-04, must not change the value on finite non-degenerate input):
   - restrict to the jointly finite positions first, following the house convention in `quantlab/utils/metrics.py:_joint`;
   - fewer than one surviving pair, or a denominator of exactly 0.0, returns `1.0` — the worst possible loss. Returning NaN would make every `EarlyStopping` comparison false, and returning 0.0 would let a fully degenerate round be recorded as the best one. Say that in the docstring.
2. An adapter matching xgboost 3.4.1's `custom_metric` contract: `(predt, dtrain) -> tuple[str, float]`, returning the name `"ccc_loss"` and the helper's value.
   Multi-label: `dtrain.get_label()` returns one flat vector for a multi-output DMatrix. BEFORE writing the reshape, measure the actual layout with a throwaway 3-row / 2-label `xgb.DMatrix` in a scratch interpreter and record the observed layout in the adapter's docstring. Then reshape labels to `(dtrain.num_row(), -1)` and `predt` to the same shape, and score the PRIMARY label (column 0) only — the same convention `MLModel._compute_metrics` already uses (`pred[..., 0]`), and already stated in this class's 多标签 docstring block. Raise `ValueError` naming both sizes if the label and prediction element counts disagree; do not silently broadcast.
   For the single-label case this project uses, column 0 IS the whole vector, so the value is identical to the reference.

3. In `_fit_model`, pass the adapter as `custom_metric=` to `xgb.train`. Unconditional, per D-05 — not gated on `config.early_stopping`, so the ccc curve is recorded on every run exactly as the rmse curve is. Leave `DEFAULT_PARAMS["eval_metric"]` at rmse, leave the `EarlyStopping(...)` construction exactly as it is (no `metric_name`, no `maximize`), and leave the callback ordering comment at lines 255-257 untouched.

Then update the one existing assertion this breaks: in `test_hyperparameters_pass_through`, the exact-equality check on the first log row's key set now has two more keys. Assert the full new expected set explicitly (the two mae keys plus the two ccc keys) — keep it an `==`, do not weaken it to a subset, so a future metric-name change still turns it red.

If any of the LEARNING assertions (`test_learns_a_factor_driven_label`, `test_unrelated_label_gives_no_ic`, `test_multi_label_predicts_every_label`, `test_train_cv_sequential`'s `cv_mean_test_ic > 0.3`) goes red, STOP and report it in the SUMMARY. That would be real information about the criterion change — never loosen a threshold to absorb it.
  </action>
  <verify>
    <automated>uv run pytest tests/test_xgb_model.py -q</automated>
  </verify>
  <done>The suite is green at 31 + the new tests (baseline was 31). `booster.best_iteration` provably tracks the validation ccc_loss series and provably does not track the validation rmse series on the pinned seed.</done>
  <reversibility rating="reversible">Deleting one `custom_metric=` kwarg restores the previous behaviour exactly; nothing is persisted in a new format.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Lock the curve keys, the loss direction, and the degenerate cases</name>
  <files>tests/test_xgb_model.py</files>
  <read_first>quantlab/ml_model/xgb.py (`_WandbEvalCallback` at 33-58), tests/test_xgb_model.py (the section added in Task 1)</read_first>
  <behavior>
    Appended to the same section:
    - CURVES: after a run with a validation segment, every per-round row recorded through the `recorders` fixture carries both `train-ccc_loss` and `val-ccc_loss`. This is the proof that `_WandbEvalCallback` needed zero changes — say so in the test docstring. If it fails, report it; do not edit the callback without saying why in the SUMMARY.
    - DIRECTION: a prediction that tracks the target well returns a strictly SMALLER value than a poorly-tracking prediction on the same target (lower is better), and a perfect prediction returns approximately 0.0.
    - DEGENERATE (must not raise, and must not emit a numpy RuntimeWarning): constant predictions against a varying target returns 1.0 (covariance 0 gives ccc 0); a single row with differing values returns 1.0; the fully undefined case (both vectors constant and equal, so the denominator is 0) returns 1.0; an all-NaN prediction vector returns 1.0.
    - MULTI-LABEL ORIENTATION: build a two-label `xgb.DMatrix` directly, call the adapter with a known `(n, 2)` prediction array, and assert the returned value equals the oracle computed on column 0 of both. A transposed or column-major reshape produces a different number, so this is what pins the layout measured in Task 1.
  </behavior>
  <action>
Add the four test groups described in `<behavior>` to the section opened in Task 1. Call the production helper and the adapter directly for the pure cases — no training run is needed for the direction, degenerate or orientation tests, and a direct call is what keeps them fast and unambiguous about which function is under test.

Wrap the degenerate assertions in `pytest.warns(None)`-equivalent strictness by running them under `np.errstate(invalid="raise", divide="raise")`, so a 0/0 that happens to produce NaN silently cannot pass — `quantlab/utils/metrics.py` already holds itself to "no RuntimeWarning on the empty case" and this metric is held to the same bar.

Keep the multi-label DMatrix construction offline and synthetic, consistent with the rest of this file.
  </action>
  <verify>
    <automated>uv run pytest tests/test_xgb_model.py -q</automated>
  </verify>
  <done>Curve keys, loss direction, all four degenerate inputs and the multi-label column-0 orientation are each covered by an assertion that fails if the behaviour changes. `_WandbEvalCallback` is unmodified.</done>
</task>

<task type="auto">
  <name>Task 3: Correct every statement that names RMSE as the criterion, and record the pooled trade-off</name>
  <files>quantlab/ml_model/xgb.py, example/model.md</files>
  <read_first>quantlab/ml_model/xgb.py (class docstring 61-143, especially the 早停 block at 69-79 and the wandb block at 96-108), example/model.md (line 486, line 937, and the 早停判据 section at 987-1001)</read_first>
  <action>
Docstrings in `quantlab/ml_model/xgb.py` are Chinese — match that. Do not touch the English test docstrings.

In quantlab/ml_model/xgb.py:
1. The 早停 block (lines 69-79): the bullet at line 74 asserting which metric is the criterion is now false. Replace it with the CCC criterion: the pooled concordance correlation coefficient loss (`1 - ccc`) on the validation set, computed by this module's adapter, registered through `xgb.train(custom_metric=...)`; the built-in `eval_metric` stays first in the list and therefore stays an observation curve only, while `EarlyStopping(metric_name=None)` resolves to the LAST entry, which is the custom metric. Say that it is a loss, lower-is-better, which is why `maximize` is not passed.
2. Same block: state that `best_score` in the W&B summary is now a CCC loss and is NOT comparable to a number recorded by any run predating this change.
3. The wandb block (lines 96-99): the per-round curve list gains the two ccc keys beside the existing rmse ones.
4. On the new helper (or the class 早停 block — pick one place and cross-reference it from the other), record the trade-off as a deliberate user decision, not an oversight: the pooled form is NOT a cross-sectional metric — it is computed over all (t, s) rows at once, so it still rewards predicting the per-date market-wide component, and its denominator penalises a correctly shrunk prediction (in low-signal data the optimal prediction's variance sits far below the label's). State plainly that the pooled form was chosen by the user. Per D-03, do not propose, sketch or leave a TODO for a per-date variant.

In example/model.md:
5. Line 937: the `eval_metric` row's 说明 cell currently claims two roles; narrow it to the per-round curve role alone and point at the 早停判据 section.
6. Lines 992-993: replace that bullet with the CCC criterion, matching what the docstring now says, and keep the neighbouring true statements (patience counted in boosting rounds; the last-metric resolution rule at 994-995; the last-step-is-`best_iteration + patience` rule; the no-validation-segment warning) intact.
7. Lines 999-1001 currently describe `custom_metric` as unimplemented. That is now false — it IS wired, for this metric. Rewrite it as a factual statement of what is present and what is absent: the custom metric hook is in use; a cross-sectional IC criterion is still not implemented because per-row timestamp grouping is not threaded into the metric. Keep it descriptive; do not turn it into a proposal, and per D-03 do not mention a per-date CCC at all.
8. Line 486: the XGB row's per-round curve cell gains the two ccc keys.
9. Leave the pasted `resolved_hyperparameters` outputs at lines 982 and 852 alone — `eval_metric` really is still rmse in `DEFAULT_PARAMS`, so those outputs remain accurate. `example/README.md` states that pasted outputs in this directory were really run; do not hand-edit one.
  </action>
  <verify>
    <automated>! grep -rn "默认 RMSE\|逐轮曲线与早停判据" quantlab/ml_model/xgb.py example/model.md && grep -c "ccc_loss" quantlab/ml_model/xgb.py example/model.md && uv run pytest tests/test_xgb_model.py -q</automated>
  </verify>
  <done>No file tells a reader the criterion is RMSE; the pooled-vs-cross-sectional trade-off is recorded as a user decision in both the code and example/model.md; the suite is still green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| none crossed | Pure in-process numeric code on already-loaded synthetic/local panels. No network, no new dependency, no package install, no credential, no file format change. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-weq-01 | Tampering | `xgb.train(custom_metric=)` metric ordering | medium | mitigate | If a later change reorders metrics or drops the custom one, early stopping silently reverts to RMSE and the feature becomes inert with a green suite. Task 1's CRITERION test asserts `best_iteration` tracks the ccc series AND diverges from the rmse series. |
| T-weq-02 | Information Disclosure | `best_score` in the W&B summary | low | mitigate | The number changes meaning without changing name or type, so historical run comparisons would be quietly wrong. Task 3 records the discontinuity in the docstring. |
| T-weq-03 | Denial of Service | degenerate metric input during training | low | mitigate | A raise or NaN inside `custom_metric` aborts training or disables early stopping mid-run. Task 2 locks four degenerate inputs to a finite 1.0 under strict `np.errstate`. |
| T-weq-SC | Tampering | npm/pip/cargo installs | n/a | accept | No package installs in this plan; no `uv add`, no dependency change. |
</threat_model>

<verification>
- `uv run pytest tests/test_xgb_model.py -q` — green, count strictly above the 31 baseline.
- `uv run pytest tests/test_ml_models.py tests/test_model_layer.py tests/test_model_cv.py tests/test_model_predict_panel.py tests/test_ml_backend.py tests/test_model_hierarchy.py tests/test_config_roundtrip.py -q` — green; these are the model-layer neighbours that touch `XGBoostRegressor` for orchestration, persistence and config round-trip. Record the counts in the SUMMARY.
- `git diff --stat` touches exactly the three files in `files_modified`.
</verification>

<success_criteria>
- Early stopping demonstrably selects the round minimising validation pooled CCC loss, with the divergence-from-rmse arm proving the assertion is not vacuous.
- The shipped metric equals the user's verbatim reference on randomised non-degenerate input, checked by an oracle that lives only in the test file.
- `train-ccc_loss` / `val-ccc_loss` reach the W&B rows with `_WandbEvalCallback` unmodified.
- `eval_metric: "rmse"` still in `DEFAULT_PARAMS`, rmse curves still recorded.
- No per-date / cross-sectional CCC variant, switch, TODO or proposal exists anywhere in the diff (D-03).
- No config field was added (D-05).
</success_criteria>

<output>
Create `.planning/quick/260915-weq-use-a-ccc-based-custom-metric-as-the-xgb/260915-weq-SUMMARY.md` when done, recording: the measured multi-output `get_label()` layout, the pinned seed for the criterion test and why that seed, the final test counts, and any planning fact that turned out wrong.
</output>
