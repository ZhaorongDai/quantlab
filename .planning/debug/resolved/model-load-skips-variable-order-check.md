---
status: resolved
trigger: "G-03.7-9 model-load-skips-variable-order-check: BaseModel.load() never checks the checkpoint's recorded factor/label variable names and order, so predict_panel outside the backtester can silently feed permuted inputs."
created: 2026-09-15T12:40:00Z
updated: 2026-09-15T20:12:13Z
---

## Current Focus

hypothesis: CONFIRMED. BaseModel.load() restores a checkpoint without comparing the training record trained_on.factor_names / label_names with the model's get_factor_names() / get_label_names(). Neither head validates names by itself: the XGBoost Booster is trained without feature_names and predicts through inplace_predict(numpy), which checks only the column count, and a torch state_dict checks only tensor shape. The only name check sits outside the model layer, in BaseBacktester._assert_checkpoint_variables, and it compares against the wrong record: factors[].factor_names from the factor config, not trained_on. Measured, it both refuses a correct checkpoint (false positive) and passes a permuted one (false negative).
test: done (scratchpad repro_model_load.py, repro_false_negative.py, xgb_names_probe.py)
expecting: n/a
next_action: return ROOT CAUSE FOUND to the orchestrator (goal: find_root_cause_only); plan-phase --gaps plans the fix
bug_class: Bohrbug (deterministic code-path omission; reproduced every run)

reasoning_checkpoint:
  hypothesis: "Permuted or different factor/label lists pass BaseModel.load() + predict_panel silently. load() never reads trained_on.factor_names/label_names, and neither xgboost (nameless Booster, numpy inplace_predict) nor torch (shape-only load_state_dict) validates variable identity."
  confirming_evidence:
    - "Repro: XGBoostRegressor trained on [f_signal,f_second,f_noise] -> [ret_30,ret_60]. A fresh model with factors [f_noise,f_second,f_signal] runs load() and predict_panel with no error; ret_30 corr with correct prediction +0.046."
    - "Repro: XGBoostRegressor with reordered labels [ret_60,ret_30] loads and predicts silently; both labels corr -0.017 (outputs mislabeled)."
    - "Repro: LinearDLHead with reordered factors loads and predicts silently (ret_30 corr +0.021); reordered labels also silent (corr -0.015)."
    - "trained booster.feature_names is None; xgb_names_probe shows inplace_predict(numpy) with swapped columns PASSES and returns different predictions."
    - "grep: no quantlab code reads trained_on.factor_names/label_names; load() reads only trained_on.symbols (model.py:336,340-350)."
  falsification_test: "If load() or predict_panel raised on any reordered-factor or reordered-label case for either head, the hypothesis would be wrong. None raised."
  fix_rationale: "Put the name/order check in BaseModel.load(), which every load path goes through (direct, backtester run(), run_cv folds). Make trained_on its source, because trained_on is written by the same get_factor_names()/get_label_names() calls that built the training arrays."
  blind_spots: "Not tested on real KunQuant/Polars factor stores end to end, only name enumeration. Not tested RNN/MLP heads specifically; the positional-encoding argument and the LinearDLHead repro cover them. Did not run the full pytest suite, since no code was changed."
  candidate_causes:
    - "code: load() lacks the check (model.py:312-338); the check lives only in the backtester (backtest.py:874-904)"
    - "data/record: the backtester compares the factor CONFIG field factors[].factor_names instead of trained_on.factor_names, and the two diverge (user-supplied config.factor_names, FactorPolars read path)"
    - "environment: a factor's derived names can drift after training (KunQuant upgrade changing Alpha101.all_alpha, an edited Polars graph), so a stale config field can equal the drifted names"
    - "code (library use): xgb.DMatrix built without feature_names (xgb.py:227,234) plus inplace_predict on numpy (xgb.py:283) give no library-level safety net"
  and_gate: "yes. Silent permutation outside the backtester needs load() to lack a record check AND the head to lack intrinsic name validation. Inside the backtester, a separate contributing cause is the wrong record source (config field instead of trained_on). root_cause is a set."

## Symptoms

expected: BaseModel.load() refuses a checkpoint whose recorded factor/label variable names or order differ from the model config, so predict_panel outside the backtester cannot silently feed permuted inputs.
actual: BaseModel._save_model writes config.json with get_config() plus trained_on = {factor_names, label_names, symbols}. Nothing ever reads trained_on.factor_names / label_names: load() only reads trained_on.symbols, via _read_trained_symbols. The only name/order check is the backtester's BaseBacktester._assert_checkpoint_variables. It reads the CONFIG field factors[i].factor_names via _saved_variable_names, not the trained_on record, and it runs only inside the backtester's _load_model_checkpoint. BaseModel.get_factor_names() derives names from factor._get_factor_names(). XGBoost builds its DMatrix without feature_names, so it checks only the feature COUNT. The orchestrator has NOT run a repro; this was inferred from code.
errors: None — silent (inferred).
reproduction: Train a model with the synthetic helpers in tests/test_model_predict_panel.py or tests/backtest_fixtures.py. Build a fresh model whose factor list or order differs, call load(checkpoint), then predict_panel. Check whether it raises or silently predicts on permuted inputs, for XGBoostRegressor and for a DL head. Run with OMP_NUM_THREADS=1 WANDB_MODE=disabled uv run python <script>.
started: Discovered during UAT of phase 03.7 (2026-09-15), after code review fixes WR-01 (backtester variable check) and WR-02 (trained_on record).

## Eliminated

- hypothesis: xgboost itself rejects reordered features at predict time
  evidence: trained XGBoostRegressor booster.feature_names is None (repro). inplace_predict on numpy validates only num_features (xgboost core.py:2691-2701). Swapped numpy columns pass and give different predictions (xgb_names_probe).
  timestamp: 2026-09-15T12:52:00Z

- hypothesis: torch load_state_dict catches a different factor order on DL heads
  evidence: reordered factors keep the same weight shape, so LinearDLHead loads and predicts silently. Only a different factor COUNT fails, and then with a raw torch "size mismatch for weight" RuntimeError, not a named error (repro).
  timestamp: 2026-09-15T12:52:00Z

- hypothesis: predict_panel's missing-factor guard (model.py:376-383) covers it
  evidence: it checks only that each name in the fresh model's own get_factor_names() is present in the feature panel. It never compares with what training used, so reordered and relabeled cases pass (repro).
  timestamp: 2026-09-15T12:53:00Z

- hypothesis: the backtester's _assert_checkpoint_variables is an equivalent substitute for a trained_on check
  evidence: case 1, config.factor_names [f_noise,f_signal,f_second] and derived [f_signal,f_second,f_noise]: the check REFUSES an identical model loading its own checkpoint (false positive). Case 2, a backtest model whose derived names drifted to equal that stale config field: the check PASSES, and predict_panel runs on permuted inputs (ret_30 corr +0.020, ret_60 corr -0.038). A trained_on comparison would catch it (repro_false_negative).
  timestamp: 2026-09-15T12:55:00Z

## Evidence

- timestamp: 2026-09-15T12:45:00Z
  checked: Phase 0 knowledge base (.planning/debug/knowledge-base.md)
  found: No knowledge-base.md exists in the worktree or in the main repo .planning/debug; the only other file there is an unrelated active session, run-cv-spurious-checkpoint-date-warning.md. MemPalace was not queried.
  implication: No known-pattern candidate.

- timestamp: 2026-09-15T12:46:00Z
  checked: quantlab/base/model.py load() :312-338, _read_trained_symbols :340-350, _save_model :282-310
  found: _save_model writes trained_on = {factor_names: get_factor_names(), label_names: get_label_names(), symbols}. load() checks the suffix, sets `self._trained_symbols = self._read_trained_symbols(p)`, then calls `_read_checkpoint(p)`. _read_trained_symbols reads only record["symbols"]. No quantlab code reads trained_on.factor_names or label_names; utils/module.py:110 only pops the key.
  implication: The model layer never validates variable names or order at load time.

- timestamp: 2026-09-15T12:47:00Z
  checked: quantlab/base/backtest.py _load_model_checkpoint :809-838, _saved_variable_names :862-872, _assert_checkpoint_variables :874-904
  found: This is the only name/order check. It reads the sidecar CONFIG fields saved["factors"][i]["factor_names"] and saved["labels"][i]["factor_names"], which are each factor's config.factor_names at save time (Factor.get_config -> config.to_dict()). It compares them with model.get_factor_names() / get_label_names() and runs before collect for DL and before model.load(). With no record it warns and skips. It never reads trained_on.
  implication: Only backtester callers (run(), run_cv()) are protected. A direct model.load() + predict/predict_panel is not: train_model.py:58, example/model.md:803/1196, notebooks.

- timestamp: 2026-09-15T12:48:00Z
  checked: quantlab/base/model.py get_factor_names :236-248 vs quantlab/base/factor.py Factor.get_factor_names :186, _maybe_resolve_factor_names :72-77, FactorPolars.cal :394-402, Alpha158Stock._get_func_stream :143-157
  found: BaseModel.get_factor_names() chains factor._get_factor_names(), the class-derived list in graph order. Factor.get_factor_names() returns factor.config.factor_names, which the user can supply and which is auto-filled from _get_factor_names() only when None. FactorPolars.cal() overwrites config.factor_names with the real output columns; read() does not. KunQuant Alpha158 emits only the names in config.factor_names, but in GRAPH order.
  implication: (a) config.factor_names, which the backtester compares, and _get_factor_names(), which training fed and trained_on records, are two sources. They coincide only when config.factor_names was left None or matches the derived list. trained_on is authoritative: _save_model writes it with the same get_factor_names()/get_label_names() calls _fit used to build the arrays.

- timestamp: 2026-09-15T12:49:00Z
  checked: quantlab/ml_model/xgb.py _fit_model :215-279 and _forward :281-286; quantlab/ml_model/backend.py
  found: dtrain and dval are xgb.DMatrix(x_rows, label=y_rows) without feature_names (:227, :234). _forward does NOT build a DMatrix; it calls self.model.inplace_predict on a numpy (T*S, F) array. MlBackend pickles the Booster with joblib.dump.
  implication: The trained Booster has no feature names, so prediction can check only the column count.

- timestamp: 2026-09-15T12:50:00Z
  checked: all load( callers. Tests: test_xgb_model.py:455,529,558; test_ml_models.py:357; test_model_cv.py:475; test_model_predict_panel.py:426,454; test_backtest_dates.py:574 (via backtester). Also train_model.py:58, quantlab/base/backtest.py:837, utils/module.py load_model_from_config (does not call load; pops trained_on).
  found: Every direct test caller builds the fresh model from the same panels and config as the trained one, with the same names in the same order. test_backtest_dates.py:574 expects ValueError match "was trained on", the checkpoint path in the message, and `computed == []` (no feature collection before the refusal). test_backtest_dates.py:650-673 unlinks config.json and asserts EXACTLY ONE warning containing "has no config.json". No test edits a sidecar's names. train_model.py:58 loads a 2025 RNNClassifier checkpoint that predates trained_on.
  implication: (c) A load()-level check breaks no current test, under three conditions. (1) If the backtester delegates, the message keeps "was trained on" plus the path. (2) The check can also run BEFORE DL feature collection: _load_model_checkpoint collects before calling load(), so it needs a pre-load entry point, not only a call inside load(). (3) A missing-sidecar warning is emitted once, or with wording that does not contain "has no config.json".

- timestamp: 2026-09-15T12:52:00Z
  checked: scratchpad repro_model_load.py (worktree code confirmed via quantlab.__file__; OMP_NUM_THREADS=1, WANDB_MODE=disabled)
  found: |
    XGBoostRegressor, trained on factors [f_signal,f_second,f_noise] and labels [ret_30,ret_60] (booster.feature_names None):
      same-order fresh load -> identical predictions
      reordered factors -> load PASS, predict_panel PASS, ret_30 corr +0.046, ret_60 corr +0.982 (silent)
      reordered labels  -> load PASS, predict_panel PASS, both labels corr -0.017 (silent mislabel)
      2 factors instead of 3 -> load PASS, predict_panel RAISE "Feature shape mismatch, expected: 3, got 2"
    LinearDLHead:
      reordered factors -> load PASS, predict_panel PASS, ret_30 corr +0.021 (silent)
      reordered labels  -> load PASS, predict_panel PASS, corr -0.015 (silent)
      2 factors -> load RAISE RuntimeError "size mismatch for weight ... [2, 3] ... [2, 2]" (unnamed)
    BaseBacktester._assert_checkpoint_variables with reordered factors -> RAISE (backtester path protected for this simple case)
  implication: G-03.7-9 is confirmed for both heads and for labels as well as factors. The symptom needs no permutation of the data, only a different declared order.

- timestamp: 2026-09-15T12:55:00Z
  checked: scratchpad repro_model_load.py section (a) and repro_false_negative.py
  found: A factor with config.factor_names [f_noise,f_signal,f_second] and derived names [f_signal,f_second,f_noise] saves factors[0].factor_names = [f_noise,f_signal,f_second] and trained_on.factor_names = [f_signal,f_second,f_noise]. The backtester check REFUSES an identical model against its own checkpoint (false positive). A backtest model whose derived names drifted to [f_noise,f_signal,f_second] PASSES the backtester check, then predict_panel predicts on permuted inputs: ret_30 corr +0.020, ret_60 corr -0.038 (false negative). `trained_on != get_factor_names()` is True, so a trained_on check catches it.
  implication: (a) Measured: the config field and trained_on disagree whenever config.factor_names differs from the derived list or the derived list drifts. trained_on is authoritative. (d) The backtester should delegate to one model-level check keyed on trained_on.

- timestamp: 2026-09-15T12:53:00Z
  checked: scratchpad xgb_names_probe.py on xgboost 3.4.1; xgboost core.py _validate_features :3292-3325, inplace_predict :2684-2701, DMatrix.feature_names setter :1298/:1307
  found: |
    Booster trained with DMatrix(feature_names=[a,b]) plus EarlyStopping(save_best=True): the sliced booster keeps feature_names [a,b] (24 rounds, best_iteration 23).
    Booster.predict(DMatrix):
      names [a,b] -> PASS
      names [b,a] -> RAISE "feature_names mismatch: ['a', 'b'] ['b', 'a']"
      names [b,a] with validate_features=False -> PASS
      no names -> RAISE "data did not contain feature names, but the following fields are expected: a, b"
      names [a,c] -> RAISE "feature_names mismatch ... expected b in input data | training data did not have the following fields: c"
      3 columns [a,b,c] -> RAISE mismatch
    inplace_predict:
      numpy -> PASS
      numpy with SWAPPED columns -> PASS (predictions differ)
      numpy with 3 columns -> RAISE "Feature shape mismatch"
      pandas [a,b] -> PASS; pandas [b,a] -> RAISE
    joblib dump/load (MlBackend path): loaded.feature_names [a,b], best_iteration 23; predict DMatrix [b,a] -> RAISE; [a,b] -> PASS.
    OLD nameless booster: feature_names None; DMatrix [a,b] PASS, [b,a] PASS, no names PASS, pandas [b,a] PASS. _validate_features returns immediately when booster.feature_names is None (core.py:3293-3294).
    Name rules: '[', ']' and '<' -> RAISE "feature_names must be string, and may not contain [, ] or <"; duplicates -> RAISE "feature_names must be unique". 'a b', 'b/c', 'a(1)', 'b.c', '', 'a,b' accepted.
    get_score: keyed by name when set (gain {'a':1.54,'b':0.016}); nameless booster keys are f0/f1. Multi-output total_gain is aggregated across outputs ({'f_signal':818,'f_second':791,'f_noise':0.04}). Never-split features are OMITTED ('const' missing).
    Real names: Alpha101 has 82 (alpha001..), Alpha158Stock has 169 (KMID, KLEN, ROC5..). None contain [ ] <, no duplicates, no overlap, only [A-Za-z0-9_]. Polars/label names in repo: momentum_{n}, past_ret_{n}, fwd_ret_{n}, ret_{n}, ret_binary_{n}.
  implication: See the XGBoost feature_names addendum in Resolution.

- timestamp: 2026-09-15T12:56:00Z
  checked: W&B recording in quantlab/ml_model/xgb.py _WandbEvalCallback :29-54, _fit_model :273-279, MLModel._evaluate model.py:1221-1236; tests/test_xgb_model.py FakeRecorder :154-180 and :264-296; tests/test_ml_models.py:280-291
  found: Per-round curves go through recorder.log({"train-rmse","val-rmse"}, step=epoch). best_iteration/best_score go to the summary after xgb.train. Final `{split}_{metric}` values go to the summary in _evaluate. test_xgb_model.py:293-296 asserts `steps == list(range(len(steps)))` and that every logged row has train-rmse and val-rmse. test_ml_models.py:281 asserts the EXACT summary key set, but only for StubMLHead (base MLModel), not XGBoostRegressor.
  implication: Feature importance belongs in XGBoostRegressor._fit_model right after the best_iteration summary update, written with summary.update. A recorder.log(...) with no step, or a row without the rmse keys, would turn test_xgb_model.py:264 red. Putting it in MLModel._evaluate would turn test_ml_models.py:281 red.

## Resolution

root_cause: |
  (1) BaseModel.load() (quantlab/base/model.py:312-338) reads only trained_on.symbols and never compares trained_on.factor_names / label_names with get_factor_names() / get_label_names(). A model whose factor or label list is reordered or changed loads and predicts silently.
  (2) Neither head type validates variable identity by itself. XGBoostRegressor trains on nameless DMatrix (xgb.py:227,234) and predicts through inplace_predict(numpy) (xgb.py:283), which checks only the column count. DL heads restore a positional state_dict, which checks only shape.
  (3) The only name check, BaseBacktester._assert_checkpoint_variables (backtest.py:874-904), lives outside the model layer. It compares against the factor config field factors[].factor_names instead of trained_on.factor_names, and those can diverge (Factor.get_factor_names = config.factor_names vs BaseModel.get_factor_names = _get_factor_names()). The result is measured false positives and false negatives.
fix: ""
verification: "diagnose-only; repro scripts in scratchpad: repro_model_load.py, repro_false_negative.py, xgb_names_probe.py"
files_changed: []

answers:
  a_config_field_vs_trained_on: |
    They can disagree. Factor.get_factor_names() returns config.factor_names, which is user-suppliable and not rewritten on the FactorPolars read path. BaseModel.get_factor_names() returns _get_factor_names(). A factor library upgrade or an edited graph can also change _get_factor_names() after training. Both directions are measured: a correct checkpoint refused, and a permuted checkpoint accepted.
    trained_on.factor_names / label_names is authoritative. It is written by the same get_factor_names()/get_label_names() calls whose order to_array used to build the training arrays.
  b_old_checkpoints_policy: |
    Follow the WR-01/WR-02 policy:
    - trained_on present: strict ValueError naming both lists and the checkpoint path.
    - No trained_on, but the sidecar has factors[].factor_names (checkpoints trained before the WR-02 fix): fall back to that legacy config field. Log a warning that it is a weaker record, and raise on mismatch, which is what the backtester does today, so no protection is lost.
    - No sidecar or no names at all: one warning, then continue (load() is silent there today).
  c_breakage_risk: |
    No current test breaks when all callers build the fresh model from identical panels. Constraints on the fix:
    - test_backtest_dates.py:574 needs "was trained on", the checkpoint path, the declared names, and the refusal BEFORE any feature collection. The backtester must call the model-level check before DLModel's _collect_all_features, not only via load().
    - test_backtest_dates.py:650-673 needs exactly one warning containing "has no config.json", so avoid a duplicate from load().
    - load_model_from_config never calls load() and already pops trained_on, so it is unaffected. A rebuilt model's derived names equal trained_on unless the factor class drifted, and refusing then is correct.
    - train_cv never calls load(). run_cv loads each fold through _load_model_checkpoint, and every fold checkpoint carries its own trained_on.
    - train_model.py:58 (pre-WR-02 checkpoint) takes the legacy/warning path.
    - For DL, check before _read_checkpoint so a count mismatch gets the named ValueError instead of torch's RuntimeError.
  d_single_source: |
    Yes. Add one model-level check keyed on trained_on (legacy fallback inside it), e.g. BaseModel._assert_trained_variables(path) that reads the sidecar itself (no data needed). load() calls it; _load_model_checkpoint calls it (or model.load's pre-read) before feature collection. _assert_checkpoint_variables and _saved_variable_names then delegate or disappear.
  xgboost_feature_names_addendum: |
    (a) Passing feature_names=[str(n) for n in self.get_factor_names()] to the DMatrix at xgb.py:227/:234 makes the Booster carry names; they survive EarlyStopping(save_best=True) slicing. That alone does NOT make today's _forward validate: _forward (xgb.py:283) uses inplace_predict on numpy, which checks only the count, and swapped columns pass. To validate, _forward must either:
    - build xgb.DMatrix(rows, feature_names=names) and call Booster.predict (validate_features=True): [b,a] raises "feature_names mismatch", no names raises "data did not contain feature names", same count with another member raises; or
    - keep inplace_predict and compare self.model.feature_names with get_factor_names() explicitly (cheaper, clearer message).
    pandas inplace_predict also validates, but would bring a DataFrame into the layer (project constraint).
    Either way this validates the fresh CONFIG against the checkpoint, like trained_on, not the column order of a hand-built array passed to predict(x).
    (b) The names persist through MlBackend's joblib pickle, and a freshly loaded Booster still raises on [b,a]. Checkpoints written today are nameless; for those xgboost skips validation entirely (core.py:3293), so they load and predict unchanged (backward compatible) with no protection.
    (c) Importance goes in XGBoostRegressor._fit_model after xgb.train, beside the best_iteration/best_score summary update:
    - Use Booster.get_score(importance_type='gain'/'total_gain'/'weight'), keyed by factor name once DMatrix has names.
    - Zero-fill factors that never split; get_score omits them.
    - Multi-output importance is aggregated across labels.
    - Write it to recorder.summary (e.g. importance_gain/<factor>), or put a wandb.Table in the summary.
    - A recorder.log(...) call breaks test_xgb_model.py:293-296 (contiguous steps, rmse keys in every row) unless that test changes.
    - Name rules: no '[', ']', '<', no duplicates, str only. All 82 Alpha101 and 169 Alpha158Stock names and the repo's Polars/label names comply. Arbitrary FactorPolars column names could violate them and would then raise at train time.
    (d) This complements the load()-level trained_on check and does not replace it. It covers only XGBoost feature variables. It does not cover DL heads, labels (output channel order), other MLModel heads, or existing nameless checkpoints. It does protect XGBoost checkpoints that have no config.json beside them (hand-copied .joblib), and it adds importance keyed by factor name.
