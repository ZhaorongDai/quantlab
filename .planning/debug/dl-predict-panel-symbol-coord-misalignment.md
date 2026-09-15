---
status: diagnosed
trigger: "G-03.7-8 dl-predict-panel-symbol-coord-misalignment — DL predict_panel can put predictions on the wrong symbol coordinate when the recorded training symbol list is not in sorted order."
created: 2026-09-15T00:00:00Z
updated: 2026-09-15T00:45:00Z
---

## Current Focus

bug_class: bohrbug (deterministic; same record order -> same mislabel, every run)
known_pattern_candidate: none (no .planning/debug/knowledge-base.md exists)
hypothesis: CONFIRMED. The model layer has two definitions of "the training symbol layout" that coincide only when the recorded list is sorted. to_array sorts the symbol axis, both when DLModel._fit trains and when predict_panel predicts. _save_model records self.symbols in raw data_backend order, and predict_panel labels output coords with the record-ordered panel.
test: done (P1-P4 probes plus in-memory fix simulations; see Evidence)
expecting: n/a
next_action: return ROOT CAUSE FOUND to orchestrator (goal: find_root_cause_only); plan-phase --gaps plans the fix

reasoning_checkpoint:
  hypothesis: "DL predict_panel mislabels symbol coords because DLModel._align_prediction_symbols returns the panel in trained_on.symbols order (model.py:1045), predict_panel takes output coords from that panel (model.py:405), and to_array re-sorts the symbol axis before building the array (model.py:272). Coords follow the record, values follow sorted order. The record is unsorted whenever _save_model (model.py:299) runs on a backend that collect() did not sort, or when trained_on.symbols is written/edited out of order."
  confirming_evidence:
    - "P1: record ['S2','S0','S1'] -> all 3 coords wrong; record ['S1','S0','S2'] -> only S2 right; ['S2','S1','S0'] -> only S1 right (exactly the positions where record order == sorted order)"
    - "P2: train() and train_cv() on a backend filled by to_internal(unsorted) record ['S2','S0','S1'] while the x captured in _train_one_batch is in SORTED layout; load->predict_panel mislabels all 3"
    - "P3.5: inside predict_panel with an unsorted record, feats.symbol == record order but to_array layout == sorted layout"
    - "P4: an MLP trained by HEAD from an unsorted backend, then loaded: every coord wrong at HEAD; all right with a post-alignment re-sort (opt1/opt1c)"
  falsification_test: "If to_array did not re-sort the symbol axis, or if predict_panel's coords came from the sorted panel to_array consumes, an unsorted record would give correct per-coord values. opt1c (re-sort after alignment) and opt2 (no symbol sort in DL to_array) both produced correct per-coord values in P1, which confirms the mechanism."
  fix_rationale: "Symbol-sorted order is what every DL checkpoint was actually trained on: to_tensor/to_array has sorted the symbol axis since the first commit (6b3c603). So only the record's MEMBERSHIP is authoritative. Its order is always sorted(membership). Re-sorting after membership alignment makes coords == the to_array layout == the training layout, for new AND existing checkpoints. Recording sorted symbols makes the record a true statement about training."
  blind_spots: "Parallel train_cv not run separately (code-identical via copy.deepcopy -> _train_one_fold -> _save_model). GPU/CUDA not exercised (the order logic is device-independent). No third-party producer of trained_on records exists in the repo. MLP is the only position-sensitive head probed; the RNN heads' position sensitivity is established by code (GRU/LSTM recur across dim1=symbols) and by 03.7-REVIEW WR-02, not by a probe here."
  candidate_causes:
    - "code: predict_panel/DLModel._align_prediction_symbols take coords from the record-ordered panel while to_array re-sorts (coord/layout mismatch) -- CONFIRMED"
    - "code: _save_model records self.symbols in backend order, not the sorted layout _fit trained on; train()/train_cv() never normalize the backend -- CONFIRMED (contributing)"
    - "data: an unsorted symbol axis reaches the model backend (to_internal without collect(); XrBackend.filter_by_symbol keeps config.symbols order; combine_by_coords keeps a same-ordered unsorted axis) -- CONFIRMED as the trigger source"
    - "data/config: hand-edited or externally written config.json trained_on.symbols in non-sorted order -- CONFIRMED equivalent trigger (P1 sets the same state)"
    - "environment: numpy/xarray vs Python string sort disagreement -- ELIMINATED (P3.4)"
    - "code: head adapters (MLP/RNNRegressor/RNNClassifier) add a symbol reorder -- ELIMINATED (code read)"
  and_gate: "yes. Misalignment needs BOTH (a) coords taken from the record-ordered panel while to_array re-sorts, AND (b) an unsorted record. With a sorted record there is no symptom (P1 first line, and all 134 existing tests). With (a) removed, an unsorted record is harmless (opt1c: P1 and P4 all correct). (b) has two producers: _save_model recording backend order when collect() was skipped, and an externally edited record. Latent today because collect() sorts before every normal train and backtester train-mode path."

## Symptoms

expected: DL predict_panel puts every prediction on the symbol coordinate whose features produced it, regardless of the order of the recorded training symbol list.
actual: BaseModel.predict_panel (quantlab/base/model.py:362-408) calls self._align_prediction_symbols(features[factors].sortby([...])) (:385). DLModel._align_prediction_symbols (:1007-1045) returns feats.sel(symbol=trained) in trained-list order. self.to_array(feats, factors) (:388) then re-sorts by ["timestamp","symbol"] (:261-276), while the output coords take feats.symbol.values (:405), i.e. trained-list order. The orchestrator reproduced this with fresh._trained_symbols = ['S2','S0','S1'].
errors: None — silent misalignment.
reproduction: order_probe.py in scratchpad (imports tests.test_model_predict_panel helpers); run with OMP_NUM_THREADS=1 uv run python <path>.
started: Discovered during UAT of phase 03.7 (2026-09-15), after code review fix WR-02 added _align_prediction_symbols and the trained_on record.

## Eliminated

- hypothesis: numpy/xarray string sort order differs from Python sorted(str), so even a "sorted" record could disagree with to_array's layout
  evidence: P3.4 -- xarray sortby matches Python sorted() for U, object and numpy StringDType symbol coords, over tickers with dots, hyphens, mixed case, digits, non-ASCII and astral code points (both sort by code point)
  timestamp: 2026-09-15T00:27:00Z

- hypothesis: the RNNRegressor / RNNClassifier / MLPRegressor _predict_panel_array adapters add a further symbol reorder
  evidence: code read. MLP reshapes [T,S,F]->[T,S*F] and back [T,S*L]->[T,S,L] in C order (exact inverse, same as _train_one_batch). ModelRCrypto/ModelRBaseCrypto treat dim0=times as the batch and dim1=symbols as the sequence (batch_first=True), and reshape (D*T)->(D,T,·), preserving [T,S]. RNNClassifier reshapes [T,S,2L]->[T,S,L,2] and softmaxes the last axis. Every _preprocess is nan_to_num. None permutes the symbol axis. They are, however, position-SENSITIVE (P2d), so the correct layout is required, not cosmetic.
  timestamp: 2026-09-15T00:27:00Z

- hypothesis: the backtester's DL load path (_load_model_checkpoint filling data_backend with the unsorted _collect_all_features() panel) writes or reads an unsorted training record
  evidence: backtest.py:835-837 -- to_internal(_collect_all_features()) happens before load(). load() sets _trained_symbols only from the sidecar (model.py:336, 340-350). DLModel._read_checkpoint uses only len() of the backend when there is no record (model.py:1053-1057). That path never calls _save_model. With no sidecar, _trained_symbols is None and predict_panel keeps the sorted panel (correct).
  timestamp: 2026-09-15T00:15:00Z

- hypothesis: "preserve trained order end-to-end" (DL to_array must not re-sort the symbol axis) is the right contract
  evidence: P2 under opt2 -- DLModel._fit's training layout becomes backend order (a training-behaviour change). P4 under opt2 -- a HEAD-trained MLP checkpoint with an unsorted record gets ALL coords' values wrong, because the net is fed a layout it never trained on. Every existing checkpoint was trained symbol-sorted (to_tensor sorted since 6b3c603).
  timestamp: 2026-09-15T00:40:00Z

## Evidence

- timestamp: 2026-09-15T00:10:00Z
  checked: quantlab/base/model.py predict_panel (362-408), to_array (261-276), DLModel._align_prediction_symbols (1007-1045), _save_model (282-310), symbols (136-140), collect (226-234), _collect_all_features (206-224), DLModel._fit (863-1005), train_cv (626-717)
  found: |
    - predict_panel sorts the features first.
    - DLModel._align_prediction_symbols then returns feats.sel(symbol=trained) in trained-list order.
    - to_array then applies .sortby(["timestamp","symbol"]) again, but the output Dataset coords use feats.symbol.values (trained-list order).
    - _save_model records self.symbols in data_backend order. Only collect() sorts the backend; train()/train_cv() never call collect.
    - _collect_all_features does NOT sortby.
    - DLModel._fit builds its training tensors via to_tensor -> to_array, so the network is always TRAINED on the symbol-SORTED layout, whatever the backend order.
  implication: the recorded list and the true training layout diverge when the backend is unsorted; predict_panel labels coords with the record's order while values follow the sorted layout.

- timestamp: 2026-09-15T00:12:00Z
  checked: git history (git log -S 'sortby(["timestamp", "symbol"' ; git show 6b3c603:base/model.py)
  found: the repo's first commit (6b3c603) already sorted in to_tensor (.sortby(["timestamp","symbol","variable"])) and in collect. 905de76 kept the symbol sort while fixing the variable axis. _align_prediction_symbols and trained_on were introduced by 554e69b (the WR-02 fix).
  implication: every DL checkpoint DLModel._fit has ever produced was trained on a symbol-sorted layout. The symbol sort in to_array is the long-standing contract; the trained-list-order selection is the newcomer.

- timestamp: 2026-09-15T00:14:00Z
  checked: 03.7-REVIEW.md WR-02, 03.7-REVIEW-FIX.md WR-02, example/model.md:147,159, tests/test_backtest_dates.py:27-31,391-411, tests/test_model_predict_panel.py:489
  found: |
    - REVIEW asked to "reindex the features onto the training symbols before prediction" (membership/position).
    - REVIEW-FIX says the hook "selects the training symbols in their training order".
    - example/model.md:147 promises coords come from the sorted panel fed to to_array; :159 promises "输出的 symbol 坐标就是训练标的，顺序与训练时相同". The two claims only coincide when the record is sorted.
    - test_backtest_dates.py documents "Predictions always come back symbol-sorted from predict_panel".
    - test_checkpoint_config_json_with_the_training_record_rebuilds_the_model already asserts trained_on.symbols == sorted(model.symbols).
  implication: the intended, relied-on contract is symbol-sorted output; WR-02's "training order" wording silently assumed training order == sorted order.

- timestamp: 2026-09-15T00:15:00Z
  checked: quantlab/base/backtest.py _prepare_model (762-782), _load_model_checkpoint (809-838), _align_and_predict (976-986); quantlab/utils/module.py load_model_from_config (91-114); quantlab/base/factor.py _auto_filter (36-43); quantlab/dataset/backend.py XrBackend.to_internal/filter_by_symbol/get_xarray_dataset (1469-1540)
  found: |
    - Backtester train mode calls model.collect() then train(), so the record is sorted.
    - Load mode for DLModel does data_backend.to_internal(model._collect_all_features()) (NOT sorted), but load() takes _trained_symbols only from the sidecar, and _read_checkpoint uses only the COUNT of the backend when no record exists.
    - load_model_from_config pops trained_on.
    - filter_by_symbol does .sel(symbol=list(config.symbols)), i.e. config order.
    - get_xarray_dataset only transposes, never sorts. to_internal stores as-is.
  implication: the backtester paths never write an unsorted record themselves. Unsorted records come from train()/train_cv() on a backend filled without collect(), from a hand-edited or externally written trained_on.symbols, or from direct assignment of _trained_symbols.

- timestamp: 2026-09-15T00:25:00Z
  checked: P1 scratchpad/g8_p1_repro.py (orchestrator probe pinned to this worktree; quantlab import path asserted)
  found: record sorted -> values on right coords True. Record ['S2','S0','S1'] -> coords ['S2','S0','S1'], per-coord-correct all False. ['S1','S0','S2'] -> only S2 correct; ['S2','S1','S0'] -> only S1 correct (exactly the positions where record order == sorted order).
  implication: reproduced deterministically; every coord whose position differs between record order and sorted order is mislabeled.

- timestamp: 2026-09-15T00:26:00Z
  checked: P2 scratchpad/g8_p2_unsorted_training.py (train() and train_cv() on a backend filled by data_backend.to_internal(panel.sel(symbol=['S2','S0','S1'])), no collect(); x captured inside _train_one_batch)
  found: |
    - train(): trained_on.symbols = ['S2','S0','S1'], but the layout _fit fed the network = sorted. train()->load->predict_panel gives coords ['S2','S0','S1'], all per-coord False.
    - train_cv(): 2 folds; fold0 trained_on.symbols = ['S2','S0','S1']; training layout = sorted.
    - MLPRegressor's per-symbol prediction is NOT invariant to symbol layout (position-sensitive).
  implication: whenever the backend is unsorted, the record is a false statement about training. train() and train_cv() (sequential, and by deepcopy parallel) both write it. Because DL heads are position-sensitive, "preserve the recorded order end-to-end" would feed existing checkpoints a layout they were never trained on.

- timestamp: 2026-09-15T00:27:00Z
  checked: P3 scratchpad/g8_p3_order_sources.py
  found: |
    - xr.combine_by_coords keeps a single or same-ordered unsorted symbol axis (['S2','S0','S1']) and only sorts when inputs disagree.
    - _collect_all_features() returns ['S2','S0','S1'] for unsorted factors; collect() sorts to ['S0','S1','S2'].
    - XrBackend.filter_by_symbol(('NVDA','AAPL')) yields ['NVDA','AAPL'] (config.symbols order), so real factors/datasets with config.symbols produce unsorted axes.
    - xarray sortby == Python sorted(str) for U, object and StringDType coords over tricky tickers (dots, hyphens, case, digits, non-ASCII, astral).
    - Inside predict_panel with an unsorted record, feats.symbol == record order, while to_array's layout == sorted layout (not feats' layout).
  implication: numpy-vs-Python string sort is NOT a contributing cause (eliminated). Unsorted axes are easy to produce upstream (config.symbols order), and only collect() normalizes them for training. The backtester's load path puts an unsorted features panel into the DL backend, but load() never records from it.

- timestamp: 2026-09-15T00:28:00Z
  checked: baseline pytest at HEAD: tests/test_model_predict_panel.py test_dl_models.py test_model_hierarchy.py test_model_cv.py test_backtest_dates.py test_backtest_persistence.py test_backtest_rebuild.py test_backtest_contracts.py
  found: 134 passed
  implication: no existing test exercises an unsorted training record (all fixtures use sorted SYMBOLS and go through collect()).

- timestamp: 2026-09-15T00:35:00Z
  checked: in-memory fix simulations on the same 8 test files as baseline (scratchpad/g8_patches.py via pytest plugin g8_plugin.py and runner g8_run_patched.py; no source edits)
  found: |
    Options simulated:
    - opt1 = record sorted(symbols) in _save_model + re-sort the symbol axis after DLModel alignment
    - opt1c = only the re-sort after alignment
    - opt2 = DLModel.to_array does not sort the symbol axis (training and prediction both use backend/record order)
    Results:
    - Test suite: opt1 134 passed, opt1c 134 passed, opt2 134 passed.
    - P1 under opt1c: every unsorted record -> coords ['S0','S1','S2'], all per-coord True.
    - P1 under opt2: coords follow the record, all per-coord True. LinearDLHead is position-invariant, so P1 cannot see layout errors.
    - P2 under opt1: record ['S0','S1','S2'], layout sorted, predict_panel all True; train_cv fold record sorted.
    - P2 under opt2: record ['S2','S0','S1'] AND the training layout becomes 'unsorted S2,S0,S1' (opt2 changes what DLModel._fit trains on); predict_panel all True.
  implication: the existing suite cannot discriminate the options (it only uses sorted SYMBOLS). opt2 is a training-behaviour change for DL (the to_tensor/_fit layout now depends on backend order), not a predict_panel-local fix. The discriminating case is a checkpoint that ALREADY exists with an unsorted record and a position-sensitive head (P4).

- timestamp: 2026-09-15T00:40:00Z
  checked: P4 scratchpad/g8_p4_existing_checkpoint.py -- MLPRegressor trained by UNPATCHED HEAD from an unsorted backend (record ['S2','S0','S1']), then loaded in the same process after applying none/opt1c/opt1/opt2. Ground truth = the trained net applied to the sorted layout HEAD's _fit trained on, read per sorted symbol.
  found: none -> coords ['S2','S0','S1'], all per-coord False. opt1c -> coords ['S0','S1','S2'], all True. opt1 -> coords ['S0','S1','S2'], all True. opt2 -> coords ['S2','S0','S1'], all False (values themselves wrong: the MLP is fed a layout it never trained on).
  implication: the sorted invariant is the correct contract. Re-sorting after membership alignment fixes existing and new checkpoints with no training change. "Preserve trained order end-to-end" (opt2) corrupts VALUES for every existing position-sensitive checkpoint with an unsorted record, and changes DL training.

## Resolution

root_cause: |
  Two contributing causes (AND-gate):
  1. predict_panel labels output coords with the record-ordered panel while to_array lays values out symbol-sorted. DLModel._align_prediction_symbols (quantlab/base/model.py:1045) returns feats.sel(symbol=trained) in trained_on.symbols order, and predict_panel builds the output coords from that panel (model.py:405). But to_array (model.py:269-276) re-sorts the symbol axis before building the array, so the values are in sorted layout. That layout is the one DLModel._fit trains on (to_tensor -> to_array, sorted since 6b3c603). The hook's "network sees the training layout" claim (docstring at 1019) is false for any unsorted record.
  2. _save_model (model.py:299) records self.symbols in raw data_backend order instead of the sorted layout _fit actually trained on. Only collect() (model.py:232) sorts the backend; train() (447) and train_cv() (626, including parallel folds via deepcopy) accept a backend filled by to_internal without collect(). A hand-edited or externally written trained_on.symbols is an equivalent producer, because load() takes the order verbatim (model.py:340-350).
  With a sorted record the two orders coincide, which is why it is latent on every collect()-based path and in every existing test.
fix:
verification:
files_changed: []
