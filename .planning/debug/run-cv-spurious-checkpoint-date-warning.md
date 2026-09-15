---
status: diagnosed
trigger: "G-03.7-7 run-cv-spurious-checkpoint-date-warning — Every normal run_cv fold logs a spurious \"config.model says ... using the checkpoint's dates\" warning, although run_cv uses the manifest's dates and the checkpoint and manifest come from the same train_cv."
created: 2026-09-15T00:00:00Z
updated: 2026-09-15T00:30:00Z
goal: find_root_cause_only
---

## Current Focus

hypothesis: CONFIRMED. Two co-occurring causes. (1) run_cv reuses _checkpoint_train_bounds, whose job is the run()-load-mode comparison against config.model, so every fold is compared with config.model's single train window even though run_cv uses the manifest's dates. (2) Both date checks compare with tuple(map(str, ...)), so the same instant written as '2024-01-01' and as np.datetime_as_string's '2024-01-01T00:00:00.000000000' counts as a mismatch.
test: done (scratchpad run_cv_example.py: 8/8 folds warn; probe_load_mode_dates.py cases A-D plus an intraday slice check)
expecting: n/a
next_action: return ROOT CAUSE FOUND to orchestrator (diagnose-only; plan-phase --gaps plans the fix)
bug_class: bohrbug (deterministic; reproduces on every run with the same inputs)
known_pattern_candidate: none

reasoning_checkpoint:
  hypothesis: "run_cv warns on every fold because it calls _checkpoint_train_bounds (backtest.py:432), which compares the checkpoint's recorded train dates with config.model's (backtest.py:799). Fold i>0 genuinely trains on a different window from config.model's single window. Fold 0 has the same instants but different text (ns string vs plain ISO date), and the comparison is on strings. The warning says 'using the checkpoint's dates', but run_cv discards that and uses fold['_train_bounds'] (backtest.py:448)."
  confirming_evidence:
    - "Repro: 8 'using the checkpoint's dates' warnings from _checkpoint_train_bounds:800 and 0 'using the manifest's dates'. Fold 0: recorded 2024-01-01T00:00:00.000000000..2024-02-09T00:00:00.000000000 vs config.model '2024-01-01'..'2024-02-09' (same instants). Folds 1-7: recorded windows shift by 6 bars each (e.g. fold 1 2024-01-09..2024-02-19), so the instants really differ from config.model."
    - "model.py:514-525 formats fold dates with np.datetime_as_string. _train_one_fold (:540-541) assigns them to config.train_start/train_end, so _save_model's config.json and cv_folds.json both carry ns strings. The BaseModel.config setter (:98-118) does not normalize train dates."
    - "Probe case C: a train_cv fold-0 checkpoint loaded through run() with config.model typed as the same instants in plain dates -> 1 warning (same instants = True). Case B: a train() checkpoint trained with plain dates, config.model typed 'YYYY-MM-DDT00:00:00' -> 1 warning. Case A: plain dates on both sides -> 0. Case D: genuinely stale train_end -> 1 (sink is not vacuous)."
  falsification_test: "If run_cv did not call the config.model comparison, or the comparison were on timestamps, fold 0 would not warn. If folds 1-7 still warned under a timestamp comparison, cause (1) is independent of cause (2). Both observed: fold 0's instants equal config.model's; folds 1-7's differ."
  fix_rationale: "Remove the config.model comparison from run_cv (the manifest is D-16's authority there, and the checkpoint-vs-manifest check at :434 already covers checkpoint drift). Replace both string comparisons with one timestamp-based comparison helper, so format differences stop warning in run() load mode and in the manifest check."
  blind_spots: "A naive pd.Timestamp equality hides a real intraday difference: as a slice endpoint, '2024-02-09' includes the whole day, while '2024-02-09T00:00:00.000000000' stops at midnight. Probe: Timestamp-equal = True, but the slice end bars are 2024-02-09 16:30 vs 2024-02-08 16:30. In run() load mode the recorded dates are used regardless, so this only affects whether a warning appears, not numbers. Parallel train_cv (deepcopy branch) and DL heads were not exercised, but they go through the same _cv_folds/_train_one_fold/_save_model code."
  candidate_causes:
    - "code: run_cv calls _checkpoint_train_bounds, which compares against config.model (wrong premise for run_cv)"
    - "code: tuple(map(str, ...)) string equality for dates at backtest.py:799 and :434-435"
    - "data: date text format differs by producer. np.datetime_as_string ns strings (train_cv folds -> config.json/cv_folds.json) vs user-typed plain ISO dates (config.model), with no normalization in the model config setter"
    - "config: backtester config.model built with the train_cv trainer's overall dates, which never equal a per-fold window (eliminated as an independent cause: that is simply how run_cv is meant to be configured; the doc and the fixture do exactly this)"
  and_gate: "yes. The 8/8 count needs both conditions: cause (1) produces the folds 1-7 warnings on its own (different instants), and cause (2) plus the ns-vs-plain data format produce fold 0's warning and the run() load-mode false positives (probe B, C). Fixing only (2) leaves 7/8 run_cv warnings; fixing only (1) leaves the run() load-mode false positives and the latent :434 string check."

## Symptoms

expected: A normal run_cv (checkpoints and cv_folds.json produced by the same train_cv) logs no checkpoint-vs-config date-mismatch warning; date mismatch checks compare timestamps, not strings. The three WR-01 rules stay: factor/label name-or-order mismatch raises; recorded train dates differing from config.model warn and the checkpoint's dates are used (run()'s load mode); a checkpoint with no config.json warns and continues.
actual: 8-fold train_cv + run_cv on tests/backtest_fixtures synthetic panel logged 8 warnings from quantlab/base/backtest.py:_checkpoint_train_bounds:800 ("checkpoint ... was trained on 2024-01-01T00:00:00.000000000..2024-02-09T00:00:00.000000000 (its config.json), but config.model says train_start='2024-01-01' ..."), and 0 of the "using the manifest's dates" warning. Re-verification counted 24 across 3 tests in tests/test_backtest_run_cv.py. No numbers affected.
errors: None — log noise only.
reproduction: UAT test 7 in .planning/phases/03.7-.../03.7-UAT.md; scratchpad run_cv_example.py (OMP_NUM_THREADS=1 uv run python <path> 2>stderr.txt; grep "using the checkpoint's dates").
started: Discovered during UAT re-verification after code review fix WR-01 (2026-09-15).

## Eliminated

- hypothesis: train_cv writes different date text to each fold's config.json and to cv_folds.json, so the checkpoint-vs-manifest check (:434) misfires
  evidence: repro logged 0 "using the manifest's dates" warnings; both files are written from the same fold dict produced by np.datetime_as_string (model.py:514-525, :540-541, :557-569)
  timestamp: 2026-09-15T00:12:00Z

- hypothesis: the model config setter normalizes train_start/train_end, so config.json never matches what the user typed
  evidence: BaseModel.config setter (model.py:98-118) only defaults start_date/end_date; probe case A config.json records train_start='2024-01-01' verbatim and produces 0 warnings
  timestamp: 2026-09-15T00:25:00Z

- hypothesis: switching the comparison to timestamps alone fixes the run_cv symptom
  evidence: folds 1-7 record windows that really differ from config.model's single window (fold 1 2024-01-09..2024-02-19 vs 2024-01-01..2024-02-09), so they would still warn under a timestamp comparison
  timestamp: 2026-09-15T00:14:00Z

- hypothesis: run()'s load mode is free of the string-vs-timestamp false positive
  evidence: probe case B (train() checkpoint, plain dates vs 'T00:00:00' text) and case C (train_cv fold checkpoint, ns vs plain, same instants) each log 1 "using the checkpoint's dates" warning
  timestamp: 2026-09-15T00:25:00Z

- hypothesis: the spurious warning changes which training window is used (numbers affected)
  evidence: run_cv passes fold["_train_bounds"] (manifest) to _backtest_window (backtest.py:444-449) and ignores `recorded` except for the :434 check; run() returns `recorded` regardless of whether it warned (:807)
  timestamp: 2026-09-15T00:07:00Z

## Evidence

- timestamp: 2026-09-15T00:00:00Z
  checked: .planning/debug/ for knowledge-base.md
  found: directory had no knowledge base and no prior sessions
  implication: no known-pattern candidate; proceed with open investigation

- timestamp: 2026-09-15T00:05:00Z
  checked: quantlab/base/backtest.py run_cv (377-449), _prepare_model (762-782), _checkpoint_train_bounds (784-807)
  found: run_cv calls self._checkpoint_train_bounds(saved, fold["checkpoint"]) at :432 for every fold. That helper compares saved (train_start, train_end) against config.model's (train_start, train_end) with tuple(map(str, ...)) != tuple(map(str, ...)) at :799 and warns "using the checkpoint's dates" at :800. run_cv then keeps only `recorded` for a second str comparison against fold["_train_bounds"] (:434-443) and passes fold["_train_bounds"] (not recorded) to _backtest_window (:448). _prepare_model (run() load mode, :776) is the only other caller.
  implication: in run_cv the config.model comparison has no effect on which dates are used, yet it still warns. The warning's claim ("using the checkpoint's dates") is false in run_cv.

- timestamp: 2026-09-15T00:06:00Z
  checked: quantlab/base/model.py _cv_folds (470-528), _train_one_fold (530-569), _save_model (282-310), config setter (98-118), get_config (250-254); quantlab/base/config.py:138-171
  found: _cv_folds formats every fold date with np.datetime_as_string (ns strings like 2024-01-01T00:00:00.000000000). _train_one_fold assigns those strings to self.config.train_start/train_end before _fit, so _save_model's config.json (get_config -> config.to_dict) records ns strings. cv_folds.json gets the same fold dicts. The model config setter does NOT normalize train_start/train_end (only defaults start_date/end_date). MLConfig/DLConfig declare train_start: str | None.
  implication: config.json and cv_folds.json agree byte-for-byte (so :434 does not fire), but config.model built by a user carries whatever text was typed ('2024-01-01'), which never string-equals an ns string even for the same instant. For fold i>0 the instants also differ (each fold has its own train window), so even a timestamp comparison against config.model would still warn in run_cv.

- timestamp: 2026-09-15T00:07:00Z
  checked: grep -rn "map(str" quantlab/; grep for other train_start/train_end equality in backtest.py, quantlab/utils/module.py, quantlab/backtest/
  found: only two string-equality date comparisons exist: backtest.py:434-435 (run_cv checkpoint vs manifest) and backtest.py:799 (_checkpoint_train_bounds checkpoint vs config.model). No overrides of _checkpoint_train_bounds/_prepare_model/run_cv in quantlab/backtest/; module.py has no date comparison.
  implication: the complete set of call sites needing timestamp comparison is these two.

- timestamp: 2026-09-15T00:12:00Z
  checked: ran scratchpad run_cv_example.py (OMP_NUM_THREADS=1 uv run python), grep stderr
  found: exit 0; 8 "using the checkpoint's dates" warnings from _checkpoint_train_bounds:800, 0 "using the manifest's dates". Fold 0 line: recorded 2024-01-01T00:00:00.000000000..2024-02-09T00:00:00.000000000 vs config.model '2024-01-01','2024-02-09' (same instants, different text). Fold 1 line: recorded 2024-01-09T..2024-02-19T vs config.model '2024-01-01','2024-02-09' (genuinely different instants).
  implication: bug is deterministic (Bohrbug). Two contributing conditions: fold 0 warns ONLY because of string comparison; folds 1..7 warn because run_cv compares each fold against config.model's single window at all, which is a wrong premise in run_cv regardless of comparison type. A timestamp-only fix would still leave 7 of 8 warnings.

- timestamp: 2026-09-15T00:14:00Z
  checked: per-fold extraction of recorded vs config.model dates from the repro stderr
  found: fold 0 recorded 2024-01-01..2024-02-09 (equal instants); folds 1-7 recorded 2024-01-09..2024-02-19, 2024-01-17..2024-02-27, 2024-01-25..2024-03-06, 2024-02-02..2024-03-14, 2024-02-12..2024-03-22, 2024-02-20..2024-04-01, 2024-02-28..2024-04-09; config.model is always '2024-01-01'..'2024-02-09'
  implication: 1 of 8 warnings is format-only; 7 of 8 come from comparing against config.model at all.

- timestamp: 2026-09-15T00:25:00Z
  checked: scratchpad probe_load_mode_dates.py, run() with model_mode='load'
  found: A (train() ckpt, plain dates both sides; config.json records '2024-01-01'/'2024-02-02') -> 0 warnings. B (train() ckpt plain; config.model '2024-01-01T00:00:00'/'2024-02-02T00:00:00') -> 1 warning. C (train_cv fold-0 ckpt, ns strings; config.model plain dates, same instants=True) -> 1 warning. D (stale train_end control) -> 1 warning.
  implication: run() load mode has the same string-vs-timestamp false positive whenever the recorded and configured text differ in format; the common path of train() plus identical plain dates is unaffected. The WR-01 true positive (D) must survive the fix.

- timestamp: 2026-09-15T00:25:00Z
  checked: pure-pandas intraday slice check in the probe
  found: pd.Timestamp('2024-02-09') == pd.Timestamp('2024-02-09T00:00:00.000000000') is True, but DatetimeIndex.slice_indexer(None, '2024-02-09') ends at bar 2024-02-09 16:30 while slice_indexer(None, '2024-02-09T00:00:00.000000000') ends at 2024-02-08 16:30.
  implication: under the model layer's slice semantics (CR-01, _slice_bound), a plain date and ns midnight are NOT the same training end on intraday data. A naive Timestamp equality would silence a warning for a genuinely different config.model window on intraday bars (no numbers affected, because the recorded dates are used either way). The fix must choose between Timestamp equality and comparing the resolved bars on the price calendar.

- timestamp: 2026-09-15T00:28:00Z
  checked: git log -S on quantlab/base/backtest.py; git show --stat 8465f1a; grep WR-01 across tests/
  found: both the run_cv call to _checkpoint_train_bounds and both map(str) comparisons were introduced together in 8465f1a "fix(03.7): WR-01 check a loaded checkpoint against config.model and use its recorded train dates". That commit added tests only in tests/test_backtest_dates.py (run() load mode) and tests/test_backtest_metrics.py; tests/test_backtest_run_cv.py has a warning_messages fixture but asserts only on OVERLAP_WARNING. The run() WR-01 tests build the trainer and the backtest model with the same _day() formatter, so string and timestamp comparison agree there.
  implication: why not caught — no test asserts that a normal run_cv emits zero WR-01 warnings, and no test builds recorded and configured dates in different text formats.

## Resolution

root_cause: "Two contributing causes (AND-gate). (1) run_cv (quantlab/base/backtest.py:432) calls _checkpoint_train_bounds, which compares the checkpoint's recorded train dates with config.model's (:795-806) and warns 'using the checkpoint's dates'. run_cv's dates come from the manifest (:448), and each fold's window legitimately differs from config.model's single window, so folds 1..7 warn even with identical instants. (2) Both date checks (:799, and run_cv's checkpoint-vs-manifest check at :434-435) compare tuple(map(str, ...)). np.datetime_as_string ns strings from train_cv (model.py:514-525 -> config.json and cv_folds.json) never string-equal plain ISO dates typed into config.model, so the same instants still mismatch (fold 0; run() load mode probes B and C)."
fix:
verification:
files_changed: []
