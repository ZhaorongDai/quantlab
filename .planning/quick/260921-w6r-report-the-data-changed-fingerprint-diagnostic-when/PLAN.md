---
phase: quick-260921-w6r
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/base/backtest.py
  - tests/test_backtest_rebuild.py
  - .planning/phases/03.11-crsp-permno-symbol-axis-migration-and-tiingo-era-dead-code-r/deferred-items.md
autonomous: true
requirements: [D-03.11-UAT-A]

estimate:
  tokens: 70000
  raw_tokens: 35000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "A `run()` that raises inside the backtest window still tells the operator the data changed: the D-27 comparison runs on whatever fingerprints exist at raise time, marked as a PARTIAL comparison."
    - "The original exception is what propagates, byte for byte, in every failure path — including when the diagnostic itself raises."
    - "A partial comparison never invents a 'present in expected_fingerprint but not read by this run' warning: under a partial comparison that condition means 'not read YET'."
    - "A successful run emits exactly the fingerprint warnings it emits today — same count, same text, no PARTIAL marker."
    - "`run_cv()` has the same failure-path diagnostic, both for a fold's window and for the stitched recording."
  artifacts:
    - quantlab/base/backtest.py
    - tests/test_backtest_rebuild.py
  key_links:
    - "`run()` / `run_cv()` exception path -> `_compare_fingerprints_on_failure` -> `_compare_fingerprints(partial=True)`"
    - "`_compare_fingerprints(partial=True)` -> skips the 'not read by this run' branch, keeps every other branch"
---

<objective>
`BaseBacktester.run()` records the D-27 data fingerprints INSIDE `_backtest_window`
(`_align_and_predict` -> `_redate_factors` -> `_record_factor_fingerprints` before
`model.predict_panel`, `_load_prices` after it) and only compares them AFTER that call
returns. So anything raising in between leaves the comparison unexecuted although its
inputs exist and already differ. `run_cv()` has the same shape. The operator gets the
downstream error and no indication that the data changed — exactly what D-27 exists to
say. Measured during 03.11 UAT (see `reproduction.py` in this directory): control run =
2 fingerprint warnings, probe run with a raise inside the window = 0 warnings while
`factor[0]:PastReturnFactor` was already recorded and already differing.

Purpose: make the "the data changed" diagnostic survive a failing run, without ever
letting the diagnostic replace the real exception, and without adding a single warning
to the happy path.
Output: a `partial` mode on `_compare_fingerprints`, a guarded failure-path call in
`run()` and `run_cv()`, and three permanent tests in `tests/test_backtest_rebuild.py`
that lock the fixed behaviour (control arm kept).
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/quick/260921-w6r-report-the-data-changed-fingerprint-diagnostic-when/reproduction.py
@quantlab/base/backtest.py
@tests/test_backtest_rebuild.py
@tests/backtest_fixtures.py

Read-first line numbers in `quantlab/base/backtest.py` (as of this plan):
`FINGERPRINT_COMPARED_FIELDS` :26, `run()` :365-411 (the window call :391, the compare
:392), `run_cv()` :413-566 (the per-fold window call :488-493, the stitched recording
:513-516), `_backtest_window` :758, `_redate_factors` :1038-1053, `_align_and_predict`
:1055-1070, `_load_prices` :1072-1092, `_record_factor_fingerprints` :1121,
`_compare_fingerprints` :1175-1218.

Measured baseline, do not re-investigate (`uv run pytest` on `reproduction.py`, wandb
disabled): control = 2 warnings, `'factor[0]:PastReturnFactor'` and `'price_dataset'`,
both "differing fields: digest, n_symbols ... n_symbols: expected 6, got 5"; probe =
0 warnings with `sorted(backtester._fingerprints) == ['factor[0]:PastReturnFactor']`.
The full `tests/test_backtest_rebuild.py` file is 16 tests, ~18s.

Project constraints that bind here: configs are constructed directly, never through
`quantlab/config/__init__.py` (D-32); no backward-compatibility obligation, so change
the signature directly with no shim; `quantlab/base/backtest.py` is commented and
documented in Chinese — keep that style for new docstrings/comments there, while
`tests/` stays English; every command below is relative to the repo root, never an
absolute `/Users/...` path (execution may run in an isolated worktree, and an absolute
main-tree path would test unchanged code and report a false green).
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: port the reproduction into tests/test_backtest_rebuild.py as three RED tests</name>
  <files>tests/test_backtest_rebuild.py</files>
  <behavior>
    Three new tests, all asserting the FIXED behaviour, so all three are red on today's code:

    1. `test_a_raise_inside_the_window_still_reports_the_changed_data` — control arm + probe arm
       over one store mutation. Control (no raise): the mismatch is reported, exactly as today.
       Probe (raise inside the window): the mismatch is reported too, marked partial, and the
       original exception propagates.
    2. `test_a_failing_partial_diagnostic_never_replaces_the_real_exception` — a diagnostic that
       raises cannot become the exception the caller sees.
    3. `test_run_cv_reports_a_partial_comparison_when_a_fold_raises` — the same failure-path
       diagnostic exists in `run_cv`.
  </behavior>
  <action>
Extend `tests/test_backtest_rebuild.py`; do NOT create a new test module and do NOT copy
`reproduction.py` into `tests/` — it is scratch input, the task-dir copy stays as the record.
Reuse the module's existing `warning_messages` fixture, `_day`, `_backtester`, `_trained`,
`_read_run_config`, `_fingerprint_warnings` helpers and the autouse `_offline_wandb` fixture
(the reproduction lacked it and hit the real wandb API; inside this module it does not).

Imports and constants: add `SYMBOLS` to the existing `from tests.backtest_fixtures import (...)`
list, and add a module constant next to `FINGERPRINT_WARNING`:
`PARTIAL_WARNING = "comparison is PARTIAL"`, the distinctive substring of the tail Task 2 adds
to `quantlab/base/backtest.py`, with a comment saying so. Do NOT import that tail from
`quantlab.base.backtest`: an ImportError at module level would break collection of the whole
file, and the other 16 tests must keep passing while these three are red.

Refactor first, so the run_cv arm does not duplicate ~30 lines: extract the body of
`test_run_cv_rebuild_reproduces_the_stitched_curve` that builds the CV setup (write the
`CV_N_BARS` store, build `model_dates` from `CV_BARS`/`CV_TRAIN_PERIODS`, `make_model` +
`collect()` + `train_cv(train_periods=CV_TRAIN_PERIODS, gap_periods=0)`, assert exactly one
`cv_folds.json` manifest, build the `USEquityCrossectionSelectStockVectorBt` over
`cv_project_dir=str(manifests[0].parent)` and the `CV_FIRST_TEST_BAR..CV_LAST_TEST_BAR`
window) into a module-level helper `_cv_original(tmp_path)` returning that backtester, and
make the existing test call it. The existing test keeps every assertion it has today.

Add a section banner comment above the new tests, in the style of the file's existing
"Task N: ..." banners: "Task 3: a failed run still reports the changed data (D-03.11-UAT-A)".

Test 1, `test_a_raise_inside_the_window_still_reports_the_changed_data(tmp_path,
warning_messages, monkeypatch)`:
  - `dataset_config, checkpoint = _trained(tmp_path)`; `first = _backtester(tmp_path,
    dataset_config, checkpoint=checkpoint).run()`; `saved = _read_run_config(first.run_dir)`.
  - A real data change at the same path: call `write_price_store(tmp_path / "store",
    symbols=SYMBOLS[:-1], n_bars=N_BARS)` — `_trained` wrote that same `store` root, so one
    symbol disappears and both the factor and the price fingerprint really differ
    (digest + n_symbols).
  - Control arm: `warning_messages.clear()`, then
    `module_utils.load_backtester_from_config(saved).run()`. Assert `_fingerprint_warnings`
    returns exactly 2 messages, one naming `'factor[0]:PastReturnFactor'` and one naming
    `'price_dataset'`, that each ends with the unchanged today-text `"(D-27); continuing"`,
    and that none contains `PARTIAL_WARNING`. This arm is what proves the mismatch is
    detectable at all and that the happy path gained no extra or reworded warning; keep it.
  - Probe arm: `warning_messages.clear()`; `rebuilt = module_utils.load_backtester_from_config(
    saved)`; `monkeypatch.setattr(rebuilt.config.model, "predict_panel", <a function raising
    ValueError("representative downstream failure")>)` — it stands in for a real
    post-fingerprint failure such as a ticker-era checkpoint against a PERMNO panel, or
    "the feature panel lacks N of the symbols this model was trained on"; then
    `with pytest.raises(ValueError, match="representative downstream failure"): rebuilt.run()`.
  - Probe assertions: `sorted(rebuilt._fingerprints) == ["factor[0]:PastReturnFactor"]` (the
    price fingerprint does not exist yet at raise time); `_fingerprint_warnings` returns
    exactly 1 message; it names `'factor[0]:PastReturnFactor'`, contains
    `"n_symbols: expected 6, got 5"` and contains `PARTIAL_WARNING`; and NO message
    contains the string `"not read by this run"` — `price_dataset` was not read YET, and
    warning about it would be a false alarm invented by the fix.

Test 2, `test_a_failing_partial_diagnostic_never_replaces_the_real_exception(tmp_path,
warning_messages, monkeypatch)`: same setup and same mutation as test 1; on the rebuilt
backtester patch `predict_panel` to raise `ValueError("representative downstream failure")`
AND patch the instance's `_compare_fingerprints` to raise `RuntimeError("the diagnostic
itself is broken")`. Assert with `pytest.raises(ValueError, match="representative downstream
failure")` that the ValueError is what propagates, assert
`"the diagnostic itself is broken" not in repr(excinfo.value)`, and assert that at least one
WARNING message reports the broken diagnostic (a message mentioning `"diagnostic"` that is
NOT one of `_fingerprint_warnings`) — the guard swallows, but it does not hide.

Test 3, `test_run_cv_reports_a_partial_comparison_when_a_fold_raises(tmp_path,
warning_messages, monkeypatch)`: `original = _cv_original(tmp_path)`; `first =
original.run_cv()`; assert `len(first.folds) > 1` (the probe depends on the fold window being
narrower than the stitched window); `rebuilt = module_utils.load_backtester_from_config(
_read_run_config(first.run_dir))`; patch `rebuilt.config.model.predict_panel` to raise
`ValueError("representative downstream failure")`; `warning_messages.clear()`; assert the
ValueError propagates out of `rebuilt.run_cv()`. Then assert `_fingerprint_warnings` is
non-empty, that EVERY one of them contains `PARTIAL_WARNING`, that one names
`'factor[0]:PastReturnFactor'`, and that none contains `"not read by this run"`. Say plainly
in the docstring what this test does NOT claim: the store is unchanged here, and the fold-0
window is narrower than the stitched window the expected fingerprint describes, so the
differing fields are `end`/`n_timestamps`/`digest` by construction. What is locked is that
the diagnostic RUNS and is marked partial, and that the original exception propagates — that
range caveat is exactly why the partial marker exists.
  </action>
  <verify>
    <automated>uv run pytest tests/test_backtest_rebuild.py -q -k "still_reports_the_changed_data or never_replaces_the_real_exception or reports_a_partial_comparison" ; test $? -ne 0</automated>
    <automated>uv run pytest tests/test_backtest_rebuild.py -q -k "not (still_reports_the_changed_data or never_replaces_the_real_exception or reports_a_partial_comparison)"</automated>
  </verify>
  <done>
The first command exits 0, meaning the three new tests failed on today's code (red). The second
passes with 16 tests, i.e. the `_cv_original` extraction changed no existing behaviour and the
module still collects. Read the red output once (`uv run pytest tests/test_backtest_rebuild.py
-k "still_reports_the_changed_data" -x -q 2>&1 | tail -30`) and confirm the failure is the
expected assertion — the probe arm emitted 0 fingerprint warnings — and not a collection error,
a fixture typo or a wandb network call.
  </done>
</task>

<task type="auto">
  <name>Task 2: partial fingerprint comparison on the failure paths of run() and run_cv()</name>
  <files>quantlab/base/backtest.py</files>
  <action>
Four edits in `quantlab/base/backtest.py`, Chinese docstrings/comments to match the file.

(a) Next to `FINGERPRINT_COMPARED_FIELDS` (:26) add a module constant
`FINGERPRINT_PARTIAL_NOTE` holding the tail that marks a partial comparison. Use this text
verbatim so the tests can import it instead of re-spelling it:
"this comparison is PARTIAL: the run failed before it finished reading, so a differing
digest/start/end/n_timestamps may reflect the interrupted read (under run_cv, a single fold's
window) rather than a data change; the original error follows". Add a one-line comment saying
it exists for D-03.11-UAT-A and that `tests/test_backtest_rebuild.py` identifies partial
comparisons by its substring `"comparison is PARTIAL"` (`PARTIAL_WARNING` there), so that
substring must survive any future rewording.

(b) `_compare_fingerprints` (:1175) takes a new keyword-only `partial: bool = False`.
Bind `tail = FINGERPRINT_PARTIAL_NOTE if partial else "continuing"` and end all three
`logger.warning` calls with `f"(D-27); {tail}"` in place of the literal `"(D-27); continuing"`,
so the default (non-partial) message text is byte-identical to today's. In the
`if key not in actual:` branch, return to the loop immediately when `partial` is true, BEFORE
the warning: under a partial comparison that condition means the key has not been read YET,
not that it was not read, and warning about it would be a false alarm. Every other branch
(key differing, key read but absent from expected) keeps its current behaviour. Extend the
Chinese docstring with the partial mode: when it is used, which branch it skips and why, and
that a partial comparison is followed by the original exception rather than by "continuing".

(c) Add `_compare_fingerprints_on_failure(self) -> None` directly after `_compare_fingerprints`:
it calls `self._compare_fingerprints(partial=True)` inside `try` and catches `BaseException`,
and in the handler writes ONE `logger.warning` naming the class and the repr of the diagnostic
error and saying the original error follows — that message must NOT contain the substring
"data fingerprint mismatch", so it never reads as a D-27 mismatch; wrap that logging call in
its own `try` / `except BaseException: pass` so a failing sink cannot resurrect the problem.
The docstring must state the rule explicitly, because it looks like an anti-pattern and a
future reader will otherwise "fix" it back: phase 03.11's WR-02 established that a guard which
hides a module's own bug is a defect, and this is the one place where swallowing is correct —
what the guard protects is that the operator still sees the REAL exception. The diagnostic is
extra information and must never replace the error; it hides nothing either, since its own
failure is reported as a separate warning and the original exception then propagates unchanged.

(d) Wire the failure path into both entry points, wrapping only the spans whose failure would
skip the D-27 comparison, and leaving the existing comparison calls outside the `try` so the
happy path emits exactly what it emits today:
  - `run()`: wrap the block from `train_bounds = self._prepare_model()` (:386) through
    `window = self._backtest_window(start_date, end_date, calendar, *train_bounds)` (:391) in
    `try:` / `except Exception:` where the handler calls `self._compare_fingerprints_on_failure()`
    and then bare-`raise`s. `_prepare_model` is inside the span on purpose: in train mode it
    records the training fingerprints (`_record_training_fingerprints`, WR-05) before
    `model.train()`, which can raise for the same data reasons. The existing
    `self._compare_fingerprints()` at :392 stays where it is, unchanged and non-partial.
  - `run_cv()`: wrap the per-fold `window = self._backtest_window(...)` call (:488-493) the same
    way, inside the loop; a failure earlier in the loop body (checkpoint resolution/loading)
    holds either no fingerprints or the previous fold's, so it is deliberately not covered.
    Then wrap the stitched pair `self._redate_factors(first_start, last_end, calendar)` and
    `stitched_prices = self._load_prices(first_start, last_end)` (:514-515) the same way,
    leaving the `self._fingerprints = {}` reset (:513) and `self._compare_fingerprints()` (:516)
    outside the `try`. Add one line to the run_cv docstring's step 5 noting that a failure in
    the fold window or in the stitched read reports a partial comparison and re-raises.

Do not add a `finally`: the diagnostic must run on the exception path only, or the happy path
would compare twice. Do not change any other message text, and do not change the behaviour
when `expected_fingerprint is None` (a first run still says nothing, before or after a failure).
  </action>
  <verify>
    <automated>uv run pytest tests/test_backtest_rebuild.py -q</automated>
    <automated>uv run pytest tests/test_backtest_run.py tests/test_backtest_run_cv.py tests/test_backtest_persistence.py tests/test_backtest_contracts.py -q</automated>
  </verify>
  <done>
All 19 tests in `tests/test_backtest_rebuild.py` pass, including the three from Task 1, and the
other backtest suites are untouched-green. The probe arm now reports the changed data
(1 warning naming `factor[0]:PastReturnFactor`, carrying the partial note) while the
ValueError still propagates, and the control arm still emits its 2 today-text warnings.
  </done>
</task>

<task type="auto">
  <name>Task 3: full-suite regression and close-out</name>
  <files>.planning/phases/03.11-crsp-permno-symbol-axis-migration-and-tiingo-era-dead-code-r/deferred-items.md</files>
  <action>
Run the project's full test command (below) and confirm it is green. If something outside the
backtest suites fails, diagnose it from the failure output; if a before/after comparison is
genuinely needed, copy the baseline out with `git show HEAD:<path>` into a temp file — never
`git stash` (an interrupted A/B leaves the working tree inside a stash).

Confirm the scratch reproduction was not smuggled into the suite: `grep -rl
"representative downstream failure" tests/` must match only `tests/test_backtest_rebuild.py`,
and `ls tests/ | grep -i reproduction` must be empty. Also make sure no `__pycache__` under
`.planning/quick/260921-w6r-.../` is staged.

Close the item: in
`.planning/phases/03.11-crsp-permno-symbol-axis-migration-and-tiingo-era-dead-code-r/deferred-items.md`,
under `## D-03.11-UAT-A`, replace the single line that currently reads
`**Status:** open — pre-existing since phase **03.7**, untouched by 03.11`
with a resolved line naming this quick task
(`260921-w6r`), the two entry points changed and the test that locks it. Use an exact-match
string replacement of that one line — never a `sed` range delete on a planning document — and
leave the rest of the section (the analysis, the measured table, the sketch of the fix) as the
historical record.

Commit each task atomically (tests, then fix, then close-out) and push every commit as it is
made — the project's standing rule; report honestly if a push fails and never force it.
  </action>
  <verify>
    <automated>uv run pytest -q --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py</automated>
    <automated>grep -rl "representative downstream failure" tests/</automated>
  </verify>
  <done>
The full suite is green (allow several minutes; macOS runs single-threaded under the
`OMP_NUM_THREADS=1` guard in `tests/conftest.py`). The grep lists exactly
`tests/test_backtest_rebuild.py`. `deferred-items.md` marks D-03.11-UAT-A resolved with one
edited line and no other change, verifiable with `git diff --stat` showing 1 insertion and
1 deletion in that file.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| on-disk data store -> backtester | The Zarr store can change between the recorded run and the rebuild; D-27 exists to report it. No new boundary is crossed by this change. |
| backtester -> operator log | Warnings carry fingerprint digests, dataset keys and exception reprs to local loguru sinks. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-w6r-01 | Tampering | changed price/factor store read by a rebuilt run that then fails | medium | mitigate | This plan: the D-27 comparison now also runs on the failure path (`_compare_fingerprints_on_failure`), so a tampered or silently re-based store is reported even when the run dies afterwards. |
| T-w6r-02 | Denial of Service | `_compare_fingerprints_on_failure` on the exception path | low | mitigate | The guard catches `BaseException` and re-raises nothing of its own; a broken diagnostic cannot turn a recoverable failure into a different one, and the log call is itself guarded. Locked by `test_a_failing_partial_diagnostic_never_replaces_the_real_exception`. |
| T-w6r-03 | Information Disclosure | warning text (digests, dataset keys, exception repr) | low | accept | Same content and same local-only loguru sinks as today's D-27 warnings; wandb stays off by default (D-28). No new data leaves the host. |
| T-w6r-SC | Tampering | npm/pip/cargo installs | high | mitigate | Not applicable: this plan installs no package and adds no dependency. If any task turns out to need one, stop and run the package-legitimacy gate first. |
</threat_model>

<verification>
- `uv run pytest tests/test_backtest_rebuild.py -q` — 19 passed.
- `uv run pytest -q --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py` — green.
- Every command above is repo-root-relative; none may be rewritten to an absolute
  `/Users/...` path, or an isolated-worktree run would exercise unchanged code and report a
  false green.
</verification>

<success_criteria>
- A `run()` whose window computation raises logs the D-27 mismatch for every fingerprint
  recorded so far, marked with `FINGERPRINT_PARTIAL_NOTE`, and re-raises the original
  exception unchanged.
- A partial comparison emits no "present in expected_fingerprint but not read by this run"
  warning.
- A diagnostic that raises is reported as its own warning and never becomes the exception the
  caller sees.
- A successful run's fingerprint warnings are unchanged in count and in text (control arm:
  exactly 2, both ending `(D-27); continuing`).
- `run_cv()` carries the same failure-path diagnostic for a fold's window and for the stitched
  recording.
- `reproduction.py` remains only in this task directory; nothing under `tests/` is named after it.
</success_criteria>

<output>
Create `.planning/quick/260921-w6r-report-the-data-changed-fingerprint-diagnostic-when/SUMMARY.md` when done.
</output>
