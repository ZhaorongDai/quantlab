---
phase: quick-260921-w6r
plan: 01
subsystem: backtest
tags: [d-27, diagnostics, error-handling, fingerprints]
status: complete
requires:
  - quantlab/base/backtest.py (D-27 fingerprint recording/comparison)
  - tests/backtest_fixtures.py
provides:
  - "quantlab.base.backtest.FINGERPRINT_PARTIAL_NOTE"
  - "BaseBacktester._compare_fingerprints(partial=True)"
  - "BaseBacktester._compare_fingerprints_on_failure()"
affects:
  - quantlab/base/backtest.py
  - tests/test_backtest_rebuild.py
tech-stack:
  added: []
  patterns:
    - "failure-path diagnostic: additive-only, swallowing guard, bare re-raise"
key-files:
  created: []
  modified:
    - quantlab/base/backtest.py
    - tests/test_backtest_rebuild.py
    - .planning/phases/03.11-crsp-permno-symbol-axis-migration-and-tiingo-era-dead-code-r/deferred-items.md
decisions:
  - "The D-27 comparison runs on the exception path too, marked PARTIAL, never in a `finally` (a successful run must compare exactly once)."
  - "Under `partial`, the 'present in expected_fingerprint but not read by this run' branch is skipped: there the condition means 'not read YET'."
  - "`_compare_fingerprints_on_failure` swallows `BaseException` - the one sanctioned exception to 03.11's WR-02, documented in its own docstring."
  - "`PARTIAL_WARNING` is spelled out in the test module rather than imported, so a missing constant cannot break collection of the whole file."
metrics:
  duration: ~35 min
  completed: 2026-09-21
actuals:
  tokens: 21000
  tasks: 3
  commits: 3
  plan_head_before: ec5c0381f41befcab2ab56feec76d1a0bd898240
---

# Quick 260921-w6r: report the data changed when a run fails - Summary

`BaseBacktester.run()` / `run_cv()` now run the D-27 fingerprint comparison on the
exception path as a PARTIAL comparison, so a run that dies inside the backtest window
still tells the operator the data changed - while the original exception propagates
unchanged, even when the diagnostic itself raises.

## What was built

**Task 1 - three RED tests** (`tests/test_backtest_rebuild.py`, commit `c57a8ce`)

Ported the measured reproduction into the existing module (never into `tests/` as its
own file). Also extracted `_cv_original(tmp_path)` from
`test_run_cv_rebuild_reproduces_the_stitched_curve` so the `run_cv` arm does not
duplicate ~30 lines of CV setup; that test kept every assertion it had.

- `test_a_raise_inside_the_window_still_reports_the_changed_data` - control arm (no
  raise: exactly 2 warnings, each ending `(D-27); continuing`, none partial) plus probe
  arm (raise inside the window: 1 partial warning naming the factor, no
  `"not read by this run"`, `ValueError` propagates).
- `test_a_failing_partial_diagnostic_never_replaces_the_real_exception`
- `test_run_cv_reports_a_partial_comparison_when_a_fold_raises`

**Task 2 - the fix** (`quantlab/base/backtest.py`, commit `af3888b`)

- `FINGERPRINT_PARTIAL_NOTE` module constant, with a comment naming D-03.11-UAT-A and
  pinning the substring `"comparison is PARTIAL"` that the tests key on.
- `_compare_fingerprints(*, partial: bool = False)`: binds
  `tail = FINGERPRINT_PARTIAL_NOTE if partial else "continuing"` and ends all three
  warnings with `f"(D-27); {tail}"`, so the non-partial text is byte-identical to before.
  Under `partial` the `key not in actual` branch `continue`s **before** the warning.
- `_compare_fingerprints_on_failure()`: calls the partial comparison inside a
  `try` / `except BaseException`, and in the handler logs one warning (which deliberately
  does NOT contain `"data fingerprint mismatch"`) inside its own
  `try` / `except BaseException: pass`.
- `run()` wraps `_prepare_model()` -> `_backtest_window(...)`; `run_cv()` wraps the
  per-fold `_backtest_window` call and the stitched `_redate_factors` /
  `_load_prices` pair. Each handler calls the diagnostic then bare-`raise`s. Both
  happy-path `_compare_fingerprints()` calls and the `self._fingerprints = {}` reset stay
  outside the `try`; no `finally` anywhere.
- `run_cv`'s docstring step 5 gained the failure-path line.

**Task 3 - regression and close-out**

`deferred-items.md` D-03.11-UAT-A marked resolved with a one-line exact-match
replacement (`git diff --stat`: 1 insertion, 1 deletion; no `sed` range delete).

## Verification (measured, not narrated)

All commands repo-root-relative, run inside the worktree.

| Command | Result |
|---|---|
| `uv run pytest tests/test_backtest_rebuild.py -q -k "<the 3 new>"` (before Task 2) | **3 failed** - the intended RED |
| `uv run pytest tests/test_backtest_rebuild.py -q -k "not (<the 3 new>)"` (before Task 2) | **16 passed** |
| `uv run pytest tests/test_backtest_rebuild.py -q` (after Task 2) | **19 passed** in 22.97s |
| `uv run pytest tests/test_backtest_run.py tests/test_backtest_run_cv.py tests/test_backtest_persistence.py tests/test_backtest_contracts.py -q` | **57 passed** in 46.68s |
| `uv run pytest -q --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py` | **55 failed, 1703 passed, 1 skipped** - see below |
| `grep -rl "representative downstream failure" tests/` | exactly `tests/test_backtest_rebuild.py` |
| `ls tests/ \| grep -i reproduction` | empty |
| `find .planning/quick/260921-w6r-.../ -name "__pycache__" -o -name "*.pyc"` | empty |

The RED output was read: the control arm passed (2 warnings, today's text) and the probe
arm failed on `assert len(probe) == 1` with `probe == []` - the expected assertion, not a
collection error, fixture typo or wandb network call.

## Full suite is NOT green - 55 pre-existing failures, reported not fixed

Task 3's `<done>` expected a green full suite. It is not green. **Zero** of the 55
failures are in a backtest suite:

| File | Failures |
|---|---|
| `tests/test_ingest_shells.py` | 14 |
| `tests/test_ingest_tiingo_universe_wiring.py` | 11 |
| `tests/test_data_dir_cli.py` | 10 |
| `tests/test_ingest_conversion_gate.py` | 6 |
| `tests/test_volume_guard.py` | 5 |
| `tests/test_factor_kunquant.py` | 3 |
| `tests/test_spot_dataset.py` | 2 |
| `tests/test_chunked_ingest.py` | 2 |
| `tests/test_universe.py` | 1 |
| `tests/test_entry_point_contracts.py` | 1 |

Evidence that they are pre-existing and out of scope (SCOPE BOUNDARY - not fixed, not
retried):

- `git diff --stat ec5c038 HEAD` touches exactly two files: `quantlab/base/backtest.py`
  and `tests/test_backtest_rebuild.py`. None of the failing modules imports either.
- The representative failure is `FileNotFoundError: 'ingest_tiingo.py'` from
  `INGEST_SCRIPTS = ("ingest_tiingo.py", ...)` - bare repo-root-relative names, while the
  scripts live at `scripts/ingest_tiingo.py`. `git show ec5c038:ingest_tiingo.py` confirms
  the file was **already absent** from the repo root at the base commit, so the test fails
  identically there.
- Both full-suite runs produced the identical count (55), i.e. deterministic, not flaky
  and not order-dependent on the new tests.

This looks like stale path assumptions left by the 03.11 script relocation. It is a real
open defect, but it belongs to the ingest/CLI suites, not to this task.

## Deviations from Plan

**1. [Reported, not auto-fixed] The full suite is not green.** Task 3's done-criterion
assumed it was. 55 pre-existing failures in ingest/CLI/factor suites are documented above
rather than fixed (SCOPE BOUNDARY: only issues directly caused by this task's changes are
auto-fixed).

**2. [Process] Commits not pushed.** The project's standing rule is to push every commit
as it is made. These commits live on the disposable worktree branch
`worktree-agent-a3b71ed12c6835e59`; pushing that branch would create a junk remote branch.
The push is deferred to the merge back onto `main`. Reported rather than silently skipped.

Everything else executed as written: the PARTIAL marker and the `run_cv` scoping both
survived contact with the code with no widening or narrowing. In particular:

- The probe arm's `sorted(rebuilt._fingerprints) == ["factor[0]:PastReturnFactor"]` held
  exactly as the plan predicted - the price fingerprint genuinely does not exist at raise
  time, which is what makes the skipped branch necessary rather than cosmetic.
- `run_cv`'s fold-0 arm produced partial warnings whose differing fields are
  `end`/`n_timestamps`/`digest` by construction (unchanged store, narrower fold window) -
  the range caveat the plan called out, and the reason the partial marker exists.

## Known Stubs

None.

## Threat Flags

None. No new network endpoint, auth path, file access pattern or schema change. The
warning text carries the same content to the same local-only loguru sinks as today's D-27
warnings (T-w6r-03, accepted in the plan). No package was installed.

## Self-Check: PASSED

- `quantlab/base/backtest.py` - FOUND
- `tests/test_backtest_rebuild.py` - FOUND
- `.planning/phases/03.11-.../deferred-items.md` - FOUND
- commit `c57a8ce` - FOUND
- commit `af3888b` - FOUND
