---
phase: quick-260915-v6i
plan: 01
subsystem: backtest-reporting
status: complete
tags: [backtest, report, drawdown, vectorbt, plotly, tdd]

requires:
  - quantlab/base/backtest.py (BaseBacktester, SimulationResult, _bar_label)
  - quantlab/backtest/engine_vectorbt.py (VectorBtBacktester)
  - quantlab/utils/backtest_report.py (write_backtest_report)
provides:
  - BaseBacktester._drawdown_span (concrete, returns None)
  - VectorBtBacktester._drawdown_span (depth-selected span payload)
  - write_backtest_report(drawdown_span=...) keyword-only parameter
  - "Deepest drawdown span" row in the report's Dates-and-setup block
affects:
  - report.html for every run() and run_cv() run
  - the engine's _report_notes() (now three notes)

tech-stack:
  added: []
  patterns:
    - "engine hook shaped exactly like _engine_stats: read simulation.native, return plain Python"
    - "module-level constant so the hook's only self-dependency is _bar_label"

key-files:
  created: []
  modified:
    - quantlab/utils/backtest_report.py
    - quantlab/base/backtest.py
    - quantlab/backtest/engine_vectorbt.py
    - tests/test_backtest_report.py
    - tests/test_backtest_engine.py
    - tests/test_backtest_persistence.py
    - example/backtest.md

decisions:
  - "The marked record is selected by DEPTH (nanargmin over valley_val/peak_val - 1), never by duration"
  - "Span length is always a bar count (end_idx - start_idx), worded as trading days, never a calendar timedelta"
  - "DRAWDOWN_RECOVERED is module-level, not a class attribute, so _drawdown_span depends only on self._bar_label"
  - "BaseBacktester._drawdown_span is concrete (returns None), keeping both __abstractmethods__ frozensets exact"

metrics:
  duration: 21 min
  completed: 2026-09-16
  tasks: 3
  files: 7

actuals:
  tokens: 10900   # chars/4 over the realized diff (43,598 chars), the plan's own scale
  tasks: 3
  commits: 3      # MEASURED: git rev-list --count 3b46d6b..HEAD
plan_head_before: 3b46d6b141ecb2b15c5056bb6392c92f300840df
---

# Quick 260915-v6i: Mark the Deepest Drawdown Span on the Report Summary

The backtest report's equity row now carries an up triangle at the bar the
**deepest** drawdown began and a down triangle at the bar it ended, with the
same span restated in words at the top of the page — selected by depth rather
than duration, and measured in trading days (bar counts) rather than calendar
days.

## What Was Built

**Task 1 — the leaf report module** (`quantlab/utils/backtest_report.py`).
`write_backtest_report` gained one keyword-only `drawdown_span` parameter, and
`_add_drawdown_span` draws two single-point traces on row 1 beside the
liquidation markers: `deepest_drawdown_start` (`triangle-up`) and
`deepest_drawdown_end` (`triangle-down`), in `#8e44ad` so they cannot be
confused with the liquidation red. The numbers are baked into a static
`hovertemplate`, and the end marker's wording branches on `recovered`. The
module stays a leaf — stdlib, pandas, xarray and plotly only.

**Task 2 — record selection in the engine** (`quantlab/backtest/engine_vectorbt.py`,
`quantlab/base/backtest.py`). `VectorBtBacktester._drawdown_span` reads
`simulation.native.drawdowns.records` and picks the deepest record with
`np.nanargmin`. `BaseBacktester._drawdown_span` is concrete and returns `None`,
so a non-vectorbt engine renders exactly today's page. Both
`write_backtest_report` call sites (`_report_and_persist` and `_persist_cv`)
compute the span once and pass it to the figure *and* to `_report_summary`, so
the picture and the text cannot disagree. A third report note states that the
triangles mark the deepest drawdown while the metric named `Max Drawdown
Duration` measures the longest one.

**Task 3 — the doc** (`example/backtest.md`). The `report.html` row and a new
paragraph record the three things a reader could otherwise get wrong: deepest
is not longest, the length is in trading days, and a never-recovered drawdown
is labelled as such.

## Verification

| Gate | Result |
|------|--------|
| Seven backtest suites, baseline re-measured at `82bede6`-equivalent tree | **122 passed, 0 failed** |
| Seven backtest suites, final | **141 passed, 0 failed** (+19 new tests) |
| `test_the_module_is_a_leaf` | passes (no project-internal import added) |
| `test_abstract_method_sets_are_exact` | passes (the new hook is concrete) |
| Run-directory file set / `metrics.json` keys | unchanged — no task writes a new metric key |

New tests: 9 in `tests/test_backtest_report.py`, 8 in
`tests/test_backtest_engine.py`, 2 in `tests/test_backtest_persistence.py`.
122 + 19 = 141 exactly, so no pre-existing test was silently dropped.

**The non-vacuity that matters.** The engine lock runs against a series whose
drawdown depths are `[-36.4%, -5.2%, -5.9%]` and whose durations are
`[1, 5, 1]` bars: the deepest record lasts 1 bar and the longest lasts 5, and
the test asserts both facts against vectorbt itself before asserting the hook
returns the 1-bar one. An implementation that consulted `max_duration()` fails
here rather than mislabelling an episode on a report a human would act on.
A second series (deepest record still open at the last bar, an earlier one
recovered) makes both constant answers to `recovered` fail one test or the
other.

## TDD

Both Tasks 1 and 2 were driven test-first, and each RED was observed and
attributed before any implementation:

- **Task 1 RED:** 10 failed / 29 passed — `TypeError: write_backtest_report()
  got an unexpected keyword argument 'drawdown_span'`. GREEN: 39 passed.
- **Task 2 RED:** collection error — `AttributeError: type object
  'VectorBtBacktester' has no attribute '_drawdown_span'`. GREEN: 39 passed
  (engine + contracts), 45 passed (persistence + run_cv).

## Deviations from Plan

**1. [Rule 3 - Blocking] The plan's verify commands pointed at the wrong checkout**

- **Found during:** Task 1 verification.
- **Issue:** every `<verify>` block begins
  `cd /Users/daizhaorong/projects/quantlab && uv run pytest ...`, which is the
  **main** checkout. This agent runs in an isolated worktree, so running them
  verbatim would have exercised code that does not contain this task's changes
  and reported a meaningless green.
- **Fix:** ran each verify from the worktree root instead. Confirmed the
  interpreter resolves correctly: `uv run` built a fresh `.venv` **inside** the
  worktree and `quantlab.__file__` points at the worktree tree, not the main
  one. (`.venv` is gitignored at `.gitignore:124`, so the tree stayed clean.)
- **Files modified:** none — execution-environment correction only.

**2. [Rule 1 - Bug] `DRAWDOWN_RECOVERED` was unreachable from the hook's own test**

- **Found during:** Task 2, first GREEN attempt.
- **Issue:** the recovered-code constant was written as a class attribute and
  read as `self.DRAWDOWN_RECOVERED`. The engine tests bind the hook onto a tiny
  stub (no config, no model, no store), so 4 of them failed with
  `AttributeError: '_SpanHost' object has no attribute 'DRAWDOWN_RECOVERED'`.
  Notably the end-to-end persistence tests were already green, so only the
  isolated tests could surface this.
- **Fix:** moved the constant to module level (the plan permitted "module- or
  class-level"). This is better than patching the stub: it leaves
  `_drawdown_span` depending on `self._bar_label` alone, which is precisely
  what makes record selection testable without constructing a backtester. The
  alternative would have made the test's own docstring claim false.
- **Files modified:** `quantlab/backtest/engine_vectorbt.py`.
- **Commit:** 5d19616.

**3. [Measurement] Every plan fact re-measured, no discrepancies**

Facts 1–6 were reproduced in this tree before being relied on, rather than
inherited: the 122-test baseline; `Drawdowns.records` as a pandas DataFrame
with those exact 10 columns; `status` as `int64` with `DrawdownStatus.Active
== 0` / `Recovered == 1`; the `.iloc` trap (`records.iloc[0]` returns
`start_idx` as `np.float64(2.0)`); deepest ≠ longest on the fact-5 series; and
`max_duration()` returning a `5 days` Timedelta while `duration.values` is the
bar count. The second fixture series (`DEEPEST_NEVER_RECOVERS`) was probed
before use and confirmed to yield a deepest record with `status == 0`, depth
−37.5%, 2 bars, alongside an earlier recovered record.

## Threat Mitigations Applied

| Threat | Disposition |
|--------|-------------|
| T-v6i-01 (note/row injection) | New note written free of `<>&"'`; already asserted as a property by `test_report_and_metrics_carry_the_lot_versus_position_note`, which loops over **all** notes |
| T-v6i-02 (report raises, deleting the staged run) | Off-axis and missing-key endpoints drop their own marker; empty records and non-finite depths return `None`. Locked by two leaf tests |
| T-v6i-03 (wrong units / wrong record) | Length is `end_idx - start_idx` worded as trading days; record chosen by depth, named "deepest", with a note distinguishing it from `Max Drawdown Duration` |
| T-v6i-04 (enum renumbering) | `DrawdownStatus.Recovered == 1` asserted directly in a dedicated test |

## Known Stubs

None. No stub, TODO, FIXME, skipped test or unrun `<verify>` was introduced —
scanned across all seven changed files. The only `NotImplementedError` in the
touched files is the pre-existing D-08 benchmark guard.

## Self-Check: PASSED

- All 7 modified files exist on disk.
- All 3 commits present in `3b46d6b..HEAD`: `1527719`, `5d19616`, `29449b8`.
- `git rev-list --count 3b46d6b..HEAD` = 3, matching `commits:` above.
- No files deleted across the plan range; working tree clean.
