---
phase: quick-260916-hro
plan: 01
subsystem: backtest-report
status: complete
tags: [backtest, report, plotly, drawdown, liquidation, layout]

requires:
  - "quantlab/base/backtest.py:BaseBacktester._drawdown_span (contract)"
  - "quantlab/backtest/engine_vectorbt.py:VectorBtBacktester (vectorbt drawdown records)"
provides:
  - "report.html triangles spanning valley -> recovery"
  - "a chart with no liquidation trace, liquidations.json unchanged"
  - "a legible three-row axis-title layout"
affects:
  - "quantlab/utils/backtest_report.py (write_backtest_report signature: `liquidations` removed)"

tech-stack:
  added: []
  patterns:
    - "payload key renamed, not shadowed: `start` -> `valley`, locked by `assert 'start' not in span`"
    - "chart/data split made explicit: the picture loses the markers, the artifact keeps them"

key-files:
  created: []
  modified:
    - quantlab/backtest/engine_vectorbt.py
    - quantlab/base/backtest.py
    - quantlab/utils/backtest_report.py
    - tests/test_backtest_engine.py
    - tests/test_backtest_report.py
    - tests/test_backtest_persistence.py
    - example/backtest.md

decisions:
  - "H-1 upheld: `write_backtest_report` DROPS the `liquidations` parameter rather than accepting and ignoring it, so a stale caller gets a TypeError instead of silently drawing nothing."
  - "H-2 upheld: the payload key is RENAMED `start` -> `valley` and the trace `deepest_drawdown_start` -> `deepest_drawdown_valley`; a key named `start` holding a valley would falsify the documented contract."
  - "H-3 upheld: D-03 fixed with an explicit `height=900` plus a shortened row-1 title. No left margin: the collision is vertical."
  - "Deviation: `start_idx` is no longer read at all in `_drawdown_span`, rather than being kept in the bounds guard as the plan's action text implied. Guarding an index that is never dereferenced can only turn a drawable span into None."

metrics:
  duration: ~35 min
  completed: 2026-09-16
  tasks: 3
  files: 7

actuals:
  tokens: 14000      # chars/4 over the realized diff (55,873 chars), the same scale as `estimate`
  tasks: 3
  commits: 3         # MEASURED: git rev-list --count ${plan_head_before}..HEAD
  plan_head_before: 5b3eec6f873aa134f5d61ffe16f9506ee2569b10
---

# Quick 260916-hro: Mark the Drawdown Span from Valley to Recovery Summary

The report's triangle pair now spans the deepest drawdown's **valley to recovery** instead of start to recovery, the liquidation markers are gone from the chart while `liquidations.json` is untouched, and the three left-hand axis titles have room to render.

## Test Results

| Run | Command | Result |
|---|---|---|
| Baseline (at `5b3eec6`, before any edit) | seven backtest suites | **141 passed** in 134.34s |
| Final | seven backtest suites | **140 passed** in 64.46s, zero failures |
| Final | `tests/test_universe_filtered_factor.py` | **84 passed**, zero failures |

### Every delta from 141 accounted for

Net **−1** (141 → 140). Measured per-suite counts afterwards: report **37**, persistence **27**, engine **31** (95 for the three changed suites, against 96 when Task 1's gate ran exactly those three before any Task 2/3 test edits).

| Suite | Change | Tests |
|---|---|---|
| `test_backtest_report.py` | **−4** deleted | `test_liquidation_markers_sit_on_the_equity_curve`, `test_a_liquidation_off_the_equity_axis_is_dropped_not_raised`, `test_no_liquidations_means_no_marker_trace`, `test_the_span_markers_are_not_the_liquidation_colour` |
| `test_backtest_report.py` | **+2** added | `test_the_figure_draws_exactly_these_five_traces`, `test_the_layout_gives_every_axis_title_room_to_render` |
| `test_backtest_persistence.py` | **+1** added | `test_the_liquidating_run_drops_the_chart_markers_and_keeps_the_json` |
| `test_backtest_engine.py` | **0** (renamed) | `test_bars_is_the_chosen_records_end_minus_start_...` → `..._end_minus_valley_...` |

The four deletions were not housekeeping: all four exercised the `liquidations` parameter, which no longer exists, so they can no longer be *written*. The surviving half of the colour test (both triangles share one colour) was folded into `test_the_span_draws_one_triangle_at_each_end_on_the_equity_curve` rather than lost.

## F-2 Version Correction

The plan's F-2 states "(vectorbt 5.24.1 / plotly 5.24.1)". **Measured in this tree: `vectorbt` is 1.1.0**; 5.24.1 is plotly's version only. F-2's substantive claim is correct and was re-verified: the drawdown records columns really are `['id','col','peak_idx','start_idx','valley_idx','end_idx','peak_val','valley_val','end_val','status']`, and `valley_idx` is available. The wrong version string was **not** propagated into any docstring, comment or doc.

## The F-3 Trap, Confirmed and Defused

Re-measured before writing any test, which is what made the lock non-vacuous:

| Fixture | start | valley | end | end−start | end−valley |
|---|---|---|---|---|---|
| `DEEPEST_IS_NOT_LONGEST` | 2 | **2** | 3 | 1 | **1** |
| `DEEPEST_NEVER_RECOVERS` | 4 | 6 | 6 | 2 | **0** |
| `VALLEY_IS_NOT_START` (new) | 2 | **4** | 7 | 5 | **3** |

On the pre-existing fixture start and valley coincide, so every old assertion passes under either rule. The new `VALLEY_IS_NOT_START = [100,120,115,110,90,95,105,125]` separates them (one record, depth −25%, recovered), and the test carries explicit non-vacuity assertions `start != valley`, `bars == end - valley`, `bars != end - start`, plus `'start' not in span`.

The `DEEPEST_NEVER_RECOVERS` edge case is locked too: the valley **is** the last bar, so `bars` legitimately becomes **0** (was 2), asserted with a comment naming the reason.

## D-03 Is Locked by a Budget, Not a Magic Number

F-5's vertical-overlap diagnosis was independently reproduced (emitted layout carried no `height`, `margin={'b':140}`, domains `[0.5032,1.0]`/`[0.2516,0.4632]`/`[0.0,0.2116]`). The new test asserts a per-row pixel budget, and I verified it is genuinely red at the old default rather than decorative:

| Row | Title | Needs | At 450px (old) | At 900px (new) |
|---|---|---|---|---|
| 1 | `value` | 32.5px | 104.3px PASS | 327.9px PASS |
| 2 | `drawdown` | 52.0px | 44.4px **FAIL** | 139.7px PASS |
| 3 | `monthly return` | 91.0px | 44.4px **FAIL** | 139.7px PASS |

This also confirms the plan's reasoning that shortening row 1 alone could never have fixed rows 2 and 3.

## T-hro-02: The Data Survived

Asserted in one test, for the fixture run that really liquidates: the page carries **no** `liquidation` trace, `liquidations.json` still strict-parses and still matches `result.simulation.liquidations` field for field (symbol, both timestamps, price), and the run directory still holds exactly the seven D-24 entries. Neither `write_json_atomically(run_dir / "liquidations.json", ...)` block was touched. The four exact run-directory set assertions are unchanged — `git status` confirmed `test_backtest_run.py`, `test_backtest_run_cv.py` and `test_universe_filtered_factor.py` were never modified.

## Deviations from Plan

### 1. [Rule 2 — missing critical correctness] Two stale-prose sites T-hro-03 did not enumerate

- **Found during:** Task 2
- **Issue:** T-hro-03 listed four prose sites. Two more still claimed the triangles marked where the drawdown *began and ended*: `quantlab/utils/backtest_report.py`'s module docstring item 4, and `quantlab/base/backtest.py`'s `_report_and_persist` (line ~1774) and `_persist_cv` (line ~1919) docstrings.
- **Fix:** Rewritten to "valley → recovery". Both of the first two sentences also contained the forced-liquidation-marker phrase Task 2 had to remove anyway, so they were corrected in the Task 2 commit rather than split awkwardly across two commits.
- **Commit:** `58671ff`

### 2. [Design refinement] `start_idx` is no longer read at all

- **Found during:** Task 1
- **Issue:** The plan said to keep the bounds guard and "extend it to cover the valley index", implying `start_idx` stays. But `start` no longer appears in the payload, so guarding it would let an out-of-range index the code never dereferences turn an otherwise drawable span into `None`.
- **Fix:** `_drawdown_span` reads and bounds-checks exactly the two indices it dereferences (`valley_idx`, `end_idx`). T-hro-01 is still satisfied: no bare index without a guard. Recorded in an inline comment.
- **Commit:** `9a381c8`

## Known Stubs

None. No stub, skipped test or unrun `<verify>` was introduced, so there is nothing to add to `.planning/WINDOWS.md`.

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or schema change. `quantlab/utils/backtest_report.py` remains a leaf (stdlib, pandas, xarray, plotly only — no import was added or removed, and `test_the_module_is_a_leaf` still passes); T-hro-01's drop-do-not-raise discipline is intact, now stated directly instead of by reference to the deleted `_add_liquidations`.

## Commits

| Task | Commit | Description |
|---|---|---|
| 1 | `9a381c8` | `feat`: mark the drawdown span from valley to recovery |
| 2 | `58671ff` | `feat`: drop the liquidation markers from the chart, keep the JSON |
| 3 | `78cbf34` | `fix`: make the three axis titles fit instead of overlapping |

No commit deleted a tracked file (`git diff --diff-filter=D` empty for all three). Docs artifacts are deliberately uncommitted — the orchestrator owns that commit.

## Self-Check: PASSED

- `.planning/quick/260916-hro-.../260916-hro-SUMMARY.md` — written
- Commits `9a381c8`, `58671ff`, `78cbf34` — all present in `git log`
- `commits: 3` measured via `git rev-list --count 5b3eec6..HEAD`, not narrated
- Seven-suite gate: 140 passed, zero failures; universe suite: 84 passed, zero failures
