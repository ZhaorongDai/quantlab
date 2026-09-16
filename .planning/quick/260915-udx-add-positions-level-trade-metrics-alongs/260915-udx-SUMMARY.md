---
phase: quick-260915-udx
plan: 01
subsystem: backtest
status: complete
tags: [backtest, vectorbt, metrics, reporting, trade-statistics]
requires:
  - quantlab/base/backtest.py (_compute_metrics, _report_notes, to_jsonable path)
  - quantlab/utils/backtest_report.py (generic dotted-path flattening)
provides:
  - metrics["whole"]["positions"] — 13 position-level trade metrics
  - a report note distinguishing lot-level from position-level trade statistics
affects:
  - metrics.json (new nested block), report.html (new prefixed rows + note)
  - wandb summary (new whole/positions/<metric> keys, no code change needed)
tech-stack:
  added: []
  patterns:
    - "Portfolio.replace(trades_type=...) — instance-scoped view switch, never global settings"
    - "nested sub-dict inside a metric block (existing turnover precedent)"
key-files:
  created: []
  modified:
    - quantlab/backtest/engine_vectorbt.py
    - quantlab/base/backtest.py
    - tests/test_backtest_engine.py
    - tests/test_backtest_persistence.py
    - example/backtest.md
decisions:
  - "Narrow scope held: only the whole block gains the positions view; in_sample / out_of_sample trade counts stay lot-level and are now documented as such in two places."
  - "Portfolio.replace is the only correct switch — trades_type is a Portfolio.__init__ parameter; stats() and get_trades() silently ignore it per-call."
  - "Only the 13 trade-derived metrics are recomputed; portfolio-level metrics are unaffected by trades type, so a second copy could only drift."
  - "Non-vacuity is enforced by a test-local fixture that trims holdings without closing them, not by the existing run fixture."
metrics:
  duration: ~15 minutes
  completed: 2026-09-15
actuals:
  tokens: 6368
  tasks: 3
  commits: 3
plan_head_before: 3c561ac21a50b4628d61fbbfea0d8350124e81a4
---

# Quick 260915-udx: Position-Level Trade Metrics Alongside Lot-Level Summary

The vectorbt engine now reports position-level trade statistics beside the existing exit-trades (lot-level) ones in the `whole` metric block, with a report note stating which is which — so a reader no longer takes `Win Rate [%]` for a stock-picking win rate when it is really a per-trim win rate.

## What Was Built

**Task 1 — the positions block (commit `7158b01`).** `VectorBtBacktester` gained `TRADE_STATS_METRICS`, the 13 trade-derived metric names, and `_engine_stats` now computes a second stats frame from `simulation.native.replace(trades_type="positions")`, attaching it under `whole["positions"]`. Each `stats()` call builds its own settings dict. Downstream needed **zero** edits, as the plan predicted: `metrics.json` inherits strict-JSON safety through `to_jsonable`, `report.html` renders `positions.<metric>` rows through the generic dotted-path flattening, and the wandb summary gains `whole/positions/<metric>` keys.

**Task 2 — the note (commit `369c0f9`).** `VectorBtBacktester._report_notes` appends one English note naming both views. It reaches `report.html` and `metrics.json` verbatim.

**Task 3 — documentation (commit `d5fe808`).** `example/backtest.md` gained a "两套交易统计" section (both views, where each appears, why the switch is instance-scoped), the `whole` row of the key table now names the nested block, and both the doc and the `_period_record_stats` docstring state plainly that the sliced blocks' two trade counts remain lot-level and are deliberately not converted.

## Measured Results

Measured on the new `RotateOneOutEqualWeight` fixture (21 bars, 6 symbols, `rebalance_periods=5`), a run in which holdings are trimmed without being closed:

| View | Closed trades | Open | Win rate | Profit factor | Expectancy |
|---|---|---|---|---|---|
| Lot-level (top level, `exittrades`) | 9 | 5 | **55.56%** | 0.7054 | -814.28 |
| Position-level (`whole.positions`) | 3 | 5 | **33.33%** | 0.5991 | -3250.82 |

Underlying records: 14 exit-trade records vs 8 position records. The lot-level view reports 3x the closed trades and a win rate 22.2 points higher on the same simulation — the misreading this task exists to prevent.

**Provenance note:** the `57.27%` vs `50.74%` figures quoted in the plan's objective and reproduced in `example/backtest.md` come from the plan's own planning-time measurement on a real 244-bar, `top_n=200` run. They were **not** re-measured here; the table above is this execution's own measurement.

## Verification

- **Full seven-suite gate: 122 passed, 0 failed** (baseline re-measured at 116 before editing, as instructed; +3 engine, +3 persistence tests). Requirement was >=116 with 0 failed.
- Both Task 1 static gates pass: no global vectorbt settings reference outside comments, and the instance-scoped `replace(trades_type=...)` is present.
- The non-vacuity lock is real: `test_the_fixture_really_diverges_lot_level_from_position_level` asserts lot-level closed > position-level closed > 0 and that the win rates differ, so a `positions` block that forgot to switch the trades type (a byte-for-byte copy) cannot pass.
- The independent derivation reads `simulation.native.positions.records_readable` directly rather than re-deriving what the engine should have produced.

**TDD evidence.** Task 1: all three new tests observed failing with `KeyError: 'positions'` before implementation, then green. Task 2: the note test observed failing (`len(notes) >= 2` with one note) before implementation. The other two Task 2 tests passed on arrival because Task 1 had already shipped the block — they are additive artifact-level regression locks, recorded here rather than presented as RED.

## Deviations from Plan

**1. [Rule 1 — plan assumption corrected] Positions leaves are not all "number or null"**
- **Found during:** Task 2 test design.
- **Issue:** The plan required asserting "every positions leaf a number or null". Two of the 13 metrics (`Avg Winning Trade Duration`, `Avg Losing Trade Duration`) are `pd.Timedelta`, which `to_jsonable` renders as a **string** (and `NaT` as null). The literal assertion would have failed against correct behavior.
- **Fix:** The lock asserts the accurate property — the two duration keys are string-or-null, the other eleven are number-or-null with every float finite. Strict-JSON parsing still rejects NaN/Infinity tokens.
- **Files:** `tests/test_backtest_persistence.py`. **Commit:** `369c0f9`.

**2. [Scope, additive] Documentation went slightly beyond the minimum**
- The plan required updating the `whole` table row and the slice paragraph. I also added a dedicated "两套交易统计" section and a cross-reference bullet under 费用与假设, because the must-have truth is that a reader can tell the views apart *without opening the source*. All within the authorized `example/backtest.md`.

**Authorization gap (raised in the brief): not triggered.** `tests/backtest_fixtures.py` was **not** modified. The diverging fixture is test-local (`RotateOneOutEqualWeight` in `tests/test_backtest_engine.py`), following the `EqualWeightEveryone` idiom the brief pointed to.

**Execution-mode note:** per the orchestrator's deliberate degrade, all three commits were made directly on `main` with no worktree. The standard protected-branch halt was intentionally not applied; branch identity was asserted before each commit and each commit staged explicit paths only.

## Known Stubs

None. No stubbed values, skipped tests, or unrun verification commands.

## Threat Flags

None. No new network endpoint, auth path, file access pattern or schema change. T-udx-01 (note text) is mitigated — the text is a hardcoded literal containing none of the five characters HTML escaping rewrites, asserted as a property in `test_report_and_metrics_carry_the_lot_versus_position_note`. T-udx-02 (global settings) is mitigated — the switch is instance-scoped and the gate asserts zero non-comment references to the global mapping.

## Self-Check: PASSED

- Commits verified present: `7158b01`, `369c0f9`, `d5fe808`.
- Files verified present: `quantlab/backtest/engine_vectorbt.py`, `quantlab/base/backtest.py`, `tests/test_backtest_engine.py`, `tests/test_backtest_persistence.py`, `example/backtest.md`.
- Working tree clean after the final task commit; no stray or untracked files.
