---
phase: quick-260915-sxx
plan: 01
subsystem: backtest-reporting
tags: [backtest, report, plotly, html, observability]
status: complete

requires:
  - "quantlab/base/backtest.py:BaseBacktester._compute_metrics (the metrics mapping)"
  - "quantlab/base/backtest.py:_bar_label / _label_ns (persisted bar labels)"
  - "SimulationResult (value, returns, liquidations, bar_interval)"
provides:
  - "report.html states its dates as text, byte-identical to the run's metrics.json"
  - "report.html carries a metric table derived generically from the metrics mapping"
  - "equity log/linear toggle, monthly-return bars, forced-liquidation markers"
  - "BaseBacktester._report_summary (presentation-only helper, both split key sets)"
affects:
  - "quantlab/utils/backtest_report.py (rewritten; still a leaf)"
  - "quantlab/base/backtest.py (two call sites + one new presentation helper)"
  - "tests/test_backtest_persistence.py (D-23 locks updated deliberately)"
  - "example/backtest.md (both report.html rows + the stitched-shading note)"

tech-stack:
  added: []
  patterns:
    - "stdlib html.escape at every interpolation point in a hand-written HTML wrapper"
    - "render-time key walking with dotted-path flattening instead of a hardcoded field list"
    - "pandas index.to_period('M') instead of a resample alias (M/ME was renamed across versions)"

key-files:
  created:
    - tests/test_backtest_report.py
  modified:
    - quantlab/utils/backtest_report.py
    - quantlab/base/backtest.py
    - tests/test_backtest_persistence.py
    - example/backtest.md

decisions:
  - "Equity y stays the RAW persisted value; the multiple of initial capital rides along as customdata"
  - "Linear is the default axis, log one click away"
  - "The title is rendered only in the escaped <h1> and is no longer given to plotly's layout title"
  - "The metric table is HTML, never a plotly go.Table"

metrics:
  duration: 18 min
  completed: 2026-09-16

estimate:
  tokens: 105000
  tasks: 3
actuals:
  tokens: 33000
  tasks: 3
  commits: 3
plan_head_before: 147284168f23939609bf51a9e23ac8877be9f67b
---

# Quick Task 260915-sxx: Backtest HTML Report Summary

Rewrote `report.html` from two undated plotly panels into one self-contained page that states its window, split and setup as text, carries the full metric set as a table derived generically from whatever keys the metrics mapping holds, and draws equity (with a log toggle and liquidation markers), drawdown and per-calendar-month returns.

## What Was Built

**The problem.** The report was two linear-axis panels with no dates and no numbers. A run that compounded to +886,077,331% over 244 bars rendered as a flat line with a terminal spike, and the reader could not see the window, the training split, or a single metric without opening `metrics.json` separately.

**Task 1 (tracer) — dates end to end.** The leaf stopped calling `fig.write_html` and started composing its own HTML document around a plotly div. That architectural move was proven with the smallest possible payload (the dates block alone), leaving every existing trace untouched so the 56-test baseline stayed green. `write_backtest_report` gained five keyword-only parameters, all defaulting to `None`, so the original five-argument call form stays legal. `BaseBacktester._report_summary` was added as a presentation-only helper.

**Task 2 — expansion.** Both call sites now pass `metrics`, `returns`, `liquidations` and `init_cash`. The page gained the generic metric table, a third chart row, the log/linear toggle and the liquidation markers.

**Task 3 — prose.** Every description of the report — two rows and a note in `example/backtest.md`, the leaf module docstring, two Chinese docstrings in `backtest.py`, and the D-23 contract prose in the persistence test's module docstring — now matches what the code does.

## Key Decisions

**Equity `y` stays the raw persisted value; the multiple is `customdata`.** Normalising `y` to a multiple of initial capital would break the page's correspondence with `equity.zarr` (the test asserting `traces["equity"]["y"] == equity.zarr's value` exists precisely to keep the picture honest), and would silently mislabel the axis whenever `init_cash` is unknown. The multiple is what makes an enormous compounding run legible, so it rides along in `customdata` and surfaces on hover instead of replacing the data.

**Linear is the default axis, log one click away.** Log is the fix for the straight-line complaint, but a run that loses everything has a non-positive value that renders an empty log panel — a default that can render nothing is the wrong default. The monthly-return panel is the complementary fix: it shows *where* the P&L came from, which a log axis alone does not.

**The metric table names no metric, and that is load-bearing rather than stylistic.** The project is moving to vectorbt-only metrics, so the hand-rolled aggregates this table renders today are being removed in a separate follow-up. Crucially, the report is written *inside a run's staging directory* — an exception while writing it deletes the entire run directory, every artifact, not just the report (T-sxx-03). A report that hardcoded metric names would raise the day the metric set changes and would take every future run with it. Hence: rows derived by walking the blocks at render time, nested dicts flattened to dotted paths by recursion, split values read with `.get`, and a dash for anything absent. Proved behaviourally, not by a source scan.

**The metric table is HTML, not a plotly `go.Table`.** The existing `_report_traces` helper does `{trace["name"]: trace for trace in traces}`, so a trace lacking a `name` raises `KeyError`. Every trace on the page carries an explicit name.

**The title is no longer given to plotly's layout title.** It is rendered only in the escaped `<h1>`. This removes the one raw-title path into the embedded JSON payload, which makes "the title can never be live markup" a total claim rather than one contingent on plotly's own escaping. No test asserted the layout title (verified before removing it), and it was visually redundant with the new heading.

## Deviations from Plan

**1. [Rule 2 — avoiding a known-recurring defect shape] Reworded a docstring so a literal scan stays honest**

- **Found during:** Task 1, on the post-edit import check.
- **Issue:** My first draft of `_report_summary`'s docstring stated the prohibition as "也不调用 `_turnover` 之类的聚合助手" — naming the very literal that the plan's own acceptance criterion greps for. The check dutifully reported `calls _turnover: True` on a method that calls nothing of the sort.
- **Why it mattered:** This is the exact defect this repo recorded **four separate times in phase 03.4** ("a prohibition stated in a docstring must be written in path form when a literal source scan is its acceptance check"). Shipping it would have planted a permanent false positive for the next person verifying the no-aggregate-helper claim.
- **Fix:** Reworded to "换手率那一类聚合助手", with an inline note recording why the method names are deliberately absent. `grep -n "_turnover"` now returns only the seven genuine metric-code references.
- **Commit:** b2ddc12

No other deviations — the plan's `<facts_measured_at_planning_time>` all held. In particular fact 3 was correct: both monkeypatch stubs are variadic, so the signature change broke neither, and neither needed editing.

## Verification

| Gate | Baseline | Result |
|---|---|---|
| `test_backtest_persistence` + `run` + `run_cv` + `metrics` + `report` | 56 passed | **88 passed, 0 failed** (50.5s) |
| `test_universe_filtered_factor -k backtester_runs_with_wrapped_factors` | passes | **1 passed** (asserts the exact run-directory file set) |
| `git diff --stat -- quantlab/backtest/ quantlab/base/config.py` | empty | **empty**, in the working tree *and* across `1472841..HEAD` |
| `grep -c '^from quantlab\|^import quantlab' backtest_report.py` | 0 | **0** |
| Metric-name tripwire (outside comments) | — | **no output** |

The baseline was re-measured in this worktree before any edit (56 passed) rather than taken on trust. The +32 is 30 new leaf tests plus 2 new persistence locks.

**Trace set the page now carries:** `equity`, `drawdown`, `monthly_return`, and `liquidation` (markers, present only when the run liquidated — the overlap fixture does). `drawdown` stays on `x2` and `equity` on `x`.

**Genericity proved behaviourally:** rendering a metrics mapping with an arbitrary unknown key, with a nested sub-dict the code has never seen, and with every currently-shipped key deleted each produces a valid page and raises nothing.

**Not verified by eye.** The plan's last verification line asks a human to open a generated `report.html` and confirm the dates read clearly and that the log button turns the near-flat curve into a readable slope. That is the one check an automated suite cannot make, and it has not been done — see Follow-ups.

## Known Stubs

None. No TODO/FIXME/placeholder markers, no skipped or xfailed tests, and no unrun `<verify>` in the three task gates.

## Follow-ups

- **Visual confirmation (recommended before relying on the report).** Open any run's `report.html` and check the dates block reads clearly and the log button does its job. Everything else is locked by tests; this one is inherently human.
- **The metric set is about to change under this report.** A separate task removes the hand-rolled aggregates (`_turnover` / `_turnover_summary`, `traded_notional` and friends) in favour of vectorbt's own. The table was built to absorb that with no report change — `tests/test_backtest_report.py::test_a_mapping_with_every_shipped_key_deleted_still_renders` is the guard. One caveat: `tests/test_backtest_persistence.py::test_report_carries_the_metric_table_and_the_axis_toggle` asserts `"turnover.sum"` is present as evidence that nested dicts really are flattened. When turnover is removed, that single assertion should be repointed at whatever nested group exists then (or dropped) — the leaf tests cover flattening independently of any metric name.

## Self-Check: PASSED

- `quantlab/utils/backtest_report.py` — FOUND (modified)
- `quantlab/base/backtest.py` — FOUND (modified)
- `tests/test_backtest_report.py` — FOUND (created)
- `tests/test_backtest_persistence.py` — FOUND (modified)
- `example/backtest.md` — FOUND (modified)
- Commit b2ddc12 — FOUND
- Commit ba9a79f — FOUND
- Commit f19bd74 — FOUND
- Working tree clean; no `.planning/` artifact in any code commit
