---
phase: quick-260915-sxx
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/utils/backtest_report.py
  - quantlab/base/backtest.py
  - tests/test_backtest_report.py
  - tests/test_backtest_persistence.py
  - example/backtest.md
autonomous: true
requirements: [260915-sxx]

estimate:
  tokens: 105000
  raw_tokens: 105000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "`report.html` states the backtest dates as TEXT, not only as a shaded band: the window's first and last bar label, the bar count, the bar interval, the model's training window (`run()`) or windows (`run_cv()`), the in-sample range(s) and the out-of-sample ranges. Every one of those strings is byte-identical to the corresponding value in the same run directory's `metrics.json`."
    - "`report.html` carries the whole / in-sample / out-of-sample metric blocks as a rendered table, one column per block. The rows are derived GENERICALLY from whatever keys the blocks actually carry at render time — never from a hardcoded metric list. Nested dicts are flattened to dotted paths by walking them, so a block gains or loses metrics without a code change. A key absent from a block renders as a dash in that column and raises nothing; a key absent from every block produces no row at all."
    - "The report survives the metric set changing under it. This is proved BEHAVIOURALLY, by tests in tests/test_backtest_report.py: rendering a metrics mapping with an arbitrary unknown key, with a nested sub-dict the code has never seen, and with every currently-shipped key deleted, each produces a valid page and raises nothing. No row set, row order or column is decided by a metric name written into `quantlab/utils/backtest_report.py` — the rows come from walking the mapping at render time."
    - "The equity panel carries a working log/linear toggle built from plotly `updatemenus`, defaulting to linear. The `equity` trace's `y` stays the RAW persisted portfolio value (identical to `equity.zarr`'s `value`), and the multiple of initial capital rides along as `customdata` rendered by the trace `hovertemplate`."
    - "The page carries named traces on three rows sharing one time axis: `equity` and `liquidation` markers on row 1, `drawdown` on row 2, `monthly_return` bars on row 3. Every trace on the page has a `name` key. `drawdown` stays on `x2` and `equity` stays on `x`."
    - "`monthly_return` makes a run whose entire P&L lands in one or two bars visible at a glance: it is the per-calendar-month compounded return of `simulation.returns`, grouped by `to_period(\"M\")`, never by a resample alias."
    - "The run directory's file set is UNCHANGED: still exactly config.json, equity.zarr, fingerprint.json, liquidations.json, metrics.json, report.html, weights.zarr (plus `folds/` for `run_cv`). The report stays one self-contained file whose only external reference is the plotly CDN."
    - "`quantlab/utils/backtest_report.py` stays a LEAF: its imports are stdlib, pandas, xarray and plotly only, and zero `quantlab.*` imports. `grep -c '^from quantlab\\|^import quantlab' quantlab/utils/backtest_report.py` prints 0."
    - "No engine, selection or metric arithmetic changed, and no metric is computed for the report. `git diff --stat -- quantlab/backtest/ quantlab/base/config.py` is empty; inside `quantlab/base/backtest.py` the only changes are the two `write_backtest_report` call sites, the new `_report_summary` presentation helper and docstrings. Neither call site calls `_turnover` or any other aggregate helper."
    - "Every text value interpolated into the hand-written HTML wrapper (title, notes, summary labels and values, metric names and values) passes through `html.escape`, so a run directory name or a note containing angle brackets cannot inject markup."
    - "Gated suites keep their pre-change result. Measured at planning time on tests/test_backtest_persistence.py + tests/test_backtest_run.py + tests/test_backtest_run_cv.py + tests/test_backtest_metrics.py: 56 passed, 0 failed, ~51s. After this task the same four files pass with 0 failures and a count of at least 56 plus the tests added here."
  artifacts:
    - path: quantlab/utils/backtest_report.py
      provides: "write_backtest_report with the widened keyword-only signature; HTML page composition (escaped header + dates block + generic metrics table + plotly div); equity/drawdown/monthly_return/liquidation traces; log-linear updatemenus"
      contains: "def write_backtest_report"
    - path: quantlab/base/backtest.py
      provides: "_report_summary presentation helper; both call sites pass summary, metrics, returns, liquidations and init_cash"
      contains: "def _report_summary"
    - path: tests/test_backtest_report.py
      provides: "Leaf unit tests: synthetic DataArrays straight into write_backtest_report, no backtest run; dates block, generic metric table, unknown/absent/deleted keys, trace set, toggle, escaping, null-block rendering"
      contains: "write_backtest_report"
    - path: tests/test_backtest_persistence.py
      provides: "Updated D-23 contract prose and report locks asserting the new trace set and the dates/metrics text against the run's own metrics.json"
      contains: "_report_traces"
    - path: example/backtest.md
      provides: "Updated report.html rows in both the run() and run_cv() output tables, and the D-23 report description"
      contains: "report.html"
  key_links:
    - from: quantlab/base/backtest.py
      to: quantlab/utils/backtest_report.py
      via: "write_backtest_report(...) in _report_and_persist and in _persist_cv"
    - from: tests/test_backtest_persistence.py
      to: quantlab/utils/backtest_report.py
      via: "monkeypatch.setattr(backtest_module, 'write_backtest_report', ...) and _report_traces(html)"
---

<objective>
Make a backtest run's `report.html` readable: state the dates as text, put the full metric set on the page, and draw charts that show what actually happened.

Purpose: the current report is two linear-axis panels with no dates and no numbers. A run that compounded to +886,077,331% over 244 bars renders as a flat line with a final spike, and the reader cannot see the window, the training split, or a single metric without opening `metrics.json` separately.

Output: a rewritten leaf report module, both call sites passing the extra presentation data, updated locks, and updated docs.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

@quantlab/utils/backtest_report.py
@quantlab/base/backtest.py
@tests/test_backtest_persistence.py
@tests/backtest_fixtures.py
@example/backtest.md
</context>

<facts_measured_at_planning_time>
Do not re-derive these. They were read from the live tree at planning time.

1. **Baseline gate:** `uv run pytest tests/test_backtest_persistence.py tests/test_backtest_run.py tests/test_backtest_run_cv.py tests/test_backtest_metrics.py -q` → **56 passed, 0 failed, 50.90s**.
2. **plotly 5.24.1**, pandas available. `updatemenus` and `to_html(full_html=False, include_plotlyjs="cdn")` are both supported.
3. **The two monkeypatch stubs are already variadic** — `tests/test_backtest_run_cv.py:372` is `def _fail(*args, **kwargs)` and `tests/test_backtest_persistence.py:303` is `def _interrupt(*args, **kwargs)`. The task brief called them "2-positional-arg" stubs; they are not. **A signature change cannot break either one**, so neither stub needs editing.
4. **`write_backtest_report` has exactly two call sites**, both in `quantlab/base/backtest.py`: line 1668 inside `_report_and_persist`'s `_write` closure, and line 1813 inside `_persist_cv`'s `_write` closure. Nothing else in the repo calls it.
5. **The run directory file set is asserted as an EXACT set in four places**: `tests/test_backtest_persistence.py:71` (`D24_ARTIFACTS`), `tests/test_backtest_run.py:119`, `tests/test_backtest_run_cv.py:596`, `tests/test_universe_filtered_factor.py:932`. Adding any sidecar asset file to the run directory turns all four red. The report must stay one file.
6. **`_report_traces(html)` (tests/test_backtest_persistence.py:666) does `{trace["name"]: trace for trace in traces}`** — a trace with no `name` key raises `KeyError`. Every trace added to the page must carry a `name`. This is why the metrics table must be HTML, not a plotly `go.Table`.
7. **Metrics shape at each call site.** `run()` (line 368-374): `metrics` has `whole`, `in_sample`, `out_of_sample`, `training_window`, `in_sample_range`, `out_of_sample_ranges`, `notes`, and `trained_checkpoint` in train mode. `run_cv()` (line 516-529): `metrics` is `{"stitched": <same shape but with training_windows / in_sample_ranges / out_of_sample_ranges and NO single in_sample_range>, "folds": [...], "notes": [...]}`. **Treat this shape as the split keys only** — the CONTENTS of the `whole` / `in_sample` / `out_of_sample` blocks are about to change (fact 8) and must never be hardcoded.
8. **The metric set is about to change under this report (user decision, 2026-09-15).** The project is moving to "metrics come from vectorbt only". The hand-rolled aggregates — `BaseBacktester._turnover` / `_turnover_summary` (D-22), `traded_notional` and friends — are being REMOVED from `metrics.json` in a separate follow-up task that owns the engine/metrics change. Consequences that bind this plan: the report renders the metric table generically from whatever keys are present, names no metric in code, and draws no turnover panel. An earlier draft of this plan had a fourth turnover row and hardcoded `turnover.*` table rows; both are removed.
9. **`SimulationResult.liquidations`** is a list of dicts with keys `symbol`, `signal_timestamp`, `fill_timestamp`, `price` (confirmed at tests/test_backtest_persistence.py:249).
10. **Docs to update:** `example/backtest.md` line 124 (CV table `report.html` row), lines 131-132 (stitched-report shading note), line 421 (the `run()` table's D-23 report row).
</facts_measured_at_planning_time>

<tasks>

<task type="tracer">
  <name>Task 1: Dates on the page, end to end — page composition proven with the smallest payload</name>
  <files>quantlab/utils/backtest_report.py, quantlab/base/backtest.py, tests/test_backtest_report.py, tests/test_backtest_persistence.py</files>
  <action>
Wire ONE new capability through every layer: leaf signature → call sites → a real persisted `report.html` → an assertion. The architectural move being proven here is page composition — the module stops calling `fig.write_html` and starts building its own HTML document around a plotly div. Prove that with the dates block alone, leaving all existing traces untouched, so the 56 baseline tests stay green.

Widen `write_backtest_report`'s signature by APPENDING keyword-only parameters that all default to None, keeping the five existing parameters in their current order and spelling:

    value, path, *, in_sample_range, notes, title,
    summary=None, metrics=None, returns=None, liquidations=None, init_cash=None

Defaulting every new parameter to None is what keeps the old call form legal and keeps the two variadic monkeypatch stubs valid (fact 3). Task 2 fills in `metrics`, `returns`, `liquidations` and `init_cash`; this task only uses `summary`.

In the leaf, replace `fig.write_html(str(path), include_plotlyjs="cdn")` with: build the figure exactly as today, get the div via `fig.to_html(full_html=False, include_plotlyjs="cdn")`, and write a complete HTML document containing, in order, an escaped `&lt;h1&gt;` of the title, a dates/setup block rendered from `summary`, the plotly div, and the notes. Keep the notes readable on the page — the existing in-plot annotation may stay, move into the wrapper, or both, as long as every `_report_notes()` line still appears in the file. `summary` is an ordered mapping of display label to already-formatted display string: the leaf renders it and formats nothing. Import `html` from the stdlib and pass EVERY interpolated string through `html.escape` (threat T-sxx-01). Add `pandas` to the imports; the module docstring already declares pandas as permitted. Do not add any `quantlab.*` import — this module is a leaf and its docstring says so.

In `quantlab/base/backtest.py`, add a presentation-only helper `_report_summary(self, simulation, block)` returning an ordered dict of display label to display string. `block` is the metrics mapping holding the split keys: `metrics` for `run()`, `metrics["stitched"]` for `run_cv()`. It must handle BOTH split key sets (fact 7) — singular `training_window` / `in_sample_range` for `run()`, plural `training_windows` / `in_sample_ranges` for `run_cv()`, with no single `in_sample_range` in the plural case. Include at least: the backtest window as first and last bar label with the bar count, the bar interval, the training window(s), the in-sample range(s), the out-of-sample ranges, and the run setup read off `self.config` (`model_mode`, `rebalance_periods`, `top_n`, `direction`, `init_cash`, `fees`). Include `trained_checkpoint` when `block` carries it. Render an absent or null value as an explicit dash, never as the word None. Read the split values off `block` by `.get(...)` rather than indexing, so a key that disappears later degrades to a dash instead of raising. The bar labels must be the SAME strings the metrics carry, so the page and `metrics.json` agree byte for byte — read them off `block` rather than reformatting timestamps. This helper only reads and formats; it computes no statistic and calls no aggregate helper.

Pass `summary=self._report_summary(simulation, metrics)` at the `_report_and_persist` call site and `summary=self._report_summary(simulation, metrics["stitched"])` at the `_persist_cv` call site.

Create `tests/test_backtest_report.py`: leaf-only tests that build small synthetic `xr.DataArray` inputs and call `write_backtest_report` directly, with no backtest run, so they stay fast. Cover: the summary labels and values appear in the written file; a title and a summary value containing angle brackets come back escaped rather than as live markup; calling with only the five original arguments still writes a valid page. Add to `tests/test_backtest_persistence.py`, reusing the existing module-scoped `overlap_run` fixture, one test asserting the real `report.html` carries the window dates, the training window and the in-sample range, each compared against the value read from that same run's `metrics.json`.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab &amp;&amp; uv run pytest tests/test_backtest_report.py tests/test_backtest_persistence.py -q 2>&amp;1 | tail -5</automated>
  </verify>
  <done>New leaf tests pass, the new persistence test passes, and every previously passing test in `tests/test_backtest_persistence.py` still passes (0 failures). A real run's `report.html` states its window, training window and in-sample range as text matching its own `metrics.json`. `grep -c '^from quantlab\|^import quantlab' quantlab/utils/backtest_report.py` prints 0.</done>
</task>

<task type="auto">
  <name>Task 2: Expand — generic metric table, log toggle, monthly returns, liquidation markers</name>
  <files>quantlab/utils/backtest_report.py, quantlab/base/backtest.py, tests/test_backtest_report.py, tests/test_backtest_persistence.py</files>
  <action>
Build out from the proven page composition. Fill the parameters Task 1 added.

At both call sites pass `metrics=`, `returns=simulation.returns`, `liquidations=simulation.liquidations` and `init_cash=self.config.init_cash`. For `_report_and_persist` pass `metrics=metrics`; for `_persist_cv` pass `metrics=metrics["stitched"]`. Change no arithmetic anywhere, and do not call `_turnover`, `_turnover_summary` or any other aggregate helper from the persist sites: this task adds no statistic and reuses none. The report displays only what `_compute_metrics` already put in the mapping.

**Metrics table (HTML, generic, not a plotly trace).** Render the `whole`, `in_sample` and `out_of_sample` blocks of `metrics` as one HTML table: three columns, one row per metric. Derive the row set by walking the blocks at render time and taking the union of the keys actually present — never from a hardcoded metric list. Flatten nested dicts generically to dotted paths by recursion (a sub-dict named `x` holding `y` becomes the row `x.y`), so a block can gain or lose nested groups without a code change. Rules: a key missing from one block renders as a dash in that column; a key present in no block produces no row; a block that is None renders as a full column of dashes rather than being dropped, so the reader can see the block exists and is empty; NaN and infinity render as a dash. Render `trained_checkpoint` when present.

This genericity is load-bearing, not stylistic: the metric set is being replaced with vectorbt's own (fact 8), and a report that names metrics in code breaks the day that lands. Do not write any metric name into this module. For the same reason, do NOT add an exposure or turnover panel — every candidate key for one is either being removed or not yet confirmed to exist in the new set; a chart hardcoding such a key is exactly the breakage this rule exists to prevent.

The table must be HTML because any plotly trace lacking a `name` key crashes the existing `_report_traces` helper (fact 6) — do not use `go.Table`.

**Charts.** Grow the figure to three rows on a shared x axis, keeping the existing rows where they are so the existing axis assertions hold: row 1 equity, row 2 drawdown, row 3 monthly returns. Every trace carries an explicit `name`: `equity`, `drawdown`, `monthly_return`, and `liquidation` for the markers. No trace on the page may be named after a benchmark — benchmark comparison is excluded this phase (D-08).

- *Equity + log toggle.* Keep the `equity` trace's `y` as the RAW persisted portfolio value so it stays identical to `equity.zarr`'s `value`. Express the multiple of initial capital as `customdata` (value divided by `init_cash`) surfaced through the trace `hovertemplate`, and name both units in the y-axis title. Rejected alternative, recorded so it is not "fixed" later: making `y` itself the multiple would break the page's correspondence with the persisted equity and would silently mislabel the axis whenever `init_cash` is None. Add plotly `updatemenus` with two buttons relayouting the row-1 y axis between `linear` and `log`. Default to **linear**: log is one click away, and a run that loses everything has a non-positive value that renders an empty log panel. The log button is the fix for the straight-line complaint; the monthly panel below is the fix for not being able to see where the P&L came from.
- *Drawdown.* Unchanged formula and unchanged row.
- *Monthly returns.* Bar chart of the per-calendar-month compounded return of `returns`, grouped with `index.to_period("M")` — not a resample alias, because the `M`/`ME` alias was renamed across pandas versions while `to_period` is stable across both. A short window legitimately yields one or two bars; that is correct, not a bug.
- *Liquidation markers.* Scatter markers on the equity row at each record's `fill_timestamp`, looked up against the equity index, with the symbol in the hover text. Drop records whose timestamp is not on the equity axis rather than raising. Omit the trace entirely when there are no liquidations.

**Update the locked tests, deliberately.** In `tests/test_backtest_persistence.py`, `test_report_has_equity_and_drawdown_and_shades_the_in_sample_range` currently pins the two-trace contract. Replace the trace-set equality with the new expected set for that fixture (it does liquidate, so it has the markers), and keep everything else the test proves: the `equity` y values still equal `equity.zarr`'s `value`, drawdown is still value over running max minus one, drawdown is still on `x2` and equity on `x`, the shaded band still carries the persisted in-sample endpoints, and every `_report_notes()` line still appears. Add assertions that the metric table carries the same numbers as `metrics.json` and that the log/linear toggle is present. `test_report_without_in_sample_overlap_has_no_shaded_range` and `test_report_and_metrics_carry_no_benchmark` should still pass unchanged — confirm that rather than assuming it, and if either goes red, fix the report, not the test. Extend `tests/test_backtest_report.py` with leaf-level coverage of: a metrics mapping carrying an arbitrary unknown key and an unseen nested sub-dict (both render, nothing raises); a mapping with every currently-shipped key deleted (still a valid page); the null-block column; the dash rendering; the month grouping; and the no-liquidations case.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab &amp;&amp; uv run pytest tests/test_backtest_report.py tests/test_backtest_persistence.py tests/test_backtest_run_cv.py -q 2>&amp;1 | tail -5</automated>
  </verify>
  <done>All three files pass with 0 failures. A real `report.html` carries the named traces, an HTML metric table whose numbers match `metrics.json`, and a working log/linear toggle. A metrics mapping with unknown keys, and one with all known keys removed, both render without raising — the behavioural proof that no metric name drives the table. The no-shaded-range and no-benchmark tests both still hold, and the run directory file set is unchanged.</done>
</task>

<task type="auto">
  <name>Task 3: Docs, docstrings and the full gate</name>
  <files>example/backtest.md, quantlab/utils/backtest_report.py, quantlab/base/backtest.py, tests/test_backtest_persistence.py</files>
  <action>
Bring every prose description of the report up to what the code now does, then run the full gate.

`example/backtest.md`: rewrite the `report.html` row of the `run()` output table (line 421) — it currently describes only the two panels, the shading and the notes. Describe the dates block, the metric table, the three panels and the log/linear toggle, and keep the two statements that are still true: no benchmark curve (D-08), and plotly.js from the CDN so each report is small but needs network access to draw. State that the metric table is rendered from whatever the metrics mapping carries, so it tracks changes to the metric set without a report change. Update the CV table's `report.html` row (line 124) and the stitched-report shading note (lines 131-132) the same way — the stitched report still shades nothing, and now states its per-fold training windows and in-sample ranges as text, which is a strictly better answer to the problem that note describes.

Update the module docstring of `quantlab/utils/backtest_report.py`: it currently says "One plotly page with two panels". Describe the current page, and record the generic-table rule and WHY it exists (the metric set is moving to vectorbt's own, fact 8), so nobody reintroduces a hardcoded metric list. Keep the leaf declaration and the CDN rationale. Docstrings in this file are English — match it. Update the `write_backtest_report` docstring to document every parameter including the new ones.

Update the two Chinese docstrings in `quantlab/base/backtest.py` that describe the report: `_report_and_persist` (the D-23 paragraph at line 1650) and `_persist_cv` (line 1778). Docstrings in that file are Chinese — match it. Document `_report_summary` as presentation-only.

Update the D-23 contract prose in the `tests/test_backtest_persistence.py` module docstring (lines 24-28), which still states the two-trace contract in prose, to the contract the tests now enforce.

Then run the full gate and record the counts in the summary.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab &amp;&amp; uv run pytest tests/test_backtest_persistence.py tests/test_backtest_run.py tests/test_backtest_run_cv.py tests/test_backtest_metrics.py tests/test_backtest_report.py -q 2>&amp;1 | tail -4 &amp;&amp; uv run pytest tests/test_universe_filtered_factor.py -k backtester_runs_with_wrapped_factors -q 2>&amp;1 | tail -3 &amp;&amp; git diff --stat -- quantlab/backtest/ quantlab/base/config.py</automated>
  </verify>
  <done>The four baseline files plus the new leaf test file pass with 0 failures and a count of at least 56 plus the tests added by this plan. The universe-filtered backtester test passes. `git diff --stat -- quantlab/backtest/ quantlab/base/config.py` prints nothing, proving no engine or config change. No document still describes the report as two panels.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| run metadata → HTML page | The run directory name, `_report_notes()` lines, summary values and metric names/values are interpolated into a hand-written HTML document. Before this task plotly owned all page generation and escaped its own payload; this task takes that responsibility on. |
| report.html → plotly CDN | The page fetches plotly.js over the network when opened (pre-existing D-23 behaviour, unchanged). |
| metrics mapping → report renderer | The report consumes a mapping whose key set is about to be replaced wholesale (fact 8). |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-sxx-01 | Tampering | `write_backtest_report` HTML wrapper | medium | mitigate | Every interpolated string (title, notes, summary labels/values, metric names/values) passes through stdlib `html.escape`. Task 1 ships a test that a title and a summary value containing angle brackets render as text, not as live markup. |
| T-sxx-02 | Information disclosure | plotly CDN fetch from `report.html` | low | accept | The page carries only local backtest numbers; the CDN fetch reveals that a plotly page was opened, not its contents. Unchanged from D-23, and the module docstring already records this trade. |
| T-sxx-03 | Denial of service | report rendering inside `_persist_run_dir`'s staging `_write` | medium | mitigate | An exception raised while writing the report deletes the staging directory and re-raises, destroying the ENTIRE run — every artifact, not just the report. A report that hardcodes metric names would raise `KeyError` the day the metric set changes (fact 8) and would take every future run with it. Mitigated by deriving the table rows from the keys actually present, reading split values with `.get(...)`, and testing a mapping with all known keys removed. |

No package-manager installs and no new third-party dependencies: the only additions are stdlib `html` and `pandas`, already declared permitted by the module docstring. No package legitimacy gate applies.
</threat_model>

<verification>
- `uv run pytest tests/test_backtest_persistence.py tests/test_backtest_run.py tests/test_backtest_run_cv.py tests/test_backtest_metrics.py tests/test_backtest_report.py -q` → 0 failures, count at least 56 plus the tests added here (baseline was 56 passed, fact 1).
- `uv run pytest tests/test_universe_filtered_factor.py -k backtester_runs_with_wrapped_factors -q` → passes (it asserts the exact run-directory file set, fact 5).
- `git diff --stat -- quantlab/backtest/ quantlab/base/config.py` → empty (no engine, selection or config change).
- `grep -c '^from quantlab\|^import quantlab' quantlab/utils/backtest_report.py` → `0` (the leaf property holds).
- No metric name drives the table. Proved by the behavioural tests above, NOT by a source scan: the module docstring is required to explain the generic-table rule, so a metric name legitimately appears there as prose. As an advisory tripwire only, `grep -vE '^\s*#' quantlab/utils/backtest_report.py | grep -niE 'traded_notional|sharpe|calmar|sortino'` should print nothing outside docstring prose; a hit inside a dict key, a row list or a conditional is a real failure, a hit in explanatory prose is not.
- Open a generated `report.html` and confirm by eye that the window dates, training window and in-sample/out-of-sample ranges are readable as text, the metric table is populated, and the log button turns the equity curve from a near-flat line into a readable slope.
</verification>

<success_criteria>
- The backtest window, bar count, bar interval, training window(s), in-sample range(s) and out-of-sample ranges appear on the page as text, matching the same run's `metrics.json` byte for byte.
- The whole / in-sample / out-of-sample metric blocks appear on the page as a table rendered from the keys actually present, with absent keys as dashes and no hardcoded metric list.
- Equity carries a working log/linear toggle and the multiple of initial capital; drawdown, monthly return bars and forced-liquidation markers are present; notes still appear.
- The report remains one self-contained `report.html`, the run directory file set is unchanged, and `quantlab/utils/backtest_report.py` remains a leaf module.
- The gate suites pass with no regression against the 56-passed baseline.
</success_criteria>

<output>
Create `.planning/quick/260915-sxx-improve-the-backtest-html-report-dates-m/260915-sxx-SUMMARY.md` when done.

Record in it: the final pass counts for the gate command versus the 56-passed baseline, the exact trace set the page now carries, and the decision record for the two judgement calls — raw equity `y` with the multiple as `customdata` (rather than normalising `y`), and linear as the default axis with log one click away.
</output>
