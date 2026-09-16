---
phase: quick-260916-hro
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/utils/backtest_report.py
  - quantlab/backtest/engine_vectorbt.py
  - quantlab/base/backtest.py
  - tests/test_backtest_engine.py
  - tests/test_backtest_report.py
  - tests/test_backtest_persistence.py
  - example/backtest.md
autonomous: true
requirements: [D-01, D-02, D-03]

estimate:
  tokens: 120000
  raw_tokens: 60000
  tasks: 3
  confidence: low        # no calibration samples for this repo yet

must_haves:
  truths:
    - "The up triangle on the equity curve sits at the DEEPEST bar of the deepest drawdown (its valley), not at the bar the drawdown began (D-01)."
    - "The down triangle still sits at the recovery bar, and a drawdown still open at the last bar still says it has not recovered."
    - "`bars` is `end_idx - valley_idx`, and nothing on the page claims that number equals Max Drawdown Duration (D-01)."
    - "No trace named `liquidation` is drawn on the page, even for a run that liquidated (D-02)."
    - "That same run still writes `liquidations.json` with byte-identical content, and the run directory still holds exactly seven entries (D-02)."
    - "The three left-hand axis titles no longer overlap each other or their neighbouring rows (D-03)."
  artifacts:
    - quantlab/utils/backtest_report.py
    - quantlab/backtest/engine_vectorbt.py
    - quantlab/base/backtest.py
    - tests/test_backtest_engine.py
    - tests/test_backtest_report.py
    - tests/test_backtest_persistence.py
    - example/backtest.md
  key_links:
    - "`VectorBtBacktester._drawdown_span` payload -> `write_backtest_report(drawdown_span=...)` -> the two triangle traces."
    - "`VectorBtBacktester._drawdown_span` payload -> `BaseBacktester._report_summary` -> the `Deepest drawdown` row of the dates-and-setup block. The picture and the text must state the SAME two bars."
    - "`simulation.liquidations` -> `liquidations.json` (SURVIVES) and, separately, -> the chart (REMOVED). Severing the second must not touch the first."
---

<objective>
Three independent changes to the backtest `report.html`, all requested directly by the user.

1. **D-01** — the pair of triangles on the equity curve now spans **valley -> recovery** instead of start -> recovery. The up triangle marks the deepest bar of the deepest drawdown.
2. **D-02** — the red X liquidation markers are removed from the chart. `liquidations.json` is untouched.
3. **D-03** — the left-hand axis titles stop overlapping.

Purpose: the report currently answers a question the user did not ask (where a drawdown began) and hides the one they did (how long from the bottom back to even), draws a marker they do not want, and renders three axis titles on top of each other.

Output: a report whose triangles mean valley-to-recovery, whose chart carries no liquidation markers while the run directory is unchanged, and whose axis titles are legible.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md

@quantlab/utils/backtest_report.py
@quantlab/backtest/engine_vectorbt.py
@tests/test_backtest_report.py
@tests/test_backtest_engine.py
@tests/test_backtest_persistence.py
@example/backtest.md
</context>

<facts>
Measured at planning time against `f6d2061` (clean tree). Trust a file over this list if they disagree, and record the discrepancy in the SUMMARY.

**F-1 — Baseline: 141 passed in 64.6s.** Measured, not assumed:
`uv run pytest tests/test_backtest_persistence.py tests/test_backtest_run.py tests/test_backtest_run_cv.py tests/test_backtest_metrics.py tests/test_backtest_report.py tests/test_backtest_engine.py tests/test_backtest_contracts.py -q`

**F-2 — `valley_idx` exists.** The drawdown records columns are exactly
`['id', 'col', 'peak_idx', 'start_idx', 'valley_idx', 'end_idx', 'peak_val', 'valley_val', 'end_val', 'status']`
(vectorbt 5.24.1 / plotly 5.24.1). Reading `valley_idx` is structurally identical to the `start_idx` read already there.

**F-3 — THE TRAP: the existing engine fixture cannot prove D-01.** Measured indices of the deepest record:

| fixture | start | valley | end | end-start | end-valley |
|---|---|---|---|---|---|
| `DEEPEST_IS_NOT_LONGEST` | 2 | **2** | 3 | 1 | **1** |
| `DEEPEST_NEVER_RECOVERS` | 4 | 6 | 6 | 2 | **0** |

On `DEEPEST_IS_NOT_LONGEST` start and valley **coincide**, so `end - start == end - valley == 1`: every existing assertion passes whether or not the change was made. A NEW fixture is mandatory (F-4). On `DEEPEST_NEVER_RECOVERS` the valley IS the last bar, so `bars` legitimately becomes `0` (was `2`).

**F-4 — A validated fixture where the two differ.** `[100, 120, 115, 110, 90, 95, 105, 125]` on `pd.bdate_range("2024-01-01", ...)` produces exactly ONE drawdown record: `start=2, valley=4, end=7`, `depth=-0.25`, `status=1` (recovered). `end-start=5`, `end-valley=3`. Bar labels: start `2024-01-03`, valley `2024-01-05`, end `2024-01-10`. Measured in this tree.

**F-5 — D-03 root cause is VERTICAL, not horizontal.** The emitted plotly layout carries `margin={'b': 140}` and **no `height`**, so the div falls back to plotly.js's default 450px. With the default top margin (100) and the overridden bottom (140), the plotting area is only `450 - 100 - 140 = 210px`, which `row_heights=[0.54, 0.23, 0.23]` splits into:

| row | domain | px at height 450 | px at height 900 |
|---|---|---|---|
| 1 equity | `[0.5032, 1.0]` | **104** | 328 |
| 2 drawdown | `[0.2516, 0.4632]` | **44** | 140 |
| 3 monthly return | `[0.0, 0.2116]` | **44** | 140 |

A y-axis title is rotated 90 degrees, so its text length is measured against the axis **height**. The 34-character row-1 title is roughly 220px of text in a 104px axis, and even `monthly return` is roughly 91px in a 44px axis. **Shortening the row-1 title alone cannot fix rows 2 and 3** — the dominant lever is an explicit `height`. A left margin would not help: the collision is between vertically stacked titles, not between a title and its tick labels. Both `height=900` figures above were rendered and measured.

**F-6 — `liquidations.json` is written independently of the chart.** In `quantlab/base/backtest.py` both persist blocks write `liquidations.json` from `simulation.liquidations` (lines ~1789 and ~1937) BEFORE and separately from the `write_backtest_report(...)` call (lines ~1799 and ~1958). Severing the chart argument cannot affect the file.

**F-7 — the run-directory file set is asserted as an EXACT set in four places**, and `liquidations.json` is in all four: `tests/test_backtest_persistence.py:97`, `tests/test_backtest_run.py:123`, `tests/test_backtest_run_cv.py:601`, `tests/test_universe_filtered_factor.py:936`. None of these four may change.

**F-8 — the chart-side liquidation references are exactly these.** Everything else named `liquidation` is about the JSON artifact and must not be touched:
- `quantlab/utils/backtest_report.py`: module docstring line ~13, the `liquidations` parameter + its docstring bullet (~136-138), the `_add_liquidations(...)` call (~166), the helper itself (~245-276), the `SPAN_COLOUR` comment (~279-281), and the precedent reference in `_add_drawdown_span`'s docstring (~298).
- `quantlab/base/backtest.py`: the two `liquidations=simulation.liquidations` report arguments ONLY (~1810, ~1969).
- `tests/test_backtest_report.py`: 308, 377-412, 523, 543-555.
- `tests/test_backtest_persistence.py`: module docstring line 29, and line 721 inside the exact trace set.
- `example/backtest.md`: line 455 only (the phrase inside the `report.html` row).

**F-9 — no consumers outside these files.** `SPAN_COLOUR` is used only at its own definition site and line 341 of the same module; the trace names `deepest_drawdown_start` / `deepest_drawdown_end` appear only in `tests/test_backtest_report.py` and `tests/test_backtest_persistence.py`; `_add_liquidations` appears only in `quantlab/utils/backtest_report.py` plus one test docstring. (Hits under `.planning/` are historical records of the prior quick task and are not edited.)

**F-10 — `timeout` and `gtimeout` do not exist on this machine.** Never wrap a verify command in one: it exits 0 having run nothing and reads as a false green. All verify commands below use relative paths so they work inside an isolated worktree.
</facts>

<decisions>
**H-1 — `write_backtest_report` DROPS its `liquidations=` parameter; it does not keep it and ignore it.**
A keyword argument that is accepted and silently does nothing is a trap: the two call sites would keep handing over real records and the next reader would spend an afternoon working out why no markers appear. Deleting the parameter turns that into an immediate `TypeError` at the one place it could happen. The blast radius is fully measured (F-8, F-9) and small: one helper, one call, one parameter, one docstring bullet, two call-site arguments. It also makes the data/picture split explicit in the code — `liquidations.json` is still written from `simulation.liquidations` a few lines above the report call (F-6), so the two now visibly diverge at exactly the right place. The two variadic monkeypatch stubs (`tests/test_backtest_persistence.py:323`, `tests/test_backtest_run_cv.py:375`) take `**kwargs`, so they are unaffected; if either is not variadic the suites fail loudly rather than silently.

**H-2 — the payload key `start` is RENAMED to `valley`, and the trace `deepest_drawdown_start` to `deepest_drawdown_valley`.**
`BaseBacktester._drawdown_span`'s docstring documents the key set `{"start", "end", "bars", "depth", "recovered"}` as a contract. Leaving a key named `start` that holds the valley would make that documented contract false, and would re-create the exact confusion the user just had to ask about. The leaf module reads every key with `.get` and drops an endpoint it cannot place, so an old-shaped payload degrades to a missing marker rather than an exception (this is the D-6 / T-sxx-03 discipline, preserved).

**H-3 — D-03 uses TWO levers: an explicit `height=900` (primary) and a shortened row-1 title (secondary). No left-margin change.**
Justified by F-5: the overlap is vertical. At the default 450px the plotting area is 210px and rows 2 and 3 are 44px tall, so `monthly return` overflows its own axis no matter what row 1 says. `height=900` gives 328/140/140px, which fits all three titles with margin. The row-1 title is separately shortened from `value (x initial capital on hover)` to `value` because the parenthetical merely describes the hover template the equity trace already carries — it is redundant, not informative. A left margin is deliberately NOT added: it separates a title from its tick labels horizontally and does nothing about titles colliding with the rows above and below.

**H-4 — the summary label is relabelled `Deepest drawdown (valley to recovery)`.**
Its text has to change anyway (it currently reads as the full span), and a row still labelled `span` beside triangles that mean valley-to-recovery would mislead exactly the reader this task is for.

**H-5 — no negative `grep` acceptance gates.** Every gate below is a pytest run. A `grep -c` gate on these files would count the docstring and comment prose that this very plan rewrites, making the gate self-invalidating.
</decisions>

<tasks>

<task type="tracer">
  <name>Task 1: Move the drawdown span to valley -> recovery (D-01)</name>
  <files>quantlab/backtest/engine_vectorbt.py, quantlab/base/backtest.py, quantlab/utils/backtest_report.py, tests/test_backtest_engine.py, tests/test_backtest_report.py, tests/test_backtest_persistence.py, example/backtest.md</files>
  <behavior>
    - Engine, on the new fixture `[100,120,115,110,90,95,105,125]` (F-4): `span["valley"] == "2024-01-05"`, `span["end"] == "2024-01-10"`, `span["bars"] == 3`, and non-vacuity `start_idx != valley_idx` so `end-start (5) != end-valley (3)`.
    - Engine, on `DEEPEST_NEVER_RECOVERS`: `span["recovered"] is False` and `span["bars"] == 0` (the valley IS the last bar, F-3).
    - Engine, on `DEEPEST_IS_NOT_LONGEST`: the deepest record is still chosen by DEPTH, never duration.
    - Report leaf: the up triangle trace is named `deepest_drawdown_valley`, keeps `triangle-up`, and sits on the equity value at the valley bar; the down triangle is unchanged.
    - Report leaf: the end marker's hover states the bar count in trading days, does not render a calendar timedelta, and does NOT claim to be Max Drawdown Duration.
    - Report leaf: a never-recovered span still says it has not recovered; an endpoint off the equity axis still drops only its own marker; a payload missing keys still renders the page.
    - Persistence: on a real run the up triangle's x equals the bar of `argmin(value / cummax - 1)` from `equity.zarr` — an independently derived valley.
  </behavior>
  <action>
In `quantlab/backtest/engine_vectorbt.py` `_drawdown_span`: read `valley_idx` alongside `start_idx`/`end_idx` using the same per-column `.to_numpy()[row]` then `int(...)` idiom already there (the row-wise `.iloc` float coercion note in the docstring still applies). Return `valley` (the `_bar_label` of `timestamps[valley]`) in place of `start`, and set `bars` to `end - valley`. Keep the existing bounds guard and extend it to cover the valley index so an out-of-range index still returns `None` rather than raising. Per D-01, the down endpoint stays `end_idx`.

Rewrite that method's Chinese docstring: it currently states the span length is what vectorbt's own duration measures. That is no longer true and is the single most load-bearing piece of prose in this change. Say instead that the span runs from the deepest bar to recovery, that `bars` is `end_idx - valley_idx` counted in trading days (bar counts, never calendar days), and state explicitly that this number is NOT `Max Drawdown Duration` and will usually be smaller. Record that this reverses the earlier quick task quick-260915-v6i's own third decision — which deliberately chose `start_idx` so the two triangles' distance would equal `Max Drawdown Duration` — and why it was reversed (asked directly, the user chose the valley). Note that decision belongs to that task's numbering and is unrelated to D-03 in this plan.

In `quantlab/base/backtest.py`: update the base `_drawdown_span` contract docstring so the documented key set reads `{"valley", "end", "bars", "depth", "recovered"}` with the valley described as the deepest bar. In `_report_summary`, read `.get('valley')` instead of `.get('start')` and relabel the row to `Deepest drawdown (valley to recovery)` per H-4; keep the existing `DASH`-on-missing handling and the `trading days` wording verbatim.

In `quantlab/utils/backtest_report.py` `_add_drawdown_span`: change the endpoints tuple's first entry to key `valley`, trace name `deepest_drawdown_valley`, symbol `triangle-up`, and hover text saying the deepest drawdown bottoms at this bar. Change the recovered tail to count from the deepest point rather than from the start, and keep the not-recovered branch counting bars so far. Update the helper's docstring and the `drawdown_span` bullet in `write_backtest_report`'s docstring to describe the valley-to-recovery contract; delete the sentence claiming the bar count is what the engine's duration measures. English in this file, Chinese in the two above (CLAUDE.md).

In `quantlab/backtest/engine_vectorbt.py` `_report_notes`: rewrite the second note so it says the up triangle is the deepest bar (the valley) and the down triangle the recovery bar, that the length is counted in trading days, and that Max Drawdown Duration is a different measurement of a possibly different episode. Obey the existing constraint recorded in that docstring: no angle brackets, ampersands, double quotes or apostrophes anywhere in note text, because each note is asserted verbatim in the page after HTML escaping. Spell out contractions and possessives.

Tests. In `tests/test_backtest_engine.py` add the F-4 fixture as a module constant with a comment recording its measured indices and WHY it is needed (on `DEEPEST_IS_NOT_LONGEST` start and valley coincide, so that series cannot distinguish the two rules). Rewrite `test_bars_is_the_chosen_records_end_minus_start_and_labels_are_bar_labels` to use the new fixture, assert `span["bars"] == end - valley` against independently derived indices, assert the `valley`/`end` bar labels, and carry an explicit non-vacuity assertion that `start != valley` on this series. Add `bars == 0` to the never-recovered test with a comment naming the reason. Leave `test_the_span_is_the_deepest_record_and_never_the_longest_one` passing but reword its assertion message, which currently describes the record's duration rather than the marked span.

In `tests/test_backtest_report.py` update the `_span()` helper's key and every `deepest_drawdown_start` reference to the new trace name, including the off-axis parametrize ids. Strengthen the hover test with an assertion that the text does not name Max Drawdown Duration.

In `tests/test_backtest_persistence.py` update both trace names, and in `test_report_marks_the_deepest_drawdown_on_the_persisted_equity_curve` add the independent-derivation assertion: recompute `value / np.maximum.accumulate(value) - 1.0` from `equity.zarr`, take its `argmin`, and assert the up triangle's x is that bar's label. Update the `Deepest drawdown` regex in `test_the_page_states_the_deepest_drawdown_span_in_words` to the new label.

In `example/backtest.md` (Chinese) rewrite the triangle sentence in the `report.html` table row and the standalone paragraph below it: the up triangle is the deepest bar, the down triangle is recovery, the distance is in trading days, and it is NOT `Max Drawdown Duration` — now for two independent reasons (different episode, and the span no longer starts where the drawdown started). Keep the existing note that a never-recovered drawdown is labelled as such and that a run with no drawdown record draws no triangles.
  </action>
  <verify>
    <automated>uv run pytest tests/test_backtest_engine.py tests/test_backtest_report.py -q</automated>
  </verify>
  <done>The engine returns a valley-keyed payload whose `bars` is `end_idx - valley_idx`, proved on a fixture where valley and start are different bars; the up triangle is named and placed at the valley; no docstring, note, hover string or doc paragraph any longer equates the span with Max Drawdown Duration.</done>
  <reversibility rating="reversible">A payload key, a trace name and prose; revertible by one `git revert`.</reversibility>
</task>

<task type="auto">
  <name>Task 2: Drop the liquidation markers from the chart, keep the JSON (D-02)</name>
  <files>quantlab/utils/backtest_report.py, quantlab/base/backtest.py, tests/test_backtest_report.py, tests/test_backtest_persistence.py, example/backtest.md</files>
  <behavior>
    - A run that really liquidated renders a page carrying NO trace named `liquidation`.
    - That same run still writes `liquidations.json`, whose parsed content still matches `result.simulation.liquidations` symbol-for-symbol and timestamp-for-timestamp.
    - The run directory still holds exactly the same seven entries, `liquidations.json` among them (F-7).
    - The persisted page's trace set is exactly `{equity, drawdown, monthly_return, deepest_drawdown_valley, deepest_drawdown_end}`.
  </behavior>
  <action>
In `quantlab/utils/backtest_report.py`: delete the `_add_liquidations` helper and its call from the figure assembly, and delete the `liquidations` parameter and its docstring bullet from `write_backtest_report` (H-1). Update the module docstring's item 4, which currently describes the equity row as carrying forced-liquidation markers. Rewrite the `SPAN_COLOUR` comment: its stated justification is that the colour must differ from the liquidation red, and that comparison no longer exists on the page — keep the colour value unchanged (no visual churn) and state the real remaining reason, that the two triangles share one colour because they are two ends of one measurement. Reword the `_add_drawdown_span` docstring sentence that cites the deleted helper as its precedent: state the drop-do-not-raise rule directly instead of pointing at a function that is gone. Keep every `.get`-everything / drop-bad-endpoint behaviour exactly as it is — this module runs inside the run's staging directory and an exception here deletes the entire run (T-hro-01).

In `quantlab/base/backtest.py`: remove the `liquidations=simulation.liquidations` argument from BOTH `write_backtest_report` calls. **Do not touch either `write_json_atomically(run_dir / "liquidations.json", ...)` block** — those run a few lines earlier and are the artifact the user is keeping (F-6). Leave the surrounding Chinese docstrings that enumerate the run-directory contents unchanged; they describe the file, which still exists.

In `tests/test_backtest_report.py`: delete `test_liquidation_markers_sit_on_the_equity_curve`, `test_a_liquidation_off_the_equity_axis_is_dropped_not_raised` and `test_no_liquidations_means_no_marker_trace` — the parameter they exercise is gone, so they can no longer be written. Delete `test_the_span_markers_are_not_the_liquidation_colour`, which has lost its comparison object, but first fold its surviving half (both triangles share one colour) into `test_the_span_draws_one_triangle_at_each_end_on_the_equity_curve` so that assertion is not lost. Remove the `liquidations=` argument from `test_every_trace_carries_a_name`. Reword the docstring at line ~523 that cites the deleted helper as a precedent. Add a test asserting the leaf figure's trace set is exactly the five names above, so a future trace cannot reappear unnoticed.

In `tests/test_backtest_persistence.py`: remove `"liquidation"` from the exact trace set at line ~721 and update the comment above it, which currently explains that the fixture run liquidates and therefore carries the markers. Keep the comparison as `==`; do not weaken it to a subset. Update the module docstring's line 29 parenthetical about liquidation markers on the equity row — and change ONLY that line; the other JSON-side references in this file (lines 5, 88, 97, 186, 262, 267-274) are about the artifact and stay. Add the pairing test that is the whole point of D-02: for the fixture run that really liquidates, assert in ONE test that the page carries no `liquidation` trace while `liquidations.json` still parses and still matches `result.simulation.liquidations`.

In `example/backtest.md`: in the `report.html` table row, remove only the phrase saying the equity panel carries forced-liquidation markers. Leave the `liquidations.json` row, the forced-liquidation section around line 227, and the example output listing the seven files exactly as they are.
  </action>
  <verify>
    <automated>uv run pytest tests/test_backtest_report.py tests/test_backtest_persistence.py -q</automated>
  </verify>
  <done>No page carries a `liquidation` trace; a liquidating run's `liquidations.json` is unchanged and the run directory still has its seven entries; `write_backtest_report` no longer accepts a `liquidations` argument and neither call site passes one.</done>
  <reversibility rating="reversible">Deletes a chart trace and a parameter; no persisted artifact changes shape.</reversibility>
</task>

<task type="auto">
  <name>Task 3: Make the axis titles fit (D-03)</name>
  <files>quantlab/utils/backtest_report.py, tests/test_backtest_report.py</files>
  <behavior>
    - The emitted layout carries an explicit `height` (not None), so the figure no longer inherits plotly's 450px default.
    - The row-1 y-axis title is `value`; rows 2 and 3 keep `drawdown` and `monthly return`.
    - At the chosen height each row's pixel height exceeds the rendered length of its own rotated title: roughly 328/140/140px against roughly 33/52/91px of text (F-5).
    - The notes annotation at paper `y=-0.16` still lands inside the 140px bottom margin.
  </behavior>
  <action>
In `quantlab/utils/backtest_report.py`'s `update_layout`, add `height=900` beside the existing `margin={"b": 140}` (H-3). Shorten the row-1 `update_yaxes` title from the 34-character string to `value`; leave rows 2 and 3 alone.

Add a short comment at the layout block recording the measured reason, because this is a number someone will later be tempted to delete as arbitrary: a y-axis title is rotated 90 degrees so its length is measured against the axis height; without an explicit height the div is 450px, the top and bottom margins take 240 of it, and `row_heights` then leaves rows 2 and 3 at roughly 44px each — shorter than their own titles. State the resulting per-row heights at 900.

Note for the docstring: the row-1 title's dropped clause described the hover text, and the equity trace's `hovertemplate` already shows the multiple of initial capital, so no information is lost — say so where the module docstring explains that the multiple rides along as `customdata`.

In `tests/test_backtest_report.py` add one test that parses the embedded layout (the existing `_traces` helper shows the `raw_decode` idiom; the layout is the next JSON object after the trace array) and asserts: `height` is set and is not None, the three y-axis titles are exactly `value` / `drawdown` / `monthly return`, and each row's domain times the plotting area is at least as large as a stated floor. Keep the floor derived from the numbers in F-5 rather than hardcoding a magic pixel count without explanation.
  </action>
  <verify>
    <automated>uv run pytest tests/test_backtest_report.py -q</automated>
  </verify>
  <done>The layout emits an explicit height, the row-1 title is `value`, and a test locks both plus the per-row pixel budget so a future `row_heights` or margin change that re-creates the overlap fails loudly.</done>
  <reversibility rating="reversible">Two layout values and a title string.</reversibility>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| run staging directory -> final run directory | The report is written INSIDE the staging directory. An exception during report writing aborts the run and deletes every artifact already staged, not just the report. |
| engine payload -> leaf report module | `_drawdown_span`'s dict crosses from the vectorbt-aware engine into a module that must stay engine-agnostic and dependency-light. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-hro-01 | Denial of Service | `_add_drawdown_span`, `_drawdown_span` | high | mitigate | Inherits T-sxx-03/T-v6i-02: every payload key is read with `.get`, an endpoint the equity axis does not carry drops only its own marker, and the engine bounds-checks indices (now including `valley_idx`) and returns `None` rather than raising. Task 1 must not introduce a bare `records[...][row]` index without the guard. |
| T-hro-02 | Tampering | `liquidations.json` | high | mitigate | The data artifact must survive the chart deletion byte-identically. Mitigated by leaving both `write_json_atomically` blocks untouched (F-6), by the four exact run-directory set assertions (F-7), and by the new pairing test in Task 2 that asserts no chart trace AND unchanged JSON in one place. |
| T-hro-03 | Information Disclosure | report prose, hover text, `example/backtest.md` | medium | mitigate | The span number changes meaning. Stale prose claiming it equals Max Drawdown Duration would misinform every future reader of a real run. Task 1 enumerates all four prose sites and the hover test asserts the claim is absent. |
| T-hro-04 | Tampering | `quantlab/utils/backtest_report.py` leaf status | medium | mitigate | The module must keep importing only stdlib, pandas, xarray and plotly. No new import is required by any task; `test_the_module_is_a_leaf` remains the gate. |
| T-hro-05 | Spoofing | notes text | low | accept | Note text is asserted verbatim after HTML escaping, so a stray apostrophe or angle bracket silently rewrites the note into entities and reddens the suite. Accepted because it fails loudly and the constraint is already recorded in `_report_notes`' docstring. |

No package-manager installs occur in this plan, so no package legitimacy gate applies.
</threat_model>

<verification>
Run the full backtest gate, from the repository root, with relative paths and **no** `timeout` wrapper (F-10):

```
uv run pytest tests/test_backtest_persistence.py tests/test_backtest_run.py tests/test_backtest_run_cv.py tests/test_backtest_metrics.py tests/test_backtest_report.py tests/test_backtest_engine.py tests/test_backtest_contracts.py -q
```

Baseline is **141 passed** (F-1). Zero failures is mandatory. The count will move, and the SUMMARY must state the final number and account for every delta — expected: minus 4 in `test_backtest_report.py` (three liquidation chart tests plus the colour-comparison test, Task 2), plus the new tests added in Tasks 1, 2 and 3.

Also run the four exact run-directory set assertions, which must be untouched by this work:

```
uv run pytest tests/test_universe_filtered_factor.py -q
```

Finally, confirm by eye that the three axis titles no longer collide, by opening any `report.html` produced during the run above.
</verification>

<success_criteria>
- The up triangle marks the valley and the down triangle the recovery bar, proved on a fixture where those are different bars from the drawdown's start.
- `bars == end_idx - valley_idx`, asserted against independently derived indices; no hover text, docstring, note or doc paragraph claims that number is Max Drawdown Duration.
- The never-recovered case still labels itself as not recovered.
- A liquidating run's page carries no `liquidation` trace while its `liquidations.json` is unchanged, asserted together.
- The run directory still holds exactly seven entries.
- The layout carries an explicit height and a short row-1 title, locked by a test.
- The seven backtest suites pass with zero failures; the final test count is stated and every delta from 141 is explained.
</success_criteria>

<output>
Create `.planning/quick/260916-hro-mark-the-drawdown-span-from-valley-to-re/260916-hro-SUMMARY.md` when done.
</output>
