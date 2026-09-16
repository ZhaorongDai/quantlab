---
phase: quick-260915-udx
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/backtest/engine_vectorbt.py
  - quantlab/base/backtest.py
  - tests/test_backtest_engine.py
  - tests/test_backtest_persistence.py
  - example/backtest.md
autonomous: true
requirements: [QUICK-260915-udx]

estimate:
  tokens: 45000
  raw_tokens: 30000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "A backtest run reports position-level trade statistics alongside the existing lot-level ones; both sets are present, neither replaces the other."
    - "A reader of report.html or metrics.json can tell which trade numbers are lot-level and which are position-level without opening the source."
    - "The position-level numbers are genuinely the positions view, not a second copy of the exit-trades view."
    - "In-sample / out-of-sample trade counts stay exit-trades-based and say so, so nobody reads them as position counts."
  artifacts:
    - quantlab/backtest/engine_vectorbt.py
    - tests/test_backtest_engine.py
    - tests/test_backtest_persistence.py
    - example/backtest.md
  key_links:
    - "_engine_stats -> _compute_metrics whole block -> to_jsonable -> metrics.json (strict JSON, non-finite as null)"
    - "_engine_stats -> _compute_metrics whole block -> backtest_report._flatten -> report.html metric table rows"
    - "_engine_stats -> _compute_metrics whole block -> _flatten_numeric -> wandb summary keys"
    - "VectorBtBacktester._report_notes -> metrics['notes'] AND write_backtest_report(notes=...) -> html.escape"
---

<objective>
Emit position-level trade statistics alongside the existing exit-trades (lot-level)
statistics in the vectorbt engine's `whole` metric block, and make the distinction
legible to a reader of the report.

Purpose: with equal-weight target-percent rebalancing, every partial trim of a winner
is recorded by vectorbt's default `exittrades` view as its own closed trade. On a real
measured run (244 bars, top_n=200, rebalance_periods=5) that reports a 57.27% win rate
where the position-level win rate is 50.74%. Both views are legitimate; today only the
lot-level one is reported, and it is reported under names (`Win Rate [%]`,
`Profit Factor`) that a reader takes for stock-picking statistics.

Output: a `positions` sub-dict inside the `whole` block carrying the 13 trade-derived
metrics computed under `trades_type="positions"`, a report note stating what each set
means, and tests that go red if the positions block is secretly a second copy of the
exit-trades block.
</objective>

<scope_decision>
**Narrow scope, deliberately: the `whole` block only.**

`in_sample` / `out_of_sample` keep deriving `closed_trade_count` / `open_trade_count`
from `simulation.trades`, which is built in `_simulate` from `pf.trades.records_readable`
— i.e. the exit-trades view. They are NOT converted, for three reasons:

1. Converting them requires `SimulationResult` to carry a second records dataset built
   in `_simulate`, plus `_period_record_stats` changes — a materially larger blast
   radius than a quick task should take.
2. The defect that motivated this work is a misleading *ratio* (win rate, profit factor,
   avg win/loss). The sliced blocks carry no ratios — only two counts.
3. No source artifact requires the sliced blocks.

The cost of the narrow choice is that `closed_trade_count` and `open_trade_count` in
the sliced blocks are lot counts, not position counts. That is not left implicit:
Task 3 states it in `example/backtest.md` and in the `_period_record_stats` docstring,
so a later reader does not take them for position counts.
</scope_decision>

<measured_context>
Measured at planning time against the live tree and a live interpreter. Trust the file
over this block if they disagree, and record the discrepancy.

**Gate baseline (measured, this machine, commit 581bf1d):** the seven backtest suites
(`test_backtest_persistence` `test_backtest_run` `test_backtest_run_cv`
`test_backtest_metrics` `test_backtest_report` `test_backtest_engine`
`test_backtest_contracts`) = **116 passed, 0 failed, 51.87s**. The task brief quoted 88
for five of those suites; 116 is the same tree with `test_backtest_engine` and
`test_backtest_contracts` added. No discrepancy.

**`timeout` does not exist on this machine** (macOS/zsh). Do not wrap any verify command
in it; it exits 0 having run nothing, which reads as green.

**vectorbt API, confirmed live in this venv:** version `1.1.0`; `Portfolio.replace`
exists; `trades_type` IS a `Portfolio.__init__` parameter and is NOT a `from_orders`
parameter; `Portfolio.get_trades` has signature `(group_by=None, **kwargs)` so it reads
the instance attribute and cannot be passed a per-call type; the global default
`vectorbt.settings["portfolio"]["trades_type"]` is `exittrades`. Therefore
`pf.replace(trades_type="positions")` is the only correct mechanism — per-call kwargs
on `stats()` / `get_trades()` are silently ignored, and mutating the global settings
mapping would leak across the process.

**Downstream is already generic — confirmed by reading, not assumed:**
- `quantlab/utils/backtest_report.py:330 _flatten` recurses into nested dicts producing
  `dotted.path` keys, and `_BLOCKS = ("whole", "in_sample", "out_of_sample")` at line 75.
  A `positions` sub-dict therefore renders as `positions.<metric>` rows with **no report
  edit**. The module docstring states row derivation is load-bearing.
- `quantlab/base/backtest.py:1594 _flatten_numeric` recurses generically, so the wandb
  summary gains `whole/positions/<metric>` with no edit.
- `metrics.json` is written as `to_jsonable(metrics)` wholesale
  (`backtest.py:1748`), and `to_jsonable` maps NaN/+inf/-inf to `None`
  (`quantlab/utils/jsonable.py:56`). A nested dict inherits strict-JSON safety
  automatically.

**Test constraints, read from the files:**
- `tests/test_backtest_engine.py:557` asserts `"Total Return [%]" in whole` (membership,
  not an exact key set) and `[key for key in whole if "Benchmark" in key] == []`. That
  iterates **top-level** keys only. A top-level key named `positions` is safe.
- `tests/test_backtest_metrics.py:580` asserts named keys and strict-JSON parse.
- `tests/test_backtest_contracts.py:102` asserts the exact `__abstractmethods__` sets.
  `_report_notes` is concrete on `BaseBacktester`, so overriding it changes nothing there.
- `tests/test_backtest_persistence.py:690`
  (`test_report_has_equity_and_drawdown_and_shades_the_in_sample_range`) asserts **every** note returned by
  `_report_notes()` appears **verbatim** in report.html, and `:812-815` asserts
  `metrics["notes"] == backtester._report_notes()` plus that the joined text contains
  `short`, `borrow`, `optimistic`. Both are membership/substring assertions, so
  **appending** a note is safe — but see the escaping hazard below.

**Escaping hazard (discovered at planning time, load-bearing):**
`backtest_report.py:428 _notes_section` renders each note through `html.escape` with
default `quote=True`. A note containing `<`, `>`, `&`, `"` or `'` is escaped in the page
and then does NOT appear verbatim, turning `test_report_carries...` RED. The new note
text must contain **no angle brackets, no ampersand, no double quote and no apostrophe**.

**`_engine_stats` is also the benchmark path** (`backtest.py:1577`
`metrics["benchmark"] = self._engine_stats(benchmark)`), but `benchmark_dataset` is
refused by the base config setter this phase (D-08), so there is no live second call.
</measured_context>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

@quantlab/backtest/engine_vectorbt.py
@quantlab/base/backtest.py
@tests/test_backtest_engine.py
@tests/test_backtest_persistence.py
</context>

<conventions>
- **Language per file.** Docstrings and comments in `quantlab/backtest/engine_vectorbt.py`
  and `quantlab/base/backtest.py` are Chinese — match that. `example/backtest.md` is
  Chinese — match that. Test docstrings in `tests/` are English — match that. The user-facing
  **note string** itself is English, because the note it is appended to
  (`BaseBacktester._report_notes`) is English and the two render as one list.
- Do not change engine arithmetic, selection logic, or the config schema. No new config
  field — the positions view is always emitted, never a toggle.
- No DataFrame crosses a layer boundary; pandas stays inside `_simulate` as it does today.
- `uv run` for every command.
</conventions>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Positions statistics through the whole stack, one path</name>
  <files>quantlab/backtest/engine_vectorbt.py, tests/test_backtest_engine.py</files>
  <read_first>
    quantlab/backtest/engine_vectorbt.py lines 48-78 (STATS_METRICS) and 287-296 (_engine_stats);
    tests/test_backtest_engine.py line 557 (the no-Benchmark test, for the fixture idiom it uses).
  </read_first>
  <behavior>
    - A real `run()` produces `result.metrics["whole"]["positions"]`, a dict.
    - Its key set is exactly the vectorbt display names of the 13 trade-derived metrics.
    - Every one of its keys also exists as a top-level key of `whole` (the lot-level twin),
      so the two views are directly comparable row by row.
    - Its `Total Closed Trades` equals the closed-record count taken independently from
      `simulation.native.positions.records_readable` — this is what goes red if the
      implementation forgot to switch the trades type and merely recomputed exit trades.
    - On the chosen fixture the lot-level and position-level closed counts genuinely
      DIFFER, asserted explicitly, so the previous check cannot pass vacuously.
    - Top-level keys are untouched: no rename, no removal, and still no key containing
      the substring that the existing D-08 test forbids.
  </behavior>
  <action>
    Add a class constant beside `STATS_METRICS` naming only the trade-derived metrics —
    total_trades, total_closed_trades, total_open_trades, open_trade_pnl, win_rate,
    best_trade, worst_trade, avg_winning_trade, avg_losing_trade,
    avg_winning_trade_duration, avg_losing_trade_duration, profit_factor, expectancy.
    Deliberately exclude every portfolio-level metric (sharpe, calmar, omega, sortino,
    max_dd, total_return, exposure, fees, start/end/period, start_value/end_value): the
    trades type does not affect them, so recomputing them would be waste that invites drift.

    In `_engine_stats`, keep the existing call exactly as it is, then compute a second
    stats frame from `simulation.native.replace(trades_type="positions")` restricted to
    the new constant, and attach it to the returned dict under the key `positions`. This
    lives in the engine layer, not in `_compute_metrics`, because the trades-type concept
    is vectorbt-specific and the base layer stays engine-agnostic; it follows the existing
    nested-sub-dict precedent that `_compute_metrics` already sets for turnover.

    Build a FRESH settings dict for each of the two `stats()` calls rather than sharing one
    object between them, so neither call can mutate what the other reads.

    Do not touch the global vectorbt settings mapping — the switch must be instance-scoped,
    because a process-wide change would silently alter every other portfolio in the process.
    Do not pass the trades type as a per-call keyword to `stats()` or `get_trades()`: both
    ignore it (measured), which would produce a positions block that is a byte-for-byte
    copy of the exit-trades block while looking correct.

    Extend the class docstring with a Chinese paragraph stating the two views, that the
    top-level trade metrics are the lot-level ones, and why both are reported.

    Write the tests first and watch them fail before implementing. Pick or build a fixture
    run in which at least one holding is trimmed without being closed — if the existing
    run fixture does not produce diverging counts, adjust the weights so it does, and
    assert the divergence, because a non-diverging fixture makes the whole lock vacuous.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_backtest_engine.py -q 2>&amp;1 | tail -5</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab && grep -v '^\s*#' quantlab/backtest/engine_vectorbt.py | grep -c 'vbt\.settings' | grep -qx 0 &amp;&amp; echo "OK: no global settings mutation"</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab && grep -v '^\s*#' quantlab/backtest/engine_vectorbt.py | grep -cE 'replace\(\s*trades_type\s*=' | grep -qvx 0 &amp;&amp; echo "OK: instance-scoped trades-type switch present"</automated>
  </verify>
  <done>
    `tests/test_backtest_engine.py` passes with the new tests included; the positions block
    exists with exactly the 13 display keys, every key has a top-level twin, its closed
    count matches an independent derivation from the positions accessor, and the fixture is
    proved non-degenerate by an explicit lot-vs-position divergence assertion.
  </done>
  <reversibility rating="reversible">Additive: one class constant, one extra stats call, one new dict key. Removing it restores the previous output exactly.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: The note that stops a reader mistaking lot-level for position-level</name>
  <files>quantlab/backtest/engine_vectorbt.py, tests/test_backtest_persistence.py</files>
  <read_first>
    quantlab/base/backtest.py lines 1582-1591 (`_report_notes`, the D-21 mechanism);
    quantlab/utils/backtest_report.py lines 428-435 (`_notes_section`, the escaping);
    tests/test_backtest_persistence.py lines 700-725 and 800-816 (the two existing note locks).
  </read_first>
  <behavior>
    - `VectorBtBacktester._report_notes()` returns the base note plus one new note.
    - The base short-side note survives verbatim, so the existing test asserting the joined
      text contains short / borrow / optimistic stays green.
    - The new note appears verbatim in report.html (it survives HTML escaping untouched)
      and verbatim in `metrics.json` under `notes`.
    - `metrics.json` still parses under a parser that rejects the NaN and Infinity tokens,
      with the nested positions block present and every leaf a number or null.
    - report.html carries metric-table rows whose names begin with the positions prefix.
  </behavior>
  <action>
    Override `_report_notes` on `VectorBtBacktester` so it returns the base implementation's
    list plus one English note. The note must say: the trade metrics at the top level are
    vectorbt exit trades, i.e. lot-level, where each partial trim of a holding is counted as
    its own closed trade and the win rate is inflated as a result; the prefixed rows are the
    position-level view, one entry-to-flat round trip per symbol. Keep it to one or two
    sentences.

    Hard constraint on the note text: it must contain no angle bracket, no ampersand, no
    double quote and no apostrophe. Every note is rendered through HTML escaping and an
    existing test asserts each note appears verbatim in the page, so any of those five
    characters turns that test red. Write out possessives and contractions in full.

    Do not rename, move or drop any existing metric key — several tests assert the current
    names directly, and the whole point is that both views stay side by side.

    Add the downstream locks to `tests/test_backtest_persistence.py` using its existing run
    fixture: the note present in both artifacts, the strict-JSON parse with the nested
    positions block, every positions leaf number-or-null, and the report carrying
    positions-prefixed rows.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_backtest_persistence.py tests/test_backtest_report.py -q 2>&amp;1 | tail -5</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_backtest_persistence.py -q -k "test_report_has_equity_and_drawdown_and_shades_the_in_sample_range" 2>&amp;1 | tail -5</automated>
  </verify>
  <done>
    A run's report.html and metrics.json both carry the new note verbatim alongside the
    short-side note; metrics.json strict-parses with the nested positions block; the report
    metric table shows positions-prefixed rows; `tests/test_backtest_persistence.py` and
    `tests/test_backtest_report.py` pass.
  </done>
  <reversibility rating="reversible">One overridden method and additive tests.</reversibility>
</task>

<task type="auto">
  <name>Task 3: Documentation — and stating what the narrow scope leaves lot-level</name>
  <files>example/backtest.md, quantlab/base/backtest.py</files>
  <read_first>
    example/backtest.md lines 330-360 (the metrics.json key table and the slice-block paragraph);
    quantlab/base/backtest.py lines 1474-1528 (`_period_record_stats`).
  </read_first>
  <action>
    In `example/backtest.md`, update the `whole` row of the metrics.json key table to state
    that it also carries a nested positions block with the position-level trade metrics, and
    extend the paragraph describing the slice blocks to state plainly that their two trade
    counts are the lot-level view and are NOT converted to positions — naming the reason
    (the sliced counts come from the records dataset built during simulation). A reader who
    compares a sliced count against the position-level whole-window count must find the
    answer here rather than infer a bug.

    Add one Chinese sentence to the `_period_record_stats` docstring in
    `quantlab/base/backtest.py` making the same statement at the place a maintainer reads
    before changing those counts. Docstring text only — no behaviour change in this file.

    Report the real measured numbers in the SUMMARY: the lot-level and position-level win
    rate and closed-trade count from an actual run, so the improvement is recorded as a
    measurement rather than a claim.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_backtest_persistence.py tests/test_backtest_run.py tests/test_backtest_run_cv.py tests/test_backtest_metrics.py tests/test_backtest_report.py tests/test_backtest_engine.py tests/test_backtest_contracts.py -q 2>&amp;1 | tail -5</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab && grep -c 'positions' example/backtest.md | grep -qv '^0$' &amp;&amp; echo "OK: docs mention the positions block"</automated>
  </verify>
  <done>
    `example/backtest.md` documents the positions block and states that the sliced trade
    counts remain lot-level; `_period_record_stats` says the same in its docstring; the full
    seven-suite gate is green at 116 or more passed with 0 failed.
  </done>
  <reversibility rating="reversible">Documentation only.</reversibility>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| metrics/notes -> report.html | Strings assembled in Python are interpolated into an HTML document opened in a browser |
| in-process vectorbt global settings | A process-wide configuration mapping shared by every portfolio object in the run |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-udx-01 | Tampering | `_report_notes` text -> `backtest_report._notes_section` | low | mitigate | Note text is a hardcoded literal, never interpolated from data, and passes through the existing `html.escape`; Task 2 additionally forbids the five characters escaping would rewrite, verified by an automated gate |
| T-udx-02 | Tampering | vectorbt global settings mapping | medium | mitigate | The trades-type switch is instance-scoped via `replace`; Task 1 gate asserts zero non-comment references to the global settings mapping in the engine module |
| T-udx-03 | Information disclosure | metrics.json / report.html | low | accept | Both artifacts already carry the full statistics set for this run and are written into the local run directory; the positions block adds no new category of information |
| T-udx-SC | Tampering | package installs | low | accept | No package is installed, added or upgraded by this plan; vectorbt 1.1.0 is already resident and pinned by the existing environment |
</threat_model>

<verification>
1. Full gate green: the seven backtest suites at **116 or more passed, 0 failed**
   (measured baseline 116; the new tests only add).
2. A real `run()` yields `metrics["whole"]["positions"]` whose numbers differ from the
   top-level trade metrics on a fixture proved to trim a holding.
3. `metrics.json` strict-parses (NaN and Infinity tokens rejected) with the nested block.
4. `report.html` shows both the top-level trade rows and the positions-prefixed rows, and
   prints both notes.
5. No metric key was renamed, moved or removed; no config field was added.
</verification>

<success_criteria>
- Both metric sets are emitted; neither replaces the other.
- The positions block is provably the positions view, locked by an independent derivation
  plus a non-vacuity assertion.
- A reader of the report can tell the two apart without reading source.
- The sliced blocks are documented as lot-level in both the example doc and the docstring.
- Zero changes to engine arithmetic, selection or config schema.
</success_criteria>

<output>
Create `.planning/quick/260915-udx-add-positions-level-trade-metrics-alongs/260915-udx-SUMMARY.md` when done.
</output>
