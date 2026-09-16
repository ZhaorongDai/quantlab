# Phase 03.8 — Context handoff

Written 2026-09-16 at the end of the session that shipped six backtest-report quick
tasks. Everything below was **measured**, not inferred. Where a measurement
contradicted an earlier assumption, the correction is recorded — several of these
assumptions were wrong the first time.

## What this phase collects

Three items the user asked for that were NOT done in the quick tasks. They are
independent and can be split into separate plans.

### 1. In/out-of-sample delta column in the metric table

Add a column showing `out_of_sample - in_sample` to the report's metric table.

**The design point that must be settled first:** the table is *generic* — rows are
derived by walking whatever keys the metrics mapping carries at render time
(`quantlab/utils/backtest_report.py`, `_BLOCKS = ("whole", "in_sample",
"out_of_sample")`). A delta column therefore cannot assume numbers:

- duration metrics (`Avg Winning Trade Duration`, `Max Drawdown Duration`) are
  rendered as **strings** by `to_jsonable` (`pd.Timedelta` → str, `NaT` → null);
- ratio metrics (Sharpe, Calmar, Profit Factor) are not meaningfully subtractable;
- a key present in one block and absent from the other must not produce a bogus 0.

Skip by type rather than trying to subtract everything. Do not hardcode a metric
list — that genericity is load-bearing and is locked behind behavioural tests
(`test_a_mapping_with_every_shipped_key_deleted_still_renders`).

### 2. Monthly-return heatmap

A year × month matrix, sourced from the same `returns` series the existing
`monthly_return` bars use. That series is grouped with `index.to_period("M")` —
**not** a resample alias, because the `M`/`ME` alias was renamed across pandas
versions while `to_period` is stable. Keep that.

Nothing named `heatmap` exists in the module today (verified: grep count 0).

### 3. Positions-only trade metrics + `order_count` in `whole`

**User decision (2026-09-16):** "如果 positions.Total Trades 更好 就不要用 Total Trades 了".

Today `_engine_stats` emits BOTH: the lot-level (`exittrades`) set at the top level
and a `positions` nested block (quick task 260915-udx). This phase switches the top
level to the positions view and stops emitting the lot-level set.

**Why it matters, measured on two independent real runs:**

| run | lot-level Win Rate | position-level Win Rate |
|---|---|---|
| 2025 single year | 57.27% | 50.74% |
| 2020–2026 six years | 57.27% | 50.81% |

A systematic **~6.5 point overstatement**, consistent across both — not sample
noise. Cause: equal-weight target-percent rebalancing means only a *winner* ever
needs trimming back to its 0.5% target, and vectorbt's exit-trades view records
every partial trim as its own closed trade. Confirming detail: the *losing* side is
nearly unaffected (−6.28% vs −6.32%) while the winning side is badly distorted —
exactly what "only winners get trimmed" predicts.

**The companion change is not optional.** The `whole` block currently has **no
execution-activity count at all** — `order_count` / `traded_notional` exist only in
the sliced `in_sample` / `out_of_sample` blocks (`_period_record_stats`, around
`quantlab/base/backtest.py:1479-1530`). Dropping the lot-level trades would leave
the whole-window view with no answer to "how many fills actually happened".
`Total Fees Paid` and `turnover` survive and cover cost, but not count.

**API facts, measured:** vectorbt is **1.1.0**. `pf.replace(trades_type="positions")`
is the only correct switch and is already in use. Two things that do NOT work and
must not be reintroduced: `pf.stats(settings=dict(trades_type=...))` (the string
never appears in `Portfolio.stats`) and `pf.get_trades(trades_type=...)` (its
signature is `(group_by=None, **kwargs)` — it reads the *instance* attribute).
Mutating the global `vbt.settings` is rejected: it leaks across the process.

## Constraints that apply to all three

- **`quantlab/utils/backtest_report.py` is a LEAF**: stdlib, pandas, xarray and
  plotly only, **zero `quantlab.*` imports**, locked by
  `test_the_module_is_a_leaf`. It has no vectorbt import and must not gain one.
- **The report is written INSIDE the run's staging directory.** An exception while
  writing it deletes the ENTIRE run — every artifact, not just the report. Hence the
  `.get`-everything / drop-bad-input discipline throughout. A drawing bug must never
  cost a trained model or a completed backtest.
- **The run-directory file set is asserted as an EXACT set in FOUR files**:
  `tests/test_backtest_persistence.py:97`, `tests/test_backtest_run.py:123`,
  `tests/test_backtest_run_cv.py:601`, `tests/test_universe_filtered_factor.py:936`.
  Adding or removing any run-directory file turns all four red.
- **`metrics.json` must parse under a strict JSON parser that rejects `NaN`** —
  non-finite values are written as `null` by `to_jsonable`.
- **Report note text is HTML-escaped and asserted VERBATIM in the page.** Any note
  you add must contain no `<`, `>`, `&`, `"` or apostrophe; write contractions and
  possessives out in full, or the verbatim assertion goes red.
- **wandb summary keys and the report table both flatten nested dicts generically**
  (`_flatten_numeric` for wandb, `_flatten` for the report). A new nested metrics
  group needs no downstream change — proved twice already.

## State at the end of the predecessor work

All six quick tasks are complete and pushed: `260915-sxx` (report readability),
`260915-udx` (position-level metrics alongside), `260915-v6i` (deepest-drawdown
triangles), `260915-weq` (CCC early-stopping criterion), `260916-gs8` (importance
charts on the wandb dashboard), `260916-hro` (valley-to-recovery span, liquidation
markers dropped, axis titles fixed).

**Test baseline:** the seven backtest suites are at **140 passed, 0 failed**
(`test_backtest_persistence test_backtest_run test_backtest_run_cv
test_backtest_metrics test_backtest_report test_backtest_engine
test_backtest_contracts`). `tests/test_universe_filtered_factor.py` is at 84 passed.
`tests/test_xgb_model.py` is at 50 passed. Re-measure before editing; do not inherit
these numbers on trust.

**Environment:** `timeout` and `gtimeout` do NOT exist on this machine — never wrap
a verify command in one, it exits 0 having run nothing and reads as a false green.
`tests/conftest.py` sets `OMP_NUM_THREADS=1` on darwin (torch/xgboost libomp clash);
do not remove it.

## Two open items that are NOT part of this phase

- **A human still has to open a `report.html` and look at it.** Outstanding since
  260915-sxx. Tests lock pixel budgets and trace names; they cannot judge whether
  the page reads well.
- **The −92.75% single-trade loss is unexplained.** It is identical under all three
  trade views, so it is a real full-position loss rather than a lot-accounting
  artifact — most likely one of the 11 forced liquidations. Worth checking whether
  it is an unadjusted corporate action or a genuine delisting fill.
