# Phase 03.11 — Deferred Items

Out-of-scope discoveries logged during execution. Per the executor scope
boundary, these are NOT fixed by the plan that found them.

---

## D-03.11-12-A — 55 pre-existing full-suite failures: tests still import the
moved top-level `ingest_*.py` shells

**Found during:** Plan 03.11-12, Task 3 (full regression)
**Status:** open — pre-existing at this phase's base commit, untouched by 03.11-12

`uv run pytest -q --ignore=tests/test_factor_hierarchy.py
--ignore=tests/test_crsp_rebuild_measurements.py` reports
**55 failed, 1675 passed, 1 skipped**. Every one of the 55 is a
`ModuleNotFoundError: No module named 'ingest_tiingo'` /
`'ingest_binance_spot'`, or the `FileNotFoundError` twin of it raised by tests
that read the shell's SOURCE text to assert on its argparse wiring.

| Test file | Failures |
|---|---|
| `tests/test_ingest_shells.py` | 14 |
| `tests/test_ingest_tiingo_universe_wiring.py` | 11 |
| `tests/test_data_dir_cli.py` | 10 |
| `tests/test_ingest_conversion_gate.py` | 6 |
| `tests/test_volume_guard.py` | 5 |
| `tests/test_factor_kunquant.py` | 3 |
| `tests/test_chunked_ingest.py` | 2 |
| `tests/test_spot_dataset.py` | 2 |
| `tests/test_entry_point_contracts.py` | 1 |
| `tests/test_universe.py` | 1 |

**Cause.** Commit `d07f06e "modify docs and move scripts to dir"` moved the
top-level ingest shells into a directory. The tests above still import them by
their old top-level module name, or open `'ingest_tiingo.py'` relative to the
repo root. `git ls-tree --name-only e95ad6b | grep -i ingest` is EMPTY at this
phase's base commit, so the breakage predates every 03.11 plan.

**Why 03.11-12 did not fix it.** None of the ten files reference
`CrspTickerLookup`, `_spell` or `predict_panel`'s guard — the three files
03.11-12 changed (`quantlab/dataset/crsp_tickers.py`, plus a docstring in
`quantlab/base/model.py` and one line of `example/wrds_crsp.md`) cannot reach
them. Repointing ten test files at relocated CLI shells is a distinct piece of
work with its own blast radius, and folding it into a gap-closure plan about a
ticker sidecar would hide it inside an unrelated commit.

**Verification that 03.11-12 is green on its own surface:**
`uv run pytest -q tests/test_crsp_ticker_sidecar.py tests/test_model_predict_panel.py
-p no:cacheprovider` -> **56 passed**.

---

## D-03.11-12-B — `test_cross_sectional_zscore.py` intermittently deadlocks the
whole suite in KunQuant's `~MultiThreadExecutor()`

**Found during:** Phase 03.11 wave-1 post-merge gate (orchestrator, not the executor)
**Status:** open — intermittent; pre-existing, unrelated to 03.11-12's changes

The post-merge full-suite run hung for **2h02m at 0.2% CPU** and had to be
killed. `sample(1)` on the stuck interpreter shows the main thread parked in:

```
KunRunner.abi3.so
  kun::MultiThreadExecutor::~MultiThreadExecutor()
    std::thread::join()  ->  _pthread_join  ->  __ulock_wait
```

while every worker sits in `kun::MultiThreadExecutor::workerMain(int)` ->
`std::condition_variable::wait`. The destructor joins workers that were never
signalled to exit — a teardown race inside KunQuant, not a slow test.

**Located at:** `tests/test_cross_sectional_zscore.py`, the module-scoped
`stream_outputs` fixture. Its last statement is `del ctx, executor` (:152),
which is what invokes the destructor. Progress stopped at collected test #510,
`test_stream_matches_pandas[z_raw]`, the first test to consume that fixture.

**Not a thread-count problem.** That fixture builds its executor with
`kr.createMultiThreadExecutor(4)` (:139). The deadlock reproduces at **4**
threads, so lowering the count does not remove it. (The separate `njobs=128`
default is logged as D-03.11-12-C below; it is a different issue.)

**Intermittent, not deterministic.** Plan 03.11-12's executor ran the same
suite to completion earlier in this phase (55 failed, 1675 passed, 1 skipped).
Only the orchestrator's re-run hung.

**Workaround used for the wave-1 gate:** added
`--ignore=tests/test_cross_sectional_zscore.py` to the configured
`workflow.test_command` ignore list for that one run only. The config was NOT
changed. A real fix needs the fixture's teardown made deterministic (or the
KunQuant race fixed upstream), not a permanent ignore — a permanently ignored
file is an untested cross-sectional z-score op.

---

### CORRECTION (2026-09-22, quick task 260922-edi post-merge gate)

**The "Located at" line above is too narrow. This is not one fixture's bug.**

Measured today: a full-suite run that ALREADY carried
`--ignore=tests/test_cross_sectional_zscore.py` hung for **69 minutes**, and
`sample(1)` on it showed the SAME stack —

```
kun::MultiThreadExecutor::~MultiThreadExecutor()  (in libKunRuntime.dylib)
  -> _pthread_join            [blocked forever]
```

so the excluded file cannot be the cause. Progress stopped at **45%**, which
maps to collected test ~788 of 1752 — the `tests/test_factor_kunquant.py`
region (`--collect-only` on the same command, checked afterwards).

**What this changes.** The race is in KunQuant's executor teardown, reachable
from ANY caller, not from one module-scoped fixture. `test_cross_sectional_zscore.py`
is where it was FIRST seen, not where it lives. Ignoring that one file does not
make a suite run safe.

**A lead worth following, stated as a lead and not as a finding.** Unlike
`test_cross_sectional_zscore.py`, which builds its own executor with an explicit
`kr.createMultiThreadExecutor(4)`, `test_factor_kunquant.py` reaches KunQuant
through `quantlab/base/factor.py:291` and `:337`, both of which call
`kr.createMultiThreadExecutor(self.config.njobs)` — and `njobs` defaults to
**128** (D-03.11-12-C, below). So the two known deadlock sites differ by a
factor of 32 in thread count.

That does NOT overturn the "Not a thread-count problem" paragraph above, which
remains true as written: the race reproduces at 4. What it adds is that the
128-thread path also deadlocks, so D-03.11-12-C is no longer obviously
independent of this entry — line 116's "this is NOT the cause of D-03.11-12-B"
was established against the 4-thread site only.

**Not proven, and deliberately not claimed:** which executor instance actually
deadlocked. The stack names the destructor, not its owner. Confirming it needs a
run under `faulthandler` or a root-enabled `py-spy dump`, neither of which was
available in that session.

**Unrelated to the refactor that found it.** Of quick task 260922-edi's 26
changed files, the only one under `quantlab/factor/` is `universe_filter.py`,
whose entire diff is one docstring path string with no executable change. The
re-run of the identical command completed in 263s.

---

## D-03.11-12-C — `FactorConfig.njobs` defaults to 128

**Found during:** Phase 03.11 wave-1 post-merge gate
**Status:** open — operator flagged it directly during execution

`quantlab/base/config.py:307` declares `njobs: int = 128`. It reaches
`kr.createMultiThreadExecutor(self.config.njobs)` at three call sites:
`quantlab/base/factor.py:291`, `quantlab/base/factor.py:337`, and
`quantlab/factor/universe_filter.py:386`. On a workstation this spawns 128
KunQuant worker threads per executor regardless of core count.

The operator's instruction during wave 1 was explicit: do not use 128 threads.

**Why it was not changed here.** It is production code in `quantlab/base/`,
outside the declared `files_modified` of both 03.11-12 and 03.11-13, and
CLAUDE.md forbids direct repo edits outside a GSD workflow's scope. It needs
its own plan that picks the replacement default (a CPU-count-derived value is
the obvious candidate) and re-measures factor throughput, since lowering it
changes performance characteristics the phase never benchmarked.

**Note:** this is NOT the cause of D-03.11-12-B — see that entry.

---

## D-03.11-18-A — the regression gate collects one more test in the main
checkout than in any worktree, because it parametrizes over the repo root

**Found during:** Plan 03.11-18, Task 3 (regression gate)
**Status:** open — a measurement artefact, not a defect in any test

`tests/test_entry_point_contracts.py:44` builds its parametrization at module
scope:

```python
ENTRY_POINTS = sorted(REPO_ROOT.glob("*.py"))
```

That glob reads the **working tree**, not git. The main checkout currently
carries an untracked root-level scratch file, `jerry_query_data.py`
(`git ls-files --error-unmatch jerry_query_data.py` → "did not match any
file(s) known to git"), which no worktree ever contains — a worktree is a
checkout of the commit, and untracked files do not travel with it. So
`test_every_main_guarded_entry_point_imports` collects one extra (passing)
case in the main tree:

| Where the gate ran | collected | failed | passed | skipped |
|---|---|---|---|---|
| main checkout | 1757 | 55 | 1701 | 1 |
| any worktree | 1756 | 55 | 1700 | 1 |

**Why this is worth a ledger entry.** The same unexplained ±1 was reported as
an open question by **03.11-09** and again by **03.11-10** (its Issues §2:
"全仓 passed 比 orchestrator 给的基线少 1 个（1667 vs 1668）… 同一处未归因的
±1"), and it cost 03.11-18 a round of forensics to rule out a regression. It
is now attributed. Anyone comparing an orchestrator-measured baseline (main
tree) against an executor-measured number (worktree) should expect exactly
this offset, and neither number is wrong.

**Not fixed here.** Two candidate fixes exist and both are scope of their own:
restrict the glob to git-tracked files (`git ls-files '*.py'`), which changes
what the contract test *claims to cover* — it currently asserts that **every**
root-level script is import-safe, including a scratch file someone dropped in
— or leave the glob and accept the offset as documented. Which of those is
right is a question about the contract, not about this phase, and
`tests/test_entry_point_contracts.py` is outside 03.11-18's `files_modified`.

---

## D-03.11-UAT-A — a raise inside `_backtest_window` swallows the D-27
"data changed" diagnostic

**Found during:** UAT round 3 (verify-work), while disposing of test 4
**Status:** resolved by quick task `260921-w6r` — `run()` and `run_cv()` now call `_compare_fingerprints_on_failure()` on the exception path (a `partial=True` comparison that never replaces the original exception); locked by `tests/test_backtest_rebuild.py::test_a_raise_inside_the_window_still_reports_the_changed_data`
**Operator decision (2026-09-21):** fix separately AFTER 03.11 closes, not inside it

**The gap.** `run()` calls `self._backtest_window(...)` (`backtest.py:391`) and only
then `self._compare_fingerprints()` (`:392`). But the factor fingerprints are already
recorded *inside* that call — `_align_and_predict` (`:1065`) runs `_redate_factors` →
`_record_factor_fingerprints` (`:1053`) **before** `model.predict_panel` (`:1068`), and
`_load_prices` (`:774`) records the price fingerprint after it. So anything that raises
between those points leaves the comparison unexecuted although its inputs exist and
already differ. `run_cv()` has the same shape (`:488`, `:514-516`).

**Consequence.** When a data change is large enough to break the run, the operator gets
the downstream error and **no** indication that the data changed — which is precisely
what D-27 exists to tell them. Realistic triggers: `DLModel._assert_symbol_types_match`
(a ticker-era checkpoint against a PERMNO panel), `"the feature panel lacks N of the
symbols this model was trained on"`, or `"no price bars between ..."`.

**Reproduction (measured, not reasoned).** Synthetic store, one symbol dropped at the
same path so both fingerprints really differ:

| | fingerprints recorded at raise time | warnings emitted |
|---|---|---|
| control — no raise | — | **2** (`factor[0]:PastReturnFactor`, `price_dataset`; digest + n_symbols) |
| probe — raise inside `_backtest_window` | `['factor[0]:PastReturnFactor']`, differing | **0** |

**Why it is NOT a 03.11 defect.** `git log -L 391,392:quantlab/base/backtest.py` attributes
both lines to `8465f1a` / `79e342d` (2026-09-15, phase 03.7). 03.11 never touched the
ordering. It surfaced here only because test 4 asked what the fingerprint warnings do.

**Sketch of the fix (not applied).** Give `_compare_fingerprints` a `partial` mode that
skips the "present in expected but not read by this run" branch — under a partial
comparison that means "not read *yet*", not "not read" — and call it from an exception
path around the window computation, then re-raise. The diagnostic itself must be guarded
so it can never replace the real exception.
