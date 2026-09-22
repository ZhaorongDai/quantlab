---
phase: quick-260922-lu2
plan: 01
type: execute
wave: 1
depends_on: []
autonomous: true
requirements: [LU2-01, LU2-02, LU2-03, LU2-04]
files_modified:
  # moved (git mv)
  - quantlab/backend.py
  - quantlab/registry.py
  - quantlab/universe.py
  - quantlab/dataset/_support/cleaning.py
  - quantlab/dataset/_support/masking.py
  - quantlab/dataset/_support/session_calendar.py
  - quantlab/acquisition/_support/sql_volume.py
  - quantlab/acquisition/_support/inspector.py
  # created (0 bytes)
  - quantlab/dataset/_support/__init__.py
  - quantlab/acquisition/_support/__init__.py
  # reference rewrites (89 files measured by the sweep below)
  - CLAUDE.md
  - example/README.md
  - example/acquisition.md
  - example/backend.md
  - example/chunking.md
  - example/constituent.md
  - example/dataset.md
  - example/factor.md
  - example/registry.md
  - example/universe.md
  - example/wrds_crsp.md
  - example/wrds_taq.md
  - quantlab/acquisition/alpaca.py
  - quantlab/acquisition/tiingo.py
  - quantlab/acquisition/wrds/__init__.py
  - quantlab/base/acquisition.py
  - quantlab/base/backtest.py
  - quantlab/base/chunking.py
  - quantlab/base/config.py
  - quantlab/base/constituent.py
  - quantlab/base/coverage.py
  - quantlab/base/data.py
  - quantlab/base/factor.py
  - quantlab/base/model.py
  - quantlab/base/rebuild.py
  - quantlab/config/__init__.py
  - quantlab/dataset/constituent.py
  - quantlab/dataset/crsp/membership.py
  - quantlab/dataset/crsp/rebuild.py
  - quantlab/dataset/crsp/tickers.py
  - quantlab/dataset/nbbo/__init__.py
  - quantlab/dataset/nbbo/resample.py
  - quantlab/dataset/spot.py
  - quantlab/dataset/stock.py
  - quantlab/enums/data.py
  - quantlab/factor/universe_filter.py
  - quantlab/ml_model/backend.py
  - quantlab/utils/cli.py
  - quantlab/utils/symbol_axis.py
  - scripts/backfill_us_equity_full.py
  - scripts/ingest_alpaca.py
  - scripts/ingest_tiingo.py
  - scripts/ingest_us_equity.py
  - scripts/ingest_wrds_crsp.py
  - scripts/ingest_wrds_taq.py
  - scripts/refresh_us_equity_universe.py
  - tests/conftest.py
  - tests/test_acquisition_progress.py
  - tests/test_backend_head.py
  - tests/test_backend_indexes.py
  - tests/test_backend_overwrite.py
  - tests/test_chunked_ingest.py
  - tests/test_chunked_panel_estimate.py
  - tests/test_cleaning.py
  - tests/test_constituent_panel.py
  - tests/test_crsp_identity.py
  - tests/test_crsp_rebuild.py
  - tests/test_crsp_ticker_sidecar.py
  - tests/test_crsp_tracer.py
  - tests/test_data_dir_cli.py
  - tests/test_extensibility_contract.py
  - tests/test_factor_update.py
  - tests/test_ingest_conversion_gate.py
  - tests/test_ingest_shells.py
  - tests/test_ingest_tiingo_universe_wiring.py
  - tests/test_ingest_wrds_crsp.py
  - tests/test_ingest_wrds_taq.py
  - tests/test_model_predict_panel.py
  - tests/test_nbbo_dataset.py
  - tests/test_nbbo_resampler.py
  - tests/test_registry_convert.py
  - tests/test_session_calendar.py
  - tests/test_source_inspector.py
  - tests/test_source_registry.py
  - tests/test_sql_volume_guard.py
  - tests/test_symbol_axis_contract.py
  - tests/test_symbol_axis_widening.py
  - tests/test_ticker_pattern_reconciliation.py
  - tests/test_universe.py
  - tests/test_universe_mask.py
  - tests/test_variable_axis_widening.py
  - tests/test_volume_guard.py
  - tests/test_widen_crash_residue.py
  - tests/test_wrds_crsp_acquisition.py
  - tests/test_wrds_vendor_seam.py
  # live .planning pointers (historical records stay untouched -- see Task 4)
  - .planning/STATE.md
  - .planning/WINDOWS.md
  - .planning/todos/pending/2026-09-07-normalize-ticker-delimiter-between-membership-panel-and-pric.md
  - .planning/todos/pending/2026-09-07-no-ticker-rename-mapping-between-membership-history-and-pric.md
  - .planning/todos/pending/2026-09-08-an-empty-zarr-store-records-symbol-as-float64.md
  - .planning/todos/pending/2026-09-21-cross-vendor-permno-ticker-merge-is-silently-all-nan.md
  - .planning/todos/pending/2026-09-07-re-point-stale-former-root-paths-in-python-docstrings.md

estimate:
  tokens: 150000
  raw_tokens: 75000
  tasks: 4
  confidence: low

must_haves:
  truths:
    - "`ls quantlab/dataset/` shows only datasets: spot, stock, constituent, crsp, nbbo, plus `_support/`."
    - "`ls quantlab/acquisition/` shows only acquisitions: alpaca, tiingo, wrds, plus `_support/`."
    - "`import quantlab.universe` runs exactly ONE package `__init__.py` (`quantlab/__init__.py`), and it is 0 bytes."
    - "Every import order of registry / universe / each vendor / each wrds submodule enumerates ['alpaca', 'tiingo', 'wrds'] from a cold interpreter."
    - "The failing node-ID set of the test suite is unchanged in BOTH directions."
    - "CLAUDE.md states which rule now governs `base/X.py` <-> `<layer>/X.py`, so no two contradictory conventions are left undocumented."
  artifacts:
    - quantlab/backend.py
    - quantlab/registry.py
    - quantlab/universe.py
    - quantlab/dataset/_support/__init__.py
    - quantlab/dataset/_support/cleaning.py
    - quantlab/dataset/_support/masking.py
    - quantlab/dataset/_support/session_calendar.py
    - quantlab/acquisition/_support/__init__.py
    - quantlab/acquisition/_support/sql_volume.py
    - quantlab/acquisition/_support/inspector.py
  key_links:
    - "quantlab/registry.py bottom module-object imports <-> quantlab/acquisition/{alpaca,tiingo,wrds} decorator imports (the cycle)"
    - "tests/test_volume_guard.py structural arm <-> quantlab/universe.py source + quantlab/__init__.py emptiness"
    - "tests/test_source_inspector.py _FORBIDDEN_INSPECTOR_MODULES <-> the moved registry's NEW dotted name"
    - "tests/test_wrds_vendor_seam.py REGISTRY_SOURCE Path <-> quantlab/registry.py on disk"
---

<objective>
Make every top-level entry of `quantlab/dataset/` a dataset and every top-level entry of
`quantlab/acquisition/` an acquisition, so browsing either directory is a menu of complete,
usable things.

Eight support modules leave that level. Three are really their own LAYER and move up to
`quantlab/` siblings; five are genuinely internal and sink into a `_support/` subpackage.
~306 references follow them.

Purpose: an operator who opens `quantlab/dataset/` should not have to know which of ten files
is importable as a dataset. Today five of the ten are not.

Output: 8 `git mv` moves, 2 new 0-byte `_support/__init__.py` files, ~306 rewritten
references across 89 files, 3 new/extended permanent gates, and a CLAUDE.md that resolves the
convention this breaks.
</objective>

<decisions_made>

## D-1 — `quantlab/universe.py`, NOT `quantlab/universe/` (deliberate, and I agree)

The option text the operator picked drew the destination as a directory. This plan writes a
flat module instead. That is a change to the chosen option and is flagged here rather than
made silently.

Why I agree with the orchestrator's simplification, on top of the "a package needs a FAMILY"
convention the previous task adopted:

- `universe.py` has no `universe_*.py` siblings. A directory would add one level for one file.
- **A package `__init__` is the exact thing the volume-guard invariant is trying to minimise.**
  `quantlab/universe/__init__.py` would put a SECOND package `__init__` back on the
  `import quantlab.universe` path — re-creating, one directory lower, precisely the hazard that
  `quantlab/acquisition/registry.py:728-737` spends ten lines warning about. The flat module is
  not merely simpler here, it is strictly safer.

If `universe.py` (113,915 bytes — the largest module in the repo) is later split into a family,
converting it to a package with the `__init__` IS the entry module is a one-commit change and
the convention already covers it. **Flagged, not done here.**

## D-2 — the `base/X.py` <-> `<layer>/X.py` pairing: which rule now governs

Moving `dataset/backend.py` to `quantlab/backend.py` breaks the mirrored-filename pairing for
`backend` while `quantlab/ml_model/backend.py` keeps it. The governing rule this plan adopts
and writes into CLAUDE.md:

> **The ABC lives in `quantlab/base/`. The concrete implementation lives with its CONSUMERS —
> not in a directory that mirrors the ABC's filename.** The mirrored filename was a coincidence
> of the first two cases, never the rule.
>
> - `base/backend.py:ModelBackend` <-> `ml_model/backend.py:MlBackend`: only the model layer
>   consumes it, so it lives in the model layer.
> - `base/backend.py:DataBackend` <-> `quantlab/backend.py` (`XrBackend`/`PlBackend`): **seven
>   quantlab modules across five layers** import it — `config/__init__.py`, `universe.py`,
>   `factor/universe_filter.py`, `base/factor.py`, `base/model.py`, `base/data.py`,
>   `base/backtest.py` (measured). No single layer owns it, so it is a top-level sibling.
> - `base/constituent.py` <-> `dataset/constituent.py` survives untouched, because a
>   constituent dataset genuinely IS a dataset.

**Flagged, not done:** by the operator's own "every file is one complete thing" logic,
`quantlab/ml_model/backend.py` is support rather than a model and has the same smell. The
operator scoped this task to `dataset/` and `acquisition/`; expanding it would be scope
creep. Recorded here so the next reader sees it was considered, not missed.

## D-3 — the 0-byte-`__init__.py` invariant has TRANSFERRED and WIDENED

**Measured before planning: `quantlab/__init__.py` is 0 bytes** (`wc -c quantlab/__init__.py`
-> `0`). So moving `universe.py` to `quantlab/universe.py` does not weaken the volume guard's
structural arm — it **strengthens** it:

| | package `__init__`s run by the import | can hide a transitive vendor import |
|---|---|---|
| before | `quantlab/` + `quantlab/acquisition/` | 2 places |
| after | `quantlab/` | 1 place |

The invariant now covers THREE files, for two different guarantees:

- `quantlab/__init__.py` — for `quantlab.universe` (the volume guard: no acquisition client is
  constructible there, whatever the call order) AND for `quantlab.registry` / `quantlab.backend`.
- `quantlab/acquisition/__init__.py` — still, for `alpaca` / `tiingo` / `wrds`.
- `quantlab/acquisition/_support/__init__.py` — NEW, and load-bearing for the same reason:
  `tests/test_source_inspector.py`'s structural arm is an `ast` scan of `inspector.py`'s OWN
  source (`_FORBIDDEN_INSPECTOR_MODULES`), and a non-empty `_support/__init__.py` would drag a
  vendor module in where that scan cannot see it.
- `quantlab/dataset/_support/__init__.py` — 0 bytes for consistency; no guarantee rides on it.

**This invariant is currently held by PROSE ALONE.** Task 1 and Task 3 give it a gate.
</decisions_made>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/quick/260922-edi-package-wrds-crsp-and-nbbo-into-import-entry-poin/SUMMARY.md
@quantlab/acquisition/registry.py
@tests/test_volume_guard.py
@tests/test_source_inspector.py
@tests/test_wrds_vendor_seam.py
</context>

<shared_apparatus>

Everything below is set up ONCE, before Task 1, and reused verbatim by every task. **Write the
helper to the scratchpad, never into the repo.** All paths are RELATIVE to the repo root: this
runs in an isolated worktree, and an absolute `/Users/daizhaorong/...` path would measure
unchanged main-tree code and report a false green.

## A. The move table (this is the whole refactor, in data)

| # | from | to | measured refs / files |
|---|---|---|---|
| 1 | `quantlab/dataset/backend.py` | `quantlab/backend.py` | 46 / 28 |
| 2 | `quantlab/acquisition/registry.py` | `quantlab/registry.py` | 93 / 30 |
| 3 | `quantlab/acquisition/universe.py` | `quantlab/universe.py` | 86 / 32 |
| 4 | `quantlab/dataset/cleaning.py` | `quantlab/dataset/_support/cleaning.py` | 36 / 18 |
| 5 | `quantlab/dataset/masking.py` | `quantlab/dataset/_support/masking.py` | 18 / 9 |
| 6 | `quantlab/dataset/session_calendar.py` | `quantlab/dataset/_support/session_calendar.py` | 6 / 4 |
| 7 | `quantlab/acquisition/sql_volume.py` | `quantlab/acquisition/_support/sql_volume.py` | 10 / 8 |
| 8 | `quantlab/acquisition/inspector.py` | `quantlab/acquisition/_support/inspector.py` | 27 / 7 |

Per-mover counts overlap on shared lines. The de-duplicated union, measured today:
**306 unique lines across 89 files.**

## B. The scripted rewrite — YOU MUST SCRIPT THIS, NOT HAND-EDIT

306 references is more than a context window survives hand-edited; the previous task nearly
ran out doing less. Write this to the scratchpad and run one row of the table per task.

```bash
# /private/tmp/claude-501/.../scratchpad/move_refs.sh   -- NOT in the repo
cd "$(git rev-parse --show-toplevel)"
ROOTS=(quantlab tests scripts example CLAUDE.md)

# Export these ONCE, before Task 1, and re-export them in every later shell. $SCRATCH is the
# session scratchpad the harness provides -- never a path inside the repo. $BASE pins the
# pre-task-1 commit so Task 4 can prove the historical .planning records were never touched,
# without assuming one commit per task.
export SCRATCH="${SCRATCH:?set this to the session scratchpad directory}"
export BASE="$(git rev-parse HEAD)"; echo "$BASE" > "$SCRATCH/base.sha"

move_refs() {                     # move_refs <old_dotted> <new_dotted> <old_path_suffix> <new_path>
  local OD="$1" ND="$2" OP="$3" NP="$4" files n
  files=$(grep -rlF -e "$OD" -e "$OP" "${ROOTS[@]}" 2>/dev/null | grep -v __pycache__ | sort -u)
  [ -n "$files" ] || { echo "REFUSE: no file matches $OD / $OP"; return 1; }
  n=$(printf '%s\n' "$files" | wc -l | tr -d ' ')
  OD="$OD" ND="$ND" OP="$OP" NP="$NP" perl -pi -e '
    s{quantlab/\Q$ENV{OP}\E}{$ENV{NP}}g;              # 1. fully-qualified path form FIRST
    s{(?<![A-Za-z0-9_/])\Q$ENV{OP}\E}{$ENV{NP}}g;     # 2. bare path form, never after a "/"
    s{\Q$ENV{OD}\E}{$ENV{ND}}g;                       # 3. dotted form
  ' $files
  echo "rewrote $OD -> $ND across $n files"
}

# `git` inside a pipeline has its status swallowed by the last stage, so a broken invocation
# would read as clean. Capture the status, then count.
hist() {                          # hist <path>   -- asserts git history survived the move
  local p="$1" out
  out=$(git log --follow --oneline -- "$p") || { echo "GIT FAILED for $p"; return 1; }
  local n; n=$(printf '%s\n' "$out" | grep -c .)
  [ "$n" -gt 1 ] || { echo "NO HISTORY (only $n commit) for $p"; return 1; }
  echo "history ok ($n commits): $p"
}
```

Three substitutions per mover, in this order and no other:

1. `quantlab/<old_path>` first, because it is a superstring of (2) — running (2) first would
   leave a `quantlab/quantlab/...` double prefix.
2. The bare path form with a `(?<![A-Za-z0-9_/])` lookbehind. The `_` half keeps
   `refresh_us_equity_universe.py`, `test_universe.py`, `universe_filter.py`,
   `test_source_inspector.py` and `test_cleaning.py` from matching. The `/` half is what the
   previous task had to add as a consistency fix mid-flight; it is here from the start.
3. The dotted form. Anchored on the full `quantlab.dataset.backend`, so
   `quantlab.base.backend` and `quantlab.ml_model.backend` are untouched.

The file LIST may be over-inclusive (fixed-string `grep -rlF`); the perl patterns are the
precise part. `move_refs` REFUSES on an empty file list, so a typo cannot pass as "done".

## C. The sweep gate — the single pattern that must reach 0

```bash
SWEEP='(quantlab\.(dataset\.(backend|cleaning|masking|session_calendar)|acquisition\.(registry|universe|sql_volume|inspector))\b)|((^|[^A-Za-z0-9_])(quantlab/)?(dataset/(backend|cleaning|masking|session_calendar)|acquisition/(registry|universe|sql_volume|inspector))\.py)'

sweep() { grep -rnE "$SWEEP" quantlab tests scripts example CLAUDE.md 2>/dev/null \
          | grep -v __pycache__ | sort -u; }
sweep | wc -l          # BEFORE: 306   AFTER (end of Task 4): 0
```

**This pattern is genuinely 0-able.** Verified against every post-move spelling:
`quantlab/dataset/_support/cleaning.py` does not contain the literal `dataset/cleaning.py`;
`quantlab.dataset._support.cleaning` does not contain `quantlab.dataset.cleaning`;
`quantlab/backend.py`, `quantlab/registry.py`, `quantlab/universe.py` match no alternative.
There is no `crsp_reference.py`-style structural collision this time — checked explicitly.

**Sweep roots deliberately exclude `.planning/`**, so this PLAN.md's own pre-move spellings
cannot pollute its own gate, and `.planning/phases/**` stays out of scope by construction.

**If the count does not reach 0, do NOT widen the pattern.** Report the residual lines
verbatim with the evidence for why they are correct post-move text, exactly as the previous
task did. A widened filter is how a real miss gets buried.

## D. The dangling-path gate — "AFTER identical to BEFORE", never "repaired"

```bash
dangling() {
  grep -rhoE '(^|[^A-Za-z0-9_])quantlab/[A-Za-z0-9_/]+\.py' quantlab tests scripts example CLAUDE.md 2>/dev/null \
  | grep -v __pycache__ | grep -oE 'quantlab/[A-Za-z0-9_/]+\.py' | sort -u \
  | while read -r p; do [ -e "$p" ] || echo "$p"; done
}
```

**BEFORE, measured today — 9 paths, and this exact list is the gate:**

```
quantlab/acquisition/progress.py
quantlab/base/factor_polars.py
quantlab/dataset/chunking.py
quantlab/dl_model/xxx.py
quantlab/factor/xxx.py
quantlab/ingest_alpaca.py
quantlab/label/spot.py
quantlab/tests/test_page_ledger.py
quantlab/vecbt/bt.py
```

Capture it to the scratchpad before Task 1 (`dangling > $SCRATCH/dangling.before`) and
`diff` after every task. **Do not repair any of the 9** — repairing them hides a regression
inside the diff, which is the entire reason this gate exists.

## E. The string-form references the sweep CANNOT see — hand-listed, because they fail silently

An import mistake is an immediate `ImportError`. These are not:

| file:line | form | failure mode if missed | sweep sees it? |
|---|---|---|---|
| `tests/test_wrds_vendor_seam.py:58` | `REGISTRY_SOURCE = REPO_ROOT / "quantlab" / "acquisition" / "registry.py"` | `FileNotFoundError` at collection | **NO — `/`-joined `Path`, not a slash string.** Hand-fix in Task 2. |
| `tests/test_source_inspector.py:365` | `"quantlab.acquisition.registry"` inside `_FORBIDDEN_INSPECTOR_MODULES` | **SILENT.** The frozenset stops matching; the test goes on passing while the inspector could freely import the registry. | yes (dotted) |
| `tests/test_volume_guard.py:834` | `quantlab/acquisition/universe.py` inside the assertion MESSAGE | silent staleness: a failure names a file that no longer exists | yes (slash) |
| `tests/test_universe.py` ×9, `tests/conftest.py:730`, `tests/test_constituent_panel.py:393` | `monkeypatch.setattr("quantlab.acquisition.universe.requests.get", ...)` | `AttributeError`/patch-no-op | yes (dotted) |

I re-derived the `Path`-join class exhaustively: `test_wrds_vendor_seam.py:58` is the **only**
sweep-invisible reference to any of the 8 movers. Every other `REPO_ROOT / "quantlab" / ...`
in `tests/` names `wrds/crsp.py`, `wrds/taq.py`, `utils/cli.py`, `dataset/stock.py`,
`base/backtest.py`, `base/factor.py`, `dataset/crsp/__init__.py` or `config/__init__.py` —
none of which move.

Structural arms that resolve their target through `inspect.getfile(module)` follow the move by
themselves and need no edit: `tests/test_volume_guard.py:830`,
`tests/test_source_inspector.py:303,582`.

## F. Bytecode caches — clear before EVERY verifying run

```bash
find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
```

This repo has been burned by stale bytecode (gap G-03.11-4). A file that moved while its
`.pyc` lingered is exactly the shape of failure it produces.

## G. Test command and baseline

```bash
uv run pytest -q \
  --ignore=tests/test_factor_hierarchy.py \
  --ignore=tests/test_crsp_rebuild_measurements.py \
  --ignore=tests/test_cross_sectional_zscore.py
```

`test_cross_sectional_zscore.py` is ignored on BOTH sides: KunQuant's
`~MultiThreadExecutor()` destructor deadlocks intermittently (`D-03.11-12-B`, whose scope note
was widened today — it is not confined to that one file). Same command both sides, so the
set-difference gate holds; absolute counts are not comparable to a run that includes it.

**Expected baseline: 55 failed, 1696 passed, 1 skipped.** The 55 are pre-existing
(`D-03.11-12-A`). **The gate is the failing node-ID set difference being empty in BOTH
directions — not green.** Capture it before Task 1:

```bash
uv run pytest -q ... 2>&1 | tee $SCRATCH/suite.before.txt
grep -E '^FAILED ' $SCRATCH/suite.before.txt | awk '{print $2}' | sort -u > $SCRATCH/failed.before
```

Task 1 and Task 4 each produce a `failed.after`; `comm -13` and `comm -23` must both be empty.

**Collected-count changes are EXPECTED and must be reported as such**, not as repairs:
Task 2 adds 2 parametrize cases to the import-order gate (3 -> 5 params = +2 passed), and
Tasks 1/3 add 2 new emptiness gates (+2 passed). Net expected: 1696 -> 1700 passed.
</shared_apparatus>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end move of ONE file — `universe.py` — through every mechanism this refactor uses</name>
  <files>
    quantlab/universe.py (from quantlab/acquisition/universe.py, git mv),
    quantlab/acquisition/registry.py,
    quantlab/dataset/constituent.py,
    quantlab/utils/cli.py,
    quantlab/enums/data.py,
    quantlab/base/acquisition.py,
    quantlab/base/coverage.py,
    tests/test_volume_guard.py,
    tests/test_universe.py,
    tests/conftest.py,
    tests/test_constituent_panel.py,
    tests/test_ingest_tiingo_universe_wiring.py,
    tests/test_sql_volume_guard.py,
    tests/test_ticker_pattern_reconciliation.py,
    tests/test_chunked_panel_estimate.py,
    tests/test_ingest_wrds_taq.py,
    tests/test_source_registry.py,
    scripts/ingest_us_equity.py,
    scripts/ingest_tiingo.py,
    scripts/ingest_alpaca.py,
    scripts/backfill_us_equity_full.py,
    scripts/refresh_us_equity_universe.py,
    example/universe.md,
    example/acquisition.md
  </files>
  <read_first>
    quantlab/acquisition/registry.py:720-758 (the invariant this task transfers),
    tests/test_volume_guard.py:729-845 (the structural arm)
  </read_first>
  <action>
This is the tracer: `universe.py` is the single file that carries every mechanism the other
seven moves need — a `git mv`, a scripted dotted+path rewrite, eleven `monkeypatch` string
targets, an assertion-message literal, a load-bearing prose invariant, and a permanent gate
that does not exist yet. Prove all six end-to-end on one file before touching the rest.

Set up the shared apparatus first (helper script in the scratchpad, `dangling.before`,
`failed.before` per section G). Then:

1. `git mv quantlab/acquisition/universe.py quantlab/universe.py`.

2. Run one row of the move table:
   `move_refs quantlab.acquisition.universe quantlab.universe acquisition/universe.py quantlab/universe.py`
   Report the file count it prints.

3. Carry the invariant's prose to where it now belongs. It currently lives at
   `quantlab/acquisition/registry.py:728-737` and argues that
   `quantlab/acquisition/__init__.py` must stay 0 bytes "because a non-empty package `__init__`
   runs on every `import quantlab.acquisition.<anything>` — including
   `quantlab.acquisition.universe`". That clause is now false: the universe module is no longer
   under that package. Do NOT delete the paragraph — the acquisition half is still true for
   `alpaca`/`tiingo`/`wrds`. Narrow it to the acquisition half, and add ONE sentence recording
   that the universe half transferred UP to `quantlab/__init__.py`, with the measurement from
   D-3 (two package `__init__`s on the import path became one; both must stay 0 bytes).

4. Give `quantlab/universe.py` a module-docstring paragraph stating its own guarantee, in the
   house style: this module must reach no acquisition client by any spelling, the enforcement
   is `tests/test_volume_guard.py`'s `ast` arm over this file's OWN source plus a
   `vars()` sweep, neither of which can see a transitive import dragged in by a package
   `__init__` — which is why `quantlab/__init__.py` stays 0 bytes, and why this module is a
   flat sibling rather than a package of its own (D-1). The module that HAS the guarantee is
   the right place for it; the reader who breaks it is reading this file.

5. In `tests/test_volume_guard.py`, inside
   `test_the_guard_constructs_no_acquisition_client_and_needs_no_credentials`, add a fourth arm
   to the structural check: assert `Path(inspect.getfile(universe_module)).parent /
   "__init__.py"` has `stat().st_size == 0`. Give it a failure message that names the
   guarantee, not the file — something an operator can act on, in the register the rest of that
   file uses. **This is the gate the invariant has never had.** Keep the arm AHEAD of the
   bound-clients scan for the same first-failing-assertion reason the file already documents at
   its existing arm ordering. Resolve the path from the module object, not from a literal, so
   the gate follows a future move the way the existing arm does.

6. Confirm the eleven `monkeypatch.setattr("quantlab.acquisition.universe.requests.get", ...)`
   targets and the assertion-message literal at what was line 834 were all rewritten by step 2
   (they are dotted/slash forms, so they should have been). Report the counts.

The rewrite is scripted. Budget your context for steps 3-5, which are prose and judgement.
  </action>
  <verify>
    <automated>
find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
test -f quantlab/universe.py && test ! -e quantlab/acquisition/universe.py
hist quantlab/universe.py
test "$(wc -c < quantlab/__init__.py)" -eq 0
grep -rnE 'quantlab\.acquisition\.universe|(^|[^A-Za-z0-9_])(quantlab/)?acquisition/universe\.py' quantlab tests scripts example CLAUDE.md | grep -v __pycache__ | wc -l    # must be 0
grep -rn 'monkeypatch.setattr("quantlab.universe.requests.get"' tests | wc -l    # must be 11
uv run python -c "import quantlab.universe as u; print(u.UniverseCatalog.__name__)"
uv run pytest -q tests/test_volume_guard.py 2>&1 | tail -5
uv run pytest -q --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py --ignore=tests/test_cross_sectional_zscore.py 2>&1 | tee "$SCRATCH/suite.t1.txt" | tail -3
grep -E '^FAILED ' "$SCRATCH/suite.t1.txt" | awk '{print $2}' | sort -u > "$SCRATCH/failed.t1"
comm -13 "$SCRATCH/failed.before" "$SCRATCH/failed.t1"    # must print nothing
comm -23 "$SCRATCH/failed.before" "$SCRATCH/failed.t1"    # must print nothing
diff <(dangling) "$SCRATCH/dangling.before"               # must print nothing
    </automated>
  </verify>
  <done>
`quantlab/universe.py` exists with history reachable through `git log --follow`; no file
remains at the pre-move path and no shim, alias or re-export stub was created anywhere. The
universe half of the sweep is 0. `quantlab/__init__.py` is 0 bytes AND a test now fails if it
stops being so. The registry comment says what is still true and nothing that is not. The
failing node-ID set is unchanged in both directions; the dangling list is byte-identical to
the 9-path baseline.
  </done>
</task>

<task type="auto">
  <name>Task 2: The other two layer movers — `registry.py` and `backend.py` — and the cycle they carry</name>
  <files>
    quantlab/registry.py (from quantlab/acquisition/registry.py, git mv),
    quantlab/backend.py (from quantlab/dataset/backend.py, git mv),
    quantlab/acquisition/alpaca.py,
    quantlab/acquisition/tiingo.py,
    quantlab/acquisition/wrds/__init__.py,
    quantlab/acquisition/inspector.py,
    quantlab/universe.py,
    quantlab/config/__init__.py,
    quantlab/factor/universe_filter.py,
    quantlab/base/factor.py,
    quantlab/base/model.py,
    quantlab/base/data.py,
    quantlab/base/backtest.py,
    quantlab/base/config.py,
    quantlab/base/rebuild.py,
    quantlab/base/acquisition.py,
    quantlab/dataset/crsp/rebuild.py,
    quantlab/utils/cli.py,
    quantlab/utils/symbol_axis.py,
    quantlab/ml_model/backend.py,
    tests/test_wrds_vendor_seam.py,
    tests/test_source_inspector.py,
    tests/test_source_registry.py,
    tests/conftest.py,
    tests/test_registry_convert.py,
    tests/test_ingest_shells.py,
    tests/test_backend_head.py,
    tests/test_backend_indexes.py,
    tests/test_backend_overwrite.py,
    tests/test_chunked_ingest.py,
    tests/test_symbol_axis_widening.py,
    tests/test_variable_axis_widening.py,
    tests/test_widen_crash_residue.py,
    tests/test_factor_update.py,
    tests/test_data_dir_cli.py,
    tests/test_crsp_identity.py,
    scripts/ingest_wrds_crsp.py,
    scripts/ingest_wrds_taq.py,
    scripts/ingest_us_equity.py,
    scripts/ingest_alpaca.py,
    scripts/ingest_tiingo.py,
    scripts/backfill_us_equity_full.py,
    example/registry.md,
    example/backend.md
  </files>
  <action>
`git mv` both, then run their two move-table rows:

```
move_refs quantlab.acquisition.registry quantlab.registry acquisition/registry.py quantlab/registry.py
<!-- planner-discipline-allow: quantlab.acquisition.registry -->
move_refs quantlab.dataset.backend      quantlab.backend  dataset/backend.py      quantlab/backend.py
```

Then four things the script cannot do:

1. **Hand-fix `tests/test_wrds_vendor_seam.py:58`.** `REGISTRY_SOURCE = REPO_ROOT /
   "quantlab" / "acquisition" / "registry.py"` is a `/`-joined `Path`; the sweep pattern
   physically cannot see it, and the file it names will not exist. Rewrite to
   `REPO_ROOT / "quantlab" / "registry.py"`. This is the single sweep-invisible reference in
   the whole task (section E) — if you fix nothing else by hand, fix this.

2. **Verify the `_FORBIDDEN_INSPECTOR_MODULES` entry was rewritten.** The frozenset at
   `tests/test_source_inspector.py:365` held the dotted name of the registry. Step 2's dotted
   substitution should have updated it. **Confirm it by reading the line**, because this is the
   one entry in the whole task whose failure mode is a green test over an empty match — the
   inspector could then import the registry freely and nothing would go red. The comment above
   that frozenset explains the registry is in the set because its bottom imports pull both
   vendor modules; that reasoning is unchanged by the move, so only the spelling changes.

3. **Rewrite the registry's bottom comment for its new address.** It now argues from
   `quantlab/registry.py`, and the cycle it describes became
   `quantlab.registry` -> `quantlab.acquisition.wrds` -> `quantlab.registry`: it crosses a
   package boundary where it used to stay inside one. State that the MODULE-OBJECT binding
   (`from quantlab.acquisition import wrds as _wrds`, never
   `from quantlab.acquisition.wrds import ...`) is what keeps it safe, and cite the five import
   orders the gate below now pins. Record which `__init__.py` files the argument depends on
   (`quantlab/` and `quantlab/acquisition/`, both 0 bytes).

4. **Make the import-order measurement a permanent gate, widened.**
   `tests/test_wrds_vendor_seam.py::test_enumeration_survives_any_wrds_import_order` already
   runs fresh interpreters through `_run_child` over 3 modules. Widen the parametrize to 5:
   `quantlab.acquisition.wrds.taq`, `quantlab.acquisition.wrds`, `quantlab.registry`,
   `quantlab.acquisition.alpaca`, `quantlab.universe`. Each must print
   `['alpaca', 'tiingo', 'wrds']` with no traceback. Rename the test so the name stops claiming
   the WRDS scope it has outgrown, and update its docstring to say the cycle now crosses a
   package boundary and that the `quantlab.universe`-first case additionally proves the guard
   module's import path stays clean of the registry's vendor pull. +2 collected tests — report
   as added, not as repaired.

Also update the two prose notes that now describe a cross-layer relationship:
`quantlab/ml_model/backend.py`'s docstring (it names `dataset/backend.py` as its symmetric
half — that symmetry is what D-2 resolves) and `quantlab/base/acquisition.py`'s references to
the registry's location.
  </action>
  <verify>
    <automated>
find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
test -f quantlab/registry.py && test -f quantlab/backend.py
test ! -e quantlab/acquisition/registry.py && test ! -e quantlab/dataset/backend.py
hist quantlab/registry.py
hist quantlab/backend.py
grep -n 'REGISTRY_SOURCE' tests/test_wrds_vendor_seam.py
grep -c 'quantlab.acquisition.registry' tests/test_source_inspector.py    # must be 0
grep -n 'quantlab.registry' tests/test_source_inspector.py               # must show the frozenset entry
grep -rnE 'quantlab\.(acquisition\.registry|dataset\.backend)\b|(^|[^A-Za-z0-9_])(quantlab/)?(acquisition/registry|dataset/backend)\.py' quantlab tests scripts example CLAUDE.md | grep -v __pycache__ | wc -l    # must be 0
for m in quantlab.registry quantlab.universe quantlab.acquisition.alpaca quantlab.acquisition.tiingo quantlab.acquisition.wrds quantlab.acquisition.wrds.taq quantlab.backend; do
  find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
  echo -n "$m -> "; uv run python -c "import $m; from quantlab.registry import DataSourceRegistry; print(sorted(d.vendor for d in DataSourceRegistry.all()))"
done    # every line must print ['alpaca', 'tiingo', 'wrds']
uv run pytest -q tests/test_wrds_vendor_seam.py tests/test_source_inspector.py tests/test_source_registry.py 2>&1 | tail -5
diff <(dangling) "$SCRATCH/dangling.before"
    </automated>
  </verify>
  <done>
Both files moved with `git log --follow` history intact and nothing left at either pre-move
path. The registry/backend halves of the sweep are 0. `REGISTRY_SOURCE` resolves to a file that
exists. `tests/test_source_inspector.py` names the registry by its NEW dotted path and by no
other. Seven cold-interpreter import orders each enumerate `['alpaca', 'tiingo', 'wrds']`, and
five of them are now a parametrized test rather than a one-off measurement.
  </done>
</task>

<task type="auto">
  <name>Task 3: The five `_support` movers, and the `_support/__init__.py` that must stay empty</name>
  <files>
    quantlab/dataset/_support/__init__.py (new, 0 bytes),
    quantlab/dataset/_support/cleaning.py,
    quantlab/dataset/_support/masking.py,
    quantlab/dataset/_support/session_calendar.py,
    quantlab/acquisition/_support/__init__.py (new, 0 bytes),
    quantlab/acquisition/_support/sql_volume.py,
    quantlab/acquisition/_support/inspector.py,
    quantlab/dataset/spot.py,
    quantlab/dataset/stock.py,
    quantlab/dataset/crsp/rebuild.py,
    quantlab/dataset/crsp/tickers.py,
    quantlab/dataset/nbbo/__init__.py,
    quantlab/dataset/nbbo/resample.py,
    quantlab/dataset/constituent.py,
    quantlab/base/constituent.py,
    quantlab/base/data.py,
    quantlab/base/chunking.py,
    quantlab/base/coverage.py,
    quantlab/factor/universe_filter.py,
    quantlab/utils/cli.py,
    tests/test_cleaning.py,
    tests/test_universe_mask.py,
    tests/test_session_calendar.py,
    tests/test_sql_volume_guard.py,
    tests/test_source_inspector.py,
    tests/test_acquisition_progress.py,
    tests/test_acquisition_batching.py,
    tests/test_nbbo_dataset.py,
    tests/test_nbbo_resampler.py,
    tests/test_constituent_panel.py,
    tests/test_crsp_rebuild.py,
    tests/test_wrds_crsp_acquisition.py,
    tests/test_ingest_wrds_crsp.py,
    scripts/ingest_wrds_crsp.py,
    scripts/ingest_wrds_taq.py,
    scripts/ingest_us_equity.py,
    example/constituent.md,
    example/dataset.md,
    example/chunking.md
  </files>
  <action>
Create the two subpackages, `git mv` the five files in, run their five move-table rows:

```
mkdir -p quantlab/dataset/_support quantlab/acquisition/_support
: > quantlab/dataset/_support/__init__.py
: > quantlab/acquisition/_support/__init__.py
git add quantlab/dataset/_support/__init__.py quantlab/acquisition/_support/__init__.py

git mv quantlab/dataset/cleaning.py          quantlab/dataset/_support/cleaning.py
git mv quantlab/dataset/masking.py           quantlab/dataset/_support/masking.py
git mv quantlab/dataset/session_calendar.py  quantlab/dataset/_support/session_calendar.py
git mv quantlab/acquisition/sql_volume.py    quantlab/acquisition/_support/sql_volume.py
git mv quantlab/acquisition/inspector.py     quantlab/acquisition/_support/inspector.py

move_refs quantlab.dataset.cleaning         quantlab.dataset._support.cleaning         dataset/cleaning.py         quantlab/dataset/_support/cleaning.py
move_refs quantlab.dataset.masking          quantlab.dataset._support.masking          dataset/masking.py          quantlab/dataset/_support/masking.py
move_refs quantlab.dataset.session_calendar quantlab.dataset._support.session_calendar dataset/session_calendar.py quantlab/dataset/_support/session_calendar.py
move_refs quantlab.acquisition.sql_volume   quantlab.acquisition._support.sql_volume   acquisition/sql_volume.py   quantlab/acquisition/_support/sql_volume.py
move_refs quantlab.acquisition.inspector    quantlab.acquisition._support.inspector    acquisition/inspector.py    quantlab/acquisition/_support/inspector.py
```

Then three things the script cannot do:

1. **Gate `quantlab/acquisition/_support/__init__.py` at 0 bytes, for a real reason.**
   `tests/test_source_inspector.py`'s structural arm is an `ast` scan of `inspector.py`'s OWN
   source. `inspector.py` now sits one package deeper, so `import
   quantlab.acquisition._support.inspector` runs TWO package `__init__`s that scan cannot see
   through. In the test that owns that arm, add an assertion that
   `quantlab/__init__.py`, `quantlab/acquisition/__init__.py` and
   `quantlab/acquisition/_support/__init__.py` are each 0 bytes, with a failure message that
   names the guarantee (no acquisition client is reachable from the read surface, whatever the
   call order). Follow the file's own documented convention — the copy at its
   `_INSPECTOR_RESOLVER_TOKEN` comment explains that a copy is preferred to a cross-test import
   because `tests/` is not a package; a second, independently-worded copy of this emptiness
   check is therefore correct house style, not duplication to be factored out.

2. **Record the `pkgutil` behaviour change.**
   `tests/test_acquisition_batching.py:_concrete_acquisition_subclasses` walks
   `pkgutil.iter_modules(acquisition.__path__, ...)` and imports every module it finds. Before
   this task that walk imported `registry`, `inspector` and `sql_volume` as a side effect; now
   it yields `alpaca`, `tiingo`, `wrds` and the `_support` package (which it does not recurse
   into). The concrete-subclass discovery is unaffected — the three vendors are still walked
   directly — but the helper's docstring claims the walk covers "every module under the
   `acquisition` package", and that claim now means something narrower and BETTER: the walk
   sees exactly the acquisitions. Update the docstring to say so. Confirm the test still passes
   and report its result explicitly; do not assume.

3. **`quantlab/factor/universe_filter.py` and `quantlab/dataset/crsp/tickers.py`** carry prose
   contrasting themselves with `UniverseMask`/`masking.py`. Those cross-references are the kind
   a reader follows; make sure the rewritten paths still land on a file that exists.

Watch for the shadowing trap the previous task hit three times: a local variable named
`cleaning`, `masking`, `inspector` or `sql_volume` in a file that now imports the module under
that name. Check before you import; if one exists, rename the LOCAL and say so, keeping the
plan's module path.
  </action>
  <verify>
    <automated>
find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
ls quantlab/dataset/    # exactly: __init__.py _support constituent.py crsp nbbo spot.py stock.py
ls quantlab/acquisition/    # exactly: __init__.py _support alpaca.py tiingo.py wrds
for f in quantlab/__init__.py quantlab/dataset/__init__.py quantlab/acquisition/__init__.py quantlab/dataset/_support/__init__.py quantlab/acquisition/_support/__init__.py; do
  test "$(wc -c < $f)" -eq 0 || echo "NOT EMPTY: $f"
done
for p in quantlab/dataset/_support/cleaning.py quantlab/dataset/_support/masking.py quantlab/dataset/_support/session_calendar.py quantlab/acquisition/_support/sql_volume.py quantlab/acquisition/_support/inspector.py; do
  hist "$p"
done
test ! -e quantlab/dataset/cleaning.py && test ! -e quantlab/dataset/masking.py && test ! -e quantlab/dataset/session_calendar.py
test ! -e quantlab/acquisition/sql_volume.py && test ! -e quantlab/acquisition/inspector.py
uv run python -c "from quantlab.dataset._support.cleaning import validate_schema; from quantlab.dataset._support.masking import UniverseMask; from quantlab.dataset._support.session_calendar import XnysSessionCalendar; from quantlab.acquisition._support.sql_volume import SqlVolumeGuard; from quantlab.acquisition._support.inspector import SourceInspector; print('ok')"
uv run pytest -q tests/test_source_inspector.py tests/test_acquisition_batching.py tests/test_cleaning.py tests/test_universe_mask.py tests/test_session_calendar.py tests/test_sql_volume_guard.py 2>&1 | tail -5
diff <(dangling) "$SCRATCH/dangling.before"
    </automated>
  </verify>
  <done>
`ls quantlab/dataset/` shows five datasets and `_support/`; `ls quantlab/acquisition/` shows
three acquisitions and `_support/`. All five moved files have `git log --follow` history and
nothing remains at any pre-move path. All five `__init__.py` files in the chain are 0 bytes,
and two independent tests now fail if any of the three load-bearing ones stops being so.
  </done>
</task>

<task type="auto">
  <name>Task 4: Sweep to zero, resolve the convention in CLAUDE.md, re-point live docs, re-verify</name>
  <files>
    CLAUDE.md,
    example/README.md,
    example/acquisition.md,
    example/backend.md,
    example/chunking.md,
    example/constituent.md,
    example/dataset.md,
    example/factor.md,
    example/registry.md,
    example/universe.md,
    example/wrds_crsp.md,
    example/wrds_taq.md,
    .planning/STATE.md,
    .planning/WINDOWS.md,
    .planning/todos/pending/2026-09-07-normalize-ticker-delimiter-between-membership-panel-and-pric.md,
    .planning/todos/pending/2026-09-07-no-ticker-rename-mapping-between-membership-history-and-pric.md,
    .planning/todos/pending/2026-09-08-an-empty-zarr-store-records-symbol-as-float64.md,
    .planning/todos/pending/2026-09-21-cross-vendor-permno-ticker-merge-is-silently-all-nan.md,
    .planning/todos/pending/2026-09-07-re-point-stale-former-root-paths-in-python-docstrings.md
  </files>
  <action>
1. **Run the full sweep (section C) and drive it to 0.** Report the count. Whatever remains,
   fix it in the file, not in the pattern. If you conclude a residual line is correct post-move
   text, say so loudly with the evidence — the file it names must exist on disk — and
   cross-check with a second, independently-constructed pattern, exactly as the previous task
   did. Do not widen `SWEEP`.

2. **Rewrite the CLAUDE.md `__init__.py` bullet (currently line 176).** It is the previous
   task's bullet and this task invalidates part of it: its closing clause says
   `quantlab/acquisition/registry.py:728-737` explains why `quantlab/acquisition/__init__.py`
   must stay 0 bytes "because a non-empty package `__init__` would run on every `import
   quantlab.acquisition.universe`". That file and that module both moved. The rewritten bullet
   must state:
   - the three-way distinction now in force: LAYER packages with empty `__init__.py` and full
     dotted imports; the three packages whose `__init__.py` IS the entry module (unchanged);
     and the new `_support/` subpackages, which are neither — they are private, have empty
     `__init__.py` files, and exist so the layer directory above them reads as a menu;
   - that **five** `__init__.py` files are 0 bytes and which guarantee each one carries
     (D-3's table), with the measurement that `import quantlab.universe` went from two package
     `__init__`s to one;
   - that the invariant is now enforced by tests in `tests/test_volume_guard.py` and
     `tests/test_source_inspector.py`, not by prose alone — and where the prose that explains
     WHY now lives.

3. **Add the new layout rule to CLAUDE.md, in its own bullet.** "Every top-level entry of
   `quantlab/dataset/` is a dataset; every top-level entry of `quantlab/acquisition/` is an
   acquisition. Support code lives in `_support/` or, if it is really its own layer, as a
   `quantlab/` sibling." Name the three siblings this task created and one sentence on why each
   is a layer rather than support.

4. **Resolve the pairing convention — D-2, in CLAUDE.md, as prose.** Write the rule from D-2
   verbatim in intent: the ABC lives in `base/`, the concrete implementation lives with its
   CONSUMERS, the mirrored filename was never the rule. Name all three cases
   (`ml_model/backend.py`, `quantlab/backend.py`, `dataset/constituent.py`) and give the
   seven-importers-across-five-layers measurement for `backend`. Update the Component
   Responsibilities rows (currently lines 92-93), the Storage Backend layer Location line
   (122) and the Key Abstractions examples (148) to the new paths while you are there. **Do not
   leave two contradictory conventions undocumented** — that is this step's whole point.

5. **`example/*.md`** — paths only, in place, Chinese prose preserved. Do not rename the doc
   files (the previous task set that policy and it holds).

6. **Live `.planning` pointers only.** Re-point exactly these, which point at code that moved:
   - `.planning/STATE.md:176, 197, 249`
   - `.planning/todos/pending/` — 8 lines across 5 files (measured; enumerate before editing)
   - `.planning/WINDOWS.md` — the two OPEN rows (id 4 `dataset/masking.py`, id 17
     `quantlab/acquisition/inspector.py`) and their JSON mirrors at lines 90 and 246
   **`.planning/phases/**`, `.planning/todos/completed/**`, `.planning/ROADMAP.md` and
   `.planning/research/` stay byte-for-byte untouched.** A SUMMARY that says "modified
   `quantlab/dataset/backend.py`" was TRUE when written; rewriting it falsifies the audit
   trail. Their absence from every sweep is by design.
   **Use precise replacements with an assert, never a `sed` range delete** — this project has
   been burned by a range delete running to EOF. Record each edited file's line count before
   and after and confirm it is unchanged.

7. **Final full-suite run and the set-difference gate**, caches cleared first. Report the
   before/after counts, the node-ID set difference in both directions, and account for every
   collected-count delta as an ADDED test (expected: +4 passed — 2 emptiness gates, 2
   import-order params).
  </action>
  <verify>
    <automated>
find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
grep -rnE '(quantlab\.(dataset\.(backend|cleaning|masking|session_calendar)|acquisition\.(registry|universe|sql_volume|inspector))\b)|((^|[^A-Za-z0-9_])(quantlab/)?(dataset/(backend|cleaning|masking|session_calendar)|acquisition/(registry|universe|sql_volume|inspector))\.py)' quantlab tests scripts example CLAUDE.md 2>/dev/null | grep -v __pycache__ | sort -u | wc -l    # must be 0
git diff --stat "$(cat "$SCRATCH/base.sha")" -- .planning/phases .planning/ROADMAP.md .planning/research .planning/todos/completed    # must be empty
grep -c '_support' CLAUDE.md    # must be > 0
uv run pytest -q --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py --ignore=tests/test_cross_sectional_zscore.py 2>&1 | tee "$SCRATCH/suite.after.txt" | tail -3
grep -E '^FAILED ' "$SCRATCH/suite.after.txt" | awk '{print $2}' | sort -u > "$SCRATCH/failed.after"
comm -13 "$SCRATCH/failed.before" "$SCRATCH/failed.after"    # must print nothing
comm -23 "$SCRATCH/failed.before" "$SCRATCH/failed.after"    # must print nothing
diff <(dangling) "$SCRATCH/dangling.before"                  # must print nothing
git grep -c 'quantlab/dataset/backend.py' -- . ':!.planning'; test $? -eq 1    # exit 1 == no match anywhere
    </automated>
    <human-check>
`ls quantlab/dataset/` and `ls quantlab/acquisition/` each read as a menu of complete, usable
things — the operator's stated goal, which no automated check can confirm.
    </human-check>
  </verify>
  <done>
The sweep is 0 with no pattern widening. CLAUDE.md states the new layout rule, the five-file
`__init__.py` invariant with its enforcement, and the resolved `base/X.py` pairing rule, with
no contradictory convention left undocumented. Every live `.planning` pointer resolves to a
file that exists and every historical record is byte-identical. The failing node-ID set is
unchanged in both directions and every collected-count delta is accounted for as an added test.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| none crossed | This task moves files and rewrites references. It opens no network endpoint, adds no auth path, changes no schema and installs no package. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-lu2-01 | Tampering | `tests/test_source_inspector.py:_FORBIDDEN_INSPECTOR_MODULES` | high | mitigate | A stale dotted literal silently disables the guard that keeps credential-demanding vendor clients out of the read surface. Task 2 step 2 requires the line be READ, not assumed rewritten; the AFTER state is asserted by `grep -c 'quantlab.acquisition.registry' tests/test_source_inspector.py == 0` plus a positive grep for the new name. |
| T-lu2-02 | Elevation of Privilege | `quantlab/acquisition/_support/__init__.py` | medium | mitigate | A non-empty `_support/__init__.py` would import a vendor client where the inspector's `ast` arm cannot see it, defeating the no-client-constructible guarantee. Task 3 step 1 adds a 0-byte assertion over all three acquisition-chain `__init__.py` files. |
| T-lu2-03 | Information Disclosure | `quantlab/__init__.py` | medium | mitigate | Same shape, one level up, for the volume guard's refuse-before-any-client-exists property. Task 1 step 5 adds the 0-byte assertion inside the test that owns that guarantee. |
| T-lu2-04 | Tampering | package installs | n/a | accept | No package-manager install task exists in this plan, so the package-legitimacy gate does not apply. |
</threat_model>

<verification>
- 8 files at their new paths, 0 at their pre-move paths, `git log --follow` reaching pre-move
  commits on all 8, no shim / alias / re-export stub anywhere.
- Sweep: 306 -> 0, with no widening of the pattern.
- Dangling `quantlab/**.py` paths: AFTER list byte-identical to the 9-path BEFORE list.
- Five `__init__.py` files 0 bytes, three of them now gated by tests.
- Seven cold-interpreter import orders enumerate `['alpaca', 'tiingo', 'wrds']`; five are a
  parametrized permanent test.
- Suite: failing node-ID set difference empty in BOTH directions; every count delta accounted
  for as an added test.
</verification>

<success_criteria>
Browsing `quantlab/dataset/` shows five datasets and one `_support/`; browsing
`quantlab/acquisition/` shows three acquisitions and one `_support/`. Every top-level entry is
one complete, usable thing. Nothing that was true before this task is false after it, and the
two invariants that were held by prose are now held by tests.
</success_criteria>

<output>
Create `.planning/quick/260922-lu2-make-every-dataset-and-acquisition-entry-a-dataset/SUMMARY.md`
when done. It must report, as numbers: the sweep before/after, the dangling before/after diff,
the suite before/after with the node-ID set difference in both directions, the cold-import
matrix, and every deviation from this plan with the evidence that the deviation was right.
</output>
