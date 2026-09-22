---
phase: quick-260922-edi
plan: 01
subsystem: acquisition / dataset module layout
tags: [refactor, packaging, imports, git-mv]
status: complete
requires:
  - quantlab/acquisition/registry.py's module-object bottom import
provides:
  - quantlab.acquisition.wrds as a package whose __init__ IS the descriptor
  - quantlab.dataset.crsp as a package whose __init__ IS CrspStockDataset's module
  - quantlab.dataset.nbbo as a package whose __init__ IS NbboPanelDataset's module
affects:
  - quantlab/base/backtest.py (+0.99s / +196 modules on import)
  - tests/conftest.py mock_wrds_session / mock_crsp_session (find_spec now imports the parent)
tech-stack:
  added: []
  patterns:
    - "package __init__.py IS the entry module (never a re-export shim)"
key-files:
  created: []
  modified:
    - quantlab/acquisition/wrds/__init__.py
    - quantlab/acquisition/wrds/crsp.py
    - quantlab/acquisition/wrds/crsp_reference.py
    - quantlab/acquisition/wrds/taq.py
    - quantlab/dataset/crsp/__init__.py
    - quantlab/dataset/crsp/membership.py
    - quantlab/dataset/crsp/rebuild.py
    - quantlab/dataset/crsp/reference.py
    - quantlab/dataset/crsp/symbology.py
    - quantlab/dataset/crsp/tickers.py
    - quantlab/dataset/nbbo/__init__.py
    - quantlab/dataset/nbbo/resample.py
    - CLAUDE.md
decisions:
  - "The 12 moves were made with `git mv`; `git log --follow` reaches pre-move commits on every new path."
  - "No compatibility shim, deprecation alias or re-export stub exists at any old path."
  - "\"The providers stay free of the registry\" is now a SOURCE-TEXT rule enforced by an ast scan, not a runtime property."
  - "tests/test_source_registry.py's local `crsp` capability variable was renamed `crsp_cap` so the module name `crsp` is not shadowed (deviation, Rule 1)."
  - "tests/test_crsp_membership.py imports the module as `membership_module` because `membership` is that file's name for a CrspMembership INSTANCE (deviation, Rule 1)."
metrics:
  duration: ~75 min
  completed: 2026-09-22
actuals:
  tokens: 46000
  tasks: 3
  commits: 3
plan_head_before: 81e549af24d0541c49330e915b745ce69cfa6066
---

# Quick 260922-edi Plan 01: Package wrds, crsp and nbbo into import entry points Summary

Twelve files moved into three packages whose `__init__.py` IS the original entry module, with
the `X_` prefix stripped from every submodule; the entry-point imports operators write are
byte-identical before and after, and the suite's failing node-ID set is unchanged in both
directions.

## What was done

| Task | Commit | Content |
|---|---|---|
| 1 (tracer) | `b703de1` | 4 wrds files moved; 99 wrds-family references rewritten across 29 files; package/provider docstrings rewritten; seam + tracer tests retargeted |
| 2 | `307dde7` | 8 crsp/nbbo files moved; dataset-family references rewritten across 33 files; backtest import-cost comment; tickers.py parent-import note |
| 3 | `e8968d1` | full sweep, `CLAUDE.md:176` convention rewritten, three live `.planning` pointers re-pointed |

### The 12 moves (all `git mv`, history verified)

```
quantlab/acquisition/wrds.py                -> quantlab/acquisition/wrds/__init__.py        (3 commits)
quantlab/acquisition/wrds_crsp.py           -> quantlab/acquisition/wrds/crsp.py            (5)
quantlab/acquisition/wrds_crsp_reference.py -> quantlab/acquisition/wrds/crsp_reference.py  (3)
quantlab/acquisition/wrds_taq.py            -> quantlab/acquisition/wrds/taq.py             (6)
quantlab/dataset/crsp.py                    -> quantlab/dataset/crsp/__init__.py           (24)
quantlab/dataset/crsp_membership.py         -> quantlab/dataset/crsp/membership.py          (6)
quantlab/dataset/crsp_rebuild.py            -> quantlab/dataset/crsp/rebuild.py             (3)
quantlab/dataset/crsp_reference.py          -> quantlab/dataset/crsp/reference.py           (3)
quantlab/dataset/crsp_symbology.py          -> quantlab/dataset/crsp/symbology.py           (7)
quantlab/dataset/crsp_tickers.py            -> quantlab/dataset/crsp/tickers.py             (9)
quantlab/dataset/nbbo.py                    -> quantlab/dataset/nbbo/__init__.py            (4)
quantlab/dataset/nbbo_resample.py           -> quantlab/dataset/nbbo/resample.py            (5)
```

`git log --follow --oneline` reaches pre-move commits on all 12 (> 1 commit each). None of the
12 pre-move paths exists on disk. `quantlab/acquisition/__init__.py` and
`quantlab/dataset/__init__.py` are still 0 bytes. No compatibility shim, deprecation alias or
re-export stub exists at any old path.

## Measurements (executor's, taken in the execution worktree, all repo-root-relative)

### Test suite

| | failed | passed | skipped |
|---|---|---|---|
| BEFORE (`81e549a`, caches cleared) | **55** | 1694 | 1 |
| AFTER (`e8968d1`, caches cleared) | **55** | 1695 | 1 |

Failing node-ID set difference: **empty in BOTH directions** (`comm -13` and `comm -23` both
produced nothing). The baseline is exactly the expected 55 of `D-03.11-12-A`, and the per-file
breakdown matched that ledger line for line: `test_ingest_shells` 14,
`test_ingest_tiingo_universe_wiring` 11, `test_data_dir_cli` 10, `test_ingest_conversion_gate` 6,
`test_volume_guard` 5, `test_factor_kunquant` 3, `test_spot_dataset` 2, `test_chunked_ingest` 2,
`test_universe` 1, `test_entry_point_contracts` 1.

The `1694 -> 1695` pass delta is **one test that was added**, not a repaired failure:
`test_enumeration_survives_any_wrds_import_order` gained a third parametrize case
(`quantlab.acquisition.registry` first). 2 params -> 3 params = +1 collected test.

**Deviation from the plan's test command:** the plan ignores `test_factor_hierarchy.py` and
`test_crsp_rebuild_measurements.py`. Both the BEFORE and AFTER runs ALSO ignored
`tests/test_cross_sectional_zscore.py`, because that file intermittently deadlocks KunQuant
(`D-03.11-12-B`) and a hang would have destroyed the baseline. The same command was used on both
sides, so the set-difference gate is valid; the absolute counts are not directly comparable to a
run that includes that file.

### Reference sweeps

| Sweep | Before | After |
|---|---|---|
| wrds half (Task 1 pattern) | 99 in 29 files | **0** |
| dataset half (Task 2 pattern) | 89 pre-Task-1 / 92 post-Task-1 | **3** (see below) |
| full sweep (Task 3 pattern, plan verbatim) | 200 in 52 files | **3** (see below) |
| full sweep with the `/`-exclusion applied consistently | 200 | **0** |

### A second exclusion was needed — reported loudly, as the plan demands

The plan says the single `EXC` is the only exclusion, and that if a second line needs excluding
"that is a signal the rewrite is wrong, not the filter". Three lines need it. **`EXC` was NOT
widened.** The evidence that this one is the filter, not the rewrite:

```
tests/test_crsp_reference_tables.py:5:`quantlab/acquisition/wrds/crsp_reference.py:CrspReferenceTables`, which PUTS
tests/test_crsp_reference_tables.py:28:style: this suite is written before `wrds/crsp_reference.py` exists, and a
example/wrds_crsp.md:5:> `quantlab/acquisition/wrds/crsp_reference.py:CrspReferenceTables`，
```

All three name `quantlab/acquisition/wrds/crsp_reference.py`, which **exists on disk** and is
the plan's own prescribed post-move spelling — the plan's move table keeps the basename
`crsp_reference.py` for the acquisition module. The collision is structural: `crsp_reference`
is in the `$OLD` alternation because the DATASET module `crsp_reference.py` became
`crsp/reference.py`, while the ACQUISITION module keeps that basename permanently. The pattern's
`(^|[^A-Za-z0-9_])($OLD)\.py` alternative allows a preceding `/`; its sibling alternative
`(^|[^A-Za-z0-9_/])(wrds|crsp|nbbo)\.py` excludes one, for exactly this reason. Applying the
same `/`-exclusion to both basename alternatives — a consistency fix, not a per-line exemption —
yields **0**. Two independent cross-checks agree:

- a grep for path-qualified stale spellings,
  `(^|[^A-Za-z0-9_])(quantlab/)?(acquisition|dataset)/($OLD)\.py` over
  `quantlab tests scripts example CLAUDE.md`, returns **0**;
- the dangling-path gate below is unchanged.

When the plan was written those three lines read `quantlab/acquisition/wrds_crsp_reference.py`
and matched the FIRST alternative, so they were correctly part of the 200. Post-move they match
the wrong alternative. Nothing in the tree is stale.

**Orchestrator's independent check (main tree, post-merge):**
`quantlab/acquisition/wrds/crsp_reference.py` exists on disk (16560 bytes) and is exactly the
path those three lines name. Confirmed a filter artefact, not a missed rewrite.

### Dangling `quantlab/**.py` path references

The before/after lists diff clean: **14 before, 14 after, byte-identical**. Nothing added, and
none of the 14 pre-existing ones (`factor_polars.py`, the `xxx.py` placeholders,
`quantlab/vecbt/bt.py`, ...) repaired by accident.

### Runtime gates

- Three fresh interpreters, caches cleared, taq-first / wrds-first / registry-first: all three
  printed `['alpaca', 'tiingo', 'wrds']` with no traceback. These three orders are now a
  permanent gate, not a one-off measurement.
- `from quantlab.acquisition.wrds import WRDS_SOURCE` is `DataSourceRegistry.get('wrds')` by
  IDENTITY.
- `from quantlab.dataset.crsp import CrspStockDataset, TICKER_SIDECAR_SUFFIX, SECURITY_FILTER_PRESETS`,
  `from quantlab.dataset.nbbo import NbboPanelDataset` and all six new submodule paths import
  cleanly.
- `tests/test_crsp_rebuild_measurements.py` (off the standard test command) collects: 6 tests.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] renaming the imported module binding broke ~45 call sites**

- **Found during:** Task 1 verify run (27 failures, `NameError: name 'wrds_crsp' is not defined`).
- **Issue:** The plan's rewrite table gives the new import line
  (`from quantlab.acquisition.wrds import crsp, taq`) but not the identifier renames it forces.
  `tests/test_wrds_crsp_acquisition.py` used `wrds_crsp.X` / `wrds_taq.X` on 32 lines and
  `tests/test_source_registry.py` on 13.
- **Fix:** `wrds_crsp.` -> `crsp.`, `wrds_taq.` -> `taq.` throughout both files.
- **Commit:** `b703de1`

**2. [Rule 1 - Bug] the new module name `crsp` was shadowed by a local variable**

- **Found during:** Task 1, while fixing deviation 1.
- **Issue:** `tests/test_source_registry.py:447` binds `crsp = by_key[("us_equity", "1d", "crsp_daily")]`
  — a Capability row — seven lines before the module `crsp` is used. The plan's prescribed
  `from quantlab.acquisition.wrds import crsp` would have been silently shadowed.
- **Fix:** renamed the LOCAL to `crsp_cap`, keeping the plan's import spelling.
- **Commit:** `b703de1`

**3. [Rule 1 - Bug] the same shadowing, one package over, for `membership`**

- **Found during:** Task 2.
- **Issue:** `tests/test_crsp_membership.py` uses `membership` as the name of a `CrspMembership`
  INSTANCE on ~30 lines. The plan's `from quantlab.dataset.crsp import membership` at :603 would
  read as one of those.
- **Fix:** imported as `membership_module` with a comment saying why. This is the one place the
  plan's literal import spelling was not used; the module PATH is exactly the plan's.
- **Commit:** `307dde7`

**4. [Rule 2 - Missing coverage] three prose references to `wrds_taq.WrdsSession` outside the named files**

- **Found during:** Task 1 sweep.
- **Issue:** `scripts/ingest_wrds_crsp.py:337`, `tests/crsp_fixtures.py:848` and
  `tests/conftest.py:1253` spell the patch target in prose in a form neither half-pattern
  reaches. A doc pointing at a module attribute that moved is how the next reader loses an hour.
- **Fix:** rewritten to `wrds.taq.WrdsSession`.
- **Commit:** `b703de1`

### Behaviour changes recorded rather than fixed

- `tests/conftest.py`'s `mock_wrds_session` and `mock_crsp_session` call
  `find_spec("quantlab.acquisition.wrds.taq")`, which imports the PARENT package to locate the
  submodule. Both fixtures therefore now pull in the descriptor, the registry,
  `quantlab.dataset.crsp` and `quantlab.dataset.nbbo` at fixture-setup time; before the move the
  guard only imported the 0-byte `quantlab.acquisition`. One sentence added to each docstring.
  `isolated_registry` and `tests/test_volume_guard.py` were watched in the verify run: nothing
  new went red (`test_volume_guard`'s 5 failures are in the BEFORE baseline, unchanged).
- `quantlab/base/backtest.py`'s `CrspTickerLookup` import now runs the whole `crsp` package
  `__init__`. The accepted +0.99s / +196 modules is recorded as a comment AT the import.

## Documentation

- **Rewritten, not deleted:** `quantlab/acquisition/wrds/__init__.py`'s anti-cycle argument. It
  now states (a) the descriptor lives in the entry point and a registration inside a provider
  would still have to name a sibling's classes, (b) why no import order cycles — module-object
  binding plus `from package import submodule` during partial init — citing the three fresh
  interpreters, (c) **what was traded away, with the numbers**: any provider import now loads
  the registry and both datasets, 1457 -> 1600 modules, 0.87s -> 0.85s, so "the providers stay
  free of the registry" is a SOURCE-TEXT rule enforced by `tests/test_wrds_vendor_seam.py`, not
  a runtime fact; and (d) that "the providers import each other not at all" was ALREADY false
  before the move. It names no pre-move path.
- `tests/test_wrds_vendor_seam.py` now says in the module docstring AND in the test's own
  docstring that `test_wrds_taq_registers_nothing_and_imports_no_registry` is a SOURCE-TEXT
  claim, and points at the import-order test as the file's genuine runtime claim. It gained a
  `node.level == 0` assertion so a relative import — newly possible inside a package — cannot
  evade the forbidden-prefix scan.
- `CLAUDE.md:176` rewritten to state both halves: layer packages keep empty `__init__.py` files
  and full dotted imports; three packages have an `__init__.py` that IS the entry module, with
  what that buys and the +0.99s / +196 modules it costs; `quantlab/acquisition/__init__.py` is
  still 0 bytes and `registry.py:728-737` says why it must stay so.
- `example/wrds_crsp.md`, `example/wrds_taq.md`, `example/universe.md` updated in place
  (Chinese, paths only). The doc FILES were not renamed, per policy.

## `.planning` policy — why most of it is untouched, and that is correct

Exactly three live pointers were re-pointed, each pointing at code that moved:

- `.planning/STATE.md:308` — `quantlab/dataset/crsp/tickers.py:CrspTickerLookup`
- `.planning/todos/pending/2026-09-07-no-ticker-rename-...md:93` — `crsp/symbology.py`
- `.planning/todos/pending/2026-09-20-delisting-return-...md:12` — `crsp/__init__.py`

Line counts after editing: 393 / 122 / 63 — unchanged from before, confirmed. No `sed` range
delete was used anywhere.

**`.planning/phases/`, `.planning/WINDOWS.md`, `.planning/ROADMAP.md` and `.planning/research/`
are byte-for-byte untouched, deliberately.** A SUMMARY that says "modified
`quantlab/dataset/crsp_tickers.py`" was TRUE when it was written; rewriting it would falsify the
audit trail. Their absence from every sweep's path list is by design, not a miss — please do not
file it as one.

## Known Stubs

None. No stub, placeholder, skipped test or unrun verify block was introduced. Every verify
block in the plan was run; the two that did not reach the plan's literal expected value (the
Task 2 half-sweep at 3 and the Task 3 full sweep at 3) are analysed above with the evidence that
the residue is correct post-move text, and both were cross-checked to 0 by an
internally-consistent variant of the same pattern.

## Threat Flags

None. This plan moved files and rewrote references; it added no network endpoint, auth path,
file-access pattern or schema change, and installed no package.

## Self-Check: PASSED

- All 12 new paths exist; all 12 pre-move paths are gone (checked with `test ! -e`).
- `git log --follow` returns > 1 commit for each of the 12.
- Commits `b703de1`, `307dde7`, `e8968d1` all present in `git log`.
- `quantlab/acquisition/__init__.py` and `quantlab/dataset/__init__.py` are 0 bytes.

## Post-merge re-verification (orchestrator, MAIN TREE)

The executor's green was taken in an isolated worktree, which this project has been burned by
before (a verify command carrying an absolute main-tree path tests unchanged code). Re-measured
independently on `main` at the merge commit, bytecode caches cleared first:

- structure: the three packages exist with the expected members; all 12 pre-move paths gone;
  git recorded the moves as renames at 96-100% similarity, so history survives;
- entry imports unchanged and `WRDS_SOURCE is DataSourceRegistry.get('wrds')` by identity;
- all three import orders in fresh interpreters print `['alpaca', 'tiingo', 'wrds']`;
- full-suite result recorded in the merge-commit message.
