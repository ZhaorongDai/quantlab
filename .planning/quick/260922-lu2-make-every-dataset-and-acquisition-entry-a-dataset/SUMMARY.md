---
phase: quick-260922-lu2
plan: 01
subsystem: repo-layout
tags: [refactor, packaging, import-graph, structural-guards]
status: complete
requires: []
provides:
  - "quantlab/backend.py, quantlab/registry.py, quantlab/universe.py as top-level layer siblings"
  - "quantlab/dataset/_support/ and quantlab/acquisition/_support/ as private subpackages"
  - "test-enforced 0-byte invariant on three load-bearing package __init__.py files"
affects:
  - "every module importing a backend, the registry, the universe catalog, cleaning, masking, session_calendar, sql_volume or the inspector"
tech-stack:
  added: []
  patterns:
    - "_support/ subpackage: private, empty __init__.py, exists so the layer directory above reads as a menu"
    - "the ABC lives in base/, the concrete implementation lives with its CONSUMERS (D-2)"
key-files:
  created:
    - quantlab/dataset/_support/__init__.py
    - quantlab/acquisition/_support/__init__.py
  modified:
    - CLAUDE.md
    - quantlab/registry.py
    - quantlab/universe.py
    - tests/test_volume_guard.py
    - tests/test_source_inspector.py
    - tests/test_wrds_vendor_seam.py
    - tests/test_acquisition_batching.py
decisions:
  - "D-1 upheld: quantlab/universe.py is a FLAT module, not a universe/ package -- a package would put a second __init__ back on the import path and re-create the exact hazard the volume guard's structural arm cannot see through"
  - "D-2 written into CLAUDE.md: the mirrored base/X.py <-> <layer>/X.py filename was never the rule; the concrete implementation lives with its consumers"
  - "D-3 discharged: the 0-byte __init__ invariant is now held by TESTS, not prose, on all three load-bearing files"
  - "Dead-path literals are not exempted from the sweep even when they appear in correct history-prose -- the meaning is kept and the literal dropped, so the gate stays meaningful"
  - "Pasted loguru transcripts in example/*.md are NOT rewritten: they are records of what a run printed, and they were already abbreviated before this task"
metrics:
  duration: ~4h
  completed: 2026-09-22
actuals:
  tokens: 96000
  tasks: 4
  commits: 4
plan_head_before: 23e51de4b71dbaa8801e8955e3023f7dcc876f6e
---

# Quick 260922-lu2: Make Every dataset/ and acquisition/ Entry a Complete Thing — Summary

Eight support modules left the top level of `quantlab/dataset/` and `quantlab/acquisition/` —
three up to `quantlab/` siblings, five down into new private `_support/` subpackages — and the
~306 references followed them. The two invariants that were held by prose alone are now held by
tests.

## The result, as an `ls`

```
quantlab/dataset/      __init__.py  _support  constituent.py  crsp  nbbo  spot.py  stock.py
quantlab/acquisition/  __init__.py  _support  alpaca.py  tiingo.py  wrds
```

Every top-level entry is one complete, usable thing.

## Numbers (executor, in the isolated worktree)

| Gate | Before | After | Verdict |
|---|---|---|---|
| Sweep (anchored pattern) | 306 | **0** | pass, with NO widening of the pattern |
| Dangling `quantlab/**.py` paths | 9 | 9 | **byte-identical** |
| Suite | 55 failed / 1695 passed / 1 skipped | 55 failed / **1697** passed / 1 skipped | pass |
| Failing node-ID set difference | — | **empty in BOTH directions** | pass |
| Collected tests | 1751 | 1753 | +2, fully accounted for |
| `git log --follow`, all 8 moved files | — | 32, 12, 28, 8, 6, 2, 3, 11 commits | history survived every move |
| Cold-interpreter import orders | — | 7/7 enumerate `['alpaca','tiingo','wrds']` | pass |
| Import-smoke over every `quantlab.*` module | — | 0 errors | pass |

### Accounting for the +2

The plan predicted +4. The measured delta is +2, and it is not a shortfall: the two emptiness
gates were added as *arms inside existing tests*, exactly as the plan's own step wording
instructs ("add a fourth arm to the structural check"). An arm adds no collected test. The real
+2 is the parametrize widening 3 -> 5.

## The gates that did not exist before

Both proved **red first, then green** — an emptiness assertion never observed to fail is
indistinguishable from one that cannot fail.

1. `tests/test_volume_guard.py` — a fourth arm asserting `quantlab/__init__.py` is 0 bytes, with
   the path resolved from the *module object* so it follows a future move.
2. `tests/test_source_inspector.py` — all three acquisition-chain `__init__.py` files, each
   tripped **independently** (a loop proved on only its first element leaves the other two
   unguarded).

`_FORBIDDEN_INSPECTOR_MODULES`'s non-vacuity was also proved rather than assumed: its stale-literal
failure mode is a *green test over an empty match*. Feeding the test's own `_resolved_imports` a
module that really does import the registry (`alpaca.py`) yields a non-empty intersection.

## Deviations from Plan

### 1. [Rule 3] A THIRD sweep-invisible reference class the plan did not enumerate

The plan states `tests/test_wrds_vendor_seam.py:58` is "the **only** sweep-invisible reference",
having exhaustively re-derived the `Path`-join class. It missed
`from quantlab.acquisition import universe as universe_module` — the from-package-import-submodule
form, which contains neither the dotted name nor the path, so no pattern of that shape can see it.

Caught by the **full-suite gate, not the sweep**:
`test_both_roster_queries_return_the_same_order_every_call` failed with
`ImportError: cannot import name 'universe' from 'quantlab.acquisition'` — the single new failure in
Task 1's run. The tracer task did its job.

Sweeping that class for all 8 movers found **11 more** naming `registry` across 7 files, three of
which (`tests/crsp_fixtures.py`, `test_crsp_constituent.py`, `test_crsp_dataset.py`) are absent from
the plan's `files_modified`.

### 2. [Rule 3] A FOURTH class: abbreviated spellings with no `quantlab.` prefix

The sweep's dotted alternative is anchored on `quantlab\.`, so bare `acquisition.universe` /
`dataset.backend` in prose are structurally invisible. **8 genuinely stale**, fixed.

**7 further hits deliberately NOT fixed** — pasted loguru transcripts in `example/*.md`. Evidence:
`git show 23e51de:example/dataset.md` shows they *already* read `dataset.cleaning` when the module's
real `__name__` was `quantlab.dataset.cleaning`. They are records of what a run printed, from an
older layout; rewriting a transcript falsifies it — the same argument the plan makes for
`.planning/phases/**`.

### 3. The sweep's last residual, each task, was the executor's OWN history-prose

Three times the final residual was a sentence just written naming a pre-move path *as history*.
Correct sentences. The pattern was **not** widened and the lines were **not** exempted: a gate that
tolerates dead literals in "good" prose cannot tell that from staleness. Meaning kept, literal
dropped; the sweep reached 0 honestly.

### 4. Two plan measurements are off (code fine, verify commands wrong)

- `monkeypatch` count: plan asserts 11. Measured 12 lines contain
  `quantlab.universe.requests.get`, of which 7 have `monkeypatch.setattr(` on the same line.
  Neither is 11. All 12 rewritten.
- D-2's layer count: plan says "seven modules across five layers". Measured seven modules across
  **four**. The seven is right and the conclusion holds; CLAUDE.md records the honest number.

## Historical records: proved untouched

`git diff --stat 23e51de -- .planning/phases .planning/ROADMAP.md .planning/research .planning/todos/completed`
-> **empty**.

Two live pointers still dangle and both pre-date this task: `quantlab/acquisition/progress.py`
(STATE.md:240 names it deliberately as the path that was *rejected*) and
`quantlab/dataset/nbbo_resample.py` (WINDOWS.md row 20, status `fixed`, closed 03.9). Repairing a
closed row would falsify it.

## Known Stubs

None.

## Post-merge re-verification (orchestrator, MAIN TREE)

The executor's green was taken in an isolated worktree, which this project has been burned by before
(`project_gsd_worktree_verify_path`). Re-measured independently on `main` after the merge, bytecode
caches cleared first:

- The menu is as promised, and all **five** `__init__.py` files measure 0 bytes:
  `quantlab/`, `quantlab/acquisition/`, `quantlab/dataset/` and both new `_support/`.
- Full suite: **55 failed, 1707 passed, 1 skipped**, against a pre-lu2 main baseline of
  55 / 1705 / 1. Failing node-ID set difference **empty in both directions**; the +2 is exactly the
  parametrize widening.

### One correction to the executor's own note

It attributed the main-tree/worktree 1-test difference to "this worktree has no `data/` directory".
That explanation does not fit: the difference it measured is in the COLLECTED count (1751 vs the
main tree's 1752), and a missing `data/` would change pass/fail, not collection. The actual cause is
`tests/test_entry_point_contracts.py:44`, `ENTRY_POINTS = sorted(REPO_ROOT.glob("*.py"))`, and the
extra file is `jerry_query_data.py` — **gitignored**, so invisible to `git status --porcelain` but
plainly visible to `glob`. Its gate was unaffected either way: before and after were both measured
inside the same tree with the same command.

## Human check still outstanding

Whether the two listings above *read* as a menu of complete, usable things is the judgement no
automated check makes.
