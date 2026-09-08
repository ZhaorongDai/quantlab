---
phase: quick-260908-dvv
plan: 01
status: complete
subsystem: testing
tags: [zarr, xarray, numpy, coordinate-encoding, fixtures, mutation-testing, worktree]
reconstructed: true
---

# Make the widening suites exercise the coordinate dtypes production writes

> **This file is a RECONSTRUCTION, and that fact is itself the first finding.**
>
> The executor wrote a summary inside its isolated worktree and, per the orchestrator's
> instruction not to commit docs artifacts, left it uncommitted. The orchestrator then
> removed the worktree with `--force`, destroying it. `git fsck` recovered no matching
> blob.
>
> The instruction "do not commit SUMMARY.md — the orchestrator handles the docs commit"
> is correct for SEQUENTIAL execution on the main tree, where an uncommitted file simply
> waits. It is **incompatible with worktree isolation**, where an uncommitted file dies
> with the worktree. Either the executor commits the summary inside the worktree, or the
> orchestrator copies it out before removal. Neither happened.
>
> What follows is rebuilt from the executor's return message and from the verifier's
> INDEPENDENT re-derivation, which reproduced every load-bearing number itself rather
> than reading them. Where the two disagree, the verifier's measurement is recorded and
> the disagreement is named.

## What was wrong

The three suites that own the axis-widening methods built their `symbol` coordinate from
python list literals. A production panel does not: it is written to zarr and re-opened,
so it carries `object` on disk and `StringDType()` in memory. Every one of those tests
exercised a dtype the production path never produces.

That let a real defect ship. `widen_data_vars` rebuilt the store's coordinates from the
opened dataset and wrote them back, raising `Mismatched dtypes for variable symbol`.
Variable widening was unreachable for every store the chunked ingest writes — reachable
that way from `Factor.update()`, which passed a 13/13 verification one task earlier.

## The number that defines the task

With `quantlab/dataset/backend.py` at `dea1e85` (pre-fix), the three OWNING suites:

| | before this task | after |
|---|---|---|
| owning suites, pre-fix backend | `34 passed` | **`8 failed, 59 passed`** |

Verifier re-derived: RED=8, VL=8, **FW=0** — all eight are `[variable_length]` ids matching
WFR-07 verbatim, every `[fixed_width]` twin green. HEAD control `67 passed`. Restored and
byte-hash verified.

The lock now lives with the methods. It previously sat, by accident, in
`tests/test_chunked_ingest.py`, which was not edited.

## What was built

- A shared coordinate-encoding helper with six self-tests, at COORDINATE granularity —
  panel granularity does not fit, because the three suites carry three incompatible panel
  idioms.
- Two-arm parametrisation (`[fixed_width]` / `[variable_length]`) over 33 of 34 tests.
- `assert_stored_symbol_encoding`, a post-widen encoding lock.
- An AST family guard, so a future test added to these suites cannot quietly skip the
  parametrisation.

## Why two arms and not one or three

Forced by two mutations that redden **disjoint** sets:

| mutation | `[fixed_width]` | `[variable_length]` |
|---|---|---|
| M1 (the shipped defect) | all green | raises |
| M4 (`assign_coords` after reindex) | green | no raise, values correct, store silently downgraded |

A third arm was refused on evidence: at natural label widths `<U3` and `<U9` measured
identical, and the `float64` store is 0×0 — an empty store, not a string encoding.

## The finding that outlived the task

**M4's defect class is invisible to a pass/fail battery.** It raises nothing and leaves
every value, label, NaN and chunk correct — it only rewrites the store's encoding.

The verifier ran a control the plan never did: M4 applied **and**
`assert_stored_symbol_encoding` neutralised to a no-op → **`67 passed`, rc=0**. So the
explicit encoding assertion is not a nicety; it is the only observable that exists for
that class.

## The spelling trap

`np.array(dtype=np.dtypes.StringDType())` — the spelling that names the DECODED dtype —
writes a fixed-width array, i.e. the WRONG arm. Only `object` (or `pd.Index`) reproduces
production. The helper's self-tests exist for this reason; the verifier reproduced the
whole five-row round-trip table, and found M5 reddens 5 of 6 self-tests where the plan
predicted 2 — stronger than claimed, not weaker.

## Deviations, none of which were written down at the time

1. **`from conftest import` instead of the plan's `from tests.conftest import`** (WFR-03),
   and an edit to `tests/test_raw_hive_layout.py`, a file outside the plan's
   `files_modified`. Cause: vectorbt **0.28.2** ships a top-level REGULAR `tests` package
   into site-packages, which Python resolves ahead of this repo's `tests/` namespace
   portion. The worktree's fresh, lock-conformant venv hit it; the ambient `~/.venv` did
   not, because it carries vectorbt 1.1.0 in violation of the lock.
   - Adding `tests/__init__.py` was tried and rejected. The verifier reproduced the
     trade-off: it trades one break for six — `ModuleNotFoundError: No module named
     'test_universe'` at `test_ticker_pattern_reconciliation.py:276` plus collection
     errors in all five new/rewired suites.
   - **Correction to the executor's account:** line 276 relies on a bare *sibling-module*
     import (`test_universe`), not on `conftest`.

2. **Task 2's `<precondition>` was never made executable.** The plan required
   `git status --porcelain -- quantlab/` to be clean BEFORE the verify checks out an old
   commit, and the orchestrator asked for it to be prepended. It was not. The verify still
   opens with `git checkout dea1e85 --` and only asserts emptiness after the restore, so
   the precondition remains prose rather than a gate. The restore path itself is sound and
   was exercised cleanly; the risk is at the entry, on a dirty tree.

## Near-miss, recorded rather than inflated

A `<U9` store meeting a `<U3` panel does raise, at HEAD too. It is not a defect, on two
grounds the verifier checked: the superset guard refuses a subset outright and a
superset's natural width cannot shrink; and an `object` panel appended to a `<U9` store
passes with no raise. The raise is only reachable by artificially pinning a wide dtype
onto short labels, and the honest route trips `append`'s label-mismatch guard first.

## Gate

`uv run pytest tests/ -q` → **634 passed** (592 + 42), zero failures. Per-task gates
614 → 631 → 634.

Independently confirmed by the orchestrator in BOTH environments after the merge: the
ambient `~/.venv` (vectorbt 1.1.0) and a fresh lock-conformant clone (`uv sync`,
vectorbt 0.28.2). Both 634.

## Not changed

No production file. `git diff` over `quantlab/` and `tests/test_chunked_ingest.py` is
empty across the whole commit range and in the working tree.
