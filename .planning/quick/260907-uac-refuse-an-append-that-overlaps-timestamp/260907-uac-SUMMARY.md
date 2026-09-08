---
phase: quick-260907-uac
plan: 01
subsystem: data / storage
status: complete
tags: [xarray, zarr, append-guard, data-integrity, tdd, mutation-testing]
requires:
  - quantlab/dataset/backend.py::XrBackend._assert_append_compatible (the existing single call site)
  - quick task 260907-sm2 (the quantlab.* namespace migration this was blocked on)
provides:
  - "XrBackend.append refuses an append_dim window overlapping the stored range, before to_zarr"
  - "XrBackend._format_append_label — human-readable append-dim labels in refusals"
  - "widen_and_append inherits the refusal verbatim, with no parallel check"
affects:
  - BaseDataset.from_raw_data_chunked (the only production caller; satisfied by construction)
  - Factor.append / the future incremental factor path (the reason this stopped being latent)
tech-stack:
  added: []
  patterns:
    - "Refuse rather than write something wrong quietly — the third guard built on this principle in quantlab/dataset/backend.py"
    - "Lock a behavioural property with a PAIR of tests when a structural proxy provably cannot span it"
    - "Assert one shared consequence clause across every fixture a message must be true of"
key-files:
  created: []
  modified:
    - quantlab/dataset/backend.py
    - tests/test_chunked_ingest.py
    - tests/test_symbol_axis_widening.py
    - .planning/todos/completed/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md
decisions:
  - "D-01 implemented, not softened: gaps are NOT guarded — only overlap is refused. M5 showed guarding gaps reddens 21 tests beyond the gap lock, because TimeChunkPlanner windows are built from observed timestamps and are sparse by construction."
  - "D-02 implemented as a PAIR of locks: the parameter tuple plus a behavioural test, because a kwargs-smuggled hatch leaves inspect.signature byte-identical (demonstrated live by M6)."
  - "D-03 implemented: widen_and_append gained no parallel check and inherits the refusal through its closing unmodified append(); proved by byte-identical messages across both paths."
  - "Refusal message states the axis is left 'no longer STRICTLY increasing' — the only consequence true of all three refusable window shapes, established by measurement rather than by wording preference."
metrics:
  duration_minutes: 14
  completed: 2026-09-07
  tasks: 3
  tests_added: 8
  suite_before: 537
  suite_after: 545
actuals:
  tokens: 10763
  tasks: 3
  commits: 3
plan_head_before: ab3049eb09332595a2b06157aacb362989f15fe1
---

# Quick 260907-uac: Refuse an Append That Overlaps Stored Timestamps — Summary

`XrBackend.append` now refuses an overlapping `append_dim` window before `to_zarr`
rather than silently producing a store whose time axis is no longer strictly
increasing — closing the last hole in `_assert_append_compatible`, which guarded
every dimension except the one being appended along.

## What Shipped

One check inside `quantlab/dataset/backend.py::XrBackend._assert_append_compatible`,
placed between the non-append coordinate loop and the dtype loop so no existing
refusal's precedence changed and no currently-green test could change outcome.

It compares the incoming `append_dim` **minimum** against the stored **maximum**,
using `.min()`/`.max()` rather than positional indexing so an unsorted axis on
either side cannot fool it, and raises when the incoming start is less than **or
equal to** the stored end. Equality must refuse: a window starting exactly on the
stored end duplicates that one label.

The check skips when either side carries no coordinate on the append dimension,
mirroring the skip the dim loop above already applies. That case is not
hypothetical — measured 2026-09-07, a panel with a `timestamp` dim and no
`timestamp` coord writes and re-appends cleanly, and without the skip a working
path becomes a crash.

Labels render through a small `_format_append_label` static helper
(`pd.Timestamp(...).isoformat()` for a `datetime64`, `str()` otherwise), because
`append_dim` is a parameter and this guard must not become timestamp-only.

Two docstrings updated: `append()` gained the append dimension as its third
enforced property and states plainly that a gap is permitted; `widen_and_append()`
gained one sentence recording the accepted half-widened side effect.

Eight tests — seven in `tests/test_chunked_ingest.py`, one in
`tests/test_symbol_axis_widening.py`. Suite 537 → 545.

## The Refusal Message, and Why It Is Worded As It Is

The message names the store path, the dimension, the stored end, the incoming
start, and the remedy `save(mode="w")`. Its stated consequence is that the axis is
left **no longer STRICTLY increasing — duplicate labels, out-of-order labels, or
both**.

That wording was measured, not chosen. All three refusable shapes, run against the
unguarded backend:

| incoming window vs. store | `is_unique` | `is_monotonic_increasing` |
|---|---|---|
| partial overlap (`01-05..01-07` into `01-04..01-06`) | False | False |
| starts exactly on the stored end (`01-06..01-08`) | False | **True** |
| ends before the stored start (`01-04` into `06-01..06-02`) | **True** | False |

A consequence phrased as "duplicate labels" is falsified by the third row; one
phrased as "out-of-order labels" is falsified by the second. Strictly-increasing is
the only property all three break. All three refusal tests assert the identical
clause via a shared `_OVERLAP_CONSEQUENCE` constant, so a wording true of only one
shape cannot survive by being checked only where it happens to hold.

## TDD Gate Compliance

**RED observed before implementation, and observed for the right reason.** The
tracer test was written first and run against the unmodified backend. It failed
with:

```
E       Failed: DID NOT RAISE ValueError
tests/test_chunked_ingest.py:404: Failed
```

That is a behavioural RED, not a collection error, an import error or a fixture
error — a RED that fails for a setup reason proves nothing. GREEN followed the
single check in `_assert_append_compatible`; no test was adjusted to fit the
implementation.

## Mutation Verification

Every lock was mutation-verified. Each mutation was applied to a clean tree, the
predicted test(s) confirmed to redden **through the assertion they claim to
exercise**, then reverted — `git diff --quiet quantlab/dataset/backend.py` clean
at the end.

| Mut | Change | Predicted | Observed |
|-----|--------|-----------|----------|
| M1 | delete the whole new check | overlap, ends-before, starts-exactly-on, widen-inherits redden together | ✅ all four reddened via `DID NOT RAISE ValueError` (lines 405/494/540/453), plus the smuggled-kwarg test via `TypeError: Dataset.to_zarr() got an unexpected keyword argument 'force'` — exactly the unguarded-tree behaviour recorded at planning time |
| M2 | relax `<=` to `<` | **exactly** starts-exactly-on reddens; overlap stays green | ✅ 1 failed / 544 passed. Only `test_append_refuses_a_window_starting_exactly_on_the_stored_end`, via `DID NOT RAISE ValueError` at line 540. The "nothing else may redden" clause held |
| M3 | compare incoming **max** vs stored max | overlap **and** starts-exactly-on; further reddenings expected via the same mechanism | ✅ both predicted reddened, plus the kwarg and widen-inherits tests for the identical reason (their windows extend past the stored end). **The ends-before test stayed GREEN**, confirming the mechanism precisely — its incoming max lies below the stored max, so the mutated comparison still refuses it |
| M4 | second, differently-worded check atop `widen_and_append` | verbatim-message test reddens on message equality | ✅ 1 failed / 544 passed, on the equality assertion, diffing the `M4 MUTATION:` string against the inherited message |
| M5 | extend the guard to also refuse a gap | gap test reddens | ✅ reddened via a `ValueError` on a legitimate gap — **plus 21 others**, see below |
| M6 | hatch popped from `**kwargs` **inside** the body, ahead of the guard call | behavioural test reddens **AND** signature test stays **GREEN** | ✅ exactly the split. `..._carrying_an_unrecognised_kwarg` FAILED on `DID NOT RAISE ValueError` (line 661); `test_append_offers_no_overwrite_escape_hatch` **PASSED**, with `inspect.signature` returning a byte-identical `('self','path','append_dim','kwargs')`. 1 failed / 544 passed |

**M6 is the result that justifies the whole exercise.** The structural proxy alone
stays green through precisely the drift D-02 exists to catch; only the pair spans
the property. Neither half is sufficient, and the half a later reader is most
likely to find is the one that does not span it — which is why both docstrings
cross-reference each other.

### M5 turned up a finding stronger than the plan anticipated

M5 was first written bluntly (`or True`, refusing every append) and re-done as a
genuine gap check (`incoming_start > stored_end + 1 day`). Both forms produced the
same 22 failures. The 21 collateral reddenings are the finding: the existing suite
**and the production chunked ingest path** are full of deliberately gapped appends,
because `TimeChunkPlanner`'s windows are built from *observed* timestamps and are
sparse by construction. D-01 is therefore not a stylistic preference — refusing
gaps would not be a stricter guard, it would break the ingest path outright. That
reasoning is recorded in the closed todo so a later reader who wants to "finish the
job" finds it before the code.

## Decisions Honoured, Not Re-opened

- **D-01 — gaps are not guarded.** Only overlap is refused, and
  `test_append_allows_a_gap_between_the_stored_end_and_the_incoming_start` reddens
  under any attempt to extend the guard to contiguity.
- **D-02 — no opt-out, declared or smuggled.** `XrBackend.append`'s parameter list
  is unchanged, and the behavioural test proves no `**kwargs` route exists. The
  mechanism: `append()` calls `_assert_append_compatible` before it touches
  `kwargs` at all.
- **D-03 — `widen_and_append` gained no parallel check.** It inherits the refusal
  through its closing, unmodified `append()` call. Verified non-vacuously: the
  widen genuinely runs (the store's symbol axis grew to `['A','B','C']`) before the
  inherited refusal fires.
- **D-04 — rolling-window warm-up remains out of scope.**

## Deviations from Plan

**None affecting behaviour.** Two execution-level notes, both recorded rather than
smoothed over:

1. **M5 was re-run in a more faithful form.** The first mutation (`or True`)
   refused every append rather than only gaps, which is broader than "extend the
   guard to also refuse a gap". It was reverted and re-applied as a genuine
   one-day-gap check. Both forms reddened the gap test identically; the second is
   the one reported.
2. **Task 3's commit used a `.planning/todos/` directory pathspec** rather than the
   two file paths. `git mv` had already staged both sides of the rename, so
   `git add` on the now-absent pending path failed with `did not match any files`.
   The directory pathspec covers both sides of the rename, is still explicit, and
   cannot reach `test.py`.

`test.py`'s pre-existing uncommitted modification was not staged, committed,
reverted or otherwise touched at any point. Every commit used explicit pathspecs;
no `git add -A`, no amend.

## Verification

- `uv run pytest tests/ -q` → **545 passed**, zero failures (537 live baseline + 8
  new). Real output, unadjusted.
- Prose gate: backticked pre-migration package paths count `1 / 0 / 0` across the
  three touched files — unchanged from planning time, so nothing was added to the
  open doc-sweep backlog. The pre-existing hit inside `widen_and_append`'s
  docstring was left byte-for-byte alone.
- Todo is under `.planning/todos/completed/` via `git mv`, `status: closed`, with a
  `## Closed 2026-09-07` section recording what shipped, the mutation results and
  the decided points.
- Post-commit deletion check: the only deletion across the three commits is the
  intentional todo rename. No untracked files left behind.

## Known Stubs

None. No `TODO`/`FIXME`/placeholder markers, no skipped or xfailed tests, and no
unrun `<verify>` blocks — every verify command in the plan was executed and its
real output reported.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 (tracer) | `406b557` | `feat`: refuse an append overlapping the stored timestamps |
| 2 | `230eb73` | `test`: lock the decided boundaries of the overlap refusal |
| 3 | `8249ded` | `chore`: close the append-dim overlap todo |

Measured: `git rev-list --count ab3049eb..HEAD` = **3**.

## Self-Check: PASSED

All four modified/moved artifacts exist on disk. All three commit hashes
(`406b557`, `230eb73`, `8249ded`) resolve in `git log`. The guard's consequence
clause is present in `quantlab/dataset/backend.py`. All eight new test functions
are present — 7 in `tests/test_chunked_ingest.py`, 1 in
`tests/test_symbol_axis_widening.py`.
