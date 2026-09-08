---
created: "2026-09-07T00:00:00.000Z"
closed: 2026-09-07
title: Guard the append dim against overlapping timestamps
area: data / storage
severity: major
status: closed
blocked_by: quick-260907-sm2
source: quick task 260907-uac
---

# Guard the append dim against overlapping timestamps

## Problem

`XrBackend._assert_append_compatible` refuses an append that would silently corrupt
the store, but it checks only the **non-append** dimensions and the variable dtypes.
The append dimension itself is skipped by the first line of its loop:

```python
for dim in self.data.dims:
    if dim == append_dim or dim not in existing.dims:
        continue
```

So nothing compares the incoming window's `timestamp` range against what the store
already holds. Two consequences, both silent at write time:

- **Overlap.** Appending a window whose dates overlap already-stored rows produces a
  store with duplicate timestamps. Nothing raises. The failure surfaces much later
  and somewhere else — `.sel()` and `to_xarray()`'s unique-index requirement both
  break on it — with no trace pointing back to the append that caused it.
- **Gap.** Appending a window that starts after the stored end leaves a store whose
  time axis is discontinuous, with no record that anything is missing. Described here
  for completeness only — see Solution: gaps are deliberately NOT guarded.

This is the same class the two existing guards already cover: zarr silently
re-attributing stored labels, and a float NaN cast into an integer store becoming a
fabricated `0`. Both were fixed on the principle of refusing rather than writing
something wrong quietly; the append dim is the remaining hole in that principle.

## Why it is not currently biting

No caller reaches it today. `BaseDataset.from_raw_data_chunked()` drives every
existing append through `ChunkLedger`, which owns the window boundaries and records
which windows are already written, so windows arrive contiguous and non-overlapping
by construction. The hazard is latent, not active — hence `major` rather than
`blocker`.

It stops being purely latent once the factor layer gets an incremental append path
(see the sibling work wiring `Factor.append()` to `widen_and_append`). A factor
panel is far more likely than a raw-market panel to be recomputed over a date range
that was already stored — re-running a factor over an overlapping window is an
ordinary thing to do by hand, and there is no ledger on that path.

## Solution

Extend `_assert_append_compatible` to compare the incoming `append_dim` coordinate
against the stored one and refuse an **overlap**, with an error message that names
the stored end, the incoming start, and what to do instead — the way the
coordinate-mismatch message already names `BaseDataset.from_raw_data_chunked()`.

Two questions were open when this was captured. Both were decided by the developer
on 2026-09-07, so whoever plans this should implement them rather than re-open them:

- **Gaps are NOT guarded.** Only overlap is. A discontinuous time axis is not treated
  as an error here.
- **There is NO opt-out for overwriting a range, and none should be added.** The
  refusal is unconditional. Recomputing a date range that is already stored is
  `save(mode="w")`'s job — replace the store — and `append()` exists to extend it.
  Giving `append()` an overwrite escape hatch would blur exactly the boundary that
  `save()` and `append()` are separate methods in order to keep sharp. If a future
  reader believes they need one, the thing they actually need is the rewrite path.

## Scope boundary

This is a **storage-layer** concern and belongs to `XrBackend`. It is not
`Factor`'s and not `Dataset`'s.

Explicitly NOT this todo: rolling-window warm-up. Warm-up is a property of `cal()` —
whether a factor value was computed with enough lookback to be valid. The storage
layer cannot distinguish a warm-up artifact from a genuine low value and must not
try; conflating the two was considered and rejected on 2026-09-07.

## Files

- `dataset/backend.py` — `XrBackend._assert_append_compatible` (the skip is the first
  line of its dim loop), `XrBackend.append`, `XrBackend.widen_and_append`
- `tests/test_symbol_axis_widening.py` — the established test idiom for this
  method's guards; a new case belongs alongside these

Paths are pre-migration. Quick task `260907-sm2` moves them under `quantlab/`.

## Blocked by

`260907-sm2` (the `quantlab.*` namespace package migration) relocates
`dataset/backend.py` to `quantlab/dataset/backend.py`. Do this after that lands.

## Discovered

2026-09-07, while reviewing guard coverage on the append path in order to wire the
factor layer to `widen_and_append`.

## Closed 2026-09-07

Closed by quick task `260907-uac`. The hole is shut: `XrBackend.append` now
refuses an overlapping window before `to_zarr`, and the store is bit-identical
after a refusal.

### What shipped

One check inside `quantlab/dataset/backend.py::XrBackend._assert_append_compatible`,
placed between the non-append coordinate loop and the dtype loop so no existing
refusal's precedence changed. It compares the incoming `append_dim` MINIMUM
against the stored MAXIMUM — `.min()`/`.max()` rather than positional indexing,
so an unsorted axis on either side cannot fool it — and raises when the incoming
start is less than **or equal to** the stored end. Equality must refuse: a window
starting exactly on the stored end duplicates that one label.

It skips when either side carries no coordinate on the append dimension, mirroring
the skip the dim loop above already applies. That case is not hypothetical:
measured 2026-09-07, a panel with a `timestamp` DIM and no `timestamp` COORD
writes and re-appends cleanly, and without the skip a working path becomes a crash.

Labels render through a small `_format_append_label` helper (`pd.Timestamp(...).isoformat()`
for a `datetime64`, `str()` otherwise), because `append_dim` is a parameter and
this guard must not become timestamp-only.

Eight tests: one tracer in `tests/test_chunked_ingest.py` plus six siblings there,
and one inheritance test in `tests/test_symbol_axis_widening.py`. Suite 537 -> 545.

### The refusal message, and why it is worded as it is

The message names the store path, the dimension, the stored end, the incoming
start, and the remedy `save(mode="w")`. Its stated consequence is that the axis
is left **no longer STRICTLY increasing — duplicate labels, out-of-order labels,
or both**.

That wording is measured, not chosen. All three refusable shapes were run against
the unguarded backend on 2026-09-07:

| incoming window vs. store | `is_unique` | `is_monotonic_increasing` |
|---|---|---|
| partial overlap (`01-05..01-07` into `01-04..01-06`) | False | False |
| starts exactly on the stored end (`01-06..01-08`) | False | **True** |
| ends before the stored start (`01-04` into `06-01..06-02`) | **True** | False |

A consequence phrased as "duplicate labels" is falsified by the third row; one
phrased as "out-of-order labels" is falsified by the second. Strictly-increasing
is the only property all three break, so it is the only honest wording — do not
weaken it. All three refusal tests assert the SAME clause via `_OVERLAP_CONSEQUENCE`,
so a wording true of only one shape cannot survive by being checked only where it
happens to hold.

### Mutation results

Every lock was mutation-verified, each mutation applied to a clean tree and
reverted afterwards (`git diff --quiet quantlab/dataset/backend.py` clean at the end).

- **M1** delete the check — the overlap, ends-before, starts-exactly-on,
  smuggled-kwarg and widen-inherits tests reddened together (4 via `DID NOT RAISE
  ValueError`; the kwarg test via `TypeError: Dataset.to_zarr() got an unexpected
  keyword argument 'force'`, which is exactly the unguarded-tree behaviour recorded
  at planning time).
- **M2** relax `<=` to `<` — **exactly one** reddening,
  `test_append_refuses_a_window_starting_exactly_on_the_stored_end`. The overlap
  test stayed green, so the fixtures do isolate what their docstrings claim. That
  boundary is covered by that single test and no other; do not delete it as redundant.
- **M3** compare the incoming MAXIMUM instead of its minimum — the overlap and
  starts-exactly-on tests reddened as predicted, plus the kwarg and widen-inherits
  tests for the identical mechanism (their windows also extend past the stored end).
  The ends-before test stayed GREEN, which confirms the mechanism precisely: its
  incoming max lies below the stored max, so the mutated comparison still refuses it.
- **M4** a second, differently-worded overlap check at the top of `widen_and_append`
  — exactly one reddening, `test_widen_and_append_inherits_the_overlap_refusal_verbatim`,
  on the message-equality assertion. D-03 is enforced, not merely requested.
- **M5** extend the guard to also refuse a gap — the gap test reddened through a
  `ValueError` on a legitimate gap. **21 other tests reddened with it**, and that is
  the finding worth carrying forward: the existing suite and the production chunked
  path are full of deliberately gapped appends, because `TimeChunkPlanner`'s windows
  are built from OBSERVED timestamps and are sparse by construction. Refusing gaps
  would not be a stricter guard, it would break the ingest path outright.
- **M6** an escape hatch popped from `**kwargs` INSIDE the method body, ahead of the
  guard call — the split the whole exercise exists to demonstrate:
  `test_append_refuses_an_overlapping_window_carrying_an_unrecognised_kwarg` reddened
  on `DID NOT RAISE ValueError`, while `test_append_offers_no_overwrite_escape_hatch`
  stayed **GREEN** with `inspect.signature` returning a byte-identical
  `('self','path','append_dim','kwargs')`. The structural proxy alone does not span
  D-02; the pair does.

### The two decided points were implemented, not re-opened

- **Gaps are NOT guarded (D-01).** Only overlap is refused. A discontinuous time
  axis is a legitimate shape this layer takes no position on — the storage layer
  cannot tell a deliberately sparse range from a missing one — and M5 above shows
  that guarding it breaks the ingest path. `test_append_allows_a_gap_between_the_stored_end_and_the_incoming_start`
  exists so a later reader cannot "finish the job"; extending the guard to
  contiguity reddens there, which is the point.
- **There is NO opt-out, declared or smuggled (D-02).** The refusal is
  unconditional. Recomputing a range the store already holds is `save(mode="w")`'s
  job — replace the store — and `append()` exists to extend it; an escape hatch
  would blur exactly the boundary those two methods are separate in order to keep
  sharp. This is locked by a PAIR of tests, and M6 shows why one is not enough. If
  you arrive here believing you need a `force=`, the thing you actually need is the
  rewrite path.

`widen_and_append` gained NO parallel check (D-03) — it inherits the refusal
through its closing, unmodified `append()` call, proved by the two paths producing
a byte-identical message once the store path is normalised out. Its docstring now
records the accepted side effect that test asserts: the widen commits before the
closing `append()` raises, so a refused window can leave the store on the GROWN
symbol axis with its timestamp axis untouched and its history intact.

Rolling-window warm-up remains out of scope (D-04), for the reason the Scope
boundary section above already gives.
