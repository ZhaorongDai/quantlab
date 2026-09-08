---
created: "2026-09-07T00:00:00.000Z"
title: Guard the append dim against overlapping timestamps
area: data / storage
severity: major
status: pending
blocked_by: quick-260907-sm2
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
