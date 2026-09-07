---
created: 2026-09-07T03:40:45.606Z
title: Support new listings in incremental zarr appends
area: database
severity: major
files:
  - dataset/backend.py:41-79 (XrBackend.append)
  - dataset/backend.py:98-135 (_assert_append_compatible — the guard that refuses)
  - base/data.py:271-300 (BaseDataset.from_raw_data_chunked — where D-02 pins the symbol axis)
---

## Problem

A cross-run incremental update **halts** when a stock has listed since the store was
created. It does not corrupt anything — but it cannot proceed either, and there is no
tooling to get past it.

`XrBackend.append()` calls `_assert_append_compatible()`, which requires every non-append
dimension's coordinate to match the store **exactly**. A new listing changes the `symbol`
union, so the append is refused:

```
XrBackend.append: refusing to append to <store> -- the 'symbol' coordinate does not
match the store (3 incoming label(s) vs 2 stored). ... Pin the 'symbol' axis over the
whole range before the first window, the way BaseDataset.from_raw_data_chunked() does.
```

**The guard is correct and must stay.** Measured 2026-09-06 on a synthetic store, the case
it prevents is genuinely silent:

| Scenario | raw `to_zarr(mode="a")` | `XrBackend.append()` |
|---|---|---|
| new listing, symbol **count** changes (2 → 3) | ValueError (zarr's own check) | ValueError, store intact |
| one delisting + one new listing, **count unchanged**, labels differ | **succeeds silently; history mis-attributed** | ValueError, store intact |

In the second row zarr overwrites the stored `symbol` coordinate labels while leaving the
data blocks in place, so every previously written `XYZ` observation ends up attributed to
`ARM`. Verified output: `rows 0-4 were written for XYZ but are now labelled: ARM`. Nothing
raises and the store carries no trace afterwards. That is what the guard exists to stop.

**The gap is the missing forward path, not the guard.** D-02's "resolve the symbol axis once
over the whole range before any window exists" solves this *within* a single chunked
backfill — every window is materialised onto one axis. It does not help across runs: the
store was written last month against last month's union, and this month's union is larger.
There is currently no code path that widens an existing store's symbol axis.

Reachable in normal operation: any periodic refresh of a US-equity store (new listings are
routine), and any market whose roster grows. Not currently exercised by tests or by any
shipped factory, because every fixture and factory builds a store in one pass.

## Solution

Two candidate directions — decide which before implementing:

1. **Widen in place.** Give `append()` (or a sibling) an "extend the symbol axis" path:
   `reindex` both the stored panel and the incoming window onto the union, backfilling the
   historical block with NaN for the newly listed symbols. Keeps the store, keeps history.
   Needs care: `reindex` on the stored side rewrites the whole store, so it is not cheap,
   and the chunk grid pinned by `APPEND_DIM_CHUNK` interacts with it.
2. **Detect and rebuild.** Have the `Dataset` layer compare the incoming union against the
   store's and trigger a full `from_raw_data_chunked()` rebuild when they differ. Simpler and
   obviously correct, but pays a full re-densify for one new ticker.

Whichever is chosen, the guard stays and the new path must be explicit — never a relaxation
of `_assert_append_compatible`, since the count-unchanged/labels-differ case above must keep
raising.

Note the interaction with the constituent panels from phase 03.1: a point-in-time membership
panel already carries an all-time symbol union, so it may already hold the union this needs.

**Verification requirement.** Any fix needs a test that appends a window containing a symbol
absent from the store *and* asserts the historical rows for pre-existing symbols are
unchanged afterwards — this project has ten recorded instances of tests that passed for the
wrong reason, and "the append succeeded" alone would not catch mis-attribution.
