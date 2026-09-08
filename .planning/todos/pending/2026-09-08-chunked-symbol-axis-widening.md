---
created: "2026-09-08T00:00:00.000Z"
title: Chunked symbol-axis widening, so the factor layer has a memory-bounded path
area: dataset / storage
severity: major
status: pending
blocked_by: 2026-09-08-no-memory-guard-before-a-symbol-axis-widen
---

# Chunked symbol-axis widening, so the factor layer has a memory-bounded path

## Problem

`XrBackend.widen_symbol_axis` rewrites the store onto a wider axis by materialising all
of it (`.reindex(...).load()`). `BaseDataset` can dodge that by rebuilding window by
window; `Factor` cannot, because it has no raw tier to re-read. So a factor panel whose
roster grows has exactly one path, and that path's peak memory is the whole store.

The pressure comes from high-frequency factors, which are computed from quotes and
trades. At 1-minute resolution, 3,000 symbols × 1 year × 20 factors is a 43.9 GiB
whole-store materialisation, triggered by a single IPO or delisting.

## Measured, 2026-09-08 — the mechanism works and is exact

Prototyped both paths against a synthetic store of 400 × 3,000 × 6 variables
(0.05 GiB), time-chunked at 20:

| approach | peak RSS increase |
|---|---|
| current `.reindex(...).load()` | **67.7 MiB** (≈ the whole store) |
| chunked loop | **1.1 MiB** (≈ one time-chunk) |

**Results were bit-identical** — same variables, same coordinates, and every value equal
under `array_equal(..., equal_nan=True)`.

The loop is small:

```python
for lo in range(0, n_t, TCHUNK):
    blk = stored.isel(timestamp=slice(lo, lo + TCHUNK)).load()
    blk = blk.reindex(symbol=new_syms)
    blk.to_zarr(out, append_dim="timestamp")   # first block: mode="w" + encoding
```

Peak becomes `time_chunk × new_symbol_count × variables × 8` instead of the total. For
the 1-minute case above, day-sized chunks put it near 190 MB.

**The chunk grid already exists.** The real store reports `chunks=(17, 7700)` — chunked
along time, spanning the full symbol axis. So widening the symbol axis reshapes every
chunk, but the chunks can be rewritten one at a time, and the boundaries are already
there to iterate over.

## Hand-written, not dask — decided 2026-09-08

dask is not installed (`import dask` fails; `xr.open_zarr(..., chunks={})` raises
`chunk manager 'dask' is not available`). Installing it would let xarray stream this
automatically. It was considered and rejected for now:

1. **It is a repo-wide behaviour change, not a local one.** `xr.open_zarr` would return
   dask arrays on EVERY read path — `to_kunquant`, the model layer, every `.values`
   call — and all 592 tests were written against the numpy backend. This repository was
   burned on 2026-09-08 by fixtures and production taking different array paths; adding
   a dependency that changes what production returns everywhere, while the test suite
   has never seen it, is the same hazard pointed the other way.
2. **The hand-written version is already proven bit-identical and is a dozen lines**,
   reusing the chunk grid the store already has rather than introducing a second
   chunking abstraction to describe it.
3. **Only this one operation needs streaming.** A plain append never reads the store at
   all — `_assert_append_compatible` touches coordinates and dtypes, not data — so the
   amortisation argument for dask has a single customer today.

If a second and third whole-store rewrite ever appear, that is the moment to re-open the
dask question. It is not this moment.

## Solution

TBD. Open questions a plan should settle rather than assume:

- **Does the chunked path replace the current one, or sit beside it?** Replacing keeps
  one path (this repo's stated preference — it has ruled twice against two live names
  for one thing). Keeping both needs a reason better than caution.
- **What picks the chunk size?** The store's own encoding is the obvious source, but a
  store chunked at 17 timestamps and one chunked at a day are very different loop
  counts. `TimeChunkPlanner` already exists for a related job.
- **The atomic swap must survive the loop.** The current `.load()` is load-bearing
  because lazily-indexed arrays read from a directory the swap renames. The prototype
  loads each block fully before writing it, so no lazy reference outlives an iteration —
  but the swap ordering needs to be re-verified against the loop, not assumed.
- **Does `widen_data_vars` need the same treatment?** It allocates one full-extent array
  per new variable rather than the whole store, so it is cheaper — but for a 1-minute
  panel one variable's full extent is still gigabytes.

## Files

- `quantlab/dataset/backend.py` — `widen_symbol_axis`, `widen_data_vars`,
  `widen_and_append`, `_append_encoding`, `APPEND_DIM_CHUNK`
- `quantlab/base/chunking.py` — `TimeChunkPlanner`, for the chunk-size question
- `quantlab/base/factor.py` — `Factor.update()`, the caller with no fallback

## Sequencing

After `[[2026-09-08-no-memory-guard-before-a-symbol-axis-widen]]`, which is what makes
this the named remedy rather than an optimisation, and after `260908-dvv`, whose
realistic fixtures should be in place before these methods are rewritten.

## Discovered

2026-09-08, from the question of whether installing dask would relieve the widening
memory cost. It would, and hand-writing the loop relieves it too, without the dependency.
