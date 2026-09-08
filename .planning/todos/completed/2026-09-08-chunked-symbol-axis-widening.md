---
created: "2026-09-08T00:00:00.000Z"
title: Chunked symbol-axis widening, so the factor layer has a memory-bounded path
area: dataset / storage
severity: major
status: completed
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

## Design revised 2026-09-08 — the guard ROUTES, it does not refuse

The developer's decision: when the estimate exceeds the budget, widening switches to the
chunked path automatically rather than refusing. That merges these two todos into one
deliverable — *widening picks its strategy by size, and says which* — and REVERSES their
order: a router cannot route to a path that does not exist, so the chunked path must land
first or both must land together.

**The threshold earns its place; measured 2026-09-08.** Chunked widening is not free:

| store | whole-store | chunked | ratio |
|---|---|---|---|
| 0.2 MiB | 0.04s | 0.04s | 1.2x |
| 34.6 MiB | 0.14s | 0.56s | **4.0x** |
| 137.3 MiB | 0.49s | 1.76s | **3.6x** |

So "always chunk" would make every routine widen ~4x slower. Where both fit, whole-store
wins; above the budget, whole-store does not run at all. The threshold sits exactly on
that boundary, which is why it is a router rather than a constant.

**The switch must be REPORTED, not silent.** Same shape as
`BaseDataset._reconcile_new_listings`, which names the qualifying symbols and their raw
row counts before a rebuild starts. Silently taking a path 4x slower reads to an operator
as their machine being slow.

Take these two files together as one brief.

## Closed

2026-09-08, by quick task `260908-g30` (commits `fc93304` router + tracer,
`1caa8c5` locks, and this documentation commit). See
`.planning/quick/260908-g30-route-symbol-axis-widening-by-size-add-a/260908-g30-SUMMARY.md`.

All FOUR open questions the brief posed were answered, and each answer now has
an address in the code:

- **Q1 -- both paths, or replace the whole-store one?** BOTH SHIP, behind one
  router. `XrBackend.widen_symbol_axis` stays the single public entry with an
  unchanged signature and no strategy parameter; `_widen_whole_store` (the
  shipped body, verbatim) and `_widen_chunked` sit behind it. The brief's own
  timing table decided it: at 1.2x / 4.0x / 3.6x on 0.2 / 34.6 / 137.3 MiB,
  "always chunk" taxes every routine widen ~4x to buy nothing. The repo's
  one-path rule forbids two live NAMES for one thing; one name with two
  size-selected private strategies is not that.
- **Q2 -- where does the block size come from?** From the BUDGET, floored onto
  the store's own chunk grid: `XrBackend._widen_block_rows`. `TimeChunkPlanner`
  was REJECTED with a stated reason -- its granularities are calendar periods,
  and a period's row count is a function of frequency and density (a month of
  1-minute bars is ~390x a month of daily bars), so it cannot bound BYTES,
  which is the entire constraint. The floor onto `APPEND_DIM_CHUNK` is
  load-bearing and measured: the first block carries the `encoding=`, so a
  100-row block leaves chunks `(100, 3)` where the whole-store path leaves
  `(512, 3)`.
- **Q3 -- does the atomic swap still hold with multiple writes?** RE-VERIFIED
  against the loop, not assumed. The loop runs inside the `try:` whose
  `finally` closes `stored` (it reads `path` through that handle for its whole
  duration); both `os.replace` calls stay outside it, in the shipped order; and
  the `except BaseException: rmtree(widening)` cleanup spans the WHOLE strategy
  call rather than one write. Locked by
  `tests/test_symbol_axis_widening.py::test_a_crash_part_way_through_the_block_loop_leaves_the_store_intact`.
- **Q4 -- does `widen_data_vars` need the same treatment?** YES, but OUT of
  scope here and FILED rather than dropped: the `append_dim` trick is
  unavailable there (extending the append dim would extend every already-full
  stored variable too), so a bounded version needs `region=` writes against a
  pre-created full-shape array -- a different write primitive. Carried with its
  measured cost (~2.4 GiB per variable on a 3,000-symbol 1-minute year) at
  `.planning/todos/pending/2026-09-08-chunked-widen-data-vars.md`.

The budget is `XrBackend.MAX_WIDEN_BYTES = 4 * 1024**3` -- the same figure as
`UniverseCatalog.MAX_DENSE_PANEL_BYTES` (same machine, same measured ceiling),
a SEPARATE constant because `quantlab/dataset/backend.py` has no import path to
the acquisition layer and must not grow one. It routes every measured scenario
in the brief correctly: 0.2 / 34.6 / 137.3 MiB and daily-7,700-symbols-x-1yr-x-20
at 0.3 GiB take whole-store; daily full history at 6.0 GiB, 1-minute 500 symbols
at 7.3 GiB and 1-minute 3,000 symbols at 43.9 GiB take chunked.

The switch is REPORTED and the report is asymmetric (D-6): the chunked branch
logs a `warning` naming both symbol counts, the estimate and the budget in GiB,
the rows per block, the block count, the per-block figure, the measured
~3.6-4.0x wall-clock cost (so a slow run does not read as a slow machine) and
how to opt back; the whole-store branch logs one `info` line. A `warning` on
every routine sub-budget widen would be noise.

The two now-false cost claims this brief's sibling identified were reconciled in
the same task: `widen_symbol_axis`'s `**Documented cost.**` paragraph,
`BaseDataset._reconcile_new_listings`'s `widen` warning, and the Chinese bullet
in `example/backend.md`. `on_new_listing="rebuild"` keeps a reason, but it is
HISTORY RECOVERY (it re-reads raw) rather than memory.
