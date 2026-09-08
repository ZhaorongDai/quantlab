---
created: "2026-09-08T00:00:00.000Z"
title: No memory guard before a symbol-axis widen
area: dataset / storage
severity: major
status: completed
blocks: 2026-09-08-chunked-symbol-axis-widening
---

# No memory guard before a symbol-axis widen

## Problem

`XrBackend.widen_symbol_axis` materialises the WHOLE store in RAM:

```python
widened = stored.reindex({dim: requested}, fill_value=fills).load()
```

Nothing checks first whether it fits. A widen on a store larger than memory does not
refuse — it OOMs, part-way through an operation the operator has usually already been
waiting on for a long time.

**A newly added symbol having no history does not reduce this at all.** The widen is
about the AXIS, not about the new symbol's data: every existing symbol's every
historical value is rewritten into the wider array layout. An IPO with zero rows costs
exactly as much as one with full history.

This is the one growth axis in the storage layer with no volume guard. The repository
already guards the comparable cases: `assert_acquisition_volume_fits` before a
download, `assert_chunked_panel_fits` before a dense-panel conversion. Widening was
missed.

## Measured, 2026-09-08

Peak materialisation for one widen, float64:

| shape | whole-store materialisation |
|---|---|
| daily, `us_all` today (7,700 × 21) × 1 var | negligible |
| daily, 7,700 × 1 year × 20 factors | 0.3 GiB |
| daily, 7,700 × full history (5,215d) × 20 factors | **6.0 GiB** |
| 1-minute, 500 × 1 year × 20 factors | **7.3 GiB** |
| 1-minute, 3,000 × 1 year × 20 factors | **43.9 GiB** |

The trigger is a roster change of one name — a single IPO or delisting is enough.

`.load()` is deliberate and cannot simply be dropped. Its comment records why: without
dask, `open_zarr` hands back lazily-indexed arrays that read from the store directory
on access, and the method's atomic swap renames that directory out from under them.

## Why the factor layer is the sharp case

`BaseDataset` has an escape: when a widen would be too large, `on_new_listing="rebuild"`
re-densifies window by window, so peak memory is one window. `Factor` has no equivalent
— there is no raw tier for it to re-read — so `Factor.update()` reaches
`widen_symbol_axis` as its only path. The safety valve that exists on the dataset side
is absent exactly where high-frequency factor panels make the problem worst.

## Solution

TBD. The shape that matches the repo's existing guards: estimate the materialisation
before `.load()` and refuse above a budget, naming the figure, the budget, and the
alternative — the way `assert_chunked_panel_fits` already names a finer `--chunk` as
its remedy.

What the refusal should point at is the open question, and it is why this todo is
paired with the chunked-widening one:

- On the **dataset** side it can say `rebuild`, which exists today.
- On the **factor** side there is currently nothing to name. Until
  `[[2026-09-08-chunked-symbol-axis-widening]]` lands, a refusal there is honest but
  leaves the operator stuck.

So this guard is worth having on its own — a stated refusal beats an OOM either way —
but the pair is what makes it actionable.

## Files

- `quantlab/dataset/backend.py` — `widen_symbol_axis` (the `.load()` and the swap it
  protects), `widen_and_append`
- `quantlab/base/data.py` — `_reconcile_new_listings`, the `widen`/`rebuild` branch and
  the warning it already prints about this cost
- `quantlab/acquisition/universe.py` — `assert_chunked_panel_fits`, the guard idiom to
  follow

## Sequencing

After `260908-dvv` (realistic widening fixtures). That task adds tests to these same
methods; changing their behaviour concurrently would collide, and landing it first
means this guard is caught by fixtures that exercise the real encoding path rather than
by the ones just shown to have a blind spot.

## Discovered

2026-09-08, while answering whether a newly added symbol with no history still forces a
full read. It does.

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
