---
created: "2026-09-08T00:00:00.000Z"
title: No memory guard before a symbol-axis widen
area: dataset / storage
severity: major
status: pending
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
