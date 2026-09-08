---
created: "2026-09-08T00:00:00.000Z"
title: Chunked widen_data_vars, so the VARIABLE axis gets the bound the symbol axis has
area: dataset / storage
severity: major
status: pending
blocked_by: null
---

# Chunked `widen_data_vars`, so the VARIABLE axis gets the bound the symbol axis has

## Problem

260908-g30 bounded `XrBackend.widen_symbol_axis`: above `XrBackend.MAX_WIDEN_BYTES` it
rewrites the store block by block along `append_dim` and holds one block. Its sibling
`XrBackend.widen_data_vars` — the same method family, reached from the same
`widen_and_append()` call and therefore from the same `Factor.update()` — was left
UNBOUNDED, deliberately (260908-g30 D-4). It still builds each new variable's filler at
the store's FULL extent, in memory, before writing it.

The cost is real and is the same order as the one just fixed. One float64 variable over
a 3,000-symbol 1-minute year is:

    3000 symbols x 390 bars/day x 252 days x 8 bytes = ~2.4 GiB

per variable added. A factor set that grows three variables in one refresh allocates
~7.1 GiB, on the machine measured to OOM at ~7.2 GiB. And, exactly as on the symbol
axis, `Factor` has no `on_new_listing="rebuild"` escape — no raw tier, so this is its
only path.

## Why 260908-g30 did not close it

**The `append_dim` trick that makes the symbol-axis loop a dozen lines is UNAVAILABLE
here.** `_widen_chunked` works because extending `append_dim` is exactly what it wants:
each block appends to a store whose variables are all short by the same amount, so
`to_zarr(append_dim=...)` lands every block in the right place.

A new variable's filler needs the opposite. It must be materialised at the store's
EXISTING extent, alongside variables that are already full-length. Writing a filler
block with `append_dim=` would extend every already-full stored variable too — leaving
ragged lengths, which `_assert_append_compatible`'s docstring records as the one
corruption that makes the store un-OPENABLE afterwards.

So a bounded `widen_data_vars` needs a DIFFERENT write primitive: create the variable at
full shape first (e.g. an empty/`full`-valued array written once with the target shape
and the store's chunk grid), then fill it with `region=` writes block by block. That is a
different deliverable with its own failure modes — a partially-filled variable is a
state the symbol-axis path never produces — and folding it into 260908-g30 would have
put two write primitives and two crash-recovery stories in one change.

## What already exists to build on

- `XrBackend._estimate_widen_bytes(stored, requested, dim, append_dim)` — sizes the
  widened panel from an already-open handle. Sizing the VARIABLE axis needs the
  analogous "bytes of the variables about to be added", which is a near-sibling and
  should probably become one estimate with two callers rather than two estimates.
- `XrBackend._widen_block_rows(row_bytes)` — the budget-to-block-length rule, floored
  onto `APPEND_DIM_CHUNK` and a multiple of it. Reusable verbatim: the same grid
  constraint applies, and for the same measured reason.
- `XrBackend.MAX_WIDEN_BYTES` — the budget, and the router idiom around it
  (`_report_widen_strategy`'s asymmetric info/warning pair).
- `tests/test_symbol_axis_widening.py::test_the_two_widen_strategies_leave_identical_stores`
  — the equivalence-test SHAPE to copy: build once, `copytree`, widen each copy under a
  different forced budget, compare values (`equal_nan=True`), both coordinates, on-disk
  chunks, and the on-disk symbol encoding read through `conftest.stored_symbol_dtype`.
  The variable-axis version belongs in `tests/test_variable_axis_widening.py`, which is
  already one of the three suites `tests/test_widening_fixture_realism.py` guards.

## Open questions

1. Does `region=` interact correctly with the store's existing chunk grid when the
   region is not chunk-aligned? (The symbol-axis path sidesteps this by making every
   block a multiple of `APPEND_DIM_CHUNK`; the same discipline probably transfers, but
   it is unmeasured for `region=`.)
2. Crash recovery: the symbol-axis path writes to a `.widening.tmp` sidecar and swaps
   atomically. Does the variable path do the same (rewrite the whole store to a sidecar,
   bounded), or fill in place — and if in place, what makes a half-filled variable
   recoverable rather than silently NaN?
