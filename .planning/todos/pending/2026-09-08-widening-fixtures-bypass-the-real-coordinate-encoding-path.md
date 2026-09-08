---
created: "2026-09-08T00:00:00.000Z"
title: Widening fixtures bypass the real coordinate encoding path
area: testing
severity: major
status: pending
---

# Widening fixtures bypass the real coordinate encoding path

## Problem

The three suites that OWN the axis-widening methods build their `symbol` coordinate
from python list literals. A panel that reaches those methods in production does not:
it is written to zarr and re-opened, and zarr records a string coordinate as `object`
while `xr.open_zarr` decodes it back to numpy's `StringDType()`.

So every one of those tests exercises a dtype the production path never produces.

This is not a hypothetical. `XrBackend.widen_data_vars` rebuilt the store's
coordinates from the opened dataset and wrote them back, which raises

```
ValueError: Mismatched dtypes for variable symbol between Zarr store on disk and
dataset to append. Store has dtype object but dataset to append has dtype
StringDType().
```

Variable widening was therefore unreachable for every store the current chunked
ingest writes — and reachable that way from `Factor.update()`, which shipped one
task earlier (`260907-vyr`) under a **13/13 passing verification**. The defect was
found and fixed in `260908-0f4` only because that task happened to build its
fixtures through the real `StockDataset.from_raw_data_chunked`.

## Why this is different from the other test gaps

Ten separate times in three days this repository has shipped a gate that passed for
the wrong reason: an unanchored grep, a substring scan matching a leftover import, a
test deriving its expected value from the same expression as the bug, a character
class terminating at `(`, a structural proxy that could not span a behavioural
property, a pass-count pattern matching `3 failed, 592 passed`.

Every one of those was a gate written incorrectly, and every one was catchable by
reading the gate carefully enough.

This one is not. The gates are all correct. What is wrong is that **an entire family
of fixtures shares one unrealistic assumption**, so the suite is rigorous only inside
that assumption. No amount of care spent on any individual test would have surfaced
it. That makes it the more expensive shape: it produces defects that pass full
verification and then fail on real data.

## Measured, 2026-09-08

**The blind spot is still open.** Reverting only `quantlab/dataset/backend.py` to the
pre-fix commit reddens 4 of the 5 new tests in `tests/test_chunked_ingest.py` — but
the suites that own the widening methods all stay green against that same broken
backend:

| suite | tests | result against the broken backend |
|---|---|---|
| `tests/test_symbol_axis_widening.py` | — | all pass |
| `tests/test_variable_axis_widening.py` | — | all pass |
| `tests/test_factor_update.py` | — | all pass |
| (combined) | 34 | **all pass** |

The lock is now incidental: it sits in the chunked-ingest suite rather than where the
method lives, and the factor path has no object-dtype fixture at all.

**Production carries at least three different symbol dtypes**, none of them the one
the fixtures use:

```
data/data/us_equity/1d/stock_alpaca.zarr   symbol dtype = float64
data/data/us_equity/1d/us_all.zarr         symbol dtype = <U9
data/data/us_equity/1m/stock_alpaca.zarr   symbol dtype = StringDType()
```

So the fixtures cover one dtype while the stores on this machine carry three. Note
also that the narrative in `260908-0f4`'s summary says "every real store in this
project" would have failed; the accurate claim is **every store the current chunked
ingest writes** — `us_all.zarr` at `<U9` would have succeeded.

**A separate oddity, noticed and not chased:** `1d/stock_alpaca.zarr` records `symbol`
as `float64`. A symbol coordinate should never be floating point. Most likely an empty
store whose coordinate was never populated, but it was not investigated and may be its
own defect.

## Solution

TBD. The property to reach is that a test exercising a widening method sees the
coordinate dtype a production panel actually carries — and that the lock lives with
the method, not incidentally in a downstream suite.

Two things a plan should decide rather than assume:

- **Whether one realistic fixture per suite is enough, or whether the dtype should be
  parametrised.** Three dtypes are live on this machine today. A fixture pinned to one
  of them re-creates the same blind spot one dtype over.
- **Whether the encoding round-trip belongs in a shared fixture helper.** The
  round-trip is what makes it realistic; a helper that builds a store by writing and
  re-opening it would make realism the default rather than something each test
  remembers.

Do NOT close this by adding an assertion to the chunked-ingest tests. Those already
catch it, by accident; the gap is that the owning suites do not.

## Files

- `tests/test_symbol_axis_widening.py`, `tests/test_variable_axis_widening.py`,
  `tests/test_factor_update.py` — the three suites with the unrealistic fixtures
- `quantlab/dataset/backend.py` — `widen_symbol_axis`, `widen_data_vars`,
  `widen_and_append` (the methods those suites own)
- `tests/test_chunked_ingest.py` — where the incidental lock currently sits

## Discovered

2026-09-08, in `260908-0f4`, when a task whose fixtures went through the real ingest
path hit a defect that three dedicated widening suites and a 13/13 verification had
both missed.
