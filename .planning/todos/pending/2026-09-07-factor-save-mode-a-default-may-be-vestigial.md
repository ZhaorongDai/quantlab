---
created: 2026-09-07T00:00:00.000Z
title: Factor.save()'s mode="a" default may be vestigial now that update() exists
area: factor / storage
severity: minor
triggers: any caller writing a second date range through `Factor.save()` without passing
  an explicit mode. Harmless while every caller already passes `mode="w"` or writes only
  once; fires the moment someone relies on the default meaning "append".
files:
  - quantlab/base/factor.py (Factor.save — the `mode: Literal["a", "w"] = "a"` default)
  - quantlab/base/factor.py (Factor.update — the incremental interface added 260907-vyr)
  - tests/test_factor_save_mode.py (test_the_default_is_still_a — pins the decision, not just the code)
---

## Problem

`Factor.save()` defaults to `mode="a"`, which READS like "append" and is not.

zarr's `"a"` means "overwrite variables in an existing store", not "append along
time". Writing a second, differently-sized date range under it fails outright --
`to_zarr` refuses to change a dimension size without an explicit `append_dim`.
Quick task 260907-fl6 made that failure actionable (it now names `mode="w"`, names
the store, and keeps zarr's own words on `__cause__`) but deliberately left the
default alone, because changing a default is a behaviour change for every existing
caller.

So `"a"` is the mode that essentially never does what a second write wants:

- writing once -- `"a"` and `"w"` are indistinguishable, the store is created either way
- rewriting wholesale -- the caller wants `"w"`, and `"a"` raises
- extending by time -- the caller wants `update()`, and `"a"` raises

There is no call shape left where the default is the right answer.

## Why it is being surfaced now rather than then

Quick task 260907-vyr added `Factor.update()`: the explicit, automatic incremental
interface. Before it existed, `"a"` at least gestured at an intent the factor layer
could not otherwise express -- there was no route from `Factor` to
`XrBackend.append` at all (measured 2026-09-07: `widen_and_append` had no production
caller anywhere in the repo). Now the split is explicit and named:

    save()    writes WHOLESALE   (save(mode="w") replaces the store)
    update()  EXTENDS           (no mode, no route to overwrite a stored range)

With that split in place, `save()`'s `"a"` default has no remaining job, and the
argument for keeping it is now purely backwards compatibility rather than meaning.

## The decision, which is the developer's and was NOT made here

Options, roughly in increasing order of disruption:

1. **Leave it.** Zero risk, but the default keeps reading as "append" to everyone
   who has not read the docstring.
2. **Flip the default to `"w"`.** Matches what "save the factor panel" almost always
   means. Behaviour change: a caller who today gets a loud `ValueError` on a second
   range would silently get a REPLACED store instead. That is arguably worse -- a
   raise is louder than a truncation.
3. **Make `mode` required (no default).** Forces every call site to say which it
   means. Loudest, most disruptive, and the only one that cannot silently change
   what an existing caller's data looks like.

Option 2 is the tempting one and the one worth the most scrutiny, precisely because
it converts a raise into a silent overwrite.

`tests/test_factor_save_mode.py::test_the_default_is_still_a` pins the CURRENT
decision, so whichever way this goes, that test is the thing to change deliberately
rather than discover.

## Status

Surfaced, not decided. 260907-vyr changed nothing about `save()` except its prose --
the two clauses claiming the method had no incremental route became false when
`update()` landed and were re-pointed at it.
