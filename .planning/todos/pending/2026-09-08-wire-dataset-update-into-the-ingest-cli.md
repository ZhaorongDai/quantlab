---
created: 2026-09-08T00:00:00.000Z
title: BaseDataset.update() ships with no CLI caller -- decide whether the ingest script should reach it
area: dataset / cli
severity: minor
triggers: an operator running a periodic refresh from `ingest_us_equity.py` who wants the
  evidence-driven strategy rather than picking `--on-new-listing` by hand. Today they must
  either choose a strategy themselves or call `update()` from Python.
files:
  - quantlab/base/data.py (BaseDataset.update -- the automatic incremental entry point)
  - quantlab/base/data.py (BaseDataset._resolve_new_listing_strategy -- the three-way rule)
  - ingest_us_equity.py (the `--on-new-listing` flag, threaded to from_raw_data_chunked)
  - quantlab/utils/cli.py (add_chunk_args -- where the flag's choices are derived)
---

## Problem

`BaseDataset.update()` landed in quick task 260908-0f4 with **no CLI caller**. The
only route to it is Python. Meanwhile `ingest_us_equity.py` still takes
`--on-new-listing`, whose whole point `update()` removes: it asks the operator for a
choice that is actually a fact about the raw tier.

So the ingest script -- the place an operator would naturally reach for a periodic
refresh -- is the one place that cannot ask for the automatic behaviour.

## Why it was not decided during 260908-0f4

Both directions have a real argument and neither is obviously right, so the task
filed the question rather than answering it by default.

**Against adding a flag (the symmetry argument).** `Factor.update()` -- the sibling
this method was built to mirror, landed one day earlier in 260907-vyr -- has no CLI
either. Adding one here breaks that symmetry, and the split it would have to express
(`--update` versus `--on-new-listing`) is a mode selector, not a value, which is a
different shape from every flag the ingest script currently carries.

**For adding one (the reachability argument).** The ingest script IS the periodic
refresh. An operator who has to drop into Python to get the safe behaviour will, in
practice, keep passing `--on-new-listing widen` because it is the one that is there
and it is fast -- which is precisely the silent data loss the task exists to prevent.

## What is NOT blocked on this

The operator-facing report already reaches the terminal. The resolver logs its
decision through loguru from inside the library -- naming how many added symbols
qualified, and each one's raw row count inside the store's extent, before any rebuild
runs. So nothing about visibility waits on a CLI decision; only convenience does.

## Options

1. **Leave it.** Symmetric with `Factor.update()`. The Python route works.
2. **Add `--update` to `ingest_us_equity.py`**, mutually exclusive with
   `--on-new-listing`. Most discoverable; needs the mutual exclusion to be explicit,
   or the two flags will silently contradict each other.
3. **Make `--on-new-listing` accept an `auto` token that routes to `update()`.**
   Tempting and probably wrong: quick task 260908-0f4 deliberately made the automatic
   sentinel a NON-string `object()` precisely so no user-typed string could reach that
   branch, and this option would undo that by construction. If it is chosen, it should
   be chosen knowing that is what it does.

## Status

Surfaced, not decided.
