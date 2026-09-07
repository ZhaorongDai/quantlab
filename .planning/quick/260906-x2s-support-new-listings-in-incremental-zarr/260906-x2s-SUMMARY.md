---
phase: quick-260906-x2s
plan: 01
subsystem: data
status: complete
tags: [zarr, incremental-append, symbol-axis, new-listings, crash-safety, cli]
requires:
  - XrBackend.append / _assert_append_compatible (260906-13w)
  - ChunkLedger + TimeChunkPlanner (260906-13w)
  - BaseDataset.from_raw_data_chunked (260906-13w)
  - BaseDataset._pin_append_dtypes
provides:
  - XrBackend.widen_symbol_axis
  - XrBackend.widen_and_append
  - ChunkLedger.rebase
  - BaseDataset.NEW_LISTING_STRATEGIES
  - BaseDataset.from_raw_data_chunked(on_new_listing=...)
  - BaseDataset._widen_fill_values (overridable seam)
  - ingest_us_equity.py --on-new-listing
affects:
  - dataset/backend.py
  - base/data.py
  - base/chunking.py
  - utils/cli.py
  - ingest_us_equity.py
tech-stack:
  added: []
  patterns:
    - "rename-aside -> rename-in -> rmtree directory swap (ChunkLedger._flush's os.replace idiom, one level up at directory granularity)"
    - "guard-satisfying opt-in: the widen ends by calling the UNCHANGED append(), so _assert_append_compatible still runs and passes by construction"
    - "choices DERIVED from a locked class literal, never restated in the CLI module"
key-files:
  created:
    - tests/test_symbol_axis_widening.py
  modified:
    - dataset/backend.py
    - base/data.py
    - base/chunking.py
    - utils/cli.py
    - ingest_us_equity.py
    - tests/test_chunked_ingest.py
    - tests/test_ingest_tiingo_universe_wiring.py
decisions:
  - "_assert_append_compatible was NOT relaxed. Both new paths are separately-named explicit opt-ins that SATISFY the guard by making the two axes agree, rather than loosening it. Verified by diff (only _append_encoding's signature changed) and by the new plain-append regression test."
  - "widen_and_append ends by calling append(), deliberately. Writing the window directly with to_zarr(mode='a') from inside the widen path would remove the guard from the widened path entirely and reintroduce the measured mis-attribution."
  - "The union is sorted(stored | incoming), matching base/constituent.py:_densify's all-time-union rule, because ChunkLedger's fingerprint is order-sensitive and the axis must be reproducible across runs."
  - "A widen refuses to NaN-backfill a non-float variable without an explicit fill_values entry. Measured: an unfilled reindex upcasts bool anomaly_flag and int64 volume to float64-with-NaN -- a silent schema change to a live store."
  - "BaseDataset._widen_fill_values() is an overridable seam returning {'anomaly_flag': False}, not a constant. _pin_append_dtypes leaves that one variable bool, so without it EVERY real widen of a cleaned market panel refuses. A future non-OHLCV Dataset subclass carries different variables."
  - "rebuild renames the store and ledger ASIDE and restores them on any exception. A rebuild that deletes first has no way back from a failure halfway through a multi-hour re-densify."
  - "A crash between the widen's two renames leaves NO store at path and REFUSES on the next run, naming the manual `mv`. Auto-recovering is deliberately not done: which directory is authoritative is not the backend's call."
  - "on_new_listing defaults to 'refuse' -- byte-identical to the pre-task behaviour. A destructive strategy is only ever reached by an explicit argument or flag."
metrics:
  duration: 14 min
  completed: 2026-09-07
actuals:
  tokens: 46609
  tasks: 3
  commits: 3
---

# Quick Task 260906-x2s: Support New Listings in Incremental Zarr Appends Summary

A store whose roster has grown since it was created now has two explicit,
independently tested forward paths — a crash-safe in-place symbol-axis widen at
the backend layer and a detect-and-rebuild strategy switch at the `Dataset`
layer — and neither relaxes the `_assert_append_compatible` guard that made the
halt correct in the first place.

## What Was Built

**Task 1 — `XrBackend.widen_symbol_axis` / `widen_and_append`** (`874cdb8`)

`widen_symbol_axis(path, symbols, dim, append_dim, fill_values)` rewrites an
existing store onto a superset symbol axis. Three guards fire before any write:
a crashed-rename detector (a `.superseded.tmp` sidecar with no store at `path`),
a superset check (a non-superset target silently deletes a delisted symbol's
whole history), and a fill-safety check (a non-float variable that would receive
a NaN backfill). The rewrite goes to a `.widening.tmp` sibling and is swapped in
with `os.replace(path, superseded)` → `os.replace(widening, path)` →
`shutil.rmtree(superseded)`.

`widen_and_append(path, ...)` is the opt-in sibling of `append()`. It widens the
store to `sorted(stored ∪ incoming)`, reindexes `self.data` onto that same axis,
and then calls the **unchanged** `append()`. That closing call is load-bearing:
both sides now share one axis, so `_assert_append_compatible` still runs and
passes on its own terms.

`_append_encoding` gained an optional `data` panel so the widen rewrite reuses
the `APPEND_DIM_CHUNK` rule instead of restating chunk arithmetic.

**Task 2 — `Dataset`-layer reconciliation** (`fa8e1d8`)

`BaseDataset.NEW_LISTING_STRATEGIES = ("refuse", "rebuild", "widen")` and
`from_raw_data_chunked(on_new_listing="refuse")`. The reconciliation compares the
pinned whole-range axis against the **store's**, reading only the coordinate the
way `ChunkLedger._store_tail` does, and branches:

- `refuse` (default) — falls through unchanged; the `ChunkLedger` roster error
  raises exactly as before, now preceded by an INFO line naming the remedy.
- `rebuild` — renames the store and ledger aside, constructs a fresh ledger, runs
  the loop as a first run, and restores both on any exception.
- `widen` — calls `widen_symbol_axis` with `_widen_fill_values()` and then
  `ChunkLedger.rebase(symbols)` in the same operation, then falls through to the
  normal loop.

`ChunkLedger.rebase` re-fingerprints the axis and deliberately leaves `windows`
untouched — a widen changes the axis, not the windows.

**Task 3 — `--on-new-listing`** (`ea46887`)

Added to `add_chunk_args` with `choices=list(BaseDataset.NEW_LISTING_STRATEGIES)`
and default `refuse`, threaded into `ingest_us_equity.py`'s existing
`from_raw_data_chunked` call.

## Verification Discipline: Observed RED Output

This project has ten recorded instances of a test passing for the wrong reason,
so every headline claim below was run against the mutation that should break it
and **observed failing**. Actual output, not paraphrase.

### Headline 1 — `test_widening_does_not_misattribute_when_the_symbol_count_is_unchanged`

Run **before any implementation existed**, against a mutant `widen_and_append`
that substituted raw `to_zarr(mode="a", append_dim=...)` for the widen:

```
        after = _stored(path)
>       assert after["symbol"].values.tolist() == ["A", "ARM", "XYZ"]
E       AssertionError: assert ['A', 'ARM'] == ['A', 'ARM', 'XYZ']
E
E         Right contains one more item: 'XYZ'
```

The same mutant, driven directly to show the corruption at value level:

```
BEFORE axis: ['A', 'XYZ'] XYZ history: [1.0, 3.0]
AFTER  axis: ['A', 'ARM'] ARM history: [1.0, 3.0, 901.0]
--> rows [1.0, 3.0] were written for XYZ but are now labelled: ARM
```

That reproduces the 2026-09-06 measurement exactly. XYZ is gone from the axis and
its two historical rows are now attributed to a symbol that did not exist when
they were written.

### Headline 2 — `test_rebuild_redensifies_every_window_onto_the_new_union`

Run before the Task 2 fix was committed, against the mutation `rebuild` → the
`widen` branch:

```
        assert rebuilt["symbol"].values.tolist() == ["A", "B", "C"]
>       assert (rebuilt["adjClose"].sel(symbol="C").values == 300.0).all()
E       AssertionError: assert np.False_
E        +    where ... = array([nan, nan, nan, nan, nan, nan, nan, nan, nan]) == 300.0.all
```

This is also the proof that `rebuild` and `widen` are **observably different** on
the same input: `rebuild` recovers C's real 300.0 history from raw across all
nine timestamps, `widen` leaves all nine NaN.

### Additional mutations run and observed RED

| Mutation | Test | Observed |
|---|---|---|
| `_assert_append_compatible` relaxed to `len(incoming) != len(stored)` only | `test_plain_append_still_refuses_a_labels_differ_axis_of_the_same_length` | `E Failed: DID NOT RAISE ValueError` |
| `encoding=` dropped from the widen rewrite | `test_the_chunk_grid_survives_a_widen` | `E assert (512, 2) == (512, 3)` — the rewrite inherits the SOURCE store's encoding, so the symbol chunk stays pinned to the pre-widen count |
| `on_new_listing` default flipped to `"widen"` | `test_the_default_still_refuses_a_roster_change` | `E Failed: DID NOT RAISE ValueError` |
| `ledger.rebase(symbols)` omitted from the widen branch | `test_a_widen_rebases_the_ledger_so_the_next_run_resumes` | `E ValueError: ChunkLedger: refusing to resume ... the pinned symbol axis has 3 symbol(s) but the ledger ... was written against 2.` |
| CLI choices restated + a 4th strategy added at the Dataset layer | `test_the_new_listing_choices_are_derived_from_the_locked_literal` | `E AssertionError: assert ['refuse','rebuild','widen'] == ['refuse','rebuild','widen','merge']` |

The `test_the_chunk_grid_survives_a_widen` docstring was corrected after this run:
it originally predicted the append-dim chunk would become 600. The measured
failure is at the **symbol** index instead, because a `to_zarr` without
`encoding=` inherits the source store's encoding. The docstring now records what
actually happens.

## Plan Verification Results

| # | Criterion | Result |
|---|---|---|
| 1 | `uv run pytest -q` zero failures, ≥ 387 + new | **405 passed**, 0 failed (baseline 387, +18 new) |
| 2 | `git diff dataset/backend.py` shows `append`'s body and `_assert_append_compatible`'s comparison logic unchanged | Confirmed — the only deleted lines across the whole task are `-from typing import Optional, Self`, `-    def _append_encoding(self, append_dim: str) -> dict:`, and the two `self.data` references inside `_append_encoding` that became `panel` |
| 3 | `--help` lists `--on-new-listing` with three derived choices, default `refuse` | Confirmed: `--on-new-listing {refuse,rebuild,widen}` |
| 4 | Every new test's docstring names its reddening mutation; both headline tests observed RED against it | Confirmed — see above |

Test-count breakdown: 387 baseline → 405. +10 in `tests/test_symbol_axis_widening.py`,
+7 in `tests/test_chunked_ingest.py`, +1 in `tests/test_ingest_tiingo_universe_wiring.py`.

## Deviations from Plan

### Additions beyond the plan's enumerated tests (Rule 2 — missing critical coverage)

**1. [Rule 2] `test_a_crash_between_the_two_renames_refuses_and_names_the_recovery`**
- **Found during:** Task 1
- **Issue:** The plan's step-1 recovery-state check was specified in `<action>`
  but had no test in `<behavior>`. It is the one crash state that can leave the
  only copy of a real store under a sidecar name, and an untested refusal is a
  refusal a later refactor deletes.
- **Fix:** Added a test that renames a real store to `.superseded.tmp` and
  asserts the widen raises naming both paths, with the real store still present.
- **Files:** `tests/test_symbol_axis_widening.py`
- **Commit:** `874cdb8`

**2. [Rule 2] `test_widen_and_append_creates_the_store_when_there_is_none`**
- **Found during:** Task 1
- **Issue:** The plan specifies "one creation path, not two" in `<action>` with
  no covering test. A second creating write would bypass `_append_encoding` and
  pin the chunk grid differently from every store `append()` creates.
- **Fix:** Added a test asserting the delegation and the resulting chunk grid.
- **Files:** `tests/test_symbol_axis_widening.py`
- **Commit:** `874cdb8`

### Process deviations

**3. Corrected docstring in `test_the_chunk_grid_survives_a_widen`** — see the
mutation table above. The named mutation still reddens the test; only the stated
mechanism was wrong and is now recorded from measurement.

**4. Mid-task `git checkout utils/cli.py` accident** — during Task 3's mutation
testing I ran `git checkout utils/cli.py`, which reverted that task's
(uncommitted) edits. Detected immediately from the mutation script's own
assertion, restored `base/data.py` and `utils/cli.py` from the committed state,
and re-applied the Task 3 edits deterministically via a script. `git diff` after
re-application was verified against the intended content and the full suite was
re-run. No committed work was lost and no earlier task was touched.

### Everything else executed as written

`_assert_append_compatible` was not relaxed. `append()`'s signature and default
path are byte-compatible. `from_raw_data_chunked(on_new_listing="refuse")` is
byte-identical to the previous behaviour, proven by the pre-existing
roster-change, resume, crash-resume and tail-mismatch tests passing untouched.

## Threat Mitigations Applied

| Threat ID | Disposition | Where mitigated |
|---|---|---|
| T-x2s-01 | mitigate | `widen_and_append` ends by calling the unrelaxed `append()`; pinned by `test_plain_append_still_refuses_a_labels_differ_axis_of_the_same_length` + `test_widening_does_not_misattribute_when_the_symbol_count_is_unchanged` |
| T-x2s-02 | mitigate | Fill-safety guard fires before any write; `test_widening_refuses_a_non_float_variable_without_an_explicit_fill_value` asserts the store is byte-identical afterwards |
| T-x2s-03 | mitigate | `.widening.tmp` → rename-aside → rename-in → rmtree; `test_a_failed_widen_leaves_the_original_store_intact`, `test_a_crash_between_the_two_renames_refuses_and_names_the_recovery` |
| T-x2s-04 | mitigate | `rebase` documented as valid only after a successful widen; its only caller is the `widen` branch immediately after `widen_symbol_axis` returns |
| T-x2s-05 | mitigate | Store and ledger renamed aside, restored on any exception; `test_a_failed_rebuild_restores_the_original_store` |
| T-x2s-06 | accept | Documented in `widen_symbol_axis`'s docstring and in the `widen` branch's WARNING log; `rebuild` named as the strategy for a store too large to hold |
| T-x2s-07 | mitigate | Default `refuse` at both the `Dataset` and CLI layers; `test_the_default_still_refuses_a_roster_change` and the derived-choices test's default assertion |
| T-x2s-SC | accept | No package installs; no new dependency (dask specifically not added) |

## Known Stubs

None. No `TODO`/`FIXME`/placeholder was introduced, no test was skipped, and every
`<verify>` block in the plan was run.

## Commits

| Task | Commit | Message |
|---|---|---|
| 1 | `874cdb8` | feat(quick-260906-x2s): crash-safe symbol-axis widen on XrBackend |
| 2 | `fa8e1d8` | feat(quick-260906-x2s): reconcile a grown roster at the Dataset layer |
| 3 | `ea46887` | feat(quick-260906-x2s): expose --on-new-listing on the ingest CLI |

## Known Costs and Follow-ups

- **A widen materialises the whole store in RAM.** dask is not installed, so
  `xr.open_zarr` yields lazily-indexed arrays that `.load()` reads in full. This
  is the exact allocation `from_raw_data_chunked` exists to avoid. Documented in
  the docstring and in the runtime WARNING; `rebuild` is the strategy for a store
  too large to hold. Adding dask was explicitly out of scope.
- **A `widen` on a store predating `_pin_append_dtypes` will refuse** on an int64
  `volume`, naming `_pin_append_dtypes` as where the promotion belongs. That is
  the intended behaviour, not a gap — the alternative is a silent schema change.
- The pre-existing unstaged deletion of `backtest/test_strategy.py` and the
  untracked `.gsd/` and `.planning/state.json` were left exactly as found; every
  commit staged files explicitly by path.

## Self-Check: PASSED

- `dataset/backend.py`, `base/data.py`, `base/chunking.py`, `utils/cli.py`,
  `ingest_us_equity.py`, `tests/test_symbol_axis_widening.py`,
  `tests/test_chunked_ingest.py`, `tests/test_ingest_tiingo_universe_wiring.py`
  — all present on disk.
- Commits `874cdb8`, `fa8e1d8`, `ea46887` — all found in `git log`.
- `uv run pytest -q` at HEAD: 405 passed, 0 failed.
