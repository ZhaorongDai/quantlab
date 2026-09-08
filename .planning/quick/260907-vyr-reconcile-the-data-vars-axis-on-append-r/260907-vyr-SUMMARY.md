---
phase: quick-260907-vyr
plan: 01
subsystem: data / storage + factor
status: complete
tags: [xarray, zarr, append-guard, data-integrity, widening, tdd, mutation-testing]
requires:
  - quantlab/dataset/backend.py::XrBackend._assert_append_compatible (the existing single call site the guard rides)
  - quick task 260907-uac (the append_dim overlap refusal update() inherits)
  - quick task 260906-x2s (widen_symbol_axis, whose superset-and-backfill rule this mirrors)
provides:
  - "XrBackend.append refuses ANY data_vars set mismatch, both directions, before to_zarr"
  - "XrBackend.widen_data_vars — materialise a new data variable over the store's existing extent"
  - "widen_and_append reconciles THREE axes (symbol, then variables, then the unchanged append)"
  - "Factor.update() — the automatic incremental write interface, with no overwrite route"
  - "Factor._widen_fill_values() — the per-subclass fill seam for a non-float variable"
affects:
  - Factor.save (prose only — behaviour, signature and mode="a" default untouched)
  - widen_and_append's five existing test callers (all single-variable, so the variable widen is a no-op on each)
tech-stack:
  added: []
  patterns:
    - "The data_vars set is an AXIS, reconciled like the symbol axis rather than assumed"
    - "Refuse rather than write something wrong quietly — the fourth guard built on this principle in quantlab/dataset/backend.py"
    - "Superset rule + explicit named opt-in, never a flag that loosens the refusal"
    - "Lock a behavioural property with a PAIR of separate test functions when a structural proxy provably cannot span it"
    - "The storage layer stays variable-neutral: 'factor' is the caller's word, never the store's"
key-files:
  created:
    - tests/test_variable_axis_widening.py
    - tests/test_factor_update.py
    - .planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md
  modified:
    - quantlab/dataset/backend.py
    - quantlab/base/factor.py
decisions:
  - "D-01/D-02 implemented: both directions refused unconditionally, MISSING checked first and naming no opt-in. Locked by M2 (drop the missing branch -> tests 2,4,5 red) and M3 (swap precedence -> exactly test 5 red on its message)."
  - "D-03 implemented: the check sits AFTER the shared-variable dtype loop. Spanned by the ONLY fixture that can span it — a combined dtype+variable-set mismatch — and proved by M4, which flips exactly that one test's message."
  - "D-06 implemented: the filler carries the INCOMING dtype. M6 (hardcode float64) is caught by the EXISTING shared-variable dtype guard at the closing append(), confirming the composition silently depended on this."
  - "D-07 implemented: the filler is written with _append_encoding. M7 (drop encoding=) reads back chunks (10,2) against the store's (4,2) — the measured divergence, which does NOT crash the next append and so had to be pinned by construction."
  - "D-08 implemented: a non-float new variable is refused without an explicit fill. M8 recorded the store it would otherwise produce — anomaly_flag all True, volume all 0. Fabricated history, exactly as measured at planning time."
  - "D-10 honoured: save()'s mode=\"a\" default is untouched and the question of whether it is now vestigial is FILED for the developer, not decided."
  - "D-11 honoured: no warm-up guard or trim anywhere. The only config.window read in factor.py remains the pre-existing _reset_dataset_config; nothing added reads it."
metrics:
  duration_minutes: 42
  completed: 2026-09-08
  tasks: 3
  tests_added: 23
  suite_before: 545
  suite_after: 568
actuals:
  tokens: 19091
  tasks: 3
  commits: 3
  plan_head_before: e549928ff0e1f9e79315ea3ec7be365e45e28ed0
---

# Quick Task 260907-vyr: Reconcile the data_vars Axis on Append Summary

The `data_vars` set is now a reconciled axis on the append path: any mismatch is
refused before `to_zarr`, a panel that legitimately grew a column has an explicit
`widen_data_vars()` opt-in, and `Factor.update()` gives the factor layer an
automatic incremental interface built on the three-axis reconcile-then-append path.

## What Shipped

Three axes cross `XrBackend.append`. Two already had a rule and a guard; the third
had neither, and was the most destructive of the three — measured on the unguarded
tree, all three mismatch shapes wrote silently and left the store **unopenable**,
and one of them destroyed a store that was valid before the call.

| Task | Commit | What |
|------|--------|------|
| 1 | `868001b` | The refusal: `_assert_append_compatible` checks the variable SET, both directions, after the dtype loop |
| 2 | `dea1e85` | `widen_data_vars()` + `widen_and_append` extended to three axes |
| 3 | `41fcf99` | `Factor.update()`, `Factor._widen_fill_values()`, `save()` prose re-pointed, todo filed |

**Suite: 545 -> 568**, gated per task at 553 / 561 / 568. Every gate was run and
every number below is the real one printed by pytest.

## Verification Results

All eight verification items pass:

1. `uv run pytest tests/ -q` -> **568 passed**, 0 failed (gates hit 553 / 561 / 568).
2. Every mutation applied, observed and reverted; `git diff --quiet` clean on both
   mutated files afterwards.
3. Every RED observed before its implementation, and behavioural (see below).
4. Storage-layer factor vocabulary: **2**, unchanged from its measured baseline —
   both hits are the pre-existing `FactorPolars` narrative in `head()`'s docstring.
   `widen_data_vars` adds none.
5. `save()`'s signature line present verbatim at `factor.py:131`;
   `tests/test_factor_save_mode.py`'s 5 tests green untouched.
6. No warm-up trim: the only `config.window` read is the pre-existing
   `_reset_dataset_config` at `factor.py:98`. Nothing added reads it. (Per the
   orchestrator's advisory, item 6 is read as scoped to NEWLY ADDED reads — the
   `_auto_filter()` call that `update()` shares with `save()` is pre-existing and
   required for the two write paths to normalise identically.)
7. Backticked pre-migration path counts: `backend.py` 1, `factor.py` 1 — both at
   baseline.
8. `git status --porcelain` shows only the pre-existing ` M test.py`, which was
   never staged, committed, reverted or touched. Confirmed: it appears in zero of
   the three commits.

## RED Evidence (observed before each implementation)

| Task | RED | Failure mode |
|------|-----|--------------|
| 1 | 6 failed, 2 passed | 5x `Failed: DID NOT RAISE ValueError`; test 8 `TypeError: Dataset.to_zarr() got an unexpected keyword argument 'force'` |
| 2 | 7 failed, 9 passed | Composed-path tests failed with the Task 1 refusal (`new=['beta']`) — behavioural, through the exact mechanism Task 2 removes |
| 3 | 6 failed, 1 passed | `AttributeError: 'PanelFactor' object has no attribute 'update'` — the interface genuinely absent |

Tests that were GREEN before their task and stayed green are precedence/no-change
locks, not misses: Task 1 test 6 (positive control) and test 7 (D-03 precedence),
Task 2 test 8 (overlap inheritance, measured to hold either way), Task 3 test 7
(`save()` unchanged).

## Mutation Verification

All eleven mutations applied and reverted. **Eight matched their predictions
exactly. Three diverged, all in the same direction — MORE tests detected the
mutation than predicted — and none implicates the implementation.**

| Mut | Predicted | Observed | Verdict |
|-----|-----------|----------|---------|
| M1 | T1 tests 1,2,3,4,5,8 red; 6,7 green | exactly that | match |
| M2 | exactly T1 tests 2,4,5 red | exactly that; test 3 green (the disjoint case caught by the NEW branch) | match |
| M3 | exactly T1 test 5, on its message | exactly that — received the widening message where the unconditional one is required | match |
| M4 | exactly T1 test 7, on its message | exactly that — received the MISSING message where the dtype message is required | match |
| M5 | SPLIT: new test 8 + chunked_ingest `:633` red, `:603` GREEN | exactly that, all three by name | match |
| M6 | exactly T2 test 3 | T2 test 3 **and T2 test 6** red | **divergence** |
| M7 | exactly T2 test 4, chunks = store extent | exactly that: `(10, 2)` vs `(4, 2)` | match |
| M8 | T2 test 5 red, test 6 green; record the store | exactly that; store held `anomaly_flag` all `True`, `volume` all `0` | match |
| M9 | exactly T2 test 3 + T3 test 3 | those two **and T3 test 6** red | **divergence** |
| M10 | SPLIT: T3 test 4 red, T3 test 5 GREEN | matches under "declared-but-unpassed"; the "forwarded" reading also reddens test 5, for an unrelated reason | **ambiguity** |
| M11 | T3 tests 2,3 red; test 1 GREEN | exactly that, **plus T3 test 6** red | **divergence** |

### The three divergences, reported not smoothed over

**M6, M9 and M11 share one root cause.** Each predicted an "EXACTLY test N" set,
but the plan's own spec for a *different* test creates a second, independent
detector of the same mutation:

- **M6** (hardcode the filler to float64) also reddens Task 2 test 6, whose
  plan-specified assertion is literally "preserves the dtype exactly on disk — no
  float64 upcast". Observed: `assert dtype('<f8') == dtype('bool')`. A float64
  filler *is* the upcast that test exists to catch, so its reddening is necessary,
  not incidental.
- **M9** (leave the short-circuit symbol-only) also reddens Task 3 test 6. That
  test's plan-specified span — "a subclass carrying a non-float variable and
  overriding the seam must widen successfully where one that does not override is
  refused" — necessarily has an agreeing symbol axis and a growing variable set,
  which is precisely the M9-observable shape. Observed: it receives the Task 1
  new-variable refusal instead of the `fill_values` one, because `widen_data_vars`
  is never reached.
- **M11** (route `update()` through `append()`) also reddens Task 3 test 6, for the
  same reason — removing the widening path removes the seam it exercises. M11's
  prediction did not say "EXACTLY", and its load-bearing half held: **test 1 stayed
  GREEN**, which is what isolates the reconciliation from the plain append.

None of these is a test passing for the wrong reason, and none is a defect in the
shipped code — in every case the extra red test detected the mutation through its
own stated assertion. The plan's predictions were under-counted relative to the
plan's own test specifications.

**M10 is an ambiguity in the mutation's wording rather than a divergence.** The
mutation is described as "give `Factor.update` a `mode` parameter *forwarded to the
backend*", but its prediction reasons about "a **declared-but-unpassed** `mode`".
Both were run:

- *declared but unpassed* (what the prediction reasons about): the predicted split
  holds exactly — test 4 FAILED, test 5 PASSED.
- *forwarded*: test 5 also reddens, but with
  `TypeError: to_zarr() got multiple values for keyword argument 'mode'` at the
  store-**creation** step. It fails at setup, not on its refusal assertion, and
  demonstrates nothing about a route past the guard.

The split M10 exists to demonstrate — structural proxy red, behavioural green — is
confirmed under the reading its own prediction states.

## Deviations from Plan

**None.** The plan was executed as written, including both orchestrator advisories:

1. **Task 3 test 3's symbol axis pinned.** Written with the same roster on both
   sides (`["AAA","BBB"]` before and after) and only the variable set growing,
   mirroring Task 2 test 3. This is what made it an M9 canary — confirmed: it
   reddened under M9 with `carries data variable(s) ['beta']`.
2. **Verification item 6 read by intent.** `update()` calls `_auto_filter()` exactly
   as `save()` does, as Task 3 requires. No finding filed about the pre-existing
   `config.window` read it shares.

No auth gates. No architectural (Rule 4) decisions. No auto-fixes were needed —
every gate passed on its first run after implementation.

## Notes for the Developer

- **`widen_and_append` now has its first production caller.** Measured at planning
  time it had none outside `tests/`. `Factor.update()` is it.
- **One open question is filed, not answered:**
  `.planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md`.
  Now that `update()` exists, there is no call shape left where `save()`'s `"a"`
  default is the right answer — but flipping it would convert a loud `ValueError`
  into a silent store replacement, which is arguably worse. That trade-off is
  yours; `tests/test_factor_save_mode.py::test_the_default_is_still_a` pins the
  current decision so it changes deliberately rather than by accident.
- **The storage layer stayed variable-neutral.** `widen_data_vars` names the xarray
  concept, not the caller's domain, so a market-data panel that grows a column later
  uses the same method with no rename.

## Known Stubs

None. No stubs, TODOs, FIXMEs, skipped tests or unrun `<verify>` blocks were
introduced. (The two `raise NotImplementedError` hits in `factor.py` are the
pre-existing `_get_features`/`_get_labels` hooks, line-shifted by this change.)

## Threat Flags

None. No new network endpoints, auth paths, file-access patterns or trust-boundary
schema changes beyond the `<threat_model>`'s existing register. Every `mitigate`
disposition in that register (T-vyr-01 through T-vyr-06) is implemented and
mutation-verified; T-vyr-07 (the half-reconciled store after an inherited refusal)
remains `accept` and is asserted as the documented side effect by Task 2 test 8.

## Self-Check: PASSED

- `quantlab/dataset/backend.py` — FOUND (modified)
- `quantlab/base/factor.py` — FOUND (modified)
- `tests/test_variable_axis_widening.py` — FOUND (created, 16 tests)
- `tests/test_factor_update.py` — FOUND (created, 7 tests)
- `.planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md` — FOUND (created)
- Commit `868001b` — FOUND
- Commit `dea1e85` — FOUND
- Commit `41fcf99` — FOUND
- `commits: 3` measured via `git rev-list --count e549928..HEAD`, not narrated
