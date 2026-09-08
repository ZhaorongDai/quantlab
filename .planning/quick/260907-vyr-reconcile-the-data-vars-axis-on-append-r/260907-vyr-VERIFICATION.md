---
phase: quick-260907-vyr
verified: 2026-09-08T01:20:00Z
status: passed
score: 13/13 must-haves verified
covered_files:
  - ".planning/quick/260907-vyr-reconcile-the-data-vars-axis-on-append-r/260907-vyr-PLAN.md"
  - ".planning/quick/260907-vyr-reconcile-the-data-vars-axis-on-append-r/260907-vyr-SUMMARY.md"
  - ".planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md"
  - "quantlab/base/factor.py"
  - "quantlab/dataset/backend.py"
  - "tests/test_factor_update.py"
  - "tests/test_variable_axis_widening.py"
covered_digest: "v1:sha256:49d4f2d27d0d37c39607d109c0c908ac84ba50a56f6030343f53378dc09dbb88"
behavior_unverified: 0
overrides_applied: 0
---

# Quick Task 260907-vyr: Reconcile the data_vars Axis on Append — Verification Report

**Task Goal:** make the `data_vars` set a reconciled axis on the append path — a
refusal in `_assert_append_compatible`, a variable-neutral `widen_data_vars` folded
into `widen_and_append`, and `Factor.update()` as the automatic three-axis interface
with no overwrite route.

**Verified:** 2026-09-08
**Status:** passed
**Re-verification:** No — initial verification

Everything below was re-derived by the verifier: the suite was run at each of the
three task commits, the original defect was reproduced on a pre-implementation
worktree, all eleven mutations were re-applied and reverted in a throwaway worktree,
and the three-axis composition was measured end to end with an independent probe.
No SUMMARY claim was accepted on its word.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| DVAR-01 | `append()` raises before `to_zarr` on an unstored incoming variable | ✓ VERIFIED | Independent probe: store `{alpha}` + incoming `{alpha,beta}` → `ValueError` from `_assert_append_compatible`; store reopens as `['alpha']` `{timestamp:2, symbol:2}`, unchanged. On a worktree at `e549928` (pre-implementation) the identical probe → **NO RAISE** and `xr.open_zarr` → `conflicting sizes for dimension 'timestamp'`. The defect was real and is gone |
| DVAR-02 | The destructive (missing-variable) direction is refused unconditionally, no opt-in | ✓ VERIFIED | Probe: store `{alpha,beta}` + incoming `{alpha}` → `ValueError`; store reopens with **both** variables at their original `(2,2)` extent. Same probe at `e549928` → no raise, store **unopenable**. No opt-in exists: `widen_data_vars` handles only the absent-in-store direction (`stored_names - incoming_names` has no widening branch anywhere) |
| DVAR-03 | Two distinct messages, MISSING checked first, placed after the dtype loop | ✓ VERIFIED | `backend.py:650-700`: `absent` block precedes `unstored` block, both after the shared-variable dtype loop. Messages differ; only the NEW one names `widen_data_vars()`/`widen_and_append()`. Mutation M3 (hoist NEW above MISSING) → **exactly** `test_a_mismatch_in_both_directions_raises_the_missing_variable_message` red. M4 (move the set check above the dtype loop) → **exactly** `test_a_shared_variable_dtype_mismatch_outranks_the_variable_set_check` red |
| DVAR-04 | No opt-out, declared or smuggled | ✓ VERIFIED | `grep -cF 'def append(self, path: str, append_dim: str = "timestamp", **kwargs) -> Self:'` = 1. Bypass-identifier count over non-comment lines of `backend.py` = **0**. M5 split reproduced exactly (see below) |
| DVAR-05 | `widen_data_vars()` materialises a new variable over the store's existing extent, variable-neutral | ✓ VERIFIED | `backend.py:290` — one definition, `Mapping[str, object]` of name→dtype, filler built from `stored.sizes`, single `to_zarr(mode="a")`. `test_widen_data_vars_backfills_the_stores_whole_existing_extent` green. Factor-vocabulary count in `backend.py` = 2, identical to the `e549928` baseline (both hits the pre-existing `FactorPolars` narrative at `:741-742`) |
| DVAR-06 | The filler carries the INCOMING dtype, not float64 | ✓ VERIFIED | M6 (hardcode `dtype="float64"`) → `test_the_filler_carries_the_incoming_variables_dtype` red, and two further tests red through their own stated assertions (`assert dtype('<f8') == dtype('bool')`) |
| DVAR-07 | The filler is written with `_append_encoding`, joining the store's chunk grid | ✓ VERIFIED | M7 (drop `encoding=`) → **exactly** `test_the_filler_joins_the_stores_existing_chunk_grid` red, `assert (10, 2) == (4, 2)` — the planning measurement reproduced verbatim |
| DVAR-08 | A non-float new variable is refused without an explicit fill | ✓ VERIFIED | Direct probe on the shipped tree: `widen_data_vars(p, {"anomaly_flag": bool, "volume": int64})` → `ValueError` naming both variables, their dtypes, and the `fill_values` remedy. M8 (disable the guard) → `test_widen_data_vars_refuses_a_non_float_variable_without_a_fill` red while `test_an_explicit_fill_widens_a_non_float_variable_and_keeps_its_dtype` stays green |
| DVAR-09 | `widen_and_append` reconciles all three axes and remains the ONE composed path | ✓ VERIFIED | Independent end-to-end probe: `{alpha}`×`[A,B]`×3 dates taking `{alpha,beta}`×`[A,B,C]`×2 later dates → `{timestamp:5, symbol:3}`, vars `['alpha','beta']`, timestamp unique **and** strictly increasing, `alpha` NaN=**3**, `beta` NaN=**9**, pre-existing history bit-identical. Every predicted number reproduced. Only three `def widen*` in the repo; no second composed entry point |
| DVAR-10 | `Factor.update()` is automatic across all three axes with NO overwrite route | ✓ VERIFIED | `factor.py:177` `def update(self, **kwargs)` — no `mode`. Behavioural lock asserts the overlap refusal fires plain **and** under `update(force=True)`, with the store byte-unchanged afterwards. M11 (route through `append()`) → tests 2, 3, 6 red, test 1 **GREEN** — the reconciliation is isolated from the plain append |
| DVAR-11 | `Factor._widen_fill_values()` seam exists, defaults `{}`, reaches the widening | ✓ VERIFIED | `factor.py:212`, returns `{}`. Reach asserted behaviourally, and independently confirmed: the seam test reddens under M6, M9 **and** M11 — in each case because the fill never reaches a working widen |
| DVAR-12 | `save()`'s behaviour and `mode="a"` default unchanged; only prose moves | ✓ VERIFIED | `git diff e549928..HEAD -- quantlab/base/factor.py` touches, inside `save`, only the Chinese docstring paragraph and the tail sentence of the wrapped message. Signature, default, matched substring, `__cause__` chaining all byte-identical. `tests/test_factor_save_mode.py` unmodified since `e549928` and its 5 tests green |
| DVAR-13 | Baseline preserved, grows only by the new tests | ✓ VERIFIED | Suite re-run by the verifier at each commit in a clean worktree: `868001b` → **553 passed**, `dea1e85` → **561 passed**, `41fcf99` → **568 passed**, zero failures at each. Matches the reported gates exactly |

**Score:** 13/13 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `quantlab/dataset/backend.py` | variable-set refusal + `widen_data_vars` + extended `widen_and_append` | ✓ VERIFIED | 886 lines; guard at `:650-700`, `widen_data_vars` at `:290`, three-axis `widen_and_append` at `:428`. All wired, all data-flowing |
| `quantlab/base/factor.py` | `update()` + `_widen_fill_values()`, `save()` prose only | ✓ VERIFIED | `update` at `:177`, `_widen_fill_values` at `:212`; `save()` diff is prose-only |
| `tests/test_variable_axis_widening.py` | 16 tests (Task 1's 8 + Task 2's 8) | ✓ VERIFIED | 16 test functions, all green, none skipped/xfailed |
| `tests/test_factor_update.py` | 7 tests | ✓ VERIFIED | 7 test functions, all green |
| `.planning/todos/pending/2026-09-07-...vestigial.md` | question filed, not decided | ✓ VERIFIED | 74 lines, three options laid out, closes "Surfaced, not decided" |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `XrBackend.append` | `_assert_append_compatible` | single existing call site | ✓ WIRED | Guard runs before `kwargs` is touched; store intact after every refusal proves it precedes `to_zarr` |
| `widen_and_append` | `widen_symbol_axis` → `widen_data_vars` → `append` | composed, in that order | ✓ WIRED | `backend.py:512-531`; closing `append()` unchanged and load-bearing (M5/M9 both observable through it) |
| `widen_data_vars` | `_append_encoding(append_dim, data=filler)` | chunk-grid single-sourcing | ✓ WIRED | `backend.py:417`; M7 proves it is load-bearing |
| `Factor.update` | `data_backend.widen_and_append(..., fill_values=self._widen_fill_values())` | the factor layer's only incremental route | ✓ WIRED | `factor.py:205` — the **only** non-test reference to `widen_and_append` in the repo. First production caller, as claimed |
| `Factor._widen_fill_values` | both widened axes | one seam, two axes | ✓ WIRED | Passed as `fill_values` into `widen_symbol_axis`'s reindex and `widen_data_vars`' filler |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Produces Real Data | Status |
|----------|------|--------|--------------------|--------|
| `widen_data_vars` filler | `shape`, `dims`, coords | read from the opened store (`stored.sizes`, `stored[name].values`), not literals | Yes — probe shows NaN counts 3/9 matching the real historical extent | ✓ FLOWING |
| `_assert_append_compatible` variable sets | `incoming_names`/`stored_names` | `self.data.data_vars` vs `xr.open_zarr(path).data_vars` | Yes | ✓ FLOWING |
| `Factor.update` fill values | `self._widen_fill_values()` | overridable per subclass; default `{}` reaches the backend | Yes — seam test reddens when the route is broken | ✓ FLOWING |

### Behavioural Spot-Checks

| Behaviour | Command | Result | Status |
|-----------|---------|--------|--------|
| All three corruption shapes refused, store intact | verifier probe against HEAD | 3/3 raise `ValueError`; store openable and unchanged in all three | ✓ PASS |
| The same three shapes on the pre-fix tree | same probe in a `e549928` worktree | 3/3 **NO RAISE**, store **unopenable** afterwards | ✓ PASS (defect confirmed real) |
| Three-axis composition end to end | verifier probe | `{timestamp:5, symbol:3}`, `alpha` NaN 3, `beta` NaN 9, history bit-identical | ✓ PASS |
| Non-float widen refusal | `widen_data_vars(p, {bool, int64})` | `ValueError` naming variables, dtypes and `fill_values` | ✓ PASS |
| Suite at each task commit | `pytest tests/ -q` in a clean worktree | 553 / 561 / 568 passed, 0 failed | ✓ PASS |
| Task 1/2/3 `<verify>` grep gates | run verbatim | all PASS | ✓ PASS |

### Mutation Re-verification (all 11 re-applied by the verifier)

| Mut | Predicted | Verifier observed | Verdict |
|-----|-----------|-------------------|---------|
| M1 | T1 1,2,3,4,5,8 red; 6,7 green | exactly that (6 failed / 10 passed) | match |
| M2 | exactly T1 2,4,5 | exactly that; test 3 green | match |
| M3 | exactly T1 5, on its message | exactly that | match |
| M4 | exactly T1 7, on its message | exactly that | match |
| M5 | SPLIT: T1 test 8 + `chunked_ingest:633` red, `:603` GREEN | exactly that — 2 failed by name, `:603` re-run alone → 1 passed | **match, split reproduced** |
| M6 | exactly T2 test 3 | T2 test 3, T2 test 6 **and T3 test 6** red | divergence (under-count) |
| M7 | exactly T2 test 4, chunks = store extent | exactly that: `assert (10, 2) == (4, 2)` | match |
| M8 | T2 test 5 red, test 6 green | exactly that | match |
| M9 | exactly T2 test 3 + T3 test 3 | those two **and T3 test 6** red; T2 test 1 green | divergence (under-count) |
| M10 | SPLIT: T3 test 4 red, T3 test 5 GREEN | *declared-but-unpassed*: exactly that (1 failed / 6 passed). *forwarded*: 6 failed, all with `TypeError: to_zarr() got multiple values for keyword argument 'mode'` at `backend.py:113`, the store-**creation** write | **ambiguity confirmed as reported** |
| M11 | T3 2,3 red; test 1 GREEN | exactly that, **plus T3 test 6** | divergence (under-count) |

`git diff --quiet` confirmed clean on both mutated files after every revert; the
mutation worktree ended `git status --porcelain` empty and was removed.

**The three divergences are under-counted predictions, not over-broad guards — verified, not accepted.**
In every case the extra red test fails through *its own plan-specified assertion*, on
the property the mutation actually breaks:

- **M6** → T2 test 6 fails on `assert dtype('<f8') == dtype('bool')`, which is
  literally its spec ("preserves the dtype exactly on disk — no float64 upcast"); T3
  test 6 fails because the bool filler becomes float64 and the closing `append()`'s
  *pre-existing* dtype guard refuses (`variable 'anomaly_flag' has dtype bool but the
  store holds float64`). Neither is a spurious detector.
- **M9** → T3 test 6's shape (agreeing symbol axis, growing variable set) is by
  construction the M9-observable shape; it receives the Task 1 new-variable refusal
  because `widen_data_vars` is never reached.
- **M11** → removing the widening path removes the seam T3 test 6 exercises.

No mutation produced a refusal on a path that should have been allowed. The positive
controls (T1 test 6, T2 tests 1/7/8, T3 test 1) stayed green under every mutation
that touched their area, which is the evidence that the guards are not over-broad.

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| DVAR-01 … DVAR-13 | see Observable Truths | ✓ SATISFIED | 13/13, each with independently reproduced evidence above |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | none | — | No `TODO`/`FIXME`/`XXX`/`TBD`/`HACK`/`PLACEHOLDER` in any of the four touched source/test files; no skipped or xfailed tests; no stub returns |

### Specific Scrutiny Items

1. **Defect actually fixed** — ✓ all three shapes reproduced on both trees. The
   destructive one (`{alpha,beta}` ← `{alpha}`) is refused with `beta` still at its
   original extent; on `e549928` the same call left the store unopenable.
2. **Both predicted SPLITS held** — ✓ M5: `test_append_offers_no_overwrite_escape_hatch`
   (`:603`) GREEN while T1 test 8 and `:633` red. M10: T3 test 4 red, T3 test 5 GREEN
   under the reading its prediction states.
3. **The three divergences** — ✓ the executor's reading is correct: under-counted
   predictions, each extra detector firing through its own stated assertion. The
   verifier found M6 under-counted by one *further* test the SUMMARY did not list
   (see Info below); the classification is unchanged.
4. **M10's ambiguity** — ✓ confirmed exactly. The *forwarded* reading fails at store
   creation (`backend.py:113`, the `mode="w"` creating write) with
   `TypeError: ... got multiple values for keyword argument 'mode'`, reddening six of
   seven tests at setup and demonstrating nothing about a route past the guard.
5. **Storage layer stayed variable-neutral** — ✓ factor-vocabulary count in
   `backend.py` = 2, byte-identical to the `e549928` baseline (both the pre-existing
   `FactorPolars` lines). Bypass-word count = **0**; no negative-grep target was
   written into source.
6. **`Factor.update()` has no overwrite route; `save()` untouched** — ✓ no `mode`
   parameter; overlap refusal fires plain and with `force=True`; `save()`'s diff is
   prose-only with `mode: Literal["a", "w"] = "a"` verbatim. The vestigial-default
   question is filed as a pending todo that closes "Surfaced, not decided" and lays
   out three options without choosing.
7. **`widen_and_append` EXTENDED, not duplicated** — ✓ exactly three `def widen*`
   methods; one composed reconcile-then-append entry point; `factor.py:205` is the
   only non-test reference to it in the repo — its first production caller.

**Working-tree hygiene:** `git status --porcelain` shows only the pre-existing
` M test.py` (mtime `2026-09-07 01:04:45`, well before all three commits) plus the
untracked SUMMARY.md awaiting the orchestrator. `test.py` appears in **none** of the
three commits (`git log --name-only e549928..HEAD` confirmed). Each commit carries an
explicit two- or three-file pathspec. `git diff HEAD -- quantlab/ tests/` is empty —
no mutation or probe residue; all verifier probes ran in scratchpad worktrees which
were removed (`git worktree list` shows only the main tree).

### Human Verification Required

None. Every truth is behaviourally exercised by a passing test *and* independently
reproduced by the verifier; the plan declared no deferred human checks.

### Info (non-blocking)

- ℹ️ **SUMMARY's M6 row is under-counted by one test.** The verifier's M6 run reddens
  `test_the_widen_fill_seam_reaches_the_widening_call` (Task 3 test 6) in addition to
  the two the SUMMARY lists. Most likely explanation: M6 is a Task 2 mutation and
  `tests/test_factor_update.py` did not exist at commit `dea1e85`. The mechanism was
  verified (the bool filler upcast to float64 is refused by the pre-existing dtype
  guard at the closing `append()`), so this changes nothing about the classification
  — it is the same "the plan's own test specs create a second detector" family the
  SUMMARY already reports for M6/M9/M11. No code implication.
- ℹ️ **Open developer decision, correctly filed not decided:**
  `.planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md`.
  This is the plan's intended sink (D-10), so it is not a verification gap.

### Gaps Summary

None. The task goal is achieved on all three fronts: the refusal exists and is
unconditional in both directions with the destructive one refused outright; a
variable-neutral `widen_data_vars` is folded into the single composed
`widen_and_append`; and `Factor.update()` is an automatic three-axis interface with
no overwrite route, with `save()`'s behaviour and `mode="a"` default untouched. The
storage layer carries no factor vocabulary beyond its pre-existing baseline.

---

_Verified: 2026-09-08_
_Verifier: Claude (gsd-verifier)_
