---
phase: quick-260908-0f4
verified: 2026-09-08T06:20:00Z
status: passed
score: 14/14 must-haves verified
covered_files:
  - ".planning/quick/260908-0f4-give-dataset-an-automatic-update-interfa/260908-0f4-PLAN.md"
  - ".planning/quick/260908-0f4-give-dataset-an-automatic-update-interfa/260908-0f4-SUMMARY.md"
  - ".planning/todos/pending/2026-09-08-wire-dataset-update-into-the-ingest-cli.md"
  - "example/chunking.md"
  - "quantlab/base/data.py"
  - "quantlab/dataset/backend.py"
  - "quantlab/dataset/stock.py"
  - "tests/test_chunked_ingest.py"
  - "tests/test_dataset_update.py"
  - "tests/test_dataset_update_evidence.py"
covered_digest: "v1:sha256:4d3c49658029c1dc1fdd701d7c4bfd6296b0bc13a6a7932795977d3de00ba65a"
behavior_unverified: 0
overrides_applied: 0
advisory:
  - finding: "`260907-vyr` shipped `Factor.update()` with its variable-widening path broken for any store carrying an object-dtype string coordinate -- i.e. every store the current chunked ingest writes. Reproduced independently at `c8bba70^`: `Factor.update()` on such a store raises `ValueError: Mismatched dtypes for variable symbol ... Store has dtype object but dataset to append has dtype StringDType()`; the same script succeeds at HEAD. That task's 13/13 verification did not catch it."
    category: architectural
    reason: "A verification gap in the PRIOR task, closed by this one. Raised so it is recorded rather than rediscovered; nothing in this task is blocked by it."
    evidence_status: "reproduced live (see Behavioural Spot-Checks, rows 3-5)"
  - finding: "The suites that OWN the widening methods -- `tests/test_variable_axis_widening.py`, `tests/test_symbol_axis_widening.py`, `tests/test_factor_update.py` (34 tests) -- ALL pass against the pre-fix, broken backend. Their fixtures build symbol coordinates from python list literals, which xarray types `<U3`; the real polars/pandas-derived panels type `object`. The blind spot is now covered only incidentally, by `tests/test_chunked_ingest.py`'s Task-3 tests. A coordinate-encoding fixture in the owning suites (and one on the factor path) would put the lock where the method lives."
    category: other
    reason: "Coverage placement, not a defect. The behaviour is correct at HEAD and IS locked by at least one suite."
    evidence_status: "reproduced live (34 passed against the pre-fix backend)"
  - finding: "SUMMARY.md says the pre-fix defect made variable widening unreachable for `every real store in this project`. Measured: of the three `.zarr` stores on disk, `data/data/us_equity/1m/stock_alpaca.zarr` carries `StringDType()`/object (would have failed) but `data/data/us_equity/1d/us_all.zarr` carries `<U9` (would have succeeded). The accurate claim is `every store the current chunked-ingest path writes`, which the Task-3 fixtures demonstrate."
    category: other
    reason: "Overstatement of scope in the narrative; the substantive claim (the defect was real, reachable, and shipped) holds."
    evidence_status: "measured live (zarr dtype scan of data/**/*.zarr)"
---

# Quick Task 260908-0f4: Automatic `Dataset.update()` — Verification Report

**Task Goal:** `BaseDataset.update()` as an automatic interface symmetric with `Factor.update()`, resolving `widen` vs `rebuild` from RAW-LAYER EVIDENCE rather than a caller-supplied flag; a REMOVED symbol resolves to `refuse`; the automatic sentinel is a non-string object unreachable from the CLI/config/JSON; the decision is REPORTED before a rebuild runs, not switched silently; and `data_vars` reconciliation reaches the chunked path.

**Verified:** 2026-09-08
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| DSUP-01 | `update()` is the one automatic entry point, no strategy parameter | ✓ VERIFIED | `quantlab/base/data.py:471` — signature is `(self, granularity="year", ledger_path=None, append_dim="timestamp")`. `test_update_exposes_no_strategy_parameter` pins `set(parameters) == {"self","granularity","ledger_path","append_dim"}` and asserts `on_new_listing`/`mode`/`force`/`overwrite` absent |
| DSUP-02 | `update()` WRAPS `from_raw_data_chunked`, forwarding everything | ✓ VERIFIED | `data.py:531-536` — a single forwarding call, the only difference being `on_new_listing=self._AUTOMATIC`. No window planning, ledger or resume logic duplicated |
| DSUP-03 | Resolution happens inside `_reconcile_new_listings`, one site | ✓ VERIFIED | `data.py:1008-1011` — the sentinel is exchanged for a published string immediately after `added`/`removed` are computed, AFTER the early fall-through at `data.py:1004`. `test_an_unchanged_roster_never_touches_the_raw_evidence_probe` makes the probe raise and the run still completes |
| DSUP-04 | Sentinel is a NON-STRING `object()`, unreachable from CLI/config/JSON | ✓ VERIFIED | `data.py:74` `_AUTOMATIC = object()`; `quantlab/utils/cli.py:192-196` `--on-new-listing` is `type=str, choices=list(NEW_LISTING_STRATEGIES), default="refuse"`. Validation at `data.py:587` admits it by IDENTITY only. Mutation M6 (sentinel → `"automatic"`) reddens 2 tests — reproduced independently |
| DSUP-05 | `NEW_LISTING_STRATEGIES` byte-identical, default still `"refuse"`, annotation widened | ✓ VERIFIED | `data.py:59` unchanged; `data.py:543` `on_new_listing: str \| object = "refuse"`. `git diff c0b20b7..HEAD -- quantlab/base/data.py` removes exactly **4** lines: the old annotation, the old validation `if`, and the two lines of the old plain-`append` call. All three strategy branch bodies untouched |
| DSUP-06 | Three-way evidence rule; REMOVED → `refuse` | ✓ VERIFIED | `_resolve_new_listing_strategy` (`data.py:898-978`): `removed` → `"refuse"`; no extent → `"widen"`; non-empty evidence → `"rebuild"`; empty → `"widen"`. Locked by `test_history_inside_the_stores_extent_resolves_to_rebuild`, `test_a_genuine_new_listing_resolves_to_widen`, `test_one_qualifying_symbol_rebuilds_the_whole_store`, `test_a_removed_symbol_resolves_to_refuse_and_leaves_the_store_alone` (which asserts `xr.testing.assert_identical` against the pre-run panel). Mutation M2 reproduced independently: 3 red |
| DSUP-07 | The evidence question is asked over the STORE's extent, not the config range | ✓ VERIFIED | `data.py:936-948` passes `_stored_append_extent(store_path, append_dim)` to the probe. `test_the_probe_is_asked_about_the_stores_extent_not_the_config_range` records the `(start,end)` the probe receives and asserts it equals the store's own first/last labels, having FIRST asserted the config range strictly brackets the extent on both sides (so an accidental equality is excluded) |
| DSUP-08 | The decision is SPOKEN before a rebuild runs, with symbols and row counts, capped at 20 | ✓ VERIFIED | `data.py:961-978` — `logger.warning` naming `N of M`, `symbol=rows` sorted descending, `NEW_LISTING_REPORT_LIMIT = 20` with an explicit truncation clause; a `logger.info` on the widen branch too. Ordering is control-flow enforced: the log precedes `return "rebuild"`, and the rebuild branch only runs on that return. `test_the_rebuild_decision_is_reported_with_symbols_and_row_counts` asserts `C=9` and `rebuild` in the captured loguru output |
| DSUP-09 | `data_vars` reconciled on the chunked path via `widen_and_append` with `_widen_fill_values()` | ✓ VERIFIED | `data.py:702-706` — exactly one `widen_and_append` call site, `fill_values=self._widen_fill_values()`. No second reconciliation in the Dataset layer (`widen_symbol_axis` has one call site, `data.py:1040`, in the pre-existing `widen` branch). **Mutation M1 applied independently:** reverting to plain `append` reddens exactly T3 1,2,3,4 and leaves T3 5 green — the predicted split |
| DSUP-10 | A window missing a stored variable is still refused; the rename residue is LOCKED | ✓ VERIFIED | `test_a_window_missing_a_stored_variable_is_still_refused` asserts the refusal, then reads the store back and asserts `sorted(data_vars) == before + ["adjCloseV2"]`, `sizes["timestamp"]` unchanged, `adjClose` values bit-identical (`assert_array_equal`), and `adjCloseV2` all-NaN. Reddens under M1 on the post-refusal set — confirmed independently |
| DSUP-11 | Overlap unconditionally refused; no gap check added | ✓ VERIFIED | Zero loosening keywords on non-comment lines (gate returns 0). No adjacency/gap comparison in the `data.py` diff. `test_agreeing_axes_still_go_through_the_unchanged_append_guard` asserts the closing unchanged `append` is reached once per written window |
| DSUP-12 | `_widen_fill_values()` honoured on BOTH widened axes of the chunked path | ✓ VERIFIED | Two call sites: `data.py:705` (per-window / data_vars axis) and `data.py:1044` (symbol axis). `test_a_non_float_new_variable_is_widened_with_the_declared_fill` asserts `halted` keeps `bool` dtype, is `False` over the store's pre-existing extent and `True` where the window supplied it |
| DSUP-13 | 568 → 592 passed, per-task gates 576 / 587 / 592 | ✓ VERIFIED | Re-run independently: `592 passed, 273 warnings in 36.17s`, RC=0. 11 + 8 new test functions in the two new files + 5 in `test_chunked_ingest.py` = 24; 568 + 24 = 592 |
| DSUP-14 | The full-suite gate cannot pass while any test failed or errored | ✓ VERIFIED | Task 3's verify line executed verbatim at HEAD → **rc=0**. Both arms present (`set -o pipefail` + `test "$RC" -eq 0`, and `test "$(grep -cE '[0-9]+ (failed\|error)' <<<"$TAIL")" -eq 0`). All three verify lines contain **no** `!` and return rc=0 with empty stderr from both `bash -n` and `zsh -n` |

**Score:** 14/14 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `quantlab/base/data.py` | `update`, sentinel, resolver, extent helper, probe default, widen_and_append swap | ✓ VERIFIED | All present, wired, exercised; 4-line removal only |
| `quantlab/dataset/stock.py` | hive-pruned probe override | ✓ VERIFIED | `stock.py:409-445`, built on `_scan_raw(start, end)` — polars group-by/len, counts only collected |
| `quantlab/dataset/backend.py` | (unplanned) `widen_data_vars` coordinate fix | ✓ VERIFIED | `c8bba70` removes the `coords={...}` from the filler; load-bearing, see Behavioural Spot-Checks |
| `tests/test_dataset_update_evidence.py` | 8 tests | ✓ VERIFIED | Exactly 8 `def test_` functions, matching the 8 declared behaviours in order |
| `tests/test_dataset_update.py` | 11 tests | ✓ VERIFIED | Exactly 11 `def test_` functions |
| `tests/test_chunked_ingest.py` | +5 tests in a new section | ✓ VERIFIED | New section at line 1255; 5 new test functions |
| `example/chunking.md` | steps (4)/(6) re-pointed, entry-point contrast | ✓ VERIFIED | 4 `update()` references; modified in `c8bba70` |
| `.planning/todos/pending/2026-09-08-wire-dataset-update-into-the-ingest-cli.md` | D-10 filed | ✓ VERIFIED | Exists, 3,262 bytes, committed in `c8bba70` |

### Key Link Verification

| From | To | Via | Status |
|---|---|---|---|
| `BaseDataset.update()` | `from_raw_data_chunked` | `on_new_listing=self._AUTOMATIC` (`data.py:535`) | ✓ WIRED |
| `from_raw_data_chunked` | `_reconcile_new_listings` | `data.py:621` | ✓ WIRED |
| `_reconcile_new_listings` | `_resolve_new_listing_strategy` | identity check at `data.py:1008` | ✓ WIRED |
| `_resolve_new_listing_strategy` | `_added_symbols_with_raw_history` | `data.py:948`, window = `_stored_append_extent` | ✓ WIRED |
| `StockDataset` probe | `_scan_raw` | `stock.py:436`, hive-pruned | ✓ WIRED (M10 reddens 5 tests) |
| per-window write | `XrBackend.widen_and_append` → unchanged `append` | `data.py:702` | ✓ WIRED (M1 reddens 4 tests) |
| `_resolve_new_listing_strategy` | the three EXISTING branches | returns a published string; branch bodies untouched in the diff | ✓ WIRED |

### Behavioural Spot-Checks

All run by this verifier, not read from SUMMARY.md.

| # | Behaviour | Command | Result | Status |
|---|---|---|---|---|
| 1 | Full suite | `set -o pipefail; uv run pytest tests/ -q` | `592 passed, 273 warnings in 36.17s`, RC=0 | ✓ PASS |
| 2 | Task 3 verify line, verbatim from PLAN.md | `bash g3.sh` | rc=0 (`87 passed` subset, `592 passed` full, all source greps) | ✓ PASS |
| 3 | **Pre-fix defect exists on a realistically-encoded store** | build a store whose `symbol` coord is object-dtype, call `widen_data_vars` under `c8bba70^` | `ValueError: Mismatched dtypes for variable symbol ... Store has dtype object but dataset to append has dtype StringDType()`; identical case under HEAD → SUCCESS, `keep` bit-identical, coords untouched, new var all-NaN | ✓ PASS (defect confirmed, fix confirmed) |
| 4 | **The new tests catch it** | `git checkout c8bba70^ -- quantlab/dataset/backend.py; pytest tests/test_chunked_ingest.py -k "variable or agreeing"` | **4 failed, 1 passed** — the 4 new data_vars tests redden with the dtype error; the agreeing-axes lock stays green | ✓ PASS |
| 5 | **`Factor.update()` was broken pre-fix for realistic stores** | `Factor.update()` twice over an object-dtype symbol coord, second run adding a variable | HEAD: `SUCCEEDED; vars = ['alpha','beta'] timestamps = 4`. `c8bba70^`: `RAISED ValueError: Mismatched dtypes for variable symbol ...` | ✓ PASS (260907-vyr shipped broken — see advisory 1) |
| 6 | **Existing widening suites are blind** | `git checkout c8bba70^ -- backend.py; pytest test_variable_axis_widening test_symbol_axis_widening test_factor_update -q` | **34 passed** against the broken backend | ✓ PASS (blind spot confirmed — advisory 2) |
| 7 | Store dtype survey | zarr scan of `data/**/*.zarr` | `us_all.zarr` `<U9`; `1m/stock_alpaca.zarr` `StringDType()`; `1d/stock_alpaca.zarr` zero-length | ✓ PASS (advisory 3) |
| 8 | D-12 gate discrimination | run the gate regex against a copy of `data.py` carrying the pre-reword docstring | pre-reword → count **1** (gate fails); current → count **0** (gate passes) | ✓ PASS |
| 9 | Gate not weakened | `git diff c0b20b7 HEAD -- ...PLAN.md` | **0 lines** — PLAN.md untouched since `c0b20b7`; no gate edited | ✓ PASS |
| 10 | Verify lines shell-safe | `bash -n` / `zsh -n` on each of the 3 extracted `<automated>` lines | all rc=0, empty stderr; `!` count = 0 in all three | ✓ PASS |

### Mutation Re-Verification (independently applied by this verifier)

Five mutations applied, full suite run, reverted, `git diff --quiet` confirmed clean on each mutated file.

| # | Mutation | SUMMARY's observed | **My observed** | Verdict |
|---|---|---|---|---|
| M1 | plain `append` per window | T3 1,2,3,4 red; T3 5 green | `4 failed, 588 passed` — exactly those four | **reproduces** |
| M2 | always widen, probe still called | T2 1 red + also T2 3, T2 5 | `3 failed`: `..._resolves_to_rebuild`, `test_one_qualifying_symbol_rebuilds_the_whole_store`, `test_the_rebuild_decision_is_reported_...`; T2 2 and T2 11 green | **reproduces** |
| M6 | sentinel is `"automatic"` | T2 7 red + also T2 8 | `2 failed`: `test_no_string_reaches_the_automatic_branch`, `test_the_sentinel_is_absent_from_the_published_strategies` | **reproduces** |
| M9 | sentinel added to the tuple | T2 8, T2 7 and `test_an_unknown_strategy_lists_the_accepted_values` red; the named wiring test **GREEN** | `3 failed`: exactly those three; `tests/test_ingest_tiingo_universe_wiring.py` **green** | **reproduces — divergence is real** |
| M10 | probe counts the whole tier | T1 5, 6, T2 2 red + also T1 7, T1 8 | `5 failed`: T1 5, 6, 7, 8 and `test_a_genuine_new_listing_resolves_to_widen` | **reproduces** |

**Scrutiny item 3 — M9's divergence.** Both halves confirmed. The named guardian, `tests/test_ingest_tiingo_universe_wiring.py:470`, asserts `list(action.choices) == list(BaseDataset.NEW_LISTING_STRATEGIES)`, and `quantlab/utils/cli.py:195` builds those choices as `list(BaseDataset.NEW_LISTING_STRATEGIES)`. Growing the tuple grows both sides identically, so the equality is invariant under M9 — the test **structurally cannot redden**, exactly as reported. The protection genuinely exists elsewhere and is broader than the plan accounted for: three tests redden, including the pre-existing `test_an_unknown_strategy_lists_the_accepted_values`, which iterates the tuple asserting each member appears in the error string (an `object()` does not). T-0f4-04 is mitigated; only the plan's identification of the guardian was wrong. The executor reported this rather than smoothing it — correct behaviour.

**Scrutiny item 4 — the four under-counted predictions.** Each extra reddened test fails through its OWN specified assertion, not through collateral breakage:
- M2 / `test_one_qualifying_symbol_rebuilds_the_whole_store` → `assert len(dataset.window_calls) == len(_STORE_YEARS) + 1` → `assert 1 == 4`. That is the test's own subject (rebuild is whole-store).
- M2 / `test_the_rebuild_decision_is_reported_...` → `assert 'C=9' in blob`, the captured log being the *widening* message. Its own subject.
- M6 / `test_the_sentinel_is_absent_from_the_published_strategies` → its own `assert not isinstance(BaseDataset._AUTOMATIC, str)`.
- M10 / `test_a_subclass_that_did_not_override_the_seam_still_gets_the_answer` → `{'SPAN':3,'STRADDLE':5,'LATER':2} != {'SPAN':3,'STRADDLE':3}`, its own agreement assertion.
- M10 / `test_the_stock_override_reaches_raw_through_the_pruned_scan` → its own `assert found == {"SPAN": 3}`.
Positive controls held in every case (M2: T2 2 and T2 11 green; M1: T3 5 green; M9: the wiring test green). These are **under-counted predictions**, not over-broad behaviour.

**Scrutiny item 5 — the two green-before tests.** Honest, and the partners genuinely drive.
- `test_no_string_reaches_the_automatic_branch` uses only `from_raw_data_chunked` and `NEW_LISTING_STRATEGIES`, both pre-existing, so it was necessarily green before implementation. Its partner `test_the_sentinel_is_absent_from_the_published_strategies` dereferences `BaseDataset._AUTOMATIC` and asserts `not isinstance(..., str)` — an `AttributeError` before the sentinel existed, so genuinely red-driven. I confirmed M6 reddens **both**, so the pair is not decorative: the lock has demonstrated discriminating power against the exact threat it names.
- `test_agreeing_axes_still_go_through_the_unchanged_append_guard` was trivially green pre-swap. It names a hypothetical RED (a second write path), not an applied mutation, and stayed green under M1 as predicted — a genuine fast-path lock rather than a driver. Its partners (T3 1-4) all drove: I reddened all four independently under both M1 and the pre-fix backend.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| — | — | none | — | Zero `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`/"not yet implemented" across all 8 modified/created source, test and doc files |

### Hygiene Checks

| Check | Result |
|---|---|
| `test.py` untouched | ✓ `git status --porcelain` shows ` M test.py` only; `git log --oneline c0b20b7..HEAD -- test.py` returns **0** lines; file mtime `2026-09-07 01:04:45`, a day before the task's commits (`2026-09-08 01:3x`) |
| Explicit pathspec per commit | ✓ verified by proxy — each commit's file set matches its task's declared `<files>` exactly (`cad9fdb`: 3 files; `65dc287`: 2; `c8bba70`: 5 incl. the unplanned `backend.py` fix). A `commit -a` would have swept in the long-standing ` M test.py`; it did not appear in any of the three |
| No mutation or probe residue | ✓ working tree after all verifier mutations is ` M test.py` + the untracked `260908-0f4-SUMMARY.md`. All probe artefacts written to the scratchpad, never to the repo |
| Three verify lines free of `!` | ✓ 0 occurrences; `bash -n` and `zsh -n` both rc=0 with empty stderr on all three |
| No gate edited | ✓ PLAN.md diff since `c0b20b7` is 0 lines |

## Gaps Summary

None. Every must-have is verified against the codebase, and the three claims that most needed independent confirmation — the latent `widen_data_vars` defect, the D-12 gate's fail-closed firing, and M9's divergence — were each reproduced from scratch rather than accepted from the narrative.

Three advisory findings are recorded in the frontmatter. The most consequential is a **verification gap in the preceding task**: `Factor.update()`'s variable-widening path was broken at `260907-vyr` for any store carrying an object-dtype string coordinate, which is what this project's own chunked ingest writes. That task passed 13/13 because its fixtures build symbol coordinates from python list literals (`<U3`) rather than through the polars/pandas path (`object`). This task found and fixed it; the recommendation is to put the lock where the method lives — an object-dtype-coordinate fixture in `tests/test_variable_axis_widening.py` and on the factor path — since those 34 tests still pass against the broken backend today.

---

_Verified: 2026-09-08_
_Verifier: Claude (gsd-verifier)_
