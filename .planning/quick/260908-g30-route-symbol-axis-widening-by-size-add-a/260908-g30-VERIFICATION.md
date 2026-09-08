---
phase: quick-260908-g30
verified: 2026-09-08T12:20:00Z
status: passed
score: 12/12 must-haves verified
covered_files:
  - ".planning/quick/260908-g30-route-symbol-axis-widening-by-size-add-a/260908-g30-PLAN.md"
  - ".planning/quick/260908-g30-route-symbol-axis-widening-by-size-add-a/260908-g30-SUMMARY.md"
  - ".planning/todos/pending/2026-09-08-chunked-widen-data-vars.md"
  - "example/backend.md"
  - "quantlab/base/data.py"
  - "quantlab/dataset/backend.py"
  - "tests/test_factor_update.py"
  - "tests/test_symbol_axis_widening.py"
  - "tests/test_widening_fixture_realism.py"
covered_digest: "v1:sha256:6837b290d4a346d51a3ee9f1d7b57ac4a1834e27fd4404079f75dd18d6170c9c"
behavior_unverified: 0
overrides_applied: 0
---

# Quick 260908-g30: Route symbol-axis widening by size — Verification Report

**Task Goal:** Route symbol-axis widening by size: add a memory-bounded chunked widening path to `XrBackend` and a materialisation estimate that switches to it above budget, reporting which path was taken.
**Verified:** 2026-09-08
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| RWG-01 | `widen_symbol_axis` picks its strategy BY SIZE and says which; one public entry, no new public parameter; chunked-branch tests run on 600-timestamp stores | ✓ VERIFIED | `quantlab/dataset/backend.py:162` — signature byte-identical to pre-task (`git diff aab7439 HEAD` shows no changed `def` line); router at L323-347 computes `estimate`/`block_rows`/`chunked` then selects `self._widen_chunked` or `self._widen_whole_store`. No strategy parameter exists. Every chunked-forcing test builds `_LONG_DATES` (600 daily rows) / `LONG` (600 hourly rows) and forces via `monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)` |
| RWG-02 | Both paths ship behind one router; no caller names a strategy and no caller changed | ✓ VERIFIED | Two private strategies (`_widen_whole_store` L514, `_widen_chunked` L545) with one identical signature; `grep -rn "widen_symbol_axis\|widen_and_append" quantlab/` — the only call sites (`backend.py:847`, `base/data.py:1046`, `base/factor.py:205`) are unchanged in argument list; `quantlab/base/factor.py` is not in the task diff at all |
| RWG-03 | Block size comes from the budget, floored onto `APPEND_DIM_CHUNK`; `TimeChunkPlanner` rejected with a stated reason | ✓ VERIFIED | `_widen_block_rows` (L411-451) implements D-2 verbatim: `raw = MAX_WIDEN_BYTES // row_bytes if row_bytes > 0 else 0; return max(chunk, (raw // chunk) * chunk)`. Docstring states the `TimeChunkPlanner` rejection with the calendar-vs-bytes reason. `test_the_block_size_rule_floors_onto_the_chunk_grid` passes and asserts floor, multiple-of, and monotonicity |
| RWG-04 | The grid constraint is load-bearing: first block carries `_append_encoding`, so the two paths land on one grid | ✓ VERIFIED | `_widen_chunked` L587-600: first block writes `mode="w", encoding=self._append_encoding(append_dim, data=block)`; later blocks append and drop non-`append_dim` vars. `test_a_chunked_widen_lands_on_the_stores_own_chunk_grid` passes: `zarr…["close"].chunks == (512, 3)`. Independently mutated: forcing `_widen_block_rows` to return 100 reddens both grid locks (see Behavioral Spot-Checks) |
| RWG-05 | Both paths observationally identical: values, both coords, on-disk chunks, on-disk symbol ENCODING, under both live encodings | ✓ VERIFIED | `test_the_two_widen_strategies_leave_identical_stores[fixed_width]` and `[variable_length]` pass; asserts `np.array_equal(..., equal_nan=True)`, both coords, `zarr.open_group(...)["close"].chunks` equality, and `stored_symbol_dtype`/`stored_symbol_encoding` read straight off zarr (not through decoded `xr.open_zarr`) |
| RWG-06 | Swap ordering re-verified against the loop: loop inside `stored`'s lifetime, `os.replace` pair outside, `rmtree` cleanup spans the whole strategy call | ✓ VERIFIED | Router L336-352: `try: strategy(...) except BaseException: shutil.rmtree(widening, ignore_errors=True); raise` inside the `try:` whose `finally: stored.close()` precedes both `os.replace` calls and the closing `rmtree(superseded)`. `test_a_crash_part_way_through_the_block_loop_leaves_the_store_intact` passes (crashes on write 2 of 2, asserts `xr.testing.assert_identical`, no `.widening.tmp`, no `.superseded.tmp`) |
| RWG-07 | `widen_data_vars` filed rather than dropped, carrying its cost and blocking reason | ✓ VERIFIED | `.planning/todos/pending/2026-09-08-chunked-widen-data-vars.md` exists (78 lines), front-matter matches sibling briefs, carries the ~2.4 GiB/variable figure, the `append_dim`-trick-unavailable reason, and the `region=` requirement |
| RWG-08 | THE SWITCH IS REPORTED — chunked warns with all figures; whole-store logs one info line | ✓ VERIFIED | `_report_widen_strategy` (L462-512) — `logger.warning` naming both symbol counts, estimate + budget in GiB, `block_rows`, block count, per-block MiB, the `~3.6-4.0x` wall-clock note, and `Raise XrBackend.MAX_WIDEN_BYTES` opt-out; `logger.info` on the routine path. `test_the_budget_routes_in_both_directions` passes and asserts BOTH directions plus levels (`INFO|` vs `WARNING|`), `MAX_WIDEN_BYTES` and `3.6-4.0x` in the warning |
| RWG-09 | `MAX_WIDEN_BYTES = 4 * 1024**3`, separate constant, sibling precedent cited, routes the measured table | ✓ VERIFIED | `quantlab/dataset/backend.py:65` = `4 * 1024**3` with a `#:` block citing `UniverseCatalog.MAX_DENSE_PANEL_BYTES` (confirmed `= 4 * 1024**3` at `universe.py:1616`) and the `MAX_RAW_BYTES` sibling precedent (confirmed at `universe.py:2096`). Backend imports only `quantlab.base.backend` — no acquisition import added. Routing table re-derived independently: 0.3 GiB daily-year < 4 GiB; 7.3 / 43.9 GiB minute panels > 4 GiB |
| RWG-10 | The factor layer reaches the bounded path with no call-site change, proved from `test_factor_update.py` | ✓ VERIFIED | `tests/test_factor_update.py::test_update_reaches_the_bounded_widen_path_with_no_call_site_change[fixed_width|variable_length]` passes: 600 HOURLY rows (so `_auto_filter()`'s 2024 window does not truncate below 512), asserts the warning names this store, block count `== 2`, block size `== APPEND_DIM_CHUNK`, history preserved, new symbol NaN over history and non-NaN after. `quantlab/base/factor.py` is untouched by the task diff |
| RWG-11 | The now-false cost claims are reconciled in all three files | ✓ VERIFIED | All three gate greps pass and the phrases occur NOWHERE under `quantlab/`, `example/`, `tests/`: `WHOLE store in RAM` gone from `backend.py`, `too large to hold` gone from `base/data.py`, `库大到装不下时` gone from `example/backend.md`. Replacements read as the routing contract; `rebuild`'s surviving justification is stated as HISTORY RECOVERY in all three (`backend.py` docstring tail, `data.py:1037-1046` warning, `example/backend.md` final new bullet) |
| RWG-12 | The 260908-dvv family guards are UPDATED, not weakened | ✓ VERIFIED | `tests/test_widening_fixture_realism.py:165` — `assert checked == 40` (was 33), with the comment re-stating the 40-vs-41 arithmetic; `found == 5` unchanged at L218; `_EXEMPT` unchanged (`frozenset({"test_update_declares_no_overwrite_parameter"})`, L84). Guard passes. Every new store-touching test generated both `[fixed_width]` and `[variable_length]` ids (observed in the mutation runs below) |

**Score:** 12/12 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `quantlab/dataset/backend.py` | Router, estimate, block rule, two strategies, report | ✓ VERIFIED | +356 lines; `MAX_WIDEN_BYTES`, `_estimate_widen_bytes`, `_widen_block_rows`, `_widen_blocks`, `_report_widen_strategy`, `_widen_whole_store`, `_widen_chunked`, router body. Three write-front guards present, unchanged, and still ahead of any write |
| `quantlab/base/data.py` | `_reconcile_new_listings` warning + `NEW_LISTING_STRATEGIES` reconciled | ✓ VERIFIED | 14 lines changed; the unconditional-materialisation sentence pair replaced by the routing statement; the evidence-based resolver untouched (chooses on raw rows) |
| `tests/test_symbol_axis_widening.py` | Tracer + equivalence + bounded + grid + threshold + rule + crash | ✓ VERIFIED | +403 lines; 6 new tests + `_LONG_DATES`/`_captured_warnings`/`_reported_block_count`. All pass under both encoding arms |
| `tests/test_factor_update.py` | Factor-side lock | ✓ VERIFIED | +86 lines; `LONG`/`AFTER_LONG` + the bounded-path lock, both arms passing |
| `tests/test_widening_fixture_realism.py` | Re-derived literals, `_EXEMPT` untouched | ✓ VERIFIED | `checked` 33 → 40 with re-stated comment; `found == 5` and `_EXEMPT` unchanged |
| `example/backend.md` | Chinese cost bullet replaced by the router contract | ✓ VERIFIED | 1 bullet removed, 5 added; cites the new tests by name in the neighbours' style; Chinese kept |
| `.planning/todos/pending/2026-09-08-chunked-widen-data-vars.md` | Filed follow-up | ✓ VERIFIED | Created, sibling front-matter shape, cost + blocking reason + reusable pieces |
| `.planning/todos/completed/…-chunked-symbol-axis-widening.md`, `…-no-memory-guard-before-a-symbol-axis-widen.md` | Moved with `git mv`, `## Closed` appended | ✓ VERIFIED | Commit `4941e8d` records both as renames (`R062`, `R056`), `status: completed`, each with a `## Closed` section (L138 / L126). Neither remains in `pending/` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `_widen_chunked` first block | `_append_encoding(append_dim, data=block)` | single-sourced chunk rule | ✓ WIRED | L590-596; grid locks red under an off-grid block size (M-C reproduced below) |
| block loop | `try/finally stored.close()` → the two `os.replace` calls | D-3 ordering | ✓ WIRED | Loop runs inside the handle's lifetime; renames strictly after `finally`; crash lock green, and red under M-A |
| `Factor.update()` | `widen_and_append()` → `widen_symbol_axis()` | unchanged call sites | ✓ WIRED | `quantlab/base/factor.py:205` unmodified; factor-side test observes the chunked warning naming its own store |
| new tests | `test_widening_fixture_realism.py` `checked`/`found` | family guard | ✓ WIRED | Guard green with `checked == 40`; `found` correctly unmoved (no new panel builder — the 600-entry date list is an argument to the existing `_panel`/`PanelFactor.cal()`) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `_widen_chunked` | each block's `close`/`alpha` values | `stored.isel(...).load().reindex(...)` from the real store | Yes — pre-existing history asserted bit-identical, new symbol NaN over history only | ✓ FLOWING |
| `_estimate_widen_bytes` | `widened_bytes` / `row_bytes` | real `variable.sizes` / `dtype.itemsize` off the open dataset (no hardcoded figures) | Yes | ✓ FLOWING |
| `_report_widen_strategy` | block count | `len(self._widen_blocks(...))` — same helper the loop walks | Yes (2 for a 600-row store; asserted from the parsed log line) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full suite green | `uv run python -m pytest -q` | `649 passed` in 41s (matches the SUMMARY's claimed count) | ✓ PASS |
| Task 3 doc gates | the three `grep -qF` literals + repo-wide residue scan over `quantlab/ example/ tests/` | all three phrases absent everywhere | ✓ PASS |
| Task 3 file-move gates | `ls .planning/todos/pending` / `completed` + `git log --name-status 4941e8d` | new todo present; both briefs absent from `pending/`, present in `completed/`, recorded as renames | ✓ PASS |
| M-A mutation (chunked collapses to one whole-store write), applied IN-MEMORY via a pytest plugin — source tree never modified | `pytest tests/test_symbol_axis_widening.py tests/test_factor_update.py -p mut_a` | `4 failed`: `test_the_chunked_widen_writes_in_bounded_blocks` + `test_a_crash_part_way_through_the_block_loop_leaves_the_store_intact`, both arms — **exactly the red set the SUMMARY records for M-A, including its stated caveat that the tracer's block-count assertion does NOT move** | ✓ PASS |
| M-C mutation (`_widen_block_rows` → 100, off the grid), same in-memory technique | `pytest … -p mut_c` | `11 failed`: tracer, equivalence, bounded-blocks, chunk-grid landing, block-size rule, factor lock (both arms each) — matches the SUMMARY's recorded M-C red set | ✓ PASS |
| Working tree unpolluted by verification | `git status --short quantlab tests example` | empty | ✓ PASS |

### Probe Execution

No probes declared or implied (no `scripts/*/tests/probe-*.sh` in the repo; PLAN and SUMMARY mention none). Step 7c: SKIPPED.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| RWG-01 … RWG-12 | `260908-g30-PLAN.md` | Plan-local requirement IDs for this quick task | ✓ SATISFIED | See the Observable Truths table — each RWG id maps 1:1 to a truth, all verified |

`.planning/REQUIREMENTS.md` maps no requirement to this quick task, so there are no orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No `TBD`/`FIXME`/`XXX`/`HACK`/`PLACEHOLDER`/stub marker in any file this task touched | — | none |
| `quantlab/dataset/backend.py` | 448-451, 453-460 | `_widen_block_rows` / `_widen_blocks` read `XrBackend.APPEND_DIM_CHUNK` / `XrBackend.MAX_WIDEN_BYTES` off the class by name while the router reads `self.MAX_WIDEN_BYTES` | ℹ️ Info | A subclass that raised the budget would route on its own value but size blocks against the base value — smaller blocks than needed, never incorrect. No subclass exists today; tests monkeypatch `XrBackend` itself, so the seam is exercised as written |
| `tests/test_symbol_axis_widening.py` | `test_the_block_size_rule_floors_onto_the_chunk_grid` | The SUMMARY's coverage entry D9 cites this test as the lock for "`MAX_WIDEN_BYTES` = 4 GiB"; the test derives `budget` FROM the constant and never asserts the literal | ℹ️ Info | The truth RWG-09 is a claim about the source, and the source is correct — so this is not a gap. But the value itself is not pinned the way the sibling `MAX_RAW_BYTES` is (`tests/test_volume_guard.py:841` asserts `== 20 * 1024**3`). A one-line literal assertion would close the asymmetry |
| `quantlab/dataset/backend.py` | 514-543 | `_widen_whole_store`'s write now runs inside `stored`'s open lifetime, where the shipped code wrote after `stored.close()` | ℹ️ Info | Source and destination are different directories and both renames still follow the close, so behaviour is preserved on POSIX; the 38 pre-existing green tests plus the 649-test suite are the lock. The SUMMARY discloses this as a deliberate resolution of the plan-checker advisory rather than burying it |

### TDD Substitution Judgment (Task 2)

Task 2 carried `tdd="true"` while its subject — the router and the chunked loop — shipped in Task 1's tracer. **The substitution is SOUND, and it is corroborated rather than merely recorded.**

- The plan itself made a literal RED phase impossible: Task 1 is a `tracer` whose `<done>` explicitly requires the over-budget path to be working and its block count asserted ("so a router that collapsed to a single write reddens HERE, in the tracer, rather than waiting for Task 2"). Writing Task 2's locks against already-shipped behaviour can only go green on first run.
- The replacement evidence is the stronger form for this situation: each test's own named reddening mutation, applied, with the red set recorded — which proves non-vacuity of every added lock, not just that a red-then-green sequence occurred.
- I independently reproduced **two of the four mutations (M-A, M-C)** without touching the source tree (pytest plugin patching `XrBackend` attributes at `pytest_configure`). Both red sets matched the SUMMARY's table exactly, including M-A's non-obvious caveat that the tracer's block-count assertion does NOT move under it. A narrated table would not predict that asymmetry correctly.
- "Source restored byte-identically" is corroborated: the three task commits contain no mutation residue, and `git status` over `quantlab/ tests/ example/` is clean at HEAD.

Residual (not a gap): M-B (inverted comparison) and M-D (cleanup no longer spanning the loop) were not independently re-run — M-B's direction is separately locked by the passing `test_the_budget_routes_in_both_directions` (both directions, both log levels), and M-D's subject is locked by the passing crash test, which M-A also reddens.

### Human Verification Required

None. Every truth is either a source fact confirmed by reading the shipped code or a behavioural claim exercised by a passing named test, and the two highest-risk behavioural claims (bounded multi-block writes, crash-safety across the loop) were additionally proved non-vacuous by reproduced mutations.

### Gaps Summary

None. The router routes, the switch is reported at `warning` with every figure the brief demanded, the chunked path is exercised over 2 blocks (`written == [512, 88]`, reported count `2`) on 600-timestamp stores so no single-iteration loop can satisfy the tests, `Factor.update()` reaches the bounded path with no call-site change, and all three of Task 3's documentation gates plus both file moves hold. The three findings above are informational and none of them touches goal achievement.

---

_Verified: 2026-09-08_
_Verifier: Claude (gsd-verifier)_
