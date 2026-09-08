---
phase: quick-260907-uac
verified: 2026-09-07T00:00:00Z
status: passed
score: 7/7 must-haves verified
covered_files:
  - ".planning/quick/260907-uac-refuse-an-append-that-overlaps-timestamp/260907-uac-PLAN.md"
  - ".planning/quick/260907-uac-refuse-an-append-that-overlaps-timestamp/260907-uac-SUMMARY.md"
  - ".planning/todos/completed/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md"
  - "quantlab/dataset/backend.py"
  - "tests/test_chunked_ingest.py"
  - "tests/test_symbol_axis_widening.py"
covered_digest: "v1:sha256:27a875b718d29c0ec70b082a8cf735b56d08803829eeeb6b4eaaec3d315089e0"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: null
  note: "Initial verification -- no prior VERIFICATION.md in this task directory."
---

# Quick 260907-uac: Refuse an Append That Overlaps Stored Timestamps — Verification Report

**Task Goal:** `XrBackend._assert_append_compatible` refuses an append whose `append_dim` coordinate overlaps timestamps already in the store, with a message naming the stored end, the incoming start and the alternative. `widen_and_append` inherits the refusal through its closing unmodified `append()` and has NO parallel check. Locked: gaps are NOT guarded; the refusal is unconditional with no overwrite escape hatch.

**Verified:** 2026-09-07
**Status:** passed
**Re-verification:** No — initial verification
**Method:** every claim below was re-derived by running the command myself against the working tree. SUMMARY.md was read for its claims and then treated as a hypothesis to falsify, not as evidence.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | APPD-01: `append()` raises `ValueError` before writing on an overlapping window; the store is unchanged | ✓ VERIFIED | Guard present at `quantlab/dataset/backend.py:398-434`, between the non-append coordinate loop and the dtype loop as specified. `test_append_refuses_a_window_overlapping_the_stored_timestamps` passes and is non-vacuous: it reddens under M1 (guard deleted) via `DID NOT RAISE ValueError`, run by me. Test asserts the store is bit-identical afterwards (`len==3`, `is_unique`, `is_monotonic_increasing`, `close` values equal to `before`). |
| 2 | APPD-02: message names store path, dimension, stored end, incoming start, and `save(mode="w")` | ✓ VERIFIED | Message body read at `backend.py:421-434`; all five elements present. Asserted in the tracer test (path, `timestamp`, `2022-01-06T00:00:00`, `2022-01-05T00:00:00`, `save(mode="w")`). Labels render via the new `_format_append_label` static helper (`backend.py:364-376`), ISO for `datetime64`, `str()` otherwise. |
| 3 | APPD-02b: the stated consequence is true of ALL THREE refused shapes | ✓ VERIFIED | **Re-measured by me** against the guard-disabled tree (see table below). The word **STRICTLY** survived into the shipped message and into the shared `_OVERLAP_CONSEQUENCE` constant (`tests/test_chunked_ingest.py:369-372`), asserted in all three refusal tests (lines 418, 505, 553) plus the smuggled-kwarg test (668). |
| 4 | APPD-03: a GAP still appends (D-01, unguarded) | ✓ VERIFIED | `test_append_allows_a_gap_between_the_stored_end_and_the_incoming_start` passes and is non-vacuous: it reddens under M5 (guard extended to refuse gaps), run by me. |
| 5 | APPD-04: no opt-out, DECLARED or SMUGGLED | ✓ VERIFIED | Live `inspect.signature(XrBackend.append)` → `('self','path','append_dim','kwargs')`. The **M6 split was reproduced by me** — see the Mutation table. Both halves of the lock exist and cross-reference each other in their docstrings. |
| 6 | APPD-05: `widen_and_append` inherits the refusal verbatim and holds NO parallel check | ✓ VERIFIED | The full `git diff ab3049e..HEAD -- quantlab/dataset/backend.py` shows `widen_and_append` gained **only** a docstring paragraph; its body is unchanged and contains zero timestamp/overlap logic (body printed via `inspect.getsource`, verified). Inheritance is not merely asserted: the widen test reddens under M1 (the shared guard deleted) and under M4 (a parallel check added), both run by me. I also confirmed independently that the widen genuinely commits before the inherited refusal fires — symbol axis grew to `['A','B','C']` while the timestamp axis stayed at the original 3 labels. |
| 7 | APPD-06: chunked ingest path unaffected; suite green | ✓ VERIFIED | `uv run pytest tests/ -q` → **545 passed**, zero failures, on a clean tree, run by me twice (before and after the mutation sweep). Baseline 537 + 8 new. |

**Score:** 7/7 truths verified (0 present, behavior-unverified)

### APPD-02b Re-measurement (the load-bearing wording check)

Measured by me on 2026-09-07 by disabling the new comparison (`if False:`), performing each append for real, and reading the resulting axis:

| incoming window vs. store | resulting labels | `is_unique` | `is_monotonic_increasing` | strictly increasing |
|---|---|---|---|---|
| partial overlap (`01-05..01-07` into `01-04..01-06`) | `[01-04, 01-05, 01-06, 01-05, 01-06, 01-07]` | False | False | **False** |
| starts exactly on stored end (`01-06..01-08`) | `[01-04, 01-05, 01-06, 01-06, 01-07, 01-08]` | False | **True** | **False** |
| ends before stored start (`01-04..01-05` into `06-01..06-02`) | `[06-01, 06-02, 01-04, 01-05]` | **True** | False | **False** |

Identical to the planning-time measurement in every cell, including both non-obvious rows. The shipped clause — `no longer STRICTLY increasing -- duplicate labels, out-of-order labels, or both` — is true of each row: row 2 is the duplicate-only case, row 3 is the out-of-order-only case, row 1 is "both". A clause naming only duplicates is falsified by row 3; one naming only disorder is falsified by row 2. The disjunction is the honest wording and it survived into the code verbatim.

### Mutation Re-derivation (run by me, not read from SUMMARY)

Each mutation was applied to a clean tree, the command run, then reverted with `git checkout --`. `git diff --quiet quantlab/dataset/backend.py` was confirmed clean after each.

| Mut | Change | Executor claimed | I observed | Match |
|-----|--------|------------------|------------|-------|
| M1 | disable the new comparison | overlap, ends-before, starts-exactly-on, widen-inherits + kwarg test redden | `5 failed, 42 passed` — exactly those five | ✓ |
| M2 | relax `<=` to `<` | **exactly one** reddening | `1 failed, 544 passed` — only `test_append_refuses_a_window_starting_exactly_on_the_stored_end` | ✓ |
| M3 | incoming `.max()` instead of `.min()` | overlap + starts-exactly-on + kwarg + widen; **ends-before stays GREEN** | `4 failed, 541 passed` — exactly those four; the ends-before test stayed green | ✓ |
| M4 | second, differently-worded check atop `widen_and_append` | verbatim-message test reddens on the equality assertion | `1 failed, 10 passed`; failure is `assert 'XrBackend.ap...' == 'M4 MUTATION:...'` at `test_symbol_axis_widening.py:464` | ✓ |
| M5 | extend the guard to refuse a gap | gap test **plus 21 others = 22** | `22 failed, 523 passed`; the gap lock is among them, the other 21 are the chunked-ingest / widening suites | ✓ **count re-derived exactly** |
| M6 | `force` popped from `**kwargs` INSIDE the body, ahead of the guard call | behavioural test RED, signature test GREEN | `1 failed, 1 passed` — `..._carrying_an_unrecognised_kwarg` failed on `DID NOT RAISE ValueError` at `test_chunked_ingest.py:661`; `test_append_offers_no_overwrite_escape_hatch` **passed**. Independently confirmed `inspect.signature` returned a byte-identical `('self','path','append_dim','kwargs')` with the hatch in | ✓ **the split is real** |
| M7 (mine, beyond the plan) | drop the coordinate-presence skip (`if True:`) | not claimed | `1 failed, 544 passed` — only `test_append_skips_the_overlap_check_without_an_append_dim_coordinate` | ✓ that lock is non-vacuous too |

**On M6 specifically (scrutiny item 1):** the split reproduced exactly. Neither "both redden" nor "both stay green" occurred. The structural proxy demonstrably does not span D-02; the pair does. This is the strongest single piece of evidence in the task and it holds up.

**On M5 (scrutiny item 4):** the 22 count is correct in both directions — not 21, not 23. The mechanism is real: `TimeChunkPlanner.plan_from_timestamps` is documented at `quantlab/base/chunking.py:100` as producing "Windows whose edges are OBSERVED timestamps", so ordinary market data (weekends, holidays) makes gapped appends the normal case. Guarding gaps would break the production ingest path, not tighten it. D-01 is load-bearing, not stylistic — the executor's stronger-than-planned finding is confirmed.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `quantlab/dataset/backend.py` | one new check + `_format_append_label` + two docstring edits | ✓ VERIFIED | Diff is exactly that and nothing else. `append`'s docstring gained the third enforced property and the explicit "A GAP is NOT an error" paragraph; `widen_and_append`'s gained one prose paragraph, no control flow. |
| `tests/test_chunked_ingest.py` | 7 new tests + shared consequence constant | ✓ VERIFIED | 7 new `def test_...` functions confirmed in the diff, plus `import inspect` and `_OVERLAP_CONSEQUENCE`. |
| `tests/test_symbol_axis_widening.py` | 1 new inheritance test | ✓ VERIFIED | `test_widen_and_append_inherits_the_overlap_refusal_verbatim`, +66 lines. |
| `.planning/todos/completed/2026-09-07-guard-...md` | moved, `status: closed`, `## Closed` section | ✓ VERIFIED | File present under `completed/`, absent from `pending/`, frontmatter `status: closed`, `## Closed 2026-09-07` section present recording what shipped, all six mutation results and the decided points. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `XrBackend.append` | `_assert_append_compatible(path, append_dim)` | single call site, `backend.py:91` | ✓ WIRED | The guard call sits ahead of `kwargs.pop("encoding", None)` and ahead of every `to_zarr`, which is the mechanism the smuggled-kwarg test depends on. |
| `XrBackend.widen_and_append` | the UNCHANGED `XrBackend.append` | closing `return self.append(path, append_dim, **kwargs)` | ✓ WIRED | Body verified unchanged; no parallel check. M1 and M4 both prove the link is load-bearing rather than decorative. |
| `BaseDataset.from_raw_data_chunked` | `data_backend.append(..., append_dim=...)` | production caller | ✓ WIRED | Unaffected: full suite green, and the chunked-ingest tests are exactly the ones M5 shows depend on gaps being permitted. |

### Behavioural Spot-Checks

| Behaviour | Command | Result | Status |
|-----------|---------|--------|--------|
| Full suite green on clean tree | `uv run pytest tests/ -q` | `545 passed` in 26.2s, zero failures | ✓ PASS |
| No declared escape hatch | `inspect.signature(XrBackend.append)` | `('self','path','append_dim','kwargs')` | ✓ PASS |
| Three refused shapes measured on unguarded tree | ad-hoc script against `if False:` guard | matches planning-time table cell for cell | ✓ PASS |
| Widen genuinely commits before the inherited refusal | ad-hoc `widen_and_append` with added symbol C | `ValueError` raised; symbol axis `['A','B','C']`, timestamp axis still the original 3 | ✓ PASS |
| Prose gate (no new pre-migration paths) | plan's `diff <(grep -c ...)` gate | counts `1 / 0 / 0`, unchanged | ✓ PASS |

### Git Hygiene

| Check | Result | Status |
|-------|--------|--------|
| `test.py`'s pre-existing modification still uncommitted | `git status --porcelain` → ` M test.py`; `git diff --stat test.py` → 3 insertions, 37 deletions | ✓ PASS |
| `test.py` untouched by this task | `git diff ab3049e..HEAD -- test.py` → empty | ✓ PASS |
| All mutations reverted | `git diff --quiet quantlab/dataset/backend.py` → clean | ✓ PASS |
| Todo moved to `completed/` | present in `completed/`, absent from `pending/`, `status: closed` | ✓ PASS |
| Three commits resolve | `406b557`, `230eb73`, `8249ded` all in `git log` | ✓ PASS |

**Deviation assessed — Task 3's `.planning/todos/` directory pathspec (scrutiny item 5): SOUND.** `git show --stat 8249ded` lists exactly two files, both sides of the todo rename. A pathspec rooted at `.planning/todos/` cannot reach `test.py`, which sits at the repository root — the containment is structural, not incidental. The other three files in `.planning/todos/pending/` were not swept in, so the pathspec did not over-collect either. The stated reason is also correct: `git mv` stages both sides of a rename, so `git add` on the now-absent pending path would fail with `did not match any files`. The narrower alternative (`git add -A .planning/todos/`) buys nothing here.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | none | — | No `TODO`/`TBD`/`FIXME`/`XXX`/`HACK`/`PLACEHOLDER` markers in any of the three modified source files. No `skip`/`xfail` in the new tests (the only `skip` hits are prose about the guard's coordinate-presence skip). |

### Human Verification Required

None. Every truth is behaviour-dependent and every one of them was exercised by a test I ran, with a mutation confirming the test is non-vacuous. Nothing was accepted on symbol presence alone.

### Gaps Summary

None. Seven of seven must-haves verified. The three claims flagged for special scrutiny all survived independent re-derivation:

1. **M6's split is real** — behavioural test red on `DID NOT RAISE ValueError`, signature test green with a byte-identical parameter tuple. The pair is genuinely necessary; neither half alone spans D-02.
2. **The consequence clause is honest** — "STRICTLY" survived into both the shipped message and the shared test constant, and re-measuring the two non-obvious shapes reproduced the exact `is_unique` / `is_monotonic_increasing` pattern that makes strictly-increasing the only defensible wording.
3. **`widen_and_append` gained no parallel guard** — docstring only, body byte-unchanged, and the inheritance is proved rather than asserted: it breaks under M1 (shared guard removed) and under M4 (parallel check added), and the widen is demonstrably non-vacuous (symbol axis grows to A,B,C before the inherited refusal fires).

The M5 finding is confirmed at exactly 22 failures, and the deviation in Task 3's commit pathspec is sound and could not have reached `test.py`.

---

_Verified: 2026-09-07_
_Verifier: Claude (gsd-verifier)_
