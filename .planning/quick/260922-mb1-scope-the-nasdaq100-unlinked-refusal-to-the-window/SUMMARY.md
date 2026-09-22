---
phase: quick-260922-mb1
plan: 01
subsystem: dataset/crsp
status: complete
tags: [crsp, membership, nasdaq100, survivorship-bias, wrds]

requires:
  - CrspMembership._nasdaq100_pieces (the whole-history refusal)
  - CrspReference (offline parquet tier)
provides:
  - "permno_intervals(..., window=) — the refusal scoped to a requested window"
  - "report['unlinked_blocking'] — the subset of unlinked spells that would refuse"
affects:
  - quantlab/dataset/constituent.py
  - quantlab/dataset/crsp/__init__.py
  - scripts/ingest_wrds_crsp.py (unmodified; unblocked)

tech-stack:
  added: []
  patterns:
    - "A widening keyword with a None default: every existing caller keeps today's behaviour, asserted rather than intended."
    - "One dict object appended to two report lists, so the two keys cannot drift apart."

key-files:
  created: []
  modified:
    - quantlab/dataset/crsp/membership.py
    - quantlab/dataset/constituent.py
    - quantlab/dataset/crsp/__init__.py
    - tests/test_crsp_membership.py
    - tests/test_crsp_constituent.py
    - example/wrds_crsp.md

decisions:
  - "The window scopes the REFUSAL only — never the returned intervals, never the report, never the log."
  - "An inverted window is refused outright rather than treated as an empty window: unchecked, it overlaps nothing and would suppress every refusal."
  - "`_member_intervals` still passes no `allow_unlinked`. Deliberately out of scope; the conservative direction is to keep refusing an in-window gap there."
  - "The two constituent `_build_intervals` bodies both pass the window, staying byte-identical apart from one explanatory comment."

metrics:
  duration: ~50min
  completed: 2026-09-22

actuals:
  tokens: 9000
  tasks: 3
  commits: 4
plan_head_before: 23e51de4b71dbaa8801e8955e3023f7dcc876f6e
---

# Quick 260922-mb1: Scope the Nasdaq-100 Unlinked Refusal to the Window — Summary

`CrspMembership` validated CRSP/Compustat link integrity across the entire Nasdaq-100 index
history however short the requested window; the requested window is now threaded to the refusal
at all three consumer sites, so an uncovered span refuses only when it overlaps the window that
could actually lose a member to it.

## Why this existed: a gate that always fires is a gate that never fires

The operator hit the refusal on a live WRDS pull for `2015-01-01..2025-12-31`. All six offending
spells END before 2015 and could not contribute a PERMNO to that roster, yet all six blocked the
run. The documented escape hatch, `--allow-unlinked-ndx`, IS genuinely lossy on a window that
overlaps a gap — so an alarm firing on every pull trains the operator to pass the flag reflexively,
and it will not be noticed on the run where it actually drops index members. Scoping the refusal is
what keeps it meaningful where it is real.

## What Was Built

**The refusal takes a window.** `permno_intervals` and `_nasdaq100_pieces` gained a keyword-only
`window`. In the per-spell loop, an uncovered span blocks when `window is None` or when any
`(gap_start, gap_end)` satisfies `gap_start <= window[1] and gap_end >= window[0]`. The
comparison runs on the `date` tuples, before the report entry stringifies them.

**The record is untouched.** The same dict object is appended to `unlinked` and, when it blocks,
to the new `blocking`. `report['unlinked']` stays complete; `report['unlinked_blocking']` names
only the entries that would refuse. Appending one object to two lists is what keeps the two keys
from drifting.

**The warning has two shapes that do not read alike.** With something blocking it keeps today's
sentence and appends the window and the in-window count. With nothing blocking it says plainly
that none of the uncovered days fall inside the requested window — and that branch now fires on
runs that did NOT pass `allow_unlinked`, which is the point: the out-of-window fact is scoped out
of the refusal, never out of the log.

**All three consumers pass their own window.** `permnos_in_range` hands down the two dates it had
already computed and already validated for inversion; `CompustatNasdaq100ConstituentDataset` and
`CrspSP500ConstituentDataset._build_intervals` pass `(config.start_date, config.end_date)`; and
`CrspStockDataset._member_intervals` — the security filter's roster exemption, and the exact frame
in the operator's traceback — passes the conversion's own window.

**An inverted window is refused.** It overlaps nothing, so left unchecked it would suppress every
refusal and return a roster indistinguishable from a complete one.

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | The window contract as seven tests | `3215f7b` | `tests/test_crsp_membership.py` |
| 2 | Scope the refusal inside `CrspMembership` | `15a27f2` | `quantlab/dataset/crsp/membership.py`, `tests/test_crsp_membership.py` |
| 3 | The other two consumers + docs | `29b916f` | `constituent.py`, `crsp/__init__.py`, both test files, `example/wrds_crsp.md` |
| 2 (follow-up) | Warning stops disagreeing with its own count | `b22ad76` | `quantlab/dataset/crsp/membership.py` |

## Verification (executor, in the isolated worktree)

| Check | Result |
|---|---|
| `tests/test_crsp_membership.py` | 29 passed (21 existing + 8 new) |
| `tests/test_crsp_constituent.py` | 15 passed (14 existing + 1 new) |
| `tests/test_crsp_{membership,constituent,identity,rebuild}.py` + `test_ingest_wrds_crsp.py` | 134 passed |
| `grep "window=(self.config.start_date, self.config.end_date)"` | 3 lines — 2 in `constituent.py`, 1 in `crsp/__init__.py` |
| Full suite (3 ignores, per plan) | 55 failed, 1704 passed, 1 skipped |
| `diff baseline.txt after.txt` | **empty, exit 0** — the failing node-ID set is identical in both directions |

The 55 failures are the pre-existing `D-03.11-12-A` set, present before and after. The gate is the
set difference, not a green suite.

### RED evidence

Task 1 — six of seven new tests red on pre-fix code: two by
`TypeError: CrspMembership.permno_intervals() got an unexpected keyword argument 'window'`, four by
the whole-history refusal firing where the window cannot lose a member. The seventh is discussed
under Deviations.

Task 3 — both consumer tests red with the fix from Task 2 already in place, each failing inside
`_nasdaq100_pieces` with `allow_unlinked = False, window = None`:

    quantlab/dataset/crsp/__init__.py:773: in _member_intervals
        ).permno_intervals(self.config.roster_universe)
    quantlab/dataset/crsp/membership.py:235: in permno_intervals
        pieces = self._nasdaq100_pieces(
    allow_unlinked = False, window = None

    quantlab/dataset/constituent.py:219: in _build_intervals
        CrspMembership(CrspReference(self.config.cache_dir)).permno_intervals(
    quantlab/dataset/crsp/membership.py:235: in permno_intervals
        pieces = self._nasdaq100_pieces(
    allow_unlinked = False, window = None

That frame is the operator's own traceback, which is why Task 3 — not Task 2 — is the half that
unblocks them. A fix that stopped at `permnos_in_range` would have left the failure in place.

### The operator-facing text, as it now reads

Window clear of the gap, no `allow_unlinked`, no raise:

    CrspMembership: 1 Nasdaq-100 membership spell(s) have days no CRSP/Compustat link covers, but
    NONE of those days fall inside the requested window 2015-01-01..2025-12-31, so the universe over
    that window is complete. They are recorded in report['unlinked'] for inspection.

Window covering the gap, refused:

    CrspMembership: 1 Nasdaq-100 membership spell(s) have days that no CRSP/Compustat link covers, so
    those membership days have no PERMNO:
      gvkey=100020 iid=01 from=1999-01-13 thru=2007-02-05 uncovered=[2007-02-01..2007-02-05]
    The refusal is scoped to the requested window 2007-01-01..2007-12-31: only spells with uncovered
    days inside it are listed above, and only they refuse. Spells whose gaps fall entirely outside it
    are recorded in report['unlinked'] instead.
    Dropping them would remove real index members from the universe -- survivorship bias that reads
    downstream as a data gap rather than an error. Pass allow_unlinked=True (the CLI's
    --allow-unlinked-ndx) to proceed with the linked days and read the rest from report['unlinked'].

## Deviations from Plan

**1. [Rule 1 - Plan expectation wrong] Test 1 of Task 1 is green before AND after, not red**

- **Found during:** Task 1 RED run.
- **Issue:** The plan states all seven new tests are red on today's code. Test 1,
  `test_an_uncovered_gap_inside_the_window_still_refuses`, cannot be: it asserts that an in-window
  gap keeps refusing, which is exactly today's behaviour. A guard on behaviour that must not change
  is green from the start by construction — the plan's own action text calls it "the guard the whole
  change must not weaken", which says the same thing.
- **Fix:** None needed in code. The test was kept exactly as specified and is the T-mb1-01
  mitigation; the RED count is six, not seven, and is recorded as such in the Task 1 commit message.
- **Commit:** `3215f7b`

**2. [Rule 1 - Bug in new test] The inverted-window assertion was too coarse**

- **Found during:** Task 2 GREEN run — 1 failed, 27 passed.
- **Issue:** The test asserted `"link" not in message.lower()`. The refusal message legitimately
  contains "unlinked refusal", naming the thing an inverted window would suppress, and "unlinked"
  contains "link".
- **Fix:** Tightened to the plan's own stated intent — the message must not mention a
  *CRSP/Compustat link*, i.e. must not send the reader to the vendor data for a typo in their own
  arguments. Assertion is now `"CRSP/Compustat" not in message`, with a comment saying why it is
  not a bare "link" search. Still red on pre-fix code (`pytest.raises(ValueError)` vs. the
  `TypeError` that a `window=` keyword raised there).
- **Files modified:** `tests/test_crsp_membership.py`
- **Commit:** `15a27f2`

**3. [Rule 2 - Operator-facing correctness] The in-window warning disagreed with its own count**

- **Found during:** post-Task-3 manual check of the two warning shapes (a success criterion with no
  test assertion behind it).
- **Issue:** The added sentence read `1 of them have uncovered days INSIDE the requested window`.
  A subject-verb disagreement in the one sentence this whole change exists to produce is the kind
  of wrong that makes an operator doubt the count itself.
- **Fix:** Reworded without a verb that has to agree with the number, and it now shows the
  denominator: `Uncovered days INSIDE the requested window 2007-01-01..2007-12-31: 1 of the 1.`
- **Files modified:** `quantlab/dataset/crsp/membership.py`
- **Commit:** `b22ad76`

## Measurement discrepancy — raised by the executor, RESOLVED by the orchestrator

The executor reported a pass count it could not attribute: the orchestrator's stated baseline was
**1696 passed**, while its own after-run of 1704 minus its 9 added tests implied a baseline of
**1695**. It declined to smooth the difference over, which was right.

The cause is `tests/test_entry_point_contracts.py:44`,
`ENTRY_POINTS = sorted(REPO_ROOT.glob("*.py"))` — the suite parameterizes over the **working tree**,
not the git index — and the file is `jerry_query_data.py`, which is **gitignored**, not merely
untracked. It sits in the main tree and is invisible to `git status --porcelain`, but `glob` sees it,
so the main tree collects 8 entry-point tests where a fresh worktree collects 7. This is the
already-filed `D-03.11-18-A`, whose original observation was the same +1 from the same glob (recorded
there as an untracked file; it is in fact ignored, which is why a `git status` check does not surface
it).

Confirmed by arithmetic on both sides after the merge: main 1696 -> 1705, worktree 1695 -> 1704,
each exactly +9 for the 9 added tests, with main consistently one higher.

Nothing was wrong with either measurement, and the binding gate was never affected: the failing
node-ID set difference was computed by each party against its own same-tree baseline with the same
command, and was empty in both directions on both sides.

## Known Stubs

None.

## Out of Scope (carried forward, unchanged)

`CrspStockDataset._member_intervals` still passes no `allow_unlinked`, because `CrspDatasetConfig`
has no field to carry the flag. An in-window gap therefore refuses a conversion with no CLI escape
hatch. This is the conservative direction, was explicitly excluded by the plan, and is now
documented in that method's docstring rather than left to be rediscovered.

## Post-merge re-verification (orchestrator, MAIN TREE)

The executor's green was taken in an isolated worktree, which this project has been burned by before
(`project_gsd_worktree_verify_path`). Re-measured independently on `main` after the merge, bytecode
caches cleared first:

- `tests/test_crsp_{membership,constituent,identity}.py` + `test_ingest_wrds_crsp.py`: 115 passed.
- Full suite (3 ignores): **55 failed, 1705 passed, 1 skipped** against a pre-mb1 main baseline of
  55 / 1696 / 1. The failing node-ID set difference is **empty in both directions**; the +9 is
  exactly the 9 tests this task added.
