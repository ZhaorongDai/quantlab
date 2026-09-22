---
phase: quick-260922-mb1
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/dataset/crsp/membership.py
  - quantlab/dataset/constituent.py
  - quantlab/dataset/crsp/__init__.py
  - tests/test_crsp_membership.py
  - tests/test_crsp_constituent.py
  - example/wrds_crsp.md
autonomous: true
requirements: [QUICK-260922-mb1]

estimate:
  tokens: 80000
  raw_tokens: 40000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "A Nasdaq-100 pull whose requested window contains no uncovered membership day no longer refuses: over a tier whose only link gap is in 2007, `permnos_in_range(NASDAQ100, '2015-01-01', '2025-12-31')` returns the roster instead of raising."
    - "An uncovered day INSIDE the requested window still refuses by default, including when the gap overlaps the window only partially or touches exactly one of its edges. The survivorship-bias guard is unchanged wherever it is real."
    - "`permno_intervals` called WITHOUT a window behaves exactly as today: any uncovered day refuses. That is the compatibility hinge and it is asserted, not merely intended."
    - "A window never filters the intervals a call returns; it scopes the refusal and nothing else."
    - "Out-of-window uncovered spans are never hidden: they stay in `report['unlinked']` and emit a warning whose text says plainly that none of them fall inside the requested window, so 'gaps, none in your window' and 'gaps, all in your window' do not read identically."
    - "All THREE window-bounded consumers of the refusal pass their own window down, so the operator's reported command reaches the end instead of failing at the next consumer with the same message."
  artifacts:
    - quantlab/dataset/crsp/membership.py
    - quantlab/dataset/constituent.py
    - quantlab/dataset/crsp/__init__.py
    - tests/test_crsp_membership.py
    - tests/test_crsp_constituent.py
    - example/wrds_crsp.md
  key_links:
    - "`permnos_in_range` -> `permno_intervals(window=)` -> `_nasdaq100_pieces(window=)` -> the per-spell `uncovered` DATE tuples -> the refusal"
    - "`CompustatNasdaq100ConstituentDataset._build_intervals` -> `permno_intervals(window=(config.start_date, config.end_date))`"
    - "`CrspStockDataset._member_intervals` -> `permno_intervals(window=(config.start_date, config.end_date))`"
---

<objective>
`CrspMembership` validates CRSP/Compustat link integrity across the ENTIRE Nasdaq-100 index
history (1995 onward) however short the requested window. `permnos_in_range`
(`membership.py:247-250`) builds the full-history intervals first and applies the window only
afterwards, so the refusal at the end of `_nasdaq100_pieces` (`membership.py:424-425`) has
already fired. On live WRDS data the operator's

    uv run python scripts/ingest_wrds_crsp.py --universe comp_nasdaq100 \
        --start-date 2015-01-01 --end-date 2025-12-31 --to-zarr

refuses with six unlinked spells whose uncovered days all lie in 1999-2008 — every one of
them ends before 2015-01-01, so not one could contribute a PERMNO to that roster. The refusal
message argues survivorship bias, which is simply untrue for a window those members never
touch.

The cost is not the annoyance. The documented escape hatch `--allow-unlinked-ndx` IS genuinely
lossy for a window that DOES overlap a gap, and an alarm that fires on every single pull trains
the operator to pass the flag reflexively — so it will not be noticed on the run where it drops
real index members. A gate that always fires is not a gate.

Purpose: thread the requested window down to the refusal so it fires only for uncovered spans
that overlap the window, leaving the returned intervals, the report and the default
(window-less) behaviour untouched.
Output: a `window` keyword on `permno_intervals` / `_nasdaq100_pieces`, passed by all three
window-bounded consumers; a `report['unlinked_blocking']` subset; a warning that distinguishes
in-window from out-of-window; a window-scoped refusal message; nine tests; and the doc bullet
in `example/wrds_crsp.md` that currently names this exact behaviour as a known follow-up.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@quantlab/dataset/crsp/membership.py
@tests/test_crsp_membership.py
@quantlab/dataset/constituent.py
@tests/test_crsp_constituent.py

Read-first line numbers (as of this plan):

- `quantlab/dataset/crsp/membership.py`: `_as_date` :80, `_uncovered` :108,
  `_reset_report` :176-186, `permno_intervals` :190-213, `permnos_in_range` :215-254,
  `LINK_GAP_TOLERANCE_DAYS` :168, `_nasdaq100_pieces` :309-434 (the per-spell `uncovered`
  loop :392-422, the refusal :424-425, the warning :426-433), `_unlinked_message` :505-532.
- `quantlab/dataset/constituent.py`: `CrspSP500ConstituentDataset._build_intervals` :170-178,
  `CompustatNasdaq100ConstituentDataset._build_intervals` :217-226. The two bodies are
  byte-identical today.
- `quantlab/dataset/crsp/__init__.py`: `_member_intervals` :745-778 (the call :773).
- `tests/test_crsp_membership.py`: `_spell` :73, `_link` :78, `_membership` :90, `_ndx_tier`
  :108, and the existing D-14 section banner at :308.
- `tests/test_crsp_constituent.py`: `_ndx_spell` :94, `_link` :99, `_ndx_reference` :170,
  `_panel_config` :571, `test_an_unlinked_nasdaq100_spell_stops_the_panel_unless_allow_unlinked`
  :716.

**THREE consumers refuse, not one.** The designed fix (window on `permnos_in_range`) closes
only the first of them; verified by reading the call graph, the operator's exact command would
then fail at the second or third with the identical message:

1. `scripts/ingest_wrds_crsp.py:410` -> `permnos_in_range(...)` — the roster. Fixed by Task 2.
2. `scripts/ingest_wrds_crsp.py:501` sets `roster_universe=args.universe` on the conversion
   config; `CrspStockDataset._apply_security_filter` (:989) -> `_roster_exemption` ->
   `_member_intervals` (:773) calls `permno_intervals(roster_universe)` with NO window AND no
   `allow_unlinked` — so on live data this one refuses even WITH `--allow-unlinked-ndx`. Task 3.
3. `scripts/ingest_wrds_crsp.py:571` builds the membership panel;
   `CompustatNasdaq100ConstituentDataset._build_intervals` calls `permno_intervals` with no
   window. Task 3.

Consumers 2 and 3 are safe to scope for exactly the same reason as 1: the CRSP derivation is
computed over `[config.start_date, config.end_date]` (module docstring, `crsp/__init__.py:11`)
and the constituent panel's own edges are `max(config.start_date, coverage_start)` ..
`min(config.end_date, horizon)` (`base/constituent.py:153-168`), both subsets of the window
being passed. A gap outside that window cannot touch a single cell either of them produces.

Missing `allow_unlinked` at consumer 2 is a SEPARATE defect (no `CrspDatasetConfig` field
exists to carry the flag) and is deliberately OUT OF SCOPE here — do not add a config field.
After this plan an in-window gap still refuses there with no escape hatch, which is the
conservative direction.

Project constraints that bind here:
- `uv` for everything. Package manager, test runner, interpreter.
- No backward-compatibility obligation (D-32 lineage): change signatures directly, no shim.
  This one happens to be a widening anyway — every existing caller that omits `window` keeps
  today's behaviour exactly.
- `quantlab/` docstrings and comments in this area are English; `tests/` is English;
  `example/wrds_crsp.md` is Chinese — match each file's existing language.
- macOS: `tests/conftest.py` already sets `OMP_NUM_THREADS=1` before any import. Do not add
  anything to conftest.
- Clear bytecode caches before any verifying run (this repo was burned by stale bytecode,
  G-03.11-4): `find . -name '__pycache__' -type d -prune -exec rm -rf {} +`.
- EVERY command below is relative to the repo root. Never rewrite one to an absolute
  `/Users/daizhaorong/...` path: execution happens in an isolated worktree and an absolute
  path would exercise unchanged main-tree code and report a false green.
- Never `git stash` to compare arms; capture the baseline list first instead (Task 1).
- `.planning/phases/**` is DO NOT TOUCH.

**Concurrency.** Quick task `260922-lu2` is executing in a separate worktree right now. Do not
touch, move or "fix imports in" any of its movers: `quantlab/dataset/backend.py`,
`quantlab/acquisition/registry.py`, `quantlab/acquisition/universe.py`,
`quantlab/dataset/cleaning.py`, `quantlab/dataset/masking.py`,
`quantlab/dataset/session_calendar.py`, `quantlab/acquisition/sql_volume.py`,
`quantlab/acquisition/inspector.py`. An import that looks stale is that task's in-flight work.

That task ALSO rewrites path references in three files this plan edits —
`quantlab/dataset/crsp/membership.py` (its module docstring, :41), `quantlab/dataset/constituent.py`
and `example/wrds_crsp.md` — so whoever merges second may see a textual conflict. The hunks are
far apart (a docstring path near the top of the file vs. the refusal logic from :309 down), and
the resolution is always KEEP BOTH: their renamed path, our window logic. Never resolve by
taking one side wholesale. `quantlab/dataset/crsp/__init__.py`, `tests/test_crsp_membership.py`
and `tests/test_crsp_constituent.py` are not in that task's file list at all.

**Suite gate.** The baseline is 55 failed / 1696 passed / 1 skipped; the 55 are pre-existing
(`D-03.11-12-A`). The gate is therefore the failing node-ID SET DIFFERENCE being empty in BOTH
directions, never "green". `tests/test_cross_sectional_zscore.py` is ignored on both sides
because of `D-03.11-12-B`, an intermittent KunQuant destructor deadlock that hung a run for 69
minutes today.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: capture the baseline, then the window contract as seven RED tests</name>
  <files>tests/test_crsp_membership.py</files>
  <behavior>
    Seven new tests in `tests/test_crsp_membership.py`, all asserting the FIXED behaviour, so
    all seven are red on today's code (five by raising the unlinked refusal where they expect
    none or a different error, two by `TypeError: unexpected keyword argument 'window'`):

    1. An uncovered gap INSIDE the requested window still raises. Write this one first; it is
       the guard the whole change must not weaken.
    2. An uncovered gap entirely OUTSIDE the window does not raise, and the PERMNOs that do
       overlap the window come back intact.
    3. Gaps that PARTIALLY overlap, or touch either edge of, the window raise; a gap one day
       clear of the edge does not.
    4. `permno_intervals` with no window (and with an explicit `window=None`) still raises on
       any gap — the unchanged-behaviour hinge.
    5. A window scopes the refusal and never filters the returned intervals.
    6. `report['unlinked']` is complete whether or not the call raised, and
       `report['unlinked_blocking']` names only the entries that would refuse.
    7. An inverted window raises its own error rather than silently blocking nothing.
  </behavior>
  <action>
FIRST, before creating or editing anything, capture the baseline failing set — the gate in
Task 3 needs a before-list and this repo forbids `git stash` for A/B arms. Run the two commands
in the first `<verify>` block below and keep `/tmp/gsd-mb1/baseline.txt`. It is scratch: never
add it to a commit. Expect 55 node IDs.

Then extend `tests/test_crsp_membership.py`. Do not create a new test module, do not add new
fixtures, and do not edit `tests/crsp_fixtures.py`. Reuse the module's own `_spell`, `_link` and
`_ndx_tier` helpers and its convention that every quantlab import lives INSIDE a test.

Add a section banner in the file's existing style, after the last D-14 test and before the
`03.11-05` section: `# 260922-mb1 -- the refusal is scoped to the requested window`, with two or
three lines of prose stating the rule: the refusal fires for an uncovered span only if that span
overlaps the requested window, because a span the window never touches cannot cost the window a
member; with no window every span blocks, which is what keeps every existing caller unchanged.

Add ONE module-level helper next to the other builders, `_gap_tier(tmp_path)`, returning
`_ndx_tier` over two synthetic spells and their links, and document in its docstring that it
reproduces the live shape the operator hit (gvkey 012884: spell 1999-01-13..2007-02-05, links
covering through 2007-01-31, five uncovered days) beside one modern member that is fully linked:

  - spell gvkey `100020` iid `01` from `1999-01-13` thru `2007-02-05`; link `100020`/`01` ->
    `81020.0` from `1999-01-01` to `2007-01-31`. Uncovered `2007-02-01..2007-02-05`, five
    days, which is one more than `LINK_GAP_TOLERANCE_DAYS`, so it is unlinked and not a
    tolerated seam. Mark it `# SYNTHETIC` and name the live gvkey it mirrors in the comment.
  - spell gvkey `100021` iid `01` from `2015-01-02` thru `None`; link `100021`/`01` ->
    `81021.0` from `2010-01-01` to `None`. Fully covered, clipped to the product end.

Test 1, `test_an_uncovered_gap_inside_the_window_still_refuses`: over `_gap_tier`, assert
`permnos_in_range(NASDAQ100, "2007-01-01", "2007-12-31")` raises `ValueError` whose message
contains `100020`, `2007-02-01` and `allow_unlinked`. Docstring: this is the survivorship-bias
guard where it is REAL — a 2007 window loses a genuine member if those days are dropped — and
scoping the refusal must not weaken it.

Test 2, `test_an_uncovered_gap_outside_the_window_no_longer_refuses`: assert
`permnos_in_range(NASDAQ100, "2015-01-01", "2025-12-31")` returns exactly `["81021"]`. Assert
in the same test that 81020 is legitimately absent BECAUSE its membership ended in 2007, not
because anything was dropped: `permno_intervals(NASDAQ100, allow_unlinked=True)` over the same
tier lists an 81020 interval ending `2007-01-31`. That pairing is what separates "the window
does not need it" from "the fix silently lost it".

Test 3, `test_a_gap_that_touches_the_window_edge_refuses_and_one_day_clear_does_not`: four
windows over `_gap_tier`, each named in the test body with the edge it probes. Raises:
`("2007-02-03", "2007-12-31")` (gap starts before the window, ends inside);
`("2006-01-01", "2007-02-01")` (window end equals gap start — the `<=` edge);
`("2007-02-05", "2030-01-01")` (window start equals gap end — the `>=` edge). Does not raise:
`("2007-02-06", "2007-12-31")`, one day clear of the gap.

Test 4, `test_permno_intervals_without_a_window_still_refuses_on_any_gap`: over `_gap_tier`,
both `permno_intervals(NASDAQ100)` and `permno_intervals(NASDAQ100, window=None)` raise, with
`100020` in each message. Docstring: this is the compatibility hinge — every existing caller
omits `window`, so this test is the one that says the widening did not move anyone's floor.

Test 5, `test_a_window_scopes_the_refusal_and_never_filters_the_intervals`: call
`permno_intervals(NASDAQ100, window=("2015-01-01", "2025-12-31"))` over `_gap_tier` and assert
the frame carries BOTH PERMNOs — `{81020, 81021}` as a set of the `permno` column — with 81020's
interval still `1999-01-13..2007-01-31`, entirely outside the window it was given. A window is
not a filter; `permnos_in_range` is where filtering happens, afterwards, and it already has a
test.

Test 6, `test_the_report_lists_every_unlinked_spell_and_names_only_the_blocking_ones`: three
arms over `_gap_tier`, each asserting on a fresh `_gap_tier` instance (the report describes the
most recent call). Arm A, window `2015-01-01..2025-12-31`, no raise: `report["unlinked"]` equals
the single-entry list `[{"gvkey": "100020", "iid": "01", "from": "1999-01-13", "thru":
"2007-02-05", "uncovered": [["2007-02-01", "2007-02-05"]]}]` and `report["unlinked_blocking"]`
equals `[]`. Arm B, `allow_unlinked=True` with no window: `report["unlinked"]` equals that same
list and `report["unlinked_blocking"]` equals it too. Arm C, `allow_unlinked=True` with window
`2007-01-01..2007-12-31`: both keys equal that list, and the returned intervals still carry
81020's linked days. Assert the exact dicts, not just lengths — the entry shape is what an
operator reads after a tolerated run.

Test 7, `test_an_inverted_window_is_refused_rather_than_silently_blocking_nothing`: assert
`permno_intervals(NASDAQ100, window=("2020-01-01", "2019-01-01"))` raises `ValueError` naming
both dates, and that the message does NOT mention a CRSP/Compustat link — an inverted window
overlaps nothing, so left unchecked it would suppress every refusal and read as success.

Do not touch any existing test in the file. `test_nasdaq100_unlinked_tail_is_reported_when_allow_unlinked`
asserts `report["unlinked"]` by exact list equality, which is precisely why Task 2 adds a
SEPARATE report key instead of a per-entry flag.
  </action>
  <verify>
    <automated>mkdir -p /tmp/gsd-mb1 && find . -name '__pycache__' -type d -prune -exec rm -rf {} + && uv run pytest -q --tb=no -rf --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py --ignore=tests/test_cross_sectional_zscore.py 2>&1 | grep '^FAILED ' | awk '{print $2}' | sort > /tmp/gsd-mb1/baseline.txt; wc -l /tmp/gsd-mb1/baseline.txt</automated>
    <automated>uv run pytest tests/test_crsp_membership.py -q -k "inside_the_window or outside_the_window or touches_the_window_edge or without_a_window or scopes_the_refusal or blocking_ones or inverted_window" ; test $? -ne 0</automated>
    <automated>uv run pytest tests/test_crsp_membership.py -q -k "not (inside_the_window or outside_the_window or touches_the_window_edge or without_a_window or scopes_the_refusal or blocking_ones or inverted_window)"</automated>
  </verify>
  <done>
`/tmp/gsd-mb1/baseline.txt` holds 55 node IDs (the pre-existing `D-03.11-12-A` failures) and is
NOT staged for commit. The second command exits 0, meaning the seven new tests are red on
today's code. The third passes with the file's existing 21 tests, i.e. the module still collects
and nothing existing moved. Read the red output once
(`uv run pytest tests/test_crsp_membership.py -k "outside_the_window" -x -q 2>&1 | tail -30`)
and confirm the failure is the expected refusal / unexpected-keyword error, not a fixture typo
or a collection error.
  </done>
</task>

<task type="auto">
  <name>Task 2: scope the refusal to the window inside CrspMembership</name>
  <files>quantlab/dataset/crsp/membership.py</files>
  <action>
Thread the window down to the refusal. Five edits, all in this file.

1. `_reset_report` (:176-186): add `"unlinked_blocking": []` beside `"unlinked"`. Keep the
   existing keys and their order. Update the class docstring's "the six keys below are reset on
   entry" (:135) to the new count.

2. `permno_intervals` (:190): add a keyword-only `window` parameter after `allow_unlinked`,
   typed `tuple[date, date] | tuple[str, str] | None`, defaulting to `None`. After the existing
   unknown-index check and BEFORE `_reset_report`, normalise it: when it is not `None`, rebuild
   it as a pair of `date` via `_as_date` on each element (idempotent on a `date`, so
   `permnos_in_range` can pass the dates it already has), then refuse an inverted pair with a
   `ValueError` naming both dates and saying what an inverted window would do — overlap nothing,
   therefore suppress every refusal and return a roster indistinguishable from a complete one.
   Pass the normalised window through to `_nasdaq100_pieces`; the S&P branch does not take one.
   Extend the docstring: `window` scopes the UNLINKED REFUSAL and nothing else — it never
   filters the rows returned — and like `allow_unlinked` it only affects the Nasdaq-100 branch.
   Do not remove or reword anything else in that docstring.

3. `permnos_in_range` (:247): pass `window=(start, end)` — the two dates the method already
   computed with `_as_date` and already validated for inversion at :241. Add two or three lines
   to the docstring saying the window now reaches the refusal, so a Nasdaq-100 roster is refused
   only for uncovered days the window can actually lose a member to. Keep the word NUMERIC and
   the whole order-contract paragraph verbatim: `tests/test_crsp_membership.py` asserts that
   substring is in this docstring.

4. `_nasdaq100_pieces` (:309): add keyword-only `window` (same type, default `None`). Build a
   second list `blocking: list[dict]` beside `unlinked`. In the per-spell loop, at the point the
   report entry is built (:410-422), append the SAME dict object to `unlinked` and — when
   `window is None` or any `(gap_start, gap_end)` in `uncovered` satisfies
   `gap_start <= window[1] and gap_end >= window[0]` — also to `blocking`. Do the comparison on
   the `date` tuples in `uncovered` while they still exist, BEFORE they are stringified into the
   entry: by the time the entry is built, `uncovered` inside it holds strings. Appending one
   dict to two lists is deliberate — the two report keys then cannot drift apart, and the memory
   cost is a second pointer.
   Then replace the tail (:424-433):
   - refuse when `blocking and not allow_unlinked`, raising
     `self._unlinked_message(blocking, window=window)`;
   - when `unlinked` is non-empty, set `self.report["unlinked"] = unlinked`,
     `self.report["unlinked_blocking"] = blocking`, and warn. The warning now has two shapes and
     they must not read alike: with a non-empty `blocking` keep today's sentence (N spells have
     days no link covers, those days are absent from the universe, `allow_unlinked=True` was
     passed so they are listed in `report['unlinked']` instead of raising) and add, when a window
     was given, how many of the N have uncovered days inside it and what the window is; with an
     EMPTY `blocking` and a window, say instead that N spells have days no link covers but that
     none of those days fall inside the requested window (name it), so the universe over that
     window is complete and the spells are recorded in `report['unlinked']` for inspection.
     Note that this second shape now fires on a run that did NOT pass `allow_unlinked` — that is
     the point: the out-of-window fact is scoped out of the refusal, never out of the log.
   Extend the method docstring's "Every membership day must end up with a PERMNO" paragraph with
   the scoping rule and its justification: a span of uncovered days that the requested window
   never touches cannot cost that window a member, so refusing on it is an alarm that fires on
   every pull — and an alarm that always fires trains the operator to pass the escape hatch
   reflexively, which is how the flag stops being noticed on the run where it really does drop
   members.

5. `_unlinked_message` (:505): add a keyword-only `window` parameter defaulting to `None`, and
   rename the parameter's meaning in the docstring to "the spells that BLOCK". When `window` is
   not `None`, insert one sentence after the listing: the refusal is scoped to the requested
   window, named as `start..end`; only spells with uncovered days inside it are listed and only
   they refuse; spells whose gaps fall outside it are recorded in `report['unlinked']` instead.
   Keep the closing survivorship-bias sentence and the `allow_unlinked` / `--allow-unlinked-ndx`
   remedy sentence exactly as they are — three existing tests assert substrings of them.

Update the module docstring only if a sentence there became false; do not rewrite it wholesale
(the concurrent `260922-lu2` task edits a path near :41).
  </action>
  <verify>
    <automated>find . -name '__pycache__' -type d -prune -exec rm -rf {} + && uv run pytest tests/test_crsp_membership.py -q</automated>
    <automated>uv run pytest tests/test_crsp_constituent.py tests/test_ingest_wrds_crsp.py tests/test_crsp_identity.py tests/test_crsp_rebuild.py -q</automated>
  </verify>
  <done>
`tests/test_crsp_membership.py` is green at 28 tests: the file's existing 21 plus the seven from
Task 1. The second command is unchanged from baseline — in particular
`test_an_unlinked_nasdaq100_spell_stops_the_panel_unless_allow_unlinked` (its 1995..2100 default
window covers the planted 2016 spell) and both CLI unlinked tests (their planted spell starts
2015-01-01 and runs open, fully inside the 2015 window) still refuse exactly as before.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: the other two consumers pass their own window, and the docs stop promising the old behaviour</name>
  <files>quantlab/dataset/constituent.py, quantlab/dataset/crsp/__init__.py, tests/test_crsp_constituent.py, tests/test_crsp_membership.py, example/wrds_crsp.md</files>
  <behavior>
    Two new tests, written and confirmed red BEFORE the two source edits:

    1. `tests/test_crsp_constituent.py` — with an unlinked spell entirely before the panel's
       configured window, `CompustatNasdaq100ConstituentDataset._build_intervals()` returns
       instead of raising; with a window that covers the gap, it still raises.
    2. `tests/test_crsp_membership.py` — with the same shape of tier,
       `CrspStockDataset._member_intervals()` returns the full-history intervals instead of
       raising when the configured conversion window is clear of the gap, and still raises when
       the window covers it.
  </behavior>
  <action>
Write both tests first, run them red, then make them green with the two source edits. Capture
the red output in the summary; it is the evidence that each test fails for the consumer's own
missing window rather than for something Task 2 already fixed.

TEST 1, in `tests/test_crsp_constituent.py`, immediately after
`test_an_unlinked_nasdaq100_spell_stops_the_panel_unless_allow_unlinked` (:716) so the pair reads
together: `test_an_unlinked_spell_before_the_panel_window_no_longer_stops_the_panel`. Reuse the
file's own `_ndx_spell`, `_link`, `_ndx_reference` and `_panel_config`; add no fixtures. Build the
tier with an explicit `spells=` and `links=` pair of the same shape Task 1 used — gvkey `100020`
iid `01`, spell `1999-01-13..2007-02-05`, link to `81020.0` ending `2007-01-31`, plus a fully
linked gvkey `100021` iid `01` from `2015-01-02` open, link to `81021.0` from `2010-01-01` open
— marked `# SYNTHETIC` with the live gvkey it mirrors named in the comment. Then assert:
`CompustatNasdaq100ConstituentDataset(_panel_config(tmp_path, reference_dir, "ndx_window",
start_date="2015-01-01", end_date="2025-12-31"))._build_intervals()` returns a frame whose
`symbol` values are `{81020, 81021}`; and that the SAME class over
`_panel_config(..., start_date="2007-01-01", end_date="2007-12-31")` raises `ValueError`
naming `100020`. Say in the docstring why the assertion is on `_build_intervals()` and not on
`from_raw_data()`: the seam under test is the refusal, and the densification either side of it
has its own tests; and why the frame still carries 81020 although the window is clear of its
membership — a window scopes the refusal, it does not filter intervals.

TEST 2, in `tests/test_crsp_membership.py` under the Task 1 banner:
`test_the_conversion_window_reaches_the_refusal_through_member_intervals`. Import
`CrspDatasetConfig` and `CrspStockDataset` inside the test, per the file's convention. Build the
tier with the module's own `_gap_tier` helper from Task 1 — take its reference directory from
`membership.reference.directory` if `CrspReference` exposes one, otherwise write the tier through
`tests.crsp_fixtures.write_reference_tables` exactly as `_membership` (:90) does and reuse that
path for both objects; do not duplicate the row builders. Construct
`CrspStockDataset(CrspDatasetConfig(zarr_file_path=..., raw_data_dir_path=..., catalog_path=...,
reference_dir=<that tier>, start_date="2015-01-01", end_date="2025-12-31",
roster_universe="comp_nasdaq100"))` — construction performs no IO, the pattern is
`tests/test_crsp_identity.py:1176-1191` — and assert `_member_intervals()` returns a frame whose
`permno` set is `{81020, 81021}`. Then build the same object with `start_date="2007-01-01"`,
`end_date="2007-12-31"` and assert `_member_intervals()` raises `ValueError` naming `100020`.
Docstring: this memo is the security filter's roster exemption
(`quantlab/dataset/crsp/__init__.py:773`), it is reached on every `--universe` conversion, and
it passes no `allow_unlinked` at all — so before this change it refused a whole conversion over
a 1999 link gap with no escape hatch on the command line.

SOURCE EDIT A, `quantlab/dataset/constituent.py`: in BOTH `_build_intervals` bodies (:170-178
and :217-226) add `window=(self.config.start_date, self.config.end_date)` to the
`permno_intervals` call. Pass it from both, for the same reason `allow_unlinked` is already
passed from both although it too only affects the Nasdaq-100 branch: the two bodies are
byte-identical today, and forking them would advertise a difference that does not exist. Add one
comment line at the Nasdaq-100 one recording why the window is safe here: the panel's own edges
are `max(config.start_date, coverage_start)`..`min(config.end_date, horizon)`, a subset of the
window being passed, so an uncovered span outside it cannot touch a cell of this panel. Extend
that class's "An unlinked spell REFUSES by default" docstring paragraph (:198-205) by one
sentence: the refusal is scoped to the panel's own configured window, and a gap outside it is
recorded in the membership `report['unlinked']` rather than refusing.

SOURCE EDIT B, `quantlab/dataset/crsp/__init__.py`: at `_member_intervals` (:773) add
`window=(self.config.start_date, self.config.end_date)` to the `permno_intervals` call. Add two
sentences to that method's docstring: the window is the conversion's own window, the derivation
is computed over exactly that range (module docstring, :11) so a link gap outside it cannot
affect a single exempted row; and note plainly that this call still passes no `allow_unlinked`,
so an IN-window gap refuses the conversion with no CLI escape hatch — a known, deliberate gap
recorded here rather than fixed under this task. Do not add a config field for it.

DOC EDIT, `example/wrds_crsp.md` (Chinese, match the file's voice):
- The bullet at :777-783 currently states the opposite of the new behaviour and calls scoping an
  optional follow-up. Rewrite that bullet: the check is now scoped to the requested window; the
  six live spells (gvkey 012884 / 063180 / 064606 / 065068 / 065489 / 106368, uncovered ranges
  all in 1999-2008) no longer stop a 2015+ pull; they remain listed in `report['unlinked']` and
  in the log; a gap that DOES fall inside the requested window still stops the run, and that is
  when `--allow-unlinked-ndx` is genuinely lossy and worth reading carefully. Remove the
  follow-up sentence — this task is that follow-up.
- The paragraph at :505-507: add one sentence that the refusal considers only uncovered days
  inside the requested window, and that the rest are recorded rather than raised.
- The live-measurement table row at :728 records `--universe comp_nasdaq100` 2024 exiting 1 on
  six unlinked spells. Do NOT rewrite a measured historical row. Add a dated 追记 note under the
  table, in the style of the existing 2026-09-21 one, saying that since 2026-09-22 the same
  command no longer exits on those six spells because their uncovered ranges are all outside a
  2024 window, and that the roster figure itself (108 PERMNOs) is unchanged because
  `--allow-unlinked-ndx` never dropped a member the window could see.
- The comment at :703 ("若因未链接的 spell 停下…") stays true and needs no edit.

Do not touch `scripts/ingest_wrds_crsp.py`: it already hands `permnos_in_range` the window
positionally and already hands the constituent config its own dates.
  </action>
  <verify>
    <automated>find . -name '__pycache__' -type d -prune -exec rm -rf {} + && uv run pytest tests/test_crsp_constituent.py tests/test_crsp_membership.py -q</automated>
    <automated>grep -n "window=(self.config.start_date, self.config.end_date)" quantlab/dataset/constituent.py quantlab/dataset/crsp/__init__.py</automated>
    <automated>uv run pytest -q --tb=no -rf --ignore=tests/test_factor_hierarchy.py --ignore=tests/test_crsp_rebuild_measurements.py --ignore=tests/test_cross_sectional_zscore.py 2>&1 | grep '^FAILED ' | awk '{print $2}' | sort > /tmp/gsd-mb1/after.txt; diff /tmp/gsd-mb1/baseline.txt /tmp/gsd-mb1/after.txt</automated>
  </verify>
  <done>
The first command is green: `tests/test_crsp_membership.py` at 29 tests and
`tests/test_crsp_constituent.py` at 15. The grep prints three lines —
two in `constituent.py`, one in `crsp/__init__.py`. The `diff` prints NOTHING and exits 0: the
failing node-ID set is identical in both directions, which is the gate (the suite is not green;
55 pre-existing `D-03.11-12-A` failures are expected on both sides). Allow the full run several
minutes; macOS is single-threaded under the `OMP_NUM_THREADS=1` guard. `/tmp/gsd-mb1/*.txt` is
scratch and is not committed. `git status` shows exactly the six files in `files_modified` and
nothing under `.planning/phases/`.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| WRDS reference tier (parquet on disk) -> `CrspMembership` | Vendor link/membership tables decide who is in a universe; a wrong answer here is survivorship bias written into every downstream backtest. |
| `CrspMembership` refusal -> operator | The refusal is the only signal that a universe would be incomplete; its credibility is the control. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-mb1-01 | Tampering | the scoped refusal in `_nasdaq100_pieces` | high | mitigate | The window narrows the refusal only where an uncovered span provably cannot cost the window a member. Locked from both sides: tests 1 and 3 of Task 1 assert an in-window or edge-touching gap still refuses; test 4 asserts a window-less call is unchanged. |
| T-mb1-02 | Repudiation | `report['unlinked']` / the warning | medium | mitigate | Out-of-window spans are still recorded AND still logged, now with text that names the window and distinguishes "none in your window" from "some in your window". Task 1 test 6 asserts the report is complete in the non-raising case. |
| T-mb1-03 | Spoofing | an inverted `window` argument | medium | mitigate | An inverted pair would overlap nothing and suppress every refusal; `permno_intervals` refuses it outright, asserted by Task 1 test 7. |
| T-mb1-04 | Denial of Service | operator habituation to `--allow-unlinked-ndx` | medium | mitigate | This is the defect being fixed: a gate that fires on every pull is not a gate. After this change the flag is requested only on runs where it really does drop members. |
| T-mb1-SC | Tampering | npm/pip/cargo installs | high | mitigate | Not applicable: this plan installs no package and adds no dependency. If a task turns out to need one, stop and run the package-legitimacy gate first. |
</threat_model>

<verification>
- `uv run pytest tests/test_crsp_membership.py -q` — 29 passed (21 existing + 8 new).
- `uv run pytest tests/test_crsp_constituent.py tests/test_ingest_wrds_crsp.py tests/test_crsp_identity.py tests/test_crsp_rebuild.py -q` — unchanged from baseline.
- `diff /tmp/gsd-mb1/baseline.txt /tmp/gsd-mb1/after.txt` — empty, exit 0. The gate is the
  failing node-ID set difference in BOTH directions, not a green suite.
- Every command is repo-root-relative. Rewriting one to an absolute `/Users/...` path would test
  unchanged main-tree code from the isolated worktree and report a false green.
</verification>

<success_criteria>
- Over a tier whose only link gap is `2007-02-01..2007-02-05`, `permnos_in_range(NASDAQ100,
  "2015-01-01", "2025-12-31")` returns `["81021"]` instead of raising.
- The same tier with a 2007 window, a window that touches either edge of the gap, or no window
  at all, still raises — message unchanged except for the added window-scoping sentence.
- `permno_intervals(..., window=...)` returns the same rows it returns without a window.
- `report['unlinked']` lists every unlinked spell in both cases; `report['unlinked_blocking']`
  lists only those that would refuse.
- The warning text for "gaps, none in your window" differs from "gaps, some in your window" and
  names the window in both.
- `quantlab/dataset/constituent.py` (both classes) and `quantlab/dataset/crsp/__init__.py`
  pass their own configured window, so the operator's reported command clears all three
  consumers.
- `example/wrds_crsp.md` no longer describes the refusal as whole-history, and no longer lists
  window-scoping as an open follow-up.
- No file under `.planning/phases/` is modified, and none of the eight `260922-lu2` movers is
  touched.
</success_criteria>

<output>
Create `.planning/quick/260922-mb1-scope-the-nasdaq100-unlinked-refusal-to-the-window/SUMMARY.md` when done.
</output>
