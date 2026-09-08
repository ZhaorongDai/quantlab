---
phase: quick-260907-uac
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/dataset/backend.py
  - tests/test_chunked_ingest.py
  - tests/test_symbol_axis_widening.py
  - .planning/todos/pending/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md
autonomous: true
requirements: [APPD-01, APPD-02, APPD-03, APPD-04, APPD-05, APPD-06]
estimate:
  tokens: 58000
  raw_tokens: 58000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "APPD-01: `XrBackend.append()` RAISES `ValueError` before writing anything when the incoming `append_dim` coordinate begins at or before the stored end. Measured 2026-09-07: today it succeeds silently and the store comes back holding `['2022-01-04','2022-01-05','2022-01-06','2022-01-05','2022-01-06','2022-01-07']` -- `is_unique=False`, `is_monotonic_increasing=False`."
    - "APPD-02: The refusal message names the store path, the append dimension, the STORED END, the INCOMING START, and the rewrite remedy `save(mode=\"w\")`. It is the message shape the sibling coordinate guard already set: say what is wrong and say what to do instead."
    - "APPD-03: A GAP is still allowed. A window starting strictly after the stored end appends normally, whatever the distance. This is a decided design point, not an oversight -- a test keeps a future reader from 'completing' the guard."
    - "APPD-04: There is NO opt-out. `XrBackend.append`'s signature stays exactly `(self, path, append_dim, **kwargs)` -- no force flag, no overwrite flag, no mode parameter. Recomputing an already-stored range is the rewrite path's job."
    - "APPD-05: `widen_and_append` inherits the refusal for free through its closing `append()` call, and carries NO parallel check of its own -- proved by the two paths producing a byte-identical message once the store path is normalized out."
    - "APPD-06: The chunked ingest path is unaffected. `TimeChunkPlanner.plan_from_timestamps` returns windows that are 'time-ordered, non-overlapping, and together cover the whole de-duplicated axis exactly once', so the new guard is satisfied by construction there. Live baseline 537 passed stays green."
  artifacts:
    - quantlab/dataset/backend.py
    - tests/test_chunked_ingest.py
    - tests/test_symbol_axis_widening.py
    - .planning/todos/completed/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md
  key_links:
    - "`XrBackend.append` -> `_assert_append_compatible(path, append_dim)` -- the single call site of the guard, already in place; the new check rides it and needs no new wiring."
    - "`XrBackend.widen_and_append` -> the UNCHANGED `XrBackend.append` -- the load-bearing closing call its docstring already names. This is the whole mechanism by which the widened path is covered; nothing may be added beside it."
    - "`BaseDataset.from_raw_data_chunked` -> `self.data_backend.append(zarr_file_path, append_dim=...)` -- the only production caller, ledger-driven and contiguous by construction."
---

<objective>
Close the last hole in `XrBackend._assert_append_compatible`: it guards every
non-append dimension's coordinate and every shared variable's dtype, but the
append dimension itself is skipped by the first line of its dim loop, so
nothing compares an incoming window's timestamps against what the store
already holds.

Purpose: an overlapping append silently produces duplicate timestamps. Measured
2026-09-07 against the live backend -- the append returns cleanly, and the
damage surfaces later and elsewhere: `.sel(timestamp=slice(...))` raises
`KeyError: 'Value based partial slicing on non-monotonic DatetimeIndexes with
non-existing keys is not allowed.'`, a `to_xarray` round-trip raises
`ValueError: cannot convert a DataFrame with a non-unique MultiIndex into
xarray`, and a point `.sel()` on a duplicated date quietly returns TWO rows
where the caller expects one. Nothing points back to the append that caused it.
This is the same class as the two guards already here -- zarr re-attributing
stored coordinate labels, and a float NaN cast into an integer store becoming a
fabricated `0` -- both built on "refuse rather than write something wrong
quietly".

Output: one added check inside `_assert_append_compatible`, seven tests, and a
closed todo.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/todos/pending/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md
@CLAUDE.md

@quantlab/dataset/backend.py
@tests/test_chunked_ingest.py
@tests/test_symbol_axis_widening.py
</context>

<decisions_already_made>
These were settled by the developer on 2026-09-07. Implement them. Do not
re-open them, do not soften them, and do not add a "for now" around either.

- **D-01: gaps are NOT guarded.** Only overlap is refused. A discontinuous time
  axis is not an error here. APPD-03 exists to keep a later reader from
  "finishing the job".
- **D-02: the refusal is unconditional.** No overwrite escape hatch, now or
  later. Recomputing an already-stored date range is the rewrite path's job
  (`save(mode="w")` replaces the store); `append()` exists to extend it. An
  opt-out would blur the exact boundary those two methods are separate in order
  to keep sharp. APPD-04 locks this structurally, on the signature, rather than
  by wording.
- **D-03: `widen_and_append` gets no separate handling.** It closes by calling
  the UNCHANGED `append()`, which its docstring already declares load-bearing
  precisely so guards keep applying on the widened path. The new guard covers it
  for free. Do NOT add a parallel check there, and do not let the widened path
  bypass it.
- **D-04: out of scope -- rolling-window warm-up.** Whether a factor value had
  enough lookback to be valid is a property of `cal()`. The storage layer cannot
  distinguish a warm-up artifact from a genuine low value and must not try.
</decisions_already_made>

<observed_state>
Everything below was read or measured live at planning time on 2026-09-07.
Prefer these over any line number quoted elsewhere.

- `quantlab/dataset/backend.py` is 565 lines. `XrBackend.append` at line 43,
  `widen_and_append` at 250, `_append_encoding` at 314,
  `_assert_append_compatible` at 342. `numpy as np`, `pandas as pd` and
  `xarray as xr` are already imported at the top.
- `_assert_append_compatible` opens the store with `xr.open_zarr(path)` inside a
  `try/finally` that closes it, runs a non-append-dim coordinate loop, then a
  dtype loop over `self.data.data_vars`.
- **Live test baseline: `uv run pytest tests/ -q` -> `537 passed` in ~27s.** The
  figure quoted in the task brief was re-derived here rather than trusted.
- The two sibling guards for this method are tested in **`tests/test_chunked_ingest.py`**
  (`test_append_refuses_a_changed_symbol_axis`,
  `test_plain_append_still_refuses_a_labels_differ_axis_of_the_same_length`,
  `test_append_refuses_a_changed_dtype`), using a local `_small_panel(dates,
  symbols, offset)` helper. `tests/test_symbol_axis_widening.py` owns
  `widen_and_append`'s contract with an equivalent `_panel(...)` helper and a
  `_stored(path)` reader. That is the split this plan follows.
- Every existing append in the suite writes a strictly-later window (2022 ->
  2023), and `TimeChunkPlanner.plan_from_timestamps` documents its windows as
  "time-ordered, non-overlapping, and together cover the whole de-duplicated
  axis exactly once". No existing green test performs an overlapping append.
- Live signatures, for APPD-04: `append` is `('self','path','append_dim','kwargs')`;
  `_assert_append_compatible` is `('self','path','append_dim')`.
- **A store can legitimately carry NO coordinate on the append dimension.**
  Measured: a panel with a `timestamp` DIM but no `timestamp` COORD writes and
  re-appends cleanly today (n=2 then n=4). The new check must skip that case the
  way the existing dim loop already skips a coordinate absent on either side, or
  it turns a working path into a crash.
- Measured shape for message formatting: the stored coordinate comes back as
  `datetime64[ns]`; `repr()` of a scalar renders as
  `np.datetime64('2022-01-07T00:00:00.000000000')` while
  `pd.Timestamp(...).isoformat()` renders `2022-01-07T00:00:00`.
- **A doc-sweep todo is open** (`.planning/todos/pending/2026-09-07-re-point-stale-former-root-paths-in-python-docstrings.md`)
  that counts backticked pre-migration package paths inside `.py` files. Current
  counts for the three files this plan touches: `quantlab/dataset/backend.py` 1
  (a pre-existing hit in `widen_and_append`'s docstring -- leave it alone, it
  belongs to that todo), both test files 0. Write every path reference in new
  prose with the `quantlab/` package prefix so this plan adds nothing to that
  backlog. Task 3 gates on the counts being unchanged.
</observed_state>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Refuse an overlapping append, end to end through the real guard</name>
  <files>quantlab/dataset/backend.py, tests/test_chunked_ingest.py</files>
  <read_first>quantlab/dataset/backend.py lines 342-378 (`_assert_append_compatible` in full), tests/test_chunked_ingest.py lines 249-355 (the `_small_panel` helper and the three existing guard tests, for idiom)</read_first>
  <behavior>
    The one path, wired through every layer it touches: a real Zarr store on
    disk, a real `XrBackend.append()` call, the real `_assert_append_compatible`
    guard, a real refusal, a store left untouched.

    - Store holds 2022-01-04, 2022-01-05, 2022-01-06 on symbols A and B.
    - Incoming window holds 2022-01-05, 2022-01-06, 2022-01-07 on the SAME
      symbol axis, so neither existing guard can fire and a green test cannot
      be green for their reason.
    - `append()` raises `ValueError`.
    - The message contains the store path, the string `timestamp`, the stored
      end and the incoming start (both rendered as readable dates), and the
      remedy `save(mode="w")`.
    - The store is unchanged afterwards: three timestamps, `is_unique` True,
      `is_monotonic_increasing` True, and the `close` values bit-identical to
      before the refused call.
  </behavior>
  <action>
    Write the test FIRST, in `tests/test_chunked_ingest.py`, immediately after
    `test_append_refuses_a_changed_dtype` so the three guards of one method sit
    together. Name it
    `test_append_refuses_a_window_overlapping_the_stored_timestamps`. Build both
    panels with the file's existing `_small_panel` helper and read the store
    back with its existing `_panel(path)` reader. Follow the neighbouring
    docstring idiom: state the measured corruption and name the mutation that
    reddens the test.

    Run it before touching `quantlab/dataset/backend.py` and CONFIRM the failure
    mode empirically: it must fail with pytest's DID NOT RAISE for `ValueError`,
    not with a collection error, an import error or a fixture error. A RED that
    fails for a setup reason proves nothing. Record the observed failure line in
    the summary.

    Then extend `XrBackend._assert_append_compatible`. Add the check INSIDE the
    existing `try` block, AFTER the non-append dimension loop and BEFORE the
    dtype loop -- that position leaves every existing message's precedence
    exactly as it is today, so no currently-green test can change outcome.

    The check: skip unless `append_dim` is present in `self.data.coords` AND in
    `existing.coords`, mirroring the guard the dim loop above already applies;
    skip when either coordinate array is empty, since there is nothing to
    compare. Otherwise take the stored maximum and the incoming minimum with
    numpy `.max()` / `.min()` rather than positional indexing, so an unsorted
    axis cannot fool the comparison, and raise when the incoming minimum is less
    than OR EQUAL TO the stored maximum. Equality must refuse: a window starting
    exactly on the stored end duplicates that one label.

    Render the two values through a tiny local formatter that returns
    `pd.Timestamp(value).isoformat()` for a `datetime64` coordinate and falls
    back to `str(value)` for anything else -- `append_dim` is a parameter and
    this guard must not become timestamp-only. Compose the message in the shape
    the coordinate-mismatch message above it already uses: name the store path
    and the dimension, state the stored end and the incoming start, say that
    zarr would extend the axis without complaint and leave duplicate labels
    behind, and route the reader to the rewrite path by naming
    `save(mode="w")` for recomputing an already-stored range while `append()`
    extends it. Per D-02, name no flag, parameter or option that would let a
    caller past this refusal -- none exists and none is coming.

    Update `XrBackend.append`'s docstring where it enumerates the enforced
    contract, adding the append dimension as the third guarded property and
    naming the measured duplicate-timestamp outcome. Per D-01, state plainly in
    that prose that a gap is permitted. Write any path reference with the
    `quantlab/` package prefix.
  </action>
  <verify>
    <automated>uv run pytest "tests/test_chunked_ingest.py::test_append_refuses_a_window_overlapping_the_stored_timestamps" -q && uv run pytest tests/ -q 2>&1 | tail -1 | grep -qE '^538 passed'</automated>
  </verify>
  <done>The named test passes, the full suite reports exactly 538 passed with zero failures, and the summary records the observed pre-implementation RED as DID NOT RAISE rather than a setup error.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Lock the decided boundaries -- gap allowed, no escape hatch, widened path inherits</name>
  <files>tests/test_chunked_ingest.py, tests/test_symbol_axis_widening.py</files>
  <read_first>tests/test_symbol_axis_widening.py lines 1-90 (module docstring, `_panel`, `_typed_panel`, `_stored`) and lines 367-412 (the two `widen_and_append` delegation tests, for idiom)</read_first>
  <behavior>
    Five more tests, each locking one decided property that the tracer test
    alone leaves unprotected. Every docstring names the mutation that reddens
    it, per this module's stated convention.

    In `tests/test_chunked_ingest.py`, beside the tracer test:
    - `test_append_allows_a_gap_between_the_stored_end_and_the_incoming_start`
      -- store 2022-01-04/2022-01-05, append 2023-06-01, succeeds, three
      timestamps. Locks D-01. RED under extending the guard to refuse gaps.
    - `test_append_refuses_a_window_that_ends_before_the_stored_start` -- store
      2022-06-01/2022-06-02, incoming 2022-01-04. No duplicate label is
      produced, but the axis goes non-monotonic and the same
      `.sel(timestamp=slice(...))` `KeyError` follows; measured 2026-09-07. This
      is the direct consequence of comparing the incoming start against the
      stored end, which is the comparison the required message shape describes.
      RED under comparing against the stored MINIMUM instead of the maximum.
    - `test_append_refuses_a_window_starting_exactly_on_the_stored_end` --
      store ends 2022-01-06, incoming starts 2022-01-06. This is the ONLY test
      that reddens when the comparison is relaxed from less-than-or-equal to
      strictly-less-than; record that in its docstring, because a mechanism
      covered by exactly one test is a weaker guarantee than it looks.
    - `test_append_skips_the_overlap_check_without_an_append_dim_coordinate` --
      a panel carrying a `timestamp` dim and no `timestamp` coord appends twice
      and reaches four rows, exactly as it does today. RED under dropping the
      coordinate-presence skip, which turns a working path into a crash.
    - `test_append_offers_no_overwrite_escape_hatch` -- assert via
      `inspect.signature(XrBackend.append)` that the parameter names are
      exactly `self`, `path`, `append_dim`, `kwargs`. Locks D-02 structurally,
      on the signature, where no message wording can drift out from under it.
      RED under adding any parameter that would let a caller past the refusal.

    In `tests/test_symbol_axis_widening.py`, at the end of the
    `widen_and_append` section:
    - `test_widen_and_append_inherits_the_overlap_refusal_verbatim` -- two
      stores under `tmp_path`. Both start with the same panel on symbols A and
      B over 2022-01-04..2022-01-06. Against the first, plain `append()` an
      overlapping 2022-01-05..2022-01-07 window on A and B and capture the
      message. Against the second, `widen_and_append()` the same overlapping
      window carrying an ADDED symbol C, so the widen genuinely runs, and
      capture that message. With each store's own path replaced by a fixed
      placeholder, the two messages must be EQUAL. That equality is the proof
      of D-03: a second, separately-worded check inside `widen_and_append`
      cannot produce it. Also assert the accepted, documented side effect --
      the widen commits before the closing `append()` raises, so the store's
      symbol axis MAY have grown to A, B, C, while its timestamp axis is still
      the original three labels, unique and monotonic, and A's and B's stored
      values on those three timestamps are unchanged. Assert that guarantee;
      do NOT assert that the symbol axis stayed narrow.
  </behavior>
  <action>
    Add the five tests to `tests/test_chunked_ingest.py` and the one test to
    `tests/test_symbol_axis_widening.py`, using each file's own existing panel
    helpers rather than introducing new ones. Import `inspect` in
    `tests/test_chunked_ingest.py` for the signature lock.

    Change NOTHING in `quantlab/dataset/backend.py` in this task. Per D-03 in
    particular, `widen_and_append` must not gain a check, a reorder, or an early
    return. If the verbatim-message test fails, the fix is in the test or in the
    shared guard, never a second check on the widened path.

    Write every path reference in new prose with the `quantlab/` package prefix.
  </action>
  <verify>
    <automated>uv run pytest tests/test_chunked_ingest.py tests/test_symbol_axis_widening.py -q && uv run pytest tests/ -q 2>&1 | tail -1 | grep -qE '^544 passed'</automated>
  </verify>
  <done>Both test files pass in full and the whole suite reports exactly 544 passed -- the live 537 baseline plus the tracer test plus these six -- with zero failures and no change to quantlab/dataset/backend.py in this task's diff.</done>
</task>

<task type="auto">
  <name>Task 3: Mutation-verify every lock, gate the prose, close the todo</name>
  <files>quantlab/dataset/backend.py, tests/test_chunked_ingest.py, tests/test_symbol_axis_widening.py, .planning/todos/pending/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md</files>
  <action>
    This repo has hit "a verification that passed for the wrong reason" four
    times in two days. Do not accept an arriving-green lock. Apply each mutation
    below to a clean tree, run the named command, confirm the test reddens AND
    that it reddens through the assertion it claims to exercise rather than
    through an unrelated error, then revert with `git checkout --` before the
    next one. Record the observed failing assertion for each in the summary.

    M1 -- delete the whole new check from `_assert_append_compatible`. Expect
    the overlap, ends-before, starts-exactly-on and widen-inherits tests to
    redden together.

    M2 -- relax the comparison from less-than-or-equal to strictly-less-than.
    Expect exactly the starts-exactly-on test to redden and the overlap test to
    stay green. If the overlap test also reddens, the fixtures are not isolating
    what the docstrings claim; fix the fixtures.

    M3 -- compare the incoming MAXIMUM against the stored maximum instead of the
    incoming minimum. Expect the overlap test to redden.

    M4 -- add a second, differently-worded overlap check at the top of
    `widen_and_append` so it raises before delegating. Expect the
    verbatim-message test to redden on the message equality assertion. This is
    the mutation that proves D-03 is enforced rather than merely requested.

    M5 -- extend the guard to also refuse a gap. Expect the gap test to redden.
    This is the mutation that proves D-01 is enforced.

    Then run the full suite once more on the clean tree, and run the prose gate
    below, which asserts that this plan added no new backticked pre-migration
    package path to the three touched files -- the counts must be exactly the
    ones measured at planning time. The gate SORTS its input: this machine's
    grep emits multi-file counts in non-deterministic order, measured at
    planning time, so an unsorted comparison fails intermittently for a reason
    that has nothing to do with the code. This gate deliberately counts comment
    and docstring lines, because prose is exactly what it is auditing.

    Close the todo: `git mv` the pending file to
    `.planning/todos/completed/`, set its frontmatter `status` to `closed`, and
    append a `## Closed 2026-09-07` section following the convention in
    `.planning/todos/completed/2026-09-07-rebuild-universe-parquet-after-ticker-reconciliation.md`.
    That section must record what shipped, the mutation results, and the two
    decided points that were implemented rather than re-opened, so a later
    reader who wants to "finish" the gap half or add an override finds the
    reasoning before the code.
  </action>
  <verify>
    <automated>uv run pytest tests/ -q 2>&1 | tail -1 | grep -qE '^544 passed' && diff <(grep -cE '`(acquisition|base|config|dataset|dl_model|enums|factor|label|ml_model|my_ops|utils|vecbt)([/.]|` )' quantlab/dataset/backend.py tests/test_chunked_ingest.py tests/test_symbol_axis_widening.py 2>/dev/null | sort) <(printf 'quantlab/dataset/backend.py:1\ntests/test_chunked_ingest.py:0\ntests/test_symbol_axis_widening.py:0\n' | sort) && test -f .planning/todos/completed/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md && test ! -f .planning/todos/pending/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md && grep -q '^status: closed' .planning/todos/completed/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md</automated>
  </verify>
  <done>All five mutations were run and each reddened the predicted test through the predicted assertion, with the observations recorded in the summary; the suite reports 544 passed; the prose gate matches its planning-time counts exactly; and the todo is under completed/ with status closed and a Closed section.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| in-memory panel -> on-disk Zarr store | An irreversible write. Once an append lands, the store alone cannot tell a correct history from a corrupted one -- which is why every guard here fires BEFORE `to_zarr`. |
| store -> every downstream reader | `.sel()`, `to_xarray()` and the model layer all assume a unique, monotonic time index. A violation is detected far from its cause, if at all. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-uac-01 | Tampering | `XrBackend.append` -> `_assert_append_compatible` | high | mitigate | The new append-dim check refuses before `to_zarr`. Measured: without it, an overlapping window yields duplicate timestamps and a non-monotonic axis with nothing raised. Task 1. |
| T-uac-02 | Tampering | `XrBackend.widen_and_append` | high | mitigate | The widened path must not bypass the guard. Proved by the byte-identical message across both paths (APPD-05), mutation M4. |
| T-uac-03 | Repudiation | the refusal message | medium | mitigate | A refusal that does not name the stored end, the incoming start and the remedy teaches a caller to route around it. Message content asserted in Task 1. |
| T-uac-04 | Elevation of Privilege | a future overwrite opt-out on `append()` | medium | mitigate | D-02: no escape hatch. Locked on the signature by `test_append_offers_no_overwrite_escape_hatch`, not on wording. |
| T-uac-05 | Denial of Service | the guard refusing a legitimate append | medium | mitigate | A guard that over-refuses gets deleted rather than obeyed. The gap case (D-01) and the no-append-dim-coordinate case are both measured-working today and are locked green by APPD-03 and its sibling. |

No package-manager installs in this plan; no legitimacy gate applies.
</threat_model>

<verification>
- `uv run pytest tests/ -q` reports `544 passed`, zero failures. The `537`
  baseline was re-measured live at planning time rather than trusted.
- Each of the seven new tests names, in its docstring, the mutation that reddens
  it -- this module's stated convention -- and Task 3 ran five of those
  mutations and observed the predicted reddening.
- `quantlab/dataset/backend.py` has exactly ONE new check, inside
  `_assert_append_compatible`. `widen_and_append` is unchanged (D-03).
- `XrBackend.append`'s parameter list is unchanged (D-02).
- The prose gate reports the planning-time counts `1 / 0 / 0`, so nothing was
  added to the open doc-sweep backlog.
</verification>

<success_criteria>
- An overlapping append raises `ValueError` before writing, and the store is
  bit-identical afterwards.
- The message names the store path, the append dimension, the stored end, the
  incoming start, and `save(mode="w")`.
- A gap still appends. A no-coordinate append dim still appends.
- `widen_and_append` refuses with the identical message and holds no check of
  its own.
- `.planning/todos/pending/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md`
  is under `completed/` via `git mv`, `status: closed`, with a `## Closed` section.
</success_criteria>

<output>
Create `.planning/quick/260907-uac-refuse-an-append-that-overlaps-timestamp/260907-uac-SUMMARY.md` when done.
</output>
