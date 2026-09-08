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
  tokens: 64000
  raw_tokens: 64000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "APPD-01: `XrBackend.append()` RAISES `ValueError` before writing anything when the incoming `append_dim` coordinate begins at or before the stored end. Measured 2026-09-07: today it succeeds silently and the store comes back holding `['2022-01-04','2022-01-05','2022-01-06','2022-01-05','2022-01-06','2022-01-07']` -- `is_unique=False`, `is_monotonic_increasing=False`."
    - "APPD-02: The refusal message names the store path, the append dimension, the STORED END, the INCOMING START, and the rewrite remedy `save(mode=\"w\")`. It is the message shape the sibling coordinate guard already set: say what is wrong and say what to do instead."
    - "APPD-02b (the consequence half of APPD-02, not a separate requirement): The consequence the message STATES is true of every window shape the guard refuses, not just the one that motivated it. Measured 2026-09-07 across all three refusable shapes: a partially overlapping window yields `is_unique` False and `is_monotonic_increasing` False; a window starting exactly on the stored end yields `is_unique` False but `is_monotonic_increasing` TRUE; a window ending before the stored start yields `is_unique` TRUE and `is_monotonic_increasing` False. The single property broken by all three is that the axis is no longer STRICTLY increasing -- so that is the form the message must take, and any consequence naming only duplicates or only disorder is disproved by one of the plan's own fixtures."
    - "APPD-03: A GAP is still allowed. A window starting strictly after the stored end appends normally, whatever the distance. This is a decided design point, not an oversight -- a test keeps a future reader from 'completing' the guard."
    - "APPD-04: There is NO opt-out, DECLARED or SMUGGLED. `XrBackend.append`'s signature stays exactly `(self, path, append_dim, **kwargs)` -- no force flag, no overwrite flag, no mode parameter -- AND no route past the refusal may be carried in `**kwargs` either. The signature alone does not span this: measured 2026-09-07, a hatch popped from `**kwargs` inside the method body made the existing guard's `ValueError` disappear while `inspect.signature` still reported a byte-identical `('self','path','append_dim','kwargs')`. The lock is therefore a PAIR -- one structural assertion on the parameter tuple and one behavioural assertion that an overlapping append carrying an unrecognised kwarg still raises. Recomputing an already-stored range is the rewrite path's job."
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

Output: one added check inside `_assert_append_compatible`, two docstring
updates, eight tests, and a closed todo.
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
- **The three refusable window shapes, measured 2026-09-07** by performing each
  append against the live UNGUARDED backend and reading the resulting axis.
  Store `2022-01-04..2022-01-06`, incoming `2022-01-05..2022-01-07`: labels come
  back `[01-04, 01-05, 01-06, 01-05, 01-06, 01-07]`, `is_unique` False,
  `is_monotonic_increasing` False. Same store, incoming `2022-01-06..2022-01-08`:
  `[01-04, 01-05, 01-06, 01-06, 01-07, 01-08]`, `is_unique` False,
  `is_monotonic_increasing` **True**. Store `2022-06-01..2022-06-02`, incoming
  `2022-01-04..2022-01-05`: `[06-01, 06-02, 01-04, 01-05]`, `is_unique` **True**,
  `is_monotonic_increasing` False. Every one of the three is non-strictly-
  increasing; no weaker property is shared by all three. This is what fixes the
  wording of the refusal message and is why APPD-02b exists.
- **An escape hatch does not have to touch the signature, measured 2026-09-07.**
  In `XrBackend.append`, `self._assert_append_compatible(path, append_dim)` runs
  BEFORE `kwargs.pop("encoding", None)` and before any `to_zarr`. Inserting
  `if kwargs.pop("force", False): self.data.to_zarr(path, mode="w"); return self`
  immediately ahead of that guard call was measured against the EXISTING
  symbol-axis guard: `inspect.signature(XrBackend.append)` still returned
  `('self','path','append_dim','kwargs')` -- byte-identical to baseline -- while
  the guard's `ValueError` vanished entirely. The file was restored afterwards
  (`git diff --stat` clean). A signature-only lock is therefore green through
  the exact drift D-02 exists to catch.
- **On the unguarded tree, `append(path, "timestamp", force=True)` over an
  overlapping window raises `TypeError: Dataset.to_zarr() got an unexpected
  keyword argument 'force'`** -- measured 2026-09-07 -- and the store is left
  unchanged at three unique monotonic labels. So the behavioural D-02 test is
  not trivially green today; once the guard lands, its `ValueError` must arrive
  from `_assert_append_compatible`, ahead of `to_zarr`.
- **A doc-sweep todo is open** (`.planning/todos/pending/2026-09-07-re-point-stale-former-root-paths-in-python-docstrings.md`)
  that counts backticked pre-migration package paths inside `.py` files. Current
  counts for the three files this plan touches: `quantlab/dataset/backend.py` 1
  (a pre-existing hit at line 272, inside `widen_and_append`'s docstring --
  leave it alone, it belongs to that todo, and Task 1 now edits that same
  docstring), both test files 0. Write every path reference in new
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
      end and the incoming start (both rendered as readable dates), the remedy
      `save(mode="w")`, and a consequence clause that holds for EVERY shape the
      guard refuses (APPD-02b), not only for this fixture's.
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
    zarr would extend the axis without complaint and leave it no longer
    STRICTLY increasing -- duplicate labels, out-of-order labels, or both --
    and route the reader to the rewrite path by naming `save(mode="w")` for
    recomputing an already-stored range while `append()` extends it.

    The consequence clause is constrained, not free prose. It must be true of
    all three shapes this guard refuses, and the measurements in
    `<observed_state>` leave exactly one form that is: a window starting on the
    stored end leaves the axis still ordered though no longer strictly rising,
    and a window ending before the stored start leaves it disordered though
    still free of repeats. A clause naming only one of those two failure modes
    is falsified by the other's Task 2 fixture. Assert the SAME clause in all
    three refusal tests, so a wording true of only one shape cannot survive by
    being checked only where it happens to hold.

    Per D-02, name no flag, parameter or option that would let a
    caller past this refusal -- none exists and none is coming.

    Update `XrBackend.append`'s docstring where it enumerates the enforced
    contract, adding the append dimension as the third guarded property and
    naming the measured corrupt-axis outcome. Per D-01, state plainly in that
    prose that a gap is permitted.

    Then add ONE sentence to `widen_and_append`'s docstring recording the
    accepted side effect that Task 2 asserts: because the widen commits before
    the closing `append()` runs, a refused overlapping window can leave the
    store carrying the GROWN symbol axis with its original timestamp axis
    intact and unharmed. This is prose only. It is not a parallel check, it
    changes no control flow, and it does not touch D-03 -- the closing
    `append()` call stays exactly as it is. It belongs in this docstring
    because someone debugging a half-widened store opens `widen_and_append`,
    and today that acknowledgement exists only in a test docstring they have no
    reason to open.

    Write every path reference in both docstrings with the `quantlab/` package
    prefix. `widen_and_append`'s docstring already holds the single
    pre-existing backticked pre-migration path this plan must not disturb
    (line 272, owned by the open doc-sweep todo) -- leave that line byte-for-
    byte alone, so Task 3's prose gate still reports 1 / 0 / 0.
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
    Seven more tests -- six in `tests/test_chunked_ingest.py` and one in
    `tests/test_symbol_axis_widening.py` -- each locking one decided property
    that the tracer test alone leaves unprotected. Every docstring names the
    mutation that reddens it, per this module's stated convention.

    Three tests across this task and Task 1 refuse a differently-shaped window
    -- the tracer's partial overlap, the ends-before window, and the window
    starting exactly on the stored end -- and all three must assert the SAME
    consequence clause lifted from the message. That shared assertion is what
    keeps APPD-02b honest: the three shapes break the axis in provably
    different ways, so a clause that survives all three is one that is true of
    all three.

    In `tests/test_chunked_ingest.py`, beside the tracer test:
    - `test_append_allows_a_gap_between_the_stored_end_and_the_incoming_start`
      -- store 2022-01-04/2022-01-05, append 2023-06-01, succeeds, three
      timestamps. Locks D-01. RED under extending the guard to refuse gaps.
    - `test_append_refuses_a_window_that_ends_before_the_stored_start` -- store
      2022-06-01/2022-06-02, incoming 2022-01-04. Measured 2026-09-07 on the
      unguarded tree: `is_unique` stays TRUE -- no label repeats -- while
      `is_monotonic_increasing` goes False and the same
      `.sel(timestamp=slice(...))` `KeyError` follows. Record both figures in
      the docstring: this fixture is the one that falsifies any message
      consequence phrased as repeated labels, and a reader who cannot see the
      `is_unique` measurement here has no way to check that claim. This is the
      direct consequence of comparing the incoming start against the stored
      end, which is the comparison the required message shape describes.
      RED under comparing against the stored MINIMUM instead of the maximum.
    - `test_append_refuses_a_window_starting_exactly_on_the_stored_end` --
      store ends 2022-01-06, incoming starts 2022-01-06. This is the ONLY test
      that reddens when the comparison is relaxed from less-than-or-equal to
      strictly-less-than; record that in its docstring, because a mechanism
      covered by exactly one test is a weaker guarantee than it looks. Record
      its measured axis state too, which runs opposite to the ends-before
      fixture: 2026-09-07 on the unguarded tree this shape gives `is_unique`
      False with `is_monotonic_increasing` still TRUE. The two fixtures break
      the axis in opposite ways, and only a consequence true of both survives
      the shared clause assertion.
    - `test_append_skips_the_overlap_check_without_an_append_dim_coordinate` --
      a panel carrying a `timestamp` dim and no `timestamp` coord appends twice
      and reaches four rows, exactly as it does today. RED under dropping the
      coordinate-presence skip, which turns a working path into a crash.
    - `test_append_offers_no_overwrite_escape_hatch` -- assert via
      `inspect.signature(XrBackend.append)` that the parameter names are
      exactly `self`, `path`, `append_dim`, `kwargs`. This locks the DECLARED
      half of D-02 only, and its docstring must say so: it is a structural
      proxy that does NOT span the property it stands for. Measured
      2026-09-07, a hatch popped from `**kwargs` inside the method body made
      the existing guard's `ValueError` disappear while this exact parameter
      tuple came back byte-identical -- so on its own this assertion stays
      green through the precise drift D-02 exists to catch. RED under adding a
      NAMED parameter that lets a caller past the refusal; deliberately NOT red
      under a smuggled one, which is the next test's job. Neither test is
      sufficient alone; both must exist.
    - `test_append_refuses_an_overlapping_window_carrying_an_unrecognised_kwarg`
      -- the BEHAVIOURAL half of D-02, and the half that spans the realistic
      drift. Reuse the tracer test's overlapping fixture and call
      `append(path, "timestamp", force=True)`; it must still raise
      `ValueError`, and the store must be unchanged afterwards. State the
      mechanism in the docstring, because it is the reason this passes:
      `XrBackend.append` calls `_assert_append_compatible` BEFORE it touches
      `kwargs` at all, so no kwarg can be consumed ahead of the refusal.
      Measured 2026-09-07 on the unguarded tree, that same call raises
      `TypeError: Dataset.to_zarr() got an unexpected keyword argument
      'force'` -- so this test is not green today by accident, and once the
      guard lands the `ValueError` must arrive from the guard rather than from
      `to_zarr`. RED under any hatch consumed from `**kwargs` before the guard
      call, which is exactly mutation M6.

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
    Add the six tests to `tests/test_chunked_ingest.py` and the one test to
    `tests/test_symbol_axis_widening.py`, using each file's own existing panel
    helpers rather than introducing new ones. Import `inspect` in
    `tests/test_chunked_ingest.py` for the signature lock.

    Keep the two D-02 tests adjacent and cross-reference them in both
    docstrings. A later reader who finds only one of them has half a lock, and
    the half they are most likely to find is the one that does not span the
    property.

    Change NOTHING in `quantlab/dataset/backend.py` in this task. Per D-03 in
    particular, `widen_and_append` must not gain a check, a reorder, or an early
    return. If the verbatim-message test fails, the fix is in the test or in the
    shared guard, never a second check on the widened path.

    Write every path reference in new prose with the `quantlab/` package prefix.
  </action>
  <verify>
    <automated>uv run pytest tests/test_chunked_ingest.py tests/test_symbol_axis_widening.py -q && uv run pytest tests/ -q 2>&1 | tail -1 | grep -qE '^545 passed'</automated>
  </verify>
  <done>Both test files pass in full and the whole suite reports exactly 545 passed -- the live 537 baseline plus the tracer test plus these seven -- with zero failures and no change to quantlab/dataset/backend.py in this task's diff.</done>
</task>

<task type="auto">
  <name>Task 3: Mutation-verify every lock, gate the prose, close the todo</name>
  <files>quantlab/dataset/backend.py, tests/test_chunked_ingest.py, tests/test_symbol_axis_widening.py, .planning/todos/pending/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md</files>
  <action>
    This repo has hit "a verification that passed for the wrong reason" five
    times in two days, the most recent being a structural proxy that did not
    span the property it stood for -- which is why M6 below is not optional.
    Do not accept an arriving-green lock. Apply each mutation
    below to a clean tree, run the named command, confirm the test reddens AND
    that it reddens through the assertion it claims to exercise rather than
    through an unrelated error, then revert with `git checkout --` before the
    next one. Record the observed failing assertion for each in the summary,
    and for M6 record the observed PASS of the signature test alongside it.

    M1 -- delete the whole new check from `_assert_append_compatible`. Expect
    the overlap, ends-before, starts-exactly-on and widen-inherits tests to
    redden together.

    M2 -- relax the comparison from less-than-or-equal to strictly-less-than.
    Expect exactly the starts-exactly-on test to redden and the overlap test to
    stay green. If the overlap test also reddens, the fixtures are not isolating
    what the docstrings claim; fix the fixtures.

    M3 -- compare the incoming MAXIMUM against the stored maximum instead of the
    incoming minimum. This still refuses only a window that ENDS at or before
    the stored end, so expect TWO reddenings, not one: the overlap test AND
    `test_append_refuses_a_window_starting_exactly_on_the_stored_end`, whose
    incoming maximum necessarily lies past the stored end. Any other fixture
    whose overlapping window extends past the stored end reddens for the same
    reason and is equally expected -- the prediction here is the mechanism, not
    a closed list. Only M2 carries a "nothing else may redden" clause, so do
    not stop to investigate a further reddening under M3.

    M4 -- add a second, differently-worded overlap check at the top of
    `widen_and_append` so it raises before delegating. Expect the
    verbatim-message test to redden on the message equality assertion. This is
    the mutation that proves D-03 is enforced rather than merely requested.

    M5 -- extend the guard to also refuse a gap. Expect the gap test to redden.
    This is the mutation that proves D-01 is enforced.

    M6 -- the escape hatch, and the mutation this whole task exists for. Insert
    a route past the refusal that is consumed from `**kwargs` INSIDE the method
    body rather than declared on it: in `XrBackend.append`, immediately ahead
    of the `self._assert_append_compatible(path, append_dim)` call, pop a
    `force` key and, when truthy, write with `to_zarr(mode="w")` and return.
    Expect BOTH of the following, and treat both halves as load-bearing:
    `test_append_refuses_an_overlapping_window_carrying_an_unrecognised_kwarg`
    reddens on its `pytest.raises(ValueError)`, AND
    `test_append_offers_no_overwrite_escape_hatch` stays GREEN. The green half
    is the demonstration that the structural proxy does not span D-02; the red
    half is the demonstration that the pair does. Verified live at planning
    time against the EXISTING symbol-axis guard, which shares this call site:
    with the hatch in, `inspect.signature(XrBackend.append)` still returned
    `('self','path','append_dim','kwargs')` while that guard's `ValueError`
    disappeared.

    If BOTH D-02 tests stay green under M6, the behavioural test is not
    reaching the hatch and the lock is decorative -- fix it and re-run M6
    before this task closes. If the SIGNATURE test also reddens, the mutation
    was written as a named parameter instead of a smuggled kwarg; that is the
    easy shape, not the one being tested, so rewrite it and re-run.

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
    <automated>uv run pytest tests/ -q 2>&1 | tail -1 | grep -qE '^545 passed' && diff <(grep -cE '`(acquisition|base|config|dataset|dl_model|enums|factor|label|ml_model|my_ops|utils|vecbt)([/.]|` )' quantlab/dataset/backend.py tests/test_chunked_ingest.py tests/test_symbol_axis_widening.py 2>/dev/null | sort) <(printf 'quantlab/dataset/backend.py:1\ntests/test_chunked_ingest.py:0\ntests/test_symbol_axis_widening.py:0\n' | sort) && test -f .planning/todos/completed/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md && test ! -f .planning/todos/pending/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md && grep -q '^status: closed' .planning/todos/completed/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md</automated>
  </verify>
  <done>All six mutations were run and each reddened the predicted test(s) through the predicted assertion, with the observations recorded in the summary; M6 additionally recorded the signature test staying GREEN while the behavioural test reddened; the suite reports 545 passed; the prose gate matches its planning-time counts exactly; and the todo is under completed/ with status closed and a Closed section.</done>
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
| T-uac-03 | Repudiation | the refusal message | medium | mitigate | A refusal that does not name the stored end, the incoming start and the remedy teaches a caller to route around it. A refusal whose stated consequence is false for the shape in hand teaches the same lesson faster. Message content asserted in Task 1 and re-asserted, as one shared consequence clause, across all three refusal shapes (APPD-02b). |
| T-uac-04 | Elevation of Privilege | a future overwrite opt-out on `append()` | medium | mitigate | D-02: no escape hatch, DECLARED or SMUGGLED. Locked by a PAIR -- `test_append_offers_no_overwrite_escape_hatch` on the parameter tuple, and `test_append_refuses_an_overlapping_window_carrying_an_unrecognised_kwarg` on behaviour. The signature alone is measurably insufficient: a hatch popped from `**kwargs` leaves it byte-identical (measured 2026-09-07). Mutation M6 demonstrates exactly that split. |
| T-uac-05 | Denial of Service | the guard refusing a legitimate append | medium | mitigate | A guard that over-refuses gets deleted rather than obeyed. The gap case (D-01) and the no-append-dim-coordinate case are both measured-working today and are locked green by APPD-03 and its sibling. |

No package-manager installs in this plan; no legitimacy gate applies.
</threat_model>

<verification>
- `uv run pytest tests/ -q` reports `545 passed`, zero failures -- the `537`
  baseline re-measured live at planning time, plus eight new tests. The gate's
  own mechanism was re-run in both directions at planning time after the count
  changed: the pattern rejects the real 537-passed tail and accepts a 545-passed
  tail, and does not match `544`, `45` or `5450`.
- Each of the eight new tests names, in its docstring, the mutation that reddens
  it -- this module's stated convention -- and Task 3 ran six of those
  mutations and observed the predicted reddening.
- The refusal message's consequence clause is true of all three refusable window
  shapes, measured rather than assumed (APPD-02b), and is asserted identically
  in all three refusal tests.
- D-02 is locked twice: structurally on the parameter tuple and behaviourally on
  an overlapping append carrying an unrecognised kwarg. M6 demonstrated that the
  structural half alone stays green through a `**kwargs`-borne hatch.
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
  incoming start, and `save(mode="w")`, and its stated consequence is true of
  every window shape the guard refuses.
- `append()` offers no route past the refusal -- not a named parameter, and not
  one carried in `**kwargs`.
- A gap still appends. A no-coordinate append dim still appends.
- `widen_and_append` refuses with the identical message and holds no check of
  its own, and its docstring records the accepted half-widened side effect.
- `.planning/todos/pending/2026-09-07-guard-the-append-dim-against-overlapping-timestamps.md`
  is under `completed/` via `git mv`, `status: closed`, with a `## Closed` section.
</success_criteria>

<output>
Create `.planning/quick/260907-uac-refuse-an-append-that-overlaps-timestamp/260907-uac-SUMMARY.md` when done.
</output>
