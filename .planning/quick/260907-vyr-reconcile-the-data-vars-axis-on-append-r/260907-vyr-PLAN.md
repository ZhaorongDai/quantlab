---
phase: quick-260907-vyr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/dataset/backend.py
  - quantlab/base/factor.py
  - tests/test_variable_axis_widening.py
  - tests/test_factor_update.py
  - .planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md
autonomous: true
requirements: [DVAR-01, DVAR-02, DVAR-03, DVAR-04, DVAR-05, DVAR-06, DVAR-07, DVAR-08, DVAR-09, DVAR-10, DVAR-11, DVAR-12, DVAR-13]
estimate:
  tokens: 24000
  raw_tokens: 24000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "DVAR-01: `XrBackend.append()` RAISES `ValueError` before `to_zarr` when the incoming panel carries a data variable the store does not have. Re-measured live at planning time on the current tree: it raises NOTHING, and the store comes back with `alpha`(4,2), `beta`(2,2), `timestamp`=4 -- after which `xr.open_zarr()` fails with `ValueError: conflicting sizes for dimension 'timestamp': length 2 on 'beta' and length 4 on ...`. The store is not merely wrong, it is UNOPENABLE."
    - "DVAR-02: `XrBackend.append()` RAISES `ValueError` before `to_zarr` when the store holds a data variable the incoming panel lacks, and there is NO opt-in that permits it. Re-measured live: it raises nothing, `alpha` grows to (4,2) while `beta` stays STUCK at (2,2), and a previously VALID two-variable store becomes unopenable. This is the destructive row -- the only one of the three that destroys pre-existing good data -- and it is why the missing-variable case is refused unconditionally rather than offered a widening path. Filling the absent variable with NaN over the incoming window would punch holes into recent dates of a variable that was complete, and afterwards the store is indistinguishable from one where those values were genuinely missing. That is the same argument `widen_symbol_axis` already makes for refusing to DROP a stored label."
    - "DVAR-03: The two refusals carry DISTINCT messages and a decided precedence. The MISSING case is checked FIRST and names no opt-in, because a caller told about the widening opt-in while still being refused for dropping would be misled. The NEW case names the widening opt-in. Both name the store path and the offending variable names. Placement is AFTER the existing shared-variable dtype loop so no existing refusal's precedence changes -- and measured live at planning time, placement cannot flip an existing test either way: the STRICTEST possible variable-set refusal, injected into `_assert_append_compatible` and run against the whole suite, left 545 passed / 0 failed. No currently-green test appends a mismatched variable set at all. Precedence against that dtype loop is observable ONLY against a COMBINED fixture, and both placements were measured live at revision time: store `{alpha:float64, beta}` taking incoming `{alpha:float32, gamma}` raises the EXISTING dtype message when the variable check sits AFTER the loop, and the MISSING variable-set message when it sits above it. A fixture with IDENTICAL variable sets raises the dtype message in BOTH placements -- the relocated check has nothing to fire on -- so it cannot span this precedence at all."
    - "DVAR-04: There is NO opt-out, DECLARED or SMUGGLED. `XrBackend.append`'s signature stays exactly `(self, path, append_dim, **kwargs)`, AND no route past the variable refusal may be carried in `**kwargs`. The signature alone provably does not span this: quick task `260907-uac` demonstrated live (mutation M6) that a hatch popped from `**kwargs` inside the body makes the guard's `ValueError` disappear while `inspect.signature` still reports a byte-identical parameter tuple. The lock is therefore a PAIR of two SEPARATE test functions, never two assertions inside one: a single pytest function cannot be half-red, so a pair folded into one function cannot demonstrate the split it exists for. The structural half already EXISTS and is NOT duplicated here -- `tests/test_chunked_ingest.py:603` (`test_append_offers_no_overwrite_escape_hatch`) asserts this exact parameter tuple for this exact method at `:628`. This plan adds only the behavioural half -- a variable-mismatched append carrying an unrecognised kwarg still raises -- cross-referencing the existing structural test from its docstring, following that file's own `:603`/`:633` precedent."
    - "DVAR-05: `XrBackend.widen_data_vars()` adds a data variable to an existing store by materialising it over the store's EXISTING extent with the fill value and writing it with `to_zarr(mode=\"a\")`. Measured live: writing a new variable whose dims match the store exactly succeeds; writing one spanning only PART of the store's append axis raises and leaves the store INTACT (`['alpha']`, dims unchanged); and reindexing onto the store's full axis first then writing succeeds, giving the new variable NaN over history. The method is VARIABLE-NEUTRAL: it names no factor concept, so a market-data panel that grows a column later uses the same method."
    - "DVAR-06: The materialised filler carries the INCOMING variable's dtype, not an assumed float64. Measured live: a float64 filler against a float32 incoming variable is refused at the closing `append()` by the EXISTING shared-variable dtype guard; a dtype-matched filler appends cleanly. This is a real trap, not a defensive nicety -- KunQuant emits float32 as readily as float64."
    - "DVAR-07: The filler is written with an explicit `encoding` from `_append_encoding`, so the new variable joins the store on the SAME chunk grid as every other variable. Measured live with `APPEND_DIM_CHUNK` shrunk to 4 over a 10-timestamp store: an unencoded filler gets chunks `(10, 2)` -- the store's whole extent -- while `alpha` holds `(4, 2)`; passing `_append_encoding`'s computed `{'beta': {'chunks': (4, 2)}}` is ACCEPTED on `mode=\"a\"` and produces `(4, 2)`. The divergence does not crash the next append, which is exactly why it needs pinning by construction rather than by a test that waits for a failure."
    - "DVAR-08: A NON-FLOAT new variable is REFUSED without an explicit fill value, mirroring `widen_symbol_axis`'s refusal for the same reason. Measured live: `np.full((2,2), np.nan, dtype=int64)` yields `0`, and `dtype=bool` yields `True`. A NaN backfill of a boolean flag would mark every historical row as FLAGGED -- the identical 'fabricated observation where the data was missing' family the append dtype guard already refuses."
    - "DVAR-09: `widen_and_append()` reconciles ALL THREE axes and remains the ONE reconcile-then-append path -- extended, not duplicated by a second composed entry point. Order is symbol widen, then variable widen, then the UNCHANGED closing `append()`. Measured live end to end: store `{alpha}` x `[A,B]` x 3 dates taking `{alpha,beta}` x `[A,B,C]` x 2 later dates gives 5 timestamps x 3 symbols x 2 variables, timestamp unique AND strictly increasing, pre-existing `alpha` history bit-identical, `alpha` NaN = 3 (the new symbol over 3 historical dates) and `beta` NaN = 9 (3 symbols x 3 historical dates). It short-circuits to a plain `append()` when BOTH axes already agree, and an absent store still takes the single creation path its docstring promises."
    - "DVAR-10: `Factor.update()` is the AUTOMATIC data-update interface: it works out for itself what changed -- later dates, new symbols, new variables -- and reconciles each axis without the caller naming which widening to perform. It offers NO route to overwrite an already-stored range: it has no `mode` parameter and inherits the unconditional `append_dim` overlap refusal shipped by `260907-uac`. Overwriting is `save()`'s job; the two interfaces are how a caller expresses which they mean."
    - "DVAR-11: `Factor` gains a `_widen_fill_values()` seam, overridable per subclass, for the same reason `BaseDataset` has one -- `widen_symbol_axis` refuses to NaN-backfill a non-float variable without an explicit fill, and `Factor` is the shared base for factors AND classification labels. Measured live: the default is `{}` because every factor and label panel is float today (KunQuant emits float arrays, and even `SpotBinaryReturn` builds its binary label from `op.ConstantOp(1.0)`/`(0.0)`, so it is float64 rather than bool). The hook is required regardless of that default: without it a future non-float subclass has no way to widen at all, and `Factor` does not have this method today (verified live: `hasattr(Factor, '_widen_fill_values')` is False)."
    - "DVAR-12: `Factor.save()`'s BEHAVIOUR and its `mode=\"a\"` DEFAULT are unchanged. Only its prose moves: the two clauses asserting this method has no incremental route are now false and are re-pointed at `update()`. `tests/test_factor_save_mode.py`'s 5 tests stay green untouched."
    - "DVAR-13: The live baseline is preserved and grows only by the new tests. Measured at planning time on the current tree: `uv run pytest tests/ -q` -> 545 passed. Expected after this plan: 545 + 8 (Task 1) + 8 (Task 2) + 7 (Task 3) = 568. Each task GATES on its own running total (553 / 561 / 568) rather than merely printing it. The gate's mechanism was tested live in BOTH directions at revision time: the pattern REJECTS today's real tail (`545 passed, 154 warnings in 28.28s`), ACCEPTS a synthetic `568 passed, ...`, and matches none of 567, 569, 1568, 5680, 68, or `568 failed`. `tee /dev/stderr` keeps the real tail visible, so the gate adds a refusal without hiding the number the executor must still report."
  artifacts:
    - quantlab/dataset/backend.py
    - quantlab/base/factor.py
    - tests/test_variable_axis_widening.py
    - tests/test_factor_update.py
    - .planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md
  key_links:
    - "`XrBackend.append` -> `_assert_append_compatible(path, append_dim)` -- the single existing call site of the guard. The variable-set check rides it and needs no new wiring, exactly as the append-dim overlap check did."
    - "`XrBackend.widen_and_append` -> `widen_symbol_axis` -> `widen_data_vars` -> the UNCHANGED `XrBackend.append`. The closing `append()` is load-bearing: both axes now AGREE by construction, so `_assert_append_compatible` still runs and PASSES on its own terms rather than being bypassed. Any refactor that writes the window directly with `to_zarr(mode=\"a\")` from inside the composed method removes the guard from the widened path entirely."
    - "`XrBackend.widen_data_vars` -> `_append_encoding(append_dim, data=filler)` -- routing the filler's chunk shape through the SAME method `widen_symbol_axis` already uses is what keeps the `APPEND_DIM_CHUNK` rule single-sourced. Restating the chunk arithmetic inline is the drift this link exists to prevent."
    - "`Factor.update` -> `self.data_backend.widen_and_append(self.config.file_path, fill_values=self._widen_fill_values())` -- the factor layer's ONLY route to the store's incremental path, and the FIRST production caller `widen_and_append` has ever had (verified live: today it is referenced only from `tests/`)."
    - "`Factor._widen_fill_values` -> `widen_and_append(fill_values=...)` -> BOTH `widen_symbol_axis`'s reindex AND `widen_data_vars`' filler -- one seam feeding both widened axes, so a subclass carrying a non-float variable states its fill ONCE."
---

<objective>
Make the `data_vars` set a reconciled axis on the append path.

Three axes cross `XrBackend.append`. Two of them already have a rule and a
guard; the third has neither, and is the most destructive of the three.

| axis | rule | status |
|---|---|---|
| `append_dim` (timestamp) | append forward; overlap refused unconditionally | done (`260907-uac`) |
| widened dim (symbol) | incoming must be a SUPERSET; widen in place, NaN over history | done (`widen_symbol_axis`) |
| `data_vars` | incoming must be a SUPERSET; new variables materialised over the store's existing extent, NaN over history | **this plan** |

Root cause, in the same method and the same shape as the guards already there:
`_assert_append_compatible`'s variable loop opens with a skip on any name not
already in the store, so a NEW variable is never examined; and a variable the
store has but the incoming panel lacks is never visited at all, because the
loop iterates the INCOMING panel's variables.

This is strictly worse than the timestamp-overlap defect just fixed. That one
produced a readable store with wrong semantics. This one produces a store that
cannot be opened at all -- and in one shape it destroys a store that was valid
before the call.

Purpose: no append can leave a store unopenable, and a panel that legitimately
grew a variable has an explicit, named way to say so.
Output: a refusal, a variable-widening capability, and an automatic factor
update interface built on it.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/260907-uac-refuse-an-append-that-overlaps-timestamp/260907-uac-SUMMARY.md

@quantlab/dataset/backend.py
@quantlab/base/factor.py
@quantlab/base/data.py
@tests/test_symbol_axis_widening.py
</context>

<measured_evidence>
Everything below was re-derived live against the current tree at planning time.
None of it is inherited on trust.

**Baseline:** `uv run pytest tests/ -q` -> **545 passed**, 154 warnings, 26.57s.

**The defect, all three shapes, `append()` raising nothing in every one:**

| store vars | incoming vars | `append()` | on-disk shapes | `xr.open_zarr()` |
|---|---|---|---|---|
| `{alpha}` | `{alpha, beta}` | NO RAISE | `alpha`(4,2) `beta`(2,2) ts=4 | ValueError: conflicting sizes for dimension 'timestamp' |
| `{alpha, beta}` | `{alpha}` | NO RAISE | `alpha`(4,2) `beta`(2,2) ts=4 | ValueError: conflicting sizes for dimension 'timestamp' |
| `{alpha}` | `{beta}` | NO RAISE | both (2,2) ts=4 | ValueError: conflicting sizes for dimension 'timestamp' |

**Zarr CAN add a variable to an existing store:**

- new variable, dims matching the store exactly, `to_zarr(mode="a")` -> SUCCESS,
  both variables present, dims unchanged.
- new variable spanning only PART of the store's append axis -> RAISES
  `ValueError: variable 'timestamp' already exists with different dimension
  sizes`, and the store is left **INTACT** (`['alpha']`, dims unchanged). That
  failure is safe.
- reindex onto the store's FULL append axis, then `to_zarr(mode="a")` ->
  SUCCESS; new variable NaN over history (4 NaN), pre-existing variable
  untouched (0 NaN).

**The three-axis composition, end to end:** store `{alpha}` x `[A,B]` x 3 dates
taking `{alpha,beta}` x `[A,B,C]` x 2 later dates -> `{timestamp: 5, symbol: 3}`,
vars `['alpha','beta']`, timestamp unique AND strictly increasing, `alpha` NaN=3,
`beta` NaN=9, pre-existing history bit-identical.

**Two traps the composition hides, both measured, both NEW findings beyond the
brief:**

1. *Filler dtype.* filler float64 / incoming beta float32 -> the closing
   `append()` is REFUSED by the existing dtype guard. filler float32 / incoming
   float32 -> OK. filler float64 / incoming float64 -> OK.
2. *Chunk grid.* With `APPEND_DIM_CHUNK` shrunk to 4 over a 10-timestamp store:
   unencoded filler -> `beta` chunks `(10, 2)` while `alpha` holds `(4, 2)`; the
   next append still succeeds, so nothing surfaces. Passing `_append_encoding`'s
   `{'beta': {'chunks': (4, 2)}}` on `mode="a"` is ACCEPTED and yields `(4, 2)`.

**Non-float fillers fabricate data:** `np.full((2,2), np.nan, dtype=int64)` ->
`[0 0]`. `np.full((2,2), np.nan, dtype=bool)` -> `[True True]`.

**The refusal reddens nothing.** The strictest possible variable-set refusal
(raise on ANY mismatch, either direction) injected into
`_assert_append_compatible` and run against the whole suite: **545 passed**.

**The Task 2 RED will be behavioural, not a setup error.** With that same probe
injected, `widen_and_append` on a variable-grown panel raised
`ValueError: PROBE var-set mismatch: new=['beta'] missing=[]`, and the store was
left carrying the GROWN symbol axis `['A','B','C']` with vars still `['alpha']`
-- the half-widened side effect `widen_and_append`'s docstring already records
as accepted.

**Factor-layer facts:** `hasattr(Factor, '_widen_fill_values')` -> **False**.
`widen_and_append` has **no production caller** today (referenced only from
`tests/`); `widen_symbol_axis` is called from `quantlab/base/data.py:682`.
`SpotBinaryReturn` builds its label from `op.ConstantOp(1.0)`/`(0.0)`, so the
"binary" label is float, not bool. Suite sizes: `test_symbol_axis_widening.py`
11 tests, `test_factor_save_mode.py` 5, `test_factor_hierarchy.py` 11.

**Precedence is observable ONLY against a COMBINED fixture.** Measured at
revision time with the same probe injected at BOTH candidate positions, the
source restored from a backup between runs:

| fixture (same symbols, later window) | check AFTER the dtype loop (D-03) | check ABOVE it (M4) |
|---|---|---|
| store `{alpha:f64, beta}` <- incoming `{alpha:f32, gamma}` | **dtype** message | **MISSING** var-set message |
| store `{alpha:f64}` <- incoming `{alpha:f32}` (identical sets) | dtype message | dtype message |

The second row is the measured reason an identical-set dtype fixture cannot
span D-03: the relocated check has nothing to fire on, so NOTHING reddens and a
green run there proves nothing. Only the first row flips.

**Two more precedence facts, measured in the same runs:**

1. `_assert_append_compatible`'s existing order is: the non-append-dim
   coordinate loop, then the append_dim OVERLAP check, then the shared-variable
   dtype loop. A new-variable panel on an OVERLAPPING window therefore raises
   the OVERLAP message -- measured -- whether or not the composed path
   reconciled the variable axis first.
2. `widen_and_append`'s short-circuit TODAY tests the symbol axis alone, which
   is exactly mutation M9's shape, so M9 can be measured before the code exists.
   Through it: a panel whose symbols AGREE and whose variables GROW takes the
   fast path and is refused at the closing `append()` (`new=['beta']`). A panel
   whose symbols GROW does not take the fast path at all.

**`widen_and_append`'s callers, re-counted live: FIVE, not four.**
`tests/test_symbol_axis_widening.py:91, :129, :367, :395, :415`. All five build
their panels with that file's `_panel` helper, which carries the single variable
`close`, so every one has an AGREEING variable set and `widen_data_vars` is a
NO-OP on all five. Three of them (`:91`, `:129`, `:415`) grow or reindex the
SYMBOL axis and therefore do NOT take the short-circuit -- they run the full
widen path and stay green because the variable widen is a no-op inside it, not
because they skip it. `:367` (equal axis) and `:395` (absent store) never reach
it.

**The pass-count gate's own mechanism, tested in both directions:**
`... | tail -2 | tee /dev/stderr | grep -qE '(^|[^0-9])568 passed'` REJECTS
today's real tail (`545 passed, 154 warnings in 28.28s`), ACCEPTS a synthetic
`568 passed, ...`, and matches none of `567`, `569`, `1568`, `5680`, `68`, or
`568 failed`. The `tee` keeps the real tail on the executor's screen.

Both probe rounds were reverted from the backup; `git diff --quiet
quantlab/dataset/backend.py` passed afterwards, and `git status --porcelain`
showed only the pre-existing ` M test.py`, which this plan does not touch at any
point.
</measured_evidence>

<decisions>
- **D-01 — the `data_vars` set follows the SUPERSET rule**, for the same reason
  the symbol axis does. `append()` refuses ANY mismatch, in BOTH directions,
  unconditionally.
- **D-02 — two distinct messages, MISSING checked first.** The missing case
  names no opt-in (there is none); the new-variable case points at the widening
  opt-in.
- **D-03 — the check is placed AFTER the existing shared-variable dtype loop**,
  so no existing refusal's precedence changes. Measured: placement cannot flip
  an existing test in any case, since no currently-green test appends a
  mismatched variable set. It is observable only against a fixture carrying
  BOTH a dtype mismatch on a shared variable AND a variable-set mismatch --
  measured in both placements, that fixture flips its message while an
  identical-variable-set fixture does not move at all. Task 1 test 7 is that
  fixture; it is what makes D-03 an enforced decision rather than a stated one.
- **D-04 — the new method is `widen_data_vars`.** Variable-neutral: it names the
  xarray concept it operates on and cannot be misread as a dimension. It must
  not mention factors -- "factor" is the caller's vocabulary, never the storage
  layer's.
- **D-05 — extend `widen_and_append`, do not add a second composed entry point.**
  Its docstring already establishes it as the ONE reconcile-then-append path,
  including "an absent store does the same, so there is ONE creation path rather
  than two". Folding the variable axis in completes that meaning rather than
  changing its contract. Weighed against its existing dedicated tests, re-counted
  live at revision time: FIVE call it, not four. All five build single-variable
  `close` panels, so their variable sets already AGREE and `widen_data_vars` is a
  NO-OP on every one of them -- that, and not the short-circuit, is what keeps
  them green. Three (`:91`, `:129`, `:415`) grow or reindex the symbol axis and
  run the FULL widen path; only `:367` short-circuits and only `:395` takes the
  creation path. Verified by the probe run above.
- **D-06 — symbol widen FIRST, then variable widen, then append.** The filler is
  built over the store's extent AFTER the symbol widen, so it is materialised
  once at the final width rather than written narrow and rewritten. This is the
  order measured end to end.
- **D-07 — the filler carries the incoming variable's dtype** and is written with
  `_append_encoding`. Both are corrections the composition silently needs; see
  the two traps above.
- **D-08 — a non-float new variable is refused without an explicit fill**,
  mirroring `widen_symbol_axis`.
- **D-09 — `Factor` gets TWO interfaces.** `save()` = first save / overwrite,
  unchanged. `update()` = automatic incremental update, no overwrite route.
- **D-10 — `save()`'s `mode="a"` default is NOT changed here**, deliberately.
  Whether it is now vestigial is a question for the developer, filed as a todo.
- **D-11 — rolling-window warm-up stays OUT of scope.** Whether a value was
  computed with enough lookback is a property of `cal()`. The storage layer
  cannot distinguish a warm-up artifact from a genuine low value, and a
  `config.window`-derived trim inside the write path was proposed and rejected
  on 2026-09-07 as a heuristic wearing a guard's clothing. Add no trim anywhere.
- **D-12 — tracer ordering is deliberately inverted.** The refusal ships FIRST,
  before the capability it protects, so there is no window in which the widening
  exists unguarded. Task 1 is still a vertical slice: it wires the property
  end-to-end through BOTH public entry points with a runnable end-to-end verify.
</decisions>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Refuse an append whose data_vars set differs from the store's</name>
  <files>quantlab/dataset/backend.py, tests/test_variable_axis_widening.py</files>
  <reversibility rating="reversible">A guard with a single call site, removable in one edit; nothing depends on its presence structurally.</reversibility>
  <read_first>
    quantlab/dataset/backend.py lines 378-450 (`_assert_append_compatible`, all
    three existing checks and their ordering), lines 43-100 (`append`, and the
    docstring paragraph enumerating the enforced properties), lines 365-377
    (`_format_append_label`, the precedent for a small message helper).
    tests/test_symbol_axis_widening.py lines 415-470 (the inherited-refusal test
    shape and the byte-identical-message technique it uses).
    tests/test_chunked_ingest.py lines 603-660 --
    `test_append_offers_no_overwrite_escape_hatch` and
    `test_append_refuses_an_overlapping_window_carrying_an_unrecognised_kwarg`:
    the structural/behavioural PAIR this plan REUSES rather than duplicates,
    the reason each is its own function, and the docstring convention each half
    uses to point at the other.
  </read_first>
  <behavior>
    Write these FIRST, in a NEW file `tests/test_variable_axis_widening.py`, and
    observe RED before touching the backend. Per D-01/D-02/D-03/D-04:

    - Test 1 (tracer, end to end): store `{alpha}` on two dates, incoming
      `{alpha, beta}` on two later dates. `append()` raises `ValueError`. Assert
      the store is still openable afterwards and still holds exactly `{alpha}`
      with its original extent -- the refusal must fire BEFORE `to_zarr`, and an
      intact store is the only proof of that which the message cannot fake.
    - Test 2: store `{alpha, beta}`, incoming `{alpha}` -> raises. This is the
      destructive shape; assert the store is intact and `beta` still carries its
      original extent.
    - Test 3: fully disjoint, store `{alpha}` / incoming `{beta}` -> raises.
      Deliberately MESSAGE-AGNOSTIC: assert only that it raises and that the
      store is intact. Pinning a message here would make M3's precedence swap
      redden two tests instead of one and blur which branch caught it.
    - Test 4: the two messages are DISTINCT and each names the offending
      variable names and the store path. The new-variable message names the
      widening opt-in; the missing-variable message does not. Use one PURE
      fixture per message (`{alpha}` -> `{alpha, beta}` and
      `{alpha, beta}` -> `{alpha}`), never the mixed one, so a precedence swap
      cannot move this test.
    - Test 5: precedence -- store `{alpha, beta}`, incoming `{alpha, gamma}`
      (both a drop AND an addition) raises the MISSING message, not the new one.
    - Test 6: identical variable sets still append normally -- the positive
      control that the new check adds no refusal to the path every other caller
      already uses.
    - Test 7 (D-03, and the ONLY fixture that spans it): store
      `{alpha: float64, beta: float64}` taking incoming
      `{alpha: float32, gamma: float64}` on a later window with the SAME symbol
      axis -- a dtype mismatch on a SHARED variable AND a variable-set mismatch
      at once. Assert the EXISTING dtype message fires, naming `alpha`, and that
      NEITHER variable-set message appears. Measured live in both placements at
      revision time: with the new check after the dtype loop this fixture raises
      the dtype message; moved above it, the same fixture raises the MISSING
      message instead. An identical-variable-set fixture raises the dtype
      message in BOTH placements and therefore proves nothing about precedence
      -- also measured, which is why it is not the fixture used here.
    - Test 8 (the BEHAVIOURAL half of D-04, as its OWN function): a
      variable-mismatched append -- store `{alpha}`, incoming `{alpha, beta}` --
      carrying an unrecognised kwarg (`force=True`) still raises `ValueError`
      from the guard, and the store is left intact. Pin the NEW-variable
      direction so a change to one branch of the refusal cannot move this test
      sideways.

      Its STRUCTURAL half is NOT rewritten here. It already exists as
      `tests/test_chunked_ingest.py::test_append_offers_no_overwrite_escape_hatch`
      (`:603`), which asserts this exact parameter tuple for this exact method
      at `:628`; a verbatim copy would give two tests that redden together on
      the same drift, which is noise rather than a split. Cross-reference it
      from this test's docstring the way that file's own pair does, and state
      what this test adds over its neighbour `:633`: `:633` locks the OVERLAP
      refusal against a smuggled kwarg, this one locks the VARIABLE-SET
      refusal, and a bypass added to one branch alone would leave the other
      green.

    These are EIGHT separate test functions, and tests 6, 7 and 8 are
    deliberately not folded into fewer. A single pytest function cannot be
    half-red, so a mutation whose whole point is that one assertion moves while
    another holds cannot be OBSERVED inside one function. That is the precedent
    `tests/test_chunked_ingest.py:603`/`:633` set in this repo, for this method,
    for this reason -- their docstrings say so outright.

    Expected RED, measured at planning time: every raising test fails with
    `Failed: DID NOT RAISE ValueError`. That is a behavioural RED. Test 7 is the
    exception and must be read differently: it raises today too, with the dtype
    message, so it is GREEN before the change and stays green after -- it is a
    precedence lock, not a RED-first test, and its value is entirely in what M4
    does to it. A RED that fails on import, collection or fixture setup proves
    nothing and must be fixed before proceeding.
  </behavior>
  <action>
    Add ONE variable-set check inside
    `quantlab/dataset/backend.py::XrBackend._assert_append_compatible`, placed
    AFTER the existing shared-variable dtype loop (D-03).

    Compare the incoming panel's `data_vars` key set against the store's. Check
    the MISSING direction first (store has, incoming lacks), then the NEW
    direction (incoming has, store lacks). Raise `ValueError` on either.

    Both messages follow the shape the three sibling refusals already set: name
    the store path, name the offending variable names sorted for reproducibility,
    state what zarr would do instead, and state the remedy.

    The MISSING message's stated consequence is measured, not chosen: the
    variable the incoming panel lacks stays at its old length while every other
    variable grows, so the store afterwards cannot be opened at all -- and the
    data destroyed was valid before the call. Its remedy is to recompute the
    full variable set, or replace the store wholesale; it names NO widening
    opt-in, because none exists and offering one would be a lie.

    The NEW message's remedy names the explicit widening opt-in that Task 2
    ships. Word it so it stays true before Task 2 lands as well as after --
    describe the opt-in by what it does rather than by promising a method that
    does not exist yet is unnecessary, since Task 2 lands in the same plan; name
    the method.

    Then extend `append()`'s docstring paragraph that enumerates the enforced
    properties: the data-variable set is now the fourth, and the SUPERSET rule
    and the reason the missing direction has no escape hatch belong there beside
    the append-dim rule that already states its own.

    Do not touch `append()`'s signature. Do not add a flag, a mode, or any
    bypass-shaped parameter, and do not read anything out of `**kwargs` ahead of
    the guard call -- the guard runs before `kwargs` is touched at all, and that
    ordering is the mechanism D-04 depends on.

    Keep that prohibition OUT of the source. This task's gate counts
    bypass-shaped identifiers in `backend.py` with word boundaries and requires
    ZERO (measured: zero today), so a docstring sentence that names one in order
    to disown it would fail the gate on its own words. `enforced` and `refactor`
    are safe -- the boundaries were measured at planning time -- but the bare
    identifiers are not. Say what the method DOES accept; say nothing about what
    it refuses to be given.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_variable_axis_widening.py -q && uv run pytest tests/ -q 2>&1 | tail -2 | tee /dev/stderr | grep -qE '(^|[^0-9])553 passed' && test "$(grep -cF 'def append(self, path: str, append_dim: str = "timestamp", **kwargs) -> Self:' quantlab/dataset/backend.py)" -eq 1 && test "$(grep -vE '^[[:space:]]*#' quantlab/dataset/backend.py | grep -cE '\b(force|overwrite_ok|allow_mismatch|allow_var_mismatch)\b')" -eq 0 && test "$(grep -oE '`(base|dataset|factor|label|utils|enums|my_ops|dl_model|ml_model|vecbt)/[A-Za-z_]+\.py' quantlab/dataset/backend.py | wc -l | tr -d ' ')" -le 1</automated>
  </verify>
  <done>
    All 8 new tests pass. The full suite GATES on 553 passed (545 live baseline
    + 8) and the executor states the REAL number it saw. Both refusal messages
    are reachable and distinct. The combined fixture raises the DTYPE message,
    not a variable-set one. `XrBackend.append`'s parameter tuple is unchanged --
    locked by the pre-existing structural test at
    `tests/test_chunked_ingest.py:603`, which stays green untouched, not by a
    duplicate. The backticked pre-migration path count in `backend.py` has not
    risen above its measured baseline of 1.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add variable widening and fold it into widen_and_append</name>
  <files>quantlab/dataset/backend.py, tests/test_variable_axis_widening.py</files>
  <reversibility rating="costly">`widen_data_vars` is a new public method name on the storage layer; renaming it after callers exist means touching every call site and its cross-referencing docstrings. The name is decided in D-04 rather than left to execution.</reversibility>
  <read_first>
    quantlab/dataset/backend.py lines 106-263 (`widen_symbol_axis` in full --
    its guard ordering, its superset refusal, its non-float refusal and the
    `fill_values` remedy prose, and the docstring paragraph arguing why widening
    is a separately named method rather than a flag that loosens the refusal),
    lines 265-335 (`widen_and_append`, especially the short-circuit and the
    paragraph on the load-bearing closing `append()`), lines 336-364
    (`_append_encoding` and its `data=` parameter).
  </read_first>
  <behavior>
    Write these FIRST and observe RED. Per D-05 through D-08:

    - Test 1 (tracer, end to end, through `widen_and_append`): store `{alpha}` x
      `[A,B]` x 3 dates, incoming `{alpha,beta}` x `[A,B,C]` x 2 later dates.
      Result: 5 timestamps, 3 symbols, 2 variables; timestamp unique AND
      strictly increasing; `alpha` NaN count 3; `beta` NaN count 9; the
      pre-existing `alpha` history bit-identical to what was there before.
      Measured RED at planning time: this fails with the Task 1 MISSING/NEW
      `ValueError` raised through the composed path -- a behavioural RED firing
      through exactly the mechanism this task removes, not an `AttributeError`
      from a missing method. Its SYMBOL axis grows, which is why M9 leaves this
      test GREEN -- the symbol-only short-circuit is not taken and the widen
      still runs. Test 3, not this one, is the M9 canary.
    - Test 2: `widen_data_vars` called directly adds the variable with the fill
      over the store's whole existing extent and leaves every pre-existing
      variable bit-identical.
    - Test 3 (D-06, and the M9 canary): through `widen_and_append`, with the
      SYMBOL axis already agreeing -- `[A,B]` on both sides -- and only the
      variable set growing: store `{alpha: float64}` x 3 dates, incoming
      `{alpha: float64, beta: float32}` x 2 later dates. The composed path
      succeeds and `beta` is float32 on disk. RED under a hardcoded float64
      filler, which the existing dtype guard refuses. ALSO red under M9, and
      that is measured rather than predicted: run against today's
      still-symbol-only short-circuit at revision time, this exact shape takes
      the fast path and is refused at the closing `append()` with
      `new=['beta']`. Keeping the symbol axis equal here is load-bearing, not
      incidental -- it is what makes M9 observable at all.
    - Test 4 (D-07): with `APPEND_DIM_CHUNK` monkeypatched small enough that the
      store exceeds it, the new variable's on-disk chunks EQUAL the pre-existing
      variable's. RED under an unencoded filler write. Read chunks from the zarr
      array directly -- xarray does not surface this. Call `widen_data_vars`
      DIRECTLY: the encoding is that method's property, and going direct keeps
      this test independent of `widen_and_append`'s short-circuit, so M9 cannot
      move it.
    - Test 5 (D-08): a non-float new variable with no `fill_values` entry is
      REFUSED, and the message names the variable, its dtype, and the
      `fill_values` remedy. Assert the store is intact after the refusal. Via
      `widen_data_vars` directly -- the guard belongs to that method, and the
      direct entry point keeps M9 out of this test too.
    - Test 6 (D-08, the positive half): the same non-float variable WITH an
      explicit `fill_values` entry succeeds and preserves the dtype exactly on
      disk -- no float64 upcast. Same direct entry point as test 5.
    - Test 7 (D-05): `widen_and_append` with variable sets and symbol axes that
      already agree still takes the plain-`append` short-circuit -- no store
      rewrite, no filler write. Assert it by monkeypatching BOTH
      `widen_symbol_axis` AND `widen_data_vars` to raise, the way
      `tests/test_symbol_axis_widening.py:367` already does for the symbol half
      alone. Stays green under M9, which only widens what this test already
      asserts is skipped.
    - Test 8 (D-05): `widen_and_append` still inherits the append_dim overlap
      refusal after the variable reconciliation runs, byte-identically to plain
      `append()` once the store path is normalized out. This is the guarantee
      that the composed path did not grow a parallel write. Measured at revision
      time: the overlap check sits AHEAD of the variable-set check inside
      `_assert_append_compatible`, so this message equality holds whether or not
      the variable axis was reconciled first -- which is precisely why this test
      is NOT an M9 canary and stays green under it.
  </behavior>
  <action>
    Add `XrBackend.widen_data_vars(self, path, variables, append_dim="timestamp",
    fill_values=None)` to `quantlab/dataset/backend.py`, placed beside
    `widen_symbol_axis`.

    `variables` is a mapping of variable name to the incoming `xr.DataArray` or
    to its dtype -- choose the narrower of the two that lets the method
    materialise a correctly-typed filler without holding the incoming panel, and
    say which in the docstring. It opens the store, computes which named
    variables are absent, and returns without writing when none are.

    For each absent variable it materialises an array over the store's EXISTING
    coordinate extent on every dimension, filled with the `fill_values` entry if
    one is named and NaN otherwise, carrying the INCOMING variable's dtype
    (D-06). It writes them in ONE `to_zarr(mode="a")` call with
    `encoding=self._append_encoding(append_dim, data=filler)` (D-07).

    Guards fire before any write, mirroring `widen_symbol_axis`'s order and its
    message shape:
    1. no store at `path` -> `FileNotFoundError`, same as its sibling.
    2. a named variable that is absent AND not floating-point AND not named in
       `fill_values` -> refuse. State the measured reason: a NaN fill of an
       integer array becomes zero and a NaN fill of a boolean array becomes
       true, so an unfilled backfill FABRICATES history rather than marking it
       absent -- the same family of invisible corruption the append dtype guard
       refuses. Name the `fill_values` remedy.

    The docstring must be VARIABLE-NEUTRAL (D-04): it describes a stored panel
    that grew a column, in storage vocabulary only. Say nothing about the
    caller's domain -- and say nothing about not saying it, either: this task's
    gate counts that domain word in `backend.py` and allows only the 2
    pre-existing hits, so a docstring clause disowning the vocabulary would
    spend the very budget it exists to protect. Carry across
    `widen_symbol_axis`'s argument for why this is a separately named opt-in
    rather than a flag that loosens the refusal, and cross-reference the Task 1
    refusal it protects a non-opting caller from. Note that unlike its sibling
    this needs NO directory swap: the measured behaviour is that a partial-extent
    write raises and leaves the store intact, so the operation is already safe to
    retry.

    Then extend `widen_and_append` (D-05): after the existing symbol-axis
    reconciliation and before the closing `append()`, call `widen_data_vars`
    with the incoming panel's variables and the same `fill_values` (D-06 order).
    Extend its short-circuit so the plain-`append` fast path is taken only when
    BOTH the symbol axis AND the variable set already agree. The closing
    `append()` stays exactly as it is: both axes now agree by construction, so
    `_assert_append_compatible` runs and passes on its own terms.

    Extend `widen_and_append`'s docstring to state that it now reconciles three
    axes, that the order is symbol then variable then append, and that the
    accepted half-widened side effect it already documents now spans the variable
    axis too. Do not add a second composed entry point.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_variable_axis_widening.py tests/test_symbol_axis_widening.py tests/test_chunked_ingest.py -q && uv run pytest tests/ -q 2>&1 | tail -2 | tee /dev/stderr | grep -qE '(^|[^0-9])561 passed' && test "$(grep -vE '^[[:space:]]*#' quantlab/dataset/backend.py | grep -cEi '\b(factor|alpha101|alpha158|kunquant)')" -le 2 && test "$(grep -c 'def widen_data_vars' quantlab/dataset/backend.py)" -eq 1</automated>
  </verify>
  <done>
    All 8 new tests pass and the 11 pre-existing `test_symbol_axis_widening.py`
    tests plus the 36 `test_chunked_ingest.py` tests stay green untouched. The
    full suite GATES on 561 passed (553 + 8); the executor states the REAL
    number it saw.
    `widen_data_vars` exists exactly once. The storage layer's factor-vocabulary
    count has not risen above its measured baseline of 2 -- both pre-existing
    hits are the `FactorPolars` reference in `head()`'s docstring, which is a
    caller-bug narrative and is left alone; the point of the gate is that
    `widen_data_vars` adds none.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Give Factor an automatic update interface built on the reconciled path</name>
  <files>quantlab/base/factor.py, tests/test_factor_update.py, .planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md</files>
  <reversibility rating="costly">`Factor.update` is a new public interface on the shared factor base; both concrete backends and every future subclass inherit it, so the name and the no-overwrite contract are decided in D-09 rather than left to execution.</reversibility>
  <read_first>
    quantlab/base/factor.py lines 21-60 (the `Factor` class docstring's
    backend-agnostic contract, `__init__`'s load-bearing ordering, and
    `_auto_filter`), lines 126-171 (`read` and `save` in full, including the
    docstring paragraph on what zarr's `"a"` mode means and the wrapped
    `ValueError`). quantlab/base/data.py lines 613-631
    (`BaseDataset._widen_fill_values` -- the seam precedent, its argument for
    being a seam rather than a constant, and its "honest value" reasoning).
    tests/test_factor_hierarchy.py (the fixture idiom for constructing a `Factor`
    subclass without a real dataset on disk). tests/test_chunked_ingest.py lines
    603-660 -- the structural/behavioural PAIR, two separate functions, that
    tests 4 and 5 below copy the SHAPE of (not the assertion: that one is about
    `XrBackend.append`, these are about `Factor.update`).
  </read_first>
  <behavior>
    Write these FIRST and observe RED. Per D-09/D-10:

    - Test 1 (tracer): a factor whose store already holds an earlier date range;
      `update()` appends a later range and the store afterwards holds both,
      timestamp unique and strictly increasing, and the earlier values
      bit-identical.
    - Test 2: `update()` with a panel carrying a NEW symbol reconciles the symbol
      axis automatically -- the caller names no widening.
    - Test 3: `update()` with a panel carrying a NEW factor/variable reconciles
      the variable axis automatically. RED before Task 2 exists; after it, this
      is the property that proves `update()` actually routes through the
      three-axis path rather than a plain append.
    - Test 4 (D-10, the STRUCTURAL half, its OWN function): `Factor.update`'s
      parameter tuple carries no `mode` and no force-shaped parameter. Its
      docstring must say that this assertion is a PROXY that measurably does NOT
      span the property it stands for -- quick task `260907-uac` demonstrated
      live that a hatch popped from `**kwargs` inside a method body leaves the
      parameter tuple byte-identical -- and must name test 5 as the half that
      does span it.
    - Test 5 (D-10, the BEHAVIOURAL half, its OWN function, and the
      load-bearing negative): `update()` on a range the store ALREADY holds
      RAISES, inheriting the unconditional overlap refusal, and still raises
      when the same call carries an unrecognised kwarg (`force=True`). Test 4
      alone does not span this; this one alone does not span a NAMED parameter.

      Tests 4 and 5 are two functions rather than two assertions in one, for the
      measured reason the repo's own pair at `tests/test_chunked_ingest.py:603`
      and `:633` is two: a single pytest function cannot be half-red, so the
      split M10 predicts -- structural red, behavioural green -- would be
      unobservable inside one function.
    - Test 6 (DVAR-11): `Factor._widen_fill_values()` exists on the base,
      returns `{}` by default, and a subclass overriding it has that mapping
      reach the widening call. Assert the reach behaviourally -- a subclass
      carrying a non-float variable and overriding the seam must widen
      successfully where one that does not override is refused. A test that only
      asserts the method exists and returns `{}` does not span the seam's
      purpose.
    - Test 7 (DVAR-12): `save()`'s default is still `"a"` and its wrapped
      `ValueError` still fires on a differently-sized second range. This is a
      no-change lock; it must redden if `save()`'s behaviour drifts.
  </behavior>
  <action>
    Add `Factor.update(self, **kwargs)` to `quantlab/base/factor.py`, beside
    `save()`.

    It runs `self._auto_filter()` exactly as `save()` does -- the same
    normalization must apply to both write paths -- then delegates to
    `self.data_backend.widen_and_append(self.config.file_path,
    fill_values=self._widen_fill_values(), **kwargs)`.

    It takes NO `mode` parameter and offers no route to overwrite an
    already-stored range (D-10). The automatic reconciliation is entirely the
    backend's: `update()` names no axis, and the caller does not choose which
    widening to perform. Wrap it in the same `Timer` idiom `save()` uses.

    Add `Factor._widen_fill_values(self) -> dict` returning `{}`, with the seam
    argument `BaseDataset._widen_fill_values` already makes: the widening
    refuses to backfill a non-float variable without an explicit fill, and this
    base is shared by factors AND classification labels. Record the measured
    reason the default is empty rather than the sibling's `{'anomaly_flag':
    False}` -- every factor and label panel is float today, including the binary
    label, which is built from float constants. The seam exists for the subclass
    that is not.

    Then re-point the two stale clauses in `save()` that tell the reader this
    method has no incremental route -- the Chinese one closing the docstring's
    second paragraph, and the English one closing the raised `ValueError`
    message. Both assert an absence that `update()` now fills. Replace each with
    a pointer naming `update()` as the incremental path and `save(mode="w")` as
    the wholesale one, and state the split the two interfaces express: `save()`
    writes wholesale, `update()` extends. Do not restate the retired claim
    alongside its replacement.

    Change nothing else about `save()` -- not its default, not its signature,
    not the wrapped exception's `__cause__` chaining, not the substring it
    matches on (D-10 forbids the default change; `tests/test_factor_save_mode.py`
    locks the rest).

    Finally, file the follow-up question as a pending todo at
    `.planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md`,
    following the frontmatter shape of the existing pending todos
    (`created`, `title`, `area`, `severity`, `status: pending`). Its content: now
    that an explicit update interface exists, `save()`'s `"a"` default may be
    vestigial, since it is the mode that essentially never does what a second
    write wants; changing a default is a behaviour change for existing callers,
    so it is the developer's call and was deliberately not made here. This is
    surfaced, not decided.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_factor_update.py tests/test_factor_save_mode.py tests/test_factor_hierarchy.py tests/test_factor_kunquant.py tests/test_factor_polars.py -q && uv run pytest tests/ -q 2>&1 | tail -2 | tee /dev/stderr | grep -qE '(^|[^0-9])568 passed' && test "$(grep -c 'is not wired to' quantlab/base/factor.py)" -eq 0 && test "$(grep -c '没接过去' quantlab/base/factor.py)" -eq 0 && test "$(grep -c 'def update' quantlab/base/factor.py)" -eq 1 && test "$(grep -c 'def _widen_fill_values' quantlab/base/factor.py)" -eq 1 && test "$(grep -c 'mode: Literal\["a", "w"\] = "a"' quantlab/base/factor.py)" -eq 1 && test -f .planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md && test "$(grep -oE '`(base|dataset|factor|label|utils|enums|my_ops|dl_model|ml_model|vecbt)/[A-Za-z_]+\.py' quantlab/base/factor.py | wc -l | tr -d ' ')" -le 1</automated>
  </verify>
  <done>
    All 7 new tests pass; `test_factor_save_mode.py`'s 5 and
    `test_factor_hierarchy.py`'s 11 stay green untouched. The full suite GATES
    on 568 passed (561 + 7); the executor states the REAL number it saw.
    `save()`'s signature line with its `"a"` default is present verbatim and
    unmodified.
    Both retired clauses are gone from `quantlab/base/factor.py`. The todo exists
    under `pending/`. The backticked pre-migration path count in `factor.py` has
    not risen above its measured baseline of 1.
  </done>
</task>

</tasks>

<mutation_verification>
This repo has hit "a gate that passes for the wrong reason" repeatedly over the
past two days -- the tally on 2026-09-07 stands at seven -- and the most recent
was a structural proxy that could not span the behavioural property it stood
for, caught only because a mutation demonstrated the split. Every lock below is
therefore mutation-verified: apply the mutation to a clean tree, confirm the
predicted test(s) redden THROUGH THE ASSERTION THEY CLAIM TO EXERCISE (report
the failing line and the failure mode, not just the count), revert, and confirm
`git diff --quiet` on the mutated file.

**How a SPLIT is observed.** Where a mutation predicts one test reddening while
its partner stays green (M5, M10), that pair is TWO SEPARATE TEST FUNCTIONS. A
single pytest function cannot be half-red, so a pair folded into one function
makes the split unobservable -- which is why the repo's own pair at
`tests/test_chunked_ingest.py:603`/`:633` is two functions, as their docstrings
state outright. Observing a split means reporting BOTH results by test NAME from
the same run: the red one with its failing line, the green one as still passing.

**Two mutations flip a MESSAGE rather than a raise** (M3, M4). A test that only
asserts `pytest.raises(ValueError)` cannot observe either one. The predicted
tests assert on message content, and the report must quote the message actually
raised.

| Mut | Change | Predicted |
|-----|--------|-----------|
| M1 | delete the whole variable-set check | Task 1 tests 1, 2, 3, 4, 5 and 8 redden together via `DID NOT RAISE ValueError`. Tests 6 and 7 stay GREEN -- 6 is the positive control, and 7's dtype message comes from the untouched existing guard |
| M2 | check ONLY the new-variable direction (drop the missing branch) | EXACTLY Task 1 tests 2, 4 and 5 redden; tests 1, 3 and 8 stay green. Test 3's incoming `{beta}` is ALSO missing `alpha`, so a GREEN test 3 under M2 is the evidence that the disjoint case is being caught by the new-variable branch -- which is why test 3 is deliberately message-agnostic |
| M3 | swap the precedence -- check NEW before MISSING | EXACTLY Task 1 test 5 reddens, on its message assertion: it receives the NEW message where the MISSING one is required. Nothing else moves -- test 2 is pure-missing, test 4 uses one PURE fixture per message, and test 3 asserts no message at all |
| M4 | move the variable-set check ABOVE the shared-variable dtype loop | EXACTLY Task 1 test 7 reddens, on its message assertion: its COMBINED fixture (`{alpha:f64, beta}` <- `{alpha:f32, gamma}`) raises the MISSING variable-set message where the DTYPE message is required. Measured in BOTH placements at revision time, so this is an observed flip rather than a reasoned one. Test 6 and every identical-variable-set fixture stay GREEN, and that is the CORRECT outcome rather than a miss: with matching sets the relocated check has nothing to fire on, which is exactly why an identical-set fixture cannot span D-03 |
| M5 | hatch popped from `**kwargs` inside `append`'s body, ahead of the guard call | A SPLIT across three named functions: Task 1 test 8 (new, behavioural) reddens via `DID NOT RAISE`; `tests/test_chunked_ingest.py::test_append_refuses_an_overlapping_window_carrying_an_unrecognised_kwarg` (`:633`) reddens with it; and `tests/test_chunked_ingest.py::test_append_offers_no_overwrite_escape_hatch` (`:603`, the structural half this plan reuses instead of duplicating) stays GREEN. Report all three by name. A run in which `:603` also reddens means the structural assertion is not what this plan believes it to be |
| M6 | build the filler as float64 unconditionally | EXACTLY Task 2 test 3 reddens, via the EXISTING append dtype refusal, not a new one. Confirm the message is the dtype guard's |
| M7 | drop `encoding=` from the filler write | EXACTLY Task 2 test 4 reddens, on the chunk-tuple comparison. Confirm the chunks read back as the store's full extent, matching the planning measurement |
| M8 | allow a non-float new variable through with a NaN fill | Task 2 test 5 reddens via `DID NOT RAISE`; Task 2 test 6 stays green. Additionally record what the store holds afterwards -- the planning measurement predicts fabricated zeros or trues, and observing that is the point of the guard |
| M9 | leave `widen_and_append`'s short-circuit testing the SYMBOL axis alone (do not extend it to the variable set) | EXACTLY Task 2 test 3 and Task 3 test 3 redden: both grow the variable set while their symbol axis AGREES, so the fast path is taken and the closing `append()` refuses the new variable. Measured at revision time against today's still-unextended short-circuit -- that shape raises `new=['beta']`. Task 2 test 1 stays GREEN because its symbol axis GROWS, so the short-circuit is not taken and the widen runs regardless; tests 4, 5 and 6 stay green because they call `widen_data_vars` directly; test 7 stays green because it asserts the short-circuit; and test 8 stays green because -- measured -- the append-dim overlap check sits AHEAD of the variable-set check, leaving its message equality untouched |
| M10 | give `Factor.update` a `mode` parameter forwarded to the backend | A SPLIT: Task 3 test 4 (structural) reddens on the parameter tuple while Task 3 test 5 (behavioural) stays GREEN -- a declared-but-unpassed `mode` opens no route past a guard that runs before `kwargs` is read. Both must be observed |
| M11 | route `Factor.update` through `append()` instead of `widen_and_append()` | Task 3 tests 2 and 3 redden; test 1 stays GREEN. A test 1 that also reddens is not isolating the reconciliation from the plain append |

M5 and M10 are the two that justify the exercise, and each predicts a SPLIT
rather than a joint reddening. A mutation whose observed result does not match
its prediction is a finding to report, not a discrepancy to smooth over.
</mutation_verification>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller -> `XrBackend` write path | An in-process boundary that is nonetheless a real trust boundary: everything past it is DURABLE and, in the measured defect, IRREVERSIBLE. A panel assembled upstream crosses here and can destroy data that was valid before the call |
| `XrBackend` -> zarr store directory | zarr enforces almost nothing about semantic compatibility; every property that matters is checked on this side or not at all |

No network boundary, no untrusted external input, and no package-manager
installs in this task -- so no `T-vyr-SC` package-legitimacy row and no
legitimacy checkpoint. `pyproject.toml` and `uv.lock` are not touched.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-vyr-01 | Tampering | `XrBackend.append` / `_assert_append_compatible`, `data_vars` axis | critical | mitigate | Refuse ANY variable-set mismatch before `to_zarr`, both directions, no opt-out (Task 1). Measured: today all three shapes write silently and leave the store unopenable |
| T-vyr-02 | Denial of Service | the zarr store directory | critical | mitigate | Same refusal. The measured outcome is total loss of read access to the store, including data written by earlier, correct calls -- availability, not just integrity |
| T-vyr-03 | Tampering | `widen_data_vars` filler for a non-float variable | high | mitigate | Refuse without an explicit `fill_values` entry (Task 2, D-08). Measured: a NaN fill becomes `0` for int64 and `True` for bool -- fabricated history, indistinguishable afterwards from real observations |
| T-vyr-04 | Tampering | `widen_data_vars` filler dtype | high | mitigate | Build the filler at the INCOMING variable's dtype (Task 2, D-06). A mismatch is currently caught by the existing dtype guard, so the failure mode is a refused append rather than corruption -- but only because that guard exists; the filler must not depend on being caught |
| T-vyr-05 | Elevation of Privilege | `Factor.update` as a disguised overwrite route | medium | mitigate | No `mode` parameter, no force-shaped kwarg, and the unconditional overlap refusal inherited through `widen_and_append` -> `append` (Task 3, D-10). Locked by a structural + behavioural PAIR, because `260907-uac` proved live that the structural half alone does not span it |
| T-vyr-06 | Tampering | chunk-grid divergence in a widened store | low | mitigate | Pin the filler's chunks via `_append_encoding` (Task 2, D-07). Layout-only -- no value is corrupted and the next append still succeeds -- which is precisely why it is mitigated by construction rather than by a test that waits for a failure |
| T-vyr-07 | Tampering | half-reconciled store after an inherited refusal | low | accept | `widen_and_append` commits its widenings before the closing `append()`, so a refused window can leave the store carrying a grown symbol axis and a grown variable set with its append dimension untouched. Measured live and confirmed harmless: the pre-existing history stays intact, no window is partially written, and a retry is clean. This is the accepted side effect `widen_and_append`'s docstring already records, extended to a third axis; duplicating the refusal to avoid it would break D-05's single-guard rule |
</threat_model>

<verification>
1. `uv run pytest tests/ -q` -> 568 passed (545 live baseline + 8 + 8 + 7), and
   each task GATES on its own running total (553 / 561 / 568) rather than merely
   printing it. The gate's mechanism was tested in BOTH directions at revision
   time: it rejects today's real `545 passed` tail, accepts a synthetic
   `568 passed` tail, and matches none of 567, 569, 1568, 5680, 68 or
   `568 failed`. `tee /dev/stderr` keeps the real tail on screen, and the
   executor still reports the REAL number it saw; a shortfall or an unexplained
   excess is a finding.
2. Every mutation in `<mutation_verification>` applied, its prediction confirmed
   or its divergence reported, reverted, and `git diff --quiet` clean on the
   mutated file afterwards.
3. Every RED observed BEFORE its implementation, and observed to be behavioural.
   Record the actual failure text. A RED that fails on import, collection or
   fixture setup is not a RED.
4. The storage layer adds no factor vocabulary (D-04): the word-boundary count
   of `factor|alpha101|alpha158|kunquant` over non-comment lines of
   `quantlab/dataset/backend.py` stays at or below its measured baseline of 2.
   Both baseline hits are the pre-existing `FactorPolars` caller-bug narrative
   in `head()`'s docstring and are left alone -- an unbounded `-eq 0` gate here
   would fail on the current tree for a reason this plan is not about, and a
   pattern without `\b` matches "refactor" and "enforced" (both measured at
   planning time, which is why neither gate is written that way).
5. `save()` unchanged: its signature line with the `"a"` default present
   verbatim, and `tests/test_factor_save_mode.py`'s 5 tests green untouched.
6. No warm-up trim anywhere: nothing added reads `config.window` inside a write
   path (D-11).
7. Prose gate: the backticked pre-migration path counts in the two touched source
   files stay at or below their measured baselines of 1 and 1, so nothing is
   added to the open doc-sweep backlog.
8. `git status --porcelain` at the end shows the plan's own files plus the
   pre-existing ` M test.py`, which must not be staged, committed, reverted or
   otherwise touched at any point. Use explicit pathspecs on every commit; no
   `git add -A`, no amend.
</verification>

<success_criteria>
- All three shapes of the measured defect now RAISE before `to_zarr`, and the
  store is intact and openable after each refusal.
- A panel that legitimately grew a variable has ONE named way to say so, on the
  storage layer, in variable-neutral vocabulary.
- `widen_and_append` reconciles three axes and is still the single
  reconcile-then-append path; no second composed entry point exists.
- `Factor` has two distinct interfaces -- `save()` for wholesale writes,
  unchanged including its default, and `update()` for automatic incremental
  updates with no overwrite route.
- The `save()` default question is filed for the developer, not decided.
- Suite green at 568, GATED per task (553 / 561 / 568) rather than merely
  printed, and every lock mutation-verified -- including the two that predict a
  split across two separate test functions.
</success_criteria>

<open_question_for_the_developer>
Surfaced, not decided, and filed as a pending todo by Task 3:

`Factor.save()`'s `mode="a"` default is kept unchanged here deliberately --
changing a default is a behaviour change for every existing caller. But zarr's
`"a"` means "overwrite variables in an existing store", not "append along time",
so it is the mode that essentially never does what a second write wants. Now
that `update()` exists as the explicit incremental interface, the `"a"` default
may be vestigial. Worth a decision in its own right.
</open_question_for_the_developer>

<output>
Create `.planning/quick/260907-vyr-reconcile-the-data-vars-axis-on-append-r/260907-vyr-SUMMARY.md` when done
</output>
