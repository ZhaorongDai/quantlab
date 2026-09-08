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
  tokens: 22000
  raw_tokens: 22000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "DVAR-01: `XrBackend.append()` RAISES `ValueError` before `to_zarr` when the incoming panel carries a data variable the store does not have. Re-measured live at planning time on the current tree: it raises NOTHING, and the store comes back with `alpha`(4,2), `beta`(2,2), `timestamp`=4 -- after which `xr.open_zarr()` fails with `ValueError: conflicting sizes for dimension 'timestamp': length 2 on 'beta' and length 4 on ...`. The store is not merely wrong, it is UNOPENABLE."
    - "DVAR-02: `XrBackend.append()` RAISES `ValueError` before `to_zarr` when the store holds a data variable the incoming panel lacks, and there is NO opt-in that permits it. Re-measured live: it raises nothing, `alpha` grows to (4,2) while `beta` stays STUCK at (2,2), and a previously VALID two-variable store becomes unopenable. This is the destructive row -- the only one of the three that destroys pre-existing good data -- and it is why the missing-variable case is refused unconditionally rather than offered a widening path. Filling the absent variable with NaN over the incoming window would punch holes into recent dates of a variable that was complete, and afterwards the store is indistinguishable from one where those values were genuinely missing. That is the same argument `widen_symbol_axis` already makes for refusing to DROP a stored label."
    - "DVAR-03: The two refusals carry DISTINCT messages and a decided precedence. The MISSING case is checked FIRST and names no opt-in, because a caller told about the widening opt-in while still being refused for dropping would be misled. The NEW case names the widening opt-in. Both name the store path and the offending variable names. Placement is AFTER the existing shared-variable dtype loop so no existing refusal's precedence changes -- and measured live at planning time, placement cannot flip an existing test either way: the STRICTEST possible variable-set refusal, injected into `_assert_append_compatible` and run against the whole suite, left 545 passed / 0 failed. No currently-green test appends a mismatched variable set at all."
    - "DVAR-04: There is NO opt-out, DECLARED or SMUGGLED. `XrBackend.append`'s signature stays exactly `(self, path, append_dim, **kwargs)`, AND no route past the variable refusal may be carried in `**kwargs`. The signature alone provably does not span this: quick task `260907-uac` demonstrated live (mutation M6) that a hatch popped from `**kwargs` inside the body makes the guard's `ValueError` disappear while `inspect.signature` still reports a byte-identical parameter tuple. The lock is therefore a PAIR -- one structural assertion on the parameter tuple, one behavioural assertion that a variable-mismatched append carrying an unrecognised kwarg still raises."
    - "DVAR-05: `XrBackend.widen_data_vars()` adds a data variable to an existing store by materialising it over the store's EXISTING extent with the fill value and writing it with `to_zarr(mode=\"a\")`. Measured live: writing a new variable whose dims match the store exactly succeeds; writing one spanning only PART of the store's append axis raises and leaves the store INTACT (`['alpha']`, dims unchanged); and reindexing onto the store's full axis first then writing succeeds, giving the new variable NaN over history. The method is VARIABLE-NEUTRAL: it names no factor concept, so a market-data panel that grows a column later uses the same method."
    - "DVAR-06: The materialised filler carries the INCOMING variable's dtype, not an assumed float64. Measured live: a float64 filler against a float32 incoming variable is refused at the closing `append()` by the EXISTING shared-variable dtype guard; a dtype-matched filler appends cleanly. This is a real trap, not a defensive nicety -- KunQuant emits float32 as readily as float64."
    - "DVAR-07: The filler is written with an explicit `encoding` from `_append_encoding`, so the new variable joins the store on the SAME chunk grid as every other variable. Measured live with `APPEND_DIM_CHUNK` shrunk to 4 over a 10-timestamp store: an unencoded filler gets chunks `(10, 2)` -- the store's whole extent -- while `alpha` holds `(4, 2)`; passing `_append_encoding`'s computed `{'beta': {'chunks': (4, 2)}}` is ACCEPTED on `mode=\"a\"` and produces `(4, 2)`. The divergence does not crash the next append, which is exactly why it needs pinning by construction rather than by a test that waits for a failure."
    - "DVAR-08: A NON-FLOAT new variable is REFUSED without an explicit fill value, mirroring `widen_symbol_axis`'s refusal for the same reason. Measured live: `np.full((2,2), np.nan, dtype=int64)` yields `0`, and `dtype=bool` yields `True`. A NaN backfill of a boolean flag would mark every historical row as FLAGGED -- the identical 'fabricated observation where the data was missing' family the append dtype guard already refuses."
    - "DVAR-09: `widen_and_append()` reconciles ALL THREE axes and remains the ONE reconcile-then-append path -- extended, not duplicated by a second composed entry point. Order is symbol widen, then variable widen, then the UNCHANGED closing `append()`. Measured live end to end: store `{alpha}` x `[A,B]` x 3 dates taking `{alpha,beta}` x `[A,B,C]` x 2 later dates gives 5 timestamps x 3 symbols x 2 variables, timestamp unique AND strictly increasing, pre-existing `alpha` history bit-identical, `alpha` NaN = 3 (the new symbol over 3 historical dates) and `beta` NaN = 9 (3 symbols x 3 historical dates). It short-circuits to a plain `append()` when BOTH axes already agree, and an absent store still takes the single creation path its docstring promises."
    - "DVAR-10: `Factor.update()` is the AUTOMATIC data-update interface: it works out for itself what changed -- later dates, new symbols, new variables -- and reconciles each axis without the caller naming which widening to perform. It offers NO route to overwrite an already-stored range: it has no `mode` parameter and inherits the unconditional `append_dim` overlap refusal shipped by `260907-uac`. Overwriting is `save()`'s job; the two interfaces are how a caller expresses which they mean."
    - "DVAR-11: `Factor` gains a `_widen_fill_values()` seam, overridable per subclass, for the same reason `BaseDataset` has one -- `widen_symbol_axis` refuses to NaN-backfill a non-float variable without an explicit fill, and `Factor` is the shared base for factors AND classification labels. Measured live: the default is `{}` because every factor and label panel is float today (KunQuant emits float arrays, and even `SpotBinaryReturn` builds its binary label from `op.ConstantOp(1.0)`/`(0.0)`, so it is float64 rather than bool). The hook is required regardless of that default: without it a future non-float subclass has no way to widen at all, and `Factor` does not have this method today (verified live: `hasattr(Factor, '_widen_fill_values')` is False)."
    - "DVAR-12: `Factor.save()`'s BEHAVIOUR and its `mode=\"a\"` DEFAULT are unchanged. Only its prose moves: the two clauses asserting this method has no incremental route are now false and are re-pointed at `update()`. `tests/test_factor_save_mode.py`'s 5 tests stay green untouched."
    - "DVAR-13: The live baseline is preserved and grows only by the new tests. Measured at planning time on the current tree: `uv run pytest tests/ -q` -> 545 passed. Expected after this plan: 545 + 7 (Task 1) + 8 (Task 2) + 6 (Task 3) = 566. The executor reports the REAL number; a shortfall or an unexplained excess is a finding, not a rounding error."
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

The probe was reverted; `git status --porcelain` showed only the pre-existing
` M test.py`, which this plan does not touch at any point.
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
  mismatched variable set.
- **D-04 — the new method is `widen_data_vars`.** Variable-neutral: it names the
  xarray concept it operates on and cannot be misread as a dimension. It must
  not mention factors -- "factor" is the caller's vocabulary, never the storage
  layer's.
- **D-05 — extend `widen_and_append`, do not add a second composed entry point.**
  Its docstring already establishes it as the ONE reconcile-then-append path,
  including "an absent store does the same, so there is ONE creation path rather
  than two". Folding the variable axis in completes that meaning rather than
  changing its contract. Weighed against its 4 existing dedicated tests: all of
  them pass single-variable panels whose variable sets already agree, so they
  take the short-circuit and stay green. Verified by the probe run above.
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
    - Test 4: the two messages are DISTINCT and each names the offending
      variable names and the store path. The new-variable message names the
      widening opt-in; the missing-variable message does not.
    - Test 5: precedence -- store `{alpha, beta}`, incoming `{alpha, gamma}`
      (both a drop AND an addition) raises the MISSING message, not the new one.
    - Test 6: identical variable sets still append normally, and a shared
      variable with a mismatched dtype still raises the EXISTING dtype message,
      unchanged -- the new check must not steal precedence from it.
    - Test 7 (the pair for D-04): a structural assertion that
      `XrBackend.append`'s parameter tuple is exactly
      `('self', 'path', 'append_dim', 'kwargs')`, AND a behavioural assertion
      that a variable-mismatched append carrying an unrecognised kwarg still
      raises `ValueError`. Cross-reference each half from the other's docstring:
      quick task `260907-uac` proved live that the structural half alone stays
      green through exactly the drift it exists to catch.

    Expected RED, measured at planning time: every raising test fails with
    `Failed: DID NOT RAISE ValueError`. That is a behavioural RED. A RED that
    fails on import, collection or fixture setup proves nothing and must be
    fixed before proceeding.
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

    Do not touch `append()`'s signature. Do not add a flag, a mode, or a force
    parameter, and do not read anything out of `**kwargs` ahead of the guard
    call -- the guard runs before `kwargs` is touched at all, and that ordering
    is the mechanism D-04 depends on.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_variable_axis_widening.py -q && uv run pytest tests/ -q 2>&1 | tail -2 && test "$(grep -cF 'def append(self, path: str, append_dim: str = "timestamp", **kwargs) -> Self:' quantlab/dataset/backend.py)" -eq 1 && test "$(grep -vE '^[[:space:]]*#' quantlab/dataset/backend.py | grep -cE '\b(force|overwrite_ok|allow_mismatch|allow_var_mismatch)\b')" -eq 0 && test "$(grep -oE '`(base|dataset|factor|label|utils|enums|my_ops|dl_model|ml_model|vecbt)/[A-Za-z_]+\.py' quantlab/dataset/backend.py | wc -l | tr -d ' ')" -le 1</automated>
  </verify>
  <done>
    All 7 new tests pass. The full suite reports 552 passed (545 live baseline +
    7); the executor states the REAL number. Both refusal messages are reachable
    and distinct. `XrBackend.append`'s parameter tuple is unchanged. The
    backticked pre-migration path count in `backend.py` has not risen above its
    measured baseline of 1.
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
      from a missing method.
    - Test 2: `widen_data_vars` called directly adds the variable with the fill
      over the store's whole existing extent and leaves every pre-existing
      variable bit-identical.
    - Test 3 (D-06): incoming variable is float32 while the store is float64;
      the composed path succeeds and the new variable is float32 on disk. RED
      under a hardcoded float64 filler, which the existing dtype guard refuses.
    - Test 4 (D-07): with `APPEND_DIM_CHUNK` monkeypatched small enough that the
      store exceeds it, the new variable's on-disk chunks EQUAL the pre-existing
      variable's. RED under an unencoded filler write. Read chunks from the zarr
      array directly -- xarray does not surface this.
    - Test 5 (D-08): a non-float new variable with no `fill_values` entry is
      REFUSED, and the message names the variable, its dtype, and the
      `fill_values` remedy. Assert the store is intact after the refusal.
    - Test 6 (D-08, the positive half): the same non-float variable WITH an
      explicit `fill_values` entry succeeds and preserves the dtype exactly on
      disk -- no float64 upcast.
    - Test 7 (D-05): `widen_and_append` with variable sets and symbol axes that
      already agree still takes the plain-`append` short-circuit -- no store
      rewrite, no filler write. Assert via the store's mtime or by asserting no
      new variable appeared and the symbol axis is untouched.
    - Test 8 (D-05): `widen_and_append` still inherits the append_dim overlap
      refusal after the variable reconciliation runs, byte-identically to plain
      `append()` once the store path is normalized out. This is the guarantee
      that the composed path did not grow a parallel write.
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
    that grew a column, with no mention of factors. Carry across
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
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_variable_axis_widening.py tests/test_symbol_axis_widening.py tests/test_chunked_ingest.py -q && uv run pytest tests/ -q 2>&1 | tail -2 && test "$(grep -vE '^[[:space:]]*#' quantlab/dataset/backend.py | grep -cEi '\b(factor|alpha101|alpha158|kunquant)')" -le 2 && test "$(grep -c 'def widen_data_vars' quantlab/dataset/backend.py)" -eq 1</automated>
  </verify>
  <done>
    All 8 new tests pass and the 11 pre-existing `test_symbol_axis_widening.py`
    tests plus the 36 `test_chunked_ingest.py` tests stay green untouched. The
    full suite reports 560 passed (552 + 8); the executor states the REAL number.
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
    subclass without a real dataset on disk).
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
    - Test 4 (D-10, the load-bearing negative): `update()` on a range the store
      ALREADY holds RAISES, inheriting the unconditional overlap refusal. Pair
      it with a structural assertion that `Factor.update`'s parameter tuple
      carries no `mode` and no force-shaped parameter, AND a behavioural
      assertion that an unrecognised kwarg does not smuggle one -- the same pair
      D-04 requires on the backend, for the same measured reason.
    - Test 5 (DVAR-11): `Factor._widen_fill_values()` exists on the base,
      returns `{}` by default, and a subclass overriding it has that mapping
      reach the widening call. Assert the reach behaviourally -- a subclass
      carrying a non-float variable and overriding the seam must widen
      successfully where one that does not override is refused. A test that only
      asserts the method exists and returns `{}` does not span the seam's
      purpose.
    - Test 6 (DVAR-12): `save()`'s default is still `"a"` and its wrapped
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
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_factor_update.py tests/test_factor_save_mode.py tests/test_factor_hierarchy.py tests/test_factor_kunquant.py tests/test_factor_polars.py -q && uv run pytest tests/ -q 2>&1 | tail -2 && test "$(grep -c 'is not wired to' quantlab/base/factor.py)" -eq 0 && test "$(grep -c '没接过去' quantlab/base/factor.py)" -eq 0 && test "$(grep -c 'def update' quantlab/base/factor.py)" -eq 1 && test "$(grep -c 'def _widen_fill_values' quantlab/base/factor.py)" -eq 1 && test "$(grep -c 'mode: Literal\["a", "w"\] = "a"' quantlab/base/factor.py)" -eq 1 && test -f .planning/todos/pending/2026-09-07-factor-save-mode-a-default-may-be-vestigial.md && test "$(grep -oE '`(base|dataset|factor|label|utils|enums|my_ops|dl_model|ml_model|vecbt)/[A-Za-z_]+\.py' quantlab/base/factor.py | wc -l | tr -d ' ')" -le 1</automated>
  </verify>
  <done>
    All 6 new tests pass; `test_factor_save_mode.py`'s 5 and
    `test_factor_hierarchy.py`'s 11 stay green untouched. The full suite reports
    566 passed (560 + 6); the executor states the REAL number. `save()`'s
    signature line with its `"a"` default is present verbatim and unmodified.
    Both retired clauses are gone from `quantlab/base/factor.py`. The todo exists
    under `pending/`. The backticked pre-migration path count in `factor.py` has
    not risen above its measured baseline of 1.
  </done>
</task>

</tasks>

<mutation_verification>
This repo has hit "a gate that passes for the wrong reason" five times in two
days; the most recent was a structural proxy that could not span the behavioural
property it stood for, caught only because a mutation demonstrated the split.
Every lock below is therefore mutation-verified: apply the mutation to a clean
tree, confirm the predicted test(s) redden THROUGH THE ASSERTION THEY CLAIM TO
EXERCISE (report the failing line and the failure mode, not just the count),
revert, and confirm `git diff --quiet` on the mutated file.

| Mut | Change | Predicted |
|-----|--------|-----------|
| M1 | delete the whole variable-set check | Task 1 tests 1, 2, 3, 5 redden together via `DID NOT RAISE ValueError` |
| M2 | check ONLY the new-variable direction (drop the missing branch) | EXACTLY tests 2 and 5 redden; test 1 and test 3 stay green. Test 3's incoming `{beta}` is also missing `alpha`, so a green test 3 under M2 would prove the disjoint case is caught by the wrong branch |
| M3 | swap the precedence -- check NEW before MISSING | EXACTLY test 5 reddens, on the message assertion. Nothing else moves |
| M4 | move the variable check ABOVE the shared-variable dtype loop | test 6's dtype half reddens on the message assertion. If nothing reddens, the dtype fixture's variable sets are not identical and the test does not span D-03 |
| M5 | hatch popped from `**kwargs` inside `append`'s body, ahead of the guard call | test 7's BEHAVIOURAL half reddens AND its structural half stays GREEN. Both halves must be observed -- a run where both redden means the pair is not spanning what D-04 says it spans |
| M6 | build the filler as float64 unconditionally | EXACTLY Task 2 test 3 reddens, via the EXISTING append dtype refusal, not a new one. Confirm the message is the dtype guard's |
| M7 | drop `encoding=` from the filler write | EXACTLY Task 2 test 4 reddens, on the chunk-tuple comparison. Confirm the chunks read back as the store's full extent, matching the planning measurement |
| M8 | allow a non-float new variable through with a NaN fill | Task 2 test 5 reddens via `DID NOT RAISE`; Task 2 test 6 stays green. Additionally record what the store holds afterwards -- the planning measurement predicts fabricated zeros or trues, and observing that is the point of the guard |
| M9 | short-circuit `widen_and_append` on the symbol axis alone | EXACTLY Task 2 test 1 and Task 3 test 3 redden; test 7 stays green |
| M10 | give `Factor.update` a `mode` parameter forwarded to the backend | Task 3 test 4's structural half reddens on the parameter tuple |
| M11 | route `Factor.update` through `append()` instead of `widen_and_append()` | Task 3 tests 2 and 3 redden; test 1 stays GREEN. A test 1 that also reddens is not isolating the reconciliation from the plain append |

M5 is the one that justifies the exercise, and its predicted result is a SPLIT,
not a joint reddening. A mutation whose observed result does not match its
prediction is a finding to report, not a discrepancy to smooth over.
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
1. `uv run pytest tests/ -q` -> 566 passed (545 live baseline + 7 + 8 + 6). The
   executor reports the REAL number; a shortfall or an unexplained excess is a
   finding.
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
- Suite green at the stated total, every lock mutation-verified.
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
