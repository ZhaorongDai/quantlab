---
phase: quick-260906-usg
plan: 01
subsystem: factors
tags: [polars, xarray, zarr, databackend, abc, factor-names, tdd, mutation-testing]

# Dependency graph
requires:
  - phase: 03-factor-computation-kunquant-polars
    provides: the Factor/FactorKunQuant/FactorPolars hierarchy (D-03), the
      D-05 dynamic-name contract, and the 03-VERIFICATION.md Gap 1 disposition
      that settled this design
provides:
  - "DataBackend.head(n): a bounded read declared abstractly on the storage interface"
  - "XrBackend.head / PlBackend.head: dimension-agnostic isel bound, and a pushed-down scan_parquet limit"
  - "BaseDataset.head(n): concrete pass-through beside get_lazyframe()"
  - "FactorPolars._get_factor_names(): factor names DERIVED from the computation graph via a bounded probe"
  - "FactorPolars is interchangeable with FactorKunQuant on base/model.py's read() path"
  - "tests/test_backend_head.py: the bounded-read contract, including the non-mutation hazard"
  - "test_kunquant_and_polars_factors_are_interchangeable_on_the_read_path: the regression Phase 3 never had"
affects: [factor computation, model layer collect(), any future DataBackend implementation]

actuals:
  tokens: 7454
  tasks: 3
  commits: 5

tech-stack:
  added: []
  patterns:
    - "Bounded read as an ABC obligation: head(n) is @abstractmethod, so a future backend cannot be constructed without one"
    - "Names derive from the GRAPH, never from the store on disk: a config/store mismatch surfaces loudly instead of being papered over"
    - "Mutation-verified tests: every test added here was proven to go red under a named mutation"

key-files:
  created:
    - tests/test_backend_head.py
  modified:
    - base/backend.py
    - dataset/backend.py
    - base/data.py
    - base/factor_polars.py
    - base/config.py
    - tests/test_factor_polars.py
    - tests/test_factor_hierarchy.py

key-decisions:
  - "Factor names derive from the computation GRAPH, not from the factor store on disk: a store written under n=5, read back under a config saying n=60, yields momentum_60 and fails loudly at lookup rather than silently reporting the stale name the data happens to carry."
  - "The bounded read is an @abstractmethod on DataBackend rather than a limit= keyword on get_lazyframe(): ABC enforcement makes a backend that omits it impossible to construct, whereas an optional keyword is satisfied by plain inheritance and only fails at whichever call site passes it."
  - "Both fresh factors in the read-path regression test leave factor_names unset (deviation from the plan's pinned-KunQuant instruction), because Alpha158SpotKline._get_factor_names() returns all 169 names regardless of any pin, so the plan's own uniform equality assertion cannot hold against a pinned instance."
  - "Constructing a FactorPolars now performs a bounded disk read (~25-50 ms, flat in store size). Accepted knowingly per 03-VERIFICATION.md Gap 1: D-04's 'computation starts at cal()' is not violated by a few rows plus metadata."

patterns-established:
  - "Non-mutating probe beside mutating filters: XrBackend.head returns a derived frame while its filter_by_* siblings mutate in place, and a dedicated test plus mutation 3 enforces the difference."
  - "Assert the PUBLIC call site, not only the private mechanism: get_factor_names()/num_factors read config.factor_names and can see the gap; _get_factor_names() derives and stays green under the regression."

requirements-completed: [FACTOR-03, FACTOR-04]

coverage:
  - id: D1
    description: "DataBackend declares a bounded read abstractly; both concrete backends and BaseDataset provide it, with BaseDataset.__abstractmethods__ unchanged"
    requirement: "FACTOR-03"
    verification:
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_is_an_interface_obligation_not_a_convenience"
        status: pass
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_returns_at_most_n_rows"
        status: pass
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_preserves_the_get_lazyframe_schema"
        status: pass
      - kind: other
        ref: "uv run python -c \"assert 'head' in DataBackend.__abstractmethods__; assert BaseDataset.__abstractmethods__ == frozenset({'_raw_data_to_xr'})\""
        status: pass
    human_judgment: false
  - id: D2
    description: "The probe read does not mutate backend state -- unlike the in-place filter_by_date/filter_by_symbol siblings on the same class"
    requirement: "FACTOR-03"
    verification:
      - kind: unit
        ref: "tests/test_backend_head.py#test_head_does_not_mutate_backend_state"
        status: pass
    human_judgment: false
  - id: D3
    description: "FactorPolars factor names are derived from the computation graph at config-assignment time; a bare-constructed Momentum reports ('momentum_5',) and num_factors == 1"
    requirement: "FACTOR-03"
    verification:
      - kind: unit
        ref: "tests/test_factor_polars.py#test_factor_names_resolve_dynamically_from_the_lazyframe_schema"
        status: pass
    human_judgment: false
  - id: D4
    description: "An explicit config.factor_names pin still wins through the inherited base-class hook, and skips the probe entirely"
    requirement: "FACTOR-03"
    verification:
      - kind: unit
        ref: "tests/test_factor_polars.py#test_an_explicit_factor_names_pin_is_not_overwritten_at_construction"
        status: pass
    human_judgment: false
  - id: D5
    description: "Both factor backends survive cal() -> save() -> fresh instance -> read() with factor_data_strategy='read', answering both name surfaces uniformly with no per-backend branch"
    requirement: "FACTOR-04"
    verification:
      - kind: integration
        ref: "tests/test_factor_hierarchy.py#test_kunquant_and_polars_factors_are_interchangeable_on_the_read_path"
        status: pass
      - kind: other
        ref: "mutation: restoring the deleted no-op override reddens this test while the cal-path sibling stays green"
        status: pass
    human_judgment: false
  - id: D6
    description: "The obsolete precondition prose is gone from every place it was stated (factor_polars.py docstring, its RuntimeError guard, the test restatement, PolarsFactorConfig's docstring)"
    verification:
      - kind: other
        ref: "grep -c 'RuntimeError' base/factor_polars.py -> 0; grep -c 'def _maybe_resolve_factor_names' base/factor_polars.py -> 0; grep -c 'Documented precondition' base/factor_polars.py -> 0; grep -c 'pytest.raises(RuntimeError' tests/test_factor_polars.py -> 0"
        status: pass
    human_judgment: false

# Metrics
duration: 41 min
completed: 2026-09-06
status: complete
---

# Quick Task 260906-usg: Close Phase 3 Gap 1 Summary

**`FactorPolars` factor names are now DERIVED from the computation graph through a bounded `DataBackend.head(n)` probe, so a read-back Polars factor is interchangeable with a KunQuant one on `base/model.py`'s `factor_data_strategy="read"` path — closed by DELETING the no-op override that had disabled the derivation channel.**

## Performance

- **Duration:** 41 min
- **Tasks:** 3
- **Files created:** 1
- **Files modified:** 7
- **Full suite:** 378 → **384 passed**, zero failures, zero regressions

## Accomplishments

- **`DataBackend.head(n)` added as an `@abstractmethod`** — a bounded read is now an interface *obligation*, not a convenience on one backend. `XrBackend` bounds every dimension via `isel({dim: slice(0, n) for dim in self.data.dims})` (dimension-name-agnostic, so a third axis is bounded too); `PlBackend` pushes the limit down through `scan_parquet`; `BaseDataset` exposes a concrete pass-through beside `get_lazyframe()`.
- **The Gap 1 fix is a deleted override, not an added one.** `base/factor.py:92-93` already implements "explicit pin wins, else derive"; `FactorPolars` had disabled the derivation channel with a no-op `_maybe_resolve_factor_names()`, leaving only `cal()` to fill `config.factor_names` in. Deleting it and making `_get_factor_names()` the actual derivation fixes `read()`, `cal()` and bare construction simultaneously, and the explicit-pin channel comes free from the base class.
- **The regression test Phase 3 never had.** The existing interchangeability proof pinned `factor_data_strategy="cal"` and hand-called `factor.cal()`, so `base/model.py`'s `read()` arm had never been driven for *either* backend — which is exactly how the asymmetry survived a green suite. The new sibling drives `cal() → save(mode="w") → fresh instance → read()` for both.
- **The obsolete precondition prose is gone everywhere it was stated** — the `FactorPolars` class docstring, its `RuntimeError` guard, `tests/test_factor_polars.py`'s restatement, and `PolarsFactorConfig`'s "expected to stay `None` until `cal()` runs" claim.
- **Every test added here was proven to go red under a named mutation.** Nine mutations run, nine expected outcomes.

## Task Commits

1. **Task 1 RED: bounded-read contract** — `85d450b` (test)
2. **Task 1 GREEN: `head(n)` on the interface and both backends** — `7cdf2a5` (feat)
3. **Task 2 RED: graph-derived names before `cal()`, plus the pin proof** — `257d9ef` (test)
4. **Task 2 GREEN: the derivation, the deleted override, the deleted precondition** — `5c4985c` (feat)
5. **Task 3: the read-path regression across both backends** — `eed186d` (test)

## Mutation Results

Every mutation below was applied, run, and reverted with `git checkout`. **Nine mutations, nine expected outcomes.**

### Task 1 — the bounded read

| # | Mutation | Test that failed | Reason |
|---|---|---|---|
| 1 | `XrBackend.head` returns `self.get_lazyframe()` (ignores `n`) | `test_head_returns_at_most_n_rows` | returned all 30 rows instead of ≤4 |
| 2 | `PlBackend.head` returns `self.data` (unbounded) | `test_head_returns_at_most_n_rows` | same, on the other backend — the shared fixture drives both |
| 3 | `XrBackend.head` assigns the slice back to `self.data` | `test_head_does_not_mutate_backend_state` | `get_lazyframe()` afterwards returned 2 rows, not 30 — **the hazard `filter_by_date`/`filter_by_symbol` set up** |
| 4 | Drop `@abstractmethod` from `DataBackend.head` | `test_head_is_an_interface_obligation_not_a_convenience` | `'head' not in DataBackend.__abstractmethods__`; the stub subclass instantiated instead of raising `TypeError` |

### Task 2 — the derivation

| # | Mutation | Outcome | Reason |
|---|---|---|---|
| 1 | Restore the no-op `_maybe_resolve_factor_names()` override | **RED** — `test_factor_names_resolve_dynamically_from_the_lazyframe_schema` (pre-`cal()` half) and `test_an_explicit_factor_names_pin_is_not_overwritten_at_construction` | `assert None == ('momentum_5',)`; and `DID NOT RAISE _ProbeCalled` — the probe was never reached at all |
| 2 | Drop the `_INDEX_COLUMNS` exclusion | **RED** — the same dynamic-name test | `('timestamp', 'symbol', 'momentum_5') != ('momentum_5',)` |
| 3 | `_get_factor_names()` returns `tuple(self.config.factor_names)` | **RED** — all four unpinned tests | `TypeError: 'NoneType' object is not iterable`, at construction |
| 4 | `_SCHEMA_PROBE_ROWS = 0` (**must stay green**) | **GREEN** — 6 passed | the derivation reads `collect_schema()` only; it depends on the schema, never on rows |

### Task 3 — the load-bearing mutation

Restore `FactorPolars._maybe_resolve_factor_names()` as a no-op override. **Both required halves held:**

| Half | Test | Outcome |
|---|---|---|
| The new test must see the gap | `test_kunquant_and_polars_factors_are_interchangeable_on_the_read_path` | **FAILED** — `Momentum.get_factor_names() is empty after read(); the model layer reads this surface, not the private one` / `assert (None is not None)` |
| The old suite must NOT have been able to see it | `test_kunquant_and_polars_factors_are_interchangeable_in_one_dlconfig` | **PASSED** (1 passed, 10 deselected) |

That pair is the proof: the phase's existing cal-path test is structurally blind to this gap, and the new read-path test is not. Note the failing assertion is on the **public** `get_factor_names()`, exactly as predicted — the private `_get_factor_names()` derives and stays green under the mutation, so a test asserting only it would have reproduced this project's recorded "mechanism proved as a function but never through its call site" failure.

## The Decision That Governs This Design

**Names derive from the GRAPH, never from the factor store on disk.**

`_get_factor_names()` runs `_get_factor_lazyframe()` over a bounded probe of the *dataset* and reads the resulting schema. It deliberately does **not** read the names out of the factor store. The consequence is intentional and was decided in `03-VERIFICATION.md` § "Gap Dispositions" → "Gap 1":

> a store written with `n=5` under a config now saying `n=60` yields `momentum_60`, and the subsequent lookup fails with a name pointing straight at the cause. Deriving from disk instead would have returned `momentum_5` — matching the data while silently ignoring that the config asked for something else.

A loud failure naming the mismatch beats a silent success that quietly answers a different question than the one the config asked. This is not a rough edge to be smoothed later.

## Files Created/Modified

- `base/backend.py` — `DataBackend.head(n) -> pl.LazyFrame` as `@abstractmethod`, with both obligations (no full materialization, no mutation of `self.data`) stated in the docstring.
- `dataset/backend.py` — `XrBackend.head` (dimension-agnostic `isel` bound; the slice is a **local**, never written back) and `PlBackend.head` (lazy `.head(n)`).
- `base/data.py` — `BaseDataset.head(n)` pass-through, concrete, shaped identically to `get_lazyframe()`.
- `base/factor_polars.py` — `_SCHEMA_PROBE_ROWS = 8`; `_get_factor_names()` rewritten as the derivation; `_maybe_resolve_factor_names()` override **deleted**; `RuntimeError` guard **deleted**; class docstring rewritten.
- `base/config.py` — `PolarsFactorConfig` docstring corrected: names are derived at config-assignment time; `None` is the normal case *because* the derivation fills it.
- `tests/test_backend_head.py` (new, 154 lines) — four bounded-read contract tests, each driving both backends from one shared fixture.
- `tests/test_factor_polars.py` — dynamic-name test rewritten (assertions moved before `cal()`, obsolete `pytest.raises` block deleted); new two-sided pin test.
- `tests/test_factor_hierarchy.py` — the read-path interchangeability regression test, appended beside its cal-path sibling.

## Decisions Made

1. **Bounded read as a distinct `@abstractmethod`, not a `limit=` keyword.** ABC enforcement makes a future backend impossible to construct without one; an optional keyword is satisfied by plain inheritance and fails only at whichever call site happens to pass it — this repo's recorded "gate whose flag was always true where it was read" shape. Mutation 4 of Task 1 is what holds this.
2. **`XrBackend.head` builds its selector from `self.data.dims`** rather than naming `timestamp`. A storage-medium-agnostic backend has no business knowing this project's panels are indexed by time and symbol.
3. **`README.md` left untouched**, per the plan and per `03-VERIFICATION.md`'s own correction: its L34-40 were false against the old code and became true with this change. `git diff --stat -- README.md` is empty.
4. **`tests/test_factor_hierarchy.py`'s five `_maybe_resolve_factor_names` references kept** (count verified unchanged at 5) — they exercise the base-class hook contract, which survives. Only `FactorPolars`'s override went.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in plan spec] Task 3's fresh KunQuant instance left UNPINNED rather than pinned**

- **Found during:** Task 3 (the read-path regression test)
- **Issue:** The plan directed that the fresh `Alpha158SpotKline` keep `factor_names=[_KUNQUANT_FACTOR_NAME]` while also asserting, uniformly across both factors, `tuple(factor.get_factor_names()) == tuple(factor._get_factor_names())`. Those two instructions are mutually unsatisfiable: `Alpha158SpotKline._get_factor_names()` returns `tuple(self._factor_names_stream())` — the **full** Alpha158 name block — regardless of any pin. Measured empirically: a pinned instance reports `public n=1, private n=169`; an unpinned one reports `169`/`169` and agrees.
- **Fix:** Both fresh instances leave `factor_names` unset. The *first*, computing instance keeps the pin so `cal()` computes one factor rather than the whole block — that pin is about cost, not naming.
- **Why this is stronger, not a weakening:** "neither instance was told what it computes, so each must derive it" is precisely the condition Gap 1 is about, and it makes the plan's own uniform equality assertion hold with no per-backend branch. Measured cost of the unpinned construction: **5 ms** (the name stream is pure Python graph construction, no KunQuant compile).
- **Files modified:** `tests/test_factor_hierarchy.py`
- **Verification:** The load-bearing mutation still produces both required halves — new test RED on the public surface, cal-path sibling GREEN. The assertion the plan called load-bearing is preserved verbatim.
- **Committed in:** `eed186d`

**2. [Rule 3 - Blocking] Test-local `_ProbeCalled` exception instead of `RuntimeError` in the pin test**

- **Found during:** Task 2
- **Issue:** The pin test needs to assert that constructing an unpinned factor reaches the probe. The plan's own acceptance grep requires `grep -c 'pytest.raises(RuntimeError' tests/test_factor_polars.py` to be `0`, so the patched `head` could not raise `RuntimeError`.
- **Fix:** Defined a module-local `_ProbeCalled(Exception)`. Strictly better than a built-in anyway: it can only come from the patch, so the assertion cannot pass by accident on an unrelated failure that happens to raise the same class.
- **Files modified:** `tests/test_factor_polars.py`
- **Verification:** grep returns `0`; Task 2 mutation 1 confirms the test goes red (`DID NOT RAISE _ProbeCalled`) when the derivation is disabled.
- **Committed in:** `257d9ef`

---

**Total deviations:** 2 auto-fixed (1 plan-spec contradiction resolved empirically, 1 blocking constraint conflict).
**Impact on plan:** No scope creep. Both deviations preserve the plan's load-bearing assertions; deviation 1 makes the read-path test more symmetric and its mutation proof unchanged.

## Issues Encountered

- **Task 1, self-inflicted:** the first mutation run used `git checkout dataset/backend.py` while the GREEN implementation was still uncommitted, wiping it. Re-applied, then re-ordered the workflow so mutations run only *after* the implementation commit — which is the correct order regardless, since `git checkout` is the documented revert step. No lost work; mutations 1 and 2 were re-run cleanly against the committed implementation.
- The known `kun::StreamContext::~StreamContext()` full-suite flake did not occur in any of the four full-suite runs.

## Verification Results

Every item in the plan's `<verification>` block:

| # | Check | Result |
|---|---|---|
| 1 | `uv run pytest -q` | **384 passed**, exit 0 (378 baseline + 6 new) |
| 2 | `'head' in DataBackend.__abstractmethods__` | OK |
| 3 | `grep -c 'RuntimeError' base/factor_polars.py` / `grep -c 'def _maybe_resolve_factor_names' base/factor_polars.py` | `0` / `0` |
| 4 | `grep -c '_maybe_resolve_factor_names' tests/test_factor_hierarchy.py` | `5` — unchanged from the pre-task count |
| 5 | `git diff --stat -- README.md` | empty |
| 6 | All 9 mutations produced their named outcome, incl. Task 3's two-half proof | Confirmed (see Mutation Results) |

Additionally: `BaseDataset.__abstractmethods__ == frozenset({'_raw_data_to_xr'})` still holds, and the no-type-dispatch grep over the new read-path test returns `0`.

## Known Stubs

None. No stub, skipped test, or unrun `<verify>` was left behind.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Phase 3 Gap 1 is closed at the root.** `factor_data_strategy="read"` now works for every factor backend, and `README.md:34-40` is true against the code for the first time.
- `DataBackend.head(n)` is available to any future consumer needing a cheap schema/dtype probe. Any **new** `DataBackend` implementation must now supply it — that is the intended enforcement, and it will surface at construction rather than at a distant call site.
- **Known boundary, unchanged and documented:** a factor whose output column names depend on data *values* (a pivot over distinct symbols, say) would derive wrong names from a bounded probe. Such a factor already violates the stated `_get_factor_lazyframe` contract, so it is out of contract rather than a regression — but it is the one shape this design cannot serve.
- **Still open, untouched by this task:** Phase 3 Gap 2 (synthesized `amount` makes `vwap` identically `close` for US equities) remains dismissed by user decision, with its consequences recorded in `03-VERIFICATION.md`.

## Self-Check: PASSED

- All 8 files verified present on disk.
- All 5 commits verified present in `git log`.
- Full suite re-run green at 384 after every task.

---
*Quick task: 260906-usg*
*Completed: 2026-09-06*
