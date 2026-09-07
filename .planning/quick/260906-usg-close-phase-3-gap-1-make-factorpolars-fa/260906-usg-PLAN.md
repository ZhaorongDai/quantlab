---
phase: quick-260906-usg
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - base/backend.py
  - dataset/backend.py
  - base/data.py
  - base/factor_polars.py
  - base/config.py
  - tests/test_backend_head.py
  - tests/test_factor_polars.py
  - tests/test_factor_hierarchy.py
autonomous: true
requirements: [FACTOR-03, FACTOR-04]
user_setup: []

estimate:
  tokens: 90000
  raw_tokens: 90000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "`DataBackend` declares a bounded read (`head(n)`) as an ABSTRACT method, so a future backend cannot be instantiated without implementing it; both `XrBackend` and `PlBackend` implement it, and `BaseDataset` exposes it as a pass-through beside the existing `get_lazyframe()`."
    - "`head(n)` returns at most `n` rows with the SAME column names and dtypes `get_lazyframe()` returns, and leaves `self.data` untouched — unlike `filter_by_date`/`filter_by_symbol` on the same classes, which mutate in place."
    - "`FactorPolars._get_factor_names()` DERIVES the names by running `_get_factor_lazyframe()` over a bounded probe read and returning `collect_schema().names()` minus the index columns; it never consults `config.factor_names` and never touches `self.data_backend`."
    - "`FactorPolars` no longer overrides `_maybe_resolve_factor_names()`, so names resolve at config-assignment time: a bare-constructed factor, a `read()` factor and a `cal()` factor all report the same names, and an explicit `config.factor_names` pin still wins (base-class hook, unchanged)."
    - "Both factor backends survive `cal() -> save() -> fresh instance -> read()` with `factor_data_strategy=\"read\"` and answer BOTH name surfaces — `_get_factor_names()` (what `base/model.py:get_factor_names` calls) and `get_factor_names()`/`num_factors` (what `config.factor_names` backs) — with no per-backend branch and no runtime type check in the test body."
    - "Restoring the deleted no-op override makes the new read-path test FAIL while the phase's existing cal-path proof (`test_kunquant_and_polars_factors_are_interchangeable_in_one_dlconfig`) still PASSES — the mutation that demonstrates the new test is the thing that catches the gap."
    - "The obsolete precondition prose is gone everywhere it was stated: `base/factor_polars.py`'s class docstring, its `RuntimeError` guard, `tests/test_factor_polars.py`'s restatement, and `base/config.py:PolarsFactorConfig`'s `\"expected to stay None until cal() runs\"` claim."
  artifacts:
    - "base/backend.py — `DataBackend.head(n) -> pl.LazyFrame` as `@abstractmethod`"
    - "dataset/backend.py — `XrBackend.head` (dimension-agnostic `isel` bound, non-mutating) and `PlBackend.head` (lazy `.head(n)`)"
    - "base/data.py — `BaseDataset.head(n)` pass-through beside `get_lazyframe()`, concrete (NOT abstract)"
    - "base/factor_polars.py — `_get_factor_names()` as the derivation; `_maybe_resolve_factor_names()` override DELETED"
    - "tests/test_backend_head.py — bounded-read contract: row bound, schema equality, non-mutation, ABC enforcement"
    - "tests/test_factor_hierarchy.py — `test_kunquant_and_polars_factors_are_interchangeable_on_the_read_path`"
  key_links:
    - "`FactorPolars._get_factor_names()` runs from inside the `Factor.config` setter, BEFORE `self.data_backend` exists (base/factor.py:36-39, locked by tests/test_factor_hierarchy.py:335). It must reach the DATASET's backend (`self.config.dataset`), never the factor's own storage backend."
    - "`head()` must not write back to `self.data`. The dataset object is SHARED with `cal()`; a mutating probe would silently leave `cal()` computing over 8 rows, and nothing downstream could tell."
    - "The new read-path test must assert the PUBLIC `get_factor_names()`/`num_factors` surface, not only the private `_get_factor_names()`. The private one derives and stays green under the regression; asserting only it reproduces this project's recorded 'mechanism proved as a function but never through its call site' failure."
    - "Names derive from the GRAPH, not from the factor store on disk: a store written with `n=5` under a config now saying `n=60` yields `momentum_60` and the lookup fails loudly. That is the decided behaviour (03-VERIFICATION.md Gap 1) — do not 'fix' it by reading names out of the factor store."
---

<!-- planner-discipline-allow: _maybe_resolve_factor_names, RuntimeError, isinstance, Documented precondition -->

<objective>
Close Phase 3 Gap 1: make `FactorPolars` factor names a DERIVED property of the
computation graph, resolved through a bounded read added to the `DataBackend`
interface.

Purpose: today `FactorPolars` is not interchangeable with `FactorKunQuant` on
`base/model.py`'s `read()` path — a Polars factor that was read rather than
computed carries `factor_names=None`, so `get_factor_names()` returns `None` and
`num_factors` raises. The root cause is not a missing assignment in `read()`; it
is that the derivation channel (`_get_factor_names()`) was disabled by a no-op
`_maybe_resolve_factor_names()` override, leaving only `cal()` to populate the
field. **The fix is a DELETED override, not an added one.**

Output: a bounded read on the `DataBackend` interface implemented by both
backends, a `FactorPolars._get_factor_names()` that derives from the graph, the
authorised deletions of the now-impossible precondition, and the regression test
the phase never had — both backends driven through
`save() -> fresh instance -> read() -> names` with `factor_data_strategy="read"`.

**The design is settled and must not be re-derived.** It is recorded with
measurements in
`.planning/phases/03-factor-computation-kunquant-polars/03-VERIFICATION.md`
§ "Gap Dispositions" → "Gap 1 — accept; close it at the root". Read that section
before Task 2. Do not re-open the three routes it measured, and do not
re-litigate the accepted consequence that constructing a `FactorPolars` now
touches disk for a bounded read (D-04's "computation starts at `cal()`" is not
violated — a few rows plus metadata is not the computation; the trade was made
knowingly).
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@.planning/phases/03-factor-computation-kunquant-polars/03-VERIFICATION.md

@base/backend.py
@dataset/backend.py
@base/data.py
@base/factor.py
@base/factor_polars.py
@factor/momentum.py
@tests/test_factor_polars.py
@tests/test_factor_hierarchy.py
</context>

<constraints>
- No new runtime dependency. `uv add` / `pip install` are forbidden.
- xarray/Zarr stays the inter-module exchange format. `head()` is the bounded
  twin of the ALREADY-PUBLIC `BaseDataset.get_lazyframe()` pass-through, not a
  new polars transport channel; the factor layer still emits `xr.Dataset` only.
- `FactorPolars.cal()` and `Factor.read()` are UNCHANGED. `read()` keeps its
  existing meaning — instantiate the specified date range of historical data —
  and plays no part in naming.
- **Do NOT touch `README.md`.** Its lines 34-40 are false against today's code
  and become TRUE once this lands. The code catches up to the doc.
- **Keep** `tests/test_factor_hierarchy.py`'s references to
  `_maybe_resolve_factor_names` (L161, L186, L340, L364, L370). They exercise
  the BASE-CLASS hook contract, which survives; only `FactorPolars`'s override
  of it goes.
- Reaching past the interface to `xr.open_zarr(...).isel(...)` is explicitly
  REJECTED: using one concrete backend's private path to fix a
  backend-dependence gap is self-contradicting.
</constraints>

<verification_discipline>
This project has eight recorded instances of tests that passed for a different
reason than their author assumed — including Gap 1 itself, where
`tests/test_factor_polars.py:95-117` restated the false precondition in its own
docstring and asserted only the `cal()` path, encoding the defect instead of
catching it.

**Every task below names mutations the executor MUST run.** A test that stays
green when you break the fix has verified nothing. Run each mutation, confirm
the named test fails for the named reason, then `git checkout` the mutated file
before continuing. Record the mutation results in the SUMMARY.

Test commands (`uv run`; `pythonpath = ["."]` is set in `pyproject.toml`):
- Focused: `uv run pytest tests/test_factor_polars.py tests/test_factor_hierarchy.py -q`
- Full suite: `uv run pytest -q` — currently **378 passed**, must not regress.
- pytest exits **5** on "no tests ran" — a scaffold gap, not green. Read exit
  codes directly; `cmd | tail` yields tail's exit code, not the command's.
- Known flake, not yours: a full run intermittently wedges in
  `kun::StreamContext::~StreamContext()` inside `KunRunner.abi3.so`. Kill and
  re-run.
</verification_discipline>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Add a bounded read to the DataBackend interface</name>
  <files>base/backend.py, dataset/backend.py, base/data.py, tests/test_backend_head.py</files>
  <behavior>
    - `head(4)` on a 30-row store returns 1..4 rows, for BOTH `XrBackend` and `PlBackend`.
    - `head(2).collect_schema()` equals `get_lazyframe().collect_schema()` — same names AND same dtypes — for both backends. (This is what catches an `XrBackend` implementation that forgets `reset_index()` and drops the index columns.)
    - After `head(2)`, `get_lazyframe().collect().height` is still the FULL row count for both backends: the probe does not mutate backend state.
    - A `DataBackend` subclass implementing every abstract member EXCEPT `head` raises `TypeError` on instantiation, and `"head"` is in `DataBackend.__abstractmethods__`.
  </behavior>
  <action>
Add the bounded read to the `DataBackend` ABC and implement it on both concrete
backends plus the dataset pass-through.

**`base/backend.py`** — declare it abstract, directly under `get_lazyframe`:
an abstract `head` taking `n: int` and returning `pl.LazyFrame`. Docstring: a
bounded read — return a lazyframe carrying AT MOST `n` rows of the backing
store, with the same column names and dtypes `get_lazyframe()` would return; an
implementation must not materialize the whole store and must not mutate
`self.data`.

Choose a distinct abstract method over a `limit=` keyword on `get_lazyframe`:
ABC enforcement makes it impossible for a future backend to be constructed
without supplying one, whereas an optional keyword is satisfied by plain
inheritance and only fails at whichever call site happens to pass it — this
repo's recorded "gate whose flag was always true where it was read" shape.

**`dataset/backend.py` / `XrBackend.head`** — bound EVERY dimension before
converting: build the selector as `{dim: slice(0, n) for dim in self.data.dims}`
and pass it to `isel`. Dimension-name-agnostic on purpose — a medium-agnostic
backend must not hardcode `timestamp`. Then `.to_dataframe().reset_index()`,
wrap with `pl.from_pandas(...).lazy()` and apply `.head(n)` so the row bound is
exact. Assign the sliced dataset to a LOCAL; never write it back to `self.data`.
`filter_by_date`/`filter_by_symbol` on this same class DO mutate in place, so an
implementation copied from them would silently truncate the dataset every
consumer shares — see the third behavior above, which exists to catch exactly
that.

**`dataset/backend.py` / `PlBackend.head`** — `self.data.head(n)`. Genuinely
lazy: `scan_parquet` pushes the limit down, so this costs nothing.

**`base/data.py` / `BaseDataset.head`** — a pass-through to
`self.data_backend.head(n)`, placed beside the existing `get_lazyframe()` and
shaped identically. It must be CONCRETE, not abstract:
`tests/test_dataset_hierarchy.py` asserts
`BaseDataset.__abstractmethods__ == frozenset({"_raw_data_to_xr"})`, and that
assertion is correct — do not weaken it.

**`tests/test_backend_head.py`** (new file) — four tests, one per behavior
above, each driving BOTH backends from one shared fixture: build a small
`xr.Dataset` (10 timestamps x 3 symbols = 30 long-format rows) for `XrBackend`
via `to_internal`, and write the equivalent long-format frame to a parquet file
under `tmp_path` for `PlBackend` to `read`. The ABC-enforcement test defines its
stub subclass locally inside the test body.
  </action>
  <verify>
    <automated>uv run pytest tests/test_backend_head.py -q; echo "pytest exit=$?"; uv run python -c "from base.backend import DataBackend; from base.data import BaseDataset; assert 'head' in DataBackend.__abstractmethods__; assert 'head' not in BaseDataset.__abstractmethods__; assert BaseDataset.__abstractmethods__ == frozenset({'_raw_data_to_xr'}); print('OK')"</automated>
    <mutation>
Run each, confirm the named failure, then `git checkout` the file:
1. `XrBackend.head` returns `self.get_lazyframe()` (ignoring `n`) -> the row-bound test MUST fail.
2. `PlBackend.head` returns `self.data` (unbounded) -> the row-bound test MUST fail.
3. `XrBackend.head` assigns the sliced dataset back to `self.data` before converting -> the non-mutation test MUST fail.
4. Drop the abstract decorator from `DataBackend.head` -> the ABC-enforcement test MUST fail.
If any mutation leaves the suite green, the corresponding test is not testing
what it claims; strengthen it before moving on.
    </mutation>
  </verify>
  <done>
`DataBackend` declares `head` abstract; `XrBackend`, `PlBackend` and
`BaseDataset` all provide it; `BaseDataset.__abstractmethods__` is unchanged;
`tests/test_backend_head.py` passes (exit 0, not 5); all four mutations produce
the named failures.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Derive FactorPolars names from the graph and delete the obsolete precondition</name>
  <files>base/factor_polars.py, base/config.py, tests/test_factor_polars.py</files>
  <behavior>
    - A BARE-CONSTRUCTED `Momentum` (no `cal()`, no `read()`) answers `get_factor_names() == ("momentum_5",)` and `num_factors == 1`.
    - The same two assertions still hold after `cal()`.
    - A config with `factor_names=["pinned_name"]` keeps the pin at construction and does NOT invoke the probe: with `BaseDataset.head` monkeypatched to raise, constructing the PINNED factor succeeds while constructing an UNPINNED one raises.
    - Nothing in `base/factor_polars.py` reads `config.factor_names` inside `_get_factor_names()`, and nothing there raises about an unresolved state.
  </behavior>
  <action>
Read `03-VERIFICATION.md` § "Gap Dispositions" → "Gap 1" first. Three edits to
`base/factor_polars.py`, plus one stale docstring elsewhere and the authorised
test deletions.

**(1) DELETE the `_maybe_resolve_factor_names()` no-op override** (currently
`base/factor_polars.py:55-61`) outright. Do not replace it with anything, do not
leave a stub. Base-class behaviour is restored: names resolve at
config-assignment time inside the `Factor.config` setter, so `read()`, `cal()`
and a bare-constructed instance all become correct at once, and "explicit pin
wins, else derive" comes free from `base/factor.py:92-93`.

**(2) REWRITE `_get_factor_names()` as the derivation** (currently
`base/factor_polars.py:63-71` — the guard body goes; the method becomes the
derivation). Add a module-level `_SCHEMA_PROBE_ROWS = 8` beside
`_INDEX_COLUMNS`. The method reads a bounded probe through the dataset
(`self.config.dataset.read().head(_SCHEMA_PROBE_ROWS)`), passes it through
`self._get_factor_lazyframe(...)`, and returns a tuple of
`collect_schema().names()` minus `_INDEX_COLUMNS`.

Four constraints on that method, each of which has a way to go wrong silently:
  - It must NOT read `self.config.factor_names`. The base hook owns the
    explicit-pin channel; consulting it here would re-couple the two.
  - It must NOT touch `self.data_backend`. This runs from inside the config
    setter, before `Factor.__init__` has assigned the storage backend
    (`base/factor.py:36-39`, locked by `tests/test_factor_hierarchy.py:335`).
    The dataset's own backend, reached via `self.config.dataset`, is a different
    object and does exist.
  - It must NOT depend on the probe returning any ROWS. Only `collect_schema()`
    is consulted; a date filter that yields zero rows still yields the right
    schema. Do not add an "empty probe" guard — `_SCHEMA_PROBE_ROWS` exists so
    real dtypes come along for free, not because rows are required.
  - Keep the `.read()` call. `XrBackend.read` returns early when data is already
    in memory (the ordinary case: `BaseDataset.__init__` already read it), so
    this is a cache hit; but a dataset whose `_reset_symbols()` is a no-op has
    nothing loaded and needs it. Note that `_reset_dataset_config()` runs AFTER
    name resolution in the setter, so the probe sees the dataset's own date
    window — irrelevant to a schema, and not something to "fix" by reordering
    the setter.

**(3) REPLACE the class-docstring paragraph** currently headed "Documented
precondition on factor names" (`base/factor_polars.py:44-50`, keeping the
closing quotes). The new paragraph states: factor names are DERIVED from the
computation graph at config-assignment time via a bounded probe read; an
explicit `config.factor_names` still wins through the inherited base-class hook;
and constructing a factor therefore performs a bounded disk read BY DECISION
(cite 03-VERIFICATION.md Gap 1 — D-04 is not violated, a few rows plus metadata
is not the computation). There is no precondition left to state, so do not
restate one in weaker words.

After this task the string `RuntimeError` must not appear ANYWHERE in
`base/factor_polars.py` — not in code, not in a comment, not in a docstring.
The same goes for a definition of the deleted override.

**(4) `base/config.py:230-234`** — `PolarsFactorConfig`'s docstring claims
`factor_names` "is expected to stay `None` until `cal()` runs". That is the same
now-false precondition stated in a second place; it was not in the authorised
deletion table only because the table enumerated `base/factor_polars.py` and
`tests/`. Correct the sentence to say names are derived at config-assignment
time from the computation graph, and that leaving `factor_names` as `None` is
the normal case because the derivation fills it.

**(5) `tests/test_factor_polars.py`** — rewrite
`test_factor_names_resolve_dynamically_from_the_lazyframe_schema`: delete the
docstring paragraph restating the precondition (L101-105) and the
`pytest.raises` block asserting it (L110-111). The two surviving assertions move
BEFORE the `cal()` call — a bare-constructed factor now resolves its names, and
pinning that is the cleanest proof the fix works. Then keep `cal()` and repeat
the two assertions, so the original coverage is not lost. Rewrite the docstring
to describe what is now true.

Add `test_an_explicit_factor_names_pin_is_not_overwritten_at_construction`:
monkeypatch `base.data.BaseDataset.head` to raise, construct a `Momentum` whose
config pins `factor_names=["pinned_name"]` and assert
`list(factor.get_factor_names()) == ["pinned_name"]` (compare as a list — the
pin goes in as a list and `get_factor_names()` returns the field verbatim),
then assert that constructing an UNPINNED `Momentum` under the same patch
raises. Two-sided: the first half proves the pin survives, the second proves the
probe is precisely what the pin skips. Note in the docstring that `cal()` still
overwrites `config.factor_names` from the collected schema — pre-existing,
unchanged behaviour, deliberately not asserted here.

Leave the other three tests in that file untouched.
  </action>
  <verify>
    <automated>uv run pytest tests/test_factor_polars.py -q; echo "pytest exit=$?"; test "$(grep -c 'RuntimeError' base/factor_polars.py)" = "0" && test "$(grep -c 'def _maybe_resolve_factor_names' base/factor_polars.py)" = "0" && test "$(grep -c 'Documented precondition' base/factor_polars.py)" = "0" && test "$(grep -c 'pytest.raises(RuntimeError' tests/test_factor_polars.py)" = "0" && echo "GREPS OK"</automated>
    <mutation>
Run each, confirm the named failure, then `git checkout` the file:
1. Restore the no-op `_maybe_resolve_factor_names()` override on `FactorPolars`
   -> `test_factor_names_resolve_dynamically_from_the_lazyframe_schema` MUST
   fail on its PRE-`cal()` assertions.
2. Drop the `_INDEX_COLUMNS` exclusion from `_get_factor_names()` -> the same
   test MUST fail (names would carry `timestamp`/`symbol`).
3. Make `_get_factor_names()` return `tuple(self.config.factor_names)` -> the
   unpinned tests MUST fail at construction.
4. MUST-STAY-GREEN probe: set `_SCHEMA_PROBE_ROWS = 0` -> the file's tests must
   STILL pass, proving the derivation depends on the schema and not on rows.
   Revert to 8 afterwards. If this one goes red, the derivation is reading rows
   it should not be.
    </mutation>
  </verify>
  <done>
`base/factor_polars.py` has no override of the base hook and no unresolved-state
raise; a bare-constructed `Momentum` reports `("momentum_5",)` and
`num_factors == 1`; the pin test passes both halves; all four greps return the
expected counts; mutations 1-3 fail as named and mutation 4 stays green.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: The read-path regression test across both backends</name>
  <files>tests/test_factor_hierarchy.py</files>
  <behavior>
    - Both an `Alpha158SpotKline` (KunQuant) and a `Momentum` (Polars) survive `cal() -> save(mode="w") -> FRESH instance -> read()` driven through `base/model.py`'s `factor_data_strategy="read"` branch.
    - For each fresh factor, uniformly and with no per-backend branch: `_get_factor_names()` is non-empty; `get_factor_names()` is non-empty and agrees with it; `num_factors >= 1`; `get_config()["factor_names"]` is non-empty.
    - The combined `xr.combine_by_coords` result has dims `{timestamp, symbol}` and carries both `KMID` and `momentum_5`.
    - Restoring the deleted no-op override makes THIS test fail while the existing cal-path proof still passes.
  </behavior>
  <action>
Append `test_kunquant_and_polars_factors_are_interchangeable_on_the_read_path`
to `tests/test_factor_hierarchy.py`, directly after
`test_kunquant_and_polars_factors_are_interchangeable_in_one_dlconfig`. It is
that test's sibling: the existing one pins `factor_data_strategy="cal"` and
hand-calls `factor.cal()`, so the `read()` branch of `base/model.py`'s call
surface has never been driven for EITHER backend. That absence is what let the
asymmetry survive a green suite.

Build both factors over one `spot_kline_zarr(periods=60, seed=0)` config exactly
as the neighbouring test does (`Alpha158SpotKline` with `factor_names` pinned to
`[_KUNQUANT_FACTOR_NAME]`; `Momentum` with `kwargs={"n": _MOMENTUM_HORIZON}`),
reusing the module-level constants. Then, in order:

1. `factor.cal().save(mode="w")` for each — the stores now exist under
   `tmp_path`. Pass `mode="w"` explicitly so the test does not lean on
   `to_zarr(mode="a")` create-if-missing semantics.
2. Construct FRESH instances of both classes against the SAME `file_path`s, each
   with a fresh `SpotKlineDataset` over the same dataset config. The Polars one
   leaves `factor_names` unset.
3. Build a `DLConfig` with `factor_data_strategy="read"` and
   `label_data_strategy="read"`, `labels=[]`, over the same date range the
   neighbouring test uses.
4. Replicate `base/model.py`'s four interactions in its own order, with the
   `read` branch this time: `_reset_factors_config` (set start/end on each
   config, call `_reset_dataset_config()`), `_get_features_batch`'s read arm
   (`factor.read().get_features()` then `xr.combine_by_coords`),
   `get_factor_names`, and `get_config`.

Assert uniformly across BOTH fresh factors, in one loop with no per-backend
branch:
  - `factor._get_factor_names()` is non-empty — the surface
    `base/model.py:get_factor_names()` actually calls;
  - `factor.get_factor_names()` is non-empty AND
    `tuple(factor.get_factor_names()) == tuple(factor._get_factor_names())`;
  - `factor.num_factors >= 1`;
  - `"factor_names" in factor.get_config()` and that value is non-empty.

**The public-surface assertion is the one that carries the gap and must not be
dropped.** `_get_factor_names()` DERIVES after Task 2, so it stays green even
with the regression reintroduced; only `get_factor_names()`/`num_factors`, which
read `config.factor_names`, can see it. Asserting only the private one would
reproduce this project's recorded "mechanism proved as a function but never
through its call site" failure — the exact shape that produced this gap.

Then assert the combined dataset has dims `{"timestamp", "symbol"}` and carries
both `_KUNQUANT_FACTOR_NAME` and `f"momentum_{_MOMENTUM_HORIZON}"`.

Follow the neighbouring test's discipline: no runtime type check of any form and
no per-backend branch anywhere in the body — if a branch were needed here,
interchangeability would be dead and the test would be documenting the coupling
D-03 exists to remove. Write the docstring to say so, and to name the mutation
below as the reason this test exists.
  </action>
  <verify>
    <automated>uv run pytest tests/test_factor_hierarchy.py -q; echo "pytest exit=$?"; test "$(awk '/^def test_kunquant_and_polars_factors_are_interchangeable_on_the_read_path/,0' tests/test_factor_hierarchy.py | grep -cE 'isinstance|type\(')" = "0" && echo "NO TYPE DISPATCH OK"; uv run pytest -q; echo "full suite exit=$?"</automated>
    <mutation>
THE load-bearing mutation. Restore `FactorPolars._maybe_resolve_factor_names()`
as a no-op override, then run `uv run pytest tests/test_factor_hierarchy.py -q`.

REQUIRED outcome, both halves:
  - the NEW read-path test FAILS, on the `get_factor_names()` / `num_factors`
    assertion for the Polars factor;
  - `test_kunquant_and_polars_factors_are_interchangeable_in_one_dlconfig`
    (the phase's existing cal-path proof) still PASSES.

The second half is the point: it demonstrates the old suite could not see this
gap and the new test can. If the new test passes under the mutation, it is not
testing the gap — strengthen it until it fails. `git checkout base/factor_polars.py`
afterwards and re-run to confirm green.

Then the full suite: `uv run pytest -q` must report 378 + the tests added by
Tasks 1-3, with zero failures. Read the exit code directly; do not pipe.
    </mutation>
  </verify>
  <done>
The new test exists beside the cal-path proof, passes, contains no runtime type
check and no per-backend branch; the mutation produces both required outcomes;
the full suite is green with no regression against the 378 baseline.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| local filesystem -> process | Zarr/parquet stores under a developer-controlled path are read; no network, no user input, no credentials touched by this change |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-usg-01 | Tampering | `XrBackend.head` writing back to `self.data` | medium | mitigate | Non-mutation test in Task 1 + mutation 3; a truncating probe would silently make `cal()` compute over 8 rows with nothing downstream able to tell |
| T-usg-02 | Information Disclosure | probe read of the dataset store at construction | low | accept | Reads a store the process already opened for the same dataset object; no new path, no new credential, no network |
| T-usg-03 | Denial of Service | bounded read at every `FactorPolars` construction | low | accept | Measured ~25-50 ms, flat in store size (03-VERIFICATION.md Gap 1); accepted knowingly in the disposition |
| T-usg-SC | Tampering | npm/pip/cargo installs | n/a | n/a | No package installs in this plan; `uv add` / `pip install` are forbidden by the constraints |
</threat_model>

<verification>
1. `uv run pytest -q` — 378 baseline plus the new tests, zero failures.
2. `uv run python -c "from base.backend import DataBackend; assert 'head' in DataBackend.__abstractmethods__"` — the bounded read is an interface obligation, not a convenience on one backend.
3. `grep -c 'RuntimeError' base/factor_polars.py` -> 0; `grep -c 'def _maybe_resolve_factor_names' base/factor_polars.py` -> 0.
4. `grep -c '_maybe_resolve_factor_names' tests/test_factor_hierarchy.py` -> unchanged from the pre-task count (the base-class hook contract survives).
5. `git diff --stat -- README.md` -> empty.
6. Every mutation in Tasks 1-3 produced its named failure, and Task 3's mutation additionally left the existing cal-path proof green.
</verification>

<success_criteria>
- A bare-constructed `Momentum`, a `read()` `Momentum` and a `cal()` `Momentum` all report `("momentum_5",)` from both name surfaces.
- `FactorPolars` carries no override of `_maybe_resolve_factor_names()` and no unresolved-state raise.
- `DataBackend` declares the bounded read abstractly; `XrBackend` and `PlBackend` implement it; `BaseDataset` passes it through; no caller reaches past the interface to `xr.open_zarr`.
- Both backends pass the `save() -> fresh instance -> read()` regression test with `factor_data_strategy="read"`.
- Restoring the deleted override reddens the new test and leaves the old cal-path test green.
- Full suite green; `README.md` untouched.
</success_criteria>

<output>
Create `.planning/quick/260906-usg-close-phase-3-gap-1-make-factorpolars-fa/260906-usg-SUMMARY.md` when done.
Record in it: the mutation results table (each mutation, the test that failed, the reason), the final full-suite count, and the decision that names derive from the GRAPH rather than from the factor store on disk.
</output>
