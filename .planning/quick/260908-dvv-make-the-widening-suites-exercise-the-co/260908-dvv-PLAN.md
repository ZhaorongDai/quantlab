---
phase: quick-260908-dvv
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/conftest.py
  - tests/test_symbol_coord_encoding.py
  - tests/test_variable_axis_widening.py
  - tests/test_symbol_axis_widening.py
  - tests/test_factor_update.py
  - tests/test_widening_fixture_realism.py
  - .planning/todos/pending/2026-09-08-an-empty-zarr-store-records-symbol-as-float64.md
autonomous: true
requirements: [WFR-01, WFR-02, WFR-03, WFR-04, WFR-05, WFR-06, WFR-07, WFR-08, WFR-09, WFR-10, WFR-11, WFR-12, WFR-13, WFR-14]
estimate:
  tokens: 60000
  raw_tokens: 60000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "WFR-01: A test exercising a widening method sees the coordinate dtype a production panel actually carries, and the lock lives in the suite that OWNS the method. `tests/test_symbol_axis_widening.py`, `tests/test_variable_axis_widening.py` and `tests/test_factor_update.py` each run their store-touching tests under BOTH live production symbol encodings. `tests/test_chunked_ingest.py`, where the lock currently sits by accident, is not edited at all."
    - "WFR-02: OPEN QUESTION 1 ANSWERED -- PARAMETRISE, over exactly two encodings, decided from a measured mutation matrix rather than assumed. Two mutations redden DISJOINT arms, so neither arm subsumes the other and neither is cost without coverage. M1 (the shipped defect -- restore `coords={...}` on `widen_data_vars`'s filler, i.e. `git checkout dea1e85 -- quantlab/dataset/backend.py`) reddens ONLY the variable-length arm; the fixed-width arm stays green, as all 34 tests do today. M4 (append `widened = widened.assign_coords({dim: requested})` after the reindex in `widen_symbol_axis`) reddens ONLY the variable-length arm of a post-widen store-encoding assertion (`assert dtype('<U9') == StringDType()`), silently rewriting a variable-length store to fixed width while every pass/fail assertion in both arms stays green. A THIRD arm is refused on the same evidence: at natural label widths `<U3` and `<U9` were measured behaviourally IDENTICAL across the whole battery (widen_data_vars / widen_and_append on either axis / widen_symbol_axis-then-append), so a second fixed-width case would be cost without coverage."
    - "WFR-03: OPEN QUESTION 2 ANSWERED -- YES, a shared helper, and at COORDINATE granularity rather than panel granularity. Panel granularity does not fit: the three suites have three incompatible panel idioms -- `_panel(dates, symbols, offset)` plus `_typed_panel(dates, symbols)` in the symbol suite, `_panel(dates, symbols, variables, offset)` plus `_typed_panel(dates, symbols, name, dtype)` in the variable suite, and a panel built INSIDE `PanelFactor.cal()` in the factor suite. A coordinate-level helper composes with all three because it replaces exactly the one expression they share. It lives in `tests/conftest.py` and is imported as `from tests.conftest import ...`, the repo's own existing idiom (`tests/test_raw_hive_layout.py:341` imports `_STOCK_PQT_COLUMNS` that way, enabled by `[tool.pytest.ini_options] pythonpath = [\".\"]`)."
    - "WFR-04: The helper carries its own self-tests because the realistic construction is COUNTER-INTUITIVE, and getting it wrong reproduces this exact blind spot one level up. Measured 2026-09-08 (numpy 2.5.2 / xarray 2026.7.0 / zarr 3.3.0), in-memory spelling to on-disk result: a python list literal and `np.array(dtype='<U9')` both land on `<U9` / `encoding['dtype']=<U9` / `BytesCodec`; `np.array(dtype=object)` and `pd.Index([...])` land on decoded `StringDType()` / `encoding['dtype']=object` / `VLenUTF8Codec`; and THE TRAP -- `np.array(dtype=np.dtypes.StringDType())`, the spelling that names the dtype production DECODES to, lands on `<U9`, i.e. on the WRONG arm. Only the object spelling reproduces the production store. Without a self-test pinning that, a later reader simplifies the object spelling to the StringDType one and the variable-length arm silently becomes a duplicate of the fixed-width arm with the suite still green."
    - "WFR-05: The production anchor, re-derived live 2026-09-08 rather than inherited from the todo. `data/data/us_equity/1d/stock_alpaca.zarr` symbol dtype `float64`; `data/data/us_equity/1d/us_all.zarr` `<U9` with `encoding['dtype']=<U9`, `BytesCodec`, 7700 symbols, longest label `SATX-WS-A` at 9 chars (so `<U9` is its NATURAL width); `data/data/us_equity/1m/stock_alpaca.zarr` decoded `StringDType()` with `encoding['dtype']=object`, `VLenUTF8Codec`, 102 symbols. That `object`/`StringDType()` pair is verbatim the pair the shipped `ValueError` names. The real ingest was traced end to end and confirms it: `_raw_data_to_xr()` yields `object`, the reindexed window stays `object`, and `from_raw_data_chunked` writes `StringDType()`."
    - "WFR-06: The blind spot was RE-DERIVED live before planning, not taken on trust. With `quantlab/dataset/backend.py` reverted to `dea1e85` (the diff to `c8bba70` is 1 file, 1 hunk, 21 insertions / 6 deletions, entirely the filler's `coords=`), `tests/test_chunked_ingest.py` goes `4 failed, 37 passed` while the three owning suites report `34 passed` -- 11 + 16 + 7. Baseline at HEAD re-measured: the three suites `34 passed in 3.24s`, whole repo `592 passed, 273 warnings in 41.52s`, exit status 0."
    - "WFR-07: THE DEMONSTRATION THE TASK EXISTS FOR, measured in advance and gated in Task 2. Against that same reverted backend, the parametrised suites go RED on exactly 8 tests, every one of them a `[variable_length]` id, in TWO of the three owning suites -- `test_variable_axis_widening.py::test_widen_and_append_reconciles_all_three_axes`, `::test_widen_data_vars_backfills_the_stores_whole_existing_extent`, `::test_the_filler_carries_the_incoming_variables_dtype`, `::test_the_filler_joins_the_stores_existing_chunk_grid`, `::test_an_explicit_fill_widens_a_non_float_variable_and_keeps_its_dtype`, `::test_widen_and_append_still_inherits_the_overlap_refusal_verbatim`, and `test_factor_update.py::test_update_reconciles_a_new_variable_without_being_told_to`, `::test_the_widen_fill_seam_reaches_the_widening_call`. Every `[fixed_width]` twin stays GREEN, which is what makes the RED attributable to the coordinate encoding rather than to the tests being new."
    - "WFR-08: `tests/test_symbol_axis_widening.py` scores ZERO under M1 and that is CORRECT, not a shortfall -- `widen_symbol_axis` never builds the filler, so the shipped defect does not reach it. Recorded so nobody later 'fixes' the zero. That suite therefore gets its OWN mutation-verified lock instead: `assert_stored_symbol_encoding(path, symbol_encoding)` after each widen, proven live to go RED on `[variable_length]` alone under M4 (`assert dtype('<U9') == StringDType()`) while `[fixed_width]` stays green. Without that assertion M4 escapes BOTH arms entirely."
    - "WFR-09: `tests/test_chunked_ingest.py` is not touched. Closing this by adding assertions there is forbidden by the brief -- those already catch it, by accident, and the gap is that the owning suites do not. Proved by `git diff 48e6bf6..HEAD -- tests/test_chunked_ingest.py` being empty, pinned to a SHA taken BEFORE any commit, because GSD commits per task and a working-tree-only check prints nothing in exactly the world where the file was edited and committed (the 03.1 DATA-06 lesson)."
    - "WFR-10: NO production behaviour changes. `quantlab/` is untouched -- `git diff 48e6bf6..HEAD -- quantlab/` and `git status --porcelain -- quantlab/` both empty. Task 2 does check out `dea1e85`'s `backend.py` transiently to run the demonstration, and restores it unconditionally with a following `;` sequencer, with the cleanliness assertion as the gate on that restore."
    - "WFR-11: THE LOOSE THREAD, CLOSED WITH EVIDENCE AND NOT CHASED. `1d/stock_alpaca.zarr` is `{timestamp: 0, symbol: 0}` carrying the full Alpaca variable set -- an EMPTY store, exactly the hypothesis the todo offered. Reproduced in three lines: an empty python list has no string information, so numpy defaults its dtype to `float64`, and writing a 0x0 panel produces a byte-identical store. It is recorded as a diagnosed finding with its one open question (whether an empty store should be written at all) and is NOT parametrised into the widening arms -- it is a degenerate 0x0 shape, not a string encoding."
    - "WFR-12: NO SECOND REAL DEFECT WAS FOUND, and the near-miss is recorded rather than inflated. A `<U9` store meeting an incoming panel whose coordinate is `<U3` does raise `Mismatched dtypes ... Store has dtype <U9 but dataset to append has dtype <U3` -- at HEAD as well as pre-fix -- but that width gap was an artefact of a synthetic fixture. It is unreachable in production on two independent grounds, both measured: `<U9` is the NATURAL width of `us_all.zarr`'s own longest label, and a widen's target axis must be a SUPERSET of the stored one so its natural width can never shrink; and the real ingest hands the widen an `object` coordinate, whose pairing with a `<U9` store was measured to PASS. Re-run with natural label widths, the whole battery is green at HEAD across all three encodings."
    - "WFR-13: Baseline arithmetic. Live baseline 592 passed (re-measured at plan time, exit status 0). 33 of the 34 tests in the three owning suites touch a store and gain a second arm; the sole exemption is `test_update_declares_no_overwrite_parameter`, which is pure `inspect` introspection and would be cost without coverage. 592 + 6 (helper self-tests) + 16 (variable suite second arm) = 614; + 11 (symbol suite) + 6 (factor suite) = 631; + 3 (family guard) = 634. Per-task gates at 614 / 631 / 634."
    - "WFR-14: The full-suite gate CANNOT pass while any test failed or errored. It carries two independent arms -- pytest's own exit status, preserved through `set -o pipefail`, and a ZERO-COUNT assertion over the same tail via the `test`/`grep -c`/`-eq 0` idiom. Both were re-demonstrated live AT 634, in isolation and in both directions: against a clean `634 passed` tail the full gate and each arm alone exit 0; against a `3 failed, 634 passed in 0.19s` tail the full gate exits 1, arm A alone exits 1 and arm B alone exits 1, while the COUNT arm alone exits 0 -- proving the count arm accepts a failing run and cannot stand alone. Near-miss patterns 633, 635, 63, 6340 and 1634 were each confirmed to refuse the real 634 tail. No `!` appears in any verify line, because `zsh -n` returns non-zero with EMPTY stderr for any line containing one."
  artifacts:
    - tests/conftest.py
    - tests/test_symbol_coord_encoding.py
    - tests/test_variable_axis_widening.py
    - tests/test_symbol_axis_widening.py
    - tests/test_factor_update.py
    - tests/test_widening_fixture_realism.py
    - .planning/todos/pending/2026-09-08-an-empty-zarr-store-records-symbol-as-float64.md
  key_links:
    - "`tests/conftest.py:symbol_coord(symbols, encoding)` -> each suite's `coords={... \"symbol\": ...}` expression -> `XrBackend.append` writing the store -> `xr.open_zarr` inside `widen_symbol_axis` / `widen_data_vars`. That chain is the whole point: the helper's output must survive a real zarr write, because the encoding only exists on the far side of one."
    - "`tests/conftest.py:symbol_encoding` (a `params=SYMBOL_COORD_ENCODINGS` fixture) -> every store-touching test in the three owning suites -> two pytest ids per test. The fixture is what makes realism the DEFAULT rather than something each test remembers."
    - "`tests/conftest.py:assert_stored_symbol_encoding(path, encoding)` -> `zarr.open_group(path)['symbol'].dtype` -> the symbol suite's post-widen assertions. This is the ONLY arm that can see M4, which raises nothing and passes every value assertion."
    - "`tests/test_widening_fixture_realism.py` -> `ast` over the three owning suite modules -> the family property itself. The original defect was a whole FAMILY of fixtures sharing one unrealistic assumption; a guard that reads the family, not any single test, is the only thing that keeps it closed."
    - "`tests/test_symbol_coord_encoding.py::test_the_variable_length_arm_matches_what_the_real_chunked_ingest_writes` -> `stock_pqt_row` / `hive_raw_tree` (existing conftest fixtures) -> `StockDataset.from_raw_data_chunked` -> the store's on-disk `symbol` dtype. This is the anchor that keeps the helper tied to production instead of to a claim about production."
---

<objective>
Move the lock on the coordinate-encoding contract from where it sits by
accident -- `tests/test_chunked_ingest.py` -- into the three suites that OWN
the axis-widening methods, and make a realistic coordinate the DEFAULT there
rather than something an individual test remembers to arrange.

The three owning suites build their `symbol` coordinate from python list
literals. A panel that reaches those methods in production does not: it is
written to zarr and re-opened, and the encoding is decided by that round trip.
Re-derived live today, the suites are blind in exactly the way the todo
describes -- with `quantlab/dataset/backend.py` reverted to its pre-fix state
the chunked-ingest suite reports `4 failed, 37 passed` while the three owning
suites report `34 passed`.

This is not the shape of the ten gates this repository has shipped that passed
for the wrong reason. Every gate here is CORRECT. What is wrong is that a whole
FAMILY of fixtures shares one unrealistic assumption, so the suite is rigorous
only inside it, and no amount of care spent on any individual test would have
surfaced it. The plan is shaped around that: a shared helper so realism is the
default, a parametrisation decided from a measured mutation matrix rather than
assumed, a self-test on the helper because the realistic construction is
counter-intuitive, and a structural guard that reads the FAMILY rather than any
single test.

Purpose: a defect in a widening method must be caught by the suite that owns
that method, on the coordinate encoding a production panel actually carries.
Output: a shared encoding helper with its own self-tests, 33 parametrised
tests across the three owning suites, a family guard, and a recorded diagnosis
of the `float64` loose thread. No file under `quantlab/` is edited.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/todos/pending/2026-09-08-widening-fixtures-bypass-the-real-coordinate-encoding-path.md
@.planning/STATE.md
@CLAUDE.md
@tests/conftest.py
@tests/test_symbol_axis_widening.py
@tests/test_variable_axis_widening.py
@tests/test_factor_update.py
@quantlab/dataset/backend.py
@quantlab/base/data.py
</context>

<measured_evidence>

Everything below was measured on this machine on 2026-09-08, before planning,
with numpy 2.5.2 / xarray 2026.7.0 / zarr 3.3.0. Nothing here is inherited
from the todo without re-derivation.

## Baselines

| Reading | Value |
|---|---|
| whole repo at HEAD | `592 passed, 273 warnings in 41.52s`, exit 0 |
| three owning suites at HEAD | `34 passed in 3.24s` (11 + 16 + 7) |
| three owning suites, backend reverted to `dea1e85` | `34 passed` -- the blind spot, confirmed still open |
| `tests/test_chunked_ingest.py`, same reverted backend | `4 failed, 37 passed` -- the incidental lock, confirmed |
| `git diff dea1e85 c8bba70 -- quantlab/dataset/backend.py` | 1 file, 1 hunk, 21 insertions / 6 deletions, entirely the filler's `coords=` |

## The three production dtypes, re-derived

| Store | zarr dtype | decoded | `encoding['dtype']` | serializer | sizes |
|---|---|---|---|---|---|
| `1d/stock_alpaca.zarr` | `float64` | `float64` | -- | -- | `{timestamp: 0, symbol: 0}` |
| `1d/us_all.zarr` | `<U9` | `<U9` | `<U9` | `BytesCodec` | 7700 symbols, longest `SATX-WS-A` (9 chars) |
| `1m/stock_alpaca.zarr` | `StringDType()` | `StringDType()` | `object` | `VLenUTF8Codec` | 102 symbols |

The real ingest, traced end to end: `_raw_data_to_xr()` -> `object`; the
reindexed window -> `object`; `from_raw_data_chunked` writes `StringDType()`.

## The round-trip mapping the helper exists to encode

| in-memory spelling | zarr | decoded | `encoding['dtype']` | serializer |
|---|---|---|---|---|
| python list literal | `<U9` | `<U9` | `<U9` | `BytesCodec` |
| `np.array(dtype="<U9")` | `<U9` | `<U9` | `<U9` | `BytesCodec` |
| `np.array(dtype=object)` | `StringDType()` | `StringDType()` | `object` | `VLenUTF8Codec` |
| `np.array(dtype=np.dtypes.StringDType())` | `<U9` | `<U9` | `<U9` | `BytesCodec` |
| `pd.Index([...])` | `StringDType()` | `StringDType()` | `object` | `VLenUTF8Codec` |

Row 4 is the trap. The spelling that NAMES the dtype production decodes to is
the one that reproduces the wrong arm.

## The mutation matrix that answers "parametrise?"

Battery = `widen_data_vars` alone; `widen_and_append` adding a symbol;
`widen_and_append` adding a variable; `widen_symbol_axis` then a plain append.
Run over three store encodings, at natural label widths.

| Mutation | fixed-width arm | variable-length arm |
|---|---|---|
| none (HEAD) | all green | all green |
| M1 -- the shipped defect | **all green** | **`widen_data_vars` and `widen_and_append`-with-a-new-variable RAISE** |
| M4 -- `assign_coords` after the reindex | all green, store stays `<U9` | no raise, every value assertion green, but the store is **silently rewritten `StringDType()` -> `<U9`** |

Two mutations, two disjoint reddening arms. Neither arm subsumes the other,
so parametrising over both is coverage rather than cost. A third arm is
refused on the same evidence: `<U3` and `<U9` were measured identical across
the whole battery once the label widths are natural, and `float64` is a 0x0
empty store rather than a string encoding.

M4 also proves something the pass/fail battery cannot: it is invisible unless
a test asserts the store's POST-WIDEN on-disk encoding. That assertion is a
required deliverable, not a nicety.

## The demonstration, pre-measured

The three owning suites with their `symbol` coordinate switched to `object`,
against the `dea1e85` backend: `8 failed, 26 passed`. The eight are named in
WFR-07. At HEAD with the same object coordinate: `34 passed`. So the arm is
green when the code is right and red when it is wrong, on both halves.

M4 against a post-widen encoding assertion, parametrised: `1 failed, 1 passed`,
the failure being `[variable_length]` with `assert dtype('<U9') == StringDType()`.

## The loose thread, closed

`1d/stock_alpaca.zarr` is `{timestamp: 0, symbol: 0}` with the full Alpaca
variable set. An empty python list carries no string information so numpy
defaults it to `float64`; writing a 0x0 panel reproduces the store exactly.
An empty store, as the todo guessed. Recorded, not chased.

## The gate, re-verified at 634

| Probe | Result |
|---|---|
| full gate vs `634 passed in 0.18s` | exit 0 |
| full gate vs `3 failed, 634 passed in 0.19s` | exit 1 |
| arm A (exit status) alone, failing tail | exit 1 |
| arm B (zero failure count) alone, failing tail | exit 1 |
| COUNT arm alone, failing tail | **exit 0 -- accepts it; cannot stand alone** |
| patterns 633 / 635 / 63 / 6340 / 1634 vs the clean 634 tail | each exit 1 |

</measured_evidence>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: the shared encoding helper, its self-tests, and the first owning suite wired onto it end to end</name>
  <files>tests/conftest.py, tests/test_symbol_coord_encoding.py, tests/test_variable_axis_widening.py</files>
  <reversibility rating="reversible">Test-only. Every edit is additive or a signature change inside `tests/`; reverting is one `git revert`.</reversibility>
  <behavior>
    - The helper's fixed-width arm round-trips through a real zarr write to a byte-encoded unicode store whose `encoding['dtype']` is fixed-width and whose serializer is `BytesCodec`.
    - The helper's variable-length arm round-trips to a decoded `StringDType()` store whose `encoding['dtype']` is `object` and whose serializer is `VLenUTF8Codec` -- verbatim the pair the shipped `ValueError` names.
    - The two arms are DISTINCT on disk, so the parametrisation cannot silently collapse into two copies of one arm.
    - The spelling that names the decoded dtype directly does NOT reproduce the variable-length store; it lands on the fixed-width one.
    - The variable-length arm equals what `StockDataset.from_raw_data_chunked` actually writes, built through the existing `stock_pqt_row` / `hive_raw_tree` fixtures.
    - A bare python list reproduces exactly one arm -- the fixed-width one -- which is the diagnosis of why 34 tests were blind.
  </behavior>
  <action>
Add to `tests/conftest.py`, which needs a plain top-level `import zarr` line
alongside its existing numpy / pandas / xarray imports:

`SYMBOL_COORD_ENCODINGS`, declared at module level exactly as
`SYMBOL_COORD_ENCODINGS = ("fixed_width", "variable_length")` -- the two names
are load-bearing, because they become the pytest parametrisation ids that
three verify lines count and that the mutation table names. An adjacent
comment carries the production anchor from `<measured_evidence>`: which real
store on this machine carries each encoding, with its decoded dtype, its
`encoding['dtype']` and its serializer.

`symbol_coord(symbols, encoding)`, defined as a module-level `def` (so the
anchored gate on it holds) and returning a numpy array. For the
fixed-width name it returns `np.asarray(list(symbols))`, letting numpy resolve
the natural `<U` width -- deliberately byte-identical to what the suites' list
literals already produce, so that arm is an unchanged CONTROL. For the
variable-length name it returns `np.array(list(symbols), dtype=object)`. Any
other name raises with a message naming the two accepted ones. The docstring
carries the full round-trip table from `<measured_evidence>` and states, in
terms a reader cannot miss, that the `np.dtypes.StringDType()` spelling is the
one that looks right and lands on the wrong arm.

`stored_symbol_dtype(path)`, also a module-level `def`, returning
`zarr.open_group(path, mode="r")["symbol"].dtype`
-- the ON-DISK dtype, not the decoded one, because the decoded value is what
hid the defect.

`assert_stored_symbol_encoding(path, encoding)`, a module-level `def`
asserting that the store at
`path` still carries the encoding it was built with, with a failure message
naming both dtypes. This is the only thing that can observe a widen which
raises nothing and changes every value correctly while rewriting the encoding.

`symbol_encoding`, a module-level `def` decorated
`@pytest.fixture(params=SYMBOL_COORD_ENCODINGS)`, returning `request.param`, so
the pytest ids are literally `[fixed_width]` and `[variable_length]`.

Create `tests/test_symbol_coord_encoding.py` with EXACTLY six tests, importing
the helpers via `from tests.conftest import ...` (the idiom
`tests/test_raw_hive_layout.py` already uses):
`test_the_fixed_width_arm_lands_on_a_bytes_encoded_unicode_store`,
`test_the_variable_length_arm_lands_on_the_object_encoded_string_store`,
`test_the_two_arms_are_distinct_on_disk`,
`test_the_obvious_string_dtype_spelling_does_not_reproduce_the_production_store`,
`test_the_variable_length_arm_matches_what_the_real_chunked_ingest_writes`,
`test_a_bare_python_list_reproduces_only_the_fixed_width_arm`.
Each is unparametrised -- they are about the helper itself, and both arms are
already named inside them. Each docstring names the mutation that reddens it;
for the fourth that mutation is swapping the variable-length arm to the
StringDType spelling, which reddens this test and the third one together. The
fifth builds a small hive tree through `stock_pqt_row` and `hive_raw_tree`,
runs `StockDataset(...).from_raw_data_chunked(granularity="year")`, and
compares the resulting store's on-disk symbol dtype with the helper's
variable-length arm -- the anchor that keeps the helper tied to production
rather than to a claim about it.

Rewire `tests/test_variable_axis_widening.py`. Its `_panel` and `_typed_panel`
gain a required `encoding` parameter and source their symbol coordinate from
`symbol_coord`; nothing else about either builder changes, so the fixed-width
arm reproduces today's behaviour exactly. All sixteen tests take the
`symbol_encoding` fixture and thread it through every builder call. The module
docstring gains a paragraph stating why: the suite owns `widen_data_vars` and
`widen_and_append`, the shipped defect lived in the first of those, and the
suite could not see it because a list literal is one encoding out of the two
that are live on disk. Do not change any assertion's meaning -- this task adds
an axis to the fixtures, it does not restate what the tests claim.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && set -o pipefail && uv run pytest tests/test_symbol_coord_encoding.py tests/test_variable_axis_widening.py -q && SELF="$(uv run pytest tests/test_symbol_coord_encoding.py --collect-only -q 2>/dev/null | grep -o '::test_' | wc -l | tr -d ' ' || true)" && VAR_VL="$(uv run pytest tests/test_variable_axis_widening.py --collect-only -q 2>/dev/null | grep -o '\[variable_length\]' | wc -l | tr -d ' ' || true)" && VAR_FW="$(uv run pytest tests/test_variable_axis_widening.py --collect-only -q 2>/dev/null | grep -o '\[fixed_width\]' | wc -l | tr -d ' ' || true)" && printf 'SELF=%s VAR_VL=%s VAR_FW=%s\n' "$SELF" "$VAR_VL" "$VAR_FW" && test "$SELF" -eq 6 && test "$VAR_VL" -eq 16 && test "$VAR_FW" -eq 16 && test "$(grep -oE '^def symbol_coord\(' tests/conftest.py | wc -l | tr -d ' ' || true)" -eq 1 && test "$(grep -oE '^def stored_symbol_dtype\(' tests/conftest.py | wc -l | tr -d ' ' || true)" -eq 1 && test "$(grep -oE '^def assert_stored_symbol_encoding\(' tests/conftest.py | wc -l | tr -d ' ' || true)" -eq 1 && test "$(grep -oE '^def symbol_encoding\(' tests/conftest.py | wc -l | tr -d ' ' || true)" -eq 1 && test "$(grep -oE '^SYMBOL_COORD_ENCODINGS = \("fixed_width", "variable_length"\)$' tests/conftest.py | wc -l | tr -d ' ' || true)" -eq 1 && test "$(grep -oE '^import zarr$' tests/conftest.py | wc -l | tr -d ' ' || true)" -eq 1 && TAIL="$(uv run pytest tests/ -q 2>&1 | tail -2)"; RC=$?; printf '%s\n' "$TAIL"; test "$RC" -eq 0 && grep -qE '(^|[^0-9])614 passed' <<<"$TAIL" && test "$(grep -cE '[0-9]+ (failed|error)' <<<"$TAIL")" -eq 0 && test -z "$(git status --porcelain -- quantlab/)"</automated>
  </verify>
  <done>
    `tests/conftest.py` exports the five named symbols and imports zarr. The
    six self-tests pass. `tests/test_variable_axis_widening.py` collects 32
    ids, 16 per arm, and passes. The full suite GATES on 614 passed (592 live
    baseline + 6 + 16) through the two-armed gate re-verified at 634, and
    `quantlab/` is untouched.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: the remaining two owning suites, the encoding-preservation lock, and the reverted-backend demonstration</name>
  <files>tests/test_symbol_axis_widening.py, tests/test_factor_update.py</files>
  <precondition>`git rev-parse --verify dea1e85` resolves and `git status --porcelain -- quantlab/` is empty before the task runs; the demonstration checks that commit's `backend.py` out transiently and restores it.</precondition>
  <behavior>
    - Every store-touching test in the symbol and factor suites runs under both live encodings.
    - After a widen, the store still carries the encoding it was built with -- the only observable that can see a widen which raises nothing and rewrites the coordinate.
    - Against `quantlab/dataset/backend.py` at `dea1e85`, the three owning suites go red on exactly eight tests, every one a `[variable_length]` id, across the variable and factor suites; every `[fixed_width]` twin stays green.
  </behavior>
  <action>
Rewire `tests/test_symbol_axis_widening.py` the same way Task 1 rewired its
sibling: `_panel` and `_typed_panel` gain a required `encoding` parameter
sourced from `symbol_coord`, and all eleven tests take the `symbol_encoding`
fixture. Then add the lock this suite needs for itself: after each call to
`widen_symbol_axis` or `widen_and_append` that is expected to SUCCEED, assert
`assert_stored_symbol_encoding(path, symbol_encoding)`. At minimum
`test_widening_preserves_history_for_pre_existing_symbols`,
`test_widening_does_not_misattribute_when_the_symbol_count_is_unchanged` and
`test_the_chunk_grid_survives_a_widen` carry it. The module docstring records
why this suite needs its own lock: the shipped defect lived in the filler that
`widen_symbol_axis` never builds, so this suite scoring zero against that
defect is correct rather than a shortfall, and its guarantee comes instead
from the encoding-preservation assertion, whose reddening mutation is stated
in each carrying test's docstring.

Rewire `tests/test_factor_update.py`. `PanelFactor.cal()` sources its symbol
coordinate from `symbol_coord` using an encoding the test supplies -- pass it
through `_factor(...)` onto an attribute `cal()` reads, or as a `cal()`
keyword; either is fine as long as the encoding reaches the coords expression
through the helper. Six of the seven tests take the `symbol_encoding` fixture.
`test_update_declares_no_overwrite_parameter` does NOT: it is pure `inspect`
introspection, never builds a panel and never opens a store, so a second id
for it would be a case with identical behaviour -- cost without coverage. Note
that exemption in its docstring so it reads as a decision rather than an
oversight.

Then run the demonstration and record it in the summary: check `backend.py`
out at `dea1e85`, run the three owning suites, restore, and confirm the red
set is exactly the eight tests named in WFR-07, all `[variable_length]`. The
verify line does this and gates on it; the restore uses a `;` sequencer so it
runs whatever happened before it, and the worktree-clean assertion is what
gates the restore rather than trusting it.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && set -o pipefail && OUT="$(mktemp)" && git checkout dea1e85 -- quantlab/dataset/backend.py && uv run pytest tests/test_symbol_axis_widening.py tests/test_variable_axis_widening.py tests/test_factor_update.py -q > "$OUT" 2>&1; git checkout HEAD -- quantlab/dataset/backend.py; RED="$(grep -o '^FAILED' "$OUT" | wc -l | tr -d ' ' || true)"; VL="$(grep '^FAILED' "$OUT" | grep -o '\[variable_length\]' | wc -l | tr -d ' ' || true)"; printf 'RED=%s VL=%s\n' "$RED" "$VL"; rm -f "$OUT"; test "$RED" -eq 8 && test "$VL" -eq 8 && test -z "$(git status --porcelain -- quantlab/)" && SYM_VL="$(uv run pytest tests/test_symbol_axis_widening.py --collect-only -q 2>/dev/null | grep -o '\[variable_length\]' | wc -l | tr -d ' ' || true)" && FAC_VL="$(uv run pytest tests/test_factor_update.py --collect-only -q 2>/dev/null | grep -o '\[variable_length\]' | wc -l | tr -d ' ' || true)" && FAC_N="$(uv run pytest tests/test_factor_update.py --collect-only -q 2>/dev/null | grep -o '::test_' | wc -l | tr -d ' ' || true)" && printf 'SYM_VL=%s FAC_VL=%s FAC_N=%s\n' "$SYM_VL" "$FAC_VL" "$FAC_N" && test "$SYM_VL" -eq 11 && test "$FAC_VL" -eq 6 && test "$FAC_N" -eq 13 && test "$(grep -o 'assert_stored_symbol_encoding' tests/test_symbol_axis_widening.py | wc -l | tr -d ' ' || true)" -ge 4 && test "$(grep -o 'symbol_coord' tests/test_factor_update.py | wc -l | tr -d ' ' || true)" -ge 2 && TAIL="$(uv run pytest tests/ -q 2>&1 | tail -2)"; RC=$?; printf '%s\n' "$TAIL"; test "$RC" -eq 0 && grep -qE '(^|[^0-9])631 passed' <<<"$TAIL" && test "$(grep -cE '[0-9]+ (failed|error)' <<<"$TAIL")" -eq 0 && DRIFT="$(git diff 48e6bf6..HEAD --name-only -- quantlab/ tests/test_chunked_ingest.py)"; test -z "$DRIFT"</automated>
  </verify>
  <done>
    The symbol suite collects 22 ids and the factor suite 13 (six parametrised
    plus the one introspection test). The encoding-preservation assertion is
    present on the symbol suite's successful widens. Against `backend.py` at
    `dea1e85` the three suites go red on exactly 8 tests and all 8 carry the
    `[variable_length]` id -- the demonstration the task exists for, gated
    rather than narrated. `backend.py` is restored, `quantlab/` and
    `tests/test_chunked_ingest.py` are unchanged since 48e6bf6, and the full
    suite GATES on 631 passed (614 + 11 + 6).
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: the family guard, and the float64 loose thread recorded</name>
  <files>tests/test_widening_fixture_realism.py, .planning/todos/pending/2026-09-08-an-empty-zarr-store-records-symbol-as-float64.md</files>
  <behavior>
    - A store-touching test in any of the three owning suites that stops requesting the encoding fixture fails the guard.
    - A symbol coordinate rebuilt from a bare sequence anywhere in the three suites fails the guard.
    - Collapsing the shared fixture to a single encoding fails the guard.
  </behavior>
  <action>
Create `tests/test_widening_fixture_realism.py` with EXACTLY three tests, all
operating on the three owning suite modules through `ast` rather than through
text scanning -- a source-text scan would be defeated by a docstring that
merely mentions the old spelling, which is the same trap that has bitten this
repository's gates before.

`test_every_store_touching_test_in_the_owning_suites_requests_the_encoding_fixture`:
parse each of the three modules; for every top-level `test_*` function that
reaches a panel builder or a store operation, assert the encoding fixture is
among its parameter names. The exemption set is a frozenset holding exactly
the one introspection test named in Task 2, and the assertion message routes a
would-be changer to that decision before the literal.

`test_no_owning_suite_builds_a_symbol_coordinate_from_a_bare_sequence`: for
every `ast.Dict` in the three modules that carries a `symbol` key, assert the
paired value node is a call to the shared helper. This is the guard that reads
the FAMILY rather than any single test, and it is what makes the original
failure mode -- a whole family sharing one unrealistic assumption -- impossible
to reintroduce by adding a new test that forgets.

`test_the_shared_fixture_offers_exactly_the_two_live_production_encodings`:
literal-equality against the two-name tuple, following the D-09 precedent in
this repository, with a message that sends a would-be changer to the measured
production dtype table rather than to the literal. This is what stops the
parametrisation being quietly reduced to one arm, which would reopen the blind
spot without failing anything else.

Then write the loose-thread record at the todo path above. It states what was
measured -- the store is `{timestamp: 0, symbol: 0}` carrying the full Alpaca
variable set, and an empty python list defaults to `float64` in numpy so a 0x0
panel reproduces it byte for byte -- names it as diagnosed rather than open,
and leaves exactly one question for whoever picks it up: whether an empty
store should be written at all, or its coordinate pinned to a string dtype
when it is. It explicitly says this was noticed during fixture work and not
investigated further, per the brief.

Also record in the SUMMARY, not as a defect but as a near-miss with its
disproof: the `<U9`-store-meets-`<U3`-panel raise, why it is unreachable in
production on two independent measured grounds, and the fact that no second
real defect was found.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && set -o pipefail && uv run pytest tests/test_widening_fixture_realism.py -q && GUARD="$(uv run pytest tests/test_widening_fixture_realism.py --collect-only -q 2>/dev/null | grep -o '::test_' | wc -l | tr -d ' ' || true)" && printf 'GUARD=%s\n' "$GUARD" && test "$GUARD" -eq 3 && test "$(grep -oE '^import ast$' tests/test_widening_fixture_realism.py | wc -l | tr -d ' ' || true)" -eq 1 && test -f .planning/todos/pending/2026-09-08-an-empty-zarr-store-records-symbol-as-float64.md && test "$(grep -o 'timestamp: 0' .planning/todos/pending/2026-09-08-an-empty-zarr-store-records-symbol-as-float64.md | wc -l | tr -d ' ' || true)" -ge 1 && TAIL="$(uv run pytest tests/ -q 2>&1 | tail -2)"; RC=$?; printf '%s\n' "$TAIL"; test "$RC" -eq 0 && grep -qE '(^|[^0-9])634 passed' <<<"$TAIL" && test "$(grep -cE '[0-9]+ (failed|error)' <<<"$TAIL")" -eq 0 && DRIFT="$(git diff 48e6bf6..HEAD --name-only -- quantlab/ tests/test_chunked_ingest.py)"; DIRTY="$(git status --porcelain -- quantlab/ tests/test_chunked_ingest.py)"; test -z "$DRIFT" && test -z "$DIRTY"</automated>
  </verify>
  <done>
    The three family-guard tests pass and read the suites through `ast`. The
    loose-thread record exists with its measured diagnosis. The full suite
    GATES on 634 passed (631 + 3). `quantlab/` and `tests/test_chunked_ingest.py`
    are provably unchanged since 48e6bf6, in both the commit range and the
    working tree.
  </done>
</task>

</tasks>

<mutation_table>

| # | Mutation | Predicted result | Basis |
|---|---|---|---|
| M1 | `git checkout dea1e85 -- quantlab/dataset/backend.py` (restore `coords={...}` on `widen_data_vars`'s filler -- the shipped defect) | SPLIT across DISTINCT functions, all on `[variable_length]` only: `test_variable_axis_widening.py::test_widen_and_append_reconciles_all_three_axes`, `::test_widen_data_vars_backfills_the_stores_whole_existing_extent`, `::test_the_filler_carries_the_incoming_variables_dtype`, `::test_the_filler_joins_the_stores_existing_chunk_grid`, `::test_an_explicit_fill_widens_a_non_float_variable_and_keeps_its_dtype`, `::test_widen_and_append_still_inherits_the_overlap_refusal_verbatim`, `test_factor_update.py::test_update_reconciles_a_new_variable_without_being_told_to`, `::test_the_widen_fill_seam_reaches_the_widening_call`. Every `[fixed_width]` twin GREEN. `test_symbol_axis_widening.py` GREEN in both arms | Measured: `8 failed, 26 passed` with the object coordinate against `dea1e85`; `34 passed` with the list coordinate against the same backend; `34 passed` with the object coordinate at HEAD. Gated by Task 2's verify |
| M2 | `test_widen_data_vars_refuses_a_non_float_variable_without_a_fill` stays GREEN under M1 | GREEN, deliberately -- the refusal fires before the filler is written, so it is unreachable by this defect | Measured; recorded so its absence from the red set is not read as a gap |
| M4 | Append `widened = widened.assign_coords({dim: requested})` after the reindex in `widen_symbol_axis` | `test_symbol_axis_widening.py`'s encoding-preservation assertion RED on `[variable_length]` only (`assert dtype('<U9') == StringDType()`); `[fixed_width]` GREEN; every value assertion in both arms GREEN | Measured live on a standalone parametrised probe: `1 failed, 1 passed`. This is the mutation that escapes a pass/fail battery entirely and is the reason the assertion is a deliverable |
| M5 | Swap the helper's variable-length arm to `np.array(syms, dtype=np.dtypes.StringDType())` | `test_symbol_coord_encoding.py::test_the_two_arms_are_distinct_on_disk` and `::test_the_obvious_string_dtype_spelling_does_not_reproduce_the_production_store` both RED | Measured: that spelling writes `<U9`, identical to the fixed-width arm. This is the mutation that would silently re-create the blind spot inside the fix |
| M6 | Drop `symbol_encoding` from any store-touching test's signature | `test_widening_fixture_realism.py::test_every_store_touching_test_in_the_owning_suites_requests_the_encoding_fixture` RED | Structural, by construction of the AST guard |
| M7 | Reduce `SYMBOL_COORD_ENCODINGS` to one name | `::test_the_shared_fixture_offers_exactly_the_two_live_production_encodings` RED, and the three per-suite id counts in Tasks 1-2 verify lines all fall to zero for the removed arm | Structural plus the collect-only counts |

</mutation_table>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| fixture -> zarr store | the only place the coordinate encoding is decided; a fixture that never crosses it cannot see the property under test |
| test suite -> the reader's belief about coverage | 34 green tests asserted a guarantee they did not hold; that gap is the asset being repaired |
| plan -> `quantlab/` | this task must not cross it at all |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-dvv-01 | Spoofing | a fixture claiming to be realistic while carrying the wrong encoding | critical | mitigate | The six helper self-tests, anchored to a store built by the real `from_raw_data_chunked`, plus mutation M5 which is the exact way this would happen |
| T-dvv-02 | Tampering | a widen that raises nothing and silently rewrites the store's coordinate encoding | high | mitigate | `assert_stored_symbol_encoding` after every successful widen in the symbol suite; proven by M4 to be the only arm that can see it |
| T-dvv-03 | Repudiation | the parametrisation quietly reduced to one arm, leaving the suite green | high | mitigate | `test_the_shared_fixture_offers_exactly_the_two_live_production_encodings` (literal equality, D-09 precedent) plus the per-suite collect-only id counts in all three verify lines; mutation M7 |
| T-dvv-04 | Information Disclosure | a new widening test added later that forgets the encoding axis, reopening the family blind spot one test at a time | high | mitigate | `tests/test_widening_fixture_realism.py`'s AST family guard, which reads the family rather than any single test; mutation M6 |
| T-dvv-05 | Elevation of Privilege | a test-integrity task quietly changing production behaviour | critical | mitigate | `git diff 48e6bf6..HEAD --name-only -- quantlab/` and `git status --porcelain -- quantlab/` both asserted empty in Tasks 2 and 3, with the SHA pinned before any commit per the 03.1 DATA-06 lesson |
| T-dvv-06 | Tampering | the transient `dea1e85` checkout in Task 2 left in place | high | mitigate | The restore runs behind a `;` sequencer so it is unconditional, and the worktree-clean assertion GATES it rather than trusting it |
| T-dvv-07 | Denial of Service | the parametrisation doubling suite runtime | low | accept | Measured: the three suites run in 3.24s today; doubling 33 of 34 tests lands near 7s against a 41.5s whole-suite baseline |
| T-dvv-SC | Tampering | npm/pip/cargo installs | n/a | accept | No package is installed by this plan. Every import used -- numpy, pandas, xarray, zarr, pytest, ast, inspect -- is already a live dependency exercised by the current suite |
</threat_model>

<verification>
1. `uv run pytest tests/ -q` -> **634 passed** (592 live baseline + 6 + 16 + 11
   + 6 + 3), with each task gating on its own running total (614 / 631 / 634)
   rather than merely printing it. The gate carries TWO independent arms --
   pytest's exit status, kept out of the pipe by `set -o pipefail`, and a
   zero-count assertion over the same tail -- and each was re-demonstrated
   live AT 634 to refuse a `3 failed, 634 passed` run, while the count arm
   alone was demonstrated to ACCEPT it.
2. No `!` appears in any verify line. `test "$(grep -c ...)" -eq 0` is used
   throughout, because `zsh -n` returns non-zero with EMPTY stderr for any
   line containing one and that is indistinguishable from a parse error.
3. The demonstration the brief demands is a GATE, not a narrative: Task 2's
   verify checks `backend.py` out at `dea1e85`, runs the three owning suites,
   restores unconditionally, and requires `RED=8` and `VL=8` -- eight red
   tests, every one of them a `[variable_length]` id.
4. `quantlab/` and `tests/test_chunked_ingest.py` are proved untouched against
   a SHA pinned before the first commit (48e6bf6), in both the commit range
   and the working tree.
5. The two open questions are answered in `must_haves` with the measured
   mutation matrix that decided each, not with a preference.
6. The one structural warning the plan checker raises (R4 -- a fallible `git`
   whose status is discarded) is DELIBERATE and is the single place it must
   be. Task 2's restore of `backend.py` sits behind a `;` sequencer precisely
   so it runs whatever happened before it; discarding its status is what makes
   it unconditional. Its status is not trusted either -- the following
   `test -z "$(git status --porcelain -- quantlab/)"` is the gate on the
   restore, so a restore that silently failed reddens the task.
</verification>

<success_criteria>
- 634 tests pass, through a gate that provably cannot pass while any test
  failed or errored, re-verified in both directions at that exact number.
- Reverting `quantlab/dataset/backend.py` to `dea1e85` reddens 8 tests in the
  suites that OWN the widening methods -- today it reddens none of them --
  and every one of the 8 is a `[variable_length]` id whose `[fixed_width]`
  twin stays green.
- `tests/test_symbol_axis_widening.py` carries its own mutation-verified lock
  (M4) rather than borrowing the variable suite's, because the shipped defect
  genuinely does not reach `widen_symbol_axis`.
- Realism is the DEFAULT: a new widening test cannot omit the encoding axis
  without failing the family guard.
- `tests/test_chunked_ingest.py` is not edited, and no file under `quantlab/`
  is edited.
- The `float64` loose thread is recorded with its measured diagnosis and not
  investigated further.
</success_criteria>

<output>
Create `.planning/quick/260908-dvv-make-the-widening-suites-exercise-the-co/260908-dvv-SUMMARY.md` when done
</output>
</content>
</invoke>
