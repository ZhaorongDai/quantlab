---
phase: quick-260907-1du
plan: 01
subsystem: dataset-ingest
tags: [ingest, zarr, cleaning, defect-fix, tdd]
status: complete
requires:
  - base/data.py:BaseDataset._stored_symbol_axis
  - base/data.py:BaseDataset._reset_symbols
  - dataset/cleaning.py:validate_schema
provides:
  - "empty-store construction fallback (BaseDataset._reset_symbols pre-probe)"
  - "one-shot pending-panel handoff (BaseDataset._construction_raw_panel/_window)"
  - "structural-null mask in validate_schema()"
affects:
  - ingest_alpaca.py
  - ingest_tiingo.py
  - ingest_binance_spot.py
  - dataset/spot.py
tech-stack:
  added: []
  patterns:
    - "pre-probe before read(): a condition read() cannot raise on must be tested before read() is called"
    - "one-shot handoff cleared on entry: removes a duplicate without becoming a cache"
    - "structural mask built from the ARGUMENT, not the module constant"
key-files:
  created:
    - tests/test_construction_raw_fallback.py
  modified:
    - base/data.py
    - dataset/cleaning.py
    - tests/test_cleaning.py
decisions:
  - "Defect 2 fixed by a one-shot pending-panel handoff, NOT by deferring name derivation — D-05's construction-time-names contract is untouched and RV-02 stays filed and open."
  - "The handoff is recorded on the probe-fallback path only, never on the read() path, and cleared unconditionally on entry to from_raw_data()."
  - "validate_schema's structural mask is folded over the required_columns ARGUMENT, because SpotKlineDataset._clean() passes Title-Case names."
metrics:
  duration: 41 min
  completed: 2026-09-07
actuals:
  tokens: 21000
  tasks: 3
  commits: 3
---

# Quick Task 260907-1du: Fix Three Ingest-Path Defects Summary

Three defects on the ingest path closed: a present-but-empty destination store now reaches the proven `from_raw_data()` fallback instead of crashing construction; one ingest converts raw exactly once instead of twice; and `validate_schema()` now discriminates a vendor-withheld column from the dense panel's own structural sparsity while null-checking required columns for the first time.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Reach the raw fallback on a present-but-EMPTY store | `b68a5c8` | `base/data.py`, `tests/test_construction_raw_fallback.py` |
| 2 | One ingest runs exactly one raw conversion | `272c934` | `base/data.py`, `tests/test_construction_raw_fallback.py` |
| 3 | Warn on withheld nulls, not structural sparsity | `7bbd1f9` | `dataset/cleaning.py`, `tests/test_cleaning.py` |

## What Changed

### Defect 1 — empty-store crash (`base/data.py:_reset_symbols`)

The fallback keyed off `FileNotFoundError` from `read()`, which an empty store never raises: `read()` SUCCEEDS and then `_filter()` -> `filter_by_symbol` -> `.sel({"symbol": [...]})` raises `ValueError` from *inside* `read()`, because zarr reads a zero-length symbol axis back as `float64`. No `except` clause on `read()` can see that.

The store is now pre-probed with the existing `_stored_symbol_axis()` helper (coordinate-only, lazy, already used against this same store by `_reconcile_new_listings`). A falsy axis — `None` for an absent store or an absent coordinate, `[]` for a present-but-empty one — routes to `from_raw_data()`. The warning names the store path and distinguishes all three cases while retaining the pre-existing `data not found, try to read from csv` wording.

`read()`, `XrBackend.read`, `XrBackend.filter_by_symbol` and `PlBackend.read` are byte-identical to `HEAD` — confirmed by `git diff` touching only `base/data.py` in this task. The inner `try/except FileNotFoundError` is kept rather than made dead: still reachable if the store is removed between the probe and the read.

### Defect 2 — double raw conversion (`base/data.py`)

Two CLASS attributes (`_construction_raw_panel`, `_construction_raw_window`) carry a ONE-SHOT handoff from the construction-time fallback to the first `from_raw_data()` call after it. `from_raw_data()` reads and clears both unconditionally on entry, then returns early only when all three hold: a panel was handed over, the backend still holds that very object (identity, not equality), and the config's `(start_date, end_date)` still match.

Class-level rather than assigned in `__init__` on purpose — `tests/test_dataset_hierarchy.py::test_base_dataset_init_assigns_the_backend_before_the_config` introspects `__init__`'s source for statement order, and a subclass that builds its config outside `BaseDataset.__init__` must not hit a missing attribute. Reads use `getattr(..., None)` so `tests/test_cleaning.py:_NoOpDataset` (which calls `BaseDataset.from_raw_data(self)` without being a subclass) is unaffected.

**Design boundary honoured, as the plan made binding:** `from_raw_data()`'s signature is still `from_raw_data(self) -> Self`; name/symbol derivation still happens eagerly at construction from a full raw materialisation; D-05's construction-time-names contract is untouched; RV-02 stays filed and open. The rejected alternative (`_raw_axes_in_range()`-based cheap `_reset_symbols()` / deferred derivation) was not taken and was never needed.

### Defect 3 — false-positive null warnings (`dataset/cleaning.py:validate_schema`)

The null loop is replaced by a structural mask: a logical AND over `isnull()` of every column in the `required_columns` **argument**, True precisely where no bar exists. A column's unexpected nulls are `isnull() & ~mask`. A column whose dims differ from the mask's falls back to the plain whole-column count rather than broadcasting.

This is a **net widening**: the loop now runs over all of `data.data_vars`, so required columns are null-checked for the first time, and a genuinely all-null column still warns. A `logger.info` now reports the panel's structural sparsity (cell count and percentage) rather than leaving it unreported. Signature, the missing-required-column `ValueError` and the not-raising behaviour are unchanged.

## Observed RED — real captured output

### Task 1 headline test

`tests/test_construction_raw_fallback.py::test_a_present_but_empty_store_falls_back_to_raw`, before the pre-probe:

```
base/data.py:110: in _filter
    self.data_backend.filter_by_symbol("symbol", self.config.symbols)
dataset/backend.py:389: in filter_by_symbol
    self.data = self.data.sel({col: list(symbols)})
...
value = array(['AAPL'], dtype='<U4'), dtype = dtype('<f8')
E           ValueError: could not convert string to float: np.str_('AAPL')
../../.venv/lib/python3.13/site-packages/xarray/core/indexes.py:625: ValueError
...
FAILED tests/test_construction_raw_fallback.py::test_a_present_but_empty_store_falls_back_to_raw
1 failed, 2 passed, 2 warnings in 1.77s
```

Exactly the `ValueError` measured at planning time. The other two Task-1 tests (absent store, populated store) passed before the fix as well as after — they are regression guards on the two behaviours the change must NOT alter.

The fixture asserts its own store shape before construction, so it cannot pass for the wrong reason:

```python
assert reopened["symbol"].dtype == np.dtype("float64")
assert reopened["symbol"].size == 0
```

Probed independently first (`scratchpad/probe_empty.py`): writing a zero-length `object` symbol coordinate to zarr reads back as `symbol dtype: float64 size: 0`.

### Task 2 headline tests

`test_one_ingest_runs_exactly_one_raw_conversion` and `test_the_constructors_panel_is_consumed_once_and_only_once`, before the handoff:

```
>       assert count_raw_conversions() == 1
E       assert 2 == 1
E        +  where 2 = <function count_raw_conversions.<locals>.<lambda> at 0x11bf78cc0>()
>       assert count_raw_conversions() == 1
E       assert 2 == 1
E        +  where 2 = <function count_raw_conversions.<locals>.<lambda> at 0x108fcaf20>()
FAILED tests/test_construction_raw_fallback.py::test_one_ingest_runs_exactly_one_raw_conversion
FAILED tests/test_construction_raw_fallback.py::test_the_constructors_panel_is_consumed_once_and_only_once
2 failed, 5 passed, 3 warnings in 1.23s
```

`2 == 1` is the measured double conversion. Every one of these four tests **counts** `StockDataset._raw_data_to_xr` calls through a `monkeypatch.setattr` counter — none of them times anything, so none can pass because the machine was fast.

### Task 3 headline tests

All four RED before the mask, with the real warning lines:

```
E       AssertionError: ["validate_schema: column 'trade_count' has 1 unexpected null value(s) — not raising, per flag-don't-delete philosophy (D-07).
E         "]
E       assert not True

E       AssertionError: validate_schema: column 'vwap' has 4 unexpected null value(s) — not raising, per flag-don't-delete philosophy (D-07).
E       assert 'has 3 null' in "validate_schema: column 'vwap' has 4 unexpected null value(s) — ..."

E       AssertionError: []
E       assert 0 == 1
E        +  where 0 = len([])

E       AssertionError: ["validate_schema: column 'Quote asset volume' has 1 unexpected null value(s) — not raising, per flag-don't-delete philosophy (D-07).
E         "]
E       assert not True
FAILED tests/test_cleaning.py::test_a_structurally_sparse_panel_logs_no_unexpected_null_warning
FAILED tests/test_cleaning.py::test_a_genuinely_withheld_column_still_warns
FAILED tests/test_cleaning.py::test_a_required_column_null_on_an_existing_bar_now_warns
FAILED tests/test_cleaning.py::test_the_structural_mask_follows_the_callers_required_columns
4 failed, 9 passed in 0.97s
```

Note the third: `AssertionError: []` / `assert 0 == 1` — **no warning at all** for `open` null on a cell where `close` exists. That is the blind spot half of the defect, measured silent, and the widening is what closes it. Note also `vwap` reporting **4** where only **3** cells carry a bar: the old count was inflated by the panel's own sparsity.

## Reddening mutation per test

| Test | Mutation that reddens it | Verified |
|------|--------------------------|----------|
| `test_a_present_but_empty_store_falls_back_to_raw` | remove the pre-probe from `_reset_symbols()` | yes — observed RED pre-fix |
| `test_an_absent_store_still_falls_back_to_raw` | delete the inner `except FileNotFoundError` fallback | named |
| `test_a_populated_store_is_still_read_and_no_raw_conversion_runs` | force the probe branch to always fall back | named |
| `test_one_ingest_runs_exactly_one_raw_conversion` | delete the early return from `from_raw_data()` | yes — observed RED pre-fix (`2 == 1`) |
| `test_the_constructors_panel_is_consumed_once_and_only_once` | do not clear the handoff on entry | yes — observed RED pre-fix (`2 == 1`) |
| `test_a_window_changed_after_construction_forces_a_reconversion` | drop the date-pair comparison | yes — mutation run, test FAILED, reverted |
| `test_a_replaced_backend_panel_forces_a_reconversion` | drop the identity comparison | yes — mutation run, test FAILED, reverted |
| `test_a_structurally_sparse_panel_logs_no_unexpected_null_warning` | restore the plain per-column null count | yes — observed RED pre-fix |
| `test_a_genuinely_withheld_column_still_warns` | invert the mask (count inside instead of outside) | yes — mutation run, test FAILED, reverted |
| `test_a_required_column_null_on_an_existing_bar_now_warns` | restore the `col not in required_columns` filter | yes — observed RED pre-fix (silent, `0 == 1`) |
| `test_the_structural_mask_follows_the_callers_required_columns` | build the mask from `REQUIRED_COLUMNS` instead of the argument (`KeyError: 'open'`) | yes — observed RED pre-fix |

Four mutations were executed against the finished code and each reddened exactly its own test; all four were reverted with `git checkout -- <file>` and the working tree was confirmed clean afterwards.

## Verification

```
uv run pytest -q
425 passed, 115 warnings in 24.01s
```

Baseline was **414 passed**. 425 = 414 + 3 (Task 1) + 4 (Task 2) + 4 (Task 3). No test was removed, skipped or weakened; the three pre-existing `validate_schema` tests pass unchanged.

Every new test builds its own `tmp_path` fixtures. None reads the developer's `data/` store, none touches the network, and none requires `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` or `TIINGO_API_KEY` (T-1du-03 mitigated).

Concurrency guard from the plan: `git status --porcelain` confirms none of `enums/data.py`, `base/acquisition.py`, `acquisition/universe.py`, `tests/test_ticker_pattern_reconciliation.py`, `tests/test_universe.py` appears in any of this plan's three commits.

## Deviations from Plan

**1. [Rule 3 — blocking issue] The absent-store warning had to move into the pre-probe branch**

- **Found during:** Task 1
- **Issue:** The plan requires the pre-probe branch to name the store path and distinguish the empty case, AND requires `test_an_absent_store_still_falls_back_to_raw` to still observe the pre-existing `data not found, try to read from csv` warning. With the pre-probe in place, an absent store no longer reaches the inner `except`, so that message would have disappeared.
- **Fix:** The probe-branch warning retains the exact legacy phrase and appends the path and the case in parentheses — satisfying both requirements rather than trading one off against the other. The inner `except` block's own warning text is untouched.
- **Files modified:** `base/data.py`
- **Commit:** `b68a5c8`

**2. [Scope narrowing, plan-directed] The handoff is recorded only on the probe-fallback path**

- **Found during:** Task 2
- **Issue:** `_reset_symbols()` now has two `from_raw_data()` call sites — the probe fallback and the retained inner `except FileNotFoundError` race path.
- **Fix:** The plan says "immediately after the fallback branch calls `self.from_raw_data()`" and "never on the `read()` path". The handoff is recorded in the probe branch only. The inner-`except` race path therefore still costs a duplicate conversion — conservative, and it cannot regress anything.
- **Files modified:** `base/data.py`
- **Commit:** `272c934`

No architectural changes were needed; no plan decision turned out wrong against the real code.

## Known Stubs

None. No stub, skipped test, `TODO`/`FIXME` or unrun `<verify>` was introduced by this task. Every `<verify>` block in the plan was executed and its real output is recorded above.

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or schema change at a trust boundary. `T-1du-01` and `T-1du-02` are mitigated as planned; `T-1du-SC` had nothing to audit — no dependency was added, removed or upgraded.

## Self-Check: PASSED

Files verified present on disk:

- `base/data.py` — FOUND
- `dataset/cleaning.py` — FOUND
- `tests/test_construction_raw_fallback.py` — FOUND
- `tests/test_cleaning.py` — FOUND

Commits verified in `git log --all`:

- `b68a5c8` — FOUND
- `272c934` — FOUND
- `7bbd1f9` — FOUND
