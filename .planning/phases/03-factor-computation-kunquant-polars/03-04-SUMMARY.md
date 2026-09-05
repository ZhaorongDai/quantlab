---
phase: 03-factor-computation-kunquant-polars
plan: 04
subsystem: factor-layer
tags: [polars, lazyframe, xarray, factor, momentum, d-04, d-05, d-06, d-07, d-08]

# Dependency graph
requires:
  - phase: 03-factor-computation-kunquant-polars
    plan: 01
    provides: "The spot_kline_zarr synthetic-Zarr fixture and the tests/test_factor_polars.py scaffold that locks the raw Title-Case `Close` column contract this plan's example factor is written against"
  - phase: 03-factor-computation-kunquant-polars
    plan: 02
    provides: "base/factor.py:Factor(ABC), the PolarsFactorConfig dataclass, and the _maybe_resolve_factor_names() seam this plan overrides"
provides:
  - "base/factor_polars.py:FactorPolars(Factor) — a batch-only second factor backend whose subclasses write factor logic as Polars lazy expressions (FACTOR-03, D-03)"
  - "The one-hook subclass contract `_get_factor_lazyframe(lf) -> pl.LazyFrame` carrying only timestamp/symbol/factor columns, deferred until cal() (D-04)"
  - "Dynamic factor-name resolution from the computed frame's own schema, with no upfront declaration (D-05)"
  - "cal() converting to xr.Dataset before anything leaves the class, so the module boundary stays xarray-only on both backends (D-06, FACTOR-04)"
  - "factor/momentum.py:Momentum — the D-08 worked example, a real config-driven per-symbol N-day momentum signal"
  - "base/factor_polars.py under the core-layer purity check in tests/test_extensibility_contract.py"
affects: [03-05, 04-return-model, 06-portfolio-optimization, polars-factor-backend, model-checkpoint-reload]

actuals:
  tokens: 5521
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Hook-takes-its-input: `_get_factor_lazyframe(lf)` receives the frame as a parameter (unlike `_get_factor_func()`, which builds its own Input nodes), which is what makes a factor's logic unit-testable against a hand-built frame with no Dataset and no store on disk"
    - "Runtime laziness proof by monkeypatching `pl.LazyFrame.collect` to raise — a contract clause that would otherwise degrade silently into a per-call performance cliff becomes an immediate hard failure"
    - "Resolution-timing override: `_maybe_resolve_factor_names()` overridden to a no-op so constructing a factor performs no I/O; names are assigned inside `cal()` from `collect_schema().names()`"
    - "`collect_schema().names()` as the schema-only primitive — it inspects without materializing, so name resolution lives inside the laziness contract rather than breaking it"

key-files:
  created:
    - base/factor_polars.py
    - factor/momentum.py
  modified:
    - tests/test_factor_polars.py
    - tests/test_extensibility_contract.py

key-decisions:
  - "The class docstring describes the sibling relationship without naming FactorKunQuant, because Task 1's own acceptance criterion requires `grep -c KunQuant base/factor_polars.py` to return 0 — the literal check is the enforceable half of 'this module carries no KunQuant dependency'"
  - "`_INDEX_COLUMNS = ('timestamp', 'symbol')` is a module constant shared by the name-resolution step and the xarray conversion, so the two places that must agree on what an index column is cannot drift apart"
  - "`Momentum.horizon` is a small public property rather than an inline kwargs lookup, so the horizon is readable from outside the hook and the hook body stays about the expression chain"
  - "Momentum sorts by ['symbol', 'timestamp'] BEFORE shifting: `.shift(n).over('symbol')` is per-symbol only if row order within each partition is chronological"

patterns-established:
  - "Mutation-checked contract tests (continued from 03-02/03-03): the D-04 laziness proof was run against a deliberately materializing `Momentum._get_factor_lazyframe`, observed to fail, then reverted"
  - "A second backend's tests build their config directly against a tmp_path fixture rather than calling the production `config/` factory, keeping wave-parallel plans free of runtime coupling"

requirements-completed: [FACTOR-03, FACTOR-04]

coverage:
  - id: D1
    description: "Momentum.cal().get_features() returns an xr.Dataset over [timestamp, symbol] whose data_vars are exactly ['momentum_5'] with finite values — the Polars backend computes a real factor and hands it back as xarray (FACTOR-03, FACTOR-04, D-06), with no raw price/volume column leaking through (D-04)"
    requirement: FACTOR-03
    verification:
      - kind: integration
        ref: "tests/test_factor_polars.py#test_momentum_cal_returns_xarray_dataset_with_only_factor_columns"
        status: pass
      - kind: other
        ref: "Numeric spot-check during execution: res['momentum_5'][t=10, symbol=S2USDT] == Close[10,2]/Close[5,2] - 1 exactly (np.isclose True) — the per-symbol window is correct, not merely non-empty"
        status: pass
    human_judgment: false
  - id: D2
    description: "Factor names are resolved dynamically from the computed lazyframe's schema after cal() (get_factor_names() == ('momentum_5',), num_factors == 1); before cal() _get_factor_names() raises RuntimeError naming the precondition (D-05, 03-RESEARCH.md Pitfall 4)"
    requirement: FACTOR-03
    verification:
      - kind: unit
        ref: "tests/test_factor_polars.py#test_factor_names_resolve_dynamically_from_the_lazyframe_schema"
        status: pass
    human_judgment: false
  - id: D3
    description: "Nothing inside _get_factor_lazyframe materializes — the hook returns a pl.LazyFrame whose schema is readable while pl.LazyFrame.collect is patched to raise (D-04, threat T-03-04-03)"
    requirement: FACTOR-03
    verification:
      - kind: unit
        ref: "tests/test_factor_polars.py#test_get_factor_lazyframe_stays_lazy_until_cal (mutation-checked: appending .collect().lazy() to Momentum's hook fails it, 1 failed / 2 passed, then reverted)"
        status: pass
    human_judgment: false
  - id: D4
    description: "FactorPolars and Momentum expose none of cal_stream/init_stream/_make/_make_stream — the Polars backend is batch-only by decision, not by omission (D-07)"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_factor_polars.py#test_polars_backend_exposes_no_streaming_surface"
        status: pass
    human_judgment: false
  - id: D5
    description: "FactorPolars and FactorKunQuant are true siblings — both subclass the shared Factor base, neither subclasses the other, and FactorPolars is abstract on exactly _get_factor_lazyframe (FACTOR-03, D-03)"
    requirement: FACTOR-04
    verification:
      - kind: other
        ref: "uv run python -c 'issubclass(FactorPolars, Factor) and issubclass(FactorKunQuant, Factor) and not issubclass(FactorPolars, FactorKunQuant)' -> siblings ok; inspect.isabstract(FactorPolars) and '_get_factor_lazyframe' in __abstractmethods__ -> ok"
        status: pass
    human_judgment: false
  - id: D6
    description: "base/factor_polars.py is registered in CORE_LAYER_FILES and passes the core-layer purity check — it names no concrete Dataset subclass and no market literal, and imports no compiled-graph symbol"
    requirement: FACTOR-04
    verification:
      - kind: unit
        ref: "tests/test_extensibility_contract.py#test_core_layer_purity_no_market_specific_logic (2 passed)"
        status: pass
      - kind: other
        ref: "grep -c 'KunQuant' base/factor_polars.py -> 0"
        status: pass
    human_judgment: false
  - id: D7
    description: "Momentum is a readable one-file worked example of the FactorPolars contract, written entirely as Polars window expressions with a config-driven horizon and no import from config/"
    requirement: FACTOR-03
    verification:
      - kind: other
        ref: "inspect.getsource(Momentum._get_factor_lazyframe) contains 'over(' and 'shift(' -> ok; grep -c '^from config\\|^import config' factor/momentum.py -> 0; issubclass(Momentum, FactorPolars) and not inspect.isabstract(Momentum) -> ok"
        status: pass
    human_judgment: false

# Metrics
duration: 8 min
completed: 2026-09-05
status: complete
---

# Phase 3 Plan 04: The Polars Factor Backend Summary

**A batch-only second factor backend, `FactorPolars(Factor)`, whose subclasses write factor logic as a single Polars lazy-expression hook — with factor names read from the computed frame's own schema, an `xr.Dataset` at the module boundary, no streaming surface, and every one of those clauses proven by a test (including a monkeypatch-based laziness proof that was mutation-checked) rather than asserted in prose.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-09-05T18:43Z (approx.)
- **Completed:** 2026-09-05T18:51:00Z
- **Tasks:** 3
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments

- **Delivered FACTOR-03 as a genuine sibling backend, not a variant.** `base/factor_polars.py:FactorPolars` subclasses the `Factor` base 03-02 extracted, carries its own `cal()`, and declares exactly one abstract member — `_get_factor_lazyframe(lf)`. `FactorPolars` and `FactorKunQuant` share a base and neither descends from the other, which is what makes a Polars factor droppable into `DLConfig.factors` with zero `base/model.py` edits (D-03).

- **Implemented the user's own stated contract clause by clause.** The user asked for a hook shaped like `_get_factor_func()` that returns a lazyframe of only symbol/date/factor columns and starts computing when `cal()` is called — that is `_get_factor_lazyframe(lf)` plus a `cal()` that collects. The user also observed that "Polars 不会返回 Function() 和因子名称" — so names are extracted from the frame's own columns: `collect_schema().names()` minus `{timestamp, symbol}` (D-05).

- **Made the laziness clause enforceable instead of aspirational.** The D-04 test replaces `pl.LazyFrame.collect` with a stub that raises, then calls the hook directly against a hand-built frame with no `Dataset` and no store on disk. It also asserts that `collect_schema().names()` still works under that stub — schema inspection must not materialize either, which is precisely why `cal()` can resolve names inside the laziness contract rather than in violation of it. The test was mutation-checked: appending `.collect().lazy()` to `Momentum`'s hook makes it fail, and only it.

- **Kept the xarray-only boundary intact on both backends (D-06 / FACTOR-04).** `cal()` converts the collected frame to an `xr.Dataset` before it leaves the class, using the exact idiom already in `dataset/backend.py:PlBackend.get_xarray_dataset()`. Polars is an internal implementation detail of one backend; nothing downstream can tell which backend produced a factor store.

- **Shipped a worked example that is actually worked (D-08).** `factor/momentum.py:Momentum` computes `Close_t / Close_{t-n} - 1` per symbol via `.shift(n).over("symbol")` over a `["symbol", "timestamp"]`-sorted frame, with `n` from `config.kwargs`. A numeric spot-check during execution confirmed the value at `(t=10, S2USDT)` equals `Close[10,2]/Close[5,2] - 1` exactly — the window really is per-symbol, not a global shift that happens to look plausible.

- **Put the new core-layer module under the standing purity check.** `base/factor_polars.py` joined `tests/test_extensibility_contract.py:CORE_LAYER_FILES`, closing the one item 03-02's SUMMARY explicitly assigned to this plan.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement FactorPolars(Factor) — the batch-only Polars factor backend** — `882583c` (feat)
2. **Task 2: Ship the Momentum example factor (D-08)** — `1a0f9be` (feat)
3. **Task 3: Prove the Polars contract — computation, laziness, dynamic names, xarray boundary** — `c2db142` (test)

**Plan metadata:** see the `docs(03-04)` commit that carries this SUMMARY.

## Files Created/Modified

- `base/factor_polars.py` — **created** (127 lines). `FactorPolars(Factor)` with `__init__`, the no-op `_maybe_resolve_factor_names()` override, `_get_factor_names()` with its documented precondition error, the abstract `_get_factor_lazyframe()` and `cal()`. Module-level `_INDEX_COLUMNS` constant. Imports: `abstractmethod`, `Self`, `polars`, `xarray`, `PolarsFactorConfig`, `Factor`, `Timer` — nothing else. No streaming member, no `config_cls`, no compiled-graph helper.
- `factor/momentum.py` — **created** (73 lines). `Momentum(FactorPolars)`: constructor taking `factor_config`, a `horizon` property reading `config.kwargs["n"]` (default 20), the `_get_factor_lazyframe` hook, and the standard `_get_features` passthrough / `_get_labels` refusal tail using the implicit-closure `__class__` idiom the other factor classes use.
- `tests/test_factor_polars.py` — **modified.** Grew from 1 test to 5: the 03-01 scaffold test is untouched, and four contract proofs plus a module-private `_momentum_config()` helper were added. The module docstring's "the real content lands in 03-04" paragraph was updated to record that it has landed.
- `tests/test_extensibility_contract.py` — **modified.** One line: `"base/factor_polars.py"` added to `CORE_LAYER_FILES`. Nothing else in the file changed.

## Decisions Made

1. **The class docstring names the sibling relationship without writing "KunQuant".** Task 1's action text asks the docstring to say `FactorPolars` is "the Polars sibling of `FactorKunQuant`", while Task 1's own acceptance criterion requires `grep -c "KunQuant" base/factor_polars.py` to return `0`. Those cannot both hold literally. The grep is the enforceable, machine-checkable half of the real requirement ("this module has no compiled-graph dependency"), so it won: the docstring says "a sibling — not a descendant — of the compiled-graph backend declared in `base/factor.py`", which conveys the same fact and points at the same file. See Deviations.

2. **`_INDEX_COLUMNS` is a module constant, used by both the name-resolution step and the xarray conversion.** Those two steps must agree on what counts as an index column — if they ever drift, `cal()` would either register `timestamp` as a factor or fail to index by it. One constant makes the drift impossible rather than merely unlikely.

3. **`Momentum.horizon` is a small property, not an inline `kwargs.get`.** The hook body stays about the expression chain, and the horizon is readable from outside without reaching into `config.kwargs`. It is used (by the hook), so it is not an unused member under QUAL-02; it is nonetheless one member more than the plan literally described, disclosed in Deviations.

4. **`Momentum` sorts before it shifts.** `.shift(n).over("symbol")` is a positional shift within each partition — it means "n bars earlier for this symbol" only if row order within the partition is chronological. `Dataset.get_lazyframe()` derives its row order from `xr.Dataset.to_dataframe()`, which this code does not want to depend on. The explicit `sort(["symbol", "timestamp"])` makes the guarantee local to the factor.

5. **The tests build `PolarsFactorConfig` directly rather than calling `config.momentum_config()`.** That factory (owned by parallel plan 03-03) points at production data paths; these tests run against the `tmp_path`-scoped `spot_kline_zarr` fixture. Result: zero runtime coupling between the two wave-3 plans in either direction, as the plan's file-ownership note requires.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Task 1's action text and its own acceptance criterion contradict each other on the string "KunQuant"**

- **Found during:** Task 1
- **Issue:** The action says the class docstring must cover "that it is the Polars sibling of `FactorKunQuant`". The acceptance criterion says `grep -c "KunQuant" base/factor_polars.py` returns `0`. Writing the docstring as specified would fail the criterion; passing the criterion means not writing that literal name.
- **Fix:** Satisfied the criterion and preserved the action's intent: the docstring reads "A sibling -- not a descendant -- of the compiled-graph backend declared in `base/factor.py`", and the module docstring explains that the isolation exists so "the Polars path carries no dependency on the batch/stream graph machinery that lives in `base/factor.py`". The underlying requirement — no compiled-graph import in this module — is fully met and now grep-enforced.
- **Files modified:** `base/factor_polars.py` (docstrings only)
- **Verification:** `grep -c "KunQuant" base/factor_polars.py` → `0`; the class docstring still states the sibling relationship and points at the file that holds the other backend
- **Commit:** `882583c`

**2. [Rule 1 - Bug] Task 3's `-k lazy` acceptance count is off by two (substring match, not exact match)**

- **Found during:** Task 3
- **Issue:** The criterion states `uv run pytest tests/test_factor_polars.py -k lazy -q` shows **1 passed**. `-k lazy` is a substring filter and matches three of the five tests, because `lazyframe` contains `lazy`: `test_get_factor_lazyframe_stays_lazy_until_cal`, `test_factor_names_resolve_dynamically_from_the_lazyframe_schema` and `test_spot_kline_lazyframe_exposes_raw_title_case_columns`. Actual result is **3 passed, 2 deselected**.
- **Fix:** No code change — the criterion's *substantive* half (that the test genuinely fails if `Momentum._get_factor_lazyframe` is edited to materialize) was executed as specified: `.collect().lazy()` was appended to the hook's expression chain, `pytest -k lazy` reported **1 failed, 2 passed** with only `test_get_factor_lazyframe_stays_lazy_until_cal` failing, and the edit was reverted with `git checkout -- factor/momentum.py`. The mutation isolating exactly one failing test is stronger evidence than the count would have been.
- **Files modified:** none
- **Verification:** mutation applied → 1 failed / 2 passed; reverted → 3 passed / 2 deselected
- **Commit:** n/a (verification-only)

### Disclosed structural addition

**`Momentum.horizon` property.** The plan says to "Read `n` from `self.config.kwargs` with a default of 20". That is done, but through a two-line `horizon` property rather than inline in the hook. It is used by the hook (so not an unused member), and it keeps the hook body focused on the expression chain. Recording it here because it is one public member more than the plan literally enumerated.

**Total deviations:** 2 auto-fixed (both Rule 1 — internal contradictions in the plan's own acceptance criteria, neither in implementation logic) + 1 disclosed structural choice.
**Impact on plan:** None on scope or outcome. Every `<success_criteria>` item is met, and every acceptance criterion passed as written except the two corrected above.

## Verification Results

Plan-level `<verification>`, re-run after all three commits:

| Command | Expected | Actual |
|---|---|---|
| `uv run pytest tests/ -q` | zero failures | **63 passed** |
| `uv run pytest tests/test_factor_polars.py -q` | 5 passed | **5 passed** |
| `issubclass(FactorPolars, Factor) and issubclass(FactorKunQuant, Factor) and not issubclass(FactorPolars, FactorKunQuant)` | siblings ok | **siblings ok** |
| `inspect.isabstract(FactorPolars)` + `'_get_factor_lazyframe' in __abstractmethods__` | ok | **ok** |
| `hasattr(FactorPolars, 'cal_stream'/'init_stream'/'_make'/'_make_stream')` | all False | **all False** |
| `grep -c "KunQuant" base/factor_polars.py` | 0 | **0** |
| `uv run pytest tests/test_extensibility_contract.py -q` | all passed with the new file in CORE_LAYER_FILES | **2 passed** |
| `issubclass(Momentum, FactorPolars)` + `not inspect.isabstract(Momentum)` | ok | **ok** |
| `inspect.getsource(Momentum._get_factor_lazyframe)` contains `over(` and `shift(` | ok | **ok** |
| `grep -c "^from config\|^import config" factor/momentum.py` | 0 | **0** |
| Numeric spot-check: `momentum_5[t=10, S2USDT] == Close[10,2]/Close[5,2] - 1` | equal | **equal (`np.isclose` True)** |

**Mutation check** (Task 3 acceptance criterion — applied, observed, then reverted with `git checkout -- factor/momentum.py`):

| Mutation | Test | Result |
|---|---|---|
| Append `.collect().lazy()` to `Momentum._get_factor_lazyframe`'s expression chain | `test_get_factor_lazyframe_stays_lazy_until_cal` | **FAILED as required** (1 failed / 2 passed; only the laziness test broke) |

The suite grew from 59 to 63 tests (4 added; the run was already at 59 after Task 1 because 03-03 had landed).

## Known Stubs

None introduced by this plan.

`PolarsFactorConfig` remains field-free (03-02's deliberate choice) and needed no field here. `FactorPolars` defines no `_get_features`/`_get_labels` — it inherits `Factor`'s pre-existing `NotImplementedError` bodies, and `Momentum` overrides both, exactly as every other concrete factor class does.

One scaffold item recorded in 03-01/03-02 remains open and is already assigned:

| Item | Status | Resolved by |
|---|---|---|
| KunQuant/Polars interchangeability integration test in `tests/test_factor_hierarchy.py` | Not written here — `FactorPolars` now exists, so 03-05 is unblocked to write it | **03-05** |

## Live Follow-Up: 03-RESEARCH.md Open Question 2

**`Dataset.get_lazyframe()` has no per-market column-normalization contract, so a Polars factor written against one market's raw column names is not portable to another without a per-market branch.**

Unlike `_to_kunquant()` — which each `Dataset` subclass overrides specifically to rename its columns to KunQuant's lowercase `open/high/low/close/volume/amount` before building a graph — `get_lazyframe()` returns whatever the underlying Zarr store holds, un-renamed. Verified shapes:

- `SpotKlineDataset` → `['timestamp', 'symbol', 'Close', 'High', 'Low', 'Open', 'Quote asset volume', 'Volume']` (Binance Title-Case)
- `StockDataset` → lowercase raw names plus an `adj*` group

`Momentum` is therefore written against Title-Case `Close` and would need a per-market branch (or a rename step) to run over US equities. This is recorded in `Momentum`'s class docstring and in `FactorPolars._get_factor_lazyframe`'s docstring, at the two points a new factor author actually reads.

**Deliberately out of scope for Phase 3** (D-08 requires *one* example factor, and 03-CONTEXT.md's Polars discussion never raises normalization). It becomes worth solving the moment a second Polars factor needs to run across both markets — the natural shape being a `get_lazyframe()`-side counterpart to `_to_kunquant()`'s per-subclass rename. Flagged for whichever phase adds that second factor.

## Issues Encountered

None blocking. Two internal contradictions in the plan's own acceptance criteria were found and resolved in-flight — see Deviations. Every fact the `<interfaces>` block asserted about the `cal()` chain (the five-step sequence, `collect_schema()` being schema-only, the `from_dataframe` idiom, the eager raw load) held exactly as written; nothing needed re-derivation.

## User Setup Required

None — zero new packages installed (`polars>=1.44.1` was already declared and installed), no environment variable, no external service, no network call. Threat `T-03-04-SC` stands as `accept`: no package legitimacy checkpoint applies.

## Next Phase Readiness

**Ready for 03-05.** Wave 3 is complete: both wave-3 plans (03-03's dual-market KunQuant factors and this one's Polars backend) have landed with the full suite green at 63 tests.

- **03-05** (streaming, BUG-02's aarch64 SIMD block width, the interchangeability integration test) now has a real second backend to test interchangeability *against*. `tests/test_factor_hierarchy.py`'s outstanding item — an integration test proving `base/model.py` consumes a `FactorPolars` and a `FactorKunQuant` identically — is unblocked; `config.momentum_config()` (delivered by 03-03) plus `factor.momentum.Momentum` (delivered here) now compose into a runnable Polars factor.
- **Phase 4** inherits 03-02's checkpoint-reload flag unchanged, and this plan makes it concrete rather than hypothetical: `utils/module.py:18` hardcodes `FactorConfig(**config)`, so a checkpoint saved from a `Momentum` factor will raise `TypeError` on the missing mandatory `mode`/`data_columns` when reloaded. That is 03-RESEARCH.md Open Question 1, resolved as "defer to Phase 4"; no Phase-3 success criterion touches it and no `config_cls` ClassVar was added here either, for the same QUAL-02 reason 03-02 gave.

**Requirement status:** neither FACTOR-03 nor FACTOR-04 was marked Complete by this plan — `requirements ready-ids` reports `0/2 ready`, because 03-05 also declares both (`requirements: [FACTOR-02, FACTOR-03, FACTOR-04]`) and has not produced its SUMMARY yet. Both flip Complete when 03-05 lands. This is the shared-ID gate working as intended, not a gap.

**Concerns:** none blocking. The one live follow-up is Open Question 2 above, which is a portability limit on future Polars factors, not a defect in anything shipped.

---
*Phase: 03-factor-computation-kunquant-polars*
*Completed: 2026-09-05*

## Self-Check: PASSED

All 4 key files verified present on disk (`base/factor_polars.py`, `factor/momentum.py`, `tests/test_factor_polars.py`, `tests/test_extensibility_contract.py`); all three task commits (`882583c`, `1a0f9be`, `c2db142`) verified present in `git log --all`; full suite re-run green (63 passed).
