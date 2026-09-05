---
phase: 03-factor-computation-kunquant-polars
plan: 03
subsystem: factor-layer
tags: [kunquant, alpha101, alpha158, us-equity, tiingo, xarray, normalization, cross-sectional, d-09]

# Dependency graph
requires:
  - phase: 03-factor-computation-kunquant-polars
    plan: 01
    provides: "Green batch KunQuant .cal() path, the stock_zarr synthetic Zarr fixture, and tests/test_factor_kunquant.py with its _factor_config() helper"
  - phase: 03-factor-computation-kunquant-polars
    plan: 02
    provides: "The Factor/FactorKunQuant hierarchy and the BaseFactorConfig/FactorConfig/PolarsFactorConfig three-way config split this plan's new classes and factories sit on"
provides:
  - "factor/alpha158.py:Alpha158Stock — the Alpha158 factor set computable in batch mode against US equities (D-01, FACTOR-01 across both markets)"
  - "dataset/stock.py:StockDataset._to_kunquant() amount = volume * close synthesis — the single central fix that unbreaks every KunQuant factor class reading US-equity data (D-02)"
  - "factor/alpha101.py:Alpha101Stock repaired — the Input('amount') node that ends the verified RuntimeError: Bad inputs crash"
  - "The D-09 four-class normalization matrix, recorded in every factor class docstring and locked by test_normalization_matrix_matches_recorded_strategy_types (NORM-01)"
  - "config/__init__.py:stock_alpha101_config / stock_alpha158_config / momentum_config — keyword-only, _data_root()-derived factories"
affects: [03-04, 03-05, 04-return-model, 06-portfolio-optimization, cross-sectional-normalization]

actuals:
  tokens: 6199
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Central boundary synthesis: a derived column every consumer needs is computed once in the Dataset's _to_kunquant() boundary method, double-guarded (requested AND absent), never duplicated per factor class"
    - "Deliberate near-duplication with a single recorded divergence: Alpha158Stock replicates its sibling verbatim so the one intentional difference (normalization) is the only thing a reader has to reason about"
    - "Design-choice docstrings + a matrix lock test: a per-class strategy decision that looks like an inconsistency gets written into the docstring AND asserted as an equality against a literal, so 'aligning' the classes fails a test with a message explaining why"
    - "Helper widening over helper duplication: _factor_config() gained a dataset_cls parameter (defaulting to the incumbent) rather than being copied for the second market"

key-files:
  created: []
  modified:
    - dataset/stock.py
    - factor/alpha101.py
    - factor/alpha158.py
    - config/__init__.py
    - tests/test_factor_kunquant.py

key-decisions:
  - "The amount = volume * close proxy lives in StockDataset._to_kunquant(), not in each factor class — one insertion point serves Alpha101Stock, Alpha158Stock and every future US-equity KunQuant factor"
  - "The proxy is double-guarded (only when 'amount' is in data_columns AND absent from data.data_vars) so it is inert on today's paths and cannot overwrite a real vendor column if one ever appears"
  - "Alpha158Stock replicates Alpha158SpotKline's double-AllData build and six declared-but-unused Input(...) nodes verbatim rather than tidying either class — D-01 fixes the invocation shape across markets, so tidying one alone would silently diverge them"
  - "Stock factor tests that compile a KunQuant graph pass 8 symbols; the stock_zarr fixture's 2-symbol default trips KunQuant's SIMD block-width requirement with 'RuntimeError: Bad shape at open'"
  - "The normalization lock is a single equality assertion against a four-entry literal, not four independent asserts — a matrix is one decision, and a diff of the whole matrix is what a future reader needs to see"

patterns-established:
  - "Mutation-checked introspection guards (continued from 03-02): the NORM-01 lock was proven to fail against an Alpha158Stock deliberately wrapped in WindowedZScore, then reverted"
  - "Config factories for a second market are keyword-only (`*,`) and take explicit market/frequency parameters that thread into the underlying dataset factory"

requirements-completed: [FACTOR-01]

coverage:
  - id: D1
    description: "Alpha101Stock(...).cal() no longer raises RuntimeError: Bad inputs, given <class 'NoneType'> — it returns an xr.Dataset of computed factor values for US equities"
    requirement: FACTOR-01
    verification:
      - kind: integration
        ref: "tests/test_factor_kunquant.py#test_alpha101_stock_bugfix_batch_cal_returns_xarray_dataset"
        status: pass
    human_judgment: false
  - id: D2
    description: "Alpha158Stock exists and .cal() returns an xr.Dataset for US equities, so the Alpha158 factor set is computable in batch mode across BOTH markets (D-01)"
    requirement: FACTOR-01
    verification:
      - kind: integration
        ref: "tests/test_factor_kunquant.py#test_alpha158_stock_batch_cal_returns_xarray_dataset"
        status: pass
      - kind: other
        ref: "vars() method-set equality between Alpha158Stock and Alpha158SpotKline -> ok"
        status: pass
    human_judgment: false
  - id: D3
    description: "StockDataset._to_kunquant() synthesizes amount = volume * close (adjusted dollar-volume) once and centrally whenever amount is requested and absent (D-02)"
    requirement: FACTOR-01
    verification:
      - kind: unit
        ref: "tests/test_factor_kunquant.py#test_stock_to_kunquant_synthesizes_amount_as_adjusted_dollar_volume"
        status: pass
    human_judgment: false
  - id: D4
    description: "The synthesis is inert when amount is not requested — the returned dict has no amount key and every other array is identical to the amount-requested run (T-03-03-01)"
    verification:
      - kind: unit
        ref: "tests/test_factor_kunquant.py#test_stock_to_kunquant_without_amount_leaves_arrays_unchanged"
        status: pass
    human_judgment: false
  - id: D5
    description: "The normalization each factor class applies is recorded in that class's docstring as a deliberate time-series-vs-cross-sectional strategy choice, and the four-class D-09 matrix is locked by an automated test (NORM-01)"
    verification:
      - kind: unit
        ref: "tests/test_factor_kunquant.py#test_normalization_matrix_matches_recorded_strategy_types (mutation-checked: wrapping Alpha158Stock's Output in WindowedZScore fails it)"
        status: pass
      - kind: other
        ref: "All four class __doc__ strings assert-checked for 'NORM-01' + 'D-09'; both *Stock docstrings for 'raw'"
        status: pass
    human_judgment: false
  - id: D6
    description: "config/__init__.py exposes stock_alpha101_config(), stock_alpha158_config() and momentum_config(), all deriving paths from _data_root() and never hardcoding an absolute path"
    verification:
      - kind: unit
        ref: "tests/test_factor_kunquant.py#test_stock_alpha158_config_wires_amount_and_a_derived_path + #test_momentum_config_returns_a_polars_factor_config"
        status: pass
      - kind: other
        ref: "grep -c '_data_root()' config/__init__.py 10 -> 11; grep -c 'from factor.momentum' config/__init__.py == 0"
        status: pass
    human_judgment: false

# Metrics
duration: 7 min
completed: 2026-09-05
status: complete
---

# Phase 3 Plan 03: Dual-Market KunQuant Factors Summary

**A one-line central `amount = volume * close` synthesis in `StockDataset._to_kunquant()` plus one `Input("amount")` node that together end the verified `RuntimeError: Bad inputs` crash on every US-equity KunQuant factor, a new `Alpha158Stock` that makes the Alpha158 set dual-market, and D-09's normalization matrix written into all four class docstrings and locked by a mutation-checked test.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-09-05T18:35:32Z
- **Completed:** 2026-09-05T18:43:00Z
- **Tasks:** 3
- **Files modified:** 5 (all modified, none created)

## Accomplishments

- **Fixed the single central cause of the US-equity factor crash, at the boundary rather than per class.** `Alpha101.AllData.__init__` and `Alpha158.AllData.__init__` both build `vwap = Div(self.amount, ...)` unconditionally, so a `None` `amount` raises `RuntimeError: Bad inputs, given <class 'NoneType'>` before any factor is evaluated. Tiingo ships no dollar-volume column. Rather than patch each factor class, the `volume * close` proxy is synthesized once in `StockDataset._to_kunquant()` — so `Alpha101Stock`, the new `Alpha158Stock`, and every future US-equity KunQuant factor get it for free. The rename runs first, so the proxy is **adjusted** dollar-volume, consistent with the rest of the adjusted-price pipeline.

- **Made the Alpha158 set genuinely dual-market (D-01, FACTOR-01).** `Alpha158Stock` mirrors `Alpha158SpotKline` method-for-method — verified by a `vars()` method-set equality assertion, not just by eye — including the double-`AllData` build and the six declared-but-unused `Input(...)` nodes. Those are replicated on purpose: D-01 fixes the invocation shape across markets, so tidying one class alone would silently diverge them and tidying both would edit working code for cosmetics.

- **Turned a locked domain decision into an enforced invariant.** The absence of a rolling z-score on the US-equity classes previously *looked* like an inconsistency, and a previous reading of this codebase treated it as a bug. It is now impossible to make that mistake silently: all four class docstrings state which normalization the class applies and why, and `test_normalization_matrix_matches_recorded_strategy_types` fails with a message that names D-09 and tells the reader to update the decision record before the test literal.

- **Kept every new config path derived, never hardcoded.** `config/__init__.py` is historically where per-developer absolute paths leaked into this repo (Phase 1, CLEAN-03). All three new factories go through `_data_root()`; `grep -c "_data_root()"` rose from 10 to 11 and `tests/test_config_paths.py` stays green.

## Task Commits

Each task was committed atomically:

1. **Task 1: Central dollar-volume proxy, Alpha101Stock amount fix, stock_alpha101_config** — `75f41d6` (fix)
2. **Task 2: Alpha158Stock, remaining config factories, NORM-01 normalization matrix** — `2575f13` (feat)
3. **Task 3: Record the locked D-09 matrix and the Phase-6 deferral** — this SUMMARY (docs)

**Plan metadata:** see the `docs(03-03)` commit that carries this SUMMARY.

## NORM-01 — final locked normalization matrix (D-09)

This section is the authoritative written record of the settled normalization
decision. Tasks 1 and 2 implemented it; this section states it, and
`test_normalization_matrix_matches_recorded_strategy_types` proves the code and
this text agree.

### The matrix

| Class | Market | Wrapper applied around each `Output(...)` | Strategy type served |
|---|---|---|---|
| `Alpha101SpotKline` | crypto spot | rolling z-score over `self.config.window` | 时序 / time-series |
| `Alpha158SpotKline` | crypto spot | rolling z-score over `self.config.window` | 时序 / time-series |
| `Alpha101Stock` | US equities | none — raw alpha values | 截面 / cross-sectional; the consumer applies its own cross-sectional normalization |
| `Alpha158Stock` (new) | US equities | none — raw alpha values | 截面 / cross-sectional; the consumer applies its own cross-sectional normalization |

### The split is per market, not per factor family

Crypto spot is traded with time-series strategies; US equities are traded with
cross-sectional ones. There is no intra-market asymmetry to reconcile and no
open question. This comes from **D-09, a locked user decision** — the user, the
domain expert on this project, ruled directly:

> "WindowedZScore是时序策略用的 截面策略用截面Z 值"
> (WindowedZScore is for time-series strategies; cross-sectional strategies use a cross-sectional Z-score.)

> "美股都用截面策略，Alpha158Stock 也不要时序标准化"
> (US equities all use cross-sectional strategies; Alpha158Stock should not have time-series standardization either.)

`my_ops/preprocess.py:WindowedZScore` normalizes each symbol's factor value
across that symbol's OWN rolling time window — a **time-series** normalization.
A cross-sectional strategy instead normalizes across SYMBOLS at each timestamp.
The presence or absence of the wrapper is therefore a **strategy-type choice,
never a market-dependent defect.**

**A future reader must not "align" a US-equity class with its crypto-spot
sibling.** Doing so would silently impose time-series normalization on a
deliberately cross-sectional factor set — factor values that are normalized in
one class and raw in another are indistinguishable to a downstream consumer at
runtime (threat T-03-03-02), which is exactly why this is written down and
locked rather than left to inspection.

### D-09 supersedes part of D-01

D-01's "调用的步骤完全一致" (identical invocation steps across markets) governs:

- the `Alpha158.AllData(...)` / `Alpha101.AllData(...)` keyword wiring,
- the `build({...})` category dict (`kbar` / `price` / `volume` / `rolling`, including the `exclude` list),
- the double-`AllData` build (names outside the builder, ops inside it),
- the declared-but-unused `Input(...)` nodes in `_get_func_stream()`.

It does **NOT** govern the normalization wrapper. `Alpha158Stock` emits
`Output(v, k)` directly.

### Explicit deferral: no cross-sectional Z op is built in this phase

No cross-sectional normalization op is added to `my_ops/` here. Reasoning,
recorded so it is reviewable:

- `my_ops/` today holds only time-series composite ops (`WindowedZScore`, `WindowedRobustStandardization`, both `WindowedCompositiveOp` subclasses). No cross-sectional op exists in this project.
- KunQuant upstream already supplies the building blocks — `CrossSectionalOp`, `SimpleCrossSectionalOp`, `Rank`, `Scale` — so the extension point exists without this phase inventing an abstraction.
- Normalization is **already selectable per factor class by construction**: it is just an op wrapped around each `Output(...)` inside that class's own `_get_factor_func()`. No new abstraction is needed to make it selectable, so none was added (QUAL-02).
- Building the op belongs with the **ARCH-01/ARCH-02 work in Phase 6** ("架构同时兼容单标的时序策略与多标的截面多因子策略"). ARCH-01/ARCH-02 are Phase 6 requirements, not Phase 3 ones; doing it now would balloon an already refactor-heavy phase.
- D-09 itself defers it: this phase emits raw US-equity factor values and leaves normalization to the downstream consumer.

### Deliberate duplication, recorded

`Alpha158Stock` replicates `Alpha158SpotKline`'s structure rather than either
class being tidied — the double-`AllData` build and the six declared-but-unused
`Input(...)` nodes in `_get_func_stream()` are copied verbatim. D-01 fixes the
invocation shape across markets, so deleting the dead nodes in `Alpha158Stock`
alone would silently diverge the two classes, and deleting them in both would
edit a currently-working class for cosmetics. **The normalization wrapper is the
single deliberate divergence between the two classes.**

### How to change this matrix

`tests/test_factor_kunquant.py:test_normalization_matrix_matches_recorded_strategy_types`
is the automated lock. It derives, per class, whether the rolling z-score op
appears in the source of the method that emits that class's `Output(...)` calls
(`_get_factor_func` for the Alpha101 classes, `_get_func_stream` for the
Alpha158 classes), and compares the whole four-entry dict against a literal in
one equality assertion.

A future change to the matrix must proceed **in this order**:

1. update **D-09** in `.planning/phases/03-factor-computation-kunquant-polars/03-CONTEXT.md`,
2. then the **four class docstrings**,
3. then the **test literal** `_EXPECTED_NORMALIZATION_MATRIX`.

Changing the test literal alone to make a red test go green is changing a locked
user decision without recording it, and the assertion message says so.

## Files Created/Modified

- `dataset/stock.py` — **modified.** `_to_kunquant()` gained the D-02 synthesis between the existing `sortby(...)` and `timestamp = ...` lines, double-guarded on `"amount" in data_columns` and `"amount" not in data.data_vars`. The `drop_vars`, `rename`, `sortby` and per-column `np.ascontiguousarray` loop are byte-identical to before (12 lines added, 0 removed).
- `factor/alpha101.py` — **modified.** `Alpha101Stock._get_factor_func()` gained `amount = Input("amount")` and passes `amount=amount` into `Alpha101.AllData(...)`, so its parameter list now matches `Alpha101SpotKline`'s. Both classes gained NORM-01 docstrings. No normalization wrapper added or removed anywhere.
- `factor/alpha158.py` — **modified.** `Alpha158Stock` added (five methods plus the `_get_labels` refusal and `_get_features` passthrough tail); `Alpha158SpotKline` gained its NORM-01 docstring. `Alpha158SpotKline`'s bodies are untouched.
- `config/__init__.py` — **modified.** `StockDataset` and `PolarsFactorConfig` imports; `stock_alpha101_config()`, `stock_alpha158_config()` and `momentum_config()` added, all keyword-only and all `_data_root()`-derived.
- `tests/test_factor_kunquant.py` — **modified.** Grew from 2 tests to 9. `_factor_config()` gained a `dataset_cls` parameter (defaulting to `SpotKlineDataset`, so 03-01's callers are untouched) instead of being duplicated.

## Decisions Made

1. **The proxy lives at the Dataset boundary, not in the factor classes.** `StockDataset._to_kunquant()` is where vendor-shaped data becomes engine-shaped arrays — the one place every KunQuant consumer of US-equity data passes through. Per-class fixes would have needed one edit per factor class forever, and each would have been a place to get the adjusted-vs-raw distinction wrong.

2. **Double-guarded, so it is inert rather than authoritative.** The synthesis fires only when `amount` is requested AND absent. An unguarded assignment would overwrite a genuine vendor `amount` column the day one appears — a silent data-integrity regression (T-03-03-01) that no factor test would catch, since a plausible-looking number would just flow downstream. `test_stock_to_kunquant_without_amount_leaves_arrays_unchanged` asserts the inert path returns arrays identical to the active path's.

3. **Deliberate near-duplication in `Alpha158Stock`, recorded rather than silently tidied.** See the NORM-01 section above. The `vars()` method-set equality assertion is the mechanical guard that the two classes stay structurally parallel.

4. **Compiled stock tests use 8 symbols, not the fixture's 2.** KunQuant's compiled TS layout requires the symbol axis to be a multiple of its SIMD block width; 2 symbols raise `RuntimeError: Bad shape at open` inside `kr.runGraph`. The two `to_kunquant()` tests never reach KunQuant and keep the fixture default, which incidentally proves the proxy works at any panel width.

5. **One equality assertion for the whole matrix, not four independent asserts.** The matrix is a single decision. A failure should show the reader the entire before/after matrix, not one flipped boolean out of context — the mismatch message and the expected/actual dicts are the audit trail.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `stock_zarr`'s 2-symbol default cannot compile a KunQuant graph**

- **Found during:** Task 1 (Step D, `test_alpha101_stock_bugfix_batch_cal_returns_xarray_dataset`)
- **Issue:** The plan's verified-mechanics block confirmed the `amount` fix at `AllData` construction time but did not exercise `kr.runGraph`. With the `stock_zarr` fixture's default 2 symbols, the graph compiled and then died at execution with `RuntimeError: Bad shape at open` — KunQuant's compiled TS layout requires the symbol axis to be a multiple of its SIMD block width (8). The crypto tests never hit this because `spot_kline_zarr` defaults to 8 symbols.
- **Fix:** Added a module-level `_STOCK_SYMBOLS` constant (8 tickers) passed by every stock test that compiles a graph, with a comment block explaining the constraint. The two `to_kunquant()` tests deliberately keep the 2-symbol default — they never reach KunQuant, and running them at a different panel width is free extra coverage of the proxy.
- **Files modified:** `tests/test_factor_kunquant.py` (test only — no implementation change)
- **Verification:** `test_alpha101_stock_bugfix_batch_cal_returns_xarray_dataset` and `test_alpha158_stock_batch_cal_returns_xarray_dataset` both pass and both assert `sizes == {"timestamp": 60, "symbol": 8}`
- **Commits:** `75f41d6`, `2575f13`

---

**Total deviations:** 1 auto-fixed (1 × Rule 3 blocking). The deviation is confined to the test file; **zero deviations touched implementation code.**

**Impact on plan:** None on scope or outcome. Every `<success_criteria>` item is met as written, and every acceptance criterion passed. The finding is worth carrying forward: it is the same class of constraint as 03-01's known BUG-02 (`_make_stream()`'s hardcoded x86 SIMD block width, 03-05 Task 1) — KunQuant's block width leaks into what shapes of data the pipeline can process, and any future US-equity factor test must respect it.

## Verification Results

Plan-level `<verification>`, re-run after both implementation commits:

| Command | Expected | Actual |
|---|---|---|
| `uv run pytest tests/ -q` | zero failures | **59 passed** |
| `uv run pytest tests/test_factor_kunquant.py -q` | all dual-market tests green in one run | **9 passed** |
| `uv run pytest tests/test_factor_kunquant.py -k "amount or alpha101_stock" -q` | 3 passed | **3 passed** |
| `uv run pytest tests/test_factor_kunquant.py -k "alpha158_stock or normalization or config" -q` | 4 passed | **4 passed** |
| `uv run pytest tests/test_factor_kunquant.py -k normalization -q` | 1 passed | **1 passed** |
| `Alpha101Stock` vs `Alpha101SpotKline` `Input(` count + `amount` present | ok | **ok** |
| `vars()` method-set equality, `Alpha158Stock` vs `Alpha158SpotKline` | equal | **equal** |
| All four class docstrings contain `NORM-01` and `D-09` | ok | **ok** |
| Both `*Stock` docstrings contain `raw` | ok | **ok** |
| `stock_alpha101_config()` — `amount` in `data_columns`, path ends `alpha101_stock.zarr` | ok | **ok** |
| `stock_alpha158_config()` is a `FactorConfig`; `momentum_config(n=5)` is a `PolarsFactorConfig` with `window==5`, `kwargs=={'n':5}` | ok | **ok** |
| `grep -c "_data_root()" config/__init__.py` | greater than pre-task value | **10 → 11** |
| `grep -c "from factor.momentum" config/__init__.py` | 0 | **0** |
| `git diff --diff-filter=D` on both task commits | no deletions | **none** |

**Mutation check** (the NORM-01 lock, following 03-02's precedent that an introspection test which cannot fail is worse than no test):

| Mutation | Test | Result |
|---|---|---|
| Wrap `Alpha158Stock`'s `Output(v, k)` in `WindowedZScore(v, self.config.window)` | `test_normalization_matrix_matches_recorded_strategy_types` | **FAILED as required** |

The suite grew from 52 to 59 tests (7 added, all in `tests/test_factor_kunquant.py`).

## Known Stubs

None introduced by this plan.

`StockDataset._get_instrument` / `_xr_to_bars` / `_to_nautilus` still `raise ValueError("Not finished")` — pre-existing, untouched, and out of scope for a KunQuant factor plan (the Nautilus path is not a Phase-3 concern).

Two scaffold items already recorded in 03-01/03-02 remain open and assigned:

| Item | Status | Resolved by |
|---|---|---|
| `tests/test_factor_polars.py` — `FactorPolars` ABC, `Momentum` factor, D-04 laziness proof | Scaffold; `momentum_config()` shipped here but the class it configures has not | **03-04** |
| `tests/test_factor_stream.py` — FACTOR-02 `cal_stream()` replay, BUG-02 aarch64 SIMD block width | Scaffold | **03-05** |

## Threat Flags

None. No file changed by this plan introduces a network endpoint, auth path, file-access pattern or schema change at a trust boundary. The two boundaries in the plan's threat register were the ones actually worked on, and both mitigations landed: `T-03-03-01` (unguarded `amount` overwrite) is neutralised by the double guard plus its inert-path test, and `T-03-03-02` (undocumented normalization semantics) by the four docstrings, the matrix lock and the section above. `T-03-03-03` (leaked absolute paths in config factories) is covered by the `_data_root()` derivation and the still-green `tests/test_config_paths.py`.

## Issues Encountered

**A mutation-check revert overshot and had to be restored.** While mutation-checking the NORM-01 lock, `git checkout -- factor/alpha158.py` was used to undo the deliberate break. Because Task 2's work was not yet committed, that reverted the file to `HEAD` and discarded `Alpha158Stock` and the `Alpha158SpotKline` docstring along with the mutation. A pre-mutation copy had been taken first, so the file was restored from it and re-verified (211 lines, `Alpha158Stock` present, exactly one raw `Output(v, k)` and one `WindowedZScore`-wrapped one) before the suite was re-run green.

No work was lost, but the lesson is worth recording for future mutation checks: **revert a mutation on an uncommitted file by restoring the explicit pre-mutation copy, never with `git checkout --`**, which resets to `HEAD` and takes all uncommitted work with it. 03-02 ran its mutation checks against `base/factor.py` at a point where its changes were already committed, which is why `git checkout --` was safe there and is not safe in general.

## User Setup Required

None — zero new packages installed, no external service configuration, no environment variable added. 03-RESEARCH.md records the Package Legitimacy Audit as "Not applicable" for this phase, and `T-03-03-SC` is `accept` on that basis.

## Next Phase Readiness

**Ready for wave 4 (`03-05`), and non-blocking for the parallel `03-04`.**

- **03-04** (`FactorPolars`, `factor/momentum.py`) can consume `momentum_config()` as-is: it returns a fully-formed `PolarsFactorConfig` with `window == n` and `kwargs == {"n": n}`, and imports nothing from `factor/momentum.py`, so the two wave-3 plans never touched each other at runtime. Remember `config/__init__.py` was owned by this plan for the whole wave.
- **03-05** (streaming) inherits an unchanged `FactorKunQuant` streaming path. Note for its Task 1: this plan hit the same underlying constraint as BUG-02 from the other side — KunQuant's SIMD block width — at graph *execution* rather than compilation. `_STOCK_SYMBOLS` in `tests/test_factor_kunquant.py` documents it for the batch path.
- **Phase 4** (return model) can now feed US-equity factor panels into `base/model.py`, but should be aware these arrive **raw**: the consumer owns cross-sectional normalization for US equities, per the matrix above.
- **Phase 6** (ARCH-01/ARCH-02) inherits the explicit deferral of the cross-sectional Z op, with the building blocks named (`CrossSectionalOp`, `SimpleCrossSectionalOp`, `Rank`, `Scale`) and the extension point identified (an op wrapped around each `Output(...)` inside a factor class's own `_get_factor_func()`).

**Requirement status:** FACTOR-01 is now satisfied across **both** markets and is marked **Complete** in REQUIREMENTS.md. It is declared by exactly three plans — 03-01, 03-02 and 03-03 — and this plan is the last of them to produce a SUMMARY, so `requirements ready-ids` reported `1/1 ready` and the shared-ID gate released it. (03-02's SUMMARY anticipated that 03-05 also declared FACTOR-01; it does not — `grep -l FACTOR-01 *-PLAN.md` returns only those three.)

**Concerns:** none blocking. The Phase-4 checkpoint-reload flag raised by 03-02 (`utils/module.py:18` hardcodes `FactorConfig(**config)`, which cannot rebuild a `PolarsFactorConfig`-backed factor) is unaffected by this plan and still stands.

---
*Phase: 03-factor-computation-kunquant-polars*
*Completed: 2026-09-05*

## Self-Check: PASSED

All 5 modified files verified present on disk (`dataset/stock.py`, `factor/alpha101.py`, `factor/alpha158.py`, `config/__init__.py`, `tests/test_factor_kunquant.py`); both task commits (`75f41d6`, `2575f13`) verified present in `git log --all`; full suite re-run green (59 passed); all three tasks' acceptance criteria re-run and passing.
