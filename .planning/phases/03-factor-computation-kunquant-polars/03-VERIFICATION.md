---
phase: 03-factor-computation-kunquant-polars
verified: 2026-09-05T16:10:00Z
status: gaps_found
score: 3/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "User can compute at least one new factor via the Polars batch backend and get output conforming to the same `xarray.Dataset` contract (ROADMAP SC3 / FACTOR-03 / D-03 interchangeability)"
    status: partial
    reason: >-
      The `cal()` path is fully verified — `Momentum.cal().get_features()` returns an
      `xr.Dataset`. But the `read()` half of the shared contract is backend-dependent,
      which is exactly what D-03 exists to eliminate. Verified empirically by the
      verifier (not taken from 03-REVIEW.md): after `Momentum(...).cal().save()`, a
      FRESH `Momentum` instance driven through `read()` returns data but leaves
      `config.factor_names is None`, so `_get_factor_names()` raises
      `RuntimeError: Momentum: factor names are not resolved yet.` The identical
      sequence on `Alpha158SpotKline` succeeds and returns `('KMID',)`. Since
      `DLConfig.factor_data_strategy` is a first-class, config-selectable
      `Literal["read", "cal"]`, a `DLConfig` with `factor_data_strategy="read"` works
      for every KunQuant factor and raises for every Polars factor at
      `base/model.py:187`. This independently CONFIRMS 03-REVIEW.md CR-01.
    artifacts:
      - path: "base/factor_polars.py"
        issue: >-
          `_maybe_resolve_factor_names()` (L55-61) is a deliberate no-op and `cal()`
          (L110) is the ONLY assignment to `config.factor_names`. The class docstring
          (L45-46) and the `RuntimeError` message (L65-70) both assert the precondition
          is satisfied by "`cal()` **or `read()`**" — the `read()` half of that claim is
          false in code.
      - path: "base/factor.py"
        issue: >-
          `Factor.read()` (L126-129) calls only `data_backend.read()` + `_auto_filter()`;
          it never resolves `config.factor_names`. There is no `FactorPolars.read()`
          override.
      - path: "tests/test_factor_hierarchy.py"
        issue: >-
          `test_kunquant_and_polars_factors_are_interchangeable_in_one_dlconfig`
          (L430-522) — the phase's live D-03 proof — pins `factor_data_strategy="cal"`
          and hand-calls `factor.cal()`. The `read()` branch of `base/model.py`'s call
          surface is never driven for EITHER backend, so the asymmetry is invisible to
          a green suite.
      - path: "tests/test_factor_polars.py"
        issue: >-
          `test_factor_names_resolve_dynamically_from_the_lazyframe_schema` (L95-117)
          restates the false precondition in its own docstring ("only valid after
          `cal()`/`read()` … `base/model.py` always calls `.cal()`/`.read()` before
          asking a factor for its names") while asserting only the `cal()` path. The
          test encodes the defect rather than catching it.
      - path: "README.md"
        issue: >-
          L34-40 claims "nothing downstream can tell which one produced a given factor
          store" and lists `read()` among the shared-contract methods the model layer
          calls. Empirically false — shipped documentation contradicted by the code.
    missing:
      - "A `FactorPolars.read()` override that resolves `config.factor_names` from the persisted store's non-index data_vars, symmetrically with `cal()`."
      - "A regression test driving BOTH backends through save() -> fresh instance -> read() -> _get_factor_names(), with factor_data_strategy='read'."
      - "Correction of the false `read()` precondition in base/factor_polars.py's docstring, its RuntimeError message, tests/test_factor_polars.py:103-105 and README.md:34-40."
    disposition: accept-and-fix
    disposition_date: "2026-09-06"
    disposition_by: user
    disposition_note: >-
      Gap stands and will be closed, but NOT via plan 03-06. Design settled with the user
      on 2026-09-06 after three routes were measured. Chosen route: make factor names a
      DERIVED PROPERTY OF THE COMPUTATION GRAPH rather than stored state. (1) Add a
      bounded-read method to the `DataBackend` interface (e.g. `head(n)` /
      `get_lazyframe(limit=n)`), implemented by both `XrBackend` and `PlBackend` -- going
      through the interface rather than reaching past it, since backend interchangeability
      (D-03) is what this gap is about. (2) `FactorPolars._get_factor_names()` reads a few
      rows through that method, runs `_get_factor_lazyframe()` on them, and returns
      `collect_schema().names()` minus `_INDEX_COLUMNS`. (3) DELETE the
      `_maybe_resolve_factor_names()` no-op override so the base-class behaviour at
      base/factor.py:92-93 is restored -- names resolve at config-assignment time, which
      fixes read(), cal() and a bare-constructed instance at once, and preserves the
      "explicit pin wins, else derive" semantics for free. The fix is a DELETED override,
      not an added one. `read()` keeps its existing meaning: instantiate the specified
      date range of historical data; it plays no part in naming. Implementation is the
      user's, done by hand outside GSD. Plan 03-06 is superseded.
  - truth: "`Alpha158Stock`/`Alpha101Stock` compute the Alpha158/101 factor set correctly for US equities (phase must-have from 03-03; the D-01 extension of ROADMAP SC1)"
    status: partial
    reason: >-
      `.cal()` returns a correctly shaped `xr.Dataset`, so the must-have as literally
      worded passes — but the values are silently corrupted. Verified empirically by
      the verifier: `dataset/stock.py:73-74` synthesizes `amount = volume * close`,
      and KunQuant's `AllData` derives `vwap = amount / volume`, so `vwap` is
      IDENTICALLY `close`. Probing `Alpha158Stock` with `factor_names=["VWAP0","VWAP1","CLOSE1"]`
      over the `stock_zarr` fixture produced `VWAP0` unique finite values `[1.0]`
      (std 5.2e-08 — a zero-variance feature) and `VWAP1` allclose-identical to
      `CLOSE1`. This independently CONFIRMS 03-REVIEW.md CR-02. Five of Alpha158's
      emitted price-block features carry zero incremental information for US equities,
      and every `Alpha101Stock` alpha referencing `vwap` (alpha025, alpha028, alpha041,
      alpha050, alpha083, …) degenerates into a close-price variant. Nothing fails
      loudly — this is worse than the graph-construction crash it replaced, because the
      crash was visible.
    artifacts:
      - path: "dataset/stock.py"
        issue: "L73-74: `data.assign(amount=data['volume'] * data['close'])` makes the derived vwap algebraically equal to close."
      - path: "tests/test_factor_kunquant.py"
        issue: >-
          `test_alpha158_stock_batch_cal_returns_xarray_dataset` (L222-247) asserts only
          on KMID/VOLUME0/STD5 and `test_stock_to_kunquant_synthesizes_amount_as_adjusted_dollar_volume`
          (L144) locks the defective formula in as expected behaviour. No test in the
          suite touches a VWAP-derived feature, which is why 66/66 green says nothing here.
    missing:
      - "A dollar-volume proxy whose implied vwap is not the close price (e.g. typical price `(high+low+close)/3 * volume`)."
      - "A test asserting `input_dict['amount'] / input_dict['volume']` is NOT allclose to `input_dict['close']`."
      - "A test asserting `Alpha158Stock`'s `VWAP0` output is not constant."
    disposition: dismissed
    disposition_date: "2026-09-06"
    disposition_by: user
    disposition_note: >-
      Dismissed by user decision on 2026-09-06 ("第二个可以忽略"). No technical
      rationale was given and none is inferred here. The finding itself is NOT
      retracted -- it was confirmed empirically by the verifier and independently by
      03-REVIEW.md CR-02 -- so the consequences below remain true and accepted:
      `dataset/stock.py:73-74` synthesizes `amount = volume * close`, KunQuant derives
      `vwap = amount / volume`, and `vwap` is therefore identically `close` for US
      equities. Five Alpha158 price-block features carry zero incremental information,
      and every Alpha101Stock alpha referencing vwap (alpha025, alpha028, alpha041,
      alpha050, alpha083, ...) degenerates into a close-price variant. Nothing fails
      loudly. Plan 03-07 is superseded. Re-open this gap if VWAP-derived features are
      ever used in a model or a backtest.
human_verification:
  - test: "Decide whether the D-02 `amount` proxy should be the typical-price dollar volume, or whether a real vendor dollar-volume column should be sourced from Tiingo instead."
    expected: "A US-equity `amount` series whose implied `vwap = amount/volume` is not identically `close`, restoring the informational content of the VWAP price block."
    why_human: "Choosing between a synthesized proxy and a vendor column is a data-sourcing decision with cost/coverage tradeoffs, not a programmatic determination."
    disposition: dismissed
    disposition_date: "2026-09-06"
    disposition_by: user
    disposition_note: "Answered by the gap-2 dismissal: neither option is taken; the existing `volume * close` proxy stands."
---

# Phase 3: Factor Computation (KunQuant + Polars) Verification Report

**Phase Goal:** Users can compute the existing Alpha158 factor set (batch and streaming) via KunQuant, and can add new factors via a new Polars batch backend, with `xarray.Dataset` as the sole exchange format.
**Verified:** 2026-09-05T16:10:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can compute the Alpha158 factor set in batch mode from stored market data and get back an `xarray.Dataset` | ✓ VERIFIED | `tests/test_factor_kunquant.py::test_alpha158_spot_batch_cal_returns_xarray_dataset` run individually — PASS. Asserts `isinstance(result, xr.Dataset)`, `sizes == {"timestamp": 60, "symbol": 8}`, `data_vars == ["KMID","STD5","VOLUME0"]`, and `np.isfinite(KMID).sum() > 0`. Real compiled-graph execution against a synthetic Zarr store, not a mock. The enabling fix is real: `my_ops/preprocess.py:12,36` now carry `def decompose(self, options: dict)`, matching installed KunQuant 0.1.11's `CompositiveOp.decompose` contract. **Caveat:** verified for crypto spot. The D-01 US-equity extension (`Alpha158Stock`) returns a correctly shaped Dataset but with corrupted VWAP values — see gap 2. |
| 2 | User can invoke KunQuant's streaming (`cal_stream`) factor computation path and get incremental factor updates without error | ✓ VERIFIED | `tests/test_factor_stream.py::test_cal_stream_replay_produces_incremental_factor_updates` run individually — PASS. A genuine behavioural test, not a presence check: replays all 60 timestamps one bar at a time through `factor.cal_stream(bar, step, symbol_list)`, asserts the result is an `xr.Dataset` of shape `{"timestamp": 1, "symbol": 8}`, and — critically — asserts `not np.array_equal(previous_kmid, last_kmid)`, so a stale or constant buffer fails. Per-bar inputs come from the real `Dataset.to_kunquant()` adapter. `init_stream()` is separately proven to bind a buffer handle for all six declared names. The enabling fix (`0a61d8e`, removal of the hardcoded x86-only SIMD block width) is what makes this runnable on this aarch64 machine at all. |
| 3 | User can compute at least one new factor via the Polars batch backend and get output conforming to the same `xarray.Dataset` contract | ✗ FAILED (partial) | `cal()` half VERIFIED: `tests/test_factor_polars.py::test_momentum_cal_returns_xarray_dataset_with_only_factor_columns` — PASS; `data_vars == ["momentum_5"]` exactly (no raw-column leakage), `sizes == {timestamp, symbol}`, finite values > 0. `read()` half FAILED: verifier probe proved a fresh `Momentum` after `read()` raises `RuntimeError` from `_get_factor_names()` while `Alpha158SpotKline` returns `('KMID',)` on the identical sequence. "the same contract" does not hold across both halves of the model layer's call surface. See gap 1. |
| 4 | No factor-pipeline code path passes a plain DataFrame between modules — inputs/outputs are `xarray.Dataset` only | ✓ VERIFIED (coincidental-reliance) | Grep over `base/ factor/ label/ dataset/ my_ops/` finds no `pd.DataFrame`/`pl.DataFrame` in any signature outside comments. The only frame-typed methods in the factor layer are the private hooks `Factor._get_lazyframe` and `FactorPolars._get_factor_lazyframe`. Both backends convert to `xr.Dataset` before anything leaves the class: `FactorKunQuant` from raw arrays, `FactorPolars.cal()` at L110-121 via `collect().to_pandas().set_index().Dataset.from_dataframe()`. `get_features`/`get_labels` return-annotated `xr.Dataset` on both classes, asserted by `test_public_factor_api_exchanges_only_xarray_datasets`. Flagged advisory — see below. |

**Score:** 3/4 truths verified (0 present, behavior-unverified)

#### Note on truth 4's advisory flag

`test_public_factor_api_exchanges_only_xarray_datasets` is a **weak lock** on a property that does happen to hold. It asserts `"DataFrame" not in str(inspect.signature(...))` — a substring match that would not catch `pl.LazyFrame`, `pl.Series`, or an aliased import — and it iterates only `(Factor, FactorKunQuant)`, never `FactorPolars`, the one backend that actually handles frames. The property holds today because `FactorPolars.cal()` returns `Self` and its frame-handling hook is private, not because the test enforces it. Additionally `FactorPolars.cal()` receives a `pl.LazyFrame` across the Dataset→Factor module boundary via `self.config.dataset.read().get_lazyframe()`; `03-CONTEXT.md` D-06 explicitly sanctions this ("Polars is used internally for the computation, but the module boundary contract … is unchanged"), and it mirrors the pre-existing `to_kunquant()` numpy-dict adapter, so it is a decision, not a violation. Recorded as advisory only — it does not change the status or the score.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `my_ops/preprocess.py` | `decompose(self, options: dict)` matching KunQuant 0.1.11 | ✓ VERIFIED | L12 and L36 both carry the corrected signature; both alpha families compile and run |
| `tests/conftest.py` | `spot_kline_zarr` / `stock_zarr` synthetic-Zarr factories | ✓ VERIFIED | L346, L486; consumed by every Phase-3 factor test with zero network access |
| `base/factor.py` | `Factor(ABC)` + `FactorKunQuant(Factor)` | ✓ VERIFIED | `class Factor(ABC)` L21, `class FactorKunQuant(Factor)` L169; `__init__` assigns `self.config` before `self.data_backend` (L36-40, Hazard 1 preserved); `mode` reads confined to `FactorKunQuant` overrides |
| `base/config.py` | `BaseFactorConfig` / `FactorConfig` / `PolarsFactorConfig` split + type widening | ✓ VERIFIED | L67, L91, L107; `DLConfig.factors`/`labels` and `MLConfig.factors` typed `list["Factor"]` (L120-121, L155) |
| `base/factor_polars.py` | `FactorPolars(Factor)` batch-only backend | ⚠️ HOLLOW | Exists, substantive, wired, and correct on `cal()` — but the `read()` path never resolves `factor_names`, breaking the contract it documents (gap 1) |
| `factor/momentum.py` | `Momentum(FactorPolars)` worked example | ✓ VERIFIED | Genuine Polars window expressions (`close.shift(n).over("symbol")`), config-driven horizon via `config.kwargs["n"]`, terminal `.select([timestamp, symbol, factor])` |
| `factor/alpha158.py` | `Alpha158Stock` US-equity sibling | ⚠️ HOLLOW | `class Alpha158Stock(FactorKunQuant)` L107 exists and computes — but its VWAP price block is degenerate (gap 2) |
| `factor/alpha101.py` | `Alpha101Stock` with `amount` wired into `AllData` | ⚠️ HOLLOW | Construction crash genuinely fixed; every vwap-referencing alpha silently degenerates (gap 2) |
| `dataset/stock.py` | `_to_kunquant()` with D-02 dollar-volume proxy | ✗ DEFECTIVE | L73-74 present and double-guarded as designed, but the chosen formula collapses vwap onto close (gap 2) |
| `config/__init__.py` | `stock_alpha101_config` / `stock_alpha158_config` / `momentum_config` | ✓ VERIFIED | L154, L225, L271; all derive from `_data_root()`, no absolute paths |
| `tests/test_extensibility_contract.py` | `CORE_LAYER_FILES` covers `base/factor_polars.py` | ✓ VERIFIED | L33; passes the core-layer purity check |
| `README.md` | Documents both backends + how to add a factor to each | ⚠️ PARTIAL | Sections present and useful, but L34-40's interchangeability claim is contradicted by the code (gap 1) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `my_ops/preprocess.py:decompose` | KunQuant `Decompose.decompose_impl` | one-positional-arg call | ✓ WIRED | Both alpha families compile and execute end to end |
| `base/factor.py:FactorKunQuant` | `base/factor.py:Factor` | class inheritance | ✓ WIRED | `class FactorKunQuant(Factor)` |
| `base/factor_polars.py:FactorPolars` | `base/factor.py:Factor` | class inheritance | ✓ WIRED | `class FactorPolars(Factor)` |
| `base/config.py:DLConfig.factors` | `base/factor.py:Factor` | `TYPE_CHECKING` forward ref | ✓ WIRED | `list["Factor"]` on both DLConfig and MLConfig |
| `base/factor_polars.py:cal` | `base/data.py:Dataset.get_lazyframe` | `self.config.dataset.read().get_lazyframe()` | ✓ WIRED | L104 |
| `factor/alpha101.py:Alpha101Stock` | `dataset/stock.py:_to_kunquant` | `data_columns` includes `amount` | ⚠️ WIRED, WRONG VALUE | Link exists and carries data; the synthesized value is algebraically degenerate |
| `base/model.py` (`strategy="cal"`) | both backends | `factor.cal().get_features()` | ✓ WIRED | Live two-backend test merges both outputs into one `xr.Dataset` |
| `base/model.py` (`strategy="read"`) | `FactorPolars` | `factor.read()` then `_get_factor_names()` | ✗ NOT_WIRED | Raises `RuntimeError` for Polars, succeeds for KunQuant — verified empirically |
| `base/factor.py:_get_lazyframe` | (any caller) | — | ⚠️ ORPHANED | Zero call sites repo-wide; dead code carried through the refactor |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `Alpha158SpotKline` | `KMID`, `VOLUME0`, `STD5` | `kr.runGraph` over `to_kunquant()` arrays | Yes — finite, varying | ✓ FLOWING |
| `FactorKunQuant` (stream) | `KMID` per bar | `cal_stream` StreamContext buffers | Yes — values change between consecutive bars | ✓ FLOWING |
| `Momentum` | `momentum_5` | `collect()` of a real Polars window chain | Yes — finite values > 0 | ✓ FLOWING |
| `Alpha158Stock` | `KMID`, `STD5`, `VOLUME0` | `kr.runGraph` over adjusted stock arrays | Yes | ✓ FLOWING |
| `Alpha158Stock` | `VWAP0` | `vwap = amount/volume` where `amount = volume*close` | **No — constant 1.0, std 5.2e-08** | ✗ DEGENERATE |
| `Alpha158Stock` | `VWAP1..VWAP4` | same | **No — bit-identical to `CLOSE1..CLOSE4`** | ✗ DEGENERATE |
| `Momentum` after `read()` | `config.factor_names` | nothing populates it | **No — stays `None`** | ✗ DISCONNECTED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full suite baseline (run once) | `uv run pytest tests/ -q` | `66 passed, 21 warnings in 8.73s` | ✓ PASS |
| SC1 batch Alpha158 | `pytest tests/test_factor_kunquant.py::test_alpha158_spot_batch_cal_returns_xarray_dataset` | pass | ✓ PASS |
| SC2 streaming replay | `pytest tests/test_factor_stream.py::test_cal_stream_replay_produces_incremental_factor_updates` | pass | ✓ PASS |
| SC3 Polars `cal()` | `pytest tests/test_factor_polars.py::test_momentum_cal_returns_xarray_dataset_with_only_factor_columns` | pass | ✓ PASS |
| SC4 boundary contract | `pytest tests/test_factor_hierarchy.py::test_public_factor_api_exchanges_only_xarray_datasets` | pass | ✓ PASS |
| D-03 live interchangeability (`cal` path only) | `pytest tests/test_factor_hierarchy.py::test_kunquant_and_polars_factors_are_interchangeable_in_one_dlconfig` | pass | ✓ PASS |
| **Verifier probe:** Polars `read()` -> `_get_factor_names()` | temporary probe, `save()` -> fresh instance -> `read()` | `RuntimeError: Momentum: factor names are not resolved yet.` (KunQuant sibling returned `('KMID',)`) | ✗ FAIL |
| **Verifier probe:** `Alpha158Stock` VWAP degeneracy | temporary probe, `factor_names=["VWAP0","VWAP1","CLOSE1"]` | `VWAP0 unique finite = [1.]`, `std = 5.1976368e-08`, `VWAP1 allclose CLOSE1 = True` | ✗ FAIL |

Both probe files were deleted after execution; `git status --porcelain` confirms no source or test file was modified by verification.

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| — | — | No `scripts/*/tests/probe-*.sh` exist and no PLAN/SUMMARY declares one | ? SKIP (no project probes) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FACTOR-01 | 03-01, 03-02, 03-03 | KunQuant 后端支持批量计算 Alpha158 因子集，输出 `xarray.Dataset` | ✓ SATISFIED (crypto) / ⚠️ PARTIAL (US equities) | Crypto batch path fully proven. `Alpha158Stock`/`Alpha101Stock` return correct-shaped Datasets but with degenerate VWAP-derived features |
| FACTOR-02 | 03-05 | KunQuant 后端保留流式（`cal_stream`）计算能力 | ✓ SATISFIED | First-ever execution of `init_stream()`/`cal_stream()` in this repo; 60-bar replay with a genuine incrementality assertion |
| FACTOR-03 | 03-04, 03-05 | 新增 Polars 批量因子计算后端接口 | ⚠️ PARTIAL | Backend, contract and worked example all real and proven on `cal()`; the `read()` path is not contract-conformant |
| FACTOR-04 | 03-02, 03-04, 03-05 | 因子计算模块间数据传输统一使用 `xarray.Dataset` | ✓ SATISFIED | No public factor method exchanges a DataFrame; both backends emit `xr.Dataset`. Lock is weaker than it appears (see truth 4 note) |

No orphaned requirements: `REQUIREMENTS.md` maps exactly FACTOR-01..04 to Phase 3, and all four are claimed by at least one plan's `requirements` frontmatter.

**Note:** `REQUIREMENTS.md` L28-31 and L106-109 already mark all four as `[x]` / `Complete`. Given the gaps above, FACTOR-01 and FACTOR-03 are marked complete prematurely.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | `TBD` / `FIXME` / `XXX` / `TODO` / `HACK` / `PLACEHOLDER` | — | **None found** across all 17 phase-modified files. Debt-marker gate passes. |
| `base/factor_polars.py` | 45-46, 65-70 | Docstring/error message asserts a precondition the code does not satisfy | 🛑 Blocker | Misleads every future reader into believing `read()` resolves names |
| `base/factor.py` | 141-144 | `_get_lazyframe()` — zero call sites repo-wide | ⚠️ Warning | Dead code carried through the refactor; `xr.Dataset.to_pandas()` on a 2-D-per-variable Dataset is also of doubtful correctness |
| `base/factor.py` | 111-112, 161-162 | `get_factor_names()` returns `self.config.factor_names` directly, bypassing `_get_factor_names()` | ⚠️ Warning | The `FactorPolars` `RuntimeError` guard is unreachable from the public surface: `num_factors` on an unresolved Polars factor gives `TypeError: object of type 'NoneType' has no len()`, and `get_factor_names()` silently returns `None`. Confirms 03-REVIEW.md WR-01 |
| `tests/test_factor_kunquant.py` | 144-167 | `test_stock_to_kunquant_synthesizes_amount_as_adjusted_dollar_volume` | ⚠️ Warning | Locks the defective `volume * close` formula in as expected behaviour, so fixing gap 2 requires editing this test |
| `factor/momentum.py` / `config/__init__.py` | 9 / 276 | `_DEFAULT_HORIZON = 20` and `n: int = 20` are two independent defaults | ℹ️ Info | Divergence would be silent |
| `tests/test_factor_hierarchy.py` | 389-414 | `"DataFrame" not in signature` substring match, `FactorPolars` not in the checked class tuple | ⚠️ Warning | The FACTOR-04 lock is weaker than the property it guards |

### Human Verification Required

#### 1. Choose the US-equity `amount` data source

**Test:** Decide whether `dataset/stock.py`'s D-02 proxy should become the typical-price dollar volume (`(high+low+close)/3 * volume`), or whether a real vendor dollar-volume column should be sourced from Tiingo instead.
**Expected:** A US-equity `amount` series whose implied `vwap = amount / volume` is not identically `close`, restoring the informational content of the five VWAP price-block features and every vwap-referencing Alpha101 formula.
**Why human:** Choosing between a synthesized proxy and a vendor column is a data-sourcing decision with cost, coverage and adjustment-consistency tradeoffs — not something the verifier can settle programmatically. The *presence* of the defect is already proven.

### Gaps Summary

The phase's headline deliverables are real, and three of the four ROADMAP success criteria hold under genuine behavioural tests. The `WindowedZScore.decompose` signature fix genuinely unblocks batch Alpha158/Alpha101; `cal_stream()` genuinely runs for the first time in this repository, on this machine, with a real incrementality assertion rather than a smoke test; `FactorPolars` is a real sibling backend with a real lazy-expression worked example; and `xr.Dataset` really is the only type crossing the factor layer's public boundary. The `Factor`/`FactorKunQuant`/`FactorPolars` hierarchy, the `BaseFactorConfig` split, and the `mode`-isolation and `__init__`-ordering hazards are all correctly executed.

Two defects block the goal, and I confirmed both independently against the code rather than accepting 03-REVIEW.md's claims.

**CR-01 is confirmed, and it is a real goal-level gap, not a documentation nit.** D-03's entire purpose — the user's own words, "我需要这两个因子类可以无缝替换" — is that `base/model.py` cannot tell the two backends apart. It can. `factor_data_strategy` is a first-class `Literal["read", "cal"]` config field, and selecting `"read"` makes every KunQuant factor work and every Polars factor raise. The phase's own live interchangeability test cannot see this because it pins `"cal"` and hand-calls `cal()`; `tests/test_factor_polars.py` cannot see it because it restates the false precondition in prose while asserting only the `cal()` path; and `README.md` ships the claim that this is impossible. The fix is small — resolve `factor_names` from the persisted store's data_vars on `read()` — but the missing *test* matters more than the missing line, because the whole D-03 contract currently rests on a test that exercises half of it.

**CR-02 is confirmed, and it is silent corruption rather than an approximation.** `amount = volume * close` makes `vwap = amount/volume` exactly `close`, which I measured: `VWAP0` is a constant 1.0 with std 5.2e-08, and `VWAP1` is allclose-identical to `CLOSE1`. Five Alpha158 features become zero-information for US equities and every vwap-referencing Alpha101 alpha degenerates into a close-price variant. The arrays are finite and the shapes are right, so nothing fails — and `test_stock_to_kunquant_synthesizes_amount_as_adjusted_dollar_volume` actively locks the defective formula in as intended behaviour. This is strictly worse than the graph-construction crash it replaced, because the crash was visible. Phase 4 consumes exactly these features to train a model.

**On the green suite.** 66/66 passing is necessary but not sufficient here, and this phase is a clean illustration of why: the suite is genuinely strong on the paths it covers (the streaming incrementality assertion and the D-04 monkeypatched-`collect` laziness proof are both better than typical), but neither defect is reachable from any assertion in it. Both were found by driving code paths the tests deliberately do not drive.

Neither gap is deferrable. No later phase's goal or success criteria specifically address the Polars `read()` path or the US-equity dollar-volume proxy; Phase 6 SC4 (stage swapping) and Phase 7 (test coverage) are too general to carry them, and per the conservative matching rule both stay as actionable gaps.

---

_Verified: 2026-09-05T16:10:00Z_
_Verifier: Claude (gsd-verifier)_

---

## Gap Dispositions — 2026-09-06

Both gaps were reviewed with the user on 2026-09-06. Neither finding is retracted; what
changed is what will be done about each. The two pre-existing gap-closure plans, `03-06`
and `03-07`, were written against these gaps but never executed (no SUMMARY, no commits
reference them) and are **both superseded** by the decisions below.

| Gap | Finding stands? | Disposition | Closing route |
|-----|-----------------|-------------|---------------|
| 1 — `FactorPolars` not interchangeable on the `read()` path | Yes | accept-and-fix | Manual fix by the user; plan 03-06 superseded |
| 2 — synthesized `amount` makes `vwap` identically `close` | Yes | **dismissed** | None; plan 03-07 superseded |

### Gap 1 — accept; close it at the root, not on the `read()` path

The gap is real and stays open until the fix lands. Plan `03-06` proposed a broader change
and is superseded. Three routes were measured with the user on 2026-09-06 before settling.

**The insight that reframed it.** `base/factor.py:92-93` already implements the right
pattern for every other factor class:

```python
def _maybe_resolve_factor_names(self):
    if self._config.factor_names is None:
        self._config.factor_names = self._get_factor_names()
```

`config.factor_names` is the **explicit-pin** channel; `_get_factor_names()` is the
**derivation** channel. `FactorPolars` overrode `_maybe_resolve_factor_names()` to a no-op
for one reason only — deriving eagerly would have meant a disk read at construction. So the
gap is not "`read()` forgot to set a field"; it is "the derivation channel was disabled and
only `cal()` was left to fill the field in". Fixing the derivation makes the override
unnecessary. **The fix is a deleted override, not an added one.**

**Chosen design.**

1. Add a bounded-read method to the `DataBackend` interface — `head(n)` or
   `get_lazyframe(limit=n)` — implemented by both `XrBackend` and `PlBackend`.
2. `FactorPolars._get_factor_names()` reads a few rows through that method, runs
   `_get_factor_lazyframe()` on them, and returns `collect_schema().names()` minus
   `_INDEX_COLUMNS`.
3. Delete the `_maybe_resolve_factor_names()` no-op override, restoring base-class
   behaviour: names resolve at config-assignment, so `read()`, `cal()` and a
   bare-constructed instance are all correct at once, and "explicit pin wins" comes free.

Going through the `DataBackend` interface rather than reaching past it to
`xr.open_zarr(...).isel(...)` is deliberate: using one concrete backend's private path to
fix a *backend-dependence* gap would be self-contradicting. On `PlBackend` the same method
is `scan_parquet().head(n)` — genuinely lazy, so the cost below drops to nothing there.

**Measurements (2026-09-06).** All three routes returned the identical name.

| Route | Cost | Note |
|---|---|---|
| Full read through today's `get_lazyframe()` | 0.160 s | on the *smallest* store tested; materializes everything, so it grows with the store |
| **Bounded read (chosen)** | **~25–50 ms** | see scaling below |
| Zero-row schema stub | 0.009 s | cheapest, but requires hand-constructing the input schema |

Bounded-read cost does **not grow with store size** — it is dominated by fixed overhead
(zarr metadata open plus pandas conversion setup), not data volume:

```
  2520 x 500  ( 30 MB)  53.5 ms   baseline
 25200 x 500  (303 MB)  26.0 ms   10x history
  2520 x 5000 (303 MB)  30.2 ms   10x symbols
```

The zero-row stub was rejected despite being cheapest: it required hardcoding every input
dtype (`pl.Float64`), so a dtype-sensitive expression — integer division, a `.str.`
operation, a cast — could derive a different name or fail spuriously. Reading real rows
carries real dtypes for free.

**Accepted consequences.**

- **Construction touches disk.** Names now resolve at config-assignment, so constructing a
  `FactorPolars` reads a few rows. D-04's "computation starts at `cal()`" is not violated —
  a bounded metadata-plus-few-rows read is not the computation — but the no-op override
  originally existed to avoid exactly this touch, and that trade is being made knowingly.
- **`read()` is unchanged.** It keeps its existing meaning: instantiate the specified date
  range of historical data. It plays no part in naming.
- **A config/store mismatch now surfaces instead of being papered over.** Because names come
  from the graph, a store written with `n=5` under a config now saying `n=60` yields
  `momentum_60`, and the subsequent lookup fails with a name pointing straight at the cause.
  Deriving from disk instead would have returned `momentum_5` — matching the data while
  silently ignoring that the config asked for something else.
- **Known boundary.** A factor whose output column names depend on data *values* (e.g. a
  pivot over distinct symbols) would derive wrong names from a few rows. Such a factor
  already violates the stated `_get_factor_lazyframe` contract ("return only `timestamp`,
  `symbol` and the computed factor column(s)"), so it is out of contract rather than a
  regression — but it is the one shape this design cannot serve.

**Deletions this design authorises** (user decision, 2026-09-06: "不使用新方法的测试可以
删掉，过时的判断也可以删掉"). Each of these is not merely mis-worded but *obsolete* — the
condition it describes cannot arise once names derive from the graph.

| Location | What goes | Why it is obsolete, not just wrong |
|---|---|---|
| `base/factor_polars.py:55-61` | the `_maybe_resolve_factor_names()` no-op override | the reason it existed (eager derivation meant a disk read) is removed by the bounded read |
| `base/factor_polars.py:63-71` | the `if self.config.factor_names is None: raise RuntimeError(...)` guard inside `_get_factor_names()` | names are always derivable from the graph, so the unresolved state the guard reports can no longer occur; the method body becomes the derivation |
| `base/factor_polars.py:44-50` | the class-docstring paragraph stating the `cal()`/`read()` precondition | there is no precondition left to state |
| `tests/test_factor_polars.py:101-105, 110-111` | the docstring paragraph restating the precondition, and the `pytest.raises(RuntimeError, match="cal")` block | this assertion **inverts** under the new design — a bare-constructed factor now resolves its names, so the test would fail if kept |

The remainder of `test_factor_names_resolve_dynamically_from_the_lazyframe_schema`
(`cal()`, then `get_factor_names() == ("momentum_5",)` and `num_factors == 1`) stays valid
and in fact gets **stronger**: under the new design those two assertions should hold
*without* the preceding `cal()` call, which is the cleanest way to pin the fix.

`tests/test_factor_hierarchy.py`'s references to `_maybe_resolve_factor_names` (L161, L186,
L340, L364, L370) are **kept** — they exercise the base-class hook contract, which survives;
only `FactorPolars`'s override of it goes.

**Correction to an earlier entry in this record.** `README.md:34-40` was previously listed
here as prose needing correction. That was wrong. It claims "nothing downstream can tell
which one produced a given factor store" and lists `read()` among the shared-contract
methods the model layer calls — statements that are false against today's code and
**become true** once this design lands. It needs no edit; the code catches up to it. It is,
in effect, a specification that was written correctly and shipped ahead of its
implementation.

**Still open under this gap, and NOT closed by the names fix alone:**

- The regression test driving **both** backends through `save() → fresh instance →
  read() → _get_factor_names()` with `factor_data_strategy="read"`. This is the one item
  that is an *addition* rather than a deletion. Without it the asymmetry stays invisible to
  a green suite, which is how it survived in the first place
  (`tests/test_factor_hierarchy.py:430-522` pins `factor_data_strategy="cal"` and hand-calls
  `factor.cal()`, so the `read()` branch of `base/model.py`'s call surface is never driven
  for either backend).

### Gap 2 — dismissed by user decision

Dismissed on 2026-09-06 at the user's direction ("第二个可以忽略"). **No technical
rationale was offered, and none is invented here.** The finding is not withdrawn — it was
established empirically by the verifier and independently by `03-REVIEW.md` CR-02 — so the
following remains true and is accepted as a known condition of the codebase:

- `dataset/stock.py:73-74` computes `amount = volume * close`.
- KunQuant's `AllData` derives `vwap = amount / volume`, so for US equities
  **`vwap` is identically `close`**.
- Probing `Alpha158Stock` with `factor_names=["VWAP0","VWAP1","CLOSE1"]` produced `VWAP0`
  with unique finite values `[1.0]` (std 5.2e-08 — a zero-variance feature) and `VWAP1`
  allclose-identical to `CLOSE1`.
- Five of Alpha158's emitted price-block features therefore carry zero incremental
  information for US equities, and every `Alpha101Stock` alpha referencing `vwap`
  (alpha025, alpha028, alpha041, alpha050, alpha083, …) degenerates into a close-price
  variant.
- `tests/test_factor_kunquant.py:144`
  (`test_stock_to_kunquant_synthesizes_amount_as_adjusted_dollar_volume`) locks the current
  formula in as expected behaviour, so the suite will stay green over it.

**Nothing fails loudly.** That is the operative risk of this dismissal: a model or backtest
consuming VWAP-derived features will silently receive a duplicate of the close price rather
than an error.

**Re-open this gap** if VWAP-derived features are used in a model (phase 4) or a backtest
(phase 6). The accompanying `human_verification` item — choosing between a typical-price
proxy and a vendor dollar-volume column from Tiingo — is dismissed with it; neither option
is taken and the existing proxy stands.

### Phase status

Unchanged: `gaps_found`. Gap 1's fix has not landed yet, so the phase is not verified. Once
it lands, gap 1's remaining items above and this disposition record are what a re-run of
verification should be measured against — not plan `03-06`, which no longer describes the
work being done.
