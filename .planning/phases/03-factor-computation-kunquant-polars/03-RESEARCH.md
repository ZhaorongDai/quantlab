# Phase 3: Factor Computation (KunQuant + Polars) - Research

**Researched:** 2026-09-05
**Domain:** Refactoring an existing KunQuant-based factor-computation layer to extract a shared abstract base, adding a new Polars-lazy batch factor backend as a sibling, and extending Alpha158 to a second market (US equities) — all while xarray.Dataset remains the sole cross-module exchange format.
**Confidence:** HIGH — every architectural claim below is verified either by reading the actual source of `base/factor.py`/`factor/alpha101.py`/`factor/alpha158.py`/`label/spot.py`/`base/config.py`/`base/model.py`/`base/data.py`/`dataset/stock.py`/`dataset/spot.py`, or by executing real code against the installed `KunQuant==0.1.11`/`polars==1.44.1` in this repo's own `uv` environment. Two items are `[ASSUMED]` and flagged (KunQuant's STREAM `num_stock`-multiple-of-8 requirement's hard-vs-soft nature; the exact runtime cost of un-normalized column names in the Polars example factor).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Alpha158 market coverage**
- **D-01:** Extend Alpha158 to US equities by adding `Alpha158Stock` (mirroring the existing `Alpha101SpotKline`/`Alpha101Stock` pattern — same `FactorKunQuant` invocation shape per market, just a different `AllData`/data-columns wiring). Per the user: "alpha158拓展到美股非常简单，调用的步骤完全一致" — extending Alpha158 to US equities is straightforward, the invocation steps are identical to the existing Alpha101 pattern. Both Alpha101 and Alpha158 should work against both crypto spot and US equities.
- **Note (Claude's discretion, flagged by user but not blocking):** some Alpha158 factors may be cross-sectional (normalized across symbols) — whether/how cross-sectional factors need different handling per market is left to the planner/researcher to determine; the user does not consider this a blocker.
- **D-02:** `Alpha158Stock` needs an `amount` (dollar-volume) input for its VWAP-related features, which Tiingo data does not provide natively (unlike Binance's native `amount`/quote-volume column). Approximate it as `volume * close` (a standard dollar-volume proxy) rather than skipping VWAP-dependent features.

**New Polars factor engine (major architecture addition)**
- **D-03:** Extract a shared abstract base class from the existing `FactorKunQuant` (`base/factor.py`) — call it `Factor` (exact name is the planner's call) — covering everything market/backend-agnostic (config lifecycle, `read()`/`save()`, `get_features()`/`get_labels()`, `get_xarray_dataset()`, symbol/date bookkeeping). `FactorKunQuant` becomes a subclass of this new shared base (refactor, not a rewrite — behavior must not change for existing KunQuant factor classes). A new `FactorPolars(Factor)` ABC becomes a sibling implementation using Polars instead of KunQuant's compiled op-graph.
- **User's explicit requirement:** "我需要这两个因子类可以无缝替换" — `FactorKunQuant`-based and `FactorPolars`-based factor objects must be seamlessly interchangeable wherever a factor object is consumed today (e.g. `base/config.py`'s `DLConfig.factors`/`DLConfig.labels`/`MLConfig.factors`/`MLConfig.labels`, currently typed `list["FactorKunQuant"]` — this blast radius must be updated to the new shared `Factor` type so both backends type-check and behave identically from the consumer's point of view).
- **D-04 (Polars factor contract):** Analogous to `FactorKunQuant`'s abstract `_get_factor_func()` (which returns a KunQuant `Function`), `FactorPolars` gets an abstract method (e.g. `_get_factor_lazyframe()`) that the user overrides to write factor logic. It must return a `pl.LazyFrame` containing ONLY `timestamp`, `symbol`, and the computed factor value column(s) — no raw price/volume passthrough columns. Per the user: "用户在类似 `_get_factor_func()` 的函数中编写因子，返回一个只有 symbol date 因子列的 polars lazyframe，然后当用户调用 `cal()` 的时候开始计算" — the lazyframe must stay lazy (uncomputed) until `cal()` is actually invoked, matching KunQuant's `cal()` semantics of "trigger computation now."
- **D-05 (dynamic factor-name resolution):** Unlike KunQuant (which needs factor names declared upfront via `_get_factor_names()` for stream-buffer wiring), the Polars path has no such requirement. Per the user: "Polars 不会返回 Function() 和因子名称，所以要做针对性的设计，比如提取 polars dataframe 的属于因子的列名" — factor names for the Polars path are derived dynamically from the lazyframe's own schema (all column names except `timestamp`/`symbol`), not declared separately.
- **D-06 (pipeline-format boundary preserved):** `cal()` on `FactorPolars` triggers `.collect()` to materialize the lazyframe, then the result is converted to `xr.Dataset` before being handed off or persisted — Polars is used internally for the computation, but the module boundary contract (xarray/Zarr only between pipeline modules, per project-wide constraint) is unchanged. This mirrors the existing pattern already used elsewhere in the codebase (e.g. `dataset/cleaning.py` uses polars internally but the `Dataset` layer's public interface is xarray).
- **D-07 (batch-only, no streaming):** Reaffirms the original project-level decision — the Polars backend has no streaming/`cal_stream()` equivalent; it is batch-only.
- **D-08 (example factor):** Ship one simple example factor in `factor/` demonstrating the new `FactorPolars` pattern end-to-end (not a factor already covered by Alpha101/158). Exact formula is Claude's/the planner's discretion — the user's own answer focused entirely on the architecture/contract, not a specific factor formula, so a simple, well-understood factor (e.g. N-day price momentum, or a simple turnover/volume-based signal) is an acceptable default as long as it's genuinely computed via Polars lazyframe operations, not a trivial pass-through.

### Claude's Discretion
- Exact naming of the new shared `Factor` ABC and the new `FactorPolars`/`_get_factor_lazyframe()` method names (the user described the shape and contract, not literal identifiers).
- Whether/how Alpha158's cross-sectional-normalized factors need per-market handling differences (D-01's note). **Resolved by this research (see Architecture Patterns → Cross-Sectional Verification): Alpha158's predefined KunQuant factor set contains zero cross-sectional operators — no special per-market handling is needed.**
- The exact formula for the one example Polars-computed factor (D-08). **This research proposes N-day price momentum (see Code Examples).**
- How `cal_stream()`/streaming-specific methods on the current `FactorKunQuant` (`init_stream`, `_make_stream`, `_stream_context`) are organized relative to the new shared `Factor` base — they are KunQuant-specific and should NOT be hoisted onto the shared base or required by `FactorPolars`, but the exact class-hierarchy placement is an implementation detail.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FACTOR-01 | KunQuant 后端支持批量计算 Alpha158 因子集，输出 `xarray.Dataset` | Already works for crypto (`Alpha158SpotKline`, verified unchanged by refactor). New `Alpha158Stock` designed below (Architecture Patterns → Pattern 3), including the `amount` proxy fix required to not crash (Common Pitfalls #1). |
| FACTOR-02 | KunQuant 后端保留流式（`cal_stream`）计算能力，为未来实时数据接入预留接口 | `cal_stream()`/`init_stream()` code exists but has **zero exercised call sites** anywhere in the repo (verified by full-repo grep — every call site is commented out in `backtest/test_strategy.py`). A concrete batch-replay smoke-test design is proposed (Common Pitfalls #3 / Validation Architecture). |
| FACTOR-03 | 新增 Polars 批量因子计算后端接口，用于实现新因子（不要求复刻 Alpha101/Alpha158 已有公式），不需要支持流式 | Full `FactorPolars` contract, config class, and one example factor (`Momentum`) designed below (Architecture Patterns → Pattern 2, Pattern 4). |
| FACTOR-04 | 因子计算模块间的数据传输统一使用 `xarray.Dataset`，不使用 DataFrame 作为流水线传输格式 | `FactorPolars.cal()` converts internally-lazy Polars results to `xr.Dataset` before ever leaving the class (D-06) — verified idiom matches `dataset/stock.py`/`dataset/spot.py`'s existing `.to_pandas().set_index([...]).to_xarray()` pattern exactly. |

</phase_requirements>

## Summary

This phase is a **behavior-preserving refactor** of `base/factor.py:FactorKunQuant` to extract a shared `Factor(ABC)` base, plus a **net-new sibling** `FactorPolars(Factor)` implementing a Polars-lazy factor contract, plus **one net-new market extension** (`Alpha158Stock`) and **one net-new example factor** (a Polars-computed N-day momentum). No new third-party packages are required — `polars==1.44.1` and `KunQuant==0.1.11` are already pinned in `pyproject.toml` and installed in this repo's `uv` environment; this phase is pure application code.

The single most consequential finding is that **`Alpha101Stock` (the existing template D-01 says to mirror) is currently broken** — constructing its KunQuant op graph raises `RuntimeError: Bad inputs, given <class 'NoneType'>` (verified by direct execution against the installed KunQuant runtime) because `KunQuant.predefined.Alpha101.AllData.__init__` unconditionally builds `self.vwap = Div(self.amount, ...)` whenever no explicit `vwap` is passed, regardless of whether `amount` was supplied. Since `Alpha101Stock` never passes `amount`, every `.cal()` call on it fails immediately. This means D-02's `amount = volume * close` proxy is not just needed for the *new* `Alpha158Stock` — it must also be threaded through to fix the *existing*, already-broken `Alpha101Stock`, in the same place (`StockDataset._to_kunquant()`), to make D-01's "invocation steps are identical" claim actually true.

The second major finding resolves the CONTEXT.md's open discretion question about cross-sectional Alpha158 factors: **Alpha158's predefined KunQuant feature set (`kbar`/`price`/`volume`/`rolling` categories) contains zero cross-sectional operators** — every rolling/window function (`TsRank`, `WindowedAvg`, etc.) operates per-symbol, in time, only. (Alpha101, by contrast, genuinely does use `Rank` — a `SimpleCrossSectionalOp` — in many of its 101 formulas, but this is pre-existing, market-agnostic behavior unaffected by this phase.) No special per-market handling is needed for Alpha158Stock's cross-sectional behavior because there isn't any.

The third major finding governs the `FactorPolars` design: `Dataset.get_lazyframe()` (existing, `base/data.py`) is **not actually lazy with respect to disk I/O** — `XrBackend.get_lazyframe()` calls `self.data.to_dataframe()` eagerly (materializing the full in-memory xarray Dataset into pandas) before wrapping it in a Polars `LazyFrame` purely for API consistency with `PlBackend`. This matches KunQuant's own `cal()` (which also eagerly reads the full dataset via `Dataset.to_kunquant()`), so it is *not* a regression — but it does mean D-04's "must stay lazy until `cal()`" guarantee applies only to the *derived-factor* computation graph (the Polars expression chain), not to the underlying raw-data load, which is eager on both backends. The planner should design `FactorPolars.cal()` accordingly: call `self.config.dataset.read().get_lazyframe()` once (this triggers the underlying read), then thread the resulting `pl.LazyFrame` through the user's `_get_factor_lazyframe(lf)` override, then `.collect()` only at the very end.

**Primary recommendation:** Introduce `base/factor.py:Factor(ABC)` as a new top-of-hierarchy class holding every market/backend-agnostic method (read/save/config-lifecycle/get_features/get_labels/get_config), make `FactorKunQuant(Factor)` a pure refactor (zero behavior change, verified against every existing subclass and call site), add `FactorPolars(Factor)` as an independent sibling with its own `cal()` and a new `_get_factor_lazyframe()` abstract hook, split `FactorConfig` into a shared `BaseFactorConfig` (using `@dataclass(kw_only=True)` to sidestep dataclass field-ordering constraints) with `FactorConfig` (KunQuant) and `PolarsFactorConfig` (new, lighter) as siblings, fix `StockDataset._to_kunquant()` to synthesize `amount = volume * close` whenever requested (fixing `Alpha101Stock` and unblocking `Alpha158Stock`), and ship one new `factor/momentum.py:Momentum(FactorPolars)` example factor computed directly off `SpotKlineDataset`'s raw (un-renamed) lazyframe.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Factor config lifecycle (date defaults, dataset lookback-window reset) | Factor / Backend layer | — | Pure in-process dataclass bookkeeping; no I/O, no market-specific logic — identical for KunQuant and Polars. |
| KunQuant compiled-graph factor computation (batch) | Factor / Backend layer | Database / Storage (reads dataset's Zarr, writes factor's own Zarr) | `FactorKunQuant.cal()` orchestrates: read dataset → build/compile op graph → run compiled native code → persist. All in-process, single-machine. |
| KunQuant streaming factor computation | Factor / Backend layer | — | `cal_stream()`/`init_stream()` — designed for a future live-data tier (not yet built) to push bars into; this phase only proves the KunQuant-side half works via replay, not a live feed. |
| Polars-lazy factor computation (batch) | Factor / Backend layer | Database / Storage (reads dataset's Zarr via `get_lazyframe()`, writes factor's own Zarr) | Same orchestration shape as KunQuant's, substituting a Polars lazy-expression graph for a compiled op graph. |
| xarray/Zarr persistence of factor output | Database / Storage | — | `XrBackend` (already exists) — both backends converge on this at the module boundary (D-06/FACTOR-04). |
| Alpha158Stock's `amount` (dollar-volume) proxy computation | Database / Storage (`StockDataset._to_kunquant()`) | — | Must happen once, at the dataset→KunQuant-input boundary, so every KunQuant factor class reading Stock data via `data_columns=[...,"amount"]` gets it for free — not duplicated per factor class. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `KunQuant` | 0.1.11 (pinned in `pyproject.toml`, confirmed installed `[VERIFIED: pip/uv environment]`) | Compiled-graph batch + streaming factor computation | Already the project's primary factor engine (project-level Key Decision); no alternative under consideration. |
| `polars` | 1.44.1 (pinned in `pyproject.toml`, confirmed installed `[VERIFIED: pip/uv environment]`) | Lazy-expression batch factor computation (new `FactorPolars` backend) | Already used internally for raw ingestion (`dataset/spot.py`, `dataset/stock.py`, `dataset/cleaning.py`) and as a read-only view (`Dataset.get_lazyframe()`); `LazyFrame.collect_schema().names()` (needed for D-05's dynamic name resolution) confirmed present and working on this exact installed version `[VERIFIED: executed against installed polars 1.44.1]`. |
| `xarray` | 2026.7.0 (pinned, installed `[VERIFIED]`) | Sole cross-module exchange format (project-wide hard constraint) | No change — both new/refactored classes converge on `xr.Dataset` at their public boundary. |

No new third-party packages are required for this phase — **Package Legitimacy Audit is not applicable** (see below).

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Reusing `FactorConfig` as-is for `FactorPolars` | A new, lighter `PolarsFactorConfig` dataclass | Reusing `FactorConfig` would carry meaningless-but-required fields (`mode: Literal["stream","batch"]`, `data_columns: list`, `njobs: int`) that a Polars user would have to fill in with dummy values, violating QUAL-02's "no unnecessary code" requirement and inviting confusion about what `mode="stream"` even means for a backend that has no streaming. A dedicated lighter dataclass, sharing a common `BaseFactorConfig` parent, is cleaner. **Recommended: new `PolarsFactorConfig`.** |
| Eagerly resolving `FactorPolars.factor_names` at config-assignment time (mirroring `FactorKunQuant`'s eager `_get_factor_names()` call in the `config` setter) | Deferring resolution to `cal()`/`read()` time | Eager resolution would require calling `.collect_schema()` on a lazyframe sourced from `self.config.dataset.get_lazyframe()`, which itself requires `dataset.data_backend.data` to already be loaded in memory — forcing an unwanted disk read as a side effect of merely *constructing* a `FactorConfig`. Deferring to `cal()` (per D-05's spirit: "when the user calls `cal()`, computation begins") avoids this side effect. **Recommended: defer — see Common Pitfalls #4 for the resulting interchangeability caveat this introduces.** |

## Package Legitimacy Audit

**Not applicable.** This phase installs zero new external packages — `polars` and `KunQuant` are already declared in `pyproject.toml` (`polars>=1.44.1`, `kunquant>=0.1.11`) and confirmed installed and importable in the project's `uv` environment. No `slopcheck`/registry verification is required.

## Architecture Patterns

### System Architecture Diagram

```
                     ┌─────────────────────────────────────────┐
                     │        Dataset (existing, unchanged)      │
                     │  SpotKlineDataset / StockDataset          │
                     │  .read() -> xr.Dataset (Zarr)             │
                     │  .to_kunquant(cols) -> dict[str,ndarray]  │  <- KunQuant-shaped
                     │  .get_lazyframe() -> pl.LazyFrame         │  <- Polars-shaped (raw, un-renamed cols)
                     └───────────────┬───────────────┬───────────┘
                                     │               │
                     ┌───────────────▼───┐   ┌───────▼─────────────┐
                     │  FactorKunQuant    │   │   FactorPolars       │
                     │  .cal():           │   │   .cal():            │
                     │   dataset.to_kunquant()   dataset.read()      │
                     │   -> compile graph  │   │      .get_lazyframe()│
                     │   -> kr.runGraph    │   │   -> _get_factor_    │
                     │   -> dict[ndarray]  │   │      lazyframe(lf)   │
                     │   -> xr.Dataset     │   │   -> .collect_schema │
                     │                     │   │      ().names()      │
                     │  .cal_stream():     │   │      (dynamic names, │
                     │   incremental,      │   │       D-05)          │
                     │   1 bar at a time   │   │   -> .collect()      │
                     │   (KunQuant-only,   │   │   -> .to_pandas()    │
                     │    D-07: no Polars  │   │      .set_index()    │
                     │    equivalent)      │   │      .to_xarray()    │
                     └──────────┬──────────┘   └──────────┬───────────┘
                                │                          │
                                └──────────┬───────────────┘
                                           ▼
                          both produce ONLY xr.Dataset here
                          (module boundary — FACTOR-04 / D-06)
                                           │
                                           ▼
                          XrBackend -> Zarr (factor's own file_path)
                                           │
                                           ▼
                    BaseModel.collect() / get_features() / get_labels()
                    (consumes list["Factor"] uniformly — D-03 interchangeability)
```

A reader can trace the primary use case (compute a factor, get back `xr.Dataset`) through either backend by following the two parallel `.cal()` paths above — both converge on the same xarray/Zarr boundary before anything downstream ever sees them.

### Recommended Project Structure
```
base/
├── factor.py          # Factor(ABC) [NEW top of hierarchy] + FactorKunQuant(Factor) [refactored, behavior-preserving]
├── factor_polars.py    # FactorPolars(Factor) [NEW] -- separate module recommended (KunQuant vs Polars imports
│                        # don't need to load together; base/factor.py already imports both today, but splitting
│                        # keeps the KunQuant-heavy imports (cfake, KunRunner) out of a pure-Polars call path)
├── config.py           # BaseFactorConfig [NEW] / FactorConfig [refactored to subclass it] /
│                        # PolarsFactorConfig [NEW] / DLConfig.factors: list["Factor"] [changed] / MLConfig ditto
factor/
├── alpha101.py         # Alpha101SpotKline, Alpha101Stock [Alpha101Stock gets the amount-input bugfix]
├── alpha158.py         # Alpha158SpotKline, Alpha158Stock [NEW]
├── momentum.py         # Momentum(FactorPolars) [NEW example factor, D-08]
dataset/
├── stock.py            # StockDataset._to_kunquant() [modified: synthesize `amount` when requested, D-02]
config/
├── __init__.py         # + stock_alpha101_config(), stock_alpha158_config(), momentum_config() [NEW factories]
```

**Design note on `base/factor_polars.py` vs. putting `FactorPolars` in `base/factor.py`:** either is workable; a separate module is a minor readability/import-hygiene win (avoids `base/factor.py` importing `KunQuant.runner.KunRunner`/`KunQuant.jit.cfake` for a class that never uses them) but is not required for correctness. This is the planner's call.

### Pattern 1: Proposed Class Hierarchy (the core deliverable)

**What:** Every method/property of the current `FactorKunQuant` (`base/factor.py`), classified and re-homed.

| Method/Property | Classification | New Home | Notes |
|---|---|---|---|
| `__repr__` | (a) generic | `Factor` | As-is. |
| `_reset_dataset_config()` | (a) generic | `Factor` | As-is — uses `self.config.window`/`.dataset.config`, no KunQuant reference. |
| `import_path` (property) | (a) generic | `Factor` | As-is. |
| `class_name` (property) | (a) generic | `Factor` | As-is. |
| `read()` | (a) generic | `Factor` | As-is. |
| `save()` | (a) generic | `Factor` | As-is. |
| `_get_lazyframe()` | (a) generic | `Factor` | As-is (already backend-agnostic: reads `self.data_backend`, wraps in polars regardless of which subclass produced the data). |
| `_get_xarray_dataset()` | (a) generic | `Factor` | As-is. |
| `_get_features()` / `get_features()` | (a) generic | `Factor` | As-is (default raise, overridable — unchanged contract for every existing subclass). |
| `_get_labels()` / `get_labels()` | (a) generic | `Factor` | As-is. |
| `get_factor_names()` | (a) generic | `Factor` | As-is (`return self.config.factor_names`). |
| `get_config()` | (a) generic | `Factor` | As-is. |
| `num_factors` (property) | (a) generic | `Factor` | As-is (`len(self.get_factor_names())`) — see Common Pitfall #4 for the FactorPolars-specific precondition this introduces. |
| `__init__` (data_backend + config assignment) | (a) generic, partial | `Factor.__init__` | `self.data_backend = XrBackend()` and `self.config = config` hoist as-is. `self._stream_context`/`self._lib`/`self._buffer_name_to_id` stay KunQuant-only, set in `FactorKunQuant.__init__` after `super().__init__(config)`. |
| `init_stream()` | (b) KunQuant-only | `FactorKunQuant` | Unchanged. |
| `cal_stream()` | (b) KunQuant-only | `FactorKunQuant` | Unchanged. |
| `_make()` / `_make_stream()` | (b) KunQuant-only | `FactorKunQuant` | Unchanged. |
| `_to_xarray_dataset()` | (b) KunQuant-only | `FactorKunQuant` | Its signature (`dict[str, np.ndarray]` + `timestamps`/`symbols` ndarrays) is exactly `KunRunner.runGraph()`'s output shape — not shared; `FactorPolars` does its own 3-line conversion inline in `cal()` (see Pattern 2). Not worth abstracting a shared helper for a 3-line idiom used differently by each backend. |
| `_get_factor_func()` (abstract) | (b) KunQuant-only | `FactorKunQuant` | Stays abstract here, not on `Factor` — `FactorPolars` has no equivalent concept. |
| `cal()` | (c) → **abstract on `Factor`, fully separate implementation per subclass** | `Factor` (abstract) / `FactorKunQuant` + `FactorPolars` (concrete) | Per CONTEXT.md's own framing: same public signature (`-> Self`), entirely different internals. Declaring it abstract on `Factor` is what makes `BaseModel`'s `factor.cal()` calls polymorphic across both backends (the actual mechanism behind "seamless interchangeability" — see Integration Verification below). |
| `_get_factor_names()` (abstract) | (c) → **stays abstract on `Factor`, but with an overridable *resolution-timing* hook** | `Factor` (abstract) + `Factor._maybe_resolve_factor_names()` (new, concrete-but-overridable hook) | See "The `_get_factor_names()` timing split" below — this is the one genuinely subtle design decision in this refactor. |
| `_auto_filter()` | (c) → **default (batch-only) body hoisted to `Factor`, `FactorKunQuant` overrides to add the stream-mode skip** | `Factor` (default: always filter) / `FactorKunQuant` (override: skip when `mode=="stream"`) | `FactorPolars` never overrides this — it is always effectively "batch", so the inherited default is correct with zero extra code. |
| `symbols` / `num_symbols` (properties) | (c) → **default (batch) body hoisted to `Factor`, `FactorKunQuant` overrides to add the stream-mode branch** | `Factor` (default: `self.config.dataset.symbols`/`.num_symbols`) / `FactorKunQuant` (override: branch on `self.config.mode`) | Same hoist-with-override shape as `_auto_filter()`. |
| `config` (property/setter) | (c) → **shared skeleton on `Factor`, with `_maybe_resolve_factor_names()` as the one overridable step** | `Factor` | Date defaults + `_reset_dataset_config()` call are generic; the `factor_names` resolution line is the overridable hook described below. |

**The `_get_factor_names()` timing split (the one non-obvious design decision):**

`FactorKunQuant`'s existing `config` setter eagerly resolves `factor_names` at config-assignment time:
```python
if self._config.factor_names is None:
    self._config.factor_names = self._get_factor_names()
```
This is free for KunQuant subclasses — `Alpha101SpotKline._get_factor_names()` just enumerates `Alpha101.all_alpha` names, no I/O. But for `FactorPolars`, resolving names (D-05) means calling `.collect_schema().names()` on the lazyframe returned by the user's `_get_factor_lazyframe()` override — which needs `self.config.dataset.get_lazyframe()`, which needs the dataset's `data_backend.data` to already be loaded in memory (verified: `XrBackend.get_lazyframe()` calls `self.data.to_dataframe()`, which raises `AttributeError` if `self.data` was never set by a prior `.read()`/`.from_raw_data()`). Forcing this at config-*construction* time would make merely building a `PolarsFactorConfig` silently trigger a disk read — a surprising side effect, and one that contradicts D-04's "stay lazy until `cal()`" spirit.

**Recommendation:** introduce one new overridable hook, `Factor._maybe_resolve_factor_names()`, called from the shared `config` setter in place of the current inline `if`. `FactorKunQuant` inherits/keeps the eager default (zero behavior change — verified against every existing subclass, see Integration Verification). `FactorPolars` overrides it to a no-op, and instead resolves + assigns `self.config.factor_names` inside its own `cal()`, immediately before `.collect()`. `FactorPolars._get_factor_names()` (the abstract method every `Factor` subclass must implement, since `BaseModel.get_factor_names()`/`get_label_names()` call `factor._get_factor_names()` directly — verified in `base/model.py`) becomes:
```python
def _get_factor_names(self) -> tuple[str, ...]:
    if self.config.factor_names is None:
        raise RuntimeError(
            f"{self.class_name}: factor_names not resolved yet — "
            f"call cal() or read() first"
        )
    return tuple(self.config.factor_names)
```
This is a **documented, narrow interchangeability caveat** (see Common Pitfalls #4), not a violation of D-03's interchangeability requirement — `BaseModel` never calls `_get_factor_names()` before `.collect()` (which calls `.cal()`/`.read()` on every factor first), so this precondition is already satisfied by existing call order.

### Pattern 2: `FactorPolars` Contract (D-04/D-05/D-06)

**What:** The abstract method a user overrides, its exact signature, and how `cal()` orchestrates it.

**When to use:** Any new factor whose logic is easier to express as a Polars lazy expression than a KunQuant op graph, and which does not need streaming (D-07).

```python
# base/factor_polars.py (proposed)
from abc import abstractmethod
from typing import Self

import polars as pl
import xarray as xr

from base.config import PolarsFactorConfig
from base.factor import Factor


class FactorPolars(Factor):
    def __init__(self, config: PolarsFactorConfig):
        super().__init__(config)

    def _maybe_resolve_factor_names(self) -> None:
        # Deferred to cal() -- see Pattern 1's timing-split rationale.
        pass

    def _get_factor_names(self) -> tuple[str, ...]:
        if self.config.factor_names is None:
            raise RuntimeError(
                f"{self.class_name}: factor_names not resolved yet — "
                f"call cal() or read() first"
            )
        return tuple(self.config.factor_names)

    @abstractmethod
    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Receive the dataset's raw (already-read, un-renamed) lazyframe;
        return a LazyFrame containing ONLY timestamp, symbol, and computed
        factor value column(s). Must remain uncomputed (no .collect() calls
        here) -- cal() triggers materialization (D-04)."""
        ...

    def cal(self) -> Self:
        lf = self.config.dataset.read().get_lazyframe()
        factor_lf = self._get_factor_lazyframe(lf)

        schema_names = factor_lf.collect_schema().names()
        factor_names = tuple(
            n for n in schema_names if n not in ("timestamp", "symbol")
        )
        self.config.factor_names = factor_names

        df = factor_lf.collect().to_pandas().set_index(["timestamp", "symbol"])
        ds = xr.Dataset.from_dataframe(df)

        self.data_backend.to_internal(ds)
        self._auto_filter()
        return self
```

**Why `_get_factor_lazyframe(self, lf)` takes the dataset lazyframe as a parameter** (rather than pulling it itself via `self.config.dataset.get_lazyframe()` internally, the way `FactorKunQuant._get_factor_func()` builds its own `Input(...)` nodes with no arguments): this makes the user's factor logic **unit-testable in isolation** — a test can call `SomeFactor(config)._get_factor_lazyframe(synthetic_lf)` directly with a hand-built `pl.LazyFrame` fixture, without needing a fully-wired `Dataset`/Zarr file on disk. `FactorKunQuant`'s no-argument pattern exists because KunQuant's `Builder()`/`Input()` context-manager idiom has no equivalent "pass the data in" mechanism — that asymmetry is inherent to the two engines, not a contract this phase should try to force into matching.

**`collect_schema()` verified on installed polars 1.44.1:**
```python
>>> import polars as pl
>>> lf = pl.LazyFrame({"a": [1], "b": [2]})
>>> lf.collect_schema().names()
['a', 'b']
```
`[VERIFIED: executed against the project's own uv-managed polars 1.44.1]` — no version-compatibility risk; `collect_schema()` has been the recommended API (over the older, deprecated `.columns` property, which the polars source itself warns triggers "a potentially expensive operation") since polars 0.20, well below the pinned `>=1.44.1` floor.

**xr.Dataset conversion idiom — verified consistent with the rest of the codebase:** `dataset/stock.py:_raw_data_to_xr()` and `dataset/spot.py:_raw_data_to_xr()` both end with `data.collect().to_pandas().set_index([...]).to_xarray()`; `dataset/backend.py:PlBackend.get_xarray_dataset()` uses the equivalent `xr.Dataset.from_dataframe(data.set_index(indexes))`. The proposed `FactorPolars.cal()` above uses the same idiom — no new conversion pattern introduced.

### Pattern 3: `Alpha158Stock` Design (D-01/D-02)

**What:** Mirror `Alpha158SpotKline` exactly, with two differences: (1) it's constructed with a `StockDataset`-backed config, and (2) `StockDataset._to_kunquant()` must synthesize an `amount` column when requested (D-02), which fixes a genuine pre-existing bug shared by `Alpha101Stock`.

**Verified bug (not hypothetical) — `Alpha101Stock` crashes today:**
```python
>>> from KunQuant.Op import Input
>>> from KunQuant.predefined import Alpha101
>>> close, low, high, vopen, vol = (Input(n) for n in ("close","low","high","open","volume"))
>>> Alpha101.AllData(low=low, high=high, close=close, open=vopen, volume=vol)
RuntimeError: Bad inputs, given <class 'NoneType'>
```
`[VERIFIED: executed against the project's own uv-managed KunQuant 0.1.11]`. Root cause: `KunQuant/predefined/Alpha101.py:AllData.__init__` (and the identical pattern in `Alpha158.py:AllData.__init__`) unconditionally computes `self.vwap = Div(self.amount, AddConst(self.volume, ...))` whenever `vwap` is not explicitly passed — **regardless of whether `amount` was supplied**. `Alpha101Stock._get_factor_func()` (`factor/alpha101.py`) never passes `amount`, so this line always raises the moment `.cal()`/`._make()` is invoked. `test.py`'s exploratory `Alpha101Stock` smoke test never actually calls `.cal()` (only `.get_features()` against an unpopulated backend, which would separately fail with `AttributeError` — confirming this class has never been exercised end-to-end).

**Fix (single location, benefits both `Alpha101Stock` and the new `Alpha158Stock`):**
```python
# dataset/stock.py: StockDataset._to_kunquant(), after the existing rename block
data = data.sortby(["timestamp", "symbol"])
if "amount" in data_columns and "amount" not in data.data_vars:
    # D-02: Tiingo has no native dollar-volume column; approximate as
    # volume * close (standard dollar-volume proxy).
    data = data.assign(amount=data["volume"] * data["close"])
timestamp = data["timestamp"].values
...
```
Then update `Alpha101Stock._get_factor_func()` to accept and pass `amount=Input("amount")` to `AllData(...)` (currently omitted — the actual bug), and add `"amount"` to the `data_columns` list in a new `stock_alpha101_config()` factory. This is a small, low-risk, necessary fix — without it, D-01's claim that Alpha101/158 "invocation steps are identical" across markets is false for any alpha that touches `vwap`.

**`Alpha158Stock` (new, `factor/alpha158.py`):**
```python
class Alpha158Stock(FactorKunQuant):
    def __init__(self, factor_config: FactorConfig):
        super().__init__(factor_config)

    def _get_factor_names(self) -> tuple[str, ...]:
        return tuple(self._factor_names_stream())

    def _get_func_names(self):
        close = Input("close"); low = Input("low"); high = Input("high")
        vopen = Input("open"); amount = Input("amount"); vol = Input("volume")
        all_data = Alpha158.AllData(
            low=low, high=high, close=close, open=vopen, amount=amount, volume=vol
        )
        # identical build() config dict to Alpha158SpotKline -- D-01's
        # "invocation steps are identical" claim, now actually true because
        # `amount` is supplied via the StockDataset._to_kunquant() proxy above.
        return all_data.build({
            "kbar": {},
            "price": {"windows": [0, 1, 2, 3, 4],
                       "feature": [("OPEN", all_data.open), ("HIGH", all_data.high),
                                   ("LOW", all_data.low), ("CLOSE", all_data.close),
                                   ("VWAP", all_data.vwap)]},
            "volume": {"windows": [0, 1, 2, 3, 4]},
            "rolling": {"windows": [5, 10, 20, 30, 60], "exclude": ["BETA", "RSQR", "RESI"]},
        })

    def _factor_names_stream(self):
        return self._get_func_names()[-1]

    def _get_func_stream(self) -> Function:
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("close"); low = Input("low"); high = Input("high")
            vopen = Input("open"); amount = Input("amount"); vol = Input("volume")
            alpha158, names = self._get_func_names()
            for v, k in zip(alpha158, names):
                if k in factor_names:
                    Output(WindowedZScore(v, self.config.window), k)
        return Function(builder.ops)

    def _get_factor_func(self):
        return self._get_func_stream()

    def _get_labels(self, data): raise RuntimeError(...)
    def _get_features(self, data): return data
```
This is a byte-for-byte structural copy of `Alpha158SpotKline` (per D-01) — the only wiring difference is which `Dataset` subclass the `FactorConfig.dataset` points at, and that difference lives entirely in the new `stock_alpha158_config()` factory, not in the factor class itself.

**New config factories (`config/__init__.py`):**
```python
def stock_alpha101_config(..., market="us_equity", frequency="1d"):
    return FactorConfig(
        file_path=str(_data_root() / "data" / "factor" / "alpha101_stock.zarr"),
        dataset=StockDataset(stock_kline_config(symbols=symbols)),
        data_columns=["high", "low", "close", "open", "volume", "amount"],
        ...
    )

def stock_alpha158_config(..., market="us_equity", frequency="1d"):
    return FactorConfig(
        file_path=str(_data_root() / "data" / "factor" / "alpha158_stock.zarr"),
        dataset=StockDataset(stock_kline_config(symbols=symbols)),
        data_columns=["high", "low", "close", "open", "volume", "amount"],
        window=128,
        ...
    )
```

### Cross-Sectional Verification (resolves CONTEXT.md's open discretion question)

Read directly from the installed `KunQuant/predefined/Alpha158.py` (`AllData.build()`): the four feature categories (`kbar`, `price`, `volume`, `rolling`) use exclusively per-symbol time-series operators — `WindowedAvg`, `WindowedStddev`, `WindowedQuantile`, `TsRank` (time-series rank *within a rolling window for one symbol*, not across symbols), `TsArgMax`/`TsArgMin`, `WindowedCorrelation`, `WindowedLinearRegressionSlope`, etc. **None of Alpha158's ~150 predefined features use `Rank`/`Scale`/any cross-sectional operator.** `[VERIFIED: read full source, `/predefined/Alpha158.py`, 242 lines]`

By contrast, `KunQuant/predefined/Alpha101.py` genuinely does use `rank(v) -> Rank(v)`, and `Rank` is declared as `class Rank(SimpleCrossSectionalOp)` in `KunQuant/Op.py` `[VERIFIED: grep on installed package]` — i.e., dozens of the 101 formulas compute a percentile rank *across every symbol in the same compiled-graph batch* at each timestamp. This is pre-existing, already-shipped behavior (identical for `Alpha101SpotKline` today) — extending it to `Alpha101Stock`/US equities introduces no new complexity: the "universe" a cross-sectional rank is computed against is simply whatever `symbols` list is passed into that `FactorConfig`, market-agnostic.

**Conclusion: no special cross-sectional handling is needed for `Alpha158Stock`** (it has no cross-sectional features to handle), and none is needed for fixing `Alpha101Stock` either (its existing cross-sectional behavior already works the same way regardless of market).

### Pattern 4: Example Polars Factor (D-08)

**What:** `factor/momentum.py:Momentum(FactorPolars)` — N-day price momentum, computed as a genuinely-lazy Polars window expression (`.over("symbol")`), built directly against `SpotKlineDataset`'s **raw** (un-renamed) lazyframe.

**Why built on `SpotKlineDataset` specifically, using its raw Title-Case columns directly** (rather than trying to build a market-agnostic column-normalization layer for `Dataset.get_lazyframe()`): `Dataset.get_lazyframe()` (existing, `base/data.py`) returns whatever raw column names the underlying Zarr store has — Title-Case (`Open`/`High`/`Low`/`Close`/`Volume`) for `SpotKlineDataset`, lowercase `adj*`-prefixed for `StockDataset`. Unlike `_to_kunquant()` (which each `Dataset` subclass overrides specifically to normalize names before feeding a KunQuant graph), there is **no equivalent normalization step for `get_lazyframe()`** today, and CONTEXT.md's D-04/D-05/D-06 discussion is scoped entirely to the `Factor`-side contract, not to adding a new `Dataset`-side normalization layer. Building the one example factor against `SpotKlineDataset`'s already-unambiguous raw OHLCV columns keeps this phase's scope tight; a market-agnostic `get_lazyframe()` naming contract is flagged as an Open Question for a future phase, not solved here.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|--------------|-----|
| Rolling-window factor value normalization | A custom Polars/KunQuant z-score op | `my_ops/preprocess.py:WindowedZScore` (KunQuant side, already exists, already used by `Alpha101SpotKline`/`Alpha158SpotKline`) | Reuse for `Alpha158Stock` exactly as `Alpha158SpotKline` does — no reason to reimplement for a second market. |
| Dynamic column-name introspection for a Polars pipeline | Manual `.columns` inspection or eager `.collect()` just to peek at schema | `pl.LazyFrame.collect_schema().names()` | Schema-only, does not materialize data — the correct primitive for D-05's dynamic factor-name resolution; avoids the deprecated, more expensive `.columns` property. |
| Cross-sectional rank/percentile computation | A hand-rolled per-timestamp groupby-rank in Polars or numpy | KunQuant's `Rank`/`SimpleCrossSectionalOp` (already used by Alpha101; not needed for this phase's new Polars factor, but relevant if a future Polars factor ever needs cross-sectional behavior — Polars' own `.rank().over("timestamp")` is the correct native primitive, not a custom loop) | Not exercised in this phase (the example factor is intentionally per-symbol time-series only), but worth noting for the planner in case a future Polars factor needs it. |

**Key insight:** every "don't hand-roll" risk in this phase is really about **reusing the existing normalization primitive (`WindowedZScore`) for the new market**, not about needing anything new — the phase's actual net-new complexity is entirely architectural (the class hierarchy split), not algorithmic.

## Integration Verification (behavior-preservation check)

Every existing `FactorKunQuant` subclass and call site was read and checked against the proposed hierarchy:

| File | Usage | Breaks under proposed hierarchy? |
|------|-------|-----------------------------------|
| `factor/alpha101.py` (`Alpha101SpotKline`, `Alpha101Stock`) | Subclasses `FactorKunQuant`, implements `_get_factor_func`/`_get_factor_names`/`_get_labels`/`_get_features` | No — all four remain exactly where they are (bucket (b)/(a) per subclass contract), zero signature changes. |
| `factor/alpha158.py` (`Alpha158SpotKline`) | Same shape, plus `_get_func_names`/`_factor_names_stream`/`_get_func_stream` (internal helpers, not part of any ABC contract) | No — internal helper methods are untouched by the base-class split. |
| `label/spot.py` (`SpotReturn`, `SpotBinaryReturn`) | Subclasses `FactorKunQuant`, uses `op.SubConst`/`op.Div`/`op.BackRef`/`op.Select` directly (KunQuant-only ops) | No — no reference to anything being hoisted differently. |
| `cal.py` | Instantiates `Alpha101SpotKline(alpha101_config())`, calls `.cal().save()` | No — public API unchanged. |
| `train_model.py` | Instantiates `Alpha101SpotKline`/`Alpha158SpotKline`, passes list into `DLConfig(factors=[...])` | No — `DLConfig.factors` type-hint changes to `list["Factor"]`, but `FactorKunQuant` instances still satisfy it (it's a widening, not a narrowing, of the type). |
| `test.py` | Constructs `FactorConfig(window=10, dataset=ds, mode="batch", data_columns=[...])` directly (all-kwarg call), then `Alpha101Stock(facfg)`, then `.get_features()` (no `.cal()`/`.read()` — already broken/unexercised, confirmed above) | No new breakage — `FactorConfig`'s public kwarg-based construction is unchanged (see `@dataclass(kw_only=True)` note below); this script's pre-existing dysfunction is a documented finding, not a regression this phase introduces. |
| `backtest/test_strategy.py` | Instantiates `Alpha101SpotKline`/`Alpha158SpotKline`, calls `.num_factors`, `.read().get_features()` — all `cal_stream()` call sites are commented out | No — every method actually invoked (`.num_factors`, `.read()`, `.get_features()`) is bucket (a), unchanged. |
| `test_nt.ipynb` | Grepped — zero references to `Factor`/`FactorKunQuant`/any factor class | Not applicable. |
| `config/__init__.py` (`alpha101_config`/`alpha158_config`/`spot_label_config`) | Return `FactorConfig` instances (never a factor instance) | No — `FactorConfig`'s field set is unchanged; only its *parent class* changes (new `BaseFactorConfig` ancestor), which is invisible to any code that only constructs it via keyword arguments (verified every call site does). |
| `base/config.py` (`DLConfig.factors`/`.labels`, `MLConfig.factors`/`.labels`) | Typed `list["FactorKunQuant"]` | **Must change** to `list["Factor"]` (this is the intended, required blast-radius fix per D-03's explicit interchangeability requirement — not a break, the fix itself). |
| `base/model.py` (`BaseModel`) | Calls, on each `factor`/`label` item: `.config.start_date =`, `.config.end_date =`, `._reset_dataset_config()`, `.cal()`, `.get_features()`/`.get_labels()`, `.read()`, `._get_factor_names()` (directly, not via the public `get_factor_names()` wrapper — verified in `base/model.py:get_factor_names()`/`get_label_names()`), `.get_config()` | No — every one of these is either concrete-generic on `Factor` or abstract-and-required on `Factor` (`cal()`, `_get_factor_names()`), so both `FactorKunQuant` and `FactorPolars` instances satisfy every call `BaseModel` makes. **This is the actual proof of "seamless interchangeability" (D-03), not just a type-hint change.** |
| `utils/module.py` (`load_factor_from_config`) | Hardcodes `FactorConfig(**config)` when reconstructing a factor from a saved JSON checkpoint config | **Not fixed in this phase** (see Open Questions #1) — this path is only reachable from Phase 4's model-checkpoint reload flow, not from any of this phase's FACTOR-01..04 success criteria. Flagged forward for Phase 4 planning, not blocking here. |

**Dataclass field-ordering note (implementation detail, not a design risk):** splitting `FactorConfig`'s fields across a new `BaseFactorConfig` parent requires either `@dataclass(kw_only=True)` (Python 3.10+; this project targets `>=3.13`, confirmed compatible) on `BaseFactorConfig`/`FactorConfig`/`PolarsFactorConfig`, or giving `FactorConfig`'s currently-mandatory `mode`/`data_columns` fields defaults. **Recommend `kw_only=True`** — it preserves "these fields are still required" semantics exactly, and every real call site in the repo already constructs `FactorConfig`/`DatasetConfig` with 100% keyword arguments (verified: `config/__init__.py`, `test.py` — zero positional-argument construction found anywhere), so this is a purely additive, zero-behavior-change constraint.

## Common Pitfalls

### Pitfall 1: `Alpha101Stock`/`AllData` crashes without an explicit `amount` or `vwap`
**What goes wrong:** `KunQuant.predefined.Alpha101.AllData.__init__` (and the identical line in `Alpha158.py`) always executes `self.vwap = Div(self.amount, AddConst(self.volume, ...))` when `vwap` is not explicitly passed, even if no downstream alpha uses `vwap` and even if `amount` itself was never passed (`amount=None` by default). The result is an immediate `RuntimeError: Bad inputs, given <class 'NoneType'>` the moment `AllData(...)` is constructed, not a lazy/deferred failure.
**Why it happens:** `AllData.__init__` does not `try`/guard the `vwap` fallback computation on whether `amount` is actually present.
**How to avoid:** Always pass `amount` (real, for `SpotKlineDataset`/Binance; approximated as `volume * close`, for `StockDataset`/Tiingo, per D-02) to every `AllData(...)` construction, for every market, for every alpha set. Implement the approximation once, centrally, in `StockDataset._to_kunquant()` (Pattern 3).
**Warning signs:** Any `KunQuant`-based factor class targeting `StockDataset` that omits `amount` from both `data_columns` and its `AllData(...)` call will fail at `.cal()`/`._make()` time with the exact `RuntimeError` above — this is a hard, immediate, 100%-reproducible failure, not a flaky edge case.

### Pitfall 2: `cal_stream()`/`init_stream()` have never been exercised — zero live call sites exist
**What goes wrong:** A full-repo grep for `cal_stream`/`init_stream` finds exactly two non-definition matches, both in `backtest/test_strategy.py`, and both are commented out (`# self.alpha101.cal_stream(...)`). There is no test, script, or notebook anywhere that has ever successfully called `cal_stream()` against the installed KunQuant runtime.
**Why it happens:** The streaming path was built for a future live-data-feed integration that doesn't exist yet in this codebase (per PROJECT.md: "保留未来实时数据接入能力" — reserved for future real-time data access, not v1's actual live trading).
**How to avoid:** FACTOR-02's success criterion ("invoke KunQuant's streaming path and get incremental factor updates without error") must be satisfied via a **synthetic batch-replay smoke test** (proposed in Validation Architecture below) — feed already-stored historical bars into `cal_stream()` one timestamp at a time, in a test, and assert (a) no exception and (b) output shape matches `(1, num_symbols)` per factor name. This is the only way to give FACTOR-02 real coverage without an actual live feed.
**Warning signs:** Do not assume `cal_stream()` "already works" because the code compiles/imports cleanly — the compiled-graph construction (`_make_stream()`, `KunCompilerConfig(input_layout="STREAM", ...)`) and the runtime buffer-wiring (`queryBufferHandle`/`pushData`/`getCurrentBuffer`) are both entirely unexercised paths.

### Pitfall 3: KunQuant's STREAM layout expects `num_stock` to be a multiple of 8 `[ASSUMED: performance-vs-correctness distinction unverified]`
**What goes wrong:** KunQuant's own documentation/tests (`tests/test_stream.py` in the upstream repo, and the general "blocking_len" guidance referenced in the README) consistently use stock counts that are multiples of 8 (`blocking_len=8` for AVX2/float32, matching `_make_stream()`'s `KunCompilerConfig(blocking_len=8, partition_factor=8, ...)` already used by this codebase). The upstream project's phrasing ("suggested," "for proper performance") leaves ambiguous whether a non-multiple-of-8 `num_stock` merely runs slower or actually produces incorrect/undefined output for STREAM-layout graphs specifically (BATCH/"TS" layout graphs, which is what `.cal()` already uses successfully for arbitrary symbol counts today, are unaffected).
**Why it happens:** SIMD blocking (AVX2, 8 float32 lanes) is a natural fit for STREAM's incremental per-tick processing model in a way that doesn't apply the same way to the batch "TS" layout.
**How to avoid:** When designing the batch-replay streaming smoke test (Pitfall 2), use exactly 8 symbols (or another multiple of 8) to sidestep the ambiguity entirely, rather than resolving whether it's a hard requirement.
**Warning signs:** If the smoke test is later run with a non-multiple-of-8 symbol count and produces silently wrong (not erroring) factor values, this is the first place to look.

### Pitfall 4: `FactorPolars.num_factors`/`_get_factor_names()` raise before `cal()`/`read()` has run — an intentional, narrow interchangeability gap
**What goes wrong:** Unlike `FactorKunQuant` subclasses (whose `_get_factor_names()` is a static, I/O-free computation, valid immediately after construction), `FactorPolars._get_factor_names()` depends on `self.config.factor_names` having already been populated by a prior `.cal()` or `.read()` call (Pattern 1's timing-split). Calling `.num_factors` or `._get_factor_names()` on a freshly-constructed `FactorPolars` instance, before either of those, raises `RuntimeError`.
**Why it happens:** D-05's dynamic-name-resolution requirement genuinely cannot be satisfied without first materializing (or at least schema-inspecting) the dataset's lazyframe, which itself requires the dataset to have been read.
**How to avoid:** Document this precondition explicitly on `FactorPolars`/`_get_factor_lazyframe()`'s docstring. Verify (already done, see Integration Verification) that `BaseModel`'s actual usage pattern always calls `.cal()`/`.read()` before `_get_factor_names()`, so this gap does not affect any code path exercised by this phase or by the Phase-4 model layer as currently written.
**Warning signs:** A future caller (e.g. a hypothetical Phase-4+ UI/introspection tool that lists factor names without computing them) hitting this `RuntimeError` unexpectedly — flag for that future phase's own research, not fixable generically here without abandoning D-05's laziness requirement.

### Pitfall 5: `Dataset.get_lazyframe()` is not actually lazy with respect to disk I/O
**What goes wrong:** `XrBackend.get_lazyframe()` (`dataset/backend.py`) implements `pl.LazyFrame` conversion as `pl.from_pandas(self.data.to_dataframe().reset_index()).lazy()` — the `.to_dataframe()` call is fully eager (materializes the entire in-memory `xr.Dataset` into a pandas DataFrame) before ever touching Polars. The `.lazy()` at the end only defers *subsequent* Polars operations, not the underlying data materialization, which has already happened by the time `_get_factor_lazyframe(lf)` receives `lf`.
**Why it happens:** `XrBackend`'s `data` attribute is already a fully-loaded, in-memory `xr.Dataset` (loaded via `xr.open_dataset()` in `.read()`) — there is no lazy/chunked xarray→polars bridge in use here.
**How to avoid:** This is not a bug to fix in this phase — `FactorKunQuant.cal()` has the identical characteristic (`Dataset.to_kunquant()` also fully reads the dataset before building the compiled graph). Simply do not describe D-04's "lazy until `cal()`" contract to the planner/executor as "the raw data load is deferred" — it isn't, on either backend. Only the *derived-factor* Polars expression chain (whatever the user's `_get_factor_lazyframe()` builds on top of the already-loaded `lf`) is actually deferred until `.collect()`.
**Warning signs:** Confusion during implementation about why `FactorPolars.cal()` seems to "already have the data" the moment `_get_factor_lazyframe(lf)` is called — this is expected, not a leak of the laziness contract.

## Code Examples

### Proposed `Factor(ABC)` base (skeleton, showing the hoisted (a)/(c) methods)
```python
# base/factor.py (proposed shared base -- generic methods only; KunQuant-only
# methods (init_stream, cal_stream, _make, _make_stream, _to_xarray_dataset,
# _get_factor_func) move to FactorKunQuant, shown separately)
from abc import ABC, abstractmethod
from typing import Literal, Self

import pandas as pd
import polars as pl
import xarray as xr

from base.config import BaseFactorConfig
from dataset.backend import XrBackend
from enums.constant import Date
from utils.timer import Timer


class Factor(ABC):
    def __init__(self, config: "BaseFactorConfig"):
        self.config = config
        self.data_backend = XrBackend()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> "BaseFactorConfig":
        return self._config

    @config.setter
    def config(self, config: "BaseFactorConfig"):
        self._config = config
        self._config.name = self.import_path
        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE
        self._maybe_resolve_factor_names()
        self._reset_dataset_config()

    def _maybe_resolve_factor_names(self) -> None:
        if self._config.factor_names is None:
            self._config.factor_names = self._get_factor_names()

    def _reset_dataset_config(self):
        start_date = pd.to_datetime(self._config.start_date)
        start_date = start_date - pd.DateOffset(days=self._config.window)
        self._config.dataset.config.start_date = start_date.strftime("%Y-%m-%d")
        self._config.dataset.config.end_date = self._config.end_date

    def _auto_filter(self):
        self.data_backend.filter_by_date(
            col="timestamp",
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        if self.config.symbols is not None:
            self.data_backend.filter_by_symbol("symbol", self.config.symbols)

    @property
    def symbols(self) -> list[str]:
        return self.config.dataset.symbols

    @property
    def num_symbols(self) -> int:
        return self.config.dataset.num_symbols

    @property
    def num_factors(self) -> int:
        return len(self.get_factor_names())

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def read(self) -> Self:
        self.data_backend.read(self.config.file_path)
        self._auto_filter()
        return self

    def save(self, mode: Literal["a", "w"] = "a", **kwargs) -> Self:
        with Timer(f"{self.__class__.__name__}: save"):
            self._auto_filter()
            self.data_backend.write(self.config.file_path, mode=mode, **kwargs)
            return self

    def _get_lazyframe(self) -> pl.LazyFrame:
        df = self.data_backend.get_xarray_dataset().to_pandas()  # type: ignore
        return pl.LazyFrame(df.reset_index())

    def _get_xarray_dataset(self) -> xr.Dataset:
        return self.data_backend.get_xarray_dataset()  # type: ignore

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError

    def get_features(self) -> xr.Dataset:
        return self._get_features(self._get_xarray_dataset())

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError

    def get_labels(self) -> xr.Dataset:
        return self._get_labels(self._get_xarray_dataset())

    def get_factor_names(self) -> tuple[str, ...]:
        return self.config.factor_names

    def get_config(self) -> dict:
        ds_config = self.config.dataset.get_config()
        cfg = self.config.to_dict()
        cfg["dataset"] = ds_config  # type: ignore
        return cfg  # type: ignore

    @abstractmethod
    def _get_factor_names(self) -> tuple[str, ...]: ...

    @abstractmethod
    def cal(self) -> Self: ...
```

### Proposed config dataclass split
```python
# base/config.py (proposed)
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .data import Dataset
    from .factor import Factor  # was: FactorKunQuant


@dataclass(kw_only=True)
class BaseFactorConfig:
    window: int
    dataset: "Dataset"
    file_path: str | None = None
    factor_names: list | None = None
    start_date: str | None = None
    end_date: str | None = None
    symbols: list | None = None
    kwargs: dict | None = None
    name: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class FactorConfig(BaseFactorConfig):
    mode: Literal["stream", "batch"]
    data_columns: list
    njobs: int = 128


@dataclass(kw_only=True)
class PolarsFactorConfig(BaseFactorConfig):
    pass  # no additional fields needed for D-04..D-07's scope


@dataclass
class DLConfig:
    factors: list["Factor"]   # was: list["FactorKunQuant"]
    labels: list["Factor"]    # was: list["FactorKunQuant"]
    ...  # rest unchanged
```

### Example Polars factor — N-day momentum (D-08)
```python
# factor/momentum.py (proposed)
import polars as pl
import xarray as xr

from base.config import PolarsFactorConfig
from base.factor_polars import FactorPolars


class Momentum(FactorPolars):
    """N-day price momentum: (close / close.shift(n)) - 1, per symbol.

    Built directly against SpotKlineDataset's raw Title-Case columns
    (Open/High/Low/Close/Volume) -- see RESEARCH.md Pattern 4 for why this
    example intentionally does not attempt a market-agnostic column-name
    contract.
    """

    def __init__(self, factor_config: PolarsFactorConfig):
        super().__init__(factor_config)

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.config.kwargs.get("n", 20) if self.config.kwargs else 20
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (pl.col("Close") / pl.col("Close").shift(n).over("symbol") - 1)
                .alias(f"momentum_{n}")
            )
            .select(["timestamp", "symbol", f"momentum_{n}"])
        )

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data

    def _get_labels(self, data: xr.Dataset):
        raise RuntimeError(f"{self.class_name} does not support get_labels()")
```

### Streaming batch-replay smoke test sketch (FACTOR-02)
```python
# tests/test_factor_stream.py (proposed shape)
def test_cal_stream_replays_batch_data_without_error():
    # 1. Build a small SpotKlineDataset fixture with exactly 8 symbols
    #    (Pitfall 3 -- sidesteps the multiple-of-8 ambiguity).
    # 2. Construct Alpha101SpotKline (or a minimal single-alpha subclass) in
    #    mode="stream" with symbols=[...8 symbols...].
    # 3. Read the batch xr.Dataset once (ground truth), then loop over its
    #    sorted unique timestamps:
    #      for ts in timestamps:
    #          bar = batch_ds.sel(timestamp=ts)
    #          data = {col: bar[col].values.astype(np.float32) for col in data_columns}
    #          factor.cal_stream(data, ts.astype("int64"), symbols)
    # 4. Assert no exception is raised across the full replay loop, and that
    #    factor.get_features() after the last step has shape
    #    (1, num_symbols) per factor name in factor.config.factor_names.
```

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | KunQuant's STREAM layout requires `num_stock` to be an exact multiple of 8 for *correctness* (not just performance) | Common Pitfalls #3 | If actually just a performance suggestion, the smoke test's choice of exactly-8-symbols is unnecessarily conservative but still correct (no downside either way — flagged purely so the planner doesn't spend time trying to prove the boundary further than needed). |
| A2 | `volume * close` is an acceptable proxy for `amount` for the *streaming* smoke test too (not just batch `Alpha158Stock`) | Pitfall 1 / Pattern 3 | Low — this is CONTEXT.md's own locked D-02 decision, not a new assumption; only the extension "also applies uniformly wherever `amount` is needed" is this research's addition, and it's low-risk since it's the same formula in the same place. |

**If this table is empty:** N/A — see above; both entries are low-risk, narrowly-scoped extrapolations of already-locked/verified findings, not load-bearing unverified claims.

## Open Questions (RESOLVED)

1. **`utils/module.py:load_factor_from_config()` hardcodes `FactorConfig(**config)`, which will break for a persisted `FactorPolars`/`PolarsFactorConfig`-based factor.** — RESOLVED: deferred to Phase 4
   - What we know: This function is only reachable from `load_model_from_config()`, used by Phase 4's model-checkpoint reload flow — not exercised by any of this phase's FACTOR-01..04 success criteria.
   - What's unclear: Whether Phase 4's planner wants a `Factor.config_cls: ClassVar[type]` pattern (each subclass declares which config dataclass it needs) or a different reload mechanism.
   - Recommendation: Do not fix in this phase (out of scope — no success criterion touches checkpoint reload of a Polars-backed factor). Flag explicitly for Phase 4 planning.

2. **Should `Dataset.get_lazyframe()` eventually gain a market-agnostic column-normalization contract** (mirroring what `_to_kunquant()` already does per-subclass), so future Polars factors can be written once and reused across `SpotKlineDataset`/`StockDataset` without per-market column-name branching? — RESOLVED: deferred, out of scope per D-08
   - What we know: Today it returns raw, un-renamed columns (Pattern 4).
   - What's unclear: Whether this is worth the added abstraction for a single example factor in this phase.
   - Recommendation: Out of scope for this phase (D-08 only requires *one* example factor, and CONTEXT.md's Polars discussion never mentions this). Worth a one-line mention to the user/planner as a natural extension if more Polars factors are added in a future phase.

## Environment Availability

Skipped — no external services/network dependencies are introduced by this phase (both `KunQuant` and `polars` are already-installed local Python libraries; no new CLI tools, databases, or network calls).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.1.1 (already installed, `[dependency-groups] dev` in `pyproject.toml`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (`pythonpath = ["."]`) — already present, reusable as-is |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -q` |

**Reusable from Phase 2:** `tests/conftest.py`'s `binance_csv_rows`/`write_binance_csv` fixtures (for building a synthetic `SpotKlineDataset` on disk without real market data) are directly reusable for constructing `FactorKunQuant`/`FactorPolars` test fixtures — no new fixture infrastructure needed for the crypto-market test paths. A new, small `_write_stock_pqt`-style helper (already exists as a private helper in `tests/test_stock_dataset.py` — can be promoted to `conftest.py` or duplicated) covers the `StockDataset`/`Alpha158Stock` test path.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FACTOR-01 | `Alpha158SpotKline.cal()` produces `xr.Dataset` with expected factor columns, unchanged after refactor | unit/regression | `uv run pytest tests/test_factor_kunquant.py -k spot -x` | ❌ Wave 0 |
| FACTOR-01 | `Alpha158Stock.cal()` (new) produces `xr.Dataset`, `amount` proxy computed correctly, no `RuntimeError` | unit | `uv run pytest tests/test_factor_kunquant.py -k stock -x` | ❌ Wave 0 |
| FACTOR-01 (regression) | `Alpha101Stock.cal()` (bugfix) no longer raises `RuntimeError: Bad inputs` | regression (locks in Pitfall 1's fix) | `uv run pytest tests/test_factor_kunquant.py -k alpha101_stock_bugfix -x` | ❌ Wave 0 |
| FACTOR-02 | `cal_stream()` batch-replay smoke test: no exception, correct output shape, across a full historical replay | unit/smoke | `uv run pytest tests/test_factor_stream.py -x` | ❌ Wave 0 |
| FACTOR-03 | `Momentum(FactorPolars).cal()` produces `xr.Dataset` with exactly the dynamically-resolved factor column(s), no raw price passthrough | unit | `uv run pytest tests/test_factor_polars.py -x` | ❌ Wave 0 |
| FACTOR-03 | `_get_factor_lazyframe()` never calls `.collect()` internally (stays lazy until `cal()`) | unit (introspection: assert `_get_factor_lazyframe()`'s return type is `pl.LazyFrame`, not `pl.DataFrame`; optionally AST-grep the method body for `.collect(` calls) | `uv run pytest tests/test_factor_polars.py -k laziness -x` | ❌ Wave 0 |
| FACTOR-04 | `Factor`/`FactorKunQuant`/`FactorPolars` public methods (`cal`, `get_features`, `get_labels`, `read`, `save`) never accept or return a bare `pandas.DataFrame`/`polars.DataFrame` (only `xr.Dataset` at the boundary, `pl.LazyFrame` only as an internal/intermediate type on `FactorPolars`) | unit (signature/type-annotation introspection via `inspect.signature`) | `uv run pytest tests/test_factor_hierarchy.py -k boundary_contract -x` | ❌ Wave 0 |
| D-03 interchangeability | `DLConfig(factors=[kunquant_instance, polars_instance], ...)` type-checks and `BaseModel._get_features_batch()` calls `.cal().get_features()` uniformly on both without `isinstance` branching in `base/model.py` | integration | `uv run pytest tests/test_factor_hierarchy.py -k interchangeability -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_factor_kunquant.py` — covers FACTOR-01 (both markets) + the `Alpha101Stock` regression fix
- [ ] `tests/test_factor_stream.py` — covers FACTOR-02 (batch-replay smoke test)
- [ ] `tests/test_factor_polars.py` — covers FACTOR-03 (`Momentum` example + laziness contract)
- [ ] `tests/test_factor_hierarchy.py` — covers FACTOR-04 (boundary-contract introspection) + D-03 interchangeability
- [ ] A small `_write_stock_pqt`-equivalent helper promoted to `tests/conftest.py` (currently a private helper duplicated in `tests/test_stock_dataset.py`) so the new `Alpha158Stock`/`Alpha101Stock` tests don't re-duplicate it a third time

*(No framework install needed — pytest, polars, and KunQuant are all already present.)*

## Security Domain

This phase introduces no new credentials, no new network calls, and no new external service integrations — it is pure in-process refactoring plus local compute (KunQuant compiled-graph execution, Polars lazy-expression execution) over already-local Zarr-backed data. No ASVS category applies beyond what Phase 1/2 already established (no secrets touched, no new attack surface). **Security gate: N/A for this phase — confirmed by design, not by omission.**

## Sources

### Primary (HIGH confidence — direct source reads + executed code)
- `base/factor.py`, `factor/alpha101.py`, `factor/alpha158.py`, `label/spot.py`, `base/config.py`, `base/model.py`, `base/data.py`, `dataset/stock.py`, `dataset/spot.py`, `dataset/backend.py`, `dataset/cleaning.py`, `config/__init__.py`, `my_ops/preprocess.py`, `utils/module.py`, `cal.py`, `train_model.py`, `test.py`, `backtest/test_strategy.py`, `test_nt.ipynb` (full reads, this repo)
- `/Users/daizhaorong/.venv/lib/python3.13/site-packages/KunQuant/predefined/Alpha101.py`, `Alpha158.py`, `KunQuant/Op.py` (full/targeted reads, installed package)
- Executed: `AllData(...)` construction without `amount` → `RuntimeError` (installed KunQuant 0.1.11)
- Executed: `pl.LazyFrame(...).collect_schema().names()` (installed polars 1.44.1)
- `pyproject.toml`, `uv.lock`, `importlib.metadata.version()` for `kunquant`/`polars`/`pytest`/`xarray`/`zarr`
- `.planning/phases/02-multi-market-data-foundation/tests/conftest.py`, `tests/test_stock_dataset.py`, `tests/test_spot_dataset.py` (existing test infrastructure/conventions)

### Secondary (MEDIUM confidence — WebFetch, cross-verified against installed package behavior where possible)
- GitHub `Menooker/KunQuant` repo structure (`doc/Stream.md`, `tests/test_stream.py` existence confirmed via GitHub API `contents` listing) — streaming API shape (`StreamContext`, `queryBufferHandle`, `pushData`, `run`, `getCurrentBuffer`) cross-verified against this repo's own `base/factor.py:cal_stream()`/`init_stream()` implementation, which already matches this shape exactly.

### Tertiary (LOW confidence — flagged in Assumptions Log)
- The "multiple of 8" `num_stock` requirement for STREAM layout being a hard correctness constraint vs. a soft performance suggestion (A1 in Assumptions Log).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; both `KunQuant`/`polars` versions confirmed installed and pinned.
- Architecture (class hierarchy): HIGH — verified against every existing subclass and call site in the repo (Integration Verification table).
- Pitfalls: HIGH for Pitfalls 1/2/4/5 (verified by direct code execution or exhaustive grep); MEDIUM for Pitfall 3 (external doc + repo-structure confirmation, not executed against a real multi-symbol stream in this session).

**Research date:** 2026-09-05
**Valid until:** 30 days (stable, local-only dependencies; no fast-moving external API surface in this phase)
