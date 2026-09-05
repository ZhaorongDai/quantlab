# Phase 3: Factor Computation (KunQuant + Polars) - Context

**Gathered:** 2026-09-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can compute the existing Alpha158 factor set (batch and streaming) via KunQuant across BOTH markets (crypto spot and US equities), and can add new factors via a new Polars batch backend that is a drop-in-compatible sibling to the existing KunQuant factor engine — `xarray.Dataset` remains the sole exchange format at the pipeline boundary either way.

</domain>

<decisions>
## Implementation Decisions

### Alpha158 market coverage
- **D-01:** Extend Alpha158 to US equities by adding `Alpha158Stock` (mirroring the existing `Alpha101SpotKline`/`Alpha101Stock` pattern — same `FactorKunQuant` invocation shape per market, just a different `AllData`/data-columns wiring). Per the user: "alpha158拓展到美股非常简单，调用的步骤完全一致" — extending Alpha158 to US equities is straightforward, the invocation steps are identical to the existing Alpha101 pattern. Both Alpha101 and Alpha158 should work against both crypto spot and US equities.
- **Note (Claude's discretion, flagged by user but not blocking):** some Alpha158 factors may be cross-sectional (normalized across symbols) — whether/how cross-sectional factors need different handling per market is left to the planner/researcher to determine; the user does not consider this a blocker.
- **D-02:** `Alpha158Stock` needs an `amount` (dollar-volume) input for its VWAP-related features, which Tiingo data does not provide natively (unlike Binance's native `amount`/quote-volume column). Approximate it as `volume * close` (a standard dollar-volume proxy) rather than skipping VWAP-dependent features.

### New Polars factor engine (major architecture addition)
- **D-03:** Extract a shared abstract base class from the existing `FactorKunQuant` (`base/factor.py`) — call it `Factor` (exact name is the planner's call) — covering everything market/backend-agnostic (config lifecycle, `read()`/`save()`, `get_features()`/`get_labels()`, `get_xarray_dataset()`, symbol/date bookkeeping). `FactorKunQuant` becomes a subclass of this new shared base (refactor, not a rewrite — behavior must not change for existing KunQuant factor classes). A new `FactorPolars(Factor)` ABC becomes a sibling implementation using Polars instead of KunQuant's compiled op-graph.
- **User's explicit requirement:** "我需要这两个因子类可以无缝替换" — `FactorKunQuant`-based and `FactorPolars`-based factor objects must be seamlessly interchangeable wherever a factor object is consumed today (e.g. `base/config.py`'s `DLConfig.factors`/`DLConfig.labels`/`MLConfig.factors`/`MLConfig.labels`, currently typed `list["FactorKunQuant"]` — this blast radius must be updated to the new shared `Factor` type so both backends type-check and behave identically from the consumer's point of view).
- **D-04 (Polars factor contract):** Analogous to `FactorKunQuant`'s abstract `_get_factor_func()` (which returns a KunQuant `Function`), `FactorPolars` gets an abstract method (e.g. `_get_factor_lazyframe()`) that the user overrides to write factor logic. It must return a `pl.LazyFrame` containing ONLY `timestamp`, `symbol`, and the computed factor value column(s) — no raw price/volume passthrough columns. Per the user: "用户在类似 `_get_factor_func()` 的函数中编写因子，返回一个只有 symbol date 因子列的 polars lazyframe，然后当用户调用 `cal()` 的时候开始计算" — the lazyframe must stay lazy (uncomputed) until `cal()` is actually invoked, matching KunQuant's `cal()` semantics of "trigger computation now."
- **D-05 (dynamic factor-name resolution):** Unlike KunQuant (which needs factor names declared upfront via `_get_factor_names()` for stream-buffer wiring), the Polars path has no such requirement. Per the user: "Polars 不会返回 Function() 和因子名称，所以要做针对性的设计，比如提取 polars dataframe 的属于因子的列名" — factor names for the Polars path are derived dynamically from the lazyframe's own schema (all column names except `timestamp`/`symbol`), not declared separately.
- **D-06 (pipeline-format boundary preserved):** `cal()` on `FactorPolars` triggers `.collect()` to materialize the lazyframe, then the result is converted to `xr.Dataset` before being handed off or persisted — Polars is used internally for the computation, but the module boundary contract (xarray/Zarr only between pipeline modules, per project-wide constraint) is unchanged. This mirrors the existing pattern already used elsewhere in the codebase (e.g. `dataset/cleaning.py` uses polars internally but the `Dataset` layer's public interface is xarray).
- **D-07 (batch-only, no streaming):** Reaffirms the original project-level decision — the Polars backend has no streaming/`cal_stream()` equivalent; it is batch-only.
- **D-08 (example factor):** Ship one simple example factor in `factor/` demonstrating the new `FactorPolars` pattern end-to-end (not a factor already covered by Alpha101/158). Exact formula is Claude's/the planner's discretion — the user's own answer focused entirely on the architecture/contract, not a specific factor formula, so a simple, well-understood factor (e.g. N-day price momentum, or a simple turnover/volume-based signal) is an acceptable default as long as it's genuinely computed via Polars lazyframe operations, not a trivial pass-through.

### Factor normalization axis (D-09 — added 2026-09-05, supersedes part of D-01)

- **D-09 (LOCKED, user's direct ruling):** `WindowedZScore`/`WindowedRobustStandardization` (`my_ops/preprocess.py`, both `WindowedCompositiveOp`) are **time-series-strategy** (时序策略) normalization — they standardize each symbol against its own rolling window. **Cross-sectional strategies** (截面策略) instead need a cross-sectional Z-score (standardize across symbols at each timestamp). The presence or absence of `WindowedZScore` in a factor class is therefore a **strategy-type design choice, never a scale bug to "fix" by aligning two classes.**
- **US equities use cross-sectional strategies.** The user's words: "美股都用截面策略，Alpha158Stock 也不要时序标准化". Both US-equity factor classes emit **raw, un-normalized** factor values:

  | Factor class | Market | Normalization |
  |---|---|---|
  | `Alpha101SpotKline` | crypto spot | `WindowedZScore` (time-series) |
  | `Alpha158SpotKline` | crypto spot | `WindowedZScore` (time-series) |
  | `Alpha101Stock` | US equities | none — raw (cross-sectional strategy) |
  | `Alpha158Stock` (new) | US equities | none — raw (cross-sectional strategy) |

- **This supersedes part of D-01.** D-01's "mirror `Alpha158SpotKline`" applies to the `AllData` wiring and factor-set construction only — NOT to the normalization wrapper. `Alpha158Stock` must emit `Output(v, k)` directly, not `Output(WindowedZScore(v, window), k)`.
- **`Alpha101Stock`'s only genuine defect** remains the missing `amount` input causing the verified `RuntimeError: Bad inputs` crash (D-02's `volume * close` proxy fixes it). Its lack of `WindowedZScore` is correct-by-design, not a bug.
- **Cross-sectional Z-score op is deferred, not forgotten.** `my_ops/` currently has only time-series ops; KunQuant upstream supplies `CrossSectionalOp`, `SimpleCrossSectionalOp`, `Rank`, `Scale` as building blocks. Building an actual cross-sectional normalization op belongs with the ARCH-01/ARCH-02 work in Phase 6 ("架构同时兼容单标的时序策略与多标的截面多因子策略"), not this phase — this phase emits raw US-equity factor values and leaves normalization to the downstream consumer.

### Claude's Discretion
- Exact naming of the new shared `Factor` ABC and the new `FactorPolars`/`_get_factor_lazyframe()` method names (the user described the shape and contract, not literal identifiers).
- Whether/how Alpha158's cross-sectional-normalized factors need per-market handling differences (D-01's note).
- The exact formula for the one example Polars-computed factor (D-08).
- How `cal_stream()`/streaming-specific methods on the current `FactorKunQuant` (`init_stream`, `_make_stream`, `_stream_context`) are organized relative to the new shared `Factor` base — they are KunQuant-specific and should NOT be hoisted onto the shared base or required by `FactorPolars`, but the exact class-hierarchy placement is an implementation detail.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level constraints
- `.planning/PROJECT.md` — Core Value, Constraints (xarray/Zarr pipeline-format-only, config-driven reproducibility, dual factor-backend priority), Key Decisions table
- `.planning/ROADMAP.md` §Phase 3 — Goal, Requirements (FACTOR-01..04), Success Criteria
- `.planning/REQUIREMENTS.md` — FACTOR-01..04 full requirement text

### Existing architecture to extend, not replace
- `base/factor.py` — abstract `FactorKunQuant` (the class this phase refactors to extract a shared `Factor` base from — read in full, every method needs a market/backend-agnostic-vs-KunQuant-specific classification before refactoring)
- `factor/alpha101.py` — `Alpha101SpotKline` + `Alpha101Stock` (the existing dual-market pattern D-01 mirrors for Alpha158)
- `factor/alpha158.py` — `Alpha158SpotKline` (the class D-01's `Alpha158Stock` mirrors; note its `_get_func_names()`/`_get_func_stream()` structure differs slightly from Alpha101's `_get_factor_func()`)
- `my_ops/preprocess.py` — `WindowedZScore`/`WindowedRobustStandardization` (existing KunQuant composite ops used by both Alpha101/158)
- `base/config.py` — `FactorConfig`, and `DLConfig.factors`/`DLConfig.labels`/`MLConfig.factors`/`MLConfig.labels` (currently `list["FactorKunQuant"]` — blast radius for D-03's shared-base type update)
- `dataset/backend.py:PlBackend` — existing Polars/LazyFrame-based backend (reference for Polars conventions already established in the codebase)
- `dataset/cleaning.py` — existing precedent for "Polars internally, xarray at the pipeline boundary" (D-06's pattern)
- `config/__init__.py` — existing `alpha101_config()`/`alpha158_config()` factory pattern to extend for the new Stock variant and any new Polars factor config

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `base/factor.py:FactorKunQuant` — read/save/config lifecycle, `_auto_filter()`, `get_config()` are all market/backend-agnostic and are strong candidates to hoist onto the new shared `Factor` base as-is.
- `factor/alpha101.py:Alpha101Stock` — the concrete template for building `Alpha158Stock` (same `AllData` wiring pattern, different KunQuant predefined module).

### Established Patterns
- Every existing factor class subclasses `FactorKunQuant` directly and implements `_get_factor_func()`/`_get_factor_names()`/`_get_labels()`/`_get_features()` — the new `FactorPolars` should offer an equivalent-shaped contract so switching between backends is a subclass-swap, not a caller-code change.
- `DatasetConfig`/`FactorConfig` are plain dataclasses threaded through constructors — no reason to change this pattern for a new `PolarsFactorConfig` if one is needed, though `FactorConfig` itself might already be reusable as-is (needs research to confirm which fields are KunQuant-specific, e.g. `njobs`, `mode: Literal["stream","batch"]` which wouldn't apply to Polars).

### Integration Points
- `base/model.py:BaseModel` consumes `DLConfig.factors`/`.labels` — after D-03's refactor, this must keep working unmodified with either backend's factor objects (this is the actual test of "seamless interchangeability," not just a type-hint change).

</code_context>

<specifics>
## Specific Ideas

- User's exact words on the interchangeability requirement: "我需要这两个因子类可以无缝替换" (I need these two factor classes to be seamlessly interchangeable).
- User's exact words on the Polars contract shape: "实现因子类后，在 factor 文件夹中加一个简单的 polars 计算因子。注意，Polars 不会返回Function()和因子名称，所以要做针对性的设计，比如提取 polars dataframe 的属于因子的列名。polars 因子计算要使用 lazyframe，我的想法是用户在类似 `_get_factor_func()` 的函数中编写因子，返回一个只有 symbol date 因子列的 polars lazyframe，然后当用户调用 `cal()`的时候开始计算。"

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-factor-computation-kunquant-polars*
*Context gathered: 2026-09-05*
