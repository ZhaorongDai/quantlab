# Phase 2: Multi-Market Data Foundation - Context

**Gathered:** 2026-09-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can ingest and store market data for multiple markets/frequencies through one extensible `Dataset`/`DataBackend` abstraction, with all data landing in canonical `xarray`/Zarr storage. v1 delivers two working sources — US equities daily (Tiingo) and Binance spot klines — through a config schema that is designed from the start to add new markets/frequencies without touching factor/model/backtest code. This phase does NOT need to actually implement a second frequency or a new Binance downloader to prove extensibility — that is validated by config/architecture design review (ROADMAP Phase 2 Success Criterion 4), not by building extra sources.

</domain>

<decisions>
## Implementation Decisions

### Storage layout / config schema
- **D-01:** `DatasetConfig` (`base/config.py`) gets explicit `market` and `frequency` fields (e.g. `market: Literal["us_equity", "crypto_spot"]`, `frequency: Literal["1d", "1m", "tick"]` — exact literal set is the planner's call). These become first-class, not implicit in hardcoded path strings.
- **D-02:** Storage path convention: `data/{market}/{frequency}/{name}.zarr` (and the equivalent for `raw_data_dir_path`/`catalog_path`). Every config factory function (`config/__init__.py`) must derive its paths from `market`+`frequency`, not hardcode market-specific segments like today's `spot/monthly/klines`.
- **Why this matters:** today's `spot_kline_config()`/`alpha101_config()`/etc. hardcode paths per-market with no `market`/`frequency` concept at all — this is exactly the gap ROADMAP Phase 2 Success Criterion 4 ("adding a new market or frequency only requires a new `Dataset` subclass + config") needs to close.

### Binance data source — scope for this phase
- **D-03:** Do NOT write a new Binance downloader (no calls to `data.binance.vision` or the Binance REST klines API) in this phase. The user's explicit priority is completing the **US equities end-to-end pipeline first**; Binance keeps its current manual-CSV-drop workflow (`SpotKlineDataset._raw_data_to_xr` reading pre-downloaded CSVs from `raw_data_dir_path`).
- **D-04:** `SpotKlineDataset`/`spot_kline_config()` still gets retrofitted into the new `market`/`frequency` config schema (D-01/D-02) for consistency — this is a config/path change only, not new ingestion logic.
- **Deferred:** an actual Binance auto-downloader is a future-phase/backlog item, not Phase 2 scope.

### Cleaning / preprocessing module (shared, reused by both sources per ROADMAP Success Criterion 3)
The module must handle, for v1:
- **D-05:** Deduplicate duplicate/overlapping `(symbol, timestamp)` rows — keep one deterministic record (planner decides first-vs-last, but must be deterministic and documented).
- **D-06:** Missing timestamps / non-trading-day gaps — leave as `NaN` in the aligned time grid. Explicitly do **NOT** forward-fill or otherwise impute, since that would introduce a look-ahead-free but still fabricated "low-risk" illusion the user wants to avoid.
- **D-07:** Anomalous price/volume values (e.g. zero/negative price, extreme jumps) are **flagged, not deleted or corrected** — e.g. an added flag column or a logged warning — so real anomalies aren't silently hidden or auto-"fixed" in a way that could mask data quality issues.
- **D-08:** Basic schema/type/non-null validation (required columns present, correct dtypes, no unexpected nulls in key columns) is in scope for v1; deeper statistical anomaly detection beyond D-07's flagging is explicitly out of scope for v1 (candidate for a later phase).
- **Why:** today's `my_ops/preprocess.py` only has factor-normalization ops (`WindowedZScore` etc.) — there is no raw-market-data cleaning logic anywhere yet. This is new code, not a refactor.

### US equities (Tiingo) ingestion — priority path for this phase
- **D-09:** No `stock_kline_config()`-equivalent factory function currently exists in `config/__init__.py` (only spot/alpha101/alpha158/label factories exist) — Phase 2 must add one, following the `market`/`frequency` schema (D-01/D-02), analogous to `spot_kline_config()`.
- **D-10 (revised — see below and [[feedback-encapsulate-new-components]] memory):** New Tiingo acquisition code must be a **properly encapsulated component following the codebase's established OOP/layered style** — a class (or small set of classes) analogous in spirit to `Dataset`/`FactorKunQuant` (ABC + concrete implementation, config-driven, single-responsibility methods) — NOT a flat procedural top-level script.
  - **User's explicit correction:** do NOT model it on `get_binance_instruments.py` or `cal.py` either — those are flat, single-purpose, run-once scripts. Even though they're "real" top-level entry points (not `scripts/` throwaways), their procedural style is still the wrong reference for this component. The correct reference is the codebase's class-based layered pattern (`base/data.py:Dataset`, `dataset/stock.py:StockDataset`, `dataset/spot.py:SpotKlineDataset`), not any existing script regardless of location.
  - **Must be designed for, from the start (even though only daily-US-equity is implemented in v1):**
    - **Multi-frequency support:** the acquisition component must be parameterized by frequency (not hardcode `frequency="daily"` deep in a function body the way `scripts/download_stock_data_from_tiingo.py` does today) so adding minute/intraday frequency later is a parameter/config change, not a rewrite.
    - **Incremental / daily-refresh updates:** must support fetching only new data since the last successful download per symbol (a "resume from last watermark" refresh), not just one-shot full-history backfill. Exact mechanism (a small state/metadata file recording last-downloaded date per symbol+frequency, or deriving it from the existing Zarr store's max timestamp) is the planner/researcher's call — but the capability itself is a locked requirement, not optional/future.
  - The existing `scripts/download_stock_data_from_tiingo.py` remains a reference for WHAT it does (Tiingo API call shape, JSON→columns mapping) but not HOW it's structured (per the original temp-vs-core-code guardrail, [[feedback-temp-vs-core-code]]) — this now also explicitly rules out copying the flat-script *style* even from non-`scripts/` files.
  - Output still lands via the existing `Dataset`/`DataBackend` abstraction into the `data/{market}/{frequency}/...` path convention (D-01/D-02) — this component handles acquisition (remote fetch → local raw files or direct hand-off), `StockDataset` still handles raw-files → xarray conversion, keeping the existing separation of concerns.

### Real-data verification for Binance ingestion (follow-up correction)
- **D-11:** This development machine currently has no Binance kline data downloaded at all (superseding the earlier assumption in D-03/D-10's discussion that "existing" local CSVs were simply outside the repo convention — there are none yet). To verify `ingest_binance_spot.py`/`SpotKlineDataset` against real (not just synthetic-fixture) data, use the user's own pre-existing, separate `binance-data-downloader` CLI tool (`/Users/daizhaorong/projects/binance-data-downloader`, installable/runnable via `uvx binance-data-downloader` — a real PyPI-shaped external project, not new code written for quantlab) to fetch a small real sample (e.g. one month of `BTCUSDT` `1d` klines) directly into the default convention path `{QUANTLAB_DATA_DIR}/downloads/crypto_spot/1d/spot/monthly/klines/`.
- **Why this doesn't violate D-03:** D-03 rules out writing NEW download code inside quantlab. Invoking an existing, already-built, separate third-party CLI tool as a one-time manual data-acquisition step is not new quantlab code — it's the moral equivalent of a user manually placing CSVs into the directory, just via a tool instead of a browser download.
- This real download should land in Wave 3 (02-05), as a task that fetches the sample data and then runs `ingest_binance_spot.py` against it, confirming a real `xr.Dataset` with plausible values gets persisted to Zarr — this supersedes/complements the earlier "point `--raw-data-dir` at wherever your CSVs already live" verification approach (that override flag stays as a useful general feature, but the phase's own verification no longer depends on the user having pre-existing data at some undocumented location).

### Claude's Discretion
- Exact `Literal` value sets for `market`/`frequency` fields.
- Whether cleaning logic lives in `my_ops/` (extending the existing package) or a new sibling package (e.g. `dataset/cleaning.py`) — planner picks based on where it best fits the existing layered architecture.
- Exact dedup tie-breaking rule (first vs. last duplicate wins) as long as it's deterministic and documented.
- Exact class/module design for the encapsulated Tiingo acquisition component (e.g. new `acquisition/` package, a method added to `StockDataset`, or a standalone class elsewhere) and exact incremental-refresh watermark mechanism — as long as it satisfies D-10's encapsulation + multi-frequency + incremental-update requirements and does not copy any existing script's flat/procedural style.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level constraints
- `.planning/PROJECT.md` — Core Value, Constraints (xarray/Zarr-only pipeline format, credential env-var-only, config-driven reproducibility), Key Decisions table, and the "代码分级" (temp-vs-core code) subsection
- `.planning/ROADMAP.md` §Phase 2 — Goal, Requirements (DATA-01..04), Success Criteria
- `.planning/REQUIREMENTS.md` — DATA-01..04 full requirement text

### Existing architecture to extend, not replace
- `base/config.py` — `DatasetConfig` dataclass (gets `market`/`frequency` fields per D-01)
- `base/data.py` — abstract `Dataset` (read/save/from_raw_data/_raw_data_to_xr contract)
- `base/backend.py`, `dataset/backend.py` — `DataBackend`/`XrBackend`/`PlBackend` (Zarr/xarray persistence contract — do not bypass)
- `dataset/stock.py` — `StockDataset` (Tiingo/NASDAQ parquet ingestion — currently has 3 unimplemented Nautilus methods, out of scope for this phase unless blocking)
- `dataset/spot.py` — `SpotKlineDataset` (Binance CSV ingestion — reference for the retrofit in D-04)
- `config/__init__.py` — existing factory function pattern (`spot_kline_config`, `alpha101_config`, `alpha158_config`, `spot_label_config`, `_data_root()` helper) to follow for the new `stock_kline_config()`-equivalent (D-09)
- `scripts/download_stock_data_from_tiingo.py` — reference for WHAT it does (Tiingo API usage, JSON→column mapping), explicitly NOT for HOW it's structured (see D-10)
- `base/data.py:Dataset`, `dataset/stock.py:StockDataset`, `dataset/spot.py:SpotKlineDataset` — reference for the class-based, encapsulated, config-driven style the new Tiingo acquisition component must follow (see D-10; `get_binance_instruments.py`/`cal.py` are explicitly NOT the right style reference despite being non-throwaway top-level scripts)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `base/data.py:Dataset` — read/save/from_raw_data lifecycle already works end-to-end for both existing sources; new sources plug in via subclassing, not modification
- `dataset/backend.py:XrBackend` — Zarr read/write already implemented and market/frequency-agnostic (paths are the only thing that need to change)
- `utils/file.py:get_csv_files`/`get_pqt_files`/`file_date_filter` — generic file-discovery helpers already reusable across both sources

### Established Patterns
- Every `Dataset` subclass only converts **already-local raw files** into xarray — none of them fetch remote data themselves today. This separation of concerns (acquisition vs. conversion) should hold, but per the user's correction, the acquisition side itself must still be built as an encapsulated, class-based component (D-10) — not a flat script like `get_binance_instruments.py` (which only refreshes YAML metadata, and is procedural style regardless).
- Config objects are dataclasses with factory functions in `config/__init__.py`, not inline construction at call sites — new sources should add a factory function, matching existing style.

### Integration Points
- `StockDataset` currently has no matching config factory in `config/__init__.py` — this is a genuine gap (D-09), not a design choice to preserve.
- `test.py` currently hand-builds a `DatasetConfig` for `StockDataset` pointing at `{QUANTLAB_DATA_DIR}/scripts/downloads/nasdaq_data` — this ad hoc wiring should be replaced by the new factory function once it exists, but `test.py` itself is flagged CLEAN-02/QUAL-02 cleanup scope (Phase 7), not this phase's concern beyond not breaking it further.

</code_context>

<specifics>
## Specific Ideas

- User's exact words on priority: "先使用已有的币安数据，目前最紧迫的目标是完成美股的全流程" (use existing Binance data as-is; the most urgent goal right now is completing the US-equities full pipeline).
- User's exact words on cleaning scope: missing timestamp/non-trading-day handling, duplicate/overlapping timestamp dedup, anomalous price/volume detection (flag, don't delete), and basic validation — all four selected together, no further prioritization given (all four are in scope for v1).

</specifics>

<deferred>
## Deferred Ideas

- **Binance automatic downloader** (calling `data.binance.vision` bulk data or the Binance REST klines API) — explicitly deferred; current manual CSV workflow stays as-is for this phase (D-03).
- **Statistical/ML-based anomaly detection** beyond simple flagging (D-07) — deferred to a later data-quality phase if ever needed.
- **Tick-level and minute-frequency production ingestion** for both markets — already deferred to v2 in `.planning/PROJECT.md` (Out of Scope) and `.planning/REQUIREMENTS.md` (DATA-V2-01/02); Phase 2 only needs to prove the config schema *could* support them, not implement them.

None — discussion stayed within phase scope beyond the above.

</deferred>

---

*Phase: 02-multi-market-data-foundation*
*Context gathered: 2026-09-05*
