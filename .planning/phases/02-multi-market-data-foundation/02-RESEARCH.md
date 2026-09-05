# Phase 2: Multi-Market Data Foundation - Research

**Researched:** 2026-09-04
**Domain:** Multi-market/multi-frequency data ingestion into a shared xarray/Zarr storage abstraction (Tiingo US-equity acquisition + Binance spot retrofit + shared cleaning module)
**Confidence:** MEDIUM-HIGH (architecture/codebase findings are VERIFIED by direct file reads and empirical tests; one external-API behavior — Tiingo's default JSON field set — is ASSUMED and flagged for live verification)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Storage layout / config schema**
- **D-01:** `DatasetConfig` (`base/config.py`) gets explicit `market` and `frequency` fields (e.g. `market: Literal["us_equity", "crypto_spot"]`, `frequency: Literal["1d", "1m", "tick"]` — exact literal set is the planner's call). These become first-class, not implicit in hardcoded path strings.
- **D-02:** Storage path convention: `data/{market}/{frequency}/{name}.zarr` (and the equivalent for `raw_data_dir_path`/`catalog_path`). Every config factory function (`config/__init__.py`) must derive its paths from `market`+`frequency`, not hardcode market-specific segments like today's `spot/monthly/klines`.
- Why this matters: today's `spot_kline_config()`/`alpha101_config()`/etc. hardcode paths per-market with no `market`/`frequency` concept at all — this is exactly the gap ROADMAP Phase 2 Success Criterion 4 ("adding a new market or frequency only requires a new `Dataset` subclass + config") needs to close.

**Binance data source — scope for this phase**
- **D-03:** Do NOT write a new Binance downloader (no calls to `data.binance.vision` or the Binance REST klines API) in this phase. Binance keeps its current manual-CSV-drop workflow (`SpotKlineDataset._raw_data_to_xr` reading pre-downloaded CSVs from `raw_data_dir_path`).
- **D-04:** `SpotKlineDataset`/`spot_kline_config()` still gets retrofitted into the new `market`/`frequency` config schema (D-01/D-02) for consistency — this is a config/path change only, not new ingestion logic.
- Deferred: an actual Binance auto-downloader is a future-phase/backlog item, not Phase 2 scope.

**Cleaning / preprocessing module (shared, reused by both sources per ROADMAP Success Criterion 3)**
- **D-05:** Deduplicate duplicate/overlapping `(symbol, timestamp)` rows — keep one deterministic record (planner decides first-vs-last, but must be deterministic and documented).
- **D-06:** Missing timestamps / non-trading-day gaps — leave as `NaN` in the aligned time grid. Explicitly do NOT forward-fill or otherwise impute.
- **D-07:** Anomalous price/volume values (e.g. zero/negative price, extreme jumps) are flagged, not deleted or corrected.
- **D-08:** Basic schema/type/non-null validation (required columns present, correct dtypes, no unexpected nulls in key columns) is in scope for v1; deeper statistical anomaly detection beyond D-07's flagging is out of scope for v1.
- Why: today's `my_ops/preprocess.py` only has factor-normalization ops (`WindowedZScore` etc.) — there is no raw-market-data cleaning logic anywhere yet. This is new code, not a refactor.

**US equities (Tiingo) ingestion — priority path for this phase**
- **D-09:** No `stock_kline_config()`-equivalent factory function currently exists in `config/__init__.py` (only spot/alpha101/alpha158/label factories exist) — Phase 2 must add one, following the `market`/`frequency` schema (D-01/D-02), analogous to `spot_kline_config()`.
- **D-10 (revised):** New Tiingo acquisition code must be a properly encapsulated component following the codebase's established OOP/layered style — a class (or small set of classes) analogous in spirit to `Dataset`/`FactorKunQuant` (ABC + concrete implementation, config-driven, single-responsibility methods) — NOT a flat procedural top-level script. Explicitly do NOT model it on `get_binance_instruments.py` or `cal.py` either.
  - Must be designed for, from the start: multi-frequency support (parameterized by frequency, not hardcoded) and incremental/daily-refresh updates (resume-from-last-watermark, not just one-shot full backfill). Exact watermark mechanism is the researcher/planner's call.
  - `scripts/download_stock_data_from_tiingo.py` remains a reference for WHAT it does (Tiingo API call shape, JSON→columns mapping), not HOW it's structured.
  - Output still lands via the existing `Dataset`/`DataBackend` abstraction into the `data/{market}/{frequency}/...` path convention — acquisition handles remote fetch → local raw files, `StockDataset` still handles raw-files → xarray conversion.

### Claude's Discretion
- Exact `Literal` value sets for `market`/`frequency` fields.
- Whether cleaning logic lives in `my_ops/` (extending the existing package) or a new sibling package (e.g. `dataset/cleaning.py`) — planner picks based on where it best fits the existing layered architecture.
- Exact dedup tie-breaking rule (first vs. last duplicate wins) as long as it's deterministic and documented.
- Exact class/module design for the encapsulated Tiingo acquisition component and exact incremental-refresh watermark mechanism — as long as it satisfies D-10's encapsulation + multi-frequency + incremental-update requirements and does not copy any existing script's flat/procedural style.

### Deferred Ideas (OUT OF SCOPE)
- Binance automatic downloader (calling `data.binance.vision` bulk data or the Binance REST klines API) — current manual CSV workflow stays as-is for this phase (D-03).
- Statistical/ML-based anomaly detection beyond simple flagging (D-07) — deferred to a later data-quality phase if ever needed.
- Tick-level and minute-frequency production ingestion for both markets — already deferred to v2 (`DATA-V2-01`/`DATA-V2-02`). Phase 2 only needs to prove the config schema *could* support them.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | 用户可以从 Tiingo 拉取美股日频行情数据并写入 xarray/Zarr 存储 | New `TiingoAcquisition` component (see Architecture Patterns) + new `stock_kline_config()` factory (D-09) + verified Tiingo client field mapping (Code Examples) |
| DATA-02 | 用户可以复用/整理现有币安现货数据接入，写入同一套存储抽象 | `SpotKlineDataset`/`spot_kline_config()` retrofit to `market`/`frequency` schema (D-04); confirmed real on-disk raw file layout (`.../spot/monthly/klines/BTCUSDT/1d/...`) informs the retrofit's raw-path handling (see Runtime State Inventory) |
| DATA-03 | 数据层抽象在设计上支持按市场与频率扩展，新增市场/频率不需要改动上层因子/模型/回测代码 | `DatasetConfig.market`/`.frequency` fields (D-01) + path-derivation convention (D-02) + full call-site blast-radius audit (see Don't Hand-Roll / Common Pitfalls) proves no factor/model/backtest code touches these fields directly |
| DATA-04 | 提供数据清洗/预处理模块（复用现有 `my_ops` 标准化算子风格），输出统一的 `xarray.Dataset` | Shared cleaning module design (D-05..D-08) + empirically verified `to_xarray()` dedup/NaN behavior (see Common Pitfalls, Code Examples) |
</phase_requirements>

## Summary

Phase 2 extends an already-working `Dataset`/`DataBackend`/`XrBackend` abstraction (verified by direct read of `base/data.py`, `base/backend.py`, `dataset/backend.py` — all functional today) with two things that do not exist yet: (1) first-class `market`/`frequency` fields on `DatasetConfig` driving a uniform storage path convention, and (2) a genuinely new, encapsulated Tiingo acquisition component that can fetch and incrementally refresh US-equity daily data. Both the `Dataset` abstraction and the `XrBackend` Zarr read/write path are proven working code — this phase is additive/refactor, not a rewrite.

Three findings materially change how the plan should be scoped. First, empirically verified: `pandas.DataFrame.set_index([...]).to_xarray()` **raises `ValueError`** on a non-unique MultiIndex, and — when the index is unique — **automatically produces the full cartesian product with `NaN` for missing `(timestamp, symbol)` combinations**. This means D-06 ("leave missing timestamps as NaN, no forward-fill") is *already* the natural behavior of the existing `to_xarray()` conversion step, and D-05 (dedup) is not just a "nice to have" but a **hard prerequisite** — the pipeline will crash without it, not silently misbehave. Second, empirically verified: `XrBackend.write()` calls `xr.Dataset.to_zarr(path)` with no `mode` kwarg, and **re-running `save()` against an already-existing Zarr store raises `FileExistsError`** (Zarr's default `mode="w-"`). This directly blocks D-10's incremental-refresh requirement unless the plan explicitly threads `mode="w"` (full overwrite-from-rebuilt-raw-files, matching the existing "rebuild from all raw files on disk" pattern) through `save()`/`write()`. Third, real on-disk evidence shows Binance's actual raw CSVs (outside the repo, e.g. `~/Downloads/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2017-08.zip`) are daily-frequency (`1d`) and live in a directory layout that does not match the new `data/{market}/{frequency}/...` convention — the D-04 retrofit is a config path change, but the *actual raw files* are not migrated by this phase, which is an open operational question for the plan.

**Primary recommendation:** Add `market`/`frequency` as `Literal` fields to `DatasetConfig` with the exact tokens `market: Literal["us_equity", "crypto_spot"]` and `frequency: Literal["1d", "1m", "tick"]`; derive all four existing config-factory paths (plus the new `stock_kline_config()`) from these two fields; build a new `base/acquisition.py:Acquisition(ABC)` + `acquisition/tiingo.py:TiingoAcquisition` pair (mirroring the `Dataset`/`FactorKunQuant` ABC+concrete pattern) that fetches raw Tiingo JSON, writes local parquet files matching `StockDataset._raw_data_to_xr`'s expected schema, and tracks a small per-symbol watermark file for incremental refresh; add a new `dataset/cleaning.py` module invoked once, centrally, inside `Dataset.from_raw_data()` (not per-subclass) so every current and future `Dataset` subclass gets dedup + NaN-gap + anomaly-flagging + schema-validation "for free," satisfying Success Criterion 4 without any factor/model/backtest code changes.

## Architectural Responsibility Map

> This project has no browser/API/CDN tiers — it is a local batch research pipeline. The standard web-tier taxonomy has been adapted to this project's own layered architecture (`base/` ABCs → concrete implementations).

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Remote data fetch (Tiingo HTTP calls) | Acquisition (new `acquisition/` package) | — | New responsibility; must stay decoupled from `Dataset` per established "Dataset only converts already-local files" pattern (`base/data.py`) |
| Incremental-refresh watermark tracking | Acquisition | — | Watermark is acquisition-side state (what's been fetched), not dataset-side state (what's been converted/cleaned) |
| Raw file → `xr.Dataset` conversion | Dataset layer (`base/data.py` + subclasses) | — | Existing, proven contract (`_raw_data_to_xr`); new sources plug in via subclassing only |
| Shared cleaning (dedup/NaN-gaps/anomaly-flag/schema-check) | Dataset layer (base class hook) | Cleaning module (`dataset/cleaning.py` or `my_ops/`) | Must run for every subclass automatically (Success Criterion 3+4) — belongs in `Dataset.from_raw_data()`, not duplicated per subclass |
| Zarr/xarray persistence | Storage Backend (`dataset/backend.py:XrBackend`) | — | Already market/frequency-agnostic; only the *path string* passed in changes |
| Config schema (`market`/`frequency`, path derivation) | Config layer (`base/config.py` + `config/__init__.py` factories) | — | Single source of truth for path convention; factor/model/backtest code never sees market/frequency directly, only resolved paths |
| Credential handling (`TIINGO_API_KEY`) | Acquisition (env var read at client construction) | — | Must never be stored on `AcquisitionConfig`/`DatasetConfig` dataclasses (which get serialized via `to_dict()` into checkpoint JSON) |

## Standard Stack

### Core
| Library | Version (installed) | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `tiingo` | 0.16.1 [VERIFIED: installed in project venv via `uv run python -c "import tiingo"`] | Official Tiingo Python client (`TiingoClient.get_ticker_price`) | Already a `pyproject.toml` dependency (added in Phase 1); reference implementation `scripts/download_stock_data_from_tiingo.py` already uses it |
| `xarray` | 2026.7.0 (pyproject pin) [VERIFIED: `pyproject.toml`] | Canonical in-memory `[timestamp, symbol]` representation | Project-wide hard constraint (PROJECT.md) |
| `zarr` | 3.3.0 (pyproject pin) [VERIFIED: `pyproject.toml`] | On-disk persistence backend for `XrBackend` | Already wired via `xr.Dataset.to_zarr`/`xr.open_dataset` |
| `polars` | 1.44.1 (pyproject pin) [VERIFIED: `pyproject.toml`] | Raw file (CSV/parquet) ingestion before conversion to pandas/xarray | Existing pattern in both `StockDataset`/`SpotKlineDataset` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandas` | 3.0.5 (pyproject pin) | Intermediate `set_index([...]).to_xarray()` conversion step | Already the conversion mechanism both existing `Dataset` subclasses use |
| `loguru` | 0.7.3 | Logging (anomaly-flag warnings, watermark updates, acquisition progress) | Consistent with existing `Dataset`/`FactorKunQuant` logging style |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| A small JSON watermark sidecar file for incremental refresh | Deriving "last fetched date" from the max timestamp already in the Zarr store | Zarr-derived watermark couples the acquisition layer to the Dataset/Zarr layer it's supposed to stay decoupled from, and can silently mis-skip data if cleaning ever drops rows downstream of what was actually fetched (see Architecture Patterns) |
| Full-rebuild-and-overwrite Zarr write on every refresh (`mode="w"`) | True incremental Zarr append (`mode="a"`, `append_dim="timestamp"`, region writes) | Append mode requires careful `region`/coordinate-alignment handling when the `symbol` dimension can also grow (new tickers), which the current single-shot `_raw_data_to_xr()` (always rebuilds from *all* raw files on disk) doesn't need — matches existing pattern with far less risk for a v1 |

**Installation:** No new packages required — `tiingo`, `xarray`, `zarr`, `polars`, `pandas` are all already declared in `pyproject.toml` (Phase 1 fixed this) and confirmed importable in the project's `uv`-managed venv.

**Version verification performed:**
```bash
uv run python3 -c "import tiingo; print(tiingo.__file__)"   # resolves inside project venv
uv run python3 -c "import KunQuant; print(KunQuant.__file__)"  # resolves inside project venv
```
Both succeeded — the environment from Phase 1 is functional; no `uv sync` blockers for Phase 2.

## Package Legitimacy Audit

**No new external packages are introduced by this phase.** All required libraries (`tiingo`, `xarray`, `zarr`, `polars`, `pandas`) are already declared in `pyproject.toml` (added/fixed during Phase 1) and were confirmed importable directly from the project's `uv`-managed virtual environment during this research session. The Package Legitimacy Gate protocol (slopcheck, registry verification) is therefore not applicable — there is nothing new to vet.

If the planner decides to add a lightweight test-mocking library (e.g. `responses` or `pytest-mock` for mocking the Tiingo HTTP client in tests — see Validation Architecture), that package selection should re-trigger this gate at plan time.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────┐
                    │   Tiingo REST API        │
                    │ (tiingo/daily/{t}/prices)│
                    └────────────┬─────────────┘
                                 │ HTTPS (TIINGO_API_KEY from env)
                                 ▼
          ┌──────────────────────────────────────────┐
          │  acquisition/tiingo.py: TiingoAcquisition │   <- NEW component (D-10)
          │  - download(symbols, start, end, freq)    │
          │  - refresh(symbols)  [reads watermark]    │
          │  - writes raw parquet files                │
          │  - updates per-symbol watermark file       │
          └───────────────────┬────────────────────────┘
                               │ writes raw files to
                               ▼
          data/downloads/us_equity/1d/{symbol}/data.pqt   (raw_data_dir_path, D-02)
                               │
                               │ read by
                               ▼
          ┌──────────────────────────────────────────┐
          │  dataset/stock.py: StockDataset            │   <- EXISTING, retrofitted
          │  ._raw_data_to_xr()                        │      (market/frequency-aware
          │    polars concat -> pandas -> set_index     │       config paths only)
          │    -> .to_xarray()                          │
          └───────────────────┬────────────────────────┘
                               │ xr.Dataset [timestamp, symbol]
                               ▼
          ┌──────────────────────────────────────────┐
          │  base/data.py: Dataset.from_raw_data()     │   <- MODIFIED: inserts
          │    data = self._raw_data_to_xr()            │      shared cleaning step
          │    data = clean_market_data(data, ...)  NEW │      here (D-05..D-08)
          │    self.data_backend.to_internal(data)      │
          └───────────────────┬────────────────────────┘
                               │
                               ▼
          ┌──────────────────────────────────────────┐
          │ dataset/backend.py: XrBackend.write()      │   <- EXISTING (needs
          │   self.data.to_zarr(path, mode="w", ...)   │      explicit mode="w" fix,
          └───────────────────┬────────────────────────┘      see Common Pitfalls)
                               ▼
          data/us_equity/1d/stock_kline.zarr   (zarr_file_path, D-02)


   Parallel path (Binance, D-03/D-04 — config retrofit only, no new fetch logic):

          Pre-downloaded CSVs (manual, outside this phase's scope)
                               │
                               ▼
          data/downloads/crypto_spot/1d/spot/monthly/klines/{SYMBOL}/...
                               │ read by
                               ▼
          dataset/spot.py: SpotKlineDataset._raw_data_to_xr()  (unchanged logic)
                               │
                               ▼
          base/data.py: Dataset.from_raw_data()  (same shared cleaning step)
                               │
                               ▼
          data/crypto_spot/1d/spot_kline.zarr
```

A reader can trace both sources end-to-end: remote/manual raw data → source-specific `_raw_data_to_xr()` → shared cleaning (new, centralized) → `XrBackend` → Zarr. The cleaning step is the single new "waist" both sources pass through, which is exactly Success Criterion 3.

### Recommended Project Structure
```
acquisition/                  # NEW package — remote data fetch only
├── __init__.py                # empty, per existing convention (no re-exports)
└── tiingo.py                  # TiingoAcquisition(Acquisition)

base/
├── acquisition.py             # NEW — Acquisition(ABC): download()/refresh()/_watermark_path()
├── config.py                  # MODIFIED — DatasetConfig gets market/frequency; new AcquisitionConfig
└── data.py                    # MODIFIED — from_raw_data() calls clean_market_data()

dataset/
├── cleaning.py                # NEW — clean_market_data(), dedup/NaN-check/anomaly-flag/schema-validate
├── stock.py                   # unchanged conversion logic (only config paths change upstream)
└── spot.py                    # unchanged conversion logic (only config paths change upstream)

config/
└── __init__.py                 # MODIFIED — market/frequency-derived paths in all 5 factories
                                 # (spot_kline_config, alpha101_config, alpha158_config,
                                 #  spot_label_config, NEW stock_kline_config)

enums/
└── data.py                    # ADD Market/Frequency Literal aliases or keep as bare Literal on DatasetConfig
```

### Pattern 1: Centralized cleaning hook in `Dataset.from_raw_data()`
**What:** Insert the shared cleaning call in the base class, not in each subclass's `_raw_data_to_xr()`.
**When to use:** Whenever a cross-cutting step must apply to *every* current and future `Dataset` subclass without each subclass author needing to remember to call it — directly required by Success Criterion 4 ("new market/frequency needs only a `Dataset` subclass + config, no other code changes").
**Example:**
```python
# Source: base/data.py (existing method, modification shown)
def from_raw_data(self) -> Self:
    data = self._raw_data_to_xr()
    data = clean_market_data(data)          # NEW — shared, unconditional
    self.data_backend.to_internal(data)
    return self
```
**Why not per-subclass:** the existing established pattern note (CONTEXT.md code_context) states cleaning is "new code, not a refactor" and must be *reused* by both sources — putting the call in each subclass risks a third future subclass forgetting to call it, silently breaking Success Criterion 4.

### Pattern 2: Dedup must happen on the tabular (long-format) data BEFORE `set_index([...]).to_xarray()`
**What:** `pandas.DataFrame.set_index(["timestamp","symbol"]).to_xarray()` raises `ValueError: cannot convert a DataFrame with a non-unique MultiIndex into xarray` when duplicate `(timestamp, symbol)` rows exist.
**When to use:** Always — this is not optional for correctness, it is a hard crash today if either raw source ever has overlapping/duplicate rows (a documented real risk for Binance's monthly CSV files near month boundaries).
**Verified empirically in this session:**
```python
# Source: empirical test run via `uv run python3 -c "..."` in this research session
import pandas as pd
df = pd.DataFrame({
    "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
    "symbol": ["A", "A", "A"],
    "close": [1.0, 2.0, 3.0],
})
df.set_index(["timestamp", "symbol"]).to_xarray()
# -> ValueError: cannot convert a DataFrame with a non-unique MultiIndex into xarray
```
**Implication for design:** the "tabular pre-clean" (dedup specifically) must happen either (a) inside each subclass's `_raw_data_to_xr()` on the polars/pandas long-format data before the final `.to_xarray()` call, or (b) the cleaning module must expose a tabular-level dedup function that each `_raw_data_to_xr()` calls explicitly (since the base-class hook in Pattern 1 only sees the *already-converted* `xr.Dataset`, by which point dedup is too late — the crash already happened). **Recommendation:** split cleaning into two entry points — `dedup_raw_frame(df)` called by each subclass just before its `.to_xarray()` call (2 call sites to touch: `StockDataset._raw_data_to_xr`, `SpotKlineDataset._raw_data_to_xr`), and `clean_market_data(xr_ds)` (NaN-gap policy is a no-op/pass-through validation only, since `to_xarray()` already produces it for free; anomaly-flagging + schema-validation) called once centrally per Pattern 1.

### Pattern 3: Missing-timestamp NaN behavior is free from `to_xarray()`, given a unique index
**What:** Once duplicates are removed, `to_xarray()` on a `(timestamp, symbol)`-indexed DataFrame automatically produces the full cartesian product of both index levels, filling any `(timestamp, symbol)` combination absent from the source data with `NaN`.
**Verified empirically in this session:**
```python
# Source: empirical test run via `uv run python3 -c "..."` in this research session
import pandas as pd
df = pd.DataFrame({
    "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
    "symbol": ["A", "A", "B"],
    "close": [1.0, 2.0, 3.0],
})
ds = df.set_index(["timestamp", "symbol"]).to_xarray()
# ds["close"].values -> [[1., 3.], [2., nan]]   (2020-01-02/B is NaN — never forward-filled)
```
**Implication:** D-06 requires no new "reindex to full grid" code — the existing conversion mechanism already satisfies it. The cleaning module's job for D-06 is narrower than it sounds: *do not add forward-fill anywhere*, and optionally add a schema/sanity check that confirms no unexpected imputation happened upstream (e.g., assert no library call like `.ffill()`/`.interpolate()` exists in the raw-to-xr path).

### Anti-Patterns to Avoid
- **Deriving the incremental-refresh watermark from the Zarr store's max timestamp:** couples the acquisition layer (remote fetch) to the Dataset/Zarr layer (conversion+persistence) that D-10's own separation-of-concerns note says must stay decoupled; also breaks if a future cleaning step ever drops/filters rows so "last row in Zarr" no longer equals "last row successfully fetched."
- **Calling `.to_zarr(path)` without an explicit `mode` for a refresh/re-run workflow:** will raise `FileExistsError` (Zarr default `mode="w-"`) — verified empirically in this session (see Common Pitfalls).
- **Putting `download()`/`refresh()` methods on `Dataset`/`StockDataset` itself:** violates the established, explicitly-noted pattern that "every `Dataset` subclass only converts already-local raw files into xarray — none of them fetch remote data themselves today" (CONTEXT.md code_context, corroborated by reading `base/data.py`/`dataset/stock.py`/`dataset/spot.py` — no network calls anywhere in the `Dataset` hierarchy today).
- **Storing the raw Tiingo API key value as a dataclass field on any `*Config` object:** these configs are serialized via `to_dict()`/`asdict()` and written into checkpoint JSON (`base/model.py:_save_model`) and reconstructed via `utils/module.py:load_dataset_from_config` — a credential field would risk re-creating the exact class of leak that caused CLEAN-01/SEC-01 in Phase 1. Read `TIINGO_API_KEY` directly from `os.environ` inside `TiingoAcquisition.__init__`, never store it on a config object.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Full-grid time/symbol alignment with NaN for gaps (D-06) | A manual reindex-to-full-calendar-grid function | `pandas.DataFrame.set_index([...]).to_xarray()` (already used) | Empirically verified to already produce this behavior for free, given a unique index — see Pattern 3 |
| Tiingo HTTP retry/session handling | Custom `requests` session/retry logic | `tiingo.TiingoClient` (already a pinned dependency, already used in the reference script) | Client already handles session reuse (`config["session"] = True`) and endpoint routing (EOD vs IEX by frequency) |
| Zarr incremental append semantics | Hand-rolled append_dim/region logic on first pass | Full-rebuild-and-overwrite (`mode="w"`) matching the existing "always rebuild from all raw files on disk" pattern | The existing `_raw_data_to_xr()` for both sources already re-globs and re-reads *all* raw files every run (`get_pqt_files`/`get_csv_files` walk the whole directory) — true incremental Zarr writes would be inconsistent with that pattern and add region/coordinate-alignment complexity not needed for a v1 |

**Key insight:** Nearly everything D-05..D-08 asks for is either already a side effect of the existing `to_xarray()` conversion path (D-06) or a small, explicit, easily-testable function (dedup, anomaly-flag, schema-check) — the risk in this phase is not "needing to build something complex," it's forgetting that dedup must run *before* the `to_xarray()` call, not after.

## Runtime State Inventory

> Included because D-04 retrofits `SpotKlineDataset`/`spot_kline_config()` onto a new path convention — this is a config/path refactor of an existing, real (locally cached) data source, not pure greenfield.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | No pre-existing Zarr stores found inside the repo or under `QUANTLAB_DATA_DIR` (`find ... -iname "*.zarr"` returned nothing). **However**, real, already-downloaded Binance raw CSVs exist **outside** the project's data root: `~/Downloads/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2017-08.zip` (and neighboring months), and a sibling project `~/projects/binance-data-downloader/downloads/spot/daily/klines/`. These are not tracked by any config in this repo today. | No Zarr migration needed (none exist yet). Raw CSVs: planner must decide whether `stock_kline_config`/`spot_kline_config`'s new `raw_data_dir_path` should point directly at these external locations, or whether the user should copy/symlink them under `{QUANTLAB_DATA_DIR}/downloads/crypto_spot/1d/...` to match the new convention — this is a real operational decision, not a code change (see Open Questions). |
| Live service config | None — no external services (n8n, Datadog, etc.) reference this project's config. | None. |
| OS-registered state | None — no cron/launchd/Task Scheduler jobs reference Tiingo or Binance ingestion today (confirmed: this is the first automated/incremental acquisition component; the existing script is manually run). | None now; if the plan schedules the new `TiingoAcquisition.refresh()` via cron/launchd for daily runs, that registration is new work, not a migration. |
| Secrets/env vars | `TIINGO_API_KEY` — already read from env in `scripts/download_stock_data_from_tiingo.py` (Phase 1 fixed this). The new `TiingoAcquisition` component must read the same env var name for continuity; no key rotation is needed for this phase (Phase 1 SUMMARY records the user explicitly deferred key rotation). | Code should read `TIINGO_API_KEY`; no secret-value migration needed. |
| Build artifacts | None relevant — no compiled/installed-package artifacts reference market/frequency path strings. | None. |

## Common Pitfalls

### Pitfall 1: Re-running `save()` against an existing Zarr path crashes with `FileExistsError`
**What goes wrong:** `XrBackend.write()` calls `self.data.to_zarr(path, **kwargs)` with no default `mode`. Zarr's own default write mode is `"w-"` ("create, fail if exists"). The very first `save()` to a new path succeeds; any subsequent `save()` to the *same* path — which is exactly what a daily-refresh workflow requires — fails.
**Why it happens:** No existing caller (`cal.py`, `train_model.py`, `test.py`) has ever re-run `save()` twice against the same path in this codebase's history, so this defect has never surfaced.
**Verified empirically in this session:**
```python
ds1.to_zarr("zarrtest.zarr")   # succeeds
ds2.to_zarr("zarrtest.zarr")   # FileExistsError: Cannot create '' with mode 'w-' ...
```
**How to avoid:** Either (a) have `save()`/`write()` pass `mode="w"` explicitly by default (full-overwrite semantics, consistent with the existing "rebuild from all raw files" pattern — recommended for v1), or (b) require every re-run caller to pass `mode="w"` via `save(**kwargs)`. Option (a) is safer since it requires no caller-side changes and matches how every existing script already behaves conceptually (always rebuild-and-persist the full dataset).
**Warning signs:** Any test or manual run that calls `.from_raw_data().save()` twice against the same `zarr_file_path` without first deleting the directory will surface this immediately — a good candidate for an automated regression test (see Validation Architecture).

### Pitfall 2: Duplicate `(timestamp, symbol)` rows crash `to_xarray()`, they don't silently produce bad data
**What goes wrong:** Teams sometimes assume "dedup" is a data-quality nicety that can be skipped for v1 and added later. Here, skipping it is not a quality issue — it is a hard runtime crash the first time overlapping raw files exist (a documented real risk for Binance's monthly-CSV-near-month-boundary case, and possible for Tiingo if `download()`+`refresh()` ranges ever overlap).
**Why it happens:** `pandas.DataFrame.set_index([...]).to_xarray()` requires a unique MultiIndex.
**How to avoid:** Dedup must run on the long-format DataFrame/LazyFrame *before* the final `.to_xarray()` call inside each subclass's `_raw_data_to_xr()` — see Pattern 2.
**Warning signs:** `ValueError: cannot convert a DataFrame with a non-unique MultiIndex into xarray` surfacing from `_raw_data_to_xr()`.

### Pitfall 3: Tiingo's default JSON field set may not include adjusted-price/corporate-action columns
**What goes wrong:** `StockDataset._to_kunquant` unconditionally expects `adjOpen`/`adjHigh`/`adjLow`/`adjClose`/`adjVolume` columns to already exist in the raw data. The existing reference script calls `client.get_ticker_price(stock, fmt="json", startDate=..., endDate=..., frequency="daily")` **without** an explicit `columns=` parameter. The `tiingo-python` client's own docstring for the `columns` parameter states: "By default, 'date', 'open', 'close', 'high' and 'low' are retrieved. 'volume' is an extra option" [CITED: github.com/hydrosquall/tiingo-python api.py docstring] — which, read literally, suggests the adjusted/corporate-action fields (`adjOpen`, `adjClose`, `divCash`, `splitFactor`, etc.) may NOT be present unless explicitly requested via `columns=`.
**Why it happens/uncertainty:** This is genuinely unclear from documentation alone — Tiingo's own end-of-day API docs page lists all available fields but does not state which subset is the JSON default [CITED: tiingo.com/documentation/end-of-day, could not confirm via WebFetch in this session]. It's possible the client docstring describes the `columns` *parameter's own default value* for filtering purposes rather than the literal server-side JSON response shape, and the actual EOD endpoint may return the full field set regardless. This cannot be resolved without a live API call.
**How to avoid:** The new `TiingoAcquisition` component should **always pass an explicit `columns=` parameter** listing every field `StockDataset._raw_data_to_xr`/`_to_kunquant` needs (`open,high,low,close,volume,adjOpen,adjHigh,adjLow,adjClose,adjVolume,divCash,splitFactor`) rather than relying on any undocumented default. This removes the ambiguity entirely and should be a design requirement, not just a suggestion.
**Warning signs:** `KeyError`/`drop_vars` failures in `StockDataset._to_kunquant` when it tries to rename `adjOpen`→`open` etc. on data that never had those columns.

## Code Examples

### Confirmed Tiingo client call shape (from the existing reference script and official client source)
```python
# Source: scripts/download_stock_data_from_tiingo.py (existing, read-only reference per D-10)
# and github.com/hydrosquall/tiingo-python/blob/master/tiingo/api.py (method signature)
data = client.get_ticker_price(
    ticker,
    fmt="json",
    startDate=start_date,   # "YYYY-MM-DD" [CITED: tiingo-python README/api.py]
    endDate=end_date,
    frequency="daily",      # routes to /tiingo/daily/{ticker}/prices [CITED: api.py _get_url]
    columns=(                # RECOMMENDED addition — do not rely on undocumented default (Pitfall 3)
        "open,high,low,close,volume,"
        "adjOpen,adjHigh,adjLow,adjClose,adjVolume,divCash,splitFactor"
    ),
)
```

### Dedup-before-conversion pattern for `_raw_data_to_xr()`
```python
# Illustrative pattern for both StockDataset._raw_data_to_xr and SpotKlineDataset._raw_data_to_xr
data = data.sort(by=["timestamp", "symbol"])
data = data.unique(subset=["timestamp", "symbol"], keep="last")  # D-05: deterministic, documented
data = data.collect().to_pandas().set_index(["timestamp", "symbol"])
return data.to_xarray()   # D-06's NaN-gap behavior now falls out for free (Pattern 3)
```
`keep="last"` is recommended (not mandated) because later-arriving files in a vendor's monthly-drop workflow more often represent corrected/reprocessed data than earlier ones; document whichever choice is made in the cleaning module's docstring per D-05.

### Explicit overwrite mode fix for `XrBackend.write()`
```python
# base/dataset/backend.py:XrBackend.write — minimal change to unblock incremental refresh (Pitfall 1)
def write(self, path: str, **kwargs) -> Self:
    if not Path(path).exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("mode", "w")   # NEW: default to full-overwrite so re-runs don't crash
    self.data.to_zarr(path, **kwargs)
    return self
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Flat procedural Tiingo script (`scripts/download_stock_data_from_tiingo.py`) hardcoding `frequency="daily"` and writing directly to a fixed local path | Encapsulated `Acquisition` ABC + `TiingoAcquisition` concrete class, config-driven, parameterized by frequency, with an incremental watermark | This phase (D-10) | Adding a future intraday frequency or a future vendor becomes a new config/class, not a rewrite of a script's internals |
| Market-specific hardcoded path segments in every config factory (`spot/monthly/klines`) | `market`/`frequency`-derived path convention (`data/{market}/{frequency}/{name}.zarr`) | This phase (D-01/D-02) | New markets/frequencies require zero changes to `config/__init__.py`'s path-building logic, only new factory calls with different `market`/`frequency` arguments |

**Deprecated/outdated:** None — this phase only adds capability, it does not remove or replace an existing, still-used mechanism.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Tiingo's EOD JSON response, when `columns=` is not explicitly specified, may omit adjusted-price and corporate-action fields (`adjOpen`, `adjClose`, `divCash`, `splitFactor`, etc.) — based on the `tiingo-python` client docstring, not a confirmed live API response | Common Pitfalls (Pitfall 3), Code Examples | If wrong in either direction: if the plan assumes explicit `columns=` is required and it turns out the API always returns everything by default, no harm (explicit is still safer/clearer). If the plan *skips* the explicit `columns=` param assuming defaults are fine and the API's actual default is the narrow 5-column set, `StockDataset._to_kunquant`'s `adjOpen`/etc. rename step will `KeyError` at runtime. Mitigation already built into the recommendation: always pass explicit `columns=`, removing the risk regardless of which assumption is correct. |
| A2 | `market: Literal["us_equity", "crypto_spot"]` and `frequency: Literal["1d", "1m", "tick"]` are the right token sets | User Constraints / Standard Stack | Low risk — CONTEXT.md D-01 explicitly delegates exact literal values to planner discretion; these are proposed defaults consistent with existing naming (`ret_1m` labels, real on-disk `.../1d/` folder), not externally-sourced facts requiring confirmation |

**A2 is a design proposal within an explicitly delegated discretion area (not an unverified factual claim) — included for completeness only.**

## Open Questions

1. **Where should the retrofitted `spot_kline_config()`'s `raw_data_dir_path` actually point?**
   - What we know: Real, already-downloaded Binance raw CSVs exist on this machine at `~/Downloads/spot/monthly/klines/BTCUSDT/1d/...` and in a sibling project `~/projects/binance-data-downloader/downloads/...` — both **outside** `QUANTLAB_DATA_DIR`/the repo's `data/` root. No Zarr caches exist yet inside the project.
   - What's unclear: Whether the user wants these files copied/symlinked into the new `{QUANTLAB_DATA_DIR}/downloads/crypto_spot/1d/...` convention (clean, but a multi-GB data move) or whether `spot_kline_config()` should simply point `raw_data_dir_path` at the existing external location (works immediately, but the "path convention" for raw dirs becomes non-uniform between sources).
   - Recommendation: Flag as a `checkpoint:human-verify` or direct question at plan/execution time — this is a real, machine-specific operational decision, not something to silently assume.

2. **Does the Tiingo EOD API actually return adjusted/corporate-action fields by default?**
   - What we know: The `tiingo-python` client docstring suggests a narrower 5-column default; Tiingo's own docs page lists all fields without stating the default subset.
   - What's unclear: Actual server response shape without a live, valid API key.
   - Recommendation: Always pass an explicit `columns=` parameter in `TiingoAcquisition` (see Code Examples/Pitfall 3) — this sidesteps the need to resolve the ambiguity at all. If the plan wants certainty anyway, a Wave 0 smoke-test task with a valid `TIINGO_API_KEY` making one real `get_ticker_price` call and inspecting the returned JSON keys would resolve this definitively.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `tiingo` Python package | `TiingoAcquisition` | ✓ | 0.16.1 [VERIFIED: `uv run python -c "import tiingo"`] | — |
| `TIINGO_API_KEY` env var | `TiingoAcquisition` runtime auth | Not currently set in this shell session (checked implicitly — script raises `RuntimeError` if unset) | — | User must export before running acquisition; existing script already enforces this pattern |
| Network access to `api.tiingo.com` | Live acquisition runs / Wave 0 smoke test | Not tested in this sandboxed research session | — | If unavailable, acquisition component must still be unit-testable via a mocked `TiingoClient` (see Validation Architecture) |
| `zarr`/`xarray` (to_zarr/open_dataset) | Persistence layer | ✓ | zarr 3.3.0, xarray 2026.7.0 [VERIFIED: pyproject.toml + empirical `to_zarr`/`to_xarray` calls in this session] | — |
| External Binance raw CSVs (`~/Downloads/spot/monthly/klines/...`) | `SpotKlineDataset` retrofit (D-04) | ✓ present on this machine, but outside the project's `data/` root | — | See Open Question 1 |

**Missing dependencies with no fallback:** None identified — all required libraries are already installed and importable.

**Missing dependencies with fallback:** `TIINGO_API_KEY`/network access — acquisition logic must be designed to be unit-testable without live network access (inject/mock the `TiingoClient`).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (not yet installed — no `pytest` in `pyproject.toml`, no `tests/` directory, confirmed via `find`/grep in this session) |
| Config file | none — see Wave 0 Gaps |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | `TiingoAcquisition` fetches + `StockDataset` persists a `[timestamp, symbol]` xr.Dataset to Zarr | unit (mocked `TiingoClient`) + integration (real Zarr roundtrip on tmp path) | `uv run pytest tests/test_tiingo_acquisition.py tests/test_stock_dataset.py -x` | ❌ Wave 0 |
| DATA-02 | Retrofitted `SpotKlineDataset`/`spot_kline_config()` still produces a correct xr.Dataset from fixture CSVs under new market/frequency paths | unit (small synthetic CSV fixtures matching `BinanceCSVHeaders.SPOT`) | `uv run pytest tests/test_spot_dataset.py -x` | ❌ Wave 0 |
| DATA-03 | Adding a new market/frequency requires only a new `Dataset` subclass + config, no factor/model/backtest changes | design/code review (per ROADMAP Success Criterion 4, explicitly not proven by building a third source) + one automated proof: a minimal test-only `FakeDataset` subclass + config exercising the full `from_raw_data()`→`save()`→`read()` lifecycle without touching non-`dataset/config` files | `uv run pytest tests/test_extensibility_contract.py -x` (automated proof) + manual code review checklist | ❌ Wave 0 |
| DATA-04 | Cleaning module: dedup (D-05), NaN-gap non-imputation (D-06), anomaly-flagging not deletion (D-07), schema validation (D-08) | unit (small crafted DataFrames/xr.Datasets per rule) | `uv run pytest tests/test_cleaning.py -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q` (fast subset relevant to the task)
- **Per wave merge:** `uv run pytest tests/ -q` (full suite)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Add `pytest` as a dev dependency: `uv add --dev pytest`
- [ ] `tests/` directory + `tests/conftest.py` — shared fixtures: a small synthetic Binance-CSV fixture (matching `BinanceCSVHeaders.SPOT`), a small synthetic Tiingo-JSON fixture (matching the confirmed field set), and a mocked `TiingoClient` fixture (no real network calls in unit tests)
- [ ] `tests/test_cleaning.py`, `tests/test_tiingo_acquisition.py`, `tests/test_stock_dataset.py`, `tests/test_spot_dataset.py`, `tests/test_extensibility_contract.py` — all net-new, no existing test infrastructure to extend (confirmed: zero `tests/`, zero pytest config anywhere in the repo)
- [ ] A regression test that explicitly reproduces Pitfall 1 (`save()` twice against the same path) to lock in the `mode="w"` fix

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | This is a local batch pipeline with no user-facing auth surface |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Yes | New `dataset/cleaning.py` schema/type/non-null validation (D-08) on all incoming raw market data before persistence |
| V6 Cryptography | No | No new crypto/hashing introduced by this phase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Credential (Tiingo API key) leaking into serialized config/checkpoint JSON | Information Disclosure | Never add the API key as a field on `AcquisitionConfig`/`DatasetConfig` (both are `asdict()`-serialized into checkpoint JSON via `base/model.py:_save_model` and reconstructed via `utils/module.py`); read `TIINGO_API_KEY` directly from `os.environ` only inside `TiingoAcquisition.__init__`, matching the project's existing SEC-01 remediation pattern from Phase 1 |
| Malformed/corrupted vendor data silently corrupting downstream factor computation | Tampering (data integrity, not adversarial) | D-07/D-08's flag-don't-delete + schema validation is the standard mitigation already locked in CONTEXT.md — do not weaken this to auto-correction |
| Unbounded/unvalidated raw file paths (`raw_data_dir_path`) pointing outside the intended data root | Tampering / Information Disclosure (path traversal-adjacent) | Low severity here since all paths are locally-configured (not user/network-supplied at runtime), but the new `stock_kline_config()` factory should still derive paths via the same `_data_root()` helper as existing factories rather than accepting arbitrary absolute strings from an untrusted source |

## Sources

### Primary (HIGH confidence)
- Direct file reads: `base/config.py`, `base/data.py`, `base/backend.py`, `dataset/backend.py`, `dataset/stock.py`, `dataset/spot.py`, `config/__init__.py`, `my_ops/preprocess.py`, `scripts/download_stock_data_from_tiingo.py`, `enums/data.py`, `enums/constant.py`, `utils/file.py`, `utils/module.py`, `test.py`, `pyproject.toml` — all read in full during this session
- Empirical tests run in this session via `uv run python3 -c "..."` (pandas `to_xarray()` duplicate-index behavior, NaN cartesian-product behavior, `xr.Dataset.to_zarr()` default-overwrite behavior) — all VERIFIED, reproducible, exact error messages captured
- `uv run python3 -c "import tiingo/KunQuant"` — confirmed both packages importable in the project's actual `uv`-managed venv

### Secondary (MEDIUM confidence)
- [tiingo-python api.py source (GitHub, hydrosquall/tiingo-python)](https://github.com/hydrosquall/tiingo-python/blob/master/tiingo/api.py) — `get_ticker_price` signature, endpoint routing, `columns` parameter docstring, `get_dataframe`'s `valid_columns` set (via WebFetch of raw source file)

### Tertiary (LOW confidence)
- [Tiingo End-of-Day API documentation](https://www.tiingo.com/documentation/end-of-day) — confirms the full field list exists but could NOT confirm which subset is the JSON default without `columns=` — explicitly flagged as unresolved in Open Questions / Assumptions Log (A1)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages, all versions confirmed installed and importable in the actual project environment
- Architecture: HIGH — based on direct reads of the actual existing code, not documentation or training-data recall; two key behaviors (dedup-crash, missing-NaN-for-free, Zarr overwrite-crash) were empirically verified with live code execution in this session, not merely asserted
- Pitfalls: HIGH for the three documented pitfalls (all empirically reproduced or directly evidenced from real on-disk file layouts) — MEDIUM for the Tiingo default-columns question specifically (A1), which remains genuinely unresolved without a live API call

**Research date:** 2026-09-04
**Valid until:** 30 days (stable, mostly-internal-codebase-driven findings; the one external-API uncertainty, A1, should be resolved by a live smoke test early in phase execution regardless of this expiry window)
