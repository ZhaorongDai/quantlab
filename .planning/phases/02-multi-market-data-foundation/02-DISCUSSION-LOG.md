# Phase 2: Multi-Market Data Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-05
**Phase:** 02-multi-market-data-foundation
**Areas discussed:** Storage layout / config schema, Binance data source scope, Cleaning/preprocessing rules

---

## Storage layout / config schema

| Option | Description | Selected |
|--------|-------------|----------|
| `data/{market}/{frequency}/{symbol集合}.zarr` | `market`/`frequency` as explicit `DatasetConfig` fields, driving directory layout directly; new markets/frequencies need no code change | ✓ |
| No strong convention, user decides details | User describes their own preferred layout | |

**User's choice:** `data/{market}/{frequency}/{symbol集合}.zarr` (recommended option)
**Notes:** Current `DatasetConfig` has no `market`/`frequency` fields at all; paths are hardcoded per-market inside each `config/__init__.py` factory function (e.g. `spot/monthly/klines`). This is the core gap blocking ROADMAP Phase 2 Success Criterion 4.

---

## Binance data source — scope for this phase

| Option | Description | Selected |
|--------|-------------|----------|
| 写自动下载器（推荐） | Call Binance's public data download endpoint / REST klines API to auto-fetch/refresh, matching the existing Tiingo script's role | |
| 只整理现有手动流程 | Don't write downloader code; formalize/document the existing "manually download CSVs into a directory" workflow | ✓ (via free-text) |

**User's choice (free text):** "先使用已有的币安数据，目前最紧迫的目标是完成美股的全流程" (use the existing Binance data as-is; the most urgent goal right now is completing the US-equities full pipeline)
**Notes:** This reprioritizes phase effort toward the US-equities (Tiingo) path. Binance/`SpotKlineDataset` only gets retrofitted into the new `market`/`frequency` config schema — no new download automation this phase. An actual Binance auto-downloader is deferred (see Deferred Ideas below).

---

## Cleaning / preprocessing rules

| Option | Description | Selected |
|--------|-------------|----------|
| 重复/重叠时间戳去重 | Deduplicate rows sharing `(symbol, timestamp)`, keep one deterministic record | ✓ |
| 缺失时间点/非交易日处理 | Align to a time grid, leave missing values as `NaN` — explicitly no forward-fill | ✓ |
| 异常价格/成交量检测（标记不删除） | Flag anomalies (e.g. extra column or log warning), never silently delete/correct | ✓ |
| 仅基础校验（schema/类型/非空） | Minimal completeness checks only, no statistical anomaly detection this phase | ✓ |

**User's choice:** All four selected together — no further prioritization among them; all four are in scope for v1's shared cleaning module.
**Notes:** Today's `my_ops/preprocess.py` only contains factor-normalization ops (`WindowedZScore`, `WindowedRobustStandardization`) — there is no raw-market-data cleaning logic anywhere in the codebase yet. This is new code.

---

## Tiingo acquisition component design (follow-up correction)

**User's correction (free text):** "tinngo 数据采集逻辑不要沿用get_binance_instruments.py/cal.py的风格，要参考代码库的风格，做好封装，为未来多频率数据、数据每日更新做准备" (the Tiingo data-acquisition logic should not follow the style of `get_binance_instruments.py`/`cal.py` — it should follow the codebase's [layered/OOP] style, be properly encapsulated, and be prepared for future multi-frequency data and daily data updates)

**Notes:** This arrived after the initial CONTEXT.md draft, which had (incorrectly) suggested `get_binance_instruments.py`/`cal.py` as the style reference for D-10. Corrected D-10 to require a class-based, encapsulated component (in the spirit of `base/data.py:Dataset`/`dataset/stock.py:StockDataset`) rather than any flat procedural script — and made multi-frequency parameterization + incremental/daily-refresh capability explicit, locked requirements rather than "nice to have later."

---

## Claude's Discretion

- Exact `Literal` value sets for the new `market`/`frequency` config fields.
- Whether the new cleaning module lives inside `my_ops/` or as a new sibling package (e.g. `dataset/cleaning.py`).
- Exact duplicate-timestamp tie-breaking rule (first vs. last), as long as deterministic and documented.
- Whether new Tiingo acquisition code is a rewritten script in place or a new standalone script — as long as it does not copy `scripts/download_stock_data_from_tiingo.py`'s throwaway patterns (hardcoded relative paths, notebook-cell style), per the project's temp-vs-core-code guardrail.

## Deferred Ideas

- Binance automatic downloader (Binance public bulk data / REST klines API) — explicitly out of scope for this phase; may become a future phase/backlog item.
- Statistical/ML-based anomaly detection beyond simple flagging — deferred to a possible later data-quality phase.
- Tick-level and minute-frequency production ingestion for both markets — already deferred to v2 per `.planning/PROJECT.md` and `.planning/REQUIREMENTS.md` (DATA-V2-01/02); this phase only needs the config schema to be *capable* of supporting them.
