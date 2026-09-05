---
phase: 02-multi-market-data-foundation
plan: 07
subsystem: data
tags: [polars, xarray, zarr, tiingo, dedup, cli]

# Dependency graph
requires:
  - phase: 02-multi-market-data-foundation (02-04)
    provides: TiingoAcquisition (acquisition/tiingo.py), AcquisitionConfig, watermark orchestration
  - phase: 02-multi-market-data-foundation (02-02)
    provides: stock_acquisition_config()/stock_kline_config() market/frequency config factories
  - phase: 02-multi-market-data-foundation (02-03)
    provides: dataset/cleaning.py (dedup_raw_frame, clean_market_data, flag_anomalies, validate_schema)
provides:
  - StockDataset._raw_data_to_xr() deduplicates overlapping (timestamp, symbol) rows before to_xarray() (D-05)
  - StockDataset._raw_data_to_xr() tolerant of raw parquet files with differing column order/set (pl.concat how="diagonal_relaxed")
  - Full mocked-network integration test proving TiingoAcquisition -> StockDataset -> Zarr round trip
  - ingest_tiingo.py -- documented, user-runnable CLI entry point for the full US-equities Tiingo-to-Zarr pipeline
affects: [phase-03-factor-computation, phase-07-quality]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "dedup_raw_frame() inserted immediately before .collect()/.to_xarray() in every Dataset subclass's _raw_data_to_xr() (now both SpotKlineDataset and StockDataset)"
    - "pl.concat(..., how=\"diagonal_relaxed\") for combining raw files from potentially-differing vendor/acquisition sources under the same raw_data_dir_path"

key-files:
  created: [ingest_tiingo.py, tests/test_stock_dataset.py]
  modified: [dataset/stock.py, README.md]

key-decisions:
  - "dedup_raw_frame(data, keep=\"last\") inserted at the identical point 02-05 used for SpotKlineDataset -- keeps the dedup insertion pattern uniform across both Dataset subclasses"
  - "pl.concat(stock_dfs) switched from default \"vertical\" to \"diagonal_relaxed\" -- vertical requires exact column order across every raw file, which crashed the moment two symbols' raw files came from sources ordering columns differently (found while building the multi-symbol Task 2 integration test)"

patterns-established:
  - "StockDataset raw-file ingestion is now schema-order-agnostic (diagonal_relaxed), matching columns by name rather than position -- future acquisition sources for this market/frequency don't need to match a fixed column order"

requirements-completed: [DATA-01]

# Metrics
duration: 25min
completed: 2026-09-05
---

# Phase 2 Plan 07: US-Equities Tiingo Vertical Slice Summary

**StockDataset now dedupes overlapping raw Tiingo rows and ingests schema-order-agnostic parquet files; `ingest_tiingo.py` is the single documented command that pulls US-equities daily data from Tiingo and persists it as xr.Dataset/Zarr, with `--refresh` for incremental updates.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-09-05T14:44:00Z (approx, worktree setup)
- **Completed:** 2026-09-05T14:51:45Z
- **Tasks:** 3/3 completed
- **Files modified:** 4 (dataset/stock.py, tests/test_stock_dataset.py, ingest_tiingo.py, README.md)

## Accomplishments
- `StockDataset._raw_data_to_xr()` no longer crashes on overlapping/duplicate raw Tiingo `(timestamp, symbol)` rows (D-05), matching the pattern already applied to `SpotKlineDataset` in 02-05
- A full, real (non-live-network) integration test proves `TiingoAcquisition.download()` -> `StockDataset.from_raw_data().save()` -> `StockDataset(...).read()` round-trips a `[timestamp, symbol]` `xr.Dataset` through a real tmp-path Zarr store, carrying `adjClose` values, an `anomaly_flag` variable (proving `clean_market_data()` ran), and NaN (not forward-filled) gaps for absent `(timestamp, symbol)` combinations
- `ingest_tiingo.py` ships as the single documented, user-runnable command for the US-equities Tiingo-to-Zarr pipeline, satisfying ROADMAP Success Criterion 1 and fully closing out DATA-01

## Task Commits

Each task was committed atomically (Task 1 and Task 2 both followed RED/GREEN TDD cycles per their `tdd="true"` markers):

1. **Task 1 (RED): add failing test for dedup_raw_frame in StockDataset** - `c0b3720` (test)
2. **Task 1 (GREEN): insert dedup_raw_frame into StockDataset._raw_data_to_xr (D-05)** - `54da1c5` (feat)
3. **Task 2: full TiingoAcquisition-to-Zarr integration test (DATA-01)** - `4c93790` (feat; includes the Rule 1/3 `pl.concat` schema-order fix required to make the test's multi-symbol NaN-gap assertion possible)
4. **Task 3: add ingest_tiingo.py entry point + README docs (DATA-01)** - `0a84662` (feat)

_Note: Task 1's RED commit intentionally also stages Task 2's integration test file skeleton (same `tests/test_stock_dataset.py` file) since both tasks' tests live in one file; Task 2's commit records the additional test logic and its own production fix separately._

## Files Created/Modified
- `dataset/stock.py` - `_raw_data_to_xr()` now calls `dedup_raw_frame(data, keep="last")` before `.collect()`, and `pl.concat(stock_dfs, how="diagonal_relaxed")` replaces the schema-order-strict default `pl.concat(stock_dfs)`
- `tests/test_stock_dataset.py` (new) - 2 dedup unit tests (overlap crash fix + non-overlap regression) + 1 full mocked-network integration test proving the DATA-01 round trip end-to-end
- `ingest_tiingo.py` (new) - thin, documented CLI: `--symbols` (required), `--start-date`, `--end-date`, `--refresh`; wires `stock_acquisition_config()`/`TiingoAcquisition` to `stock_kline_config()`/`StockDataset`; never prints/logs the raw `TIINGO_API_KEY` value or the `TiingoClient` config dict
- `README.md` - documents `ingest_tiingo.py` under Entry Points; updates the `TIINGO_API_KEY` bullet under Environment Variables to reference it as the current documented entry point

## Decisions Made
- Kept the dedup insertion point identical to 02-05's `SpotKlineDataset` pattern (immediately before `.collect()`/`.to_xarray()`) for consistency across `Dataset` subclasses.
- Switched `pl.concat`'s combine strategy to `"diagonal_relaxed"` rather than requiring all raw parquet writers to produce identically-ordered columns -- more robust to future acquisition sources without adding project-specific column-order conventions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1/3 - Bug/Blocking] `pl.concat(stock_dfs)` required exact column order across every raw parquet file**
- **Found during:** Task 2 (building the multi-symbol NaN-gap assertion in the integration test)
- **Issue:** `pl.concat()`'s default `"vertical"` strategy raises `polars.exceptions.InvalidOperationError` the moment two raw parquet files under `raw_data_dir_path` have columns in a different order (even with an identical column *set* and matching dtypes) -- confirmed by direct reproduction with a minimal `pl.concat` example. This would crash `StockDataset.from_raw_data()` in production whenever raw files under the same directory originate from more than one writer/acquisition path.
- **Fix:** Changed `pl.concat(stock_dfs)` to `pl.concat(stock_dfs, how="diagonal_relaxed")`, which matches columns by name (not position) and tolerates a differing-but-compatible column set/order.
- **Files modified:** `dataset/stock.py`
- **Verification:** `uv run pytest tests/test_stock_dataset.py -v` (all 3 tests pass) and `uv run pytest tests/ -v` (all 29 tests pass)
- **Committed in:** `4c93790` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1/3 -- bug that also blocked task completion)
**Impact on plan:** Necessary for correctness/robustness of raw-file ingestion; no scope creep -- fix is scoped to the exact `pl.concat` call already touched by this plan's `dataset/stock.py` changes.

## Issues Encountered
- Initial integration-test fixture (`_write_stock_pqt`) for the multi-symbol NaN-gap assertion did not match `TiingoAcquisition`'s real written schema (missing `divCash`/`splitFactor`, mismatched `volume` dtype, different column order) -- fixed by aligning the test fixture's schema/column order to `acquisition/tiingo.py`'s actual output, which in turn surfaced the `pl.concat` column-order bug documented above.

## User Setup Required

None - no external service configuration required. Running `ingest_tiingo.py` for real requires a `TIINGO_API_KEY` (already documented in README.md prior to this plan; this plan adds `ingest_tiingo.py` as an additional consumer of that same variable).

## Next Phase Readiness

- DATA-01 is now fully satisfied: a real, testable, end-to-end path from Tiingo's API shape through to a Zarr-persisted `xr.Dataset` exists (`ingest_tiingo.py`), matching the Binance spot-kline slice already completed in 02-05.
- Both `Dataset` subclasses (`SpotKlineDataset`, `StockDataset`) now share the same dedup-before-`to_xarray()` pattern; a future third market/frequency `Dataset` subclass should follow the same insertion point.
- `StockDataset`'s raw-file ingestion is more robust to differing acquisition sources (`diagonal_relaxed`), reducing risk for Phase 3's factor computation layer, which reads `StockDataset` output.
- No blockers identified for downstream phases.

---
*Phase: 02-multi-market-data-foundation*
*Completed: 2026-09-05*

## Self-Check: PASSED

All created/modified files verified present on disk:
- `dataset/stock.py`, `tests/test_stock_dataset.py`, `ingest_tiingo.py`, `README.md`, this SUMMARY.md

All task commits verified present in `git log`:
- `c0b3720` (test), `54da1c5` (feat), `4c93790` (feat), `0a84662` (feat), `5774b86` (docs: this summary)
