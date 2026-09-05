---
phase: 02-multi-market-data-foundation
plan: 05
subsystem: data
tags: [polars, xarray, zarr, binance, dataset, dedup, cli]

# Dependency graph
requires:
  - phase: 02-multi-market-data-foundation
    provides: "02-02's market/frequency config schema (spot_kline_config()); 02-03's shared dataset/cleaning.py module and overridable Dataset._clean() hook"
provides:
  - "SpotKlineDataset._raw_data_to_xr() dedups overlapping (timestamp, symbol) rows via dedup_raw_frame() before .to_xarray(), preventing a non-unique-MultiIndex crash"
  - "SpotKlineDataset._clean() override that validates against Binance's actual Title-Case OHLCV column names, fixing a previously-unconditional crash of SpotKlineDataset.from_raw_data() introduced by 02-03's centralized lowercase-column cleaning hook"
  - "ingest_binance_spot.py: documented top-level CLI rebuilding the Binance spot-kline Zarr store from locally-dropped CSVs, with a --raw-data-dir override for CSVs stored outside the repo's data/{market}/{frequency}/... convention"
  - "Real-data smoke check (D-11): a real BTCUSDT 1d monthly CSV fetched via the user's binance-data-downloader tool, round-tripped through ingest_binance_spot.py into Zarr"
affects: [phase-03-factor-computation, phase-04-return-model]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dataset._clean() override pattern: subclasses whose raw column casing/naming doesn't match dataset/cleaning.py's lowercase OHLCV convention override _clean() to call validate_schema()/flag_anomalies() with subclass-appropriate required_columns, rather than forcing every Dataset subclass onto one hardcoded schema."
    - "Thin-script CLI convention extended: ingest_binance_spot.py factors argparse-Namespace-to-config construction into one small _build_dataset_config(args) helper so override behavior is unit-testable without invoking argparse, matching cal.py's overall shape otherwise."

key-files:
  created: [ingest_binance_spot.py, tests/test_spot_dataset.py, .planning/phases/02-multi-market-data-foundation/02-05-REAL-DATA-CHECK.md]
  modified: [dataset/spot.py, README.md]

key-decisions:
  - "Fixed a pre-existing (not plan-anticipated) bug: SpotKlineDataset.from_raw_data() was unconditionally broken by 02-03's centralized Dataset._clean() hook, which validates lowercase open/high/low/close/volume columns -- Binance's raw columns are Title-Case (Open/High/Low/Close/Volume). Fixed via a SpotKlineDataset._clean() override (Rule 1 auto-fix) rather than renaming columns pipeline-wide, keeping the change confined to dataset/spot.py and off the D-04 scope boundary the plan set for _raw_data_to_xr() itself."
  - "Real-data check (Task 3) used 2026-07 instead of 2026-08 ('most recent fully-completed month' per this machine's system clock) because Binance's public data.vision archive had not yet published a 2026-08 monthly file at run time -- confirmed via a direct bucket listing before falling back to the last month actually available, and documented in 02-05-REAL-DATA-CHECK.md."

patterns-established:
  - "Dataset._clean() override pattern (see tech-stack.patterns)"

requirements-completed: [DATA-02]

# Metrics
duration: ~35min (resumed after a session rate-limit interruption; no prior commits existed for this plan, so all 3 tasks were executed fresh in this session)
completed: 2026-09-05
---

# Phase 02 Plan 05: Binance Spot Dedup + ingest_binance_spot.py CLI Summary

**Binance spot-kline dedup via dedup_raw_frame() plus a new ingest_binance_spot.py CLI (--raw-data-dir override), verified against both fixture tests and a real BTCUSDT month fetched via binance-data-downloader.**

## Performance

- **Duration:** ~35 min (this session; plan was interrupted by a session rate limit before any commits landed, then resumed and executed end-to-end)
- **Completed:** 2026-09-05
- **Tasks:** 3/3 completed
- **Files modified:** 5 (dataset/spot.py, ingest_binance_spot.py [new], tests/test_spot_dataset.py [new], README.md, 02-05-REAL-DATA-CHECK.md [new])

## Accomplishments
- `SpotKlineDataset._raw_data_to_xr()` no longer crashes on overlapping/duplicate `(timestamp, symbol)` rows near a Binance monthly-CSV boundary (`dedup_raw_frame(res, keep="last")` inserted between `.sort()` and `.collect()`).
- Fixed a latent, plan-unanticipated bug where `SpotKlineDataset.from_raw_data()` was unconditionally broken by 02-03's shared `_clean()` hook (lowercase-column schema check vs. Binance's Title-Case raw columns) -- without this fix, Task 1's own acceptance criteria (`.from_raw_data()` succeeds) would have been unreachable regardless of the dedup fix.
- New `ingest_binance_spot.py` top-level CLI, matching `cal.py`'s thin-script convention, with `--symbols`/`--start-date`/`--end-date`/`--raw-data-dir` and zero new network-fetch code (verified via grep).
- README.md documents the new entry point and its `--raw-data-dir` override.
- Real, non-fixture verification (D-11): fetched a real BTCUSDT July-2026 monthly CSV via the user's separate `binance-data-downloader` tool and ran it through `ingest_binance_spot.py` into a real Zarr store (31 rows, plausible OHLCV values) -- see `02-05-REAL-DATA-CHECK.md`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Insert dedup_raw_frame() into SpotKlineDataset._raw_data_to_xr() (D-05)** - `4e4cc23` (feat)
2. **Task 2: ingest_binance_spot.py CLI with --raw-data-dir override + README update** - `1c18d89` (feat)
3. **Task 3: Real-data smoke check via binance-data-downloader (D-11)** - `44211b2` (docs)

**Plan metadata:** this SUMMARY.md commit (docs)

## Files Created/Modified
- `dataset/spot.py` - Added `dedup_raw_frame()` call before `.to_xarray()` conversion (D-05); added `_clean()` override validating Title-Case OHLCV columns (Rule 1 bug fix, not in original plan scope)
- `ingest_binance_spot.py` - New thin CLI entry point rebuilding the Binance spot-kline Zarr store from local CSVs, with `--raw-data-dir` override; no network code
- `tests/test_spot_dataset.py` - 4 unit tests: overlapping-CSV dedup, non-duplicate-CSV regression, `--raw-data-dir` no-op, `--raw-data-dir` override-applies
- `README.md` - Documents `ingest_binance_spot.py` under Entry Points
- `.planning/phases/02-multi-market-data-foundation/02-05-REAL-DATA-CHECK.md` - Real BTCUSDT smoke-check evidence (command transcript, Zarr path, row count, sample OHLCV rows)

## Decisions Made
- Fixed `SpotKlineDataset.from_raw_data()`'s pre-existing incompatibility with 02-03's centralized `_clean()` hook via a `SpotKlineDataset._clean()` override (Rule 1), rather than renaming Binance's raw columns to lowercase pipeline-wide -- the latter would have required touching `_to_kunquant()` and `_xr_to_bars()` as well, exceeding the plan's explicit D-04 scope boundary ("not a rewrite of the CSV-discovery or column-mapping logic").
- Task 3 used 2026-07 instead of the literally-most-recent calendar month (2026-08) because Binance's public archive hadn't published 2026-08 data yet at run time; confirmed via a direct S3 bucket listing before falling back. Documented transparently in `02-05-REAL-DATA-CHECK.md` rather than silently substituting without explanation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed SpotKlineDataset.from_raw_data() unconditional crash from 02-03's lowercase-column schema check**
- **Found during:** Task 1 (writing the dedup test, running `dataset.from_raw_data()` for the first time against the shared `_clean()` hook)
- **Issue:** `base/data.py:Dataset.from_raw_data()` calls `self._clean(data)`, which defaults to `dataset/cleaning.py:clean_market_data()` -> `validate_schema()`, hardcoded to require lowercase `open/high/low/close/volume` data variables. `SpotKlineDataset._raw_data_to_xr()` produces Title-Case columns (`Open/High/Low/Close/Volume`, per `enums.data.BinanceCSVHeaders.SPOT`) and only lowercases them inside `_to_kunquant()`/`_xr_to_bars()` for downstream consumers. This meant `SpotKlineDataset.from_raw_data()` raised `ValueError: validate_schema: required column(s) missing` on every call, regardless of the dedup fix -- a pre-existing bug from 02-03's integration that had never been exercised end-to-end against `SpotKlineDataset` before this plan's tests did so.
- **Fix:** Added `SpotKlineDataset._clean()`, overriding the base hook to call `validate_schema(data, required_columns=("Open","High","Low","Close","Volume"))` and still run `flag_anomalies(data)` (a documented no-op for spot data, since its price-column list is lowercase-only -- no regression vs. spot's previous behavior, which had no anomaly-flagging integration at all).
- **Files modified:** `dataset/spot.py`
- **Verification:** `tests/test_spot_dataset.py`'s Task 1 tests call `.from_raw_data()` directly and pass; `uv run pytest tests/test_spot_dataset.py tests/test_cleaning.py tests/test_config_paths.py -v` -- 17/17 passed, no regressions.
- **Committed in:** `4e4cc23` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix, Rule 1)
**Impact on plan:** Necessary for Task 1's own stated acceptance criteria (`.from_raw_data()` succeeds) to be achievable at all. Confined to `dataset/spot.py` (already in the plan's `files_modified` list); no scope creep into `_to_kunquant()`, `_xr_to_bars()`, or `dataset/cleaning.py`.

## Issues Encountered
- Task 3's literal instruction ("first day of the most recent fully-completed month" relative to this machine's system clock, 2026-09-05) initially targeted 2026-08, which returned 0/108 matching files from `binance-data-downloader` -- Binance's public archive had not yet published that month's data. Resolved by listing the actual bucket contents directly and using the most recent month genuinely available (2026-07). No fabrication; documented transparently in `02-05-REAL-DATA-CHECK.md`.

## User Setup Required
None - no external service configuration required. `binance-data-downloader` is a pre-existing, user-owned tool invoked via `uvx`; no new credentials or environment variables were introduced.

## Next Phase Readiness
- `SpotKlineDataset` is now safely usable end-to-end via `.from_raw_data().save()`, matching `StockDataset`'s reliability level for the shared `Dataset`/`DataBackend` contract that Phase 3 (factor computation) and beyond will build on.
- `ingest_binance_spot.py` gives any future phase/user a documented, tested way to (re)populate the Binance Zarr store from local CSVs without network access, including from non-convention paths via `--raw-data-dir`.
- No blockers for downstream phases identified.

---
*Phase: 02-multi-market-data-foundation*
*Completed: 2026-09-05*

## Self-Check: PASSED

All created/modified files verified present on disk (`dataset/spot.py`, `ingest_binance_spot.py`,
`tests/test_spot_dataset.py`, `README.md`, `02-05-REAL-DATA-CHECK.md`, this `02-05-SUMMARY.md`).
All 4 commit hashes (`4e4cc23`, `1c18d89`, `44211b2`, `311cca0`) verified present in `git log`.
