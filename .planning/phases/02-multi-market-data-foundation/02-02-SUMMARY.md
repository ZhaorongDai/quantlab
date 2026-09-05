---
phase: 02-multi-market-data-foundation
plan: 02
subsystem: database
tags: [dataclass, config, xarray, zarr, market-data]

# Dependency graph
requires:
  - phase: 02-multi-market-data-foundation (plan 01)
    provides: pytest test infrastructure, XrBackend overwrite fix
provides:
  - "DatasetConfig.market/.frequency required fields (Market/Frequency Literal aliases)"
  - "AcquisitionConfig dataclass (credential-free) for Wave 3's TiingoAcquisition"
  - "config/__init__.py:_market_data_root()/_market_downloads_root() shared path helpers"
  - "config/__init__.py:stock_kline_config() and stock_acquisition_config() factories"
  - "data/{market}/{frequency}/{name}.zarr storage path convention, applied to spot_kline_config() and stock_kline_config()"
affects: [02-multi-market-data-foundation (Wave 3: Tiingo acquisition, Binance retrofit)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Config factories derive all paths from market+frequency via two shared private helpers, never hardcoded market-specific path segments"
    - "Credential-free serializable config dataclasses (AcquisitionConfig mirrors DatasetConfig's to_dict()/asdict() shape but has no api_key/token field, forcing credentials to stay in os.environ-only code)"

key-files:
  created: [tests/test_config_paths.py]
  modified: [enums/data.py, base/config.py, config/__init__.py, test.py]

key-decisions:
  - "Market/Frequency Literal token sets locked as [\"us_equity\",\"crypto_spot\"] / [\"1d\",\"1m\",\"tick\"] per 02-RESEARCH.md Assumptions Log A2"
  - "AcquisitionConfig deliberately excludes any credential-shaped field; TIINGO_API_KEY will be read directly from os.environ inside Wave 3's TiingoAcquisition, never threaded through a serializable config object (mirrors Phase 1's SEC-01 fix)"

patterns-established:
  - "New market/frequency combinations require zero changes to existing factory functions — only new market/frequency keyword values passed to shared helpers"

requirements-completed: [DATA-01, DATA-02, DATA-03]

# Metrics
duration: 5min
completed: 2026-09-04
---

# Phase 02 Plan 02: Market/Frequency Config Contract Summary

**DatasetConfig/AcquisitionConfig gain required market/frequency fields, and every config factory now derives storage paths from a shared `data/{market}/{frequency}/{name}.zarr` convention, closing the missing `stock_kline_config()` gap.**

## Performance

- **Duration:** ~5 min (task execution only; excludes environment/context read time)
- **Started:** 2026-09-05T02:15:00Z (approx)
- **Completed:** 2026-09-05T02:20:32Z
- **Tasks:** 2
- **Files modified:** 4 (+1 created)

## Accomplishments
- `DatasetConfig` and new `AcquisitionConfig` both carry required `market: Market`/`frequency: Frequency` fields backed by shared `Literal` type aliases in `enums/data.py`
- Every config factory (`spot_kline_config`, new `stock_kline_config`, new `stock_acquisition_config`) derives paths via two shared private helpers (`_market_data_root`, `_market_downloads_root`) — no more hardcoded `spot/monthly/klines`-style path segments under `data/`
- `stock_kline_config()` and `stock_acquisition_config()` now exist, closing D-09's gap and giving Wave 3's Tiingo acquisition component a config contract to build on
- `test.py`'s hand-built `DatasetConfig(...)` call updated with `market`/`frequency` kwargs so it keeps constructing without a `TypeError`
- `cal.py`, `train_model.py`, `backtest/test_strategy.py`, `test_nt.ipynb` all remain syntactically valid and their indirect `DatasetConfig` construction paths (via `spot_kline_config()`'s new defaulted params) still resolve — confirmed via `ast.parse`-based static smoke-checks

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Market/Frequency type aliases and DatasetConfig/AcquisitionConfig fields** - `63033f5` (feat)
2. **Task 2: Retrofit config factories to the data/{market}/{frequency}/... path convention + add stock_kline_config()** - `c18e37c` (test, RED) then `f2dabf3` (feat, GREEN)

**Plan metadata:** committed separately after this SUMMARY (worktree mode — orchestrator handles STATE.md/ROADMAP.md centrally)

_Note: Task 2 used TDD (RED test commit, then GREEN implementation commit); no REFACTOR commit was needed._

## Files Created/Modified
- `enums/data.py` - added `Market`/`Frequency` `Literal` type aliases
- `base/config.py` - added required `market`/`frequency` fields to `DatasetConfig`; added new `AcquisitionConfig` dataclass (no credential field)
- `config/__init__.py` - added `_market_data_root()`/`_market_downloads_root()` helpers; retrofitted `spot_kline_config()`; added `stock_kline_config()` and `stock_acquisition_config()`
- `test.py` - added `market="us_equity", frequency="1d"` kwargs to the existing `DatasetConfig(...)` call
- `tests/test_config_paths.py` (created) - 4 tests covering the path convention, credential-free `AcquisitionConfig`, and the `alpha101_config()` regression

## Decisions Made
- `AcquisitionConfig` deliberately has no `api_key`/`credential`/`token`/`secret` field (verified via grep in Task 1's acceptance criteria) — Wave 3's `TiingoAcquisition` will read `TIINGO_API_KEY` directly from `os.environ`, consistent with the Phase 1 SEC-01 credential-leak fix.
- `alpha101_config()`, `alpha158_config()`, `spot_label_config()` left untouched — they call `spot_kline_config(symbols=symbols)` with no `market`/`frequency` override, which now resolves via the new defaulted params with zero edits needed (proven by Test 4 in `tests/test_config_paths.py`).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The worktree's HEAD had diverged from the expected wave-2 base commit (`8a9851c8af1b813a89223abf355e9f915618c394`) at agent startup — `git merge-base` found no common ancestor with the worktree's single "Initial commit". Per the mandatory `<worktree_branch_check>` protocol, since the working tree was clean (no uncommitted work to lose), `git reset --hard` to the expected base commit was performed to correct this before any task work began. No plan-related code was affected.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Wave 3's Tiingo acquisition component (`TiingoAcquisition`) can now consume `stock_acquisition_config()` for its `raw_data_dir_path`/`watermark_path`/`market`/`frequency` inputs.
- Wave 3's Binance retrofit work can rely on `spot_kline_config()`'s new `market`/`frequency` parameters already being in place.
- No blockers identified for downstream Wave 3 work.

---
*Phase: 02-multi-market-data-foundation*
*Completed: 2026-09-04*
