---
phase: 01-codebase-cleanup-security-hardening
plan: 01
subsystem: security
tags: [tiingo, credentials, env-vars, portability, pathlib]

# Dependency graph
requires: []
provides:
  - "scripts/download_stock_data_from_tiingo.py reads TIINGO_API_KEY from the environment with a fail-fast RuntimeError guard, no hardcoded key literal"
  - "config/__init__.py's four factory functions (spot_kline_config, alpha101_config, alpha158_config, spot_label_config) build paths from a single _data_root() helper, configurable via QUANTLAB_DATA_DIR"
  - "test.py and train_model.py no longer contain per-developer-machine absolute paths; both configurable via env vars with repo-relative defaults"
affects: [02-data-infrastructure, 07-quality-and-testing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_data_root() / os.environ.get(...) pattern for env-var-configurable, repo-relative default paths (QUANTLAB_DATA_DIR, QUANTLAB_NASDAQ_STOCKS_PARQUET, QUANTLAB_CHECKPOINT_PATH)"

key-files:
  created: []
  modified:
    - scripts/download_stock_data_from_tiingo.py
    - config/__init__.py
    - test.py
    - train_model.py

key-decisions:
  - "Kept the explicit config[\"api_key\"] = os.environ[\"TIINGO_API_KEY\"] assignment (rather than passing os.environ directly to TiingoClient) so the intent is self-documenting in the script, per the plan's SEC-01 guidance"
  - "Deleted the dead commented-out config-loading block in train_model.py referencing a stale crypto_quant path instead of rewriting it to the new env-var pattern (smaller diff, code was already unreachable)"

patterns-established:
  - "Env-var-configurable path with repo-relative default: Path(os.environ.get(\"VAR\", <repo-relative-default>))"

requirements-completed: [SEC-01, CLEAN-02]

# Metrics
duration: 15min
completed: 2026-09-04
---

# Phase 1 Plan 1: Remove Leaked Credential and Hardcoded Paths Summary

**Tiingo API key now read exclusively from `TIINGO_API_KEY` env var with fail-fast RuntimeError; all four `config/__init__.py` factories plus `test.py`/`train_model.py` now derive paths from `QUANTLAB_DATA_DIR`/`QUANTLAB_NASDAQ_STOCKS_PARQUET`/`QUANTLAB_CHECKPOINT_PATH` env vars with repo-relative defaults, eliminating every remaining `/home/zhrdai/...` hardcoded path in Phase 1's scope.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-09-04T16:00:00Z
- **Completed:** 2026-09-04T16:15:02Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Removed the leaked Tiingo API key literal from `scripts/download_stock_data_from_tiingo.py`; script now fails fast with a clear `RuntimeError` if `TIINGO_API_KEY` is unset, and its `nasdaq_stocks.parquet` lookup is configurable via `QUANTLAB_NASDAQ_STOCKS_PARQUET`
- Introduced a single `_data_root()` helper in `config/__init__.py`, consumed by all four config factory functions, replacing every `/home/zhrdai/projects/crypto_quant/...` absolute path with a `QUANTLAB_DATA_DIR`-configurable, repo-relative default
- Fixed `test.py`'s `StockDataset` paths and `train_model.py`'s model checkpoint load path to use env-var-configurable, repo-relative defaults instead of stale per-developer absolute paths (including one pointing at a different, stale project name `crypto_quant`)
- Repo-wide grep confirmed no other hardcoded-secret-shaped literals exist beyond the Tiingo key (per threat model T-01-01-02)

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove hardcoded Tiingo API key and hardcoded nasdaq_stocks.parquet path** - `1648674` (fix)
2. **Task 2: Make config/__init__.py path factories configurable, not per-machine hardcoded** - `d0cb064` (fix)
3. **Task 3: Fix hardcoded per-developer paths in test.py and train_model.py** - `e834b11` (fix)

_Note: SUMMARY.md commit is handled separately by the worktree executor's metadata commit step._

## Files Created/Modified
- `scripts/download_stock_data_from_tiingo.py` - Reads `TIINGO_API_KEY` from env with explicit guard; removed dead premature `TiingoClient()` call; `nasdaq_stocks.parquet` path configurable via `QUANTLAB_NASDAQ_STOCKS_PARQUET`
- `config/__init__.py` - Added `_data_root()` helper (`QUANTLAB_DATA_DIR` env var or repo-relative `data/` default); all four factory functions (`spot_kline_config`, `alpha101_config`, `alpha158_config`, `spot_label_config`) now build paths from it
- `test.py` - `StockDataset` `raw_data_dir_path`/`zarr_file_path` built from `QUANTLAB_DATA_DIR`-configurable, repo-relative root
- `train_model.py` - Checkpoint path configurable via `QUANTLAB_CHECKPOINT_PATH`; deleted dead commented-out block referencing a stale `crypto_quant` absolute path

## Decisions Made
- Kept `config["api_key"] = os.environ["TIINGO_API_KEY"]` as an explicit line (rather than relying on `TiingoClient()`'s implicit env var pickup) to keep the source self-documenting, per plan instructions
- Deleted (rather than rewrote) the dead commented-out config-loading block in `train_model.py` since it was unreachable code referencing a stale project name — smaller diff, no loss of functionality

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. All acceptance criteria verified via automated grep checks and standalone Python logic simulations (since the project's actual dependencies — KunQuant, torch, nautilus_trader, etc. — are not installed in this environment and importing the modules directly was out of scope for this plan).

## User Setup Required

None - no external service configuration required by this plan. Note: the previously-leaked Tiingo API key itself still needs to be revoked/rotated in the Tiingo dashboard by the user (tracked as a pre-existing blocker in STATE.md, addressed separately in Phase 1's later git-history-reset work, not in this plan).

## Next Phase Readiness
- `scripts/download_stock_data_from_tiingo.py`, `config/__init__.py`, `test.py`, and `train_model.py` are now portable across machines with zero source edits (only env vars need to be set): `TIINGO_API_KEY`, `QUANTLAB_DATA_DIR`, `QUANTLAB_NASDAQ_STOCKS_PARQUET`, `QUANTLAB_CHECKPOINT_PATH`
- No blockers introduced by this plan. Git history still contains the leaked key in earlier commits — that is addressed by a separate, later Phase 1 plan (git history reset), not this one.

---
*Phase: 01-codebase-cleanup-security-hardening*
*Completed: 2026-09-04*

## Self-Check: PASSED

- FOUND: scripts/download_stock_data_from_tiingo.py
- FOUND: config/__init__.py
- FOUND: test.py
- FOUND: train_model.py
- FOUND commit: 1648674
- FOUND commit: d0cb064
- FOUND commit: e834b11
- FOUND commit: 07c8126
