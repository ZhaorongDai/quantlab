---
phase: 01-codebase-cleanup-security-hardening
plan: 03
subsystem: backtest
tags: [vectorbt, refactor, dead-code, duplicate-code]

# Dependency graph
requires: []
provides:
  - "Working vecbt/bt.py:backtest_from_signals that passes through its own arguments to vbt.Portfolio.from_signals and returns the Portfolio"
  - "Single-source Binance exchange-info fetch/parse logic (get_binance_instruments.py now imports utils.binance instead of duplicating it)"
affects: [phase-6-backtest]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - vecbt/bt.py
    - get_binance_instruments.py

key-decisions:
  - "Kept backtest_from_signals' existing signature unchanged; only fixed the body to pass entries/exits/short_entries/short_exits through and added a return statement"
  - "Removed get_binance_exchange_info()/parse_symbol_info() entirely from get_binance_instruments.py rather than keeping thin wrappers, since utils.binance's underscore-prefixed functions are already this repo's established public surface for this logic"

patterns-established: []

requirements-completed: [CLEAN-02]

# Metrics
duration: 3min
completed: 2026-09-04
---

# Phase 1 Plan 03: Fix Broken/Duplicate Code Summary

**Fixed `vecbt/bt.py:backtest_from_signals`'s zero-argument `vbt.Portfolio.from_signals()` crash and consolidated `get_binance_instruments.py`'s duplicate Binance exchange-info parsing into `utils/binance.py`**

## Performance

- **Duration:** 3 min
- **Started:** 2026-09-04T16:13:39Z
- **Completed:** 2026-09-04T16:16:40Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments
- `vecbt/bt.py:backtest_from_signals` now calls `vbt.Portfolio.from_signals(close, entries=long_entries, exits=long_exits, short_entries=short_entries, short_exits=short_exits)` and returns the resulting `Portfolio` object, instead of crashing on a zero-argument call
- `get_binance_instruments.py` no longer maintains its own copy of `get_binance_exchange_info()`/`parse_symbol_info()`; both call sites (`update_instruments_config`, `get_all_usdt_pairs`) now import and call `utils.binance._get_binance_exchange_info`/`_parse_symbol_info`

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix vecbt/bt.py:backtest_from_signals to pass its own parameters through** - `688422b` (fix)
2. **Task 2: Consolidate duplicate Binance exchange-info parsing into utils/binance.py** - `43f86a2` (refactor)

_No TDD test/feat split — Task 1 was marked `tdd="true"` in frontmatter but its `<verify>` block is a single smoke-test script, not a separate RED/GREEN cycle; verified via the plan's automated command directly._

## Files Created/Modified
- `vecbt/bt.py` - `backtest_from_signals` now passes close/entries/exits/short_entries/short_exits through to `vbt.Portfolio.from_signals` and returns the Portfolio
- `get_binance_instruments.py` - removed duplicate `get_binance_exchange_info`/`parse_symbol_info`, imports `utils.binance._get_binance_exchange_info`/`_parse_symbol_info` instead; `update_instruments_config`/`get_all_usdt_pairs`/CLI behavior unchanged

## Decisions Made
- Kept `backtest_from_signals`' public signature exactly as specified in the plan (`close, long_entries, long_exits, short_entries, short_exits, index`) — only the broken body changed.
- Fully removed the duplicate functions from `get_binance_instruments.py` (no re-export shim), matching the plan's explicit instruction and keeping a true single source of truth.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Task 2's plan-specified verify command (`uv run python -c "import get_binance_instruments; print('OK')"`) fails in this worktree's environment with `ModuleNotFoundError: No module named 'yaml'` (and, once `yaml` is stubbed, `No module named 'loguru'`). This is a pre-existing environment gap, not caused by this plan's changes: `get_binance_instruments.py` already imported `yaml` before this plan touched it, and `utils/binance.py` (unmodified) already imported `loguru`. `pyproject.toml` currently declares zero dependencies and the checked-in `uv.lock` is stale — reconciling `pyproject.toml`/`uv.lock` with the ~20 actually-imported third-party packages (including PyYAML and loguru) is explicitly owned by parallel plan 01-02 ("Declare and lock actual dependencies via uv add"), which is out of this plan's scope and modifies files (`pyproject.toml`, `uv.lock`) this plan does not touch.

Verified the code change is functionally correct via alternate means:
- `grep -cE "^def (get_binance_exchange_info|parse_symbol_info)\("` returns `0` (no duplicate definitions remain)
- `grep -c "from utils.binance import"` returns `1`
- `python3 -m py_compile get_binance_instruments.py` succeeds (no syntax errors)
- Ran `uv run python` with `yaml`/`loguru` stubbed via `sys.modules` injection (installed-package-independent) confirming `get_binance_instruments` imports successfully, exposes `_get_binance_exchange_info`/`_parse_symbol_info` from `utils.binance`, and no longer exposes the old duplicate names — i.e. no `NameError`/`ImportError` originating from this plan's own code changes

Once plan 01-02 lands `pyproject.toml`'s dependency declarations, the plan's literal verify command will pass unmodified — no further code change is needed from this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `vecbt/bt.py:backtest_from_signals` is now usable and ready for the Phase 6 vectorized-backtest work (CLAUDE.md's "回测技术栈" constraint: vectorbt is the primary backtest path).
- `get_binance_instruments.py` and `utils/binance.py` now share one implementation; future changes to Binance filter-parsing only need to happen in `utils/binance.py`.
- No blockers for this plan's scope. The `import get_binance_instruments` smoke test will only pass end-to-end once plan 01-02's dependency reconciliation (`pyproject.toml`/`uv sync`) is merged — tracked there, not a blocker for this plan's completion.

---
*Phase: 01-codebase-cleanup-security-hardening*
*Completed: 2026-09-04*

## Self-Check: PASSED

- FOUND: vecbt/bt.py
- FOUND: get_binance_instruments.py
- FOUND: .planning/phases/01-codebase-cleanup-security-hardening/01-03-SUMMARY.md
- FOUND: 688422b (Task 1 commit)
- FOUND: 43f86a2 (Task 2 commit)
- FOUND: 46c7ad3 (SUMMARY commit)
