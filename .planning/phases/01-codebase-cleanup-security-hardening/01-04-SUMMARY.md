---
phase: 01-codebase-cleanup-security-hardening
plan: 04
subsystem: docs
tags: [readme, documentation, uv]

# Dependency graph
requires:
  - phase: 01-codebase-cleanup-security-hardening (01-01, 01-02, 01-03)
    provides: env-var-based Tiingo key handling, portable config paths, uv-managed dependencies, deduped/fixed scripts
provides:
  - Accurate README.md describing actual module layout, entry points, uv-based install, and required env vars
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified: [README.md]

key-decisions:
  - "Documented QUANTLAB_DATA_DIR, TIINGO_API_KEY, and WANDB_API_KEY as the three environment variables a new contributor needs, matching config/__init__.py and base/model.py behavior"

patterns-established: []

requirements-completed: [CLEAN-03]

# Metrics
duration: 6min
completed: 2026-09-04
---

# Phase 01 Plan 04: README Rewrite Summary

**Rewrote README.md from a stale "Crypto Quant Trading Models" (LSTM/XGBoost) description to an accurate description of the actual quantlab pipeline architecture, module layout, entry points, and uv-based setup.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-09-04T16:20:00Z (approx.)
- **Completed:** 2026-09-04T16:26:06Z
- **Tasks:** 1 completed
- **Files modified:** 1

## Accomplishments
- Replaced all references to non-existent files/classes (`models/lstm_model.py`, `examples/train_models.py`, `ModelConfig`, `XGBoostModel`, `requirements.txt`) with the actual codebase structure
- Documented the real top-level package layout (`base/`, `dataset/`, `factor/`, `label/`, `my_ops/`, `dl_model/`, `ml_model/`, `backtest/`, `vecbt/`, `config/`, `enums/`, `utils/`, `scripts/`) with one-line purpose per package, verified each directory exists on disk
- Documented the real entry-point scripts (`cal.py`, `train_model.py`, `test.py`, `get_binance_instruments.py`, `read_mock_data_sink.py`, `scripts/download_stock_data_from_tiingo.py`) and noted there is no unified CLI
- Documented `uv sync` installation (replacing the old `pip install -r requirements.txt` instructions) and the three environment variables a fresh clone needs: `TIINGO_API_KEY`, `WANDB_API_KEY`, `QUANTLAB_DATA_DIR`
- Documented the actual config dataclasses (`DatasetConfig`, `FactorConfig`, `DLConfig`, `MLConfig`) and model classes (`MLPRegressor`, `RNNRegressor`, `RNNClassifier`)

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite README.md to match the actual codebase** - `1a4eac3` (docs)

**Plan metadata:** (this commit, docs: complete plan)

## Files Created/Modified
- `README.md` - Full rewrite: project description, module layout, entry points, uv-based install, required env vars, config dataclasses, model classes

## Decisions Made
- Documented `QUANTLAB_DATA_DIR`, `TIINGO_API_KEY`, and `WANDB_API_KEY` as the three environment variables a new contributor needs — cross-checked against `config/__init__.py:_data_root()` and `base/model.py:_init_wandb` to confirm actual usage before documenting.

## Deviations from Plan

None - plan executed exactly as written. Task 1 was the plan's only task; content and acceptance criteria matched the codebase as verified by reading `pyproject.toml`, `config/__init__.py`, and grepping class definitions across `base/`, `dataset/`, `factor/`, `label/`, `dl_model/`.

## Issues Encountered

The worktree's local branch (`worktree-agent-adb12be4e637a2568`) was initially behind the expected wave-2 base commit (`faca6b9a`) — it pointed at a stale "Initial commit" with no `.planning/` directory present. Per the plan's `<worktree_branch_check>` step, verified no uncommitted changes existed, then `git reset --hard` to the expected base commit before starting work. This is expected/documented worktree setup behavior, not a plan deviation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- README.md now accurately reflects the post-Wave-1 codebase state (env-var-based Tiingo key, portable `config/__init__.py` paths, `uv`-declared dependencies, deduped Binance parsing logic, fixed `vecbt/bt.py` signal passthrough).
- No blockers for subsequent phases. Phase 01 Plan 04 was the last plan in Wave 2 depending on 01-01/01-02/01-03.

---
*Phase: 01-codebase-cleanup-security-hardening*
*Completed: 2026-09-04*

## Self-Check: PASSED

- FOUND: README.md
- FOUND: .planning/phases/01-codebase-cleanup-security-hardening/01-04-SUMMARY.md
- FOUND commit: 1a4eac3 (docs(01-04): rewrite README.md to match actual codebase)
- FOUND commit: ebc33fd (docs(01-04): complete README rewrite plan)
