---
phase: 01-codebase-cleanup-security-hardening
plan: 02
subsystem: infra
tags: [uv, dependency-management, pyproject, packaging, kunquant, vectorbt, plotly]

# Dependency graph
requires: []
provides:
  - "pyproject.toml declaring the ~20 third-party packages actually imported by the codebase"
  - "regenerated uv.lock resolved against the quantlab project (stale crypto-quant lock removed)"
  - "working `uv sync` producing a `.venv` with all first-party modules importable"
  - "ModelBackend abstract base class (base/backend.py), completing the DataBackend/ModelBackend persistence-abstraction pair"
affects: [02-data-infrastructure, 03-factor-engineering, 04-return-models, 05-portfolio-optimization, 06-backtesting, 07-quality-hardening]

# Tech tracking
tech-stack:
  added: [uv-managed pyproject.toml dependency declarations, KunQuant (PyPI 0.1.11), nautilus-trader 1.231.0, vectorbt 0.28.2, plotly (pinned <6 for vectorbt compatibility)]
  patterns: ["ModelBackend ABC mirrors the existing DataBackend ABC pattern in base/backend.py (property + abstractmethod read/write/to_internal)"]

key-files:
  created: [pyproject.toml, .planning/phases/01-codebase-cleanup-security-hardening/01-02-SUMMARY.md]
  modified: [uv.lock, base/backend.py]

key-decisions:
  - "Created pyproject.toml from scratch (it did not exist in this worktree/git history at all -- only as an untracked file in the main checkout) rather than editing an existing stale one"
  - "Pinned plotly<6 because vectorbt 0.28.2 executes a module-level `scattermapbox` Plotly trace registration at import time, and plotly 6.0+ removed that trace entirely, raising ValueError on import"
  - "Added a minimal ModelBackend ABC to base/backend.py (mirroring DataBackend) rather than restructuring ml_model/backend.py, since MlBackend already implements the exact contract needed"

patterns-established:
  - "Third-party dependency versions may be constrained below `uv add`'s default lower-bound-only resolution when a downstream package (vectorbt) has a hard incompatibility with a newer major version of one of its own dependencies (plotly) -- pin explicitly and document the reason in the commit message"

requirements-completed: [CLEAN-04]

# Metrics
duration: 8min
completed: 2026-09-04
---

# Phase 1 Plan 2: Reconcile pyproject.toml/uv.lock with Actual Dependencies Summary

**Rebuilt `pyproject.toml` from scratch (it didn't exist in this worktree) with 20 `uv add`-resolved dependencies against the `quantlab` project, regenerated `uv.lock` to replace the stale 3-package `crypto-quant` lock, and fixed two blockers (a missing `ModelBackend` ABC and a vectorbt/plotly version incompatibility) so all 27 first-party modules import cleanly under `uv sync`.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-09-04T16:11:06Z
- **Completed:** 2026-09-04T16:19:01Z
- **Tasks:** 2 completed
- **Files modified:** 3 (pyproject.toml created, uv.lock regenerated, base/backend.py extended)

## Accomplishments
- `pyproject.toml` now declares all ~20 packages actually imported by the codebase (numpy, pandas, polars, xarray, torch, KunQuant, nautilus-trader, vectorbt, wandb, loguru, joblib, tqdm, bottleneck, requests, pyyaml, tiingo, psutil, plotly, scikit-learn, zarr)
- `uv.lock` regenerated against the `quantlab` project (124 locked packages), no remaining reference to the old `crypto-quant` project name
- `uv sync` exits 0 and produces a working `.venv`
- All 27 first-party top-level modules (`base.*`, `dataset.*`, `factor.*`, `label.*`, `dl_model.*`, `ml_model.backend`, `backtest.test_strategy`, `vecbt.bt`, `utils.*`, `enums.*`, `config`) import without error under the resulting environment

## Task Commits

Each task was committed atomically:

1. **Task 1: Declare and lock actual dependencies via uv add** - `1321a73` (feat)
2. **Task 2: Verify every first-party module imports cleanly** - `b2ea47b` (fix)

**Plan metadata:** (pending — final metadata commit, see below)

## Files Created/Modified
- `pyproject.toml` - Created fresh; declares `quantlab` project metadata and the 20-package dependency list resolved via `uv add`
- `uv.lock` - Regenerated against `quantlab` (124 locked packages, replacing the stale 3-package `crypto-quant` lock)
- `base/backend.py` - Added `ModelBackend` ABC (mirrors `DataBackend`: `get_model`/`read`/`write`/`to_internal` abstract methods + `model` property), fixing a pre-existing broken import in `ml_model/backend.py`

## Decisions Made
- **pyproject.toml did not exist in this worktree** (git history never committed it — it only existed as an untracked file in the main repo checkout, alongside `main.py` and `.planning/codebase/`). Created it from scratch with `[project] name = "quantlab"`, `requires-python = ">=3.13"`, matching the metadata-only shape described in the plan's context (no `[tool.*]` sections).
- **KunQuant resolved directly from PyPI** (`kunquant==0.1.11`) — the plan's git-dependency fallback (`https://github.com/Menooker/KunQuant`) was not needed.
- **Pinned `plotly<6`** — `vectorbt` 0.28.2 calls `vbt._settings.reset_theme()` at import time, which registers a Plotly template containing a `scattermapbox` trace. Plotly 6.0+ removed `scattermapbox` (renamed/consolidated into `scattermap`), so importing `vectorbt` under plotly 7.0.0 raised `ValueError: Invalid property specified for object of type plotly.graph_objs.layout.template.Data: 'scattermapbox'`. Downgraded to plotly 5.24.1 (the latest 5.x release), which resolved the import cleanly. This is a real, load-bearing version constraint, not a cosmetic downgrade — `uv add plotly` alone would silently break every module that transitively imports `vecbt.bt`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] pyproject.toml did not exist in this worktree**
- **Found during:** Task 1 (read_first step)
- **Issue:** The plan's `<read_first>` step assumed `pyproject.toml` existed with `dependencies = []` (per STACK.md, which also does not exist in this worktree — `.planning/codebase/` was never committed to git; it only existed as an untracked directory in the main checkout at plan-authoring time). In this isolated worktree, neither file was present at all.
- **Fix:** Created `pyproject.toml` from scratch with the exact metadata shape (`name`, `version`, `description`, `readme`, `requires-python`, empty `dependencies`) that STACK.md/the plan described, then proceeded with `uv add` as planned.
- **Files modified:** pyproject.toml
- **Verification:** `uv add` succeeded against the new file; `uv sync` exits 0.
- **Committed in:** `1321a73` (Task 1 commit)

**2. [Rule 1 - Bug] Added missing `ModelBackend` ABC to fix broken import**
- **Found during:** Task 2 (first-party import verification)
- **Issue:** `ml_model/backend.py` does `from base.backend import ModelBackend` and subclasses it, but `base/backend.py` only defined `DataBackend` — `ModelBackend` was never defined anywhere in the codebase. This is a pre-existing bug unrelated to the dependency changes in Task 1, but it directly blocked Task 2's "all first-party modules import cleanly" acceptance criterion.
- **Fix:** Added a minimal `ModelBackend(ABC)` class to `base/backend.py`, mirroring the existing `DataBackend` pattern exactly (`model` property + `get_model`/`read`/`write`/`to_internal` abstract methods) — matching the contract `MlBackend` already implements.
- **Files modified:** base/backend.py
- **Verification:** `uv run python -c "import ml_model.backend"` (and the full 27-module import command) now succeeds.
- **Committed in:** `b2ea47b` (Task 2 commit)

**3. [Rule 3 - Blocking] Pinned `plotly<6` to fix vectorbt import failure**
- **Found during:** Task 2 (first-party import verification)
- **Issue:** `vecbt.bt` imports `vectorbt`, which at import time (module load, not call time) registers a Plotly template referencing the `scattermapbox` trace type. `uv add plotly` (no version constraint) resolved plotly 7.0.0, which removed `scattermapbox`, causing `ValueError` on import — a hard blocker for Task 2's acceptance criterion, not a bug in this plan's own code.
- **Fix:** `uv add "plotly<6"`, which resolved to plotly 5.24.1 (latest compatible with vectorbt's Mapbox-trace registration).
- **Files modified:** pyproject.toml, uv.lock
- **Verification:** Full 27-module import command exits 0 with no traceback; re-ran Task 1's `uv sync` verification afterward to confirm it still passes (uv.lock still has no `crypto-quant` reference, 124 `name = ` entries ≥ 19 floor).
- **Committed in:** `b2ea47b` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (1 missing-file/Rule 3, 1 bug/Rule 1, 1 blocking-version-conflict/Rule 3)
**Impact on plan:** All three were necessary preconditions for the plan's stated success criteria ("uv sync exits 0"; "every first-party module imports without error"). None expand scope beyond making the existing prototype's declared imports actually resolvable and loadable. No architectural changes were made.

## Issues Encountered
- `.planning/codebase/STACK.md` and `.planning/codebase/STRUCTURE.md`, referenced in the plan's `<context>` block, do not exist in this worktree (never committed to git — only present as untracked files in the main repo checkout at plan-authoring time). Proceeded without them since Task 1's `<action>` already enumerates every package name explicitly inline; no information was missing for execution.

## User Setup Required

None - no external service configuration required. (Note: Phase 1's separate Tiingo API key rotation, tracked in STATE.md Blockers/Concerns, is out of scope for this plan and is being handled by a sibling plan in this wave/phase.)

## Next Phase Readiness
- `uv sync` now produces a working, importable environment for every first-party module in the repo — this was the hard precondition blocking all of Phase 2+ (CLEAN-04).
- `pyproject.toml`/`uv.lock` are now the source of truth for the `quantlab` project's dependencies; future plans should use `uv add`/`uv remove` to modify them rather than hand-editing.
- No blockers for downstream phases from this plan's scope.

---
*Phase: 01-codebase-cleanup-security-hardening*
*Completed: 2026-09-04*

## Self-Check: PASSED

- FOUND: pyproject.toml
- FOUND: uv.lock
- FOUND: base/backend.py
- FOUND: .planning/phases/01-codebase-cleanup-security-hardening/01-02-SUMMARY.md
- FOUND commit: 1321a73
- FOUND commit: b2ea47b
- FOUND commit: 21de639
