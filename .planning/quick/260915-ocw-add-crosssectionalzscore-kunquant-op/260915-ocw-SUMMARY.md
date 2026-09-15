---
phase: quick-260915-ocw
plan: 01
status: complete
subsystem: factor-ops
tags: [kunquant, cross-sectional, normalization, zscore]
requires: []
provides:
  - "quantlab/my_ops/preprocess.py:CrossSectionalZScore (GenericCrossSectionalOp, TS + STREAM)"
affects:
  - example/factor.md
tech-stack:
  added: []
  patterns:
    - "Custom KunQuant cross-sectional op via GenericCrossSectionalOp.generate_body C++ loop (no attrs)"
key-files:
  created:
    - tests/test_cross_sectional_zscore.py
  modified:
    - quantlab/my_ops/preprocess.py
    - example/factor.md
decisions:
  - "CrossSectionalZScore is a GenericCrossSectionalOp, not a CompositiveOp: decompose can only unfold into time-series ops"
  - "Degenerate cross-sections (<2 valid values or sd == 0) produce a whole NaN row; the op does no fillna"
  - "The start>0 KunQuant 0.1.11 CrossSectionalDataHolder bug is documented, not asserted (non-deterministic); only start=0 is tested"
  - "No factor class wired to the op; wiring stays deferred to ARCH-02 (US-equity factors still raw per D-09)"
metrics:
  duration: "3m27s"
  started: "2026-09-15T21:40:12Z"
  completed: "2026-09-15T21:43:39Z"
plan_head_before: 5f483214e5802a6149f132a99d5f16b676ba68ef
actuals:
  tokens: 3904     # diff bytes/4 over the realized diff (15614 bytes, 3 files)
  tasks: 3
  commits: 3       # git rev-list --count 5f48321..HEAD
---

# Phase quick-260915-ocw Plan 01: CrossSectionalZScore KunQuant op Summary

NaN-aware cross-sectional Z-score (`(x - mean) / sample std`, ddof=1) added as a KunQuant `GenericCrossSectionalOp` with a C++ loop body. Locked against pandas in both TS batch (start=0) and STREAM layouts, with its four pitfalls recorded in a Chinese docstring. `example/factor.md` now says the op exists but is not wired into any factor class.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Tracer: op + batch start=0 pandas-match test | 643ae1b | quantlab/my_ops/preprocess.py, tests/test_cross_sectional_zscore.py |
| 2 | Stream layout, intermediate node, TS consumer, degenerate rows, structural lock | 921fff0 | tests/test_cross_sectional_zscore.py |
| 3 | Dated doc update in example/factor.md | 6a8f8dd | example/factor.md |

## Verification

- RED confirmed before implementation: `tests/test_cross_sectional_zscore.py` failed at collection with `ImportError: cannot import name 'CrossSectionalZScore'` (resolved against the worktree copy of preprocess.py).
- Tracer feedback gate: after Task 1, 1 passed; re-run end to end before expanding.
- `uv run pytest tests/test_cross_sectional_zscore.py` gives 9 passed in 2.60s: 1 tracer, 2 batch, 3 stream (parametrized), 1 degenerate rows, 1 moments, 1 structural. The 1e-4 moments tolerance held on real output with no widening.
- `uv run pytest tests/test_factor_kunquant.py` gives 3 failed, 7 passed, the three baseline failures by name: `test_stock_to_kunquant_synthesizes_amount_as_adjusted_dollar_volume`, `test_stock_to_kunquant_without_amount_leaves_arrays_unchanged`, `test_alpha101_stock_bugfix_batch_cal_returns_xarray_dataset`.
- The test file has exactly 2 `cfake.compileit(` calls.
- `git status --porcelain -- quantlab/factor quantlab/base/factor.py tests/test_factor_kunquant.py` is empty.
- `git diff` of `quantlab/my_ops/preprocess.py` is add-only, so `WindowedZScore` is byte-identical.
- example/factor.md greps: `CrossSectionalZScore`=2, `2026-09-15`=1, xarray snippet line=1 (unchanged), `ddof=1`=2. The diff touches only the "现在还没有的东西" subsection.
- quantlab/base/model.py and quantlab/base/backtest.py (owned by 260915-o5y) were not touched. `git stash` was never used.
- Tests ran with the main checkout's venv (`UV_PROJECT_ENVIRONMENT=... UV_NO_SYNC=1`). `import quantlab` resolved to the worktree copy. cfake builds went to a system `tempfile.mkdtemp` dir, which is removed after load.

## Deviations from Plan

None - plan executed exactly as written.

## Notes

- The plan's `<verify>` commands `cd` into the main checkout. They were run from the worktree root instead, per the orchestrator's worktree-isolation instruction. The commands and assertions are otherwise identical.
- Observation, out of scope and not edited: the "常见坑" item 1 heading in example/factor.md says the SIMD alignment limit applies "KunQuant 批量模式下". The new op's docstring, following the plan's measured facts, says it applies to both TS and STREAM layouts. A future doc pass may want to widen that heading.

## Threat Flags

None. `generate_body` is a constant C++ literal with no attrs or runtime input spliced in (T-260915-ocw-01, locked by the structural test). The start>0 bug is mitigated by documentation plus the start=0-only caller (T-260915-ocw-02).

## Known Stubs

None.

## Self-Check: PASSED

- FOUND: quantlab/my_ops/preprocess.py (class CrossSectionalZScore)
- FOUND: tests/test_cross_sectional_zscore.py
- FOUND: example/factor.md (2026-09-15 update)
- FOUND: 643ae1b, 921fff0, 6a8f8dd in `git log`
