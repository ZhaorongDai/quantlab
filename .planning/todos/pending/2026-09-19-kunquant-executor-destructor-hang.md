---
title: "Intermittent full-suite hang: KunQuant MultiThreadExecutor destructor deadlock"
area: "factor / tests"
created: 2026-09-19
priority: high
---

## Problem

The full pytest suite intermittently hangs forever at 0% CPU at the start of
`tests/test_cross_sectional_zscore.py` (seen 3 times on 2026-09-19: once in the main tree for 66 min,
twice in executor worktrees for > 1 h). The file passes on its own (9 passed in 2.9 s).

`sample` of a hung process (evidence: `.planning/debug/evidence/2026-09-19-kunquant-executor-hang-sample.txt`):

- main thread: `KunRunner.abi3.so` → `kun::MultiThreadExecutor::~MultiThreadExecutor()` →
  `std::thread::join()` → `__ulock_wait` (holding the GIL)
- ~40 KunQuant worker threads parked in `_pthread_cond_wait` (never woken to exit)
- one Python thread in `take_gil`

So the executor destructor joins workers that are waiting on a condition variable that was never
signalled (lost wakeup / shutdown race), not the macOS torch+xgboost OpenMP clash.

## Ideas

- Keep one executor per process (module-level cache) instead of creating/destroying one per `cal()`.
- Or release the GIL / explicitly shut the executor down before dropping it; check KunQuant upstream.
- Test-side stopgap: pytest-timeout on factor tests so a hang fails fast instead of blocking the gate.
