# Phase 03.9 Deferred Items

Out-of-scope discoveries logged during plan execution. Not fixed here.

## From 03.9-06

- **Order-dependent full-suite hang at `tests/test_cross_sectional_zscore.py` (macOS).**
  A single-process `uv run pytest -q --ignore=tests/test_factor_hierarchy.py -p no:cacheprovider`
  hung at test 495 of 1461, the first tests of this file (KunQuant compile/stream), with
  0% CPU for more than 10 minutes. The file passes on its own in about 3s. A pytest run in
  another agent's worktree hung at the same point for over an hour, so this predates
  03.9-06 and is unrelated to the NBBO code. The likely suspect is a KunQuant
  multi-thread executor or libomp interaction with a module an earlier test imported
  (compare the torch+xgboost libomp note in CLAUDE.md). Workaround used: run the suite in
  two processes split at that file; the failing ids then match the baseline exactly.
