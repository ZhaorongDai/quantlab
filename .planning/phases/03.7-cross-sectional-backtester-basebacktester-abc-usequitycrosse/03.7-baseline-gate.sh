#!/usr/bin/env bash
# Phase 03.7 full-suite gate: "no failures beyond the pre-phase baseline".
#
# The suite is NOT green before this phase, so "exit 0" is not the gate.
# Baseline measured 2026-09-14 by the 03.7 planner, on BOTH the working tree
# carrying the user's uncommitted edits AND a clean `git archive HEAD` copy:
# identical 56 failing node ids, with the per-file counts below, plus a
# collection error in tests/test_factor_hierarchy.py (imports the renamed
# SpotReturn), which is ignored here exactly as the baseline run ignored it.
#
# Fails (exit 1) when:
#   - any `ERROR tests/...` line appears (a new collection/setup error);
#   - no "N passed" summary line appears (pytest did not actually run);
#   - a file outside the baseline set has a failure;
#   - a baseline file has MORE failures than its baseline count.
# Fewer failures than baseline passes (a worktree may legitimately fix one).
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

uv run pytest tests/ -q -p no:cacheprovider \
  --ignore=tests/test_factor_hierarchy.py -rfE >"$OUT" 2>&1
tail -1 "$OUT"

if grep -qE '^ERROR tests/' "$OUT"; then
  echo "NEW ERROR(S):"
  grep -E '^ERROR tests/' "$OUT"
  exit 1
fi

if ! grep -qE '[0-9]+ passed' "$OUT"; then
  echo "NO 'passed' SUMMARY LINE: pytest did not run the suite"
  exit 1
fi

{ grep -E '^FAILED tests/' "$OUT" || true; } \
  | sed -E 's#^FAILED (tests/[^:]+)::.*#\1#' \
  | sort | uniq -c \
  | awk '
BEGIN {
  b["tests/test_chunked_ingest.py"] = 2
  b["tests/test_data_dir_cli.py"] = 10
  b["tests/test_entry_point_contracts.py"] = 1
  b["tests/test_factor_kunquant.py"] = 4
  b["tests/test_ingest_conversion_gate.py"] = 6
  b["tests/test_ingest_shells.py"] = 14
  b["tests/test_ingest_tiingo_universe_wiring.py"] = 11
  b["tests/test_spot_dataset.py"] = 2
  b["tests/test_universe.py"] = 1
  b["tests/test_volume_guard.py"] = 5
  bad = 0
}
{
  base = ($2 in b) ? b[$2] : 0
  if ($1 > base) {
    print "NEW FAILURE(S) in " $2 ": " $1 " failed (baseline " base ")"
    bad = 1
  }
}
END {
  if (bad) exit 1
  print "03.7 baseline gate: no failures beyond the 56-test pre-phase baseline"
}'
