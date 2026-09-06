---
schema_version: 1
open_count: 4
waived_count: 0
fixed_count: 0
total_count: 4
last_updated: 2026-09-06T05:16:52.451Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 03.1 | deviation | .planning/phases/03.1-index-historical-constituents/03.1-01-PLAN.md | 454 | Plan <verification> grep 'import Dataset' is over-broad — it also matches the live 'import DatasetConfig'; the word-boundary form was run instead | open |  | 2026-09-06T02:57:36.166Z |  |
| 2 | quick-260906-0iy | unrun-verify | ingest_us_equity.py |  | Plan verification step 5 (TIINGO_API_KEY=... ingest_us_equity.py --limit 5) not run: no Tiingo credential available in this environment; offline tests cover resume/isolation but the real 5-symbol backfill is unexercised | open |  | 2026-09-06T04:46:32.180Z |  |
| 3 | quick-260906-13w | unrun-verify | ingest_us_equity.py |  | Real full-market --to-zarr backfill not run: TIINGO_API_KEY unset, no raw parquet to convert | open |  | 2026-09-06T05:16:52.332Z |  |
| 4 | quick-260906-13w | unrun-verify | dataset/masking.py |  | UniverseMask.report() not run against real membership vs real us_all market data; a non-empty missing-member list would falsify D-07 | open |  | 2026-09-06T05:16:52.451Z |  |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "03.1",
    "file": ".planning/phases/03.1-index-historical-constituents/03.1-01-PLAN.md",
    "line": 454,
    "description": "Plan <verification> grep 'import Dataset' is over-broad — it also matches the live 'import DatasetConfig'; the word-boundary form was run instead",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T02:57:36.166Z",
    "resolved_at": null
  },
  {
    "id": 2,
    "kind": "unrun-verify",
    "phase": "quick-260906-0iy",
    "file": "ingest_us_equity.py",
    "line": null,
    "description": "Plan verification step 5 (TIINGO_API_KEY=... ingest_us_equity.py --limit 5) not run: no Tiingo credential available in this environment; offline tests cover resume/isolation but the real 5-symbol backfill is unexercised",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T04:46:32.180Z",
    "resolved_at": null
  },
  {
    "id": 3,
    "kind": "unrun-verify",
    "phase": "quick-260906-13w",
    "file": "ingest_us_equity.py",
    "line": null,
    "description": "Real full-market --to-zarr backfill not run: TIINGO_API_KEY unset, no raw parquet to convert",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T05:16:52.332Z",
    "resolved_at": null
  },
  {
    "id": 4,
    "kind": "unrun-verify",
    "phase": "quick-260906-13w",
    "file": "dataset/masking.py",
    "line": null,
    "description": "UniverseMask.report() not run against real membership vs real us_all market data; a non-empty missing-member list would falsify D-07",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T05:16:52.451Z",
    "resolved_at": null
  }
]
````
