---
schema_version: 1
open_count: 1
waived_count: 0
fixed_count: 0
total_count: 1
last_updated: 2026-09-06T02:57:36.166Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 03.1 | deviation | .planning/phases/03.1-index-historical-constituents/03.1-01-PLAN.md | 454 | Plan <verification> grep 'import Dataset' is over-broad — it also matches the live 'import DatasetConfig'; the word-boundary form was run instead | open |  | 2026-09-06T02:57:36.166Z |  |

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
  }
]
````
