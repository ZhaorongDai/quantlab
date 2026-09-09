---
schema_version: 1
open_count: 11
waived_count: 0
fixed_count: 0
total_count: 11
last_updated: 2026-09-09T01:40:00.775Z
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
| 5 | 03.2 | unrun-verify | ingest_alpaca.py |  | 03.2-07 open verification A (O-1): free-tier Alpaca SIP access unresolved; needs one live curl with real credentials | open |  | 2026-09-06T22:47:40.434Z |  |
| 6 | 03.2 | unrun-verify | ingest_alpaca.py |  | 03.2-07 open verification B (SC-5 manual half): no real Alpaca round-trip performed; no credentials on this machine | open |  | 2026-09-06T22:47:40.650Z |  |
| 7 | 03.2 | unrun-verify | ingest_alpaca.py |  | 03.2-07 open verification C (O-2): real Alpaca symbols-per-request ceiling unprobed; DEFAULT_BATCH_SIZE=100 is a conservative working value | open |  | 2026-09-06T22:47:40.800Z |  |
| 8 | quick-260907-sm2 | deviation | README.md |  | Doc-path sweep regex excluded ':' so backticked refs like base/model.py:BaseModel were invisible; widened and fixed (~40 refs) | open |  | 2026-09-08T01:29:08.360Z |  |
| 9 | 03.4 | unmet-truth | tests/test_tiingo_quota.py |  | The runtime quota suite does not pin that _attempt_batch's first statement honours the vendor abort: mutating _should_stop to consult only the cancel token leaves all 24 tests green (max_workers=2 means joblib pre_dispatch withholds most batches). Only the structural abort_is_first guard catches it. Pre-existing; surfaced by 03.4-05 mutation M6. | open |  | 2026-09-09T01:14:26.385Z |  |
| 10 | 03.4 | unmet-truth | ingest_us_equity.py |  | Empty-roster refusal (if not symbols: parser.error) is carried as a backstop truth in 03.4-06: the branch is untouched and sits before the only run(SOURCE, ...) call, but no test drives it | open |  | 2026-09-09T01:39:55.444Z |  |
| 11 | 03.4 | unmet-truth | ingest_us_equity.py | 449 | The stamp-legacy-watermarks flag reaches stamp_watermarks() only through SOURCE.acquisition_cls; the WRITE itself is unexercised in-repo and remains the open blocking-human checkpoint from quick task 260906-26o | open |  | 2026-09-09T01:40:00.775Z |  |

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
  },
  {
    "id": 5,
    "kind": "unrun-verify",
    "phase": "03.2",
    "file": "ingest_alpaca.py",
    "line": null,
    "description": "03.2-07 open verification A (O-1): free-tier Alpaca SIP access unresolved; needs one live curl with real credentials",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T22:47:40.434Z",
    "resolved_at": null
  },
  {
    "id": 6,
    "kind": "unrun-verify",
    "phase": "03.2",
    "file": "ingest_alpaca.py",
    "line": null,
    "description": "03.2-07 open verification B (SC-5 manual half): no real Alpaca round-trip performed; no credentials on this machine",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T22:47:40.650Z",
    "resolved_at": null
  },
  {
    "id": 7,
    "kind": "unrun-verify",
    "phase": "03.2",
    "file": "ingest_alpaca.py",
    "line": null,
    "description": "03.2-07 open verification C (O-2): real Alpaca symbols-per-request ceiling unprobed; DEFAULT_BATCH_SIZE=100 is a conservative working value",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-06T22:47:40.800Z",
    "resolved_at": null
  },
  {
    "id": 8,
    "kind": "deviation",
    "phase": "quick-260907-sm2",
    "file": "README.md",
    "line": null,
    "description": "Doc-path sweep regex excluded ':' so backticked refs like base/model.py:BaseModel were invisible; widened and fixed (~40 refs)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-08T01:29:08.360Z",
    "resolved_at": null
  },
  {
    "id": 9,
    "kind": "unmet-truth",
    "phase": "03.4",
    "file": "tests/test_tiingo_quota.py",
    "line": null,
    "description": "The runtime quota suite does not pin that _attempt_batch's first statement honours the vendor abort: mutating _should_stop to consult only the cancel token leaves all 24 tests green (max_workers=2 means joblib pre_dispatch withholds most batches). Only the structural abort_is_first guard catches it. Pre-existing; surfaced by 03.4-05 mutation M6.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-09T01:14:26.385Z",
    "resolved_at": null
  },
  {
    "id": 10,
    "kind": "unmet-truth",
    "phase": "03.4",
    "file": "ingest_us_equity.py",
    "line": null,
    "description": "Empty-roster refusal (if not symbols: parser.error) is carried as a backstop truth in 03.4-06: the branch is untouched and sits before the only run(SOURCE, ...) call, but no test drives it",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-09T01:39:55.444Z",
    "resolved_at": null
  },
  {
    "id": 11,
    "kind": "unmet-truth",
    "phase": "03.4",
    "file": "ingest_us_equity.py",
    "line": 449,
    "description": "The stamp-legacy-watermarks flag reaches stamp_watermarks() only through SOURCE.acquisition_cls; the WRITE itself is unexercised in-repo and remains the open blocking-human checkpoint from quick task 260906-26o",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-09T01:40:00.775Z",
    "resolved_at": null
  }
]
````
