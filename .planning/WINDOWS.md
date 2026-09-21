---
schema_version: 1
open_count: 18
waived_count: 0
fixed_count: 10
total_count: 28
last_updated: 2026-09-21T16:24:00.443Z
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
| 12 | 03.4 | deviation | .planning/phases/03.4-data-source-registry/03.4-07-PLAN.md | 397 | Task 3 verify counts every '**Resolved:**' literal in the Open Questions section, including one the lead-in sentence legitimately contains; it read 7 against correct content. The section's own lead-in was reworded and a discriminating list-item form was additionally run. Same class as plan 01's 'no tests ran' and plan 05's src.count('Parallel('). | open |  | 2026-09-09T02:02:11.204Z |  |
| 13 | 03.4 | deviation | quantlab/base/acquisition.py |  | REVIEW CR-01: _merge_unattempted_failures was wired into the 'if cancelled:' branch only, so both quota-abort exits fell through to _write_failure_manifest with only this run's failures and overwrote the previous run's manifest with {} -- which _write_failure_manifest's own docstring defines as 'the last run was clean'. wait_for_quota defaults to False, so this was the DEFAULT quota-abort behaviour. The D-18 invariant set(result.failures) == set(manifest) could not catch it: both sides are built from one dict, so result and manifest were consistently wrong together. Fixed by plan 03.4-08 (merge relocated outside the resume loop, unconditional, immediately before the write). | fixed |  | 2026-09-09T03:54:15.213Z | 2026-09-09T03:54:20.534Z |
| 14 | 03.5 | unrun-verify | tests/test_ingest_conversion_gate.py |  | test_refusal_precedes_every_symbol_bearing_dataset_construction has zero live coverage since df7bfe9 deleted BaseDataset._reset_symbols; kept as a regression guard, docstring states it | open |  | 2026-09-12T01:41:21.710Z |  |
| 15 | 03.5 | unrun-verify | ingest_us_equity.py |  | 03.5-05 Task 1 verify 'ingest_us_equity.py --dry-run' not run: the command uses a --symbols flag this shell has no, and the corrected form needs a universe.parquet this worktree has no data/ dir for | open |  | 2026-09-12T01:41:26.439Z |  |
| 16 | 03.5 | deviation | quantlab/base/data.py |  | BaseDataset.__init__ ordering comment still claims the config setter reaches the backend via _reset_symbols()->read(); df7bfe9 deleted that method, so the stated AttributeError invariant needs re-testing (plan 04 owns this file) | open |  | 2026-09-12T01:41:30.958Z |  |
| 17 | 03.5 | deviation | quantlab/acquisition/inspector.py | 99 | _RawTierReader._reset_symbols overrides a method df7bfe9 deleted from BaseDataset, so the override is inert and its class docstring still describes a construction-time from_raw_data() fallback that no longer happens. Same class as open window 16, one file over. Out of plan 06's scope (files_modified does not include inspector.py). | open |  | 2026-09-12T01:58:32.071Z |  |
| 18 | 03.6 | deviation | quantlab/base/chunking.py |  | Class docstring first line corrected (Rule 1) beyond the plan's two named constructs | open |  | 2026-09-12T20:56:42.552Z |  |
| 19 | 03.6 | deviation | quantlab/base/data.py |  | Plan 03.6-09 Rule 3: the pre-try rebuild_rolled_back seed was annotated (bool) so the plan's AST gate, which forbids any ast.Constant-valued assignment, could pass; semantics unchanged | open |  | 2026-09-13T21:34:19.364Z |  |
| 20 | 03.9 | stub | quantlab/dataset/nbbo_resample.py | 272 | n_ambiguous_ties emitted as 0.0 until plan 03.9-05 adds tie collapse and the ambiguity count (D-19) | fixed |  | 2026-09-19T18:49:10.437Z | 2026-09-19T19:14:23.404Z |
| 21 | 03.11 | deviation | tests/test_crsp_constituent.py |  | test_membership_symbols_agree_with_the_crsp_price_panel red between 03.11-03 and 03.11-05: price panel is on the int64 PERMNO axis, the membership panel is still ticker-keyed. Owned by plan 03.11-05. | fixed |  | 2026-09-21T04:00:24.635Z | 2026-09-21T04:36:55.739Z |
| 22 | 03.11 | deviation | tests/test_ingest_wrds_crsp.py |  | 4 tests red between 03.11-03 and 03.11-08: they assert ticker axis labels (AAPL / QQQ) that the int64 PERMNO axis no longer carries. Owned by plan 03.11-08. | fixed | UAT 03.11 test 6 裁定为 fixed。交棒 owner plan 03.11-08 已执行，现场复核：`uv run pytest tests/test_ingest_wrds_crsp.py tests/test_no_identity_residue.py -q` = 32 passed；strict-xfail 分组 `DELETED_IN_03_11_08` 已不存在，`symbol_overrides` / `nan_adj_at_permno_seam` 已提升进 `tests/test_no_identity_residue.py:58` 的 `DELETED_IN_03_11_07`，该处注释记录了提升来由。 | 2026-09-21T04:00:30.770Z | 2026-09-21T16:24:00.204Z |
| 23 | 03.11 | deviation | tests/test_ingest_wrds_crsp.py |  | test_the_universe_conversion_also_writes_the_membership_panel red from 03.11-05: it asserts the ticker label AAPL on the membership panel's symbol axis, which is now the int64 PERMNO 14593. Same file, same cause and same owner as ledger entry 22 -- plan 03.11-08. Out of 03.11-05's files_modified, so handed over rather than edited. | fixed | UAT 03.11 test 6 裁定为 fixed。交棒 owner plan 03.11-08 已执行，现场复核：`uv run pytest tests/test_ingest_wrds_crsp.py tests/test_no_identity_residue.py -q` = 32 passed；strict-xfail 分组 `DELETED_IN_03_11_08` 已不存在，`symbol_overrides` / `nan_adj_at_permno_seam` 已提升进 `tests/test_no_identity_residue.py:58` 的 `DELETED_IN_03_11_07`，该处注释记录了提升来由。 | 2026-09-21T04:37:01.261Z | 2026-09-21T16:24:00.323Z |
| 24 | 03.11 | deviation | tests/test_no_identity_residue.py |  | strict-xfail group DELETED_IN_03_11_08 (symbol_overrides / nan_adj_at_permno_seam) is red by design until plan 03.11-08 deletes those config fields and promotes the names into DELETED_IN_03_11_07 | fixed | UAT 03.11 test 6 裁定为 fixed。交棒 owner plan 03.11-08 已执行，现场复核：`uv run pytest tests/test_ingest_wrds_crsp.py tests/test_no_identity_residue.py -q` = 32 passed；strict-xfail 分组 `DELETED_IN_03_11_08` 已不存在，`symbol_overrides` / `nan_adj_at_permno_seam` 已提升进 `tests/test_no_identity_residue.py:58` 的 `DELETED_IN_03_11_07`，该处注释记录了提升来由。 | 2026-09-21T05:57:52.914Z | 2026-09-21T16:24:00.443Z |
| 25 | 03.11 | deviation | example/wrds_crsp.md | 364 | 03.11-09 added the .crsp_tickers.json sidecar; this doc's two sidecar enumerations (the prose at :364 and the ASCII tree at :456-457) list only the adjustment/filter/symbology trio. Out of 03.11-09's files_modified and inside the doc set plan 03.11-10 already owns, so handed over rather than edited. | fixed |  | 2026-09-21T07:19:31.356Z | 2026-09-21T07:53:48.709Z |
| 26 | 03.11 | deviation | example/backtest.md | 227 | 03.11-09 added an axis_symbol field to every forced-liquidation record and made symbol the period-correct ticker; example/backtest.md:227 and :445 still describe the pre-09 record. Handed to plan 03.11-10's doc pass. | fixed |  | 2026-09-21T07:19:36.837Z | 2026-09-21T07:53:54.261Z |
| 27 | 03.11 | deviation | example/constituent.md | 468 | 03.11-09 added missing_labels to UniverseMask.report(); the transcribed REPL output at example/constituent.md:468 shows the three-key dict and is now short one key. Handed to plan 03.11-10's doc pass. | fixed |  | 2026-09-21T07:19:42.728Z | 2026-09-21T07:53:54.380Z |
| 28 | 03.11 | deviation | data/data/us_equity/1d/wrds_crsp_custom_1d.zarr |  | The custom CRSP store is still a ticker-axis panel written 2026-09-20 by pre-migration code, and its stale .crsp_symbology_report.json sidecar survives beside it. Plan 03.11-10's D-15 scope was sp500/2024 only, and rebuilding custom needs a window/roster this phase never specified. data/ is gitignored, so this blocks nothing in git -- but any read of that store returns a panel the current tree cannot reproduce. | fixed | Operator 裁定删除（03.11-11 Task 2 checkpoint）。编排器已在主仓库工作树备份至 data/_backup_deleted_custom_store_03.11/（784K，zarr 本体 + 四个 sidecar，含 58,846 B 的 .crsp_symbology_report.json），随后删除 wrds_crsp_custom_1d.zarr 及其 .chunks.json / .crsp_adjustment.json / .crsp_filter_report.json / .crsp_symbology_report.json。删除后实测 1d/ 目录内 crsp_symbology_report 命中数为 0，symbology 旁车归零；这顺带关闭了 plan 03.11-10 记录的那条返回 1 的未满足验收条件。 | 2026-09-21T07:53:10.814Z | 2026-09-21T10:55:47.471Z |

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
  },
  {
    "id": 12,
    "kind": "deviation",
    "phase": "03.4",
    "file": ".planning/phases/03.4-data-source-registry/03.4-07-PLAN.md",
    "line": 397,
    "description": "Task 3 verify counts every '**Resolved:**' literal in the Open Questions section, including one the lead-in sentence legitimately contains; it read 7 against correct content. The section's own lead-in was reworded and a discriminating list-item form was additionally run. Same class as plan 01's 'no tests ran' and plan 05's src.count('Parallel(').",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-09T02:02:11.204Z",
    "resolved_at": null
  },
  {
    "id": 13,
    "kind": "deviation",
    "phase": "03.4",
    "file": "quantlab/base/acquisition.py",
    "line": null,
    "description": "REVIEW CR-01: _merge_unattempted_failures was wired into the 'if cancelled:' branch only, so both quota-abort exits fell through to _write_failure_manifest with only this run's failures and overwrote the previous run's manifest with {} -- which _write_failure_manifest's own docstring defines as 'the last run was clean'. wait_for_quota defaults to False, so this was the DEFAULT quota-abort behaviour. The D-18 invariant set(result.failures) == set(manifest) could not catch it: both sides are built from one dict, so result and manifest were consistently wrong together. Fixed by plan 03.4-08 (merge relocated outside the resume loop, unconditional, immediately before the write).",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-09T03:54:15.213Z",
    "resolved_at": "2026-09-09T03:54:20.534Z"
  },
  {
    "id": 14,
    "kind": "unrun-verify",
    "phase": "03.5",
    "file": "tests/test_ingest_conversion_gate.py",
    "line": null,
    "description": "test_refusal_precedes_every_symbol_bearing_dataset_construction has zero live coverage since df7bfe9 deleted BaseDataset._reset_symbols; kept as a regression guard, docstring states it",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-12T01:41:21.710Z",
    "resolved_at": null
  },
  {
    "id": 15,
    "kind": "unrun-verify",
    "phase": "03.5",
    "file": "ingest_us_equity.py",
    "line": null,
    "description": "03.5-05 Task 1 verify 'ingest_us_equity.py --dry-run' not run: the command uses a --symbols flag this shell has no, and the corrected form needs a universe.parquet this worktree has no data/ dir for",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-12T01:41:26.439Z",
    "resolved_at": null
  },
  {
    "id": 16,
    "kind": "deviation",
    "phase": "03.5",
    "file": "quantlab/base/data.py",
    "line": null,
    "description": "BaseDataset.__init__ ordering comment still claims the config setter reaches the backend via _reset_symbols()->read(); df7bfe9 deleted that method, so the stated AttributeError invariant needs re-testing (plan 04 owns this file)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-12T01:41:30.958Z",
    "resolved_at": null
  },
  {
    "id": 17,
    "kind": "deviation",
    "phase": "03.5",
    "file": "quantlab/acquisition/inspector.py",
    "line": 99,
    "description": "_RawTierReader._reset_symbols overrides a method df7bfe9 deleted from BaseDataset, so the override is inert and its class docstring still describes a construction-time from_raw_data() fallback that no longer happens. Same class as open window 16, one file over. Out of plan 06's scope (files_modified does not include inspector.py).",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-12T01:58:32.071Z",
    "resolved_at": null
  },
  {
    "id": 18,
    "kind": "deviation",
    "phase": "03.6",
    "file": "quantlab/base/chunking.py",
    "line": null,
    "description": "Class docstring first line corrected (Rule 1) beyond the plan's two named constructs",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-12T20:56:42.552Z",
    "resolved_at": null
  },
  {
    "id": 19,
    "kind": "deviation",
    "phase": "03.6",
    "file": "quantlab/base/data.py",
    "line": null,
    "description": "Plan 03.6-09 Rule 3: the pre-try rebuild_rolled_back seed was annotated (bool) so the plan's AST gate, which forbids any ast.Constant-valued assignment, could pass; semantics unchanged",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-13T21:34:19.364Z",
    "resolved_at": null
  },
  {
    "id": 20,
    "kind": "stub",
    "phase": "03.9",
    "file": "quantlab/dataset/nbbo_resample.py",
    "line": 272,
    "description": "n_ambiguous_ties emitted as 0.0 until plan 03.9-05 adds tie collapse and the ambiguity count (D-19)",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-19T18:49:10.437Z",
    "resolved_at": "2026-09-19T19:14:23.404Z"
  },
  {
    "id": 21,
    "kind": "deviation",
    "phase": "03.11",
    "file": "tests/test_crsp_constituent.py",
    "line": null,
    "description": "test_membership_symbols_agree_with_the_crsp_price_panel red between 03.11-03 and 03.11-05: price panel is on the int64 PERMNO axis, the membership panel is still ticker-keyed. Owned by plan 03.11-05.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-21T04:00:24.635Z",
    "resolved_at": "2026-09-21T04:36:55.739Z"
  },
  {
    "id": 22,
    "kind": "deviation",
    "phase": "03.11",
    "file": "tests/test_ingest_wrds_crsp.py",
    "line": null,
    "description": "4 tests red between 03.11-03 and 03.11-08: they assert ticker axis labels (AAPL / QQQ) that the int64 PERMNO axis no longer carries. Owned by plan 03.11-08.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-21T04:00:30.770Z",
    "resolved_at": "2026-09-21T16:24:00.204Z"
  },
  {
    "id": 23,
    "kind": "deviation",
    "phase": "03.11",
    "file": "tests/test_ingest_wrds_crsp.py",
    "line": null,
    "description": "test_the_universe_conversion_also_writes_the_membership_panel red from 03.11-05: it asserts the ticker label AAPL on the membership panel's symbol axis, which is now the int64 PERMNO 14593. Same file, same cause and same owner as ledger entry 22 -- plan 03.11-08. Out of 03.11-05's files_modified, so handed over rather than edited.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-21T04:37:01.261Z",
    "resolved_at": "2026-09-21T16:24:00.323Z"
  },
  {
    "id": 24,
    "kind": "deviation",
    "phase": "03.11",
    "file": "tests/test_no_identity_residue.py",
    "line": null,
    "description": "strict-xfail group DELETED_IN_03_11_08 (symbol_overrides / nan_adj_at_permno_seam) is red by design until plan 03.11-08 deletes those config fields and promotes the names into DELETED_IN_03_11_07",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-21T05:57:52.914Z",
    "resolved_at": "2026-09-21T16:24:00.443Z"
  },
  {
    "id": 25,
    "kind": "deviation",
    "phase": "03.11",
    "file": "example/wrds_crsp.md",
    "line": 364,
    "description": "03.11-09 added the .crsp_tickers.json sidecar; this doc's two sidecar enumerations (the prose at :364 and the ASCII tree at :456-457) list only the adjustment/filter/symbology trio. Out of 03.11-09's files_modified and inside the doc set plan 03.11-10 already owns, so handed over rather than edited.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-21T07:19:31.356Z",
    "resolved_at": "2026-09-21T07:53:48.709Z"
  },
  {
    "id": 26,
    "kind": "deviation",
    "phase": "03.11",
    "file": "example/backtest.md",
    "line": 227,
    "description": "03.11-09 added an axis_symbol field to every forced-liquidation record and made symbol the period-correct ticker; example/backtest.md:227 and :445 still describe the pre-09 record. Handed to plan 03.11-10's doc pass.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-21T07:19:36.837Z",
    "resolved_at": "2026-09-21T07:53:54.261Z"
  },
  {
    "id": 27,
    "kind": "deviation",
    "phase": "03.11",
    "file": "example/constituent.md",
    "line": 468,
    "description": "03.11-09 added missing_labels to UniverseMask.report(); the transcribed REPL output at example/constituent.md:468 shows the three-key dict and is now short one key. Handed to plan 03.11-10's doc pass.",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-21T07:19:42.728Z",
    "resolved_at": "2026-09-21T07:53:54.380Z"
  },
  {
    "id": 28,
    "kind": "deviation",
    "phase": "03.11",
    "file": "data/data/us_equity/1d/wrds_crsp_custom_1d.zarr",
    "line": null,
    "description": "The custom CRSP store is still a ticker-axis panel written 2026-09-20 by pre-migration code, and its stale .crsp_symbology_report.json sidecar survives beside it. Plan 03.11-10's D-15 scope was sp500/2024 only, and rebuilding custom needs a window/roster this phase never specified. data/ is gitignored, so this blocks nothing in git -- but any read of that store returns a panel the current tree cannot reproduce.",
    "status": "fixed",
    "reason": "Operator 裁定删除（03.11-11 Task 2 checkpoint）。编排器已在主仓库工作树备份至 data/_backup_deleted_custom_store_03.11/（784K，zarr 本体 + 四个 sidecar，含 58,846 B 的 .crsp_symbology_report.json），随后删除 wrds_crsp_custom_1d.zarr 及其 .chunks.json / .crsp_adjustment.json / .crsp_filter_report.json / .crsp_symbology_report.json。删除后实测 1d/ 目录内 crsp_symbology_report 命中数为 0，symbology 旁车归零；这顺带关闭了 plan 03.11-10 记录的那条返回 1 的未满足验收条件。",
    "recorded_at": "2026-09-21T07:53:10.814Z",
    "resolved_at": "2026-09-21T10:55:47.471Z"
  }
]
````
