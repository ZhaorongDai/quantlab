---
phase: quick-260909-idh
plan: 01
subsystem: data-acquisition
tags: [cli, argparse, polars, zarr, reproducibility, ingest]

requires:
  - phase: 03.4-data-source-registry
    provides: the three thin ingest shells, UniverseCatalog's two roster queries, and the UAT that found both gaps
provides:
  - "UniverseCatalog.get_symbols_in_range / get_symbols_as_of return an ascending, content-determined order, stated as a contract in both docstrings and in --limit's help"
  - "StockDataset.has_raw_data(): the raw-presence predicate, defined once and read by both _scan_raw and the CLI"
  - "quantlab.utils.cli.refuse_conversion_without_raw_data: a clean non-zero exit when a run fetched nothing onto an empty raw tree"
  - "quantlab.utils.cli.add_to_zarr_arg: --to-zarr registered once, carried by all three ingest shells, off by default"
  - "ingest_alpaca.py refuses --frequency tick with --to-zarr at exit 2"
affects: [ingest, universe, dataset, documentation]

actuals:
  tokens: 16806
  tasks: 3
  commits: 3
plan_head_before: f0abcd8c9b84075bff299c0ab5bae790908ab7cc

tech-stack:
  added: []
  patterns:
    - "One predicate, one home: a fact two layers decide on lives in one method (has_raw_data), never as two spellings"
    - "Guards scoped by reachability (every __main__ that densifies), never by script name"
    - "Shared argparse registrars parameterise the genuinely different half (add_to_zarr_arg(mode=...)), following add_concurrency_args"

key-files:
  created:
    - tests/test_ingest_conversion_gate.py
  modified:
    - quantlab/acquisition/universe.py
    - quantlab/utils/cli.py
    - quantlab/dataset/stock.py
    - ingest_tiingo.py
    - ingest_alpaca.py
    - ingest_us_equity.py
    - tests/test_universe.py
    - tests/test_ingest_shells.py
    - tests/test_ingest_tiingo_universe_wiring.py
    - README.md
    - example/acquisition.md

key-decisions:
  - "Sort the roster rather than unique(maintain_order=True): sorting makes the order a function of the membership SET; maintain_order pins it to the parquet ROW order, which refresh_us_equity_universe.py rewrites — the same reproducibility hole one layer down"
  - "The zero-success guard probes the disk, not the success count: a run whose symbols were all skipped by watermark also reports zero successes and must still convert"
  - "--to-zarr defaults to off in ingest_tiingo.py and ingest_alpaca.py — a deliberate behaviour change to their existing default, locked by the user before planning"
  - "assert_dense_panel_fits becomes conditional on --to-zarr, but does not move: sizing a densification that will not happen would refuse fetches that are fine, while moving it would break the SC-6 AST ordering assertions"
  - "tick + --to-zarr is refused at exit 2 rather than ignored; silently dropping the flag is the same silence this task exists to end"
  - "tests/test_ingest_tiingo_universe_wiring.py's hand-built Namespace gained to_zarr=False so _validate_data_type reads the attribute directly — a getattr(..., False) default would let the tick refusal go quietly missing"

requirements-completed: [G-03.4-1, G-03.4-2]

coverage:
  - id: D1
    description: "Both roster queries return an ascending, call-to-call stable order, so --limit N truncates to the same N symbols every run"
    requirement: "G-03.4-2"
    verification:
      - kind: unit
        ref: "tests/test_universe.py::test_both_roster_queries_return_the_same_order_every_call"
        status: pass
      - kind: unit
        ref: "uv run pytest tests/test_universe.py -q (77 passed, incl. the 26 pre-existing in_range/as_of assertions)"
        status: pass
    human_judgment: false
  - id: D2
    description: "A run that fetched nothing onto an empty raw tree exits non-zero with a readable message instead of an uncaught ValueError traceback; a run whose symbols were all skipped still converts"
    requirement: "G-03.4-1"
    verification:
      - kind: unit
        ref: "tests/test_ingest_conversion_gate.py::test_zero_success_and_an_empty_raw_root_refuses_with_a_readable_message"
        status: pass
      - kind: unit
        ref: "tests/test_ingest_conversion_gate.py::test_zero_success_but_raw_data_on_disk_converts_anyway"
        status: pass
      - kind: unit
        ref: "tests/test_ingest_conversion_gate.py::test_every_entry_point_that_densifies_refuses_first"
        status: pass
    human_judgment: false
  - id: D3
    description: "All three ingest shells gate their Zarr conversion behind --to-zarr, default to raw, and say so; tick refuses the flag at exit 2"
    requirement: "G-03.4-1"
    verification:
      - kind: unit
        ref: "tests/test_ingest_conversion_gate.py::test_every_shell_registers_to_zarr_and_defaults_to_not_converting"
        status: pass
      - kind: integration
        ref: "tests/test_ingest_conversion_gate.py::test_alpaca_refuses_tick_with_to_zarr_at_exit_2 (real subprocess)"
        status: pass
      - kind: other
        ref: "uv run python -c \"import ingest_alpaca, ingest_tiingo, ingest_us_equity; ...\" -> all three True"
        status: pass
    human_judgment: false
  - id: D4
    description: "No sentence in the repo still claims these two scripts convert unconditionally; example/ output is unfalsified"
    verification:
      - kind: other
        ref: "the plan's doc truth gate: module docstrings and README bullets both mention --to-zarr -> 'doc truth gate OK'"
        status: pass
      - kind: unit
        ref: "uv run pytest tests/test_ingest_shells.py tests/test_volume_guard.py -q (43 passed)"
        status: pass
    human_judgment: true
    rationale: "Completeness of a sentence-by-sentence sweep cannot be asserted by a scan — the literal gate only checks two files' worth of the claim. A reader confirming no surviving false sentence is the real check."

duration: 47min
completed: 2026-09-09
status: complete
---

# Quick 260909-idh: Two Phase-03.4 UAT Gaps Summary

**Both silences turned into signals: the roster now resolves in a content-determined ascending order so `--limit` is reproducible, and a run that fetched nothing exits cleanly instead of dying inside the conversion — with the Zarr conversion itself now an explicit `--to-zarr` opt-in across all three ingest shells.**

## Performance

- **Duration:** ~47 min
- **Tasks:** 3 of 3
- **Files modified:** 11 modified, 1 created
- **Full suite:** 744 passed, 0 failed. No pre-change full-suite baseline was taken, so no delta is claimed here; the one measured baseline is `tests/test_universe.py` at 76 before Task 1 and 77 after.

## Accomplishments

- **G-03.4-2 closed.** `get_symbols_in_range` and `get_symbols_as_of` sort ascending by symbol after `.unique()`. Two calls with the same arguments now return element-wise equal lists, because the order is a function of the membership set rather than of the parquet row order or polars' threading. `resolve_symbols`' `symbols[:limit]` therefore truncates to the same batch every run, so a second run meets the watermarks the first one wrote and resume/skip can fire.
- **G-03.4-1(a) closed.** `refuse_conversion_without_raw_data` sits in front of all three densification sites and raises `SystemExit` with the raw root, both counts and the failure-manifest entry point — no credential value, no vendor response body. The previous behaviour was `quantlab/dataset/stock.py`'s absent-root `ValueError` traceback, which misattributed a fetch failure to the conversion layer.
- **G-03.4-1(b) closed.** `--to-zarr` is registered once in `quantlab/utils/cli.py:add_to_zarr_arg` and carried by all three shells, off by default. `ingest_tiingo.py` and `ingest_alpaca.py` no longer convert unconditionally, and the default path prints that it skipped. `ingest_us_equity.py`'s chunked help text survives verbatim.
- **The third door was fixed without being asked.** `ingest_us_equity.py --to-zarr` reaches the identical absent-root `ValueError` and appeared in neither bug report. The guard was wired by reachability — every `__main__` that calls `from_raw_data` / `from_raw_data_chunked` — and a new AST test enforces that scoping.
- **One predicate, one home.** `StockDataset.has_raw_data()` is now the only raw-presence test in the repo; `_scan_raw` reads it, the CLI reads it, and a test asserts no second spelling appears in either file.

## Task Commits

1. **Task 1: roster order is content-determined (G-03.4-2)** — `eebf2b3` (fix)
2. **Task 2: conversion guard + `--to-zarr` opt-in on three doors (G-03.4-1)** — `1d71521` (fix)
3. **Task 3: sentence-by-sentence sweep of the now-false docs** — `41e4b4a` (docs)

## Mutation Verification (Task 1)

The plan required the new lock be mutation-verified rather than accepted for arriving green. Both `.sort("symbol")` calls were removed, `tests/test_universe.py::test_both_roster_queries_return_the_same_order_every_call` was re-run, each arm was probed individually, and the sorts were restored.

| Assertion arm | Under mutation |
|---|---|
| A1 `in_range` two calls equal | **RED** |
| A1 `as_of` two calls equal | **RED** |
| A2 `in_range == sorted(...)` | **RED** |
| A2 `as_of == sorted(...)` | **RED** |
| A2 `in_range` de-duplicated | green (expected) |
| A2 `as_of` de-duplicated | green (expected) |
| A3 AST: `.sort(...)` encloses `.unique()` in `get_symbols_in_range` | **RED** |
| A3 AST: same in `get_symbols_as_of` | **RED** |

Six of eight arms went red. The two that stayed green are the de-duplication assertions, and correctly so: dedup is `.unique()`'s job and the mutation removed only the sort — a green there is the arms doing their separate jobs, not a vacuum.

Worth recording: the plan anticipated that arm A1 ("two calls equal") might be vacuously green on a fixture this small, and that the AST arm would be the only sharp instrument. It was not vacuous — polars' `unique()` genuinely returned a different order between two calls on the same in-memory catalog (`['DUAL1', 'UA...']` vs `['PBR-A', 'C-...']`). The gap reproduces on a 20-row fixture, not merely at 15.4k symbols.

## Sentence-by-Sentence Doc Sweep (Task 3)

Enumeration was run at execution time (`grep -rn "ingest_tiingo\|ingest_alpaca" README.md example/ ingest_*.py tests/` → 110 hits; `grep -rni "zarr" ingest_alpaca.py ingest_tiingo.py README.md` → 36 hits), then read sentence by sentence. **~24 distinct claims judged; 12 changed.** One rule per row:

| Site | Verdict | Reason |
|---|---|---|
| `README.md` "the three convert in three different modes" | changed | all three now gate on `--to-zarr`; the mode (chunked vs whole-window), not the fact of converting, is what differs |
| `README.md` `ingest_tiingo.py` bullet | changed | "full Tiingo-to-Zarr pipeline … then converts" is false by default |
| `README.md` `ingest_us_equity.py` bullet | changed | `--to-zarr` is no longer what distinguishes it; the chunked, resumable conversion is |
| `README.md` `ingest_alpaca.py` bullet | changed | "then converts bars to Zarr" is false; tick now refuses the flag rather than merely stopping |
| `README.md` other Zarr mentions (L19/95/100-101/153/170/206/253/347/359) | kept | all about the Dataset layer, constituent panels or Binance — unrelated to these two scripts |
| `ingest_tiingo.py:1` docstring title | changed | promised persistence unconditionally |
| `ingest_tiingo.py:3-6` "Full Tiingo-to-Zarr pipeline" | changed | the causal chain is now conditional |
| `ingest_tiingo.py` argparse `description` | changed | **found by the enumeration, not on the plan's list** — same false promise, third copy |
| `ingest_tiingo.py` Usage block | changed | a raw-only default section plus one `--to-zarr` example; also states the `--limit` ascending contract |
| `ingest_alpaca.py` "Two pre-flight guards" section | changed | tick is no longer the only reason the dense guard is skipped; an absent `--to-zarr` is now the common one |
| `ingest_alpaca.py` Usage block (6 commands) | changed | a leading note that every command lands raw; one added `--to-zarr` variant; the tick example deliberately does **not** gain the flag and says why |
| `ingest_alpaca.py` `DEFAULT_STORE_NAME` comment | changed | true as written but implied the conversions always run |
| `ingest_alpaca.py` tick branch "a Zarr store this phase never writes" | kept | still true |
| `example/acquisition.md:484` 例 2 command | changed | **command only** — the prose below claims both guards ran pre-client, which now requires `--to-zarr`. Output byte-identical (the dense guard prints nothing when it admits); a note records that the command was amended and the output was not |
| `example/acquisition.md:502` traceback line numbers | kept | historical record, per the plan |
| `example/acquisition.md:525` 例 3 command | kept | its listed artefacts are the raw tree and watermarks only, reproducible as written |
| `example/acquisition.md:650` 例 4 tick command | kept | must **not** gain `--to-zarr` — it would exit 2 |
| `example/acquisition.md:818` point 12 | changed | "tick stops at raw" is no longer distinguishing; every frequency now does. Tick's property is the refusal |
| `example/acquisition.md:16, 392, 401` | kept | acquisition-layer and chunking statements, still true |
| `tests/test_ingest_shells.py:562` "the rest are declared in the script" | changed | false: `--to-zarr` now arrives through `add_to_zarr_arg` |
| `tests/test_ingest_shells.py:9` header | changed | the chunking is still exclusive, the flag is not |
| `tests/test_ingest_shells.py:611` seven-differences docstring | changed | same distinction made explicit |
| `tests/test_volume_guard.py:1328-1345` docstring | kept | historical plus a reachability claim, both still true |
| `tests/test_volume_guard.py:1450` "the --to-zarr sizing guard is gone" | kept | still true |

**No pasted output in `example/` was altered.** This machine has no Alpaca or Tiingo credentials; nothing was re-run and nothing was invented.

## Files Created/Modified

- `quantlab/acquisition/universe.py` — ascending sort on both roster queries, the argument for sort-over-`maintain_order` in source, ordering contract in both docstrings (`get_symbols_as_of` had none at all)
- `quantlab/utils/cli.py` — `ConversionMode` / `_TO_ZARR_HELP`, `add_to_zarr_arg`, `refuse_conversion_without_raw_data`, corrected `--limit` help, corrected module-docstring dependency claim
- `quantlab/dataset/stock.py` — `has_raw_data()` extracted; `_scan_raw` now reads it
- `ingest_tiingo.py` / `ingest_alpaca.py` — `--to-zarr` opt-in, guard wired, RAM guard conditionalised (position unchanged), tick+`--to-zarr` refused (alpaca), docs corrected
- `ingest_us_equity.py` — registers the shared flag, guard wired in front of its chunked densification
- `tests/test_ingest_conversion_gate.py` (new, 9 tests) — three guard behaviours, parser-level flag registration, tick exit 2, reachability AST arm
- `tests/test_universe.py` — the three-arm ordering regression
- `tests/test_ingest_shells.py`, `tests/test_ingest_tiingo_universe_wiring.py` — corrected claims and the Namespace fixture
- `README.md`, `example/acquisition.md` — the sweep

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `tests/test_ingest_tiingo_universe_wiring.py`'s hand-built Namespace lacked `to_zarr`**
- **Found during:** Task 2
- **Issue:** `_alpaca_args()` constructs an `argparse.Namespace` by hand rather than through the real parser. Adding the tick+`--to-zarr` check to `_validate_data_type` made that test raise `AttributeError: 'Namespace' object has no attribute 'to_zarr'`. Caught by the full-suite run, not by the plan's per-task verification set.
- **Fix:** added `to_zarr=False` to the fixture's base dict, mirroring the parser default, with a comment on why. The alternative — `getattr(args, "to_zarr", False)` in the validator — was rejected: a tolerant default would let the tick refusal go quietly missing on any Namespace that forgot the flag, which is precisely the silence this task removes.
- **Files modified:** `tests/test_ingest_tiingo_universe_wiring.py`
- **Verification:** `uv run pytest -q` → 744 passed
- **Committed in:** `1d71521` (part of the Task 2 commit)

**2. [Rule 1 - Bug] my own first draft of a test arm used the instrument the plan warns against**
- **Found during:** Task 2
- **Issue:** the "one predicate, one home" arm in the new test file counted the substring `rglob("*.pqt")` in `quantlab/dataset/stock.py` and failed at 2 — the second hit was the prose inside `has_raw_data`'s own docstring. A literal scan counting a sentence about a probe as a probe.
- **Fix:** rewritten to walk the AST and assert the owning function names of every `rglob` **Call** node equal `["has_raw_data"]`.
- **Files modified:** `tests/test_ingest_conversion_gate.py`
- **Verification:** `uv run pytest tests/test_ingest_conversion_gate.py -q` → 9 passed
- **Committed in:** `1d71521`

### Process deviation (not auto-fixed, recorded honestly)

**Task 1 carried `tdd="true"` but was implemented before its test was written.** The plan's frontmatter is `type: execute`, not `type: tdd`, and no `TDD_MODE` was passed, so no plan-level gate applied. The RED evidence the cycle exists to produce was nonetheless obtained, and obtained more strongly than a written-first test would have: the mutation run above re-created the pre-fix code exactly and showed six of eight arms failing against it. Stated rather than glossed, because "the mutation is equivalent to RED" is a claim a reader should be able to check, not assume.

---

**Total deviations:** 2 auto-fixed (1× Rule 3, 1× Rule 1) + 1 recorded process deviation
**Impact on plan:** none on scope. Both auto-fixes were consequences of this task's own changes; neither touched a file outside the plan's `files_modified` except `tests/test_ingest_tiingo_universe_wiring.py`, which the change made fail.

## Issues Encountered

- The plan's per-task verification set for Task 2 did not include `tests/test_ingest_tiingo_universe_wiring.py`, so the fixture break surfaced only on the full-suite run. Running `uv run pytest -q` after each task, as the plan's overall verification requires, is what caught it — a per-file verification set is not a substitute for it in a repo with this many source-scanning structural tests.

## Threat Flags

None. The task adds no network surface, no new dependency and no path construction from user input. The one new output surface — the refusal message — was written to the register's `T-idh-01` requirement: it carries the raw root path, this run's succeeded/failed counts and the name of the manifest reader, and no credential value or vendor response body. `T-idh-02` (the conditionalised RAM guard) is held by `tests/test_volume_guard.py`'s two AST ordering assertions plus the new reachability arm; the guard's lexical position was not moved. `T-idh-03` (roster sort) is held by the 26 pre-existing membership assertions staying green. `T-idh-05` — `--to-zarr` is a `store_true` taking no value and participating in no path construction — is asserted directly in `test_every_shell_registers_to_zarr_and_defaults_to_not_converting`.

## Known Stubs

None. No `TODO`, `FIXME`, placeholder, skipped test or unrun `<verify>` was introduced; every `<verify>` block in the plan was executed, and the two greps the doc sweep specified were run rather than assumed.

## Verification Results

| Plan verification item | Result |
|---|---|
| 1. `uv run pytest -q` all green | **744 passed** |
| 2. `--to-zarr` on all three parsers (baseline: alpaca False / tiingo False / us_equity True) | **alpaca True / tiingo True / us_equity True** |
| 3. `cal.py` / `pyproject.toml` / `test.py` / `uv.lock` never committed | **confirmed** — `git diff --name-only f0abcd8..HEAD` lists none of them. Note: this worktree never carried those modifications; they exist only in the primary checkout |
| 4. Mutation results recorded, at least one arm red | **six of eight arms red**, table above |
| Task 3's doc truth gate | **`doc truth gate OK`** |

## Self-Check: PASSED

- `tests/test_ingest_conversion_gate.py` — FOUND
- commit `eebf2b3` — FOUND
- commit `1d71521` — FOUND
- commit `41e4b4a` — FOUND
- `git rev-list --count f0abcd8..HEAD` → **3**, matching the `commits: 3` recorded above

---

## Orchestrator follow-up: the guard shipped one line too late (cf215bc)

Added after this summary was written, during the orchestrator's own verification pass.

**The miss.** Re-running the exact UAT reproduction — `ingest_alpaca.py` with present-but-invalid
Alpaca credentials, all 3 symbols 401, `--to-zarr` — still ended on the same
`ValueError: StockDataset: no raw data for vendor 'alpaca'` traceback the task set out to
remove. The guard was present, correct, and unreachable.

**Why.** `BaseDataset`'s config setter calls `_reset_symbols()` for any non-None symbol list,
which calls `read()`, catches the not-yet-written store's `FileNotFoundError`, and falls back to
`from_raw_data()` — a full densification at CONSTRUCTION time. `ingest_alpaca.py` and
`ingest_tiingo.py` both build `ds_config` with `symbols=list(symbols)`, so
`dataset = StockDataset(ds_config)` raised before `refuse_conversion_without_raw_data(dataset,
result)` on the next line could run. `ingest_us_equity.py` was unaffected: its `ds_config`
carries `symbols=None` from the factory — the one shell where the guard actually worked was the
third door this task added, not either of the two the bug reports named. This is the same trap
`ingest_us_equity.py` already documents at its own `stock_kline_config` call.

**Why the suite stayed green.** `test_the_refusal_precedes_the_densification`'s AST scan compares
the refusal against explicit `from_raw_data` / `from_raw_data_chunked` call sites. The implicit
densification inside `StockDataset(...)` carries no such token, so the shipped ordering read
correctly and passed. That is the blind spot, not an oversight in the assertion's own terms.

**The fix.** All three shells now hand the guard a symbol-free probe,
`StockDataset(replace(ds_config, symbols=None))`, and construct the real dataset only after it
returns. `ingest_us_equity.py`'s `replace` is redundant at runtime and kept deliberately: it
states the property at the call site where it is relied on rather than one factory call away,
and it is what lets a single rule cover all three shells.

`test_refusal_precedes_every_symbol_bearing_dataset_construction` orders the refusal against the
CONSTRUCTION, exempting only the `replace(..., symbols=None)` shape. Mutation-verified: restoring
the shipped two-line order in `ingest_alpaca.py` turns it red.

**Verified after the fix:** the reproduction exits 1 with the guard's message and no traceback;
`uv run pytest` → 745 passed.

**Deviation record.** This was applied by the orchestrator rather than the executor, whose
worktree had already been merged and removed. Same atomic-commit discipline; the four unrelated
working-tree files (`cal.py`, `pyproject.toml`, `test.py`, `uv.lock`) were never staged.
