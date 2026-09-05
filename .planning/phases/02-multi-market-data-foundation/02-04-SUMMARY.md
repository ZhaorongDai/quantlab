---
phase: 02-multi-market-data-foundation
plan: 04
subsystem: data-acquisition
tags: [tiingo, polars, acquisition, watermark, tdd, us-equity]

# Dependency graph
requires:
  - phase: 02-02
    provides: "AcquisitionConfig dataclass (market, frequency, raw_data_dir_path, watermark_path, symbols, start_date, end_date, kwargs, name) with no credential field"
  - phase: 02-03
    provides: "clean_market_data()/validate_schema()/flag_anomalies() shared cleaning module, consumed downstream by StockDataset once raw files land"
provides:
  - "base/acquisition.py:Acquisition(ABC) -- config lifecycle mirroring Dataset, per-symbol JSON watermark read/write (tolerant of missing/corrupt files), download()/refresh() orchestration, abstract _fetch_and_write()"
  - "acquisition/tiingo.py:TiingoAcquisition(Acquisition) -- config-driven, frequency-parameterized, incrementally-refreshable Tiingo EOD data acquisition with zero network calls in tests"
  - "enums/data.py:TiingoColumns.EOD -- explicit, auditable columns= constant for get_ticker_price()"
  - "Full unit test coverage (4/4) proving credential safety, incremental refresh, and watermark write, all via a mocked TiingoClient"
affects: [02-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Acquisition(ABC) as a new sibling lifecycle to Dataset -- network I/O and local-file writes only, never touches xarray/Zarr; Dataset stays responsible for converting already-local raw files"
    - "Per-symbol JSON watermark sidecar ({raw|watermark}_dir/{symbol}.json holding {\"last_date\": ...}) for incremental refresh, corrupt/missing-file-tolerant by design (falls back to None/config.start_date, never raises)"
    - "TIINGO_API_KEY read from os.environ only inside __init__, checked before any network call, never assigned to config or any dataclass-facing attribute"

key-files:
  created:
    - base/acquisition.py
    - acquisition/__init__.py
    - acquisition/tiingo.py
    - tests/test_tiingo_acquisition.py
  modified:
    - enums/data.py

key-decisions:
  - "Parse Tiingo's tz-aware ISO-8601 date string (trailing Z/UTC offset) via str.to_datetime(time_zone=\"UTC\") then dt.replace_time_zone(None), producing a naive pl.Datetime timestamp column consistent with the naive-timestamp convention used elsewhere in the codebase (StockDataset's str.to_datetime() filter bounds)"
  - "Fixed Test 4 (credential-surface safety) to assert against config.to_dict().keys() rather than a JSON-serialized substring match -- the original substring check produced a false positive because pytest's tmp_path embeds the test function name, which can coincidentally contain a forbidden substring"

patterns-established:
  - "Acquisition/Dataset separation of concerns: Acquisition subclasses fetch over the network and write local raw files under config.raw_data_dir_path; the matching Dataset subclass's _raw_data_to_xr() is solely responsible for converting those files into the canonical xarray representation. No download()/refresh()-shaped method exists anywhere on Dataset."

requirements-completed: [DATA-01]

# Metrics
duration: ~35min (across an interrupted session, resumed and completed)
completed: 2026-09-05
---

# Phase 02 Plan 04: Tiingo Acquisition Layer Summary

**Class-based `Acquisition(ABC)` + `TiingoAcquisition`, mirroring the `Dataset`/`FactorKunQuant` ABC+concrete lifecycle pattern, supporting frequency-parameterized fetch and incremental "since-last-watermark" refresh via a mocked `TiingoClient` -- zero live network calls required.**

## Performance

- **Duration:** ~35 min total (session was interrupted by a rate-limit error mid-Task-2 GREEN phase and resumed to completion)
- **Started:** 2026-09-04T22:30:06Z (Task 1 commit)
- **Completed:** 2026-09-05 (this session)
- **Tasks:** 2 completed
- **Files modified:** 5 (2 created new modules, 1 new package, 1 new test file, 1 enum file extended)

## Accomplishments
- `Acquisition(ABC)` established as a new sibling lifecycle to `Dataset`, fully decoupled from the Dataset/Zarr layer -- config property/setter idiom, per-symbol JSON watermark I/O, `download()`/`refresh()` orchestration, abstract `_fetch_and_write()`
- `TiingoAcquisition` implemented: reads `TIINGO_API_KEY` from `os.environ` only (raises `RuntimeError` before any network call if unset), always passes an explicit `columns=TiingoColumns.EOD`, supports frequency parameterization via `_FREQUENCY_MAP` (not hardcoded `"daily"`), writes to the exact per-symbol subdirectory layout `StockDataset._raw_data_to_xr()` already discovers via `rglob("*.pqt")`
- 4/4 TDD behaviors proven green with a mocked `TiingoClient` fixture: missing-key gate, download+watermark write, incremental refresh (uses watermark date, not `config.start_date`), and credential-surface safety (never on `AcquisitionConfig.to_dict()` or any config-facing attribute)

## Task Commits

Each task was committed atomically:

1. **Task 1: Define base/acquisition.py:Acquisition(ABC)** - `2f86164` (feat)
2. **Task 2: Implement acquisition/tiingo.py:TiingoAcquisition (TDD)**
   - RED: `32cc929` (test) -- failing tests added, `ModuleNotFoundError` as expected (acquisition package did not exist yet)
   - GREEN: `a679073` (feat) -- TiingoAcquisition implementation + enums/data.py:TiingoColumns + test-file fix, all 4 tests passing

_Note: this plan's execution was interrupted between the RED and GREEN commits by a session rate-limit error (not a real failure); the GREEN-phase implementation files (`acquisition/tiingo.py`, the `TiingoColumns` addition to `enums/data.py`, and a test-file fix) were present uncommitted in the worktree at resume time, verified correct, and committed as the single GREEN commit above once a pre-existing date-parsing bug (see Deviations) was fixed._

**Plan metadata:** (this commit) - `docs: complete plan`

## Files Created/Modified
- `base/acquisition.py` - `Acquisition(ABC)`: config lifecycle, watermark read/write, `download()`/`refresh()` orchestration, abstract `_fetch_and_write()`
- `acquisition/__init__.py` - empty, per the project's no-re-export package convention
- `acquisition/tiingo.py` - `TiingoAcquisition(Acquisition)`: credential gate, `_FREQUENCY_MAP`, `_fetch_and_write()` (explicit `columns=`, tz-aware date parsing, per-symbol parquet write)
- `enums/data.py` - added `TiingoColumns` dataclass-of-constants (`EOD` = 12-field comma-joined string)
- `tests/test_tiingo_acquisition.py` - 4 tests covering the full behavior contract, using `mock_tiingo_client`/`tiingo_json_response` fixtures from `tests/conftest.py`

## Decisions Made
- Naive-timestamp normalization: Tiingo's `date` field carries a `Z` (UTC) suffix; parsed as UTC then stripped of tz info to stay consistent with the naive-timestamp convention `StockDataset` already assumes elsewhere in the pipeline (see Deviations below for why this was necessary, not optional).
- Test 4 assertion changed from a serialized-JSON substring check to a `dict.keys()` membership check, to eliminate a false-positive risk from `tmp_path`'s test-name-derived directory components.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed polars `ComputeError` on Tiingo's tz-aware date strings**
- **Found during:** Task 2 (GREEN phase, verifying `_fetch_and_write`)
- **Issue:** The fixture's `tiingo_json_response` `date` field is an ISO-8601 string with a trailing `Z` (e.g. `"2024-01-02T00:00:00.000Z"`). Calling `pl.col("date").str.to_datetime()` with no format and no time zone on a tz-aware string string raises `polars.exceptions.ComputeError` ("was called with no format and no time zone, but a time zone is part of the data") in the installed polars version -- this broke 2 of the 4 planned tests (`test_download_writes_parquet_and_watermark`, `test_refresh_uses_watermark_not_config_start_date`).
- **Fix:** Changed the date-parsing expression to `pl.col("date").str.to_datetime(time_zone="UTC").dt.replace_time_zone(None)`, producing a naive `pl.Datetime` column consistent with the naive-timestamp convention `StockDataset`'s own `str.to_datetime()` filter bounds already assume.
- **Files modified:** `acquisition/tiingo.py`
- **Verification:** `uv run pytest tests/test_tiingo_acquisition.py -v` -- 4/4 passed (previously 2/4).
- **Committed in:** `a679073` (Task 2 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix, Rule 1)
**Impact on plan:** Fix was necessary for the plan's own stated behavior (Test 2/Test 3) to pass at all -- no scope creep, no architectural change.

## Issues Encountered
- Session was interrupted by a rate-limit error partway through Task 2's GREEN-phase implementation. On resume, git state was inspected directly (`git log`, `git status`, `git diff`) rather than trusting the interruption summary; the uncommitted `enums/data.py`/`tests/test_tiingo_acquisition.py` changes and untracked `acquisition/` directory were verified to be complete, correct implementation work (not partial/broken), fixed for the one outstanding bug above, and committed as a single GREEN commit.

## User Setup Required
None - no external service configuration required. `TIINGO_API_KEY` must be exported by the user before running acquisition against the live Tiingo API in Wave 4 (02-07), but that is unchanged from the plan's design and requires no action for this plan's test suite (all tests use a mocked client).

## Next Phase Readiness
- `Acquisition(ABC)` + `TiingoAcquisition` are ready for Wave 4 (02-07) to wire end-to-end with `StockDataset` and add the user-facing entry-point script.
- No blockers. `base/config.py` still has zero credential-shaped fields (`grep -iE "api_key|credential|token|secret" base/config.py` returns no matches).

---
*Phase: 02-multi-market-data-foundation*
*Completed: 2026-09-05*
