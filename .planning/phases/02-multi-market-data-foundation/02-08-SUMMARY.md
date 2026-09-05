---
phase: 02-multi-market-data-foundation
plan: 08
subsystem: data
tags: [tiingo, wikipedia, polars, lxml, point-in-time, survivorship-bias, universe]

# Dependency graph
requires:
  - phase: 02-multi-market-data-foundation (plans 01-07)
    provides: TiingoAcquisition, StockDataset, stock_acquisition_config()/stock_kline_config(), PlBackend
provides:
  - "acquisition/universe.py: NasdaqUniverseFetcher, SP500MembershipFetcher, UniverseCatalog"
  - "Point-in-time, survivorship-bias-free US-equity universe reference table (data/reference/universe.parquet)"
  - "ingest_tiingo.py --universe {sp500,nasdaq_all} --as-of-date symbol resolution"
  - "refresh_us_equity_universe.py CLI entry point"
affects: [phase-05-return-model, phase-06-portfolio-optimization, backtest-universe-construction]

# Tech tracking
tech-stack:
  added: [lxml 6.1.3]
  patterns:
    - "Standalone reference-data class (not Acquisition(ABC) subclass) for roster/interval-table shapes that don't fit the per-symbol-watermark OHLCV contract"
    - "Forward-chronological event simulation for point-in-time interval reconstruction from an anchor + change-log pair"
    - "Fetch-validate-fallback-to-cache pattern for scrape-fragile external sources (never overwrite cache on validation failure)"

key-files:
  created:
    - acquisition/universe.py
    - refresh_us_equity_universe.py
    - tests/test_universe.py
    - tests/test_ingest_tiingo_universe_wiring.py
    - .planning/phases/02-multi-market-data-foundation/02-08-REAL-DATA-CHECK.md
  modified:
    - pyproject.toml
    - uv.lock
    - enums/data.py
    - base/config.py
    - config/__init__.py
    - ingest_tiingo.py
    - tests/conftest.py

key-decisions:
  - "Universe/interval table is reference/metadata, exempt from the xarray/Zarr pipeline-format constraint (Locked Decision A1) -- persisted via existing PlBackend/parquet, same footing as config/instruments.yaml"
  - "NASDAQ-listed common stock only, no OTC/Expert-Market tiers (Locked Decision A4)"
  - "Nasdaq-universe source is Tiingo's own supported_tickers.csv, not nasdaqlisted.txt -- the latter cannot represent delisted history at all"
  - "S&P 500 membership reconstructed via forward-chronological event simulation over Wikipedia's Historical_components_of_the_S&P_500 change log, anchored against datahub's constituents.csv"
  - "PIT_COVERAGE_START = 1976-07-01 is a hard, explicitly-rejected boundary -- queries before it raise ValueError rather than silently answering incompletely"
  - "Alpaca material in 02-08-RESEARCH.md is a hallucination artifact and was not referenced or acted upon anywhere in this implementation"

patterns-established:
  - "Reference/config-tier persistence (PlBackend/parquet) is a valid alternative to the Dataset/Factor/Model xarray/Zarr pipeline tier for non-timeseries metadata tables"
  - "get_symbols_as_of(category, as_of_date) must be called per-rebalance-date in future walk-forward backtests, not once at setup time (documented in UniverseCatalog's docstring for forward reference)"

requirements-completed: [DATA-01]

# Metrics
duration: ~25min
completed: 2026-09-05
---

# Phase 2 Plan 08: Point-in-Time US Equity Universe Summary

**Survivorship-bias-free, point-in-time US-equity universe (NASDAQ roster including delisted stocks + Wikipedia-reconstructed S&P 500 membership intervals) persisted via PlBackend/parquet and wired into `ingest_tiingo.py` via a new `--universe`/`--as-of-date` option.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-09-05T15:15:00Z (approx.)
- **Completed:** 2026-09-05T15:41:00Z
- **Tasks:** 4 completed
- **Files modified:** 12 (7 modified, 5 created)

## Accomplishments
- `acquisition/universe.py` with three single-responsibility classes (`NasdaqUniverseFetcher`, `SP500MembershipFetcher`, `UniverseCatalog`), none subclassing `Acquisition(ABC)` since a roster/interval table is a different shape than per-symbol OHLCV.
- Point-in-time interval reconstruction via forward-chronological event simulation, correctly handling re-entry (added→removed→re-added), left-censoring (removal with no matching prior add, sentineled at `PIT_COVERAGE_START`), and graceful degradation (Wikipedia parse failure/schema-drift/row-count-regression falls back to cached snapshot without corrupting it).
- `ingest_tiingo.py --universe sp500 --as-of-date <date>` resolves a real symbol list with zero code changes inside `TiingoAcquisition`/`StockDataset`.
- Real-data verification against live Tiingo/Wikipedia/GitHub sources (network was available): confirmed Tesla's actual 2020-12-21 S&P 500 addition and `ATVI`'s actual 2023-10-13 NASDAQ delisting date, and confirmed `TWTR` is correctly excluded from `nasdaq_all` (it trades on NYSE, not NASDAQ) — see `02-08-REAL-DATA-CHECK.md`.

## Task Commits

Each task was committed atomically:

1. **Task 1: lxml dependency + UniverseConfig contract + config factory** - `331fa1b` (feat)
2. **Task 2: acquisition/universe.py — NASDAQ roster + point-in-time S&P 500 reconstruction** - `60099a8` (feat)
3. **Task 3: Tests — point-in-time correctness, left-censoring, re-entry, graceful degradation** - `7502379` (test)
4. **Task 4: Wire universe resolution into ingest_tiingo.py + standalone refresh entry point** - `5f25f51` (feat)

**Additional:** `3d6b981` (docs: real-data verification results, non-gating per plan design)

_Note: Task 2 is marked `tdd="true"` in the plan, but the plan itself explicitly splits implementation (Task 2) and the full behavioral test suite (Task 3) into separate tasks, with Task 2's `<done>` criteria stating "full behavioral proof is provided by Task 3's test suite." This means the commit order is feat→test rather than canonical TDD's test→feat. This is the plan's own explicit design (not a shortcut taken during execution) — see "TDD Gate Compliance" below._

## Files Created/Modified
- `acquisition/universe.py` - `NasdaqUniverseFetcher`, `SP500MembershipFetcher`, `UniverseCatalog`
- `refresh_us_equity_universe.py` - thin CLI entry point (`UniverseCatalog(config).build().save()`)
- `base/config.py` - `UniverseConfig` dataclass
- `config/__init__.py` - `universe_config()` factory (`data/reference/` path convention)
- `enums/data.py` - `UniverseCategory` Literal alias
- `ingest_tiingo.py` - `--universe {sp500,nasdaq_all} --as-of-date` option, argparse validation
- `tests/conftest.py` - `sp500_anchor_csv_rows`, `sp500_changes_html_fixture`, `mock_universe_fetchers` fixtures
- `tests/test_universe.py` - 7 tests covering all D-12 behavioral requirements
- `tests/test_ingest_tiingo_universe_wiring.py` - 3 tests proving the wiring never touches `TiingoAcquisition`/`StockDataset`
- `pyproject.toml`/`uv.lock` - `lxml>=6.1.3` dependency (required by `pandas.read_html()`)
- `.planning/phases/02-multi-market-data-foundation/02-08-REAL-DATA-CHECK.md` - live verification results

## TDD Gate Compliance

Task 2 (`tdd="true"`) implementation landed in commit `60099a8` (feat) before
the corresponding test suite landed in `7502379` (test) — this is the
opposite of canonical RED-before-GREEN ordering. This was **not** a shortcut
taken during execution: `02-08-PLAN.md` explicitly structures Task 2 and
Task 3 as separate tasks, with Task 2's own `<done>` criteria stating "full
behavioral proof is provided by Task 3's test suite" — the plan's author
deliberately split implementation and test-writing across two commits rather
than requiring a single-task RED/GREEN/REFACTOR cycle. All 7 behavioral
tests in Task 3 pass against the Task 2 implementation with zero
implementation changes required, so no regression or gap resulted from this
ordering.

## Decisions Made
- Followed the plan's Locked Decisions A1 (universe table is reference/metadata, `PlBackend`/parquet, not xarray/Zarr) and A4 (NASDAQ-listed common stock only) exactly as specified — no deviation.
- Used a temporary `loguru` sink (not `caplog`) to assert on the left-censoring warning message, since `loguru` does not propagate to stdlib `logging` by default and `caplog` alone would not capture it — the plan explicitly allowed "via caplog or a loguru sink."
- Did not reference or act on the "Alpaca" material in `02-08-RESEARCH.md` anywhere in this implementation, per explicit instruction that it was a hallucination artifact from the research session.

## Deviations from Plan

None - plan executed exactly as written. All acceptance criteria for all four tasks passed on first or second attempt (one test-only fix: switched from `caplog` to a `loguru` sink for the left-censoring warning assertion, within the plan's own stated allowance).

## Issues Encountered
- Initial `test_reconstruct_intervals_left_censored` test used pytest's `caplog` fixture, which does not capture `loguru` output by default (no stdlib-`logging` propagation configured anywhere in the codebase). Fixed by attaching a temporary `loguru.logger.add(...)` sink for the duration of the test instead — no production code change needed, test-only fix.
- This worktree's branch initially had a single disconnected "Initial commit" (`64200da`) not descended from the expected base (`c6e8095`, containing all of Phase 2's merged work + this plan's own research/plan-authoring commits). Corrected via `git reset --hard c6e8095b3bbb449dc28963f8b35dcd731aeecd49` per the mandatory `<worktree_branch_check>` step before any task work began.

## User Setup Required

None - no external service configuration required. Universe-table construction needs no `TIINGO_API_KEY` (confirmed both by test suite and live network verification) — only the existing, unchanged OHLCV-fetch step (`TiingoAcquisition`) requires that credential, and this plan does not touch it.

## Next Phase Readiness
- `UniverseCatalog.get_symbols_as_of(category, as_of_date)` is ready for future walk-forward backtest phases (Phase 5/6) to resolve a correct, look-ahead-free historical universe at any rebalance date — its docstring explicitly notes it must be called per-rebalance-date, not once at setup time, to avoid look-ahead bias.
- `data/reference/universe.parquet` (built once via `refresh_us_equity_universe.py`) is the persisted artifact those future phases will read via `UniverseCatalog.load(universe_config())`.
- No blockers. Full test suite (`uv run pytest tests/ -x -q`) passes: 39/39, including all Phase 2 plans 01-07's pre-existing tests (no regressions).

---
*Phase: 02-multi-market-data-foundation*
*Completed: 2026-09-05*

## Self-Check: PASSED

All claimed files verified present on disk:
- FOUND: acquisition/universe.py
- FOUND: refresh_us_equity_universe.py
- FOUND: tests/test_universe.py
- FOUND: tests/test_ingest_tiingo_universe_wiring.py
- FOUND: .planning/phases/02-multi-market-data-foundation/02-08-REAL-DATA-CHECK.md
- FOUND: .planning/phases/02-multi-market-data-foundation/02-08-SUMMARY.md

All claimed commits verified present in `git log`:
- FOUND: 331fa1b, 60099a8, 7502379, 5f25f51, 3d6b981, e5710e2

No missing items.
