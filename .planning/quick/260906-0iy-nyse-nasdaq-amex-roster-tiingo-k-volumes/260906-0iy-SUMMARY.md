---
phase: quick-260906-0iy
plan: 01
subsystem: data
tags: [tiingo, polars, universe, survivorship-bias, joblib, threading, xarray, zarr]

requires:
  - phase: 03.1-index-historical-constituents
    provides: "UniverseCatalog, MEMBERSHIP_FETCHERS registry, NasdaqUniverseFetcher, PlBackend-backed universe.parquet"
  - phase: 02-data-acquisition
    provides: "Acquisition base with per-symbol watermarks, TiingoAcquisition, stock_acquisition_config/stock_kline_config path convention"
provides:
  - "`us_all` universe category — the full NYSE + NASDAQ + AMEX common-stock roster, delisted included (16,138 rows / 15,425 distinct tickers, measured live)"
  - "`TiingoRosterFetcher` base making an exchange roster a data change (three class constants), not a code change"
  - "`UniverseCatalog.get_symbols_in_range()` — interval-overlap roster query, the survivorship-bias-free backfill primitive"
  - "`UniverseCatalog.estimate_dense_panel()` / `assert_dense_panel_fits()` — dense-panel storage sizing and a 4 GiB RAM guard"
  - "`ConcurrentTiingoAcquisition` — resumable, concurrent, failure-isolated bulk ingest with a credential-scrubbed failure manifest"
  - "`ingest_us_equity.py` — glue-only bulk backfill entry point with a zero-request `--dry-run`"
  - "`subdir`/`store_name` arguments on the stock config factories, so a second roster lands beside the first without a second path root"
affects: [factor computation, model training, backtesting, any cross-sectional US-equity research]

actuals:
  tokens: 19627
  tasks: 3
  commits: 6

tech-stack:
  added: []
  patterns:
    - "Roster-as-data: a second exchange roster is three class constants on TiingoRosterFetcher, mirroring the IndexMembershipFetcher precedent"
    - "Two-registry catalog: ROSTER_FETCHERS (boundary-free) vs MEMBERSHIP_FETCHERS (PIT_COVERAGE_START-bound), so a roster cannot inherit a coverage boundary by accident"
    - "Orchestration-only subclassing: ConcurrentTiingoAcquisition overrides download/refresh but inherits _fetch_and_write untouched, leaving base/acquisition.py's sequential default intact"
    - "Credential scrubbing at a single choke point: every captured exception message passes through _scrub() before it is logged or written"
    - "Sizing guard before allocation: assert_dense_panel_fits() raises with the measured numbers rather than letting the pandas densification OOM"

key-files:
  created:
    - ingest_us_equity.py
  modified:
    - acquisition/universe.py
    - acquisition/tiingo.py
    - config/__init__.py
    - enums/data.py
    - ingest_tiingo.py
    - tests/conftest.py
    - tests/test_universe.py
    - tests/test_tiingo_acquisition.py
    - tests/test_config_paths.py

key-decisions:
  - "The AMEX is filtered under BOTH `AMEX` and `NYSE MKT` because Tiingo never re-labelled its historical rows across the exchange's rename history; there is no `NYSE American` token and omitting either silently drops ~224 real tickers"
  - "`NYSE ARCA` / `NYSE NAT` / `BATS` are excluded despite the shared brand — they are different exchanges and ARCA is predominantly ETFs"
  - "Roster fetchers live in their own ROSTER_FETCHERS registry, deliberately outside MEMBERSHIP_FETCHERS, so neither roster inherits a PIT_COVERAGE_START boundary it must not have (D-02)"
  - "The 2006-01-01 cut lives in get_symbols_in_range() as interval overlap, not as a MIN_END_DATE constant on the fetcher — baking a window into the reference table would make it unusable for any other window"
  - "The two roster fetchers each download supported_tickers.zip independently; a shared cache would either leak mocked bytes across tests or change NasdaqUniverseFetcher.fetch()'s observable behaviour, which D-02 forbids"
  - "max_workers/resume are read from config.kwargs rather than becoming constructor arguments, keeping the concurrency knobs reachable from a config file per CLAUDE.md"
  - "MAX_DENSE_PANEL_BYTES is 4 GiB — below the measured 7.19 GiB that OOMs this 16 GiB machine and above the windows that fit. Disk is not the constraint (120 GiB free); RAM is"
  - "--to-zarr runs the sizing guard BEFORE the download, not merely before from_raw_data(), so a user is refused immediately rather than after a multi-hour backfill"

patterns-established:
  - "Roster-as-data: adding an exchange roster is EXCHANGE_FILTER + MIN_ROSTER_ROWS + CATEGORY, nothing else"
  - "Safety envelopes sized to their own magnitude: MIN_ROSTER_ROWS 1000/~10k (nasdaq_all), 8000/~16k (us_all), MIN_ANCHOR_ROWS 50/~102 (NDX)"
  - "Failure manifests as the audit trail for partially-completed long jobs: _failures.json beside the per-symbol watermarks, describing the latest run"

requirements-completed: [DATA-05]

coverage:
  - id: D1
    description: "`us_all` universe category holding the full NYSE + NASDAQ + AMEX common-stock roster, delisted tickers included"
    requirement: "DATA-05"
    verification:
      - kind: unit
        ref: "tests/test_universe.py#test_us_equity_fetcher_filters_nyse_nasdaq_amex_and_excludes_others"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_catalog_build_emits_all_four_categories"
        status: pass
      - kind: integration
        ref: "QUANTLAB_DATA_DIR=/Volumes/SSD/data uv run python refresh_us_equity_universe.py -> us_all = 16,138 rows / 15,425 distinct"
        status: pass
    human_judgment: false
  - id: D2
    description: "`nasdaq_all` and NasdaqUniverseFetcher.EXCHANGE_FILTER provably unchanged (Locked Decision A4 / D-02)"
    verification:
      - kind: unit
        ref: "tests/test_universe.py#test_nasdaq_roster_exchange_filter_and_symbol_set_are_unchanged"
        status: pass
      - kind: other
        ref: "python -c \"from acquisition.universe import NasdaqUniverseFetcher as N; assert N.EXCHANGE_FILTER == ('NASDAQ',)\""
        status: pass
      - kind: integration
        ref: "live rebuild: nasdaq_all = 9,318 rows / 8,967 distinct, get_symbols_as_of('nasdaq_all','2026-09-04') = 4,848"
        status: pass
    human_judgment: false
  - id: D3
    description: "Interval-overlap roster query keeping post-2006 delistings and dropping only pre-2006-ended tickers"
    requirement: "DATA-05"
    verification:
      - kind: unit
        ref: "tests/test_universe.py#test_get_symbols_in_range_keeps_post_2006_delistings"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_get_symbols_in_range_deduplicates_a_dual_listed_ticker"
        status: pass
      - kind: integration
        ref: "live: 15,425 distinct roster tickers -> 15,424 in 2006-01-01..2026-09-06 (exactly one 1997 delisting dropped, matching planning measurement)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Resumable, concurrent, failure-isolated bulk Tiingo acquisition"
    verification:
      - kind: unit
        ref: "tests/test_tiingo_acquisition.py#test_second_download_skips_symbols_already_at_the_watermark"
        status: pass
      - kind: unit
        ref: "tests/test_tiingo_acquisition.py#test_one_symbol_failure_does_not_abort_the_others"
        status: pass
      - kind: unit
        ref: "tests/test_tiingo_acquisition.py#test_concurrent_refresh_starts_each_symbol_from_its_own_watermark"
        status: pass
    human_judgment: true
    rationale: "Resume, concurrency and failure isolation are proven against the mocked client offline. TIINGO_API_KEY is not set in this environment (a hard constraint of the run), so the real ~15.4k-symbol backfill and its wall-clock/rate-limit behaviour under a paid-tier key have not been exercised end to end."
  - id: D5
    description: "No credential reachable from any log or artifact (T-0iy-01)"
    verification:
      - kind: unit
        ref: "tests/test_tiingo_acquisition.py#test_failure_manifest_never_contains_the_api_key"
        status: pass
      - kind: unit
        ref: "tests/test_tiingo_acquisition.py#test_credential_never_exposed_on_config_surface"
        status: pass
    human_judgment: false
  - id: D6
    description: "Dry-run mode reporting roster and storage estimate with zero price requests, plus a sizing guard blocking the full-market densification"
    verification:
      - kind: unit
        ref: "tests/test_universe.py#test_estimate_dense_panel_reports_a_coherent_density"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_assert_dense_panel_fits_raises_with_the_numbers"
        status: pass
      - kind: integration
        ref: "QUANTLAB_DATA_DIR=/Volumes/SSD/data uv run python ingest_us_equity.py --dry-run (no TIINGO_API_KEY set) -> 15,424 symbols, 7.19 GiB dense, density 0.368, zero requests"
        status: pass
    human_judgment: false
  - id: D7
    description: "All storage rooted at QUANTLAB_DATA_DIR — no hardcoded volume, no second path knob"
    verification:
      - kind: unit
        ref: "tests/test_config_paths.py#test_stock_config_defaults_are_byte_identical_without_the_new_arguments"
        status: pass
      - kind: unit
        ref: "tests/test_config_paths.py#test_stock_config_subdir_and_store_name_redirect_under_the_same_root"
        status: pass
      - kind: other
        ref: "grep -rn 'Volumes/SSD' ingest_us_equity.py config/__init__.py acquisition/ -> no matches"
        status: pass
    human_judgment: false

duration: 15 min
completed: 2026-09-06
status: complete
---

# Quick 260906-0iy: NYSE + NASDAQ + AMEX Roster and Bulk Tiingo Ingest Summary

**A `us_all` full-US-market universe category (15,425 distinct tickers, delisted included) as a sibling of the untouched `nasdaq_all`, plus a resumable, concurrent, credential-scrubbing Tiingo bulk ingest gated by a measured 4 GiB dense-panel RAM guard.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-09-06T04:31:00Z
- **Completed:** 2026-09-06T04:46:00Z
- **Tasks:** 3
- **Files modified:** 10 (1 created, 9 modified)

## Accomplishments

- **`us_all` roster.** `USEquityUniverseFetcher` filters Tiingo's 108,561-row directory on `exchange in (NASDAQ, NYSE, AMEX, NYSE MKT)` + `Stock` + `USD`, yielding exactly the planning-time measurement when run live: **16,138 rows / 15,425 distinct tickers**. The AMEX needs two tokens because Tiingo never re-labelled its historical rows across the AMEX → NYSE Amex → NYSE MKT → NYSE American renames; there is no `NYSE American` token at all.
- **`nasdaq_all` provably untouched.** The shared body moved onto a new `TiingoRosterFetcher` base, but `NasdaqUniverseFetcher.EXCHANGE_FILTER == ("NASDAQ",)` is now pinned by direct equality (not a grep), and the live rebuild reproduces 9,318 rows / 8,967 distinct exactly as before.
- **Survivorship-bias-free backfill query.** `get_symbols_in_range()` returns every symbol whose listing interval *overlaps* the window. Live: 15,425 roster tickers → **15,424** over 2006-01-01..2026-09-06 — the 2006 cut drops exactly one ticker (a 1997 delisting) while keeping all ~6.9k that delisted *inside* the window.
- **Bulk ingest that survives a 15k-symbol, multi-hour job.** `ConcurrentTiingoAcquisition` fans out over threads, skips symbols already at the target watermark, captures per-symbol failures instead of propagating them, and writes only successful watermarks so failures retry next run.
- **Measured sizing guard.** `estimate_dense_panel()` + `assert_dense_panel_fits()` reproduce the planning figures live: 15,424 × 5,212 trading days = 80,389,888 dense cells, 29,571,621 real observations, **density 0.368**, **7.19 GiB** dense float64 against a **4.00 GiB** budget — so `--to-zarr` on the full window is refused with a legible message instead of OOMing a 16 GiB machine.
- **Zero-request dry run.** `ingest_us_equity.py --dry-run` resolved the full roster and printed the complete size breakdown **with `TIINGO_API_KEY` unset**, confirming it issues no price request and needs no credential.

## Task Commits

1. **Task 1: `us_all` roster category (tracer, TDD)** — `fe7d577` (test, RED) → `03dfdee` (feat, GREEN)
2. **Task 2: concurrent/resumable acquisition (TDD)** — `69bf31c` (test, RED) → `5b3ec2d` (feat, GREEN)
3. **Task 3: sizing guard, config wiring, entry point (TDD)** — `26d0329` (test, RED) → `4ae175e` (feat, GREEN)

## Files Created/Modified

- `ingest_us_equity.py` — **created.** Glue-only bulk backfill entry point. Resolves the roster by interval overlap, drives `ConcurrentTiingoAcquisition`, and offers `--dry-run` / `--limit` / `--refresh` / `--max-workers` / `--to-zarr`.
- `acquisition/universe.py` — `TiingoRosterFetcher` base, `USEquityUniverseFetcher`, `ROSTER_FETCHERS` registry, `get_symbols_in_range()`, `estimate_dense_panel()`, `assert_dense_panel_fits()`, shared `_validate_category`/`_validate_iso_date`.
- `acquisition/tiingo.py` — `ConcurrentTiingoAcquisition` (threading fan-out, watermark resume, failure isolation, `_failures.json`, `_scrub()`).
- `config/__init__.py` — optional `subdir`/`store_name` on `stock_acquisition_config()`/`stock_kline_config()`.
- `enums/data.py` — `us_all` added to `UniverseCategory`, with an explicit warning that it must not become an alias of `nasdaq_all`.
- `ingest_tiingo.py` — `us_all` registered in `_UNIVERSE_CATEGORY_MAP`; help text points at `ingest_us_equity.py` for interval-overlap backfills.
- `tests/conftest.py` — synthetic roster CSV extended (append-only) with NYSE / AMEX / NYSE MKT / NYSE ARCA / dual-listed / pre-2006 rows.
- `tests/test_universe.py`, `tests/test_tiingo_acquisition.py`, `tests/test_config_paths.py` — 21 new tests.

## Decisions Made

See `key-decisions` in the frontmatter. The two most load-bearing:

1. **Two registries, not one.** `ROSTER_FETCHERS` is deliberately separate from `MEMBERSHIP_FETCHERS`. Registering a roster in the latter would give it a `PIT_COVERAGE_START` boundary, making a pre-boundary query *raise* instead of answering correctly from the roster's own listing dates. `test_roster_fetchers_registry_is_disjoint_from_membership_fetchers` makes the separation a tested property rather than a convention.
2. **The date window lives in the query, not the table.** No `MIN_END_DATE` constant was added to any fetcher. `universe.parquet` stores raw listing intervals and `get_symbols_in_range()` applies the window — otherwise the reference table would be frozen to one backfill's dates and unusable for any other, breaking the config-driven reproducibility constraint.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Sizing guard moved ahead of the download in `ingest_us_equity.py`**

- **Found during:** Task 3 (entry point wiring)
- **Issue:** The plan specified calling `assert_dense_panel_fits()` immediately before `StockDataset(...).from_raw_data()`. That satisfies "fires before the densification allocates", but it means a user running `--to-zarr` on the full window would spend several hours downloading ~15.4k symbols and only *then* be told the conversion is refused.
- **Fix:** The guard now runs right after the roster is resolved and before `ConcurrentTiingoAcquisition` is constructed, so the refusal is immediate. The stated invariant ("before `from_raw_data()` allocates") is strictly preserved — the check simply happens earlier.
- **Files modified:** `ingest_us_equity.py`
- **Verification:** Guard fires with the full-window message (7.19 GiB vs 4.00 GiB budget) and passes for a narrowed window; `uv run pytest tests/ -q` green.
- **Committed in:** `4ae175e`

**2. [Rule 1 - Bug] Case-sensitive assertion in a newly-written test**

- **Found during:** Task 3 (RED → GREEN transition)
- **Issue:** `test_assert_dense_panel_fits_raises_with_the_numbers` asserted `"narrow" in message`, but the error message begins that sentence with a capital `"Narrow"`. The test failed against a correct implementation.
- **Fix:** Assertion changed to `"narrow" in message.lower()`. This is a fix to a test written in this same plan, not a weakening of any pre-existing assertion.
- **Files modified:** `tests/test_universe.py`
- **Verification:** `uv run pytest tests/test_universe.py -q` green.
- **Committed in:** `4ae175e`

**3. [Rule 1 - Bug] Cosmetic formatting and a stale flag name in the new entry point**

- **Found during:** Task 3 (live dry-run verification)
- **Issue:** The dry-run printed `dense-panel budget:4.00 GiB` (missing space), and the module docstring referenced a `--symbols-limit` flag that is actually named `--limit`.
- **Fix:** Both corrected.
- **Files modified:** `ingest_us_equity.py`
- **Verification:** Re-ran the dry run; output aligned and the docstring now matches the parser.
- **Committed in:** `4ae175e`

---

**Total deviations:** 3 auto-fixed (1 missing critical, 2 bugs)
**Impact on plan:** No scope creep. Deviation 1 strengthens the guard's stated purpose; deviations 2–3 are corrections to code written in this same plan. Every Locked Decision (A4 / D-01..D-05) is intact.

## Issues Encountered

- **`get_symbols_as_of('nasdaq_all', '<today>')` returns `[]`.** Investigated and confirmed **pre-existing and correct**, not a regression: Tiingo populates `endDate` for *currently-listed* names with the last trading day (e.g. `2026-09-04`), so an as-of query for a later calendar date matches nothing. `get_symbols_as_of('nasdaq_all', '2026-09-04')` returns 4,848 and `'2024-01-02'` returns 3,992. This is exactly why `ingest_us_equity.py` uses `get_symbols_in_range()` rather than `get_symbols_as_of()` — a backfill must not depend on the roster's right edge landing on a trading day.
- **`us_all` as-of `1970-01-01` returns 9 symbols** (BA, GE, IBM, KO, DIS, …) rather than the empty list the synthetic fixture produces. Correct: those tickers carry real pre-1970 Tiingo `startDate` values. The tested property — that a roster answers from its own dates instead of raising a coverage-boundary error — holds either way.

## Verification

Plan `<verification>` steps, run in order:

| # | Step | Result |
|---|------|--------|
| 1 | `uv run pytest tests/ -q` | **144 passed** (baseline 123; +21 new, none weakened) |
| 2 | `NasdaqUniverseFetcher.EXCHANGE_FILTER == ('NASDAQ',)` | **exit 0** |
| 3 | `QUANTLAB_DATA_DIR=/Volumes/SSD/data uv run python refresh_us_equity_universe.py` | **`us_all` = 16,138 rows / 15,425 distinct**, in the required 15,000–16,000 range |
| 4 | `QUANTLAB_DATA_DIR=/Volumes/SSD/data uv run python ingest_us_equity.py --dry-run` | **15,424 symbols, 7.19 GiB dense estimate, paths rooted at `/Volumes/SSD/data`, zero price requests, ran with `TIINGO_API_KEY` unset** |
| 5 | `TIINGO_API_KEY=... ingest_us_equity.py --limit 5` | **NOT RUN — no credential available.** `TIINGO_API_KEY` is unset in this environment and the run's hard constraints forbid obtaining or fabricating one. Recorded in `.planning/WINDOWS.md` as an unrun verify. |

Additional guard check (not in the numbered list): `assert_dense_panel_fits('us_all', '2006-01-01', '2026-09-06')` raises naming 15,424 symbols, 7.19 GiB and the 4.00 GiB budget; a narrowed 2026 window returns cleanly.

## Known Stubs

None. Every code path shipped in this plan is wired to a real data source; no placeholder values, no hardcoded empties, no `TODO`/`FIXME` markers introduced.

## Threat Flags

None. Every trust boundary in the plan's `<threat_model>` is either mitigated as specified (T-0iy-01 credential scrubbing, T-0iy-02 `MIN_ROSTER_ROWS`, T-0iy-03 sizing guard, T-0iy-04 registry-derived population check, T-0iy-06 category/date validation, T-0iy-07 watermark + failure manifest) or accepted as documented (T-0iy-05 unauthenticated TLS fetch, same posture as the already-shipped roster; T-0iy-SC no package installs). No new network endpoint, auth path, file-access pattern or trust-boundary schema change was introduced beyond those the plan already modelled.

## User Setup Required

**External service configuration is needed before the real backfill can run.** No `USER-SETUP.md` was generated (this is a quick task), so the two variables are recorded here:

| Variable | Source | Status |
|---|---|---|
| `TIINGO_API_KEY` | Tiingo Dashboard → API → Token (paid tier per D-03) | **Not set.** Required for `ingest_us_equity.py` without `--dry-run`. Never hardcode it; never commit it. |
| `QUANTLAB_DATA_DIR` | Set to `/Volumes/SSD/data` per D-04 | Works when exported; `config/__init__.py:_data_root()` already reads it. No competing knob was added. |

Also outstanding from earlier state: the previously leaked Tiingo key in git history should be **revoked/rotated** in the Tiingo dashboard before a new key is exported.

## Next Phase Readiness

**Ready.** The universe table at `/Volumes/SSD/data/data/reference/universe.parquet` now carries all four categories and the `us_all` roster resolves 15,424 in-range symbols. The remaining step is operational, not code: export a valid `TIINGO_API_KEY` and run

```
QUANTLAB_DATA_DIR=/Volumes/SSD/data uv run python ingest_us_equity.py --limit 5   # smoke test
QUANTLAB_DATA_DIR=/Volumes/SSD/data uv run python ingest_us_equity.py             # full backfill
```

**Carry-forward for whoever consumes this data:** the full-market Zarr densification does **not** fit this machine (7.19 GiB dense vs 4 GiB budget vs 16 GiB RAM). The guard is a deliberate deliverable, not an obstacle — `--to-zarr` needs a narrowed `--start-date` or `--limit`. A chunked/lazy densification path in `StockDataset._raw_data_to_xr()` is the real fix and is out of scope here.

## Self-Check: PASSED

- `ingest_us_equity.py` exists on disk — **FOUND**
- Commits `fe7d577`, `03dfdee`, `69bf31c`, `5b3ec2d`, `26d0329`, `4ae175e` — all **FOUND** in `git log`
- `uv run pytest tests/ -q` → **144 passed**, ≥ 123 baseline, no pre-existing assertion weakened
- No hardcoded `/Volumes/SSD` in any source file — **CONFIRMED** by grep
- `NasdaqUniverseFetcher.EXCHANGE_FILTER == ("NASDAQ",)` — **CONFIRMED**

---
*Phase: quick-260906-0iy*
*Completed: 2026-09-06*
