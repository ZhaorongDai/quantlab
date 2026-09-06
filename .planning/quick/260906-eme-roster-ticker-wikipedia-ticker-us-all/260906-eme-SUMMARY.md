---
phase: quick-260906-eme
plan: 01
subsystem: data
tags: [universe, tiingo, wikipedia, polars, regex, survivorship-bias, roster]

requires:
  - phase: 03.1
    provides: IndexMembershipFetcher / TiingoRosterFetcher and the universe.parquet reference table
  - phase: quick-260906-0iy
    provides: USEquityUniverseFetcher (us_all) as a sibling of NasdaqUniverseFetcher
provides:
  - "IndexMembershipFetcher._normalize_ticker_cell() + _WELL_FORMED_TICKER: Wikipedia change-log ticker cells are normalized, the correction is announced, and anything still malformed raises"
  - "TiingoRosterFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES (default False) with _PREFERRED_SHARE_PATTERN / _BABY_BOND_PATTERN, opted into by USEquityUniverseFetcher only"
  - "us_all drops ~940 distinct preferred / baby-bond tickers; nasdaq_all is byte-for-byte unchanged"
affects: [constituent-panel, chunked-ingest, tiingo-acquisition, universe-mask]

actuals:
  tokens: 8873
  tasks: 2
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Opt-in roster filtering via a base-class boolean constant defaulting to False, so a frozen category's semantics cannot be changed by a shared filter"
    - "Normalize -> log the correction -> validate -> raise, instead of a bare raise, for a publicly-editable upstream source"

key-files:
  created: []
  modified:
    - acquisition/universe.py
    - tests/conftest.py
    - tests/test_universe.py

key-decisions:
  - "_WELL_FORMED_TICKER admits digits ([A-Z0-9]) although zero live cells carry one — the plan's literal [A-Z] would have raised on the repo's own ADDED1/TEMP1/GONE1 synthetic fixture symbols (28 references across 4 test files, two of them outside this plan's file scope)"
  - "Normalization lives on IndexMembershipFetcher, not on either subclass, so no index can ship a parse that skips it (safety property 1)"
  - "Only LEADING/TRAILING delimiters are stripped; an interior one raises, because it means two cells were merged by a parser regression rather than an editor typo"
  - "EXCLUDE_NON_COMMON_SECURITY_TYPES is a class constant, not an overridable method — a roster is DATA in this module, and the criterion then lives in exactly one place"
  - "us_all is no longer a strict superset of nasdaq_all; the asymmetry is documented in USEquityUniverseFetcher's docstring and pointed at Locked Decision A4 / D-02"

patterns-established:
  - "Aggregated one-line-per-column correction warnings: an ongoing 3-cell upstream typo stays readable, a systematic regression shows up as one huge list rather than flooding a cron log"
  - "A roster-shrinking filter runs BEFORE its row-count floor, so the floor validates the count that is actually persisted"

requirements-completed: [DATA-05]

coverage:
  - id: D1
    description: "Wikipedia change-log ticker cells carrying trailing delimiter residue (ALLE |, JCP |, ITT |) normalize to clean tickers and the correction is announced by a logger.warning naming the index, the column and every raw->normalized pair"
    requirement: DATA-05
    verification:
      - kind: unit
        ref: "tests/test_universe.py#test_normalize_ticker_cell_strips_the_real_observed_delimiter_residue"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_residue_bearing_ticker_cells_parse_clean_and_announce_themselves"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_a_clean_change_log_emits_no_normalization_warning"
        status: pass
    human_judgment: false
  - id: D2
    description: "A residue-only cell becomes the existing None no-change sentinel; a cell still malformed after normalization raises naming every offending cell"
    requirement: DATA-05
    verification:
      - kind: unit
        ref: "tests/test_universe.py#test_a_residue_only_ticker_cell_becomes_the_no_change_sentinel"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_a_cell_still_malformed_after_normalization_names_every_offending_cell"
        status: pass
    human_judgment: false
  - id: D3
    description: "USEquityUniverseFetcher.fetch() drops preferred shares and baby bonds while keeping every class share and every warrant/unit/right"
    requirement: DATA-05
    verification:
      - kind: unit
        ref: "tests/test_universe.py#test_the_exclusion_criterion_matches_the_measured_directory"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_us_equity_fetcher_drops_preferred_shares_and_baby_bonds"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_us_equity_fetcher_keeps_class_shares_and_warrants"
        status: pass
    human_judgment: false
  - id: D4
    description: "nasdaq_all is unaffected: one payload yields kept-in-nasdaq_all / dropped-from-us_all, and MIN_ROSTER_ROWS is evaluated on the post-exclusion count"
    requirement: DATA-05
    verification:
      - kind: unit
        ref: "tests/test_universe.py#test_one_payload_yields_kept_in_nasdaq_all_and_dropped_from_us_all"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_min_roster_rows_is_evaluated_on_the_post_exclusion_count"
        status: pass
      - kind: unit
        ref: "tests/test_universe.py#test_nasdaq_roster_exchange_filter_and_symbol_set_are_unchanged"
        status: pass
    human_judgment: false
  - id: D5
    description: "The fixes reach the user's persisted universe.parquet and the derived Zarr panels only after a manual refresh + full store rebuild"
    verification: []
    human_judgment: true
    rationale: "Requires the user to run refresh_us_equity_universe.py against the real Tiingo/Wikipedia sources and to rebuild the Zarr stores — nothing under /Volumes/SSD/data was touched by this task, by constraint."

duration: 9min
completed: 2026-09-06
status: complete
---

# Quick Task 260906-eme: Roster Ticker Hygiene Summary

**Two independent roster-hygiene fixes in `acquisition/universe.py`: Wikipedia change-log ticker cells are now normalized-then-validated (recovering `JCP` and `ITT` as real S&P 500 members and de-duplicating `ALLE`), and `us_all` drops ~940 preferred / baby-bond tickers while `nasdaq_all` stays byte-for-byte frozen.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-09-06T10:41:49Z
- **Completed:** 2026-09-06T10:51:00Z
- **Tasks:** 2 of 2
- **Files modified:** 3

## Accomplishments

- **Closed a survivorship-bias hole.** Three live S&P 500 change-log ticker cells carry a trailing wikitable delimiter (`ALLE |`, `ITT |`, `JCP |`). They entered the membership table verbatim, so `JCP` and `ITT` — two multi-decade index members — were representable in the panel ONLY as phantom symbols matching no market data, while `ALLE` was double-counted with a fabricated 2013-12-02 → 2026-08-18 interval. `_normalize_ticker_cell()` now strips the residue, `logger.warning` announces every correction, and anything still malformed raises.
- **Removed ~940 non-common-stock lines from `us_all`.** `_PREFERRED_SHARE_PATTERN` / `_BABY_BOND_PATTERN` drop 965 rows / 940 distinct tickers (932 preferred + 8 baby bonds) from the measured 16,138-row roster, leaving 15,173 — so downstream ingestion stops spending Tiingo requests and panel columns on preferred series and notes.
- **Kept every class share.** `BRK-A`, `BRK-B`, `BF-A`, `BF-B`, `PBR-A`, `HEI-A`, `MOG-A`, `LEN-B`, `UA-C`, `MKC-V`, `AGM-A`, `CRD-A`, `LGF-A`, `GEF-B`, `STZ-B`, `UHAL-B`, `CWEN-A` are common stock carrying a hyphen — the precise trap here. Requiring a `P` immediately after the delimiter separates them from preferreds; the survival is asserted in a test against the real measured literals.
- **Froze `nasdaq_all`.** `EXCLUDE_NON_COMMON_SECURITY_TYPES` defaults to `False` on `TiingoRosterFetcher`; only `USEquityUniverseFetcher` opts in. A dedicated test proves ONE payload yields kept-in-`nasdaq_all` / dropped-from-`us_all`.
- **Ordered both guards deliberately.** Normalization precedes the `_BLANK_TICKER_CELLS` sentinel test; the exclusion precedes the `MIN_ROSTER_ROWS` floor. Both orderings are commented as key links and pinned by tests.

## Task Commits

1. **Task 1 (RED): failing normalization tests** — `f799965` (test)
2. **Task 1 (GREEN): normalize + validate change-log tickers** — `8f85f0d` (feat)
3. **Task 2 (RED): failing exclusion tests + append-only fixture rows** — `e3d8972` (test)
4. **Task 2 (GREEN): us_all preferred / baby-bond exclusion** — `5e8c999` (feat)

No REFACTOR commits — neither implementation had cleanup to do after going green.

## Files Created/Modified

- `acquisition/universe.py` — `_WELL_FORMED_TICKER`, `IndexMembershipFetcher._normalize_ticker_cell()`, the rewritten per-column loop and post-loop validation raise in `_parse_changes_table()`; `_PREFERRED_SHARE_PATTERN`, `_BABY_BOND_PATTERN`, `TiingoRosterFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES` and the exclusion branch in `fetch()`; docstring/comment updates on both roster subclasses.
- `tests/conftest.py` — 18 rows appended to the `mock_universe_fetchers` roster CSV (every one NYSE or AMEX, honouring the fixture's APPEND-ONLY rule).
- `tests/test_universe.py` — 10 new tests in two new sections, plus the `IndexMembershipFetcher` / `re` imports.

## Decisions Made

- **`_WELL_FORMED_TICKER` admits digits.** See Deviations below — this is the one departure from the plan's literal text.
- **Normalization lives on the base class**, not on either subclass, matching safety property 1: `_parse_changes_table` is concrete and shared precisely so no index can ship a parse that skips the base's validation. The Nasdaq-100 log has zero malformed cells today but is the same publicly-editable MediaWiki surface.
- **Only leading/trailing delimiters are stripped.** An interior pipe raises: it is not the observed upstream typo shape and much more likely means two cells were merged by a parser regression, which must be loud.
- **`EXCLUDE_NON_COMMON_SECURITY_TYPES` is a class constant, not an overridable method**, matching this module's "a roster is DATA — three class constants, not a code change" idiom, and keeping the criterion in exactly one place.
- **The `us_all` / `nasdaq_all` asymmetry is documented as intentional.** `us_all` is no longer a strict superset: a NASDAQ-listed preferred such as `ONB-P-A` is in `nasdaq_all` and not in `us_all`. `USEquityUniverseFetcher`'s docstring routes anyone wanting to "fix" that at Locked Decision A4 / D-02 rather than at the code.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] `_WELL_FORMED_TICKER` widened from `[A-Z]` to `[A-Z0-9]`**

- **Found during:** Task 1 (GREEN phase, before implementing — predicted from the fixture vocabulary and confirmed by inspection).
- **Issue:** The plan specifies `^[A-Z]{1,7}(?:[.-][A-Z]{1,2})?$`, pinned to the measured live vocabulary (`A-Z` only, no digits, in all 1,190 live cells). But this repo's synthetic change-log fixtures name their symbols `ADDED1`, `TEMP1`, `GONE1` — 28 references across `tests/conftest.py`, `tests/test_universe.py`, `tests/test_constituent_panel.py` and `tests/test_ndx_constituent.py`. The literal regex would raise `ValueError` on every one of them, breaking ~10 pre-existing tests.
- **Fix:** Admitted digits in the character class: `^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2})?$`. The alternative — renaming the fixture symbols — would have touched two test files outside this plan's declared `files_modified` scope and destroyed the deliberate naming convention that makes a synthetic symbol unmistakable for a real ticker.
- **Why this is the right call, not just the convenient one:** a false positive in this validator is expensive in exactly the way the plan's own design argument warns about — it raises, `fetch_changes()` falls back to the stale cache, and `build()` then refuses until a human edits Wikipedia. Admitting a character class that has not yet appeared upstream but plausibly could costs nothing today. Every malformation this guard actually exists to catch — an interior delimiter, an embedded space, lowercase, an over-long cell — is still rejected, and the tests prove it.
- **Files modified:** `acquisition/universe.py`
- **Verification:** The measurement and both reasons are recorded in the constant's comment. `test_a_cell_still_malformed_after_normalization_names_every_offending_cell` pins that `AL|LE` and `logi` still raise.
- **Committed in:** `8f85f0d`

**2. [Rule 2 — Missing critical documentation] Corrected the now-false "strict SUPERSET" claim**

- **Found during:** Task 2.
- **Issue:** `USEquityUniverseFetcher`'s docstring asserted that `us_all` "is a strict SUPERSET" of `nasdaq_all`. The exclusion makes that statement false, and leaving it would have set a future reader up to reason from a stale invariant.
- **Fix:** Removed the superset claim from the opening paragraph and replaced it with an explicit asymmetry section naming the mechanism and the decision that locks it.
- **Files modified:** `acquisition/universe.py`
- **Verification:** `test_one_payload_yields_kept_in_nasdaq_all_and_dropped_from_us_all` asserts the asymmetry directly.
- **Committed in:** `5e8c999`

---

**Total deviations:** 2 auto-fixed (1x Rule 3, 1x Rule 2)
**Impact on plan:** Both are narrow and inside the plan's own reasoning. No scope creep: `git diff --stat` touches exactly the three declared files. Every locked decision held — `nasdaq_all` unchanged, class shares surviving, normalize-then-log-then-validate shape intact, both orderings as specified.

## Issues Encountered

None. Both tasks went RED → GREEN on the first implementation pass.

## Verification

- `uv run pytest tests/ -q` → **226 passed** (216 baseline + 10 new), zero failures, no pre-existing assertion weakened.
- Both per-task inline `uv run python -c` checks pass (`normalization OK`, `criterion OK`).
- `git diff --stat` touches only `acquisition/universe.py`, `tests/conftest.py`, `tests/test_universe.py`.
- Nothing under `/Volumes/SSD/data` was read-modified, written, moved or deleted. Every test is offline; `TIINGO_API_KEY` is unset and was not needed.
- The open human-verify checkpoint in quick task 260906-26o (Task 3, legacy Tiingo watermark stamping) was not touched.

## Known Stubs

None.

## Threat Flags

None — no new network endpoint, auth path, file-access pattern or trust-boundary schema change was introduced. Both changes tighten existing boundaries (T-eme-01 through T-eme-05 all mitigated as planned).

## User Setup Required — ACTION NEEDED

**These fixes do not reach your data until you rebuild the reference table.** The code is correct as of this commit, but `universe.parquet` on disk still holds the old roster and the old phantom tickers.

1. **Re-run the universe refresh:**

   ```
   cd /Users/daizhaorong/projects/quantlab && uv run python refresh_us_equity_universe.py
   ```

   Expect `us_all` to shrink by roughly 940 symbols (15,425 → ~14,485 distinct), and `JCP` / `ITT` to appear in `sp500_constituent` as real, market-data-matchable symbols with the duplicate `ALLE` gone.

2. **Rebuild the Zarr stores — a full rebuild, not an incremental append.** A changed `us_all` roster makes `ChunkLedger.assert_consistent` refuse an incremental append to the existing store, so the store must be rebuilt from scratch. This adds no new burden: that rebuild was ALREADY required for an unrelated reason (the existing store covers only 9,945 of 15,424 roster symbols). The S&P membership panel store must also be rebuilt for the phantom-ticker fix to reach it.

## Next Phase Readiness

`acquisition/universe.py` is ready. The blocker for anything downstream is the manual refresh + rebuild above — until it runs, every consumer of `universe.parquet` and of the constituent Zarr panels is still reading the pre-fix roster.

---
*Phase: quick-260906-eme*
*Completed: 2026-09-06*
