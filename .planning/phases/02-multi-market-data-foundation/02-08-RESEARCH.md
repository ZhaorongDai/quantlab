# Phase 2 (Supplement): Survivorship-Bias-Free, Point-in-Time US Equity Universe — Research

**Researched:** 2026-09-05
**Domain:** US equity reference-data acquisition (delisted-inclusive symbol universe + point-in-time index membership)
**Confidence:** MEDIUM-HIGH (data-source mechanics verified live; long-run Wikipedia maintenance reliability and Tiingo rate limits are lower confidence — see Assumptions Log)

## Summary

This supplements Phase 2's already-completed D-12 decision. Two of D-12's original source choices need to be **revised**, not just implemented, based on this research:

1. **Tiingo has no native index-constituent/membership product** — confirmed absent from every documented Tiingo endpoint category (End-of-Day, IEX, Fundamentals, Fund Fees, Dividends, Splits, Search) `[CITED: tiingo.com/documentation]`. The user's own suggested alternative, **Alpaca**, was also checked and is **not a viable substitute**: Alpaca has no index-constituent endpoint either, and its Corporate Actions API explicitly does **not** cover delistings/reorganizations `[CITED: docs.alpaca.markets/us/reference/corporateactions-1]` — so Alpaca is strictly worse than Tiingo for this pipeline's two hard requirements (delisted-stock coverage AND index membership). **Decision: no change from D-12's Wikipedia-based approach for constituent membership; Tiingo/Alpaca are both ruled out for that specific sub-problem.**
2. **D-12's Nasdaq-universe source needs to change.** `nasdaqlisted.txt` (D-12's original choice) lists **only currently-listed** securities — it cannot satisfy "including historically delisted stocks" no matter how it's post-processed, because delisted symbols are removed from that file entirely, not flagged. The correct source for a delisted-inclusive Nasdaq roster is **Tiingo's own `supported_tickers.csv`** (`exchange == "NASDAQ"`, `assetType == "Stock"`, `priceCurrency == "USD"` → 9,318 rows verified live, vs. `nasdaqlisted.txt`'s 5,593 current-only rows), each row carrying a real `startDate`/`endDate` window `[VERIFIED: apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip, fetched 2026-09-05]`. `nasdaqlisted.txt` can still be kept as a secondary source for security-name/ETF-flag/test-issue metadata on **currently**-listed names, but must not be the sole/primary source for universe membership going forward.
3. **The Wikipedia "Historical components of the S&P 500" table is real, actively maintained, and cleanly parseable via `pandas.read_html`** — verified live (407 rows, 2026-08-18 most recent entry, updated within days of the actual S&P announcement). **But it is left-censored at 1976-07-01** — this is a new, load-bearing finding not in the background brief: the page's own prose claims coverage "between January 1, 1963, and December 31, 2014" but the actual table's earliest row is **July 1, 1976**, not 1963. Point-in-time queries for dates before 1976-07-01 cannot be correctly answered by this pipeline and must be explicitly rejected/flagged, not silently answered with an incomplete history.

**Primary recommendation:** Build a new, standalone `acquisition/universe.py` module (not subclassing `Acquisition(ABC)` — confirmed correct per D-12's own reasoning, this is structurally a roster/interval table, not an OHLCV time series) with three single-responsibility classes — a Tiingo-backed Nasdaq roster fetcher, a Wikipedia+datahub-backed S&P 500 point-in-time interval builder, and a `UniverseCatalog` orchestrator that merges both into one `(symbol, category, start_date, end_date)` reference table persisted as parquet via the existing `PlBackend`. Wire `ingest_tiingo.py` with a new `--universe {sp500,nasdaq_all,all} --as-of-date` option that resolves to a plain symbol list before calling the existing `TiingoAcquisition`/`StockDataset` — no changes needed to either of those classes.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Fetch/parse Nasdaq-ever-listed roster (Tiingo `supported_tickers.csv`) | Acquisition (`acquisition/universe.py`) | — | Remote fetch + filter, no time-series/OHLCV shape — same tier as `TiingoAcquisition` but a different class (per D-12) |
| Fetch/parse S&P 500 current-anchor + historical change log (datahub CSV + Wikipedia) | Acquisition (`acquisition/universe.py`) | — | Same tier; two independent remote sources merged by one algorithm |
| Point-in-time interval reconstruction (chronological event simulation) | Acquisition (`acquisition/universe.py`) | — | Pure computation over already-fetched reference data; no network/no xarray involved |
| Persisted universe/interval table | Storage (reference/metadata, parquet via `PlBackend`) | — | Explicitly **not** part of the `Dataset`→`Factor`→`Model` xarray/Zarr pipeline contract (see Architecture Patterns, Anti-Pattern note) — same category as `config/instruments.yaml`, just larger, hence parquet not YAML |
| Symbol-list resolution (`get_symbols_as_of`) | Acquisition (`acquisition/universe.py`) | Config/CLI (`ingest_tiingo.py`) | Resolves to a plain `list[str]`/`tuple[str,...]` consumed by existing `AcquisitionConfig.symbols`/`DatasetConfig.symbols` — no new data shape crosses into the Dataset/Factor tiers |
| OHLCV acquisition for resolved symbols | Acquisition (`TiingoAcquisition`, existing, unmodified) | Dataset (`StockDataset`, existing, unmodified) | Already built in Phase 2 Wave 3/4 — this supplement only changes what symbol list feeds in, never how OHLCV is fetched/converted |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `requests` | already a pyproject dependency (`>=2.34.2`) | Fetch `nasdaqlisted.txt`, `supported_tickers.zip`, datahub CSV, Wikipedia HTML/API | Already used elsewhere in the codebase (`utils/binance.py`); no new dependency needed |
| `polars` | already a pyproject dependency (`>=1.44.1`) | Build/filter the interval table, write parquet via `PlBackend` | Matches existing `PlBackend`/`dataset/cleaning.py` tabular convention — do not introduce pandas as the storage-layer dtype here |
| `pandas` | already a pyproject dependency (`>=3.0.5`) | `pandas.read_html()` to parse the Wikipedia wikitable | Already a dependency (used elsewhere for date filtering, e.g. `utils/file.py`); only used transiently inside the Wikipedia parser, immediately converted to polars for storage — does not violate the "no DataFrame between pipeline modules" rule since this never crosses a `Dataset`/`Factor` boundary |
| `lxml` | 6.1.3 (verified via `pip index versions lxml`, 2026-09-05) `[VERIFIED: PyPI registry, slopcheck OK]` | HTML parser backend for `pandas.read_html()` | `pandas.read_html()` requires `lxml` or `bs4`+`html5lib` installed; without one of these it raises `ImportError` at call time (verified in this research session: initial run without `lxml` in the `uv run --with` set would fail) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `loguru` | already a dependency | Log parse-schema-drift warnings, left-censoring warnings, and anchor/change-log consistency mismatches | Matches existing logging convention throughout `base/`, `dataset/`, `acquisition/` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pandas.read_html` on rendered HTML | Raw wikitext regex parsing (`action=parse&prop=wikitext`) | Background brief already flagged wikitext cell-formatting as inconsistent (`\|value` vs `\|\| value`); **verified in this session** that `pandas.read_html` cleanly handles the real page (407×7 table, correct `NaN` for one-sided add/remove rows) with zero custom regex — strictly less code, less fragile. Use HTML parsing, not wikitext regex. |
| `nasdaqlisted.txt` as Nasdaq-universe source | Tiingo `supported_tickers.csv` filtered to `exchange=="NASDAQ"` | `nasdaqlisted.txt` is current-listings-only (no delisted history at all — not a matter of degree, the data literally does not exist in that file). Tiingo's file has explicit `startDate`/`endDate` per ticker across 9,318 NASDAQ-tagged Stock/USD rows (current + historical). **This is the load-bearing change vs. D-12's original design.** |
| Alpaca (user-suggested) for index membership or delisted coverage | Tiingo `supported_tickers.csv` + Wikipedia | Alpaca has no index-constituent endpoint and its Corporate Actions API explicitly excludes delistings/reorganizations `[CITED: docs.alpaca.markets/us/reference/corporateactions-1]`; Alpaca's own operating/data history is also shorter (IEX-feed-based, platform launched ~2018) than Tiingo's, so its "inactive" asset coverage for stocks delisted well before Alpaca existed is likely weaker. Not recommended for either sub-problem in this phase. |

**Installation:**
```bash
uv add lxml
```

**Version verification:** `lxml` 6.1.3 confirmed current via `pip index versions lxml` on 2026-09-05; `requests`/`polars`/`pandas` already pinned in `pyproject.toml`, no version bump needed.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `lxml` | PyPI | ~20 years (long-standing) | very high (foundational HTML/XML parsing library) | github.com/lxml/lxml | `OK` (verified via `slopcheck scan --pkg pypi lxml --json`, 2026-09-05) | Approved |
| `requests` | PyPI | already installed | — | github.com/psf/requests | `OK` (verified via `slopcheck scan --pkg pypi requests --json`) | Approved (no new install needed) |

**Packages removed due to slopcheck `[SLOP]` verdict:** none
**Packages flagged as suspicious `[SUS]`:** none

slopcheck was successfully installed and run in this research session (`uv tool install slopcheck`, v0.6.1) — both candidate/verification packages returned `OK`, no `[ASSUMED]` fallback needed for these two.

## Architecture Patterns

### System Architecture Diagram

```text
                         ┌─────────────────────────────────────────────┐
                         │        acquisition/universe.py (NEW)         │
                         │                                               │
  Tiingo                │  ┌─────────────────────────┐                  │
  supported_tickers.zip ─┼─▶│ NasdaqUniverseFetcher    │──┐               │
  (no auth needed)      │  │ .fetch() -> pl.DataFrame │  │               │
                         │  │ (symbol,start,end)       │  │               │
                         │  └─────────────────────────┘  │               │
                         │                                 ▼               │
  datahub                │  ┌─────────────────────────┐  ┌─────────────┐ │
  constituents.csv       ─┼─▶│ SP500MembershipFetcher   │─▶│ UniverseCatalog│
  (current anchor)       │  │  .fetch_anchor()          │  │ .build()      │
                         │  │  .fetch_changes()  (wiki) │  │ .save()       │
  Wikipedia               │  │  .reconstruct_intervals()│  │ .get_symbols_ │
  Historical_components_  ─┼─▶│  -> pl.DataFrame          │  │   as_of()     │
  of_the_S%26P_500        │  │  (symbol,start,end)       │  └──────┬──────┘ │
  (HTML table, id=changes)│  └─────────────────────────┘         │        │
                         └───────────────────────────────────────┼────────┘
                                                                   ▼
                                                    data/reference/universe.parquet
                                                    (symbol, category, start_date, end_date)
                                                    category ∈ {sp500_constituent, general_market}
                                                                   │
                                                                   ▼
                                              ingest_tiingo.py --universe sp500 --as-of-date 2015-06-01
                                                          │  resolves to plain list[str]
                                                          ▼
                                    stock_acquisition_config(symbols=[...]) / stock_kline_config(symbols=[...])
                                                          │  (existing, UNCHANGED)
                                                          ▼
                                    TiingoAcquisition.download()/.refresh()  →  StockDataset.from_raw_data().save()
                                                          │
                                                          ▼
                                          data/{market}/{frequency}/stock.zarr  (existing Zarr contract, unchanged)
```

### Recommended Project Structure
```
acquisition/
├── __init__.py          # empty, per existing convention
├── tiingo.py             # existing, unmodified
└── universe.py           # NEW — NasdaqUniverseFetcher, SP500MembershipFetcher, UniverseCatalog
base/
└── config.py              # MODIFY — add UniverseConfig dataclass
config/
└── __init__.py            # MODIFY — add universe_config() factory
data/
└── reference/
    └── universe.parquet   # NEW — persisted output, plus small cache snapshots (see Pitfall: Wikipedia degradation)
```

### Pattern 1: Standalone reference-data class, not an `Acquisition(ABC)` subclass
**What:** `NasdaqUniverseFetcher`/`SP500MembershipFetcher`/`UniverseCatalog` are plain classes, config-driven, single-responsibility methods (`fetch()`, `reconstruct_intervals()`, `save()`, `get_symbols_as_of()`), mirroring the codebase's `Dataset`/`FactorKunQuant` style (config property, `_`-prefixed private helpers) but **without** inheriting `base/acquisition.py:Acquisition`.
**When to use:** Any time a "which symbols exist and when" roster/interval question is a fundamentally different shape than a per-symbol OHLCV time series with a single watermark — `Acquisition.download()/.refresh()` loop over `self.config.symbols` and call `_fetch_and_write(symbol, start, end)` once per symbol; a universe fetch is a **single bulk fetch** producing a **table of many symbols' date ranges**, not a per-symbol watermark refresh. Forcing this into `Acquisition(ABC)`'s contract (per D-12's own reasoning, confirmed correct in this research) would require overriding almost every method meaninglessly.
**Example:**
```python
# acquisition/universe.py — class skeleton (new)
from pathlib import Path
from typing import Literal, Self

import polars as pl
import requests
from loguru import logger

from base.config import UniverseConfig


class NasdaqUniverseFetcher:
    """Fetches Tiingo's full historical ticker directory and filters to the
    NASDAQ-listed Stock universe (current + delisted). Source: Tiingo's own
    supported_tickers.csv — NOT nasdaqlisted.txt, which only lists currently-
    listed securities and cannot represent delisted history at all."""

    SOURCE_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"

    def fetch(self) -> pl.DataFrame:
        # download+unzip supported_tickers.zip, read supported_tickers.csv
        # filter: exchange == "NASDAQ", assetType == "Stock", priceCurrency == "USD"
        # rename: ticker -> symbol, startDate -> start_date, endDate -> end_date
        ...


class SP500MembershipFetcher:
    """Reconstructs point-in-time S&P 500 membership intervals from a current
    anchor snapshot plus a dated historical change log. See Common Pitfall
    'Wikipedia table left-censored at 1976' before using output before that
    date."""

    ANCHOR_URL = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
        "main/data/constituents.csv"
    )
    CHANGES_URL = (
        "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
    )
    PIT_COVERAGE_START = "1976-07-01"  # verified earliest row in changes table

    def fetch_anchor(self) -> pl.DataFrame: ...
    def fetch_changes(self) -> pl.DataFrame: ...
    def reconstruct_intervals(
        self, anchor: pl.DataFrame, changes: pl.DataFrame
    ) -> pl.DataFrame: ...
```

### Pattern 2: Wikipedia table parsing — HTML via `pandas.read_html`, not wikitext regex
**What:** Fetch the rendered page HTML (`requests.get(url, headers={"User-Agent": "..."})`), then `pandas.read_html(io.StringIO(html), attrs={"id": "changes"})[0]`.
**When to use:** Always, for this specific table — verified in this research session to return a clean `(407, 7)` frame with a 2-level column MultiIndex (`("Effective Date","Effective Date")`, `("Added","Ticker")`, `("Added","Security")`, `("Removed","Ticker")`, `("Removed","Security")`, `("Reason","Reason")`, `("Refs","Refs")`) and correct `NaN` in `Added.*`/`Removed.*` columns for one-sided rows (spin-offs / market-cap-driven single removals with no simultaneous addition).
**Example:**
```python
# Source: verified live in this research session, 2026-09-05
import io
import requests
import pandas as pd

resp = requests.get(
    "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500",
    headers={"User-Agent": "quantlab-research (contact: dzr233@gmail.com)"},
    timeout=30,
)
resp.raise_for_status()
tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "changes"})
changes = tables[0]
changes.columns = [
    "effective_date", "added_ticker", "added_security",
    "removed_ticker", "removed_security", "reason", "refs",
]
changes["effective_date"] = pd.to_datetime(changes["effective_date"])
# verified: 407 rows, min date 1976-07-01, max date 2026-08-18,
# 19 removal-only rows, 23 addition-only rows, 365 paired rows
```
**Requires `lxml` installed** (`pandas.read_html` raises `ImportError: lxml not found, please install it` without it or an equivalent bs4/html5lib backend — confirmed by needing to add it in this research session's test harness).

### Pattern 3: Point-in-time interval reconstruction — forward chronological event simulation
**What:** Rather than "roll the anchor backward" (error-prone to reason about correctly), walk the change log **forward** in ascending date order, simulating open/close events per ticker, then reconcile the result against the anchor set.
**Algorithm:**
```python
# acquisition/universe.py — SP500MembershipFetcher.reconstruct_intervals()
def reconstruct_intervals(self, anchor: pl.DataFrame, changes: pl.DataFrame) -> pl.DataFrame:
    anchor_symbols = set(anchor["symbol"])
    anchor_date_added = dict(zip(anchor["symbol"], anchor["date_added"]))  # from datahub CSV's own "Date added" column

    changes_sorted = changes.sort("effective_date")  # ASCENDING — oldest first
    open_intervals: dict[str, str] = {}   # symbol -> currently-open start_date
    closed: list[tuple[str, str, str]] = []  # (symbol, start_date, end_date)

    for row in changes_sorted.iter_rows(named=True):
        eff = row["effective_date"]
        if row["removed_ticker"] is not None:
            sym = row["removed_ticker"]
            if sym in open_intervals:
                closed.append((sym, open_intervals.pop(sym), eff))
            else:
                # Removal predates the earliest row in our change log
                # (left-censored: this symbol was already a member before
                # PIT_COVERAGE_START). Record with an explicit sentinel,
                # never silently drop.
                logger.warning(f"{sym}: removal at {eff} has no matching "
                                f"prior 'added' event — left-censored interval, "
                                f"using PIT_COVERAGE_START as start_date")
                closed.append((sym, self.PIT_COVERAGE_START, eff))
        if row["added_ticker"] is not None:
            sym = row["added_ticker"]
            if sym in open_intervals:
                logger.warning(f"{sym}: duplicate 'added' event at {eff} "
                                f"while an interval was already open since "
                                f"{open_intervals[sym]} — data quality issue, "
                                f"keeping the earlier open date")
            else:
                open_intervals[sym] = eff

    # Reconcile remaining open intervals against the current anchor.
    for sym, start in open_intervals.items():
        end_date = None if sym in anchor_symbols else eff  # should always be in anchor; log if not
        if sym not in anchor_symbols:
            logger.warning(f"{sym}: open interval since {start} but not in "
                            f"current anchor set — anchor CSV may be stale "
                            f"relative to the Wikipedia change log")
        closed.append((sym, start, end_date))

    # Anchor members with NO 'added' event anywhere in the change log are
    # either original constituents or were added before PIT_COVERAGE_START.
    seen_symbols = {c[0] for c in closed}
    for sym in anchor_symbols - seen_symbols:
        start = anchor_date_added.get(sym, self.PIT_COVERAGE_START)
        closed.append((sym, start, None))  # open-ended, currently a member

    return pl.DataFrame(closed, schema=["symbol", "start_date", "end_date"], orient="row")
```
**Handles, as required by the research questions:**
- **Non-contiguous re-membership** (added → removed → re-added): naturally produces multiple rows for the same `symbol`, since each add/remove pair independently appends to `closed` and a later `added_ticker` event reopens a fresh `open_intervals[sym]` entry.
- **Currently-active, never-removed members:** `end_date = None` (open-ended) — either resolved directly from the anchor's own `Date added` column (no need to touch the change log at all for the common case), or from the change-log walk when the anchor lacks a reliable "Date added" (defensive fallback).
- **Reason field (bankruptcy vs. rebalance):** confirmed via live data sample it is irrelevant to the pipeline's logic — only `effective_date` + ticker matter. `reason`/`refs` can optionally be retained as audit-only columns, never used in `get_symbols_as_of()`.
- **Ticker renames:** Wikipedia's own editorial note (verified in the fetched wikitext) explicitly excludes renames from this table ("Company name changes and ticker changes are not changes to the index and should not be in this table") — no special handling needed; a renamed ticker is treated as two unrelated symbols, which is correct for this pipeline since downstream OHLCV acquisition is keyed by ticker symbol, not company identity.

### Anti-Patterns to Avoid
- **Storing the universe/interval table inside the `Dataset`/`Factor`/`Model` xarray pipeline:** this table has no `[timestamp, symbol]` OHLCV shape and is never consumed by KunQuant/factor code directly — it is a pre-acquisition symbol-list resolver, structurally identical in role to `config/instruments.yaml` (reference/config metadata), just bigger (thousands of rows) hence parquet instead of YAML. Do not wrap it in `XrBackend`/Zarr; use `PlBackend` (already exists, already supports parquet read/write). **This interpretation of the xarray/Zarr pipeline-format boundary should be confirmed with the user/planner before locking in** — see Assumptions Log A1.
- **Using `nasdaqlisted.txt` as the sole/primary Nasdaq-universe source:** it silently omits every delisted symbol; a pipeline built on it alone reintroduces exactly the survivorship bias the user is trying to eliminate.
- **Treating "left-censored, unknown start date" the same as "no data, skip it":** an S&P 500 member removed in, say, 1980 with no matching "added" row anywhere in the parsed table (because the table starts 1976-07-01) genuinely has an unknown-before-1976 start. Silently using `None`/omitting it (rather than flagging + using the coverage-start sentinel) produces a table that looks complete but silently under-covers pre-1976 history — this is itself a subtle survivorship/look-ahead-adjacent bug, worth a loud warning, not a silent default.
- **Backward-rolling from "today" is harder to get right than forward simulation:** the background research questions phrased the algorithm as "roll the anchor backward" — this research recommends the mathematically equivalent but much easier to verify forward-chronological event simulation instead (Pattern 3 above), because forward simulation's invariants (an "add" opens, a "remove" closes) are simpler to unit-test than backward differencing.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parsing an HTML wikitable with merged/irregular cell markup | Custom wikitext regex parser | `pandas.read_html(io.StringIO(html), attrs={"id": "changes"})` | Verified in this session to correctly handle every row shape (paired add/remove, addition-only, removal-only) with zero custom parsing code |
| Detecting whether a ticker is currently tradable/delisted | A hand-maintained "known delisted" list | Tiingo `supported_tickers.csv`'s own `startDate`/`endDate` columns | Tiingo already computes and publishes this, refreshed daily, across its entire 108k-row ticker universe — verified with real examples (`ATVI` end 2023-10-13, `TWTR` end 2022-10-28) |
| Interval-overlap "as of date" queries | Manual date-range loops | `polars` boolean filter: `(pl.col("start_date") <= as_of) & (pl.col("end_date").is_null() | (pl.col("end_date") >= as_of))` | One-line vectorized filter, no hand-rolled interval tree needed at this row count (~10k rows total) |

**Key insight:** every part of this problem that looks like it needs custom logic (delisted detection, wikitable parsing, interval-overlap queries) already has a library or vendor-published field that solves it directly — the only genuinely new logic this phase must write is the ~40-line forward-chronological reconciliation in Pattern 3, and even that reduces to simple dict/list bookkeeping once framed correctly.

## Common Pitfalls

### Pitfall 1: Wikipedia table left-censored at 1976-07-01 despite page text claiming 1963 coverage
**What goes wrong:** A caller requests `get_symbols_as_of("sp500", "1970-01-01")` and silently receives a wrong/incomplete answer instead of an explicit "coverage starts 1976-07-01" error.
**Why it happens:** The article's prose ("Between January 1, 1963, and December 31, 2014, 1,186 index components were replaced") does not match the actual `id="changes"` table content, whose earliest row (verified live) is `July 1, 1976`. This is either an editorial inconsistency in the Wikipedia article itself, or a separate pre-1976 table that was not preserved when this page was split out of the main "S&P 500" article on 2026-08-11.
**How to avoid:** Store `PIT_COVERAGE_START = "1976-07-01"` as an explicit constant on `SP500MembershipFetcher`; `get_symbols_as_of()` must raise or loudly warn (not silently answer) for `as_of_date < PIT_COVERAGE_START`.
**Warning signs:** Any interval row with `start_date == PIT_COVERAGE_START` for a symbol that was clearly a much older company (e.g., an original-1957 constituent) is a left-censoring artifact, not a real 1976 addition date — do not present it to users as the "true" addition date without the caveat.

### Pitfall 2: This specific Wikipedia page is very young (created 2026-08-11) — don't over-read its short edit history as "unreliable"
**What goes wrong:** A naive freshness check ("this page only has ~10 revisions") could wrongly conclude the source is unmaintained/risky.
**Why it happens:** The page was **split out of** the long-standing, heavily-trafficked main "S&P 500" Wikipedia article (verified: that article is edited multiple times per week, was even briefly edit-protected for "disruptive editing" on 2026-08-29) on 2026-08-11 — the content itself (the changes table) is not new, only its placement on a separate page is. The split page's `dateModified` (2026-08-23) already reflects a post-split correction ("renamings are not index changes"), and the table itself already includes the Aug 18, 2026 RDDT/AVB change (added within days of the real S&P announcement on Aug 13, 2026).
**How to avoid:** Judge freshness by the **content's edit cadence relative to real S&P announcements** (verified: days-scale lag), not by this specific page's creation date. Do implement the caching/fallback strategy below regardless — that protects against a future page-structure change (e.g., another split/merge), not against staleness.
**Warning signs:** none currently — flagged here purely so a future maintainer doesn't misdiagnose "young page" as "unreliable source."

### Pitfall 3: A scrape failure or schema-drift on refresh day must not corrupt the universe table
**What goes wrong:** `pandas.read_html()` finds zero tables, or the `id="changes"` table's columns change shape (e.g., Wikipedia editors restructure it), and a naive daily-refresh job overwrites `universe.parquet` with a broken/empty result.
**Why it happens:** No third-party guarantees this page's table `id`/column structure stays stable forever; MediaWiki content can change without a version bump the pipeline would notice.
**How to avoid:** Cache the last-successfully-parsed `changes` table (a small parquet snapshot with a `fetched_at` timestamp) alongside the main output. On each refresh: (1) fetch + parse; (2) if parse fails, OR the resulting row count is *lower* than the cached snapshot's row count, OR expected columns are missing → log via `loguru.logger.error` (not a bare warning — this should be alert-worthy) and **fall back to the cached snapshot** rather than writing a corrupted/incomplete universe table; (3) only overwrite the cache after a successful, schema-valid, row-count-non-decreasing parse.
**Warning signs:** `universe.parquet`'s S&P 500 row count decreasing between two consecutive daily refreshes (memberships only close, they don't retroactively vanish from history) is itself a good automatic sanity-check trigger.

### Pitfall 4: Bulk-backfilling the full delisted-inclusive Nasdaq universe (thousands of symbols × full history) can hit Tiingo rate limits
**What goes wrong:** Running `TiingoAcquisition.download()` over ~9,000+ resolved Nasdaq symbols in one loop (existing `Acquisition.download()` has no built-in throttling/backoff) can exceed Tiingo's per-hour/per-day request caps, especially on a free-tier key.
**Why it happens:** `Acquisition.download()`/`.refresh()` (existing code, `base/acquisition.py`) currently issue one `_fetch_and_write()` call per symbol in a tight loop with no delay or retry/backoff logic.
**How to avoid:** This research could not obtain an authoritative current numeric rate limit from Tiingo's own docs (see Assumptions Log A3 — community reports were inconsistent, ranging from "25 requests/day" to "2400/hour", and Tiingo's own docs state limits are "approximate and can change"). The planner should treat a full-universe backfill as requiring: (a) an explicit `checkpoint:human-verify` before the first full-scale run so the user can confirm against their actual Tiingo plan's dashboard-displayed limit, and (b) a resumable, batched execution (the existing per-symbol watermark mechanism already makes re-running safe/idempotent — this is a scheduling/pacing concern, not a data-model concern).
**Warning signs:** HTTP 429 responses from Tiingo (the `tiingo` Python client is known, per a linked GitHub issue, to not always surface server rate-limit errors gracefully — treat any unexpected empty/partial response during a bulk run as a possible silent rate-limit hit, not just "no data for this symbol").

## Code Examples

### Filtering Tiingo's supported_tickers.csv to the Nasdaq Stock universe
```python
# Source: verified live against the real file, 2026-09-05
import polars as pl

df = pl.read_csv("supported_tickers.csv")
nasdaq_stocks = df.filter(
    (pl.col("exchange") == "NASDAQ")
    & (pl.col("assetType") == "Stock")
    & (pl.col("priceCurrency") == "USD")
)
# verified: 9,318 rows (vs. nasdaqlisted.txt's 5,593 CURRENT-only rows)
nasdaq_stocks = nasdaq_stocks.rename(
    {"ticker": "symbol", "startDate": "start_date", "endDate": "end_date"}
)
```

### Point-in-time query
```python
def get_symbols_as_of(
    universe: pl.DataFrame, category: str, as_of_date: str
) -> list[str]:
    matched = universe.filter(
        (pl.col("category") == category)
        & (pl.col("start_date") <= as_of_date)
        & (pl.col("end_date").is_null() | (pl.col("end_date") >= as_of_date))
    )
    return matched["symbol"].unique().to_list()
```

## State of the Art

| Old Approach (D-12, pre-this-research) | Current Approach (this research) | When Changed | Impact |
|--------------------------------------|-----------------------------------|---------------|--------|
| `nasdaqlisted.txt` as the Nasdaq-universe source | Tiingo `supported_tickers.csv` filtered to `exchange=="NASDAQ"` | This research session, 2026-09-05 | `nasdaqlisted.txt` cannot represent delisted history at all (not a completeness gap, a structural absence); the pivot is required, not optional, to satisfy the user's explicit new requirement |
| Static "current S&P 500 members" classification (`sp500_constituent` vs `general_market`, no dates) | Point-in-time interval table (`symbol, category, start_date, end_date`) | This research session | Enables `get_symbols_as_of(category, as_of_date)` — the user's explicit "no look-ahead / real historical membership" requirement cannot be satisfied by a single static label |

**Deprecated/outdated:** D-12's "one static label per symbol" classification model is superseded by the interval model above for the `sp500_constituent` category specifically; `general_market` (all Nasdaq-ever-listed, non-S&P500 names) can remain a simpler `(symbol, start_date, end_date)` = Tiingo's own listing window without needing the Wikipedia reconstruction machinery.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The universe/interval reference table is exempt from the "xarray/Zarr only between pipeline modules" hard constraint, on the same footing as `config/instruments.yaml` | Architecture Patterns, Anti-Patterns | If the user intends the constraint to cover *all* persisted data without exception, this design needs revisiting (e.g. wrapping the table in an `xr.Dataset` with a dummy time dimension) — worth a one-line confirmation before planning locks it in |
| A2 | 1976-07-01 is a hard left-censoring boundary for the entire Wikipedia-sourced change log, not an artifact of this one fetch (e.g. a temporary rendering truncation) | Common Pitfalls, Pitfall 1 | If wrong (e.g. older rows exist in a collapsed/paginated section not captured by `pandas.read_html`), the recommended `PIT_COVERAGE_START` constant would be overly conservative; low-cost to re-verify by re-fetching and checking `min(effective_date)` periodically |
| A3 | Tiingo's actual current rate limits (used in Pitfall 4's reasoning) | Common Pitfalls, Pitfall 4 | Community-sourced numbers were inconsistent (25/day to 2400/hour) and Tiingo's own docs state limits are approximate/subject to change; the user should check their own dashboard-displayed limit before a full-universe backfill — does not block building the interval-table logic itself, only the eventual bulk-download execution |
| A4 | `EXPM` (OTC Expert Market) and other OTC-tier exchange values in Tiingo's file are out of scope for "Nasdaq-listed market" and should be excluded, not unioned in | Standard Stack, Alternatives Considered | If the user's intended scope is broader ("full US market" rather than specifically "Nasdaq-listed"), the filter should be relaxed — this reading follows the objective's literal wording ("full Nasdaq-listed market") |

**Confirm A1 and A4 explicitly with the user during planning/discuss-phase** — both are reasoned interpretations of scope/constraint boundaries the user has not directly confirmed for this specific supplement, even though both are well-supported by existing precedent in the codebase (A1) and the objective's literal wording (A4).

## Open Questions

1. **Does the planner need a `general_market` interval table at all, or is a static "currently on Nasdaq" label sufficient for that category?**
   - What we know: `sp500_constituent` clearly needs full point-in-time reconstruction (the user's explicit ask). `general_market` (D-12's second category) is "everything Nasdaq-ever-listed that isn't S&P 500" — Tiingo's `supported_tickers.csv` already gives every such symbol its own `start_date`/`end_date`, so no separate reconstruction algorithm is needed for it; it literally falls out of the Tiingo filter for free.
   - What's unclear: whether the planner should still route `general_market` through the same `get_symbols_as_of()` interface (recommended, for consistency) or treat it as always-current (simpler, but silently reintroduces survivorship bias for the general-market category specifically).
   - Recommendation: route both categories through `get_symbols_as_of()` uniformly — the marginal cost is near zero since the Tiingo data already carries the needed dates.

2. **How does the eventual backtest/portfolio-optimization phase consume `get_symbols_as_of()` to avoid look-ahead bias in a walk-forward simulation** (i.e., re-querying membership at every historical rebalance date, not just once at backtest-setup time)?
   - What we know: This phase only needs to build the queryable interval table and the OHLCV acquisition pivot; using it correctly inside a rolling backtest is a downstream (Phase 5/6-era) concern.
   - What's unclear: not this phase's scope to resolve, but worth a forward-reference note so a future phase's research doesn't have to rediscover that `get_symbols_as_of()` must be called per-rebalance-date, not once.
   - Recommendation: note this explicitly in `UniverseCatalog`'s docstring so future phases inherit the context.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Network access to `nasdaqtrader.com`, `apimedia.tiingo.com`, `raw.githubusercontent.com`, `en.wikipedia.org` | All fetchers in `acquisition/universe.py` | ✓ (all four verified reachable, no auth, in this research session) | — | — |
| `lxml` | `pandas.read_html()` in `SP500MembershipFetcher.fetch_changes()` | ✗ (not yet in `pyproject.toml`) | 6.1.3 available on PyPI | `uv add lxml` — no viable in-repo fallback since `pandas.read_html` hard-requires an HTML backend |
| `TIINGO_API_KEY` env var | `TiingoAcquisition` (existing, unmodified) — only needed for the OHLCV-fetch step, not for building the universe table itself | Not verified in this session (no key used, by design — see conversation) | — | Universe-table construction itself needs **no** Tiingo API key (the `supported_tickers.zip` file and the EOD price endpoints are separate; only price data requires auth) |

**Missing dependencies with no fallback:** none — `lxml` has a trivial, safe fallback (just add it).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.1.1 (`[dependency-groups] dev = ["pytest>=9.1.1"]`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (`pythonpath = ["."]`) |
| Quick run command | `uv run pytest tests/test_universe.py -x -q` |
| Full suite command | `uv run pytest tests/` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D-12 (Nasdaq roster) | `NasdaqUniverseFetcher.fetch()` filters exchange/assetType/currency correctly | unit | `pytest tests/test_universe.py::test_nasdaq_fetcher_filters -x` | ❌ Wave 0 |
| D-12 (PIT reconstruction) | `reconstruct_intervals()` produces correct multi-interval output for a symbol added→removed→re-added | unit | `pytest tests/test_universe.py::test_reconstruct_intervals_reentry -x` | ❌ Wave 0 |
| D-12 (PIT reconstruction) | Left-censored removal (no matching prior add) uses `PIT_COVERAGE_START` sentinel + logs a warning | unit | `pytest tests/test_universe.py::test_reconstruct_intervals_left_censored -x` | ❌ Wave 0 |
| D-12 (query interface) | `get_symbols_as_of()` returns correct membership for a known historical date (e.g. a symbol known to have been removed before the query date is excluded) | unit | `pytest tests/test_universe.py::test_get_symbols_as_of -x` | ❌ Wave 0 |
| D-12 (graceful degradation) | A simulated parse failure/schema-drift falls back to the cached snapshot and logs an error, does not overwrite with bad data | unit | `pytest tests/test_universe.py::test_wikipedia_parse_failure_falls_back_to_cache -x` | ❌ Wave 0 |
| Wiring | `ingest_tiingo.py --universe sp500 --as-of-date ...` resolves to a symbol list without touching `TiingoAcquisition`/`StockDataset` | integration | `pytest tests/test_ingest_tiingo_universe_wiring.py -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_universe.py -x -q`
- **Per wave merge:** `uv run pytest tests/`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_universe.py` — covers all D-12 rows above; must mock all four network fetches (`requests.get`/`.iter_lines()` for `nasdaqlisted.txt`/`supported_tickers.zip`/datahub CSV/Wikipedia HTML) via `monkeypatch`, following the existing `mock_tiingo_client` fixture pattern in `tests/conftest.py` — no test in this suite should make a real network call.
- [ ] `tests/conftest.py` — add fixtures: `sp500_anchor_csv_rows`, `sp500_changes_html_fixture` (a small synthetic HTML snippet with `id="changes"`, not the full live page), `mock_universe_fetchers`.
- [ ] `lxml` — add to `pyproject.toml` before any test importing `pandas.read_html` can pass.

## Security Domain

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | This module makes no authenticated requests — `supported_tickers.zip`, `nasdaqlisted.txt`, the datahub CSV, and Wikipedia are all public/no-auth endpoints |
| V5 Input Validation | yes | Validate fetched CSV/HTML schema before use (Pitfall 3's schema-drift check); never `eval`/`exec` any fetched content |
| V6 Cryptography | no | No secrets/credentials handled by this specific module (the existing `TiingoAcquisition`'s `TIINGO_API_KEY` handling, already implemented and unchanged, remains the only credential-bearing code touched by this supplement) |

### Known Threat Patterns for this module
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed/adversarial HTML causing `pandas.read_html`/`lxml` to hang or behave unexpectedly | Denial of Service | Set an explicit `requests` timeout (e.g. `timeout=30`) on every fetch; treat any exception from the parse step as a fallback-to-cache event (Pitfall 3), never a crash |
| A user pasting a live API key into chat/logs/files (observed during this research session) | Information Disclosure | Never write any credential into a file, git-tracked or scratch; this research session declined to use a key the user pasted directly into the conversation, consistent with the project's existing `TIINGO_API_KEY`-env-var-only constraint and its prior real leak incident |

## Sources

### Primary (HIGH confidence)
- `https://www.tiingo.com/documentation/` — full product/category listing, fetched and reviewed 2026-09-05, no index-constituent endpoint anywhere
- `https://www.tiingo.com/documentation/fundamentals` — Fundamentals API endpoint list, no index-membership data
- `https://www.tiingo.com/documentation/end-of-day` — confirms `supported_tickers.zip` is "updated daily," no index-membership mention
- `https://docs.alpaca.markets/us/reference/corporateactions-1` — explicit statement that delistings/reorganizations are not available via Alpaca's Corporate Actions API
- `https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip` — downloaded and inspected directly in this session (108,562 rows, verified exchange/assetType value enumeration)
- `https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt` — downloaded and inspected directly (5,594 lines incl. footer, confirmed current-listings-only)
- `https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv` — downloaded and inspected directly (504 rows, includes a `Date added` column)
- `https://api.github.com/repos/datasets/s-and-p-500-companies/commits` — confirmed daily automated commits (last: 2026-09-05, same day as this research)
- `https://en.wikipedia.org/w/api.php?action=query&prop=revisions...` — page revision history for both "Historical components of the S&P 500" and the main "S&P 500" article, fetched directly
- `https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500` — full HTML fetched and parsed live via `pandas.read_html` (407×7 table, verified date range 1976-07-01 to 2026-08-18)
- Existing codebase reads: `base/acquisition.py`, `acquisition/tiingo.py`, `base/config.py`, `config/__init__.py`, `dataset/stock.py`, `dataset/backend.py`, `base/backend.py`, `enums/data.py`, `dataset/cleaning.py`, `ingest_tiingo.py`, `tests/conftest.py`, `tests/test_tiingo_acquisition.py`, `.planning/phases/02-multi-market-data-foundation/02-CONTEXT.md`, `02-PATTERNS.md`, `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`

### Secondary (MEDIUM confidence)
- `slopcheck scan --pkg pypi lxml/requests --json` — run live in this session (slopcheck v0.6.1, installed via `uv tool install slopcheck`), both `OK`
- `pip index versions lxml` / `pip index versions beautifulsoup4` — run live via `uv run --with pip`, confirms current PyPI versions

### Tertiary (LOW confidence)
- Tiingo rate-limit figures (WebSearch aggregation of forum/blog posts, no single authoritative current number found) — see Assumptions Log A3
- General characterization of Alpaca's historical-delisted-symbol coverage depth (inferred from Alpaca's operating history/IEX-feed basis, not from an explicit Alpaca statement) — directionally reasonable but not independently verified against Alpaca's actual asset database

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library already a dependency except `lxml`, which was version-checked and slopcheck-verified live
- Architecture (universe module placement, storage format): MEDIUM-HIGH — strongly grounded in existing codebase precedent (D-12's own reasoning, `PlBackend`/`config/instruments.yaml` precedent), but the xarray/Zarr-constraint boundary interpretation (A1) is a reasoned judgment call, not something the user has directly confirmed for this specific case
- Pitfalls: HIGH for the data-source mechanics (left-censoring, schema fragility, delisted-coverage gap) — all verified against live data in this session; MEDIUM-LOW for the Tiingo rate-limit specifics (A3)

**Research date:** 2026-09-05
**Valid until:** 30 days for the architecture/library recommendations; the Wikipedia table's exact row count/date range and Tiingo's exact row counts should be re-verified at implementation time regardless (both are live, mutating data sources, not static references)
