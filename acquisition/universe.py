"""Survivorship-bias-free, point-in-time US-equity symbol universe.

This module deliberately does NOT subclass `base/acquisition.py:Acquisition`.
`Acquisition` subclasses are per-symbol, watermark-driven OHLCV time-series
fetchers (`download()`/`refresh()` loop over `self.config.symbols` and call
`_fetch_and_write(symbol, start, end)` once per symbol). A symbol roster /
membership-interval table is a fundamentally different shape: it is a single
bulk fetch producing a table of *many* symbols' date ranges, not a per-symbol
watermark refresh. Forcing this into `Acquisition`'s contract would require
overriding almost every method meaninglessly -- this preempts future attempts
to force it into that hierarchy (see 02-08-PLAN.md / 02-08-RESEARCH.md
Pattern 1).

Two independent reference-data problems are solved here:

- `TiingoRosterFetcher` and its two data-only subclasses,
  `NasdaqUniverseFetcher` (NASDAQ-listed only) and `USEquityUniverseFetcher`
  (the full NYSE + NASDAQ + AMEX listed market): exchange-scoped Common Stock
  rosters including historically delisted symbols, sourced from Tiingo's own
  `supported_tickers.csv` (NOT `nasdaqlisted.txt`, which only lists
  currently-active tickers and cannot represent delisted history at all).
  Adding a roster is a data change -- three class constants -- not a code
  change. The two are SIBLINGS: `us_all` is a superset of `nasdaq_all` and
  neither replaces the other (260906-0iy D-01/D-02).
- `IndexMembershipFetcher` and its two data-only subclasses,
  `SP500MembershipFetcher` and `Nasdaq100MembershipFetcher`: point-in-time
  index constituent membership, reconstructed via forward-chronological
  event simulation over Wikipedia's "Historical components of ..." change
  log, anchored against a known-correct current snapshot. Adding a third
  index is a data change -- nine class constants plus one parse method
  (`fetch_anchor()`) -- not a code change; the reconstruction algorithm, the
  change-log parse and the whole fetch/cache safety envelope live once, on
  the base.

`UniverseCatalog` merges both into one `(symbol, category, start_date,
end_date, end_date_is_inferred)` reference table, persisted via the existing `PlBackend` as
parquet -- per Locked Decision A1 (02-08-PLAN.md), this table is
reference/metadata, not xarray/Zarr pipeline data, on the same footing as
`config/instruments.yaml`.
"""

import datetime
import io
import os
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import pandas as pd
import polars as pl
import requests
from loguru import logger

from base.config import UniverseConfig
from dataset.backend import PlBackend
from enums.data import UniverseCategory

#: Contact string sent in the outbound `User-Agent` when scraping Wikipedia,
#: whose bot policy asks for one. Read from the environment per CLAUDE.md's
#: env-vars-for-anything-sensitive convention: the previous value hardcoded a
#: developer's PERSONAL email address into source, transmitted it to Wikipedia
#: on every fetch, and followed the repo to every future contributor and any
#: public fork. The default is a neutral project URL, so an unset variable is
#: still a polite User-Agent.
_CONTACT = os.environ.get(
    "QUANTLAB_CONTACT", "https://github.com/quantlab/quantlab"
)


class TiingoRosterFetcher:
    """Shared machinery for filtering Tiingo's full historical ticker
    directory down to one exchange-scoped common-stock roster.

    A roster is DATA here, not code -- exactly the `IndexMembershipFetcher`
    precedent already in this module. A subclass supplies a tuple of exchange
    tokens, a row-count floor and a category token, and inherits the whole
    download/unzip/filter/rename/guard body unchanged.

    Source: Tiingo's own `supported_tickers.csv` -- NOT `nasdaqlisted.txt`,
    which only lists currently-listed securities and cannot represent
    delisted history at all (02-08-RESEARCH.md finding #2). This is the
    property that makes the resulting rosters survivorship-bias free.

    Subclass-bound class constants:

    - ``EXCHANGE_FILTER`` -- exact-match exchange tokens to keep.
    - ``MIN_ROSTER_ROWS`` -- the structural-drift floor, sized to the
      subclass's own observed magnitude.
    - ``CATEGORY`` -- the ``enums.data.UniverseCategory`` token.

    `fetch()` filters on exact-match string literals, so a casing/spelling
    change or a renamed column value in Tiingo's feed yields ZERO rows
    WITHOUT raising -- and `UniverseCatalog.build()` would concatenate that
    empty frame happily, after which `save()` overwrites the previous good
    reference table. `MIN_ROSTER_ROWS` is the guard that turns that silent
    corruption into a loud failure, and it lives on this base so every roster
    gets it BY CONSTRUCTION.
    """

    SOURCE_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
    ASSET_TYPE = "Stock"
    PRICE_CURRENCY = "USD"

    #: Exact-match `exchange` tokens kept by `fetch()`.
    EXCHANGE_FILTER: tuple[str, ...]
    #: Structural-drift floor -- see the class docstring.
    MIN_ROSTER_ROWS: int
    #: Typed as the literal, not `str`: a fetcher registered with a token
    #: absent from `enums.data.UniverseCategory` is then a type error rather
    #: than something only a set-comparing test notices at test time.
    CATEGORY: UniverseCategory

    def fetch(self) -> pl.DataFrame:
        response = requests.get(self.SOURCE_URL, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            with archive.open("supported_tickers.csv") as csv_file:
                data = pl.read_csv(csv_file)

        data = data.filter(
            pl.col("exchange").is_in(self.EXCHANGE_FILTER)
            & (pl.col("assetType") == self.ASSET_TYPE)
            & (pl.col("priceCurrency") == self.PRICE_CURRENCY)
        )
        data = data.rename(
            {"ticker": "symbol", "startDate": "start_date", "endDate": "end_date"}
        )
        data = data.select(["symbol", "start_date", "end_date"])
        if len(data) < self.MIN_ROSTER_ROWS:
            raise ValueError(
                f"Tiingo supported_tickers filtered down to only {len(data)} "
                f"rows (exchange={self.EXCHANGE_FILTER}, "
                f"assetType={self.ASSET_TYPE!r}, "
                f"priceCurrency={self.PRICE_CURRENCY!r}), fewer than the "
                f"minimum {self.MIN_ROSTER_ROWS} -- the source's token "
                f"vocabulary has drifted. Refusing to continue rather than "
                f"overwriting the reference table with a truncated roster."
            )
        return data


class NasdaqUniverseFetcher(TiingoRosterFetcher):
    """The full NASDAQ-listed Common Stock roster, including historically
    delisted symbols.

    Data-only subclass: every constant below is byte-for-byte what it was
    before the shared body was extracted onto `TiingoRosterFetcher`, and
    `test_nasdaq_roster_exchange_filter_and_symbol_set_are_unchanged` pins
    that by direct equality rather than by grep.
    """

    # Locked Decision A4 (02-08-PLAN.md): NASDAQ-listed common stock only, no
    # OTC/Expert-Market tiers. Matches the objective's literal "Nasdaq
    # market" framing. 260906-0iy D-02 re-locks it: the full-US-market roster
    # is a NEW SIBLING (`USEquityUniverseFetcher`), never a widening of this.
    EXCHANGE_FILTER = ("NASDAQ",)

    # The same safety envelope the index anchors already have, applied to what
    # was the LARGEST category in the table. 1000 is far below the real ~10k so
    # a legitimate shrink never trips it. See `TiingoRosterFetcher` for why an
    # unguarded zero-row filter would destroy `universe.parquet`.
    MIN_ROSTER_ROWS = 1000

    CATEGORY: UniverseCategory = "nasdaq_all"


class USEquityUniverseFetcher(TiingoRosterFetcher):
    """The full US listed-equity roster -- NYSE + NASDAQ + AMEX common stock
    priced in USD, delisted names included (260906-0iy D-01).

    A NEW SIBLING of `NasdaqUniverseFetcher`, not a replacement: `nasdaq_all`
    keeps its exact prior semantics per D-02, and `us_all` is a strict
    SUPERSET of it. Both are deliberately retained.

    **Why the AMEX needs two tokens.** Tiingo's `exchange` column carries the
    following distinct values over the 108,561-row directory (measured
    2026-09-06)::

        NMFQS 49827 | PINK 20695 | NASDAQ 10969 | NYSE 9296 | SHE 3808
        SHG 3429 | BATS 2016 | OTCMKTS 1992 | NYSE ARCA 1532 | OTCGREY 1518
        EXPM 931 | OTCBB 602 | OTCQB 447 | OTCCE 389 | AMEX 386
        NYSE MKT 224 | OTCQX 204 | (empty) 135 | OTCD 61 | SHGB 44
        SHEB 42 | LSE 11 | NYSE NAT 3

    The American Stock Exchange appears under BOTH `AMEX` (386) and
    `NYSE MKT` (224), because Tiingo never re-labelled its historical rows
    across the exchange's AMEX -> NYSE Amex -> NYSE MKT -> NYSE American
    rename history. There is no `NYSE American` token at all. Omitting either
    silently drops ~224 real tickers.

    **Why some NYSE-prefixed tokens are excluded.** `NYSE ARCA` (1532) and
    `NYSE NAT` (3) are DIFFERENT exchanges that merely share the brand --
    ARCA is predominantly ETFs -- and `BATS` (2016) is a different exchange
    outright. D-01 scopes this roster to NYSE, NASDAQ and AMEX, so all three
    are deliberately out. The empty-string exchange is excluded for free by
    exact-match `is_in`.

    Observed yield of this exact filter: 16,138 rows / 15,425 distinct
    tickers (NASDAQ 9318, NYSE 6284, AMEX 336, NYSE MKT 200). Rows exceed
    tickers because ~700 tickers carry more than one exchange row, which is
    why `UniverseCatalog`'s interval queries de-duplicate on symbol.
    """

    EXCHANGE_FILTER = ("NASDAQ", "NYSE", "AMEX", "NYSE MKT")

    # Roughly half the observed 16,138 rows: low enough that a legitimate
    # market contraction never trips it, high enough that a token-vocabulary
    # drift which silently zeroes the filter always does. Same safety-envelope
    # idiom as `NasdaqUniverseFetcher.MIN_ROSTER_ROWS` (1000 vs ~10k) and
    # `Nasdaq100MembershipFetcher.MIN_ANCHOR_ROWS` (50 vs ~102), sized to its
    # own magnitude rather than shared as one number across all three.
    MIN_ROSTER_ROWS = 8000

    CATEGORY: UniverseCategory = "us_all"


#: Cell values that mean "no ticker on this side of the change row". Kept
#: explicit because `pd.read_html` is now told NOT to infer NA at all (see
#: `IndexMembershipFetcher._parse_changes_table`): its default NA vocabulary
#: overlaps the ticker namespace, and `NA` is a real US equity ticker. The
#: dash variants are the em/en/hyphen glyphs Wikipedia uses for "none".
_BLANK_TICKER_CELLS = frozenset({"", "-", "–", "—"})


class IndexMembershipFetcher(ABC):
    """Shared machinery for reconstructing point-in-time index membership
    intervals from a current-constituent anchor plus a dated change log.

    An index is DATA here, not code: a subclass supplies nine class constants
    and ONE parse method (`fetch_anchor()`), and inherits the whole
    reconstruction algorithm, the change-log parse and the whole fetch/cache
    safety envelope unchanged.

    Subclass-bound class constants:

    - ``ANCHOR_URL`` -- current-constituent snapshot source.
    - ``CHANGES_URL`` -- dated add/remove change-log source.
    - ``PIT_COVERAGE_START`` -- the earliest date the change log actually
      covers. Point-in-time queries before it cannot be correctly answered
      and must be rejected, never silently answered with a partial history.
    - ``CACHE_FILENAME`` -- per-index snapshot filename under ``cache_dir``.
      Two indices must never share one cache file.
    - ``INDEX_LABEL`` -- human-readable index name, used in log/error text.
    - ``CATEGORY`` -- the ``enums.data.UniverseCategory`` token.
    - ``EXPECTED_SOURCE_HEADER`` -- the exact flattened header the change-log
      table must have. This is the identity the table is SELECTED by and the
      contract its columns are read against.
    - ``DATE_HEADER`` -- which ``EXPECTED_SOURCE_HEADER`` entry carries the
      effective date (the two pages word it differently).
    - ``CHANGES_TABLE_ATTRS`` -- optional ``pd.read_html(attrs=...)``
      pre-filter when the page marks its change log with an id/class.

    Three behaviours are load-bearing SAFETY properties, not incidental
    implementation. They live on this base precisely so every subclass gets
    them BY CONSTRUCTION:

    1. **Source-header validation.** `_parse_changes_table()` SELECTS the
       change-log table by matching its flattened header against
       ``EXPECTED_SOURCE_HEADER``, and reads the ticker columns BY NAME off
       that verified header. Previously each subclass assigned column names
       positionally and the base then checked that the names it had just
       assigned were present -- an unconditionally-true check, while the one
       drift a scraped source most easily produces (a header REORDER, same
       column count) was accepted and silently inverted add/remove. Selecting
       by header identity also removes the old `tables[0]` positional pick,
       so a table inserted ahead of the change log is no longer parsed as the
       change log.
    2. **Row-count monotonicity.** A live table with fewer rows than the
       cached snapshot is treated as a parse failure / schema drift, because
       memberships only close, they don't retroactively vanish.
    3. **Non-destructive fallback.** On any fetch/parse failure the cached
       snapshot is returned and the cache file is NOT overwritten, so a
       single bad parse cannot poison every future run.
    """

    ANCHOR_URL: str
    CHANGES_URL: str
    PIT_COVERAGE_START: str
    CACHE_FILENAME: str
    INDEX_LABEL: str
    #: Typed as the literal, not `str`: a fetcher registered with a token
    #: absent from `enums.data.UniverseCategory` is then a type error rather
    #: than something only `test_catalog_build_emits_all_three_categories`
    #: notices, at test time, and only because it compares sets.
    CATEGORY: UniverseCategory

    #: The exact flattened change-log header (see `_flatten_header`). Both the
    #: table SELECTOR and the column contract -- a source whose header drifts
    #: raises instead of being parsed against assumptions that no longer hold.
    EXPECTED_SOURCE_HEADER: tuple[str, ...]
    #: Which `EXPECTED_SOURCE_HEADER` entry carries the effective date.
    DATE_HEADER: str
    #: The two ticker columns actually consumed. Defaulted because both live
    #: sources word them identically; still validated against
    #: `EXPECTED_SOURCE_HEADER` so an inconsistent subclass fails loudly.
    ADDED_TICKER_HEADER: str = "Added Ticker"
    REMOVED_TICKER_HEADER: str = "Removed Ticker"
    #: Optional `pd.read_html(attrs=...)` pre-filter, when the page marks its
    #: change log (the S&P 500 page has `id="changes"`; the Nasdaq-100 page
    #: has no such marker and relies on header identity alone).
    CHANGES_TABLE_ATTRS: dict[str, str] | None = None

    def __init__(self, cache_dir: str):
        self._cache_path = Path(cache_dir) / self.CACHE_FILENAME
        #: Provenance of the frame `fetch_changes()` last returned. The
        #: cached-snapshot fallback is correct and deliberate, but it must not
        #: be INDISTINGUISHABLE from a fresh fetch once persisted: a
        #: permanently-broken source would otherwise freeze the universe at
        #: the cache date with only a logger.error line -- easily lost in a
        #: cron log -- to record it. `UniverseCatalog.build()` reads these.
        self.changes_are_stale = False
        self.changes_source_asof: str | None = None

    @abstractmethod
    def fetch_anchor(self) -> pl.DataFrame:
        """Return the current-constituent anchor as a `[symbol, date_added]`
        frame. `date_added` may be entirely null when the source carries no
        such column -- `reconstruct_intervals()` falls back to
        `PIT_COVERAGE_START` for those symbols.
        """

    @staticmethod
    def _flatten_header(columns) -> tuple[str, ...]:
        """Flatten a (possibly two-level) `pd.read_html` header to plain
        strings: `("Added", "Ticker")` -> `"Added Ticker"`, and a label
        repeated across both header rows (`("Reason", "Reason")`) -> `"Reason"`.
        Whitespace is collapsed so a stray NBSP or line break in the source
        markup does not read as a different header.
        """
        flattened: list[str] = []
        for column in columns:
            parts = column if isinstance(column, tuple) else (column,)
            cleaned: list[str] = []
            for part in parts:
                text = " ".join(str(part).split())
                if text and (not cleaned or cleaned[-1] != text):
                    cleaned.append(text)
            flattened.append(" ".join(cleaned))
        return tuple(flattened)

    def _parse_changes_table(self, html_text: str) -> pd.DataFrame:
        """Parse this index's change-log HTML into an `effective_date` /
        `added_ticker` / `removed_ticker` frame.

        Concrete and SHARED, deliberately (see safety property 1 on the class
        docstring). The change-log table is selected by matching its flattened
        header against `EXPECTED_SOURCE_HEADER`, and the three consumed
        columns are then read BY NAME off that verified header rather than by
        position. A subclass supplies the header constants, not a parse body,
        so no index can ship a parse that skips the validation.
        """
        # `keep_default_na=False, na_values=[]` is load-bearing, not tidiness.
        # pandas' default NA vocabulary (`NA`, `N/A`, `NULL`, `NaN`, `None`,
        # `nan`, `-`, `1.#IND`, ...) OVERLAPS the ticker namespace -- `NA` is a
        # real, historically-listed US equity ticker. Coerced to NaN it becomes
        # the exact sentinel `reconstruct_intervals()` reads as "no change on
        # this side", so the add/remove event for such a ticker was silently
        # discarded from the change log entirely. Blank cells are recovered
        # explicitly below instead.
        read_kwargs: dict = {
            "flavor": "lxml",
            "keep_default_na": False,
            "na_values": [],
        }
        if self.CHANGES_TABLE_ATTRS is not None:
            read_kwargs["attrs"] = self.CHANGES_TABLE_ATTRS
        tables = pd.read_html(io.StringIO(html_text), **read_kwargs)

        for header in (
            self.DATE_HEADER,
            self.ADDED_TICKER_HEADER,
            self.REMOVED_TICKER_HEADER,
        ):
            if header not in self.EXPECTED_SOURCE_HEADER:
                raise ValueError(
                    f"{self.INDEX_LABEL}: {header!r} is not one of "
                    f"EXPECTED_SOURCE_HEADER {self.EXPECTED_SOURCE_HEADER} -- "
                    f"this class's header constants contradict each other."
                )

        matched = next(
            (
                table
                for table in tables
                if self._flatten_header(table.columns)
                == self.EXPECTED_SOURCE_HEADER
            ),
            None,
        )
        if matched is None:
            raise ValueError(
                f"{self.INDEX_LABEL}: no table at {self.CHANGES_URL} carries "
                f"the expected change-log header "
                f"{self.EXPECTED_SOURCE_HEADER}; saw "
                f"{[self._flatten_header(t.columns) for t in tables]}. "
                f"Refusing to parse a table whose header was not verified -- "
                f"a reordered header has the same column count as a correct "
                f"one, so accepting it would silently invert add/remove."
            )

        changes = matched.copy()
        changes.columns = self._flatten_header(changes.columns)

        parsed = pd.DataFrame(
            {
                "effective_date": changes[self.DATE_HEADER],
                "added_ticker": changes[self.ADDED_TICKER_HEADER],
                "removed_ticker": changes[self.REMOVED_TICKER_HEADER],
            }
        )
        # Re-derive the "no change on this side" sentinel EXPLICITLY, now that
        # pandas is no longer allowed to guess it. `reconstruct_intervals()`
        # tests `is not None`, so a blank must be a real `None` and every other
        # cell -- including the ticker `NA` -- must survive as itself.
        for column in ("added_ticker", "removed_ticker"):
            stripped = parsed[column].astype(str).str.strip().tolist()
            # `dtype=object` keeps the sentinel a real `None`; a plain list
            # assignment lets pandas re-infer a string dtype and turn it back
            # into `nan`. Both survive `pl.from_pandas` as null, but only the
            # explicit form says so at the layer a reader is looking at.
            parsed[column] = pd.Series(
                [
                    None if value in _BLANK_TICKER_CELLS else value
                    for value in stripped
                ],
                dtype=object,
                index=parsed.index,
            )

        # `format="mixed"` silences the `Could not infer format ... falling
        # back to dateutil` warning both pages provoked, and `errors="coerce"`
        # turns an unparseable cell into NaT instead of raising. That
        # distinction matters: Wikipedia routinely carries footnote markers and
        # date ranges in date columns, and a raised DateParseError is caught by
        # fetch_changes()'s broad `except Exception`, which converts one bad
        # cell into a PERMANENT, near-silent fallback to the stale cache.
        # Surfacing the offending rows instead keeps the failure legible.
        parsed_dates = pd.to_datetime(
            parsed["effective_date"], format="mixed", errors="coerce"
        )
        unparseable = parsed["effective_date"][parsed_dates.isna()]
        if len(unparseable):
            raise ValueError(
                f"{self.INDEX_LABEL}: unparseable effective_date cells in the "
                f"change log at {self.CHANGES_URL}: "
                f"{unparseable.tolist()}. Refusing to reconstruct membership "
                f"from a table whose dates were silently dropped."
            )
        parsed["effective_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
        return parsed

    def fetch_changes(self) -> pl.DataFrame:
        cached_row_count = 0
        if self._cache_path.exists():
            try:
                cached_row_count = len(pl.read_parquet(self._cache_path))
            except Exception as exc:
                # Swallowing this silently DISABLES the row-count monotonicity
                # guard below (`len(changes) < 0` can never be true) -- the
                # very property the class docstring calls load-bearing. The
                # fallback is still 0 (a corrupt cache must not block a fresh
                # fetch), but it must be loud: this is the one state in which
                # a shrunken table would be accepted as new truth.
                logger.error(
                    f"{self.INDEX_LABEL}: could not read the cached changes "
                    f"snapshot at {self._cache_path}: {exc}. The row-count "
                    f"monotonicity guard is DISABLED for this run -- a "
                    f"shrunken live table will not be rejected. Delete the "
                    f"file to re-seed it from a good fetch."
                )
                cached_row_count = 0

        try:
            response = requests.get(
                self.CHANGES_URL,
                headers={"User-Agent": f"quantlab (contact: {_CONTACT})"},
                timeout=30,
            )
            response.raise_for_status()

            changes = self._parse_changes_table(response.text)

            # Post-condition backstop, NOT the source-header guard. The real
            # validation is `_parse_changes_table`'s `EXPECTED_SOURCE_HEADER`
            # match against the SOURCE's own header; this only catches a
            # subclass that overrides the (concrete) base parse and returns
            # the wrong shape. It is deliberately no longer described as
            # validating the source: when each subclass assigned these very
            # names positionally, this check was unconditionally true while
            # nothing looked at the source header at all.
            required_columns = {"effective_date", "added_ticker", "removed_ticker"}
            if not required_columns.issubset(set(changes.columns)):
                raise ValueError(
                    f"{self.INDEX_LABEL}: _parse_changes_table() returned a "
                    f"frame missing required columns: expected "
                    f"{required_columns}, got {set(changes.columns)}"
                )
            if len(changes) < cached_row_count:
                raise ValueError(
                    f"Parsed Wikipedia changes table has fewer rows "
                    f"({len(changes)}) than the cached snapshot "
                    f"({cached_row_count}) -- memberships only close, they "
                    f"don't retroactively vanish; treating this as a parse "
                    f"failure/schema-drift."
                )

            parsed = pl.from_pandas(changes)
        except Exception as exc:
            logger.error(
                f"Failed to fetch/parse {self.INDEX_LABEL} changes from Wikipedia "
                f"({self.CHANGES_URL}): {exc}. Falling back to cached "
                f"snapshot; NOT overwriting the cache file."
            )
            cached = self._load_cache()
            self.changes_are_stale = True
            self.changes_source_asof = datetime.datetime.fromtimestamp(
                self._cache_path.stat().st_mtime
            ).isoformat(timespec="seconds")
            return cached

        # Write atomically. A plain in-place `write_parquet` leaves a
        # TRUNCATED parquet if the run is interrupted mid-write, which is
        # exactly the corrupt-cache state that disables the monotonicity guard
        # above -- the cache write could manufacture its own blind spot.
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._cache_path.with_suffix(".parquet.tmp")
        parsed.write_parquet(tmp_path)
        tmp_path.replace(self._cache_path)
        self.changes_are_stale = False
        self.changes_source_asof = datetime.date.today().isoformat()
        return parsed

    def _load_cache(self) -> pl.DataFrame:
        if self._cache_path.exists():
            return pl.read_parquet(self._cache_path)
        raise RuntimeError(
            f"No cached {self.INDEX_LABEL} changes snapshot available and live "
            f"fetch failed -- cannot build {self.CATEGORY} intervals."
        )

    def reconstruct_intervals(
        self, anchor: pl.DataFrame, changes: pl.DataFrame
    ) -> pl.DataFrame:
        """Replay the change log forward and reconcile it against the anchor.

        The anchor is AUTHORITATIVE for "is this symbol a member today"; the
        change log is authoritative for "when did that change". Those two
        sources disagree in exactly three ways, and all three are reconciled
        here -- none may pass silently, because a wrong-but-plausible roster
        is indistinguishable downstream from a correct one:

        1. **Open interval, symbol absent from the anchor.** The log is
           missing a removal. Closed at `last_eff` -- the last date the change
           log covers at all, which is NOT an observation about this symbol --
           and flagged `end_date_is_inferred=True`.
        2. **The log's last event for the symbol was a REMOVAL, yet the
           anchor still lists it as a current constituent.** The log is
           missing a re-addition -- the live Nasdaq-100 log is known to be
           asymmetric (16 drop-only rows), so this is a real shape, not a
           hypothetical. Membership is re-opened from that removal date with
           a warning. Without this branch the symbol is silently recorded as
           a FORMER member and `get_symbols_as_of(category, today)` omits a
           current constituent.
        3. **Anchor member with no event anywhere in the log.** Either an
           original constituent or added before `PIT_COVERAGE_START`; opened
           at `date_added or PIT_COVERAGE_START`.

        **`end_date_is_inferred`.** Carried through to the persisted table so a
        FABRICATED interval end is distinguishable from an observed one. Only
        case 1 sets it: that `end_date` is the change log's own right edge, an
        unrelated event's date, chosen because the anchor proves the symbol is
        no longer a member while the log never says when it stopped. Writing
        that as though it were observed -- with only a `logger.warning` to
        distinguish it -- is the same silent-plausible-answer failure the rest
        of this module exists to prevent. Consumers that care about exact
        removal dates must filter on this column; the point-in-time queries in
        `UniverseCatalog.get_symbols_as_of` deliberately do not, because a
        bounded end is still much closer to the truth than an open one.
        """
        anchor_symbols = set(anchor["symbol"])
        anchor_date_added = dict(zip(anchor["symbol"], anchor["date_added"]))

        changes_sorted = changes.sort("effective_date")
        open_intervals: dict[str, str] = {}
        # 4-tuples: (symbol, start_date, end_date, end_date_is_inferred).
        closed: list[tuple[str, str, str | None, bool]] = []

        last_eff = self.PIT_COVERAGE_START
        for row in changes_sorted.iter_rows(named=True):
            eff = row["effective_date"]
            last_eff = eff
            if row["removed_ticker"] is not None:
                sym = row["removed_ticker"]
                if sym in open_intervals:
                    closed.append((sym, open_intervals.pop(sym), eff, False))
                else:
                    logger.warning(
                        f"{sym}: removal at {eff} has no matching prior "
                        f"'added' event -- left-censored interval, using "
                        f"PIT_COVERAGE_START ({self.PIT_COVERAGE_START}) as "
                        f"start_date"
                    )
                    closed.append((sym, self.PIT_COVERAGE_START, eff, False))
            if row["added_ticker"] is not None:
                sym = row["added_ticker"]
                if sym in open_intervals:
                    logger.warning(
                        f"{sym}: duplicate 'added' event at {eff} while an "
                        f"interval was already open since "
                        f"{open_intervals[sym]} -- data quality issue, "
                        f"keeping the earlier open date"
                    )
                else:
                    open_intervals[sym] = eff

        # Reconcile remaining open intervals against the current anchor.
        for sym, start in open_intervals.items():
            if sym not in anchor_symbols:
                logger.warning(
                    f"{sym}: open interval since {start} but not in current "
                    f"anchor set -- anchor CSV may be stale relative to the "
                    f"Wikipedia change log"
                )
                closed.append((sym, start, last_eff, True))
            else:
                closed.append((sym, start, None, False))

        # Third reconciliation direction: the change log's LAST event for a
        # symbol is a REMOVAL, yet the anchor still lists it as a current
        # constituent -- the log is missing a re-addition. Without this branch
        # the symbol falls through both loops (it is in `seen_symbols`, so the
        # anchor-only loop below skips it) and is silently persisted as a
        # FORMER member, which makes `get_symbols_as_of(category, today)` omit
        # a current constituent and the densified panel read `is_member=False`
        # at the live edge. The anchor is authoritative for "member today", so
        # membership is re-opened rather than left closed.
        symbols_still_open = set(open_intervals)
        closed_by_symbol: dict[str, list[int]] = {}
        for position, (sym, _start, _end, _inferred) in enumerate(closed):
            closed_by_symbol.setdefault(sym, []).append(position)

        for sym in sorted(anchor_symbols - symbols_still_open):
            positions = closed_by_symbol.get(sym)
            if not positions:
                continue  # never seen in the log -- handled by the loop below
            latest = max(positions, key=lambda i: closed[i][2] or "")
            last_removal = closed[latest][2]
            if last_removal is None:
                continue  # already open-ended; nothing to reconcile
            logger.warning(
                f"{sym}: the change log's last event is a removal at "
                f"{last_removal}, but the current anchor still lists it as a "
                f"constituent -- the change log is missing a re-addition. "
                f"Re-opening membership from that removal date rather than "
                f"silently recording a current constituent as a former one."
            )
            closed.append((sym, last_removal, None, False))

        # Anchor members with NO 'added'/'removed' event anywhere in the
        # change log are either original constituents or were added before
        # PIT_COVERAGE_START.
        seen_symbols = {c[0] for c in closed}
        for sym in anchor_symbols - seen_symbols:
            start = anchor_date_added.get(sym) or self.PIT_COVERAGE_START
            closed.append((sym, start, None, False))

        return pl.DataFrame(
            closed,
            schema=[
                "symbol",
                "start_date",
                "end_date",
                "end_date_is_inferred",
            ],
            orient="row",
        )

    def build_intervals(self) -> pl.DataFrame:
        anchor = self.fetch_anchor()
        changes = self.fetch_changes()
        return self.reconstruct_intervals(anchor, changes)


class SP500MembershipFetcher(IndexMembershipFetcher):
    """Reconstructs point-in-time S&P 500 membership intervals from a
    current-anchor snapshot plus a dated historical change log.

    `PIT_COVERAGE_START` ("1976-07-01") is the verified earliest row in the
    Wikipedia `id="changes"` table -- NOT the same as the page's own prose
    claim of 1963 coverage (02-08-RESEARCH.md Pitfall 1). Point-in-time
    queries before this date cannot be correctly answered and must be
    explicitly rejected, never silently answered with an incomplete history.
    """

    ANCHOR_URL = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
        "main/data/constituents.csv"
    )
    CHANGES_URL = (
        "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
    )
    PIT_COVERAGE_START = "1976-07-01"
    CACHE_FILENAME = "sp500_changes_snapshot.parquet"
    INDEX_LABEL = "S&P 500"
    CATEGORY = "sp500_constituent"

    # Seven columns: this page also carries `Refs`, which the Nasdaq-100 page
    # does not. The `id="changes"` marker narrows `read_html` before the
    # header identity check does the real work.
    EXPECTED_SOURCE_HEADER = (
        "Effective Date",
        "Added Ticker",
        "Added Security",
        "Removed Ticker",
        "Removed Security",
        "Reason",
        "Refs",
    )
    DATE_HEADER = "Effective Date"
    CHANGES_TABLE_ATTRS = {"id": "changes"}

    def fetch_anchor(self) -> pl.DataFrame:
        response = requests.get(self.ANCHOR_URL, timeout=30)
        response.raise_for_status()
        data = pl.read_csv(io.StringIO(response.text))
        data = data.rename({"Symbol": "symbol", "Date added": "date_added"})
        return data.select(["symbol", "date_added"])


class Nasdaq100MembershipFetcher(IndexMembershipFetcher):
    """Reconstructs point-in-time Nasdaq-100 (NDX) membership intervals from
    a current-constituent anchor snapshot plus a dated historical change log.

    `PIT_COVERAGE_START` ("2007-02-01") is the verified earliest row the
    Wikipedia "Historical components of the Nasdaq-100" table actually
    contains (`LOGI` added / `CMVT` removed, 03.1-RESEARCH.md Finding 2).
    Point-in-time NDX queries before that date cannot be correctly answered
    and must be explicitly rejected, never silently answered with an
    incomplete roster. Note this left edge is ~31 years later than the S&P
    500's -- the two categories must not be silently unioned onto one axis
    that implies coverage neither has.

    **The anchor is the least stable input in this data layer.** Unlike the
    S&P 500, whose anchor is a GitHub-hosted CSV, the NDX has no such
    analogue: Wikipedia's `Nasdaq-100` page renders its components through a
    navbox template with no parseable constituents table (RESEARCH Finding
    5), so the anchor is scraped from the commercial page
    `https://stockanalysis.com/list/nasdaq-100-stocks/`. If that source dies,
    the documented alternative is `https://www.slickcharts.com/nasdaq100`
    (also 102 rows, columns `# / Company / Symbol / Weight / Price / Chg /
    % Chg`) -- switch `ANCHOR_URL` and `fetch_anchor()`'s column handling to
    it rather than implementing both. This is exactly why the base class's
    cached-snapshot fallback matters MORE here than for the S&P 500, and why
    `fetch_anchor()` carries an explicit shape guard of its own.

    **102 rows, not 100.** The live anchor probed at 102 constituents because
    the index carries multiple share classes for some issuers (GOOGL/GOOG,
    FOX/FOXA). A `== 100` expectation is wrong against correct data.

    Neither anchor source carries a `date_added` column, so `fetch_anchor()`
    synthesises an explicit all-null one; the base's
    `anchor_date_added.get(sym) or self.PIT_COVERAGE_START` fallback then
    fires as written instead of raising `KeyError`.
    """

    ANCHOR_URL = "https://stockanalysis.com/list/nasdaq-100-stocks/"
    CHANGES_URL = (
        "https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100"
    )
    PIT_COVERAGE_START = "2007-02-01"
    CACHE_FILENAME = "nasdaq100_changes_snapshot.parquet"
    INDEX_LABEL = "Nasdaq-100"
    CATEGORY = "nasdaq100_constituent"

    # Six columns after `pd.read_html` flattens the page's two-level
    # `Date | Added(Ticker, Security) | Removed(Ticker, Security) | Reason`
    # header -- one fewer than the S&P 500 table, which also has `Refs`, and
    # the date column is worded `Date` rather than `Effective Date`.
    #
    # This page carries no id/class on its change log, so there is no
    # `CHANGES_TABLE_ATTRS` pre-filter: the header identity IS the selector.
    # That replaces the previous `tables[0]` positional pick, under which any
    # table Wikipedia inserted ahead of the change log became the change log.
    EXPECTED_SOURCE_HEADER = (
        "Date",
        "Added Ticker",
        "Added Security",
        "Removed Ticker",
        "Removed Security",
        "Reason",
    )
    DATE_HEADER = "Date"

    # A structurally-drifted commercial page must fail loudly rather than
    # yield a three-symbol "index": a silently truncated anchor closes every
    # unmentioned membership and quietly reintroduces the survivorship bias
    # this whole data layer exists to remove. 50 is deliberately far below
    # the real 102 so a legitimate index resize never trips it.
    MIN_ANCHOR_ROWS = 50

    def fetch_anchor(self) -> pl.DataFrame:
        response = requests.get(self.ANCHOR_URL, timeout=30)
        response.raise_for_status()

        # `keep_default_na=False, na_values=[]` for the same reason as the
        # change-log parse: pandas' default NA vocabulary overlaps the ticker
        # namespace. Here the corruption ran the other way -- the old
        # `str(sym)` turned a coerced NaN into the literal string "nan", which
        # entered the anchor as a FABRICATED permanent constituent (and hence
        # an always-True column in the densified panel) while the real ticker
        # `NA` vanished from a table whose whole purpose is to contain every
        # symbol that was ever a member.
        tables = pd.read_html(
            io.StringIO(response.text),
            flavor="lxml",
            keep_default_na=False,
            na_values=[],
        )
        anchor = next(
            (table for table in tables if "Symbol" in table.columns), None
        )
        if anchor is None:
            raise ValueError(
                f"Parsed {self.INDEX_LABEL} anchor has no table carrying a "
                f"'Symbol' column ({self.ANCHOR_URL}) -- the anchor source is "
                f"a commercial scraped page whose markup has drifted. "
                f"Refusing to continue rather than reconstructing membership "
                f"from a structurally wrong anchor."
            )

        # Drop genuinely-absent cells (ragged rows are padded with real NaN by
        # `read_html` whatever `na_values` says) and blanks, by NULLNESS rather
        # than by comparing against the string "nan" -- the latter would drop a
        # ticker literally named `NAN`, reintroducing this very bug.
        raw = anchor["Symbol"]
        symbols = [
            symbol
            for symbol in raw[raw.notna()].astype(str).str.strip().tolist()
            if symbol
        ]

        # Counted AFTER cleaning: a page that renders 102 rows of which only
        # three carry a ticker is exactly as drifted as one rendering 3 rows.
        if len(symbols) < self.MIN_ANCHOR_ROWS:
            raise ValueError(
                f"Parsed {self.INDEX_LABEL} anchor yielded only "
                f"{len(symbols)} symbols ({self.ANCHOR_URL}), fewer than the "
                f"minimum {self.MIN_ANCHOR_ROWS} -- the anchor source is a "
                f"commercial scraped page whose markup has drifted. Refusing "
                f"to continue: a truncated anchor closes every unmentioned "
                f"membership and silently reintroduces survivorship bias."
            )

        return pl.DataFrame(
            {
                "symbol": symbols,
                "date_added": [None] * len(symbols),
            },
            schema={"symbol": pl.String, "date_added": pl.String},
        )


class UniverseCatalog:
    """Point-in-time US-equity universe reference table.

    Combines every exchange roster in `ROSTER_FETCHERS` with every index
    membership fetcher in `MEMBERSHIP_FETCHERS` into one
    `(symbol, category, start_date, end_date)` table, persisted via
    `PlBackend`/parquet (Locked Decision A1, 02-08-PLAN.md).

    IMPORTANT for future backtest phases: `get_symbols_as_of()` must be
    called per-rebalance-date in a walk-forward backtest, not once at setup
    time, to avoid look-ahead bias (02-08-RESEARCH.md Open Question 2).
    """

    #: The index-membership fetchers this catalog carries. **Membership of
    #: this tuple is what gives a category its point-in-time coverage
    #: boundary**: `build()` derives each category token from `cls.CATEGORY`
    #: and `get_symbols_as_of()` derives each boundary from
    #: `cls.PIT_COVERAGE_START`, so a fourth index inherits the guard by
    #: being registered here rather than by someone remembering to add an
    #: `if` branch. The previous hardcoded single-category guard meant any
    #: second index silently answered pre-coverage queries with an
    #: incomplete roster -- precisely the failure DATA-05 exists to prevent
    #: (RESEARCH Finding 6 bullet 4).
    #:
    #: Both roster fetchers are deliberately ABSENT: they are full-exchange
    #: rosters with no membership-interval semantics and no coverage start,
    #: and D-02 locks `nasdaq_all` semantics exactly as they are. Adding
    #: either here would impose a boundary it must not have.
    MEMBERSHIP_FETCHERS: tuple[type[IndexMembershipFetcher], ...] = (
        SP500MembershipFetcher,
        Nasdaq100MembershipFetcher,
    )

    #: The exchange-roster fetchers this catalog carries, looped in `build()`
    #: exactly the way `MEMBERSHIP_FETCHERS` is, with each category token
    #: derived from `cls.CATEGORY`. Adding a roster is a registration, not an
    #: `if` branch.
    #:
    #: **Kept in its OWN registry, separate from `MEMBERSHIP_FETCHERS`, on
    #: purpose.** A roster has per-symbol listing dates but no index-membership
    #: concept and therefore no `PIT_COVERAGE_START`; registering one in
    #: `MEMBERSHIP_FETCHERS` would impose a point-in-time coverage boundary
    #: neither roster may have, making pre-boundary queries raise instead of
    #: answering correctly from the roster's own dates (D-02).
    #:
    #: The two roster fetchers each download `supported_tickers.zip`
    #: independently. That is deliberate: two 794 KB downloads per build cost
    #: a couple of seconds, whereas a shared cache would either leak mocked
    #: bytes across tests or change `NasdaqUniverseFetcher.fetch()`'s
    #: observable behaviour -- which D-02 forbids.
    ROSTER_FETCHERS: tuple[type[TiingoRosterFetcher], ...] = (
        NasdaqUniverseFetcher,
        USEquityUniverseFetcher,
    )

    def __init__(self, config: UniverseConfig):
        self.config = config
        self._backend = PlBackend()

    def build(self, allow_stale: bool = False) -> Self:
        """Fetch every category and stage the combined reference table.

        `allow_stale` must be set explicitly to build from a fetcher that fell
        back to its cached snapshot. The fallback itself is correct and
        deliberate -- one bad parse must not poison every future run -- but
        `save()` would otherwise persist that stale reconstruction into
        `universe.parquet` INDISTINGUISHABLY from a fresh one, so a
        permanently-broken source silently freezes the universe at the cache
        date with only a `logger.error` line, easily lost in a cron log, to
        record it. Refusing by default makes the degradation a decision.
        """
        frames = [
            roster_cls()
            .fetch()
            .with_columns(
                pl.lit(roster_cls.CATEGORY).alias("category"),
                # Tiingo reports real listing/delisting dates, so no end here
                # is ever inferred. Stated explicitly rather than left null so
                # the column means the same thing in every category.
                pl.lit(False).alias("end_date_is_inferred"),
            )
            # `vertical_relaxed` concat matches on column ORDER, not name, so
            # every frame is projected onto the canonical order before it is
            # appended -- otherwise adding a column to one producer silently
            # transposes values into the wrong columns of another.
            .select(self.CATALOG_COLUMNS)
            for roster_cls in self.ROSTER_FETCHERS
        ]
        stale: list[str] = []
        for fetcher_cls in self.MEMBERSHIP_FETCHERS:
            fetcher = fetcher_cls(cache_dir=self.config.cache_dir)
            frames.append(
                fetcher.build_intervals()
                .with_columns(pl.lit(fetcher_cls.CATEGORY).alias("category"))
                .select(self.CATALOG_COLUMNS)
            )
            if fetcher.changes_are_stale:
                stale.append(
                    f"{fetcher_cls.CATEGORY} (cached snapshot from "
                    f"{fetcher.changes_source_asof})"
                )

        if stale and not allow_stale:
            raise ValueError(
                f"Refusing to build the universe table from stale cached "
                f"snapshots: {stale}. The live change-log fetch/parse failed "
                f"for those categories, so persisting this would freeze the "
                f"universe at the cache date while looking exactly like a "
                f"fresh build. Fix the source, or pass allow_stale=True to "
                f"accept a knowingly-frozen table."
            )
        if stale:
            logger.warning(
                f"Building the universe table from STALE cached snapshots "
                f"(allow_stale=True): {stale}."
            )

        combined = pl.concat(frames, how="vertical_relaxed")
        self._backend.to_internal(combined.lazy())
        return self

    def _assert_every_category_is_populated(self) -> None:
        """Refuse to persist a table missing any category's rows.

        `save()` overwrites `universe.parquet` in place (that is what
        `refresh_us_equity_universe.py` calls), so a category that silently
        came back empty would DESTROY the previous good roster and the only
        symptom would be downstream ingestion quietly doing nothing. Checked
        against the registry so a fourth index is covered by registration.
        """
        counts = (
            self._backend.get_lazyframe()
            .group_by("category")
            .agg(pl.len().alias("row_count"))
            .collect()
        )
        populated = {
            category
            for category, row_count in zip(
                counts["category"].to_list(), counts["row_count"].to_list()
            )
            if row_count > 0
        }
        missing = sorted(self.known_categories() - populated)
        if missing:
            raise ValueError(
                f"Refusing to write {self.config.output_path}: categories "
                f"{missing} have no rows. save() overwrites the reference "
                f"table in place, so persisting this would destroy the "
                f"previous good roster and leave downstream ingestion "
                f"silently resolving an empty symbol list."
            )

    def save(self) -> Self:
        self._assert_every_category_is_populated()
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        self._backend.write(self.config.output_path)
        return self

    @classmethod
    def load(cls, config: UniverseConfig) -> "UniverseCatalog":
        catalog = cls(config)
        catalog._backend.read(config.output_path)
        return catalog

    #: Canonical column order of the persisted reference table. `end_date_is_
    #: inferred` marks a FABRICATED interval end (see
    #: `IndexMembershipFetcher.reconstruct_intervals`) so a consumer can tell
    #: it from an observed one instead of trusting a log line.
    CATALOG_COLUMNS = (
        "symbol",
        "category",
        "start_date",
        "end_date",
        "end_date_is_inferred",
    )

    def known_categories(self) -> set[str]:
        """Every category this catalog can answer for.

        The UNION of both registries, so a category cannot exist without a
        fetcher and a fetcher cannot be registered without becoming a known
        category. `_assert_every_category_is_populated()` derives its check
        from this, which is what makes a newly-registered roster covered by
        registration rather than by someone remembering to add a guard.
        """
        return {roster_cls.CATEGORY for roster_cls in self.ROSTER_FETCHERS} | {
            fetcher_cls.CATEGORY for fetcher_cls in self.MEMBERSHIP_FETCHERS
        }

    def _validate_category(self, category: str) -> None:
        """Reject an unknown category token.

        Shared by both query methods because every wrong input otherwise
        produces `[]`, which is a LEGITIMATE return value (a pre-listing
        roster query returns it), so the caller cannot distinguish "no
        members" from "you asked wrong" -- a typo'd category would silently
        ingest nothing instead of the requested universe.
        """
        known = self.known_categories()
        if category not in known:
            raise ValueError(
                f"Unknown universe category {category!r}; known categories "
                f"are {sorted(known)}."
            )

    @staticmethod
    def _validate_iso_date(value: str, field: str) -> None:
        """Reject a non-ISO date string.

        The table stores ISO date strings and compares them
        LEXICOGRAPHICALLY, so a non-ISO value does not merely fail to match --
        it compares wrong and returns a plausible, silently incorrect roster.
        """
        try:
            datetime.date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be an ISO YYYY-MM-DD string, got {value!r}. "
                f"The table stores ISO date strings and compares them "
                f"LEXICOGRAPHICALLY, so a non-ISO value does not merely fail "
                f"to match -- it compares wrong and returns a plausible, "
                f"silently incorrect roster."
            ) from exc

    def get_symbols_in_range(
        self, category: str, start_date: str, end_date: str
    ) -> list[str]:
        """Every symbol whose listing interval OVERLAPS `[start_date,
        end_date]`, de-duplicated.

        This is the query a full-window BACKFILL wants;
        `get_symbols_as_of()` is the query a walk-forward backtest wants,
        per-rebalance. The overlap predicate is
        `start_date <= end AND (end_date IS NULL OR end_date >= start)`.

        **This is the D-05 filter's home.** Dropping tickers whose Tiingo
        `endDate` precedes the window start falls out of interval overlap, so
        no `MIN_END_DATE` constant is baked into any fetcher: baking a window
        into the reference table would make that table unusable for any other
        window and would break the config-driven reproducibility constraint
        (CLAUDE.md). Equally important is what overlap KEEPS -- every ticker
        that delisted INSIDE the window. Roughly 6.9k of the 15.4k US-equity
        tickers ended before today; excluding them is precisely the
        survivorship bias this layer exists to remove.

        Both dates and the category are validated for the same reason
        `get_symbols_as_of()` validates them: `[]` is a legitimate answer, so
        a typo must raise rather than silently ingest nothing.
        """
        self._validate_category(category)
        self._validate_iso_date(start_date, "start_date")
        self._validate_iso_date(end_date, "end_date")

        matched = self._backend.get_lazyframe().filter(
            (pl.col("category") == category)
            & (pl.col("start_date") <= end_date)
            # Null-tolerant even though Tiingo always populates `endDate` in
            # this subset (currently-listed names carry the last trading day):
            # the membership categories DO produce real nulls for open
            # intervals, and this query answers for those too.
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= start_date))
        )
        return matched.select("symbol").unique().collect()["symbol"].to_list()

    def get_symbols_as_of(self, category: str, as_of_date: str) -> list[str]:
        # `category` and `as_of_date` arrive unvalidated -- `as_of_date` comes
        # straight off `ingest_tiingo.py`'s `--as-of-date` CLI argument. Both
        # are validated HERE because every wrong input otherwise produces `[]`,
        # which is a LEGITIMATE return value (a pre-listing `nasdaq_all` query
        # returns it), so the caller cannot distinguish "no members" from "you
        # asked wrong". A typo'd category or a non-ISO date would silently
        # ingest nothing instead of the requested index -- the same class of
        # silent-wrong-answer the coverage-start guard below raises to prevent.
        self._validate_category(category)
        self._validate_iso_date(as_of_date, "as_of_date")

        # Every registered membership category carries its own boundary; a
        # category absent from the map (i.e. either exchange roster,
        # `nasdaq_all` and `us_all`) is boundary-free by design (D-02).
        coverage_starts = {
            fetcher_cls.CATEGORY: fetcher_cls.PIT_COVERAGE_START
            for fetcher_cls in self.MEMBERSHIP_FETCHERS
        }
        coverage_start = coverage_starts.get(category)
        if coverage_start is not None and as_of_date < coverage_start:
            raise ValueError(
                f"Cannot answer {category} membership before "
                f"{coverage_start} -- the "
                f"Wikipedia-sourced change log is left-censored at that "
                f"date and this query cannot be answered correctly, rather "
                f"than silently defaulting to an incomplete/wrong answer."
            )

        matched = self._backend.get_lazyframe().filter(
            (pl.col("category") == category)
            & (pl.col("start_date") <= as_of_date)
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= as_of_date))
        )
        return matched.select("symbol").unique().collect()["symbol"].to_list()
