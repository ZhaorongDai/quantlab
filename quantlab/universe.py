"""Point-in-time, survivorship-bias-free US equity symbol universe.

This module builds and queries the reference table that tells the rest of
the pipeline which symbols existed, and which index they belonged to, on any
given date. Two kinds of fetcher produce its rows. ``TiingoRosterFetcher``
subclasses (``NasdaqUniverseFetcher``, ``USEquityUniverseFetcher``) download
exchange-wide common-stock rosters, delisted names included, from Tiingo's
ticker directory. ``IndexMembershipFetcher`` subclasses
(``SP500MembershipFetcher``, ``Nasdaq100MembershipFetcher``) rebuild index
membership intervals from a current-constituent snapshot plus Wikipedia's
historical change log. ``UniverseCatalog`` merges them into one
``(symbol, category, start_date, end_date, end_date_is_inferred)`` parquet
table and answers point-in-time queries against it.

``UniverseCatalog`` also hosts the acquisition volume guard, which prices a
download and refuses one that would exceed the disk, request or wall-clock
ceiling. This module must therefore never import an acquisition client, by
any spelling, and the package ``__init__`` on its import path stays empty:
the guard has to refuse an over-budget download before any client can exist,
and keeping every client out of this module's import graph makes that true
whatever order callers use.

See ``docs/constituent.md`` for the guide.
"""

import datetime
import io
import math
import os
import re
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import pandas as pd
import polars as pl
import requests
from loguru import logger

from quantlab.base.config import UniverseConfig
from quantlab.backend import PlBackend
from quantlab.enums.data import TRADEABLE_TICKER_PATTERN, UniverseCategory

#: Contact string sent in the ``User-Agent`` header when scraping Wikipedia,
#: whose bot policy asks for one. Read from ``QUANTLAB_CONTACT`` so that no
#: personal address is committed to source; the default is a neutral
#: project URL.
_CONTACT = os.environ.get(
    "QUANTLAB_CONTACT", "https://github.com/quantlab/quantlab"
)


#: Regex matching a preferred-share ticker in every notation Tiingo's
#: directory uses: a ``-`` or ``/`` delimiter, optional whitespace, a ``P``,
#: an optional single letter (the series letter in ``BC/PA`` or the ``R`` of
#: ``PR``), optional whitespace, then another delimiter or the end of the
#: ticker. Requiring the ``P`` right after the delimiter is what keeps
#: hyphenated class shares such as ``BRK-B`` and ``BF-A``, which are common
#: stock, out of the match.
_PREFERRED_SHARE_PATTERN = r"[-/]\s*P[A-Z]?\s*(?:[-/]|$)"

#: Regex matching a baby bond or note, whose ticker embeds a coupon and
#: sometimes a maturity (``NEE 6.219``, ``ASRV 8.45 06-30-28``). A space
#: followed by a digit is the tell; no common stock ticker contains one.
_BABY_BOND_PATTERN = r"\s\d"


class TiingoRosterFetcher:
    """Base class for exchange-scoped common-stock rosters built from Tiingo.

    Downloads Tiingo's ``supported_tickers.csv`` directory, which lists every
    ticker the vendor has ever carried together with its listing and delisting
    dates, and filters it down to one roster. Using the full directory rather
    than a currently-listed feed is what keeps delisted names in the roster and
    the resulting universe free of survivorship bias.

    A subclass is data, not code: it sets ``EXCHANGE_FILTER``,
    ``MIN_ROSTER_ROWS`` and ``CATEGORY`` (and optionally
    ``EXCLUDE_NON_COMMON_SECURITY_TYPES``) and inherits ``fetch`` unchanged.

    ``fetch`` filters on exact string tokens, so a vocabulary change in the
    vendor's feed would yield zero rows without raising. ``MIN_ROSTER_ROWS``
    turns that silent truncation into an error before the catalog can overwrite
    a good reference table with an empty one.

    Example:
        >>> class NyseUniverseFetcher(TiingoRosterFetcher):
        ...     EXCHANGE_FILTER = ("NYSE",)
        ...     MIN_ROSTER_ROWS = 1000
        ...     CATEGORY = "us_all"
        >>> roster = NyseUniverseFetcher().fetch()  # downloads from Tiingo
        >>> roster.columns
        ['symbol', 'start_date', 'end_date']
    """

    SOURCE_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
    ASSET_TYPE = "Stock"
    PRICE_CURRENCY = "USD"

    #: Exact ``exchange`` tokens kept by ``fetch``.
    EXCHANGE_FILTER: tuple[str, ...]
    #: Minimum row count ``fetch`` accepts; fewer rows means the vendor's
    #: token vocabulary drifted and the roster is refused.
    MIN_ROSTER_ROWS: int
    #: The catalog category this roster is stored under. Typed as the
    #: literal so an unknown token is a type error rather than a runtime
    #: surprise.
    CATEGORY: UniverseCategory
    #: When true, ``fetch`` also drops preferred shares and baby bonds (see
    #: ``_PREFERRED_SHARE_PATTERN`` and ``_BABY_BOND_PATTERN``). Off by
    #: default so a roster's contents never change unless its subclass
    #: opts in.
    EXCLUDE_NON_COMMON_SECURITY_TYPES: bool = False

    def fetch(self) -> pl.DataFrame:
        """Download Tiingo's ticker directory and return this roster as a frame.

        The directory is filtered to ``EXCHANGE_FILTER``, ``ASSET_TYPE`` and
        ``PRICE_CURRENCY``; preferred shares and baby bonds are dropped when
        ``EXCLUDE_NON_COMMON_SECURITY_TYPES`` is set; and tickers that do not match
        ``TRADEABLE_TICKER_PATTERN`` are always dropped, because the acquisition
        layer would refuse to fetch them. The row-count floor is checked last, on
        the rows that will actually be persisted.

        Returns:
            A frame with columns ``symbol``, ``start_date`` and ``end_date``, one
            row per exchange listing (a ticker that moved venue has two rows).

        Raises:
            ValueError: If fewer than ``MIN_ROSTER_ROWS`` rows survive the filter.

        Example:
            >>> roster = USEquityUniverseFetcher().fetch()  # downloads from Tiingo
            >>> roster.columns
            ['symbol', 'start_date', 'end_date']
        """
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
        if self.EXCLUDE_NON_COMMON_SECURITY_TYPES:
            # Applied to the source's own `ticker` column, before the rename
            # below, and before the row-count floor so the floor checks the
            # rows that are actually persisted.
            before = len(data)
            data = data.filter(
                ~pl.col("ticker").str.contains(_PREFERRED_SHARE_PATTERN)
                & ~pl.col("ticker").str.contains(_BABY_BOND_PATTERN)
            )
            logger.info(
                f"{self.CATEGORY}: excluded {before - len(data)} preferred / "
                f"baby-bond rows ({before} -> {len(data)})."
            )

        # Malformed tickers are dropped for every roster, not only those that
        # opt into the exclusion above: the acquisition layer refuses such a
        # symbol before issuing a request, so persisting one would abort a
        # whole-roster download. This also runs before the row-count floor.
        # `str.contains` is a search, so the pattern's own anchors are what
        # make it a whole-string match.
        before = len(data)
        data = data.filter(
            pl.col("ticker").str.contains(
                TRADEABLE_TICKER_PATTERN.pattern, literal=False
            )
        )
        logger.info(
            f"{self.CATEGORY}: dropped {before - len(data)} malformed / "
            f"unfetchable ticker rows ({before} -> {len(data)})."
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
    """Roster of every NASDAQ-listed common stock, delisted names included.

    Preferred shares and baby bonds listed on NASDAQ stay in this roster; only
    ``USEquityUniverseFetcher`` opts into dropping them.

    Example:
        >>> roster = NasdaqUniverseFetcher().fetch()  # downloads from Tiingo
        >>> roster.columns
        ['symbol', 'start_date', 'end_date']
    """

    # NASDAQ-listed only: no OTC or expert-market tiers.
    EXCHANGE_FILTER = ("NASDAQ",)

    # Far below the roughly ten thousand rows the real roster has, so a
    # genuine shrink never trips it while a zeroed filter always does.
    MIN_ROSTER_ROWS = 1000

    CATEGORY: UniverseCategory = "nasdaq_all"


class USEquityUniverseFetcher(TiingoRosterFetcher):
    """Roster of US common stock listed on NYSE, NASDAQ and AMEX, priced in USD.

    Delisted names are included. The AMEX appears in Tiingo's directory under
    both ``AMEX`` and ``NYSE MKT``, because historical rows were never
    relabelled across the exchange's renames, so both tokens are kept;
    ``NYSE ARCA``, ``NYSE NAT`` and ``BATS`` are different exchanges and are
    excluded. Several hundred tickers carry more than one exchange row, which
    is why the catalog's interval queries de-duplicate on symbol.

    This roster opts into ``EXCLUDE_NON_COMMON_SECURITY_TYPES``, so preferred
    shares and baby bonds are dropped while hyphenated class shares such as
    ``BRK-B``, and warrants, units and rights, are kept. It is therefore not a
    strict superset of ``NasdaqUniverseFetcher``'s roster: a NASDAQ-listed
    preferred share appears there and not here, by design.

    Example:
        >>> roster = USEquityUniverseFetcher().fetch()  # downloads from Tiingo
        >>> roster.columns
        ['symbol', 'start_date', 'end_date']
    """

    EXCHANGE_FILTER = ("NASDAQ", "NYSE", "AMEX", "NYSE MKT")

    EXCLUDE_NON_COMMON_SECURITY_TYPES = True

    # About half the observed row count: a market contraction never trips
    # it, a zeroed filter always does.
    MIN_ROSTER_ROWS = 8000

    CATEGORY: UniverseCategory = "us_all"


#: Cell values meaning "no ticker on this side of the change row". Listed
#: explicitly because ``pd.read_html`` is told not to infer missing values:
#: its default vocabulary overlaps the ticker namespace (``NA`` is a real US
#: ticker). The dashes are the glyphs Wikipedia uses for "none".
_BLANK_TICKER_CELLS = frozenset({"", "-", "–", "—"})

#: Shape a change-log ticker cell must have after
#: ``IndexMembershipFetcher._normalize_ticker_cell``: up to seven upper-case
#: letters or digits, optionally followed by one ``.`` or ``-`` delimited
#: class suffix (``BRK.B``, ``BRK-B``). Deliberately narrower than
#: ``TRADEABLE_TICKER_PATTERN``, which admits two suffix segments for the
#: warrants in Tiingo's directory: here an interior delimiter means two
#: HTML cells were merged by a parser regression, which is exactly what
#: this validator exists to catch.
_WELL_FORMED_TICKER = re.compile(r"^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2})?$")


class IndexMembershipFetcher(ABC):
    """Base class for reconstructing point-in-time index membership intervals.

    An index is described by data: a subclass sets the class constants below
    and implements ``fetch_anchor``, and inherits the change-log parse, the
    reconstruction algorithm and the caching behaviour of ``fetch_changes``.

    Two sources are combined. The anchor is a snapshot of the current
    constituents and is authoritative for who is a member today. The change
    log is Wikipedia's dated table of additions and removals and is
    authoritative for when membership changed. ``reconstruct_intervals``
    replays the log forward and reconciles it against the anchor.

    A subclass sets ``ANCHOR_URL``, ``CHANGES_URL``, ``PIT_COVERAGE_START``
    (the earliest date the change log covers; queries before it are refused),
    ``CACHE_FILENAME`` (the per-index change-log snapshot under ``cache_dir``),
    ``INDEX_LABEL``, ``CATEGORY``, ``EXPECTED_SOURCE_HEADER`` (the flattened
    header that identifies the change-log table) and ``DATE_HEADER``;
    ``CHANGES_TABLE_ATTRS`` is optional.

    Three safety properties live on this base so every index gets them. The
    change-log table is selected by matching its header against
    ``EXPECTED_SOURCE_HEADER`` and its columns are read by name, so a reordered
    header raises instead of silently swapping additions and removals. A live
    table with fewer rows than the cached snapshot is treated as a parse
    failure, because memberships only close and never vanish. On any fetch or
    parse failure the cached snapshot is returned and the cache file is left
    untouched.

    Example:
        >>> fetcher = SP500MembershipFetcher(cache_dir="data/reference/_cache")
        >>> intervals = fetcher.build_intervals()  # downloads anchor and change log
        >>> intervals.columns
        ['symbol', 'start_date', 'end_date', 'end_date_is_inferred']
    """

    ANCHOR_URL: str
    CHANGES_URL: str
    PIT_COVERAGE_START: str
    CACHE_FILENAME: str
    INDEX_LABEL: str
    #: The catalog category this index is stored under. Typed as the
    #: literal so an unknown token is a type error rather than a runtime
    #: surprise.
    CATEGORY: UniverseCategory

    #: Exact flattened header (see ``_flatten_header``) of the change-log
    #: table. It both selects the table and names the columns read from it.
    EXPECTED_SOURCE_HEADER: tuple[str, ...]
    #: The ``EXPECTED_SOURCE_HEADER`` entry carrying the effective date.
    DATE_HEADER: str
    #: The two ticker columns consumed. Both live sources word them
    #: identically; they are still validated against
    #: ``EXPECTED_SOURCE_HEADER``.
    ADDED_TICKER_HEADER: str = "Added Ticker"
    REMOVED_TICKER_HEADER: str = "Removed Ticker"
    #: Optional ``pd.read_html(attrs=...)`` pre-filter for a page that marks
    #: its change-log table with an id or class.
    CHANGES_TABLE_ATTRS: dict[str, str] | None = None

    def __init__(self, cache_dir: str):
        """Set the cache path under ``cache_dir`` and clear the staleness flags."""
        self._cache_path = Path(cache_dir) / self.CACHE_FILENAME
        #: Whether the frame ``fetch_changes`` last returned came from the
        #: cached snapshot rather than a live fetch, and the date that frame
        #: is current to. ``UniverseCatalog.build`` reads both so a stale
        #: reconstruction is never persisted silently.
        self.changes_are_stale = False
        self.changes_source_asof: str | None = None

    @abstractmethod
    def fetch_anchor(self) -> pl.DataFrame:
        """Return the current constituents as a ``[symbol, date_added]`` frame.

        ``date_added`` may be entirely null when the source has no such column;
        ``reconstruct_intervals`` then falls back to ``PIT_COVERAGE_START`` for
        those symbols.

        Example:
            A subclass whose source is a CSV with ``Symbol`` and ``Date added``
            columns::

                def fetch_anchor(self) -> pl.DataFrame:
                    response = requests.get(self.ANCHOR_URL, timeout=30)
                    response.raise_for_status()
                    data = pl.read_csv(io.StringIO(response.text))
                    data = data.rename({"Symbol": "symbol", "Date added": "date_added"})
                    return data.select(["symbol", "date_added"])
        """

    @staticmethod
    def _normalize_ticker_cell(value: str) -> str:
        """Strip leading and trailing whitespace and ``|`` residue from one cell.

        The observed upstream typo is a trailing ``|`` left by a wikitable editor
        (``"ALLE |"``). Only leading and trailing delimiters are removed; an
        interior one most likely means two cells were merged and is left in place
        so that ``_WELL_FORMED_TICKER`` rejects it. A cell that is nothing but
        residue becomes the empty string, which reads as "no ticker".
        """
        return value.strip().strip("|").strip()

    @staticmethod
    def _flatten_header(columns) -> tuple[str, ...]:
        """Flatten a possibly two-level ``pd.read_html`` header to plain strings.

        ``("Added", "Ticker")`` becomes ``"Added Ticker"`` and a label repeated
        across both header rows (``("Reason", "Reason")``) becomes ``"Reason"``.
        Whitespace is collapsed so a stray non-breaking space or line break in the
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
        """Parse the change-log HTML into a three-column pandas frame.

        The table whose flattened header equals ``EXPECTED_SOURCE_HEADER`` is
        selected and its date and ticker columns are read by name. Ticker cells are
        normalized, blank cells become ``None``, and anything left that is not a
        well-formed ticker raises. Dates are parsed and re-rendered as ISO strings.

        Args:
            html_text: The page body of ``CHANGES_URL``.

        Returns:
            A frame with columns ``effective_date``, ``added_ticker`` and
            ``removed_ticker``; the ticker columns hold ``None`` where a row has no
            ticker on that side.

        Raises:
            ValueError: If the header constants contradict each other, no table
                carries the expected header, a ticker cell is malformed after
                normalization, or a date cell cannot be parsed.
        """
        # Do not let pandas infer missing values: its default vocabulary
        # (`NA`, `N/A`, `-`, ...) overlaps the ticker namespace, and `NA` is
        # a real ticker. Blank cells are recovered explicitly below.
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
        # Blank cells become a real `None`, which `reconstruct_intervals`
        # reads as "no change on this side"; every other cell, including
        # the ticker `NA`, survives as itself.
        malformed: list[tuple[str, str]] = []
        for column in ("added_ticker", "removed_ticker"):
            stripped = parsed[column].astype(str).str.strip().tolist()

            # Normalize before anything else: Wikipedia's change logs carry
            # delimiter typos such as `ALLE |`, and passing them through would
            # create phantom symbols matching no market data.
            normalized = [self._normalize_ticker_cell(v) for v in stripped]

            # Every corrected cell is logged, one line per column, so a parser
            # regression shows up as one long list instead of being absorbed.
            corrections = [
                (raw, clean)
                for raw, clean in zip(stripped, normalized)
                if raw != clean
            ]
            if corrections:
                logger.warning(
                    f"{self.INDEX_LABEL}: normalized {len(corrections)} "
                    f"{column} cell(s) carrying delimiter residue from "
                    f"{self.CHANGES_URL}: "
                    + ", ".join(f"{raw!r} -> {clean!r}" for raw, clean in corrections)
                )

            # The blank test runs on the normalized value: a cell that was only
            # residue must become `None`, not reach the validator below.
            # `dtype=object` (further down) keeps the sentinel a real `None`;
            # a plain list assignment would let pandas turn it back into `nan`.
            cleaned = [
                None if value in _BLANK_TICKER_CELLS else value
                for value in normalized
            ]

            # Malformed cells are collected across both columns so one run
            # reports all of them.
            malformed.extend(
                (column, value)
                for value in cleaned
                if value is not None and not _WELL_FORMED_TICKER.match(value)
            )

            parsed[column] = pd.Series(cleaned, dtype=object, index=parsed.index)

        if malformed:
            # `fetch_changes` catches this and falls back to the cached snapshot
            # without overwriting it, so raising here is loud but harmless.
            raise ValueError(
                f"{self.INDEX_LABEL}: change-log ticker cells at "
                f"{self.CHANGES_URL} are still malformed after delimiter "
                f"normalization: {malformed}. Refusing to reconstruct "
                f"membership from cells that would enter the panel as "
                f"phantom symbols matching no market data. An INTERIOR "
                f"delimiter is not the observed upstream typo -- it means two "
                f"cells were merged, i.e. a parser regression."
            )

        # `format="mixed"` avoids the format-inference warning both pages
        # provoke, and `errors="coerce"` turns an unparseable cell into NaT so
        # the offending rows can be named below instead of a bare parse error.
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
        """Fetch the change log, validate it and update the cached snapshot.

        On success the parsed table is written atomically to ``CACHE_FILENAME``
        under the cache directory. On any fetch or parse failure, or when the live
        table has fewer rows than the cached one, the cached snapshot is returned
        instead, the cache file is left untouched, and ``changes_are_stale`` is set
        so the catalog can refuse to persist the result.

        Returns:
            A frame with columns ``effective_date``, ``added_ticker`` and
            ``removed_ticker``.

        Raises:
            RuntimeError: If the live fetch failed and no cached snapshot exists.

        Example:
            >>> fetcher = SP500MembershipFetcher(cache_dir="data/reference/_cache")
            >>> changes = fetcher.fetch_changes()  # downloads from Wikipedia
            >>> changes.columns
            ['effective_date', 'added_ticker', 'removed_ticker']
            >>> fetcher.changes_are_stale
            False
        """
        cached_row_count = 0
        if self._cache_path.exists():
            try:
                cached_row_count = len(pl.read_parquet(self._cache_path))
            except Exception as exc:
                # A corrupt cache must not block a fresh fetch, but falling back
                # to 0 disables the row-count guard below for this run, so say
                # so loudly.
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

            # Backstop for a subclass that overrides the base parse and returns
            # the wrong shape; the source header itself is validated inside
            # `_parse_changes_table`.
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

        # Write atomically: an in-place write interrupted midway would leave a
        # truncated parquet, the corrupt-cache state that disables the
        # row-count guard above.
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._cache_path.with_suffix(".parquet.tmp")
        parsed.write_parquet(tmp_path)
        tmp_path.replace(self._cache_path)
        self.changes_are_stale = False
        self.changes_source_asof = datetime.date.today().isoformat()
        return parsed

    def _load_cache(self) -> pl.DataFrame:
        """Return the cached change-log snapshot, or raise if there is none."""
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

        The anchor is authoritative for who is a member today; the change log is
        authoritative for when membership changed. Three disagreements between
        them are reconciled, each with a warning:

        1. An interval still open at the end of the log whose symbol is absent
           from the anchor is closed at the last date the log covers and flagged
           ``end_date_is_inferred=True``, because the log never says when it
           ended.
        2. A symbol whose last event is a removal but which the anchor still lists
           is re-opened from that removal date; the log is missing a re-addition.
        3. An anchor member with no event in the log is opened at its
           ``date_added``, or at ``PIT_COVERAGE_START`` when that is null.

        A removal with no prior addition opens at ``PIT_COVERAGE_START``, and a
        duplicate addition keeps the earlier open date.

        Args:
            anchor: A ``[symbol, date_added]`` frame as returned by
                ``fetch_anchor``.
            changes: A frame as returned by ``fetch_changes``, with ``None`` where
                a row has no ticker on one side.

        Returns:
            A frame with columns ``symbol``, ``start_date``, ``end_date`` (null for
            a current member) and ``end_date_is_inferred``.

        Example:
            >>> fetcher = SP500MembershipFetcher(cache_dir="/tmp/cache")
            >>> anchor = pl.DataFrame(
            ...     {"symbol": ["AAA", "BBB"], "date_added": ["1999-01-01", None]}
            ... )
            >>> changes = pl.DataFrame(
            ...     {
            ...         "effective_date": ["2010-05-03", "2015-09-21"],
            ...         "added_ticker": ["BBB", "DDD"],
            ...         "removed_ticker": ["ZZZ", None],
            ...     }
            ... )
            >>> fetcher.reconstruct_intervals(anchor, changes).sort("symbol")
            shape: (4, 4)
            ┌────────┬────────────┬────────────┬──────────────────────┐
            │ symbol ┆ start_date ┆ end_date   ┆ end_date_is_inferred │
            │ ---    ┆ ---        ┆ ---        ┆ ---                  │
            │ str    ┆ str        ┆ str        ┆ bool                 │
            ╞════════╪════════════╪════════════╪══════════════════════╡
            │ AAA    ┆ 1999-01-01 ┆ null       ┆ false                │
            │ BBB    ┆ 2010-05-03 ┆ null       ┆ false                │
            │ DDD    ┆ 2015-09-21 ┆ 2015-09-21 ┆ true                 │
            │ ZZZ    ┆ 1976-07-01 ┆ 2010-05-03 ┆ false                │
            └────────┴────────────┴────────────┴──────────────────────┘
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

        # Case 2: the log's last event for an anchor member is a removal, so
        # the log is missing a re-addition. Re-open from that removal date;
        # otherwise the symbol would be persisted as a former member.
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

        # Case 3: anchor members with no event in the log are original
        # constituents or were added before PIT_COVERAGE_START.
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
        """Fetch the anchor and change log and return the reconstructed intervals.

        Example:
            >>> fetcher = Nasdaq100MembershipFetcher(cache_dir="data/reference/_cache")
            >>> intervals = fetcher.build_intervals()  # downloads anchor and change log
            >>> intervals.columns
            ['symbol', 'start_date', 'end_date', 'end_date_is_inferred']
        """
        anchor = self.fetch_anchor()
        changes = self.fetch_changes()
        return self.reconstruct_intervals(anchor, changes)


class SP500MembershipFetcher(IndexMembershipFetcher):
    """Point-in-time S&P 500 membership from a GitHub CSV anchor and Wikipedia.

    The anchor is the ``datasets/s-and-p-500-companies`` CSV, which carries a
    ``Date added`` column. ``PIT_COVERAGE_START`` is the earliest row of the
    Wikipedia change table (1976-07-01), not the earlier coverage the page's
    prose claims; queries before it are refused.

    Example:
        >>> fetcher = SP500MembershipFetcher(cache_dir="data/reference/_cache")
        >>> intervals = fetcher.build_intervals()  # downloads anchor and change log
        >>> intervals.columns
        ['symbol', 'start_date', 'end_date', 'end_date_is_inferred']
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

    # Seven columns: this page also carries `Refs`. The `id="changes"`
    # marker narrows `read_html` before the header match selects the table.
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
        """Download the constituents CSV and return ``[symbol, date_added]``.

        Example:
            >>> fetcher = SP500MembershipFetcher(cache_dir="/tmp/cache")
            >>> anchor = fetcher.fetch_anchor()  # downloads from GitHub
            >>> anchor.columns
            ['symbol', 'date_added']
        """
        response = requests.get(self.ANCHOR_URL, timeout=30)
        response.raise_for_status()
        data = pl.read_csv(io.StringIO(response.text))
        data = data.rename({"Symbol": "symbol", "Date added": "date_added"})
        return data.select(["symbol", "date_added"])


class Nasdaq100MembershipFetcher(IndexMembershipFetcher):
    """Point-in-time Nasdaq-100 membership from a scraped anchor and Wikipedia.

    ``PIT_COVERAGE_START`` is the earliest row of the Wikipedia change table
    (2007-02-01), about three decades later than the S&P 500's, so the two
    categories must not be unioned onto one axis that implies coverage neither
    has.

    The anchor is scraped from a commercial listing page (``ANCHOR_URL``),
    because Wikipedia's Nasdaq-100 page renders its components through a
    template with no parseable table. It is the least stable input in this
    module, which is why ``fetch_anchor`` has a row-count floor of its own. The
    live anchor has 102 rows, not 100, because some issuers carry two share
    classes (``GOOGL``/``GOOG``, ``FOX``/``FOXA``). Neither source carries a
    ``date_added`` column, so the anchor's is all null and every member without
    a change-log event opens at ``PIT_COVERAGE_START``.

    Example:
        >>> fetcher = Nasdaq100MembershipFetcher(cache_dir="data/reference/_cache")
        >>> intervals = fetcher.build_intervals()  # downloads anchor and change log
        >>> intervals.columns
        ['symbol', 'start_date', 'end_date', 'end_date_is_inferred']
    """

    ANCHOR_URL = "https://stockanalysis.com/list/nasdaq-100-stocks/"
    CHANGES_URL = (
        "https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100"
    )
    PIT_COVERAGE_START = "2007-02-01"
    CACHE_FILENAME = "nasdaq100_changes_snapshot.parquet"
    INDEX_LABEL = "Nasdaq-100"
    CATEGORY = "nasdaq100_constituent"

    # Six columns after `pd.read_html` flattens the page's two-level header:
    # no `Refs`, and the date column is worded `Date`. The page marks its
    # change log with no id or class, so the header match alone selects it.
    EXPECTED_SOURCE_HEADER = (
        "Date",
        "Added Ticker",
        "Added Security",
        "Removed Ticker",
        "Removed Security",
        "Reason",
    )
    DATE_HEADER = "Date"

    # Far below the real 102 so an index resize never trips it, while a
    # drifted page that yields a handful of symbols always does.
    MIN_ANCHOR_ROWS = 50

    def fetch_anchor(self) -> pl.DataFrame:
        """Scrape the constituents table and return ``[symbol, date_added]``.

        The first table carrying a ``Symbol`` column is used and ``date_added`` is
        all null. Missing cells are dropped by nullness, never by comparing against
        the string ``"nan"``, so that pandas' missing-value vocabulary cannot
        swallow a real ticker.

        Raises:
            ValueError: If no table carries a ``Symbol`` column, or fewer than
                ``MIN_ANCHOR_ROWS`` symbols survive cleaning.

        Example:
            >>> fetcher = Nasdaq100MembershipFetcher(cache_dir="/tmp/cache")
            >>> anchor = fetcher.fetch_anchor()  # downloads the listing page
            >>> anchor.columns
            ['symbol', 'date_added']
        """
        response = requests.get(self.ANCHOR_URL, timeout=30)
        response.raise_for_status()

        # As in the change-log parse, pandas must not infer missing values:
        # `NA` is a real ticker, and a coerced NaN would otherwise enter the
        # anchor as the literal string "nan".
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

        # Ragged rows are padded with real NaN whatever `na_values` says. Drop
        # them by nullness, never by comparing against the string "nan",
        # which would also drop a ticker literally named `NAN`.
        raw = anchor["Symbol"]
        symbols = [
            symbol
            for symbol in raw[raw.notna()].astype(str).str.strip().tolist()
            if symbol
        ]

        # Counted after cleaning: a page rendering 102 rows of which three
        # carry a ticker is as drifted as one rendering three rows.
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
    """Point-in-time US equity universe reference table.

    Builds one ``(symbol, category, start_date, end_date, end_date_is_inferred)``
    table from every roster fetcher in ``ROSTER_FETCHERS`` and every index
    membership fetcher in ``MEMBERSHIP_FETCHERS``, persists it as parquet
    through ``PlBackend``, and answers point-in-time queries against it. Dates
    are ISO ``YYYY-MM-DD`` strings throughout and are compared
    lexicographically.

    A walk-forward backtest must call ``get_symbols_as_of`` on each rebalance
    date rather than once at setup, otherwise the roster carries look-ahead
    bias.

    The catalog also prices downloads: ``estimate_acquisition_volume`` and
    ``assert_acquisition_volume_fits`` work purely from the listing intervals,
    issue no vendor request and construct no client.

    Example:
        Build the table from the live sources and persist it::

            config = UniverseConfig(
                output_path="data/reference/universe.parquet",
                cache_dir="data/reference/_cache",
            )
            UniverseCatalog(config).build().save()  # downloads from Tiingo/Wikipedia

        Load a table already on disk and query it (here a three-row table
        written by hand):

        >>> pl.DataFrame({
        ...     "symbol": ["AAPL", "MSFT", "OLD1"],
        ...     "category": ["us_all"] * 3,
        ...     "start_date": ["1980-12-12", "1986-03-13", "1980-01-01"],
        ...     "end_date": [None, None, "1997-06-30"],
        ...     "end_date_is_inferred": [False] * 3,
        ... }).write_parquet(config.output_path)
        >>> catalog = UniverseCatalog.load(config)
        >>> catalog.get_symbols_as_of("us_all", "2020-01-01")
        ['AAPL', 'MSFT']
    """

    #: The index membership fetchers this catalog carries. Registration here
    #: is what gives a category its point-in-time coverage boundary:
    #: ``build`` takes the category token from ``cls.CATEGORY`` and the
    #: queries take the boundary from ``cls.PIT_COVERAGE_START``. Roster
    #: fetchers do not belong here; they have listing dates but no boundary.
    MEMBERSHIP_FETCHERS: tuple[type[IndexMembershipFetcher], ...] = (
        SP500MembershipFetcher,
        Nasdaq100MembershipFetcher,
    )

    #: The exchange roster fetchers this catalog carries, looped in ``build``
    #: the same way as ``MEMBERSHIP_FETCHERS``. Kept as a separate registry
    #: because a roster has no coverage boundary, so a pre-listing query
    #: answers from the roster's own dates instead of raising. Each fetcher
    #: downloads the ticker directory independently; that costs a couple of
    #: seconds per build.
    ROSTER_FETCHERS: tuple[type[TiingoRosterFetcher], ...] = (
        NasdaqUniverseFetcher,
        USEquityUniverseFetcher,
    )

    def __init__(self, config: UniverseConfig):
        """Bind ``config`` and an empty ``PlBackend``; nothing is read or fetched."""
        self.config = config
        self._backend = PlBackend()

    def build(self, allow_stale: bool = False) -> Self:
        """Fetch every category and stage the combined table in memory.

        Rosters come first, then index memberships; each frame is projected onto
        ``CATALOG_COLUMNS`` before concatenation. Nothing is written to disk until
        ``save``.

        Args:
            allow_stale: Whether to accept a membership fetcher that fell back to
                its cached change-log snapshot. Off by default, because a stale
                reconstruction persisted by ``save`` is indistinguishable from a
                fresh one and would silently freeze the universe at the cache date.

        Returns:
            ``self``, so ``build`` and ``save`` chain.

        Raises:
            ValueError: If a membership fetcher is stale and ``allow_stale`` is
                false.

        Example:
            >>> catalog = UniverseCatalog(config).build()  # downloads every source
            >>> catalog.save()
        """
        frames = [
            roster_cls()
            .fetch()
            .with_columns(
                pl.lit(roster_cls.CATEGORY).alias("category"),
                # Tiingo reports real listing and delisting dates, so no roster
                # end is inferred; stated explicitly so the column means the
                # same thing in every category.
                pl.lit(False).alias("end_date_is_inferred"),
            )
            # `vertical_relaxed` concat matches on column order, not name, so
            # every frame is projected onto the canonical order first.
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
        """Refuse to persist a table in which any known category has no rows.

        ``save`` overwrites the reference table in place, so a category that came
        back empty would destroy the previous good roster and the only symptom
        would be downstream ingestion quietly doing nothing.
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
        """Write the staged table to ``config.output_path``.

        Parent directories are created as needed.

        Returns:
            ``self``.

        Raises:
            ValueError: If any known category has no rows in the staged table.

        Example:
            >>> UniverseCatalog(config).build().save()  # downloads, then writes
        """
        self._assert_every_category_is_populated()
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        self._backend.write(self.config.output_path)
        return self

    @classmethod
    def load(cls, config: UniverseConfig) -> "UniverseCatalog":
        """Return a catalog whose table is read from ``config.output_path``.

        Args:
            config: The same config the table was built with; only
                ``output_path`` is read here.

        Example:
            >>> catalog = UniverseCatalog.load(config)
            >>> sorted(catalog.known_categories())
            ['nasdaq100_constituent', 'nasdaq_all', 'sp500_constituent', 'us_all']
        """
        catalog = cls(config)
        catalog._backend.read(config.output_path)
        return catalog

    #: Canonical column order of the persisted table. ``end_date_is_inferred``
    #: marks an interval end that ``reconstruct_intervals`` inferred rather
    #: than observed in a source.
    CATALOG_COLUMNS = (
        "symbol",
        "category",
        "start_date",
        "end_date",
        "end_date_is_inferred",
    )

    def known_categories(self) -> set[str]:
        """Return every category token this catalog can answer for.

        The union of both registries, so a category cannot exist without a fetcher
        and a registered fetcher is automatically a known category.

        Example:
            >>> sorted(catalog.known_categories())
            ['nasdaq100_constituent', 'nasdaq_all', 'sp500_constituent', 'us_all']
        """
        return {roster_cls.CATEGORY for roster_cls in self.ROSTER_FETCHERS} | {
            fetcher_cls.CATEGORY for fetcher_cls in self.MEMBERSHIP_FETCHERS
        }

    def _validate_category(self, category: str) -> None:
        """Raise ``ValueError`` for a category token that is not registered.

        An empty list is a legitimate query result, so a misspelt category must
        raise rather than silently select nothing.
        """
        known = self.known_categories()
        if category not in known:
            raise ValueError(
                f"Unknown universe category {category!r}; known categories "
                f"are {sorted(known)}."
            )

    @staticmethod
    def _normalize_iso_date(value: str, field: str) -> str:
        """Validate an ISO date string and return it as ``YYYY-MM-DD``.

        The table stores ISO date strings and compares them lexicographically, so
        a value in any other shape compares wrong rather than failing to match.
        ``date.fromisoformat`` also accepts the basic form (``"20070115"``) and
        week dates, which would compare wrong in the same way; callers must
        therefore use the returned string, not the argument they passed.

        Args:
            value: The date string to check.
            field: The argument name used in the error message.

        Raises:
            ValueError: If ``value`` is not an ISO date.
        """
        try:
            return datetime.date.fromisoformat(value).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be an ISO YYYY-MM-DD string, got {value!r}. "
                f"The table stores ISO date strings and compares them "
                f"LEXICOGRAPHICALLY, so a non-ISO value does not merely fail "
                f"to match -- it compares wrong and returns a plausible, "
                f"silently incorrect roster."
            ) from exc

    def _coverage_start(self, category: str) -> str | None:
        """Return ``category``'s point-in-time coverage start, or ``None``.

        Derived from ``MEMBERSHIP_FETCHERS``, so a registered index carries a
        boundary without any per-category code. Roster categories are not
        registered there and have no boundary: a pre-listing query on one answers
        from the roster's own dates.
        """
        return {
            fetcher_cls.CATEGORY: fetcher_cls.PIT_COVERAGE_START
            for fetcher_cls in self.MEMBERSHIP_FETCHERS
        }.get(category)

    def _assert_within_coverage(
        self, category: str, date: str, field: str
    ) -> None:
        """Raise if ``date`` precedes ``category``'s coverage start.

        Shared by both membership queries so they agree on the boundary. The
        comparison is a strict ``<``: a date equal to the coverage start is
        answerable and accepted. ``field`` names the rejected argument in the
        message.
        """
        coverage_start = self._coverage_start(category)
        if coverage_start is not None and date < coverage_start:
            raise ValueError(
                f"Cannot answer {category} membership before "
                f"{coverage_start} -- {field}={date!r} precedes it. The "
                f"Wikipedia-sourced change log is left-censored at that "
                f"date and this query cannot be answered correctly, rather "
                f"than silently defaulting to an incomplete/wrong answer."
            )

    def get_symbols_in_range(
        self, category: str, start_date: str, end_date: str
    ) -> list[str]:
        """Return every symbol whose interval overlaps ``[start_date, end_date]``.

        The overlap predicate is ``start_date <= end AND (end_date IS NULL OR
        end_date >= start)``, so a symbol that delisted inside the window is kept;
        excluding such symbols is the survivorship bias this table exists to
        remove. This is the query a full-window backfill wants;
        ``get_symbols_as_of`` is the one a walk-forward backtest wants at each
        rebalance.

        The result is de-duplicated and sorted ascending, and the order is part of
        the contract: callers slice it for ``--limit``, so it must depend only on
        the membership set and not on the parquet layout.

        Args:
            category: A token from ``known_categories``.
            start_date: ISO date, inclusive. For an index category it must not
                precede that index's coverage start; it is refused, not clamped,
                because clamping would return the same truncated roster silently.
            end_date: ISO date, inclusive.

        Returns:
            Sorted, de-duplicated symbols.

        Raises:
            ValueError: For an unknown category, a non-ISO date, or a
                ``start_date`` before the category's coverage start.

        Example:
            >>> catalog.get_symbols_in_range("us_all", "1995-01-01", "2000-12-31")
            ['AAPL', 'MSFT', 'OLD1']
        """
        self._validate_category(category)
        start_date = self._normalize_iso_date(start_date, "start_date")
        end_date = self._normalize_iso_date(end_date, "end_date")
        self._assert_within_coverage(category, start_date, "start_date")

        matched = self._backend.get_lazyframe().filter(
            (pl.col("category") == category)
            & (pl.col("start_date") <= end_date)
            # Null-tolerant: membership categories carry real nulls for open
            # intervals.
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= start_date))
        )
        # `.sort()` after `.unique()`: the order is part of the contract (see
        # the docstring). `unique(maintain_order=True)` would pin it to this
        # file's row order instead, which a refresh rewrites.
        return (
            matched.select("symbol")
            .unique()
            .sort("symbol")
            .collect()["symbol"]
            .to_list()
        )

    #: Trading days per calendar year and the calendar year they are scaled
    #: against. An approximation for sizing only; an exchange calendar would
    #: sharpen a row count by a couple of percent and change no decision.
    TRADING_DAYS_PER_YEAR = 252
    CALENDAR_DAYS_PER_YEAR = 365.25

    def _roster_window_profile(
        self,
        category: str,
        start_date: str,
        end_date: str,
        bars_per_day: int = 1,
    ) -> dict:
        """Return roster arithmetic for a window: symbols, timestamps and density.

        Everything is derived from this catalog's own intervals clipped to the
        window and reduced to one span per symbol, so a dual-listed ticker is
        counted once. ``density`` is the share of the dense ``symbols x
        timestamps`` grid that is a real observation; it is well below 1 for a
        full-market roster because most symbols are listed for only part of any
        long window.

        Args:
            category: A token from ``known_categories``.
            start_date: ISO date, inclusive.
            end_date: ISO date, inclusive.
            bars_per_day: Rows one symbol produces per trading day; 1 for daily
                bars, 390 for minute bars.

        Returns:
            A dict with ``symbols``, ``trading_days``, ``bars_per_day``,
            ``timestamps``, ``dense_cells``, ``observed_cells`` and ``density``.
            Counts are cells, never bytes.

        Raises:
            ValueError: For an unknown category, a non-ISO date or
                ``bars_per_day < 1``.
        """
        self._validate_category(category)
        start_date = self._normalize_iso_date(start_date, "start_date")
        end_date = self._normalize_iso_date(end_date, "end_date")

        window_days = (
            datetime.date.fromisoformat(end_date)
            - datetime.date.fromisoformat(start_date)
        ).days + 1
        trading_days = max(
            round(
                window_days
                * self.TRADING_DAYS_PER_YEAR
                / self.CALENDAR_DAYS_PER_YEAR
            ),
            1,
        )

        # Clip each interval to the window, then reduce to one span per
        # symbol so a dual-listed ticker is counted once.
        overlapping = self._backend.get_lazyframe().filter(
            (pl.col("category") == category)
            & (pl.col("start_date") <= end_date)
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= start_date))
        )
        spans = (
            overlapping.with_columns(
                pl.max_horizontal(
                    pl.col("start_date"), pl.lit(start_date)
                ).alias("clip_start"),
                pl.min_horizontal(
                    pl.col("end_date").fill_null(end_date), pl.lit(end_date)
                ).alias("clip_end"),
            )
            .group_by("symbol")
            .agg(
                pl.col("clip_start").min().alias("clip_start"),
                pl.col("clip_end").max().alias("clip_end"),
            )
            .with_columns(
                (
                    pl.col("clip_end").str.to_date()
                    - pl.col("clip_start").str.to_date()
                )
                .dt.total_days()
                .add(1)
                .alias("span_days")
            )
            .collect()
        )

        if bars_per_day < 1:
            raise ValueError(f"bars_per_day must be >= 1, got {bars_per_day!r}.")

        symbols = spans.height
        # The timestamp axis is what a dense panel is allocated on; it equals
        # the trading-day count only for daily bars.
        timestamps = trading_days * bars_per_day
        dense_cells = symbols * timestamps
        observed_cells = min(
            round(
                float(spans["span_days"].sum() or 0)
                * self.TRADING_DAYS_PER_YEAR
                / self.CALENDAR_DAYS_PER_YEAR
            )
            * bars_per_day,
            dense_cells,
        )
        density = observed_cells / dense_cells if dense_cells else 0.0

        return {
            "symbols": symbols,
            "trading_days": trading_days,
            "bars_per_day": bars_per_day,
            "timestamps": timestamps,
            "dense_cells": dense_cells,
            "observed_cells": observed_cells,
            "density": density,
        }

    #: Rows one symbol-day yields at each ``Frequency`` token. ``1d`` is one
    #: bar per trading day. ``1m`` is 390, the regular 09:30-16:00 ET session;
    #: this assumes regular hours only, and including extended hours
    #: (04:00-20:00 ET) would raise it to about 960. ``tick`` is deliberately
    #: absent: tick volume is not derivable from a calendar, so
    #: ``estimate_acquisition_volume`` requires ``rows_per_symbol_day`` for
    #: it instead of guessing.
    BARS_PER_DAY_BY_FREQUENCY: dict[str, int] = {"1d": 1, "1m": 390}

    #: Bytes one raw row occupies on disk before any densification: roughly
    #: what a timestamp plus a handful of float OHLCV columns compress to in
    #: parquet. A sizing figure, like ``TRADING_DAYS_PER_YEAR``.
    BYTES_PER_RAW_ROW = 60

    #: Requests per minute assumed when the caller names none: the vendor's
    #: free-tier ceiling for the historical API. The rate limit dominates a
    #: backfill's wall clock, which is why wall clock is a ceiling of its
    #: own.
    DEFAULT_RATE_LIMIT_PER_MIN = 200

    def _resolve_volume_knobs(
        self,
        frequency: str,
        batch_size: int,
        page_limit: int,
        rate_limit_per_min: int | None,
    ) -> int:
        """Validate the sizing knobs and return the resolved rate limit.

        Each knob arrives from a CLI flag or ``config.kwargs``, so it is checked
        here rather than surfacing later as a ``ZeroDivisionError``.

        Raises:
            ValueError: For a frequency this estimator cannot size, or a knob
                below 1.
        """
        if frequency not in self.BARS_PER_DAY_BY_FREQUENCY and frequency != "tick":
            raise ValueError(
                f"Unknown frequency {frequency!r}; this estimator prices "
                f"{sorted(self.BARS_PER_DAY_BY_FREQUENCY) + ['tick']}. A "
                f"frequency with no bars-per-day figure cannot be sized, and "
                f"defaulting one would invent the number the guard exists to "
                f"defend."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}.")
        if page_limit < 1:
            raise ValueError(f"page_limit must be >= 1, got {page_limit!r}.")
        resolved = (
            self.DEFAULT_RATE_LIMIT_PER_MIN
            if rate_limit_per_min is None
            else rate_limit_per_min
        )
        if resolved < 1:
            raise ValueError(
                f"rate_limit_per_min must be >= 1, got {rate_limit_per_min!r}."
            )
        return resolved

    def estimate_acquisition_volume(
        self,
        category: str,
        start_date: str,
        end_date: str,
        *,
        frequency: str,
        batch_size: int,
        page_limit: int = 10_000,
        rate_limit_per_min: int | None = None,
        rows_per_symbol_day: int | None = None,
    ) -> dict:
        """Price a download before it starts: rows, raw bytes, requests and hours.

        Pure arithmetic over this catalog's listing intervals; it issues no vendor
        request and constructs no client. It bounds disk, request count and wall
        clock, not memory, and is a sizing figure in the same spirit as the 252-day
        year: precise enough to separate an eight-minute fetch from a fifty-hour
        one.

        For bar frequencies the row count uses observed cells, not the dense grid,
        because a full-market roster is listed only part of the time. For
        ``frequency="tick"`` the dense count is used and ``rows_per_symbol_day`` is
        required, since tick volume cannot be derived from a calendar.

        Args:
            category: A token from ``known_categories``.
            start_date: ISO date, inclusive.
            end_date: ISO date, inclusive.
            frequency: ``"1d"``, ``"1m"`` or ``"tick"``.
            batch_size: Symbols per request; a fetch issues at least one request
                per batch.
            page_limit: Maximum rows one response can carry.
            rate_limit_per_min: Requests per minute; ``None`` uses
                ``DEFAULT_RATE_LIMIT_PER_MIN``.
            rows_per_symbol_day: Measured rows per symbol-day, required for
                ``"tick"`` and ignored otherwise.

        Returns:
            A dict with ``symbols``, ``trading_days``, ``density``,
            ``bars_per_day``, ``rows``, ``raw_bytes``, ``requests`` and
            ``wall_clock_hours``, plus the inputs and resolved knobs so a caller
            can print a refusal without recomputing.

        Raises:
            ValueError: For an unsizable frequency, a knob below 1, or a tick
                estimate without ``rows_per_symbol_day``.

        Example:
            >>> estimate = catalog.estimate_acquisition_volume(
            ...     "us_all", "2020-01-01", "2020-12-31", frequency="1d", batch_size=100
            ... )
            >>> estimate["symbols"], estimate["trading_days"], estimate["requests"]
            (2, 253, 1)
        """
        rate_limit_per_min = self._resolve_volume_knobs(
            frequency, batch_size, page_limit, rate_limit_per_min
        )

        # Delegated so the roster, the trading-day count and the density come
        # from the one method that derives them.
        panel = self._roster_window_profile(category, start_date, end_date)
        symbols = panel["symbols"]
        trading_days = panel["trading_days"]

        if frequency == "tick":
            if rows_per_symbol_day is None:
                raise ValueError(
                    f"rows_per_symbol_day is REQUIRED for frequency='tick' "
                    f"and has no default. Tick volume is not derivable from a "
                    f"calendar the way a bar count is -- it depends on the "
                    f"symbol's liquidity and the day's activity, and the "
                    f"order-of-magnitude figures available (~100k trades and "
                    f"10-20x that in quotes per liquid symbol-day) are "
                    f"unmeasured against this vendor. Guessing here would make "
                    f"this guard confidently wrong in exactly the regime it "
                    f"exists for, so it refuses instead. Pass a measured "
                    f"rows_per_symbol_day (sample one symbol-day and count)."
                )
            if rows_per_symbol_day < 1:
                raise ValueError(
                    f"rows_per_symbol_day must be >= 1, got "
                    f"{rows_per_symbol_day!r}."
                )
            bars_per_day = rows_per_symbol_day
            # Dense, not density-adjusted: a tick estimate covers a narrow
            # window over liquid names that are listed throughout.
            rows = symbols * trading_days * rows_per_symbol_day
        else:
            bars_per_day = self.BARS_PER_DAY_BY_FREQUENCY[frequency]
            # Observed cells, never dense: a full-market roster over a decade
            # is listed only about a third of the time, and a guard that
            # overstates gets ignored.
            rows = panel["observed_cells"] * bars_per_day

        raw_bytes = rows * self.BYTES_PER_RAW_ROW
        # Two independent floors: pages, because a response carries at most
        # `page_limit` rows; batches, because a fetch issues at least one
        # request per batch even when every row would fit on one page.
        requests_needed = max(
            math.ceil(rows / page_limit), math.ceil(symbols / batch_size)
        )
        return {
            "category": category,
            "start_date": start_date,
            "end_date": end_date,
            "frequency": frequency,
            "symbols": symbols,
            "trading_days": trading_days,
            "density": panel["density"],
            "bars_per_day": bars_per_day,
            "rows": rows,
            "raw_bytes": raw_bytes,
            "requests": requests_needed,
            "wall_clock_hours": requests_needed / rate_limit_per_min / 60,
            "batch_size": batch_size,
            "page_limit": page_limit,
            "rate_limit_per_min": rate_limit_per_min,
        }

    #: Ceiling on the raw bytes one fetch may write to disk. A disk
    #: constraint, sized to admit a full-market daily backfill or a year of
    #: index-constituent minute bars while refusing full-market minute
    #: history or a day of full-market quotes. Denominated in bytes because
    #: a tick fetch can pass a request-count ceiling and still fill the
    #: volume.
    MAX_RAW_BYTES = 20 * 1024**3

    #: Ceiling on the vendor requests one fetch may issue. Independent of the
    #: byte ceiling: a fetch can be small on disk and still pathological in
    #: request count (a low ``page_limit``, or a wide roster at
    #: ``batch_size=1``), and quota is spent per request.
    MAX_ACQUISITION_REQUESTS = 50_000

    #: Ceiling on the wall-clock hours one fetch may take: the request
    #: ceiling at the default rate limit. Kept as its own constant because
    #: it is the number a user feels, and because a paid tier changes the
    #: hours without changing the request count.
    MAX_ACQUISITION_WALL_CLOCK_HOURS = 4.0

    def _narrowing_that_fits(
        self,
        estimate: dict,
        overshoot: float,
        ceilings: tuple[int, int, float],
        rows_per_symbol_day: int | None,
    ) -> tuple[str, dict | None, int]:
        """Return a shorter window that would pass the ceilings, with its estimate.

        The window is scaled down by ``overshoot`` and re-estimated rather than
        divided through, because the request count has a ceiling and a batch floor
        in it. If the candidate still does not fit it is halved a bounded number
        of times; when nothing fits, the per-batch floor alone is over a ceiling
        and only a smaller roster can help.

        Returns:
            ``(window_end, narrowed_estimate or None, max_symbols)``, where
            ``max_symbols`` is the roster size that would fit the original window.
        """
        start = datetime.date.fromisoformat(estimate["start_date"])
        window_days = (
            datetime.date.fromisoformat(estimate["end_date"]) - start
        ).days + 1
        max_symbols = max(int(estimate["symbols"] / overshoot), 1)

        candidate_days = max(int(window_days / overshoot), 1)
        narrowed_end = estimate["end_date"]
        for _attempt in range(8):
            narrowed_end = (
                start + datetime.timedelta(days=candidate_days - 1)
            ).isoformat()
            narrowed = self.estimate_acquisition_volume(
                estimate["category"],
                estimate["start_date"],
                narrowed_end,
                frequency=estimate["frequency"],
                batch_size=estimate["batch_size"],
                page_limit=estimate["page_limit"],
                rate_limit_per_min=estimate["rate_limit_per_min"],
                rows_per_symbol_day=rows_per_symbol_day,
            )
            if not self._crossed_ceilings(narrowed, ceilings):
                return narrowed_end, narrowed, max_symbols
            if candidate_days == 1:
                break
            candidate_days = max(candidate_days // 2, 1)
        return narrowed_end, None, max_symbols

    #: ``(label, estimate key, class constant, keyword)`` for each ceiling, in
    #: the order a refusal reports them. One declaration, so the constant a
    #: reader is told to edit and the keyword they are told to pass cannot
    #: drift from the value being checked.
    ACQUISITION_CEILINGS: tuple[tuple[str, str, str, str], ...] = (
        ("raw-bytes", "raw_bytes", "MAX_RAW_BYTES", "max_raw_bytes"),
        ("request", "requests", "MAX_ACQUISITION_REQUESTS", "max_requests"),
        (
            "wall-clock",
            "wall_clock_hours",
            "MAX_ACQUISITION_WALL_CLOCK_HOURS",
            "max_wall_clock_hours",
        ),
    )

    @classmethod
    def _crossed_ceilings(
        cls, estimate: dict, ceilings: tuple[int, int, float]
    ) -> list[tuple[str, str, float, float, str, str]]:
        """Return every ceiling ``estimate`` exceeds, as report tuples.

        All three are checked independently so a refusal names every reason at
        once. Each tuple is ``(label, key, actual, ceiling, constant, keyword)``.
        """
        return [
            (label, key, estimate[key], ceiling, constant, keyword)
            for (label, key, constant, keyword), ceiling in zip(
                cls.ACQUISITION_CEILINGS, ceilings
            )
            if estimate[key] > ceiling
        ]

    def assert_acquisition_volume_fits(
        self,
        category: str,
        start_date: str,
        end_date: str,
        *,
        frequency: str,
        batch_size: int,
        page_limit: int = 10_000,
        rate_limit_per_min: int | None = None,
        rows_per_symbol_day: int | None = None,
        max_raw_bytes: int | None = None,
        max_requests: int | None = None,
        max_wall_clock_hours: float | None = None,
        force: bool = False,
    ) -> dict:
        """Return the estimate, or raise if it crosses a disk, request or hour ceiling.

        Call this before the acquisition client is constructed and before any
        request is issued, so a fetch that cannot complete is refused immediately
        instead of hours in. The three ceilings are independent because any one
        alone lets a real case through: a request-count check passes a tick fetch
        that fills the disk, and a byte check passes a small, slow fetch that runs
        overnight.

        The refusal names every crossed ceiling, the class constant behind it, the
        keyword that raises it, and a concrete narrowing (fewer symbols or a
        shorter window) that would fit.

        Args:
            category: A token from ``known_categories``.
            start_date: ISO date, inclusive.
            end_date: ISO date, inclusive.
            frequency: ``"1d"``, ``"1m"`` or ``"tick"``.
            batch_size: Symbols per request.
            page_limit: Maximum rows one response can carry.
            rate_limit_per_min: Requests per minute; ``None`` uses the default.
            rows_per_symbol_day: Required for ``"tick"``, ignored otherwise.
            max_raw_bytes: Overrides ``MAX_RAW_BYTES`` when given.
            max_requests: Overrides ``MAX_ACQUISITION_REQUESTS`` when given.
            max_wall_clock_hours: Overrides ``MAX_ACQUISITION_WALL_CLOCK_HOURS``
                when given.
            force: Skip the raise but still compute the estimate. There is no
                environment variable or config key that disables the guard.

        Returns:
            The dict from ``estimate_acquisition_volume``.

        Raises:
            ValueError: If any ceiling is crossed and ``force`` is false, or for
                the input errors ``estimate_acquisition_volume`` raises.

        Example:
            >>> estimate = catalog.assert_acquisition_volume_fits(
            ...     "us_all", "2020-01-01", "2020-12-31", frequency="1d", batch_size=100
            ... )
            >>> estimate["rows"]
            505
            >>> catalog.assert_acquisition_volume_fits(
            ...     "us_all", "2020-01-01", "2020-12-31", frequency="1d",
            ...     batch_size=100, max_raw_bytes=1000,
            ... )
            Traceback (most recent call last):
            ...
            ValueError: Refusing to fetch us_all 1d over 2020-01-01..2020-12-31: ...
        """
        estimate = self.estimate_acquisition_volume(
            category,
            start_date,
            end_date,
            frequency=frequency,
            batch_size=batch_size,
            page_limit=page_limit,
            rate_limit_per_min=rate_limit_per_min,
            rows_per_symbol_day=rows_per_symbol_day,
        )
        ceilings = (
            self.MAX_RAW_BYTES if max_raw_bytes is None else max_raw_bytes,
            self.MAX_ACQUISITION_REQUESTS if max_requests is None else max_requests,
            (
                self.MAX_ACQUISITION_WALL_CLOCK_HOURS
                if max_wall_clock_hours is None
                else max_wall_clock_hours
            ),
        )
        crossed = self._crossed_ceilings(estimate, ceilings)
        if not crossed or force:
            return estimate

        gib = 1024**3
        overshoot = max(
            actual / ceiling for _l, _k, actual, ceiling, _c, _kw in crossed
        )

        def _render(label: str, actual: float, ceiling: float) -> str:
            """Format one crossed ceiling as ``actual > ceiling`` in its own unit."""
            if label == "raw-bytes":
                return f"{actual / gib:.2f} GiB > {ceiling / gib:.2f} GiB"
            if label == "request":
                return f"{actual:,.0f} > {ceiling:,.0f}"
            return f"{actual:.1f} h > {ceiling:.1f} h"

        # Not `.capitalize()`: it would lowercase `GiB` and the constant names
        # a reader is meant to go and edit.
        reasons = "Over the " + "; over the ".join(
            f"{label} ceiling ({_render(label, actual, ceiling)}, {constant})"
            for label, _key, actual, ceiling, constant, _kw in crossed
        )
        # Only the crossed ceilings' keywords are offered, so a refusal never
        # tells the user to raise a ceiling they are nowhere near.
        keywords = " / ".join(keyword for *_rest, keyword in crossed)
        narrowed_end, narrowed, max_symbols = self._narrowing_that_fits(
            estimate, overshoot, ceilings, rows_per_symbol_day
        )
        if narrowed is None:
            # No window is short enough: the per-batch floor alone is over a
            # ceiling, so only a smaller roster or a bigger batch can help.
            cure = (
                f"No shorter window fits -- at batch_size="
                f"{estimate['batch_size']} the per-batch floor alone is over "
                f"the ceiling, so narrow the ROSTER to <= {max_symbols:,} "
                f"symbol(s) (a smaller --universe), or raise --batch-size."
            )
        else:
            cure = (
                f"A narrowing that fits: the same window at "
                f"<= {max_symbols:,} symbol(s) (a smaller --universe, e.g. an "
                f"index-constituent category), or this roster over "
                f"<= {(datetime.date.fromisoformat(narrowed_end) - datetime.date.fromisoformat(start_date)).days + 1:,} "
                f"calendar day(s) ({start_date}..{narrowed_end}), which is "
                f"~{narrowed['requests']:,} request(s), "
                f"~{narrowed['raw_bytes'] / gib:.2f} GiB, "
                f"~{narrowed['wall_clock_hours']:.1f} h."
            )
        raise ValueError(
            f"Refusing to fetch {category} {frequency} over "
            f"{start_date}..{end_date}: {estimate['symbols']:,} symbol(s) x "
            f"{estimate['trading_days']:,} trading day(s) x "
            f"{estimate['bars_per_day']:,} row(s)/symbol-day = "
            f"{estimate['rows']:,} row(s) -> {estimate['requests']:,} "
            f"request(s), {estimate['raw_bytes'] / gib:.2f} GiB, "
            f"{estimate['wall_clock_hours']:.1f} h at "
            f"{estimate['rate_limit_per_min']:,} req/min "
            f"(batch_size={estimate['batch_size']:,}, "
            f"page_limit={estimate['page_limit']:,}). "
            f"{reasons} -- {overshoot:.1f}x the tightest "
            f"ceiling. {cure} Or raise that ceiling deliberately via the "
            f"{keywords} keyword (readable from config.kwargs), or pass "
            f"force=True (--force-volume) to proceed anyway."
        )

    def get_symbols_as_of(self, category: str, as_of_date: str) -> list[str]:
        """Return every symbol whose interval contains ``as_of_date``.

        This is point-in-time membership on one day, the query a walk-forward
        backtest wants at each rebalance; ``get_symbols_in_range`` is the one a
        full-window backfill wants. The result is de-duplicated and sorted
        ascending, and the order is part of the contract for the same reason as
        there.

        Args:
            category: A token from ``known_categories``.
            as_of_date: ISO date. For an index category it must not precede that
                index's coverage start.

        Returns:
            Sorted, de-duplicated symbols.

        Raises:
            ValueError: For an unknown category, a non-ISO date, or a date before
                the category's coverage start.

        Example:
            >>> catalog.get_symbols_as_of("us_all", "1990-01-01")
            ['AAPL', 'MSFT', 'OLD1']
            >>> catalog.get_symbols_as_of("sp500_constituent", "1970-01-01")
            Traceback (most recent call last):
            ...
            ValueError: Cannot answer sp500_constituent membership before 1976-07-01 ...
        """
        # Both arguments arrive unvalidated from CLI flags. An empty list is a
        # legitimate answer, so a typo must raise rather than silently select
        # nothing.
        self._validate_category(category)
        as_of_date = self._normalize_iso_date(as_of_date, "as_of_date")

        # Roster categories have no coverage boundary; index categories take
        # theirs from the registry through the helper shared with
        # `get_symbols_in_range`.
        self._assert_within_coverage(category, as_of_date, "as_of_date")

        matched = self._backend.get_lazyframe().filter(
            (pl.col("category") == category)
            & (pl.col("start_date") <= as_of_date)
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= as_of_date))
        )
        # Sorted for the same reason as in `get_symbols_in_range`: the order
        # is a contract callers slice against.
        return (
            matched.select("symbol")
            .unique()
            .sort("symbol")
            .collect()["symbol"]
            .to_list()
        )
