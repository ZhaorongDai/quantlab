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

# `base.chunking` is a LEAF (stdlib + pandas/xarray, zero project-internal
# imports), so importing it here cannot create a cycle -- this module
# already reaches into `base.config` and `dataset.backend`.
from quantlab.base.chunking import TimeChunkPlanner
from quantlab.base.config import UniverseConfig
from quantlab.dataset.backend import PlBackend
from quantlab.enums.data import TRADEABLE_TICKER_PATTERN, UniverseCategory

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


#: A preferred-share ticker, in every notation Tiingo's directory actually
#: uses. Read as: a delimiter (`-` or `/`), optional whitespace, a `P`, an
#: optional single letter (which absorbs both the series letter in `BC/PA`
#: and the `R` of the `PR` spelling), optional whitespace, then either
#: another delimiter or end-of-ticker.
#:
#: Measured against the live directory on 2026-09-06, over the 15,425
#: distinct tickers `USEquityUniverseFetcher`'s exchange/assetType/currency
#: filter yields. It matches every one of these observed shapes:
#:
#:     ROOT-P-SERIES     823   AAM-P-A
#:     ROOT-P-SERIES-X    84   ZB-P-F-CL
#:     ROOT-P             20   MTB-P
#:     ROOT/PSERIES        3   BC/PA
#:     ROOT--P-X           3   SCE--P-D, IMH-P--B, IMH-P--C
#:     ROOT- PR-X          1   NYCB- PR-U
#:     -P-SERIES           1   -P-HIZ
#:
#: **The trap this shape exists to avoid.** `BRK-A`, `BRK-B`, `BF-A`, `BF-B`,
#: `PBR-A`, `HEI-A`, `MOG-A`, `LEN-B`, `UA-C`, `MKC-V`, `AGM-A`, `CRD-A`,
#: `LGF-A`, `GEF-B`, `STZ-B`, `UHAL-B` and `CWEN-A` are COMMON STOCK carrying
#: a hyphen. A pattern that merely looked for `-<letter>` would delete
#: Berkshire Hathaway from the full-market roster, silently. Requiring the
#: `P` immediately after the delimiter is what separates the two populations.
#:
#: Cross-checked against an independent segment-split reference implementation
#: (split on `[-/]`, whitespace-strip each segment, match any segment at index
#: >= 1 against `^P(?:R|[A-Z])?$`): the two agree on all 15,425 tickers
#: exactly. Combined with `_BABY_BOND_PATTERN` it removes 965 rows / 940
#: distinct tickers (932 preferred + 8 baby bonds, zero overlap) and ZERO
#: legitimate common stocks.
_PREFERRED_SHARE_PATTERN = r"[-/]\s*P[A-Z]?\s*(?:[-/]|$)"

#: A baby bond / note, whose ticker embeds a coupon and sometimes a maturity:
#: `ASRV 8.45 06-30-28`, `SO 6.75 08-01-22`, `NEE 6.219`, `CHNG 6`. A space
#: followed by a digit is the whole tell -- 8 distinct live tickers, and no
#: common stock in the directory contains one.
_BABY_BOND_PATTERN = r"\s\d"


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
    #: Opt-in: drop preferred shares and baby bonds (`_PREFERRED_SHARE_PATTERN`
    #: / `_BABY_BOND_PATTERN`) from this roster.
    #:
    #: **The `False` default is the entire mechanism protecting Locked
    #: Decision A4 / D-02.** `nasdaq_all`'s semantics are FROZEN, and a shared
    #: unconditional filter here would have silently changed what that
    #: category means -- the one outcome this flag exists to prevent. Only
    #: `USEquityUniverseFetcher` opts in.
    #:
    #: A class constant rather than an overridable method because a roster is
    #: DATA in this module (see the class docstring: "adding a roster is a
    #: data change, three class constants, not a code change"), so the opt-in
    #: belongs exactly where `EXCHANGE_FILTER`, `MIN_ROSTER_ROWS` and
    #: `CATEGORY` already live -- and the criterion itself then stays in ONE
    #: place instead of being duplicated per subclass.
    EXCLUDE_NON_COMMON_SECURITY_TYPES: bool = False

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
        if self.EXCLUDE_NON_COMMON_SECURITY_TYPES:
            # Filtered on the SOURCE's own `ticker` column, before the rename
            # below, so the criterion reads against the vocabulary it was
            # measured on.
            #
            # KEY LINK: this runs BEFORE the `MIN_ROSTER_ROWS` check, so that
            # guard validates the count that actually gets PERSISTED rather
            # than a pre-exclusion count it would then bless without ever
            # having seen the real roster. Measured headroom: 15,173
            # surviving rows against a floor of 8,000 -- 1.90x -- so the
            # exclusion cannot trip the guard it is not meant to trip.
            before = len(data)
            data = data.filter(
                ~pl.col("ticker").str.contains(_PREFERRED_SHARE_PATTERN)
                & ~pl.col("ticker").str.contains(_BABY_BOND_PATTERN)
            )
            # Logged at INFO so a future criterion change is visible in a
            # refresh log, not only as a diff in the resulting parquet.
            logger.info(
                f"{self.CATEGORY}: excluded {before - len(data)} preferred / "
                f"baby-bond rows ({before} -> {len(data)})."
            )

        # Build-time WELL-FORMEDNESS drop (260907-10t). Filtered on the
        # SOURCE's own `ticker` column, before the rename below, matching the
        # exclusion block's stated convention.
        #
        # UNCONDITIONAL -- deliberately NOT behind
        # `EXCLUDE_NON_COMMON_SECURITY_TYPES`. A malformed entry is
        # unfetchable for EVERY roster: `Acquisition._validate_symbols`
        # refuses it before a single request is issued, so persisting one
        # guarantees that `download()`'s whole-roster pre-flight aborts a
        # multi-hour job. `nasdaq_all` halts on its own 7 today and does not
        # opt into the exclusion, so gating this on that flag would fix
        # `us_all` alone and leave `nasdaq_all` permanently unfetchable.
        #
        # THE AXIS IS WELL-FORMEDNESS, NOT SECURITY TYPE. `nasdaq_all`'s
        # frozen preferred shares (`FITB-P-A/-I/-K/-M`, `AAM-P-A`, `MTB-P`)
        # are all well-formed and are untouched -- Locked Decision A4 / D-02
        # is unaffected, and `test_the_malformed_drop_does_not_touch_nasdaq_
        # alls_preferred_shares` keeps the two axes from being conflated.
        #
        # MEASURED 2026-09-07 against the live reference table (`us_all`
        # 14,485 unique symbols, `nasdaq_all` 8,967).
        # `TRADEABLE_TICKER_PATTERN` rejects exactly and only:
        #
        #   us_all     (6): CAPTW(EXP20260807), DTV_1, ETP-, NSPR-WSB,
        #                   NXT(EXP20091224), OXY-WSW
        #   nasdaq_all (7): -P-HIZ, ASRV 8.45 06-30-28, CAPTW(EXP20260807),
        #                   CHNG 6, DTV_1, NSPR-WSB, NXT(EXP20091224)
        #
        # Two of those were argued rather than assumed, on the symbol FAMILY
        # each belongs to in that same table:
        #   - `OXY-WSW`: family `['OXY', 'OXY-WS', 'OXY-WS-W', 'OXY-WSW']`.
        #     The properly delimited form of the SAME warrant is already in
        #     the roster, so this drop loses no security at all.
        #   - `NSPR-WSB`: family `['NSPR', 'NSPR-WS', 'NSPR-WSB']`. There is
        #     no `NSPR-WS-B`, so this drop DOES lose one microcap warrant
        #     series -- a named, measured, carried-forward finding in the same
        #     idiom 260906-eme used for the retained warrants. Admitting it
        #     would mean widening every suffix segment from {1,2} to {1,3} for
        #     all 14,485 symbols on the evidence of two outliers, one of them
        #     redundant, against 77 that follow the delimiter convention.
        #
        # KEY LINK: like the exclusion block above, this runs BEFORE the
        # `MIN_ROSTER_ROWS` check, so that guard validates the count that
        # actually gets PERSISTED. A vocabulary drift that zeroed this filter
        # then trips the floor rather than silently persisting a truncated
        # roster (T-10t-03).
        #
        # `str.contains` is a SEARCH, not a match -- the pattern's `^...$`
        # anchors are what make this total. Verified directly rather than
        # assumed: an unanchored search would keep `ETP-` and `DTV_1`, both of
        # which CONTAIN a well-formed substring
        # (`test_the_well_formedness_filter_matches_the_whole_string_not_a_
        # substring`).
        before = len(data)
        data = data.filter(
            pl.col("ticker").str.contains(
                TRADEABLE_TICKER_PATTERN.pattern, literal=False
            )
        )
        # Same shape as the exclusion log above -- count before and after,
        # plus the CATEGORY -- so a future vocabulary drift is visible in a
        # refresh log rather than only as a diff in the resulting parquet.
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
    # 260906-eme keeps that lock: this roster deliberately does NOT opt into
    # `EXCLUDE_NON_COMMON_SECURITY_TYPES`, so it still carries its preferred
    # shares and baby bonds exactly as it always has.
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
    keeps its exact prior semantics per D-02. Both are deliberately retained.

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

    **Non-common-stock exclusion (260906-eme).** This roster additionally
    opts into `EXCLUDE_NON_COMMON_SECURITY_TYPES`, dropping preferred shares
    and baby bonds so downstream ingestion stops spending Tiingo requests and
    panel columns on preferred series and notes. Measured against the live
    directory on 2026-09-06::

        ROOT-P-SERIES     823   preferred            AAM-P-A
        ROOT-P-SERIES-X    84   preferred            ZB-P-F-CL
        ROOT-P             20   preferred            MTB-P
        ROOT/PSERIES        3   preferred            BC/PA
        ROOT--P-X           3   preferred            SCE--P-D
        ROOT- PR-X          1   preferred            NYCB- PR-U
        -P-SERIES           1   preferred            -P-HIZ
        ROOT <coupon>       8   baby bonds / notes   NEE 6.219

    16,138 rows -> **15,173 rows**: 965 rows / 940 distinct tickers removed
    (932 preferred + 8 baby bonds, zero overlap), and ZERO legitimate common
    stocks. Verified survivors include every class share -- `BRK-A`, `BRK-B`,
    `BF-A`, `BF-B`, `PBR-A`, `HEI-A`, `MOG-A`, `MOG-B`, `LEN-B`, `CWEN-A`,
    `UA-C`, `MKC-V`, `AKO-A`, `AKO-B`, `AGM-A`, `NYLD-A`, `TAP-A`, `LGF-A`,
    `LGF-B`, `GTN-A`, `HVT-A`, `BWL-A`, `BH-A`, `BNRE-A`, `CRD-A`, `CRD-B`,
    `RDS-A`, `RDS-B`, `FCE-A`, `GEF-B`, `STZ-B`, `UHAL-B`, `WSO-B`, `BIO-B`,
    `CIG-C`, `EBR-B`, `TI-A`, `GGO-C`, `BALY-T`, `SPWR-V`. Class shares are
    common stock carrying a hyphen, and they are the precise trap the
    criterion is shaped around.

    **Deliberately still IN, stated rather than silently omitted:** the 1,124
    warrant / unit / right / when-issued lines (`-WS` 391, `-U` 360, `-W` 164,
    `-R` 138, `-CL` 57, `-WD` 11, `-WI` 3). The scope is preferred shares and
    baby bonds; this is a named, measured, carried-forward finding, and their
    retention is asserted in the tests so a later widening must be deliberate.
    **They are now FETCHABLE as well as retained (260907-10t):** 77 of them
    are three-segment `ROOT-X-Y` (`NXG-R-W`, `BAC-WS-A`, `UA-C-W`), which the
    fetch-time guard used to refuse -- a real full-market `download()` aborted
    its whole-roster pre-flight on `NXG-R-W` before issuing one request.
    Retaining them and being unable to fetch them were two separately
    deliberate decisions that had never been pinned against each other.

    **Malformed entries are now dropped at BUILD time (260907-10t), and two of
    those drops are themselves named carried-forward findings.** The filter's
    axis is well-formedness, never security type; see
    `TiingoRosterFetcher.fetch()` for the measurement and for the family
    evidence behind `OXY-WSW` (redundant -- `OXY-WS-W` is already in the
    roster) and `NSPR-WSB` (NOT redundant -- no `NSPR-WS-B` exists, so this
    one genuinely loses a microcap warrant series, accepted rather than widen
    the segment bound for all 14,485 symbols).

    **The resulting asymmetry with `nasdaq_all` is intentional.** `us_all` is
    NO LONGER a strict superset of `nasdaq_all`: a NASDAQ-listed preferred
    such as `ONB-P-A` appears in `nasdaq_all` and not in `us_all`. That is not
    an inconsistency to fix in this code -- `nasdaq_all`'s semantics are
    FROZEN by Locked Decision A4 / D-02, and the `False` default of
    `EXCLUDE_NON_COMMON_SECURITY_TYPES` on `TiingoRosterFetcher` is what
    freezes them. Anyone wanting to align the two should reopen that decision,
    not widen the flag.
    """

    EXCHANGE_FILTER = ("NASDAQ", "NYSE", "AMEX", "NYSE MKT")

    EXCLUDE_NON_COMMON_SECURITY_TYPES = True

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

#: What a change-log ticker cell must look like AFTER
#: `IndexMembershipFetcher._normalize_ticker_cell()` has run. Pinned against a
#: live measurement (2026-09-06) of BOTH change logs pushed through
#: `_parse_changes_table`: 1,190 non-null ticker cells (S&P 500 772,
#: Nasdaq-100 418), observed lengths 1-6, character vocabulary `A-Z` plus --
#: in exactly THREE S&P cells -- a space and a trailing `|`. After
#: normalization all 1,190 match this pattern.
#:
#: The optional `[.-][A-Z0-9]{1,2}` tail is deliberate headroom for the
#: `BRK.B` / `BRK-B` class-share notations: neither table carries one today,
#: but it is the one shape a future row could legitimately hold, and a
#: validator that rejected it would break every refresh on the day it
#: appeared.
#:
#: Digits are admitted even though ZERO live cells carry one. Two reasons,
#: both deliberate: (1) a false positive here is expensive in exactly the way
#: hard-won by the normalization design -- it raises, `fetch_changes()` falls
#: back to the stale cache, and `build()` then refuses until a human edits
#: Wikipedia; (2) this repo's synthetic change-log fixtures name their
#: symbols `ADDED1` / `TEMP1` / `GONE1` precisely so a synthetic symbol is
#: never mistakable for a real ticker, and that convention is worth more than
#: a character class narrowed to a vocabulary that could widen upstream at
#: any time. The malformations this guard actually exists to catch -- an
#: interior delimiter, an embedded space, lowercase, an over-long cell --
#: are all still rejected.
#:
#: DO NOT ALIGN THIS WITH `enums.data.TRADEABLE_TICKER_PATTERN` (260907-10t).
#: That one is its deliberately WIDER sibling, and the difference is the
#: design, not a drift left over from a refactor. The two answer different
#: questions on different inputs:
#:
#:   TRADEABLE_TICKER_PATTERN  -- "can this symbol safely become a path
#:     segment and a query value?" Input: Tiingo's ticker directory, which
#:     legitimately contains 77 three-segment `ROOT-X-Y` warrants and
#:     when-issued lines in `us_all` alone. Admits up to TWO suffix segments.
#:
#:   _WELL_FORMED_TICKER (this) -- "did the Wikipedia change-log parser hand
#:     me one cell or two?" Input: scraped HTML cells. An INTERIOR DELIMITER
#:     here means two cells were MERGED, i.e. a parser regression, so a
#:     three-segment value is the signal this guard exists to raise on.
#:     Admits at most ONE.
#:
#: And the input vocabulary cannot need the widening: index constituents are
#: common stock, and BOTH constituent categories have ZERO pattern failures
#: across 1,151 symbols (`sp500_constituent` 876, `nasdaq100_constituent` 275,
#: measured 2026-09-07 against the live reference table). Widening this would
#: delete a real guard to buy nothing. Pinned by
#: `tests/test_ticker_pattern_reconciliation.py:
#: test_the_changelog_guard_is_deliberately_narrower_than_the_fetch_guard`.
_WELL_FORMED_TICKER = re.compile(r"^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2})?$")


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
    def _normalize_ticker_cell(value: str) -> str:
        """Strip upstream wikitable delimiter residue off one ticker cell.

        Applied in order: whitespace strip, leading/trailing `|` strip,
        whitespace strip again -- so the real observed `"ALLE |"` becomes
        `"ALLE"` and a cell that is nothing but residue (`" | "`) becomes the
        empty string, which the `_BLANK_TICKER_CELLS` sentinel then reads as
        "no change on this side".

        **On this base, deliberately -- not on either subclass.**
        `_parse_changes_table` is already concrete and shared precisely so no
        index can ship a parse that skips the base's validation (safety
        property 1 on the class docstring), and that property exists because
        the previous per-subclass parse meant nothing looked at the source
        header at all. The Nasdaq-100 change log has zero malformed cells
        today, but it is the SAME publicly-editable MediaWiki surface, and the
        S&P 500 table's three cells arrived by editor typo alone. Putting the
        normalization on one subclass would reintroduce exactly the shape that
        refactor removed.

        Only LEADING/TRAILING delimiters are stripped. An interior one is not
        the observed upstream shape and much more likely means two cells were
        merged by a parser regression, so it is left in place to fail
        `_WELL_FORMED_TICKER` loudly.
        """
        return value.strip().strip("|").strip()

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
        malformed: list[tuple[str, str]] = []
        for column in ("added_ticker", "removed_ticker"):
            stripped = parsed[column].astype(str).str.strip().tolist()

            # 1. Normalization runs FIRST, on the whitespace-stripped raw
            #    values. Wikipedia is publicly editable and its change logs
            #    carry ongoing delimiter typos (measured 2026-09-06: three
            #    live S&P 500 cells, `ALLE |` / `ITT |` / `JCP |`). A bare
            #    raise on those would break every refresh until somebody
            #    edited Wikipedia; passing them through wrote phantom symbols
            #    that match NO market data, so `JCP` and `ITT` -- two
            #    multi-decade members -- were simply absent from the panel
            #    while `ALLE` was double-counted.
            normalized = [self._normalize_ticker_cell(v) for v in stripped]

            # 2. Every cell the normalization CHANGED is announced. ONE
            #    aggregated line per column, not one per cell: the ongoing
            #    three-cell upstream typo stays a single readable line, while
            #    a systematic parser regression shows up as one huge list
            #    rather than flooding a cron log. The logging is load-bearing
            #    -- silent correction would swallow that regression, which is
            #    the whole reason this module's other guards are loud.
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

            # 3. KEY LINK: the `_BLANK_TICKER_CELLS` sentinel test runs on the
            #    NORMALIZED value, never the raw one. A cell whose whole
            #    content is residue (`" | "`) must become the `None`
            #    "no change on this side" sentinel; testing the raw value
            #    first would send it to the validator below and raise.
            #
            #    `dtype=object` keeps the sentinel a real `None`; a plain list
            #    assignment lets pandas re-infer a string dtype and turn it
            #    back into `nan`. Both survive `pl.from_pandas` as null, but
            #    only the explicit form says so at the layer a reader is
            #    looking at.
            cleaned = [
                None if value in _BLANK_TICKER_CELLS else value
                for value in normalized
            ]

            # 4. Anything non-blank that is STILL not a well-formed ticker is
            #    accumulated across BOTH columns, so one run reports all of
            #    them rather than one per re-run.
            malformed.extend(
                (column, value)
                for value in cleaned
                if value is not None and not _WELL_FORMED_TICKER.match(value)
            )

            parsed[column] = pd.Series(cleaned, dtype=object, index=parsed.index)

        if malformed:
            # Same shape, and same reasoning, as the `unparseable
            # effective_date` guard immediately below: `fetch_changes()`
            # catches this, falls back to the cached snapshot WITHOUT
            # overwriting it, and `build()` then refuses unless
            # `allow_stale=True`. Raising is therefore loud-but-non-
            # destructive, which is what makes it safe to raise on a shape
            # that has not been observed.
            raise ValueError(
                f"{self.INDEX_LABEL}: change-log ticker cells at "
                f"{self.CHANGES_URL} are still malformed after delimiter "
                f"normalization: {malformed}. Refusing to reconstruct "
                f"membership from cells that would enter the panel as "
                f"phantom symbols matching no market data. An INTERIOR "
                f"delimiter is not the observed upstream typo -- it means two "
                f"cells were merged, i.e. a parser regression."
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
    def _normalize_iso_date(value: str, field: str) -> str:
        """Reject a non-ISO date string, and return it CANONICALISED to
        zero-padded `YYYY-MM-DD`.

        The table stores ISO date strings and compares them
        LEXICOGRAPHICALLY, so a non-ISO value does not merely fail to match --
        it compares wrong and returns a plausible, silently incorrect roster.

        **Returning the canonical form is load-bearing, not tidiness, and it
        is why this normalises rather than merely validating.**
        `date.fromisoformat` accepts ANY valid ISO 8601 date since 3.11, not
        the `YYYY-MM-DD` shape the message below promises -- ISO BASIC form
        (`"20070115"`) and week dates (`"2020-W01-1"`) parse happily. Both
        then compare WRONG against this table's dashed strings, in two
        different places and in two different directions:

        - `"20070101" < "2007-02-01"` is False (`'0'` 0x30 beats `'-'` 0x2D at
          index 4), so a pre-coverage date sails straight past
          `_assert_within_coverage()` -- the coverage guard's own bypass.
        - `"2018-03-02" >= "20180101"` is False for the same reason, so every
          membership interval closing in the query date's year is silently
          dropped from the polars filter's result.

        Callers must therefore USE the return value in place of the argument
        they passed; validating and discarding the parse is exactly the bug.
        Same reasoning, and the same shape, as `BaseDataset._normalize_date`
        one layer over.
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
        """This category's point-in-time coverage start, or `None` if it has
        none.

        Built from `MEMBERSHIP_FETCHERS` -- each fetcher's own `CATEGORY` ->
        `PIT_COVERAGE_START` -- so a fifth index inherits a coverage boundary
        by REGISTRATION, never by someone remembering to add an `if`. The
        registry is the single source, and no category token is written into
        either query method's body.

        Returning `None` for an unregistered category is the D-02 CONTRACT,
        not an oversight. `nasdaq_all` and `us_all` live in the separate
        `ROSTER_FETCHERS` registry: they are full-exchange rosters with
        per-symbol listing dates and no index-membership concept, so they have
        no left-censored change log and therefore no coverage start. A
        pre-listing query on either must answer from the roster's own dates
        rather than raising.
        """
        return {
            fetcher_cls.CATEGORY: fetcher_cls.PIT_COVERAGE_START
            for fetcher_cls in self.MEMBERSHIP_FETCHERS
        }.get(category)

    def _assert_within_coverage(
        self, category: str, date: str, field: str
    ) -> None:
        """Refuse a date preceding `category`'s point-in-time coverage start.

        THE coverage guard, shared by BOTH membership queries
        (`get_symbols_as_of` and `get_symbols_in_range`) so there is one
        contract rather than two that can drift apart. Two that could drift is
        exactly how `get_symbols_in_range` came to answer pre-coverage windows
        silently while its sibling raised -- the gap 03.1-VERIFICATION.md
        records as truth 3.

        **The boundary is INCLUSIVE: the comparison is a strict `<`, so a
        `date` exactly EQUAL to the coverage start is ACCEPTED.** That is
        deliberate and load-bearing. The coverage start is the earliest date
        the change log actually covers, so it is answerable; membership
        intervals are closed on both ends everywhere else in this layer; and a
        `<=` here would make the two membership queries disagree on the
        boundary day by exactly one day.

        `field` names WHICH date argument was rejected, so a caller passing
        two dates can tell them apart in a log without reading this source.
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

        **A `start_date` before the category's `PIT_COVERAGE_START` RAISES; it
        is not clamped.** The window's left edge is the only date checked --
        a window whose left edge is inside coverage cannot reach left-censored
        territory -- and the check is the same `_assert_within_coverage()`
        `get_symbols_as_of()` uses, so both membership queries share one
        boundary and agree on the boundary day (which is INCLUSIVE: a
        `start_date` exactly equal to the coverage start is answered).

        Raise rather than clamp, because clamping would buy nothing and cost
        the caller the signal: every membership interval already starts at or
        after its own coverage start, so raising `1900-01-01` to `1976-07-01`
        leaves the overlap predicate's result set IDENTICAL. Clamping would
        hand back the same truncated roster with only a log line to
        distinguish it from a complete one -- the exact
        silent-incomplete-roster failure DATA-05 names, and the one the
        paragraph above already argues against for a typo'd category.

        **`IndexConstituentDataset._clamp_coverage_start()` deliberately does
        the OPPOSITE on the panel side, and the two must NOT be "aligned".**
        The panel receives `enums/constant.py:Date.START_DATE` -- a
        framework-supplied config default nobody typed -- so raising there
        would make every default construction explode; clamping is right. A
        query date is one somebody actually asked, so a wrong value is a
        question, and refusing it is right here.
        """
        self._validate_category(category)
        start_date = self._normalize_iso_date(start_date, "start_date")
        end_date = self._normalize_iso_date(end_date, "end_date")
        self._assert_within_coverage(category, start_date, "start_date")

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

    #: Trading days per calendar year, and the calendar year they are scaled
    #: against. A deliberate APPROXIMATION: `estimate_dense_panel()` produces
    #: a SIZING figure, and taking an exchange-calendar dependency (holidays,
    #: half-days, the 1968 paperwork crisis) to sharpen a "how many GiB is
    #: this" answer by a couple of percent would buy nothing and cost a
    #: package.
    TRADING_DAYS_PER_YEAR = 252
    CALENDAR_DAYS_PER_YEAR = 365.25

    #: Ceiling on the DENSE `[timestamp, symbol]` grid a caller may ask
    #: `StockDataset` to materialise, enforced by `assert_dense_panel_fits()`.
    #:
    #: Justified from measurements taken on the target machine (2026-09-06):
    #: the full `us_all` roster over 2006-01-01..today is 15,424 symbols x
    #: ~5,215 trading days = 80.4M dense cells, of which only ~29.6M are real
    #: observations (density 0.368). At 12 Tiingo EOD variables that dense
    #: grid is ~7.2 GiB of float64. `StockDataset._raw_data_to_xr()` reaches
    #: it through `.collect().to_pandas().set_index([...]).to_xarray()`,
    #: holding the 29.6M-row frame, the dense array AND conversion scratch
    #: simultaneously -- on a 16 GiB box.
    #:
    #: **Disk is not the binding constraint: 120 GiB is free on the target
    #: volume. RAM is.** 4 GiB sits below the ~7.2 GiB that OOMs and above
    #: the windows that comfortably fit, so the guard fires as a legible
    #: error naming the numbers rather than as an OOM three hours into a
    #: backfill. Same safety-envelope idiom as `MIN_ROSTER_ROWS`,
    #: `MIN_ANCHOR_ROWS` and `_assert_every_category_is_populated()`.
    MAX_DENSE_PANEL_BYTES = 4 * 1024**3

    def estimate_dense_panel(
        self,
        category: str,
        start_date: str,
        end_date: str,
        num_variables: int = 12,
        bytes_per_value: int = 8,
        bars_per_day: int = 1,
    ) -> dict:
        """Size the dense `[timestamp, symbol]` panel a window would produce.

        Returns `symbols`, `trading_days`, `bars_per_day`, `timestamps`,
        `dense_cells`, `observed_cells`, `density`, `dense_bytes` and
        `observed_bytes`.

        `bars_per_day` is the length of ONE trading day's timestamp axis, and
        it defaults to 1 because a daily panel has exactly one row per symbol
        per session. It exists because the timestamp axis -- not the trading-day
        count -- is what `to_xarray()` allocates against: at `1m` a session is
        390 rows (`BARS_PER_DAY_BY_FREQUENCY`), so a minute window is 390x the
        dense grid of the same window at `1d`. Sizing a minute fetch with the
        default would admit a panel three orders of magnitude over the budget
        while reporting a number that looks fine, which is the single easiest
        way for this guard to be confidently wrong.

        Everything is derived from this catalog's own interval table clipped
        to the window, because the catalog is the ONLY object that knows when
        each symbol was actually listed -- which is exactly what makes the
        dense grid so much larger than the real observation count. `density`
        below 1 is not an error: it is the survivorship-bias-free roster's
        defining property, ~0.368 for the full US market since 2006.

        `num_variables` defaults to 12 to match `enums.data.TiingoColumns.EOD`
        and `bytes_per_value` to 8 for float64, the dtype
        `StockDataset._raw_data_to_xr()` produces.
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

        # Clip each interval to the window, then reduce to ONE span per
        # symbol. Reducing first is what keeps a dual-listed ticker (~700 of
        # them carry two exchange rows) from being counted twice.
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
        # The TIMESTAMP axis, which is what a dense panel is allocated on --
        # trading days only equals it at `1d`.
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
            "dense_bytes": dense_cells * num_variables * bytes_per_value,
            "observed_bytes": observed_cells * num_variables * bytes_per_value,
        }

    def assert_dense_panel_fits(
        self,
        category: str,
        start_date: str,
        end_date: str,
        num_variables: int = 12,
        bytes_per_value: int = 8,
        bars_per_day: int = 1,
    ) -> None:
        """Raise if densifying this window would exceed
        `MAX_DENSE_PANEL_BYTES`.

        Call this BEFORE `StockDataset.from_raw_data()`, never after: the
        whole point is to fail before the pandas densification allocates
        (T-0iy-03). See `MAX_DENSE_PANEL_BYTES` for why RAM rather than disk
        sets the ceiling.

        **Pass `bars_per_day` for any intraday frequency.** This guard sizes
        the timestamp axis, and at `1m` a session is 390 rows rather than 1
        (`BARS_PER_DAY_BY_FREQUENCY`). Left at the default, it would admit the
        very fetch it exists to refuse -- an S&P-500 minute year is ~4 TB dense
        against a 4 GiB budget, and it would report ~10 GiB.
        """
        estimate = self.estimate_dense_panel(
            category,
            start_date,
            end_date,
            num_variables,
            bytes_per_value,
            bars_per_day,
        )
        if estimate["dense_bytes"] <= self.MAX_DENSE_PANEL_BYTES:
            return

        gib = 1024**3
        raise ValueError(
            f"Refusing to densify {category} over {start_date}..{end_date}: "
            f"the dense [timestamp, symbol] grid is "
            f"{estimate['symbols']} symbol(s) x {estimate['trading_days']} "
            f"trading days x {estimate['bars_per_day']} row(s)/day x "
            f"{num_variables} variables = "
            f"{estimate['dense_bytes'] / gib:.2f} GiB, over the "
            f"{self.MAX_DENSE_PANEL_BYTES / gib:.2f} GiB budget. Only "
            f"{estimate['density']:.1%} of that grid is real observations, "
            f"but StockDataset._raw_data_to_xr() materialises ALL of it -- "
            f"plus the row frame and conversion scratch -- at once, so this "
            f"would exhaust memory rather than merely be wasteful. Narrow "
            f"the date window or the symbol set, or raise "
            f"MAX_DENSE_PANEL_BYTES deliberately if this machine has the RAM."
        )

    def assert_chunked_panel_fits(
        self,
        category: str,
        start_date: str,
        end_date: str,
        granularity: str = "year",
        num_variables: int = 12,
        bytes_per_value: int = 8,
    ) -> dict:
        """Size a CHUNKED densification: refuse per chunk, advise on the total.

        The sibling of `assert_dense_panel_fits()`, not its replacement. That
        one answers "does this whole window fit in RAM at once", which is the
        right question for `from_raw_data()`. This one answers "does one
        `granularity` window fit", which is the right question for
        `from_raw_data_chunked()` -- and it deliberately does NOT raise merely
        because the whole-range total is over budget, because making that
        total achievable is precisely what chunking is for (D-05). The total
        is still returned and printed, as a non-raising advisory, so a caller
        sees what they are committing to.

        **Each chunk is sized on the PINNED WHOLE-RANGE symbol count**, never
        on the roster that overlaps that chunk. `from_raw_data_chunked()`
        resolves the symbol axis once over the entire range and materialises
        EVERY window on it (D-02), so a 2025 window still allocates a column
        for a ticker that delisted in 2009. Calling `estimate_dense_panel()`
        scoped to one chunk would count only the symbols listed during it,
        understate the real allocation, and let the OOM back in -- which is
        the single easiest thing to get subtly wrong here.

        **Why calendar windows are correct in this method and nowhere else.**
        Sizing runs BEFORE the download, when no timestamp axis exists to
        plan against, so `TimeChunkPlanner.plan_calendar()` is the only
        option; and the whole estimator is already a 252/365.25
        approximation, so calendar edges cost nothing here. They must never
        be handed to a densifier -- the write loop uses
        `plan_from_timestamps()` against the real observed axis.

        Returns `{"granularity", "advisory", "chunks", "max_chunk",
        "max_chunk_bytes"}`, so the caller can print without recomputing.
        """
        self._validate_category(category)
        start_date = self._normalize_iso_date(start_date, "start_date")
        end_date = self._normalize_iso_date(end_date, "end_date")

        planner = TimeChunkPlanner(granularity)
        advisory = self.estimate_dense_panel(
            category, start_date, end_date, num_variables, bytes_per_value
        )
        pinned_symbols = advisory["symbols"]

        gib = 1024**3
        chunks: list[dict] = []
        for window_start, window_end in planner.plan_calendar(start_date, end_date):
            window_days = (
                datetime.date.fromisoformat(window_end)
                - datetime.date.fromisoformat(window_start)
            ).days + 1
            trading_days = max(
                round(
                    window_days
                    * self.TRADING_DAYS_PER_YEAR
                    / self.CALENDAR_DAYS_PER_YEAR
                ),
                1,
            )
            dense_cells = pinned_symbols * trading_days
            dense_bytes = dense_cells * num_variables * bytes_per_value
            chunk = {
                "start": window_start,
                "end": window_end,
                "symbols": pinned_symbols,
                "trading_days": trading_days,
                "dense_cells": dense_cells,
                "dense_bytes": dense_bytes,
            }
            if dense_bytes > self.MAX_DENSE_PANEL_BYTES:
                raise ValueError(
                    f"Refusing to densify {category} in {granularity} chunks: "
                    f"the window {window_start}..{window_end} alone is "
                    f"{pinned_symbols} pinned symbol(s) x {trading_days} "
                    f"trading days x {num_variables} variables = "
                    f"{dense_bytes / gib:.2f} GiB, over the "
                    f"{self.MAX_DENSE_PANEL_BYTES / gib:.2f} GiB budget. Every "
                    f"window is materialised on the whole-range symbol axis, "
                    f"so a chunk does not get smaller by containing fewer "
                    f"listed tickers -- only by covering less time. Pass a "
                    f"finer --chunk (year -> quarter -> month), or raise "
                    f"MAX_DENSE_PANEL_BYTES deliberately if this machine has "
                    f"the RAM."
                )
            chunks.append(chunk)

        max_chunk = max(chunks, key=lambda c: c["dense_bytes"]) if chunks else None
        return {
            "granularity": granularity,
            "advisory": advisory,
            "chunks": chunks,
            "max_chunk": max_chunk,
            "max_chunk_bytes": max_chunk["dense_bytes"] if max_chunk else 0,
        }

    #: Rows a single symbol-day yields at each `enums.data.Frequency` token.
    #:
    #: `1d` is one bar per trading day by definition. `1m` is **390** -- the
    #: regular 09:30-16:00 ET session, 6.5 hours x 60 minutes. This is an
    #: ASSUMPTION, not a measurement: including extended hours (04:00-20:00 ET)
    #: would raise it to ~960, a 2.5x error in every minute estimate. It is
    #: stated rather than hidden because the conservative ceilings below absorb
    #: a 2.5x understatement -- the full-market minute scenario is refused ~17x
    #: over at 390 and would merely be refused ~41x over at 960 -- while a
    #: reader who needs the exact figure must be able to see which one is
    #: encoded (03.2-RESEARCH.md Pattern 6, row-count input provenance).
    #:
    #: `tick` is deliberately ABSENT: see `estimate_acquisition_volume`'s
    #: `rows_per_symbol_day`. A number here would be a guess, and a guess here
    #: is what makes a guard confidently wrong in the one regime it exists for.
    BARS_PER_DAY_BY_FREQUENCY: dict[str, int] = {"1d": 1, "1m": 390}

    #: Bytes one RAW row occupies on disk, before any densification.
    #:
    #: DERIVED, not invented. 03.2-RESEARCH.md Pattern 6's Volume Arithmetic
    #: sizes ~15.3M daily rows at ~0.9 GB and ~6.0B minute rows at ~358 GB;
    #: both land at ~60 bytes per raw parquet row, which is what a handful of
    #: f64 OHLCV columns plus a timestamp compress to in practice. It is a
    #: SIZING figure in the same deliberate-approximation spirit as
    #: `TRADING_DAYS_PER_YEAR`: sharpening it by measuring a real shard would
    #: move a 20 GiB ceiling by a fraction of a GiB and change no decision.
    BYTES_PER_RAW_ROW = 60

    #: Requests per minute assumed when the caller names none: the free
    #: (Basic) tier's documented historical-API ceiling.
    #:
    #: The paid tier is 10,000/min -- 50x -- and **the rate limit is the
    #: dominant cost driver for this phase**, because historical depth,
    #: available fields and the recency floor are all IDENTICAL between the
    #: tiers for a backfill. That is why wall clock is a ceiling of its own
    #: rather than a derived note: the same fetch that takes ~50 hours on
    #: Basic takes ~1 hour paid, and nothing else about it differs.
    DEFAULT_RATE_LIMIT_PER_MIN = 200

    def _resolve_volume_knobs(
        self,
        frequency: str,
        batch_size: int,
        page_limit: int,
        rate_limit_per_min: int | None,
    ) -> int:
        """Validate the four knobs and return the resolved rate limit.

        Every one of these arrives off a CLI flag or `config.kwargs`, so each
        is checked where it is consumed. A `batch_size` of 0 would otherwise
        surface as `ZeroDivisionError` from inside a sizing method -- a guard
        that crashes on a typo'd flag has failed at being a guard.
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
        """Price a fetch before it starts: rows, raw bytes, requests, hours.

        **Issues zero vendor requests and constructs no acquisition client.**
        It is arithmetic over this catalog's own listing intervals, which is
        the entire point: an estimate that costs a vendor request has defeated
        itself. Call it BEFORE the client is constructed and before a single
        request -- the 03.2 analogue of `assert_dense_panel_fits`'s "call this
        BEFORE `StockDataset.from_raw_data()`, never after".

        The SIBLING of `estimate_dense_panel`, not its replacement. That one
        bounds RAM for a dense `[timestamp, symbol]` panel; this phase never
        densifies (D-18 fences the conversion out), so the binding constraints
        here are disk, request count and wall clock. Both are SIZING figures
        in the same deliberate-approximation spirit as the 252/365.25 calendar:
        precise enough to separate an 8-minute fetch from a 50-hour one, and
        not pretending to be more.

        Returns `symbols`, `trading_days`, `rows`, `raw_bytes`, `requests`,
        `wall_clock_hours`, plus the resolved knobs (`frequency`,
        `batch_size`, `page_limit`, `rate_limit_per_min`, `bars_per_day`,
        `density`) so a caller can print a refusal without recomputing.

        `rows_per_symbol_day` is REQUIRED for `frequency="tick"` and ignored
        otherwise: tick volume is not derivable from a calendar, so this method
        refuses rather than guessing (T-03.2-20).
        """
        rate_limit_per_min = self._resolve_volume_knobs(
            frequency, batch_size, page_limit, rate_limit_per_min
        )

        # Delegated rather than recomputed: the roster, the trading-day count
        # and the measured 0.368 density all come from the one estimator that
        # already derives them, so the two cannot disagree about the window
        # they are both sizing.
        panel = self.estimate_dense_panel(category, start_date, end_date)
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
            # Dense, not density-adjusted: a tick estimate is asked for a
            # narrow window over liquid names, where the roster is listed
            # throughout and the observed/dense distinction that matters over
            # a decade of full-market history does not apply.
            rows = symbols * trading_days * rows_per_symbol_day
        else:
            bars_per_day = self.BARS_PER_DAY_BY_FREQUENCY[frequency]
            # `observed_cells`, never `dense_cells`: over 2016-2026 the full
            # US roster is only 36.8% listed on average, so a dense count
            # overstates a daily backfill by ~2.7x. A guard that overstates
            # refuses fetches that would have been fine, which is how a guard
            # gets deleted.
            rows = panel["observed_cells"] * bars_per_day

        raw_bytes = rows * self.BYTES_PER_RAW_ROW
        # Two independent floors. Pages, because a response cannot carry more
        # than `page_limit` rows; batches, because a fetch issues at least one
        # request per batch even when every row would fit on one page. Taking
        # only the page term understates a wide, short daily fetch by orders
        # of magnitude.
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

    #: Ceiling on the RAW bytes a single fetch may write to disk, enforced by
    #: `assert_acquisition_volume_fits()`.
    #:
    #: **A DIFFERENT constraint from `MAX_DENSE_PANEL_BYTES`, not a
    #: replacement for it.** That one bounds RAM for a dense
    #: `[timestamp, symbol]` panel; this phase never materialises one (D-18
    #: fences the raw-to-Zarr conversion out), so what binds here is the disk
    #: the raw tier lands on -- ~120 GiB free on the target volume, per
    #: `MAX_DENSE_PANEL_BYTES`'s own measurement.
    #:
    #: 20 GiB sits comfortably under that while admitting every
    #: capability-scale scenario in 03.2-RESEARCH.md Pattern 6's Volume
    #: Arithmetic (`us_all` daily 2016-2026 at ~0.9 GB; S&P-500 minute for a
    #: year at ~3 GB) and refusing every v2-scale one (`us_all` minute at
    #: ~358 GB; one day of full-market quotes at ~30-60 GB).
    #:
    #: **Denominated in BYTES precisely because a row-count check is not
    #: enough**: a tick request passes a request-count ceiling comfortably and
    #: still fills the volume, because its rows-per-request is orders of
    #: magnitude higher than a bar fetch's.
    MAX_RAW_BYTES = 20 * 1024**3

    #: Ceiling on the number of vendor requests a single fetch may issue.
    #:
    #: Admits S&P-500-minute-per-year at ~4,900 requests and refuses `us_all`
    #: minute 2016-2026 at ~597,000 (03.2-RESEARCH.md Pattern 6). Independent
    #: of the byte ceiling on purpose: a fetch can be small on disk and still
    #: be pathological in request count (a low `page_limit`, or a wide roster
    #: at `batch_size=1`), and quota is spent per request, not per byte.
    MAX_ACQUISITION_REQUESTS = 50_000

    #: Ceiling on the wall-clock hours a single fetch may take.
    #:
    #: DERIVED: `MAX_ACQUISITION_REQUESTS / DEFAULT_RATE_LIMIT_PER_MIN / 60` =
    #: 50,000 / 200 / 60 = 4.0 h. Kept as its own constant rather than
    #: computed, because it is the knob a user actually FEELS -- nobody has an
    #: intuition for 50,000 requests and everybody has one for "this will take
    #: four hours" -- and because it must stay independently raisable: on the
    #: paid tier the same request count takes ~5 minutes, so wall clock and
    #: request count genuinely diverge.
    MAX_ACQUISITION_WALL_CLOCK_HOURS = 4.0

    def _narrowing_that_fits(
        self,
        estimate: dict,
        overshoot: float,
        ceilings: tuple[int, int, float],
        rows_per_symbol_day: int | None,
    ) -> tuple[str, dict | None, int]:
        """A CONCRETE alternative window that would pass, and its own numbers.

        Scales the window down by the overshoot ratio, then RE-ESTIMATES it
        (still zero requests) rather than dividing the original figures
        through: `requests` has a `ceil` and a batch floor in it, so a scaled
        arithmetic figure would be a plausible-looking lie. Halves further, a
        bounded number of times, if the first candidate still does not fit --
        which happens exactly when the batch floor rather than the row count
        is what is over, and in that case no window is short enough and the
        honest cure is a smaller roster.

        Returns `(window_end, narrowed_estimate | None, max_symbols)`.
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

    #: `(label, estimate key, class constant, keyword)` for each of the three
    #: independent ceilings, in the order a refusal reports them. ONE
    #: declaration, so the constant a reader is told to edit and the keyword
    #: they are told to pass cannot drift apart from the value being checked.
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
        """Which of the three ceilings this estimate crosses.

        Each is checked INDEPENDENTLY and all crossings are reported, so a
        refusal names every reason rather than only the first -- a caller who
        raises the one ceiling they were told about, only to hit the next,
        learns to distrust the message.
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
        """Raise if this fetch would exceed the disk, request or wall-clock
        ceiling; otherwise return the estimate.

        **Call this BEFORE the acquisition client is constructed and before a
        single request** -- the same position `ingest_us_equity.py` already
        chose for `assert_chunked_panel_fits()`, and for the same documented
        reason: sparing the user a multi-hour backfill that ends in a refusal
        they could have been told about immediately. A guard that runs after
        the client exists has already spent the thing it was meant to save.

        **A SIBLING of `assert_dense_panel_fits()` / `assert_chunked_panel_
        fits()`, never a replacement.** Those bound RAM for a dense panel;
        this bounds disk, request count and wall clock, which are three
        independent quantities. Any one alone lets a real scenario through: a
        request-count check passes a tick fetch that fills the volume, and a
        byte check passes a small, slow, many-request fetch that runs
        overnight.

        Each `max_*` keyword defaults to `None` meaning "use the class
        constant", so a caller reading `config.kwargs` can raise ONE ceiling
        deliberately without disturbing the others. That is the deliberate
        path. `force=True` is the blunt one: it skips the RAISE and never the
        arithmetic, and it is an explicit named parameter -- there is no
        environment variable and no config key that disables the guard
        wholesale (T-03.2-21).
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
            if label == "raw-bytes":
                return f"{actual / gib:.2f} GiB > {ceiling / gib:.2f} GiB"
            if label == "request":
                return f"{actual:,.0f} > {ceiling:,.0f}"
            return f"{actual:.1f} h > {ceiling:.1f} h"

        # NOT `"; ".join(...).capitalize()`: `str.capitalize()` LOWERCASES the
        # rest of the string, which would render `GiB` as `gib` and
        # `MAX_RAW_BYTES` as `max_raw_bytes` -- corrupting the unit and the
        # constant name a reader is meant to go and edit.
        reasons = "Over the " + "; over the ".join(
            f"{label} ceiling ({_render(label, actual, ceiling)}, {constant})"
            for label, _key, actual, ceiling, constant, _kw in crossed
        )
        # Only the CROSSED ceilings' keywords are offered. Listing all three
        # every time would tell a user to raise a ceiling they are nowhere
        # near, which is how a refusal teaches the habit of raising everything.
        keywords = " / ".join(keyword for *_rest, keyword in crossed)
        narrowed_end, narrowed, max_symbols = self._narrowing_that_fits(
            estimate, overshoot, ceilings, rows_per_symbol_day
        )
        if narrowed is None:
            # No window is short enough: the per-batch floor alone is over a
            # ceiling, so only a smaller roster (or a bigger batch) can help.
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
        # `category` and `as_of_date` arrive unvalidated -- `as_of_date` comes
        # straight off `ingest_tiingo.py`'s `--as-of-date` CLI argument. Both
        # are validated HERE because every wrong input otherwise produces `[]`,
        # which is a LEGITIMATE return value (a pre-listing `nasdaq_all` query
        # returns it), so the caller cannot distinguish "no members" from "you
        # asked wrong". A typo'd category or a non-ISO date would silently
        # ingest nothing instead of the requested index -- the same class of
        # silent-wrong-answer the coverage-start guard below raises to prevent.
        self._validate_category(category)
        as_of_date = self._normalize_iso_date(as_of_date, "as_of_date")

        # Every registered membership category carries its own boundary; a
        # category absent from the registry (i.e. either exchange roster,
        # `nasdaq_all` and `us_all`) is boundary-free by design (D-02).
        # The lookup and the raise live in `_assert_within_coverage()` because
        # `get_symbols_in_range()` needs the identical boundary: one helper,
        # so the two membership queries cannot drift into two contracts.
        self._assert_within_coverage(category, as_of_date, "as_of_date")

        matched = self._backend.get_lazyframe().filter(
            (pl.col("category") == category)
            & (pl.col("start_date") <= as_of_date)
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= as_of_date))
        )
        return matched.select("symbol").unique().collect()["symbol"].to_list()
