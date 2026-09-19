import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class BinanceCSVHeaders:
    SPOT = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]


# Locked literal token sets shared by base/config.py, config/__init__.py, and
# base/acquisition.py (02-RESEARCH.md Assumptions Log A2) — do not add values
# without revisiting that decision.
Market = Literal["us_equity", "crypto_spot"]
Frequency = Literal["1d", "1m", "tick"]

# The bar size of a panel RESAMPLED LOCALLY from tick records -- a separate
# noun from `Frequency` above, which stays the ACQUISITION frequency and stays
# locked. This is the D-08 option "separate panel frequency parameter", chosen
# by 03.9 D-26: extending `Frequency` would break every table keyed on it
# (`RAW_HIVE_KEYS`, `StockDataset.HIVE_SCHEMA_BY_FREQUENCY`, the ingest
# scripts' choices) for tokens only one Dataset understands. Carried by
# `NbboDatasetConfig.bar_interval`.
#
# Every token divides BOTH a 390-minute regular session and a 210-minute
# half day, so a bar never straddles the close on either kind of day.
BarInterval = Literal[
    "1s", "5s", "10s", "15s", "30s", "1m", "5m", "10m", "15m", "30m"
]

#: `BarInterval` token -> its length in seconds.
BAR_INTERVAL_SECONDS: dict[str, int] = {
    "1s": 1,
    "5s": 5,
    "10s": 10,
    "15s": 15,
    "30s": 30,
    "1m": 60,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "30m": 1800,
}

# US-equity universe reference categories (02-08-PLAN.md / 02-CONTEXT.md D-12).
# "nasdaq_all" = every symbol ever listed on NASDAQ as Common Stock priced in
# USD per Tiingo's supported_tickers.csv (current + delisted, via
# start_date/end_date). "sp500_constituent" = point-in-time S&P 500
# membership reconstructed from Wikipedia's historical-components table.
# "nasdaq100_constituent" = point-in-time Nasdaq-100 (NDX) membership,
# reconstructed from Wikipedia's Nasdaq-100 historical-components change log
# anchored against a current-constituent snapshot (03.1-CONTEXT.md D-02).
#
# WARNING: "nasdaq100_constituent" is NOT "nasdaq_all". They are different
# universes that merely share the word "Nasdaq": the former is the ~100-name
# NDX index (~350 in the all-time union, left-censored at 2007-02-01), the
# latter is every symbol ever listed on the NASDAQ *exchange* (10,000+, no
# index membership concept at all). 03.1-CONTEXT.md D-02 records this exact
# confusion being raised with the user and resolved deliberately -- do not
# "unify" the two, and do not answer an index-membership question with
# "nasdaq_all".
#
# "us_all" = every symbol ever listed on NYSE, NASDAQ or the AMEX as Common
# Stock priced in USD per Tiingo's supported_tickers.csv (current + delisted),
# ~15.4k distinct tickers (260906-0iy D-01).
#
# WARNING: "us_all" is a strict SUPERSET of "nasdaq_all", and both are
# DELIBERATELY retained (260906-0iy D-01/D-02). "nasdaq_all" must NOT be
# redefined as an alias of "us_all" and its NASDAQ-only semantics must not be
# widened: `NasdaqUniverseFetcher.EXCHANGE_FILTER == ("NASDAQ",)` is pinned by
# direct equality in tests/test_universe.py. The full-market roster is a NEW
# SIBLING fetcher, never a widening of the existing one -- code already
# written against "nasdaq_all" keeps resolving the exact symbol set it
# always did.
#
# NASDAQ exchange scope for "nasdaq_all" is locked per Locked Decision A4
# (02-08-PLAN.md) -- NASDAQ only, no OTC/Expert-Market tiers. "us_all" adds
# NYSE and the AMEX (which Tiingo spells under BOTH "AMEX" and "NYSE MKT",
# unmigrated across the exchange's rename history) but still excludes every
# OTC tier, plus "NYSE ARCA"/"NYSE NAT"/"BATS", which are different exchanges.
UniverseCategory = Literal[
    "nasdaq_all", "us_all", "sp500_constituent", "nasdaq100_constituent"
]


@dataclass
class TiingoColumns:
    """Explicit `columns=` value passed to `TiingoClient.get_ticker_price()`.

    Always pass this explicitly -- never rely on the client's undocumented
    default field set (02-RESEARCH.md Pitfall 3): StockDataset._to_kunquant()
    unconditionally expects the `adj*` columns to be present, which fails
    downstream (not at acquisition time) if they are silently omitted.
    """

    EOD = (
        "open,high,low,close,volume,adjOpen,adjHigh,adjLow,adjClose,"
        "adjVolume,divCash,splitFactor"
    )


# Locked vendor token set (03.2-CONTEXT.md D-10/D-11) — the same discipline as
# `Market`/`Frequency` above: do not add a value without revisiting
# 02-RESEARCH.md Assumptions Log A2. A vendor token is not merely a label; it
# is a PATH SEGMENT (`downloads/{market}/{frequency}/{subdir}/{vendor}/`) and a
# literal column written into every raw shard, so adding one commits to an
# on-disk layout that cannot be renamed without relaying the raw tier.
Vendor = Literal["tiingo", "alpaca", "wrds"]

# Hive partition key(s) per frequency for the raw tier (D-08 / D-19).
#
# ONE definition, imported by BOTH the writer (`base/acquisition.py`'s
# `_hive_partition_values`) and the reader (`dataset/stock.py`'s
# `_hive_window_predicate`), so the two cannot drift. A writer and a reader
# that disagree about the key produce a scan that prunes nothing and silently
# returns fewer rows than the tree holds.
#
# WHY `month=` FOR DAILY rather than the `date=/symbol=` D-08 illustrates.
# 03.2-RESEARCH.md Pattern 5 did the arithmetic against the real `us_all`
# roster (15,424 symbols x ~2,690 trading days over 2016-2026):
#
#   date=/symbol=  ->  2,690 x 15,424  ~= 41,000,000 leaf directories, each
#                      holding a one-row file. Inode exhaustion on most
#                      filesystems before the backfill finishes.
#   date=          ->  2,690 dirs x ~155 batches ~= 417,000 files of ~90 rows.
#                      Survivable but pathological.
#   month=         ->  ~130 dirs; ~155 batches x 130 ~= 20,000 files of ~800
#                      rows each. This is the right size, and `month=` was
#                      MEASURED to prune correctly (2 of 5 files opened).
#
# For `tick` the calculus inverts: one symbol-day of quotes is large enough to
# justify its own directory and `symbol=` pruning genuinely pays.
#
# WHY `tick` CARRIES A LEADING `data_type=` KEY that RESEARCH's recommended
# layout does not. Quotes and trades have DIFFERENT column sets, so without
# that key a directory scan of one tick root meets two schemas and raises.
# Expressing the distinction as a hive KEY rather than as two new `Frequency`
# tokens leaves this file's locked literal set untouched and keeps both
# prunable. Only the tick writer/reader added in 03.2-06 consumes it; `1d` and
# `1m` are unaffected. This is a planner refinement taken under
# 03.2-CONTEXT.md's explicit grant of partition-key choice to Claude's
# discretion, confirmed by the developer as D-19 contract 5/6.
#
# NOTE for the intraday keys: the `date=` VALUE is the US/Eastern SESSION date,
# not the naive-UTC date (D-19 contract 7). Timestamp values stay naive UTC and
# unchanged; only the derived partition key converts. A UTC-derived key files
# the last ~4 hours of every US session (20:00-24:00 UTC) under the FOLLOWING
# day, making a one-trading-day query wrong at both edges in a way that looks
# like sparse data rather than like a bug.
RAW_HIVE_KEYS: dict[str, tuple[str, ...]] = {
    "1d": ("month",),
    "1m": ("date",),
    "tick": ("data_type", "date", "symbol"),
}


# The ONE well-formedness rule a US-equity ticker must satisfy before it
# becomes a filesystem path segment or a query-string value.
#
# Declared HERE, and bound by BOTH ends of the symbol lifecycle -- the roster
# builder (`acquisition/universe.py:TiingoRosterFetcher.fetch`) and the fetch
# guard (`base/acquisition.py:Acquisition._validate_symbols`) -- for exactly
# the reason `RAW_HIVE_KEYS` above lives here (D-19): one definition, imported
# by the writer and the reader, so the two cannot drift.
#
# They HAD drifted. `base/acquisition.py` carried a standalone `re.compile` of
# the same literal while its comment claimed the pattern was imported from
# `acquisition/universe.py`, and the guard admitted only ONE suffix segment
# while `us_all` deliberately retained 1,124 warrant / unit / right /
# when-issued lines (260906-eme). The result was that `download()`'s
# whole-roster pre-flight raised on `NXG-R-W` and killed a multi-hour
# full-market job before a single request was issued (quick task 260907-10t).
#
# WHY THIS MODULE. Neither of the two binders may import the other:
# `quantlab/base/acquisition.py` importing the universe module inverts the
# layering, and `quantlab/acquisition/universe.py` importing the acquisition
# base breaks `tests/test_volume_guard.py`, which resolves universe.py's
# imports with `ast` and asserts none of them is an acquisition module -- the
# volume guard's "refuse before any client exists" property is STRUCTURAL.
# `quantlab/enums/data.py` is the shared lower module both already import.
#
# THE SUFFIX BOUND IS `{0,2}`, MEASURED. Against the live reference table on
# 2026-09-07 (`us_all` 14,485 unique symbols, `nasdaq_all` 8,967):
#
#   77 `us_all` symbols are three-segment `ROOT-X-Y` with each suffix 1-2
#      chars -- `ACP-R-W` (47 of shape 3-1-1), `AST-WS-W`, `KODK-WS-A`,
#      `GM-WS-A`, `DB-R-W`, `UA-C-W`, `C-WS-A`.
#    4 `nasdaq_all` symbols likewise: `FITB-P-A/-I/-K/-M`, the frozen
#      preferred shares Locked Decision A4 / D-02 keeps. They are ADMITTED by
#      this bound, never dropped -- the roster filter's axis is
#      well-formedness, never security type.
#
# WHY THE SEGMENT BOUND STAYS `{1,2}` AND NOT `{1,3}`. Exactly two live
# symbols would need `{1,3}`, and both were argued rather than assumed:
#   - `OXY-WSW`: the family is `['OXY', 'OXY-WS', 'OXY-WS-W', 'OXY-WSW']`.
#     The properly-delimited form of the same warrant is ALREADY in the
#     roster, so refusing the un-delimited variant loses no security at all.
#   - `NSPR-WSB`: the family is `['NSPR', 'NSPR-WS', 'NSPR-WSB']` -- there is
#     no `NSPR-WS-B`, so refusing it DOES lose one microcap warrant series.
#     Accepted deliberately and recorded as a named, measured,
#     carried-forward finding in the same idiom 260906-eme used for the
#     retained warrants: a 50% loosening of the segment bound for all 14,485
#     symbols, to save one redundant and one microcap warrant, against 77
#     symbols that follow the delimiter convention, does not clear the bar.
# A later widener should confront that argument, not the regex.
#
# WHAT IS DELIBERATELY UNCHANGED. Only the suffix REPETITION COUNT widened
# (from the prior `?`). The character class `[A-Z0-9]`, the delimiter class
# `[.-]`, the 1-7 root bound (the longest delimiter-free live symbol is 7:
# `ALLPDCL`, `ALLYPRA`) and BOTH anchors are untouched -- which is what keeps
# a path separator, a parent reference, an embedded comma, a space and
# lowercase unrepresentable rather than merely unmatched (T-10t-01).
#
# NOT THE SAME AS `acquisition/universe.py:_WELL_FORMED_TICKER`, which is
# DELIBERATELY NARROWER and must not be "aligned" with this one. That one
# validates Wikipedia-scraped change-log CELLS, where an interior delimiter
# means two cells were merged by a parser regression; a three-segment value is
# that exact signal on that input, and both constituent categories have ZERO
# pattern failures across 1,151 measured symbols. Pinned by
# `tests/test_ticker_pattern_reconciliation.py:
# test_the_changelog_guard_is_deliberately_narrower_than_the_fetch_guard`.
TRADEABLE_TICKER_PATTERN = re.compile(
    r"^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2}){0,2}$"
)
