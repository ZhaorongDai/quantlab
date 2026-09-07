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
Vendor = Literal["tiingo", "alpaca"]

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
