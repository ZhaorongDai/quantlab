"""Shared literal types, column layouts and naming rules for market data.

The ``Literal`` aliases here (``Market``, ``Frequency``, ``Vendor``,
``BarInterval``, ``UniverseCategory``) are the vocabulary every config,
acquisition and dataset class agrees on. Several of them are also written to
disk, as path segments of the raw data tree or as columns of raw shards, so
adding a token is an on-disk format decision and not just a new label.
"""

import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class BinanceCSVHeaders:
    """Column names of the raw Binance kline CSV files, which ship without a header.

    Example:
        >>> BinanceCSVHeaders.SPOT[:4]
        ['Open time', 'Open', 'High', 'Low']
        >>> len(BinanceCSVHeaders.SPOT)
        12
    """

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


#: The markets a dataset or acquisition can belong to. Shared by the config
#: dataclasses, the config factories and the acquisition base class.
Market = Literal["us_equity", "crypto_spot"]

#: The acquisition frequency of a raw data tree: daily bars, minute bars or
#: raw tick records.
Frequency = Literal["1d", "1m", "tick"]

#: The bar size of a panel resampled locally from tick records. This is a
#: separate notion from ``Frequency``, which stays the acquisition frequency,
#: so that tables keyed on ``Frequency`` are not widened by tokens only one
#: dataset understands. Every token divides both a 390-minute regular session
#: and a 210-minute half day, so a bar never straddles the close.
BarInterval = Literal[
    "1s", "5s", "10s", "15s", "30s", "1m", "5m", "10m", "15m", "30m"
]

#: ``BarInterval`` token to its length in seconds.
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

#: The US-equity symbol universes a roster can be built for.
#:
#: ``"nasdaq_all"`` is every symbol ever listed on the NASDAQ exchange as
#: common stock priced in USD (current and delisted), per Tiingo's supported
#: tickers file, and ``"us_all"`` is the same across NYSE, NASDAQ and AMEX
#: (roughly 15k tickers; OTC tiers and the NYSE ARCA, NYSE NAT and BATS
#: venues are excluded). ``"us_all"`` is a strict superset of ``"nasdaq_all"``
#: and the two are kept as separate categories.
#:
#: ``"sp500_constituent"`` and ``"nasdaq100_constituent"`` are point-in-time
#: index memberships reconstructed from Wikipedia's historical-components
#: tables. ``"nasdaq100_constituent"`` (about 100 names at a time, roughly 350
#: in the all-time union) is unrelated to ``"nasdaq_all"`` (10,000+ names,
#: no index concept); they merely share a word.
UniverseCategory = Literal[
    "nasdaq_all", "us_all", "sp500_constituent", "nasdaq100_constituent"
]


@dataclass
class TiingoColumns:
    """Explicit ``columns=`` value passed to ``TiingoClient.get_ticker_price()``.

    Always pass this rather than relying on the client's default field set:
    the stock dataset expects the ``adj*`` columns to be present, and their
    absence only surfaces at conversion time, not at download time.

    Example:
        >>> TiingoColumns.EOD.split(",")[:4]
        ['open', 'high', 'low', 'close']
    """

    EOD = (
        "open,high,low,close,volume,adjOpen,adjHigh,adjLow,adjClose,"
        "adjVolume,divCash,splitFactor"
    )


#: The data vendors acquisition can download from. A vendor token is a path
#: segment of the raw tree (``downloads/{market}/{frequency}/{subdir}/{vendor}/``)
#: and a literal column in every raw shard, so it cannot be renamed later.
Vendor = Literal["tiingo", "alpaca", "wrds"]

#: Hive partition keys of the raw tier, per frequency. One definition, imported
#: by both the writer (the acquisition base class) and the reader (the stock
#: dataset), so the two cannot disagree; a writer and a reader that disagree
#: about the key produce a scan that prunes nothing and silently returns fewer
#: rows than the tree holds.
#:
#: Daily data is partitioned by ``month`` rather than by date or symbol: a
#: full-market roster over a decade would otherwise produce tens of millions
#: of one-row files. For tick data one symbol-day is large enough to justify
#: its own directory, and the leading ``data_type`` key separates quotes from
#: trades, which have different column sets. For the intraday keys the
#: ``date`` value is the US/Eastern session date, not the naive-UTC date, so a
#: one-trading-day query does not lose the last hours of the session to the
#: following day; timestamp values themselves stay naive UTC.
RAW_HIVE_KEYS: dict[str, tuple[str, ...]] = {
    "1d": ("month",),
    "1m": ("date",),
    "tick": ("data_type", "date", "symbol"),
}


#: The well-formedness rule a US-equity ticker must satisfy before it becomes
#: a filesystem path segment or a query-string value: an uppercase
#: alphanumeric root of 1 to 7 characters followed by up to two suffix segments
#: of 1 to 2 characters, each introduced by ``.`` or ``-``.
#:
#: It is defined here because both the roster builder and the acquisition
#: fetch guard bind it and neither may import the other. Two suffix segments
#: are allowed because the full-market roster deliberately retains warrant,
#: unit and right lines such as ``ACP-R-W``; the anchors and character classes
#: keep path separators, spaces, commas and lowercase unrepresentable. This
#: pattern is intentionally wider than the one the universe module uses to
#: validate scraped change-log cells, where an interior delimiter signals a
#: parsing error.
TRADEABLE_TICKER_PATTERN = re.compile(
    r"^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2}){0,2}$"
)
