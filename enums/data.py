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
# Exchange scope is locked per Locked Decision A4 (02-08-PLAN.md) -- NASDAQ
# only, no OTC/Expert-Market tiers.
UniverseCategory = Literal[
    "nasdaq_all", "sp500_constituent", "nasdaq100_constituent"
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
