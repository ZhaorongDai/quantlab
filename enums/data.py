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
