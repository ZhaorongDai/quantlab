"""Offline stand-ins for Massive's daily files and its condition table.

Nothing here touches the network. **Provenance rule.** File names, S3 keys,
column names and their order are VERBATIM from Massive's flat files
(``us_stocks_sip/trades_v1`` and ``minute_aggs_v1``, read 2026-10-09). The
condition ids, names, ``data_types`` and consolidated update rules in
``CONDITION_RECORDS`` are VERBATIM from ``/v3/reference/conditions``
(2026-10-09), a subset of the 94 the endpoint lists. Every trade, price,
size and ticker is invented for the test (``# SYNTHETIC``); no row of a
Massive file is in this repository: the data is licensed.
"""

from __future__ import annotations

import gzip
import io
import json
from datetime import date, datetime
from pathlib import Path

import polars as pl


def _rules(high_low: bool, open_close: bool, volume: bool) -> dict:
    flags = {"updates_high_low": high_low, "updates_open_close": open_close, "updates_volume": volume}
    return {"consolidated": dict(flags), "market_center": dict(flags)}


#: VERBATIM condition records (the fields the conversion reads).
CONDITION_RECORDS: list[dict] = [
    {"id": 1, "type": "sale_condition", "name": "Acquisition", "asset_class": "stocks",
     "update_rules": _rules(True, True, True), "data_types": ["trade"]},
    {"id": 2, "type": "sale_condition", "name": "Average Price Trade", "asset_class": "stocks",
     "update_rules": _rules(False, False, True), "data_types": ["trade"]},
    {"id": 7, "type": "sale_condition", "name": "Cash Sale", "asset_class": "stocks",
     "update_rules": _rules(False, False, True), "data_types": ["trade"]},
    {"id": 12, "type": "sale_condition", "name": "Form T/Extended Hours", "asset_class": "stocks",
     "update_rules": _rules(False, False, True), "data_types": ["trade"]},
    {"id": 14, "type": "sale_condition", "name": "Intermarket Sweep", "asset_class": "stocks",
     "update_rules": _rules(True, True, True), "data_types": ["trade"]},
    {"id": 16, "type": "sale_condition", "name": "Market Center Official Open", "asset_class": "stocks",
     "update_rules": _rules(False, False, False), "data_types": ["trade"]},
    {"id": 32, "type": "sale_condition", "name": "Sold (Out Of Sequence)", "asset_class": "stocks",
     "update_rules": _rules(True, False, True), "data_types": ["trade"]},
    {"id": 37, "type": "sale_condition", "name": "Odd Lot Trade", "asset_class": "stocks",
     "update_rules": _rules(False, False, True), "data_types": ["trade"]},
    {"id": 38, "type": "sale_condition", "name": "Corrected Consolidated Close (per listing market)",
     "asset_class": "stocks", "update_rules": _rules(True, True, False), "data_types": ["trade"]},
    {"id": 41, "type": "trade_thru_exempt", "name": "Trade Thru Exempt", "asset_class": "stocks",
     "update_rules": _rules(True, True, True), "data_types": ["trade"]},
    {"id": 60, "type": "short_sale_restriction_indicator", "name": "Short Sale Restriction In Effect",
     "asset_class": "stocks", "data_types": ["trade"]},
    # Ids are reused across data types: these two are quote conditions.
    {"id": 12, "type": "quote_condition", "name": "Manual Bid and Ask", "asset_class": "stocks",
     "data_types": ["bbo", "nbbo"]},
    {"id": 999, "type": "quote_condition", "name": "SYNTHETIC quote-only id", "asset_class": "stocks",
     "data_types": ["nbbo"]},
]

#: VERBATIM header of a trade file.
TRADE_HEADER = (
    "ticker,conditions,correction,exchange,id,participant_timestamp,price,"
    "sequence_number,sip_timestamp,size,tape,trf_id,trf_timestamp"
)

#: VERBATIM header of a minute- or day-aggregate file.
AGGREGATE_HEADER = "ticker,volume,open,close,high,low,window_start,transactions"


def conditions_frame() -> pl.DataFrame:
    """``CONDITION_RECORDS`` as ``read_conditions`` returns them."""
    import tempfile

    from quantlab.dataset.massive.raw import read_conditions

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "conditions.json"
        path.write_text(json.dumps(CONDITION_RECORDS), encoding="utf-8")
        return read_conditions(path)


def ns(moment: datetime) -> int:
    """Nanoseconds since the epoch of a naive-UTC ``moment``."""
    return int((moment - datetime(1970, 1, 1)).total_seconds() * 1_000_000) * 1000


def trade_line(
    ticker: str,
    moment: datetime,
    price: float,
    size: float,
    *,
    conditions: str = "",
    correction: int = 0,
    sequence: int = 1,
    trf_id: int = 0,
) -> str:
    """One trade-file line; ``moment`` is naive UTC. Every value is SYNTHETIC."""
    quoted = f'"{conditions}"' if "," in conditions else conditions
    stamp = ns(moment)
    return (
        f"{ticker},{quoted},{correction},12,{sequence}00,{stamp - 300},{price},"
        f"{sequence},{stamp},{size:g},1,{trf_id},0"
    )


def gzip_text(lines: list[str]) -> bytes:
    """The gzip bytes of ``lines`` joined by newlines, with a trailing one."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(("\n".join(lines) + "\n").encode())
    return buffer.getvalue()


def write_trades(vendor_root: Path, day: date, lines: list[str]) -> Path:
    """Write one day's trade file into the raw tier."""
    from quantlab.dataset.massive.raw import raw_file

    path = raw_file(vendor_root, "trades", day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip_text([TRADE_HEADER, *lines]))
    return path


def write_conditions(vendor_root: Path, pulled_at: datetime, records: list[dict] | None = None) -> Path:
    """Write a condition-table snapshot into the raw tier."""
    from quantlab.dataset.massive.raw import conditions_file

    path = conditions_file(vendor_root, pulled_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(CONDITION_RECORDS if records is None else records), encoding="utf-8")
    return path


def aggregate_line(ticker: str, start: datetime, o: float, h: float, low: float, c: float, volume: float,
                   transactions: int) -> str:
    """One minute-aggregate line; ``start`` is the bar's naive-UTC start. Every value is SYNTHETIC."""
    return f"{ticker},{volume:g},{o},{c},{h},{low},{ns(start)},{transactions}"


def write_aggregates(vendor_root: Path, day: date, lines: list[str], data_type: str = "minute_aggs") -> Path:
    """Write one day's minute- (or day-) aggregate file into the raw tier."""
    from quantlab.dataset.massive.raw import raw_file

    path = raw_file(vendor_root, data_type, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip_text([AGGREGATE_HEADER, *lines]))
    return path
