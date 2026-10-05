"""Offline stand-ins for Sharadar's API (`api.sharadar.com/v1.0`).

Nothing here touches the network. `FakeTransport` replaces the one transport
function `SharadarClient` sends every HTTP call through, and answers each
request from a script of responses.

**Provenance rule for every value in this module.** Column names and their
order are VERBATIM from Sharadar's published PostgreSQL schemas
(`GET api.sharadar.com/v1.0/schema/{table}?format=postgres`, "As of
2026-08-18"). The `tickers.table` values are VERBATIM too: the bulk TICKERS
file labels rows with the legacy codes (`SEP`, `SFP`, `SF1`, ...; live bulk
pull, 2026-10-05), while the REST API labels them with the API names
(`stocks`, `funds`, ...). Every row value is invented for the test and carries
a `# SYNTHETIC` comment. No row returned by Sharadar, including from its public
test key, is in this repository: the data is licensed.
"""

from __future__ import annotations

import io
import threading
import zipfile
from dataclasses import dataclass, field
from urllib.parse import urlencode

#: VERBATIM `stocks` (SEP) column order.
SEP_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closeadj",
    "closeunadj",
    "lastupdated",
)

#: VERBATIM `tickers` column order.
TICKERS_COLUMNS: tuple[str, ...] = (
    "table",
    "permaticker",
    "ticker",
    "name",
    "exchange",
    "isdelisted",
    "category",
    "cusips",
    "siccode",
    "sicsector",
    "sicindustry",
    "figi",
    "famaindustry",
    "sector",
    "industry",
    "scalemarketcap",
    "scalerevenue",
    "relatedtickers",
    "currency",
    "location",
    "lastupdated",
    "firstadded",
    "firstpricedate",
    "lastpricedate",
    "firstquarter",
    "lastquarter",
    "secfilings",
    "companysite",
)

#: VERBATIM `actions` column order.
ACTIONS_COLUMNS: tuple[str, ...] = (
    "date",
    "action",
    "ticker",
    "name",
    "value",
    "contraticker",
    "contraname",
)

#: VERBATIM `descriptions` (INDICATORS) column order.
INDICATORS_COLUMNS: tuple[str, ...] = (
    "table",
    "indicator",
    "isfilter",
    "isprimarykey",
    "title",
    "description",
    "unittype",
)


def csv_text(columns: tuple[str, ...], rows: list[dict]) -> str:
    """Render ``rows`` as Sharadar CSV: a header, then one line per row.

    A key missing from a row is an empty field, which is how the vendor writes
    a null.
    """
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join("" if row.get(c) is None else str(row[c]) for c in columns))
    return "\n".join(lines) + "\n"


def bulk_zip(member: str, text: str) -> bytes:
    """Return a zip archive holding one CSV member, as a bulk download does."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


def sep_row(ticker: str, date: str, close: float, **overrides) -> dict:
    """One SEP row whose split-adjusted and raw prices agree (no split since).

    ``open``/``high``/``low`` sit around ``close``; ``closeadj`` is slightly
    below it, as a later dividend would put it.
    """
    row = {
        "ticker": ticker,
        "date": date,
        "open": close - 1.0,  # SYNTHETIC
        "high": close + 2.0,  # SYNTHETIC
        "low": close - 2.0,  # SYNTHETIC
        "close": close,  # SYNTHETIC
        "volume": 1000.0,  # SYNTHETIC
        "closeadj": close * 0.99,  # SYNTHETIC
        "closeunadj": close,  # SYNTHETIC
        "lastupdated": "2026-08-10",  # SYNTHETIC
    }
    row.update(overrides)
    return row


def action_row(date: str, action: str, ticker: str, value: float | None) -> dict:
    """One ACTIONS row. ``N/A`` is how the vendor fills an unused contra column."""
    return {
        "date": date,
        "action": action,  # VERBATIM action type, e.g. `dividend`, `split`
        "ticker": ticker,
        "name": f"{ticker} CORP",  # SYNTHETIC
        "value": value,  # SYNTHETIC
        "contraticker": "N/A",
        "contraname": "N/A",
    }


def tickers_row(table: str, permaticker: int, ticker: str, **overrides) -> dict:
    """One TICKERS row; every value but the key is SYNTHETIC filler."""
    row = {
        "table": table,
        "permaticker": permaticker,  # SYNTHETIC
        "ticker": ticker,  # SYNTHETIC
        "name": f"{ticker} CORP",  # SYNTHETIC
        "exchange": "NYSE",  # SYNTHETIC
        "isdelisted": "N",  # SYNTHETIC
        "category": "Domestic Common Stock",  # SYNTHETIC
        "siccode": 3571,  # SYNTHETIC
        "currency": "USD",  # SYNTHETIC
        "lastupdated": "2026-08-10",  # SYNTHETIC
        "firstpricedate": "2020-01-02",  # SYNTHETIC
        "lastpricedate": "2026-08-10",  # SYNTHETIC
    }
    row.update(overrides)
    return row


INDICATORS_ROWS: list[dict] = [
    {
        "table": "stocks",
        "indicator": "closeunadj",
        "isfilter": "N",  # SYNTHETIC
        "isprimarykey": "N",  # SYNTHETIC
        "title": "Close Unadjusted",  # SYNTHETIC
        "description": "Synthetic description",  # SYNTHETIC
        "unittype": "USD",  # SYNTHETIC
    },
]


@dataclass
class Reply:
    """One scripted answer: an HTTP status, headers and a body."""

    status: int
    body: bytes = b""
    headers: dict = field(default_factory=dict)
    #: Whether a 200 reply answers a ``Range`` request with the 206 slice, as
    #: file storage does. ``False`` ignores the header and sends everything.
    ranges: bool = True
    #: Whether the connection drops after the first half of the body, as a
    #: broken stream does.
    breaks: bool = False


@dataclass
class Call:
    """One request the client sent through the transport."""

    url: str
    params: dict
    headers: dict


class FakeTransport:
    """A transport function answering from scripted replies, recording each call.

    ``routes`` maps a URL (without its query string) to a list of replies, served in
    order; the last reply repeats once the list runs out.
    """

    def __init__(self, routes: dict[str, list[Reply]]):
        self.routes = {url: list(replies) for url, replies in routes.items()}
        self.calls: list[Call] = []
        #: How many responses the client closed.
        self.closed = 0
        # The client may call from several threads at once.
        self._lock = threading.Lock()

    def __call__(self, url: str, *, params: dict, headers: dict):
        from quantlab.acquisition.sharadar.client import Response

        route = url.split("?")[0]
        with self._lock:
            self.calls.append(Call(url, dict(params), dict(headers)))
            if route not in self.routes:
                raise AssertionError(
                    f"unexpected request {url} {urlencode(params)}; scripted: {sorted(self.routes)}"
                )
            replies = self.routes[route]
            reply = replies.pop(0) if len(replies) > 1 else replies[0]
        status, body, reply_headers = reply.status, reply.body, dict(reply.headers)
        requested = headers.get("Range")
        if requested and status == 200 and reply.ranges:
            first, last = (int(v) for v in requested.removeprefix("bytes=").split("-"))
            last = min(last, len(body) - 1)
            status, body = 206, body[first : last + 1]
            reply_headers["Content-Range"] = f"bytes {first}-{last}/{len(reply.body)}"
        # Two chunks, so the client must join a streamed body.
        half = len(body) // 2
        return Response(
            status=status,
            headers=reply_headers,
            body=_broken(body[:half]) if reply.breaks else iter([body[:half], body[half:]]),
            close=self._close,
        )

    def _close(self) -> None:
        with self._lock:
            self.closed += 1


def _broken(first: bytes):
    """Yield ``first``, then fail as a dropped connection does."""
    from quantlab.acquisition.sharadar.client import SharadarTransportError

    yield first
    raise SharadarTransportError("connection broken (SYNTHETIC)")


API = "https://api.sharadar.com/v1.0/data"
#: A pre-signed bulk URL, as the 302 points at. Invented.
SIGNED = "https://bulk.example.invalid/{table}.csv.zip?signature=SYNTHETIC"


def bulk_routes(tables: dict[str, str]) -> dict[str, list[Reply]]:
    """Routes answering a bulk pull of each ``{api_table: csv_text}`` with 302 then the zip."""
    routes: dict[str, list[Reply]] = {}
    for api_table, text in tables.items():
        signed = SIGNED.format(table=api_table)
        routes[f"{API}/{api_table}"] = [Reply(302, headers={"Location": signed})]
        routes[signed.split("?")[0]] = [Reply(200, bulk_zip(f"{api_table}.csv", text))]
    return routes
