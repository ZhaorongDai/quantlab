"""A fake StockFit vendor that replays the recorded responses in ``tests/fixtures/stockfit``.

``FakeVendor`` is a transport for ``StockFitClient(http_get=...)``: it is
called as ``vendor(url, params=..., headers=...)`` and returns an
``HttpResponse`` the way ``https://api.stockfit.io/v1`` would:

- a per-entity request (``/api/financials/income-statement?cik=...&period=...``)
  is answered with the recorded body whose path and identifying parameters
  (every parameter but ``page``, ``pageSize`` and ``limit``) match, and with
  404 when nothing was recorded for it;
- a market-wide list (``/api/symbols``, ``/api/company/delisted``, the
  declared split and dividend calendars) is served
  from ``vendor.lists[path]``, filtered on the query parameters that are
  fields of its rows (``status``, ``type``) and paged by ``page`` and
  ``pageSize`` like the vendor's paged objects;
- every response carries the ``X-RateLimit-*`` headers, computed from the
  calls made in the last 60 seconds of ``clock`` and from a daily budget; a
  call past either limit gets 429;
- a request without ``Authorization: Bearer <KEY>`` gets 401.

``vendor.bodies`` and ``vendor.lists`` can be edited to make the vendor serve
a changed response (a restatement, a new company). ``replies`` are canned
answers served first, an exception being raised instead of answered.

Every value written here (the key, the list rows) is SYNTHETIC; the recorded
bodies are VERBATIM StockFit responses.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

FIXTURES = Path(__file__).parent / "fixtures" / "stockfit"

KEY = "synthetic-stockfit-key"  # SYNTHETIC

#: Parameters that page or bound a response rather than say what it is of.
PAGING = frozenset({"page", "pageSize", "limit"})

#: The CIKs of the recorded companies.
PLUG, TSM, TWTR, GOOGL = 1093691, 1046179, 1418091, 1652044


def fixture(name: str):
    """Return one recorded response body."""
    return json.loads((FIXTURES / f"{name}.json").read_text())


def company_row(cik: int, name: str, status: str = "operating", kind: str = "stock") -> dict:
    """Return one row of ``/api/symbols`` in the vendor's shape."""
    return {  # SYNTHETIC values, VERBATIM field names
        "symbols": [] if status == "delisted" else [name[:4].upper()],
        "exchanges": [] if status == "delisted" else ["Nasdaq"],
        "name": name,
        "cik": cik,
        "seriesId": 0,
        "type": kind,
        "sector": "Industrials",
        "industry": "Electrical Equipment",
        "industryGroup": None,
        "compositeFigi": None,
        "ein": None,
        "formerNames": [],
        "status": status,
        "priorTickers": [],
    }


def _identity(params) -> frozenset:
    return frozenset((str(k), str(v)) for k, v in params.items() if k not in PAGING)


class FakeClock:
    """A UTC clock that only moves when slept on or advanced."""

    def __init__(self, start: datetime = datetime(2026, 11, 16, 14, tzinfo=UTC)):
        self.at = start
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.at

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.at += timedelta(seconds=seconds)

    def advance(self, **delta) -> None:
        self.at += timedelta(**delta)


class FakeVendor:
    """StockFit's API over the recorded fixtures; see the module docstring."""

    def __init__(
        self,
        clock: FakeClock,
        *,
        companies: list[dict] | None = None,
        per_minute: int = 60,
        per_day: int = 25_000,
        replies: list | None = None,
    ):
        index = json.loads((FIXTURES / "index.json").read_text())
        self.clock = clock
        self.bodies: dict[tuple[str, frozenset], object] = {
            (entry["path"], _identity(entry["params"])): fixture(name) for name, entry in index.items()
        }
        default = [
            company_row(PLUG, "Plug Power Inc"),
            company_row(TSM, "Taiwan Semiconductor Manufacturing Co Ltd"),
            company_row(TWTR, "Twitter, Inc.", "delisted"),
        ]
        rows = default if companies is None else companies
        self.lists: dict[str, list[dict]] = {
            "/api/symbols": rows,
            "/api/company/delisted": [dict(row, delistedAt="2022-11-08") for row in rows if row["status"] == "delisted"],
            # The declared split and dividend calendars: the recorded first page's rows.
            "/api/company/upcoming-stock-splits": fixture("upcoming_stock_splits_page1")["data"],
            "/api/earnings/upcoming-dividends": fixture("earnings_upcoming_dividends_page1")["data"],
        }
        self.per_minute = per_minute
        self.per_day = per_day
        self.used_today = 0
        self.replies = list(replies or [])
        self.calls: list[tuple[str, dict, dict]] = []
        self.statuses: list[int] = []
        self._times: list[datetime] = []

    def requested(self, path: str) -> list[dict]:
        """Return the parameters of every call made to ``path``, in order."""
        return [params for url, params, _ in self.calls if urlsplit(url).path.endswith(path)]

    def _limits(self) -> dict[str, str]:
        recent = [t for t in self._times if self.clock.now() - t < timedelta(seconds=60)]
        return {
            "X-RateLimit-Limit-Minute": str(self.per_minute),
            "X-RateLimit-Remaining-Minute": str(max(self.per_minute - len(recent), 0)),
            "X-RateLimit-Limit-Day": str(self.per_day),
            "X-RateLimit-Remaining-Day": str(max(self.per_day - self.used_today, 0)),
        }

    def __call__(self, url, *, params, headers):
        response = self._answer(url, params, headers)
        self.statuses.append(response.status)
        return response

    def _answer(self, url, params, headers):
        from quantlab.acquisition._support.http import HttpResponse

        self.calls.append((url, dict(params), dict(headers)))
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return HttpResponse(*reply)
        if headers.get("Authorization") != f"Bearer {KEY}":
            return HttpResponse(401, b'{"error":"Unauthorized"}')
        now = self.clock.now()
        recent = [t for t in self._times if now - t < timedelta(seconds=60)]
        if len(recent) >= self.per_minute or self.used_today >= self.per_day:
            return HttpResponse(429, b'{"error":"Too Many Requests"}', self._limits())
        self._times.append(now)
        self.used_today += 1
        path = urlsplit(url).path.removeprefix("/v1")
        if path in self.lists:
            body = self._page(self.lists[path], params)
        else:
            body = self.bodies.get((path, _identity(params)))
            if body is None:
                return HttpResponse(404, b'{"error":"Not found"}', self._limits())
        return HttpResponse(200, json.dumps(body).encode(), self._limits())

    @staticmethod
    def _page(rows: list[dict], params) -> dict:
        rows = [r for r in rows if all(str(r[k]) == str(v) for k, v in params.items() if k in r)]
        page, size = int(params.get("page", 1)), int(params.get("pageSize", 100))
        return {
            "page": page,
            "pageSize": size,
            "totalPages": max(-(-len(rows) // size), 1),
            "totalResults": len(rows),
            "data": rows[(page - 1) * size : page * size],
        }
