"""StockFit's endpoints as data: one class per pull shape, one table row per endpoint.

A pull shape says which entities an endpoint is pulled for and how one
entity's pull is asked for; an endpoint is a row of ``ENDPOINTS``: its name
(the raw tier's directory), its path and its fixed query parameters. Adding
an endpoint of an existing shape is one row.

Shapes:

- ``MarketList``: a market-wide paged list (``/api/symbols``), one entity
  per named variant of its query (the operating and the delisted companies),
  every page of the vendor's paged object (``page``, ``pageSize``,
  ``totalPages``, ``data``) kept in one pull;
- ``SinceFeed``: a market-wide feed asked from a day (``since``), from the
  start of the archive first and from the last pull's day afterwards;
- ``PerCompany``: one request per company, selected by ``cik``; the
  companies are those of the newest company lists in the raw tier
  (``quantlab.dataset.stockfit.raw.company_ciks``), or the CIKs a refresh is
  given.

A pull is written through ``quantlab.acquisition.stockfit.raw_tier``: staged,
then published by one rename, or dropped when equal to the entity's newest
pull. A 404 means the vendor holds nothing for the entity: nothing is
published and the watermark moves.

Examples
--------
>>> ENDPOINTS["income_statement_quarter"].query("1093691")
{'period': 'quarter', 'limit': 80, 'cik': 1093691}
>>> list(ENDPOINTS)[:3]
['symbols', 'company_delisted', 'income_statement_annual']
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

from quantlab.acquisition.stockfit.client import StockFitClient, StockFitResponseError
from quantlab.acquisition.stockfit.raw_tier import StagedPull, write_entity_watermark
from quantlab.dataset.stockfit.raw import (
    COMPANY_DELISTED,
    CONGRESS_HOUSE_TRADES,
    CONGRESS_SENATE_TRADES,
    SYMBOLS,
    WHOLE_LIST,
    company_ciks,
    read_watermark,
)
from quantlab.dataset.stockfit.raw import (
    COMPANY_DETAILS,
    STOCK_SPLITS,
    UPCOMING_DIVIDENDS,
    UPCOMING_STOCK_SPLITS,
)

#: The most periods one statement request returns.
STATEMENT_LIMIT = 80

#: The largest page ``/api/symbols`` and ``/api/company/delisted`` serve.
LIST_PAGE_SIZE = 1000

#: The days ahead the declared split and dividend calendars are asked for (the most served).
FORWARD_DAYS = 28

#: The largest page the declared split and dividend calendars serve.
FORWARD_PAGE_SIZE = 500

#: The largest page the Congress feeds serve.
CONGRESS_PAGE_SIZE = 250


class Outcome(Enum):
    """What one entity's pull did to the raw tier."""

    PUBLISHED = "published"
    UNCHANGED = "unchanged"
    MISSING = "missing"


@dataclass(frozen=True)
class PullContext:
    """What a pull shape may consult to list its entities.

    Parameters
    ----------
    vendor_root : Path
        ``<download-dir>/stockfit``.
    ciks : tuple of int or None, default None
        The companies to pull; those of the raw tier's company lists when
        ``None``.
    """

    vendor_root: Path
    ciks: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Endpoint(ABC):
    """One StockFit endpoint: a row of ``ENDPOINTS``, pulled entity by entity.

    Parameters
    ----------
    name : str
        The endpoint's directory in the raw tier.
    path : str
        The API path, appended to ``client.BASE_URL``.
    params : Mapping, default {}
        Query parameters sent with every request of the endpoint.
    page_size : int or None, default None
        For a paged object, the ``pageSize`` asked for; every page is
        followed. ``None`` for a one-page response.
    """

    name: str
    path: str
    params: Mapping[str, object] = field(default_factory=dict)
    page_size: int | None = None

    @abstractmethod
    def entities(self, context: PullContext) -> list[str]:
        """Return the entities to pull, by directory name, in pull order."""

    @abstractmethod
    def query(self, entity: str) -> dict:
        """Return the query of one entity's pull, without paging."""

    def query_now(self, vendor_root: Path, entity: str) -> dict:
        """Return the query of the entity's next pull, which may depend on the raw tier; ``query`` by default."""
        return self.query(entity)

    def pull(self, client: StockFitClient, vendor_root: Path, entity: str) -> Outcome:
        """Pull one entity: stage every page, then publish or drop it; the watermark moves.

        Raises
        ------
        StockFitResponseError
            If a paged body lacks ``data`` or ``totalPages``; nothing is
            published.
        Exception
            The client's errors, from the request that failed; nothing is
            published and the watermark stays.
        """
        pulled_at = client.now()
        query = self.query_now(vendor_root, entity)
        if self.page_size is None:
            body = client.get_json(self.path, query)
            if body is None:
                write_entity_watermark(vendor_root, self.name, entity, pulled_at)
                return Outcome.MISSING
            staged = StagedPull(vendor_root, self.name, entity, pulled_at, query=query)
            staged.add_page(body)
        else:
            staged = StagedPull(vendor_root, self.name, entity, pulled_at, records="data", query=query)
            page, pages = 1, 1
            while page <= pages:
                body = client.get_json(self.path, {**query, "page": page, "pageSize": self.page_size})
                if not isinstance(body, dict) or not isinstance(body.get("data"), list) or "totalPages" not in body:
                    raise StockFitResponseError(f"{self.path} page {page} is not a paged object with data.")
                staged.add_page(body)
                pages = int(body["totalPages"])
                page += 1
        return Outcome.PUBLISHED if staged.commit().published else Outcome.UNCHANGED


@dataclass(frozen=True)
class MarketList(Endpoint):
    """A market-wide list, one entity per variant of its query.

    Parameters
    ----------
    variants : Mapping[str, Mapping], default {"all": {}}
        Entity name to the query parameters that select it, added to
        ``params``.

    Examples
    --------
    >>> ENDPOINTS["symbols"].entities(PullContext(Path("/d/stockfit")))
    ['operating', 'delisted']
    """

    variants: Mapping[str, Mapping[str, object]] = field(default_factory=lambda: {WHOLE_LIST: {}})

    def entities(self, context: PullContext) -> list[str]:
        """Return the variants, in table order."""
        return list(self.variants)

    def query(self, entity: str) -> dict:
        """Return the fixed parameters and the variant's."""
        return {**self.params, **self.variants[entity]}


@dataclass(frozen=True)
class PerCompany(Endpoint):
    """One request per company, selected by its CIK.

    Examples
    --------
    >>> ENDPOINTS["balance_sheet_ttm"].entities(PullContext(Path("/d/stockfit"), ciks=(1093691,)))
    ['1093691']
    """

    def entities(self, context: PullContext) -> list[str]:
        """Return the given CIKs, or every company of the raw tier's company lists."""
        ciks: Sequence[int] = context.ciks if context.ciks is not None else company_ciks(context.vendor_root)
        return [str(cik) for cik in ciks]

    def query(self, entity: str) -> dict:
        """Return the fixed parameters and ``cik``."""
        return {**self.params, "cik": int(entity)}


@dataclass(frozen=True)
class SinceFeed(Endpoint):
    """A market-wide feed asked for everything filed on or after a day (``since``).

    The first pull asks from ``start``, the start of the vendor's archive,
    and pages through the whole history; each later pull asks from the day
    of the entity's last pull, ``overlap_days`` back, so a report the vendor
    added late is caught. Each pull holds only its window; the panels join
    every pull of the feed. The feed is one entity, ``WHOLE_LIST``.

    Parameters
    ----------
    start : str, default "2012-01-01"
        The ``since`` of the first pull (ISO date).
    overlap_days : int, default 30
        How far before the last pull's day a later pull reaches back.

    Examples
    --------
    >>> feed = ENDPOINTS["congress_house_trades"]
    >>> feed.query_now(Path("/nonexistent/stockfit"), "all")
    {'since': '2012-01-01'}
    """

    start: str = "2012-01-01"
    overlap_days: int = 30

    def entities(self, context: PullContext) -> list[str]:
        """Return the one entity of the feed."""
        return [WHOLE_LIST]

    def query(self, entity: str) -> dict:
        """Return the fixed parameters and the first pull's ``since``."""
        return {**self.params, "since": self.start}

    def query_now(self, vendor_root: Path, entity: str) -> dict:
        """Return the fixed parameters and ``since``: the last pull's day less ``overlap_days``, or ``start``."""
        last = read_watermark(vendor_root, self.name, entity)
        if last is None:
            return self.query(entity)
        return {**self.params, "since": max(last - timedelta(days=self.overlap_days), date.fromisoformat(self.start)).isoformat()}


def _statements() -> list[Endpoint]:
    paths = {
        "income_statement": "/api/financials/income-statement",
        "balance_sheet": "/api/financials/balance-sheet",
        "cash_flow_statement": "/api/financials/cash-flow-statement",
    }
    return [
        PerCompany(f"{name}_{period}", path, {"period": period, "limit": STATEMENT_LIMIT})
        for period in ("annual", "quarter", "ttm")
        for name, path in paths.items()
    ]


#: Every endpoint the refresh pulls, by name, in pull order: the company
#: lists first, since the per-company endpoints take their CIKs from them.
ENDPOINTS: dict[str, Endpoint] = {endpoint.name: endpoint for endpoint in (
    # Every company, operating and delisted: the CIKs of the per-company pulls.
    MarketList(
        SYMBOLS, "/api/symbols", {"type": "stock"}, page_size=LIST_PAGE_SIZE,
        variants={"operating": {"status": "operating"}, "delisted": {"status": "delisted"}},
    ),
    # The delisted companies with their delisting day.
    MarketList(COMPANY_DELISTED, "/api/company/delisted", page_size=LIST_PAGE_SIZE),
    # The three statements, annual, quarterly and TTM, as many periods as served.
    *_statements(),
    # Corporate events: each company's profile (its successions), its XBRL
    # split history, and the declared splits and dividends of the next
    # FORWARD_DAYS days, a forward snapshot kept on every refresh.
    PerCompany(COMPANY_DETAILS, "/api/company/details"),
    PerCompany(STOCK_SPLITS, "/api/company/stock-splits"),
    MarketList(UPCOMING_STOCK_SPLITS, "/api/company/upcoming-stock-splits", {"days": FORWARD_DAYS},
               page_size=FORWARD_PAGE_SIZE),
    MarketList(UPCOMING_DIVIDENDS, "/api/earnings/upcoming-dividends", {"days": FORWARD_DAYS},
               page_size=FORWARD_PAGE_SIZE),
    # Congress members' periodic transaction reports, House and Senate, by filing day.
    SinceFeed(CONGRESS_HOUSE_TRADES, "/api/congress/house/trades/recent", page_size=CONGRESS_PAGE_SIZE),
    SinceFeed(CONGRESS_SENATE_TRADES, "/api/congress/senate/trades/recent", page_size=CONGRESS_PAGE_SIZE),
)}
