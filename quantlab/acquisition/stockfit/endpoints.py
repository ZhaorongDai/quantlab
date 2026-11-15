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
- ``PerCompany``: one request per company, selected by ``cik``; the
  companies are those of the newest company lists in the raw tier
  (``quantlab.dataset.stockfit.raw.company_ciks``), or the CIKs a refresh is
  given;
- ``PerCompanyQuarter``: one request per company and calendar quarter
  (``cik`` and ``reportDate``), the entity ``<cik>_<quarter end>``;
- ``PerManager``: one request per 13F manager (``cik``), the managers named
  in the raw tier's 13F holder rows;
- ``PerManagerQuarter``: one request per 13F manager and quarter its 13F
  filings report.

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
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from pathlib import Path

from quantlab.acquisition.stockfit.client import StockFitClient, StockFitResponseError
from quantlab.acquisition.stockfit.raw_tier import StagedPull, write_entity_watermark
from quantlab.dataset.stockfit.raw import (
    COMPANY_DELISTED,
    INSTITUTIONAL_HOLDERS,
    MANAGER_13F_FILINGS,
    MANAGER_PORTFOLIO,
    SYMBOLS,
    WHOLE_LIST,
    company_ciks,
    quarter_entity,
    split_quarter_entity,
    thirteen_f_managers,
    thirteen_f_periods,
)

#: The most periods one statement request returns.
STATEMENT_LIMIT = 80

#: The largest page ``/api/symbols`` and ``/api/company/delisted`` serve.
LIST_PAGE_SIZE = 1000


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
    end : date or None, default None
        The day the refresh runs for; today (UTC) when ``None``.
    """

    vendor_root: Path
    ciks: tuple[int, ...] | None = None
    #: The day the refresh runs for: an endpoint asked per quarter pulls the
    #: quarters ended by it. ``None`` is today (UTC).
    end: date | None = None


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
        query = self.query(entity)
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
                if body is None and page == 1:  # nothing held for the entity
                    write_entity_watermark(vendor_root, self.name, entity, pulled_at)
                    return Outcome.MISSING
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


#: The first quarter StockFit serves 13F holdings for.
FIRST_13F_QUARTER = date(2013, 3, 31)

#: The largest page the 13F endpoints serve.
THIRTEEN_F_PAGE_SIZE = 500

#: The largest page ``/api/filings`` serves.
FILINGS_PAGE_SIZE = 100


def quarter_ends(first: date, end: date) -> list[date]:
    """Return the calendar quarter ends from ``first`` to ``end``, both inclusive, newest first.

    Examples
    --------
    >>> quarter_ends(date(2024, 1, 15), date(2024, 7, 1))
    [datetime.date(2024, 6, 30), datetime.date(2024, 3, 31)]
    """
    found = []
    year, month = first.year, (first.month - 1) // 3 * 3 + 3
    while True:
        quarter = (date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
        if quarter > end:
            return found[::-1]
        if quarter >= first:
            found.append(quarter)
        year, month = (year, month + 3) if month < 12 else (year + 1, 3)


def _today(context: PullContext) -> date:
    return context.end if context.end is not None else datetime.now(UTC).date()


@dataclass(frozen=True)
class PerCompanyQuarter(Endpoint):
    """One request per company and calendar quarter, selected by ``cik`` and ``reportDate``.

    The quarters are those ended from ``first_quarter`` to the refresh's day,
    newest first; each quarter is asked of every company before the one
    before it. An entity is ``<cik>_<quarter end>``.

    Parameters
    ----------
    first_quarter : date, default ``FIRST_13F_QUARTER``
        The oldest quarter pulled.

    Examples
    --------
    >>> row = ENDPOINTS["institutional_holders"]
    >>> row.entities(PullContext(Path("/d/stockfit"), ciks=(1652044,), end=date(2024, 7, 1)))[:2]
    ['1652044_2024-06-30', '1652044_2024-03-31']
    >>> row.query("1652044_2024-03-31")
    {'cik': 1652044, 'reportDate': '2024-03-31'}
    """

    first_quarter: date = FIRST_13F_QUARTER

    def entities(self, context: PullContext) -> list[str]:
        """Return every company's entity for each quarter, newest quarter first."""
        ciks: Sequence[int] = context.ciks if context.ciks is not None else company_ciks(context.vendor_root)
        return [
            quarter_entity(cik, quarter)
            for quarter in quarter_ends(self.first_quarter, _today(context))
            for cik in ciks
        ]

    def query(self, entity: str) -> dict:
        """Return the fixed parameters, ``cik`` and ``reportDate``."""
        cik, quarter = split_quarter_entity(entity)
        return {**self.params, "cik": cik, "reportDate": quarter.isoformat()}


@dataclass(frozen=True)
class PerManager(Endpoint):
    """One request per 13F manager, selected by its CIK.

    The managers are those named in the raw tier's 13F holder rows
    (``quantlab.dataset.stockfit.raw.thirteen_f_managers``), so the holder
    endpoint is pulled first.

    Examples
    --------
    >>> ENDPOINTS["manager_13f_filings"].query("1067983")
    {'type': '13F-HR', 'cik': 1067983}
    """

    def entities(self, context: PullContext) -> list[str]:
        """Return every manager of the raw tier's holder rows."""
        return [str(cik) for cik in thirteen_f_managers(context.vendor_root)]

    def query(self, entity: str) -> dict:
        """Return the fixed parameters and ``cik``."""
        return {**self.params, "cik": int(entity)}


@dataclass(frozen=True)
class PerManagerQuarter(PerCompanyQuarter):
    """One request per 13F manager and quarter its filings report.

    The quarters of a manager are the periods of its newest 13F filing list
    (``quantlab.dataset.stockfit.raw.thirteen_f_periods``) from
    ``first_quarter`` on, newest first, so the filing endpoint is pulled
    first. An entity is ``<manager cik>_<quarter end>``.

    Examples
    --------
    >>> ENDPOINTS["manager_portfolio"].query("1067983_2023-09-30")
    {'cik': 1067983, 'reportDate': '2023-09-30'}
    """

    def entities(self, context: PullContext) -> list[str]:
        """Return each manager's entity for each quarter its filings report."""
        last = _today(context)
        return [
            quarter_entity(manager, quarter)
            for manager in thirteen_f_managers(context.vendor_root)
            for quarter in thirteen_f_periods(context.vendor_root, manager)
            if self.first_quarter <= quarter <= last
        ]


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
    # 13F, last: the holder rows of each company per quarter (every share
    # class, by CUSIP), then the filings of each manager they name (the
    # filing date of each row's accession), then each manager's portfolio
    # per quarter it reported (raw only, the largest pull).
    PerCompanyQuarter(
        INSTITUTIONAL_HOLDERS, "/api/ownership/institutional-holders", page_size=THIRTEEN_F_PAGE_SIZE
    ),
    PerManager(MANAGER_13F_FILINGS, "/api/filings", {"type": "13F-HR"}, page_size=FILINGS_PAGE_SIZE),
    PerManagerQuarter(MANAGER_PORTFOLIO, "/api/ownership/portfolio", page_size=THIRTEEN_F_PAGE_SIZE),
)}
