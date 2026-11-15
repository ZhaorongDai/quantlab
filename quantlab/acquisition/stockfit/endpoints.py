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
  given;
- ``PerCompanyQuarter``: one request per company and calendar quarter
  (``cik`` and ``reportDate``), the entity ``<cik>_<quarter end>``;
- ``PerManager``: one request per 13F manager (``cik``), the managers named
  in the raw tier's 13F holder rows;
- ``PerManagerQuarter``: one request per 13F manager and quarter its 13F
  filings report;
- ``LatestFilings``: the market-wide feed of recent filings
  (``/api/filings/latest``), one entity per form type, read newest first
  back to the entity's last pull. The refresh takes from it the filers whose
  new filings drive a re-pull.

A row also says how the refresh keeps it current. A market-wide list is
pulled on every run (a snapshot). Any other endpoint is backfilled once per
entity, then re-pulled when a filing of one of its ``triggers`` form types
is made for the entity, and, with ``repull_days``, once its last pull is
that many days old (the weekly re-pull of the statements).

A pull is written through ``quantlab.acquisition.stockfit.raw_tier``: staged,
then published by one rename, or dropped when equal to the entity's newest
pull. A 404 means the vendor holds nothing for the entity: nothing is
published and the watermark moves.

Examples
--------
>>> ENDPOINTS["income_statement_quarter"].query("1093691")
{'period': 'quarter', 'limit': 80, 'cik': 1093691}
>>> list(ENDPOINTS)[:4]
['symbols', 'company_delisted', 'filings_latest', 'income_statement_annual']
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from pathlib import Path

from quantlab.acquisition.stockfit.client import StockFitClient, StockFitResponseError
from quantlab.acquisition.stockfit.raw_tier import StagedPull, write_entity_watermark
from quantlab.dataset.stockfit.raw import (
    COMPANY_DELISTED,
    CONGRESS_HOUSE_TRADES,
    CONGRESS_SENATE_TRADES,
    INSTITUTIONAL_HOLDERS,
    MANAGER_13F_FILINGS,
    MANAGER_PORTFOLIO,
    SYMBOLS,
    WHOLE_LIST,
    company_ciks,
    last_pulled,
    manager_holdings,
    quarter_entity,
    read_watermark,
    split_quarter_entity,
    thirteen_f_managers,
    thirteen_f_periods,
)
from quantlab.dataset.stockfit.raw import entities as raw_entities

#: The most periods one statement request returns.
STATEMENT_LIMIT = 80

#: The largest page ``/api/symbols`` and ``/api/company/delisted`` serve.
LIST_PAGE_SIZE = 1000

#: The largest page the Congress feeds serve.
CONGRESS_PAGE_SIZE = 250

#: The endpoint of the recent-filings feed (``/api/filings/latest``).
FILINGS_LATEST = "filings_latest"

#: The largest page, and the most pages, ``/api/filings/latest`` serves.
FEED_PAGE_SIZE = 100
FEED_MAX_PAGES = 100

#: The form types whose new filings drive a re-pull, by the feed's entity name.
FEED_FORMS: dict[str, str] = {
    form.replace(" ", "-"): form
    for form in ("10-K", "10-Q", "20-F", "4", "13F-HR", "NPORT-P", "13D", "13G", "DEF 14A")
}

#: The forms of a company's financial reports: they drive the statements.
REPORT_FORMS = frozenset({"10-K", "10-Q", "20-F"})

#: Matches the filer's CIK in a filing's EDGAR URL (``.../edgar/data/<cik>/...``).
_FILER_CIK = re.compile(r"/edgar/data/(\d+)/")


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
    triggers : frozenset of str, default frozenset()
        The form types (as ``/api/filings/latest`` names them) whose new
        filing re-pulls the filer's entities (``filer_entities``).
    repull_days : int or None, default None
        Re-pull an entity once its last pull is this many UTC days old;
        ``None`` re-pulls it only on a new filing of ``triggers``.
    """

    name: str
    path: str
    params: Mapping[str, object] = field(default_factory=dict)
    page_size: int | None = None
    triggers: frozenset[str] = frozenset()
    repull_days: int | None = None

    @abstractmethod
    def entities(self, context: PullContext) -> list[str]:
        """Return the entities to pull, by directory name, in pull order."""

    @abstractmethod
    def query(self, entity: str) -> dict:
        """Return the query of one entity's pull, without paging."""

    def query_now(self, vendor_root: Path, entity: str) -> dict:
        """Return the query of the entity's next pull, which may depend on the raw tier; ``query`` by default."""
        return self.query(entity)
    def filer_entities(self, ciks: Iterable[int], context: PullContext) -> dict[int, list[str]]:
        """Return, per filer CIK, the entities its filing of ``triggers`` touches; none by default.

        Examples
        --------
        >>> ENDPOINTS["symbols"].filer_entities([1093691], PullContext(Path("/d/stockfit")))
        {}
        """
        return {}

    def last_page(self, body: dict, vendor_root: Path, entity: str) -> bool:
        """Return whether a paged pull stops after ``body``, before the vendor's last page; never by default.

        Examples
        --------
        >>> ENDPOINTS["symbols"].last_page({"page": 1, "data": []}, Path("/d/stockfit"), "operating")
        False
        """
        return False

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
                if body is None and page == 1:  # nothing held for the entity
                    write_entity_watermark(vendor_root, self.name, entity, pulled_at)
                    return Outcome.MISSING
                if not isinstance(body, dict) or not isinstance(body.get("data"), list) or "totalPages" not in body:
                    raise StockFitResponseError(f"{self.path} page {page} is not a paged object with data.")
                staged.add_page(body)
                pages = int(body["totalPages"])
                if self.last_page(body, vendor_root, entity):
                    break
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

    def filer_entities(self, ciks: Iterable[int], context: PullContext) -> dict[int, list[str]]:
        """Return each filer's own company entity, for the companies this refresh pulls.

        Examples
        --------
        >>> ENDPOINTS["balance_sheet_ttm"].filer_entities([1093691], PullContext(Path("/d/stockfit")))
        {1093691: ['1093691']}
        """
        wanted = None if context.ciks is None else set(context.ciks)
        return {cik: [str(cik)] for cik in ciks if wanted is None or cik in wanted}


@dataclass(frozen=True)
class LatestFilings(MarketList):
    """The recent-filings feed, one entity per form type, read back to the entity's last pull.

    The vendor serves the last 30 days, newest first, at most
    ``FEED_MAX_PAGES`` pages. A pull stops after the first page that reaches
    a day before the entity's last pull, so the rows filed since are all in
    it, or at the vendor's last page. Its rows carry no CIK; ``filer_cik``
    reads the filer's from the EDGAR URL.

    Examples
    --------
    >>> ENDPOINTS[FILINGS_LATEST].query("DEF-14A")
    {'type': 'DEF 14A'}
    """

    def last_page(self, body: dict, vendor_root: Path, entity: str) -> bool:
        """Return whether the page reaches a day before the entity's last pull, or is the most served."""
        if int(body.get("page", 0)) >= FEED_MAX_PAGES:
            return True
        previous = last_pulled(vendor_root, self.name, entity)
        filed = [row["dateFiled"] for row in body["data"] if row.get("dateFiled")]
        return previous is not None and bool(filed) and date.fromisoformat(min(filed)) < previous.date() - timedelta(days=1)


def filer_cik(row: Mapping) -> int | None:
    """Return the filer's CIK of a row of ``/api/filings/latest``, read from its EDGAR URL.

    The URL's ``edgar/data/<cik>/`` is the filer's: the reporting company of
    a 10-K, the manager of a 13F, the issuer of a 13G.

    Examples
    --------
    >>> filer_cik({"url": "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000084/x.htm"})
    1652044
    >>> filer_cik({}) is None
    True
    """
    found = _FILER_CIK.search(str(row.get("url") or ""))
    return int(found.group(1)) if found else None


@dataclass(frozen=True)
class SinceFeed(MarketList):
    """A market-wide feed asked for everything filed on or after a day (``since``).

    The first pull asks from ``start``, the start of the vendor's archive,
    and pages through the whole history; each later pull asks from the day
    of the entity's last pull, ``overlap_days`` back, so a report the vendor
    added late is caught. Each pull holds only its window; the panels join
    every pull of the feed. The feed is one entity, ``WHOLE_LIST``. Like
    any market-wide list it is pulled on every run, from its watermark.

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

    def query(self, entity: str) -> dict:
        """Return the fixed parameters and the first pull's ``since``."""
        return {**self.params, "since": self.start}

    def query_now(self, vendor_root: Path, entity: str) -> dict:
        """Return the fixed parameters and ``since``: the last pull's day less ``overlap_days``, or ``start``."""
        last = read_watermark(vendor_root, self.name, entity)
        if last is None:
            return self.query(entity)
        return {**self.params, "since": max(last - timedelta(days=self.overlap_days), date.fromisoformat(self.start)).isoformat()}


#: The first quarter StockFit serves 13F holdings for.
FIRST_13F_QUARTER = date(2013, 3, 31)

#: Days after a quarter's end before its 13F holder rows are first pulled:
#: the 45 days managers have to file, and a few for the vendor to load them.
SETTLE_DAYS = 50

#: How many of a company's (or a manager's) newest quarters a new 13F filing re-pulls.
REPULLED_QUARTERS = 2

#: The largest page the 13F endpoints serve.
THIRTEEN_F_PAGE_SIZE = 500

#: The largest page ``/api/filings`` serves.
FILINGS_PAGE_SIZE = 100

#: The form whose new filing re-pulls the 13F endpoints, as the feed names it.
THIRTEEN_F_FORMS = frozenset({"13F-HR"})


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


def _newest_quarters(names: Iterable[str]) -> dict[int, list[str]]:
    """Return, per owner, its ``REPULLED_QUARTERS`` newest quarter entities among ``names``."""
    by_owner: dict[int, list[tuple[date, str]]] = {}
    for name in names:
        owner, quarter = split_quarter_entity(name)
        by_owner.setdefault(owner, []).append((quarter, name))
    return {
        owner: [name for _, name in sorted(found, reverse=True)[:REPULLED_QUARTERS]]
        for owner, found in by_owner.items()
    }


@dataclass(frozen=True)
class PerCompanyQuarter(Endpoint):
    """One request per company and calendar quarter, selected by ``cik`` and ``reportDate``.

    The quarters are those from ``first_quarter`` whose filings are all due
    by the refresh's day (``SETTLE_DAYS`` after the quarter end), newest
    first; each quarter is asked of every company before the one before it.
    An entity is ``<cik>_<quarter end>``. A new 13F filing of a manager
    re-pulls the ``REPULLED_QUARTERS`` newest pulled quarters of every
    company the manager's holder rows name (a late filing, an amendment).

    Parameters
    ----------
    first_quarter : date, default ``FIRST_13F_QUARTER``
        The oldest quarter pulled.

    Examples
    --------
    >>> row = ENDPOINTS["institutional_holders"]
    >>> row.entities(PullContext(Path("/d/stockfit"), ciks=(1652044,), end=date(2024, 9, 1)))[:2]
    ['1652044_2024-06-30', '1652044_2024-03-31']
    >>> row.query("1652044_2024-03-31")
    {'cik': 1652044, 'reportDate': '2024-03-31'}
    """

    first_quarter: date = FIRST_13F_QUARTER

    def entities(self, context: PullContext) -> list[str]:
        """Return every company's entity for each settled quarter, newest quarter first."""
        ciks: Sequence[int] = context.ciks if context.ciks is not None else company_ciks(context.vendor_root)
        settled = _today(context) - timedelta(days=SETTLE_DAYS)
        return [quarter_entity(cik, quarter) for quarter in quarter_ends(self.first_quarter, settled) for cik in ciks]

    def query(self, entity: str) -> dict:
        """Return the fixed parameters, ``cik`` and ``reportDate``."""
        cik, quarter = split_quarter_entity(entity)
        return {**self.params, "cik": cik, "reportDate": quarter.isoformat()}

    def filer_entities(self, ciks: Iterable[int], context: PullContext) -> dict[int, list[str]]:
        """Return, per filing manager, the newest pulled quarters of the companies its holder rows name.

        Examples
        --------
        >>> ENDPOINTS["institutional_holders"].filer_entities([1067983], PullContext(Path("/nonexistent")))
        {}
        """
        holdings = manager_holdings(context.vendor_root)
        newest = _newest_quarters(raw_entities(context.vendor_root, self.name))
        wanted = None if context.ciks is None else set(context.ciks)
        touched = {}
        for manager in ciks:
            companies = {split_quarter_entity(name)[0] for name in holdings.get(manager, ())}
            names = [
                name for company in sorted(companies) if wanted is None or company in wanted
                for name in newest.get(company, [])
            ]
            if names:
                touched[manager] = names
        return touched


@dataclass(frozen=True)
class PerManager(Endpoint):
    """One request per 13F manager, selected by its CIK.

    The managers are those named in the raw tier's 13F holder rows
    (``quantlab.dataset.stockfit.raw.thirteen_f_managers``), so a manager
    first appears in the run after its holder rows land. A manager's new
    13F filing re-pulls it.

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

    def filer_entities(self, ciks: Iterable[int], context: PullContext) -> dict[int, list[str]]:
        """Return each filing manager's own entity.

        Examples
        --------
        >>> ENDPOINTS["manager_13f_filings"].filer_entities([1067983], PullContext(Path("/nonexistent")))
        {1067983: ['1067983']}
        """
        return {cik: [str(cik)] for cik in ciks}


@dataclass(frozen=True)
class PerManagerQuarter(PerCompanyQuarter):
    """One request per 13F manager and quarter its filings report.

    The quarters of a manager are the periods of its newest 13F filing list
    (``quantlab.dataset.stockfit.raw.thirteen_f_periods``) from
    ``first_quarter`` on, newest first, so a quarter first appears in the
    run after the manager's filing list shows it. An entity is
    ``<manager cik>_<quarter end>``. A manager's new 13F filing re-pulls its
    ``REPULLED_QUARTERS`` newest pulled quarters.

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

    def filer_entities(self, ciks: Iterable[int], context: PullContext) -> dict[int, list[str]]:
        """Return each filing manager's newest pulled quarters.

        Examples
        --------
        >>> ENDPOINTS["manager_portfolio"].filer_entities([1067983], PullContext(Path("/nonexistent")))
        {}
        """
        newest = _newest_quarters(raw_entities(context.vendor_root, self.name))
        return {cik: newest[cik] for cik in ciks if cik in newest}


def _statements() -> list[Endpoint]:
    paths = {
        "income_statement": "/api/financials/income-statement",
        "balance_sheet": "/api/financials/balance-sheet",
        "cash_flow_statement": "/api/financials/cash-flow-statement",
    }
    return [
        PerCompany(
            f"{name}_{period}", path, {"period": period, "limit": STATEMENT_LIMIT},
            triggers=REPORT_FORMS, repull_days=7,  # re-pulled on a new report, and weekly
        )
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
    # The filings of the last 30 days per form type: whose new filing re-pulls what.
    LatestFilings(
        FILINGS_LATEST, "/api/filings/latest", page_size=FEED_PAGE_SIZE,
        variants={entity: {"type": form} for entity, form in FEED_FORMS.items()},
    ),
    # The three statements, annual, quarterly and TTM, as many periods as served.
    *_statements(),
    # Congress members' periodic transaction reports, House and Senate, by filing day.
    SinceFeed(CONGRESS_HOUSE_TRADES, "/api/congress/house/trades/recent", page_size=CONGRESS_PAGE_SIZE),
    SinceFeed(CONGRESS_SENATE_TRADES, "/api/congress/senate/trades/recent", page_size=CONGRESS_PAGE_SIZE),
    # 13F, last: the holder rows of each company per quarter (every share
    # class, by CUSIP), then the filings of each manager they name (the
    # filing date of each row's accession), then each manager's portfolio
    # per quarter it reported (raw only, the largest pull).
    PerCompanyQuarter(
        INSTITUTIONAL_HOLDERS, "/api/ownership/institutional-holders",
        page_size=THIRTEEN_F_PAGE_SIZE, triggers=THIRTEEN_F_FORMS,
    ),
    PerManager(
        MANAGER_13F_FILINGS, "/api/filings", {"type": "13F-HR"},
        page_size=FILINGS_PAGE_SIZE, triggers=THIRTEEN_F_FORMS,
    ),
    PerManagerQuarter(
        MANAGER_PORTFOLIO, "/api/ownership/portfolio", page_size=THIRTEEN_F_PAGE_SIZE, triggers=THIRTEEN_F_FORMS,
    ),
)}
