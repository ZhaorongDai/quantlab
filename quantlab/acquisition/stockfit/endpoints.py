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
- ``CompanySnapshot``: a per-company snapshot (the predicted filing
  calendar, the latest headlines), one request per operating company,
  re-pulled once a day from what the day's requests leave;
- ``PerCompanyQuarter``: one request per company and calendar quarter
  (``cik`` and ``reportDate``), the entity ``<cik>_<quarter end>``;
- ``PerManager``: one request per 13F manager (``cik``), the managers named
  in the raw tier's 13F holder rows;
- ``PerManagerQuarter``: one request per 13F manager and quarter its 13F
  filings report;
- ``LatestFilings``: the market-wide feed of recent filings
  (``/api/filings/latest``), one entity per form type, read newest first
  back to the entity's last pull. The refresh takes from it the filers whose
  new filings drive a re-pull;
- ``PerFundSeries``: one request per fund series (``<cik>-<series id>``) of
  the newest fund lists (``quantlab.dataset.stockfit.raw.fund_listings``),
  asked for by its ticker or Composite FIGI (``fund_selector``);
  ``PerFundQuarter``: one per series and quarter, each asked once;
  ``PerAdviser``: one per adviser CRD found in the series' N-CEN service
  providers.

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
from functools import lru_cache
from pathlib import Path

from quantlab.acquisition.stockfit.client import StockFitClient, StockFitResponseError
from quantlab.acquisition.stockfit.raw_tier import StagedPull, write_entity_watermark
from quantlab.dataset.stockfit.beneficial_owners import BENEFICIAL_OWNERS
from quantlab.dataset.stockfit.executives import (
    EXECUTIVES_COMPENSATION,
    EXECUTIVES_GOVERNANCE,
    EXECUTIVES_PERFORMANCE_MEASURES,
)
from quantlab.dataset.stockfit.filings import FILINGS
from quantlab.dataset.stockfit.footnotes import FOOTNOTES, PERIODS, footnote_endpoint
from quantlab.dataset.stockfit.insiders import INSIDER_TRANSACTIONS
from quantlab.dataset.stockfit.raw import (
    COMPANY_DELISTED,
    COMPANY_DETAILS,
    CONGRESS_HOUSE_TRADES,
    CONGRESS_SENATE_TRADES,
    EARNINGS_UPCOMING,
    FILINGS_CALENDAR,
    FUND_DAILY_SUPPORTED,
    FUND_LISTS,
    FUND_SYMBOLS,
    INSTITUTIONAL_HOLDERS,
    IPO_CALENDAR,
    MANAGER_13F_FILINGS,
    MANAGER_PORTFOLIO,
    NEWS,
    RECENT_OFFERINGS,
    STOCK_SPLITS,
    SYMBOLS,
    UPCOMING_DIVIDENDS,
    UPCOMING_STOCK_SPLITS,
    WHOLE_LIST,
    company_ciks,
    daily_supported,
    fund_listings,
    last_pulled,
    manager_holdings,
    newest_pull_dir,
    pulls,
    quarter_entity,
    read_pull,
    read_watermark,
    split_quarter_entity,
    thirteen_f_managers,
    thirteen_f_periods,
)
from quantlab.dataset.stockfit.raw import entities as raw_entities

#: The most periods one statement request returns.
STATEMENT_LIMIT = 80

#: The most periods ``/api/earnings/dividend-history`` returns.
DIVIDEND_HISTORY_LIMIT = 40

#: Days after which a company's scores and trends (snapshots without history) are re-pulled.
SNAPSHOT_REPULL_DAYS = 30

#: The largest page ``/api/symbols`` and ``/api/company/delisted`` serve.
LIST_PAGE_SIZE = 1000

#: The forms whose new filing by a trust re-pulls its series' N-PORT endpoints.
NPORT_FORMS = frozenset({"NPORT-P"})

#: Days after which a fund series' N-PORT, N-1A and N-CSR endpoints are re-pulled without a filing.
FUND_REPULL_DAYS = 30

#: Days after which the N-CEN endpoints and the adviser lookup are re-pulled (N-CEN is annual).
NCEN_REPULL_DAYS = 90

#: The largest page of the fund fee and performance tables.
FUND_PAGE_SIZE = 100

#: The largest page of N-PORT and daily fund holdings.
HOLDINGS_PAGE_SIZE = 1000

#: The largest page of the funds of one adviser.
ADVISER_PAGE_SIZE = 500

#: The most N-CEN reporting years ``/api/fund/service-providers`` returns.
PROVIDER_YEARS = 40
#: The days ahead the declared split and dividend calendars are asked for (the most served).
FORWARD_DAYS = 28

#: The largest page the declared split and dividend calendars serve.
FORWARD_PAGE_SIZE = 500

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

#: The forms that disclose a split: the reports that tag it in XBRL, and an 8-K (when the feed follows it).
SPLIT_FORMS = REPORT_FORMS | {"8-K"}

#: Matches the filer's CIK in a filing's EDGAR URL (``.../edgar/data/<cik>/...``).
_FILER_CIK = re.compile(r"/edgar/data/(\d+)/")

#: The largest page ``/api/filings`` serves.
FILINGS_PAGE_SIZE = 100

#: The largest page ``/api/insider-transactions`` serves.
INSIDER_PAGE_SIZE = 250


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
    fund_series : tuple of str or None, default None
        The fund series (``<cik>-<series id>``) to pull; every series of the
        raw tier's fund lists when ``None``.
    end : date or None, default None
        The day the refresh runs for; today (UTC) when ``None``.
    """

    vendor_root: Path
    ciks: tuple[int, ...] | None = None
    fund_series: tuple[str, ...] | None = None
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
                if body is None and page == 1:  # the vendor holds nothing for the entity
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


# -- fund series ------------------------------------------------------------------


def _listings_key(vendor_root: Path) -> tuple:
    """The newest pulls the fund listings are read from: a new pull of any makes a new key."""
    return tuple(
        str(newest_pull_dir(vendor_root, endpoint, entity))
        for endpoint, entity in (*FUND_LISTS, (FUND_DAILY_SUPPORTED, WHOLE_LIST))
    )


@lru_cache(maxsize=4)
def _listings_at(vendor_root: Path, key: tuple) -> tuple[dict[str, dict], frozenset[str]]:
    del key  # only keys the cache
    return fund_listings(vendor_root), frozenset(daily_supported(vendor_root))


def _listings(vendor_root: Path) -> tuple[dict[str, dict], frozenset[str]]:
    """The fund series of the raw tier's newest fund lists, and the tickers with daily holdings."""
    vendor_root = Path(vendor_root)
    return _listings_at(vendor_root, _listings_key(vendor_root))


def fund_selector(listing: Mapping, supported: frozenset[str] | None = None) -> dict | None:
    """Return the selector that asks StockFit's fund endpoints for one fund series; ``None`` if nothing does.

    Fund endpoints take a security selector, not the series id, so a series
    is asked for by one of its tickers (the first listed, or with
    ``supported`` the first StockFit has daily holdings for), else by its
    Composite FIGI. A ticker is today's; the fund lists are today's too.

    Examples
    --------
    >>> fund_selector({"symbols": ["VOO"], "composite_figi": "BBG000BHTMY2"})
    {'symbol': 'VOO'}
    >>> fund_selector({"symbols": [], "composite_figi": "BBG000BHTMY2"})
    {'composite_figi': 'BBG000BHTMY2'}
    >>> fund_selector({"symbols": ["VFIAX", "VOO"], "composite_figi": None}, frozenset({"VOO"}))
    {'symbol': 'VOO'}
    """
    if supported is not None:
        tickers = [ticker for ticker in listing["symbols"] if ticker in supported]
        return {"symbol": tickers[0]} if tickers else None
    if listing["symbols"]:
        return {"symbol": listing["symbols"][0]}
    if listing.get("composite_figi"):
        return {"composite_figi": listing["composite_figi"]}
    return None


@dataclass(frozen=True)
class PerFundSeries(Endpoint):
    """One request per fund series (``<cik>-<series id>``), asked for by its selector (``fund_selector``).

    The series are those of the newest fund lists in the raw tier
    (``quantlab.dataset.stockfit.raw.fund_listings``), or the ones a
    refresh is given; a series nothing selects is not pulled.

    Parameters
    ----------
    daily : bool, default False
        Pull only the series with a ticker StockFit has daily website
        holdings for (``/api/fund/holdings/daily/supported-funds``), asked
        for by that ticker.

    Examples
    --------
    >>> ENDPOINTS["fund_profile"].query("36405-2839", {"symbol": "VOO"})
    {'symbol': 'VOO'}
    """

    daily: bool = False

    def _selectors(self, context: PullContext) -> dict[str, dict]:
        listings, supported = _listings(context.vendor_root)
        wanted = listings if context.fund_series is None else {
            name: listings[name] for name in context.fund_series if name in listings
        }
        selectors = {name: fund_selector(listing, supported if self.daily else None)
                     for name, listing in wanted.items()}
        return {name: selector for name, selector in selectors.items() if selector is not None}

    def entities(self, context: PullContext) -> list[str]:
        """Return the fund series to pull, sorted by CIK, then series id."""
        return list(self._selectors(context))

    def filer_entities(self, ciks: Iterable[int], context: PullContext) -> dict[int, list[str]]:
        """Return each filer's fund series: a trust files each series' N-PORT under its own CIK.

        The feed's rows name the trust, not the series, so a trust's filing
        touches every series of it the refresh pulls (the series whose entity
        starts with ``<cik>-``).

        Examples
        --------
        >>> ENDPOINTS["fund_profile"].filer_entities([36405], PullContext(Path("/nonexistent/stockfit")))
        {}
        """
        by_trust: dict[int, list[str]] = {}
        for series in self.entities(context):
            by_trust.setdefault(int(series.split("-")[0]), []).append(series)
        return {cik: by_trust[cik] for cik in ciks if cik in by_trust}

    def query(self, entity: str, selector: Mapping | None = None) -> dict:
        """Return the fixed parameters and the series' selector.

        Raises
        ------
        ValueError
            If ``selector`` is omitted: a series id alone selects nothing.
        """
        if selector is None:
            raise ValueError(f"StockFit fund series {entity!r} is asked for by a selector; none was given.")
        return {**self.params, **selector}

    def query_now(self, vendor_root: Path, entity: str) -> dict:
        """Return the query with the series' selector from the raw tier's fund lists."""
        selector = self._selectors(PullContext(vendor_root, fund_series=(entity,))).get(entity)
        return self.query(entity, selector)


@dataclass(frozen=True)
class PerFundQuarter(PerFundSeries):
    """One request per fund series and quarter end its N-PORT summary lists, each asked once.

    The quarters are the ``reportDate`` of the series' newest
    ``fund_profile`` pull; the entity is ``<series>_<YYYY-MM-DD>``. A quarter
    already asked for (its watermark is set, whether or not the vendor
    served it) is not asked again: a past quarter's N-PORT does not change.

    Examples
    --------
    >>> ENDPOINTS["fund_holdings"].query("36405-2839_2026-06-30", {"symbol": "VOO"})
    {'symbol': 'VOO', 'reportDate': '2026-06-30'}
    """

    quarters_from: str = "fund_profile"

    def filer_entities(self, ciks: Iterable[int], context: PullContext) -> dict[int, list[str]]:
        """Return none: a new quarter is a new entity, backfilled once its ``fund_profile`` lists it."""
        return {}

    def entities(self, context: PullContext) -> list[str]:
        """Return each series' quarters not yet asked for, series by series, newest quarter first."""
        found = []
        for series in self._selectors(context):
            if newest_pull_dir(context.vendor_root, self.quarters_from, series) is None:
                continue
            rows = read_pull(context.vendor_root, self.quarters_from, series)
            for day in dict.fromkeys(str(row["reportDate"]) for row in rows if row.get("reportDate")):
                entity = f"{series}_{day}"
                if read_watermark(context.vendor_root, self.name, entity) is None:
                    found.append(entity)
        return found

    def query(self, entity: str, selector: Mapping | None = None) -> dict:
        """Return the fixed parameters, the series' selector and the quarter's ``reportDate``."""
        series, day = entity.split("_")
        return {**super().query(series, selector), "reportDate": day}

    def query_now(self, vendor_root: Path, entity: str) -> dict:
        """Return the query with the series' selector from the raw tier's fund lists."""
        series = entity.split("_")[0]
        selector = self._selectors(PullContext(vendor_root, fund_series=(series,))).get(series)
        return self.query(entity, selector)


@dataclass(frozen=True)
class PerAdviser(Endpoint):
    """One request per investment adviser, by CRD number, from the fund series' N-CEN service providers.

    The advisers are the providers whose role names an adviser in the
    newest pull of every series of ``providers_from``.

    Examples
    --------
    >>> ENDPOINTS["fund_by_adviser"].query("105958")
    {'crd': '105958'}
    """

    providers_from: str = "fund_service_providers"

    def entities(self, context: PullContext) -> list[str]:
        """Return the advisers' CRD numbers, sorted."""
        crds: set[str] = set()
        for series in raw_entities(context.vendor_root, self.providers_from):
            body = read_pull(context.vendor_root, self.providers_from, series)
            for year in body.get("providersByYear") or [] if isinstance(body, dict) else []:
                crds.update(
                    str(provider["crdNumber"]) for provider in year.get("providers") or []
                    if provider.get("crdNumber") and "advis" in str(provider.get("providerRole", "")).lower()
                )
        return sorted(crds, key=lambda crd: (len(crd), crd))

    def query(self, entity: str) -> dict:
        """Return the fixed parameters and ``crd``."""
        return {**self.params, "crd": entity}


def _funds() -> list[Endpoint]:
    """The fund endpoints, in pull order: the lists first, the adviser lookup after its providers."""
    return [
        # Every fund series, by kind and listing status: the series of the per-series pulls.
        MarketList(
            FUND_SYMBOLS, "/api/symbols", page_size=LIST_PAGE_SIZE,
            variants={entity: dict(zip(("type", "status"), entity.split("_"))) for _, entity in FUND_LISTS},
        ),
        # The tickers with daily website holdings.
        MarketList(FUND_DAILY_SUPPORTED, "/api/fund/holdings/daily/supported-funds"),
        # N-PORT summary and monthly flows, per quarter (panels): re-pulled on the trust's new N-PORT.
        PerFundSeries("fund_profile", "/api/fund/profile", triggers=NPORT_FORMS, repull_days=FUND_REPULL_DAYS),
        PerFundSeries("fund_flows", "/api/fund/flows", triggers=NPORT_FORMS, repull_days=FUND_REPULL_DAYS),
        # N-1A prospectus fee tables (panel) and N-CSR performance per share class (raw only); the feed
        # follows neither form, so they are re-pulled on age only.
        PerFundSeries("fund_fees", "/api/fund/fees", page_size=FUND_PAGE_SIZE, repull_days=FUND_REPULL_DAYS),
        PerFundSeries(
            "fund_performance", "/api/fund/performance", page_size=FUND_PAGE_SIZE, repull_days=FUND_REPULL_DAYS,
        ),
        # N-PORT holdings, each quarter once (raw only).
        PerFundQuarter("fund_holdings", "/api/fund/holdings", page_size=HOLDINGS_PAGE_SIZE),
        # Daily website holdings, a snapshot a day: due daily, after the backfill and the filing-driven pulls.
        PerFundSeries(
            "fund_holdings_daily", "/api/fund/holdings/daily", page_size=HOLDINGS_PAGE_SIZE, daily=True, repull_days=1,
        ),
        # N-CEN: fund health and fee analysis (panel), service providers (raw only); an annual form the
        # feed does not follow, re-pulled on age only.
        PerFundSeries("fund_health", "/api/fund/fund-health", repull_days=NCEN_REPULL_DAYS),
        PerFundSeries("fund_fee_analysis", "/api/fund/fee-analysis", repull_days=NCEN_REPULL_DAYS),
        PerFundSeries(
            "fund_service_providers", "/api/fund/service-providers", {"limit": PROVIDER_YEARS},
            repull_days=NCEN_REPULL_DAYS,
        ),
        # The funds of each adviser (raw only): a lookup, re-pulled on age only.
        PerAdviser(
            "fund_by_adviser", "/api/fund/by-adviser", page_size=ADVISER_PAGE_SIZE, repull_days=NCEN_REPULL_DAYS,
        ),
    ]


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


@dataclass(frozen=True)
class CompanySnapshot(PerCompany):
    """A per-company forward snapshot: each operating company, backfilled once, then re-pulled daily.

    The endpoint keeps no history, so the raw tier's pulls are its only
    record. A market-wide list is pulled on every run before the planner
    runs; one request per company would spend a day's requests before the
    backfill got any, so these rows are planned instead: never-pulled
    companies with the backfill, then each company again once its last pull
    is ``repull_days`` (one) UTC days old, oldest first, from what the day's
    requests leave. The companies are those of the newest operating list (a
    delisted company has no calendar and no news), or the CIKs a refresh is
    given.

    Examples
    --------
    >>> ENDPOINTS["news"].query("1652044")
    {'limit': 50, 'cik': 1652044}
    >>> ENDPOINTS["news"].repull_days
    1
    """

    repull_days: int | None = 1

    def entities(self, context: PullContext) -> list[str]:
        """Return the given CIKs, or every company of the newest operating list."""
        if context.ciks is not None:
            return [str(cik) for cik in context.ciks]
        if not pulls(context.vendor_root, SYMBOLS, "operating"):
            return []
        rows = read_pull(context.vendor_root, SYMBOLS, "operating")
        return [str(cik) for cik in sorted({int(row["cik"]) for row in rows if row.get("cik")})]


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


def _as_reported_and_derived() -> list[Endpoint]:
    """The as-reported facts, the derived metrics, the share classes, the scores and the trends."""
    on_report = {"triggers": REPORT_FORMS, "repull_days": None}  # re-pulled on a new report only
    return [
        *(
            PerCompany(
                f"as_reported_{period}", "/api/financials/as-reported",
                {"period": period, "limit": STATEMENT_LIMIT}, **on_report,
            )
            for period in ("annual", "quarter")
        ),
        PerCompany("growth_quarter", "/api/financials/growth", {"period": "quarter", "limit": STATEMENT_LIMIT}, **on_report),
        PerCompany(
            "key_metrics_quarter", "/api/financials/key-metrics", {"period": "quarter", "limit": STATEMENT_LIMIT},
            **on_report,
        ),
        # Per-share values as filed: a split-adjusted history would change on every split.
        PerCompany(
            "eps_history_quarter", "/api/earnings/eps-history",
            {"period": "quarter", "limit": STATEMENT_LIMIT, "splitAdjust": "false"}, **on_report,
        ),
        PerCompany(
            "dividend_history_quarter", "/api/earnings/dividend-history",
            {"period": "quarter", "limit": DIVIDEND_HISTORY_LIMIT, "splitAdjust": "false"}, **on_report,
        ),
        PerCompany("share_classes", "/api/ownership/share-classes", **on_report),
        # No history: each pull is a snapshot, re-pulled on a new report and monthly.
        PerCompany("scores", "/api/financials/scores", triggers=REPORT_FORMS, repull_days=SNAPSHOT_REPULL_DAYS),
        PerCompany("trends", "/api/earnings/trends", triggers=REPORT_FORMS, repull_days=SNAPSHOT_REPULL_DAYS),
    ]


def _footnotes() -> list[Endpoint]:
    return [
        PerCompany(
            footnote_endpoint(name, period), spec.path, {"period": period, "limit": STATEMENT_LIMIT},
            triggers=REPORT_FORMS,  # re-pulled on a new report
        )
        for period in PERIODS
        for name, spec in FOOTNOTES.items()
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
    # The 17 footnote tables, annual and quarterly, as many periods as served; re-pulled on a new report.
    *_footnotes(),
    # As-reported facts, derived metrics, share classes, scores and trends.
    *_as_reported_and_derived(),
    # Corporate events: each company's profile (its successions), its XBRL
    # split history, and the declared splits and dividends of the next
    # FORWARD_DAYS days, a forward snapshot kept on every refresh.
    # The profile follows any filing (an 8-K12B succession among them); the split history follows
    # the reports and 8-Ks that tag a split. Both are re-pulled weekly too.
    PerCompany(COMPANY_DETAILS, "/api/company/details", triggers=frozenset(FEED_FORMS.values()), repull_days=7),
    PerCompany(STOCK_SPLITS, "/api/company/stock-splits", triggers=SPLIT_FORMS, repull_days=7),
    MarketList(UPCOMING_STOCK_SPLITS, "/api/company/upcoming-stock-splits", {"days": FORWARD_DAYS},
               page_size=FORWARD_PAGE_SIZE),
    MarketList(UPCOMING_DIVIDENDS, "/api/earnings/upcoming-dividends", {"days": FORWARD_DAYS},
               page_size=FORWARD_PAGE_SIZE),
    # Congress members' periodic transaction reports, House and Senate, by filing day.
    SinceFeed(CONGRESS_HOUSE_TRADES, "/api/congress/house/trades/recent", page_size=CONGRESS_PAGE_SIZE),
    SinceFeed(CONGRESS_SENATE_TRADES, "/api/congress/senate/trades/recent", page_size=CONGRESS_PAGE_SIZE),
    # Insiders, 13D/13G stakes and proxy (DEF 14A) executive data, per company.
    # Every insider transaction of officers and directors (Forms 4 and 5), re-pulled on a new Form 4.
    PerCompany(
        INSIDER_TRANSACTIONS, "/api/insider-transactions", page_size=INSIDER_PAGE_SIZE, triggers=frozenset({"4"}),
    ),
    # Every Schedule 13D/13G filed on the company, re-pulled on a new 13D or 13G.
    PerCompany(BENEFICIAL_OWNERS, "/api/ownership/beneficial-owners/history", triggers=frozenset({"13D", "13G"})),
    # Pay versus performance, governance flags and company-selected measures, re-pulled on a new proxy.
    PerCompany(EXECUTIVES_COMPENSATION, "/api/executives/compensation", triggers=frozenset({"DEF 14A"})),
    PerCompany(EXECUTIVES_GOVERNANCE, "/api/executives/governance", triggers=frozenset({"DEF 14A"})),
    PerCompany(
        EXECUTIVES_PERFORMANCE_MEASURES, "/api/executives/performance-measures", triggers=frozenset({"DEF 14A"}),
    ),
    # Each company's SEC filing list, every form, newest first: the filing panel and accession lookups.
    # Re-pulled when the company files any form the feed follows, and weekly.
    PerCompany(
        FILINGS, "/api/filings", page_size=FILINGS_PAGE_SIZE,
        triggers=frozenset(FEED_FORMS.values()), repull_days=7,
    ),
    # Fund series: lists, N-PORT, N-1A, N-CSR, N-CEN and daily holdings.
    *_funds(),
    # Forward snapshots: the vendor keeps no history of these, so every refresh pulls them.
    # Companies reporting in the next 28 days, with consensus EPS (keyed by ticker).
    MarketList(EARNINGS_UPCOMING, "/api/earnings/upcoming", {"days": 28}, page_size=500),
    # IPO deals, one entity per stage.
    MarketList(
        IPO_CALENDAR, "/api/company/ipo-calendar", page_size=500,
        variants={stage: {"status": stage} for stage in ("filed", "upcoming", "priced", "withdrawn")},
    ),
    # Each company's latest S-1/S-1A of the last year.
    MarketList(RECENT_OFFERINGS, "/api/filings/recent-offerings", {"days": 365}, page_size=100),
    # Per operating company: the predicted next 10-K, 10-Q and DEF 14A, and the latest 50 headlines.
    CompanySnapshot(FILINGS_CALENDAR, "/api/filings/calendar"),
    CompanySnapshot(NEWS, "/api/lookup/news", {"limit": 50}),
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
