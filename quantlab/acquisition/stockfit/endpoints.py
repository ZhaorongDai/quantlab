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
  given.
- ``PerFundSeries``: one request per fund series (``<cik>-<series id>``) of
  the newest fund lists (``quantlab.dataset.stockfit.raw.fund_listings``),
  asked for by its ticker or Composite FIGI (``fund_selector``);
  ``PerFundQuarter``: one per series and quarter, each asked once;
  ``PerAdviser``: one per adviser CRD found in the series' N-CEN service
  providers.

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
from enum import Enum
from functools import lru_cache
from pathlib import Path

from quantlab.acquisition.stockfit.client import StockFitClient, StockFitResponseError
from quantlab.acquisition.stockfit.raw_tier import StagedPull, write_entity_watermark
from quantlab.dataset.stockfit.raw import (
    COMPANY_DELISTED,
    FUND_DAILY_SUPPORTED,
    FUND_LISTS,
    FUND_SYMBOLS,
    SYMBOLS,
    WHOLE_LIST,
    company_ciks,
    daily_supported,
    fund_listings,
    newest_pull_dir,
    read_pull,
    read_watermark,
)
from quantlab.dataset.stockfit.raw import entities as raw_entities

#: The most periods one statement request returns.
STATEMENT_LIMIT = 80

#: The largest page ``/api/symbols`` and ``/api/company/delisted`` serve.
LIST_PAGE_SIZE = 1000

#: The largest page of the fund fee and performance tables.
FUND_PAGE_SIZE = 100

#: The largest page of N-PORT and daily fund holdings.
HOLDINGS_PAGE_SIZE = 1000

#: The largest page of the funds of one adviser.
ADVISER_PAGE_SIZE = 500

#: The most N-CEN reporting years ``/api/fund/service-providers`` returns.
PROVIDER_YEARS = 40


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
    """

    vendor_root: Path
    ciks: tuple[int, ...] | None = None
    fund_series: tuple[str, ...] | None = None


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

    def _query_in(self, vendor_root: Path, entity: str) -> dict:
        """Return the query of one entity's pull from this raw tier; ``query(entity)`` unless the shape needs the tier."""
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
        query = self._query_in(vendor_root, entity)
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

    def _query_in(self, vendor_root: Path, entity: str) -> dict:
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

    def _query_in(self, vendor_root: Path, entity: str) -> dict:
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
        # N-PORT summary and monthly flows, per quarter (panels).
        PerFundSeries("fund_profile", "/api/fund/profile"),
        PerFundSeries("fund_flows", "/api/fund/flows"),
        # N-1A prospectus fee tables (panel) and N-CSR performance per share class (raw only).
        PerFundSeries("fund_fees", "/api/fund/fees", page_size=FUND_PAGE_SIZE),
        PerFundSeries("fund_performance", "/api/fund/performance", page_size=FUND_PAGE_SIZE),
        # N-PORT holdings, each quarter once (raw only).
        PerFundQuarter("fund_holdings", "/api/fund/holdings", page_size=HOLDINGS_PAGE_SIZE),
        # Daily website holdings, a snapshot each refresh (raw only).
        PerFundSeries("fund_holdings_daily", "/api/fund/holdings/daily", page_size=HOLDINGS_PAGE_SIZE, daily=True),
        # N-CEN: fund health and fee analysis (panel), service providers (raw only).
        PerFundSeries("fund_health", "/api/fund/fund-health"),
        PerFundSeries("fund_fee_analysis", "/api/fund/fee-analysis"),
        PerFundSeries("fund_service_providers", "/api/fund/service-providers", {"limit": PROVIDER_YEARS}),
        # The funds of each adviser (raw only): a lookup.
        PerAdviser("fund_by_adviser", "/api/fund/by-adviser", page_size=ADVISER_PAGE_SIZE),
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
    # Fund series: lists, N-PORT, N-1A, N-CSR, N-CEN and daily holdings.
    *_funds(),
)}
