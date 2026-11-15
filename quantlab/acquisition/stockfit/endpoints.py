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
    last_pulled,
    read_watermark,
)

#: The most periods one statement request returns.
STATEMENT_LIMIT = 80

#: The most periods ``/api/earnings/dividend-history`` returns.
DIVIDEND_HISTORY_LIMIT = 40

#: Days after which a company's scores and trends (snapshots without history) are re-pulled.
SNAPSHOT_REPULL_DAYS = 30

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
    # As-reported facts, derived metrics, share classes, scores and trends.
    *_as_reported_and_derived(),
    # Congress members' periodic transaction reports, House and Senate, by filing day.
    SinceFeed(CONGRESS_HOUSE_TRADES, "/api/congress/house/trades/recent", page_size=CONGRESS_PAGE_SIZE),
    SinceFeed(CONGRESS_SENATE_TRADES, "/api/congress/senate/trades/recent", page_size=CONGRESS_PAGE_SIZE),
)}
