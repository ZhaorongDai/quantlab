"""StockFit's raw tier: where each pull lives, and how it is read.

StockFit (``https://api.stockfit.io/v1``) serves SEC-filing data as JSON, one
request per endpoint and entity (a company's CIK, or a market-wide list).
The raw tier keeps every pull of one endpoint for one entity as a directory
named by the pull's UTC stamp, under the StockFit directory of the download
root::

    <root>/downloads/stockfit/symbols/operating/20261116T140000000000/000001.json
    <root>/downloads/stockfit/symbols/operating/20261116T140000000000/_pull.json
    <root>/downloads/stockfit/income_statement_quarter/1093691/20261116T140000000000/000001.json
    <root>/downloads/stockfit/income_statement_quarter/1093691/_watermark.json

A pull directory holds the response body of each page as the vendor returned
it (``000001.json``, ``000002.json``, ... for a paged list; one file
otherwise) and a manifest (``_pull.json``: the endpoint, the entity, the
pull time, the page count, the key a paged body keeps its records under, and
the content hash of the pages). A pull waits in the entity's ``_pending``
directory, which nothing reads, and appears by one directory rename, so a
pull directory is always a whole pull (ADR 0053). A pull whose content equals
the entity's newest pull is not published: only the entity's watermark
(``_watermark.json``, the UTC day of the last pull) moves. So the pulls of an
entity are the changes the vendor made to it, each kept.

The writer (``quantlab.acquisition.stockfit.raw_tier``) takes every path from
here; the readers are ``pulls``, ``read_pull``, ``read_pull_as_of``,
``entities``, ``read_watermark``, ``last_pulled`` and ``company_ciks``, and
for 13F ``thirteen_f_managers`` and ``thirteen_f_periods``. This module only
reads.

An endpoint asked per quarter (13F holders of a company, a manager's
portfolio) has one entity per owner and quarter, named
``<cik>_<quarter end>`` (``quarter_entity``).

Examples
--------
>>> pull_dir("/d/stockfit", "income_statement_quarter", "1093691", datetime(2026, 11, 16, 14, tzinfo=UTC))
PosixPath('/d/stockfit/income_statement_quarter/1093691/20261116T140000000000')
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from quantlab.utils.watermark import WATERMARK_FILE
from quantlab.utils.watermark import read_watermark as _read_watermark

#: The vendor directory under the download root.
VENDOR_DIR = "stockfit"

#: The pull stamp's format, as in the Massive and Sharadar raw tiers.
STAMP_FORMAT = "%Y%m%dT%H%M%S%f"

#: The directory, in an entity's, where a pull waits until it is whole.
PENDING_DIR = "_pending"

#: The manifest file of one pull.
MANIFEST_FILE = "_pull.json"

#: Suffix of one page file.
PAGE_SUFFIX = ".json"

#: Matches the directory of one pull, named by its stamp.
_PULL_DIR_GLOB = "[0-9]*T[0-9]*"

#: The endpoint of the company list (``/api/symbols``), one entity per listing status.
SYMBOLS = "symbols"

#: The endpoint of the delisted companies with their delisting day (``/api/company/delisted``).
COMPANY_DELISTED = "company_delisted"

#: The endpoint of each company's profile, with its successions (``/api/company/details``).
COMPANY_DETAILS = "company_details"

#: The endpoint of each company's XBRL split history (``/api/company/stock-splits``).
STOCK_SPLITS = "stock_splits"

#: The forward snapshot of the declared splits (``/api/company/upcoming-stock-splits``).
UPCOMING_STOCK_SPLITS = "upcoming_stock_splits"

#: The forward snapshot of the declared dividends (``/api/earnings/upcoming-dividends``).
UPCOMING_DIVIDENDS = "upcoming_dividends"

#: The endpoints of the House and Senate members' trade reports (``/api/congress/*/trades/recent``).
CONGRESS_HOUSE_TRADES = "congress_house_trades"
CONGRESS_SENATE_TRADES = "congress_senate_trades"

#: The forward snapshots, pulled on every refresh: the market-wide earnings
#: calendar with consensus (``/api/earnings/upcoming``), IPO deals by stage
#: (``/api/company/ipo-calendar``), recent S-1 offerings
#: (``/api/filings/recent-offerings``), and per company the predicted filing
#: calendar (``/api/filings/calendar``) and the latest headlines (``/api/lookup/news``).
EARNINGS_UPCOMING = "earnings_upcoming"
IPO_CALENDAR = "ipo_calendar"
RECENT_OFFERINGS = "recent_offerings"
FILINGS_CALENDAR = "filings_calendar"
NEWS = "news"

#: The entity of an endpoint that is one market-wide list.
WHOLE_LIST = "all"

#: The pulls whose rows name the companies: (endpoint, entity).
COMPANY_LISTS: tuple[tuple[str, str], ...] = (
    (SYMBOLS, "operating"),
    (SYMBOLS, "delisted"),
    (COMPANY_DELISTED, WHOLE_LIST),
)


def _plain(name: str, what: str) -> str:
    if not name or "/" in name or name.startswith((".", "_")):
        raise ValueError(f"StockFit {what} {name!r} must be a plain directory name.")
    return name


def endpoint_dir(vendor_root: str | Path, endpoint: str) -> Path:
    """Return the directory of one endpoint's entities.

    Examples
    --------
    >>> endpoint_dir("/d/stockfit", "symbols")
    PosixPath('/d/stockfit/symbols')
    """
    return Path(vendor_root) / _plain(endpoint, "endpoint")


def entity_dir(vendor_root: str | Path, endpoint: str, entity: str) -> Path:
    """Return the directory of one entity's pulls and watermark.

    Examples
    --------
    >>> entity_dir("/d/stockfit", "balance_sheet_quarter", "1093691")
    PosixPath('/d/stockfit/balance_sheet_quarter/1093691')
    """
    return endpoint_dir(vendor_root, endpoint) / _plain(entity, "entity")


def pull_stamp(pulled_at: datetime) -> str:
    """Return the name of the pull made at ``pulled_at``: its UTC stamp.

    Examples
    --------
    >>> pull_stamp(datetime(2026, 11, 16, 14, tzinfo=UTC))
    '20261116T140000000000'
    """
    return pulled_at.astimezone(UTC).strftime(STAMP_FORMAT)


def pull_dir(vendor_root: str | Path, endpoint: str, entity: str, pulled_at: datetime) -> Path:
    """Return the directory of the pull made at ``pulled_at`` (UTC).

    Examples
    --------
    >>> pull_dir("/d/stockfit", "symbols", "operating", datetime(2026, 11, 16, tzinfo=UTC)).name
    '20261116T000000000000'
    """
    return entity_dir(vendor_root, endpoint, entity) / pull_stamp(pulled_at)


def pending_dir(vendor_root: str | Path, endpoint: str, entity: str) -> Path:
    """Return where an entity's pull waits until it is whole; nothing reads it.

    Examples
    --------
    >>> pending_dir("/d/stockfit", "symbols", "operating")
    PosixPath('/d/stockfit/symbols/operating/_pending')
    """
    return entity_dir(vendor_root, endpoint, entity) / PENDING_DIR


def page_name(page: int) -> str:
    """Return the file name of page ``page`` (from 1) of a pull.

    Examples
    --------
    >>> page_name(3)
    '000003.json'
    """
    return f"{page:06d}{PAGE_SUFFIX}"


def watermark_file(vendor_root: str | Path, endpoint: str, entity: str) -> Path:
    """Return the watermark file of one entity.

    Examples
    --------
    >>> watermark_file("/d/stockfit", "symbols", "operating").name
    '_watermark.json'
    """
    return entity_dir(vendor_root, endpoint, entity) / WATERMARK_FILE


def read_watermark(vendor_root: str | Path, endpoint: str, entity: str) -> date | None:
    """Return the UTC day an entity was last pulled, published or not; ``None`` if never.

    Examples
    --------
    >>> read_watermark("/nonexistent/stockfit", "symbols", "operating") is None
    True
    """
    return _read_watermark(watermark_file(vendor_root, endpoint, entity))


def last_pulled(vendor_root: str | Path, endpoint: str, entity: str) -> datetime | None:
    """Return when an entity was last pulled (UTC), published or not; ``None`` if never.

    The time of the pull, which the watermark file records beside its day;
    the refresh compares it with the filings made since.

    Examples
    --------
    >>> last_pulled("/nonexistent/stockfit", "symbols", "operating") is None
    True
    """
    path = watermark_file(vendor_root, endpoint, entity)
    if not path.exists():
        return None
    stamp = json.loads(path.read_text(encoding="utf-8"))["pulled_at"]
    return datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=UTC)


def _pull_dirs(vendor_root: str | Path, endpoint: str, entity: str) -> dict[datetime, Path]:
    root = entity_dir(vendor_root, endpoint, entity)
    found = sorted(root.glob(_PULL_DIR_GLOB), key=lambda path: path.name) if root.exists() else []
    for path in found:
        if not path.is_dir():
            raise ValueError(f"StockFit file {str(path)!r} is where a pull's directory belongs: move it.")
    return {datetime.strptime(path.name, STAMP_FORMAT).replace(tzinfo=UTC): path for path in found}


def newest_pull_dir(vendor_root: str | Path, endpoint: str, entity: str) -> Path | None:
    """Return the directory of an entity's newest published pull; ``None`` if it has none.

    Examples
    --------
    >>> newest_pull_dir("/nonexistent/stockfit", "symbols", "operating") is None
    True
    """
    found = _pull_dirs(vendor_root, endpoint, entity)
    return found[max(found)] if found else None


def pulls(vendor_root: str | Path, endpoint: str, entity: str) -> list[datetime]:
    """Return the pull times (UTC) of an entity's published pulls, oldest first.

    Each is a change: a pull equal to the one before it is never published.

    Examples
    --------
    >>> pulls("/nonexistent/stockfit", "symbols", "operating")
    []
    """
    return list(_pull_dirs(vendor_root, endpoint, entity))


def entities(vendor_root: str | Path, endpoint: str) -> list[str]:
    """Return the entities of an endpoint with at least one published pull, sorted by name.

    Examples
    --------
    >>> entities("/nonexistent/stockfit", "income_statement_quarter")
    []
    """
    root = endpoint_dir(vendor_root, endpoint)
    if not root.exists():
        return []
    return sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_")) and any(path.glob(_PULL_DIR_GLOB))
    )


def read_manifest(directory: str | Path) -> dict:
    """Return the manifest of one pull directory.

    Examples
    --------
    Needs a raw tier on disk::

        read_manifest(pull_dir(root, "symbols", "operating", pulled_at))["sha256"]
    """
    return json.loads((Path(directory) / MANIFEST_FILE).read_text(encoding="utf-8"))


def read_pull_dir(directory: str | Path) -> object:
    """Return what one pull directory holds: the body, or a paged list's records in order.

    A paged list (its manifest names the key its records are under) is
    returned as one list of every page's records; any other pull as its one
    body, verbatim.

    Examples
    --------
    Needs a raw tier on disk::

        read_pull_dir(pull_dir(root, "symbols", "operating", pulled_at))
    """
    directory = Path(directory)
    manifest = read_manifest(directory)
    bodies = [
        json.loads((directory / page_name(page)).read_text(encoding="utf-8"))
        for page in range(1, manifest["pages"] + 1)
    ]
    key = manifest.get("records")
    if key is None:
        (body,) = bodies
        return body
    return [record for body in bodies for record in body[key]]


def read_pull(vendor_root: str | Path, endpoint: str, entity: str, pulled_at: datetime | None = None) -> object:
    """Return one published pull of an entity, the newest by default.

    Parameters
    ----------
    vendor_root : str or Path
        ``<root>/downloads/stockfit``.
    endpoint, entity : str
        The endpoint's and the entity's directory names.
    pulled_at : datetime, optional
        The pull's time, from ``pulls``; the newest pull when omitted.

    Returns
    -------
    object
        The response body, verbatim; for a paged list, every page's records
        as one list.

    Raises
    ------
    ValueError
        If the entity has no such pull.

    Examples
    --------
    Needs a raw tier on disk; every version of a company's quarterly income statement::

        root = "/data/downloads/stockfit"
        versions = [read_pull(root, "income_statement_quarter", "1093691", at)
                    for at in pulls(root, "income_statement_quarter", "1093691")]
    """
    found = _pull_dirs(vendor_root, endpoint, entity)
    if not found or (pulled_at is not None and pulled_at not in found):
        raise ValueError(f"StockFit {endpoint}/{entity} has no pull made at {pulled_at} under {str(vendor_root)!r}.")
    return read_pull_dir(found[max(found) if pulled_at is None else pulled_at])


def read_pull_as_of(vendor_root: str | Path, endpoint: str, entity: str, at: datetime) -> object | None:
    """Return the newest pull of an entity made at or before ``at``; ``None`` if none was.

    Because unchanged pulls are not published, this is what the vendor
    served for the entity at ``at``, as far as the raw tier saw it.

    Examples
    --------
    >>> read_pull_as_of("/nonexistent/stockfit", "symbols", "operating", datetime(2026, 1, 1, tzinfo=UTC)) is None
    True
    """
    found = [stamp for stamp in _pull_dirs(vendor_root, endpoint, entity) if stamp <= at]
    return read_pull(vendor_root, endpoint, entity, max(found)) if found else None


def company_ciks(vendor_root: str | Path) -> list[int]:
    """Return the CIK of every company in the newest company lists, sorted.

    The companies are the rows of ``COMPANY_LISTS`` (the operating and the
    delisted companies); a list never pulled adds none.

    Examples
    --------
    >>> company_ciks("/nonexistent/stockfit")
    []
    """
    ciks: set[int] = set()
    for endpoint, entity in COMPANY_LISTS:
        if pulls(vendor_root, endpoint, entity):
            ciks.update(int(row["cik"]) for row in read_pull(vendor_root, endpoint, entity) if row.get("cik"))
    return sorted(ciks)


def as_number(value) -> float:
    """Return a served value as a panel's float: the one coercion every StockFit panel uses.

    A JSON number is its float; a boolean is 1.0 or 0.0 (a flag is a
    number in a panel); anything else (a null, text, a nested object) is NaN,
    never ``None``, so a missing value is NaN in every panel.

    Examples
    --------
    >>> as_number(3), as_number(2.5), as_number(True), as_number(False)
    (3.0, 2.5, 1.0, 0.0)
    >>> as_number(None), as_number("12.5"), as_number({})
    (nan, nan, nan)
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return float("nan")


# -- fund series -----------------------------------------------------------------

#: The endpoint of the fund lists (``/api/symbols`` by fund kind), one entity per kind and listing status.
FUND_SYMBOLS = "fund_symbols"

#: StockFit's fund kinds (its ``SecurityType`` but ``stock``): ETFs, open-end and closed-end funds,
#: and series whose kind is not yet evidenced.
FUND_KINDS = ("etf", "mf", "cef", "fund")

#: The pulls whose rows name the fund series: (endpoint, entity), operating funds first.
FUND_LISTS: tuple[tuple[str, str], ...] = tuple(
    (FUND_SYMBOLS, f"{kind}_{status}") for status in ("operating", "delisted") for kind in FUND_KINDS
)

#: The endpoint of the tickers with daily website holdings (``/api/fund/holdings/daily/supported-funds``).
FUND_DAILY_SUPPORTED = "fund_daily_supported"


def fund_series_name(cik: int, series_id: int) -> str:
    """Return the name of a fund series: the trust's CIK and the series id, ``<cik>-<series id>``.

    It is the series' entity in the raw tier and its symbol in every fund panel.

    Examples
    --------
    >>> fund_series_name(36405, 2839)
    '36405-2839'
    """
    return f"{int(cik)}-{int(series_id)}"


def fund_series_key(name: str) -> tuple[int, int]:
    """Return a fund series' CIK and series id, the order fund series sort in.

    Examples
    --------
    >>> fund_series_key("36405-2839")
    (36405, 2839)
    """
    cik, series_id = name.split("-")
    return int(cik), int(series_id)


def fund_listings(vendor_root: str | Path) -> dict[str, dict]:
    """Return every fund series of the newest fund lists with what selects it, in series order.

    A row of a fund list is one series of a trust (``cik`` and a non-zero
    ``seriesId``); a row without a series id is not a fund series. A series
    listed by several rows keeps its first row's Composite FIGI and kind and
    every row's tickers, operating lists first.

    Returns
    -------
    dict
        Series name (``fund_series_name``) to ``{"symbols": [...],
        "composite_figi": str or None, "kind": str}``, sorted by CIK, then
        series id.

    Examples
    --------
    >>> fund_listings("/nonexistent/stockfit")
    {}
    """
    found: dict[str, dict] = {}
    for endpoint, entity in FUND_LISTS:
        if not pulls(vendor_root, endpoint, entity):
            continue
        for row in read_pull(vendor_root, endpoint, entity):
            if not row.get("cik") or not row.get("seriesId"):
                continue
            listing = found.setdefault(fund_series_name(row["cik"], row["seriesId"]), {
                "symbols": [], "composite_figi": row.get("compositeFigi"), "kind": row.get("type"),
            })
            listing["symbols"] += [s for s in row.get("symbols") or [] if s not in listing["symbols"]]
    return {name: found[name] for name in sorted(found, key=fund_series_key)}


def fund_series(vendor_root: str | Path) -> list[str]:
    """Return every fund series of the newest fund lists, sorted by CIK, then series id.

    Two series of one trust are two fund series.

    Examples
    --------
    >>> fund_series("/nonexistent/stockfit")
    []
    """
    return list(fund_listings(vendor_root))


def daily_supported(vendor_root: str | Path) -> list[str]:
    """Return the tickers whose daily website holdings StockFit serves, as of the newest pull; ``[]`` if none.

    Examples
    --------
    >>> daily_supported("/nonexistent/stockfit")
    []
    """
    if not pulls(vendor_root, FUND_DAILY_SUPPORTED, WHOLE_LIST):
        return []
    return [str(ticker) for ticker in read_pull(vendor_root, FUND_DAILY_SUPPORTED, WHOLE_LIST)]


def newest_watermark(vendor_root: str | Path, endpoint: str) -> date | None:
    """Return the latest watermark among an endpoint's entities: the last UTC day it was pulled.

    Every entity counts, its pull published or not (an entity the vendor
    holds nothing for still has a watermark). ``None`` if the endpoint was
    never pulled.

    Examples
    --------
    >>> newest_watermark("/nonexistent/stockfit", "stock_splits") is None
    True
    """
    root = endpoint_dir(vendor_root, endpoint)
    if not root.exists():
        return None
    days = [
        day for path in root.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
        and (day := read_watermark(vendor_root, endpoint, path.name)) is not None
    ]
    return max(days, default=None)


# -- 13F -----------------------------------------------------------------------

#: The endpoint of a company's 13F holder rows for one quarter
#: (``/api/ownership/institutional-holders?cik=...&reportDate=...``): one row
#: per manager and security (CUSIP) of the company; entity ``<cik>_<quarter end>``.
INSTITUTIONAL_HOLDERS = "institutional_holders"

#: The endpoint of a manager's 13F filings, originals and amendments
#: (``/api/filings?cik=...&type=13F-HR``); entity ``<manager cik>``.
MANAGER_13F_FILINGS = "manager_13f_filings"

#: The endpoint of a manager's 13F portfolio for one quarter
#: (``/api/ownership/portfolio?cik=...&reportDate=...``), raw only; entity
#: ``<manager cik>_<quarter end>``.
MANAGER_PORTFOLIO = "manager_portfolio"


def quarter_entity(owner: int | str, quarter: date) -> str:
    """Return the entity of one owner's (a company's or a manager's) pull for one quarter.

    Examples
    --------
    >>> quarter_entity(1652044, date(2024, 3, 31))
    '1652044_2024-03-31'
    """
    return f"{owner}_{quarter.isoformat()}"


def split_quarter_entity(entity: str) -> tuple[int, date]:
    """Return the owner's CIK and the quarter end of a ``quarter_entity``.

    Examples
    --------
    >>> split_quarter_entity("1652044_2024-03-31")
    (1652044, datetime.date(2024, 3, 31))
    """
    owner, quarter = entity.rsplit("_", 1)
    return int(owner), date.fromisoformat(quarter)


def manager_holdings(vendor_root: str | Path) -> dict[int, list[str]]:
    """Return, per 13F manager, the holder entities (``<cik>_<quarter end>``) whose newest pull names it.

    Examples
    --------
    >>> manager_holdings("/nonexistent/stockfit")
    {}
    """
    held: dict[int, list[str]] = {}
    for entity in entities(vendor_root, INSTITUTIONAL_HOLDERS):
        managers = {
            int(row["managerCik"]) for row in read_pull(vendor_root, INSTITUTIONAL_HOLDERS, entity)
            if row.get("managerCik")
        }
        for manager in managers:
            held.setdefault(manager, []).append(entity)
    return held


def thirteen_f_managers(vendor_root: str | Path) -> list[int]:
    """Return the CIK of every manager in the newest 13F holder rows of every company and quarter, sorted.

    Examples
    --------
    >>> thirteen_f_managers("/nonexistent/stockfit")
    []
    """
    return sorted(manager_holdings(vendor_root))


def thirteen_f_periods(vendor_root: str | Path, manager: int | str) -> list[date]:
    """Return the quarter ends a manager's newest 13F filing list reports, newest first.

    Examples
    --------
    >>> thirteen_f_periods("/nonexistent/stockfit", 1067983)
    []
    """
    if not pulls(vendor_root, MANAGER_13F_FILINGS, str(manager)):
        return []
    periods = {
        date.fromisoformat(row["periodEndDate"][:10])
        for row in read_pull(vendor_root, MANAGER_13F_FILINGS, str(manager))
        if row.get("periodEndDate")
    }
    return sorted(periods, reverse=True)
