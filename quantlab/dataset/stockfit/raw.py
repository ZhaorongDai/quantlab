"""StockFit's raw tier: where each pull lives, and how it is read.

StockFit (``https://api.stockfit.io/v1``) serves SEC-filing data as JSON, one
request per endpoint and entity (a company's CIK, or a market-wide list).
The raw tier keeps every pull of one endpoint for one entity as a directory
named by the pull's UTC stamp, under the StockFit directory of the download
root::

    <download-dir>/stockfit/symbols/operating/20261116T140000000000/000001.json
    <download-dir>/stockfit/symbols/operating/20261116T140000000000/_pull.json
    <download-dir>/stockfit/income_statement_quarter/1093691/20261116T140000000000/000001.json
    <download-dir>/stockfit/income_statement_quarter/1093691/_watermark.json

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
``entities``, ``read_watermark``, ``last_pulled`` and ``company_ciks``. This module only
reads.

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
    >>> entity_dir("/d/stockfit", "balance_sheet_ttm", "1093691")
    PosixPath('/d/stockfit/balance_sheet_ttm/1093691')
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
        ``<download-dir>/stockfit``.
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

        root = "/data/quantlab/downloads/stockfit"
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
