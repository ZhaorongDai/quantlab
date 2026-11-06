"""The shapes of a Massive REST pull, and the one page iterator under them.

Every endpoint of the raw tier (``quantlab.acquisition.massive.rest``) is pulled
in one of six shapes. Each shape is one config class, holding only the fields
that shape uses, and its ``pull``:

- ``DatedEndpoint``: rows sorted on a date field, from the day after the
  watermark; the watermark moves after every page.
- ``WindowedEndpoint``: rows filtered on a date field that will not sort on it,
  a window of days at a time; the watermark moves after a whole window.
- ``SnapshotEndpoint``: the whole table, once a day.
- ``AsOfEndpoint``: the whole table as of every session after the watermark.
- ``IdsEndpoint``: one id at a time, for the ids an as-of endpoint shows as new
  or changed.
- ``ConditionTableEndpoint``: the whole table in one file.

``MassiveClient.pull`` hands an endpoint a ``PullRun``, the transport and the
day's bounds, and ``RestEndpoint.pull`` does the rest. ``PullRun.pages`` is the
only loop over ``next_url``, and takes the key out of every URL it follows.
Files land through ``quantlab.acquisition.massive.raw_tier``; this module knows
nothing of the layout. A new shape is one more class here.

Examples
--------
>>> endpoint = DatedEndpoint("splits", "https://api.massive.com/stocks/v1/splits", "execution_date", date(1978, 1, 1))
>>> endpoint.sort_params
{'sort': 'execution_date.asc,ticker.asc'}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote

from loguru import logger

from quantlab.acquisition._support.sessions import xnys_sessions
from quantlab.acquisition.massive.raw_tier import (
    PageFiles,
    stage_as_of,
    stage_snapshot,
    write_conditions,
    write_rest_watermark,
)
from quantlab.dataset.massive.raw import read_rest_as_of, read_rest_records, read_rest_watermark, rest_as_of_days
from quantlab.utils.watermark import holds, resume_from


class MassiveResponseError(ValueError):
    """Massive answered 200 with a page the client cannot read: not JSON, or a record missing a field it needs.

    A ``ValueError``, but the vendor's: the refresh records the endpoint as
    failed, where a plain ``ValueError`` (a bug of ours) is raised.
    """


@dataclass(frozen=True)
class RestPull:
    """What ``MassiveClient.pull`` did for one endpoint.

    Attributes
    ----------
    endpoint : str
        The endpoint's name.
    pages : int
        Pages fetched.
    rows : int
        Records written (a record re-read past the watermark counts again).
    through : date or None
        The endpoint's watermark after the pull; ``None`` for the condition
        table, or while nothing is pulled through a day yet.
    files : tuple of Path
        The page files written, in order.

    Examples
    --------
    >>> RestPull("splits", 2, 7000, date(2026, 10, 15)).rows
    7000
    """

    endpoint: str
    pages: int
    rows: int
    through: date | None
    files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Page:
    """One page of a pull.

    Attributes
    ----------
    records : list of dict
        The page's records, as the vendor returned them.
    more : bool
        Whether another page follows.
    answered : bool, default True
        Whether the page held a ``results`` list at all; a page without one
        has no ``records``, and a shape that must not lose rows refuses it.

    Examples
    --------
    >>> Page([{"ticker": "AAPL"}], more=False).records
    [{'ticker': 'AAPL'}]
    """

    records: list[dict]
    more: bool
    answered: bool = True


@dataclass(frozen=True)
class PullRun:
    """What one pull needs from the client: the transport and the day's bounds.

    Parameters
    ----------
    get_json : callable
        ``get_json(url, params, missing_ok=False)`` returns a page's JSON, or
        ``None`` for a 404 with ``missing_ok``; it retries and raises.
    keyless : callable
        Returns a URL the vendor handed back with any key taken out of it.
    vendor_root : Path
        ``<download-dir>/massive``.
    today : date
        Today in New York.
    end : date or None, default None
        The last day asked for; no upper bound when omitted.
    pulled_at : datetime, default now
        When the pull started, in UTC.

    Examples
    --------
    >>> run = PullRun(None, None, Path("/d/massive"), date(2024, 3, 15), end=date(2024, 3, 1))
    >>> run.last_day
    datetime.date(2024, 3, 1)
    """

    get_json: Callable[..., dict | None]
    keyless: Callable[[str], str]
    vendor_root: Path
    today: date
    end: date | None = None
    pulled_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def last_day(self) -> date:
        """The last day a pull may hold: yesterday in New York, or ``end`` when earlier.

        Today's and future rows (a split announced ahead) are asked for again
        by the next pull.

        Examples
        --------
        >>> PullRun(None, None, Path("/d/massive"), date(2024, 3, 15)).last_day
        datetime.date(2024, 3, 14)
        """
        yesterday = self.today - timedelta(days=1)
        return yesterday if self.end is None else min(yesterday, self.end)

    def pages(self, url: str, params: Mapping[str, str]) -> Iterator[Page]:
        """Yield each page from ``url``, following every ``next_url`` with the key taken out of it.

        A ``next_url`` is followed with no other parameter: its cursor carries
        the filters.

        Parameters
        ----------
        url : str
            The first page's URL.
        params : mapping of str to str
            The first page's query.

        Yields
        ------
        Page
            One per page, in order.

        Examples
        --------
        >>> replies = iter([{"results": [{"id": 1}], "next_url": "https://x/p?cursor=2"}, {"results": []}])
        >>> run = PullRun(lambda url, params: next(replies), str, Path("/d/massive"), date(2024, 3, 15))
        >>> [(page.records, page.more) for page in run.pages("https://x/p", {"limit": "1"})]
        [([{'id': 1}], True), ([], False)]
        """
        following: str | None = url
        while following:
            page = self.get_json(following, params)
            following = page.get("next_url")
            results = page.get("results")
            yield Page(results or [], more=bool(following), answered=results is not None)
            following, params = (self.keyless(following) if following else None), {}


class _Count:
    """The pages and rows of a pull, counted as they land."""

    def __init__(self) -> None:
        self.pages = 0
        self.rows = 0

    def land(self, page: Page, keep: Callable[[list[dict]], None]) -> None:
        """Count ``page`` and ``keep`` its records unless it has none."""
        self.pages += 1
        if page.records:
            keep(page.records)
            self.rows += len(page.records)


@dataclass(frozen=True)
class RestEndpoint(ABC):
    """One REST endpoint: its name and URL, and how a shape pulls it.

    Parameters
    ----------
    name : str
        The endpoint's directory under ``<download-dir>/massive/rest/``.
    url : str
        The endpoint's URL, without a query.
    params : dict, default empty
        Fixed query parameters of every first page (filters of the endpoint's
        own).
    limit : int, default 5000
        Rows per page; the endpoint's maximum.

    Examples
    --------
    >>> SnapshotEndpoint("float", "https://api.massive.com/stocks/vX/float").limit
    5000
    """

    name: str
    url: str
    params: dict[str, str] = field(default_factory=dict, kw_only=True)
    limit: int = field(default=5000, kw_only=True)

    @abstractmethod
    def pull(self, run: PullRun) -> RestPull:
        """Pull the endpoint's new rows into the raw tier.

        Parameters
        ----------
        run : PullRun
            The transport and the day's bounds.

        Returns
        -------
        RestPull
            The pages and rows written and the watermark after the pull.

        Examples
        --------
        Needs the credentials and the network; ``MassiveClient.pull`` builds
        the run::

            client.pull(ENDPOINTS["splits"], "/data/quantlab/downloads")
        """

    @property
    def _query(self) -> dict[str, str]:
        """The query every first page carries."""
        return {**self.params, "limit": str(self.limit)}

    def _done(self, count: _Count, through: date | None, files: tuple[Path, ...]) -> RestPull:
        """Log the pull and return what it did."""
        logger.info(f"Massive: {self.name}: {count.rows:,} row(s) in {count.pages} page(s); watermark {through}.")
        return RestPull(self.name, count.pages, count.rows, through, files)


@dataclass(frozen=True)
class _DateRangedEndpoint(RestEndpoint):
    """An endpoint asked for a range of a date field, resumed from its watermark."""

    date_field: str
    earliest: date
    overlap_days: int = field(default=0, kw_only=True)

    def _resume(self, run: PullRun) -> tuple[date | None, date]:
        """Return the watermark and the first day to ask for."""
        watermark = read_rest_watermark(run.vendor_root, self.name)
        start = resume_from(watermark, self.earliest, self.overlap_days)
        logger.info(f"Massive: pulling {self.name} from {self.date_field} {start} (watermark {watermark}).")
        return watermark, start

    def _advance(self, run: PullRun, watermark: date | None, through: date) -> date | None:
        """Move the watermark to ``through`` and return it; a REST watermark never moves back."""
        if holds(watermark, through):
            return watermark
        write_rest_watermark(run.vendor_root, self.name, self.date_field, through)
        return through


class DatedSort(StrEnum):
    """How a ``DatedEndpoint`` asks for its rows in date order.

    Examples
    --------
    >>> DatedSort.DATE.value
    'date'
    """

    #: ``sort=<date field>.asc,ticker.asc``: the pages of one day in a stable order.
    DATE_THEN_TICKER = "date_then_ticker"
    #: ``sort=<date field>.asc``.
    DATE = "date"
    #: ``sort=<date field>&order=asc``, as the ``/vX/reference`` IPOs and the news take it.
    ORDER_PARAM = "order_param"


@dataclass(frozen=True)
class DatedEndpoint(_DateRangedEndpoint):
    """An endpoint sorted on its date field, pulled from the day after its watermark.

    The first page asks for ``<date_field>.gte`` the day after the watermark
    less ``overlap_days`` (``earliest`` without one), and ``.lte`` ``end`` when
    given. The watermark moves after each page: to the day before its last
    row's day while more pages follow (that day may continue on the next page),
    and to the last day asked for once the last page is in.

    Parameters
    ----------
    name, url, params, limit
        As on ``RestEndpoint``.
    date_field : str
        The ``yyyy-mm-dd`` (or ISO timestamp) field the pull filters and sorts on.
    earliest : date
        The first day a pull without a watermark asks for.
    overlap_days : int, default 0
        Days before the watermark asked for again on every pull, for rows a
        vendor adds or revises after their day.
    sort : DatedSort, default DATE_THEN_TICKER
        How the endpoint takes its order.
    timestamped : bool, default False
        ``date_field`` holds a timestamp, not a day (news' ``published_utc``):
        a pull bounded by ``end`` asks for ``<date_field>.lt`` the day after
        it, so the whole of ``end`` is kept.

    Examples
    --------
    >>> DatedEndpoint("short_volume", "https://api.massive.com/stocks/v1/short-volume", "date",
    ...               date(2024, 2, 6)).date_field
    'date'
    """

    sort: DatedSort = field(default=DatedSort.DATE_THEN_TICKER, kw_only=True)
    timestamped: bool = field(default=False, kw_only=True)

    @property
    def sort_params(self) -> dict[str, str]:
        """The query parameters ordering the pull.

        Returns
        -------
        dict of str to str
            ``sort``, and ``order`` with ``DatedSort.ORDER_PARAM``.

        Examples
        --------
        >>> DatedEndpoint("ipos", "https://x/ipos", "listing_date", date(2008, 12, 1),
        ...               sort=DatedSort.ORDER_PARAM).sort_params
        {'sort': 'listing_date', 'order': 'asc'}
        """
        if self.sort is DatedSort.ORDER_PARAM:
            return {"sort": self.date_field, "order": "asc"}
        if self.sort is DatedSort.DATE:
            return {"sort": f"{self.date_field}.asc"}
        return {"sort": f"{self.date_field}.asc,ticker.asc"}

    def pull(self, run: PullRun) -> RestPull:
        """Pull the rows after the watermark, moving it after each page.

        Parameters
        ----------
        run : PullRun
            The transport and the day's bounds.

        Returns
        -------
        RestPull
            The pages and rows written and the watermark after the pull.

        Raises
        ------
        MassiveResponseError
            If a record lacks the date field the watermark needs.

        Examples
        --------
        Needs the credentials and the network::

            client.pull(ENDPOINTS["dividends"], "/data/quantlab/downloads")
        """
        watermark, start = self._resume(run)
        cap = run.last_day
        bounds = {f"{self.date_field}.gte": start.isoformat()}
        if run.end is not None and self.timestamped:
            bounds[f"{self.date_field}.lt"] = (run.end + timedelta(days=1)).isoformat()
        elif run.end is not None:
            bounds[f"{self.date_field}.lte"] = run.end.isoformat()
        files = PageFiles(run.vendor_root, self.name, run.pulled_at)
        count = _Count()
        for page in run.pages(self.url, {**bounds, **self._query, **self.sort_params}):
            count.land(page, files.add_page)
            through: date | None = cap
            if page.more:
                # The last row's day may continue on the next page.
                through = None
                if page.records:
                    through = _day_of(page.records[-1], self.date_field, self.name) - timedelta(days=1)
            if through is not None:
                watermark = self._advance(run, watermark, min(through, cap))
        return self._done(count, watermark, files.written)


@dataclass(frozen=True)
class WindowedEndpoint(_DateRangedEndpoint):
    """An endpoint that filters on its date field but will not sort on it, pulled by windows.

    Each window is asked with ``.gte``/``.lte`` and no ``sort``. Its rows come
    in no date order, so the watermark moves to a window's last day only once
    all its pages are in, and an interrupted window is asked again whole.

    Parameters
    ----------
    name, url, params, limit
        As on ``RestEndpoint``.
    date_field : str
        The ``yyyy-mm-dd`` field the pull filters on.
    earliest : date
        The first day a pull without a watermark asks for.
    overlap_days : int, default 0
        Days before the watermark asked for again on every pull.
    window_days : int
        Days in one window.

    Raises
    ------
    ValueError
        If ``window_days`` is below 1.

    Examples
    --------
    >>> WindowedEndpoint("10k_sections", "https://x/sections", "filing_date", date(1993, 1, 1),
    ...                  window_days=31).window_days
    31
    """

    window_days: int = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.window_days < 1:
            raise ValueError(f"Massive REST endpoint {self.name!r}: window_days is a positive day count.")

    def pull(self, run: PullRun) -> RestPull:
        """Pull the days after the watermark, one window at a time.

        Parameters
        ----------
        run : PullRun
            The transport and the day's bounds.

        Returns
        -------
        RestPull
            The pages and rows written and the watermark after the pull.

        Examples
        --------
        Needs the credentials and the network::

            client.pull(ENDPOINTS["10k_sections"], "/data/quantlab/downloads")
        """
        watermark, first = self._resume(run)
        cap = run.last_day
        files = PageFiles(run.vendor_root, self.name, run.pulled_at)
        count = _Count()
        while first <= cap:
            last = min(first + timedelta(days=self.window_days - 1), cap)
            bounds = {f"{self.date_field}.gte": first.isoformat(), f"{self.date_field}.lte": last.isoformat()}
            for page in run.pages(self.url, {**self._query, **bounds}):
                count.land(page, files.add_page)
            watermark = self._advance(run, watermark, last)
            first = last + timedelta(days=1)
        return self._done(count, watermark, files.written)


@dataclass(frozen=True)
class SnapshotEndpoint(RestEndpoint):
    """A table pulled whole, one snapshot a day.

    A pull on a day (in New York) that already has one asks for nothing, and
    the watermark is the day of the newest snapshot. The pages appear only once
    the last one is in (``raw_tier.stage_snapshot``).

    Parameters
    ----------
    name, url, params, limit
        As on ``RestEndpoint``.

    Examples
    --------
    >>> SnapshotEndpoint("float", "https://api.massive.com/stocks/vX/float").name
    'float'
    """

    def pull(self, run: PullRun) -> RestPull:
        """Pull the table unless today's snapshot is kept.

        Parameters
        ----------
        run : PullRun
            The transport and the day's bounds.

        Returns
        -------
        RestPull
            The pages and rows written, and the day of the newest snapshot.

        Examples
        --------
        Needs the credentials and the network::

            client.pull(ENDPOINTS["float"], "/data/quantlab/downloads")
        """
        watermark = read_rest_watermark(run.vendor_root, self.name)
        if holds(watermark, run.today):
            logger.info(f"Massive: {self.name}: today's snapshot ({run.today}) is already kept.")
            return RestPull(self.name, 0, 0, watermark)
        staged = stage_snapshot(run.vendor_root, self.name, run.pulled_at)
        count = _Count()
        for page in run.pages(self.url, self._query):
            count.land(page, staged.add_page)
        files = staged.commit()
        write_rest_watermark(run.vendor_root, self.name, "snapshot", run.today)
        return self._done(count, run.today, files)


@dataclass(frozen=True)
class AsOfEndpoint(RestEndpoint):
    """A table served as of a past day, pulled whole as of every session after the watermark.

    The sessions run from ``earliest`` through yesterday in New York (or
    ``end``), each kept apart (``raw.read_rest_as_of``) and appearing once it is
    whole (``raw_tier.stage_as_of``); the watermark is the last session kept.

    Parameters
    ----------
    name, url, params, limit
        As on ``RestEndpoint``.
    as_of_param : str
        The query parameter naming the day.
    earliest : date
        The first session a pull without a watermark asks for.

    Examples
    --------
    >>> AsOfEndpoint("tickers", "https://x/tickers", "date", date(2016, 10, 10)).as_of_param
    'date'
    """

    as_of_param: str
    earliest: date

    def pull(self, run: PullRun) -> RestPull:
        """Pull the table as of every session after the watermark.

        Parameters
        ----------
        run : PullRun
            The transport and the day's bounds.

        Returns
        -------
        RestPull
            The pages and rows written, and the last session kept.

        Examples
        --------
        Needs the credentials and the network::

            client.pull(ENDPOINTS["tickers"], "/data/quantlab/downloads")
        """
        watermark = read_rest_watermark(run.vendor_root, self.name)
        start = resume_from(watermark, self.earliest)
        days = xnys_sessions(start, run.last_day)
        logger.info(f"Massive: pulling {self.name} as of {len(days)} session(s) from {start} (watermark {watermark}).")
        count = _Count()
        files: list[Path] = []
        for day in days:
            staged = stage_as_of(run.vendor_root, self.name, day, datetime.now(UTC))
            for page in run.pages(self.url, {self.as_of_param: day.isoformat(), **self._query}):
                count.land(page, staged.add_page)
            files.extend(staged.commit())
            write_rest_watermark(run.vendor_root, self.name, self.as_of_param, day)
            watermark = day
        return self._done(count, watermark, tuple(files))


@dataclass(frozen=True)
class IdsEndpoint(RestEndpoint):
    """An endpoint served one id at a time, for the ids an as-of endpoint shows as new or changed.

    A pull reads the source's sessions after this endpoint's watermark and asks
    for every id with no record kept yet, or whose ``watch_field`` (the ticker)
    changed on one of them, against the session before. It writes
    one record per id (the vendor's ``results`` object), ``limit`` records a page
    file, and moves the watermark to the last session read once every id is
    asked. An id the vendor does not know (404) is skipped, and asked again with
    the next new session.

    Parameters
    ----------
    name, url, params
        As on ``RestEndpoint``; ``url`` holds ``{id}``.
    limit : int, default 5000
        Records per page file: the vendor answers one record an id.
    ids_from : AsOfEndpoint
        The endpoint whose tables give the ids.
    key : str
        The field of a record naming its id.
    id_field : str, default "composite_figi"
        The source records' field giving the ids.
    watch_field : str, default "ticker"
        The source records' field whose change asks for an id again.

    Raises
    ------
    ValueError
        If ``url`` has no ``{id}``, or ``ids_from`` is not an as-of endpoint.

    Examples
    --------
    >>> source = AsOfEndpoint("tickers", "https://x/tickers", "date", date(2016, 10, 10))
    >>> IdsEndpoint("ticker_events", "https://x/{id}/events", source, key="composite_figi").id_field
    'composite_figi'
    """

    ids_from: AsOfEndpoint
    key: str = field(kw_only=True)
    id_field: str = field(default="composite_figi", kw_only=True)
    watch_field: str = field(default="ticker", kw_only=True)

    def __post_init__(self) -> None:
        if "{id}" not in self.url:
            raise ValueError(f"Massive REST endpoint {self.name!r}: an id endpoint needs {{id}} in its url.")
        if not isinstance(self.ids_from, AsOfEndpoint):
            raise ValueError(f"Massive REST endpoint {self.name!r}: ids_from is an as-of endpoint.")

    def _wanted(self, run: PullRun, days: list[date], watermark: date | None) -> set[str]:
        """Return the ids with no record kept yet, or changed on one of ``days``."""
        source = self.ids_from
        kept = {record[self.key] for record in read_rest_records(run.vendor_root, self.name, self.key)}
        previous: dict[str, object] = {}
        if watermark is not None and watermark in rest_as_of_days(run.vendor_root, source.name):
            previous = {record[self.id_field]: record.get(self.watch_field)
                        for record in read_rest_as_of(run.vendor_root, source.name, watermark)
                        if record.get(self.id_field)}
        wanted: set[str] = set()
        for day in days:
            for record in read_rest_as_of(run.vendor_root, source.name, day):
                ident = record.get(self.id_field)
                if not ident:
                    continue
                value = record.get(self.watch_field)
                if ident not in kept or (ident in previous and previous[ident] != value):
                    wanted.add(ident)
                previous[ident] = value
        return wanted

    def pull(self, run: PullRun) -> RestPull:
        """Pull the ids new or changed in the source's new sessions.

        Parameters
        ----------
        run : PullRun
            The transport and the day's bounds.

        Returns
        -------
        RestPull
            The ids asked for (as ``pages``), the records written, and the last session read.

        Raises
        ------
        MassiveResponseError
            If a record is not the one its id asked for.

        Examples
        --------
        Needs the credentials and the network::

            client.pull(ENDPOINTS["ticker_events"], "/data/quantlab/downloads")
        """
        source = self.ids_from
        watermark = read_rest_watermark(run.vendor_root, self.name)
        days = [day for day in rest_as_of_days(run.vendor_root, source.name)
                if not holds(watermark, day) and (run.end is None or day <= run.end)]
        if not days:
            logger.info(f"Massive: {self.name}: no new session of {source.name} after {watermark}.")
            return RestPull(self.name, 0, 0, watermark)
        wanted = self._wanted(run, days, watermark)
        logger.info(f"Massive: pulling {self.name} for {len(wanted):,} {self.id_field}(s) "
                    f"of {source.name} through {days[-1]}.")
        files = PageFiles(run.vendor_root, self.name, run.pulled_at)
        batch: list[dict] = []
        rows = missing = 0
        for ident in sorted(wanted):
            page = run.get_json(self.url.format(id=quote(ident, safe="")), self.params, missing_ok=True)
            if page is None:
                missing += 1
                continue
            record = page.get("results")
            if not isinstance(record, dict) or record.get(self.key) != ident:
                raise MassiveResponseError(
                    f"Massive {self.name} for {ident!r} returned no record keyed by {self.key!r}: {record}"
                )
            batch.append(record)
            rows += 1
            if len(batch) >= self.limit:
                files.add_page(batch)
                batch = []
        if batch:
            files.add_page(batch)
        write_rest_watermark(run.vendor_root, self.name, source.name, days[-1])
        logger.info(f"Massive: {self.name}: {rows:,} record(s), {missing:,} unknown to Massive (404); "
                    f"watermark {days[-1]}.")
        return RestPull(self.name, len(wanted), rows, days[-1], files.written)


@dataclass(frozen=True)
class ConditionTableEndpoint(RestEndpoint):
    """The condition table: every page, kept as one file named by the pull's UTC time.

    It keeps no watermark, and a day conversion reads the newest file
    (``quantlab.dataset.massive.raw.conditions_file``).

    Parameters
    ----------
    name, url, params, limit
        As on ``RestEndpoint``.

    Examples
    --------
    >>> ConditionTableEndpoint("conditions", "https://api.massive.com/v3/reference/conditions").name
    'conditions'
    """

    def pull(self, run: PullRun) -> RestPull:
        """Pull every page and write the one file.

        Parameters
        ----------
        run : PullRun
            The transport and the day's bounds.

        Returns
        -------
        RestPull
            The pages and rows pulled and the one file; ``through`` is ``None``.

        Raises
        ------
        MassiveResponseError
            If a page holds no ``results``; no file is written, so the newest stays whole.

        Examples
        --------
        Needs the credentials and the network; ``MassiveClient.condition_table``
        is the usual way in::

            client.pull(CONDITIONS, "/data/quantlab/downloads").files[0]
        """
        records: list[dict] = []
        count = _Count()
        for page in run.pages(self.url, self._query):
            # The newest file is the one a conversion reads: a part of the table must not become it.
            if not page.answered:
                raise MassiveResponseError(f"Massive {self.name}: a page holds no results; nothing is written.")
            count.land(page, records.extend)
        return self._done(count, None, (write_conditions(run.vendor_root, run.pulled_at, records),))


def _day_of(record: Mapping[str, object], date_field: str, endpoint: str) -> date:
    """Return the day of a record's date field (a date, or an ISO timestamp's date)."""
    value = record.get(date_field)
    if not value:
        raise MassiveResponseError(
            f"A Massive {endpoint} record has no {date_field!r}; cannot move its watermark: {record}"
        )
    return date.fromisoformat(str(value)[:10])
