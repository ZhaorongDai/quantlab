"""When a StockFit row becomes available: the one rule every StockFit panel places its rows by.

StockFit dates a filing by its EDGAR acceptance time (``acceptedAt``, UTC to
the second) and its filing date (``dateFiled``). A row is available on:

- with an acceptance time: its New York day when it was accepted before
  that day's NYSE close (16:00, or 13:00 on a half day), otherwise the next
  NYSE trading day; a row accepted on a day without a session rolls to the
  next trading day. So a close-based backtest never trades on a filing made
  after the close;
- with only a filing date: that day, or the next trading day when it is not
  one (the Sharadar convention: a filing counts on its filing date).

A pull of a snapshot the vendor keeps no history of counts by the same rule,
its pull time taken as the acceptance time (``pull_day``).

``release_day`` gives each row the calendar day the rule picks and
``place_released`` rolls it onto a ``TradingCalendar``; ``pull_day`` is the
rule for one time. ``nyse_calendar`` is the calendar every StockFit panel
uses: NYSE's sessions through the day the endpoints it reads are complete
through, the day before the oldest of their newest pulls
(``quantlab.dataset.stockfit.raw.newest_watermark``). The daily refresh
re-pulls only the entities a filing touched, so an endpoint is dated by the
newest pull of any of its entities, not the oldest.

Examples
--------
A filing accepted at 16:30 on Friday 2026-11-06 is available on Monday:

>>> import polars as pl
>>> from datetime import date
>>> rows = pl.DataFrame({"accepted_at": ["2026-11-06T21:30:00Z"], "date_filed": [date(2026, 11, 6)]}).with_columns(
...     pl.col("accepted_at").str.to_datetime(time_zone="UTC"))
>>> release_day(rows).get_column("release_day").to_list()
[datetime.date(2026, 11, 7)]
>>> calendar = TradingCalendar.of_nyse(date(2026, 11, 13), start="2026-11-01", end="2026-11-30")
>>> place_released(rows, calendar).get_column("available").dt.date().to_list()
[datetime.date(2026, 11, 9)]
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import polars as pl

from quantlab.dataset._support.point_in_time import NYSE_FIRST_DAY, TradingCalendar
from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.stockfit.raw import newest_watermark

#: The time zone of the sessions.
NEW_YORK = "America/New_York"


class _EdgarSessions(XnysSessionCalendar):
    """NYSE's sessions from ``NYSE_FIRST_DAY``, as far back as EDGAR's acceptance times."""

    CALENDAR_START = NYSE_FIRST_DAY


@lru_cache(maxsize=1)
def _sessions() -> _EdgarSessions:
    return _EdgarSessions()


def _closes(days: list[date]) -> pl.DataFrame:
    """Return the close of each session among ``days`` as a UTC time: ``_local_day``, ``_close``."""
    sessions = _sessions()
    calendar = sessions.calendar
    first, last = calendar.first_session.date(), calendar.last_session.date()
    inside = [day for day in days if first <= day <= last and sessions.is_session(day)]
    return sessions.session_bounds(inside).select(
        pl.col("date").alias("_local_day"), pl.col("close").dt.replace_time_zone("UTC").alias("_close")
    )


def parse_accepted(text: str | None) -> datetime | None:
    """Return an ISO acceptance time as an aware UTC datetime, or ``None``.

    Examples
    --------
    >>> parse_accepted("2026-11-09T22:01:51Z")
    datetime.datetime(2026, 11, 9, 22, 1, 51, tzinfo=datetime.timezone.utc)
    >>> parse_accepted(None) is None
    True
    """
    return None if not text else datetime.fromisoformat(text).astimezone(UTC)


def release_day(rows: pl.DataFrame, accepted: str = "accepted_at", filed: str = "date_filed") -> pl.DataFrame:
    """Return ``rows`` with ``release_day``: the calendar day each becomes public, before rolling onto trading days.

    Parameters
    ----------
    rows : pl.DataFrame
        Rows with an acceptance-time column ``accepted`` (UTC datetime,
        null when unknown) and a filing-date column ``filed`` (date).
    accepted, filed : str
        The two columns' names.

    Returns
    -------
    pl.DataFrame
        ``rows`` and ``release_day`` (date): the New York day of the
        acceptance when before that session's close, the day after it
        otherwise; the filing date when there is no acceptance time; null
        when there is neither.

    Examples
    --------
    NYSE closed at 13:00 on 2024-07-03:

    >>> import polars as pl
    >>> from datetime import date
    >>> rows = pl.DataFrame({
    ...     "accepted_at": ["2024-07-03T16:59:00Z", "2024-07-03T17:00:00Z", None],
    ...     "date_filed": [date(2024, 7, 3), date(2024, 7, 3), date(2024, 7, 6)],
    ... }).with_columns(pl.col("accepted_at").str.to_datetime(time_zone="UTC"))
    >>> release_day(rows).get_column("release_day").to_list()
    [datetime.date(2024, 7, 3), datetime.date(2024, 7, 4), datetime.date(2024, 7, 6)]
    """
    at = pl.col(accepted).dt.convert_time_zone("UTC")
    rows = rows.with_columns(at.dt.convert_time_zone(NEW_YORK).dt.date().alias("_local_day"))
    closes = _closes(rows.get_column("_local_day").drop_nulls().unique().sort().to_list())
    before_close = (pl.col(accepted).dt.convert_time_zone("UTC") < pl.col("_close")).fill_null(False)
    return (
        rows.join(closes, on="_local_day", how="left")
        .with_columns(
            pl.when(pl.col(accepted).is_null())
            .then(pl.col(filed))
            .when(before_close)
            .then(pl.col("_local_day"))
            .otherwise(pl.col("_local_day") + timedelta(days=1))
            .alias("release_day")
        )
        .drop("_local_day", "_close")
    )


def place_released(
    rows: pl.DataFrame,
    calendar: TradingCalendar,
    accepted: str = "accepted_at",
    filed: str = "date_filed",
    *,
    drop_earlier: bool = True,
) -> pl.DataFrame:
    """Return ``rows`` with ``available``: the first trading day on or after each row's ``release_day``.

    A row with neither time, or past the day the calendar is complete
    through, is dropped; so, with ``drop_earlier``, is a row dated before
    the calendar's first day (an event), which is otherwise placed on that
    day (a state that still holds). Row order is not kept.

    Examples
    --------
    A filing dated Saturday with no acceptance time lands on Monday:

    >>> import polars as pl
    >>> from datetime import date
    >>> rows = pl.DataFrame({"accepted_at": [None], "date_filed": [date(2026, 11, 7)]},
    ...                     schema={"accepted_at": pl.Datetime("us", "UTC"), "date_filed": pl.Date})
    >>> calendar = TradingCalendar.of_nyse(date(2026, 11, 13), start="2026-11-01", end="2026-11-30")
    >>> place_released(rows, calendar).get_column("available").dt.date().to_list()
    [datetime.date(2026, 11, 9)]
    """
    dated = release_day(rows, accepted, filed).filter(pl.col("release_day").is_not_null())
    return calendar.place(dated, on="release_day", drop_earlier=drop_earlier)


@lru_cache(maxsize=4096)
def _close_of(day: date) -> datetime | None:
    """Return the NYSE close of ``day`` as a UTC time; ``None`` without a session."""
    closes = _closes([day])
    return closes.get_column("_close").item() if closes.height else None


def pull_day(pulled_at: datetime) -> date:
    """Return the day what a pull made at ``pulled_at`` saw counts from: ``release_day`` of one time.

    A pull made before its New York day's NYSE close (13:00 on a half day)
    counts on that day; one made at or after the close, or on a day without
    a session, on the next day, which ``TradingCalendar.place`` rolls onto
    the next trading day.

    Parameters
    ----------
    pulled_at : datetime
        The pull's time, timezone-aware.

    Returns
    -------
    date
        A New York calendar day.

    Examples
    --------
    >>> pull_day(datetime(2026, 11, 16, 14, tzinfo=UTC))  # 09:00 in New York
    datetime.date(2026, 11, 16)
    >>> pull_day(datetime(2024, 7, 3, 18, tzinfo=UTC))  # 14:00, after the 13:00 close
    datetime.date(2024, 7, 4)
    """
    at = pulled_at.astimezone(UTC)
    day = pl.Series([at]).dt.convert_time_zone(NEW_YORK).dt.date().item()
    close = _close_of(day)
    return day if close is not None and at < close else day + timedelta(days=1)


def nyse_calendar(
    vendor_root: str | Path, endpoints: str | Sequence[str], *, start, end, owner: str
) -> TradingCalendar:
    """Return NYSE's trading days through the day the endpoints a panel reads are complete through.

    That is the day before the oldest of their newest watermarks (each the
    UTC day of the latest pull of any of the endpoint's entities, published
    or not, or of the latest refresh that found none of them due,
    ``quantlab.dataset.stockfit.raw.newest_watermark``): a refresh made on a
    day holds every row public by the close of the day before, whatever
    time it ran.

    Parameters
    ----------
    vendor_root : str or Path
        ``<root>/downloads/stockfit``.
    endpoints : str or sequence of str
        The raw tier's endpoints the panel reads.
    start, end : date-like
        The configured window.
    owner : str
        Who asks, for the error message.

    Raises
    ------
    ValueError
        If an endpoint was never pulled.

    Examples
    --------
    With a raw tier on disk::

        nyse_calendar("/data/downloads/stockfit", "filings", start="2020-01-01", end=None, owner="me")
    """
    names = (endpoints,) if isinstance(endpoints, str) else tuple(endpoints)
    marks = {name: newest_watermark(vendor_root, name) for name in names}
    missing = [name for name, mark in marks.items() if mark is None]
    if missing:
        raise ValueError(
            f"{owner}: StockFit's {', '.join(map(repr, missing))} endpoint(s) were never pulled under "
            f"{str(vendor_root)!r}; refresh the StockFit raw tier first."
        )
    return TradingCalendar.of_nyse(min(marks.values()) - timedelta(days=1), start=start, end=end)
