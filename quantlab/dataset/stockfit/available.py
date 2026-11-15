"""When a StockFit row becomes available: the one rule every StockFit panel places its rows by.

A row that carries the SEC's acceptance time (``acceptedAt``) is available on
the NYSE trading day it was accepted on when accepted before that session's
close (16:00 New York time, 13:00 on an early close), and on the next trading
day when accepted at or after the close or on a day without a session. So a
filing made after the close never reaches the day's bar. A row with only a
filing date (``dateFiled``) is available on that day, or on the next trading
day when it is not one, as Sharadar's filings are.

``release_day`` gives each row the calendar day the rule picks, and
``place_released`` rolls it onto the trading calendar (``TradingCalendar.place``).
``nyse_calendar`` is the calendar every StockFit panel uses: NYSE's sessions
through the day the endpoint's raw tier is complete through
(``raw_through``).

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

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import polars as pl

from quantlab.dataset._support.point_in_time import NYSE_FIRST_DAY, TradingCalendar
from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.stockfit.raw import endpoint_dir, read_watermark

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
    closes = sessions.session_bounds(inside).select(
        pl.col("date").alias("_local_day"), pl.col("close").dt.replace_time_zone("UTC").alias("_close")
    )
    return closes


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
    rows: pl.DataFrame, calendar: TradingCalendar, accepted: str = "accepted_at", filed: str = "date_filed"
) -> pl.DataFrame:
    """Return ``rows`` with ``available``: the first trading day on or after each row's ``release_day``.

    A row with neither time, dated before the calendar's first day, or past
    the day the calendar is complete through is dropped.

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
    return calendar.place(dated, on="release_day", drop_earlier=True)


def raw_through(vendor_root: str | Path, endpoint: str) -> date | None:
    """Return the last day an endpoint's raw tier is complete through; ``None`` if never pulled.

    That is the day before its newest watermark (the UTC day of the latest
    pull of any of its entities): a refresh made on a day holds every row
    public by the close of the day before, whatever time it ran. The daily
    refresh re-pulls only the entities with new filings, so the newest
    watermark, not the oldest, dates the endpoint.

    Examples
    --------
    >>> raw_through("/nonexistent/stockfit", "filings") is None
    True
    """
    root = endpoint_dir(vendor_root, endpoint)
    if not root.exists():
        return None
    marks = [
        mark
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
        if (mark := read_watermark(vendor_root, endpoint, path.name)) is not None
    ]
    return max(marks) - timedelta(days=1) if marks else None


def nyse_calendar(vendor_root: str | Path, endpoint: str, *, start, end, owner: str) -> TradingCalendar:
    """Return NYSE's trading days through the day ``endpoint``'s raw tier is complete through.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/stockfit``.
    endpoint : str
        The raw tier's endpoint the panel reads.
    start, end : date-like
        The configured window.
    owner : str
        Who asks, for the error message.

    Raises
    ------
    ValueError
        If the endpoint was never pulled.

    Examples
    --------
    With a raw tier on disk::

        nyse_calendar("/data/quantlab/downloads/stockfit", "filings", start="2020-01-01", end=None, owner="me")
    """
    through = raw_through(vendor_root, endpoint)
    if through is None:
        raise ValueError(
            f"{owner}: StockFit's {endpoint!r} endpoint was never pulled under {str(vendor_root)!r}; "
            f"refresh the StockFit raw tier first."
        )
    return TradingCalendar.of_nyse(through, start=start, end=end)
