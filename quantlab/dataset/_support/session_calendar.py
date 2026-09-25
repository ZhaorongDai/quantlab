"""Exchange session calendar for intraday panels.

A *session* is one trading day of an exchange, with its open and close time.
The NBBO (National Best Bid and Offer, the best bid and ask across all US
exchanges) is published by the SIP (the securities information processor)
from 04:00 to 20:00 ET (US Eastern time), and the raw quotes cover that whole
range. The New York Stock Exchange (exchange code XNYS) also closes early,
at 13:00 ET, on a few days a year. A panel built on a fixed 09:30 to 16:00
window would, on such a day, treat three hours of after-close quotes as
regular trading.

``XnysSessionCalendar`` uses the ``exchange_calendars`` library to get every
session's actual open and close, half days included, and turns a
configurable ET window into UTC bounds for each session. The module depends
only on that library.
"""

import re
from datetime import date, datetime, time
from typing import Iterable, Optional

import exchange_calendars as xcals
import pandas as pd
import polars as pl
from loguru import logger

_TIME_PATTERN = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")


def _parse_time(value: str, name: str) -> time:
    """Parse an ``HH:MM`` or ``HH:MM:SS`` string into a ``time``.

    Parameters
    ----------
    value : str
        The text to parse.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    time
        The parsed time of day.

    Raises
    ------
    ValueError
        If ``value`` is not a string of that shape or is not a
        valid time of day; ``name`` is used in the message.
    """
    if not isinstance(value, str) or not _TIME_PATTERN.match(value):
        raise ValueError(f"{name} must be 'HH:MM' or 'HH:MM:SS', got {value!r}")
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid time of day: {value!r}") from exc


class XnysSessionCalendar:
    """Per-session open and close bounds of a configurable NYSE (XNYS) window.

    The window, given by ``session_start`` and ``session_end`` as ET clock
    times, defaults to regular trading hours, 09:30 to 16:00. It may be set
    anywhere inside the SIP publication window of 04:00 to 20:00 ET.

    Each edge is handled on its own. An edge inside regular hours (09:30 to
    16:00 inclusive) is clipped to the session's actual exchange open or
    close; this is how a half day shortens a regular-hours window. An edge
    in extended hours (before 09:30 or after 16:00) is a plain clock time
    and stays unchanged even on a half day. On 2024-11-29 (13:00 ET early
    close) this gives:

    - 09:30 to 16:00 becomes 09:30 to 13:00 ET;
    - 13:30 to 16:00 becomes empty, so the date yields no row (logged at info);
    - 04:00 to 20:00 stays 04:00 to 20:00 ET, the close landing on the next
      UTC calendar day;
    - 07:00 to 16:00 becomes 07:00 to 13:00 ET, since only 16:00 is a
      regular-hours edge.

    A window that spans the early close has no gap, so bars after 13:00 on
    such a day hold after-close quotes; use a regular-hours end edge if that
    is not wanted. Returned timestamps are naive UTC. Each edge is converted
    from ``America/New_York`` on its own date, so the same ET window maps to
    different UTC times in winter and summer (daylight saving time).

    Parameters
    ----------
    session_start : str, default "09:30"
        Window start as ``"HH:MM"`` or ``"HH:MM:SS"`` ET.
    session_end : str, default "16:00"
        Window end in the same format.

    Attributes
    ----------
    session_start, session_end : datetime.time
        The parsed window edges.

    Raises
    ------
    ValueError
        If an edge is malformed, lies outside 04:00 to 20:00 ET, or
        ``session_start`` is not before ``session_end``.

    Examples
    --------
    >>> cal = XnysSessionCalendar("09:30", "16:00")
    >>> cal.is_session(date(2024, 11, 29))
    True
    >>> bounds = cal.session_bounds([date(2024, 11, 29)])
    >>> bounds["close"][0]  # the 13:00 ET early close, as naive UTC
    datetime.datetime(2024, 11, 29, 18, 0)
    """

    EXCHANGE = "XNYS"
    TIME_ZONE = "America/New_York"
    REGULAR_OPEN = "09:30"
    REGULAR_CLOSE = "16:00"
    EXTENDED_OPEN = "04:00"
    EXTENDED_CLOSE = "20:00"
    # TAQ (NYSE's Trade and Quote database) has millisecond data from 2003.
    # The calendar library's default history does not go back that far.
    CALENDAR_START = "2003-01-01"

    def __init__(self, session_start: str = "09:30", session_end: str = "16:00"):
        """Initialize the calendar; see the class docstring for parameters."""
        start = _parse_time(session_start, "session_start")
        end = _parse_time(session_end, "session_end")
        lo = time.fromisoformat(self.EXTENDED_OPEN)
        hi = time.fromisoformat(self.EXTENDED_CLOSE)
        for name, edge in (("session_start", start), ("session_end", end)):
            if not lo <= edge <= hi:
                raise ValueError(
                    f"{name} {edge.isoformat()} lies outside the extended window "
                    f"{self.EXTENDED_OPEN}-{self.EXTENDED_CLOSE} ET"
                )
        if start >= end:
            raise ValueError(
                f"session_start {start.isoformat()} must be before "
                f"session_end {end.isoformat()}"
            )
        self.session_start = start
        self.session_end = end
        self._calendar: Optional[xcals.ExchangeCalendar] = None

    @property
    def calendar(self) -> xcals.ExchangeCalendar:
        """Return the ``exchange_calendars`` XNYS calendar, building it on first use.

        Examples
        --------
        >>> cal.calendar.name, cal.calendar.first_session
        ('XNYS', Timestamp('2003-01-02 00:00:00'))
        """
        if self._calendar is None:
            self._calendar = xcals.get_calendar(
                self.EXCHANGE, start=self.CALENDAR_START
            )
        return self._calendar

    def _label(self, day: date) -> pd.Timestamp:
        """Return ``day`` as the calendar's session label (a ``pd.Timestamp``).

        Raises
        ------
        ValueError
            If ``day`` lies outside the calendar's date range.
        """
        cal = self.calendar
        label = pd.Timestamp(day)
        if label < cal.first_session or label > cal.last_session:
            raise ValueError(
                f"{day.isoformat()} is outside the {self.EXCHANGE} calendar range "
                f"{cal.first_session.date()}..{cal.last_session.date()}"
            )
        return label

    def is_session(self, day: date) -> bool:
        """Return whether ``day`` is an XNYS trading session.

        Parameters
        ----------
        day : date
            Calendar date to check.

        Returns
        -------
        bool
            True if the exchange trades on ``day``.

        Examples
        --------
        >>> cal.is_session(date(2024, 11, 28))  # Thanksgiving
        False
        >>> cal.is_session(date(2024, 11, 29))  # the half day after it
        True
        """
        return bool(self.calendar.is_session(self._label(day)))

    def _is_regular(self, edge: time) -> bool:
        """Return whether ``edge`` lies within regular trading hours, 09:30 to 16:00."""
        return (
            time.fromisoformat(self.REGULAR_OPEN)
            <= edge
            <= time.fromisoformat(self.REGULAR_CLOSE)
        )

    def _wall_clock_utc(self, day: date, edge: time) -> datetime:
        """Convert the ET clock time ``edge`` on ``day`` to a naive UTC datetime."""
        return (
            pd.Timestamp(datetime.combine(day, edge))
            .tz_localize(self.TIME_ZONE)
            .tz_convert("UTC")
            .tz_localize(None)
            .to_pydatetime()
        )

    def session_bounds(self, dates: Iterable[date]) -> pl.DataFrame:
        """Return the UTC open and close of the window for each session date.

        Parameters
        ----------
        dates : Iterable[date]
            Session dates; duplicates are collapsed and the result is
            sorted by date.

        Returns
        -------
        pl.DataFrame
            A frame with columns ``date`` (Date), ``open`` and ``close``
            (naive UTC ``Datetime("ns")``). A session whose window is empty
            after clipping is left out and logged at info level.

        Raises
        ------
        ValueError
            If a date is not an XNYS session or lies outside the
            calendar's range.

        Examples
        --------
        >>> bounds = cal.session_bounds([date(2024, 11, 27), date(2024, 11, 29)])
        >>> bounds["close"].dt.hour().to_list()  # 16:00 ET, then the 13:00 close
        [21, 18]
        >>> XnysSessionCalendar("13:30", "16:00").session_bounds(
        ...     [date(2024, 11, 29)]
        ... ).height  # the window is empty after clipping, so no row
        0
        """
        cal = self.calendar
        rows = []
        for day in sorted(set(dates)):
            label = self._label(day)
            if not cal.is_session(label):
                raise ValueError(
                    f"{day.isoformat()} is not an {self.EXCHANGE} session"
                )
            exchange_open = cal.session_open(label).tz_localize(None).to_pydatetime()
            exchange_close = cal.session_close(label).tz_localize(None).to_pydatetime()

            open_ = self._wall_clock_utc(day, self.session_start)
            if self._is_regular(self.session_start):
                open_ = max(open_, exchange_open)
            close = self._wall_clock_utc(day, self.session_end)
            if self._is_regular(self.session_end):
                close = min(close, exchange_close)

            if open_ >= close:
                logger.info(
                    f"{day.isoformat()}: window {self.session_start.isoformat()}-"
                    f"{self.session_end.isoformat()} ET is empty after clipping to the "
                    f"exchange session ({exchange_open}..{exchange_close} UTC); omitted"
                )
                continue
            rows.append({"date": day, "open": open_, "close": close})

        return pl.DataFrame(
            rows,
            schema={
                "date": pl.Date,
                "open": pl.Datetime("ns"),
                "close": pl.Datetime("ns"),
            },
        )
