"""XNYS session calendar for intraday panels (phase 03.9, D-09 / D-23 / D-29).

Why a calendar at all: raw NBBO is collected 04:00-20:00 ET, and NYSE closes
early (13:00 ET) on a handful of days a year. A panel built on a fixed
09:30-16:00 window would, on such a half day, treat three hours of
after-hours quotes as regular-session state -- a plausible-looking but wrong
panel. `exchange_calendars` (the maintained fork of Quantopian's
trading_calendars) is the source of every session's actual open/close,
historical half days included.

This module depends on the calendar library only; it has no dependency on
the acquisition layer.
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
    if not isinstance(value, str) or not _TIME_PATTERN.match(value):
        raise ValueError(f"{name} must be 'HH:MM' or 'HH:MM:SS', got {value!r}")
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid time of day: {value!r}") from exc


class XnysSessionCalendar:
    """Per-session (open, close) bounds of a configurable XNYS window.

    The window (`session_start`/`session_end`, ET wall clock) defaults to
    regular trading hours 09:30-16:00 (D-09) and may be set anywhere inside
    the SIP publication window 04:00-20:00 ET (D-29). An edge outside that
    range, a malformed edge, or `session_start >= session_end` is refused at
    construction with `ValueError`.

    Edge rule (D-29), applied to each edge independently:

    - an edge inside regular hours [09:30, 16:00] is clipped to that
      session's actual exchange open/close: open = max(edge, exchange open),
      close = min(edge, exchange close). This is where a half day caps a
      regular-hours window;
    - an edge in extended hours (before 09:30 or after 16:00) is wall clock
      and unchanged, even on a half day.

    On 2024-11-29 (13:00 ET early close) this gives:

    - 09:30-16:00 -> 09:30-13:00 ET (14:30Z-18:00Z);
    - 10:00-15:30 -> 10:00-13:00 ET (15:30 is clipped to the early close);
    - 13:30-16:00 -> empty window, so the date yields no row (logged at INFO);
    - 04:00-20:00 -> 04:00-20:00 ET unchanged (09:00Z-01:00Z next day);
    - 09:30-17:00 -> 09:30-17:00 ET (17:00 is an after-hours edge);
    - 07:00-16:00 -> 07:00-13:00 ET (16:00 is a regular-hours edge).

    A window spanning the early close is continuous: it is not split into a
    regular and an after-hours part, so bars after 13:00 on such a day carry
    post-close quote state. Choose a regular-hours end edge if that is not
    wanted.

    Timestamps are naive UTC, like every other timestamp in the codebase.
    DST is handled by localizing each edge on its own date in
    `America/New_York` before converting to UTC, so the same ET window maps
    to different UTC instants in winter and summer. An extended close
    (e.g. 20:00 ET) lands on the next UTC calendar day; that is expected.
    """

    EXCHANGE = "XNYS"
    TIME_ZONE = "America/New_York"
    REGULAR_OPEN = "09:30"
    REGULAR_CLOSE = "16:00"
    EXTENDED_OPEN = "04:00"
    EXTENDED_CLOSE = "20:00"
    # TAQ millisecond history begins 2003-09-10; exchange_calendars' default
    # lookback does not reach that far, so the start is explicit.
    CALENDAR_START = "2003-01-01"

    def __init__(self, session_start: str = "09:30", session_end: str = "16:00"):
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
        if self._calendar is None:
            self._calendar = xcals.get_calendar(
                self.EXCHANGE, start=self.CALENDAR_START
            )
        return self._calendar

    def _label(self, day: date) -> pd.Timestamp:
        cal = self.calendar
        label = pd.Timestamp(day)
        if label < cal.first_session or label > cal.last_session:
            raise ValueError(
                f"{day.isoformat()} is outside the {self.EXCHANGE} calendar range "
                f"{cal.first_session.date()}..{cal.last_session.date()}"
            )
        return label

    def is_session(self, day: date) -> bool:
        return bool(self.calendar.is_session(self._label(day)))

    def _is_regular(self, edge: time) -> bool:
        return (
            time.fromisoformat(self.REGULAR_OPEN)
            <= edge
            <= time.fromisoformat(self.REGULAR_CLOSE)
        )

    def _wall_clock_utc(self, day: date, edge: time) -> datetime:
        return (
            pd.Timestamp(datetime.combine(day, edge))
            .tz_localize(self.TIME_ZONE)
            .tz_convert("UTC")
            .tz_localize(None)
            .to_pydatetime()
        )

    def session_bounds(self, dates: Iterable[date]) -> pl.DataFrame:
        """Return `date` (Date), `open`, `close` (naive-UTC Datetime("ns")).

        Rows are sorted by date. A non-session date raises `ValueError`
        naming it; a session whose clipped window is empty is omitted.
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
