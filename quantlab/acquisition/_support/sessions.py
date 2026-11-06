"""The XNYS sessions of a date range, for the acquisition code that walks trading days.

Examples
--------
>>> from datetime import date
>>> xnys_sessions(date(2024, 3, 8), date(2024, 3, 11))
[datetime.date(2024, 3, 8), datetime.date(2024, 3, 11)]
"""

from __future__ import annotations

from datetime import date

import exchange_calendars as xcals


def xnys_sessions(start: date, end: date) -> list[date]:
    """Return the XNYS sessions from ``start`` through ``end``, ascending.

    Parameters
    ----------
    start, end : date
        The first and last day, inclusive.

    Returns
    -------
    list of date
        The sessions; empty when ``end`` is before ``start``.

    Examples
    --------
    >>> xnys_sessions(date(2024, 3, 11), date(2024, 3, 8))
    []
    """
    if end < start:
        return []
    sessions = xcals.get_calendar("XNYS").sessions_in_range(start.isoformat(), end.isoformat())
    return [stamp.date() for stamp in sessions]
