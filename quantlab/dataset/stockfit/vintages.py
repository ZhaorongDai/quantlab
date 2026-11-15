"""What a row served without a restatement trail was known as, pull by pull.

StockFit's footnote and as-reported endpoints serve each period's latest
value with the date of the newest filing that reported it, and keep no trail
of the values it replaced. Placed at that date, a value from the first pull of
an entity may already include a later restatement. From the first pull on,
the raw tier keeps every pull that changed (``quantlab.dataset.stockfit.raw``),
so each re-pull can be compared with the one before it, and what changed is
known from the moment it was seen:

- every row of an entity's first pull is a version at its filing date,
  flagged as one that may include a later restatement;
- a row a later pull adds, or whose values it changes, is a version that is
  not flagged; it counts from its filing date when that date is after the
  previous pull's day (the filing that changed it came after the last look),
  and from the pull's time otherwise (the vendor changed it silently);
- a row a pull serves unchanged, or no longer serves, adds nothing.

``observed_versions`` applies these rules to the rows of each pull;
``pull_day`` turns a pull's time into the day its content counts from.

Examples
--------
>>> from datetime import UTC, date, datetime
>>> first = datetime(2026, 1, 5, 14, tzinfo=UTC)
>>> second = datetime(2026, 3, 2, 14, tzinfo=UTC)
>>> pulls = [
...     (first, {"2025-12-31": ServedRow(date(2026, 1, 3), {"value": 1.0})}),
...     (second, {"2025-12-31": ServedRow(date(2026, 1, 3), {"value": 2.0})}),
... ]
>>> [(v.values["value"], v.seen_at is None, v.may_include_restatement) for v in observed_versions(pulls)]
[(1.0, True, True), (2.0, False, False)]
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache

import pandas as pd

#: The time zone the NYSE session close is read in.
NEW_YORK = "America/New_York"


@dataclass(frozen=True)
class ServedRow:
    """One row of one pull: the filing date it is served with and its values.

    Parameters
    ----------
    filed : date
        The date of the newest filing the vendor says reported the row.
    values : Mapping[str, float]
        The row's values, NaN where the vendor serves none.

    Examples
    --------
    >>> from datetime import date
    >>> ServedRow(date(2026, 2, 5), {"value": 1.0}).filed
    datetime.date(2026, 2, 5)
    """

    filed: date
    values: Mapping[str, float]


@dataclass(frozen=True)
class Version:
    """One version of a row: its values and when they became known.

    Parameters
    ----------
    key : str
        The row's key within its entity (a fiscal period end, say).
    filed : date
        The filing date the version was served with.
    values : Mapping[str, float]
        The version's values.
    seen_at : datetime or None
        The time of the pull that first served the version when it counts
        from then; ``None`` when it counts from its filing date.
    may_include_restatement : bool
        Whether the version comes from the entity's first pull, so that its
        values may include a restatement made after its filing date.

    Examples
    --------
    >>> from datetime import date
    >>> Version("2025-12-31", date(2026, 2, 5), {"value": 1.0}, None, True).key
    '2025-12-31'
    """

    key: str
    filed: date
    values: Mapping[str, float]
    seen_at: datetime | None
    may_include_restatement: bool


def _same(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    """Return whether two rows hold the same values, NaN equal to NaN."""
    if a.keys() != b.keys():
        return False
    for name, value in a.items():
        other = b[name]
        if not (value == other or (_isnan(value) and _isnan(other))):
            return False
    return True


def _isnan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def observed_versions(pulls: Sequence[tuple[datetime, Mapping[str, ServedRow]]]) -> list[Version]:
    """Return every version of an entity's rows, from its pulls oldest first.

    Parameters
    ----------
    pulls : sequence of (datetime, Mapping[str, ServedRow])
        Each pull's UTC time and its rows by key, oldest pull first (the
        published pulls of one entity: ``quantlab.dataset.stockfit.raw.pulls``).

    Returns
    -------
    list of Version
        The first pull's rows, then each later pull's new and changed rows,
        in pull order. See the module docstring for the rules.

    Examples
    --------
    A period filed after the first pull is placed at its filing date:

    >>> from datetime import UTC, date, datetime
    >>> pulls = [
    ...     (datetime(2026, 1, 5, tzinfo=UTC), {}),
    ...     (datetime(2026, 3, 2, tzinfo=UTC), {"2025-12-31": ServedRow(date(2026, 2, 5), {"value": 1.0})}),
    ... ]
    >>> observed_versions(pulls)[0].seen_at is None
    True
    """
    found: list[Version] = []
    previous: Mapping[str, ServedRow] | None = None
    previous_day: date | None = None
    for pulled_at, rows in pulls:
        for key, row in rows.items():
            if previous is None:
                found.append(Version(key, row.filed, row.values, None, True))
                continue
            before = previous.get(key)
            if before is not None and _same(before.values, row.values):
                continue
            filed_since = previous_day is not None and row.filed > previous_day
            found.append(Version(key, row.filed, row.values, None if filed_since else pulled_at, False))
        previous, previous_day = rows, pd.Timestamp(pulled_at).tz_convert(NEW_YORK).date()
    return found


def pull_day(pulled_at: datetime) -> date:
    """Return the day what a pull made at ``pulled_at`` saw counts from.

    A pull made before the NYSE session's close (13:00 New York time on a
    half day) counts on its New York day; one made at or after the close, or
    on a day without a session, on the next day, which a trading calendar's
    ``place`` rolls onto the next trading day.

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
    >>> from datetime import UTC, datetime
    >>> pull_day(datetime(2026, 11, 16, 14, tzinfo=UTC))  # 09:00 in New York
    datetime.date(2026, 11, 16)
    >>> pull_day(datetime(2024, 7, 3, 18, tzinfo=UTC))  # 14:00, after the 13:00 close
    datetime.date(2024, 7, 4)
    """
    local = pd.Timestamp(pulled_at).tz_convert(NEW_YORK)
    day = local.date()
    close = _session_close(day)
    return day if close is not None and local.time() < close else day + timedelta(days=1)


@lru_cache(maxsize=1)
def _nyse():
    """Return the XNYS session calendar, built once."""
    from quantlab.dataset._support.session_calendar import XnysSessionCalendar

    return XnysSessionCalendar().calendar


def _session_close(day: date) -> time | None:
    """Return the NYSE close of ``day`` in New York time; 16:00 past the calendar's end; ``None`` without a session."""
    calendar = _nyse()
    label = pd.Timestamp(day)
    if label > calendar.last_session:
        return time(16) if day.weekday() < 5 else None
    if label < calendar.first_session or not calendar.is_session(label):
        return None
    return calendar.session_close(label).tz_convert(NEW_YORK).time()
