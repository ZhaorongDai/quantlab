"""Helpers for inclusive ``(start, end)`` date-range requests on panels.

Datasets answer ``panel(start, end)`` and factors answer ``read(start, end)``
and ``compute(start, end)``. Both ends are inclusive, and a date-only ISO
string such as ``"2024-01-05"`` covers every bar of that day, the way
``xarray`` slices a ``timestamp`` index with it.
"""

import pandas as pd

from quantlab.utils.resample import resample_seconds


def as_label(value):
    """Return ``value`` as a label ``xarray`` slices ``timestamp`` with.

    Strings pass through, so a date-only end keeps covering its whole day;
    anything else becomes a ``pd.Timestamp``.

    Examples
    --------
    >>> as_label("2024-01-05")
    '2024-01-05'
    >>> as_label(datetime.date(2024, 1, 5))
    Timestamp('2024-01-05 00:00:00')
    """
    return value if isinstance(value, str) else pd.Timestamp(value)


def last_moment(value) -> pd.Timestamp:
    """Return the latest instant an inclusive ``end`` label covers.

    A date-only ISO string covers its whole day, as ``xarray`` slices it.

    Examples
    --------
    >>> last_moment("2024-01-05")
    Timestamp('2024-01-05 23:59:59.999999999')
    >>> last_moment("2024-01-05 09:30")
    Timestamp('2024-01-05 09:30:00')
    """
    if isinstance(value, str) and len(value) == len("YYYY-MM-DD"):
        return pd.Timestamp(value) + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
    return pd.Timestamp(value)


def check_range(start, end, owner: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first and last instants of ``[start, end]``.

    Parameters
    ----------
    start, end : str, datetime.date or pd.Timestamp
        The inclusive range.
    owner : str
        The ``Class.method()`` quoted in the error message.

    Raises
    ------
    ValueError
        If ``start`` is after ``end``.

    Examples
    --------
    >>> check_range("2024-01-02", "2024-01-05", "Demo.panel()")
    (Timestamp('2024-01-02 00:00:00'), Timestamp('2024-01-05 23:59:59.999999999'))
    """
    first, last = pd.Timestamp(start), last_moment(end)
    if first > last:
        raise ValueError(f"{owner}: start {start!r} is after end {end!r}.")
    return first, last


def resample_padding(freq: str) -> pd.Timedelta:
    """Return how far outside a range a resampled request reads source bars.

    A resampled bar inside the range draws on source bars up to one
    resample period, plus the rest of the end day, outside it.

    Examples
    --------
    >>> resample_padding("1h")
    Timedelta('1 days 01:00:00')
    """
    return pd.Timedelta(seconds=resample_seconds(freq)) + pd.Timedelta(days=1)


def range_text(value) -> str:
    """Return a range end as it is recorded: strings as given, else ISO.

    Examples
    --------
    >>> range_text("2024-01-05")
    '2024-01-05'
    >>> range_text(datetime.date(2024, 1, 5))
    '2024-01-05T00:00:00'
    """
    return value if isinstance(value, str) else pd.Timestamp(value).isoformat()
