"""The available date of a StockFit row: the one rule every StockFit panel places its rows by.

StockFit dates a filing by its EDGAR acceptance time (``acceptedAt``, UTC to
the second) and its filing date (``dateFiled``). A row is *available* on:

- with ``acceptedAt``: its New York day when it was accepted before that
  day's NYSE close (16:00, or 13:00 on a half day), otherwise the next NYSE
  trading day; a row accepted on a day without a session rolls to the next
  trading day;
- with only ``dateFiled``: that day, or the next trading day when it is not
  one (the Sharadar convention: a filing counts on its filing date).

So a close-based backtest never trades on a filing made after the close.
``available_dates`` applies the rule to a frame of rows, on any
``TradingCalendar`` (StockFit's panels use NYSE's, ``TradingCalendar.of_nyse``);
it is the only place the rule is written.

Examples
--------
2026-11-09 was a Monday; 21:00 UTC is 16:00 in New York that day:

>>> from datetime import date, datetime, UTC
>>> import polars as pl
>>> from quantlab.dataset._support.point_in_time import TradingCalendar
>>> calendar = TradingCalendar.of_nyse(date(2026, 11, 13), start="2026-11-01", end="2026-11-13")
>>> rows = pl.DataFrame({
...     "accepted_at": [datetime(2026, 11, 9, 20, 59, tzinfo=UTC), datetime(2026, 11, 9, 21, 0, tzinfo=UTC), None],
...     "date_filed": [date(2026, 11, 9), date(2026, 11, 9), date(2026, 11, 8)],
... })
>>> available_dates(rows.with_row_index(), calendar).sort("index").get_column("available").dt.date().to_list()
[datetime.date(2026, 11, 9), datetime.date(2026, 11, 10), datetime.date(2026, 11, 9)]
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl

from quantlab.dataset._support.point_in_time import NYSE_FIRST_DAY, TradingCalendar

#: The exchange whose session closes decide the day of an acceptance time.
_EXCHANGE = "XNYS"

#: New York's time zone, where a session's day is counted.
NEW_YORK = "America/New_York"


@lru_cache(maxsize=1)
def _session_closes() -> pl.DataFrame:
    """Return every NYSE session from 1993 with its actual close: ``_local_day``, ``_close`` (UTC)."""
    import exchange_calendars as xcals

    closes = xcals.get_calendar(_EXCHANGE, start=NYSE_FIRST_DAY).closes
    return pl.DataFrame(
        {
            "_local_day": list(closes.index.date),
            "_close": closes.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(),
        },
        schema={"_local_day": pl.Date, "_close": pl.Datetime("ns")},
    ).with_columns(pl.col("_close").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))


def available_dates(
    rows: pl.DataFrame,
    calendar: TradingCalendar,
    *,
    accepted: str = "accepted_at",
    filed: str = "date_filed",
) -> pl.DataFrame:
    """Return ``rows`` with ``available``, the trading day each became public by StockFit's rule.

    Parameters
    ----------
    rows : pl.DataFrame
        Rows with an acceptance time column ``accepted`` (UTC datetime, may
        be null) and a filing date column ``filed`` (date, may be null).
    calendar : TradingCalendar
        The trading days the rows are placed on.
    accepted, filed : str
        The two columns' names.

    Returns
    -------
    pl.DataFrame
        ``rows`` and ``available`` (ns datetime, a trading day). A row with
        neither date, or one the calendar does not reach yet, is dropped;
        row order is not kept.

    Examples
    --------
    See the module docstring.
    """
    at = pl.col(accepted).cast(pl.Datetime("us", "UTC"))
    local_day = at.dt.convert_time_zone(NEW_YORK).dt.date()
    dated = rows.with_columns(local_day.alias("_local_day")).join(_session_closes(), on="_local_day", how="left")
    before_close = (at < pl.col("_close")).fill_null(False)
    day = (
        pl.when(pl.col(accepted).is_null())
        .then(pl.col(filed))
        .when(before_close)
        .then(pl.col("_local_day"))
        .otherwise(pl.col("_local_day") + pl.duration(days=1))
    )
    dated = dated.with_columns(day.alias("_day")).drop("_local_day", "_close").filter(pl.col("_day").is_not_null())
    return calendar.place(dated, "_day").drop("_day")
