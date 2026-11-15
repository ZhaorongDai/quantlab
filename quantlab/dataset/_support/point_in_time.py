"""Point-in-time release panels, whatever vendor the rows come from.

Many vendor tables are *releases*: each row became public on some day (a
filing on its SEC filing date, a 13F quarter 45 days after its end, an 8-K
on its filing date), and a panel built from them must never show a row
before that day. This module owns that rule and the calendar and axis
plumbing around it, so a dataset supplies only its rows and a spec:

- ``TradingCalendar`` is a vendor's trading days up to the day its raw
  tables are complete through: SEP's (Sharadar's
  ``quantlab.dataset.sharadar.point_in_time.sep_calendar``) or NYSE's
  (``TradingCalendar.of_nyse``). ``place`` gives each row its *available
  date*, the first trading day on or after its date (plus a lag), and drops
  a row the calendar does not reach yet.
- ``as_of`` is the point-in-time rule: on every trading day each security
  shows its latest row whose available date is on or before that day, and
  shows nothing once that row's release date is more than
  ``stale_after_days`` before the day.
- ``shown_symbols``, ``on_grid`` and ``to_panel`` are the axes: the
  securities with a row available by a window's last day, the
  ``(timestamp, symbol)`` cell grid, and the cells shaped into an
  ``xarray.Dataset``. The symbol axis is int64 when every symbol is an
  integer (or a digit string), such as a permaticker or a CIK, and text
  otherwise, such as a Composite FIGI or a fund series
  (``axis_symbols``).
- ``ReleasePanelDataset`` is the dataset of a release table: a subclass
  implements ``_released`` (its rows, each with ``symbol`` and
  ``available``) and ``_spec`` (a ``ReleaseSpec``: the variables, the date
  variables, the column staleness is measured from), and inherits the axes,
  the windows, the symbols an update adds and the widening fill. A
  subclass also supplies its calendar (``_trading_calendar``).
- ``FilingPanelDataset`` is a release panel of events: a filing shows on
  its own day only, every other day holds the subclass's empty value.

Which rows a dataset keeps stays the dataset's own: the fundamentals panel
drops a later filing of an older period, the fiscal-year history turns the
rows into slots, the 13F panel shows one quarter for every security, and a
filing panel shows a filing only on its own day. The last two are not shown
as of their available date: they override the window and reuse the
calendar, the grid and the shaping.

Examples
--------
A row released on Saturday 2024-01-06 is available on Monday and shown from
then on:

>>> import polars as pl
>>> from datetime import date
>>> days = pl.Series("timestamp", ["2024-01-04", "2024-01-05", "2024-01-08"]).str.to_datetime(time_unit="ns")
>>> calendar = TradingCalendar(days, through=date(2024, 1, 8), start="2024-01-01", end="2024-12-31")
>>> rows = calendar.place(pl.DataFrame({"symbol": [101], "date": [date(2024, 1, 6)], "value": [1.0]}))
>>> rows.get_column("available").dt.date().to_list()
[datetime.date(2024, 1, 8)]
>>> spec = ReleaseSpec(variables={"value": {}}, stamp=None)
>>> as_of(rows, calendar.days_in("2024-01-01", "2024-12-31"), [101], spec)["value"].values.ravel().tolist()
[nan, nan, 1.0]
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawDataset
from quantlab.utils.symbol_axis import sort_symbol_axis

#: The first day of ``TradingCalendar.of_nyse``: EDGAR's electronic filings
#: start in 1993, and the ``exchange_calendars`` default starts much later.
NYSE_FIRST_DAY = "1993-01-01"


@lru_cache(maxsize=1)
def _nyse_sessions():
    """Return the ``exchange_calendars`` XNYS calendar from ``NYSE_FIRST_DAY``, built once."""
    import exchange_calendars as xcals

    from quantlab.dataset._support.session_calendar import XnysSessionCalendar

    return xcals.get_calendar(XnysSessionCalendar.EXCHANGE, start=NYSE_FIRST_DAY)


def axis_symbols(symbols: Iterable, text: bool = False) -> list:
    """Return ``symbols`` as the labels of a panel's symbol axis: ints when every one is integral, else text.

    An integer, or a string of digits, is integral (a permaticker, a CIK);
    anything else (a Composite FIGI, a fund series ``<cik>-<series id>``)
    makes the whole axis text. With ``text`` the axis is text whatever the
    labels: a CUSIP can be all digits (``037833100``) and keeps its leading
    zero. Order is kept.

    Examples
    --------
    >>> axis_symbols(["101", 202])
    [101, 202]
    >>> axis_symbols(["BBG000B9XRY4", "0000036405-S000002277"])
    ['BBG000B9XRY4', '0000036405-S000002277']
    >>> axis_symbols(["037833100"], text=True)
    ['037833100']
    """
    labels = list(symbols)
    if text:
        return [str(s) for s in labels]
    integral = all(
        not isinstance(s, bool) and (isinstance(s, (int, np.integer)) or (isinstance(s, str) and s.isdigit()))
        for s in labels
    )
    return [int(s) for s in labels] if integral else [str(s) for s in labels]


def _is_text(symbols: list) -> bool:
    """Return whether an ``axis_symbols`` list is a text axis."""
    return bool(symbols) and isinstance(symbols[0], str)


class TradingCalendar:
    """The trading days a release panel is placed on, up to the day its raw data is complete through.

    Build one from NYSE's sessions with ``of_nyse``, from Sharadar's SEP with
    ``quantlab.dataset.sharadar.point_in_time.sep_calendar``, or from any
    sorted day series with the constructor.

    Parameters
    ----------
    days : pl.Series
        The trading days, sorted, as ns datetimes named ``timestamp``.
    through : date
        The last day the raw tables behind the panel are complete through;
        a row dated after it is not read.
    start, end : date-like
        The configured window. ``days_in`` never returns a day outside it;
        ``place`` uses every day, so a row released before the window is
        still placed.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> days = pl.Series("timestamp", ["2024-01-04", "2024-01-05", "2024-01-08"]).str.to_datetime(time_unit="ns")
    >>> calendar = TradingCalendar(days, through=date(2024, 1, 8), start="2024-01-05", end="2024-12-31")
    >>> calendar.days_in("2024-01-01", "2024-01-31").dt.date().to_list()
    [datetime.date(2024, 1, 5), datetime.date(2024, 1, 8)]
    """

    def __init__(self, days: pl.Series, *, through: date, start, end):
        self.days = days
        self.through = through
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)

    @classmethod
    def of_nyse(cls, through: date, *, start, end) -> TradingCalendar:
        """Return NYSE's sessions from 1993 up to ``through``.

        The sessions are those of the ``exchange_calendars`` XNYS calendar
        behind ``XnysSessionCalendar``
        (``quantlab.dataset._support.session_calendar``), from
        ``NYSE_FIRST_DAY``: weekends and NYSE holidays are not trading days,
        half days are. A row dated before 1993 is placed on the first
        session unless ``place`` drops it (``drop_earlier``).

        Parameters
        ----------
        through : date
            The last day the raw data behind the panel is complete through;
            a row dated after it is not placed.
        start, end : date-like
            The configured window.

        Returns
        -------
        TradingCalendar
            The sessions, as ns datetimes named ``timestamp``.

        Examples
        --------
        >>> from datetime import date
        >>> calendar = TradingCalendar.of_nyse(date(2024, 7, 5), start="2024-07-01", end="2024-12-31")
        >>> calendar.days_in("2024-07-01", "2024-07-31").dt.date().to_list()[-2:]  # no 4 July
        [datetime.date(2024, 7, 3), datetime.date(2024, 7, 5)]
        """
        sessions = _nyse_sessions()
        last = min(pd.Timestamp(through), sessions.last_session)
        labels = sessions.sessions_in_range(sessions.first_session, last)
        days = pl.Series("timestamp", labels.tz_localize(None).to_numpy(), dtype=pl.Datetime("ns"))
        return cls(days, through=through, start=start, end=end)

    def days_in(self, start, end) -> pl.Series:
        """Return the trading days in both ``start`` to ``end`` and the configured window.

        Examples
        --------
        >>> import polars as pl
        >>> from datetime import date
        >>> days = pl.Series("timestamp", ["2024-01-04", "2024-01-08"]).str.to_datetime(time_unit="ns")
        >>> calendar = TradingCalendar(days, through=date(2024, 1, 8), start="2024-01-01", end="2024-12-31")
        >>> calendar.days_in("2024-01-08", "2024-01-08").dt.date().to_list()
        [datetime.date(2024, 1, 8)]
        """
        first = max(pd.Timestamp(start), self.start)
        last = min(pd.Timestamp(end), self.end)
        return self.days.filter(
            self.days.is_between(pl.lit(first.to_datetime64()), pl.lit(last.to_datetime64()))
        )

    def place(
        self,
        rows: pl.DataFrame,
        on: str = "date",
        *,
        lag: timedelta | None = None,
        drop_earlier: bool = False,
    ) -> pl.DataFrame:
        """Return ``rows`` with their available date, the first trading day on or after ``on`` + ``lag``.

        A row the calendar does not reach (dated after its last day) is
        dropped: it is not available yet.

        Parameters
        ----------
        rows : pl.DataFrame
            Rows with a date column ``on``.
        on : str, default ``"date"``
            The column holding the day a row became public (or the day its
            lag starts from).
        lag : timedelta, optional
            Added to ``on`` before placing (13F holdings: 45 days after the
            quarter end).
        drop_earlier : bool, default False
            Drop a row dated before the calendar's first day instead of
            placing it there. A state still holds on the first day; an event
            (a filing) would be misdated.

        Returns
        -------
        pl.DataFrame
            ``rows`` and an ``available`` column (ns datetime, a trading
            day), row order not kept.

        Examples
        --------
        2024-01-02 + 3 days is a Friday; a row past the calendar's last day
        is dropped, and so, on request, is a row before its first:

        >>> import polars as pl
        >>> from datetime import date, timedelta
        >>> days = pl.Series("timestamp", ["2024-01-04", "2024-01-05", "2024-01-08"]).str.to_datetime(time_unit="ns")
        >>> calendar = TradingCalendar(days, through=date(2024, 1, 8), start="2024-01-01", end="2024-12-31")
        >>> rows = pl.DataFrame({"date": [date(2024, 1, 2), date(2024, 1, 9)]})
        >>> calendar.place(rows, lag=timedelta(days=3)).get_column("available").dt.date().to_list()
        [datetime.date(2024, 1, 5)]
        >>> calendar.place(rows).get_column("available").dt.date().to_list()
        [datetime.date(2024, 1, 4)]
        >>> calendar.place(rows, drop_earlier=True).height
        0
        """
        days = self.days
        if drop_earlier and days.len():
            rows = rows.filter(pl.col(on) >= days.min().date())
        due = pl.col(on) if lag is None else pl.col(on) + lag
        calendar = days.to_frame("available").with_columns(
            pl.col("available").cast(pl.Date).alias("_due")
        )
        available = (
            rows.select(pl.col(on).unique())
            .with_columns(due.alias("_due"))
            .sort("_due")
            .join_asof(calendar, on="_due", strategy="forward")
            .filter(pl.col("available").is_not_null())
            .drop("_due")
        )
        return rows.join(available, on=on, how="inner")


@dataclass(frozen=True)
class ReleaseSpec:
    """What a release panel shows: its variables and the column its staleness is measured from.

    Parameters
    ----------
    variables : Mapping[str, Mapping]
        Each variable of the panel and its attributes, in panel order.
    dates : tuple of str, default ()
        The datetime variables among them: a store widened by a new symbol
        fills its stored days with NaT (``ReleasePanelDataset``).
    stamp : str or None, default ``"release_date"``
        The column holding the release date of the row a cell shows: the
        cell is stale once ``stale_after_days`` have passed since it, and a
        cell where it is null shows nothing. ``None`` for a panel without
        staleness.
    stale_after_days : int or None, default None
        Days after its release date a row no newer row has replaced stops
        being shown; ``None`` never.

    Examples
    --------
    >>> ReleaseSpec(variables={"revenue": {"unit": "currency"}}, stale_after_days=365).stamp
    'release_date'
    """

    variables: Mapping[str, Mapping] = field(default_factory=dict)
    dates: tuple[str, ...] = ()
    stamp: str | None = "release_date"
    stale_after_days: int | None = None


def shown_symbols(rows: pl.DataFrame, days: pl.Series) -> list:
    """Return the symbols with a row available by the last of ``days``, sorted.

    With no day, every symbol with a row.

    Examples
    --------
    >>> import polars as pl
    >>> rows = pl.DataFrame({"symbol": [202, 101], "available": ["2024-01-05", "2024-01-09"]}).with_columns(pl.col("available").str.to_datetime(time_unit="ns"))
    >>> days = pl.Series("timestamp", ["2024-01-05", "2024-01-08"]).str.to_datetime(time_unit="ns")
    >>> shown_symbols(rows, days)
    [202]
    """
    if days.len():
        rows = rows.filter(pl.col("available") <= days.max())
    return sort_symbol_axis(rows.get_column("symbol").unique().to_list())


def on_grid(days: pl.Series, symbols: list, *, text: bool = False) -> pl.DataFrame:
    """Return every ``(timestamp, symbol)`` cell, day by day, with its row and column index.

    Returns
    -------
    pl.DataFrame
        ``timestamp``, ``_day``, ``symbol`` (int64, or text for a text axis;
        ``axis_symbols``, whose ``text`` this passes on) and ``_column``, sorted by ``_day`` then
        ``_column``: the order ``to_panel`` reshapes.

    Examples
    --------
    >>> import polars as pl
    >>> days = pl.Series("timestamp", ["2024-01-05", "2024-01-08"]).str.to_datetime(time_unit="ns")
    >>> on_grid(days, [101, 202]).get_column("symbol").to_list()
    [101, 202, 101, 202]
    """
    symbols = axis_symbols(symbols, text)
    dtype = pl.String if text or _is_text(symbols) else pl.Int64
    return (
        days.to_frame()
        .with_row_index("_day")
        .join(
            pl.DataFrame({"symbol": symbols}, schema={"symbol": dtype}).with_row_index("_column"),
            how="cross",
        )
        .sort("_day", "_column")
    )


def to_panel(
    cells: pl.DataFrame,
    days: pl.Series,
    symbols: list,
    variables: Mapping[str, Mapping],
    *,
    text: bool = False,
) -> xr.Dataset:
    """Shape grid-ordered cells into a ``(timestamp, symbol)`` panel.

    Parameters
    ----------
    cells : pl.DataFrame
        One row per cell in ``on_grid`` order, a column per variable.
    days, symbols
        The axes the cells were built on.
    variables : Mapping[str, Mapping]
        Each variable to take and its attributes (copied), in panel order.
    text : bool, default False
        Make the symbol axis text whatever the labels (``axis_symbols``).

    Examples
    --------
    >>> import polars as pl
    >>> days = pl.Series("timestamp", ["2024-01-05", "2024-01-08"]).str.to_datetime(time_unit="ns")
    >>> cells = on_grid(days, [101]).with_columns(value=pl.Series([1.0, 2.0]))
    >>> to_panel(cells, days, [101], {"value": {"unit": "USD"}})["value"].values.tolist()
    [[1.0], [2.0]]
    """
    symbols = axis_symbols(symbols, text)
    shape = (days.len(), len(symbols))
    return xr.Dataset(
        {
            name: (("timestamp", "symbol"), cells.get_column(name).to_numpy().reshape(shape), dict(attrs))
            for name, attrs in variables.items()
        },
        coords={
            "timestamp": pd.DatetimeIndex(days.to_list()),
            "symbol": np.asarray(symbols, dtype=object if text or _is_text(symbols) else np.int64),
        },
    )


def as_of(
    rows: pl.DataFrame, days: pl.Series, symbols: list, spec: ReleaseSpec, *, text: bool = False
) -> xr.Dataset:
    """Return the point-in-time panel: per day and symbol, the latest row available by then.

    A cell shows the symbol's row with the latest ``available`` on or before
    the day; nothing before its first one. When ``spec.stale_after_days`` is
    set, a cell whose row was released (``spec.stamp``) more than that many
    days before the day shows nothing.

    Parameters
    ----------
    rows : pl.DataFrame
        ``symbol``, ``available`` (ns datetime), ``spec.stamp`` when set and
        every variable; at most one row per ``(symbol, available)``, sorted
        by ``available``.
    days : pl.Series
        The panel's trading days.
    symbols : list
        The panel's symbols, in axis order (``axis_symbols``).
    spec : ReleaseSpec
        The variables and the staleness rule.
    text : bool, default False
        Make the symbol axis text whatever the labels (``axis_symbols``).

    Examples
    --------
    A row released on 2024-01-02 and stale after 3 days:

    >>> import polars as pl
    >>> days = pl.Series("timestamp", ["2024-01-02", "2024-01-03", "2024-01-08"]).str.to_datetime(time_unit="ns")
    >>> rows = pl.DataFrame({"symbol": [101], "available": days[:1], "release_date": days[:1], "value": [5.0]})
    >>> spec = ReleaseSpec(variables={"value": {}}, stale_after_days=3)
    >>> as_of(rows, days, [101], spec)["value"].values.ravel().tolist()
    [5.0, 5.0, nan]
    """
    symbols = axis_symbols(symbols, text)
    cells = (
        on_grid(days, symbols, text=text)
        .join_asof(
            rows.filter(pl.col("symbol").is_in(symbols)),
            left_on="timestamp",
            right_on="available",
            by="symbol",
            strategy="backward",
            # Both sides are sorted by time; polars cannot check that per group.
            check_sortedness=False,
        )
        .sort("_day", "_column")
    )
    if spec.stale_after_days is not None:
        limit = timedelta(days=spec.stale_after_days)
        fresh = (pl.col("timestamp") - pl.col(spec.stamp)) <= pl.lit(limit)
        cells = cells.with_columns(
            pl.when(fresh).then(pl.col(name)).otherwise(None).alias(name)
            for name in spec.variables
        )
    return to_panel(cells, days, symbols, spec.variables, text=text)


class ReleasePanelDataset(RawDataset):
    """Daily point-in-time panel of one release table, keyed by an int64 or a text symbol.

    A subclass sets ``config_cls`` and implements ``_trading_calendar`` (its
    ``TradingCalendar``, cached in ``_calendar_cache``), ``_released`` (its
    rows, each at its available date, usually through
    ``_trading_calendar().place``), ``_spec`` (a ``ReleaseSpec``) and
    ``_nothing_shown`` (the start of the message refusing an empty window).
    It inherits the axes of a window (the trading days, and the securities
    with a row available by its end), the window itself (``as_of``), the
    symbols an update adds (those with a shown cell in the new days), the
    widening fill (NaT for ``spec.dates``) and no OHLCV cleaning. The
    staleness limit is the config's ``stale_after_days`` when it has one.

    A panel whose rows are not shown as of their available date (13F
    holdings, filings) overrides ``_raw_data_to_xr_window`` and
    ``_added_symbols_with_raw_history`` and keeps the rest. A panel whose
    symbols are text even when all digits (a CUSIP) sets ``TEXT_SYMBOLS``.

    Examples
    --------
    ``SharadarFundamentalsDataset`` is one, on SEP's calendar
    (``quantlab.dataset.sharadar.point_in_time.SepReleasePanelDataset``)::

        SharadarFundamentalsDataset(config).panel("2024-01-02", "2024-01-05")["revenue"]
    """

    #: Whether the symbol axis is text whatever the labels (``axis_symbols``).
    TEXT_SYMBOLS: bool = False

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        self._calendar_cache: TradingCalendar | None = None
        self._released_cache: pl.DataFrame | None = None

    # -- what a subclass provides -------------------------------------------

    def _trading_calendar(self) -> TradingCalendar:
        """Return the calendar the rows are placed on, built once per config (in ``_calendar_cache``)."""
        raise NotImplementedError

    def _released(self) -> pl.DataFrame:
        """Return the rows the panel shows, each at its available date.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (int64 or text), ``available`` (a trading day, ns
            datetime) and one column per variable of ``_spec()``, sorted by
            ``available`` when the inherited window reads it.
        """
        raise NotImplementedError

    def _spec(self) -> ReleaseSpec:
        """Return the panel's variables and staleness rule."""
        raise NotImplementedError

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks, e.g. ``"no ARQ row is shown"``."""
        raise NotImplementedError

    # -- the shared part ----------------------------------------------------

    def _released_rows(self) -> pl.DataFrame:
        """Return ``_released()``, built once per config (cached)."""
        if self._released_cache is None:
            self._released_cache = self._released()
        return self._released_cache

    def _release_spec(self) -> ReleaseSpec:
        """Return ``_spec()`` with the config's ``stale_after_days``, if it has one."""
        spec = self._spec()
        return dataclasses.replace(
            spec, stale_after_days=getattr(self.config, "stale_after_days", spec.stale_after_days)
        )

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the symbols with a row available by the window's end, and its trading days.

        Raises
        ------
        ValueError
            If the window holds no trading day or no row is available by its end.
        """
        days = self._trading_calendar().days_in(self.config.start_date, self.config.end_date)
        symbols = shown_symbols(self._released_rows(), days)
        if not days.len() or not symbols:
            raise ValueError(
                f"{self.class_name}: {self._nothing_shown()} in "
                f"[{self.config.start_date}, {self.config.end_date}] for the "
                f"configured universe."
            )
        return symbols, pd.DatetimeIndex(days.to_list())

    def _window_axes(self, start_date, end_date, symbols) -> tuple[pl.Series, list]:
        """Return a window's trading days and its symbols, those shown by its end when ``None``."""
        days = self._trading_calendar().days_in(start_date, end_date)
        if symbols is None:
            symbols = shown_symbols(self._released_rows(), days)
        return days, axis_symbols(symbols, self.TEXT_SYMBOLS)

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list | None = None
    ) -> xr.Dataset:
        """Return one window of the panel: every trading day by every symbol, as of the day.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list, optional
            Symbols to build; ``None`` is every one with a row available by
            the window's end.
        """
        days, symbols = self._window_axes(start_date, end_date, symbols)
        return as_of(self._released_rows(), days, symbols, self._release_spec(), text=self.TEXT_SYMBOLS)

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the panel for the whole configured window."""
        symbols, _ = self._raw_axes_in_range()
        return self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=symbols
        )

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count the cells each added symbol shows between ``start`` and ``end``.

        A symbol whose first row is available after the store's last
        day shows nothing in it, so it is a new one and widens the store. A
        cell counts when any variable shows a value: the staleness column
        (``spec.stamp``) need not be one of the panel's variables.
        """
        if not added:
            return {}
        window = self._raw_data_to_xr_window(start, end, symbols=axis_symbols(added, self.TEXT_SYMBOLS))
        counts = window.notnull().to_dataarray().any("variable").sum("timestamp")
        return {
            str(symbol): int(count)
            for symbol, count in zip(window["symbol"].values.tolist(), counts.values.tolist())
            if count
        }

    def _widen_fill_values(self) -> dict:
        """Fill a new symbol's stored days of the date variables with NaT."""
        return {name: np.datetime64("NaT", "ns") for name in self._spec().dates}

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel unchanged: a release table is not OHLCV market data."""
        return data


class FilingPanelDataset(ReleasePanelDataset):
    """Daily panel of filings, each shown on its available date only.

    Some tables are events, not states: an 8-K or an insider's Form 4 is
    filed on one day and says something about that day only. A panel of
    them has a value on the day a filing became public and nothing in
    between: a day with no filing holds the subclass's empty value
    (``False``, ``0.0``), never NaN, since nothing was filed.

    A subclass implements what a release panel does (``_trading_calendar``,
    ``_released`` with one row per symbol and available day,
    ``_nothing_shown``) plus ``_empty`` (each variable's value on a day
    without a filing, in panel order) and optionally ``_attrs``. The spec,
    the windows, the symbols an update adds and the widening fill are
    inherited.

    Examples
    --------
    ``SharadarEventsDataset`` is one, the 8-K items of Sharadar's EVENTS::

        SharadarEventsDataset(config).panel("2024-01-02", "2024-01-05")["event_22"]
    """

    # -- the subclass's part --------------------------------------------------

    def _empty(self) -> dict[str, object]:
        """Return each variable's value on a day without a filing, in variable order."""
        raise NotImplementedError

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's attributes; none by default."""
        return {}

    # -- the release table ------------------------------------------------------

    def _spec(self) -> ReleaseSpec:
        """Return the variables, in ``_empty()`` order, with their attributes."""
        attrs = self._attrs()
        return ReleaseSpec(
            variables={name: attrs.get(name, {}) for name in self._empty()}, stamp=None
        )

    # -- windows: a filing on its own day only ----------------------------------

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list | None = None
    ) -> xr.Dataset:
        """Return one window of the panel: a day's filings on that day, the empty value elsewhere.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list, optional
            Symbols to build; ``None`` is every one with a filing by the
            window's end.
        """
        days, symbols = self._window_axes(start_date, end_date, symbols)
        cells = (
            on_grid(days, symbols, text=self.TEXT_SYMBOLS)
            .join(
                self._released_rows(),
                left_on=["timestamp", "symbol"],
                right_on=["available", "symbol"],
                how="left",
            )
            .sort("_day", "_column")
            .with_columns(pl.col(name).fill_null(value) for name, value in self._empty().items())
        )
        return to_panel(cells, days, symbols, self._spec().variables, text=self.TEXT_SYMBOLS)

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count each added symbol's filing days between ``start`` and ``end``.

        A symbol whose first filing is after the store's last day has none
        in it, so it widens the store.
        """
        if not added:
            return {}
        counts = (
            self._released_rows()
            .filter(
                pl.col("symbol").is_in(axis_symbols(added, self.TEXT_SYMBOLS))
                & pl.col("available").is_between(
                    pl.lit(pd.Timestamp(start).to_datetime64()),
                    pl.lit(pd.Timestamp(end).to_datetime64()),
                )
            )
            .group_by("symbol")
            .len()
        )
        return {str(symbol): int(rows) for symbol, rows in counts.iter_rows()}

    def _widen_fill_values(self) -> dict:
        """Fill a new symbol's stored days with the empty values: it filed nothing then."""
        return self._empty()
