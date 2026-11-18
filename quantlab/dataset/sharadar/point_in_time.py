"""Sharadar's side of the point-in-time release panels: SEP's trading calendar.

The point-in-time rule, the calendar, the axes and the release and filing
panel base classes are vendor-neutral and live in
``quantlab.dataset._support.point_in_time``. A Sharadar panel (and a Massive
panel on Sharadar's permaticker axis) is placed on SEP's trading days, so it
lines up bar for bar with the Sharadar price panels; this module supplies
that calendar:

- ``sep_calendar`` is SEP's trading days up to the day every raw table a
  panel reads is complete through.
- ``SepReleasePanelDataset`` is a release panel on that calendar: a
  subclass sets ``RAW_TABLES`` (``"sep"`` among them) and supplies its rows
  and spec as any release panel does.

Examples
--------
Needs Sharadar's raw tier, so it is not run here::

    calendar = sep_calendar("/data/quantlab/downloads/sharadar", ("sf1", "sep"), start="2024-01-01", end="2024-12-31")
    calendar.through  # datetime.date(2024, 1, 11)
"""

from __future__ import annotations

from pathlib import Path

from quantlab.dataset._support.point_in_time import ReleasePanelDataset, TradingCalendar
from quantlab.dataset.sharadar.tables import raw_through, trading_days


def sep_calendar(vendor_root: str | Path, tables: tuple[str, ...], *, start, end) -> TradingCalendar:
    """Return SEP's trading days up to the day every one of ``tables`` is complete through.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    tables : tuple of str
        The raw tables the panel reads, ``"sep"`` among them
        (``quantlab.dataset.sharadar.tables.raw_through``).
    start, end : date-like
        The configured window.

    Returns
    -------
    TradingCalendar
        SEP's trading days, ``through`` the day the tables are complete through.

    Examples
    --------
    Needs Sharadar's raw tier, so it is not run here::

        calendar = sep_calendar(root, ("sf1", "sep"), start="2024-01-01", end="2024-12-31")
        calendar.through  # datetime.date(2024, 1, 11)
    """
    through = raw_through(vendor_root, tables)
    return TradingCalendar(trading_days(vendor_root, through), through=through, start=start, end=end)


class SepReleasePanelDataset(ReleasePanelDataset):
    """A release panel placed on SEP's trading days, keyed by permaticker.

    A subclass sets ``config_cls`` (a config with ``raw_data_dir_path``, the
    Sharadar raw tier) and ``RAW_TABLES``, and implements ``_released``,
    ``_spec`` and ``_nothing_shown`` as any ``ReleasePanelDataset`` does. The
    calendar is SEP's trading days through the day every one of
    ``_raw_tables()`` is complete through.

    Examples
    --------
    ``SharadarFundamentalsDataset`` is one::

        class SharadarFundamentalsDataset(SepReleasePanelDataset):
            RAW_TABLES = ("sf1", "sep")
    """

    #: Raw tables that must be complete through a day for it to be built;
    #: ``"sep"`` (the calendar) among them.
    RAW_TABLES: tuple[str, ...] = ("sep",)

    def _raw_tables(self) -> tuple[str, ...]:
        """Return the raw tables bounding the calendar; ``RAW_TABLES`` by default."""
        return self.RAW_TABLES

    def _trading_calendar(self) -> TradingCalendar:
        """Return SEP's trading days through the day ``_raw_tables()`` are complete through (cached)."""
        if self._calendar_cache is None:
            self._calendar_cache = sep_calendar(
                self.config.raw_data_dir_path,
                self._raw_tables(),
                start=self.config.start_date,
                end=self.config.end_date,
            )
        return self._calendar_cache
