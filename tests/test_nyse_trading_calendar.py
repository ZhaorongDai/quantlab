"""The NYSE trading calendar of a release panel that has no Sharadar price table to borrow days from.

A vendor without its own price table (SEC-filing data keyed by CIK) places
its rows on NYSE's sessions instead of SEP's trading days. Every row here is
`# SYNTHETIC`; the holidays are NYSE's own.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from quantlab.dataset._support.point_in_time import TradingCalendar


def _nyse(through=date(2024, 7, 9)) -> TradingCalendar:
    return TradingCalendar.of_nyse(through, start="2024-07-01", end="2024-12-31")


def test_the_nyse_calendar_holds_nyse_sessions_up_to_the_day_the_data_is_complete_through():
    days = _nyse().days_in("2024-07-01", "2024-12-31").dt.date().to_list()

    # Independence Day (Thursday 2024-07-04) and the weekend are not sessions.
    assert days == [
        date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3),
        date(2024, 7, 5), date(2024, 7, 8), date(2024, 7, 9),
    ]


def test_a_row_released_on_an_nyse_holiday_is_available_on_the_next_session():
    rows = pl.DataFrame({"symbol": [101], "date": [date(2024, 7, 4)]})  # SYNTHETIC

    placed = _nyse().place(rows)

    assert placed.get_column("available").dt.date().to_list() == [date(2024, 7, 5)]


def test_a_row_past_the_day_the_data_is_complete_through_is_not_available_yet():
    rows = pl.DataFrame({"symbol": [101], "date": [date(2024, 7, 10)]})  # SYNTHETIC

    assert _nyse(through=date(2024, 7, 9)).place(rows).height == 0


def test_a_row_released_before_the_window_is_still_placed_on_a_session():
    rows = pl.DataFrame({"symbol": [101], "date": [date(2023, 12, 25)]})  # SYNTHETIC, Christmas

    placed = _nyse().place(rows)

    assert placed.get_column("available").dt.date().to_list() == [date(2023, 12, 26)]


def test_a_filing_from_the_early_years_of_edgar_lands_on_its_own_nyse_session():
    calendar = TradingCalendar.of_nyse(date(1997, 1, 10), start="1996-01-01", end="1997-12-31")
    rows = pl.DataFrame({"symbol": [101], "date": [date(1996, 12, 25)]})  # SYNTHETIC, Christmas

    placed = calendar.place(rows, drop_earlier=True)

    assert placed.get_column("available").dt.date().to_list() == [date(1996, 12, 26)]
