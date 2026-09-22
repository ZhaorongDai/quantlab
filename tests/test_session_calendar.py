"""Locks for `XnysSessionCalendar` (phase 03.9, D-09 / D-23 / D-29).

Every expected value is a known 2024 (or 2016) XNYS fact expressed as naive
UTC: winter sessions open 14:30Z, the first EDT session (2024-03-11) opens
13:30Z, the day after Thanksgiving 2024 closes 13:00 ET = 18:00Z, and
Thanksgiving itself is not a session.
"""

import tomllib
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from quantlab.dataset._support.session_calendar import XnysSessionCalendar

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bounds(cal: XnysSessionCalendar, day: date) -> tuple[datetime, datetime]:
    frame = cal.session_bounds([day])
    assert frame.height == 1, frame
    row = frame.row(0, named=True)
    assert row["date"] == day
    return row["open"], row["close"]


# --- regular hours (default window) -----------------------------------------


def test_winter_session_default_window():
    assert _bounds(XnysSessionCalendar(), date(2024, 1, 24)) == (
        datetime(2024, 1, 24, 14, 30),
        datetime(2024, 1, 24, 21, 0),
    )


def test_dst_first_edt_session():
    assert _bounds(XnysSessionCalendar(), date(2024, 3, 11)) == (
        datetime(2024, 3, 11, 13, 30),
        datetime(2024, 3, 11, 20, 0),
    )


def test_half_day_early_close():
    assert _bounds(XnysSessionCalendar(), date(2024, 11, 29)) == (
        datetime(2024, 11, 29, 14, 30),
        datetime(2024, 11, 29, 18, 0),
    )


def test_winter_2016_session():
    assert _bounds(XnysSessionCalendar(), date(2016, 12, 7)) == (
        datetime(2016, 12, 7, 14, 30),
        datetime(2016, 12, 7, 21, 0),
    )


def test_non_session_refused():
    with pytest.raises(ValueError, match="2024-11-28"):
        XnysSessionCalendar().session_bounds([date(2024, 11, 28)])


def test_is_session():
    cal = XnysSessionCalendar()
    assert cal.is_session(date(2024, 1, 24)) is True
    assert cal.is_session(date(2024, 1, 27)) is False


def test_schema_and_sorting():
    frame = XnysSessionCalendar().session_bounds(
        [date(2024, 11, 29), date(2024, 1, 24), date(2024, 3, 11)]
    )
    assert frame.columns == ["date", "open", "close"]
    assert frame.schema["date"] == pl.Date
    assert frame.schema["open"] == pl.Datetime("ns")
    assert frame.schema["close"] == pl.Datetime("ns")
    assert frame["date"].to_list() == [
        date(2024, 1, 24),
        date(2024, 3, 11),
        date(2024, 11, 29),
    ]


# --- custom window inside regular hours -------------------------------------


def test_custom_rth_window_winter_and_half_day():
    cal = XnysSessionCalendar(session_start="10:00", session_end="15:30")
    assert _bounds(cal, date(2024, 1, 24)) == (
        datetime(2024, 1, 24, 15, 0),
        datetime(2024, 1, 24, 20, 30),
    )
    # 15:30 lies inside regular hours -> clipped to the 13:00 early close.
    assert _bounds(cal, date(2024, 11, 29)) == (
        datetime(2024, 11, 29, 15, 0),
        datetime(2024, 11, 29, 18, 0),
    )


def test_half_day_empty_clipped_window_is_omitted():
    cal = XnysSessionCalendar(session_start="13:30", session_end="16:00")
    frame = cal.session_bounds([date(2024, 1, 24), date(2024, 11, 29)])
    assert frame["date"].to_list() == [date(2024, 1, 24)]
    row = frame.row(0, named=True)
    assert (row["open"], row["close"]) == (
        datetime(2024, 1, 24, 18, 30),
        datetime(2024, 1, 24, 21, 0),
    )


# --- extended window (D-29) --------------------------------------------------


def test_extended_window_winter_crosses_utc_midnight():
    cal = XnysSessionCalendar(session_start="04:00", session_end="20:00")
    assert _bounds(cal, date(2024, 1, 24)) == (
        datetime(2024, 1, 24, 9, 0),
        datetime(2024, 1, 25, 1, 0),
    )


def test_extended_window_dst():
    cal = XnysSessionCalendar(session_start="04:00", session_end="20:00")
    assert _bounds(cal, date(2024, 3, 11)) == (
        datetime(2024, 3, 11, 8, 0),
        datetime(2024, 3, 12, 0, 0),
    )


def test_extended_window_half_day_unchanged():
    cal = XnysSessionCalendar(session_start="04:00", session_end="20:00")
    assert _bounds(cal, date(2024, 11, 29)) == (
        datetime(2024, 11, 29, 9, 0),
        datetime(2024, 11, 30, 1, 0),
    )


def test_extended_mixed_after_hours_close_on_half_day():
    cal = XnysSessionCalendar(session_start="09:30", session_end="17:00")
    # 17:00 is an after-hours edge -> wall clock, unchanged on the half day.
    assert _bounds(cal, date(2024, 11, 29)) == (
        datetime(2024, 11, 29, 14, 30),
        datetime(2024, 11, 29, 22, 0),
    )


def test_extended_mixed_premarket_open_on_half_day():
    cal = XnysSessionCalendar(session_start="07:00", session_end="16:00")
    # 16:00 is a regular-hours edge -> clipped to the 13:00 early close.
    assert _bounds(cal, date(2024, 11, 29)) == (
        datetime(2024, 11, 29, 12, 0),
        datetime(2024, 11, 29, 18, 0),
    )


def test_extended_bounds_accepted():
    XnysSessionCalendar(session_start="04:00", session_end="20:00")
    XnysSessionCalendar(session_start="04:00:00", session_end="20:00:00")


@pytest.mark.parametrize(
    "start, end",
    [
        ("03:59", "16:00"),  # extended: before 04:00
        ("09:30", "20:01"),  # extended: after 20:00
        ("16:00", "09:30"),  # start >= end
        ("10:00", "10:00"),  # start == end
        ("9h30", "16:00"),  # malformed
        ("09:30", "25:00"),  # malformed hour
    ],
)
def test_extended_invalid_window_refused(start, end):
    with pytest.raises(ValueError):
        XnysSessionCalendar(session_start=start, session_end=end)


# --- dependency declaration (D-23) -------------------------------------------


def test_dependencies_declared():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    names = [
        dep.split(";")[0].split("[")[0].split("<")[0].split(">")[0].split("=")[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dep in project["dependencies"]
    ]
    assert "exchange-calendars" in names
    assert "psycopg2-binary" in names


def test_module_does_not_import_acquisition():
    source = (REPO_ROOT / "quantlab/dataset/_support/session_calendar.py").read_text()
    assert "quantlab.acquisition" not in source
