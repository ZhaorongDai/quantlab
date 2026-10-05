"""The split module's purge, tested on synthetic timestamp axes.

``purge_segments`` takes a sorted axis, ordered inclusive ``(start, end)``
segments and the lookahead L, and returns each segment's usable timestamps:
every segment that has a later one loses its last L bars, so no label fitted
on it reads a bar of the next segment.
"""

import numpy as np
import pandas as pd

from quantlab.model.split import purge_segments

DAYS = pd.date_range("2024-01-01", periods=10, freq="D").values


def day(i):
    return DAYS[i]


def test_two_segments_drop_the_last_lookahead_bars_of_the_first():
    train, test = purge_segments(
        DAYS, [(day(0), day(5)), (day(6), day(9))], lookahead=2
    )
    np.testing.assert_array_equal(train, DAYS[0:4])
    np.testing.assert_array_equal(test, DAYS[6:10])


def test_zero_lookahead_keeps_every_bar():
    train, test = purge_segments(
        DAYS, [(day(0), day(5)), (day(6), day(9))], lookahead=0
    )
    np.testing.assert_array_equal(train, DAYS[0:6])
    np.testing.assert_array_equal(test, DAYS[6:10])


def test_every_segment_but_the_last_is_purged():
    train, val, test = purge_segments(
        DAYS, [(day(0), day(3)), (day(4), day(6)), (day(7), day(9))], lookahead=1
    )
    np.testing.assert_array_equal(train, DAYS[0:3])
    np.testing.assert_array_equal(val, DAYS[4:6])
    np.testing.assert_array_equal(test, DAYS[7:10])


def test_a_segment_no_longer_than_the_lookahead_comes_back_empty():
    train, test = purge_segments(
        DAYS, [(day(0), day(1)), (day(2), day(9))], lookahead=3
    )
    assert len(train) == 0
    np.testing.assert_array_equal(test, DAYS[2:10])


def test_bounds_between_bars_select_the_bars_inside():
    axis = DAYS[::2]  # every other day
    train, test = purge_segments(
        axis,
        [(day(1), day(7)), (np.datetime64("2024-01-08T12"), day(9))],
        lookahead=1,
    )
    # Bars inside [day 1, day 7] are days 2, 4, 6; the last one is purged.
    np.testing.assert_array_equal(train, [day(2), day(4)])
    np.testing.assert_array_equal(test, [day(8)])


def test_a_date_string_bound_covers_the_whole_day_on_an_intraday_axis():
    hours = pd.date_range("2024-01-01", periods=72, freq="h").values
    first, second = purge_segments(
        hours, [("2024-01-01", "2024-01-02"), ("2024-01-03", "2024-01-03")], lookahead=0
    )
    np.testing.assert_array_equal(first, hours[:48])
    np.testing.assert_array_equal(second, hours[48:])
