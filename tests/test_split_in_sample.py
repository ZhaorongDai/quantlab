"""The split module's backtest in-sample line, tested on synthetic axes.

``in_sample_window`` turns the fitted training window into the bars the model
has seen: the training bars plus the L bars the last training label reads.
``split_ranges`` cuts a backtest's bars into in-sample runs and the
out-of-sample runs between them.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.utils.split import in_sample_window, split_ranges

BDAYS = pd.bdate_range("2024-01-01", periods=20).values


def test_in_sample_window_ends_lookahead_bars_after_train_end_across_a_weekend():
    friday = "2024-01-05"
    assert pd.Timestamp(friday).day_name() == "Friday"

    window = in_sample_window(BDAYS, "2024-01-01", friday, lookahead=2)

    # Friday + 2 bars = Monday, Tuesday. Two calendar days would give Sunday.
    assert window == (np.datetime64("2024-01-01"), np.datetime64("2024-01-09"))


#: Three 7-bar hourly sessions (10:00..16:00); no bar sits at midnight.
SESSION_DAYS = ("2024-01-01", "2024-01-02", "2024-01-03")
SESSION_HOURS = tuple(range(10, 17))


def _sessions(days, hours=SESSION_HOURS) -> np.ndarray:
    return pd.DatetimeIndex(
        [pd.Timestamp(f"{day} {hour:02d}:00") for day in days for hour in hours]
    ).values


def test_intraday_in_sample_window_ends_lookahead_bars_into_the_next_session():
    """A date-only ``train_end`` covers its whole session, as the model slices it.

    The last training label sits at 16:00 and reads the next two bars, 10:00
    and 11:00 of the FOLLOWING session.
    """
    calendar = _sessions(SESSION_DAYS)
    model_layer_slice = xr.DataArray(
        np.arange(calendar.size), dims="timestamp", coords={"timestamp": calendar}
    ).sel(timestamp=slice("2024-01-01", "2024-01-02"))
    assert pd.Timestamp(model_layer_slice.timestamp.values[-1]) == pd.Timestamp(
        "2024-01-02 16:00"
    )

    window = in_sample_window(calendar, "2024-01-01", "2024-01-02", lookahead=2)

    assert window == (
        np.datetime64("2024-01-01T10:00"),
        np.datetime64("2024-01-03T11:00"),
    )


@pytest.mark.parametrize(
    "train_end",
    [
        "2024-01-02T13:00:00",
        "2024-01-02T13:00:00.000000000",
        pd.Timestamp("2024-01-02 13:00"),
        np.datetime64("2024-01-02T13:00"),
    ],
    ids=["iso", "fold-style-ns", "timestamp", "datetime64"],
)
def test_in_sample_window_honours_a_time_of_day_train_end(train_end):
    """A trained run's ``run.json`` stores nanosecond strings; they are not cut to a date."""
    window = in_sample_window(
        _sessions(SESSION_DAYS), "2024-01-01", train_end, lookahead=2
    )

    assert window == (
        np.datetime64("2024-01-01T10:00"),
        np.datetime64("2024-01-02T15:00"),
    )


def test_zero_lookahead_ends_the_window_on_train_end():
    window = in_sample_window(BDAYS, "2024-01-01", "2024-01-05", lookahead=0)

    assert window == (np.datetime64("2024-01-01"), np.datetime64("2024-01-05"))


def test_in_sample_window_stops_at_the_last_calendar_bar():
    window = in_sample_window(BDAYS, "2024-01-01", BDAYS[-2], lookahead=5)

    assert window == (np.datetime64("2024-01-01"), BDAYS[-1])


def test_in_sample_window_with_no_training_bar_is_none():
    assert in_sample_window(BDAYS, "2023-01-01", "2023-06-30", lookahead=1) is None


def test_split_ranges_cuts_one_window_into_in_sample_and_the_runs_around_it():
    window = (BDAYS[3], BDAYS[5])

    in_sample, out_of_sample = split_ranges(BDAYS[:10], [window])

    assert in_sample == [(BDAYS[3], BDAYS[5])]
    assert out_of_sample == [(BDAYS[0], BDAYS[2]), (BDAYS[6], BDAYS[9])]
