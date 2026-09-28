"""Split a timestamp axis into segments and purge each split boundary.

A label at bar t reads bars up to t + L, L being its lookahead. A segment
followed by another therefore loses its last L bars, so no label fitted on
it reads a bar of the next segment. The model's train/validation/test
splits and the walk-forward folds all go through ``purge_segments``.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def purge_segments(
    timestamps: np.ndarray,
    segments: Sequence[tuple],
    lookahead: int,
) -> list[np.ndarray]:
    """Return each segment's usable timestamps after purging.

    Parameters
    ----------
    timestamps : np.ndarray
        The sorted timestamp axis the segments are cut from.
    segments : sequence of (start, end)
        Segment bounds in time order, both ends inclusive, resolved as
        ``xarray``'s ``sel(timestamp=slice(start, end))`` resolves them: a
        bound need not be a bar of the axis, and a date string covers its
        whole day.
    lookahead : int
        L, the largest ``lookahead_bars()`` among the model's labels.

    Returns
    -------
    list[np.ndarray]
        One array per segment: the axis bars inside its bounds, minus the
        last ``lookahead`` bars for every segment but the last. A segment
        with no more than ``lookahead`` bars comes back empty.
    """
    timestamps = np.asarray(timestamps)
    index = pd.DatetimeIndex(timestamps)
    result = []
    for i, (start, end) in enumerate(segments):
        lo, hi, _ = index.slice_indexer(start, end).indices(len(index))
        if i < len(segments) - 1:
            hi = max(lo, hi - lookahead)
        result.append(timestamps[lo:hi])
    return result


def in_sample_window(
    calendar: np.ndarray,
    train_start,
    train_end,
    lookahead: int,
) -> tuple[np.datetime64, np.datetime64] | None:
    """Return the first and last bar a fitted model has seen.

    The model fitted labels on the bars of ``[train_start, train_end]``, and
    the label of the last one reads ``lookahead`` bars further, so those bars
    are in-sample too.

    Parameters
    ----------
    calendar : np.ndarray
        The sorted bar axis the backtest trades on.
    train_start, train_end
        The fitted training window, both ends inclusive, resolved as
        ``purge_segments`` resolves segment bounds. ``train_end`` is the last
        bar fitted, after any purge.
    lookahead : int
        L, the largest ``lookahead_bars()`` among the model's labels.

    Returns
    -------
    tuple of np.datetime64 or None
        The first training bar and the bar ``lookahead`` bars after the last
        one, clamped to the calendar's last bar; ``None`` when the window
        holds no bar of the calendar.

    Examples
    --------
    >>> days = pd.bdate_range("2024-01-01", periods=10).values
    >>> in_sample_window(days, "2024-01-01", "2024-01-05", 2)
    (np.datetime64('2024-01-01T00:00:00.000000000'), np.datetime64('2024-01-09T00:00:00.000000000'))
    """
    calendar = np.asarray(calendar).astype("datetime64[ns]")
    index = pd.DatetimeIndex(calendar)
    lo, hi, _ = index.slice_indexer(train_start, train_end).indices(len(index))
    if lo >= hi:
        return None
    return calendar[lo], calendar[min(hi - 1 + lookahead, len(calendar) - 1)]


def split_ranges(
    timestamps: np.ndarray,
    windows: Sequence[tuple | None],
) -> tuple[list[tuple], list[tuple]]:
    """Cut a backtest's bars into in-sample and out-of-sample runs.

    Parameters
    ----------
    timestamps : np.ndarray
        The sorted bars of the backtest.
    windows : sequence of (first, last) or None
        In-sample windows such as ``in_sample_window`` returns, both ends
        inclusive and compared as exact timestamps; ``None`` entries are
        skipped.

    Returns
    -------
    in_sample : list of (first, last)
        The contiguous runs of bars inside any window, in time order.
    out_of_sample : list of (first, last)
        The contiguous runs of the remaining bars, in time order.

    Examples
    --------
    >>> days = pd.bdate_range("2024-01-01", periods=5).values
    >>> inside, outside = split_ranges(days, [(days[1], days[2])])
    >>> [(str(a)[:10], str(b)[:10]) for a, b in inside + outside]
    [('2024-01-02', '2024-01-03'), ('2024-01-01', '2024-01-01'), ('2024-01-04', '2024-01-05')]
    """
    timestamps = np.asarray(timestamps).astype("datetime64[ns]")
    mask = np.zeros(timestamps.size, dtype=bool)
    for window in windows:
        if window is not None:
            first, last = (np.datetime64(pd.Timestamp(bound), "ns") for bound in window)
            mask |= (timestamps >= first) & (timestamps <= last)
    return _runs(timestamps, mask), _runs(timestamps, ~mask)


def _runs(timestamps: np.ndarray, mask: np.ndarray) -> list[tuple]:
    """Return the first and last timestamp of every contiguous run of ``mask``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(timestamps[a], timestamps[b]) for a, b in zip(starts, ends)]
