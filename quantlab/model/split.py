"""Splits of a timestamp axis: purged segments, in/out-of-sample ranges and walk-forward folds.

A label at bar t reads bars up to t + L, L being its lookahead. A segment followed
by another therefore loses its last L bars, so no label fitted on it reads a bar of
the next segment. The model's train/validation/test splits, the backtest's
in-sample/out-of-sample split (``in_sample_window``, ``split_ranges``) and the
walk-forward folds all go through ``purge_segments``.

A walk-forward cross-validation trains on a window of bars and tests on the bars
right after it, then steps forward by one test segment. Fold ``i`` tests on the
``test_periods`` bars from position ``i * test_periods + train_periods`` on, and
its training window ends right before them. The window starts at position
``i * test_periods`` (sliding) or at the first bar (expanding), so both modes test
on the same bars. There are ``max(1, (T - train_periods) // test_periods)`` folds,
``T`` being the number of bars, and a fold whose test segment runs past the end is
skipped. ``test_periods`` defaults to ``train_periods // 5``. The training window
loses its last ``purge_bars`` bars before it is fitted; each ``Fold`` carries both
windows, the one configured (which a model is given and purges itself) and the one
actually fitted.

The module imports no quantlab layer: the caller computes ``purge_bars`` from its
labels and cuts the axis to its own date range. Model and ensemble
cross-validation both plan their folds here.

Examples
--------
>>> import numpy as np
>>> bars = np.arange("2024-01-01", "2024-01-21", dtype="datetime64[D]")
>>> folds = walk_forward_folds(bars, 10, test_periods=5, purge_bars=2)
>>> len(folds)
2
>>> folds[1]
Fold(index=1, train_window=('2024-01-06', '2024-01-15'), fitted_train_window=('2024-01-06', '2024-01-13'), test_window=('2024-01-16', '2024-01-20'))
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


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


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold; every window is ``(start, end)``, both ends inclusive.

    Dates are ``np.datetime_as_string`` values of the axis' bars.

    Attributes
    ----------
    index : int
        The fold's position in the walk, from 0. A skipped fold leaves its
        index unused.
    train_window : tuple[str, str]
        The training window before the purge: what a model is configured with.
    fitted_train_window : tuple[str, str]
        The training window after the purge: the bars actually fitted.
    test_window : tuple[str, str]
        The test window.

    Examples
    --------
    >>> Fold(0, ("2024-01-01", "2024-01-10"), ("2024-01-01", "2024-01-08"),
    ...      ("2024-01-11", "2024-01-15")).fitted_train_window
    ('2024-01-01', '2024-01-08')
    """

    index: int
    train_window: tuple[str, str]
    fitted_train_window: tuple[str, str]
    test_window: tuple[str, str]


def walk_forward_folds(
    timestamps,
    train_periods: int,
    test_periods: int | None = None,
    expanding: bool = False,
    purge_bars: int = 0,
) -> tuple[Fold, ...]:
    """Return the walk-forward folds of a timestamp axis.

    Parameters
    ----------
    timestamps : array-like of datetime64
        The sorted bars to split, already cut to the caller's date range.
    train_periods : int
        Bars in each training window (the first one, when ``expanding``).
    test_periods : int, optional
        Bars in each test window. Defaults to ``train_periods // 5``.
    expanding : bool, default False
        Start every training window at the first bar instead of sliding it.
    purge_bars : int, default 0
        Bars dropped from the end of each training window before fitting:
        the largest ``lookahead_bars()`` of the labels.

    Returns
    -------
    tuple[Fold, ...]
        The folds in order. Empty when the axis is too short for one.

    Raises
    ------
    ValueError
        If ``test_periods`` is given and below 1, or not given and
        ``train_periods`` is below 5 (the one-fifth test window would be
        empty); if ``purge_bars`` is negative; or if the purge leaves a fold
        no training bar.

    Examples
    --------
    >>> import numpy as np
    >>> bars = np.arange("2024-01-01", "2024-01-21", dtype="datetime64[D]")
    >>> [fold.test_window for fold in walk_forward_folds(bars, 10)]
    [('2024-01-11', '2024-01-12'), ('2024-01-13', '2024-01-14'), ('2024-01-15', '2024-01-16'), ('2024-01-17', '2024-01-18'), ('2024-01-19', '2024-01-20')]
    >>> {fold.train_window[0] for fold in walk_forward_folds(bars, 10, expanding=True)}
    {'2024-01-01'}
    """
    test_periods = _test_periods(train_periods, test_periods)
    if purge_bars < 0:
        raise ValueError(f"purge_bars={purge_bars} cannot be negative.")
    timestamps = np.asarray(timestamps)
    total_periods = len(timestamps)
    n_splits = max(1, (total_periods - train_periods) // test_periods)

    folds = []
    for i in range(n_splits):
        train_end = i * test_periods + train_periods
        train_start = 0 if expanding else i * test_periods
        test_end = train_end + test_periods
        if test_end > total_periods:
            logger.warning(f"Skipping fold {i}: test set exceeds data range")
            continue
        train_window = _window(timestamps, train_start, train_end - 1)
        test_window = _window(timestamps, train_end, test_end - 1)
        fitted = _purged_train_window(timestamps, i, train_window, test_window, purge_bars)
        folds.append(Fold(i, train_window, fitted, test_window))
    return tuple(folds)


def _test_periods(train_periods: int, test_periods: int | None) -> int:
    """Return the test length: ``test_periods``, else ``train_periods // 5``."""
    if test_periods is not None:
        if test_periods < 1:
            raise ValueError(f"test_periods={test_periods} needs at least 1 test bar per fold.")
        return int(test_periods)
    if train_periods < 5:
        raise ValueError(
            f"train_periods={train_periods} needs at least 5 training bars, "
            f"since each fold tests on train_periods // 5 bars; or pass test_periods."
        )
    return train_periods // 5


def _window(timestamps: np.ndarray, first: int, last: int) -> tuple[str, str]:
    """Return the dates of positions ``first`` and ``last`` as a window."""
    return (
        str(np.datetime_as_string(timestamps[first])),
        str(np.datetime_as_string(timestamps[last])),
    )


def _purged_train_window(
    timestamps, index: int, train_window, test_window, purge_bars: int
) -> tuple[str, str]:
    """Return ``train_window`` with its end moved to the last bar the purge keeps."""
    usable, _ = purge_segments(timestamps, [train_window, test_window], purge_bars)
    if len(usable) == 0:
        raise ValueError(
            f"Fold {index}: purging the last {purge_bars} bars leaves no "
            f"training bar; raise train_periods."
        )
    return train_window[0], str(np.datetime_as_string(usable[-1]))
