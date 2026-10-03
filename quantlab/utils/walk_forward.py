"""Walk-forward folds: how a timestamp axis is split for cross-validation.

A walk-forward cross-validation trains on a window of bars and tests on the
bars right after it, then steps forward by one test segment. Fold ``i`` tests
on the ``test_periods`` bars from position ``i * test_periods +
train_periods`` on, and its training window ends right before them. The
window starts at position ``i * test_periods`` (sliding) or at the first bar
(expanding), so both modes test on the same bars. There are ``max(1, (T -
train_periods) // test_periods)`` folds, ``T`` being the number of bars, and a
fold whose test segment runs past the end is skipped. ``test_periods``
defaults to ``train_periods // 5``.

A label at bar t reads bars up to t + L, so the training window loses its last
``purge_bars`` bars before it is fitted (``quantlab.utils.split.purge_segments``).
Each ``Fold`` carries both windows: the one configured, which a model is given
and purges itself, and the one actually fitted.

The module imports no quantlab layer: the caller computes ``purge_bars`` from
its labels and cuts the axis to its own date range. Model and ensemble
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

from dataclasses import dataclass

import numpy as np
from loguru import logger

from quantlab.utils.split import purge_segments


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
