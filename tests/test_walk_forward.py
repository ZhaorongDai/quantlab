"""Walk-forward folds from one public module (#121).

``walk_forward_folds`` turns a timestamp axis and the CV settings into
immutable folds, each with its configured training window, the training window
actually fitted after the purge, and its test window. What is locked here,
through the function alone, on a hand-built axis of 30 daily bars:

- fold ``i`` tests on the ``test_periods`` bars from ``i * test_periods +
  train_periods`` on, its training window ends right before them and starts at
  ``i * test_periods`` (sliding) or at the first bar (expanding);
- there are ``max(1, (T - train_periods) // test_periods)`` folds, a fold
  whose test segment runs past the end is skipped, and ``test_periods``
  defaults to ``train_periods // 5``;
- the purge moves the fitted training window's end back by ``purge_bars`` bars
  and leaves the configured window alone;
- an invalid length, or a purge leaving no training bar, is refused.

That model and ensemble cross-validation train exactly these folds is locked
by the CV tests (``tests/test_model_cv*.py``, ``tests/test_seed_ensemble_cv.py``).
"""

import dataclasses

import numpy as np
import pytest

from quantlab.utils.walk_forward import Fold, walk_forward_folds

BARS = np.arange("2024-01-01", "2024-01-31", dtype="datetime64[D]").astype("datetime64[ns]")


def _day(i: int) -> str:
    """The date of bar ``i``."""
    return np.datetime_as_string(BARS[i])


def _window(start: int, end: int) -> tuple[str, str]:
    """The window from bar ``start`` to bar ``end``."""
    return _day(start), _day(end)


@pytest.mark.parametrize(
    "train_periods, test_periods, expanding, purge_bars, expected",
    [
        # 30 bars, 10 training bars, the default test length 10 // 5 = 2:
        # (30 - 10) // 2 = 10 folds.
        (10, None, False, 0, [(i, (2 * i, 2 * i + 9), (2 * i, 2 * i + 9), (2 * i + 10, 2 * i + 11)) for i in range(10)]),
        # Expanding: every fold trains from the first bar, tests on the same bars.
        (10, None, True, 0, [(i, (0, 2 * i + 9), (0, 2 * i + 9), (2 * i + 10, 2 * i + 11)) for i in range(10)]),
        # An explicit test length: (30 - 10) // 7 = 2 folds.
        (10, 7, False, 0, [(0, (0, 9), (0, 9), (10, 16)), (1, (7, 16), (7, 16), (17, 23))]),
        # A purge of 3 bars ends each fitted window 3 bars early.
        (10, 7, False, 3, [(0, (0, 9), (0, 6), (10, 16)), (1, (7, 16), (7, 13), (17, 23))]),
        (10, 7, True, 3, [(0, (0, 9), (0, 6), (10, 16)), (1, (0, 16), (0, 13), (17, 23))]),
    ],
)
def test_folds(train_periods, test_periods, expanding, purge_bars, expected):
    folds = walk_forward_folds(
        BARS, train_periods, test_periods=test_periods, expanding=expanding, purge_bars=purge_bars
    )

    assert folds == tuple(
        Fold(index, _window(*train), _window(*fitted), _window(*test))
        for index, train, fitted, test in expected
    )


def test_a_fold_running_past_the_end_is_skipped():
    # 28 training bars leave max(1, (30 - 28) // 5) = 1 fold, whose 5 test
    # bars would end at bar 32.
    assert walk_forward_folds(BARS, 28, test_periods=5) == ()


def test_a_short_axis_gives_no_fold():
    assert walk_forward_folds(BARS[:3], 10) == ()


def test_a_fold_is_immutable():
    fold = walk_forward_folds(BARS, 10)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        fold.index = 1  # type: ignore[misc]


@pytest.mark.parametrize("train_periods", [0, 4])
def test_the_default_test_length_needs_five_training_bars(train_periods):
    with pytest.raises(ValueError, match=f"train_periods={train_periods}.*at least 5"):
        walk_forward_folds(BARS, train_periods)


def test_a_small_training_length_is_accepted_with_an_explicit_test_length():
    assert len(walk_forward_folds(BARS, 4, test_periods=4)) == (30 - 4) // 4


@pytest.mark.parametrize("test_periods", [0, -1])
def test_a_non_positive_test_length_is_refused(test_periods):
    with pytest.raises(ValueError, match=f"test_periods={test_periods}"):
        walk_forward_folds(BARS, 10, test_periods=test_periods)


def test_a_negative_purge_is_refused():
    with pytest.raises(ValueError, match="purge_bars=-1"):
        walk_forward_folds(BARS, 10, purge_bars=-1)


def test_a_purge_leaving_no_training_bar_is_refused():
    with pytest.raises(ValueError, match="Fold 0.*purging the last 10 bars"):
        walk_forward_folds(BARS, 10, purge_bars=10)
