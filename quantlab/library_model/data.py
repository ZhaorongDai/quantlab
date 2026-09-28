"""The rows a ``LibraryModel`` head is fed.

The base class turns the collected panel into flat rows, one per
``(bar, symbol)`` cell with a valid training target, and hands them to the
head's ``_fit_model``. Every row carries its cell in ``where``, so a head
that needs the bar of a row (a per-bar objective, a grouped split) can read
it without rebuilding the panel.
"""

from typing import NamedTuple

import numpy as np


class Rows(NamedTuple):
    """The training or validation rows of one fit.

    Rows are ordered by bar, then by symbol.

    Attributes
    ----------
    x : np.ndarray
        Features after ``_transform_feature``, ``[n, F]`` float32. NaN is kept
        for the library's own missing-value handling.
    y : np.ndarray
        Training target, ``[n, L]`` float32, finite everywhere: the output of
        the head's ``_transform_target`` on the row's bar.
    y_raw : np.ndarray
        Raw labels, ``[n, L]`` float32.
    where : tuple[np.ndarray, np.ndarray]
        Timestamp and symbol index of each row in the collected panel,
        two ``[n]`` int64 arrays.

    Examples
    --------
    >>> rows = Rows(
    ...     x=np.zeros((2, 3), dtype=np.float32), y=np.zeros((2, 1), dtype=np.float32),
    ...     y_raw=np.zeros((2, 1), dtype=np.float32),
    ...     where=(np.array([0, 0]), np.array([0, 1])),
    ... )
    >>> len(rows.x), rows.where[1].tolist()
    (2, [0, 1])
    """

    x: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    where: tuple[np.ndarray, np.ndarray]
