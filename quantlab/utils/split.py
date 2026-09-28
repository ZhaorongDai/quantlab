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
