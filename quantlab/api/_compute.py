"""Computing a factor or label over every bar of a caller's panel.

``compute_factors`` and ``forward_returns`` share one shape: the caller's frame becomes a
panel, the panel a ``FrameDataset``, a factor or label is built on that dataset, computed
over every bar at once, and the result goes back out in the caller's library.
"""

import os
from collections.abc import Callable

import pandas as pd
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.utils.frame import INDEX_COLUMNS, Library, library_of, to_frame

#: KunQuant threads a computation runs on: every CPU.
NJOBS = os.cpu_count() or 1


def output_library(data, as_xarray: bool) -> Library:
    """Return the library a result for ``data`` goes back out in.

    ``data`` is checked even with ``as_xarray``, so an input of another type raises the
    same ``TypeError`` either way.
    """
    library = library_of(data)
    return "xarray" if as_xarray else library


def compute_over(panel: xr.Dataset, build: Callable[[FrameDataset], object], library: Library):
    """Compute ``build(FrameDataset(panel))`` over every bar of ``panel``, in ``library``.

    Parameters
    ----------
    panel : xr.Dataset
        The caller's data, already under the field names the factor reads.
    build : callable
        Returns the factor or label to compute, given the dataset holding ``panel``.
    library : {"pandas", "polars", "xarray"}
        The library of the result.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame or xarray.Dataset
        The computed variables on ``(timestamp, symbol)``.
    """
    factor = build(FrameDataset(panel))
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    result = factor.compute(timestamps[0], timestamps[-1])
    return to_frame(result.transpose(*INDEX_COLUMNS), library)
