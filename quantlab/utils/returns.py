"""One-bar returns of a price panel: ``price[t] / price[t - 1] - 1``.

``one_bar_returns`` is the one copy of the formula the factor, risk and
portfolio layers share, so they agree on its rule: the first bar has no
return, and a bar where either price is missing has none either (NaN, never
0). A caller that wants a halt to show as a zero return forward-fills the
prices first.
"""

import numpy as np
import xarray as xr


def one_bar_returns(prices: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
    """Return ``prices[t] / prices[t - 1] - 1`` along the time axis, NaN on the first bar.

    A return is NaN where either price is missing; a previous price of 0
    gives what the division gives (``inf`` or NaN), with no warning. The
    result keeps the input's float precision (an integer input gives
    float64).

    Parameters
    ----------
    prices : np.ndarray or xr.DataArray
        An array with time on its first axis (``[T]`` or ``[T, S]``), or a
        data array with a ``timestamp`` dimension anywhere.

    Returns
    -------
    np.ndarray or xr.DataArray
        The returns, the shape of ``prices``; a data array keeps its
        name, dimensions, their order, coordinates and attributes, as
        arithmetic on it does.

    Examples
    --------
    >>> import numpy as np
    >>> one_bar_returns(np.array([[10.0, 20.0], [11.0, np.nan], [12.1, 22.0]]))
    array([[nan, nan],
           [0.1, nan],
           [0.1, nan]])
    """
    if isinstance(prices, xr.DataArray):
        dims = prices.dims
        ordered = prices.transpose("timestamp", ...)
        return ordered.copy(data=one_bar_returns(ordered.values)).transpose(*dims)
    prices = np.asarray(prices)
    returns = np.full(prices.shape, np.nan, dtype=np.promote_types(prices.dtype, np.float32))
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[1:] = prices[1:] / prices[:-1] - 1.0
    return returns


__all__ = ["one_bar_returns"]
