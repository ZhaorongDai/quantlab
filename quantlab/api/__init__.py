"""Frame-in, frame-out functions for callers who hold their own DataFrames.

``quantlab.api`` lets a researcher with market data in a pandas or polars DataFrame use
one quantlab capability without adopting the project's stores and configs (ADR 0011). A
function takes a *frame*, a DataFrame in long form with one row per ``timestamp`` and
``symbol``, and returns a frame of the same library; ``as_xarray=True`` returns the
library's own ``xarray`` panel instead. Nothing is written to disk.

The canonical columns are ``timestamp``, ``symbol``, ``open``, ``high``, ``low``,
``close`` and ``volume``, plus ``amount`` (traded value) where a function needs it. The
``columns`` argument maps differently named columns onto them. The input rules:

- a pandas ``(timestamp, symbol)`` MultiIndex is reset automatically;
- a repeated ``(timestamp, symbol)`` pair raises, listing the first ones;
- missing ``(timestamp, symbol)`` cells become NaN;
- timezone-aware timestamps are converted to UTC and made naive;
- symbols are converted to ``str``.

Underneath, the frame becomes a ``quantlab.dataset.memory.FrameDataset``, a market
dataset held in memory, which advanced users can also plug into the full pipeline.

Examples
--------
>>> import quantlab.api as qa
>>> factors = qa.compute_factors(frame, "alpha158")
"""

from collections.abc import Mapping

from quantlab.api import _factors


def compute_factors(
    frame,
    factor,
    *,
    columns: Mapping[str, str] | None = None,
    as_xarray: bool = False,
):
    """Compute a factor set over every bar of a frame.

    The whole frame is computed at once, with no history before its first bar, so the
    leading bars of each rolling window are NaN rather than dropped.

    Parameters
    ----------
    frame : pandas.DataFrame or polars.DataFrame
        Bars in long form: ``timestamp``, ``symbol`` and the price and volume columns the
        factor reads.
    factor : str or type
        A short name, or any ``quantlab.base.factor.Factor`` subclass:

        - ``"alpha158"``, ``"alpha101"``: the equity variants (``Alpha158Stock``,
          ``Alpha101Stock``), z-scored across symbols with VWAP taken as
          ``(high + low + close) / 3``; read ``open``, ``high``, ``low``, ``close``,
          ``volume``, meant to be adjusted prices.
        - ``"alpha158_crypto"``, ``"alpha101_crypto"``: the crypto spot variants
          (``Alpha158SpotKline``, ``Alpha101SpotKline``), z-scored along time over 20
          bars; also read ``amount``.

        A class from this list behaves as its short name. Any other class is built with
        its ``config_cls``, no warm-up and batch mode, on a dataset holding every column
        of the frame under its canonical name; a KunQuant factor is fed every numeric
        column. Factors needing fundamentals or factor-return series
        (``LiteratureAlpha``, ``ResidualMomentumFF3``, ``MarketFeatures``) are not in the
        catalog.
    columns : mapping of str to str, optional
        Renames the frame's columns onto the canonical names, ``{"date": "timestamp",
        "ticker": "symbol", "Close": "close"}``.
    as_xarray : bool, default False
        Return the ``xarray.Dataset`` panel instead of a frame.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame or xarray.Dataset
        One row per ``(timestamp, symbol)`` of the frame's full grid, columns
        ``timestamp``, ``symbol`` and one per factor, in the frame's library; or the panel
        on ``(timestamp, symbol)`` with ``as_xarray=True``.

    Raises
    ------
    ValueError
        If the short name is unknown (the message lists the valid ones), a column the
        factor reads is missing (the message names it), ``columns`` names an absent
        column, or a ``(timestamp, symbol)`` pair repeats.
    TypeError
        If ``frame`` is not a pandas or polars DataFrame, or ``factor`` is neither a short
        name nor a ``Factor`` subclass.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import quantlab.api as qa
    >>> rng = np.random.default_rng(0)
    >>> timestamps = pd.bdate_range("2024-01-01", periods=30)
    >>> frame = pd.DataFrame({
    ...     "timestamp": np.repeat(timestamps, 3),
    ...     "symbol": ["AAA", "BBB", "CCC"] * 30,
    ...     "close": 100 + rng.normal(size=90).cumsum(),
    ... })
    >>> frame["open"] = frame["close"] * 0.99
    >>> frame["high"] = frame["close"] * 1.01
    >>> frame["low"] = frame["close"] * 0.98
    >>> frame["volume"] = 1e6
    >>> factors = qa.compute_factors(frame, "alpha158")
    >>> factors.shape
    (90, 171)
    >>> factors[["timestamp", "symbol", "KMID"]].head(3)
       timestamp symbol      KMID
    0 2024-01-01    AAA  0.865969
    1 2024-01-01    BBB -1.089445
    2 2024-01-01    CCC  0.251410
    >>> int(factors["STD5"].isna().sum())  # 4 warm-up bars x 3 symbols
    12
    """
    return _factors.compute_factors(frame, factor, columns=columns, as_xarray=as_xarray)
