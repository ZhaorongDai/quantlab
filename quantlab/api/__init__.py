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
>>> import pandas as pd
>>> import quantlab.api as qa
>>> frame = pd.DataFrame({
...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
...     "symbol": ["AAA", "BBB", "AAA", "BBB"],
...     "open": [10.0, 20.0, 10.5, 19.0], "high": [11.0, 21.0, 11.5, 20.5],
...     "low": [9.5, 19.5, 10.0, 18.5], "close": [10.5, 20.5, 11.0, 19.5],
...     "volume": [1e6, 2e6, 1.1e6, 1.8e6],
... })
>>> qa.compute_factors(frame, "alpha158")[["timestamp", "symbol", "KMID"]]
   timestamp symbol      KMID
0 2024-01-02    AAA  0.707107
1 2024-01-02    BBB -0.707107
2 2024-01-03    AAA  0.707107
3 2024-01-03    BBB -0.707107
"""

from collections.abc import Mapping

from quantlab.api import _factors, _labels


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

        A class from this list, or a subclass of one, reads the columns of its short
        name. Any other class is built with its ``config_cls``, no warm-up and batch mode,
        on a dataset holding every column of the frame under its canonical name; a
        KunQuant factor is fed every numeric column. Factors needing fundamentals or
        factor-return series (``LiteratureAlpha``, ``ResidualMomentumFF3``,
        ``MarketFeatures``) are not in the catalog.
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


def forward_returns(
    frame,
    *,
    price: str = "open",
    span: int = 1,
    delay: int = 1,
    binary: bool = False,
    columns: Mapping[str, str] | None = None,
    as_xarray: bool = False,
):
    """Compute the forward return of each bar, the label factors are judged against.

    The label at bar ``t`` is the return of a position entered ``delay`` bars later and
    held ``span`` bars, both at the ``price`` column::

        price[t + delay + span] / price[t + delay] - 1

    ``delay + span`` is the label's lookahead: the last ``delay + span`` bars of the
    frame have no later bars to read and are NaN. The defaults match the library's
    ``Return`` label, a signal at bar ``t`` filled at bar ``t + 1``'s open. The whole frame
    is computed at once.

    Parameters
    ----------
    frame : pandas.DataFrame or polars.DataFrame
        Bars in long form: ``timestamp``, ``symbol`` and the ``price`` column.
    price : str, default "open"
        The column the return is computed on, after ``columns`` renames. There is no
        fallback to another column: use the column a backtest fills at, so the label and
        the fill agree.
    span : int, default 1
        Bars the return is held over; at least 1.
    delay : int, default 1
        Bars between the signal bar and the entry bar; at least 0. With 0 the position is
        entered at the signal bar's own price.
    binary : bool, default False
        Return 1.0 where the forward return is positive and 0.0 elsewhere (the library's
        ``BinaryReturn``) instead of the return (``Return``).
    columns : mapping of str to str, optional
        Renames the frame's columns onto the canonical names, ``{"date": "timestamp",
        "ticker": "symbol", "Open": "open"}``.
    as_xarray : bool, default False
        Return the ``xarray.Dataset`` panel instead of a frame.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame or xarray.Dataset
        One row per ``(timestamp, symbol)`` of the frame's full grid, columns
        ``timestamp``, ``symbol`` and ``ret_{span}`` (``ret_binary_{span}`` with
        ``binary=True``), in the frame's library; or the panel on ``(timestamp, symbol)``
        with ``as_xarray=True``. The values are float32, computed by the library's
        KunQuant label path, so compare them with float64 returns at a float32-level
        tolerance, not exactly.

    Raises
    ------
    ValueError
        If the ``price`` column is missing (the message lists the columns present and
        suggests one as ``price=``),
        ``span`` is below 1, ``delay`` is below 0, ``columns`` names an absent column, or
        a ``(timestamp, symbol)`` pair repeats.
    TypeError
        If ``frame`` is not a pandas or polars DataFrame, ``price`` is not a string,
        ``span`` or ``delay`` is not an int, or ``binary`` is not a bool.

    Examples
    --------
    >>> import pandas as pd
    >>> import quantlab.api as qa
    >>> frame = pd.DataFrame({
    ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
    ...     "symbol": ["AAA"] * 3 + ["BBB"] * 3,
    ...     "open": [10.0, 11.0, 12.1, 20.0, 19.0, 19.0],
    ...     "close": [10.5, 11.5, 12.5, 19.5, 19.2, 18.0],
    ... })
    >>> qa.forward_returns(frame)
       timestamp symbol  ret_1
    0 2024-01-02    AAA    0.1
    1 2024-01-02    BBB    0.0
    2 2024-01-03    AAA    NaN
    3 2024-01-03    BBB    NaN
    4 2024-01-04    AAA    NaN
    5 2024-01-04    BBB    NaN
    >>> qa.forward_returns(frame, price="close", delay=0, binary=True)
       timestamp symbol  ret_binary_1
    0 2024-01-02    AAA           1.0
    1 2024-01-02    BBB           0.0
    2 2024-01-03    AAA           1.0
    3 2024-01-03    BBB           0.0
    4 2024-01-04    AAA           NaN
    5 2024-01-04    BBB           NaN
    """
    return _labels.forward_returns(
        frame,
        price=price,
        span=span,
        delay=delay,
        binary=binary,
        columns=columns,
        as_xarray=as_xarray,
    )
