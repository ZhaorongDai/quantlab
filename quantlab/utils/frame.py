"""Conversion between a caller's frame and a panel.

A *frame* is a pandas or polars DataFrame held by a caller outside the pipeline, in long
form with one row per ``(timestamp, symbol)``. A *panel* is the ``xarray.Dataset`` on
``(timestamp, symbol)`` every layer exchanges. ``to_panel`` turns a frame (or a panel) into
a dense panel under one set of input rules, ``to_field_panel`` does the same for a
single-field input that may also come wide (one column per symbol), and ``to_frame`` turns
a result panel back into a frame of the caller's library. ``quantlab.api`` and ``quantlab.dataset.memory`` share
these rules, so a frame means the same thing wherever it enters.

The input rules:

- a pandas ``(timestamp, symbol)`` MultiIndex is moved into columns;
- ``columns`` renames the caller's columns first (``{"date": "timestamp"}``);
- ``timestamp`` and ``symbol`` columns are required, with no missing values; every
  other column becomes a variable;
- a repeated ``(timestamp, symbol)`` pair raises, listing the first ones;
- timezone-aware timestamps are converted to UTC and made naive;
- symbols are cast to ``str``;
- missing ``(timestamp, symbol)`` cells become NaN, so the panel is a full grid sorted by
  timestamp and symbol.
"""

from collections.abc import Iterable, Mapping
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.utils.symbol_axis import sort_symbol_axis

#: The two index columns of a long frame.
INDEX_COLUMNS = ("timestamp", "symbol")

#: Offending rows or pairs named in an error message.
ROWS_SHOWN = 5

Library = Literal["pandas", "polars", "xarray"]


def library_of(data) -> Library:
    """Return which library ``data`` belongs to.

    Parameters
    ----------
    data : pandas.DataFrame, polars.DataFrame, polars.LazyFrame or xarray.Dataset
        A caller's frame or panel.

    Returns
    -------
    {"pandas", "polars", "xarray"}
        The library a result for this input is returned in.

    Raises
    ------
    TypeError
        If ``data`` is none of the above.

    Examples
    --------
    >>> library_of(pl.DataFrame({"timestamp": [], "symbol": []}))
    'polars'
    """
    if isinstance(data, pd.DataFrame):
        return "pandas"
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return "polars"
    if isinstance(data, xr.Dataset):
        return "xarray"
    raise TypeError(
        f"Expected a pandas or polars DataFrame in long form (one row per timestamp and "
        f"symbol) or an xarray.Dataset panel, got {type(data).__name__}."
    )


def to_panel(
    data,
    *,
    columns: Mapping[str, str] | None = None,
    required: Iterable[str] = (),
    purpose: str = "the frame",
) -> xr.Dataset:
    """Return ``data`` as a dense panel on ``(timestamp, symbol)``.

    Parameters
    ----------
    data : pandas.DataFrame, polars.DataFrame, polars.LazyFrame or xarray.Dataset
        A long frame, or a panel with ``timestamp`` and ``symbol`` dimensions.
    columns : mapping of str to str, optional
        Renames the caller's columns (or a panel's variables and dimensions) before
        anything else, ``{caller_name: name}``.
    required : iterable of str, default ()
        Variable names the panel must hold, besides ``timestamp`` and ``symbol``.
    purpose : str, default "the frame"
        What reads the data, for error messages (``"'alpha158'"``).

    Returns
    -------
    xr.Dataset
        The panel, sorted by timestamp and symbol, NaN where the input had no row.

    Raises
    ------
    TypeError
        If ``data`` is not a pandas or polars frame or an xarray panel.
    ValueError
        If ``columns`` names a column ``data`` lacks, a required column is missing, a row
        has no timestamp or symbol, or a ``(timestamp, symbol)`` pair repeats.

    Examples
    --------
    >>> frame = pd.DataFrame({"date": ["2024-01-02", "2024-01-03", "2024-01-02"],
    ...                       "ticker": [7, 7, 9], "close": [1.0, 2.0, 3.0]})
    >>> panel = to_panel(frame, columns={"date": "timestamp", "ticker": "symbol"})
    >>> panel["close"].values
    array([[ 1.,  3.],
           [ 2., nan]])
    >>> panel["symbol"].values.tolist()
    ['7', '9']
    """
    if library_of(data) == "xarray":
        return _panel_to_panel(data, columns, tuple(required), purpose)
    frame = _as_pandas(data)
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    frame = _rename(frame, columns, list(frame.columns), purpose)
    _check_not_in_index(frame, purpose)
    _check_present(list(frame.columns), INDEX_COLUMNS + tuple(required), purpose)
    _check_no_null_keys(frame, purpose)

    timestamps = pd.to_datetime(frame["timestamp"])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    frame = frame.assign(timestamp=timestamps, symbol=frame["symbol"].astype(str))
    _check_unique(frame, purpose)

    panel = xr.Dataset.from_dataframe(frame.set_index(list(INDEX_COLUMNS)))
    return _sorted(panel)


def to_field_panel(
    data,
    field: str,
    *,
    columns: Mapping[str, str] | None = None,
    purpose: str = "the frame",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return a single-field input as one panel variable, plus where it had a row.

    A single-field input (weights, scores) carries one number per timestamp and symbol,
    so besides the long form it may come wide, one row per timestamp and one column per
    symbol. The shapes accepted:

    - long: ``timestamp`` and ``symbol`` columns (or a pandas ``(timestamp, symbol)``
      MultiIndex) and exactly one value column, of any name;
    - wide: the timestamps in a ``timestamp`` column or, for pandas, a ``DatetimeIndex``,
      and every other column a symbol;
    - an ``xarray.DataArray`` or a one-variable ``xarray.Dataset`` on ``(timestamp,
      symbol)``.

    ``columns`` renames the names ``data`` has among its keys and ignores the others, so
    one mapping written for a caller's price frame also serves the single-field frames
    beside it. The input rules of ``to_panel`` then apply.

    Parameters
    ----------
    data : pandas.DataFrame, polars.DataFrame, xarray.DataArray or xarray.Dataset
        The input, long or wide.
    field : str
        The name the returned variable carries (``"weight"``).
    columns : mapping of str to str, optional
        Renames ``{caller_name: name}``, applied to the names ``data`` has.
    purpose : str, default "the frame"
        What the input is, for error messages (``"weights"``).

    Returns
    -------
    values : xr.DataArray
        The field on ``(timestamp, symbol)``, named ``field``, NaN where the input had no
        row or held NaN.
    given : xr.DataArray
        Boolean on the same axes: ``True`` where a long frame had a row, and everywhere
        for a wide frame or a panel, whose cells all exist.

    Raises
    ------
    TypeError
        If ``data`` is none of the accepted types.
    ValueError
        If a long frame has other than one value column, a wide frame has no timestamps,
        a panel has more than one variable, or a ``to_panel`` rule fails.

    Examples
    --------
    >>> long = pd.DataFrame({"timestamp": ["2024-01-02", "2024-01-02", "2024-01-03"],
    ...                      "symbol": ["A", "B", "A"], "w": [0.5, 0.5, 1.0]})
    >>> values, given = to_field_panel(long, "weight")
    >>> values.values
    array([[0.5, 0.5],
           [1. , nan]])
    >>> given.values
    array([[ True,  True],
           [ True, False]])
    >>> wide = pd.DataFrame({"A": [0.5, 1.0], "B": [0.5, np.nan]},
    ...                     index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    >>> bool(to_field_panel(wide, "weight")[1].all())
    True
    """
    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name=field)
    if library_of(data) == "xarray":
        present = list(data.dims) + list(data.data_vars)
        panel = _panel_to_panel(data, _present_keys(columns, present), (), purpose)
        if len(panel.data_vars) != 1:
            raise ValueError(
                f"{purpose} must hold one variable on (timestamp, symbol), got "
                f"{list(panel.data_vars)}."
            )
        values = panel[next(iter(panel.data_vars))].rename(field).astype(float)
        return values, xr.ones_like(values, dtype=bool)

    frame = _as_pandas(data)
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.rename_axis(frame.index.name or "timestamp").reset_index()
    names = list(frame.columns)
    frame = _rename(frame, _present_keys(columns, names), names, purpose)

    if "symbol" in frame.columns:
        values = [name for name in frame.columns if name not in INDEX_COLUMNS]
        if len(values) != 1:
            raise ValueError(
                f"{purpose} in long form needs one value column beside timestamp and "
                f"symbol, got {', '.join(repr(v) for v in values) or 'none'}."
            )
        frame = frame.rename(columns={values[0]: field}).assign(_given=True)
    elif "timestamp" in frame.columns:
        frame = frame.melt(id_vars="timestamp", var_name="symbol", value_name=field)
        frame = frame.assign(_given=True)
    else:
        raise ValueError(
            f"{purpose} is neither long (timestamp and symbol columns and one value "
            f"column) nor wide (timestamps in a 'timestamp' column or a DatetimeIndex, "
            f"one column per symbol). Present columns: {list(frame.columns)}."
        )
    panel = to_panel(frame, purpose=purpose)
    return panel[field].astype(float), panel["_given"].notnull()


def to_frame(panel: xr.Dataset, library: Library):
    """Return ``panel`` in ``library``: a long frame, or the panel itself.

    Parameters
    ----------
    panel : xr.Dataset
        A panel on ``(timestamp, symbol)``.
    library : {"pandas", "polars", "xarray"}
        The library to return, usually ``library_of`` the caller's input.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame or xarray.Dataset
        One row per ``(timestamp, symbol)`` of the panel, columns ``timestamp``,
        ``symbol`` and one per variable; the panel unchanged for ``"xarray"``.

    Examples
    --------
    >>> panel = xr.Dataset({"x": (("timestamp", "symbol"), [[1.0, 2.0]])},
    ...                    coords={"timestamp": [pd.Timestamp("2024-01-02")],
    ...                            "symbol": ["A", "B"]})
    >>> to_frame(panel, "polars").columns
    ['timestamp', 'symbol', 'x']
    """
    if library == "xarray":
        return panel
    frame = panel.transpose(*INDEX_COLUMNS).to_dataframe().reset_index()
    frame = frame[list(INDEX_COLUMNS) + list(panel.data_vars)]
    frame["symbol"] = frame["symbol"].astype(str)
    if library == "polars":
        return pl.from_pandas(frame)
    return frame


def _as_pandas(data) -> pd.DataFrame:
    """Return a pandas copy of a pandas or polars frame."""
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    return data.copy()


def _rename(obj, columns, present: list, purpose: str):
    """Apply ``columns`` to ``obj``, refusing a name ``obj`` does not have."""
    if not columns:
        return obj
    absent = [name for name in columns if name not in present]
    if absent:
        raise ValueError(
            f"columns= maps {absent} but {purpose} has no such column. Present columns: "
            f"{present}."
        )
    if isinstance(obj, xr.Dataset):
        return obj.rename(dict(columns))
    return obj.rename(columns=dict(columns))


def _present_keys(columns, present: list) -> dict:
    """Return the entries of ``columns`` whose caller name is among ``present``."""
    return {name: target for name, target in (columns or {}).items() if name in present}


def _check_present(present: list, needed: tuple, purpose: str) -> None:
    """Raise naming every needed column ``present`` lacks."""
    missing = [name for name in needed if name not in present]
    if missing:
        raise ValueError(
            f"{purpose} needs column(s) {', '.join(repr(m) for m in missing)}, which the "
            f"data does not have. Present columns: {present}. Pass columns={{'yours': "
            f"{missing[0]!r}}} to map one of yours onto it."
        )


def _check_not_in_index(frame: pd.DataFrame, purpose: str) -> None:
    """Raise suggesting ``reset_index()`` when an index column sits in the index."""
    indexed = [
        name
        for name in INDEX_COLUMNS
        if name not in frame.columns and name in (frame.index.names or [])
    ]
    if indexed:
        raise ValueError(
            f"{purpose} holds {', '.join(repr(n) for n in indexed)} in its index, not as a "
            f"column. Call frame.reset_index() first, or index the frame by both "
            f"timestamp and symbol."
        )


def _check_no_null_keys(frame: pd.DataFrame, purpose: str) -> None:
    """Raise counting and showing the rows whose timestamp or symbol is missing."""
    null = frame["timestamp"].isna() | frame["symbol"].isna()
    if not null.any():
        return
    shown = frame.loc[null, list(INDEX_COLUMNS)].head(ROWS_SHOWN)
    listed = ", ".join(f"row {i}: ({t}, {s!r})" for i, t, s in shown.itertuples())
    raise ValueError(
        f"{purpose} has {int(null.sum())} row(s) with a missing timestamp or symbol, for "
        f"example {listed}. Every row needs both; drop or fill those rows first."
    )


def _check_unique(frame: pd.DataFrame, purpose: str) -> None:
    """Raise listing the first repeated ``(timestamp, symbol)`` pairs."""
    repeated = frame.duplicated(list(INDEX_COLUMNS), keep=False)
    if not repeated.any():
        return
    pairs = frame.loc[repeated, list(INDEX_COLUMNS)].drop_duplicates()
    shown = ", ".join(
        f"({pd.Timestamp(t)}, {s!r})"
        for t, s in pairs.head(ROWS_SHOWN).itertuples(index=False)
    )
    raise ValueError(
        f"{purpose} has {len(pairs)} duplicate (timestamp, symbol) pair(s), for example "
        f"{shown}. Each pair must appear once; drop or aggregate the repeats first."
    )


def _panel_to_panel(
    panel: xr.Dataset, columns, required: tuple, purpose: str
) -> xr.Dataset:
    """Apply the input rules to a panel: rename, check, cast symbols, sort."""
    names = list(panel.dims) + list(panel.data_vars)
    panel = _rename(panel, columns, names, purpose)
    _check_present(list(panel.dims), INDEX_COLUMNS, purpose)
    _check_present(list(panel.data_vars), required, purpose)
    panel = panel.drop_vars(
        [name for name, var in panel.data_vars.items() if set(var.dims) != set(INDEX_COLUMNS)]
    )
    if pd.isna(panel["timestamp"].values).any() or pd.isna(panel["symbol"].values).any():
        raise ValueError(f"{purpose} has a missing timestamp or symbol label.")
    panel = panel.assign_coords(symbol=panel["symbol"].values.astype(str).astype(object))
    if len(np.unique(panel["timestamp"].values)) != panel.sizes["timestamp"] or len(
        np.unique(panel["symbol"].values)
    ) != panel.sizes["symbol"]:
        raise ValueError(f"{purpose} has a repeated timestamp or symbol label.")
    return _sorted(panel.transpose(*INDEX_COLUMNS))


def _sorted(panel: xr.Dataset) -> xr.Dataset:
    """Return ``panel`` sorted by timestamp, symbols in ``sort_symbol_axis`` order."""
    order = sort_symbol_axis(panel["symbol"].values.tolist())
    panel = panel.sortby("timestamp").reindex(symbol=order)
    return panel.transpose(*INDEX_COLUMNS)
