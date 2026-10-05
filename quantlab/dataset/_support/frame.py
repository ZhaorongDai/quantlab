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
from typing import Literal, NamedTuple

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
    return to_panel_with_zone(data, columns=columns, required=required, purpose=purpose)[0]


def to_panel_with_zone(
    data,
    *,
    columns: Mapping[str, str] | None = None,
    required: Iterable[str] = (),
    purpose: str = "the frame",
) -> tuple[xr.Dataset, str | None]:
    """Return ``to_panel(data)`` and the time zone its timestamps came in.

    The panel's timestamps are naive UTC whatever the input's zone, so two inputs whose
    bars fail to line up may differ only in the zone they were written in; the zone lets
    an error message say so.

    Parameters
    ----------
    data, columns, required, purpose
        As for ``to_panel``.

    Returns
    -------
    panel : xr.Dataset
        As ``to_panel`` returns it.
    zone : str or None
        The name of the input timestamps' time zone, ``None`` when they were naive (an
        ``xarray`` panel's always are).

    Examples
    --------
    >>> frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02 09:30"]),
    ...                       "symbol": ["A"], "close": [1.0]})
    >>> frame["timestamp"] = frame["timestamp"].dt.tz_localize("America/New_York")
    >>> panel, zone = to_panel_with_zone(frame)
    >>> zone, str(pd.Timestamp(panel["timestamp"].values[0]))
    ('America/New_York', '2024-01-02 14:30:00')
    """
    if library_of(data) == "xarray":
        return _panel_to_panel(data, columns, tuple(required), purpose), None
    frame = _as_pandas(data)
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    frame = _rename(frame, columns, list(frame.columns), purpose)
    _check_not_in_index(frame, purpose)
    _check_present(list(frame.columns), INDEX_COLUMNS + tuple(required), purpose)
    _check_no_null_keys(frame, purpose)

    timestamps = pd.to_datetime(frame["timestamp"])
    zone = None if timestamps.dt.tz is None else str(timestamps.dt.tz)
    if zone is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    frame = frame.assign(timestamp=timestamps, symbol=frame["symbol"].astype(str))
    _check_unique(frame, purpose)

    panel = xr.Dataset.from_dataframe(frame.set_index(list(INDEX_COLUMNS)))
    return _sorted(panel), zone


class FieldPanel(NamedTuple):
    """A single-field input as a panel variable; see ``to_field_panel``."""

    #: The field on ``(timestamp, symbol)``, NaN where the input had no value.
    values: xr.DataArray
    #: ``True`` where the input gave a cell: a row of a long frame, a non-NaN cell of a
    #: wide frame, every cell of a panel.
    given: xr.DataArray
    #: The input timestamps' time zone, ``None`` when naive.
    zone: str | None


def to_field_panel(
    data,
    field: str,
    *,
    columns: Mapping[str, str] | None = None,
    purpose: str = "the frame",
) -> FieldPanel:
    """Return a single-field input as one panel variable, where it gave cells, its zone.

    A single-field input (weights, scores) carries one number per timestamp and symbol,
    so besides the long form it may come wide, one row per timestamp and one column per
    symbol. The shapes accepted:

    - long: ``timestamp`` and ``symbol`` columns (or a pandas ``(timestamp, symbol)``
      MultiIndex) and exactly one value column, of any name;
    - wide: the timestamps in a ``timestamp`` column or in a pandas ``DatetimeIndex``
      (whatever its name), and every other column a symbol;
    - an ``xarray.DataArray`` or a one-variable ``xarray.Dataset`` on ``(timestamp,
      symbol)``.

    A NaN cell of a wide frame counts as not given, like a row a long frame leaves out,
    so a sparse long frame and its pivot mean the same thing. ``columns`` renames the
    names ``data`` has among its keys and ignores the others (``columns_present``), so
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
    FieldPanel
        ``values`` (the field on ``(timestamp, symbol)``, named ``field``, NaN where not
        given or NaN), ``given`` (boolean on the same axes: a row of a long frame, a
        non-NaN cell of a wide frame, every cell of a panel) and ``zone`` (the input
        timestamps' time zone, ``None`` when naive).

    Raises
    ------
    TypeError
        If ``data`` is none of the accepted types.
    ValueError
        If a long frame has other than one value column, a wide frame has no timestamps
        or both a ``DatetimeIndex`` and a ``timestamp`` column, a panel has more than one
        variable, or a ``to_panel`` rule fails.

    Examples
    --------
    >>> long = pd.DataFrame({"timestamp": ["2024-01-02", "2024-01-02", "2024-01-03"],
    ...                      "symbol": ["A", "B", "A"], "w": [0.5, 0.5, 1.0]})
    >>> values, given, zone = to_field_panel(long, "weight")
    >>> values.values
    array([[0.5, 0.5],
           [1. , nan]])
    >>> given.values
    array([[ True,  True],
           [ True, False]])
    >>> wide = pd.DataFrame({"A": [0.5, 1.0], "B": [0.5, np.nan]},
    ...                     index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    >>> to_field_panel(wide, "weight").given.values
    array([[ True,  True],
           [ True, False]])
    """
    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name=field)
    if library_of(data) == "xarray":
        panel = _panel_to_panel(data, columns_present(columns, data), (), purpose)
        if len(panel.data_vars) != 1:
            raise ValueError(
                f"{purpose} must hold one variable on (timestamp, symbol), got "
                f"{list(panel.data_vars)}."
            )
        values = panel[next(iter(panel.data_vars))].rename(field).astype(float)
        return FieldPanel(values, xr.ones_like(values, dtype=bool), None)

    frame = _as_pandas(data)
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif isinstance(frame.index, pd.DatetimeIndex):
        if "timestamp" in frame.columns:
            raise ValueError(
                f"{purpose} has both a DatetimeIndex and a 'timestamp' column; keep the "
                f"timestamps in one of them."
            )
        frame = frame.rename_axis("timestamp").reset_index()
    frame = _rename(frame, columns_present(columns, frame), list(frame.columns), purpose)

    if "symbol" in frame.columns:
        value_columns = [name for name in frame.columns if name not in INDEX_COLUMNS]
        if len(value_columns) != 1:
            raise ValueError(
                f"{purpose} in long form needs one value column beside timestamp and "
                f"symbol, got {', '.join(repr(v) for v in value_columns) or 'none'}."
            )
        frame = frame.rename(columns={value_columns[0]: field}).assign(_given=True)
    elif "timestamp" in frame.columns:
        frame = frame.melt(id_vars="timestamp", var_name="symbol", value_name=field)
        frame = frame.assign(_given=frame[field].notna())
    else:
        raise ValueError(
            f"{purpose} is neither long (timestamp and symbol columns and one value "
            f"column) nor wide (timestamps in a 'timestamp' column or a DatetimeIndex, "
            f"one column per symbol). Present columns: {list(frame.columns)}."
        )
    panel, zone = to_panel_with_zone(frame, purpose=purpose)
    given = panel["_given"].fillna(False).astype(bool)
    return FieldPanel(panel[field].astype(float), given, zone)


def columns_present(columns: Mapping[str, str] | None, data) -> dict:
    """Return the entries of ``columns`` whose caller name ``data`` has.

    The names looked at are a frame's columns and, for pandas, its index level names,
    or a panel's dimensions and variables. A caller's one mapping can then be applied
    to every frame of a call, each renamed where it has the name.

    Parameters
    ----------
    columns : mapping of str to str or None
        ``{caller_name: name}``.
    data : pandas.DataFrame, polars.DataFrame, polars.LazyFrame, xarray.Dataset or xarray.DataArray
        The frame or panel.

    Returns
    -------
    dict
        The entries naming something ``data`` has; empty for ``None``.

    Raises
    ------
    TypeError
        If ``data`` is none of the accepted types.

    Examples
    --------
    >>> weights = pd.DataFrame({"date": [], "ticker": [], "w": []})
    >>> columns_present({"date": "timestamp", "ticker": "symbol", "Close": "close"}, weights)
    {'date': 'timestamp', 'ticker': 'symbol'}
    """
    if isinstance(data, xr.DataArray):
        names = set(data.dims)
    elif library_of(data) == "pandas":
        names = set(data.columns) | {name for name in data.index.names if name}
    elif library_of(data) == "xarray":
        names = set(data.dims) | set(data.data_vars)
    else:
        names = set(data.collect_schema().names())
    return {name: target for name, target in (columns or {}).items() if name in names}


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
    """Return ``panel`` sorted by timestamp, symbols in ``sort_symbol_axis`` order.

    The timestamps are cast to ``datetime64[ns]``, the resolution every store is read
    back in, so a panel held in memory and its copy on disk agree to the nanosecond
    (a duration averaged over them would otherwise round differently).
    """
    panel = panel.assign_coords(
        timestamp=panel["timestamp"].values.astype("datetime64[ns]")
    )
    order = sort_symbol_axis(panel["symbol"].values.tolist())
    panel = panel.sortby("timestamp").reindex(symbol=order)
    return panel.transpose(*INDEX_COLUMNS)
