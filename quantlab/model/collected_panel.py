"""The collected panel: what a model's ``collect()`` read, and the arrays built from it.

A model reads its factors and labels over its own date range into one panel
indexed by ``(timestamp, symbol)``. ``CollectedPanel`` holds that panel sorted
on both axes, the record of the reads and what it was read for, and builds the
numpy arrays the model fits and predicts on once each. Every model holding the
same collected panel, each walk-forward fold of a model and the members of a
seed ensemble, reads the same arrays. The module loads no torch.

Examples
--------
>>> panel = CollectedPanel(dataset, record={}, source=None)
>>> x, timestamps, symbols = panel.array(["f_a", "f_b"], np.float32)
>>> x.shape, x.dtype
((40, 3, 2), dtype('float32'))
"""

import numpy as np
import xarray as xr

__all__ = ["CollectedPanel", "sorted_panel", "to_array"]

#: ``to_array`` loads this many variables at a time (a lazy panel is never
#: loaded whole beside the result) and writes them a block of this many bars
#: at a time: each write along the variable axis is strided, and a small block
#: keeps those writes in cache (about 5x faster than writing whole variables,
#: measured on a 1,500 x 10,000 x 40 panel).
GROUP_VARIABLES = 32
BLOCK_BARS = 16

_AXES = ("timestamp", "symbol")


def sorted_panel(data: xr.Dataset) -> xr.Dataset:
    """Return ``data`` with both axes increasing, sorting only when they are not.

    Examples
    --------
    >>> sorted_panel(panel.isel(symbol=[2, 0, 1])).symbol.values.tolist()
    ['S0', 'S1', 'S2']
    """
    if all(data.indexes[axis].is_monotonic_increasing for axis in _AXES):
        return data
    return data.sortby(list(_AXES))


def _array_dtype(data: xr.Dataset, variables: list[str], dtype) -> np.dtype:
    """``dtype``, or the variables' common type when it is None."""
    if dtype is None:
        return np.result_type(*(data[name].dtype for name in variables))
    return np.dtype(dtype)


#: The default type a model reads its features as (``BaseModel.feature_dtype``):
#: its fit, its evaluation and its predictions all read one array of the
#: model's one type. A torch model storing its panel in float16 reads float16.
FEATURE_DTYPE = np.dtype(np.float32)


def _refuse_overflow(source: np.ndarray, cast: np.ndarray, name: str) -> None:
    """Raise when ``cast`` turned a finite value of ``source`` into an infinity.

    Raises
    ------
    ValueError
        If any value finite in ``source`` is infinite in ``cast``, as a value
        beyond ±65504 becomes in float16; the message names the variable.
    """
    overflow = int(np.count_nonzero(np.isfinite(source) & ~np.isfinite(cast)))
    if overflow:
        raise ValueError(
            f"{overflow} value(s) of {name!r} are finite but too large for {cast.dtype}; "
            f"read the features in a wider type (a torch model's panel_dtype='float32') "
            f"or rescale the factor"
        )


def _below_feature_dtype(source: np.dtype, target: np.dtype) -> bool:
    """Whether a cast of ``source`` to ``target`` goes below ``FEATURE_DTYPE`` (to float16).

    Such a cast can overflow, so it is checked. A cast to ``FEATURE_DTYPE``
    itself is not: the default float32 read behaves as it always has.
    """
    return target.kind == "f" and target.itemsize < FEATURE_DTYPE.itemsize <= source.itemsize


def as_feature_dtype(data: xr.Dataset, dtype: "np.typing.DTypeLike" = FEATURE_DTYPE) -> xr.Dataset:
    """Return ``data`` with every variable cast to ``dtype``, ``FEATURE_DTYPE`` by default.

    A variable already of that type is kept as it is, not copied. Every
    variable is cast, integer ones included: a model reads its features in
    its ``feature_dtype`` whatever their type, each value rounded once,
    straight from its own type. A cast below float32 (to float16) is checked
    for overflow one variable at a time.

    Parameters
    ----------
    data : xr.Dataset
        A feature panel.
    dtype : numpy dtype, optional
        The type to cast to; ``FEATURE_DTYPE`` (float32) by default.

    Returns
    -------
    xr.Dataset
        ``data`` with every variable of ``dtype``; ``data`` itself when every
        variable already is.

    Raises
    ------
    ValueError
        If a finite value is too large for ``dtype`` (beyond ±65504 for
        float16).

    Examples
    --------
    >>> wide = panel.assign(f_a=panel["f_a"].astype(np.float64))
    >>> as_feature_dtype(wide)["f_a"].dtype
    dtype('float32')
    >>> as_feature_dtype(wide, np.float16)["f_a"].dtype
    dtype('float16')
    """
    dtype = np.dtype(dtype)
    cast = {}
    for name in data.data_vars:
        variable = data[name]
        if variable.dtype == dtype:
            continue
        if _below_feature_dtype(variable.dtype, dtype):
            values = variable.values
            narrowed = values.astype(dtype)
            _refuse_overflow(values, narrowed, str(name))
            cast[name] = variable.copy(data=narrowed)
        else:
            cast[name] = variable.astype(dtype)
    return data.assign(cast) if cast else data


def to_array(data: xr.Dataset, variables: list[str], dtype: "np.typing.DTypeLike | None" = None) -> np.ndarray:
    """Convert a panel to a ``[num_times, num_symbols, len(variables)]`` array.

    Both axes are sorted, and the last axis follows the order of
    ``variables`` exactly (not alphabetical order), so ``x[..., i]`` is
    always ``variables[i]``.

    Parameters
    ----------
    data : xr.Dataset
        A dataset indexed by ``(timestamp, symbol)``.
    variables : list[str]
        The data variables to stack, in the wanted order.
    dtype : numpy dtype, optional
        The array's type; by default the variables' common type.

    Returns
    -------
    np.ndarray
        The stacked variables.

    Raises
    ------
    ValueError
        If ``dtype`` is below float32 (float16) and a finite value is too
        large for it (checked a block of bars at a time).

    Examples
    --------
    >>> to_array(panel, ["f_b", "f_a"]).shape
    (40, 3, 2)
    >>> to_array(panel, ["f_b", "f_a"], dtype=np.float32).dtype
    dtype('float32')
    """
    # One preallocated array: the peak is the result itself, not the several
    # full copies of a stacked DataArray.
    data = sorted_panel(data[variables])
    dtype = _array_dtype(data, variables, dtype)
    num_times = data.sizes["timestamp"]
    out = np.empty((num_times, data.sizes["symbol"], len(variables)), dtype=dtype)
    for first in range(0, len(variables), GROUP_VARIABLES):
        names = variables[first:first + GROUP_VARIABLES]
        columns = [data[name].transpose(*_AXES).values for name in names]
        checked = [_below_feature_dtype(column.dtype, dtype) for column in columns]
        for start in range(0, num_times, BLOCK_BARS):
            stop = min(num_times, start + BLOCK_BARS)
            for i, column in enumerate(columns):
                if checked[i]:
                    block = column[start:stop]
                    out[start:stop, :, first + i] = block
                    _refuse_overflow(block, out[start:stop, :, first + i], str(names[i]))
                else:
                    out[start:stop, :, first + i] = column[start:stop]
    return out


class CollectedPanel:
    """A model's collected panel: the data sorted on both axes, its read record, its arrays.

    Parameters
    ----------
    data : xr.Dataset
        The factors and labels on ``(timestamp, symbol)``; sorted on both
        axes here when it is not.
    record : dict, optional
        What the reads recorded, keyed by component path within the model
        that collected (``BaseModel.training_record``).
    source : dict, optional
        What the panel was read for: the factor and label configs and the
        dates (``BaseModel.use_panel`` compares it).

    Examples
    --------
    >>> panel = model.collect().collected_panel
    >>> panel.data.sizes
    Frozen({'timestamp': 40, 'symbol': 3})
    >>> panel.array(["f_a"])[0] is panel.array(["f_a"])[0]
    True
    """

    @classmethod
    def assemble(
        cls,
        features: xr.Dataset,
        labels: xr.Dataset,
        source: dict | None = None,
        feature_names: list[str] | None = None,
        dtype: "np.typing.DTypeLike" = FEATURE_DTYPE,
    ) -> "CollectedPanel":
        """Join a model's features and labels into one panel on the features' bars and symbols.

        The features decide the axes: the labels are reindexed onto them, so
        a label's symbol or bar without features is dropped (nothing can be
        fitted or predicted on it) and a feature's warm-up bar has NaN
        labels. The features are held once: they are stacked into one
        ``[T, S, F]`` array of ``dtype`` in ``feature_names`` order, the
        array ``array(feature_names, dtype)`` returns, and the panel's
        feature variables are views of it.

        Parameters
        ----------
        features : xr.Dataset
            The factor panels on ``(timestamp, symbol)``.
        labels : xr.Dataset
            The label panels on ``(timestamp, symbol)``.
        source : dict, optional
            What the panel was read for (see the class).
        feature_names : list[str], optional
            The feature variables in the order the model reads them; the
            order of ``features``' variables when not given.
        dtype : numpy dtype, optional
            The features' type, the reading model's ``feature_dtype``;
            ``FEATURE_DTYPE`` (float32) by default.

        Returns
        -------
        CollectedPanel
            The panel, with an empty record.

        Raises
        ------
        ValueError
            If no label symbol is on the features' symbol axis (an integer
            permaticker against a text one), which would leave every label
            NaN, or a feature is too large for ``dtype`` (see ``to_array``).

        Examples
        --------
        With ``features`` on symbols ``A`` and ``B`` and ``labels`` on ``A``,
        ``B`` and ``C``:

        >>> CollectedPanel.assemble(features, labels).data["symbol"].values.tolist()
        ['A', 'B']
        """
        features = sorted_panel(features)
        symbols = features.indexes["symbol"]
        label_symbols = labels.indexes["symbol"]
        if len(label_symbols) and len(symbols) and not label_symbols.isin(symbols).any():
            raise ValueError(
                f"CollectedPanel: none of the {len(label_symbols)} label symbol(s) is on the "
                f"features' symbol axis (labels {label_symbols[:3].tolist()} of type "
                f"{label_symbols.dtype} against {symbols[:3].tolist()} of type {symbols.dtype}); "
                f"every label would be NaN. Check that both are on the same axis."
            )
        names = list(features.data_vars) if feature_names is None else list(feature_names)
        dtype = np.dtype(dtype)
        x = to_array(features, names, dtype)
        dims = ("timestamp", "symbol")
        data = xr.Dataset(
            {name: (dims, x[..., i]) for i, name in enumerate(names)},
            coords={axis: features.indexes[axis] for axis in dims},
        )
        labels = labels.reindex(timestamp=features.indexes["timestamp"], symbol=symbols)
        panel = cls(data.assign(labels.data_vars), source=source)
        panel._arrays[(tuple(names), dtype.str)] = x
        return panel

    def __init__(self, data: xr.Dataset, record: dict | None = None, source: dict | None = None):
        """Hold ``data`` sorted; see the class docstring for parameters."""
        self._data = sorted_panel(data)
        self._record = dict(record or {})
        self._source = source
        self._arrays: dict = {}

    @property
    def data(self) -> xr.Dataset:
        """The panel on ``(timestamp, symbol)``, both axes increasing.

        Examples
        --------
        >>> panel.data.indexes["symbol"].is_monotonic_increasing
        True
        """
        return self._data

    @property
    def record(self) -> dict:
        """What the reads recorded, keyed by component path (a copy).

        Examples
        --------
        >>> sorted(panel.record)
        ['factors.0.dataset', 'labels.0.factor.dataset']
        """
        return dict(self._record)

    @property
    def source(self) -> dict | None:
        """What the panel was read for: factor and label configs and dates.

        Examples
        --------
        >>> sorted(panel.source)
        ['end_date', 'factors', 'labels', 'start_date']
        """
        return self._source

    def with_record(self, record: dict) -> "CollectedPanel":
        """Return this panel with ``record`` as its read record, its data and arrays shared.

        Parameters
        ----------
        record : dict
            What the reads recorded, keyed by component path.

        Returns
        -------
        CollectedPanel
            A new panel on the same data, source and built arrays.

        Examples
        --------
        >>> panel.with_record({"factors.0.dataset": "r"}).record
        {'factors.0.dataset': 'r'}
        """
        panel = CollectedPanel(self._data, record, self._source)
        panel._arrays = self._arrays
        return panel

    def array(
        self, names: list[str], dtype: "np.typing.DTypeLike | None" = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``names`` as one ``[T, S, len(names)]`` array, with its timestamps and symbols.

        Built by ``to_array`` once per variable list and resolved type, then
        kept with the panel: every caller gets the same array, and none may
        write into it.

        Parameters
        ----------
        names : list[str]
            The variables, in order.
        dtype : numpy dtype, optional
            The array's type; by default the variables' common type.

        Returns
        -------
        tuple
            The array, the panel's timestamps and its symbols.

        Examples
        --------
        >>> x, timestamps, symbols = panel.array(["f_a", "f_b"], np.float32)
        >>> x.shape, len(timestamps), len(symbols)
        ((40, 3, 2), 40, 3)
        """
        names = list(names)
        resolved = _array_dtype(self._data, names, dtype)
        key = (tuple(names), resolved.str)
        if key not in self._arrays:
            self._arrays[key] = to_array(self._data, names, resolved)
        return self._arrays[key], self._data.timestamp.values, self._data.symbol.values
