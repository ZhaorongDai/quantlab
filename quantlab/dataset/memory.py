"""A market dataset held in memory, built from a caller's frame or panel.

``FrameDataset`` is the one seam a caller's own data enters the library through (ADR
0011). It is a real ``MarketDataset``: a factor, label or backtester that reads a dataset
reads it through the same public requests (``panel``, ``bar_before``, ``head``,
``to_kunquant``), answered from the panel it holds instead of from a Zarr store. The frame
is converted by ``quantlab.utils.frame.to_panel``, the same rules ``quantlab.api`` applies.
"""

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.backend import XrBackend
from quantlab.base.config import FrameDatasetConfig
from quantlab.base.data import MarketDataset
from quantlab.utils.date_range import as_label, check_range
from quantlab.utils.frame import to_panel


class FrameDataset(MarketDataset):
    """A market dataset whose panel is held in memory.

    Built from a long pandas or polars frame (one row per ``timestamp`` and ``symbol``) or
    from an ``xarray.Dataset`` panel. The input rules of ``quantlab.utils.frame`` apply: a
    pandas ``(timestamp, symbol)`` MultiIndex is reset, ``columns`` renames first, a
    repeated ``(timestamp, symbol)`` raises, timezone-aware timestamps become naive UTC,
    symbols become ``str`` and missing cells become NaN. Every other column becomes a
    variable under its own name, and ``to_kunquant`` exports the variables as they are
    named, so name them as the factor reading them expects (``adjClose`` for the stock
    factors, ``close`` for the crypto ones).

    Nothing is read from or written to disk. There are no raw files, so
    ``from_raw_data``, ``from_raw_data_chunked`` and ``update`` refuse, as does ``save``;
    a stream-mode factor refuses it too, since a stream is fed live bars rather than a
    held panel. ``resample()`` returns a dataset holding the resampled panel.

    Parameters
    ----------
    data : pandas.DataFrame, polars.DataFrame or xarray.Dataset
        The bars, as a long frame or a panel on ``(timestamp, symbol)``.
    columns : mapping of str to str, optional
        Renames the frame's columns before conversion, ``{caller_name: name}``.

    Raises
    ------
    TypeError
        If ``data`` is not a pandas or polars frame or an xarray panel.
    ValueError
        If ``timestamp`` or ``symbol`` is missing after renaming, ``columns`` names an
        absent column, or a ``(timestamp, symbol)`` pair repeats.

    Examples
    --------
    >>> import pandas as pd
    >>> from quantlab.dataset.memory import FrameDataset
    >>> frame = pd.DataFrame({
    ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
    ...     "symbol": ["AAA", "BBB", "AAA"],
    ...     "close": [10.0, 20.0, 11.0],
    ... })
    >>> ds = FrameDataset(frame)
    >>> ds.panel("2024-01-02", "2024-01-03")["close"].values
    array([[10., 20.],
           [11., nan]])
    >>> ds.bar_before("2024-01-03", 1)
    Timestamp('2024-01-02 00:00:00')
    """

    # Narrower type annotation for readers and type checkers only.
    config: FrameDatasetConfig

    #: The config class of this dataset.
    config_cls = FrameDatasetConfig

    def __init__(self, data, *, columns: Mapping[str, str] | None = None):
        """Convert and hold ``data``; see the class docstring for parameters."""
        panel = to_panel(data, columns=columns, purpose="FrameDataset").load()
        super().__init__(FrameDatasetConfig())
        self.data_backend.to_internal(panel)

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a ``FrameDataset`` with an equal config and panel.

        The panel is compared too, since two frames give the same config.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> ds == FrameDataset(frame.copy()), ds == FrameDataset(frame.assign(close=0.0))
        (True, False)
        """
        if type(other) is not type(self):
            return NotImplemented
        return self.config == other.config and self._held().equals(other._held())

    def _held(self) -> xr.Dataset:
        """Return the panel this dataset holds."""
        return self.data_backend.get_xarray_dataset(["timestamp", "symbol"])

    def copy(self) -> Self:
        """Return a copy with its own config, holding the same panel.

        The panel is shared, not duplicated: no method of this dataset changes it.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> other = ds.copy()
        >>> other == ds, other.config is ds.config
        (True, False)
        """
        other = super().copy()
        other.data_backend.to_internal(self._held())
        return other

    def resample(self, freq: str, how: Mapping[str, str] | str) -> Self:
        """Return a dataset holding this panel resampled onto ``freq``, in memory.

        The bars are cut and aggregated as on a Zarr-backed dataset: UTC-clock buckets
        labelled at their start, one ``ResampleMethod`` per variable, NaN cells skipped.
        Nothing is written: there is no sibling store, and ``store_path`` stays ``None``.
        The copy is still a ``FrameDataset``, so it refuses ``from_raw_data``,
        ``from_raw_data_chunked``, ``update``, ``save`` and stream mode. This dataset is
        not changed.

        Parameters
        ----------
        freq : str
            A ``ResampleFrequency`` token, coarser than the held bars.
        how : mapping of str to str, or str
            One ``ResampleMethod`` for every variable, or a ``{variable: method}``
            mapping naming every variable.

        Returns
        -------
        FrameDataset
            A new dataset whose config carries ``resample_freq`` and ``resample_how``.

        Raises
        ------
        ValueError
            If ``freq`` or a method is not a known token, ``freq`` is not coarser than
            the held bars, or ``how`` does not name every variable.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.date_range("2024-01-02", periods=4, freq="12h").repeat(2),
        ...     "symbol": ["AAA", "BBB"] * 4,
        ...     "close": [10.0, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0],
        ...     "volume": [1.0] * 8,
        ... })
        >>> daily = FrameDataset(frame).resample("1d", {"close": "last", "volume": "sum"})
        >>> panel = daily.panel("2024-01-02", "2024-01-03")
        >>> panel["close"].values
        array([[11., 21.],
               [13., 23.]])
        >>> panel["volume"].values
        array([[2., 2.],
               [2., 2.]])
        >>> daily.store_path is None
        True
        >>> daily.save()
        Traceback (most recent call last):
        ValueError: FrameDataset.save(): the panel is held in memory, ...
        """
        return super().resample(freq, how)

    def panel(
        self,
        start: "str | pd.Timestamp",
        end: "str | pd.Timestamp",
        symbols: Sequence | None = None,
    ) -> xr.Dataset:
        """Return the held panel from ``start`` to ``end``, both inclusive.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return. A date-only ``end`` includes every bar of that day.
        symbols : sequence, optional
            Symbols to keep, in the order given. ``None`` keeps every symbol.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.
        KeyError
            If a requested symbol is not held.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> dict(ds.panel("2024-01-03", "2024-01-03", symbols=["BBB"]).sizes)
        {'timestamp': 1, 'symbol': 1}
        """
        check_range(start, end, f"{self.class_name}.panel()")
        data = self._held().sel(timestamp=slice(as_label(start), as_label(end)))
        if symbols is not None:
            data = data.sel(symbol=list(symbols))
        return XrBackend().to_internal(data).get_xarray_dataset(["timestamp", "symbol"])

    def _calendar(self) -> pd.DatetimeIndex:
        """Return the held panel's timestamps."""
        return pd.DatetimeIndex(self._held()["timestamp"].values)

    def _calendar_source(self) -> str:
        """Return a description of what ``_calendar`` reads, for error messages."""
        return "the panel held in memory"

    def head(self, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows of the held panel as a ``LazyFrame``.

        Parameters
        ----------
        n : int
            Maximum number of rows.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> ds.head(2).collect().columns
        ['timestamp', 'symbol', 'close']
        """
        held = self._held()
        bounded = held.isel(timestamp=slice(0, n), symbol=slice(0, n))
        return XrBackend().to_internal(bounded).get_lazyframe().head(n)

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns under their own names."""
        return self._kunquant_arrays(data, data_columns)

    def _refuse(self, method: str):
        """Raise: a held panel has no raw files and no store."""
        raise ValueError(
            f"{self.class_name}.{method}(): the panel is held in memory, handed over at "
            f"construction; there are no raw files to build it from and no store to write. "
            f"Build a new FrameDataset from updated data instead."
        )

    def save(self, **kwargs):
        """Refuse: a held panel has no store.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> ds.save()
        Traceback (most recent call last):
        ValueError: FrameDataset.save(): the panel is held in memory, ...
        """
        self._refuse("save")

    def from_raw_data(self) -> Self:
        """Refuse: a held panel has no raw files.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> ds.from_raw_data()
        Traceback (most recent call last):
        ValueError: FrameDataset.from_raw_data(): the panel is held in memory, ...
        """
        self._refuse("from_raw_data")

    def from_raw_data_chunked(self, *args, **kwargs) -> Self:
        """Refuse: a held panel has no raw files.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> ds.from_raw_data_chunked()
        Traceback (most recent call last):
        ValueError: FrameDataset.from_raw_data_chunked(): the panel is held in memory, ...
        """
        self._refuse("from_raw_data_chunked")

    def update(self, *args, **kwargs) -> Self:
        """Refuse: a held panel has no raw files.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> ds.update()
        Traceback (most recent call last):
        ValueError: FrameDataset.update(): the panel is held in memory, ...
        """
        self._refuse("update")

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Refuse: a held panel has no raw files."""
        self._refuse("_raw_data_to_xr")

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None) -> xr.Dataset:
        """Refuse: a held panel has no raw files."""
        self._refuse("_raw_data_to_xr_window")
