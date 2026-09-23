"""Abstract storage contracts for data and for trained models.

This module defines the two interfaces the rest of the pipeline uses to keep
"where something is stored" separate from "what it means". ``DataBackend`` is
the contract a dataset or factor object talks to when it reads, writes,
filters or converts its panel; concrete implementations live in
``quantlab/backend.py`` (``XrBackend`` for Zarr, ``PlBackend`` for Parquet).
``ModelBackend`` is the much smaller counterpart for persisting a fitted
model object. Neither contract assumes a particular schema, so a backend can
be swapped without touching the layers above it. See ``docs/backend.md``.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Self

import polars as pl
import xarray as xr


class DataBackend(ABC):
    """Contract for a storage medium that holds one tabular or panel dataset.

    A backend owns a single in-memory object, exposed as ``data``, and knows
    how to move it to and from a path, narrow it in place, and convert it to
    the two exchange formats used across the pipeline: an ``xarray.Dataset``
    and a ``polars.LazyFrame``. Every mutating method returns ``self`` so
    calls can be chained.

    Two details of the contract are easy to get wrong when implementing it.
    ``filter_by_date`` and ``filter_by_symbol`` narrow ``data`` in place, and
    every object sharing the backend instance sees the result. ``head`` must
    do the opposite: open the store at the given path, read at most ``n``
    rows without materialising the whole store, and leave ``data`` untouched.

    Example:
        >>> backend = XrBackend().read("prices.zarr")
        >>> panel = backend.filter_by_date("timestamp", "2022-01-01",
        ...                                "2022-12-31")
        >>> ds = panel.get_xarray_dataset(["timestamp", "symbol"])
    """

    def __repr__(self) -> str:
        """Return the class name followed by empty parentheses."""
        return f"{self.__class__.__name__}()"

    @property
    def data(self):
        """The object this backend currently holds.

        Raises:
            AttributeError: If nothing has been loaded yet. Accessing an
                unpopulated backend fails immediately rather than returning
                an empty dataset that downstream code would mistake for
                "no data in this range".
        """
        try:
            return self._data
        except AttributeError:
            raise AttributeError("Please cal 'read' or 'to_internal' first.")

    @data.setter
    def data(self, data):
        """Replace the held object."""
        self._data = data

    @abstractmethod
    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        """Return the held data as an ``xarray.Dataset``.

        Args:
            indexes: The dimensions the returned dataset must be indexed by,
                in order. ``None`` asks for no particular shape.
        """
        ...

    @abstractmethod
    def get_lazyframe(self) -> pl.LazyFrame:
        """Return the held data as a ``polars.LazyFrame``."""
        ...

    @abstractmethod
    def head(self, path: str, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows from the store at ``path``.

        Implementations must not materialise the whole store, must not touch
        ``data``, and must raise ``FileNotFoundError`` when ``path`` does not
        exist rather than deferring the failure to a later collect.
        """
        ...

    @abstractmethod
    def read(self, path: str, **kwargs) -> Self:
        """Load the store at ``path`` into ``data`` and return ``self``."""
        ...

    @abstractmethod
    def write(self, path: str, **kwargs) -> Self:
        """Persist ``data`` to ``path`` and return ``self``."""
        ...

    @abstractmethod
    def to_internal(self, data) -> Self:
        """Adopt an in-memory object as ``data``, bypassing disk."""
        ...

    @abstractmethod
    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        """Narrow ``data`` in place to ``start_date..end_date`` on ``col``."""
        ...

    @abstractmethod
    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        """Narrow ``data`` in place to the rows whose ``col`` is in ``symbols``."""
        ...


class ModelBackend(ABC):
    """Contract for a storage medium that holds one fitted model object.

    The model-side twin of ``DataBackend``: it knows nothing about
    dimensions or coordinates, only how to load a model from a path, save it
    back, or adopt one already in memory. The concrete implementation used
    by the tree-model layer is ``quantlab/ml_model/backend.py:MlBackend``.
    """

    def __repr__(self) -> str:
        """Return the class name followed by empty parentheses."""
        return f"{self.__class__.__name__}()"

    @property
    def model(self):
        """The model object this backend currently holds.

        Raises:
            AttributeError: If nothing has been loaded yet.
        """
        try:
            return self._model
        except AttributeError:
            raise AttributeError("Please call 'read' or 'to_internal' first.")

    @model.setter
    def model(self, model):
        """Replace the held model."""
        self._model = model

    @abstractmethod
    def get_model(self):
        """Return the held model object."""
        ...

    @abstractmethod
    def read(self, path: str, **kwargs) -> Self:
        """Load the model stored at ``path`` and return ``self``."""
        ...

    @abstractmethod
    def write(self, path: str, **kwargs) -> Self:
        """Persist the held model to ``path`` and return ``self``."""
        ...

    @abstractmethod
    def to_internal(self, model) -> Self:
        """Adopt an in-memory model object, bypassing disk."""
        ...
