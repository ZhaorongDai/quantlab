import xarray as xr
import polars as pl
from typing import Literal, Self
from abc import abstractmethod, ABC


class DataBackend(ABC):

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def data(self):
        try:
            return self._data
        except AttributeError:
            raise AttributeError("Please cal 'read' or 'to_internal' first.")

    @data.setter
    def data(self, data):
        self._data = data

    @abstractmethod
    def get_xarray_dataset(self, indexes: list[str]) -> xr.Dataset: ...

    @abstractmethod
    def get_lazyframe(self) -> pl.LazyFrame: ...

    @abstractmethod
    def read(self, path: str, **kwargs) -> Self: ...

    @abstractmethod
    def write(self, path: str, **kwargs) -> Self: ...

    @abstractmethod
    def to_internal(self, data) -> Self: ...

    @abstractmethod
    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self: ...

    @abstractmethod
    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self: ...


class ModelBackend(ABC):

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def model(self):
        try:
            return self._model
        except AttributeError:
            raise AttributeError("Please call 'read' or 'to_internal' first.")

    @model.setter
    def model(self, model):
        self._model = model

    @abstractmethod
    def get_model(self): ...

    @abstractmethod
    def read(self, path: str, **kwargs) -> Self: ...

    @abstractmethod
    def write(self, path: str, **kwargs) -> Self: ...

    @abstractmethod
    def to_internal(self, model) -> Self: ...
