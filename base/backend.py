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
    def head(self, path: str, n: int) -> pl.LazyFrame:
        """A BOUNDED read: at most `n` rows of the store at `path`.

        The bounded twin of `get_lazyframe()`. The returned lazyframe carries
        the same column names and the same dtypes `get_lazyframe()` would
        return -- real dtypes matter, because callers run real expressions
        over this probe to learn what those expressions produce.

        **Takes the path and OPENS the store itself**, path first, mirroring
        `read(path, **kwargs)` -- it does not read `self.data`. That is what
        makes it usable before anything has read the dataset, and it is not a
        convenience: reaching the store through `read()` meant the caller had
        to call `BaseDataset.read()`, which runs `_filter()` and narrows
        `data_backend.data` IN PLACE. `XrBackend.read()`'s cache early-return
        then made that narrowing survive every later read, so a probe caller
        silently truncated the shared dataset (RV-01, `03-VERIFICATION.md`).
        An implementation that goes back to `self.data` reintroduces it.

        Three obligations on any implementation:

        - It must NOT materialize the whole store. That is the entire point;
          an implementation that reads everything and slices afterwards
          satisfies the signature and defeats the purpose.
        - It must NOT mutate `self.data` -- nor assign to it at all.
          `filter_by_date`/`filter_by_symbol` on this same interface DO filter
          in place, so an implementation written by analogy with them would
          silently truncate the store its caller shares with everything else
          holding that backend.
        - It must raise `FileNotFoundError` for an absent store, at the call
          rather than at `.collect()` time, exactly as `read()` does.

        Abstract rather than a `limit=` keyword on `get_lazyframe()` on
        purpose: ABC enforcement makes a backend that omits the bounded read
        impossible to construct, whereas an optional keyword is satisfied by
        plain inheritance and only fails at whichever call site passes it.
        """
        ...

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
