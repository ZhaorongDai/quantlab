import xarray as xr
import polars as pl
from typing import Literal, Optional, Self
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
    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        """转成全流水线唯一的层间交换格式：一个 `xr.Dataset`。

        **`indexes` 就是结果的索引维度**，而且实现必须真的照做——返回的
        `Dataset` 的维度恰好是 `indexes`，顺序也照给定的来，铺在其他维度上的
        数据变量要丢掉。全项目绝大多数调用点传的是 `["timestamp", "symbol"]`，
        因为「时间戳 + 标的」是 CLAUDE.md 里的硬约束，不是本方法的可选项；但
        「传了就得算数」是本方法的契约，不是那条约束的推论。

        `indexes=None` 表示「不做形状要求，原样给我」。它存在是因为多数调用点
        只想拿到面板本身，而不是想断言它的形状。

        实现方要注意的两点：

        - 请求了当前数据没有的维度必须**报错**，并把实际有的维度列出来。静默
          返回一个形状不符的 `Dataset` 会让错误在三层之外才现形。
        - `XrBackend` 在 `indexes=None` 时返回的是它持有的那个对象本身，不是
          副本；调用方不应就地改写返回值。

        历史：`XrBackend` 的实现曾经整个忽略这个参数（函数体就是
        `return self.data`），于是 `BaseDataset.time_interval` 这种依赖「只要
        时间轴」的调用点在该后端下根本跑不通（2026-09-07 修复）。
        """
        ...

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
