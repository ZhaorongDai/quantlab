import xarray as xr
import polars as pl
from typing import Literal, Self
from abc import abstractmethod, ABC


class DataBackend(ABC):
    """数据存储后端的抽象基类，把"数据存在哪里"和"数据是什么"彻底分开。

    每个子类实现一种存储介质（Zarr、Parquet），上层的数据集与因子只持有一个后端
    实例、只调用这里声明的方法，因此换一种存储介质不需要改动上层任何一行代码。
    """

    def __repr__(self) -> str:
        """给出后端的简短标识：日志里只需要认出这是哪一类后端，不需要看到数据本身。

        Returns:
            str: 只含类名的标识字符串。
        """
        return f"{self.__class__.__name__}()"

    @property
    def data(self):
        """已经载入内存的数据；还没读取就访问会直接报错，而不是返回一个空值。

        宁可报错也不给空值：一个空数据集会被下游当成"这段时间确实没有行情"继续算
        下去，错误要跑到很远的地方才暴露出来，那时已经很难追回源头。

        Returns:
            object: 后端当前持有的数据，具体类型由子类的存储介质决定——Zarr 后端
                持有 xarray 数据集，Parquet 后端持有 polars 惰性帧。

        Raises:
            AttributeError: 尚未读取数据、也未从内存接管数据时。
        """
        try:
            return self._data
        except AttributeError:
            raise AttributeError("Please cal 'read' or 'to_internal' first.")

    @data.setter
    def data(self, data):
        """直接接管一份已经在内存里的数据，让不经过磁盘的构造路径也能用这个后端。

        Args:
            data (object): 要接管的数据，类型需与子类的存储介质相符。
        """
        self._data = data

    @abstractmethod
    def get_xarray_dataset(self, indexes: list[str]) -> xr.Dataset:
        """把后端数据转成规范形状的 xarray 数据集，作为各层之间唯一的交换格式。

        Args:
            indexes (list[str]): 用作坐标轴的维度名。流水线固定使用时间戳与标的
                这两个维度，这是全项目的硬约束，不是本方法的可选项。

        Returns:
            xr.Dataset: 以 indexes 为坐标的数据集。
        """
        ...

    @abstractmethod
    def get_lazyframe(self) -> pl.LazyFrame:
        """以惰性帧的形式暴露数据，让 Polars 因子能在不落地中间结果的前提下计算。

        Returns:
            pl.LazyFrame: 覆盖存储全量数据的惰性帧。
        """
        ...

    @abstractmethod
    def head(self, path: str, n: int) -> pl.LazyFrame:
        """有界读取：最多取 n 行，用于在不加载全量数据的前提下探查存储的结构。

        A BOUNDED read: at most `n` rows of the store at `path`.

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

        Args:
            path (str): 要探查的存储根路径；本方法自己打开它，不依赖任何先前的
                读取，因此可以在数据集尚未读取时调用。
            n (int): 最多返回的行数。

        Returns:
            pl.LazyFrame: 至多 n 行的探查结果，列名与真实数据类型都与全量读取
                一致，因此可以拿真实表达式在上面试算。

        Raises:
            FileNotFoundError: 路径下没有存储时，在调用当下抛出。
        """
        ...

    @abstractmethod
    def read(self, path: str, **kwargs) -> Self:
        """把磁盘上的存储读入内存，返回自身，方便与后续的收窄操作串成一条链。

        Args:
            path (str): 存储根路径。
            **kwargs (object): 透传给底层存储引擎的读取参数。

        Returns:
            Self: 已载入数据的后端自身。
        """
        ...

    @abstractmethod
    def write(self, path: str, **kwargs) -> Self:
        """把内存中的数据落盘，返回自身，方便继续链式调用。

        Args:
            path (str): 写入的目标路径。
            **kwargs (object): 透传给底层存储引擎的写入参数。

        Returns:
            Self: 完成写入后的后端自身。
        """
        ...

    @abstractmethod
    def to_internal(self, data) -> Self:
        """接管一份外部数据并转成后端的内部表示，让数据不经磁盘就进入流水线。

        Args:
            data (object): 外部数据，类型需与子类的存储介质相符。

        Returns:
            Self: 已持有该数据的后端自身。
        """
        ...

    @abstractmethod
    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        """按时间区间就地收窄已载入的数据，返回自身，方便继续链式调用。

        收窄是就地发生的：所有共享这一个后端实例的调用方都会看到被收窄后的数据。
        只想看一眼存储而不想影响别人，应当用有界读取而不是这里。

        Args:
            col (str): 时间维度的名字。
            start_date (str): 区间起点，含当天。
            end_date (str): 区间终点，含当天。

        Returns:
            Self: 已收窄到该时间区间的后端自身。
        """
        ...

    @abstractmethod
    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        """按标的清单就地收窄已载入的数据，返回自身，方便继续链式调用。

        与按时间收窄一样是就地生效的，同一个后端实例的其他持有者也会受影响。

        Args:
            col (str): 标的维度的名字。
            symbols (tuple[str, ...]): 需要保留的标的清单。

        Returns:
            Self: 只剩下这些标的的后端自身。
        """
        ...


class ModelBackend(ABC):
    """模型持久化后端的抽象基类，与数据后端对称，把"模型存在哪里"独立出来。

    训练层只负责产出模型对象，至于它是被序列化成权重文件还是别的什么格式，由子类
    决定；因此换一种持久化方式不需要改训练代码。
    """

    def __repr__(self) -> str:
        """给出后端的简短标识：日志里只需要认出这是哪一类后端，不需要看到模型本身。

        Returns:
            str: 只含类名的标识字符串。
        """
        return f"{self.__class__.__name__}()"

    @property
    def model(self):
        """已经载入内存的模型；还没读取就访问会直接报错，而不是返回一个空对象。

        Returns:
            object: 后端当前持有的模型对象，具体类型取决于训练时用的框架。

        Raises:
            AttributeError: 尚未读取模型、也未从内存接管模型时。
        """
        try:
            return self._model
        except AttributeError:
            raise AttributeError("Please call 'read' or 'to_internal' first.")

    @model.setter
    def model(self, model):
        """直接接管一个内存中的模型对象，让刚训练完的模型无需落盘即可继续使用。

        Args:
            model (object): 要接管的模型对象。
        """
        self._model = model

    @abstractmethod
    def get_model(self):
        """取出后端持有的模型，交给推理或继续训练的调用方使用。

        Returns:
            object: 后端持有的模型对象。
        """
        ...

    @abstractmethod
    def read(self, path: str, **kwargs) -> Self:
        """从磁盘载入模型，返回自身，方便继续链式调用。

        Args:
            path (str): 模型文件路径。
            **kwargs (object): 透传给底层序列化库的读取参数。

        Returns:
            Self: 已载入模型的后端自身。
        """
        ...

    @abstractmethod
    def write(self, path: str, **kwargs) -> Self:
        """把内存中的模型落盘，返回自身，方便继续链式调用。

        Args:
            path (str): 写入的目标路径。
            **kwargs (object): 透传给底层序列化库的写入参数。

        Returns:
            Self: 完成写入后的后端自身。
        """
        ...

    @abstractmethod
    def to_internal(self, model) -> Self:
        """接管一个外部模型对象并转成后端的内部表示，跳过磁盘直接可用。

        Args:
            model (object): 外部模型对象。

        Returns:
            Self: 已持有该模型的后端自身。
        """
        ...
