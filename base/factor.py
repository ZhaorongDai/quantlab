from abc import ABC, abstractmethod

# from prefect import task, flow
from typing import Literal, Self

import KunQuant.runner.KunRunner as kr
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Stage import Function

from base.config import BaseFactorConfig, FactorConfig
from dataset.backend import XrBackend
from enums.constant import Date
from utils.timer import Timer


class Factor(ABC):
    """与后端无关的因子共享契约：模型层需要从一个因子那里得到的一切都在这里。

    基类刻意不含任何 KunQuant 概念，这样换一种因子计算后端时，模型层一行都不用改。

    The shared, backend-agnostic factor contract (D-03).

    Everything the model layer needs from a factor lives here: the `config`
    lifecycle, the `XrBackend` storage round-trip (`read`/`save`), the
    `xr.Dataset` feature/label accessors, and the two abstract members
    (`cal`, `_get_factor_names`) each backend implements for itself.

    The base is deliberately free of any KunQuant concept -- no streaming, no
    compiled graph, and no read of the KunQuant-only `mode`/`data_columns`/
    `njobs` config fields. That is what makes a non-KunQuant factor backend
    droppable into `DLConfig.factors` with zero edits to `base/model.py`.
    """

    def __init__(self, config: BaseFactorConfig):
        """构造一个因子实例：先接下配置，再准备好存放因子数据的地方。

        这两步的先后顺序是有意的，不是随手写的，下面的注释说明了颠倒之后会发生
        什么。

        Args:
            config (BaseFactorConfig): 因子配置。
        """
        # Ordering is load-bearing: assigning `self.config` fires the property
        # setter below, which runs before `self.data_backend` exists. No
        # setter-reachable method may read the storage backend.
        self.config = config
        self.data_backend = XrBackend()

    def __repr__(self) -> str:
        """给出因子的标识：调试时最需要知道的是这个因子是拿哪份配置算出来的。

        Returns:
            str: 含类名与配置内容的标识字符串。
        """
        return f"{self.__class__.__name__}(config={self.config})"

    def _auto_filter(self):
        """把已载入的因子数据收窄到配置声明的时间区间与标的范围。

        读取和落盘都要先过这一道：因子计算需要比请求区间更长的历史来预热窗口，
        如果不收窄，多出来的那段预热数据会一路流到模型那里。

        标的清单没有配置时不收窄，因为缺失的标的在面板里本来就是空值，坐标轴仍然
        完整，不需要再切一刀。
        """
        self.data_backend.filter_by_date(
            col="timestamp",
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        if self.config.symbols is not None:
            self.data_backend.filter_by_symbol("symbol", self.config.symbols)

    @property
    def config(self) -> BaseFactorConfig:
        """当前生效的因子配置；它在赋值时已被补全，读到的是补全后的版本。

        Returns:
            BaseFactorConfig: 因子配置对象。
        """
        return self._config

    @config.setter
    def config(self, config: BaseFactorConfig):
        """设置因子配置文件 使用因子配置覆盖数据集配置

        Args:
            config (FactorConfig): 配置类
        """
        self._config = config
        self._config.name = self.import_path

        # 初始化时间
        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

        # 初始化因子名
        self._maybe_resolve_factor_names()

        # 重置数据集配置
        # batch模式下, 数据集实例化时会初始化数据集文件(若不存在), 存在则会读取
        self._reset_dataset_config()

    def _maybe_resolve_factor_names(self) -> None:
        """在配置赋值的当下就把因子名定下来，而不是等到真正开算的时候。

        这是一个可以覆写的接缝：如果某个后端的因子名要等数据读进来才知道，它可以
        把这里改成空实现，改在自己的计算流程里解析。

        Resolve `config.factor_names` eagerly, at config-assignment time.

        The default is exactly the behaviour every KunQuant factor and label
        class has always had: if the caller did not pin an explicit list, ask
        the subclass to enumerate its factor names now.

        Overridable seam: a backend whose factor names can only be known after
        the data is read (a lazyframe schema, say) overrides this to a no-op
        and resolves them inside its own `cal()`.
        """
        if self._config.factor_names is None:
            self._config.factor_names = self._get_factor_names()

    def _reset_dataset_config(self):
        """用因子配置反过来改写数据集配置的取数区间，让窗口有历史可以预热。

        起始日期按因子窗口长度往前推：不多取这一段，请求区间开头的那些因子值会
        因为窗口没填满而算不出来。

        标的轴不动，原因写在下面的注释里。
        """
        # 时间
        start_date = pd.to_datetime(self._config.start_date)
        start_date = start_date - pd.DateOffset(days=self._config.window)
        self._config.dataset.config.start_date = start_date.strftime("%Y-%m-%d")
        self._config.dataset.config.end_date = self._config.end_date

        # symbol
        # 可以不reset, 因为xarray缺失的数据设为null 但是.symbol还是存在
        # self._config.dataset._reset_symbols()

    @property
    def num_symbols(self) -> int:
        """因子面板在标的方向上的宽度，计算与落盘时用它对齐数据形状。

        Returns:
            int: 标的数量。
        """
        return self.config.dataset.num_symbols

    @property
    def num_factors(self) -> int:
        """这个因子类会产出多少列因子，模型层据此确定输入特征的宽度。

        Returns:
            int: 因子列数。
        """
        return len(self.get_factor_names())

    @property
    def import_path(self) -> str:
        """本类的完整导入路径，配置里记下它才能从磁盘上的配置反向重建出对象。

        Returns:
            str: 模块路径加类名。
        """
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def symbols(self) -> list[str]:
        """因子面板的标的轴，与数据集保持一致，避免两边各算各的对不上。

        Returns:
            list[str]: 标的代码列表。
        """
        return self.config.dataset.symbols

    @property
    def class_name(self) -> str:
        """类名；日志、计时器和编译产物的命名都以它为准。

        Returns:
            str: 类名。
        """
        return self.__class__.__name__

    def read(self) -> Self:
        """从磁盘读回已经算好的因子，并立刻收窄到配置请求的区间。

        Returns:
            Self: 已载入并收窄好数据的因子自身，可继续链式调用。
        """
        self.data_backend.read(self.config.file_path)
        self._auto_filter()
        return self

    def save(self, mode: Literal["a", "w"] = "a", **kwargs) -> Self:
        """把算好的因子落盘；写之前先收窄，免得把预热用的历史一起写进去。

        Args:
            mode (Literal["a", "w"]): 追加写还是覆盖写，默认追加。
            **kwargs (object): 透传给存储后端的写入参数。

        Returns:
            Self: 完成写入的因子自身，可继续链式调用。
        """
        with Timer(f"{self.__class__.__name__}: save"):
            self._auto_filter()
            self.data_backend.write(
                self.config.file_path,
                mode=mode,
                **kwargs,
            )
            return self

    def _get_lazyframe(self) -> pl.LazyFrame:
        """把因子面板转成惰性帧，供需要按行处理的内部逻辑使用。

        这是内部用法：层与层之间的交换格式始终是 xarray，惰性帧不跨越因子层边界。

        Returns:
            pl.LazyFrame: 把时间与标的展开成列之后的惰性帧。
        """
        df = self.data_backend.get_xarray_dataset().to_pandas()  # type: ignore
        df = pl.LazyFrame(df.reset_index())
        return df

    def _get_xarray_dataset(self) -> xr.Dataset:
        """取出后端持有的因子面板，作为对外各个取数方法的统一入口。

        Returns:
            xr.Dataset: 以时间与标的为坐标的因子面板。
        """
        return self.data_backend.get_xarray_dataset()  # type: ignore

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """从面板里挑出要当作特征的那几列；基类不作实现，交给具体类自己决定。

        同一个基类同时服务因子类和标签类：因子类只实现取特征这一半，标签类只实现
        取标签那一半，各自没实现的那半留在这里直接报错，而不是返回一份空数据让
        错误悄悄传下去。

        Args:
            data (xr.Dataset): 已读入的因子面板。

        Returns:
            xr.Dataset: 只含特征列的面板。

        Raises:
            NotImplementedError: 具体类没有实现取特征这一半时。
        """
        raise NotImplementedError

    def get_features(self) -> xr.Dataset:
        """对外取特征：模型层只认这个方法，不关心特征是哪个后端算出来的。

        Returns:
            xr.Dataset: 只含特征列的面板。
        """
        return self._get_features(self._get_xarray_dataset())

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        """从面板里挑出要当作标签的那几列；基类不作实现，交给具体类自己决定。

        与取特征互为一半，同样宁可报错也不返回空数据。

        Args:
            data (xr.Dataset): 已读入的面板。

        Returns:
            xr.Dataset: 只含标签列的面板。

        Raises:
            NotImplementedError: 具体类没有实现取标签这一半时。
        """
        raise NotImplementedError

    def get_labels(self) -> xr.Dataset:
        """对外取标签：模型层只认这个方法，与取特征形成对称的一对入口。

        Returns:
            xr.Dataset: 只含标签列的面板。
        """
        return self._get_labels(self._get_xarray_dataset())

    def get_factor_names(self) -> tuple[str, ...]:
        """这个因子会产出哪几列，模型层用它把特征矩阵的列对上号。

        名字在配置赋值时就已确定，因此这里读的是既成事实，不会触发任何计算。

        Returns:
            tuple[str, ...]: 因子名列表。
        """
        return self.config.factor_names

    def get_config(self) -> dict:
        """导出可落盘的完整配置，把数据集配置一并嵌进去。

        嵌套而不是只记一个引用，是为了让这份配置单独存在时也能把整条链路复现出来
        ——只留引用的话，换台机器就找不回当时用的是哪份数据。

        Returns:
            dict: 因子配置字典，其中数据集字段已被替换成数据集自己的配置字典。
        """
        ds_config = self.config.dataset.get_config()
        cfg = self.config.to_dict()
        cfg["dataset"] = ds_config  # type: ignore
        return cfg  # type: ignore

    @abstractmethod
    def _get_factor_names(self) -> tuple[str, ...]:
        """由具体后端给出自己会产出哪些因子名，这是配置补全时要问的问题。

        Returns:
            tuple[str, ...]: 因子名列表。
        """
        ...

    @abstractmethod
    def cal(self) -> Self:
        """真正把因子算出来；每个后端用自己的方式实现，对上层是同一个动作。

        Returns:
            Self: 已持有计算结果的因子自身，可继续链式调用。
        """
        ...


class FactorKunQuant(Factor):
    """KunQuant 因子后端：编译计算图与流式增量计算的全部细节都圈在这个类里。

    把这些留在子类而不是提到基类，是为了让只做批量计算的后端不必被迫实现流式接口。

    The KunQuant factor backend.

    Owns everything compiled-graph- and streaming-specific: `cal()` over
    `kr.runGraph`, the `init_stream()`/`cal_stream()` incremental path, the
    two `cfake.compileit` wrappers, and the three `mode`-branching overrides
    that keep the batch/stream distinction off the shared base (D-07).
    """

    def __init__(self, config: FactorConfig):
        """构造 KunQuant 因子；编译产物与流式上下文都留到真正用到时再建。

        构造时不编译：编译一次的开销不小，而很多场景（比如只是把算好的因子读回来）
        根本不需要编译产物。

        Args:
            config (FactorConfig): KunQuant 因子配置。
        """
        super().__init__(config)
        self._stream_context: kr.StreamContext = None
        self._lib = None
        self._buffer_name_to_id = dict()

    def _auto_filter(self):
        """只有批量模式才收窄数据；流式模式下没有可收窄的历史面板。

        流式计算是逐个时间点推进来的，每次手里只有当前这一条，按区间去切它没有
        意义，还会把唯一的一条数据切掉。
        """
        if self.config.mode == "batch":
            super()._auto_filter()

    @property
    def num_symbols(self) -> int:
        """因子面板的标的数量；两种计算模式下这个数字的来源不同。

        批量模式下标的轴由已读入的数据决定；流式模式还没有数据可读，只能以配置
        声明的标的清单为准，而这个数字要用来开出固定大小的流式缓冲区。

        Returns:
            int: 标的数量。

        Raises:
            ValueError: 配置中的计算模式不是批量或流式时。
        """
        if self.config.mode == "batch":
            return super().num_symbols
        elif self.config.mode == "stream":
            return len(self.config.dataset.config.symbols)
        else:
            raise ValueError(f"mode {self.config.mode} is not supported")

    @property
    def symbols(self) -> list[str]:
        """因子面板的标的清单；与标的数量同理，两种模式的来源不同。

        Returns:
            list[str]: 标的代码列表。

        Raises:
            ValueError: 配置中的计算模式不是批量或流式时。
        """
        if self.config.mode == "batch":
            return super().symbols
        elif self.config.mode == "stream":
            return list(self.config.dataset.config.symbols)
        else:
            raise ValueError(f"mode {self.config.mode} is not supported")

    def init_stream(self) -> Self:
        """建立流式计算上下文，并把每个输入列与因子列的缓冲区句柄一次性查好。

        句柄提前查好并缓存起来，是因为流式计算每来一条数据都要用到它们；按名字
        现查会把这条本该很轻的热路径拖慢。

        Returns:
            Self: 已建好流式上下文的因子自身，可继续链式调用。
        """
        with Timer(f"{self.__class__.__name__}: init stream"):
            lib = self._make_stream()
            modu = lib.getModule(f"{self.__class__.__name__}_stream")  # type: ignore

            executor = kr.createMultiThreadExecutor(self.config.njobs)
            stream = kr.StreamContext(executor, modu, self.num_symbols)

            buffer_name_to_id = {}
            for name in self.config.data_columns:
                buffer_name_to_id[name] = stream.queryBufferHandle(name)
            for name in self.config.factor_names:
                buffer_name_to_id[name] = stream.queryBufferHandle(name)

            self._stream_context = stream
            self._buffer_name_to_id = buffer_name_to_id
            return self

    def _to_xarray_dataset(
        self,
        raw_factor: dict[str, np.ndarray],
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ):
        """把后端算出的裸数组装回带坐标的面板，这是因子层对外的唯一交付形态。

        裸数组只有形状没有含义，一旦离开这个方法就分不清哪一行是哪个时间、哪一列
        是哪个标的；所以计算结果一算完就立刻贴上坐标。

        Args:
            raw_factor (dict[str, np.ndarray]): 因子名到二维数组的映射，数组的
                两个轴依次是时间与标的。
            timestamps (np.ndarray): 时间轴取值。
            symbols (np.ndarray): 标的轴取值。

        Returns:
            Self: 已持有该面板并完成收窄的因子自身。
        """
        ds = xr.Dataset(
            {k: (["timestamp", "symbol"], v) for k, v in raw_factor.items()},
            coords={
                "timestamp": timestamps,
                "symbol": symbols,
            },
        )
        self.data_backend.to_internal(ds)
        self._auto_filter()
        return self

    @abstractmethod
    def _get_factor_func(self) -> Function:
        """由具体因子类给出自己的计算图，这是写一个新 KunQuant 因子唯一要写的东西。

        因子以声明式的算子图描述，再交给编译器生成本地代码，而不是写成逐行的数值
        运算，所以这里返回的是一张图而不是一段计算结果。

        Returns:
            Function: 描述该因子计算过程的算子图。
        """
        ...

    def cal(self) -> Self:
        """批量算出整段历史的因子：编译计算图、跑一遍，再把结果装回面板。

        编译产物用完即弃，不长期挂在实例上：它占的是本地代码的内存，而一次批量
        计算通常只跑一遍。

        Returns:
            Self: 已持有计算结果的因子自身，可继续链式调用。
        """
        input_dict, symbols, timestamp = self.config.dataset.to_kunquant(
            data_columns=self.config.data_columns
        )
        # 随便拿一个确定时间
        # [time, stocks]
        num_time = next(iter(input_dict.values())).shape[0]

        if self._lib is None:
            self._lib = self._make()

        modu = self._lib.getModule(f"{self.__class__.__name__}")  # type: ignore

        executor = kr.createMultiThreadExecutor(self.config.njobs)
        with Timer(f" {self.__class__.__name__}: cal"):
            out_dict = kr.runGraph(executor, modu, input_dict, 0, num_time)

        self._lib = None

        self._to_xarray_dataset(out_dict, timestamp, symbols)

        return self

    def cal_stream(
        self, data: dict[str, np.ndarray], timestamp: int, symbols: list[str]
    ) -> Self:
        """推入一个时间点的数据，增量算出该时刻的因子值。

        这条路径是为将来接实时行情预留的：状态留在流式上下文里，因此每来一条数据
        只需要算增量，不必把整段历史重算一遍。

        Args:
            data (dict[str, np.ndarray]): 输入列名到该时刻各标的取值的映射。
            timestamp (int): 这一条数据对应的时间戳。
            symbols (list[str]): 这一条数据的标的顺序，须与流式上下文一致。

        Returns:
            Self: 已持有该时刻因子值的因子自身，可继续链式调用。
        """
        if self._stream_context is None:
            self.init_stream()

        for name in self.config.data_columns:
            self._stream_context.pushData(
                self._buffer_name_to_id[name], data[name]
            )

        self._stream_context.run()

        out_dict = {}
        for factor in self.config.factor_names:
            alpha = self._stream_context.getCurrentBuffer(
                self._buffer_name_to_id[factor]
            )[:]
            out_dict[factor] = np.expand_dims(alpha, axis=0)

        self._to_xarray_dataset(
            out_dict, np.array([timestamp]), np.array(symbols)
        )

        return self

    def _make(self):
        """把因子计算图编译成批量模式的本地代码。

        Returns:
            object: 编译产物句柄，可从中取出与本类同名的计算模块。
        """
        with Timer(f" {self.__class__.__name__}: make"):
            return cfake.compileit(
                [
                    (
                        f"{self.__class__.__name__}",
                        self._get_factor_func(),
                        KunCompilerConfig(
                            input_layout="TS",
                            output_layout="TS",
                        ),
                    )
                ],
                f"{self.__class__.__name__}",
                cfake.CppCompilerConfig(),
            )

    def _make_stream(self):
        """把同一张因子计算图编译成流式模式的本地代码。

        与批量编译分成两个方法，是因为两者的输入输出布局不同，一份产物无法兼用。

        Returns:
            object: 编译产物句柄，可从中取出流式计算模块。
        """
        with Timer(f"{self.__class__.__name__}: make stream"):
            # The SIMD block width is deliberately left unset so KunQuant picks
            # it per architecture. It is architecture-dependent -- KunQuant's
            # own defaults for float are 8 on x86_64 and 4 on aarch64
            # (`KunQuant/Driver.py`) -- so pinning the x86 value here raised
            # `RuntimeError: Blocking length 8 is not supported for float on
            # aarch64` on Apple Silicon, i.e. on this project's own dev
            # machine. Leaving it unset preserves the previous x86_64 behaviour
            # exactly, because 8 is what KunQuant selects there anyway.
            # `partition_factor` below is unrelated: it controls graph
            # partitioning, not SIMD width.
            return cfake.compileit(
                [
                    (
                        f"{self.__class__.__name__}_stream",
                        self._get_factor_func(),
                        KunCompilerConfig(
                            partition_factor=8,
                            input_layout="STREAM",
                            output_layout="STREAM",
                            options={"opt_reduce": False, "fast_log": True},
                        ),
                    )
                ],
                f"{self.__class__.__name__}_stream",
                cfake.CppCompilerConfig(),
            )
