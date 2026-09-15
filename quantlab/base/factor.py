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

from quantlab.base.config import (
    BaseFactorConfig,
    FactorConfig,
    PolarsFactorConfig,
)
from quantlab.dataset.backend import XrBackend
from quantlab.enums.constant import Date
from quantlab.utils.timer import Timer


class Factor(ABC):
    def __init__(self, config: BaseFactorConfig):
        # Ordering is load-bearing: assigning `self.config` fires the property
        # setter below, which runs before `self.data_backend` exists. No
        # setter-reachable method may read the storage backend.
        self.config = config
        self.data_backend = XrBackend()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"

    def _auto_filter(self):
        self.data_backend.filter_by_date(
            col="timestamp",
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        if self.config.symbols is not None:
            self.data_backend.filter_by_symbol("symbol", self.config.symbols)

    @property
    def config(self) -> BaseFactorConfig:
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
        """
        Resolve `config.factor_names` eagerly, at config-assignment time.
        """
        if self._config.factor_names is None:
            self._config.factor_names = self._get_factor_names()

    def _reset_dataset_config(self):
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
        return self.config.dataset.num_symbols

    @property
    def num_factors(self) -> int:
        return len(self.get_factor_names())

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def symbols(self) -> list[str]:
        return self.config.dataset.symbols

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def read(self, overwrite: bool = False) -> Self:
        """Read the factor store and narrow it to the configured window.

        The default keeps the cached read: once the backend holds data it is
        not re-opened, and `_auto_filter()` narrows that cached panel in place.
        Pass `overwrite=True` to re-open the store. It is required after
        mutating `config.start_date`/`end_date`, which is what the
        backtester's D-14 re-dating does (RESEARCH Pitfall 1).
        """
        self.data_backend.read(self.config.file_path, overwrite=overwrite)
        self._auto_filter()
        return self

    def save(self, mode: Literal["a", "w"] = "a", **kwargs) -> Self:
        with Timer(f"{self.__class__.__name__}: save"):
            self._auto_filter()
            try:
                self.data_backend.write(
                    self.config.file_path,
                    mode=mode,
                    **kwargs,
                )
            except ValueError as exc:
                if "already exists with different dimension sizes" not in str(
                    exc
                ):
                    raise
                raise ValueError(
                    f'{self.class_name}.save(mode="a"): cannot write this '
                    f"date range into the existing store at "
                    f'{self.config.file_path}. zarr\'s "a" means "overwrite '
                    f'variables in an existing store", NOT "append along '
                    f'time", so a second, differently-sized date range is '
                    f'rejected. Use save(mode="w") to replace the store, or '
                    f"delete it first. To EXTEND it with a later date range "
                    f"instead, call update(), which reconciles the timestamp, "
                    f"symbol and variable axes automatically and inherits "
                    f"XrBackend.append()'s guards. save() writes wholesale; "
                    f"update() extends. "
                    f"Original error: {exc}"
                ) from exc
            return self

    def update(self, **kwargs) -> Self:
        with Timer(f"{self.__class__.__name__}: update"):
            self._auto_filter()
            self.data_backend.widen_and_append(
                self.config.file_path,
                fill_values=self._widen_fill_values(),
                **kwargs,
            )
            return self

    def _widen_fill_values(self) -> dict:
        return {}

    def _get_lazyframe(self) -> pl.LazyFrame:
        df = self.data_backend.get_xarray_dataset().to_pandas()  # type: ignore
        df = pl.LazyFrame(df.reset_index())
        return df

    def _get_xarray_dataset(self) -> xr.Dataset:
        return self.data_backend.get_xarray_dataset()  # type: ignore

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError

    def get_features(self) -> xr.Dataset:
        return self._get_features(self._get_xarray_dataset())

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError

    def get_labels(self) -> xr.Dataset:
        return self._get_labels(self._get_xarray_dataset())

    def get_factor_names(self) -> tuple[str, ...]:
        return self.config.factor_names

    def get_config(self) -> dict:
        ds_config = self.config.dataset.get_config()
        cfg = self.config.to_dict()
        cfg["dataset"] = ds_config  # type: ignore
        return cfg  # type: ignore

    @abstractmethod
    def _get_factor_names(self) -> tuple[str, ...]: ...

    @abstractmethod
    def cal(self) -> Self: ...


class FactorKunQuant(Factor):
    """The KunQuant factor backend."""

    # D-26: the config class `quantlab/utils/module.py` rebuilds this factor with.
    config_cls = FactorConfig

    def __init__(self, config: FactorConfig):
        super().__init__(config)
        self._stream_context: kr.StreamContext = None
        self._lib = None
        self._buffer_name_to_id = dict()

    def _auto_filter(self):
        if self.config.mode == "batch":
            super()._auto_filter()

    @property
    def num_symbols(self) -> int:
        if self.config.mode == "batch":
            return super().num_symbols
        elif self.config.mode == "stream":
            return len(self.config.dataset.config.symbols)
        else:
            raise ValueError(f"mode {self.config.mode} is not supported")

    @property
    def symbols(self) -> list[str]:
        if self.config.mode == "batch":
            return super().symbols
        elif self.config.mode == "stream":
            return list(self.config.dataset.config.symbols)
        else:
            raise ValueError(f"mode {self.config.mode} is not supported")

    def init_stream(self) -> Self:
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
    def _get_factor_func(self) -> Function: ...

    def cal(self) -> Self:
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
        with Timer(f"{self.__class__.__name__}: make stream"):
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


class FactorPolars(Factor):
    """
    Batch-only factor backend whose factor logic is written in Polars.
    """

    # D-26: the config class `quantlab/utils/module.py` rebuilds this factor with.
    config_cls = PolarsFactorConfig

    _INDEX_COLUMNS = ("timestamp", "symbol")

    _SCHEMA_PROBE_ROWS = 8

    def __init__(self, config: PolarsFactorConfig):
        super().__init__(config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """
        Derive the factor names by asking the graph what it produces.
        """
        probe = self.config.dataset.head(self._SCHEMA_PROBE_ROWS)
        factor_lf = self._get_factor_lazyframe(probe)
        return tuple(
            name
            for name in factor_lf.collect_schema().names()
            if name not in self._INDEX_COLUMNS
        )

    @abstractmethod
    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Write the factor here. This is the one method a subclass overrides.
        """
        ...

    def cal(self) -> Self:
        lf = self.config.dataset.read().get_lazyframe()
        factor_lf = self._get_factor_lazyframe(lf)

        self.config.factor_names = tuple(
            name
            for name in factor_lf.collect_schema().names()
            if name not in self._INDEX_COLUMNS
        )

        with Timer(f"{self.__class__.__name__}: cal"):
            frame = factor_lf.collect().to_pandas()
            frame = frame.set_index(list(self._INDEX_COLUMNS))
            data = xr.Dataset.from_dataframe(frame)

        self.data_backend.to_internal(data)
        self._auto_filter()
        return self
