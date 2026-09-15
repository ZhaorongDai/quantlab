from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.base.model import BaseModel, DLModel
from quantlab.dataset.backend import XrBackend
from quantlab.enums.constant import Date
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.jsonable import to_jsonable

from .config import BacktestConfig


@dataclass(frozen=True)
class MarketSpec:
    """一个市场的回测约定：成交价列、估值价列与年化口径（03.7 D-04）。

    列名只允许出现在具体市场的规格实例上，回测器的方法体里不写任何列名。
    """

    fill_price_column: str
    valuation_price_column: str
    trading_days_per_year: int
    session_minutes_per_day: int

    def year_freq(self, bar_interval) -> pd.Timedelta:
        """一年的时长，用 vectorbt 的口径表示：`year_freq / freq` 即每年 bar 数。

        日及以上频率：每年 bar 数 = 交易日数 x (一天 / bar 间隔)；日内频率：
        每年 bar 数 = 交易日数 x 每日交易分钟数 / bar 分钟数。日频得 252，
        1 分钟得 252 x 390（03.7-RESEARCH.md Pitfall 6：vectorbt 默认按 365 天）。
        """
        interval = pd.Timedelta(bar_interval)
        if interval <= pd.Timedelta(0):
            raise ValueError(f"bar_interval must be positive, got {interval}")
        one_day = pd.Timedelta(days=1)
        if interval >= one_day:
            bars_per_year = self.trading_days_per_year * (one_day / interval)
        else:
            minutes = interval / pd.Timedelta(minutes=1)
            bars_per_year = (
                self.trading_days_per_year * self.session_minutes_per_day / minutes
            )
        return interval * bars_per_year


@dataclass
class SimulationResult:
    """引擎模拟的产出，除 `native` 外全部是引擎无关的 xarray / 纯 Python 值。

    - `value` / `returns`：组合净值与收益，维度 `timestamp`；
    - `orders`：维度 `order`，变量 `timestamp`、`symbol`、`size`、`price`、`fees`、`side`；
    - `liquidations`：强制平仓记录；
    - `bar_interval`：模拟使用的 bar 间隔；
    - `native`：引擎自己的结果对象，只由产出它的引擎读取。
    """

    value: xr.DataArray
    returns: xr.DataArray
    orders: xr.Dataset
    liquidations: list[dict]
    bar_interval: np.timedelta64
    native: object | None = None


@dataclass
class BacktestResult:
    """`run()` 的返回值；`run_dir` 是这次回测落盘的目录。"""

    run_dir: Path
    predictions: xr.Dataset
    weights: xr.Dataset
    simulation: SimulationResult
    metrics: dict = field(default_factory=dict)


class BaseBacktester(ABC):
    """回测层的基类（03.7 D-01、D-02）。

    **D-01：引擎靠继承变化，市场与选股逻辑靠组合变化。** 层级是
    `BaseBacktester`（本类）-> 引擎层（如 `VectorBtBacktester`）-> 具名的具体类
    （如 `USEquityCrossectionSelectStockVectorBt`）。具体类只是把一个市场规格
    （`MARKET`，一个 `MarketSpec`）和一个选股组件组合到某个引擎上。这样市场 x
    风格 x 引擎不会乘出一大堆类，将来的时序兄弟类、事件驱动兄弟类也不需要改动
    本类。

    **D-02：公开入口是本类上的模板方法，子类从不覆盖。** `run()` 按固定顺序
    执行：准备模型 -> 对齐因子日期并预测 -> 生成信号 -> 模拟 -> 基准 -> 指标 ->
    报告与落盘。可变的步骤是下面的抽象钩子；引擎相关的 `_simulate` /
    `_simulate_benchmark` / `_engine_stats` 由引擎层实现，`_generate_signals`
    由具体类实现。

    两个类属性由具体类用**普通类属性**满足：`config_cls`（接受的配置类，
    `config` setter 第一件事就检查它）与 `MARKET`（市场规格）。
    """

    MARKET: MarketSpec | None = None

    def __init__(self, config: BacktestConfig):
        self.config = config

    @property
    @abstractmethod
    def config_cls(self) -> type:
        """这个回测器接受的配置类（具体类用类属性覆盖）。"""

    @property
    def config(self) -> BacktestConfig:
        return self._config

    @config.setter
    def config(self, config: BacktestConfig):
        # 类型检查必须是第一条语句，先于任何校验与赋值（与 BaseModel 同一规则）。
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} requires a {self.config_cls.__name__}, "
                f"got {type(config).__name__}"
            )
        if self.MARKET is None:
            raise TypeError(
                f"{self.class_name} declares no MARKET spec; a concrete "
                f"backtester must set the MARKET class attribute"
            )
        if not isinstance(config.model, BaseModel):
            raise TypeError(
                f"{self.class_name}: config.model must be a BaseModel, got "
                f"{type(config.model).__name__}"
            )

        if config.model_mode not in ("train", "load"):
            raise ValueError(
                f"{self.class_name}: model_mode must be 'train' or 'load', got "
                f"{config.model_mode!r}"
            )
        # run() 的 load 需要 checkpoint 路径；run_cv 读的是 cv_project_dir。
        # 03.7-10 把这条放宽成「二者有其一」时，只改下面这一行条件。
        if config.model_mode == "load" and config.checkpoint is None:
            raise ValueError(
                f"{self.class_name}: model_mode='load' requires a checkpoint path"
            )
        if config.rebalance_periods < 1:
            raise ValueError(
                f"{self.class_name}: rebalance_periods must be >= 1, got "
                f"{config.rebalance_periods}"
            )
        if config.fees < 0 or config.slippage < 0:
            raise ValueError(
                f"{self.class_name}: fees and slippage must be >= 0, got "
                f"fees={config.fees}, slippage={config.slippage}"
            )
        if config.init_cash <= 0:
            raise ValueError(
                f"{self.class_name}: init_cash must be > 0, got {config.init_cash}"
            )
        if pd.Timestamp(config.start_date) > pd.Timestamp(config.end_date):
            raise ValueError(
                f"{self.class_name}: start_date {config.start_date} is after "
                f"end_date {config.end_date}"
            )
        if config.benchmark_dataset is not None:
            raise NotImplementedError(
                f"{self.class_name}: benchmark comparison is excluded from phase "
                f"03.7 by D-08 until directly-downloaded index price data "
                f"exists; the benchmark_dataset config slot is kept, leave it None"
            )

        self._config = config
        self._config.name = self.import_path
        self._validate_config()

    def _validate_config(self) -> None:
        """具体类的额外构造期校验；默认什么都不做。"""

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def get_config(self) -> dict:
        """标量字段 + 逐个嵌套的数据集与模型配置；不对整个配置 `asdict`。"""
        cfg = self.config.to_dict()
        cfg["price_dataset"] = self.config.price_dataset.get_config()
        cfg["model"] = self.config.model.get_config()
        cfg["benchmark_dataset"] = (
            None
            if self.config.benchmark_dataset is None
            else self.config.benchmark_dataset.get_config()
        )
        return cfg

    @staticmethod
    def _iso_date(value) -> str:
        """把任意日期样式的值规范成 ISO `YYYY-MM-DD` 字符串（03.7-RESEARCH.md Pitfall 10）。

        本模块写进数据集或因子配置的每一个日期都经过这里：数据集 setter 的 ISO
        规范化只在整份配置赋值时发生，直接改 `config.start_date` 不会经过它，而
        下游的日期比较是字符串字典序。先 `str(value)`，因为 `pd.Timestamp` 不接受
        `numpy.str_`；真实数据上的折日期形如 `'2026-08-07T00:00:00.000000000'`。
        """
        return pd.Timestamp(str(value)).strftime("%Y-%m-%d")

    def run(self) -> BacktestResult:
        """模型回测的模板方法（D-02）。子类不覆盖。"""
        start_date = self._iso_date(self.config.start_date)
        end_date = self._iso_date(self.config.end_date)

        self._prepare_model()
        calendar = self._price_calendar(end_date)
        predictions = self._align_and_predict(start_date, end_date, calendar)

        prices = self._load_prices(start_date, end_date)
        if prices.sizes.get("timestamp", 0) == 0:
            raise ValueError(
                f"{self.class_name}: no price bars between {start_date} and "
                f"{end_date}"
            )

        # 预测铺到价格数据集的全部标的上：缺的标的是 NaN，也就不可选（D-06）。
        predictions = predictions.reindex(
            timestamp=prices.timestamp.values, symbol=prices.symbol.values
        )

        model = self.config.model
        split = self._split_window(
            prices.timestamp.values,
            self._training_window(
                calendar, model.config.train_start, model.config.train_end
            ),
        )

        weights = self._generate_signals(predictions, prices)
        self._assert_weights_contract(weights, prices)

        simulation = self._simulate(weights, prices)
        benchmark = self._simulate_benchmark(start_date, end_date)
        metrics = self._compute_metrics(simulation, benchmark, split)
        run_dir = self._report_and_persist(predictions, weights, simulation, metrics)

        return BacktestResult(
            run_dir=run_dir,
            predictions=predictions,
            weights=weights,
            simulation=simulation,
            metrics=metrics,
        )

    def _prepare_model(self) -> None:
        """按 `model_mode` 准备模型（D-13）。

        - `train`：`collect()` 再 `train()`，用的是**模型自己**配置里的
          train/test 日期。回测窗口从不写进这些日期：回测窗口只决定预测区间和
          样本内/外的划分，改写它们会让训练集跟着回测参数漂移。
        - `load`：
          1. 先检查 checkpoint 文件存在，缺了直接报错并写明路径。这一步必须在
             任何特征计算之前，否则一个拼错的路径要白算一遍特征才暴露。
          2. 只对 `DLModel`，先把特征面板放进模型的 data backend。
             `DLModel._read_checkpoint` 用 `num_symbols` 重建网络，而
             `num_symbols` 读的正是这个 backend，空着就会在 `load()` 里报错
             （03.7-RESEARCH.md Pitfall 11）。`MLModel` 的 checkpoint 就是完整
             模型，不调 `_init_model`，所以跳过这一步。
          3. `model.load(checkpoint)`。
        """
        model = self.config.model
        if self.config.model_mode == "load":
            checkpoint = Path(self.config.checkpoint)  # type: ignore[arg-type]
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"{self.class_name}: checkpoint {checkpoint} does not exist"
                )
            if isinstance(model, DLModel):
                model.data_backend.to_internal(model._collect_all_features())
            model.load(checkpoint)
        else:
            model.collect()
            model.train()

    def _price_calendar(self, end_date: str) -> np.ndarray:
        """价格数据集自己的交易日历（截至 `end_date`），用于按 bar 计数。

        日期直接写 ISO 字符串进数据集配置：数据集 setter 的 ISO 规范化只在整份
        配置赋值时发生。`overwrite=True` 绕过 XrBackend 的读缓存，否则拿回的是
        之前更窄的一次读取（03.7-RESEARCH.md Pitfall 1）。
        """
        dataset = self.config.price_dataset
        dataset.config.start_date = Date.START_DATE
        dataset.config.end_date = end_date
        dataset.read(overwrite=True)
        return np.sort(dataset.get_xarray_dataset().timestamp.values)

    def _warmup_start(self, calendar: np.ndarray, start_date: str) -> str:
        """预热起点：在价格日历上从 `start_date` 往前数最大因子窗口个 bar（D-15）。

        按 bar 计数，不做日历日减法。`Factor._reset_dataset_config` 自己减的日历日
        只是额外缓冲，这里不依赖它。
        """
        factors = self.config.model.config.factors
        window = max((int(f.config.window) for f in factors), default=0)
        idx = int(
            np.searchsorted(
                calendar, np.datetime64(pd.Timestamp(start_date)), side="left"
            )
        )
        warmup = self._iso_date(calendar[max(idx - window, 0)])
        if idx - window < 0:
            logger.warning(
                f"{self.class_name}: warm-up needs {window} bars before "
                f"{start_date} but the price calendar has only {idx}; short by "
                f"{window - idx} bar(s), clamping the warm-up start to the "
                f"first bar {warmup}"
            )
        return warmup

    def _refresh_factor_reads(self, factor) -> None:
        """改过日期之后，强制重读因子背后的数据（D-14，03.7-RESEARCH.md Pitfall 1）。

        XrBackend 的 `read` 一旦已经持有数据就直接返回，不再打开存储；而
        `BaseDataset.read()` 的 `_filter` 与 `Factor.read()` 的 `_auto_filter`
        都是**就地**收窄这份缓存。所以模型先按自己的日期 collect/train 过之后，
        再把因子日期放宽到「预热 + 回测窗口」重读，拿回的仍是先前那段更窄的
        窗口：不报错，只是缺 bar——预热悄悄变短，或首个调仓日整行没有预测。

        - 数据集总是 `read(overwrite=True)`：`cal` 策略的因子从数据集现算；
        - `factor_data_strategy == "read"` 时，因子库本身也 `read(overwrite=True)`
          （03.7-02 加的开关）：`read` 策略的特征直接来自因子库的缓存。
        """
        factor.config.dataset.read(overwrite=True)
        if self.config.model.config.factor_data_strategy == "read":
            factor.read(overwrite=True)

    def _align_and_predict(
        self, start_date: str, end_date: str, calendar: np.ndarray
    ) -> xr.Dataset:
        """改因子配置日期（含预热）-> 只算特征 -> 预测 -> 切回回测窗口（D-14）。"""
        model = self.config.model
        warmup = self._warmup_start(calendar, start_date)
        for factor in model.config.factors:
            factor.config.start_date = warmup
            factor.config.end_date = end_date
            factor._reset_dataset_config()
            self._refresh_factor_reads(factor)

        features = model._collect_all_features()
        return model.predict_panel(features).sel(
            timestamp=slice(start_date, end_date)
        )

    def _load_prices(self, start_date: str, end_date: str) -> xr.Dataset:
        """回测窗口内的成交价与估值价两列，深拷贝后返回。

        深拷贝：价格数据集对象可能与某个因子共用，之后再改日期不能改到这里。
        """
        dataset = self.config.price_dataset
        dataset.config.start_date = start_date
        dataset.config.end_date = end_date
        ds = dataset.read(overwrite=True).get_xarray_dataset()

        fill = self.MARKET.fill_price_column  # type: ignore[union-attr]
        valuation = self.MARKET.valuation_price_column  # type: ignore[union-attr]
        for column in (fill, valuation):
            if column not in ds.data_vars:
                raise ValueError(
                    f"{self.class_name}: price column {column!r} not found in "
                    f"{dataset.config.zarr_file_path}"
                )
        return ds[[fill, valuation]].load().copy(deep=True)

    def _assert_weights_contract(
        self, weights: xr.Dataset, prices: xr.Dataset
    ) -> None:
        """目标权重契约（D-03）：非调仓行全 NaN，调仓行全有限且毛敞口 <= 1。"""
        if "weight" not in weights.data_vars:
            raise ValueError(
                f"{self.class_name}: weights must carry a 'weight' variable, got "
                f"{list(weights.data_vars)}"
            )
        weight = weights["weight"]
        if weight.dims != ("timestamp", "symbol"):
            raise ValueError(
                f"{self.class_name}: weight dims must be ('timestamp', 'symbol'), "
                f"got {weight.dims}"
            )
        if not np.array_equal(weight.timestamp.values, prices.timestamp.values):
            raise ValueError(
                f"{self.class_name}: weight timestamps do not match the price "
                f"timestamps"
            )
        if not np.array_equal(weight.symbol.values, prices.symbol.values):
            raise ValueError(
                f"{self.class_name}: weight symbols do not match the price symbols"
            )

        values = weight.values
        timestamps = weight.timestamp.values
        all_nan = np.isnan(values).all(axis=1)
        all_finite = np.isfinite(values).all(axis=1)
        mixed = ~(all_nan | all_finite)
        if mixed.any():
            first = timestamps[int(np.argmax(mixed))]
            raise ValueError(
                f"{self.class_name}: weight row at {first} mixes NaN and finite "
                f"values; a row must be all-NaN (hold) or all-finite (rebalance)"
            )
        gross = np.where(all_finite, np.abs(np.nan_to_num(values)).sum(axis=1), 0.0)
        over = gross > 1 + 1e-9
        if over.any():
            idx = int(np.argmax(over))
            raise ValueError(
                f"{self.class_name}: weight row at {timestamps[idx]} has gross "
                f"exposure {gross[idx]} > 1"
            )

    @abstractmethod
    def _generate_signals(
        self, predictions: xr.Dataset, prices: xr.Dataset
    ) -> xr.Dataset:
        """预测 + 价格 -> 满足 D-03 契约的目标权重 `xr.Dataset`。"""

    @abstractmethod
    def _simulate(self, weights: xr.Dataset, prices: xr.Dataset) -> SimulationResult:
        """按目标权重模拟组合；bar t 的信号在 bar t+1 成交（D-05）。"""

    @abstractmethod
    def _simulate_benchmark(
        self, start_date: str, end_date: str
    ) -> SimulationResult | None:
        """基准的买入持有模拟；没有基准时返回 None（D-08）。"""

    @abstractmethod
    def _engine_stats(self, simulation: SimulationResult) -> dict:
        """引擎自己的整段统计指标，键为指标名。"""

    def _label_horizon_bars(self) -> int:
        """模型所有标签里最大的 `n_forward_periods`，单位是 bar（D-17）。

        `train_end` 那一 bar 的标签读的是之后 n 个 bar 的价格，所以这 n 个 bar
        也属于样本内。标签的 `config.kwargs` 为 None 或没有这个键时，它贡献 0，
        并且 warning 写明标签类名：不静默猜一个值。
        """
        horizon = 0
        for label in self.config.model.config.labels:
            kwargs = label.config.kwargs
            if kwargs is None or "n_forward_periods" not in kwargs:
                logger.warning(
                    f"{self.class_name}: label {type(label).__name__} has no "
                    f"n_forward_periods in config.kwargs; it contributes a "
                    f"0-bar horizon to the effective training window (D-17)"
                )
                continue
            horizon = max(horizon, int(kwargs["n_forward_periods"]))
        return horizon

    def _training_window(
        self, calendar: np.ndarray, train_start, train_end
    ) -> tuple[str, str] | None:
        """模型的有效训练窗口 `[train_start, train_end + 标签期限]`（D-17），ISO 日期对。

        期限在价格日历上按 bar 数，不做日历日加法：周五的 `train_end` 加 2 个
        bar 是下周二。
        - 起点：日历上第一个不早于 `train_start` 的 bar；
        - 终点：日历上最后一个不晚于 `train_end` 的 bar 再往后数期限个 bar，
          超出日历时截到最后一个 bar。

        任一日期为 None 时返回 None 并 warning：没有训练日期就无从判断样本内，
        metrics 里记显式的 null，而不是猜。
        """
        if train_start is None or train_end is None:
            logger.warning(
                f"{self.class_name}: model config has train_start={train_start!r}, "
                f"train_end={train_end!r}; the effective training window is "
                f"unknown, so metrics record training_window as null and every "
                f"backtest bar as out-of-sample (D-17)"
            )
            return None

        calendar = np.asarray(calendar).astype("datetime64[ns]")
        start = np.datetime64(pd.Timestamp(self._iso_date(train_start)), "ns")
        end = np.datetime64(pd.Timestamp(self._iso_date(train_end)), "ns")
        last = calendar.size - 1

        start_idx = int(np.searchsorted(calendar, start, side="left"))
        end_idx = (
            int(np.searchsorted(calendar, end, side="right"))
            - 1
            + self._label_horizon_bars()
        )
        window_start = (
            self._iso_date(calendar[start_idx])
            if start_idx <= last
            else self._iso_date(train_start)
        )
        window_end = (
            self._iso_date(calendar[min(end_idx, last)])
            if end_idx >= 0
            else self._iso_date(train_end)
        )
        return window_start, window_end

    def _split_window(
        self, window_timestamps: np.ndarray, training_window: tuple[str, str] | None
    ) -> dict:
        """把回测窗口的 bar 切成样本内与样本外（D-17）。

        返回三个键，原样并入 metrics 顶层：
        - `training_window`：有效训练窗口的 ISO 日期对，或 None；
        - `in_sample_range`：回测窗口与训练窗口重叠部分的首尾 bar，或 None；
        - `out_of_sample_ranges`：重叠之外的 bar 组成的连续段，0、1 或 2 段。

        比较按日期（`datetime64[D]`）做，与训练窗口的 ISO 日期口径一致。两个
        窗口都是区间，所以重叠部分一定连续。重叠非空时 warning 写明两个窗口，
        并说明样本内外分开报告；回测照常继续。
        """
        timestamps = np.asarray(window_timestamps).astype("datetime64[ns]")
        split = {
            "training_window": training_window,
            "in_sample_range": None,
            "out_of_sample_ranges": [],
        }
        if timestamps.size == 0:
            return split

        days = timestamps.astype("datetime64[D]")
        if training_window is None:
            in_sample = np.zeros(days.size, dtype=bool)
        else:
            first = np.datetime64(training_window[0], "D")
            last = np.datetime64(training_window[1], "D")
            in_sample = (days >= first) & (days <= last)

        pieces = []
        if in_sample.any():
            idx = np.flatnonzero(in_sample)
            lo, hi = int(idx[0]), int(idx[-1])
            split["in_sample_range"] = (
                self._iso_date(timestamps[lo]),
                self._iso_date(timestamps[hi]),
            )
            if lo > 0:
                pieces.append((0, lo - 1))
            if hi < timestamps.size - 1:
                pieces.append((hi + 1, timestamps.size - 1))
            window = (self._iso_date(timestamps[0]), self._iso_date(timestamps[-1]))
            logger.warning(
                f"{self.class_name}: backtest window {window[0]}..{window[1]} "
                f"overlaps the model's effective training window "
                f"{training_window[0]}..{training_window[1]} (train_start.."  # type: ignore[index]
                f"train_end + label horizon, D-17); bars "
                f"{split['in_sample_range'][0]}..{split['in_sample_range'][1]} "
                f"are in-sample. Continuing: in-sample and out-of-sample results "
                f"are reported separately"
            )
        else:
            pieces.append((0, timestamps.size - 1))

        split["out_of_sample_ranges"] = [
            (self._iso_date(timestamps[a]), self._iso_date(timestamps[b]))
            for a, b in pieces
        ]
        return split

    def _compute_metrics(
        self,
        simulation: SimulationResult,
        benchmark: SimulationResult | None,
        split: dict,
    ) -> dict:
        metrics = {"whole": self._engine_stats(simulation)}
        if benchmark is not None:
            metrics["benchmark"] = self._engine_stats(benchmark)
        for key in ("training_window", "in_sample_range", "out_of_sample_ranges"):
            metrics[key] = split[key]
        return metrics

    def _run_dir_name(self) -> str:
        """`{class}_{timestamp}`（D-24）；带微秒，同一秒内的两次运行不会撞名。"""
        return f"{self.class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _report_and_persist(
        self,
        predictions: xr.Dataset,
        weights: xr.Dataset,
        simulation: SimulationResult,
        metrics: dict,
    ) -> Path:
        """建新的运行目录并写入配置、权重、净值与指标（D-24）；从不覆盖已有目录。"""
        run_dir = Path(self.config.output_dir) / self._run_dir_name()
        if run_dir.exists():
            raise RuntimeError(f"{run_dir} already exists")
        run_dir.mkdir(parents=True)

        write_json_atomically(
            run_dir / "config.json", to_jsonable(self.get_config()), indent=2
        )
        XrBackend().to_internal(weights).write(str(run_dir / "weights.zarr"))
        XrBackend().to_internal(
            xr.Dataset({"value": simulation.value, "returns": simulation.returns})
        ).write(str(run_dir / "equity.zarr"))
        write_json_atomically(
            run_dir / "metrics.json", to_jsonable(metrics), indent=2
        )
        return run_dir
