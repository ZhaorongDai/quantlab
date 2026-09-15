import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import xarray as xr
from loguru import logger

from quantlab.base.model import BaseModel, DLModel
from quantlab.dataset.backend import XrBackend
from quantlab.enums.constant import Date
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.backtest_report import write_backtest_report
from quantlab.utils.fingerprint import dataset_fingerprint
from quantlab.utils.jsonable import to_jsonable

from .config import BacktestConfig, FactorConfig

#: 数据指纹比较的字段（D-27）：任一不同就 warning。
FINGERPRINT_COMPARED_FIELDS = ("digest", "start", "end", "n_timestamps", "n_symbols")

#: 一个日历年的平均天数。长于一天的 bar 是日历跨度（周线、月线），按它年化
#: （`MarketSpec.year_freq`，代码审查 CR-02）。
CALENDAR_DAYS_PER_YEAR = 365.25


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

        - 日内频率：每年 bar 数 = 交易日数 x 每日交易分钟数 / bar 分钟数，
          1 分钟得 252 x 390（03.7-RESEARCH.md Pitfall 6：vectorbt 默认按 365 天）；
        - 恰好一天：每年 bar 数 = 交易日数，日频得 252。交易日历上的日线
          时间戳差分的众数是 1 天，一个 bar 就是一个交易日；
        - 长于一天：一个 bar 是一段**日历**跨度（周线每个日历周一个 bar，
          节假日不会让一周消失；月线同理），所以每年 bar 数 =
          `CALENDAR_DAYS_PER_YEAR` / bar 天数，并以交易日数封顶：一个 bar 不会
          短于一个交易日。周线约 52.18，30 天约 12.18，31 天约 11.78。

        这个分段在一天处连续、且随间隔单调不增：`min(交易日数, 365.25 / 天数)`
        在一天时取 252，到约 1.45 天两者相等，之后按日历跨度下降。

        **代码审查 CR-02 更正。** 以前长于一天的频率用「交易日数 x (一天 /
        间隔)」，把**交易日**计数除以**日历日**间隔：周线得 36、31 天月线得
        8.13，周线的 Sharpe / Sortino 被低估约 sqrt(52/36) 倍，Calmar、年化
        收益和年化换手也跟着错，而且不报任何错。被放弃的另两种做法：审查建议
        的「日历天数 x 5/7 取整作为每 bar 交易日数」（周线 50.4、31 天月线
        11.45，把节假日当成会删掉整周/整月的 bar），以及对长于一天的频率直接
        报错（项目是多频率的，D-18 明确调仓与频率无关）。
        """
        interval = pd.Timedelta(bar_interval)
        if interval <= pd.Timedelta(0):
            raise ValueError(f"bar_interval must be positive, got {interval}")
        one_day = pd.Timedelta(days=1)
        if interval >= one_day:
            bars_per_year = min(
                float(self.trading_days_per_year),
                CALENDAR_DAYS_PER_YEAR * (one_day / interval),
            )
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
    - `trades`：维度 `trade`，变量 `symbol`、`entry_timestamp`、`exit_timestamp`、
      `pnl`、`return`、`status`（`Open` / `Closed`）；没有交易时是空 Dataset；
    - `native`：引擎自己的结果对象，只由产出它的引擎读取。
    """

    value: xr.DataArray
    returns: xr.DataArray
    orders: xr.Dataset
    liquidations: list[dict]
    bar_interval: np.timedelta64
    trades: xr.Dataset | None = None
    native: object | None = None


@dataclass
class BacktestResult:
    """`run()` 的返回值；`run_dir` 是这次回测落盘的目录。"""

    run_dir: Path
    predictions: xr.Dataset
    weights: xr.Dataset
    simulation: SimulationResult
    metrics: dict = field(default_factory=dict)


@dataclass
class _BacktestWindow:
    """一个回测窗口跑完、尚未落盘的中间产物；`run()` 与 `run_cv()` 的每折共用。"""

    predictions: xr.Dataset
    prices: xr.Dataset
    weights: xr.Dataset
    simulation: SimulationResult
    split: dict
    metrics: dict


@dataclass
class CVBacktestResult:
    """`run_cv()` 的返回值（D-16、D-35）。

    - `run_dir`：这次 CV 回测落盘的目录；
    - `folds`：每折一条记录，含 `fold`、四个日期、`checkpoint`，以及该折自己的
      `predictions`、`weights`、`simulation`、`metrics`（独立的逐折模拟）；
    - `weights` / `simulation`：拼接后的样本外权重与**一次**连续模拟；
    - `metrics`：与 metrics.json 相同的结构，`stitched`、`folds`、`notes`。
    """

    run_dir: Path
    folds: list[dict]
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
        # 指纹状态先于 config 赋值（D-27），setter 与校验钩子里都可以放心读它们。
        # `expected_fingerprint`：从已存的 fingerprint.json（或 config.json 的
        # `data_fingerprint`）重建回测时设置，run() 用它比对本次读到的数据。
        self.expected_fingerprint: dict | None = None
        self._fingerprints: dict = {}
        # train 模式下本次 run() 训练出的 checkpoint 绝对路径；load 模式与未运行时
        # 为 None（代码审查 WR-04）。get_config 与 metrics 据此记录。
        self._trained_checkpoint: str | None = None
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
        # load 模式二者至少有其一：run() 读 checkpoint，run_cv() 读 cv_project_dir。
        # 缺的恰好是某个入口自己要的那一个时，由该入口在运行时拒绝（D-13、D-16）。
        if (
            config.model_mode == "load"
            and config.checkpoint is None
            and config.cv_project_dir is None
        ):
            raise ValueError(
                f"{self.class_name}: model_mode='load' requires a checkpoint path "
                f"(for run()) or a cv_project_dir (for run_cv())"
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

        # 路径字段一律存绝对路径（代码审查 WR-03）：config.json 会在另一个进程、
        # 另一个工作目录里重建回测（D-25），相对路径到那里指向别处。
        for name in ("checkpoint", "cv_project_dir", "output_dir"):
            value = getattr(config, name)
            if value is not None:
                setattr(config, name, str(Path(value).absolute()))

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
        """标量字段 + 逐个嵌套的数据集与模型配置；不对整个配置 `asdict`。

        跑过一次之后，顶层多一个 `data_fingerprint`（D-25、D-27）：本次读到的
        每个数据集的指纹。落盘的 config.json 因此带着重建时比对所需的记录。
        """
        cfg = self.config.to_dict()
        cfg["price_dataset"] = self.config.price_dataset.get_config()
        cfg["model"] = self.config.model.get_config()
        cfg["benchmark_dataset"] = (
            None
            if self.config.benchmark_dataset is None
            else self.config.benchmark_dataset.get_config()
        )
        if self._fingerprints:
            cfg["data_fingerprint"] = dict(self._fingerprints)
        # train 模式跑过之后再多一个记录 `trained_checkpoint`（代码审查 WR-04）：
        # 本次训练出的 checkpoint，用 load 模式可以精确回放同一个模型。
        if self._trained_checkpoint is not None:
            cfg["trained_checkpoint"] = self._trained_checkpoint
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

    @staticmethod
    def _bar_label(value) -> str:
        """一个 bar 时间戳的持久化标签：午夜的 bar 写 ISO 日期，其余写完整 ISO 时间。

        样本内外的区间端点（`training_window`、`in_sample_range`、
        `out_of_sample_ranges`）都经过这里（代码审查 CR-01）。日线的 bar 都在
        午夜，所以日线的标签与以前一样是日期；日内 bar 保留时刻，区间端点不会
        被截到当天。读回标签时一律按**精确时间戳**比较（`_label_ns`），不再按天。
        """
        ts = pd.Timestamp(str(value)) if isinstance(value, str) else pd.Timestamp(value)
        if ts == ts.normalize():
            return ts.strftime("%Y-%m-%d")
        return ts.isoformat()

    @staticmethod
    def _label_ns(label) -> np.datetime64:
        """把 `_bar_label` 写出的标签还原成精确的 `datetime64[ns]`（日期即午夜）。"""
        return np.datetime64(pd.Timestamp(str(label)).to_datetime64(), "ns")

    @staticmethod
    def _slice_bound(value):
        """模型层切片端点的原样形式：字符串保持字符串，其余转 `pd.Timestamp`。

        模型层按 `data.sel(timestamp=slice(train_start, train_end))` 训练，而
        xarray 把这个切片交给 pandas 的 `slice_indexer`。字符串端点按其**分辨率**
        解释：`"2024-05-17"` 包含当天全部 bar，`"2024-05-17T13:00"` 只到 13:00。
        `_training_window` 用同一个 `slice_indexer` 定位训练段，所以必须把端点
        原样交过去，不能先截成日期（代码审查 CR-01）。`str()` 是因为
        `numpy.str_` 不是所有 pandas 入口都接受。
        """
        if isinstance(value, str):
            return str(value)
        return pd.Timestamp(value)

    def run(self) -> BacktestResult:
        """模型回测的模板方法（D-02）。子类不覆盖。

        load 模式下 D-17 用 checkpoint 自己 `config.json` 记录的训练日期
        （代码审查 WR-01）。记录与 `config.model` 的日期选中的训练 bar 不同时
        （`_warn_if_config_model_dates_differ`，G-03.7-7）warning 一次，仍用
        记录的日期。这个比较需要价格日历，所以在 `_price_calendar` 之后做。
        """
        if self.config.model_mode == "load" and self.config.checkpoint is None:
            raise ValueError(
                f"{self.class_name}: run() with model_mode='load' requires "
                f"config.checkpoint; cv_project_dir is read only by run_cv()"
            )
        start_date = self._iso_date(self.config.start_date)
        end_date = self._iso_date(self.config.end_date)
        # 每次运行重新记录指纹：同一个回测器跑第二次，不能带着上一次的记录。
        self._fingerprints = {}
        # 训练出的 checkpoint 同理（代码审查 WR-04）：只有本次 train 模式才有。
        self._trained_checkpoint = None

        # load 模式下训练日期取自 checkpoint 自己的 config.json（代码审查 WR-01）。
        train_bounds = self._prepare_model()
        calendar = self._price_calendar(end_date)
        # 与 config.model 的日期比较按日历上选中的 bar 做（G-03.7-7），所以在日历之后。
        if self.config.model_mode == "load":
            self._warn_if_config_model_dates_differ(calendar, train_bounds)
        window = self._backtest_window(start_date, end_date, calendar, *train_bounds)
        self._compare_fingerprints()

        metrics = window.metrics
        if self._trained_checkpoint is not None:
            metrics["trained_checkpoint"] = self._trained_checkpoint
        metrics["notes"] = self._report_notes()
        run_dir = self._report_and_persist(
            window.predictions, window.weights, window.simulation, metrics
        )
        # wandb 默认关闭（D-28）：只有显式打开才会有任何数据离开本机。
        if self.config.use_wandb:
            self._log_to_wandb(run_dir, metrics)

        return BacktestResult(
            run_dir=run_dir,
            predictions=window.predictions,
            weights=window.weights,
            simulation=window.simulation,
            metrics=metrics,
        )

    def run_cv(self) -> CVBacktestResult:
        """模型 CV 回测的模板方法（D-02、D-16、D-17、D-35、D-36）。子类不覆盖。

        回放一次 `train_cv` 的交叉验证：读它在项目目录里写的 `cv_folds.json`，
        每折用**该折自己的** checkpoint，只回测该折的样本外测试段。顺序：

        1. 读清单并校验格式（`_read_cv_folds`），只留测试段落在回测窗口内的折
           （`_select_folds`）；
        2. 在价格日历上断言这些折的测试段首尾相接、互不重叠
           （`_assert_contiguous_folds`）。这一步先于任何模型加载与模拟：拼接
           一个有缺口或重叠的序列得到的曲线不对应任何真实的交易路径；
        3. 逐折：加载该折 checkpoint，`_backtest_window` 回测该折测试段（含预热）。
           样本内/外按**该折**的 train 日期加标签期限划分（D-17 逐折），所以
           gap 为 0 时每折开头的期限个 bar 是样本内，并各自 warning。

        4. 拼接（D-35）：各折测试段权重沿 `timestamp` 拼起来，对「首折
           test_start .. 末折 test_end」的价格跑**一次**连续模拟，资金在折边界
           不重置；逐折指标仍来自上面各自独立的逐折模拟。拼接曲线按构造是样本
           外的，只有每折开头的标签期限个 bar 是样本内，`in_sample_ranges`
           逐段记下它们（`_stitched_split`）；
        5. 指纹覆盖整个拼接窗口（D-27）：因子重新定到「首折预热起点 .. 末折
           test_end」并重读后记录，价格指纹取自拼接价格，然后比对；
        6. 落盘（`_persist_cv`），可选 wandb，返回 `CVBacktestResult`。

        折日期经 `_iso_date` 规范成 ISO 日期（真实数据上是纳秒字符串），所以
        判定按日期粒度进行。

        训练日期（WR-01、G-03.7-7）：每折以清单的训练日期为准。checkpoint 自己
        记录的训练日期只与清单核对，按两对日期在价格日历上选中的 bar 比较
        （`_same_training_bars`），不同时 warning 一次。每折的训练段本来就与
        回测模型配置里那一段不同，所以这里从不与它比较；那个比较只属于
        `run()` 的 load 模式。
        """
        if self.config.cv_project_dir is None:
            raise ValueError(
                f"{self.class_name}: run_cv() requires config.cv_project_dir, the "
                f"train_cv project directory holding "
                f"{BaseModel.CV_FOLDS_FILENAME}"
            )
        if self.config.model_mode != "load":
            raise ValueError(
                f"{self.class_name}: run_cv() replays the checkpoints of an "
                f"existing train_cv run and requires model_mode='load', got "
                f"{self.config.model_mode!r}"
            )
        self._fingerprints = {}
        # run_cv 只加载不训练：不能带着之前某次 train 模式 run() 的记录（WR-04）。
        self._trained_checkpoint = None

        folds = self._select_folds(self._read_cv_folds())
        calendar = self._price_calendar(folds[-1]["test_end"])
        self._assert_contiguous_folds(folds, calendar)

        records: list[dict] = []
        for fold in folds:
            # 在项目目录下解析，而不是按进程工作目录（代码审查 WR-03）；解析后的
            # 路径也就是逐折记录与 metrics.json 里落盘的 checkpoint。
            fold["checkpoint"] = self._resolve_fold_checkpoint(fold["checkpoint"])
            saved = self._load_model_checkpoint(fold["checkpoint"])
            # D-16 以清单的折日期为准。checkpoint 自己记的训练日期只与清单核对
            # （代码审查 WR-01、G-03.7-7）：两者本应出自同一次 train_cv。每折的
            # 训练段本来就各不相同，所以这里从不与回测模型配置里的日期比较；
            # 比较的是两对日期在价格日历上选中的 bar，不是文本。
            recorded = self._recorded_train_bounds(saved)
            manifest_bounds = fold["_train_bounds"]
            if recorded is not None and not self._same_training_bars(
                calendar, recorded, manifest_bounds
            ):
                logger.warning(
                    f"{self.class_name}: fold {fold['fold']} checkpoint "
                    f"{fold['checkpoint']} records training dates "
                    f"{recorded[0]}..{recorded[1]}, but the manifest says "
                    f"{manifest_bounds[0]}..{manifest_bounds[1]}; using the "
                    f"manifest's dates (D-16, WR-01)"
                )
            window = self._backtest_window(
                fold["test_start"],
                fold["test_end"],
                calendar,
                *fold["_train_bounds"],
            )
            records.append(
                {
                    **{key: fold[key] for key in self._CV_RECORD_KEYS},
                    "predictions": window.predictions,
                    "weights": window.weights,
                    "simulation": window.simulation,
                    "metrics": window.metrics,
                }
            )

        # D-35：拼接权重，对整段拼接价格跑一次连续模拟（资金不在折边界重置）。
        first_start = folds[0]["test_start"]
        last_end = folds[-1]["test_end"]
        stitched_weights = xr.concat(
            [record["weights"] for record in records], dim="timestamp"
        )

        # D-27：逐折记录的只是最后一折的窗口。重置后把因子定到整个拼接窗口
        # （含首折预热）重读并记录，价格指纹取自拼接价格，然后比对。
        self._fingerprints = {}
        self._redate_factors(first_start, last_end, calendar)
        stitched_prices = self._load_prices(first_start, last_end)
        self._compare_fingerprints()

        if not np.array_equal(
            stitched_weights.timestamp.values.astype("datetime64[ns]"),
            stitched_prices.timestamp.values.astype("datetime64[ns]"),
        ):
            raise ValueError(
                f"{self.class_name}: the concatenated fold weights do not cover "
                f"exactly the price bars {first_start}..{last_end}"
            )
        self._assert_weights_contract(stitched_weights, stitched_prices)
        stitched_simulation = self._simulate(stitched_weights, stitched_prices)
        stitched_metrics = self._compute_metrics(
            stitched_simulation,
            self._simulate_benchmark(first_start, last_end),
            self._stitched_split(stitched_prices.timestamp.values, records),
        )

        notes = self._report_notes() + [
            f"run_cv: the stitched curve is one continuous simulation over folds "
            f"{[fold['fold'] for fold in folds]} ({first_start}..{last_end}), "
            f"capital carried across fold boundaries; per-fold metrics come from "
            f"separate per-fold simulations. Each fold's first label-horizon "
            f"bars are in-sample (metrics stitched.in_sample_ranges) and are not "
            f"shaded in this report."
        ]
        metrics = {
            "stitched": stitched_metrics,
            "folds": [
                {
                    **{key: record[key] for key in self._CV_RECORD_KEYS},
                    "metrics": record["metrics"],
                }
                for record in records
            ],
            "notes": notes,
        }
        run_dir = self._persist_cv(
            records, stitched_weights, stitched_simulation, metrics
        )
        # wandb 默认关闭（D-28）；打开时记的是拼接曲线的指标。
        if self.config.use_wandb:
            self._log_to_wandb(run_dir, stitched_metrics)

        return CVBacktestResult(
            run_dir=run_dir,
            folds=records,
            weights=stitched_weights,
            simulation=stitched_simulation,
            metrics=metrics,
        )

    #: 每折记录与 metrics.json 里逐折条目共有的清单字段。
    _CV_RECORD_KEYS = (
        "fold",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "checkpoint",
    )

    def _read_cv_folds(self) -> list[dict]:
        """读 `cv_project_dir` 下的折清单并校验（D-36），按 `fold` 排序返回。

        - 文件不存在：FileNotFoundError，写明路径；
        - 没有 `format_version`，或不等于 `BaseModel.CV_FOLDS_FORMAT_VERSION`：
          ValueError，写明读到的值与支持的版本。清单是持久化格式，猜着读一个
          不认识的版本会悄悄读错旧（或将来）的训练 run；
        - `folds` 不是非空 list：ValueError；
        - 某折缺清单字段，或测试段起点晚于终点：ValueError，写明折号。

        每折的四个日期经 `_iso_date` 规范；返回的是新 dict，不改动读到的对象。
        """
        path = Path(self.config.cv_project_dir) / BaseModel.CV_FOLDS_FILENAME  # type: ignore[arg-type]
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.class_name}: CV fold manifest {path} does not exist; "
                f"cv_project_dir must be the project directory a train_cv run "
                f"wrote"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        supported = BaseModel.CV_FOLDS_FORMAT_VERSION
        if not isinstance(payload, dict) or "format_version" not in payload:
            raise ValueError(
                f"{self.class_name}: {path} has no format_version (supported: "
                f"{supported}); it is not a cv_folds manifest this reader "
                f"understands (D-36)"
            )
        version = payload["format_version"]
        if isinstance(version, bool) or version != supported:
            raise ValueError(
                f"{self.class_name}: {path} format_version {version!r} is not "
                f"supported (supported: {supported}) (D-36)"
            )
        raw_folds = payload.get("folds")
        if not isinstance(raw_folds, list):
            raise ValueError(
                f"{self.class_name}: {path} 'folds' must be a list, got "
                f"{type(raw_folds).__name__}"
            )
        if not raw_folds:
            raise ValueError(
                f"{self.class_name}: {path} lists no folds; the train_cv run "
                f"produced no fold to backtest"
            )

        folds = []
        for entry in raw_folds:
            missing = [
                key
                for key in self._CV_RECORD_KEYS
                if not isinstance(entry, dict) or key not in entry
            ]
            if missing:
                raise ValueError(
                    f"{self.class_name}: {path} fold entry "
                    f"{entry.get('fold') if isinstance(entry, dict) else entry!r} "
                    f"is missing {missing}"
                )
            fold = dict(entry)
            # 训练段端点原样保留一份给 `_training_window`（代码审查 CR-01）：
            # 清单里是 `np.datetime_as_string` 的纳秒字符串，模型层按它精确切片；
            # 截成日期后日内折的整个 train_end 当天都会被当成训练段。
            fold["_train_bounds"] = (entry["train_start"], entry["train_end"])
            for key in ("train_start", "train_end", "test_start", "test_end"):
                fold[key] = self._iso_date(fold[key])
            if fold["test_start"] > fold["test_end"]:
                raise ValueError(
                    f"{self.class_name}: {path} fold {fold['fold']} test segment "
                    f"starts {fold['test_start']} after it ends {fold['test_end']}"
                )
            folds.append(fold)
        return sorted(folds, key=lambda fold: fold["fold"])

    def _resolve_fold_checkpoint(self, recorded) -> str:
        """清单里一折的 checkpoint 记录 -> 实际要加载的文件（代码审查 WR-03）。

        `train_cv` 的布局固定是 `{cv_project_dir}/{experiment_name}/{model_name}`，
        所以先在 `cv_project_dir` 下按记录路径的**最后两段**找。项目目录整体
        搬走了、或者清单里是相对 `model_save_dir` 写出的相对路径，都能找到，
        而且与进程的工作目录无关。

        找不到时，再接受记录里本身存在的**绝对**路径（比如只把清单拷到别处，
        checkpoint 仍在原处）。相对路径从不按当前工作目录解析：那样换个目录
        运行会找不到，更糟的是会悄悄加载工作目录下另一次训练的同名
        checkpoint。两处都没有时 FileNotFoundError，写明两个候选路径。
        """
        project_dir = Path(self.config.cv_project_dir)  # type: ignore[arg-type]
        recorded_path = Path(str(recorded))
        in_project = project_dir / recorded_path.parent.name / recorded_path.name
        if in_project.is_file():
            return str(in_project)
        if recorded_path.is_absolute() and recorded_path.is_file():
            return str(recorded_path)
        raise FileNotFoundError(
            f"{self.class_name}: fold checkpoint {recorded!r} was found neither "
            f"inside cv_project_dir as {in_project} nor as an existing absolute "
            f"path; relative manifest entries are never resolved against the "
            f"working directory (WR-03)"
        )

    def _select_folds(self, folds: list[dict]) -> list[dict]:
        """只留测试段整段落在 `[config.start_date, config.end_date]` 内的折。

        一个也不剩时 ValueError，写明回测窗口与清单测试段的覆盖范围。ISO 日期
        字符串的字典序就是时间序。
        """
        start = self._iso_date(self.config.start_date)
        end = self._iso_date(self.config.end_date)
        selected = [
            fold
            for fold in folds
            if fold["test_start"] >= start and fold["test_end"] <= end
        ]
        if not selected:
            raise ValueError(
                f"{self.class_name}: no fold's test segment lies within the "
                f"backtest window {start}..{end}; the manifest's test segments "
                f"span {folds[0]['test_start']}..{folds[-1]['test_end']}"
            )
        if len(selected) < len(folds):
            logger.info(
                f"{self.class_name}: run_cv backtests folds "
                f"{[fold['fold'] for fold in selected]} of "
                f"{[fold['fold'] for fold in folds]} (window {start}..{end})"
            )
        return selected

    def _assert_contiguous_folds(self, folds: list[dict], calendar: np.ndarray) -> None:
        """在价格日历上断言各折测试段首尾相接、互不重叠（D-35）。

        每折测试段在日历上的首 bar 是第一个不早于 `test_start` 的 bar，末 bar
        是最后一个不晚于 `test_end` 当天的 bar。相邻两折必须满足「后一折首 bar =
        前一折末 bar + 1」：
        - 更大：缺口，中间的 bar 不属于任何折，拼接曲线会悄悄跳过它们；
        - 更小或相等：重叠，同一批 bar 会被两个模型各交易一次。
        两种情况都 ValueError，写明两折的折号与交界处的两个日期。某折测试段在
        日历上没有 bar 同样报错。
        """
        cal = np.asarray(calendar).astype("datetime64[ns]")
        one_day = np.timedelta64(1, "D")

        def _span(fold: dict) -> tuple[int, int]:
            day_start = np.datetime64(fold["test_start"], "D").astype("datetime64[ns]")
            day_after_end = (np.datetime64(fold["test_end"], "D") + one_day).astype(
                "datetime64[ns]"
            )
            first = int(np.searchsorted(cal, day_start, side="left"))
            last = int(np.searchsorted(cal, day_after_end, side="left")) - 1
            if first > last:
                raise ValueError(
                    f"{self.class_name}: fold {fold['fold']} test segment "
                    f"{fold['test_start']}..{fold['test_end']} has no price bars "
                    f"on the price calendar"
                )
            return first, last

        previous = folds[0]
        _, previous_last = _span(previous)
        for fold in folds[1:]:
            first, last = _span(fold)
            expected = previous_last + 1
            if first > expected:
                raise ValueError(
                    f"{self.class_name}: fold test segments are not contiguous: "
                    f"gap between fold {previous['fold']} ending "
                    f"{previous['test_end']} and fold {fold['fold']} starting "
                    f"{fold['test_start']}; {first - expected} price bar(s) in "
                    f"between belong to no fold, so a stitched out-of-sample "
                    f"curve would silently skip them (D-35)"
                )
            if first < expected:
                raise ValueError(
                    f"{self.class_name}: fold test segments overlap: fold "
                    f"{fold['fold']} starts {fold['test_start']}, on or before "
                    f"fold {previous['fold']} ends {previous['test_end']}; "
                    f"{expected - first} price bar(s) would be traded by two "
                    f"models (D-35)"
                )
            previous, previous_last = fold, last

    def _backtest_window(
        self,
        start_date: str,
        end_date: str,
        calendar: np.ndarray,
        train_start,
        train_end,
    ) -> _BacktestWindow:
        """一个回测窗口的全部步骤，不落盘；`run()` 与 `run_cv()` 的每折共用。

        对齐因子日期并预测 -> 读价格 -> 预测铺到价格轴（D-06）-> 按
        `[train_start, train_end + 标签期限]` 划分样本内外（D-17）-> 生成信号 ->
        权重契约 -> 模拟 -> 基准 -> 指标。调用前模型必须已经准备好。
        """
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

        split = self._split_window(
            prices.timestamp.values,
            self._training_window(calendar, train_start, train_end),
        )

        weights = self._generate_signals(predictions, prices)
        self._assert_weights_contract(weights, prices)

        simulation = self._simulate(weights, prices)
        benchmark = self._simulate_benchmark(start_date, end_date)
        metrics = self._compute_metrics(simulation, benchmark, split)
        return _BacktestWindow(
            predictions=predictions,
            prices=prices,
            weights=weights,
            simulation=simulation,
            split=split,
            metrics=metrics,
        )

    def _prepare_model(self) -> tuple:
        """按 `model_mode` 准备模型（D-13），返回 D-17 用的训练段端点 `(train_start, train_end)`。

        - `train`：`collect()` 再 `train()`，用的是**模型自己**配置里的
          train/test 日期。回测窗口从不写进这些日期：回测窗口只决定预测区间和
          样本内/外的划分，改写它们会让训练集跟着回测参数漂移。返回模型配置里
          的两个日期。
        - `load`：`_load_model_checkpoint(config.checkpoint)`。checkpoint 旁
          `config.json` 记录的才是这个模型真正训练过的日期（代码审查 WR-01），
          所以有记录（`_recorded_train_bounds`）时返回记录的日期，没有时返回
          `config.model` 的日期。
        """
        model = self.config.model
        if self.config.model_mode == "load":
            saved = self._load_model_checkpoint(self.config.checkpoint)
            recorded = self._recorded_train_bounds(saved)
            if recorded is not None:
                return recorded
            return model.config.train_start, model.config.train_end
        model.collect()
        # 训练段数据的指纹（代码审查 WR-05）：collect 刚读完、train 之前记录。
        self._record_training_fingerprints()
        # 记下训练出的 checkpoint（代码审查 WR-04），config.json 与 metrics.json 据此记录。
        self._trained_checkpoint = str(model.train())
        return model.config.train_start, model.config.train_end

    def _warn_if_config_model_dates_differ(self, calendar, train_bounds: tuple) -> None:
        """`run()` 的 load 模式：`train_bounds` 与 `config.model` 的训练日期选中的 bar 不同时 warning。

        `config.model` 的日期可能是手工重建模型时写的、或者早就过时的：拿它们
        判定样本内，会把真正训练过的 bar 悄悄算成样本外（代码审查 WR-01）。
        所以 D-17 用 checkpoint 记录的日期，两者不同时写明两对日期与
        checkpoint 路径。比较走 `_same_training_bars`（G-03.7-7），同一段训练
        bar 换一种文本写法不算不同。

        checkpoint 没有记录时 `train_bounds` 就是 `config.model` 自己的日期，
        两对原样相同，不会 warning。只在 `run()` 调用：`run_cv` 每折只与清单比较。
        """
        model = self.config.model
        configured = (model.config.train_start, model.config.train_end)
        if self._same_training_bars(calendar, train_bounds, configured):
            return
        logger.warning(
            f"{self.class_name}: checkpoint {self.config.checkpoint} was trained on "
            f"{train_bounds[0]}..{train_bounds[1]} (its config.json), but config.model "
            f"says train_start={configured[0]!r}, train_end={configured[1]!r}; "
            f"using the checkpoint's dates for the effective training window "
            f"(D-17, WR-01)"
        )

    @staticmethod
    def _recorded_train_bounds(saved: dict | None) -> tuple | None:
        """checkpoint 旁 `config.json` 记录的 `(train_start, train_end)`；缺任一个时 None。

        纯读取：不 warning，也不看 `config.model`。`_save_model` 在训练时写下
        这两个日期，它们是这个 checkpoint 真正用过的训练段（代码审查 WR-01）。
        与谁比较、不同时怎么办由调用方决定：`run()` 的 load 模式与
        `config.model` 比，`run_cv` 每折只与清单比（G-03.7-7）。
        """
        if not isinstance(saved, dict):
            return None
        if saved.get("train_start") is None or saved.get("train_end") is None:
            return None
        return saved["train_start"], saved["train_end"]

    @classmethod
    def _same_training_bars(cls, calendar, a: tuple, b: tuple) -> bool:
        """两对 `(train_start, train_end)` 是否选中价格日历上同一段训练 bar（G-03.7-7）。

        所有训练日期的一致性判定都走这里，不比较日期的文本。同一个时刻可以写成
        `"2024-01-01"`、`"2024-01-01T00:00:00"` 或 `np.datetime_as_string` 的
        纳秒字符串；按文本比较会把它们判成不同，每次正常的 `run_cv` 都会
        误报。

        也不用裸的 `pd.Timestamp` 相等。模型层按
        `data.sel(timestamp=slice(train_start, train_end))` 训练，字符串端点
        按分辨率解释（代码审查 CR-01）：日内数据上 `"2024-02-09"` 包含当天全部
        bar，而纳秒字符串的午夜只到前一个交易日的最后一个 bar。两者 Timestamp
        相等，训练段却不同。所以这里与 `_training_window` 共用同一个 pandas
        `slice_indexer` 和 `_slice_bound`，比较两对端点选中的 bar 位置。

        规则：
        - 两对原样相同：相同；
        - 否则任一端点为 None：不同；
        - 日历为空：两对端点逐个 `pd.Timestamp` 相等才相同；
        - 否则两对选中的 (start, stop) bar 位置必须相同，并且落在日历首尾 bar
          之外的端点必须与对应端点 `pd.Timestamp` 相等。`run()` 的日历只到
          回测 `end_date`，两个都晚于它的不同 `train_end` 会截到同一个 stop；
          没有这一条，过时的日期就不会被发现。

        结果只决定是否 warning，从不决定用哪对日期。无法解析的日期文本由
        pandas 直接报错，不会被悄悄当成相同。
        """
        if tuple(a) == tuple(b):
            return True
        if any(value is None for value in (*a, *b)):
            return False
        a_ts = tuple(pd.Timestamp(str(value)) for value in a)
        b_ts = tuple(pd.Timestamp(str(value)) for value in b)
        bars = np.sort(np.asarray(calendar).astype("datetime64[ns]"))
        if bars.size == 0:
            return a_ts == b_ts

        index = pd.DatetimeIndex(bars)
        a_slice = index.slice_indexer(cls._slice_bound(a[0]), cls._slice_bound(a[1]))
        b_slice = index.slice_indexer(cls._slice_bound(b[0]), cls._slice_bound(b[1]))
        if (a_slice.start, a_slice.stop) != (b_slice.start, b_slice.stop):
            return False

        first, last = index[0], index[-1]
        for x, y in zip(a_ts, b_ts):
            outside = not (first <= x <= last) or not (first <= y <= last)
            if outside and x != y:
                return False
        return True

    def _load_model_checkpoint(self, checkpoint) -> dict | None:
        """把 `checkpoint` 加载进 `config.model`；`run()` 与 `run_cv()` 的每折共用。

        1. 先检查 checkpoint 文件存在，缺了直接报错并写明路径。这一步必须在
           任何特征计算之前，否则一个拼错的路径要白算一遍特征才暴露。
        2. 读 checkpoint 旁的 `config.json`（`_read_checkpoint_config`），核对
           因子与标签的变量名和顺序（`_assert_checkpoint_variables`，代码审查
           WR-01）。同样先于任何特征计算。
        3. 只对 `DLModel`，先把特征面板放进模型的 data backend。
           `DLModel._read_checkpoint` 用 `num_symbols` 重建网络，而
           `num_symbols` 读的正是这个 backend，空着就会在 `load()` 里报错
           （03.7-RESEARCH.md Pitfall 11）。`MLModel` 的 checkpoint 就是完整
           模型，不调 `_init_model`，所以跳过这一步。
        4. `model.load(checkpoint)`。

        返回读到的 `config.json`（没有时 None），调用方从中取训练日期。
        """
        model = self.config.model
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.class_name}: checkpoint {path} does not exist"
            )
        saved = self._read_checkpoint_config(path)
        if saved is not None:
            self._assert_checkpoint_variables(saved, path)
        if isinstance(model, DLModel):
            model.data_backend.to_internal(model._collect_all_features())
        model.load(path)
        return saved

    def _read_checkpoint_config(self, path: Path) -> dict | None:
        """checkpoint 旁由 `_save_model` 写下的 `config.json`；没有时 warning 并返回 None。

        没有它就无从核对因子、标签与训练日期，只能照 `config.model` 原样使用。
        这不一定是错（比如手工拷贝的 checkpoint），所以只 warning 不中断。
        """
        sidecar = path.parent / "config.json"
        if not sidecar.is_file():
            logger.warning(
                f"{self.class_name}: checkpoint {path} has no config.json beside "
                f"it, so its factors, labels and training dates cannot be checked "
                f"against config.model (WR-01); continuing with config.model as "
                f"given"
            )
            return None
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError(
                f"{self.class_name}: {sidecar} is not a model config object"
            )
        return saved

    @staticmethod
    def _saved_variable_names(entries) -> list[str] | None:
        """`config.json` 里一组因子（或标签）配置按顺序展开的变量名；记录不全时 None。"""
        if not isinstance(entries, list):
            return None
        names: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("factor_names") is None:
                return None
            names.extend(str(name) for name in entry["factor_names"])
        return names

    def _assert_checkpoint_variables(self, saved: dict, path: Path) -> None:
        """checkpoint 训练时的因子与标签变量（名字与顺序）必须与 `config.model` 一致。

        模型只按位置消费特征：xgboost 只核对特征**个数**，DL 头连个数都只在
        第一层才核对。一个用别的因子集、或同样因子不同顺序训练出来的
        checkpoint，会对错位的特征悄悄给出预测（代码审查 WR-01）。标签决定
        预测变量的名字与 `score_label` 的含义，同理。不一致时 ValueError，写明
        两边的变量名与 checkpoint 路径；记录不全（旧的或手写的 `config.json`）
        时对该类变量 warning 并跳过核对。
        """
        model = self.config.model
        for kind, entries, declared in (
            ("factor", saved.get("factors"), model.get_factor_names()),
            ("label", saved.get("labels"), model.get_label_names()),
        ):
            recorded = self._saved_variable_names(entries)
            current = [str(name) for name in declared]
            if recorded is None:
                logger.warning(
                    f"{self.class_name}: {path.parent / 'config.json'} does not "
                    f"record the {kind} variable names, so they cannot be checked "
                    f"against config.model (WR-01)"
                )
                continue
            if recorded != current:
                raise ValueError(
                    f"{self.class_name}: checkpoint {path} was trained on {kind} "
                    f"variables {recorded}, but config.model declares {current}; "
                    f"predicting with it would feed the model different or "
                    f"permuted inputs (WR-01)"
                )

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

    def _redate_factors(
        self, start_date: str, end_date: str, calendar: np.ndarray
    ) -> None:
        """把每个因子的日期定到「预热起点 .. end_date」、强制重读，并记录因子指纹。

        预热按 bar 计（`_warmup_start`，D-15），重读绕过读缓存（D-14），重读后
        立刻记录指纹：此时数据集持有的正是「预热 + 窗口」（D-27）。`run()` 的
        窗口与 `run_cv()` 的每折、以及拼接窗口的指纹都经过这里。
        """
        warmup = self._warmup_start(calendar, start_date)
        for factor in self.config.model.config.factors:
            factor.config.start_date = warmup
            factor.config.end_date = end_date
            factor._reset_dataset_config()
            self._refresh_factor_reads(factor)
        self._record_factor_fingerprints()

    def _align_and_predict(
        self, start_date: str, end_date: str, calendar: np.ndarray
    ) -> xr.Dataset:
        """改因子配置日期（含预热）-> 只算特征 -> 预测 -> 切回回测窗口（D-14）。"""
        model = self.config.model
        self._redate_factors(start_date, end_date, calendar)

        features = model._collect_all_features()
        return model.predict_panel(features).sel(
            timestamp=slice(start_date, end_date)
        )

    def _load_prices(self, start_date: str, end_date: str) -> xr.Dataset:
        """回测窗口内的成交价与估值价两列，深拷贝后返回，并记录价格数据集的指纹。

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
        prices = ds[[fill, valuation]].load().copy(deep=True)
        self._record_price_fingerprint(prices)
        return prices

    def _record_price_fingerprint(self, prices: xr.Dataset) -> None:
        """价格数据集的指纹，键 `price_dataset`，只覆盖成交价与估值价两列（D-27）。"""
        columns = [
            self.MARKET.fill_price_column,  # type: ignore[union-attr]
            self.MARKET.valuation_price_column,  # type: ignore[union-attr]
        ]
        self._fingerprints["price_dataset"] = dataset_fingerprint(prices, columns)

    @staticmethod
    def _dataset_variables_fingerprint(factor) -> dict:
        """一个因子（或标签）背后数据集当前持有范围的指纹，覆盖它真正消费的列。

        KunQuant 因子（`FactorConfig`）是 `data_columns`；Polars 因子消费整个
        lazyframe，所以是数据集的全部数据变量。
        """
        ds = factor.config.dataset.get_xarray_dataset()
        if isinstance(factor.config, FactorConfig):
            variables = list(factor.config.data_columns)
        else:
            variables = list(ds.data_vars)
        return dataset_fingerprint(ds, variables)

    @staticmethod
    def _store_fingerprint(ds: xr.Dataset) -> dict:
        """因子库或标签库读出的面板（`get_features()` / `get_labels()`）的指纹，覆盖全部变量。"""
        return dataset_fingerprint(ds, list(ds.data_vars))

    def _record_factor_fingerprints(self) -> None:
        """每个因子背后数据集的指纹，键 `factor[{i}]:{类名}`（D-27）。

        覆盖的变量口径见 `_dataset_variables_fingerprint`。时间范围是因子数据集
        当前持有的范围，调用时机保证它包含预热。

        `factor_data_strategy == "read"` 时再记一个键 `factor_store[{i}]:{类名}`
        （代码审查 WR-05）。read 策略的特征来自**因子库**（`factor.read`），不是
        数据集，而以前只给数据集算指纹：因子库被重算或改动后，预测和权重都变了，
        指纹却照样匹配、没有任何警告，这正是 D-27 要抓的情况。这个键覆盖
        `factor.get_features()` 的全部变量，调用时机保证因子库已按「预热 + 窗口」
        重读。
        """
        strategy = self.config.model.config.factor_data_strategy
        for i, factor in enumerate(self.config.model.config.factors):
            name = type(factor).__name__
            self._fingerprints[f"factor[{i}]:{name}"] = (
                self._dataset_variables_fingerprint(factor)
            )
            if strategy == "read":
                self._fingerprints[f"factor_store[{i}]:{name}"] = (
                    self._store_fingerprint(factor.get_features())
                )

    def _record_training_fingerprints(self) -> None:
        """train 模式训练所用数据的指纹；在 `collect()` 之后、`train()` 之前调用（代码审查 WR-05）。

        回测窗口的指纹（价格窗口、因子的「预热 + 窗口」）不含训练段。训练窗口里
        的一次回溯重算复权会让重建时训练出另一个模型，以前不会有任何警告。所以
        对模型的每个因子与标签各记一个键，范围是 `collect()` 刚读过的范围（模型
        自己的 start_date..end_date，另含因子自带的日历日缓冲）：

        - `cal` 策略：`train_factor[{i}]:{类名}` / `train_label[{i}]:{类名}`，
          覆盖背后数据集真正消费的列（`_dataset_variables_fingerprint`）；
        - `read` 策略：`train_factor_store[{i}]:{类名}` /
          `train_label_store[{i}]:{类名}`，覆盖因子库 / 标签库读出的全部变量。
          read 策略训练时不读数据集，只有库是训练真正用到的数据。
        """
        model_config = self.config.model.config
        for prefix, items, strategy, getter in (
            ("train_factor", model_config.factors, model_config.factor_data_strategy, "get_features"),
            ("train_label", model_config.labels, model_config.label_data_strategy, "get_labels"),
        ):
            for i, item in enumerate(items):
                name = type(item).__name__
                if strategy == "read":
                    self._fingerprints[f"{prefix}_store[{i}]:{name}"] = (
                        self._store_fingerprint(getattr(item, getter)())
                    )
                else:
                    self._fingerprints[f"{prefix}[{i}]:{name}"] = (
                        self._dataset_variables_fingerprint(item)
                    )

    def _compare_fingerprints(self) -> None:
        """与 `expected_fingerprint` 比对本次记录的指纹（D-27）；只 warning，不中断。

        只在 `expected_fingerprint` 不为 None 时比对。某个键只出现在一侧，或
        `digest` / `start` / `end` / `n_timestamps` / `n_symbols` 任一不同，都对
        该键发一条 warning，写明键名和不同的字段。数据集会被追加，Tiingo 也会在
        新分红后回溯重算复权价，所以重建出来的回测必须能察觉数据变了，而不是
        悄悄得出不同的结果；但变了的数据仍然可以回测，所以继续运行。
        """
        expected = self.expected_fingerprint
        if expected is None:
            return
        actual = to_jsonable(self._fingerprints)
        for key in sorted(set(expected) | set(actual)):  # type: ignore[arg-type]
            if key not in actual:
                logger.warning(
                    f"{self.class_name}: data fingerprint mismatch for {key!r}: "
                    f"present in expected_fingerprint but not read by this run "
                    f"(D-27); continuing"
                )
                continue
            if key not in expected:
                logger.warning(
                    f"{self.class_name}: data fingerprint mismatch for {key!r}: "
                    f"read by this run but absent from expected_fingerprint "
                    f"(D-27); continuing"
                )
                continue
            wanted, got = expected[key], actual[key]
            differing = [
                name
                for name in FINGERPRINT_COMPARED_FIELDS
                if wanted.get(name) != got.get(name)
            ]
            if differing:
                details = "; ".join(
                    f"{name}: expected {wanted.get(name)!r}, got {got.get(name)!r}"
                    for name in differing
                )
                logger.warning(
                    f"{self.class_name}: data fingerprint mismatch for {key!r} "
                    f"(differing fields: {', '.join(differing)}): {details}. The "
                    f"data changed since the expected run (D-27); continuing"
                )

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

    @abstractmethod
    def _period_returns_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """只看 `ranges` 内收益的收益类统计（D-34）。

        组合对象不能按时间切片（03.7-RESEARCH.md Pitfall 5），重新模拟一段又会
        重置资金、改变路径，所以切片统计只能取**同一次**连续模拟的收益序列，
        截到 `ranges`（ISO 日期对，含两端）后交给引擎的收益统计。多段时把各段
        收益按时间顺序拼起来算。
        """

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
        """模型的有效训练窗口 `[train_start, train_end + 标签期限]`（D-17），bar 标签对。

        期限在价格日历上按 bar 数，不做日历日加法：周五的 `train_end` 加 2 个
        bar 是下周二。
        - 训练段：日历上与模型层 `data.sel(timestamp=slice(train_start, train_end))`
          **同一个** pandas `slice_indexer` 选中的 bar。端点原样交过去
          （`_slice_bound`），所以 `"2024-05-17"` 包含当天全部 bar，
          `"2024-05-17T13:00"` 只到 13:00，纳秒精度的折日期精确匹配；
        - 起点：训练段的第一个 bar；
        - 终点：训练段的最后一个 bar 再往后数期限个 bar，超出日历时截到最后一个 bar。

        返回的两端经 `_bar_label`：日线是日期，日内保留时刻。

        **代码审查 CR-01 更正。** 以前先把 `train_end` 截成当天午夜再找「最后一个
        不晚于它的 bar」。日内数据上那一天的 bar 全都晚于午夜，于是落到**前一个
        交易日**的最后一个 bar 再加期限，终点又被截成日期、按天比较：`train_end`
        当天全算样本内，而真正读过训练标签的、落在下一个交易日的那期限个 bar
        却被算成样本外，污染样本外指标。日线的 bar 就在午夜，不受影响。

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

        calendar = np.sort(np.asarray(calendar).astype("datetime64[ns]"))
        last = calendar.size - 1
        trained = pd.DatetimeIndex(calendar).slice_indexer(
            self._slice_bound(train_start), self._slice_bound(train_end)
        )
        start_idx = int(trained.start)
        end_idx = int(trained.stop) - 1 + self._label_horizon_bars()
        window_start = (
            self._bar_label(calendar[start_idx])
            if start_idx <= last
            else self._bar_label(train_start)
        )
        window_end = (
            self._bar_label(calendar[min(end_idx, last)])
            if end_idx >= 0
            else self._bar_label(train_end)
        )
        return window_start, window_end

    def _split_window(
        self, window_timestamps: np.ndarray, training_window: tuple[str, str] | None
    ) -> dict:
        """把回测窗口的 bar 切成样本内与样本外（D-17）。

        返回三个键，原样并入 metrics 顶层：
        - `training_window`：有效训练窗口的 bar 标签对（`_bar_label`），或 None；
        - `in_sample_range`：回测窗口与训练窗口重叠部分的首尾 bar，或 None；
        - `out_of_sample_ranges`：重叠之外的 bar 组成的连续段，0、1 或 2 段。

        比较按**精确的 bar 时间戳**做（代码审查 CR-01）：以前按日期比较，日内
        数据上 `train_end` 所在那一天整天算样本内，而下一天读过训练标签的 bar
        算样本外。两个窗口都是区间，所以重叠部分一定连续。重叠非空时 warning
        写明两个窗口，并说明样本内外分开报告；回测照常继续。
        """
        timestamps = np.asarray(window_timestamps).astype("datetime64[ns]")
        split = {
            "training_window": training_window,
            "in_sample_range": None,
            "out_of_sample_ranges": [],
        }
        if timestamps.size == 0:
            return split

        if training_window is None:
            in_sample = np.zeros(timestamps.size, dtype=bool)
        else:
            first = self._label_ns(training_window[0])
            last = self._label_ns(training_window[1])
            in_sample = (timestamps >= first) & (timestamps <= last)

        pieces = []
        if in_sample.any():
            idx = np.flatnonzero(in_sample)
            lo, hi = int(idx[0]), int(idx[-1])
            split["in_sample_range"] = (
                self._bar_label(timestamps[lo]),
                self._bar_label(timestamps[hi]),
            )
            if lo > 0:
                pieces.append((0, lo - 1))
            if hi < timestamps.size - 1:
                pieces.append((hi + 1, timestamps.size - 1))
            window = (self._bar_label(timestamps[0]), self._bar_label(timestamps[-1]))
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
            (self._bar_label(timestamps[a]), self._bar_label(timestamps[b]))
            for a, b in pieces
        ]
        return split

    @classmethod
    def _in_ranges(cls, timestamps: np.ndarray, ranges: list[tuple[str, str]]) -> np.ndarray:
        """`timestamps` 中落在任一 bar 标签对内（精确时间戳、含两端）的布尔掩码。

        标签是 `_bar_label` 写出的端点，日期即午夜。不按天比较（代码审查
        CR-01）：日内数据上一个午夜 bar 的标签按天比较会把当天其余 bar 也算进来。
        """
        ts = np.asarray(timestamps).astype("datetime64[ns]")
        mask = np.zeros(ts.size, dtype=bool)
        for start, end in ranges:
            mask |= (ts >= cls._label_ns(start)) & (ts <= cls._label_ns(end))
        return mask

    def _turnover(self, simulation: SimulationResult) -> xr.DataArray:
        """每个有成交的 bar 的换手率，维度 `timestamp`（D-22，口径由本方法定义）。

        换手率 = 该 bar 所有订单的单边成交额之和（`|size| x 成交价`）/ 上一个
        bar 的组合净值；窗口第一个 bar 没有上一个 bar，用 `config.init_cash`。
        单边口径：从空仓全仓买入约为 1，整个组合换成另一批标的（先卖后买）约为 2。
        分母取成交前的净值而不是成交 bar 的净值，这样换手率不含成交当 bar 的盈亏。
        没有订单时返回空数组。
        """
        orders = simulation.orders
        if orders.sizes.get("order", 0) == 0:
            return xr.DataArray(
                np.array([], dtype=np.float64),
                dims=("timestamp",),
                coords={"timestamp": np.array([], dtype="datetime64[ns]")},
            )

        order_ts = orders["timestamp"].values.astype("datetime64[ns]")
        notional = np.abs(orders["size"].values.astype(np.float64)) * orders[
            "price"
        ].values.astype(np.float64)
        fill_bars, inverse = np.unique(order_ts, return_inverse=True)
        traded = np.zeros(fill_bars.size, dtype=np.float64)
        np.add.at(traded, inverse, notional)

        value_ts = simulation.value.timestamp.values.astype("datetime64[ns]")
        idx = np.searchsorted(value_ts, fill_bars)
        if (idx >= value_ts.size).any() or not np.array_equal(
            value_ts[np.minimum(idx, value_ts.size - 1)], fill_bars
        ):
            raise ValueError(
                f"{self.class_name}: an order timestamp is not on the equity "
                f"timestamp axis"
            )
        values = np.asarray(simulation.value.values, dtype=np.float64)
        previous = np.where(
            idx > 0, values[np.maximum(idx - 1, 0)], float(self.config.init_cash)
        )
        return xr.DataArray(
            traded / previous, dims=("timestamp",), coords={"timestamp": fill_bars}
        )

    def _turnover_summary(self, turnover: xr.DataArray, bar_interval) -> dict:
        """换手率汇总：每次调仓均值、总和、年化。

        年化 = 每次调仓均值 x 每年 bar 数 / `rebalance_periods`；每年 bar 数取
        `MARKET.year_freq(bar_interval) / bar_interval`。没有成交 bar 时均值与年化
        是 NaN（落盘为 null），总和是 0。
        """
        values = np.asarray(turnover.values, dtype=np.float64)
        interval = pd.Timedelta(bar_interval)
        bars_per_year = self.MARKET.year_freq(interval) / interval  # type: ignore[union-attr]
        mean = float(values.mean()) if values.size else float("nan")
        return {
            "mean_per_rebalance": mean,
            "sum": float(values.sum()),
            "annualized": mean * bars_per_year / self.config.rebalance_periods,
        }

    def _period_record_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """按时间段过滤的订单、交易与换手统计（D-34）。

        - `order_count` / `fees_paid` / `traded_notional`：成交时间落在段内的订单；
        - `closed_trade_count`：平仓时间落在段内、状态为 Closed 的交易；
        - `open_trade_count`：段末仍未平仓的交易（入场不晚于段末，且尚未平仓
          或平仓晚于段末）；
        - `turnover`：段内成交 bar 的 `_turnover_summary`。

        多段时：订单与已平仓交易按段求和（段互不重叠，等于逐段相加），段末
        持仓数逐段相加，换手率汇总取所有段内成交 bar 的并集。
        """
        orders = simulation.orders
        if orders.sizes.get("order", 0) > 0:
            in_range = self._in_ranges(orders["timestamp"].values, ranges)
            sizes = np.abs(orders["size"].values.astype(np.float64))[in_range]
            prices = orders["price"].values.astype(np.float64)[in_range]
            fees = orders["fees"].values.astype(np.float64)[in_range]
            order_count = int(in_range.sum())
            fees_paid = float(fees.sum())
            traded_notional = float((sizes * prices).sum())
        else:
            order_count, fees_paid, traded_notional = 0, 0.0, 0.0

        trades = simulation.trades
        closed_trade_count = open_trade_count = 0
        if trades is not None and trades.sizes.get("trade", 0) > 0:
            status = trades["status"].values.astype(str)
            closed = status == "Closed"
            closed_trade_count = int(
                (closed & self._in_ranges(trades["exit_timestamp"].values, ranges)).sum()
            )
            # 精确时间戳比较，不按天（代码审查 CR-01）：日内段末是某个 bar，
            # 段末之后同一天才入场的交易不算「段末仍未平仓」。
            entry_ts = trades["entry_timestamp"].values.astype("datetime64[ns]")
            exit_ts = trades["exit_timestamp"].values.astype("datetime64[ns]")
            for _, end in ranges:
                end_ts = self._label_ns(end)
                open_at_end = (entry_ts <= end_ts) & (~closed | (exit_ts > end_ts))
                open_trade_count += int(open_at_end.sum())

        turnover = self._turnover(simulation)
        turnover = turnover.isel(
            timestamp=self._in_ranges(turnover.timestamp.values, ranges)
        )
        return {
            "order_count": order_count,
            "fees_paid": fees_paid,
            "traded_notional": traded_notional,
            "closed_trade_count": closed_trade_count,
            "open_trade_count": open_trade_count,
            "turnover": self._turnover_summary(turnover, simulation.bar_interval),
        }

    def _compute_metrics(
        self,
        simulation: SimulationResult,
        benchmark: SimulationResult | None,
        split: dict,
    ) -> dict:
        """整段、样本内、样本外三块指标，全部取自同一次连续模拟（D-17、D-22、D-34）。

        - `whole`：引擎的整段统计（不含基准）加 `turnover` 汇总；
        - `in_sample`：样本内区间的收益统计（`_period_returns_stats`）合并按区间
          过滤的订单/交易/换手统计（`_period_record_stats`）；没有样本内区间时 None；
        - `out_of_sample`：同上，作用于样本外各段。两段时收益统计用拼接后的
          样本外收益，记录统计把两段相加；没有样本外区间时 None；
        - `benchmark`：只在有基准时出现（D-08）；
        - `split` 的每个键原样并入顶层：`run()` 是 `_split_window` 的
          `training_window` / `in_sample_range` / `out_of_sample_ranges`；
          `run_cv()` 的拼接曲线是 `_stitched_split` 的 `training_windows` /
          `in_sample_ranges` / `out_of_sample_ranges`。`split` 带
          `in_sample_ranges`（可以多段）时样本内切片用它，否则用单段的
          `in_sample_range`。

        本方法与本模块的任何路径都不会发起第二次模拟：组合不能切片，切片重算
        会重置资金、改变路径。
        """
        whole = self._engine_stats(simulation)
        whole["turnover"] = self._turnover_summary(
            self._turnover(simulation), simulation.bar_interval
        )
        metrics: dict = {"whole": whole}

        def _slice(ranges: list[tuple[str, str]]) -> dict | None:
            if not ranges:
                return None
            return {
                **self._period_returns_stats(simulation, ranges),
                **self._period_record_stats(simulation, ranges),
            }

        if "in_sample_ranges" in split:
            in_sample_ranges = list(split["in_sample_ranges"])
        else:
            in_sample_range = split["in_sample_range"]
            in_sample_ranges = [in_sample_range] if in_sample_range else []
        metrics["in_sample"] = _slice(in_sample_ranges)
        metrics["out_of_sample"] = _slice(list(split["out_of_sample_ranges"]))

        if benchmark is not None:
            metrics["benchmark"] = self._engine_stats(benchmark)
        for key, value in split.items():
            metrics[key] = value
        return metrics

    def _report_notes(self) -> list[str]:
        """报告与 metrics.json 里附带的说明（D-21）。

        默认只有一条：本阶段不模拟融券费与做空融资成本，所以空头一侧的收益偏
        乐观。模拟了借券成本的引擎覆盖本方法，去掉或改写这条说明。
        """
        return [
            "No borrow or short-financing cost is modelled, so short-side "
            "returns are optimistic."
        ]

    @classmethod
    def _flatten_numeric(cls, prefix: str, value, out: dict) -> None:
        """把嵌套 dict 里有限的数值叶子摊平成 `prefix/key` 写进 `out`。

        布尔、NaN、inf、时间、字符串都跳过：wandb summary 只收可比较的数。
        """
        if isinstance(value, dict):
            for key, item in value.items():
                cls._flatten_numeric(f"{prefix}/{key}", item, out)
            return
        if isinstance(value, (bool, np.bool_)):
            return
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value) if isinstance(value, (float, np.floating)) else int(value)
            if np.isfinite(number):
                out[prefix] = number

    def _log_to_wandb(self, run_dir: Path, metrics: dict) -> None:
        """把指标与报告记到一个单独的回测 wandb run（D-28），只在 `use_wandb` 时调用。

        - project 是 `{类名}_backtest`，run 名是运行目录名，与模型训练的 run 分开；
        - run config 是 `to_jsonable(get_config())`（含数据指纹）；
        - summary 收 `whole` / `in_sample` / `out_of_sample` 三块里有限的数值叶子，
          键形如 `whole/<指标>`，嵌套的换手率是 `whole/turnover/<键>`；
        - `report` 记 report.html 的内容，然后 finish。
        """
        run = wandb.init(
            project=f"{self.class_name}_backtest",
            name=run_dir.name,
            config=to_jsonable(self.get_config()),
        )
        summary: dict = {}
        for block in ("whole", "in_sample", "out_of_sample"):
            if metrics.get(block) is not None:
                self._flatten_numeric(block, metrics[block], summary)
        run.summary.update(summary)
        run.log({"report": wandb.Html((run_dir / "report.html").read_text())})
        run.finish()

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
        """建新的运行目录并写入 D-24 的全部产物；从不覆盖已有目录。

        config.json、weights.zarr、equity.zarr（value、returns）、
        liquidations.json、metrics.json、report.html、fingerprint.json。每个
        JSON 都先经 `to_jsonable`（NaN/inf 记为 null，时间记为 ISO 字符串）再
        原子写入。

        report.html（D-23）：净值与回撤两栏共用时间轴，样本内区间取 metrics 里
        实际算出的 `in_sample_range` 涂灰（两个区间的交集，必然是一段），并印出
        `_report_notes()`；本阶段没有基准曲线（D-08）。
        """
        # 全部产物先写进暂存目录，写完才改名成运行目录（代码审查 WR-08）。
        def _write(run_dir: Path, name: str) -> None:
            write_json_atomically(
                run_dir / "config.json", to_jsonable(self.get_config()), indent=2
            )
            self._write_weights_and_equity(run_dir, weights, simulation)
            write_json_atomically(
                run_dir / "liquidations.json",
                to_jsonable(simulation.liquidations),
                indent=2,
            )
            write_json_atomically(
                run_dir / "metrics.json", to_jsonable(metrics), indent=2
            )
            write_backtest_report(
                simulation.value,
                run_dir / "report.html",
                in_sample_range=metrics.get("in_sample_range"),
                notes=self._report_notes(),
                title=name,
            )
            write_json_atomically(
                run_dir / "fingerprint.json", to_jsonable(self._fingerprints), indent=2
            )

        return self._persist_run_dir(_write)

    def _persist_run_dir(self, write) -> Path:
        """建 `output_dir/{class}_{timestamp}/`（D-24）并写入产物；从不覆盖，也不留下半成品。

        `write(directory, name)` 往 `directory` 写全部产物，`name` 是最终的运行
        目录名（报告标题用它）。

        **先暂存，写完才改名（代码审查 WR-08）。** 以前先建运行目录、先写
        config.json，再写 zarr、指标、报告和指纹。其中任何一步失败（zarr 写错、
        plotly、磁盘满、Ctrl-C）都会留下一个带着合法 config.json、却没有指标或
        指纹的目录：看起来和跑完的一样，`load_backtester_from_config` 还会照样
        去「复现」它。现在：

        1. 最终目录已存在就 RuntimeError，从不覆盖；
        2. 在同一父目录下建隐藏的暂存目录 `.{name}.partial`，`write` 写进去；
        3. 全部写完后再确认一次最终目录不存在，然后把暂存目录改名成最终目录
           （同一文件系统上的 rename 是原子的）；
        4. 任何异常（含 KeyboardInterrupt）：删掉本次自己建的暂存目录，原样
           抛出。所以 `output_dir` 下要么是完整的运行目录，要么什么都没有。
        """
        import shutil

        final = Path(self.config.output_dir) / self._run_dir_name()
        if final.exists():
            raise RuntimeError(f"{final} already exists")
        staging = final.parent / f".{final.name}.partial"
        staging.mkdir(parents=True)
        try:
            write(staging, final.name)
            if final.exists():
                raise RuntimeError(f"{final} already exists")
            staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return final

    @staticmethod
    def _write_weights_and_equity(
        directory: Path, weights: xr.Dataset, simulation: SimulationResult
    ) -> None:
        """在 `directory` 下写 weights.zarr 与 equity.zarr（value、returns）。"""
        XrBackend().to_internal(weights).write(str(directory / "weights.zarr"))
        XrBackend().to_internal(
            xr.Dataset({"value": simulation.value, "returns": simulation.returns})
        ).write(str(directory / "equity.zarr"))

    def _stitched_split(
        self, timestamps: np.ndarray, records: list[dict]
    ) -> dict:
        """拼接曲线的样本内/外划分，由各折自己的划分拼成（D-17 逐折、D-35）。

        拼接曲线按构造是样本外的：每折只交易自己的测试段。例外是每折开头与该折
        有效训练窗口重叠的那几个 bar（标签期限），所以样本内是**多段**：
        - `training_windows`：各折的有效训练窗口，按折顺序；
        - `in_sample_ranges`：各折非空的 `in_sample_range`，按折顺序；没有任何
          一折重叠时是空 list；
        - `out_of_sample_ranges`：拼接窗口里不在任何样本内段内的 bar 组成的连续段。

        单段的 `in_sample_range` 刻意不出现：多段样本内塞不进一个日期对。
        """
        in_sample_ranges = [
            record["metrics"]["in_sample_range"]
            for record in records
            if record["metrics"]["in_sample_range"] is not None
        ]
        ts = np.asarray(timestamps).astype("datetime64[ns]")
        out_mask = ~self._in_ranges(ts, in_sample_ranges)
        pieces = []
        idx = np.flatnonzero(out_mask)
        if idx.size:
            breaks = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate(([idx[0]], idx[breaks + 1]))
            ends = np.concatenate((idx[breaks], [idx[-1]]))
            pieces = [
                (self._bar_label(ts[a]), self._bar_label(ts[b]))
                for a, b in zip(starts, ends)
            ]
        return {
            "training_windows": [
                record["metrics"]["training_window"] for record in records
            ],
            "in_sample_ranges": in_sample_ranges,
            "out_of_sample_ranges": pieces,
        }

    def _persist_cv(
        self,
        records: list[dict],
        weights: xr.Dataset,
        simulation: SimulationResult,
        metrics: dict,
    ) -> Path:
        """`run_cv()` 的运行目录（D-24 按 D-35 展开）；从不覆盖已有目录。

        顶层描述**拼接曲线**，与 `run()` 的运行目录同名同义：
        config.json、weights.zarr、equity.zarr、metrics.json（`stitched`、
        `folds`、`notes`）、liquidations.json（`stitched` 与逐折 `folds`）、
        fingerprint.json（拼接窗口）、report.html（拼接曲线，不涂样本内：多段
        样本内由 notes 说明）。每折的逐折模拟另存在 `folds/fold_{i}/` 下的
        weights.zarr 与 equity.zarr，`i` 是清单里的折号。
        """
        # 与 run() 相同：先写暂存目录，全部写完才改名（代码审查 WR-08）。
        def _write(run_dir: Path, name: str) -> None:
            write_json_atomically(
                run_dir / "config.json", to_jsonable(self.get_config()), indent=2
            )
            self._write_weights_and_equity(run_dir, weights, simulation)
            for record in records:
                fold_dir = run_dir / "folds" / f"fold_{record['fold']}"
                fold_dir.mkdir(parents=True)
                self._write_weights_and_equity(
                    fold_dir, record["weights"], record["simulation"]
                )
            write_json_atomically(
                run_dir / "liquidations.json",
                to_jsonable(
                    {
                        "stitched": simulation.liquidations,
                        "folds": [
                            {
                                "fold": record["fold"],
                                "liquidations": record["simulation"].liquidations,
                            }
                            for record in records
                        ],
                    }
                ),
                indent=2,
            )
            write_json_atomically(
                run_dir / "metrics.json", to_jsonable(metrics), indent=2
            )
            write_backtest_report(
                simulation.value,
                run_dir / "report.html",
                in_sample_range=None,
                notes=metrics["notes"],
                title=name,
            )
            write_json_atomically(
                run_dir / "fingerprint.json", to_jsonable(self._fingerprints), indent=2
            )

        return self._persist_run_dir(_write)
