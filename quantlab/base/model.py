import copy
import json
import random
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
import wandb
import wandb.sdk
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from quantlab.dataset.backend import XrBackend
from quantlab.enums.constant import Date
from quantlab.ml_model.backend import MlBackend
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.jsonable import to_jsonable
from quantlab.utils.metrics import regression_panel_metrics

from .config import DLConfig, MLConfig


class BaseModel(ABC):
    """模型层的框架无关基类，三层结构的顶层（260914-lno，2026-09-14）。

    模型层拆成三层：

    - `BaseModel`（本类）：与训练框架无关的共享生命周期。配置与日期注入、
      因子/标签收集、checkpoint 目录约定与 `config.json`、交叉验证的折几何
      （`_cv_folds`，全仓唯一一份折边界算术）、wandb run 的创建。公开的
      `train` / `train_cv` / `load` / `predict` 只在这里实现一份，任何具体的头
      都不覆盖它们（`tests/test_model_hierarchy.py` 锁）。
    - `DLModel`：torch 变体。device、张量转换、DataLoader 按 epoch 的训练循环、
      state_dict checkpoint、refit 优化器都是它的细节；子类实现五个张量钩子。
    - `MLModel`：numpy 变体（xgboost 这类树模型）。没有 epoch 循环，训练、早停
      与最优模型回滚交给所属库的原生机制；子类实现四个钩子。

    为什么按框架拆：把非 torch 的分支硬塞进一个 torch 形状的基类，每个方法都会
    长出「模型是不是 nn.Module」的分支。按框架拆开后，每一层只暴露自己需要的
    钩子，而训练编排之外的东西——尤其是 CV 折边界——仍然只有一份。

    两个抽象类属性由具体变体用**普通类属性**满足：

    - `config_cls`：这个变体接受的配置类。`config` setter 第一件事就是检查它，
      `utils/module.py:load_model_from_config` 也在实例化之前从类上读它，
      所以它必须是类属性，不能只存在于实例上。
    - `checkpoint_suffix`：checkpoint 文件后缀。`train` / `train_cv` 用它拼文件名，
      `load()` 用它在构建任何模型之前拒收错误类型的文件。
    """

    def __init__(self, config: DLConfig | MLConfig):
        self.config = config
        self._set_random_seed(self.config.random_seed)

        self.model = None

        self.data_backend = XrBackend()
        # self._pre_feature: Optional[xr.Dataset] = None
        self._wandb_recorder: wandb.sdk.wandb_run.Run = None  # type: ignore

    @property
    @abstractmethod
    def config_cls(self) -> type:
        """这个变体接受的配置类（具体变体用类属性覆盖）。"""

    @property
    @abstractmethod
    def checkpoint_suffix(self) -> str:
        """checkpoint 文件后缀，含点号（具体变体用类属性覆盖）。"""

    @staticmethod
    def _set_random_seed(seed: int):
        """只播种框架无关的两处：python `random` 与 numpy。

        torch 的种子由 `DLModel._set_random_seed` 在此之上补齐。
        """
        random.seed(seed)
        np.random.seed(seed)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> DLConfig | MLConfig:
        return self._config

    @config.setter
    def config(self, config: DLConfig | MLConfig):
        # 类型检查必须是第一条语句：先于给 `_config` 赋值，也先于触碰任何因子或
        # 标签。拿错配置类的模型如果先把日期写进因子再报错，调用方手里的因子
        # 对象就已经被改过了。
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} requires a {self.config_cls.__name__}, "
                f"got {type(config).__name__}"
            )
        self._config = config
        self._config.name = self.import_path

        # 初始化时间
        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

        self._reset_factors_config()
        self._reset_labels_config()

        if self._config.backtest_data is not None:
            self._reset_backtest_dataset_config()

    @property
    def num_times(self) -> int:
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).timestamp.size

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    @property
    def num_symbols(self) -> int:
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.size

    @property
    def symbols(self) -> list[str]:
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.values.tolist()

    @property
    def num_null(self) -> int:
        return int(
            self.data_backend.get_xarray_dataset(["timestamp", "symbol"])
            .isnull()
            .sum()
            .to_dataarray()
            .sum()
            .item()
        )

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def num_factors(self) -> int:
        return len(self.get_factor_names())

    @property
    def num_labels(self) -> int:
        return len(self.get_label_names())

    def _reset_factors_config(self):
        for factor in self._config.factors:
            # 覆盖因子配置文件日期
            factor.config.start_date = self._config.start_date
            factor.config.end_date = self._config.end_date

            # 因子类重置数据集配置
            factor._reset_dataset_config()

    def _reset_labels_config(self):
        for label in self._config.labels:
            label.config.start_date = self._config.start_date
            label.config.end_date = self._config.end_date

            # 因子类重置数据集配置
            label._reset_dataset_config()

    def _reset_backtest_dataset_config(self):
        self.config.backtest_data.start_date = self._config.start_date
        self.config.backtest_data.end_date = self._config.end_date

    def _collect_all_labels(self) -> xr.Dataset:
        """把 `config.labels` 里的**每一个**标签取出来合成一块面板。

        它以前叫 `_get_labels_batch`。这个模块里 `batch` 已经有一个确定的意思
        ——`DataLoader` 切出来的 mini-batch（见 `_train_one_batch`）——而这里
        既不切也不采样，是「全部收齐再 `combine_by_coords`」。同一个词在同一个
        文件里指两件相反的事，名字就得让一个。
        """
        all_ds = []
        for label in self.config.labels:
            match self.config.label_data_strategy:
                case "cal":
                    ds = label.cal().get_labels()
                case "read":
                    ds = label.read().get_labels()
                case _:
                    raise ValueError(
                        f"label_data_strategy {self.config.label_data_strategy} is not supported"
                    )
            all_ds.append(ds)
        data: xr.Dataset = xr.combine_by_coords(all_ds)  # type: ignore
        data = data.sortby(["timestamp", "symbol"])
        return data

    def _collect_all_features(self) -> xr.Dataset:
        """把 `config.factors` 里的**每一个**因子取出来合成一块面板。

        命名理由见 `_collect_all_labels`：这里没有任何 mini-batch 语义。
        """
        all_ds = []
        for factor in self.config.factors:
            match self.config.factor_data_strategy:
                case "cal":
                    ds = factor.cal().get_features()
                case "read":
                    ds = factor.read().get_features()
                case _:
                    raise ValueError(
                        f"data_strategy {self.config.factor_data_strategy} not supported"
                    )
            all_ds.append(ds)
        data: xr.Dataset = xr.combine_by_coords(all_ds)  # type: ignore
        return data

    def collect(
        self,
    ) -> Self:
        feature = self._collect_all_features()
        label = self._collect_all_labels()
        d = xr.combine_by_coords([feature, label])
        d = d.sortby(["timestamp", "symbol"])
        self.data_backend.to_internal(d)  # type: ignore
        return self

    def get_factor_names(self):
        return list(
            chain.from_iterable(
                [factor._get_factor_names() for factor in self.config.factors]
            )
        )

    def get_label_names(self):
        return list(
            chain.from_iterable(
                [label._get_factor_names() for label in self.config.labels]
            )
        )

    def get_config(self) -> dict:
        cfg = self.config.to_dict()
        cfg["factors"] = [factor.get_config() for factor in self.config.factors]  # type: ignore
        cfg["labels"] = [label.get_config() for label in self.config.labels]  # type: ignore
        return cfg  # type: ignore

    def _get_config_with_extra_kv(self, extra_kv: dict) -> dict:
        cfg = self.get_config()
        cfg.update(extra_kv)
        return cfg

    def to_array(self, data: xr.Dataset, variables: list[str]) -> np.ndarray:
        """把 `(timestamp, symbol)` 面板转成 `[num_times, num_symbols, len(variables)]` 的数组。

        最后一维严格按 `variables` 给定的顺序排列，这是这个方法存在的全部理由。
        以前这条链上是 `.sortby(["timestamp", "symbol", "variable"])`，最后一维
        于是按变量**名**字母序排，`y[..., 0]` 拿到的是 `ret_120` 而不是声明在
        第一位的 `ret_30`。`.sel(variable=variables)` 才是按声明顺序取。
        """
        return (
            data[variables]
            .to_dataarray()
            .sortby(["timestamp", "symbol"])
            .sel(variable=variables)
            .transpose("timestamp", "symbol", "variable")
            .values
        )

    def _save_model(self, p: Path):
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not initialized")

        if p.parent.exists():
            raise RuntimeError(f"{p.parent} already exists")
        else:
            p.parent.mkdir(parents=True)

        # 先保存config为json
        with open(p.parent / Path("config.json"), "w") as f:
            json.dump(self.get_config(), f, indent=4)

        self._write_checkpoint(p)

    def load(self, p: Path | str) -> Self:
        """从 checkpoint 恢复模型，返回 self。

        后缀校验先于任何模型构建：把 `.pth` 交给 joblib、或把 `.joblib` 交给
        `torch.load`，失败方式要么是难懂的反序列化报错，要么是悄悄反序列化出
        一个错误类型的对象。
        """
        if isinstance(p, str):
            p = Path(p)

        if not p.exists():
            raise FileNotFoundError(f"{p} not found")

        if p.suffix != self.checkpoint_suffix:
            raise ValueError(
                f"Unsupported file type: {p.suffix!r}; {self.class_name} "
                f"checkpoints use {self.checkpoint_suffix!r} ({p})"
            )

        self._read_checkpoint(p)
        return self

    def predict(
        self, data: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """对 `[T, S, F]` 输入给出 `[T, S, L]` 预测；具体转换由变体的 `_predict` 决定。"""
        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "Model not initialized, please call load() or train() first"
            )
        return self._predict(data)

    def predict_panel(self, features: xr.Dataset) -> xr.Dataset:
        """对 `(timestamp, symbol)` 特征面板给出同维度的预测面板（03.7 D-29）。

        返回的 `xr.Dataset` 每个标签名一个变量，维度 `("timestamp", "symbol")`，
        坐标取自 `to_array` 实际消费的那块排序后的面板，所以坐标与数值不会错位。
        xarray -> 数组 -> 预测 -> xarray 这条管道只在模型层实现一份，DL 与 ML
        共用；变体之间的差别只在 `_predict_panel_array` 这一个钩子里。

        所有特征都是 NaN 的 `(t, s)` 位置，所有标签的预测都置为 NaN：xgboost 与
        先 `nan_to_num` 输入的 DL 头会给还没上市的标的算出有限预测，不屏蔽的话
        回测会把它们选进去（03.7-RESEARCH.md Pitfall 7）。

        走公开的 `predict`，「Model not initialized」的守卫在那里。
        """
        factors = self.get_factor_names()
        labels = self.get_label_names()
        missing = [name for name in factors if name not in features.data_vars]
        if missing:
            raise ValueError(
                f"{self.class_name}.predict_panel: features are missing factor "
                f"variable(s) {missing}"
            )

        feats = features[factors].sortby(["timestamp", "symbol"])
        x = self.to_array(feats, factors)
        y = np.asarray(self._predict_panel_array(x), dtype=np.float64)
        expected = (x.shape[0], x.shape[1], len(labels))
        if y.shape != expected:
            raise ValueError(
                f"{self.class_name}.predict_panel: expected prediction shape "
                f"{expected} [num_times, num_symbols, num_labels], got {y.shape}"
            )

        y = y.copy()
        y[np.isnan(x).all(axis=-1)] = np.nan
        return xr.Dataset(
            {
                name: (("timestamp", "symbol"), y[..., i])
                for i, name in enumerate(labels)
            },
            coords={
                "timestamp": feats.timestamp.values,
                "symbol": feats.symbol.values,
            },
        )

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """`predict_panel` 的变体钩子：`[T, S, F]` numpy 进，`[T, S, L]` numpy 出。

        刻意是普通方法而不是抽象方法：抽象的话每一层的 `__abstractmethods__`
        都会变，`tests/test_model_hierarchy.py` 锁的正是这些集合。
        """
        raise NotImplementedError(
            f"{self.class_name} does not implement _predict_panel_array"
        )

    def train(self):
        project_name = f"{self.class_name}_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_name = f"{self.class_name}_total"
        model_name = f"{experiment_name}{self.checkpoint_suffix}"
        self._init_wandb(
            project_name=project_name,
            experiment_name=experiment_name,
        )
        self._fit(
            project_name=project_name,
            experiment_name=experiment_name,
            model_name=model_name,
        )

    @staticmethod
    def _cv_folds(
        timestamps, train_periods: int, gap_periods: int
    ) -> list[dict]:
        """滚动前推交叉验证的折几何。这是折边界算术在全仓**唯一**的实现。

        `train_cv` 的顺序分支与并行分支、DL 与 ML 都消费这一个生成器，所以它们
        不可能各自漂移（重构前两个分支各抄了一份同样的算术）。几何逐字沿用
        重构前的写法，由 `tests/test_model_cv.py` 里重构前捕获的 golden 锁定：

        - `test_periods = train_periods // 5`；
        - 第 i 折训练段是下标 `[i*test_periods, i*test_periods + train_periods)`，
          之后空出 `gap_periods` 个时间点，再接 `test_periods` 个时间点的测试段；
        - 折数 `max(1, (总长 - train_periods - gap_periods) // test_periods)`；
          测试段越过数据末尾的折记 warning 并跳过，所以数据不足时返回 `[]`。

        每折产出 `{"fold", "train_start", "train_end", "test_start", "test_end"}`，
        日期由 `np.datetime_as_string` 生成，两端都是闭区间。
        """
        total_periods = len(timestamps)
        test_periods = train_periods // 5  # Test set is 20% of training set
        n_splits = max(
            1, (total_periods - train_periods - gap_periods) // test_periods
        )

        folds: list[dict] = []
        for i in range(n_splits):
            # Calculate indices for each fold
            train_start_idx = i * test_periods
            train_end_idx = train_start_idx + train_periods
            test_start_idx = train_end_idx + gap_periods
            test_end_idx = test_start_idx + test_periods

            # Check if test set exceeds data range
            if test_end_idx > total_periods:
                logger.warning(
                    f"Skipping fold {i}: test set exceeds data range"
                )
                continue

            # Convert indices to timestamps
            folds.append(
                {
                    "fold": i,
                    "train_start": np.datetime_as_string(
                        timestamps[train_start_idx]
                    ),
                    "train_end": np.datetime_as_string(
                        timestamps[train_end_idx - 1]
                    ),
                    "test_start": np.datetime_as_string(
                        timestamps[test_start_idx]
                    ),
                    "test_end": np.datetime_as_string(
                        timestamps[test_end_idx - 1]
                    ),
                }
            )
        return folds

    def _train_one_fold(self, fold: dict, project_name: str) -> dict:
        """在**本实例**上训练一折，返回该折的结果 dict。

        结果 = 折 dict 的五个键 + `experiment_name` + `checkpoint`（该折落盘的
        文件路径）+ `_fit` 返回的 test 指标（不产出指标的变体没有这部分）。
        """
        self.config.train_start = fold["train_start"]
        self.config.train_end = fold["train_end"]
        self.config.test_start = fold["test_start"]
        self.config.test_end = fold["test_end"]

        experiment_name = f"{self.class_name}_cv_fold_{fold['fold']}"
        model_name = f"{experiment_name}{self.checkpoint_suffix}"

        self._init_wandb(
            project_name=project_name,
            experiment_name=experiment_name,
        )
        metrics = self._fit(
            project_name=project_name,
            experiment_name=experiment_name,
            model_name=model_name,
        )
        return {
            **fold,
            "experiment_name": experiment_name,
            "checkpoint": str(
                Path(self.config.model_save_dir)
                / project_name
                / experiment_name
                / model_name
            ),
            **(metrics or {}),
        }

    def _train_fold_with_config(self, fold: dict, project_name: str) -> dict:
        """并行分支用：在本实例的深拷贝上训练一折。

        每折一份 `copy.deepcopy(self)`，折之间不共享配置日期、模型与 wandb run；
        代价是面板数据被复制 njobs 份。
        """
        return copy.deepcopy(self)._train_one_fold(fold, project_name)

    #: `train_cv` 在项目目录里写的折清单文件名（03.7 D-30）。
    CV_FOLDS_FILENAME = "cv_folds.json"
    #: 折清单的格式版本（03.7 D-36）。`run_cv` 会读旧训练 run 的清单，并拒收
    #: 它不认识的版本；改动清单结构时必须递增这里。
    CV_FOLDS_FORMAT_VERSION = 1

    #: 折 dict 自带的键。`test_start` / `test_end` 也以 `test_` 开头，但它们是
    #: 日期不是指标，求 CV 均值时必须排除。
    _CV_FOLD_KEYS = frozenset(
        {"fold", "train_start", "train_end", "test_start", "test_end"}
    )

    @staticmethod
    def _cv_mean_metrics(results: list[dict]) -> dict:
        """对各折的 `test_*` 指标求均值，键名 `cv_mean_{key}`，另加 `cv_n_folds`。

        只取有限的数值；某个指标在所有折上都不是有限值时均值为 NaN（显式计数，
        不走会发 RuntimeWarning 的空集求均值）。没有任何 `test_*` 指标时——比如
        DL 变体的 `_fit` 不返回指标——返回空 dict，`train_cv` 也就不开 summary run。
        """
        keys: list[str] = []
        for result in results:
            for key, value in result.items():
                if (
                    key.startswith("test_")
                    and key not in BaseModel._CV_FOLD_KEYS
                    and isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, bool)
                    and key not in keys
                ):
                    keys.append(key)
        if not keys:
            return {}

        means: dict = {}
        for key in keys:
            finite = [
                float(r[key])
                for r in results
                if key in r and np.isfinite(float(r[key]))
            ]
            means[f"cv_mean_{key}"] = (
                sum(finite) / len(finite) if finite else float("nan")
            )
        means["cv_n_folds"] = len(results)
        return means

    def train_cv(
        self,
        train_periods: int,
        gap_periods: int = 0,
        parallel: bool = False,
        njobs: int = -1,
    ) -> list[dict]:
        """滚动前推交叉验证，返回逐折结果 list。

        折几何见 `_cv_folds`。每折有自己的 wandb run 与 checkpoint 目录
        `{class}_cv_fold_{i}/`。各折的 `test_*` 指标求均值后写进一个独立的
        `{class}_cv_summary` run 的 summary（`cv_mean_test_*` 与 `cv_n_folds`）：
        每折的 `_fit` 结束时已经 finish 了自己的 run，均值算出来时没有还开着的
        run 可写。`parallel=True` 在 joblib threading 后端上为每折深拷贝本实例。

        返回前把折清单写到 `{model_save_dir}/{project_name}/cv_folds.json`
        （03.7 D-30 / D-36），内容是 `{"format_version": 1, "folds": [...]}`：
        `folds` 就是本方法返回的这个 list 的 JSON 形式（`to_jsonable` 转换，NaN /
        inf 写成 null，所以文件是严格 JSON），不多不少。折数为 0 时照样写，
        `folds` 为 `[]`。经 `write_json_atomically` 落盘，中断的 run 不会留下
        写了一半的文件。

        之所以带 `format_version`：`run_cv` 要读**旧**训练 run 留下的清单来回测
        每折的样本外段，清单因此是一个持久化格式，结构改动必须递增
        `CV_FOLDS_FORMAT_VERSION`，`run_cv` 拒收不认识的版本。清单不改变返回值：
        返回的折 dict 里没有任何清单的键。
        """
        start_date = self.config.start_date
        end_date = self.config.end_date

        project_name = f"{self.class_name}_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"])

        # Filter data by date range
        data_in_range = data.sel(timestamp=slice(start_date, end_date))
        timestamps = data_in_range.timestamp.values

        if len(timestamps) == 0:
            raise ValueError(
                f"No data found between {start_date} and {end_date}"
            )

        logger.info(
            f"Starting CV from {start_date} to {end_date} with {train_periods} training periods and {gap_periods} periods gap"
        )

        folds = self._cv_folds(timestamps, train_periods, gap_periods)

        logger.info(f"Total {len(folds)} folds will be created")
        for fold in folds:
            logger.info(
                f"Fold {fold['fold']}: Train [{fold['train_start']} to {fold['train_end']}], Test [{fold['test_start']} to {fold['test_end']}]"
            )

        if parallel:
            logger.info(f"Starting parallel training of {len(folds)} folds")
            results = list(
                Parallel(n_jobs=njobs, backend="threading")(
                    delayed(self._train_fold_with_config)(fold, project_name)
                    for fold in folds
                )
            )
        else:
            results = [
                self._train_one_fold(fold, project_name) for fold in folds
            ]

        means = self._cv_mean_metrics(results)
        if means:
            self._init_wandb(
                project_name=project_name,
                experiment_name=f"{self.class_name}_cv_summary",
            )
            if self._wandb_recorder is not None:
                self._wandb_recorder.summary.update(means)
                self._wandb_recorder.finish()

        # 放在两个分支之后：顺序与并行都经过这里。写入的是 `to_jsonable` 转出的
        # 新对象，`results` 本身原样返回（D-30 要求返回值不变）。
        write_json_atomically(
            Path(self.config.model_save_dir)
            / project_name
            / self.CV_FOLDS_FILENAME,
            {
                "format_version": self.CV_FOLDS_FORMAT_VERSION,
                "folds": to_jsonable(results),
            },
            indent=2,
        )

        return results

    def _assert_shape_match_y(self, data: np.ndarray | torch.Tensor):
        num_symbols, num_labels = (
            self.num_symbols,
            self.num_labels,
        )
        if num_symbols != data.shape[1] or num_labels != data.shape[2]:
            raise ValueError(
                f"Train y shape mismatch: [num_times, {num_symbols}, {num_labels}] vs {data.shape}"
            )

    def _assert_shape_match_x(self, data: np.ndarray | torch.Tensor):
        num_symbols, num_features = (
            self.num_symbols,
            self.num_factors,
        )
        if num_symbols != data.shape[1] or num_features != data.shape[2]:
            raise ValueError(
                f"Train x shape mismatch: [num_times, {num_symbols}, {num_features}] vs {data.shape}"
            )

    def _init_wandb(self, project_name: str, experiment_name: str):
        self._wandb_recorder = wandb.init(
            project=project_name, name=experiment_name, config=self.get_config()
        )

    @abstractmethod
    def _fit(
        self, project_name: str, experiment_name: str, model_name: str
    ) -> dict | None:
        """按 `config` 里的四个日期训练一次、评估、落盘，并 finish 当前 wandb run。

        checkpoint 写到 `model_save_dir / project_name / experiment_name / model_name`。
        返回 test 指标 dict（键带 `test_` 前缀，`train_cv` 据此求 CV 均值）；
        不产出指标的实现返回 None。
        """

    @abstractmethod
    def _predict(self, data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """`predict` 的变体实现；调用时 `self.model` 保证已存在。"""

    @abstractmethod
    def _write_checkpoint(self, path: Path) -> None:
        """把 `self.model` 写到 `path`（目录与 `config.json` 已由 `_save_model` 备好）。"""

    @abstractmethod
    def _read_checkpoint(self, path: Path) -> None:
        """从 `path` 恢复 `self.model`；后缀已由 `load()` 校验过。"""


class DLModel(BaseModel):
    """torch 变体：DataLoader 按 epoch 的训练循环。

    device、`to_tensor`、epoch 循环、按 epoch 的早停与最优 state_dict 回滚、
    `.pth` checkpoint、refit 优化器都在这一层。子类实现五个张量钩子：
    `_init_model`、`_train_one_batch`、`_val_one_batch`、`_test_one_batch`、
    `_preprocess`。

    DL 的早停保持按 epoch 判定并回滚到最优 epoch：神经网络一次参数更新作用于
    整个模型，没有「前 k 棵树」那种可以廉价切片回退的结构，所以只能在 epoch
    边界上快照权重。
    """

    config_cls = DLConfig
    checkpoint_suffix = ".pth"

    @staticmethod
    def _set_random_seed(seed: int):
        BaseModel._set_random_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _to_default_float(tensor: torch.Tensor) -> torch.Tensor:
        """浮点张量统一到 torch 的默认浮点 dtype；非浮点张量原样返回。

        `torch.from_numpy` 保留 numpy 的 dtype，而 pandas/polars 这两条数据路径
        给出的是 float64，模型参数却是 float32，不统一就在第一个 Linear 层报
        dtype 不匹配。
        """
        default_dtype = torch.get_default_dtype()
        if tensor.is_floating_point() and tensor.dtype != default_dtype:
            tensor = tensor.to(default_dtype)
        return tensor

    def to_tensor(self, data: xr.Dataset, variables: list[str]) -> torch.Tensor:
        """把 `(timestamp, symbol)` 面板转成 `[num_times, num_symbols, len(variables)]`。

        `BaseModel.to_array` 的张量包装：变量轴顺序的保证来自 `to_array`，
        这里只负责转张量并统一浮点 dtype。
        """
        return self._to_default_float(
            torch.from_numpy(self.to_array(data, variables))
        )

    def _predict(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = self._to_default_float(torch.from_numpy(data))
        elif not isinstance(data, torch.Tensor):
            raise TypeError(f"Unsupported data type: {type(data)}")
        # 推理必须切 eval + no_grad。`load()` 新建的 nn.Module 默认处在 training
        # 模式，以前这里两样都没做：dropout 是开着的（train_model.py 的配置里
        # 首层就是 0.5），同一份输入每次调用给出的结果都不一样；而且整张计算图
        # 被留下来白吃内存。切过之后模型就留在 eval 模式——要继续训练的话，
        # 训练循环开头本来就会调 `self.model.train()`。
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            data = data.to(self.device)
            data = self._preprocess(data)
            return self.model(data)  # type: ignore[misc]

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """张量输出转 numpy；`forward` 返回 tuple 的头必须自己覆盖本钩子（D-33）。"""
        raw = self.predict(x)
        if isinstance(raw, torch.Tensor):
            return raw.detach().cpu().numpy()
        if isinstance(raw, (tuple, list)):
            raise TypeError(
                f"{self.class_name}: forward returns a {type(raw).__name__}, not "
                f"a [T, S, L] tensor; the head must override "
                f"_predict_panel_array to map it onto one channel per label"
            )
        raise TypeError(
            f"{self.class_name}: unsupported prediction type "
            f"{type(raw).__name__}; expected a torch.Tensor"
        )

    def _init_model_and_optim(self):
        self.model = self._init_model(
            num_symbols=self.num_symbols,
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.config.hyperparameters,
        )
        self.model = self.model.to(self.device)  # type: ignore
        optim = self._init_optim(self.model)  # type: ignore
        if optim is not None:
            self.optim = optim

    def _fit(
        self,
        project_name: str,
        experiment_name: str,
        model_name: str,
        backtest: bool = False,
    ):
        if backtest:
            raise NotImplementedError(
                "_fit(backtest=True) is not supported: end-to-end "
                "backtesting is owned by Phase 6 and _do_vecbt is still a "
                "skeleton. Train with backtest=False and run the backtest "
                "separately."
            )

        train_start, train_end, test_start, test_end = (
            self.config.train_start,
            self.config.train_end,
            self.config.test_start,
            self.config.test_end,
        )
        if not train_start or not train_end or not test_start or not test_end:
            raise ValueError(
                "Training and testing start and end dates must be specified."
            )

        self._init_model_and_optim()

        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"])
        train_data = data.sel(timestamp=slice(train_start, train_end))
        test_data = data.sel(timestamp=slice(test_start, test_end))
        factors = self.get_factor_names()
        labels = self.get_label_names()
        train_x = train_data[factors]
        train_y = train_data[labels]
        test_x = test_data[factors]
        test_y = test_data[labels]

        datas = [
            self.to_tensor(d, names)
            for d, names in [
                (train_x, factors),
                (train_y, labels),
                (test_x, factors),
                (test_y, labels),
            ]
        ]
        datas = [self._preprocess(d) for d in datas]

        train_x_t_all, train_y_t_all, test_x_t, test_y_t = datas
        for d in [train_x_t_all, test_x_t]:
            self._assert_shape_match_x(d)
        for d in [train_y_t_all, test_y_t]:
            self._assert_shape_match_y(d)

        train_split = int(train_x_t_all.shape[0] * (1 - self.config.val_size))
        train_x_t = train_x_t_all[:train_split]
        train_y_t = train_y_t_all[:train_split]
        # 切点用 `train_split:` 而不是 `train_split + 1:`：后者会让第 train_split
        # 行既不在训练集也不在验证集，被静默丢掉。
        val_x_t = train_x_t_all[train_split:]
        val_y_t = train_y_t_all[train_split:]

        train_loader = DataLoader(
            TensorDataset(train_x_t, train_y_t),
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.num_workers,
        )

        val_test_loaders = [
            DataLoader(
                TensorDataset(*d),
                batch_size=self.config.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.config.num_workers,
            )
            for d in [
                (val_x_t, val_y_t),
                (test_x_t, test_y_t),
            ]
        ]
        val_loader, test_loader = val_test_loaders

        best_loss = float("inf")
        early_stopping = False
        patience = self.config.early_stopping_patience
        counter = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in tqdm(
            range(self.config.epochs),
            desc=f"{self.class_name}_train",
        ):
            self.model.train()  # type: ignore
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                self._train_one_batch(epoch, x_batch, y_batch)

            self.model.eval()  # type: ignore
            with torch.no_grad():
                val_loss_sum = 0.0
                val_sample_count = 0
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    val_loss = self._val_one_batch(epoch, x_batch, y_batch)

                    batch_samples = int(x_batch.shape[0])
                    val_loss_sum += float(val_loss) * batch_samples
                    val_sample_count += batch_samples

                if self.config.early_stopping and val_sample_count > 0:
                    epoch_val_loss = val_loss_sum / val_sample_count
                    if epoch_val_loss < best_loss:
                        best_loss = epoch_val_loss
                        counter = 0
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()  # type: ignore[union-attr]
                        }
                    else:
                        counter += 1
                        if counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            early_stopping = True

                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    self._test_one_batch(epoch, x_batch, y_batch)

                if early_stopping:
                    break

        if self.config.early_stopping and best_state is not None:
            self.model.load_state_dict(best_state)  # type: ignore[union-attr]

        self._save_model(
            Path(self.config.model_save_dir)
            / project_name
            / experiment_name
            / model_name,
        )

        if self._wandb_recorder:
            self._wandb_recorder.finish()

        self.optim = None

    def _write_checkpoint(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)  # type: ignore[union-attr]

    def _read_checkpoint(self, path: Path) -> None:
        self.model = self._init_model(
            num_symbols=self.num_symbols,
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.config.hyperparameters,
        ).to(self.device)
        self.model.load_state_dict(torch.load(path))  # type: ignore[union-attr]

    @abstractmethod
    def _init_model(
        self,
        num_symbols: int,
        num_features: int,
        num_labels: int,
        hyperparameters: dict,
    ): ...

    @abstractmethod
    def _test_one_batch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor: ...

    @abstractmethod
    def _train_one_batch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor: ...

    @abstractmethod
    def _val_one_batch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor: ...

    @abstractmethod
    def _preprocess(self, data: torch.Tensor) -> torch.Tensor: ...

    def _init_optim(self, model: torch.nn.Module):
        raise NotImplementedError

    def _get_refit_optim(self) -> torch.optim.Optimizer:
        key = (self.model, self.config.lr_refit)
        cached = getattr(self, "_refit_optim_cache", None)
        if (
            cached is not None
            and cached[0][0] is key[0]
            and cached[0][1] == key[1]
        ):
            return cached[1]

        optim = torch.optim.AdamW(
            self.model.parameters(),  # type: ignore[union-attr]
            lr=self.config.lr_refit,
        )
        self._refit_optim_cache = (key, optim)
        return optim


class MLModel(BaseModel):
    """numpy 变体：非 torch 的 ML / 树模型（xgboost 等）。

    **没有 epoch 循环，也不做任何模型拷贝回滚。** 训练、早停与最优模型的选取
    全部交给所属库的原生机制（`_fit_model` 负责），理由有三：

    - 粒度：xgboost / LightGBM / CatBoost 都是逐棵树（逐轮）判定早停，外层按
      「epoch」包一圈只会把判定粒度变粗；
    - 成本：库内的验证分数靠预测缓存增量计算，总成本随轮数线性增长；外层每个
      epoch 用全部树把验证集重算一遍，总成本是二次的；
    - 回滚：树模型回到最优轮只需切片保留前 k 棵树（xgboost 的
      `EarlyStopping(save_best=True)`），不需要拷贝整个模型。

    DL 仍按 epoch 走（见 `DLModel`），因为神经网络没有这种可切片的结构。

    子类实现四个钩子：`_init_model`、`_preprocess`、`_fit_model`、`_forward`。
    `_loss` / `_compute_metrics` / `_evaluate` 有可用的默认实现，可按需覆盖。
    checkpoint 是 `.joblib`，经 `ml_model/backend.py:MlBackend` 读写——本质是
    pickle，只加载自己信任的文件。
    """

    config_cls = MLConfig
    checkpoint_suffix = ".joblib"

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """根据特征数、标签数与超参准备模型；返回值原样赋给 `self.model`。

        树模型往往要到 `_fit_model` 里才真正建出模型，此时可以只解析超参并返回 None。
        `load()` 不调用本方法：checkpoint 里就是完整模型。
        """

    @abstractmethod
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """对 `[T, S, *]` 数组做预处理，返回新数组；不得原地修改入参。

        训练时四份数组（train/test 的 x 与 y）各调用一次，推理时对输入调用一次。
        """

    @abstractmethod
    def _fit_model(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> None:
        """训练模型。入参是 `[T, S, F]` / `[T, S, L]`。

        验证段为空（`val_size == 0`）时 `val_x` 与 `val_y` 都是 None。早停与最优
        模型回滚由本方法用库的原生机制完成，遵循 `config.early_stopping` 与
        `config.early_stopping_patience`；返回时 `self.model` 必须已经是要保存的
        模型。
        """

    @abstractmethod
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """对已预处理的 `[T, S, F]` 输入返回 `[T, S, L]` 预测。"""

    def _resolved_hyperparameters(self) -> dict | None:
        """实际生效的超参记录（默认值合并用户覆盖之后），默认 None 表示不记录。

        头在 `_init_model` 里解析出库参数后覆盖本钩子返回它们。非 None 时，
        `_fit` 把它写进 wandb run config，`get_config` 把它放进 `config.json`
        的顶层 `resolved_hyperparameters`——这样日后库默认值或头的默认参数改了，
        这次训练仍能按记录复现。它是记录不是输入：`config.hyperparameters`
        保持用户原样，`load_model_from_config` 重建配置时丢弃这个键。
        """
        return None

    def get_config(self) -> dict:
        """在 `BaseModel.get_config` 之上，追加非 None 的 `resolved_hyperparameters`。"""
        cfg = super().get_config()
        resolved = self._resolved_hyperparameters()
        if resolved is not None:
            cfg["resolved_hyperparameters"] = dict(resolved)
        return cfg

    def _loss(self, y: np.ndarray, pred: np.ndarray) -> float:
        """默认损失：所有标签都有限的 `(t, s)` 位置上、对全部标签求 MSE。

        没有有效位置时返回 NaN（显式计数，不发 RuntimeWarning）。
        """
        y = np.asarray(y, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        rows = np.isfinite(y).all(axis=-1)
        n = int(rows.sum())
        if n == 0:
            return float("nan")
        diff = pred[rows] - y[rows]
        return float(np.sum(diff * diff) / diff.size)

    def _compute_metrics(self, y: np.ndarray, pred: np.ndarray) -> dict:
        """默认指标：主标签（最后一维第 0 个）上的 `regression_panel_metrics`。"""
        return regression_panel_metrics(pred[..., 0], y[..., 0])

    def _evaluate(
        self, split: str, x: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        """在一个切分上评估，返回带前缀的指标 dict 并写入 wandb summary。

        键为 `{split}_loss` 与 `{split}_{mse,rmse,mae,r2,ic,rank_ic}`（下划线）。
        写的是 summary 里的最终值、不带 step，不和逐轮 `log(step=...)` 的曲线
        争抢 step。
        """
        pred = self._forward(x)
        metrics = {f"{split}_loss": self._loss(y, pred)}
        for key, value in self._compute_metrics(y, pred).items():
            metrics[f"{split}_{key}"] = value
        if self._wandb_recorder is not None:
            self._wandb_recorder.summary.update(metrics)
        return metrics

    def _fit(
        self, project_name: str, experiment_name: str, model_name: str
    ) -> dict:
        """一次 ML 训练：切分 -> `_fit_model` 一次 -> 评估 -> 落盘 -> finish。

        验证段是训练段尾部的 `val_size` 比例（与 DL 的切法一致）。空切分一律
        跳过评估：没有验证段就没有 `val_*` 指标，测试段为空时返回 `{}`。
        返回 test 指标 dict。
        """
        train_start, train_end, test_start, test_end = (
            self.config.train_start,
            self.config.train_end,
            self.config.test_start,
            self.config.test_end,
        )
        if not train_start or not train_end or not test_start or not test_end:
            raise ValueError(
                "Training and testing start and end dates must be specified."
            )

        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.config.hyperparameters,
        )
        resolved = self._resolved_hyperparameters()
        if resolved is not None and self._wandb_recorder is not None:
            # run 在 `_init_wandb` 时已经打开，那时参数还没解析；这里补记。
            self._wandb_recorder.config.update(
                {"resolved_hyperparameters": dict(resolved)},
                allow_val_change=True,
            )

        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"])
        train_data = data.sel(timestamp=slice(train_start, train_end))
        test_data = data.sel(timestamp=slice(test_start, test_end))
        factors = self.get_factor_names()
        labels = self.get_label_names()

        train_x_all, train_y_all, test_x, test_y = [
            self._preprocess(self.to_array(d, names))
            for d, names in [
                (train_data, factors),
                (train_data, labels),
                (test_data, factors),
                (test_data, labels),
            ]
        ]
        for d in [train_x_all, test_x]:
            self._assert_shape_match_x(d)
        for d in [train_y_all, test_y]:
            self._assert_shape_match_y(d)

        n_train_times = train_x_all.shape[0]
        train_split = int(n_train_times * (1 - self.config.val_size))
        if train_split == 0:
            raise ValueError(
                f"Empty training segment: val_size={self.config.val_size} "
                f"leaves 0 of {n_train_times} training timestamps for fitting."
            )
        train_x = train_x_all[:train_split]
        train_y = train_y_all[:train_split]
        if train_split < n_train_times:
            val_x = train_x_all[train_split:]
            val_y = train_y_all[train_split:]
        else:
            val_x = val_y = None

        self._fit_model(train_x, train_y, val_x, val_y)

        self._evaluate("train", train_x, train_y)
        if val_x is not None:
            self._evaluate("val", val_x, val_y)
        test_metrics = (
            self._evaluate("test", test_x, test_y) if test_x.shape[0] > 0 else {}
        )

        self._save_model(
            Path(self.config.model_save_dir)
            / project_name
            / experiment_name
            / model_name,
        )

        if self._wandb_recorder is not None:
            self._wandb_recorder.finish()

        return test_metrics

    def _predict(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")
        return self._forward(self._preprocess(data))

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """ML 头的 `_forward` 本来就返回 `[T, S, L]` numpy，这里只做 `np.asarray`。"""
        return np.asarray(self.predict(x))

    def _write_checkpoint(self, path: Path) -> None:
        MlBackend().to_internal(self.model).write(str(path))

    def _read_checkpoint(self, path: Path) -> None:
        # 不调 `_init_model`：joblib 文件里就是完整模型，重建一个空模型再覆盖既
        # 多余，又要求新实例先 collect 才知道特征数。
        self.model = MlBackend().read(str(path)).get_model()
