import json
import random
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Self

import joblib
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

from dataset.backend import XrBackend
from enums.constant import Date

from .config import DLConfig, MLConfig


class BaseModel(ABC):
    def __init__(self, config: DLConfig | MLConfig):
        self.config = config
        self._set_random_seed(self.config.random_seed)

        self.model = None

        self.data_backend = XrBackend()
        # self._pre_feature: Optional[xr.Dataset] = None
        self._wandb_recorder: wandb.sdk.wandb_run.Run = None  # type: ignore

    @staticmethod
    def _set_random_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> DLConfig | MLConfig:
        return self._config

    @config.setter
    def config(self, config: DLConfig | MLConfig):
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
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

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
        """整块面板上 NaN 单元格的总数（跨全部变量、全部 timestamp、全部 symbol）。

        结尾曾经是 `.values[0]`，但它前面那个 `.sum()` 已经把 `variable` 维也加
        掉了，得到的是一个 **0 维** DataArray——于是**每一次读取**都是
        `IndexError: too many indices for array: array is 0-dimensional, but 1
        were indexed`。这个属性的注解写着 `-> int`，`example/model.md` 还把它
        推荐为「训练前先看一眼缺失值」的入口，所以它是一条被文档化、被推荐、
        却从来没有跑通过的路（2026-09-07 修复，
        `tests/test_model_layer.py::test_num_null_counts_missing_cells_and_returns_an_int`
        锁住）。

        现在取的是 0 维数组本身，并显式 `int()` 兑现注解——`.item()` 出来的是
        numpy 标量，直接返回会让 `-> int` 继续说谎。
        """
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

        if isinstance(self.model, torch.nn.Module):
            torch.save(self.model.state_dict(), p)
        else:
            joblib.dump(self.model, p)

    def load(self, p: Path | str) -> Self:
        if isinstance(p, str):
            p = Path(p)

        if not p.exists():
            raise FileNotFoundError(f"{p} not found")

        self.model = self._init_model(
            num_symbols=self.num_symbols,
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.config.hyperparameters,
        ).to(self.device)

        if p.suffix == ".pth":
            self.model.load_state_dict(torch.load(p))
        elif p.suffix == ".joblib":
            self.model = joblib.load(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")
        return self

    def to_tensor(
        self, data: xr.Dataset, variables: list[str]
    ) -> torch.Tensor:
        """把 `(timestamp, symbol)` 面板转成 `[num_times, num_symbols, len(variables)]`。

        最后一维**严格按 `variables` 给定的顺序**排列，这是这个方法存在的全部理由。
        以前这段逻辑内联在 `_train_dl` 里，写的是
        `.sortby(["timestamp", "symbol", "variable"])`——`variable` 也被排进去了，
        于是最后一维变成**字母序**而不是调用方声明的顺序。它不报错、不警告，
        但把 `train_model.py` 里 `labels=[ret_30, ret_60, ret_120]` 的第一个标签
        换成了字母序最小的 `ret_120`，而 `RNNClassifier` 把 `y[:, :, 0]` 当作
        primary target——训了两个月的模型学的是 120 期收益。

        `timestamp` / `symbol` 仍然要排序：特征面板和标签面板的坐标顺序不保证一致，
        不排序就会「第 3 行的特征配上第 7 行的标签」。只有 `variable` 不能排。

        推理侧也应该走这个方法，而不是手抄一遍转换逻辑——训练和推理的列顺序一旦
        不一致，同样是静默错位。
        """
        return torch.from_numpy(
            data[variables]
            .to_dataarray()
            .sortby(["timestamp", "symbol"])
            .sel(variable=variables)
            .transpose("timestamp", "symbol", "variable")
            .values
        )

    def _predict_nn(self, data: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "Model not initialized, please call load() or train() first"
            )
        # 推理必须切 eval + no_grad。`load()` 新建的 nn.Module 默认处在 training
        # 模式，以前这里两样都没做：dropout 是开着的（train_model.py 的配置里
        # 首层就是 0.5），同一份输入每次调用给出的结果都不一样；而且整张计算图
        # 被留下来白吃内存。切过之后模型就留在 eval 模式——要继续训练的话，
        # 训练循环开头本来就会调 `self.model.train()`。
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            data = data.to(self.device)
            data = self._preprocess(data)
            return self.model(data)

    def predict(
        self, data: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        if isinstance(data, torch.Tensor):
            return self._predict_nn(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

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

    def _do_vecbt(self):
        if self.config.backtest_data is None:
            raise ValueError("Backtest dataset must be specified.")

        if self.config.backtest_data.config.symbols is None:
            raise ValueError("Backtest dataset symbols must be specified")

        data = self.config.backtest_data
        data.read()
        price = data.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).sel(timestamp=slice(self.config.test_start, self.config.test_end))

    def _train_dl(
        self,
        project_name: str,
        experiment_name: str,
        model_name: str,
        backtest: bool = False,
    ):
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

        # 这四个变量必须无条件初始化：epoch 循环末尾的 `if early_stopping: break`
        # 是无条件执行的，一旦只在 `if self.config.early_stopping:` 里绑定，
        # `early_stopping=False` 的普通配置就会在第一个 epoch 结束时抛
        # UnboundLocalError（见 tests/test_model_layer.py 的 A 用例）。
        best_loss = float("inf")
        early_stopping = False
        patience = self.config.early_stopping_patience
        counter = 0

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
                # 早停要比较的是「整个 epoch 的验证损失」。`_val_one_batch` 返回的
                # 是单个 batch 的损失，所以这里按样本数加权累加，循环结束后再折算成
                # 一个 epoch 级别的标量——counter 才是「连续多少个 epoch 没有改善」。
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

        self._save_model(
            Path(self.config.model_save_dir)
            / project_name
            / experiment_name
            / model_name,
        )

        if self._wandb_recorder:
            self._wandb_recorder.finish()

        # 训练结束后**保留** `self.model`：以前这里是 `del self.model`，于是
        # `train()` 之后紧接着 `predict()` 会抛「Model not initialized」，
        # 必须先把刚存下来的权重再 `load()` 回来，纯属多此一举。
        #
        # 优化器状态（Adam 的一阶/二阶动量，约 2 倍参数量）在训练之外没有任何
        # 用处，显式丢掉——这才是当初 `del` 想省的那部分显存。所有 `self.optim`
        # 的读取点都在 `_train_one_batch` 里，而 `_init_model_and_optim()` 会在
        # 下一次训练（包括 CV 的下一折）开头重新建一个。
        self.optim = None

    def _auto_train(self, project_name, experiment_name, model_name):
        if isinstance(self.config, DLConfig):
            logger.info(f"Training DL model: {model_name}")
            return self._train_dl(
                project_name=project_name,
                experiment_name=experiment_name,
                model_name=model_name,
            )
        elif isinstance(self.config, MLConfig):
            raise NotImplementedError("ML training not implemented")
        else:
            raise ValueError("Unsupported configuration type")

    def train(self):
        project_name = f"{self.class_name}_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_name = f"{self.class_name}_total"
        model_name = f"{experiment_name}.pth"
        self._init_wandb(
            project_name=project_name,
            experiment_name=experiment_name,
        )
        self._auto_train(
            project_name=project_name,
            experiment_name=experiment_name,
            model_name=model_name,
        )

    def train_cv(
        self,
        train_periods: int,
        gap_periods: int = 0,
        parallel: bool = False,
        njobs: int = -1,
    ):
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

        # Calculate number of folds
        total_periods = len(timestamps)
        test_periods = train_periods // 5  # Test set is 20% of training set
        n_splits = max(
            1, (total_periods - train_periods - gap_periods) // test_periods
        )

        logger.info(f"Total {n_splits} folds will be created")

        if parallel:
            # Prepare all fold configurations for parallel execution
            fold_configs = []
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
                train_start_ts = np.datetime_as_string(
                    timestamps[train_start_idx]
                )
                train_end_ts = np.datetime_as_string(
                    timestamps[train_end_idx - 1]
                )
                test_start_ts = np.datetime_as_string(
                    timestamps[test_start_idx]
                )
                test_end_ts = np.datetime_as_string(
                    timestamps[test_end_idx - 1]
                )

                experiment_name = f"{self.class_name}_cv_fold_{i}"

                fold_configs.append(
                    {
                        "fold_idx": i,
                        "train_start_ts": train_start_ts,
                        "train_end_ts": train_end_ts,
                        "test_start_ts": test_start_ts,
                        "test_end_ts": test_end_ts,
                        "experiment_name": experiment_name,
                        "project_name": project_name,
                    }
                )

                logger.info(
                    f"Fold {i}: Train [{train_start_ts} to {train_end_ts}], Test [{test_start_ts} to {test_end_ts}]"
                )

            # Execute folds in parallel
            logger.info(
                f"Starting parallel training of {len(fold_configs)} folds"
            )
            Parallel(n_jobs=njobs, backend="threading")(
                delayed(self._train_fold_with_config)(fold_config)
                for fold_config in fold_configs
            )
        else:
            # Sequential execution (original logic)
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
                train_start_ts = np.datetime_as_string(
                    timestamps[train_start_idx]
                )
                train_end_ts = np.datetime_as_string(
                    timestamps[train_end_idx - 1]
                )
                test_start_ts = np.datetime_as_string(
                    timestamps[test_start_idx]
                )
                test_end_ts = np.datetime_as_string(
                    timestamps[test_end_idx - 1]
                )

                experiment_name = f"{self.class_name}_cv_fold_{i}"
                self.config.train_start = train_start_ts
                self.config.train_end = train_end_ts
                self.config.test_start = test_start_ts
                self.config.test_end = test_end_ts

                logger.info(
                    f"Fold {i}: Train [{train_start_ts} to {train_end_ts}], Test [{test_start_ts} to {test_end_ts}]"
                )

                self._init_wandb(
                    project_name=project_name,
                    experiment_name=experiment_name,
                )
                self._auto_train(
                    project_name=project_name,
                    experiment_name=experiment_name,
                    model_name=f"{experiment_name}.pth",
                )

    def _train_fold_with_config(self, fold_config: dict):
        """Train a single fold with given configuration by creating a new model instance"""
        import copy

        # Create a deep copy of the current model with its own configuration
        fold_model = copy.deepcopy(self)

        # Update the fold model's configuration with fold-specific dates
        fold_model.config.train_start = fold_config["train_start_ts"]
        fold_model.config.train_end = fold_config["train_end_ts"]
        fold_model.config.test_start = fold_config["test_start_ts"]
        fold_model.config.test_end = fold_config["test_end_ts"]

        # Initialize wandb for this fold
        fold_model._init_wandb(
            project_name=fold_config["project_name"],
            experiment_name=fold_config["experiment_name"],
        )

        # Train this fold
        fold_model._auto_train(
            project_name=fold_config["project_name"],
            experiment_name=fold_config["experiment_name"],
            model_name=f"{fold_config['experiment_name']}.pth",
        )

    def _assert_shape_match_y(self, data: torch.Tensor):
        num_symbols, num_labels = (
            self.num_symbols,
            self.num_labels,
        )
        if num_symbols != data.shape[1] or num_labels != data.shape[2]:
            raise ValueError(
                f"Train y shape mismatch: [num_times, {num_symbols}, {num_labels}] vs {data.shape}"
            )

    def _assert_shape_match_x(self, data: torch.Tensor):
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
    ) -> torch.Tensor:
        """一次调用 = **一个 batch** 的测试步。

        调用点在 `_train_dl` 的 `for x_batch, y_batch in test_loader:` 里，外面
        已经是 `model.eval()` + `torch.no_grad()`。只记指标，返回值基类不使用。

        命名说明见 `_train_one_batch`。
        """
        ...

    @abstractmethod
    def _train_one_batch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """一次调用 = **一个 batch** 的完整优化步。

        调用点在 `_train_dl` 的 `for x_batch, y_batch in train_loader:` 里，
        实现方要自己走完 `zero_grad` → forward → loss → `backward` → `step`。
        `epoch` 只是透传下来的 epoch 序号，用于把指标 log 到正确的 step 上——
        它不表示「本次调用覆盖了一整个 epoch」。

        这三个钩子曾经叫 `_train_one_epoch` / `_val_one_epoch` /
        `_test_one_epoch`。那个名字不是无害的措辞问题：早停计数器一度被写在
        `_val_one_epoch` 的调用点旁边、照名字理解成「每个 epoch 执行一次」，
        实际却落在验证 batch 循环内部，于是 patience 数的是 batch 而不是 epoch
        （2026-09-07 修复，`tests/test_model_layer.py::
        test_early_stopping_patience_counts_epochs_not_batches` 锁住）。
        名字保留下去就是把同一个坑留给下一个读者，所以一并改名。

        **想做单步 / 在线训练的不要来改这里。** 这个钩子属于
        `_train_dl` 的批量训练循环。在线学习的入口是各模型头的 `update()`，
        它用 `_get_refit_optim()` 拿一个**跨调用复用**的微调优化器
        （复用是必要的：每步新建会把 AdamW 的动量清零）。
        """
        ...

    @abstractmethod
    def _val_one_batch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """一次调用 = **一个 batch** 的验证步。

        调用点在 `_train_dl` 的 `for x_batch, y_batch in val_loader:` 里，外面
        已经是 `model.eval()` + `torch.no_grad()`——不要再自己包一层，也不要
        backward。

        **必须返回一个能 `float()` 的标量 loss。** 基类把每个 batch 的返回值按
        样本数加权累加，循环结束后折算成一个 epoch 级别的验证损失，那个标量才是
        早停判据。返回 `None` 会在 `float(None)` 处直接 `TypeError`。

        命名说明见 `_train_one_batch`。
        """
        ...

    @abstractmethod
    def _preprocess(self, data: torch.Tensor) -> torch.Tensor: ...

    def _init_optim(self, model: torch.nn.Module):
        raise NotImplementedError

    def _get_refit_optim(self) -> torch.optim.Optimizer:
        """在线学习（`update()`）用的优化器，**跨调用复用**。

        每个 `update()` 以前都是现场 `torch.optim.AdamW(...)` 新建一个。AdamW 的
        一阶/二阶动量存在优化器实例里，所以「每步新建」等于每一步都把动量清零——
        它不报错，只是悄悄退化成一个带古怪 warmup 的 SGD。而 `update()` 的用途正是
        真正的在线 / 单步训练，动量的累积就是它的全部意义所在。

        **失效条件是 `self.model` 被换掉**：优化器持有的是参数张量的引用，
        `load()` 或再次 `_init_model()` 之后 `self.model` 指向一个全新的
        `nn.Module`，旧优化器手里那些张量已经跟当前模型无关了——继续拿它 step
        会静默地更新一堆游离张量，比原来的 bug 更糟。缓存因此按
        `(self.model 这个对象, lr_refit)` 命中：模型换了、或者学习率改了，都重建。
        用 `is` 比较对象身份而不是记一个「脏」标志，好处是 `load()` /
        `_init_model_and_optim()` 一行都不用改，也就不可能有人改了模型却忘了失效。

        这跟 `self.optim` 是**两回事**。`self.optim` 是训练循环的优化器，
        `_train_dl` 结束时会被显式置 None 释放显存（见上文），那是有意的；
        微调优化器是另一个生命周期，不要用这个方法去复活 `self.optim`。
        """
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

    def _vecbt(self, prices: pd.Series, signals: pd.Series):
        raise NotImplementedError
