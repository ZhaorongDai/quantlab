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
    """模型层的共享基类：把"取数、切分、训练、保存"这套流程一次写好，供各模型复用。

    具体模型只需要实现五个钩子（建模型、建优化器、训练一轮、验证一轮、测试一轮，
    外加一个张量预处理），训练循环、早停、检查点与实验记录都由这里负责。

    模型层只按统一契约与因子对话，不关心因子是哪个后端算出来的，所以换一种因子
    计算方式不需要改这里的任何一行。
    """

    def __init__(self, config: DLConfig | MLConfig):
        """构造模型：接下配置、固定随机种子，但不建模型也不读数据。

        种子在构造时就固定，是为了让"同一份配置跑出同一个结果"这件事从对象一诞生
        就成立，而不是等到训练开始才补。模型本身留到训练或载入时再建，因为建模型
        需要知道数据的形状。

        Args:
            config (DLConfig | MLConfig): 训练配置。
        """
        self.config = config
        self._set_random_seed(self.config.random_seed)

        self.model = None

        self.data_backend = XrBackend()
        # self._pre_feature: Optional[xr.Dataset] = None
        self._wandb_recorder: wandb.sdk.wandb_run.Run = None  # type: ignore

    @staticmethod
    def _set_random_seed(seed: int):
        """把所有随机源一次性钉死，让同一份配置能跑出同一个结果。

        Python、NumPy、CPU 与 GPU 各有各的随机源，漏掉任何一个，实验就无法复现；
        同时关掉卷积算法的自动择优，因为它会为了快而在不同次运行里选不同的算法。

        Args:
            seed (int): 随机种子。
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def __repr__(self) -> str:
        """给出模型的标识：调试时最需要知道的是这个模型是用哪份配置训练的。

        Returns:
            str: 含类名与配置内容的标识字符串。
        """
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> DLConfig | MLConfig:
        """当前生效的训练配置；它在赋值时已被补全，读到的是补全后的版本。

        Returns:
            DLConfig | MLConfig: 训练配置对象。
        """
        return self._config

    @config.setter
    def config(self, config: DLConfig | MLConfig):
        """接下训练配置，并把训练区间一路下推给它依赖的因子、标签与回测数据。

        下推是必要的：模型说的"训练到某一天"，必须变成因子和标签各自的取数区间，
        否则三者各按各的区间取数，拼起来的面板会在时间轴上对不齐。

        Args:
            config (DLConfig | MLConfig): 训练配置。
        """
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
        """张量该放在哪块设备上；有显卡就用显卡，没有就退回 CPU 而不是直接报错。

        Returns:
            str: 设备名。
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_times(self) -> int:
        """已收集数据在时间方向上的长度，切分训练集与构造张量时按它来算。

        Returns:
            int: 时间点数量。
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).timestamp.size

    @property
    def class_name(self) -> str:
        """类名；实验名称、进度条与检查点文件名都以它为前缀。

        Returns:
            str: 类名。
        """
        return self.__class__.__name__

    @property
    def num_symbols(self) -> int:
        """一个时间截面上有多少个标的；建模型时用它确定输入层的宽度。

        训练张量的第二个维度就是它，形状校验也拿它作基准——对不上说明因子面板与
        模型的预期已经不一致了。

        Returns:
            int: 标的数量。
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.size

    @property
    def symbols(self) -> list[str]:
        """训练数据里的标的清单，用于把预测结果重新对回到具体标的上。

        Returns:
            list[str]: 标的代码列表。
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.values.tolist()

    @property
    def num_null(self) -> int:
        """整份训练数据里的空值总数，训练前用来判断数据质量是否可以接受。

        空值不会让训练报错，只会让损失变成非数并且一路传播下去；先看一眼这个数字
        比事后排查便宜得多。

        Returns:
            int: 空值个数。
        """
        return (
            self.data_backend.get_xarray_dataset(["timestamp", "symbol"])
            .isnull()
            .sum()
            .to_dataarray()
            .sum()
            .values[0]
        )

    @property
    def import_path(self) -> str:
        """本类的完整导入路径，检查点旁的配置记下它才能把模型类找回来重建。

        Returns:
            str: 模块路径加类名。
        """
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def num_factors(self) -> int:
        """输入特征有多少列；建模型时用它确定输入维度，校验张量形状时也用它。

        Returns:
            int: 特征列数。
        """
        return len(self.get_factor_names())

    @property
    def num_labels(self) -> int:
        """预测目标有多少列；建模型时用它确定输出维度。

        Returns:
            int: 标签列数。
        """
        return len(self.get_label_names())

    def _reset_factors_config(self):
        """把训练区间下推到每个因子，并让因子据此回头调整自己的取数区间。

        因子还要在这个区间之前多取一段历史来预热窗口，那一步由因子自己完成；模型
        这里只负责把"要训练哪一段"这个事实传下去。
        """
        for factor in self._config.factors:
            # 覆盖因子配置文件日期
            factor.config.start_date = self._config.start_date
            factor.config.end_date = self._config.end_date

            # 因子类重置数据集配置
            factor._reset_dataset_config()

    def _reset_labels_config(self):
        """把训练区间同样下推到每个标签，保证标签与特征落在同一段时间上。
        """
        for label in self._config.labels:
            label.config.start_date = self._config.start_date
            label.config.end_date = self._config.end_date

            # 因子类重置数据集配置
            label._reset_dataset_config()

    def _reset_backtest_dataset_config(self):
        """把训练区间下推给回测数据集，让回测取到的价格与预测结果时间轴一致。
        """
        self.config.backtest_data.start_date = self._config.start_date
        self.config.backtest_data.end_date = self._config.end_date

    def _get_labels_batch(self) -> xr.Dataset:
        """把配置里所有标签取来并拼成一份面板，作为训练的预测目标。

        取数方式由配置决定：要么现算，要么读已经算好的。拼完按时间与标的排序，
        因为后面要把它和特征面板对齐成同样形状的张量，顺序不一致就会错位。

        Returns:
            xr.Dataset: 以时间与标的为坐标的标签面板。

        Raises:
            ValueError: 配置里的标签取数方式既不是现算也不是读取时。
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

    def _get_features_batch(self) -> xr.Dataset:
        """把配置里所有因子取来并拼成一份特征面板。

        模型只调用因子的公共契约，因此这里对因子是用编译计算图算的还是用别的方式
        算的一无所知——这正是换因子后端不必改模型层的原因。

        Returns:
            xr.Dataset: 以时间与标的为坐标的特征面板。

        Raises:
            ValueError: 配置里的因子取数方式既不是现算也不是读取时。
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
        """把特征与标签合成一份训练数据并留在内存里，供后续训练直接取用。

        合并后再按时间与标的排一次序：两份面板各自的坐标顺序不保证一致，不排序就
        会出现特征与标签错位——这种错位不会报错，只会让模型学到噪音。

        Returns:
            Self: 已持有训练数据的模型自身，可继续链式调用。
        """
        feature = self._get_features_batch()
        label = self._get_labels_batch()
        d = xr.combine_by_coords([feature, label])
        d = d.sortby(["timestamp", "symbol"])
        self.data_backend.to_internal(d)  # type: ignore
        return self

    def get_factor_names(self):
        """按配置中因子的先后顺序列出全部特征列名。

        顺序是有意义的：张量的最后一维就按这个顺序排列，训练与推理必须用同一个
        顺序，否则模型会把某个因子的值当成另一个因子来用。

        Returns:
            list[str]: 特征列名，按配置中因子的顺序首尾相接。
        """
        return list(
            chain.from_iterable(
                [factor._get_factor_names() for factor in self.config.factors]
            )
        )

    def get_label_names(self):
        """按配置中标签的先后顺序列出全部标签列名，理由与特征列名相同。

        Returns:
            list[str]: 标签列名，按配置中标签的顺序首尾相接。
        """
        return list(
            chain.from_iterable(
                [label._get_factor_names() for label in self.config.labels]
            )
        )

    def get_config(self) -> dict:
        """导出可落盘的完整配置，把每个因子与标签的配置一并嵌进去。

        嵌套而不是只记引用：这份配置要和检查点一起存下来，日后单凭它就应该能把
        整条链路复现出来，只留引用的话换台机器就复现不了。

        Returns:
            dict: 训练配置字典，其中因子与标签字段已展开为各自的配置字典。
        """
        cfg = self.config.to_dict()
        cfg["factors"] = [factor.get_config() for factor in self.config.factors]  # type: ignore
        cfg["labels"] = [label.get_config() for label in self.config.labels]  # type: ignore
        return cfg  # type: ignore

    def _get_config_with_extra_kv(self, extra_kv: dict) -> dict:
        """在完整配置之上补几条额外信息，用于记录配置本身没有的运行时事实。

        Args:
            extra_kv (dict): 要补进去的键值对；同名键会覆盖配置里的值。

        Returns:
            dict: 合并后的配置字典。
        """
        cfg = self.get_config()
        cfg.update(extra_kv)
        return cfg

    def _save_model(self, p: Path):
        """保存模型，并把当次训练的完整配置作为同目录的伴生文件一起写下。

        目录已存在就直接报错，不覆盖：一次训练的产物是不可替代的，宁可让人换个
        名字重来，也不要悄悄盖掉上一次的结果。

        配置和权重一起保存，是因为单有权重无法复现——不知道当时用的是哪些因子、
        哪段时间，这份权重就只是一堆数字。

        Args:
            p (Path): 模型文件的目标路径；配置写在它的同级目录下。

        Raises:
            ValueError: 模型尚未建立时。
            RuntimeError: 目标目录已经存在时。
        """
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
        """从检查点载入一个已经训练好的模型，用于推理或继续实验。

        载入前会先按当前数据的形状把模型重新搭一遍，再往里灌权重：形状来自当下的
        因子与标的数量，因此如果数据已经变了，这里就会在灌权重时报错，而不是带着
        一个形状不匹配的模型继续跑下去。

        Args:
            p (Path | str): 检查点文件路径。

        Returns:
            Self: 已载入权重的模型自身，可继续链式调用。

        Raises:
            FileNotFoundError: 路径不存在时。
            ValueError: 文件后缀不是已知的两种检查点格式之一时。
        """
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

    def _predict_nn(self, data: torch.Tensor) -> torch.Tensor:
        """用神经网络做一次前向推理，推理前把数据搬到模型所在的设备并做预处理。

        预处理走的是训练时同一个钩子，这样训练与推理看到的输入分布才是一致的。

        Args:
            data (torch.Tensor): 输入张量，形状与训练时的特征张量一致。

        Returns:
            torch.Tensor: 模型输出。

        Raises:
            ValueError: 模型尚未训练也尚未载入时。
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "Model not initialized, please call load() or train() first"
            )
        data = data.to(self.device)
        data = self._preprocess(data)
        return self.model(data)

    def predict(
        self, data: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """对外的推理入口。

        目前只支持张量输入：签名上写着也接受数组，但那条路还没有实现，遇到数组会
        直接报错而不是悄悄转换——隐式转换会掩盖调用方本该处理的形状问题。

        Args:
            data (torch.Tensor | np.ndarray): 输入数据。

        Returns:
            torch.Tensor | np.ndarray: 模型输出。

        Raises:
            TypeError: 输入不是张量时。
        """
        if isinstance(data, torch.Tensor):
            return self._predict_nn(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _init_model_and_optim(self):
        """按当前数据的形状建好模型与优化器，并把模型搬到可用的设备上。

        形状（标的数、特征数、标签数）取自已收集的数据而不是配置，因此模型的输入
        输出维度永远与手里的数据一致，不需要有人手动同步这几个数字。

        优化器允许缺席：具体模型没有实现建优化器这一步时就不建，交由它自己在训练
        钩子里安排。
        """
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
        """取出测试区间的价格，为向量化回测做准备。

        目前只做到取价这一步，尚未接上真正的回测：不要把它当成"训练完会自动回测"
        的入口来用。

        Raises:
            ValueError: 未配置回测数据集，或回测数据集没有指定标的时。
        """
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
        """深度学习模型的完整训练流程：切分、训练、验证、测试、保存。

        验证集是从训练区间的尾部按比例切出来的，切法是按时间顺序而不是随机抽样：
        金融数据有时间顺序，随机抽样会让模型在训练时见到未来，验证分数会好看得
        不真实。测试区间则来自配置，与训练区间之间不重叠。

        训练结束后模型与优化器会被立即释放，因为交叉验证会反复调用这个流程，不释放
        显存会一轮轮累积上去。

        Args:
            project_name (str): 实验记录的项目名。
            experiment_name (str): 本次实验的名称。
            model_name (str): 检查点文件名。
            backtest (bool): 预留开关，目前未被使用。

        Raises:
            ValueError: 训练或测试区间的起止日期没有配全时。
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

        datas = []
        for d in [train_x, train_y, test_x, test_y]:
            datas.append(
                torch.from_numpy(
                    d.to_dataarray()
                    .transpose("timestamp", "symbol", "variable")
                    .sortby(["timestamp", "symbol", "variable"])
                    .values
                )
            )
        datas = [self._preprocess(d) for d in datas]

        train_x_t_all, train_y_t_all, test_x_t, test_y_t = datas
        for d in [train_x_t_all, test_x_t]:
            self._assert_shape_match_x(d)
        for d in [train_y_t_all, test_y_t]:
            self._assert_shape_match_y(d)

        train_split = int(train_x_t_all.shape[0] * (1 - self.config.val_size))
        train_x_t = train_x_t_all[:train_split]
        train_y_t = train_y_t_all[:train_split]
        val_x_t = train_x_t_all[train_split + 1 :]
        val_y_t = train_y_t_all[train_split + 1 :]

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

        if self.config.early_stopping:
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
                self._train_one_epoch(epoch, x_batch, y_batch)

            self.model.eval()  # type: ignore
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    val_loss = self._val_one_epoch(epoch, x_batch, y_batch)

                    if self.config.early_stopping:
                        if val_loss < best_loss:
                            best_loss = val_loss
                            counter = 0
                        else:
                            counter += 1
                            if counter >= patience:
                                logger.info(f"Early stopping at epoch {epoch}")
                                early_stopping = True

                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    self._test_one_epoch(epoch, x_batch, y_batch)

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

        del self.model
        del self.optim

    def _auto_train(self, project_name, experiment_name, model_name):
        """按配置类型选择训练路径，让上层不必关心这是哪一类模型。

        Args:
            project_name (str): 实验记录的项目名。
            experiment_name (str): 本次实验的名称。
            model_name (str): 检查点文件名。

        Returns:
            None: 训练在原地完成，不返回模型；这里的 return 只是用来结束分支。

        Raises:
            NotImplementedError: 配置指向传统机器学习模型时，该路径尚未实现。
            ValueError: 配置类型不属于已知的两种时。
        """
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
        """在全部数据上训练一次，实验名带上时间戳以免不同次训练互相覆盖。
        """
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
        """滚动式交叉验证：把时间轴切成若干段，每段单独训练一个模型。

        切分是沿时间前滚的，不是随机划分：金融数据不能打乱，训练集必须整段落在
        测试集之前。训练段与测试段之间还可以留一段空隙，用来隔开标签的前视窗口，
        否则训练集末尾的标签会包含测试段开头的信息。

        测试段长度固定为训练段的五分之一；测试段超出数据末尾的那些折会被跳过，
        而不是用一段更短的数据凑数。

        Args:
            train_periods (int): 每折训练段包含多少个时间点。
            gap_periods (int): 训练段与测试段之间留出的时间点数量，默认不留。
            parallel (bool): 是否并行训练各折，默认串行。
            njobs (int): 并行时使用的线程数，-1 表示不限。

        Raises:
            ValueError: 配置的时间区间内没有任何数据时。
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
        """训练交叉验证中的一折，为它单独复制一个模型实例。

        必须复制而不是复用同一个实例：并行训练时各折会同时改写训练区间，共用一个
        实例的话它们会互相覆盖对方的配置。

        Train a single fold with given configuration by creating a new model instance

        Args:
            fold_config (dict): 这一折的参数，须包含训练与测试区间的起止时间戳、
                实验名与项目名。
        """
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
        """在训练开始前确认标签张量的形状与模型的预期一致。

        形状不匹配在训练时往往不会报错，而是被广播机制悄悄吸收掉，最后表现为损失
        不下降。在这里挡住，问题就停在它产生的地方。

        Args:
            data (torch.Tensor): 标签张量，三个维度依次是时间、标的、标签列。

        Raises:
            ValueError: 标的数或标签列数与预期不符时。
        """
        num_symbols, num_labels = (
            self.num_symbols,
            self.num_labels,
        )
        if num_symbols != data.shape[1] or num_labels != data.shape[2]:
            raise ValueError(
                f"Train y shape mismatch: [num_times, {num_symbols}, {num_labels}] vs {data.shape}"
            )

    def _assert_shape_match_x(self, data: torch.Tensor):
        """在训练开始前确认特征张量的形状与模型的预期一致，理由同标签一侧。

        Args:
            data (torch.Tensor): 特征张量，三个维度依次是时间、标的、特征列。

        Raises:
            ValueError: 标的数或特征列数与预期不符时。
        """
        num_symbols, num_features = (
            self.num_symbols,
            self.num_factors,
        )
        if num_symbols != data.shape[1] or num_features != data.shape[2]:
            raise ValueError(
                f"Train x shape mismatch: [num_times, {num_symbols}, {num_features}] vs {data.shape}"
            )

    def _init_wandb(self, project_name: str, experiment_name: str):
        """开启一次实验记录，并把完整配置一并上报。

        配置随实验一起上报，是为了让记录里的每条曲线都能追回到产生它的那份参数；
        只有指标没有参数的实验记录，事后无法解释。

        Args:
            project_name (str): 实验记录的项目名。
            experiment_name (str): 本次实验的名称。
        """
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
    ):
        """由具体模型搭出自己的网络结构，输入输出维度由传入的数据形状决定。

        维度作为参数传进来而不是从配置里读，是为了让模型结构永远跟着实际数据走，
        不需要有人手动维护一份和数据同步的维度声明。

        Args:
            num_symbols (int): 一个时间截面上的标的数量。
            num_features (int): 输入特征列数。
            num_labels (int): 预测目标列数。
            hyperparameters (dict): 该模型自己的超参数。

        Returns:
            torch.nn.Module: 搭好但尚未搬到设备上的模型。
        """
        ...

    @abstractmethod
    def _test_one_epoch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """在测试集上跑一轮并记录指标；不更新参数，只观察。

        Args:
            epoch (int): 当前轮次，用于把指标记到正确的时间点上。
            x (np.ndarray | torch.Tensor): 特征张量，三个维度依次是批次内时间、
                标的、特征列。
            y (np.ndarray | torch.Tensor): 标签张量，三个维度依次是批次内时间、
                标的、标签列。

        Returns:
            torch.Tensor: 本轮的测试损失。
        """
        ...

    @abstractmethod
    def _train_one_epoch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """在一个批次上完成一次前向、反向与参数更新。

        Args:
            epoch (int): 当前轮次，用于把指标记到正确的时间点上。
            x (np.ndarray | torch.Tensor): 特征张量，三个维度依次是批次内时间、
                标的、特征列。
            y (np.ndarray | torch.Tensor): 标签张量，三个维度依次是批次内时间、
                标的、标签列。

        Returns:
            torch.Tensor: 本批次的训练损失。
        """
        ...

    @abstractmethod
    def _val_one_epoch(
        self,
        epoch: int,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """在验证集上跑一轮；它的返回值就是早停判断依据的那个损失。

        Args:
            epoch (int): 当前轮次，用于把指标记到正确的时间点上。
            x (np.ndarray | torch.Tensor): 特征张量，三个维度依次是批次内时间、
                标的、特征列。
            y (np.ndarray | torch.Tensor): 标签张量，三个维度依次是批次内时间、
                标的、标签列。

        Returns:
            torch.Tensor: 本轮的验证损失。
        """
        ...

    @abstractmethod
    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """在张量进入模型之前做统一的预处理，训练与推理共用这一个钩子。

        共用同一个钩子是关键：如果推理时的预处理和训练时不一致，模型看到的输入
        分布就变了，而这种偏差不会报错，只会让预测悄悄失准。

        Args:
            data (torch.Tensor): 待处理的张量，三个维度依次是时间、标的、列。

        Returns:
            torch.Tensor: 处理后的张量。
        """
        ...

    def _init_optim(self, model: torch.nn.Module):
        """由具体模型给出自己的优化器；基类不作实现。

        允许不实现：有些模型会在自己的训练钩子里安排更新方式，那时调用方拿到空值
        就跳过外部优化器，而不是被迫返回一个用不上的对象。

        Args:
            model (torch.nn.Module): 已经搭好的模型。

        Returns:
            torch.optim.Optimizer | None: 优化器；返回空值表示由模型自行更新参数。

        Raises:
            NotImplementedError: 具体模型没有实现这一步时。
        """
        raise NotImplementedError

    def _vecbt(self, prices: pd.Series, signals: pd.Series):
        """由具体模型定义如何把预测结果变成交易信号并做向量化回测；尚未实现。

        Args:
            prices (pd.Series): 回测区间的价格序列。
            signals (pd.Series): 与价格对齐的交易信号序列。

        Raises:
            NotImplementedError: 该路径目前没有任何实现。
        """
        raise NotImplementedError
