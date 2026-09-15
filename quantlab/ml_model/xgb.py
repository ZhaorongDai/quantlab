"""XGBoost 回归头：用 xgboost 原生早停预测未来收益（260914-lno）。

文件名刻意叫 `xgb.py`：叫 `xgboost.py` 会在包内遮蔽顶层的 `xgboost` 包。
"""

import numpy as np
import xgboost as xgb
from loguru import logger

from quantlab.base.config import MLConfig
from quantlab.base.model import MLModel

#: sklearn 风格别名 -> xgboost 原生（`xgb.train`）键名。
#:
#: 必须在合并默认参数**之前**、只在用户字典上归一化：xgboost 对这些别名的处理
#: 不一致（3.4.1 实测）——`learning_rate` 与默认 `eta` 同时出现时谁生效只取决于
#: 字典顺序；`n_estimators` 被忽略、只给一条 "not used" 警告，轮数仍取
#: `num_boost_round`；`random_state` 在已有 `seed` 时被**静默**忽略，连警告都没有。
_PARAM_ALIASES: dict[str, str] = {
    "n_estimators": "num_boost_round",
    "learning_rate": "eta",
    "random_state": "seed",
    "n_jobs": "nthread",
    "reg_alpha": "alpha",
    "reg_lambda": "lambda",
}

#: 训练结束写进 wandb summary 的 `Booster.get_score` 重要性类型：分裂次数、
#: 平均每次分裂的增益、总增益（见 `XGBoostRegressor._record_feature_importance`）。
_IMPORTANCE_TYPES: tuple[str, ...] = ("weight", "gain", "total_gain")


class _WandbEvalCallback(xgb.callback.TrainingCallback):
    """逐轮把 xgboost 的 eval 结果写进模型头**当前**的 wandb run。

    构造时只持有模型头的引用，每轮调用时再去读 `head._wandb_recorder`：并行交叉
    验证下每折是 deepcopy 出来的头，而回调是在 `_fit_model` 里现场用 `self`
    构造的，所以它指向的是折副本和折副本自己的 run。

    键用 xgboost 原生的连字符形式（`train-rmse`、`val-rmse`），`step` 是轮次
    （从 0 开始）；刻意与 `MLModel._evaluate` 写进 summary 的下划线形式
    （`val_rmse`）区分——前者是曲线，后者是最终值。
    """

    def __init__(self, head: "XGBoostRegressor"):
        super().__init__()
        self._head = head

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        recorder = self._head._wandb_recorder
        if recorder is not None:
            row = {
                f"{data_name}-{metric}": float(values[-1])
                for data_name, metrics in evals_log.items()
                for metric, values in metrics.items()
            }
            recorder.log(row, step=epoch)
        return False


class XGBoostRegressor(MLModel):
    """XGBoost 回归头：预测未来收益，输出 `[T, S, L]`。

    训练
        `xgb.train` 在展平后的 `(T*S, F)` 行上训练；标签含 NaN 的行被丢弃（任一
        标签缺失即丢弃整行），特征里的 ±inf 转成 NaN 交给 xgboost 当缺失值处理
        （xgboost 遇到 inf 会直接报错）。

    早停
        `config.early_stopping=True` 且验证段里有标签有限的行时，追加
        `xgb.callback.EarlyStopping(rounds=patience, data_name="val", save_best=True)`：

        - patience 按 **boosting 轮数**计，不是 epoch；
        - 判据是验证集上的 `eval_metric`（默认 RMSE），**不是 IC**；
        - `save_best=True` 让返回的 Booster 已截断到 `best_iteration + 1` 棵树，
          落盘的 `.joblib` 就是最优模型；`best_iteration` / `best_score` 同时写进
          wandb summary。

        没有可用的验证段时记一条 warning，跳过早停，训练满 `num_boost_round` 轮。

    超参（`config.hyperparameters`）
        `num_boost_round`（默认 1000）单独取出，不进 xgboost params；其余键逐键
        覆盖 `DEFAULT_PARAMS`，用户键优先，未指定的键保留默认。`seed` 默认取
        `config.random_seed`。sklearn 风格别名（`n_estimators`、`learning_rate`、
        `random_state`、`n_jobs`、`reg_alpha`、`reg_lambda`）先归一化成原生键再
        合并；别名与原生键同时给出时抛 `ValueError`。`config.hyperparameters`
        本身不会被修改——它记录用户输入；实际生效的参数（含轮数）经
        `_resolved_hyperparameters` 写进 `config.json` 的
        `resolved_hyperparameters` 与 wandb run config，日后 `DEFAULT_PARAMS`
        改了也能复现这次训练。

    多标签
        每个标签一个输出（xgboost 多输出回归）。头条指标（`{split}_ic` 等）只看
        主标签，即最后一维第 0 个标签。

    wandb
        逐轮曲线 `train-rmse` / `val-rmse`（连字符，`step=iteration`）；训练结束
        summary 里是 `{split}_loss` 与 `{split}_{mse,rmse,mae,r2,ic,rank_ic}`
        （下划线），以及早停启用时的 `best_iteration` / `best_score`。

        另有每个因子的重要性 `importance_{weight,gain,total_gain}/{因子名}`
        （03.7-18，G-03.7-9）：训练结束后对 Booster 调 `get_score`，只进 summary，
        从不进逐轮曲线。Booster 不带特征名，键 `f{i}` 按列下标映射到
        `get_factor_names()[i]`；从未分裂的因子补 0.0。多标签（多输出）模型的
        重要性由 xgboost 在各输出之间汇总，不分标签。变量顺序由
        `BaseModel.load()` 核对，不靠 xgboost 的特征名。重要性尽力而为：
        `gblinear` 整段跳过，拿不到或非标量的类型记 warning 后跳过，从不因此
        丢掉 checkpoint（REVIEW CR-01）。

    交叉验证
        `train_cv` 继承自 `BaseModel`：每折在折内训练段尾部的验证段上独立做
        原生早停，每折一个 `.joblib`。`parallel=True` 时各折在 joblib threading
        后端上并发，而 xgboost 默认用满全部核，会造成 CPU 超额订阅——建议在
        hyperparameters 里设 `nthread ≈ 核数 // njobs`。本类原样透传 `nthread`，
        不会替用户改写。

    用法::

        from quantlab.base.config import MLConfig
        from quantlab.label.fret import Return
        from quantlab.ml_model.xgb import XGBoostRegressor

        model = XGBoostRegressor(
            MLConfig(
                factors=[...],
                labels=[Return(...)],
                model_save_dir="checkpoints",
                factor_data_strategy="read",
                label_data_strategy="read",
                train_start="2020-01-01", train_end="2023-12-31",
                test_start="2024-01-01", test_end="2024-12-31",
                early_stopping=True,
                early_stopping_patience=50,
                hyperparameters={"num_boost_round": 1000},
            )
        ).collect()
        model.train()

        # 滚动交叉验证：4 折并发，每折 xgboost 用 核数 // 4 个线程
        results = model.train_cv(
            train_periods=500, gap_periods=5, parallel=True, njobs=4
        )  # 同时在 hyperparameters 里设 "nthread": os.cpu_count() // 4
    """

    DEFAULT_PARAMS: dict = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "device": "cpu",
        "eval_metric": "rmse",
    }
    DEFAULT_NUM_BOOST_ROUND = 1000

    def __init__(self, config: MLConfig):
        super().__init__(config)
        self._params: dict | None = None
        self._num_boost_round: int | None = None

    @staticmethod
    def _normalize_aliases(hyperparameters: dict) -> dict:
        """返回用户超参的副本，sklearn 风格别名换成原生键名（见 `_PARAM_ALIASES`）。

        别名与原生键同时出现（如 `eta` 与 `learning_rate`）时抛 `ValueError`，
        点名两个键——绝不静默挑一个。入参不被修改。
        """
        user = dict(hyperparameters)
        for alias, canonical in _PARAM_ALIASES.items():
            if alias not in user:
                continue
            if canonical in user:
                raise ValueError(
                    f"hyperparameters set both {alias!r} and {canonical!r}, which "
                    f"are the same XGBoost parameter; keep only one of them."
                )
            user[canonical] = user.pop(alias)
        return user

    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """解析超参；Booster 要到 `_fit_model` 里由 `xgb.train` 建出，这里返回 None。

        顺序：先在用户字典副本上归一化别名，再取出 `num_boost_round`，最后逐键
        覆盖 `DEFAULT_PARAMS`（用户键优先，未指定的键保留默认）。所以别名同样能
        覆盖默认值，例如 `learning_rate=0.3` 覆盖默认 `eta=0.05`、`random_state=7`
        覆盖 `config.random_seed`。
        """
        user = self._normalize_aliases(hyperparameters)
        num_boost_round = int(
            user.pop("num_boost_round", self.DEFAULT_NUM_BOOST_ROUND)
        )
        if num_boost_round < 1:
            raise ValueError(
                f"num_boost_round must be >= 1, got {num_boost_round}"
            )
        self._num_boost_round = num_boost_round
        self._params = {
            **self.DEFAULT_PARAMS,
            "seed": self.config.random_seed,
            **user,
        }
        return None

    def _resolved_hyperparameters(self) -> dict | None:
        """实际交给 `xgb.train` 的参数（默认值 + 用户覆盖 + 别名归一化）加轮数。"""
        if self._params is None:
            return None
        return {**self._params, "num_boost_round": self._num_boost_round}

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """返回新的 float32 数组，±inf 替换为 NaN，NaN 保留；不修改入参。"""
        out = np.array(data, dtype=np.float32, copy=True)
        out[np.isinf(out)] = np.nan
        return out

    @staticmethod
    def _to_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`[T,S,F]`、`[T,S,L]` 展平成 `(T*S, F)`、`(T*S, L)`，只保留标签全部有限的行。"""
        n_times, n_symbols, n_features = x.shape
        x_rows = x.reshape(n_times * n_symbols, n_features)
        y_rows = y.reshape(n_times * n_symbols, y.shape[-1])
        keep = np.isfinite(y_rows).all(axis=1)
        return x_rows[keep], y_rows[keep]

    def _fit_model(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> None:
        x_rows, y_rows = self._to_rows(train_x, train_y)
        if x_rows.shape[0] == 0:
            raise ValueError(
                "The training segment has no rows with finite labels."
            )
        dtrain = xgb.DMatrix(x_rows, label=y_rows)
        evals = [(dtrain, "train")]

        dval = None
        if val_x is not None:
            val_x_rows, val_y_rows = self._to_rows(val_x, val_y)
            if val_x_rows.shape[0] > 0:
                dval = xgb.DMatrix(val_x_rows, label=val_y_rows)
                evals.append((dval, "val"))
            else:
                logger.warning(
                    f"{self.class_name}: the validation segment has no rows "
                    "with finite labels; training without a validation set."
                )

        # 顺序必须是记录回调在前、EarlyStopping 在后：xgboost 的回调容器按短路
        # 方式依次调用，排在 EarlyStopping 之后的回调会漏掉触发停止的那一轮
        # （xgboost 3.4.1 实测）。
        callbacks: list[xgb.callback.TrainingCallback] = [
            _WandbEvalCallback(self)
        ]
        use_early_stopping = bool(self.config.early_stopping) and dval is not None
        if use_early_stopping:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self.config.early_stopping_patience,
                    data_name="val",
                    save_best=True,
                )
            )
        elif self.config.early_stopping:
            logger.warning(
                f"{self.class_name}: early_stopping=True but there is no usable "
                f"validation segment; early stopping skipped, training all "
                f"{self._num_boost_round} rounds."
            )

        self.model = xgb.train(
            self._params,
            dtrain,
            num_boost_round=self._num_boost_round,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=False,
        )

        if use_early_stopping and self._wandb_recorder is not None:
            self._wandb_recorder.summary.update(
                {
                    "best_iteration": int(self.model.best_iteration),
                    "best_score": float(self.model.best_score),
                }
            )

        if self._wandb_recorder is not None:
            self._record_feature_importance()

    def _record_feature_importance(self) -> None:
        """把每个因子的重要性写进 wandb summary（03.7-18，G-03.7-9）。

        Booster 不带特征名：`get_score` 的键是 `f{i}`，`i` 是列下标，而
        `_fit_model` 按 `get_factor_names()` 的顺序排列，所以 `f{i}` 就是第 `i`
        个因子名。从未分裂的因子 `get_score` 不返回，这里补 0.0。只走
        `summary.update`，从不 `log`，逐轮曲线的 step 保持连续。早停时
        `self.model` 已是截断后的 Booster，重要性描述的就是落盘的模型。

        键解析不成 `f<int>` 或下标越界时抛 `ValueError`：那说明 Booster 不是这里
        按列顺序训练出来的，悄悄丢掉会把重要性记错到别的因子上。

        重要性只是遥测，尽力而为（REVIEW CR-01）：本方法在 `_fit_model` 里、
        `_save_model` 之前运行，一个合法配置在这里抛异常就会丢掉整次训练的
        checkpoint。所以：

        - `booster="gblinear"` 没有分裂重要性（`gain` 直接 XGBoostError，多标签
          时 `weight` 还返回每个输出一个值的列表），整段跳过，记一条 info；
        - 某个重要性类型 `get_score` 报 XGBoostError，或返回的不是标量（多输出
          按输出分列），记一条 warning 并跳过**该类型**，其余类型照常写入；
          绝不对列表做 `float()`。
        """
        booster_type = str((self._params or {}).get("booster", "gbtree"))
        if booster_type == "gblinear":
            logger.info(
                f"{self.class_name}: booster='gblinear' has no split feature "
                f"importance; importance recording skipped."
            )
            return

        names = [str(name) for name in self.get_factor_names()]
        for importance_type in _IMPORTANCE_TYPES:
            try:
                scores = self.model.get_score(  # type: ignore[union-attr]
                    importance_type=importance_type
                )
            except xgb.core.XGBoostError as exc:
                logger.warning(
                    f"{self.class_name}: feature importance {importance_type!r} "
                    f"is unavailable for this Booster and was skipped: {exc}"
                )
                continue
            values = {name: 0.0 for name in names}
            non_scalar_key = None
            for key, score in scores.items():
                digits = key[1:] if key.startswith("f") else ""
                index = int(digits) if digits.isdigit() else -1
                if not 0 <= index < len(names):
                    raise ValueError(
                        f"{self.class_name}: Booster.get_score returned feature "
                        f"key {key!r}, which is not f<index> for one of the "
                        f"{len(names)} factors {names}."
                    )
                if not np.isscalar(score):
                    non_scalar_key = key
                    break
                values[names[index]] = float(score)
            if non_scalar_key is not None:
                logger.warning(
                    f"{self.class_name}: feature importance {importance_type!r} "
                    f"returned a non-scalar score for {non_scalar_key!r} "
                    f"(one value per output); skipped."
                )
                continue
            self._wandb_recorder.summary.update(
                {
                    f"importance_{importance_type}/{name}": value
                    for name, value in values.items()
                }
            )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """`[T, S, F]` -> `[T, S, L]`。

        先走 `inplace_predict`（树模型的快路径）；`booster="gblinear"` 不支持
        inplace 预测（xgboost 3.4.1 实测报 "Inplace predict is not supported by
        the current booster"），此时退回 `predict(DMatrix)`。两条路径 NaN 都当
        缺失值，树模型上结果一致。否则 gblinear 会在 `_fit` 的 `_evaluate`
        （`_save_model` 之前）报错，照样丢掉 checkpoint（REVIEW CR-01）。
        """
        n_times, n_symbols, n_features = x.shape
        rows = x.reshape(n_times * n_symbols, n_features)
        try:
            pred = self.model.inplace_predict(rows)  # type: ignore[union-attr]
        except xgb.core.XGBoostError as exc:
            if "Inplace predict is not supported" not in str(exc):
                raise
            pred = self.model.predict(xgb.DMatrix(rows))  # type: ignore[union-attr]
        return np.asarray(pred).reshape(n_times, n_symbols, -1)
