# 收益模型

[English](../model.md) | 简体中文

收益模型根据一组因子预测标签（label），标签通常是未来收益或收益排名。因子和标签都是以 `(timestamp, symbol)` 为索引的 `xarray.Dataset` 面板（panel），预测结果与之形状相同，可以直接交给组合优化和回测环节。模型层提供统一的训练流程：数据收集、训练与测试窗口、walk-forward 交叉验证、检查点（checkpoint）、提前停止、评估指标以及 Weights & Biases 日志；具体的模型头（head）只需要实现拟合本身。

## 前置条件

示例都在 CPU 上运行，不需要联网。每次训练都会调用 Weights & Biases（W&B），本地实验时把它关掉。在 macOS 上还要在导入 `torch` 或 `xgboost` 之前限制 OpenMP 线程数（见注意事项）。

```bash
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1   # 仅 macOS
```

## 基础

### 输入与输出

模型头用一个因子对象列表（特征）和一个标签对象列表（目标）来配置，二者都来自因子层（见 factor 指南）。`collect()` 读取或计算它们，按 `(timestamp, symbol)` 合并，并把面板保存在内存中。模型头内部处理形状为 `[num_times, num_symbols, num_features]` 的数组，最后一个轴的顺序与 `get_factor_names()` 完全一致，输出形状为 `[num_times, num_symbols, num_labels]` 的预测。

下面的示例用一个内存中的小型替身来代替因子和标签对象，因此不需要任何数据存储。它只实现了模型层会调用的几个方法。标签是两个因子的带噪线性函数。

```python
>>> import numpy as np, xarray as xr
>>> from types import SimpleNamespace
>>> rng = np.random.default_rng(0)
>>> coords = {"timestamp": np.datetime64("2024-01-01") + np.arange(200),
...           "symbol": [f"S{i:02d}" for i in range(20)]}
>>> f_a, f_b = rng.standard_normal((2, 200, 20))
>>> ret = 0.05 * f_a - 0.02 * f_b + 0.05 * rng.standard_normal((200, 20))
>>> class Panel:
...     """因子或标签对象的最小替身。"""
...     def __init__(self, **variables):
...         data = {k: (("timestamp", "symbol"), v) for k, v in variables.items()}
...         self.ds = xr.Dataset(data, coords=coords)
...         self.config = SimpleNamespace(start_date=None, end_date=None)
...     def _reset_dataset_config(self): pass
...     def _get_factor_names(self): return list(self.ds.data_vars)
...     def read(self): return self
...     def get_features(self): return self.ds
...     def get_labels(self): return self.ds
...     def get_config(self): return {"factor_names": self._get_factor_names()}
>>> factor, label = Panel(f_a=f_a, f_b=f_b), Panel(ret=ret)
>>> from loguru import logger
>>> logger.remove()  # quantlab 把进度日志写到 stderr，这里关掉
```

### 训练与检查点

配置对象包含因子和标签对象、检查点的保存目录，以及四个日期：训练窗口和测试窗口（两端都包含）。训练窗口末尾的 `val_size` 比例（默认 0.2）被留作验证段。`factor_data_strategy` 和 `label_data_strategy` 决定是读取已存储的值（`"read"`）还是先计算（`"cal"`）。`XGBoostRegressor` 是树模型的模型头，`hyperparameters` 会传给它。

```python
>>> from quantlab.base.config import MLConfig
>>> from quantlab.ml_model.xgb import XGBoostRegressor
>>> config = MLConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     hyperparameters={"num_boost_round": 50, "max_depth": 3},
... )
>>> model = XGBoostRegressor(config).collect()
>>> model.num_times, model.num_symbols, model.get_factor_names(), model.get_label_names()
(200, 20, ['f_a', 'f_b'], ['ret'])
>>> checkpoint = model.train()
>>> checkpoint.name
'XGBoostRegressor_total.joblib'
```

`train()` 返回检查点的绝对路径。每次调用都会新建一个试验目录 `checkpoints/XGBoostRegressor_trial_<时间戳>/XGBoostRegressor_total/`，里面有检查点文件和旁边的 `config.json`。`config.json` 保存完整配置，以及一份 `trained_on` 记录：模型训练时见过的特征名、标签名和标的。

```python
>>> sorted(p.name for p in checkpoint.parent.iterdir())
['XGBoostRegressor_total.joblib', 'config.json']
>>> import json
>>> record = json.loads((checkpoint.parent / "config.json").read_text())
>>> record["trained_on"]["factor_names"], record["trained_on"]["label_names"], len(record["trained_on"]["symbols"])
(['f_a', 'f_b'], ['ret'], 20)
```

### 预测与加载

`predict_panel` 接收特征面板，返回一个面板，每个标签名对应一个变量。所有特征都为 NaN 的位置，预测也是 NaN。`predict` 是数组层面的对应接口：对 `XGBoostRegressor` 而言，输入 `[T, S, F]`，输出 `[T, S, L]`。拿不准时用 `predict_panel`，因为 `predict` 的数组约定由各个模型头自己决定。

```python
>>> predictions = model.predict_panel(factor.get_features())
>>> predictions
<xarray.Dataset> Size: 34kB
Dimensions:    (timestamp: 200, symbol: 20)
Coordinates:
  * timestamp  (timestamp) datetime64[s] 2kB 2024-01-01 ... 2024-07-18
  * symbol     (symbol) <U3 240B 'S00' 'S01' 'S02' 'S03' ... 'S17' 'S18' 'S19'
Data variables:
    ret        (timestamp, symbol) float64 32kB -0.001596 -0.01424 ... -0.03463
>>> model.predict(np.zeros((5, 20, 2))).shape
(5, 20, 1)
```

`load()` 把检查点恢复到一个用相同因子和标签构造的模型中。它先检查文件后缀，再把 `config.json` 里记录的变量名与模型自己声明的变量名做比较。

```python
>>> restored = XGBoostRegressor(config).load(checkpoint)
>>> bool((restored.predict_panel(factor.get_features())["ret"] == predictions["ret"]).all())
True
```

### 评估指标

`quantlab.utils.metrics` 对 `[T, S]` 面板打分，只有预测和目标同时有限的单元格才参与计算。除了 MSE、RMSE、MAE 和 R2，还有两个截面指标。IC 是同一时间点上、跨标的的预测与目标之间的 Pearson 相关系数，再对时间取平均。RankIC 在每个时间点的排名上做同样的计算，因此衡量的是排序能力，与量纲无关。树模型在主标签（第一个标签）上计算全部六个指标，覆盖训练、验证和测试三段，并以 `train_*`、`val_*`、`test_*` 的名字写入 W&B 运行摘要。

```python
>>> from quantlab.utils.metrics import regression_panel_metrics
>>> test = slice("2024-06-01", "2024-07-18")
>>> scores = regression_panel_metrics(
...     predictions["ret"].sel(timestamp=test).values,
...     label.ds["ret"].sel(timestamp=test).values,
... )
>>> {name: round(value, 3) for name, value in scores.items()}
{'mse': 0.003, 'rmse': 0.051, 'mae': 0.041, 'r2': 0.48, 'ic': 0.703, 'rank_ic': 0.683}
```

### 类层次

所有模型头都通过两个变体之一继承自 `BaseModel`。两个变体的差别在于训练框架，以及子类必须实现哪些方法；`train`、`train_cv`、`load`、`predict` 和 `predict_panel` 只在 `BaseModel` 中实现一次，子类不覆盖。

| 类 | 框架 | 配置 | 检查点 | 模型头需要实现的方法 |
|---|---|---|---|---|
| `DLModel` | torch，按 `DataLoader` 批次做 epoch 循环 | `DLConfig` | `.pth` | `_init_model`、`_init_optim`、`_preprocess`、`_train_one_batch`、`_val_one_batch`、`_test_one_batch` |
| `MLModel` | numpy，使用库自带的提前停止 | `MLConfig` | `.joblib` | `_init_model`、`_preprocess`、`_fit_model`、`_forward` |

自带的模型头有：`XGBoostRegressor`（`MLModel`）、`MLPRegressor`（作用在展平后的截面上的两隐层感知机）、`RNNRegressor`（每个标签一个 GRU 或 LSTM 塔，至少需要两个标签）和 `RNNClassifier`（预测收益的正负，并把上涨概率作为排序分数返回）。完整的配置字段见 `quantlab/base/model.py` 和 `quantlab/base/config.py` 的 docstring。

## 常见任务

### 提前停止

设置 `early_stopping=True` 后，当验证损失连续 `early_stopping_patience` 轮没有改善时停止训练（`MLModel` 的单位是 boosting 轮数，`DLModel` 的单位是 epoch），并保留最优模型。对 `XGBoostRegressor`，检查点会被截断到最优的那一轮。判据是验证段上的 pooled 一致性相关系数（concordance correlation）损失。

```python
>>> from dataclasses import replace
>>> stopping = replace(config, early_stopping=True, early_stopping_patience=5,
...                    hyperparameters={"num_boost_round": 500, "max_depth": 3})
>>> stopped = XGBoostRegressor(stopping).collect()
>>> _ = stopped.train()
>>> stopped.model.num_boosted_rounds(), stopped.model.best_iteration
(181, 180)
```

### walk-forward 交叉验证

`train_cv(train_periods, gap_periods)` 在 `start_date` 到 `end_date` 之间的时间戳上滑动训练窗口。测试段长 `train_periods // 5` 个时间戳，起点在训练段结束后再间隔 `gap_periods` 个时间戳；每一折向前平移一个测试段的长度。每一折都有自己的检查点和自己的 W&B 运行，返回值是每折一个字典，包含该折的日期（两端都包含）、检查点路径和 `test_*` 指标。

```python
>>> results = model.train_cv(train_periods=100, gap_periods=2)
>>> len(results)
4
>>> [(str(r["train_start"])[:10], str(r["test_start"])[:10], str(r["test_end"])[:10]) for r in results]
[('2024-01-01', '2024-04-12', '2024-05-01'), ('2024-01-21', '2024-05-02', '2024-05-21'), ('2024-02-10', '2024-05-22', '2024-06-10'), ('2024-03-01', '2024-06-11', '2024-06-30')]
>>> [round(r["test_rank_ic"], 3) for r in results]
[0.669, 0.68, 0.688, 0.66]
```

所有折共用一个试验目录。除了每折一个子目录，目录里还有 `cv_folds.json`，即包含 `format_version` 和折列表的清单文件。回测器根据这个文件回放一次交叉验证。

```python
>>> from pathlib import Path
>>> trial = Path(results[0]["checkpoint"]).parent.parent
>>> sorted(p.name for p in trial.iterdir())
['XGBoostRegressor_cv_fold_0', 'XGBoostRegressor_cv_fold_1', 'XGBoostRegressor_cv_fold_2', 'XGBoostRegressor_cv_fold_3', 'cv_folds.json']
>>> manifest = json.loads((trial / "cv_folds.json").read_text())
>>> manifest["format_version"], len(manifest["folds"])
(1, 4)
```

`parallel=True` 用线程并发训练各折（`njobs` 指定线程池大小）。每一折操作的是模型的深拷贝，因此内存占用随任务数增长。树模型库本身已经占满所有核心，建议把 `hyperparameters` 里的 `nthread` 设为大约 `os.cpu_count() // njobs`。

### 训练 torch 模型

`DLConfig` 增加了 `epochs`、`batch_size`、`lr` 和 `num_workers`（`DataLoader` 的工作进程数，默认 4；数据小且在内存中时用 0）。网络一次看到一个 bar 上的所有标的，因此会编码标的的位置。`predict_panel` 会先把输入对齐到训练时的标的：缺少其中任何一个，输入会被拒绝；多出来的标的会被丢弃并给出警告，也不会得到预测。

```python
>>> from quantlab.base.config import DLConfig
>>> from quantlab.dl_model.mlp import MLPRegressor
>>> dl_config = DLConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     epochs=30, batch_size=32, num_workers=0,
...     early_stopping=True, early_stopping_patience=5,
...     hyperparameters={"hidden_size1": 16, "hidden_size2": 8},
... )
>>> mlp = MLPRegressor(dl_config).collect()
>>> mlp_checkpoint = mlp.train()
>>> mlp_checkpoint.name
'MLPRegressor_total.pth'
>>> one_more = factor.get_features().isel(symbol=[0]).assign_coords(symbol=["S99"])
>>> wider = xr.concat([factor.get_features(), one_more], dim="symbol")
>>> mlp.predict_panel(wider).symbol.size
20
```

### 记录到 Weights & Biases

每次 `train()` 以及 `train_cv()` 的每一折都会打开一个 W&B 运行：运行名是实验名，所在项目名是试验目录名，并附带完整配置。`XGBoostRegressor` 记录每一轮的训练和验证曲线，把最终指标和各因子的重要性写入运行摘要；`train_cv` 还会额外打开一个 `<类名>_cv_summary` 运行，把各 `test_*` 指标的均值记为 `cv_mean_test_*`。torch 模型头每个 epoch 记录指标。`WANDB_MODE=disabled` 会关闭全部记录；`WANDB_MODE=offline` 把运行写到本地的 `wandb/` 目录，之后可以用 `wandb sync` 同步。两者都不设置时，`wandb.init` 需要已登录的账号。

## 扩展

新的模型头继承 `MLModel` 或 `DLModel`，实现上表列出的方法即可，其余都不用改。之后它就能使用 `train`、`train_cv`、`load`、`predict_panel` 和各个回测器。

`MLModel` 的模型头拿到的是数组形式的 `[T, S, F]` 特征和 `[T, S, L]` 标签。`_fit_model` 必须把拟合好的对象放到 `self.model` 里，检查点保存的就是这个对象（通过 joblib）。`_preprocess` 会作用在每一个数组上，包括标签，必须返回拷贝。真正的模型在拟合过程中才创建时，`_init_model` 可以返回 `None`。

```python
>>> from quantlab.base.model import MLModel
>>> class RidgeHead(MLModel):
...     """所有标的共用的闭式岭回归。"""
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         self.alpha = hyperparameters.get("alpha", 1.0)
...         return None  # 真正的模型在 _fit_model 里构建
...     def _preprocess(self, data):
...         return np.array(data, dtype=np.float64, copy=True)  # 返回拷贝，不原地修改
...     def _fit_model(self, train_x, train_y, val_x, val_y):
...         x = np.nan_to_num(train_x.reshape(-1, train_x.shape[-1]))
...         y = train_y.reshape(-1, train_y.shape[-1])
...         keep = np.isfinite(y).all(axis=1)  # 丢弃没有标签的行
...         x1 = np.c_[x[keep], np.ones(keep.sum())]  # 加一列截距
...         penalty = self.alpha * np.eye(x1.shape[1])
...         penalty[-1, -1] = 0.0  # 截距不做收缩
...         self.model = np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y[keep])
...     def _forward(self, x):
...         rows = np.nan_to_num(x.reshape(-1, x.shape[-1]))
...         out = np.c_[rows, np.ones(len(rows))] @ self.model
...         return out.reshape(x.shape[0], x.shape[1], -1)
>>> ridge = RidgeHead(replace(config, hyperparameters={"alpha": 1.0})).collect()
>>> ridge_results = ridge.train_cv(train_periods=100, gap_periods=2)
>>> [round(r["test_rank_ic"], 3) for r in ridge_results]
[0.678, 0.691, 0.697, 0.677]
>>> ridge.model.round(3).ravel().tolist()
[0.05, -0.019, 0.001]
```

`DLModel` 的模型头在 `_init_model` 里根据面板形状构建 `nn.Module`，基类负责把它放到对应设备上、运行 epoch 循环，并在开启提前停止时恢复最优 epoch 的权重。`_train_one_batch` 在一个 `[batch, num_symbols, num_features]` 的批次上执行一步优化。`_val_one_batch` 必须以标量形式返回验证损失，因为基类会把它平均成驱动提前停止的 epoch 损失。`_test_one_batch` 每个 epoch 在测试段上调用一次，模型头通常在这里记录指标。`_preprocess` 由训练和推理共用。

```python
>>> import torch, torch.nn as nn
>>> from quantlab.base.model import DLModel
>>> class LinearHead(DLModel):
...     """对一个 bar 上的每个标的应用同一个 nn.Linear。"""
...     def _init_model(self, num_symbols, num_features, num_labels, hyperparameters):
...         return nn.Linear(num_features, num_labels)
...     def _init_optim(self, model):
...         return torch.optim.Adam(model.parameters(), lr=self.config.lr)
...     def _preprocess(self, data):
...         return torch.nan_to_num(data, nan=0.0)
...     def _train_one_batch(self, epoch, x, y):
...         self.optim.zero_grad()
...         loss = nn.functional.mse_loss(self.model(x), y)
...         loss.backward()
...         self.optim.step()
...         return loss.detach()
...     def _val_one_batch(self, epoch, x, y):
...         return nn.functional.mse_loss(self.model(x), y)
...     def _test_one_batch(self, epoch, x, y):
...         return nn.functional.mse_loss(self.model(x), y)
>>> linear = LinearHead(replace(dl_config, epochs=300, lr=0.05, early_stopping_patience=20, hyperparameters={})).collect()
>>> linear_checkpoint = linear.train()
>>> [round(w, 3) for w in linear.model.weight[0].tolist()]
[0.05, -0.019]
>>> LinearHead.checkpoint_suffix, linear_checkpoint.suffix
('.pth', '.pth')
```

## 注意事项

下面的报错都是原样引用，路径缩写为 `...`。

模型头在构造的第一步就会拒绝错误的配置类。

```text
TypeError: XGBoostRegressor requires a MLConfig, got DLConfig
```

`predict` 和 `predict_panel` 需要已训练或已加载的模型。

```text
ValueError: Model not initialized, please call load() or train() first
```

`load` 会报告文件不存在、文件属于另一个变体，以及检查点记录的变量与模型声明的因子或标签不一致，因此被调换或替换的输入不会送进模型。声明的因子和标签要与训练时相同，顺序也相同。

```text
FileNotFoundError: checkpoints/missing.joblib not found
ValueError: Unsupported file type: '.pth'; XGBoostRegressor checkpoints use '.joblib' (...)
ValueError: XGBoostRegressor: checkpoint ... was trained on factor variables ['f_a', 'f_b'] (trained_on in its config.json), but this model declares ['f_z', 'f_b']; loading it would feed the model different or permuted inputs (...)
```

`predict_panel` 需要全部因子变量，torch 模型头还需要全部训练时的标的。

```text
ValueError: XGBoostRegressor.predict_panel: features are missing factor variable(s) ['f_b']
ValueError: MLPRegressor.predict_panel: the feature panel lacks 5 of the 20 symbols this model was trained on: ['S15', 'S16', 'S17', 'S18', 'S19']. A DL head encodes symbol position, so it cannot predict without them (...)
```

`train` 需要四个窗口日期齐全。在配置里设置它们，或者用 `train_cv`，它会为每一折设置日期。

```text
ValueError: Training and testing start and end dates must be specified.
```

`train_cv` 会用最后一折的日期覆盖配置里的四个 `train_*` 和 `test_*` 日期，之后再调用 `train()` 时请新建配置。如果 `train_periods` 太长、放不下测试段，它会记录一条 `Skipping fold 0: test set exceeds data range` 的日志，并返回空列表（`[]`），不会抛出异常。torch 模型头的 `train_cv` 不返回 `test_*` 指标，因此每折的字典里只有日期和路径，也不会打开汇总运行。

`train()` 只返回检查点路径。单次运行的测试指标记录在 W&B 摘要里；对 `MLModel` 的模型头，`train_cv` 会直接返回这些指标。

检查点是 pickle 文件（`MLModel` 用 `joblib`，`DLModel` 用 `torch.load`）。只加载自己生成或可信的文件。

进度信息通过 `loguru` 和 `tqdm` 输出到 stderr。`logger.remove()` 可以关掉日志行。在没有 CUDA 的机器上，`DataLoader` 可能打印一条 `pin_memory` 警告，可以忽略。

在 macOS 上，`xgboost` 的 wheel 链接的是 Homebrew 的 OpenMP 运行时，而 `torch` 自带另一份。同一个进程里同时使用两者可能崩溃或卡死。在第一次导入其中任何一个库之前设置 `OMP_NUM_THREADS=1` 可以避免，代价是树模型和 torch 代码变成单线程。Linux 不受影响。

## 另请参阅

factor 指南（`docs/factor.md`）介绍因子和标签如何生成，backtest 指南（`docs/backtest.md`）介绍 `predict_panel` 的输出和 `cv_folds.json` 清单如何进入回测。backend 指南（`docs/backend.md`）介绍面板使用的 Zarr 与 xarray 存储。API 细节见 `quantlab/base/model.py`、`quantlab/base/config.py`（`DLConfig`、`MLConfig`）、`quantlab/dl_model/`、`quantlab/ml_model/xgb.py`、`quantlab/ml_model/backend.py` 和 `quantlab/utils/metrics.py` 的 docstring。
