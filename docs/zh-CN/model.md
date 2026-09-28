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

模型头用一个因子对象列表（特征）和一个标签对象列表（目标）来配置，二者都来自因子层（见 factor 指南）。`collect()` 用 `read(start, end)` 或 `compute(start, end)` 向每个对象请求模型 `start_date` 到 `end_date` 的面板，按 `(timestamp, symbol)` 合并，并把面板保存在内存中。模型头内部处理形状为 `[num_times, num_symbols, num_features]` 的数组，最后一个轴的顺序与 `get_factor_names()` 完全一致，输出形状为 `[num_times, num_symbols, num_labels]` 的预测。

标签会读取 t 之后的 bar，因此绝不能用作特征。模型靠 `lookahead_bars()` 区分这两种角色：每个标签都有这个方法，因子都没有；构造时，因子列表里出现标签、或标签列表里出现因子，都会被拒绝（见注意事项）。真实的标签是 `quantlab.label.forward.Forward`，即向前平移 `delay + span` 个 bar 的因子，`lookahead_bars()` 返回这个和（见 factor 指南）。

下面的示例用一个内存中的小型替身来代替因子和标签对象，因此不需要任何数据存储。它只实现了模型层会调用的几个方法。标签是两个因子的带噪线性函数，它的替身报告 2 个 bar 的前瞻（lookahead），与默认 delay 为 1 的一期未来收益相同。

```python
>>> import numpy as np, xarray as xr
>>> rng = np.random.default_rng(0)
>>> coords = {"timestamp": np.datetime64("2024-01-01") + np.arange(200),
...           "symbol": [f"S{i:02d}" for i in range(20)]}
>>> f_a, f_b = rng.standard_normal((2, 200, 20))
>>> ret = 0.05 * f_a - 0.02 * f_b + 0.05 * rng.standard_normal((200, 20))
>>> class Panel:
...     """因子对象的最小替身。"""
...     def __init__(self, **variables):
...         data = {k: (("timestamp", "symbol"), v) for k, v in variables.items()}
...         self.ds = xr.Dataset(data, coords=coords)
...     def _get_factor_names(self): return list(self.ds.data_vars)
...     def read(self, start, end): return self.ds.sel(timestamp=slice(start, end))
...     def get_config(self): return {"factor_names": self._get_factor_names()}
>>> class LabelPanel(Panel):
...     """标签的最小替身，它在 t 的值读取 t+1 和 t+2 两个 bar。"""
...     def lookahead_bars(self): return 2
>>> factor, label = Panel(f_a=f_a, f_b=f_b), LabelPanel(ret=ret)
>>> from loguru import logger
>>> logger.remove()  # quantlab 把进度日志写到 stderr，这里关掉
```

### 训练与检查点

配置对象包含因子和标签对象、检查点的保存目录，以及四个日期：训练窗口和测试窗口（两端都包含）。训练窗口末尾的 `val_size` 比例（默认 0.2）被留作验证段。`factor_data_strategy` 和 `label_data_strategy` 决定是读取已存储的值（`"read"`）还是先计算（`"cal"`）。`XGBoostRegressor` 是树模型的模型头，`hyperparameters` 会传给它。

```python
>>> from quantlab.base.config import ModelConfig
>>> from quantlab.library_model.xgb import XGBoostRegressor
>>> config = ModelConfig(
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

`train()` 返回检查点的绝对路径。每次调用都会新建一个试验目录 `checkpoints/XGBoostRegressor_trial_<时间戳>/XGBoostRegressor_total/`，里面有检查点文件、旁边的 `config.json`、`metrics.json`、`ic_series.csv` 和 `test_predictions.zarr`。`config.json` 保存完整配置，以及一份 `trained_on` 记录：模型训练时见过的特征名、标签名和标的。`metrics.json` 保存这次运行的评分（见下文“评估指标”），另外两个文件保存逐 bar 的 IC 序列和测试段的预测（见下文“IC 序列与保存的预测”）。

```python
>>> sorted(p.name for p in checkpoint.parent.iterdir())
['XGBoostRegressor_total.joblib', 'config.json', 'ic_series.csv', 'metrics.json', 'test_predictions.zarr']
>>> import json
>>> record = json.loads((checkpoint.parent / "config.json").read_text())
>>> record["trained_on"]["factor_names"], record["trained_on"]["label_names"], len(record["trained_on"]["symbols"])
(['f_a', 'f_b'], ['ret'], 20)
```

### 清除标签前瞻

bar t 上的标签会读到 t + L 为止的 bar，L 是模型所有标签中最大的 `lookahead_bars()`。因此每个切分边界都会丢掉前一段的最后 L 个 bar（purge），使参与拟合的标签不会读到后一段的任何 bar。`train()` 先按位置把训练窗口切成训练段和验证段，再对训练/验证、验证/测试两个边界做清除；测试段保留全部 bar。`val_size=0` 时，训练段直接相对测试段清除。所以每个边界都会让拟合少用 L 个 bar。L 不是参数，而是由标签决定。

上面的示例中 L 为 2。训练窗口 2024-01-01 到 2024-05-31 共 152 个 bar；前 121 个（到 2024-04-30）用于训练，其余 31 个用于验证。清除之后，训练段止于 2024-04-28，验证段止于 2024-05-29。切分由 `quantlab.utils.split.purge_segments` 完成，下文的 walk-forward 各折也用它。

```python
>>> from quantlab.utils.split import purge_segments
>>> train_bars, val_bars, test_bars = purge_segments(
...     coords["timestamp"],
...     [("2024-01-01", "2024-04-30"), ("2024-05-01", "2024-05-31"), ("2024-06-01", "2024-07-18")],
...     label.lookahead_bars(),
... )
>>> len(train_bars), str(train_bars[-1]), len(val_bars), str(val_bars[-1]), len(test_bars)
(119, '2024-04-28', 29, '2024-05-29', 48)
```

### 预测与加载

`predict_panel` 接收特征面板，返回一个面板，每个标签名对应一个变量。所有特征都为 NaN 的位置，预测也是 NaN。`predict` 是数组层面的对应接口：对 `XGBoostRegressor` 而言，输入 `[T, S, F]`，输出 `[T, S, L]`。拿不准时用 `predict_panel`，因为 `predict` 的数组约定由各个模型头自己决定。

```python
>>> predictions = model.predict_panel(factor.ds)
>>> predictions
<xarray.Dataset> Size: 34kB
Dimensions:    (timestamp: 200, symbol: 20)
Coordinates:
  * timestamp  (timestamp) datetime64[s] 2kB 2024-01-01 ... 2024-07-18
  * symbol     (symbol) <U3 240B 'S00' 'S01' 'S02' 'S03' ... 'S17' 'S18' 'S19'
Data variables:
    ret        (timestamp, symbol) float64 32kB 0.002621 -0.01783 ... -0.04321
>>> model.predict(np.zeros((5, 20, 2))).shape
(5, 20, 1)
```

`load()` 把检查点恢复到一个用相同因子和标签构造的模型中。它先检查文件后缀，再把 `config.json` 里记录的变量名与模型自己声明的变量名做比较。

```python
>>> restored = XGBoostRegressor(config).load(checkpoint)
>>> bool((restored.predict_panel(factor.ds)["ret"] == predictions["ret"]).all())
True
```

### 评估指标

`quantlab.utils.metrics` 对 `[T, S]` 面板打分，只有预测和目标同时有限的单元格才参与计算。除了 MSE、RMSE、MAE 和 R2，还有两个截面指标。IC 是同一时间点上、跨标的的预测与目标之间的 Pearson 相关系数，再对时间取平均。RankIC 在每个时间点的排名上做同样的计算，因此衡量的是排序能力，与量纲无关。某个时间点上预测和目标同时有限的标的少于两个，或者预测或目标在截面上是常数时，这个时间点没有 IC，求平均时直接跳过，而不是当作 0。ICIR 和 RankICIR 衡量信号的稳定性：逐时间点 IC（或 RankIC）的均值除以它的样本标准差（`ddof=1`）。有 IC 的时间点少于两个时，它们是 NaN。每个模型头都在主标签（第一个标签）的原始值上计算全部八个指标和拟合用的 `loss`，覆盖训练、验证和测试三段。这些指标以 `train_*`、`val_*`、`test_*` 的名字写入 W&B 运行摘要，`train()` 还把同一个字典写到 `config.json` 旁边的 `metrics.json`，NaN 和无穷大写成 null。没有验证段时（`val_size=0`）不会有 `val_*` 键。torch 模型头报告同样的键；其中的 `loss` 是模型头的 `_val_one_batch` 在该段上的均值，默认就是变换后目标上的 `_loss`（见“训练 torch 模型”）。

```python
>>> metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
>>> sorted(metrics)[:9]
['test_ic', 'test_icir', 'test_loss', 'test_mae', 'test_mse', 'test_r2', 'test_rank_ic', 'test_rank_icir', 'test_rmse']
>>> {k: round(v, 3) for k, v in metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.707, 'val_rank_ic': 0.687, 'test_rank_ic': 0.679}
>>> {k: round(v, 3) for k, v in metrics.items() if k.endswith("icir")}
{'train_icir': 6.897, 'train_rank_icir': 6.202, 'val_icir': 6.144, 'val_rank_icir': 5.323, 'test_icir': 5.524, 'test_rank_icir': 4.813}
```

`regression_panel_metrics` 可以对任意面板计算同样的评分：

```python
>>> from quantlab.utils.metrics import regression_panel_metrics
>>> test = slice("2024-06-01", "2024-07-18")
>>> scores = regression_panel_metrics(
...     predictions["ret"].sel(timestamp=test).values,
...     label.ds["ret"].sel(timestamp=test).values,
... )
>>> {name: round(value, 3) for name, value in scores.items()}
{'mse': 0.003, 'rmse': 0.051, 'mae': 0.04, 'r2': 0.491, 'ic': 0.698, 'rank_ic': 0.679, 'icir': 5.524, 'rank_icir': 4.813}
```

IC 和 RankIC 背后的逐时间点数值由 `cross_sectional_ic_series` 和 `cross_sectional_rank_ic_series` 给出（被跳过的时间点为 NaN），`information_ratio` 把这样的序列变成 ICIR。`regression_panel_metrics(pred, target, return_series=True)` 会把两条序列和指标一起返回。

### IC 序列与保存的预测

每次运行还会在 `metrics.json` 旁边写两个文件，之后要算新指标或做集成时可以直接从磁盘读取，不必重新预测：

- `ic_series.csv` 有 `split`、`timestamp`、`ic` 和 `rank_ic` 四列：每个参与评估的段（`train`、`val`、`test`，各段内按时间排序）的每个 bar 一行，记录该 bar 在主标签原始值上的 IC 和 RankIC。它们和 `metrics.json` 来自同一份预测：某段 `ic` 列的均值就是 `<split>_ic`，其 ICIR 就是 `<split>_icir`。没有 IC 的 bar（有效标的少于两个，或截面为常数）不写行。只有当两个值一个存在、另一个不存在时，才会出现空单元格。
- `test_predictions.zarr` 是测试段的预测面板：在收集到的特征上调用 `predict_panel`，覆盖测试段的所有 bar 和收集到的所有标的，每个标签一个变量。测试段没有 bar 时不写这个存储。

```python
>>> import pandas as pd
>>> series = pd.read_csv(checkpoint.parent / "ic_series.csv", parse_dates=["timestamp"])
>>> series.head(3)
   split  timestamp        ic   rank_ic
0  train 2024-01-01  0.662219  0.690226
1  train 2024-01-02  0.798276  0.780451
2  train 2024-01-03  0.811311  0.826627
>>> series.groupby("split", sort=False).size().to_dict()
{'train': 119, 'val': 29, 'test': 48}
>>> test_ic = series[series["split"] == "test"]["ic"]
>>> round(test_ic.mean() / test_ic.std(), 3), round(metrics["test_icir"], 3)
(5.524, 5.524)
>>> saved = xr.open_zarr(checkpoint.parent / "test_predictions.zarr").load()
>>> dict(saved.sizes), list(saved.data_vars)
({'timestamp': 48, 'symbol': 20}, ['ret'])
>>> bool((saved["ret"] == predictions["ret"].sel(timestamp=saved.timestamp)).all())
True
```

### 类层次

所有模型头都通过两个变体之一继承自 `BaseModel`。两个变体的差别在于训练框架，以及子类必须实现哪些方法；`train`、`train_cv`、`load`、`predict` 和 `predict_panel` 只在 `BaseModel` 中实现一次，子类不覆盖。

| 类 | 框架 | 检查点 | 模型头需要实现的方法 |
|---|---|---|---|
| `TorchModel` | torch，每一步一个标的截面 | `.pth` | `window_bars`、`_init_model`、`_loss`；其余是带默认实现的可选钩子（见“训练 torch 模型”） |
| `LibraryModel` | numpy，使用库自带的提前停止 | `.joblib` | `_init_model`、`_preprocess`、`_fit_model`、`_forward` |

自带的模型头有 `XGBoostRegressor`、`XGBTDRegressor` 和 `RealMLPRegressor`，都是 `LibraryModel`；目前还没有自带的 torch 模型头。torch 模型头放在 `quantlab/torch_model/`，库模型头放在 `quantlab/library_model/`。完整的配置字段见 `quantlab/base/model.py` 和 `quantlab/base/config.py` 的 docstring。

### 配置与保留超参数

所有模型头都使用同一个 `ModelConfig`。它只包含两个变体都会读取的字段：因子和标签、保存目录、数据策略、日期、`val_size`、`random_seed` 和 `hyperparameters`。所有训练设置都放进 `hyperparameters` 这一个扁平字典，它会被记录到 `config.json`，因此仅凭 `config.json` 就能重建模型。

基类和自带的模型头会自己从中读取下面这些键（`quantlab.base.model.RESERVED_HYPERPARAMETERS`，即 `TORCH_RESERVED_HYPERPARAMETERS` 与 `LIBRARY_RESERVED_HYPERPARAMETERS` 的并集）：

| 键 | 读取方 | 默认值 |
|---|---|---|
| `epochs` | `TorchModel`：训练 epoch 数的上限；不是正整数时，训练开始时抛出 `ValueError` | 100 |
| `lr` | `TorchModel`：默认 `_init_optim` 的学习率 | `1e-3` |
| `early_stopping` | 自带的库模型头：开启库自带的提前停止 | `False` |
| `early_stopping_patience` | 自带的库模型头：容忍多少轮（或库自己的单位）没有改善 | 5 |
| `batch_size`、`num_workers`、`panel_device`、`panel_dtype` | 预留给 torch 的数据加载器和训练面板 | |

其余的键都属于模型头自己。`_init_model(num_features, num_labels, hyperparameters)` 拿到的是整个字典，保留键也在其中。不要把它整个展开传给网络或库的构造函数（`nn.GRU(**hyperparameters)`、`Regressor(**hyperparameters)`）：按名字读取模型头需要的键，或者先用模型头的 `head_hyperparameters` 方法去掉它所属变体保留的键。自带的库模型头用的是后一种做法：去掉提前停止的两个键，保留 `lr`，因为 pytabkit 把它当作自己的学习率。

## 常见任务

### 提前停止

在 `hyperparameters` 里设置 `"early_stopping": True` 后，当验证损失连续 `early_stopping_patience` 个 boosting 轮没有改善时停止训练，并保留最优模型。这两个是库模型头自己读取的保留键，不会传给库；torch 模型头改用自己的 `_should_stop` 钩子决定何时停止（见“训练 torch 模型”）。对 `XGBoostRegressor`，检查点会被截断到最优的那一轮。判据是验证段上的 RMSE。模型本身以 pooled 一致性相关系数（concordance correlation）损失 `1 - ccc` 为训练目标（见 `quantlab/library_model/xgb.py` 中的 `ccc_objective`）；在 `hyperparameters` 里指定 `objective` 则改回 xgboost 的内置目标。

```python
>>> from dataclasses import replace
>>> stopping = replace(config, hyperparameters={
...     "num_boost_round": 500, "max_depth": 3,
...     "early_stopping": True, "early_stopping_patience": 5,
... })
>>> stopped = XGBoostRegressor(stopping).collect()
>>> _ = stopped.train()
>>> stopped.model.num_boosted_rounds(), stopped.model.best_iteration
(52, 51)
```

### walk-forward 交叉验证

`train_cv(train_periods, parallel=False, njobs=-1)` 在 `start_date` 到 `end_date` 之间的时间戳上滑动训练窗口。每一折在 `train_periods` 个时间戳上训练，在紧随其后的 `train_periods // 5` 个时间戳上测试；下一折晚一个测试段的长度开始。每一折都像 `train()` 一样在自己的日期上拟合，因此训练窗口在测试段之前丢掉最后 L 个 bar，内部再切分成训练段和验证段并做清除。每一折都有自己的检查点和自己的 W&B 运行，检查点目录里还有该折的 `ic_series.csv` 和 `test_predictions.zarr`（该折的指标本身写在下文的 `cv_folds.json` 里）。返回值是每折一个字典，包含该折的日期（两端都包含）、检查点路径以及 `train_*`、`val_*` 和 `test_*` 指标。其中 `train_end` 是清除之后实际拟合的最后一个 bar。

```python
>>> results = model.train_cv(train_periods=100)
>>> len(results)
5
>>> [(r["train_start"][:10], r["train_end"][:10], r["test_start"][:10], r["test_end"][:10]) for r in results]
[('2024-01-01', '2024-04-07', '2024-04-10', '2024-04-29'), ('2024-01-21', '2024-04-27', '2024-04-30', '2024-05-19'), ('2024-02-10', '2024-05-17', '2024-05-20', '2024-06-08'), ('2024-03-01', '2024-06-06', '2024-06-09', '2024-06-28'), ('2024-03-21', '2024-06-26', '2024-06-29', '2024-07-18')]
>>> [round(r["test_rank_ic"], 3) for r in results]
[0.691, 0.649, 0.695, 0.656, 0.697]
>>> [round(r["test_icir"], 3) for r in results]
[6.943, 5.013, 7.982, 5.508, 5.464]
>>> from pathlib import Path
>>> sorted(p.name for p in Path(results[0]["checkpoint"]).parent.iterdir())
['XGBoostRegressor_cv_fold_0.joblib', 'config.json', 'ic_series.csv', 'test_predictions.zarr']
```

所有折共用一个试验目录。除了每折一个子目录，目录里还有 `cv_folds.json`，即清单文件，包含 `format_version`（2）、与返回值相同的折列表（含清除后的 `train_end`），以及 `cv_mean` 块：每个 `train_*`、`val_*`、`test_*` 指标在各折上的均值，记为 `cv_mean_<指标>`，另有 `cv_n_folds`。非有限的折值不参与平均，NaN 和无穷大写成 null。回测器根据这个文件回放一次交叉验证；它拒绝读取第 1 版清单（出现 `cv_mean` 块之前写的），旧项目请重新运行 `train_cv`。每折的 `config.json` 记录的是该折配置时的日期，即清除之前的日期，所以它的 `train_end` 比清单里的晚 L 个 bar。

```python
>>> trial = Path(results[0]["checkpoint"]).parent.parent
>>> sorted(p.name for p in trial.iterdir())
['XGBoostRegressor_cv_fold_0', 'XGBoostRegressor_cv_fold_1', 'XGBoostRegressor_cv_fold_2', 'XGBoostRegressor_cv_fold_3', 'XGBoostRegressor_cv_fold_4', 'cv_folds.json']
>>> manifest = json.loads((trial / "cv_folds.json").read_text())
>>> manifest["format_version"], len(manifest["folds"])
(2, 5)
>>> {k: round(v, 3) for k, v in manifest["cv_mean"].items() if k.endswith("rank_ic")}
{'cv_mean_train_rank_ic': 0.709, 'cv_mean_val_rank_ic': 0.687, 'cv_mean_test_rank_ic': 0.677}
>>> fold_0 = json.loads((Path(results[0]["checkpoint"]).parent / "config.json").read_text())
>>> fold_0["train_end"], manifest["folds"][0]["train_end"]
('2024-04-09T00:00:00', '2024-04-07T00:00:00')
```

`parallel=True` 用线程并发训练各折（`njobs` 指定线程池大小）。每一折操作的是模型的深拷贝，因此内存占用随任务数增长。树模型库本身已经占满所有核心，建议把 `hyperparameters` 里的 `nthread` 设为大约 `os.cpu_count() // njobs`。

### 训练 torch 模型

torch 模型头（`TorchModel`）每一步训练一个 bar：这个 bar 的*截面*，即在该 bar 上至少有一个有限特征值的标的，每个标的带着自己最近 `window_bars` 个 bar 的特征。网络看到的是 `[S_t, N, F]`，其中标的数 S_t 逐 bar 变化，所以网络不能依赖标的的顺序或数量。训练之后才加入的标的同样会得到预测，标签缺失的标的仍作为上下文留在输入里。

模型头需要写三样东西：`window_bars`（N）、`_init_model(num_features, num_labels, hyperparameters)`（网络）和 `_loss(output, batch)`（一个 bar 的损失）。`output` 是网络的原始输出，`batch` 是一个 `CrossSectionBatch`：`x`（`[S_t, N, F]`）、`y`（目标，`[S_t, L]`）、`mask`（`y` 有效的位置为 True）、`y_raw`（原始标签）、`symbols` 和 `timestamp`。缺失的标签已经被掩码并在 `y` 里置 0，所以损失只需统计 `mask` 为 True 的位置，`quantlab.torch_model.training` 里的 `masked_mse` 就是这样做的。其他所有选择都是带默认实现的可选钩子：

| 钩子 | 默认 |
|---|---|
| `_transform_feature(x)`：原始窗口（缺失处为 NaN）到网络输入 | 截断到 ±3，NaN 变 0 |
| `_transform_target(y, training)`：一个 bar 的原始标签到 `(target, keep)`；`keep` 把标的从这一步里去掉 | `(y, None)`；工具函数 `cs_rank_norm`（Qlib `CSRankNorm`）、`cs_zscore`、`drop_extreme` |
| `_init_optim(model)` | Adam，学习率 `hyperparameters["lr"]`（`1e-3`） |
| `_train_one_batch(epoch, batch)`：一步优化，返回损失 | 前向、`_loss`、反向传播、按 `grad_clip_value`（3.0）截断梯度值、step |
| `_val_one_batch(epoch, batch)`：一个 bar 的验证损失 | `_loss` |
| `_test_one_batch(epoch, batch)`：每个 epoch 之后对每个测试 bar 调用 | 什么都不做 |
| `_forward(x)`：`[S_t, L]` 的预测，用于指标和 `predict_panel` | `self.model(x)` |
| `_on_fit_start()`、`_should_stop(epoch, train_loss, val_loss)`、`_on_fit_end()` | 跑满 `hyperparameters["epochs"]`（100）个 epoch，保留最后的权重 |

`train_loss` 和 `val_loss` 是各个逐步钩子返回值的均值；没有验证段时 `val_loss` 为 None。训练的上限是超参数 `epochs`，默认优化器读取 `lr`（见“配置与保留超参数”）。没有 batch size：各个 bar 按打乱后的顺序逐一训练，一步一个 bar。指标始终用原始的第一个标签计算。

`window_bars` 为 N 的模型在预测的第一个 bar 之前需要 N - 1 个 bar 的历史。`collect()` 以及回测的特征请求会向每个因子多要这么多个 bar，按因子自己的数据集日历计数，数据不够早时给出警告。这里的替身面板没有数据集，所以这个模型头用一个 bar 的窗口；它按 bar 对目标做 z-score，验证损失连续五个 epoch 没有下降就停止。

```python
>>> import torch.nn as nn
>>> from quantlab.base.config import ModelConfig
>>> from quantlab.base.model import TorchModel
>>> from quantlab.torch_model.training import cs_zscore, masked_mse
>>> class LastBar(nn.Module):
...     """对每个标的最新一个 bar 做线性映射。"""
...     def __init__(self, num_features, num_labels):
...         super().__init__()
...         self.linear = nn.Linear(num_features, num_labels)
...     def forward(self, x):              # x: [S_t, N, F]
...         return self.linear(x[:, -1])   # [S_t, L]
>>> class LinearHead(TorchModel):
...     window_bars = 1
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         return LastBar(num_features, num_labels)
...     def _loss(self, output, batch):
...         return masked_mse(output, batch.y, batch.mask)
...     def _transform_target(self, y, training):
...         return cs_zscore(y), None
...     def _on_fit_start(self):
...         self.best, self.bad = float("inf"), 0
...     def _should_stop(self, epoch, train_loss, val_loss):
...         if val_loss < self.best:
...             self.best, self.bad = val_loss, 0
...         else:
...             self.bad += 1
...         return self.bad >= 5
>>> torch_config = ModelConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     hyperparameters={"epochs": 50, "lr": 1e-2},
... )
>>> linear = LinearHead(torch_config).collect()
>>> linear_checkpoint = linear.train()
>>> linear_checkpoint.name
'LinearHead_total.pth'
>>> torch_metrics = json.loads((linear_checkpoint.parent / "metrics.json").read_text())
>>> {k: round(v, 3) for k, v in torch_metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.698, 'val_rank_ic': 0.69, 'test_rank_ic': 0.69}
>>> one_more = factor.ds.isel(symbol=[0]).assign_coords(symbol=["S99"])
>>> wider = xr.concat([factor.ds, one_more], dim="symbol")
>>> linear.predict_panel(wider).symbol.size
21
```

### 记录到 Weights & Biases

每次 `train()` 以及 `train_cv()` 的每一折都会打开一个 W&B 运行：运行名是实验名，所在项目名是试验目录名，并附带完整配置。`XGBoostRegressor` 记录每一轮的训练和验证曲线，把最终指标和各因子的重要性写入运行摘要。`XGBTDRegressor` 记录每一轮的验证曲线（`val-rmse`，多标签时为 `val-rmse/<label>`）、选中的轮数和实际训练的轮数，以及同样的特征重要性图表，靠一个注入 pytabkit 内部 `xgboost.train` 调用的回调实现。`RealMLPRegressor` 记录每个 epoch 的平均训练损失（`train-loss`）和验证误差（`val-rmse`），以 epoch 为 `step`，摘要里另有 `best_val_rmse`、`epochs_trained` 和停止 epoch，靠一个注入 pytabkit trainer 的 Lightning 回调实现（`quantlab.library_model.tabkit.active_callbacks`）。`train_cv` 还会额外打开一个 `<类名>_cv_summary` 运行，其摘要就是清单里的 `cv_mean` 块。torch 模型头每个 epoch 记录 `train_loss` 和 `val_loss`，并把最终指标写入运行摘要。`WANDB_MODE=disabled` 会关闭全部记录；`WANDB_MODE=offline` 把运行写到本地的 `wandb/` 目录，之后可以用 `wandb sync` 同步。两者都不设置时，`wandb.init` 需要已登录的账号。

## 扩展

新的模型头继承 `LibraryModel` 或 `TorchModel`，实现上表列出的方法即可，其余都不用改。之后它就能使用 `train`、`train_cv`、`load`、`predict_panel` 和各个回测器。

`LibraryModel` 的模型头拿到的是数组形式的 `[T, S, F]` 特征和 `[T, S, L]` 标签。`_fit_model` 必须把拟合好的对象放到 `self.model` 里，检查点保存的就是这个对象（通过 joblib）。`_preprocess` 会作用在每一个数组上，包括标签，必须返回拷贝。真正的模型在拟合过程中才创建时，`_init_model` 可以返回 `None`。

```python
>>> from quantlab.base.model import LibraryModel
>>> class RidgeHead(LibraryModel):
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
>>> ridge_results = ridge.train_cv(train_periods=100)
>>> [round(r["test_rank_ic"], 3) for r in ridge_results]
[0.685, 0.667, 0.707, 0.672, 0.716]
>>> ridge.model.round(3).ravel().tolist()
[0.05, -0.02, -0.001]
```

`TorchModel` 的模型头就是窗口、网络和损失，再加上它覆写的可选钩子；“训练 torch 模型”里的 `LinearHead` 就是一个完整的例子。窗口、warm-up、目标掩码、bar 的顺序、epoch 循环、指标和检查点由基类负责。

## 注意事项

下面的报错都是原样引用，路径缩写为 `...`。

模型头在构造的第一步就会拒绝 `ModelConfig` 以外的任何配置。

```text
TypeError: XGBoostRegressor requires a ModelConfig, got dict
```

torch 模型头的超参数 `epochs` 不是正整数时，训练一开始就会失败。

```text
ValueError: LinearHead: hyperparameters['epochs'] must be a positive integer, got 0
```

它还会拒绝因子列表里的标签和标签列表里的因子。要预测一个因子，用 `Forward` 包装它。

```text
TypeError: XGBoostRegressor: factors[0] is the label LabelPanel, which reads bars after t; pass it in labels, not factors.
TypeError: XGBoostRegressor: labels[0] is Panel, which is not a label; wrap it in quantlab.label.forward.Forward to predict it.
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

`predict_panel` 需要全部因子变量；torch 网络必须为截面里的每个标的返回一行、为每个标签返回一列。

```text
ValueError: XGBoostRegressor.predict_panel: features are missing factor variable(s) ['f_b']
ValueError: LinearHead: the network must map [S_t, N, F] to a [S_t, L] = [20, 1] tensor, got <class 'tuple'>
```

`train` 需要四个窗口日期齐全。在配置里设置它们，或者用 `train_cv`，它会为每一折设置日期。

```text
ValueError: Training and testing start and end dates must be specified.
```

清除之后必须还剩训练 bar。清除前训练段不超过 L 个 bar 时（这里 L = 2，训练段 2 个 bar），`train` 抛出异常；`train_periods` 不大于 L 时（这里用前瞻为 10 的标签），`train_cv` 抛出异常。

```text
ValueError: Empty training segment: purging the last 2 bars leaves 0 of 2 training timestamps for fitting.
ValueError: Fold 0: purging the last 10 bars leaves no training bar; raise train_periods.
```

每一折在 `train_periods // 5` 个 bar 上测试，因此 `train_periods` 小于 5 时，`train_cv` 在训练任何一折之前就会拒绝。

```text
ValueError: XGBoostRegressor: train_cv(train_periods=4) needs at least 5 training bars, since each fold tests on train_periods // 5 bars.
```

`train_cv` 会用最后一折的日期覆盖配置里的四个 `train_*` 和 `test_*` 日期，之后再调用 `train()` 时请新建配置。如果 `train_periods` 太长、放不下测试段，它会记录一条 `Skipping fold 0: test set exceeds data range` 的日志，并返回空列表（`[]`），不会抛出异常。torch 模型头的 `train_cv` 不返回指标，因此每折的字典里只有日期和路径，`cv_mean` 为空，也不会打开汇总运行。

`train()` 只返回检查点路径，这次运行的指标在旁边的 `metrics.json` 里。对 `LibraryModel` 的模型头，`train_cv` 会直接返回这些指标。

检查点是 pickle 文件（`LibraryModel` 用 `joblib`，`TorchModel` 用 `torch.load`）。只加载自己生成或可信的文件。

进度信息通过 `loguru` 和 `tqdm` 输出到 stderr。`logger.remove()` 可以关掉日志行。

在 macOS 上，`xgboost` 的 wheel 链接的是 Homebrew 的 OpenMP 运行时，而 `torch` 自带另一份。同一个进程里同时使用两者可能崩溃或卡死。在第一次导入其中任何一个库之前设置 `OMP_NUM_THREADS=1` 可以避免，代价是树模型和 torch 代码变成单线程。Linux 不受影响。

## 另请参阅

factor 指南（`docs/factor.md`）介绍因子和标签如何生成，backtest 指南（`docs/backtest.md`）介绍 `predict_panel` 的输出和 `cv_folds.json` 清单如何进入回测。backend 指南（`docs/backend.md`）介绍面板使用的 Zarr 与 xarray 存储。API 细节见 `quantlab/base/model.py`、`quantlab/base/config.py`（`ModelConfig`、`ModelConfig`）、`quantlab/torch_model/`、`quantlab/library_model/xgb.py`、`quantlab/library_model/backend.py` 和 `quantlab/utils/metrics.py` 的 docstring。
