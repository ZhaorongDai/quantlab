# 模型层（Model）

> 代码位置：`quantlab/base/model.py`（三层模型类：框架无关的 `BaseModel`、torch 变体 `DLModel`、
> numpy / 树模型变体 `MLModel`）、`quantlab/base/config.py`（`DLConfig` / `MLConfig`）、
> `quantlab/dl_model/rnn_classification.py`、`quantlab/dl_model/rnn.py`、`quantlab/dl_model/mlp.py`（三个 `DLModel` 头）、
> `quantlab/ml_model/xgb.py`（`XGBoostRegressor`，`MLModel` 头）、`quantlab/ml_model/backend.py`（`MLModel` 的 checkpoint 持久化后端）、
> `quantlab/utils/metrics.py`（截面 IC / RankIC 等面板指标）、`quantlab/utils/module.py`（按点分路径重建类）。
> 一个真实的端到端调用脚本：`train_model.py`。

---

## 2026-09-14 重构：名字对照

模型层在 260914-lno 里从「一个 torch 形状的 `BaseModel`」拆成了三层（理由见下一节）。
后文「常见坑」「已知的不完整之处」里的**历史叙述**保留了当时的旧名字，对照这张表读：

| 旧名字 | 现在 |
|---|---|
| `BaseModel._train_dl` | `DLModel._fit`（函数体原样搬过去，早停语义逐字不变） |
| `BaseModel._predict_nn` | `DLModel._predict`（「模型未初始化」检查上提到公开的 `BaseModel.predict`） |
| `BaseModel._auto_train` | **已删除**，不留别名：`train` / `train_cv` 直接调变体的 `_fit` |
| `to_tensor`（原先定义在基类上） | 迁到 `DLModel.to_tensor`，它包装 `BaseModel.to_array`（numpy，DL 与 ML 共用） |
| `_train_fold_with_config(fold_config)` | `_train_fold_with_config(fold, project_name)`，消费 `_cv_folds` 产出的折 dict |
| 模型头继承 `BaseModel` | DL 头继承 `DLModel`，ML 头继承 `MLModel`；`from quantlab.base.model import BaseModel` 仍可用 |

---

## 一句话

把「拿因子面板和标签面板去训练一个模型」这件事里**所有跟训练框架无关的部分**——取数、
对齐、按日期切分、交叉验证的折边界、存检查点与 `config.json`、记实验——一次性写在
`BaseModel` 里；训练本身按框架分两条路：

- **`DLModel`（torch）**：基类替你跑 epoch 循环、按 epoch 早停并回滚到最优 epoch。写一个新模型
  只需要回答五个问题：网络长什么样、一个 batch 怎么训、怎么验、怎么测、张量进模型前怎么洗。
- **`MLModel`（numpy，树模型）**：没有 epoch 循环。写一个新模型回答四个问题：模型/超参怎么
  准备、数组怎么洗、怎么训（早停交给库自己）、怎么前向。

公开的 `train` / `train_cv` / `load` / `predict` 只在 `BaseModel` 上实现一份，任何头都不覆盖
（`tests/test_model_hierarchy.py` 锁）。之所以按框架拆：把非 torch 的分支硬塞进 torch 形状的
基类，每个方法都要长出「模型是不是 `nn.Module`」的分支；拆开后每层只暴露自己要的钩子，
而折边界这类跟框架无关的算术仍然只有一份。

---

## 它吃什么、吐什么

### 吃：两份 xarray 面板

模型层不直接读磁盘、不直接碰行情。它拿到的是配置里塞进来的**因子对象**和**标签对象**，
然后调用它们的公共契约取数（`quantlab/base/model.py:_collect_all_features` / `_collect_all_labels`）：

```python
# factor_data_strategy / label_data_strategy 决定走哪条
ds = factor.cal().get_features()    # "cal"：现算
ds = factor.read().get_features()   # "read"：读已经算好的 zarr
```

拿到的每一份都是坐标为 `(timestamp, symbol)` 的 `xr.Dataset`，多个因子/标签用
`xr.combine_by_coords` 拼成一整块，最后 `collect()` 把特征和标签再拼一次、
按 `["timestamp", "symbol"]` 排序，存进 `self.data_backend`（一个 `XrBackend`）。

排序这一步不是洁癖。两份面板各自的坐标顺序**不保证一致**，不排序就会出现
「第 3 行的特征配上了第 7 行的标签」——这种错位不会报任何错，只会让模型安静地学噪音。

### 吐：预测

`predict()` 返回模型的原始输出。它是什么含义完全由具体模型定义：
`RNNRegressor` 吐未来收益的回归值，`RNNClassifier` 吐涨跌两分类的 logits
（注意 `RNNClassifier` 的 `forward` 返回的是 `(primary_pred, all_direct_preds)` 元组，
所以调用方要写 `predicts, _ = model.predict(data)`，见 `train_model.py:78`）。

按 CLAUDE.md 的架构契约，模型层的产物是「未来收益 / 收益排名预测」，
下游由组合优化模块把它变成目标持仓。**这条下游目前还没有接上**——
`train_model.py` 里是手写的一段 vectorbt 信号回测，不是一个组件。

### 张量形状：`[num_times, num_symbols, num_features]`

从 xarray 变成数组/张量的那段是整层最该看懂的地方。它现在被封装成了
`BaseModel.to_array(data, variables)`（给出 numpy 数组，DL 与 ML 共用），
`DLModel.to_tensor(data, variables)` 只是在它外面 `torch.from_numpy` 并统一浮点 dtype。
训练和推理走的是同一份实现（下面是 `to_tensor` 的等价展开）：

```python
torch.from_numpy(
    data[variables]
    .to_dataarray()
    .sortby(["timestamp", "symbol"])   # 只排这两个
    .sel(variable=variables)           # 最后一维按调用方声明的顺序钉死
    .transpose("timestamp", "symbol", "variable")
    .values
)
```

`to_dataarray()` 把 Dataset 的每个变量（每个因子）堆成新的一维 `variable`，
于是二维面板 `(timestamp, symbol)` 变成三维 `(timestamp, symbol, variable)`。

`sortby` 里**没有** `variable`，这是 2026-09-07 修掉的一个静默错位：
把 `variable` 一起排会让最后一维变成字母序而不是配置里的顺序，
详见「常见坑」第 3 条。

**为什么是这个顺序而不是别的？** 因为 `DataLoader` 只会在**第 0 维**上切 batch。
把 `timestamp` 放第 0 维，一个 batch 就是「若干个完整的时间截面」——
截面内的所有标的原封不动地待在一起。这对量化是刚需：截面上做排序、做中性化、
做 GRU 的序列建模（`ModelRBaseCrypto.forward` 把 `(D, T, F)` 喂给 `nn.GRU(batch_first=True)`，
把 symbol 当成序列维），全都要求同一时刻的标的不能被拆散。
如果按 `(symbol, timestamp)` 排，切 batch 就会把一个截面切碎。

顺带一提：`DLModel._fit` 的 `DataLoader` 用了 `shuffle=True`。打乱的是**截面之间**的顺序，
截面内部完好，所以对 MLP 这类逐截面模型没问题；但对把 symbol 当序列的 GRU 也没问题，
因为它的「序列」是 symbol 而不是时间。要做真正的时间序列窗口模型，
需要在 `_preprocess` 或自定义 Dataset 里自己造滑窗——基类不提供。

### 为什么不经 DataFrame

CLAUDE.md 把「模块间统一使用 xarray，不用 DataFrame 作为层间传输格式」列为硬约束，
模型层是这条约束最吃力也最受益的地方：

1. **`(timestamp, symbol)` 是天然的二维，`(时间, 标的, 因子)` 是天然的三维。**
   DataFrame 只有二维，装三维要靠 MultiIndex，而 MultiIndex → numpy 的 reshape
   顺序对不对，只能靠人脑保证；xarray 的 `transpose("timestamp","symbol","variable")`
   是**按名字**指定的，写错了会报错而不是静默错位。
2. **稀疏与对齐是白送的。** 不同因子覆盖的标的、时间不完全一样，
   `combine_by_coords` 按坐标对齐并自动填 NaN；换成 DataFrame 要写一堆 merge。
3. **少一次全量拷贝。** `.to_dataarray().values` 直接给出连续内存，
   `torch.from_numpy` 零拷贝接管。走 DataFrame 要多一轮 pivot + to_numpy。

代价是 NaN 要自己处理——这正是 `_preprocess` 存在的原因。

---

## 基类替你做了什么

按调用顺序过一遍。「来自」一列说明这一步实现在哪一层：

| 你调什么 | 来自 | 做了什么 |
|---|---|---|
| `__init__(config)` | `BaseModel` | 接下配置；`_set_random_seed` 钉死随机源（`BaseModel` 播种 python / numpy，`DLModel` 再补 torch CPU / CUDA 与 cudnn 确定性）。**注意此时既不建模型也不读数据**——建模型要先知道数据形状。 |
| `config = ...`（setter） | `BaseModel` | **第一条语句**检查配置类型：不是本变体的 `config_cls`（`DLModel`→`DLConfig`，`MLModel`→`MLConfig`）就 `TypeError`，先于触碰任何因子。之后把训练区间**下推**给每一个因子和标签，并把 `config.name` 写成本类的完整导入路径。 |
| `collect()` | `BaseModel` | 取特征、取标签、`combine_by_coords`、`sortby`、灌进 `XrBackend`。返回 `self`，可以链式写 `Model(cfg).collect().load(ckpt)`。 |
| `train()` | `BaseModel` | 生成带时间戳的实验名，文件名后缀取 `checkpoint_suffix`（`.pth` / `.joblib`）→ `_init_wandb` → 变体的 `_fit`。 |
| `_fit()`（DL） | `DLModel` | 切 train/test（按配置日期）→ `to_tensor` 转张量 → 形状校验 → 从训练段**尾部**按 `val_size` 切验证集 → 建 `DataLoader` → epoch 循环 → 按 epoch 早停（并快照最优 epoch 的权重）→ 回滚到最优权重 → `_save_model` → `wandb.finish()` → `self.optim = None`（模型保留）。返回 None。 |
| `_fit()`（ML） | `MLModel` | 切 train/test → `to_array` + `_preprocess` → 形状校验 → 尾部切验证集（为空时传 None）→ **调一次** `_fit_model`（早停在库里）→ `_evaluate` train / val / test 写 W&B summary → `_save_model` → `finish()`。返回 test 指标 dict。 |
| `train_cv(...)` | `BaseModel` | 用唯一的折生成器 `_cv_folds` 沿时间前滚切多折，每折独立训练一个模型；返回逐折结果 list。ML 头额外把各折 `test_*` 指标的均值写进一个独立的 `{cls}_cv_summary` run。 |
| `load(path)` | `BaseModel` → 变体 | 先校验文件存在、后缀等于 `checkpoint_suffix`，再调变体的 `_read_checkpoint`（DL：按当前数据形状重建网络再灌权重；ML：直接读回整个模型）。 |
| `predict(x)` | `BaseModel` → 变体 | 模型未训练/未加载时报错；DL：`model.eval()` → `to(device)` → `_preprocess` → `model(x)`，整段在 `torch.no_grad()` 里，ndarray 输入先转张量；ML：tensor 输入先转 numpy → `_preprocess` → `_forward`。 |

几个设计选择值得单独说明，因为它们都是**只在量化场景才成立**的：

**验证集是从训练段尾部按时间切的，不是随机抽的**：

```python
train_split = int(train_x_t_all.shape[0] * (1 - self.config.val_size))
train_x_t = train_x_t_all[:train_split]
val_x_t   = train_x_t_all[train_split:]
```

随机抽样会让模型在训练时见到未来，验证分数会好看得不真实。

（切点原本写的是 `train_split + 1:`，会静默丢掉一行；2026-09-07 修掉了，
见「常见坑」第 6 条。）

**`train_cv` 是滚动切分，还带 `gap_periods`。** 训练段整段落在测试段之前；
中间可以留一段空隙，用来隔开标签自身的前视窗口——
`Return`（`quantlab/label/fret.py`）的标签是 `shift(timestamp=-n_forward_periods)` 得来的，
不留 gap 的话训练段末尾那 n 根 bar 的标签里已经包含了测试段开头的信息。
测试段长度**固定**为训练段的 1/5（`test_periods = train_periods // 5`，写死的，不可配）。
折几何的全部细节见「ML / 树模型」一节下的「ML 交叉验证」——DL 与 ML 用的是同一个 `_cv_folds`。

**模型的输入输出维度取自数据而不是配置**（`_init_model_and_optim`）：
`num_symbols` / `num_factors` / `num_labels` 都是从已收集的 xarray 面板上现算的属性。
所以你加一个因子，网络的输入层自动变宽，不需要在任何地方同步一个数字。

**形状校验挡在训练之前**（`_assert_shape_match_x/y`）：形状不匹配在 torch 里
常常被广播悄悄吸收掉，最后表现为「loss 不下降」。挡在这里，问题停在它产生的地方。
但要注意它**只校验个数，不校验列名顺序**——见下面「常见坑」。

**并行 CV 的每折先 `copy.deepcopy(self)`**（`_train_fold_with_config(fold, project_name)`，
消费 `_cv_folds` 产出的折 dict）：并行跑 CV 时各折会同时改写 `config.train_start` 等字段，
共用一个实例会互相覆盖。顺序分支直接在本实例上逐折 `_train_one_fold`。

---

## DLModel 的五个张量钩子

`DLModel` 有 5 个 `@abstractmethod`（`BaseModel` 自己的抽象成员——`config_cls`、`checkpoint_suffix`、
`_fit`、`_predict`、`_write_checkpoint`、`_read_checkpoint`——已由 `DLModel` 实现），少实现一个类就实例化不了
（这不是理论——`quantlab/dl_model/mlp.py:MLPRegressor` 曾经漏了 `_val_one_batch`，
`MLPRegressor.__abstractmethods__` 实测是 `frozenset({'_val_one_batch'})`，
连构造都做不到。已于 2026-09-07 修复，现在三个具体模型头都实现齐了 5 个方法，
由 `tests/test_dl_models.py::test_mlp_regressor_has_no_unimplemented_abstract_methods` 锁住）。

### `_init_model(num_symbols, num_features, num_labels, hyperparameters) -> nn.Module`

搭网络。四个参数全是**基类算好递给你的**：前三个来自当前数据的实际形状，
第四个是 `config.hyperparameters` 这个自由字典。返回一个还没搬到设备上的 `nn.Module`
（基类会 `.to(self.device)`）。

职责边界：这里**只建结构，不建优化器**。

### `_init_optim(model) -> Optimizer | None`（非抽象，但基本都要写）

基类默认 `raise NotImplementedError`，但 `_init_model_and_optim` 里的调用**没有** try/except，
所以不实现就会直接炸。允许返回 `None`——表示「我在训练钩子里自己更新参数」，
这时基类就不设置 `self.optim`。（以前 `_train_dl` 结尾是无条件 `del self.optim`，
返回 `None` 会在训练结束时 `AttributeError`；2026-09-07 改成了 `self.optim = None`，
这条路不再炸。实务上还是老实返回一个优化器。）

### `_train_one_batch(epoch, x, y) -> Tensor`

**一次调用 = 一个 batch。** 基类在 `for x_batch, y_batch in train_loader:` 里调它，
你要在里面完成 `zero_grad` → forward → loss → `backward` → `step`，
外加自己 log 指标。基类已经替你做了 `model.train()` 和 `x.to(device)`。

> 这三个钩子曾经叫 `_train_one_epoch` / `_val_one_epoch` / `_test_one_epoch`
> （**已于 2026-09-07 改名**）。名字骗人不是文风问题：批次 1 修的那个早停 bug，
> 正是因为作者把计数器写在 `_val_one_epoch` 旁边、照着名字读成「每个 epoch 一次」，
> 于是 `counter += 1` 落在验证 batch 循环里（见「常见坑」第 2 条）。
> 名字留着就等于把同一个坑留给下一个人，所以改了。第一个参数 `epoch` 仍然是
> epoch 序号——基类透传它只是为了让你 log 到正确的 step 上。

`x` 形状 `(batch内的时间点数, num_symbols, num_features)`，`y` 是 `(..., num_labels)`。

### `_val_one_batch(epoch, x, y) -> Tensor`

同样是一次调用一个 batch。基类已经在 `model.eval()` + `torch.no_grad()` 里了，
所以**不要**再自己包 `no_grad`，也不要 backward。

**它的返回值就是早停判据。** 基类把每个 batch 的返回值按样本数加权平均成一个
epoch 级别的验证损失，再拿它去比 `best_loss`（2026-09-07 之前是逐 batch 直接比，
见「常见坑」第 2 条）。**它同时决定存哪一轮的权重**——比 `best_loss` 小的那个
epoch 会被快照下来，训练结束回滚（见「常见坑」第 13 条）。
所以它必须返回一个能 `float()` 的标量 loss——
返回 `None` 会在 `float(None)` 处直接 `TypeError`。

> 这条不是假想。`quantlab/dl_model/rnn.py:RNNRegressor._val_one_batch` 以前只记 metrics
> **什么都不返回**，注解写的却是 `-> torch.Tensor`。加权平均那行是无条件执行的
> （跟 `early_stopping` 开不开无关），所以 `RNNRegressor.train()` 在第 0 个 epoch
> 就是 `TypeError: float() argument must be a string or a real number, not
> 'NoneType'`。2026-09-07 已按 `rnn_classification.py` 里同名方法的写法补上
> `return val_loss.detach()`，由
> `tests/test_dl_models.py::test_rnn_regressor_val_one_batch_returns_a_floatable_loss` 锁住。

### `_test_one_batch(epoch, x, y) -> Tensor`

每个 epoch 在测试集上跑一遍，只记指标不更新参数。返回值目前基类不使用。
（是的，这意味着测试集指标在训练过程中一直可见——这在方法论上是有争议的，
但代码就是这么写的。）

### `_preprocess(data) -> Tensor`

训练与推理**共用**这一个钩子（`DLModel._fit` 和 `DLModel._predict` 都调它）。
共用是关键：如果推理时的预处理和训练时不一致，模型看到的输入分布就变了，
而这种偏差不会报错，只会让预测悄悄失准。

现有三个模型的实现都是一句 `torch.nan_to_num(data, nan=0.0)`。
NaN 从哪来？因子的滚动窗口预热期、标签的 `shift` 尾部、稀疏覆盖的标的。
`num_null` 这个属性就是给你训练前先看一眼用的：它返回整块面板上 NaN 单元格的
总数（跨全部变量、全部 timestamp、全部 symbol）。

> 它曾经**每次读取都抛异常**（**已于 2026-09-07 修复**）：实现结尾是 `.values[0]`，
> 但前一个 `.sum()` 已经把 `variable` 维加掉了，拿到的是 0 维数组，于是
> `IndexError: too many indices for array: array is 0-dimensional, but 1 were
> indexed`。一条被文档推荐、注解写着 `-> int`、却从来没跑通过的路。现在取 0 维
> 数组本身再显式 `int()`，由 `tests/test_model_layer.py::
> test_num_null_counts_missing_cells_and_returns_an_int` 锁住（配套的
> `test_num_null_is_zero_on_a_dense_panel` 保证它不是恒返回某个常数）。

---

## MLModel 的四个钩子与可覆盖默认实现

`MLModel` 是给 xgboost / LightGBM / CatBoost 这类非 torch 模型的变体。四个抽象钩子
（`tests/test_model_hierarchy.py` 钉成精确集合）：

### `_init_model(num_features, num_labels, hyperparameters)`

准备模型或只解析超参，返回值原样赋给 `self.model`（允许 None——树模型通常要到训练时才由库建出来）。
**`load()` 不调它**：`.joblib` 里就是完整模型，未 `collect()` 的新实例也能直接 `load` 再 `predict`。

### `_preprocess(data: np.ndarray) -> np.ndarray`

训练时对四份数组（train/test 的 x 与 y）各调一次，推理时对输入调一次。返回**新数组**，不许原地改入参。

### `_fit_model(train_x, train_y, val_x, val_y)`

入参是 `[T, S, F]` / `[T, S, L]`。训练段尾部按 `val_size` 切出的验证段为空（`val_size=0`）时，
`val_x` 与 `val_y` **都是 None**——不是长度为 0 的数组（xgboost 对空输入会给出 `(0, 0)` 这种形状）。
早停与「回到最优轮」由它用库的原生机制完成，遵循 `config.early_stopping` / `config.early_stopping_patience`；
返回时 `self.model` 必须已经是要保存的那个模型。每次 `_fit` 恰好调用它一次。

### `_forward(x: np.ndarray) -> np.ndarray`

对已预处理的 `[T, S, F]` 返回 `[T, S, L]`。

### 可覆盖的默认实现

| 方法 | 默认行为 |
|---|---|
| `_loss(y, pred)` | 所有标签都有限的 `(t, s)` 位置上、对全部标签求 MSE；没有有效位置时返回 NaN（不发 RuntimeWarning） |
| `_compute_metrics(y, pred)` | 主标签（最后一维第 0 个）上的 `regression_panel_metrics`：`mse, rmse, mae, r2, ic, rank_ic` |
| `_evaluate(split, x, y)` | `_forward` → 组装 `{split}_loss` 与 `{split}_{mse,rmse,mae,r2,ic,rank_ic}` → 写进 W&B run 的 **summary**（最终值，不带 step）→ **返回这个带前缀的 dict** |
| `_resolved_hyperparameters()` | 返回 None。头覆盖它返回「实际生效的超参」后，`_fit` 把它写进 W&B run config，`config.json` 多一个顶层 `resolved_hyperparameters`（见「ML / 树模型」） |

IC / RankIC 来自 `quantlab/utils/metrics.py`：逐时间戳的截面 Pearson / Spearman 再对时间求均值，
只统计预测和目标**两边都有限**的格子，RankIC 先联合掩码再排名（平局取平均秩），
有效标的 <2 或截面为常数的时间戳被跳过。两个函数都是向量化实现，没有 Python 行级循环。

### 为什么 ML 不用 DL 那种按 epoch 的循环

`MLModel` 没有 epoch 循环，也不做任何模型拷贝回滚（`tests/test_model_hierarchy.py` 用 AST 锁住类体里没有 `deepcopy`）：

- **粒度**：xgboost / LightGBM / CatBoost 都是逐棵树（逐轮）判定早停；外面再包一层「epoch」只会让判定变粗。
- **成本**：库内的验证分数靠预测缓存增量计算，总成本随轮数**线性**增长；外层每个 epoch 用全部树把验证集重算一遍，总成本是**二次**的。
- **回滚**：树模型回到最优轮只需切片保留前 k 棵树（xgboost 的 `EarlyStopping(save_best=True)`），不需要拷贝整个模型。

DL 仍按 epoch 走，因为神经网络没有「前 k 棵树」这种可以廉价切回去的结构，只能在 epoch 边界上快照权重。

---

## 简单用法：训练一个已有的模型

`train_model.py` 是仓库里唯一一条真实的端到端路径。骨架是这样的：

```python
from quantlab.base.config import DLConfig
from quantlab.config import alpha101_config, alpha158_config, spot_label_config
from quantlab.dl_model.rnn_classification import RNNClassifier
from quantlab.factor.alpha101 import Alpha101SpotKline
from quantlab.factor.alpha158 import Alpha158SpotKline
from quantlab.label.spot import SpotReturn

label1 = SpotReturn(spot_label_config("ret_1m", n_forward_periods=30,  symbols=["BTCUSDT"]))
label2 = SpotReturn(spot_label_config("ret_1m", n_forward_periods=60,  symbols=["BTCUSDT"]))
label3 = SpotReturn(spot_label_config("ret_1m", n_forward_periods=120, symbols=["BTCUSDT"]))
alpha101 = Alpha101SpotKline(alpha101_config(symbols=["BTCUSDT"]))
alpha158 = Alpha158SpotKline(alpha158_config(symbols=["BTCUSDT"]))

mc = DLConfig(
    start_date="2020-01-01", end_date="2025-01-01",
    train_start="2022-01-01", train_end="2022-08-01",
    test_start="2022-08-02",  test_end="2022-10-01",
    factors=[alpha158, alpha101],
    labels=[label1, label2, label3],
    model_save_dir="./model_ckpt",
    factor_data_strategy="read",   # 因子读已算好的 zarr
    label_data_strategy="cal",     # 标签现算
    batch_size=30000, epochs=50, lr=1e-3,
    early_stopping=True, early_stopping_patience=5,
    hyperparameters={
        "hidden_sizes": [1024, 512, 256, 128, 64],
        "dropout_rates": [0.5, 0.3, 0.3, 0.3, 0.3],
        "hidden_sizes_linear": [64, 32, 16],
        "dropout_rates_linear": [0.3, 0.3, 0.3],
        "model_type": "gru",
    },
)

model = RNNClassifier(mc)
model.collect()
model.train()
```

> **此例未实际运行**：它依赖本机不存在的 Binance BTCUSDT 分钟线 zarr 数据，
> 且 `quantlab/config/__init__.py` 里的路径是另一台机器的绝对路径。上面的代码抄自
> `train_model.py:19-58`，只是把 `model.load(...)` 换回了 `model.train()`。

推理这一段值得单独看（`train_model.py:66-74`）：

```python
model.load(ckpt_path)
data = model.data_backend.get_xarray_dataset()
data = data.sel(timestamp=slice("2024-01-01", "2024-03-01"))
factors = model.get_factor_names()
data = model.to_tensor(data[factors].fillna(0), factors)
predicts, _ = model.predict(data)
```

这里以前是**手抄**了一份 `_train_dl` 里的转换逻辑（`to_dataarray → transpose →
sortby(["timestamp","symbol","variable"])`）。抄一份的代价不是重复，是**两份会漂**：
训练侧一旦改了列顺序，推理侧不跟着改就是静默错位。2026-09-07 把它提成了基类上的 `to_tensor`；2026-09-14 起它是
`DLModel.to_tensor(data, variables)`（包装 `BaseModel.to_array`），训练和推理仍共用同一份实现。

---

## 扩展：接入一个新模型

下面这段是**真的跑通过的**：合成数据、CPU、5 个 epoch，不需要 GPU、不需要行情数据、
不需要 W&B 账号。

它同时演示了一件容易被忽略的事：`collect()` 只要求因子/标签对象实现**六个方法**
（`config` 属性、`_reset_dataset_config`、`_get_factor_names`、`cal()`/`read()`、
`get_features()`/`get_labels()`、`get_config()`），所以做单元测试时完全可以拿一个
几十行的假面板顶上，不必拖进 KunQuant 和 zarr。

### 关于 W&B

`train()` 无条件调用 `_init_wandb`，里面是 `wandb.init(...)`——**没有开关可以跳过**。
`DLConfig` 里也没有任何 `use_wandb` 字段。绕过办法是环境变量：

```bash
WANDB_MODE=disabled uv run python example/min_model.py
```

`disabled` 模式下 `wandb.init` 返回一个 no-op 的 Run 对象，`.log()` / `.finish()` /
`.summary.update()` / `.config.update()` 都能正常调用，不联网、不需要登录、不落盘。要保留本地记录但不上传，用 `WANDB_MODE=offline`。
（下面的实测输出就是在 `WANDB_MODE=disabled` 下跑的。）

记了什么，按路径分：

| 路径 | 逐步曲线（`log(step=...)`） | 训练结束写 summary | 其他 |
|---|---|---|---|
| DL 头 | 头自己在 `_*_one_batch` 里记，`step=epoch` | 无统一约定 | — |
| XGB 头 | 每轮 `train-rmse` / `val-rmse`（**连字符**，xgboost 原生写法，`step=iteration`，从 0 连续） | `{split}_loss` 与 `{split}_{mse,rmse,mae,r2,ic,rank_ic}`（**下划线**），早停启用时另有 `best_iteration` / `best_score` | run config 里的 `resolved_hyperparameters` |
| `train_cv`（ML） | 每折一个 run，内容同上 | 另开一个名为 `{cls}_cv_summary` 的 run：`cv_mean_test_*` 与 `cv_n_folds` | DL 的 `_fit` 不返回指标，不开这个 run |

两种键写法是刻意区分的：连字符的是曲线，下划线的是最终值。

> macOS 注意：本节示例只用 torch，不受影响；但同一进程里一旦同时训练 torch 与 xgboost，
> macOS 上必须 `OMP_NUM_THREADS=1`，见「ML / 树模型」下的「macOS / Linux：OpenMP 冲突」。
> 下面这次重跑（2026-09-14）为了与测试环境一致，也带了这个变量。

### 代码

```python
"""最小可跑示例：用合成数据在 CPU 上训练一个自定义模型。

运行方式（仓库根目录）：
    WANDB_MODE=disabled uv run python example/min_model.py
"""

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from quantlab.base.config import DLConfig
from quantlab.base.model import DLModel

# ---------------------------------------------------------------- 合成的因子/标签
N_TIMES, N_SYMBOLS = 200, 4
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype("timedelta64[D]")
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]


class FakePanel:
    """假装自己是一个因子/标签对象，只实现模型层真正会调用的那几个方法。"""

    def __init__(self, names, seed):
        self.names = names
        rng = np.random.default_rng(seed)
        self._ds = xr.Dataset(
            {
                n: (("timestamp", "symbol"),
                    rng.standard_normal((N_TIMES, N_SYMBOLS)).astype("float32"))
                for n in names
            },
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )
        self.config = SimpleNamespace(start_date=None, end_date=None)

    # 模型层 config setter 会调用
    def _reset_dataset_config(self):
        pass

    # 模型层拼列名会调用
    def _get_factor_names(self):
        return list(self.names)

    # factor_data_strategy / label_data_strategy = "cal" 时走这条
    def cal(self):
        return self

    def get_features(self):
        return self._ds

    def get_labels(self):
        return self._ds

    def get_config(self):
        return {"name": "FakePanel", "factor_names": list(self.names)}


# ---------------------------------------------------------------- DLModel 的五个张量钩子
class TinyRegressor(DLModel):
    def __init__(self, config: DLConfig):
        super().__init__(config)
        self.criterion = nn.MSELoss()

    def _init_model(self, num_symbols, num_features, num_labels, hyperparameters):
        hidden = hyperparameters.get("hidden", 16)
        # 输入 (T, S, F)，nn.Linear 作用在最后一维，输出 (T, S, L)
        return nn.Sequential(
            nn.Linear(num_features, hidden), nn.ReLU(), nn.Linear(hidden, num_labels)
        )

    def _init_optim(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        # NaN 归零。dtype 不用管了：`to_tensor` 已经统一成 torch 的默认 dtype
        # （见「常见坑」#4），`.float()` 现在只是个恒等操作。
        return torch.nan_to_num(data, nan=0.0)

    def _train_one_batch(self, epoch, x, y):
        self.optim.zero_grad()
        loss = self.criterion(self.model(x), y)
        loss.backward()
        self.optim.step()
        if self._wandb_recorder:
            self._wandb_recorder.log({"train_loss": loss.item()}, step=epoch)
        print(f"  epoch {epoch} train_loss={loss.item():.4f}")
        return loss

    def _val_one_batch(self, epoch, x, y):
        loss = self.criterion(self.model(x), y)   # 基类已经在 no_grad 里了
        print(f"  epoch {epoch}   val_loss={loss.item():.4f}")
        return loss                                # 早停就看这个返回值

    def _test_one_batch(self, epoch, x, y):
        loss = self.criterion(self.model(x), y)
        print(f"  epoch {epoch}  test_loss={loss.item():.4f}")
        return loss


# ---------------------------------------------------------------- 跑起来
if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "disabled")

    save_dir = tempfile.mkdtemp(prefix="tiny_ckpt_")
    cfg = DLConfig(
        factors=[FakePanel(["f0", "f1", "f2"], seed=1)],
        labels=[FakePanel(["y0"], seed=2)],
        model_save_dir=save_dir,
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date="2024-01-01", end_date="2024-07-18",
        train_start="2024-01-01", train_end="2024-05-01",
        test_start="2024-05-02",  test_end="2024-07-18",
        epochs=5,
        batch_size=64,
        num_workers=0,
        lr=1e-2,
        early_stopping=True,          # 必须为 True，见「常见坑」第 1 条
        early_stopping_patience=5,
        hyperparameters={"hidden": 16},
    )

    model = TinyRegressor(cfg)
    model.collect()
    print("device:", model.device)
    print("num_times / num_symbols / num_factors / num_labels =",
          model.num_times, model.num_symbols, model.num_factors, model.num_labels)
    print("factor names:", model.get_factor_names(),
          "label names:", model.get_label_names())
    model.train()
    print("checkpoint dir:", save_dir)
    for root, _, files in os.walk(save_dir):
        for f in files:
            print("  ", os.path.join(root, f).replace(save_dir, "<save_dir>"))
```

### 真实输出

```
$ WANDB_MODE=disabled OMP_NUM_THREADS=1 uv run python <仓库外临时目录>/min_model.py

device: cpu
num_times / num_symbols / num_factors / num_labels = 200 4 3 1
factor names: ['f0', 'f1', 'f2'] label names: ['y0']


TinyRegressor_train: 100%|██████████| 5/5 [00:00<00:00, 193.73it/s]
  epoch 0 train_loss=1.0673
  epoch 0 train_loss=1.1245
  epoch 0   val_loss=1.0620
  epoch 0  test_loss=1.0013
  epoch 0  test_loss=1.1546
  epoch 1 train_loss=1.0064
  epoch 1 train_loss=1.1186
  epoch 1   val_loss=1.0531
  epoch 1  test_loss=0.9912
  epoch 1  test_loss=1.1352
  epoch 2 train_loss=1.0252
  epoch 2 train_loss=1.0306
  epoch 2   val_loss=1.0516
  epoch 2  test_loss=0.9860
  epoch 2  test_loss=1.1045
  epoch 3 train_loss=0.9785
  epoch 3 train_loss=1.0815
  epoch 3   val_loss=1.0547
  epoch 3  test_loss=0.9836
  epoch 3  test_loss=1.0759
  epoch 4 train_loss=1.0135
  epoch 4 train_loss=0.9877
  epoch 4   val_loss=1.0581
  epoch 4  test_loss=0.9835
  epoch 4  test_loss=1.0476
checkpoint dir: /var/folders/.../T/tiny_ckpt_y35yla00
   <save_dir>/TinyRegressor_trial_20260914_193249/TinyRegressor_total/config.json
   <save_dir>/TinyRegressor_trial_20260914_193249/TinyRegressor_total/TinyRegressor_total.pth
```

（数据是纯随机的，loss 在 1.0 附近不下降是**正确的**——特征和标签之间本来就没有关系。
另有一条 `UserWarning: 'pin_memory' ... not supported on MPS` 被略去，
来自 `DLModel._fit` 里写死的 `pin_memory=True`，在 Mac 上无害。）

（2026-09-14 按 `DLModel` 重跑，上面是这次的原样输出，仅把临时目录路径缩写。与 2026-09-07 那次相比：
旧输出第一行 `Training DL model: TinyRegressor_total.pth` 那条 INFO 日志来自已删除的 `_auto_train`，
不再出现；`train_loss` / `test_loss` 逐位相同，`val_loss` 数值不同——原因没有追查，不把它写成结论。）

注意每个 epoch 里 `train_loss` 打印了 2 行、`test_loss` 打印了 2 行——
这就是「`_train_one_batch` 是 per-batch」的直接证据——改名之前它叫
`_train_one_epoch`，这段输出跟那个名字是直接矛盾的。

---

## ML / 树模型：XGBoostRegressor

`quantlab/ml_model/xgb.py:XGBoostRegressor(MLModel)` 预测未来收益：因子是 x，未来收益标签是 y，
`predict` 返回 `[T, S, L]`。文件名刻意叫 `xgb.py`——叫 `xgboost.py` 会在包内遮蔽顶层 `xgboost` 包。

它只实现 `MLModel` 的四个钩子加超参处理：

- `_fit_model`：把 `[T, S, F]` / `[T, S, L]` 展平成行，**任一标签非有限的行整行丢弃**；特征里的 ±inf
  转成 NaN 交给 xgboost 当缺失值（xgboost 遇到 inf 直接报错）；然后 `xgb.train`。
- 验证段有时间戳、但标签全是 NaN 时，按「没有验证段」处理并记 warning，不会把空 DMatrix 交给 xgboost。
- 多标签时每个标签一个输出（xgboost 多输出回归），头条指标只看主标签（第 0 个）。

### 最小示例（真的跑过）

合成面板（未来收益 = 0.1 × 信号因子 + 噪声，另有一个纯噪声因子）、CPU，一次 `train()` 加一次小规模 `train_cv`。
脚本放在仓库外的临时目录运行，没有提交。

```python
"""最小可跑示例：合成面板 + XGBoostRegressor，CPU，原生早停 + 一次小规模 train_cv。

运行方式（仓库根目录）：
    WANDB_MODE=disabled uv run python min_xgb_model.py
macOS 上本进程经模型层同时加载 torch 与 xgboost，必须带 OMP_NUM_THREADS=1：
    WANDB_MODE=disabled OMP_NUM_THREADS=1 uv run python min_xgb_model.py
"""

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import xarray as xr

from quantlab.base.config import MLConfig
from quantlab.ml_model.xgb import XGBoostRegressor
from quantlab.utils.metrics import regression_panel_metrics

N_TIMES, N_SYMBOLS = 160, 30
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype("timedelta64[D]")
SYMBOLS = [f"S{i:02d}" for i in range(N_SYMBOLS)]


class FakePanel:
    """假装自己是一个因子/标签对象，只实现模型层真正会调用的那几个方法。"""

    def __init__(self, arrays):
        self.names = list(arrays)
        self._ds = xr.Dataset(
            {n: (("timestamp", "symbol"), a.astype("float32")) for n, a in arrays.items()},
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )
        self.config = SimpleNamespace(start_date=None, end_date=None)

    def _reset_dataset_config(self):
        pass

    def _get_factor_names(self):
        return list(self.names)

    def cal(self):
        return self

    def get_features(self):
        return self._ds

    def get_labels(self):
        return self._ds

    def get_config(self):
        return {"name": "FakePanel", "factor_names": list(self.names)}


def make_config(save_dir):
    rng = np.random.default_rng(0)
    f_signal, f_noise = rng.standard_normal((2, N_TIMES, N_SYMBOLS))
    # 未来收益 = 0.1 * 信号因子 + 噪声
    ret = 0.1 * f_signal + 0.05 * rng.standard_normal((N_TIMES, N_SYMBOLS))
    return MLConfig(
        factors=[FakePanel({"f_signal": f_signal, "f_noise": f_noise})],
        labels=[FakePanel({"ret_fwd": ret})],
        model_save_dir=save_dir,
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date="2024-01-01", end_date="2024-06-08",
        train_start="2024-01-01", train_end="2024-04-29",   # 前 120 个时间点
        test_start="2024-04-30", test_end="2024-06-08",     # 后 40 个时间点
        early_stopping=True,
        early_stopping_patience=20,                          # 按 boosting 轮数计
        hyperparameters={"num_boost_round": 300, "max_depth": 3, "eta": 0.1, "nthread": 2},
    )


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "disabled")
    save_dir = tempfile.mkdtemp(prefix="xgb_ckpt_")

    model = XGBoostRegressor(make_config(save_dir))
    model.collect()
    model.train()

    booster = model.model
    print("rounds kept:", booster.num_boosted_rounds(), "| best_iteration:", booster.best_iteration)

    data = model.data_backend.get_xarray_dataset(["timestamp", "symbol"]).sel(
        timestamp=slice("2024-04-30", "2024-06-08")
    )
    test_x = model.to_array(data, model.get_factor_names())
    test_y = model.to_array(data, model.get_label_names())
    pred = model.predict(test_x)
    print("predict shape:", pred.shape)
    metrics = regression_panel_metrics(pred[..., 0], test_y[..., 0])
    print("test IC / RankIC:", round(metrics["ic"], 4), "/", round(metrics["rank_ic"], 4))

    ckpt = next(
        os.path.join(r, f) for r, _, fs in os.walk(save_dir) for f in fs if f.endswith(".joblib")
    )
    fresh = XGBoostRegressor(make_config(tempfile.mkdtemp(prefix="unused_"))).load(ckpt)
    print("reloaded predictions identical:", np.array_equal(fresh.predict(test_x), pred))

    import json

    saved = json.load(open(os.path.join(os.path.dirname(ckpt), "config.json")))
    print("config.json hyperparameters:", saved["hyperparameters"])
    print("config.json resolved_hyperparameters:", saved["resolved_hyperparameters"])

    # ---------------------------------------------------------------- 小规模 train_cv
    cv_model = XGBoostRegressor(make_config(tempfile.mkdtemp(prefix="xgb_cv_")))
    cv_model.collect()
    results = cv_model.train_cv(train_periods=60, gap_periods=2)
    print("folds:", len(results))
    for r in results[:2]:
        print(
            f"  fold {r['fold']}: train {r['train_start'][:10]}..{r['train_end'][:10]}, "
            f"test {r['test_start'][:10]}..{r['test_end'][:10]}, test_ic={r['test_ic']:.4f}"
        )
    print("result keys:", sorted(results[0]))
    print("mean test_ic:", round(float(np.mean([r["test_ic"] for r in results])), 4))
```

#### 真实输出

stdout（原样）：

```
$ WANDB_MODE=disabled OMP_NUM_THREADS=1 uv run python <仓库外临时目录>/min_xgb_model.py
rounds kept: 80 | best_iteration: 79
predict shape: (40, 30, 1)
test IC / RankIC: 0.8811 / 0.8663
reloaded predictions identical: True
config.json hyperparameters: {'num_boost_round': 300, 'max_depth': 3, 'eta': 0.1, 'nthread': 2}
config.json resolved_hyperparameters: {'objective': 'reg:squarederror', 'tree_method': 'hist', 'eta': 0.1, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'device': 'cpu', 'eval_metric': 'rmse', 'seed': 42, 'nthread': 2, 'num_boost_round': 300}
folds: 8
  fold 0: train 2024-01-01..2024-02-29, test 2024-03-03..2024-03-14, test_ic=0.8959
  fold 1: train 2024-01-13..2024-03-12, test 2024-03-15..2024-03-26, test_ic=0.8937
result keys: ['checkpoint', 'experiment_name', 'fold', 'test_end', 'test_ic', 'test_loss', 'test_mae', 'test_mse', 'test_r2', 'test_rank_ic', 'test_rmse', 'test_start', 'train_end', 'train_start']
mean test_ic: 0.8878
```

stderr 里是 `train_cv` 的 loguru INFO 日志，共 10 行，这里原样贴前 3 行：

```
2026-09-14 19:35:02.538 | INFO     | quantlab.base.model:train_cv:515 - Starting CV from 2024-01-01 to 2024-06-08 with 60 training periods and 2 periods gap
2026-09-14 19:35:02.538 | INFO     | quantlab.base.model:train_cv:521 - Total 8 folds will be created
2026-09-14 19:35:02.538 | INFO     | quantlab.base.model:train_cv:523 - Fold 0: Train [2024-01-01T00:00:00 to 2024-02-29T00:00:00], Test [2024-03-03T00:00:00 to 2024-03-14T00:00:00]
```

怎么读：

- `rounds kept` = `best_iteration + 1`：`EarlyStopping(save_best=True)` 把返回的 Booster 截断到了最优轮，
  落盘的 `.joblib` 就是最优模型。
- 测试 IC 高是因为标签就是由因子构造的——这是「管线把 x 和 y 对齐了」的证据，不是策略效果。
- 结果 dict 里 `test_start` / `test_end` 也以 `test_` 开头，但它们是日期，CV 均值只对数值型的 `test_*` 指标求。

### 真实数据用法（此例未实际运行）

> **此例未实际运行**：它依赖本机不存在的 Binance 分钟线 zarr 数据，且 `quantlab/config/__init__.py`
> 的工厂函数需要数据根目录（`--data-dir` / `QUANTLAB_DATA_DIR`）。

```python
import os

from quantlab.base.config import MLConfig
from quantlab.config import alpha158_config, spot_label_config
from quantlab.factor.alpha158 import Alpha158SpotKline
from quantlab.label.fret import Return
from quantlab.ml_model.xgb import XGBoostRegressor

alpha158 = Alpha158SpotKline(alpha158_config(symbols=["BTCUSDT", "ETHUSDT"]))
label = Return(spot_label_config("ret_1m", n_forward_periods=30, symbols=["BTCUSDT", "ETHUSDT"]))

model = XGBoostRegressor(
    MLConfig(
        factors=[alpha158],
        labels=[label],
        model_save_dir="./model_ckpt",
        factor_data_strategy="read",
        label_data_strategy="cal",
        start_date="2022-01-01", end_date="2022-10-01",
        train_start="2022-01-01", train_end="2022-08-01",
        test_start="2022-08-02", test_end="2022-10-01",
        early_stopping=True,
        early_stopping_patience=50,           # 按 boosting 轮数计
        hyperparameters={"num_boost_round": 1000},
    )
).collect()
model.train()

# 滚动交叉验证：4 折并发，每折 xgboost 用 核数 // 4 个线程
cv = XGBoostRegressor(
    MLConfig(
        factors=[alpha158], labels=[label], model_save_dir="./model_ckpt",
        factor_data_strategy="read", label_data_strategy="cal",
        start_date="2022-01-01", end_date="2022-10-01",
        early_stopping=True, early_stopping_patience=50,
        hyperparameters={"num_boost_round": 1000, "nthread": os.cpu_count() // 4},
    )
).collect()
results = cv.train_cv(train_periods=100_000, gap_periods=30, parallel=True, njobs=4)
```

`gap_periods=30` 对应标签的 `n_forward_periods=30`，理由见下面「ML 交叉验证」。

### 超参：默认值、合并规则、别名

`XGBoostRegressor.DEFAULT_PARAMS`：

| 键 | 默认值 | 说明 |
|---|---|---|
| `objective` | `"reg:squarederror"` | 回归 |
| `tree_method` | `"hist"` | |
| `eta` | `0.05` | 学习率 |
| `max_depth` | `6` | |
| `subsample` | `0.8` | |
| `colsample_bytree` | `0.8` | |
| `device` | `"cpu"` | |
| `eval_metric` | `"rmse"` | 逐轮曲线与早停判据 |
| `seed` | `config.random_seed` | 不在字典里，合并时注入 |
| `nthread` | **不设** | 交给 xgboost 默认（用满全部核） |

另有 `num_boost_round`，默认 `1000`。**它不是 xgboost 的参数**：从超参里单独取出，作为 `xgb.train`
的轮数上限，不进 params；小于 1 时 `ValueError`。

**合并规则：逐键覆盖。** `config.hyperparameters` 里给了的键覆盖默认值，没给的键保留默认值，
用户键优先（包括 `seed` 与 `nthread`，原样透传，代码不会改写）。`config.hyperparameters` 本身
不被修改——它记录的是你传进去的东西。

**sklearn 风格别名**先在用户字典上归一化成原生键，**再**合并默认值，所以别名同样能覆盖默认值
（例如 `learning_rate=0.3` 覆盖默认 `eta=0.05`，`random_state=7` 覆盖 `config.random_seed`）：

| 别名 | 原生键 |
|---|---|
| `n_estimators` | `num_boost_round` |
| `learning_rate` | `eta` |
| `random_state` | `seed` |
| `n_jobs` | `nthread` |
| `reg_alpha` | `alpha` |
| `reg_lambda` | `lambda` |

为什么必须归一化（xgboost 3.4.1 实测，直接把别名传给 `xgb.train`）：`learning_rate` 与默认 `eta`
同时出现时谁生效只取决于字典顺序；`n_estimators` 被忽略，只给一条 `Parameters: { "n_estimators" } are not used`
警告，轮数仍然取 `num_boost_round`；`random_state` 在已有 `seed` 时被**静默**忽略，连警告都没有。

**同一个参数的别名和原生键同时给出**（例如同时写 `eta` 和 `learning_rate`）时 `train()` 抛
`ValueError`，消息点名这两个键——绝不静默挑一个。

### 记录实际生效的参数

合并出来的参数加上轮数会被记下来：

- `config.json` 顶层的 `resolved_hyperparameters`；
- W&B run 的 config（`config.update({"resolved_hyperparameters": ...}, allow_val_change=True)`，
  在 `_init_model` 解析出参数之后补记——run 在那之前已经打开）。

为什么要记：`hyperparameters` 只记录你**传了什么**。日后 `DEFAULT_PARAMS` 一改，同一份
`hyperparameters` 训出来的就是另一个模型；`resolved_hyperparameters` 记录的是这次**实际用了什么**，
复现以它为准。它是记录不是输入：`load_model_from_config` 重建配置时丢掉这个键。

上面示例那次运行的原样输出：

```
config.json resolved_hyperparameters: {'objective': 'reg:squarederror', 'tree_method': 'hist', 'eta': 0.1, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'device': 'cpu', 'eval_metric': 'rmse', 'seed': 42, 'nthread': 2, 'num_boost_round': 300}
```

注意用户没给的 `subsample` / `colsample_bytree` 等都以默认值出现，`seed` 取自 `config.random_seed`。

### 早停判据

- `config.early_stopping=True` 且有可用验证段时，追加
  `xgb.callback.EarlyStopping(rounds=early_stopping_patience, data_name="val", save_best=True)`。
- **patience 按 boosting 轮数计**，不是 epoch。
- **判据是验证集上的 `eval_metric`，默认 RMSE**——不是 IC。对「x = 因子、y = 未来收益」的回归，
  RMSE 是和训练目标一致的默认；IC / RankIC 只在训练结束时算出来记进 summary，**不参与早停**。
- `eval_metric` 给成列表时，`EarlyStopping` 用的是**最后一个**：它的 `metric_name=None` 解析为
  `list(data_log.keys())[-1]`（xgboost 3.4.1 `xgboost/callback.py` 第 482 行）。
- 早停触发时，逐轮曲线的最后一个 step 是 `best_iteration + patience`，即触发停止的那一轮。
  W&B 回调必须排在 `EarlyStopping` **之前**：回调容器短路调用，排在后面会漏掉这一轮（实测）。
- `early_stopping=True` 但 `val_size=0`（或验证段标签全 NaN）：记 warning，跳过早停，跑满 `num_boost_round`。
- 以后想按 IC 早停，路子是 xgboost 的 `custom_metric`（在验证集上算截面 IC）加
  `EarlyStopping(metric_name=..., maximize=True)`。**现在没有实现**：截面 IC 需要知道每行属于哪个时间戳，
  得先把分组信息带进自定义指标。

### ML 交叉验证

`train_cv` 继承自 `BaseModel`，DL 与 ML 共用；以下几何由重构前捕获的 golden
（`tests/test_model_cv.py`）锁定。

**折几何**（`BaseModel._cv_folds`，全仓唯一一份折边界算术）：

- 滚动前推：训练段整段在测试段之前，每折整体向后挪一个测试段的长度。
- `train_periods` 的单位是**时间点个数**（bar 数），不是天数。
- `test_periods = train_periods // 5`（写死）。
- 第 i 折：训练段是下标 `[i·test_periods, i·test_periods + train_periods)`，之后空出 `gap_periods` 个
  时间点，再接 `test_periods` 个时间点的测试段；日期两端闭区间。
- 折数 `max(1, (总长 − train_periods − gap_periods) // test_periods)`；测试段越过数据末尾的折记 warning 并跳过，
  所以数据不够一折时返回空 list。
- 例：160 个时间点、`train_periods=60, gap_periods=2` → 测试段 12、共 8 折（上面示例的输出）。

**`gap_periods` 的用途**：隔开标签自身的前视窗口。`Return` 用
`shift(timestamp=-n_forward_periods)` 生成标签，不留 gap 的话，训练段末尾那几根 bar 的标签
已经包含了测试段开头的价格信息。一般取 `gap_periods >= n_forward_periods`。

**每折内部**：训练段尾部按 `val_size` 切出验证段，驱动**该折自己的**原生早停；每折一个 W&B run
（`{cls}_cv_fold_{i}`）和一个 `{cls}_cv_fold_{i}/{cls}_cv_fold_{i}.joblib`。

**返回值**：逐折结果 list，每项含 `fold`、`train_start`、`train_end`、`test_start`、`test_end`、
`experiment_name`、`checkpoint`（该折 `.joblib` 的路径），以及 `test_loss, test_mse, test_rmse, test_mae, test_r2, test_ic, test_rank_ic`。
各折 `test_*` 指标（只取有限值）的均值写进一个**独立的** `{cls}_cv_summary` run：`cv_mean_test_*` 与 `cv_n_folds`。
之所以单开一个 run：每折的 `_fit` 结束时已经 finish 了自己的 run，均值算出来时没有还开着的 run 可写。
（DL 头的结果只有日期、`experiment_name` 与 `checkpoint`，也不开 summary run。）

**并行**：`parallel=True` 在 joblib 的 threading 后端上为每折 `copy.deepcopy(self)`，面板数据会被复制
njobs 份，内存按此估算。xgboost 默认用满全部核，njobs 个折同时跑会严重超额订阅 CPU——建议在
`hyperparameters` 里设 `nthread ≈ 核数 // njobs`。**代码不会替你改写 `nthread`**（测试锁住并行 CV 之后它仍是用户给的值）。
顺序与并行两个分支消费同一个 `_cv_folds`，产出相同的折与 checkpoint。

### macOS / Linux：OpenMP 冲突

**症状（macOS 开发机）。** xgboost 3.4.1 的 macOS wheel 链接 Homebrew 的
`/opt/homebrew/opt/libomp/lib/libomp.dylib`，torch 自带一份 `torch/lib/libomp.dylib`。同一进程里两份
OpenMP 运行库冲突：

- 先加载 torch、再 `xgb.train` → `OMP: Error #179: Function pthread_mutex_init failed`，然后段错误；
- 先训练 xgboost、再跑 torch 运算 → **死锁**（进程挂住，CPU 为 0）。

`quantlab/base/model.py` 顶层就 import torch，所以任何用模型层训练 `XGBoostRegressor`、又在同一进程里
碰 torch 的场景都会撞上——包括把 DL 测试和 XGB 测试放在一次 pytest 里跑。

**试过并否决的方案：ctypes 预加载 Homebrew libomp。** 在 torch 之前
`ctypes.CDLL(".../libomp.dylib", mode=ctypes.RTLD_GLOBAL)` 确实能让「先 torch 后 xgboost」不再段错误，
但它让 **torch 自己**的 GRU 前向和一次 20000 行的 `cross_entropy` 段错误，跟 xgboost 用不用无关
（2026-09-14 实测，每种组合各起一个新子进程）。「先 import xgboost 再 import torch」则在第一个 torch 运算处死锁。
`KMP_DUPLICATE_LIB_OK=TRUE` 与 xgboost 的 `nthread=1` 也都无效。

**采用的方案：单线程 OpenMP。** 在 import torch **之前**设 `OMP_NUM_THREADS=1`，是唯一通过完整混合序列
（torch → xgboost → torch → 线程里并行 xgboost → torch）的设置。

- **测试**：`tests/conftest.py` 的第一段可执行代码就是
  `if sys.platform == "darwin": os.environ.setdefault("OMP_NUM_THREADS", "1")`，排在一切 import 之前；
  `setdefault` 让你显式设置的值优先。由 `tests/test_macos_openmp_guard.py` 锁住（守卫位置、非 macOS 不生效、
  不覆盖显式值，以及一个真实规模混合序列的子进程测试）。代价：macOS 上跑测试时 torch 与 xgboost 都是单线程。
- **你自己的脚本 / notebook（macOS）**：只要同一进程里同时用 torch 和 xgboost，就必须
  `export OMP_NUM_THREADS=1` 再启动，或者在 import torch（以及任何 quantlab 模型层模块）**之前**
  `os.environ["OMP_NUM_THREADS"] = "1"`。notebook 里要放在第一个 cell 的最前面，内核已经加载过 torch 就只能重启。
  上面两个示例都是这样跑的。

**Linux（生产训练环境）。** 守卫只在 `darwin` 上生效，Linux 上 `tests/conftest.py` 不设任何变量，包代码也没有
任何平台相关逻辑。Linux 上 torch 与 xgboost 的 wheel 通常都用 GNU OpenMP，预期没有这个冲突——但**这一点没有在
Linux 上验证过**。建议在 Linux 工作站上跑一次：

```bash
uv run pytest tests/test_xgb_model.py tests/test_ml_models.py -q
```

要留意的已知失败形态是 aarch64 Linux 上的 `cannot allocate memory in static TLS block`（导入 torch / xgboost / sklearn
时出现，通常与 libgomp 的加载顺序有关）。

---

## 保存与加载

### 存：`.pth` 与 `.joblib` 两条路

分流不再看模型对象的类型，而是看**变体**：`BaseModel._save_model(p)` 建目录、写 `config.json`，
然后调变体的 `_write_checkpoint(p)`；文件名后缀来自变体的 `checkpoint_suffix`。

| 变体 | 后缀 | 写 | 读 |
|---|---|---|---|
| `DLModel` | `.pth` | `torch.save(self.model.state_dict(), p)`——**只有权重，没有结构** | 先按当前数据形状 `_init_model` 重建网络，再 `load_state_dict` |
| `MLModel` | `.joblib` | `MlBackend().to_internal(self.model).write(p)`——整个对象序列化 | `MlBackend().read(p).get_model()`，**不调 `_init_model`** |

`quantlab/ml_model/backend.py:MlBackend` 就是 ML 这条路的持久化后端。`.joblib` 本质是 **pickle**：
反序列化会执行文件里的代码，**只加载自己训练、自己信任的文件**。

落盘目录是 `{model_save_dir}/{project_name}/{experiment_name}/{model_name}`，
`project_name` 带训练时刻的时间戳。如果目标目录已存在，`_save_model` **直接 `RuntimeError`，
不覆盖**——一次训练的产物不可替代，宁可让人换个名字重来。

### `config.json`：为什么权重旁边必须躺一份配置

`_save_model` 会在权重的同级目录写一个 `config.json`，内容是 `get_config()`：
`DLConfig` 摊平之后，把 `factors` / `labels` 两个字段**就地替换成每个因子/标签自己的配置字典**。

嵌套而不是只记引用，是因为这份 JSON 的目的是「单凭它就能把整条链路复现出来」。
只留个引用，换台机器就复现不了。

上面那次真实训练（2026-09-14 重跑）产出的 `config.json`（节选，略去了 `num_workers` / `lr_refit` 两个字段）：

```json
{
    "factors": [{"name": "FakePanel", "factor_names": ["f0", "f1", "f2"]}],
    "labels":  [{"name": "FakePanel", "factor_names": ["y0"]}],
    "model_save_dir": "/var/folders/.../T/tiny_ckpt_y35yla00",
    "factor_data_strategy": "cal",
    "label_data_strategy": "cal",
    "start_date": "2024-01-01",
    "end_date": "2024-07-18",
    "hyperparameters": {"hidden": 16},
    "lr": 0.01, "epochs": 5,
    "early_stopping": true, "early_stopping_patience": 5,
    "batch_size": 64, "val_size": 0.2, "random_seed": 42,
    "train_start": "2024-01-01", "train_end": "2024-05-01",
    "test_start": "2024-05-02",  "test_end": "2024-07-18",
    "name": "__main__.TinyRegressor"
}
```

ML 头的 `config.json` 另有一个顶层 `resolved_hyperparameters`：实际交给库的参数（默认值合并用户覆盖之后），
是记录不是输入，见「ML / 树模型」。DL 头没有这个键。

关键在最后那个 `name`。它由 `BaseModel.import_path` 生成：

```python
f"{self.__class__.__module__}.{self.__class__.__qualname__}"
```

这个字符串就是**重建这个类所需的全部信息**。`quantlab/utils/module.py` 拿它做反向解析：

```python
def get_cls_from_path(path: str):
    module_path, class_name = path.rsplit(".", 1)      # "dl_model.rnn_classification" + "RNNClassifier"
    module = importlib.import_module(module_path)      # 动态 import 那个模块
    return getattr(module, class_name)                 # 从模块里取出类对象
```

于是 `load_model_from_config(cfg)` 可以递归地把整棵树重建回来：
先把每个 factor 的 `name` 解析成因子类、每个 factor 里的 `dataset.name` 解析成数据集类，
最后把模型类实例化——配置类取**模型类自己声明的** `cls.config_cls`（以前写死 `DLConfig`，
ML 的配置会被静默建成 `DLConfig`）；`resolved_hyperparameters` 这个记录键在构建配置前被丢弃，
只丢它，其他未知键照样 `TypeError`。这是这个项目里**唯一的插件注册机制**——
没有注册表、没有 DI 容器，就靠「点分路径 + `importlib`」。

代价是路径变成了序列化契约的一部分：**重命名或移动一个模型/因子的模块，
所有旧检查点的 `config.json` 就都失效了**。

（另见「常见坑」第 5 条：在脚本里直接定义模型类会让这里存成 `__main__.X`，无法重建。）

### 取：`load()`

```python
model = TinyRegressor(cfg).collect().load(ckpt_path)
pred = model.predict(torch.randn(3, model.num_symbols, model.num_factors))
# -> torch.Size([3, 4, 1])   （实测）
```

`load()` 先查文件存在，再查后缀等于本变体的 `checkpoint_suffix`——对 ML 模型传 `x.pth`
会在构建任何东西之前 `ValueError`，不会把文件交给错误的反序列化器。然后调变体的 `_read_checkpoint`。

DL 头：先按**当下数据的形状**调 `_init_model` 把网络搭出来，再灌权重。
这有个重要的副作用：**必须先 `collect()`**，否则 `self.num_symbols` 会因为
`XrBackend` 还没有数据而抛 `AttributeError: Please cal 'read' or 'to_internal' first.`。

ML 头：直接读回整个模型，不调 `_init_model`，所以**不需要先 `collect()`**。

好处是：如果因子数量变了，`load_state_dict` 会当场报形状不匹配，
而不是带着一个错的模型继续跑。

---

## 已知的不完整之处

这些都是 2026-09-07 读代码时逐条核实过的，不美化。

**1. 已实现：ML / 树模型路径。**（**2026-09-14 实现**，260914-lno）
2026-09-07 记录时，`MLConfig` 这个 dataclass 和 `MlBackend` 都存在，但没有任何一条路把它们接起来，
`_auto_train` 对 `MLConfig` 直接抛 `NotImplementedError`。现在模型层拆成了
`BaseModel` / `DLModel` / `MLModel` 三层，`MLModel` 是非 torch 模型的正式变体，
`quantlab/ml_model/xgb.py:XGBoostRegressor` 是第一个出厂头，交叉验证同样可用。
见「MLModel 的四个钩子」与「ML / 树模型」两节。

**2/3/4. 回测骨架还是空的——但现在它会说出来。**（**已于 2026-09-07 改造**）

`_do_vecbt`、`_vecbt`、`RNNClassifier._vecbt`、`_train_dl(backtest=...)`
是四块互不相连的半成品。它们**没有被删掉**：CLAUDE.md 已经确认
`MLConfig`/xgboost 这条非 torch 路径要做，一个日后同时服务 torch 和非 torch
模型的回测钩子挂在基类上位置是对的；端到端回测归 **Phase 6**，只是现在还没内容。

改的是它们**失败的方式**——空实现要么报错，要么就不该存在，
「安静地返回 None」是两者里最糟的一种：

| 曾经 | 现在 |
|---|---|
| `_do_vecbt` 读完 `backtest_data`、算出一个局部变量 `price`，函数就结束了，调用方拿到 `None` | 两个既有参数检查保留在前（`backtest_data` 没配是今天就能改的错误），之后 `NotImplementedError`，消息里点名 Phase 6 |
| `_train_dl(backtest=...)` 签名里声明、函数体里一次都没引用 | 传真值时在**训练开始之前**就 `NotImplementedError`。拒绝必须前置：这参数真实调用里只传一次，后面那段训练要跑几个小时，训完再说「其实我不支持」跟不说差别不大 |
| `BaseModel._vecbt` 是一句光秃秃的 `raise NotImplementedError`，异常消息是空字符串 | 仍是 stub（**刻意保留**），但消息点名 Phase 6 |
| `RNNClassifier._vecbt` 算完四个 pandas Series 就到文件末尾——不返回、不调用 vectorbt、不报错 | 先抛 `NotImplementedError`；那四行原样抄进 docstring 保留 |

最后一格值得单独说：那四行**跑不起来**。写测试时实测到

```
pandas.errors.IndexingError: Unalignable boolean Series provided as indexer
```

——`long_exits = long_entries[short_entries == 1]` 拿 `short_entries` 的布尔掩码
去索引 `long_entries`，而这两个是同一个 Series 的互补子集、index 天然不相交。
除了「一个多头信号都没有」的退化输入，任何信号序列都会炸（`[1,0,1]`、`[0,1]`、
`[1,1]`、`[1,0,0,1,1]` 全部抛异常）。让它先执行，等于把「Phase 6 还没做」换成
一句莫名其妙的 pandas 索引错误——那不是变诚实，只是换了一种骗法。所以四行
降级成 docstring 里的记录（保留作者的进出场约定：0=做空、1=做多），
`short_exits` 那行同样可疑，Phase 6 接手时两行都要重新推导，不要照抄。

**全库仍然没有任何地方调用 `_vecbt` / `_do_vecbt`**，真正跑回测的代码还是
`train_model.py` 脚本里手写的那段。不要以为「训练完会自动回测」——只是现在
你如果那么以为，会立刻收到一个点名 Phase 6 的异常，而不是一片安静。

由 `tests/test_model_layer.py::test_train_dl_rejects_a_truthy_backtest_flag`
（并断言拒绝发生在训练之前）、`::test_train_dl_still_trains_when_backtest_is_falsy`
（保证默认路径没被这道闸门误伤）、
`::test_do_vecbt_says_it_is_unbuilt_instead_of_returning_none`、
`::test_vecbt_stub_is_still_a_stub_and_names_phase_6` 和
`tests/test_dl_models.py::test_rnn_classifier_vecbt_raises_instead_of_returning_none`
共同锁住。

**5. `quantlab/dl_model/mlp.py:MLPRegressor` 曾经是坏的，三处。**（**已于 2026-09-07 修复**）
以前它同时踩了三个坑，而且是层层挡在后面的三个——修掉一个才能看见下一个：
- 缺 `_val_one_batch`，是抽象类，**根本实例化不了**（实测
  `MLPRegressor.__abstractmethods__ == frozenset({'_val_one_batch'})`）；
- `_init_model(self, num_symbols, num_features, num_labels)` 少了 `hyperparameters` 参数，
  而基类是用关键字 `hyperparameters=` 调它的 →
  `TypeError: MLPRegressor._init_model() got an unexpected keyword argument 'hyperparameters'`；
- `_preprocess` 的实现是 `data.fillna(0.0)`，那是 xarray 的 API，
  但传进来的是 `torch.Tensor` →
  `AttributeError: 'Tensor' object has no attribute 'fillna'`。

现在三处都补齐了：`_val_one_batch` 存在且**返回** `val_loss.detach()`（返回值契约见
「常见坑」#2 旁注——epoch 循环要 `float()` 它）；`_init_model` 收 `hyperparameters`，
两个隐藏层宽度从 `hidden_size1`/`hidden_size2` 读，缺省仍是原来硬编码的 512/256；
`_preprocess` 换成 `torch.nan_to_num(data, nan=0.0)`，跟两个 RNN 头一致。
`tests/test_dl_models.py::test_mlp_regressor_trains_two_epochs_and_predicts` 真的跑了
两个 epoch 并断言 `fc1` 权重发生了变化——「能 import」不算证据。

它**没有**被删掉：CLAUDE.md 的架构表把它列为具名组件，而后续阶段需要的正是一个
baseline 回归模型。

遗留的一处（没修，是有意的）：`MLPRegressor` 的 reshape 写在
`_train_one_batch`/`_test_one_batch` 里，而 `MLP.forward` 只是一串 `nn.Linear`，
所以 `predict()` 要求调用方传**已经拍平**的 `[num_times, num_symbols * num_features]`，
不是训练时那个三维张量。补这个缺口要么改 `quantlab/base/model.py:DLModel._predict`，要么改公开的
`MLP` 模块接受什么，两者都超出了这次的范围。

**6. `quantlab/dl_model/rnn.py` 里那个 `RNNClassifier` 是一份坏掉的旧副本。**（**已于 2026-09-07 删除**）
`rnn.py` 的 `ModelRBaseCrypto` 最后一层是 `nn.Linear(..., 1)`（回归用），
但它里面的 `RNNClassifier._train_one_batch` 却写了 `primary_pred.reshape(D * T, 2)`
——元素个数对不上，必炸。真正在用的分类器是 `quantlab/dl_model/rnn_classification.py:RNNClassifier`
（那份的基础块输出 2 类，`self.out = nn.Linear(num_aux * 2, 2)`，逻辑自洽），
`train_model.py` 导入的也是它。

**两个同名类、一个能跑一个不能，本身就是个陷阱**，所以坏的那份删掉了：删除前
grep 确认全仓对 `RNNClassifier` 的引用无一例外解析到 `rnn_classification.py`，
从 `dl_model.rnn` 导入的只有 `RNNRegressor`（`cal.py`、`tests/test_dl_models.py`）。
`ModelRBaseCrypto` / `ModelRCrypto` 留着，`RNNRegressor` 在用。随之清掉了六个
只被那份副本用到的 sklearn 分类指标 import。要找回它 `git show` 即可——从没跑通过
的实现，日后从历史里捞出来比现在维护它便宜。

现在 `quantlab/dl_model/rnn.py` 里只有 `RNNRegressor`。

**7. `update()`（在线学习）曾经引用不存在的配置字段。**（**已于 2026-09-07 修复**）
`rnn.py` 和 `rnn_classification.py` 的 `update()` 第一行是 `if self.config.lr_refit <= 0.0`，
但 `DLConfig` **没有 `lr_refit` 字段** →
`AttributeError: 'DLConfig' object has no attribute 'lr_refit'`，整条在线学习路径不可用。

现在 `DLConfig` 有 `lr_refit: float = 0.0`。**补字段而不是删掉这处读取**，理由在代码本身：
`update()` 自己写着「取零即 return」，作者本来就是按「配置里的一个开关」设计的；
删掉读取就必须替微调步骤挑一个学习率，而 docstring 明确要求它要**小于**训练用的 `lr`
——那是个建模决策，代码里没有依据。默认 0.0 意味着没人显式开启时 `update()` 是纯 no-op，
所以这个字段不改变任何既有行为。
`tests/test_dl_models.py` 两头都锁：默认配置下 `update()` 一个参数都不动，
`lr_refit > 0` 时必须真的走一步优化器（否则「字段加了但没人读」也能骗过测试）。

**同一天修的第二处：`update()` 以前每次调用都现场新建一个 AdamW。**
Adam 的一阶/二阶动量存在优化器实例里，「每步新建」就是每步清零——不报错，
只是悄悄退化成一个带古怪 warmup 的 SGD，而 `update()` 的用途正是真正的在线 /
单步训练，动量累积是它的全部意义。现在走
`DLModel._get_refit_optim()`，实例级缓存，按 `(self.model 这个对象, lr_refit)`
命中：`load()` 或再次 `_init_model()` 换掉 `self.model` 之后缓存自动失效，
不用任何调用点记得去手动作废——一个指向旧参数张量的陈旧优化器会静默更新一堆
游离张量，比原来的 bug 更糟。锁它的测试断言的是**状态**（两次 `update()` 之后
每个参数的 `state[p]["step"] == 2`，且 `exp_avg` 非零），不是 `id()` 相等：
后者在状态被清空时照样能通过。

它跟 `self.optim` 是两回事——`_train_dl` 结束时 `self.optim = None` 是有意的
（见「常见坑」#8 旁注），微调优化器没有把它复活。

**8. 「从 xarray 到推理张量」没有被封装。**（**已于 2026-09-07 修复**）
以前 `_train_dl` 内联了转换逻辑，推理方要手抄一遍。现在是
`DLModel.to_tensor(data, variables)`（2026-09-14 之前定义在基类上），训练和推理共用；`train_model.py` 已改为调用它。
仍然**没有**一个 `predict_from_xarray()` 把「切时间窗 + 选因子 + 填 NaN + 转张量 + 推理」
一次做完，调用方还是要自己写那三行。

---

## 常见坑

**1. `early_stopping=False` 会直接崩。**（**已于 2026-09-07 修复**）

曾经：`_train_dl` 里 `best_loss` / `early_stopping` / `patience` / `counter` 四个变量
只在 `if self.config.early_stopping:` 里初始化，但 epoch 循环末尾的
`if early_stopping: break` 是**无条件**执行的，于是：

```
UnboundLocalError: cannot access local variable 'early_stopping'
where it is not associated with a value
```

现在这四个变量在 epoch 循环之前**无条件初始化**，`early_stopping=False` 是完全正常的配置，
会老老实实跑满 `epochs` 个 epoch。回归锁：
`tests/test_model_layer.py::test_early_stopping_disabled_runs_all_epochs`。

**2. 早停的计数器是按 batch 走的，不是按 epoch。**（**已于 2026-09-07 修复**）

曾经：`counter += 1` 写在验证 batch 循环**内部**，一个 epoch 里有几个验证 batch，
counter 就可能加几次。实测（`batch_size=16`，每 epoch 2 个验证 batch，`patience=3`）
在 epoch 2 就被「早停」了；那时设 patience 得按 `patience / 每epoch验证batch数` 折算。

现在验证循环只负责按样本数加权累加，循环结束后折算出**一个 epoch 级别的验证损失**，
早停判断在循环外每个 epoch 只做一次。`early_stopping_patience=N` 就是字面意思：
**连续 N 个 epoch 的验证损失没有改善**。回归锁：
`tests/test_model_layer.py::test_early_stopping_patience_counts_epochs_not_batches`
（它刻意让每个 epoch 有 4 个验证 batch——只有一个 batch 的用例区分不出这两种语义）。

注意 `_val_one_batch` 的返回值现在会被 `float()` 转成标量参与加权平均，
所以它必须返回一个 0 维张量或 python 数（原本就是这么约定的）。

> 早停还有另外一半——「把最好的那一轮**留下来**」——它曾经完全没做，见第 13 条。

**3. 张量的因子列顺序是「字母序」，不是 `get_factor_names()` 的顺序。**
（**已于 2026-09-07 修复**）

曾经是最阴的一个。`_train_dl` 里的 `.sortby(["timestamp", "symbol", "variable"])`
把 `variable` 坐标**按字母排序**了，而 `get_factor_names()` 返回的是**配置里的顺序**：

```python
sub = ds[['zeta', 'alpha', 'mid']]
sub.to_dataarray().coords['variable']                       # ['zeta', 'alpha', 'mid']
sub.to_dataarray().sortby([...,'variable']).coords['variable']  # ['alpha', 'mid', 'zeta']
```

**它在标签上是有实际后果的。** `RNNClassifier` 把 `y[:, :, 0]` 当作 primary target，
而 `train_model.py` 写的是 `labels=[label1(30期), label2(60期), label3(120期)]`，
名字分别是 `ret_30` / `ret_60` / `ret_120`。字母序是 `ret_120 < ret_30 < ret_60`，
所以真正被当作 primary target 的是 120 期收益，不是作者写在第一位的 30 期——
不报错、不警告。

现在转换统一走 `DLModel.to_tensor(data, variables)`（底下是 `BaseModel.to_array`）：`sortby` 只排
`timestamp` / `symbol`，最后一维用 `.sel(variable=variables)` 按调用方声明的顺序**钉死**。
`train_model.py` 的推理路径也改成调用同一个方法，两侧不会再漂。回归锁：
`tests/test_model_layer.py::test_tensor_variable_axis_follows_declared_order`
（因子声明成 `zeta, alpha, mid`、标签声明成 `ret_30, ret_60, ret_120`，
两组都刻意不是字母序，否则这个测试会因为错误的原因通过）。

> **不做向后兼容**（用户 2026-09-07 锁定的决定）：本仓库是实验性质的，
> 旧检查点的列顺序与新代码不一致，直接作废重训即可，不加兼容开关、不加版本戳。

顺带一提，`_assert_shape_match_x` / `_assert_shape_match_y` 只查列**数**不查列**名**，
所以这类错位从来指望不上它们。

**4. dtype 曾经基类不管，float64 面板根本训不了。**（**已于 2026-09-07 修复**）
`torch.from_numpy` 忠实继承 numpy 的 dtype，而 `quantlab/dl_model/` 里每个 `nn.Module`
的权重都是默认的 float32。三个出厂模型头的 `_preprocess` 都只做
`torch.nan_to_num`，一个 `.float()` 都没有，于是真实面板一进 forward 就死：

```
ValueError: RNN input dtype (torch.float64) does not match weight dtype
(torch.float32). Convert input: input.to(torch.float32), or convert model:
model.to(torch.float64)
```

float64 不是假想的——它就是两条非 KunQuant 数据路径的产物：Polars 那条
（`FactorPolars` / `PlBackend`）和 pandas 那条（`StockDataset` 读 Tiingo
parquet）给回来的都是 float64。也就是说 CLAUDE.md 写明的第二个因子后端
**训不了**，而 `DLConfig(factors=[kunquant 因子, polars 因子])` 正是 Phase 03
D-03 要保证的可互换性。

现在 `DLModel.to_tensor` 在**转换的那一个接缝上**统一 dtype：浮点面板一律转成
`torch.get_default_dtype()`。放在这里而不是放进各个头的 `_preprocess`，是因为
`_preprocess` 有三份实现、第四个头一定会忘；`to_tensor` 是面板变成张量的唯一入口。
取 `get_default_dtype()` 而不是写死 `float32`，是为了跟随 torch 的全局设置——
谁要是 `torch.set_default_dtype(torch.float64)` 建了 float64 的模型，写死 float32
就是把同一个 bug 镜像了一遍。

**这是一个明写的取舍**：float64 → float32 会掉精度。对行情因子来说这是对的交易
（torch 模块本来就是 float32），但它是个决定，不是个意外。只转**浮点**：整型 /
布尔面板（成分股掩码、类别编码）原样穿过，静默转成浮点会把含义糊掉。

回归锁：`tests/test_dl_models.py::test_a_shipped_head_trains_on_a_float64_panel`
（参数化 `RNNRegressor` / `RNNClassifier`，面板走
`DataFrame.set_index([...]).to_xarray()` 这条真实摄取路径造出真的 float64——
预先 `.astype("float32")` 的面板什么都证明不了，那正是这个缺陷两次逃逸的原因）、
`::test_to_tensor_downcasts_a_float64_panel`、
`::test_to_tensor_follows_torchs_default_dtype_not_a_hardcoded_float32`、
`::test_to_tensor_leaves_non_floating_panels_alone`。

自己写模型头时**不必**再在 `_preprocess` 里 `.float()`（下面示例里那一句留着是无害的
恒等操作），但仍然要处理 NaN。

**5. 模型类不要定义在 `__main__` 脚本里。**
`import_path` 用 `self.__class__.__module__`，脚本里定义的类会存成 `__main__.TinyRegressor`
（上面真实的 `config.json` 就是这样），`get_cls_from_path` 之后没法把它 import 回来。
要走 `load_model_from_config` 的话，模型类必须住在一个真实模块里（比如 `quantlab/dl_model/xxx.py`）。

**6. 验证集切分丢一行。**（**已于 2026-09-07 修复**）
曾经是 `val_x_t = train_x_t_all[train_split + 1:]`，第 `train_split` 行既不在训练集
也不在验证集。现在切点是 `train_split:`，训练集加验证集的行数正好等于训练区间的
时间点数。回归锁：`tests/test_model_layer.py::test_val_split_keeps_every_training_row`。

注意这**不是**时序留白（purge/embargo）：训练段的最后一根和验证段的第一根是相邻的
bar，标签又是前视收益，泄漏是存在的。真要做留白得自己按 `n_forward_periods` 切，
基类不提供。

**7. `train_cv` 的测试段长度写死为训练段的 1/5。**
`test_periods = train_periods // 5`，没有参数可调。
另外 `train_periods` 的单位是**时间点个数**（bar 数），不是天数。

**8. 训练完模型就没了。**（**已于 2026-09-07 修复**）
曾经 `_train_dl` 最后是 `del self.model; del self.optim`（为了 CV 时不累积显存），
于是 `model.train()` 之后不能直接 `model.predict(...)`，要先把刚存下来的权重
`model.load(...)` 回来。

现在只丢优化器（`self.optim = None`）——Adam 的一阶/二阶动量约是参数量的 2 倍，
那才是真正占显存的部分，而且训练之外没人读它。模型留着，`train()` 之后可以直接
`predict()`。回归锁：`tests/test_model_layer.py::test_model_is_usable_immediately_after_train`。

**9. `pin_memory=True` 是写死的。**
`DataLoader` 里硬编码，Mac（MPS）上会每次打印 UserWarning。无害，但吵。

**10. `predict()` 不会自动切 `eval()`，也不在 `no_grad` 里。**
（**已于 2026-09-07 修复**）

曾经：`_predict_nn` 只做了 `to(device)` + `_preprocess` + `model(data)`，
没有 `model.eval()`、没有 `torch.no_grad()`；而 `load()` 新建的 `nn.Module`
默认就处在 training 模式。实测：

```
model.training after load(): True
requires_grad on output:     True
```

后果是**推理时 dropout 是开着的**——`train_model.py` 的配置里
`dropout_rates=[0.5, 0.3, 0.3, 0.3, 0.3]`，预测结果里混着一半的随机丢弃，
每次调用还都不一样；同时因为没有 `no_grad`，整张计算图被留着白吃内存。

现在 `_predict_nn` 自己会 `self.model.eval()` 并在 `torch.no_grad()` 里前向，
调用方直接 `model.predict(x)` 就行，不用再手动包一层。**副作用是模型会留在
eval 模式**——要接着训练不用管，`_train_dl` 每个 epoch 开头本来就会调
`self.model.train()`。回归锁：
`tests/test_model_layer.py::test_predict_runs_in_eval_mode_without_grad`
（同一份输入连调两次，断言结果逐位相等）。

**11. 配置 setter 有副作用。**
`BaseModel.config = cfg` 会**就地修改**你传进来的因子和标签对象的 `config.start_date` /
`end_date`。同一个因子实例喂给两个日期区间不同的模型，后者会把前者的区间改掉。
CV 并行那条路 `copy.deepcopy(self)` 就是为了躲这个。

**12. `RNNRegressor._init_model` 曾经收下 `hyperparameters` 然后原样扔掉。**（**已于 2026-09-07 修复**）
签名上写着 `hyperparameters: dict`，函数体里每一个值都写死：
`hidden_sizes=[256, 128, 64]`、`dropout_rates=[0.1, 0.1, 0.1]`、
`hidden_sizes_linear=[32]`、`model_type="gru"`。`config.hyperparameters` 被静默丢掉。
改了配置、跑完一轮、拿到一个跟改之前逐位相同的模型——没有任何地方提示你配置没生效。

这跟这一批一起删掉的另外三处「声明了、收下了、从不引用」是同一个谎
（`_train_dl(backtest=...)`、`get_crypot_currency(name=...)`、
`XrBackend.get_xarray_dataset(indexes)`），而同一批里 `MLPRegressor._init_model`
已经改成读它了——修那三个、留这一个说不过去。

它藏得住是有具体原因的：测试给 `RNNRegressor` 喂的是 MLP 形状的
`{"hidden_size1": 16, "hidden_size2": 8}`，**正因为参数被忽略**才通过；真读了反而会炸。
这是「因为错误的原因而变绿」。

现在它跟 `RNNClassifier` 一样真的读这个 dict，但每个值都用
`.get(..., <原来写死的字面量>)` 取，**任何既有配置建出来的模型结构都不变**。
回归锁：`tests/test_dl_models.py::test_rnn_head_hyperparameters_reach_the_built_module`
（断言在**建出来的模块**上——「读了」和「收下就扔」唯一的区别就是那个数字有没有出现在
某一层里）和 `::test_rnn_regressor_defaults_preserve_the_previously_hardcoded_shape`。
所有 RNN 测试也一并改成显式传 RNN 形状的 hyperparameters。

> 三个头对同一个抽象钩子仍然有两种取值习惯：`MLPRegressor` / `RNNRegressor` 用
> `.get()` 带默认值，`RNNClassifier` 用 `[...]`（缺键直接 `KeyError`）。统一它们
> 是另一个决定，这次**没有**做——`RNNClassifier` 没有「原来写死的值」可以当默认值，
> 硬给一个等于替使用者拍板网络结构。

**13. 早停算出了最优 epoch，存下来的却是等待期里最差的那一轮。**（**已于 2026-09-07 修复**）

曾经：`best_loss` 是个**只写变量**——它唯一的用途是喂 patience 计数器，从来没有人
把产生它的那份权重快照下来。而 `_save_model` 在整个 epoch 循环**之后**才跑一次，
存的就是循环退出那一刻内存里的东西。早停触发时，退出的那个 epoch 按定义是
「连续 `patience` 个没有改善」里的最后一个：

```
epoch 0  val_loss 3.0   ← counter 归零
epoch 1  val_loss 1.0   ← 最优
epoch 2  val_loss 2.0   ← counter 1
epoch 3  val_loss 2.0   ← counter 2
epoch 4  val_loss 2.0   ← counter 3 == patience，break
                          checkpoint 里存的是 epoch 4 的权重
```

早停本来是两件事：**挑出最好的**、**别再浪费时间**。这份实现只做了后一件，
还把前一件反着做了——花完 patience 预算找到的最优点，恰恰是被丢掉的那个。
`best_loss` 越算越像个功能，实际上没有任何下游。

现在：`epoch_val_loss < best_loss` 成立时顺手快照一份 `state_dict`，循环结束后、
`_save_model` **之前**把它灌回模型。三个细节：

- **快照放 CPU**（`v.detach().cpu().clone()`）。`load_state_dict` 是原地拷贝，
  CPU 张量灌回 CUDA 模型完全正常，所以 GPU 训练不用为这份快照多付一倍显存；
  代价是主存里多一份参数（不含优化器状态）——这就是这个修复的价格，写在这里。
- **两条退出路径都回滚**：早停 `break` 出来的，和 epoch 跑满自然退出的。后者也回滚，
  因为 `early_stopping=True` 表达的是「按验证损失挑 checkpoint」这个意图，循环是撞上
  patience 还是撞上 `epochs` 上限纯属排期的偶然；同一份配置一种退法存最优、另一种
  退法存最后一轮，说不通。
- **只在早停打开时回滚**。`early_stopping=False` 那条路根本不维护 `best_loss`，
  也没表达过任何按验证损失择优的意思，替它改掉存哪一轮是没人要求过的行为变更——
  它仍然老老实实存最后一个 epoch。

回归锁三条，都断言在**从磁盘读回来的 checkpoint** 上（断言内存里的 `self.model`
测不到 `_save_model` 写了什么）：
`tests/test_model_layer.py::test_early_stopping_saves_the_best_epoch_not_the_waited_out_one`、
`::test_early_stopping_saves_the_best_epoch_when_epochs_run_out`、
`::test_early_stopping_off_still_saves_the_last_epoch`（守住上面那条边界）。
用例里验证损失是**先降后升**的脚本，且每个 epoch 把全部参数刷成 `float(epoch)`——
只有这样「最优」和「最后」才是两个可区分的数字，否则这个断言会因为错误的原因变绿。
