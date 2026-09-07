# 模型层（Model）

> 代码位置：`base/model.py`（抽象基类 `BaseModel`）、`base/config.py`（`DLConfig` / `MLConfig`）、
> `dl_model/rnn_classification.py`、`dl_model/rnn.py`、`dl_model/mlp.py`（三个具体模型头）、
> `ml_model/backend.py`（非 torch 模型的持久化后端）、`utils/module.py`（按点分路径重建类）。
> 一个真实的端到端调用脚本：`train_model.py`。

---

## 一句话

把「拿因子面板和标签面板去训练一个模型」这件事里**所有跟模型无关的部分**——取数、
对齐、切分训练/验证/测试、跑 epoch、早停、存检查点、记实验——一次性写在基类里；
写一个新模型只需要回答五个问题：网络长什么样、一个 batch 怎么训、怎么验、怎么测、
张量进模型前怎么洗。

---

## 它吃什么、吐什么

### 吃：两份 xarray 面板

模型层不直接读磁盘、不直接碰行情。它拿到的是配置里塞进来的**因子对象**和**标签对象**，
然后调用它们的公共契约取数（`base/model.py:_collect_all_features` / `_collect_all_labels`）：

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

从 xarray 变成 torch 张量的那段是整层最该看懂的地方。它现在被封装成了
`BaseModel.to_tensor(data, variables)`，训练和推理走的是同一份实现：

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

顺带一提：`_train_dl` 的 `DataLoader` 用了 `shuffle=True`。打乱的是**截面之间**的顺序，
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

按调用顺序过一遍 `BaseModel`：

| 你调什么 | 基类做了什么 |
|---|---|
| `__init__(config)` | 接下配置；`_set_random_seed` 把 python / numpy / torch CPU / torch CUDA 四个随机源和 cudnn 的算法自动择优一次性钉死。**注意此时既不建模型也不读数据**——建模型要先知道数据形状。 |
| `config = ...`（setter） | 把训练区间**下推**给每一个因子和标签（`_reset_factors_config` / `_reset_labels_config`），并把 `config.name` 写成本类的完整导入路径。 |
| `collect()` | 取特征、取标签、`combine_by_coords`、`sortby`、灌进 `XrBackend`。返回 `self`，可以链式写 `Model(cfg).collect().load(ckpt)`。 |
| `train()` | 生成带时间戳的实验名 → `_init_wandb` → `_auto_train` → `_train_dl`。 |
| `_train_dl()` | 切 train/test（按配置日期）→ `to_tensor` 转张量 → 形状校验 → 从训练段**尾部**按 `val_size` 切验证集 → 建 `DataLoader` → epoch 循环 → 按 epoch 早停 → `_save_model` → `wandb.finish()` → `self.optim = None`（模型保留）。 |
| `train_cv(...)` | 沿时间前滚切多折，每折独立训练一个模型。 |
| `load(path)` | 按当前数据形状重建网络，再灌权重。 |
| `predict(tensor)` | `model.eval()` → `to(device)` → `_preprocess` → `model(x)`，整段在 `torch.no_grad()` 里。 |

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
`SpotReturn` 的标签是 `shift(timestamp=-n_forward_periods)` 得来的，
不留 gap 的话训练段末尾那 n 根 bar 的标签里已经包含了测试段开头的信息。
测试段长度**固定**为训练段的 1/5（`test_periods = train_periods // 5`，写死的，不可配）。

**模型的输入输出维度取自数据而不是配置**（`_init_model_and_optim`）：
`num_symbols` / `num_factors` / `num_labels` 都是从已收集的 xarray 面板上现算的属性。
所以你加一个因子，网络的输入层自动变宽，不需要在任何地方同步一个数字。

**形状校验挡在训练之前**（`_assert_shape_match_x/y`）：形状不匹配在 torch 里
常常被广播悄悄吸收掉，最后表现为「loss 不下降」。挡在这里，问题停在它产生的地方。
但要注意它**只校验个数，不校验列名顺序**——见下面「常见坑」。

**每折训练前 `copy.deepcopy(self)`**（`_train_fold_with_config`）：
并行跑 CV 时各折会同时改写 `config.train_start` 等字段，共用一个实例会互相覆盖。

---

## 子类的五方法契约

`BaseModel` 有 5 个 `@abstractmethod`，少实现一个类就实例化不了
（这不是理论——`dl_model/mlp.py:MLPRegressor` 曾经漏了 `_val_one_batch`，
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
见「常见坑」第 2 条）。所以它必须返回一个能 `float()` 的标量 loss——
返回 `None` 会在 `float(None)` 处直接 `TypeError`。

> 这条不是假想。`dl_model/rnn.py:RNNRegressor._val_one_batch` 以前只记 metrics
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

训练与推理**共用**这一个钩子（`_train_dl:586` 和 `_predict_nn:462` 都调它）。
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

## 简单用法：训练一个已有的模型

`train_model.py` 是仓库里唯一一条真实的端到端路径。骨架是这样的：

```python
from base.config import DLConfig
from config import alpha101_config, alpha158_config, spot_label_config
from dl_model.rnn_classification import RNNClassifier
from factor.alpha101 import Alpha101SpotKline
from factor.alpha158 import Alpha158SpotKline
from label.spot import SpotReturn

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
> 且 `config/__init__.py` 里的路径是另一台机器的绝对路径。上面的代码抄自
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
训练侧一旦改了列顺序，推理侧不跟着改就是静默错位。2026-09-07 把它提成了
`BaseModel.to_tensor(data, variables)`，训练和推理现在共用同一份实现。

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

`disabled` 模式下 `wandb.init` 返回一个 no-op 的 Run 对象，`.log()` / `.finish()`
都能正常调用，不联网、不需要登录、不落盘。要保留本地记录但不上传，用 `WANDB_MODE=offline`。
（下面的实测输出就是在 `WANDB_MODE=disabled` 下跑的。）

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

from base.config import DLConfig
from base.model import BaseModel

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


# ---------------------------------------------------------------- 五方法契约
class TinyRegressor(BaseModel):
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
$ WANDB_MODE=disabled uv run python example/min_model.py

2026-09-07 10:51:17.793 | INFO | base.model:_auto_train:693 - Training DL model: TinyRegressor_total.pth
device: cpu
num_times / num_symbols / num_factors / num_labels = 200 4 3 1
factor names: ['f0', 'f1', 'f2'] label names: ['y0']
TinyRegressor_train: 100%|██████████| 5/5 [00:00<00:00, 88.90it/s]
  epoch 0 train_loss=1.0673
  epoch 0 train_loss=1.1245
  epoch 0   val_loss=1.0695
  epoch 0  test_loss=1.0013
  epoch 0  test_loss=1.1546
  epoch 1 train_loss=1.0064
  epoch 1 train_loss=1.1186
  epoch 1   val_loss=1.0660
  epoch 1  test_loss=0.9912
  epoch 1  test_loss=1.1352
  epoch 2 train_loss=1.0252
  epoch 2 train_loss=1.0306
  epoch 2   val_loss=1.0695
  epoch 2  test_loss=0.9860
  epoch 2  test_loss=1.1045
  epoch 3 train_loss=0.9785
  epoch 3 train_loss=1.0815
  epoch 3   val_loss=1.0759
  epoch 3  test_loss=0.9836
  epoch 3  test_loss=1.0759
  epoch 4 train_loss=1.0135
  epoch 4 train_loss=0.9877
  epoch 4   val_loss=1.0814
  epoch 4  test_loss=0.9835
  epoch 4  test_loss=1.0476
checkpoint dir: /var/folders/.../T/tiny_ckpt_4qksldir
   <save_dir>/TinyRegressor_trial_20260907_105117/TinyRegressor_total/config.json
   <save_dir>/TinyRegressor_trial_20260907_105117/TinyRegressor_total/TinyRegressor_total.pth
```

（数据是纯随机的，loss 在 1.0 附近不下降是**正确的**——特征和标签之间本来就没有关系。
另有一条 `UserWarning: 'pin_memory' ... not supported on MPS` 被略去，
来自 `_train_dl` 里写死的 `pin_memory=True`，在 Mac 上无害。）

注意每个 epoch 里 `train_loss` 打印了 2 行、`test_loss` 打印了 2 行——
这就是「`_train_one_batch` 是 per-batch」的直接证据——改名之前它叫
`_train_one_epoch`，这段输出跟那个名字是直接矛盾的。

---

## 保存与加载

### 存：`.pth` 与 `.joblib` 两条路

`_save_model(p)` 按模型对象的类型自动分流：

```python
if isinstance(self.model, torch.nn.Module):
    torch.save(self.model.state_dict(), p)   # 只存权重，不存结构
else:
    joblib.dump(self.model, p)               # 整个对象序列化
```

torch 那条存的是 `state_dict`——**只有权重，没有结构**。所以加载时必须先把网络重建出来。
非 torch 那条（sklearn / LightGBM 之类）用 joblib 整个 pickle 掉，结构和权重一起走。
`ml_model/backend.py:MlBackend` 是给这条路准备的持久化后端，
但目前**仓库里没有任何地方使用它**（全库 grep 只有定义处一个命中）。

落盘目录是 `{model_save_dir}/{project_name}/{experiment_name}/{model_name}`，
`project_name` 带训练时刻的时间戳。如果目标目录已存在，`_save_model` **直接 `RuntimeError`，
不覆盖**——一次训练的产物不可替代，宁可让人换个名字重来。

### `config.json`：为什么权重旁边必须躺一份配置

`_save_model` 会在权重的同级目录写一个 `config.json`，内容是 `get_config()`：
`DLConfig` 摊平之后，把 `factors` / `labels` 两个字段**就地替换成每个因子/标签自己的配置字典**。

嵌套而不是只记引用，是因为这份 JSON 的目的是「单凭它就能把整条链路复现出来」。
只留个引用，换台机器就复现不了。

上面那次真实训练产出的 `config.json`（节选）：

```json
{
    "factors": [{"name": "FakePanel", "factor_names": ["f0", "f1", "f2"]}],
    "labels":  [{"name": "FakePanel", "factor_names": ["y0"]}],
    "model_save_dir": "/var/folders/.../tiny_ckpt_4qksldir",
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

关键在最后那个 `name`。它由 `BaseModel.import_path` 生成：

```python
f"{self.__class__.__module__}.{self.__class__.__qualname__}"
```

这个字符串就是**重建这个类所需的全部信息**。`utils/module.py` 拿它做反向解析：

```python
def get_cls_from_path(path: str):
    module_path, class_name = path.rsplit(".", 1)      # "dl_model.rnn_classification" + "RNNClassifier"
    module = importlib.import_module(module_path)      # 动态 import 那个模块
    return getattr(module, class_name)                 # 从模块里取出类对象
```

于是 `load_model_from_config(cfg)` 可以递归地把整棵树重建回来：
先把每个 factor 的 `name` 解析成因子类、每个 factor 里的 `dataset.name` 解析成数据集类，
最后把模型类实例化。这是这个项目里**唯一的插件注册机制**——
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

`load()` 做两件事：先按**当下数据的形状**调 `_init_model` 把网络搭出来，再灌权重。
这有个重要的副作用：**必须先 `collect()`**，否则 `self.num_symbols` 会因为
`XrBackend` 还没有数据而抛 `AttributeError: Please cal 'read' or 'to_internal' first.`。

好处是：如果因子数量变了，`load_state_dict` 会当场报形状不匹配，
而不是带着一个错的模型继续跑。

---

## 已知的不完整之处

这些都是 2026-09-07 读代码时逐条核实过的，不美化。

**1. `MLConfig` 分支根本没实现。**
`_auto_train` 里：

```python
elif isinstance(self.config, MLConfig):
    raise NotImplementedError("ML training not implemented")
```

`MLConfig` 这个 dataclass 存在、`MlBackend` 存在，但没有任何一条路把它们接起来。
今天想接 LightGBM 之类，要么自己写一条 `_train_ml`，要么用 `DLConfig` 硬套。

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

**5. `dl_model/mlp.py:MLPRegressor` 曾经是坏的，三处。**（**已于 2026-09-07 修复**）
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
不是训练时那个三维张量。补这个缺口要么改 `base/model.py:_predict_nn`，要么改公开的
`MLP` 模块接受什么，两者都超出了这次的范围。

**6. `dl_model/rnn.py` 里那个 `RNNClassifier` 是一份坏掉的旧副本。**（**已于 2026-09-07 删除**）
`rnn.py` 的 `ModelRBaseCrypto` 最后一层是 `nn.Linear(..., 1)`（回归用），
但它里面的 `RNNClassifier._train_one_batch` 却写了 `primary_pred.reshape(D * T, 2)`
——元素个数对不上，必炸。真正在用的分类器是 `dl_model/rnn_classification.py:RNNClassifier`
（那份的基础块输出 2 类，`self.out = nn.Linear(num_aux * 2, 2)`，逻辑自洽），
`train_model.py` 导入的也是它。

**两个同名类、一个能跑一个不能，本身就是个陷阱**，所以坏的那份删掉了：删除前
grep 确认全仓对 `RNNClassifier` 的引用无一例外解析到 `rnn_classification.py`，
从 `dl_model.rnn` 导入的只有 `RNNRegressor`（`cal.py`、`tests/test_dl_models.py`）。
`ModelRBaseCrypto` / `ModelRCrypto` 留着，`RNNRegressor` 在用。随之清掉了六个
只被那份副本用到的 sklearn 分类指标 import。要找回它 `git show` 即可——从没跑通过
的实现，日后从历史里捞出来比现在维护它便宜。

现在 `dl_model/rnn.py` 里只有 `RNNRegressor`。

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
`BaseModel._get_refit_optim()`，实例级缓存，按 `(self.model 这个对象, lr_refit)`
命中：`load()` 或再次 `_init_model()` 换掉 `self.model` 之后缓存自动失效，
不用任何调用点记得去手动作废——一个指向旧参数张量的陈旧优化器会静默更新一堆
游离张量，比原来的 bug 更糟。锁它的测试断言的是**状态**（两次 `update()` 之后
每个参数的 `state[p]["step"] == 2`，且 `exp_avg` 非零），不是 `id()` 相等：
后者在状态被清空时照样能通过。

它跟 `self.optim` 是两回事——`_train_dl` 结束时 `self.optim = None` 是有意的
（见「常见坑」#8 旁注），微调优化器没有把它复活。

**8. 「从 xarray 到推理张量」没有被封装。**（**已于 2026-09-07 修复**）
以前 `_train_dl` 内联了转换逻辑，推理方要手抄一遍。现在是
`BaseModel.to_tensor(data, variables)`，训练和推理共用；`train_model.py` 已改为调用它。
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

现在转换统一走 `BaseModel.to_tensor(data, variables)`：`sortby` 只排
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
`torch.from_numpy` 忠实继承 numpy 的 dtype，而 `dl_model/` 里每个 `nn.Module`
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

现在 `BaseModel.to_tensor` 在**转换的那一个接缝上**统一 dtype：浮点面板一律转成
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
要走 `load_model_from_config` 的话，模型类必须住在一个真实模块里（比如 `dl_model/xxx.py`）。

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
