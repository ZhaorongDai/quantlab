# 因子层（Factor）

> 代码位置：`base/factor.py`（共享契约 + KunQuant 后端）、`base/factor_polars.py`（Polars 后端）
> 已有因子：`factor/alpha101.py`、`factor/alpha158.py`、`factor/momentum.py`
> 标签：`label/spot.py`　自定义算子：`my_ops/preprocess.py`
> 配置：`base/config.py` 里的 `BaseFactorConfig` / `FactorConfig` / `PolarsFactorConfig`
> 工厂函数：`config/__init__.py`（`alpha101_config` / `alpha158_config` / `momentum_config` / `spot_label_config` …）
> 测试：`tests/test_factor_hierarchy.py`、`tests/test_factor_kunquant.py`、`tests/test_factor_polars.py`、`tests/test_factor_stream.py`

本文所有例子都在本机真跑过（macOS / arm64、Python 3.13、KunQuant 0.1.11、Apple clang 21），贴的是真实输出。

---

## 一句话

因子层拿数据层的 `[timestamp, symbol]` 面板做输入，算出同样形状的因子面板交给模型层；它自己不关心因子是怎么算出来的，所以底下可以挂两套完全不同的计算引擎，而模型层一行都不用改。

---

## 两个后端，为什么并存

类的继承关系是这样的（`tests/test_factor_hierarchy.py` 把它整个钉死了）：

```
Factor（base/factor.py，抽象）
├── FactorKunQuant（base/factor.py）        批量 + 流式，编译成原生代码
│   ├── Alpha101SpotKline / Alpha101Stock
│   ├── Alpha158SpotKline / Alpha158Stock
│   └── SpotReturn / SpotBinaryReturn（标签，label/spot.py）
└── FactorPolars（base/factor_polars.py）    只有批量
    └── Momentum（factor/momentum.py）
```

注意两个后端是**兄弟不是父子**。这不是审美问题，是一个具体的约束：`FactorPolars` 只做批量，如果流式接口留在共同的基类上，它就被迫要实现一个它永远不会实现的东西——要么抛 `NotImplementedError`，要么留一个空方法在那里骗人。所以 `init_stream` / `cal_stream` / `_make` / `_make_stream` / `_get_factor_func` 这些全部只长在 `FactorKunQuant` 上，`tests/test_factor_hierarchy.py::test_streaming_members_stay_on_the_kunquant_subclass` 是这条的自动锁。

### KunQuant

因子写成一张**声明式的算子图**，编译成本地代码再跑。它的两个不可替代的地方：

1. **速度**。Alpha101/Alpha158 这种一次算上百列、每列都是多层滚动窗口套滚动窗口的场景，编译成 SIMD 化的原生代码和用 numpy 逐算子跑不是一个量级。
2. **同一张图能编成流式**。`_make()` 和 `_make_stream()` 把**同一个** `_get_factor_func()` 返回的图编译两次，一次 `input_layout="TS"`（整段历史一次算完），一次 `input_layout="STREAM"`（每来一根 bar 增量推进一步）。这条流式路径现在没有任何生产调用方（`backtest/test_strategy.py` 里两个调用点都被注释掉了，`tests/test_factor_stream.py` 是它在这个仓库里唯一的存活证明），但 CLAUDE.md 明确要"保留未来实时数据接入能力"——将来接实盘时，**因子逻辑一行都不用重写**就能从回测切到实时。这是 Polars 给不了的。

代价：写一个新因子要用 KunQuant 的算子词汇（`op.Div`、`op.WindowedAvg`、`op.BackRef`…）来表达，能不能表达、怎么表达都受算子集限制；每次 `cal()` 都要重新编一次图（见下文）。

### Polars

因子就是一串 Polars 表达式，写起来跟平时做研究一样，`.rolling_mean()`、`.shift().over("symbol")` 张口就来，不用先去查有没有对应的算子。子类只覆写**一个**方法。

代价：**没有流式**，而且这是决定，不是遗漏（`tests/test_factor_polars.py::test_polars_backend_exposes_no_streaming_surface` 断言 `FactorPolars` 和 `Momentum` 身上一个流式成员都不能有）。

### 那条优先级的意思

CLAUDE.md 写的是"能用 xarray/KunQuant 完成的处理，优先不用 Polars"。落到因子层，这句话的实际含义是：

**一个因子将来有没有可能要跑在实时数据上，是选后端时唯一真正重要的问题。** 用 Polars 写的因子，将来要上实时就得整个重写一遍——重写就意味着回测里验证过的那个东西和实盘里跑的那个东西不再是同一份代码，这是量化里最贵的一类 bug。KunQuant 写的因子没有这个断层。

所以实际选法：

- **进入模型、要跟着上实盘的核心因子** → KunQuant。Alpha101/Alpha158 全家都在这边。
- **快速试一个想法、做研究性的探索、或者逻辑用算子图实在别扭** → Polars。跑通了觉得有用，再决定要不要翻译成 KunQuant。
- **KunQuant 算子集表达不了的** → Polars，没得选。

`Momentum`（`factor/momentum.py`）是官方给的 Polars 范例，写一个新的照着它抄就行。

---

## 核心契约

### 输入

因子的输入不是一份数据，是一个**数据集对象**：`BaseFactorConfig.dataset` 直接持有一个 `MarketDataset` 实例。因子在需要数据时自己去问它要：

- KunQuant 走 `dataset.to_kunquant(data_columns)` —— 各个 `Dataset` 子类在这里把自己市场的原始列名改成 KunQuant 认的小写 `open/high/low/close/volume/amount`，并转成 `float32` 的 `[time, symbol]` 连续数组。
- Polars 走 `dataset.read().get_lazyframe()` —— 这条路**不改名**，拿到的是底层 store 里原封不动的列名。

这个差异是 Polars 那边最容易踩的坑，后面单说。

### 输出

统一是 `xr.Dataset`，坐标就是 `timestamp` 和 `symbol`，和输入面板同形状。`tests/test_factor_hierarchy.py::test_public_factor_api_exchanges_only_xarray_datasets` 把这条钉在公共 API 上：`cal` / `read` / `save` / `get_features` / `get_labels` / `get_factor_names` / `get_config` 之间流动的只能是 xarray。Polars 只是 `FactorPolars` 内部的实现手段，结果在离开这个类之前就被 `xr.Dataset.from_dataframe` 转回面板了。

模型层（`base/model.py:_collect_all_features` / `_collect_all_labels`）只调这几个方法，从不检查因子的具体类型——`tests/test_factor_hierarchy.py::test_base_model_does_not_dispatch_on_concrete_factor_types` 会 grep `base/model.py`，出现 `FactorKunQuant` / `FactorPolars` / `isinstance` 就红。这就是"换后端不用改模型层"这句话的兑现方式。

### 一份配置，一个因子实例

`BaseFactorConfig` 里跟两个后端都相关的字段：

| 字段 | 作用 |
|---|---|
| `dataset` | 数据从哪来（持有的是对象，不是路径） |
| `window` | **一词两用**，见「常见坑」第 3 条 |
| `file_path` | 因子面板落盘的 zarr 路径 |
| `start_date` / `end_date` | 要哪一段。不填由 `enums/constant.py:Date` 兜底成 `1900-01-01` ~ `2100-01-01` |
| `symbols` | 要哪些标的，`None` 表示不筛 |
| `factor_names` | 产出哪几列；**留 `None` 是常态**，见下 |
| `kwargs` | 每个因子自己的参数逃生口（`n`、`n_forward_periods` …） |

`FactorConfig`（KunQuant 专属）额外三个：`mode`（`"batch"` / `"stream"`）、`data_columns`（图的输入列名）、`njobs`（执行器线程数，默认 **128**）。`PolarsFactorConfig` 一个都不加。

反过来说：共享基类 `Factor` 的任何方法都**不许读** `mode` / `data_columns` / `njobs`，因为 `PolarsFactorConfig` 上根本没有这些字段，读了就是给兄弟后端埋一个 `AttributeError`。`tests/test_factor_hierarchy.py::test_shared_factor_base_never_reads_config_mode` 用源码 grep 守着。

### 因子名在**构造期**就确定

这是这一层最不直观、也最值得讲的一条设计。

`Factor.config` 的 setter（`base/factor.py`）在赋值的当下就调 `_maybe_resolve_factor_names()`：如果调用方没有显式钉 `factor_names`，就问子类"你会产出哪些列"，当场填进配置里。也就是说，**`Factor(config)` 一构造出来，`get_factor_names()` 和 `num_factors` 就已经有答案了，不需要先 `cal()`**。

为什么要这么早？因为模型层需要它。`base/model.py` 建网络的时候要知道输入层多宽（`num_factors`），而这时候因子可能根本还没算——特别是 `factor_data_strategy="read"` 这条路：因子是从盘上读回来的，从来没在这个进程里算过。如果名字要等 `cal()` 才知道，读回来的因子就报不出自己算的是什么。（`tests/test_factor_hierarchy.py::test_kunquant_and_polars_factors_are_interchangeable_on_the_read_path` 就是这个 bug 的回归锁：以前 Polars 因子走 read 路径时 `factor_names` 是 `None`，`num_factors` 直接炸，而同样的操作 KunQuant 因子没事——"一个后端能用另一个报错"就不叫可互换。）

两个后端回答这个问题的方式不一样：

- **KunQuant**：`_get_factor_names()` 是一个**声明**。`Alpha101SpotKline` 返回 `Alpha101.all_alpha` 全部 82 个名字，`SpotReturn` 返回 `(f"ret_{n}",)`。
- **Polars**：`_get_factor_names()` 是一次**推导**。它拿 `dataset.head(8)` 从 store 里读 8 行（只为了带上真实 dtype），把这 8 行喂给 `_get_factor_lazyframe()`，然后读结果的 schema——**图产出什么列，因子名就是什么**，除掉 `timestamp` / `symbol` 两个索引列。声明和实现不可能对不上，因为压根没有"声明"这回事。

Polars 这条推导有两个直接后果，都是有意的：

1. **构造一个 Polars 因子会碰盘**（约 25–50 ms，且几乎不随 store 大小变化，因为开销主要在开元数据）。所以数据集的 store 必须已经在磁盘上，否则连构造都构造不出来。
2. **配置里的 `n` 变了，名字立刻跟着变**。一个用 `n=5` 算好存下来的 store，被一个现在写着 `n=60` 的配置读回来，因子名是 `momentum_60`，取列的时候当场报错——而不是安安静静地把盘上那份 `momentum_5` 当成 60 日动量喂给模型。

显式钉 `factor_names` 永远优先，推导只在没钉的时候发生。KunQuant 那边经常这么用：`Alpha158SpotKline` 默认要产 169 列，跑测试或试验时钉成 `["KMID", "STD5"]`，编译出来的图就只有这两个 `Output` 可达，编译时间从几十秒掉到 1 秒以内。

### 窗口预热：`_reset_dataset_config()`

因子要 2 月的值，就得比 2 月多读一段历史，不然 2 月头几天的滚动窗口是空的。这件事在 `Factor.config` 的 setter 里自动做掉：

```python
start_date = pd.to_datetime(self._config.start_date)
start_date = start_date - pd.DateOffset(days=self._config.window)
self._config.dataset.config.start_date = start_date.strftime("%Y-%m-%d")
```

然后 `read()` 和 `save()` 之前都会先跑 `_auto_filter()`，把面板收窄回配置真正请求的区间——**预热用的那段历史不会流到模型那里，也不会写进因子 store**。

标的轴不收窄（除非配置里显式给了 `symbols`），因为 xarray 里缺数据的标的是 NaN，`symbol` 坐标轴本身是完整的，没必要再切一刀。

流式模式下 `_auto_filter()` 整个跳过（`FactorKunQuant._auto_filter` 里判了 `mode`）：流式每次手上只有当前这一根 bar，按区间去切它没有意义，还会把唯一的那条切掉。

---

## KunQuant 是怎么跑的

用白话说三步。

**第一步：把因子录成一张图。**

`Builder()` 是一个**录制上下文**。在 `with builder:` 里，你写的每一个 `Input("close")`、`op.Div(a, b)`、`Output(x, "名字")` 都不会当场算任何东西——它们只是往 builder 里记一个节点。`Input` 是图的入口（名字要和 `config.data_columns` 对得上），`Output` 是图的出口（名字就是因子名），中间是算子节点。`with` 块结束后 `Function(builder.ops)` 把这堆节点封成一张完整的图。

这就是为什么 `_get_factor_func()` 的返回类型是 `Function` 而不是一份数据：它交出去的是**做法**，不是结果。

**第二步：把图编译成原生代码。**

`cfake.compileit(...)` 拿这张图，生成 C++，调系统编译器（本机是 Apple clang）编成一个动态库加载进来。`cfake` 里的 `c` 就是 C++——所以**这条路需要机器上有能用的 C++ 编译器**，没有的话因子层的 KunQuant 那半边整个跑不了。

`KunCompilerConfig` 里最重要的是 `input_layout` / `output_layout`：`"TS"` 是批量（整个 `[time, symbol]` 矩阵一次进去），`"STREAM"` 是流式。同一张图编两次，这就是"回测和实盘用同一份因子逻辑"的实现方式。

`_make_stream()` 里那段很长的注释值得看一眼：SIMD 块宽度**故意不写死**，让 KunQuant 按架构自己选（x86_64 上 float 是 8，aarch64 上是 4）。写死成 8 的话，在这个项目自己的开发机（Apple Silicon）上会直接 `RuntimeError: Blocking length 8 is not supported for float on aarch64`。

编译是有成本的。本机实测：两列 Alpha158 大约 0.87 秒，一个只有四五个算子的自定义因子 1.22 秒。全量 Alpha158（169 列）会显著更久。而且 `cal()` 跑完会把 `self._lib = None`，**下一次 `cal()` 重新编一遍**——占的是本地代码的内存，而批量计算通常一个进程里只跑一次，所以是有意用完即弃。试验阶段一定要钉 `factor_names` 把图缩小。

**第三步：多线程跑一遍，再把结果贴回坐标。**

```python
executor = kr.createMultiThreadExecutor(self.config.njobs)   # njobs 默认 128
out_dict = kr.runGraph(executor, modu, input_dict, 0, num_time)
```

`runGraph` 吐出来的是 `{因子名: 二维 float32 数组}`，**只有形状没有含义**——哪一行是哪天、哪一列是哪个标的，全靠调用方记着。所以 `_to_xarray_dataset()` 紧接着就把 `timestamps` 和 `symbols` 贴回去，变成 `xr.Dataset`。裸数组在这一层活不过一个方法。

流式那条路（`init_stream()` / `cal_stream()`）形状一样但状态留在 `StreamContext` 里：`init_stream()` 一次性把每个输入列、每个因子列的缓冲区句柄查好缓存起来（`queryBufferHandle` 按名字现查会拖慢热路径），之后每来一根 bar 就 `pushData` → `run()` → `getCurrentBuffer` 取出这一刻的因子值，不用把历史重算一遍。

---

## 简单用法：算一个已有的因子

先造一份合成的币安现货日线 store（真实用法里换成 `config.spot_kline_config()` 指向你自己的数据即可）：

```python
# make_store.py
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

DEMO = Path("/tmp/quantlab_factor_demo")
STORE = DEMO / "klines.zarr"

def build_store(periods: int = 120, n_symbols: int = 8, seed: int = 0) -> Path:
    symbols = [f"S{i}USDT" for i in range(n_symbols)]
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="D")
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.02, size=(periods, n_symbols))
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    volume = np.abs(rng.normal(1000.0, 100.0, size=(periods, n_symbols)))
    ds = xr.Dataset(
        {
            "Open":  (["timestamp", "symbol"], close * 0.99),
            "High":  (["timestamp", "symbol"], close * 1.02),
            "Low":   (["timestamp", "symbol"], close * 0.98),
            "Close": (["timestamp", "symbol"], close),
            "Volume": (["timestamp", "symbol"], volume),
            "Quote asset volume": (["timestamp", "symbol"], volume * close),
        },
        coords={"timestamp": timestamps, "symbol": symbols},
    )
    STORE.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(STORE, mode="w")
    return STORE
```

（列名是币安 CSV 的原始 Title-Case 名字，`SpotKlineDataset._to_kunquant()` 会在喂给 KunQuant 之前改成小写；`Quote asset volume` 改名成 `amount`。symbol 数取 8，原因见「常见坑」第 1 条。）

然后算 Alpha158 里的两列：

```python
from make_store import DEMO, STORE, build_store
from base.config import DatasetConfig, FactorConfig
from dataset.spot import SpotKlineDataset
from factor.alpha158 import Alpha158SpotKline

build_store()

dataset_config = DatasetConfig(
    zarr_file_path=str(STORE),
    raw_data_dir_path=str(DEMO / "raw"),
    catalog_path=str(DEMO / "catalog"),
    market="crypto_spot",
    frequency="1d",
)

factor = Alpha158SpotKline(
    FactorConfig(
        window=10,
        dataset=SpotKlineDataset(dataset_config),
        mode="batch",
        data_columns=["open", "close", "volume"],
        factor_names=["KMID", "STD5"],          # 钉住，图才小，编译才快
        file_path=str(DEMO / "alpha158_demo.zarr"),
        njobs=4,
    )
)

print("因子名:", factor.get_factor_names())     # 还没 cal() 就已经知道
print("因子列数:", factor.num_factors)

panel = factor.cal().get_features()
print(panel)
```

真实输出：

```
2026-09-07 10:50:30.306 | INFO | utils.timer:__enter__:11 - Starting SpotKlineDataset: to kunquant
2026-09-07 10:50:30.310 | INFO | utils.timer:__exit__:18 - SpotKlineDataset: to kunquant consumed time: 0.00s
2026-09-07 10:50:30.310 | INFO | utils.timer:__enter__:11 - Starting  Alpha158SpotKline: make
2026-09-07 10:50:31.178 | INFO | utils.timer:__exit__:18 -  Alpha158SpotKline: make consumed time: 0.87s
2026-09-07 10:50:31.179 | INFO | utils.timer:__enter__:11 - Starting  Alpha158SpotKline: cal
2026-09-07 10:50:31.179 | INFO | utils.timer:__exit__:18 -  Alpha158SpotKline: cal consumed time: 0.00s
因子名: ['KMID', 'STD5']
因子列数: 2
<xarray.Dataset> Size: 9kB
Dimensions:    (timestamp: 120, symbol: 8)
Coordinates:
  * timestamp  (timestamp) datetime64[ns] 960B 2024-01-01 ... 2024-04-29
  * symbol     (symbol) <U6 192B 'S0USDT' 'S1USDT' ... 'S6USDT' 'S7USDT'
Data variables:
    KMID       (timestamp, symbol) float32 4kB nan nan nan ... -0.3443 0.5548
    STD5       (timestamp, symbol) float32 4kB nan nan nan ... -1.804 -0.8937
```

注意计时器：**编译 0.87 秒，真正算只用了 0.00 秒**。这就是为什么试验时要钉 `factor_names`——你付的绝大部分时间是编译费。

前几行是 NaN，因为 `WindowedZScore(window=10)` 要 10 根才填满窗口：

```
KMID 第 20-22 个时间点 × 前 3 个标的:
symbol        S0USDT    S1USDT    S2USDT
timestamp
2024-01-21 -0.884518  0.117749  0.057778
2024-01-22  0.000000  1.241422  0.596314
2024-01-23  0.219617 -0.732428  1.694411

每列 NaN 数量: {'KMID': 72, 'STD5': 104}
```

72 = 9 行 × 8 标的（z-score 窗口 10 预热掉 9 行）；104 = 13 行 × 8 标的（STD5 自己要 5 根，再叠 10 根 z-score，共 13 行）。

### 顺带说标签

标签类（`label/spot.py`）跟因子共用**同一套机制**，只是 `_get_labels()` 那一半有实现。这里有一个必须理解的细节：

`SpotReturn` 的算子图算的是 `close / BackRef(close, n) - 1`，也就是**过去 n 根的收益**（KunQuant 的图只能往回看，看不到未来）。真正把它变成"未来 n 根的收益"的，是 `_get_labels()` 里那一行 `data.shift(timestamp=-n)`。

```python
label = SpotReturn(FactorConfig(..., data_columns=["close"], kwargs={"n_forward_periods": 3}))
label.cal()
label.get_features()   # 未平移：过去 3 根的收益
label.get_labels()     # 平移后：未来 3 根的收益  ← 模型层用的是这个
```

真实输出（同一份数据，对照看错位）：

```
get_features()（未平移，是过去 3 根的收益）:
symbol        S0USDT    S1USDT
timestamp
2024-01-04 -0.006867 -0.029316
2024-01-05  0.004008  0.006392
2024-01-06 -0.010244  0.043912

get_labels()（向前平移 3 期，成为未来 3 根的收益）:
symbol        S0USDT    S1USDT
timestamp
2024-01-01 -0.006867 -0.029316
2024-01-02  0.004008  0.006392
2024-01-03 -0.010244  0.043912
```

`base/model.py` 走的是 `get_labels()`。**自己写脚本时千万别顺手用 `get_features()` 当标签**——那样训出来的模型是在用过去预测过去。

---

## 扩展一：用 Polars 写一个新因子

一个完整的、能跑的最小子类。因子逻辑：**相对成交量** = 当前成交量 / 过去 n 根的平均成交量 − 1。

```python
# ex2_polars.py（放在仓库根目录，或保证 PYTHONPATH 包含仓库根）
from typing import NoReturn

import polars as pl
import xarray as xr
from make_store import DEMO, STORE, build_store

from base.config import DatasetConfig, PolarsFactorConfig
from base.factor_polars import FactorPolars
from dataset.spot import SpotKlineDataset


class RelativeVolume(FactorPolars):
    """相对成交量：当前成交量 / 过去 n 根 K 线的平均成交量 - 1。"""

    @property
    def window_n(self) -> int:
        # 参数从 config.kwargs 读，不写死在源码里 —— 换个配置就是另一个因子
        return (self.config.kwargs or {}).get("n", 20)

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.window_n
        name = f"rel_volume_{n}"
        volume = pl.col("Volume")          # ← store 里的原始列名，不是小写
        return (
            # 先排序，下面的滚动窗口才是"这个 symbol 自己过去 n 根"
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (volume / volume.rolling_mean(window_size=n).over("symbol") - 1.0)
                .alias(name)
            )
            # 只留 timestamp / symbol / 因子列。少了这行 select，
            # Close、Volume 会被当成因子一起写进 store
            .select(["timestamp", "symbol", name])
        )

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError("RelativeVolume does not support get_labels()")


build_store()

dataset_config = DatasetConfig(
    zarr_file_path=str(STORE),
    raw_data_dir_path=str(DEMO / "raw"),
    catalog_path=str(DEMO / "catalog"),
    market="crypto_spot",
    frequency="1d",
)


def make_config() -> PolarsFactorConfig:
    return PolarsFactorConfig(
        window=10,                                   # 回看天数，见常见坑第 3 条
        dataset=SpotKlineDataset(dataset_config),
        file_path=str(DEMO / "rel_volume.zarr"),
        start_date="2024-02-01",
        end_date="2024-03-01",
        kwargs={"n": 10},
    )


factor = RelativeVolume(make_config())
print("构造完成，因子名已确定:", factor.get_factor_names())

factor.cal().save(mode="w")

# 全新实例，只读盘不算
fresh = RelativeVolume(make_config())
panel = fresh.read().get_features()
print(panel)
print(panel["rel_volume_10"].isel(timestamp=slice(0, 4), symbol=slice(0, 3)).to_pandas())
```

真实输出：

```
2026-09-07 10:51:01.288 | INFO | utils.timer:__enter__:11 - Starting RelativeVolume: cal
2026-09-07 10:51:01.297 | INFO | utils.timer:__exit__:18 - RelativeVolume: cal consumed time: 0.01s
2026-09-07 10:51:01.298 | INFO | utils.timer:__enter__:11 - Starting RelativeVolume: save
2026-09-07 10:51:01.310 | INFO | utils.timer:__exit__:18 - RelativeVolume: save consumed time: 0.01s
构造完成，因子名已确定: ('rel_volume_10',)
<xarray.Dataset> Size: 2kB
Dimensions:        (timestamp: 30, symbol: 8)
Coordinates:
  * timestamp      (timestamp) datetime64[ns] 240B 2024-02-01 ... 2024-03-01
  * symbol         (symbol) StringDType() 128B 'S0USDT' 'S1USDT' ... 'S7USDT'
Data variables:
    rel_volume_10  (timestamp, symbol) float64 2kB ...

symbol        S0USDT    S1USDT    S2USDT
timestamp
2024-02-01  0.033227 -0.112180 -0.067456
2024-02-02  0.090046  0.055531  0.038985
2024-02-03  0.075490 -0.071853 -0.079787
2024-02-04  0.121407 -0.292443 -0.026173
```

几个值得留意的点：

- **`构造完成，因子名已确定` 这行打在 `cal()` 之前**，而且名字里的 `10` 来自 `kwargs["n"]`。这就是前面说的构造期推导。
- **`2024-02-01` 那天就有值，没有 NaN**。因为 `window=10` 让数据集往前多读了 10 天，滚动窗口在请求区间的第一天就已经填满了。
- 没有 `mode` 字段、没有 `data_columns`、没有 `njobs`——`PolarsFactorConfig` 什么都不加。
- 整个 `_get_factor_lazyframe` 里**没有 `.collect()`**。这是硬约束（`tests/test_factor_polars.py::test_get_factor_lazyframe_stays_lazy_until_cal` 会把 `pl.LazyFrame.collect` 替换成抛异常的桩来验证），触发计算的是 `cal()`。
- `_get_factor_lazyframe` 的输入是**参数**，不是自己去取的。所以你可以拿一个手搓的 `pl.LazyFrame` 直接测因子逻辑，不需要 `Dataset`、不需要盘上有 store。

---

## 扩展二：用 KunQuant 写一个新因子

**本环境跑通了。** KunQuant 0.1.11 在 `/Users/daizhaorong/.venv/lib/python3.13/site-packages/KunQuant/`，系统编译器是 Apple clang 21（arm64），`cfake.compileit` 正常工作。

因子逻辑：**收盘价相对 n 日均线的偏离**，再套一层时序 Z 标准化。

```python
# ex3_kunquant.py
from typing import NoReturn

import KunQuant.ops as op
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from make_store import DEMO, STORE, build_store

from base.config import DatasetConfig, FactorConfig
from base.factor import FactorKunQuant
from dataset.spot import SpotKlineDataset
from my_ops.preprocess import WindowedZScore


class MaDeviation(FactorKunQuant):
    """收盘价相对 n 日均线的偏离，再做时序 Z 标准化。"""

    @property
    def n(self) -> int:
        return self.config.kwargs["n"]

    def _get_factor_names(self) -> tuple[str, ...]:
        # 注意：这个方法在 config setter 里就会被调，
        # 此时只能读 config，不能读 self.data_backend（它还没被赋值）
        return (f"ma_dev_{self.config.kwargs['n']}",)

    def _get_factor_func(self) -> Function:
        builder = Builder()
        with builder:
            close = Input("close")                     # 名字要和 data_columns 对上
            dev = op.SubConst(op.Div(close, op.WindowedAvg(close, self.n)), 1.0)
            Output(
                WindowedZScore(dev, self.config.window),
                self._get_factor_names()[0],           # Output 的名字就是因子名
            )
        return Function(builder.ops)

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError("MaDeviation does not support get_labels()")


build_store()

dataset_config = DatasetConfig(
    zarr_file_path=str(STORE),
    raw_data_dir_path=str(DEMO / "raw"),
    catalog_path=str(DEMO / "catalog"),
    market="crypto_spot",
    frequency="1d",
)

factor = MaDeviation(
    FactorConfig(
        window=20,
        dataset=SpotKlineDataset(dataset_config),
        mode="batch",
        data_columns=["close"],           # 图里只有 Input("close")，就只给 close
        file_path=str(DEMO / "ma_dev.zarr"),
        kwargs={"n": 5},
        start_date="2024-03-01",
        end_date="2024-03-31",
        njobs=4,
    )
)

print("因子名:", factor.get_factor_names())
factor.cal().save(mode="w")
panel = factor.read().get_features()
print(panel)
```

真实输出：

```
2026-09-07 10:51:28.928 | INFO | utils.timer:__enter__:11 - Starting SpotKlineDataset: to kunquant
2026-09-07 10:51:28.931 | INFO | utils.timer:__exit__:18 - SpotKlineDataset: to kunquant consumed time: 0.00s
2026-09-07 10:51:28.931 | INFO | utils.timer:__enter__:11 - Starting  MaDeviation: make
2026-09-07 10:51:30.152 | INFO | utils.timer:__exit__:18 -  MaDeviation: make consumed time: 1.22s
2026-09-07 10:51:30.152 | INFO | utils.timer:__enter__:11 - Starting  MaDeviation: cal
2026-09-07 10:51:30.153 | INFO | utils.timer:__exit__:18 -  MaDeviation: cal consumed time: 0.00s
因子名: ('ma_dev_5',)
<xarray.Dataset> Size: 1kB
Dimensions:    (timestamp: 31, symbol: 8)
Coordinates:
  * timestamp  (timestamp) datetime64[ns] 248B 2024-03-01 ... 2024-03-31
  * symbol     (symbol) <U6 192B 'S0USDT' 'S1USDT' ... 'S6USDT' 'S7USDT'
Data variables:
    ma_dev_5   (timestamp, symbol) float32 992B nan nan nan ... -1.143 1.273

symbol        S0USDT    S1USDT    S2USDT
timestamp
2024-03-01       NaN       NaN       NaN
2024-03-02       NaN       NaN       NaN
2024-03-03       NaN       NaN       NaN
2024-03-04  1.137148  0.137248 -1.189091
```

**这三行 NaN 不是随机的，正好是「常见坑」第 3 条的现场。** 同一次运行打出来的：

```
因子请求区间: 2024-03-01 -> 2024-03-31
数据集实际取数区间: 2024-02-10 -> 2024-03-31
实际算过的时间点数: 51
NaN 数量: 24 = 3 行 x 8 标的
```

`window=20` 让数据集回退到 2024-02-10，多给了 20 根。但这个因子真正需要的预热是 **5（均线）+ 20（z-score）= 24 根**。24 > 20，所以请求区间开头的 3 天填不满，出 NaN。想干净就把 `window` 设大，或者理解成"`window` 是给数据集回看用的天数，不等于因子的窗口需求"。

**要写一个新的 KunQuant 因子，你只需要实现 `_get_factor_func()` 和 `_get_factor_names()`**（`_get_features` / `_get_labels` 按需要挑一半实现，另一半让基类报错即可）。批量、流式、编译、落盘、读回、区间收窄全部继承。

---

## 标准化算子与截面 vs 时序

这一节最重要，因为它看起来像 bug，其实是设计。

### `WindowedZScore` 干的是什么

`my_ops/preprocess.py`：

```python
rolling_mean = WindowedAvg(self.inputs[0], window)
rolling_std  = WindowedStddev(self.inputs[0], window)
z_score = Div(Sub(self.inputs[0], rolling_mean), rolling_std)
```

对每一个标的，拿它**自己过去 window 根**的均值和标准差来标准化自己。这是**时序（time-series）标准化**：横向（同一时刻的不同标的之间）它什么都没做。

同文件里曾经还有一个 `WindowedRobustStandardization`（中位数 + MAD，对异常值更稳），但全仓没有任何地方用它——写好了备着，从没被谁调用过。**2026-09-07 已删除**：一个从未被执行过的 op 摆在这里，读的人会以为它是个可用的入口，而它连一次编译都没进过。真要用鲁棒标准化，`git show` 把它捞回来比信任一份没验证过的实现便宜。

### 现状的四格矩阵

| 因子类 | 市场 | 标准化 |
|---|---|---|
| `Alpha101SpotKline` | 加密现货 | `WindowedZScore`（时序） |
| `Alpha158SpotKline` | 加密现货 | `WindowedZScore`（时序） |
| `Alpha101Stock` | 美股 | **无，输出原始值** |
| `Alpha158Stock` | 美股 | **无，输出原始值** |

`tests/test_factor_kunquant.py::test_normalization_matrix_matches_recorded_strategy_types` 用源码内省把这四格锁住了，改一格测试就红，报错信息直接把理由写在里面。

### 为什么不一样

**因为这两个市场在这个项目里跑的是不同类型的策略。**

- **加密现货 → 时序策略**：判断的是"这个币现在相对它自己最近的状态是贵还是便宜"。它要的就是拿自己的历史当基准，`WindowedZScore` 正是这个。
- **美股 → 截面策略**：判断的是"在今天这一批股票里，哪些更好"。它要的是**在每个时间截面上、跨标的**做标准化——把今天所有股票的这个因子值放在一起排序/标准化。

**时序标准化对截面策略不但没用，还会破坏信息。** 一个股票的因子值除以它自己的历史波动之后，同一天不同股票之间的可比性就变了——原本"A 比 B 高"可能因为两者历史波动不同而翻过来。所以美股这两个类**故意输出原始值**，把截面标准化留给下游消费方去做。

反过来说，如果有人为了"统一风格"给 `Alpha101Stock` 加上 `WindowedZScore`，那是往一套截面因子上硬套了一个时序标准化——代码不会报错，回测也照跑，只是结果不对。**这是本仓库明确记录过的一类"别对齐它"。**

### 现在还没有的东西

**仓库里没有截面 Z-score 算子。** `my_ops/preprocess.py` 里两个都是 `WindowedCompositiveOp`（时序）。真正的截面标准化算子被推迟到后续阶段（对应 `.planning/` 里的 ARCH-01/ARCH-02，目标是"架构同时兼容单标的时序策略与多标的截面多因子策略"）。现阶段美股这条路的约定是：**因子层出原始值，截面标准化由消费方自己做。**

如果你现在就需要，在 xarray 层做是最直接的（因子面板已经是 `[timestamp, symbol]`，截面就是沿 `symbol` 维）：

```python
panel = factor.cal().get_features()
cs_z = (panel - panel.mean(dim="symbol")) / panel.std(dim="symbol")
```

（这段是示意，不是仓库里现有的代码。）

---

## 常见坑

以下每一条都在本机复现过，贴的是真实报错。

### 1. KunQuant 批量模式下，symbol 数必须是 SIMD 块宽度的整数倍

8 个标的能跑，3 个不行：

```
坑A: RuntimeError Bad shape at open
```

`TS` layout 编出来的代码按 SIMD 块处理标的轴（x86_64 上 float 是 8，aarch64 是 4）。`tests/test_factor_kunquant.py` 里美股那几个测试专门凑了 8 个 ticker（`_STOCK_SYMBOLS`），就是为了绕这个。报错信息 `Bad shape at open` 完全不提标的数量，第一次撞上会查很久。

### 2. 流式模式下，`data_columns` 不能比图实际消费的宽

KunQuant 会把**没有任何可达 `Output` 用到的输入节点剪掉**。`init_stream()` 却对 `config.data_columns` 里每一个名字都去 `queryBufferHandle`，剪掉的那个查不到：

```
坑C: RuntimeError Cannot find the buffer name
```

**批量模式没有这个问题**，多给输入是无害的：

```
坑B: batch 模式多给输入没问题, 结果 {'timestamp': 60, 'symbol': 8}
```

所以这个坑只在你钉了一小撮 `factor_names` 又去跑流式的时候出现（生产配置用全量因子集，六个输入全都被消费，撞不上）。

### 3. `window` 一词两用

同一个 `config.window` 字段被用在两个完全不同的地方：

1. 因子代码里当**滚动窗口长度**（`WindowedZScore(alpha, self.config.window)`，单位是 **bar 数**）；
2. `_reset_dataset_config()` 里当**数据集回看天数**（`pd.DateOffset(days=self._config.window)`，单位是**日历天**）。

两个后果：

- **回看量可能不够**。上面 `MaDeviation` 的例子：`window=20` 只回看 20 天，而因子真正需要 5+20=24 根，请求区间开头 3 天出 NaN，没有任何警告。
- **有休市的市场上，天数换不来同样多的 bar**。加密现货一周七天，20 个日历天正好 20 根日线，所以上面那个例子的算术很干净。美股不是：20 个日历天大约只有 14 根日线，`window=20` 的滚动窗口铁定填不满。日内频率则相反——1 分钟线上 `window=128` 意味着回看 128 **天**，远远超出 128 根 bar 的需求，只是白读数据。总之写因子时自己按频率算一遍这个数，别按直觉设。

### 4. `save()` 默认的 `mode="a"` 不是"按时间追加"

`Factor.save()` 默认 `mode="a"`，直接透传给 `to_zarr`。zarr 的 `"a"` 是"改写已有 store 里的变量"，**不是**沿时间轴追加。所以先存 2 月、再存 3 月：

```
第一次 save 后: {'timestamp': 29, 'symbol': 8}
第二次 save 报错: ValueError variable 'timestamp' already exists with different
dimension sizes: {'timestamp': 29} != {'timestamp': 31}. to_zarr() only supports
changing dimension sizes when explicitly appending, but append_dim=None
```

结论：**因子落盘基本都该用 `save(mode="w")`**。真的要增量追加，得走 `XrBackend.append()` / `widen_and_append()`（`dataset/backend.py`），那边有坐标一致性和 dtype 的检查，而 `Factor.save()` 现在没接过去。

**2026-09-07 起这条报错自己会说该怎么办**（上面那段 zarr 原文现在只是 `__cause__`）：

```
ValueError: PanelFactor.save(mode="a"): cannot write this date range into the
existing store at /tmp/.../alpha.zarr. zarr's "a" means "overwrite variables in
an existing store", NOT "append along time", so a second, differently-sized
date range is rejected. Use save(mode="w") to replace the store, or delete it
first. True incremental appends go through XrBackend.append(), which
Factor.save() is not wired to. Original error: ...
```

**默认值没有改**，仍然是 `mode="a"`——改默认值对任何依赖它的调用方都是行为变更，而这里真正的伤害是「错误信息隔着两层看不懂」，不是「默认值错了」。由 `tests/test_factor_save_mode.py` 锁（包括「默认值仍是 `a`」这一条，以及「别的 `ValueError` 不能被顺手改写成这句话」）。

### 5. Polars 因子写的是 store 的**原始列名**，跨市场不可移植

`Dataset.get_lazyframe()` **不改名**，给什么就是什么。`Momentum` 写的是 `pl.col("Close")`（币安 Title-Case），拿去配美股的 store：

```
美股 store 的列名 = ['timestamp', 'symbol', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen',
 'adjVolume', 'anomaly_flag', 'close', 'divCash', 'high', 'low', 'open',
 'splitFactor', 'volume']
坑D: ColumnNotFoundError unable to find column "Close"; valid columns: [...]
```

KunQuant 那边没这个问题，因为每个 `Dataset` 子类都覆写了 `_to_kunquant()` 把自己的列改成统一的小写名字。给 `get_lazyframe()` 也定一个市场无关的命名契约是已知的待办（03-RESEARCH.md Open Question 2），现在还没做。

注意这个错**发生在构造期**（`RelativeVolume(config)` 这一行），不是 `cal()`——因为因子名推导要拿 8 行样本跑一遍图。

### 6. Polars 因子构造时就要碰盘

承上：`_get_factor_names()` 走 `dataset.head(8)`，**直接按路径打开 store**。所以

- 数据集的 zarr 必须**已经存在**，否则连因子对象都构造不出来（这是已记录的收窄，编号 RV-02，故意没有关掉）；
- 这个探查**绝对不能改成走 `read()`**。`BaseDataset.read()` 会跑 `_filter()` 原地收窄数据，而 `XrBackend.read()` 有缓存早返回，收窄会一直留到 `cal()`——而探查发生在 `_reset_dataset_config()` 拓宽窗口**之前**，`filter_by_date` 又只会缩不会扩，结果就是**因子的整段回看被静默丢掉**，因子列前 n 行全是 NaN 而没有任何地方报错。这就是编号 RV-01 的事故，`tests/test_factor_polars.py::test_a_dated_dataset_config_keeps_the_factor_lookback_window` 是它的回归锁。`base/factor_polars.py` 的类文档里写着 "Do not put `.read()` back in front of it."

### 7. `_get_factor_lazyframe` 忘了最后那个 `.select(...)`

返回的 lazyframe 里**除 `timestamp` / `symbol` 之外的每一列都会被当成因子**——名字是它，落盘的是它，喂给模型的也是它。忘了 select，`Close`、`Volume` 就会作为"因子"被持久化。`tests/test_factor_polars.py::test_momentum_cal_returns_xarray_dataset_with_only_factor_columns` 断言 `data_vars == ["momentum_5"]` 就是在守这一条。

### 8. 标签别用 `get_features()`

见前面「顺带说标签」。`get_features()` 是**过去** n 期收益，`get_labels()` 才是平移后的**未来** n 期收益。`base/model.py` 用的是后者。

### 9. 因子类调 `get_labels()` 会抛 `RuntimeError`，反之亦然

`Factor` 基类的 `_get_features` / `_get_labels` 默认都是 `raise NotImplementedError`，具体类只实现自己那一半：因子类（Alpha101/158/Momentum）的 `_get_labels` 显式 `raise RuntimeError(...)`；标签类（`SpotReturn` / `SpotBinaryReturn`）两个都实现了。**基类宁可报错也不返回一份空面板**——空面板会一路往下游流，错误要到很后面才现形。

### 10. `_get_factor_names()` 里不能碰 `self.data_backend`

`Factor.__init__` 的顺序是先 `self.config = config`（触发 setter，setter 里就会调 `_get_factor_names()`），**再** `self.data_backend = XrBackend()`。所以名字解析跑的时候，因子自己的存储后端还不存在。`tests/test_factor_hierarchy.py::test_factor_init_assigns_config_before_the_storage_backend` 锁着这个顺序——它是有承载力的，别"整理"成先建 backend。

（数据集的 backend 是另一个对象，通过 `self.config.dataset` 拿，那个是存在的，`FactorPolars` 的探查就是这么做的。）

### 11. `WindowedZScore` 的 docstring 说了它没做的事（**已于 2026-09-07 修复**）

docstring 曾经写"先将缺失值替换为 0，然后进行滚动标准化"，但 `decompose()` 里只有 `WindowedAvg` / `WindowedStddev` / `Sub` / `Div`，**没有任何缺失值替换**。

改的是**文字**不是代码，这是有意的：补一个 `fillna(0)` 会改变这个 op 产出的每一个因子值——而 `Alpha101SpotKline` / `Alpha158SpotKline` 的每一个 `Output(...)` 都裹着它——也就是让已经落盘的每一份因子、以及依赖它们训出来的每一个模型，全部对不上。反过来，那句描述所说的行为一天都没有存在过，所以没有任何调用方能真的依赖它。现在的 docstring 明说「不做任何缺失值处理」，并指出窗口未填满时的前 window-1 根 NaN 是滚动 op 的正常行为，不是缺失值处理的替代品。要填充语义请在调用方显式做。

### 12. 每次 `cal()` 都重新编译一次图

`FactorKunQuant.cal()` 末尾把 `self._lib = None`。同一个实例连着 `cal()` 两次，就编两次。这是有意的（编译产物占的是本地代码内存，批量通常一个进程只跑一次），但如果你在 notebook 里反复调 `cal()` 调试，那一两秒是每次都要付的。想省就钉 `factor_names` 把图缩到最小。
