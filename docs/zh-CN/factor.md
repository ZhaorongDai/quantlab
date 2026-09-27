# 因子（Factors）

[English](../factor.md) | 简体中文

因子把数据集持有的 `(timestamp, symbol)` 面板转换成同样形状的特征面板；标签（label）则是输出为预测目标的因子。quantlab 有两个因子后端：`FactorKunQuant` 把 KunQuant 算子图编译成本地代码，支持批量和流式两种运行方式；`FactorPolars` 的逻辑是一条 Polars 表达式链，只支持批量。两者共用基类 `Factor`，所以模型层对它们一视同仁。

## 前置条件

KunQuant 因子在运行时编译 C++，需要可用的 C++ 编译器；Polars 因子没有这个要求。下面的例子都在工作目录下用合成数据运行。加密现货数据集存的是 Binance 原始列名（`Close`、`Volume`）；数据集配置的各个字段见数据集指南 `dataset.md`。

## 基础

### 两个后端

| | `FactorKunQuant` | `FactorPolars` |
|---|---|---|
| 逻辑写在 | `_get_factor_func` 中的 KunQuant 计算图 | `_get_factor_lazyframe` 中的 Polars 表达式链 |
| 运行方式 | 批量和流式 | 仅批量 |
| 配置类 | `FactorConfig` | `PolarsFactorConfig` |
| 输入列名 | 共享列名：存储自己的列名，数据集定义了 `COLUMN_MAP` 时按它改名（现货存储的 `Close` 变成 `close`） | 存储中的原始列名（`Close`）；合并输入时为共享列名 |
| 输出 dtype | float32 | float64 |
| 开销 | 每次 `compute()` 都要编译计算图 | 无 |

KunQuant 是主后端：现有的 alpha 因子库用到的滚动和截面算子它都有，并且只有它能在实盘数据上逐 bar 运行。`FactorPolars` 是新因子的补充路径，适合用 DataFrame 表达式更容易描述的因子，或者需要 KunQuant 没有的运算。需要同时支持流式运行的因子必须写成 KunQuant 因子。

### 配置

两个后端使用的配置字段如下。`FactorConfig` 另外有 `mode`（`"batch"` 或 `"stream"`）、`data_columns`（输入计算图的数据集变量）和 `njobs`（执行器线程数）；`PolarsFactorConfig` 没有额外字段。

| 字段 | 含义 |
|---|---|
| `warmup_bars` | 在请求的起点之前读取的历史 bar 数，用来让滚动算子预热；在数据集自己的日历上数 |
| `dataset` | 因子读取的数据集，或由它合并的数据集列表（见下文） |
| `file_path` | `build` 写入、`read` 读取的 Zarr 存储 |
| `factor_names` | 输出列名；为 `None` 时由因子自己推出 |
| `kwargs` | 因子类自行读取的自由选项 |

配置是冻结的。因子持有的是归一化后的副本，其中 `name` 和 `factor_names` 已填好；你传入的配置从不被改动，保存下来的 `config.json` 能重建出配置相等的因子。

配置只说明算什么，不说明算哪段时间。日期区间是 `compute(start, end)`、`build(start, end)` 和 `read(start, end)` 的参数，这些调用都不会改动因子自己或其数据集的配置。

### 计算一个因子

下面的会话先写入一个小的合成现货存储，并定义一个基于它构造数据集的辅助函数。`Momentum` 是 Polars 后端的参考因子：`Close_t / Close_{t-n} - 1`，`n` 从 `kwargs` 读取。因子名在对象构造完成时就已确定，不需要先计算。

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.backend import XrBackend
>>> from quantlab.base.config import DatasetConfig, PolarsFactorConfig
>>> from quantlab.dataset.spot import SpotKlineDataset
>>> from quantlab.factor.momentum import Momentum
>>> rng = np.random.default_rng(0)
>>> symbols = [f"S{i}USDT" for i in range(8)]
>>> close = 100 + np.cumsum(rng.normal(size=(90, 8)), axis=0)
>>> scale = {"Open": 0.99, "High": 1.02, "Low": 0.98, "Close": 1.0, "Volume": 10.0, "Quote asset volume": 1000.0}
>>> raw = xr.Dataset(
...     {name: (["timestamp", "symbol"], close * k) for name, k in scale.items()},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=90), "symbol": symbols},
... )
>>> XrBackend().to_internal(raw).write("data/klines.zarr")
XrBackend()
>>> def make_dataset():
...     return SpotKlineDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/klines.zarr",
...         market="crypto_spot", frequency="1d",
...     ))
...
```

`compute(start, end)` 从 `start` 之前 `warmup_bars` 根 bar 开始读取数据集（在数据集自己的日历上数，没有数据的日子会被跳过），计算后只返回 `start` 到 `end`。结果与对全部历史计算的值一致。`get_features(panel)` 把返回的面板转换成模型特征，即以 `(timestamp, symbol)` 为索引的 `xarray.Dataset`。

```python
>>> factor = Momentum(PolarsFactorConfig(
...     warmup_bars=20, dataset=make_dataset(), file_path="data/factors/momentum.zarr",
...     kwargs={"n": 5},
... ))
>>> factor.get_factor_names(), factor.warmup_bars
(('momentum_5',), 20)
>>> panel = factor.get_features(factor.compute("2024-02-01", "2024-02-29"))
>>> dict(panel.sizes), list(panel.data_vars)
({'timestamp': 29, 'symbol': 8}, ['momentum_5'])
>>> panel["momentum_5"].isel(timestamp=0, symbol=slice(0, 3)).values.round(4)
array([ 0.0161,  0.0079, -0.0007])
```

当数据集在 `start` 之前不足 `warmup_bars` 根 bar 时，会发出一条 `UserWarning` 说明差多少根，计算从现有的第一根 bar 开始：

```text
UserWarning: Momentum.compute(): 20 warm-up bar(s) are needed before '2024-01-05' but SpotKlineDataset holds only 4; the first bars are short by 16 bar(s) of warm-up.
```

`compute` 之后因子不持有面板；每次调用都经 `dataset.panel` 重新读取数据集。

## 常见任务

### 构建、读取和扩充因子存储

`build(start, end)` 把 `compute(start, end)` 写成 `store_path` 处的存储（替换原有内容），并把区间记录在它旁边的 `<store>.range.json` 中。`read(start, end)` 惰性地从存储返回一个区间；记录的区间不包含所请求的区间时拒绝。`extend(end)` 计算记录终点之后的 bar（从数据集取预热历史），追加到存储，并把记录的终点后移。它调用 `XrBackend.widen_and_append`，因此 timestamp、symbol 和变量三个轴会按需加宽，后端指南中描述的追加检查同样适用。

```python
>>> factor.build("2024-01-21", "2024-02-29").store_range()
('2024-01-21', '2024-02-29')
>>> dict(factor.read("2024-02-10", "2024-02-15").sizes)
{'timestamp': 6, 'symbol': 8}
>>> factor.read("2024-02-10", "2024-03-10")
Traceback (most recent call last):
ValueError: Momentum.read(): the store at data/factors/momentum.zarr covers 2024-01-21 to 2024-02-29, which does not contain 2024-02-10 to 2024-03-10. Extend it with extend(end) or rebuild it with build(start, end).
>>> factor.extend("2024-03-20").store_range()
('2024-01-21', '2024-03-20')
>>> factor.read("2024-01-21", "2024-03-20").sizes["timestamp"]
60
>>> stored = xr.open_zarr("data/factors/momentum.zarr")
>>> stored.sizes["timestamp"], str(stored["timestamp"].values[-1])[:10]
(60, '2024-03-20')
```

不是由 `build` 写出的存储，`store_range()` 返回 `None`，`read` 和 `extend` 都会拒绝这样的存储。KunQuant 因子只在 batch 模式下应答这些调用。

### 从配置重建因子

`get_config()` 返回描述因子及其数据集的 dict，`load_factor_from_config` 可以据此重建因子。

```python
>>> cfg = factor.get_config()
>>> cfg["name"], cfg["kwargs"], cfg["dataset"]["market"]
('quantlab.factor.momentum.Momentum', {'n': 5}, 'crypto_spot')
>>> from quantlab.utils.module import load_factor_from_config
>>> rebuilt = load_factor_from_config(cfg)
>>> type(rebuilt).__name__, rebuilt.get_factor_names()
('Momentum', ('momentum_5',))
```

### 把多个数据集合并成一个输入

`dataset` 也接受一个数据集列表。因子把它们合并成一个 `MergedDataset`：每个输入先按自己的 `COLUMN_MAP` 改成共享变量名（现货存储的 `Close` 变成 `close`；股票存储保持原名），然后在 timestamp 和 symbol 上做外连接，某个输入没有值的格子为 NaN。这覆盖了指数存储加 ETF 存储（变量相同、标的不同）和价格加报价（标的相同、变量不同）两种情况，也可以跨数据集类合并。预热在各输入日历的并集上数。下面的例子接着上文的会话：把现货存储拆成四个现货标的和四个用共享名存储的标的，一个 Polars 因子从两者读取共享列名。

```python
>>> import polars as pl
>>> from quantlab.base.factor import FactorPolars
>>> from quantlab.dataset.merged import MergedDataset
>>> from quantlab.dataset.stock import StockDataset
>>> XrBackend().to_internal(raw.sel(symbol=symbols[:4])).write("data/spot_half.zarr")
XrBackend()
>>> XrBackend().to_internal(
...     raw.sel(symbol=symbols[4:]).rename(SpotKlineDataset.COLUMN_MAP)
... ).write("data/stock_half.zarr")
XrBackend()
>>> spot_half = SpotKlineDataset(DatasetConfig(
...     raw_data_dir_path="data/raw", zarr_file_path="data/spot_half.zarr",
...     market="crypto_spot", frequency="1d",
... ))
>>> stock_half = StockDataset(DatasetConfig(
...     raw_data_dir_path="data/raw", zarr_file_path="data/stock_half.zarr",
...     market="us_equity", frequency="1d",
... ))
>>> panel = MergedDataset([spot_half, stock_half]).panel("2024-02-01", "2024-02-29")
>>> dict(panel.sizes), sorted(panel.data_vars)
({'timestamp': 29, 'symbol': 8}, ['amount', 'close', 'high', 'low', 'open', 'volume'])
>>> class Range(FactorPolars):
...     def _get_factor_lazyframe(self, lf):
...         return lf.with_columns(
...             ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range")
...         ).select(["timestamp", "symbol", "range"])
...     def _get_features(self, data):
...         return data
...
>>> factor_range = Range(PolarsFactorConfig(
...     warmup_bars=0, dataset=[spot_half, stock_half], file_path="data/factors/range.zarr",
... ))
>>> type(factor_range.config.dataset).__name__
'MergedDataset'
>>> dict(factor_range.compute("2024-02-01", "2024-02-29").sizes)
{'timestamp': 29, 'symbol': 8}
>>> cfg = factor_range.get_config()
>>> cfg["dataset"]["name"], [d["zarr_file_path"] for d in cfg["dataset"]["datasets"]]
('quantlab.dataset.merged.MergedDataset', ['data/spot_half.zarr', 'data/stock_half.zarr'])
>>> load_factor_from_config(cfg) == factor_range
True
```

合并从不按输入顺序取值。同一个格子在两个输入里都有值时抛出 `ValueError: MergedDataset: variable 'close' holds a value in both SpotKlineDataset(data/spot_half.zarr) and SpotKlineDataset(data/overlap.zarr), for example at symbol 'S3USDT' on 2024-02-01 00:00:00. ...`；bar 间隔不同的输入抛出 `ValueError: MergedDataset: the inputs have different bar spacing (SpotKlineDataset(data/spot_half.zarr): 1 days 00:00:00, SpotKlineDataset(data/hourly.zarr): 0 days 01:00:00). ...`。合并输入上的 KunQuant 因子在 `data_columns` 里写共享列名，Polars 因子在表达式里也用共享列名（`close` 而不是 `Close`）。流式模式在构造因子时就拒绝合并输入：`ValueError: MaDeviation: stream mode takes one dataset, got a merge of 2. ...`。`MergedDataset` 本身就是数据集，有 `panel` 和 `bar_before`；它没有自己的存储，所以 `store_path`、`save`、`resample` 和构建路径都会拒绝。要先对各输入重采样再合并；合并输入上的因子本身可以重采样，前提是各输入切 bar 的方式相同。合并数据集目前还不能作为回测的 `price_dataset` 或 `benchmark_dataset`，它们需要存储路径。

### 把因子重采样到更粗的 bar

`resample(freq, how)` 返回因子的一个副本，其 `compute`、`read` 和 `build` 在更粗的 bar 上应答。因子仍然在其 dataset 自身的 bar 上计算，只有输出被聚合，因此分钟 bar 上的动量会变成"每日最后一分钟的值"这一日频序列，而它衡量的东西没有变。`freq` 和 `how` 的取值与 `BaseDataset.resample` 相同（见 dataset 指南），`how` 也可以只给一个字符串，表示所有因子变量都用这种方法。bar 的切分方式沿用因子所用 dataset 的切分方式。

下面的会话在 dataset 指南里构造的两天分钟 store 上运行 `Momentum`（`config` 就是那个 store 的 `DatasetConfig`）。该 store 从第一根请求的 bar 开始，所以 `compute` 还会警告缺少那一根预热 bar。

```python
>>> factor = Momentum(PolarsFactorConfig(
...     warmup_bars=1, dataset=SpotKlineDataset(config),
...     file_path="data/factors/momentum.zarr", kwargs={"n": 1},
... ))
>>> factor.compute("2024-01-02", "2024-01-03")["momentum_1"].to_pandas().round(3)
symbol               AAAUSDT  BBBUSDT
timestamp                            
2024-01-02 00:00:00      NaN      NaN
2024-01-02 00:01:00    1.000    1.000
2024-01-02 00:02:00    0.500    0.500
2024-01-02 00:03:00    0.333    0.333
2024-01-03 00:00:00    0.250    0.250
2024-01-03 00:01:00    0.200    0.200
2024-01-03 00:02:00    0.167    0.167
2024-01-03 00:03:00    0.143    0.143
>>> daily = factor.resample("1d", "last")
>>> daily.compute("2024-01-02", "2024-01-03")["momentum_1"].to_pandas().round(3)
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02    0.333    0.333
2024-01-03    0.143    0.143
>>> daily.config.dataset.config.resample_freq, daily.config.resample_freq
(None, '1d')
```

副本有自己的 dataset 对象和空的编译状态。`build()` 写到 `store_path`，即源 store 旁边的那个 store；副本上的 `read()` 在该 store 存在时直接打开它，否则对源因子的 store 做重采样。这次请求能经 `get_config()` 和 `load_factor_from_config` 往返重建。

```python
>>> daily.store_path
'data/factors/momentum_resample_1d.zarr'
>>> daily.build("2024-01-02", "2024-01-03").store_range()
('2024-01-02', '2024-01-03')
>>> daily.read("2024-01-02", "2024-01-03").sizes
Frozen({'timestamp': 2, 'symbol': 2})
>>> cfg = daily.get_config()
>>> cfg["resample_freq"], cfg["resample_how"], cfg["dataset"]["resample_freq"]
('1d', 'last', None)
>>> load_factor_from_config(cfg).compute("2024-01-02", "2024-01-03").sizes
Frozen({'symbol': 2, 'timestamp': 2})
```

如果想在已经重采样的 bar 上计算因子，先对 dataset 做重采样，再把重采样后的 dataset 交给因子。

### 计算标签

标签是 `get_labels()` 返回前瞻值的 KunQuant 因子。`quantlab.label.fret` 中的 `Return` 是从下一根 bar 的复权开盘价到其后 `n` 根 bar 的复权开盘价的收益，`BinaryReturn` 在该收益为正时取 1.0。计算图算出的是滞后收益（KunQuant 只能向后看），`get_labels()` 再把它向前平移 `n_forward_periods + 1` 根 bar，因此最后 `n_forward_periods + 1` 根 bar 是 NaN。标签读取 `adjOpen`，所以数据集必须带复权价格：美股数据集有，加密现货数据集没有。下面的会话从存储的第一根 bar 开始，所以 `compute` 会警告缺少 5 根预热 bar；`ret_2` 只需要之前 2 根 bar，滞后收益的前 2 根 bar 在平移前是 NaN。

```python
>>> from quantlab.base.config import FactorConfig
>>> from quantlab.dataset.stock import StockDataset
>>> from quantlab.label.fret import Return
>>> px = 50 + np.cumsum(rng.normal(size=(30, 8)), axis=0)
>>> stock = xr.Dataset(
...     {"adjOpen": (["timestamp", "symbol"], px)},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=30), "symbol": [f"T{i}" for i in range(8)]},
... )
>>> XrBackend().to_internal(stock).write("data/stock.zarr")
XrBackend()
>>> label = Return(FactorConfig(
...     warmup_bars=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 2},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/stock.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     file_path="data/labels/ret.zarr", njobs=2,
... ))
>>> label.get_factor_names()
('ret_2',)
>>> ret = label.get_labels(label.compute("2024-01-01", "2024-01-30"))["ret_2"]
>>> dict(ret.sizes)
{'timestamp': 30, 'symbol': 8}
>>> ret.isel(symbol=0).values[:2].round(5)
array([-0.03945,  0.01532], dtype=float32)
>>> round(float(px[3, 0] / px[1, 0] - 1), 5)
-0.03945
>>> ret.isel(symbol=0).values[-3:]
array([nan, nan, nan], dtype=float32)
```

`get_labels(panel)` 在传入的面板内部平移，因此任何请求区间的最后 `n_forward_periods + 1` 根 bar 都是 NaN。对标签调用 `get_features(panel)` 得到的是未平移的滞后收益，不能拿来当预测目标。

### 分析一个因子

`analyze(start, end, ...)` 仿照 alphalens 库，报告从 `start` 到 `end` 因子按未来收益给标的排序的能力。把一个或多个前瞻收益标签传给 `frets`；每个因子变量（默认是 `get_factor_names()` 的全部，也可用 `factor_names` 指定）与每个标签变量两两配对。因子和每个标签都按 `[start, end]` 请求面板：`data_strategy="cal"`（默认）时用 `compute(start, end)`，`data_strategy="read"` 时用 `read(start, end)`，此时所有存储都必须已在该区间上构建。`data_strategy` 取其他值时抛出 `ValueError`。两个面板最常见的 bar 间隔必须相同（与 `BaseDataset.time_interval` 的规则一致），否则 `analyze()` 抛出 `ValueError` 并写明两个间隔；之后两者按共同的时间戳和标的做内连接。

每个配对得到：

| 类别 | 指标 |
|---|---|
| 信息系数 | 每期 IC（跨标的的 Spearman 秩相关）、IC 均值、标准差、IR（均值 / 标准差）、t 统计量、p 值、偏度、超额峰度、正值占比、月度平均 IC |
| 收益 | 每个因子分位组的平均未来收益（第 1 组是因子值最低的一组）、每期最高组减最低组的价差、各分位组和多空组合的累计收益 |
| 换手 | 每个分位组中上一期不在该组的标的占比，以及因子滞后一期的秩自相关 |

`quantiles`（默认 5）决定等数量分组的组数。标签跨 `n` 根 bar（`kwargs["n_forward_periods"]`）时，累计收益按每根 bar 的收益率 `(1 + r) ** (1 / n) - 1` 复利。下面的例子在与第一节 `factor` 相同的八个标的上构造一个单 bar 的 `Return` 标签，再用它分析 `momentum_5`。数据是随机游走，所以 IC 接近零，这符合预期。

```python
>>> import os
>>> from quantlab.base.config import FactorConfig
>>> opens = xr.Dataset(
...     {"adjOpen": (["timestamp", "symbol"], close * 0.99)},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=90), "symbol": symbols},
... )
>>> XrBackend().to_internal(opens).write("data/spot_open.zarr")
XrBackend()
>>> fwd = Return(FactorConfig(
...     warmup_bars=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 1},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/spot_open.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     file_path="data/labels/fwd_ret.zarr", njobs=2,
... ))
>>> result = factor.analyze(
...     "2024-02-01", "2024-02-29", frets=[fwd], quantiles=4,
...     output_dir="data/analysis/momentum",
... )
>>> list(result.pairs)
['momentum_5__ret_1']
>>> pair = result.pairs["momentum_5__ret_1"]
>>> round(pair.summary["ic_mean"], 4), round(pair.summary["ir"], 4), pair.summary["n_periods"]
(-0.0494, -0.1475, 29)
>>> pair.mean_quantile_returns.round(4).tolist()
[0.001, -0.0025, 0.0004, -0.001]
>>> sorted(os.listdir("data/analysis/momentum"))
['config.json', 'ic.csv', 'momentum_5__ret_1.png', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv']
>>> import json
>>> from quantlab.utils.module import load_factor_from_config
>>> cfg = json.load(open("data/analysis/momentum/config.json"))
>>> list(cfg), type(load_factor_from_config(cfg["frets"][0])).__name__
(['factor', 'frets'], 'Return')
```

结果对象包含 `pairs`（按 `"<factor>__<fret>"` 索引的 `PairAnalysis`，内有 IC 序列、其累计和 `cumulative_ic`、分位收益、换手和 `summary` 字典）、`figures`（每个配对一张 matplotlib 图，只在不传 `output_dir` 时保留），以及 `summary_table()`、`ic_table()`、`monthly_ic_table()`、`quantile_returns_table()`、`turnover_table()` 返回的整洁表。传入 `output_dir` 时，这些表写成 CSV，标量指标写成 `summary.json`，每张图写成 `<factor>__<fret>.png`，`config.json` 保存因子和各标签的配置，每一项都能用 `load_factor_from_config` 重建；此时各图在所有 CPU 上并行绘制并直接写成 PNG，不保留在内存里，因为对整个因子库做报告时画图是主要开销。不传 `output_dir` 则不写任何文件。图不经过 `pyplot` 创建，因此不会弹出显示，也不需要关闭；`fig.savefig(path)` 即可保存。IC 面板的右轴画累计 IC；换手率和排名自相关两个面板不画逐期原始值，而是画半透明的滚动区间（`rolling_window` 期内的最小到最大值，默认 22 期）和滚动均值线。指标用 polars 计算：每一批因子变量（`chunk_size`，默认 32）在 `(timestamp, symbol)` 长表上构成一个惰性计划，只 collect 一次。实现位于 `quantlab.analysis.factor_report`，其中 `FactorAnalyzer.run(factor, frets, features=..., labels=[...], factor_names=None, output_dir=None)` 显式接收特征面板和标签面板。

除了秩 IC，每个配对还报告 Pearson IC（`pearson_ic`，汇总里有 `pearson_ic_mean`、`pearson_ic_std`、`pearson_ir`、`pearson_ic_t_stat`）；Pearson IC 与秩 IC 差距大，说明线性关系由少数极端值驱动。IC 均值另有 Newey-West t 值 `ic_nw_t_stat` 和 `ic_nw_p_value`，用 Bartlett 权重，滞后阶数 `ic_nw_lags` 取 `horizon - 1`（重叠的多期前瞻收益带来的自相关）与经验规则 `floor(4 * (n / 100) ** (2 / 9))` 中的较大者。标签跨多根 bar 时，普通的 `ic_t_stat` 会高估显著性，应看 `ic_nw_t_stat`。秩自相关在 `FactorAnalyzer(autocorrelation_lags=(1, 5, 10, 20))` 的每个滞后阶上计算：`rank_autocorrelations` 每个滞后阶一列，汇总里是 `rank_autocorrelation_lag<k>`，`turnover_table()` 每个滞后阶一列，图里在同一个面板中画出每个滞后阶的滚动均值，滞后越短颜色越深，图例标出各自的均值。多空组合新增 `long_short_annual_return`、`long_short_annual_volatility`、`long_short_sharpe` 和 `long_short_max_drawdown`，按从数据测出的 `periods_per_year` 年化（日频股票约 252，日频加密货币约 365）；净值跌到 0 后保持为 0。传入两个或以上的 fret 时，`ic_decay_table()` 按期限列出每个配对的 IC 均值及其 95% Newey-West 区间，写入 `ic_decay.csv`，并且每张配对图都多一个面板，画出该因子 IC 均值随期限的变化，当前配对的 fret 用圆圈标出。和自相关面板一起看，就能知道信号衰减得多快、因子变化得多慢，两者一起提示合适的持有期。要得到它，传入多个期限的 fret，例如 `n_forward_periods` 为 1、5、20 的几个 `Return` 标签。

分析两个或以上的因子变量时，结果还带有 `correlation`，即这些变量两两之间的 `FactorCorrelation`（`quantlab.analysis.factor_correlation`）；只有一个变量时为 `None`。两个变量的相关性是它们在每个时间点上跨标的的 Spearman 秩相关，再对时间取平均，同时给出标准差和所用的时间点数。每个变量只在自己有效的标的中排一次秩，所以两个变量覆盖的标的不同时，结果是 Spearman 相关的近似值而不是精确值；这样几百个变量每个时间点也只需几次矩阵乘法。变量按 `1 - |相关|` 做平均连接的层次聚类，所以一个因子和它的相反数会落在同一组，矩阵也按聚类顺序排列；`FactorAnalyzer(correlation_threshold=0.7)` 决定在哪里切分成组。`pairs_table()` 按强度从高到低列出每一对，`cluster_summary()` 列出每个至少两个变量的组及其大小、组内平均 `|相关|` 和成员，`cluster_table()` 给出每个变量所属的组和位置。传 `output_dir` 时写出 `factor_correlation.csv`（排序后的矩阵）、`factor_correlation_pairs.csv`、`factor_clusters.csv` 和 `factor_correlation.png`，`summary.json` 增加一个 `correlation` 条目；不传时图保存在 `correlation_figure`。图里是固定 -1 到 1 发散色标（红为负、灰为无、蓝为正）的排序矩阵，恒为 1 的对角线留白，每个至少两个变量的组都有框线；旁边是按 `|相关|` 画、按正负着色的最强相关对，最大的几个组及其大小、组内 `|相关|` 和前几个成员，以及所有配对的分布（对数纵轴，让少数强相关对也看得见）。变量不超过 `FactorCorrelationFigure(label_limit=800)` 个时标出每行每列的名字：热力图会放大到每行至少 6.5 磅高，PNG 以 150 dpi 输出，所以超过约一百个变量时需要放大阅读名字（300 个变量得到 5450×4445 像素的图）。超过上限时只标组，每个变量的位置见 `factor_clusters.csv`。在 2500 天、500 个标的的面板上，300 个变量约需 25 秒。

### 沿时间或跨标的做标准化

`quantlab.my_ops.preprocess` 提供四个 KunQuant 算子。`WindowedZScore` 让每个标的相对自己的滚动窗口做标准化，属于时间序列标准化；`CrossSectionalZScore` 在每个时间点上跨所有标的做标准化。用哪一个取决于使用该因子的策略。`Alpha101SpotKline` 和 `Alpha158SpotKline` 对每个输出应用 `WindowedZScore`，窗口是 `kwargs["zscore_window"]` 根 bar（默认 20），与 `warmup_bars` 相互独立；要让第一个请求的 bar 完全标准化，`warmup_bars` 必须覆盖 alpha 自身的回看长度再加 `zscore_window - 1` 根 bar（完整的 Alpha158 最多是 60 + 19）。`zscore_window` 不是正整数时，构造因子就会被拒绝；`Alpha101Stock` 和 `Alpha158Stock` 对每个输出应用 `CrossSectionalZScore`。“扩展”一节中的 KunQuant 因子同时用了两个算子。

该模块还有两个截面去极值算子。`CrossSectionalWinsorize(v, lower=0.01, upper=0.99)`（缩尾）在每个时间点把取值截到该时点所有标的的 `lower` 和 `upper` 分位数之间；`CrossSectionalTrim(v, lower=0.01, upper=0.99)`（截尾）把严格落在这两个分位数之外的值设为 NaN。分位数忽略 NaN，并按线性插值计算，与 `np.nanquantile` 一致。常见用法是 `CrossSectionalZScore(CrossSectionalWinsorize(v))`，避免少数极端标的主导均值和标准差。KunQuant 0.1.11 没有内置这两个算子：它的 `Clip` 按固定常数截断，`WindowedQuantile` 是沿时间方向的。

### 已有的因子

| 类 | 后端 | 说明 |
|---|---|---|
| `Momentum` | Polars | Polars 参考因子，读取 `Close` |
| `Alpha101SpotKline`、`Alpha101Stock` | KunQuant | KunQuant 的 Alpha101 库 |
| `Alpha158SpotKline`、`Alpha158Stock` | KunQuant | Alpha158 特征；试验时建议固定 `factor_names` |
| `ResidualMomentumFF3` | KunQuant | Fama-French 三因子残差动量；因子序列来自 Fama-French CSV 或面板本身 |
| `LiteratureAlpha` | KunQuant | 覆盖价格、风险、流动性、基本面和盈利事件的 8 个原始值/排名因子 |
| `Return`、`BinaryReturn` | KunQuant | 前瞻收益标签 |

每个类的 docstring 里都有配置示例。

### 文献型股票 Alpha 因子包

作者：[Jerry](https://github.com/j38903016-lgtm)

`LiteratureAlpha` 是一个与 universe 解耦的通用 KunQuant 因子类。它计算
8 个特征，每个特征同时输出原始值和截面排名。历史成分股筛选应在输入市场
面板上单独完成；被掩码成 NaN 的股票会自动被 `Rank` 排除。

| 输出名前缀 | 定义与方向 | 文献 |
|---|---|---|
| `high_52week_proximity` | `拆股调整价 / 252 日滚动最高价`；高值为正向 | [George 和 Hwang（2004）](https://doi.org/10.1111/j.1540-6261.2004.00695.x) |
| `short_reversal` | 21 日复合收益的负值；高值表示前一个月表现更差 | [Jegadeesh（1990）](https://doi.org/10.1111/j.1540-6261.1990.tb05110.x) |
| `low_max` | 21 日最大单日收益的负值；高值回避彩票型股票 | [Bali、Cakici 和 Whitelaw（2011）](https://www.nber.org/papers/w14804) |
| `low_idiosyncratic_volatility` | 21 日 FF3 回归残差标准差的负值 | [Ang、Hodrick、Xing 和 Zhang（2006）](https://doi.org/10.1111/j.1540-6261.2006.00836.x) |
| `amihud_illiquidity` | `abs(收益)/(未复权收盘价*成交股数)` 的均值取对数；越大越不流动 | [Amihud（2002）](https://doi.org/10.1016/S1386-4181(01)00024-6) |
| `gross_profitability` | 最新已公开毛利润除以总资产；高值为正向 | [Novy-Marx（2013）](https://www.nber.org/papers/w15940) |
| `conservative_asset_growth` | 年度总资产增长率的负值；高值代表更保守的投资 | [Cooper、Gulen 和 Schill（2008）](https://doi.org/10.1111/j.1540-6261.2008.01370.x) |
| `standardized_unexpected_earnings` | `(实际 EPS - 公告前一致预期 EPS)/缩放价格`；高值为正向 | [Livnat 和 Mendenhall（2006）](https://doi.org/10.1111/j.1475-679X.2006.00196.x) |

每个前缀都有 `<stem>_raw` 和 `<stem>_rank`。`factor_names` 可以选择任意
子集；类会裁剪无关公式，并且只要求该子集真正依赖的面板字段。窗口和字段名
均可在 `kwargs` 中修改。完整默认图读取 `ret`、`adjClose`、`close`、
`volume`、4 个 FF3 输入、3 个财务字段和 3 个盈利事件字段。FF3 也可以通过
`kwargs={"fama_french_csv": "..."}` 使用与 `ResidualMomentumFF3` 相同的 CSV。

PIT 正确性由数据层负责。`gross_profit`、`total_assets` 和
`prior_year_total_assets` 只能从财报公开后开始生效。三个盈利事件字段必须
冻结实际 EPS、公告前已经存在的一致预期以及事件缩放价格；不能把当前修订后
的预期连接到历史实际值。52 周高点应使用仅拆股调整的价格，Amihud 美元成交额
则需要未复权成交价和原始成交股数。

```python
from quantlab.base.config import FactorConfig
from quantlab.factor.literature_alpha import LiteratureAlpha

factor = LiteratureAlpha(FactorConfig(
    warmup_bars=400,
    dataset=dataset,
    mode="batch",
    data_columns=(
        "adjClose", "ret", "risk_free", "mkt_rf", "smb", "hml",
        "close", "volume", "gross_profit", "total_assets",
        "prior_year_total_assets", "eps_actual_event",
        "eps_consensus_event", "eps_scale_price_event",
    ),
    factor_names=None,  # 全部 16 个 raw/rank 输出
    file_path="data/factors/literature_alpha.zarr",
))
features = factor.get_features(factor.compute("2020-01-01", "2024-12-31"))
```

## 扩展

### 一个 Polars 因子

继承 `FactorPolars` 并实现 `_get_factor_lazyframe`。它接收 `polars.LazyFrame` 形式的数据集，返回只包含 `timestamp`、`symbol` 和因子列的惰性表。因子名从返回表的 schema 读出。还需要覆写 `_get_features` 并返回面板，否则 `get_features(panel)` 会抛出 `NotImplementedError`。

```python
>>> import polars as pl
>>> from quantlab.base.factor import FactorPolars
>>> class RelativeVolume(FactorPolars):
...     def _get_factor_lazyframe(self, lf):
...         volume = pl.col("Volume")
...         return (
...             lf.sort(["symbol", "timestamp"])
...             .with_columns(
...                 (volume / volume.rolling_mean(5).over("symbol") - 1.0).alias("rel_volume_5")
...             )
...             .select(["timestamp", "symbol", "rel_volume_5"])
...         )
...     def _get_features(self, data):
...         return data
...
>>> rv = RelativeVolume(PolarsFactorConfig(
...     warmup_bars=10, dataset=make_dataset(), file_path="data/factors/rel_volume.zarr",
... ))
>>> rv.get_factor_names()
('rel_volume_5',)
>>> out = rv.get_features(rv.compute("2024-02-01", "2024-02-29"))
>>> dict(out.sizes), str(out["rel_volume_5"].dtype)
({'timestamp': 29, 'symbol': 8}, 'float64')
>>> int(out["rel_volume_5"].isnull().sum())
0
```

### 一个 KunQuant 因子

继承 `FactorKunQuant`，实现 `_get_factor_names`、`_get_factor_func`（KunQuant 计算图：`data_columns` 的每一项对应一个 `Input`，每个因子名对应一个 `Output`）和 `_get_features`。下面的计算图输出一个均线偏离度，分别给出原始值、沿时间的 z-score 和跨标的的 z-score。KunQuant 在每次 `compute()` 时编译（这里约一秒）。字段之间互相约束的因子（例如 `data_columns` 要和参数对应）重写 `_validate_config`，读取 `self.config`，不合格时抛 `ValueError`；它在每次给 config 赋值时运行，包括 `copy()` 和 `resample()` 所做的赋值，被拒绝的 config 不会生效，因子保留原来的 config。`LiteratureAlpha` 和 `ResidualMomentumFF3` 就是这样检查 `data_columns` 的。

```python
>>> import KunQuant.ops as op
>>> from KunQuant.Op import Builder, Input, Output
>>> from KunQuant.Stage import Function
>>> from quantlab.base.config import FactorConfig
>>> from quantlab.base.factor import FactorKunQuant
>>> from quantlab.my_ops.preprocess import CrossSectionalZScore, WindowedZScore
>>> class MaDeviation(FactorKunQuant):
...     def _get_factor_names(self):
...         return ("ma_dev_5", "ma_dev_ts", "ma_dev_cs")
...     def _get_features(self, data):
...         return data
...     def _get_factor_func(self):
...         builder = Builder()
...         with builder:
...             close = Input("close")
...             dev = op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0)
...             Output(dev, "ma_dev_5")
...             Output(WindowedZScore(dev, 10), "ma_dev_ts")
...             Output(CrossSectionalZScore(dev), "ma_dev_cs")
...         return Function(builder.ops)
...
```

```python
>>> kq = MaDeviation(FactorConfig(
...     warmup_bars=10, dataset=make_dataset(), mode="batch", data_columns=("close",),
...     file_path="data/factors/ma_dev.zarr", njobs=2,
... ))
>>> out = kq.get_features(kq.compute("2024-02-01", "2024-02-29"))
>>> list(out.data_vars), dict(out.sizes)
(['ma_dev_5', 'ma_dev_ts', 'ma_dev_cs'], {'timestamp': 29, 'symbol': 8})
>>> float(abs(out["ma_dev_cs"].mean("symbol")).max()) < 1e-5
True
>>> out["ma_dev_cs"].std("symbol", ddof=1).values[:3].round(4)
array([1., 1., 1.], dtype=float32)
>>> int(out["ma_dev_ts"].isel(symbol=0).notnull().values.argmax())
3
```

跨标的输出在每个时间点上均值为 0、标准差为 1。时间序列输出在两层嵌套窗口填满之前是 NaN：5 根 bar 的均线加上 10 根 bar 的 z-score 一共需要 14 根 bar，而 `warmup_bars=10` 在第一个请求日期之前提供了 10 根 bar，所以第一个有效值出现在下标 3。流式模式下，`cal_stream` 每次把编译好的计算图推进一根 bar 并返回因子本身，且需要在数据集配置上固定 `symbols`。不传面板的 `get_features()` 返回最近一次 `cal_stream()` 算出的那根 bar。流式结果与批量公式一致：

```python
>>> import numpy as np
>>> stream = MaDeviation(FactorConfig(
...     warmup_bars=10, mode="stream", data_columns=("close",), njobs=2,
...     dataset=SpotKlineDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/klines.zarr",
...         market="crypto_spot", frequency="1d",
...         symbols=tuple(symbols),
...     )),
... ))
>>> for step in range(6):
...     row = stream.cal_stream({"close": close[step].astype("float32")}, step, symbols).get_features()
...
>>> dict(row.sizes)
{'timestamp': 1, 'symbol': 8}
>>> row["ma_dev_5"].values[0, :3].round(5)
array([-0.00857,  0.01526,  0.00988], dtype=float32)
>>> (close[5] / close[1:6].mean(axis=0) - 1)[:3].round(5)
array([-0.00857,  0.01526,  0.00988])
```

## 注意事项

在 macOS 上，批量模式要求标的数是 SIMD 块宽度的整数倍，`compute()` 会用全 NaN 的假标的把标的轴补到 8 的倍数、算完再裁掉，因此任何标的数都能运行；不补的话 5 个标的会报 `RuntimeError: Bad shape at close`，报错信息没有提到标的数。在 Linux x86（AVX2）上任何标的数都能运行，不做补齐。

流式模式下，`data_columns` 的每一项都必须被某个 `Output` 用到，因为 KunQuant 会剪掉没用到的输入。多给一列会在 `init_stream()` 中报 `RuntimeError: Cannot find the buffer name`。批量模式容忍多余的输入。

因子类没有覆写 `_get_features` 时，`get_features(panel)` 会抛出一个不带消息的 `NotImplementedError`。同样，在只产出特征的类上调用 `get_labels(panel)` 会抛 `RuntimeError: Momentum does not support get_label()`。

Polars 因子引用了存储中不存在的列时，构造对象就会失败，因为因子名是通过在少量行上运行表达式推出来的：`polars.exceptions.ColumnNotFoundError: unable to find column "close"; valid columns: ["timestamp", "symbol", "Close", ...]`。要用存储自己的列名（`Close`），而不是 KunQuant 的列名（`close`）；合并输入例外，它带的是共享列名（`close`）。

`read(start, end)` 和 `extend(end)` 需要 `build` 记录的区间：用其他方式写出的存储会抛出 `ValueError: Momentum.read(): the store at data/factors/nob.zarr has no recorded range, so it cannot answer a date-range request; write it with build(start, end).` 给 `extend(end)` 传入记录区间已经覆盖到的 `end` 会抛出 `ValueError: Momentum.extend(): the store at data/factors/momentum.zarr already covers 2024-01-21 to 2024-03-20; extend() appends only bars after 2024-03-20, got end '2024-03-10'.`

`warmup_bars` 是数据集自己日历上的 bar 数，没有数据的日子会被跳过，不计入。嵌套的滚动窗口需要两个窗口长度之和。

`FactorKunQuant.symbols` 和 `num_symbols` 只在流式模式下存在，取值是数据集配置上固定的标的。批量模式下计算覆盖所请求面板中的标的，访问它们会抛出 `ValueError: MaDeviation.symbols: only a stream-mode factor has a fixed symbol list; a batch computation runs over the symbols of the requested panel. config.mode is 'batch'.` 同样，在流式模式的因子上调用 `compute()` 会抛出 `ValueError: MaDeviation.compute(): a date-range computation runs the batch graph, but config.mode is 'stream'.`

重采样后的因子是其源面板的一个视图：`extend()`、`init_stream()` 和 `cal_stream()` 会拒绝：`Momentum.extend(): a resampled factor (resample_freq='1d') is a view of its source panel and does not support extend. Compute or update the source factor, then resample it.` 构建出的重采样 store 是缓存：重新构建源因子不会刷新它。

`FactorKunQuant.compute()` 每次调用都会重新编译计算图。把 `factor_names` 固定为需要的列可以让计算图保持较小。

`CrossSectionalZScore` 在某个时间点有效值少于两个或没有离散度时输出 NaN。批量运行必须从第 0 根 bar 开始，`compute()` 总是这样做。`CrossSectionalWinsorize` 和 `CrossSectionalTrim` 同样要求从第 0 根 bar 开始。每组不同的 `(lower, upper)` 会编译出各自的 C++ 函数，所以构造得到的对象属于一个生成的子类，例如 `CrossSectionalWinsorize_0p01_0p99`；`isinstance(op, CrossSectionalWinsorize)` 仍然成立。

## 另请参阅

`backend.md` 介绍 `XrBackend` 以及 `extend()` 背后的追加检查；`dataset.md` 介绍因子读取的数据集；`model.md` 介绍模型如何使用 `get_features()` 和 `get_labels()`。相关模块：`quantlab.base.factor`（`Factor`、`FactorKunQuant`、`FactorPolars`）、`quantlab.base.config`（`FactorConfig`、`PolarsFactorConfig`）、`quantlab.factor`、`quantlab.label.fret` 和 `quantlab.my_ops.preprocess`。
