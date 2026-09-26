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
| 输入列名 | 由数据集重命名（`close`、`amount`、`adjClose`） | 存储中的原始列名（`Close`） |
| 输出 dtype | float32 | float64 |
| 开销 | 每次 `cal()` 都要编译计算图 | 无 |

KunQuant 是主后端：现有的 alpha 因子库用到的滚动和截面算子它都有，并且只有它能在实盘数据上逐 bar 运行。`FactorPolars` 是新因子的补充路径，适合用 DataFrame 表达式更容易描述的因子，或者需要 KunQuant 没有的运算。需要同时支持流式运行的因子必须写成 KunQuant 因子。

### 配置

两个后端使用的配置字段如下。`FactorConfig` 另外有 `mode`（`"batch"` 或 `"stream"`）、`data_columns`（输入计算图的数据集变量）和 `njobs`（执行器线程数）；`PolarsFactorConfig` 没有额外字段。

| 字段 | 含义 |
|---|---|
| `window` | 回看的日历天数，在 `start_date` 之前读取，用来让滚动算子预热 |
| `dataset` | 因子读取的数据集 |
| `file_path` | 因子保存和读取所用的 Zarr 存储 |
| `factor_names` | 输出列名；为 `None` 时由因子自己推出 |
| `start_date`、`end_date` | 计算的起止日期（含） |
| `symbols` | 限定的标的 |
| `kwargs` | 因子类自行读取的自由选项 |

输出的日期范围由因子自己的 `start_date` 和 `end_date` 决定。它们没设置时默认是无界范围，同时因子会把数据集的日期改成从 `start_date` 往前 `window` 天到 `end_date`。日期请设在因子配置上，而不是数据集配置上。

### 计算并保存一个因子

下面的会话先写入一个小的合成现货存储，并定义一个基于它构造数据集的辅助函数。`Momentum` 是 Polars 后端的参考因子：`Close_t / Close_{t-n} - 1`，`n` 从 `kwargs` 读取。因子名在对象构造完成时就已确定，不需要先计算；数据集的起始日期已被提前 20 天，用来预热窗口。

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

```python
>>> config = PolarsFactorConfig(
...     window=20, dataset=make_dataset(), file_path="data/factors/momentum.zarr",
...     kwargs={"n": 5}, start_date="2024-02-01", end_date="2024-02-29",
... )
>>> factor = Momentum(config)
>>> factor.get_factor_names()
('momentum_5',)
>>> factor.config.dataset.config.start_date
'2024-01-12'
>>> panel = factor.cal().get_features()
>>> dict(panel.sizes), list(panel.data_vars)
({'timestamp': 29, 'symbol': 8}, ['momentum_5'])
>>> panel["momentum_5"].isel(timestamp=0, symbol=slice(0, 3)).values.round(4)
array([ 0.0161,  0.0079, -0.0007])
>>> factor.save(mode="w") is factor
True
```

`cal()` 计算面板并把它放在因子的存储后端里，`get_features()` 以 `(timestamp, symbol)` 为索引的 `xarray.Dataset` 返回。`save(mode="w")` 把它写到 `file_path` 的 Zarr 存储中，替换原有内容。

## 常见任务

### 用更晚的日期扩充已保存的因子

`update()` 把当前持有的面板追加到已有存储：先加宽 timestamp、symbol 和变量三个轴，再追加。它调用 `XrBackend.widen_and_append`，因此后端指南中描述的追加检查同样适用。替换存储用 `save(mode="w")`，扩充存储用 `update()`。

```python
>>> later = Momentum(PolarsFactorConfig(
...     window=20, dataset=make_dataset(), file_path="data/factors/momentum.zarr",
...     kwargs={"n": 5}, start_date="2024-03-01", end_date="2024-03-20",
... ))
>>> later.cal().get_features().sizes["timestamp"]
20
>>> later.update() is later
True
>>> stored = xr.open_zarr("data/factors/momentum.zarr")
>>> stored.sizes["timestamp"], str(stored["timestamp"].values[-1])[:10]
(49, '2024-03-20')
```

### 读回因子，并从配置重建

`read()` 打开存储并按配置的日期收窄。`get_config()` 返回描述因子及其数据集的 dict，`load_factor_from_config` 可以据此重建因子。

```python
>>> reader = Momentum(PolarsFactorConfig(
...     window=20, dataset=make_dataset(), file_path="data/factors/momentum.zarr",
...     kwargs={"n": 5}, start_date="2024-03-10", end_date="2024-03-15",
... ))
>>> reader.read().get_features().sizes["timestamp"]
6
>>> cfg = factor.get_config()
>>> cfg["name"], cfg["kwargs"], cfg["dataset"]["market"]
('quantlab.factor.momentum.Momentum', {'n': 5}, 'crypto_spot')
>>> from quantlab.utils.module import load_factor_from_config
>>> rebuilt = load_factor_from_config(cfg)
>>> type(rebuilt).__name__, rebuilt.get_factor_names()
('Momentum', ('momentum_5',))
```

### 把因子重采样到更粗的 bar

`resample(freq, how)` 返回因子的一个副本，其计算出的面板被聚合到更粗的 bar 上。因子仍然在其 dataset 自身的 bar 上计算，只有输出被聚合，因此分钟 bar 上的动量会变成"每日最后一分钟的值"这一日频序列，而它衡量的东西没有变。`freq` 和 `how` 的取值与 `BaseDataset.resample` 相同（见 dataset 指南），`how` 也可以只给一个字符串，表示所有因子变量都用这种方法。bar 的切分方式沿用因子所用 dataset 的切分方式。

下面的会话在 dataset 指南里构造的两天分钟 store 上运行 `Momentum`（`config` 就是那个 store 的 `DatasetConfig`）。

```python
>>> factor = Momentum(PolarsFactorConfig(
...     window=1, dataset=SpotKlineDataset(config),
...     file_path="data/factors/momentum.zarr", kwargs={"n": 1},
... ))
>>> minute = factor.cal()
>>> minute.get_features()["momentum_1"].to_pandas().round(3)
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
>>> daily = minute.resample("1d", "last")
>>> daily.get_features()["momentum_1"].to_pandas().round(3)
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02    0.333    0.333
2024-01-03    0.143    0.143
>>> daily.config.dataset.config.resample_freq, daily.config.resample_freq
(None, '1d')
```

副本有自己的 dataset 对象和空的编译状态，所以在它上面调用 `cal()` 会重新计算分钟面板再重采样。`save()` 写到 `store_path`，即源 store 旁边的那个 store；副本上的 `read()` 在该 store 存在时直接打开它，否则读源 store 再重采样。这次请求能经 `get_config()` 和 `load_factor_from_config` 往返重建。

```python
>>> daily.store_path
'data/factors/momentum_resample_1d.zarr'
>>> daily.cal().get_features().sizes
Frozen({'symbol': 2, 'timestamp': 2})
>>> cfg = daily.get_config()
>>> cfg["resample_freq"], cfg["resample_how"], cfg["dataset"]["resample_freq"]
('1d', 'last', None)
>>> load_factor_from_config(cfg).cal().get_features().sizes
Frozen({'symbol': 2, 'timestamp': 2})
```

如果想在已经重采样的 bar 上计算因子，先对 dataset 做重采样，再把重采样后的 dataset 交给因子。

### 计算标签

标签是 `get_labels()` 返回前瞻值的 KunQuant 因子。`quantlab.label.fret` 中的 `Return` 是从下一根 bar 的复权开盘价到其后 `n` 根 bar 的复权开盘价的收益，`BinaryReturn` 在该收益为正时取 1.0。计算图算出的是滞后收益（KunQuant 只能向后看），`get_labels()` 再把它向前平移 `n_forward_periods + 1` 根 bar，因此最后 `n_forward_periods + 1` 根 bar 是 NaN。标签读取 `adjOpen`，所以数据集必须带复权价格：美股数据集有，加密现货数据集没有。

```python
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
...     window=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 2},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/stock.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     file_path="data/labels/ret.zarr", njobs=2,
... ))
>>> label.get_factor_names()
('ret_2',)
>>> ret = label.cal().get_labels()["ret_2"]
>>> dict(ret.sizes)
{'timestamp': 30, 'symbol': 8}
>>> ret.isel(symbol=0).values[:2].round(5)
array([-0.03945,  0.01532], dtype=float32)
>>> round(float(px[3, 0] / px[1, 0] - 1), 5)
-0.03945
>>> ret.isel(symbol=0).values[-3:]
array([nan, nan, nan], dtype=float32)
```

对标签调用 `get_features()` 得到的是未平移的滞后收益，不能拿来当预测目标。

### 分析一个因子

`analyze()` 仿照 alphalens 库，报告因子按未来收益给标的排序的能力。把一个或多个前瞻收益标签传给 `frets`；每个因子变量（默认是 `get_factor_names()` 的全部，也可用 `factor_names` 指定）与每个标签变量两两配对。因子和标签都必须已经持有面板（先调用 `cal()` 或 `read()`）。两个面板最常见的 bar 间隔必须相同（与 `BaseDataset.time_interval` 的规则一致），否则 `analyze()` 抛出 `ValueError` 并写明两个间隔；之后两者按共同的时间戳和标的做内连接。

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
...     window=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 1},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/spot_open.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     start_date="2024-02-01", end_date="2024-02-29",
...     file_path="data/labels/fwd_ret.zarr", njobs=2,
... ))
>>> fwd.cal() is fwd
True
>>> result = factor.analyze(frets=[fwd], quantiles=4, output_dir="data/analysis/momentum")
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

结果对象包含 `pairs`（按 `"<factor>__<fret>"` 索引的 `PairAnalysis`，内有 IC 序列、分位收益、换手和 `summary` 字典）、`figures`（每个配对一张 matplotlib 图），以及 `summary_table()`、`ic_table()`、`monthly_ic_table()`、`quantile_returns_table()`、`turnover_table()` 返回的整洁表。传入 `output_dir` 时，这些表写成 CSV，标量指标写成 `summary.json`，每张图写成 `<factor>__<fret>.png`，`config.json` 保存因子和各标签的配置，每一项都能用 `load_factor_from_config` 重建。不传 `output_dir` 则不写任何文件。图不经过 `pyplot` 创建，因此不会弹出显示，也不需要关闭；`fig.savefig(path)` 即可保存。实现位于 `quantlab.analysis.factor_report`。

### 沿时间或跨标的做标准化

`quantlab.my_ops.preprocess` 提供四个 KunQuant 算子。`WindowedZScore` 让每个标的相对自己的滚动窗口做标准化，属于时间序列标准化；`CrossSectionalZScore` 在每个时间点上跨所有标的做标准化。用哪一个取决于使用该因子的策略。`Alpha101SpotKline` 和 `Alpha158SpotKline` 对每个输出应用 `WindowedZScore`；`Alpha101Stock` 和 `Alpha158Stock` 对每个输出应用 `CrossSectionalZScore`。“扩展”一节中的 KunQuant 因子同时用了两个算子。

该模块还有两个截面去极值算子。`CrossSectionalWinsorize(v, lower=0.01, upper=0.99)`（缩尾）在每个时间点把取值截到该时点所有标的的 `lower` 和 `upper` 分位数之间；`CrossSectionalTrim(v, lower=0.01, upper=0.99)`（截尾）把严格落在这两个分位数之外的值设为 NaN。分位数忽略 NaN，并按线性插值计算，与 `np.nanquantile` 一致。常见用法是 `CrossSectionalZScore(CrossSectionalWinsorize(v))`，避免少数极端标的主导均值和标准差。KunQuant 0.1.11 没有内置这两个算子：它的 `Clip` 按固定常数截断，`WindowedQuantile` 是沿时间方向的。

### 已有的因子

| 类 | 后端 | 说明 |
|---|---|---|
| `Momentum` | Polars | Polars 参考因子，读取 `Close` |
| `Alpha101SpotKline`、`Alpha101Stock` | KunQuant | KunQuant 的 Alpha101 库 |
| `Alpha158SpotKline`、`Alpha158Stock` | KunQuant | Alpha158 特征；试验时建议固定 `factor_names` |
| `ResidualMomentumFF3` | KunQuant | 月频数据上的 Fama-French 三因子残差动量 |
| `Return`、`BinaryReturn` | KunQuant | 前瞻收益标签 |

每个类的 docstring 里都有配置示例。

## 扩展

### 一个 Polars 因子

继承 `FactorPolars` 并实现 `_get_factor_lazyframe`。它接收 `polars.LazyFrame` 形式的数据集，返回只包含 `timestamp`、`symbol` 和因子列的惰性表。因子名从返回表的 schema 读出。还需要覆写 `_get_features` 并返回面板，否则 `get_features()` 会抛出 `NotImplementedError`。

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
...     window=10, dataset=make_dataset(), file_path="data/factors/rel_volume.zarr",
...     start_date="2024-02-01", end_date="2024-02-29",
... ))
>>> rv.get_factor_names()
('rel_volume_5',)
>>> out = rv.cal().get_features()
>>> dict(out.sizes), str(out["rel_volume_5"].dtype)
({'timestamp': 29, 'symbol': 8}, 'float64')
>>> int(out["rel_volume_5"].isnull().sum())
0
```

### 一个 KunQuant 因子

继承 `FactorKunQuant`，实现 `_get_factor_names`、`_get_factor_func`（KunQuant 计算图：`data_columns` 的每一项对应一个 `Input`，每个因子名对应一个 `Output`）和 `_get_features`。下面的计算图输出一个均线偏离度，分别给出原始值、沿时间的 z-score 和跨标的的 z-score。KunQuant 在第一次 `cal()` 时编译（这里约一秒）。

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
...     window=10, dataset=make_dataset(), mode="batch", data_columns=("close",),
...     file_path="data/factors/ma_dev.zarr", start_date="2024-02-01",
...     end_date="2024-02-29", njobs=2,
... ))
>>> out = kq.cal().get_features()
>>> list(out.data_vars), dict(out.sizes)
(['ma_dev_5', 'ma_dev_ts', 'ma_dev_cs'], {'timestamp': 29, 'symbol': 8})
>>> float(abs(out["ma_dev_cs"].mean("symbol")).max()) < 1e-5
True
>>> out["ma_dev_cs"].std("symbol", ddof=1).values[:3].round(4)
array([1., 1., 1.], dtype=float32)
>>> int(out["ma_dev_ts"].isel(symbol=0).notnull().values.argmax())
3
```

跨标的输出在每个时间点上均值为 0、标准差为 1。时间序列输出在两层嵌套窗口填满之前是 NaN：5 根 bar 的均线加上 10 根 bar 的 z-score 一共需要 14 根 bar，而 `window=10` 在第一个请求日期之前提供了 10 根 bar，所以第一个有效值出现在下标 3。流式模式下，`cal_stream` 每次把编译好的计算图推进一根 bar，且需要在数据集配置上固定 `symbols`。流式结果与批量公式一致：

```python
>>> import numpy as np
>>> stream = MaDeviation(FactorConfig(
...     window=10, mode="stream", data_columns=("close",), njobs=2,
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

批量模式要求标的数是所在机器 SIMD 块宽度的整数倍。在运行这些示例的机器上，4、8、12 个标的可以运行，5 个会报 `RuntimeError: Bad shape at close`。报错信息没有提到标的数，需要补齐或裁剪标的轴。

流式模式下，`data_columns` 的每一项都必须被某个 `Output` 用到，因为 KunQuant 会剪掉没用到的输入。多给一列会在 `init_stream()` 中报 `RuntimeError: Cannot find the buffer name`。批量模式容忍多余的输入。

因子类没有覆写 `_get_features` 时，`get_features()` 会抛出一个不带消息的 `NotImplementedError`。同样，在只产出特征的类上调用 `get_labels()` 会抛 `RuntimeError: Momentum does not support get_label()`。

Polars 因子引用了存储中不存在的列时，构造对象就会失败，因为因子名是通过在少量行上运行表达式推出来的：`polars.exceptions.ColumnNotFoundError: unable to find column "close"; valid columns: ["timestamp", "symbol", "Close", ...]`。要用存储自己的列名（`Close`），而不是 KunQuant 的列名（`close`）。

`save()` 默认 `mode="a"`，在 Zarr 里它表示改写已有存储中的变量，不是沿时间追加。保存长度不同的面板会抛出 `ValueError: Momentum.save(mode="a"): cannot write this date range into the existing store at data/f/m.zarr. zarr's "a" means "overwrite variables in an existing store", NOT "append along time", ...`。替换存储用 `save(mode="w")`，扩充存储用 `update()`。

`window` 是数据集回看的日历天数，不是 bar 数。市场在某些日子休市时，同样的天数对应的 bar 更少；嵌套的滚动窗口需要两个窗口长度之和。

重采样后的因子是其源面板的一个视图：`update()`、`init_stream()` 和 `cal_stream()` 会拒绝：`Momentum.update(): a resampled factor (resample_freq='1d') is a view of its source panel and does not support update. Compute or update the source factor, then resample it.` 保存下来的重采样 store 是缓存：重新计算源因子不会刷新它。

`FactorKunQuant.cal()` 每次调用都会重新编译计算图。把 `factor_names` 固定为需要的列可以让计算图保持较小。

`CrossSectionalZScore` 在某个时间点有效值少于两个或没有离散度时输出 NaN。批量运行必须从第 0 根 bar 开始，`cal()` 总是这样做。`CrossSectionalWinsorize` 和 `CrossSectionalTrim` 同样要求从第 0 根 bar 开始。每组不同的 `(lower, upper)` 会编译出各自的 C++ 函数，所以构造得到的对象属于一个生成的子类，例如 `CrossSectionalWinsorize_0p01_0p99`；`isinstance(op, CrossSectionalWinsorize)` 仍然成立。

## 另请参阅

`backend.md` 介绍 `XrBackend` 以及 `update()` 背后的追加检查；`dataset.md` 介绍因子读取的数据集；`model.md` 介绍模型如何使用 `get_features()` 和 `get_labels()`。相关模块：`quantlab.base.factor`（`Factor`、`FactorKunQuant`、`FactorPolars`）、`quantlab.base.config`（`FactorConfig`、`PolarsFactorConfig`）、`quantlab.factor`、`quantlab.label.fret` 和 `quantlab.my_ops.preprocess`。
