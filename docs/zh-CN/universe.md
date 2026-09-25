# 价格与流动性股票池过滤（universe）

[English](../universe.md) | 简体中文

`UniverseFilteredFactor` 把 KunQuant 因子或标签限制在每根 bar 上可交易的标的范围内：原始收盘价高于价格下限，且滚动平均成交额高于流动性下限。它是一个因子包装类。包装后的对象本身仍是 `FactorKunQuant`，因此可以直接放进 `MLConfig.factors` 和 `MLConfig.labels`，也可以配合回测器使用，模型层和回测层无需任何改动。

这个过滤器与指数成分是两回事（见 `constituent` 指南）。成分回答"某天哪些标的属于某个指数"，这个过滤器回答"哪些标的足够贵、足够活跃，可以交易"。两者可以组合使用。

## 前置条件

因子后端是 KunQuant，它会编译因子计算图，需要可用的 C++ 编译器。批量计算要求标的数量是当前主机 SIMD 块宽度的整数倍；在编写本页所用的机器上，16 个标的可以运行。

## 基础

### 判定规则

一个标的在第 `t` 根 bar 上属于股票池，需同时满足两个条件：`t` 时刻的原始 `close` 不低于 `min_price`；截至 `t` 的 `window` 根 bar 上，原始 `close * volume` 的均值不低于 `min_dollar_volume`。两者都使用原始列，不使用复权列，因为复权历史会被后来的拆股和分红改变，无法说明当时这只股票是不是低价股。`t` 之后的任何数据都不影响 `t` 时刻的判定。窗口未满或窗口内含 NaN，一律视为不在池内。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `min_price` | `5.0` | 原始收盘价下限 |
| `min_dollar_volume` | `1_000_000.0` | 原始 `close * volume` 滚动均值下限 |
| `window` | `20` | 滚动均值的 bar 数 |

示例使用一份合成的 30 根 bar、16 个标的的面板：十三个普通标的 `S00` 到 `S12`；`PENY`（原始收盘价 1 美元，但复权收盘价是全面板最高，所以不过滤时排名会把它排第一）；`ILQD`（价格正常，成交量极小）；`DRPX`（到第 20 根 bar 前都在池内，之后原始收盘价跌到 1 美元）。第一个会话构建存储，并定义一个小的 KunQuant 因子，含一个截面输出（`Rank`）和一个时序输出（`WindowedAvg`）。

```python
>>> import os, tempfile
>>> import numpy as np, pandas as pd, xarray as xr
>>> from KunQuant.Op import Builder, Input, Output, Rank
>>> from KunQuant.ops import WindowedAvg
>>> from KunQuant.Stage import Function
>>> from quantlab.base.config import DatasetConfig, FactorConfig
>>> from quantlab.base.factor import FactorKunQuant
>>> from quantlab.dataset.stock import StockDataset
>>> from quantlab.factor.universe_filter import UniverseFilteredFactor
>>> def write_store(root, symbols):
...     n_bars, n = 30, len(symbols)
...     rng = np.random.default_rng(0)
...     adjusted = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_bars, n)), axis=0))
...     raw = adjusted * 1.7
...     opened = adjusted * 1.001
...     volume = np.full((n_bars, n), 1e6)
...     at = symbols.index
...     raw[:, at("PENY")] = 1.0
...     adjusted[:, at("PENY")] = 900.0
...     volume[:, at("ILQD")] = 100.0
...     raw[20:, at("DRPX")] = 1.0
...     adjusted[:, at("DRPX")] = 700.0
...     dims = ("timestamp", "symbol")
...     panel = xr.Dataset(
...         {"adjOpen": (dims, opened), "adjClose": (dims, adjusted),
...          "close": (dims, raw), "volume": (dims, volume)},
...         coords={"timestamp": pd.bdate_range("2024-01-01", periods=n_bars), "symbol": symbols},
...     )
...     store = os.path.join(root, "stock.zarr")
...     panel.to_zarr(store, mode="w")
...     return DatasetConfig(
...         raw_data_dir_path=os.path.join(root, "raw"), zarr_file_path=store,
...         market="us_equity", frequency="1d",
...     )
>>> class RankClose(FactorKunQuant):
...     def _get_factor_names(self):
...         return ("rank_close", "ma_close")
...     def _get_factor_func(self):
...         builder = Builder()
...         with builder:
...             close = Input("adjClose")
...             Output(Rank(close), "rank_close")
...             Output(WindowedAvg(close, 3), "ma_close")
...         return Function(builder.ops)
...     def _get_features(self, data):
...         return data
...     def _get_labels(self, data):
...         raise RuntimeError("RankClose is a feature")
>>> symbols = [f"S{i:02d}" for i in range(13)] + ["PENY", "ILQD", "DRPX"]
>>> dataset_config = write_store(tempfile.mkdtemp(), symbols)
>>> def make_factor():
...     config = FactorConfig(window=3, dataset=StockDataset(dataset_config), mode="batch",
...                           data_columns=("adjClose",), njobs=4)
...     return RankClose(config)
```

### 包装一个因子

包装类接收内部因子和三个参数。它的 `config` 就是内部因子自己的 config 对象，不是副本，所以模型或回测器写入的日期会落在内部因子上。

```python
>>> wrapped = UniverseFilteredFactor(make_factor(), min_price=5.0, min_dollar_volume=1_000_000.0, window=3)
>>> wrapped.config is wrapped.factor.config
True
```

`compute_universe_mask` 返回一个面板的掩码：在池内为 1.0，其余为 NaN。前 `window - 1` 行对所有标的都是 NaN，因为滚动均值需要满窗口。PENY 每根 bar 都因价格不达标被排除，ILQD 则因成交额不达标被排除。DRPX 到第 19 根 bar（2024-01-26）都在池内，从第 20 根 bar 起出池。

```python
>>> panel = StockDataset(dataset_config).read().get_xarray_dataset()
>>> mask = wrapped.compute_universe_mask(panel)
>>> mask.sel(symbol=["S00", "PENY", "ILQD", "DRPX"]).isel(timestamp=[0, 1, 2, 19, 20]).to_pandas()
symbol      S00  PENY  ILQD  DRPX
timestamp                        
2024-01-01  NaN   NaN   NaN   NaN
2024-01-02  NaN   NaN   NaN   NaN
2024-01-03  1.0   NaN   NaN   1.0
2024-01-26  1.0   NaN   NaN   1.0
2024-01-29  1.0   NaN   NaN   NaN
```

### 出池只是置 NaN，不删除标的

`cal()` 运行编译后的计算图，`get_features()` 返回已应用掩码的结果。出池只会把格子置空，绝不会删掉一列。输出的标的轴与输入完全一致，整个窗口都出池的标的会保留为全 NaN 列。因此标的轴不依赖日期窗口，用一个窗口训练出的模型可以拿到另一个窗口的面板。

```python
>>> features = wrapped.cal().get_features()
>>> features.sizes
Frozen({'timestamp': 30, 'symbol': 16})
>>> bool(features["rank_close"].sel(symbol="PENY").isnull().all())
True
```

### 截面算子只看池内标的

对 `Rank` 这类算子来说，只把出池标的的输出置空是不够的，因为它仍然参与了每一次排名。所以包装类会改写因子计算图：把掩码作为一个额外输入加入，并把每个截面算子的每个输入都除以它。除以 1.0 不改变数值，除以 NaN 得到 NaN，因此出池标的从所有排名和截面 z-score 中消失。时序算子不做改写，仍然能看到完整历史。

下面最后一根 bar 上的排名展示了效果。不过滤时，PENY 在 16 个标的中排第一（1.0）；过滤后 PENY 不存在，其余标的在 13 个之间排名。

```python
>>> unfiltered = make_factor().cal().get_features()
>>> unfiltered["rank_close"].isel(timestamp=-1).sel(symbol=["S00", "S01", "PENY"]).to_pandas()
symbol
S00     0.4375
S01     0.5000
PENY    1.0000
Name: rank_close, dtype: float32
>>> features["rank_close"].isel(timestamp=-1).sel(symbol=["S00", "S01", "PENY", "ILQD", "DRPX"]).to_pandas()
symbol
S00     0.461538
S01     0.538462
PENY         NaN
ILQD         NaN
DRPX         NaN
Name: rank_close, dtype: float32
```

纯时序输出 `ma_close` 在过滤后输出有定义的位置上，与不过滤时完全相同。

```python
>>> in_universe = features["ma_close"].notnull()
>>> bool((features["ma_close"] == unfiltered["ma_close"]).where(in_universe, True).all())
True
```

## 常见任务

### 同时包装模型的因子和标签

因子和标签都要包装。只包装因子，标签里仍会保留出池标的的行；只包装标签，因子里的截面算子仍会受到这些标的的影响。交给回测器的价格数据集不要包装：持仓标的的价格必须一直可用。模型和回测器的调用这里只展示形状。

```python
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.universe_filter import UniverseFilteredFactor
from quantlab.label.fret import Return

factors = [UniverseFilteredFactor(Alpha101Stock(factor_config))]
labels = [UniverseFilteredFactor(Return(label_config))]
model = XGBoostRegressor(MLConfig(factors=factors, labels=labels, ...))
```

包装类会把内部数据集的起始日期提前 `2 * window + 10` 个日历日，使第一根被请求的 bar 上滚动均值已经是满窗口。它只会把起始日期往前挪，不会往后挪。

### 标签在它自己的时间戳上打掩码

第 `t` 根 bar 上的前向收益标签描述的是 `t` 之后开仓的仓位。包装类在标签做完前向位移之后才应用掩码，所以 `t` 时刻的掩码决定 `t` 时刻的标签。一个标的在其最后一根池内 bar 上仍保留该 bar 的标签，包括它离开股票池时赚到的那段收益。如果先打掩码再位移，就会用较晚一根 bar 的股票池状态去决定较早一根 bar 的标签。

`n_forward_periods=1` 的 `Return` 是从下一个开盘价到再下一个开盘价的开盘对开盘收益。DRPX 在第 20 根 bar（2024-01-29）出池；它在第 19 根 bar 的标签保持不变，从第 20 根 bar 起的标签为空。

```python
>>> from quantlab.label.fret import Return
>>> def make_label():
...     config = FactorConfig(window=0, dataset=StockDataset(dataset_config), mode="batch",
...                           data_columns=("adjOpen",), njobs=4, kwargs={"n_forward_periods": 1})
...     return Return(config)
>>> label = UniverseFilteredFactor(make_label(), min_price=5.0, min_dollar_volume=1_000_000.0, window=3)
>>> labels = label.cal().get_labels()
>>> labels["ret_1"].sel(symbol="DRPX").isel(timestamp=slice(17, 23)).to_pandas()
timestamp
2024-01-24    0.013593
2024-01-25   -0.006783
2024-01-26    0.041368
2024-01-29         NaN
2024-01-30         NaN
2024-01-31         NaN
Name: ret_1, dtype: float32
>>> make_label().cal().get_labels()["ret_1"].sel(symbol="DRPX").isel(timestamp=slice(17, 23)).to_pandas()
timestamp
2024-01-24    0.013593
2024-01-25   -0.006783
2024-01-26    0.041368
2024-01-29   -0.012877
2024-01-30    0.002454
2024-01-31    0.008919
Name: ret_1, dtype: float32
```

### 掉出股票池的持仓

一个标的出池后，特征全为 NaN，模型对它的预测也是 NaN，而 NaN 分数不可被选中。过滤器不会直接操作持仓。掉出池之后的第一个调仓 bar 上，该标的不再可选，目标权重变为零，仓位在下一根 bar 的开盘价平掉。资格只在调仓 bar 上重新评估，所以持仓在离开股票池之后最多还会被持有 `rebalance_periods - 1` 根 bar。`rebalance_periods` 越小，延迟越短。

下面的会话用上面算出的排名充当分数，用回测层的 top-2 选择器，在一个六根 bar 窗口的第 0 行和第 3 行调仓。DRPX 在第一次调仓（2024-01-25，出池之前）被选中，在第二次调仓（2024-01-30）权重为 0.0。没有调仓的行整行都是 NaN，意思是"保持现有仓位"。

```python
>>> from quantlab.backtest.selection import CrossSectionTopNSelector, rebalance_mask
>>> scores = features["rank_close"].isel(timestamp=slice(18, 24))
>>> selector = CrossSectionTopNSelector(direction="long_only", top_n=2)
>>> weights = selector.select(scores, xr.full_like(scores, 100.0), rebalance_mask(6, 3))["weight"]
>>> held = weights.isel(timestamp=[0, 3]).to_pandas()
>>> held.loc[:, (held != 0).any()]
symbol      DRPX  S07  S08
timestamp                 
2024-01-25   0.5  0.5  0.0
2024-01-30   0.0  0.5  0.5
```

### 保存与读回

通过包装类写出的因子库，保存的是改写后计算图的输出，也就是打输出掩码之前的值。`read()` 会根据数据集的原始收盘价和成交量重新计算掩码，并把它应用到 `get_features()` 返回的结果上。因此因子库必须由包装类写出：由未包装的内部因子写出的库，其截面值已经包含了出池标的，经包装类读取也无法把它们去掉。

```python
>>> root = tempfile.mkdtemp()
>>> def make_stored():
...     config = FactorConfig(window=3, dataset=StockDataset(dataset_config), mode="batch",
...                           data_columns=("adjClose",), njobs=4,
...                           file_path=os.path.join(root, "rank.zarr"),
...                           start_date="2024-01-01", end_date="2024-02-09")
...     return UniverseFilteredFactor(RankClose(config), window=3)
>>> _ = make_stored().cal().save(mode="w")
>>> back = make_stored().read().get_features()
>>> back.sizes
Frozen({'timestamp': 30, 'symbol': 16})
>>> bool(back["rank_close"].sel(symbol="PENY").isnull().all())
True
```

### 序列化与重建

`get_config()` 返回包装类的三个参数，以及放在 `"factor"` 键下的内部因子配置。`from_config()` 根据这个字典重建包装类和内部因子。它不会用当前默认值补全缺失的参数，所以保存过的运行不会被重建成另一个股票池。

```python
>>> cfg = wrapped.get_config()
>>> sorted(cfg)
['factor', 'min_dollar_volume', 'min_price', 'name', 'window']
>>> cfg["window"], cfg["min_price"], cfg["min_dollar_volume"]
(3, 5.0, 1000000.0)
>>> UniverseFilteredFactor.from_config({"factor": cfg["factor"], "window": 3})
Traceback (most recent call last):
  ...
ValueError: UniverseFilteredFactor.from_config: refusing to rebuild -- missing key(s) ['min_dollar_volume', 'min_price'], unknown key(s) []. Missing parameters are NOT filled from the current defaults, ...
```

## 扩展

阈值是参数；规则本身是方法 `compute_universe_mask(panel)`，它返回一个 `(timestamp, symbol)` 数组，在池内为 1.0，其余为 NaN。子类可以通过收窄这个结果来增加条件。批量路径（`cal()` 和 `read()`）会调用这个方法；流式路径在 `cal_stream()` 内部自己构造掩码这一行，不会调用它，所以需要同时作用于流式的规则要在那里重复实现。

示例增加了原始收盘价的上限。普通标的起始价在 85 左右并随机漂移，所以最后一根 bar 上 13 个里只有 5 个低于 80。

```python
>>> class CappedUniverse(UniverseFilteredFactor):
...     def compute_universe_mask(self, panel):
...         mask = super().compute_universe_mask(panel)
...         return mask.where(panel["close"] <= 80.0).rename(mask.name)
>>> capped = CappedUniverse(make_factor(), min_price=5.0, min_dollar_volume=1_000_000.0, window=3)
>>> capped.cal().get_features()["rank_close"].isel(timestamp=-1).notnull().sum().item()
5
>>> features["rank_close"].isel(timestamp=-1).notnull().sum().item()
13
```

## 注意事项

包装类只包装 `FactorKunQuant` 子类。Polars 因子会被拒绝，因为它的截面逻辑是 Polars 表达式，无法改写，而只掩输出会把出池标的留在每一次排名里面。包装一个已包装的对象也会被拒绝，因为两层掩码会悄悄叠加。`window` 至少为一根 bar。

```python
>>> UniverseFilteredFactor(object())
Traceback (most recent call last):
  ...
TypeError: UniverseFilteredFactor wraps a FactorKunQuant, got object. A Polars factor's cross-sectional expressions are polars expressions, not a KunQuant op graph, so they cannot be rewritten -- and masking only the OUTPUTS would leave every out-of-universe symbol sitting inside each rank/zscore, which is exactly what this class exists to prevent.
>>> UniverseFilteredFactor(wrapped)
Traceback (most recent call last):
  ...
TypeError: UniverseFilteredFactor cannot wrap another UniverseFilteredFactor: the inner wrapper would mask the cross-sections a second time, and the two masks' parameters would silently compose. Wrap the innermost factor once, with the parameters you want.
>>> UniverseFilteredFactor(make_factor(), window=0)
Traceback (most recent call last):
  ...
ValueError: window must be >= 1 bar, got 0; it is the number of bars the trailing dollar-volume mean is taken over.
```

掩码需要数据集里有原始的 `close` 和 `volume` 变量。缺少任何一个时，`compute_universe_mask` 会抛出 `ValueError`，并列出当前存在的变量。在 `cal()`、`read()` 或 `cal_stream()` 之前调用 `get_features()` 或 `get_labels()` 会抛出 `RuntimeError`。

```python
>>> wrapped.compute_universe_mask(panel.drop_vars("volume"))
Traceback (most recent call last):
  ...
ValueError: UniverseFilteredFactor needs the RAW column 'volume' to decide universe membership, and the dataset panel does not carry it (present: ['adjClose', 'adjOpen', 'close']). The mask reads RAW close/volume, never the adjusted columns: adjusted history is depressed by splits and dividends, so a penny stock today can look like a $50 stock in 2015.
>>> UniverseFilteredFactor(make_factor(), window=3).get_features()
Traceback (most recent call last):
  ...
RuntimeError: UniverseFilteredFactor: no universe mask has been computed yet, so the outputs cannot be masked. Call cal(), read() or cal_stream() first.
```

KunQuant 有两条限制，对所有调用方都适用。批量计算总是从第 0 根 bar 开始，因为在 KunQuant 0.1.11 中，非零的起点会让所有截面算子给出错误结果。标的数量必须是 SIMD 块宽度的整数倍。在编写本页的机器上，13 个标的的面板会以下面这个 KunQuant 错误失败，解决办法是补齐或裁剪标的集合。

```python
>>> symbols13 = [f"S{i:02d}" for i in range(10)] + ["PENY", "ILQD", "DRPX"]
>>> config13 = write_store(tempfile.mkdtemp(), symbols13)
>>> config = FactorConfig(window=3, dataset=StockDataset(config13), mode="batch",
...                       data_columns=("adjClose",), njobs=4)
>>> UniverseFilteredFactor(RankClose(config), window=3).cal()
Traceback (most recent call last):
  ...
RuntimeError: Bad shape at adjClose
```

流式模式下，传给 `cal_stream()` 的每根 bar 的 `data` 字典，除了 `config.data_columns` 之外，还必须包含原始 `close` 和 `volume` 数组；继承来的推送只发送 `data_columns`，缺少键会抛出 `ValueError`。截面算子之上的时序算子，在标的重新入池之后的一整个窗口内都是 NaN，因为标的出池期间它的输入是 NaN。

## 另请参阅

`constituent` 指南介绍指数成分面板，它回答的是另一个问题，可以与这个过滤器组合使用。`factor` 指南介绍 `FactorKunQuant` 和算子，`model` 介绍如何把包装后的因子交给模型，`backtest` 介绍调仓以及仓位如何被平掉。类文档字符串：`quantlab.factor.universe_filter.UniverseFilteredFactor`。
