# Frame API

[English](../api.md) | 简体中文

`quantlab.api` 面向已经把行情数据放在 pandas 或 polars DataFrame 里的用户：只想用 quantlab 的某一项功能，例如 Alpha158 因子、前瞻收益标签、因子报告或对自己信号的回测，而不想写 Zarr store、构造配置对象，也不想先学数据集类。每个函数接收一个 *frame*，即长格式的 DataFrame，每个 `timestamp` 与 `symbol` 一行，返回同一个库的 frame。除非你要求，不写任何文件，也不启动 Weights & Biases。

| 函数 | 输入 | 返回 |
|------|------|------|
| `compute_factors(frame, factor)` | bar | 每个因子一列 |
| `forward_returns(frame, price=, span=, delay=)` | bar | 前瞻收益标签 |
| `analyze_factors(factors, returns)` 或 `analyze_factors(factors, prices=)` | 因子，以及前瞻收益或 bar | `FactorReport` |
| `backtest(prices, weights=)` 或 `backtest(prices, scores=, top_n=)` | bar，以及权重或分数 | `BacktestReport` |

在内部，每个 frame 都变成一个 `FrameDataset`，即保存在内存中的市场数据集，各函数在它上面运行库自身的因子、标签、因子报告和回测器。同一个 `FrameDataset` 也可以直接用于完整流水线（见[从 frame 到完整流水线](#从-frame-到完整流水线)）。

## 前置条件

在仓库根目录用 `uv run python` 运行示例。本页的示例构成一个 Python 会话：每个代码块都接着前面的代码块。计算因子和标签时会编译 KunQuant 代码，每次调用需要几秒，并在标准错误输出上打印计时信息；这些行本页不展示。

## 第一个会话

会话从 8 个标的、120 个工作日的合成日频 bar 开始。frame 使用标准列名，所以不需要映射。

```python
>>> import numpy as np
>>> import pandas as pd
>>> import quantlab.api as qa
>>> rng = np.random.default_rng(42)
>>> bars = pd.bdate_range("2024-01-01", periods=120)
>>> symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
>>> close = 50 * np.exp(rng.normal(0.0003, 0.02, (120, 8)).cumsum(axis=0))
>>> open_ = close * np.exp(rng.normal(0, 0.005, (120, 8)))
>>> prices = pd.DataFrame({
...     "timestamp": np.repeat(bars, 8), "symbol": symbols * 120,
...     "open": open_.ravel(), "high": (np.maximum(open_, close) * 1.01).ravel(),
...     "low": (np.minimum(open_, close) * 0.99).ravel(), "close": close.ravel(),
...     "volume": rng.integers(100_000, 1_000_000, 960).astype(float),
... })
>>> prices.head(3)
   timestamp symbol       open       high        low      close    volume
0 2024-01-01    AAA  50.729202  51.236494  49.817534  50.320741  737969.0
1 2024-01-01    BBB  49.017553  49.507729  48.495596  48.985450  167515.0
2 2024-01-01    CCC  50.517525  51.279054  50.012350  50.771340  732593.0
```

计算 Alpha158 因子，按各因子对标的 5 根 bar 前瞻收益的排序能力排名，再把最好的一个当作分数回测：持有分数最高的两个标的，每 5 根 bar 调仓一次。

```python
>>> factors = qa.compute_factors(prices, "alpha158")
>>> factors.shape  # timestamp, symbol and 169 factor columns
(960, 171)
>>> report = qa.analyze_factors(factors, prices=prices, span=5, plot=False)
>>> summary = report.summary().sort_values("rank_icir", ascending=False)
>>> summary[["factor", "rank_ic", "rank_icir", "turnover"]].head(3).round(4)
     factor  rank_ic  rank_icir  turnover
43    ROC60   0.1098     0.3219    0.0519
133  SUMN60   0.1049     0.3068    0.0425
58    MAX60   0.1100     0.2857    0.1065
>>> best = summary["factor"].iloc[0]
>>> result = qa.backtest(prices, scores=factors[["timestamp", "symbol", best]],
...                      top_n=2, rebalance_periods=5)
>>> result
BacktestReport(120 bars x 8 symbols, total return -7.16%)
>>> round(result.metrics["whole"]["Sharpe Ratio"], 4)
-0.9251
```

因子是在回测所用的同一段 bar 上选出来的，所以这次回测是样本内的。本页其余部分逐一介绍每一步。

## Frame

### 标准列

一个 frame 有 `timestamp` 列、`symbol` 列，以及每个字段一列。价格和成交量字段是 `open`、`high`、`low`、`close` 和 `volume`，加密货币因子集还需要 `amount`（成交额）。每个函数只读取它需要的列：`forward_returns` 读取 `price` 指定的列，`backtest` 读取 `fill` 和 `valuation` 指定的列。

`columns=` 把你的列名映射到标准列名，不需要先重命名 frame。

```python
>>> mine = pd.DataFrame({
...     "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
...     "ticker": ["AAA"] * 3 + ["BBB"] * 3,
...     "Open": [10.0, 11.0, 12.1, 20.0, 19.0, 19.0],
... })
>>> qa.forward_returns(mine, columns={"date": "timestamp", "ticker": "symbol", "Open": "open"})
   timestamp symbol  ret_1
0 2024-01-02    AAA    0.1
1 2024-01-02    BBB    0.0
2 2024-01-03    AAA    NaN
3 2024-01-03    BBB    NaN
4 2024-01-04    AAA    NaN
5 2024-01-04    BBB    NaN
```

### 长格式、宽格式与带索引的 frame

长格式，即每个 `(timestamp, symbol)` 一行，是所有函数都接受的格式。以 `(timestamp, symbol)` MultiIndex 为索引的 pandas frame 可以直接传入：

```python
>>> indexed = prices.set_index(["timestamp", "symbol"])
>>> qa.forward_returns(indexed).equals(qa.forward_returns(prices))
True
```

只含单个字段的输入，即 `backtest` 的 `weights` 和 `scores`、`analyze_factors` 的 `returns`，也可以是宽格式：每个时间戳一行，每个标的一列，时间戳放在 `timestamp` 列或 pandas `DatetimeIndex` 中。[回测](#回测)一节展示了两种格式。

### 输入规则

无论 frame 从哪里进入，都适用同一套规则：

- 重复的 `(timestamp, symbol)` 会报错，并列出前几个重复的组合；
- 缺少时间戳或标的的行会报错；
- 没有对应行的 `(timestamp, symbol)` 单元格变成 NaN，所以参差不齐的数据（上市、退市、假日）不需要先补齐；
- 带时区的时间戳转换为 UTC 并去掉时区；
- 标的转换为 `str`。

```python
>>> repeated = pd.concat([mine, mine.iloc[:1]])
>>> qa.forward_returns(repeated, columns={"date": "timestamp", "ticker": "symbol", "Open": "open"})
Traceback (most recent call last):
    ...
ValueError: forward_returns(price='open') has 1 duplicate (timestamp, symbol) pair(s), for example (2024-01-02 00:00:00, 'AAA'). Each pair must appear once; drop or aggregate the repeats first.
>>> qa.forward_returns(mine.assign(ticker=["AAA", None, "AAA", "BBB", "BBB", "BBB"]),
...                    columns={"date": "timestamp", "ticker": "symbol", "Open": "open"})
Traceback (most recent call last):
    ...
ValueError: forward_returns(price='open') has 1 row(s) with a missing timestamp or symbol, for example row 1: (2024-01-03 00:00:00, nan). Every row needs both; drop or fill those rows first.
>>> ragged = pd.DataFrame({
...     "timestamp": pd.to_datetime(["2024-01-02 14:30", "2024-01-02 14:30",
...                                  "2024-01-03 14:30"]).tz_localize("America/New_York"),
...     "symbol": [1, 2, 1],
...     "open": [10.0, 20.0, 11.0],
... })
>>> out = qa.forward_returns(ragged, delay=0)
>>> out
            timestamp symbol  ret_1
0 2024-01-02 19:30:00      1    0.1
1 2024-01-02 19:30:00      2    NaN
2 2024-01-03 19:30:00      1    NaN
3 2024-01-03 19:30:00      2    NaN
>>> out["symbol"].map(type).unique().tolist()
[<class 'str'>]
```

标的 `2` 在 1 月 3 日没有行，所以该单元格为 NaN，它的收益也是 NaN。

### 返回什么

结果以 frame 传入时所属的库返回：传入 pandas 返回 pandas，传入 polars 返回 polars。有多个 frame 输入的函数以第一个输入为准：`analyze_factors` 看 `factors`，`backtest` 看 `prices`。`compute_factors` 和 `forward_returns` 传 `as_xarray=True` 时，改为返回库自身在 `(timestamp, symbol)` 上的 `xarray.Dataset` 面板。

```python
>>> import polars as pl
>>> type(qa.forward_returns(pl.from_pandas(mine), columns={
...     "date": "timestamp", "ticker": "symbol", "Open": "open"})).__name__
'DataFrame'
>>> qa.forward_returns(prices, as_xarray=True)
<xarray.Dataset> Size: 5kB
Dimensions:    (timestamp: 120, symbol: 8)
Coordinates:
  * timestamp  (timestamp) datetime64[ns] 960B 2024-01-01 ... 2024-06-14
  * symbol     (symbol) object 64B 'AAA' 'BBB' 'CCC' 'DDD' ... 'FFF' 'GGG' 'HHH'
Data variables:
    ret_1      (timestamp, symbol) float32 4kB 0.008614 -0.02089 ... nan nan
```

## 计算因子

`compute_factors(frame, factor)` 在 frame 的每根 bar 上计算一个因子集，frame 完整网格上的每个 `(timestamp, symbol)` 返回一行，每个因子一列。`factor` 是一个简称或一个因子类。

| 简称 | 类 | 读取 | 标准化 |
|------|----|------|--------|
| `"alpha158"` | `Alpha158Stock` | `open`、`high`、`low`、`close`、`volume` | 每根 bar 上跨标的 z-score |
| `"alpha101"` | `Alpha101Stock` | `open`、`high`、`low`、`close`、`volume` | 每根 bar 上跨标的 z-score |
| `"alpha158_crypto"` | `Alpha158SpotKline` | 上述各列和 `amount` | 沿时间在 20 根 bar 上做 z-score |
| `"alpha101_crypto"` | `Alpha101SpotKline` | 上述各列和 `amount` | 沿时间在 20 根 bar 上做 z-score |

股票因子集针对经过拆股和分红复权的价格，VWAP 取 `(high + low + close) / 3`；加密货币因子集从 `amount` 读取成交额。需要基本面数据或因子收益序列的因子（`LiteratureAlpha`、`ResidualMomentumFF3`、`MarketFeatures`）没有简称。

整个 frame 一次算完，第一根 bar 之前没有历史，所以每个滚动窗口开头的几根 bar 是 NaN，而不是被丢弃。因子值是 float32。

```python
>>> factors[["timestamp", "symbol", "KMID", "STD5"]].head(3)
   timestamp symbol      KMID  STD5
0 2024-01-01    AAA -2.025160   NaN
1 2024-01-01    BBB -0.390322   NaN
2 2024-01-01    CCC  0.864862   NaN
>>> int(factors["STD5"].isna().sum())  # 4 warm-up bars x 8 symbols
32
>>> factors["KMID"].dtype
dtype('float32')
```

未知的简称会报错并列出有效的简称，缺少的列会被指出：

```python
>>> qa.compute_factors(prices, "alpha159")
Traceback (most recent call last):
    ...
ValueError: Unknown factor short name 'alpha159'. Valid names: 'alpha101', 'alpha158', 'alpha101_crypto', 'alpha158_crypto'; or pass a Factor subclass.
>>> qa.compute_factors(prices, "alpha158_crypto")
Traceback (most recent call last):
    ...
ValueError: 'alpha158_crypto' needs column(s) 'amount', which the data does not have. Present columns: ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']. Pass columns={'yours': 'amount'} to map one of yours onto it.
```

也可以不用简称，直接传入任意 `quantlab.base.factor.Factor` 子类。目录中某个类的子类读取该简称对应的列。其他类构建在一个数据集上，该数据集以标准列名持有 frame 的每一列。例如一个 Polars 因子：

```python
>>> from quantlab.factor.polars import FactorPolars
>>> class RelativeVolume(FactorPolars):
...     def _get_factor_lazyframe(self, lf):
...         volume = pl.col("volume")
...         return (
...             lf.sort(["symbol", "timestamp"])
...             .with_columns((volume / volume.rolling_mean(5).over("symbol") - 1.0)
...                           .alias("rel_volume_5"))
...             .select(["timestamp", "symbol", "rel_volume_5"])
...         )
>>> qa.compute_factors(prices, RelativeVolume).dropna().head(3)
    timestamp symbol  rel_volume_5
32 2024-01-05    AAA      0.177489
33 2024-01-05    BBB     -0.135541
34 2024-01-05    CCC     -0.555079
```

## 前瞻收益

`forward_returns(frame, price="open", span=1, delay=1)` 计算用来评判因子的标签：在第 `t` 根 bar 之后 `delay` 根 bar 建仓、持有 `span` 根 bar 的收益，两端都用 `price` 列的价格。

```text
label[t] = price[t + delay + span] / price[t + delay] - 1
```

`delay + span` 是标签的前瞻：第 `t` 根 bar 上的值要等第 `t + delay + span` 根 bar 收盘后才知道，frame 最后 `delay + span` 根 bar 是 NaN。默认值与库的 `Return` 标签一致：第 `t` 根 bar 上形成的信号在第 `t + 1` 根 bar 的开盘成交。`price` 没有备选列；请使用回测成交所用的列，让标签与成交一致。输出列名是 `ret_{span}`。`delay=0` 时在信号 bar 自身的价格建仓：下面 `AAA` 在 1 月 2 日的 0.10 就是 11.0 / 10.0 - 1。

```python
>>> labels = qa.forward_returns(mine, columns={"date": "timestamp", "ticker": "symbol",
...                                            "Open": "open"}, delay=0)
>>> labels
   timestamp symbol  ret_1
0 2024-01-02    AAA   0.10
1 2024-01-02    BBB  -0.05
2 2024-01-03    AAA   0.10
3 2024-01-03    BBB   0.00
4 2024-01-04    AAA    NaN
5 2024-01-04    BBB    NaN
>>> qa.forward_returns(prices, price="close", span=5).dropna().head(2)
   timestamp symbol     ret_5
0 2024-01-01    AAA  0.018679
1 2024-01-01    BBB -0.028371
>>> qa.forward_returns(mine, columns={"date": "timestamp", "ticker": "symbol"})
Traceback (most recent call last):
    ...
ValueError: forward_returns: price column 'open' is not in the frame. Present columns: ['timestamp', 'symbol', 'Open']. Pass price='Open' to use one of them, or columns={'yours': 'open'} to map one of yours onto it.
```

只要两个价格之一缺失，收益就是 NaN，最后几根 bar 也是如此。`binary=True` 返回库的 `BinaryReturn`，列名为 `ret_binary_{span}`：前瞻收益为正时取 1.0，为零或为负时取 0.0，收益为 NaN 时取 NaN，而不是 0.0。

```python
>>> gap = pd.DataFrame({
...     "timestamp": pd.bdate_range("2024-01-01", periods=5), "symbol": "AAA",
...     "open": [10.0, 11.0, np.nan, 10.0, 10.5],
... })
>>> returns = qa.forward_returns(gap, delay=0)
>>> binary = qa.forward_returns(gap, delay=0, binary=True)
>>> returns.merge(binary)
   timestamp symbol  ret_1  ret_binary_1
0 2024-01-01    AAA   0.10           1.0
1 2024-01-02    AAA    NaN           NaN
2 2024-01-03    AAA    NaN           NaN
3 2024-01-04    AAA   0.05           1.0
4 2024-01-05    AAA    NaN           NaN
```

这些值是 float32，与库的标签走同一条 KunQuant 计算路径；与 float64 收益比较时，请按 float32 精度比较。

## 分析因子

`analyze_factors(factors, ...)` 把每个因子列与前瞻收益配对，在两者共有的 bar 和标的上对每个配对运行库的因子报告：每根 bar 上的信息系数（IC，跨标的的 Pearson 相关和秩相关）及其统计量；按因子值分成 `quantiles` 个等数量分组后各组的平均前瞻收益；最高组减最低组的差；换手率和秩自相关。有两个或以上因子时，还会度量因子之间的相关性。

前瞻收益恰好来自以下一个来源：

- `prices=`：从 bar 计算，与 `forward_returns(prices, price=price, span=span, delay=delay)` 的算法相同。`span` 默认为 1。
- `returns`：你自己的收益，长格式（`timestamp`、`symbol` 和一个值列，其列名成为收益的名字）或宽格式（收益名为 `"returns"`）。此时必须给出 `span`，因为只有你知道这些收益的期限：它决定分组累计收益复利所用的每根 bar 收益率 `(1 + r) ** (1 / span) - 1`，以及 IC 的 Newey-West t 统计量所考虑的重叠。

```python
>>> momentum = factors[["timestamp", "symbol", "ROC5", "ROC20"]]
>>> own = qa.forward_returns(prices, span=5).pivot(index="timestamp", columns="symbol",
...                                                values="ret_5")
>>> qa.analyze_factors(momentum, own)
Traceback (most recent call last):
    ...
ValueError: analyze_factors: returns= needs span=, the number of bars your returns span; it sets how returns compound and the IC's Newey-West lags.
>>> by_returns = qa.analyze_factors(momentum, own, span=5, plot=False)
>>> by_prices = qa.analyze_factors(momentum, prices=prices, span=5, plot=False)
>>> by_returns
FactorReport(2 pairs: ROC5__returns, ROC20__returns)
>>> by_returns.summary().drop(columns="fret").equals(by_prices.summary().drop(columns="fret"))
True
```

`summary()` 每个因子与收益的配对一行：

| 列 | 含义 |
|----|------|
| `factor`、`fret` | 因子和前瞻收益 |
| `ic`、`rank_ic` | Pearson IC 均值和秩 IC 均值 |
| `icir`、`rank_icir` | 各自的均值除以标准差 |
| `long_short_return` | 每根 bar 最高分位减最低分位的前瞻收益均值，按收益的 span 计 |
| `turnover` | 最高与最低分位平均换手率的平均 |

```python
>>> print(by_prices.summary().round(4).to_string())
  factor   fret      ic  rank_ic    icir  rank_icir  long_short_return  turnover
0   ROC5  ret_5 -0.0115   0.0090 -0.0288     0.0234             0.0019    0.3681
1  ROC20  ret_5  0.0409   0.0301  0.1010     0.0702             0.0148    0.2688
```

`quantiles`（默认 5）决定每根 bar 的分组数，取值在 2 与最大截面之间。`plot=True`（默认）为每个配对画一张 matplotlib 图，放在 `report.figures` 中；报告较大时画图是主要开销，`plot=False` 时只在保存报告时才画图。`raw` 是库的 `FactorAnalysis`，包含全部指标和表格。`save(directory)` 把表格写成 CSV，把指标写成 `summary.json`，每个配对写一张 PNG。

```python
>>> import tempfile
>>> terciles = qa.analyze_factors(momentum, prices=prices, span=5, quantiles=3)
>>> sorted(terciles.figures)
['ROC20__ret_5', 'ROC5__ret_5']
>>> out = terciles.save(tempfile.mkdtemp())
>>> sorted(path.name for path in out.iterdir())
['ROC20__ret_5.png', 'ROC5__ret_5.png', 'config.json', 'factor_clusters.csv', 'factor_correlation.csv', 'factor_correlation.png', 'factor_correlation_pairs.csv', 'ic.csv', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv']
```

## 回测

`backtest(prices, ...)` 用库的 vectorbt 回测器在 `prices` 的 bar 上模拟一个组合。信号恰好是以下之一：

- `weights=`：目标权重，即每根 bar 之后每个标的占组合价值的比例，正为多头、负为空头，每根 bar 的总敞口（绝对权重之和）不超过 1；
- `scores=`：在每根 bar 上给标的排序的数值，越高越好，例如一个因子或模型的预测，需同时给出 `top_n=`。每 `rebalance_periods` 根 bar，分数最高的 `top_n` 个标的等权持有；`direction="long_short"` 时，分数最低的 `top_n` 个标的还会等权做空。一个标的有分数、且在该 bar 上有成交价时才可入选；没有成交价的持仓保持原权重，与库里的回测器一致。

第 `t` 根 bar 上形成的权重在第 `t + 1` 根 bar 以 `fill` 价格（默认 `open`）成交，组合按 `valuation` 价格（默认 `close`）估值。bar 间隔取时间戳之间最常见的间距。没有训练窗口，所以指标覆盖整个区间。

### 权重

权重可以是长格式，每个 `(timestamp, symbol)` 一行，带一个任意名字的值列；也可以是宽格式。权重 frame 可以省略一些内容：

- 在一根有其他标的权重的 bar 上，没有权重的标的取权重 0：长格式中省略的行，或宽格式中的 NaN 单元格；
- 完全没有权重的 bar 保持原有仓位：长格式中不出现，或宽格式中整行为 NaN。因此只列出调仓 bar 的 frame 就够了；要在某根 bar 上清仓，需显式写 0；
- 长格式中与同一根 bar 上的有限权重并列写出的 NaN 表示该标的在这根 bar 上保持持仓、不交易；权重总敞口超过 1 的 bar 会报错。

下面从第一根 bar 起等额持有 `AAA` 和 `BBB`，在第 60 根 bar 清仓。frame 只列出这两根 bar，第 60 根上只有 `AAA`：省略的 `BBB` 同样取权重 0。在同一个 frame 的宽格式透视表中，`BBB` 在该 bar 上的单元格是 NaN，回测结果相同：

```python
>>> weights = pd.DataFrame({
...     "timestamp": [bars[0], bars[0], bars[59]], "symbol": ["AAA", "BBB", "AAA"],
...     "weight": [0.5, 0.5, 0.0],
... })
>>> held = qa.backtest(prices, weights=weights)
>>> held
BacktestReport(120 bars x 8 symbols, total return -2.91%)
>>> held.orders
   timestamp symbol          size      price        fees  side
0 2024-01-02    AAA   9890.799541  50.577307  250.125000   Buy
1 2024-01-02    BBB  10357.098123  48.203681  249.625125   Buy
2 2024-03-25    BBB  10357.098123  47.903232  248.069237  Sell
3 2024-03-25    AAA   9890.799541  48.053934  237.645913  Sell
>>> wide = weights.pivot(index="timestamp", columns="symbol", values="weight")
>>> wide
symbol      AAA  BBB
timestamp           
2024-01-01  0.5  0.5
2024-03-22  0.0  NaN
>>> qa.backtest(prices, weights=wide).equity.equals(held.equity)
True
>>> bad = pd.DataFrame({"timestamp": bars[0], "symbol": ["AAA", "BBB"], "weight": [0.8, 0.4]})
>>> qa.backtest(prices, weights=bad)
Traceback (most recent call last):
    ...
ValueError: WeightsVectorBt: weight row at 2024-01-01 has gross exposure 1.2000000000000002 > 1
```

### 分数

使用分数时必须给出 `top_n`，`direction` 选择方向。某根 bar 上没有分数的标的（省略的行或 NaN 单元格）在该 bar 上不会被选中。

```python
>>> roc = factors[["timestamp", "symbol", "ROC20"]]
>>> long_short = qa.backtest(prices, scores=roc, top_n=2, direction="long_short",
...                          rebalance_periods=5)
>>> long_short.weights[long_short.weights["timestamp"] == bars[20]]
     timestamp symbol  weight
160 2024-01-29    AAA    0.25
161 2024-01-29    BBB    0.00
162 2024-01-29    CCC   -0.25
163 2024-01-29    DDD   -0.25
164 2024-01-29    EEE    0.00
165 2024-01-29    FFF    0.00
166 2024-01-29    GGG    0.00
167 2024-01-29    HHH    0.25
>>> qa.backtest(prices, scores=roc)
Traceback (most recent call last):
    ...
ValueError: scores= needs top_n=, the number of names held per side.
```

### 交易假设

`fees` 和 `slippage` 是每笔交易的比例成本，默认各为 0.0005；`init_cash` 是初始资金，默认 1,000,000，均与库的 `BacktestConfig` 相同。`fill` 和 `valuation` 指定价格列。`market` 决定比率类指标的年化方式：`"equity"` 为每年 252 个交易日、每天 390 分钟，`"crypto"` 为每年 365 天、每天 1,440 分钟。`trading_days_per_year` 和 `session_minutes_per_day` 可以分别覆盖其中一项。

```python
>>> costless = qa.backtest(prices, weights=weights, fees=0.0, slippage=0.0, init_cash=10_000.0)
>>> costless.equity.head(3)
   timestamp         value
0 2024-01-01  10000.000000
1 2024-01-02   9976.079154
2 2024-01-03   9920.937496
>>> at_close = qa.backtest(prices, weights=weights, fill="close")
>>> at_close.orders[["timestamp", "symbol", "price"]]
   timestamp symbol      price
0 2024-01-02    AAA  50.344088
1 2024-01-02    BBB  48.195339
2 2024-03-25    BBB  47.816942
3 2024-03-25    AAA  48.213889
>>> [round(qa.backtest(prices, weights=weights, market=m).metrics["whole"]["Sharpe Ratio"], 4)
...  for m in ("equity", "crypto")]
[-0.3192, -0.3842]
```

### 基准

`benchmark=` 接收单个标的的 bar，需带 `fill` 和 `valuation` 列，在同样的 bar 上以同样的成本和资金买入并持有。它在指标中增加 `"benchmark"` 和 `"relative"`（超额收益与超额回撤统计），并在报告中加入基准曲线。

```python
>>> index = prices.groupby("timestamp", as_index=False)[["open", "close"]].mean()
>>> index["symbol"] = "INDEX"
>>> compared = qa.backtest(prices, weights=weights, benchmark=index)
>>> sorted(compared.metrics)
['benchmark', 'execution', 'notes', 'relative', 'whole']
>>> compared.benchmark.tail(2)
     timestamp          value   returns
118 2024-06-13  967050.957213 -0.016379
119 2024-06-14  970206.298863  0.003263
```

### 报告

`BacktestReport` 中的 frame 属于 `prices` 传入时所属的库：

| 属性 | 内容 |
|------|------|
| `equity` | `timestamp`、`value`：每根 bar 之后的组合价值 |
| `returns` | `timestamp`、`returns`：每根 bar 上的组合收益 |
| `weights` | `timestamp`、`symbol`、`weight`：模拟所用的目标权重，保持仓位的 bar 上为 NaN |
| `orders` | 每次成交一行：`timestamp`、`symbol`、`size`、`price`、`fees`、`side` |
| `trades` | 从建仓到清仓的每个来回一行：`symbol`、`entry_timestamp`、`exit_timestamp`、`pnl`、`return`、`status` |
| `metrics` | 一个 dict：`"whole"` 统计量和 `"notes"`；有基准时还有 `"benchmark"` 和 `"relative"` |
| `benchmark` | 基准的 `timestamp`、`value`、`returns`，没有基准时为 `None` |
| `raw` | 库的 `BacktestResult`，全部是 xarray |

```python
>>> held.trades
  symbol entry_timestamp exit_timestamp           pnl    return  status
0    AAA      2024-01-02     2024-03-25 -25445.945655 -0.050866  Closed
1    BBB      2024-01-02     2024-03-25  -3609.470215 -0.007230  Closed
>>> {k: round(held.metrics["whole"][k], 4) for k in ("Total Return [%]", "Max Drawdown [%]")}
{'Total Return [%]': -2.9055, 'Max Drawdown [%]': 11.0196}
```

`plot()` 以 plotly 图形返回库报告中 Performance 标签页的图表：净值、回撤和月度收益，有基准时基准曲线并列其中（报告页面把相对基准的超额单独放在另一个标签页）。对它调用 `.show()` 即可显示。

```python
>>> figure = compared.plot()
>>> sorted({trace.name for trace in figure.data if trace.name})
['benchmark_drawdown', 'benchmark_equity', 'benchmark_monthly_return', 'deepest_drawdown_end', 'deepest_drawdown_valley', 'drawdown', 'equity', 'monthly_return']
```

### 保留一次运行

默认不写任何文件。`output_dir=` 在该目录下写出库的运行目录，名为 `WeightsVectorBt_<timestamp>`；`report.save(directory)` 在事后写出同样的目录。运行目录包含 `config.json`、`weights.zarr`、`equity.zarr`、`metrics.json`、`settlements.json`、`fingerprint.json` 和 `report.html`，`inputs/` 下是价格和基准面板，`config.json` 以相对于运行目录的路径指向它们。运行目录可以移动；下面先移动它，再用两行代码从中重建这次运行：

```python
>>> import json
>>> import shutil
>>> from pathlib import Path
>>> from quantlab.backend import XrBackend
>>> from quantlab.utils.module import load_backtester_from_config
>>> kept = qa.backtest(prices, weights=weights, benchmark=index, output_dir=tempfile.mkdtemp())
>>> kept.raw.run_dir.name.startswith("WeightsVectorBt_")
True
>>> run_dir = Path(shutil.move(kept.raw.run_dir, tempfile.mkdtemp()))
>>> sorted(path.name for path in run_dir.iterdir())
['config.json', 'equity.zarr', 'fingerprint.json', 'inputs', 'metrics.json', 'report.html', 'settlements.json', 'weights.zarr']
>>> sorted(path.name for path in (run_dir / "inputs").iterdir())
['benchmark_dataset.zarr', 'price_dataset.zarr']
>>> rebuilt = load_backtester_from_config(json.loads((run_dir / "config.json").read_text()), run_dir=run_dir)
>>> replay = rebuilt.run_weights(XrBackend().read(run_dir / "weights.zarr").data)
>>> bool((replay.simulation.value == kept.raw.simulation.value).all())
True
>>> replay.run_dir.parent == run_dir.parent, replay.run_dir != run_dir
(False, True)
```

重放会写出自己的运行目录，因为重建出的配置保留了 `output_dir`。对没有传 `output_dir` 的报告，`save(directory)` 用同样的权重再模拟一次，写出运行目录并返回它：

```python
>>> saved = held.save(tempfile.mkdtemp())
>>> sorted(path.name for path in (saved / "inputs").iterdir())
['price_dataset.zarr']
>>> json.loads((saved / "metrics.json").read_text())["whole"]["Total Return [%]"] == held.metrics["whole"]["Total Return [%]"]
True
```

## 从 frame 到完整流水线

上面的函数各自提供一项功能。要在自己的 frame 上训练模型，或者在库接受市场数据集的任何地方（因子配置、回测的价格或基准数据集）使用它，就自己构造 `FrameDataset`：

```python
>>> from quantlab.dataset.memory import FrameDataset
>>> dataset = FrameDataset(mine, columns={"date": "timestamp", "ticker": "symbol", "Open": "open"})
>>> dataset.panel("2024-01-02", "2024-01-04")["open"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-02  10.0  20.0
2024-01-03  11.0  19.0
2024-01-04  12.1  19.0
```

`FrameDataset` 遵循上面的输入规则，并保留每一列自己的列名，所以要按读取它的因子所期望的名字命名各列（股票因子类读取 `adjOpen`、`adjClose` 等；`quantlab.api` 会替你完成这一重命名）。数据集指南中的[在内存中持有自己的 frame](dataset.md#在内存中持有自己的-frame) 完整介绍了它：在它上面计算因子、`resample()`、`to_zarr()`，以及回测运行目录如何保存其面板。之后的流水线见[因子](factor.md)、[模型](model.md)和[回测](backtest.md)指南。
