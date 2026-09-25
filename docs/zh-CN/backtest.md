# 回测（Backtesting）

[English](../backtest.md) | 简体中文

回测拿一个训练好的收益模型和一份价格数据集，展示模型的预测如果拿来交易会得到什么结果。模型对每个标的、每根 bar 给出一个分数，选股规则把分数变成目标权重，模拟引擎按这些权重成交并记录净值曲线。每次运行都会写出一个运行目录，里面有权重、净值曲线、指标、HTML 报告，以及重建这次运行所需的配置。

主要的类有：`BaseBacktester`（`quantlab/base/backtest.py`）、vectorbt 引擎 `VectorBtBacktester`（`quantlab/backtest/engine_vectorbt.py`）、选股规则 `CrossSectionTopNSelector`（`quantlab/backtest/selection.py`），以及美股回测器 `USEquityCrossectionSelectStockVectorBt`（`quantlab/backtest/us_equity.py`）。

## 前置条件

在仓库根目录用 `uv run python` 运行示例。在 macOS 上，同一进程导入 torch 或 xgboost 之前要设置 `OMP_NUM_THREADS=1`，并设置 `WANDB_MODE=disabled` 关闭实验跟踪。

回测需要一份价格数据集，其存储里有 `adjOpen` 和 `adjClose` 两列；还需要一个模型，checkpoint 由 `train()` 或 `train_cv()` 写出。下面的会话使用一套合成数据：六个标的、一个因子、一个标签，以及一个无需拟合的模型，它的分数就是过去一根 bar 的收益率。最后一个标的 `FFF` 从第 36 根 bar 起不再有价格。把下面的代码保存为 `demo_parts.py`。

<details>
<summary>demo_parts.py</summary>

```python
"""合成价格、一个因子、一个标签，以及一个无需拟合的模型。"""
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.base.config import DatasetConfig, MLConfig, PolarsFactorConfig
from quantlab.base.factor import FactorPolars
from quantlab.base.model import MLModel
from quantlab.dataset.stock import StockDataset

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def write_price_store(root, n_bars=60, delist=None):
    """写出日频复权价格 Zarr；delist={"FFF": 36} 表示 FFF 从第 36 根 bar 起不再有价格。"""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    shape = (n_bars, len(SYMBOLS))
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, shape), axis=0))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(rng.normal(0, 0.01, shape))
    for symbol, bar in (delist or {}).items():
        close[bar:, SYMBOLS.index(symbol)] = np.nan
        open_[bar:, SYMBOLS.index(symbol)] = np.nan
    dims = ["timestamp", "symbol"]
    xr.Dataset(
        {"adjOpen": (dims, open_), "adjClose": (dims, close)},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=n_bars), "symbol": SYMBOLS},
    ).to_zarr(root / "prices.zarr", mode="w")
    return DatasetConfig(
        raw_data_dir_path=str(root / "raw"), zarr_file_path=str(root / "prices.zarr"),
        market="us_equity", frequency="1d",
    )


def prices_of(cfg):
    return StockDataset(dataclasses.replace(cfg))


class PastReturn(FactorPolars):
    """特征 past_ret_1：单 bar 收益率。"""

    def _get_factor_lazyframe(self, lf):
        c = pl.col("adjClose")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((c / c.shift(1).over("symbol") - 1).alias("past_ret_1"))
                .select(["timestamp", "symbol", "past_ret_1"]))

    def _get_features(self, data):
        return data


class ForwardReturn(FactorPolars):
    """标签 fwd_ret_1：下一根 bar 的收益率。"""

    def _get_factor_lazyframe(self, lf):
        c = pl.col("adjClose")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((c.shift(-1).over("symbol") / c - 1).alias("fwd_ret_1"))
                .select(["timestamp", "symbol", "fwd_ret_1"]))

    def _get_labels(self, data):
        return data


class MomentumHead(MLModel):
    """直接预测第一个特征，所以分数就是过去收益率。"""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def make_model(root, cfg, days, train_end=39):
    day = lambda i: str(days[i].date())
    factor = PastReturn(PolarsFactorConfig(window=5, dataset=prices_of(cfg)))
    label = ForwardReturn(PolarsFactorConfig(
        window=0, dataset=prices_of(cfg), kwargs={"n_forward_periods": 1}))
    return MomentumHead(MLConfig(
        factors=[factor], labels=[label], model_save_dir=str(root / "models"),
        factor_data_strategy="cal", label_data_strategy="cal", val_size=0.0,
        start_date=day(0), end_date=day(len(days) - 1),
        train_start=day(0), train_end=day(train_end),
        test_start=day(train_end + 1), test_end=day(len(days) - 1),
    ))


def train_checkpoint(model):
    model.collect()
    return model.train()


def train_cv_project(model, train_periods):
    model.collect()
    model.train_cv(train_periods=train_periods, gap_periods=0)
    return next(Path(model.config.model_save_dir).rglob("cv_folds.json")).parent
```

</details>

## 基础

### 一次运行做了什么

`BaseBacktester.run()` 在 `start_date` 到 `end_date` 的窗口上回测一个模型。它加载 checkpoint（或先训练模型），在窗口上计算特征，为每个标的、每根 bar 预测分数，向具体的回测器类要目标权重，模拟成交，计算指标，最后写出运行目录。`run_cv()` 对一次 `train_cv` 的每一折做同样的事，并把各折拼接成一条曲线。

第一个会话先训练一个 checkpoint，然后回测一条规则：持有分数最高的两个标的，每五根 bar 调仓一次。日志输出到 stderr，这里没有显示。

```python
>>> import dataclasses, json, tempfile
>>> from pathlib import Path
>>> import pandas as pd
>>> import xarray as xr
>>> from demo_parts import *
>>> from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
>>> from quantlab.base.config import CrossSectionBacktestConfig
>>> root = Path(tempfile.mkdtemp())
>>> cfg = write_price_store(root, delist={"FFF": 36})
>>> days = pd.bdate_range("2024-01-01", periods=60)
>>> checkpoint = train_checkpoint(make_model(root / "train", cfg, days))
>>> backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
...     price_dataset=prices_of(cfg),
...     model=make_model(root / "backtest", cfg, days),
...     model_mode="load",
...     checkpoint=str(checkpoint),
...     start_date="2024-02-12",
...     end_date="2024-03-22",
...     output_dir=str(root / "runs"),
...     rebalance_periods=5,
...     direction="long_only",
...     top_n=2,
... ))
>>> result = backtester.run()
>>> sorted(p.name for p in result.run_dir.iterdir())
['config.json', 'equity.zarr', 'fingerprint.json', 'liquidations.json', 'metrics.json', 'report.html', 'weights.zarr']
```

运行目录位于 `output_dir` 之下，返回的结果里有预测、权重、模拟结果和指标。

### 目标权重契约

权重是 `(timestamp, symbol)` 上的 `weight` 变量。每一行要么全是 NaN，表示这根 bar 不调仓、保持原有仓位；要么全是有限值，表示把组合调整到这些占组合价值的比例。调仓行的总敞口（绝对权重之和）不超过 1。调仓 bar 上没被选中的标的权重是 `0.0`，不能是 NaN。

```python
>>> result.weights["weight"].to_pandas().iloc[:7].round(2)
symbol      AAA  BBB  CCC  DDD  EEE  FFF
timestamp                               
2024-02-12  0.0  0.0  0.5  0.0  0.0  0.5
2024-02-13  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-14  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-15  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-16  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-19  0.0  0.5  0.0  0.5  0.0  0.0
2024-02-20  NaN  NaN  NaN  NaN  NaN  NaN
```

窗口的第一根 bar 调仓，之后每隔 `rebalance_periods` 根 bar 调仓一次。最后一根 bar 从不调仓，因为在它上面形成的信号在窗口内已经没有下一根 bar 可供成交。`direction="long_short"` 时，分数最高的 `top_n` 个标的各得 `+0.5/top_n`，分数最低的 `top_n` 个各得 `-0.5/top_n`。

### t 时刻的信号在 t+1 时刻成交

在 bar `t` 形成的权重行，按 bar `t + 1` 的成交价执行。美股的成交价是复权开盘价（`adjOpen`），组合按复权收盘价（`adjClose`）估值。下面第一批订单出现在 2024-02-13，也就是第一次调仓后的下一根 bar。买入价等于当天开盘价加上默认的 0.05% 滑点。

```python
>>> result.simulation.orders.to_dataframe().head(3)
       timestamp symbol         size      price        fees  side
order                                                            
0     2024-02-13    CCC  9465.316230  52.850849  250.125000   Buy
1     2024-02-13    FFF  8221.655639  60.723809  249.625125   Buy
2     2024-02-20    FFF  8221.655639  62.012933  254.924491  Sell
>>> adj_open = xr.open_zarr(root / "prices.zarr")["adjOpen"]
>>> float(adj_open.sel(timestamp="2024-02-13", symbol="CCC"))
52.82443690819098
```

手续费和滑点（`fees` 与 `slippage`，默认都是 0.0005）按每笔成交额的比例收取。目标百分比以其执行那根 bar 的成交价所对应的组合价值为基数。

### 退市的持仓

模拟之前，两列价格都会做前向填充。某标的在一次调仓后被持有，而下一根 bar 上没有原始成交价，就会在那根 bar 上按最后已知价格卖出，其余标的照常调仓，这次卖出会记为一条强制平仓记录。`FFF` 从 2024-02-20 起没有价格，并且在第一个组合里，所以出现了这条记录。选股规则从不选择下一根 bar 没有价格的标的，所以上面第二个组合里没有 `FFF`。

```python
>>> result.simulation.liquidations
[{'symbol': 'FFF', 'axis_symbol': 'FFF', 'signal_timestamp': Timestamp('2024-02-19 00:00:00'), 'fill_timestamp': Timestamp('2024-02-20 00:00:00'), 'price': 62.04395518050185}]
```

窗口开头没有价格、且从未被持有的标的，被视为尚未上市，价格出现后正常交易。

### 预热

模型的因子需要 `start_date` 之前的历史。回测器在窗口之前多读取的 bar 数，等于模型各因子 `window` 中的最大值，按价格 bar 数计算，而不是按日历天数。价格日历不够长时，预热起点被限制在第一根 bar，并输出一条给出缺口大小的警告。预测恰好覆盖窗口内的 bar；没有预测的价格标的分数为 NaN，不会被选中。

### 样本内与样本外

模型在 `train_start` 到 `train_end` 的 bar 上训练。它的标签向前看 `n_forward_periods` 根 bar，所以有效训练窗口比 `train_end` 多延伸这么多根 bar。回测窗口里落在有效训练窗口内的 bar 是样本内，其余是样本外。load 模式下，训练日期取自 checkpoint 旁边的 `config.json`。回测窗口与训练窗口重叠时，运行会记录一条警告并继续。

```python
>>> m = result.metrics
>>> m["training_window"], m["in_sample_range"], m["out_of_sample_ranges"]
(('2024-01-01', '2024-02-26'), ('2024-02-12', '2024-02-26'), [('2024-02-27', '2024-03-22')])
```

各部分来自同一次连续的模拟，所以资金和持仓会跨过边界延续。`whole` 是整个窗口的引擎统计；`in_sample` 和 `out_of_sample` 是各自 bar 上基于收益序列的统计，外加成交笔数和换手。

```python
>>> for part in ("whole", "in_sample", "out_of_sample"):
...     print(part, round(m[part]["Total Return [%]"], 2), round(m[part]["Sharpe Ratio"], 2), m[part]["order_count"])
whole -5.85 -2.32 19
in_sample -1.09 -0.79 6
out_of_sample -4.82 -3.76 13
>>> list(m["whole"])[:6]
['Start', 'End', 'Period', 'Start Value', 'End Value', 'Total Return [%]']
>>> round(m["whole"]["turnover"]["mean_per_rebalance"], 2)
1.34
```

换手是某根 bar 的单边成交额除以成交前的组合价值，所以从现金一次性建仓约为 1，整本书全部换掉约为 2。这里的分数来自随机游走收益，负收益没有任何含义。

### 运行目录

每次运行在 `output_dir` 下写一个新目录 `{ClassName}_{timestamp}`。文件先写入一个隐藏的暂存目录，全部写完后才改名，所以 `output_dir` 里只会有完整的运行。

| 文件 | 内容 |
| --- | --- |
| `config.json` | 配置，嵌套着价格数据集和模型，以及数据指纹。 |
| `weights.zarr` | `(timestamp, symbol)` 上的目标权重。 |
| `equity.zarr` | `timestamp` 上的组合 `value` 与每根 bar 的 `returns`。 |
| `metrics.json` | 与 `result.metrics` 相同的映射；NaN 和无穷大写成 null。 |
| `liquidations.json` | 强制平仓记录。 |
| `fingerprint.json` | 本次运行读取的价格数据和因子数据的摘要。 |
| `report.html` | 净值、回撤、月度收益图表，指标表和备注。 |

## 常见任务

### 在回测中训练模型

`model_mode="train"` 时，模型先按自己配置的日期训练，不需要 `checkpoint`。回测窗口不会改变训练日期。本次运行写出的 checkpoint 记录在 `metrics["trained_checkpoint"]` 中。这个会话同时换成了每侧一个标的的多空组合。

```python
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=make_model(root / "train_mode", cfg, days),
...     model_mode="train", checkpoint=None, direction="long_short", top_n=1,
... )).run()
>>> Path(trained.metrics["trained_checkpoint"]).name
'MomentumHead_total.joblib'
>>> trained.weights["weight"].to_pandas().iloc[0]
symbol
AAA    0.0
BBB   -0.5
CCC    0.5
DDD    0.0
EEE    0.0
FFF    0.0
Name: 2024-02-12 00:00:00, dtype: float64
```

### 回放一次交叉验证

`train_cv` 会为每个滚动折写一个 checkpoint，并在项目目录里写一个 `cv_folds.json` 清单。`run_cv()` 读取清单，用各折自己的 checkpoint 回测该折的测试段，再把拼接后的权重整体模拟一次。`cv_project_dir` 指向项目目录，`model_mode` 必须是 `"load"`。只使用测试段落在窗口内的折，这些测试段必须逐 bar 首尾相接。

```python
>>> cfg2 = write_price_store(root / "cv", n_bars=80)
>>> days2 = pd.bdate_range("2024-01-01", periods=80)
>>> project_dir = train_cv_project(make_model(root / "cv_train", cfg2, days2, train_end=29), 30)
>>> manifest = json.loads((project_dir / "cv_folds.json").read_text())
>>> len(manifest["folds"]), manifest["folds"][0]["test_start"][:10], manifest["folds"][-1]["test_end"][:10]
(8, '2024-02-12', '2024-04-17')
>>> cv_config = dataclasses.replace(
...     backtester.config,
...     price_dataset=prices_of(cfg2),
...     model=make_model(root / "cv_backtest", cfg2, days2, train_end=29),
...     cv_project_dir=str(project_dir),
...     start_date=str(days2[30].date()),
...     end_date=str(days2[77].date()),
...     rebalance_periods=2,
... )
>>> cv = USEquityCrossectionSelectStockVectorBt(cv_config).run_cv()
>>> len(cv.folds), dict(cv.weights.sizes), sorted(cv.metrics)
(8, {'timestamp': 48, 'symbol': 6}, ['folds', 'notes', 'stitched'])
>>> sorted(p.name for p in (cv.run_dir / "folds").iterdir())[:2]
['fold_0', 'fold_1']
```

运行目录顶层的文件描述的是拼接后的曲线，`folds/fold_{i}/` 存放每一折自己的权重和净值。拼接曲线是一次模拟，所以资金会跨折延续。每一折另有一次从 `init_cash` 起步的独立模拟，各折的指标来自这些独立模拟。这里的标签向前看一根 bar，所以每一折的第一根 bar 对该折的模型来说是样本内。

```python
>>> stitched = cv.metrics["stitched"]
>>> stitched["in_sample_ranges"][:2], stitched["out_of_sample_ranges"][:2]
([('2024-02-12', '2024-02-12'), ('2024-02-20', '2024-02-20')], [('2024-02-13', '2024-02-19'), ('2024-02-21', '2024-02-27')])
>>> round(stitched["whole"]["Total Return [%]"], 2), round(cv.folds[0]["metrics"]["whole"]["Total Return [%]"], 2)
(-3.11, -1.62)
```

### 从配置重建一次运行

`config.json` 用点分导入路径记录每个类，所以 `load_backtester_from_config` 能重建出相同的回测器，包括它的价格数据集和模型，`run()` 会把这次回测重做一遍，写入新目录。如果自原始运行以来数据发生了变化，重建的运行会对每个变化的数据集记录一条警告并继续。

```python
>>> from quantlab.utils.module import load_backtester_from_config
>>> config = json.loads((result.run_dir / "config.json").read_text())
>>> config["name"], config["direction"], config["top_n"]
('quantlab.backtest.us_equity.USEquityCrossectionSelectStockVectorBt', 'long_only', 2)
>>> again = load_backtester_from_config(config).run()
>>> again.metrics["whole"] == result.metrics["whole"]
True
>>> again.run_dir == result.run_dir
False
```

这些类必须能按点分路径导入。定义在脚本里的类名为 `__main__.X`，换一个进程就找不到，所以数据集、因子、模型和回测器应放在模块里。train 模式写出的配置在重建时会重新训练；要重放同一个模型，请把 `model_mode` 设为 `"load"`，并把 `checkpoint` 设为记录下来的 `trained_checkpoint`。

## 扩展

新的选股规则是 `VectorBtBacktester` 的子类，需要三个成员：`config_cls`、`MARKET` 和 `_generate_signals(predictions, prices)`。该方法返回一个数据集，其 `weight` 变量满足上面的契约。`predictions` 与 `prices` 共用同一套 `(timestamp, symbol)` 坐标轴。下面的规则让每个可交易标的的权重与其正分数成比例，没有正分数时空仓。它复用了 `rebalance_mask` 和 `US_EQUITY_MARKET` 的价格约定。保存为 `score_weighted.py`。

```python
"""一条新的选股规则：多头权重与正分数成比例。"""

import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import rebalance_mask
from quantlab.backtest.us_equity import US_EQUITY_MARKET
from quantlab.base.config import BacktestConfig


class ScoreWeightedBacktester(VectorBtBacktester):
    config_cls = BacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions, prices):
        label = self.config.model.get_label_names()[0]
        scores = predictions[label].transpose("timestamp", "symbol")
        # 下一根 bar 没有价格的标的无法成交，不参与选择。
        next_fill = prices[self.MARKET.fill_price_column].shift(timestamp=-1)
        positive = scores.where(next_fill.notnull()).clip(min=0).fillna(0.0)
        total = positive.sum("symbol")
        weight = (positive / total.where(total > 0)).fillna(0.0)  # 没有正分数则空仓
        rebalance = xr.DataArray(
            rebalance_mask(prices.sizes["timestamp"], self.config.rebalance_periods),
            dims="timestamp", coords={"timestamp": prices.timestamp},
        )
        return weight.where(rebalance).to_dataset(name="weight")  # 非调仓 bar 为 NaN
```

它的配置类是 `BacktestConfig`，所以不需要 `direction` 和 `top_n`。

```python
>>> from score_weighted import ScoreWeightedBacktester
>>> from quantlab.base.config import BacktestConfig
>>> custom = ScoreWeightedBacktester(BacktestConfig(
...     price_dataset=prices_of(cfg),
...     model=make_model(root / "custom", cfg, days),
...     model_mode="load",
...     checkpoint=str(checkpoint),
...     start_date="2024-02-12",
...     end_date="2024-03-22",
...     output_dir=str(root / "runs"),
...     rebalance_periods=5,
... ))
>>> w = custom.run().weights["weight"].to_pandas()
>>> w.iloc[[0, 1, 5]].round(3)
symbol        AAA    BBB    CCC    DDD  EEE    FFF
timestamp                                         
2024-02-12  0.000  0.000  0.541  0.000  0.0  0.459
2024-02-13    NaN    NaN    NaN    NaN  NaN    NaN
2024-02-19  0.044  0.767  0.000  0.189  0.0  0.000
```

如果想沿用 top-N 规则、只换分数，`CrossSectionTopNSelector(direction, top_n).select(scores, next_fill_price, rebalance)` 接受任意分数面板并返回同样的 `weight` 数据集。换一个市场就是换一个 `MarketSpec`，其中有自己的成交价列、估值价列和年化常数。

## 注意事项

回测不模拟借券费用或做空融资成本，所以空头一侧的收益偏乐观；指标里的 `notes` 也有说明。交易统计采用持仓视角：一笔交易是某个标的从建仓到清仓的一次完整往返，把持仓减回目标权重不算一笔已平仓交易。`order_count` 是成交笔数。

`benchmark_dataset` 是保留字段，传入会抛出 `NotImplementedError`。具体的回测器必须设置 `MARKET`。load 模式下 `run()` 需要 `checkpoint`，`run_cv()` 需要 `cv_project_dir` 和 `model_mode="load"`。价格存储旁没有 CRSP ticker 附属文件时，回测器会记录一条警告，说明改用坐标轴上的标的名作为标签，运行本身不受影响。

配置类用错时：

```python
>>> USEquityCrossectionSelectStockVectorBt(custom.config)
Traceback (most recent call last):
  ...
TypeError: USEquityCrossectionSelectStockVectorBt requires a CrossSectionBacktestConfig, got BacktestConfig
```

模型没有声明的 score 标签：

```python
>>> USEquityCrossectionSelectStockVectorBt(dataclasses.replace(backtester.config, score_label="fwd_ret_5"))
Traceback (most recent call last):
  ...
ValueError: score_label 'fwd_ret_5' is not one of the model's labels ['fwd_ret_1']
```

没有 `cv_project_dir` 就调用 `run_cv()`：

```python
>>> backtester.run_cv()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: run_cv() requires config.cv_project_dir, the train_cv project directory holding cv_folds.json
```

保存的配置缺少字段时，会被拒绝，而不是用当前默认值补上：

```python
>>> del config["top_n"]
>>> load_backtester_from_config(config)
Traceback (most recent call last):
  ...
ValueError: quantlab.backtest.us_equity.USEquityCrossectionSelectStockVectorBt config is missing field(s) ['top_n']; refusing to fill them from the current dataclass defaults, which may differ from the values the stored backtest ran with (...)
```

如果 `cv_folds.json` 中间缺了一折，`run_cv()` 拒绝跨缺口拼接，报错信息包含 `fold test segments are not contiguous: gap between fold 2 ending 2024-03-06 and fold 4 starting 2024-03-15; 6 price bar(s) in between belong to no fold, so a stitched out-of-sample curve would silently skip them`。恢复清单，或者把 `start_date` 与 `end_date` 收窄到一段连续的折。

## 另请参阅

- [model](model.md)：`train`、`train_cv`、`cv_folds.json` 与 `predict_panel`。
- [dataset](dataset.md)：价格数据集；[factor](factor.md)：模型使用的因子和标签。
- [backend](backend.md)：权重和净值曲线所写入的 Zarr 存储。
- `quantlab/base/backtest.py` 中的 `BaseBacktester`、`BacktestResult`、`CVBacktestResult`、`MarketSpec`；`quantlab/base/config.py` 中的 `BacktestConfig` 与 `CrossSectionBacktestConfig`；`quantlab/utils/module.py` 中的 `load_backtester_from_config`。
