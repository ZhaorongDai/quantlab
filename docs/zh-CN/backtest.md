# 回测（Backtesting）

[English](../backtest.md) | 简体中文

回测拿一个训练好的收益模型和一份价格数据集，展示模型的预测如果拿来交易会得到什么结果。模型对每个标的、每根 bar 给出一个分数，选股规则把分数变成目标权重，模拟引擎按这些权重成交并记录净值曲线。每次运行都会写出一个运行目录，里面有权重、净值曲线、指标、HTML 报告，以及重建这次运行所需的配置。

主要的类有：`BaseBacktester`（`quantlab/base/backtest.py`）、vectorbt 引擎 `VectorBtBacktester`（`quantlab/backtest/engine_vectorbt.py`）、调仓时点（`quantlab/backtest/selection.py`）、配置里 `constructor` 持有的组合构建规则 `TopNConstructor`（`quantlab/portfolio/predefined/top_n.py`，继承 `quantlab/base/portfolio.py` 中的 `PortfolioConstructor`），以及美股回测器 `USEquityCrossectionSelectStockVectorBt`（`quantlab/backtest/predefined/us_equity.py`）。

## 前置条件

在仓库根目录用 `uv run python` 运行示例。在 macOS 上，同一进程导入 torch 或 xgboost 之前要设置 `OMP_NUM_THREADS=1`，并设置 `WANDB_MODE=disabled` 关闭实验跟踪。

回测需要一份价格数据集，其存储里有 `adjOpen` 和 `adjClose` 两列；还需要一个模型，checkpoint 由 `train()` 或 `train_cv()` 写出。下面的会话使用一套合成数据：六个标的、一个因子、一个标签，以及一个无需拟合的模型，它的分数就是过去一根 bar 的收益率。标签是用 `Forward` 包装的因子 `open_ret_1`，`span=1`，`delay` 取默认值 1：它在 bar t 的值是从 t+1 到 t+2 的开盘价收益率，所以前视（lookahead）为 2 根 bar。最后一个标的 `FFF` 从第 36 根 bar 起不再有价格。把下面的代码保存为 `demo_parts.py`。

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

from quantlab.base.config import DatasetConfig, ForwardConfig, ModelConfig, PolarsFactorConfig
from quantlab.factor.polars import FactorPolars
from quantlab.model.library_model import LibraryModel
from quantlab.dataset.stock import StockDataset
from quantlab.label.forward import Forward

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
    """特征 past_ret_1：单 bar 收盘价收益率。"""

    def _get_factor_lazyframe(self, lf):
        c = pl.col("adjClose")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((c / c.shift(1).over("symbol") - 1).alias("past_ret_1"))
                .select(["timestamp", "symbol", "past_ret_1"]))


class OpenReturn(FactorPolars):
    """open_ret_1：单 bar 开盘价收益率，只用到 t 及之前的 bar。"""

    def _get_factor_lazyframe(self, lf):
        o = pl.col("adjOpen")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((o / o.shift(1).over("symbol") - 1).alias("open_ret_1"))
                .select(["timestamp", "symbol", "open_ret_1"]))


class MomentumHead(LibraryModel):
    """直接预测第一个特征，所以分数就是过去收益率。"""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def make_label(cfg, delay=1):
    """标签 open_ret_1 在 t 的值：从 t+delay 到 t+delay+1 的开盘价收益率。"""
    factor = OpenReturn(PolarsFactorConfig(warmup_bars=1, dataset=prices_of(cfg)))
    return Forward(ForwardConfig(factor=factor, span=1, delay=delay))


def make_model(root, cfg, days, train_end=39, delay=1):
    day = lambda i: str(days[i].date())
    factor = PastReturn(PolarsFactorConfig(warmup_bars=5, dataset=prices_of(cfg)))
    return MomentumHead(ModelConfig(
        factors=[factor], labels=[make_label(cfg, delay)], model_save_dir=str(root / "models"),
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
    model.train_cv(train_periods=train_periods)
    return next(Path(model.config.model_save_dir).rglob("cv_folds.json")).parent
```

</details>

## 基础

### 一次运行做了什么

`BaseBacktester.run()` 在 `start_date` 到 `end_date` 的窗口上回测一个模型。它先检查每个标签的延迟是否等于引擎的成交延迟，然后加载 checkpoint（或先训练模型），在窗口上计算特征，为每个标的、每根 bar 预测分数，向具体的回测器类要目标权重，模拟成交，计算指标，最后写出运行目录。`run_cv()` 对一次 `train_cv` 的每一折做同样的事，并把各折拼接成一条曲线。`run_weights(weights)` 不经过模型，直接回测已有的目标权重（见[回测预先算好的权重](#回测预先算好的权重)）。

第一个会话先训练一个 checkpoint，然后回测一条规则：持有分数最高的两个标的，每五根 bar 调仓一次。日志输出到 stderr，这里没有显示。

```python
>>> import dataclasses, json, tempfile
>>> from pathlib import Path
>>> import pandas as pd
>>> import xarray as xr
>>> from demo_parts import *
>>> from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
>>> from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
>>> from quantlab.portfolio.predefined.top_n import TopNConstructor
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
...     constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
... ))
>>> result = backtester.run()
>>> sorted(p.name for p in result.run_dir.iterdir())
['config.json', 'equity.zarr', 'fingerprint.json', 'metrics.json', 'report.html', 'settlements.json', 'weights.zarr']
```

运行目录位于 `output_dir` 之下，返回的结果里有预测、权重、模拟结果和指标。

### 目标权重契约

权重是 `(timestamp, symbol)` 上的 `weight` 变量。有限值表示该标的在这根 bar 成交后应占组合价值的比例；NaN 表示保持该标的的持仓、不交易。全为 NaN 的一行表示这根 bar 不调仓；一行也可以两者混合，例如只保留某一笔持仓不动。一行目标的总敞口（绝对权重之和）不超过 1。库自带的组合构建规则在调仓 bar 上给每个标的有限权重，没被选中的是 `0.0`。规则只依据这根 bar 上已知的信息做决定（ADR 0014）：价格数据集的 `tradable_bars` 认为可交易的标的才是*可交易*的，默认即该 bar 上有成交价；持有但不可交易的标的是*锁定仓位*，保持当前权重。规则若改动锁定仓位，或给既不可交易也未持有的标的分配权重，回测器会拒绝。

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
2     2024-02-20    FFF  8221.655639  61.958954    0.000000  Sell
>>> adj_open = xr.open_zarr(root / "prices.zarr")["adjOpen"]
>>> float(adj_open.sel(timestamp="2024-02-13", symbol="CCC"))
52.82443690819098
```

手续费和滑点（`fees` 与 `slippage`，默认都是 0.0005）按每笔成交额的比例收取。目标百分比以其执行那根 bar 的成交价所对应的组合价值为基数。

### 标签延迟与成交延迟

引擎声明 `fill_delay_bars`，即权重形成的 bar 与成交的 bar 之间相隔的 bar 数；`VectorBtBacktester` 把它设为 1。标签的 `delay` 是信号形成的 bar 与标签开始计算的第一根 bar 之间相隔的 bar 数。`run()` 和 `run_cv()` 在训练、加载或模拟之前，逐个比较标签的 `delay` 与 `fill_delay_bars`，不相等时抛出 `ValueError`，报错信息给出该标签、它的延迟和成交延迟。

```python
>>> USEquityCrossectionSelectStockVectorBt.fill_delay_bars
1
>>> label = make_label(cfg)
>>> label.config.delay, label.span_bars(), label.lookahead_bars()
(1, 1, 2)
>>> same_bar = dataclasses.replace(backtester.config, model=make_model(root / "same_bar", cfg, days, delay=0))
>>> USEquityCrossectionSelectStockVectorBt(same_bar).run()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: labels[0] Forward ('open_ret_1',) has delay=0, but the engine fills a weight fill_delay_bars=1 bar(s) after the bar it forms on; the model would learn a return the backtest never trades
```

### 被拒订单与退市的持仓

模拟之前，两列价格都会做前向填充；随后每根成交 bar 按市场的方式执行（ADR 0014）。订单在成交 bar 上没有原始成交价（标的停牌）时是一笔*被拒订单*：持仓保持不变，订单作废，由下一次调仓重新决策。本会成交的被拒订单列在 `result.simulation.rejected_orders` 和指标的 `execution` 块里，同时给出 `rejected_order_count` 和 `max_target_deviation`，即目标权重与其成交 bar 之后实际持有权重的最大差距（含手续费和现金的影响）。

价格在窗口内中止的标的，在它最后一根有价格的 bar 上视为退市（`MarketDataset.delisting_bars`；知道停牌信息的数据集可以覆盖它）。下一根 bar 上，对它的持仓按最后的估值价转为现金，不收手续费和滑点，并记为一条*退市结算*。在 CRSP 数据上，最后的复权收盘价已经包含退市收益。`FFF` 从 2024-02-20 起没有价格，并且在第一个组合里，于 2024-02-20 按 2024-02-19 的收盘价结算；上面价格为 61.96 的那笔订单就是这次结算。

```python
>>> result.simulation.settlements
[{'symbol': 'FFF', 'axis_symbol': 'FFF', 'delisting_timestamp': Timestamp('2024-02-19 00:00:00'), 'settlement_timestamp': Timestamp('2024-02-20 00:00:00'), 'price': 61.95895375478968}]
>>> result.metrics["execution"]["rejected_order_count"]
0
```

窗口开头没有价格、且从未被持有的标的，被视为尚未上市，价格出现后正常交易。

### 预热

模型的因子需要 `start_date` 之前的历史。回测器按日期范围向每个因子请求窗口：`"cal"` 策略下调用 `compute(start_date, end_date)`，`"read"` 策略下调用 `read(start_date, end_date)`。计算型因子会在 `start_date` 之前读取自身的 `warmup_bars` 根 bar，按其数据集自己的日历计数，而不是按日历天数。因此回测与对同一窗口单独调用 `compute` 得到的因子值相同。数据集中的 bar 不够时，因子从第一根 bar 开始计算，并发出一条给出 bar 缺口数的 `UserWarning`。回测不修改任何数据集、因子或标签的 config，所以价格数据集可以与某个因子的数据集是同一个对象。预测恰好覆盖窗口内的 bar；没有预测的价格标的分数为 NaN，不会被选中。

### 样本内与样本外

记 L 为模型各标签 `lookahead_bars()` 的最大值。模型在 `train_start` 到 `train_end` 的 bar 上拟合，但要扣掉清洗（purge）部分，即测试段之前的最后 L 根 bar。最后一根参与拟合的 bar 上的标签还要再往后读 L 根 bar，所以有效训练窗口从 `train_start` 开始，到最后一根拟合 bar 之后第 L 根 bar 为止，按价格日历计数（`quantlab.utils.split.in_sample_window`）。测试段紧接训练段时，这个窗口恰好结束于配置的 `train_end`。回测窗口里落在有效训练窗口内的 bar 是样本内，其余是样本外。load 模式下，训练日期取自 checkpoint 旁边的 `config.json`。回测窗口与训练窗口重叠时，运行会记录一条警告并继续。

```python
>>> m = result.metrics
>>> m["training_window"], m["in_sample_range"], m["out_of_sample_ranges"]
(('2024-01-01', '2024-02-23'), ('2024-02-12', '2024-02-23'), [('2024-02-26', '2024-03-22')])
```

各部分来自同一次连续的模拟，所以资金和持仓会跨过边界延续。`whole` 是整个窗口的引擎统计；`in_sample` 和 `out_of_sample` 是各自 bar 上基于收益序列的统计，外加成交笔数和换手。

```python
>>> for part in ("whole", "in_sample", "out_of_sample"):
...     print(part, round(m[part]["Total Return [%]"], 2), round(m[part]["Sharpe Ratio"], 2), m[part]["Total Orders"])
whole -5.87 -2.33 19
in_sample -0.27 -0.11 6
out_of_sample -5.61 -4.27 13
>>> list(m["whole"])[:6]
['Start', 'End', 'Period', 'Start Value', 'End Value', 'Total Return [%]']
>>> round(m["whole"]["Turnover per Rebalance [%]"])
134
```

换手是某根 bar 的单边成交额除以成交前的组合价值，和其他带 `[%]` 的行一样以百分数表示，所以从现金一次性建仓约为 100，整本书全部换掉约为 200。这里的分数来自随机游走收益，负收益没有任何含义。

### 运行目录

每次运行在 `output_dir` 下写一个新目录 `{ClassName}_{timestamp}`。文件先写入一个隐藏的暂存目录，全部写完后才改名，所以 `output_dir` 里只会有完整的运行。`output_dir=None` 时什么都不写（见[只在内存中运行](#只在内存中运行)）。

| 文件 | 内容 |
| --- | --- |
| `config.json` | 配置，嵌套着价格数据集和模型，以及数据指纹。 |
| `weights.zarr` | `(timestamp, symbol)` 上的目标权重。 |
| `equity.zarr` | `timestamp` 上的组合 `value` 与每根 bar 的 `returns`。 |
| `metrics.json` | 与 `result.metrics` 相同的映射；NaN 和无穷大写成 null。每次运行都记录 `execution`（被拒订单和最大目标偏差）。`run()` 和 `run_cv()` 的每个折还记录 `portfolio_construction`：`failed_bar_count` 和 `failed_bars`，即组合构建规则无法决定（优化失败或不可行）、回测改为维持原仓位的调仓 bar。 |
| `settlements.json` | 退市结算记录。 |
| `fingerprint.json` | 本次运行读取的价格数据和因子数据的摘要。 |
| `report.html` | 净值、回撤、月度收益图表，指标表和备注。 |
| `inputs/` | 仅当价格或基准数据集是保存在内存中的 `FrameDataset` 时写出：它的面板存为 `price_dataset.zarr` 或 `benchmark_dataset.zarr`，`config.json` 以相对运行目录的路径指向它（见[重建一次给定权重的运行](#重建一次给定权重的运行)）。 |

## 常见任务

### 在回测中训练模型

`model_mode="train"` 时，模型先按自己配置的日期训练，不需要 `checkpoint`。回测窗口不会改变训练日期。本次运行写出的 checkpoint 记录在 `metrics["trained_checkpoint"]` 中。这个会话同时换成了每侧一个标的的多空组合。

```python
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=make_model(root / "train_mode", cfg, days),
...     model_mode="train", checkpoint=None,
...     constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=1)),
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

运行目录顶层的文件描述的是拼接后的曲线，`folds/fold_{i}/` 存放每一折自己的权重和净值。拼接曲线是一次模拟，所以资金会跨折延续。每一折另有一次从 `init_cash` 起步的独立模拟，各折的指标来自这些独立模拟。`train_cv` 对每一折的训练段清洗掉最后 L 根 bar，并把清洗后的 `train_end` 记入清单。一折的样本内窗口结束于该 `train_end` 之后第 L 根 bar，也就是该折测试段之前的那根 bar，所以拼接曲线上没有样本内的 bar。`quantlab.utils.split.split_ranges` 把拼接后的 bar 切分为 `in_sample_ranges` 和 `out_of_sample_ranges`。

```python
>>> stitched = cv.metrics["stitched"]
>>> stitched["in_sample_ranges"][:2], stitched["out_of_sample_ranges"][:2]
([], [('2024-02-12', '2024-04-17')])
>>> round(stitched["whole"]["Total Return [%]"], 2), round(cv.folds[0]["metrics"]["whole"]["Total Return [%]"], 2)
(-3.11, -1.62)
```

### 回测预先算好的权重

`run_weights(weights)` 在没有模型的情况下回测一个已有的目标权重面板，例如别的工具算出的权重，或一次早先运行保存下来的权重。配置不需要 `model` 和 `model_mode`；这两项要么同时设置，要么同时为 `None`，只设置其中一项的配置在构造回测器时就会被拒绝。回测器读取 `start_date` 到 `end_date` 窗口内的成交价和估值价，在恰好这些 bar 和标的上按[目标权重契约](#目标权重契约)检查权重，并以同样的 t+1 成交方式模拟。权重面板可以是带 `weight` 变量的数据集，也可以是数据数组，坐标轴顺序不限，会对齐到价格的坐标轴上。基准的处理与 `run()` 相同。这类运行没有训练窗口，所以指标只有全窗口的部分（`whole`；设置了基准时还有 `benchmark` 和 `relative`，各自只含 `whole`），没有样本内/样本外的拆分，报告里也不出现拆分相关的行。把第一段会话得到的权重传进去（那段会话的配置没有设置基准），就能复现那次运行，因此指标只有 `whole`、`execution` 和 `notes`。

```python
>>> weights_config = dataclasses.replace(backtester.config, model=None, model_mode=None, checkpoint=None)
>>> weights_backtester = USEquityCrossectionSelectStockVectorBt(weights_config)
>>> replay = weights_backtester.run_weights(result.weights)
>>> sorted(replay.metrics), replay.predictions is None
(['execution', 'notes', 'whole'], True)
>>> replay.metrics["whole"] == result.metrics["whole"]
True
>>> sorted(p.name for p in replay.run_dir.iterdir())
['config.json', 'equity.zarr', 'fingerprint.json', 'metrics.json', 'report.html', 'settlements.json', 'weights.zarr']
```

`run()` 和 `run_cv()` 仍然需要模型：

```python
>>> weights_backtester.run()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: run() requires config.model, but it is None; set config.model and config.model_mode, or backtest precomputed weights with run_weights()
```

违反契约的权重会被拒绝，错误信息指出出问题的 bar：

```python
>>> broken = result.weights.copy(deep=True)
>>> broken["weight"][5, 0] = 0.9
>>> weights_backtester.run_weights(broken)
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: weight row at 2024-02-19 has gross exposure 1.9 > 1
>>> weights_backtester.run_weights(result.weights.isel(timestamp=slice(1, None)))
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: the weight bars must be exactly the price bars of the backtest window: 1 missing ['2024-02-12'], 0 extra [], 0 duplicated
```

`WeightsVectorBt`（`quantlab/backtest/predefined/weights.py`）可以在任意市场上回测给定的权重：成交价列、估值价列和年化参数（`trading_days_per_year`、`session_minutes_per_day`）写在它的 `WeightsBacktestConfig` 里，而不是类常量；它不接受模型。`quantlab.api.backtest` 运行的就是这个回测器。价格数据集可以是保存在内存中的 `FrameDataset`；没有 store 也就没有 ticker 附属文件，标的按原样显示，不会记录警告。

```python
>>> import numpy as np
>>> import pandas as pd
>>> import xarray as xr
>>> from quantlab.backtest.predefined.weights import WeightsVectorBt
>>> from quantlab.base.config import WeightsBacktestConfig
>>> from quantlab.dataset.memory import FrameDataset
>>> bars = pd.bdate_range("2024-01-01", periods=5)
>>> prices = FrameDataset(pd.DataFrame({
...     "timestamp": np.repeat(bars, 2),
...     "symbol": ["AAA", "BBB"] * 5,
...     "open": [10.0, 20.0, 11.0, 20.0, 12.0, 21.0, 12.0, 22.0, 13.0, 22.0],
...     "close": [10.5, 20.0, 11.5, 20.5, 12.0, 21.5, 12.5, 22.0, 13.0, 22.5],
... }))
>>> backtester = WeightsVectorBt(WeightsBacktestConfig(
...     price_dataset=prices, start_date="2024-01-01", end_date="2024-01-05",
...     output_dir=None, rebalance_periods=1, fees=0.0, slippage=0.0,
...     fill_price_column="open", valuation_price_column="close",
...     trading_days_per_year=252, session_minutes_per_day=390,
... ))
>>> weights = xr.DataArray(
...     [[1.0, 0.0]] + [[np.nan, np.nan]] * 4, dims=("timestamp", "symbol"),
...     coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
... )
>>> result = backtester.run_weights(weights)
>>> result.simulation.value.values.round(2).tolist()
[1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]
```

### 只在内存中运行

`output_dir=None` 时运行不写任何文件：没有运行目录，也没有报告。`result.run_dir` 为 `None`，其余内容都在返回的结果里。`run()`、`run_cv()` 和 `run_weights()` 都是如此。`output_dir` 没有默认值，需要显式传入 `None`。`output_dir=None` 只管回测自己的运行目录：在 `model_mode="train"` 下，模型仍会把 checkpoint 写到它自己的配置指定的位置。

```python
>>> in_memory = USEquityCrossectionSelectStockVectorBt(
...     dataclasses.replace(weights_config, output_dir=None)
... ).run_weights(result.weights)
>>> in_memory.run_dir is None, round(in_memory.metrics["whole"]["Total Return [%]"], 2)
(True, -5.85)
```

`report_figure(result)` 以 plotly 图形返回 `report.html` 中嵌入的那张图（净值、回撤、月度收益，设置了基准时还有基准相关的行），因此只在内存中运行的结果也能查看。它接受 `run()` 或 `run_weights()` 的结果。

```python
>>> figure = weights_backtester.report_figure(in_memory)
>>> type(figure).__name__
'Figure'
```

### 与基准对比

把 `benchmark_dataset` 设为只含一个标的的市场数据集，例如 `scripts/wrds/etf.py --etf qqq` 写出的 QQQ store（`CrspDatasetConfig.qqq_benchmark`）。它和价格数据集一样是 `(timestamp, symbol)` 面板，放在单独的 store 里，带有相同的 `adjOpen` / `adjClose` 列。直接传入数据集对象：

```python
>>> from quantlab.dataset.crsp import CrspStockDataset
>>> qqq = CrspStockDataset(CrspDatasetConfig.qqq_benchmark(
...     zarr_file_path="data/us_equity/1d/wrds_crsp_qqq_1d.zarr",
...     raw_data_dir_path="data/downloads/us_equity/1d/crsp/wrds",
...     reference_dir="data/reference/crsp",
... ))
>>> config = CrossSectionBacktestConfig(..., benchmark_dataset=qqq)
>>> result = USEquityCrossectionSelectStockVectorBt(config).run()
>>> sorted(result.metrics["relative"]["whole"])[:4]
['Annualized Excess Return [%]', 'Bars', 'Benchmark Total Return [%]', 'Beta']
```

基准按回测窗口读取，并对齐到策略自己的 bar 上；基准缺失的 bar 沿用前一个价格（记录一条警告）。基准晚于窗口开始、或面板中不止一个标的时会报错。基准按策略的同一套执行约定买入并持有：在第二根 bar 的开盘价全仓买入，初始资金 `init_cash`、手续费和滑点都与策略相同，因此两条净值曲线可以逐 bar 比较。`result.benchmark` 是它的 `SimulationResult`。

运行结果多出两个指标块，每块都有 `whole`、`in_sample`、`out_of_sample` 三个切片：

- `benchmark`：基准的 `symbol` 及其自身的收益统计（总收益、年化收益、波动率、Sharpe、最大回撤等）；
- `relative`：组合相对基准的表现，命名沿用 vectorbt 的风格，所有带 `[%]` 的行都是百分数。*相对净值* = 组合净值 / 基准净值。`Excess Return [%]` 是期末相对净值减 1（即通常所说的超额收益 alpha），`Annualized Excess Return [%]` 为其年化值，`Excess Max Drawdown [%]` 是相对净值从其历史高点的最大回落（*超额回撤*，为负数或 0），另有 `Strategy Total Return [%]`、`Benchmark Total Return [%]`、`Total Return Difference [%]`、`Tracking Error [%]`、`Information Ratio`、`Beta`、`Correlation`、`CAPM Alpha [%]`（年化回归截距）和 `Win Rate vs Benchmark [%]`。

`report.html` 在组合净值的同一面板上画出基准净值（灰色虚线），其下是超额收益和超额回撤两行，回撤和月度收益面板中并列显示基准，另有“Excess over benchmark”和“Benchmark (buy and hold)”两张表。`equity.zarr` 额外保存 `benchmark_value` 和 `benchmark_returns`，`fingerprint.json` 在 `benchmark_dataset` 下记录基准数据指纹，`config.json` 可以重建基准。`run_cv()` 对拼接曲线和每个 fold 做同样的对比。

### 从配置重建一次运行

`config.json` 用点分导入路径记录每个类，所以 `load_backtester_from_config` 能重建出相同的回测器，包括它的价格数据集和模型，`run()` 会把这次回测重做一遍，写入新目录。如果自原始运行以来数据发生了变化，重建的运行会对每个变化的数据集记录一条警告并继续。

```python
>>> from quantlab.utils.module import load_backtester_from_config
>>> config = json.loads((result.run_dir / "config.json").read_text())
>>> config["name"], config["constructor"]
('quantlab.backtest.predefined.us_equity.USEquityCrossectionSelectStockVectorBt', {'direction': 'long_only', 'top_n': 2, 'score_label': None, 'name': 'quantlab.portfolio.predefined.top_n.TopNConstructor'})
>>> again = load_backtester_from_config(config).run()
>>> again.metrics["whole"] == result.metrics["whole"]
True
>>> again.run_dir == result.run_dir
False
```

这些类必须能按点分路径导入。定义在脚本里的类名为 `__main__.X`，换一个进程就找不到，所以数据集、因子、模型和回测器应放在模块里。train 模式写出的配置在重建时会重新训练；要重放同一个模型，请把 `model_mode` 设为 `"load"`，并把 `checkpoint` 设为记录下来的 `trained_checkpoint`。

### 重建一次给定权重的运行

`run_weights()` 的运行没有模型可以重新预测权重，所以要用它保存下来的权重重放：用 `XrBackend().read(path).data` 读取运行目录里的 `weights.zarr`，把这个面板传给 `run_weights()`。当价格或基准数据集是 `FrameDataset` 时（每次 `quantlab.api.backtest` 运行，以及上面的 `WeightsVectorBt` 会话），它的面板没有自己的 store，因此由数据集自己（它的 `persist_with_run`）把面板写到 `inputs/` 下，`config.json` 以相对运行目录的路径指向这个 store。把读取配置所在的目录作为 `run_dir` 传入；运行目录可以移动。从项目 store 读取的数据集什么都不写，路径保持不变。接着 `WeightsVectorBt` 的会话：

```python
>>> import dataclasses, json, shutil, tempfile
>>> from pathlib import Path
>>> from quantlab.backend import XrBackend
>>> from quantlab.utils.module import load_backtester_from_config
>>> kept = WeightsVectorBt(
...     dataclasses.replace(backtester.config, output_dir=tempfile.mkdtemp())
... ).run_weights(weights)
>>> sorted(p.name for p in kept.run_dir.iterdir())
['config.json', 'equity.zarr', 'fingerprint.json', 'inputs', 'metrics.json', 'report.html', 'settlements.json', 'weights.zarr']
>>> config = json.loads((kept.run_dir / "config.json").read_text())
>>> config["price_dataset"]["zarr_file_path"]
'inputs/price_dataset.zarr'
>>> run_dir = Path(shutil.move(kept.run_dir, tempfile.mkdtemp()))
>>> rebuilt = load_backtester_from_config(config, run_dir=run_dir)
>>> replay = rebuilt.run_weights(XrBackend().read(run_dir / "weights.zarr").data)
>>> replay.simulation.value.values.round(2).tolist()
[1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]
>>> json.loads((replay.run_dir / "metrics.json").read_text()) == json.loads(
...     (run_dir / "metrics.json").read_text())
True
>>> json.loads((replay.run_dir / "fingerprint.json").read_text()) == rebuilt.expected_fingerprint
True
```

重建出的 `FrameDataset` 把 store 读进内存；重放会再次写出自己的 `inputs/`，所以它的目录同样可以独立重建。它的标的按原样显示：`FrameDataset` 不指定任何用于查找 CRSP ticker 附属文件的 store（`ticker_store()` 为 `None`），即使它是从 `inputs/` 读回来的。相对的 store 路径永远不会按当前工作目录解析：

```python
>>> load_backtester_from_config(config)
Traceback (most recent call last):
  ...
ValueError: quantlab.dataset.memory.FrameDataset reads the store 'inputs/price_dataset.zarr', which is relative to the run directory the config was saved in; pass run_dir= (the directory holding config.json) to rebuild it. It is never resolved against the working directory.
```

## 扩展

新的选股规则是 `VectorBtBacktester` 的子类，需要三个成员：`config_cls`、`MARKET` 和 `_generate_signals(predictions, prices)`。该方法返回一个数据集，其 `weight` 变量满足上面的契约。`predictions` 与 `prices` 共用同一套 `(timestamp, symbol)` 坐标轴。下面的规则让每个可交易标的的权重与其正分数成比例，没有正分数时空仓。它复用了 `rebalance_mask` 和 `US_EQUITY_MARKET` 的价格约定。保存为 `score_weighted.py`。

```python
"""一条新的选股规则：多头权重与正分数成比例。"""

import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import rebalance_mask
from quantlab.backtest.predefined.us_equity import US_EQUITY_MARKET
from quantlab.base.config import BacktestConfig


class ScoreWeightedBacktester(VectorBtBacktester):
    config_cls = BacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions, prices):
        label = list(predictions.data_vars)[0]  # 模型的第一个标签
        scores = predictions[label].transpose("timestamp", "symbol")
        # 在这根 bar 上没有成交价的标的不能交易，不参与选择（ADR 0014）。
        tradable = self.config.price_dataset.tradable_bars(prices, self.MARKET.fill_price_column)
        positive = scores.where(tradable).clip(min=0).fillna(0.0)
        total = positive.sum("symbol")
        weight = (positive / total.where(total > 0)).fillna(0.0)  # 没有正分数则空仓
        rebalance = xr.DataArray(
            rebalance_mask(prices.sizes["timestamp"], self.config.rebalance_periods),
            dims="timestamp", coords={"timestamp": prices.timestamp},
        )
        return weight.where(rebalance).to_dataset(name="weight")  # 非调仓 bar 为 NaN
```

它的配置类是 `BacktestConfig`，所以不需要 `constructor`。

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
symbol        AAA    BBB    CCC   DDD  EEE    FFF
timestamp                                        
2024-02-12  0.000  0.000  0.541  0.00  0.0  0.459
2024-02-13    NaN    NaN    NaN   NaN  NaN    NaN
2024-02-19  0.037  0.647  0.000  0.16  0.0  0.156
```

如果想沿用 top-N 规则、只换分数，`TopNConstructor(TopNConfig(direction, top_n)).construct_panel(scores, tradable, rebalance)` 接受任意分数面板（每个标签一个变量的数据集）和一个布尔型可交易面板（例如价格数据集的 `tradable_bars(prices, fill_column)`），并返回同样的 `weight` 数据集；再传入 `fill_price=`、`valuation_price=` 和 `delisted=`，每根 bar 拿到的就是模拟中实际持有的仓位。它的逐 bar 方法 `construct(context)` 根据一个 `PortfolioContext` 决定一根 bar 的权重，自定义规则就是这样写的：继承 `quantlab.base.portfolio` 中的 `PortfolioConstructor` 并实现 `construct`。换一个市场就是换一个 `MarketSpec`，其中有自己的成交价列、估值价列和年化常数。

### 回测任意预测器

`config.model` 不必是 `BaseModel`。回测器只依赖 `quantlab.base.backtest` 中的 `Predictor` 协议，`BaseModel` 不继承它也满足它。由多个模型组合成的集成、或包装一个模型的对象，只要具备全部成员，回测器无需任何改动即可回测：

| 成员 | 回测器的用途 |
|---|---|
| `labels`、`label_delays` | 标签延迟检查、清除（purge）与样本内划分（`lookahead_bars()`）、预测变量名 |
| `train_bounds`、`test_bounds` | 配置中的训练窗口和测试窗口 |
| `predict_window(start, end)` | 一个窗口的预测面板；预测器自己请求特征和预热 |
| `fingerprint_inputs(start, end)`、`training_fingerprint_inputs()` | `(key, 因子或标签, 策略, first, last)` 条目，回测器把它们哈希进 `data_fingerprint` |
| `collect()`、`train()` | 训练模式；`train` 返回检查点，其旁边的 `config.json` 记录训练日期 |
| `check_checkpoint(path)`、`load(path)` | 加载模式；检查在计算任何特征之前运行 |
| `get_config()`、`from_config(config)` | `config.json`，以及 `load_backtester_from_config` 通过 `"name"` 指明的类进行重建 |

回测器不读取模型配置，也不调用模型的其他方法。`model` 缺少成员的配置在构造时被拒绝，抛出 `TypeError` 并列出缺少的成员。

```python
>>> from typing import get_protocol_members
>>> from quantlab.base.backtest import Predictor
>>> sorted(get_protocol_members(Predictor))
['check_checkpoint', 'collect', 'fingerprint_inputs', 'from_config', 'get_config', 'label_delays', 'labels', 'load', 'predict_window', 'test_bounds', 'train', 'train_bounds', 'training_fingerprint_inputs']
```

`SeedEnsemble`（见 model 指南的“平均多个种子”）就是这样的预测器。训练模式下，`run()` 把每个种子训练到同一个集成目录，并把其中的 `ensemble.json` 记为 `trained_checkpoint`；加载模式下，`checkpoint` 就是这个 `ensemble.json`，样本内划分所用的训练日期从它旁边的集成级 `config.json` 读取，与单个模型的检查点相同。预测是各成员截面 z-score 的平均。各成员读取相同的输入，所以数据指纹的键与单个模型相同；`load_backtester_from_config` 用运行目录 `config.json` 中的 `get_config()` 重建集成。`MomentumHead` 没有需要拟合的内容，三个种子的结果一致，所以权重与第一段会话中单个模型的权重相同。

```python
>>> from quantlab.model.predefined.seed_ensemble import SeedEnsemble
>>> ensemble = SeedEnsemble(make_model(root / "ensemble", cfg, days), seeds=[0, 1, 2])
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=ensemble, model_mode="train", checkpoint=None,
... )).run()
>>> manifest = Path(trained.metrics["trained_checkpoint"])
>>> manifest.name, sorted(p.name for p in manifest.parent.iterdir())
('ensemble.json', ['config.json', 'ensemble.json', 'ic_series.csv', 'member_0', 'member_1', 'member_2', 'metrics.json', 'test_predictions.zarr'])
>>> replayed = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config,
...     model=SeedEnsemble(make_model(root / "replay", cfg, days, train_end=20), seeds=[0, 1, 2]),
...     checkpoint=str(manifest),
... )).run()
>>> replayed.metrics["training_window"]
('2024-01-01', '2024-02-23')
>>> bool((replayed.weights["weight"].fillna(0) == result.weights["weight"].fillna(0)).all())
True
>>> saved = json.loads((replayed.run_dir / "config.json").read_text())
>>> saved["model"]["seeds"], sorted(saved["data_fingerprint"])
([0, 1, 2], ['factor[0]:PastReturn', 'price_dataset'])
```

`run_cv()` 以同样的方式回放集成的交叉验证。`SeedEnsemble.train_cv`（见 model 指南的“平均多个种子”）写出的 `cv_folds.json` 与单个模型的 `train_cv` 格式相同，其中的 `checkpoint` 是每个 `fold_{i}/` 的 `ensemble.json`。以集成为 `model`、以这个目录为 `cv_project_dir` 时，每一折加载自己的集成，该折集成级 `config.json` 记录的训练日期与清单中的日期相互核对，与单个模型的折相同。回测器为此无需任何改动。`MomentumHead` 的各个种子结果仍然一致，所以拼接后的权重与上文单个模型交叉验证的权重相同。

```python
>>> cv_ensemble = SeedEnsemble(make_model(root / "ensemble_cv", cfg2, days2, train_end=29), seeds=[0, 1, 2])
>>> folds = cv_ensemble.collect().train_cv(train_periods=30)
>>> ensemble_cv_dir = Path(folds[0]["checkpoint"]).parent.parent
>>> ensemble_cv = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     cv_config,
...     model=SeedEnsemble(make_model(root / "ensemble_cv_backtest", cfg2, days2, train_end=29), seeds=[0, 1, 2]),
...     cv_project_dir=str(ensemble_cv_dir),
... )).run_cv()
>>> len(ensemble_cv.folds), Path(ensemble_cv.folds[0]["checkpoint"]).relative_to(ensemble_cv_dir).as_posix()
(8, 'fold_0/ensemble.json')
>>> bool((ensemble_cv.weights["weight"].fillna(0) == cv.weights["weight"].fillna(0)).all())
True
>>> ensemble_cv.metrics["stitched"]["in_sample_ranges"]
[]
```

## 注意事项

回测不模拟借券费用或做空融资成本，所以空头一侧的收益偏乐观；指标里的 `notes` 也有说明。交易统计采用持仓视角：一笔交易是某个标的从建仓到清仓的一次完整往返，把持仓减回目标权重不算一笔已平仓交易。`Total Orders` 是成交笔数。

`benchmark_dataset` 必须只含一个标的（见[与基准对比](#与基准对比)）。具体的回测器必须设置 `MARKET`。`run()` 和 `run_cv()` 需要 `model` 和 `model_mode`，`run_weights()` 不使用这两项。load 模式下 `run()` 需要 `checkpoint`，`run_cv()` 需要 `cv_project_dir` 和 `model_mode="load"`。每个标签的 `delay` 必须等于引擎的 `fill_delay_bars`（见[标签延迟与成交延迟](#标签延迟与成交延迟)）。价格存储旁没有 CRSP ticker 附属文件时，回测器会记录一条警告，说明改用坐标轴上的标的名作为标签，运行本身不受影响；保存在内存中的数据集（`FrameDataset`）没有存储，直接使用标的名，不记录警告。

配置类用错时：

```python
>>> USEquityCrossectionSelectStockVectorBt(custom.config)
Traceback (most recent call last):
  ...
TypeError: USEquityCrossectionSelectStockVectorBt requires a CrossSectionBacktestConfig, got BacktestConfig
```

模型没有声明的 score 标签：

```python
>>> USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config,
...     constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2, score_label="fwd_ret_5")),
... ))
Traceback (most recent call last):
  ...
ValueError: score_label 'fwd_ret_5' is not one of the predictor's labels ['open_ret_1']
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
>>> del config["constructor"]
>>> load_backtester_from_config(config)
Traceback (most recent call last):
  ...
ValueError: quantlab.backtest.predefined.us_equity.USEquityCrossectionSelectStockVectorBt config is missing field(s) ['constructor']; refusing to fill them from the current dataclass defaults, which may differ from the values the stored backtest ran with
```

如果 `cv_folds.json` 中间缺了一折，`run_cv()` 拒绝跨缺口拼接，报错信息包含 `fold test segments are not contiguous: gap between fold 2 ending 2024-03-06 and fold 4 starting 2024-03-15; 6 price bar(s) in between belong to no fold, so a stitched out-of-sample curve would silently skip them`。恢复清单，或者把 `start_date` 与 `end_date` 收窄到一段连续的折。

## 另请参阅

- [model](model.md)：`train`、`train_cv`、`cv_folds.json` 与 `predict_panel`。
- [dataset](dataset.md)：价格数据集；[factor](factor.md)：模型使用的因子和标签。
- [backend](backend.md)：权重和净值曲线所写入的 Zarr 存储。
- `quantlab/base/backtest.py` 中的 `BaseBacktester`、`BacktestResult`、`CVBacktestResult`、`MarketSpec`；`quantlab/base/config.py` 中的 `BacktestConfig` 与 `CrossSectionBacktestConfig`；`quantlab/utils/module.py` 中的 `load_backtester_from_config`。
