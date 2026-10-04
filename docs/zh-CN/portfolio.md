# 组合构建（Portfolio construction）

[English](../portfolio.md) | 简体中文

组合构建是模型和回测之间的一步。在每个调仓 bar 上，它拿到模型对这根 bar 的预测和当前持有的权重，返回这根 bar 之后要持有的权重，回测再去交易这些权重。

每条规则都继承 `PortfolioConstructor`（`quantlab/base/portfolio.py`）。`quantlab/portfolio/predefined/` 里自带两条规则：

- `TopNConstructor`：分数最高的 `top_n` 个标的等权，可以只做多，也可以多空。
- `MeanVarianceOptimizer`：带换手惩罚的 Markowitz 权重，用 cvxpy 求解。它用风险模型给风险定价；自带的风险模型是 `LedoitWolfRiskModel`，即对历史收益样本协方差做 Ledoit-Wolf 收缩。

回测把规则放在配置的 `constructor` 字段里（见[回测](backtest.md)）。

## 前置条件

在仓库根目录用 `uv run python` 运行下面的会话。这些会话手工构造每根 bar 的输入，所以不需要 store、模型或 GPU。在回测中，回测器会从价格数据集和模型构造同样的输入。

## 基础

### 一次只看一根 bar

规则只需要实现一个方法：`construct(context)`。它收到的 `PortfolioContext` 描述一根 bar，不含任何更晚的 bar 的信息：

| 字段 | 内容 |
| --- | --- |
| `timestamp` | 这根 bar。返回的权重在下一根 bar 成交。 |
| `predictions` | 这根 bar 上每个标签的预测，每个标签一个变量，维度为 `symbol`。 |
| `tradable` | 每个标的在这根 bar 上能否交易：默认看它在这根 bar 上有没有真实成交价。 |
| `current_weights` | 当前持有的权重，按这根 bar 估值。第一次调仓之前全为 0.0。 |
| `returns` | 截至这根 bar 的最近 `lookback_bars` 个单 bar 收益，由规则最近 `history_bars` 个原始估值价格在该窗口内前向填充后算出。`lookback_bars` 为 0 的规则拿到的是空窗口。 |
| `staleness` | 每个标的在最近 `history_bars` 根 bar 内距上一个真实价格已过了多少根 bar；窗口内没有真实价格时为 NaN。 |
| `factors` | 规则在 `required_factors()` 中声明的因子在这根 bar 上的值；没有声明时为 `None`。 |

规则为每个标的返回一个权重：

- 权重都是有限值，未持有的为 0.0，总敞口不超过 1。
- 整行为 NaN 表示"保持当前仓位"。
- 持有但不可交易的标的是*锁定仓位*，由 `context.locked` 标出，必须保持当前权重。既不可交易也未持有的标的权重为 0.0。

规则在 bar 之间不保存任何状态。它只能看到一根 bar，所以不可能看到未来；它也可以直接在事件驱动引擎的 bar 处理函数里调用。

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.base.config import TopNConfig
>>> from quantlab.base.portfolio import PortfolioContext
>>> from quantlab.portfolio.predefined.top_n import TopNConstructor
>>> symbols = ["AAA", "BBB", "CCC", "DDD"]
>>> def on_symbols(values):
...     return xr.DataArray(values, dims="symbol", coords={"symbol": symbols})
>>> context = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, -0.1, 0.5, 0.2])}),
...     tradable=on_symbols([True, True, True, False]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.25]),
... )
>>> context.locked.values
array([False, False, False,  True])
>>> TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(context).values
array([0.375, 0.   , 0.375, 0.25 ])

```

`DDD` 被持有时停牌，所以保持 0.25；剩下的 0.75 由可交易标的中分数最高的两个平分。

分数相同的标的按标的顺序排列。当某一侧的截断点落在一组同分标的中间时，持有哪几只是由标的顺序而不是模型决定的，这一行会把被排除的并列标的数量记为 `tie_at_cutoff` 事件。下面 `BBB`、`CCC` 和 `DDD` 分数相同，`BBB` 排在最前，另外两只被排除。如果模型只输出少数几个不同的预测值，大多数调仓都会出现这个事件。

```python
>>> tied = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, 0.5, 0.5, 0.5])}),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.0]),
... )
>>> weights = TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(tied)
>>> weights.values
array([0.5, 0.5, 0. , 0. ])
>>> weights.attrs
{'events': {'tie_at_cutoff': 2}}

```

### 在回测中

规则只负责决策。它的*决策输入*，即在一根 bar 上能读到的除持仓以外的一切，由同一个模块组装：`quantlab/portfolio/decision_inputs.py` 中的 `DecisionInputs`，回测和执行器都用它。它由价格数据集、成交价列和估值价列、已绑定的规则、调仓周期、*锚点*（预测面板的第一根 bar，调仓日程从它开始计数）以及执行设置构造。向量化回测调用它的 `weights(predictions, delisted=...)`，它在调仓 bar 上逐根调用规则的单 bar 决策（`decide`，见[回测之外决定一根 bar](#回测之外决定一根-bar)），并为每根 bar 构造 context：

- 可交易性取自价格数据集的 `tradable_bars`；因子值是规则的 `required_factors()` 在窗口上计算的结果，各带自己的预热期。
- 当前权重是之前各次调仓实际留下的持仓，由执行模块（`quantlab.utils.execution`）按模拟引擎完全相同的方式重放，包括被拒订单、退市结算、sizing basis、手续费和滑点。回测器传入自己 config 中的 `execution` 设置，以及交给引擎的同一份退市标记；不传设置时按成交价定仓位、不计成本。
- 调仓 bar 是从锚点起每 `rebalance_periods` 根中的一根；最后一根 bar 从不调仓，因为在那里决定的订单没有下一根 bar 可以成交（同一模块中的 `rebalance_mask`）。
- 收益窗口用每个标的最后已知的价格计算，所以一次停牌表现为若干个零收益，然后在复牌当天出现整段涨跌。
- 每根 bar 只读截至（含）它的最近 `history_bars` 个原始估值价格（默认 `lookback_bars + 1`；Ledoit-Wolf 为 `lookback_bars + 1 + max_stale_bars`；均值方差取其风险模型的值），所以决策与价格历史从哪里开始无关。回测的预热期包含第一根 bar 之前的 `history_bars - 1` 根 bar，所以第一根 bar 就有完整的窗口。

回测器在构造时调用规则的 `bind(labels)`，这时还没有读任何数据、也没有训练任何模型。`labels` 为每个预测变量给出一个 `LabelSpec(name, scale, delay, span)`，由回测器用 `quantlab.base.backtest.label_specs` 从预测器推导；不是 `Forward` 标签的 `span` 为 `None`。规则对预测能知道的只有这些规格，永远拿不到模型本身。规则在这里检查自己需要的标签，所以配置错误会立刻报错。

规则无法决定的 bar 会抛出 `PortfolioConstructionError`，例如优化不可行或求解器失败。`decide` 把它变成在这根 bar 上保持当前仓位，并记一条警告。`metrics.json` 在 `portfolio_construction` 下列出所有这样的 bar（`failed_bar_count`、`failed_bars`），以及规则报告的事件，比如上文的 `tie_at_cutoff` 或下文的 `closed_without_risk`，带 `count`（所有 bar 上涉及的标的总数）和每个 bar 一条记录。

运行的配方记录了规则的全部参数和它的风险模型；`BacktestRun.rebuild("constructor")` 重建规则，`rebuild_backtester()` 重建整个回测器（见回测指南）。

### 不加载模型重建一次运行的决策输入

带模型的运行（`run()` 或 `run_cv()`）还会保存规则读到的预测及其标签规格，格式是 `PredictionPanel`，由 `BacktestRun.predictions()` 返回。`DecisionInputs.from_run(run_dir)` 通过 `BacktestRun` 读取该运行：重建规则（绑定到预测面板的规格上）和价格数据集（内存数据集从运行目录下的副本读取），取市场价格列、执行设置和调仓周期，并取预测面板的第一根 bar 作为锚点。它从不导入模型、因子、标签或回测层，所以研究流水线之外的执行器（例如事件驱动回测）只凭运行目录就能重放该运行的决策；在预测面板的预测上调用 `weights` 会复现该运行的权重。设 `result` 是[回测指南](backtest.md#一次运行做了什么)第一个会话的 `run()`，规则持有前 2 名、每五根 bar 调仓一次：

```python
>>> from quantlab.portfolio.decision_inputs import DecisionInputs
>>> from quantlab.runs.backtest_run import BacktestRun
>>> run = BacktestRun.open(result.run_dir)
>>> run_inputs = DecisionInputs.from_run(run.path)
>>> run_inputs.constructor == run.rebuild("constructor"), run_inputs.rebalance_periods
(True, 5)
>>> panel = run.predictions()
>>> panel.labels
(LabelSpec(name='open_ret_1', scale='raw', delay=1, span=1),)
>>> run_inputs.anchor == panel.predictions.timestamp.values[0]
True
>>> run_inputs.weights(panel.predictions).equals(run.weights())
True

```

预测面板的每个标签对应 `(timestamp, symbol)` 上的一个变量，与其规格一起保存。`run_weights()` 的运行没有模型，也没有预测面板：它的 `predictions()` 是 `None`。

### 回测之外决定一根 bar

自己维护账本的执行器（例如事件驱动回测或实盘账户）用同一个 `DecisionInputs` 和规则的 `decide` 决定一根 bar：

- `rebalances(t)` 回答 bar `t` 是否是日程中的调仓 bar；构造时传入 `end=` 可以让回放的最后一根 bar 不调仓。
- `context(t, predictions, current_weights)` 构造这根 bar 的 `PortfolioContext`。`predictions` 和 `current_weights` 是这根 bar 在 `symbol` 上的取值；`current_weights` 中缺失的标的视为未持有，没有预测的持仓标的以 NaN 预测加入这根 bar。可交易性、截至 `t` 的最近 `history_bars` 个估值价格得出的收益窗口和停牌时长（staleness），以及 `t` 上的因子值，都从数据集读取，所以执行器最多只需保留 `history_bars` 根 bar 的价格。给定回测重放出的持仓，得到的 context 与回测构造的相同。
- `decide(context)` 调用 `construct`，并按权重契约检查这一行。它返回 `Decision(weights, failure, events)`：context 各标的上的权重，全 NaN 表示保持；使这根 bar 保持仓位的 `PortfolioConstructionError` 的消息，或 `None`；以及这一行报告的事件。违反契约的行（NaN 与有限权重混合、改动了锁定仓位、给既不可交易也未持有的标的分配权重）是规则的 bug，抛出 `ValueError`。

这里直接构造 `inputs`，即 `DecisionInputs(dataset, rule, fill_column=..., valuation_column=..., rebalance_periods=..., anchor=...)`，价格只有三根 bar，`DDD` 在 `context.timestamp` 上没有价格（`from_run` 从一次运行构造同样的输入）：

```python
>>> from quantlab.base.portfolio import LabelSpec
>>> from quantlab.dataset.memory import FrameDataset
>>> bars = pd.bdate_range(end=context.timestamp, periods=3)
>>> close = xr.DataArray(
...     [[10.0, 20.0, 30.0, 40.0], [10.5, 20.5, 30.5, 40.0], [11.0, 21.0, 31.0, np.nan]],
...     dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": symbols},
... )
>>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
>>> rule.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
>>> inputs = DecisionInputs(
...     FrameDataset(xr.Dataset({"open": close, "close": close})), rule,
...     fill_column="open", valuation_column="close", rebalance_periods=1,
...     anchor=context.timestamp,
... )
>>> inputs.rebalances(context.timestamp)
True
>>> bar = inputs.context(
...     context.timestamp,
...     context.predictions,
...     xr.DataArray([0.25], dims="symbol", coords={"symbol": ["DDD"]}),
... )
>>> bar.locked.values
array([False, False, False,  True])
>>> decision = rule.decide(bar)
>>> decision.weights.values, decision.failure, decision.events
(array([0.375, 0.   , 0.375, 0.25 ]), None, {})

```

## 均值-方差优化

### 优化问题

在每个调仓 bar 上，`MeanVarianceOptimizer` 求解

    maximise    w'mu - risk_aversion / 2 * w'Sigma w - turnover_penalty * |w - w_current|_1

- `mu` 是每个标的的预期收益。
- `Sigma` 是它们的协方差。
- `w_current` 是 context 里的当前权重。

约束取决于 `direction`：

| `direction` | 约束 |
| --- | --- |
| `"long_only"` | `sum(w) = 1`，`0 <= w <= weight_cap`：满仓。 |
| `"long_short"` | `sum(w) = 0`，`sum(abs(w)) <= 1`，`abs(w) <= weight_cap`：美元中性。 |

多空组合中总敞口 1 是上限，不是目标。预期收益不足以抵偿风险和交易成本时，组合会有一部分不投资，最极端时不持有任何仓位。

换手惩罚按相对实际持有权重的变化收取，所以它定价的是真正会发生的交易。锁定仓位保持原有权重：它计入风险项，其他标的分配剩下的预算。

候选标的需要同时满足：

- 可交易；
- 未被锁定；
- 风险模型覆盖它；
- 有有限的预期收益预测，或者当前持有。

持有但没有预测的候选标的，预期收益记为 0.0，由它的换手成本决定是否平仓。其他标的权重都是 0.0。

### 第一次优化

优化器需要知道它读取的标签的 span 和尺度，回测会通过 `bind` 以标签规格的形式交给它。这里的规格描述 5 根 bar 的收益 `ret_5` 和 5 根 bar 的波动率 `vol_5`，两者都以标签自身的单位预测（`"raw"`，见[校准](#校准)）。

```python
>>> from quantlab.base.config import LedoitWolfConfig, MeanVarianceConfig
>>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
>>> from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
>>> from quantlab.base.portfolio import LabelSpec
>>> specs = [
...     LabelSpec(name="ret_5", scale="raw", delay=1, span=5),
...     LabelSpec(name="vol_5", scale="raw", delay=1, span=5),
... ]
>>> rng = np.random.default_rng(0)
>>> window = rng.normal(0.0, 1.0, size=(60, 4)) * [0.010, 0.015, 0.020, 0.025]
>>> context = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-26"),
...     predictions=xr.Dataset({
...         "ret_5": on_symbols([0.8, -0.1, -0.3, 0.2]),
...         "vol_5": on_symbols([0.05, 0.03, 0.04, 0.06]),
...     }),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.0]),
...     returns=xr.DataArray(window, dims=("timestamp", "symbol"), coords={
...         "timestamp": pd.bdate_range("2023-12-29", periods=60), "symbol": symbols}),
... )
>>> optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
... ))
>>> optimizer.bind(specs)
>>> optimizer.lookback_bars, optimizer.span
(60, 5)
>>> weights = optimizer.construct(context)
>>> weights.values.round(3)
array([0.4  , 0.4  , 0.009, 0.191])

```

组合满仓且不超过权重上限。预期收益这么小时，组合偏向低波动的 `AAA` 和 `BBB`。`optimizer.problem_inputs(context)` 返回候选标的及其 `mu`、`Sigma` 和当前权重，可以用来查看优化器为什么这样选。

### 校准

模型的预测通常是一个分数而不是收益：它能给标的排序，但尺度取决于模型本身，以及训练目标做过什么变换。`calibration` 决定预测如何变成预期收益 `mu`：

- `"grinold"`（默认）令 `mu = ic * sigma * z`。其中 `z` 是预测在候选标的上的截面 z-score，`sigma` 是每个标的在 span 内的波动率（取自协方差矩阵的对角线），`ic` 是模型的信息系数，例如一次滚动交叉验证的平均 IC。任何模型的输出都能接入，并且换模型时 `risk_aversion` 的含义不变。
- `"raw"` 直接把预测当作 `mu`，只适合预测值本身就是收益单位的模型。

每个预测器都会按标签报告预测是否用标签自身的单位：`label_scales` 把每个标签映射为 `"raw"` 或 `"standardized"`，标签规格以 `scale` 携带它。模型在未经变换的标签上训练时报告 `"raw"`。ensemble 对多个成员平均的标签报告 `"standardized"`，因为它会先对每个成员做 z-score。`bind` 会拒绝在 `"standardized"` 标签上使用 `"raw"` 校准：

```python
>>> import dataclasses
>>> ranked = [dataclasses.replace(specs[0], scale="standardized"), specs[1]]
>>> MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5", calibration="raw",
...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
...     risk_aversion=5.0,
... )).bind(ranked)
Traceback (most recent call last):
    ...
ValueError: calibration='raw' reads the prediction of 'ret_5' as a return, but its label spec reports its scale as 'standardized', not 'raw' (a model fitted on a transformed target, or a label an ensemble averages); use calibration='grinold'

```

### Span

`mu`、`sigma` 和 `Sigma` 都以 `expected_return_label` 的 span 为时间尺度：`ret_5` 是 5 根 bar。span 从标签规格读取，不需要配置；没有 span 的标签（不是 `Forward` 标签）会被拒绝。风险模型估计的是单 bar 收益的协方差，优化器把它乘以 span，因为方差随时间线性增长：

```python
>>> inputs = optimizer.problem_inputs(context)
>>> one_bar = LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)).estimate(context)
>>> bool(np.allclose(inputs.covariance, 5 * one_bar.covariance))
True

```

每隔一个 span 调仓一次（`rebalance_periods` 等于 span）时，优化器规划的时间跨度就等于组合实际持有的时间。

### 用模型预测波动率

波动率变化快，也比较好预测；相关性变化慢，更适合用历史估计。设置 `volatility_label` 后，优化器从模型的预测中取每个标的的波动率：

- 协方差变成预测波动率夹着风险模型的历史相关性：`Sigma = D C D`。
- Grinold 公式里的 `sigma` 就是这个预测值。

波动率标签（例如 `Volatility`）是 span 尺度的波动率。它的 span 必须与预期收益标签相同，尺度必须是 `"raw"`，`bind` 会检查这两点。

```python
>>> with_volatility = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5", volatility_label="vol_5",
...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
... ))
>>> with_volatility.bind(specs)
>>> covariance = with_volatility.problem_inputs(context).covariance
>>> np.sqrt(np.diag(covariance)).round(4)
array([0.05, 0.03, 0.04, 0.06])
>>> with_volatility.construct(context).values.round(3)
array([0.4  , 0.399, 0.   , 0.201])

```

没有有限正值波动率预测的标的没有风险估计，处理方式和历史数据不足的标的相同，见[风险模型](#风险模型)。

两个标签必须来自同一个预测器。一个由收益模型和波动率模型组成的 `ModelEnsemble` 就能做到：只有一个成员预测的标签会原样传出，并保留该成员的尺度（见[模型](model.md)）。

### 多空组合与候选池

`direction="long_short"` 时组合是美元中性的：

```python
>>> long_short = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4, direction="long_short",
... ))
>>> long_short.bind(specs)
>>> weights = long_short.construct(context)
>>> weights.values.round(3)
array([ 0.4  , -0.212, -0.261,  0.073])
>>> bool(abs(weights.sum()) < 1e-12), float(abs(weights).sum().round(3))
(True, 0.945)

```

对上千个标的在每次调仓时都做优化会很慢。`candidate_top_k` 把每根 bar 限制在一个候选池里：

- `mu` 最大的 `candidate_top_k` 个标的（多空时按 `abs(mu)` 取最大）；
- 加上当前持有的所有标的，这样跌出前列的持仓仍能按换手成本平仓。

候选池以外的标的权重为 0.0。Grinold 校准的 z-score 在截取候选池之前、对全部候选标的计算，所以 `mu` 不受 `candidate_top_k` 影响。

## 风险模型

风险模型在一根 bar 上估计单 bar 收益的协方差。它的 `estimate(context, volatility=None)` 返回一个 `CovarianceEstimate`：覆盖的标的及其协方差。风险模型是估计器，每根 bar 都根据 context 里的内容重新计算，不需要训练，也没有 checkpoint。喂给它的预测（例如预测波动率）来自模型，经由预测器传入。

`LedoitWolfRiskModel` 读取最近 `lookback_bars` 个单 bar 收益，用 Ledoit-Wolf 系数把样本协方差向缩放的单位阵收缩（`sklearn.covariance.ledoit_wolf`），这样标的数多于 bar 数时估计仍然良态。然后它把协方差拆成相关性和波动率；传入 `volatility` 时用给定的波动率替换。

标的只有同时满足以下条件才会被覆盖：

- 窗口内每个收益都是有限值；
- 这些收益不全相等；
- staleness 不超过 `max_stale_bars`（默认 5）。

其他标的不在估计里，优化器也不会选它们。风险模型不覆盖的持仓会被平掉，并在这一行里报告为 `closed_without_risk` 事件：

```python
>>> short_history = window.copy()
>>> short_history[:5, 2] = np.nan  # CCC listed five bars into the window
>>> import dataclasses
>>> late = dataclasses.replace(
...     context,
...     current_weights=on_symbols([0.0, 0.0, 0.3, 0.0]),
...     returns=context.returns.copy(data=short_history),
... )
>>> weights = optimizer.construct(late)
>>> weights.values.round(3)
array([0.4, 0.4, 0. , 0.2])
>>> weights.attrs["events"]
{'closed_without_risk': ['CCC']}

```

### 预留：因子风险模型

接口为因子风险模型留出了位置，但目前还没有实现：

- 风险模型可以返回 `FactorCovarianceEstimate`，即因子形式的协方差 `B F B' + diag(D)`：暴露 `B`、因子协方差 `F` 和特异方差 `D`。它的 `factor_form()` 让优化器把风险写成 `|F^(1/2) B' w|^2 + w' diag(D) w`，不需要构造稠密矩阵。
- 风险模型可以在 `required_factors()` 中声明它要读取的 `Factor` 面板，例如暴露。回测在自己的窗口上计算这些因子（各自带预热），并把它们在这根 bar 上的值放进 `context.factors`。

接入这样的模型时，优化器、驱动循环和回测器都不需要改动。

## 完整示例

`examples/wrds_us_equity/sp500_xgb_mvo.py` 是一个 S&P 500 指数增强示例：它在 point-in-time 的指数成分股上跑完整流程。

1. Alpha101 和 Alpha158 因子。
2. 一个 `Return` 标签和一个 `Volatility` 标签，都是 5 根 bar。
3. 由两个 `XGBoostRegressor` 组成的 `ModelEnsemble`，每个标签一个模型。
4. 在当天成分股上运行的均值-方差优化器，设置了 `volatility_label`，候选池 200 个标的，每 5 根 bar 调仓，与买入持有的 SPY 对比。被调出指数的股票按最后收盘价结算，与退市相同。

它需要 `scripts/wrds/index.py --index sp500` 和 `scripts/wrds/etf.py --etf spy` 生成的 S&P 500 store 和 SPY store（见[示例说明](../../examples/wrds_us_equity/README.zh-CN.md)）。

在训练服务器上（一块 GPU；提交 `211b324`，2026-09-30；store 就绪后约 8 分钟）跑一次，样本外窗口 2020–2024 的结果如下：

| 模型 | 测试集 IC | 测试集 Rank IC |
| --- | --- | --- |
| 收益模型（`ret_5`） | -0.0005 | -0.0004 |
| 波动率模型（`vol_5`） | 0.513 | 0.481 |

| 回测 | 策略 | SPY |
| --- | --- | --- |
| 总收益 | 41.6% | 96.8% |
| Sharpe | 0.51 | |
| 最大回撤 | 32.3% | |
| 年化超额收益 | -6.4% | |
| 跟踪误差 | 10.1% | |
| 信息比率 | -0.75 | |
| Beta | 0.68 | |

没有一次调仓失败，也没有订单被拒。有 10 次调仓一共平掉了 11 个持仓，原因是风险模型没有它们的估计（`closed_without_risk`）。

这些数字说明流程能跑通，不代表一个有效的策略。收益模型没有样本外预测能力：测试集 IC 为零。`mu` 里没有信号时，权重由风险项决定，组合变成一个 beta 为 0.68 的低波动组合，在上涨的市场里跑输指数。波动率模型确实有信息：它的 IC 是预测波动率与实际波动率的相关，衡量的是对风险的排序能力，而不是 alpha。此外，优化器既不约束跟踪误差，也不约束相对指数的主动权重，所以这是一个在指数成分股里选股的组合，还不是受控的指数增强。

## 编写规则

继承 `PortfolioConstructor`：

1. 把 `config_cls` 设为规则参数的 frozen dataclass。
2. 实现 `construct`。
3. 规则读取收益窗口时重写 `lookback_bars`（需要多于 `lookback_bars + 1` 个原始价格时再重写 `history_bars`），读取因子面板时重写 `required_factors`，需要检查标签规格时重写 `bind`。

不要重写 `decide`：它是回测与执行器共用的唯一决策路径。规则不含组装代码，它的 context 由 `DecisionInputs` 构造。

`get_config` 和 `from_config` 把规则序列化为配置的各字段加上类的导入路径。字段里如果是另一个组件（例如风险模型），会嵌套序列化，所以重建一次运行不需要额外代码。

新的风险模型同样继承 `RiskModel` 并实现 `estimate`。见[扩展 quantlab](../developer-guide/extending.md)。

## 另请参阅

- [回测](backtest.md)：`constructor` 字段、成交方式和运行目录。
- [模型](model.md)：`label_scales`，以及组合收益模型与波动率模型的 ensemble。
- [因子](factor.md)：`Return` 和 `Volatility` 标签。
