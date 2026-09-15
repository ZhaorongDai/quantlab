# 回测层（Backtest）

> 代码位置：`quantlab/base/backtest.py`（抽象基类 `BaseBacktester`、市场规格 `MarketSpec`、结果类
> `SimulationResult` / `BacktestResult` / `CVBacktestResult`）、`quantlab/base/config.py`（`BacktestConfig`、
> `CrossSectionBacktestConfig`）、`quantlab/backtest/engine_vectorbt.py`（vectorbt 引擎层 `VectorBtBacktester`）、
> `quantlab/backtest/selection.py`（`rebalance_mask`、`resolve_score_label`、`CrossSectionTopNSelector`）、
> `quantlab/backtest/us_equity.py`（`US_EQUITY_MARKET`、`USEquityCrossectionSelectStockVectorBt`）、
> `quantlab/utils/module.py`（`load_backtester_from_config`）。
> 辅助的叶子模块：`quantlab/utils/fingerprint.py`（数据指纹）、`quantlab/utils/backtest_report.py`（HTML 报告）。
> 行为由 `tests/test_backtest_*.py` 锁住。本文所说的 D-xx 是阶段 03.7 的决定编号，原文在
> `.planning/phases/03.7-cross-sectional-backtester-basebacktester-abc-usequitycrosse/03.7-CONTEXT.md`。

---

## 一句话

回测器把一个训练好（或现场训练）的收益模型变成一次**可复现**的回测：按回测窗口对齐因子日期并预测，
把预测变成**目标权重**，在下一个 bar 的开盘价成交，算出整段、样本内、样本外三块指标，把配置、权重、
净值、指标、报告和数据指纹写进一个新的运行目录。这个目录里的 `config.json` 能原样重建回测器并重跑出
同一条曲线。对外只有两个入口：`run()`（一个模型的回测）和 `run_cv()`（回放一次 `train_cv` 的逐折样本外回测）。

---

## 类层次

**引擎靠继承变化，市场和选股逻辑靠组合变化（D-01）。** 不这样拆的话，「市场 × 风格 × 引擎」会乘出一堆类。

```
BaseBacktester                         quantlab/base/backtest.py      模板方法 run() / run_cv()，全部与引擎无关的步骤
└── VectorBtBacktester                 quantlab/backtest/engine_vectorbt.py   引擎层：模拟、统计、切片统计；仍是抽象类
    └── USEquityCrossectionSelectStockVectorBt   quantlab/backtest/us_equity.py  具名组合：美股市场规格 + 截面 TopN 选股
```

各层还剩哪些抽象成员，由 `tests/test_backtest_contracts.py::test_abstract_method_sets_are_exact` 钉成精确集合：

| 类 | 抽象成员 |
|---|---|
| `BaseBacktester` | `config_cls`、`_generate_signals`、`_simulate`、`_simulate_benchmark`、`_engine_stats`、`_period_returns_stats` |
| `VectorBtBacktester` | `config_cls`、`_generate_signals` |
| `USEquityCrossectionSelectStockVectorBt` | 无 |

具体类只声明三样东西：

- `config_cls = CrossSectionBacktestConfig`：`config` setter 的**第一条语句**就检查它，类型不对立刻 `TypeError`；
- `MARKET = US_EQUITY_MARKET`：一个 `MarketSpec`，成交价列、估值价列、年化口径都在这里（见「价格与成交」）；
- `_generate_signals`：把预测交给一个 `CrossSectionTopNSelector`。打分标签和选股组件在 `_validate_config` 里构造期就解析好。

**预留但没有实现的兄弟类名**：`USEquityTimeseriesVectorBt`（时序策略，同一个 vectorbt 引擎）和
`USEquityCrossectionEventDrivenBt`（事件驱动，NautilusTrader，归 Phase 6 BT-02）。加它们不需要改基类：
`tests/test_backtest_engine.py` 里有一个测试内的 `EqualWeightEveryone(VectorBtBacktester)`，只声明
`config_cls`、`MARKET`、`_generate_signals` 三样就能完整跑通 `run()`。

`run` / `run_cv` / `get_config` 是 `BaseBacktester` 仅有的公开方法，子类**从不覆盖**它们
（`test_run_lives_only_on_base_backtester` 锁）。可变的部分只有上表那些下划线钩子。

---

## run() 的七步

`run()` 是基类上的模板方法（D-02），顺序固定：

| # | 步骤 | 实现 |
|---|---|---|
| 1 | 准备模型（训练或加载） | `_prepare_model`，见「模型准备与日期对齐」 |
| 2 | 对齐因子日期，然后预测 | `_align_and_predict`：`_redate_factors`（预热 + 强制重读 + 记因子指纹）→ `model._collect_all_features()`（只算特征，不算标签）→ `model.predict_panel` → 切回回测窗口 |
| 3 | 生成信号 | `_load_prices` 读两列价格 → 预测 `reindex` 到价格的全部 timestamp / symbol → `_generate_signals` → `_assert_weights_contract` |
| 4 | 模拟 | `_simulate`（引擎层） |
| 5 | 基准 | `_simulate_benchmark`，本阶段恒为 `None`（D-08） |
| 6 | 指标 | `_split_window`（样本内外划分）+ `_compute_metrics` |
| 7 | 报告与落盘 | `_compare_fingerprints` → `_report_and_persist`；`use_wandb=True` 时再 `_log_to_wandb` |

第 2 到第 6 步封装在 `_backtest_window` 里，`run()` 调一次，`run_cv()` 每折调一次，所以两条入口的单窗口逻辑只有一份。

返回 `BacktestResult`：`run_dir`、`predictions`（已铺到价格轴）、`weights`、`simulation`（`SimulationResult`：
`value`、`returns`、`orders`、`trades`、`liquidations`、`bar_interval`、引擎原生对象 `native`）、`metrics`。

## run_cv()

`run_cv()` **回放**一次已经跑完的 `train_cv`：每折加载该折自己的 checkpoint，只回测该折的样本外测试段，
用来检验模型 CV 的真实交易能力（D-16、D-35、D-36）。

**前提。** `model_mode="load"`，并且 `cv_project_dir` 指向 `train_cv` 的项目目录（里面有 `cv_folds.json`，
格式见 `example/model.md` 的「cv_folds.json」一节）。`model_mode="train"` 会被拒绝：`run_cv` 从不训练。
构造期只要求 `load` 模式下 `checkpoint` 与 `cv_project_dir` **至少有其一**；缺的恰好是某个入口要的那个时，
由那个入口在运行时报错（`run()` 要 `checkpoint`，`run_cv()` 要 `cv_project_dir`）。

**路径不依赖工作目录（2026-09-15，代码审查 WR-03）。** 以前 `train_cv` 把 `model_save_dir` 原样拼进清单，
相对的 `model_save_dir` 就写出相对路径，`run_cv` 再按**当前工作目录**解析：换个目录运行找不到，
更糟的是会加载工作目录下另一次训练的同名 checkpoint。现在：

- `train_cv` 在清单与返回值里写 checkpoint 的绝对路径；
- `run_cv` 按 `{cv_project_dir}/{实验目录}/{文件名}` 在项目目录下找每折的 checkpoint（项目整体搬走也找得到），
  找不到才接受记录里本身存在的绝对路径，从不按工作目录解析；两处都没有时 `FileNotFoundError`；
- `BacktestConfig` 的 `checkpoint`、`cv_project_dir`、`output_dir` 在构造时规范成绝对路径，所以 `config.json` 里记的也是绝对路径。

**顺序。**

1. `_read_cv_folds` 读清单并校验：文件不存在 `FileNotFoundError`；没有 `format_version`、版本不等于
   `BaseModel.CV_FOLDS_FORMAT_VERSION`（目前是 1，`True` 也拒收）、`folds` 不是非空 list、某折缺
   `fold` / 四个日期 / `checkpoint`、测试段起点晚于终点，都是 `ValueError`。四个日期统一规范成 ISO 日期。
2. `_select_folds` 只留**测试段整段**落在 `[config.start_date, config.end_date]` 内的折；一个都不剩时报错。
3. `_assert_contiguous_folds` 在价格日历上断言相邻折的测试段首尾相接：后一折的首 bar 必须恰好是前一折末 bar 的下一个。
   有缺口报「gap」并写出有几个 bar 不属于任何折，重叠报「overlap」并写出几个 bar 会被两个模型各交易一次。
   **这一步先于任何模型加载和模拟**：拼接一个有缺口或重叠的序列，得到的曲线不对应任何真实交易路径。
4. 逐折：加载该折 checkpoint，`_backtest_window` 回测该折测试段（含预热）。样本内外按**该折自己的**
   train 日期加标签期限划分（D-17 逐折），所以折间 `gap_periods=0` 时每折开头的标签期限个 bar 是样本内，每折各报一条 warning。
5. 拼接（D-35）：各折权重沿 `timestamp` 拼起来，对「首折 test_start .. 末折 test_end」的价格跑**一次**连续模拟，
   资金在折边界**不重置**。逐折指标仍来自第 4 步各自独立的模拟，那些模拟每折都从 `init_cash` 起步。
6. 指纹覆盖整个拼接窗口（首折预热起点 .. 末折 test_end）后比对，落盘，可选 wandb（记拼接曲线的指标）。

返回 `CVBacktestResult`：`run_dir`、`folds`（每折的清单字段加该折的 `predictions` / `weights` / `simulation` / `metrics`）、
拼接后的 `weights` / `simulation`、`metrics`。

**CV 运行目录。** 顶层描述拼接曲线，文件名与 `run()` 的运行目录同名同义；逐折结果放在子目录：

| 位置 | 内容 |
|---|---|
| `config.json` | `get_config()`，含 `data_fingerprint` |
| `weights.zarr`、`equity.zarr` | 拼接权重；拼接模拟的 `value` 与 `returns` |
| `folds/fold_{i}/weights.zarr`、`folds/fold_{i}/equity.zarr` | 第 i 折（清单里的折号）独立模拟的权重与净值 |
| `metrics.json` | `stitched`、`folds`、`notes` 三个键 |
| `liquidations.json` | `{"stitched": [...], "folds": [{"fold": i, "liquidations": [...]}, ...]}` |
| `fingerprint.json` | 拼接窗口的指纹 |
| `report.html` | 拼接曲线 |

`metrics.json` 的 `stitched` 块与 `run()` 的指标同构，但划分键不同：是 `training_windows`（每折一个）、
`in_sample_ranges`（每折开头的样本内段，**多段**）和 `out_of_sample_ranges`，**没有**单段的 `in_sample_range`，
因为多段样本内塞不进一个日期对。`folds` 里每项是六个清单字段（`fold`、四个日期、`checkpoint`）加该折的 `metrics`，
与 `run()` 的指标同构。

**拼接报告不涂样本内。** `report.html` 只能画一段阴影，而拼接曲线的样本内是每折一小段，所以拼接报告
一段也不涂；`notes` 里专门有一条说明样本内段记在 `metrics.json` 的 `stitched.in_sample_ranges` 里。

---

## 权重契约

**信号生成的产出是目标权重（D-03）。** 这是回测器与选股（以及 Phase 5 的组合优化器）之间的接口，
也会原样落盘成 `weights.zarr`，将来的事件驱动引擎可以直接复用。所以它是一个代价高的契约，改它要动选股、所有引擎和已存的运行。

- 一个 `xr.Dataset`，变量 `weight`，维度严格是 `("timestamp", "symbol")`，两个轴与价格数据完全相同；
- **非调仓 bar：整行 NaN**，意思是「保持现有仓位」；
- **调仓 bar：整行有限值**，没被选中的标的写 `0.0`；
- 调仓行的毛敞口 `sum(|w|) <= 1`（容差 1e-9），即不加杠杆；
- 可选数为 0 的调仓行是整行 `0.0`，也就是清仓，不是 NaN。

`_assert_weights_contract` 在模拟前检查以上全部，引擎的 `_simulate` 在任何 pandas 转换之前**再**检查一次
「每行要么全 NaN、要么全有限」（直接调 `_simulate` 的人会绕过基类那道检查）。

**为什么调仓行上绝不能出现 NaN。** vectorbt 把 NaN 读成「这个标的保持原仓位」。实测：调仓行 `[1.0, NaN]`
（A 买满、B 保持）一笔订单都没有成交，B 占着资金，A 买不进去，而且不报任何错（`03.7-RESEARCH.md` Pitfall 3）。
所以调仓行上「不持有」必须写成 `0.0`，混着 NaN 与有限值的一行直接 `ValueError`，写出该行的时间戳。

---

## 价格与成交

**价格列来自市场规格，不写在方法里（D-04）。** 美股的规格是：

```python
US_EQUITY_MARKET = MarketSpec(
    fill_price_column="adjOpen",        # 成交价：复权开盘价
    valuation_price_column="adjClose",  # 估值价：复权收盘价
    trading_days_per_year=252,
    session_minutes_per_day=390,
)
```

两个列名只出现在这个实例上，回测层任何方法体里都不写列名（`test_price_column_literals_never_appear_inside_a_method_body`
扫源码锁住）。价格经 `dataset.read(overwrite=True).get_xarray_dataset()` 取出这两列并深拷贝。
不复权的 `open` / `close` 从不参与盈亏计算。

**成交时点：bar t 收盘形成的信号，在 bar t+1 开盘成交（D-05）。** 实现就是把权重整体后移一个 bar：

```python
vbt.Portfolio.from_orders(
    close=adjClose.ffill(), price=adjOpen.ffill(), size=weights.shift(1),
    size_type="targetpercent", direction="both",
    group_by=True, cash_sharing=True, call_seq="auto",
    fees=config.fees, slippage=config.slippage, init_cash=config.init_cash,
    freq=bar 间隔,
)
```

- `direction="both"`：多空翻转在一次调仓里完成。
- **目标百分比的基数按下单价计**：vectorbt 的 `val_price` 默认是「当前订单价」，所以目标 1/6 是按整组资产在
  **t+1 开盘价**上的估值算的，不是 t 收盘。
- 滑点作用在成交价上（买入价 ×(1+slippage)，卖出价 ×(1−slippage)），手续费 = 成交数量 × 滑点后价格 × fees；
  目标仓位的数量已经把手续费算进去（Pitfall 4 的实测）。
- bar 间隔取价格时间戳差分的众数；少于 2 个 bar 无法模拟，直接报错。
- ~~年化口径来自市场规格：日及以上频率每年 bar 数 = 252 × (一天 / bar 间隔)，日内频率 = 252 × 390 / bar 分钟数。~~
  vectorbt 默认按 365 天年化，对美股是错的，所以整段统计和切片统计都显式传了 `year_freq`。

  > **更正（2026-09-15，代码审查 CR-02）：上面删除线那句的「日及以上频率」公式已作废，保留原文作记录。**
  > 它把交易日计数（252）除以日历日间隔：周线（7 天）只得 36 个 bar/年、31 天月线只得 8.13，周线 Sharpe / Sortino
  > 被低估约 sqrt(52/36) 倍，Calmar、年化收益、年化换手一起错。现在的口径（`MarketSpec.year_freq`）：
  >
  > | bar 间隔 | 每年 bar 数 |
  > |---|---|
  > | 短于一天（日内） | 252 × 390 / bar 分钟数（未变） |
  > | 恰好一天 | 252（交易日历上的日线，一个 bar 是一个交易日） |
  > | 长于一天 | `min(252, 365.25 / bar 天数)`：一个 bar 是一段日历跨度，周线约 52.18，30 天约 12.18，31 天约 11.78 |
  >
  > 长于一天按日历跨度计，是因为周线、月线每个日历周期一个 bar，节假日不会让某一周消失；封顶 252 保证一个 bar 不短于一个交易日，
  > 也让这个分段在一天处连续、随间隔单调不增。代价：一个「每隔 N 个交易日取一个 bar」的序列（众数间隔 2 天或 3 天）也会按日历跨度年化，
  > 比按交易日计多算约 40%。这种重采样本项目目前没有产出路径。

---

## 标的池与退市

**标的池就是价格数据集里的全部标的（D-06）。** 本阶段不做指数成分过滤。预测在选股之前按**标签**
（不是按位置）`reindex` 到价格数据集的 timestamp 与 symbol 上：价格里有、预测里没有的标的得到 NaN 分数，也就不可选。

**退市规则：持仓标的的价格变成 NaN 后，沿用最后价格，并在下一个调仓 bar 强制平仓（D-07）。**

为什么必须这样：vectorbt 遇到持仓标的价格为 NaN 时，会按最后价值继续持有它，并且**整组**之后的所有调仓都被
悄悄跳过，冻结的不只是退市的那一只（Pitfall 2 实测）。所以引擎在模拟前把成交价与估值价两列都做 `ffill`。
ffill 之后，退市标的在下一个调仓 bar 按最后价格被卖掉，其余标的照常调仓。

判定一次强制平仓：调仓行 t（且 t+1 仍在窗口内）、t 收盘时该标的持仓非零（由订单记录累计得出）、
t+1 的**原始**（未 ffill）成交价是 NaN。开头价格为 NaN、此前从未持有过的标的（晚上市）不算退市。
选股时「t+1 有没有成交价」也用原始价格判断，所以已经退市的标的不会被重新选进来（D-12）。

每次强制平仓在 `SimulationResult.liquidations` 里记一条，并打一行 INFO 日志。落盘到 `liquidations.json` 时
时间戳经 `to_jsonable` 变成 ISO 字符串，一条记录长这样（下面是 `to_jsonable` 对一条手写记录的实际输出，
不是某次回测的结果）：

```python
{'symbol': 'CCC', 'signal_timestamp': '2024-02-01T00:00:00', 'fill_timestamp': '2024-02-02T00:00:00', 'price': 41.2}
```

- `symbol`：标的名；
- `signal_timestamp`：发出平仓信号的调仓 bar t；
- `fill_timestamp`：成交 bar t+1；
- `price`：t+1 上 ffill 后的成交价，即该标的最后一个有限的复权开盘价。

注意从价格变 NaN 到下一个调仓 bar 之间，这个仓位按最后价格冻结着估值；`rebalance_periods` 越大，这段越长。

---

## 选股规则

`CrossSectionTopNSelector(direction, top_n)` 在每个调仓 bar 上做截面等权选股，规则由 `tests/test_backtest_selection.py` 锁住：

| 规则 | 内容 |
|---|---|
| 方向（D-09） | `long_only`：分数最高的 k 个各 `1/k`，合计 100%。`long_short`：最高 k 个各 `+0.5/k`，最低 k 个各 `-0.5/k`，毛敞口 100%、净敞口 0 |
| 数量（D-10） | 固定 `top_n`，没有分位数模式 |
| 打分（D-11） | `score_label` 是模型的一个标签名；`None` 取模型第一个标签；未知名字在**构造期**就 `ValueError`，先于任何训练 |
| 可选（D-12） | 分数有限（NaN、inf 都不行）并且 t+1 的原始成交价有限 |
| 不足 top_n（D-12） | `long_only` 的 k = min(top_n, 可选数)；`long_short` 的 k = min(top_n, 可选数 // 2)，所以两本书永远不共享标的。k < top_n 时 warning 写出该 bar 时间戳；k == 0 时整行 0.0 |
| 平分 | 稳定排序，平分时按标的轴顺序决定，同一面板永远得到同一组权重（复现要求，D-25） |

**调仓日程（D-18）。** 锚点是回测窗口的第一个 bar，之后每 `rebalance_periods` 个 bar 调仓一次，中间持仓不动。
`rebalance_periods` 是 bar 数，与频率无关（日线上 5 就是一周左右）。

**最后一个 bar 从不调仓。** 它的信号在窗口内没有 t+1 成交 bar，`shift(1)` 会把它丢掉，所以 `rebalance_mask`
直接把最后一个 bar 标成非调仓（Pitfall 13）。`run_cv` 里每折测试段的最后一个 bar 同理。

---

## 模型准备与日期对齐

**`model_mode`（D-13）。**

- `"train"`：`model.collect()` 再 `model.train()`，用的是**模型自己**配置里的 `train_start` / `train_end` /
  `test_start` / `test_end`。回测窗口从不写进这些日期：回测窗口只决定预测区间和样本内外划分，
  改写它们会让训练集跟着回测参数漂移。
- `"load"`：先检查 `config.checkpoint` 文件存在（缺了直接 `FileNotFoundError`，不白算一遍特征），
  然后只对 `DLModel` 先把特征面板放进模型的 data backend，最后 `model.load(checkpoint)`。

  **核对 checkpoint 自己的记录（2026-09-15，代码审查 WR-01）。** 加载前读 checkpoint 旁由训练写下的 `config.json`：
  - 因子与标签的变量名（含顺序）必须与 `config.model` 一致，否则 `ValueError`，写明两边的变量名。
    xgboost 只核对特征**个数**，同样个数、不同因子（或不同顺序）训练出来的 checkpoint 以前会对错位的特征悄悄给出预测。
  - D-17 的样本内判定用**记录里**的 `train_start` / `train_end`：它们才是这个模型真正训练过的日期。
    与 `config.model` 的日期不同时 `logger.warning` 写明两对日期，然后用记录的日期；`run_cv` 每折同样核对，日期以 `cv_folds.json` 为准。
  - 没有 `config.json`（比如手工拷贝的 checkpoint）时 `logger.warning` 说明无法核对，照 `config.model` 原样继续。

**DL checkpoint 需要先有面板。** `DLModel` 的 `.pth` 只有权重，加载时要按 `num_symbols` 重建网络，而
`num_symbols` 读的正是 data backend，空着会报 `Please cal 'read' or 'to_internal' first.`（Pitfall 11）。
回测器替你做了 `model.data_backend.to_internal(model._collect_all_features())`。`MLModel` 的 `.joblib`
就是完整模型，跳过这一步。

**日期对齐靠直接改因子配置（D-14）。** 不另开一条切片或取数的路：对每个因子设
`factor.config.start_date` / `end_date`，调 `_reset_dataset_config()` 把日期推给它的数据集，然后**强制重读**：

- 数据集总是 `read(overwrite=True)`；
- `factor_data_strategy == "read"` 时，因子库本身也 `factor.read(overwrite=True)`。

强制重读是必须的：`XrBackend.read` 一旦已经持有数据就直接返回，`read()` 之后的过滤又是**就地**收窄这份缓存。
train 模式下模型先按自己的日期 collect 过，再把因子日期放宽到「预热 + 回测窗口」时，不强制重读拿回的仍是那段
更窄的数据：不报错，只是缺 bar（Pitfall 1）。预测只算**特征**（`_collect_all_features`），不算标签。

**预热按 bar 计（D-15）。** 取所有因子 `config.window` 的最大值，在**价格数据集自己的交易日历**上从回测起点往前数
这么多个 bar，作为因子的起始日期；预测完再切回回测窗口。日历是把价格数据集从最早日期读到回测终点得到的时间轴。
历史不够时截到第一个 bar 并 warning，写明要多少、有多少、差多少、截到哪天。

所以 `window` 要写成因子真实的回看长度（bar 数）：上面例子里 5 日动量因子的 `window=5`。
`Factor._reset_dataset_config` 自己还会从数据集起点再减 `window` 个**日历日**，那只是额外缓冲，回测器不依赖它。

**预测面板。** `model.predict_panel(features)` 进出都是 `(timestamp, symbol)` 的 `xr.Dataset`，每个标签一个变量；
所有特征都是 NaN 的位置预测为 NaN（否则 xgboost 会给还没上市的标的算出有限分数）。各种模型头怎么适配见
`example/model.md` 的「面板预测：predict_panel」。注意 `RNNClassifier` 给出的分数是**上涨概率**，不是收益。

---

## 样本内外

**有效训练窗口是 `[train_start, train_end + 标签期限]`（D-17）。** 标签期限是模型所有标签
`config.kwargs["n_forward_periods"]` 的最大值，单位 bar：`train_end` 那一 bar 的标签读的是之后 n 个 bar 的价格，
所以这 n 个 bar 也见过训练。期限在价格日历上按 bar 数加，不做日历日加法（周五加 2 个 bar 是下周二）。

- 标签的 `kwargs` 里没有 `n_forward_periods`：warning 写明标签类名，按 0 计，不猜；
- 模型没有 train 日期：`training_window` 记 `null`，整段算样本外，并 warning。

回测窗口与有效训练窗口重叠时，`logger.warning` 写明两个窗口，回测**照常继续**，指标分开报：

| `metrics.json` 的键 | 内容 |
|---|---|
| `training_window` | 有效训练窗口的 bar 标签对（午夜的 bar 写日期，其余写完整 ISO 时间，见本节末尾的更正），或 `null` |
| `in_sample_range` | 重叠部分的首尾 bar（两个区间的交集，必然是一段），或 `null` |
| `out_of_sample_ranges` | 重叠之外的连续段，0、1 或 2 段 |
| `whole` | 引擎的整段统计：vectorbt `Portfolio.stats()` 全套指标（去掉 `benchmark_return`），加 `turnover` |
| `in_sample` / `out_of_sample` | 切片统计，没有对应区间时是 `null` |
| `notes` | 附带说明，见下文 |

**为什么切片都来自同一次模拟（D-17、D-34）。** vectorbt 的分组 `Portfolio` 不能按时间切片（`pf.loc[a:b]` 直接
`IndexingError`，Pitfall 5）；把样本内外各跑一次模拟又会重置资金、改变路径，得到的不是同一个策略。所以：

- `whole` 是完整的 `pf.stats()`；
- 切片块是**同一次模拟**的收益序列截到区间后，用 vectorbt 收益访问器算的收益类统计（`Total Return [%]`、
  `Sharpe Ratio`、`Max Drawdown [%]` 等），加上按时间过滤的记录统计：`order_count`、`fees_paid`、
  `traded_notional`、`closed_trade_count`（平仓时间在段内）、`open_trade_count`（段末仍未平仓）、`turnover`。
  两段样本外时，收益按时间顺序拼接后计算，记录统计逐段相加。

~~日期比较按天做：日内数据上 `train_end` 那一天的所有 bar 都算样本内，这是偏保守的方向。~~

> **更正（2026-09-15，代码审查 CR-01）：上面这句已作废，保留原文作记录。** 它说的方向是错的，而且那种做法本身有缺陷。
> 旧实现先把 `train_end` 截成当天午夜，再在日历上找「最后一个不晚于午夜的 bar」。日内数据上那一天的 bar 都晚于午夜，
> 所以找到的是**前一个交易日**的最后一个 bar，加上标签期限后又截成日期、按天比较。结果是 `train_end` 当天整天算样本内，
> 而模型最后几个训练标签真正读过的、落在**下一个交易日**开头的那期限个 bar 被算成了样本外，样本外指标因此被训练信息污染。
> 这是不保守的方向。日线不受影响，因为日线的 bar 就在午夜。
>
> 现在的做法：
> - 训练段用与模型层 `data.sel(timestamp=slice(train_start, train_end))` 同一个 pandas `slice_indexer` 定位。
>   `"2024-05-17"` 包含当天全部 bar，`"2024-05-17T13:00"` 只到 13:00；`cv_folds.json` 里的纳秒字符串精确匹配。
> - 终点是训练段最后一个 bar 再往后数期限个 bar。
> - `training_window`、`in_sample_range`、`out_of_sample_ranges` 的端点是 bar 标签：午夜的 bar 写日期（所以日线的输出与以前一样），
>   其余写完整 ISO 时间（如 `2024-01-03T11:00:00`）。
> - 样本内外划分、切片收益、订单与交易的按段过滤，一律按精确的 bar 时间戳比较，不再按天。

**换手率的口径（D-22，由 `_turnover` 定义）。** 每个有成交的 bar：

```
换手率 = 该 bar 所有订单的 |size| × 成交价 之和 / 上一个 bar 的组合净值
```

窗口第一个 bar 的分母用 `init_cash`。这是单边口径：从空仓全仓买入约为 1，整个组合换成另一批标的（先卖后买）约为 2。
分母用成交前一个 bar 的净值，这样换手率不含成交当 bar 的盈亏。汇总成三个数：`mean_per_rebalance`（有成交 bar 的均值）、
`sum`、`annualized`（均值 × 每年 bar 数 / `rebalance_periods`）。没有成交时均值与年化是 `null`，总和是 0。

---

## 费用与假设

默认值（D-19），全部可以在 `BacktestConfig` 里改：

| 字段 | 默认 | 含义 |
|---|---|---|
| `fees` | `0.0005` | 5bp，按成交额 |
| `slippage` | `0.0005` | 5bp，作用在成交价上 |
| `init_cash` | `1_000_000.0` | 初始资金 |

其余假设：

- **允许零碎股（D-20）**，目标权重按资金比例精确成交；
- **不计融券费与做空融资成本（D-21），所以空头一侧的收益是偏乐观的。** 这句话同时写在 `metrics.json` 的
  `notes` 和 `report.html` 底部：`No borrow or short-financing cost is modelled, so short-side returns are optimistic.`
  `long_short` 的结果尤其要带着这个折扣读；
- 收益用复权价计算，拆股与分红已经体现在 `adjOpen` / `adjClose` 里；
- 构造期校验：`rebalance_periods >= 1`、`fees` 与 `slippage` 非负、`init_cash > 0`、`start_date` 不晚于 `end_date`。

---

## 输出目录

每次 `run()` 建一个新目录 `{output_dir}/{类名}_{YYYYmmdd_HHMMSS_ffffff}/`（D-24）。时间戳带微秒，同一秒内的两次运行不会撞名；
目录已存在就 `RuntimeError`，**从不覆盖**。所有 JSON 先经 `to_jsonable`（NaN / inf 写成 `null`，时间写成 ISO 字符串）
再原子写入，所以都是严格 JSON。

**运行目录要么完整、要么不存在（2026-09-15，代码审查 WR-08）。** 以前先建目录、先写 `config.json`，之后 zarr、指标、报告、指纹
任何一步失败（zarr 写错、plotly、磁盘满、Ctrl-C），都会留下一个带着合法 `config.json`、却没有指标和指纹的目录，
看起来和跑完的一样，`load_backtester_from_config` 也会照样去「复现」它。现在 `run()` 与 `run_cv()` 都先把全部产物写进
同一父目录下的隐藏暂存目录 `.{运行目录名}.partial`，写完才改名成运行目录。中途出任何异常（含 `KeyboardInterrupt`），
都会删掉这个暂存目录再原样抛出。所以 `output_dir` 下看得到的运行目录一定带着全部产物。

| 文件 | 内容 |
|---|---|
| `config.json` | `get_config()`：`CrossSectionBacktestConfig` 的标量字段，加上逐个嵌套的 `price_dataset`、`model`（含因子、标签、`checkpoint` 引用）、`benchmark_dataset` 配置；另有顶层 `data_fingerprint` |
| `weights.zarr` | 目标权重，`weight` 变量，非调仓行的 NaN 原样保留 |
| `equity.zarr` | `value`（组合净值）与 `returns`，维度 `timestamp` |
| `liquidations.json` | 强制平仓记录的 list |
| `metrics.json` | `whole`、`in_sample`、`out_of_sample`、`training_window`、`in_sample_range`、`out_of_sample_ranges`、`notes` |
| `report.html` | plotly 交互报告（D-23）：上面净值、下面回撤（`value / 历史最高 - 1`），共用时间轴，`in_sample_range` 涂灰，底部印 `notes`。没有基准曲线（D-08）。plotly.js 从 CDN 加载，所以每份报告只有几 KB，但离线打不开图 |
| `fingerprint.json` | 本次读到的数据的指纹（D-27） |

**`data_fingerprint` 不是 `BacktestConfig` 的字段。** 它是「这次跑的时候读到了什么数据」的记录，重建时由加载器取走
（见下一节），直接 `CrossSectionBacktestConfig(**config)` 会因为这个键 `TypeError`。

**数据指纹（D-27）。** 每个键一份 `{"algorithm": "sha256", "digest", "variables", "start", "end", "n_timestamps", "n_symbols"}`：

- `price_dataset`：价格数据集在回测窗口内的成交价与估值价两列；
- `factor[{i}]:{因子类名}`：第 i 个因子背后的数据集，时间范围**含预热**。KunQuant 因子覆盖 `data_columns`，
  Polars 因子消费整个 lazyframe，所以覆盖数据集的全部数据变量。

digest 在排序后的 timestamp、symbol 和各变量的 float64 值上计算，NaN 统一成一个比特模式、`-0.0` 统一成 `0.0`，
所以同一份数据读两次得到同一个 digest。~~标签背后的数据**不在**指纹里。~~

> **更正（2026-09-15，代码审查 WR-05）：上面删除线那句已作废，保留原文作记录。** 指纹以前没覆盖预测和训练真正读过的数据：
> read 策略的特征来自因子库，却只给原始数据集算了指纹；train 模式训练段的因子、标签数据完全不在指纹里。
> 因子库被重算、或训练窗口里的复权价被回溯改写后，重建出来的结果不同，却没有任何警告。现在另有这些键：
>
> | 键 | 什么时候有 | 覆盖 |
> |---|---|---|
> | `factor_store[{i}]:{因子类名}` | `factor_data_strategy="read"` | 因子库在「预热 + 窗口」上读出的全部因子变量（预测真正用的数据） |
> | `train_factor[{i}]:{因子类名}` / `train_label[{i}]:{标签类名}` | `model_mode="train"`，对应策略是 `cal` | `collect()` 读过的训练范围上，因子 / 标签背后数据集消费的列 |
> | `train_factor_store[{i}]:…` / `train_label_store[{i}]:…` | `model_mode="train"`，对应策略是 `read` | 训练范围上因子库 / 标签库读出的全部变量 |
>
> load 模式的标签数据仍不在指纹里：load 模式根本不读标签。

**wandb（D-28）。** `use_wandb` 默认 `False`，关着时回测层不发任何数据出本机。设成 `True` 时，另开一个
project 为 `{类名}_backtest`、run 名为运行目录名的 wandb run，config 是 `get_config()`，summary 收三块指标里有限的数值
（键形如 `whole/Sharpe Ratio`、`whole/turnover/mean_per_rebalance`），再记一份 `report.html`，然后 finish。
注意这个开关只管回测层：`model_mode="train"` 时模型层的 `train()` 仍会无条件 `wandb.init`，
离线跑要设 `WANDB_MODE=disabled`（见 `example/model.md`「关于 W&B」）。

---

## 从 config 重建与复现

**一份回测的 `config.json` 能重建整个回测器并重跑出同一条曲线（D-25）。**

```python
import json

from quantlab.utils.module import load_backtester_from_config

config = json.load(open("<运行目录>/config.json"))
backtester = load_backtester_from_config(config)
result = backtester.run()          # CV 运行目录的 config.json 就调 backtester.run_cv()
```

`load_backtester_from_config(config)` 依次：

1. 深拷贝输入（调用方的 dict 原样不动）；
2. 取走 `data_fingerprint`（以及 train 模式运行留下的记录 `trained_checkpoint`，见下文）；
3. 解析 `config["name"]`，**不是 `BaseBacktester` 子类就 `TypeError`**，此时还没有构造任何数据集或模型；
4. **配置类的每个字段都必须在 `config` 里（2026-09-15，代码审查 WR-06）**，缺任何一个就 `ValueError`，写出缺了哪些，
   同样先于构造任何数据集或模型。以前缺的字段会被 `**config` 悄悄填成**当前**的 dataclass 默认值：
   默认值将来一改（比如 `fees`），旧的或手改过的 `config.json` 就会重建成另一个回测，不报任何错；
5. 用 `load_dataset_from_config` 重建 `price_dataset`，用 `load_model_from_config` 重建模型（因子、标签递归重建），
   `benchmark_dataset` 不为 `None` 时一样重建；
6. 用该类声明的 `config_cls` 构造配置和回测器（D-26：因子、数据集、模型、回测器都各自声明 `config_cls`，
   Polars 因子因此不会再被建成 `FactorConfig`）；
7. 把取走的记录设成回测器的 `expected_fingerprint`。

`tests/test_backtest_rebuild.py` 在 load、train、`run_cv` 三种情况下都锁住了：原始运行与重建运行的 `weights.zarr`
完全相同、净值 `value` 逐位相等。

**数据变了会警告，但照跑（D-27）。** 重建后的 `run()` 重新记录指纹并与 `expected_fingerprint` 比对。某个键只出现在一侧，
或 `digest` / `start` / `end` / `n_timestamps` / `n_symbols` 任一不同，就对该键 `logger.warning`，
例如 `data fingerprint mismatch for 'price_dataset' (differing fields: digest): ...`，然后继续回测。
数据集会被追加，Tiingo 也会在新分红之后回溯重算复权价，所以「重跑出不同结果」必须能被察觉；
但变了的数据仍然可以回测，所以不中断。

**因子类、标签类、模型类必须住在可 import 的模块里，不能定义在 `__main__` 脚本里。** `config.json` 用
`类.__module__ + "." + 类.__qualname__` 记类，脚本里定义的类记成 `__main__.X`。重建时 `importlib` 在**新进程**里
导入的 `__main__` 是那个新进程自己的脚本，找不到这个类。下面「最小可运行例子」为了单文件可读，把两个 Polars 类写在了脚本里，
它的运行目录因此**不能**在另一个进程里重建。实测（2026-09-15，在另一个脚本里读那次运行的 `config.json`）：

```
backtester name: quantlab.backtest.us_equity.USEquityCrossectionSelectStockVectorBt
factor name: __main__.PastReturn
has data_fingerprint: True ['factor[0]:PastReturn', 'price_dataset']
AttributeError: module '__main__' has no attribute 'PastReturn'
```

要能重建，把因子类放进 `quantlab/factor/xxx.py` 这样的模块，从那里 import。

**train 模式重建会再训练一个模型。** ~~模型层的项目目录名只精确到秒（`{类名}_trial_{YYYYmmdd_HHMMSS}`），
而 `_save_model` 遇到已存在的目录会 `RuntimeError`。同一个 `model_save_dir` 下，原始运行和重建运行如果落在同一秒内，
第二次训练会撞名失败；测试里为此等了 1.1 秒。load 模式和 `run_cv` 不训练，没有这个问题。~~

> **更正（2026-09-15，代码审查 WR-04）：上面删除线部分已作废，保留原文作记录。**
> - 项目目录名现在是 `{类名}_trial_{YYYYmmdd_HHMMSS_ffffff}`，目录已存在时再追加 `_1`、`_2`……，所以连续两次训练永不撞名，测试里的等待已删除。
> - `BaseModel.train()` 返回它写出的 checkpoint 绝对路径。train 模式的 `run()` 把它记进 `config.json` 与 `metrics.json` 的
>   顶层 `trained_checkpoint`（和 `data_fingerprint` 一样是记录，不是配置字段，重建时加载器取走它）。
> - 重建一个 train 模式的 `config.json` 仍然会**再训练**；要精确回放当初那个模型（torch / GPU 训练不能逐位复现），
>   把 `model_mode` 改成 `"load"`、`checkpoint` 设成记录里的 `trained_checkpoint` 再重建。

**信任边界。** `config.json` 会指定要 import 的类和要读的路径，`.joblib` checkpoint 本质是 pickle。只加载自己产出、自己信任的运行目录。

---

## 最小可运行例子

合成美股日线（160 个工作日 × 12 只股票，Tiingo EOD 形状）、一个 Polars 5 日动量因子、一个 5 日远期收益标签、
20 轮的 `XGBoostRegressor`，`model_mode="train"` 现场训练，然后跑一次 `long_short` 的截面 TopN 回测。
全程离线，不需要凭证、不需要真实行情、不需要 W&B 账号。所有配置都**直接构造**
（`DatasetConfig`、`PolarsFactorConfig`、`MLConfig`、`CrossSectionBacktestConfig`），不经 `quantlab/config` 里的工厂函数（D-32）。

回测窗口故意从 `train_end` 的下一个 bar 开始，好让你看到样本内判定：标签期限是 5 个 bar，所以窗口开头 5 个 bar 算样本内。

脚本放在仓库外的临时目录运行，没有提交。

```python
"""最小可跑示例：合成美股日线 + Polars 因子/标签 + XGBoostRegressor + 截面 TopN 回测。

全程离线：价格库是现写的合成 Zarr，W&B 关闭，不需要任何凭证或真实行情。
运行方式（在仓库外的任意目录）：
    OMP_NUM_THREADS=1 WANDB_MODE=disabled uv run --project <仓库根目录> python backtest_example.py
"""

import os

# 必须在 import quantlab 之前：模型层 import torch，本例又训练 xgboost，
# macOS 上同一进程混用两者要单线程 OpenMP（见 example/model.md）。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["WANDB_MODE"] = "disabled"

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import (
    CrossSectionBacktestConfig,
    DatasetConfig,
    MLConfig,
    PolarsFactorConfig,
)
from quantlab.base.factor import FactorPolars
from quantlab.dataset.stock import StockDataset
from quantlab.ml_model.xgb import XGBoostRegressor

root = Path(tempfile.mkdtemp(prefix="bt_example_"))

# ---------------------------------------------------------------- 1. 合成价格库
# 160 个工作日 x 12 只股票，Tiingo EOD 形状（复权列 adj* + 不复权列）。
timestamps = pd.bdate_range("2024-01-01", periods=160)
symbols = [f"S{i:02d}" for i in range(12)]
rng = np.random.default_rng(0)
close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (160, 12)), axis=0))
open_ = np.vstack([close[:1], close[:-1]]) * np.exp(rng.normal(0.0, 0.01, (160, 12)))
adjusted = {
    "adjOpen": open_,
    "adjHigh": np.maximum(open_, close) * 1.01,
    "adjLow": np.minimum(open_, close) * 0.99,
    "adjClose": close,
    "adjVolume": rng.uniform(1e5, 1e6, (160, 12)),
}
variables = {**adjusted, **{k[3:].lower(): v * 1.7 for k, v in adjusted.items()}}
store = root / "stock" / "stock.zarr"
xr.Dataset(
    {name: (("timestamp", "symbol"), values) for name, values in variables.items()},
    coords={"timestamp": timestamps, "symbol": symbols},
).to_zarr(store, mode="w")

# 先写库、再建配置：数据集对象构造时就会读库。
price_config = DatasetConfig(
    zarr_file_path=str(store),
    raw_data_dir_path=str(root / "stock" / "raw"),
    catalog_path=str(root / "stock" / "catalog"),
    market="us_equity",
    frequency="1d",
)


def dataset() -> StockDataset:
    """每个使用方一个独立的数据集对象（配置也复制一份），互不改写日期。"""
    return StockDataset(dataclasses.replace(price_config))


# ---------------------------------------------------------------- 2. Polars 因子与标签
class PastReturn(FactorPolars):
    """past_ret_n = adjClose / n 个 bar 前的 adjClose - 1。"""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.config.kwargs["n"]
        close = pl.col("adjClose")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((close / close.shift(n).over("symbol") - 1.0).alias(f"past_ret_{n}"))
            .select(["timestamp", "symbol", f"past_ret_{n}"])
        )

    def _get_features(self, data):
        return data

    def _get_labels(self, data):
        raise RuntimeError("PastReturn is a feature")


class ForwardReturn(FactorPolars):
    """fwd_ret_n = n 个 bar 后的 adjClose / adjClose - 1。"""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.config.kwargs["n_forward_periods"]
        close = pl.col("adjClose")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((close.shift(-n).over("symbol") / close - 1.0).alias(f"fwd_ret_{n}"))
            .select(["timestamp", "symbol", f"fwd_ret_{n}"])
        )

    def _get_labels(self, data):
        return data

    def _get_features(self, data):
        raise RuntimeError("ForwardReturn is a label")


factor = PastReturn(PolarsFactorConfig(window=5, dataset=dataset(), kwargs={"n": 5}))
# 回测器从 kwargs["n_forward_periods"] 读标签期限（D-17）。
label = ForwardReturn(
    PolarsFactorConfig(window=0, dataset=dataset(), kwargs={"n_forward_periods": 5})
)

# ---------------------------------------------------------------- 3. 模型（直接构造 MLConfig）
model = XGBoostRegressor(
    MLConfig(
        factors=[factor],
        labels=[label],
        model_save_dir=str(root / "models"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date="2024-01-01", end_date="2024-08-09",
        train_start="2024-01-01", train_end="2024-05-17",
        test_start="2024-05-24", test_end="2024-08-09",
        hyperparameters={"num_boost_round": 20, "max_depth": 2, "nthread": 1},
    )
)

# ---------------------------------------------------------------- 4. 回测（直接构造 CrossSectionBacktestConfig）
config = CrossSectionBacktestConfig(
    price_dataset=dataset(),
    model=model,
    model_mode="train",              # 用模型自己的 train/test 日期训练（D-13）
    start_date="2024-05-20",         # 故意从 train_end 的下一个 bar 开始，演示样本内判定
    end_date="2024-08-09",
    output_dir=str(root / "backtests"),
    rebalance_periods=5,             # 每 5 个 bar 调仓一次
    direction="long_short",          # 多 3 只各 +1/6，空 3 只各 -1/6
    top_n=3,
)
result = USEquityCrossectionSelectStockVectorBt(config).run()

# ---------------------------------------------------------------- 5. 看结果
metrics = result.metrics
print("run dir:", result.run_dir.name)
print("files:", sorted(p.name for p in result.run_dir.iterdir()))
print("training_window:", metrics["training_window"])
print("in_sample_range:", metrics["in_sample_range"])
print("out_of_sample_ranges:", metrics["out_of_sample_ranges"])
print("Total Return [%]:", round(metrics["whole"]["Total Return [%]"], 4))
print("Sharpe Ratio:", round(metrics["whole"]["Sharpe Ratio"], 4))
print("turnover mean per rebalance:", round(metrics["whole"]["turnover"]["mean_per_rebalance"], 4))
print("out-of-sample Total Return [%]:", round(metrics["out_of_sample"]["Total Return [%]"], 4))
print("orders:", result.simulation.orders.sizes["order"], "| liquidations:", result.simulation.liquidations)
first = result.weights["weight"].isel(timestamp=0)
print("first rebalance", pd.Timestamp(first.timestamp.values).date(), "weights:",
      {s: round(float(w), 4) for s, w in zip(first.symbol.values, first.values) if w != 0})
print("notes:", metrics["notes"])
```

### 真实输出

stdout，原样。命令行里的仓库路径缩写成了 `<仓库根目录>`；脚本只打印运行目录的**名字**，不打印临时目录的完整路径
（它在 `tempfile.mkdtemp` 建的 `bt_example_*` 下面），所以输出本身没有做替换。运行目录名里的时间戳是本机当地时间。

```
$ OMP_NUM_THREADS=1 WANDB_MODE=disabled uv run --project <仓库根目录> python backtest_example.py
run dir: USEquityCrossectionSelectStockVectorBt_20260915_032506_974380
files: ['config.json', 'equity.zarr', 'fingerprint.json', 'liquidations.json', 'metrics.json', 'report.html', 'weights.zarr']
training_window: ('2024-01-01', '2024-05-24')
in_sample_range: ('2024-05-20', '2024-05-24')
out_of_sample_ranges: [('2024-05-27', '2024-08-09')]
Total Return [%]: 3.2102
Sharpe Ratio: 1.0364
turnover mean per rebalance: 1.2695
out-of-sample Total Return [%]: -0.0442
orders: 105 | liquidations: []
first rebalance 2024-05-20 weights: {'S04': 0.1667, 'S05': -0.1667, 'S07': 0.1667, 'S09': -0.1667, 'S10': -0.1667, 'S11': 0.1667}
notes: ['No borrow or short-financing cost is modelled, so short-side returns are optimistic.']
```

stderr 共 13 行：3 条 zarr 的 `ZarrUserWarning: Consolidated metadata is currently not part in the Zarr format 3 specification`
（各占两行）、6 行因子计算计时的 loguru INFO，以及下面这条 D-17 的样本内 warning（原样，只截掉了行首的时间与代码位置）：

```
WARNING  | USEquityCrossectionSelectStockVectorBt: backtest window 2024-05-20..2024-08-09 overlaps the model's effective training window 2024-01-01..2024-05-24 (train_start..train_end + label horizon, D-17); bars 2024-05-20..2024-05-24 are in-sample. Continuing: in-sample and out-of-sample results are reported separately
```

怎么读：

- `training_window` 的终点是 `2024-05-24`：`train_end` 是周五 `2024-05-17`，加 5 个 bar 是下一个周五，不是日历上的 5 天后。
- 样本外那段的 `Total Return [%]` 是 -0.04%，整段是 +3.21%：整段里包含了 5 个样本内 bar。价格是纯随机游走，
  这些数字**没有任何策略含义**，例子只证明管线接通了，以及样本内外确实分开报了。
- `turnover mean per rebalance` 约 1.27：`long_short` 每次调仓大半个组合换手，单边口径下完全换一遍是 2。
- 首个调仓日的权重是 3 个 `+1/6`、3 个 `-1/6`，毛敞口 1、净敞口 0（D-09）。
- 合成数据没有退市，所以 `liquidations` 是空的。
- 这个运行目录不能在另一个进程里重建，原因是两个 Polars 类定义在 `__main__`，见上一节。

---

## 真实数据用法（此例未实际运行）

> **此例未实际运行。** 本机 `data/data/us_equity/1d/us_all.zarr` 只有约 21 个 bar（`2026-08-07`..`2026-09-04`，
> 7700 个标的），远不够一个真实因子集的预热（Alpha101 这类窗口动辄上百个 bar），训练段和回测窗口也放不下，
> 所以没有跑过。下面只说明形状：价格数据集指向真实库，因子与标签的 `data_columns` 与各自类的 `Input(...)` 一致。

```python
from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, DatasetConfig, FactorConfig, MLConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.label.fret import Return
from quantlab.ml_model.xgb import XGBoostRegressor


def us_equity() -> StockDataset:
    return StockDataset(
        DatasetConfig(
            zarr_file_path="data/data/us_equity/1d/us_all.zarr",
            raw_data_dir_path="data/downloads/<厂商原始分片目录>",
            catalog_path="data/catalog",
            market="us_equity",
            frequency="1d",
        )
    )


ADJUSTED = ("adjHigh", "adjLow", "adjClose", "adjOpen", "adjVolume")   # Alpha101Stock 的图输入是 adj* 列
alpha101 = Alpha101Stock(FactorConfig(window=250, dataset=us_equity(), mode="batch", data_columns=ADJUSTED))
# Return 读哪一列以 quantlab/label/fret.py 里的 Input(...) 为准，data_columns 要与之一致
label = Return(FactorConfig(window=0, dataset=us_equity(), mode="batch", data_columns=("adjClose",),
                            kwargs={"n_forward_periods": 5}))

model = XGBoostRegressor(MLConfig(
    factors=[alpha101], labels=[label], model_save_dir="./model_ckpt",
    factor_data_strategy="cal", label_data_strategy="cal",
    start_date="2020-01-01", end_date="2026-09-04",
    train_start="2020-01-01", train_end="2024-12-31",
    test_start="2025-01-10", test_end="2026-09-04",
    early_stopping=True, early_stopping_patience=50,
    hyperparameters={"num_boost_round": 1000},
))

result = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
    price_dataset=us_equity(), model=model, model_mode="train",
    start_date="2025-01-10", end_date="2026-09-04", output_dir="./backtests",
    rebalance_periods=5, direction="long_short", top_n=50,
)).run()
```

`raw_data_dir_path` / `catalog_path` 是 `DatasetConfig` 的必填字段，回测不读它们，按你本机的数据布局填。
训练加回测的进程同时有 torch 和 xgboost，macOS 上照样要 `OMP_NUM_THREADS=1`。

---

## benchmark

`BacktestConfig.benchmark_dataset` 这个槽位**保留着**（D-08），设计上是一个同频率的单标的数据集（可以是指数），
基类也留着 `_simulate_benchmark` 钩子和 `metrics["benchmark"]` 的位置。但**基准对比本阶段没有实现**：
本机没有直接下载的指数价格数据，而用成分股拼一个「指数」不是用户要的基准。

所以现在它只能是 `None`：给一个非 `None` 的值，`config` setter 在构造期就抛 `NotImplementedError`，
消息里写着 D-08 和「leave it None」。`None` 时报告和指标里没有任何基准曲线或基准指标
（vectorbt `stats()` 默认带的 `Benchmark Return [%]` 也被刻意去掉了，它是持有全部交易标的的收益，不是基准）。

---

## 常见坑

**1. 改了因子日期，读回来的还是旧窗口。** `XrBackend.read` 已经持有数据时直接返回，过滤又是就地收窄，
所以改 `config.start_date` 之后普通的 `read()` 拿回的是上一次更窄的缓存：不报错，只是缺 bar，预热悄悄变短，
或首个调仓日整行没有预测。回测器内部一律 `read(overwrite=True)`；你自己在回测器外改日期取数时也要这样做（Pitfall 1）。

**2. 一只退市股冻结整个组合。** vectorbt 里持仓标的价格为 NaN 时，**整组**之后的调仓全部被悄悄跳过，
净值变成一条买入持有的直线，订单数在中途停止增长。回测器已经 ffill 两列价格并强制平仓；如果你自己写引擎，
这是第一个要处理的问题（Pitfall 2）。

**3. 调仓行上的 NaN 权重不是「不持有」，是「保持原仓位」。** 它还会占着资金挡住同一行的其余订单。
自己写 `_generate_signals` 时，调仓行上不持有必须写 `0.0`，NaN 只能出现在整行都不调仓的 bar 上（Pitfall 3）。

**4. `Portfolio` 不能按时间切。** `pf.loc[a:b]` 直接 `IndexingError`。想要某一段的指标，切**收益序列**再算，
不要切组合，更不要把那一段重新模拟一遍：重新模拟会重置资金，得到的是另一条路径（Pitfall 5）。

**5. 真实数据上的折日期是纳秒字符串。** `train_cv` 用 `np.datetime_as_string` 生成折日期，
真实库上是 `'2026-08-07T00:00:00.000000000'`；`datetime.date.fromisoformat` 不认它，`pd.Timestamp(np.str_)`
直接 `TypeError`。回测器统一经 `pd.Timestamp(str(x))` 规范成 ISO 日期；你自己处理 `cv_folds.json` 里的日期时也要先 `str()`（Pitfall 10）。

**6. 最后一个 bar 不调仓。** 它的信号在窗口内没有 t+1 成交 bar，所以最后一个 bar 恒为非调仓。
回测窗口只有 `rebalance_periods` 个 bar 时，实际只会调仓一次（Pitfall 13）。

**7. `run_cv` 按「天」判断折是否相接。** 折日期先被规范成 ISO 日期，相接判定也按日历上的整天映射到 bar。
日内数据上，前一折结束和后一折开始落在同一天时（比如 12:00 结束、13:00 开始），会被当成**重叠**拒绝，
即便它们在 bar 上其实首尾相接。日内频率的 CV 回测目前只能让折边界落在不同的日子上。

**8. 因子类写在脚本里就重建不了。** 见「从 config 重建与复现」：`__main__.X` 在新进程里找不到。

**9. `RNNClassifier` 的分数是概率。** 它的预测变量是 P(上涨)，取值 [0, 1]。按它排序选股没问题，
但不要把它当收益幅度读，也不要拿它和回归头的分数混着比较。

~~**10. train 模式的两次运行落在同一秒会撞名。** 模型层项目目录只精确到秒，第二次训练 `RuntimeError`。
快速连续跑两次 train 模式回测（比如原始运行后立刻重建重跑）时，换一个 `model_save_dir` 或者隔一秒。~~
（2026-09-15 代码审查 WR-04 已修复，保留原文作记录：项目目录名带微秒并在已存在时追加序号，不再撞名；见「从 config 重建与复现」。）

---

## 已知的不完整之处

下面这些都是本阶段**明确推迟**的（`03.7-CONTEXT.md` 的 Deferred Ideas），记在这里是为了不让人以为它们存在，不是承诺。

**1. 基准对比。** 等有了直接下载的指数价格数据再做；槽位和钩子已留好，见上面「benchmark」。

**2. 融券费与空头融资成本。** 不模拟（D-21），`long_short` 的空头一侧收益偏乐观。

**3. 分批调仓（staggered rebalancing）。** 现在锚点固定在回测窗口第一个 bar，整本书在同一天换，
结果对「哪天开始回测」敏感。把书拆成 `rebalance_periods` 份错开调仓可以降低这种日期运气，没有实现。

**4. 等权 TopN 之外的组合构造。** 没有优化器、没有风险模型、没有行业或市值中性化。Phase 5 的组合优化器会产出同一个
权重契约（D-03），届时可以替换 `CrossSectionTopNSelector` 而不改回测器。

**5. 事件驱动引擎与时序兄弟类。** `USEquityCrossectionEventDrivenBt`（NautilusTrader，Phase 6 BT-02）和
`USEquityTimeseriesVectorBt` 只有预留的名字。

**6. 指数成分标的池。** 标的池是价格库里的全部标的（D-06），没有时点成分过滤，因此也有幸存者偏差方面的顾虑，
见 `constituent.md`。

**7. 拼接报告不画样本内阴影、日内折按天判定。** 分别见「run_cv()」和「常见坑」第 7 条。

~~**8. 标签数据不在指纹里。** 指纹覆盖价格数据集和每个因子的数据集；train 模式下训练用的标签数据变了，重建时不会有警告。~~
（2026-09-15 代码审查 WR-05 已修复，保留原文作记录：train 模式记录训练段的因子与标签数据指纹，read 策略记录因子库指纹；见「输出目录」一节的数据指纹更正。）
