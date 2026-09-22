# 股票池过滤（UniverseFilteredFactor）

> 代码位置：`quantlab/factor/universe_filter.py`
> 配置重建：`quantlab/utils/module.py:load_factor_from_config`（`from_config` 协议）
> 测试：`tests/test_universe_filtered_factor.py`、`tests/universe_fixtures.py`
> 相关：[factor.md](factor.md)（因子层）、[backtest.md](backtest.md)（回测层）、[constituent.md](constituent.md)（指数成分）

本文的代码例子**未实际运行**（按 [README.md](README.md) 的约定标注）；文中所有**数字**都是真实测量结果，测量脚本与口径写在对应小节里。

---

## 一句话

`UniverseFilteredFactor` 把「一个标的在 t 时刻值不值得交易」这个判断，用真实交易员会用的点时点规则，做成一个**因子包装类**——包住任意 KunQuant 因子或标签，它自己还是一个因子，所以模型层和回测层一行都不用改。

---

## 它解决的那个事故

美股 Alpha101 → 5 日 Return → XGBoostRegressor → `USEquityCrossectionSelectStockVectorBt` 这条链，在约 7,700 个 Tiingo `us_all` 标的上跑出过 **244 天 +886,077,331%**，而且 `best_iteration=0`（模型根本没学到东西）。

原因不是模型，是**股票池**：认股权证、权利、单位、交易所测试代码和低价股。`ZWZZT`（交易所测试代码）从 0.007 涨到 10.05，57 次强制平仓里约 50 次是这类标的。截面排序里只要混进一个从 0.007 变成 10.05 的东西，它就会稳定地排在第一位，然后被买进去。

**这里面「证券类型」那一半，2026-09-21 起不再由本类负责。** 本类现在只做价格与流动性两条阈值；认股权证 / 权利 / 单位 / 测试代码由数据采集侧的 CRSP `security_filter`（`equity_common`）按日期排除。为什么是这样切分，见下文「为什么是删而不是关」。

---

## 怎么用

**因子和标签都要包。** 只包因子，标签仍然带着出池标的的行；只包标签，截面算子继续被垃圾标的污染。

```python
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.universe_filter import UniverseFilteredFactor
from quantlab.label.fret import Return

model = XGBoostRegressor(
    MLConfig(
        factors=[UniverseFilteredFactor(Alpha101Stock(factor_config))],
        labels=[UniverseFilteredFactor(Return(label_config))],
        ...
    )
)

backtester = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        # 价格数据集保持**未过滤**：已持仓标的的价格必须一直拿得到，
        # 否则 vectorbt 会冻结整组调仓（D-07）。
        price_dataset=make_stock_dataset(dataset_config),
        model=model,
        ...
    )
)
```

构造参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `factor` | 必填 | 被包的 KunQuant 因子或标签 |
| `min_price` | `5.0` | **原始**收盘价下限 |
| `min_dollar_volume` | `1_000_000.0` | 滚动成交额（原始 `close*volume`）均值下限 |
| `window` | `20` | 成交额均值的窗口，单位是 bar |

只有这三个阈值参数。原先还有第四个 `exclude_non_common`（静态普通股代码规则的开关），**2026-09-21 连同规则本体一起删除**，理由见下文「为什么是删而不是关」。旧配置里带着这个键的，`from_config` 会按 D-25/WR-06 点名报 `ValueError`，不会悄悄重建成另一个股票池。

---

## 语义（LS-1 / LS-2 / LS-3）

### LS-1 掩码是点时点的

标的在 t 时刻在池内，当且仅当两条同时成立：

1. **原始** `close[t] >= min_price`；
2. 截至 t 的 `window` 根 **原始** `close*volume` 均值 `>= min_dollar_volume`。

（原先是三条，第一条「代码是普通股」的静态规则 2026-09-21 删除；证券类型的判定移到数据采集侧的 CRSP `security_filter`。见下文「为什么是删而不是关」。）

**为什么必须用原始价、不能用复权价。** 复权历史会被后来的拆股和分红**压低**。一个今天 3 块钱的仙股，如果中间做过 1:10 的反向拆股，它 2015 年的复权价可能是 30 块——拿复权价去问「它当年是不是仙股」，答案会系统性地偏向「不是」。判断当时能不能交易，只能用当时屏幕上的那个价格。

**窗口不满或窗口里有缺失，一律算出池。** 实现上是 `rolling(window, min_periods=window)`，`min_periods` 数的是**有效**观测数，所以窗口里任何一个 NaN 都让结果变成 NaN。「不知道它当时有多活跃」必须读作出池，不能读作在池。

**不使用 `anomaly_flag`。** 掩码在池内是 `1.0`、出池是 `NaN`，铺在 `(timestamp, symbol)` 上。

t 之后的任何一根 bar 都不会改变 t 时刻的掩码——这条由 `test_mask_is_point_in_time` 锁住（把 t 之后的数据整体乘以 1e-3 或 1e3，`mask[:t+1]` 逐位不变）。一个会偷看未来的股票池，会精准地选出后来涨得好的那些名字。

### LS-2 截面算子只看池内，时序算子看完整历史

见下文「截面算子改写」。

### LS-3 输出掩码打在标签**自己**的时间戳上

因子输出和标签算完之后，在掩码为 NaN 的位置置 NaN。标签来自**未过滤**的价格，并且**只看标签自己那个时间戳 t 的掩码**，绝不看 t+h 的池状态。

顺序是 load-bearing 的：`Return._get_labels` 做的是 `shift(timestamp=-n)`，**先位移再打掩码**，掩码打的就是 t。反过来会用 t+n 的池状态决定 t 的标签，那是一个前视错误——而且是会安静地删掉「每一次成功离场」的那种前视错误。

---

## 标的轴规则：永不删列

**没有任何规则会让标的离开标的轴。** 出池只表现为 NaN 格子；整窗 NaN 的标的**保留**为 NaN 列。任意窗口、任意标的集合下，`_mask_panel` 输出的 symbol 轴与输入**逐元素相等**。

这条规则演进过两次：最初设计是「整窗 NaN 就删列」；2026-09-15 用户决定收窄成「只删被静态代码规则排除的标的」；2026-09-21 条件 (a) 整体删除之后，连那唯一一个删列出口也消失了。

**为什么这是件好事。** 按「整窗 NaN 删列」，标的轴会**依赖日期窗口**：同一个标的在窗口 A 里在、窗口 B 里没了。而 DL 头按标的**位置**编码输入（`MLPRegressor` 把每根 bar 展平成 `[S*F]`），`DLModel._align_prediction_symbols` 在面板缺少训练过的标的时会直接 `ValueError`。于是「用 2020 年训练、在 2024 年预测」这件最普通的事会随机报错。

2026-09-15 那版靠的是一个**论证**：代码规则与日期无关，所以它删掉的标的在任何窗口里都不存在，也就不可能被训练过，删它永远安全。现在连论证都不需要了——**「符号轴与日期窗口无关」变成平凡成立**，因为根本没有删列这个动作。`test_mask_panel_never_drops_a_column` 是这条的锁（三种窗口 × ticker 轴/int64 PERMNO 轴）。

**保留下来的 NaN 列对两类头分别意味着什么：**

- **ML 头（树模型）：中性。** `XGBoostRegressor._to_rows` 只保留标签全部有限的行（`quantlab/ml_model/xgb.py:225`），NaN 列自然被丢掉。
- **DL 头的训练：不中性，且这是已接受的代价，本次不修。** `DLModel._fit` 会对**标签**张量也调用 `_preprocess`（`quantlab/base/model.py:1069`），而 `MLPRegressor._preprocess` 是 `torch.nan_to_num(nan=0.0)`（`quantlab/dl_model/mlp.py:185`）。于是每一个被掩掉的格子——包括整列保留下来的 NaN 列——都变成一个「特征为 0、目标为 0」的样本被网络训练，把损失函数带偏。

  必须说清楚的是：**这是 `DLModel` 对任何 NaN 的既有行为**，上市前的空档和退市后的尾巴早就在这么干了，股票池包装只是让这类格子变多。修它要动模型层，而本次任务的边界是不碰模型层。

- **预测与选股不受影响，两类头都一样。** 全 NaN 特征让 `predict_panel` 返回 NaN（`quantlab/base/model.py:567`），NaN 分数在选股器里不可选（D-12）。所以这些标的永远不会被买进去。

**结论：DL 头可以跨窗口工作，这不是一条限制。** `test_dl_head_predicts_across_windows_when_a_trained_symbol_leaves_the_universe` 就是这条的锁：`MLPRegressor` 在窗口 A 训练，在一个「某个训练过的标的全程出池」的窗口 B 上 `predict_panel` 不抛任何异常，该标的的预测全是 NaN。

---

## 截面算子改写

**问题**：出池标的不能只在输出上被抹掉。`Rank`、`CrossSectionalZScore` 这类**截面**算子在计算时会把当期所有标的排在一起——一个从 0.007 涨到 10.05 的测试代码，会改变**每一个**在池标的的排名值。事后把它那一列置 NaN，救不回已经被污染的其余列。

**做法**：把掩码作为一个额外的图 `Input`，然后对图里每一个 `CrossSectionalOp` 的每一个输入 `v`，替换成 `Div(v, universe_mask)`。

除以 `1.0` 不改变数值，除以 `NaN` 得到 `NaN`——于是出池标的在截面算子眼里**根本不存在**，而时序算子拿到的仍是完整历史。

原型实测（Alpha101 全量 101 个因子，把「永远不在池内」的标的数据整体放大 1000 倍）：

| | 受扰动而改变的因子数 |
|---|---|
| 未改写的图 | **37** |
| 改写后的图 | **0** |

改写给在池格子带来的额外 NaN：平均 **0.16%**，最大 **3.9%**（`alpha071`）。

两个 KunQuant 硬约束：

- **批量一律 `start=0`**。KunQuant 0.1.11 的 `CrossSectionDataHolder` 在 `num_time` 被赋值之前就用它算 `base_time`，`start>0` 时**所有** `GenericCrossSectionalOp` 的结果都是错的且不确定。
- **标的数必须与 SIMD 块宽对齐**（本机 aarch64 上实测 16 可以、13 不行）。

图里没有任何截面算子时，改写是一个 no-op，并且**不会**声明掩码输入——KunQuant 会把没有被任何 `Output` 消费的输入剪掉，此时还去 `queryBufferHandle` 会抛 `RuntimeError: Cannot find the buffer name`。

---

## 回测里掉出股票池的持仓，什么时候卖出（LS-4）

回测器**没有改**，`price_dataset` 保持未过滤。掉出池的持仓走的是这条链：

全 NaN 特征 → `predict_panel` 给 NaN 预测 → 下一个**调仓 bar** 不可选 → 在**再下一根 bar 的开盘**卖出（D-05 的 t+1 成交）。

所以：

| `rebalance_periods` | 最晚多久卖掉 |
|---|---|
| `5` | 最多晚 **4** 根 bar |
| `1` | 次日 |

这是刻意的：资格只在调仓 bar 上重新评估。想要更快，就把 `rebalance_periods` 调小。

价格数据集**必须**保持未过滤——已持仓标的的价格一旦变成 NaN，vectorbt 会按最后价值继续持有，并且**整组**之后的调仓都被悄悄跳过（03.7-RESEARCH.md Pitfall 2）。

---

## 已知代价（LS-5）

**截面算子之上的时序算子**，在标的（重新）入池之后的一整个窗口内是 NaN。

例如 `correlation(rank(x), rank(y), 10)`：标的出池期间 `rank` 是 NaN，重新入池后前 10 根 bar 的滚动窗口里仍然含 NaN，所以输出是 NaN。

这与真实交易一致——你确实没有那段时间的观测——所以它是记录下来的代价，不是要绕过的 bug。`test_time_series_over_cross_sectional_is_nan_after_reentry` 锁住了这条（`WindowedAvg(Rank(close), 3)` 在重新入池后前 2 根 bar 是 NaN，第 3 根恢复）。

---

## 为什么是删而不是关（2026-09-21）

原先有一条静态、与日期无关的**票代码规则**（条件 (a)）：九条正则对大写代码做 `re.search`，判断它**看起来**是不是普通股。

| 组 | 正则 | 在 Tiingo `us_all` 里的排除数 |
|---|---|---|
| `nasdaq_fifth_letter` | `^[A-Z]{4}[WRU]$` | 2,310 |
| `six_char_warrant` | `^[A-Z]{4}WS$` | 13 |
| `delimited_suffix` | `[-.](?:WS\|WT\|W\|U\|UN\|R\|RT)(?:[-.]\|$)` | 1,056 |
| `when_issued_or_called` | `-(?:WD\|WI\|CL)$` | 71 |
| `test_symbol_zzzt` | `^Z[A-Z]ZZT$` | 7 |
| `test_symbol_xtest` | `^[A-Z]TEST(?:-\|$)` | 72 |
| `test_symbol_zxyz` | `^ZXYZ(?:-\|$)` | 1 |
| `preferred_share` | （引用采集层常量） | 0 |
| `baby_bond` | （同上） | 0 |

并集 3,519 / 14,481（24.3%），测量口径是 2026-09-15 `data/data/reference/universe.parquet` 里 `category == "us_all"` 的 14,481 个不同代码。**整条规则连同这段测量注释一起删除了。**

### (i) 机制论证：它不是被取代，它是失效

**九条正则全部要求字母。** `^[A-Z]{4}[WRU]$` 要四个大写字母加一个后缀字母；`^Z[A-Z]ZZT$` 要 `ZZT` 这三个字面字符；`-(?:WD|WI|CL)$` 要一个连字符加两个字母。**对 `"10107"` 这样的 PERMNO 数字串，九条一条都不匹配。**

于是那条静态判定**恒返回 True**，整条过滤在 CRSP 面板上**无声地变成 no-op**——而它**看起来**在工作：开关默认 `True`，`to_dict()` 照样把它序列化进每一份 run config，文档照样描述它，分支照样在每次 `compute_universe_mask` 里执行一遍。

这就是为什么「把 `exclude_non_common` 默认改成 `False`」不是一个可接受的替代：那留下的是一个**仍然存在、仍会被序列化、仍在文档里、但什么都不做**的旋钮，比删掉更难发现。

这条机制证据的分量和当初那次 14,481 个代码的测量**等量齐观**：那次测量证明了规则在 Tiingo ticker 轴上有效，这条机制证明了它在 PERMNO 轴上**不可能**有效。轴换了，规则的前提就没了。

（代码里这九条正则是**唯一**的删除对象。`quantlab/acquisition/universe.py` 的 `_PREFERRED_SHARE_PATTERN` / `_BABY_BOND_PATTERN` **本体一个字未改**——它们是 Tiingo 采集基础设施；`quantlab/enums/data.py` 的 `TRADEABLE_TICKER_PATTERN` 也一个字未改——它是路径段安全检查，不是证券类型过滤。`universe_filter.py` 删掉的只是那两行 `import`。）

### (ii) 职责的承接者：CRSP 的 `security_filter`

CRSP 的 `security_filter` 预设 `equity_common`（`quantlab/dataset/crsp/__init__.py`）不是等价替换，是**更强**的替换：

- **CRSP 的类型词表里根本不存在** warrant / right / preferred / test-code 的编码，所以九条正则里有 7 类在 CRSP 面板上**不可能存在**；
- 唯一真实对应的是 "unit"，在 CRSP 里是 `sharetype=UG`，**已被 `equity_common` 排除**；
- CRSP 还额外排除了正则**排不掉**的 ADR（`AD`）、ETF/CEF、以及未知类型行；
- 判定是**按日期**做的（一个标的的证券类型会变），而正则是静态的；
- 而且附一份**审计报告**，排除了什么、排除了多少都可核对。

原正则的论证基础，连那份文档自己都承认有窟窿：2,310 个五字母命中里有 284 个没有旁证。

**顺带结清一个悬案：** `when_issued_or_called`（`-WD` / `-WI` / `-CL`）是九条里**唯一一条从未被单独测量过**的规则——它的 71 个命中从来没有像五字母那批一样被逐个复核过。条件 (a) 整体删除之后，这个悬案不再需要一次全库 distinct 统计来结清：被删掉的规则不需要证明自己是对的。

### 红利：`_mask_panel` 从此永不删列

`_mask_panel` 里唯一一条会在标的轴上删列的分支，就是按这九条正则剔除标的那一条。它消失之后本方法成为**纯阈值**：出池只表现为 NaN 格子，标的轴原样返回。详见上文「标的轴规则：永不删列」。

---

## 限制

1. **Polars 因子会被拒绝**（`TypeError`）。Polars 因子的截面逻辑是 polars 表达式，不是 KunQuant 算子图，改写不了；而只掩输出会把出池标的留在每一个 rank/zscore 里面——正是这个类要防的事。
2. **`read` 策略的因子库必须经由包装类写出来。** 落盘的是**打掩码之前**的值，输出掩码是 `read()` 之后再施加的。用未包装的内层因子写出来的库，其**截面值已经被污染**，读的时候再怎么打掩码也救不回来。
3. **回测的数据指纹覆盖 `data_columns`，不覆盖掩码读的原始 close/volume**（D-27）。也就是说，只改了原始 close/volume 的一次数据重算，指纹不会报警。修它要动回测层，本次任务不碰。
4. **标签里的截面算子** 会在标签图自己的计算时间戳上求值。当前的 `Return` / `BinaryReturn` 里没有截面算子，所以现在不是问题。
5. **KunQuant 约束**：批量一律 `start=0`；标的数必须是 SIMD 块宽的倍数（aarch64 上 8 的倍数）。
6. **流式模式**每根 bar 的 `data` 字典里必须**额外**带上原始 `close` 与 `volume`——继承来的推送只发 `config.data_columns`，这两个键是掩码专用的。

---

## 与 `quantlab/dataset/masking.py:UniverseMask` 的区别

两者名字像，做的事正交，可以叠加使用：

| | `UniverseMask` | `UniverseFilteredFactor` |
|---|---|---|
| 回答的问题 | **某天谁是指数成分股** | **某天这只票够不够贵、够不够活** |
| 依据 | 指数成分历史（外部事实） | 原始价格、原始成交额（阈值） |
| 层次 | 数据集层 | 因子层（包装类） |
| 文档 | [constituent.md](constituent.md) | 本文 |

想「在标普 500 成分股里，只交易够贵够活的那些」，就两个一起用。
