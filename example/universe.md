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
| `exclude_non_common` | `True` | 是否启用静态普通股代码规则 |

---

## 语义（LS-1 / LS-2 / LS-3）

### LS-1 掩码是点时点的

标的在 t 时刻在池内，当且仅当三条同时成立：

1. 代码是普通股（静态规则，见下文「票代码规则」）；
2. **原始** `close[t] >= min_price`；
3. 截至 t 的 `window` 根 **原始** `close*volume` 均值 `>= min_dollar_volume`。

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

## 标的轴规则（2026-09-15 用户决定）

**只有静态代码规则会让标的离开标的轴，并且在每一个窗口里都离开。** 仅仅因为价格或成交额阈值而整窗 NaN 的标的，**保留**为 NaN 列。

这条取代了最初设计的「整窗 NaN 就删列」。

**为什么改。** 按「整窗 NaN 删列」，标的轴会**依赖日期窗口**：同一个标的在窗口 A 里在、窗口 B 里没了。而 DL 头按标的**位置**编码输入（`MLPRegressor` 把每根 bar 展平成 `[S*F]`），`DLModel._align_prediction_symbols` 在面板缺少训练过的标的时会直接 `ValueError`。于是「用 2020 年训练、在 2024 年预测」这件最普通的事会随机报错。代码规则与日期无关，它删掉的标的在任何窗口里都不存在，也就**不可能被训练过**，所以删它永远安全。

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

## 票代码规则

静态、与日期无关的一条规则，判断代码**看起来**是不是普通股。

**测量口径**：2026-09-15，`data/data/reference/universe.parquet` 里 `category == "us_all"` 的 **14,481** 个不同代码。

| 组 | 排除数 | 例子 |
|---|---|---|
| `nasdaq_fifth_letter` `^[A-Z]{4}[WRU]$` | 2,310 | AACIW、AACBR、AACBU |
| `six_char_warrant` `^[A-Z]{4}WS$` | 13 | AACTWS、ACNDWS |
| `delimited_suffix` | 1,056 | AAC-WS、AAC-U、ACP-R、ACP-R-W |
| `when_issued_or_called` | 71 | DD-WD、JNJ-WD、AED-CL |
| `test_symbol_zzzt` `^Z[A-Z]ZZT$` | 7 | ZWZZT、ZVZZT、ZXZZT |
| `test_symbol_xtest` | 72 | ATEST-*、CTEST-*、NTEST-* |
| `test_symbol_zxyz` | 1 | ZXYZ-A |
| `preferred_share` | 0 | （见下） |
| `baby_bond` | 0 | （见下） |

并集 **3,519 / 14,481（24.3%）**，剩下 10,962 个普通股。

**优先股和小额债券在 `us_all` 里是 0，规则却必须保留。** 按 A4/D-02，`us_all` 在**采集期**就已经把它们剔除了，但用 `nasdaq_all` 建的库里仍然有。这两条正则直接从 `quantlab/acquisition/universe.py` 引用（`_PREFERRED_SHARE_PATTERN` / `_BABY_BOND_PATTERN`），单一真源，不抄第二份。

### 这条规则要避开的陷阱

**带连字符的不都是优先股，五个字母的不都是权证。**

- `BRK-A`、`BRK-B`、`BF-A`、`BF-B`、`HEI-A`、`LEN-B`、`MOG-A`、`UA-C`、`MKC-V`、`CWEN-A`、`PBR-A` 是**普通股**（类别股）；
- `GOOGL`、`CMCSA`、`BATRK`、`DISCA`、`LBTYA`、`RYAAY` 是**五个字母的普通股**。

一条只看「有没有连字符」或「是不是五个字母」的规则，会把伯克希尔和康卡斯特从全市场名单里悄悄删掉。所以第五字符规则只对**恰好五个字母**生效（`ACIW`、`AAWW`、`ACHR`、`AMKR`、`ALTR` 这些 4 字母代码不受影响），连字符规则要求分隔符后面**紧跟**特定后缀。

未被任何规则命中的带分隔符代码共 45 个，尾巴只有 6 种：`-A`(21)、`-B`(15)、`-1`(3)、`-C`(3)、`-V`(2)、`-T`(1)，全是普通股或类别股。

### 两个证伪器

规则的危险方向是**误删普通股**（少选几只权证只是少赚，删掉伯克希尔是数据错误）。所以测量主动去找反例：

**证伪器一（机械）**：把每一组与 `sp500_constituent` + `nasdaq100_constituent` 的 **966** 个不同代码求交。指数成分股必然是普通股，任何一个命中都是一次错误排除。结果：**九组全部为 0**。

**证伪器二（人工复核）**：2,310 个五字母命中里，有 **284** 个在名单里找不到佐证（没有 4 字母词根、没有同词根的其他 W/R/U、也没有 `ROOT-WS`/`-U`/`-R` 兄弟），中位挂牌 **1.9 年**，其中 36 个挂牌 ≥ 5 年。逐个按 NASDAQ 第五字符约定（R=权利、U=单位、W=认股权证）判读：**没有一个是普通股**，因此**不设** `COMMON_TICKER_ALLOWLIST`。

值得记下来的是**为什么它们看起来没有佐证**：挂牌最久的那批恰好是「**3 个字符的词根 + 双写后缀**」这个形状，而佐证规则是按 4 字符词根去找的，结构上就看不见它们——`TMCWW`/`TMC`、`HTZWW`/`HTZ`、`VLYWW`/`VLY`、`XOSWW`/`XOS`、`RNWWW`/`RNW`、`SMXWW`/`SMX`、`AUROW`/`AUR`、`QSIAW`/`QSI`，13 个可核对的短词根 13 个都在 `us_all` 里。其余（`SBNYW`、`GSMGW`、`CMPOW`、`IMAQU`…）的普通股已退市或被并购，与「SPAC 权证/单位」完全吻合。

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
| 回答的问题 | **某天谁是指数成分股** | **某天这只票值不值得交易** |
| 依据 | 指数成分历史（外部事实） | 价格、成交额、代码形态（规则） |
| 层次 | 数据集层 | 因子层（包装类） |
| 文档 | [constituent.md](constituent.md) | 本文 |

想「在标普 500 成分股里，只交易够贵够活的那些」，就两个一起用。
