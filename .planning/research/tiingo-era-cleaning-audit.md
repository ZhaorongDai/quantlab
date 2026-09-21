# 「Tiingo 时代清洗/过滤代码」盘点 — 哪些能删，哪些不能

调研日期：2026-09-20，HEAD `295ca51`。含对两个已落盘 CRSP store 的只读实测。

## 结论摘要

1. **`universe_filter.py` 不是 Tiingo 适配层**，是因子层的可交易性过滤器。3 个条件里
   **只有 1 个**被 CRSP 权威替代，另 2 个 CRSP 没有任何等价物。
2. **但它在生产代码里零调用** —— 只有 `tests/` 和 `example/` 引用。所以整体删除技术上零阻力。
3. **`cleaning.py` 绝大部分不是 Tiingo 补丁，而且它在 CRSP 上抓到了真 bug**。不要删。
4. **⚠ 已落盘的 CRSP store 是 CR-01/CR-02 修复前的产物**，所有"CRSP 有多干净"的数字都要重测。

## universe_filter.py 的三个条件

| 条件 | 行号 | 是 Tiingo 补丁吗 | CRSP 替代 | 建议 |
|---|---|---|---|---|
| **(a) 普通股代码正则**（9 条） | `:158-221`、`:355-362`、`:408-420`、`:544-548`、`:608-614` | **是，100%**。整个论证都是对 Tiingo `us_all` 14,481 个 ticker 的测量 | **完全替代且更强** | **删** |
| (b) `close[t] >= min_price`（默认 5.0） | `:225`、`:404-406`、`:541-543` | **不是**。流动性过滤 | **无** | 见决策 |
| (c) 20 根 `close*volume` 均值 >= 1e6 | `:226-227`、`:398-406`、`:530-543` | **不是**。流动性过滤 | **无** | 见决策 |

### 为什么 (a) 的替代是「更强」而非「等价」

CRSP Stock v2 的 ShareType/SecurityType/SecuritySubType 词表（`03.10-LIVE-CHECK-2.json` `L2_2`）
**根本没有 warrant / right / preferred / 测试码的编码**。全集只有
`EQTY/FUND/DERV(ATR,1983-92)` × `COM/CEF/ETF/ATR/UNK` × `NS/AD/SB/UG/CE/N-A`。
那 9 条正则要打的 7 类东西**在 CRSP 里压根不存在**；唯一真实存在的 unit 是 `sharetype=UG`，
`equity_common` 已排除（实测 sp500 报告里就丢了 CCL 的 `UG/EQTY/COM/CORP/N` 252 行）。

反过来 CRSP 还能排掉正则**做不到**的：ADR（`AD`）、ETF/CEF（`securitysubtype`）、类型未知行
（NULL 不匹配）、以及**逐行 per-date 判定**（某段是普通股、某段不是），并写审计报告。

而正则是静态、与日期无关、无审计、还可能误删——文档 `:184-191` 自己承认 2,310 个 5 字母命中里
有 284 个"找不到佐证"，靠人工判读才没设白名单。

### 为什么 (b)(c) 没有 CRSP 替代（三条证据）

1. `crsp.py:163-177` 注释明说 `FILTERABLE_COLUMNS` 是封闭的、每列都是 per-day **TYPE** 列，
   "a price column would also filter, but a ticker-picked panel that looks like a type-filtered
   one is exactly the silent substitution the report cannot catch"。
2. `resolve_security_filter`（`:294-302`）对非 TYPE 列直接 `ValueError`。
3. 全仓 grep `min_price`/`min_dollar_volume` 只在 `universe_filter.py` + 文档 + 测试里出现。

### 删 (a) 的附带红利

`_mask_panel` 的标的轴删列（`:608-614`）一并删掉后，LS-3 的「标的轴与日期窗口无关」变成
**天然成立**（掩码只剩阈值，永不删列），连带 `:596-599` 那段复杂论证和
`DLModel._align_prediction_symbols` 的风险一起消失。这是真正的简化。

### 生产零调用的证据

`grep -rn "UniverseFilteredFactor\|universe_filter"` 全部命中：自身、`utils/module.py:97`（仅注释
举例）、`tests/test_universe_filtered_factor.py`、`tests/universe_fixtures.py`、`example/universe.md`、
`example/README.md`。`main.py` / `train_model.py` / `scripts/*.py` / 任何 config **都没用**。

整体删除的代价：1,156 行测试 + 385 行 fixture + 一篇文档，以及**失去全仓唯一的「截面算子只看池内
标的」机制**（`:86-123` 的图改写，把 `Div(v, mask)` 注入每个 `CrossSectionalOp` 输入）。
`dataset/masking.py:UniverseMask` 做的是指数成分对齐，**不做**截面算子改写，两者不可互替。

## cleaning.py — 不要删（它在 CRSP 上抓到了真 bug）

实测已落盘的 `wrds_crsp_sp500_1d.zarr`（2024, 522×252）：
`anomaly_flag` True **699** 个 = `adjClose<=0` 686 ∪ 17 个 >50% 跳变（4 个重叠）。
那 686 个**正是 CR-01/CR-02 那个缺陷**（PXD/WRK/MRO/CTLT 四只退市股，`close=0.0`、
`is_delisting=1.0`、`volume=NaN`）。**`flag_anomalies` 是唯一把它顶到日志上的东西。**

| 函数 | 行号 | 为什么数据源无关 |
|---|---|---|
| `dedup_raw_frame` | `:63-76` | `to_xarray()` 在非唯一 MultiIndex 上**直接 raise**，是 API 硬约束。CRSP 不走它（用 `_assert_unique_panel_keys` 拒绝而非去重，更严格） |
| `validate_schema` 缺列 raise | `:203-208` | schema 违约；CRSP 靠它保证 12 个 Tiingo 变量名齐全 —— 这正是 drop-in 承诺的执行点 |
| `validate_schema` 结构性掩码 | `:210-273` | 稠密面板笛卡尔积的设计后果。没它 CRSP 会刷出 91,862 条 `numtrd` 假警报（实测，NYSE 不报 trade count） |
| `validate_schema` 空摄取 error | `:217-233` | 空厂商响应/全失败 backfill |
| `flag_anomalies` | `:79-136` | 见上，已在 CRSP 抓到真 bug |
| `clean_membership_panel` / `clean_nbbo_panel` | `:300-356` / `:359-436` | 形状契约 |
| 禁止 ffill/interpolate/fillna | `:25-27`、`:290-293` | D-06 反向约束，越权威的源越该守 |

### 删 `anomaly_flag` 的连带面（比想象大 —— 不建议）

它是全仓**唯一的布尔数据变量**，被当成「非 float64 变量」的测试标本：
`base/data.py:872`、`dataset/backend.py:339/489/1004/1412/1507`（bool→float64 静默转换守卫，
错误信息直接举它为例）、`dataset/masking.py:167-177`、`dataset/nbbo.py:368-375`，
测试 `test_cleaning.py`、`test_backend_indexes.py`、`test_symbol_axis_widening.py`、
`test_variable_axis_widening.py`、`test_constituent_panel.py:468`。
且三个已落盘 store 里都有这个变量，`03.10-06/13-SUMMARY` 把「变量集 == 12 Tiingo + 15 CRSP
extras + anomaly_flag」写成了 drop-in 契约的一部分。

## 绝对不要删

| 项 | 行号 | 理由 |
|---|---|---|
| `enums/data.py:TRADEABLE_TICKER_PATTERN` | `:183-237` | 不是证券类型过滤，是「能不能安全变成文件路径段」。真实事故：全 roster 预检在 `NXG-R-W` 上 raise，杀掉多小时全市场作业 |
| `acquisition/universe.py` 的 `_PREFERRED_SHARE_PATTERN` / `_BABY_BOND_PATTERN` | `:73-104` / `:106-110` | Tiingo roster 构建器的一部分；删了等于放弃 Tiingo 采集 |
| `_WELL_FORMED_TICKER` | `:483`、`:761` | Wikipedia cell 解析守卫 |
| `MIN_ROSTER_ROWS` | `:150`、`:279-287` | 防止词表漂移静默写入截断 roster |
| **volume guard / `--force-volume`** | `utils/cli.py:482-520+` | **与清洗无关** —— 是下载体积预检；CRSP 自己也在用（`test_ingest_wrds_crsp.py:283`） |
| `acquisition/tiingo.py:226-283` schema 稳定化 | | 纯 Tiingo 补丁，但删了 Tiingo 就抓不动 |
| `engine_vectorbt.py:38-52` 退市 ffill/强平 | | 回测语义，数据源无关 |

## ⚠ 必须先做：重建 CRSP store

`wrds_crsp_sp500_1d.zarr` 的 sidecar mtime 是 2026-09-20 13:46/13:47，而修 CR-01/CR-02 的提交
`3cbe787` 是 16:03。**盘上的 store 是修复前的产物。** 上面的 682/686/699 全是修复前的数字。

重建后必须重测：`anomaly_flag` 还剩多少、`adjClose<=0` 是否归零、`adjVolume` NaN 是否从 2,202 降下来。
这同时还掉 03.10 遗留的 rebuild debt。

## 附带发现：一个数据源无关的假阳性

CRSP 的 17 个 >50% raw close 跳变里 **12 个伴随 `splitFactor != 1`** —— `flag_anomalies` 读 raw
`close`（`:109-123`），所以在**任何**数据源上都会把拆股日误报为异常。这是**改进**（改用
`close/splitFactor` 或 `ret` 判跳变），不是删除。剩 5 个跳变成因未核实。

## 仍需实测

1. 重建 store 后的 `anomaly_flag` 统计（见上，最重要）。
2. `when_issued_or_called`（`-WD`/`-WI`/`-CL`）在 CRSP 里是否真不存在 —— 其他 6 类有词表直接佐证，
   这一类没有单独测量。建议对 CRSP 全库跑一次三个 type 列的 distinct 统计。
3. `validate_schema` 是否该给 CRSP 一份自己的 `required_columns`（`numtrd` 的 91,862 个 NaN 是
   厂商真不提供，每次转换 warn 一条 —— 噪音还是信号需要定）。
4. 有没有已落盘的、含 `UniverseFilteredFactor` 的 run artifact（未扫 `wandb/` 和 backtest 输出目录）。
5. CRSP 全库（非 S&P/NDX 名册）的仙股/低流动性占比未测 —— 这决定删 (b)(c) 的代价是 0 还是不小。
   当前两个 store 上：`close<5` 只有 4 个格子（全是哨兵 0.0）、`close*volume<1e6` **0 个格子**。
