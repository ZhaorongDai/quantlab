# 指数成分与标的池（constituent / universe）

## 一句话

这一层回答一个问题：**"在历史上的某一天，哪些股票属于这个池子？"**——把"标普 500 今天有哪些成分股"这种**当前快照**，变成"1997 年 3 月 12 日标普 500 有哪些成分股"这种**时点（point-in-time）**查询，从而让回测不带幸存者偏差。

它由三部分组成：

| 层 | 文件 | 职责 |
|---|---|---|
| 采集 + 参考表 | `acquisition/universe.py` | 抓数据、重建历史成分区间、落成一张 `universe.parquet` 参考表，并提供时点查询 |
| 面板化 | `base/constituent.py` + `dataset/constituent.py` | 把"区间表"稠密化成 `(timestamp, symbol) -> is_member` 的布尔面板，存 Zarr |
| 掩码 | `dataset/masking.py` | 把布尔面板盖到价格面板上，非成分的格子置 NaN，并报告覆盖缺口 |

---

## 不用它会怎样

**幸存者偏差（survivorship bias）会系统性地把回测收益抬高，而且不会报错。**

想象一个最常见的错误做法：去某个网站下载"标普 500 当前成分股列表"，拿到 503 个代码，然后把这 503 个代码的 2007 年至今的价格拉下来，做一个等权组合回测。

问题在于：**这 503 个名字是"活下来的人"。**

用本仓库真实的表跑一下（下面例 2 有完整代码），可以量化这件事：

```
2007-06-29 那天的标普 500 成分股：476 个
2026-09-01 那天的标普 500 成分股：503 个
2007 年在、今天不在的：              231 个
```

也就是说，**2007 年的成分股里有 231 个（约 49%）在这十九年里被踢出了指数**。它们是怎么走的？破产（`LEH`，雷曼兄弟，实测 `'LEH' in 2007 → True`，`in 2026 → False`）、被收购、被降级到中盘指数、退市（`ABK` 安巴克金融、`CFC` 全国金融、`FNM`/`FRE` 两房、`EK` 柯达）。

这些恰恰是**表现最差的一批公司**。只拿"今天的 503 个"回测过去十九年，等于事先知道了"哪些公司挺过了金融危机"，把雷曼、两房、柯达全部从样本里删掉了。回测出来的夏普比率会好看得不像话，实盘会立刻打回原形。

同一个偏差还有一个更隐蔽的版本：即使你意识到要用退市股，如果价格数据源只提供"当前上市公司"的历史（比如 `nasdaqlisted.txt`），你根本拿不到已退市公司的价格。所以 `TiingoRosterFetcher.SOURCE_URL` 用的是 Tiingo 的 `supported_tickers.zip` 而**不是** `nasdaqlisted.txt`——后者按定义就无法表达退市历史（见 `acquisition/universe.py` 模块 docstring）。实测 `us_all` 的 15,167 行里有 7,018 行的 `end_date` 早于 2026-01-01——那都是已退市的代码，它们还在表里。

---

## 核心概念

### point-in-time（时点）

一张参考表，如果只记录"现在的状态"，它就永远只能回答现在。时点表记录的是**区间**：某个代码在某个类别里，从哪天开始、到哪天结束。查询"1997-03-12 谁是成分股"就退化成一次区间过滤。

`UniverseCatalog` 的 docstring 里有一句必须记住的话：

> `get_symbols_as_of()` must be called **per-rebalance-date** in a walk-forward backtest, not once at setup time, to avoid look-ahead bias.

在回测循环里每次调仓都重新问一次，而不是开局问一次然后一路用到底。后者就是前视偏差。

### 成分区间（interval）与区间的两端闭合

参考表 `data/data/reference/universe.parquet` 的 schema（实测）：

```
symbol: String, category: String, start_date: String, end_date: String, end_date_is_inferred: Boolean
```

- `start_date` / `end_date` 是 **ISO 字符串**，比较是**字典序**比较。这就是为什么 `UniverseCatalog._validate_iso_date` 必须拒绝非 ISO 日期——`"2020/01/02"` 不会"匹配不上"，它会**比较错**，然后返回一个看起来合理的错名单。
- `end_date` 为 `null` 表示"至今仍是成员"。
- **区间两端都是闭区间。** 一个在 D 日被剔除的股票，D 日读作 `True`，D+1 日才是 `False`。这个约定在两个地方必须完全一致：`UniverseCatalog.get_symbols_as_of` 的 `start_date <= as_of_date & (end_date.is_null() | end_date >= as_of_date)`，和 `IndexConstituentDataset._densify` 的 `(timestamps >= start) & (timestamps <= end)`。`base/constituent.py` 的类 docstring 明确点出：两边一旦不一致，面板和查询会在**历史上每一次剔除**都差一天，而运行时什么都不会报。

### 区间重叠查询 vs 时点查询

两个查询长得像，用途完全不同：

| 方法 | 谓词 | 用途 |
|---|---|---|
| `UniverseCatalog.get_symbols_as_of(category, as_of_date)` | `start <= d <= end` | **walk-forward 回测每次调仓**问"今天谁在池子里" |
| `UniverseCatalog.get_symbols_in_range(category, start, end)` | `start_date <= end AND (end_date IS NULL OR end_date >= start)` | **全窗口下载数据**时问"这段时间里出现过谁"（要把窗口内退市的也下下来） |

`get_symbols_in_range` 的 docstring 说得很清楚，它保留的正是"在窗口中间退市的那些票"。实测 `us_all` 的 15,167 行里，有 **7,018 行的 `end_date` 早于 2026-01-01**（即已退市），把它们排除掉就是幸存者偏差本身。

### 可回溯起点 `PIT_COVERAGE_START`

历史成分是从维基百科的**变更日志**重建的，而变更日志本身有个最早的一行。在那行之前，源数据**根本没有信息**。

- `SP500MembershipFetcher.PIT_COVERAGE_START = "1976-07-01"` —— 是那张 `id="changes"` 表实测最早的一行；注意**不是**该页正文自称的 1963 年（`acquisition/universe.py` docstring 明确写了这个坑）。
- `Nasdaq100MembershipFetcher.PIT_COVERAGE_START = "2007-02-01"` —— 是纳指 100 变更表实测最早的一行（`LOGI` 纳入 / `CMVT` 剔除）。

两个日期差了 31 年。docstring 里专门警告：**这两个类别不能被悄悄 union 到同一条时间轴上**，那会暗示一段谁都没有的覆盖范围。

在这个日期之前查询，`get_symbols_as_of` 会**抛 `ValueError`**，而不是返回一份不完整的名单（见例 2）。原因很直白：返回一份"1970 年的标普 500，只有 327 个名字"看起来完全正常，没人会发现它错了；抛异常会立刻停下来。

在面板层，同一件事表现为 `IndexConstituentDataset._clamp_coverage_start()`：配置里的 `start_date` 会被抬到覆盖起点。不抬会怎样？`BaseDatasetConfig` 继承的默认 `Date.START_DATE` 是 `"1900-01-01"`，面板会平白多出 76 年全 `False` 的行——而 `False` 的语义是"不是成员"，不是"不知道"。

### 四个 category

`enums/data.py`：

```python
UniverseCategory = Literal[
    "nasdaq_all", "us_all", "sp500_constituent", "nasdaq100_constituent"
]
```

实测规模（`data/data/reference/universe.parquet`，共 25,696 行 / 14,593 个唯一代码）：

| category | 行数 | 唯一代码 | 最早 start | 开区间数 | `end_date_is_inferred` |
|---|---|---|---|---|---|
| `us_all` | 15,167 | 14,479 | 1962-01-02 | 0 | 0 |
| `nasdaq_all` | 9,311 | 8,960 | 1970-01-02 | 0 | 0 |
| `sp500_constituent` | 905 | 876 | 1957-03-04 | 503 | 18 |
| `nasdaq100_constituent` | 313 | 275 | 2007-02-01 | 102 | 3 |

它们分成**两族，语义刻意不同**：

**族一：交易所名册（roster），由 `UniverseCatalog.ROSTER_FETCHERS` 注册**

- `nasdaq_all`（`NasdaqUniverseFetcher`）：`EXCHANGE_FILTER = ("NASDAQ",)`，`MIN_ROSTER_ROWS = 1000`。
- `us_all`（`USEquityUniverseFetcher`）：`EXCHANGE_FILTER = ("NASDAQ", "NYSE", "AMEX", "NYSE MKT")`，`MIN_ROSTER_ROWS = 8000`，并且 `EXCLUDE_NON_COMMON_SECURITY_TYPES = True`（剔除优先股与 baby bond）。

名册里的 `start_date`/`end_date` 是 **Tiingo 报告的真实上市/退市日**，所以 `end_date_is_inferred` 恒为 `False`，而且**没有 `PIT_COVERAGE_START`**——名册天然没有"指数成员"这个概念，也就没有覆盖边界。实测 `get_symbols_as_of('us_all', '1970-01-02')` 正常返回 28 个，不报错。

两者是**兄弟而非替代**：`us_all` 是 `nasdaq_all` 的超集，但 `nasdaq_all` 的语义被锁定（Locked Decision A4 / D-02）不允许被改动，这就是为什么"剔除优先股"是一个 `EXCLUDE_NON_COMMON_SECURITY_TYPES` 开关而不是共享的无条件过滤——只有 `us_all` 打开它。

**族二：指数成分（membership），由 `UniverseCatalog.MEMBERSHIP_FETCHERS` 注册**

- `sp500_constituent`（`SP500MembershipFetcher`）
- `nasdaq100_constituent`（`Nasdaq100MembershipFetcher`）

这一族才有 `PIT_COVERAGE_START`，才有 `end_date_is_inferred=True` 的可能，才有真正的 `null`（开区间）。

**为什么必须是两个独立注册表？** `UniverseCatalog.MEMBERSHIP_FETCHERS` 的注释写明：注册进 `MEMBERSHIP_FETCHERS` 就等于获得一个时点覆盖边界；把名册注册进去会给它强加一个它不该有的边界，导致早期的合法查询开始报错。反过来，`get_symbols_as_of` 的边界字典是从 `MEMBERSHIP_FETCHERS` **推导**出来的，所以加第三个指数只需要注册，不需要有人记得再加一个 `if` 分支。

---

## 数据是怎么重建出来的

核心算法在 `IndexMembershipFetcher.reconstruct_intervals(anchor, changes)`（`acquisition/universe.py:899`）。输入两样东西：

1. **anchor（当前成分快照）**：`fetch_anchor()`，标普 500 来自 GitHub 上的 `constituents.csv`，纳指 100 来自 stockanalysis.com 抓取。它是 **"今天谁是成员" 的权威**。
2. **changes（带日期的变更日志）**：`fetch_changes()`，来自维基百科 "Historical components of ..." 页面的表格，每行是 `(生效日, 纳入代码, 剔除代码)`。它是 **"什么时候变的" 的权威**。

### 算法：正向重放 + 三向对账

```
last_eff = PIT_COVERAGE_START
open_intervals = {}          # symbol -> start_date
closed = []                  # (symbol, start, end, end_date_is_inferred)

按 effective_date 升序遍历变更日志:
    last_eff = eff
    如果这行有剔除代码 sym:
        如果 sym 在 open_intervals 里  -> 闭掉：(sym, 开始日, eff, False)
        否则                          -> 左截断！(sym, PIT_COVERAGE_START, eff, False) + warning
    如果这行有纳入代码 sym:
        如果 sym 已经开着            -> warning（数据质量问题），保留更早的开始日
        否则                          -> open_intervals[sym] = eff
```

遍历完之后，日志和 anchor 会在**恰好三个方向**上打架，全部要对账，一个都不能默默放过：

1. **区间还开着，但 anchor 里没有这个代码。** 说明日志漏了一次剔除。做法：用 `last_eff`（日志的右边缘，**不是**关于这个代码的任何观测）闭掉它，并把 `end_date_is_inferred=True`。
2. **日志里这个代码的最后一个事件是剔除，但 anchor 说它现在还是成分股。** 说明日志漏了一次重新纳入（纳指 100 的日志已知有 16 行只有一边）。做法：**从那个剔除日重新开一个开区间**，并 warning。不处理的话，这个当前成分股会被永久记成"前成分股"，`get_symbols_as_of(category, today)` 会少一个人。
3. **anchor 里的代码在日志里从没出现过。** 要么是原始成分股，要么在 `PIT_COVERAGE_START` 之前就纳入了。做法：`start = anchor 的 date_added or PIT_COVERAGE_START`。

这就是为什么 `sp500_constituent` 的最早 `start_date` 实测是 **1957-03-04** 而不是 1976-07-01：那是 anchor CSV 自带的 `date_added`，走的是第 3 条分支。而**没有** `date_added` 的纳指 100（`fetch_anchor()` 造一列全 null），最早 `start_date` 实测恰好就是 `2007-02-01`。

### 左截断（left-censored）区间为什么用 `PIT_COVERAGE_START` 兜底

日志里出现一条"1993 年剔除 XYZ"，但从来没有过"纳入 XYZ"——因为 XYZ 是 1970 年进的指数，而日志从 1976 才开始。这就是左截断。

`start_date` 必须填个东西。填 `PIT_COVERAGE_START` 的含义是："**在我们能看到的最早那天，它已经是成员了**"——这是源数据支持的最强、也是唯一诚实的陈述。填 `null` 会被 `_densify()` 直接拒绝（见下），填一个瞎猜的日期则是凭空捏造。

实测例子（柯达）：

```
symbol  category           start_date   end_date     end_date_is_inferred
EK      sp500_constituent  1976-07-01   2010-12-17   false
```

`1976-07-01` 就是 `SP500MembershipFetcher.PIT_COVERAGE_START`。柯达当然不是 1976 年才进标普 500 的——这个日期读作"截止我们知道的最早时刻，它已经在里面了"。

### 为什么超出可回溯起点必须报错

因为 `[]` 和"一份不完整的名单"都是**合法的返回值形状**。

- `get_symbols_as_of('nasdaq_all', '1970-01-02')` 返回一个很短的列表是**对的**——那时候上市公司就这么少。
- `get_symbols_as_of('sp500_constituent', '1970-01-01')` 如果返回 327 个名字，那是**错的**，但它长得跟对的一模一样。

所以 `get_symbols_as_of` 在 `as_of_date < coverage_start` 时抛 `ValueError`。同样的逻辑覆盖了另外两种输入错误：未知 category 抛错（否则一个 typo 会静默地什么都不下载），非 ISO 日期抛错（否则字典序比较会返回一份"看起来合理但错了"的名单）。三种错误都在例 2 里实跑了。

### 一些安全阀（都在基类上，子类白拿）

`IndexMembershipFetcher` docstring 列了三条，它们是**安全属性**不是实现细节：

1. **表头校验**：`_parse_changes_table()` 用 `EXPECTED_SOURCE_HEADER` **选表**并按名字取列。以前是按位置取列然后校验自己刚赋的名字（恒真的校验），而抓取源最容易发生的漂移恰恰是**表头重排**——列数不变、顺序变了，结果就是纳入/剔除被静默颠倒。
2. **行数单调性**：线上表比缓存快照行数还少 = 解析失败/schema 漂移。成分关系只会关闭，不会追溯性消失。
3. **非破坏性回退**：任何抓取/解析失败都返回缓存快照，且**不覆盖缓存文件**，一次坏解析毒不了后面所有的运行。

配套的是 `UniverseCatalog.build(allow_stale=False)`：如果某个 fetcher 走了缓存回退，默认**拒绝**构建。因为 `save()` 写出去的 stale 表和新鲜表长得一模一样，一个永久坏掉的源会静默地把整个 universe 冻结在缓存日期上，只留一行 `logger.error` 在 cron 日志里。

### 稠密化：区间表 → 布尔面板

`IndexConstituentDataset._densify()`（`base/constituent.py:149`）把区间表变成 `(timestamp, symbol) -> is_member` 的布尔网格。几条规则值得记：

- **symbol 轴取全时段并集**，在任何日期过滤**之前**算。一个成分历史完全落在面板窗口之外的代码，仍然会得到一整列 `False`。这既是幸存者偏差保证，也让 `XrBackend.filter_by_symbol` 的 `.sel(symbol=[...])` 在请求一个从没当过成分的代码时**大声 `KeyError`**，而不是悄悄消失。
- **左边界** = `max(config.start_date, _pit_coverage_start())`。
- **右边界** = `min(config.end_date, horizon)`。如果**存在**开区间，`horizon = max(observed, today)`——因为开区间按定义就是"现在还在"，最后一次变更事件只是右边界的**下界**；停在 `observed` 会让面板比现在短几周甚至几个月，而那正是掩码最常被用到的地方。`today` 取 `config.as_of`（如果设了）否则取墙上时钟。**要可复现就必须钉 `as_of`**：不钉的话同一份配置两天跑出来是两个不同形状的 store，而 `save()` 是 `mode="w"` 覆盖写。
- **时间轴是日历日**（`pd.date_range(..., freq="D")`），不是交易日。周末和节假日的值是上一个交易日的成分关系顺延。所以和 OHLCV 面板 join 时**必须 reindex 或 `.sel()`**，不能假设两条轴对齐。`UniverseMask` 就是这么做的（时间戳做 inner join，且**故意不报告**被丢掉的行——那几千行是构造使然）。
- **null `start_date` 直接抛 `ValueError`**，不容忍。`pd.Timestamp(None)` 是 `NaT`，而 `NaT` 的所有比较都是 `False`，`max()` 会静默返回它碰巧先拿到的那个值，把整个 horizon 算错；同一个 null 再走到填充循环里会产出一整列 `False`，和"从没当过成分"完全无法区分。

### 掩码：`dataset/masking.py:UniverseMask`

`apply()` 返回 `market.where(mask)`——非成分的格子变 NaN，**所有变量统一处理，布尔标志位也不例外**（`anomaly_flag` 会变成 float64 + NaN）。理由：在池子之外，一个 flag 是"未定义"，不是 `False`，保留 `False` 等于断言了这个掩码并不知道的事。

`apply()` 会先跑 `report()`，所以打掩码永远不可能变成一种绕过覆盖报告的安静做法。报告有**两个刻意的不对称**：

- **成分股在价格面板里找不到 → 逐个点名**（`logger.warning`，完整列表，永不截断永不采样）。这是**覆盖缺口**，静默丢掉等于从最难拿到的那批名字（长期退市股）重新引入幸存者偏差。
- **价格面板里的代码不是成分股 → 一声不吭丢掉**。那只是"不在池子里"，不是任何缺口。

---

## 完整例子

### 例 1：直接读参考表（polars，无需任何凭证）

```python
import polars as pl

df = pl.read_parquet("data/data/reference/universe.parquet")
print(df.shape)
print(df.head(8))
print(
    df.group_by("category").agg(
        pl.len().alias("rows"),
        pl.col("symbol").n_unique().alias("symbols"),
        pl.col("start_date").min().alias("min_start"),
        pl.col("end_date").max().alias("max_end"),
        pl.col("end_date").is_null().sum().alias("open_intervals"),
        pl.col("end_date_is_inferred").sum().alias("inferred"),
    ).sort("category")
)
print("unique symbols total", df["symbol"].n_unique())
```

实跑输出：

```
(25696, 5)
shape: (8, 5)
┌────────┬────────────┬────────────┬────────────┬──────────────────────┐
│ symbol ┆ category   ┆ start_date ┆ end_date   ┆ end_date_is_inferred │
╞════════╪════════════╪════════════╪════════════╪══════════════════════╡
│ AAAP   ┆ nasdaq_all ┆ 2015-11-11 ┆ 2026-09-04 ┆ false                │
│ AABA   ┆ nasdaq_all ┆ 1996-04-12 ┆ 2019-11-06 ┆ false                │
│ AACB   ┆ nasdaq_all ┆ 2025-04-07 ┆ 2026-09-04 ┆ false                │
│ AACBR  ┆ nasdaq_all ┆ 2025-04-07 ┆ 2026-08-19 ┆ false                │
│ AACBU  ┆ nasdaq_all ┆ 2025-02-13 ┆ 2026-09-04 ┆ false                │
│ AACC   ┆ nasdaq_all ┆ 2004-02-05 ┆ 2013-10-17 ┆ false                │
│ AACG   ┆ nasdaq_all ┆ 2008-01-29 ┆ 2026-09-04 ┆ false                │
│ AACI   ┆ nasdaq_all ┆ 2021-11-10 ┆ 2026-09-04 ┆ false                │
└────────┴────────────┴────────────┴────────────┴──────────────────────┘
shape: (4, 7)
┌───────────────────────┬───────┬─────────┬────────────┬────────────┬────────────────┬──────────┐
│ category              ┆ rows  ┆ symbols ┆ min_start  ┆ max_end    ┆ open_intervals ┆ inferred │
╞═══════════════════════╪═══════╪═════════╪════════════╪════════════╪════════════════╪══════════╡
│ nasdaq100_constituent ┆ 313   ┆ 275     ┆ 2007-02-01 ┆ 2026-08-04 ┆ 102            ┆ 3        │
│ nasdaq_all            ┆ 9311  ┆ 8960    ┆ 1970-01-02 ┆ 2026-09-04 ┆ 0              ┆ 0        │
│ sp500_constituent     ┆ 905   ┆ 876     ┆ 1957-03-04 ┆ 2026-08-18 ┆ 503            ┆ 18       │
│ us_all                ┆ 15167 ┆ 14479   ┆ 1962-01-02 ┆ 2026-09-04 ┆ 0              ┆ 0        │
└───────────────────────┴───────┴─────────┴────────────┴────────────┴────────────────┴──────────┘
unique symbols total 14593
```

读法：

- `nasdaq100_constituent` 有 **102** 个开区间，正好对上 `Nasdaq100MembershipFetcher` docstring 说的 "102 rows, not 100"——GOOGL/GOOG、FOX/FOXA 这类同一发行人的多个股份类别。
- 两个 roster 类别的 `open_intervals` 和 `inferred` 都是 **0**：Tiingo 报告真实的上市/退市日，永远不需要推断。

**一家公司的多个区间**（Netflix，实测）：

```
┌────────┬───────────────────────┬────────────┬────────────┬──────────────────────┐
│ symbol ┆ category              ┆ start_date ┆ end_date   ┆ end_date_is_inferred │
╞════════╪═══════════════════════╪════════════╪════════════╪══════════════════════╡
│ NFLX   ┆ nasdaq100_constituent ┆ 2010-12-20 ┆ 2012-12-24 ┆ false                │
│ NFLX   ┆ nasdaq100_constituent ┆ 2013-06-06 ┆ null       ┆ false                │
│ NFLX   ┆ nasdaq_all            ┆ 2002-05-23 ┆ 2026-09-04 ┆ false                │
│ NFLX   ┆ sp500_constituent     ┆ 2010-12-17 ┆ null       ┆ false                │
│ NFLX   ┆ us_all                ┆ 2002-05-23 ┆ 2026-09-04 ┆ false                │
└────────┴───────────────────────┴────────────┴────────────┴──────────────────────┘
```

Netflix 2002-05-23 上市（roster 层，两个类别一致），2010-12-17 进标普 500 至今，纳指 100 则进了两次：2010-12-20 进、2012-12-24 被踢、2013-06-06 又回来。**如果只看"当前成分"，2013 上半年 Netflix 会被错误地算作纳指 100 成分股。**

**已退市公司（Yahoo）**：

```
│ YHOO   ┆ nasdaq100_constituent ┆ 2007-02-01 ┆ 2017-06-19 ┆ false                │
│ YHOO   ┆ nasdaq_all            ┆ 1996-04-12 ┆ 2017-06-26 ┆ false                │
│ YHOO   ┆ sp500_constituent     ┆ 1999-12-08 ┆ 2017-06-19 ┆ false                │
│ YHOO   ┆ us_all                ┆ 1996-04-12 ┆ 2017-06-26 ┆ false                │
```

注意 `nasdaq100_constituent` 的 `start_date` 恰好是 `2007-02-01` = `Nasdaq100MembershipFetcher.PIT_COVERAGE_START`：Yahoo 当然在 2007 年之前就在纳指 100 里了，这是左截断兜底的结果。同时它在 `nasdaq_all` 里的真实上市日是 1996-04-12，**证明退市公司确实留在了价格 roster 里**（Yahoo 后来更名 `AABA`，`1996-04-12 → 2019-11-06`）。

### 例 2：用 `UniverseCatalog` 做时点查询，并撞覆盖边界

```python
from base.config import UniverseConfig
from acquisition.universe import UniverseCatalog

cfg = UniverseConfig(
    output_path="data/data/reference/universe.parquet",
    cache_dir="data/data/reference/_cache",
)
cat = UniverseCatalog.load(cfg)          # 只读 parquet，不联网

print("known_categories:", sorted(cat.known_categories()))
for d in ["1980-01-02", "1990-01-02", "2000-01-03", "2005-01-03",
          "2010-06-30", "2015-06-30", "2020-06-30", "2026-09-01"]:
    print(d, len(cat.get_symbols_as_of("sp500_constituent", d)))

s2007 = set(cat.get_symbols_as_of("sp500_constituent", "2007-06-29"))
s2026 = set(cat.get_symbols_as_of("sp500_constituent", "2026-09-01"))
print("2007 有 / 2026 没有 的数量:", len(s2007 - s2026))
print("LEH in 2007?", "LEH" in s2007, "| in 2026?", "LEH" in s2026)

# 越过可回溯起点
for cat_name, d in [("sp500_constituent", "1970-01-01"),
                    ("nasdaq100_constituent", "2006-12-31")]:
    try:
        cat.get_symbols_as_of(cat_name, d)
    except ValueError as e:
        print("ValueError:", e)

# roster 没有边界
print("us_all 1970-01-02 ->", len(cat.get_symbols_as_of("us_all", "1970-01-02")))

# 另外两种静默错误也会抛
for args in [("sp500", "2020-01-02"), ("sp500_constituent", "2020/01/02")]:
    try:
        cat.get_symbols_as_of(*args)
    except ValueError as e:
        print("ValueError:", e)
```

实跑输出：

```
known_categories: ['nasdaq100_constituent', 'nasdaq_all', 'sp500_constituent', 'us_all']
1980-01-02 327
1990-01-02 363
2000-01-03 420
2005-01-03 458
2010-06-30 507
2015-06-30 516
2020-06-30 521
2026-09-01 503

2007 有 / 2026 没有 的数量: 231
LEH in 2007? True | in 2026? False

ValueError: Cannot answer sp500_constituent membership before 1976-07-01 -- the Wikipedia-sourced change log is left-censored at that date and this query cannot be answered correctly, rather than silently defaulting to an incomplete/wrong answer.

ValueError: Cannot answer nasdaq100_constituent membership before 2007-02-01 -- the Wikipedia-sourced change log is left-censored at that date and this query cannot be answered correctly, rather than silently defaulting to an incomplete/wrong answer.

us_all 1970-01-02 -> 28

ValueError: Unknown universe category 'sp500'; known categories are ['nasdaq100_constituent', 'nasdaq_all', 'sp500_constituent', 'us_all'].

ValueError: as_of_date must be an ISO YYYY-MM-DD string, got '2020/01/02'. The table stores ISO date strings and compares them LEXICOGRAPHICALLY, so a non-ISO value does not merely fail to match -- it compares wrong and returns a plausible, silently incorrect roster.
```

两个日期都精确等于对应类的 `PIT_COVERAGE_START` 常量，而 `us_all` 完全没有边界（1970 年正常返回 28 个），正是前面说的两族语义差异。

### 例 3（附加）：区间 → 面板 → 掩码，全离线

这段代码自己造区间表，不联网、不读任何 store，可以直接跑：

```python
import numpy as np, pandas as pd, polars as pl, xarray as xr
from base.config import ConstituentDatasetConfig
from base.constituent import IndexConstituentDataset
from dataset.masking import UniverseMask


class DemoPanel(IndexConstituentDataset):
    def _pit_coverage_start(self) -> str:
        return "2020-01-01"

    def _build_intervals(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                ("AAA", "2020-01-01", None),          # 一直在指数里
                ("BBB", "2020-01-01", "2020-01-05"),  # 2020-01-05 被剔除
                ("CCC", "2020-01-04", None),          # 2020-01-04 被纳入
            ],
            schema=["symbol", "start_date", "end_date"], orient="row",
        )


cfg = ConstituentDatasetConfig(
    zarr_file_path="/tmp/demo_panel.zarr", cache_dir="/tmp/demo_cache",
    start_date="2019-06-01",   # 早于 coverage start，会被夹住
    end_date="2020-01-08",
    as_of="2020-01-08",        # 钉住右边界 -> 可复现
)
ds = DemoPanel(cfg)
print("config.start_date after clamp =", ds.config.start_date)

panel = ds.from_raw_data().get_xarray_dataset()
print(panel["is_member"].to_pandas().astype(int))

ts = pd.date_range("2020-01-01", "2020-01-08", freq="D")
syms = ["AAA", "BBB", "DDD"]   # DDD 不是成分股；CCC 价格里没有 -> 覆盖缺口
close = xr.Dataset(
    {"close": (["timestamp", "symbol"],
               np.arange(len(ts) * len(syms), dtype=float).reshape(len(ts), len(syms)))},
    coords={"timestamp": ts, "symbol": syms},
)
mask = UniverseMask(close, panel)
print(mask.report())
print(mask.apply()["close"].to_pandas())
```

实跑输出（含 loguru 的告警）：

```
WARNING | base.constituent:_clamp_coverage_start:110 - DemoPanel: requested start_date 2019-06-01 is before this index's point-in-time coverage start 2020-01-01; the panel's left edge was clamped to 2020-01-01. Membership before that date cannot be answered from the source.
WARNING | dataset.masking:report:152 - UniverseMask: 1 of 3 in-window index member(s) are absent from the market panel entirely and are dropped by the alignment. Every dropped name is a survivorship-bias hole, so the COMPLETE list follows: ['CCC']

config.start_date after clamp = 2020-01-01

symbol      AAA  BBB  CCC
timestamp
2020-01-01    1    1    0
2020-01-02    1    1    0
2020-01-03    1    1    0
2020-01-04    1    1    1
2020-01-05    1    1    1
2020-01-06    1    0    1
2020-01-07    1    0    1
2020-01-08    1    0    1

{'in_window_members': 3, 'missing_count': 1, 'missing_symbols': ['CCC']}

symbol       AAA   BBB
timestamp
2020-01-01   0.0   1.0
2020-01-02   3.0   4.0
2020-01-03   6.0   7.0
2020-01-04   9.0  10.0
2020-01-05  12.0  13.0
2020-01-06  15.0   NaN
2020-01-07  18.0   NaN
2020-01-08  21.0   NaN
```

四件事一次看清：

1. **左边界被夹到 2020-01-01**，并且因为是显式请求的所以发了 warning（如果 `start_date` 还是继承来的 `Date.START_DATE` 哨兵，会静默夹住——不然每次默认构造都告警，读者会学会无视日志）。
2. **闭区间约定**：`BBB` 的 `end_date` 是 `2020-01-05`，面板上 `2020-01-05` 读 `1`，`2020-01-06` 才是 `0`。
3. **覆盖缺口被点名**：`CCC` 是成分股但价格面板没有它，`report()` 给出完整列表。`DDD` 在价格面板里但不是成分股，一声不吭地丢掉——这是那两个刻意的不对称。
4. **掩码结果**：`BBB` 出局后是 `NaN` 而不是 `0`。

### 重建这张表

```
uv run python refresh_us_equity_universe.py --help
```

```
usage: refresh_us_equity_universe.py [-h] [--allow-stale]

Build/refresh the point-in-time US-equity universe reference table. Requires
no API key.

options:
  -h, --help     show this help message and exit
  --allow-stale  Persist even if a change-log fetch/parse failed and its
                 cached snapshot was used. Off by default: a stale table
                 written over universe.parquet looks exactly like a fresh one.
```

不需要 API key（维基 + GitHub CSV + stockanalysis.com + Tiingo 的公开 `supported_tickers.zip`）。会联网。

---

## 已知的坑

### 坑 1（**必读，major**）：成分面板和价格 roster 的代码词表跨改名对不上

**今天（2026-09-07）实测：876 个 `sp500_constituent` 代码里，有 89 个在 `us_all` 价格 roster 里查无此符号。**

```python
sp = set(df.filter(pl.col("category") == "sp500_constituent")["symbol"])
us = set(df.filter(pl.col("category") == "us_all")["symbol"])
missing = sorted(sp - us)
# len(sp) = 876, len(missing) = 89
```

根源是**两个数据源用不同的代码词表**：

- **成分历史**来自维基百科变更日志，记录的是**事件当天的代码**。所以标普 500 面板里合法地存在 `FB`。
- **价格**来自 Tiingo 目录，Tiingo 会把改名证券的**全部历史重写到新代码下**。`FB` 在那边根本不存在，`META` 的 `start_date = 2012-05-18`（Facebook 的 IPO 日）。

三类原因，实测确认：

**(a) 改名**——旧代码只在成分表里，新代码只在价格表里：

| 旧（成分表有，价格表无） | 新（价格表有） |
|---|---|
| `FB` | `META` |
| `CDAY` | `DAY` |
| `DWDP` | `DD` |
| `COG` | `CTRA` |
| `ADS` | `BFH` |

实测 `{k: (k in us, v in us)}` → 五对全是 `(False, True)`。

而且 `FB` 这一行在表里长这样：

```
symbol  category           start_date   end_date     end_date_is_inferred
FB      sp500_constituent  2013-12-23   2026-08-18   true
```

`end_date_is_inferred = True` 正是 `reconstruct_intervals` 的**第 1 类对账**：日志里 `FB` 的区间一直开着（因为改名不是一次"剔除"事件），但 anchor 里只有 `META`，于是用 `last_eff`（日志右边缘）把它闭掉并打上 inferred 标记。同时 `META` 自己也有一行 `sp500_constituent 2013-12-23 → null`。**同一家公司在成分表里有两条 900 天重叠不上的记录。**

**(b) 记法差异**——同一只证券，两边分隔符不同：

```
dot-spelled missing:              ['BF.B', 'BRK.B']
present in us_all after '.'->'-': ['BF-B', 'BRK-B']
```

实测 `BRK.B` 在 `sp500_constituent`（`2010-02-16 → null`），`BRK-B` 在 `us_all`（`1996-05-09 → 2026-09-04`）。伯克希尔在两边都在，只是写法不同。注意 `TRADEABLE_TICKER_PATTERN`（`enums/data.py`）**两种分隔符都接受**，所以这纯粹是 join 问题，不是校验问题。

**(c) 交易所过滤 / 长期退市**——`CBOE` 实测在 `us_all` 和 `nasdaq_all` 里都没有，因为它在 CBOE 交易所上市，被 `USEquityUniverseFetcher.EXCHANGE_FILTER = ("NASDAQ", "NYSE", "AMEX", "NYSE MKT")` 排除了。另外 `EK`（柯达）、`BS`（伯利恒钢铁）、`CEPH`、`ABK`、`CFC`、`FNM`、`FRE` 等年代久远的退市股 Tiingo 目录也没有。

**为什么这个坑会在回测阶段咬人：** 失败是**静默的**，而且给出**错误答案**而不是错误。一个成分股如果它的时点代码在价格面板里没有对应列，读起来就是"这个代码在这个窗口没有数据"——和真正的数据缺失完全无法区分。回测会安安静静地把 Facebook 从每一个 2022 年之前的标普 500 组合里剔掉，而不是拒绝运行。

**现在还打不到，所以是 filed 而不是 fixed**：当前用法传的 `--as-of-date` 都是近期日期，那时候的成分代码已经是当前代码了。**Phase 4+ 一旦有模型或回测在历史里 walk membership，它就变成活的了。**

两条 todo：

- `.planning/todos/pending/2026-09-07-no-ticker-rename-mapping-between-membership-history-and-pric.md`（major，改名，候选方案：Tiingo `permaTicker` 身份键 / 显式改名映射表 / 构建期对账）
- `.planning/todos/pending/2026-09-07-normalize-ticker-delimiter-between-membership-panel-and-pric.md`（minor，`.` vs `-`）

todo 里有一条设计约束值得单独记住：**无论选哪个方案，join 在遇到无法解析的成分代码时必须大声失败，而不是产出一个空列。这条性质比任何具体的映射机制都值钱。**

顺带：`UniverseMask.report()` 就是**目前唯一能让这个坑现形的机制**（它会把每一个"是成分但价格面板没有"的代码完整列出来）。`dataset/masking.py` 的 docstring 说得很直白——报告非空是一个**需要处理的发现**，不是一个可以在这里糊过去的洞。

### 坑 2：重建出来的历史成分数量不精确

实测各年份 `sp500_constituent` 的成分数：

```
1980-01-02  327      2010-06-30  507
1990-01-02  363      2015-06-30  516
2000-01-03  420      2020-06-30  521
2005-01-03  458      2026-09-01  503
```

标普 500 一直是 ~500 只（多几只是因为多股份类别）。但重建出来的历史值：

- **早期偏少**（1980 年只有 327）：维基变更日志并不完整，很多早年的纳入/剔除根本没记录，那些公司既不在 anchor 里也不在日志里，就完全不存在于表中。
- **中段偏多**（2010–2020 年 507–521）：日志漏掉的剔除会让区间开得太久（这正是第 1 类对账用 `end_date_is_inferred` 标记的那类，实测 18 条），加上按闭区间约定纳入/剔除同日会两边都算。

结论：**这张表消除了幸存者偏差的主要部分，但它不是一份精确到个位的官方历史成分名单。** 越靠近现在越准（右端有 anchor 校准），越往回越糙。做长历史回测时应当把它当成"高质量近似"，并留意 `end_date_is_inferred` 这一列——`reconstruct_intervals` 的 docstring 明确说：在意精确剔除日的消费者必须自己过滤这一列，而 `get_symbols_as_of` **故意不过滤**（一个有界的结束日仍然比一个开着的区间更接近真相）。

### 坑 3：`as_of` 不钉就不可复现

`ConstituentDatasetConfig.as_of` 默认 `None` = "今天"。只要区间表里存在开区间（`sp500_constituent` 实测 503 个），面板右边界就跟着墙上时钟走。同一份配置隔天再跑，出来的是**另一个形状**的 Zarr store，而 `save()` 是 `mode="w"` 直接覆盖。要复现就必须显式设 `as_of`（`tests/test_constituent_panel.py::test_as_of_pins_the_right_edge_making_the_panel_reproducible` 就是钉这条的）。

### 坑 4：日历日轴 vs 交易日轴

面板时间轴是连续的 `freq="D"`，带周末和节假日（值为上一交易日顺延）。下游和 OHLCV join **必须**显式 reindex 或 `.sel()`。`UniverseMask` 已经处理了（时间戳做 inner join），但如果你绕开 `UniverseMask` 手写 join，就得自己记住这件事。

### 坑 5：纳指 100 的 anchor 是这一层最脆弱的输入

维基百科的 `Nasdaq-100` 页用 navbox 模板渲染成分，没有可解析的表格，所以 anchor 是从**商业网站** `stockanalysis.com` 抓的。它挂了怎么办？`Nasdaq100MembershipFetcher` 的 docstring 记了备选：`https://www.slickcharts.com/nasdaq100`（同样 102 行，列为 `# / Company / Symbol / Weight / Price / Chg / % Chg`），做法是**切换 `ANCHOR_URL` 和 `fetch_anchor()` 的列处理，而不是两个都实现**。类上还有 `MIN_ANCHOR_ROWS = 50` 的形状守卫——一个结构漂移的商业页面必须大声失败，否则一个被静默截断的 anchor 会关闭所有没被提到的成分关系，把这一层存在的意义（消除幸存者偏差）原样还回去。

---

## 加一个新指数要改什么

按设计，**指数是数据不是代码**：

1. 在 `acquisition/universe.py` 加一个 `IndexMembershipFetcher` 子类：九个类常量（`ANCHOR_URL`、`CHANGES_URL`、`PIT_COVERAGE_START`、`CACHE_FILENAME`、`INDEX_LABEL`、`CATEGORY`、`EXPECTED_SOURCE_HEADER`、`DATE_HEADER`、可选 `CHANGES_TABLE_ATTRS`）+ 一个 `fetch_anchor()`。重建算法、变更日志解析、整套抓取/缓存安全阀都从基类白拿。
2. 把它注册进 `UniverseCatalog.MEMBERSHIP_FETCHERS`——覆盖边界守卫和 `known_categories()` 会自动跟上，不需要有人记得加 `if` 分支。
3. 在 `enums/data.py` 的 `UniverseCategory` Literal 里加上新 token（`CATEGORY` 标注的是 Literal 而不是 `str`，所以漏了会是类型错误）。
4. 在 `dataset/constituent.py` 加一个 `IndexConstituentDataset` 子类：只实现 `_pit_coverage_start()` 和 `_build_intervals()` 两个钩子。URL 和覆盖常量**不要**在这里重复一遍——这个类只是"指数"和"面板机器"之间的绑定。

`base/` 里不放任何和具体指数相关的东西，这就是 "加一个新指数不需要动上层" 在结构上成立、而不只是口头承诺的原因。
