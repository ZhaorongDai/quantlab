# WRDS TAQ NBBO：从逐笔最优报价到 bar 面板

> 代码位置：采集 `quantlab/acquisition/wrds/taq.py`（`WrdsSession`、`WrdsTaqNbboAcquisition`、
> `WrdsNbboVolumeProbe`），体量护栏 `quantlab/acquisition/sql_volume.py:SqlVolumeGuard`，
> 面板 `quantlab/dataset/nbbo.py:NbboPanelDataset`，重采样 `quantlab/dataset/nbbo_resample.py`，
> 交易日历 `quantlab/dataset/session_calendar.py:XnysSessionCalendar`，
> 命令行入口 `scripts/ingest_wrds_taq.py`。
> 相关文档：采集引擎通用契约见 [acquisition.md](acquisition.md)，数据源登记表见 [registry.md](registry.md)，
> 分块转换见 [chunking.md](chunking.md)，时点成分见 [constituent.md](constituent.md)。

---

## 一句话

从 WRDS 的 NYSE TAQ 毫秒库把**每一条全国最优买卖报价（NBBO）记录原样**拉到本地 parquet，
再在本地按你选的 bar 大小（`1s` 到 `30m`）重采样成标准的 `[timestamp, symbol]` Zarr 面板。
服务器端只做 `SELECT ... WHERE`，不做任何聚合：原始记录留在本地，以后换 bar 大小、换过滤规则
都不用再去 WRDS 拉一次。

一条命令跑完整条链路：

```bash
export WRDS_USERNAME=<你的 WRDS 用户名>     # 密码只放在 ~/.pgpass
uv run python scripts/ingest_wrds_taq.py --symbols AAPL,MSFT,BRK.B \
    --start-date 2024-01-24 --end-date 2024-01-25 --to-zarr --bar-interval 1m
```

---

## 不用它会怎样

这一层的每条规则背后，都是一个会让数据悄悄出错、或让人手机被 Duo 推送轰炸的具体故障。

### 1. 用 `nbbom` 表会漏掉 NBBO 状态

TAQ 毫秒库里有两张 NBBO 表。`nbbom_YYYYMMDD` 只记录由 NBBO 变动**本身**触发的那些行；
当最优价由某一家交易所单独报出、NBBO 状态只写在该交易所报价行上时，`nbbom` 里没有这一行。
拿它重建「某一时刻的最优价」会缺状态。所以本阶段只读
`taqm_{YYYY}.complete_nbbo_{YYYYMMDD}`：它把这些单交易所状态也并进来，每行就是那一刻完整的 NBBO（D-18）。

### 2. PostgreSQL 不保证行序，而 `ORDER BY` 会打乱同时间戳的记录

WRDS 那份 PROC SQL 说明讲的是 SAS：普通 `SELECT` 会保持物理顺序。PostgreSQL 没有这个保证，
不写 `ORDER BY` 时返回什么顺序都有可能。但加上 `ORDER BY time_m` 也不行：2018 年以前的表
只有微秒精度，约 5% 的行和**另一个不同的 NBBO 状态**共享同一个 (symbol, 微秒)，
`ORDER BY` 会任意重排这些并列行，「这根 bar 最后一个 NBBO」就变成随机的了。

做法（D-19）：用 `COPY (SELECT ... WHERE ...) TO STDOUT`，**不写** `ORDER BY` / `GROUP BY` / `DISTINCT`，
把每条记录在这次查询里的到达序号记成 `wrds_row_ord` 写进原始数据；本地排序用的全序键是
`(date, symbol, time_m, time_m_nano, wrds_row_ord)`，并且是稳定排序。

### 3. 半日市按 16:00 收盘

感恩节次日（如 2024-11-29）、平安夜等半日市 13:00 收盘。固定按 16:00 切 bar 的话，
13:00 之后的三小时会被当成交易时段，1 分钟面板里每个标的多出 180 根 carry-forward 出来的假 bar。
现在的会话边界来自 `exchange_calendars` 的 XNYS 日历（D-23），半日市、夏令时切换、
非交易日都按交易所实际日程处理。

### 4. Duo 推送风暴

每开一条新的 WRDS 连接，账号持有人的手机就可能收到一次 Duo 推送，而且 WRDS 角色最多允许 7 条连接。
按批次开连接、并发 worker、失败后自动重连，每一种都会变成几十次推送。所以：一次运行只有**一个**
`WrdsSession`，体量探测、拉取和转换共用它；`max_workers` 固定为 1（设成别的值会被拒绝）；
会话一旦断开就不再重连，这次运行停下，下次重跑从已记录的页续上。

### 5. `wrds.Connection` 会弹交互式密码提示，而且在 pandas 3 下坏掉

官方 `wrds` 库的 `Connection` 在找不到密码时会**交互式地**问密码，还会主动提出帮你创建 `.pgpass`；
它的 `raw_sql` 在 pandas 3 下也不能用。无人值守的脚本卡在一个密码提示上，或者把密码敲进终端，
这两种情况都不能接受。现在直接用 `psycopg2` 连固定的主机，密码只由 libpq 从 `~/.pgpass` 读取，
连接前先检查这个文件存在、权限是 600、并且有匹配的一行，缺哪样就**立刻报错并说明怎么修**（D-20）。

### 6. 一个没订阅的年份把每一批都记成失败

如果账号没有某一年的 TAQ 订阅（实测这个账号没有 2012），不提前检查的话，
那一年的每一天、每一批都会失败一次，失败清单会把几百个完全正常的股票记成「失败」。
现在在任何 `count(*)` 和 COPY 之前，先对窗口里每一年查一次 `taqm_YYYY` 的 USAGE 权限；
没有权限就整次运行立即停下，错误信息里点名 `taqm_YYYY`（D-21）。

---

## 核心概念

### `complete_nbbo` 与 `nbbom`

| 表 | 每行是什么 | 本阶段 |
|---|---|---|
| `taqm_YYYY.complete_nbbo_YYYYMMDD` | 那一刻完整的 NBBO，包括单交易所报出的最优价状态 | **使用** |
| `taqm_YYYY.nbbom_YYYYMMDD` | 只有 NBBO 变动本身触发的记录 | 不用（会漏状态） |
| `taqmsec.*` | 上面这些表的视图 | 不用（同一份数据） |

### 原始列，以及为什么 `date` 存成 `taq_date`

每个原始分片的列（`WrdsTaqNbboAcquisition.RAW_COLUMNS`）：

```
timestamp, symbol, vendor, taq_date, time_m, time_m_nano, sym_root, sym_suffix,
qu_cond, natbbo_ind, qu_source, nbbo_qu_cond,
best_bid, best_bidsizeshares, best_ask, best_asksizeshares, wrds_row_ord
```

- `timestamp` 是 naive UTC 的 `Datetime("ns")`（和代码库里其他时间戳一样），由 ET 的 `date + time_m (+ time_m_nano)` 换算而来。
- TAQ 的 `date` 列改名为 `taq_date`：写分片时会从时间戳派生一个 `date=` hive 键，
  同名的原始列会先被覆盖再被丢掉。
- `time_m_nano` 从 **2018-01-02** 起才有；更早的表里这一列是类型为 Int16 的空值，
  所以 2016 年和 2024 年的分片 schema 完全相同。
- 目录布局：`<数据根>/downloads/us_equity/tick/wrds_taq/wrds/data_type=nbbo/date=YYYY-MM-DD/symbol=XXX/*.pqt`，
  `date=` 是美东交易日。水位线在同级的 `_watermarks/wrds/`。

### `wrds_row_ord` 与全序键（D-19）

`wrds_row_ord` 是一条记录在它那次 (交易日, symbol 批次) COPY 查询里的到达序号。
本地全序键是 `(date, symbol, time_m, time_m_nano, wrds_row_ord)`：

- **2018 年起**：`(symbol, time_m, time_m_nano)` 实测唯一（2024-01-24 的 AAPL/MSFT/BRK，
  2,065,354 行里有 2,065,354 个不同的键），顺序完全确定，`wrds_row_ord` 用不上。
- **2018 年以前**：只有微秒。2016-12-07 抽 5 个名字：1,999,275 行只有 1,897,424 个不同的时间戳，
  其中完全相同的行只有 786 行，也就是说大多数并列行是**不同的** NBBO 状态。表里没有序列号字段，
  唯一的依据就是物理到达顺序，也就是 `wrds_row_ord`。

### `n_ambiguous_ties`，以及 2018-01-02 这条分界线

面板里每根 bar 带一个 `n_ambiguous_ties`：这根 bar 里有多少条记录和一个**取值不同**的记录共享同一个时间戳
（过滤之后、合并并列行之前计数）。2018-01-02 以后它几乎总是 0；更早的数据里它非零的 bar，
「最后一个 NBBO」取决于 WRDS 返回的物理顺序。对这类 bar 敏感的研究可以按它过滤。

### 右闭 bar、以 bar 结束时刻为标签（D-22）

会话开盘 `o`、bar 长 `d`，标签是 `o + k*d`（`k = 1..N`），第 `k` 根 bar 是区间 `(o+(k-1)d, o+kd]` 的状态。
`09:31` 这根 1 分钟 bar 描述的是 `(09:30, 09:31]`，恰好落在 09:31:00 的记录属于它。
标签就是信息可得的时刻，所以用标签对齐其他数据不会有前视。

### 会话窗口（D-09、D-29）

- 默认是常规交易时段 **09:30–16:00 ET**。
- 可以用 `--session-start` / `--session-end` 设在 **04:00–20:00 ET** 内的任意位置；
  超出这个范围，或开始不早于结束，都会在建任何连接之前被拒绝。
- 半日市规则，两个边分别判断：落在常规时段 [09:30, 16:00] 内的边会被截到当天交易所实际的开/收盘；
  落在盘前盘后的边按挂钟时间保持不变。2024-11-29（13:00 早收）的例子：

| 窗口 | 当天实际窗口 |
|---|---|
| 09:30–16:00 | 09:30–13:00 ET（14:30Z–18:00Z），210 根 1m bar |
| 10:00–15:30 | 10:00–13:00 |
| 13:30–16:00 | 空窗口，当天不产生 bar |
| 04:00–20:00 | 04:00–20:00 不变（09:00Z 到次日 01:00Z） |
| 09:30–17:00 | 09:30–17:00（17:00 是盘后的边） |
| 07:00–16:00 | 07:00–13:00（16:00 是常规时段的边） |

跨过早收的窗口是连续的：13:00 之后的 bar 带的是收盘后的报价状态。
盘后收盘（如 20:00 ET）换成 UTC 会落到下一个日历日，这是正常的。

### 只在一天之内 carry-forward

没有更新的 bar 沿用上一个 NBBO，`n_updates = 0`：NBBO 在被替换之前一直有效，所以这是观测到的状态，不是编造的数据。
每个窗口的起点用开盘时刻（含）之前最后一条有效 NBBO 作种子。**状态绝不跨交易日**；
当天第一条有效 NBBO 之前的 bar 是 NaN。一个 symbol 某天完全没有记录，整天所有变量都是 NaN。

### 单边为空 = NaN

实测 2024-01-24 的 298 万行里有 202 行一侧或两侧为空（常规时段内 158 行）。空的一侧表示那一侧没有报价，
价格和数量是 NaN，**不是 0**；`mid`、`spread`、`spread_bps`、`imbalance` 在缺一侧时也是 NaN。
写成 0 的话，时间加权平均会被拉向 0，看起来像一个窄得离谱的真实报价。

### 默认过滤与 filter-stats 旁车文件（D-10）

过滤只在重采样时做，原始数据里全部保留。默认值在 `NbboDatasetConfig` 上：

| 字段 | 默认 | 含义 |
|---|---|---|
| `drop_nonpositive_price` | `True` | 丢掉存在的买价或卖价 `<= 0` 的记录 |
| `keep_qu_cond` | `None` | 设了就只保留这些 `qu_cond`；`None` 全保留 |
| `drop_crossed` | `True` | 丢掉 `bid > ask`（两侧都在时才判断） |
| `drop_locked` | `False` | `bid == ask`；实测占 4.4%，是合法状态，默认保留 |

每条被丢的记录只按第一个命中的原因计一次（nonpositive_price > condition > crossed > locked）。
计数按 (交易日, symbol) 写进 Zarr 旁边的 `<store>.nbbo_filter_stats.json`（和 store 同级，不在 store 里面，
也不在原始数据目录下），里面有 `config`（bar 大小、会话窗口、过滤策略）、`by_session[date][symbol]` 和 `totals`。
重新转换某一天会**替换**那一天的条目并重算合计，不会重复计数。命令行跑完会打印这个文件的路径。

### `BarInterval`：为什么只有这几个

`--bar-interval` 可选 `1s, 5s, 10s, 15s, 30s, 1m, 5m, 10m, 15m, 30m`（`quantlab/enums/data.py:BarInterval`）。
每一个都能同时整除常规交易日的 390 分钟和半日市的 210 分钟，所以任何一天都不会有一根 bar 跨过收盘。
`1h` 不在里面：390 不是 60 的整数倍。注意它是**面板**的 bar 大小，和采集频率 `frequency="tick"` 是两回事（D-26），
共享的 `Frequency` 字面量没有扩展。

### 面板变量

`bid, ask, bid_size, ask_size, mid, spread, spread_bps, imbalance, n_updates, tw_spread, tw_bid_size, tw_ask_size, n_ambiguous_ties`，
全部是 float64，维度 `(timestamp, symbol)`。快照类变量取标签时刻（含）之前的最后一条记录；
`tw_*` 是 bar 内的时间加权平均，每条记录的权重是它到下一条记录的时长，同一时间戳的并列记录只有最后一条有权重。
`imbalance = (bid_size - ask_size) / (bid_size + ask_size)`。

---

## 它是怎么工作的

```
scripts/ingest_wrds_taq.py
  │  参数校验：--symbols/--universe 二选一、两个日期都必填、点号记法、
  │  会话窗口（XnysSessionCalendar 构造时检查）——全部在建连接之前
  ▼
UniverseCatalog.get_symbols_in_range      （仅 --universe；区间重叠、点号记法）
  ▼
SOURCE.config_factory(...)                （= WrdsTaqNbboAcquisition.build_config，D-27）
  ▼
WrdsSession.shared()  ── 一次运行一个会话 = 最多一次 Duo 推送 ──────────────┐
  ▼                                                                       │
WrdsNbboVolumeProbe.count_rows_by_day                                     │
  │  1. assert_entitled：窗口内每一年的 taqm_YYYY 都要有 USAGE 权限           │
  │  2. 每个 (交易日, symbol 批次) 一次 count(*)，WHERE 与 COPY 完全相同         │
  ▼                                                                       │
SqlVolumeGuard.assert_acquisition_volume_fits   （超限拒绝，零 COPY）        │
  ▼                                                                       │
registry.run(SOURCE, ...)                                                 │
  │  每个 (交易日, 批次) 一次 COPY；COPY 前再 count(*) 一次核对行数             │
  ▼                                                                       │
原始分片 downloads/us_equity/tick/wrds_taq/wrds/data_type=nbbo/...          │
  ▼   （仅 --to-zarr）                                                     │
registry.convert(SOURCE, NbboDatasetConfig, data_type="nbbo")             │
  ▼                                                                       │
NbboPanelDataset → NbboResampler（过滤 → 排序 → 种子 → 右闭 bar）             │
  ▼                                                                       │
data/us_equity/tick/wrds_nbbo_{bar}_{开始}-{结束}.zarr                       │
  + .nbbo_filter_stats.json 旁车文件                                        │
                                                        finally: close_shared()
```

几个要点：

- **拒绝都发生在拉数据之前。** 参数错误和会话窗口错误在连接之前；没订阅的年份在任何 `count(*)` 之前；
  体量超限在任何 COPY 之前。
- **一次运行只有一个会话。** 脚本里只调一次 `WrdsSession.shared()`，采集类构造时拿到的也是同一个实例；
  `finally` 里 `close_shared()` 关掉它。
- **续跑粒度是「一天一页」。** 某天某批失败，下次重跑从那一天接着拉；会话中途断掉算全局停止，不写失败清单。
- **store 名字里带着会话窗口**，例如 `wrds_nbbo_1m_0930-1600.zarr`。分块转换的台账只对 symbol 轴做指纹，
  如果两个不同窗口共用一个 store，第二个窗口已经「写过」的日期会被静默跳过。

---

## 凭证

- 用户名：环境变量 `WRDS_USERNAME`。
- 密码：**只**放在 `~/.pgpass`（或 `$PGPASSFILE` 指向的文件），权限必须是 600，其中一行形如：

  ```
  wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>
  ```

  ```bash
  chmod 600 ~/.pgpass
  ```

- 密码从来不进本仓库的代码：连接时由 libpq 自己读。检查 `.pgpass` 时只解析前 4 个字段，
  密码字段不会被读进任何变量，报错信息里用户名写成 `$WRDS_USERNAME`，不打印它的值。
- 设置了 `PGHOSTADDR`、`PGSERVICE` 或 `PGSERVICEFILE` 会被拒绝：它们能把连接或连接参数引到固定主机以外的地方。
  `PGHOST` 无所谓，因为主机总是显式传入。
- 命令行**没有**任何用户名或密码参数（有结构测试锁着）。**不要**把密码敲进任何工具或命令行，
  它会留在 shell 历史和进程列表里；本仓库已经因为硬编码 Tiingo key 真实泄露过一次。

---

## 体量

2024-01-24 的实测（`03.9-LIVE-CHECK-*.json`，D-24）：

| 标的 | 当天 `complete_nbbo` 行数 |
|---|---|
| AAPL | 1,243,426 |
| XOM | 760,575 |
| MSFT | 669,127 |
| JNJ | 157,359 |
| BRK | 152,801 |
| **全市场** | **313,568,856 行，9,703 个 root** |

- 护栏按 `count(*)` 计出的真实行数定价，默认上限 **20 GiB 原始字节**（和 `UniverseCatalog.MAX_RAW_BYTES` 相同）
  和 **7 亿行**。
- 每行字节数 `bytes_per_row = 30` 是**假设值**，不是量出来的；在实测分片大小之前，打印的估算会标明 `ASSUMPTION`。
- 按这个量级，20 GiB 只够 S&P 500 的**几天**。长区间要分成多个日期段，每段单独过护栏。
  被拒绝时，报错会给出从 `--start-date` 起能放得下的最长一段，例如 `--start-date 2024-01-24 --end-date 2024-01-26`，
  照着跑完再跑下一段即可。
- `--force-volume` 跳过拒绝，但估算照算照打印，并多打一行说明哪条上限被越过了。
  没有环境变量、也没有配置项能整体关掉护栏。
- 按单个 symbol 过滤的 `count(*)` 在服务器上不到 1 秒；整张表的 `count(*)` 要约 76 秒，所以只按批次计数，
  从不数整张表。

---

## 完整例子

### 例 1：离线真跑过的参数拒绝（不连接 WRDS）

这几条在建任何连接之前就被拒绝，以下是实际输出：

```bash
$ uv run python scripts/ingest_wrds_taq.py --symbols AAPL,BRK-B \
      --start-date 2024-01-24 --end-date 2024-01-25
ingest_wrds_taq.py: error: --symbols ['BRK-B'] use a hyphen; WRDS TAQ uses dot notation (e.g. BRK.B for root BRK, suffix B).

$ uv run python scripts/ingest_wrds_taq.py --symbols AAPL \
      --start-date 2024-01-24 --end-date 2024-01-25 --to-zarr --session-start 03:59
ingest_wrds_taq.py: error: --session-start/--session-end: session_start 03:59:00 lies outside the extended window 04:00-20:00 ET

$ env -u WRDS_USERNAME uv run python scripts/ingest_wrds_taq.py --symbols AAPL \
      --start-date 2024-01-24 --end-date 2024-01-25
WRDS_USERNAME environment variable must be set to your WRDS username. The password is never read from config or from this code: libpq reads it from ~/.pgpass (chmod 600), so store it there before running a WRDS acquisition.
```

整条链路（探测、护栏、拉取、转换、单连接）在 `tests/test_ingest_wrds_taq.py` 里用离线的 `FakeWrdsSession` 端到端跑过。

### 例 2：真实拉取（此例未实际运行，需要 WRDS 凭证，每次运行可能触发一次 Duo 推送）

```bash
export WRDS_USERNAME=<你的 WRDS 用户名>

# 只拉原始记录，停在 raw
uv run python scripts/ingest_wrds_taq.py --symbols AAPL,MSFT,BRK.B \
    --start-date 2024-01-24 --end-date 2024-01-25

# 拉完重采样成 1 分钟 bar（默认常规时段、按天分块）
uv run python scripts/ingest_wrds_taq.py --symbols AAPL,MSFT,BRK.B \
    --start-date 2024-01-24 --end-date 2024-01-25 --to-zarr --bar-interval 1m

# 盘前盘后全时段、30 分钟 bar（每天 32 根）
uv run python scripts/ingest_wrds_taq.py --symbols AAPL \
    --start-date 2024-01-24 --end-date 2024-01-24 --to-zarr \
    --session-start 04:00 --session-end 20:00 --bar-interval 30m

# 时点 S&P 500 成分（窗口内任意时刻是成分的都算），分段跑
uv run python scripts/ingest_wrds_taq.py --universe sp500 \
    --start-date 2024-01-24 --end-date 2024-01-24 --to-zarr
```

命令行每次都会先探测、再拉取。原始数据已经在盘上、只想换 bar 大小或会话窗口重新转换时，
不需要连 WRDS：在 Python 里直接构造 `NbboDatasetConfig`（`raw_data_dir_path` 指向上面的原始目录）
并调用 `quantlab.acquisition.registry.convert(DataSourceRegistry.get("wrds"), cfg, data_type="nbbo", granularity="day")`。

---

## 常见坑

- **用连字符写 symbol。** `BRK-B` 是 Tiingo 花名册的写法，WRDS 要写 `BRK.B`（root `BRK`、suffix `B`；`BF.B` 同理）。
  脚本会直接拒绝，不会猜。
- **`--universe` 只有 `sp500` 和 `nasdaq100`。** `nasdaq_all` / `us_all` 是交易所全量名单，用连字符记法，
  而且对逐笔报价来说大得离谱。
- **两个日期都必填。** 窗口要先数行数、过护栏，没有默认窗口。
- **`--rows-per-symbol-day` 在这里没有意义。** 它是 Alpaca tick 护栏用的；WRDS 的行数在服务器上用 `count(*)` 数出来，传了会报错。
- **2018 年以前的数据看 `n_ambiguous_ties`。** 非零的 bar，快照取决于 WRDS 的物理返回顺序。
- **盘后收盘落在下一个 UTC 日。** 04:00–20:00 窗口的最后一根 bar 标签是次日 01:00Z（冬令时），这是对的。
- **想要不含早收后状态的数据，用常规时段内的结束边。** 例如 `--session-end 16:00` 在半日市会自动截到 13:00，
  而 `--session-end 17:00` 不会。
- **没订阅的年份整次停下。** 报错点名 `taqm_YYYY`；把窗口改到有订阅的年份再跑。
- **会话断了不会自动重连。** 这是有意的（每次重连都可能推送 Duo）；重跑即可从断点续上。
  `wait_for_quota` 对 WRDS 没有意义，命令行也没有提供。
- **30 B/行是假设。** 实测分片大小之前，护栏的字节估算可能偏大或偏小；行数上限（7 亿）是独立的第二道线。
- **亚分钟 bar 用 `--chunk day`（默认值）。** 1 秒 bar 的一整天 S&P 500 约 1170 万个 bar 行。
