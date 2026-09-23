# WRDS CRSP Stock v2 日频：从 PERMNO 原始行到可直接替换 Tiingo 的面板

> 代码位置：采集 `quantlab/acquisition/wrds/crsp.py`（`CrspQueries`、`WrdsCrspDailyAcquisition`、
> `CrspVolumeProbe`、`CrspProductEndError`、`CrspVintageError`），参考表采集
> `quantlab/acquisition/wrds/crsp_reference.py:CrspReferenceTables`，
> 数据源描述符 `quantlab/acquisition/wrds/__init__.py`，体量护栏 `quantlab/acquisition/_support/sql_volume.py:SqlVolumeGuard`，
> 参考表读取 `quantlab/dataset/crsp/reference.py`，
> PERMNO → ticker 区间表 `quantlab/dataset/crsp/symbology.py:CrspSymbology`
> （**它现在只喂 ticker 旁车，不再决定面板的列叫什么**）与旁车读侧
> `quantlab/dataset/crsp/tickers.py:CrspTickerLookup`，
> 面板 `quantlab/dataset/crsp/__init__.py:CrspStockDataset`，成分 `quantlab/dataset/crsp/membership.py:CrspMembership`
> 与 `quantlab/dataset/constituent.py`（`CrspSP500ConstituentDataset`、`CompustatNasdaq100ConstituentDataset`），
> 命令行入口 `scripts/ingest_wrds_crsp.py`。
> 相关文档：采集引擎通用契约见 [acquisition.md](acquisition.md)，数据源登记表见 [registry.md](registry.md)，
> 同厂商的逐笔报价路径见 [wrds_taq.md](wrds_taq.md)，面板形态见 [dataset.md](dataset.md)，
> 时点成分见 [constituent.md](constituent.md)。

---

## 一句话

从 WRDS 的 **CRSP US Stock Database Version 2（CIZ 格式，annual update 产品）** 把日频证券数据
按 **PERMNO** 原样拉到本地 parquet，再在本地把它转换成和 `StockDataset` 变量完全一致的
`[timestamp, symbol]` Zarr 面板——**同一套因子、标签、模型和回测代码不改一行就能换厂商**。
**这个面板的 `symbol` 轴就是 int64 的 PERMNO 本身**（D-01，phase 03.11），不是 ticker：
名字不进面板，而是写在旁车 `{zarr}.crsp_tickers.json` 里，人要读的时候按日期查。
代价是一整套 CRSP 特有的数据语义（退市收益、总收益复权、股份类别、年度产品边界），
这篇文档就是把每一条语义写清楚。

一条命令跑完整条链路：

```bash
export WRDS_USERNAME=<你的 WRDS 用户名>     # 密码只放在 ~/.pgpass
uv run python scripts/ingest_wrds_crsp.py --universe crsp_sp500 \
    --start-date 2024-01-01 --end-date 2024-12-31 --to-zarr
```

---

## 不用它会怎样

这一层的每条规则背后，都是一个会让数据**安静地出错**（不是报错，是给你一个干净、可信、完全错误的数字）的具体故障。

### 1. 缺了退市收益 = 幸存者偏差

一只股票退市那天的 -60% 如果不在收益序列里，回测里它就是「某天之后没有数据了」，
和「这只票涨到停牌」长得一模一样。组合会安静地在退市前一天把它卖在一个好价钱上。

CIZ 的做法是：退市收益**本身就是一条日行**（雷曼 PERMNO 80599，2008-09-18，
`dlydelflg='Y'`、`dlyprcflg='DP'`、`dlyret=-0.6`）。所以本模块从不把 `stkdelists.delret`
再加一次——`stkdelists` 只作为事件原始数据保存。面板上 `is_delisting` 只是一个**标记**，
没有任何地方乘以它。雷曼那三天的连乘 `(1+0.428571)(1-0.566667)(1-0.6)` 在测试里被钉死。

**但退市行有两种形态，而且现代 CIZ 写的是第二种。** 这一点不是细节，
是 2026-09-20 两个已修缺口（GAP-A / GAP-B）的全部成因：

| `dlyprcflg` | 含义 | `dlyprc` | 同一行的 `dlyclose` / `dlyvol` / `dlycumfacpr` / `dlycumfacshr` |
|---|---|---|---|
| `DP` | 退市**价格**，一个真实价格 | 雷曼 2008-09-18 是 `0.052` | 正常有值 |
| `DA` | 退市**清算金额**，**不是市场价格** | 一律 `0.000000`，这是**「无价格」哨兵值** | **全部为 NULL** |

本项目实际拉下来的那一档原始数据里，退市行的 flag 分布是 `TR` 138,888 / **`DA` 5** / `DP` 0
——`DA` 是 **5 比 5**，雷曼那种 `DP` 形态是 **0 比 5**。

面板对 `DA` 行的处理，逐条：

- 原始 `close` 在这条行上是 **NaN**，不是一笔 $0.00 的成交。`abs(0.0)` 仍然是 `0.0`，
  所以哨兵值必须在 `abs()` **之前**就被排除掉（`quantlab/dataset/crsp/__init__.py` 的
  `_NO_PRICE_FLAGS`）；否则面板会一边写着 `ret = -0.56%`，一边写着一笔 -100% 的成交。
- 这条行**永远不可能成为复权锚点**。原来的判据只是「一个非空收盘」（03.12 换向之后是
  「**第一个非空收盘**」，漏洞的形状一模一样），而 `0.0` 不是空值，
  于是 `dlyprcflg='DA'` 的哨兵行成了锚点，`adjClose = 0.0 × G_t / G_anchor`
  在这只证券**整段历史**上恒为 0.0。
- `ret` 照旧只带这笔退市收益**一次**，累计连乘 `G` 也照旧包含它。这次修复**没有动收益链**，
  动的只是这条链被锚定在哪个**水平**上（WestRock：从整段历史恒为 0.0，变回由它
  **第一个**真实收盘定下的水平——在 `tests/test_crsp_dataset.py` 那个 2024-07 窗口里，
  锚是 2024-07-03 的 49.75）。
- 一个 PERMNO 在窗口内**没有任何一行**同时带着正的 `dlyprc` 和非空的 `dlycumfacshr` 时，
  转换会**点名拒绝**（`ValueError` 里带上 PERMNO、配置的起止日期和所需谓词），
  而不是发布一列全 0 的 `adjClose` 或一列全 NaN 的 `adjVolume`。
  拒绝比给一列零或一列 NaN 更好，理由很直接：`factor/alpha158.py` 和 `label/fret.py`
  只读那五个 `adj*`，一列 0.0 会变成 `0/0 → NaN`、`x/0 → inf` 和一个天天垫底的横截面排名，
  一列 NaN 会让任何流动性筛选安静地剔掉每一只退过市的证券——
  两者都不报错，而且两者中招的都正好是退市的那些票。

所以这一节的幸存者偏差论证仍然成立，但要把话说准：**原始 `close`、`ret` 和 `is_delisting`
一直是对的，`adj*` 那五列在 2026-09-20 之前不是**。上面这些规则是把保证收窄到代码真正
交付的范围（GAP-A / GAP-B 的详细复盘见 `03.10-REVIEW.md` CR-01 / CR-02）。

那条退市行的 **ticker 是 NULL**，它的 security-info 区间也是 NULL ticker。
**这件事曾经有一半是致命的，现在只剩另一半。**

- **死掉的那一半：「没有名字就进不了面板」。** 在 ticker 轴上，这条唯一带着 -60% 的行会因为
  贴不上标签而被丢掉——幸存者偏差刚被 CIZ 修好，又被我们自己的标签规则放了回来。
  PERMNO 轴上这个问题**不存在**：这条行的键是 80599，有没有名字都在。
  （替换掉的那条「退市 symbol carry」规则连同符号学的另外三个机制一起，
  在 phase 03.11-07 被**删除**而不是留作守卫——见下文「原始层按 PERMNO」一节。
  它唯一剩下的作用是让旁车能给一只死掉的证券的最后一天写上名字。）
- **还活着的那一半：类型列也是空的。** 退市行的 `sharetype` / `securitytype` /
  `securitysubtype` 同样为 NULL，于是证券过滤会判它「类型未知 → 丢」。
  所以 `quantlab/dataset/crsp/__init__.py` 里有一条**判决继承**规则：`dlydelflg='Y'` 的行
  继承前一天的过滤判决。这一条**必须留着**，它就是这一节的反幸存者偏差论证本身。
  两条规则在代码里曾经挨着写，这正是第二条容易被连坐删掉的原因。

### 2. 一个代码被两家公司先后用过 → 凭空捏出一个跨公司收益（**已从根上消掉**）

ticker 是会被回收的。如果面板的 `symbol` 轴上「ABC」这一列前半段是 A 公司、后半段是 B 公司，
那么交接那一天的「收益」= B 公司的首日价 / A 公司的末日价——一个两家公司之间的比值，
**在数学上完全合法，在金融上毫无意义**，而且没有任何迹象表明它不对。
更糟的是两个 PERMNO 在**同一天**顶着同一个 symbol：一个格子里塞了两条记录，
继承来的 `dedup_raw_frame(keep="last")` 会把两家公司压成一条价格序列，不留痕迹。

**这两种故障在 PERMNO 轴上都不是「被防住了」，而是不可拼写。** 一列换东家需要
`symbol` 列能改指一家公司，而 PERMNO 列永远不会；同日撞车需要两个 PERMNO 落进同一个
`(date, symbol)` 格子，而原始层本身就断言 `(permno, dlycaldt)` 唯一
（`quantlab/acquisition/wrds/crsp.py`）。所以 phase 03.11-07 把为这两件事而生的机制
**整套删掉**，而不是留成永远只会说「没发生」的守卫：

| 曾经的机制 | 它防的是什么 | 现状 |
|---|---|---|
| PERMNO seam：接手方第一行把五个 `adj*` 置 NaN | 代码回收造出的跨公司收益 | **删除**（连同它的 opt-out 开关） |
| 同日撞车拒绝：活的优先于退市的、成分优先于非成分，分不开就 `ValueError` | 一个格子两条记录 | **删除** |
| 类别撞车 pass：重叠区间上重名的两个 PERMNO 各自重拼 `base.cls` | 伯克希尔 A/B 撞同一列 | **删除**（见下一节） |
| 退市 symbol carry | 没名字的退市行进不了面板 | **删除**（判决继承是另一条规则，仍在，见上一节） |

改名（FB → META，PERMNO 全程 13407）在 PERMNO 轴上**本来就是一列**，不需要任何规则去
「区分它和代码回收」——那个区分曾经是 seam 判据存在的全部理由。
两个名字都没丢：它们是旁车 `{zarr}.crsp_tickers.json` 里 13407 名下的两条区间。

### 3. `BRK.B`：类别后缀现在只是**拼法**，不再是身份

`stksecurityinfohist.ticker` 存的是**词根**（`BRK`、`BF`），A 类和 B 类**两条线拼出来一模一样**。
在 ticker 轴上这是致命的：伯克希尔两个 PERMNO 会撞在同一列上，两家（严格说两个类别）的价格
被压成一条序列。所以当时有一整套重拼规则，包括一条**类别撞车 pass**——
重叠区间上重名的两个 PERMNO 各自按自己的 `shareclass` 重拼成 `base.cls`。

**PERMNO 轴上撞不了。** `BRK.A` 是 17778，`BRK.B` 是 83443，它们在面板上是两列整数，
不管 CRSP 把它们的 ticker 词根写成什么。于是「把两个类别分开」这件事不再需要任何规则，
那条撞车 pass 在 phase 03.11-07 被删掉了。

类别后缀**仍然存在**，但只活在 ticker 的**拼写**里，也就是旁车里那个给人读的名字。
`CrspSymbology` 现在只剩四条规则，而且它的唯一下游是旁车：

1. `base = ticker.strip().upper()`；
2. `cls = shareclass`（为空/`None` 时视为没有类别）；
3. `cls` 有值且 `tradingsymbol == base + cls` 时拼成 `base.cls`（`BRK` + `BRKB` + `B` → `BRK.B`），
   否则就是 `base`（GOOGL、META、FB）；
4. ticker 为空的区间继承该 PERMNO 上一段的拼写——雷曼（80599）2008-09-18 那条退市区间就是这一条。
   在 PERMNO 轴上它**不再决定那一行在不在面板里**（行的键是 80599，怎样都在），
   只决定旁车能不能给一只死掉的证券的最后一天写上名字。

**股票池和价格仍然用同一个 `CrspSymbology` 实例**，所以两边对 `BRK.B` 的拼法按构造一致；
而且现在连这一致性都不承重了——面板和成分面板都按 PERMNO 对齐，拼法只影响人看到的字符串。
待办事项 `2026-09-07-no-ticker-rename-mapping-...` 记的那个 89/876 缺口，在这个厂商身上不存在。

### 4. 两个 CRSP 年度版本混进同一个原始目录

`crsp_a_stock` 是**年度更新**产品。每年刷新时 CRSP 会**修订历史**。如果今年拉了一半、
明年接着拉另一半，同一个原始目录里就会有两个版本的数据，而且没有任何东西会告诉你。

做法：第一次运行时在 `.../{subdir}/_vintage/wrds.json` 打一个
`{"product_end": "..."}` 的戳——它是原始根和水位线根**两者的兄弟目录**，不在任何一个里面
（水位线目录里多一个 `*.json` 会被 `CoverageLedger` 读成一个幽灵 PERMNO；原始根里多一个
非 parquet 文件会被 `StockDataset._scan_raw` 走到）。之后任何一次探到不同版本的运行，
**在任何 COPY 之前**被拒绝，报错同时点名两个版本和两条出路（换一个 `subdir`，
或者把原始根连同它的 `_watermarks/wrds` 和 `_vintage` 兄弟一起删掉）。

同样的逻辑也管着复权锚点，见下文「复权」。

### 5. 请求 2026 年的数据，安静地返回空

年度产品的最后一天（当前版本 **2025-12-31**）是一条**硬边界**，不是「数据还没到」。
普通的 `WHERE dlycaldt BETWEEN ...` 对着它只会返回零行，而零行和「这些票那段时间没交易」
长得一样。

所以每次运行先探一次 `max(dlycaldt)`：`--end-date` 超过它会被**裁剪**，而且裁剪这件事
逐字打印出来——

```
clipped end 2026-06-30 -> 2025-12-31 (crsp_a_stock annual product end)
```

`--start-date` 超过它则**直接拒绝**并点名产品末日，因为整段窗口都在边界之外，裁剪救不了。
这一次探测的结果同时喂给窗口算术和上面的版本检查，一次运行只探一次。

### 6. 用 `wrds_dsfv2_query` 会多出重复行

WRDS 提供了一张预连接的宽视图 `wrds_dsfv2_query`（98 列，带分派和市场收益）。
实测 2020 年它比 `dsf_v2` 多 **360** 行——分派日一天多条记录时它会把日行复制一遍。
把重复行喂进连乘复权，那一天的收益就被算了两次。

`dsf_v2` 在 `(permno, dlycaldt)` 上**唯一**（2020 年 1,950,357 行 = 1,950,357 个不同键），
所以本阶段用它。即便如此，**每一页都会重新断言一次唯一性**——因为万一错了，
后果在下游完全看不见。

### 7. Duo 推送风暴

每开一条新的 WRDS 连接，账号持有人的手机就可能收到一次 Duo 推送。权限探测、产品末日探测、
参考表拉取、体量探测、日频拉取和三次转换如果各开一条连接，一次运行就是六七次推送。
所以一次运行只有**一个** `WrdsSession.shared()`，`finally` 里关掉；没有提高 worker 数的开关。

---

## 核心概念

### 为什么是 `dsf_v2`

| 表 | 每行是什么 | 本阶段 |
|---|---|---|
| `crsp_a_stock.dsf_v2` | 一个 PERMNO 的一天，带当日 `sharetype`/`securitytype`/`shrout`/`dlycumfacpr` 等（`ticker` 也在，但面板不读它——名字走旁车） | **使用**（`(permno, dlycaldt)` 唯一，这条唯一性正是同日撞车不可拼写的原因） |
| `crsp_a_stock.stkdlysecuritydata` | 同上但不含每日类型/身份列 | 不用（还要自己连 security-info） |
| `crsp_a_stock.wrds_dsfv2_query` | 预连接宽视图，98 列 | 不用（2020 年有 360 条重复） |
| `crsp.*` | 上面这些表的同名视图 | 不用（同一份数据） |
| `crsp_m_*` / `crsp_q_*` | 月度/季度更新产品 | **本账号未订阅** |

一行代码的回退开关：`WrdsCrspDailyAcquisition.DAILY_TABLE`。

### 原始层按 PERMNO，面板也按 PERMNO，ticker 只进旁车

这是整条链路里最重要的一条分界线，而且 phase 03.11 之后它**贯通了**：
原始分片的 `symbol` 列是 **PERMNO 字符串**（`"14593"`），面板的 `symbol` 轴是
**int64 PERMNO**（`14593`），两端同一个身份，中间不再有一次 ticker 转译。
目录是 `<数据根>/downloads/us_equity/1d/wrds_crsp/wrds/month=YYYY-MM/`，水位线在
`.../_watermarks/wrds/`，版本戳在 `.../_vintage/wrds.json`，参考表在 `.../_reference/`
（三者都是原始根的**兄弟**，不在它下面）。

所以一次改名（FB → META，同一个 PERMNO 13407）**不碰任何水位线、任何分片路径、任何续跑点**，
**也不碰面板的任何一列**：13407 从头到尾是同一列。

**名字去哪了。** `registry.convert` 仍然从 `stksecurityinfohist` 派生 period-correct ticker，
但它不再产生任何面板上的东西——它写成一份旁车 `{zarr}.crsp_tickers.json`，形态是**区间表**：

```json
{
  "generated_from": "stksecurityinfohist",
  "vintage_product_end": "2025-12-31",
  "intervals": {
    "13407": [
      {"ticker": "FB",   "start": "2012-05-18", "end": "2022-06-08"},
      {"ticker": "META", "start": "2022-06-09", "end": "2025-12-31"}
    ]
  }
}
```

区间而不是「每个 PERMNO 的最后一个 ticker」，是因为后者会把 13407 的 2012 年也答成 META——
那正是 D-03 否决掉 1-D `ticker(symbol)` coord 的那个缺陷。旁车**只写这个面板自己的 PERMNO**
（参考表里有 40,518 个），读侧是 `quantlab/dataset/crsp/tickers.py:CrspTickerLookup`：
`as_of(permno, day)` 是严格的单值提问（缺文件会抛），`label(permnos, day)` 是展示层的批量入口
（**永不抛**，旁车缺失或损坏——含解析不了与解析得了但形状不对——都原样回落成数字）。
「解析不了」是三种：字节不是合法 **UTF-8**、字节不是 JSON、以及**嵌套**深到解析器自己爆栈
（`RecursionError` 是 `RuntimeError` 子类，03.11-15 之前从这里逃出去过，G-03.11-3 / WR-01）；
这三种在 `as_of` 侧对应的都是一次 shaped 拒绝——点名类名、旁车绝对路径与重建补救的 `ValueError`。
强平日志、模型的 missing/extra 清单、`UniverseMask.report()` 三个调用点都走后者；
`browse_zarr` 的拒绝文案只是在散文里点名这个类，并不调用它。

判据是**盘上有没有那个旁车文件**，不是面板属于哪个厂商——所以 Tiingo / Alpaca 的面板输出一字未变。

一页 = **一个 PERMNO 批次 × 一个日历年**，续跑粒度就是「某个 PERMNO 批次的某一年」。
页的边界由 `year_pages()` 这一个函数定义，拉取和体量估算共用它——两份内联拷贝会漂移，
而漂移在危险的方向上是看不见的（估算用的边界和实际拉的边界略有不同，磁盘占用被低估）。

### 身份轴决策的反转（Phase 03.10 → 03.11，D-08）

这一节记录的是一次**被推翻的决策**，不是历史背景。Phase 03.10 曾经明确拍板：

> Identifier: the panel `symbol` dimension stays the **ticker** valid at each date
> (matching the universes, factors and backtester); **PERMNO is kept as a data variable**,
> mapped via the CRSP ticker history.

**REVERSED by Phase 03.11（D-08）。** 现在的事实与它逐条相反：面板的 `symbol` 轴是
int64 的 PERMNO；`permno` 不再是一个数据变量（它就是轴本身，所以重建后面板的变量从 28 变成 27）；
period-correct ticker 不再进面板，只进旁车。

反转的理由，按证据强弱排：

1. **「贴不上标签就丢行」是一条没人声明过的准入规则。** ticker 轴要求每一行都能贴上名字，
   于是 `label_rows` 会**丢掉**任何 ticker 区间覆盖不到的行。在完整参考层上实测：
   191,048 条区间行里 34,839 条（18.2%）ticker 为 NULL，牵涉 40,518 个 PERMNO 中的 30,197 个；
   **1,012 个 PERMNO（2.5%）从来没有过任何 ticker**——它们在 ticker 轴上永远进不了面板，
   而没有任何配置项、任何报告说过这件事。这不是删掉死代码，是把一条隐式责任翻到台面上：
   替代它的是一条写明的准入条件，折进既有的 `security_filter`，并且把反事实计数写进
   `crsp_filter_report.json` 的 `admitted_without_ticker`。
2. **撞车 / seam / 类别重拼这三套机制的存在理由全是 ticker 轴。** 换轴之后它们防的故障
   不是变得不太可能，而是**不可拼写**（见上文「不用它会怎样 / 2」「/ 3」）。
3. **原始层本来就是 PERMNO。** 保留 ticker 轴意味着在一条两端都是 PERMNO 的链路中间
   插一次转译，而那次转译正是 1、2 两条的来源。

被反转的不是「ticker 有用」——它当然有用，只是它是**名字**不是**身份**，
所以它去了旁车。另外两处反转记录在 `.planning/ROADMAP.md` 与 `.planning/STATE.md`。

### 六张参考表

| 表 | 用途 | 什么时候拉 |
|---|---|---|
| `crsp_a_stock.stksecurityinfohist` | **ticker 旁车 `{zarr}.crsp_tickers.json` 的唯一数据源**（PERMNO → period-correct ticker 区间、股份类别）；另供每日类型列 | 总是 |
| `crsp_a_stock.stkdelists` | 退市事件（`delret`、`deldlydt`…），**只作事件数据** | 总是 |
| `crsp_a_stock.stkdistributions` | 分派事件（除息日、金额、因子） | 总是 |
| `crsp_a_indexes.dsp500list_v2` | CRSP 自己的 S&P 500 时点成分 | `--universe crsp_sp500` |
| `comp.idxcst_his` | Compustat 指数成分史（`gvkeyx='000208'` = Nasdaq 100） | `--universe comp_nasdaq100` |
| `crsp_a_ccm.ccmxpf_lnkhist` | CCM 链接表，gvkey → PERMNO | `--universe comp_nasdaq100` |

整表拉取，先 `count(*)` 再 COPY，超过 `MAX_REFERENCE_ROWS` 直接拒绝；每张表用
tmp + `os.replace` 落盘；**manifest 最后写**，所以被打断的一次拉取绝不会声称自己完整。
同一个 CRSP 版本**只拉一次**：跳过与否是从 manifest 的 `product_end` 加盘上的文件判断的，
在权限探测和任何 `count(*)` 之前——重复运行零次往返。`--refresh-reference` 强制重拉。

权限探测**只覆盖这次运行真正要读的 schema**：`crsp_a_stock` 总是探，`crsp_a_indexes`
只在 `crsp_sp500` 时探，`comp` + `crsp_a_ccm` 只在 `comp_nasdaq100` 时探。
大多数 CRSP 订阅**不含** Compustat，为一次 S&P 运行去探 `comp` 只会得到一个这次运行不需要的「否」。

### 证券过滤：预设，以及 D-17 的那个读法

**先说规则，它比预设表更重要：过滤筛的是一个「没有明说边界的总体」，它不会推翻一份显式名册。**
用户 2026-09-20 拍板的原话是「优先保证成分股不缺」。落到代码里是三种情况：

- **`--universe` 跑**（`CrspDatasetConfig.roster_universe`）：成分由指数提供方定了，
  所以一个成分在它的**成分区间内**（逐日判定，读 `dsp500list_v2` 的 membership spell）
  永远不会被类型过滤丢掉。区间**之外**它又回到「未指定的总体」，过滤照常生效——
  豁免是有范围的，不是一刀切放宽。
- **`--permnos` 跑**（`CrspDatasetConfig.permnos`）：证券是用户点名的，
  所以它的**每一天**都豁免，不看类型列。
- **没有显式名册的宽筛**：过滤完全照旧。在那里排除 ADR（`AD`）和 unit（`UG`）是有意义的，
  因为没人说过要谁。

豁免**永不静默**。`crsp_filter_report.json` 多了一个 `roster_overrides` 段：

| key | 内容 |
|---|---|
| `sources` | 哪几份名册在起作用，各自覆盖多少 PERMNO / 多少 membership spell |
| `rows_rescued` | 「本来会被丢、结果留下来了」的行数 |
| `permnos` | 每个被豁免的 PERMNO：被拒绝的类型组合、行数、首末日期（JSON key 本身就是 PERMNO） |

这份清单里**没有 ticker/名字字段**：D-01 把面板的 `symbol` 轴换成 PERMNO 之后，
原先那个字段逐字节重复 JSON key，于是在 G-03.11-2 删掉了 ——
「丢/留的是谁」由 key 回答，「为什么」由类型组合回答，审计链是完整的。

没有配名册时这个 key **也在**，只是计数为 0——这样「没发生豁免」和「这个 store 比该功能更早」
可以靠 key 在不在区分开。`rows_rescued` 非 0 时还会打一条 `logger.warning`。
被豁免的行不出现在 `dropped_permnos` 里（它没被丢），
`rows_kept + rows_dropped == rows_total` 依旧成立。

**手上拿着 phase 03.11 之前的 `config.json`？它读不回来了**，这是 D-04 接受的后果
（项目未进生产，不写迁移也不写兼容层；出路是重建 store，而 CRSP 一年才发一次数据）。
`CrspDatasetConfig` 现在**只认这四个**名册/过滤字段，其余都已删除：

| 字段 | 作用 | 备注 |
|---|---|---|
| `permnos` | 转换限定在这些 PERMNO（数字串）；`None` = 原始层里的每一个 | 同时是**显式名册**，覆盖 `security_filter`；空元组 `()` 在赋值时被拒（它可能指「一只都不要」也可能指「全都要」） |
| `roster_universe` | 哪个指数的成分是这次转换的显式名册 | 它做且只做这一件事：在成分区间内覆盖 `security_filter`。字段名在 03.11-08 改过，因为它原先那个名字取自一份**第二职责**——判定同一格里的两个 PERMNO 谁是成分——而 PERMNO 轴上两个证券永不同格，那个问题不可拼写 |
| `security_filter` | 面板收哪些证券 | 预设名或显式谓词字典 |
| `reference_dir` | 参考层位置 | 必填，不从原始根推导 |

基类继承下来的 ticker 侧名册字段 `symbols` **在 CRSP 上被拒绝**，拒绝发生在 **config 赋值**
那一刻并点名 `permnos`。它命名的是一个 CRSP 面板上不存在的轴；替换掉的旧故障是跑到一半时
`.sel(list[str])` 撞上整数索引抛 `KeyError`——一条看起来在怪数据、实际在怪字段选错了的报错。
基类字段本身没动：十几个非 CRSP 读者还在用它，而 PERMNO 是 Binance / Alpaca / WRDS TAQ
永远不会有的标识符。

03.11-08 还删掉了另外两个字段：一个 per-PERMNO 的 ticker 钉子（见下文 QQQ 一节）和一个
「在换东家处置 NaN」的开关（见上文「不用它会怎样 / 2」）——它们各自防的情形在 PERMNO 轴上
都不可拼写，所以是删除而不是留成惰性字段。逐字的旧名 → 新名对照在
`.planning/phases/03.11-*/03.11-08-SUMMARY.md` 里。

| 预设 | 保留什么 | 说明 |
|---|---|---|
| `equity_common`（默认） | `securitytype='EQTY'` + `securitysubtype='COM'` + `sharetype ∈ (NS, SB, CE)` | 保留 REIT（含 8 个 `SB` 的）和非美国注册发行人；丢掉 ADR（`AD`）、unit（`UG`）、基金/ETF、类型未知。**但它不会推翻显式名册**：`--universe` 成分在成分期内、`--permnos` 点名的证券都豁免于本预设，见上面的「名册优先」规则 |
| `shrcd_10_11` | 上面再加 `usincflg='Y'`、`issuertype ∈ (ACOR, CORP)`，`sharetype` 只留 `NS` | 复刻传统 `shrcd in (10,11)`；**会丢掉 REIT 和非美国发行人**，而它们是合法的 S&P 500 / Nasdaq-100 成分 |
| `none` | 全部 | QQQ 基准 store 用的就是它 |

**关于 D-17 的读法，必须说清楚。** D-17 写的谓词是 `securitytype='EQTY' AND securitysubtype='COM'`，
同时又说这个过滤要丢掉 ADR 和 unit。在 live 的 S&P 行上这两句话**互相矛盾**：
一个 ADR 读作 `AD/EQTY/COM/CORP/N`，一个 unit 读作 `UG/EQTY/COM/CORP/N`，
两者都**满足**那个两列谓词。所以 `equity_common` 额外带了 `sharetype` 白名单
（CRSP flag 字典里定义的每一个 ShareType 代码，**除了** `AD` 和 `UG`），
这样才给出 D-17 列举的每一条「丢」和每一条「留」。

**但 2026-09-20 的 live 跑推翻了「恰好」这两个字。** 在 CRSP 里 `UG` 同时标着一只证券的
**公开上市有限合伙（LP）时期**，而那些时期属于货真价实的指数成分——黑石、KKR、嘉年华都在内；
`AD` 则覆盖了皇家荷兰石油的**全部生命周期**。把它们从默认预设里排除，
换来的不是「少了一只 ADR」，而是**一只保留下来的证券被从中间截断**，下游完全看不出来。
证据（7 个中招的 PERMNO、各自丢掉的区间、总体分布）见下文
[已知问题 GAP-1](#gap-1equity_common-会安静地截断真实指数成分的历史)；
**修复已经落地**，就是上面那条「名册优先」规则。

这个读法本身**可逆且零成本**：把字面的两列谓词
`{"securitytype": ["EQTY"], "securitysubtype": ["COM"]}` 作为 `security_filter` 字典传进去就行，
而过滤报告两种情况下都会把差别摆出来。它现在是一个 escape hatch，
**而不是**成分股缺失的解法——那个由名册豁免负责，对指数成分和点名 PERMNO 自动生效，
不需要用户先知道有这个坑。

三条更细的规则：
- 判定是**逐日**的，读 `dsf_v2` 自己的当日类型列——一只后来变成封闭式基金的股票，
  只保留它还是普通股的那一段。
- `dlydelflg='Y'` 的行**继承该 PERMNO 前一天的判定**。退市行正是 CRSP 类型列变空的地方，
  而它带着退市**收益**；按它自己的空类型判，幸存者偏差就从过滤这一步一行一行地回来了。
- 过滤发生在**复权连乘之后**。先过滤的话，下一个被保留的日子的复权涨跌会横跨一段面板上
  已经看不见的收益——一个凭空出现、没有可见成因的数字。

`primaryexch` / `conditionaltype` / `tradingstatusflg` / `exchangetier` 是**可过滤但不在任何预设里**的：
它们在一只证券的一生中会变，默认按它们过滤会在序列里打洞，还可能丢掉退市行。

### 变量表：Tiingo 的 12 个，加 15 个 CRSP 特有

前 12 个和 `StockDataset` 完全同名同义，这就是「drop-in」的全部含义：

`open, high, low, close, volume, adjOpen, adjHigh, adjLow, adjClose, adjVolume, divCash, splitFactor`

15 个 CRSP 额外变量：

| 变量 | CRSP 来源 | 单位 / 含义 |
|---|---|---|
| `permno` | `permno` | 证券永久编号（float64，见下） |
| `permco` | `permco` | 公司永久编号 |
| `ret` | `dlyret` | 含分红的日收益；缺失保持 **NaN**，绝不是 0 |
| `retx` | `dlyretx` | 不含分红的日收益 |
| `shrout` | `dlyshrout` × 1000 | **股**（CRSP 存的是千股） |
| `market_cap` | `dlycap` × 1000 | **美元**（CRSP 存的是千美元） |
| `bid` / `ask` | `dlybid` / `dlyask` | 买/卖价 |
| `prc_is_bidask` | `dlyprcflg == 'BA'` | 1.0 = 当天没成交，价格是买卖中点；flag 为空时 NaN |
| `is_delisting` | `dlydelflg == 'Y'` | **标记**，没有任何地方乘以它 |
| `numtrd` | `dlynumtrd` | 成交笔数 |
| `cumfacpr` / `cumfacshr` | `dlycumfacpr` / `dlycumfacshr` | 累计价格/股数因子 |
| `facprc` | `dlyfacprc` | 当日价格因子：平常 1.0，AAPL 2020-08-31 四拆一那天 4.0 |
| `close_trade` | `dlyclose` | **成交**收盘价，bid/ask 日、1992 年前 Nasdaq、退市行上为空 |

两条容易踩的：

- **`close = abs(dlyprc)`，不是 `dlyclose`。** `dlyclose` 在上面那三种情况下是空的，
  而 `dlyret` 是**由 `dlyprc` 算出来的**。用 `dlyclose` 会把一条收益链和一条和它对不上的
  价格序列放进同一个面板，而且分歧是静默的，只表现为一个错的 `adjClose` 水平。
  `abs()` 是防一条遗留形态行的护栏，不是「没成交」的探测器——CIZ 里实测**零**条负价格
  （2000 年 0 条负价 vs 122,471 条 `BA` 行），靠符号判断等于什么也没标出来。
- **除了 `anomaly_flag` 全部是 float64**，`permno` 也是。稠密的笛卡尔面板需要一个 NaN
  来表达「这个 symbol 那时还不存在」。
- **`adjVolume` 在 `dlycumfacshr` 为空的那天是空的**（`adjVolume = dlyvol × dlycumfacshr /
  锚点行的 dlycumfacshr`），退市的 `DA` 行正是这种行。而一只证券在整个窗口内**一天都没有**
  「正的 `dlyprc` + 非空的 `dlycumfacshr`」时，转换会点名**拒绝**，而不是给它一整列 NaN——
  一列 NaN 长得像「没有数据」，实际含义却是「数据在那里，复权把它丢了」，
  任何流动性筛选都会安静地剔掉它。

### 复权：锚点是 store 内第一个可用收盘（后复权），以及为什么延长窗口是安全的

`adj*` 是**总收益复权**（拆股 + 分红），和 Tiingo 的 `adjClose` 一样把两者都算进去，
但**连乘方向是往后的**：把 CRSP 的 `dlyret` 从**该 PERMNO 在 store 内第一个可用收盘**
往**后**连乘，所以**最早**那天的 `adjClose` 就等于它的原始 `close`，越晚的日子 `adjClose`
相对名义价越**高**（长期赢家尤其明显——整条序列是用最早那天的美元重述的）。
`adjOpen/High/Low` 仍按同一个日因子缩放，`adjVolume` 仍用 `dlycumfacshr` 派生的股数因子
（**D-07**：总收益、仅拆股、成交量是**三个不同的**复权因子，不要互相代用）。

换向买到的是一个**不可变的 store**：锚只是每个 PERMNO **首行**的函数。
只向前 append（`start_date` 固定、`end_date` 延长）时首行不动 ⇒ 锚不动 ⇒
已经写出去的每一个 `adj*` 值都不会被下一次追加改写。所以不再需要把转换窗口钉成旁车，
也不再需要跨运行闸门——它们连同「锚点会随窗口移动」这件事一起，在 03.12 被删掉了。
这个不可变性的边界写在本节末尾的「已知限制」里。

四条实现上的讲究：

- 锚点是该 PERMNO 在窗口内**第一行同时满足「`dlyprc` 严格为正」和「`dlycumfacshr` 非空」**
  的行，不是简单的第一行，也不只是「第一个非空收盘」。两个条件写在**同一个**谓词里，
  是为了让 `adjClose`、`adjOpen`、`adjHigh`、`adjLow`、`adjVolume` 五个量**全部**从
  **同一行**上读出来：只按收盘挑锚点、再去那行读一个 NULL 的 `dlycumfacshr`，
  就是 `adjVolume` 整段历史变 NaN 的那个机制（GAP-B）。
  「非空」曾经是个漏洞：CRSP 的无价格哨兵是**数字** `0.0`，它不是空值（GAP-A）。
- 一个 PERMNO 没有任何合格行，或者它的 `G_anchor` 是 `0.0` / 非有限值
  （`dlyret = -1.0`，一次合法的血本无归），转换都会**点名拒绝**。
  后者的理由和前者一样：`close_A × G_t / 0.0` 在 IEEE 语义下是 `inf`，一路传下去不报错。
  方向性说明：`.first()` 之下第二条**更难触发**——`G_anchor` 是**首行**处的累积值，
  要零化它，那个 `dlyret == -1.0` 必须发生在锚**之前**，而锚就是首个可用行。
  代价写在明处：发生在锚**之后**的 -1.0 不再撞上这道拒绝，它之后那一段 `adjClose`
  会被写成精确的 `0.0`（而原始 `close` 仍然为正）。这是一个**已登记的已知缺口**
  （`.planning/WINDOWS.md`），由 `tests/test_crsp_dataset.py` 里的
  `test_a_total_loss_after_the_anchor_is_not_refused_and_zeroes_the_tail` 钉住。
- 空的 `dlyret` 在连乘里贡献因子 **1**，不是 0。CIZ 的收益会跨过空缺回溯到 `DlyPrevDt`
  （`DlyRetDurFlg` 的 `D3`/`D4`），下一个有效收益已经覆盖了缺失那天，填 0 等于重复计一次。
- 没有价格的那天 `adjClose` 是 **NaN**。连乘在那天是有定义的，不加这个 mask 的话，
  锚点的水平会被当成那天的复权价发布出去——凭空造出一个价格，而收益序列随后会对着它做差。

两个旁车文件（`crsp_filter_report.json` / `crsp_tickers.json`）
在 store **已存在**时不再重写（`quantlab/dataset/crsp/__init__.py` 的
`_write_identity_reports` 开头那道 store-exists 守卫）。后者不是审计报告而是一张
PERMNO → ticker 的**区间表**（见上文「原始层按 PERMNO」）。
代价是：一次追加不会刷新这两份文件，
它们描述的是 store **最初**写成时的那个面板；换来的是一次**被拒绝**的转换
（比如换了更宽的名册、撞上 `on_new_listing='refuse'`）不会把活着的那个 store 的
D-17 审计记录、或者它那份名字表，换成一个从没被写出来过的面板的数字。

所以**把 `--end-date` 往后延、在原有 store 上原地扩展现在是被支持的**：新追加的那一段用的是
同一个锚（每个 PERMNO 的首行没有动），实测**逐位相等**（66,356 行，`max rel 0.000e+00`），
由 `tests/test_crsp_first_anchor.py::test_a_full_and_incremental_build_agree_bit_for_bit` 钉住。
反过来，**动 `start_date` 不是 append，是重建**——为什么，见下。

#### 已知限制：历史值在什么情况下会被静默改写

「已经写出去的值不会再变」只在**一种** append 形态下成立：`start_date` 固定、`end_date` 延长。
另外两种形态**会改写**已经落盘的历史值，而且代码一条都不拦（**D-10**）：

| 场景 | 实测后果 | 代码检查吗 |
|---|---|---|
| **`start_date` 往后移**（换一个更晚的起点重新转换同一个 store） | 重叠区 **112,359 / 112,934 行**改变，max rel **0.75** | ❌ |
| **raw tier 向前回填**（`start_date` 一个字没改，但原始层里出现了比原锚更早的行） | **756 / 756 行**全部被改写，max rel **0.76** | ❌ |

第二行是 `.first()` 锚引入的**新**风险方向：`.last()` 锚怕 `end_date` 变，`.first()` 锚怕
`start_date` 变——以及它的等价物，原始层往**前**长。两条都是实测出来的数字，不是理论推演。

**没有任何代码检查它们。** `ChunkLedger.assert_consistent`
（`quantlab/base/chunking.py:307-372`）看的是符号轴指纹、store 与台账的空/非空是否一致、
store 的末尾是否等于最后一个窗口的 `end`——**它不看 `start_date`**。所以这两种形态会安静地
跑完，留下一个每个 `adj*` 都变过、而所有一致性检查都绿的 store。

这是一次**明示的取舍**（D-10：纯重算，不加锚点旁车、不加反解比对）：逐位相等是「实测是 0」，
不是「结构上不可能不是 0」。它依赖三个**外部**事实——`start_date` 不动、raw tier 不向前长、
CRSP 年度 vintage 不改写锚行——而代码一件都不检查。操作者的动作因此是明确的：改了
`start_date`、或者发现原始层往前长了，就按下文「重建 CRSP store 的完整步骤」**重建**，
不要当成一次 append。

### 事件落在除息日

`divCash` = `dlyorddivamt + dlynonorddivamt`，落在除息日（AAPL 2020-08-07 是 0.82，
两者都为空时是 0.0）；`splitFactor` 来自 `dlycumfacpr`（2020-08-31 是 4.0，平常 1.0）；
`facprc` = `dlyfacprc`。`stkdelists` 和 `stkdistributions` 整表原样保存在 `_reference/`，
但**没有任何东西**把 `delret` 合并进收益序列。

### 两个股票池，以及它们的覆盖边界

| `--universe` | 来源 | 时点覆盖从 | 说明 |
|---|---|---|---|
| `crsp_sp500` | `crsp_a_indexes.dsp500list_v2` | **1925-12-31** | CRSP 自己的成分史，比 Wikipedia 那套早五十年 |
| `comp_nasdaq100` | `comp.idxcst_his`（`gvkeyx='000208'`）经 CCM 链到 PERMNO | **1995-01-01** | Compustat 在这天**左截断**（100 条 spell 同一天开始），不是那天才成立 |

两个池的每一个区间端点都是**显式**的，而且都被裁到 CRSP 产品末日 `2025-12-31`。
这不是洁癖：`IndexConstituentDataset._densify` 会把**开区间**的右端延到**挂钟今天**，
一个 NULL 端点就能把 CRSP 股票池推到价格覆盖之外好几个月。
起点晚于产品末日的 spell 直接丢弃并计数（live 上有 10 条）。

CCM 的连接是 `gvkey` **且** `iid = liid`，**不是** `linkprim`。常见的
`linkprim IN ('P','C')` 写法在这里会留下 GOOGL、丢掉 GOOG——而 Alphabet 的两个类别
在同一个 gvkey（160329）下**都是**真实的 Nasdaq-100 成分。

没有任何有效链接覆盖的成分日**绝不会被安静丢掉**：默认 `ValueError` 列出这些 spell
和未覆盖区间，`--allow-unlinked-ndx` 才放行（并把它们记进 `report['unlinked']`）。
这个拒绝**只看落在你请求窗口之内的未覆盖日**：窗口碰不到的缺口拿不走这个窗口的任何一个
成分，所以它们照样记进 `report['unlinked']` 并打日志，但不再报错。
两条链接之间 4 天以内的缝隙算链接表的接缝，容忍并记录；**完全没有链接**的 spell
无论多短都算 unlinked——容忍度是用来桥接两条链接的，没有链接就没有东西可桥。

### QQQ：只是数据

`--qqq` 拉 PERMNO **86755**，写进**它自己的** store `wrds_crsp_qqq_1d.zarr`，
一个标的一列——而且现在是**按构造**的一列：轴就是 86755。
（在 ticker 轴上这里曾经要一个 per-PERMNO 的 ticker 钉子，因为 CRSP 给 86755 的
period-correct ticker 在 2004-12-01…2011-03-22 真的是 `QQQQ`，会把一只工具的历史劈成
带洞的两列。PERMNO 轴上没有东西需要钉，那个字段在 03.11-08 被删掉而不是留成惰性字段；
`QQQQ` 那段没丢，它是 ticker 旁车里的一条区间。）
`security_filter="none"` 是**承重**的：QQQ 是 `FUND`/`ETF`，
用股票默认过滤建的基准 store 会是**空的**，而不是看起来不对。

它**不是**权益面板的一列——一个 ETF 和它自己持有的成分在同一个截面里排序，
就是指数在和自己比。而且它是按**名册**移出权益面板的，不只是靠过滤：
`--security-filter none` 是一个用户会做的合法选择，靠过滤挡住的基准会在那一刻悄悄溜回截面。

本阶段 QQQ **只是数据**：没有任何东西把它接到回测器的 `benchmark_dataset` 上
（那个位置仍然 `NotImplementedError`，phase 03.7 D-08），这一点由两半测试锁着——
标识符扫描证明没有 CRSP 模块去碰 `benchmark_dataset`，AST 检查证明那个守卫还在。

`--qqq` **单独**配 `--to-zarr`（不给 `--permnos` / `--universe`）是受支持的：
权益名册为空时脚本会跳过权益面板那次转换并打印一行以 `Skipping the equity conversion:`
开头的说明，QQQ store 照常写出来。这一条以前会崩，
见下文[已知问题 GAP-2](#gap-2--qqq-单独配---to-zarr-会崩在权益面板转换上)（已修复，计划 03.10-15）。

---

## 它是怎么工作的

```
scripts/ingest_wrds_crsp.py
  │  参数校验：--universe/--permnos/--qqq 至少一个、两个日期都必填、
  │  PERMNO 必须是数字串、预设名合法——全部在建连接之前
  ▼
WrdsSession.shared()  ── 一次运行一个会话 = 最多一次 Duo 推送 ──────────────┐
  ▼                                                                       │
schema_usable(...)    权限探测，只探这次运行真正要读的 schema               │
  ▼                                                                       │
max(dlycaldt)         产品末日探一次 → 裁剪 / 拒绝 / 版本戳比对              │
  ▼                                                                       │
CrspReferenceTables.pull()   六张小表 → _reference/ （同版本只拉一次）        │
  │   count(*) → 行数上限 → COPY → 行数核对 → cast → 原子落盘 → manifest 最后写 │
  ▼                                                                       │
CrspMembership.permnos_in_range(...)   股票池 → PERMNO 名册（数字序排序）     │
  ▼                                                                       │
CrspVolumeProbe.count_rows_by_year     每 (年, PERMNO 批次) 一次 count(*)     │
  ▼                                                                       │
SqlVolumeGuard.assert_acquisition_volume_fits   超限拒绝，零 COPY            │
  ▼                                                                       │
registry.run(SOURCE, ...)   每 (年, 批次) 一次 COPY；核对行数、               │
  │                         (permno, dlycaldt) 唯一性、页归属                │
  ▼                                                                       │
原始分片 downloads/us_equity/1d/wrds_crsp/wrds/month=YYYY-MM/               │
  ▼   （仅 --to-zarr）                                                     │
registry.convert(...)  → CrspStockDataset                                 │
  │   全局复权（每 PERMNO 一个锚点，缓存在实例上）                            │
  │   → PERMNO 轴（symbol = int64 permno）→ 证券过滤 → ticker 区间表          │
  ▼                                                                       │
data/us_equity/1d/wrds_crsp_{sp500|nasdaq100|custom}_1d.zarr               │
  + .crsp_adjustment.json  +  .crsp_filter_report.json                     │
  + .crsp_tickers.json     （+ .chunks.json，分块台账）                      │
  ├─ （--qqq）      wrds_crsp_qqq_1d.zarr                                   │
  └─ （--universe） wrds_crsp_{sp500|nasdaq100}_membership.zarr             │
                                                        finally: close_shared()
```

几个要点：

- **拒绝都发生在拉数据之前。** 参数错误在连接之前；未订阅的 schema 在参考表拉取之前
  （所以不会留下半填的 `_reference/`）；产品边界在任何日频 COPY 之前；
  体量超限在任何 COPY 之前。这个顺序有九条结构锁钉着，其中一条做过变异验证
  （把护栏调用删掉，锁确实会失败）。
- **参考表必须在名册解析之前。** 名册是从参考表里解析出来的，而且必须在到达采集 config
  **之前**就已经是 PERMNO——否则一个 ticker 名册会被记成「WRDS 拒绝了这些证券」的逐批失败。
- **复权锚点在一次转换里只算一次**，按整个配置窗口算，然后切片给每个分块窗口用。
  按 `year` 和按 `month` 转出来的 store 是 `assert_identical` 相等的。
- **旁车文件是审计线索**，不是日志。「S&P 面板少了那个 ADR 成分」现在是文件里的一行，
  而不是一列没人注意到它不见了。

---

## 凭证

- 用户名：环境变量 **`WRDS_USERNAME`**。
- 密码：**只**放在 `~/.pgpass`（或 `$PGPASSFILE` 指向的文件），权限必须是 600，其中一行形如：

  ```
  wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>
  ```

  ```bash
  chmod 600 ~/.pgpass
  ```

- 和 TAQ 路径共用同一套 `WrdsSession`：密码从不进本仓库的代码，由 libpq 自己从 `.pgpass` 读；
  检查 `.pgpass` 时只解析前 4 个字段，密码字段不会被读进任何变量；
  报错信息里用户名写成 `$WRDS_USERNAME`，不打印它的值。
- 设置了 `PGHOSTADDR`、`PGSERVICE` 或 `PGSERVICEFILE` 会被拒绝。
- 命令行**没有**任何用户名或密码参数（有 AST 结构测试锁着），而且这个脚本打印的任何东西
  都不含凭证（有一个「种一个假用户名值进环境、扫 stdout 和 stderr」的测试）。
  **不要**把密码敲进任何命令行——它会留在 shell 历史和进程列表里；
  本仓库已经因为硬编码 Tiingo key 真实泄露过一次。

---

## 体量

live check（`03.10-LIVE-CHECK*.json`）量到的真实规模：

| 对象 | 行数 / 范围 |
|---|---|
| `crsp_a_stock.dsf_v2` | **110,257,376** 行，1925-12-31 … **2025-12-31** |
| 其中 2020 一年 | 1,950,357 行（`(permno, dlycaldt)` 全不重复） |
| `stksecurityinfohist` | 191,048 行 |
| `stkdistributions` | 1,101,681 行 |
| `stkdelists` | 29,833 行 |
| `dsp500list_v2` | 2,084 行（其中 503 条当前开放，`mbrenddt='2025-12-31'`） |
| `comp.idxcst_his`（gvkeyx 000208） | 523 条 spell / 436 家公司，1995-01-01 … 2026-07-07 |
| QQQ（PERMNO 86755） | 6,746 行，1999-03-10 … 2025-12-31 |

- 行数在**拉取之前**用 `count(*)` 按 (日历年, PERMNO 批次) 数出来，护栏默认上限是
  **20 GiB 原始字节**和 **7 亿行**。
- 打印出来的估算里 bucket 叫 **`year buckets:`**（TAQ 那边叫 `trading days:`），
  因为 CRSP 的一页就是一个日历年——这样被拒绝时给出的边界是一个重跑命令**真的能用**的边界。
- **每行字节数 `DEFAULT_BYTES_PER_ROW = 150` 是假设值**，打印的估算会标明 `ASSUMPTION`。
  2026-09-20 的 live 跑第一次把它量了出来：在 S&P 500 规模上（96 个分片、134,054 行、
  10,495,581 字节）实测 **78.3 字节/行**，也就是 150 这个假设**高估了约 1.9 倍**。
  对一个体量护栏来说高估是**安全**的方向（宁可提前拦住，不要事后爆盘），所以它不是 bug，
  代码没有改；想收紧成实测值是一个可选的后续项。
  注意只拉 3 个 PERMNO 时量到的是 381 字节/行——那是 60 个极小分片上的 parquet 元数据开销，
  不是有代表性的数字，别拿它去调常量。
  行数上限是独立的第二道线。
- `--force-volume` 跳过**拒绝**，不跳过**算术**：估算照算照打印，并额外说明是被强制放行的。
  没有环境变量、也没有配置项能整体关掉护栏。

---

## 完整例子

### 例 1：离线真跑过的参数拒绝（不连接 WRDS）

这几条在建任何连接之前就被拒绝，以下是**实际输出**：

```bash
$ uv run python scripts/ingest_wrds_crsp.py --permnos AAPL \
      --start-date 2024-01-01 --end-date 2024-12-31
ingest_wrds_crsp.py: error: --permnos ['AAPL'] are not PERMNOs. CRSP's raw tier is keyed by PERMNO (a digit string, e.g. 14593 for AAPL), not by ticker; resolve a ticker roster to PERMNOs first, or use --universe.

$ uv run python scripts/ingest_wrds_crsp.py --permnos 14593 --start-date 2024-01-01
ingest_wrds_crsp.py: error: --start-date and --end-date are both required: the window is counted and priced before any data is pulled, and it is checked against the CRSP annual product end.

$ env -u WRDS_USERNAME uv run python scripts/ingest_wrds_crsp.py --permnos 14593 \
      --start-date 2024-01-01 --end-date 2024-12-31
WRDS_USERNAME environment variable must be set to your WRDS username. The password is never read from config or from this code: libpq reads it from ~/.pgpass (chmod 600), so store it there before running a WRDS acquisition.
```

第一条拒绝在 phase 03.11 之后**更强了一层**，而且它现在有一个同形的兄弟：CRSP 面板的
`symbol` 轴就是 PERMNO，所以 ticker 在这条链路上**没有任何入口**。

| 你写的 | 什么时候被拒 | 该用什么 |
|---|---|---|
| `--permnos AAPL` | 参数解析时，建连接之前 | `--permnos 14593`，或 `--universe crsp_sp500` |
| `CrspDatasetConfig(symbols=("AAPL",))` | **config 赋值那一刻**（不是运行到一半） | `permnos=("14593",)`。`symbols` 是基类的 **ticker 侧**名册，它命名的轴在 CRSP 面板上不存在；拒绝消息直接点名 `permnos` |

第二条替换掉的旧故障值得记一笔：它从前不报错，而是一路跑到 `.sel(list[str])` 撞上整数索引才抛
`KeyError`——一条看起来在说数据有问题、实际在说字段选错了的报错。
想按名字找 PERMNO 就去查 ticker 旁车（`CrspTickerLookup.as_of`），那是名字唯一的住处；
**不要**拿名字去 `.sel()`。

整条链路（权限探测、产品边界、参考表、名册、护栏、拉取、转换、单连接）在
`tests/test_ingest_wrds_crsp.py` 里用离线的 `FakeCrspSession` 端到端跑过（29 个测试）。

### 例 2：真实拉取（**这 6 条命令 2026-09-20 由账号持有人真实跑过**，每条一次 Duo 推送）

```bash
export WRDS_USERNAME=<你的 WRDS 用户名>

# 几个 PERMNO，只拉原始分片，停在 raw
# 14593=AAPL  13407=FB→META  83443=BRK.B
uv run python scripts/ingest_wrds_crsp.py --permnos 14593,13407,83443 \
    --start-date 2019-01-01 --end-date 2023-12-31

# 同一条命令再跑一次：每个 PERMNO 都按水位线跳过

# 加上转换，得到面板和它的旁车文件
uv run python scripts/ingest_wrds_crsp.py --permnos 14593,13407,83443 \
    --start-date 2019-01-01 --end-date 2023-12-31 --to-zarr

# CRSP 自己的 S&P 500 时点成分，同时写成分面板
uv run python scripts/ingest_wrds_crsp.py --universe crsp_sp500 \
    --start-date 2024-01-01 --end-date 2024-12-31 --to-zarr

# Compustat 的 Nasdaq-100（经 CCM 链到 PERMNO）
uv run python scripts/ingest_wrds_crsp.py --universe comp_nasdaq100 \
    --start-date 2024-01-01 --end-date 2024-12-31
#   若因未链接的 spell 停下，确认后再加 --allow-unlinked-ndx

# QQQ 基准，单独一个 store；窗口会被裁到产品末日并打印裁剪行
#   权益名册是空的，所以权益面板那次转换会被跳过并打印一行说明（GAP-D，已修复）
uv run python scripts/ingest_wrds_crsp.py --qqq \
    --start-date 2024-01-01 --end-date 2026-06-30 --to-zarr

# 起点就在产品末日之后：在任何拉取之前被拒绝，点名 2025-12-31
uv run python scripts/ingest_wrds_crsp.py --permnos 14593 \
    --start-date 2026-01-05 --end-date 2026-06-30
```

live 跑出来的关键数字（全部与离线契约一致）：

| 检查点 | 实测 |
|---|---|
| AAPL（14593）2020 年日行数 | **253** |
| 三个 PERMNO 在 2019-01-01…2023-12-31 的行数 | 各 1,258，合计 3,774（估算打印 `year buckets: 5`） |
| 第二次同参数重跑 | `Resume: skipping 3/3 symbols already covering ...; 0 remaining.`，参考表也整体跳过，零次查询 |
| AAPL `adjClose` 2020-08-31 ÷ 2020-08-28 | **1.033912**（原始 close 499.23 → 129.04，四拆一） |
| FB / META 交接 | FB 866 个非空交易日止于 2022-06-08，META 392 个起于 2022-06-09，无重叠 |
| `--universe crsp_sp500` 2024 名册 | **520** 个 PERMNO（>500 是因为区间重叠保留了年中离开指数的证券——正是反幸存者偏差的设计），520/520 成功，面板 522 列 |
| `wrds_crsp_sp500_membership.zarr` | dims `{timestamp: 366, symbol: 2003}`，首日和末日都是 **503** 个成分 |
| S&P 过滤报告 | 130,280 行进，130,028 行留，丢 **252** 行，全部来自 `UG/EQTY/COM/CORP/N`（PERMNO 75154 = CCL 嘉年华）——**这就是 GAP-1** |
| S&P 准入反事实（`admitted_without_ticker`） | `{"permnos": [], "rows": 0}`——2024 这一窗内没有任何一行是「只有 PERMNO 轴才收得进来」的；这个字段**始终写入**，所以「没发生」和「这个 store 比该字段更早」分得开 |
| `--universe comp_nasdaq100` 2024 | 默认因 6 条未链接 spell 退出（exit 1）；`--allow-unlinked-ndx` 后名册 **108** 个 PERMNO、27,017 行，其中只有 **18** 个是新拉的（另外 90 个已被 S&P 那次覆盖——跨股票池增量续跑生效） |
| QQQ（86755） | 裁剪行逐字打印 `clipped end 2026-06-30 -> 2025-12-31 (crsp_a_stock annual product end)`；拉到 502 行（2024-01-02…2025-12-31），2025-12-31 收盘 **614.31** |
| 2026 起点 | 在任何拉取之前拒绝，点名 `2025-12-31` 和「annual update 产品」 |
| 落盘 | 原始层 31M（132 分片）+ `_reference` 18M + `_watermarks` 2.1M；`wrds_crsp_sp500_1d.zarr` 9.9M；版本戳 `{"product_end": "2025-12-31"}` |

> **2026-09-22 追记（quick 260922-mb1，未链接判断按窗口收窄）。** 表里
> `--universe comp_nasdaq100` 2024 那一行记的是 2026-09-20 的实测，**不改**。
> 但从 2026-09-22 起同一条命令不会再因那 6 条 spell 退出：它们的未覆盖区间全在
> 1999–2008，落在 2024 窗口之外，现在只记录不报错。名册数字本身没变，仍是
> **108** 个 PERMNO——因为 `--allow-unlinked-ndx` 从来没有丢掉过这个窗口看得见的成分，
> 它当初拦下的是一段 2024 年根本用不到的历史。

> **2026-09-21 追记（phase 03.11-10，PERMNO 轴最终重建）。** 上面这张表是 2026-09-20 那次
> live 拉取 + 转换的实测，**面板那几行描述的是 ticker 轴的 store**。盘上的
> `wrds_crsp_sp500_1d.zarr` 已用当前代码在同一窗口（2024-01-01…2024-12-31）重建过，
> 下表是重建后的实测，两张表**并存**：上表是拉取与厂商事实，下表是当前面板的事实。

| 指标 | ticker 轴（2026-09-20） | **PERMNO 轴（2026-09-21 重建）** | 说明 |
|---|---|---|---|
| dims | 252 × **523** | 252 × **520** | 少 3 列，逐个可数：2024 窗口内发生的三次改名 `FLT→CPAY`(12449)、`CDAY→DAY`(17700)、`PEAK→DOC`(67598) 在 ticker 轴上各占两列，在 PERMNO 轴上各是一列 |
| `data_vars` | 28 | **27** | 少的那个是 `permno`——它不再是数据变量，它就是轴 |
| `symbol` 落盘 dtype | `StringDType` | **`int64`** | 读的是 zarr 原始 dtype，不是解码后的值 |
| `anomaly_flag` True | 13 | **13** | 12 个拆股日误报 + `GL 2024-04-11` 的真实 −53%；换轴不改变这些事实 |
| `adjClose <= 0` | 0 | **0** | |
| `close == 0` | 0 | **0** | |
| `adjClose` NaN == `adjVolume` NaN == 结构性空格 | 1520 | **764** | 数变小是因为格子变少（252×520 = 131,040，而非 131,796）且合并的列互相补上了对方的空格。**成立的是那个等式**，不是那个数：没有一个多余的 NaN |
| 旁车文件个数 | 4（含一份 58 KB 的符号学审计） | **4**（那份审计没了，ticker 区间表顶上） | 那份审计的六个字段全是关于 ticker 轴的陈述，在 PERMNO 轴上只可能永远报「没发生」；机制与文件一并删除，重建会把盘上残留的旧文件清掉 |
| 旁车 `.crsp_tickers.json` | 不存在 | **存在，520 个 PERMNO** | 区间表的 PERMNO 与面板列**按集合**相等，不只是个数相等 |

这几条由真实数据门 `tests/test_crsp_rebuild_measurements.py` 自动锁定；它**不挂在日常回归上**
（挂上去就等于每跑一次测试重建一次真实面板），跑法写在那个文件的模块 docstring 里。

原始数据已经在盘上、只想换过滤预设或分块粒度重新转换时，**不需要连 WRDS**：
在 Python 里直接构造 `CrspDatasetConfig`（`raw_data_dir_path` 指向上面的原始目录，
`reference_dir` 指向它的 `_reference/` 兄弟）并调用
`quantlab.registry.convert(DataSourceRegistry.get("wrds"), cfg, data_type="crsp_daily")`。
注意复权锚点由 `start_date`/`end_date` 决定，改窗口会被旁车文件拒绝。

---

## 常见坑

- **改名不再劈列，但列名是数字。** FB → META 在面板上**始终是 13407 这一列**，
  没有一列结束、没有一列开始（phase 03.11 之前不是这样，那时它是两列，会被只看
  「这一列还有没有数据」的回测读成退市）。代价是 `symbol` 轴上全是整数：
  想知道 13407 在某一天叫什么，查 `{zarr}.crsp_tickers.json`
  （`CrspTickerLookup.as_of(13407, day)`），**不要**拿名字去 `.sel()`。
  强平日志、模型的 missing/extra 清单、`UniverseMask.report()` 已经替你查过了。
- **1992 年以前的 Nasdaq 行没有 OHLC。** `dlyopen/high/low` 在那之前普遍为空，
  `close_trade` 也是；`close`（= `abs(dlyprc)`）还在。用到最高最低价的因子在那段历史上会大面积 NaN。
- **窗口一旦定下就别原地改。** 延长 `--end-date` 重跑会在写入之前被拒绝，
  报错点名两个锚点、旁车文件路径和原因。出路是删掉 store **连同它全部的旁车**重建——
  `quantlab/dataset/crsp/rebuild.py:CrspStoreRebuilder` 就是这件事的执行者，
  它的清场清单是唯一一份权威列表（只删 `.zarr` 目录会留下描述**上一个**面板的审计文件）。
- **换 CRSP 年度版本 = 换原始目录。** 版本戳不匹配同样在 COPY 之前拒绝，
  两条出路（换 `subdir` / 删原始根及其 `_watermarks`、`_vintage` 兄弟）都写在报错里。
- **Nasdaq-100 停在未链接的 spell 上不是 bug。** 先看报错列出的 spell 和未覆盖区间，
  确认那确实是链接表的缺口而不是你的窗口选错了，再用 `--allow-unlinked-ndx`；
  用了之后这件事会记进面板自己的 `config.json`。
  这个检查**只看你请求的窗口**（2026-09-22 起）：live 上曾经拦路的那 6 条 spell
  （gvkey 012884 / 063180 / 064606 / 065068 / 065489 / 106368）未覆盖区间全都落在
  **1999–2008**，所以它们不再拦住一次 2015 年以后的拉取；它们仍然列在
  `report['unlinked']` 里，也仍然打进日志，只是不再报错。
  反过来，落在**请求窗口之内**的缺口照样停住整个流程——那才是真会丢成分的情形，
  也正是 `--allow-unlinked-ndx` 有损的那一次，值得逐条读完再加。
- **`--universe` 只有 `crsp_sp500` 和 `comp_nasdaq100`。** 这是 CRSP 厂商自己的两个池；
  Wikipedia 那套 `sp500` / `nasdaq100` 仍然服务于其他厂商，两者不互通。
- **显式名册的 store 叫 `custom`，不叫 `sp500`。** 就算你列的 PERMNO 恰好都是 S&P 成分，
  你也没有建出一个 S&P 面板，而 store 的名字会跟它一辈子。
- **两个日期都必填。** 窗口要先数行数、过护栏、对产品边界，没有默认窗口。
- **`--rows-per-symbol-day` 在这里没有意义。** 它是 Alpaca tick 护栏用的；传了会报错。
- **150 B/行是假设，实测是 78.3。** 护栏的字节估算偏大约 1.9 倍，是安全的方向，没改。
- **会话断了不会自动重连**（每次重连都可能推送 Duo）；重跑即可从「某批次的某一年」续上。

---

## 已知问题

2026-09-20 由账号持有人跑完 7 条 live 命令后发现的两个问题，**都还没修**。
它们被记录在 `03.10-11-SUMMARY.md` 里，留给后续的缺口修复计划。

> **2026-09-20 追记（计划 03.10-13）：** 这一节写完之后，代码审查又查出两个**复权列**的缺陷
> ——退市哨兵行成了复权锚点（GAP-A，`03.10-REVIEW.md` CR-01）、锚点行的 NULL
> `dlycumfacshr` 让 `adjVolume` 整段变 NaN（GAP-B，CR-02）。**这两个现在都已修好**，
> 见上文「缺了退市收益」和「复权」两节；另外两个转换旁车缺陷（WR-02：非分块入口不写旁车、
> 不过锚点闸；WR-03：被拒绝的转换会覆盖活着的 store 的审计报告）也一并修好了。
> **下面的 GAP-1 和 GAP-2 仍然是开的**，各自由后续计划负责。
> 用旧代码转出来的 store 需要重建：那四只 2024 年真实退出 S&P 500 的票
> （CTLT / MRO / PXD / WRK）在旧 store 里的五列 `adj*` 是错的。
>
> **2026-09-20 再追记（计划 03.10-14）：GAP-1 也修好了**，实现的就是上文
> 「证券过滤」一节开头那条「名册优先」规则——`--universe` 成分在成分期内、`--permnos`
> 点名的证券都不再被类型过滤丢掉，豁免记在 `crsp_filter_report.json` 的
> `roster_overrides` 段里。下面的 GAP-1 小节保留了它的**证据**（7 个 PERMNO 的表和总体分布），
> 因为那是这条规则存在的理由，但它描述的行为已经不是现在的行为了；
> 小节标题按本项目的 D-18 规矩原样留着，只在下面加了状态行。
> **上面那句「两个都还开着」因此已被推翻：现在只剩第二个还开着**
> （`--qqq` 单独跑崩在权益面板转换上，小节在本节末尾）。
>
> **2026-09-20 第三次追记（计划 03.10-15）：第二个也修好了，本节已经没有开着的缺口。**
> `--qqq` 单独配 `--to-zarr` 现在会跳过权益面板那次转换、打印一行说明、
> 正常写出 `wrds_crsp_qqq_1d.zarr`；连带把底下那颗雷也拆了——
> `permnos=()` 曾经被真值判断读成「原始层里的每一个 PERMNO」，
> 现在在 config 赋值时就被拒绝（WR-01）。详见本节末尾 GAP-2 小节里的「已交付的行为」。
> 这次改动**不要求**重建任何 store：改的是「哪次转换会被发起」和「空名册是什么意思」，
> 不是任何一列的数值。
> 同样地，在 GAP-1 修好之前转出来的 store 也需要重建才能拿到成分股的完整历史——
> 过滤规则变了就意味着面板里的行变了，而 CRSP 一年才发一次新数据，
> 一整个 S&P 日频面板重转只要几分钟。

### GAP-1：`equity_common` 会安静地截断真实指数成分的历史

> **状态：已修复（计划 03.10-14，2026-09-20）。** 标题这句话描述的是**修复前**的行为，
> 按 D-18 原样保留（也为了上文那些锚点链接不断）。现在的行为是上文
> 「[证券过滤](#证券过滤预设以及-d-17-的那个读法)」一节开头的「名册优先」规则。
> 下面留着的是**证据**，不是现状。

**严重程度：高（数据正确性）。**

默认预设排除 `AD` 和 `UG` 两个 `sharetype`。问题在于 CRSP 用 `UG` 标一只证券的
**公开上市有限合伙（LP）时期**，而这些时期属于实打实的大盘指数成分；`AD` 也不只是「外国 ADR」。
后果**不是**「这只证券不见了」，而是**一只保留下来的证券被从中间截断**——
截断后的序列和「这家公司上市得晚」在下游长得一模一样，没有任何东西会提示你。

从 live 的 `_reference/stksecurityinfohist.parquet` 里查出来的证据：
1,956 个历史 S&P 500 成分 PERMNO 里有 **7 个**中招。

| PERMNO | 名字 | 被丢掉的区间 |
|---|---|---|
| 92108 | Blackstone（`BLACKSTONE GROUP LP`） | 2007-06-22 … 2019-06-30（**12 年**，序列看起来从 2019-07 才开始） |
| 11990 | KKR（`K K R & CO LP`） | 2010-07-15 … 2018-07-01（**8 年**） |
| 75154 | Carnival（CCL） | 2003-04-21 … 2025-12-31（**2003 年之后全没了**；就是 2024 S&P 那次丢掉的 252 行） |
| 75592 | Plum Creek Timber | 1989-06-02 … 1999-01-03 |
| 14617 | Ares Management | 2014-05-02 … 2018-03-01 |
| 75241 | Pioneer / Parker & Parsley | 1987-12-22 … 1991-02-19 |
| 25267 | Royal Dutch Petroleum | `AD` 覆盖它 1954–2005 的**全部生命周期**，整只证券消失，尽管它做了几十年 S&P 500 成分 |

同一张表里 `EQTY/COM` 行按 `sharetype` 的分布：
`NS` 127,317 行 / 30,263 permno，`AD` 6,696 / 1,379，`UG` 1,508 / 394，`SB` 1,134 / 298，`CE` 43 / 18。

**已交付的行为（用户 2026-09-20 明确拍板：「优先保证成分股不缺」）。** 原则是
*证券过滤筛的是一个没有明说边界的总体，它不能推翻一份显式名册*，
代码里由 `quantlab/dataset/crsp/__init__.py:CrspStockDataset._roster_exemption` 实现：

- `--universe` 跑：成分由指数提供方定了 → 一个成分在它的成分区间内**永远不会**被过滤掉，
  逐日判定，读的是 `CrspMembership.permno_intervals` ——
  和成分面板用的是**同一份** interval 表，所以两处不可能对「谁是成分」有不同意见；
- `--permnos` 跑：证券是用户点名的 → 它的每一天都不过滤；
- 没有显式名册的宽筛：过滤照旧，排除 `AD`/`UG` 在那里仍然有意义。

豁免是**可审计**的：`crsp_filter_report.json` 的 `roster_overrides` 段记着
`sources` / `rows_rescued` / `permnos`（每个 PERMNO 的被拒绝的类型组合、行数、首末日期；PERMNO 本身就是 JSON key），
`rows_rescued` 非 0 时还会打一条 `logger.warning`。没配名册时这个 key 也在，计数为 0。
这**取代**了 `03.10-08-SUMMARY.md` 里「传字面两列谓词字典即可回退」那条注记作为解法——
那个字典仍然是个合法的 escape hatch，但不再是成分股缺失的答案。

**要读什么：** 转换完打开 `crsp_filter_report.json` 的 `roster_overrides`。
`rows_rescued` 就是「本来会被丢、结果因为名册留下来了」的行数，
上面 7 只票在一次覆盖它们成分期的 `--universe crsp_sp500` 跑里应该出现在 `permnos` 里。
`equity_common` 之外还想要 CRSP 自己的全集，`security_filter="none"` 依旧可用。

### GAP-2：`--qqq` 单独配 `--to-zarr` 会崩在权益面板转换上

> **状态：已修复（计划 03.10-15，2026-09-20）。** 标题这句话描述的是**修复前**的行为，
> 按 D-18 原样保留（上文有锚点链接指过来）。现在的行为写在本小节末尾
> 「已交付的行为」里。下面第一段留着的是**证据**，不是现状。

**严重程度：中（命令行缺陷，不影响数据正确性）。**

只给 `--qqq`、不给 `--permnos` / `--universe` 时，权益名册是空的，
但 `scripts/ingest_wrds_crsp.py` 仍然去调权益面板那次 `convert()`（会打印 `Converting 0 PERMNO(s)`）。
空名册解析出来的 store 名是 `custom`，于是撞上之前用别的窗口建好的
`wrds_crsp_custom_1d.zarr`（锚点 `end_date=2023-12-31`），`_assert_anchor_unchanged` 抛 `ValueError`，
**整条命令 exit 1，QQQ 那个 store 根本没被写出来**——尽管 QQQ 的原始拉取是成功的、数据也是对的
（502 行，2025-12-31 收盘 614.31，已核对）。

锚点守卫的行为是**正确**的；错的是 CLI：权益名册为空时就该跳过权益那次转换。

**已交付的行为（计划 03.10-15）。** 这个缺口是两层，分开修的：

1. **CLI：权益名册为空时不转换，并且把这件事打出来。** `--to-zarr` 里整条权益路径
   （`ds_config`、探针、原始数据缺失的拒绝、`Converting N PERMNO(s)` 那行、`convert()`、
   旁车路径）现在都在「`equity_permnos` 非空」这个条件里面。为空时打一行以
   `Skipping the equity conversion:` 开头的话，说清名册里只有那只基准、以及 QQQ 和成分
   掩码两步照常跑。**为什么要打印**：空名册是一个需要被看见的结果，
   不打印的跳过和「转了但什么都没写」在日志里长得一样。
   QQQ 那一块和成分掩码那一块原地不动、也不在这个条件里面——它们是**互相独立的输出**，
   而这个缺陷的全部内容就是其中一个失败会把另外两个一起带走。
2. **`permnos=()` 不再可能被读成「全部」（WR-01，`03.10-REVIEW.md`）。** 这才是底下那颗雷：
   `_derivation()` 以前用 `if self.config.permnos:` 这个**真值判断**卡名册过滤器，
   空元组是 falsy，于是 `permnos=()` 转的是**整个原始层**。现在
   `CrspStockDataset` 的 config setter 直接**拒绝**空元组，报错里把两种可能的含义都点名
   （空元组一只证券都不选；`None` 才是「原始层里的每一个 PERMNO」），
   而所有读这个字段的地方都改成了显式的 `is not None`。
   只修 CLI 不修这一层的话，雷还埋着：下一个写 `permnos=()` 的调用方会拿到
   522 只 S&P 成分，写进一个用另外两只票命名的 store 里，锚点和过滤报告都描述整个原始层，
   而没有任何东西会提示他。

所以现在 `--qqq --start-date ... --end-date ... --to-zarr` 单独跑会 exit 0，
写出 `wrds_crsp_qqq_1d.zarr`（symbol 轴只有一个 `QQQ`），不读也不写
`wrds_crsp_custom_1d.zarr`——**即使**磁盘上已经有一个用别的窗口建好的同名 store 和它的旁车。
这一条由 `tests/test_ingest_wrds_crsp.py::test_qqq_alone_writes_only_the_benchmark_store_over_a_stale_custom_store`
钉住：它会先把那个旧 store 和一份窗口不一致的 `.crsp_adjustment.json` **种下去**，
再断言旧旁车的字节没变——「没写权益 store」是用字节证明的，不是用「文件还在」证明的。

**旧 store 不需要因为这个改动重建。** 这两处改的是「哪次转换会被发起」和
「空名册是什么意思」，不是任何一列的数值；已经写出来的面板里的数字没有变。
（GAP-1 和 GAP-A/GAP-B 那两次重建要求仍然有效，见上文。）
