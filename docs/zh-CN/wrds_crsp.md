# CRSP 日频股票数据（WRDS）

[English](../wrds_crsp.md) | 简体中文

CRSP（Center for Research in Security Prices）提供美国股票的日频价格、收益和公司行为数据，其中包括后来退市的公司。quantlab 通过 WRDS 下载这张表，以永久性的证券标识符 PERMNO 作为主键，再转换成与 Tiingo 股票数据相同的 `(timestamp, symbol)` Zarr 面板，因此因子、模型和回测可以不做修改直接读取它。

磁盘上有三层数据。原始层（raw tier）按 CRSP 提供的样子原样保存；参考层（reference tier）保存几张小的查询表（证券历史、退市、指数成分）；存储（store）是转换后的 Zarr 面板，可以只凭前两层重建，不需要联网。

## 前置条件

下载需要一个开通了 CRSP 订阅的 WRDS 账号。用户名放在环境变量里，密码放在 `~/.pgpass`（权限 600，一行格式为 `wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>`）。quantlab 自己从不读取密码，也没有任何命令行参数接受这两个值。

```bash
export WRDS_USERNAME=<your-wrds-username>
```

Nasdaq-100 股票池还需要 Compustat 和 CRSP/Compustat Merged（CCM）两个 schema 的权限。转换、读取和重建 store 不需要账号，也不需要网络。

本页的会话都是在一份很小的合成原始层上运行的，目录布局见下文，所以数字很小，但都是实际观察到的输出。

## 基础

### PERMNO 与幸存者偏差

CRSP 日频表（`crsp_a_stock.dsf_v2`）里，每只证券每个交易日一行。证券用 PERMNO 标识，这是一个整数，公司改名或换交易所时它跟着证券走，不会变。股票代码（ticker）没有这个性质：Facebook 和 Meta 是同一个 PERMNO 的两个代码，同一个代码在不同时期也可能属于不同的公司。

用今天的指数成分股构造股票池，会漏掉所有破产、被收购或被剔除的公司，使历史业绩虚高，这就是幸存者偏差（survivorship bias）。CRSP 会保留每只证券直到退市当天的全部行情，所以由 CRSP 构造的面板既有赢家也有输家。下文的名单同样会选出成分资格与请求窗口有重叠的所有 PERMNO，而不只是窗口末尾还在指数里的那些。

因此 CRSP 面板的 `symbol` 轴是整数 PERMNO。ticker 只是显示名称，放在单独的查询表里（见“查询 ticker”）。

### 按 PERMNO 下载

`scripts/wrds/` 下的三个脚本通过厂商登记表驱动下载，每种数据一个：`index.py` 拉取一个指数的时点成分股，`market.py` 拉取整个美股市场，`etf.py` 按 PERMNO 拉取一只或多只 ETF。每个脚本都接受 `--start`、可选的 `--end`（默认今天，并截到 CRSP 年度版本的最后一天）、`--refresh` 和 `--data-dir`，并且总是转换成 Zarr。这些命令需要 WRDS 账号，因此不展示输出。

```bash
# CRSP 自己的时点 S&P 500：成分股的日线和成分面板。
uv run python scripts/wrds/index.py --index sp500 --start 2000-01-01

# Compustat 的 Nasdaq-100，通过 CCM 链接到 PERMNO，固定窗口。
uv run python scripts/wrds/index.py --index nasdaq100 --start 2010-01-01 --end 2024-12-31
```

在复制第一行日频数据之前，脚本会检查账号的 schema 权限，把结束日期截到年度产品的最后一天，拉取参考表并解析名单。所有内容都写在 data root 下：

```text
data/downloads/us_equity/1d/wrds_crsp/
    wrds/month=YYYY-MM/     原始 parquet 分片，每个 (permno, date) 一行
    _reference/             参考表 parquet 和 manifest.json
    _watermarks/wrds/       每个 PERMNO 的进度，供 --refresh 使用
    _vintage/wrds.json      原始层来自 CRSP 的哪一个年度版本
data/data/us_equity/1d/     转换后的 Zarr store 及其 JSON 边车文件
```

采集类的文档见 `quantlab.acquisition.wrds.crsp` 和 `quantlab.acquisition.wrds.crsp_reference`。

### 把原始层转换成面板

`CrspDatasetConfig` 指明原始层、参考层和 store。默认的证券过滤器是 `equity_common`，`permnos=None` 表示原始层里的所有 PERMNO。

```python
>>> from quantlab.base.config import CrspDatasetConfig
>>> from quantlab.dataset.crsp import CrspStockDataset
>>> config = CrspDatasetConfig(
...     zarr_file_path="data/data/us_equity/1d/crsp.zarr",
...     raw_data_dir_path="data/downloads/us_equity/1d/wrds_crsp/wrds",
...     reference_dir="data/downloads/us_equity/1d/wrds_crsp/_reference",
...     start_date="2020-08-03",
...     end_date="2020-08-31",
... )
>>> config.security_filter
'equity_common'
>>> CrspStockDataset(config).from_raw_data().save()
>>> panel = CrspStockDataset(config).read().get_xarray_dataset()
>>> panel.symbol.values.tolist()
[10107, 14593, 99002]
>>> panel.sizes
Frozen({'timestamp': 21, 'symbol': 3})
```

`raw_data_dir_path` 必须以厂商目录 `wrds` 结尾，`reference_dir` 是它旁边的 `_reference` 目录。脚本流程里，同样的转换通过 `quantlab.registry.convert` 完成，它按窗口逐段转换，中断后可以续跑。

面板包含 Tiingo 日频面板的十二个变量，外加 CRSP 的扩展变量：

```python
>>> sorted(panel.data_vars)
['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'anomaly_flag', 'ask', 'bid', 'close', 'close_trade', 'cumfacpr', 'cumfacshr', 'divCash', 'facprc', 'high', 'is_delisting', 'low', 'market_cap', 'numtrd', 'open', 'permco', 'prc_is_bidask', 'ret', 'retx', 'shrout', 'splitFactor', 'volume']
```

价格类和 CRSP 变量都是 float64，证券当天没有交易或尚未存在时为 NaN。扩展变量在下表里概括，完整说明见 `quantlab.dataset.crsp` 的模块 docstring。

| 变量 | 含义 |
| --- | --- |
| `ret`、`retx` | 含股息与不含股息的日总收益；CRSP 没有时为 NaN |
| `shrout`、`market_cap` | 流通股数和市值，单位分别是股和美元 |
| `bid`、`ask`、`prc_is_bidask` | 买卖报价；价格是买卖中间价（没有成交）时 `prc_is_bidask` 为 1.0 |
| `is_delisting` | 携带退市收益的那一行为 1.0 |
| `permco`、`cumfacpr`、`cumfacshr`、`facprc`、`numtrd`、`close_trade` | CRSP 公司编号、累计因子、当日价格因子、成交笔数、收盘成交价 |

### 总收益复权价格

`close` 是 CRSP 日价格的绝对值。`adjClose` 从每个 PERMNO 在窗口内第一个可用收盘价起步，之后按 CRSP 的日总收益连乘，因此包含股息，并且在拆股前后连续。`adjOpen`、`adjHigh`、`adjLow` 用同一个因子缩放，`adjVolume` 用 CRSP 的累计股本因子缩放。`splitFactor` 是当天的拆股比例，`divCash` 是除息日的每股现金分红。Apple 在 2020-08-31 的 4 比 1 拆股体现在 `close` 和 `splitFactor` 里，但不体现在 `adjClose` 里：

```python
>>> aapl = panel.sel(symbol=14593).to_pandas().dropna(subset=["close"])
>>> aapl[["close", "adjClose", "ret", "splitFactor", "divCash"]]
             close    adjClose       ret  splitFactor  divCash
timestamp                                                     
2020-08-06  455.61  455.610000  0.034889          1.0     0.00
2020-08-07  444.45  445.269931 -0.022695          1.0     0.82
2020-08-28  499.23  444.548594 -0.001620          1.0     0.00
2020-08-31  129.04  459.624126  0.033912          4.0     0.00
>>> (aapl["adjClose"] / aapl["adjClose"].shift() - 1).round(6).tolist()
[nan, -0.022695, -0.00162, 0.033912]
```

`adjClose` 的逐日变化等于 `ret`。缺失的收益在连乘中按 0 处理，因为 CRSP 的收益本来就覆盖了回到上一个有效价格之间的空档。复权后的价格水平取决于窗口，因为它以窗口内第一个收盘价为锚；跨不同起始日期的 store 比较时，应比较收益或比值，而不是绝对水平。

### 退市收益

CRSP 把退市收益写在退市当天自己的那一行上，用 `dlydelflg = "Y"` 标记。quantlab 像对待其他行一样把它连乘进去，并通过 `is_delisting` 暴露出来。不会再把 `stkdelists` 里的退市收益叠加一次，否则同一笔损失会被计算两次。以 2008 年 9 月的雷曼兄弟（PERMNO 80599）为例：

```python
>>> leh = panel.sel(symbol=80599).to_pandas()
>>> leh[["close", "adjClose", "ret", "is_delisting"]]
            close  adjClose       ret  is_delisting
timestamp                                          
2008-09-12  3.650  3.650000 -0.135071           0.0
2008-09-15  0.210  0.209999 -0.942466           0.0
2008-09-16  0.300  0.299999  0.428571           0.0
2008-09-17  0.130  0.129999 -0.566667           0.0
2008-09-18  0.052  0.052000 -0.600000           1.0
>>> (leh["adjClose"] / leh["adjClose"].shift() - 1).round(6).tolist()
[nan, -0.942466, 0.428571, -0.566667, -0.6]
```

新版数据里有些退市行给的是清算金额而不是价格（`dlyprcflg = "DA"`，`dlyprc = 0`）。这类行的 `close` 是 NaN，因此不会发布价格为零的成交。

## 常见任务

### 选择保留哪些证券

`security_filter` 按 CRSP 每日的类型列筛选，所以中途变过类型的证券只保留符合条件的那段时期。`equity_common` 保留普通股，包括 REIT 和非美国注册的发行人，剔除 ADR、unit、基金和 ETF。`shrcd_10_11` 复现较早的 `shrcd in (10, 11)` 筛法，它还会剔除非美国发行人。`none` 全部保留。也可以传入基于可过滤列的 `{列名: 允许值}` 字典（见“扩展”）。合成数据里有一只美国股票、一只 ADR、一家非美国发行人、一只基金，还有另一只美国股票：

```python
>>> from dataclasses import replace
>>> def symbols_for(store, security_filter="equity_common", permnos=None):
...     cfg = replace(config, zarr_file_path=store,
...                   security_filter=security_filter, permnos=permnos)
...     CrspStockDataset(cfg).from_raw_data().save()
...     return CrspStockDataset(cfg).read().get_xarray_dataset().symbol.values.tolist()
>>> symbols_for("data/a.zarr")
[10107, 14593, 99002]
>>> symbols_for("data/b.zarr", security_filter="shrcd_10_11")
[10107, 14593]
>>> symbols_for("data/c.zarr", security_filter="none")
[10107, 14593, 86755, 99001, 99002]
```

创建 store 的那次转换会在它旁边写一个 `<store>.crsp_filter_report.json`，列出过滤器删掉了什么；`dropped_by_type` 的键依次是 `sharetype/securitytype/securitysubtype/issuertype/usincflg`。过滤发生在转换阶段，所以改过滤器只需要重新转换，不用重新下载。

```python
>>> import json
>>> report = json.load(open("data/a.zarr.crsp_filter_report.json"))
>>> report["rows_kept"], report["rows_dropped"]
(46, 42)
>>> report["dropped_by_type"]
{'AD/EQTY/COM/CORP/Y': 21, 'NS/FUND/ETF/ACOR/Y': 21}
```

在 `permnos` 里列出的 PERMNO 是显式名单，显式名单优先于过滤器：它们的所有行都会保留，报告里也会记录这次覆盖。

```python
>>> symbols_for("data/e.zarr", permnos=("14593", "86755"))
[14593, 86755]
>>> report = json.load(open("data/e.zarr.crsp_filter_report.json"))
>>> report["roster_overrides"]["rows_rescued"]
21
```

`roster_universe`（`"crsp_sp500"` 或 `"comp_nasdaq100"`）的作用方式相同：在某个 PERMNO 的成分区间内，它的行不受过滤器约束；区间之外，过滤器照常生效。

### 查询 ticker

股份类别（share class）指同一家公司的不同类股票，例如 Berkshire 的 A 和 B、Alphabet 的 GOOG 和 GOOGL。每个类别是独立的证券，有自己的 PERMNO。这与 CRSP 的 `sharetype` 无关，后者说明的是股份的种类（普通股、ADR、unit），过滤器读取的是它。

转换会写出 `<store>.crsp_tickers.json`，这是从 `stksecurityinfohist` 推导出的 `{PERMNO: [{ticker, start, end}]}` 区间表。类别会显示为 `BRK.B`，而退市当天那一行没有自己的 ticker，会沿用前一个名字。`CrspTickerLookup` 回答“这个 PERMNO 在这一天叫什么”：

```python
>>> from datetime import date
>>> from quantlab.dataset.crsp.tickers import CrspTickerLookup
>>> lookup = CrspTickerLookup.beside_store(config.zarr_file_path)
>>> lookup.as_of(13407, date(2022, 6, 8)), lookup.as_of(13407, date(2022, 6, 9))
('FB', 'META')
>>> lookup.as_of(83443, date(2022, 6, 9))
'BRK.B'
>>> lookup.label([13407, 83443, 99999], date(2022, 6, 9))
['META', 'BRK.B', '99999']
```

边车文件缺失或无法读取时，`as_of` 会抛出异常；`label` 从不抛异常，而是退回到 PERMNO 数字，适合日志和报告。

### 选择指数股票池或整个市场

成分数据来自参考层，因此这些调用不需要连接。`CrspMembership` 提供 CRSP 的 S&P 500（`crsp_sp500`，自 1925 年起）和 Compustat 的 Nasdaq-100（`comp_nasdaq100`，自 1995 年起，通过 CCM 映射到 PERMNO）。区间是闭区间，仍在成分内的记录以产品最后一天为终点。

```python
>>> from quantlab.dataset.crsp.reference import CrspReference
>>> from quantlab.dataset.crsp.membership import CrspMembership
>>> from quantlab.dataset.crsp.market import CrspMarketRoster
>>> reference = CrspReference("data/downloads/us_equity/1d/wrds_crsp/_reference")
>>> membership = CrspMembership(reference)
>>> membership.permnos_in_range("crsp_sp500", "2020-01-01", "2020-12-31")
['14593']
>>> membership.permnos_in_range("comp_nasdaq100", "2010-01-01", "2020-12-31")
['14542', '90319']
>>> roster = CrspMarketRoster(reference)
>>> roster.permnos_in_range("2020-01-01", "2020-12-31")
['13407', '14593', '21186', '83443', '90319']
>>> roster.permnos_in_range("2020-01-01", "2020-12-31", security_filter="none")
['13407', '14593', '21186', '83443', '86755', '90319']
```

这里的参考层是很小的合成数据，所以每个列表都很短。`CrspMarketRoster` 读取 `stksecurityinfohist` 里通过过滤器的所有证券，因此全市场名单不需要额外的查询。对真实的 2025 年参考层，默认过滤器在单独一年里给出约 5,500 个 PERMNO，跨 1999 到 2025 年约 16,800 个。

成分关系本身是一个面板 `is_member(timestamp, symbol)`，由成分数据集（`CrspSP500ConstituentDataset`、`CompustatNasdaq100ConstituentDataset`、`CrspMarketConstituentDataset`）在不需要连接的情况下构造，与价格面板使用同一条 PERMNO 轴。见 [constituent.md](constituent.md)。

市场脚本接受同样的窗口参数，外加 `--security-filter`（默认 `equity_common`，也可以是 `shrcd_10_11` 或 `none`）：

```bash
uv run python scripts/wrds/market.py --start 2024-01-01 --end 2024-12-31
```

它会写出 `wrds_crsp_market_1d.zarr` 和上市状态掩码 `wrds_crsp_market_membership.zarr`。指数成分面板由 `index.py` 写出。这两个 store 在 2026-09-25 之前叫 `wrds_crsp_all_*`：已有的 store 手动改名，或者重新运行 `market.py`，从未改动的原始层重新转换。

### 增量更新 store

`--refresh` 从每个 PERMNO 已记录的水位继续，向前延长 `--end` 是受支持的方向。市场 store 的名单在两次刷新之间会增长。脚本使用库的默认转换选项；`quantlab.registry.convert(..., on_new_listing=...)` 决定新 PERMNO 怎么办：`refuse`（默认）直接停止；`widen` 加入新的列，历史部分为 NaN，适合真正的新上市；`rebuild` 会对每个窗口重新稠密化，适合本来就有历史的 PERMNO。

### 加入基准 ETF

让 ETF 与它持有的股票一起排名，等于让它和自己竞争，所以基准放在单独的 store 里。`CrspDatasetConfig.etf_benchmark(permno=...)` 固定了两个关键设置：PERMNO 和 `security_filter="none"`；`qqq_benchmark` 是 QQQ（`QQQ_PERMNO`，86755）的同一配置，`SPY_PERMNO`（84398）是 S&P 500 的 ETF。`scripts/wrds/etf.py --etf spy,qqq --start 1999-01-01` 按 PERMNO 下载 ETF，每只一个 store（`wrds_crsp_spy_1d.zarr`、`wrds_crsp_qqq_1d.zarr`）；其他 ETF 写成 `name=PERMNO`。ETF 从不作为指数或市场面板的一列。

```python
>>> etf = CrspDatasetConfig.qqq_benchmark(
...     zarr_file_path="data/qqq.zarr",
...     raw_data_dir_path=config.raw_data_dir_path,
...     reference_dir=config.reference_dir,
...     start_date="2020-08-03", end_date="2020-08-31")
>>> etf.permnos, etf.security_filter
(('86755',), 'none')
```

### 重建 store

store 是由原始层和参考层派生出来的，所以转换代码或过滤器变了，只需重建，不必访问 WRDS。`CrspStoreRebuilder` 先检查输入是否存在，备份 store，连同五个边车文件一起删除，再重新转换并测量结果。

```python
>>> from pathlib import Path
>>> from quantlab.dataset.crsp.rebuild import CrspStoreRebuilder
>>> rebuilder = CrspStoreRebuilder(config, data_root=".")
>>> result = rebuilder.rebuild(backup_dir=Path("backup"))
>>> result.dims, result.data_var_count
({'timestamp': 21, 'symbol': 3}, 27)
```

### 在使用 Tiingo 的地方换用 CRSP 面板

`CrspStockDataset` 继承自 `StockDataset`，带有 Tiingo 的全部十二个变量，所以读取 `adjClose` 或 `adjVolume` 的因子可以接受任一数据集。区别在于整数的 `symbol` 轴，以及对 ticker 侧选择字段（数据集 config 和因子 config 上的 `symbols`）的拒绝，它们会抛出 `ValueError`；应改用数据集上的 `permnos`。下面是在一个含八只合成证券的 CRSP store 上运行 Alpha101 因子：

```python
>>> from quantlab.base.config import FactorConfig
>>> from quantlab.factor.alpha101 import Alpha101Stock
>>> dataset = CrspStockDataset(config).read()
>>> factor = Alpha101Stock(FactorConfig(
...     window=20,
...     dataset=dataset,
...     mode="batch",
...     data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
...     factor_names=("alpha001",),
...     file_path="data/data/us_equity/1d/alpha101_crsp.zarr",
... ))
>>> features = factor.cal().get_features()
>>> features["alpha001"].isel(timestamp=-1).values.round(3)
array([0.875, 0.625, 0.25 , 0.875, 0.25 , 0.875, 0.25 , 0.5  ],
      dtype=float32)
```

因子、模型和回测各指南（[factor.md](factor.md)、[model.md](model.md)、[backtest.md](backtest.md)）无需改动即可适用。需要时点股票池时，套用成分掩码（[constituent.md](constituent.md)）。

## 扩展

数据集类上的 `EXTRA_VARIABLES` 指定在 Tiingo 的十二个变量之外加入哪些 CRSP 列。子类收窄它就得到更精简的面板，其余转换逻辑继承而来。子类直接使用即可，因为登记表是根据厂商的能力声明来解析数据集类的。

```python
>>> class SlimCrspDataset(CrspStockDataset):
...     EXTRA_VARIABLES = ("ret", "market_cap")
>>> slim = replace(config, zarr_file_path="data/data/us_equity/1d/slim.zarr")
>>> SlimCrspDataset(slim).from_raw_data().save()
>>> sorted(SlimCrspDataset(slim).read().get_xarray_dataset().data_vars)
['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'anomaly_flag', 'close', 'divCash', 'high', 'low', 'market_cap', 'open', 'ret', 'splitFactor', 'volume']
```

新的过滤规则不需要子类：把 `{列名: 允许值}` 字典传给 `security_filter` 即可。它可以使用的列列在 `FILTERABLE_COLUMNS` 里。

## 注意事项

转换的日志输出到 stderr。过滤器删掉了行时出现的警告是正常的，同样的数字也在过滤报告里。

`anomaly_flag` 标记原始 `close` 为零、为负或相对前一根 bar 剧烈跳变的 bar。拆股会让原始 `close` 剧烈变化，所以拆股日会被标记（见上文的 2020-08-31）。收益应从 `adjClose` 或 `ret` 计算。

复权锚点是每个 PERMNO 在窗口内第一个可用的行。把结束日期向后延长，不会改变任何历史值；把 `start_date` 往后移，或往前补充更早的原始行，会重新缩放相关 PERMNO 的整段复权历史，而且不会被检测到；这之后需要重建 store。

原始层记录了它所来自的 CRSP 年度版本，把两个版本混在同一个 raw 根目录里会抛出 `CrspVintageError`。新版本请使用新的原始层。成分数据只回答到该版本最后一天为止。

用户会遇到的错误，按实际抛出的内容引用：

```text
RuntimeError: WRDS_USERNAME environment variable must be set to your WRDS username.
```
设置该环境变量，并把密码写入 `~/.pgpass`。

```text
CrspProductEndError: end_date 2026-06-30 is past the CRSP product end 2025-12-31.
```
年度产品没有更晚的数据。请调低结束日期；脚本会自动截断并打印截断信息。以编程方式调用时可以设置 `kwargs["clip_to_product_end"]`。

```text
ValueError: WrdsCrspDailyAcquisition: symbol 'AAPL' is not a PERMNO.
```
原始层以 PERMNO 为键。请先把 ticker 解析成 PERMNO，例如通过成分名单。

```text
FileNotFoundError: CrspReference: no stksecurityinfohist.parquet under '<dir>'.
```
参考层由 ingest 脚本单独拉取，与日频行情分开。运行脚本，或把 `reference_dir` 指向已有参考表的目录。

```text
ValueError: CrspStockDataset: config.symbols is not selectable on a CRSP panel; got ('AAPL',).
```
请改用 `permnos`。同一个类还会拒绝 `security_filter="common"`（预设只有 `equity_common`、`none` 和 `shrcd_10_11`），也会拒绝空的 `permnos=()`，因为它既可以理解为一个都不要，也可以理解为全部。

```text
ValueError: CrspStockDataset: raw_data_dir_path '<dir>' has basename 'wrds_crsp' but the configured vendor is 'wrds'.
```
让 `raw_data_dir_path` 以 `wrds` 目录结尾。

## 另请参阅

[wrds_taq.md](wrds_taq.md)（共用同一个会话的 WRDS 逐笔数据）、[constituent.md](constituent.md)（成分面板）、[dataset.md](dataset.md)（面板契约）、[chunking.md](chunking.md)（分窗口转换）、[registry.md](registry.md) 与 [acquisition.md](acquisition.md)（下载机制）、[pageledger.md](pageledger.md)（断点续传）。类的 docstring：`CrspStockDataset`、`CrspDatasetConfig`、`WrdsCrspDailyAcquisition`、`CrspReferenceTables`、`CrspMembership`、`CrspMarketRoster`、`CrspSymbology`、`CrspTickerLookup`、`CrspStoreRebuilder`。
