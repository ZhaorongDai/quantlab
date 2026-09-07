# 数据集层（Dataset）

## 一句话

数据集层负责把**各家 vendor 各不相同的原始文件**，变成**全流水线唯一认识的那一种东西**：一个维度恒为 `[timestamp, symbol]` 的稠密 `xarray.Dataset` 面板，并把它在 Zarr 上存取。新增一个市场，等于新增一个 `Dataset` 子类，上层一行都不用改。

## 它在流水线里的位置

上游是 acquisition 层落在磁盘上的原始文件，下游是因子层。数据集层是这条边界上唯一的翻译器——它是**最后一个知道 vendor 长什么样的地方**。

```
                外部行情源（Tiingo / Alpaca / Binance）
                              │
   acquisition 层 ────────────┤   见 example/acquisition.md、example/pageledger.md
                              ▼
   downloads/{market}/{frequency}/{subdir}/{vendor}/month=YYYY-MM/*.pqt   （美股）
   downloads/crypto_spot/1d/spot/monthly/klines/*.csv                     （币安现货）
                              │
                              │   _raw_data_to_xr()   ← 子类唯一必须实现的方法
                              ▼
              去重 dedup → to_xarray() 稠密化 → _clean() 校验/打标
                              │
                              ▼
        ┌─────────────────────────────────────────────────┐
        │  canonical panel : xr.Dataset                    │
        │  dims == (timestamp, symbol)   ← 全流水线的硬约束 │
        └─────────────────────────────────────────────────┘
                              │   save()  /  read()
                              ▼
        data/{market}/{frequency}/*.zarr        （XrBackend，Zarr 落盘）
                              │
     ┌────────────────────────┼─────────────────────────┐
     ▼                        ▼                         ▼
 to_kunquant()          get_lazyframe()            to_nautilus()
 KunQuant 因子           Polars 因子                Nautilus 数据目录
 base/factor.py          base/factor_polars.py      （spot 已实现，stock 未实现）
```

三条出口是三种下游，不是三种实现：`to_kunquant()` 把面板拍成 KunQuant 计算图要的 `[time, symbol]` C 连续 `float32` 数组；`get_lazyframe()` 是 Polars 因子那条路（`base/factor_polars.py:197` 直接调 `dataset.read().get_lazyframe()`）；`to_nautilus()` 写 Nautilus 的 `ParquetDataCatalog`。**只有第一条和第三条需要子类实现**，第二条是基类白送的。

分块摄取（`from_raw_data_chunked()`、`ChunkLedger`、新上市三策略）另见 `example/chunking.md`；指数成分股面板另见 `example/constituent.md`。

## 核心契约

规范形态只有一句话：**一个 `xr.Dataset`，`dims` 恒为 `(timestamp, symbol)`，每个数据变量都铺在这两维上**。`base/backend.py:DataBackend.get_xarray_dataset` 的中文文档已经把它写死了——"流水线固定使用时间戳与标的这两个维度，这是全项目的硬约束，不是本方法的可选项"。

为什么不是 DataFrame？这是 CLAUDE.md 里的硬约束，理由不是审美：

**1. 面板天然是矩阵，因子和模型吃的就是矩阵。** KunQuant 的编译图要的是 `[time, symbol]` 的连续数组，从 xarray 一步 `data[col].to_numpy()` 就是（`dataset/stock.py:_to_kunquant`）；`base/model.py` 假设的张量形状是 `[num_times, num_symbols, num_features]`。如果层间传的是 long-format DataFrame，每一层都要自己 pivot 一次，而 pivot 的列顺序、缺失填充、排序规则会在每一层各写一遍，迟早各写各的。

**2. "没有这一格" 和 "这一格是 NaN" 必须能区分。** long-format 里一个标的当天没交易就是"没有这一行"，跟"数据缺了"长得一模一样。稠密面板把它变成一个显式的 NaN 格子——`dataset/cleaning.py:validate_schema` 正是靠这一点区分**结构性空缺**（必需列全为 null，说明这根 bar 根本不存在）和**异常空缺**（bar 在、某一列却是 null）。这个区分在 DataFrame 上表达不出来。

**3. 多个变量共享同一套坐标。** 一个美股面板有 13 个数据变量（`open/high/low/close/volume/adj*/divCash/splitFactor/anomaly_flag`），它们共用一组 `timestamp` 和一组 `symbol`。xarray 存一份坐标，long DataFrame 把坐标重复 13 遍。

**4. Zarr 的分块与追加是 xarray 的原生能力。** 沿 `timestamp` 增量 append、按 `symbol` 轴加宽、按块读取——这些是 `dataset/backend.py:XrBackend` 直接用 `to_zarr(append_dim=...)` 做的，换成 parquet 要自己造一套。

**但 polars 并没有被赶走，它只是不做层间格式。** 它出现在两个地方：一是子类内部解析原始文件（`pl.scan_csv` / `pl.scan_parquet`，在成为面板*之前*），二是 Polars 因子后端通过 `get_lazyframe()` / `head()` 取数（在面板*之后*）。层内用什么都行，**层与层之间只认 xarray**。

配置这边也做了同一件事的镜像切分（`base/config.py`）：

| 配置类 | 谁用 | 比 `BaseDatasetConfig` 多的字段 |
|---|---|---|
| `BaseDatasetConfig` | `BaseDataset` | —（`zarr_file_path` / `start_date` / `end_date` / `symbols` / `kwargs` / `name`） |
| `DatasetConfig` | `MarketDataset` | `raw_data_dir_path`、`catalog_path`、`market`、`frequency`、`vendor` |
| `ConstituentDatasetConfig` | `IndexConstituentDataset` | `cache_dir`、`as_of` |

规则很硬：**共享基类的方法真的会读的字段才放进 `BaseDatasetConfig`**。一个成分股面板没有 nautilus 目录，就不该有 `catalog_path` 这个字段可填。`tests/test_dataset_hierarchy.py::test_market_only_config_fields_stay_off_the_shared_base` 用集合相等（不是"不包含"）把这三行差值钉死了。

## 从原始文件到面板：一次 read() 都发生了什么

先说清楚一件容易搞混的事：**清洗、去重、稠密化都不在 `read()` 里，它们在 `from_raw_data()` 里**。`read()` 只做两件事——打开 Zarr、按配置窗口收窄。之所以两条路看起来是一条，是因为构造函数在 store 缺失时会自己掉进 `from_raw_data()`。

完整链路，按发生顺序编号：

**A. 构造期（`BaseDataset.__init__`）**

1. `self.data_backend = XrBackend()`，**然后**才 `self.config = config`。顺序是承重的：config 的 setter 会经 `_reset_symbols()` 调到 `read()`，backend 还不存在就 `AttributeError`。（注意这跟 `base/factor.py` 的顺序**正好相反**，那边是先 config 后 backend，两边都对，别去"统一"。）
2. config setter：填 `name = import_path`；`start_date`/`end_date` 为 `None` 时用 `enums/constant.py:Date.START_DATE`（`"1900-01-01"`）和 `Date.END_DATE`（`"2100-01-01"`）兜底；然后把两个日期用 `datetime.date.fromisoformat` **规范成补零 ISO**。这一步不是洁癖：全流水线的日期比较都是**字符串字典序**（`_densify` 里的 `max(start_date, coverage_start)`、`_clamp_coverage_start` 里的 `requested >= coverage_start`），`"2007-2-1"` 不会匹配失败，它会**比错**，然后悄悄跳过某个保护。
3. 只有当 `config.symbols is not None` 时，才把它转成 tuple 并调 `_reset_symbols()`。`_reset_symbols()` 先用 `_stored_symbol_axis()` 只读坐标探一下 store，分三种情况：store 不存在 → `from_raw_data()`；store 有且非空 → `read()`；**store 有但 symbol 轴为空** → 也 `from_raw_data()`。第三种情况必须靠"探"而不能靠 `except`，因为零行 store 的 `read()` 不会抛 `FileNotFoundError`，它会成功，然后在 `_filter()` 里以 `ValueError: could not convert string to float` 炸掉（zarr 把零长 symbol 轴读回成 float64）。

**B. 原始文件 → 面板（`from_raw_data()`）**

4. 一次性交接检查：如果第 3 步的兜底刚刚已经建好过面板、backend 还持着同一个对象、日期窗口也没变，就直接返回，跳过一次重复转换（实测一次 ingest 会转两遍原始树）。这个交接**只对一次调用有效**，进入方法就无条件清空。
5. `_raw_data_to_xr()`——**子类唯一必须实现的方法**。它内部要做完三件事：定位/解析原始文件、**去重**、`to_xarray()`。去重走 `dataset/cleaning.py:dedup_raw_frame(keep="last")`，必须在 `to_xarray()` 之前：非唯一的 `(timestamp, symbol)` MultiIndex 会让 `to_xarray()` 直接抛 `ValueError: cannot convert a DataFrame with a non-unique MultiIndex into xarray`。`keep="last"` 是因为 vendor 的月度重发里，后到的文件更可能是修正后的数据。
6. **稠密化不需要写代码**。`pandas.DataFrame.set_index(["timestamp","symbol"]).to_xarray()` 本身就产出完整的笛卡尔积，缺的格子自动是 NaN。这就是为什么 `dataset/cleaning.py` 里一行 fill/interpolate 都没有——模块开头写得很直白：加 forward-fill 等于**编造流水线从未观测到的数据**。
7. `_clean(data)`。默认实现是 `clean_market_data()` = `validate_schema()` + `flag_anomalies()`。前者对缺列**硬抛**，对 null 只 `logger.warning` 不抛（flag-don't-delete）；后者加一个布尔变量 `anomaly_flag`，在任何 price-like 列 ≤ 0、或 `close` 单步涨跌幅超过 `_EXTREME_JUMP_THRESHOLD`（0.5）处置 True，**从不修改原值**。这是个可覆写的钩子，非 OHLCV 的数据集必须覆写它。
8. `data_backend.to_internal(data)`——面板进内存，此时还没落盘。

**C. 落盘与再读**

9. `save()` → `_filter()` → `data_backend.write()`（`mode="w"`）。注意 `save()` 会先按当前 config 窗口收窄**再**写，且是整目录覆写。
10. `read()` → `XrBackend.read(zarr_file_path)`（`xr.open_dataset`）→ `_filter()`。`_filter()` 做两件事：`filter_by_date("timestamp", start, end)` 和——仅当 `config.symbols` 非 None 时——`filter_by_symbol("symbol", symbols)`。**两者都是就地收窄**，改的是共享的那个 backend 对象。

## 简单用法

### 1. 从已有 Zarr 读一个面板

配置一律经 `config/__init__.py` 的工厂函数构造，路径根由环境变量 `QUANTLAB_DATA_DIR` 决定，默认落在仓库根的 `data/`。

```python
from config import stock_kline_config
from dataset.stock import StockDataset

cfg = stock_kline_config(subdir="us_all", store_name="us_all.zarr", vendor="tiingo",
                         start_date="2026-08-10", end_date="2026-08-20",
                         symbols=["AAPL", "MSFT", "NVDA"])
panel = StockDataset(cfg).read().get_xarray_dataset()
print(panel)
```

真实输出：

```
<xarray.Dataset> Size: 3kB
Dimensions:       (timestamp: 9, symbol: 3)
Coordinates:
  * timestamp     (timestamp) datetime64[ns] 72B 2026-08-10 ... 2026-08-20
  * symbol        (symbol) <U9 108B 'AAPL' 'MSFT' 'NVDA'
Data variables: (12/13)
    adjClose      (timestamp, symbol) float64 216B ...
    adjHigh       (timestamp, symbol) float64 216B ...
    adjLow        (timestamp, symbol) float64 216B ...
    adjOpen       (timestamp, symbol) float64 216B ...
    adjVolume     (timestamp, symbol) float64 216B ...
    anomaly_flag  (timestamp, symbol) bool 27B ...
    ...            ...
    divCash       (timestamp, symbol) float64 216B ...
    high          (timestamp, symbol) float64 216B ...
    low           (timestamp, symbol) float64 216B ...
    open          (timestamp, symbol) float64 216B ...
    splitFactor   (timestamp, symbol) float64 216B ...
    volume        (timestamp, symbol) float64 216B ...
```

`timestamp: 9` 而不是 11：日期窗口是**字典序切片**，落在周末的日子本来就不在 store 的时间轴上。`anomaly_flag` 是 `_clean()` 在落盘之前加上去的，它现在是这个 store 的一部分。

### 2. 不读数据，只探一眼结构（`head`）

`head(n)` 是 `get_lazyframe()` 的有界孪生：**自己按路径打开 store**，不碰 `self.data`，也不触发 `_filter()`。Polars 因子层用它推导计算图会产出哪些列名——曾经这个探针走 `read()`，结果把共享数据集的日期窗口悄悄收窄、把因子的整个回看期吃掉了（RV-01）。

```python
from config import stock_kline_config
from dataset.stock import StockDataset

ds = StockDataset(stock_kline_config(subdir="us_all", store_name="us_all.zarr", vendor="tiingo"))
probe = ds.head(3).collect()
print(probe.select(["timestamp", "symbol", "open", "close", "volume"]))
print("列名:", probe.columns)
print("读过数据了吗? ", hasattr(ds.data_backend, "_data"))
```

真实输出：

```
shape: (3, 5)
┌─────────────────────┬────────┬────────┬────────┬────────────┐
│ timestamp           ┆ symbol ┆ open   ┆ close  ┆ volume     │
│ ---                 ┆ ---    ┆ ---    ┆ ---    ┆ ---        │
│ datetime[ns]        ┆ str    ┆ f64    ┆ f64    ┆ f64        │
╞═════════════════════╪════════╪════════╪════════╪════════════╡
│ 2026-08-07 00:00:00 ┆ A      ┆ 141.43 ┆ 145.97 ┆ 1.525223e6 │
│ 2026-08-07 00:00:00 ┆ AA     ┆ 48.75  ┆ 50.17  ┆ 5.991598e6 │
│ 2026-08-07 00:00:00 ┆ AAAC   ┆ 20.075 ┆ 20.075 ┆ 18.0       │
└─────────────────────┴────────┴────────┴────────┴────────────┘
列名: ['timestamp', 'symbol', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'anomaly_flag', 'close', 'divCash', 'high', 'low', 'open', 'splitFactor', 'volume']
读过数据了吗?  False
```

### 3. 从原始 parquet 现场造一个面板（`from_raw_data`）

这条路把上一节 B 段的每一步都跑一遍，日志正好把它们逐条打出来：

```python
from config import stock_kline_config
from dataset.stock import StockDataset

cfg = stock_kline_config(subdir="us_all", store_name="us_all_demo.zarr", vendor="tiingo",
                         start_date="2026-08-10", end_date="2026-08-14")
ds = StockDataset(cfg)
ds.from_raw_data()
panel = ds.get_xarray_dataset()
print(panel.sizes, list(panel.data_vars))
```

真实输出：

```
INFO     | utils.timer:__enter__:11 - Starting  StockDataset: from pqt
INFO     | utils.timer:__exit__:18 -  StockDataset: from pqt consumed time: 1.09s
INFO     | dataset.cleaning:validate_schema:200 - validate_schema: 73/38165 (0.2%) (timestamp, symbol) cell(s) hold no bar at all — null in every required column. That is the dense panel's cartesian product (D-06), not an anomaly; nulls on those cells are excluded from the counts below.
WARNING  | dataset.cleaning:flag_anomalies:127 - flag_anomalies: flagged 68 anomalous (timestamp, symbol) data point(s) (zero/negative price or extreme jump) — values left unmodified, see `anomaly_flag`.
Frozen({'timestamp': 5, 'symbol': 7633}) ['open', 'high', 'low', 'close', 'volume', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume', 'divCash', 'splitFactor', 'anomaly_flag']
```

`73/38165` 那行就是第 6 步稠密化的直接后果——73 个格子里根本没有 bar。这个数被单独报出来而不是混进 null 告警，是因为在真实分钟面板上它会产生上百万条假警报，而假警报会训练人忽略真警报。

### 4. 面板属性

```python
ds = StockDataset(stock_kline_config(subdir="us_all", store_name="us_all.zarr", vendor="tiingo"))
ds.read()
print("num_symbols =", ds.num_symbols)
print("symbols[:5] =", ds.symbols[:5])
```

真实输出：

```
num_symbols = 7700
symbols[:5] = ['A', 'AA', 'AAAC', 'AAAP', 'AAC']
```

（`ds.time_interval` 这个属性在 `XrBackend` 下是坏的，见"常见坑"第 6 条。）

## 扩展：新增一个市场 = 新增一个 Dataset 子类

### 先选基类

层次是三层，选哪一层取决于**你的数据有没有 bar**：

```
BaseDataset  (ABC)            抽象方法: {_raw_data_to_xr}
├── MarketDataset (ABC)       额外抽象: {_to_kunquant, _to_nautilus}
│   ├── SpotKlineDataset      币安现货 K 线（CSV）
│   └── StockDataset          美股（hive 分区 parquet）
└── IndexConstituentDataset   指数成分股面板（见 example/constituent.md）
    ├── SP500ConstituentDataset
    └── Nasdaq100ConstituentDataset
```

跑一遍就能确认：

```
BaseDataset  : ['_raw_data_to_xr']
MarketDataset: ['_raw_data_to_xr', '_to_kunquant', '_to_nautilus']
```

**共享基类只有一个抽象方法，这是整个设计的核心。** 一个没有 bar、没有 nautilus 目录、没有 KunQuant 输入的数据集（比如布尔成分股面板），只要实现一个 `_raw_data_to_xr()` 就能跑完整套持久化生命周期，而不必背两个"抛 `NotImplementedError`"的空壳方法。

### 子类必须实现的方法及其契约

| 方法 | 在哪层 | 契约 |
|---|---|---|
| `_raw_data_to_xr(self) -> xr.Dataset` | `BaseDataset`（必须） | 返回**已去重、已稠密化**的 `[timestamp, symbol]` 面板，覆盖 `config.start_date..end_date` 全窗口。去重要在 `to_xarray()` 之前，否则直接抛。返回的面板还没被清洗——`_clean()` 由 `from_raw_data()` 负责调。 |
| `_to_kunquant(self, data, data_columns)` | `MarketDataset`（必须） | 返回 `(input_dict, symbols, timestamp)`。`input_dict[col]` 必须是 `np.ascontiguousarray(...).astype(np.float32)` 的 `[time, symbol]` 数组。这里是**列名对齐 KunQuant 词汇表**的地方（`open/high/low/close/volume/amount`），也是复权价替换和 `amount` 合成的地方。 |
| `_to_nautilus(self, data, venue, n_jobs)` | `MarketDataset`（必须） | 返回 `(list[list[Bar]], list[Instrument])`。不打算走事件驱动回测就照 `StockDataset` 的做法抛掉——**抽象方法必须实现，但实现可以是拒绝**。 |

三个**可选覆写的接缝**，不覆写也能跑，覆写了会更好：

| 接缝 | 默认行为 | 什么时候该覆写 |
|---|---|---|
| `_clean(data)` | `clean_market_data()`（OHLCV 校验 + 异常打标） | 列名不是小写 OHLCV（`SpotKlineDataset`），或者数据根本不是 OHLCV 形状（成分股面板） |
| `_raw_axes_in_range()` | 调 `_raw_data_to_xr()` 拿全量再取两个轴 | 原始源能把列投影下推（`pl.LazyFrame`）。默认实现**正确但不省内存**，正是分块要避免的那次分配 |
| `_raw_data_to_xr_window(start, end, symbols)` | 全量稠密化后再切片 | 原始源能把日期谓词下推。不覆写时 `from_raw_data_chunked()` 会打一条 warning 明说"内存收益缺席" |
| `_reset_symbols()` | 构造期从 store 解析 symbol 轴，失败则回落到 `from_raw_data()` | 你的 symbol 轴来自数据源自身、或者回落路径会发网络请求——那就改成 no-op，在 `_raw_data_to_xr()` 里解析 |
| `_widen_fill_values()` | `{"anomaly_flag": False}` | 你的面板有别的非浮点变量。见 `example/chunking.md` |

### 基类已经免费给你的

不要重复造这些：

- **配置生命周期**：日期兜底与 ISO 规范化、`name` 自动写成 `import_path`、`symbols` 归一成 tuple。
- **存取**：`read()` / `save()` / `head(n)` / `get_xarray_dataset()` / `get_lazyframe()` / `get_config()`，以及 `_filter()` 的日期+标的双重收窄。
- **属性**：`num_symbols` / `symbols` / `class_name` / `import_path`。
- **`from_raw_data()` 的编排**：转换 → `_clean()` → 交给 backend，外加那个防止一次 ingest 转两遍原始树的一次性交接。
- **整套分块摄取**：`from_raw_data_chunked()`、sidecar ledger 断点续跑、`NEW_LISTING_STRATEGIES = ("refuse", "rebuild", "widen")` 三种新上市策略、`_pin_append_dtypes()`（把整型变量提升成 float64，否则先写的那个窗口决定 store 的 dtype，后来的 NaN 会被静默 cast 成 0——**在缺数据的地方伪造出一个观测值**）。细节见 `example/chunking.md`。
- **`MarketDataset` 额外送的**：`to_kunquant()` / `to_nautilus()` 的编排、`_write_catalog()`。

**特别注意：Polars 因子那条路完全免费。** `get_lazyframe()` 是基类的透传，所以任何新子类一落地就能被 Polars 因子消费，不需要写任何东西。要接 KunQuant 才需要 `_to_kunquant()`。

### 对照着看：同一个基类下两个差异极大的市场

`SpotKlineDataset`（币安现货）和 `StockDataset`（美股）几乎在每一个可变的点上都不一样，而基类一行都不知道这些差异：

| | `SpotKlineDataset` | `StockDataset` |
|---|---|---|
| 原始格式 | 月度 CSV，**无表头**，列名从 `enums/data.py:BinanceCSVHeaders.SPOT` 补 | vendor 命名空间下的 hive 分区 `*.pqt` |
| 时间戳 | epoch 整数，且 2024 及以前是**毫秒**、2025 起是**微秒**，两条分支 | parquet 原生 datetime |
| 列名大小写 | Title-Case（`Open`/`High`/…） | 小写（`open`/`high`/…） |
| `_clean()` | **覆写**：`validate_schema(required_columns=_RAW_REQUIRED_COLUMNS)` + `flag_anomalies()`。`_RAW_REQUIRED_COLUMNS = ("Open","High","Low","Close","Volume")` | **继承默认** `clean_market_data()` |
| symbol 从哪来 | 文件名 `csv_file.name.split("-")[0]` | 数据里的 `symbol` 列（tick 层则是 hive path segment） |
| 窗口下推 | 无。先按文件名过滤（`utils/file.py:file_date_filter`），再 `pl.concat` 全量 | `_scan_raw()` 下推**两个**谓词：hive 键谓词在 plan 期剪目录，`timestamp` 谓词修窗口边缘 |
| 分块接缝 | 未覆写 → 退化成"全量稠密化后切片"，跑分块时会收到 warning | 覆写了 `_raw_axes_in_range()` 和 `_raw_data_to_xr_window()`，真正拿到内存上界 |
| `_to_kunquant()` | 把 `Quote asset volume` 改名成 `amount` | 丢掉未复权 OHLCV、把 `adj*` 改名顶上，再按需合成 `amount = volume * close`（Tiingo 不给成交额） |
| `_to_nautilus()` | 完整实现（`BarDataWrangler` + joblib 并行） | `raise ValueError("Not finished")` |
| provenance 校验 | 无 | 三重相互加固：① 路径 basename 必须等于 `config.vendor`（防止扫描根开在上一层把两家 vendor 静默合并）；② 扫出来的 `vendor` 列必须唯一且与配置相符——且必须在 dedup **之前**校验，因为 dedup 会把两家的重叠行任意折叠成一行，证据就没了；③ `pl.scan_parquet` 的 `extra_columns`/`missing_columns` 刻意保留会抛的默认值，混 schema 的分片会直接报错 |

从这张表里能读出一条规律：**差异全部落在"原始文件长什么样"和"这个市场的词汇表叫什么"，一条都没有渗进基类**。`tests/test_extensibility_contract.py::test_core_layer_purity_no_market_specific_logic` 用 grep 把这件事钉死：`base/factor.py`、`base/factor_polars.py`、`base/model.py`、`base/backend.py` 的非注释行里不许出现 `SpotKlineDataset`、`StockDataset`、`crypto_spot`、`us_equity` 这四个字符串。

### 完整可跑的最小子类

下面这段是**完整的、从零跑通的**：造几行合成 CSV，写一个最小 `MarketDataset` 子类，`read()` 出一个真的面板。整段可以直接存成文件、用 `uv run python <文件>` 跑（在仓库根目录下跑，或者设 `PYTHONPATH` 指向仓库根）。

```python
"""最小 Dataset 子类：从两个 CSV 文件读出一个 [timestamp, symbol] 面板。"""
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

from base.config import DatasetConfig
from base.data import MarketDataset
from dataset.cleaning import dedup_raw_frame

TMP = Path("/tmp/quantlab_mini_demo")


def make_raw():
    raw = TMP / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "AAA.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01,10.0,10.5,9.8,10.2,1000\n"
        "2024-01-02,10.2,10.9,10.1,10.8,1200\n"
        "2024-01-03,10.8,11.2,10.6,11.0,900\n"
    )
    # BBB 缺 2024-01-02 这一天：稠密化之后它会变成 NaN，而不是被悄悄丢掉
    (raw / "BBB.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01,50.0,51.0,49.5,50.5,300\n"
        "2024-01-03,52.0,53.0,51.5,52.5,400\n"
    )
    return raw


class MiniCsvDataset(MarketDataset):
    """一个市场 = 一个子类。这里只实现三件事。"""

    def _raw_data_to_xr(self) -> xr.Dataset:
        frames = [
            pl.scan_csv(csv)
            .with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%d"),
                pl.lit(csv.stem).alias("symbol"),
            )
            for csv in sorted(Path(self.config.raw_data_dir_path).glob("*.csv"))
        ]
        frame = pl.concat(frames).sort(["timestamp", "symbol"])
        frame = dedup_raw_frame(frame, keep="last")          # 必须在 to_xarray 之前
        pdf = frame.collect().to_pandas().set_index(["timestamp", "symbol"])
        return pdf.to_xarray()                                # 稠密化在这一步自动发生

    def _to_kunquant(self, data, data_columns):
        data = data.sortby(["timestamp", "symbol"])
        inputs = {
            col: np.ascontiguousarray(data[col].to_numpy().astype(np.float32))
            for col in data_columns
        }
        return inputs, data["symbol"].values, data["timestamp"].values

    def _to_nautilus(self, data, venue="BINANCE", n_jobs=16):
        raise NotImplementedError("MiniCsvDataset 不走 nautilus 这条路")


if __name__ == "__main__":
    raw = make_raw()
    cfg = DatasetConfig(
        raw_data_dir_path=str(raw),
        zarr_file_path=str(TMP / "mini.zarr"),
        catalog_path=str(TMP / "catalog"),
        market="us_equity",
        frequency="1d",
    )
    ds = MiniCsvDataset(cfg)
    ds.from_raw_data().save()

    panel = MiniCsvDataset(cfg).read().get_xarray_dataset()
    print(panel)
    print()
    print("close ---")
    print(panel["close"].to_pandas())
    print()
    print("symbols =", MiniCsvDataset(cfg).read().symbols)
```

真实输出：

```
INFO     | dataset.cleaning:validate_schema:200 - validate_schema: 1/6 (16.7%) (timestamp, symbol) cell(s) hold no bar at all — null in every required column. That is the dense panel's cartesian product (D-06), not an anomaly; nulls on those cells are excluded from the counts below.
INFO     | utils.timer:__enter__:11 - Starting MiniCsvDataset: save
INFO     | utils.timer:__exit__:18 - MiniCsvDataset: save consumed time: 0.16s
<xarray.Dataset> Size: 302B
Dimensions:       (timestamp: 3, symbol: 2)
Coordinates:
  * timestamp     (timestamp) datetime64[ns] 24B 2024-01-01 ... 2024-01-03
  * symbol        (symbol) StringDType() 32B 'AAA' 'BBB'
Data variables:
    anomaly_flag  (timestamp, symbol) bool 6B ...
    close         (timestamp, symbol) float64 48B ...
    high          (timestamp, symbol) float64 48B ...
    low           (timestamp, symbol) float64 48B ...
    open          (timestamp, symbol) float64 48B ...
    volume        (timestamp, symbol) float64 48B ...

close ---
symbol       AAA   BBB
timestamp
2024-01-01  10.2  50.5
2024-01-02  10.8   NaN
2024-01-03  11.0  52.5

symbols = ['AAA', 'BBB']
```

四件事值得注意：

1. **`anomaly_flag` 是白送的**，我没写任何清洗代码——继承的 `_clean()` 默认值把它加上了。
2. **`1/6 (16.7%)` 那条日志**精确指出了 BBB 缺的那一格，并且明说这是笛卡尔积的设计而非异常。
3. **`BBB` 在 2024-01-02 是 NaN 而不是消失**——这正是稠密面板相对 long-format 的关键差别。
4. **`volume` 是 `float64` 而不是 `int64`**：稠密化引入的那个 NaN 把整列升了型。dtype 取决于**这个窗口的稠密程度**，这就是 `_pin_append_dtypes()` 存在的理由（见 `example/chunking.md`）。

如果你的数据根本不是 OHLCV（比如一个布尔面板），就直接继承 `BaseDataset`，实现 `_raw_data_to_xr()`，并覆写 `_clean()` 为恒等函数——`tests/test_dataset_hierarchy.py` 里的 `PanelDataset` 就是这么一个只有 30 行的完整例子，它跑通了 `from_raw_data() → save() → read() → get_xarray_dataset()` 全程。

### `tests/test_dataset_hierarchy.py` 会拦住哪些错误做法

这个文件是扩展契约的**可执行版本**。它有 7 个测试，其中好几个是**源码内省**而不是行为测试——因为这些不变式今天在运行时不会失败，破坏它们的代价要到很久以后、在下一个新数据集上才显形。它把那个失败挪到了测试时。

跑一下是绿的：

```
$ uv run pytest tests/test_dataset_hierarchy.py -q
.......                                                                  [100%]
7 passed, 1 warning in 0.99s
```

它拦住的是：

1. **把 nautilus/KunQuant 成员挪到共享基类**（`test_market_only_members_stay_on_market_dataset`）。`to_kunquant`、`_to_kunquant`、`to_nautilus`、`_to_nautilus`、`_write_catalog` 五个名字必须在 `MarketDataset.__dict__` 里、必须不在 `BaseDataset.__dict__` 里。一旦漏上去（尤其是漏成 `@abstractmethod`），每个未来的非行情数据集都要背一个没意义的 `raise NotImplementedError`。
2. **给共享基类的抽象方法集合加成员**（`test_base_dataset_is_abstract_and_market_dataset_subclasses_it`）。它用**集合相等**断言 `BaseDataset.__abstractmethods__ == {"_raw_data_to_xr"}`。
3. **把市场专属字段搬进 `BaseDatasetConfig`**（`test_market_only_config_fields_stay_off_the_shared_base`）。同样用集合相等，正向声明差值恰好是那五个/两个字段。承重的那个是 `catalog_path`——一个没有 catalog 的数据集，**不该有办法拿到一个 catalog 路径**。
4. **把 `BaseDataset.__init__` 里两行赋值调换顺序**（`test_base_dataset_init_assigns_the_storage_backend_before_the_config`）。它真的去 `inspect.getsource` 里找 `"self.data_backend ="` 和 `"self.config = config"` 的位置。它还特意写明：这跟 `tests/test_factor_hierarchy.py` 的同名断言**正好相反**，两边都对，**不要去"统一"**。
5. **把 `_reset_symbols()` 从共享基类推下去，或者让 `MarketDataset` 覆写它**（`test_reset_symbols_seam_suppresses_construction_time_io`）。这个测试同时用行为证明了接缝有效：`PanelDataset` 把它覆写成 no-op 之后，即使 store 路径不存在，构造依然成功、调用方传的 symbols 元组原样保留、`_raw_data_to_xr()` 一次都没被调到。对一个原始源是远程抓取的数据集，**"构造对象"不该发出网络请求**。
6. **回归到 `BaseDataset` 就能干完全套生命周期**（`test_non_market_dataset_round_trips_through_base_dataset`）。它还额外钉了 dtype：布尔变量必须以 `bool` 而不是 float 或 int8 走完 Zarr 往返，面板也不能被转置——这两种错误在下游读起来都像一个"貌似合理但错误的"标的池掩码。
7. **把 `Dataset` 这个旧名字作为别名加回来**（`test_base_data_module_exposes_only_the_two_split_classes`）。旧名字是被**退休**了，不是被别名了。如果你 `from base.data import Dataset` 失败到这里：有 bar 有 catalog 的用 `MarketDataset`，其他一律 `BaseDataset`。

## 常见坑

**1. `read()` 有缓存，`_filter()` 是就地收窄——窗口只能变窄，不能变宽。**
`XrBackend.read()` 在 `self.data` 已存在且 `overwrite=False` 时直接早返回，然后 `_filter()` 把**新**窗口套在**已经被收窄过**的数据上。

```python
cfg = stock_kline_config(subdir='us_all', store_name='us_all.zarr', vendor='tiingo',
                         start_date='2026-08-10', end_date='2026-08-14')
ds = StockDataset(cfg)
print("窄窗口:", ds.read().get_xarray_dataset().sizes)
ds.config.start_date = '2026-08-07'; ds.config.end_date = '2026-09-04'
print("改宽 config 后再 read():", ds.read().get_xarray_dataset().sizes)
print("read(overwrite=True):   ", ds.read(overwrite=True).get_xarray_dataset().sizes)
```

真实输出：

```
窄窗口: Frozen({'timestamp': 5, 'symbol': 7700})
改宽 config 后再 read(): Frozen({'timestamp': 5, 'symbol': 7700})
read(overwrite=True):    Frozen({'timestamp': 21, 'symbol': 7700})
```

要重新放宽必须 `read(overwrite=True)`。只是想探一眼 store 就用 `head(n)`，它压根不碰 `self.data`。

**2. `save()` 会先 `_filter()` 再整目录覆写。**
`save()` 的第一件事是 `self._filter()`，第二件事是 `data_backend.write()`，而 `write()` 默认 `mode="w"`。所以拿一个窄窗口 config 的实例去 `save()`，会把 store **整个换成**那个窄窗口。要增量追加就走 `from_raw_data_chunked()`（见 `example/chunking.md`），不要用 `save()`。

**3. 一旦传了 `symbols`，构造函数就会做 I/O，标的写错会在构造处炸。**

```python
cfg = stock_kline_config(subdir='us_all', store_name='us_all.zarr', vendor='tiingo',
                         symbols=['AAPL', 'NOT_A_TICKER'])
StockDataset(cfg)   # 就这一行
```

真实的调用栈（只留仓库内的帧）：

```
  File "dataset/stock.py", line 54, in __init__
  File "base/data.py", line 92, in __init__          # self.config = config
  File "base/data.py", line 174, in config           # config setter
  File "base/data.py", line 262, in _reset_symbols
  File "base/data.py", line 278, in read
  File "base/data.py", line 132, in _filter
  File "dataset/backend.py", line 389, in filter_by_symbol
  ... 之后进入 xarray 的 .sel ...
KeyError: "not all values found in index 'symbol'"
```

`symbols=None`（分块摄取那条路）则完全不会触发 `_reset_symbols()`。

**4. 零行 store 不会抛 `FileNotFoundError`，它会被当成"没有 store"。**
一次抓空的采集、一次中断的分块摄取，都会留下一个 `timestamp: 0, symbol: 0` 的 store。`_reset_symbols()` 用坐标级探针识别这种情况，然后回落到 `from_raw_data()`：

```
WARNING | base.data:_reset_symbols:243 - StockDataset data not found, try to read from csv (.../stock_alpaca.zarr: the store at that path is present but its symbol axis is EMPTY (a zero-row panel from an earlier run))
INFO    | utils.timer - Starting  StockDataset: from pqt
构造后 config.symbols = ('AAPL',)
```

这就是为什么这里不能用 `try/except FileNotFoundError` 代替探针——零行 store 的 `read()` 会**成功**，然后在更里面炸出一个跟"没数据"毫无关系的 `ValueError`。

**5. 非行情数据别继承默认的 `_clean()`。**

```python
panel = xr.Dataset({"is_member": (["timestamp","symbol"], ...)}, ...)
clean_market_data(panel)
```
```
ValueError: validate_schema: required column(s) missing from dataset: ['open', 'high', 'low', 'close', 'volume']
```

即使绕过这一关，`flag_anomalies()` 还会给一个只有布尔变量的面板再挂一个全 False 的 `anomaly_flag`——**盘面尺寸翻倍，记录的信息为零**。用 `dataset/cleaning.py:clean_membership_panel()`，或者写你自己的。

**6. `time_interval` 属性在 `XrBackend` 下是坏的。**
`BaseDataset.time_interval` 写的是 `get_xarray_dataset(["timestamp"]).diff(...).to_series().mode()`，但 `XrBackend.get_xarray_dataset()` **完全忽略 `indexes` 参数**，直接返回整个 `Dataset`。于是两个问题接连出现：

```
TypeError: numpy boolean subtract, the `-` operator, is not supported ...   # .diff 撞上 anomaly_flag
AttributeError: 'Dataset' object has no attribute 'to_series'               # 去掉布尔变量之后
```

（两条都是在 `data/data/us_equity/1d/us_all.zarr` 上真跑出来的。）它目前唯一的调用点是 `dataset/spot.py:_xr_to_bars`，也就是 nautilus 那条路，所以主线不受影响——但**别在新代码里用它**，除非先修好。

**7. 日期必须是补零 ISO，否则不是匹配失败而是比错。**

```
ValueError: Mini: start_date must be an ISO YYYY-MM-DD date string, got '2024-1-2'. Dates are compared lexicographically throughout this pipeline, so a non-ISO value compares wrong rather than failing to match.
```

config setter 在边界上一次性拦掉了它，所以下游所有比较可以放心写成普通字符串比较。

**8. 数值列的 dtype 取决于窗口的稠密度。**
一个"每个标的每天都有 bar"的窗口，`volume` 会保留 pandas 的 `int64`；只要有一个缺口，为了放 NaN 就升成 `float64`。分块写入时这意味着 store 的 dtype 由**碰巧第一个被写进去的窗口**决定，而后来的 float64 NaN 写进 int64 变量会被静默 cast 成 0——在缺数据的地方伪造出一个观测值。`_pin_append_dtypes()` 通过把整型统一提升成 float64 让这个失败不可达（布尔的 `anomaly_flag` 例外，它是标志不是测量值）。

**9. 清洗里永远不要加 fill / interpolate。**
`dataset/cleaning.py` 的模块文档写死了这一条：那等于编造流水线从未观测到的数据。异常只**打标**不修正（`anomaly_flag`），空缺只**报告**不填补。想改这个行为之前，先想清楚你是打算让一个 NaN 在三层之外变成一个看起来很正常的因子值。
