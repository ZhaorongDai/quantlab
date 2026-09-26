# Dataset（数据集）

[English](../dataset.md) | 简体中文

Dataset 负责把 vendor（数据供应商）交付的原始文件转换成 quantlab 后续各阶段共用的面板：一个以 `(timestamp, symbol)` 为索引的 `xarray.Dataset`，以 Zarr 格式落盘。每个市场或每个 vendor 对应一个 `BaseDataset` 或 `MarketDataset` 的子类。因子、模型和回测只读取面板，不接触原始格式。

## 前置条件

原始文件由 acquisition 层下载（见 acquisition 指南）。下面的示例在临时目录里造了一棵很小的合成原始数据树，不需要网络，也不需要任何凭证。这棵树遵循 `StockDataset` 读取的布局：以 vendor 名命名的目录下，每月一个 Parquet 分片，每一行带有 `vendor` 列。

```python
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

logger.remove()  # quantlab 通过 loguru 输出 INFO 日志；这里先关掉

root = Path(tempfile.mkdtemp())
raw = root / "downloads/us_equity/1d/us_all/tiingo"
for month, days in {"2024-01": [2, 3, 4], "2024-02": [1, 2]}.items():
    rows = [
        dict(timestamp=datetime.fromisoformat(f"{month}-{d:02d}"), symbol=sym,
             open=px + d, high=px + d + 1, low=px + d - 1, close=px + d,
             volume=1_000.0)
        for d in days for sym, px in [("AAPL", 100.0), ("MSFT", 300.0)]
    ]
    part = raw / f"month={month}"
    part.mkdir(parents=True)
    pl.DataFrame(rows).with_columns(vendor=pl.lit("tiingo")).write_parquet(part / "part-0.pqt")
```

## 基础

### 面板

所有 dataset 产出的对象形状相同：一个 `xarray.Dataset`，全部数据变量都落在 `timestamp` 和 `symbol` 两个维度上。面板是稠密的：某个标的在某个时间戳没有 bar 时，该单元格在所有变量中都是 NaN，而不是缺少一行。变量沿用 vendor 自己的列名（`open`、`close`、`adjClose`、`Volume` 等），清洗步骤会再加一个布尔变量 `anomaly_flag`。

### 配置

Dataset 由一个 config dataclass 构造。`BaseDatasetConfig` 含所有 dataset 都需要的字段：`zarr_file_path`、`start_date`、`end_date`、`symbols` 和自由格式的 `kwargs` 字典。`DatasetConfig` 在此基础上增加行情面板的字段：`raw_data_dir_path`、`market`、`frequency` 和 `vendor`。Dataset 类不会根据 `market` 或 `frequency` 分支，它们只是 vendor registry 用来挑选转换器的标签。

```python
>>> import dataclasses
>>> from quantlab.base.config import DatasetConfig
>>> from quantlab.dataset.stock import StockDataset
>>> config = DatasetConfig(
...     raw_data_dir_path=str(raw),
...     zarr_file_path=str(root / "data/us_all.zarr"),
...     market="us_equity",
...     frequency="1d",
...     vendor="tiingo",
...     start_date="2024-01-01",
...     end_date="2024-02-29",
... )
>>> ds = StockDataset(config)
>>> ds.config.name
'quantlab.dataset.stock.StockDataset'
```

给 dataset 赋值 config 时，会把 `name` 填成类的点分导入路径，保存下来的 config 就是靠它还原成对象的。缺少 `start_date` 或 `end_date` 时分别取 `1900-01-01` 和 `2100-01-01`，这样日期过滤总有两个端点。两个日期都必须是 ISO `YYYY-MM-DD` 字符串，因为流水线里所有日期比较都是字符串比较。

```python
>>> open_ended = dataclasses.replace(config, start_date=None, end_date=None)
>>> d = StockDataset(open_ended)
>>> d.config.start_date, d.config.end_date
('1900-01-01', '2100-01-01')
>>> bad = dataclasses.replace(config, end_date="2024-2-29")
>>> try:
...     StockDataset(bad)
... except ValueError as exc:
...     print(exc)
...
StockDataset: end_date must be an ISO YYYY-MM-DD date string, got '2024-2-29'. Dates are compared lexicographically throughout this pipeline, so a non-ISO value compares wrong rather than failing to match.
```

### 转换、保存和读取

存储生命周期由三个方法完成。`from_raw_data()` 读取配置范围内的原始文件，运行该 dataset 的清洗步骤，并把结果留在内存里。`save()` 把它写到 `zarr_file_path`。`read()` 之后再打开 Zarr store，并按配置的日期和标的收窄。每个方法都返回 dataset 本身，所以可以链式调用；`get_xarray_dataset()` 返回面板。

```python
>>> ds = ds.from_raw_data()
>>> ds.get_xarray_dataset()
<xarray.Dataset> Size: 466B
Dimensions:       (timestamp: 5, symbol: 2)
Coordinates:
  * timestamp     (timestamp) datetime64[us] 40B 2024-01-02 ... 2024-02-02
  * symbol        (symbol) object 16B 'AAPL' 'MSFT'
Data variables:
    open          (timestamp, symbol) float64 80B 102.0 302.0 ... 102.0 302.0
    high          (timestamp, symbol) float64 80B 103.0 303.0 ... 103.0 303.0
    low           (timestamp, symbol) float64 80B 101.0 301.0 ... 101.0 301.0
    close         (timestamp, symbol) float64 80B 102.0 302.0 ... 102.0 302.0
    volume        (timestamp, symbol) float64 80B 1e+03 1e+03 ... 1e+03 1e+03
    anomaly_flag  (timestamp, symbol) bool 10B False False False ... False False
>>> ds.save()
>>> panel = StockDataset(config).read().get_xarray_dataset()
>>> panel["close"].to_pandas()
symbol       AAPL   MSFT
timestamp               
2024-01-02  102.0  302.0
2024-01-03  103.0  303.0
2024-01-04  104.0  304.0
2024-02-01  101.0  301.0
2024-02-02  102.0  302.0
```

时间轴上只出现原始文件里确实存在的交易日；`2024-01-01` 到 `2024-02-29` 这个窗口不会为没有数据的日子造出行。

### 查看已存储的面板

读取过的 dataset 提供几个开销很小的属性。`time_interval` 是相邻时间戳之间最常见的间隔，所以周末或假期不会改变它。`get_lazyframe()` 以长格式 polars `LazyFrame` 返回同样的数据；`head(n)` 按路径打开 store，最多返回 `n` 行，不影响已加载的面板。

```python
>>> ds = StockDataset(config).read()
>>> ds.symbols, ds.num_symbols
(['AAPL', 'MSFT'], 2)
>>> ds.time_interval
np.timedelta64(86400000000000,'ns')
>>> ds.get_lazyframe().collect().shape
(10, 8)
>>> ds.head(2).collect().columns
['timestamp', 'symbol', 'anomaly_flag', 'close', 'high', 'low', 'open', 'volume']
```

## 常见任务

### 读取时限定日期或标的

`read()` 会应用 `start_date`、`end_date`，以及设置了时的 `symbols`。同一个 store 配不同的 config，得到不同的视图。

```python
>>> feb = dataclasses.replace(config, start_date="2024-02-01", symbols=("MSFT",))
>>> StockDataset(feb).read().get_xarray_dataset()["close"].to_pandas()
symbol       MSFT
timestamp        
2024-02-01  301.0
2024-02-02  302.0
```

### 读取异常标记

清洗在 `from_raw_data()` 内部运行。它检查必需列是否存在，报告空值，并添加 `anomaly_flag`。当某个价格为零或负数，或者 `close` 相对于为正的前一个收盘价变动超过 50% 时，该单元格被标记。数值本身不会被修改、填充或删除，标记只是做记号。这些函数可以对任意面板调用。

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.dataset._support.cleaning import flag_anomalies, validate_schema
>>> close = np.array([[10.0, 5.0], [10.5, 0.0], [20.0, 5.2], [20.5, 5.3]])
>>> panel = xr.Dataset(
...     {name: (("timestamp", "symbol"), close) for name in ("open", "high", "low", "close")}
...     | {"volume": (("timestamp", "symbol"), np.full((4, 2), 1000.0))},
...     coords={"timestamp": pd.date_range("2024-01-02", periods=4), "symbol": ["AAA", "BBB"]},
... )
>>> flagged = flag_anomalies(panel)
>>> flagged["anomaly_flag"].to_pandas()
symbol        AAA    BBB
timestamp               
2024-01-02  False  False
2024-01-03  False   True
2024-01-04   True  False
2024-01-05  False  False
>>> bool(flagged["close"].equals(panel["close"]))
True
```

`BBB` 在 2024-01-03 的收盘价为零，`AAA` 在 2024-01-04 从 10.5 跳到 20.0。缺少必需列是清洗中唯一会抛异常的情形：

```python
>>> validate_schema(panel.drop_vars("volume"))
Traceback (most recent call last):
    ...
ValueError: validate_schema: required column(s) missing from dataset: ['volume']
```

在把 frame 转成 xarray 之前，必须先去掉重复的 `(timestamp, symbol)` 行。`dedup_raw_frame(frame, keep="last")` 在 polars `LazyFrame` 上完成这件事，默认保留最后一行，因为后到的 vendor 文件更可能带有更正；`keep="first"` 则保留较早的一行。

### 导出 KunQuant 数组

`MarketDataset.to_kunquant` 读取 store，返回由连续的 `[time, symbol]` float32 数组组成的字典，外加 symbol 轴和 timestamp 轴。因子层会调用它，也可以直接调用。

```python
>>> inputs, symbols, timestamps = ds.to_kunquant(("open", "close"))
>>> inputs["close"].shape, inputs["close"].dtype
((5, 2), dtype('float32'))
>>> symbols.tolist()
['AAPL', 'MSFT']
```

### 按窗口分段转换

历史很长时，`from_raw_data_chunked()` 每次转换一个月、一个季度或一年，并把每段追加到 store；`update()` 则接着已有的 store 继续。两者的细节见 chunking 指南。

```python
>>> monthly = dataclasses.replace(config, zarr_file_path=str(root / "data/monthly.zarr"))
>>> ds = StockDataset(monthly).from_raw_data_chunked(granularity="month")
>>> result = ds.last_chunk_result
>>> result.windows_planned, result.windows_written, result.rows_written
(2, 2, 5)
```

### 重采样到更粗的 bar

`resample(freq, how)` 返回 dataset 的一个副本，其面板被聚合到更粗的 bar 上，例如分钟 bar 变日 bar。`freq` 取 `1s`、`5s`、`10s`、`15s`、`30s`、`1m`、`5m`、`10m`、`15m`、`30m`、`1h`、`1d` 之一，且必须比 store 自身的 bar 更粗。`how` 为每个变量指定一种方法，可选 `first`、`last`、`max`、`min`、`sum`、`mean`、`count`；也可以只给一个字符串，表示所有变量都用这种方法。NaN 单元格会被跳过。副本与源不共享任何内存，源本身不会被改变。

下面的会话写入一个两天的分钟 store，并通过 `SpotKlineDataset` 读取。

```python
>>> minutes = pd.DatetimeIndex(np.concatenate([
...     pd.date_range(f"2024-01-0{d} 00:00", periods=4, freq="min").values for d in (2, 3)
... ]))
>>> close = np.arange(1.0, 9.0)[:, None] * np.array([[1.0, 10.0]])
>>> xr.Dataset(
...     {"Open": (["timestamp", "symbol"], close - 0.5),
...      "Close": (["timestamp", "symbol"], close),
...      "Volume": (["timestamp", "symbol"], np.ones((8, 2)))},
...     coords={"timestamp": minutes, "symbol": ["AAAUSDT", "BBBUSDT"]},
... ).to_zarr("data/klines.zarr", mode="w")
>>> config = DatasetConfig(raw_data_dir_path="downloads/spot", zarr_file_path="data/klines.zarr",
...                        market="crypto_spot", frequency="1m")
>>> minute = SpotKlineDataset(config).read()
>>> daily = minute.resample("1d", {"Open": "first", "Close": "last", "Volume": "sum"})
>>> daily.get_xarray_dataset()["Close"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0     40.0
2024-01-03      8.0     80.0
>>> daily.get_xarray_dataset()["Volume"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0      4.0
2024-01-03      4.0      4.0
>>> daily.time_interval, minute.time_interval
(np.timedelta64(86400000000000,'ns'), np.timedelta64(60000000000,'ns'))
>>> minute.get_xarray_dataset().sizes["timestamp"], minute.config.resample_freq
(8, None)
```

副本的 config 把这次请求记录在 `resample_freq` 和 `resample_how` 里，因此能经 `get_config()` 和 `load_dataset_from_config` 往返重建。带着这两个字段构造的 dataset 会在 `read()` 时重采样。`save()` 把重采样后的面板写到 `store_path`：与源 store 同目录、名字里带 `_resample_<freq>` 的一个 store；之后带同样字段的 `read()` 会直接打开这个 store，而不再重采样。

```python
>>> daily.config.resample_freq, daily.config.resample_how
('1d', {'Open': 'first', 'Close': 'last', 'Volume': 'sum'})
>>> daily.store_path
'data/klines_resample_1d.zarr'
>>> daily.save()
>>> sorted(p.name for p in Path("data").iterdir())
['klines.zarr', 'klines_resample_1d.zarr']
>>> reader = SpotKlineDataset(dataclasses.replace(
...     config, resample_freq="1d", resample_how={"Open": "first", "Close": "last", "Volume": "sum"}))
>>> reader.read().get_xarray_dataset()["Close"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0     40.0
2024-01-03      8.0     80.0
```

默认按 UTC 时钟切 bar，标签取 bar 的起点，适合以开盘时间打标签的 bar。按交易时段切 bar 的 dataset 可覆盖 `_resample_labels`；`NbboPanelDataset` 按 NYSE 交易时段切分，因此 `"1d"` 给每个时段打上该日期零点的标签，能与日线 store 对齐。

## 扩展

### 新增一个市场数据源

新增一个数据源只需要一个 `MarketDataset` 子类和一个 config。必须实现三个方法。`_raw_data_to_xr` 返回整个配置范围的面板，已去重，`(timestamp, symbol)` 唯一。`_raw_data_to_xr_window` 返回一个日期窗口，给出 `symbols` 时要 reindex 到这些标的；最简单的写法是对整段结果做切片。`_to_kunquant` 把面板映射成数组。下面的例子每个标的读一个 CSV，保存为 `csv_daily.py`。

```python
# csv_daily.py
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.base.data import MarketDataset


class CsvDailyDataset(MarketDataset):
    """每个标的一个 CSV 文件，列为 date,open,high,low,close,volume。"""

    def _raw_data_to_xr(self) -> xr.Dataset:
        frames = []
        for path in sorted(Path(self.config.raw_data_dir_path).glob("*.csv")):
            df = pd.read_csv(path, parse_dates=["date"])
            frames.append(df.rename(columns={"date": "timestamp"}).assign(symbol=path.stem))
        df = pd.concat(frames).drop_duplicates(["timestamp", "symbol"], keep="last")
        return df.set_index(["timestamp", "symbol"]).sort_index().to_xarray()

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None) -> xr.Dataset:
        data = self._raw_data_to_xr().sel(timestamp=slice(start_date, end_date))
        return data if symbols is None else data.reindex(symbol=list(symbols))

    def _to_kunquant(self, data, data_columns):
        data = data.sortby(["timestamp", "symbol"])
        inputs = {c: np.ascontiguousarray(data[c].to_numpy().astype(np.float32))
                  for c in data_columns}
        return inputs, data["symbol"].values, data["timestamp"].values
```

`to_xarray()` 会构造稠密网格，所以下面的 `BBB` 在它没有文件行的那一天得到 NaN。子类不需要其他改动，就能配合流水线的其余部分工作。

```python
>>> csv_raw = root / "csv"
>>> csv_raw.mkdir()
>>> for sym, base, skip in [("AAA", 10.0, 0), ("BBB", 20.0, 1)]:
...     dates = pd.bdate_range("2024-01-02", periods=5)[skip:]
...     px = base + np.arange(len(dates))
...     pd.DataFrame({"date": dates, "open": px, "high": px + 1, "low": px - 1,
...                   "close": px, "volume": 1000.0}).to_csv(csv_raw / f"{sym}.csv", index=False)
...
>>> from csv_daily import CsvDailyDataset
>>> csv_config = DatasetConfig(
...     raw_data_dir_path=str(csv_raw),
...     zarr_file_path=str(root / "csv_daily.zarr"),
...     market="us_equity",
...     frequency="1d",
... )
>>> CsvDailyDataset(csv_config).from_raw_data().save()
>>> ds = CsvDailyDataset(csv_config).read()
>>> ds.get_xarray_dataset()["close"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-02  10.0   NaN
2024-01-03  11.0  20.0
2024-01-04  12.0  21.0
2024-01-05  13.0  22.0
2024-01-08  14.0  23.0
>>> inputs, symbols, timestamps = ds.to_kunquant(("close",))
>>> inputs["close"].shape, symbols.tolist()
((5, 2), ['AAA', 'BBB'])
```

缺少必需方法的子类无法被构造：

```python
>>> from quantlab.base.data import BaseDataset
>>> class Incomplete(BaseDataset):
...     pass
...
>>> Incomplete(csv_config)
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Incomplete without an implementation for abstract method '_raw_data_to_xr'
```

如果原始数据源能在加载前按日期过滤，就让 `_raw_data_to_xr_window` 只读取那个窗口。`StockDataset` 通过裁剪 Parquet 分区做到这一点，内存上限由窗口决定；上面的切片写法只限制了写入的量。

### 非 OHLCV 的 dataset

没有价格列的面板直接继承 `BaseDataset`，使用 `BaseDatasetConfig`，并覆写 `_clean`。默认的 `_clean` 要求 OHLCV 列，所以布尔型成分面板改用自己的校验函数。`clean_membership_panel` 检查 dtype、维度和时间顺序，并原样返回面板。

```python
>>> from quantlab.base.config import BaseDatasetConfig
>>> from quantlab.base.data import BaseDataset
>>> from quantlab.dataset._support.cleaning import clean_membership_panel
>>> class InIndexDataset(BaseDataset):
...     def _raw_data_to_xr(self) -> xr.Dataset:
...         days = pd.date_range("2024-01-02", periods=3)
...         member = np.array([[True, False], [True, True], [True, True]])
...         return xr.Dataset({"is_member": (("timestamp", "symbol"), member)},
...                           coords={"timestamp": days, "symbol": ["AAA", "BBB"]})
...     def _clean(self, data: xr.Dataset) -> xr.Dataset:
...         return clean_membership_panel(data)
...
>>> member_config = BaseDatasetConfig(zarr_file_path=str(root / "member.zarr"))
>>> InIndexDataset(member_config).from_raw_data().save()
>>> InIndexDataset(member_config).read().get_xarray_dataset()["is_member"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-02  True  False
2024-01-03  True   True
2024-01-04  True   True
```

## 注意事项

清洗属于 `from_raw_data()`。`read()` 只负责打开 store 并收窄，所以早先写入的 store 会按保存时的样子返回。

`save()` 会先把面板收窄到 config 窗口，再替换整个 store 目录。写入时 Zarr 可能打印关于 consolidated metadata 的 `ZarrUserWarning`，可以忽略。

`read()` 就地收窄已加载的面板，并且当 dataset 已持有数据时什么也不做。把 `dataset.config` 改成更宽的窗口后再调用 `read()`，得到的仍是收窄后的面板。传入 `overwrite=True` 才会从磁盘重新加载 store。

```python
>>> ds = StockDataset(dataclasses.replace(config, end_date="2024-01-03")).read()
>>> ds.config = dataclasses.replace(config, end_date="2024-02-29")
>>> ds.read().get_xarray_dataset().sizes["timestamp"]
2
>>> ds.read(overwrite=True).get_xarray_dataset().sizes["timestamp"]
5
```

清洗从不填充或修复任何数值。分块转换按窗口逐个清洗，因此跨越窗口边界的价格跳变不会被标记。

读取一个不存在的 store 会抛出 `FileNotFoundError: File .../missing.zarr does not exist.`

`StockDataset` 只读取单个 vendor 的目录。原始数据根目录必须以 vendor 名结尾，并且必须设置 `DatasetConfig.vendor`，否则扫描会被拒绝，例如 `StockDataset: DatasetConfig.vendor is not set, so there is no way to check that ... holds exactly one vendor's data.` 或 `StockDataset: raw_data_dir_path '...' has basename 'tiingo' but the configured vendor is 'alpaca'.` 原始数据树为空或不存在时抛出 `StockDataset: no raw data for vendor 'tiingo' at frequency '1d' under '...'.` 当范围内没有任何月度文件时，`SpotKlineDataset` 抛出 `No CSV file matching the configured date range was found under ...`

重采样后的 dataset 是其源 store 的一个视图。`from_raw_data()`、`from_raw_data_chunked()` 和 `update()` 会拒绝：`SpotKlineDataset.from_raw_data(): a resampled dataset (resample_freq='1d') is a view of its source store and cannot be built from raw files. Build or update the source dataset, then resample it.` `how` 字典必须列出每个变量：`SpotKlineDataset: resample_how does not name ['Open', 'Volume']; every variable of the panel needs a method (or pass one method as a str).` 目标频率不比 store 的 bar 更粗时拒绝：`SpotKlineDataset: resample_freq='1m' (60s) is not coarser than the panel's own bars (60s).` 保存下来的重采样 store 和因子 store 一样是缓存：重建源 store 不会刷新它。删掉它，或从新的重采样副本再 `save()` 一次。

盘中 dataset 用 `XnysSessionCalendar`（`quantlab.dataset._support.session_calendar`）把东部时间窗口转换成每个日期实际的交易所开收盘时间，半日市也考虑在内，结果是不带时区的 UTC 时间戳。

两个内置 dataset 都实现了 `to_kunquant()`。

两行相同的 `(timestamp, symbol)` 到达 `to_xarray()` 时会抛出 `ValueError: cannot convert a DataFrame with a non-unique MultiIndex into xarray`，需要先去重。

## 另请参阅

chunking 指南介绍 `from_raw_data_chunked()`、`update()` 和断点续跑。acquisition 与 registry 指南说明原始文件如何下载、如何根据 config 选择转换器。backend 指南介绍 `XrBackend`，factor 指南说明因子如何读取 dataset。相关模块：`quantlab.base.data`、`quantlab.base.config`、`quantlab.dataset.spot`、`quantlab.dataset.stock`、`quantlab.dataset._support.cleaning` 和 `quantlab.dataset._support.session_calendar`。
