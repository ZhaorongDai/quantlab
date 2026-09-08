# 存储后端（DataBackend）

## 一句话

`DataBackend` 是"数据存在哪里"这个问题的唯一答案所在地——上层的数据集、因子、模型都只持有一个后端实例、只调用它声明的那几个方法，所以把 Zarr 换成 Parquet、换成 CSV、换成任何东西，上层一行都不用改。

## 为什么要把"存在哪里"和"数据是什么"分开

这个抽象买到的东西，看三个实现的差异最清楚。它们差得越远，说明这层契约越值钱。

| 实现 | 介质 | `self.data` 是什么 | 惰性吗 |
|---|---|---|---|
| `XrBackend`（`quantlab/dataset/backend.py`） | Zarr 目录 | `xr.Dataset` | 否，`xr.open_dataset` 之后基本就在内存里 |
| `PlBackend`（`quantlab/dataset/backend.py`） | 单个 Parquet 文件 | `pl.LazyFrame` | 是，`scan_parquet` 全程惰性 |
| `MlBackend`（`quantlab/ml_model/backend.py`） | joblib 序列化文件 | 一个 Python 对象（模型） | 无所谓 |

`MlBackend` 实现的是另一个 ABC——`quantlab/base/backend.py:ModelBackend`，它跟 `DataBackend` 是对称的两半：一个管数据落在哪，一个管模型落在哪。它整个类只有十几行，没有任何维度、坐标、时间轴的概念，`read` 就是 `joblib.load`。它能和 `XrBackend` 长在同一套设计里，恰恰是因为契约里没有一句话假设"数据是个带 timestamp/symbol 的面板"。

这层分离在项目里是真的被用起来的，不是纸面上的：

- `quantlab/base/data.py:BaseDataset` 和 `quantlab/base/factor.py` 在 `__init__` 里各自 `self.data_backend = XrBackend()`，之后所有读写都走 `self.data_backend.xxx`，没有一处直接 `to_zarr`。
- `quantlab/acquisition/universe.py:UniverseCatalog` 用的是 `PlBackend()`——因为美股 universe 参考表是"元数据"不是"流水线面板"（`quantlab/config/__init__.py:universe_config` 的注释写明了这个决定），它需要的是 parquet 长表而不是 Zarr 面板。同一套 `read/write/filter_by_*` 调用，换了个介质就成立。
- `quantlab/base/model.py:BaseModel` 也有 `self.data_backend = XrBackend()`，用它来装训练集合并后的面板（`collect()` → `to_internal`）。注意训练用的这份数据从来没落过盘，是 `to_internal` 直接从内存接管的——这条"不经磁盘也能进流水线"的路径就是 `to_internal` 存在的理由。

反过来说，如果没有这层：`BaseDataset` 里会散落 `xr.open_dataset` / `to_zarr` / `sel`，`UniverseCatalog` 里会散落 `scan_parquet`，而 `head()` 那个探查语义（下面细说）会在每个调用点各写一遍、各写错一遍。

## 核心契约

`quantlab/base/backend.py:DataBackend` 声明了 8 个抽象方法，加上一个 `data` 属性。

**`data` 属性**：还没读就访问，直接 `AttributeError("Please cal 'read' or 'to_internal' first.")`，不返回空值。理由值得记住：一个空数据集会被下游当成"这段时间确实没有行情"继续算下去，错误跑到很远才暴露。宁可当场炸。

**`read(path, **kwargs) -> Self` / `write(path, **kwargs) -> Self`**：磁盘 ↔ 内存，都返回 `self`，所以能串成链。

**`to_internal(data) -> Self`**：直接接管一份内存里的数据，跳过磁盘。

**`get_xarray_dataset(indexes=None) -> xr.Dataset`**：转成全项目唯一的层间交换格式。**`indexes` 就是结果的索引维度**——返回的 `Dataset` 的 `dims` 恰好是它，顺序也照给。铺在 `indexes` 之外维度上的数据变量会被丢掉，用不到的维度连同坐标一起丢掉，剩下的 `transpose` 成给定轴序；请求一个不存在的维度会当场 `ValueError`（消息里列出实际有哪些维度）。传 `None` 表示「不做形状要求，原样给我」，这也是全仓大多数调用点的写法。

全项目实际只会传 `["timestamp", "symbol"]` 或者什么都不传，因为「时间戳 + 标的」是 CLAUDE.md 里的硬约束；但「传了就得算数」是这个方法自己的契约，不是那条约束的推论。

**`get_lazyframe() -> pl.LazyFrame`**：给 Polars 因子用的视角。

**`filter_by_date(col, start, end) -> Self` / `filter_by_symbol(col, symbols) -> Self`**：**就地**收窄。这两个字是契约的一部分，不是实现细节——所有共享这个后端实例的调用方都会看到收窄后的数据。想看一眼而不影响别人，用 `head()`。

### `head()` 的三条实现义务

`head(path, n) -> pl.LazyFrame` 是"有界读取"：最多 n 行，用来在不加载全量数据的前提下探查存储的结构。它的 docstring 写得比其他任何一个方法都长，因为它是一条真实事故（RV-01）的修复结论。转成白话：

**第一，不许物化全量。** 这是它存在的全部意义。一个"先全读进来再切前 n 行"的实现完全满足签名，同时彻底废掉了这个方法。

**第二，不许碰 `self.data`——连赋值都不许。** 这一条是最容易写错的，因为**同一个接口上的 `filter_by_date`/`filter_by_symbol` 就是就地改 `self.data` 的**。照着它们的样子写 `head`，就会静默截断一个被所有人共享的 store。

**第三，路径不存在时必须在调用当下抛 `FileNotFoundError`**，而不是等到 `.collect()` 才炸。`PlBackend.head` 的注释说得很直白：`scan_parquet` 在缺文件时只在 collect 时失败，那时离出错的那次调用已经很远了。

还有一条设计上的选择，写在签名里：**`head` 自己拿 path 开 store，不读 `self.data`**（path 在前，跟 `read(path, **kwargs)` 对齐）。这不是为了方便，是 RV-01 的修复本身。事故链条是这样的：

1. `BaseDataset.read()` 会跑 `_filter()`，而 `_filter()` 调 `filter_by_date` —— 就地把 `data_backend.data` 收窄了；
2. `XrBackend.read()` 开头有个缓存早退：`if not overwrite and hasattr(self, "data"): return self` —— 于是那次收窄在后面每一次 `read()` 里都活着；
3. `quantlab/base/factor_polars.py:_get_factor_names` 只是想探一下计算图产出什么列，它跑在 `_reset_dataset_config()` 把窗口按因子 `window` 天数放宽**之前**，而 `filter_by_date` 只会收窄不会放宽；
4. 结果：一个只想看 schema 的探针，静默地把因子整个 lookback 窗口砍掉了。因子列悄悄变成部分 NaN，没有任何地方报错。

这条链现在可以当场复现出来（见下面"简单用法"第 2 段的对照组）。

最后一个问题：为什么 `head` 是抽象方法，而不是 `get_lazyframe(limit=n)` 上的一个关键字？docstring 给的论证是**强制力**：ABC 会让一个漏掉有界读取的后端**根本无法实例化**；而一个可选关键字，靠普通继承就"满足"了，只在某个恰好传了这个参数的调用点才失败。这个区别在下面"扩展"一节里是能跑出来看的。

## Zarr 后端：一次 write 和一次 append 分别发生了什么

`XrBackend` 有两条落盘路径，语义完全不同。

### `write()`：整库替换

```python
kwargs.setdefault("mode", "w")
self.data.to_zarr(path, **kwargs)
```

`mode="w"` 替换整个 store 目录。这有个连带后果，`quantlab/base/chunking.py:ChunkLedger` 的注释专门记了一笔：断点续传用的 ledger 必须写在 store **旁边**而不是里面，因为 `write()` 会把整个目录换掉，恰恰是"需要续传"的那个操作会毁掉记录续传进度的东西。

### `append()`：沿 `append_dim` 延长

`append()` 一个方法承担两件事——store 不存在就建，存在就延长：

```python
if not target.exists():
    kwargs.setdefault("encoding", self._append_encoding(append_dim))
    self.data.to_zarr(path, mode="w", **kwargs)
    return self

self._assert_append_compatible(path, append_dim)
kwargs.pop("encoding", None)   # append 上给 encoding，xarray 直接拒绝
self.data.to_zarr(path, mode="a", append_dim=append_dim, **kwargs)
```

**chunk 网格是在第一次写的时候被钉死的。** `_append_encoding` 给每个带 `append_dim` 的变量算出 chunk 形状：append 维上是 `min(XrBackend.APPEND_DIM_CHUNK, size)`（`APPEND_DIM_CHUNK = 512`，`quantlab/dataset/backend.py:34`），其余每一维取该维的完整长度。

为什么要显式钉？因为不给 `encoding` 的话，zarr 会**拿第一个窗口自己的长度当 chunk 大小**。于是 store 的物理布局取决于"谁碰巧第一个被写进去"——第一批是 700 天就 700，是 90 天就 90——之后每一次长度不同的追加（一个短交易年、一个不完整的末月）都跟磁盘上的网格错开。钉一个固定值，布局才是 **store 的属性**而不是**第一个窗口的属性**。

跑出来看（本节所有输出都是真跑的）：

```
建库后 close 的 chunk: (512, 2)
APPEND_DIM_CHUNK = 512
追加后 shape / chunk: (790, 2) (512, 2)
裸 to_zarr 建库的 chunk: (700, 2)
```

最后一行是对照组：同样 700×2 的面板，不给 encoding 直接 `to_zarr`，chunk 就是 `(700, 2)`。

> 窗口是怎么切出来的、ledger 怎么记录断点、切块对清洗有什么代价——那是分块摄取的故事，见 [`chunking.md`](chunking.md)。这里只讲存储介质这一侧：**一次 append 落到磁盘上会发生什么**。

### 为什么 append 前必须有坐标守卫

`_assert_append_compatible` 在**不可逆的写入之前**检查两件事：每个非 append 维的坐标标签必须和 store 完全一致；每个同名变量的 dtype 必须和 store 一致。

理由是：`to_zarr(mode="a", append_dim=...)` 这两条**一条都不查**，而它出错的方式是静默的。

**第一种静默腐蚀：坐标标签被覆写。** store 里存着 `{A, XYZ}`，进来的窗口是 `{A, ARM}`（一个退市 + 一个新上市，**数量都没变**）。裸 `to_zarr` 会成功，然后把 symbol 坐标改写成 `['A', 'ARM']`，XYZ 已经写进去的历史就挂到了 ARM 名下。代码注释里记了这次实测（`quantlab/dataset/backend.py:113-114`，measured 2026-09-06：`rows [1.0, 3.0] were written for XYZ but are now labelled: ARM`）。原样复现：

```
裸 append 之后的 symbol 轴: ['A', 'ARM']
原本属于 XYZ 的历史 [1.0, 3.0] 现在挂在: [1. 3.] -> 标签 ARM，没有任何报错
```

而 `XrBackend.append()` 对同一件事的反应是：

```
append 拒绝: XrBackend.append: refusing to append to /tmp/.../guarded.zarr -- the
'symbol' coordinate does not match the store (2 incoming label(s) vs 2 stored).
Zarr would OVERWRITE the stored labels without complaint, silently re-attributing
every previously written row. Pin the 'symbol' axis over the whole range before the
first window, the way BaseDataset.from_raw_data_chunked() does.
```

注意 `2 incoming vs 2 stored`——数量一致，所以任何靠"对比长度"的检查都抓不到，必须逐标签比。

**第二种静默腐蚀：dtype 被静默转换。** store 里是 int64 的 `volume`，追加一个 float64 的 NaN 进去，zarr 不吭声地转成 int：

```
裸 append 后 volume dtype: int64
那个缺失值现在是: [0 7]
```

那个 `0` 是**凭空捏造的一次观测**，出现在数据本来缺失的地方。事后从 store 里根本看不出来。`XrBackend.append` 的第二条守卫拦的就是它：

```
append 拒绝: XrBackend.append: refusing to append to /tmp/.../dtype.zarr -- variable
'volume' has dtype float64 but the store holds int64. Zarr would cast silently, and a
float NaN cast into an integer store becomes 0: a fabricated observation where the
data was missing.
```

两种腐蚀的共同点：**事后从 store 本身完全看不出来**。这就是为什么检查必须放在写之前。

### `widen_and_append()`：花名册长大了的显式入口

守卫一旦生效，"新增了一只票"这种日常情况也会被拒。`widen_and_append` 是绕过这个拒绝的**显式 opt-in**——注意是绕过，不是削弱：

它把 store 和进来的窗口**双双**reindex 到 `sorted(stored | incoming)`，然后调用**没有任何改动的** `append()`。守卫照跑，而且是靠构造满足的，不是靠豁免。`tests/test_symbol_axis_widening.py` 的模块 docstring 把这件事说得很清楚，并且专门在隔壁 `tests/test_chunked_ingest.py` 钉了"没 opt-in 的调用方仍然被拒"这一半。

```
widen 后的 symbol 轴: ['A', 'ARM', 'XYZ']
symbol          A    ARM  XYZ
timestamp
2022-01-04    0.0    NaN  1.0
2022-06-15    2.0    NaN  3.0
2023-01-04  900.0  901.0  NaN
```

XYZ 的历史还在 XYZ 名下（退市之后新行是 NaN），ARM 的历史段是 NaN，新行三只票各归各位。

存储侧还有三个细节值得知道：

- **写前守卫，共三条**（`widen_symbol_axis` 的 docstring 按顺序列出）：崩溃残留的 `.superseded.tmp` 侧车 + `path` 下无 store → 拒绝并指名手工恢复命令，不自动恢复（"哪个目录是权威的，不是这个方法该替你决定的"）；目标轴必须是存储轴的**超集**（`reindex` 会悄悄删掉没点名的标签，删完之后跟"从来没有过"无法区分）；带 `symbol` 维的变量必须是浮点，否则必须在 `fill_values` 里点名（不填的 `reindex` 会把 bool/int64 **upcast 成 float64+NaN**，等于对一个活着的 store 做静默 schema 变更）。
- **换库是 rename 三步走**：先把原 store 改名到 `XrBackend.SUPERSEDED_SUFFIX`（`.superseded.tmp`），再把写好的 `XrBackend.WIDENING_SUFFIX`（`.widening.tmp`）改名到位，最后删掉旧的。两次 rename 同父目录、因而是原子的。**先改名旧的**是刻意的：中间崩了的话 `path` 下什么都没有，下一次读会大声失败，而不是把一个改写到一半的 store 当权威。
- **重写走的仍然是 `_append_encoding`**，所以 chunk 规则单点定义。`tests/test_symbol_axis_widening.py:test_the_chunk_grid_survives_a_widen` 用 600 个时间戳（刻意大于 512，否则编码与不编码的 chunk 大小会撞在一起）钉住这一点，并记了实测：忘了传 `encoding=` 的话，重写出来的 store 会继承源 store 的 encoding，三只票的数组 chunk 回到 `(512, 2)`。
- **代价是写在文档里的**：这里没装 dask，`xr.open_zarr` 给的是惰性索引数组，`.load()` 会把**整个 store** 拉进内存——正是分块摄取想避免的那笔分配。库大到装不下时，改用 `on_new_listing="rebuild"`（`quantlab/base/data.py`），它逐窗重建，而且能从原始数据里找回新票的**真实**历史，而不是回填 NaN。

## 简单用法

以下每段都真跑过（`uv run python <文件>`，从仓库根目录）。为了篇幅只贴关键输出。

### 1. 一个完整的往返

```python
import numpy as np, pandas as pd, xarray as xr
from quantlab.dataset.backend import XrBackend

panel = xr.Dataset(
    {"close": (["timestamp", "symbol"], np.arange(6, dtype=float).reshape(3, 2))},
    coords={"timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "symbol": ["AAPL", "MSFT"]},
)

XrBackend().to_internal(panel).write(path)          # 内存 -> 磁盘

b = XrBackend().read(path)                          # 磁盘 -> 内存
print(dict(b.get_xarray_dataset().sizes))
b.filter_by_date("timestamp", "2024-01-03", "2024-01-04") \
 .filter_by_symbol("symbol", ("MSFT",))             # 链式收窄
print(dict(b.get_xarray_dataset().sizes))
print(dict(b.data.sizes))                           # 再问一次

XrBackend().data                                    # 没读就访问
print(XrBackend().read(path).get_lazyframe().collect())   # 换个视角
```

```
读进来的形状: {'timestamp': 3, 'symbol': 2}
收窄后的形状: {'timestamp': 2, 'symbol': 1}
[[3.]
 [5.]]
再问一次还是收窄后的: {'timestamp': 2, 'symbol': 1}
未读取就访问: Please cal 'read' or 'to_internal' first.
shape: (6, 3)
┌─────────────────────┬────────┬───────┐
│ timestamp           ┆ symbol ┆ close │
│ datetime[ns]        ┆ str    ┆ f64   │
╞═════════════════════╪════════╪═══════╡
│ 2024-01-02 00:00:00 ┆ AAPL   ┆ 0.0   │
│ ...                 ┆ ...    ┆ ...   │
└─────────────────────┴────────┴───────┘
```

`get_lazyframe()` 在 `XrBackend` 上会把面板拍平成长表（3 时间 × 2 标的 = 6 行）——它先 `to_dataframe()` 再转 polars，所以**不惰性**，全量在内存里过一遍。`PlBackend.get_lazyframe()` 是真惰性（直接返回 `scan_parquet` 的 LazyFrame）。同一个方法名，两种成本，这是介质差异，不是 bug。

### 2. 有界探查，和它为什么必须自己开 store

```python
probe = XrBackend().head(zpath, 3)          # 不需要先 read
print(probe.collect_schema())

b = XrBackend()
b.head(zpath, 3).collect()
b.data                                       # 仍然是空的

PlBackend().head(ppath, 3).collect()         # limit 下推进 parquet reader
XrBackend().head(str(tmp / "nope.zarr"), 3)  # 当场 FileNotFoundError
```

```
XrBackend.head 的 schema: Schema({'timestamp': Datetime(time_unit='ns', time_zone=None),
                                  'symbol': String, 'close': Float64, 'volume': Int64})
shape: (3, 4) ... 三行真数据 ...
head 之后 self.data 依然是空的: Please cal 'read' or 'to_internal' first.
XrBackend.head 缺路径: File /tmp/.../nope.zarr does not exist.
PlBackend.head 缺路径: File /tmp/.../nope.parquet does not exist.
```

注意 schema 里 `volume` 是 `Int64` 而不是 float——"真实 dtype"是契约里明写的一条，因为调用方要拿**真表达式**在这个探针上试算，dtype 错了试算结果就错了。

对照组，把 RV-01 那条链跑出来：

```python
shared = XrBackend().read(zpath)
shared.filter_by_date("timestamp", "2024-01-01", "2024-01-05")   # 某个调用方收窄了
shared.read(zpath)                                                # 另一个调用方“重新读”
```

```
read 之后: {'timestamp': 100, 'symbol': 2}
再 read 一次: {'timestamp': 5, 'symbol': 2} <- 收窄活下来了
```

第二次 `read()` 什么也没做——`XrBackend.read` 的缓存早退看到 `self.data` 已经在了就直接返回。这就是为什么 `head` 必须绕开 `read()` 这条路，也是为什么 `base/data.py:BaseDataset.head()` 只是一句 `return self.data_backend.head(self.config.zarr_file_path, n)`：路径由数据集提供（它本来就拥有这个路径），有界读取由后端实现，数据集不加自己的意见。

### 3. append 与守卫

见上一节的输出。可复现的最小脚本骨架：

```python
XrBackend().to_internal(panel(["2022-01-04", "2022-06-15"], ["A", "XYZ"])).append(path)
XrBackend().to_internal(panel(["2023-01-04"], ["A", "ARM"], 900.0)).append(path)
#   -> ValueError: refusing to append ... the 'symbol' coordinate does not match

XrBackend().to_internal(panel(["2023-01-04"], ["A", "ARM"], 900.0)).widen_and_append(path)
#   -> OK，symbol 轴变成 ['A', 'ARM', 'XYZ']
```

花名册没变时 `widen_and_append` 会直接退化成 `append`，不重写 store（`widen_and_append` 里 `if union == stored_labels and union == incoming: return self.append(...)`），所以在例行刷新里无条件调用它是廉价的——这一点由 `tests/test_symbol_axis_widening.py:test_widen_and_append_with_an_unchanged_axis_takes_the_plain_append_path` 用 monkeypatch 钉住。

## 扩展：新增一种存储介质

### 子类必须实现什么

继承 `DataBackend`，实现全部 8 个抽象方法：`read`、`write`、`to_internal`、`filter_by_date`、`filter_by_symbol`、`get_lazyframe`、`get_xarray_dataset`、`head`。

ABC 的强制力是真的：漏掉任何一个，类**根本无法实例化**。这就是 `head` 被做成抽象方法而不是 `get_lazyframe(limit=...)` 上一个可选关键字的全部理由——可选关键字靠普通继承就"满足"了，只会在某个恰好传了它的调用点炸；抽象方法在构造那一刻就炸。跑出来是这样的：

```
少实现一个方法会怎样:
Can't instantiate abstract class Incomplete without an implementation for abstract method 'head'
```

`append` / `widen_symbol_axis` / `widen_and_append` / `APPEND_DIM_CHUNK` 都**不在** `DataBackend` 上，它们是 `XrBackend` 自己的方法。这是有道理的：追加语义是 Zarr 这个介质才有的能力。但反过来说，`base/data.py:BaseDataset.from_raw_data_chunked()` 里写死了 `self.data_backend.append(...)`，所以一个没有 `append` 的后端目前跑不了分块摄取路径——用得上分块的新介质需要自己实现 `append`。

### 一个完整可跑的最小后端

下面这个 `CsvBackend` 把面板存成一个 CSV 长表。它整个是跑通了的（`uv run python` 从仓库根目录），输出贴在后面。

```python
"""一个最小的 CSV 存储后端：换介质，不换上层。"""
from pathlib import Path
from typing import Self

import pandas as pd
import polars as pl
import xarray as xr

from quantlab.base.backend import DataBackend


class CsvBackend(DataBackend):
    """把面板存成单个 CSV 文件（长表：timestamp / symbol / 各列）。"""

    def read(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = pl.scan_csv(path, try_parse_dates=True, **kwargs)
        return self

    def write(self, path: str, **kwargs) -> Self:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.data.collect().write_csv(path, **kwargs)
        return self

    def to_internal(self, data: pl.LazyFrame) -> Self:
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        self.data = self.data.filter(
            pl.col(col).is_between(
                pl.lit(pd.to_datetime(start_date)),
                pl.lit(pd.to_datetime(end_date)),
            )
        )
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        self.data = self.data.filter(pl.col(col).is_in(symbols))
        return self

    def get_lazyframe(self) -> pl.LazyFrame:
        return self.data

    def get_xarray_dataset(self, indexes: list[str]) -> xr.Dataset:
        frame = self.data.collect().to_pandas().set_index(indexes)
        return xr.Dataset.from_dataframe(frame)

    def head(self, path: str, n: int) -> pl.LazyFrame:
        # 三条义务：自己开 store（不读 self.data）、不物化全量（limit 下推到
        # scan_csv）、缺路径当场抛。
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        return pl.scan_csv(path, try_parse_dates=True).head(n)


if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    path = str(tmp / "panel.csv")

    frame = pl.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "close": [185.6, 374.7, 184.2, 370.6],
        }
    ).lazy()

    CsvBackend().to_internal(frame).write(path)

    b = CsvBackend().read(path)
    print("xarray 视角:")
    print(b.get_xarray_dataset(["timestamp", "symbol"]))

    print("\n收窄后:")
    print(
        CsvBackend()
        .read(path)
        .filter_by_symbol("symbol", ("MSFT",))
        .get_lazyframe()
        .collect()
    )

    print("\n有界探查（不需要先 read）:")
    print(CsvBackend().head(path, 2).collect())

    print("\n少实现一个方法会怎样:")

    class Incomplete(DataBackend):
        def read(self, path, **kwargs): ...
        def write(self, path, **kwargs): ...
        def to_internal(self, data): ...
        def filter_by_date(self, col, start_date, end_date): ...
        def filter_by_symbol(self, col, symbols): ...
        def get_lazyframe(self): ...
        def get_xarray_dataset(self, indexes): ...

    try:
        Incomplete()
    except TypeError as e:
        print(e)
```

真实输出：

```
xarray 视角:
<xarray.Dataset> Size: 64B
Dimensions:    (timestamp: 2, symbol: 2)
Coordinates:
  * timestamp  (timestamp) datetime64[us] 16B 2024-01-02 2024-01-03
  * symbol     (symbol) object 16B 'AAPL' 'MSFT'
Data variables:
    close      (timestamp, symbol) float64 32B 185.6 374.7 184.2 370.6

收窄后:
shape: (2, 3)
┌─────────────────────┬────────┬───────┐
│ timestamp           ┆ symbol ┆ close │
│ ---                 ┆ ---    ┆ ---   │
│ datetime[μs]        ┆ str    ┆ f64   │
╞═════════════════════╪════════╪═══════╡
│ 2024-01-02 00:00:00 ┆ MSFT   ┆ 374.7 │
│ 2024-01-03 00:00:00 ┆ MSFT   ┆ 370.6 │
└─────────────────────┴────────┴───────┘

有界探查（不需要先 read）:
shape: (2, 3)
┌─────────────────────┬────────┬───────┐
│ timestamp           ┆ symbol ┆ close │
│ ---                 ┆ ---    ┆ ---   │
│ datetime[μs]        ┆ str    ┆ f64   │
╞═════════════════════╪════════╪═══════╡
│ 2024-01-02 00:00:00 ┆ AAPL   ┆ 185.6 │
│ 2024-01-02 00:00:00 ┆ MSFT   ┆ 374.7 │
└─────────────────────┴────────┴───────┘

少实现一个方法会怎样:
Can't instantiate abstract class Incomplete without an implementation for abstract method 'head'
```

写一个新后端时，逐条自查：

1. `read` 缺路径抛 `FileNotFoundError`（消息里带上路径）。
2. `head` 三条义务：自己开 store、真的有界、缺路径当场抛。**不要照着 `filter_by_*` 的样子写它。**
3. `read`/`write`/`to_internal`/`filter_by_*` 都 `return self`，否则链式调用会碎。`quantlab/ml_model/backend.py:MlBackend` 曾经是反例（2026-09-07 已修，见常见坑第 5 条）。
4. `get_xarray_dataset` 必须能吐出 `[timestamp, symbol]` 形状——这是全流水线的硬约束——而且必须**真的按 `indexes` 收窄**：请求了没有的维度要报错，不能静默返回一个形状不符的 `Dataset`。它还必须**不改写** `self.data`（这一点跟 `filter_by_*` 相反）。
5. 如果这个介质要走分块摄取，还得自己实现 `append`（含坐标 / dtype 守卫）。

这几条的可执行版本在 `tests/test_backend_head.py`（有界读取的三条义务）、`tests/test_backend_overwrite.py`（缓存早退与 `overwrite=`）和 `tests/test_symbol_axis_widening.py`（加宽路径的全部不变量）里——19 个用例，`uv run python -m pytest` 跑得通。每个用例的 docstring 都写明了"改成什么样它会变红"，改后端时先读它们比读实现快。

## 常见坑

**1. `filter_by_*` 是就地的，`head` 不是。** 这是同一个接口上两种相反的语义，也是唯一一条已经真出过事的。想探查就用 `head`，别用 `read().filter()`。

**2. `XrBackend.read()` 的缓存早退很容易被忽略。** `read(path)` 在 `self.data` 已存在时**什么也不做**，哪怕磁盘上的 store 已经变了。想强制重读要传 `overwrite=True`：

```
to_internal 之后再 read: {'timestamp': 1, 'symbol': 2} -> 没有重新读盘
overwrite=True 之后:     {'timestamp': 2, 'symbol': 2}
```

**3. `XrBackend.get_xarray_dataset(indexes)` 曾经完全忽略 `indexes`。**（**已于 2026-09-07 修复**）以前它的函数体就是 `return self.data`——形状是存进去时的形状，参数只是为了跟 ABC 签名对齐，传胡说八道也照样返回整个面板：

```
indexes 传胡说八道也没事: ['timestamp', 'symbol']
```

而同一个 ABC 方法在 `PlBackend` 那边是真的用（`set_index(indexes)` 然后 `Dataset.from_dataframe`）——一个方法两套含义。

现在两边对齐到 `PlBackend` 一直以来的语义：**`indexes` 就是结果的索引维度**。`XrBackend` 会校验、丢掉不属于这些维度的变量、丢掉用不到的维度、再 `transpose` 成给定轴序；请求不存在的维度会 `ValueError`。由 `tests/test_backend_indexes.py` 锁。

**对既有调用点是无操作**：全仓传的要么是 `["timestamp", "symbol"]`，要么什么都不传，而规范面板的每个变量都恰好铺在这两维上——什么都不会被丢掉，只是轴序被钉死。真正因此改变的只有 `["timestamp"]` 那条路，也就是 `BaseDataset.time_interval`（见 dataset.md 「常见坑」第 6 条）。

所以现在**可以**指望传 `indexes` 收窄一个 Zarr 面板了；但它仍然不会帮你把 long-format 表转成面板，那是 `PlBackend` 的活。

**4. `PlBackend.write()` 不建父目录，`XrBackend.write()` 建。** 前者直接 `self.data.collect().write_parquet(path)`：

```
PlBackend.write 到不存在的目录: FileNotFoundError No such file or directory (os error 2): /tmp/...
```

**5. `MlBackend` 的三个方法都不返回 `self`。**（**已于 2026-09-07 修复**）`ModelBackend` 的 ABC 签名写的是 `-> Self`，但 `quantlab/ml_model/backend.py` 的实现以前全部隐式返回 `None`，所以链式写法会当场挂：

```
MlBackend.to_internal 返回: None
链式会挂: AttributeError 'NoneType' object has no attribute 'write'
```

现在三个方法都 `return self`，跟 `XrBackend` / `PlBackend` 一致，`MlBackend().to_internal(m).write(path)` 和 `MlBackend().read(path).get_model()` 都能直接写；ABC 早就声明的 `**kwargs` 也补上并真的透传给 joblib。由 `tests/test_ml_backend.py` 锁（含一条 `write(..., compress=3)` 的透传断言，防止 `**kwargs` 变成摆设）。

`MlBackend` 在仓库里仍然**没有任何调用点**——`quantlab/base/model.py` 是直接用 `torch.save`/`joblib.dump` 的。但它**不是死代码**：`BaseModel.predict()` 签名里的 `np.ndarray` 分支是有意留的，为的是 `MLConfig` 那条非 torch 模型（xgboost 之类）的路，而 `MlBackend` 就是那条路的持久化。它是尚未建成的既定路线的脚手架——正因为如此，才值得在第一个调用方出现之前把它修好，而不是让它在第一次被按文档使用时就挂掉。

**6. `XrBackend.head(path, n)` 的"n 行"是先对每一维都切 n，再取前 n 行。** 实现是 `opened.isel({dim: slice(0, n) for dim in opened.dims})`，然后 `to_dataframe()`，最后 `.head(n)`。所以中间物化的是最多 `n^(维数)` 行——二维面板下 n=3 会先展开成 6 行再切到 3 行。这仍然是有界的（这是"不物化全量"的要求），但如果你把 n 调到几千、维数又多，中间那步不是免费的。用 `isel(dims)` 而不是写死 `timestamp`，是因为一个介质无关的后端不该假设这个项目的面板恰好按时间和标的索引。

**7. `XrBackend.head` 用 `xr.open_dataset`，`_assert_append_compatible` / `widen_*` 用 `xr.open_zarr`。** 这是刻意的：`head` 跟 `read()` 用同一个 opener（一个类里对同一个 store 用两个 opener 早晚出事），而 append 侧的 `open_zarr` 是另一件事——它要看 store 的既有坐标和 dtype。

**8. 落盘时会有一条 zarr 警告**：`Consolidated metadata is currently not part in the Zarr format 3 specification`。上面的示例输出里都过滤掉了。它是 zarr 3.3.0 对 consolidated metadata 的提醒，不是这个模块的问题。
