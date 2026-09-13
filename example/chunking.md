# 时间分块与分块台账（chunking）

> 涉及代码：`quantlab/base/chunking.py`、`quantlab/base/data.py:BaseDataset.from_raw_data_chunked()`、
> `quantlab/dataset/backend.py:XrBackend.append/widen_symbol_axis/widen_and_append`、
> `quantlab/dataset/stock.py` 的两个 seam、`quantlab/acquisition/universe.py:_roster_window_profile()`、
> `ingest_us_equity.py`。测试在 `tests/test_chunked_ingest.py`。
>
> **2026-09-12 更新**：本文原本还记述 `UniverseCatalog` 上那组稠密面板内存估算/守卫，
> 以及 `TimeChunkPlanner` 的第二个 planner。两者都已在 phase 03.6 删除，见文末
> 「2026-09-12 —— 内存守卫被删除了（phase 03.6）」一节。

## 一句话

把「一次性把整段历史铺成稠密 `[timestamp, symbol]` 面板再落盘」拆成「一个时间窗口铺一次、
铺完就 append 进 Zarr」，让峰值内存跟**窗口**大小成正比而不是跟**整段区间**成正比；
同时用一个 JSON 台账记住哪些窗口已经写完，跑了三小时挂掉可以从断点继续。

---

## 不用它会怎样

问题不在磁盘，在内存。稠密面板的大小是纯粹的乘法：

```
cells = 时间点数 × 标的数
bytes = cells × 变量数 × 每格字节数
```

这套乘法是读者判断「一个粒度够不够小」的基本工具，它没有过时；过时的只是**谁替你算**。
格数那一半现在由 `UniverseCatalog._roster_window_profile()` 给出
（标的数 / 交易日数 / 时间点数 / 稠密格数 / 观测格数 / 密度，**不含字节**），
字节那一半从 2026-09-12 起没有任何代码替你算了：乘上「变量数 × 每格字节数」要自己来。
习惯取值是 `num_variables=12`（对应 `enums.data.TiingoColumns.EOD`）、
`bytes_per_value=8`（float64，`StockDataset._raw_data_to_xr()` 实际产出的 dtype）。

**真实数字之一（见下方"完整例子"里 2026-09-12 重跑的 dry-run 输出）**：
`us_all` 全市场、日频、一年区间 `2025-09-07..2026-09-06`，
8,157 个标的 × 252 个交易日 = 2,055,564 格；按 12 × 8 算是 **0.18 GiB**。
这个量级毫无压力。注意 dry-run 现在只打印格数相关的量（观测格数、密度），
字节是上面这行手算出来的 —— 它不再是任何一个守卫的输出。

**真实数字之二（把区间拉长到二十年就翻天覆地）**：
2026-09-06 在目标机器上的实测（这次测量本身没有被删，被删的只是承载它的那个常量）：
`us_all` 全量在 `2006-01-01..今天` 是
**15,424 个标的 × ~5,215 个交易日 = 80.4M 格**，其中只有 ~29.6M 格是真实观测
（密度 0.368 —— 因为绝大多数标的并非全程都在上市）。
按 12 变量 × 8 字节算就是 **~7.2 GiB 的 float64 稠密网格**。

而旧的整段路径 `StockDataset._raw_data_to_xr()` 走的是
`.collect().to_pandas().set_index([...]).to_xarray()`，
这条链上**同时**持有：2,960 万行的 pandas frame、7.2 GiB 的稠密数组、以及转换过程的
临时内存。目标机器只有 16 GiB —— 于是 OOM。这就是 quick task
`260906-13w-k-zarr-16gib-oom` 的由来（`.planning/quick/260906-13w-k-zarr-16gib-oom/`），
分块模块是它的产物。

注意这里两个数字不要混淆：**7.2 GiB 是稠密面板，16 GiB 是机器内存**。
OOM 不是因为面板本身超过 16 GiB，而是因为面板 + 行式 frame + 转换 scratch 三份叠加。

还有两点值得记住：

- **稀疏并不能救你**。密度 0.368 说明 63% 的格子是 NaN，但 `to_xarray()` 产生的是完整
  笛卡尔积，NaN 也要占 8 字节。
- **频率是乘数**。分钟频一个交易日是 390 行（`UniverseCatalog.BARS_PER_DAY_BY_FREQUENCY`），
  同一个窗口在 `1m` 下的稠密网格是 `1d` 的 390 倍。S&P 500 的一个分钟年稠密后约 **4 TB**，
  按日频算会报成 ~10 GiB —— 差三个数量级。`_roster_window_profile()` 的时间点数是按真实
  时间轴数的，不是按交易日数的，所以这一步不会再算错；但既然没有守卫了，算错的后果也不再
  是一条报错，而是 OOM。

4 GiB 这条线曾经是一个常量（`4 * 1024**3`），画在"能舒服跑的窗口"和"7.2 GiB 会 OOM"
之间。**这个常量在 2026-09-12 随守卫一起被删除了**，它作为经验参考值仍然有用：
在一台 16 GiB 的机器上，让单个窗口的稠密面板停在几个 GiB 以内是安全的，
但今天没有任何代码会替你检查这一点。

---

## 核心概念

### chunk window（分块窗口）

一段连续时间。`TimeChunkPlanner` 负责把整段区间切成一串窗口，
粒度取自 `TimeChunkPlanner.GRANULARITIES`——从粗到细五级：
`year` / `quarter` / `month` / `day` / `hour`。这里刻意不把这个五元组抄成另一份常量：
它是唯一定义处，`--chunk` 的 `choices` 也是从它派生的，加一级不需要改 CLI，也不需要改本文。

它只有**一个** planner：

| 方法 | 什么时候用 | 边界是什么 |
|---|---|---|
| `plan_from_timestamps(timestamps)` | **写入时**，真实时间轴已经存在 | 边界一定是**观测到的**时间戳 |

为什么边界必须来自观测值：**交易日 ≠ 日历日**。如果窗口边界是 `2022-12-31`（那天可能是周日），
把它交给稠密化逻辑就等于凭空造出一个市场根本没开的行。现在调用方能拿到的每一个边界
都是真实观测到的时间戳，所以这类错误已经不可达 —— 详见文末的 2026-09-12 一节。

分组规则由私有方法 `_period_key()` 与 `_group_by_period()` 承担，
"一个年/季/月/日/小时到底怎么算"在整个模块里**只定义一次**。
这个类今天真正要防的漂移在 `GRANULARITIES` 与 `_period_key` 之间：新增一级要改两处，
只改前者过去会**静默**落到 month 尺寸的窗口 —— 错误的窗口尺寸就是错误的内存上界，
而且是 fail-open。`_period_key` 末尾现在有一条同时点名两个列表的 `raise ValueError`；
两者一致时它不可达（`__init__` 先拒绝不在 `GRANULARITIES` 里的 token），
不可达正是它作为漂移探测器而非死代码的原因。

### pinned symbol axis（钉死的标的轴，D-02）

**整段区间的 symbol 轴必须在第一个窗口开始之前就算好、固定下来，
之后每个窗口都 reindex 到这条轴上。**

为什么必须这样？因为 Zarr 的 append 只沿 `append_dim`（这里是 `timestamp`）延长，
其他维度的坐标是**整体覆写**的。假设：

- 2022 窗口里只有 `A` 在交易，于是它自己推导出的 symbol 轴是 `["A"]`；
- 2023 窗口里 `A` 和 `B` 都在，轴变成 `["A", "B"]`。

裸 `to_zarr(mode="a", append_dim="timestamp")` 在这里**不会报错**，它会默默把 store 的
symbol 坐标改成新的，历史行就归属错了。所以 `from_raw_data_chunked()` 先调
`_raw_axes_in_range()` 拿到整段区间的**全时并集**（这条规则跟
`quantlab/base/constituent.py:_densify` 的 all-time-union 一致，不是新发明的机制），
每个窗口都强制 `reindex(symbol=pinned)`。

代价是：一个 2025 年的窗口也会给 2009 年就退市的票留一整列 NaN。
这正是为什么估算一个 chunk 的内存时必须用**钉死的全区间标的数**，
而不是这个 chunk 期间在市的标的数 —— 按后者算会低估真实分配量，OOM 就又溜回来了。
（历史注记：这条规则过去由一个 pre-flight 守卫强制执行；守卫已于 2026-09-12 删除，
规则本身不变，只是现在由你自己在心算时遵守。）

`StockDataset._raw_axes_in_range()` 的实现值得看一眼：它做的是**两次单列 unique 扫描**
（polars 各自做列投影），峰值内存是原始 frame 的一列，而不是稠密网格。

### ledger（台账）

`ChunkLedger` 是一个写在 Zarr store **旁边**的 JSON 文件，
路径由 `ChunkLedger.default_path()` 生成，等于 `<store路径> + ChunkLedger.SUFFIX`，
即 `.chunks.json`。

**为什么在旁边而不在里面**：`XrBackend.write()` 用 `mode="w"`，它替换整个 store **目录**。
台账放里面的话，会被恰恰是"让恢复变得必要"的那个操作删掉。
`tests/test_chunked_ingest.py::test_the_ledger_lives_beside_the_store_not_inside_it` 钉住了这一点。

内容长这样（下方例子里是真实输出）：

```json
{
  "append_dim": "timestamp",
  "symbol_count": 3,
  "symbol_fingerprint": "2e70d7...",
  "windows": [{"start": "...", "end": "...", "rows": 3}, ...]
}
```

`ChunkLedger.record()` 用「写临时文件 → `os.fsync` → `os.replace`」的方式落盘，
所以崩在写台账中间只会留下"旧的完整台账"或"新的完整台账"，
不会留下一个解析不了的半截文件 —— 那会让下次连恢复都做不了。

### fingerprint（指纹）

`ChunkLedger.fingerprint(symbols)` = 对换行连接后的 symbol 列表做 sha256。

- **为什么存指纹不存列表**：15,000 个 ticker 会把 sidecar 撑爆，而恢复时唯一要问的问题是
  "还是同一批标的吗"，指纹能精确回答。
- **为什么对顺序敏感**：钉死的轴是一条**有序坐标**，同一批标的两种顺序会产生两个对不齐的 store。
  这也是 `XrBackend.widen_and_append` 里那个并集要 `sorted(...)` 的原因。

### resume（恢复）

重跑同一条命令即可。`from_raw_data_chunked()` 对每个窗口先问
`ledger.is_written(start, end)`，写过就跳过。
跑完的 store 再跑一次，densify 零次（`test_second_run_against_a_complete_store_densifies_zero_windows`）。

关键在 `ChunkLedger.assert_consistent()`：**台账和 store 是同一个事实的两份独立记录，
写在两个不同的瞬间**。append 是不可逆的，事后从 store 单独看不出来对不对，
所以恢复时两边都不单独信任，只要不一致就直接拒绝。四种情况按顺序检查：

1. store 不存在 + 台账为空 → 正常的首次运行，放行；
2. 台账里记录的指纹 ≠ 当前钉死的 symbol 列表 → 两次运行之间名单变了（新上市/退市），
   store 里每一列都建立在旧轴上 → 拒绝；
3. store 存在但没有台账 → 不知道里面已经有哪些窗口，盲 append 会重复或漏写 → 拒绝；
4. store 的最后一个 `timestamp` ≠ 台账最后一条窗口的 `end` → 崩溃恰好落在
   `to_zarr` 成功和台账写入之间，重跑会写重 → 拒绝。

---

## 它是怎么工作的

```
ingest_us_equity.py --to-zarr --chunk year --on-new-listing refuse
        │
        ▼
（2026-09-12 起：这里没有内存尺寸守卫了。窗口太大就直接 OOM，
  不会有一条指明「换更细的 --chunk」的可读拒绝。）
        │
        ▼
StockDataset(...).from_raw_data_chunked(granularity="year", on_new_listing="refuse")
        │
        ├─(0) 校验 on_new_listing ∈ BaseDataset.NEW_LISTING_STRATEGIES
        │     如果子类没覆写 _raw_data_to_xr_window，warning：分块只限制了写、没限制稠密化
        │
        ├─(1) symbols, timestamps = self._raw_axes_in_range()      ← 轴在这里钉死
        │
        ├─(2) windows = TimeChunkPlanner(granularity).plan_from_timestamps(timestamps)
        │
        ├─(3) ledger = ChunkLedger(<store>.chunks.json)
        │
        ├─(4) _reconcile_new_listings(...)   ← refuse / rebuild / widen 三选一
        │     走 update() 进来时，这里的策略是从**原始数据层的证据**解析出来的，
        │     不是调用方给的：_resolve_new_listing_strategy(added, removed, ...)
        │
        ├─(5) ledger.assert_consistent(symbols, store_path)   ← 在第一次不可逆写入之前
        │
        └─(6) for (start, end) in windows:
                  if ledger.is_written(start, end): continue      ← 断点续跑
                  window = self._raw_data_to_xr_window(start, end, symbols)
                  assert window.symbol == symbols  逐个标签比对，不一致直接报错
                  window = self._clean(window)                    ← 逐窗口清洗
                  window = self._pin_append_dtypes(window)        ← int → float64
                  backend.to_internal(window).widen_and_append(       ← 三条轴一起对齐
                      store, append_dim="timestamp",
                      fill_values=self._widen_fill_values())
                  ledger.record(start, end, rows, symbols)         ← 原子写台账
              最后：若存在内部边界，warning 说明 flag_anomalies 的边界损失
```

### 为什么 symbol 轴要在第一个窗口之前就定死

三条理由叠在一起，缺一不可：

1. **Zarr 的 append 会静默覆写非 append 维的坐标。** 每个窗口自己推轴 → 历史行归属错误，
   而且事后从 store 里完全看不出来。
2. **稠密化的输出形状取决于输入。** `to_xarray()` 产生的是这个窗口里出现过的
   `(timestamp × symbol)` 笛卡尔积；不 reindex 的话不同窗口列数不同，根本没法拼。
3. **台账的指纹是对整条轴算的。** 轴要是每个窗口一变，恢复就无从校验。

所以 `_raw_axes_in_range()` 必须是「不稠密化就能拿到两条轴」的 —— 这就是它被单独做成
一个 overridable seam 的原因。`BaseDataset` 给的默认实现是**正确但不省内存的**
（它调 `_raw_data_to_xr()` 铺整段再取轴），子类如果能把日期谓词下推到数据源
（`StockDataset` 下推到 polars 的 parquet scan），覆写它才真正拿到内存上界。
没覆写会在运行时被 warning 点名 —— 这是「有正确默认实现的 seam」而不是
`raise NotImplementedError` 的桩，老子类不改也能跑。

`_raw_data_to_xr_window()` 是同一个套路的第二个 seam。

### 逐窗口清洗的代价（已知且会被告知）

`_clean()` 是逐窗口跑的，所以在每个 chunk 的第一个时间戳上，
`flag_anomalies` 没有前一个样本可以做差分，跨边界的单步跳变不会被标记。
这是**有界的、有文档的**后果，不是 bug，代码会在最后打一条 warning 报出边界数量。
注意方向：`--chunk` 越细，边界越多，不是越少。

---

## 分块与追加的交界

分块负责"每次只做多少"，`XrBackend` 负责"这些字节怎么增量落到介质上"。两者在
`backend.to_internal(window).append(path, append_dim="timestamp")` 这一行相遇。

### 首次写入：`XrBackend.APPEND_DIM_CHUNK = 512`

store 不存在时走 `mode="w"` 创建，并显式传 `encoding`（由 `_append_encoding()` 生成）：
append 维的 chunk 长度固定为 512，其他维取满长度。
**为什么必须显式指定**：不给 `encoding` 的话，Zarr 会拿第一个窗口自己的长度当 chunk 大小，
之后每一个长度不同的窗口（短交易年、不完整的最后一个月）都跟磁盘上的 chunk 网格错位。
固定值让存储布局成为 **store 的属性**，而不是"碰巧第一个被写进去的那个窗口"的属性。

append 时则显式 `kwargs.pop("encoding", None)` —— xarray 在 append 上直接拒绝 encoding，
而且网格已经在创建时钉好了。

### `_assert_append_compatible`：为什么这个守卫必须存在

它在每次 append 前检查两件事，任一不符就 `raise ValueError`：

1. **每个非 append 维的坐标值必须逐个相等。**
2. **每个共享的 data variable 必须保持 dtype。**

裸 `to_zarr(mode="a", append_dim=...)` 这两条都不管：

- **坐标被静默覆写。** 2026-09-06 实测记录在 `widen_symbol_axis` 的 docstring 里：
  一个 `{A, ARM}` 的窗口写进 `{A, XYZ}` 的 store，结果是
  `rows [1.0, 3.0] were written for XYZ but are now labelled: ARM` —— 一声不吭。
  注意这里**长度是一样的**（一个退市 + 一个新上市 → 数量不变、标签变了），
  所以只比长度的检查抓不到它，必须逐标签比。
  `tests/test_chunked_ingest.py::test_plain_append_still_refuses_a_labels_differ_axis_of_the_same_length`
  就是这个的回归守卫。
- **dtype 被静默转换。** 往 int64 变量里 append 一个 float64 的 NaN，
  Zarr 会把它转成 **0** —— 数据缺失变成了一个凭空捏造的观测值。

两种破坏事后从 store 里都看不出来，而 append 是不可逆的，所以只能在写之前拒绝。

上游还有一层对称的预防：`BaseDataset._pin_append_dtypes()` 在 append 前把所有整数变量
提升成 float64。因为**一个窗口的 dtype 取决于它自己的稠密程度** —— 全满的窗口
`volume` 保持 int64，有一格缺失的窗口就为了 NaN 升成 float64。不统一的话，store 的
dtype 就由"碰巧第一个写进去的窗口"决定了。bool 的 `anomaly_flag` 故意不动：它是标记不是测量。

实跑一遍这个守卫（真实输出）：

```
append 守卫: XrBackend.append: refusing to append to /tmp/quantlab_chunk_errs.zarr -- the
'symbol' coordinate does not match the store (2 incoming label(s) vs 2 stored). Zarr would
OVERWRITE the stored labels without complaint, silently re-attributing every previously
written row. Pin the 'symbol' axis over the whole range before the first window, the way
BaseDataset.from_raw_data_chunked() does.
store 仍是: ['A', 'XYZ']
```

### 新上市时的三条出路

定期刷新时名单几乎一定会变。`BaseDataset.NEW_LISTING_STRATEGIES = ("refuse", "rebuild", "widen")`，
对应 `ingest_us_equity.py --on-new-listing`（CLI 的 `choices` 是从这个元组**派生**的，
`quantlab/utils/cli.py:add_chunk_args`，不在 CLI 里重复写一遍）。

| 策略 | 做什么 | 代价 |
|---|---|---|
| `refuse`（默认） | 什么都不改，让 `assert_consistent` 的 roster 错误抛出，store 原封不动。先打一条 INFO 告诉你另外两条路存在 | 无。但是运行会停 |
| `rebuild` | 把 store 和台账**改名挪走**，然后从原始数据把**每一个窗口**在新并集轴上重新稠密化 | 全量重跑，慢。但新上市标的的**真实历史**被找回来了 |
| `widen` | `XrBackend.widen_symbol_axis()` 就地把 store 的 symbol 轴加宽，新标的整段历史填 NaN，然后 `ChunkLedger.rebase()` 重新算指纹 | 快，但**不重读原始数据**，供应商已经有的历史拿不回来；而且加宽要把**整个 store 读进内存**（这里没有 dask，`.load()` 是必须的），store 太大就用不了 —— 那种情况恰恰该用 `rebuild` |

两个策略各自的关键安全设计：

- **`rebuild` 是"改名挪走"而不是"删掉"**（`BaseDataset.SUPERSEDED_SUFFIX = ".superseded.tmp"`）。
  删了之后在多小时重建的一半崩掉就没有退路了。`from_raw_data_chunked()` 用
  `try/except BaseException` 包住整个循环，失败就 `_restore_rebuild_asides()` 把原件改名回来，
  成功才 `_discard_rebuild_asides()` 丢掉副本。
- **`widen` 必须在同一次操作里 `rebase` 台账**。指纹是对顺序敏感的，加宽改了轴 ——
  不 rebase 的话，加宽后的 store 下一次运行会撞上它刚刚绕过去的那个 roster 错误，变成不可恢复。
- `widen_symbol_axis` 的目录交换是「原件改名到 `.superseded.tmp` → 新件改名到正位 → 删除副本」，
  两次 rename 同父目录因而原子。**改名在前是故意的**：崩在两次 rename 之间会导致
  `path` 上**没有** store，下次读会响亮地失败，而不是把一个半加宽的 store 当成权威。
  这时它会拒绝自动恢复并告诉你手动执行 `mv`，因为"哪个目录才是权威的"不是这个方法该替你决定的。
- `widen` 还必须给非浮点变量显式 fill：`BaseDataset._widen_fill_values()` 返回
  `{"anomaly_flag": False}`。不给的话 `reindex` 会把 bool 升成 float64+NaN，
  等于悄悄改了一个**在线 store** 的 schema，所以 `widen_symbol_axis` 直接拒绝。
  一个真实的清洗过的行情面板恰好只有 `anomaly_flag` 一个非浮点变量（其余都被
  `_pin_append_dtypes` 浮点化了），所以这个 fill 是**必需的**，不是学术性的。

还有一个 `XrBackend.widen_and_append()`：先把 store 和待写数据都对齐到
`sorted(stored ∪ incoming)`，再调**没有被改动过的** `append()`。
最后那次 `append()` 调用是承重的 —— 两边现在共用同一条轴，
`_assert_append_compatible` 照样跑而且按它自己的规则**通过**，
守卫是被**构造性地满足**了，而不是被绕过或放松了。没有 opt-in 的调用方仍然会被拒绝。

**逐窗口的那次写入现在就走 `widen_and_append()`**（260908-0f4），所以 `data_vars`
这条轴在分块路径上也被对齐了：供应商在两次定期刷新之间加了一列，以前整个运行会被
`append` 的「incoming panel carries data variable(s) [...]」拒掉，现在这一列会先在
store 的**已有区间**上被物化成 NaN（非浮点变量按 `_widen_fill_values()` 给的值），
再正常追加。

这是**被论证过的行为改变**，不是顺手改的：

1. 两条轴都一致时（现有的每一个测试、每一个生产调用）`widen_and_append` 直接委派给
   **同一个** `append()`，store 不存在时也一样委派，所以旧行为是被**构造性**保留的；
2. `on_new_listing` 表达的只是关于 **symbol 轴**的意图，没有任何调用方在变量轴上
   声明过什么被推翻；
3. 真正危险的形状 —— 供应商**改列名** —— 表现为「少一个 + 多一个」，而**少的那一半仍然被
   无条件拒绝**，所以它照样**停下来**：没有窗口被写入，没有历史被截断。

**改列名这个形状有一个被接受的副作用，实测记录在此，免得后来人当 bug 重新发现一遍。**
因为加宽是在最后那次 `append()` 之前**提交**的，被拒之后 store 里会**同时**留着旧列名和
新引入的列名（新的那个在 store 已有区间上是全 NaN）；而它替换掉的那个朴素 `append()`
面对同样的形状是**完全不碰 store** 的。append 维没有增长、每一个已存值逐位不变，所以这是
「继承拒绝」的代价而不是半截写入 —— `widen_and_append` 自己的 docstring 早就写明了这一点。
由 `tests/test_chunked_ingest.py::test_a_window_missing_a_stored_variable_is_still_refused`
锁住。

顺带修掉的一个真 bug（同一次改动）：`widen_data_vars` 以前会把 store 的坐标读出来再写回去，
而 `xr.open_zarr` 解码出来的 dtype 和 zarr 记录的 dtype 可能不是同一个（实测：zarr 存
`symbol` 为 `object`，解码成 numpy 的 `StringDType()`），写回去直接
`Mismatched dtypes for variable symbol` —— 也就是说**任何带字符串坐标的 store**（这里的
每一个真实 store）都用不了变量加宽。填充块现在只带 dims 不带 coords，坐标原样留在 store 里。

---

## 两个入口：`from_raw_data_chunked()` 和 `update()`

| | `from_raw_data_chunked(...)` | `update(...)` |
|---|---|---|
| 语义 | **转换**：把原始数据按窗口稠密化并落盘 | **增量更新**：把已有 store 补到最新 |
| 策略从哪来 | 调用方给的 `on_new_listing`，默认 `refuse` | **从原始数据层的证据里读出来**，没有这个参数 |
| CLI | `ingest_us_equity.py --on-new-listing` | 暂时没有（见 `.planning/todos/pending/`） |
| 对称物 | `Factor.save()` | `Factor.update()` |

`update()` **不给策略参数，这正是它的功能**。widen 和 rebuild 的区别不是偏好，而是关于
原始数据层的一个**事实**，而猜错的一方是**无声地丢数据**。所以它被读出来，而不是被问出来。
三条分支，全部来自证据：

- **任何**新增标的在 store 自己的 append 维区间内**已经有原始数据行** → `rebuild`。
  这些行是 widen 会用 NaN 顶掉的真实历史，顶掉之后 store 和「这份数据从来不存在」
  长得一模一样。rebuild 是**整库**操作，不是按标的来的。
- **没有**任何新增标的带这样的行 → `widen`。它们是真正的新上市，NaN 在 store 的历史上
  就是正确的值，rebuild 纯属浪费。
- **任何**标的被移出名单 → `refuse`。`widen` 根本表达不了「少一个标签」
  （`widen_symbol_axis` 拒绝非超集的目标轴），`rebuild` 会**悄悄丢掉**那个标签已存的历史。
  两条都不该在没有人的情况下选，所以这条路停下来，而不是替你编一个答案。

**决定会在 rebuild 真的跑起来之前被说出来**：日志会报有几个新增标的够格，并逐个列出标的
和它在 store 区间内的原始数据行数（按行数降序，最多 20 个，截断时会讲明自己截断了）。
无声地切换策略，和给错一个 flag 是同一种不透明，只是方向相反。

这个探针只在 symbol 轴**真的漂移**了的时候才会被付费 —— `_reconcile_new_listings` 里那个
「轴一致就直接返回」的早退是**唯一**的漂移检测点，探针在它后面。而且它问的窗口是
**store 自己的区间**，不是 config 的日期范围：用后者的话每一个真正的新上市都会看起来像证据，
从而触发一次没必要的全量重建（实测本仓库真实 store，两个窗口差了十一个月）。

成本记的是**次序**而不是秒数（秒数换台机器就过期，而且下一个读者没法验证）：

```
探 store 区间  <  探整个原始数据层  <  _raw_axes_in_range()
                                      ↑ 每一次分块运行本来就无条件在付这一笔
```

也就是说探针**从来不会**比这条路径本来就要走的一步更贵。2026-09-08 在本仓库真实数据层
（26,584 个 `.pqt`、13 个 `month=` 分区、153.1 MB）热缓存下的那次读数是
1.32–1.75 s / 2.24–2.38 s / 3.22–3.32 s ——**只有次序是承重的**。

```python
# 定期刷新：不用想 widen 还是 rebuild，它自己读
StockDataset(config).update(granularity="year")
```

---

## 完整例子

### 例一：真实的分块计划（不需要任何凭证）

```bash
env -u TIINGO_API_KEY -u APCA_API_KEY_ID -u APCA_API_SECRET_KEY \
    uv run python ingest_us_equity.py --start-date 2025-09-07 --end-date 2026-09-06 --dry-run
```

真实输出（**2026-09-12 重跑**，三个凭证环境变量都用 `env -u` 显式清掉）：

```
DRY RUN -- category=us_all, no price requests issued
  symbols resolved:  8157
  preview:           ['A', 'AA', 'AAAC', 'AAAP', 'AAC', 'AAC-U', 'AAC-WS', 'AACB', 'AACBR', 'AACBU']
  window:            2025-09-07 .. 2026-09-06
  trading days (~):  252
  real observations: 1,840,211
  density:           0.895
  raw-data path:     /Users/daizhaorong/projects/quantlab/data/downloads/us_equity/1d/us_all/tiingo
  watermark path:    /Users/daizhaorong/projects/quantlab/data/downloads/us_equity/1d/us_all/_watermarks/tiingo
  zarr path:         /Users/daizhaorong/projects/quantlab/data/data/us_equity/1d/us_all.zarr
  coverage report:
  already covered:   831 (would be skipped)
  re-fetch, widened: 6922 (recorded coverage starts after --start-date)
  legacy, no start:  0 (stamp via --stamp-legacy-watermarks)
  would fetch:       7326/8157
```

**这份输出是重跑出来的，不是把旧输出手改的。** 本目录的约定是贴真跑过的输出
（见 [README.md](README.md)），而 2026-09-09 那一版里的
`dense grid cells` / `dense float64` / `observed float64` / `chunk granularity` /
`chunk count` / `whole-range total` / `largest chunk` / `per-chunk budget` 八行
今天已经一行都不打印了 —— 它们全部属于 phase 03.6 删掉的那组估算/守卫。
两次之间标的数从 8,159 变成 8,157、密度从 0.890 变成 0.895，是 roster 本身在这三天里动了，
跟本次删除无关；预览的十个符号变了，是因为这是排序后的前十个而非随机十个。

几个可以对照上文读的点：

- **分块相关的行整体消失了**：粒度、窗口数、整段总量、最大 chunk、每 chunk 预算都不再打印。
  `--chunk` 本身没有变，它照常被接受、照常传到 `from_raw_data_chunked()`；
  消失的是那份**内存报告**，不是分块能力。
- 现在还在的格子类数字（`real observations`、`density`）来自
  `UniverseCatalog._roster_window_profile()`，就是上文「不用它会怎样」一节用的那套算术。
- 密度 0.895 是因为区间只有一年；拉到 2006 起就掉到 0.368。
- 覆盖报告**不需要凭证**（03.4 起）：上面这次是把三个凭证环境变量全部清掉跑的。
  `re-fetch, widened: 6924` 说的正是本文关心的那件事——盘上那批边车记录的覆盖起点
  晚于这里请求的 `--start-date`，所以历史比要求的浅，要重抓。
  这个判断走的是真实 run 走的同一个 `CoverageLedger.partition_by_coverage`
  对象，见 [registry.md](registry.md)。

### 例二：造合成数据、中途崩溃、恢复、看台账（真跑）

把下面的脚本存成 `/tmp/chunk_demo.py`，在仓库根目录执行
`PYTHONPATH=. uv run python /tmp/chunk_demo.py`：

```python
"""分块写入 + 中断 + 恢复 的最小可跑示例。"""
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl
import xarray as xr

from quantlab.base.chunking import ChunkLedger, TimeChunkPlanner
from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset

TMP = Path("/tmp/quantlab_chunk_demo")
shutil.rmtree(TMP, ignore_errors=True)
RAW, STORE = TMP / "raw", TMP / "demo.zarr"

COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume",
        "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
        "divCash", "splitFactor", "vendor"]

def row(date_str, symbol, close=100.0):
    return {"timestamp": datetime.fromisoformat(date_str), "symbol": symbol,
            "open": close, "high": close, "low": close, "close": close,
            "volume": 1000.0, "adjOpen": close, "adjHigh": close,
            "adjLow": close, "adjClose": close, "adjVolume": 1000.0,
            "divCash": 0.0, "splitFactor": 1.0, "vendor": "tiingo"}

# --- 造原始数据：A 全程都在，B 只在 2023 上市，C 只在 2024 上市 -------------
rows = []
for year in (2022, 2023, 2024):
    for day in ("01-04", "06-15", "12-28"):
        d = f"{year}-{day}"
        rows.append(row(d, "A"))
        if year == 2023: rows.append(row(d, "B"))
        if year == 2024: rows.append(row(d, "C"))

frame = pl.DataFrame(rows).with_columns(
    pl.col("timestamp").dt.strftime("%Y-%m").alias("month"))
for (month,), part in frame.partition_by("month", as_dict=True).items():
    d = RAW / "tiingo" / f"month={month}"
    d.mkdir(parents=True, exist_ok=True)
    part.select(COLS).write_parquet(d / "part-demo-00000.pqt")

config = DatasetConfig(
    raw_data_dir_path=str(RAW / "tiingo"), zarr_file_path=str(STORE),
    catalog_path=str(TMP / "catalog"), market="us_equity",
    frequency="1d", vendor="tiingo")

print("=== 1. 先看轴：symbol 轴在任何窗口之前就钉死 ===")
symbols, timestamps = StockDataset(config)._raw_axes_in_range()
print("pinned symbols :", symbols)
print("observed stamps:", [str(t.date()) for t in timestamps])
print("year windows   :", [(str(s.date()), str(e.date()))
                           for s, e in TimeChunkPlanner("year").plan_from_timestamps(timestamps)])

print("\n=== 2. 模拟第 2 个窗口崩溃 ===")
class CrashOnSecondWindow(StockDataset):
    seen = 0
    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None):
        if CrashOnSecondWindow.seen == 1:
            CrashOnSecondWindow.seen += 1
            raise RuntimeError("模拟第 2 个窗口崩溃")
        CrashOnSecondWindow.seen += 1
        return super()._raw_data_to_xr_window(start_date, end_date, symbols)

try:
    CrashOnSecondWindow(config).from_raw_data_chunked(granularity="year")
except RuntimeError as exc:
    print("崩溃:", exc)

print("崩溃后 store 里的时间点数:", xr.open_zarr(STORE).sizes["timestamp"])
ledger_path = ChunkLedger.default_path(str(STORE))
print("台账路径:", ledger_path)
print(Path(ledger_path).read_text())

print("=== 3. 原地重跑，从第 2 个窗口继续 ===")
StockDataset(config).from_raw_data_chunked(granularity="year")

store = xr.open_zarr(STORE)
print("\n恢复后 store:", dict(store.sizes))
print("symbol 轴:", store["symbol"].values.tolist())
print("adjClose:\n", store["adjClose"].to_pandas())
print("\n最终台账:")
print(Path(ledger_path).read_text())
print("fingerprint 校验:", ChunkLedger(ledger_path).symbol_fingerprint
      == ChunkLedger.fingerprint(symbols))
```

真实输出（loguru 日志走 stderr，下面按时间顺序单独列出，只去掉了时间戳前缀）：

```
INFO  | base.data:from_raw_data_chunked - CrashOnSecondWindow: chunked ingestion over 3 year window(s), 3 pinned symbol(s), 9 observed timestamp(s).
INFO  | dataset.cleaning:validate_schema - validate_schema: 6/9 (66.7%) (timestamp, symbol) cell(s) hold no bar at all — null in every required column. That is the dense panel's cartesian product (D-06), not an anomaly; nulls on those cells are excluded from the counts below.
INFO  | base.data:from_raw_data_chunked - CrashOnSecondWindow: appended window 2022-01-04..2022-12-28 (3 row(s)).
INFO  | base.data:from_raw_data_chunked - StockDataset: chunked ingestion over 3 year window(s), 3 pinned symbol(s), 9 observed timestamp(s).
INFO  | base.data:from_raw_data_chunked - StockDataset: window 2022-01-04..2022-12-28 already recorded in the ledger, skipping.
INFO  | dataset.cleaning:validate_schema - validate_schema: 3/9 (33.3%) ...
INFO  | base.data:from_raw_data_chunked - StockDataset: appended window 2023-01-04..2023-12-28 (3 row(s)).
INFO  | dataset.cleaning:validate_schema - validate_schema: 3/9 (33.3%) ...
INFO  | base.data:from_raw_data_chunked - StockDataset: appended window 2024-01-04..2024-12-28 (3 row(s)).
WARNING | base.data:from_raw_data_chunked - StockDataset: cleaning ran per window, so at 2 chunk-boundary timestamp(s) `flag_anomalies` had no prior sample to diff against and a single-step jump across that boundary is not flagged. A bounded, documented consequence of chunking -- finer --chunk granularity produces more such boundaries, not fewer.
```

```
=== 1. 先看轴：symbol 轴在任何窗口之前就钉死 ===
pinned symbols : ['A', 'B', 'C']
observed stamps: ['2022-01-04', '2022-06-15', '2022-12-28', '2023-01-04', '2023-06-15', '2023-12-28', '2024-01-04', '2024-06-15', '2024-12-28']
year windows   : [('2022-01-04', '2022-12-28'), ('2023-01-04', '2023-12-28'), ('2024-01-04', '2024-12-28')]

=== 2. 模拟第 2 个窗口崩溃 ===
崩溃: 模拟第 2 个窗口崩溃
崩溃后 store 里的时间点数: 3
台账路径: /tmp/quantlab_chunk_demo/demo.zarr.chunks.json
{
  "append_dim": "timestamp",
  "symbol_count": 3,
  "symbol_fingerprint": "2e70d7238a20934f7a8a145e8750ee44d6e043a7a8c9b3a1a0979d640e62af8c",
  "windows": [
    {
      "start": "2022-01-04T00:00:00",
      "end": "2022-12-28T00:00:00",
      "rows": 3
    }
  ]
}
=== 3. 原地重跑，从第 2 个窗口继续 ===

恢复后 store: {'timestamp': 9, 'symbol': 3}
symbol 轴: ['A', 'B', 'C']
adjClose:
 symbol          A      B      C
timestamp
2022-01-04  100.0    NaN    NaN
2022-06-15  100.0    NaN    NaN
2022-12-28  100.0    NaN    NaN
2023-01-04  100.0  100.0    NaN
2023-06-15  100.0  100.0    NaN
2023-12-28  100.0  100.0    NaN
2024-01-04  100.0    NaN  100.0
2024-06-15  100.0    NaN  100.0
2024-12-28  100.0    NaN  100.0

最终台账:
{
  "append_dim": "timestamp",
  "symbol_count": 3,
  "symbol_fingerprint": "2e70d7238a20934f7a8a145e8750ee44d6e043a7a8c9b3a1a0979d640e62af8c",
  "windows": [
    {
      "start": "2022-01-04T00:00:00",
      "end": "2022-12-28T00:00:00",
      "rows": 3
    },
    {
      "start": "2023-01-04T00:00:00",
      "end": "2023-12-28T00:00:00",
      "rows": 3
    },
    {
      "start": "2024-01-04T00:00:00",
      "end": "2024-12-28T00:00:00",
      "rows": 3
    }
  ]
}
fingerprint 校验: True
```

对着输出可以确认的几件事：

- **窗口边界是观测到的时间戳**：第一个窗口开在 `2022-01-04` 而不是 `2022-01-01`。
- **每个窗口都带完整的 3 列**：`B` 在 2022 从没交易过，2022 那个窗口里它依然是一整列 NaN。
  这跟整段稠密化产出的值完全一样（`test_chunked_store_matches_the_unchunked_store` 逐变量比对过）。
- **崩溃后 store 有 3 个时间点，台账里有 1 条窗口** —— 两份记录一致。
- **重跑跳过窗口 1，只做窗口 2 和 3**，最终 9 个时间点，一行不多一行不少。
- **边界 warning 说的是 2 个** —— 三个窗口有两个内部边界，整条轴的第一个时间戳在
  不分块的路径下也没有前置样本，所以不算。

---

## 常见坑

**1.（历史，已不可达）拿日历算术切出来的窗口去稠密化。**
这曾经是一个真实的坑：`TimeChunkPlanner` 过去还有一个只用于估算的 planner，
它的边界是日历算术的产物，会给出市场根本没开的日期，而唯一拦着你别把这种窗口
交给稠密化逻辑的东西，是它 docstring 第一行的一句 `SIZING ONLY` 警告。
phase 03.6 删掉了那个 planner（它唯一的生产调用方是同时被删的内存估算器），
于是这个坑**在结构上**消失了：今天调用方能拿到的每一个边界都是观测到的时间戳，
剩下这一个 planner 因此不需要任何这类警告。保留这一条是为了说明为什么它不再出现。

**2. 以为一个更细的 `--chunk` 能减少边界告警。**
方向反了。窗口越细，chunk 边界越多，`flag_anomalies` 丢掉的跨边界差分越多。
`--chunk` 是拿"异常检测的边界损失"换"更低的峰值内存 + 更细的恢复粒度"。

**3. 手动删掉台账想"重来一遍"。**
store 还在、台账没了 → `assert_consistent` 的第 3 种情况 → 直接拒绝，
因为没法知道 store 里已经有哪些窗口。要重来就**两个都删**。反过来只删 store 留台账
也一样会被拒（第 4 种情况的兄弟分支）。

**4. 以为 `widen` 能把新上市标的的历史补回来。**
不能。`widen` **不重读原始数据**，新标的整个历史块是 NaN，即使供应商那里有数据。
要真实历史只有 `rebuild`。这两条路的差异被
`test_widen_keeps_history_and_backfills_the_new_listing_with_nan` 和
`test_rebuild_redensifies_every_window_onto_the_new_union` 成对钉死。

**5. 以为 `widen` 总是"便宜的那个"。**
它在墙钟时间上便宜，但它要 `.load()` 整个 store 进内存重写一遍 ——
这里没装 dask，所以这正是分块存在要避免的那个分配。
store 大到装不下的时候，该用的是 `rebuild`（逐窗口重建）。

**6. 给 intraday 频率估算时按日频的行数算。**
日频一个交易日一行，分钟频是 390 行（`BARS_PER_DAY_BY_FREQUENCY`）。
按日频算一个分钟窗口，会把 4 TB 报成 ~10 GiB，差三个数量级。
2026-09-12 之前这会让守卫「自信地」放行一个超出预算三个数量级的请求；
今天没有守卫了，所以它直接就是一次 OOM。

**7. 假设 `--chunk month` 一定比 `--chunk year` 小很多。**
每个窗口都建在**钉死的全区间 symbol 轴**上，所以 chunk 变小**只**因为覆盖的时间变短，
不会因为那段时间在市的标的少而变小。心算一个 chunk 的大小时要用全区间标的数
（`_roster_window_profile()` 的 `symbols`），不要逐 chunk 重算在市标的数。

**8. 在子类里没覆写 `_raw_data_to_xr_window` 就指望省内存。**
默认实现是"铺整段再切片"，功能正确但一点内存都不省。这时候分块只限制了**写**、
给了你**可恢复性**，没限制**稠密化**。代码会 warning 说
`_raw_data_to_xr_window has not been overridden`。

**9. 改了 symbol 的排序。**
指纹是对顺序敏感的 sha256。同一批标的换个顺序 = 不同的指纹 = 恢复被拒。
`StockDataset._raw_axes_in_range()` 用 `sorted()`，`widen_and_append` 的并集也用 `sorted()`，
都是为了跨运行可复现。

**10. 用裸 `to_zarr(mode="a", append_dim=...)` 绕开 `XrBackend.append()`。**
两种静默破坏立刻回来：坐标标签被覆写（历史行归属错误）、float NaN 被转成整数 0
（缺失变成捏造的观测值）。两者事后都不可见，而 append 不可逆。
守卫在写之前拒绝，是唯一能起作用的位置。
（这个守卫**还在**，跟下面那一节说的内存守卫不是同一个东西。）

---

## 2026-09-12 —— 内存守卫被删除了（phase 03.6）

把这一节单独写出来，是因为「文档里干脆不提一个被删掉的守卫」会让读者以为它从来不存在，
或者以为是漏写。它确实存在过，是被**决定**删掉的。

**被删掉的是什么。** `UniverseCatalog` 上那组稠密面板内存估算/守卫**整组**消失了 ——
估算的那一半和拒绝的那一半都没了，连同它们共用的 4 GiB 预算常量。开发者的判断是：
一个 roster 目录（「谁在池子里、什么时候在」）不该同时承担内存尺寸策略，
而 `quantlab/utils/cli.py` 为了从这个对象里问出一个纯算术答案，
一度不得不**伪造**一个假的 catalogue 再 override 它的估算方法。

**代价，明确接受。** 一个过大的转换窗口现在**直接 OOM**，
不再有一条指明「换更细的 `--chunk`」的可读拒绝。
决定于 2026-09-11 做出、2026-09-12 在缩小 phase 03.6 范围时重申；
记录在案的先例是 quick task `260906-13w-k-zarr-16gib-oom`
（~7.2 GiB 稠密网格 + ~2,960 万行 frame 打爆一台 16 GiB 机器，日频全市场，不是 tick）。

**操作者手上还剩什么。**

1. **更细的 `--chunk`，而且比以前更细。** 粒度阶梯从三级扩到了五级：
   `year` / `quarter` / `month` / `day` / `hour`。被删掉的是**拒绝**，不是**调节手段** ——
   调节手段反而变强了。注意 `hour` 的 IO 代价是已知且被接受的：原始层按 `date=` 分片，
   一个小时窗口要读整天的 parquet 再丢掉大部分。
2. **acquisition-volume 守卫，完整保留。** `estimate_acquisition_volume()` 与
   `assert_acquisition_volume_fits()` 仍然会回答、仍然会拒绝 —— 但它们量的是
   **原始磁盘字节、请求数、墙钟时间**，也就是钱和时间，**不是 RAM**。
   烧掉的 API 配额不可回收，内存不够重跑一次就行，这是两者区别对待的理由。见
   [acquisition.md](acquisition.md)。
3. **自己算。** `UniverseCatalog._roster_window_profile()` 给格数、交易日数、密度；
   乘上变量数和每格字节数就是本文开头那套乘法。它不再是守卫，是一把尺子。

**没被删掉的东西，别搞混。** `XrBackend.append()` 的坐标/dtype 守卫、
`ChunkLedger.assert_consistent()` 的四种拒绝、`on_new_listing` 的三选一 —— 全都还在。
本节说的只有稠密面板的**内存**守卫这一个。
