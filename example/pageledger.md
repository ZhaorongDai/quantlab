# 分页台账（PageLedger）

> 代码位置：`quantlab/base/pageledger.py`（459 行，纯 stdlib，零项目内 import）
> 唯一调用方：`quantlab/base/acquisition.py`（`Acquisition._ledger_for` / `Acquisition._fetch_batch`）
> 测试：`tests/test_page_ledger.py`

---

## 一句话

`PageLedger` 是一个 JSON 小文件，记录**某一个批次（batch）的分页抓取走到了第几页、下一页的 token 是什么、这些页的数据落到了哪些 parquet 分片上**，好让一次中断的下载能从**批次中间**继续，而不是从批次开头重来、也不是从错误的位置跳过去。

---

## 不用它会怎样

先说清楚场景。Alpaca 这类行情商，一次请求可以带一整批 symbol（`AlpacaAcquisition.DEFAULT_BATCH_SIZE = 100`），返回结果按 **symbol 优先、再按 bar 时间戳** 排序，用一个不透明的 `next_page_token` 串成页链，每页最多 `page_limit` 行（`quantlab/acquisition/alpaca.py` 里 `self._knob("page_limit", 10_000)`，默认 10000）。一个 `us_all` 回补要发几千个这样的批次请求，一个批次内部可能有几十上百页。

现在假设第 37 页失败了（限流、网络断、Ctrl-C）。没有台账的话，只有两个选择，**两个都是错的**：

1. **从批次第 0 页重来。** 数据不会错，但前 36 页的配额、时间、带宽全部白烧。全市场回补里这不是"有点浪费"，是把一次跑不完的任务变成永远跑不完。
2. **猜一个位置接着抓。** 因为返回是 symbol-major 的，第 37 页开头是哪个 symbol、哪个时间点，你在批次外部是不知道的。猜错就**静默丢掉中间那些 symbol 的数据**——没有报错、没有异常，下一次读盘的时候也查不出来，只是那几十个 symbol 的分钟线少了一段。

`PageLedger` 防的是**第 2 种**：静默漏数据。顺带把第 1 种的浪费也省掉了。

还有一个更隐蔽的坑，`PageLedger.symbols_seen` 的 docstring 专门写了（对应 `03.2-RESEARCH.md` 的 Pitfall 4）：因为是 symbol-major 排序，**一个 100 symbol 批次的第 0 页可能只包含 1 个 symbol 的数据**。如果按"每页算一次 `requested - seen`"来判断"这个 symbol 查过了但没数据"，就会把另外 99 个 symbol 标记成"无数据"、推进它们的 watermark、然后**永远跳过**它们。这是 99% 的静默数据丢失，而且看起来是一次成功的运行。所以 `symbols_with_data` 是**整个批次累积**的，且只有 `is_complete()` 之后才有意义。

---

## 核心概念

### batch（批次）

一次向 vendor 发出的、带一组 symbol 的请求，加上一个时间窗 `[start_date, end_date]` 和一个频率。批次由 `Acquisition._batches()` / `_refresh_batches()` 切出来。**一个批次 = 一个台账文件**。

### batch key（批次键）

`PageLedger.batch_key(vendor, frequency, start_date, end_date, symbols)`，返回 sha256 的**前 16 位十六进制**。

为什么要有它：台账文件名要在两次运行之间**稳定**，否则重跑时找不到上次的台账，恢复就无从谈起。为什么是 16 位：全市场回补大约 155 个批次，16 位（64 bit）不用担心碰撞，同时文件名还读得下去（docstring 原话）。

**symbol 在哈希前会排序**，因为批次是一个"集合"。请求 `["B","A"]` 和 `["A","B"]` 发出去是同一个请求、拿回来是同一批行，当成两个批次就会重复下载已经在盘上的数据。

> 这里和 `quantlab/base/chunking.py` 的 `ChunkLedger.fingerprint` **故意不一样**：那边钉住的是 Zarr 的 symbol **轴**，是有序的，两种顺序会产生两个对不齐的 Zarr store，所以顺序是身份的一部分，不排序。`tests/test_page_ledger.py::test_the_batch_key_fingerprint_is_a_function_of_the_set_not_the_order` 把这个差异写成了断言。

### fingerprint（名单指纹）

`PageLedger.fingerprint(symbols)`：把 symbol 排序、换行连接、取完整 sha256（不截断）。存在台账的 `symbol_fingerprint` 字段里。

**它防的是"参数变了但台账还在"这一类错误。** 具体是这样：

`batch_key` 里已经包含了 symbol 集合，理论上名单变了 key 就变了、文件名就变了、根本不会撞上。但文件是留在磁盘上的，而磁盘上的东西会被人和外部工具动：手改过的台账、从别处恢复的备份、部分还原的目录、旧版本代码写的文件。一旦一个记着"抓到第 12 页"的台账被一个**不同名单**的批次读到，恢复就会从第 12 页开始——而中间那些页压根没为新名单里的 `D` 抓过。

所以 `PageLedger.__init__` 收一个 `symbols` 参数（当前名单），`_load()` 在读盘之后做校验：

- 存的指纹 ≠ 当前名单的指纹 → **整个台账读成空**，从第 0 页重抓。代价是重抓一次，是正确的。
- 存的指纹是 `None`、**但已经有 pages** → 同样读成空。这一条很关键：**"没有指纹"不等于"没有意见"**，而等于"这些页是为一个现在没人能识别的名单抓的"，往上面恢复就是往未知上恢复。
- 存的指纹是 `None`、**且没有 pages** → 保留原样，不清空。因为没东西可恢复，清空反而会丢掉新版本写进去的、当前代码还不认识的额外字段。

（对应测试：`test_a_ledger_whose_roster_fingerprint_differs_is_not_resumed_onto`、`test_a_ledger_with_pages_but_no_fingerprint_is_never_resumed_onto`、`test_an_identityless_ledger_with_no_pages_keeps_its_forward_compatible_keys`。）

**注意：`symbols` 是可选的。** `PageLedger(path)` 不传名单就完全不做这个校验（下面的例子第 7 步验证了这一点）。真实调用方 `Acquisition._ledger_for` 永远传。

### page（页）

台账 `pages` 数组里的一条记录，由 `record_page()` 追加：

| 字段 | 含义 |
|---|---|
| `index` | 页序号，从 0 开始 |
| `next_token` | vendor 返回的下一页 token，**逐字原样存**；最后一页是 `null` |
| `rows` | 这一页的行数 |
| `shards` | 这一页的数据写到了哪些 parquet 文件（绝对路径列表） |
| `last_symbol` / `last_timestamp` | 这一页最后一行的位置，token 的兜底 |

`next_token` **绝不自己重新编码**。vendor 的编码是没有文档的、随时可能改；自己拼一个 `symbol|timeframe|timestamp` 一旦对不上，就会从一个 vendor 从没同意过的位置继续。

### resume point（恢复点）

`resume_point()` 返回 `(下一页序号, page_token)`：

```python
last = pages[-1]
return int(last["index"]) + 1, last.get("next_token")
```

返回的是**最后一条成功记录**所携带的 `next_token`——也就是说，恢复后发的第一个请求，正是**上次被打断的那个请求**，而不是上次已经成功的那个。没有任何记录时返回 `(0, None)`，这就是首次运行的状态。

### last position（无 token 兜底）

每条 page 记录里都带着 `last_symbol` / `last_timestamp`。这是设计文档 D-03 要求的**免 token 降级路径**的原料：vendor 对 token 的有效期没有任何公开说明，而它自己文档里的示例 token 解码出来就是一个 `SYMBOL|TIMEFRAME|TIMESTAMP` 的位置三元组。所以万一存的 token 被拒了，恢复可以退化成"把 `start` 收窄到这个时间戳、把名单裁到这个 symbol 及其之后"重发。浪费，但正确。

> 注：**这条降级路径目前只是"信息被记下来了"**，没有任何读取它的代码——`quantlab/base/acquisition.py` 的 `_fetch_batch` 只用 `resume_point()`。
>
> 曾经有一个 `last_position()` 方法返回 `(last_symbol, last_timestamp)`，但它从来没有被任何地方调用过，**2026-09-07 已删除**。删除的理由和删掉 `WindowedRobustStandardization` 是同一条：一个从未被执行过的公开方法，读的人会当它是个可用的入口。真要用的时候直接读 `pages[-1]` 的 `last_symbol` / `last_timestamp`，或者 `git show` 把它捞回来——那时候至少会有一个真实调用方来验证它。

### symbols seen

`symbols_seen()`：整个批次里、在**任何一页**上出现过至少一行数据的 symbol 集合。见上面"不用它会怎样"最后一段。调用方必须遵守"只有 complete 之后才可信"这条纪律——`quantlab/base/acquisition.py` 确实遵守了：

```python
marked: set[str] = set()
if outcome.complete and not self._abort.is_set():
    marked = set(symbols) - outcome.symbols_with_data
```

### complete

`mark_complete()` 在 vendor 返回 `next_token is None` 时调用，表示这条页链走到头了。它**不**表示"这个批次不该再抓了"——见下面"常见坑"。

---

## 磁盘上长什么样

台账放在 `{watermark_path}/_pages/` 下，是原始数据目录（raw tier）的**兄弟**，绝不放在里面。原因很实际：polars 的目录扫描会走遍给定根目录下的每一个文件，raw 树里混一个 `.json` 会让 `pl.scan_parquet` 直接崩（D-19 契约 2）。

文件名 = `{batch_key}` + `PageLedger.SUFFIX`（`".pages.json"`），目录名 = `PageLedger.DIRNAME`（`"_pages"`），由 `PageLedger.default_path()` 拼出来。

仓库里现在真实存在的树：

```
data/downloads/us_equity/1m/nasdaq_data/_watermarks/alpaca/
├── AAPL.json            <- 每 symbol 一个 watermark（102 个），另一层，不是 PageLedger
├── ABNB.json
├── ... (共 102 个)
├── _failures.json       <- 失败清单，Acquisition 的，不是 PageLedger
└── _pages/              <- PageLedger.DIRNAME
    ├── 3c6a2e14da4b501b.pages.json    8737 B
    ├── 89e3be2e3dc0ecee.pages.json     314 B
    ├── 996302adf9708c91.pages.json  400415 B
    ├── a475c5d794da2324.pages.json     316 B
    └── aff618a1772f46da.pages.json     649 B
```

关于那个 `_failures.json`：它是 `Acquisition` 的东西，不是 `PageLedger` 的，
两者的粒度也完全不同——页台账是**批内**断点（一批一个文件），失败清单是
**最近一次 run** 的整体快照（`{symbol: 错误消息}`，每次 run 覆盖重写）。
它**不是续跑输入**：`quantlab/` 里没有任何代码读它，续跑完全由水位边车的存在与否驱动；
保留它是因为进程崩掉之后它还在，而运维控制台要读失败原因
（详见 [acquisition.md](acquisition.md) 的「核心概念」与 [registry.md](registry.md)）。

**三种 JSON 边车现在都是原子写的**——页台账（`PageLedger._flush`）、
水位边车和这个失败清单，加上转换层的 `ChunkLedger._flush`，四个写入方
全部委托给同一个 `quantlab/utils/atomic.py:write_json_atomically`
（临时文件 + `fsync` + `os.replace`，且失败时不留下 `.tmp`）。
两个台账本来就是这么写的，另外两个是 03.4 补上的：在此之前一次批次边界上的中断
会把一个完好的水位边车先截断成空文件再开始写，而取消恰恰发生在批次边界上。

### 一个真实的、完整的台账（`aff618a1772f46da.pages.json`，649 字节，单页批次）

```json
{
  "batch_key": "aff618a1772f46da",
  "vendor": "alpaca",
  "frequency": "1m",
  "start_date": "2024-01-02",
  "end_date": "2024-01-02",
  "symbol_count": 1,
  "symbol_fingerprint": "1eb44d625271a4eb75016f276d13783617c012685cb598f20ae93249164c0121",
  "complete": true,
  "pages": [
    {
      "index": 0,
      "next_token": null,
      "rows": 797,
      "shards": [
        "/Users/daizhaorong/projects/quantlab/data/downloads/us_equity/1m/nasdaq_data/alpaca/date=2024-01-02/part-aff618a1772f46da-00000.pqt"
      ],
      "last_symbol": "AAPL",
      "last_timestamp": "2024-01-02 23:59:00"
    }
  ],
  "symbols_with_data": [
    "AAPL"
  ]
}
```

注意分片文件名 `part-aff618a1772f46da-00000.pqt`：格式是 `part-{batch_key}-{page_index:05d}.pqt`（`Acquisition._shard_path`）。**完全确定性**——没有时间戳、没有 uuid、没有计数器。这一点后面还会用到。

### 一个"只 describe 了、一页都没抓"的台账（`89e3be2e3dc0ecee.pages.json`，314 字节）

```json
{
  "batch_key": "89e3be2e3dc0ecee",
  "vendor": "alpaca",
  "frequency": "1m",
  "start_date": "2026-08-01",
  "end_date": "2026-09-05",
  "symbol_count": 2,
  "symbol_fingerprint": "4e3b23ec9a2beafa1c39db8eccaa26e1b60034c33de914b3e279a4767c714562",
  "complete": false,
  "pages": [],
  "symbols_with_data": []
}
```

这就是 `describe()` 会 flush 的意义：批次在第 0 页就死了，磁盘上留下的仍然是一个**身份完整**的台账，不是一个没人认得的空壳。

### 一个真正的多页批次（`3c6a2e14da4b501b.pages.json`，4 页，2 个 symbol）

节选，能看出 symbol-major 分页的形状：

```json
    {
      "index": 0,
      "next_token": "QUFQTHxNfDE3ODcwNTQzNDAwMDAwMDAwMDA=",
      "rows": 10000,
      "shards": [ ... 12 个 date= 分区，2026-08-03 到 2026-08-18 ... ],
      "last_symbol": "AAPL",
      "last_timestamp": "2026-08-18 11:58:00"
    },
    {
      "index": 1,
      "next_token": "QUFQTHxNfDE3ODg0MzMyNjAwMDAwMDAwMDA=",
      "rows": 10000,
      "last_symbol": "AAPL",
      "last_timestamp": "2026-09-03 11:00:00"
    },
    {
      "index": 2,
      "next_token": "RkFTVHxNfDE3ODgyNzMyNDAwMDAwMDAwMDA=",
      "rows": 10000,
      ...
    },
    {
      "index": 3,
      "next_token": null,
      "rows": 1522,
      "last_symbol": "FAST",
      "last_timestamp": "2026-09-04 20:15:00"
    }
```

几个可以直接读出来的事实：

- 前 3 页都是 10000 行整 —— 就是 `page_limit` 打满了。
- 第 0、1 页的 `last_symbol` 都是 `AAPL`：**光 AAPL 一个 symbol 就吃掉了两页多**。第二个 symbol `FAST` 直到第 2 页才出现（token 前缀 base64 解开是 `FAST|M|...`）。这正是 Pitfall 4 描述的形状：如果按页算"没数据"，第 0 页就会把 `FAST` 判死。
- 一页的数据会散落到十几个 `date=` 分区里，所以 `shards` 是个数组，不是单个路径。

---

## 它是怎么工作的

调用方是 `Acquisition._fetch_batch`（`quantlab/base/acquisition.py`）。完整时间线：

```
【首次运行】

  _ledger_for(symbols, start, end)
     │
     ├─ batch_key = PageLedger.batch_key(VENDOR, frequency, start, end, symbols)
     ├─ ledger    = PageLedger(default_path(watermark_root, batch_key), symbols=symbols)
     │                └─ _load(): 文件不存在 → 空 payload
     └─ ledger.describe(...)  ──flush──▶  磁盘上出现身份完整、pages 为空的台账
                                          （= 上面 89e3be2e 那个样子）
  _fetch_batch
     │
     ├─ ledger.assert_consistent(raw_root)   # 空台账，直接通过
     ├─ if ledger.is_complete(): ledger.reset()      # 首次不会命中
     ├─ outcome = BatchOutcome(symbols_with_data=ledger.symbols_seen(), ...)
     ├─ page_index, page_token = ledger.resume_point()   →  (0, None)
     │
     └─ while True:
          ① frame, next_token = self._fetch_page(symbols, start, end, page_token)
          ② shards = self._write_shard(frame, batch_key, page_index)   ← 先写盘
          ③ 从 frame 算 seen / last_symbol / last_timestamp
          ④ ledger.record_page(page_index, next_token, rows, seen, shards, ...)
                                                              ──flush──▶ 后记账
          ⑤ if not next_token: ledger.mark_complete(); return outcome
          ⑥ if next_token == page_token: raise ValueError(vendor 回吐同一个 token)
          ⑦ page_index += 1; page_token = next_token
```

**第 ② 步在第 ④ 步之前，这个顺序是刻意的，而且不能反。**

- 崩在 ②④ 之间：盘上有分片、台账没记。下次重抓这一页，因为文件名确定性（`part-{batch_key}-{page:05d}.pqt`），**写的是同一个路径、覆盖掉**。代价 = 一次重抓，**绝不会产生重复行**。
- 如果顺序反过来（先记账后写盘），崩在中间就会留下"台账说第 N 页有、盘上没有"的洞，而且**后面任何一次读取都发现不了**。

这就是为什么 `assert_consistent()` 只查一个方向：**台账领先于磁盘**才是真正的分歧。它会拒绝两种情况（错误信息里自称 error 1 of 2 / 2 of 2）：

1. 某一页被记录了、但 `shards` 是空的 —— 没有任何关于这页数据落在哪的记录。
2. 某一页的 shard 路径在盘上不存在 —— 分片被删了，或者 raw 根目录被搬了。

两种都给出同一个 CURE：删掉台账文件，从第 0 页重抓；确定性文件名保证已经落地的页是被**覆盖**而不是**复制**。

```
【中途崩溃后重跑】

  _ledger_for(...)  → 同样的 symbols/window → 同样的 batch_key → 同一个文件
     │
     ├─ PageLedger._load() 读到 3 条 pages、complete=false
     ├─ 校验 symbol_fingerprint：一致 → 保留；不一致 → 全部丢弃，当空台账
     └─ describe() 再 flush 一次（身份字段原样重写）
  _fetch_batch
     ├─ assert_consistent(raw_root)：逐页核对 3 条记录的 shards 是否都在盘上
     ├─ outcome.symbols_with_data ← ledger.symbols_seen()   ← 继承前 3 页的发现
     ├─ outcome.pages ← len(ledger.pages) = 3
     ├─ resume_point() → (3, "上次第 3 页记下的 next_token")
     └─ 循环从第 3 页继续，第 0/1/2 页一个请求都不再发
```

`tests/test_page_ledger.py::test_a_resumed_run_does_not_re_request_page_zero` 断言了恢复后**没有任何一个请求的 `page_token` 是 `None`**（`None` 就意味着回到了第 0 页），而且总请求数只有 2 个。

**异常是往上抛的，但一定是在已完成的页都 flush 之后才抛。** `_fetch_page` 报错时，前面每一页都已经落了台账。吞掉异常会把一个"短了一截"的批次报成成功；先记账再抛，才把下一次运行从"重来"变成"恢复"。

---

## 完整例子

下面这段可以直接跑。存成任意文件，在仓库根目录执行：

```bash
uv run python 你的文件.py
```

```python
"""PageLedger 演示：写几页 -> 模拟崩溃 -> 恢复 -> 一致性检查 -> 指纹失配。"""

import json
import shutil
from pathlib import Path

from quantlab.base.pageledger import PageLedger

TMP = Path("/tmp/pageledger_demo")
shutil.rmtree(TMP, ignore_errors=True)
WATERMARK = TMP / "_watermarks" / "alpaca"
RAW = TMP / "raw"


def shard(page_index: int, day: str, batch_key: str) -> str:
    """造一个假的 parquet 分片文件，只为让 assert_consistent 有东西可查。"""
    path = RAW / f"date={day}" / f"part-{batch_key}-{page_index:05d}.pqt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake parquet")
    return str(path)


print("1) batch_key 是「集合」的函数，不是顺序的函数")
args = ("alpaca", "1m", "2024-01-02", "2024-01-05")
key_ab = PageLedger.batch_key(*args, ["AAPL", "MSFT"])
key_ba = PageLedger.batch_key(*args, ["MSFT", "AAPL"])
key_other = PageLedger.batch_key(*args, ["AAPL", "NVDA"])
print("batch_key([AAPL, MSFT]) =", key_ab)
print("batch_key([MSFT, AAPL]) =", key_ba, "-> 相同:", key_ab == key_ba)
print("batch_key([AAPL, NVDA]) =", key_other, "-> 相同:", key_ab == key_other)

batch_key = key_ab
path = PageLedger.default_path(str(WATERMARK), batch_key)
print("台账路径:", path)

print("2) describe()：第一页还没发出去，身份就已经落盘")
ledger = PageLedger(path, symbols=["AAPL", "MSFT"])
print("刚 new 出来（文件不存在）:", ledger)
ledger.describe(batch_key, "alpaca", "1m", "2024-01-02", "2024-01-05", ["AAPL", "MSFT"])
print("describe() 之后文件是否存在:", Path(path).exists())
print(Path(path).read_text())

print("3) 抓前两页（每页先写分片，再记台账）")
ledger.record_page(
    index=0,
    next_token="QUFQTHxNfDE3MDQyNDA=",
    rows=10000,
    seen=["AAPL"],
    shard_paths=[shard(0, "2024-01-02", batch_key), shard(0, "2024-01-03", batch_key)],
    last_symbol="AAPL",
    last_timestamp="2024-01-03 15:30:00",
)
ledger.record_page(
    index=1,
    next_token="TVNGVHxNfDE3MDQyNDA=",
    rows=10000,
    seen=["AAPL", "MSFT"],
    shard_paths=[shard(1, "2024-01-04", batch_key)],
    last_symbol="MSFT",
    last_timestamp="2024-01-04 20:00:00",
)
print(ledger)
print("resume_point() =", ledger.resume_point())
print("symbols_seen() =", ledger.symbols_seen())
print("is_complete()  =", ledger.is_complete())
# 无 token 兜底的位置直接从最后一条 page 记录里读（`last_position()` 已于
# 2026-09-07 删除，见上文）。
_last = json.loads(Path(path).read_text())["pages"][-1]
print("last position  =", (_last["last_symbol"], _last["last_timestamp"]))

print("4) 进程崩溃 —— 丢掉内存对象，只从磁盘重新读")
del ledger
resumed = PageLedger(path, symbols=["MSFT", "AAPL"])  # 顺序换了也认得
print("重新加载:", resumed)
print("下一次请求从第几页、带什么 token:", resumed.resume_point())
_last = json.loads(Path(path).read_text())["pages"][-1]
print("没有 token 时的兜底位置:", (_last["last_symbol"], _last["last_timestamp"]))
resumed.assert_consistent(str(RAW))
print("assert_consistent(raw_root) 通过")

print("5) 台账说有、磁盘上没有 —— 拒绝恢复")
victim = Path(json.loads(Path(path).read_text())["pages"][1]["shards"][0])
victim.unlink()
try:
    resumed.assert_consistent(str(RAW))
except ValueError as exc:
    print(type(exc).__name__, ":")
    print(str(exc))
victim.parent.mkdir(parents=True, exist_ok=True)
victim.write_bytes(b"fake parquet")  # 还原

print("6) 抓完最后一页（next_token 为 None）并标记完成")
resumed.record_page(
    index=2,
    next_token=None,
    rows=1522,
    seen=["MSFT"],
    shard_paths=[shard(2, "2024-01-05", batch_key)],
    last_symbol="MSFT",
    last_timestamp="2024-01-05 20:15:00",
)
resumed.mark_complete()
print(resumed)
print("symbols_seen() =", resumed.symbols_seen(), "（complete 之后才可信）")
print("resume_point() =", resumed.resume_point())

print("7) 名单变了：同一个文件，用不同 roster 打开会读成空")
mismatch = PageLedger(path, symbols=["AAPL", "NVDA"])
print("换名单打开:", mismatch)
print("resume_point() =", mismatch.resume_point(), " symbols_seen() =", mismatch.symbols_seen())
print("原名单打开:", PageLedger(path, symbols=["AAPL", "MSFT"]).resume_point())
print("不传名单打开（不做校验）:", PageLedger(path).resume_point())

print("8) reset()：清页、留身份")
again = PageLedger(path, symbols=["AAPL", "MSFT"])
again.reset()
print("reset() 后:", again, " resume_point =", again.resume_point())
print("身份还在:", again.symbol_fingerprint[:16], "... symbol_count =", again.symbol_count)
print("注意：reset() 不落盘，磁盘上仍是完成态:",
      json.loads(Path(path).read_text())["complete"])
```

### 真实输出

（以下是实际执行结果，未做任何修饰）

```text
============================================================
1) batch_key 是「集合」的函数，不是顺序的函数
============================================================
batch_key([AAPL, MSFT]) = 91c9dc202fdc2cc1
batch_key([MSFT, AAPL]) = 91c9dc202fdc2cc1 -> 相同: True
batch_key([AAPL, NVDA]) = d4d87015a721f320 -> 相同: False
台账路径: /tmp/pageledger_demo/_watermarks/alpaca/_pages/91c9dc202fdc2cc1.pages.json

============================================================
2) describe()：第一页还没发出去，身份就已经落盘
============================================================
刚 new 出来（文件不存在）: PageLedger(path='/tmp/pageledger_demo/_watermarks/alpaca/_pages/91c9dc202fdc2cc1.pages.json', pages=0, complete=False)
describe() 之后文件是否存在: True
{
  "batch_key": "91c9dc202fdc2cc1",
  "vendor": "alpaca",
  "frequency": "1m",
  "start_date": "2024-01-02",
  "end_date": "2024-01-05",
  "symbol_count": 2,
  "symbol_fingerprint": "4a1c2f2b7fca8c6aabc5de0a87bbe5aa8db9166547ed70ea29a4fb51af88af7f",
  "complete": false,
  "pages": [],
  "symbols_with_data": []
}
============================================================
3) 抓前两页（每页先写分片，再记台账）
============================================================
PageLedger(path='/tmp/pageledger_demo/_watermarks/alpaca/_pages/91c9dc202fdc2cc1.pages.json', pages=2, complete=False)
resume_point() = (2, 'TVNGVHxNfDE3MDQyNDA=')
symbols_seen() = {'AAPL', 'MSFT'}
is_complete()  = False
last position  = ('MSFT', '2024-01-04 20:00:00')

============================================================
4) 进程崩溃 —— 丢掉内存对象，只从磁盘重新读
============================================================
重新加载: PageLedger(path='/tmp/pageledger_demo/_watermarks/alpaca/_pages/91c9dc202fdc2cc1.pages.json', pages=2, complete=False)
下一次请求从第几页、带什么 token: (2, 'TVNGVHxNfDE3MDQyNDA=')
没有 token 时的兜底位置: ('MSFT', '2024-01-04 20:00:00')
assert_consistent(raw_root) 通过

============================================================
5) 台账说有、磁盘上没有 —— 拒绝恢复
============================================================
ValueError :
PageLedger: refusing to resume /tmp/pageledger_demo/_watermarks/alpaca/_pages/91c9dc202fdc2cc1.pages.json -- error 2 of 2: the ledger records page 1 but its shard /tmp/pageledger_demo/raw/date=2024-01-04/part-91c9dc202fdc2cc1-00001.pqt does not exist on disk. The two disagree, which means a shard was deleted or the raw root was moved after the ledger was written; resuming would leave a hole in the batch that no later read could detect. CURE: delete /tmp/pageledger_demo/_watermarks/alpaca/_pages/91c9dc202fdc2cc1.pages.json to re-fetch this batch from page 0, or restore the missing shard under /tmp/pageledger_demo/raw.

============================================================
6) 抓完最后一页（next_token 为 None）并标记完成
============================================================
PageLedger(path='/tmp/pageledger_demo/_watermarks/alpaca/_pages/91c9dc202fdc2cc1.pages.json', pages=3, complete=True)
symbols_seen() = {'AAPL', 'MSFT'} （complete 之后才可信）
resume_point() = (3, None)

============================================================
7) 名单变了：同一个文件，用不同 roster 打开会读成空
============================================================
换名单打开: PageLedger(path='/tmp/pageledger_demo/_watermarks/alpaca/_pages/91c9dc202fdc2cc1.pages.json', pages=0, complete=False)
resume_point() = (0, None)  symbols_seen() = set()
原名单打开: (3, None)
不传名单打开（不做校验）: (3, None)

============================================================
8) reset()：清页、留身份
============================================================
reset() 后: PageLedger(path='/tmp/pageledger_demo/_watermarks/alpaca/_pages/91c9dc202fdc2cc1.pages.json', pages=0, complete=False)  resume_point = (0, None)
身份还在: 4a1c2f2b7fca8c6a ... symbol_count = 2
注意：reset() 不落盘，磁盘上仍是完成态: True
```

第 7 步值得多看一眼：**同一个文件、同一个路径**，用 `["AAPL","NVDA"]` 打开就是 `(0, None)`（从头来），用 `["AAPL","MSFT"]` 打开就是 `(3, None)`（已完成）。这就是 fingerprint 在做的事。而不传 `symbols` 的话完全不校验——所以别自己 `PageLedger(path)` 然后往上恢复。

> **此处未实际运行的部分**：本例子里没有真的发网络请求。真实抓取中才会出现的行为有两类，本文没有实测：
> - `next_token` 的真实有效期，以及 token 被 vendor 拒绝时的降级（page 记录里 `last_symbol` / `last_timestamp` 的用途）——如上文所说，这条降级路径当前在 `quantlab/base/acquisition.py` 里还没有调用方。
> - `_fetch_batch` 里"vendor 回吐同一个 token"的死循环保护（`next_token == page_token` 分支），需要一个行为异常的 vendor 才能触发。

---

## 常见坑

**1. `is_complete()` 不是"别再抓了"的否决权。**
`_fetch_batch` 读到一个已完成的台账时做的是 `ledger.reset()`，然后照抓不误。原因写在注释里：能走到 `_fetch_batch` 就说明**上一层**（per-symbol watermark，D-05 的两层里的另一层）已经决定这个批次要抓了，比如用户显式给了 `download(resume=False)`。让批次内部的记录去覆盖 symbol 层的策略，等于悄悄吞掉用户设的开关。**页台账只回答"这个批次抓到哪了"，从不回答"这个批次该不该抓"。**

**2. `reset()` 不落盘。**
上面例子第 8 步验证了：`reset()` 之后内存里是空的，但磁盘上 `complete` 还是 `true`。它靠随后的 `record_page()` / `mark_complete()` 去 flush。所以别指望 `reset()` 能把磁盘清干净——要清磁盘就删文件。

**3. 一个批次一个文件，这是并发安全的**唯一**依据。**
`_flush()` 是原子的（同目录临时文件 + `os.fsync` + `os.replace`），但它前面那个 `payload["pages"].append(...)` 不是。多线程 worker 同时写**一个**共享 manifest 会静默丢记录，而且要到很久以后表现为"某一页被重抓了"才看得出来。类里的原话是一个**跟着代码走的条件**：**如果哪天把这些文件合并成一个总 manifest，`threading.Lock` 就从可选变成必须，而且必须在同一个 commit 里加上。**

**4. 台账文件永远不会被自动清理。**
我在代码里没有找到任何删除 `_pages/*.pages.json` 的逻辑（`PageLedger` 里也没有任何 retention 常量——整个类只有 `SUFFIX` 和 `DIRNAME` 两个常量）。每一次 `refresh()` 因为窗口变了就是一个新的 `batch_key`，也就是一个新文件。所以 `_pages/` 会随时间单调增长，需要人工清。文件很小（几百字节到几百 KB），但目录条目数会涨。

**5. 手改台账 JSON 是可以的，但删 `symbol_fingerprint` 会让整个台账作废。**
只要 `pages` 非空而 `symbol_fingerprint` 是 `null`，`_load()` 就当整个文件不存在。这是故意的（见"核心概念 / fingerprint"）。

**6. 损坏的 JSON 不会报错，会静默当空台账。**
`_load()` 捕获 `json.JSONDecodeError` 和 `OSError`，返回空 payload。也就是说一个坏掉的台账**只会**让这一个批次从头重抓，不会让整个 run 挂掉——一个批次一个文件正是为了把爆炸半径限制在这里。代价是：你不会收到任何提示。

**7. `refresh()` 的重抓会在同一个分区目录里留下**两个**文件，不是覆盖。**
确定性覆盖只在**同一个窗口**内成立。`batch_key` 里哈希了 `start_date`/`end_date`，所以同样的行在不同窗口下重抓，会落到同一个 `date=` 目录里的**另一个**文件名。`1d`/`1m` 靠 `dedup_raw_frame` 吸收；tick 按 D-16 从不 dedup，所以 `_write_shard` 走 `_clear_superseded_shards` 主动删掉被取代的分片。这不是 `PageLedger` 的逻辑，但它是"确定性文件名 → 重抓安全"这条推理的**边界**，容易被误当成无条件成立。

**8. `symbols_seen()` 在批次没完成时是"目前抓到了谁"，不是"这个批次有谁"。**
调用方必须自己守住 `complete` 这道门。`quantlab/base/acquisition.py` 守住了（`if outcome.complete and not self._abort.is_set()`），任何新的调用方也必须守。守不住的后果是本文开头说的静默 99% 数据丢失。

---

## 相关文件速查

| 路径 | 作用 |
|---|---|
| `/Users/daizhaorong/projects/quantlab/base/pageledger.py` | `PageLedger` 本体 |
| `/Users/daizhaorong/projects/quantlab/base/acquisition.py` | 唯一调用方：`_ledger_for`（979-1002 行）、`_fetch_batch`（1004-1125 行）、`_shard_path`（826-848 行） |
| `/Users/daizhaorong/projects/quantlab/base/chunking.py` | 姊妹类 `ChunkLedger`，`_flush()` 的原型；其 `fingerprint` 有序，这里无序 |
| `/Users/daizhaorong/projects/quantlab/acquisition/alpaca.py` | `DEFAULT_BATCH_SIZE = 100`、`page_limit` 默认 `10_000`、凭证环境变量 `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` |
| `/Users/daizhaorong/projects/quantlab/tests/test_page_ledger.py` | 611 行，把上面每一条不变式都写成了可执行断言 |
