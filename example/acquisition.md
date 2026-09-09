# 数据采集层（Acquisition）

> 代码位置：`quantlab/base/acquisition.py`（抽象基类）、`quantlab/acquisition/tiingo.py`、`quantlab/acquisition/alpaca.py`、
> 命令行入口 `ingest_us_equity.py`（Tiingo）与 `ingest_alpaca.py`（Alpaca）。
> 相关但独立成文的两个模块：分页断点账本见 [pageledger.md](pageledger.md)，
> 时间窗切分与转换断点见 [chunking.md](chunking.md)。

---

## 一句话

把「从外部厂商 API 把行情原始数据抓下来落到本地磁盘」这件事做成一个**可中断、可续跑、
单点失败不连坐、凭证不外泄**的可复用引擎；每个厂商只需要写「怎么发一次请求」，
其余全部由基类提供。

注意它的边界：**采集层只写原始文件（parquet 分片），绝不碰 xarray / Zarr**。
从原始文件到 canonical `[timestamp, symbol]` 面板是 `quantlab/dataset/stock.py:StockDataset`
的工作（`_raw_data_to_xr()` / `from_raw_data_chunked()`）。这条分界线是硬的，
`Acquisition` 类文档里写得很直白。

---

## 不用它会怎样

这一层几乎每一行防御性代码背后都有一次真实事故或一次真实推演。挑最能说明问题的讲：

### 1. 配额耗尽后继续烧请求

2026-09-06 真实发生过：Tiingo 账号的**每小时请求配额**用完之后，剩下约 1 万个 symbol
仍然被并发地发出去，每个都在几毫秒内 429 失败——观测到约 260 symbol/s 的空转速度。
这不但没抓到任何数据，还可能加深封禁。

现在的做法：Tiingo 把 429 归类成 `"quota"`（全局条件），一旦命中就 set 一个
`threading.Event` 全局中止旗标；`Acquisition._attempt_batch` 的**第一行**就检查这个旗标，
命中即返回 `"skipped"`，零请求。为什么必须在方法第一行而不是在任务生成器里？
因为 joblib 无法取消已经排队的任务，生成器里的 `abort.is_set()` 只是个优化。

### 2. 把全局故障当成单个股票的错

反过来也一样致命。如果把「配额耗尽」写进每个 symbol 的失败清单，那清单就在诬告
一万只完全正常的股票；下次跑的时候你分不清哪些是真的 404 退市，哪些只是撞上了配额。
所以 `"quota"` **故意不进** `_failures.json`。

同一个 HTTP 状态码在两个厂商那里含义相反，这是整个错误分类 seam 存在的原因：

| 厂商 | 429 的含义 | 分类 | 反应 |
|---|---|---|---|
| Tiingo | 每小时**配额额度**用尽，要等约一小时 | `"quota"` | 停掉整个 run 的派发 |
| Alpaca | 每**分钟** 200 次的限速（免费档），几十秒就恢复 | `"rate_limited"` | 只在本 worker 里退避重试 |

如果把 Tiingo 的读法搬到 Alpaca，一个健康的全市场 run 会在启动几秒内就自我中止；
反过来把 Alpaca 的读法搬到 Tiingo，就回到第 1 条的空转烧请求。所以基类里
**没有** `QUOTA_STATUS_CODES` 这种东西，只有可被覆盖的 `_classify_error()`。

### 3. 一只退市股票的 404 把 15000 只股票的任务拖垮

`_attempt_batch` 承诺**永不抛异常**。如果异常向上冒泡，`joblib.Parallel` 的整个 fan-out
会被拆掉，其他所有还在跑的批次一起死。所以异常在这里被捕获、分类、转成
`(symbols, status, message)` 三元组返回。

### 4. 断点续跑丢数据 / 重复数据

两处独立的断点：**symbol 级**（watermark 边车文件）和**page 级**（`PageLedger`）。
关键是写入顺序：**先写 parquet 分片，再记账本**。这个顺序不能反：

- 崩在两者之间 → 页数据在盘上但账本没记 → 下次重抓这一页，**文件名是确定性的**
  （`part-{batch_key}-{page_index:05d}.pqt`），所以是覆盖，不是多出一份。代价只是一次重抓。
- 如果反过来先记账本 → 账本说第 N 页抓完了但盘上没有 → 续跑会跳过这一页，
  留下一个**任何后续读取都发现不了的洞**。

`PageLedger.assert_consistent()` 专门检查「账本超前于磁盘」这一种不一致并拒绝续跑。

### 5. 加宽 `start_date` 后静默跳过（真实缺陷 D-03）

早期的 watermark 只记 `last_date`（覆盖到哪一天为止）。于是你把
`--start-date` 从 2020 改成 2016 重跑，所有 symbol 的 `last_date` 都等于 `end_date`，
全部被判为「已覆盖」而跳过——你以为你拿到了 2016 年起的历史，实际上一行没多。
数据集里每只股票的历史深度不一致，而且**完全没有任何警告**。

修法：watermark 记的是**区间**而不是端点（`{"start_date", "last_date", "no_data"}`），
`_classify_coverage()` 增加了 `"widened"` 状态——end 对得上但记录的起点晚于请求的起点，
就必须重抓。

### 6. 无凭证泄露的日志

这个仓库**已经真实泄露过一次 Tiingo key**（硬编码在下载脚本里）。所以：

- `AcquisitionConfig` **没有也永远不能有**凭证字段——它的 `to_dict()` 就是 `asdict(self)`，
  会被写进落盘的配置和模型 checkpoint 旁边的 JSON。
- 凭证只在 client 的 `__init__` 里从 `os.environ` 读，只进内存 client 对象。
- 所有被捕获的异常消息都经过 `Acquisition._scrub()`：按 `CREDENTIAL_ENV_VARS` 声明的
  环境变量名去读取其**值**，在消息里替换成 `REDACTION`。为什么需要？
  因为 Tiingo 客户端的 HTTP 错误消息里常常回显整个请求 URL，而 token 就在 query string 里。
- `CREDENTIAL_ENV_VARS` 是**数据**（模块级常量），不是从 client 类上取的属性——
  因为 client 类在测试里会被整个替换掉，一个走 client 类间接层的安全控制可以被一个
  「恰好没定义这些名字」的桩悄悄关掉。

### 7. 请求超时挂死 worker

`_AlpacaMarketDataClient.TIMEOUT_SECONDS = 60`。没有 timeout 的 `requests.get` 撞上
不响应的服务端会**永久挂住一个 worker 线程**；在 8 个 worker 的 15000 symbol 回填里，
这等于整个 run 静默死锁。

### 8. schema 漂移让整个目录读不出来

polars 扫一个目录时，从第一个文件推断出一份 schema 然后强制套到所有文件上。
只要有一个分片多了一列、少了一列、或者列顺序不同，**整个 vendor 根目录直接读不了**，
而且报错报的是文件名不是原因，看起来还是间歇性的（取决于 polars 先打开哪个文件）。
所以 `_write_shard` 一律 `frame.select(self.RAW_COLUMNS)` 投影，而且两个厂商都还额外
按 `RAW_SCHEMA` 做 dtype cast——因为一只股票某个窗口里 `divCash` 全是整数 0、
或者 `volume` 全 null，`pl.DataFrame(json)` 推断出来的类型就和兄弟文件不一样。

### 9. 幸存者偏差从一个被 `requests` 丢掉的参数溜进来

Alpaca 请求里的 `asof` 参数如果不传，厂商默认用**今天的** ticker→实体映射，
于是一个已退市代码会返回「今天占用这个代码的公司」的历史——干净、可信、完全错误的数据。
而 `"asof": None` **等于没传**：`requests` 在拼 query string 之前会丢掉值为 `None` 的参数。
所以必须发一个真实可编码的值，`AlpacaAcquisition.ASOF_NO_MAPPING = "-"`。

### 10. 盘中 K 线按 UTC 日切分区

`1m` 频率按 `date=` 做 hive 分区。如果这个 date 用 UTC 日期截断，
美股常规时段 14:30–21:00 UTC、盘后延伸到次日 01:00 UTC，那么**每天最后约 4 小时会被归到第二天**。
查「某一个交易日」时两头都是错的，而且错的形状是「数据稀疏」而不是「报错」——
收盘不见了，前一天的尾巴混进来了。所以有 `Acquisition.SESSION_TIME_ZONE`，
`Alpaca` 设为 `"America/New_York"`；**没声明就直接抛异常**，绝不退化成朴素截断。
注意只有**派生的分区键**做时区转换，时间戳**值**始终是 naive UTC。

### 11. 厂商回吐同一个 page token → 撑爆磁盘

分页循环唯一的退出条件是 token 为空。如果厂商（或代理、或部分故障）把你给它的 token
原样还回来，循环永不结束；而 `page_index` 每轮递增，所以**文件名一直在变**，
不是覆盖而是不停新增——每个请求看起来都成功，磁盘被填满。
`_fetch_batch` 里显式检查 `next_token == page_token` 并拒绝。

---

## 核心概念

- **vendor（厂商）**：`tiingo` / `alpaca`（`enums.data.Vendor`）。它不是装饰性字段：
  它是 `raw_data_dir_path` 的最后一段目录、`watermark_path` 的兄弟目录段、
  写进每个分片的一个字面量列、以及 `PageLedger.batch_key` 的哈希输入之一。
  两个厂商的数据因此在磁盘上物理隔离，且合并读取时仍能分辨来源。

- **raw 层（原始层）**：`data/downloads/{market}/{frequency}/{subdir}/{vendor}/` 下的
  hive 分区 parquet 树。**这个根目录下只能有 `.pqt` 文件**，任何一个 `.json`
  都会让 `pl.scan_parquet` 直接崩掉——这就是所有账本类文件都放在**兄弟目录**
  `.../{subdir}/_watermarks/{vendor}/` 而不是放在 raw 树里的原因。

- **hive key（分区键）**：由 `enums.data.RAW_HIVE_KEYS` 唯一声明，写方和读方共用一份：
  `1d → ("month",)`、`1m → ("date",)`、`tick → ("data_type", "date", "symbol")`。
  声明的**顺序**是目录嵌套顺序，重排会静默改写每一段路径。

- **batch（批次）**：一次请求携带的 symbol 集合。`Acquisition.DEFAULT_BATCH_SIZE = 1`
  是基类默认；`TiingoAcquisition.DEFAULT_BATCH_SIZE = 1`（EOD 端点单 symbol，这个 1 是
  **有承载意义的**，不是保守值）；`AlpacaAcquisition.DEFAULT_BATCH_SIZE = 100`。
  **batch 是成败的最小单位**：一批里任何一个环节失败，整批进失败清单、整批没有 watermark。

- **page（页）**：一次 HTTP 请求返回的一页数据。厂商用 `next_page_token` 串起来。
  **分页循环只在基类 `_fetch_batch` 里实现一次**；厂商只实现 `_fetch_page`（发一次请求）。
  不分页的厂商每次都返回 `None` token——这是对契约的**完整实现**，不是占位。

- **watermark（水位边车文件）**：`{watermark_root}/{symbol}.json`，内容形如
  `{"last_date": "...", "start_date": "...", "no_data": true}`。它记的是**覆盖区间**
  而不是一个端点。`start_date` 未知时**整个 key 省略**（不写 null），`no_data` 为假时
  也**整个省略**——因为「缺省」必须是默认值，这样旧文件读回来才是对的。

- **覆盖状态（coverage status）**：`_classify_coverage()` 给出的四选一：
  - `uncovered`：没边车，或 `last_date != config.end_date` → 要抓
  - `covered`：end 对得上且记录的起点 ≤ 请求起点 → 跳过
  - `widened`：end 对得上但记录的起点**晚于**请求起点 → 历史比要求的浅，要重抓
  - `legacy`：end 对得上但**起点未知**（老 schema 写的文件）→ 见下

- **legacy watermark 策略**：`Acquisition.LEGACY_WATERMARK_POLICIES = ("warn", "refetch")`，
  默认 `DEFAULT_LEGACY_WATERMARK_POLICY = "warn"`。`warn` = 跳过，但**每次运行都大声报数
  并打印修复命令**；`refetch` = 当作没覆盖重抓。为什么不猜起点？因为猜错就等于把
  第 5 条那个静默历史缺口原样复现一遍，而且更隐蔽。修复靠人工显式执行
  `Acquisition.STAMP_COMMAND_HINT` 里写明的命令。

- **失败清单（failure manifest）**：`{watermark_root}/_failures.json`
  （`Acquisition.FAILURE_MANIFEST_NAME`），`{symbol: 错误消息}`。**每次 run 覆盖重写**，
  所以它永远描述**最近一次 run**；空的 `{}` 是一句有意义的话：上次跑干净了。
  日志里只列头 `Acquisition._FAILURE_LOG_SAMPLE = 5` 个，全量在文件里。

  **它不是续跑输入。** 续跑完全由**水位边车的存在与否**驱动
  （03.4 D-18 的 FACTUAL CORRECTION 更正了此前相反的说法）。
  它被保留下来是因为它是**崩溃后仍然存在的运维记录**：进程死掉就没有
  `AcquisitionResult` 可以返回了。仓库内读它的有**两处**——`SourceInspector.failures()`
  （不带凭证的运维视图）和 `Acquisition._merge_unattempted_failures`（写清单前的合并）——
  两处都经由唯一那个容错读取器 `CoverageLedger.read_failure_manifest` 去读
  （两个容错读取器就是两份会各自漂移的失败策略）。
  （2026-09-09 更正，plan 03.4-09：这里原先写着仓库内没有代码读它、读者只有一个；
  03.4-05 把第二个读者加进来之后，那句话就不成立了。上面这两处是写这段时从 `quantlab/`
  源码里数出来的，不是从旧文案抄过来的。）

  写清单之前的那次**合并**——把盘上已有的、本轮从没轮到的条目折回来——自 03.4-08 起是
  **无条件**的：它紧挨在写入之前，覆盖续跑循环的**每一条退出路径**：取消、第一轮
  `pending` 就是空的、正常跑完、`wait_for_quota` 关着时的配额中止、`quota_max_waits`
  用尽的配额中止。于是 `set(result.failures) == set(_failures.json)` 在每一条退出路径上
  都成立——但要说清这个等式**是什么**：两边由同一个 dict 在同一处组装出来，所以它是
  「结果和清单是一起拼出来的」的**回执**，不是对任何一边是否正确的检查（阶段验收时它
  就曾在一个刚被清空的清单上读出 True）。真正保护运维的是清单**内容**的存活：默认路径
  上配额中止之后，上一轮的条目仍然在文件里，由 `tests/test_acquisition_progress.py`
  把文件从盘上读回来断言。

  它的写入现在是**原子**的，和水位边车一样，走
  `quantlab/utils/atomic.py:write_json_atomically`（临时文件 + `fsync` + `os.replace`）。
  以前是裸 `open(..., "w")` + `json.dump`：一次在批次边界上的中断，会把一个完好的
  边车先截断成空文件，然后才开始写——而取消恰恰就发生在批次边界上。

- **「查过了，没数据」标记（`no_data`）**：这是第四种状态，和「失败」「从没抓过」都不同。
  三个存储事实映射出四种状态：

  | 状态 | 边车文件 | `_failures.json` 条目 | `no_data` 键 |
  |---|---|---|---|
  | 从没抓过 | 无 | 无 | — |
  | 抓取失败 | 无 | **有** | — |
  | 抓到了数据 | 有 | 无 | 缺省 |
  | 问过了，厂商没有 | 有 | 无 | `true` |

  这个标记**只在整批 `complete`（厂商给了空 token）且没有全局中止时**、
  **按整批算一次**，绝不按页算。原因：Alpaca 是 symbol-major 排序的，
  100 个 symbol 的批次第 0 页合法地只包含 1 个 symbol；按页算差集会把另外 99 个
  标成「没数据」、推进水位、从此永久跳过——一次看起来完全成功的 99% 静默数据丢失。

- **配额（quota）与限速（rate limit）**：见上文表格。相关常量：
  `Acquisition.DEFAULT_WAIT_FOR_QUOTA = False`（默认**不**等，免得一个 run 悄悄占着
  厂商的一整个额度窗口）、`DEFAULT_QUOTA_WAIT_SECONDS = 3600`、`DEFAULT_QUOTA_MAX_WAITS = 3`
  （来自实测算术：约 4600 请求/窗口 vs 约 14674 个 symbol ≈ 三个窗口）；
  `DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 5.0`、`DEFAULT_RATE_LIMIT_MAX_RETRIES = 6`
  （5s × 6 ≈ 30s 耐心，够一个分钟窗口翻两次；超过就降级成 `failed` 进清单，
  让**下一次** run 去重试，而不是让**这一次** run 永远跑不完）。

- **knob（旋钮）**：所有每次运行可调的参数都走 `config.kwargs`，用 `Acquisition._knob()` 读，
  **不做构造函数参数**——这样配置文件永远能触达它，符合 CLAUDE.md 的「可复现性」约束。
  已有的：`batch_size`、`max_workers`、`progress`、`resume`、`legacy_watermarks`、
  `wait_for_quota`、`quota_wait_seconds`、`quota_max_waits`、
  `rate_limit_backoff_seconds`、`rate_limit_max_retries`；
  Alpaca 另有 `data_type`、`feed`、`adjustment`、`asof`、`page_limit`。

---

## 它是怎么工作的

### 整体流程

```
download(symbols)                 refresh(symbols)
   │  from_watermark=False           │  from_watermark=True
   └───────────────┬─────────────────┘
                   ▼
            Acquisition._run()
                   │
   ①  _validate_symbols(全部 symbol)      ← 在任何路径拼接之前
                   │
   ┌───────────────▼───────────────────────────────┐
   │  外层「等配额」重试循环（最多 quota_max_waits 次）│
   │                                               │
   │  ②  _partition_by_coverage()  ← 读每个边车一次 │
   │      → pending / covered / widened / legacy   │
   │        / no_data 计数，_report_coverage 打日志 │
   │      （resume=False 时整段跳过，全部 pending）  │
   │                                               │
   │  ③  _run_once(pending)                        │
   │      ├ _reset_abort()      新的全局中止 Event  │
   │      ├ 切批：_batches() 或 _refresh_batches()  │
   │      ├ emit run_started（total = 批次数）      │
   │      ├ joblib.Parallel(backend="threading",   │
   │      │      n_jobs=max_workers,               │
   │      │      return_as="generator_unordered")  │
   │      │    └─► _attempt_batch(batch)  × N 并发  │
   │      ├ 排干结果流，每落地一批 emit             │
   │      │      batch_completed(completed/total)   │
   │      │      首次观测到配额中止 → quota_exhausted│
   │      │      首次观测到取消     → cancelled      │
   │      └ finally: emit run_finished + 关闭 reporter│
   │                                               │
   │  ④  合并本轮 failures / succeeded              │
   │      被取消 → break（在配额分支之前！）        │
   │      未中止 → break；中止且 wait_for_quota →   │
   │      _sleep(quota_wait_seconds) 后重来         │
   └───────────────┬───────────────────────────────┘
                   ▼
   ⑤  （无条件，覆盖每一条退出路径）_merge_unattempted_failures()
       ← 把盘上本轮从没轮到的条目折回来
   ⑥  _write_failure_manifest()  →  _failures.json（原子写）
   ⑦  组装 AcquisitionResult 并挂到 last_result 上，供
      registry.run() 返回给调用方（requested / succeeded /
      failures / cancelled / quota_aborted / coverage）
```

进度和取消是这一层给**进程内调用方**（运维控制台）的两个把手，
`registry.run(descriptor, config, *, refresh, reporter, cancel)` 把它们透传进来；
细节见 [registry.md](registry.md)。三条要点：

- **进度是事件对象，不是 stderr。** `_run_once` 不再自己构造 `tqdm`，它 `_emit`
  `ProgressEvent`；`TqdmProgressReporter` 用同一个 `total`、同一个 `desc`、
  同一个 `unit="batch"` 把它渲染回原来的样子。渲染的**人**变了，渲染的**内容**没变。
  （测试断言的是 `tqdm` 的**构造参数**，不是捕获 stderr——进度条的实际字符宽度取决于终端；
  `_report_coverage` 那四条日志是逐字保留的，`coverage` 事件是**附加**的。）
  事件的 `message` 永远是发送方
  `_scrub` 过的——进度事件是异常字符串的一条新外泄路径，绕过那个唯一收口
  就是下一次泄露的形状。
- **一个 reporter 停不掉一次 run。** `emit` 返回 `None` 且调用方忽略返回值，
  所以「忘了返回 True」不会中止采集；`_emit` 又把每次调用包在 try/except 加一条
  脱敏 warning 里，所以 reporter 抛异常也中止不了。try/except 放在**调用方**
  而不是 reporter 里，因为异常要用厂商自己的 `CREDENTIAL_ENV_VARS` 脱敏，
  只有采集对象知道那些名字。
- **取消是一个独立的令牌，检查点在 `_attempt_batch` 的第一行。**
  `_should_stop()` = 配额中止 OR 取消令牌，位置和原来那条止血的
  `_abort.is_set()` 完全相同。为什么不是输入生成器：joblib 取消不了它已经排队的工作，
  `pre_dispatch` 默认 `2 * n_jobs`，生成器里的 `break` 只能拦住还没排队的那部分。
  过了这一行的批次会跑完并写下自己的水位，没过的批次一个边车都没有——
  这就是「取消之后仍可续跑」这句承诺的实际内容。
  **取消不是配额中止**：`_run` 在配额分支**之前**就 break 掉取消分支，
  不 `_sleep`、不等待、不打「额度耗尽」，结果对象上是
  `cancelled=True, quota_aborted=False`。

### 一个批次内部（`_attempt_batch` → `_fetch_batch`）

```
_attempt_batch(symbols, from_watermark)
  │
  ├─ **第一行**：if self._should_stop(): return "skipped"    ← 真正止血的地方
  │     （_should_stop() = self._abort.is_set() or self._is_cancelled()；
  │      配额中止和调用方的取消令牌共用这一个位置）
  │
  ├─ 读这批每个 symbol 的 coverage（一次）
  ├─ 决定请求起点：download → config.start_date
  │                refresh  → min(本批各 symbol 的 last_date)
  │
  ├─ while True:
  │     try: outcome = _fetch_batch(...)
  │     except Exception as exc:
  │         message = _scrub(f"{类型}: {exc}")        ← 凭证脱敏唯一出口
  │         verdict = _classify_error(exc)            ← 唯一的厂商策略 seam
  │           "quota"        → _abort.set(); return "quota"（不进清单）
  │           "rate_limited" → 退避 backoff 秒，重试同一批（最多 max_retries 次）
  │                            首次退避时记一条日志，附上厂商的 X-RateLimit-* 头
  │           "failed"       → return "failed"（整批进清单，无 watermark）
  │     break
  │
  ├─ 计算 no_data 集合：complete 且未中止时，symbols - outcome.symbols_with_data
  └─ 逐个写 watermark（last_date = config.end_date）


_fetch_batch(symbols, start, end)
  │
  ├─ _validate_symbols(symbols)                 ← 第二道，防 query string 注入
  ├─ ledger, batch_key = _ledger_for(...)       ← batch_key = sha256(vendor|freq|
  │                                                start|end|排序后的symbols)[:16]
  ├─ ledger.assert_consistent(raw_root)         ← 账本超前于磁盘 → 拒绝续跑
  ├─ ledger.is_complete() → ledger.reset()      ← 页账本只回答「从哪续」，
  │                                                不回答「该不该跑」
  ├─ page_index, page_token = ledger.resume_point()
  └─ while True:
        frame, next_token = _fetch_page(...)    ← 唯一的抽象方法，厂商实现
        shards = _write_shard(frame, ...)       ← **先落盘**
        ledger.record_page(...)                 ← **后记账**
        if not next_token: mark_complete(); return
        if next_token == page_token: raise      ← 防无限分页撑爆磁盘
        page_index += 1
```

### 并发、失败隔离、断点分别在哪

| 关注点 | 位置 |
|---|---|
| 并发 | `_run_once` 的 `joblib.Parallel(backend="threading")`，`DEFAULT_MAX_WORKERS = 8`。用线程不用进程：任务是网络 IO 密集的，且 client 对象是共享的 |
| 进度 | `_run_once` 发 `ProgressEvent`，默认由 `TqdmProgressReporter` 渲染成原来那条 stderr 进度条。事件发在**结果流**上（`return_as="generator_unordered"`），一格 = 一批真的落盘了；发在调用上会先空转几小时再瞬间满格 |
| 取消 | `CancelToken`，检查点是 `_attempt_batch` 的**第一行**（和全局中止同一位置，`_should_stop()` 是两者的 OR）。见 [registry.md](registry.md) |
| 失败隔离 | `_attempt_batch` 的 try/except，粒度 = 一个 batch |
| 全局中止 | `Acquisition._abort`（`threading.Event`），只有 `"quota"` 能触发 |
| symbol 级断点 | watermark 边车 + `_partition_by_coverage`。**每一轮都从磁盘重新算 pending**，所以「续跑逻辑」和「跳过逻辑」是同一份代码，不存在会漂移的平行账 |
| page 级断点 | `PageLedger`，每批一个文件在 `{watermark_root}/_pages/{batch_key}.pages.json`。见 [pageledger.md](pageledger.md) |
| 转换级断点 | 不在这一层。raw → Zarr 的分窗续跑是 `ChunkLedger`，见 [chunking.md](chunking.md) |

### 与 pageledger / chunking 的关系（一句话各自）

- **`quantlab/base/pageledger.py`**：采集层**内部**的断点记录。`_fetch_batch` 用它决定
  「这批从第几页、哪个 token 续」，并用 `assert_consistent()` 交叉验证磁盘。
  一批一个文件，所以没有跨线程共享可变状态，不需要锁。细节见 [pageledger.md](pageledger.md)。
- **`quantlab/base/chunking.py`**：采集层**下游**的东西，`Acquisition` 完全不 import 它。
  它服务于 `StockDataset.from_raw_data_chunked()`——把已经落盘的 raw parquet
  按年/季/月分窗，一窗一窗地稠密化并追加进 Zarr，让峰值内存随**窗口**而不是随**整段区间**增长。
  细节见 [chunking.md](chunking.md)。

---

## 完整例子

### 例 1：真跑过的最小例子（`--dry-run`，不需要任何凭证）

`ingest_us_equity.py --dry-run` 会解析标的池、算体量、打印路径、报告水位覆盖情况，
然后退出，**一个行情请求都不发，也不需要任何凭证**。

> 只想问「盘上已经有什么」而不想跑一个脚本？那就直接用 `SourceInspector`——
> 存量、覆盖、失败原因、行级浏览四件事它都在无凭证、零厂商请求的前提下回答。
> 见 [registry.md](registry.md) 的「只读检视器」一节。这个 `--dry-run` 里的覆盖报告
> 本身就是它算的。

```bash
cd /Users/daizhaorong/projects/quantlab
env -u TIINGO_API_KEY -u APCA_API_KEY_ID -u APCA_API_SECRET_KEY \
    uv run python ingest_us_equity.py --dry-run
```

真实输出（2026-09-09 在本机实际执行，三个凭证环境变量都用 `env -u` 显式清掉）：

```
DRY RUN -- category=us_all, no price requests issued
  symbols resolved:  13729
  preview:           ['NETDU', 'NBP', 'PRCP', 'PAYX', 'LIII-U', 'OPA-WS', 'PEPLU', 'SDSTW', 'CELL', 'SKT']
  window:            2016-01-01 .. 2026-09-08
  trading days (~):  2694
  dense grid cells:  36,985,926
  real observations: 17,145,396
  density:           0.464
  dense float64:     3.31 GiB
  observed float64:  1.53 GiB
  chunk granularity: year
  chunk count:       11
  whole-range total: 3.31 GiB (advisory -- chunking never materialises this at once)
  largest chunk:     0.31 GiB (2016-01-01..2016-12-31, 13729 pinned symbols x 253 trading days)
  per-chunk budget:  4.00 GiB (a finer --chunk is the remedy above this)
  raw-data path:     /Users/daizhaorong/projects/quantlab/data/downloads/us_equity/1d/us_all/tiingo
  watermark path:    /Users/daizhaorong/projects/quantlab/data/downloads/us_equity/1d/us_all/_watermarks/tiingo
  zarr path:         /Users/daizhaorong/projects/quantlab/data/data/us_equity/1d/us_all.zarr
  coverage report:
  already covered:   0 (would be skipped)
  re-fetch, widened: 0 (recorded coverage starts after --start-date)
  legacy, no start:  0 (stamp via --stamp-legacy-watermarks)
  would fetch:       13729/13729
```

几个值得看的点：

- `raw-data path` 以 `/tiingo` **结尾**，而 `watermark path` 是它的**兄弟**
  `us_all/_watermarks/tiingo`。前面说过：账本 JSON 绝不能落在 raw 树里。
- **覆盖报告是无条件打印的，一个凭证都不需要。** 这条命令上面那次运行是在
  `TIINGO_API_KEY` / `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` 全部被
  `env -u` 显式清掉的情况下跑的。以前这里会打印
  `coverage report: skipped (export TIINGO_API_KEY to see it)`——因为算它的唯一入口
  是 `Acquisition.coverage_report()`，而 `TiingoAcquisition.__init__` 在**构造时**
  就要凭证。于是「跑大任务之前先看看要抓多少」这个命令只对已经有 key 的人有用，
  恰好是最不需要它的那批人。现在它走 `SourceInspector`：不构造 client、
  不 import 任何厂商模块，因此**结构上**不可能发请求。判断口径没有第二份实现——
  `SourceInspector.coverage` 走的是真实 run 走的同一个
  `CoverageLedger.partition_by_coverage` 对象（D-09），所以这份报告和紧接着的抓取
  不可能对「什么叫覆盖」有分歧。详见 [registry.md](registry.md)。
- `would fetch: 13729/13729`：这台机器上 `us_all` 的水位树是空的，
  所以全部待抓。有边车的时候这四行会分别告诉你「跳过多少」「因为加宽而要重抓多少」
  「多少个是没有起点的 legacy 边车」。
- `密度 0.464`：全市场日线是稀疏的（退市股票只在自己活着的那段有数据），
  所以稠密面板会浪费一半以上的内存——这正是 `chunking` 存在的理由之一。

### 例 2：真跑过的第二个例子（没凭证时体量护栏和凭证拒绝的实际行为）

这条命令在**没有** Alpaca 凭证的环境里执行，能看到「护栏先跑、client 后建」的顺序。

> 注意这里演示的是**构造 client 时的凭证拒绝**——那是 `Acquisition` 的 fail-fast 守卫，
> 是对的，也不该被挪走。但如果你的问题其实是「没凭证的机器上我还能问什么」，
> 答案不是这条命令，而是 `SourceInspector`：它不构造任何 client，
> 所以存量 / 覆盖 / 失败原因 / 行级浏览四件事在这台机器上全都能答。
> 见 [registry.md](registry.md)。

```bash
uv run python ingest_alpaca.py --symbols AAPL,MSFT --frequency 1m \
    --start-date 2026-08-01 --end-date 2026-09-05
```

真实输出（同样是本机实际执行）：

```
Pre-flight volume estimate (zero vendor requests issued):
  roster:            (explicit --symbols list)
  window:            2026-08-01 .. 2026-09-05
  symbols:           2
  trading days (~):  25
  rows (~):          19,500 (390/symbol-day, density 1.000)
  raw on disk (~):   0.00 GiB
  requests (~):      2 (batch_size=100, page_limit=10,000)
  wall clock (~):    0.0 h at 200 req/min
Acquiring 2 symbol(s) from Alpaca (frequency=1m, data_type=None, refresh=False)
Traceback (most recent call last):
  File "/Users/daizhaorong/projects/quantlab/ingest_alpaca.py", line 348, in <module>
    acquisition = AlpacaAcquisition(acq_config)
  ...
RuntimeError: APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables must both be set. Alpaca market-data credentials are read from the environment and are never stored on the config (D-15) -- export them before running acquisition (see your Alpaca dashboard for the key pair).
```

这段输出证明了两件事：体量护栏（`assert_acquisition_volume_fits` +
`assert_dense_panel_fits`）在**任何 client 被构造之前**就跑完了；以及凭证只可能来自环境变量，
命令行里根本没有可以传 key 的地方（传了会被 argparse 拒绝）。

### 例 3：真实下载的例子（本次未再执行，但盘上的产物是真的）

**此例的命令行未实际运行**（当前环境没有 Alpaca 凭证）。但它的**产物是真的**：
今天确实跑过一次 Nasdaq-100 分钟线回填（2026-08-01..2026-09-05），
raw 树此刻就在仓库的 `data/` 下，下面所有数字都是我实测 `find` / `du` /
读文件得到的，不是编的。命令行本身是我按磁盘上的路径反推出来的形状
（`nasdaq_data` 是 `stock_acquisition_config` 的默认 `subdir`，
`nasdaq100` 是 `--universe` 的合法取值），仅供参考：

```bash
export APCA_API_KEY_ID=...        # 只写变量名，绝不写值
export APCA_API_SECRET_KEY=...

uv run python ingest_alpaca.py --universe nasdaq100 --as-of-date 2026-08-01 \
    --frequency 1m --start-date 2026-08-01 --end-date 2026-09-05
```

落盘产物：

```
data/downloads/us_equity/1m/nasdaq_data/
├── alpaca/                          ← raw 根，只有 .pqt
│   ├── date=2026-08-03/
│   │   ├── part-3c6a2e14da4b501b-00000.pqt
│   │   └── ... （每个 date= 目录约 105 个分片）
│   ├── ...
│   └── date=2026-09-04/
│                                    合计 2,645 个 .pqt / 48 MB
│                                    （另有一个 date=2024-01-02/，是更早的
│                                      单 symbol 试跑留下的，同样计入上面的总数）
└── _watermarks/alpaca/              ← 账本根，raw 的兄弟目录
    ├── AAPL.json  ...（102 个 symbol 边车）
    ├── _failures.json
    └── _pages/                      ← 5 个 PageLedger 文件（每批一个）
        └── 3c6a2e14da4b501b.pages.json
```

102 个 symbol、`batch_size=100` → 这一轮是 2 个批次。实测这 2 个批次的账本都是
`complete: true`（100 symbol / 133 页，2 symbol / 4 页）；`_pages/` 下另外 3 个文件是
更早的批次留下的，其中 2 个是 `complete: false` 且 `pages: []`——
`_fetch_batch` 一开始就会 `describe()` 并落盘账本骨架，所以一个还没抓到任何一页
就中断的批次也会留下文件。它们不影响后续运行：`batch_key` 里含 symbol 集合与窗口，
对不上就不会被复用。

`_failures.json` 的实际内容：

```json
{}
```

——空的，意思是最近一次 run 干净收尾，没有任何 symbol 失败。

`AAPL.json` 的实际内容：

```json
{"last_date": "2026-09-05", "start_date": "2026-08-01"}
```

——**两个端点都在**，所以下次如果你把 `--start-date` 提前到 2026-07-01，
`_classify_coverage` 会判成 `"widened"` 并重抓，而不是静默跳过。
没有 `no_data` 键，说明这只股票确实抓到了数据。

一个页账本（`3c6a2e14da4b501b.pages.json`）的开头，节选：

```json
{
  "batch_key": "3c6a2e14da4b501b",
  "vendor": "alpaca",
  "frequency": "1m",
  "start_date": "2026-08-01",
  "end_date": "2026-09-05",
  "symbol_count": 2,
  "symbol_fingerprint": "458eca15bcfe98346b7680d9c4171bc0ed70cc3a43cc55145842adedb8575165",
  "complete": true,
  "pages": [
    {
      "index": 0,
      "next_token": "QUFQTHxNfDE3ODcwNTQzNDAwMDAwMDAwMDA=",
      "rows": 10000,
      "shards": [
        ".../alpaca/date=2026-08-03/part-3c6a2e14da4b501b-00000.pqt",
        ".../alpaca/date=2026-08-04/part-3c6a2e14da4b501b-00000.pqt",
        "..."
      ]
    }
  ]
}
```

注意 `rows: 10000` —— 正好等于 `AlpacaAcquisition._fetch_page` 里 `page_limit`
这个 knob 的默认值（内联字面量 `10_000`，不是具名常量），也就是说这一页满了，
所以有 `next_token`，所以有第 1 页。**一页会同时写出多个分片**，
因为一页里的 10000 行横跨多个交易日，而 `1m` 按 `date=` 分区。

一个分片的实际内容（`date=2026-09-04/part-3c6a2e14da4b501b-00002.pqt`，885 行 × 10 列）：

```
┌─────────────────────┬────────┬────────┬────────┬───┬────────┬─────────┬─────────────┬────────────┐
│ timestamp           ┆ symbol ┆ vendor ┆ open   ┆ … ┆ close  ┆ volume  ┆ trade_count ┆ vwap       │
│ datetime[μs]        ┆ str    ┆ str    ┆ f64    ┆   ┆ f64    ┆ f64     ┆ f64         ┆ f64        │
╞═════════════════════╪════════╪════════╪════════╪═══╪════════╪═════════╪═════════════╪════════════╡
│ 2026-09-04 08:00:00 ┆ AAPL   ┆ alpaca ┆ 326.5  ┆ … ┆ 326.7  ┆ 25033.0 ┆ 1796.0      ┆ 327.452511 │
│ 2026-09-04 08:01:00 ┆ AAPL   ┆ alpaca ┆ 327.18 ┆ … ┆ 327.19 ┆ 18108.0 ┆ 952.0       ┆ 327.097793 │
└─────────────────────┴────────┴────────┴────────┴───┴────────┴─────────┴─────────────┴────────────┘
```

- 列名和顺序正好是 `AlpacaAcquisition.RAW_COLUMNS_BY_DATA_TYPE["bars"]`；
  `vendor` 是一个字面量列，所以就算有人把两个厂商的目录合并读，来源仍然可分辨。
- `timestamp` 是 **naive UTC**：`08:00:00` = 美东 04:00（盘前）。
  分区键 `date=2026-09-04` 是把它转成 `America/New_York` 之后取的日期。
  实测这份数据里所有行的 UTC 日期恰好都等于分区日期
  （这批 symbol 的分钟线止于 23:59 UTC，没有跨 UTC 午夜的盘后 bar），
  所以这里看不出转换的效果——但机制在，`1d` 频率也不受影响（日线按 `month=` 分区）。
- 分区目录名里没有 `symbol=`：`1m` 的 hive key 只有 `("date",)`，
  所以一个 `date=` 目录是被**所有并发批次共享**的。这就是
  `_clear_superseded_shards` 只在 `_partition_is_per_symbol`（即 tick）时才敢删旧分片的原因——
  在共享目录里删别人的文件会毁掉兄弟批次的数据。

### 例 4：几条典型运维命令（未实际执行，因为都需要凭证）

```bash
# 全市场 Tiingo 日线回填，约 15.4k symbol、数小时。中断后原样重跑即可续跑。
export TIINGO_API_KEY=...
uv run python ingest_us_equity.py

# 配额耗尽时坐等重置再续（默认关闭；关闭时会停止派发并提示你稍后重跑）
uv run python ingest_us_equity.py --wait-for-quota

# 只把已有回填顶到今天：每个 symbol 从自己的 last_date 起算
uv run python ingest_us_equity.py --refresh

# 一次性迁移：给所有缺 start_date 的老边车补上你自己知道的覆盖起点，零请求后退出。
# 注意仍然需要 export TIINGO_API_KEY —— TiingoAcquisition 在**构造时**就要凭证，
# 那时它还不知道这条路径一个请求都不会发。
uv run python ingest_us_equity.py --stamp-legacy-watermarks 2016-01-01

# Alpaca 全分辨率 quotes（--data-type 在 tick 下必填、在其他频率下会被拒绝）
uv run python ingest_alpaca.py --symbols AAPL --frequency tick \
    --data-type quotes --rows-per-symbol-day 1000000 \
    --start-date 2026-09-04 --end-date 2026-09-04
```

---

## 怎么接入一个新厂商

这个抽象存在的**全部意义**就是：新厂商只写「怎么发一次请求」和「怎么读懂一个异常」，
并发、续跑、失败隔离、脱敏、分片布局、分页全部白拿。

### 必须实现的（就三样）

```python
import polars as pl
from quantlab.base.acquisition import Acquisition


class MyVendorAcquisition(Acquisition):
    # 1) 厂商标识。写进每个分片的 vendor 列、每条路径的最后一段、
    #    以及 PageLedger.batch_key 的哈希输入。需要先加进 enums.data.Vendor。
    VENDOR = "myvendor"

    # 2) 分片的列投影 + 列顺序，钉死。_write_shard 会 select 它。
    #    必须以 ("timestamp", "symbol", "vendor") 开头。
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", "open", "high", "low", "close", "volume")

    # 3) 唯一的 @abstractmethod：发一次请求，返回 (rows, next_page_token)。
    def _fetch_page(self, symbols, start_date, end_date, page_token=None):
        ...
        return frame.select(self.RAW_COLUMNS), next_token_or_None
```

关于 `_fetch_page` 的三条硬要求：

1. 返回的 frame 必须能被 `select(self.RAW_COLUMNS)`，而且 **dtype 也要钉死**。
   两个现有厂商都定义了一个 `RAW_SCHEMA` 并显式 `.cast(...)`。
   注意：`RAW_SCHEMA` **不是基类契约的一部分**（基类里根本没引用它），
   它是两个厂商各自实现的同一个约定——但你也应该照做，理由见「不用它会怎样」第 8 条。
2. 空响应要返回**带 schema 的空 frame**，不能是裸 `pl.DataFrame()`。
   `_fetch_batch` 会去读它的 `symbol` 列，裸空 frame 会抛异常而不是报告「没数据」。
3. 不分页的厂商每次都返回 `None` token。这是完整实现，不是偷懒。

### 按需覆盖的

| 要覆盖的东西 | 什么时候需要 | 参考 |
|---|---|---|
| `CREDENTIAL_ENV_VARS` + `REDACTION` | 只要有凭证。**必须**从模块级常量取，不要从 client 类上取 | 两个厂商都有 |
| `DEFAULT_BATCH_SIZE` | 端点支持多 symbol 时。单 symbol 端点保持 1 —— 否则会在 `_fetch_page` 里偷偷循环，把失败粒度、续跑粒度和配额检查粒度一起放大 N 倍 | `AlpacaAcquisition.DEFAULT_BATCH_SIZE = 100` |
| `_classify_error(exc)` | 厂商有全局条件（配额）或瞬时条件（限速）时。基类默认全部归为 `"failed"`，这是最保守也最安全的默认 | `TiingoAcquisition._classify_error` / `AlpacaAcquisition._classify_error` |
| `RATE_LIMIT_STATUS_CODES` | 有「稍后重试就好」的状态码时。基类是空 frozenset —— 表示「本类对任何状态码不做限速判断」 | `AlpacaAcquisition.RATE_LIMIT_STATUS_CODES = frozenset({429})` |
| `_rate_limit_headers(exc)` | 厂商会回 `X-RateLimit-*` 之类的头时。**只用于日志**，不能拿来算恢复时刻 | `AlpacaAcquisition.RATE_LIMIT_HEADERS` |
| `SESSION_TIME_ZONE` | **只要支持任何盘中频率就必须设**。不设的话第一次盘中写盘就抛 `NotImplementedError` —— 这是故意的 | `AlpacaAcquisition.SESSION_TIME_ZONE = "America/New_York"` |
| `_session_date(expr)` | 厂商的时间戳**本来就是**交易所本地时间时，覆盖成一句 `expr.dt.date()`。这样它是个明确的声明而不是默认值 | 基类默认实现是 UTC→`SESSION_TIME_ZONE`→取日期 |
| `_data_type` | 只有当频率的 `RAW_HIVE_KEYS` 里含 `data_type` 时（目前只有 `tick`）。必须自己校验取值，并且**同一个解析结果既选端点又选投影**，否则目录名和列内容会互相矛盾 | `AlpacaAcquisition._data_type` |
| `RAW_COLUMNS` 做成 `@property` | 一个类要服务多个端点/多套列时 | `AlpacaAcquisition.RAW_COLUMNS` |

### 基类白送的（不要重写）

并发 fan-out、进度条、全局中止 Event、`resume` 分区与报告、`_refresh_batches` 的
按水位分组打包、分页循环、分片写入与 hive 分区、`batch_key` 计算与页账本接线、
watermark 读写与四态分类、`legacy` 策略、失败清单、`_scrub` 脱敏、
`_validate_symbols`、配额等待循环、`coverage_report()`、`stamp_watermarks()`。

### 还要做的三件配套事

1. 在 `quantlab/enums/data.py` 的 `Vendor` Literal 里加上你的厂商名。
2. 在 `quantlab/config/__init__.py` 的工厂（`stock_acquisition_config`）里通过 `vendor=` 参数走，
   **不要在调用点手工拼 `AcquisitionConfig`**。工厂是「raw 根以 vendor 结尾、
   watermark 是它的兄弟」这条约定唯一被推导的地方；在调用点手拼就是这条约定开始漂移的方式
   （`ingest_alpaca.py` 的模块 docstring 明确说了这点，还有测试断言它没有直接构造这两个 dataclass）。
3. **在你的采集类旁边注册一个描述符**（03.4 起新增的一步）。写完一个 `Acquisition`
   子类还不够——不注册，运维界面就枚举不到它，这个仓库的 ingest 薄壳也没法通过
   `DataSourceRegistry.get(...)` 拿到它：

   ```python
   MYVENDOR_SOURCE = register_source(
       SourceDescriptor(
           vendor="myvendor",
           display_name="My Vendor",
           acquisition_cls=MyVendorAcquisition,
           config_factory=functools.partial(stock_acquisition_config, vendor="myvendor"),
           capabilities=(Capability(market="us_equity", frequency="1d"),),
           required_env=("MYVENDOR_API_KEY",),   # 只有名字，永远不写值
       )
   )
   ```

   然后在 `quantlab/acquisition/registry.py` 的**文件底部**加一行
   `from quantlab.acquisition import myvendor as _myvendor`。

   **不要加到 `quantlab/acquisition/__init__.py` 里**，那个文件必须保持 0 字节：
   非空的包 `__init__` 会在每一次 `import quantlab.acquisition.<任何东西>` 时执行，
   包括 `quantlab.acquisition.universe`——而那个模块的全部结构性保证就是
   「这里不可能构造出任何 acquisition client」，并且这条保证会被**静默**侵蚀
   （体量护栏的结构臂是对 `universe.py` 自己源码的 AST 扫描，看不见传递 import）。
   底部这个位置同时也是描述符能写在厂商类旁边的原因：
   `tiingo.py` / `alpaca.py` import `registry.py` 拿装饰器，反过来不成立。

   描述符的字段含义、能力列表为什么不是叉乘、`is_configured` / `run()` /
   reporter / 取消令牌 / 检视器怎么用，全部见 [registry.md](registry.md)。

---

## 常见坑

1. **别往 raw 根目录里放任何非 parquet 文件。** polars 的目录扫描会走遍根下每个文件，
   一个 `.json` 就让整棵树读不出来。所有边车都必须在 `_watermarks/{vendor}/` 这个**兄弟**目录下。

2. **`refresh()` 不会加宽历史。** 它按 `[symbol 的 last_date, config.end_date]` 抓，
   **完全忽略**你调宽的 `config.start_date`；`_classify_coverage(from_watermark=True)`
   甚至只看 end 一个条件。想往前补历史只能用 `download()`。
   （为什么？因为 refresh 抓不到更早的数据，如果按加宽的 start 去判 pending，
   每次运行都会把所有 symbol 判成待抓，而重抓又永远填不上那个洞——一个无声的无限配额消耗。）

3. **`refresh()` 只能清除 `no_data` 标记，不能新打标记。** 它只查询了区间的后半段，
   却在往边车上盖 `[covered_start, end_date]` 这个更大的区间——它对前半段没有任何证据。
   代码里是 `no_data = no_data and bool(coverages[symbol].get("no_data"))` 这一行。

4. **legacy watermark 默认是「跳过 + 每次大声报警」，不是「重抓」。** 如果你看到
   `N symbol(s) carry a legacy watermark with NO recorded covered start`，
   那就是在告诉你：这 N 个被跳过了，而它们到底覆盖了什么区间**磁盘上无从得知**。
   要么执行 `--stamp-legacy-watermarks <START_DATE>`（值只能你自己提供，代码绝不猜），
   要么 `--legacy-watermarks refetch`。忽略它 = 接受一个你看不见的历史缺口。

5. **`_failures.json` 是跨 run 累积的运维记录，每次 run 覆盖重写。** 想留证据自己拷走。
   自 03.4-08 起，写盘之前会把「盘上已有、本轮从没轮到」的旧条目折回来，所以空的 `{}`
   只说明「本轮有消息的符号都干净、且盘上也没有别的遗留条目」，不是「还没跑过」。
   **它也不是续跑输入**——续跑只看水位边车在不在；仓库内读它的有**两处**
   （`SourceInspector.failures()` 与写盘前的 `Acquisition._merge_unattempted_failures`），
   两处都经由 `CoverageLedger.read_failure_manifest`。
   保留它是为了「进程崩了也还有一份记录」，以及给运维控制台读原因。

6. **`"quota"` 故意不进失败清单。** 所以配额中止之后，清单是空的，
   但这**不代表**所有 symbol 都成功了——被 `"skipped"` 的那些既不在成功集也不在失败集，
   因为这一轮对它们一无所知。判断进度要看 watermark，不要看清单。

7. **`download(resume=False)` 会无视页账本重跑整批。** `_fetch_batch` 里
   `ledger.is_complete()` 时调的是 `ledger.reset()` 而不是直接返回：
   页账本只回答「批内从哪续」，从来不回答「这批该不该跑」。后者是 symbol 级策略层的事。

8. **`batch_key` 里含窗口，所以换个窗口重抓会产生第二个文件名。**
   `1d`/`1m` 靠下游 `dedup_raw_frame(keep="last")` 吸收；
   **tick 故意不做去重**（真实的 quote 和 trade 合法地共享 `(timestamp, symbol)`），
   所以 tick 走 `_clear_superseded_shards` 删掉旧 key 的分片。
   这个删除**只在分区目录是单 symbol 时才安全**（tick 的 hive key 含 `symbol=`）。
   如果你以后给 `1m` 加了 `symbol=` 分区键，这条逻辑会自动生效——请确认那是你想要的。

9. **不要把 `enums.data.TRADEABLE_TICKER_PATTERN` 和
   `quantlab/acquisition/universe.py:_WELL_FORMED_TICKER` 「对齐」。** 它们守的是不同的输入：
   前者守「即将变成路径段和 query 参数的 symbol」，后者守「从 Wikipedia 抓来的变更日志单元格」，
   在后者那里出现三段式恰恰是解析出错的信号。曾经有过两份自由漂移的副本，
   结果 `NXG-R-W` 让一个多小时的全市场任务在发出第一个请求前就崩了
   （quick task 260907-10t 修的就是这个）。现在是**同一个编译对象**被两端绑定。

10. **不要把「重采样一下 tick 省点空间」当成优化。** `_fetch_page` 到分片之间
    没有任何聚合、分桶、去重，这是 tick 层的全部契约（D-16）。
    同理，Alpaca 的 quotes/trades 时间戳是**纳秒**（`RAW_SCHEMA_BY_DATA_TYPE` 里写死了
    `pl.Datetime("ns")`），用 polars 默认的微秒去解析就是一次伪装成解析器的重采样，
    它会在唯一不去重的那一层里制造出无法与真实同时性区分的时间戳并列。

11. **Alpaca 的 `feed`（SIP vs IEX）没有代码内默认值，两个方向都没有。**
    厂商自己的文档互相矛盾（免费档到底能不能取历史 SIP 数据），项目没有凭证去实证，
    所以不设时**整个参数被省略**，由厂商按订阅档位自己选。这不是遗漏，是刻意的留白。
    同样，`AlpacaAcquisition.DEFAULT_BATCH_SIZE = 100` 也是一个**保守工作值**，
    不是已验证的厂商上限（真实上限没有文档）。

12. **`tick` 频率的 run 到 raw 就停了。** `ingest_alpaca.py` 不会去转 Zarr，
    因为稠密 `[timestamp, symbol]` 面板表达不了不规则事件轴，这是另一套数据模型，
    留给后续 phase。脚本会明说，不会让你干等一个永远不会出现的 Zarr store。

13. **watermark 根目录会按 `data_type` 再套一层——但只对 tick。**
    `_watermark_root` 在 `RAW_HIVE_KEYS` 含 `data_type` 时会加一段
    `{watermark_path}/{data_type}/`。原因：quotes 和 trades 共用同一个 vendor raw 根，
    如果边车也共用，一次跑完的 quotes 回填会让后续的 trades run 以为整个池子都已覆盖，
    于是跳过全部标的、报告成功、实际一行没抓。`1d` / `1m` 的路径完全不变。

14. **`AcquisitionConfig` 里永远不要加凭证字段。** `to_dict()` 就是 `asdict(self)`，
    它会落进持久化配置和模型 checkpoint 旁边的 JSON。这个仓库已经这样泄露过一次真 key。
