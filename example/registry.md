# 数据源登记表（DataSourceRegistry）

> 代码位置：`quantlab/registry.py`（登记表、描述符、`run()`）、
> `quantlab/acquisition/_support/inspector.py`（只读检视器）、`quantlab/base/progress.py`（进度事件与取消令牌）。
> 描述符本身定义在各厂商模块里：`quantlab/acquisition/tiingo.py`、`quantlab/acquisition/alpaca.py`。
> 采集引擎本身见 [acquisition.md](acquisition.md)；分页断点见 [pageledger.md](pageledger.md)。

---

## 一句话

把「这个项目到底能下载什么」做成**一份可枚举的登记表**：每个厂商写一个描述符，
描述符自带能力列表、凭证变量名、采集类和 config 工厂；
加一个新数据源 = 在厂商类旁边注册一个描述符，而不是改五个调用点。

它服务两个消费者：本仓库的三个 ingest 脚本，以及仓库之外的运维控制台
`quantlab-console`（另一个仓库，依赖 quantlab，反向永不成立）。
两者从**同一份定义**里枚举数据源——这就是这一层存在的全部理由。

---

## 不用它会怎样

下面每一条都是这个仓库在 2026-09-08 之前的真实状态，不是假想。

### 1. 厂商类硬编码在每个调用点上，没有任何可枚举的清单

`ingest_tiingo.py` 里写死 `TiingoAcquisition`，`ingest_alpaca.py` 里写死 `AlpacaAcquisition`。
一个运维界面想问「这个项目支持哪些数据源」，**没有地方可以问**——答案只存在于
三个脚本的 import 语句里。加第三个厂商意味着再复制一遍这套调用点，
而不是新增一条数据。

### 2. argparse 的默认值在 parser 定义时就把厂商钉死了

这是比构造函数调用更隐蔽的一处绑定：`--max-workers` 的默认值、
`--legacy-watermarks` 的 `choices` 和默认策略、两个配额旋钮的默认值……
在 `_build_arg_parser()` 里就要读厂商类常量。`ingest_us_equity.py` 里有**六处**，
`ingest_alpaca.py` 里还有 tick 的 `choices`、`DEFAULT_BATCH_SIZE` 和 bars 列集。
只把构造函数换成登记表、不动这些默认值，脚本表面上「不再点名厂商」了，
实际上每次 `--help` 仍然在读那个类——一个看起来达标、实际没达标的重构。
现在它们全部走 `SOURCE.acquisition_cls`，并且是**动态**证明的：
测试把一个桩描述符塞进去，断言 `--help` 里出现的是桩的荒唐常量。

### 3. 一个纯本地文件读的报告，因为没有 key 而被跳过

`ingest_us_equity.py --dry-run` 里的覆盖报告本质上只是 `open()` + `json.load()`
读一堆水位边车，一个行情请求都不发。但它以前会打印
`coverage report: skipped (export TIINGO_API_KEY to see it)`——
因为算这个报告的唯一入口是 `Acquisition.coverage_report()`，
而 `TiingoAcquisition.__init__` 在**构造时**就要凭证。
于是「跑大任务之前先看看要抓多少」这个命令，只对已经有 key 的人有用，
恰好是最不需要它的那批人。`SourceInspector` 把这条路打通了：
不构造任何 client、不 import 任何厂商模块，因此**结构上**不可能发请求。

### 4. 「能力」如果用市场 × 频率的笛卡尔积表达，会广告出厂商拒绝的组合

Alpaca 的 `tick` 只有 `us_equity`，而且还要再分 `quotes` / `trades`；
Tiingo 只有 `us_equity` 的 `1d`。用两个扁平元组做叉乘，就会告诉运维
「Tiingo 支持 tick」。一个宣传了却抓不到的组合，比什么都不宣传更糟。
所以能力是一个 `Capability` **列表**，不是叉乘（D-01 / D-02）。

### 5. 一个多小时的回填，在进程内跑起来之后没有任何办法看进度或停下来

写入路径以前的契约是「控制台 spawn 一个子进程跑 quantlab 的 CLI」，
那时候「取消」就是 `kill`、「进度」就是读 stderr。2026-09-08 这条契约被推翻，
写入改成**进程内**调用（D-12）。代价是取消和进度必须由 quantlab 自己提供，
否则控制台只能眼睁睁看着自己的 UI 线程被占住几个小时。

---

## 核心概念

- **描述符（`SourceDescriptor`）**：**一个厂商一个**，不是 `(厂商, 市场, 频率)` 三元组一个。
  厂商级别的事实（凭证、显示名、采集类）只说一次；随市场/频率变化的东西
  放进 `capabilities` 里。frozen dataclass。

- **能力（`Capability`）**：一个 `(market, frequency, data_type)` 组合。
  `data_type=None` 表示「厂商这里没有这个区分」（Tiingo 的 EOD 端点只供一种东西，
  也没给它起名字），**不是通配符**。可选字段 `earliest_available` / `entitlement`
  是**每能力**的，不是每描述符的——只有部分行能填的字段应该往下放，而不是把容器加宽。

- **`acquisition_cls` 是直接类引用**（D-03），和 `UniverseCatalog.MEMBERSHIP_FETCHERS`
  一个路子，不是 `utils/module.py:get_cls_from_path` 的点分路径。
  接受的代价：import 登记表就会 import 所有厂商 SDK。这是 import 开销，不是脆弱性——
  `tiingo` 和 `requests` 本来就是硬依赖。

- **`config_factory`**：通常是 `functools.partial(stock_acquisition_config, vendor="...")`。
  全仓库只有**一个**采集 config 工厂同时服务两个厂商，所以「工厂」就是一个钉死了
  vendor 的 partial；写成两个每厂商函数只会是同一段函数体差一个关键字。
  用 partial 而不是给描述符加方法，是为了让描述符保持一个纯 frozen dataclass。

- **`required_env`：只有变量名，永远没有值。** `is_configured()` 返回 `bool`，
  `credential_status()` 返回 `{名字: bool}`。没有 API 会返回、打印或嵌入一个凭证值，
  **也没有「打码版本」**——打了一半码的 key 看起来很负责任，实际只是缩小了搜索空间，
  下一次泄露就是这个形状。这个仓库真的泄露过一次 Tiingo key（Phase 1）。

- **描述符里没有传输目标**：没有 base URL、没有 host、没有端点前缀。
  把线路目的地做成可配置项，就等于把运维控制台变成一个凭证外发面。
  `_AlpacaMarketDataClient.BASE_URL` 留在它自己的模块里钉死。

- **`universe_categories` 是纯建议性的**：标的池来自 `UniverseCatalog` 而不是厂商，
  所以这个字段只是给控制台一个「这个源旁边可以列哪些池」的展示提示，
  **`run()` 完全不读它**。两个厂商都是美股，四个 category 对谁都成立，
  写窄了反而是在编码一个并不存在的区分。

- **注册用装饰器**（`register_source`，D-06）：定义即注册，忘不掉。
  它原样返回描述符（按 identity），所以模块级的那个名字确实是描述符而不是 `None`。
  拒绝两件事，都抛 `ValueError`：重复的 vendor（第二个市场是新增一个 `Capability`，
  不是新增一个描述符），以及**空的 capabilities**（一个什么都不供的源是定义错误，
  不是可枚举的源）。

- **`SOURCES` 是 tuple，靠 `+=` 重新绑定，绝不 `.append`。** 这不是风格偏好：
  测试夹具 `isolated_registry` 用 `monkeypatch.setattr` 隔离，它保存的是**旧对象**
  并在拆卸时放回去；`list.append` 会原地改掉那个正要被放回去的对象，
  夹具于是静默地不再隔离，一个假描述符会漏进本次会话后面所有测试里。

---

## 它是怎么工作的

### 描述符定义在哪、厂商模块什么时候被 import

```
quantlab/registry.py
  ├─ 顶部：Capability / SourceDescriptor / DataSourceRegistry /
  │        register_source / is_configured / credential_status / run
  │        —— 这一段必须保持「不认识任何厂商」
  │
  └─ 底部（文件最后两行）：
        from quantlab.acquisition import alpaca as _alpaca
        from quantlab.acquisition import tiingo as _tiingo
```

三件事同时由「底部」这个位置解决：

1. **冷启动枚举完整性（D-07）。** 装饰器填出来的登记表，完整性只等于「已经被 import 的模块集合」。
   所以 import 这个模块必须把所有厂商模块带进来。
2. **`quantlab/acquisition/__init__.py` 保持 0 字节。** 非空的包 `__init__` 会在
   **每一次** `import quantlab.acquisition.<任何东西>` 时执行——包括
   `quantlab.universe`，而那个模块的全部结构性保证就是
   「这里不可能构造出任何 acquisition client」。更糟的是它会**静默**地被侵蚀：
   `tests/test_volume_guard.py` 的结构臂是对 `universe.py` **自己源码**的 AST 扫描，
   看不见被包 `__init__` 拖进来的传递 import。（D-07 的 AMENDED 块记录了这次机制变更。）
3. **描述符可以定义在它自己的厂商类旁边**：`tiingo.py` / `alpaca.py` import
   `registry.py` 拿装饰器，所以 `registry.py` 不能在顶部 import 它们。

用的是**模块对象**形式（`from quantlab.acquisition import tiingo`）而不是
`from ...tiingo import TiingoAcquisition`：当调用方先 import 了 `tiingo`，
本模块会在 `tiingo` 只初始化了一半的时候运行，绑定模块对象是安全的，
读它的属性会抛异常。两行的先后无所谓——`all()` 会排序。

### 为什么 `all()` 排序而不是按 import 顺序

枚举顺序会变成运维界面的**显示顺序**，而 import 顺序取决于调用方碰巧先 import 了哪个模块：
`import quantlab.acquisition.tiingo` 和 `import quantlab.registry`
会把同一套安装渲染成两种顺序。排序还顺带让针对这个方法的断言变成一次字面元组比较。

### 三个 ingest 脚本怎么消费它

`ingest_tiingo.py` / `ingest_alpaca.py` / `ingest_us_equity.py` 现在都是**薄壳**：

```python
SOURCE = DataSourceRegistry.get("tiingo")      # 脚本的身份就是它的厂商
...
acq_config = SOURCE.config_factory(...)        # 不再直接调 stock_acquisition_config
result = run(SOURCE, acq_config, refresh=args.refresh)
```

一个细节值得记住：SC-1 的「调用点不得点名厂商」指的是不得点名厂商**类**，
不是不得出现厂商**字符串**。`DataSourceRegistry.get("tiingo")` 留着是对的——
一个脚本的身份就是它的厂商；更严格的读法需要一个 `--source` 参数，
而那正是被否决掉的「合并成一个 CLI」。

---

## 程序化启动一次采集：`run()`

```python
run(descriptor, config, *, refresh=False, reporter=None, cancel=None) -> AcquisitionResult
```

- 厂商只通过 `descriptor.acquisition_cls` 触达，调用方不点名任何采集类。
- **只做采集，到 raw parquet 层为止**（D-14 AMENDED）。它不做 raw→Zarr 转换：
  三个入口的转换模式本来就不一样（无条件整窗 / 按频率 / `--to-zarr` 下分块），
  各自带一个大小不同的内存护栏，合成一个调用等于让其中两个用错护栏。
  控制台真要转换，那是**另一个** registry 级调用，不是 `run()` 上的一个 flag。
- 构造 `descriptor.acquisition_cls(config)` 是**第一个**要凭证的地方，这是故意的：
  那是厂商类自己的 fail-fast 守卫，往后挪就把 fail-fast 变成了 fail-late。
- 返回 `AcquisitionResult`：`vendor` / `requested` / `succeeded` / `failures` /
  `cancelled` / `quota_aborted` / `coverage`。`failures` 的值**已经脱敏过**——
  它们就是 `_attempt_batch` 经 `_scrub` 产出的那些字符串（Tiingo 的报错会把
  带 token 的完整 URL 回显出来）。

### 进度：事件对象 + 可插拔 reporter（D-16）

`_run_once` 不再自己构造 `tqdm`，它 `_emit` 事件；渲染的人变了，渲染的内容没变。
`TqdmProgressReporter` 用的是同一个 `total`、同一个 `desc`、同一个 `unit="batch"`。

| 事件 kind | 什么时候发 |
|---|---|
| `run_started` | 一轮 fan-out 即将开始，`total` = 本轮批次数，`message` = 进度条描述 |
| `coverage` | 本轮的续跑分区，`detail` 就是 `_report_coverage` 打的那几个数 |
| `batch_completed` | 一个批次的结果落地，`completed` 从 1 递增到 N |
| `batch_failed` | **保留未用**，目前没有任何地方发它 |
| `quota_exhausted` | 厂商配额耗尽，剩余批次在被排干而不是在抓 |
| `cancelled` | 轮内观测到取消令牌 |
| `run_finished` | 结果流被排干到底 |

三条规则：

1. **`message` 永远是发送方脱敏过的**，绝不是厂商异常原文。进度事件是异常字符串的
   一条**新**外泄路径，绕过 `_scrub` 这个唯一收口就是下一次泄露的形状。
2. **一个 reporter 停不掉一次 run**，两个方向都停不掉：`emit` 返回 `None` 且调用方忽略返回值
   （所以「忘了返回 True」不会中止采集），而 `_emit` 把每次调用包在 try/except 里
   并打一条脱敏过的 warning（所以 reporter 抛异常也中止不了）。
   try/except 放在**调用方**而不是 reporter 里，是因为异常要用厂商自己的
   `CREDENTIAL_ENV_VARS` 去脱敏，而只有采集对象知道那些名字——这样
   `progress.py` 才能保持一个「不认识凭证」的叶子模块。
3. `reporter=None` 得到的是**今天的行为**：那条 stderr 进度条。`progress=False` 旋钮
   解析成 `NullProgressReporter`（根本不构造 bar 对象）。

### 取消：独立的令牌，检查点在批次边界（D-17）

`CancelToken` 是**独立**的，不是 reporter 的返回值。检查它的地方是
`Acquisition._attempt_batch` 的**第一行**（`_should_stop()` = 配额中止 OR 取消令牌）。

为什么必须是第一行而不是输入生成器：joblib 取消不了它已经排队的工作。
`pre_dispatch` 默认是 `2 * n_jobs`，所以生成器里的 `break` 只能拦住还没排队的那部分；
真正止血的是每个批次进来时的那次检查——剩下每个批次都变成微秒级的空操作，
零个厂商请求。

**取消不是配额中止。** `_run` 在配额分支**之前**就 break 掉取消分支：
不 `_sleep`、不等待、不续跑、不打「额度耗尽」的消息，
`AcquisitionResult` 上是 `cancelled=True, quota_aborted=False`。

**取消之后 store 仍然可续跑**：过了那一行的批次都跑完并写了自己的水位，
没过的批次一个边车都没有。加上边车写入现在是原子的
（`quantlab/utils/atomic.py:write_json_atomically`），这条承诺才真的成立。

---

## 只读检视器：`SourceInspector`

它回答「盘上已经有什么」，**不枚举数据源**——那是登记表的活。
在这里 import 登记表会把两个厂商模块拖进来，把一个结构性保证降级成一句约定。

**零厂商请求是结构性的，不是靠调用顺序。** 这个模块不 import 任何厂商 client、
不绑定任何 `Acquisition` 子类，所以这里**没有任何东西能**打开一个 socket。

| 方法 | 回答什么 | 必填参数 |
|---|---|---|
| `coverage(config, symbols=None)` | 请求区间对现有水位的四态分类 + `no_data` 计数 | `config` |
| `failures(config)` | `_failures.json` 里**当前已知仍在失败**的全部符号（跨 run 累积，可能含本轮根本没请求过的符号），`{symbol: 原因}` | `config` |
| `inventory(config, dataset_config=None)` | raw 层和 Zarr 层**分开**报告的存量 | `config` |
| `browse_raw(dataset_config, symbols, start_date, end_date)` | raw parquet 层的**惰性** `pl.LazyFrame` | 四个全部必填 |
| `browse_zarr(dataset_config, symbols, start_date, end_date)` | Zarr 层的 `xr.Dataset` 切片 | 四个全部必填 |

几条设计约束：

- **覆盖判断不在这里重新实现**（D-09）。`coverage()` 走的是
  `CoverageLedger.partition_by_coverage`——真实 run 走的**同一个对象**。
  四态规则加上正交的 `no_data`，细微到「简单重写一遍」一定会把 `legacy` 搞错，
  而两个今天一致的答案正是运维日后信错一个的方式。
- **`symbols` 和日期区间是必填的**（D-11），没有默认值，少传一个就是调用点 `TypeError`。
  惰性对象技术上允许你要走一整层，堵住这一点的是**参数**而不是返回类型：
  要全量就得自己显式把整个 roster 传进来（`us_all` 约 1.54 万 symbol × 约 5200 交易日）。
- **`browse_raw` 不自己开 parquet 扫描**，它走 `StockDataset._scan_raw` 再追加
  symbol 谓词和排序。`_scan_raw` 已经带了四个分别实测过的修复
  （vendor 根 basename 断言、tick 的 `_scan_root` 下钻、钉死的 `hive_schema`、
  保持抛错的 `extra_columns` / `missing_columns`）。代价是要付
  `nautilus_trader` 的冷 import（约 1.7 秒），这个取舍写在模块 docstring 里：
  用一秒换掉四个已经修好的 bug，而不是在这里重开一个扫描——
  那正是日后有人为了「让它跑起来」加上 `extra_columns="ignore"` 的地方。
- **无状态是契约，不是巧合。** 每个方法自己建 reader 和 ledger，实例上不存任何东西。
  `XrBackend.filter_by_date` / `filter_by_symbol` 会赋值回 `self.data`，
  共享一个 backend 会**永久变窄**（quick task 260906-w3t 修的就是这个）。
- **不做缓存。** 实测：遍历 26,584 个文件的 raw 根约 0.05 秒（热），
  读 7,756 个水位边车约 1.65 秒——贵的是边车那一半，和直觉相反。
  刷新策略归控制台，它才知道操作员是刚敲了个键还是刚跑完一次回填。

---

## 完整例子

### 例 1：枚举所有数据源和凭证状态（真跑过，无凭证）

```bash
cd /Users/daizhaorong/projects/quantlab
env -u TIINGO_API_KEY -u APCA_API_KEY_ID -u APCA_API_SECRET_KEY uv run python - <<'PY'
from quantlab.registry import DataSourceRegistry, is_configured, credential_status

for d in DataSourceRegistry.all():
    print(f"{d.vendor:8} {d.display_name:22} configured={is_configured(d)}")
    print(f"         required_env={d.required_env}")
    print(f"         status={credential_status(d)}")
    for c in d.capabilities:
        print(f"         - {c.market} / {c.frequency} / {c.data_type}")
PY
```

真实输出（2026-09-09 本机实际执行，三个凭证变量都被显式 unset）：

```
alpaca   Alpaca Market Data     configured=False
         required_env=('APCA_API_KEY_ID', 'APCA_API_SECRET_KEY')
         status={'APCA_API_KEY_ID': False, 'APCA_API_SECRET_KEY': False}
         - us_equity / 1d / bars
         - us_equity / 1m / bars
         - us_equity / tick / quotes
         - us_equity / tick / trades
tiingo   Tiingo EOD             configured=False
         required_env=('TIINGO_API_KEY',)
         status={'TIINGO_API_KEY': False}
         - us_equity / 1d / None
```

注意三件事：没有凭证也能完整回答；输出里只有变量**名**；`alpaca` 排在 `tiingo` 前面
是因为 `all()` 排序，不是因为它先被 import。

### 例 2：能力查询与错误信息（真跑过）

```python
from quantlab.registry import DataSourceRegistry

t = DataSourceRegistry.get("tiingo")
a = DataSourceRegistry.get("alpaca")
print("tiingo supports us_equity/1d      :", t.supports("us_equity", "1d"))
print("tiingo supports us_equity/tick    :", t.supports("us_equity", "tick"))
print("alpaca supports us_equity/tick    :", a.supports("us_equity", "tick"))
print("alpaca supports tick/quotes       :", a.supports("us_equity", "tick", "quotes"))
print("alpaca supports tick/bars         :", a.supports("us_equity", "tick", "bars"))
print("alpaca universe_categories        :", a.universe_categories)
DataSourceRegistry.get("polygon")
```

真实输出：

```
tiingo supports us_equity/1d      : True
tiingo supports us_equity/tick    : False
alpaca supports us_equity/tick    : True
alpaca supports tick/quotes       : True
alpaca supports tick/bars         : False
alpaca universe_categories        : ('nasdaq_all', 'us_all', 'sp500_constituent', 'nasdaq100_constituent')
ValueError: No data source is registered for vendor 'polygon'. Registered vendors: ['alpaca', 'tiingo']. A source registers itself when its module is imported; if this vendor's module was never imported, the registry cannot know about it (see the vendor imports at the bottom of quantlab/registry.py).
```

（最后一行的 traceback 正文省略了，只留 `ValueError` 那行；消息文本原样。）

`tiingo.supports("us_equity", "tick")` 是 `False` 而不是被 Tiingo 自己那条
`data_type=None` 的能力意外命中成 `True`——`data_type=None` 参数表示「不关心」，
而能力里的 `data_type=None` 表示「厂商这里没有这个区分」。

### 例 3：冷进程里的枚举完整性（真跑过）

这是 D-07 真正的验收条件。必须在**子进程**里跑：当前 pytest / REPL 会话可能
早就 import 过两个厂商模块了，在进程内断言只会测出会话状态，测不出 import 图。

```python
import subprocess, sys
code = "from quantlab.registry import DataSourceRegistry;" \
       "print(tuple(d.vendor for d in DataSourceRegistry.all()))"
print(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip())
```

真实输出：

```
('alpaca', 'tiingo')
```

### 例 4：用描述符构造 config（真跑过）

```python
from quantlab.registry import DataSourceRegistry

S = DataSourceRegistry.get("tiingo")
cfg = S.config_factory(market="us_equity", frequency="1d",
                       start_date="2024-01-01", end_date="2024-01-31",
                       symbols=("AAPL", "MSFT"))
print("vendor        :", cfg.vendor)
print("raw path      :", cfg.raw_data_dir_path)
print("watermark path:", cfg.watermark_path)
```

真实输出：

```
vendor        : tiingo
raw path      : /Users/daizhaorong/projects/quantlab/data/downloads/us_equity/1d/nasdaq_data/tiingo
watermark path: /Users/daizhaorong/projects/quantlab/data/downloads/us_equity/1d/nasdaq_data/_watermarks/tiingo
```

raw 根以 `/tiingo` **结尾**，水位目录是它的**兄弟**——这条约定只在工厂里被推导一次，
所以调用点绝不该手拼 `AcquisitionConfig`。

### 例 5：检视器回答存量、覆盖和失败（真跑过，无凭证）

用的是仓库 `data/` 下真实存在的一次 Nasdaq-100 分钟线回填。

```python
from quantlab.acquisition._support.inspector import SourceInspector
from quantlab.registry import DataSourceRegistry

S = DataSourceRegistry.get("alpaca")
cfg = S.config_factory(market="us_equity", frequency="1m", subdir="nasdaq_data",
                       start_date="2024-01-01", end_date="2024-01-31",
                       symbols=("AAPL", "MSFT"))
insp = SourceInspector()
inv = insp.inventory(cfg)
for k, v in inv["raw"].items():
    print(f"   {k}: {v}")
print("zarr        :", inv["zarr"])
print("coverage    :", insp.coverage(cfg))
print("failures    :", insp.failures(cfg))
```

真实输出：

```
   vendor: alpaca
   market: us_equity
   frequency: 1m
   data_type: None
   root: /Users/daizhaorong/projects/quantlab/data/downloads/us_equity/1m/nasdaq_data/alpaca
   exists: True
   shards: 2645
   bytes: 44833357
   watermark_root: /Users/daizhaorong/projects/quantlab/data/downloads/us_equity/1m/nasdaq_data/_watermarks/alpaca
   symbols_with_watermark: 102
   coverage_start: 2026-08-01
   coverage_last_date: 2026-09-05
   no_data: 0
   failures: 0
zarr        : None
coverage    : {'requested': 2, 'pending': 2, 'skipped': 0, 'covered': 0, 'widened': 0, 'legacy': 0, 'no_data': 0}
failures    : {}
```

`zarr` 是 `None` 而不是一堆 0，因为没传 `dataset_config`——「没问」和「没有」是两回事。
`coverage` 里 `pending=2`：盘上那 102 个 symbol 的水位覆盖的是 2026-08-01..2026-09-05，
而这里问的是 2024-01，端点对不上，所以两个都要抓。

### 例 6：惰性行级浏览（真跑过）

```python
import polars as pl
from quantlab.acquisition._support.inspector import SourceInspector
from quantlab.config import stock_kline_config

ds_cfg = stock_kline_config(
    market="us_equity", frequency="1m", subdir="nasdaq_data", vendor="alpaca",
    start_date="2026-08-01", end_date="2026-09-05", symbols=("AAPL", "MSFT"),
)
insp = SourceInspector()
lf = insp.browse_raw(ds_cfg, ["AAPL"], "2026-08-03", "2026-08-05")
print(type(lf).__name__)
print(lf.select("timestamp", "symbol", "open", "close", "volume").head(3).collect())
insp.browse_raw(ds_cfg, [], "2026-08-03", "2026-08-05")
```

真实输出：

```
LazyFrame
shape: (3, 5)
┌─────────────────────┬────────┬────────┬────────┬─────────┐
│ timestamp           ┆ symbol ┆ open   ┆ close  ┆ volume  │
│ ---                 ┆ ---    ┆ ---    ┆ ---    ┆ ---     │
│ datetime[μs]        ┆ str    ┆ f64    ┆ f64    ┆ f64     │
╞═════════════════════╪════════╪════════╪════════╪═════════╡
│ 2026-08-03 08:00:00 ┆ AAPL   ┆ 309.0  ┆ 308.66 ┆ 26055.0 │
│ 2026-08-03 08:01:00 ┆ AAPL   ┆ 308.66 ┆ 308.66 ┆ 52549.0 │
│ 2026-08-03 08:02:00 ┆ AAPL   ┆ 308.8  ┆ 308.01 ┆ 13594.0 │
└─────────────────────┴────────┴────────┴────────┴─────────┘
ValueError: browse_raw: symbols must be a NON-EMPTY sequence. D-11 makes symbols and the date window required arguments precisely so the lazy handle is already narrow when it is handed out (us_all is ~15.4k symbols x ~5.2k trading days, ~30M rows); an empty list is the same unbounded request wearing a different hat, so it is refused rather than answered with zero rows.
```

（上面两段 `ValueError` 都省略了 traceback 正文，只留最后一行；消息文本本身是原样贴的。）

返回的是 `LazyFrame`，什么时候 `collect()` 由调用方决定；空 symbol 列表被明确拒绝，
不会被解释成「给我全部」。

### 例 7：reporter 与取消令牌（真跑过，不需要厂商）

```python
from quantlab.base.progress import (
    CallbackProgressReporter, CancelToken, ProgressEvent, EVENT_KINDS,
)

seen = []
reporter = CallbackProgressReporter(seen.append)
reporter.emit(ProgressEvent(kind="run_started", vendor="tiingo", total=3,
                            message="TIINGO 2024-01-01..2024-01-31"))
reporter.emit(ProgressEvent(kind="batch_completed", vendor="tiingo",
                            completed=1, total=3, symbols=("AAPL",)))
reporter.close()
for e in seen:
    print(f"{e.kind:16} {e.completed}/{e.total} symbols={e.symbols} message={e.message!r}")

token = CancelToken()
print("token           :", repr(token), token.is_cancelled())
token.cancel()
print("after cancel()  :", repr(token), token.is_cancelled())
token.cancel()
print("cancel() twice  :", token.is_cancelled())
token.reset()
print("after reset()   :", token.is_cancelled())
```

真实输出：

```
run_started      0/3 symbols=() message='TIINGO 2024-01-01..2024-01-31'
batch_completed  1/3 symbols=('AAPL',) message=None
token           : CancelToken(cancelled=False) False
after cancel()  : CancelToken(cancelled=True) True
cancel() twice  : True
after reset()   : False
```

`cancel()` 调两次、或者在 run 结束之后调，都是无操作：不抛异常、不会再写一次
失败清单、不会再产出一个结果。

### 例 8：控制台侧的完整调用形状（**此例未实际运行**，需要真实凭证）

```python
import threading

from quantlab.registry import DataSourceRegistry, is_configured, run
from quantlab.base.progress import CallbackProgressReporter, CancelToken

SOURCE = DataSourceRegistry.get("tiingo")
if not is_configured(SOURCE):
    raise SystemExit(f"未配置：{SOURCE.required_env}")   # 只报名字，不报值

cfg = SOURCE.config_factory(market="us_equity", frequency="1d",
                            start_date="2024-01-01", end_date="2024-01-31",
                            symbols=("AAPL", "MSFT"))

cancel = CancelToken()
threading.Timer(5.0, cancel.cancel).start()             # 5 秒后在批次边界停下

def on_event(event):
    print(f"[{event.kind}] {event.completed}/{event.total}")

result = run(SOURCE, cfg, reporter=CallbackProgressReporter(on_event), cancel=cancel)
print(result.vendor, len(result.succeeded), len(result.failures), result.cancelled)
```

**此例未在本机执行**（没有 Tiingo 凭证），所以上面没有贴输出。
它的每一个组成部分都被测试覆盖：事件序列、取消后 store 仍可续跑，以及结果对象与失败
清单之间的关系。

那个关系**不是相等**（REVIEW CR-01 起）：`len(result.failures)` 是「**本轮**发现了几个
失败」，而 `_failures.json` 是跨 run 累积的运维记录，可能还留着别的 run 遗留、本轮根本
没请求过的条目。成立的是包含关系——`set(result.failures) ⊆ set(_failures.json)`，且共有
key 上消息一致；再加上 `set(result.failures) ⊆ set(result.requested)`。旧的
`set(result.failures) == set(_failures.json)` 已作废：两边本是同一个 dict 在同一处组装
出来的回执，不是对任何一边是否正确的检查，而且它会让一次换了 roster 的正常跑把上一轮的
404 当成自己的报出来。真正的回归在 `tests/test_acquisition_progress.py`：默认路径上配额
中止之后、以及换成不相交 roster 正常跑完之后，上一轮的条目仍然在清单**内容**里，而结果
对象是空的。

---

## 扩展：加第三个数据源

只有一步是登记表这层的事：**在你的采集类旁边注册一个描述符**。
采集类本身怎么写见 [acquisition.md](acquisition.md) 的「怎么接入一个新厂商」。

```python
import functools

import polars as pl

from quantlab.base.acquisition import Acquisition
from quantlab.registry import (
    Capability, SourceDescriptor, register_source,
)
from quantlab.config import stock_acquisition_config


class DemoAcquisition(Acquisition):
    VENDOR = "demo"
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", "close")
    CREDENTIAL_ENV_VARS = ("DEMO_API_KEY",)

    def _fetch_page(self, symbols, start_date, end_date, page_token=None):
        ...


# 就在类下面，同一个文件里。
DEMO_SOURCE = register_source(
    SourceDescriptor(
        vendor="demo",
        display_name="Demo Vendor",
        acquisition_cls=DemoAcquisition,
        config_factory=functools.partial(stock_acquisition_config, vendor="demo"),
        capabilities=(
            Capability(market="us_equity", frequency="1d", data_type=None),
        ),
        # 只有名字。这里永远不会出现一个值。
        required_env=("DEMO_API_KEY",),
    )
)
```

真实输出（把上面这段存成一个模块跑起来，2026-09-09 本机实际执行）：

```
all() -> ('alpaca', 'demo', 'tiingo')
get('demo') -> Demo Vendor
supports us_equity/1d -> True
is_configured -> False
duplicate -> vendor 'demo' is already registered ('Demo Vendor'). ONE descriptor per VENDOR (D-01) -- express a second market/frequency/data_type as another Capability on the existing descriptor, not as a second descriptor.
empty capabilities -> Refusing to register vendor 'empty' with an empty `capabilities` tuple. A source that serves nothing is a definition error, not an enumerable source -- declare at least one Capability(market=..., frequency=...).
```

还要做的三件配套事：

1. 在 `quantlab/enums/data.py` 的 `Vendor` Literal 里加上 `"demo"`。
2. 在 `quantlab/registry.py` **底部**加一行
   `from quantlab.acquisition import demo as _demo`——不是加到
   `quantlab/acquisition/__init__.py` 里，理由见上面「它是怎么工作的」。
3. 如果这个厂商需要一个自己的 ingest 薄壳，照 `ingest_tiingo.py` 的形状写：
   `DataSourceRegistry.get("demo")` 拿描述符，所有厂商常量走
   `SOURCE.acquisition_cls`，config 走 `SOURCE.config_factory`，下载走 `run()`。

---

## 常见坑

1. **不要给描述符加 base URL / host / 端点前缀。** 把线路目的地做成可配置项，
   就是把运维控制台变成一个凭证外发面。

2. **不要给凭证做「打码版本」的 API。** `is_configured` 返回 `bool`、
   `credential_status` 返回 `{名字: bool}`，就这样。半打码的 key 只是缩小了搜索空间。

3. **不要把 `SOURCES` 换成 list。** 保持 tuple + `+=` 重新绑定。
   换成 `list` 哪怕仍然写 `SOURCES += (d,)`，那也是 `list.__iadd__` 的原地 extend，
   `isolated_registry` 夹具的隔离会静默失效。

4. **不要在 `quantlab/acquisition/__init__.py` 里放 import。** 它必须保持 0 字节。
   厂商 import 在 `registry.py` 底部。

5. **不要用 `market × frequency` 叉乘表达能力。** Alpaca 的 tick 只有 us_equity，
   还要再分 quotes / trades；叉乘会广告出厂商拒绝的组合。

6. **`universe_categories` 是建议性的，别拿它当门禁。** `run()` 不读它。
   标的池来自 `UniverseCatalog`，不来自厂商。

7. **不要在 `SourceInspector` 里 import 登记表。** 那会把两个厂商模块拖进来，
   把「结构上不可能发请求」降级成「我们碰巧没调 client」。
   控制台问登记表有哪些源，问检视器某个源盘上有什么。

8. **不要给 `browse_raw` / `browse_zarr` 的 `symbols` 和日期加默认值。**
   D-11 让它们必填，正是因为惰性对象本身拦不住「给我一整层」。

9. **不要把取消做成 reporter 的返回值。** 一个只想记日志的 reporter
   不该因为忘了返回某个值就中止一次几小时的回填。

10. **`run()` 不转 Zarr，别给它加 `to_zarr=` 参数。** 三个入口的转换模式和内存护栏
    各不相同；真要暴露转换，那是另一个 registry 级调用。

11. **`_failures.json` 不是续跑输入。** 续跑完全由水位边车的**存在与否**驱动。
    它被保留是因为它是崩溃后仍然存在的运维记录（进程死掉就没有 `AcquisitionResult` 了）。
    仓库内读它的有两处——`SourceInspector.failures()`（检视器那一侧）和写清单前的
    `Acquisition._merge_unattempted_failures`——两处都经由唯一那个容错读取器
    `CoverageLedger.read_failure_manifest` 去读（2026-09-09 更正，plan 03.4-09：
    这里原先写着 quantlab 里没有代码读它，03.4-05 把第二个读者加进来之后就不成立了）。
