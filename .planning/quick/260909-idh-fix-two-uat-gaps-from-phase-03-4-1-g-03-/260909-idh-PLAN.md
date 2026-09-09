---
phase: quick-260909-idh
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/acquisition/universe.py
  - quantlab/utils/cli.py
  - quantlab/dataset/stock.py
  - ingest_alpaca.py
  - ingest_tiingo.py
  - ingest_us_equity.py
  - tests/test_universe.py
  - tests/test_ingest_conversion_gate.py
  - README.md
  - example/acquisition.md
autonomous: true
requirements: [G-03.4-1, G-03.4-2]
estimate:
  tokens: 75000
  raw_tokens: 75000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "G-03.4-2：`UniverseCatalog.get_symbols_in_range` 与 `get_symbols_as_of` 对同一组参数每次返回**逐元素相等**的 list，且这个顺序由**内容**决定（symbol 升序），不由 parquet 的行序、文件发现顺序或 polars 的多线程 collect 决定。因此 `quantlab/utils/cli.py:resolve_symbols` 的 `symbols[:limit]` 每次截到同一批符号，第二次 run 能命中第一次写下的 watermark 而跳过 —— 这是 CLAUDE.md「全流程参数尽量通过配置文件驱动、服务于实验可复现」这条硬约束在 roster 解析这一层的直接体现。"
    - "`--limit` 的 help 文本承诺的「the first N resolved symbols」现在是一个真实存在的顺序，并在 help 里点明它是升序；两个查询方法的 docstring 把「返回值有序、调用方会切片」写成契约，而不是留给下一个人从实现里猜。"
    - "G-03.4-1（a）：`result.succeeded` 为空**且**转换要读的 raw 根上没有任何 shard 时，ingest 脚本打印一段能读懂的说明（vendor raw 根、本轮 succeeded/failed 计数、失败清单去哪读、没有写出任何 Zarr）并以**非零退出码**结束，不再冒出 `quantlab/dataset/stock.py:304` 的未捕获 ValueError traceback。"
    - "这个守卫的两半条件缺一不可：全部符号被 **skip**（watermark 已覆盖）的 run 同样 `succeeded` 为空，但盘上有 raw 数据，它必须照常转换。区分二者的正是那次 raw 根探测，而不是计数。"
    - "G-03.4-1（b）：三个 ingest 脚本的 Zarr 转换**都**只在显式 `--to-zarr` 时发生。默认不转换，并在默认路径上明说「跳过转换，用 --to-zarr 转」，让缺省行为可见而不是沉默。这是对 `ingest_alpaca.py` / `ingest_tiingo.py` 既有默认行为的**故意变更**（用户已锁定）。"
    - "raw 根「有没有数据」这个判据在仓库里只有一处定义，`StockDataset._scan_raw` 的 absent-root 分支和 CLI 守卫读的是**同一个**判据；两处各写一份 `root.exists() / rglob` 就是这两个 gap 共同的祖先形状（一个契约裂成两份）。"
    - "守卫按**可达性**接线，不按脚本名：任何在 `__main__` 里densify 的入口都带着它 —— 包括 `ingest_us_equity.py --to-zarr`，它的 `from_raw_data_chunked` 有同一个洞。`tests/test_volume_guard.py:test_every_entry_point_that_densifies_guards_the_dense_panels_ram` 的教训（按脚本名圈定作用域，于是第二扇门的缺口对它不可见）在这里被复用而不是重犯。"
    - "RAM 护栏 `assert_dense_panel_fits` 跟着转换一起变成条件性的（`if args.to_zarr`），照抄 `ingest_us_equity.py` 对 `assert_chunked_panel_fits` 的既有形状。否则一次不转换的取数会被一个「为不会发生的稠密化算出来的」体量拒绝 —— 那是这次修复自己造出来的新缺陷。护栏的**词法位置**仍在 `run(...)` 之前，`tests/test_volume_guard.py` 的两个 AST 顺序断言仍然为真。"
    - "`--frequency tick` 搭 `--to-zarr` 在 argparse 层被 `parser.error` 拒绝（exit 2），照 `_validate_data_type` 的既有形状；静默忽略这个组合，正是本任务通篇在反对的那种沉默。"
    - "仓库里没有任何一句文档 / docstring / 注释仍然声称这两个脚本会无条件转 Zarr。判定按**逐句**做（先 grep 枚举候选，再逐句判读），字面量扫描只是从属的 tripwire —— 03.4 阶段同一形状的缺陷复发四次的教训。"
    - "example/ 里贴的真实输出一律不伪造：凡是贴出的输出里含转换，就把**命令行**改成能复现该输出的那一条（补 `--to-zarr`），绝不改输出本身（03.4-07 的 example/ 约定）。"
    - "新加的回归测试被 **mutation 验证**过，不是「到手就绿因此接受」：把两处 `.unique()` 改回无序形式后，至少有一条断言转红，并在 SUMMARY 里写清哪几条红了、哪几条没红。小 fixture 上「两次调用相等」这条完全可能天然为绿——本阶段已经因为这种真空锁付过四次代价。"
  artifacts:
    - quantlab/acquisition/universe.py
    - quantlab/utils/cli.py
    - quantlab/dataset/stock.py
    - ingest_alpaca.py
    - ingest_tiingo.py
    - ingest_us_equity.py
    - tests/test_universe.py
    - tests/test_ingest_conversion_gate.py
    - README.md
    - example/acquisition.md
  key_links:
    - "`get_symbols_in_range` / `get_symbols_as_of` 的排序 → `utils/cli.py:resolve_symbols` 的 `symbols[:limit]` → 每个符号的 watermark sidecar：顺序确定性是 resume/skip 能工作的前提。"
    - "`StockDataset` 的 raw-root 判据 → CLI 守卫的探测：同源，不可复制。"
    - "CLI 守卫的位置 → 每个 `from_raw_data` / `from_raw_data_chunked` 调用点之前（词法上、AST 上都成立）。"
    - "`args.to_zarr` → `assert_dense_panel_fits` 的条件 → `run(...)` 之前的词法位置：三者同时成立才不破坏既有的 SC-6 顺序断言。"
---

<objective>
关掉 phase 03.4 UAT 留下的两个 major gap，两个都是「沉默」而不是「报错」：

- **G-03.4-1**：`ingest_alpaca.py` / `ingest_tiingo.py` 在全部符号取数失败后仍无条件进入
  Zarr 转换，以 `quantlab/dataset/stock.py:304` 的未捕获 ValueError traceback 收场。修两半：
  (a) 转换前加 zero-success 守卫，干净地非零退出；(b) 把 Zarr 转换改成显式 `--to-zarr`
  opt-in，与 `ingest_us_equity.py` 对齐，终止三个 ingest 脚本之间的行为分歧。
- **G-03.4-2**：`UniverseCatalog` 的两个 roster 查询以顺序不定的 `.unique()` 收尾，
  同一份 config 每次解析出**不同顺序**的 roster，`--limit N` 于是每次截到任意的另一批符号，
  resume/skip 永远不触发。这直接违反 CLAUDE.md 的可复现性硬约束。

Purpose: 让「同一份配置 → 同一批符号」和「取数全失败 → 干净退出」这两件本该不言自明的事
成为被测试钉住的性质，并顺手把守卫按可达性铺到第三扇门（`ingest_us_equity.py --to-zarr`
有同一个洞）。

Output: 两个查询的确定性排序 + 三个 ingest 入口共享的一个转换守卫和一个 `--to-zarr` 开关，
以及被 mutation 验证过的回归测试与一次逐句文档真值扫除。
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/03.4-data-source-registry/03.4-UAT.md
@CLAUDE.md

工作树里有**与本任务无关**的既有未提交改动（`cal.py`、`pyproject.toml`、`test.py`、
`uv.lock`）。不要碰它们，不要把它们纳入任何一次 commit —— 每个 task 只 `git add` 自己
`<files>` 里列出的路径。
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: roster 解析变成内容决定的确定顺序（G-03.4-2）</name>
  <files>quantlab/acquisition/universe.py, quantlab/utils/cli.py, tests/test_universe.py</files>
  <behavior>
    - 同一 catalog 上连续两次 `get_symbols_in_range("us_all", s, e)` 返回逐元素相等的 list。
    - 同一 catalog 上连续两次 `get_symbols_as_of("us_all", d)` 返回逐元素相等的 list。
    - 两者的返回值都等于 `sorted(返回值)`（升序契约），且无重复（既有去重性质不回退）。
    - 结构性：`quantlab/acquisition/universe.py` 里这两个方法的返回表达式中，
      `.unique()` 被一个 `.sort(...)` 包在外层；去掉那层 `.sort(...)` 会让这条断言转红。
    - 既有的 26 条 in_range / as_of 断言（边界、拒绝、去重、覆盖起点）全部保持绿。
  </behavior>
  <action>
在 `quantlab/acquisition/universe.py` 的两个收尾表达式（`get_symbols_in_range`，UAT 记为
第 1587 行；`get_symbols_as_of`，第 2355 行 —— 已核对，两处仍是
`matched.select("symbol").unique().collect()["symbol"].to_list()`）之间插入按 symbol 升序
的排序，使返回顺序由内容而非存储布局决定。两处改成同一形状，不要一处排序一处不排。

在其中一处（另一处以一行注释指向它，不要写两份论证）写清为什么是排序而不是
`maintain_order=True`：`maintain_order=True` 只把顺序钉在**当前这份 parquet 的行序**上；
universe 表是由 `refresh_us_equity_universe.py` 周期性重写的，一次重写就会悄悄换掉
`--limit` 截到的那批符号 —— 同一个可复现性窟窿往下挪了一层。按 symbol 排序则让顺序成为
membership 集合的函数：同一个集合永远给出同一个 roster 顺序，与文件数、hive 布局、
polars 引擎内部实现都无关。`quantlab/base/acquisition.py:1099` 的 `maintain_order=True`
是另一个问题（保持调用方已经排好的帧序），不是本处的先例。

把返回顺序写进两个方法的 docstring，作为**契约**而不是实现细节：调用方会切片
（`utils/cli.py:resolve_symbols` 的 `symbols[:limit]`），所以顺序是被依赖的公开行为。
一并说明这条契约存在的理由（同一 config 每次解析出同一 roster，CLAUDE.md 可复现性约束），
让下一个想「顺序反正没人看，删掉排序省一次 sort」的人先撞上这段话。

在 `quantlab/utils/cli.py` 的 `--limit` help（第 225-231 行，现文本承诺
"Process only the first N resolved symbols"）里把「first N」是**哪个**顺序的 first N
点明为升序，使承诺与现在真实存在的顺序一致。

在 `tests/test_universe.py` 末尾按该文件既有形状（`mock_universe_fetchers` + `tmp_path`
+ `_make_config`，见第 1103 行的去重测试）加回归测试，三条断言分工写清：
两次调用相等（UAT 点名的性质）、等于 `sorted(...)`（顺序契约，小 fixture 上不会真空）、
AST 结构断言（`.unique()` 外层必须有 `.sort(...)`——这条是 fixture 恰好稳定时唯一还能红的
那条）。测试 docstring 里记一句：若将来出现 null symbol，`sorted()` 会当场 TypeError 而不是
和 polars 的 nulls-first 静默分歧，这是刻意选择的响亮失败。

写完后**跑一次 mutation**：把两处排序临时去掉，重跑本任务新加的测试，记下哪几条转红、
哪几条仍绿，然后还原。结果写进 SUMMARY。本仓库自己的规矩：a lock that passes on arrival
is mutation-verified rather than accepted（03.4 阶段已为此付过四次代价）。
  </action>
  <verify>
    <automated>uv run pytest tests/test_universe.py -q</automated>
  </verify>
  <done>tests/test_universe.py 全绿（含 3 条新断言）；两处返回表达式均带升序排序且论证记在源码里；`--limit` help 与两个 docstring 都陈述了升序契约；mutation 结果（哪条红、哪条没红）已记入 SUMMARY。</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: 转换守卫 + `--to-zarr` opt-in，按可达性铺到三扇门（G-03.4-1）</name>
  <files>quantlab/dataset/stock.py, quantlab/utils/cli.py, ingest_alpaca.py, ingest_tiingo.py, ingest_us_equity.py, tests/test_ingest_conversion_gate.py</files>
  <reversibility rating="costly">改的是两个既有脚本的默认行为（默认不再转 Zarr）。代码层面一行可逆，但已经在用这两个脚本的人的调用方式会变；用户已锁定要改，故只标记不设 checkpoint。</reversibility>
  <behavior>
    - `succeeded` 为空 **且** raw 根无 shard → 守卫抛 `SystemExit`，退出码非零，消息里出现
      vendor raw 根路径、本轮 succeeded/failed 计数、失败清单的读取入口，且不含任何凭证值。
    - `succeeded` 为空 **但** raw 根有 shard（全部符号被 watermark 跳过的 run）→ 守卫放行，
      照常转换。这条是守卫为什么要探盘而不是只看计数。
    - `succeeded` 非空 → 守卫放行（不再探盘）。
    - 三个 ingest 脚本的 parser 都注册 `--to-zarr`，且都默认 `False`。
    - `ingest_alpaca.py --frequency tick --to-zarr` 被 `parser.error` 拒绝，退出码 2。
    - AST：每个在 `__main__` 里调用 `from_raw_data` / `from_raw_data_chunked` 的入口，
      其守卫调用的行号都严格小于该densify 调用的行号（照
      `tests/test_volume_guard.py:_call_linenos` 的既有做法，按 Call 节点而非子串计数）。
    - `tests/test_volume_guard.py` 与 `tests/test_ingest_shells.py` 保持全绿
      （含 SC-6 的两条顺序断言与 densify-RAM-guard 可达性断言）。
  </behavior>
  <action>
**判据只写一处。** 在 `quantlab/dataset/stock.py` 里把 `_scan_raw`（第 294-311 行）
absent-root 分支现在内联的那个「根存在且至少有一个 `.pqt`」判断，提成一个公开的、
供 shell 调用的谓词（例如 `StockDataset.has_raw_data()`，用同一个 `_scan_root()` 求根），
并让 `_scan_raw` 改为调用它 —— 两处各写一份 `exists()/rglob` 正是这两个 gap 共同的
祖先形状。既有的 absent-root ValueError 文案与行为一个字都不改：它是正确的下层信号，
本任务只是给它一个上层调用点把它翻译成干净退出。

**拒绝只写一处。** 在 `quantlab/utils/cli.py` 加一个共享守卫（例如
`refuse_conversion_without_raw_data(dataset, result)`）：`result.succeeded` 非空、或
`dataset.has_raw_data()` 为真时直接返回；否则 `raise SystemExit(<消息>)`。消息要点名
vendor raw 根、本轮 succeeded/failed 计数、失败清单从哪读（`SourceInspector.failures()`），
并明说没有写出任何 Zarr、这不是转换层的 bug 而是取数没拿到任何数据。绝不打印凭证值或
厂商响应体。这个函数**只接受已经构造好的对象**，不新增任何 module-scope import ——
`quantlab/utils/cli.py` 的 module docstring 与
`tests/test_data_dir_cli.py:test_utils_cli_does_not_import_config_at_module_scope`
把这个模块的轻依赖钉死了。如果该 docstring 里「opens no file, issues no request」这句
因为这次改动不再逐字为真，就在同一次提交里把它改准确，不要留一句被推翻的陈述站在那里
（03.4-07 的规矩）。

**`--to-zarr` 也只定义一处。** `ingest_us_equity.py` 第 361-370 行已有一份带 chunk 语义的
定义。把 flag 名、`action="store_true"` 和共有的那句 help 提到 `quantlab/utils/cli.py` 的
一个注册函数里（模式差异——分块 vs 整窗——作为参数传入，照
`add_concurrency_args(default_max_workers=...)` 的既有先例），三个脚本都用它注册。
`ingest_us_equity.py` 的 help 内容不要被削掉：
`tests/test_ingest_shells.py:test_us_equity_keeps_every_capability_that_makes_it_distinct`
把它列为七项区别之一。

**接线。** `ingest_alpaca.py` 第 403-405 行与 `ingest_tiingo.py` 第 190-192 行的转换分支，
以及 `ingest_us_equity.py` 第 542-547 行的分块转换分支，三处都改成：先过守卫，再在
`args.to_zarr` 为真时转换；为假时打印一行「默认跳过转换，用 --to-zarr 转」。
`ingest_alpaca.py` 的 tick 分支现有的那段 D-18 说明保持不变。守卫按**可达性**铺满三扇门，
不按脚本名圈定：`ingest_us_equity.py --to-zarr` 在全失败时会撞上同一个 absent-root
ValueError，按脚本名圈定作用域正是
`tests/test_volume_guard.py:test_every_entry_point_that_densifies_guards_the_dense_panels_ram`
的 docstring 记下的那次教训。

**RAM 护栏跟着转换一起条件化。** `ingest_alpaca.py` 第 421-450 行区间的
`assert_dense_panel_fits`（现条件 `args.frequency != "tick"`）与 `ingest_tiingo.py`
第 176 行的同名调用，都再加上 `args.to_zarr` 这一半，照 `ingest_us_equity.py`
第 506 行 `if args.to_zarr:` 包住 `assert_chunked_panel_fits` 的既有形状。理由写在注释里：
这个护栏度量的是稠密化的 RAM，稠密化不发生时用它拒绝取数，就是这次修复自己造出来的
新缺陷。**位置不动** —— 仍在 `run(...)` 之前，两条 AST 顺序断言（SC-6 的
guard-precedes-fetch、densify-RAM-guard）读的是行号，嵌进 `if` 不影响，但改完必须实跑
`tests/test_volume_guard.py` 确认，不要靠推理。

**tick + `--to-zarr` 显式拒绝。** 在 `ingest_alpaca.py` 的 `_validate_data_type` 旁按同样
形状（`parser.error`，exit 2）拒绝这个组合，理由指向 D-18：不规则事件轴的转换不存在。
静默忽略这个 flag 组合，就是本任务通篇在反对的那种沉默。

新建 `tests/test_ingest_conversion_gate.py`，覆盖 `<behavior>` 里的七条：三条守卫行为
（用一个最小的假 result 对象和一个真实 tmp_path 上的 raw 根构造，跳过的那条要在盘上真的
放一个 `.pqt`）、三个 parser 的 flag 存在性（走**真实** parser 而不是扫源码 ——
`tests/test_ingest_shells.py` 的 L-5 教训：注册被挪进一个永不执行的分支时扫源码看不出来）、
tick 组合的 exit 2、以及那条按 Call 节点计数的 AST 顺序断言。
  </action>
  <verify>
    <automated>uv run pytest tests/test_ingest_conversion_gate.py tests/test_volume_guard.py tests/test_ingest_shells.py tests/test_stock_dataset.py tests/test_data_dir_cli.py -q</automated>
  </verify>
  <done>三个脚本的 parser 都有 `--to-zarr` 且默认不转换；zero-success 守卫接在三扇门上、两半条件都被测试钉住；tick+`--to-zarr` 以 exit 2 被拒；上述五个测试文件全绿。</done>
</task>

<task type="auto">
  <name>Task 3: 逐句扫除仍在声称「无条件转 Zarr」的文档与注释</name>
  <files>README.md, example/acquisition.md, ingest_alpaca.py, ingest_tiingo.py, tests/test_ingest_shells.py, tests/test_volume_guard.py</files>
  <action>
默认行为变了，所以一批**当时为真、现在为假**的句子仍站在仓库里。按 03.4 阶段的教训做：
先枚举候选，再逐句判读；字面量扫描只是从属 tripwire，不构成完备性主张。

枚举（execution time 实跑，不要信这份清单本身）：
`grep -rn "ingest_tiingo\|ingest_alpaca" README.md example/ ingest_*.py tests/` 与
`grep -rni "zarr" ingest_alpaca.py ingest_tiingo.py README.md`，把命中逐句读一遍，
判「这句在新默认下是否仍然为真」。

已知必须处理的站点（核对过行号，判读时以实际内容为准）：
- `README.md:266-268`：把三扇门都描述为会转换、只是模式不同的那句。
- `README.md:269-272`：`ingest_tiingo.py` 条目里「fetches … then converts … into a Zarr
  store」的因果链。
- `README.md:280-285`：`ingest_alpaca.py` 条目里「then converts bars to Zarr」。
  条目里关于 tick 停在 raw 的那句仍然为真，保留。
- `ingest_tiingo.py:1-6`：模块 docstring 第一段把整条链描述成一次必然发生的流水线。
- `ingest_tiingo.py:19-30`：Usage 里的四条示例命令 —— 凡是意在展示落 Zarr 的，补 `--to-zarr`。
- `ingest_tiingo.py:170-176` 与 `:186-189`：两段注释各自断言了底部那次转换总会发生。
- `ingest_alpaca.py:64-72`（两个护栏那段）与 `:84-108`（Usage 的六条示例命令）：
  同样按「这条命令还会不会转」逐条判。
- `example/acquisition.md:818-820`（编号 12）：现在读起来像「只有 tick 停在 raw」，
  而新默认下**所有**频率默认都停在 raw，tick 的特殊之处变成「即使给了 `--to-zarr` 也不转」。
- `example/acquisition.md:484`、`:525`、`:650` 的示例命令。
- `tests/test_ingest_shells.py:600-614` 与 `tests/test_volume_guard.py:1328-1345` 的
  docstring：判一下它们对「哪扇门会densify、`--to-zarr` 是谁的独有能力」的陈述是否仍准确
  （分块仍是 `ingest_us_equity.py` 独有，flag 本身不再是）。

**贴出的真实输出一律不伪造**（example/README.md 对整个目录声明的约定，03.4-07 复述过）：
凡是某段贴出的输出里含转换，就把上面那条**命令行**改成能复现该输出的形状（补 `--to-zarr`），
绝不改输出本身；本机没有 Alpaca / Tiingo 凭证，不要为了「重跑一遍」去伪造任何一行。
`example/acquisition.md:502` 那段 traceback 贴的是旧行号，属于历史记录，不动。

把逐句判读的结果（候选句数、判为需要改的句数、每处一行理由）写进 SUMMARY，
让下一次扫除能从这份枚举出发，而不是从某个人的记忆出发。
  </action>
  <verify>
    <automated>uv run pytest tests/test_ingest_shells.py tests/test_volume_guard.py -q && uv run python -c "
import pathlib
readme = pathlib.Path('README.md').read_text(encoding='utf-8')
for name in ('ingest_tiingo.py', 'ingest_alpaca.py'):
    src = pathlib.Path(name).read_text(encoding='utf-8')
    doc = src.split(chr(34)*3)[1]
    assert '--to-zarr' in doc, name + ': module docstring never mentions the flag its Zarr conversion now needs'
    i = readme.index('- \`' + name + '\`')
    j = readme.index(chr(10) + '- \`', i + 1)
    assert '--to-zarr' in readme[i:j], name + ': README bullet still describes conversion without the flag'
print('doc truth gate OK')
"</automated>
  </verify>
  <done>枚举实跑过并逐句判读；两个模块 docstring 与两个 README 条目都把转换描述为 `--to-zarr` 条件性的；example/ 里贴出的输出一行未改、其上的命令行已补齐；判读结果记入 SUMMARY。</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| 厂商 API → ingest 脚本 | 401 / 配额 / 空响应等不可信外部结果穿过这里，本任务改的正是它失败后的处理路径 |
| 环境变量 → 进程 | `TIINGO_API_KEY` / `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`，新增的拒绝消息不得触及 |
| 磁盘 raw 树 → 转换层 | 新增的 raw 根探测跨越这里；探测只读、不写、不删 |
| CLI 参数 → roster 解析 | `--limit` / `--universe` 决定实际取哪些符号；顺序确定性使这个决定可复现 |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-idh-01 | Information Disclosure | `refuse_conversion_without_raw_data` 的拒绝消息 | high | mitigate | 消息只允许出现路径、计数与失败清单入口；禁止拼入 `result.failures` 的厂商响应体或任何环境变量值。本仓库已因硬编码 key 真实泄露过一次，新增的任何输出面都按同一标准处理 |
| T-idh-02 | Denial of Service（自伤） | `assert_dense_panel_fits` 条件化 | high | mitigate | 护栏只在**转换不会发生**时才被跳过；每个densify 调用点之前必须仍有护栏，由 `tests/test_volume_guard.py` 的 AST 行号断言与本计划新增的同形断言双向钉住 |
| T-idh-03 | Tampering | roster 排序改动 | medium | mitigate | 排序只改顺序不改集合；既有 26 条 in_range / as_of 断言（成员、边界、去重）全部保持绿即为证 |
| T-idh-04 | Repudiation | 全失败 run 的可追溯性 | medium | mitigate | 守卫的非零退出不改变失败清单的写入路径；`_failures.json` 仍由 acquisition 层在守卫之前写完，守卫只在其后拒绝转换 |
| T-idh-05 | Elevation of Privilege | 新增 CLI flag 面 | low | accept | `--to-zarr` 是 `store_true`，不接受任何值，不参与路径拼接 |
| T-idh-SC | Tampering | npm/pip/cargo installs | low | accept | 本计划不引入任何新依赖，无 package-manager 安装任务；`pyproject.toml` / `uv.lock` 属于工作树里与本任务无关的既有改动，明确不碰 |
</threat_model>

<verification>
1. `uv run pytest -q` 全绿（本仓库有多个扫描源码的结构性测试，一次 docstring 或注释改动
   不是自动测试中性的）。
2. `uv run python -c "import ingest_alpaca, ingest_tiingo, ingest_us_equity; [print(m.__name__, '--to-zarr' in m._build_arg_parser()._option_string_actions) for m in (ingest_alpaca, ingest_tiingo, ingest_us_equity)]"`
   三个都是 `True`（当前基线：alpaca False / tiingo False / us_equity True）。
3. `git status --porcelain` 里 `cal.py` / `pyproject.toml` / `test.py` / `uv.lock` 仍是
   未暂存的既有改动，未被任何一次 commit 带入。
4. Task 1 的 mutation 结果已记入 SUMMARY，且至少一条断言在 mutation 下转红。
</verification>

<success_criteria>
- 同一组参数两次解析 roster 逐元素相等，且顺序为 symbol 升序；`--limit` 的承诺与之一致。
- 全部符号取数失败的 ingest run 以可读消息 + 非零退出码结束，无未捕获 traceback。
- 全部符号被跳过（watermark 已覆盖）的 run 不受守卫影响，照常转换。
- 三个 ingest 脚本的 Zarr 转换一致地由 `--to-zarr` 控制，默认路径会说明自己跳过了转换。
- 仓库内不再有声称这两个脚本无条件转 Zarr 的句子；example/ 的真实输出一行未被伪造。
</success_criteria>

<output>
Create `.planning/quick/260909-idh-fix-two-uat-gaps-from-phase-03-4-1-g-03-/260909-idh-SUMMARY.md` when done
</output>
