# CRSP symbol 轴从 ticker 迁移到 PERMNO — 调研结论

调研日期：2026-09-20。两个只读调研 agent 的合并结论，行号对应当时的 HEAD
（`d91bd30`，03.10 gap set 全部合并后）。

## 已定决策（用户 2026-09-20）

| 决策 | 选择 | 理由 |
|---|---|---|
| symbol 轴类型 | **int64**（不是 digit-string） | PERMNO 本质是整数；彻底消除排序陷阱。项目未进生产，约 40 处 `str(symbol)` 强转是一次性成本 |
| 其他数据源 | **不管**。本次只做 CRSP | 不做 (ticker,date)→PERMNO 映射层；Tiingo/Alpaca/Binance 保持 ticker 轴 |
| ticker 是否进面板 | **不进**。留在 sidecar 做旁路查找 | 见下「为什么 ticker 不能进面板」 |
| 向后兼容 | **不做**。已落盘 store 直接重建 | 项目未进生产（见 memory `project-no-backward-compat`） |
| WR-05（03.10 遗留） | 由本次迁移一次性消灭 | `resolve_collisions` 整体删除，不先加计数再删 |

## 迁移的本质：删掉中间一层，不是新建管线

CRSP 的 **raw tier 已经是 PERMNO 键**：`quantlab/acquisition/wrds_crsp.py:317-320` —— raw
`symbol` 列就是 PERMNO 的字符串形式，另有 typed `permno` Int64 列随行，且 `"14593"` 恰好满足
`TRADEABLE_TICKER_PATTERN`（`quantlab/enums/data.py:235-237`），所以采集层的路径守卫和
watermark 文件名（`quantlab/base/coverage.py:245-246`）**已经在跑 PERMNO**。

ticker 化只发生在两行：

```
quantlab/dataset/crsp.py:494   frame = frame.drop("symbol")        # ← PERMNO 轴在这里被丢弃
quantlab/dataset/crsp.py:495   frame = self._symbology.label_rows(frame)  # ← ticker 在这里成为 symbol
```

成分股侧的对应转换点：`quantlab/dataset/crsp_membership.py:288`
（`pieces.append((str(symbol), piece_start, piece_end))`）。

## int64 轴会破坏的硬点（必须修，按严重度排序）

### A. `XrBackend.widen_symbol_axis` — 静默清空整个 store（最危险）

`quantlab/dataset/backend.py:451` → `795` / `869`

```python
requested = [str(symbol) for symbol in symbols]                      # 451
widened = stored.reindex({dim: requested}, fill_value=fills).load()  # 795  mode="w"
block = block.reindex({dim: requested}, fill_value=fills)            # 869
```

`stored[dim]` 是 int64、`requested` 是 str 列表 → `reindex` 认为所有请求标签都是新的，把 store
每一列丢掉换成全 NaN。superset 守卫（`455-468`）两侧都 `str()` 过所以**不会报警**，紧接着
`os.replace(widening, target)`（`backend.py:543-545`）把空库变成权威库，原库被 `rmtree`。
**无异常、无日志。** 这是真 bug，不是兼容问题。

### B. `DLModel._align_prediction_symbols` — KeyError，位置编码模型不可用

`quantlab/base/model.py:1220`：`return feats.sel(symbol=sorted(trained))`。
`trained` 来自 `_read_trained_symbols`（`model.py:507-514`），**永远是 `list[str]`**（从 JSON 读回）。
前面的成员校验（`:1201-1203`）两侧都 `str()` 所以会通过，错误发生在最后一行，报错信息完全误导。
另有排序隐患：`sorted(trained)` 是字典序，而 `to_array` 的 `sortby("symbol")` 在 int 轴上是数值序。

### C. 稠密化 / 窗口 reindex 链 — 每个窗口全 NaN

- `quantlab/base/data.py:326`（钉轴时 `str()`）→ `:340` reindex
- `quantlab/dataset/stock.py:506-516` → `:549`
- `quantlab/dataset/crsp.py:1372-1377` → `:1441-1450`
- `quantlab/base/data.py:824-866`（`_added_symbols_with_raw_history`）

`quantlab/dataset/stock.py:35-42` 的注释已经点名这个坑（03.2-RESEARCH Pitfall 8：
未钉 dtype 的数字型 hive 值被推断成 Int64，之后字符串比较静默匹配不到任何东西）。

### D. polars 侧 `is_in([str,...])` — 类型不匹配或零匹配

`dataset/backend.py:1611-1613`、`dataset/stock.py:531`、`dataset/crsp.py:1442-1450`、`crsp.py:481-483`。
CRSP 那几处尤其要注意：`config.permnos` 是 digit-string 而 `permno` 列是 Int64，代码靠显式
`pl.col("permno").cast(pl.String).is_in(...)`（`crsp.py:866-868`、`crsp_symbology.py:216`、`:267`）
桥接；面板 symbol 变 int 后这些不对称 cast 会打结。

### E. 测试的 symbol 坐标编码模型只认字符串

`tests/conftest.py:1460-1536` 的 `SYMBOL_COORD_ENCODINGS = ("fixed_width","variable_length")`；
`stored_symbol_encoding` 对 int64 dtype 直接 `raise AssertionError`（`:1532-1536`）。
5 个文件参数化在这两个 arm 上：`test_symbol_axis_widening.py`、`test_variable_axis_widening.py`、
`test_factor_update.py`、`test_widening_fixture_realism.py`、`test_symbol_coord_encoding.py`。
需要加第三个 arm 或重新设计。

**int64 的附带好处**：`dataset/backend.py:1020-1044` 记录的
`ValueError: Mismatched dtypes for variable symbol`（zarr `object` vs `StringDType()` 往返）
是字符串坐标特有的，int64 坐标无歧义，这个坑消失，`widen_data_vars` 里「filler 故意不带 coords」
的绕法不再必要。

## 与轴类型无关、但迁移必须显式处理的语义破坏

### F. `universe_filter` 的 9 条 ticker 正则会静默失效

`quantlab/factor/universe_filter.py:201-222` 的 `NON_COMMON_TICKER_PATTERNS` 全部对 ticker
文本做 `re.search`（`^[A-Z]{4}[WRU]$`、`^[A-Z]{4}WS$`、`[-.](?:WS|WT|W|U|UN|R|RT)(?:[-.]|$)`、
`^Z[A-Z]ZZT$`、`^[A-Z]TEST(?:-|$)`、优先股 `[-/]\s*P[A-Z]?\s*(?:[-/]|$)`、baby bond `\s\d` …）。
对 PERMNO 全部不匹配 → `is_common_ticker`（`:356-361`）恒为 True → LS-1 的非普通股排除
**整条静默消失**（文档记录实测剔除 24.3% / 3,519 个代码）。调用点：`:410-418`、`:546`、`:610-614`。

**处理方式**：CRSP 的 `security_filter`（`crsp.py:200-215` 的 `equity_common` 预设）用 per-date 的
`sharetype`/`securitytype`/`securitysubtype` 做同一件事且更权威。应在 CRSP 面板上**显式关掉**这条
并在报告里说明，而不是让它变成静默 no-op。

### G. constituent / masking 轴不匹配（会大声失败，好事）

`dataset/constituent.py:130-136` 的不变量是「constituent 面板的 ticker 轴 == CRSP 价格面板的
ticker 轴，因为两边都出自同一条 `CrspSymbology`」。价格面板换 PERMNO 后
`dataset/masking.py:101-105` 的交集变空 → `:181-191` 抛
`ValueError: the two panels do not overlap`。这是四个跨面板接点里**唯一设计对了的**（会报错而非静默）。

**修法明确、成本低**：`CrspSP500ConstituentDataset._build_intervals`（`dataset/constituent.py:155-161`）
和 `CompustatNasdaq100ConstituentDataset._build_intervals`（`:202-208`）从 `symbol_intervals(...)`
换成**已经存在的** `permno_intervals(...)`（`crsp_membership.py:185-208`），并把 `permno` 列改名成
`symbol`。注意 `base/constituent.py:198` 的 `sorted({str(...)})` 也要改。

### H. 「有没有 ticker」目前隐式充当准入过滤器 ← 最容易漏

`crsp_symbology.label_rows`（`:409-425`）会**丢弃**所有找不到 ticker 区间覆盖的行。实测
`stksecurityinfohist` 全历史 191,048 个区间行里：

| | 数量 |
|---|---|
| `ticker` 为 NULL 的区间行 | 34,839（18.2%） |
| 涉及的 PERMNO | 30,197 / 40,518（74.5%） |
| NULL-ticker 区间中位长度 | 1 天（多为退市当天，正是 symbology carry 规则在救的行） |
| 最长 | 20,467 天 |
| **从未有过任何 ticker 的 PERMNO** | **1,012（2.5%）** |

那 1,012 个 PERMNO 在 ticker 轴下**永远进不了面板**；PERMNO 轴下会首次出现。

**这不是删死代码，是认出被删代码承担的隐式职责。** PERMNO 轴需要一个**显式准入条件**替代它
（按 `securitytype`/`sharetype`/`securityactiveflg`），并且应该并进已有的 `security_filter` 机制，
不要新建一套。

### I. 跨 vendor 合并会静默出全 NaN（用户已决定不管，仅记录）

- `base/model.py:207`/`228`/`237`：`xr.combine_by_coords`。ticker 轴面板 + PERMNO 轴面板 →
  外连接出 `N_ticker + N_permno` 长的轴，交集为空，全格 NaN，不抛错。
  `_assert_shape_match_x`（`model.py:899-905`）只比数量，也过得去。
- `base/backtest.py:756-757`：`predictions.reindex(symbol=prices.symbol.values)`。因子在 ticker 轴、
  价格在 PERMNO 轴 → 预测全 NaN → `selection.py` 判定无可选标的 → **空仓回测**，只有
  `selection.py:163` 一条 warning。

用户决定本轮不管其他数据源。**但「不混用」和「混用时不报错地算出错结果」是两件事**——最小防护是在
这两个合并点加一条 dtype/交集断言（几行），把静默错误变成大声拒绝。建议作为一个可选小任务。

## 排序陷阱（int64 选择消除了它，记录以免回退）

全仓 10+ 处按 symbol 排序，只有 `crsp_membership.py:326-345` 一处显式处理了数值序，其 docstring
写得很清楚：

> **顺序是契约的一部分，而且是数值序。** PERMNO 是渲染成字符串的整数，所以对文本 `sorted()`
> 会把 `"14593"` 排在 `"7000"` 之前。`resolve_symbols` 会对这个列表切 `--limit`；不稳定的顺序会让
> 每次运行截断到不同批次，而第二次运行永远碰不上第一次写下的 watermark。

其余 9 处是纯字典序：`stock.py:506-511`、`crsp.py:1372-1375`、`base/constituent.py:198`、
`masking.py:105`、`nbbo.py:173/185`、`model.py:310`、**`model.py:1220`**、`fingerprint.py:53`、
`chunking.py:203-211`。

历史 PERMNO 全是 5 位数（约 10000–93436），所以字典序恰好等于数值序 —— **但这是巧合，代码里没有
任何保证**（`scripts/ingest_wrds_crsp.py:296-304`、`base/config.py:377-380` 只查 `isdigit()`）。
int64 轴让这个问题从根上消失。

## 为什么 ticker 不能进面板

- 二维 `ticker(timestamp, symbol)` 字符串变量撞三道现有守卫：
  `dataset/backend.py:470-505`（`widen_symbol_axis` 拒绝带 symbol 维、非浮点、且未在 `fill_values`
  点名的变量；`CrspStockDataset` 没覆写 `_widen_fill_values`，继承 `{"anomaly_flag": False}`）、
  `backend.py:984-1005`（`widen_data_vars` 同样守卫）、`backend.py:1020-1044`（字符串 dtype 往返）。
  且 `crsp.py:1220-1229` 把所有数据变量一律 `cast(pl.Float64)`。
- 一维 `ticker(symbol)` 坐标只能表达「最后一个 ticker」，丢掉 FB→META 这类改名史
  （`crsp_symbology.py:5` 明确说 PERMNO 13407 全程不变而 ticker 变了）。

**结论**：ticker 走 sidecar（`CrspSymbology.symbol_intervals()` /
`{zarr}.crsp_symbology_report.json`），展示层按需做 as-of 查询。与项目既有 sidecar 模式一致。

## 人类可见 symbol 的完整清单（需要 ticker 还原的全部位置）

逐个核查过，只有这 6 处 + 2 个 sidecar：

1. `backtest/engine_vectorbt.py:297` 强平日志 `forced liquidation of {symbol}`
2. `liquidations.json` 的 `symbol` 字段（`base/backtest.py:1798-1804` + `engine_vectorbt.py:291`）
3. `base/model.py:1207`/`1216` missing/extra symbols 列表
4. `dataset/masking.py:148-163` `missing_symbols` 完整列表（明确声明永不截断、永不采样）
5. `acquisition/inspector.py:485-510` `browse_zarr`（运维工具）
6. `utils/cli.py:129-133` `--symbols` help 文案
7. sidecar：`{zarr}.crsp_filter_report.json`（`crsp.py:1095-1126` 的 `_permno_breakdown`
   **已经是 `{PERMNO: {symbol,...}}` 结构，两边都带**）、`{zarr}.crsp_symbology_report.json`

**不需要还原**：`report.html`（`utils/backtest_report.py` 全文只有 2 处 `symbol`，都是 plotly 的
`marker={"symbol":"triangle-up"}`，**一个标的名都不打**）、`metrics.json`、`weights.zarr`、
`equity.zarr`、fingerprint。`selection.py:163` 和 CRSP 各类日志只报数量不报名字。

## 完全不透明的层（symbol 换成什么都零成本）

- **KunQuant**：`base/factor.py:283-306` — 只吃 `input_dict`（float32 `[time,symbol]` 连续数组），
  symbols 只作为 opaque ndarray 传回，**从不看值**。唯一约束是轴长为 SIMD block width（8）的倍数
  （`tests/test_factor_kunquant.py:124-127`，否则 `kr.runGraph` 抛 `RuntimeError: Bad shape at open`）。
- **张量转换**：`base/model.py:266-282` `to_array` — symbol 的值从不进数组，只有 `sortby` 的位置。
- **所有因子/标签定义**：`factor/alpha101.py`、`alpha158.py`、`momentum.py`、`label/fret.py` —— 只用
  列名，polars 路径用 `.over("symbol")` 分组。
- **selection**：`backtest/selection.py:131-177` 全走 `.values` + 位置索引。
- **cleaning**：`dataset/cleaning.py` 只校验 `dims == ("timestamp","symbol")`，从不看值。
- **`get_xarray_dataset(["timestamp","symbol"])`**：`backend.py:1489-1545` 只校验维度名。
- **Nautilus**：`dataset/stock.py:573-583` 的美股 Nautilus 路径全是 `raise ValueError("Not finished")`，
  所以 PERMNO 对 Nautilus **零影响**。`utils/nautilus.py` 的 `parse_symbol_currencies` 要求
  `endswith("USDT")`，是 crypto 专用。

## 死代码清单（PERMNO 轴下可删）

`crsp_symbology.py` 590 行中约 88% 变成死代码。

| 位置 | 内容 | 命运 |
|---|---|---|
| `crsp_symbology.py:241-321` | `_class_collision_pass`（class 后缀重拼 BRK.A/BRK.B） | 删 |
| `crsp_symbology.py:455-590` | `resolve_collisions` + `_collision_message` + `_MAX_LISTED_COLLISIONS` | 删 |
| `crsp_symbology.py:90-110` | `_member_spans` | 删 |
| `crsp_symbology.py:382-404` | `label_rows` 的 delisting **symbol** carry | 删（⚠ 见坑 1） |
| `crsp_symbology.py:202-222` | ticker forward-fill carry + overrides 注入 | 删 |
| `crsp_symbology.py:56-77` | `SUFFIX_DELIMITER`/`_NO_CLASS`/`_OPEN_END`/`_DELISTING_FLAG` | 删 |
| `crsp.py:660-746` | `_resolve_identity` 全体（含 PERMNO seam 打 NaN） | 删 |
| `crsp.py:748-750` | `symbology_report_path()` | 删 |
| `crsp.py:1423-1429` | `_write_identity_reports` 的 symbology 分支 | 删 |
| `crsp.py:222-225` | `SYMBOLOGY_REPORT_SUFFIX` | 删 |
| `crsp_membership.py:210-308` | `symbol_intervals()` | 删 |
| `crsp_membership.py:627-646` | `_symbol_frame()` | 删 |
| `crsp_membership.py:59-65` | `_Key = int \| str` 的 str 分支 | 收窄为 int |
| `base/config.py:133-137` | `symbol_overrides` | 删 |
| `base/config.py:156-161` | `nan_adj_at_permno_seam` | 删 |
| `base/config.py:243` | `qqq_benchmark()` 里的 `symbol_overrides={QQQ_PERMNO:"QQQ"}` | 删该项，保留 `permnos` + `security_filter` |
| symbology report 6 个字段 | `collisions`/`class_suffixed`/`seams`/`delisting_carried`/`unlabelled`/`nonconforming_symbols` | 全删 |
| `crsp.py:135` | `permno` 作为数据变量（Float64） | 删（与坐标重复；`permco` 保留） |

## 必须保留的（不要误删）— 六个坑

1. **两个 "delisting carry"，只有一个能删。**
   `crsp_symbology.py:382-404`（给退市行**补 ticker 标签**）→ 删。
   `crsp.py:960-976` +`crsp.py:228` 的 `_DELISTING_FLAG`（给退市行**继承 security filter 前一日判决**，
   因为 type 列在退市行会变空）→ **必须保留**，这是 D-10 反生存者偏差。两者在
   `crsp.py:930-934` 的 docstring 里并排提到，极易误删第二个。

2. **`collision_universe` 一名两责。**
   职责 1（D-04，collision tie-break）：消费点 `crsp.py:690-692`、`crsp_symbology.py:455-534`、`:585-589`
   → PERMNO 轴下死透，删。
   职责 2（03.10-14 / GAP-C 之后，「显式 roster 覆盖 security_filter」）：消费点 `crsp.py:801-813`、
   `:872-874`、`:884-919`、`:754-781` → **完全保留**。
   → **字段不能删，但该改名**（`config.py:180-189` 自己写了这个 trade-off）。`crsp.py:754-781` 的
   「现在有两个读者」docstring 必须重写。

3. **删 `unlabelled` 会让面板变宽** — 见上文 H，是行为变化，不是纯删代码。

4. **`_assert_unique_panel_keys`（`crsp.py:1487-1521`）不是纯适配代码。** 它防的是基类
   `dedup_raw_frame(keep="last")` 静默合并。PERMNO 轴下这个风险由
   `wrds_crsp.py:818-840` 的 `(permno, dlycaldt)` 断言在上游覆盖，但留一个便宜的下游 backstop 合理。
   只需改文案（`:1520` 提到 `symbol_overrides`）。

5. **`config.symbols` 与 `config.permnos` 语义合流。** `crsp.py:1367-1371` 用 `config.symbols` 做
   ticker 侧过滤、`config.permnos` 做 raw 侧过滤，注释反复强调二者之分（`crsp.py:159-160`、
   `config.py:112-113`）。PERMNO 轴下两者过滤同一个东西 → 需决定合成一个。

6. **`wrds_crsp.py:317-319` 看着像 ticker 适配，其实是反向的。** 它说的是「PERMNO 数字串恰好满足
   `TRADEABLE_TICKER_PATTERN`，所以基类路径守卫原样可用」。**别动。**

## 连带影响

- **ChunkLedger fingerprint**（`base/chunking.py:203-211`）顺序敏感 → 轴一换指纹必变，现有 ledger
  全部作废，store 需重建（项目未进生产，可接受）。
- **backtest fingerprint**（`utils/fingerprint.py:53`、`:58`）同理 → `_compare_fingerprints`
  （`backtest.py:1145-1188`）会对每个已存 run 发 warning（只 warn，不中断）。
- **checkpoint 的 `trained_on.symbols`**（`model.py:310`、`507-514`）是 `list[str]` → 见破坏点 B。

## 测试影响量级

- `tests/` 下硬编码 ticker 字面量约 **832 处 / 39 个文件**；提到 `symbol` 的文件 84 个、2,625 次。
- CRSP 相关（必改）：`test_crsp_dataset.py` 21、`crsp_fixtures.py` 17、`test_crsp_identity.py` 14、
  `test_crsp_constituent.py` 9、`test_crsp_symbology.py` 6、`test_crsp_tracer.py` 2。
- **基本整文件删除**：`tests/test_crsp_symbology.py`（21 个测试，大半是 class share / collision /
  carry / 正则）。
- **需要重新设计而非替换**：上文 E 的 5 个 `symbol_coord` 参数化文件、
  `tests/test_ticker_pattern_reconciliation.py`、`tests/test_universe_filtered_factor.py`（正则语义）、
  `tests/test_universe_mask.py` + `test_crsp_constituent.py`（轴换 PERMNO）。
- 很多回测/模型测试用合成短名（`"A"`/`"B"`、`"S1"`/`"S2"`），改成 PERMNO 是纯机械替换。

## 文档需同步

`example/wrds_crsp.md`：`:103`（resolve_collisions 三规则）、`:228`（seam）、`:235`
（`--universe`/`collision_universe`）、`:364`+`:457`（symbology sidecar）、`:405`
（`symbol_overrides` 钉 QQQ）。

**⚠ ROADMAP 的 03.10 locked decision 需要显式推翻**：现写着「the panel `symbol` dimension stays
the **ticker** valid at each date；**PERMNO is kept as a data variable**」。本 phase 反转它，必须
作为显式的决策变更记录，不能悄悄改。

## 仍待确认（规划时决定）

1. `config.symbols` 在 CRSP 上的去留（坑 5）—— 是否保留「按 ticker 选股」这个入口。
2. `collision_universe` 是否改名（坑 2）。项目未进生产，改名成本已消失。
3. `UniverseFilteredFactor.exclude_non_common` 在 CRSP 面板上关掉，需要一次与
   `universe_filter.py:158-200` 那段测量（对 `us_all` 14,481 个代码测出 24.3% 命中率、两条证伪、
   逐个人工复核）对等的论证。
4. vectorbt 对整数列索引的行为未实跑验证（`engine_vectorbt.py:128-142` 用 `group_by=True` +
   `cash_sharing=True` + `call_seq="auto"`）。理论自洽（`records_readable["Column"]` 经
   `wrapper.columns` 反查后 `.astype(str)`），但应在 tracer 里实测。
5. 坑 3（面板变宽）的准入条件具体用哪些 CRSP 列。
