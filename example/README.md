# quantlab 模块说明文档

这里是给**要读懂、要扩展这套代码的人**写的中文说明。每篇都包含可以直接复制运行的例子，
绝大多数例子在写文档时真跑过、贴的是真实输出；跑不了的（需要凭证、需要 GPU、需要别的
机器上的数据）都明确标注了「此例未实际运行」，没有伪造过输出。

## 怎么读

按下面的顺序读，是从「会用」到「能改」的最短路径。

### 第一步：先看数据是怎么进来的

| 文档 | 讲什么 | 什么时候读 |
|---|---|---|
| [acquisition.md](acquisition.md) | 采集引擎：从厂商 API 到磁盘上的原始分片。并发、失败隔离、配额、断点在哪 | 想下载数据、或下载出问题时 |
| [registry.md](registry.md) | 数据源登记表：一个厂商一个描述符（能力、凭证变量名、采集类），程序化 `run()`、进度事件、取消令牌，以及无凭证的只读检视器 | 想知道能下载哪些源、想在程序里发起采集、或没凭证只想看盘上有什么时 |
| [pageledger.md](pageledger.md) | 分页台账：一次多页抓取中途崩了，凭什么能接着跑而不重复不遗漏 | 想搞懂断点续跑，或看到 `_pages/` 里的文件时 |
| [constituent.md](constituent.md) | 时点成分与标的池：怎么避免幸存者偏差，四个 category 分别是什么 | 要选标的池、要做回测时 |

### 第二步：数据是怎么变成面板的

| 文档 | 讲什么 | 什么时候读 |
|---|---|---|
| [dataset.md](dataset.md) | 数据集层：原始文件 → 规范的 `[timestamp, symbol]` xarray 面板 | 想理解全流水线的数据形态时 |
| [chunking.md](chunking.md) | 时间分块：为什么不能一次性densify 全区间，以及分块和追加怎么衔接 | 数据量大到内存放不下时 |
| [backend.md](backend.md) | 存储后端：把「存在哪里」和「数据是什么」分开 | 想换存储介质、或看到 append 报错时 |

### 第三步：算因子、训模型

| 文档 | 讲什么 | 什么时候读 |
|---|---|---|
| [factor.md](factor.md) | 因子层：KunQuant 与 Polars 两个后端，各自适合什么 | 要写新因子时 |
| [model.md](model.md) | 模型层：基类替你做了什么，子类要实现哪五个方法 | 要接新模型时 |

### 第四步：回测

| 文档 | 讲什么 | 什么时候读 |
|---|---|---|
| [backtest.md](backtest.md) | 回测层：`run()` / `run_cv()` 的模板步骤、目标权重契约、t+1 开盘成交、退市强平、截面 TopN 选股、样本内外分开报告、运行目录与数据指纹、从 `config.json` 重建重跑；附一个离线真跑过的最小例子 | 训完模型想看它能不能交易、要回放一次 `train_cv`、或要写新的回测引擎/市场/选股规则时 |

## 想直接上手扩展

这五篇各自带一个**从零写到跑通的最小扩展**（新数据源是注册一个描述符，其余四篇是一个最小子类），是最快的入门方式：

- 新增一个数据源 → [registry.md](registry.md) 的「扩展」一节（注册一个描述符）
- 新增一个市场 → [dataset.md](dataset.md) 的「扩展」一节（`MiniCsvDataset`）
- 新增一种存储介质 → [backend.md](backend.md) 的「扩展」一节（`CsvBackend`）
- 写一个新因子 → [factor.md](factor.md) 的「扩展一 / 扩展二」（`RelativeVolume` 走 Polars、`MaDeviation` 走 KunQuant）
- 接一个新模型 → [model.md](model.md) 的「扩展」一节（`TinyRegressor`）

## 写文档时发现的缺陷

这些是核实文档内容时撞见的真实问题，不是推测。按严重程度排：

| 位置 | 问题 | 详见 |
|---|---|---|
| ~~`quantlab/base/model.py`~~ | ~~`early_stopping=False` 会 `UnboundLocalError` 直接崩；早停计数器按**验证 batch** 递增而非 epoch；张量列序是**字母序**不是你传入的顺序，主目标可能不是你以为的那个~~ **已于 2026-09-07 修复**（`tests/test_model_layer.py`） | [model.md](model.md) 「常见坑」 |
| ~~`quantlab/dl_model/`~~ | ~~`MLPRegressor` 三处坏掉无法实例化；`rnn.py` 里的 `RNNClassifier` 是过期坏副本（活的那个在 `rnn_classification.py`）；`update()` 读了 `DLConfig` 没有的字段~~ **已于 2026-09-07 修复/删除**（`tests/test_dl_models.py`） | [model.md](model.md) |
| ~~`quantlab/base/model.py:num_null`~~ | ~~结尾 `.values[0]` 索引一个 0 维数组，**每次读取都 `IndexError`**~~ **已于 2026-09-07 修复**。一条被文档推荐、注解写着 `-> int`、却从来没跑通过的属性（`tests/test_model_layer.py`） | [model.md](model.md) |
| ~~回测骨架~~ | ~~`_do_vecbt` / `_vecbt` / `RNNClassifier._vecbt` / `DLModel._fit(backtest=...)`（2026-09-14 前叫 `_train_dl`）四块半成品互不相连，全部**安静地什么都不做**。**保留**（端到端回测归 Phase 6，钩子位置是对的），但 2026-09-07 起改为显式 `NotImplementedError` 点名 Phase 6——空实现要么报错，要么就不该存在~~ **已删除**，连同锁它们的测试：`_do_vecbt` / `_vecbt` 于 2026-09-14（`d07f06e`），`RNNClassifier._vecbt`、`DLModel._fit(backtest=...)` 与 `backtest_data` 于阶段 03.7（D-37，`tests/test_model_hierarchy.py::test_stale_backtest_hooks_are_deleted` 锁住不再回来）。回测现在只在 `quantlab/backtest/` | [model.md](model.md)、[backtest.md](backtest.md) |
| 骗人的命名 | ~~`_train_one_epoch` / `_val_one_epoch` / `_test_one_epoch` 其实是 per-**batch**；`_get_features_batch` / `_get_labels_batch` 里的 `batch` 又是相反的意思（收齐全部）~~ **已于 2026-09-07 改名**为 `_*_one_batch` / `_collect_all_*`。前者的名字实际造成过一个早停缺陷 | [model.md](model.md) |
| ~~`quantlab/utils/nautilus.py`~~ | ~~`get_crypot_currency` 拼错了（"crypot"），且有一个被接收又完全忽略的 `name` 参数~~ **已于 2026-09-07 更正并删参**（`tests/test_spot_dataset.py`） | — |
| ~~`quantlab/dataset/backend.py`~~ | ~~`XrBackend.get_xarray_dataset()` **完全忽略** `indexes` 参数，连带 `BaseDataset.time_interval` 在该后端下不可用~~ **已于 2026-09-07 修复**（`tests/test_backend_indexes.py`） | [backend.md](backend.md)、[dataset.md](dataset.md) |
| 成分 vs 价格 | 跨改名的代码词表对不上：876 个 sp500 成分符号里 89 个在价格 roster 查无此符号 | [constituent.md](constituent.md) 「已知的坑」 |
| `quantlab/base/factor.py` | `save()` 默认 `mode="a"`，但它**不是时间追加**，第二段日期会直接报错。默认值有意保留；2026-09-07 起报错信息会直接点名 `mode="w"`（`tests/test_factor_save_mode.py`） | [factor.md](factor.md) |
| 死代码 | ~~`WindowedRobustStandardization`、`PageLedger.last_position()`~~ **已于 2026-09-07 删除**（均零调用点，从未被执行过）。`MlBackend` 当时保留，现在是 `MLModel` 的 checkpoint 持久化后端（2026-09-14 起 `XGBoostRegressor` 经它读写 `.joblib`） | 各篇 |

## 关于例子

例子里用到的临时目录一律在 `/tmp` 下，不会污染仓库。需要真实行情的例子会说明数据从哪来。
凭证一律只出现变量名（`TIINGO_API_KEY`、`APCA_API_KEY_ID`、`APCA_API_SECRET_KEY`），
文档里不会有任何真实密钥。
