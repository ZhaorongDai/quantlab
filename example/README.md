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

## 想直接上手扩展

这四篇各自带一个**从零写到跑通的最小子类**，是最快的入门方式：

- 新增一个市场 → [dataset.md](dataset.md) 的「扩展」一节（`MiniCsvDataset`）
- 新增一种存储介质 → [backend.md](backend.md) 的「扩展」一节（`CsvBackend`）
- 写一个新因子 → [factor.md](factor.md) 的「扩展一 / 扩展二」（`RelativeVolume` 走 Polars、`MaDeviation` 走 KunQuant）
- 接一个新模型 → [model.md](model.md) 的「扩展」一节（`TinyRegressor`）

## 写文档时发现的缺陷

这些是核实文档内容时撞见的真实问题，不是推测。按严重程度排：

| 位置 | 问题 | 详见 |
|---|---|---|
| `base/model.py` | `early_stopping=False` 会 `UnboundLocalError` 直接崩；早停计数器按**验证 batch** 递增而非 epoch；张量列序是**字母序**不是你传入的顺序，主目标可能不是你以为的那个 | [model.md](model.md) 「已知的不完整之处」 |
| `dl_model/` | `MLPRegressor` 三处坏掉无法实例化；`rnn.py` 里的 `RNNClassifier` 是过期坏副本（活的那个在 `rnn_classification.py`）；`update()` 读了 `DLConfig` 没有的字段 | [model.md](model.md) |
| ~~`dataset/backend.py`~~ | ~~`XrBackend.get_xarray_dataset()` **完全忽略** `indexes` 参数，连带 `BaseDataset.time_interval` 在该后端下不可用~~ **已于 2026-09-07 修复**（`tests/test_backend_indexes.py`） | [backend.md](backend.md)、[dataset.md](dataset.md) |
| 成分 vs 价格 | 跨改名的代码词表对不上：876 个 sp500 成分符号里 89 个在价格 roster 查无此符号 | [constituent.md](constituent.md) 「已知的坑」 |
| `base/factor.py` | `save()` 默认 `mode="a"`，但它**不是时间追加**，第二段日期会直接报错 | [factor.md](factor.md) |
| 死代码 | `MlBackend`、`WindowedRobustStandardization`、`PageLedger.last_position()` 均零调用点 | 各篇 |

## 关于例子

例子里用到的临时目录一律在 `/tmp` 下，不会污染仓库。需要真实行情的例子会说明数据从哪来。
凭证一律只出现变量名（`TIINGO_API_KEY`、`APCA_API_KEY_ID`、`APCA_API_SECRET_KEY`），
文档里不会有任何真实密钥。
