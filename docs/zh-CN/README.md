# quantlab 文档

[English](../README.md) | 简体中文

这是 quantlab 的用户指南。每一篇介绍流水线的一个部分，并通过可运行的例子讲解。可以直接读你需要的那个阶段，
也可以按顺序读：指南的顺序就是数据的流向，从数据源一直到回测报告。

凡是打印了输出的代码示例，输出都来自实际运行。需要厂商账号的示例会说明所需条件，并且不会打印结果。

## 获取数据

| 指南 | 内容 |
|------|------|
| [采集引擎](acquisition.md) | 一次下载如何运行：分批、失败隔离、增量刷新、原始文件布局和体量护栏 |
| [数据源登记表](registry.md) | 数据源目录、`run()` 与 `convert()` 入口、进度事件和只读检视器 |
| [断点续传](pageledger.md) | 多页下载中断后如何接着运行 |
| [WRDS CRSP 日频股票](wrds_crsp.md) | 按 PERMNO 组织的研究级美股日频数据、总收益复权和退市收益 |
| [WRDS TAQ 报价](wrds_taq.md) | 最优买卖报价（NBBO）及其重采样为 bar |

## 股票池

| 指南 | 内容 |
|------|------|
| [指数成分](constituent.md) | 标普 500 与纳斯达克 100 的时点成分面板 |
| [股票池过滤](universe.md) | 以因子包装类实现的价格与流动性过滤 |

## 构建面板

| 指南 | 内容 |
|------|------|
| [数据集](dataset.md) | 从原始文件到 `(timestamp, symbol)` 面板，以及如何新增一个市场 |
| [分块转换](chunking.md) | 按时间窗口逐个转换很长的日期范围 |
| [存储后端](backend.md) | Zarr 与 Parquet 存储、追加写入，以及如何编写新的后端 |

## 研究

| 指南 | 内容 |
|------|------|
| [因子](factor.md) | KunQuant 与 Polars 两个因子后端、标签和标准化 |
| [模型](model.md) | 模型层级、训练、交叉验证和检查点 |
| [回测](backtest.md) | 目标权重、模拟、指标和运行目录 |

## 包结构

```text
quantlab/
  base/         抽象契约：数据集、因子、模型、回测器、采集
  acquisition/  Tiingo、Alpaca、WRDS 下载器
  dataset/      具体数据集：现货 K 线、美股、CRSP、NBBO、指数成分
  factor/       因子集合：Alpha101、Alpha158、动量、股票池过滤
  label/        未来收益标签
  dl_model/     PyTorch 模型头：MLP、GRU、LSTM
  ml_model/     XGBoost 模型头，以及非 torch 模型的检查点存储
  backtest/     vectorbt 引擎、TopN 选股、美股回测器
  backend.py    Zarr 与 Parquet 存储后端
  registry.py   数据源登记表，以及 run() 与 convert() 入口
  universe.py   时点股票池
  config/       配置工厂与随包发布的标的元数据
  utils/        命令行辅助、指标、序列化、报告生成
scripts/wrds/   WRDS 下载脚本：index.py、market.py、etf.py、nbbo.py
tests/          测试套件
```

每个公开的类和函数也都有 numpydoc 格式的 docstring，其中包含 `Examples` 小节。可以用 `help()` 查看，
例如 `help(quantlab.base.model.DLModel)`。
