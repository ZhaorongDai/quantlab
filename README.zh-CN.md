# quantlab

[English](README.md) | 简体中文

quantlab 是一个用于量化股票研究的 Python 后端。它用五个步骤把原始行情数据变成一个经过回测的交易策略：
下载价格数据，整理成干净的面板，计算因子和标签，训练预测未来收益的模型，再对这些预测所对应的投资组合做回测。
每一步都由一个小的配置对象驱动，所以任何一次运行都可以保存、重建并完全复现。

- **文档：** [docs/README.md](docs/README.md)（英文），[docs/zh-CN/README.md](docs/zh-CN/README.md)（中文）
- **示例：** [examples/](examples/README.md)
- **源代码：** https://github.com/ZhaorongDai/quantlab2
- **问题反馈：** https://github.com/ZhaorongDai/quantlab2/issues
- **参与贡献：** [CONTRIBUTING.md](CONTRIBUTING.md)

```text
 数据源     ->  数据集      ->  因子与标签    ->  模型       ->  回测
 Tiingo,        原始文件         KunQuant 或       XGBoost,       目标权重,
 Alpaca,        转成面板         Polars            PyTorch,       vectorbt,
 WRDS                                              pytabkit       HTML 报告
```

## 它能做什么

各个步骤之间只用一种数据格式交换数据：一个 `xarray.Dataset`，其中每个变量都排列在 `timestamp`（时间）
和 `symbol`（标的）两个维度上。我们把这样的数据集称为*面板*。面板以 Zarr 格式存盘，模型直接在面板上训练，
步骤之间不需要来回转换成 DataFrame。

数据来自 Tiingo、Alpaca 和 WRDS（CRSP 日频股票数据和 TAQ 报价数据）。下载可以中断后继续，
而且在请求发出之前会先检查数据量，过大的请求会被直接拒绝。为了避免*幸存者偏差*（只用今天仍然存在的公司做测试所导致的偏差），
quantlab 可以根据历史上的指数成分、以及包含已退市股票的全市场名单来构建股票池。

因子可以用 [KunQuant](https://github.com/Menooker/KunQuant) 计算，它把因子公式编译成本地代码，
既能对整段历史批量计算，也能逐根 K 线流式计算；也可以用 Polars 做快速的批量实验。树模型和神经网络共用同一套接口，
并内置滚动（walk-forward）交叉验证。回测基于 [vectorbt](https://vectorbt.dev/)，样本内和样本外的结果分开报告，
每次回测都会写出一个运行目录，之后可以据此重建并重新运行。

quantlab 是一个研究后端，没有网页前端，也不会向券商发送订单。

## 安装

quantlab 需要 Python 3.13 或更高版本、[uv](https://docs.astral.sh/uv/) 以及一个 C++ 编译器
（KunQuant 在运行时编译因子代码）。

```bash
git clone https://github.com/ZhaorongDai/quantlab2.git
cd quantlab2
uv sync
```

神经网络模型在有 CUDA GPU 时使用 GPU，否则使用 CPU。GPU 和 macOS 的注意事项见
[安装指南](docs/getting-started/installation.md)。

## 快速上手

了解完整流程最快的方法是运行端到端示例。它会生成一个合成的价格面板，计算因子，训练一个 XGBoost 模型，
做回测，并根据保存的配置重建这次运行。它不需要联网，也不需要任何凭证，在笔记本电脑上大约半分钟跑完。

```bash
uv run python examples/quickstart.py
```

[快速上手指南](docs/getting-started/quickstart.md)（英文）会一步一步讲解这个示例。
英文版 [README](README.md) 中还有一段更短的代码，演示数据源登记表和面板格式这两个基础概念。

## 凭证

quantlab 只从环境变量中读取凭证。凭证从不通过命令行传入，也从不写进配置文件或日志。

| 变量 | 用途 |
|------|------|
| `TIINGO_API_KEY` | Tiingo 美股日终价格 |
| `APCA_API_KEY_ID`、`APCA_API_SECRET_KEY` | Alpaca 的 K 线、报价和成交数据 |
| `WRDS_USERNAME` | WRDS（CRSP 和 TAQ）；密码从 `~/.pgpass` 读取 |
| `WANDB_API_KEY` | 可选，训练时的 Weights & Biases 日志 |
| `QUANTLAB_DATA_DIR` | 可选，下载数据和转换后数据的根目录 |

下载脚本位于 `scripts/`，每个脚本都可以用 `--help` 查看选项，例如
`uv run python scripts/ingest_tiingo.py --help`。文件写到哪里、下载中断后如何继续，见
[数据源指南](docs/user-guide/data-sources.md)。

## 文档

[英文文档](docs/README.md)分为三部分：*入门*介绍安装和快速上手；*用户指南*为流水线的每个步骤各写一页，
包括数据源、WRDS、数据集、股票池、因子、模型和回测；*开发者指南*说明如何添加自己的数据源、数据集、存储后端、因子、
模型或回测规则，并解释让长时间任务可以安全中断的内部机制。

每个公开的类和函数都有 [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) 格式的文档字符串，
可以在 Python 中用 `help()` 查看。

## 测试

测试套件完全离线运行，不需要任何凭证：

```bash
uv run pytest
```

KunQuant 因子测试需要编译 C++，要花几分钟。`tests/test_crsp_rebuild_measurements.py` 需要一个真实的 CRSP 数据目录，
除非 `QUANTLAB_DATA_ROOT` 指向这样的目录，否则会失败并给出说明；加上
`--ignore=tests/test_crsp_rebuild_measurements.py` 可以跳过它。

## 项目状态

quantlab 仍在积极开发中，接口可能还会变化。基于 NautilusTrader 的事件驱动回测、服务层和网页前端已在计划中，但尚未实现。

## 参与贡献

欢迎提交问题报告、提问和拉取请求。提交拉取请求之前，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。
