# quantlab

[English](README.md) | 简体中文

quantlab 是一个由配置驱动的量化股票研究后端，覆盖从原始行情数据到因子、收益预测、目标持仓权重和回测的完整流程。

- **文档：** [docs/zh-CN/README.md](docs/zh-CN/README.md)
- **源代码：** https://github.com/ZhaorongDai/quantlab2
- **参与贡献：** 见[欢迎贡献](#欢迎贡献)
- **问题反馈：** https://github.com/ZhaorongDai/quantlab2/issues

它提供：

- 全流程统一的数据格式：以 `(timestamp, symbol)` 为索引的 `xarray.Dataset`，以 Zarr 落盘，因子和模型不会经过
  DataFrame
- 支持断点续传的 Tiingo、Alpaca 和 WRDS（CRSP 日频股票、TAQ 报价）下载器，并带有体量护栏，过大的请求会在发出之前被拒绝
- 没有幸存者偏差的股票池：时点指数成分和全市场名册
- 两个因子后端：KunQuant（批量与流式）和 Polars（批量）
- 深度学习与树模型共用一套接口，支持滚动交叉验证
- 基于 vectorbt 的向量化回测，样本内外分开报告，每次运行的目录都可以重建并重新运行

quantlab 是研究后端，没有网页前端，也没有下单路由引擎。

```text
 数据源   ->  数据集   ->  因子 / 标签  ->  模型     ->  回测
 (Tiingo,    (原始文件    (KunQuant 或     (torch 或    (目标权重,
  Alpaca,     转成面板)     Polars)         xgboost)     vectorbt, 报告)
  WRDS)
```

## 安装

quantlab 需要 Python 3.13 或更高版本，并使用 [uv](https://docs.astral.sh/uv/) 管理环境。

```bash
git clone https://github.com/ZhaorongDai/quantlab2.git
cd quantlab2
uv sync
```

深度学习模型在有 CUDA GPU 时使用 GPU，否则回退到 CPU。在 macOS 上，如果同一进程同时导入
PyTorch 和 XGBoost，需要设置 `OMP_NUM_THREADS=1`，因为两个库自带的 OpenMP 运行时会冲突。

## 快速开始

下面的例子可以离线运行。它先列出 quantlab 已知的数据源，再构造一个很小的价格面板，并通过数据集对象读回。

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.registry import DataSourceRegistry, credential_status

for source in DataSourceRegistry.all():
    print(source.vendor, credential_status(source))

root = Path(tempfile.mkdtemp())
timestamps = pd.date_range("2024-01-01", periods=5, freq="B")
symbols = ["AAPL", "MSFT", "NVDA"]
close = 100 + np.random.default_rng(0).normal(size=(5, 3)).cumsum(axis=0)
xr.Dataset(
    {"adjClose": (("timestamp", "symbol"), close)},
    coords={"timestamp": timestamps, "symbol": symbols},
).to_zarr(root / "prices.zarr", mode="w")

dataset = StockDataset(
    DatasetConfig(
        zarr_file_path=str(root / "prices.zarr"),
        raw_data_dir_path=str(root / "raw"),
        catalog_path=str(root / "catalog"),
        market="us_equity",
        frequency="1d",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
)
print(dataset.read().get_xarray_dataset())
```

输出：

```text
alpaca {'APCA_API_KEY_ID': False, 'APCA_API_SECRET_KEY': False}
tiingo {'TIINGO_API_KEY': False}
wrds {'WRDS_USERNAME': False}
<xarray.Dataset> Size: 208B
Dimensions:    (timestamp: 5, symbol: 3)
Coordinates:
  * timestamp  (timestamp) datetime64[ns] 40B 2024-01-01 ... 2024-01-05
  * symbol     (symbol) <U4 48B 'AAPL' 'MSFT' 'NVDA'
Data variables:
    adjClose   (timestamp, symbol) float64 120B ...
```

`False` 表示该数据源所需的凭证没有在当前环境中设置。[用户指南](docs/zh-CN/README.md)
从这里接着讲下载数据、构建因子、训练模型和运行回测。

## 数据源与凭证

quantlab 的所有凭证都只从环境变量读取。不接受命令行传入，也不会写进配置文件或日志。

| 变量 | 用途 |
|------|------|
| `TIINGO_API_KEY` | Tiingo 美股日线数据 |
| `APCA_API_KEY_ID`、`APCA_API_SECRET_KEY` | Alpaca 行情数据（K 线、报价、逐笔成交） |
| `WRDS_USERNAME` | WRDS（CRSP、TAQ）。密码来自 `~/.pgpass` 文件。 |
| `WANDB_API_KEY` | 模型训练时的 Weights & Biases 日志（可选） |
| `QUANTLAB_DATA_DIR` | 下载数据与转换结果的根目录（可选） |

数据根目录按以下顺序确定：下载脚本的 `--data-dir` 参数，其次是 `QUANTLAB_DATA_DIR`，最后是仓库根目录下的
`data/`。在根目录之下，转换后的 Zarr 存储位于 `data/<market>/<frequency>/`，原始下载位于
`downloads/<market>/<frequency>/`。

`scripts/` 中的脚本负责下载与转换数据。每个脚本都可以用 `--help` 查看参数。

```bash
uv run python scripts/ingest_tiingo.py --help
```

## 测试

测试套件不需要网络，也不需要凭证。

```bash
uv run pytest
```

`tests/test_crsp_rebuild_measurements.py` 会重建真实的 CRSP 存储，因此在 `QUANTLAB_DATA_ROOT` 没有指向
真实存储时会报错并给出说明。可以加 `--ignore=tests/test_crsp_rebuild_measurements.py` 跳过它。KunQuant
因子测试需要编译 C++，耗时几分钟。

## 状态

quantlab 仍在积极开发中，接口可能变化。基于 NautilusTrader 的事件驱动回测、服务层和网页前端尚未实现。

## 欢迎贡献

欢迎提交 issue 和 pull request。提交前请先运行 `uv run pytest`，并让 docstring 保持代码库统一的 numpydoc 格式：
一行摘要，必要时补充 `Parameters`、`Returns`、`Raises`，再加一个简短的 `Examples` 小节。
