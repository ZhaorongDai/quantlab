# WRDS 美股日线 Pipeline：CRSP 日线 -> Alpha101/Alpha158 -> 模型或因子分析

[English](README.md) | 简体中文

每个"股票池 × 模型"一个自成一体的脚本，外加每个股票池一个因子分析脚本，都跑在 CRSP 日线数据上：point-in-time 的 S&P 500 或 Nasdaq-100 成分股，或 CRSP 全市场（所有上市普通股）。每个文件只 import quantlab，可以单独拷走修改。

| 股票池 | 模型 pipeline | 因子分析 |
| --- | --- | --- |
| S&P 500 | `sp500_xgb.py`、`sp500_xgb_td.py`、`sp500_realmlp.py` | `sp500_factor_analysis.py` |
| Nasdaq-100 | `nasdaq100_xgb.py`、`nasdaq100_xgb_td.py`、`nasdaq100_realmlp.py` | `nasdaq100_factor_analysis.py` |
| CRSP 全市场 | `market_xgb.py`、`market_xgb_td.py`、`market_realmlp.py` | `market_factor_analysis.py` |

三个模型分别是 `XGBoostRegressor`（`xgb.train`，原生早停）、`XGBTDRegressor`（pytabkit 调优默认参数的 XGBoost）和 `RealMLPRegressor`（pytabkit 调优默认参数的 MLP）。每个模型 pipeline 都跑同样的五步：

1. **数据读取**：读取已转换的 CRSP 数据仓库及其成分股面板，写出两个派生仓库（`prices`、`members`）。
2. **因子计算**：在复权价格上计算 `Alpha101Stock` 和 `Alpha158Stock`，存为 Zarr。
3. **标签**：`Return`，即 t+1 开盘到 t+1+`HORIZON` 开盘的收益，只在成分股行上计算。
4. **模型训练**：在训练窗口上训练一次。
5. **回测**：`USEquityCrossectionSelectStockVectorBt`，在样本外窗口上做截面 TopN 组合，并与买入持有的 SPY（S&P 500 和全市场）或 QQQ（Nasdaq-100）对比，记录到 Weights & Biases。

全市场脚本没有第 1 步：全市场 store 在转换时已按天筛成普通股，所以每一步都通过 `CrspStockDataset` 直接读它，不写派生仓库。因子分析 pipeline 跑数据、因子、标签几步，然后对两个因子库的每一列调用 `Factor.analyze()`，不训练模型。不使用命令行参数，也没有设置对象：每个文件顶部只有几个常量（`DATA_ROOT`、日期、`HORIZON`、`WANDB_MODE`），quantlab 的各个 config 都在用到的地方直接构造（`DatasetConfig`、`FactorConfig`、`MLConfig`、`CrossSectionBacktestConfig`），每一步做什么就是它拿到的 config。

## 前置条件

先用 WRDS 账户下载并转换所需指数的成分股数据（一次即可，见 [docs/zh-CN/wrds_crsp.md](../../docs/zh-CN/wrds_crsp.md)）：

```bash
export WRDS_USERNAME=<your-wrds-username>   # 密码放在 ~/.pgpass
# S&P 500（CRSP 自带的成分股记录，从 1925 年起）
uv run python scripts/wrds/index.py --index sp500 --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --zarr-dir data/data/us_equity/1d
# Nasdaq-100（Compustat 成分股，经 CCM 映射到 PERMNO，从 1995 年起；
# 需要 Compustat 和 CCM 权限）
uv run python scripts/wrds/index.py --index nasdaq100 --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --zarr-dir data/data/us_equity/1d
# CRSP 全市场（所有上市普通股；`--security-filter` 选证券类型）
uv run python scripts/wrds/market.py --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --zarr-dir data/data/us_equity/1d
# 基准 ETF，按 CRSP PERMNO 下载（SPY 84398、QQQ 86755），每个 ETF 一个仓库
uv run python scripts/wrds/etf.py --etf spy,qqq --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --zarr-dir data/data/us_equity/1d
```

`--end` 默认为今天，并截到 CRSP 年度发布的最后一天；每个脚本都会转换成 Zarr；`--refresh` 让每个 PERMNO 从各自的水位继续。`--download-dir` 和 `--zarr-dir` 默认为当前目录；上面这组相对仓库根目录的取值会把 store 放到 pipeline 读取的位置。

每次 `index.py` 运行在 `data/data/us_equity/1d/` 下写出两个仓库：`wrds_crsp_<index>_1d.zarr`（窗口内曾经是成分股的所有 PERMNO 的价格）和 `wrds_crsp_<index>_membership.zarr`（每日的 `is_member`），其中 `<index>` 为 `sp500` 或 `nasdaq100`。`market.py` 写出 `wrds_crsp_market_1d.zarr`（全市场脚本直接读它）和 `wrds_crsp_market_membership.zarr`（上市面板，全市场脚本用不到）。`etf.py` 写出 `wrds_crsp_spy_1d.zarr` 和 `wrds_crsp_qqq_1d.zarr`。pipeline 从同一个数据根目录读取它们（`QUANTLAB_DATA_DIR`、仓库旁的 `data/`，或每个脚本顶部的 `DATA_ROOT`）。

KunQuant 需要编译因子计算图，因此需要 C++ 编译器。模型脚本在 macOS 上会自动设置 `OMP_NUM_THREADS=1`（xgboost 与 torch 同进程）。

Weights & Biases 记录默认开启（`wandb_mode="online"`）：先运行一次 `wandb login`；或者把 `wandb_mode` 设为 `"offline"`（写到本地 `wandb/`，之后用 `wandb sync` 上传）或 `"disabled"`。

## 运行

打开对应股票池和模型的脚本，如果仓库不在 quantlab 默认的数据根目录下就改 `DATA_ROOT`，再修改想改的常量和 config（日期、`hyperparameters`、`top_n` 等），然后运行：

```bash
uv run python examples/wrds_us_equity/sp500_xgb.py
uv run python examples/wrds_us_equity/nasdaq100_factor_analysis.py
```

也可以在 VS Code / Jupyter 里逐个运行 `# %%` 单元。每一步都是一个函数（`prepare_stores`、`compute_factors`、`train`、`backtest`，或 `analyze`），在 notebook 里可以只重跑改动的那一步。

## 设置项

所有设置都在每个脚本的顶部，顺序如下：

| 位置 | 内容 |
| --- | --- |
| `DATA_ROOT`、`STORES`、`RAW`、`REFERENCE`、`WORK` | 数据根目录（`get_data_root()`：`QUANTLAB_DATA_DIR` 或仓库旁的 `data/`）及其下的输入输出位置 |
| `START`、`END` | 数据窗口；因子预热数据从 `START` 之前读取 |
| `TRAIN_START` ... `TEST_END` | 训练窗口与样本外测试窗口（模型 pipeline） |
| `HORIZON` | 标签周期（bar 数） |
| `WANDB_MODE` | `"online"`、`"offline"` 或 `"disabled"`（模型 pipeline） |
| `factors_and_label()` | 两个因子库的 `FactorConfig`（`window=400`、`njobs=16`、`factor_names` 不设即全部列）和标签的 `FactorConfig` |
| `build_model()` | `MLConfig`：早停、`val_size` 和模型自己的 `hyperparameters`（`xgb.train` 参数，或 pytabkit 构造参数） |
| `backtest()` | `CrossSectionBacktestConfig`：`rebalance_periods`、`top_n`（S&P 500 为 50，Nasdaq-100 为 10，全市场为 100）、`direction`、成本，以及 ETF `benchmark_dataset` |
| `analyze()` | `Factor.analyze()` 的 `quantiles` 和 `factor_names`（因子分析 pipeline） |

## 输出

所有内容都写在 `<数据根目录>/data/pipeline/wrds_<universe>/` 下：

```text
prices.zarr, members.zarr     派生价格仓库（第 1 步；仅指数脚本）
factor/alpha101.zarr, factor/alpha158.zarr, label/ret_<h>.zarr
models/<model>/...            checkpoint、config.json
backtests/<model>/...         权重、净值、metrics.json、report.html
analysis/alpha101/, analysis/alpha158/
                              summary.json 和 .csv、ic.csv、monthly_ic.csv、
                              quantile_returns.csv、turnover.csv、每列一张 PNG、
                              config.json（因子和标签的配置）
```

## Weights & Biases 记录的内容

- **训练**：每次 `train()` 一个 run，项目名取自 trial 目录。内容包括完整配置和最终生效的超参数，train/val/test 指标（MSE、RMSE、MAE、R²、IC、RankIC），以及各模型特有的内容：`xgb` 的逐轮 `train-`/`val-` 曲线和特征重要性，`xgb_td` 的最优轮数，`realmlp` 的停止 epoch。
- **回测**：在 `USEquityCrossectionSelectStockVectorBt_backtest` 项目下一个 run，以运行目录命名：带数据指纹的回测配置、全区间/样本内/样本外指标（写入 summary；有基准时还有 `benchmark/...` 和 `relative/...`），以及 HTML 报告。

## 基准对比

基准数据是 ETF 在 CRSP 日线表（`crsp_a_stock.dsf_v2`，按 PERMNO 选取）中的逐日记录，和其他 CRSP 面板一样转换：`adjOpen`/`adjClose` 是全收益复权价，所以买入持有包含 ETF 的分红（扣除管理费，和真实持有一致）。它是可交易的 ETF，不是指数点位。

启用基准时（默认启用），回测会用同样的 `init_cash`、手续费、滑点和"下一根 bar 开盘成交"的规则买入并持有 ETF，所以两条净值曲线可以逐 bar 对比。每个 ETF 放在自己的单标的仓库里（`wrds_crsp_spy_1d.zarr`、`wrds_crsp_qqq_1d.zarr`），不会进入股票面板，否则它会和自己的成分股一起参与排序。`metrics.json` 会多出两个指标块，各自按全区间/样本内/样本外拆分：

- `benchmark`：ETF 自身的收益统计。
- `relative`：组合相对 ETF 的表现：`excess_return`（相对净值 − 1）、`excess_return_annualized`、`excess_max_drawdown`、`tracking_error`、`information_ratio`、`beta`、`correlation`、`capm_alpha`、`win_rate_vs_benchmark`。

`report.html` 会在组合净值旁画出基准净值，并增加超额收益和超额回撤两行。pipeline 的日志行会打印核心数字。在 `backtest()` 里传 `benchmark_dataset=None` 可跳过对比。

回测的运行目录可以用 `quantlab.utils.module.load_backtester_from_config` 重建并重跑，见 [docs/zh-CN/backtest.md](../../docs/zh-CN/backtest.md)。

## 股票池的处理

- **幸存者偏差**：CRSP 成分股名单包含窗口内任何时候是成分股的所有 PERMNO（含已退市的），CRSP 也带有退市收益。
- **point-in-time 成分**：`members.zarr` 是把非成分股格子置为 NaN 的价格面板。标签读取它，所以训练样本只包含成分股行；回测用它定价，所以只能买入当时的成分股，被剔除出指数的持仓会在下一个 bar 卖出。因子读取 `prices.zarr`，滚动窗口能看到完整历史。

## 注意事项

- Alpha101/Alpha158 输出是原始值（未标准化）。树模型不需要标准化，RealMLP 会自己做 robust scaling；但 pytabkit 的两个模型（`xgb_td`、`realmlp`）会把缺失特征填成 0，而原生 `xgb` 会把 NaN 当作缺失值处理。
- 实验时可以只用因子子集来减小模型规模。设置 `alpha101_names`/`alpha158_names` 需要 `BaseModel.get_factor_names` 使用配置里的 `factor_names`。
- 全市场脚本要读几千个 PERMNO；第一次跑先缩小 `START`/`END` 或固定 `factor_names`。内存随 symbol 数 × 天数 × 特征数增长：1,000 个 PERMNO、13 年、全部 251 个特征，float32 大约 3 GB。
