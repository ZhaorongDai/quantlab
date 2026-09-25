# WRDS 美股日线 Pipeline：CRSP 日线 -> Alpha101/Alpha158 -> 模型 -> 回测

[English](README.md) | 简体中文

`pipeline.py` 在 CRSP 日线数据上、针对 point-in-time 的指数成分股跑完整的研究流程，支持 S&P 500（`universe="sp500"`）和 Nasdaq-100（`universe="nasdaq100"`），并把训练和回测记录到 Weights & Biases：

1. **数据读取**：读取已转换的 CRSP 数据仓库及其成分股面板，写出两个派生仓库（`prices`、`members`）。
2. **因子计算**：在复权价格上计算 `Alpha101Stock` 和 `Alpha158Stock`，存为 Zarr。
3. **标签**：`Return`，即 t+1 开盘到 t+1+`horizon` 开盘的收益，只在成分股行上计算。
4. **模型训练**：`xgb`（`XGBoostRegressor`）、`xgb_td`（`XGBTDRegressor`）或 `realmlp`（`RealMLPRegressor`），单次训练或 walk-forward 交叉验证。
5. **回测**：`USEquityCrossectionSelectStockVectorBt`，在样本外窗口上做截面 TopN 组合。

不使用命令行参数，所有设置都在 `pipeline.py` 顶部的 `Settings` dataclass 里。

## 前置条件

先用 WRDS 账户下载并转换所需指数的成分股数据（一次即可，见 [docs/zh-CN/wrds_crsp.md](../../docs/zh-CN/wrds_crsp.md)）：

```bash
export WRDS_USERNAME=<your-wrds-username>   # 密码放在 ~/.pgpass
# S&P 500（CRSP 自带的成分股记录，从 1925 年起）
uv run python scripts/ingest_wrds_crsp.py --universe crsp_sp500 \
    --start-date 2010-01-01 --end-date 2024-12-31 --to-zarr
# Nasdaq-100（Compustat 成分股，经 CCM 映射到 PERMNO，从 1995 年起；
# 需要 Compustat 和 CCM 权限）
uv run python scripts/ingest_wrds_crsp.py --universe comp_nasdaq100 \
    --start-date 2010-01-01 --end-date 2024-12-31 --to-zarr
```

每条命令在 `data/data/us_equity/1d/` 下写出两个仓库：`wrds_crsp_<universe>_1d.zarr`（窗口内曾经是成分股的所有 PERMNO 的价格）和 `wrds_crsp_<universe>_membership.zarr`（每日的 `is_member`），其中 `<universe>` 为 `sp500` 或 `nasdaq100`。pipeline 从同一个数据根目录读取两者（`QUANTLAB_DATA_DIR`、仓库旁的 `data/`，或 `Settings.data_root`）。

KunQuant 需要编译因子计算图，因此需要 C++ 编译器。脚本在 macOS 上会自动设置 `OMP_NUM_THREADS=1`（xgboost 与 torch 同进程）。

Weights & Biases 记录默认开启（`wandb_mode="online"`）：先运行一次 `wandb login`；或者把 `wandb_mode` 设为 `"offline"`（写到本地 `wandb/`，之后用 `wandb sync` 上传）或 `"disabled"`。

## 运行

修改 `Settings`（至少改 `universe`、`model`、日期和超参数），然后：

```bash
uv run python examples/wrds_us_equity/pipeline.py
```

也可以在 VS Code / Jupyter 里逐个运行 `# %%` 单元。每个阶段都是一个函数（`prepare_stores`、`compute_factors`、`train`、`backtest`），在 notebook 里可以只重跑改动的那一步：

```python
import pipeline as p
s = p.Settings(universe="nasdaq100", model="realmlp", use_cv=True)
p.main(s)
```

## 设置项

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `universe` | `"sp500"` | `"sp500"` 或 `"nasdaq100"`；决定读取哪组输入仓库、成分股面板和输出目录 |
| `wandb_mode` | `"online"` | `"online"`、`"offline"` 或 `"disabled"` |
| `model` | `"xgb"` | `"xgb"`、`"xgb_td"` 或 `"realmlp"` |
| `hyperparameters` | `{}` | 覆盖 `DEFAULT_HYPERPARAMETERS[model]`；键名是各模型自己的（`xgb.train` 参数，或 pytabkit 构造参数） |
| `early_stopping`、`early_stopping_patience`、`val_size` | `True`、`50`、`0.2` | 在训练窗口末尾 `val_size` 比例上早停；patience 对 xgb/xgb_td 是 boosting 轮数，对 realmlp 是 epoch |
| `start_date`、`end_date` | 2012-01-01、2024-12-31 | 数据窗口；因子预热数据从它之前读取 |
| `train_start` ... `test_end` | 2012-2019 / 2020-2024 | 训练窗口与样本外测试窗口 |
| `use_cv`、`cv_train_periods`、`cv_gap_periods` | `False`、1250、5 | walk-forward 折（`train_cv`），回测时拼接成一条样本外曲线（`run_cv`） |
| `factor_window` | 400 | 因子回看长度（自然日） |
| `alpha101_names`、`alpha158_names` | `None` | 各因子库的子集；`None` 表示全部 82 / 169 列 |
| `horizon` | 5 | 标签周期（bar 数） |
| `rebalance_periods`、`top_n`、`direction` | 5、`None`、`"long_only"` | 每 5 个 bar 调仓到得分最高的 `top_n` 只（`None` 时 sp500 为 50、nasdaq100 为 10）；`"long_short"` 同时做空得分最低的 `top_n` 只 |
| `fees`、`slippage`、`init_cash` | 0.0005、0.0005、1e6 | 按比例的手续费和滑点，以及初始资金 |

## 输出

所有产物都写在 `<数据根目录>/data/pipeline/wrds_<universe>/` 下：

```text
prices.zarr, members.zarr     派生价格仓库（第 1 步）
factor/alpha101.zarr, factor/alpha158.zarr, label/ret_<h>.zarr
models/<model>/...            检查点、config.json、cv_folds.json
backtests/<model>/...         权重、净值、metrics.json、report.html
```

## Weights & Biases 记录的内容

- **训练**：每次 `train()` 一个 run；CV 时每折一个 run，外加一个记录各折均值的 `<Model>_cv_summary` run，项目名取自 trial 目录。内容包括完整配置和最终生效的超参数，train/val/test 指标（MSE、RMSE、MAE、R²、IC、RankIC），以及各模型特有的内容：`xgb` 的逐轮 `train-`/`val-` 曲线和特征重要性，`xgb_td` 的最优轮数，`realmlp` 的停止 epoch。
- **回测**：在 `USEquityCrossectionSelectStockVectorBt_backtest` 项目下一个 run，以运行目录命名：带数据指纹的回测配置、全区间/样本内/样本外指标（写入 summary），以及 HTML 报告。

回测的运行目录可以用 `quantlab.utils.module.load_backtester_from_config` 重建并重跑，见 [docs/zh-CN/backtest.md](../../docs/zh-CN/backtest.md)。

## 股票池的处理

- **幸存者偏差**：CRSP 成分股名单包含窗口内任何时候是成分股的所有 PERMNO（含已退市的），CRSP 也带有退市收益。
- **point-in-time 成分**：`members.zarr` 是把非成分股格子置为 NaN 的价格面板。标签读取它，所以训练样本只包含成分股行；回测用它定价，所以只能买入当时的成分股，被剔除出指数的持仓会在下一个 bar 卖出。因子读取 `prices.zarr`，滚动窗口能看到完整历史。
- **补齐 symbol**：symbol 轴用全 NaN 的 PERMNO（-1、-2、……）补齐到 16 的倍数，因为 KunQuant 批量计算要求 symbol 数是 SIMD 宽度的倍数。补齐列既没有标签也没有价格，因此不会被训练，也不会被交易。

## 注意事项

- Alpha101/Alpha158 输出是原始值（未标准化）。树模型不需要标准化，RealMLP 会自己做 robust scaling；但 pytabkit 的两个模型（`xgb_td`、`realmlp`）会把缺失特征填成 0，而原生 `xgb` 会把 NaN 当作缺失值处理。
- 实验时可以只用因子子集来减小模型规模。设置 `alpha101_names`/`alpha158_names` 需要 `BaseModel.get_factor_names` 使用配置里的 `factor_names`。
- 内存随 symbol 数 × 天数 × 特征数增长：1,000 个 PERMNO、13 年、全部 251 个特征，float32 大约 3 GB。
