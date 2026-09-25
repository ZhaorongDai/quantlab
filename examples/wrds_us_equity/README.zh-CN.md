# WRDS 美股日线 Pipeline：CRSP 日线 -> Alpha101/Alpha158 -> 模型 -> 回测

[English](README.md) | 简体中文

`pipeline.py` 在 CRSP 日线数据上、针对 point-in-time 的 S&P 500 跑完整的研究流程：

1. **数据读取**：读取已转换的 CRSP 数据仓库及其成分股面板，写出两个派生仓库（`prices`、`members`）。
2. **因子计算**：在复权价格上计算 `Alpha101Stock` 和 `Alpha158Stock`，存为 Zarr。
3. **标签**：`Return`，即 t+1 开盘到 t+1+`horizon` 开盘的收益，只在成分股行上计算。
4. **模型训练**：`xgb`（`XGBoostRegressor`）、`xgb_td`（`XGBTDRegressor`）或 `realmlp`（`RealMLPRegressor`），单次训练或 walk-forward 交叉验证。
5. **回测**：`USEquityCrossectionSelectStockVectorBt`，在样本外窗口上做截面 TopN 组合。

不使用命令行参数，所有设置都在 `pipeline.py` 顶部的 `Settings` dataclass 里。

## 前置条件

先用 WRDS 账户下载并转换 CRSP 的 S&P 500 成分股（一次即可，见 [docs/zh-CN/wrds_crsp.md](../../docs/zh-CN/wrds_crsp.md)）：

```bash
export WRDS_USERNAME=<your-wrds-username>   # 密码放在 ~/.pgpass
uv run python scripts/ingest_wrds_crsp.py --universe crsp_sp500 \
    --start-date 2010-01-01 --end-date 2024-12-31 --to-zarr
```

它会写出 `data/data/us_equity/1d/wrds_crsp_sp500_1d.zarr`（窗口内曾经是成分股的所有 PERMNO 的价格）和 `wrds_crsp_sp500_membership.zarr`（每日的 `is_member`）。pipeline 从同一个数据根目录读取两者（`QUANTLAB_DATA_DIR`、仓库旁的 `data/`，或 `Settings.data_root`）。

KunQuant 需要编译因子计算图，因此需要 C++ 编译器。macOS 上请设置 `OMP_NUM_THREADS=1`（xgboost 与 torch 同进程）；未登录 Weights & Biases 时设置 `WANDB_MODE=disabled` 或 `offline`。

## 运行

修改 `Settings`（至少改 `model`、日期和超参数），然后：

```bash
WANDB_MODE=disabled uv run python examples/wrds_us_equity/pipeline.py
```

也可以在 VS Code / Jupyter 里逐个运行 `# %%` 单元。每个阶段都是一个函数（`prepare_stores`、`compute_factors`、`train`、`backtest`），在 notebook 里可以只重跑改动的那一步：

```python
import pipeline as p
s = p.Settings(model="realmlp", use_cv=True)
p.main(s)
```

## 设置项

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `model` | `"xgb"` | `"xgb"`、`"xgb_td"` 或 `"realmlp"` |
| `hyperparameters` | `{}` | 覆盖 `DEFAULT_HYPERPARAMETERS[model]`；键名是各模型自己的（`xgb.train` 参数，或 pytabkit 构造参数） |
| `early_stopping`、`early_stopping_patience`、`val_size` | `True`、`50`、`0.2` | 在训练窗口末尾 `val_size` 比例上早停；patience 对 xgb/xgb_td 是 boosting 轮数，对 realmlp 是 epoch |
| `start_date`、`end_date` | 2012-01-01、2024-12-31 | 数据窗口；因子预热数据从它之前读取 |
| `train_start` ... `test_end` | 2012-2019 / 2020-2024 | 训练窗口与样本外测试窗口 |
| `use_cv`、`cv_train_periods`、`cv_gap_periods` | `False`、1250、5 | walk-forward 折（`train_cv`），回测时拼接成一条样本外曲线（`run_cv`） |
| `factor_window` | 400 | 因子回看长度（自然日） |
| `alpha101_names`、`alpha158_names` | `None` | 各因子库的子集；`None` 表示全部 82 / 169 列 |
| `horizon` | 5 | 标签周期（bar 数） |
| `rebalance_periods`、`top_n`、`direction` | 5、50、`"long_only"` | 每 5 个 bar 调仓到得分最高的 50 只；`"long_short"` 同时做空得分最低的 50 只 |
| `fees`、`slippage`、`init_cash` | 0.0005、0.0005、1e6 | 按比例的手续费和滑点，以及初始资金 |

## 输出

所有产物都写在 `<数据根目录>/data/pipeline/wrds_sp500/` 下：

```text
prices.zarr, members.zarr     派生价格仓库（第 1 步）
factor/alpha101.zarr, factor/alpha158.zarr, label/ret_<h>.zarr
models/<model>/...            检查点、config.json、cv_folds.json
backtests/<model>/...         权重、净值、metrics.json、report.html
```

回测的运行目录可以用 `quantlab.utils.module.load_backtester_from_config` 重建并重跑，见 [docs/zh-CN/backtest.md](../../docs/zh-CN/backtest.md)。

## 股票池的处理

- **幸存者偏差**：CRSP 成分股名单包含窗口内任何时候是成分股的所有 PERMNO（含已退市的），CRSP 也带有退市收益。
- **point-in-time 成分**：`members.zarr` 是把非成分股格子置为 NaN 的价格面板。标签读取它，所以训练样本只包含成分股行；回测用它定价，所以只能买入当时的成分股，被剔除出指数的持仓会在下一个 bar 卖出。因子读取 `prices.zarr`，滚动窗口能看到完整历史。
- **补齐 symbol**：symbol 轴用全 NaN 的 PERMNO（-1、-2、……）补齐到 16 的倍数，因为 KunQuant 批量计算要求 symbol 数是 SIMD 宽度的倍数。补齐列既没有标签也没有价格，因此不会被训练，也不会被交易。

## 注意事项

- Alpha101/Alpha158 输出是原始值（未标准化）。树模型不需要标准化，RealMLP 会自己做 robust scaling；但 pytabkit 的两个模型（`xgb_td`、`realmlp`）会把缺失特征填成 0，而原生 `xgb` 会把 NaN 当作缺失值处理。
- 实验时可以只用因子子集来减小模型规模。设置 `alpha101_names`/`alpha158_names` 需要 `BaseModel.get_factor_names` 使用配置里的 `factor_names`。
- 内存随 symbol 数 × 天数 × 特征数增长：1,000 个 PERMNO、13 年、全部 251 个特征，float32 大约 3 GB。
