# WRDS US-equity pipelines: CRSP daily -> Alpha101/Alpha158 -> model or factor analysis

English | [简体中文](README.zh-CN.md)

One self-contained script per universe and model, plus one factor analysis per universe, all on CRSP daily data for a point-in-time index. Each file imports only quantlab, so it can be copied out and edited on its own.

| Universe | Model pipelines | Factor analysis |
| --- | --- | --- |
| S&P 500 | `sp500_xgb.py`, `sp500_xgb_td.py`, `sp500_realmlp.py` | `sp500_factor_analysis.py` |
| Nasdaq-100 | `nasdaq100_xgb.py`, `nasdaq100_xgb_td.py`, `nasdaq100_realmlp.py` | `nasdaq100_factor_analysis.py` |

The heads are `XGBoostRegressor` (`xgb.train`, native early stopping), `XGBTDRegressor` (pytabkit tuned-default XGBoost) and `RealMLPRegressor` (pytabkit tuned-default MLP). Every model pipeline runs the same five steps:

1. **Data**: read the converted CRSP store and its membership panel, then write two derived stores (`prices`, `members`).
2. **Factors**: `Alpha101Stock` and `Alpha158Stock` on adjusted prices, saved as Zarr.
3. **Label**: `Return`, the open-to-open return from t+1 to t+1+`HORIZON`, computed on member rows only.
4. **Model**: trained once on the training window.
5. **Backtest**: `USEquityCrossectionSelectStockVectorBt`, a TopN cross-sectional portfolio over the out-of-sample window, compared against buy-and-hold SPY (S&P 500) or QQQ (Nasdaq-100), logged to Weights & Biases.

A factor-analysis pipeline runs steps 1 to 3 and then `Factor.analyze()` on every column of both libraries instead of a model. There is no command-line interface and no settings object: the top of each file holds a few constants (`DATA_ROOT`, the dates, `HORIZON`, `WANDB_MODE`) and every quantlab config is constructed in place (`DatasetConfig`, `FactorConfig`, `MLConfig`, `CrossSectionBacktestConfig`), so what a step does is the config it is given.

## Prerequisites

Download and convert the roster of the index you want once (this needs a WRDS account; see [docs/wrds_crsp.md](../../docs/wrds_crsp.md)):

```bash
export WRDS_USERNAME=<your-wrds-username>   # password in ~/.pgpass
# S&P 500 (CRSP's own membership, from 1925)
uv run python scripts/wrds/index.py --index sp500 --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --zarr-dir data/data/us_equity/1d
# Nasdaq-100 (Compustat membership linked through CCM, from 1995;
# needs the Compustat and CCM schemas)
uv run python scripts/wrds/index.py --index nasdaq100 --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --zarr-dir data/data/us_equity/1d
# The benchmark ETFs, by CRSP PERMNO (SPY 84398, QQQ 86755), one store each
uv run python scripts/wrds/etf.py --etf spy,qqq --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --zarr-dir data/data/us_equity/1d
```

`--end` defaults to today and is clipped to the last day of the CRSP release; every script converts to Zarr; `--refresh` continues each PERMNO from its watermark. `--download-dir` and `--zarr-dir` default to the current directory; the values above, relative to the repository root, put the stores where the pipeline reads them.

Each `index.py` run writes two stores under `data/data/us_equity/1d/`: `wrds_crsp_<index>_1d.zarr` (prices of every PERMNO that was a member at some point in the window) and `wrds_crsp_<index>_membership.zarr` (`is_member` per day), with `<index>` = `sp500` or `nasdaq100`. `etf.py` writes `wrds_crsp_spy_1d.zarr` and `wrds_crsp_qqq_1d.zarr`. The pipeline reads them from the same data root (`QUANTLAB_DATA_DIR`, or `data/` beside the repository, or `DATA_ROOT` at the top of each script).

KunQuant compiles the factor graphs, so a C++ compiler is required. The model scripts set `OMP_NUM_THREADS=1` on macOS themselves (xgboost and torch in one process).

Weights & Biases logging is on by default (`wandb_mode="online"`): run `wandb login` once, or set `wandb_mode` to `"offline"` (runs are written to `wandb/` and uploaded later with `wandb sync`) or `"disabled"`.

## Run

Open the script for your universe and model, change `DATA_ROOT` if the stores are not under quantlab's default data root, edit the constants and the config objects you want to change (dates, `hyperparameters`, `top_n`, ...), then run it:

```bash
uv run python examples/wrds_us_equity/sp500_xgb.py
uv run python examples/wrds_us_equity/nasdaq100_factor_analysis.py
```

or run the `# %%` cells one at a time in VS Code or Jupyter. Every step is a function (`prepare_stores`, `compute_factors`, `train`, `backtest`, or `analyze`), so a notebook can rerun only the step it changed.

## Settings

Everything lives at the top of each script, in this order:

| Where | What |
| --- | --- |
| `DATA_ROOT`, `STORES`, `RAW`, `REFERENCE`, `WORK` | the data root (`get_data_root()`: `QUANTLAB_DATA_DIR` or `data/` beside the repository) and the input and output locations under it |
| `START`, `END` | data window; the factor warm-up is read before `START` |
| `TRAIN_START` ... `TEST_END` | training and out-of-sample test windows (model pipelines) |
| `HORIZON` | label horizon in bars |
| `WANDB_MODE` | `"online"`, `"offline"` or `"disabled"` (model pipelines) |
| `factors_and_label()` | the two `FactorConfig`s of the alpha libraries (`window=400`, `njobs=16`, `factor_names` unset = all columns) and the label's |
| `build_model()` | the `MLConfig`: early stopping, `val_size` and the head's `hyperparameters` (`xgb.train` parameters, or the pytabkit constructor arguments) |
| `backtest()` | the `CrossSectionBacktestConfig`: `rebalance_periods`, `top_n` (50 for the S&P 500, 10 for the Nasdaq-100), `direction`, costs, and the ETF `benchmark_dataset` |
| `analyze()` | `quantiles` and `factor_names` of `Factor.analyze()` (factor-analysis pipelines) |

## Outputs

Everything is written under `<data root>/data/pipeline/wrds_<universe>/`:

```text
prices.zarr, members.zarr     derived price stores (step 1)
factor/alpha101.zarr, factor/alpha158.zarr, label/ret_<h>.zarr
models/<model>/...            checkpoints, config.json
backtests/<model>/...         weights, equity, metrics.json, report.html
analysis/alpha101/, analysis/alpha158/
                              summary.json and .csv, ic.csv, monthly_ic.csv,
                              quantile_returns.csv, turnover.csv, one PNG
                              per column, config.json (factor and label configs)
```

## What is logged to Weights & Biases

- **Training**: one run per `train()`, in a project named after the trial directory. The runs hold the full config and resolved hyperparameters, the train/val/test metrics (MSE, RMSE, MAE, R², IC, RankIC) and, per head, the per-round `train-`/`val-` curves and feature importance (`xgb`), the best round (`xgb_td`) or the stopping epoch (`realmlp`).
- **Backtest**: one run in the `USEquityCrossectionSelectStockVectorBt_backtest` project, named after the run directory: the backtest config with data fingerprints, the whole / in-sample / out-of-sample metrics as summary values (plus `benchmark/...` and `relative/...` when a benchmark ran), and the HTML report.

## Benchmark comparison

The benchmark is the ETF's own daily rows from CRSP (`crsp_a_stock.dsf_v2`, selected by PERMNO), converted like any CRSP panel: `adjOpen`/`adjClose` are total-return adjusted, so the buy-and-hold includes the ETF's dividends (net of its expense ratio, like a real holding). It is the tradable ETF, not the index level.

With a benchmark (the default), the backtest also buys and holds the ETF from the same `init_cash`, with the same fees, slippage and next-bar-open fills, so the two curves compare bar for bar. Each ETF lives in its own single-symbol store (`wrds_crsp_spy_1d.zarr`, `wrds_crsp_qqq_1d.zarr`), never in the equity panel, where it would be ranked against its own constituents. `metrics.json` gains two blocks, each split whole / in-sample / out-of-sample:

- `benchmark`: the ETF's own return statistics.
- `relative`: the portfolio against the ETF: `excess_return` (relative NAV − 1), `excess_return_annualized`, `excess_max_drawdown`, `tracking_error`, `information_ratio`, `beta`, `correlation`, `capm_alpha`, `win_rate_vs_benchmark`.

`report.html` draws the benchmark NAV beside the portfolio's and adds excess-return and excess-drawdown rows; The pipeline log line prints the headline numbers. Pass `benchmark_dataset=None` in `backtest()` to skip the comparison.

A backtest run directory can be rebuilt and re-run with `quantlab.utils.module.load_backtester_from_config`; see [docs/backtest.md](../../docs/backtest.md).

## How the universe is handled

- **Survivorship**: the CRSP roster contains every PERMNO that was a member at any time in the window, delisted ones included, and CRSP carries delisting returns.
- **Point-in-time membership**: `members.zarr` is the price panel with non-member cells set to NaN. The label reads it, so training rows are member rows only. The backtest prices from it, so only current members can be bought, and a holding that leaves the index is sold on the next bar. Factors read `prices.zarr`, so their rolling windows see full history.
- **Symbol padding**: the symbol axis is padded with all-NaN PERMNOs (-1, -2, ...) to a multiple of 16, because KunQuant batch runs need a multiple of the SIMD block width. Padded columns never have a label or a price, so they are never trained on or traded.

## Notes

- Alpha101/Alpha158 outputs are raw (not normalized). Tree models do not need normalization and RealMLP robust-scales its inputs itself, but missing features are filled with 0 by the pytabkit heads (`xgb_td`, `realmlp`); the plain `xgb` head keeps NaN as missing.
- A factor subset keeps the models small while experimenting. Pinning `alpha101_names`/`alpha158_names` needs `BaseModel.get_factor_names` to honor the configured `factor_names`.
- Memory grows with symbols × days × features: 1,000 PERMNOs over 13 years with all 251 features is roughly 3 GB in float32.
