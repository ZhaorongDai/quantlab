# WRDS US-equity pipelines: CRSP daily -> Alpha101/Alpha158 -> model or factor analysis

English | [简体中文](README.zh-CN.md)

One file per model, plus a factor-analysis pipeline, all on CRSP daily data for a point-in-time index universe, the S&P 500 (`universe="sp500"`) or the Nasdaq-100 (`universe="nasdaq100"`):

| File | What it runs |
| --- | --- |
| `xgb.py` | `XGBoostRegressor` (`xgb.train`, native early stopping) -> TopN backtest |
| `xgb_td.py` | `XGBTDRegressor` (pytabkit tuned-default XGBoost) -> TopN backtest |
| `realmlp.py` | `RealMLPRegressor` (pytabkit tuned-default MLP) -> TopN backtest |
| `factor_analysis.py` | `Factor.analyze()` on every Alpha101 and Alpha158 column: an alphalens-style report per column |
| `common.py` | the data root constant, the shared settings dataclasses and the steps every pipeline calls |

Every model pipeline runs the same five steps:

1. **Data**: read the converted CRSP store and its membership panel, then write two derived stores (`prices`, `members`).
2. **Factors**: `Alpha101Stock` and `Alpha158Stock` on adjusted prices, saved as Zarr.
3. **Label**: `Return`, the open-to-open return from t+1 to t+1+`horizon`, computed on member rows only.
4. **Model**: trained once, or walk-forward.
5. **Backtest**: `USEquityCrossectionSelectStockVectorBt`, a TopN cross-sectional portfolio over the out-of-sample window, compared against a buy-and-hold ETF benchmark (SPY for the S&P 500, QQQ for the Nasdaq-100), logged to Weights & Biases.

The factor-analysis pipeline runs steps 1 to 3 and then `analyze()` instead of a model. There is no command-line interface: the data root is the `DATA_ROOT` constant in `common.py`, and every other setting is a field of a dataclass.

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

Each `index.py` run writes two stores under `data/data/us_equity/1d/`: `wrds_crsp_<index>_1d.zarr` (prices of every PERMNO that was a member at some point in the window) and `wrds_crsp_<index>_membership.zarr` (`is_member` per day), with `<index>` = `sp500` or `nasdaq100`. `etf.py` writes `wrds_crsp_spy_1d.zarr` and `wrds_crsp_qqq_1d.zarr`. The pipeline reads them from the same data root (`QUANTLAB_DATA_DIR`, or `data/` beside the repository, or the `DATA_ROOT` constant in `common.py`).

KunQuant compiles the factor graphs, so a C++ compiler is required. `common.py` sets `OMP_NUM_THREADS=1` on macOS itself (xgboost and torch in one process).

Weights & Biases logging is on by default (`wandb_mode="online"`): run `wandb login` once, or set `wandb_mode` to `"offline"` (runs are written to `wandb/` and uploaded later with `wandb sync`) or `"disabled"`.

## Run

Set `DATA_ROOT` in `common.py` if the stores are not under quantlab's default data root, edit the `Settings` at the top of the pipeline you want (the universe and dates in `data`, the windows in `train`, the head's `hyperparameters`), then run it:

```bash
uv run python examples/wrds_us_equity/xgb.py
uv run python examples/wrds_us_equity/factor_analysis.py
```

or run the `# %%` cells one at a time in VS Code or Jupyter. Every step is a function in `common.py` (`prepare_stores`, `compute_factors`, `build_model`, `train_model`, `run_backtest`), so a notebook can rerun only the step it changed:

```python
import realmlp
from common import DataSettings, TrainSettings
s = realmlp.Settings(
    data=DataSettings(universe="nasdaq100"),
    train=TrainSettings(use_cv=True),
)
realmlp.main(s)
```

## Settings

Each pipeline's `Settings` nests the shared dataclasses from `common.py` and adds its own fields.

`DataSettings` (`s.data`, every pipeline):

| Field | Default | Meaning |
| --- | --- | --- |
| `universe` | `"sp500"` | `"sp500"` or `"nasdaq100"`; picks the input stores, the membership panel and the output directory |
| `start_date`, `end_date` | 2012-01-01, 2024-12-31 | data window; factor warm-up is read before it |
| `factor_window` | 400 | factor lookback in calendar days |
| `alpha101_names`, `alpha158_names` | `None` | subsets of each library; `None` means all 82 / 169 columns |
| `njobs` | 16 | KunQuant executor threads |
| `horizon` | 5 | label horizon in bars |

`TrainSettings` (`s.train`, model pipelines):

| Field | Default | Meaning |
| --- | --- | --- |
| `train_start` ... `test_end` | 2012-2019 / 2020-2024 | training and out-of-sample test windows |
| `early_stopping`, `early_stopping_patience`, `val_size` | `True`, `50`, `0.2` | early stopping on the trailing `val_size` of the training window; patience is in boosting rounds (xgb, xgb_td) or epochs (realmlp) |
| `use_cv`, `cv_train_periods`, `cv_gap_periods` | `False`, 1250, 5 | walk-forward folds (`train_cv`), backtested as one stitched out-of-sample curve (`run_cv`) |

`BacktestSettings` (`s.backtest`, model pipelines):

| Field | Default | Meaning |
| --- | --- | --- |
| `benchmark` | `"auto"` | buy-and-hold benchmark: `"auto"` (SPY for sp500, QQQ for nasdaq100), `"spy"`, `"qqq"` or `None` |
| `rebalance_periods`, `top_n`, `direction` | 5, `None`, `"long_only"` | rebalance every 5 bars into the top `top_n` scores (`None`: 50 for sp500, 10 for nasdaq100); `"long_short"` also shorts the bottom `top_n` |
| `fees`, `slippage`, `init_cash` | 0.0005, 0.0005, 1e6 | proportional costs and starting capital |

Fields of each model pipeline's own `Settings`:

| Field | Default | Meaning |
| --- | --- | --- |
| `hyperparameters` | per head | the head's own keys: `xgb.train` parameters for `xgb.py`, the pytabkit constructor arguments for `xgb_td.py` and `realmlp.py` |
| `wandb_mode` | `"online"` | `"online"`, `"offline"` or `"disabled"` |

Fields of `factor_analysis.py`'s `Settings`:

| Field | Default | Meaning |
| --- | --- | --- |
| `quantiles` | 5 | equal-count factor buckets per day |
| `recompute` | `True` | `False` reads the stores an earlier run wrote instead of rebuilding them |
| `output_dir` | `None` | report directory; `None` means `<work>/analysis/<library>` |

## Outputs

Everything is written under `<data root>/data/pipeline/wrds_<universe>/`:

```text
prices.zarr, members.zarr     derived price stores (step 1)
factor/alpha101.zarr, factor/alpha158.zarr, label/ret_<h>.zarr
models/<model>/...            checkpoints, config.json, cv_folds.json
backtests/<model>/...         weights, equity, metrics.json, report.html
analysis/alpha101/, analysis/alpha158/
                              summary.json and .csv, ic.csv, monthly_ic.csv,
                              quantile_returns.csv, turnover.csv, one PNG
                              per column, config.json (factor and label configs)
```

## What is logged to Weights & Biases

- **Training**: one run per `train()`, or one per CV fold plus a `<Model>_cv_summary` run with the fold means, in a project named after the trial directory. The runs hold the full config and resolved hyperparameters, the train/val/test metrics (MSE, RMSE, MAE, R², IC, RankIC) and, per head, the per-round `train-`/`val-` curves and feature importance (`xgb`), the best round (`xgb_td`) or the stopping epoch (`realmlp`).
- **Backtest**: one run in the `USEquityCrossectionSelectStockVectorBt_backtest` project, named after the run directory: the backtest config with data fingerprints, the whole / in-sample / out-of-sample metrics as summary values (plus `benchmark/...` and `relative/...` when a benchmark ran), and the HTML report.

## Benchmark comparison

The benchmark is the ETF's own daily rows from CRSP (`crsp_a_stock.dsf_v2`, selected by PERMNO), converted like any CRSP panel: `adjOpen`/`adjClose` are total-return adjusted, so the buy-and-hold includes the ETF's dividends (net of its expense ratio, like a real holding). It is the tradable ETF, not the index level.

With a benchmark (the default), the backtest also buys and holds the ETF from the same `init_cash`, with the same fees, slippage and next-bar-open fills, so the two curves compare bar for bar. Each ETF lives in its own single-symbol store (`wrds_crsp_spy_1d.zarr`, `wrds_crsp_qqq_1d.zarr`), never in the equity panel, where it would be ranked against its own constituents. `metrics.json` gains two blocks, each split whole / in-sample / out-of-sample:

- `benchmark`: the ETF's own return statistics.
- `relative`: the portfolio against the ETF: `excess_return` (relative NAV − 1), `excess_return_annualized`, `excess_max_drawdown`, `tracking_error`, `information_ratio`, `beta`, `correlation`, `capm_alpha`, `win_rate_vs_benchmark`.

`report.html` draws the benchmark NAV beside the portfolio's and adds excess-return and excess-drawdown rows; with `use_cv=True` the stitched curve and every fold are compared. The pipeline log line prints the headline numbers. Set `BacktestSettings.benchmark=None` to skip the comparison.

A backtest run directory can be rebuilt and re-run with `quantlab.utils.module.load_backtester_from_config`; see [docs/backtest.md](../../docs/backtest.md).

## How the universe is handled

- **Survivorship**: the CRSP roster contains every PERMNO that was a member at any time in the window, delisted ones included, and CRSP carries delisting returns.
- **Point-in-time membership**: `members.zarr` is the price panel with non-member cells set to NaN. The label reads it, so training rows are member rows only. The backtest prices from it, so only current members can be bought, and a holding that leaves the index is sold on the next bar. Factors read `prices.zarr`, so their rolling windows see full history.
- **Symbol padding**: the symbol axis is padded with all-NaN PERMNOs (-1, -2, ...) to a multiple of 16, because KunQuant batch runs need a multiple of the SIMD block width. Padded columns never have a label or a price, so they are never trained on or traded.

## Notes

- Alpha101/Alpha158 outputs are raw (not normalized). Tree models do not need normalization and RealMLP robust-scales its inputs itself, but missing features are filled with 0 by the pytabkit heads (`xgb_td`, `realmlp`); the plain `xgb` head keeps NaN as missing.
- A factor subset keeps the models small while experimenting. Pinning `alpha101_names`/`alpha158_names` needs `BaseModel.get_factor_names` to honor the configured `factor_names`.
- Memory grows with symbols × days × features: 1,000 PERMNOs over 13 years with all 251 features is roughly 3 GB in float32.
