# WRDS US-equity pipeline: CRSP daily -> Alpha101/Alpha158 -> model -> backtest

English | [简体中文](README.zh-CN.md)

`pipeline.py` runs the whole research loop on CRSP daily data for the point-in-time S&P 500:

1. **Data**: read the converted CRSP store and its membership panel, then write two derived stores (`prices`, `members`).
2. **Factors**: `Alpha101Stock` and `Alpha158Stock` on adjusted prices, saved as Zarr.
3. **Label**: `Return`, the open-to-open return from t+1 to t+1+`horizon`, computed on member rows only.
4. **Model**: `xgb` (`XGBoostRegressor`), `xgb_td` (`XGBTDRegressor`) or `realmlp` (`RealMLPRegressor`), trained once or walk-forward.
5. **Backtest**: `USEquityCrossectionSelectStockVectorBt`, a TopN cross-sectional portfolio over the out-of-sample window.

There is no command-line interface. Every setting is a field of the `Settings` dataclass at the top of `pipeline.py`.

## Prerequisites

Download and convert CRSP's S&P 500 roster once (this needs a WRDS account; see [docs/wrds_crsp.md](../../docs/wrds_crsp.md)):

```bash
export WRDS_USERNAME=<your-wrds-username>   # password in ~/.pgpass
uv run python scripts/ingest_wrds_crsp.py --universe crsp_sp500 \
    --start-date 2010-01-01 --end-date 2024-12-31 --to-zarr
```

This writes `data/data/us_equity/1d/wrds_crsp_sp500_1d.zarr` (prices of every PERMNO that was a member at some point in the window) and `wrds_crsp_sp500_membership.zarr` (`is_member` per day). The pipeline reads both from the same data root (`QUANTLAB_DATA_DIR`, or `data/` beside the repository, or `Settings.data_root`).

KunQuant compiles the factor graphs, so a C++ compiler is required. On macOS set `OMP_NUM_THREADS=1` (xgboost and torch in one process); set `WANDB_MODE=disabled` or `offline` unless you are logged in to Weights & Biases.

## Run

Edit `Settings` (at least `model`, the dates and the hyperparameters), then:

```bash
WANDB_MODE=disabled uv run python examples/wrds_us_equity/pipeline.py
```

or run the `# %%` cells one at a time in VS Code or Jupyter. Every stage is also a function (`prepare_stores`, `compute_factors`, `train`, `backtest`), so a notebook can rerun only the step it changed:

```python
import pipeline as p
s = p.Settings(model="realmlp", use_cv=True)
p.main(s)
```

## Settings

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | `"xgb"` | `"xgb"`, `"xgb_td"` or `"realmlp"` |
| `hyperparameters` | `{}` | merged over `DEFAULT_HYPERPARAMETERS[model]`; the keys are the head's own (`xgb.train` parameters, or the pytabkit constructor arguments) |
| `early_stopping`, `early_stopping_patience`, `val_size` | `True`, `50`, `0.2` | early stopping on the trailing `val_size` of the training window; patience is in boosting rounds (xgb, xgb_td) or epochs (realmlp) |
| `start_date`, `end_date` | 2012-01-01, 2024-12-31 | data window; factor warm-up is read before it |
| `train_start` ... `test_end` | 2012-2019 / 2020-2024 | training and out-of-sample test windows |
| `use_cv`, `cv_train_periods`, `cv_gap_periods` | `False`, 1250, 5 | walk-forward folds (`train_cv`), backtested as one stitched out-of-sample curve (`run_cv`) |
| `factor_window` | 400 | factor lookback in calendar days |
| `alpha101_names`, `alpha158_names` | `None` | subsets of each library; `None` means all 82 / 169 columns |
| `horizon` | 5 | label horizon in bars |
| `rebalance_periods`, `top_n`, `direction` | 5, 50, `"long_only"` | rebalance every 5 bars into the top 50 scores; `"long_short"` also shorts the bottom 50 |
| `fees`, `slippage`, `init_cash` | 0.0005, 0.0005, 1e6 | proportional costs and starting capital |

## Outputs

Everything is written under `<data root>/data/pipeline/wrds_sp500/`:

```text
prices.zarr, members.zarr     derived price stores (step 1)
factor/alpha101.zarr, factor/alpha158.zarr, label/ret_<h>.zarr
models/<model>/...            checkpoints, config.json, cv_folds.json
backtests/<model>/...         weights, equity, metrics.json, report.html
```

A backtest run directory can be rebuilt and re-run with `quantlab.utils.module.load_backtester_from_config`; see [docs/backtest.md](../../docs/backtest.md).

## How the universe is handled

- **Survivorship**: the CRSP roster contains every PERMNO that was a member at any time in the window, delisted ones included, and CRSP carries delisting returns.
- **Point-in-time membership**: `members.zarr` is the price panel with non-member cells set to NaN. The label reads it, so training rows are member rows only. The backtest prices from it, so only current members can be bought, and a holding that leaves the index is sold on the next bar. Factors read `prices.zarr`, so their rolling windows see full history.
- **Symbol padding**: the symbol axis is padded with all-NaN PERMNOs (-1, -2, ...) to a multiple of 16, because KunQuant batch runs need a multiple of the SIMD block width. Padded columns never have a label or a price, so they are never trained on or traded.

## Notes

- Alpha101/Alpha158 outputs are raw (not normalized). Tree models do not need normalization and RealMLP robust-scales its inputs itself, but missing features are filled with 0 by the pytabkit heads (`xgb_td`, `realmlp`); the plain `xgb` head keeps NaN as missing.
- A factor subset keeps the models small while experimenting. Pinning `alpha101_names`/`alpha158_names` needs `BaseModel.get_factor_names` to honor the configured `factor_names`.
- Memory grows with symbols × days × features: 1,000 PERMNOs over 13 years with all 251 features is roughly 3 GB in float32.
