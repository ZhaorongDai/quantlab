# Examples

Each script in this directory is a complete, runnable program. They all work offline: they
generate synthetic prices in a temporary directory, need no credentials and no GPU, and finish
in well under a minute on a laptop CPU. Run any of them from the repository root with `uv`:

```bash
uv run python examples/quickstart.py
```

| Script | What it shows | Guide |
|--------|---------------|-------|
| [`quickstart.py`](quickstart.py) | The whole pipeline: a price panel, factors and a label, an XGBoost model, a backtest, and rebuilding the run from its `config.json` | [Quickstart](../docs/getting-started/quickstart.md) |
| [`inspect_data_sources.py`](inspect_data_sources.py) | The data-source registry, credential checks, a resumable and cancellable download with a stand-in vendor, inspecting files on disk, and the volume check | [Data sources](../docs/user-guide/data-sources.md) |
| [`build_panel.py`](build_panel.py) | Converting raw files into a panel, requesting a date range and symbols, the storage backends, chunked conversion and updates, and masking a panel with a point-in-time universe | [Datasets](../docs/user-guide/datasets.md), [Universes](../docs/user-guide/universes.md) |
| [`train_model.py`](train_model.py) | A custom factor, a forward-return label, training and evaluating an XGBoost model, reloading it from its checkpoint, and walk-forward cross-validation | [Factors](../docs/user-guide/factors.md), [Models](../docs/user-guide/models.md) |
| [`backtest.py`](backtest.py) | Long-only and long/short backtests, a delisted holding, rebuilding a run, and backtesting cross-validation folds as one stitched curve | [Backtesting](../docs/user-guide/backtesting.md) |

## Real data

| Script | What it shows | Guide |
|--------|---------------|-------|
| [`wrds_us_equity/`](wrds_us_equity/) | The full pipeline on CRSP daily data, one self-contained script per universe (S&P 500, Nasdaq-100, the whole CRSP market) and model (`xgb`, `xgb_td`, `realmlp`): Alpha101 + Alpha158 factors, a forward-return label, the model, a TopN backtest against an ETF benchmark, Weights & Biases logging; plus one factor-analysis script per universe running `Factor.analyze()` on every alpha column, and `market_residual_momentum.py`, the Fama-French residual-momentum factor analyzed on the whole market. Needs a WRDS account and a converted CRSP store; settings are a few constants and the quantlab config objects at the top of each script | [README](wrds_us_equity/README.md), [WRDS](../docs/wrds_crsp.md) |
| [`sharadar_us_equity/`](sharadar_us_equity/) | `sp500_xgb.py`, the WRDS S&P 500 XGBoost pipeline with every input from Sharadar: prices and the SPY benchmark on the permaticker axis, membership from Sharadar's SP500 table. Needs a Sharadar subscription and the stores of `scripts/sharadar/download.py`; the tracker is offline, because the data is licensed for personal use | [README](sharadar_us_equity/README.md), [Sharadar](../docs/sharadar.md) |

The prices are random walks, sometimes with a small planted effect so the model has something
to find. The numbers the scripts print show what the output looks like; they say nothing about
real markets.

The offline examples track nothing: no config names a tracker, so the default null tracker is used. The
WRDS model pipelines name a `WandbTracker` in their model and backtest configs (`TRACKER` at the top of each
file). On macOS the examples also set `OMP_NUM_THREADS=1`, because PyTorch and XGBoost ship
conflicting OpenMP runtimes.
