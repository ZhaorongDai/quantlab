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
| [`build_panel.py`](build_panel.py) | Converting raw files into a panel, filtering by date and symbol, the storage backends, chunked conversion and updates, and masking a panel with a point-in-time universe | [Datasets](../docs/user-guide/datasets.md), [Universes](../docs/user-guide/universes.md) |
| [`train_model.py`](train_model.py) | A custom factor, a forward-return label, training and evaluating an XGBoost model, reloading it from its checkpoint, and walk-forward cross-validation | [Factors](../docs/user-guide/factors.md), [Models](../docs/user-guide/models.md) |
| [`backtest.py`](backtest.py) | Long-only and long/short backtests, a delisted holding, rebuilding a run, and backtesting cross-validation folds as one stitched curve | [Backtesting](../docs/user-guide/backtesting.md) |

## Real data

| Script | What it shows | Guide |
|--------|---------------|-------|
| [`wrds_us_equity/`](wrds_us_equity/) | The full pipeline on CRSP daily data, one self-contained script per universe (S&P 500, Nasdaq-100) and model (`xgb`, `xgb_td`, `realmlp`): Alpha101 + Alpha158 factors, a forward-return label, the model, a TopN backtest against an ETF benchmark, Weights & Biases logging; plus one factor-analysis script per universe running `Factor.analyze()` on every alpha column. Needs a WRDS account and a converted CRSP store; settings are a few constants and the quantlab config objects at the top of each script | [README](wrds_us_equity/README.md), [WRDS](../docs/wrds_crsp.md) |

The prices are random walks, sometimes with a small planted effect so the model has something
to find. The numbers the scripts print show what the output looks like; they say nothing about
real markets.

The offline examples switch Weights & Biases off by setting `WANDB_MODE=disabled` before anything is
imported; the WRDS model pipelines log to it by default (`wandb_mode` in their settings). On macOS they also set `OMP_NUM_THREADS=1`, because PyTorch and XGBoost ship
conflicting OpenMP runtimes.
