# Sharadar US-equity pipeline: Sharadar daily -> Alpha101/Alpha158 -> XGBoost -> backtest

`sp500_xgb.py` is the pipeline of [`wrds_us_equity/sp500_xgb.py`](../wrds_us_equity/sp500_xgb.py) with every input read from a Sharadar store ([docs/sharadar.md](../../docs/sharadar.md)): the point-in-time S&P 500 from Sharadar's SP500 table, prices on the permaticker axis, and SPY from the fund prices as the benchmark. It reads no WRDS store and imports only quantlab, so it can be copied out and edited on its own.

The steps, settings and outputs are those of the WRDS script; see [its README](../wrds_us_equity/README.md). What differs:

| | WRDS (`wrds_us_equity/sp500_xgb.py`) | Sharadar (`sharadar_us_equity/sp500_xgb.py`) |
| --- | --- | --- |
| Symbol axis | CRSP PERMNO | Sharadar permaticker |
| Price store | `wrds_crsp_sp500_1d.zarr` (`CrspStockDataset`) | `sharadar_sp500_1d.zarr` (`SharadarStockDataset`, `roster_universe="sp500"`) |
| Membership | `wrds_crsp_sp500_membership.zarr` | `sharadar_sp500_membership.zarr` (`SharadarSP500ConstituentDataset`) |
| Benchmark | `wrds_crsp_spy_1d.zarr` | `sharadar_spy_1d.zarr` (`SPY_PERMATICKER`) |
| Delistings | CRSP's delisting return is on the delisting day's row | settled at the last close; Sharadar has no delisting return |
| Tracker | `WandbTracker(mode="online")` | `WandbTracker(mode="offline")` |
| Outputs | `<data root>/data/pipeline/wrds_sp500/` | `<data root>/pipeline/sharadar_sp500/` |

## Prerequisites

Download every table and build the stores once (this needs a Sharadar subscription; see [docs/sharadar.md](../../docs/sharadar.md)):

```bash
export SHARADAR_API_KEY=<your-sharadar-key>
uv run python scripts/sharadar/download.py \
    --download-dir /data/quantlab/downloads --zarr-dir /data/quantlab/zarrs
```

and point the script at that root: `QUANTLAB_DATA_DIR=/data/quantlab`, or set `DATA_ROOT` at the top of the file. The script reads `<data root>/zarrs/` and writes under `<data root>/pipeline/sharadar_sp500/`. `scripts/sharadar/update.py` keeps the stores current.

The data is licensed for personal use: the data root must be outside the repository (the script refuses one inside it), and runs stay off public trackers. The tracker is offline (runs are written to `wandb/` and never uploaded); set it to `"disabled"` to write nothing.

## Run

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/sp500_xgb.py
```

or run the `# %%` cells one at a time. Every step is a function (`prepare_stores`, `compute_factors`, `train`, `backtest`).
