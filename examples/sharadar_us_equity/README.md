# Sharadar US-equity pipeline: Sharadar daily -> Alpha101/Alpha158 -> XGBoost -> backtest

`sp500_xgb.py` is the pipeline of [`wrds_us_equity/sp500_xgb.py`](../wrds_us_equity/sp500_xgb.py) with every input read from a Sharadar store ([docs/sharadar.md](../../docs/sharadar.md)): the point-in-time S&P 500 from Sharadar's SP500 table, prices on the permaticker axis, and SPY from the fund prices as the benchmark. It reads no WRDS store and imports only quantlab, so it can be copied out and edited on its own.

The steps, settings and outputs are those of the WRDS script; see [its README](../wrds_us_equity/README.md). As there, the label is computed on the unmasked `prices.zarr` and wrapped in `MembershipMaskedLabel`, so a sample exists where the stock is an S&P 500 member at t, whatever its membership later. What differs:

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

## Barra style exposures

`barra_style.py` builds `BarraStyle`, the Barra USE4-style exposures (12 styles, 20 descriptors, the industry code and the estimation-universe mask), for every Sharadar common stock over 2001-2026, prints each style's coverage of the estimation universe and writes a factor report of the styles against the 21-bar forward return. It reads the SEP, DAILY, SF1 ART, fiscal-year history, industry and share-class stores that `scripts/sharadar/download.py` builds, and downloads FRED's 3-month T-bill rate (no key) itself. Everything goes under `<data root>/pipeline/sharadar_barra/`. See [Style factors](../../docs/developer-guide/style-factors.md).

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/barra_style.py
```

The full history needs a large machine: on the training server the build takes about 6 minutes and the whole script 10 minutes, with a peak of about 290 GB of memory. Narrow `START` for a smaller one.

## Factor risk model and bias statistics

`risk_model.py` builds `Use4RiskModel` on those exposures: the regression store (each bar's factor returns for the country, the Fama-French 48 industries and the 12 styles, and every symbol's specific return) from 2001, then the estimate store (factor covariance and specific risk: exponentially weighted with USE4S half-lives, Newey-West on the factors) from 2007, once the 1512-bar correlation window fits. It then prints the bias statistics (`quantlab.risk.bias.risk_model_bias_statistics`) of every pure factor, every symbol's specific risk and 100 random active portfolios, over one-bar returns and over non-overlapping 21-bar returns (USE4 tests monthly; Newey-West adjusts for the latter), writes them to `bias_summary.json` and plots the rolling one-year mean, 5th/95th percentile and MRAD in `bias_h1.png` and `bias_h21.png`. It reads the SEP, DAILY and FRED stores and the exposures of `barra_style.py`; everything goes under `<data root>/pipeline/sharadar_risk/`.

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/risk_model.py
```

On the training server the whole script takes about 10 minutes (the estimate store in 32 processes, about 2 minutes), with a peak of about 25 GB of memory.
