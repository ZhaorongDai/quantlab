# quantlab

An end-to-end, config-driven quantitative research backend: from multi-market, multi-frequency
market data, through factor computation and return prediction, to portfolio optimization,
target-position generation, and backtesting. The current milestone targets a backend-only
research pipeline; the layered architecture is designed to leave room for future
platformization (services, multiple users, online factor/model editing, a web frontend)
without requiring a rewrite, though none of that is implemented yet.

## Core Value

A single, config-driven, reproducible quant pipeline (data -> factors -> return model ->
portfolio optimization -> target positions -> backtest -> results), where each stage
communicates through a clear input/output contract so it can be replaced or extended
independently.

## Data Format

`xarray.Dataset` (dims `[timestamp, symbol]`), persisted to disk via Zarr, is the canonical
in-memory and on-disk representation used between pipeline layers (dataset -> factor/label ->
model). Model training consumes `xarray` data directly rather than converting through a
`pandas.DataFrame`.

## Project Structure

- `base/` -- Abstract base classes that define the layer contracts: `DataBackend`/`ModelBackend`
  (`backend.py`), `Dataset` (`data.py`), `FactorKunQuant` (`factor.py`), `BaseModel`
  (`model.py`), plus the dataclass configs (`config.py`: `DatasetConfig`, `FactorConfig`,
  `DLConfig`, `MLConfig`).
- `dataset/` -- Concrete `Dataset`/`DataBackend` implementations: `SpotKlineDataset` (Binance
  spot klines), `StockDataset` (NASDAQ/Tiingo, partially implemented), `XrBackend`
  (Zarr-backed), `PlBackend` (Parquet/Polars-backed).
- `factor/` -- Concrete factor sets computed via KunQuant: `Alpha101SpotKline`, `Alpha101Stock`,
  `Alpha158SpotKline`.
- `label/` -- Forward-return prediction targets: `SpotReturn` (regression), `SpotBinaryReturn`
  (classification).
- `my_ops/` -- Custom KunQuant composite ops used inside factor/label graphs (e.g.
  `WindowedZScore`, `WindowedRobustStandardization`).
- `dl_model/` -- Concrete PyTorch model heads trained through `base/model.py:BaseModel`:
  `MLPRegressor`, `RNNRegressor`, `RNNClassifier`.
- `ml_model/` -- `joblib`-based persistence helper (`MlBackend`) for non-torch models; no
  concrete `MLConfig`-driven model is implemented yet.
- `backtest/` -- Nautilus Trader live/backtest `Strategy` (`test_strategy.py`) that loads a
  trained model checkpoint and generates/submits orders from live bars.
- `vecbt/` -- vectorbt-based signal backtest helper (`backtest_from_signals`).
- `config/` -- Config factory functions (`__init__.py`) that build `DatasetConfig`/
  `FactorConfig` instances, and `instruments.yaml` (exchange instrument metadata).
- `enums/` -- Shared constants and enums used across layers.
- `utils/` -- Cross-cutting helpers: timing (`timer.py`), file I/O (`file.py`), Binance REST
  calls (`binance.py`), Nautilus Trader conversions (`nautilus.py`), dynamic
  import-by-dotted-path (`module.py`), dataclass serialization (`asdict.py`).
- `scripts/` -- One-off/exploratory scripts, not part of the core architecture:
  `download_stock_data_from_tiingo.py` (Tiingo downloader).

## Entry Points

There is no single unified CLI -- each script independently builds its own configs and
imports the layers it needs:

- `cal.py` -- computes and saves Alpha101 factors for a fixed config.
- `train_model.py` -- builds an `RNNClassifier` (`DLConfig`), collects data, trains or loads a
  checkpoint, generates predictions over a date range, and runs a vectorbt backtest.
- `test.py` -- interactive smoke test of `StockDataset`/`Alpha101Stock` against local NASDAQ
  parquet data (meant to be run cell-by-cell, e.g. in VS Code/Jupyter).
- `get_binance_instruments.py` -- CLI to refresh `config/instruments.yaml` from the live
  Binance API.
- `read_mock_data_sink.py` -- memory-profiling scratch script for reading a parquet hive
  dataset.
- `scripts/download_stock_data_from_tiingo.py` -- parallel Tiingo downloader for NASDAQ
  tickers.

## Installation

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/) (Python >=3.13):

```bash
uv sync
```

This creates a `.venv` and installs the pinned dependencies from `uv.lock`. GPU/CUDA is
expected for deep-learning model training (`base/model.py` selects `cuda` when available,
falling back to `cpu`).

## Environment Variables

- `TIINGO_API_KEY` -- required to run `scripts/download_stock_data_from_tiingo.py`. Never
  hardcode this key; the script reads it from the environment and raises if it is unset.
- `WANDB_API_KEY` -- required for Weights & Biases experiment tracking during model training
  (`base/model.py:_init_wandb`).
- `QUANTLAB_DATA_DIR` -- optional. Overrides the default data root used by `config/__init__.py`'s
  factory functions (raw downloads, Zarr factor/label stores, Nautilus catalog). Defaults to a
  `data/` directory at the repo root if unset.

## Configuration

Config objects are dataclasses defined in `base/config.py`, threaded through every layer's
constructor:

- `DatasetConfig` -- raw data paths, Zarr/catalog paths, date range, symbols.
- `FactorConfig` -- factor computation window, mode (`batch`/`stream`), symbols, data columns,
  embeds a `Dataset`.
- `DLConfig` -- deep-learning training config (factors, labels, model hyperparameters).
- `MLConfig` -- non-torch model config (persistence via `ml_model/backend.py`; no concrete
  model implementation yet).

`config/__init__.py` provides factory functions (`spot_kline_config`, `alpha101_config`,
`alpha158_config`, `spot_label_config`) that build these configs using paths derived from
`QUANTLAB_DATA_DIR` (or the repo-root `data/` default), so a fresh clone works without manual
path edits.
