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

## Factor Backends

Factor computation has **two interchangeable backends**, both subclasses of the single
abstract `Factor` base in `base/factor.py`:

| Backend | Class | Config | Modes | Write factors as |
|---------|-------|--------|-------|------------------|
| KunQuant | `base/factor.py:FactorKunQuant` | `FactorConfig` | batch **and** streaming | a compiled `KunQuant.Stage.Function` graph |
| Polars | `base/factor_polars.py:FactorPolars` | `PolarsFactorConfig` | batch only (by decision) | a `polars` lazy-expression chain |

They are siblings, not parent and child, and nothing downstream can tell which one produced
a given factor store. A factor object from either backend drops into `DLConfig.factors` /
`MLConfig.factors` with zero changes to `base/model.py` — the model layer only ever calls the
shared contract (`cal()` / `read()` / `get_features()` / `_get_factor_names()` /
`get_config()` / `_reset_dataset_config()`) and never branches on a factor's runtime type.
`tests/test_factor_hierarchy.py` locks that property, including a live test that drives one
KunQuant factor and one Polars factor through the same loop and merges their outputs.

**`xarray.Dataset` is the only exchange format at the factor layer's boundary, regardless of
backend.** Polars is an internal implementation detail of one backend: `FactorPolars.cal()`
converts to `xr.Dataset` before anything leaves the class, exactly as the KunQuant backend
does with its raw output arrays. No public factor method accepts or returns a bare
`pandas`/`polars` DataFrame.

**Prefer the KunQuant backend.** Use Polars for a new factor only when the logic is awkward
to express as a KunQuant graph — and never when `xarray`/KunQuant can already do the job.

### Adding a KunQuant factor

1. Subclass `FactorKunQuant` (see `factor/alpha158.py`, the dual-market example: the same
   factor set is exposed as `Alpha158SpotKline` for crypto spot and `Alpha158Stock` for US
   equities).
2. Implement `_get_factor_func()`, returning the `KunQuant.Stage.Function` built from
   `Input(...)`/`Output(...)` nodes, and `_get_factor_names()`, returning the factor names the
   graph emits.
3. Add a config factory in `config/__init__.py` that builds a `FactorConfig` with paths
   derived from `_data_root()` — never a hardcoded absolute path.

The inherited `cal()` compiles and runs the graph in batch mode; `init_stream()` /
`cal_stream()` drive the same graph incrementally, one bar at a time, for live data.

Note on streaming: `init_stream()` binds a buffer handle for every name in
`config.data_columns` *and* every name in `config.factor_names`. KunQuant prunes declared
inputs that no selected output consumes, so a `data_columns` list wider than the chosen
factor subset actually needs raises `RuntimeError: Cannot find the buffer name`. Full factor
sets consume every input and are unaffected.

### Adding a Polars factor

1. Subclass `FactorPolars` (see `factor/momentum.py`, the worked example — an N-day per-symbol
   momentum signal in about a dozen lines).
2. Implement the single hook `_get_factor_lazyframe(lf) -> pl.LazyFrame`. It receives the
   dataset's already-read lazyframe and must return a lazyframe carrying **only** `timestamp`,
   `symbol` and the computed factor column(s) — whatever non-index columns come back *are* the
   factors, and are persisted as such. Nothing in the hook may materialize (no `.collect()`);
   `cal()` is what triggers computation.
3. Add a config factory in `config/__init__.py` that builds a `PolarsFactorConfig`.

There is no `_get_factor_names()` to write: names are read from the computed frame's own
schema inside `cal()`. There is likewise no streaming counterpart — the Polars backend is
batch-only by design, and adding a dormant streaming surface to it would be an unused member
rather than an extension point.

Because `Dataset.get_lazyframe()` performs no per-market column normalization (unlike
`_to_kunquant()`, which each `Dataset` subclass overrides to rename its columns), a Polars
factor is written against one market's raw column names — `factor/momentum.py` targets the
crypto-spot store's Title-Case `Close`.

## Project Structure

- `base/` -- Abstract base classes that define the layer contracts: `DataBackend`/`ModelBackend`
  (`backend.py`), `Dataset` (`data.py`), the shared `Factor` base and its `FactorKunQuant`
  backend (`factor.py`), the `FactorPolars` backend (`factor_polars.py`), `BaseModel`
  (`model.py`), plus the dataclass configs (`config.py`: `DatasetConfig`, `BaseFactorConfig`,
  `FactorConfig`, `PolarsFactorConfig`, `DLConfig`, `MLConfig`).
- `dataset/` -- Concrete `Dataset`/`DataBackend` implementations: `SpotKlineDataset` (Binance
  spot klines), `StockDataset` (NASDAQ/Tiingo, partially implemented), `XrBackend`
  (Zarr-backed), `PlBackend` (Parquet/Polars-backed).
- `factor/` -- Concrete factor sets. Computed via KunQuant: `Alpha101SpotKline`,
  `Alpha101Stock`, `Alpha158SpotKline`, `Alpha158Stock`. Computed via Polars: `Momentum`
  (`momentum.py`, the worked example of the Polars backend). See
  [Factor Backends](#factor-backends).
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
- `ingest_binance_spot.py` -- rebuilds the Binance spot-kline Zarr store from locally-dropped
  monthly CSVs via `SpotKlineDataset`/`spot_kline_config()`. Does not download anything (Binance
  keeps its manual-CSV-drop workflow). Use `--raw-data-dir` to point at CSVs stored outside the
  default `data/{market}/{frequency}/...` convention path (e.g. a pre-existing download
  directory) with no filesystem migration required.
- `ingest_tiingo.py` -- full Tiingo-to-Zarr pipeline for US equities: fetches raw EOD data via
  `TiingoAcquisition`, then converts/cleans/persists it through `StockDataset` into a Zarr
  store. Requires `TIINGO_API_KEY`. Pass `--refresh` to incrementally update from each symbol's
  last recorded watermark instead of a full backfill.
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

- `TIINGO_API_KEY` -- required to run `ingest_tiingo.py` (the current, documented entry point)
  and `scripts/download_stock_data_from_tiingo.py`. Never hardcode this key; both read it from
  the environment and raise if it is unset.
- `WANDB_API_KEY` -- required for Weights & Biases experiment tracking during model training
  (`base/model.py:_init_wandb`).
- `QUANTLAB_DATA_DIR` -- optional. Overrides the default data root used by `config/__init__.py`'s
  factory functions (raw downloads, Zarr factor/label stores, Nautilus catalog). Defaults to a
  `data/` directory at the repo root if unset.

## Configuration

Config objects are dataclasses defined in `base/config.py`, threaded through every layer's
constructor:

- `DatasetConfig` -- raw data paths, Zarr/catalog paths, date range, symbols.
- `BaseFactorConfig` -- the fields both factor backends share: window, symbols, date range,
  output path; embeds a `Dataset`.
- `FactorConfig` -- `BaseFactorConfig` plus the KunQuant-only fields: mode (`batch`/`stream`),
  data columns, executor thread count.
- `PolarsFactorConfig` -- `BaseFactorConfig` with nothing added; the Polars backend is
  batch-only, so it deliberately has no `mode`.
- `DLConfig` -- deep-learning training config (factors, labels, model hyperparameters).
- `MLConfig` -- non-torch model config (persistence via `ml_model/backend.py`; no concrete
  model implementation yet).

`config/__init__.py` provides factory functions (`spot_kline_config`, `alpha101_config`,
`alpha158_config`, `spot_label_config`) that build these configs using paths derived from
`QUANTLAB_DATA_DIR` (or the repo-root `data/` default), so a fresh clone works without manual
path edits.
