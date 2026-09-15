<!-- refreshed: 2026-09-04 -->
# Architecture

**Analysis Date:** 2026-09-04

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          Ad Hoc Entry Scripts                        │
│   `main.py` (stub) · `cal.py` · `train_model.py` · `test.py`         │
│   `read_mock_data_sink.py` · `get_binance_instruments.py`            │
│   `scripts/download_stock_data_from_tiingo.py` · `test_nt.ipynb`     │
└───────────────┬────────────────────────────────┬─────────────────────┘
                │                                │
                ▼                                ▼
┌───────────────────────────────┐   ┌───────────────────────────────────┐
│      Model Layer               │   │      Backtest Layer                │
│  `dl_model/`, `ml_model/`      │   │  `quantlab/backtest/` (vectorbt      │
│  subclasses of                 │   │   engine + TopN selection) on        │
│  `base/model.py:BaseModel`     │   │  `base/backtest.py:BaseBacktester`   │
└───────────────┬─────────────────┘   └───────────────┬─────────────────┘
                │  reads factors+labels                │ reads model + price
                ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Factor / Label Layer                              │
│   `factor/alpha101.py`, `factor/alpha158.py` (features)              │
│   `label/spot.py` (forward-return labels)                            │
│   built on `base/factor.py:FactorKunQuant` (KunQuant compiled graphs) │
│   custom ops in `my_ops/preprocess.py`                                │
└───────────────┬────────────────────────────────────────────────────┘
                │  reads xarray.Dataset from
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Dataset Layer                                 │
│  `dataset/spot.py:SpotKlineDataset` (Binance CSV, nautilus bars)     │
│  `dataset/stock.py:StockDataset` (Tiingo/NASDAQ parquet)             │
│  built on `base/data.py:Dataset` (abstract)                           │
└───────────────┬────────────────────────────────────────────────────┘
                │  persists via
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 Storage Backend Layer                                │
│  `dataset/backend.py:XrBackend` (zarr, via xarray)                   │
│  `dataset/backend.py:PlBackend` (parquet, via polars)                │
│  built on `base/backend.py:DataBackend` (abstract)                    │
└─────────────────────────────────────────────────────────────────────┘

Cross-cutting: `base/config.py` (dataclass configs consumed by every layer)
               `enums/`, `config/` (constants + instrument metadata)
               `utils/` (timer, file globbing, binance/nautilus helpers)
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `DataBackend` (abstract) | Defines read/write/filter contract for any storage medium | `base/backend.py` |
| `XrBackend` | Zarr-backed storage for `xarray.Dataset` (canonical `[timestamp, symbol]` shape) | `dataset/backend.py` |
| `PlBackend` | Parquet-backed storage via `polars.LazyFrame` | `dataset/backend.py` |
| `Dataset` (abstract) | Loads raw market data into the internal xarray representation; converts to KunQuant/Nautilus formats | `base/data.py` |
| `SpotKlineDataset` | Binance spot kline CSV ingestion, conversion to Nautilus `Bar` objects | `dataset/spot.py` |
| `StockDataset` | NASDAQ/Tiingo parquet ingestion | `dataset/stock.py` |
| `FactorKunQuant` (abstract) | Compiles and executes KunQuant factor graphs (batch and streaming modes) | `base/factor.py` |
| `Alpha101SpotKline` / `Alpha158SpotKline` | Concrete factor sets (Alpha101 formulaic factors, Alpha158 factor library) | `factor/alpha101.py`, `factor/alpha158.py` |
| `SpotReturn` / `SpotBinaryReturn` | Forward-return regression/classification labels | `label/spot.py` |
| `WindowedZScore` / `WindowedRobustStandardization` | Custom KunQuant composite ops for factor normalization | `my_ops/preprocess.py` |
| `BaseModel` (abstract) | Shared training loop: data collection, train/val/test split, epoch loop, checkpointing, W&B logging, CV | `base/model.py` |
| `MLPRegressor`, `RNNRegressor`, `RNNClassifier` | Concrete torch model heads (MLP, GRU/LSTM regressor, GRU/LSTM classifier with auxiliary-label architecture) | `dl_model/mlp.py`, `dl_model/rnn.py`, `dl_model/rnn_classification.py` |
| `MlBackend` | joblib-based persistence for non-torch (ML) models | `ml_model/backend.py` |
| `Test` (Strategy) | (file deleted 2026-09-07) Nautilus Trader live/backtest strategy: loads a trained model, generates predictions on live bars, sizes and submits orders (with TWAP execution) | `backtest/test_strategy.py` |
| `BaseBacktester` (abstract) | Backtester ABC: config type guard, the two public entry points `run()` (model backtest) and `run_cv()` (replays a `train_cv` run's `cv_folds.json` as one stitched curve), bar-counted warm-up, D-17 in/out-of-sample split, metrics, run directory, data fingerprints; engine hooks `_simulate`, `_simulate_benchmark`, `_engine_stats`, `_period_returns_stats` | `quantlab/base/backtest.py` |
| `VectorBtBacktester` (abstract) | vectorbt engine: `Portfolio.from_orders` with target-percent weights, a signal at bar t fills at bar t+1's open (D-05), forced-liquidation records for delisted holdings; the only quantlab module importing vectorbt | `quantlab/backtest/engine_vectorbt.py` |
| `CrossSectionTopNSelector` | TopN long-only or disjoint long/short target weights on rebalance bars (D-03 weights contract) | `quantlab/backtest/selection.py` |
| `USEquityCrossectionSelectStockVectorBt` | Concrete US-equity backtester composing `US_EQUITY_MARKET`, the selector and the vectorbt engine; rebuilt from a run's `config.json` by `quantlab/utils/module.py:load_backtester_from_config`. It replaces the `vecbt/bt.py:backtest_from_signals` helper, retired in phase 03.7 (D-31) | `quantlab/backtest/us_equity.py` |
| Config factories | Hardcoded-path factory functions producing `DatasetConfig`/`FactorConfig` for spot klines, alpha101, alpha158, labels | `config/__init__.py` |
| `DatasetConfig`/`FactorConfig`/`DLConfig`/`MLConfig` | Dataclass configuration objects threaded through every layer | `base/config.py` |

## Pattern Overview

**Overall:** Layered abstract-base-class pipeline for quantitative research: `DataBackend → Dataset → Factor/Label → Model → Backtest`, each layer defined by an ABC in `base/` and implemented by concrete subclasses in a sibling top-level package (`dataset/`, `factor/`, `label/`, `dl_model/`/`ml_model/`, `backtest/`). The former `vecbt/` helper package was retired in phase 03.7 (D-31).

**Key Characteristics:**
- Every domain object (`Dataset`, `FactorKunQuant`, `BaseModel`) shares the same lifecycle idiom: a `config` property setter that normalizes dates/names on assignment, a `read()`/`cal()`/`save()` trio for lazy-vs-eager data materialization, and `xarray.Dataset` as the universal in-memory exchange format between layers (always indexed by `timestamp` and `symbol`).
- Factor/label computation is offloaded to **KunQuant**, a compiled-graph engine (`Builder`/`Op`/`Function` → `cfake.compileit` → `KunRunner`), not plain numpy/pandas — factors are defined declaratively as op graphs, then JIT-compiled to native code and executed via a multi-thread executor (`kr.createMultiThreadExecutor`).
- Model layer (`base/model.py:BaseModel`) is torch-centric: training assumes a `[num_times, num_symbols, num_features]` tensor shape, uses `TensorDataset`/`DataLoader`, and every concrete model implements the same 5-method contract (`_init_model`, `_train_one_batch`, `_val_one_batch`, `_test_one_batch`, `_preprocess`).
- Configuration is dataclass-based (not env-var or YAML-based, except `config/instruments.yaml` for exchange instrument metadata) and is **hardcoded with absolute filesystem paths per developer machine** rather than parameterized (see `config/__init__.py`).
- No dependency injection framework, no plugin registry beyond `utils/module.py:get_cls_from_path` (dynamic import-by-dotted-path used to reconstruct a `Dataset`/`Factor`/`Model` from a saved JSON config).

## Layers

**Storage Backend (`dataset/backend.py`, abstract in `base/backend.py`):**
- Purpose: Abstracts "how data is persisted" from "what the data means."
- Location: `dataset/backend.py` (concrete: `XrBackend`, `PlBackend`), `base/backend.py` (abstract `DataBackend`).
- Contains: `read`/`write`/`to_internal`/`filter_by_date`/`filter_by_symbol`/`get_xarray_dataset`/`get_lazyframe`.
- Depends on: `xarray`, `polars`, `pandas`.
- Used by: `Dataset` and `FactorKunQuant`, each of which owns a `self.data_backend` instance.

**Dataset (`base/data.py`, `dataset/`):**
- Purpose: Converts raw external data (Binance CSV klines, Tiingo/NASDAQ parquet) into the canonical `xarray.Dataset` and persists it via a `DataBackend`. Also converts to KunQuant input arrays (`to_kunquant`) and Nautilus Trader bar/catalog objects (`to_nautilus`).
- Location: `base/data.py` (ABC `Dataset`), `dataset/spot.py` (`SpotKlineDataset`), `dataset/stock.py` (`StockDataset`, several methods unimplemented — raise `ValueError("Not finished")`).
- Depends on: Storage Backend layer, `nautilus_trader` model/persistence types, `utils/file.py`, `utils/nautilus.py`, `utils/timer.py`.
- Used by: Factor/Label layer (each `FactorConfig` embeds a `Dataset` instance) and directly by scripts (`test.py`, `cal.py`).

**Factor / Label (`base/factor.py`, `factor/`, `label/`, `my_ops/`):**
- Purpose: Computes engineered features (factors) and prediction targets (labels) from dataset data using compiled KunQuant graphs, in either batch mode (`cal()`, operates on a full historical window) or streaming mode (`cal_stream()`, incremental per-bar updates for live trading).
- Location: `base/factor.py` (ABC `FactorKunQuant`), `factor/alpha101.py`, `factor/alpha158.py`, `label/spot.py`, `my_ops/preprocess.py` (custom `WindowedCompositiveOp` subclasses).
- Depends on: Dataset layer (each factor config embeds a `Dataset`), `KunQuant`.
- Used by: Model layer (`DLConfig.factors`/`DLConfig.labels`), and directly by the live `Strategy` in `backtest/test_strategy.py`.

**Model (`base/model.py`, `dl_model/`, `ml_model/`):**
- Purpose: Orchestrates the full training lifecycle — pulling factor/label data into a combined `xarray.Dataset` (`collect()`), splitting into train/val/test or k-fold CV windows, running the epoch loop, early stopping, checkpointing (`.pth` via `torch.save` or `.joblib` via `joblib.dump`), and W&B logging.
- Location: `base/model.py` (ABC `BaseModel`), `base/config.py` (`DLConfig`/`MLConfig`), `dl_model/mlp.py`, `dl_model/rnn.py`, `dl_model/rnn_classification.py`, `ml_model/backend.py` (persistence helper only — no concrete `MLConfig`-based model implementation currently exists; `BaseModel._auto_train` raises `NotImplementedError` for `MLConfig`).
- Depends on: Factor/Label layer, `torch`, `wandb`, `sklearn.metrics`, `joblib`.
- Used by: Top-level scripts (`train_model.py`, `test.py`) and the live `Strategy` (`backtest/test_strategy.py`, which loads a trained `.pth` checkpoint directly into a raw `nn.Module`, bypassing `BaseModel.load()`).

**Backtest (`quantlab/backtest/`, `quantlab/base/backtest.py`):**
- Purpose: Evaluates a model's trading performance on a cross-sectional universe. `BaseBacktester.run()` is the model backtest: train mode trains the model on its own dates, load mode loads a checkpoint, and the model then predicts the window through `predict_panel`. `run_cv()` is the model-CV backtest: it reads a `train_cv` run's `cv_folds.json`, backtests each fold with its own checkpoint on its own test segment, then simulates the concatenated fold weights once as a stitched curve. Signals are D-03 target weights (a `weight` variable on `(timestamp, symbol)`), and a signal formed at bar t fills at bar t+1's open (D-05). Metrics are reported for the whole window and split in-sample/out-of-sample against the model's training window (D-17). Every run writes its own run directory (`config.json`, `weights.zarr`, `equity.zarr`, `metrics.json`, `liquidations.json`, `fingerprint.json`, `report.html`), which `quantlab/utils/module.py:load_backtester_from_config` can rebuild and re-run. The event-driven (Nautilus) path is still reserved for Phase 6, with no current implementation.
- Location: `quantlab/base/backtest.py` (`BaseBacktester`, `MarketSpec`, result dataclasses), `quantlab/backtest/engine_vectorbt.py` (`VectorBtBacktester`), `quantlab/backtest/selection.py` (`CrossSectionTopNSelector`), `quantlab/backtest/us_equity.py` (`USEquityCrossectionSelectStockVectorBt`), `quantlab/base/config.py` (`BacktestConfig`/`CrossSectionBacktestConfig`). The former Nautilus `backtest/test_strategy.py` was deleted 2026-09-07, and the former `vecbt/bt.py` helper was retired in phase 03.7 (D-31). See `example/backtest.md`.
- Depends on: Model layer (`predict_panel`, checkpoints, `cv_folds.json`), Factor/Label layer (re-dated to cover warm-up), Dataset layer (prices), `vectorbt`, `plotly`, optionally `wandb`.
- Used by: Nothing else in quantlab except the config loader in `quantlab/utils/module.py`. This is a terminal/output layer, and `tests/test_backtest_contracts.py` locks that direction.

## Data Flow

### Primary Training Path

1. A config factory builds a `FactorConfig` embedding a `Dataset` instance with hardcoded storage paths (`config/__init__.py:alpha101_config`, `alpha158_config`, `spot_label_config`).
2. `BaseModel.__init__` receives a `DLConfig` bundling `factors: list[FactorKunQuant]` and `labels: list[FactorKunQuant]` (`base/model.py:28`).
3. `BaseModel.collect()` calls `_collect_all_features()`/`_collect_all_labels()`, which each either `.cal()` (compute via KunQuant) or `.read()` (load from zarr) per `factor_data_strategy`/`label_data_strategy`, then combines all factor/label `xarray.Dataset`s via `xr.combine_by_coords` (`base/model.py:158-182`).
4. `_train_dl()` slices the combined dataset into `train`/`val`/`test` windows by timestamp, converts each to a `torch.Tensor` of shape `[time, symbol, variable]`, wraps in `DataLoader`, and runs the epoch loop calling the concrete model's `_train_one_batch`/`_val_one_batch`/`_test_one_batch`/`_preprocess` once per batch (`base/model.py:291-420`).
5. Checkpoints are written to `{model_save_dir}/{project_name}/{experiment_name}/{model_name}.pth`, plus a sibling `config.json` (`base/model.py:_save_model`).

### Factor Computation Path (KunQuant)

1. `FactorKunQuant.cal()` calls `self.config.dataset.to_kunquant(data_columns)`, which reads the dataset and reshapes each requested column into a contiguous `[time, symbol]` numpy array (`dataset/spot.py:_to_kunquant`, `dataset/stock.py:_to_kunquant`).
2. `_make()` compiles the factor's op graph (`_get_factor_func()`, e.g. `factor/alpha101.py:_get_factor_func`) via `KunQuant.jit.cfake.compileit` into a native module (`base/factor.py:247-262`).
3. `kr.runGraph(executor, modu, input_dict, 0, num_time)` executes the compiled graph on a multi-thread executor, returning a dict of factor arrays (`base/factor.py:198-219`).
4. Result is wrapped back into an `xarray.Dataset` and stored via `XrBackend` (`_to_xarray_dataset`).

### Live Prediction Path (Nautilus Strategy)

1. `Test.__init__` (`backtest/test_strategy.py`) instantiates `Alpha101SpotKline`/`Alpha158SpotKline` and loads a raw `ModelRCrypto` (`nn.Module`, not `BaseModel`) directly via `torch.load`/`load_state_dict`, bypassing the `BaseModel.load()` path used offline.
2. `on_bar()` fires on each new bar; `_generate_prediction()` currently reads precomputed factor data by timestamp lookup (`self.alpha101.read().get_features().sel(timestamp=...)`) rather than the intended streaming (`cal_stream`) path — the streaming code is present but commented out.
3. Predictions are queued with a target execution time (`prediction_horizon`), then `_check_and_execute_predictions()` triggers `_execute_buy`/`_execute_sell` (TWAP-configured market orders) once ready and above `confidence_threshold`.

**State Management:**
- No shared application state / no server process. State lives in: on-disk zarr/parquet stores (dataset/factor/label caches), on-disk `.pth`/`.joblib` model checkpoints + `config.json`, and in-memory instance attributes (`self.data_backend.data`, `self.predictions_history` in the live strategy).

## Key Abstractions

**`DataBackend` (storage abstraction):**
- Purpose: Represents "a place data is stored," independent of its schema.
- Examples: `dataset/backend.py:XrBackend` (zarr/xarray), `dataset/backend.py:PlBackend` (parquet/polars).
- Pattern: Abstract Base Class with `read`/`write`/`to_internal`/`filter_by_*`.

**`xarray.Dataset` indexed by `(timestamp, symbol)` (canonical data shape):**
- Purpose: The single in-memory representation flowing between Dataset → Factor/Label → Model layers. All `.sel()`, `.combine_by_coords()`, and tensor-conversion code assumes this exact 2-D coordinate shape.
- Examples: `base/data.py`, `base/factor.py`, `base/model.py`.
- Pattern: Every layer's `get_xarray_dataset(["timestamp", "symbol"])` call enforces this shape at the boundary.

**Config dataclasses (`base/config.py`):**
- Purpose: Typed, serializable (`to_dict()`/`asdict`) parameter bags threaded through every domain object's constructor; each object's `config` property setter mutates the config on assignment (injecting inherited dates, resolved factor names, etc.) rather than the config being immutable.
- Examples: `DatasetConfig`, `FactorConfig`, `DLConfig`, `MLConfig`.
- Pattern: Dataclass + `to_dict()`, consumed both for object construction and for JSON-serialized checkpoint metadata (`base/model.py:_save_model`).

**KunQuant op-graph factor definition:**
- Purpose: Factors are defined as declarative dataflow graphs (`Builder`/`Input`/`Output`/`Op` composition), not imperative pandas/numpy transforms, then compiled to native code for performance.
- Examples: `factor/alpha101.py:_get_factor_func`, `my_ops/preprocess.py:WindowedZScore`.
- Pattern: Subclass `FactorKunQuant`, implement `_get_factor_func()` returning a `KunQuant.Stage.Function`.

## Entry Points

**`main.py`:**
- Location: `main.py` (repo root).
- Triggers: Manual `python main.py` (or `uv run main.py`).
- Responsibilities: None currently — 7-line `uv init` stub (`def main(): print("Hello from quantlab!")`). Not wired into any other module in the codebase.

**Ad hoc top-level scripts (the codebase's actual entry points):**
- `cal.py` — computes and saves Alpha101 factors for a fixed config (repo root).
- `train_model.py` — builds an `RNNClassifier` (`DLConfig`), collects data, trains/loads a checkpoint, generates predictions over a date range, and runs a vectorbt backtest, writing `portfolio_plot.html`.
- `test.py` — smoke-tests `StockDataset`/`Alpha101Stock` against local NASDAQ parquet data (`# %%` cell markers indicate this is meant to be run interactively, e.g. in VS Code/Jupyter).
- `get_binance_instruments.py` — standalone CLI (`argparse`) to refresh `config/instruments.yaml` from the live Binance API.
- `read_mock_data_sink.py` — memory-profiling scratch script for reading a `mock_data_sink` parquet hive dataset.
- `scripts/download_stock_data_from_tiingo.py` — parallel Tiingo downloader for NASDAQ tickers (Jupyter-cell-style `# %%` script).
- `test_nt.ipynb` — Nautilus Trader exploration notebook.
- **There is no single unified CLI/entry point** — each script independently constructs its own configs and imports the layers it needs.

## Architectural Constraints

- **Threading:** Single-process, but KunQuant factor computation explicitly uses a configurable multi-thread executor (`kr.createMultiThreadExecutor(self.config.njobs)`, default `njobs=128` in `FactorConfig`), and cross-validation folds can run in parallel threads via `joblib.Parallel(backend="threading")` (`base/model.py:train_cv`).
- **Global state:** None at module level observed (no module-level singletons/mutable globals); state is instance-scoped on `Dataset`/`FactorKunQuant`/`BaseModel` objects.
- **Hardcoded paths:** `config/__init__.py`'s factory functions (`spot_kline_config`, `alpha101_config`, `alpha158_config`, `spot_label_config`) hardcode absolute Linux paths (`/home/zhrdai/projects/crypto_quant/...`), while `train_model.py`/`test.py` hardcode different absolute macOS paths (`/Users/daizhaorong/projects/quantlab/...` and `/home/zhrdai/projects/crypto_quant/...` again for checkpoint loading). Any new environment (including this one) requires manually editing these paths before the pipeline will run.
- **Circular imports:** None observed; the layering (`base` → `dataset`/`factor`/`label` → `dl_model`/`ml_model` → `backtest` (`quantlab/backtest/`)) is consistently one-directional based on import statements read.
- **No `__init__.py` re-exports:** Every package's `__init__.py` (`base/`, `dataset/`, `factor/`, `label/`, `dl_model/`, `ml_model/`, `my_ops/`, `utils/`, `enums/`) is empty — all imports use full dotted paths to the implementation module (e.g. `from factor.alpha101 import Alpha101SpotKline`), never `from factor import Alpha101SpotKline`.

## Anti-Patterns

### Duplicated helper logic between script and library code

**What happens:** `get_binance_instruments.py` (repo root) re-implements `_get_binance_exchange_info`/`_parse_symbol_info` almost line-for-line identically to `utils/binance.py:get_instrument_info`/`_get_binance_exchange_info`/`_parse_symbol_info`.
**Why it's wrong:** Two independently-maintained copies of the same Binance API parsing logic will drift; a fix or filter-type change made in one won't propagate to the other.
**Do this instead:** `get_binance_instruments.py` should import and call `utils/binance.py:get_instrument_info` rather than redefining the HTTP/parsing logic.

### Broken/incomplete backtest helper committed as-is

**Resolved:** the helper was retired in phase 03.7 (D-31), and `quantlab/backtest/` replaced it. The description below is kept as history and no longer describes the code.

**What happened (history, helper retired):** `vecbt/bt.py:backtest_from_signals` calls `vbt.Portfolio.from_signals()` with **no arguments**, immediately after reindexing `close` — the entries/exits/short_entries/short_exits/index parameters accepted by the function are never passed through, so calling this function will raise at runtime (`vectorbt` requires at least `close` and entries).
**Why it was wrong (history, helper retired):** This function is unusable as written; any script importing it (it's currently commented out in `test.py`: `# from vecbt.bt import backtest_from_signals, print_performance`) would fail immediately if uncommented.
**Do this instead (superseded: use `quantlab/backtest/`, see `example/backtest.md`):** Pass the function's own parameters through to `vbt.Portfolio.from_signals(close, entries=long_entries, exits=long_exits, short_entries=short_entries, short_exits=short_exits, ...)`, following the working pattern already used directly in `test.py:130-138`.

## Error Handling

**Strategy:** Mostly fail-fast via explicit `raise ValueError`/`raise RuntimeError`/`raise FileNotFoundError` at layer boundaries (e.g. `base/backend.py` requires `read()`/`to_internal()` to be called before `.data` is accessed; `base/model.py:_save_model` refuses to overwrite an existing checkpoint directory). The live trading `Strategy` (`backtest/test_strategy.py`) is the one place with broad `try/except Exception` blocks (around balance/position lookups) that fall back to default values (e.g. `balance = 10000.0`) and log-and-continue rather than propagate.

**Patterns:**
- Config setters validate/derive values eagerly (e.g. `FactorKunQuant.config` setter auto-fills `start_date`/`end_date`/`factor_names` if unset) rather than deferring to call time.
- Unimplemented/partial functionality is signaled by raising inside the method body rather than via `NotImplementedError`-only stubs consistently — `dataset/stock.py` uses `raise ValueError("Not finished")` for `_get_instrument`/`_xr_to_bars`/`_to_nautilus`, while `base/model.py:_auto_train` uses `raise NotImplementedError("ML training not implemented")` for the `MLConfig` branch.

## Cross-Cutting Concerns

**Logging:** `loguru.logger` used directly (no wrapper/adapter), imported per-module (`base/data.py`, `base/factor.py`, `dataset/spot.py`, `utils/timer.py`, `utils/nautilus.py`, `utils/binance.py`). `utils/timer.py:Timer` is a reusable context manager (`with Timer("task name"): ...`) that logs start/elapsed-time, used consistently across dataset/factor save/load/compute operations.

**Validation:** Shape assertions in the model layer (`base/model.py:_assert_shape_match_x`/`_assert_shape_match_y`) verify tensor dimensions against `num_symbols`/`num_factors`/`num_labels` before training. No schema/type validation framework (no pydantic) — dataclasses provide typing only, not runtime validation.

**Authentication:** Not applicable within the codebase itself; see `INTEGRATIONS.md` for the one embedded API credential.

---

*Architecture analysis: 2026-09-04*
*Backtest sections updated 2026-09-15 (phase 03.7-12): the backtester layer is described; the vecbt helper is retired.*
