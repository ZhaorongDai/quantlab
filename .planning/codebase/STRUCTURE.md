# Codebase Structure

**Analysis Date:** 2026-09-04

## Directory Layout

```
quantlab/
├── base/                 # Abstract base classes shared by every layer
│   ├── backend.py         #   DataBackend ABC (storage abstraction)
│   ├── backtest.py        #   BaseBacktester ABC (run/run_cv templates, MarketSpec, result dataclasses)
│   ├── config.py          #   DatasetConfig/FactorConfig/DLConfig/MLConfig dataclasses
│   ├── data.py             #   Dataset ABC (raw data -> xarray)
│   ├── factor.py           #   FactorKunQuant ABC (KunQuant-compiled factor/label computation)
│   ├── model.py             #   BaseModel ABC (training loop, CV, checkpointing, W&B)
│   └── __init__.py          #   empty
├── dataset/               # Concrete Dataset + DataBackend implementations
│   ├── backend.py           #   XrBackend (zarr/xarray), PlBackend (parquet/polars)
│   ├── spot.py               #   SpotKlineDataset (Binance CSV -> xarray/Nautilus bars)
│   ├── stock.py               #   StockDataset (Tiingo/NASDAQ parquet; partially unimplemented)
│   └── __init__.py            #   empty
├── factor/                # Concrete factor sets (KunQuant op graphs)
│   ├── alpha101.py           #   Alpha101SpotKline / Alpha101Stock
│   ├── alpha158.py            #   Alpha158SpotKline
│   └── __init__.py            #   empty
├── label/                 # Forward-return label definitions
│   ├── spot.py                #   SpotReturn, SpotBinaryReturn
│   └── __init__.py             #   empty
├── my_ops/                # Custom KunQuant composite ops
│   ├── preprocess.py          #   WindowedZScore, WindowedRobustStandardization
│   └── __init__.py             #   empty
├── dl_model/               # Torch model implementations (subclass BaseModel)
│   ├── mlp.py                  #   MLP / MLPRegressor
│   ├── rnn.py                    #   ModelRCrypto (GRU/LSTM), RNNRegressor
│   ├── rnn_classification.py      #   ModelRCrypto classifier variant, RNNClassifier
│   └── __init__.py                #   empty
├── ml_model/                # Non-torch model persistence (no concrete MLConfig model yet)
│   ├── backend.py                #   MlBackend (joblib-based ModelBackend)
│   └── __init__.py                #   empty
├── backtest/                # Cross-sectional backtester layer (phase 03.7), on base/backtest.py:BaseBacktester
│   ├── engine_vectorbt.py        #   VectorBtBacktester (vectorbt from_orders, target percent, t+1 open fills)
│   ├── selection.py               #   CrossSectionTopNSelector, rebalance_mask, resolve_score_label
│   ├── us_equity.py                #   US_EQUITY_MARKET, USEquityCrossectionSelectStockVectorBt
│   └── __init__.py                 #   empty
│   (Nautilus test_strategy.py deleted 2026-09-07; the vecbt/ helper package was retired in phase 03.7, D-31)
├── config/                    # Config factories + static instrument metadata
│   ├── __init__.py                #   spot_kline_config/alpha101_config/alpha158_config/spot_label_config (hardcoded absolute paths)
│   └── instruments.yaml            #   Binance instrument precision/fees/margin metadata
├── enums/                      # Shared constants
│   ├── constant.py                #   Date.START_DATE / Date.END_DATE
│   ├── data.py                     #   BinanceCSVHeaders
│   └── __init__.py                  #   empty
├── utils/                       # Cross-cutting helpers
│   ├── timer.py                    #   Timer context manager (loguru start/elapsed logging)
│   ├── file.py                      #   CSV/parquet file globbing + date filtering
│   ├── binance.py                    #   Binance exchangeInfo fetch/parse
│   ├── nautilus.py                    #   Nautilus CurrencyPair/instrument construction, bar-type string generation
│   ├── module.py                       #   get_cls_from_path + load_*_from_config (dynamic reconstruction from JSON)
│   ├── asdict.py                        #   asdict_customized (deep-copy-based dataclass serialization)
│   └── __init__.py                      #   empty
├── scripts/                       # One-off data-download scripts
│   └── download_stock_data_from_tiingo.py  # Tiingo NASDAQ downloader (contains hardcoded API key)
├── .planning/                       # GSD planning artifacts (this codebase map lives here)
│   └── codebase/
├── main.py                          # uv-generated hello-world stub (7 lines, unused elsewhere)
├── cal.py                           # Ad hoc script: computes and saves Alpha101 factors
├── train_model.py                   # Ad hoc script: train/load RNNClassifier + vectorbt backtest
├── test.py                          # Ad hoc smoke-test script for StockDataset/Alpha101Stock
├── get_binance_instruments.py       # Standalone CLI to refresh config/instruments.yaml
├── read_mock_data_sink.py           # Memory-profiling scratch script
├── test_nt.ipynb                    # Nautilus Trader exploration notebook
├── pyproject.toml                   # Project metadata (uv-generated; dependencies = [], stale vs. actual imports)
├── uv.lock                          # Stale lockfile (locks old project name "crypto-quant"; only 3 packages)
├── README.md                        # Describes an aspirational structure (models/, examples/) that does not match the actual layout
├── .gitignore
└── .git/
```

## Directory Purposes

**`base/`:**
- Purpose: Defines the abstract contracts (`DataBackend`, `Dataset`, `FactorKunQuant`, `BaseModel`) and the dataclass configs (`DatasetConfig`, `FactorConfig`, `DLConfig`, `MLConfig`) that every concrete implementation elsewhere in the repo depends on.
- Contains: Abstract base classes only, no concrete/runnable pipelines.
- Key files: `base/model.py` (largest and most central file in the repo — the shared training loop).

**`dataset/`, `factor/`, `label/`, `my_ops/`:**
- Purpose: The data pipeline — raw market data ingestion (`dataset/`), engineered features (`factor/`), prediction targets (`label/`), and shared factor-graph building blocks (`my_ops/`).
- Contains: One concrete class per data source/factor-set/label-set; each file is short (60–210 lines) and follows the same ABC-subclass pattern.

**`dl_model/`, `ml_model/`:**
- Purpose: Model definitions and training-loop specializations. `dl_model/` is by far the more developed of the two (torch models fully implemented); `ml_model/` currently only contains a persistence backend with no concrete `MLConfig`-based model — `BaseModel._auto_train` explicitly raises `NotImplementedError` for the ML path.
- Contains: `nn.Module` definitions plus `BaseModel` subclasses that implement the 5-method training contract.

**`backtest/` (`quantlab/backtest/`, with `quantlab/base/backtest.py`):**
- Purpose: The cross-sectional backtester layer. `base/backtest.py:BaseBacktester` owns the two public entry points, `run()` (backtest one model over a window) and `run_cv()` (replay a `train_cv` run's folds as one stitched curve), and writes each run to its own run directory. `backtest/` holds the vectorbt engine, the TopN selector and the US-equity composition. Event-driven (Nautilus) backtesting is reserved for Phase 6.
- Contains: `engine_vectorbt.py:VectorBtBacktester`, `selection.py:CrossSectionTopNSelector`, `us_equity.py:USEquityCrossectionSelectStockVectorBt`, and an empty `__init__.py`. (Superseded: the Nautilus `backtest/test_strategy.py` was deleted 2026-09-07, and the `vecbt/` helper package was retired in phase 03.7, D-31.)

**`config/`:**
- Purpose: Central location for both static YAML metadata (`instruments.yaml`) and Python factory functions that build fully-populated `DatasetConfig`/`FactorConfig` objects for the three "known" pipelines (spot klines, alpha101, alpha158, spot labels).
- Contains: All paths inside `config/__init__.py` are hardcoded absolute Linux paths pointing at a specific developer's home directory — this file must be edited (or refactored to read from env vars) before running on any other machine.

**`enums/`:**
- Purpose: Small shared constant containers (default date range, Binance CSV column headers). Not true Python `enum.Enum` types — implemented as plain `@dataclass` classes with class-level attributes.

**`utils/`:**
- Purpose: Generic cross-cutting helpers with no dependency on `base/`'s domain model (except `module.py`, which imports `base.config`). Grab-bag of timing, file globbing, Binance API access, Nautilus instrument construction, and dynamic class loading.

**`scripts/`:**
- Purpose: Holds standalone, rarely-run data-acquisition scripts (currently just the Tiingo downloader). Distinct from the top-level ad hoc scripts (`cal.py`, `train_model.py`, etc.), though the distinction is not strictly enforced — most one-off scripts live at the repo root instead.

**`.planning/codebase/`:**
- Purpose: GSD-generated codebase maps (this document and its siblings). Not part of the application; documentation only.

## Key File Locations

**Entry Points:**
- `main.py`: uv-generated stub, disconnected from the rest of the codebase.
- `train_model.py`, `cal.py`, `test.py`, `get_binance_instruments.py`, `read_mock_data_sink.py`, `scripts/download_stock_data_from_tiingo.py`: the actual, independently-run entry points.

**Configuration:**
- `base/config.py`: dataclass config schema definitions.
- `config/__init__.py`: config factory functions (hardcoded paths).
- `config/instruments.yaml`: static per-exchange instrument metadata.
- `pyproject.toml`, `uv.lock`: Python project/dependency metadata (currently out of sync with actual code — see `STACK.md`).

**Core Logic:**
- `base/model.py`: training loop, CV, checkpointing (the file to read first to understand the system).
- `base/factor.py`: KunQuant compile/execute machinery.
- `base/data.py`: dataset read/write/convert machinery.

**Testing:**
- None. No `tests/` directory exists.

## Naming Conventions

**Files:**
- `snake_case.py` throughout, matching the primary class defined inside where there's a 1:1 mapping (e.g. `base/backend.py:DataBackend`, `base/model.py:BaseModel`), but not strictly enforced elsewhere (`dataset/spot.py` defines `SpotKlineDataset`; `dataset/stock.py` defines `StockDataset`).
- Domain-specific concrete implementation files are named after the data source/factor set they implement (`spot.py`, `stock.py`, `alpha101.py`, `alpha158.py`), not after the class name.

**Directories:**
- Top-level directories double as Python packages (no `src/` layout) and are named after their pipeline stage in singular or plural form inconsistently: `dataset/` (plural concept, singular dir name), `factor/` (singular), `label/` (singular), `dl_model/`/`ml_model/` (singular), `my_ops/` (plural). No enforced convention.

## Where to Add New Code

**New data source (e.g. a new exchange or vendor):**
- Subclass `base/data.py:Dataset`, add the concrete class under `dataset/` (e.g. `dataset/futures.py`), implementing `_raw_data_to_xr`, `_to_kunquant`, `_to_nautilus`.
- Add a corresponding config factory function to `config/__init__.py` (or, preferably when extending, parameterize paths via environment variables rather than continuing the hardcoded-path pattern).

**New factor set:**
- Subclass `base/factor.py:FactorKunQuant`, add under `factor/` (e.g. `factor/alpha191.py`), implementing `_get_factor_func` (KunQuant `Builder`/`Op` graph) and `_get_factor_names`. Reuse `my_ops/preprocess.py` ops for normalization where applicable, or add new composite ops there.

**New label:**
- Subclass `base/factor.py:FactorKunQuant` (same base as factors — labels and factors share the same class hierarchy), add under `label/`, implementing `_get_labels`/`_get_features` to shift/transform the target column.

**New model architecture:**
- Subclass `base/model.py:BaseModel`, add under `dl_model/` (torch) or `ml_model/` (non-torch — note this path is currently unimplemented in `BaseModel._auto_train`, expect to need to add ML training support there too), implementing `_init_model`, `_train_one_batch`, `_val_one_batch`, `_test_one_batch`, `_preprocess` (renamed from `_*_one_epoch` on 2026-09-07 — they are called once per BATCH, and the old name had already caused an early-stopping defect).

**New backtest:**
- New engine: subclass `quantlab/base/backtest.py:BaseBacktester` under `quantlab/backtest/` (e.g. `engine_<name>.py`), implementing the engine hooks `_simulate`, `_simulate_benchmark`, `_engine_stats` and `_period_returns_stats`. `run()` and `run_cv()` stay on the base and must not be overridden (locked by `tests/test_backtest_contracts.py`).
- New selection style or market: subclass `quantlab/backtest/engine_vectorbt.py:VectorBtBacktester`, declaring `config_cls`, a `MARKET` (`MarketSpec`), and a `_generate_signals` that composes a selector, as `us_equity.py:USEquityCrossectionSelectStockVectorBt` does with `CrossSectionTopNSelector`. Price column names live only on the `MarketSpec` (D-04). See `example/backtest.md`.
- Event-driven (Nautilus) backtests are reserved for Phase 6. (Superseded guidance: the `backtest/test_strategy.py` pattern was deleted 2026-09-07, and the `vecbt/bt.py` helper was retired in phase 03.7, D-31.)

**Utilities:**
- Shared, domain-agnostic helpers go in `utils/`. Domain-specific helpers (e.g. Binance-only, Nautilus-only) already have dedicated files there (`utils/binance.py`, `utils/nautilus.py`) — follow that per-integration file split rather than adding to a catch-all.

**Tests (none exist yet):**
- No convention established. If adding a test suite, introduce a `tests/` directory and a test runner dependency (e.g. `pytest`) in `pyproject.toml`, since neither currently exists.

## Special Directories

**`.planning/`:**
- Purpose: GSD workflow state (phase plans, codebase maps). Not part of the running application.
- Generated: Yes (by GSD commands).
- Committed: Repository-dependent; check `.gitignore` before assuming.

**No `.venv/`, `downloads/`, `data/`, `model_ckpt/` directories are present in the repo** despite being referenced throughout the code (`config/__init__.py` paths, `train_model.py`'s `model_save_dir="./model_ckpt"`, `scripts/download_stock_data_from_tiingo.py`'s `downloads/nasdaq_data`) — these are expected to be created at runtime or exist only on the original developer's machine, and are excluded from version control per `.gitignore`.

---

*Structure analysis: 2026-09-04*
*Backtest entries updated 2026-09-15 (phase 03.7-12): `quantlab/backtest/` and `quantlab/base/backtest.py` described; the vecbt helper package is retired.*
