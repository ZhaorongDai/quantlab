## Project

**quantlab**

一个端到端的量化研究后端平台：从多市场、多频率行情数据出发，经因子计算、收益预测、组合优化，生成目标持仓并完成回测，全流程通过配置文件驱动、可复现。当前阶段只做后台，面向未来平台化（服务化、多用户、因子/模型在线编辑与测试、网页前端）预留架构空间，但不在本阶段实现。

**Core Value:** 一条打通的、config 驱动可复现的量化流水线（数据→因子→收益模型→组合优化→目标持仓→回测→结果），模块间用清晰的输入输出契约组合，任何一环都能独立替换/扩展而不需要推倒重来。

### Constraints

- **数据格式**: 模块间统一使用 xarray（Zarr 落盘），不使用 DataFrame 作为流水线层间传输格式；模型训练直接消费 xarray — 用户明确要求，是贯穿整个流水线的硬约束
- **因子计算后端**: 双后端支持——KunQuant（批量 + 流式，保留未来实时数据接入能力）为主，Polars 为新因子的补充计算路径（仅批量，不需要流式）；能用 xarray/KunQuant 完成的处理，优先不用 Polars — 用户明确的技术选型优先级
- **回测技术栈**: 向量化回测优先用 vectorbt 打通；事件驱动回测（NautilusTrader）作为预留扩展能力，非 v1 交付重点。原型 `backtest/test_strategy.py` 已于 2026-09-07 删除（早于当前 Dataset/Factor 契约），重启时按当前契约重建，不复活旧原型
- **凭证安全**: API Key 等敏感信息一律通过环境变量读取，不硬编码 — 一个早已删除的脚本曾硬编码 Tiingo Key 并造成一次真实泄露
- **可复现性**: 全流程参数尽量通过配置文件驱动 — 用户明确要求，服务于实验可复现
- **架构契约**: 数据模块输出数据、因子模块输出因子、收益模型输出未来收益/收益排名预测、组合优化模型输出每个标的目标持仓百分比——各模块通过清晰的输入输出契约组合 — 便于未来插拔式扩展与平台化
- **包管理**: 使用 `uv` — 用户明确要求，延续现有项目的包管理方式
- **范围**: 当前阶段只实现后台，不做网页前端 — 用户明确排除

## Technology Stack

## Languages
- Python — `requires-python = ">=3.13"` per `pyproject.toml`. The system `python3` currently resolves to 3.9.6 (`python3 --version`), so a 3.13 interpreter must be provisioned via `uv` before the project will run (`uv` 0.8.14 is installed at `~/.local/bin/uv`).
- YAML — instrument/venue metadata (`config/instruments.yaml`).
- Jupyter — no notebook is checked in; runnable walkthroughs live under `examples/`.
## Runtime
- CPython >=3.13 (declared, not currently installed as default `python3` on this machine).
- GPU/CUDA expected for deep-learning model code: `base/model.py` calls `torch.cuda.manual_seed_all` and `DLModel.device` selects `"cuda"` when available, falling back to `"cpu"`.
- `uv` (evidenced by `uv.lock` and the bare `pyproject.toml` layout uv generates).
- Lockfile: present (`uv.lock`) but **stale/mismatched** — it locks only `bottleneck`, `numpy`, and a self-reference to a project named `crypto-quant` (not `quantlab`), and declares `requires-python = ">=3.12"` (vs. `>=3.13` in `pyproject.toml`). This lockfile predates the current `pyproject.toml` and does not reflect the packages actually imported by the code (see Key Dependencies below). Running `uv sync` today will not install a working environment.
## Frameworks
- No web/API framework. This is a research/trading codebase (data pipeline + ML models + backtesting), not a service.
- **PyTorch** (`torch`) — deep-learning models (`dl_model/mlp.py`, `dl_model/rnn.py`, `dl_model/rnn_classification.py`), trained through the `base/model.py:DLModel` epoch loop (`DataLoader`/`TensorDataset`).
- **scikit-learn** (`sklearn.metrics`) — evaluation metrics (accuracy, F1, ROC-AUC, R², RMSE, etc.) used inside DL training loops, not for model fitting itself.
- **KunQuant** — JIT-compiled factor computation graph library. Used throughout `base/factor.py`, `factor/alpha101.py`, `factor/alpha158.py`, `label/spot.py`, `my_ops/preprocess.py` to build and compile (`cfake.compileit`) high-performance alpha factor pipelines (`KunRunner`, `Function`, `Builder`, `Op`, `Stage`).
- **Nautilus Trader** (`nautilus_trader`) — declared as a dependency but imported by no code: the `to_nautilus` export, `DatasetConfig.catalog_path` and `utils/nautilus.py` were removed 2026-09-25. Its live/backtest trading engine is not used by any code in the repo either: the only `Strategy` subclass, `backtest/test_strategy.py`, was deleted 2026-09-07.
- **vectorbt** (`vectorbt`) — vector-based backtesting/portfolio simulation, used by `quantlab/backtest/engine_vectorbt.py` (`VectorBtBacktester`: `Portfolio.from_orders` with target-percent weights) and by the ad hoc `test.py` (`vbt.Portfolio.from_signals`). The former `vecbt/bt.py` signal helper was retired in phase 03.7 (D-31).
- None detected. No `pytest`/`unittest` configuration, no test runner dependency, no `tests/` directory. `test.py` at the repo root is an untracked ad hoc script, not part of a test suite.
- No linter/formatter config detected (no `.eslintrc`, `ruff.toml`, `.flake8`, `pyproject.toml` `[tool.ruff]`/`[tool.black]` sections).
- No CI configuration (no `.github/workflows`, no other CI YAML).
## Key Dependencies
- `numpy`, `pandas`, `polars`, `xarray` — the core numerical/tabular/labeled-array stack. `xarray.Dataset` (dims `timestamp`, `symbol`) is the canonical in-memory data representation passed between dataset, factor, label, and model layers.
- `torch` — model definition and training (`dl_model/*`).
- `xgboost` — tree-model future-return regression (`ml_model/xgb.py:XGBoostRegressor`), trained with `xgb.train` and XGBoost's native `EarlyStopping(save_best=True)`.
- `scipy` — `scipy.stats.rankdata` for the cross-sectional RankIC in `utils/metrics.py`; declared directly in `pyproject.toml`.
- `KunQuant` — compiled factor computation (`factor/*`, `base/factor.py`, `label/spot.py`, `my_ops/preprocess.py`). Appears to be a specialized/possibly local or pinned package, not a mainstream PyPI package with a standard lockfile entry.
- `nautilus_trader` — declared but currently unused (the Nautilus export was removed 2026-09-25); kept for the reserved event-driven backtest.
- `vectorbt` — cross-sectional backtest engine (`quantlab/backtest/engine_vectorbt.py`, the only quantlab module importing it; `Portfolio.from_orders` with target-percent weights) and `test.py`. The former `vecbt/bt.py` helper was retired in phase 03.7 (D-31).
- `wandb` — experiment tracking, initialized in every training run (`base/model.py:_init_wandb`).
- `loguru` — logging throughout (`base/data.py`, `base/factor.py`, `utils/timer.py`, `utils/binance.py`).
- `joblib` — parallelism (`Parallel`/`delayed` for CV folds) and non-torch model persistence through `ml_model/backend.py:MlBackend` (the `.joblib` checkpoint backend of `base/model.py:MLModel`).
- `tqdm` — progress bars across data/factor/CV loops.
- `bottleneck` — the one dependency actually declared/locked (`pyproject.toml`/`uv.lock` under the old `crypto-quant` name); likely used for fast rolling/window numpy ops (not directly observed via `import` grep, may be an `xarray`/`pandas` accelerator dependency).
- `requests` — Binance REST calls (`quantlab/utils/binance.py`), the Tiingo/Alpaca clients and the Wikipedia fetchers in `quantlab/universe.py`.
- `PyYAML` (`yaml`) — reading/writing `config/instruments.yaml`.
- `tiingo` — Tiingo market-data API client (`quantlab/acquisition/tiingo.py`).
- `plotly` — backtest report rendering (`quantlab/utils/backtest_report.py`).
## Configuration
- No `.env` file present at the repo root.
- Credentials come from environment variables only (`WRDS_USERNAME`, with the password read by libpq from `~/.pgpass`; `TIINGO_API_KEY`, `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY` for the library-only vendors). No file in the repo holds a key; the script that once hardcoded one was deleted 2026-09-25.
- No other secret/credential files detected (no `credentials.json`, `.npmrc`, private keys).
- `pyproject.toml` — project metadata only (`name`, `version`, `readme`, `requires-python`, empty `dependencies = []`). No `[tool.*]` sections, no build-system customization, no optional dependency groups.
- `uv.lock` — present but stale (see Runtime/Package Manager above); does not currently reflect a resolvable, working dependency set for this codebase.
## Platform Requirements
- macOS (current dev host is Darwin/arm64 per environment) or Linux — code contains Linux-style absolute paths hardcoded into config factories (see `config/__init__.py`, e.g. `/home/zhrdai/projects/crypto_quant/...`), implying the primary development/training environment is a Linux workstation, not this machine.
- macOS dev host OpenMP clash: xgboost's wheel links Homebrew libomp while torch bundles its own, so a process mixing both segfaults or deadlocks. `tests/conftest.py` sets `OMP_NUM_THREADS=1` on darwin before any import (locked by `tests/test_macos_openmp_guard.py`); user scripts/notebooks on macOS that mix torch and xgboost must set it too. Linux is untouched. See `example/model.md`.
- A working `uv`-managed Python 3.13 environment must be created and `pyproject.toml` dependencies must be reconciled with actual imports before the code can run; currently `uv sync` alone is insufficient.
- No deployment target detected (no Dockerfile, no cloud config, no server entry point). This is a local research/trading pipeline intended to run on a workstation/server with GPU access for model training and disk access to large local datasets (CSV/Parquet klines, zarr stores).

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.

## Architecture

## System Overview
```text
```
## Component Responsibilities
| Component | Responsibility | File |
|-----------|----------------|------|
| `DataBackend` (abstract) | Defines read/write/filter contract for any storage medium | `base/backend.py` |
| `XrBackend` | Zarr-backed storage for `xarray.Dataset` (canonical `[timestamp, symbol]` shape) | `quantlab/backend.py` |
| `PlBackend` | Parquet-backed storage via `polars.LazyFrame` | `quantlab/backend.py` |
| `Dataset` (abstract) | Loads raw market data into the internal xarray representation; converts to KunQuant input arrays | `base/data.py` |
| `SpotKlineDataset` | Binance spot kline CSV ingestion | `dataset/spot.py` |
| `StockDataset` | NASDAQ/Tiingo parquet ingestion | `dataset/stock.py` |
| `FactorKunQuant` (abstract) | Compiles and executes KunQuant factor graphs (batch and streaming modes) | `base/factor.py` |
| `Alpha101SpotKline` / `Alpha158SpotKline` | Concrete factor sets (Alpha101 formulaic factors, Alpha158 factor library) | `factor/alpha101.py`, `factor/alpha158.py` |
| `SpotReturn` / `SpotBinaryReturn` | Forward-return regression/classification labels | `label/spot.py` |
| `WindowedZScore` | Custom KunQuant composite op for time-series factor normalization (`WindowedRobustStandardization` sat beside it with zero call sites and was deleted 2026-09-07) | `my_ops/preprocess.py` |
| `BaseModel` (abstract) | Framework-agnostic shared lifecycle: config type guard, factor/label collection, the public `train`/`train_cv`/`load`/`predict` (implemented once), checkpoint dir + `config.json`, the single CV fold generator `_cv_folds`, per-fold CV results and the `{cls}_cv_summary` W&B run | `base/model.py` |
| `DLModel` (abstract) | torch variant: device, `to_tensor`, DataLoader epoch loop with per-epoch early stopping and best-epoch rollback, `.pth` checkpoints; five tensor hooks | `base/model.py` |
| `MLModel` (abstract) | numpy variant for tree/ML models: one `_fit_model` call with the library's native early stopping, `_evaluate` writing `{split}_*` metrics to the W&B summary, `.joblib` checkpoints via `MlBackend`, resolved-hyperparameter recording; four hooks | `base/model.py` |
| `XGBoostRegressor` | Future-return regression with `xgb.train` + native `EarlyStopping(save_best=True)`, per-round W&B callback, sklearn-alias normalization, CV via inherited `train_cv` | `ml_model/xgb.py` |
| Panel metrics | Vectorized MSE/RMSE/MAE/R² and cross-sectional IC/RankIC over `[T, S]` panels | `utils/metrics.py` |
| `MLPRegressor`, `RNNRegressor`, `RNNClassifier` | Concrete torch model heads (MLP, GRU/LSTM regressor, GRU/LSTM classifier with auxiliary-label architecture) | `dl_model/mlp.py`, `dl_model/rnn.py`, `dl_model/rnn_classification.py` |
| `MlBackend` | joblib-based checkpoint persistence for `MLModel` heads | `ml_model/backend.py` |
| `BaseBacktester` (abstract) | Backtester ABC: config type guard, the two public entry points `run()` (model backtest) and `run_cv()` (replays a `train_cv` run's `cv_folds.json` as one stitched curve), bar-counted warm-up, D-17 in/out-of-sample split, single-symbol benchmark loading and the `relative` excess statistics, metrics, run directory, data fingerprints; engine hooks `_simulate`, `_simulate_benchmark`, `_engine_stats`, `_period_returns_stats` | `quantlab/base/backtest.py` |
| `VectorBtBacktester` (abstract) | vectorbt engine: `Portfolio.from_orders` with target-percent weights, a signal at bar t fills at bar t+1's open (D-05), forced-liquidation records for delisted holdings; the only quantlab module importing vectorbt | `quantlab/backtest/engine_vectorbt.py` |
| `CrossSectionTopNSelector` | TopN long-only or disjoint long/short target weights on rebalance bars (D-03 weights contract) | `quantlab/backtest/selection.py` |
| `USEquityCrossectionSelectStockVectorBt` | Concrete US-equity backtester composing `US_EQUITY_MARKET`, the selector and the vectorbt engine; rebuilt from a run's `config.json` by `quantlab/utils/module.py:load_backtester_from_config`. It replaces the `vecbt/bt.py:backtest_from_signals` helper, retired in phase 03.7 (D-31) | `quantlab/backtest/us_equity.py` |
| Config factories | Root-relative factories for the US-equity dataset (`stock_kline_config`), acquisition (`stock_acquisition_config`) and universe table (`universe_config`), plus `set_data_root`/`get_data_root` | `quantlab/config/__init__.py` |
| `DatasetConfig`/`FactorConfig`/`DLConfig`/`MLConfig` | Dataclass configuration objects threaded through every layer | `base/config.py` |
## Pattern Overview
- Every domain object (`Dataset`, `FactorKunQuant`, `BaseModel`) shares the same lifecycle idiom: a `config` property setter that normalizes dates/names on assignment, a `read()`/`cal()`/`save()` trio for lazy-vs-eager data materialization, and `xarray.Dataset` as the universal in-memory exchange format between layers (always indexed by `timestamp` and `symbol`).
- Factor/label computation is offloaded to **KunQuant**, a compiled-graph engine (`Builder`/`Op`/`Function` → `cfake.compileit` → `KunRunner`), not plain numpy/pandas — factors are defined declaratively as op graphs, then JIT-compiled to native code and executed via a multi-thread executor (`kr.createMultiThreadExecutor`).
- Model layer (`base/model.py`) is a three-layer hierarchy: framework-agnostic `BaseModel` owns the public `train`/`train_cv`/`load`/`predict` and CV fold geometry; `DLModel` (torch, `[num_times, num_symbols, num_features]` tensors through `TensorDataset`/`DataLoader`) requires five hooks (`_init_model`, `_train_one_batch`, `_val_one_batch`, `_test_one_batch`, `_preprocess`); `MLModel` (numpy, native library early stopping, no epoch loop) requires four (`_init_model`, `_preprocess`, `_fit_model`, `_forward`). Each variant declares `config_cls` (`DLConfig`/`MLConfig`) and `checkpoint_suffix` (`.pth`/`.joblib`).
- Configuration is dataclass-based (not env-var or YAML-based, except `config/instruments.yaml` for exchange instrument metadata) and is **hardcoded with absolute filesystem paths per developer machine** rather than parameterized (see `config/__init__.py`).
- No dependency injection framework, no plugin registry beyond `utils/module.py:get_cls_from_path` (dynamic import-by-dotted-path used to reconstruct a `Dataset`/`Factor`/`Model` from a saved JSON config).
## Layers
- Purpose: Abstracts "how data is persisted" from "what the data means."
- Location: `quantlab/backend.py` (concrete: `XrBackend`, `PlBackend`), `base/backend.py` (abstract `DataBackend`).
- Contains: `read`/`write`/`to_internal`/`filter_by_date`/`filter_by_symbol`/`get_xarray_dataset`/`get_lazyframe`.
- Depends on: `xarray`, `polars`, `pandas`.
- Used by: `Dataset` and `FactorKunQuant`, each of which owns a `self.data_backend` instance.
- Purpose: Converts raw external data (Binance CSV klines, Tiingo/NASDAQ parquet) into the canonical `xarray.Dataset` and persists it via a `DataBackend`. Also converts to KunQuant input arrays (`to_kunquant`).
- Location: `base/data.py` (ABC `Dataset`), `dataset/spot.py` (`SpotKlineDataset`), `dataset/stock.py` (`StockDataset`).
- Depends on: Storage Backend layer, `utils/file.py`, `utils/timer.py`.
- Used by: Factor/Label layer (each `FactorConfig` embeds a `Dataset` instance) and directly by the WRDS download scripts (`scripts/wrds/*.py`).
- Purpose: Computes engineered features (factors) and prediction targets (labels) from dataset data using compiled KunQuant graphs, in either batch mode (`cal()`, operates on a full historical window) or streaming mode (`cal_stream()`, incremental per-bar updates for live trading).
- Location: `base/factor.py` (ABC `FactorKunQuant`), `factor/alpha101.py`, `factor/alpha158.py`, `label/spot.py`, `my_ops/preprocess.py` (custom `WindowedCompositiveOp` subclasses).
- Depends on: Dataset layer (each factor config embeds a `Dataset`), `KunQuant`.
- Used by: Model layer (`DLConfig.factors`/`DLConfig.labels`).
- Purpose: Orchestrates the full training lifecycle — pulling factor/label data into a combined `xarray.Dataset` (`collect()`), splitting into train/val/test or rolling walk-forward CV windows (one `_cv_folds` generator for both variants and both sequential/parallel branches), training (DL: epoch loop with per-epoch early stopping; ML: one native-early-stopping fit), checkpointing (`.pth` via `torch.save` in `DLModel`, `.joblib` via `MlBackend` in `MLModel`), and W&B logging.
- Location: `base/model.py` (`BaseModel` / `DLModel` / `MLModel`), `base/config.py` (`DLConfig`/`MLConfig`), `dl_model/mlp.py`, `dl_model/rnn.py`, `dl_model/rnn_classification.py` (all `DLModel`), `ml_model/xgb.py` (`XGBoostRegressor`, an `MLModel`), `ml_model/backend.py` (`MlBackend`), `utils/metrics.py` (panel metrics).
- Depends on: Factor/Label layer, `torch`, `xgboost`, `wandb`, `sklearn.metrics`, `scipy`, `joblib`.
- Used by: `examples/train_model.py` and the ad hoc `test.py`.
- Purpose: Evaluates a model's trading performance on a cross-sectional universe. `BaseBacktester.run()` is the model backtest: train mode trains the model on its own dates, load mode loads a checkpoint, and the model then predicts the window through `predict_panel`. `run_cv()` is the model-CV backtest: it reads a `train_cv` run's `cv_folds.json`, backtests each fold with its own checkpoint on its own test segment, then simulates the concatenated fold weights once as a stitched curve. Signals are D-03 target weights (a `weight` variable on `(timestamp, symbol)`), and a signal formed at bar t fills at bar t+1's open (D-05). Metrics are reported for the whole window and split in-sample/out-of-sample against the model's training window (D-17). Every run writes its own run directory (`config.json`, `weights.zarr`, `equity.zarr`, `metrics.json`, `liquidations.json`, `fingerprint.json`, `report.html`), which `quantlab/utils/module.py:load_backtester_from_config` can rebuild and re-run. The event-driven (Nautilus) path is still reserved for Phase 6, with no current implementation.
- Location: `quantlab/base/backtest.py` (`BaseBacktester`, `MarketSpec`, result dataclasses), `quantlab/backtest/engine_vectorbt.py` (`VectorBtBacktester`), `quantlab/backtest/selection.py` (`CrossSectionTopNSelector`), `quantlab/backtest/us_equity.py` (`USEquityCrossectionSelectStockVectorBt`), `quantlab/base/config.py` (`BacktestConfig`/`CrossSectionBacktestConfig`). The former `vecbt/bt.py` helper was retired in phase 03.7 (D-31). See `example/backtest.md`.
- Depends on: Model layer (`predict_panel`, checkpoints, `cv_folds.json`), Factor/Label layer (re-dated to cover warm-up), Dataset layer (prices), `vectorbt`, `plotly`, optionally `wandb`.
- Used by: Nothing else in quantlab except the config loader in `quantlab/utils/module.py`. This is a terminal/output layer, and `tests/test_backtest_contracts.py` locks that direction.
## Data Flow
### Primary Training Path
### Factor Computation Path (KunQuant)
- No shared application state / no server process. State lives in: on-disk zarr/parquet stores (dataset/factor/label caches), on-disk `.pth`/`.joblib` model checkpoints + `config.json`, and in-memory instance attributes (`self.data_backend.data`, `self.predictions_history` in the live strategy).
## Key Abstractions
- Purpose: Represents "a place data is stored," independent of its schema.
- Examples: `quantlab/backend.py:XrBackend` (zarr/xarray), `quantlab/backend.py:PlBackend` (parquet/polars).
- Pattern: Abstract Base Class with `read`/`write`/`to_internal`/`filter_by_*`.
- Purpose: The single in-memory representation flowing between Dataset → Factor/Label → Model layers. All `.sel()`, `.combine_by_coords()`, and tensor-conversion code assumes this exact 2-D coordinate shape.
- Examples: `base/data.py`, `base/factor.py`, `base/model.py`.
- Pattern: layers call `get_xarray_dataset(["timestamp", "symbol"])` at the boundary, and as of 2026-09-07 this DOES enforce the shape on both backends. `indexes` names the dimensions the returned dataset is indexed by, in order: variables laid out on any other dimension are dropped, unused dimensions are dropped with their coordinates, the survivors are transposed onto `indexes`, and requesting a dimension the data does not have raises `ValueError`. Passing `None` (the common no-arg call) means "no shape request, return as-is". `XrBackend.get_xarray_dataset` previously ignored the argument entirely (its body was `return self.data`), so the convention was carried only by `from_raw_data()` densifying onto the pinned axes; the knock-on was that `BaseDataset.time_interval` raised under `XrBackend`. Both are fixed and locked by `tests/test_backend_indexes.py`. See `example/backend.md` and `example/dataset.md`.
- Purpose: Typed, serializable (`to_dict()`/`asdict`) parameter bags threaded through every domain object's constructor; each object's `config` property setter mutates the config on assignment (injecting inherited dates, resolved factor names, etc.) rather than the config being immutable.
- Examples: `DatasetConfig`, `FactorConfig`, `DLConfig`, `MLConfig`.
- Pattern: Dataclass + `to_dict()`, consumed both for object construction and for JSON-serialized checkpoint metadata (`base/model.py:_save_model`).
- Purpose: Factors are defined as declarative dataflow graphs (`Builder`/`Input`/`Output`/`Op` composition), not imperative pandas/numpy transforms, then compiled to native code for performance.
- Examples: `factor/alpha101.py:_get_factor_func`, `my_ops/preprocess.py:WindowedZScore`.
- Pattern: Subclass `FactorKunQuant`, implement `_get_factor_func()` returning a `KunQuant.Stage.Function`.
## Entry Points
- Location: `main.py` (repo root).
- Triggers: Manual `python main.py` (or `uv run main.py`).
- Responsibilities: None currently — 7-line `uv init` stub (`def main(): print("Hello from quantlab!")`). Not wired into any other module in the codebase.
- `scripts/wrds/index.py` — one index's point-in-time CRSP daily bars plus its membership panel (`--index sp500|nasdaq100 --start [--end] [--refresh] [--data-dir]`).
- `scripts/wrds/market.py` — the CRSP daily market plus its listing panel (`--start [--end] [--security-filter] [--refresh] [--data-dir]`); stores are `wrds_crsp_market_*`.
- `scripts/wrds/etf.py` — one store per ETF by PERMNO (`--etf spy,qqq,name=PERMNO --start [--end] [--refresh] [--data-dir]`).
- `scripts/wrds/nbbo.py` — TAQ NBBO quotes resampled into a bar panel (`--symbols|--index, --start [--end] [--interval] [--session HH:MM-HH:MM] [--refresh] [--data-dir]`).
- Every script always converts to Zarr, clips `--end` (default today) to the product's last date, checks entitlement before downloading and closes the shared WRDS session in a `finally`. `scripts/` is not on the pytest `pythonpath`: `scripts/wrds/` must never be importable, because it would shadow the `wrds` PyPI package. Tiingo, Alpaca and Binance have library interfaces only.
- `examples/` — runnable walkthroughs (`quickstart.py`, `build_panel.py`, `train_model.py`, `backtest.py`, `inspect_data_sources.py`, `wrds_us_equity/`).
- `test.py` — untracked ad hoc scratch script at the repo root.
- **There is no single unified CLI/entry point** — each script independently constructs its own configs and imports the layers it needs.
## Architectural Constraints
- **Threading:** Single-process, but KunQuant factor computation explicitly uses a configurable multi-thread executor (`kr.createMultiThreadExecutor(self.config.njobs)`, default `njobs=128` in `FactorConfig`), and cross-validation folds can run in parallel threads via `joblib.Parallel(backend="threading")` (`base/model.py:train_cv`).
- **Global state:** None at module level observed (no module-level singletons/mutable globals); state is instance-scoped on `Dataset`/`FactorKunQuant`/`BaseModel` objects.
- **Hardcoded paths:** `config/__init__.py` now holds only `stock_kline_config`, `stock_acquisition_config` and `universe_config`, all rooted in `get_data_root()` (the Binance spot, Alpha101/Alpha158, momentum, constituent and spot-label factories were deleted 2026-09-25). The untracked `test.py` hardcodes absolute macOS paths (`/Users/daizhaorong/projects/quantlab/...` and `/home/zhrdai/projects/crypto_quant/...` for checkpoint loading); the WRDS scripts take the root from `--data-dir` / `QUANTLAB_DATA_DIR` / the repo `data/` directory.
- **Circular imports:** None observed; the layering (`base` → `dataset`/`factor`/`label` → `dl_model`/`ml_model` → `backtest` (`quantlab/backtest/`)) is consistently one-directional based on import statements read.
- **No `__init__.py` re-exports, and three `__init__.py` files that ARE the module:** The LAYER packages — `base/`, `factor/`, `label/`, `dl_model/`, `ml_model/`, `my_ops/`, `utils/`, `enums/`, and `acquisition/` and `dataset/` themselves — still have empty `__init__.py` files, and every import of them spells the full dotted path to the implementation module (e.g. `from factor.alpha101 import Alpha101SpotKline`, never `from factor import Alpha101SpotKline`). Three packages are different: `quantlab/acquisition/wrds/`, `quantlab/dataset/crsp/` and `quantlab/dataset/nbbo/`, where `__init__.py` IS the entry module — the file that was `wrds.py` / `crsp.py` / `nbbo.py`, moved by `git mv`, not a re-export shim written over it. What that buys: `from quantlab.dataset.crsp import CrspStockDataset` is ONE name for one subsystem (the spelling is byte-identical before and after the move), and the subsystem's parts group by directory (`crsp/membership.py`, `crsp/tickers.py`) instead of by a shared filename prefix, so a fourth CRSP module is a file rather than a naming convention. What it costs: importing any SUBMODULE runs the entry module first — measured at **+0.99s / +196 modules** on `quantlab/base/backtest.py`, which imports `quantlab.dataset.crsp.tickers`, and it is why importing a WRDS provider now loads the registry when it used to not. The distinction being adopted is "the `__init__` IS the module", never "the `__init__` re-exports other modules"; do not add a re-export list to any `__init__.py`.

  Since 260922-lu2 there are **three** kinds of package here, not two. The third is `_support/` — `quantlab/dataset/_support/` and `quantlab/acquisition/_support/`. These are neither layer packages nor entry-module packages: they are PRIVATE, their `__init__.py` files are empty, and they exist so the layer directory above them reads as a menu (see the layout rule below). Import their contents by full dotted path like a layer package (`from quantlab.dataset._support.masking import UniverseMask`); the leading underscore is the whole signal that nothing outside that layer should be reaching in.

- **Five `__init__.py` files are 0 bytes, and one guarantee rides on three of them:** a non-empty package `__init__` runs on EVERY import beneath it, and the structural guard that keeps acquisition clients out of the credential-free read surface is an `ast` scan of the guarded module's OWN source plus a `vars()` sweep — neither of which can see a transitive import dragged in by an `__init__`. So:

  | file | carries |
  |---|---|
  | `quantlab/__init__.py` | `quantlab.universe`, `quantlab.registry` and `quantlab.backend`, which are imported directly by the read surface |
  | `quantlab/acquisition/__init__.py` | `alpaca` / `tiingo` / `wrds`, and everything under `_support/` |
  | `quantlab/acquisition/_support/__init__.py` | the read surface — `inspector.py` now sits one package deeper, so THREE `__init__`s run ahead of it |
  | `quantlab/dataset/__init__.py` | consistency only; no guarantee rides on it |
  | `quantlab/dataset/_support/__init__.py` | consistency only; no guarantee rides on it |

  Enforced by `tests/test_source_inspector.py` (`test_inspector_binds_no_client`, asserting all three acquisition-chain files). The volume guard that used to give `quantlab.universe` its own import-order rule, and `tests/test_volume_guard.py` that locked it, were removed 2026-09-25 (`docs/adr/0001-no-download-volume-guard.md`); the rationale for the empty files is in `docs/developer-guide/internals.md`.

- **Every top-level entry of `quantlab/dataset/` is a dataset; every top-level entry of `quantlab/acquisition/` is an acquisition** (260922-lu2). Browsing either directory is a menu of complete, usable things — `dataset/` shows `spot.py`, `stock.py`, `constituent.py`, `crsp/`, `nbbo/`; `acquisition/` shows `alpaca.py`, `tiingo.py`, `wrds/`. Support code goes in `_support/`, or — if it is really its own LAYER — becomes a `quantlab/` sibling. Three modules became siblings, each for a measured reason:

  - `quantlab/backend.py` (`XrBackend`/`PlBackend`) — imported by seven modules across four layers (`config/__init__.py`, `universe.py`, `factor/universe_filter.py`, and `base/data.py`/`factor.py`/`model.py`/`backtest.py`). No single layer owns it.
  - `quantlab/registry.py` — the vendor registry is the thing an operator surface asks "what can this project download"; it is not itself an acquisition, and it imports all three vendors at its bottom.
  - `quantlab/universe.py` — the point-in-time symbol universe. Deliberately a flat module rather than a `universe/` package (D-1): a package would put a second `__init__` on its import path.

- **The `base/X.py` ↔ `<layer>/X.py` pairing: the mirrored filename was never the rule** (260922-lu2 D-2). **The ABC lives in `quantlab/base/`. The concrete implementation lives with its CONSUMERS** — not in a directory that mirrors the ABC's filename. The mirror was a coincidence of the first two cases. All three cases today:

  - `base/backend.py:ModelBackend` ↔ `ml_model/backend.py:MlBackend` — only the model layer consumes it, so it lives in the model layer. The filename still matches; that is incidental.
  - `base/backend.py:DataBackend` ↔ `quantlab/backend.py` — seven importers across four layers (measured above), so it is a top-level sibling and the filename no longer mirrors a directory.
  - `base/constituent.py` ↔ `dataset/constituent.py` — untouched, because a constituent dataset genuinely IS a dataset.

  Considered and NOT done: by the same "every file is one complete thing" logic, `quantlab/ml_model/backend.py` is support rather than a model. 260922-lu2 was scoped to `dataset/` and `acquisition/`; expanding it would have been scope creep. Recorded so the next reader sees it was weighed, not missed.
## Anti-Patterns
### Duplicated helper logic between script and library code
### Broken/incomplete backtest helper committed as-is
**Resolved:** the helper (`vecbt/bt.py:backtest_from_signals`) was retired in phase 03.7 (D-31), and `quantlab/backtest/` replaced it. The heading is kept as history.
## Error Handling
- Config setters validate/derive values eagerly (e.g. `FactorKunQuant.config` setter auto-fills `start_date`/`end_date`/`factor_names` if unset) rather than deferring to call time.
- Unimplemented/partial functionality is signaled by raising inside the method body rather than via `NotImplementedError`-only stubs. (The former backtest guard that refused `benchmark_dataset` (D-08) was removed when benchmark comparison landed on 2026-09-25: the slot now takes a single-symbol `MarketDataset`, bought and held on the strategy's bars and reported in the `benchmark`/`relative` metric blocks and the report's NAV, excess-return and excess-drawdown rows. The former model-layer guard `DLModel._fit(backtest=True)` was deleted in phase 03.7, D-37.)
- A model given the wrong config class raises `TypeError` as the first statement of the `BaseModel.config` setter, before any factor/label is touched; `load()` rejects a checkpoint whose suffix differs from the variant's `checkpoint_suffix` before building a model.
## Cross-Cutting Concerns

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues (`ZhaorongDai/quantlab`), operated through the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles map one-to-one to labels of the same name (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` at the repo root plus `docs/adr/`, both created lazily by `/domain-modeling`. See `docs/agents/domain.md`.
