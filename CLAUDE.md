<!-- GSD:project-start source:PROJECT.md -->
## Project

**quantlab**

一个端到端的量化研究后端平台：从多市场、多频率行情数据出发，经因子计算、收益预测、组合优化，生成目标持仓并完成回测，全流程通过配置文件驱动、可复现。当前阶段只做后台，面向未来平台化（服务化、多用户、因子/模型在线编辑与测试、网页前端）预留架构空间，但不在本阶段实现。

**Core Value:** 一条打通的、config 驱动可复现的量化流水线（数据→因子→收益模型→组合优化→目标持仓→回测→结果），模块间用清晰的输入输出契约组合，任何一环都能独立替换/扩展而不需要推倒重来。

### Constraints

- **数据格式**: 模块间统一使用 xarray（Zarr 落盘），不使用 DataFrame 作为流水线层间传输格式；模型训练直接消费 xarray — 用户明确要求，是贯穿整个流水线的硬约束
- **因子计算后端**: 双后端支持——KunQuant（批量 + 流式，保留未来实时数据接入能力）为主，Polars 为新因子的补充计算路径（仅批量，不需要流式）；能用 xarray/KunQuant 完成的处理，优先不用 Polars — 用户明确的技术选型优先级
- **回测技术栈**: 向量化回测优先用 vectorbt 打通；事件驱动回测（NautilusTrader）作为预留扩展能力，非 v1 交付重点。原型 `backtest/test_strategy.py` 已于 2026-09-07 删除（早于当前 Dataset/Factor 契约），重启时按当前契约重建，不复活旧原型
- **凭证安全**: API Key 等敏感信息一律通过环境变量读取，不硬编码 — 现有代码已经因硬编码 Tiingo Key 造成一次真实泄露
- **可复现性**: 全流程参数尽量通过配置文件驱动 — 用户明确要求，服务于实验可复现
- **架构契约**: 数据模块输出数据、因子模块输出因子、收益模型输出未来收益/收益排名预测、组合优化模型输出每个标的目标持仓百分比——各模块通过清晰的输入输出契约组合 — 便于未来插拔式扩展与平台化
- **包管理**: 使用 `uv` — 用户明确要求，延续现有项目的包管理方式
- **范围**: 当前阶段只实现后台，不做网页前端 — 用户明确排除
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python — `requires-python = ">=3.13"` per `pyproject.toml`. The system `python3` currently resolves to 3.9.6 (`python3 --version`), so a 3.13 interpreter must be provisioned via `uv` before the project will run (`uv` 0.8.14 is installed at `~/.local/bin/uv`).
- YAML — instrument/venue metadata (`config/instruments.yaml`).
- Jupyter (`.ipynb`) — exploratory/scratch work, e.g. `test_nt.ipynb`.
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
- **Nautilus Trader** (`nautilus_trader`) — currently used ONLY as a data model / `ParquetDataCatalog` for storing bar/instrument data (`base/data.py`, `dataset/spot.py`). Its live/backtest trading engine is not used by any code in the repo: the only `Strategy` subclass, `backtest/test_strategy.py`, was deleted 2026-09-07.
- **vectorbt** (`vectorbt`) — vector-based backtesting/portfolio simulation, used by `quantlab/backtest/engine_vectorbt.py` (`VectorBtBacktester`: `Portfolio.from_orders` with target-percent weights) and by the ad hoc `test.py` (`vbt.Portfolio.from_signals`). The former `vecbt/bt.py` signal helper was retired in phase 03.7 (D-31).
- None detected. No `pytest`/`unittest` configuration, no test runner dependency, no `tests/` directory. Files named `test.py` and `test_nt.ipynb` at the repo root are ad hoc exploratory scripts/notebooks, not an automated test suite.
- No linter/formatter config detected (no `.eslintrc`, `ruff.toml`, `.flake8`, `pyproject.toml` `[tool.ruff]`/`[tool.black]` sections).
- No CI configuration (no `.github/workflows`, no other CI YAML).
## Key Dependencies
- `numpy`, `pandas`, `polars`, `xarray` — the core numerical/tabular/labeled-array stack. `xarray.Dataset` (dims `timestamp`, `symbol`) is the canonical in-memory data representation passed between dataset, factor, label, and model layers.
- `torch` — model definition and training (`dl_model/*`).
- `xgboost` — tree-model future-return regression (`ml_model/xgb.py:XGBoostRegressor`), trained with `xgb.train` and XGBoost's native `EarlyStopping(save_best=True)`.
- `scipy` — `scipy.stats.rankdata` for the cross-sectional RankIC in `utils/metrics.py`; declared directly in `pyproject.toml`.
- `KunQuant` — compiled factor computation (`factor/*`, `base/factor.py`, `label/spot.py`, `my_ops/preprocess.py`). Appears to be a specialized/possibly local or pinned package, not a mainstream PyPI package with a standard lockfile entry.
- `nautilus_trader` — data catalog, instrument/currency model (`dataset/spot.py`, `utils/nautilus.py`). The trading engine itself is unused.
- `vectorbt` — cross-sectional backtest engine (`quantlab/backtest/engine_vectorbt.py`, the only quantlab module importing it; `Portfolio.from_orders` with target-percent weights) and `test.py`. The former `vecbt/bt.py` helper was retired in phase 03.7 (D-31).
- `wandb` — experiment tracking, initialized in every training run (`base/model.py:_init_wandb`).
- `loguru` — logging throughout (`base/data.py`, `base/factor.py`, `utils/timer.py`, `utils/nautilus.py`, `utils/binance.py`).
- `joblib` — parallelism (`Parallel`/`delayed` for CV folds and nautilus bar conversion) and non-torch model persistence through `ml_model/backend.py:MlBackend` (the `.joblib` checkpoint backend of `base/model.py:MLModel`).
- `tqdm` — progress bars across data/factor/CV loops.
- `bottleneck` — the one dependency actually declared/locked (`pyproject.toml`/`uv.lock` under the old `crypto-quant` name); likely used for fast rolling/window numpy ops (not directly observed via `import` grep, may be an `xarray`/`pandas` accelerator dependency).
- `requests` — Binance REST calls (`utils/binance.py`, `get_binance_instruments.py`).
- `PyYAML` (`yaml`) — reading/writing `config/instruments.yaml`.
- `tiingo` — Tiingo market-data API client (`scripts/download_stock_data_from_tiingo.py`).
- `psutil` — memory-usage diagnostics (`read_mock_data_sink.py`).
- `plotly` (`plotly.io`) — backtest result visualization (`test.py`).
- `decimal` (stdlib) — precise price/fee representation in instrument config (`dataset/spot.py`, `utils/nautilus.py`).
## Configuration
- No `.env` file present at the repo root.
- `scripts/download_stock_data_from_tiingo.py` reads a Tiingo API key — the script comment says to set `TIINGO_API_KEY` as an environment variable, but the script as written **hardcodes an API key literal** in the `config["api_key"]` assignment instead of reading from the environment. Treat this file as containing a leaked credential.
- No other secret/credential files detected (no `credentials.json`, `.npmrc`, private keys).
- `pyproject.toml` — project metadata only (`name`, `version`, `readme`, `requires-python`, empty `dependencies = []`). No `[tool.*]` sections, no build-system customization, no optional dependency groups.
- `uv.lock` — present but stale (see Runtime/Package Manager above); does not currently reflect a resolvable, working dependency set for this codebase.
## Platform Requirements
- macOS (current dev host is Darwin/arm64 per environment) or Linux — code contains Linux-style absolute paths hardcoded into config factories (see `config/__init__.py`, e.g. `/home/zhrdai/projects/crypto_quant/...`), implying the primary development/training environment is a Linux workstation, not this machine.
- macOS dev host OpenMP clash: xgboost's wheel links Homebrew libomp while torch bundles its own, so a process mixing both segfaults or deadlocks. `tests/conftest.py` sets `OMP_NUM_THREADS=1` on darwin before any import (locked by `tests/test_macos_openmp_guard.py`); user scripts/notebooks on macOS that mix torch and xgboost must set it too. Linux is untouched. See `example/model.md`.
- A working `uv`-managed Python 3.13 environment must be created and `pyproject.toml` dependencies must be reconciled with actual imports before the code can run; currently `uv sync` alone is insufficient.
- No deployment target detected (no Dockerfile, no cloud config, no server entry point). This is a local research/trading pipeline intended to run on a workstation/server with GPU access for model training and disk access to large local datasets (CSV/Parquet klines, zarr stores).
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## System Overview
```text
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
| `WindowedZScore` | Custom KunQuant composite op for time-series factor normalization (`WindowedRobustStandardization` sat beside it with zero call sites and was deleted 2026-09-07) | `my_ops/preprocess.py` |
| `BaseModel` (abstract) | Framework-agnostic shared lifecycle: config type guard, factor/label collection, the public `train`/`train_cv`/`load`/`predict` (implemented once), checkpoint dir + `config.json`, the single CV fold generator `_cv_folds`, per-fold CV results and the `{cls}_cv_summary` W&B run | `base/model.py` |
| `DLModel` (abstract) | torch variant: device, `to_tensor`, DataLoader epoch loop with per-epoch early stopping and best-epoch rollback, `.pth` checkpoints; five tensor hooks | `base/model.py` |
| `MLModel` (abstract) | numpy variant for tree/ML models: one `_fit_model` call with the library's native early stopping, `_evaluate` writing `{split}_*` metrics to the W&B summary, `.joblib` checkpoints via `MlBackend`, resolved-hyperparameter recording; four hooks | `base/model.py` |
| `XGBoostRegressor` | Future-return regression with `xgb.train` + native `EarlyStopping(save_best=True)`, per-round W&B callback, sklearn-alias normalization, CV via inherited `train_cv` | `ml_model/xgb.py` |
| Panel metrics | Vectorized MSE/RMSE/MAE/R² and cross-sectional IC/RankIC over `[T, S]` panels | `utils/metrics.py` |
| `MLPRegressor`, `RNNRegressor`, `RNNClassifier` | Concrete torch model heads (MLP, GRU/LSTM regressor, GRU/LSTM classifier with auxiliary-label architecture) | `dl_model/mlp.py`, `dl_model/rnn.py`, `dl_model/rnn_classification.py` |
| `MlBackend` | joblib-based checkpoint persistence for `MLModel` heads | `ml_model/backend.py` |
| `BaseBacktester` (abstract) | Backtester ABC: config type guard, the two public entry points `run()` (model backtest) and `run_cv()` (replays a `train_cv` run's `cv_folds.json` as one stitched curve), bar-counted warm-up, D-17 in/out-of-sample split, metrics, run directory, data fingerprints; engine hooks `_simulate`, `_simulate_benchmark`, `_engine_stats`, `_period_returns_stats` | `quantlab/base/backtest.py` |
| `VectorBtBacktester` (abstract) | vectorbt engine: `Portfolio.from_orders` with target-percent weights, a signal at bar t fills at bar t+1's open (D-05), forced-liquidation records for delisted holdings; the only quantlab module importing vectorbt | `quantlab/backtest/engine_vectorbt.py` |
| `CrossSectionTopNSelector` | TopN long-only or disjoint long/short target weights on rebalance bars (D-03 weights contract) | `quantlab/backtest/selection.py` |
| `USEquityCrossectionSelectStockVectorBt` | Concrete US-equity backtester composing `US_EQUITY_MARKET`, the selector and the vectorbt engine; rebuilt from a run's `config.json` by `quantlab/utils/module.py:load_backtester_from_config`. It replaces the `vecbt/bt.py:backtest_from_signals` helper, retired in phase 03.7 (D-31) | `quantlab/backtest/us_equity.py` |
| Config factories | Hardcoded-path factory functions producing `DatasetConfig`/`FactorConfig` for spot klines, alpha101, alpha158, labels | `config/__init__.py` |
| `DatasetConfig`/`FactorConfig`/`DLConfig`/`MLConfig` | Dataclass configuration objects threaded through every layer | `base/config.py` |
## Pattern Overview
- Every domain object (`Dataset`, `FactorKunQuant`, `BaseModel`) shares the same lifecycle idiom: a `config` property setter that normalizes dates/names on assignment, a `read()`/`cal()`/`save()` trio for lazy-vs-eager data materialization, and `xarray.Dataset` as the universal in-memory exchange format between layers (always indexed by `timestamp` and `symbol`).
- Factor/label computation is offloaded to **KunQuant**, a compiled-graph engine (`Builder`/`Op`/`Function` → `cfake.compileit` → `KunRunner`), not plain numpy/pandas — factors are defined declaratively as op graphs, then JIT-compiled to native code and executed via a multi-thread executor (`kr.createMultiThreadExecutor`).
- Model layer (`base/model.py`) is a three-layer hierarchy: framework-agnostic `BaseModel` owns the public `train`/`train_cv`/`load`/`predict` and CV fold geometry; `DLModel` (torch, `[num_times, num_symbols, num_features]` tensors through `TensorDataset`/`DataLoader`) requires five hooks (`_init_model`, `_train_one_batch`, `_val_one_batch`, `_test_one_batch`, `_preprocess`); `MLModel` (numpy, native library early stopping, no epoch loop) requires four (`_init_model`, `_preprocess`, `_fit_model`, `_forward`). Each variant declares `config_cls` (`DLConfig`/`MLConfig`) and `checkpoint_suffix` (`.pth`/`.joblib`).
- Configuration is dataclass-based (not env-var or YAML-based, except `config/instruments.yaml` for exchange instrument metadata) and is **hardcoded with absolute filesystem paths per developer machine** rather than parameterized (see `config/__init__.py`).
- No dependency injection framework, no plugin registry beyond `utils/module.py:get_cls_from_path` (dynamic import-by-dotted-path used to reconstruct a `Dataset`/`Factor`/`Model` from a saved JSON config).
## Layers
- Purpose: Abstracts "how data is persisted" from "what the data means."
- Location: `dataset/backend.py` (concrete: `XrBackend`, `PlBackend`), `base/backend.py` (abstract `DataBackend`).
- Contains: `read`/`write`/`to_internal`/`filter_by_date`/`filter_by_symbol`/`get_xarray_dataset`/`get_lazyframe`.
- Depends on: `xarray`, `polars`, `pandas`.
- Used by: `Dataset` and `FactorKunQuant`, each of which owns a `self.data_backend` instance.
- Purpose: Converts raw external data (Binance CSV klines, Tiingo/NASDAQ parquet) into the canonical `xarray.Dataset` and persists it via a `DataBackend`. Also converts to KunQuant input arrays (`to_kunquant`) and Nautilus Trader bar/catalog objects (`to_nautilus`).
- Location: `base/data.py` (ABC `Dataset`), `dataset/spot.py` (`SpotKlineDataset`), `dataset/stock.py` (`StockDataset`, several methods unimplemented — raise `ValueError("Not finished")`).
- Depends on: Storage Backend layer, `nautilus_trader` model/persistence types, `utils/file.py`, `utils/nautilus.py`, `utils/timer.py`.
- Used by: Factor/Label layer (each `FactorConfig` embeds a `Dataset` instance) and directly by scripts (`test.py`, `cal.py`).
- Purpose: Computes engineered features (factors) and prediction targets (labels) from dataset data using compiled KunQuant graphs, in either batch mode (`cal()`, operates on a full historical window) or streaming mode (`cal_stream()`, incremental per-bar updates for live trading).
- Location: `base/factor.py` (ABC `FactorKunQuant`), `factor/alpha101.py`, `factor/alpha158.py`, `label/spot.py`, `my_ops/preprocess.py` (custom `WindowedCompositiveOp` subclasses).
- Depends on: Dataset layer (each factor config embeds a `Dataset`), `KunQuant`.
- Used by: Model layer (`DLConfig.factors`/`DLConfig.labels`).
- Purpose: Orchestrates the full training lifecycle — pulling factor/label data into a combined `xarray.Dataset` (`collect()`), splitting into train/val/test or rolling walk-forward CV windows (one `_cv_folds` generator for both variants and both sequential/parallel branches), training (DL: epoch loop with per-epoch early stopping; ML: one native-early-stopping fit), checkpointing (`.pth` via `torch.save` in `DLModel`, `.joblib` via `MlBackend` in `MLModel`), and W&B logging.
- Location: `base/model.py` (`BaseModel` / `DLModel` / `MLModel`), `base/config.py` (`DLConfig`/`MLConfig`), `dl_model/mlp.py`, `dl_model/rnn.py`, `dl_model/rnn_classification.py` (all `DLModel`), `ml_model/xgb.py` (`XGBoostRegressor`, an `MLModel`), `ml_model/backend.py` (`MlBackend`), `utils/metrics.py` (panel metrics).
- Depends on: Factor/Label layer, `torch`, `xgboost`, `wandb`, `sklearn.metrics`, `scipy`, `joblib`.
- Used by: Top-level scripts (`train_model.py`, `test.py`).
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
- Examples: `dataset/backend.py:XrBackend` (zarr/xarray), `dataset/backend.py:PlBackend` (parquet/polars).
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
- **No `__init__.py` re-exports, and three `__init__.py` files that ARE the module:** The LAYER packages — `base/`, `factor/`, `label/`, `dl_model/`, `ml_model/`, `my_ops/`, `utils/`, `enums/`, and `acquisition/` and `dataset/` themselves — still have empty `__init__.py` files, and every import of them spells the full dotted path to the implementation module (e.g. `from factor.alpha101 import Alpha101SpotKline`, never `from factor import Alpha101SpotKline`). Three packages are different: `quantlab/acquisition/wrds/`, `quantlab/dataset/crsp/` and `quantlab/dataset/nbbo/`, where `__init__.py` IS the entry module — the file that was `wrds.py` / `crsp.py` / `nbbo.py`, moved by `git mv`, not a re-export shim written over it. What that buys: `from quantlab.dataset.crsp import CrspStockDataset` is ONE name for one subsystem (the spelling is byte-identical before and after the move), and the subsystem's parts group by directory (`crsp/membership.py`, `crsp/tickers.py`) instead of by a shared filename prefix, so a fourth CRSP module is a file rather than a naming convention. What it costs: importing any SUBMODULE runs the entry module first — measured at **+0.99s / +196 modules** on `quantlab/base/backtest.py`, which imports `quantlab.dataset.crsp.tickers`, and it is why importing a WRDS provider now loads the registry when it used to not. The distinction being adopted is "the `__init__` IS the module", never "the `__init__` re-exports other modules"; do not add a re-export list to any `__init__.py`. `quantlab/acquisition/__init__.py` is still 0 bytes, and `quantlab/acquisition/registry.py:728-737` explains in prose why it must stay that way (a non-empty package `__init__` would run on every `import quantlab.acquisition.universe` and silently erode the volume guard's structural arm).
## Anti-Patterns
### Duplicated helper logic between script and library code
### Broken/incomplete backtest helper committed as-is
**Resolved:** the helper (`vecbt/bt.py:backtest_from_signals`) was retired in phase 03.7 (D-31), and `quantlab/backtest/` replaced it. The heading is kept as history.
## Error Handling
- Config setters validate/derive values eagerly (e.g. `FactorKunQuant.config` setter auto-fills `start_date`/`end_date`/`factor_names` if unset) rather than deferring to call time.
- Unimplemented/partial functionality is signaled by raising inside the method body rather than via `NotImplementedError`-only stubs consistently — `dataset/stock.py` uses `raise ValueError("Not finished")` for `_get_instrument`/`_xr_to_bars`/`_to_nautilus`, while `quantlab/base/backtest.py:BaseBacktester`'s config setter raises `NotImplementedError` when a `benchmark_dataset` is supplied, because benchmark comparison waits for directly-downloaded index price data (D-08). (The former model-layer guard `DLModel._fit(backtest=True)` was deleted in phase 03.7, D-37.)
- A model given the wrong config class raises `TypeError` as the first statement of the `BaseModel.config` setter, before any factor/label is touched; `load()` rejects a checkpoint whose suffix differs from the variant's `checkpoint_suffix` before building a model.
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
