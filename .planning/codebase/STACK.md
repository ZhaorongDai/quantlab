# Technology Stack

**Analysis Date:** 2026-09-04

## Languages

**Primary:**
- Python — `requires-python = ">=3.13"` per `pyproject.toml`. The system `python3` currently resolves to 3.9.6 (`python3 --version`), so a 3.13 interpreter must be provisioned via `uv` before the project will run (`uv` 0.8.14 is installed at `~/.local/bin/uv`).

**Secondary:**
- YAML — instrument/venue metadata (`config/instruments.yaml`).
- Jupyter (`.ipynb`) — exploratory/scratch work, e.g. `test_nt.ipynb`.

## Runtime

**Environment:**
- CPython >=3.13 (declared, not currently installed as default `python3` on this machine).
- GPU/CUDA expected for deep-learning model code: `base/model.py` calls `torch.cuda.manual_seed_all` and `BaseModel.device` selects `"cuda"` when available, falling back to `"cpu"`.

**Package Manager:**
- `uv` (evidenced by `uv.lock` and the bare `pyproject.toml` layout uv generates).
- Lockfile: present (`uv.lock`) but **stale/mismatched** — it locks only `bottleneck`, `numpy`, and a self-reference to a project named `crypto-quant` (not `quantlab`), and declares `requires-python = ">=3.12"` (vs. `>=3.13` in `pyproject.toml`). This lockfile predates the current `pyproject.toml` and does not reflect the packages actually imported by the code (see Key Dependencies below). Running `uv sync` today will not install a working environment.

## Frameworks

**Core:**
- No web/API framework. This is a research/trading codebase (data pipeline + ML models + backtesting), not a service.
- **PyTorch** (`torch`) — deep-learning models (`dl_model/mlp.py`, `dl_model/rnn.py`, `dl_model/rnn_classification.py`), trained through the shared `base/model.py:BaseModel` training loop (`DataLoader`/`TensorDataset`).
- **scikit-learn** (`sklearn.metrics`) — evaluation metrics (accuracy, F1, ROC-AUC, R², RMSE, etc.) used inside DL training loops, not for model fitting itself.
- **KunQuant** — JIT-compiled factor computation graph library. Used throughout `base/factor.py`, `factor/alpha101.py`, `factor/alpha158.py`, `label/spot.py`, `my_ops/preprocess.py` to build and compile (`cfake.compileit`) high-performance alpha factor pipelines (`KunRunner`, `Function`, `Builder`, `Op`, `Stage`).
- **Nautilus Trader** (`nautilus_trader`) — dual role: (1) data model / `ParquetDataCatalog` for storing bar/instrument data (`base/data.py`, `dataset/spot.py`), and (2) live/backtest trading engine — `backtest/test_strategy.py` implements a `nautilus_trader.trading.strategy.Strategy` subclass.
- **vectorbt** (`vectorbt`) — vector-based backtesting/portfolio simulation, used in `vecbt/bt.py` (incomplete stub) and `test.py` (`vbt.Portfolio.from_signals`).

**Testing:**
- None detected. No `pytest`/`unittest` configuration, no test runner dependency, no `tests/` directory. Files named `test.py` and `test_nt.ipynb` at the repo root are ad hoc exploratory scripts/notebooks, not an automated test suite.

**Build/Dev:**
- No linter/formatter config detected (no `.eslintrc`, `ruff.toml`, `.flake8`, `pyproject.toml` `[tool.ruff]`/`[tool.black]` sections).
- No CI configuration (no `.github/workflows`, no other CI YAML).

## Key Dependencies

**Critical (imported throughout the codebase but absent from `pyproject.toml` dependencies and largely absent from `uv.lock`):**
- `numpy`, `pandas`, `polars`, `xarray` — the core numerical/tabular/labeled-array stack. `xarray.Dataset` (dims `timestamp`, `symbol`) is the canonical in-memory data representation passed between dataset, factor, label, and model layers.
- `torch` — model definition and training (`dl_model/*`).
- `KunQuant` — compiled factor computation (`factor/*`, `base/factor.py`, `label/spot.py`, `my_ops/preprocess.py`). Appears to be a specialized/possibly local or pinned package, not a mainstream PyPI package with a standard lockfile entry.
- `nautilus_trader` — trading engine, data catalog, instrument/currency model (`dataset/spot.py`, `backtest/test_strategy.py`, `utils/nautilus.py`).
- `vectorbt` — signal-based backtesting (`vecbt/bt.py`, `test.py`).
- `wandb` — experiment tracking, initialized in every training run (`base/model.py:_init_wandb`).
- `loguru` — logging throughout (`base/data.py`, `base/factor.py`, `utils/timer.py`, `utils/nautilus.py`, `utils/binance.py`).
- `joblib` — parallelism (`Parallel`/`delayed` for CV folds and nautilus bar conversion) and non-torch model persistence (`joblib.dump`/`load` in `base/model.py`, `ml_model/backend.py`).
- `tqdm` — progress bars across data/factor/CV loops.
- `bottleneck` — the one dependency actually declared/locked (`pyproject.toml`/`uv.lock` under the old `crypto-quant` name); likely used for fast rolling/window numpy ops (not directly observed via `import` grep, may be an `xarray`/`pandas` accelerator dependency).

**Infrastructure / integration-adjacent:**
- `requests` — Binance REST calls (`utils/binance.py`, `get_binance_instruments.py`).
- `PyYAML` (`yaml`) — reading/writing `config/instruments.yaml`.
- `tiingo` — Tiingo market-data API client (`scripts/download_stock_data_from_tiingo.py`).
- `psutil` — memory-usage diagnostics (`read_mock_data_sink.py`).
- `plotly` (`plotly.io`) — backtest result visualization (`test.py`).
- `decimal` (stdlib) — precise price/fee representation in instrument config (`dataset/spot.py`, `utils/nautilus.py`).

## Configuration

**Environment:**
- No `.env` file present at the repo root.
- `scripts/download_stock_data_from_tiingo.py` reads a Tiingo API key — the script comment says to set `TIINGO_API_KEY` as an environment variable, but the script as written **hardcodes an API key literal** in the `config["api_key"]` assignment instead of reading from the environment. Treat this file as containing a leaked credential.
- No other secret/credential files detected (no `credentials.json`, `.npmrc`, private keys).

**Build:**
- `pyproject.toml` — project metadata only (`name`, `version`, `readme`, `requires-python`, empty `dependencies = []`). No `[tool.*]` sections, no build-system customization, no optional dependency groups.
- `uv.lock` — present but stale (see Runtime/Package Manager above); does not currently reflect a resolvable, working dependency set for this codebase.

## Platform Requirements

**Development:**
- macOS (current dev host is Darwin/arm64 per environment) or Linux — code contains Linux-style absolute paths hardcoded into config factories (see `config/__init__.py`, e.g. `/home/zhrdai/projects/crypto_quant/...`), implying the primary development/training environment is a Linux workstation, not this machine.
- A working `uv`-managed Python 3.13 environment must be created and `pyproject.toml` dependencies must be reconciled with actual imports before the code can run; currently `uv sync` alone is insufficient.

**Production:**
- No deployment target detected (no Dockerfile, no cloud config, no server entry point). This is a local research/trading pipeline intended to run on a workstation/server with GPU access for model training and disk access to large local datasets (CSV/Parquet klines, zarr stores).

---

*Stack analysis: 2026-09-04*
