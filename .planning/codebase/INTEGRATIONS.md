# External Integrations

**Analysis Date:** 2026-09-04

## APIs & External Services

**Market Data:**
- Binance REST API (`https://api.binance.com/api/v3/exchangeInfo`, `.../ticker/24hr`) — fetches spot trading-pair rules (price/size precision, min notional, etc.) and 24h volume rankings.
  - Client: raw `requests` calls, no SDK.
  - Implementation: `utils/binance.py:_get_binance_exchange_info`/`get_instrument_info`, duplicated (near-identically) in `get_binance_instruments.py` at the repo root.
  - Auth: none required — these are public Binance endpoints. No API key used.
  - Also referenced indirectly by `utils/nautilus.py:_load_instrument_config`, which falls back to a live Binance fetch when a symbol is missing from `config/instruments.yaml`.
- Tiingo API — historical daily stock price data for NASDAQ tickers.
  - Client: `tiingo.TiingoClient` SDK.
  - Implementation: `scripts/download_stock_data_from_tiingo.py`.
  - Auth: API key. **The script hardcodes a literal API key value in `config["api_key"]` instead of reading `TIINGO_API_KEY` from the environment as its own comment instructs** — this is a credential embedded directly in source code, not merely referenced by env var name.

## Data Storage

**Databases:**
- None (no SQL/NoSQL database client or connection string detected).

**File Storage:**
- Local filesystem only. All data paths observed are local absolute paths (Linux-style, e.g. `/home/zhrdai/projects/crypto_quant/data/...` in `config/__init__.py`, and Mac-style, e.g. `/Users/daizhaorong/projects/quantlab/...` in `train_model.py`/`test.py`), indicating per-developer-machine hardcoded paths rather than a shared/networked or cloud storage layer.
- **Zarr** — primary on-disk format for time-indexed `xarray.Dataset` objects (klines, factors, labels). Read/write handled by `dataset/backend.py:XrBackend` (`xr.open_dataset` / `Dataset.to_zarr`).
- **Parquet** — used for (a) raw stock data ingestion (`dataset/stock.py` reads `.pqt` files via `polars.scan_parquet`) and (b) Nautilus Trader's `ParquetDataCatalog`, written via `base/data.py:Dataset.to_nautilus`/`_write_catalog`.
- **CSV** — raw Binance kline downloads consumed by `dataset/spot.py:SpotKlineDataset._raw_data_to_xr` (`polars.scan_csv`).

**Caching:**
- None detected.

## Authentication & Identity

**Auth Provider:**
- None — this is not a multi-user application; there is no login/session/auth layer. "Auth" here is limited to third-party API credentials (Tiingo key, see above) and none for Binance's public endpoints.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry/Rollbar/equivalent).

**Logs:**
- `loguru` used for structured console logging across data/factor/model code (`base/data.py`, `base/factor.py`, `dataset/spot.py`, `dataset/stock.py`, `utils/timer.py`, `utils/binance.py`, `utils/nautilus.py`). No log aggregation/shipping configured — logs go to stdout/stderr only.
- **Weights & Biases (`wandb`)** — experiment tracking for all model training runs. `base/model.py:_init_wandb` calls `wandb.init(project=..., name=..., config=self.get_config())` at the start of every `train()`/`train_cv()` fold, and per-epoch metrics are logged via `self._wandb_recorder.log(...)` from each model subclass (`dl_model/rnn.py`, `dl_model/rnn_classification.py`, `dl_model/mlp.py`). This requires a wandb account/API key to be configured in the environment (not found hardcoded — presumably relies on the standard `WANDB_API_KEY` env var or local `wandb login`).

## CI/CD & Deployment

**Hosting:**
- None — no deployment target, container, or cloud infra config found.

**CI Pipeline:**
- None — no `.github/workflows`, no other CI configuration.

## Environment Configuration

**Required env vars:**
- `TIINGO_API_KEY` — intended per the comment in `scripts/download_stock_data_from_tiingo.py`, but the script does not actually read it (see APIs section — key is hardcoded instead).
- `WANDB_API_KEY` (implicit) — required for `wandb.init()` to authenticate; not referenced explicitly in code, relies on wandb's own environment/config resolution.
- No `.env` file exists in the repo; no `.env.example` either.

**Secrets location:**
- No dedicated secrets file/directory. The one credential found is embedded directly in `scripts/download_stock_data_from_tiingo.py` (see above) — this should be treated as compromised/rotated and moved to an environment variable if the code is to be reused.

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None (no webhook dispatch code detected). Nautilus Trader's live-trading `Strategy` (`backtest/test_strategy.py`) submits orders via its internal `order_factory`/`submit_order` API against the configured venue/exchange adapter, but no explicit outbound webhook integration is present in this repo.

---

*Integration audit: 2026-09-04*
