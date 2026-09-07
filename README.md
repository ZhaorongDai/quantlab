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

## Index Constituent Panels

Point-in-time index membership, as pipeline data. A panel is an `xarray.Dataset` with dims
`timestamp` x `symbol` and exactly one `bool` variable, `is_member`, persisted to Zarr under
`data/us_equity/1d/` -- **one store per index**, never a shared one.

| Index | Class | Config factory | Store | Coverage starts |
|-------|-------|----------------|-------|-----------------|
| S&P 500 | `dataset/constituent.py:SP500ConstituentDataset` | `sp500_constituent_config()` | `sp500_constituent.zarr` | `1976-07-01` |
| Nasdaq-100 (NDX) | `dataset/constituent.py:Nasdaq100ConstituentDataset` | `nasdaq100_constituent_config()` | `nasdaq100_constituent.zarr` | `2007-02-01` |

The coverage start is the earliest date the underlying change log actually covers, and it is
enforced in both directions: a panel's left edge is clamped up to it (so the panel never
contains an all-False region where the truth is *unknown*), and
`UniverseCatalog.get_symbols_as_of(category, date)` **raises** for an earlier date rather than
answering with an incomplete roster. The two indices' starts are ~31 years apart, which is why
they get two stores: unioning them onto one timestamp axis would imply 1976 Nasdaq-100 coverage
that does not exist. A consumer that wants both opens both and joins on the intersection of
their timestamp axes.

Two axis conventions matter to anyone consuming a panel:

- **Membership intervals are CLOSED on both ends.** A symbol removed with effective date `D`
  reads `True` on `D` and `False` on `D+1`. This matches `get_symbols_as_of()`'s
  `start_date <= as_of_date` / `end_date >= as_of_date` comparison exactly, so the panel and
  that query never disagree by a day at a removal.
- **The `timestamp` axis is CALENDAR days, not trading days.** It is a contiguous daily range
  including weekends and holidays, whose values carry the last trading day's membership
  forward. A join against OHLCV data, which only has trading days, must reindex or `.sel()`
  the panel onto the price panel's timestamps rather than assuming the axes align.

**Survivorship bias.** The symbol axis is the ALL-TIME union of every symbol that was ever a
member, computed before any date filtering. A delisted former member is therefore a real,
mostly-False column rather than an absent one -- an absent column is indistinguishable from a
symbol that was never a member, which is exactly how survivorship bias re-enters.

Building and reloading a panel:

```python
from config import nasdaq100_constituent_config, sp500_constituent_config
from dataset.constituent import Nasdaq100ConstituentDataset, SP500ConstituentDataset

# Build from source and persist (one Zarr store per index).
SP500ConstituentDataset(sp500_constituent_config()).from_raw_data().save()
Nasdaq100ConstituentDataset(nasdaq100_constituent_config()).from_raw_data().save()

# Reload later, with no network access.
panel = (
    Nasdaq100ConstituentDataset(nasdaq100_constituent_config())
    .read()
    .get_xarray_dataset()
)
members_on_a_day = panel["is_member"].sel(timestamp="2020-06-15")
```

`from_raw_data()` performs live HTTP requests to public, unauthenticated sources (Wikipedia
change logs plus a current-constituent anchor per index); no API key is involved. If a change-log
fetch or parse fails, the fetcher falls back to its cached snapshot under
`data/reference/_cache/` and does **not** overwrite that cache, so one bad parse cannot poison
future runs. `read()` touches only the local Zarr store.

**Adding a third index** requires no change under `base/`:

1. Add a data-only `IndexMembershipFetcher` subclass in `acquisition/universe.py` (nine class
   constants + `fetch_anchor()`). The change-log parse is inherited: you declare the source's
   `EXPECTED_SOURCE_HEADER`/`DATE_HEADER` rather than writing a `_parse_changes_table()`, which
   is what makes header validation apply to every index by construction.
2. Add the category token to `UniverseCategory` in `enums/data.py`.
3. Register the fetcher in `UniverseCatalog.MEMBERSHIP_FETCHERS` so it inherits the coverage
   guard.
4. Add the CLI token to `ingest_tiingo.py`'s `_UNIVERSE_CATEGORY_MAP` (the `--universe` choices
   are derived from that map), or the category is unreachable from the only CLI consumer.
5. Subclass `IndexConstituentDataset` with `_pit_coverage_start()` and `_build_intervals()`.
6. Add a config factory in `config/__init__.py`.

Steps 2 and 3 are enforced together by `tests/test_universe.py`:
`test_catalog_build_emits_all_three_categories` asserts the built categories equal
`get_args(UniverseCategory)` exactly, so a fetcher registered without its enum token -- or an
enum token with no fetcher -- fails there;
`test_universe_category_literal_has_exactly_three_values` pins the literal itself.

## Project Structure

- `base/` -- Abstract base classes that define the layer contracts: `DataBackend`/`ModelBackend`
  (`backend.py`), `BaseDataset` and its market-data specialization `MarketDataset` (`data.py`),
  `IndexConstituentDataset` (`constituent.py`), the shared `Factor` base and its `FactorKunQuant`
  backend (`factor.py`), the `FactorPolars` backend (`factor_polars.py`), `BaseModel`
  (`model.py`), plus the dataclass configs (`config.py`: `BaseDatasetConfig`, `DatasetConfig`,
  `ConstituentDatasetConfig`, `BaseFactorConfig`, `FactorConfig`, `PolarsFactorConfig`,
  `DLConfig`, `MLConfig`).
- `dataset/` -- Concrete dataset/`DataBackend` implementations. `MarketDataset` subclasses:
  `SpotKlineDataset` (Binance spot klines), `StockDataset` (NASDAQ/Tiingo, partially
  implemented). `BaseDataset` subclasses that are deliberately *not* `MarketDataset`s --
  a membership panel has no bar or KunQuant representation -- `SP500ConstituentDataset` and
  `Nasdaq100ConstituentDataset` (`constituent.py`), see
  [Index Constituent Panels](#index-constituent-panels). Backends: `XrBackend` (Zarr-backed),
  `PlBackend` (Parquet/Polars-backed).
- `factor/` -- Concrete factor sets. Computed via KunQuant: `Alpha101SpotKline`,
  `Alpha101Stock`, `Alpha158SpotKline`, `Alpha158Stock`. Computed via Polars: `Momentum`
  (`momentum.py`, the worked example of the Polars backend). See
  [Factor Backends](#factor-backends).
- `label/` -- Forward-return prediction targets: `SpotReturn` (regression), `SpotBinaryReturn`
  (classification).
- `my_ops/` -- Custom KunQuant composite ops used inside factor/label graphs
  (`WindowedZScore`).
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
- `ingest_alpaca.py` -- the second US-equity source (03.2 D-10: parallel to Tiingo, not a
  replacement). Fetches daily bars, minute bars, quotes or trades from Alpaca Market Data into
  the vendor-namespaced raw path, then converts bars to Zarr. Requires `APCA_API_KEY_ID` and
  `APCA_API_SECRET_KEY`. `--frequency tick` stops after the raw shards land: the quotes/trades
  raw-to-xarray conversion needs an irregular event axis the dense `[timestamp, symbol]` panel
  cannot express, and arrives in phase 03.3 (D-18). Every entry point runs a pre-flight volume
  estimate first and refuses an over-budget fetch before constructing a client; `--force-volume`
  is the explicit override.
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
- `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` -- required to run `ingest_alpaca.py`. These are
  Alpaca **market-data** credentials only: they grant no trading or broker access, and there is
  no paper/live distinction to make, because Alpaca's market-data API does not have one (03.2
  D-15). Both are read from the environment inside `acquisition/alpaca.py`'s client constructor
  and are never assigned to a config dataclass -- `AcquisitionConfig.to_dict()` is `asdict()`
  and lands in persisted configs and in the JSON saved beside model checkpoints. Neither is
  accepted as a command-line argument, which would put it in shell history and in every process
  listing. Get a key pair from the Alpaca dashboard (https://app.alpaca.markets/).
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

`config/__init__.py` provides factory functions (`spot_kline_config`, `stock_kline_config`,
`sp500_constituent_config`, `nasdaq100_constituent_config`, `alpha101_config`,
`alpha158_config`, `spot_label_config`) that build these configs using paths derived from
`QUANTLAB_DATA_DIR` (or the repo-root `data/` default), so a fresh clone works without manual
path edits.
