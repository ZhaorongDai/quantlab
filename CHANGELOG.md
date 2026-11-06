# Changelog

quantlab follows [semantic versioning](https://semver.org/). While the version is 0.x, any
minor release may break the API, the config format or the store layout; there are no
compatibility shims or config migrations.

## 0.3.0 (2026-11-05)

### Added

- `LivePredictionStore.panel(start, end)`, the live predictions of a date range on
  `(timestamp, symbol)`, and `LivePredictionStore.check_owner(run)`, the reader's check that
  the store holds the predictions of that run's current checkpoint and labels.
- `BaseDataset.last_bar()`, the last bar of a dataset's calendar (or `None`);
  `quantlab.backtest.live.last_bar` answers with it.
- `BacktestRun.price_dataset`, the run's price dataset rebuilt once and shared by every reader
  of the open run, and `DecisionInputs.from_run` takes an open `BacktestRun` as well as a
  directory and reads its prices there, so a process reading the run itself opens it once and
  reads one price dataset.
- `TickerLookup.listings(symbols, day)` and `Listing`: the ticker and primary exchange of each
  symbol as of a day, `None` when unknown, for code that sends orders. The Sharadar ticker
  sidecar records each permaticker's current exchange from TICKERS on its current ticker's span,
  and `SharadarTickerLookup.listings` reads it; a merged dataset lists each symbol through the
  first input that knows its ticker.
- `LiquidConstituentDataset` (`quantlab.dataset.liquid`, config
  `LiquidConstituentDatasetConfig`): a liquid universe, the members of another membership
  panel (`members`, such as the us3000 `EstuConstituentDataset`) whose median dollar volume,
  the unadjusted `close` times `volume` of `prices`, over the `window` bars before t clears
  `floor`. Bar t is not read, so membership at t is known at t's open; an unpriced bar has no
  dollar volume, and fewer than `min_bars` priced bars, or a member the prices do not hold,
  is not a member (defaults `window=60`, `min_bars=45`); `price_floor`, off by default, also
  requires the previous bar's price. The prices must be daily; the panel runs to their last
  bar, days without a member included.
- A membership dataset may override `_pit_coverage_end(config)`, the last date its source has
  observed: the panel then runs to it, days on which no symbol is a member included, and an
  empty interval table gives a panel without symbols instead of raising. The default reads
  the horizon off the intervals as before.
- The `LibraryModel` training target `"cs_rank_gauss"`
  (`quantlab.model.torch_training.cs_rank_gauss`): each bar's rank-Gauss (van der Waerden)
  score of the label, `Phi^-1(rank / (n + 1))` with ties averaged, beside `"cs_rank"` and
  `"cs_zscore"`.
- The 192 factors of the price-volume factor list. `PriceVolumeDailyAlpha`
  (`quantlab.factor.predefined.price_volume_daily`) computes its 102 daily factors in one
  double-precision KunQuant graph: SEP computes `SEP_OUTPUTS`, the Massive day bars the
  trade-count `HIGH_DEAL_REVERSE` family, and `IDIO_VOL_FF3` reads a Fama-French CSV.
  `PriceVolumeIntradayStats` (`quantlab.factor.predefined.price_volume_intraday`) computes 15 session
  statistics of one-minute bars, and `PriceVolumeIntradayAlpha` (config `PriceVolumeIntradayConfig`) their
  1- to 60-session means, the list's 90 minute-bar factors. Both take a `ranked` option
  (`kwargs={"ranked": True}`, `PriceVolumeIntradayConfig.ranked`) giving market-wide percentile
  ranks, the input `NeutralizedFactor` should wrap.
  `examples/sharadar_us_equity/price_volume_factors.py` builds the whole-market stores, raw and
  size-neutral, `price_volume_us3000_report.py` reports them on the point-in-time us3000, and
  `price_volume_us3000_xgb.py` trains a walk-forward XGBoost on them and
  backtests top-N and TopkDropout books over the test folds (`run_cv`), without a risk
  model.
- `FactorStoreDataset` (`quantlab.dataset.factor_store`, config `FactorStoreDatasetConfig`):
  reads a factor store's Zarr panel as panel variables, under an optional prefix, so stored
  daily statistics merge beside prices. It shifts nothing: a store's value at day t must be
  known at t's close.
- `DatasetView` (`quantlab.dataset.view`), the root class of dataset views: with one input it
  answers every read member but `panel` from that input, and with several a member the view
  does not define raises. `RosterDataset`, `PermatickerDataset`, `MergedDataset` and
  `BadPrintMaskedDataset` inherit it.
- `FactorKunQuant.graph()`, the KunQuant graph a factor compiles, and
  `quantlab.factor.kunquant_nan`, the NaN-preserving stand-ins (`SetInfOrNanToValue`, `Clip`,
  `Max`, `Min`, `Select`, `missing_bars_only`), as public API.
- `examples/data_update/`: `sharadar_data.py` and `massive_data.py` download a vendor's data
  and build its stores on the first run and append the new days on a later one, so a daily
  cron job is the same command.

### Removed

- Dataset resampling. Dataset configs lose `resample_freq` and `resample_how`, and
  `resample()`, resampling on read and the sibling resampled store are gone;
  `resample_labels` and factor resampling stay. A saved config holding a removed field is
  refused with the unknown-key error.
- `sharadar_sp500_1d`: `build_stores` and `update_stores` no longer write the S&P 500 roster
  price store. Read the whole-market SEP store on a roster instead.
- The examples whose data is gone: `examples/wrds_us_equity/`, and
  `sharadar_us_equity/sp500_xgb.py`, `sp500_xgb_mvo.py` and `us3000_h1_mvo.py`.

### Changed

- The start date, end date, symbols and kwargs of a dataset config move to
  `RawDatasetConfig`, so `BaseDatasetConfig` holds only the store path and the name, and
  `FrameDatasetConfig` loses the build fields.
- A dataset with a delisting rule of its own declares the symbols it judges in
  `_own_delisting_symbols`, and a view applies the rule only to those. A leaf that overrides
  `delisting_bars` without it raises `TypeError` when its class is defined.
- The factor report treats infinite values as missing, puts tied factor values in the bucket
  of their average rank, and sorts the panels by timestamp. Without ties or infinite values
  the results are unchanged. A discrete factor can now leave a bucket empty: a bucket that is
  always empty has NaN cumulative and long-short metrics.
- `XrBackend.resample` groups with xarray instead of a pandas round trip, about 5x faster from
  minute bars to `"1d"` and about 2x slower to many short bars such as `"5m"`. NaN semantics
  are unchanged; float sums may differ in the last bits.
- `cache_dir` moves off `ConstituentDatasetConfig` onto a new subclass,
  `FileConstituentDatasetConfig` (`quantlab.dataset.config`), where it is required. The
  classes that read their membership source from local files take it
  (`SP500ConstituentDataset`, `Nasdaq100ConstituentDataset`, `CrspSP500ConstituentDataset`,
  `CompustatNasdaq100ConstituentDataset`, `CrspMarketConstituentDataset`,
  `SharadarSP500ConstituentDataset`); a membership derived from another store
  (`EstuConstituentDataset`, `CommonStockConstituentDataset`) takes
  `ConstituentDatasetConfig`, which no longer has the field. `IndexConstituentDataset`
  refuses a config that is not its `config_cls` with a `TypeError`. A saved config of a
  derived membership holding `cache_dir` no longer rebuilds: drop the key.
- The coverage hook of a membership dataset takes the config being normalised,
  `_pit_coverage_start(self, config)`, instead of reading state set on the dataset, so
  normalisation changes nothing on the dataset and a refused config reads no store. A
  subclass written against the old hook adds the parameter.

### Fixed

- A first bulk pull of Sharadar's SF2 failed: the vendor now writes `transactionvalue` with a
  decimal point (`12507.0`). It is parsed as a float and stored as the same `Int64`, so raw
  files already on disk need no change.
- `BarraStyle` with `risk_free_symbol` answers `calendar`, `bar_before` and `bar_after` on the
  bars of its price inputs. It used to include dates only the rate has, so a chunked compute
  left uninitialised rows for them and warm-up counted them. `BarraStyle.resample` and a
  config with `resample_freq` are refused.
- A view keeps a delisting rule of its input: `MergedDataset` passes on the rule of each input,
  so `BadPrintMaskedDataset` no longer drops it, and a view over a plain input no longer
  delists a symbol whose prices move to another input.
- `write_json_atomically` gives its file the mode a plain write would, as
  `write_parquet_atomically` does. Sidecars rewritten by the daily update became owner-only,
  and other users' reads of shared stores failed with `PermissionError`. Both writers set the
  mode through the open descriptor.
- Processes compiling the same KunQuant graph no longer remove each other's intact cache entry,
  which made a third process fail with "cannot be loaded". Entries are published under a
  per-key lock.
- A release panel whose staleness column is not one of its variables can widen again:
  `MassiveTenKChangesDataset` raised `KeyError('release_date')` the first time an update brought
  a new filer.

## 0.2.0 (2026-10-20)

### Added

- `UnscaledZScoreExpectedReturn` (`quantlab.portfolio.predefined.expected_return`, config
  `UnscaledZScoreExpectedReturnConfig`): an expected return model whose `mu` is the
  candidates' cross-sectional z-score of the prediction, unscaled, with no volatility tilt.
- Ranged downloads resume: an interrupted download keeps its `.part` file and a sidecar
  recording the size and the object identity (an S3 ETag), and the next attempt continues
  from where it stopped when the object is unchanged. The Massive flat-file client uses it.
  A download resumes only with an object identity (a call without one, such as the Sharadar
  bulk zip, starts over) and a well-formed sidecar naming only this download's range starts.
  A download that does not resume now also leaves its `.part` and `.part.json` behind after a
  transient failure, where it used to remove the `.part`; the next call overwrites them.
- The Massive flat-file client conditions every ranged GET on the ETag its HEAD read
  (`If-Match`): a file replaced mid-download (S3's 412) raises the new
  `MassiveObjectChangedError` inside the client, and the day starts over from a fresh HEAD
  instead of mixing two versions. When the client's retries run out it propagates: a
  `RuntimeError`, not a `MassiveTransportError`.

### Fixed

- `predict_live_bar` treats a store with bars inside the day (a one-minute store) as holding
  the daily bar t when it holds any bar of t's day, so daily factors read from minute stores
  no longer stop the live prediction for a missing bar.
- Sharadar ACTIONS: the seven `acquisitionbesteffort*` action names Sharadar started
  publishing on 2026-10-19 are mapped (cash, stock or other), so the actions table builds again.
- `BlackLittermanOptimizer` warns for `UnscaledZScoreExpectedReturn` used as total-return views
  (`views_relative_to_prior=False`), as it does for the other centred alphas.

## 0.1.0 (2026-10-18)

The first release, published to PyPI as `quantlab-py` and imported as `quantlab`.

### What it does

One config-driven pipeline from market data to a backtest, every stage behind its own
contract:

- **Data**: Sharadar, Massive, FRED, Tiingo and Alpaca acquisition into raw tiers and
  `xarray` datasets stored as Zarr; point-in-time universes and index membership; causal
  bad-print masking; resampling.
- **Factors**: KunQuant (batch and streaming) and Polars backends; Alpha158, Alpha101, Barra
  USE4 style factors, fundamental, alternative, daily-anomaly and intraday factors;
  neutralization; factor reports. Each factor answers for its own calendar, so a daily
  factor computed from minute bars counts warm-up and lookahead in trading days; compiled
  KunQuant graphs are cached on disk across processes.
- **Labels**: forward returns and their variants.
- **Models**: XGBoost, LightGBM, CatBoost, RealMLP and torch heads (MLP, RNN, TCN,
  Transformer, GATs, MASTER), seed ensembles, walk-forward training.
- **Risk**: a USE4-style factor risk model.
- **Portfolio**: top-N, mean-variance (with turnover, exposure bounds, active risk,
  `max_names`), Black-Litterman.
- **Backtest**: the vectorbt engine with point-in-time execution rules, a fast approximate
  backtest for sweeps, attribution and an HTML report; every run directory rebuilds the run
  that wrote it.
- **Frame API** (`quantlab.api`) for pandas or polars users who want one capability without
  the stores.

### Requirements

- Python 3.13 or newer, on Linux or macOS. Windows is not supported.
- A C++ compiler, since KunQuant compiles factor graphs at run time.
- On macOS, Homebrew's OpenMP runtime (`brew install libomp`) for xgboost.

### Optional extras

`optimize`, `tabkit`, `gbdt`, `wandb`, `mlflow`, `massive` and `mosek`; `all` installs every
one except `mosek`, which needs a commercial licence. Importing a module without its extra
raises an `ImportError` that names the extra.
