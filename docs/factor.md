# Factors

English | [简体中文](zh-CN/factor.md)

A factor turns the `(timestamp, symbol)` panel held by a dataset into a panel of engineered features of the same shape. A label is a factor shifted forward in time, the value a model learns to predict. quantlab has two factor backends: `FactorKunQuant`, which compiles a KunQuant operator graph to native code and runs it in batch or streaming mode, and `FactorPolars`, whose logic is a Polars expression chain and which runs in batch mode only. Both share the base class `Factor`, so the model layer treats them alike.

## Prerequisites

KunQuant factors compile C++ at run time and need a working C++ compiler. Polars factors do not. The examples below run against synthetic data written under the working directory. A crypto spot dataset stores raw Binance column names (`Close`, `Volume`); the dataset guide (`dataset.md`) describes the dataset config fields.

## The basics

### Two backends

| | `FactorKunQuant` | `FactorPolars` |
|---|---|---|
| Logic is written as | a KunQuant graph in `_get_factor_func` | a Polars expression chain in `_get_factor_lazyframe` |
| Modes | batch and streaming | batch only |
| Config class | `FactorConfig` | `PolarsFactorConfig` |
| Input column names | the shared names: the store's own, renamed by the dataset's `COLUMN_MAP` where it has one (a spot store's `Close` becomes `close`) | the store's own names (`Close`); the shared names for a merged input |
| Output dtype | float32 | float64 |
| Cost | compiles the graph on every `compute()` | none |

KunQuant is the primary backend: it has the rolling and cross-sectional operators the shipped alpha libraries are built from, and it is the only one that can run bar by bar on live data. `FactorPolars` is the supplementary path for new factors that are easier to state as DataFrame expressions or that use operations KunQuant lacks. A factor that must also run in streaming mode has to be a KunQuant factor.

### The config

Both backends take a config with the fields below. `FactorConfig` adds `mode` (`"batch"` or `"stream"`), `data_columns` (the dataset variables fed to the graph) and `njobs` (executor threads). `PolarsFactorConfig` adds nothing.

| Field | Meaning |
|---|---|
| `warmup_bars` | bars of history read before the requested start so rolling operators are warm, counted on the dataset's own calendar |
| `dataset` | the dataset the factor reads, or a list of datasets it merges (see below) |
| `file_path` | Zarr store `build` writes and `read` reads |
| `factor_names` | output column names; derived from the factor when left `None` |
| `kwargs` | free-form options a factor class reads |

Configs are frozen. The factor holds a normalised copy with `name` and `factor_names` filled in; the config you pass is never edited, and a saved `config.json` rebuilds into a factor with an equal config.

The config says what is computed, not when. The date range is an argument of `compute(start, end)`, `build(start, end)` and `read(start, end)`, and none of these calls changes the factor's config or its dataset's config.

### Compute a factor

The session first writes a small synthetic spot store and defines a helper that builds a dataset over it. `Momentum` is the reference Polars factor: `Close_t / Close_{t-n} - 1`, with `n` read from `kwargs`. Factor names are known as soon as the object is constructed, before anything is computed.

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.backend import XrBackend
>>> from quantlab.base.config import DatasetConfig, PolarsFactorConfig
>>> from quantlab.dataset.spot import SpotKlineDataset
>>> from quantlab.factor.predefined.momentum import Momentum
>>> rng = np.random.default_rng(0)
>>> symbols = [f"S{i}USDT" for i in range(8)]
>>> close = 100 + np.cumsum(rng.normal(size=(90, 8)), axis=0)
>>> scale = {"Open": 0.99, "High": 1.02, "Low": 0.98, "Close": 1.0, "Volume": 10.0, "Quote asset volume": 1000.0}
>>> raw = xr.Dataset(
...     {name: (["timestamp", "symbol"], close * k) for name, k in scale.items()},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=90), "symbol": symbols},
... )
>>> XrBackend().to_internal(raw).write("data/klines.zarr")
XrBackend()
>>> def make_dataset():
...     return SpotKlineDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/klines.zarr",
...         market="crypto_spot", frequency="1d",
...     ))
...
```

`compute(start, end)` reads the dataset from `warmup_bars` bars before `start`, counted on the dataset's own calendar so days without data are skipped, computes, and returns only `start` to `end`, as an `xarray.Dataset` over `(timestamp, symbol)` with one variable per factor name. The values match a computation over the whole history.

```python
>>> factor = Momentum(PolarsFactorConfig(
...     warmup_bars=20, dataset=make_dataset(), file_path="data/factors/momentum.zarr",
...     kwargs={"n": 5},
... ))
>>> factor.get_factor_names(), factor.warmup_bars
(('momentum_5',), 20)
>>> panel = factor.compute("2024-02-01", "2024-02-29")
>>> dict(panel.sizes), list(panel.data_vars)
({'timestamp': 29, 'symbol': 8}, ['momentum_5'])
>>> panel["momentum_5"].isel(timestamp=0, symbol=slice(0, 3)).values.round(4)
array([ 0.0161,  0.0079, -0.0007])
```

When the dataset holds fewer than `warmup_bars` bars before `start`, a `UserWarning` states the shortfall and the computation starts from the first bar there is:

```text
UserWarning: Momentum.compute(): 20 warm-up bar(s) are needed before '2024-01-05' but SpotKlineDataset holds only 4; the first bars are short by 16 bar(s) of warm-up.
```

The factor holds no panel after `compute`; each call reads the dataset again through `dataset.panel`.

## Common tasks

### Build, read and extend a factor store

`build(start, end)` writes `compute(start, end)` as the store at `store_path`, replacing what was there, and records the range beside it, in `<store>.range.json`. `read(start, end)` returns a range from the store, opened lazily, and refuses one the recorded range does not contain. `extend(end)` computes the bars after the recorded end, warmed from the dataset, appends them and moves the recorded end. It uses `XrBackend.widen_and_append`, so the timestamp, symbol and variable axes widen to fit and the append checks described in the backend guide apply.

```python
>>> factor.build("2024-01-21", "2024-02-29").store_range()
('2024-01-21', '2024-02-29')
>>> dict(factor.read("2024-02-10", "2024-02-15").sizes)
{'timestamp': 6, 'symbol': 8}
>>> factor.read("2024-02-10", "2024-03-10")
Traceback (most recent call last):
ValueError: Momentum.read(): the store at data/factors/momentum.zarr covers 2024-01-21 to 2024-02-29, which does not contain 2024-02-10 to 2024-03-10. Extend it with extend(end) or rebuild it with build(start, end).
>>> factor.extend("2024-03-20").store_range()
('2024-01-21', '2024-03-20')
>>> factor.read("2024-01-21", "2024-03-20").sizes["timestamp"]
60
>>> stored = xr.open_zarr("data/factors/momentum.zarr")
>>> stored.sizes["timestamp"], str(stored["timestamp"].values[-1])[:10]
(60, '2024-03-20')
```

`store_range()` is `None` for a store that `build` did not write, and `read` and `extend` refuse such a store. A KunQuant factor answers these calls in batch mode only.

### Rebuild a factor from its config

`get_config()` returns a dict describing the factor and its dataset, and `load_factor_from_config` rebuilds the factor from it.

```python
>>> cfg = factor.get_config()
>>> cfg["name"], cfg["kwargs"], cfg["dataset"]["market"]
('quantlab.factor.predefined.momentum.Momentum', {'n': 5}, 'crypto_spot')
>>> from quantlab.utils.module import load_factor_from_config
>>> rebuilt = load_factor_from_config(cfg)
>>> type(rebuilt).__name__, rebuilt.get_factor_names()
('Momentum', ('momentum_5',))
```

### Merge several datasets into one input

`dataset` also takes a list of datasets. The factor merges them into one `MergedDataset`: each input is renamed to the shared variable names with its own `COLUMN_MAP` (a spot store's `Close` becomes `close`; a stock store keeps its names), then the inputs are outer-joined on timestamp and symbol, NaN where an input has no value. This covers an index store plus an ETF store (same variables, different symbols) and prices plus quotes (same symbols, different variables), across dataset classes. Warm-up is counted on the union of the inputs' calendars. The example continues the session above: the spot store is split into four spot symbols and four symbols stored under the shared names, and a Polars factor reads the shared names from both.

```python
>>> import polars as pl
>>> from quantlab.factor.polars import FactorPolars
>>> from quantlab.dataset.merged import MergedDataset
>>> from quantlab.dataset.stock import StockDataset
>>> XrBackend().to_internal(raw.sel(symbol=symbols[:4])).write("data/spot_half.zarr")
XrBackend()
>>> XrBackend().to_internal(
...     raw.sel(symbol=symbols[4:]).rename(SpotKlineDataset.COLUMN_MAP)
... ).write("data/stock_half.zarr")
XrBackend()
>>> spot_half = SpotKlineDataset(DatasetConfig(
...     raw_data_dir_path="data/raw", zarr_file_path="data/spot_half.zarr",
...     market="crypto_spot", frequency="1d",
... ))
>>> stock_half = StockDataset(DatasetConfig(
...     raw_data_dir_path="data/raw", zarr_file_path="data/stock_half.zarr",
...     market="us_equity", frequency="1d",
... ))
>>> panel = MergedDataset([spot_half, stock_half]).panel("2024-02-01", "2024-02-29")
>>> dict(panel.sizes), sorted(panel.data_vars)
({'timestamp': 29, 'symbol': 8}, ['amount', 'close', 'high', 'low', 'open', 'volume'])
>>> class Range(FactorPolars):
...     def _get_factor_lazyframe(self, lf):
...         return lf.with_columns(
...             ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range")
...         ).select(["timestamp", "symbol", "range"])
...
>>> factor_range = Range(PolarsFactorConfig(
...     warmup_bars=0, dataset=[spot_half, stock_half], file_path="data/factors/range.zarr",
... ))
>>> type(factor_range.config.dataset).__name__
'MergedDataset'
>>> dict(factor_range.compute("2024-02-01", "2024-02-29").sizes)
{'timestamp': 29, 'symbol': 8}
>>> cfg = factor_range.get_config()
>>> cfg["dataset"]["name"], [d["zarr_file_path"] for d in cfg["dataset"]["datasets"]]
('quantlab.dataset.merged.MergedDataset', ['data/spot_half.zarr', 'data/stock_half.zarr'])
>>> load_factor_from_config(cfg) == factor_range
True
```

A merge never picks a value by input order. A cell holding a value in two inputs raises `ValueError: MergedDataset: variable 'close' holds a value in both SpotKlineDataset(data/spot_half.zarr) and SpotKlineDataset(data/overlap.zarr), for example at symbol 'S3USDT' on 2024-02-01 00:00:00. ...`, and inputs on different bars raise `ValueError: MergedDataset: the inputs have different bar spacing (SpotKlineDataset(data/spot_half.zarr): 1 days 00:00:00, SpotKlineDataset(data/hourly.zarr): 0 days 01:00:00). ...`. A KunQuant factor over a merge lists the shared names in `data_columns`, as a Polars factor spells them in its expressions (`close`, not `Close`). Stream mode refuses a merge when the factor is constructed: `ValueError: MaDeviation: stream mode takes one dataset, got a merge of 2. ...`. `MergedDataset` is itself a dataset, with `panel` and `bar_before`; it holds no store, so `store_path`, `save`, `resample` and the build path refuse. Resample the inputs before merging them; a factor over a merge can itself be resampled when every input cuts bars the same way. A merged dataset cannot yet be a backtest's `price_dataset` or `benchmark_dataset`, which need a store path.

### Broadcast index or ETF features to every symbol

`MarketFeatures` (`quantlab.factor.predefined.market`) computes market-wide features from one or more index or ETF series and gives every symbol of a target panel the same values. These are the market inputs of MASTER (`research/qlib-gats-master.md`, section 2.1). Its config is `MarketFeatureConfig`. `dataset` is the target: its symbols receive the features, and `warmup_bars` is counted on its calendar. `series` maps a name to a single-symbol dataset. For each series the factor computes 21 features on the series' own bars: `<name>_ret`, the bar return `close / close[t-1] - 1`, and for d in 5, 10, 20, 30 and 60 bars `<name>_ret_mean_<d>` and `<name>_ret_std_<d>` (the mean and standard deviation of the return over d bars) and `<name>_amount_mean_<d>` and `<name>_amount_std_<d>` (the same for the traded amount, divided by the bar's own amount). The amount is volume times close unless `kwargs["amount_column"]` names a column that holds it. `warmup_bars` defaults to 60, the longest window. The session below writes three small stores, six stocks (`FFF` lists on 1 April) and two ETFs, and computes the features over March and April.

```python
>>> import json
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.base.config import DatasetConfig, MarketFeatureConfig
>>> from quantlab.dataset.stock import StockDataset
>>> from quantlab.factor.predefined.market import MarketFeatures
>>> from quantlab.utils.module import load_factor_from_config
>>> market_days = pd.bdate_range("2024-01-01", periods=120)
>>> market_rng = np.random.default_rng(1)
>>> def market_write_store(path, symbols):
...     close = 100 * np.exp(np.cumsum(market_rng.normal(0, 0.01, (120, len(symbols))), axis=0))
...     volume = market_rng.uniform(1e6, 5e6, (120, len(symbols)))
...     if "FFF" in symbols:  # FFF lists on 1 April
...         close[market_days < "2024-04-01", symbols.index("FFF")] = np.nan
...     _ = xr.Dataset(
...         {"close": (["timestamp", "symbol"], close),
...          "adjClose": (["timestamp", "symbol"], close),
...          "adjVolume": (["timestamp", "symbol"], volume)},
...         coords={"timestamp": market_days, "symbol": symbols},
...     ).to_zarr(path, mode="w")
...     return StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path=path,
...         market="us_equity", frequency="1d",
...     ))
...
>>> stocks_ds = market_write_store("data/stocks.zarr", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
>>> spy_ds = market_write_store("data/spy.zarr", ["84398"])
>>> qqq_ds = market_write_store("data/qqq.zarr", ["86755"])
>>> market_factor = MarketFeatures(MarketFeatureConfig(
...     dataset=stocks_ds, series={"spy": spy_ds, "qqq": qqq_ds},
...     file_path="data/factors/market.zarr",
... ))
>>> market_factor.warmup_bars, market_factor.num_factors
(60, 42)
>>> market_factor.get_factor_names()[:5]
('spy_ret', 'spy_ret_mean_5', 'spy_ret_std_5', 'spy_amount_mean_5', 'spy_amount_std_5')
>>> market_panel = market_factor.compute("2024-03-01", "2024-04-30")
>>> dict(market_panel.sizes)
{'timestamp': 43, 'symbol': 6}
>>> market_panel["spy_ret_mean_20"].sel(timestamp="2024-03-01").values.round(5)
array([0.00035, 0.00035, 0.00035, 0.00035, 0.00035,     nan],
      dtype=float32)
>>> market_panel["spy_ret_mean_20"].sel(timestamp="2024-04-01").values.round(5)
array([-0.00302, -0.00302, -0.00302, -0.00302, -0.00302, -0.00302],
      dtype=float32)
>>> market_factor.options
{'close_column': 'adjClose', 'volume_column': 'adjVolume', 'amount_column': None, 'presence_column': 'close'}
>>> market_cfg = json.loads(json.dumps(market_factor.get_config()))
>>> list(market_cfg["series"]), market_cfg["series"]["spy"]["zarr_file_path"]
(['spy', 'qqq'], 'data/spy.zarr')
>>> load_factor_from_config(market_cfg) == market_factor
True
>>> market_factor.build("2024-03-01", "2024-04-30").store_range()
('2024-03-01', '2024-04-30')
```

On each bar the values go to every target symbol that has a bar there, that is, whose `kwargs["presence_column"]` (default `close`) is not missing. `FFF` is therefore NaN before it lists, and a model does not see market features on a bar where a symbol had no data. The rolling windows run over each series' own bars, and a target bar that a series lacks is NaN. A window is defined only when all of its bars are. The standard deviations use `ddof=1`, as pandas and Qlib do, an amount of 0 gives NaN rather than an infinite ratio, and the panel is float32. The series columns are `kwargs["close_column"]` (default `adjClose`) and `kwargs["volume_column"]` (default `adjVolume`), looked up after the dataset's `COLUMN_MAP` renaming, so a crypto spot series is read as `close`, `volume` and `amount`. `get_config()` nests each series dataset's config under `series`, and `load_factor_from_config` rebuilds them. If the warm-up is short, the long windows stay NaN on the first bars and `compute` warns: `UserWarning: MarketFeatures.compute(): 60 warm-up bar(s) are needed before '2024-01-10' but StockDataset holds only 7; the first bars are short by 53 bar(s) of warm-up.` A series store holding more than one symbol is refused when a panel is computed: `ValueError: MarketFeatures: series 'stocks' must hold one symbol, its StockDataset holds 6; give each series its own single-symbol dataset.`

For US equities from WRDS, give each ETF its own CRSP store, as for a backtest benchmark. `scripts/wrds/etf.py --etf spy,qqq,iwm` downloads SPY, QQQ and IWM (the S&P 500, the Nasdaq-100 and the Russell 2000) into one store each, and `CrspDatasetConfig.etf_benchmark` keeps the ETF, which the default security filter drops as a fund:

```python
from quantlab.base.config import (
    IWM_PERMNO, QQQ_PERMNO, SPY_PERMNO, CrspDatasetConfig, MarketFeatureConfig,
)
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.factor.predefined.market import MarketFeatures

def etf(permno, path):
    return CrspStockDataset(CrspDatasetConfig.etf_benchmark(
        permno=permno, zarr_file_path=path,
        raw_data_dir_path="/data/downloads/us_equity/1d/crsp/wrds",
        reference_dir="/data/reference/crsp",
    ))

market = MarketFeatures(MarketFeatureConfig(
    dataset=stocks,  # the CRSP panel the model trains on
    series={"spy": etf(SPY_PERMNO, "/data/zarrs/wrds_crsp_spy_1d.zarr"),
            "qqq": etf(QQQ_PERMNO, "/data/zarrs/wrds_crsp_qqq_1d.zarr"),
            "iwm": etf(IWM_PERMNO, "/data/zarrs/wrds_crsp_iwm_1d.zarr")},
    file_path="/data/factors/market.zarr",
))
```

The factor goes into a model like any other. `MASTERRegressor` takes its variable names, `list(market.get_factor_names())`, as the `gate_features` hyperparameter (see Train MASTER with market features in the model guide).

### Resample a factor onto coarser bars

`resample(freq, how)` returns a copy of the factor whose `compute`, `read` and `build` answer on coarser bars. The factor is still computed on its dataset's own bars; only the output is aggregated, so a minute-bar momentum becomes a daily series of the last minute's value without changing what it measures. `freq` and `how` take the same values as `BaseDataset.resample` (see the dataset guide), and `how` may be one method as a string for every factor variable. Bars are cut the way the factor's dataset cuts them.

The session below writes a two-day minute store, the one of the dataset guide's resample section, and runs `Momentum` over it. The store starts on the first requested bar, so `compute` also warns that the one warm-up bar is missing.

```python
>>> from quantlab.dataset.spot import SpotKlineDataset
>>> minutes = pd.DatetimeIndex(np.concatenate([
...     pd.date_range(f"2024-01-0{d} 00:00", periods=4, freq="min").values for d in (2, 3)
... ]))
>>> minute_close = np.arange(1.0, 9.0)[:, None] * np.array([[1.0, 10.0]])
>>> _ = xr.Dataset(
...     {"Open": (["timestamp", "symbol"], minute_close - 0.5),
...      "Close": (["timestamp", "symbol"], minute_close),
...      "Volume": (["timestamp", "symbol"], np.ones((8, 2)))},
...     coords={"timestamp": minutes, "symbol": ["AAAUSDT", "BBBUSDT"]},
... ).to_zarr("data/minute_klines.zarr", mode="w")
>>> minute_config = DatasetConfig(raw_data_dir_path="downloads/spot",
...                               zarr_file_path="data/minute_klines.zarr",
...                               market="crypto_spot", frequency="1m")
>>> minute = Momentum(PolarsFactorConfig(
...     warmup_bars=1, dataset=SpotKlineDataset(minute_config),
...     file_path="data/factors/minute_momentum.zarr", kwargs={"n": 1},
... ))
>>> minute.compute("2024-01-02", "2024-01-03")["momentum_1"].to_pandas().round(3)
symbol               AAAUSDT  BBBUSDT
timestamp                            
2024-01-02 00:00:00      NaN      NaN
2024-01-02 00:01:00    1.000    1.000
2024-01-02 00:02:00    0.500    0.500
2024-01-02 00:03:00    0.333    0.333
2024-01-03 00:00:00    0.250    0.250
2024-01-03 00:01:00    0.200    0.200
2024-01-03 00:02:00    0.167    0.167
2024-01-03 00:03:00    0.143    0.143
>>> daily = minute.resample("1d", "last")
>>> daily.compute("2024-01-02", "2024-01-03")["momentum_1"].to_pandas().round(3)
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02    0.333    0.333
2024-01-03    0.143    0.143
>>> daily.config.dataset.config.resample_freq, daily.config.resample_freq
(None, '1d')
```

The copy has its own dataset object and an empty compiled state. `build()` writes to `store_path`, beside the source store, and `read()` on a copy opens that store when it exists and otherwise resamples the source factor's store. The request round-trips through `get_config()` and `load_factor_from_config`.

```python
>>> daily.store_path
'data/factors/minute_momentum_resample_1d.zarr'
>>> daily.build("2024-01-02", "2024-01-03").store_range()
('2024-01-02', '2024-01-03')
>>> daily.read("2024-01-02", "2024-01-03").sizes
Frozen({'timestamp': 2, 'symbol': 2})
>>> cfg = daily.get_config()
>>> cfg["resample_freq"], cfg["resample_how"], cfg["dataset"]["resample_freq"]
('1d', 'last', None)
>>> load_factor_from_config(cfg).compute("2024-01-02", "2024-01-03").sizes
Frozen({'symbol': 2, 'timestamp': 2})
```

To compute a factor on already-resampled bars instead, resample the dataset and give the factor the resampled dataset.

### Compute a label

A label is a factor shifted forward in time. `Forward(ForwardConfig(factor, span, delay=1))` in `quantlab.label.forward` places the wrapped factor's value at bar t + delay + span at bar t. `span` is the number of bars the label accumulates over, such as the n bars of an n-bar return. `delay` is the number of bars between the bar a signal forms on and the first bar the label counts: 1 by default, because a signal at bar t fills at bar t+1's open. `lookahead_bars()` returns `delay + span`, the bars past t the label at t reads, and `span_bars()` returns `span`. The wrapped factor must use only bars up to t for its value at t; `Forward` relies on this and cannot check it.

Any factor becomes a label this way. Wrapping the 5-bar `Momentum` `factor` of the first section with `span=5` gives the close-to-close return from bar t+1 to bar t+6. The label keeps the factor's variable names and rebuilds from its config like a factor.

```python
>>> from quantlab.base.config import ForwardConfig
>>> from quantlab.label.forward import Forward
>>> label = Forward(ForwardConfig(factor=factor, span=5))
>>> label.lookahead_bars(), label.span_bars(), label.get_factor_names()
(6, 5, ('momentum_5',))
>>> fwd_mom = label.compute("2024-02-01", "2024-02-29")["momentum_5"]
>>> round(float(fwd_mom.isel(timestamp=0, symbol=0)), 6)
-0.012116
>>> round(float(close[37, 0] / close[32, 0] - 1), 6)
-0.012116
>>> load_factor_from_config(label.get_config()) == label
True
```

`compute(start, end)` and `read(start, end)` request the wrapped factor up to `lookahead_bars()` bars past `end`, counted on the dataset's calendar and clipped at its last bar, then shift and trim back to the request. A label is therefore NaN at the end of a range only where the future bars do not exist yet. The spot store ends on 2024-03-30, so February is filled to its last day and only the last 6 bars of March have no label:

```python
>>> int(fwd_mom.isnull().sum())
0
>>> tail = label.compute("2024-03-20", "2024-03-30")["momentum_5"]
>>> int(tail.isnull().any("symbol").sum())
6
```

A `Forward` owns no store: `read` reads the wrapped factor's store, which must reach `lookahead_bars()` bars past `end` wherever the dataset has those bars, and `build(start, end)` and `extend(end)` build or extend the factor's store that far. One factor store thus serves as a feature and, wrapped, as a label. The momentum store built above ends on 2024-03-20:

```python
>>> dict(label.read("2024-02-01", "2024-02-29").sizes)
{'timestamp': 29, 'symbol': 8}
>>> label.read("2024-02-01", "2024-03-20")
Traceback (most recent call last):
ValueError: Momentum.read(): the store at data/factors/momentum.zarr covers 2024-01-21 to 2024-03-20, which does not contain 2024-02-01 to 2024-03-26 00:00:00. Extend it with extend(end) or rebuild it with build(start, end).
```

`Return` and `BinaryReturn` in `quantlab.label.predefined.fret` are `Forward` labels over a private trailing-return KunQuant factor, and are labels only. `Return` at bar t is `adjOpen[t + n + 1] / adjOpen[t + 1] - 1`, the return of a position entered at the next bar's adjusted open and held for n bars, with n read from `kwargs["n_forward_periods"]`; `BinaryReturn` is 1.0 where that return is positive and 0.0 elsewhere. Both have `span = n` and `delay = 1`, so their lookahead is n + 1. They read `adjOpen`, so the dataset must carry adjusted prices; a US equity dataset does, the crypto spot dataset does not. The session below starts at the store's first bar, so `compute` warns that the 5 warm-up bars are missing, and the store ends on 2024-01-30, so the last 3 bars have no label.

```python
>>> from quantlab.base.config import FactorConfig
>>> from quantlab.dataset.stock import StockDataset
>>> from quantlab.label.predefined.fret import Return
>>> px = 50 + np.cumsum(rng.normal(size=(30, 8)), axis=0)
>>> stock = xr.Dataset(
...     {"adjOpen": (["timestamp", "symbol"], px)},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=30), "symbol": [f"T{i}" for i in range(8)]},
... )
>>> XrBackend().to_internal(stock).write("data/stock.zarr")
XrBackend()
>>> ret_label = Return(FactorConfig(
...     warmup_bars=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 2},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/stock.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     file_path="data/labels/ret.zarr", njobs=2,
... ))
>>> ret_label.get_factor_names(), ret_label.lookahead_bars(), ret_label.span_bars()
(('ret_2',), 3, 2)
>>> ret = ret_label.compute("2024-01-01", "2024-01-30")["ret_2"]
>>> dict(ret.sizes)
{'timestamp': 30, 'symbol': 8}
>>> ret.isel(symbol=0).values[:2].round(5)
array([-0.03945,  0.01532], dtype=float32)
>>> round(float(px[3, 0] / px[1, 0] - 1), 5)
-0.03945
>>> ret.isel(symbol=0).values[-4:].round(5)
array([0.00357,     nan,     nan,     nan], dtype=float32)
```

A model takes labels, objects with `lookahead_bars()` such as `Forward`, only in its `labels` and refuses them among its `factors`. At every split boundary it drops the last L bars of the earlier segment, L being the largest lookahead among its labels (see `model.md`). A backtest refuses a label whose `delay` differs from its engine's fill delay (see `backtest.md`).

### Analyze a factor

`analyze(start, end, ...)` reports how well a factor orders symbols by their forward return from `start` to `end`, in the manner of the alphalens library. Pass one or more forward-return labels (`Forward` objects such as `Return`) as `frets`; every factor variable (all of `get_factor_names()`, or the ones named in `factor_names`) is paired with every label variable. The factor and every label are asked for their panels over `[start, end]`: with `compute(start, end)` under `data_strategy="cal"` (the default), or with `read(start, end)` under `data_strategy="read"`, which needs every store built over that range. Any other `data_strategy` raises `ValueError`. The two panels must have the same most common bar spacing, the rule `BaseDataset.time_interval` uses, or `analyze()` raises `ValueError` naming both spacings; they are then joined on their common timestamps and symbols.

Each pair gets:

| Group | Metrics |
|---|---|
| Information | per-period IC (Spearman rank correlation across symbols), IC mean, std, IR (mean / std), t-statistic, p-value, skew, excess kurtosis, share of positive periods, monthly mean IC |
| Returns | mean forward return per factor quantile (bucket 1 holds the lowest values), top-minus-bottom spread per period, cumulative return per quantile and long-short |
| Turnover | share of each quantile's symbols that were not in it the period before, lag-1 factor rank autocorrelation |

`quantiles` (default 5) sets the number of equal-count buckets. When a label spans `n` bars (its `span_bars()`), cumulative returns compound the per-bar rate `(1 + r) ** (1 / n) - 1`. The example below builds a one-bar `Return` label over the same eight symbols as `factor` from the first section, then analyzes `momentum_5` against it. On this random walk the IC is near zero, as it should be.

```python
>>> import os
>>> from quantlab.base.config import FactorConfig
>>> opens = xr.Dataset(
...     {"adjOpen": (["timestamp", "symbol"], close * 0.99)},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=90), "symbol": symbols},
... )
>>> XrBackend().to_internal(opens).write("data/spot_open.zarr")
XrBackend()
>>> fwd = Return(FactorConfig(
...     warmup_bars=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 1},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/spot_open.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     file_path="data/labels/fwd_ret.zarr", njobs=2,
... ))
>>> result = factor.analyze(
...     "2024-02-01", "2024-02-29", frets=[fwd], quantiles=4,
...     output_dir="data/analysis/momentum",
... )
>>> list(result.pairs)
['momentum_5__ret_1']
>>> pair = result.pairs["momentum_5__ret_1"]
>>> round(pair.summary["ic_mean"], 4), round(pair.summary["ir"], 4), pair.summary["n_periods"]
(-0.0279, -0.0799, 29)
>>> pair.mean_quantile_returns.round(4).tolist()
[0.0008, -0.0027, 0.0003, -0.0006]
>>> sorted(os.listdir("data/analysis/momentum"))
['config.json', 'ic.csv', 'momentum_5__ret_1.png', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv']
>>> import json
>>> from quantlab.utils.module import load_factor_from_config
>>> cfg = json.load(open("data/analysis/momentum/config.json"))
>>> list(cfg), type(load_factor_from_config(cfg["frets"][0])).__name__
(['factor', 'frets'], 'Return')
```

The result carries `pairs` (a `PairAnalysis` per `"<factor>__<fret>"`, with the IC series, its running sum `cumulative_ic`, quantile returns, turnover and a `summary` dict), `figures` (one matplotlib figure per pair, held only when no `output_dir` is given) and tidy tables from `summary_table()`, `ic_table()`, `monthly_ic_table()`, `quantile_returns_table()` and `turnover_table()`. With `output_dir`, those tables are written as CSV, the scalar metrics as `summary.json`, each figure as `<factor>__<fret>.png`, and `config.json` holds the factor's and the labels' configs, each rebuildable with `load_factor_from_config`; the figures are then drawn on every CPU in parallel straight to the PNG files and not kept, since drawing dominates the cost of a report over a whole alpha library. Without `output_dir` nothing is written. The figures are built without `pyplot`, so they are never shown and need no closing; `fig.savefig(path)` writes one. The IC panel draws the cumulative IC on its right axis; the turnover and rank-autocorrelation panels draw each series as a translucent rolling range (minimum to maximum over `rolling_window` periods, 22 by default) with its rolling mean, not the raw per-period values. The metrics are computed with polars: every factor variable of a chunk (`chunk_size`, default 32) is one lazy plan over the long `(timestamp, symbol)` frame, collected once. The machinery lives in `quantlab.analysis.factor_report`, where `FactorAnalyzer.run(factor, frets, features=..., labels=[...], factor_names=None, output_dir=None)` takes the feature and label panels explicitly.

Every pair reports the Pearson IC beside the rank IC (`pearson_ic`, and `pearson_ic_mean`, `pearson_ic_std`, `pearson_ir`, `pearson_ic_t_stat` in the summary); a Pearson IC far from the rank IC means a few extreme values drive the linear relation. The mean IC has a Newey-West t-statistic, `ic_nw_t_stat` and `ic_nw_p_value`, with Bartlett weights over `ic_nw_lags` lags: the larger of `horizon - 1`, the autocorrelation overlapping multi-bar forward returns induce, and the rule of thumb `floor(4 * (n / 100) ** (2 / 9))`. For a label spanning several bars the plain `ic_t_stat` overstates significance; read `ic_nw_t_stat` instead. The rank autocorrelation is measured at every lag of `FactorAnalyzer(autocorrelation_lags=(1, 5, 10, 20))`: `rank_autocorrelations` holds one column per lag, the summary `rank_autocorrelation_lag<k>`, `turnover_table()` one column per lag, and the figure draws the rolling range and rolling mean of every lag in one panel, darker for shorter lags. The long-short portfolio is summarized by `long_short_annual_return`, `long_short_annual_volatility`, `long_short_sharpe` and `long_short_max_drawdown`, annualized with `periods_per_year`, which is measured from the data (about 252 for daily stock bars, 365 for daily crypto bars); a value that falls to zero stays there. With two or more frets, `ic_decay_table()` lists every pair's mean IC by horizon with its 95% Newey-West interval, `ic_decay.csv` holds it, and every pair figure includes a panel of the factor's mean IC against the horizon with the pair's own fret ringed. Read beside the autocorrelation panel, it shows how fast the signal fades and how slowly the factor changes, which together suggest a holding period. Pass frets of several horizons, for example `Return` labels with `n_forward_periods` of 1, 5 and 20, to get it.

When two or more factor variables are analyzed, the result carries `correlation`, a `FactorCorrelation` (`quantlab.analysis.factor_correlation`) of the analyzed variables with each other; with a single variable it is `None`. The correlation of two variables is their Spearman rank correlation across the symbols of each timestamp, averaged over timestamps, with its standard deviation and the number of timestamps used. Each variable is ranked once among its own finite symbols, so when two variables cover different symbols the value is a close approximation of Spearman's rather than the exact one; this keeps several hundred variables to a few matrix products per timestamp. The variables are clustered by `1 - |correlation|` (average linkage), so a factor and its negation fall in one cluster, and the matrices are ordered by that clustering; `FactorAnalyzer(correlation_threshold=0.7)` sets where the clusters are cut. `pairs_table()` lists every pair strongest first, `cluster_summary()` every cluster of two or more variables with its size, mean inner `|correlation|` and members, and `cluster_table()` every variable's cluster and position. With `output_dir` these are written as `factor_correlation.csv` (the ordered matrix), `factor_correlation_pairs.csv`, `factor_clusters.csv` and `factor_correlation.png`, and `summary.json` holds a `correlation` entry; without it the figure is held in `correlation_figure`. The figure shows the ordered matrix on a fixed -1 to 1 diverging scale (red negative, gray none, blue positive) with the always-1 diagonal left blank and every cluster of two or more outlined, the strongest pairs as `|correlation|` bars colored by sign, the largest clusters with their size, inner `|correlation|` and first members, and the distribution of all pairs on a log count scale so the few strong pairs stay visible. Up to `FactorCorrelationFigure(label_limit=800)` variables it names every row and column: the heatmap grows so each row is at least 6.5 points high and the PNG is written at 150 dpi, so past about a hundred variables the names are read zoomed in (300 variables give a 5,450 by 4,445 pixel image). Beyond the limit it names the clusters instead, and `factor_clusters.csv` maps each variable to its position. On a 2,500-day, 500-symbol panel, 300 variables take about 25 seconds.

### Normalize over time or across symbols

`quantlab.my_ops.preprocess` has four KunQuant operators. `WindowedZScore` standardizes each symbol against its own trailing window, a time-series normalization. `CrossSectionalZScore` standardizes each timestamp across all symbols. Which one is right depends on the strategy consuming the factor. `Alpha101SpotKline` and `Alpha158SpotKline` apply `WindowedZScore` to every output over `kwargs["zscore_window"]` bars (default 20), independent of `warmup_bars`; for the first requested bar to be fully normalized, `warmup_bars` must cover the alpha's own lookback plus `zscore_window - 1` bars (up to 60 + 19 for the full Alpha158 set). A `zscore_window` that is not a positive integer is refused when the factor is constructed; `Alpha101Stock` and `Alpha158Stock` apply `CrossSectionalZScore` to every output. A symbol with no bar that day (not yet listed, delisted, or an all-NaN column) is NaN in every output of these two classes, so it enters neither their ranks nor the z-score; on a bar with data their values are KunQuant's, including the 0 its formulas give a value undefined on real data, such as a correlation over a window of constant values. The KunQuant factor under Extending applies both operators.

The module also has two cross-sectional outlier operators. `CrossSectionalWinsorize(v, lower=0.01, upper=0.99)` (winsorizing) clips each timestamp's values to that bar's `lower` and `upper` quantiles across symbols, and `CrossSectionalTrim(v, lower=0.01, upper=0.99)` (trimming) sets values strictly outside those quantiles to NaN. Quantiles ignore NaN and interpolate linearly, like `np.nanquantile`. A common chain is `CrossSectionalZScore(CrossSectionalWinsorize(v))`, so a few extreme symbols do not dominate the mean and standard deviation. KunQuant 0.1.11 has no built-in operator for either: its `Clip` bounds by a fixed constant and `WindowedQuantile` works along time.

### Shipped factors

| Class | Backend | Notes |
|---|---|---|
| `Momentum` | Polars | reference Polars factor, reads `Close` |
| `Alpha101SpotKline`, `Alpha101Stock` | KunQuant | KunQuant's Alpha101 library; the `Stock` class builds it from a copy in `quantlab.factor.predefined._support.kunquant_alpha101` that is NaN on a bar with no data |
| `Alpha158SpotKline`, `Alpha158Stock` | KunQuant | Alpha158 features, the `Stock` class from the copy in `quantlab.factor.predefined._support.kunquant_alpha158`; pin `factor_names` while experimenting |
| `ResidualMomentumFF3` | KunQuant | Fama-French three-factor residual momentum; the factor series come from a Fama-French CSV or from the panel |
| `LiteratureAlpha` | KunQuant | Eight raw/ranked equity characteristics spanning price, risk, liquidity, fundamentals and earnings events |
| `MarketFeatures` | xarray | 21 return and amount features per index or ETF series, the same for every symbol with a bar; config class `MarketFeatureConfig` |
| `Forward` | any | shifts a factor forward into a label |
| `Return`, `BinaryReturn` | KunQuant | forward-return labels, `Forward` subclasses |

Each class docstring shows its config.

### Literature-backed equity alpha bundle

Author: [Jerry](https://github.com/j38903016-lgtm)

`LiteratureAlpha` is one universe-agnostic KunQuant factor class. It computes
eight characteristics and emits a raw value plus a cross-sectional rank for
each one. Apply a point-in-time universe to the market panel separately; a
symbol masked to NaN is ignored by `Rank` automatically.

| Output stem | Definition and direction | Reference |
|---|---|---|
| `high_52week_proximity` | `split_adjusted_close / rolling_max(split_adjusted_close, 252)`; high is positive | [George and Hwang (2004)](https://doi.org/10.1111/j.1540-6261.2004.00695.x) |
| `short_reversal` | negative compounded return over 21 bars; high means a worse prior month | [Jegadeesh (1990)](https://doi.org/10.1111/j.1540-6261.1990.tb05110.x) |
| `low_max` | negative maximum daily return over 21 bars; high avoids lottery-like stocks | [Bali, Cakici and Whitelaw (2011)](https://www.nber.org/papers/w14804) |
| `low_idiosyncratic_volatility` | negative standard deviation of residuals from a 21-bar FF3 regression | [Ang, Hodrick, Xing and Zhang (2006)](https://doi.org/10.1111/j.1540-6261.2006.00836.x) |
| `amihud_illiquidity` | log average of `abs(return) / (raw_close * volume)`; larger means less liquid | [Amihud (2002)](https://doi.org/10.1016/S1386-4181(01)00024-6) |
| `gross_profitability` | latest public gross profit divided by total assets; high is positive | [Novy-Marx (2013)](https://www.nber.org/papers/w15940) |
| `conservative_asset_growth` | negative annual total-asset growth; high means more conservative investment | [Cooper, Gulen and Schill (2008)](https://doi.org/10.1111/j.1540-6261.2008.01370.x) |
| `standardized_unexpected_earnings` | `(actual EPS - pre-announcement consensus EPS) / scale price`; high is positive | [Livnat and Mendenhall (2006)](https://doi.org/10.1111/j.1475-679X.2006.00196.x) |

Every stem has `<stem>_raw` and `<stem>_rank`. `factor_names` may select any
subset; the class then prunes unrelated formulas and requires only the panel
columns reachable from that subset. Column names and lookbacks can be changed
in `kwargs`. By default the complete graph reads `ret`, `adjClose`, `close`,
`volume`, the four FF3 inputs, three accounting fields and three earnings-event
fields. The FF3 series can instead come from the same CSV accepted by
`ResidualMomentumFF3` through `kwargs={"fama_french_csv": "..."}`.

The data layer owns point-in-time correctness. `gross_profit`, `total_assets`
and `prior_year_total_assets` must become visible only after their filing is
public. The three earnings columns must freeze actual EPS, the consensus that
existed before the announcement and the scale price at that event; do not join
a revised current consensus to a historical actual. The 52-week price should
be split-adjusted, while Amihud dollar volume needs an as-traded close and raw
share volume.

```python
from quantlab.base.config import FactorConfig
from quantlab.factor.predefined.literature_alpha import LiteratureAlpha

factor = LiteratureAlpha(FactorConfig(
    warmup_bars=400,
    dataset=dataset,
    mode="batch",
    data_columns=(
        "adjClose", "ret", "risk_free", "mkt_rf", "smb", "hml",
        "close", "volume", "gross_profit", "total_assets",
        "prior_year_total_assets", "eps_actual_event",
        "eps_consensus_event", "eps_scale_price_event",
    ),
    factor_names=None,  # all 16 raw/rank outputs
    file_path="data/factors/literature_alpha.zarr",
))
features = factor.compute("2020-01-01", "2024-12-31")
```

## Extending

### A Polars factor

Subclass `FactorPolars` and implement `_get_factor_lazyframe`. It receives the dataset as a `polars.LazyFrame` and returns a lazy frame with only `timestamp`, `symbol` and the factor columns. Factor names are read from the returned schema.

```python
>>> import polars as pl
>>> from quantlab.factor.polars import FactorPolars
>>> class RelativeVolume(FactorPolars):
...     def _get_factor_lazyframe(self, lf):
...         volume = pl.col("Volume")
...         return (
...             lf.sort(["symbol", "timestamp"])
...             .with_columns(
...                 (volume / volume.rolling_mean(5).over("symbol") - 1.0).alias("rel_volume_5")
...             )
...             .select(["timestamp", "symbol", "rel_volume_5"])
...         )
...
>>> rv = RelativeVolume(PolarsFactorConfig(
...     warmup_bars=10, dataset=make_dataset(), file_path="data/factors/rel_volume.zarr",
... ))
>>> rv.get_factor_names()
('rel_volume_5',)
>>> out = rv.compute("2024-02-01", "2024-02-29")
>>> dict(out.sizes), str(out["rel_volume_5"].dtype)
({'timestamp': 29, 'symbol': 8}, 'float64')
>>> int(out["rel_volume_5"].isnull().sum())
0
```

### A KunQuant factor

Subclass `FactorKunQuant` and implement `_get_factor_names` and `_get_factor_func` (the KunQuant graph, with one `Input` per entry of `data_columns` and one `Output` per factor name). The graph below outputs a moving-average deviation raw, z-scored over time and z-scored across symbols. KunQuant compiles on every `compute()` (about a second here). A factor whose fields constrain each other (its `data_columns` against its parameters, say) overrides `_validate_config`, reads `self.config` and raises `ValueError`; it runs on every config assignment, including those made by `copy()` and `resample()`, and a refused config leaves the factor's previous one in place. `LiteratureAlpha` and `ResidualMomentumFF3` check their `data_columns` this way.

```python
>>> import KunQuant.ops as op
>>> from KunQuant.Op import Builder, Input, Output
>>> from KunQuant.Stage import Function
>>> from quantlab.base.config import FactorConfig
>>> from quantlab.factor.kunquant import FactorKunQuant
>>> from quantlab.my_ops.preprocess import CrossSectionalZScore, WindowedZScore
>>> class MaDeviation(FactorKunQuant):
...     def _get_factor_names(self):
...         return ("ma_dev_5", "ma_dev_ts", "ma_dev_cs")
...     def _get_factor_func(self):
...         builder = Builder()
...         with builder:
...             close = Input("close")
...             dev = op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0)
...             Output(dev, "ma_dev_5")
...             Output(WindowedZScore(dev, 10), "ma_dev_ts")
...             Output(CrossSectionalZScore(dev), "ma_dev_cs")
...         return Function(builder.ops)
...
```

```python
>>> kq = MaDeviation(FactorConfig(
...     warmup_bars=10, dataset=make_dataset(), mode="batch", data_columns=("close",),
...     file_path="data/factors/ma_dev.zarr", njobs=2,
... ))
>>> out = kq.compute("2024-02-01", "2024-02-29")
>>> list(out.data_vars), dict(out.sizes)
(['ma_dev_5', 'ma_dev_ts', 'ma_dev_cs'], {'timestamp': 29, 'symbol': 8})
>>> float(abs(out["ma_dev_cs"].mean("symbol")).max()) < 1e-5
True
>>> out["ma_dev_cs"].std("symbol", ddof=1).values[:3].round(4)
array([1., 1., 1.], dtype=float32)
>>> int(out["ma_dev_ts"].isel(symbol=0).notnull().values.argmax())
3
```

The cross-sectional output has mean 0 and standard deviation 1 at every timestamp. The time-series output is NaN until the two nested windows fill. The 5-bar average and the 10-bar z-score together need 14 bars, and `warmup_bars=10` provides 10 bars before the first requested date, so the first valid value is at index 3. In stream mode, `cal_stream` advances the compiled graph by one bar and returns that bar's panel, and `symbols` must be pinned on the dataset config. The streaming result equals the batch formula:

```python
>>> import numpy as np
>>> stream = MaDeviation(FactorConfig(
...     warmup_bars=10, mode="stream", data_columns=("close",), njobs=2,
...     dataset=SpotKlineDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/klines.zarr",
...         market="crypto_spot", frequency="1d",
...         symbols=tuple(symbols),
...     )),
... ))
>>> for step in range(6):
...     row = stream.cal_stream({"close": close[step].astype("float32")}, step, symbols)
...
>>> dict(row.sizes)
{'timestamp': 1, 'symbol': 8}
>>> row["ma_dev_5"].values[0, :3].round(5)
array([-0.00857,  0.01526,  0.00988], dtype=float32)
>>> (close[5] / close[1:6].mean(axis=0) - 1)[:3].round(5)
array([-0.00857,  0.01526,  0.00988])
```

## Notes

On macOS, batch mode needs the number of symbols to be a multiple of the SIMD block width, and `compute()` pads the symbol axis with all-NaN dummy symbols to a multiple of 8 and cuts them back, so any count runs; 5 symbols without that padding fail with `RuntimeError: Bad shape at close`, a message that does not mention symbols. The padded symbols are NaN in every output of `Alpha101Stock` and `Alpha158Stock`, so the padding does not change their values. On Linux x86 (AVX2) any count runs and nothing is padded.

In stream mode every entry of `data_columns` must be consumed by an `Output`, because KunQuant prunes unused inputs. An extra column fails in `init_stream()` with `RuntimeError: Cannot find the buffer name`. Batch mode tolerates extra inputs.

A Polars factor that names a column its store does not have fails when the object is constructed, because the factor names are derived by running the expression on a few rows: `polars.exceptions.ColumnNotFoundError: unable to find column "close"; valid columns: ["timestamp", "symbol", "Close", ...]`. Use the store's own names (`Close`), not KunQuant's (`close`), except over a merged input, which carries the shared names (`close`).

`read(start, end)` and `extend(end)` need the range `build` records: a store written some other way raises `ValueError: Momentum.read(): the store at data/factors/nob.zarr has no recorded range, so it cannot answer a date-range request; write it with build(start, end).` `extend(end)` with an `end` the recorded range already reaches raises `ValueError: Momentum.extend(): the store at data/factors/momentum.zarr already covers 2024-01-21 to 2024-03-20; extend() appends only bars after 2024-03-20, got end '2024-03-10'.`

`Forward` refuses a `span` below 1 (`ValueError: Forward: span must be at least 1, got 0.`), a negative `delay`, and a factor it cannot shift: a stream-mode factor (`ValueError: Return: _TrailingOpenReturn is in 'stream' mode; a label reads bars after t, which a stream never has.`) or a resampled one (`ValueError: Forward: Momentum is resampled to '1d'; a label counts its lookahead on the dataset's own bars, so wrap an unresampled factor.`).

`warmup_bars` is a number of bars on the dataset's own calendar, so days without data are skipped, not counted. A rolling window nested inside another needs the sum of both lengths.

`FactorKunQuant.symbols` and `num_symbols` exist only in stream mode, where they are the symbols pinned on the dataset config. In batch mode a computation runs over the symbols of the requested panel, and they raise `ValueError: MaDeviation.symbols: only a stream-mode factor has a fixed symbol list; a batch computation runs over the symbols of the requested panel. config.mode is 'batch'.` Likewise `compute()` on a stream-mode factor raises `ValueError: MaDeviation.compute(): a date-range computation runs the batch graph, but config.mode is 'stream'.`

A resampled factor is a view of its source panel: `extend()`, `init_stream()` and `cal_stream()` refuse with `Momentum.extend(): a resampled factor (resample_freq='1d') is a view of its source panel and does not support extend. Compute or update the source factor, then resample it.` The built resampled store is a cache: rebuilding the source factor does not refresh it.

`FactorKunQuant.compute()` compiles the graph each time it is called. Pin `factor_names` to the columns needed to keep the graph small.

`CrossSectionalZScore` gives NaN for a timestamp with fewer than two valid values or zero spread. Batch runs must start at bar 0, which `compute()` always does. The same bar-0 rule applies to `CrossSectionalWinsorize` and `CrossSectionalTrim`. Each distinct `(lower, upper)` pair compiles its own C++ function, so the object you get back has a generated class such as `CrossSectionalWinsorize_0p01_0p99`; `isinstance(op, CrossSectionalWinsorize)` still holds.

## See also

`backend.md` for `XrBackend` and the append checks behind `extend()`; `dataset.md` for the datasets factors read; `model.md` for how models consume factors and labels and purge each split; `backtest.md` for the check of a label's delay against the engine's fill delay. Modules: `quantlab.base.factor` (`Factor`, `FactorKunQuant`, `FactorPolars`), `quantlab.base.config` (`FactorConfig`, `PolarsFactorConfig`, `MarketFeatureConfig`), `quantlab.factor`, `quantlab.label.forward`, `quantlab.label.predefined.fret` and `quantlab.my_ops.preprocess`.
