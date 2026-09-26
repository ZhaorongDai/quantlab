# Factors

English | [简体中文](zh-CN/factor.md)

A factor turns the `(timestamp, symbol)` panel held by a dataset into a panel of engineered features of the same shape. A label is a factor whose output is a prediction target. quantlab has two factor backends: `FactorKunQuant`, which compiles a KunQuant operator graph to native code and runs it in batch or streaming mode, and `FactorPolars`, whose logic is a Polars expression chain and which runs in batch mode only. Both share the base class `Factor`, so the model layer treats them alike.

## Prerequisites

KunQuant factors compile C++ at run time and need a working C++ compiler. Polars factors do not. The examples below run against synthetic data written under the working directory. A crypto spot dataset stores raw Binance column names (`Close`, `Volume`); the dataset guide (`dataset.md`) describes the dataset config fields.

## The basics

### Two backends

| | `FactorKunQuant` | `FactorPolars` |
|---|---|---|
| Logic is written as | a KunQuant graph in `_get_factor_func` | a Polars expression chain in `_get_factor_lazyframe` |
| Modes | batch and streaming | batch only |
| Config class | `FactorConfig` | `PolarsFactorConfig` |
| Input column names | renamed by the dataset (`close`, `amount`, `adjClose`) | the store's own names (`Close`) |
| Output dtype | float32 | float64 |
| Cost | compiles the graph on every `cal()` | none |

KunQuant is the primary backend: it has the rolling and cross-sectional operators the shipped alpha libraries are built from, and it is the only one that can run bar by bar on live data. `FactorPolars` is the supplementary path for new factors that are easier to state as DataFrame expressions or that use operations KunQuant lacks. A factor that must also run in streaming mode has to be a KunQuant factor.

### The config

Both backends take a config with the fields below. `FactorConfig` adds `mode` (`"batch"` or `"stream"`), `data_columns` (the dataset variables fed to the graph) and `njobs` (executor threads). `PolarsFactorConfig` adds nothing.

| Field | Meaning |
|---|---|
| `window` | lookback in calendar days, read before `start_date` so rolling operators are warm |
| `dataset` | the dataset the factor reads |
| `file_path` | Zarr store the factor is saved to and read from |
| `factor_names` | output column names; derived from the factor when left `None` |
| `start_date`, `end_date` | first and last date to compute, inclusive |
| `symbols` | restrict to these symbols |
| `kwargs` | free-form options a factor class reads |

The factor's own `start_date` and `end_date` decide the output range. When they are left unset they default to an open-ended range, and the factor moves the dataset's dates to `start_date` minus `window` days through `end_date`. Set the dates on the factor config, not on the dataset config.

### Compute and save a factor

The session first writes a small synthetic spot store and defines a helper that builds a dataset over it. `Momentum` is the reference Polars factor: `Close_t / Close_{t-n} - 1`, with `n` read from `kwargs`. Factor names are known as soon as the object is constructed, before anything is computed, and the dataset's start date has been moved 20 days earlier to warm the window.

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.backend import XrBackend
>>> from quantlab.base.config import DatasetConfig, PolarsFactorConfig
>>> from quantlab.dataset.spot import SpotKlineDataset
>>> from quantlab.factor.momentum import Momentum
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

```python
>>> config = PolarsFactorConfig(
...     window=20, dataset=make_dataset(), file_path="data/factors/momentum.zarr",
...     kwargs={"n": 5}, start_date="2024-02-01", end_date="2024-02-29",
... )
>>> factor = Momentum(config)
>>> factor.get_factor_names()
('momentum_5',)
>>> factor.config.dataset.config.start_date
'2024-01-12'
>>> panel = factor.cal().get_features()
>>> dict(panel.sizes), list(panel.data_vars)
({'timestamp': 29, 'symbol': 8}, ['momentum_5'])
>>> panel["momentum_5"].isel(timestamp=0, symbol=slice(0, 3)).values.round(4)
array([ 0.0161,  0.0079, -0.0007])
>>> factor.save(mode="w") is factor
True
```

`cal()` computes the panel and holds it in the factor's storage backend. `get_features()` returns it as an `xarray.Dataset` over `(timestamp, symbol)`. `save(mode="w")` writes it to `file_path` as a Zarr store, replacing what was there.

## Common tasks

### Extend a stored factor with later dates

`update()` appends the held panel to the existing store, widening the timestamp, symbol and variable axes first. It uses `XrBackend.widen_and_append`, so the append checks described in the backend guide apply. Use `save(mode="w")` to replace a store, and `update()` to extend it.

```python
>>> later = Momentum(PolarsFactorConfig(
...     window=20, dataset=make_dataset(), file_path="data/factors/momentum.zarr",
...     kwargs={"n": 5}, start_date="2024-03-01", end_date="2024-03-20",
... ))
>>> later.cal().get_features().sizes["timestamp"]
20
>>> later.update() is later
True
>>> stored = xr.open_zarr("data/factors/momentum.zarr")
>>> stored.sizes["timestamp"], str(stored["timestamp"].values[-1])[:10]
(49, '2024-03-20')
```

### Read a factor back, and rebuild it from its config

`read()` opens the store and narrows it to the configured dates. `get_config()` returns a dict describing the factor and its dataset, and `load_factor_from_config` rebuilds the factor from it.

```python
>>> reader = Momentum(PolarsFactorConfig(
...     window=20, dataset=make_dataset(), file_path="data/factors/momentum.zarr",
...     kwargs={"n": 5}, start_date="2024-03-10", end_date="2024-03-15",
... ))
>>> reader.read().get_features().sizes["timestamp"]
6
>>> cfg = factor.get_config()
>>> cfg["name"], cfg["kwargs"], cfg["dataset"]["market"]
('quantlab.factor.momentum.Momentum', {'n': 5}, 'crypto_spot')
>>> from quantlab.utils.module import load_factor_from_config
>>> rebuilt = load_factor_from_config(cfg)
>>> type(rebuilt).__name__, rebuilt.get_factor_names()
('Momentum', ('momentum_5',))
```

### Resample a factor onto coarser bars

`resample(freq, how)` returns a copy of the factor whose computed panel is aggregated onto coarser bars. The factor is still computed on its dataset's own bars; only the output is aggregated, so a minute-bar momentum becomes a daily series of the last minute's value without changing what it measures. `freq` and `how` take the same values as `BaseDataset.resample` (see the dataset guide), and `how` may be one method as a string for every factor variable. Bars are cut the way the factor's dataset cuts them.

The session below runs `Momentum` over the two-day minute store built in the dataset guide (`config` is that store's `DatasetConfig`).

```python
>>> factor = Momentum(PolarsFactorConfig(
...     window=1, dataset=SpotKlineDataset(config),
...     file_path="data/factors/momentum.zarr", kwargs={"n": 1},
... ))
>>> minute = factor.cal()
>>> minute.get_features()["momentum_1"].to_pandas().round(3)
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
>>> daily.get_features()["momentum_1"].to_pandas().round(3)
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02    0.333    0.333
2024-01-03    0.143    0.143
>>> daily.config.dataset.config.resample_freq, daily.config.resample_freq
(None, '1d')
```

The copy has its own dataset object and an empty compiled state, so `cal()` on it computes the minute panel again and resamples it. `save()` writes to `store_path`, beside the source store, and `read()` on a copy opens that store when it exists and otherwise resamples the source store. The request round-trips through `get_config()` and `load_factor_from_config`.

```python
>>> daily.store_path
'data/factors/momentum_resample_1d.zarr'
>>> daily.cal().get_features().sizes
Frozen({'symbol': 2, 'timestamp': 2})
>>> cfg = daily.get_config()
>>> cfg["resample_freq"], cfg["resample_how"], cfg["dataset"]["resample_freq"]
('1d', 'last', None)
>>> load_factor_from_config(cfg).cal().get_features().sizes
Frozen({'symbol': 2, 'timestamp': 2})
```

To compute a factor on already-resampled bars instead, resample the dataset and give the factor the resampled dataset.

### Compute a label

A label is a KunQuant factor whose `get_labels()` returns a forward-looking value. `Return` in `quantlab.label.fret` is the return from the next bar's adjusted open to the adjusted open `n` bars after that, and `BinaryReturn` is 1.0 when that return is positive. The graph computes a trailing return, since KunQuant can only look backwards, and `get_labels()` shifts it forward by `n_forward_periods + 1` bars. The last `n_forward_periods + 1` bars are NaN. The labels read `adjOpen`, so the dataset must carry adjusted prices; a US equity dataset does, the crypto spot dataset does not.

```python
>>> from quantlab.dataset.stock import StockDataset
>>> from quantlab.label.fret import Return
>>> px = 50 + np.cumsum(rng.normal(size=(30, 8)), axis=0)
>>> stock = xr.Dataset(
...     {"adjOpen": (["timestamp", "symbol"], px)},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=30), "symbol": [f"T{i}" for i in range(8)]},
... )
>>> XrBackend().to_internal(stock).write("data/stock.zarr")
XrBackend()
>>> label = Return(FactorConfig(
...     window=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 2},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/stock.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     file_path="data/labels/ret.zarr", njobs=2,
... ))
>>> label.get_factor_names()
('ret_2',)
>>> ret = label.cal().get_labels()["ret_2"]
>>> dict(ret.sizes)
{'timestamp': 30, 'symbol': 8}
>>> ret.isel(symbol=0).values[:2].round(5)
array([-0.03945,  0.01532], dtype=float32)
>>> round(float(px[3, 0] / px[1, 0] - 1), 5)
-0.03945
>>> ret.isel(symbol=0).values[-3:]
array([nan, nan, nan], dtype=float32)
```

`get_features()` on a label returns the unshifted trailing return, which must not be used as a target.

### Analyze a factor

`analyze()` reports how well a factor orders symbols by their forward return, in the manner of the alphalens library. Pass one or more forward-return labels as `frets`; every factor variable (all of `get_factor_names()`, or the ones named in `factor_names`) is paired with every label variable. Both the factor and the labels must already hold their panels, from `cal()` or `read()`. The two panels must have the same most common bar spacing, the rule `BaseDataset.time_interval` uses, or `analyze()` raises `ValueError` naming both spacings; they are then joined on their common timestamps and symbols.

Each pair gets:

| Group | Metrics |
|---|---|
| Information | per-period IC (Spearman rank correlation across symbols), IC mean, std, IR (mean / std), t-statistic, p-value, skew, excess kurtosis, share of positive periods, monthly mean IC |
| Returns | mean forward return per factor quantile (bucket 1 holds the lowest values), top-minus-bottom spread per period, cumulative return per quantile and long-short |
| Turnover | share of each quantile's symbols that were not in it the period before, lag-1 factor rank autocorrelation |

`quantiles` (default 5) sets the number of equal-count buckets. When a label spans `n` bars (`kwargs["n_forward_periods"]`), cumulative returns compound the per-bar rate `(1 + r) ** (1 / n) - 1`. The example below builds a one-bar `Return` label over the same eight symbols as `factor` from the first section, then analyzes `momentum_5` against it. On this random walk the IC is near zero, as it should be.

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
...     window=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 1},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/spot_open.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     start_date="2024-02-01", end_date="2024-02-29",
...     file_path="data/labels/fwd_ret.zarr", njobs=2,
... ))
>>> fwd.cal() is fwd
True
>>> result = factor.analyze(frets=[fwd], quantiles=4, output_dir="data/analysis/momentum")
>>> list(result.pairs)
['momentum_5__ret_1']
>>> pair = result.pairs["momentum_5__ret_1"]
>>> round(pair.summary["ic_mean"], 4), round(pair.summary["ir"], 4), pair.summary["n_periods"]
(-0.0494, -0.1475, 29)
>>> pair.mean_quantile_returns.round(4).tolist()
[0.001, -0.0025, 0.0004, -0.001]
>>> sorted(os.listdir("data/analysis/momentum"))
['config.json', 'ic.csv', 'momentum_5__ret_1.png', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv']
>>> import json
>>> from quantlab.utils.module import load_factor_from_config
>>> cfg = json.load(open("data/analysis/momentum/config.json"))
>>> list(cfg), type(load_factor_from_config(cfg["frets"][0])).__name__
(['factor', 'frets'], 'Return')
```

The result carries `pairs` (a `PairAnalysis` per `"<factor>__<fret>"`, with the IC series, its running sum `cumulative_ic`, quantile returns, turnover and a `summary` dict), `figures` (one matplotlib figure per pair, held only when no `output_dir` is given) and tidy tables from `summary_table()`, `ic_table()`, `monthly_ic_table()`, `quantile_returns_table()` and `turnover_table()`. With `output_dir`, those tables are written as CSV, the scalar metrics as `summary.json`, each figure as `<factor>__<fret>.png`, and `config.json` holds the factor's and the labels' configs, each rebuildable with `load_factor_from_config`; the figures are then drawn on every CPU in parallel straight to the PNG files and not kept, since drawing dominates the cost of a report over a whole alpha library. Without `output_dir` nothing is written. The figures are built without `pyplot`, so they are never shown and need no closing; `fig.savefig(path)` writes one. The IC panel draws the cumulative IC on its right axis; the turnover and rank-autocorrelation panels draw each series as a translucent rolling range (minimum to maximum over `rolling_window` periods, 22 by default) with its rolling mean, not the raw per-period values. The metrics are computed with polars: every factor variable of a chunk (`chunk_size`, default 32) is one lazy plan over the long `(timestamp, symbol)` frame, collected once. The machinery lives in `quantlab.analysis.factor_report`.

### Normalize over time or across symbols

`quantlab.my_ops.preprocess` has four KunQuant operators. `WindowedZScore` standardizes each symbol against its own trailing window, a time-series normalization. `CrossSectionalZScore` standardizes each timestamp across all symbols. Which one is right depends on the strategy consuming the factor. `Alpha101SpotKline` and `Alpha158SpotKline` apply `WindowedZScore` to every output; `Alpha101Stock` and `Alpha158Stock` apply `CrossSectionalZScore` to every output. The KunQuant factor under Extending applies both operators.

The module also has two cross-sectional outlier operators. `CrossSectionalWinsorize(v, lower=0.01, upper=0.99)` (winsorizing) clips each timestamp's values to that bar's `lower` and `upper` quantiles across symbols, and `CrossSectionalTrim(v, lower=0.01, upper=0.99)` (trimming) sets values strictly outside those quantiles to NaN. Quantiles ignore NaN and interpolate linearly, like `np.nanquantile`. A common chain is `CrossSectionalZScore(CrossSectionalWinsorize(v))`, so a few extreme symbols do not dominate the mean and standard deviation. KunQuant 0.1.11 has no built-in operator for either: its `Clip` bounds by a fixed constant and `WindowedQuantile` works along time.

### Shipped factors

| Class | Backend | Notes |
|---|---|---|
| `Momentum` | Polars | reference Polars factor, reads `Close` |
| `Alpha101SpotKline`, `Alpha101Stock` | KunQuant | KunQuant's Alpha101 library |
| `Alpha158SpotKline`, `Alpha158Stock` | KunQuant | Alpha158 features; pin `factor_names` while experimenting |
| `ResidualMomentumFF3` | KunQuant | Fama-French three-factor residual momentum on monthly data |
| `Return`, `BinaryReturn` | KunQuant | forward-return labels |

Each class docstring shows its config.

## Extending

### A Polars factor

Subclass `FactorPolars` and implement `_get_factor_lazyframe`. It receives the dataset as a `polars.LazyFrame` and returns a lazy frame with only `timestamp`, `symbol` and the factor columns. Factor names are read from the returned schema. Override `_get_features` to return the panel; without it `get_features()` raises `NotImplementedError`.

```python
>>> import polars as pl
>>> from quantlab.base.factor import FactorPolars
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
...     def _get_features(self, data):
...         return data
...
>>> rv = RelativeVolume(PolarsFactorConfig(
...     window=10, dataset=make_dataset(), file_path="data/factors/rel_volume.zarr",
...     start_date="2024-02-01", end_date="2024-02-29",
... ))
>>> rv.get_factor_names()
('rel_volume_5',)
>>> out = rv.cal().get_features()
>>> dict(out.sizes), str(out["rel_volume_5"].dtype)
({'timestamp': 29, 'symbol': 8}, 'float64')
>>> int(out["rel_volume_5"].isnull().sum())
0
```

### A KunQuant factor

Subclass `FactorKunQuant` and implement `_get_factor_names`, `_get_factor_func` (the KunQuant graph, with one `Input` per entry of `data_columns` and one `Output` per factor name) and `_get_features`. The graph below outputs a moving-average deviation raw, z-scored over time and z-scored across symbols. KunQuant compiles on the first `cal()` (about a second here).

```python
>>> import KunQuant.ops as op
>>> from KunQuant.Op import Builder, Input, Output
>>> from KunQuant.Stage import Function
>>> from quantlab.base.config import FactorConfig
>>> from quantlab.base.factor import FactorKunQuant
>>> from quantlab.my_ops.preprocess import CrossSectionalZScore, WindowedZScore
>>> class MaDeviation(FactorKunQuant):
...     def _get_factor_names(self):
...         return ("ma_dev_5", "ma_dev_ts", "ma_dev_cs")
...     def _get_features(self, data):
...         return data
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
...     window=10, dataset=make_dataset(), mode="batch", data_columns=("close",),
...     file_path="data/factors/ma_dev.zarr", start_date="2024-02-01",
...     end_date="2024-02-29", njobs=2,
... ))
>>> out = kq.cal().get_features()
>>> list(out.data_vars), dict(out.sizes)
(['ma_dev_5', 'ma_dev_ts', 'ma_dev_cs'], {'timestamp': 29, 'symbol': 8})
>>> float(abs(out["ma_dev_cs"].mean("symbol")).max()) < 1e-5
True
>>> out["ma_dev_cs"].std("symbol", ddof=1).values[:3].round(4)
array([1., 1., 1.], dtype=float32)
>>> int(out["ma_dev_ts"].isel(symbol=0).notnull().values.argmax())
3
```

The cross-sectional output has mean 0 and standard deviation 1 at every timestamp. The time-series output is NaN until the two nested windows fill. The 5-bar average and the 10-bar z-score together need 14 bars, and `window=10` provides 10 bars before the first requested date, so the first valid value is at index 3. In stream mode, `cal_stream` advances the compiled graph by one bar at a time, and `symbols` must be pinned on the dataset config. The streaming result equals the batch formula:

```python
>>> import numpy as np
>>> stream = MaDeviation(FactorConfig(
...     window=10, mode="stream", data_columns=("close",), njobs=2,
...     dataset=SpotKlineDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/klines.zarr",
...         market="crypto_spot", frequency="1d",
...         symbols=tuple(symbols),
...     )),
... ))
>>> for step in range(6):
...     row = stream.cal_stream({"close": close[step].astype("float32")}, step, symbols).get_features()
...
>>> dict(row.sizes)
{'timestamp': 1, 'symbol': 8}
>>> row["ma_dev_5"].values[0, :3].round(5)
array([-0.00857,  0.01526,  0.00988], dtype=float32)
>>> (close[5] / close[1:6].mean(axis=0) - 1)[:3].round(5)
array([-0.00857,  0.01526,  0.00988])
```

## Notes

On macOS, batch mode needs the number of symbols to be a multiple of the SIMD block width, and `cal()` pads the symbol axis with all-NaN dummy symbols to a multiple of 8 and cuts them back, so any count runs; 5 symbols without that padding fail with `RuntimeError: Bad shape at close`, a message that does not mention symbols. On Linux x86 (AVX2) any count runs and nothing is padded.

In stream mode every entry of `data_columns` must be consumed by an `Output`, because KunQuant prunes unused inputs. An extra column fails in `init_stream()` with `RuntimeError: Cannot find the buffer name`. Batch mode tolerates extra inputs.

`get_features()` raises a bare `NotImplementedError` when the factor class does not override `_get_features`. Likewise `get_labels()` on a features-only class raises `RuntimeError: Momentum does not support get_label()`.

A Polars factor that names a column its store does not have fails when the object is constructed, because the factor names are derived by running the expression on a few rows: `polars.exceptions.ColumnNotFoundError: unable to find column "close"; valid columns: ["timestamp", "symbol", "Close", ...]`. Use the store's own names (`Close`), not KunQuant's (`close`).

`save()` defaults to `mode="a"`, which in Zarr means overwrite variables of an existing store and is not an append along time. Saving a panel of a different length raises `ValueError: Momentum.save(mode="a"): cannot write this date range into the existing store at data/f/m.zarr. zarr's "a" means "overwrite variables in an existing store", NOT "append along time", ...`. Use `save(mode="w")` to replace the store or `update()` to extend it.

`window` is a number of calendar days for the dataset lookback, not a number of bars. Where the market is closed on some days the same number gives fewer bars, and a rolling window nested inside another needs the sum of both lengths.

A resampled factor is a view of its source panel: `update()`, `init_stream()` and `cal_stream()` refuse with `Momentum.update(): a resampled factor (resample_freq='1d') is a view of its source panel and does not support update. Compute or update the source factor, then resample it.` The saved resampled store is a cache: recomputing the source factor does not refresh it.

`FactorKunQuant.cal()` compiles the graph each time it is called. Pin `factor_names` to the columns needed to keep the graph small.

`CrossSectionalZScore` gives NaN for a timestamp with fewer than two valid values or zero spread. Batch runs must start at bar 0, which `cal()` always does. The same bar-0 rule applies to `CrossSectionalWinsorize` and `CrossSectionalTrim`. Each distinct `(lower, upper)` pair compiles its own C++ function, so the object you get back has a generated class such as `CrossSectionalWinsorize_0p01_0p99`; `isinstance(op, CrossSectionalWinsorize)` still holds.

## See also

`backend.md` for `XrBackend` and the append checks behind `update()`; `dataset.md` for the datasets factors read; `model.md` for how models consume `get_features()` and `get_labels()`. Modules: `quantlab.base.factor` (`Factor`, `FactorKunQuant`, `FactorPolars`), `quantlab.base.config` (`FactorConfig`, `PolarsFactorConfig`), `quantlab.factor`, `quantlab.label.fret` and `quantlab.my_ops.preprocess`.
