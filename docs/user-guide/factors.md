# Factors and labels

This page explains how quantlab turns a price panel into model inputs
(factors) and prediction targets (labels). It covers the two computation
backends, KunQuant and Polars, and when to use each; the built-in factor
sets; computing, saving and reading factor values; writing your own factor;
the normalisation operators; the forward-return labels; and KunQuant's
streaming mode. Read it after [Datasets](datasets.md) and before
[Models](models.md).

## Factors, labels and panels

A panel is an `xarray.Dataset` indexed by two dimensions, `timestamp` and
`symbol`. Every variable in it is a `[time, symbol]` array. A dataset turns
raw vendor files into such a panel of prices. A factor reads that panel and
produces a new panel of the same shape. Each variable in the new panel is one
engineered feature, such as a five-day return or a volatility estimate.

A label has the same shape. It holds the value a model should learn to
predict at each `(timestamp, symbol)`, usually the return of the next few
bars. quantlab builds labels with the same classes as factors. The only
difference is which method you call. `get_features()` returns a factor's
values as model inputs and `get_labels()` returns a label's values as
targets. The built-in factor classes raise `RuntimeError` from
`get_labels()`, and the label classes return something other than the label
from `get_features()` (see [Forward-return labels](#forward-return-labels)).

Every factor is configured by a dataclass and built around a dataset object.
The config fields shared by both backends live on
`quantlab.base.config.BaseFactorConfig`:

- `window`: warm-up in calendar days. The factor reads this many days of
  history before `start_date` so that rolling computations already have a full
  window on the first requested bar.
- `dataset`: the dataset the factor reads prices from.
- `file_path`: the Zarr store the factor values are saved to and read from.
- `factor_names`: which outputs to produce. Left `None`, it is filled with
  every name the class can produce.
- `start_date`, `end_date`, `symbols`: the window to compute. Left `None`,
  they cover everything the dataset holds.
- `kwargs`: free-form options a particular class reads, such as a horizon.

## Choosing a backend

quantlab has two factor backends. Both produce the same kind of panel, so a
model can take factors from both at once.

| | KunQuant (`FactorKunQuant`) | Polars (`FactorPolars`) |
|---|---|---|
| Config class | `FactorConfig` | `PolarsFactorConfig` |
| Factor logic | a graph of KunQuant operators, compiled to native code | a Polars lazy expression chain |
| Modes | batch (`cal()`) and streaming (`cal_stream()`) | batch only |
| Built-in sets | Alpha101, Alpha158, residual momentum, labels | `Momentum` (a reference example) |
| Input columns | named in `data_columns` | whatever columns the store holds |

KunQuant is the main backend. It runs the same compiled graph over a whole
history in batch mode or one bar at a time in streaming mode. A factor you
validate in a backtest therefore runs unchanged on live data. Use it for any
factor that will trade live and for anything the built-in libraries already
provide. Compilation needs a C++ compiler and takes a few seconds, which
dominates the run time on small panels.

Polars is the lighter path for research factors that will only ever run in
batch. You write an ordinary Polars expression, and nothing is compiled.
Prefer KunQuant when the factor can be expressed with its operators. Reach
for Polars when the computation is awkward as an operator graph, or while you
are still trying out an idea.

## A price store to follow along

The examples on this page share one small synthetic US-equity store with
eight symbols and 120 business days. It has the same layout the Tiingo
converter writes: raw `open`, `high`, `low`, `close` and `volume`, plus their
split-adjusted versions `adjOpen` to `adjVolume`.

```python
import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import replace

from quantlab.base.config import DatasetConfig
from quantlab.dataset.stock import StockDataset

rng = np.random.default_rng(0)
symbols = [f"S{i:02d}" for i in range(8)]
dates = pd.bdate_range("2024-01-01", periods=120)
close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (120, 8)), axis=0))
volume = rng.uniform(1e5, 1e6, (120, 8))
fields = {"open": close * 0.995, "high": close * 1.01, "low": close * 0.99,
          "close": close, "volume": volume}
fields |= {"adj" + k.capitalize(): v for k, v in fields.items()}   # adjOpen, ...
xr.Dataset({k: (("timestamp", "symbol"), v) for k, v in fields.items()},
           coords={"timestamp": dates, "symbol": symbols}).to_zarr("prices.zarr", mode="w")

price_config = DatasetConfig(zarr_file_path="prices.zarr", raw_data_dir_path="raw",
                             market="us_equity", frequency="1d")
```

Give every factor its own dataset object, for example
`StockDataset(replace(price_config))`. A factor moves its dataset's start
date back by `window` days for warm-up, so two factors sharing one dataset
object would change each other's dates.

## Compute a built-in factor set

The built-in sets are thin wrappers around KunQuant's predefined libraries:

| Class | Module | Reads | Output |
|---|---|---|---|
| `Alpha101SpotKline` | `quantlab.factor.alpha101` | crypto klines: `open`, `high`, `low`, `close`, `volume`, `amount` | Alpha101 formulas, z-scored along time |
| `Alpha101Stock` | `quantlab.factor.alpha101` | US equities: `adjOpen` to `adjVolume` | Alpha101 formulas, z-scored across symbols |
| `Alpha158SpotKline` | `quantlab.factor.alpha158` | crypto klines, as above | 169 Alpha158 features, z-scored along time |
| `Alpha158Stock` | `quantlab.factor.alpha158` | US equities: `adjOpen` to `adjVolume` | 169 Alpha158 features, z-scored across symbols |
| `ResidualMomentumFF3` | `quantlab.factor.residual_momentum` | monthly returns plus Fama-French factors | residual momentum and regression diagnostics |

Alpha101 is the public list of 101 formulaic trading signals from
Kakushadze (2016). Alpha158 is the feature library of Microsoft's Qlib
project. It has candle-shape features (`KMID`, `KLEN`, ...), prices and
volumes lagged 0 to 4 bars (`CLOSE1`, `VOLUME3`, ...) and rolling statistics
over 5 to 60 bars (`ROC5`, `STD20`, `CORR60`, ...). The crypto variants
z-score every output against its own trailing window, which suits strategies
that follow one asset over time. The equity variants z-score every output
across the symbols of the same bar (see
[Normalisation operators](#normalisation-operators)).

Compute three Alpha158 features for February onwards:

```python
from quantlab.base.config import FactorConfig
from quantlab.factor.alpha158 import Alpha158Stock

alpha = Alpha158Stock(FactorConfig(
    window=30,
    dataset=StockDataset(replace(price_config)),
    mode="batch",
    data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
    factor_names=("KMID", "ROC5", "STD20"),
    start_date="2024-02-01",
    file_path="factors/alpha158.zarr",
    njobs=4,
))
panel = alpha.cal().get_features()
print(dict(panel.sizes), list(panel.data_vars))
print(alpha.config.dataset.config.start_date)
```

```text
{'timestamp': 97, 'symbol': 8} ['KMID', 'ROC5', 'STD20']
2024-01-02
```

The panel starts on the requested 1 February. The dataset itself was read
from 2 January, 30 calendar days earlier, so that `STD20` already has twenty
bars behind it on the first bar you asked for. `mode="batch"` compiles the
graph for whole-history runs, and `njobs` sets the number of executor
threads (the default is 128). Pinning `factor_names` keeps the compiled graph
small, and the full Alpha158 set compiles noticeably more slowly. Without the
pin, `factor_names` resolves to all 169 names as soon as the object is built.

The US-equity stores carry no dollar-volume (`amount`) column, so
`Alpha101Stock` and `Alpha158Stock` both use the adjusted typical price
`(adjHigh + adjLow + adjClose) / 3` as VWAP.

`ResidualMomentumFF3` expects a monthly panel that already carries each
stock's return and the Fama-French market, size and value factors
(`mkt_rf`, `smb`, `hml`, `risk_free`), broadcast over symbols. It estimates
the three-factor regression over a rolling window and ranks stocks on their
residual return. See its class docstring for the parameters it reads from
`kwargs`.

## Save and read factor values

`save()` writes the held panel to `config.file_path` as a Zarr store with one
array per factor. `read()` opens that store later and narrows it to the
configured dates, without computing anything:

```python
alpha.save(mode="w")

again = Alpha158Stock(replace(alpha.config, dataset=StockDataset(replace(price_config))))
print(dict(again.read().get_features().sizes))
```

```text
{'timestamp': 97, 'symbol': 8}
```

`mode="w"` replaces the store. The default `mode="a"` rewrites variables of
an existing store in place. It does not append along time, and it refuses a
panel whose time or symbol axis differs from the stored one. To extend a store
with a later date range, compute the new range and call `update()`, which
widens the time, symbol and variable axes as needed and appends. If you change
the dates of a factor that has already read its store, call
`read(overwrite=True)` so the store is opened again rather than the cached,
already narrowed panel being reused.

A model decides between the two paths through `factor_data_strategy` and
`label_data_strategy`. `"cal"` computes every factor when the model collects
its data, and `"read"` loads the stores you saved earlier (see
[Models](models.md)).

## Write a Polars factor

A Polars factor is a subclass of `quantlab.base.factor.FactorPolars` that
overrides one method, `_get_factor_lazyframe`. It receives the dataset as a
`polars.LazyFrame` with `timestamp` and `symbol` columns plus the store's own
columns, and returns a lazy frame with exactly `timestamp`, `symbol` and the
factor columns. `cal()` collects it and turns it into a panel. The factor
names are read from the schema of the returned frame, so you never declare
them separately:

```python
import polars as pl
from quantlab.base.config import PolarsFactorConfig
from quantlab.base.factor import FactorPolars

class VolumeSurprise(FactorPolars):
    """Today's volume relative to its trailing mean, per symbol."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = (self.config.kwargs or {}).get("n", 20)
        volume = pl.col("adjVolume")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (volume / volume.rolling_mean(n).over("symbol") - 1.0)
                .alias(f"vol_surprise_{n}")
            )
            .select(["timestamp", "symbol", f"vol_surprise_{n}"])
        )

    def _get_features(self, data):
        return data

surprise = VolumeSurprise(PolarsFactorConfig(
    window=30,
    dataset=StockDataset(replace(price_config)),
    kwargs={"n": 10},
    file_path="factors/vol_surprise.zarr",
))
print(surprise.get_factor_names())
print(dict(surprise.cal().get_features().sizes))
```

```text
('vol_surprise_10',)
{'timestamp': 120, 'symbol': 8}
```

Three details matter. Sort by symbol and time and use `.over("symbol")` so
that shifts and rolling windows stay within one symbol. End with the
`.select(...)`: any price column left in the frame would be stored as a
factor. And spell columns exactly as the store does. No renaming happens on
the Polars path, which is why the built-in `quantlab.factor.momentum.Momentum`
reads `Close` and works only on the crypto kline store. Reading the names
from the schema means the constructor reads a few rows of the store, so the
store must exist before you build the factor.

## Write a KunQuant factor

A KunQuant factor subclasses `quantlab.base.factor.FactorKunQuant` and
implements `_get_factor_names` and `_get_factor_func`. The second one builds
an operator graph. It has an `Input` for each column in `data_columns`,
KunQuant operators such as `WindowedAvg` or `BackRef` (the value `n` bars
earlier), and one `Output` per factor name. Override `_get_features` to return
the panel unchanged. The next section's example builds such a factor.

KunQuant operators look only backwards in time, so a factor graph cannot
peek at the future by accident. The operator catalogue lives in
`KunQuant.ops`, and the Alpha101 and Alpha158 sources in `quantlab/factor/`
show larger graphs.

## Normalisation operators

Raw factor values often live on very different scales, and a model or a
ranking rule usually wants them standardised. `quantlab.my_ops.preprocess`
provides two KunQuant operators that standardise along different axes:

- `WindowedZScore(x, window)` is a time-series z-score. Each symbol is
  compared with its own trailing `window` bars, `(x - rolling mean) / rolling
  std`. The first `window - 1` bars are NaN.
- `CrossSectionalZScore(x)` is a cross-sectional z-score. At every timestamp
  it subtracts the mean over all symbols and divides by their sample standard
  deviation, ignoring NaN.

Which one to use depends on the strategy. A strategy that ranks stocks
against each other on the same day wants the cross-sectional version. A
strategy that trades one asset when it is unusually high relative to its own
history wants the time-series version. The following factor computes one
feature, the close's distance from its ten-day mean, in all three forms:

```python
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from quantlab.base.factor import FactorKunQuant
from quantlab.my_ops.preprocess import CrossSectionalZScore, WindowedZScore

class MaDeviation(FactorKunQuant):
    """Distance of the close from its 10-day mean, raw and normalised."""

    def _get_factor_names(self):
        return ("ma_dev_10", "ma_dev_10_ts", "ma_dev_10_cs")

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("adjClose")
            dev = op.SubConst(op.Div(close, op.WindowedAvg(close, 10)), 1.0)
            Output(dev, "ma_dev_10")
            Output(WindowedZScore(dev, 20), "ma_dev_10_ts")      # along time
            Output(CrossSectionalZScore(dev), "ma_dev_10_cs")    # across symbols
        return Function(builder.ops)

    def _get_features(self, data):
        return data

dev = MaDeviation(FactorConfig(
    window=40, dataset=StockDataset(replace(price_config)), mode="batch",
    data_columns=("adjClose",), file_path="factors/ma_dev.zarr", njobs=4,
)).cal().get_features()
day = dev.isel(timestamp=-1)
print(round(day["ma_dev_10_cs"].mean().item(), 6), round(day["ma_dev_10_cs"].std(ddof=1).item(), 6))
print(np.round(day["ma_dev_10_cs"].values, 2))
```

```text
-0.0 1.0
[ 0.3  -1.5  -0.93  1.12 -0.87  1.27  0.27  0.35]
```

On the last day the cross-sectional column has mean 0 and standard deviation
1 across the eight symbols, as expected. `CrossSectionalZScore` has two
restrictions of its own: a batch run must start at bar 0, which `cal()`
always does, and it has no parameters, so a variant with different behaviour
needs a class of its own.

## Forward-return labels

`quantlab.label.fret` provides the two standard labels. Both read `adjOpen`
and take the horizon `n` from `kwargs["n_forward_periods"]`.

- `Return` is the regression target `ret_{n}`, the return from the open of
  bar `t + 1` to the open of bar `t + n + 1`.
- `BinaryReturn` is the classification target `ret_binary_{n}`: 1.0 when
  that return is positive and 0.0 otherwise.

The label at bar `t` starts at the next bar's open, because a signal formed
at the close of bar `t` cannot trade before then. The last `n + 1` bars have
no label and are NaN.

```python
from quantlab.label.fret import BinaryReturn, Return

ret = Return(FactorConfig(
    window=0, dataset=StockDataset(replace(price_config)), mode="batch",
    data_columns=("adjOpen",), kwargs={"n_forward_periods": 5},
    file_path="labels/ret_5.zarr", njobs=4,
))
labels = ret.cal().get_labels()
print(list(labels.data_vars), int(labels["ret_5"].isnull().all("symbol").sum()))

o = xr.open_zarr("prices.zarr")["adjOpen"]
t = 10
print(float(labels["ret_5"][t, 0]), float(o[t + 6, 0] / o[t + 1, 0] - 1))

up = BinaryReturn(replace(ret.config, dataset=StockDataset(replace(price_config)),
                          file_path="labels/up_5.zarr", factor_names=None))
print(up.get_factor_names(), np.unique(up.cal().get_labels()["ret_binary_5"].values[:-6]))
```

```text
['ret_5'] 6
0.023827195167541504 0.0238272174419778
('ret_binary_5',) [0. 1.]
```

The check on bar 10 matches the hand computation up to float32 rounding.
KunQuant can only look backwards, so the label graph computes the trailing
return and `get_labels()` shifts it `n + 1` bars earlier. `get_features()` on
a label returns the unshifted trailing return, which is not a label. Always
put label objects in a model's `labels` list, where the model calls
`get_labels()`.

`factor_names=None` in the `BinaryReturn` call matters. `replace` copies the
`Return` config, and that config's `factor_names` was already filled in with
`("ret_5",)`. Resetting it lets the new class fill in its own name.

## Streaming mode

In streaming mode a KunQuant factor is compiled for one bar at a time and
keeps its rolling state between calls. This is how a factor runs on live
data. Build the factor with `mode="stream"`, pin the symbol list on the
dataset config, and feed one `float32` array per input column on each call
to `cal_stream`. Here the whole history is replayed, and the last streamed
bar matches the batch result from the previous section:

```python
stream_config = replace(price_config, symbols=tuple(symbols))
live = MaDeviation(FactorConfig(
    window=40, dataset=StockDataset(stream_config), mode="stream",
    data_columns=("adjClose",), factor_names=("ma_dev_10", "ma_dev_10_ts"),
    njobs=4,
))
history = xr.open_zarr("prices.zarr")["adjClose"].values.astype("float32")
for step, row in enumerate(history):
    live.cal_stream({"adjClose": np.ascontiguousarray(row)}, step, symbols)

latest = live.get_features()
print(dict(latest.sizes))
print(np.round(latest["ma_dev_10"].values[0], 4))
print(np.round(dev["ma_dev_10"].values[-1], 4))
```

```text
{'timestamp': 1, 'symbol': 8}
[-0.0137 -0.0654 -0.049   0.0099 -0.0472  0.0142 -0.0145 -0.0122]
[-0.0137 -0.0654 -0.049   0.0099 -0.0472  0.0142 -0.0145 -0.0122]
```

`get_features()` in streaming mode holds only the most recent bar. The
stream is compiled on the first call, or explicitly with `init_stream()`.
Every column in `data_columns` must be used by one of the requested outputs,
because KunQuant drops unused inputs from the compiled stream and the lookup
of a dropped input fails. Streaming is a KunQuant
feature only; Polars factors have no streaming mode.

## Things to watch

- The docstrings of `FactorKunQuant` and `CrossSectionalZScore` ask for a
  symbol count that is a multiple of KunQuant's SIMD block width, 8. With the
  KunQuant version the project locks (0.1.11), a six-symbol panel computed
  correctly in batch and streaming mode, including `CrossSectionalZScore`.
  Multiples of 8 remain the safe choice, and the examples use them.
- `window` counts calendar days of warm-up, not bars. Twenty trading days
  need about thirty calendar days.
- Each `cal()` compiles the KunQuant graph again. Compute once, `save()`,
  and let later runs `read()`.
- A KunQuant graph's `Input` names must match `data_columns`, which name
  variables in the store. The graph for a US-equity store reads `adjClose`,
  not `close`.
- When a factor feeds a model, the model uses the factor's pinned
  `factor_names`, so a built-in set limited to a few features, such as
  `Alpha158Stock` with three, trains on exactly those three. An unpinned
  factor contributes every name its class can produce.

## See also

- [Models](models.md) for training on these panels.
- [Universes](universes.md) for point-in-time universes.
- The docstrings of `quantlab.base.factor.Factor`, `FactorKunQuant` and
  `FactorPolars` for every method and parameter.
