# Quickstart

This page is a ten-minute tour of the whole quantlab pipeline on synthetic data. You will
write a small price panel to disk, compute a set of factors and a forward-return label, train
an XGBoost model to predict that return, backtest a strategy that buys the stocks with the
highest predictions, and inspect what the backtest wrote. Nothing needs the network, a
credential or a GPU. Read it after [installation](installation.md); the ideas it touches are
explained in more depth in [concepts](../user-guide/concepts.md).

The code below is taken from `examples/quickstart.py`, which you can run in one go:

```bash
uv run python examples/quickstart.py
```

It takes about half a minute on a laptop CPU, most of it spent compiling the factor graph. All
output shown on this page is real output of that script.

## Before you import anything

One environment variable has to be set before PyTorch or XGBoost is imported, so it comes
first in the script:

```python
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
```

`OMP_NUM_THREADS=1` avoids the macOS OpenMP
clash between PyTorch and XGBoost described in [installation](installation.md).

## Step 1: write a price panel

quantlab passes data between its stages as an `xarray.Dataset` whose variables all sit on
the two dimensions `timestamp` and `symbol`. Such a dataset is called a panel: one row per
bar (a bar is one period's open, high, low, close and volume), one column per stock, and one
variable per field. Panels are stored on disk as Zarr, a chunked array format that xarray
reads and writes natively.

Normally a panel is converted from files downloaded from a data vendor. Here we fabricate a
random walk for 16 symbols over 400 business days, with the same variable names the Tiingo
converter produces. `adjClose` and its siblings are prices adjusted for splits and dividends;
`close` and its siblings are the unadjusted prices (identical here, as nothing splits). In the
snippets on this page, `root` is the temporary directory the script works in.

```python
import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.config import DatasetConfig

SYMBOLS = [f"S{i:02d}" for i in range(16)]
N_BARS = 400
ADJUSTED = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")

rng = np.random.default_rng(0)
timestamps = pd.bdate_range("2022-01-03", periods=N_BARS)
shape = (N_BARS, len(SYMBOLS))
close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, shape), axis=0))
# ... open, high, low and volume are built the same way (see the script)

panel = xr.Dataset(
    {name: (("timestamp", "symbol"), values) for name, values in columns.items()},
    coords={"timestamp": timestamps, "symbol": SYMBOLS},
)
panel.to_zarr(root / "data" / "stock.zarr", mode="w")
```

A dataset object is how the rest of the pipeline reaches that store. It is built from a
config dataclass, `DatasetConfig`, that says where the store lives and what it holds:

```python
from quantlab.dataset.stock import StockDataset

dataset_config = DatasetConfig(
    zarr_file_path=str(root / "data" / "stock.zarr"),
    raw_data_dir_path=str(root / "downloads" / "tiingo"),
    market="us_equity",
    frequency="1d",
    vendor="tiingo",
)
prices = StockDataset(dataset_config)
panel = prices.panel("2000-01-01", "2100-01-01")
print("Price panel:", dict(panel.sizes), "variables:", list(panel.data_vars))
```

```text
Price panel: {'timestamp': 400, 'symbol': 16} variables: ['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'close', 'high', 'low', 'open', 'volume']
```

`raw_data_dir_path` points at a directory that does not exist: it is only used when
converting raw downloads, and the Zarr store already exists. `panel(start, end)` returns the
stored bars from `start` to `end`, both inclusive, opened lazily; the range here covers the
whole store. The dataset holds nothing afterwards and its config is not changed, so one
dataset object can answer any number of requests.

## Step 2: define factors and a label

A factor is a number computed for every `(timestamp, symbol)` cell from data available at
that time, for example a 20-day return or a volatility. A label is the quantity a model learns
to predict, here the return over the next five bars. A factor is a subclass of
`quantlab.factor.base.Factor`; a label is a factor shifted forward in time by
`quantlab.label.forward.Forward`. Both produce a panel.

We use two ready-made classes. `Alpha158Stock` computes the Alpha158 feature library (169
columns in this build: candlestick shapes, lagged prices, rolling returns, volatilities and
volume statistics) from adjusted prices. `Return` is the forward open-to-open return: the
value at bar t is `adjOpen[t + 6] / adjOpen[t + 1] - 1` for `n_forward_periods=5`, which is
what a position entered at the next bar's open and held five bars earns. It is a `Forward`
label with `span=5` (the bars it accumulates over) and `delay=1` (a signal at bar t fills at
bar t + 1), so its value at t reads `lookahead_bars()` = 6 bars ahead. Both are computed by
KunQuant, a library that compiles a declarative graph of operators to native code.

```python
import dataclasses

from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.label.predefined.fret import Return

factor = Alpha158Stock(
    FactorConfig(
        warmup_bars=60,
        dataset=StockDataset(dataclasses.replace(dataset_config)),
        mode="batch",
        data_columns=ADJUSTED,
        file_path=str(root / "factors" / "alpha158.zarr"),
        njobs=4,
    )
)
label = Return(
    FactorConfig(
        warmup_bars=0,
        dataset=StockDataset(dataclasses.replace(dataset_config)),
        mode="batch",
        data_columns=("adjOpen",),
        kwargs={"n_forward_periods": 5},
        file_path=str(root / "labels" / "ret_5.zarr"),
        njobs=4,
    )
)
```

`warmup_bars` is a warm-up period in bars. A rolling feature such as a 60-bar moving
average has no value until 60 bars of history exist, so asked for a date range with
`compute(start, end)`, the factor reads `warmup_bars` bars before `start`, counted on the
dataset's own calendar, and trims them off afterwards. The config holds no dates: the range
is always an argument. `mode="batch"` compiles the graph for a whole date range at once;
`"stream"` would compile it for bar-by-bar updates. `file_path` is where `build(start, end)`
would write the factor values; this example computes them in memory and never builds a store.

## Step 3: train a model

A model takes a list of factors and a list of labels. `XGBoostRegressor` is a
gradient-boosted tree model configured with `ModelConfig`. The dates split the model's data in
time: it trains on bars 0 to 249, holding the last 20 % of that span (`val_size=0.2`) out for
early stopping, and is evaluated on bars 250 to 299. Keeping the test period strictly after
the training period is what makes the evaluation honest; a random split would let the model
see the future. At every split boundary the model also drops the last six bars (the label's
lookahead) of the earlier segment, so no label it fits on reads a bar of the later segment:
the last label it trains on reads bar 249.

```python
from quantlab.base.config import ModelConfig
from quantlab.model.predefined.xgb import XGBoostRegressor

model = XGBoostRegressor(
    ModelConfig(
        factors=[factor],
        labels=[label],
        model_save_dir=str(root / "models"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        hyperparameters={
            "num_boost_round": 50, "max_depth": 3, "eta": 0.1,
            "early_stopping": True, "early_stopping_patience": 10,
        },
        val_size=0.2,
        start_date="2022-01-03",  # bar 0
        end_date="2023-02-24",    # bar 299
        train_start="2022-01-03",
        train_end="2022-12-16",   # bar 249
        test_start="2022-12-19",  # bar 250
        test_end="2023-02-24",
    )
)
features = model.get_factor_names()
print(f"{len(features)} features, e.g. {features[:4]}; labels: {model.get_label_names()}")
```

```text
169 features, e.g. ['KMID', 'KLEN', 'KMID2', 'KUP']; labels: ['ret_5']
```

`factor_data_strategy="cal"` tells the model to compute the factors over its
`start_date`..`end_date` now; `"read"` would read that range from the Zarr stores a previous
`build()` wrote. `collect()` gathers every factor and
label into one panel, and `train()` fits the model and returns the path of the checkpoint it
wrote:

```python
model.collect()
collected = model.data_backend.get_xarray_dataset()
print("Collected panel:", dict(collected.sizes), f"{len(collected.data_vars)} variables")
checkpoint = model.train()
print("Checkpoint:", checkpoint.relative_to(root))
```

```text
Collected panel: {'timestamp': 300, 'symbol': 16} 170 variables
Checkpoint: models/XGBoostRegressor_trial_20260927_212057_563434/XGBoostRegressor_total.joblib
```

Each call to `train()` creates a new timestamped trial directory, so earlier checkpoints are
never overwritten. Next to the `.joblib` checkpoint sit a `config.json` recording the model,
its factors and labels, and a `run.json` with the training windows, the metrics and the
hyperparameters XGBoost actually used, which `quantlab.runs.trained_run.TrainedRun` reads back. The training and test
metrics also go to the config's `tracker`, which by default sends them nowhere
(see [experiment tracking](installation.md#experiment-tracking)). Walk-forward cross-validation
through `train_cv()` is covered in [models](../user-guide/models.md).

## Step 4: backtest the model

A backtest replays the model over a period and simulates trading its predictions.
`USEquityCrossectionSelectStockVectorBt` implements a cross-sectional strategy: every
`rebalance_periods` bars its `constructor`, here a `TopNConstructor`, ranks all symbols by
their predicted return and holds the `top_n` best in equal weights. Orders decided at the close of bar t are filled at the open of bar t+1,
so the strategy never trades on a price it could not have seen. The simulation itself is done
by vectorbt.

```python
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor

backtester = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        price_dataset=StockDataset(dataclasses.replace(dataset_config)),
        model=make_model(root, dataset_config, dates),  # same config as in step 3
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date="2023-01-02",  # bar 260
        end_date="2023-07-14",    # bar 399
        output_dir=str(root / "backtests"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=4)),
    )
)
result = backtester.run()
```

With `model_mode="load"` the backtester builds the model from its config and restores the
trained booster from `checkpoint`; `model_mode="train"` would train it first. It then
computes the factors over the backtest window, each reading its own `warmup_bars` earlier
bars to warm up,
predicts, turns the predictions into target weights and simulates them. Fees and slippage
default to 5 basis points each, and the portfolio starts with 1,000,000 in cash.

The backtester counts the bars from `train_start` up to the bar the last fitted label reads
as in-sample and reports them separately (results on data the model was fitted to are
optimistic by construction). Here that is bar 249, so this window, from bar 260, has none. The
backtest also checks that the label's `delay` equals the engine's fill delay, one bar for
vectorbt, so the model learns the return the backtest trades.

## Step 5: look at the results

`run()` returns a `BacktestResult` holding the predictions, the target weights, the
simulation and the metrics, and it writes all of them to a new run directory under
`output_dir`, which `BacktestRun` reads back:

```python
from quantlab.runs.backtest_run import BacktestRun

run = BacktestRun.open(result.run_dir)
print("Run:", run.kind, run.window, run.market)
print("Predictions:", list(result.predictions.data_vars), dict(result.predictions.sizes))
first_row = result.weights["weight"].isel(timestamp=0)
held = first_row.where(first_row > 0, drop=True)
print("First rebalance:", dict(zip(held.symbol.values.tolist(), held.values.tolist())))
```

```text
Run: run ('2023-01-02', '2023-07-14') Market(fill_price_column='adjOpen', valuation_price_column='adjClose')
Predictions: ['ret_5'] {'timestamp': 140, 'symbol': 16}
First rebalance: {'S03': 0.25, 'S07': 0.25, 'S08': 0.25, 'S09': 0.25}
```

The predictions are themselves a panel, one variable per label. The weights panel holds the
target fraction of the portfolio for every symbol on rebalance bars (here the four chosen
symbols get 25 % each and the rest 0) and NaN on the bars in between, meaning "no change".
The run directory also holds `report.html`, an interactive equity-curve report you can open in
a browser.

The metrics come in groups: `whole` for the whole window, `in_sample` and `out_of_sample` for
the parts that do and do not overlap the model's training data, plus the date ranges of each
part and some notes on how the numbers are computed:

```python
whole = result.metrics["whole"]
for key in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"):
    print(f"  {key:<18} {whole[key]:.3f}")
print("Metric groups:", sorted(result.metrics))
print("In-sample range:", result.metrics["in_sample_range"])
print("Out-of-sample ranges:", result.metrics["out_of_sample_ranges"])
```

```text
  Total Return [%]   10.984
  Sharpe Ratio       1.410
  Max Drawdown [%]   8.645
Metric groups: ['execution', 'in_sample', 'in_sample_range', 'notes', 'out_of_sample', 'out_of_sample_ranges', 'portfolio_construction', 'training_window', 'whole']
In-sample range: None
Out-of-sample ranges: [('2023-01-02', '2023-07-14')]
```

Do not read anything into these numbers: the prices are a random walk, so any profit is luck.
The point is the shape of the output.

## Step 6: reproduce the run from its directory

Every object in the run was built from a config dataclass, and the run directory keeps them
all as its recipe, nested: the backtester's parameters, the price dataset, the model with its
factors and labels, and the checkpoint. Each entry names the class that built it by its dotted
import path. `rebuild_backtester()` reconstructs the backtester from the recipe, so the run can
be repeated without the script that created it:

```python
rebuilt = run.rebuild_backtester()
again = rebuilt.run()
same = np.allclose(again.simulation.value.values, result.simulation.value.values)
print("Rebuilt from its run directory, same equity curve:", same)
```

```text
Rebuilt from its run directory, same equity curve: True
```

The re-run also compares the data it reads against the fingerprints recorded in the first
run (a hash of every request for price and factor inputs) and warns if the underlying data has
changed; the rebuild warns as well when the code of a component or a library version changed.

## Where to go next

- [Concepts](../user-guide/concepts.md) explains the stages, panels, configs and run
  directories used above.
- [Data sources](../user-guide/data-sources.md) and [datasets](../user-guide/datasets.md)
  replace the synthetic panel with real downloads.
- [Factors](../user-guide/factors.md) shows how to write your own factor with KunQuant or
  Polars.
- [Models](../user-guide/models.md) covers the other model heads and walk-forward
  cross-validation.
- [Backtesting](../user-guide/backtesting.md) covers long/short books, cross-validated
  backtests and the report.
