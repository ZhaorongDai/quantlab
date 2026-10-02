# Backtesting

English | [简体中文](zh-CN/backtest.md)

A backtest takes a trained return model and a price dataset and shows how the model's predictions would have traded. The model predicts a score for every symbol on every bar, a selection rule turns the scores into target weights, and a simulation engine trades those weights and records an equity curve. Each run writes a run directory with the weights, the equity curve, metrics, an HTML report and the configuration needed to rebuild it.

The main classes are `BaseBacktester` (`quantlab/base/backtest.py`), the vectorbt engine `VectorBtBacktester` (`quantlab/backtest/engine_vectorbt.py`), the rebalance schedule (`quantlab/backtest/selection.py`), the portfolio construction rule that the config's `constructor` holds (a `PortfolioConstructor` from `quantlab/base/portfolio.py`: `TopNConstructor` here, or the mean-variance optimiser of [Portfolio construction](portfolio.md)) and the US-equity backtester `USEquityCrossectionSelectStockVectorBt` (`quantlab/backtest/predefined/us_equity.py`).

## Prerequisites

Run the examples from the repository root with `uv run python`. On macOS, set `OMP_NUM_THREADS=1` before torch or xgboost is imported in the same process. Nothing is tracked unless a config names a tracker (see Track a backtest).

A backtest needs a price dataset whose store has the columns `adjOpen` and `adjClose`, and a model with a checkpoint written by `train()` or `train_cv()`. The sessions below use a synthetic setup: six symbols, one factor, one label and a model head with nothing to fit whose score is the past one-bar return. The label is the factor `open_ret_1` wrapped in `Forward` with `span=1` and the default `delay=1`: its value at bar t is the open-to-open return from t+1 to t+2, so its lookahead is 2 bars. The last symbol, `FFF`, stops trading at bar 36. Save this as `demo_parts.py`.

<details>
<summary>demo_parts.py</summary>

```python
"""Synthetic prices, a factor, a label and a model with nothing to fit."""
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.base.config import DatasetConfig, ForwardConfig, ModelConfig, PolarsFactorConfig
from quantlab.factor.polars import FactorPolars
from quantlab.model.library_model import LibraryModel
from quantlab.dataset.stock import StockDataset
from quantlab.label.forward import Forward

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def write_price_store(root, n_bars=60, delist=None):
    """Daily adjusted prices in Zarr; delist={"FFF": 36} ends FFF at bar 36."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    shape = (n_bars, len(SYMBOLS))
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, shape), axis=0))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(rng.normal(0, 0.01, shape))
    for symbol, bar in (delist or {}).items():
        close[bar:, SYMBOLS.index(symbol)] = np.nan
        open_[bar:, SYMBOLS.index(symbol)] = np.nan
    dims = ["timestamp", "symbol"]
    xr.Dataset(
        {"adjOpen": (dims, open_), "adjClose": (dims, close)},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=n_bars), "symbol": SYMBOLS},
    ).to_zarr(root / "prices.zarr", mode="w")
    return DatasetConfig(
        raw_data_dir_path=str(root / "raw"), zarr_file_path=str(root / "prices.zarr"),
        market="us_equity", frequency="1d",
    )


def prices_of(cfg):
    return StockDataset(dataclasses.replace(cfg))


class PastReturn(FactorPolars):
    """Feature past_ret_1: the one-bar close-to-close return."""

    def _get_factor_lazyframe(self, lf):
        c = pl.col("adjClose")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((c / c.shift(1).over("symbol") - 1).alias("past_ret_1"))
                .select(["timestamp", "symbol", "past_ret_1"]))


class OpenReturn(FactorPolars):
    """open_ret_1: the one-bar open-to-open return, using bars up to t only."""

    def _get_factor_lazyframe(self, lf):
        o = pl.col("adjOpen")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((o / o.shift(1).over("symbol") - 1).alias("open_ret_1"))
                .select(["timestamp", "symbol", "open_ret_1"]))


class MomentumHead(LibraryModel):
    """Predicts its first feature, so the score is the past return."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def make_label(cfg, delay=1):
    """Label open_ret_1 at t: the open-to-open return from t+delay to t+delay+1."""
    factor = OpenReturn(PolarsFactorConfig(warmup_bars=1, dataset=prices_of(cfg)))
    return Forward(ForwardConfig(factor=factor, span=1, delay=delay))


def make_model(root, cfg, days, train_end=39, delay=1):
    day = lambda i: str(days[i].date())
    factor = PastReturn(PolarsFactorConfig(warmup_bars=5, dataset=prices_of(cfg)))
    return MomentumHead(ModelConfig(
        factors=[factor], labels=[make_label(cfg, delay)], model_save_dir=str(root / "models"),
        factor_data_strategy="cal", label_data_strategy="cal", val_size=0.0,
        start_date=day(0), end_date=day(len(days) - 1),
        train_start=day(0), train_end=day(train_end),
        test_start=day(train_end + 1), test_end=day(len(days) - 1),
    ))


def train_checkpoint(model):
    model.collect()
    return model.train()


def train_cv_project(model, train_periods):
    model.collect()
    model.train_cv(train_periods=train_periods)
    return next(Path(model.config.model_save_dir).rglob("cv_folds.json")).parent
```

</details>

## The basics

### What a run does

`BaseBacktester.run()` backtests one model over the window `start_date` to `end_date`. It checks every label's delay against the engine's fill delay, loads the checkpoint (or trains the model first), computes the features on the window, predicts a score per symbol and bar, asks the concrete class for target weights, simulates them, computes metrics and writes the run directory. `run_cv()` does the same for every fold of a `train_cv` run and stitches the folds into one curve. `run_weights(weights)` skips the model and backtests target weights you already have (see [Backtest precomputed weights](#backtest-precomputed-weights)).

The first session trains a checkpoint and backtests a rule that holds the two highest-scoring symbols and rebalances every five bars. Log lines go to stderr and are not shown.

```python
>>> import dataclasses, json, tempfile
>>> from pathlib import Path
>>> import pandas as pd
>>> import xarray as xr
>>> from demo_parts import *
>>> from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
>>> from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
>>> from quantlab.portfolio.predefined.top_n import TopNConstructor
>>> root = Path(tempfile.mkdtemp())
>>> cfg = write_price_store(root, delist={"FFF": 36})
>>> days = pd.bdate_range("2024-01-01", periods=60)
>>> checkpoint = train_checkpoint(make_model(root / "train", cfg, days))
>>> backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
...     price_dataset=prices_of(cfg),
...     model=make_model(root / "backtest", cfg, days),
...     model_mode="load",
...     checkpoint=str(checkpoint),
...     start_date="2024-02-12",
...     end_date="2024-03-22",
...     output_dir=str(root / "runs"),
...     rebalance_periods=5,
...     constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
... ))
>>> result = backtester.run()
>>> sorted(p.name for p in result.run_dir.iterdir())
['config.json', 'equity.zarr', 'fingerprint.json', 'metrics.json', 'report.html', 'settlements.json', 'weights.zarr']
```

The run directory sits under `output_dir`, and the result holds the predictions, weights, simulation and metrics.

### The target-weight contract

The weights are a `weight` variable on `(timestamp, symbol)`. A finite weight is the fraction of portfolio value the symbol should hold after the bar fills; a NaN keeps the symbol's holding untraded. An all-NaN row is a bar without a rebalance, and a row may mix the two, for example to leave one holding alone. The targets of a row have a gross exposure (the sum of absolute weights) of at most 1. The shipped portfolio construction rules give every symbol a finite weight on a rebalance bar, `0.0` for one that is not selected. A rule decides from what is known at the bar (ADR 0014): a symbol is *tradable* where the price dataset's `tradable_bars` says so, by default when it has a fill price at that bar, and a held symbol that is not tradable is a *locked position*, which keeps its current weight. The backtester refuses a rule that moves a locked position or gives weight to a symbol that is neither tradable nor held.

```python
>>> result.weights["weight"].to_pandas().iloc[:7].round(2)
symbol      AAA  BBB  CCC  DDD  EEE  FFF
timestamp                               
2024-02-12  0.0  0.0  0.5  0.0  0.0  0.5
2024-02-13  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-14  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-15  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-16  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-19  0.0  0.5  0.0  0.5  0.0  0.0
2024-02-20  NaN  NaN  NaN  NaN  NaN  NaN
```

The first bar of the window rebalances, and so does every `rebalance_periods`-th bar after it. The last bar never rebalances, because a signal formed there has no following bar to fill on. With `direction="long_short"` the top `top_n` symbols get `+0.5/top_n` each and the bottom `top_n` get `-0.5/top_n` each.

### A signal at bar t fills at bar t+1

A weight row formed at bar `t` executes at the fill price of bar `t + 1`. For US equities the fill price is the adjusted open (`adjOpen`) and the portfolio is valued at the adjusted close (`adjClose`). Below, the first orders are on 2024-02-13, the bar after the first rebalance. The buy price is that day's open plus the default slippage of 0.05 percent.

```python
>>> result.simulation.orders.to_dataframe().head(3)
       timestamp symbol         size      price        fees  side
order                                                            
0     2024-02-13    CCC  9465.316230  52.850849  250.125000   Buy
1     2024-02-13    FFF  8221.655639  60.723809  249.625125   Buy
2     2024-02-20    FFF  8221.655639  61.958954    0.000000  Sell
>>> adj_open = xr.open_zarr(root / "prices.zarr")["adjOpen"]
>>> float(adj_open.sel(timestamp="2024-02-13", symbol="CCC"))
52.82443690819098
```

Fees and slippage (`fees` and `slippage`, both 0.0005 by default) are proportional to each trade. A target percentage is measured against the portfolio value at the fill price of the bar it executes on.

### Label delay and fill delay

The engine declares `fill_delay_bars`, the number of bars between the bar a weight forms on and the bar it fills on; `VectorBtBacktester` sets it to 1. A label's `delay` is the number of bars between the bar a signal forms on and the first bar the label counts. `run()` and `run_cv()` compare every label's `delay` with `fill_delay_bars` before training, loading or simulating, and raise `ValueError` naming the label, its delay and the fill delay when they differ.

```python
>>> USEquityCrossectionSelectStockVectorBt.fill_delay_bars
1
>>> label = make_label(cfg)
>>> label.config.delay, label.span_bars(), label.lookahead_bars()
(1, 1, 2)
>>> same_bar = dataclasses.replace(backtester.config, model=make_model(root / "same_bar", cfg, days, delay=0))
>>> USEquityCrossectionSelectStockVectorBt(same_bar).run()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: labels[0] Forward ('open_ret_1',) has delay=0, but the engine fills a weight fill_delay_bars=1 bar(s) after the bar it forms on; the model would learn a return the backtest never trades
```

### Rejected orders and delisted holdings

Both price columns are forward-filled before the simulation, and each fill bar is then executed the way a market would (ADR 0014). An order whose raw fill price is missing on its fill bar, because the symbol is halted, is a *rejected order*: the holding is kept, the order expires, and the next rebalance decides again. Rejected orders that would have traded are listed in `result.simulation.rejected_orders` and in the `execution` block of the metrics, with `rejected_order_count` and `max_target_deviation`, the largest gap between a target weight and the weight held right after its fill bar (fees and cash included).

A symbol whose prices stop inside the window is delisted on its last priced bar (`MarketDataset.delisting_bars`; a dataset that knows its halts may override it). On the next bar a holding in it is settled into cash at its last valuation price, with no fee or slippage, and the *delisting settlement* is recorded. On a CRSP store the last adjusted close already carries the delisting return. `FFF` has no price from 2024-02-20, is in the first portfolio, and is settled on 2024-02-20 at its 2024-02-19 close; the order at price 61.96 above is that settlement.

```python
>>> result.simulation.settlements
[{'symbol': 'FFF', 'axis_symbol': 'FFF', 'delisting_timestamp': Timestamp('2024-02-19 00:00:00'), 'settlement_timestamp': Timestamp('2024-02-20 00:00:00'), 'price': 61.95895375478968}]
>>> result.metrics["execution"]["rejected_order_count"]
0
```

A symbol with no prices at the start of the window that was never held is treated as not yet listed and trades once its prices begin.

### Warm-up

The model's factors need history before `start_date`. The backtester asks each factor for the window by date range, `compute(start_date, end_date)` under the `"cal"` strategy or `read(start_date, end_date)` under `"read"`, and a computed factor reads its own `warmup_bars` bars before `start_date`, counted on its dataset's calendar rather than in calendar days. A backtest and a standalone `compute` over the same window therefore give the same factor values. If the dataset holds fewer bars, the factor starts from the first one and a `UserWarning` gives the shortfall in bars. No dataset, factor or label config is changed, so the price dataset may be the same object as a factor's dataset. Predictions cover exactly the window bars, and a price symbol without a prediction gets NaN scores and is never selected.

### In-sample and out-of-sample

Let L be the largest `lookahead_bars()` among the model's labels. The model fits on the bars `train_start` to `train_end` less the purge, which drops the last L bars before the test segment. The label on the last fitted bar reads L bars further, so the effective training window runs from `train_start` to the last fitted bar plus L bars on the price calendar (`quantlab.utils.split.in_sample_window`). For a test segment that follows the training segment, this window ends on the configured `train_end`. Window bars inside it are in-sample and the rest are out-of-sample. In load mode the training dates come from the `config.json` stored beside the checkpoint. When the window overlaps the training window the run logs a warning and continues.

```python
>>> m = result.metrics
>>> m["training_window"], m["in_sample_range"], m["out_of_sample_ranges"]
(('2024-01-01', '2024-02-23'), ('2024-02-12', '2024-02-23'), [('2024-02-26', '2024-03-22')])
```

All parts come from one continuous simulation, so capital and positions carry across the boundary. `whole` holds the engine statistics of the full window. `in_sample` and `out_of_sample` hold return-based statistics over their own bars, plus fill counts and turnover.

```python
>>> for part in ("whole", "in_sample", "out_of_sample"):
...     print(part, round(m[part]["Total Return [%]"], 2), round(m[part]["Sharpe Ratio"], 2), m[part]["Total Orders"])
whole -5.87 -2.33 19
in_sample -0.27 -0.11 6
out_of_sample -5.61 -4.27 13
>>> list(m["whole"])[:6]
['Start', 'End', 'Period', 'Start Value', 'End Value', 'Total Return [%]']
>>> round(m["whole"]["Turnover per Rebalance [%]"])
134
```

Turnover is the one-sided traded value of a bar divided by the portfolio value before the fills, in percent like every other `[%]` row, so a full buy-in from cash is about 100 and replacing the whole book about 200. The scores here are random-walk returns, so the negative results carry no meaning.

### The run directory

Each run writes a new directory `{ClassName}_{timestamp}` under `output_dir`. Files go to a hidden staging directory that is renamed when every file has been written, so `output_dir` only holds complete runs. With `output_dir=None` nothing is written (see [Keep a run in memory](#keep-a-run-in-memory)).

| File | Content |
| --- | --- |
| `config.json` | The configuration, with the price dataset and the model nested, and a data fingerprint. Its `market` block names the market's `fill_price_column` and `valuation_price_column`, so a tool reading the run learns them without importing the backtester class; a rebuild drops it and takes them from the class again. |
| `weights.zarr` | The target weights on `(timestamp, symbol)`. |
| `equity.zarr` | The portfolio `value` and per-bar `returns` on `timestamp`. |
| `metrics.json` | The same mapping as `result.metrics`; NaN and infinity are written as null. Every run records `execution` (rejected orders and the largest target deviation). A `run()`, a `run_cv()` fold and the stitched `run_cv()` pass also record `portfolio_construction`: `failed_bar_count` and `failed_bars`, the rebalance bars the constructor could not decide (an optimisation that failed or was infeasible), which the backtest held instead, and any event the constructor reported, such as the mean-variance optimiser's `closed_without_risk` (held symbols closed because the risk model had no estimate for them) or the top-n rule's `tie_at_cutoff` (tied symbols a book's cut left out, so the picks were decided by symbol order), with its `count` (symbols over all its bars) and its `bars`, one record per bar that names the symbols or, for `tie_at_cutoff`, counts them. |
| `settlements.json` | The delisting settlements. |
| `fingerprint.json` | A digest of the price and factor data the run read. |
| `report.html` | Headline numbers, grouped metric tables, and chart tabs for performance, excess return, rolling one-year statistics and the portfolio's structure (see [The report page](#the-report-page)). |
| `inputs/` | Only when the price or benchmark dataset is a `FrameDataset` held in memory: its panel as `price_dataset.zarr` or `benchmark_dataset.zarr`, which `config.json` names relative to the run directory (see [Rebuild a run of given weights](#rebuild-a-run-of-given-weights)). |

## Common tasks

### Train the model inside the backtest

With `model_mode="train"` the model is trained on its own configured dates first, and `checkpoint` is not needed. The window never changes the training dates. The checkpoint the run wrote is recorded in `metrics["trained_checkpoint"]`. This session also switches to a long-short book with one name per side.

```python
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=make_model(root / "train_mode", cfg, days),
...     model_mode="train", checkpoint=None,
...     constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=1)),
... )).run()
>>> Path(trained.metrics["trained_checkpoint"]).name
'MomentumHead_total.joblib'
>>> trained.weights["weight"].to_pandas().iloc[0]
symbol
AAA    0.0
BBB   -0.5
CCC    0.5
DDD    0.0
EEE    0.0
FFF    0.0
Name: 2024-02-12 00:00:00, dtype: float64
```

### Replay a cross-validation run

`train_cv` writes one checkpoint per walk-forward fold and a `cv_folds.json` manifest into its project directory. `run_cv()` reads the manifest, backtests each fold's test segment with the fold's own checkpoint, then turns the concatenated fold predictions into weights in one pass, so the holdings carry across fold boundaries as in one account, and simulates them once. `cv_project_dir` points at the project directory and `model_mode` must be `"load"`. Only folds whose test segment lies inside the window are used, and those segments must follow each other bar for bar.

```python
>>> cfg2 = write_price_store(root / "cv", n_bars=80)
>>> days2 = pd.bdate_range("2024-01-01", periods=80)
>>> project_dir = train_cv_project(make_model(root / "cv_train", cfg2, days2, train_end=29), 30)
>>> manifest = json.loads((project_dir / "cv_folds.json").read_text())
>>> len(manifest["folds"]), manifest["folds"][0]["test_start"][:10], manifest["folds"][-1]["test_end"][:10]
(8, '2024-02-12', '2024-04-17')
>>> cv_config = dataclasses.replace(
...     backtester.config,
...     price_dataset=prices_of(cfg2),
...     model=make_model(root / "cv_backtest", cfg2, days2, train_end=29),
...     cv_project_dir=str(project_dir),
...     start_date=str(days2[30].date()),
...     end_date=str(days2[77].date()),
...     rebalance_periods=2,
... )
>>> cv = USEquityCrossectionSelectStockVectorBt(cv_config).run_cv()
>>> len(cv.folds), dict(cv.weights.sizes), sorted(cv.metrics)
(8, {'timestamp': 48, 'symbol': 6}, ['folds', 'notes', 'stitched'])
>>> sorted(p.name for p in (cv.run_dir / "folds").iterdir())[:2]
['fold_0', 'fold_1']
```

The top-level files of the run directory describe the stitched curve, and `folds/fold_{i}/` holds each fold's own weights and equity. The stitched curve is one simulation, so capital carries across fold boundaries. Each fold also has an independent simulation that starts from `init_cash`, and the per-fold metrics come from those. `train_cv` purges the last L bars of every fold's training segment and records the purged `train_end` in the manifest. A fold's in-sample window ends L bars after that `train_end`, on the bar before the fold's test segment, so no stitched bar is in-sample. `quantlab.utils.split.split_ranges` cuts the stitched bars into `in_sample_ranges` and `out_of_sample_ranges`.

```python
>>> stitched = cv.metrics["stitched"]
>>> stitched["in_sample_ranges"][:2], stitched["out_of_sample_ranges"][:2]
([], [('2024-02-12', '2024-04-17')])
>>> round(stitched["whole"]["Total Return [%]"], 2), round(cv.folds[0]["metrics"]["whole"]["Total Return [%]"], 2)
(-3.11, -1.62)
```

### Backtest precomputed weights

`run_weights(weights)` backtests a target-weight panel that already exists, for example weights built by another tool or saved by an earlier run, without a model. The config needs no `model` and no `model_mode`; the two are set together or left `None` together, and a config with only one of them is refused when the backtester is built. The backtester reads the fill and valuation prices of the window `start_date` to `end_date`, checks the weights against [the target-weight contract](#the-target-weight-contract) on exactly those bars and symbols, and simulates them with the same t+1 fill. The panel is a dataset with a `weight` variable or a data array, in either axis order; it is aligned to the price axes. The benchmark works as in `run()`. There is no training window, so the metrics hold whole-window blocks only (`whole`, and `benchmark` and `relative` with a `whole` slice each when a benchmark is set), with no in-sample or out-of-sample split, and the report leaves out the split lines. Fed the weights of the first session, whose config has no benchmark, it reproduces that run, so the metrics are `whole`, `execution` and `notes` only.

```python
>>> weights_config = dataclasses.replace(backtester.config, model=None, model_mode=None, checkpoint=None)
>>> weights_backtester = USEquityCrossectionSelectStockVectorBt(weights_config)
>>> replay = weights_backtester.run_weights(result.weights)
>>> sorted(replay.metrics), replay.predictions is None
(['execution', 'notes', 'whole'], True)
>>> replay.metrics["whole"] == result.metrics["whole"]
True
>>> sorted(p.name for p in replay.run_dir.iterdir())
['config.json', 'equity.zarr', 'fingerprint.json', 'metrics.json', 'report.html', 'settlements.json', 'weights.zarr']
```

`run()` and `run_cv()` still need a model:

```python
>>> weights_backtester.run()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: run() requires config.model, but it is None; set config.model and config.model_mode, or backtest precomputed weights with run_weights()
```

Weights that break the contract are refused, naming the bar:

```python
>>> broken = result.weights.copy(deep=True)
>>> broken["weight"][5, 0] = 0.9
>>> weights_backtester.run_weights(broken)
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: weight row at 2024-02-19 has gross exposure 1.9 > 1
>>> weights_backtester.run_weights(result.weights.isel(timestamp=slice(1, None)))
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: the weight bars must be exactly the price bars of the backtest window: 1 missing ['2024-02-12'], 0 extra [], 0 duplicated
```

`WeightsVectorBt` (`quantlab/backtest/predefined/weights.py`) backtests given weights on any market: its `WeightsBacktestConfig` names the fill and valuation columns and the annualization (`trading_days_per_year`, `session_minutes_per_day`) instead of a class constant, and it takes no model. It is the backtester `quantlab.api.backtest` runs. The price dataset may be a `FrameDataset` held in memory; with no store there is no ticker sidecar, so symbols are shown as they are, without a warning.

```python
>>> import numpy as np
>>> import pandas as pd
>>> import xarray as xr
>>> from quantlab.backtest.predefined.weights import WeightsVectorBt
>>> from quantlab.base.config import WeightsBacktestConfig
>>> from quantlab.dataset.memory import FrameDataset
>>> bars = pd.bdate_range("2024-01-01", periods=5)
>>> prices = FrameDataset(pd.DataFrame({
...     "timestamp": np.repeat(bars, 2),
...     "symbol": ["AAA", "BBB"] * 5,
...     "open": [10.0, 20.0, 11.0, 20.0, 12.0, 21.0, 12.0, 22.0, 13.0, 22.0],
...     "close": [10.5, 20.0, 11.5, 20.5, 12.0, 21.5, 12.5, 22.0, 13.0, 22.5],
... }))
>>> backtester = WeightsVectorBt(WeightsBacktestConfig(
...     price_dataset=prices, start_date="2024-01-01", end_date="2024-01-05",
...     output_dir=None, rebalance_periods=1, fees=0.0, slippage=0.0,
...     fill_price_column="open", valuation_price_column="close",
...     trading_days_per_year=252, session_minutes_per_day=390,
... ))
>>> weights = xr.DataArray(
...     [[1.0, 0.0]] + [[np.nan, np.nan]] * 4, dims=("timestamp", "symbol"),
...     coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
... )
>>> result = backtester.run_weights(weights)
>>> result.simulation.value.values.round(2).tolist()
[1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]
```

### Keep a run in memory

With `output_dir=None` a run writes nothing: no run directory, no report. `result.run_dir` is `None` and everything else is in the result. This holds for `run()`, `run_cv()` and `run_weights()`. `output_dir` has no default, so pass `None` explicitly. `output_dir=None` covers the backtest's own run directory only: with `model_mode="train"` the model still writes its checkpoint where its own config points.

```python
>>> in_memory = USEquityCrossectionSelectStockVectorBt(
...     dataclasses.replace(weights_config, output_dir=None)
... ).run_weights(result.weights)
>>> in_memory.run_dir is None, round(in_memory.metrics["whole"]["Total Return [%]"], 2)
(True, -5.85)
```

`report_figure(result)` returns the Performance chart `report.html` would embed (equity, drawdown and monthly returns, with the benchmark beside the portfolio when a benchmark ran) as a plotly figure, so a run kept in memory can be looked at too. It takes the result of `run()` or `run_weights()`.

```python
>>> figure = weights_backtester.report_figure(in_memory)
>>> type(figure).__name__
'Figure'
```

### Compare against a benchmark

Set `benchmark_dataset` to a market dataset that holds exactly one symbol, for example the QQQ store written by `scripts/wrds/etf.py --etf qqq` (`CrspDatasetConfig.qqq_benchmark`). It is a `(timestamp, symbol)` panel like the price dataset, in a store of its own, with the same `adjOpen` / `adjClose` columns. Pass the dataset object itself:

```python
>>> from quantlab.dataset.crsp import CrspStockDataset
>>> qqq = CrspStockDataset(CrspDatasetConfig.qqq_benchmark(
...     zarr_file_path="data/us_equity/1d/wrds_crsp_qqq_1d.zarr",
...     raw_data_dir_path="data/downloads/us_equity/1d/crsp/wrds",
...     reference_dir="data/reference/crsp",
... ))
>>> config = CrossSectionBacktestConfig(..., benchmark_dataset=qqq)
>>> result = USEquityCrossectionSelectStockVectorBt(config).run()
>>> sorted(result.metrics["relative"]["whole"])[:4]
['Annualized Excess Return [%]', 'Bars', 'Benchmark Total Return [%]', 'Beta']
```

The benchmark is read over the window and put on the strategy's own bars; a bar it lacks carries its previous price forward (one warning), and a benchmark that starts after the window, or a panel with more than one symbol, is refused. It is bought and held with the strategy's conventions: all-in at the second bar's open, from the same `init_cash`, with the same fees and slippage, so the two value curves compare bar for bar. `result.benchmark` is its `SimulationResult`.

The run then carries two more metric blocks, each with `whole`, `in_sample` and `out_of_sample` slices:

- `benchmark`: the benchmark's `symbol` and its own return statistics (total and annualized return, volatility, Sharpe, max drawdown, ...);
- `relative`: the portfolio against the benchmark, named like vectorbt's statistics, with every `[%]` row in percent. The *relative NAV* is portfolio value divided by benchmark value. `Excess Return [%]` is that NAV minus 1 at the end (the alpha in the everyday sense), `Annualized Excess Return [%]` the same over one year, `Excess Max Drawdown [%]` the deepest fall of the relative NAV from its running peak (the *excess drawdown*, negative or 0), plus `Strategy Total Return [%]`, `Benchmark Total Return [%]`, `Total Return Difference [%]`, `Tracking Error [%]`, `Information Ratio`, `Beta`, `Correlation`, `CAPM Alpha [%]` (the annualized regression intercept), `Win Rate vs Benchmark [%]` (the share of bars with a higher return than the benchmark's), `Rebalance Win Rate vs Benchmark [%]` (the share of holding periods, from a bar with fills to the bar before the next, whose compounded return beats the benchmark's) and `Monthly Win Rate vs Benchmark [%]` (the same per calendar month). The strategy's own slices carry `Rebalance Win Rate [%]` and `Monthly Win Rate [%]`, the shares with a gain, with or without a benchmark.

`report.html` draws the benchmark (dashed grey) beside the portfolio on the Performance tab (NAV, drawdown and monthly returns), puts the benchmark in the second column of the "Strategy vs" table, adds the "Relative to" table and the Excess tab, and turns the Rolling tab to excess return, information ratio and beta (see [The report page](#the-report-page)). `equity.zarr` also stores `benchmark_value` and `benchmark_returns`, `fingerprint.json` records the benchmark data under `benchmark_dataset`, and `config.json` rebuilds it. `run_cv()` compares the stitched curve and every fold the same way.

### The report page

`report.html` is one page in three parts:

- **Headline numbers**: total return, excess return, information ratio, win rate, Sharpe ratio, max drawdown, beta and annualised turnover, each with the benchmark's value or a related figure beneath. Without a benchmark: total return, annualised return, win rate, Sharpe ratio, max drawdown, volatility and turnover. The win rate is the share of holding periods (from a bar with fills to the bar before the next) that beat the benchmark, with the share of calendar months beneath; without a benchmark, the share with a gain.
- **Tables**, on the left: "Windows", a timeline of the training and backtest windows (traded out-of-sample bars green, traded bars inside a training window red, training windows light blue), one row for `run()` and, for `run_cv()`, the backtest window above one row per fold, every window's dates in its tooltip; "Setup", the settings no table shows (bar interval, benchmark, deepest-drawdown dates, model mode, rebalancing, portfolio construction, fees); "Strategy vs *benchmark*", grouped into returns, risk and risk-adjusted ratios, with the benchmark and the difference (in percentage points for percents) beside the strategy; "Relative to *benchmark*" (geometric and arithmetic excess, excess drawdown, tracking error, information ratio, beta, correlation, CAPM alpha); "Trading" (turnover, fees, orders, round trips, rejected orders, portfolio-construction failures and events); and, when the run has an in-sample part, "In-sample vs out-of-sample", with the in-sample and out-of-sample values, their difference and the whole window side by side. Hover a metric's name for its definition. A metric the page does not know, from the strategy, the benchmark or the relative block, is listed under "Other".
- **Charts**, on the right, in tabs: *Performance* (NAV with the linear/log toggle, drawdown, monthly returns and the year-by-month heatmap); *Excess*, with a benchmark (the cumulative excess return, switchable between log, `Σ log((1+r)/(1+b))`, whose exponential minus 1 is the geometric excess, and arithmetic, `Σ(r − b)`, read like a cumulative IC; and the excess drawdown under it); *Rolling* (one-year excess return, information ratio and beta, or one-year return, volatility and Sharpe ratio without a benchmark); *Portfolio* (turnover per fill bar, the number of holdings and the gross exposure of the target weights, and the net exposure when anything is short).

When the run has an in-sample part, the headline numbers and the main tables are its out-of-sample slice, the bars the model never saw, and every chart shades the in-sample range. Drawdowns are negative everywhere on the page. The excess drawdown, the fall of the relative NAV from its peak, is drawn only on the Excess tab, apart from the two NAVs' own drawdowns, because the numbers are not comparable.

### Track a backtest

A backtest is tracked through its config's `tracker`, as a model is (see Track experiments in the model guide). The default, `NullTracker()`, sends nothing anywhere. `run()`, `run_cv()` and `run_weights()` each open one run in the project `<ClassName>_backtest`, unless the tracker sets `project`, named after the run directory and carrying the backtest's config. Its summary holds the `whole`, `in_sample` and `out_of_sample` blocks as `whole/<metric>` and so on, plus `benchmark` and `relative` when a benchmark ran; for `run_cv()` these are the stitched metrics. On MLflow a character it refuses in a key becomes `_`, so `whole/Total Return [%]` is logged as `whole/Total Return ___`. `report.html` is attached to the run when a run directory exists; a run kept in memory is tracked without it. The run opens before the backtest, so a backtest that raises is recorded as failed. With `model_mode="train"`, the model's training goes through the model config's own tracker. The tracker is written to `config.json` and rebuilt by `load_backtester_from_config`.

```python
>>> backtester.config.tracker
NullTracker(project=None)
>>> from quantlab.tracking.wandb import WandbTracker
>>> tracked = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, tracker=WandbTracker(project="momentum_backtests", mode="offline")
... )).run()
>>> json.loads((tracked.run_dir / "config.json").read_text())["tracker"]
{'project': 'momentum_backtests', 'entity': None, 'mode': 'offline', 'name': 'quantlab.tracking.wandb.WandbTracker'}
```

With `mode="offline"` the run is written under `wandb/` (or `WANDB_DIR`) as the run `USEquityCrossectionSelectStockVectorBt_<timestamp>` of the project `momentum_backtests`, with `whole/Total Return [%]` and the other metrics in its summary and the report as an HTML panel named `report`.

### Rebuild a run from its config

`config.json` names every class by its dotted import path, so `load_backtester_from_config` builds the same backtester, including its price dataset and model, and `run()` repeats the backtest into a new directory. When the data changed since the original run, the rebuilt run logs a warning per changed dataset and continues.

```python
>>> from quantlab.utils.module import load_backtester_from_config
>>> config = json.loads((result.run_dir / "config.json").read_text())
>>> config["name"], config["constructor"]
('quantlab.backtest.predefined.us_equity.USEquityCrossectionSelectStockVectorBt', {'direction': 'long_only', 'top_n': 2, 'score_label': None, 'name': 'quantlab.portfolio.predefined.top_n.TopNConstructor'})
>>> again = load_backtester_from_config(config).run()
>>> again.metrics["whole"] == result.metrics["whole"]
True
>>> again.run_dir == result.run_dir
False
```

The classes must be importable by dotted path. A class defined in a script is named `__main__.X` and cannot be found from another process, so the dataset, factors, model and backtester belong in modules. A config written by a train-mode run retrains when it is rebuilt; set `model_mode` to `"load"` and `checkpoint` to the recorded `trained_checkpoint` to replay the same model.

### Rebuild a run of given weights

A `run_weights()` run has no model to predict its weights again, so it is replayed from the weights it saved: read the run directory's `weights.zarr` with `XrBackend().read(path).data` and pass the panel to `run_weights()`. When the price or benchmark dataset is a `FrameDataset` (every `quantlab.api.backtest` run, and the `WeightsVectorBt` session above), its panel has no store of its own, so the dataset writes it under `inputs/` (its `persist_with_run`) and `config.json` names that store relative to the run directory. Pass the directory the config was read from as `run_dir`; the run directory can be moved. A dataset read from a project store writes nothing and keeps its path. Continuing the `WeightsVectorBt` session:

```python
>>> import dataclasses, json, shutil, tempfile
>>> from pathlib import Path
>>> from quantlab.backend import XrBackend
>>> from quantlab.utils.module import load_backtester_from_config
>>> kept = WeightsVectorBt(
...     dataclasses.replace(backtester.config, output_dir=tempfile.mkdtemp())
... ).run_weights(weights)
>>> sorted(p.name for p in kept.run_dir.iterdir())
['config.json', 'equity.zarr', 'fingerprint.json', 'inputs', 'metrics.json', 'report.html', 'settlements.json', 'weights.zarr']
>>> config = json.loads((kept.run_dir / "config.json").read_text())
>>> config["price_dataset"]["zarr_file_path"]
'inputs/price_dataset.zarr'
>>> run_dir = Path(shutil.move(kept.run_dir, tempfile.mkdtemp()))
>>> rebuilt = load_backtester_from_config(config, run_dir=run_dir)
>>> replay = rebuilt.run_weights(XrBackend().read(run_dir / "weights.zarr").data)
>>> replay.simulation.value.values.round(2).tolist()
[1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]
>>> json.loads((replay.run_dir / "metrics.json").read_text()) == json.loads(
...     (run_dir / "metrics.json").read_text())
True
>>> json.loads((replay.run_dir / "fingerprint.json").read_text()) == rebuilt.expected_fingerprint
True
```

The rebuilt `FrameDataset` reads the store into memory; the replay writes its own `inputs/` again, so its directory rebuilds on its own too. Its symbols are shown as they are: a `FrameDataset` names no store for a CRSP ticker sidecar (`ticker_store()` is `None`), even when read back from `inputs/`. A relative store path is never resolved against the working directory:

```python
>>> load_backtester_from_config(config)
Traceback (most recent call last):
  ...
ValueError: quantlab.dataset.memory.FrameDataset reads the store 'inputs/price_dataset.zarr', which is relative to the run directory the config was saved in; pass run_dir= (the directory holding config.json) to rebuild it. It is never resolved against the working directory.
```

## Extending

A new selection rule is a subclass of `VectorBtBacktester` with three members: `config_cls`, `MARKET` and `_generate_signals(predictions, prices)`. The method returns a dataset whose `weight` variable satisfies the contract above. `predictions` and `prices` share the same `(timestamp, symbol)` axes. The rule below weights each tradable symbol in proportion to its positive score and stays flat when no score is positive. It reuses `rebalance_mask` and the `US_EQUITY_MARKET` price conventions. Save it as `score_weighted.py`.

```python
"""A new selection rule: long-only weights proportional to positive scores."""

import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import rebalance_mask
from quantlab.backtest.predefined.us_equity import US_EQUITY_MARKET
from quantlab.base.config import BacktestConfig


class ScoreWeightedBacktester(VectorBtBacktester):
    config_cls = BacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions, prices):
        label = list(predictions.data_vars)[0]  # the model's first label
        scores = predictions[label].transpose("timestamp", "symbol")
        # A symbol without a fill price at the bar cannot be traded there (ADR 0014).
        tradable = self.config.price_dataset.tradable_bars(prices, self.MARKET.fill_price_column)
        positive = scores.where(tradable).clip(min=0).fillna(0.0)
        total = positive.sum("symbol")
        weight = (positive / total.where(total > 0)).fillna(0.0)  # flat if no score
        rebalance = xr.DataArray(
            rebalance_mask(prices.sizes["timestamp"], self.config.rebalance_periods),
            dims="timestamp", coords={"timestamp": prices.timestamp},
        )
        return weight.where(rebalance).to_dataset(name="weight")  # NaN on hold bars
```

Its config class is `BacktestConfig`, so no `constructor` is required.

```python
>>> from score_weighted import ScoreWeightedBacktester
>>> from quantlab.base.config import BacktestConfig
>>> custom = ScoreWeightedBacktester(BacktestConfig(
...     price_dataset=prices_of(cfg),
...     model=make_model(root / "custom", cfg, days),
...     model_mode="load",
...     checkpoint=str(checkpoint),
...     start_date="2024-02-12",
...     end_date="2024-03-22",
...     output_dir=str(root / "runs"),
...     rebalance_periods=5,
... ))
>>> w = custom.run().weights["weight"].to_pandas()
>>> w.iloc[[0, 1, 5]].round(3)
symbol        AAA    BBB    CCC   DDD  EEE    FFF
timestamp                                        
2024-02-12  0.000  0.000  0.541  0.00  0.0  0.459
2024-02-13    NaN    NaN    NaN   NaN  NaN    NaN
2024-02-19  0.037  0.647  0.000  0.16  0.0  0.156
```

To keep the top-N rule with another score, `TopNConstructor(TopNConfig(direction, top_n)).construct_panel(scores, tradable, rebalance)` accepts any score panel (a dataset with one variable per label) with a boolean tradability panel, such as a price dataset's `tradable_bars(prices, fill_column)`, and returns the same `weight` dataset; pass `fill_price=`, `valuation_price=` and `delisted=` as well to hand each bar the holdings the simulation would carry. Its per-bar method `construct(context)` decides one bar from a `PortfolioContext`, which is how a rule is written: subclass `PortfolioConstructor` (`quantlab.base.portfolio`) and implement `construct`. Another market is a `MarketSpec` with its own fill and valuation columns and annualization constants.

### Backtest any predictor

`config.model` does not have to be a `BaseModel`. The backtester depends only on the `Predictor` protocol in `quantlab.base.backtest`, which `BaseModel` satisfies without inheriting it. An ensemble that composes several models, or a wrapper around a model, is backtested unchanged as long as it has every member:

| Member | What the backtester uses it for |
|---|---|
| `labels`, `label_delays` | the label-delay check, the purge and the in-sample split (`lookahead_bars()`), the prediction variable names |
| `train_bounds`, `test_bounds` | the configured training and test windows |
| `predict_window(start, end)` | the prediction panel of a window; the predictor requests its own features and warm-up |
| `fingerprint_inputs(start, end)`, `training_fingerprint_inputs()` | `(key, factor or label, strategy, first, last)` entries that the backtester hashes into `data_fingerprint` |
| `collect()`, `train()` | train mode; `train` returns the checkpoint, whose `config.json` holds the training dates |
| `check_checkpoint(path)`, `load(path)` | load mode; the check runs before any feature is computed |
| `get_config()`, `from_config(config)` | `config.json`, and the rebuild in `load_backtester_from_config` through the class named in `"name"` |

The backtester reads no model config and calls no other model method. A config whose `model` lacks a member is refused at construction with a `TypeError` naming the missing members.

```python
>>> from typing import get_protocol_members
>>> from quantlab.base.backtest import Predictor
>>> sorted(get_protocol_members(Predictor))
['check_checkpoint', 'collect', 'fingerprint_inputs', 'from_config', 'get_config', 'label_delays', 'labels', 'load', 'predict_window', 'test_bounds', 'train', 'train_bounds', 'training_fingerprint_inputs']
```

A `SeedEnsemble` (see Average several seeds in the model guide) is such a predictor. In train mode `run()` trains every seed into one ensemble directory and records its `ensemble.json` as `trained_checkpoint`; in load mode `checkpoint` is that `ensemble.json`, and the training dates for the in-sample split are read from the ensemble-level `config.json` beside it, as for one model's checkpoint. The predictions are the members' averaged cross-sectional z-scores. The members read the same inputs, so the data fingerprints carry the keys of a single model, and `load_backtester_from_config` rebuilds the ensemble from its `get_config()` in the run's `config.json`. `MomentumHead` has nothing to fit, so its three seeds agree and the weights equal the single model's in the first session.

```python
>>> from quantlab.model.predefined.seed_ensemble import SeedEnsemble
>>> ensemble = SeedEnsemble(make_model(root / "ensemble", cfg, days), seeds=[0, 1, 2])
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=ensemble, model_mode="train", checkpoint=None,
... )).run()
>>> manifest = Path(trained.metrics["trained_checkpoint"])
>>> manifest.name, sorted(p.name for p in manifest.parent.iterdir())
('ensemble.json', ['config.json', 'ensemble.json', 'ic_series.csv', 'member_0', 'member_1', 'member_2', 'metrics.json', 'test_predictions.zarr'])
>>> replayed = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config,
...     model=SeedEnsemble(make_model(root / "replay", cfg, days, train_end=20), seeds=[0, 1, 2]),
...     checkpoint=str(manifest),
... )).run()
>>> replayed.metrics["training_window"]
('2024-01-01', '2024-02-23')
>>> bool((replayed.weights["weight"].fillna(0) == result.weights["weight"].fillna(0)).all())
True
>>> saved = json.loads((replayed.run_dir / "config.json").read_text())
>>> saved["model"]["seeds"], sorted(saved["data_fingerprint"])
([0, 1, 2], ['factor[0]:PastReturn', 'price_dataset'])
```

`run_cv()` replays an ensemble's cross-validation the same way. `SeedEnsemble.train_cv` (see Average several seeds in the model guide) writes a `cv_folds.json` in the format of a single model's `train_cv`, whose `checkpoint` entries are the `ensemble.json` of each `fold_{i}/`. With an ensemble as `model` and that directory as `cv_project_dir`, each fold loads its own ensemble, and the training dates of its in-sample split are cross-checked against the fold's ensemble-level `config.json`, as for a single model's fold. The backtester needs no change for it. With `MomentumHead` the seeds agree again, so the stitched weights equal those of the single model's cross-validation above.

```python
>>> cv_ensemble = SeedEnsemble(make_model(root / "ensemble_cv", cfg2, days2, train_end=29), seeds=[0, 1, 2])
>>> folds = cv_ensemble.collect().train_cv(train_periods=30)
>>> ensemble_cv_dir = Path(folds[0]["checkpoint"]).parent.parent
>>> ensemble_cv = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     cv_config,
...     model=SeedEnsemble(make_model(root / "ensemble_cv_backtest", cfg2, days2, train_end=29), seeds=[0, 1, 2]),
...     cv_project_dir=str(ensemble_cv_dir),
... )).run_cv()
>>> len(ensemble_cv.folds), Path(ensemble_cv.folds[0]["checkpoint"]).relative_to(ensemble_cv_dir).as_posix()
(8, 'fold_0/ensemble.json')
>>> bool((ensemble_cv.weights["weight"].fillna(0) == cv.weights["weight"].fillna(0)).all())
True
>>> ensemble_cv.metrics["stitched"]["in_sample_ranges"]
[]
```

## Notes

No borrow or short-financing cost is modelled, so short-side returns are optimistic; the metrics `notes` say so. Trade statistics use the position view: one trade is one symbol's round trip from entry to flat, and trimming a holding back to its target weight is not a closed trade. `Total Orders` is the number of fills.

`benchmark_dataset` must hold exactly one symbol (see [Compare against a benchmark](#compare-against-a-benchmark)). A concrete backtester must set `MARKET`. `run()` and `run_cv()` need `model` and `model_mode`, which `run_weights()` ignores. `run()` in load mode needs `checkpoint`, and `run_cv()` needs `cv_project_dir` and `model_mode="load"`. Every label's `delay` must equal the engine's `fill_delay_bars` (see [Label delay and fill delay](#label-delay-and-fill-delay)). On a price store without a CRSP ticker sidecar the backtester logs one warning that it falls back to labelling symbols by their axis names, and the run is unaffected; a dataset held in memory (`FrameDataset`) has no store and uses the axis names without a warning.

A backtester built with the wrong config class:

```python
>>> USEquityCrossectionSelectStockVectorBt(custom.config)
Traceback (most recent call last):
  ...
TypeError: USEquityCrossectionSelectStockVectorBt requires a CrossSectionBacktestConfig, got BacktestConfig
```

A score label the model does not declare:

```python
>>> USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config,
...     constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2, score_label="fwd_ret_5")),
... ))
Traceback (most recent call last):
  ...
ValueError: score_label 'fwd_ret_5' is not one of the predictor's labels ['open_ret_1']
```

`run_cv()` without `cv_project_dir`:

```python
>>> backtester.run_cv()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: run_cv() requires config.cv_project_dir, the train_cv project directory holding cv_folds.json
```

A stored config with a missing field is refused instead of being filled from current defaults:

```python
>>> del config["constructor"]
>>> load_backtester_from_config(config)
Traceback (most recent call last):
  ...
ValueError: quantlab.backtest.predefined.us_equity.USEquityCrossectionSelectStockVectorBt config is missing field(s) ['constructor']; refusing to fill them from the current dataclass defaults, which may differ from the values the stored backtest ran with
```

If a fold is missing from the middle of `cv_folds.json`, `run_cv()` refuses to stitch across the gap with `fold test segments are not contiguous: gap between fold 2 ending 2024-03-06 and fold 4 starting 2024-03-15; 6 price bar(s) in between belong to no fold, so a stitched out-of-sample curve would silently skip them`. Restore the manifest, or narrow `start_date` and `end_date` to a contiguous range of folds.

## See also

- [portfolio](portfolio.md) for the rules from predictions to weights: top-n, the mean-variance optimiser and its risk models.
- [model](model.md) for `train`, `train_cv`, `cv_folds.json` and `predict_panel`.
- [dataset](dataset.md) for the price dataset and [factor](factor.md) for the factors and labels a model consumes.
- [backend](backend.md) for the Zarr stores the weights and equity curve are written to.
- `BaseBacktester`, `BacktestResult`, `CVBacktestResult` and `MarketSpec` in `quantlab/base/backtest.py`; `BacktestConfig` and `CrossSectionBacktestConfig` in `quantlab/base/config.py`; `load_backtester_from_config` in `quantlab/utils/module.py`.
