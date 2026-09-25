# Backtesting

English | [简体中文](zh-CN/backtest.md)

A backtest takes a trained return model and a price dataset and shows how the model's predictions would have traded. The model predicts a score for every symbol on every bar, a selection rule turns the scores into target weights, and a simulation engine trades those weights and records an equity curve. Each run writes a run directory with the weights, the equity curve, metrics, an HTML report and the configuration needed to rebuild it.

The main classes are `BaseBacktester` (`quantlab/base/backtest.py`), the vectorbt engine `VectorBtBacktester` (`quantlab/backtest/engine_vectorbt.py`), the selection rule `CrossSectionTopNSelector` (`quantlab/backtest/selection.py`) and the US-equity backtester `USEquityCrossectionSelectStockVectorBt` (`quantlab/backtest/us_equity.py`).

## Prerequisites

Run the examples from the repository root with `uv run python`. On macOS, set `OMP_NUM_THREADS=1` before torch or xgboost is imported in the same process, and `WANDB_MODE=disabled` to keep experiment tracking off.

A backtest needs a price dataset whose store has the columns `adjOpen` and `adjClose`, and a model with a checkpoint written by `train()` or `train_cv()`. The sessions below use a synthetic setup: six symbols, one factor, one label and a model head with nothing to fit whose score is the past one-bar return. The last symbol, `FFF`, stops trading at bar 36. Save this as `demo_parts.py`.

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

from quantlab.base.config import DatasetConfig, MLConfig, PolarsFactorConfig
from quantlab.base.factor import FactorPolars
from quantlab.base.model import MLModel
from quantlab.dataset.stock import StockDataset

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
    """Feature past_ret_1: the one-bar return."""

    def _get_factor_lazyframe(self, lf):
        c = pl.col("adjClose")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((c / c.shift(1).over("symbol") - 1).alias("past_ret_1"))
                .select(["timestamp", "symbol", "past_ret_1"]))

    def _get_features(self, data):
        return data


class ForwardReturn(FactorPolars):
    """Label fwd_ret_1: the next-bar return."""

    def _get_factor_lazyframe(self, lf):
        c = pl.col("adjClose")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((c.shift(-1).over("symbol") / c - 1).alias("fwd_ret_1"))
                .select(["timestamp", "symbol", "fwd_ret_1"]))

    def _get_labels(self, data):
        return data


class MomentumHead(MLModel):
    """Predicts its first feature, so the score is the past return."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def make_model(root, cfg, days, train_end=39):
    day = lambda i: str(days[i].date())
    factor = PastReturn(PolarsFactorConfig(window=5, dataset=prices_of(cfg)))
    label = ForwardReturn(PolarsFactorConfig(
        window=0, dataset=prices_of(cfg), kwargs={"n_forward_periods": 1}))
    return MomentumHead(MLConfig(
        factors=[factor], labels=[label], model_save_dir=str(root / "models"),
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
    model.train_cv(train_periods=train_periods, gap_periods=0)
    return next(Path(model.config.model_save_dir).rglob("cv_folds.json")).parent
```

</details>

## The basics

### What a run does

`BaseBacktester.run()` backtests one model over the window `start_date` to `end_date`. It loads the checkpoint (or trains the model first), computes the features on the window, predicts a score per symbol and bar, asks the concrete class for target weights, simulates them, computes metrics and writes the run directory. `run_cv()` does the same for every fold of a `train_cv` run and stitches the folds into one curve.

The first session trains a checkpoint and backtests a rule that holds the two highest-scoring symbols and rebalances every five bars. Log lines go to stderr and are not shown.

```python
>>> import dataclasses, json, tempfile
>>> from pathlib import Path
>>> import pandas as pd
>>> import xarray as xr
>>> from demo_parts import *
>>> from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
>>> from quantlab.base.config import CrossSectionBacktestConfig
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
...     direction="long_only",
...     top_n=2,
... ))
>>> result = backtester.run()
>>> sorted(p.name for p in result.run_dir.iterdir())
['config.json', 'equity.zarr', 'fingerprint.json', 'liquidations.json', 'metrics.json', 'report.html', 'weights.zarr']
```

The run directory sits under `output_dir`, and the result holds the predictions, weights, simulation and metrics.

### The target-weight contract

The weights are a `weight` variable on `(timestamp, symbol)`. A row is either all NaN, meaning no rebalance on that bar and positions are kept, or all finite, meaning the portfolio is rebalanced to those fractions of its value. A rebalance row has a gross exposure (the sum of absolute weights) of at most 1. A symbol that is not selected has the weight `0.0` on a rebalance bar, never NaN.

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
2     2024-02-20    FFF  8221.655639  62.012933  254.924491  Sell
>>> adj_open = xr.open_zarr(root / "prices.zarr")["adjOpen"]
>>> float(adj_open.sel(timestamp="2024-02-13", symbol="CCC"))
52.82443690819098
```

Fees and slippage (`fees` and `slippage`, both 0.0005 by default) are proportional to each trade. A target percentage is measured against the portfolio value at the fill price of the bar it executes on.

### Delisted holdings

Both price columns are forward-filled before the simulation. A symbol that is held after a rebalance and has no raw fill price on the next bar is sold on that bar at its last known price, while the other symbols rebalance normally, and the sale is recorded as a forced liquidation. `FFF` has no price from 2024-02-20 and is in the first portfolio. The selector never picks a symbol that has no price on the next bar, so `FFF` is absent from the second portfolio above.

```python
>>> result.simulation.liquidations
[{'symbol': 'FFF', 'axis_symbol': 'FFF', 'signal_timestamp': Timestamp('2024-02-19 00:00:00'), 'fill_timestamp': Timestamp('2024-02-20 00:00:00'), 'price': 62.04395518050185}]
```

A symbol with no prices at the start of the window that was never held is treated as not yet listed and trades once its prices begin.

### Warm-up

The model's factors need history before `start_date`. The backtester reads as many bars before the window as the largest `window` among the model's factors, counted in price bars rather than calendar days. If the price calendar is shorter, the warm-up start is clamped to the first bar and a warning gives the shortfall. Predictions cover exactly the window bars, and a price symbol without a prediction gets NaN scores and is never selected.

### In-sample and out-of-sample

The model was trained on the bars `train_start` to `train_end`. Its labels look `n_forward_periods` bars ahead, so the effective training window extends that many bars past `train_end`. Window bars inside it are in-sample and the rest are out-of-sample. In load mode the training dates come from the `config.json` stored beside the checkpoint. When the window overlaps the training window the run logs a warning and continues.

```python
>>> m = result.metrics
>>> m["training_window"], m["in_sample_range"], m["out_of_sample_ranges"]
(('2024-01-01', '2024-02-26'), ('2024-02-12', '2024-02-26'), [('2024-02-27', '2024-03-22')])
```

All parts come from one continuous simulation, so capital and positions carry across the boundary. `whole` holds the engine statistics of the full window. `in_sample` and `out_of_sample` hold return-based statistics over their own bars, plus fill counts and turnover.

```python
>>> for part in ("whole", "in_sample", "out_of_sample"):
...     print(part, round(m[part]["Total Return [%]"], 2), round(m[part]["Sharpe Ratio"], 2), m[part]["order_count"])
whole -5.85 -2.32 19
in_sample -1.09 -0.79 6
out_of_sample -4.82 -3.76 13
>>> list(m["whole"])[:6]
['Start', 'End', 'Period', 'Start Value', 'End Value', 'Total Return [%]']
>>> round(m["whole"]["turnover"]["mean_per_rebalance"], 2)
1.34
```

Turnover is the one-sided traded value of a bar divided by the portfolio value before the fills, so a full buy-in from cash is about 1 and replacing the whole book about 2. The scores here are random-walk returns, so the negative results carry no meaning.

### The run directory

Each run writes a new directory `{ClassName}_{timestamp}` under `output_dir`. Files go to a hidden staging directory that is renamed when every file has been written, so `output_dir` only holds complete runs.

| File | Content |
| --- | --- |
| `config.json` | The configuration, with the price dataset and the model nested, and a data fingerprint. |
| `weights.zarr` | The target weights on `(timestamp, symbol)`. |
| `equity.zarr` | The portfolio `value` and per-bar `returns` on `timestamp`. |
| `metrics.json` | The same mapping as `result.metrics`; NaN and infinity are written as null. |
| `liquidations.json` | The forced liquidations. |
| `fingerprint.json` | A digest of the price and factor data the run read. |
| `report.html` | Equity, drawdown and monthly-return charts, a metrics table and notes. |

## Common tasks

### Train the model inside the backtest

With `model_mode="train"` the model is trained on its own configured dates first, and `checkpoint` is not needed. The window never changes the training dates. The checkpoint the run wrote is recorded in `metrics["trained_checkpoint"]`. This session also switches to a long-short book with one name per side.

```python
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=make_model(root / "train_mode", cfg, days),
...     model_mode="train", checkpoint=None, direction="long_short", top_n=1,
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

`train_cv` writes one checkpoint per walk-forward fold and a `cv_folds.json` manifest into its project directory. `run_cv()` reads the manifest, backtests each fold's test segment with the fold's own checkpoint, and simulates the concatenated weights once. `cv_project_dir` points at the project directory and `model_mode` must be `"load"`. Only folds whose test segment lies inside the window are used, and those segments must follow each other bar for bar.

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

The top-level files of the run directory describe the stitched curve, and `folds/fold_{i}/` holds each fold's own weights and equity. The stitched curve is one simulation, so capital carries across fold boundaries. Each fold also has an independent simulation that starts from `init_cash`, and the per-fold metrics come from those. The label here looks one bar ahead, so the first bar of every fold is in-sample for that fold's model.

```python
>>> stitched = cv.metrics["stitched"]
>>> stitched["in_sample_ranges"][:2], stitched["out_of_sample_ranges"][:2]
([('2024-02-12', '2024-02-12'), ('2024-02-20', '2024-02-20')], [('2024-02-13', '2024-02-19'), ('2024-02-21', '2024-02-27')])
>>> round(stitched["whole"]["Total Return [%]"], 2), round(cv.folds[0]["metrics"]["whole"]["Total Return [%]"], 2)
(-3.11, -1.62)
```

### Rebuild a run from its config

`config.json` names every class by its dotted import path, so `load_backtester_from_config` builds the same backtester, including its price dataset and model, and `run()` repeats the backtest into a new directory. When the data changed since the original run, the rebuilt run logs a warning per changed dataset and continues.

```python
>>> from quantlab.utils.module import load_backtester_from_config
>>> config = json.loads((result.run_dir / "config.json").read_text())
>>> config["name"], config["direction"], config["top_n"]
('quantlab.backtest.us_equity.USEquityCrossectionSelectStockVectorBt', 'long_only', 2)
>>> again = load_backtester_from_config(config).run()
>>> again.metrics["whole"] == result.metrics["whole"]
True
>>> again.run_dir == result.run_dir
False
```

The classes must be importable by dotted path. A class defined in a script is named `__main__.X` and cannot be found from another process, so the dataset, factors, model and backtester belong in modules. A config written by a train-mode run retrains when it is rebuilt; set `model_mode` to `"load"` and `checkpoint` to the recorded `trained_checkpoint` to replay the same model.

## Extending

A new selection rule is a subclass of `VectorBtBacktester` with three members: `config_cls`, `MARKET` and `_generate_signals(predictions, prices)`. The method returns a dataset whose `weight` variable satisfies the contract above. `predictions` and `prices` share the same `(timestamp, symbol)` axes. The rule below weights each eligible symbol in proportion to its positive score and stays flat when no score is positive. It reuses `rebalance_mask` and the `US_EQUITY_MARKET` price conventions. Save it as `score_weighted.py`.

```python
"""A new selection rule: long-only weights proportional to positive scores."""

import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import rebalance_mask
from quantlab.backtest.us_equity import US_EQUITY_MARKET
from quantlab.base.config import BacktestConfig


class ScoreWeightedBacktester(VectorBtBacktester):
    config_cls = BacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions, prices):
        label = self.config.model.get_label_names()[0]
        scores = predictions[label].transpose("timestamp", "symbol")
        # A symbol without a price on the next bar cannot be filled: not eligible.
        next_fill = prices[self.MARKET.fill_price_column].shift(timestamp=-1)
        positive = scores.where(next_fill.notnull()).clip(min=0).fillna(0.0)
        total = positive.sum("symbol")
        weight = (positive / total.where(total > 0)).fillna(0.0)  # flat if no score
        rebalance = xr.DataArray(
            rebalance_mask(prices.sizes["timestamp"], self.config.rebalance_periods),
            dims="timestamp", coords={"timestamp": prices.timestamp},
        )
        return weight.where(rebalance).to_dataset(name="weight")  # NaN on hold bars
```

Its config class is `BacktestConfig`, so `direction` and `top_n` are not required.

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
symbol        AAA    BBB    CCC    DDD  EEE    FFF
timestamp                                         
2024-02-12  0.000  0.000  0.541  0.000  0.0  0.459
2024-02-13    NaN    NaN    NaN    NaN  NaN    NaN
2024-02-19  0.044  0.767  0.000  0.189  0.0  0.000
```

To keep the top-N rule with another score, `CrossSectionTopNSelector(direction, top_n).select(scores, next_fill_price, rebalance)` accepts any score panel and returns the same `weight` dataset. Another market is a `MarketSpec` with its own fill and valuation columns and annualization constants.

## Notes

No borrow or short-financing cost is modelled, so short-side returns are optimistic; the metrics `notes` say so. Trade statistics use the position view: one trade is one symbol's round trip from entry to flat, and trimming a holding back to its target weight is not a closed trade. `order_count` is the number of fills.

`benchmark_dataset` is reserved; supplying one raises `NotImplementedError`. A concrete backtester must set `MARKET`. `run()` in load mode needs `checkpoint`, and `run_cv()` needs `cv_project_dir` and `model_mode="load"`. On a price store without a CRSP ticker sidecar the backtester logs one warning that it falls back to labelling symbols by their axis names, and the run is unaffected.

A backtester built with the wrong config class:

```python
>>> USEquityCrossectionSelectStockVectorBt(custom.config)
Traceback (most recent call last):
  ...
TypeError: USEquityCrossectionSelectStockVectorBt requires a CrossSectionBacktestConfig, got BacktestConfig
```

A score label the model does not declare:

```python
>>> USEquityCrossectionSelectStockVectorBt(dataclasses.replace(backtester.config, score_label="fwd_ret_5"))
Traceback (most recent call last):
  ...
ValueError: score_label 'fwd_ret_5' is not one of the model's labels ['fwd_ret_1']
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
>>> del config["top_n"]
>>> load_backtester_from_config(config)
Traceback (most recent call last):
  ...
ValueError: quantlab.backtest.us_equity.USEquityCrossectionSelectStockVectorBt config is missing field(s) ['top_n']; refusing to fill them from the current dataclass defaults, which may differ from the values the stored backtest ran with (...)
```

If a fold is missing from the middle of `cv_folds.json`, `run_cv()` refuses to stitch across the gap with `fold test segments are not contiguous: gap between fold 2 ending 2024-03-06 and fold 4 starting 2024-03-15; 6 price bar(s) in between belong to no fold, so a stitched out-of-sample curve would silently skip them`. Restore the manifest, or narrow `start_date` and `end_date` to a contiguous range of folds.

## See also

- [model](model.md) for `train`, `train_cv`, `cv_folds.json` and `predict_panel`.
- [dataset](dataset.md) for the price dataset and [factor](factor.md) for the factors and labels a model consumes.
- [backend](backend.md) for the Zarr stores the weights and equity curve are written to.
- `BaseBacktester`, `BacktestResult`, `CVBacktestResult` and `MarketSpec` in `quantlab/base/backtest.py`; `BacktestConfig` and `CrossSectionBacktestConfig` in `quantlab/base/config.py`; `load_backtester_from_config` in `quantlab/utils/module.py`.
