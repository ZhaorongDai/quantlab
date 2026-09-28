# Return models

English | [简体中文](zh-CN/model.md)

A return model learns to predict a label, usually a forward return or a return rank, from a set of factors. Factors and labels are `xarray.Dataset` panels indexed by `(timestamp, symbol)`, and the prediction has the same shape, so the portfolio and backtest stages can consume it directly. The model layer supplies the shared training lifecycle (data collection, train and test windows, walk-forward cross-validation, checkpoints, early stopping, metrics, Weights & Biases logging); a concrete model head only implements the fitting itself.

## Prerequisites

The examples run on CPU without network access. Weights & Biases (W&B) is called on every training run, so switch it off for local experiments. On macOS, also limit OpenMP threads before importing `torch` or `xgboost` (see Notes).

```bash
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1   # macOS only
```

## The basics

### Inputs and outputs

A head is configured with a list of factor objects (the features) and a list of label objects (the targets). Both come from the factor layer (see the factor guide). `collect()` asks each of them for its panel from the model's `start_date` to `end_date`, with `read(start, end)` or `compute(start, end)`, merges them on `(timestamp, symbol)` and keeps the panel in memory. Internally a head works on arrays of shape `[num_times, num_symbols, num_features]` whose last axis follows `get_factor_names()` exactly, and produces `[num_times, num_symbols, num_labels]` predictions.

A label reads bars after t, so it must never be a feature. The model tells the two roles apart by `lookahead_bars()`: every label has it and no factor does, and construction refuses a label among the factors or a factor among the labels (see Notes). A real label is a `quantlab.label.forward.Forward`, a factor shifted forward by `delay + span` bars; `lookahead_bars()` returns that sum (see the factor guide).

The sessions below use a small in-memory stand-in for the factor and label objects, so they need no data store. It implements the few methods the model layer calls. The label is a noisy linear function of two factors, and its stand-in reports a lookahead of 2 bars, as a one-bar forward return with the default delay of 1 does.

```python
>>> import numpy as np, xarray as xr
>>> rng = np.random.default_rng(0)
>>> coords = {"timestamp": np.datetime64("2024-01-01") + np.arange(200),
...           "symbol": [f"S{i:02d}" for i in range(20)]}
>>> f_a, f_b = rng.standard_normal((2, 200, 20))
>>> ret = 0.05 * f_a - 0.02 * f_b + 0.05 * rng.standard_normal((200, 20))
>>> class Panel:
...     """Minimal stand-in for a factor object."""
...     def __init__(self, **variables):
...         data = {k: (("timestamp", "symbol"), v) for k, v in variables.items()}
...         self.ds = xr.Dataset(data, coords=coords)
...     def _get_factor_names(self): return list(self.ds.data_vars)
...     def read(self, start, end): return self.ds.sel(timestamp=slice(start, end))
...     def get_config(self): return {"factor_names": self._get_factor_names()}
>>> class LabelPanel(Panel):
...     """Minimal stand-in for a label whose value at t reads bars t+1 and t+2."""
...     def lookahead_bars(self): return 2
>>> factor, label = Panel(f_a=f_a, f_b=f_b), LabelPanel(ret=ret)
>>> from loguru import logger
>>> logger.remove()  # quantlab logs progress to stderr; silence it here
```

### Training and checkpoints

The config carries the factor and label objects, where checkpoints go, and four dates: the training window and the test window (all inclusive). The last `val_size` share of the training window (0.2 by default) is held out as a validation segment. `factor_data_strategy` and `label_data_strategy` say whether to read stored values (`"read"`) or compute them first (`"cal"`). `XGBoostRegressor` is a tree-model head; `hyperparameters` is passed to it.

```python
>>> from quantlab.base.config import ModelConfig
>>> from quantlab.library_model.xgb import XGBoostRegressor
>>> config = ModelConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     hyperparameters={"num_boost_round": 50, "max_depth": 3},
... )
>>> model = XGBoostRegressor(config).collect()
>>> model.num_times, model.num_symbols, model.get_factor_names(), model.get_label_names()
(200, 20, ['f_a', 'f_b'], ['ret'])
>>> checkpoint = model.train()
>>> checkpoint.name
'XGBoostRegressor_total.joblib'
```

`train()` returns the absolute path of the checkpoint. Each call writes a new trial directory `checkpoints/XGBoostRegressor_trial_<timestamp>/XGBoostRegressor_total/`, holding the checkpoint, a `config.json` sidecar, `metrics.json`, `ic_series.csv` and `test_predictions.zarr`. The sidecar stores the full config plus a `trained_on` record: the feature names, label names and symbols the model saw. `metrics.json` holds the scores of the run (see Metrics below), and the other two files hold the per-bar IC series and the test-segment predictions (see IC series and saved predictions below).

```python
>>> sorted(p.name for p in checkpoint.parent.iterdir())
['XGBoostRegressor_total.joblib', 'config.json', 'ic_series.csv', 'metrics.json', 'test_predictions.zarr']
>>> import json
>>> record = json.loads((checkpoint.parent / "config.json").read_text())
>>> record["trained_on"]["factor_names"], record["trained_on"]["label_names"], len(record["trained_on"]["symbols"])
(['f_a', 'f_b'], ['ret'], 20)
```

### Purging the label lookahead

The label at bar t reads bars up to t + L, where L is the largest `lookahead_bars()` among the model's labels. Every split boundary therefore drops the last L bars of the earlier segment, so no label used for fitting reads a bar of the later segment. `train()` cuts the training window into train and validation by position, then purges the train/validation and validation/test boundaries; the test segment keeps all its bars. With `val_size=0` the train segment is purged against test directly. Fitting thus loses L bars at each boundary. L is never a parameter: it follows from the labels.

In the session above L is 2. The training window 2024-01-01 to 2024-05-31 has 152 bars; the first 121 (to 2024-04-30) train and the other 31 validate. After the purge the train segment ends on 2024-04-28 and the validation segment on 2024-05-29. The splitting is done by `quantlab.utils.split.purge_segments`, which walk-forward folds (below) share.

```python
>>> from quantlab.utils.split import purge_segments
>>> train_bars, val_bars, test_bars = purge_segments(
...     coords["timestamp"],
...     [("2024-01-01", "2024-04-30"), ("2024-05-01", "2024-05-31"), ("2024-06-01", "2024-07-18")],
...     label.lookahead_bars(),
... )
>>> len(train_bars), str(train_bars[-1]), len(val_bars), str(val_bars[-1]), len(test_bars)
(119, '2024-04-28', 29, '2024-05-29', 48)
```

### Predicting and loading

`predict_panel` takes a feature panel and returns a panel with one variable per label name. Positions where every feature is NaN get NaN predictions. `predict` is the array-level counterpart: `[T, S, F]` in, `[T, S, L]` out for `XGBoostRegressor`. Use `predict_panel` when in doubt, because the array contract of `predict` belongs to each head.

```python
>>> predictions = model.predict_panel(factor.ds)
>>> predictions
<xarray.Dataset> Size: 34kB
Dimensions:    (timestamp: 200, symbol: 20)
Coordinates:
  * timestamp  (timestamp) datetime64[s] 2kB 2024-01-01 ... 2024-07-18
  * symbol     (symbol) <U3 240B 'S00' 'S01' 'S02' 'S03' ... 'S17' 'S18' 'S19'
Data variables:
    ret        (timestamp, symbol) float64 32kB 0.002621 -0.01783 ... -0.04321
>>> model.predict(np.zeros((5, 20, 2))).shape
(5, 20, 1)
```

`load()` restores a checkpoint into a model built from the same factors and labels. It first checks the file suffix, then compares the variable names recorded in `config.json` with the model's own.

```python
>>> restored = XGBoostRegressor(config).load(checkpoint)
>>> bool((restored.predict_panel(factor.ds)["ret"] == predictions["ret"]).all())
True
```

### Metrics

`quantlab.utils.metrics` scores `[T, S]` panels. Only cells where both prediction and target are finite count. Besides MSE, RMSE, MAE and R2 it provides two cross-sectional measures. IC is the Pearson correlation between prediction and target across the symbols of one timestamp, averaged over time. RankIC does the same on the per-timestamp ranks, so it measures ordering and ignores scale. A timestamp with fewer than two symbols where both are finite, or with a constant prediction or target, has no IC and is left out of the mean rather than counted as 0. ICIR and RankICIR measure how stable the signal is: the mean of the per-timestamp IC (or RankIC) divided by its sample standard deviation (`ddof=1`). They are NaN when fewer than two timestamps have an IC. Every head computes all eight on the raw values of the primary label (the first one) for the train, validation and test segments, plus `loss`: the head's loss on the training target (the label after the head's per-bar `_transform_target`, see Extending), computed per bar and averaged over bars, so every bar weighs the same whatever its number of symbols. They go to the W&B run summary as `train_*`, `val_*` and `test_*`, and `train()` writes the same dict to `metrics.json` beside `config.json`, with NaN and infinity as null. There are no `val_*` keys when the run has no validation segment (`val_size=0`). For a library head that loss is `_loss` (MSE by default); for a torch head it is `_val_one_batch`, by default its `_loss` (see Train a torch model).

```python
>>> metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
>>> sorted(metrics)[:9]
['test_ic', 'test_icir', 'test_loss', 'test_mae', 'test_mse', 'test_r2', 'test_rank_ic', 'test_rank_icir', 'test_rmse']
>>> {k: round(v, 3) for k, v in metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.707, 'val_rank_ic': 0.687, 'test_rank_ic': 0.679}
>>> {k: round(v, 3) for k, v in metrics.items() if k.endswith("icir")}
{'train_icir': 6.897, 'train_rank_icir': 6.202, 'val_icir': 6.144, 'val_rank_icir': 5.323, 'test_icir': 5.524, 'test_rank_icir': 4.813}
```

`regression_panel_metrics` computes the same scores for any panel:

```python
>>> from quantlab.utils.metrics import regression_panel_metrics
>>> test = slice("2024-06-01", "2024-07-18")
>>> scores = regression_panel_metrics(
...     predictions["ret"].sel(timestamp=test).values,
...     label.ds["ret"].sel(timestamp=test).values,
... )
>>> {name: round(value, 3) for name, value in scores.items()}
{'mse': 0.003, 'rmse': 0.051, 'mae': 0.04, 'r2': 0.491, 'ic': 0.698, 'rank_ic': 0.679, 'icir': 5.524, 'rank_icir': 4.813}
```

The per-timestamp values behind IC and RankIC are `cross_sectional_ic_series` and `cross_sectional_rank_ic_series` (NaN on a skipped timestamp), and `information_ratio` turns such a series into an ICIR. `regression_panel_metrics(pred, target, return_series=True)` returns both series with the metrics.

### IC series and saved predictions

Every run also writes two files beside `metrics.json`, so a new metric or an ensemble can be computed from disk without predicting again:

- `ic_series.csv` has the columns `split`, `timestamp`, `ic` and `rank_ic`: one row per bar of each evaluated segment (`train`, `val`, `test`, each in time order), holding the IC and RankIC of that bar on the raw primary label. They come from the same predictions as `metrics.json`: the mean of a segment's `ic` column is its `<split>_ic`, and its ICIR is `<split>_icir`. A bar without an IC (fewer than two valid symbols, or a constant cross-section) has no row. A cell is empty only when one of the two values exists and the other does not.
- `test_predictions.zarr` is the prediction panel of the test segment: `predict_panel` on the collected features, over the test bars and every collected symbol, with one variable per label. There is no store when the test segment has no bars.

```python
>>> import pandas as pd
>>> series = pd.read_csv(checkpoint.parent / "ic_series.csv", parse_dates=["timestamp"])
>>> series.head(3)
   split  timestamp        ic   rank_ic
0  train 2024-01-01  0.662219  0.690226
1  train 2024-01-02  0.798276  0.780451
2  train 2024-01-03  0.811311  0.826627
>>> series.groupby("split", sort=False).size().to_dict()
{'train': 119, 'val': 29, 'test': 48}
>>> test_ic = series[series["split"] == "test"]["ic"]
>>> round(float(test_ic.mean() / test_ic.std()), 3), round(metrics["test_icir"], 3)
(5.524, 5.524)
>>> saved = xr.open_zarr(checkpoint.parent / "test_predictions.zarr").load()
>>> dict(saved.sizes), list(saved.data_vars)
({'timestamp': 48, 'symbol': 20}, ['ret'])
>>> bool((saved["ret"] == predictions["ret"].sel(timestamp=saved.timestamp)).all())
True
```

### The class hierarchy

Every head derives from `BaseModel` through one of two variants. The variants differ in the training framework and in what a subclass must implement; `train`, `train_cv`, `load`, `predict` and `predict_panel` are written once in `BaseModel` and not overridden.

| Class | Framework | Checkpoint | Methods a head implements |
|---|---|---|---|
| `TorchModel` | torch, one cross-section of symbols per step | `.pth` | `window_bars`, `_init_model`, `_loss`; optional hooks with defaults (see Train a torch model) |
| `LibraryModel` | numpy rows, the library's own early stopping | `.joblib` | `_init_model`, `_fit_model`, `_forward`; optional `_transform_feature`, `_transform_target`, `_loss` (see Extending) |

Shipped heads: `XGBoostRegressor`, `XGBTDRegressor` and `RealMLPRegressor`, all `LibraryModel` heads, and the `TorchModel` heads `GATsRegressor` (`quantlab.torch_model.gats`, Qlib's GATs on the cross-section) and `MASTERRegressor` (`quantlab.torch_model.master`, the market-guided transformer MASTER). Torch heads live in `quantlab/torch_model/` and library heads in `quantlab/library_model/`. See the docstrings of `quantlab/base/model.py` and `quantlab/base/config.py` for the full config fields.

### Configuration and reserved hyperparameters

Every head takes one `ModelConfig`. It holds only what both variants read: the factors and labels, the save directory, the data strategies, the dates, `val_size`, `random_seed` and `hyperparameters`. Every training setting goes in `hyperparameters`, one flat dict that is recorded in `config.json`, so `config.json` alone rebuilds the model.

The base classes and the shipped heads read these keys from it themselves (`quantlab.base.model.RESERVED_HYPERPARAMETERS`, the union of `TORCH_RESERVED_HYPERPARAMETERS` and `LIBRARY_RESERVED_HYPERPARAMETERS`):

| Key | Read by | Default |
|---|---|---|
| `epochs` | `TorchModel`: the cap on training epochs; a value that is not a positive integer raises `ValueError` when training starts | 100 |
| `lr` | `TorchModel`: the learning rate of the default `_init_optim` | `1e-3` |
| `early_stopping` | the shipped library heads: turn on the library's native early stopping | `False` |
| `early_stopping_patience` | the shipped library heads: rounds (or the library's own unit) without improvement | 5 |
| `batch_size`, `num_workers` | `TorchModel`: the default `_dataloader` | `None` (one item per step), 0 |
| `panel_device` | `TorchModel`: where the training panel lives, `"auto"`, `"cuda"` or `"cpu"` (see Keep the training panel on the GPU) | `"auto"` |
| `panel_dtype` | `TorchModel`: the precision the features are stored in, `"float32"` or `"float16"` | `"float32"` |

Every other key is the head's own. `_init_model(num_features, num_labels, hyperparameters)` receives the whole dict, reserved keys included. Do not splat it into a network or a library constructor (`nn.GRU(**hyperparameters)`, `Regressor(**hyperparameters)`): read the keys the head needs by name, or pass the dict through the head's `head_hyperparameters` method first, which drops the keys its own variant reserves. The shipped library heads do the latter: they drop the early-stopping keys and keep `lr`, which pytabkit takes as its own learning rate.

## Common tasks

### Stop training early

With `"early_stopping": True` in `hyperparameters`, training stops when the validation loss has not improved for `early_stopping_patience` boosting rounds and the best model is kept. These are reserved keys the library heads read and never pass to the library; a torch head stops through its own `_should_stop` hook instead (see Train a torch model). For `XGBoostRegressor` the checkpoint is truncated to the best round. The metric is the RMSE on the validation segment. The booster itself is fit on a pooled concordance correlation loss (`1 - ccc`, see `ccc_objective` in `quantlab/library_model/xgb.py`); giving `objective` in `hyperparameters` switches back to a built-in xgboost objective.

```python
>>> from dataclasses import replace
>>> stopping = replace(config, hyperparameters={
...     "num_boost_round": 500, "max_depth": 3,
...     "early_stopping": True, "early_stopping_patience": 5,
... })
>>> stopped = XGBoostRegressor(stopping).collect()
>>> _ = stopped.train()
>>> stopped.model.num_boosted_rounds(), stopped.model.best_iteration
(52, 51)
```

### Cross-validate over walk-forward folds

`train_cv(train_periods, parallel=False, njobs=-1)` slides a training window over the timestamps between `start_date` and `end_date`. Each fold trains on `train_periods` timestamps and tests on the `train_periods // 5` timestamps right after them; the next fold starts one test length later. Each fold is fitted like `train()` on its own dates, so its training window loses its last L bars before the test segment, and is split and purged into train and validation inside. Every fold gets its own checkpoint and its own W&B run, and its checkpoint directory also holds the fold's `ic_series.csv` and `test_predictions.zarr` (the fold's metrics themselves go to `cv_folds.json`, below). The return value has one dict per fold with its dates (both ends inclusive), checkpoint path and `train_*`, `val_*` and `test_*` metrics. Its `train_end` is the last bar fitted, after the purge.

```python
>>> results = model.train_cv(train_periods=100)
>>> len(results)
5
>>> [(r["train_start"][:10], r["train_end"][:10], r["test_start"][:10], r["test_end"][:10]) for r in results]
[('2024-01-01', '2024-04-07', '2024-04-10', '2024-04-29'), ('2024-01-21', '2024-04-27', '2024-04-30', '2024-05-19'), ('2024-02-10', '2024-05-17', '2024-05-20', '2024-06-08'), ('2024-03-01', '2024-06-06', '2024-06-09', '2024-06-28'), ('2024-03-21', '2024-06-26', '2024-06-29', '2024-07-18')]
>>> [round(r["test_rank_ic"], 3) for r in results]
[0.691, 0.649, 0.695, 0.656, 0.697]
>>> [round(r["test_icir"], 3) for r in results]
[6.943, 5.013, 7.982, 5.508, 5.464]
>>> from pathlib import Path
>>> sorted(p.name for p in Path(results[0]["checkpoint"]).parent.iterdir())
['XGBoostRegressor_cv_fold_0.joblib', 'config.json', 'ic_series.csv', 'test_predictions.zarr']
```

All folds share one trial directory. Besides one sub-directory per fold it contains `cv_folds.json`, a manifest with `format_version` (2), the fold list as returned, purged `train_end` included, and a `cv_mean` block: the mean over folds of every `train_*`, `val_*` and `test_*` metric as `cv_mean_<metric>`, plus `cv_n_folds`. Non-finite fold values are left out of a mean, and NaN and infinity are written as null. A backtester replays a cross-validation run from this file; it refuses a version 1 manifest, written before the `cv_mean` block, so rerun `train_cv` for an old project. Each fold's `config.json` records the dates the fold was configured with, before the purge, so its `train_end` lies L bars after the manifest's.

```python
>>> trial = Path(results[0]["checkpoint"]).parent.parent
>>> sorted(p.name for p in trial.iterdir())
['XGBoostRegressor_cv_fold_0', 'XGBoostRegressor_cv_fold_1', 'XGBoostRegressor_cv_fold_2', 'XGBoostRegressor_cv_fold_3', 'XGBoostRegressor_cv_fold_4', 'cv_folds.json']
>>> manifest = json.loads((trial / "cv_folds.json").read_text())
>>> manifest["format_version"], len(manifest["folds"])
(2, 5)
>>> {k: round(v, 3) for k, v in manifest["cv_mean"].items() if k.endswith("rank_ic")}
{'cv_mean_train_rank_ic': 0.709, 'cv_mean_val_rank_ic': 0.687, 'cv_mean_test_rank_ic': 0.677}
>>> fold_0 = json.loads((Path(results[0]["checkpoint"]).parent / "config.json").read_text())
>>> fold_0["train_end"], manifest["folds"][0]["train_end"]
('2024-04-09T00:00:00', '2024-04-07T00:00:00')
```

`parallel=True` trains the folds concurrently on threads (`njobs` sets the pool size). Each fold works on a deep copy of the model, so memory grows with the number of jobs. Tree libraries already use every core, so set `nthread` in `hyperparameters` to roughly `os.cpu_count() // njobs`.

### Train a torch model

A torch head (`TorchModel`) is fed through standard PyTorch components. The base class builds a *training panel* of torch tensors from the collected data: features `x` (`[T, S, F]`), the training target (`[T, S, L]`), its `mask` (`[T, S]`), the raw labels `y_raw` and `present` (`[T, S]`, a cell with at least one finite feature), with the timestamps and symbols. The head's `_dataset(panel, bars, training)` returns a `torch.utils.data.Dataset` over some bars and `_dataloader(dataset, training)` batches it. The default dataset, `CrossSectionDataset` in `quantlab.torch_model.data`, gives one item per bar: the bar's *cross-section*, meaning its present symbols, each carrying its own last `window_bars` bars of features. The network then sees `[S_t, N, F]`, where the number of symbols S_t changes from bar to bar, so it must not depend on the order or the number of symbols. A symbol that joins after training still gets a prediction, and a symbol whose label is missing stays in the input as context.

Every item is a `Batch`: `x`, `y` (the training target, 0 where invalid), `mask` (True where the sample has a valid training target in every label), `y_raw` (the raw labels) and `where`, the timestamp index and symbol index of every sample, shaped like `mask`. For one cross-section `mask` is `[S_t]` and `y` is `[S_t, L]`. Predictions for every split and for `predict_panel` come from the `training=False` dataset and are put back into `[T, S, L]` through `where`; a dataset that leaves a present cell unpredicted, or predicts it twice, raises `ValueError` naming the bar.

The training target is computed once per fit, before the first epoch: `_transform_target(y, training)` receives each bar's raw labels, with `training=True` on the training bars only. The `keep` it returns only removes symbols from the loss. A target that should change every epoch, such as label noise, belongs in the head's own `_train_one_batch`.

A head writes three things: `window_bars` (N), `_init_model(num_features, num_labels, hyperparameters)` (the network, or several in an `nn.ModuleDict`) and `_loss(output, batch)`, the loss of one batch. `output` is whatever the network returned. Missing labels are already masked and set to 0 in `y`, so a loss only has to count the `mask` samples, as `masked_mse` in `quantlab.torch_model.training` does. Every other choice is an optional hook with a working default:

| Hook | Default |
|---|---|
| `_dataset(panel, bars, training)`: the PyTorch `Dataset` over `bars` | `CrossSectionDataset`, one item per bar; `SymbolSequenceDataset` gives Qlib-style per-symbol samples (see below) |
| `_dataloader(dataset, training)`: the `DataLoader` | `batch_size` and `num_workers` from the hyperparameters (`None`, one item per step, and 0); shuffled only in training, with a generator seeded from `random_seed`; the last batch never dropped; memory pinned only for a CPU panel read by workers of a CUDA model |
| `_transform_feature(x)`: a batch's raw `x`, NaN where missing, to the network input | clip to ±3, NaN to 0 |
| `_transform_target(y, training)`: one bar's raw labels to `(target, keep)`; `keep` drops symbols from the loss | `(y, None)`; helpers `cs_rank_norm` (Qlib `CSRankNorm`), `cs_zscore`, `drop_extreme` |
| `_init_optim(model)`: anything the training step understands, such as a dict of optimizers | Adam at `hyperparameters["lr"]` (`1e-3`) |
| `_train_one_batch(epoch, batch)`: one optimisation step, returns the loss | forward, `_loss`, backward, gradient values clipped to `grad_clip_value` (3.0), step |
| `_val_one_batch(epoch, batch)`: evaluation loss of one batch | `_loss` |
| `_test_one_batch(epoch, batch)`: called on every test batch after each epoch | nothing |
| `_forward(x)`: the prediction, `mask.shape + (L,)`, for metrics and `predict_panel` | `self.model(x)` |
| `_on_fit_start()`, `_should_stop(epoch, train_loss, val_loss)`, `_on_fit_end()` | run `hyperparameters["epochs"]` (100) epochs, keep the last weights |

The base moves each batch to the device and applies `_transform_feature`, checking that the shape is kept and every value is finite. Training runs in `train()` mode; validation, the test hook and prediction run under `no_grad` in `eval()` mode. `train_loss` and `val_loss` are the means of what the step hooks return; `val_loss` is None without a validation segment. `{split}_loss` is the mean of `_val_one_batch` over the split's batches, so with the default dataset every bar weighs the same whatever its number of symbols. The cap on training is the `epochs` hyperparameter and the default optimizer reads `lr` (see Configuration and reserved hyperparameters). Metrics are always computed on the raw first label.

A model with `window_bars` N needs N - 1 bars of history before the first bar it predicts. `collect()`, and a backtest's feature request, ask each factor for that many extra bars, counted on the factor's own dataset calendar, and warn when the data does not reach that far back. The purge before each split covers only the label lookahead, never the window. The stand-in panels here have no dataset, so these heads use a one-bar window.

The smallest head is a window, a network and a loss:

```python
>>> import torch
>>> import torch.nn as nn
>>> from quantlab.base.config import ModelConfig
>>> from quantlab.base.model import TorchModel
>>> from quantlab.torch_model.training import cs_zscore, masked_mse
>>> class LastBar(nn.Module):
...     """A linear map of each symbol's latest bar."""
...     def __init__(self, num_features, num_labels):
...         super().__init__()
...         self.linear = nn.Linear(num_features, num_labels)
...     def forward(self, x):              # x: [S_t, N, F]
...         return self.linear(x[:, -1])   # [S_t, L]
>>> class MinimalHead(TorchModel):
...     window_bars = 1
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         return LastBar(num_features, num_labels)
...     def _loss(self, output, batch):
...         return masked_mse(output, batch.y, batch.mask)
>>> torch_config = ModelConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     hyperparameters={"epochs": 20, "lr": 1e-2},
... )
>>> minimal = MinimalHead(torch_config).collect()
>>> minimal_checkpoint = minimal.train()
>>> minimal_checkpoint.name
'MinimalHead_total.pth'
>>> torch_metrics = json.loads((minimal_checkpoint.parent / "metrics.json").read_text())
>>> {k: round(v, 3) for k, v in torch_metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.697, 'val_rank_ic': 0.693, 'test_rank_ic': 0.69}
>>> one_more = factor.ds.isel(symbol=[0]).assign_coords(symbol=["S99"])
>>> wider = xr.concat([factor.ds, one_more], dim="symbol")
>>> minimal.predict_panel(wider).symbol.size
21
```

This head chooses its own optimizer, loss and stopping rule. It z-scores the target per bar, trains with SGD and momentum on the negative Pearson correlation between prediction and target, and keeps the weights of its best validation epoch, stopping after five epochs without improvement:

```python
>>> def masked_neg_corr(pred, y, mask):
...     """Minus the correlation of the first label over the valid samples."""
...     p, t = pred[mask, 0], y[mask, 0]
...     p, t = p - p.mean(), t - t.mean()
...     return -(p * t).sum() / (p.norm() * t.norm() + 1e-8)
>>> class CorrHead(MinimalHead):
...     def _transform_target(self, y, training):
...         return cs_zscore(y), None
...     def _init_optim(self, model):
...         return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
...     def _loss(self, output, batch):
...         return masked_neg_corr(output, batch.y, batch.mask)
...     def _on_fit_start(self):
...         self.best, self.bad, self.best_state = float("inf"), 0, None
...     def _should_stop(self, epoch, train_loss, val_loss):
...         if val_loss < self.best:
...             self.best, self.bad = val_loss, 0
...             self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
...         else:
...             self.bad += 1
...         return self.bad >= 5
...     def _on_fit_end(self):
...         self.model.load_state_dict(self.best_state)
>>> corr = CorrHead(replace(torch_config, hyperparameters={"epochs": 50})).collect()
>>> corr_metrics = json.loads((corr.train().parent / "metrics.json").read_text())
>>> {k: round(v, 3) for k, v in corr_metrics.items() if k in ("val_loss", "test_rank_ic")}
{'val_loss': -0.721, 'test_rank_ic': 0.691}
```

Qlib's sequence models (GRU, LSTM, ALSTM, Transformer) train on random `(timestamp, symbol)` samples rather than whole cross-sections. `SymbolSequenceDataset` in `quantlab.torch_model.data` gives that sample shape: one item per cell, holding the symbol's last `window_bars` bars as `[N, F]`, and PyTorch's default collation batches the items to `[B, N, F]` with `mask` and `where` shaped `[B]`. In training it holds only the cells with a valid training target; in evaluation it holds every present cell, so prediction still covers the whole cross-section. The training target is computed per bar over the whole cross-section before any batch is drawn, so a batch that mixes bars still sees each bar's cross-sectional rank or z-score, and `{split}_loss` still weighs every bar the same because the base splits a mixed batch by bar. The dataset gathers a whole batch of windows with one indexing call (`__getitems__`), and `window_bars=1` gives row samples for a torch row model.

The head below is Qlib's GRU on this dataset: `_dataset` returns the sequence dataset and `_dataloader` batches 800 samples, as Qlib does. A window longer than one bar needs warm-up bars, so the stand-in factor gets a calendar to count them on:

```python
>>> import pandas as pd
>>> from types import SimpleNamespace
>>> from torch.utils.data import DataLoader
>>> from quantlab.torch_model.data import SymbolSequenceDataset
>>> from quantlab.torch_model.training import cs_rank_norm
>>> days = pd.DatetimeIndex(coords["timestamp"])
>>> seq_factor = Panel(f_a=f_a, f_b=f_b)
>>> seq_factor.config = SimpleNamespace(dataset=SimpleNamespace(   # the bar n bars before date
...     bar_before=lambda date, n: days[max(days.searchsorted(pd.Timestamp(date)) - n, 0)]))
>>> seq_factor.store_range = lambda: None
>>> class GRUNet(nn.Module):
...     """Qlib's GRU: a GRU over each window, a linear map of its last step."""
...     def __init__(self, num_features, num_labels, hidden_size):
...         super().__init__()
...         self.rnn = nn.GRU(num_features, hidden_size, batch_first=True)
...         self.fc_out = nn.Linear(hidden_size, num_labels)
...     def forward(self, x):                  # x: [B, N, F]
...         out, _ = self.rnn(x)
...         return self.fc_out(out[:, -1])     # [B, L]
>>> class GRUHead(TorchModel):
...     window_bars = 8
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         return GRUNet(num_features, num_labels, hyperparameters["hidden_size"])
...     def _loss(self, output, batch):
...         return masked_mse(output, batch.y, batch.mask)
...     def _transform_target(self, y, training):
...         return cs_rank_norm(y), None
...     def _dataset(self, panel, bars, training):
...         return SymbolSequenceDataset(panel, bars, self.window_bars, training)
...     def _dataloader(self, dataset, training):
...         generator = torch.Generator().manual_seed(self.config.random_seed)
...         return DataLoader(dataset, batch_size=800, shuffle=training, generator=generator)
>>> gru = GRUHead(replace(
...     torch_config, factors=[seq_factor],
...     hyperparameters={"epochs": 30, "lr": 1e-2, "hidden_size": 16},
... )).collect()
>>> gru_checkpoint = gru.train()
>>> gru_metrics = json.loads((gru_checkpoint.parent / "metrics.json").read_text())
>>> {k: round(v, 3) for k, v in gru_metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.701, 'val_rank_ic': 0.693, 'test_rank_ic': 0.696}
>>> reloaded = GRUHead(gru.config).load(gru_checkpoint)
>>> gru_prediction = reloaded.predict_panel(seq_factor.ds)
>>> gru_prediction["ret"].shape, bool(np.isfinite(gru_prediction["ret"]).all())
((200, 20), True)
>>> xr.testing.assert_allclose(gru_prediction, gru.predict_panel(seq_factor.ds))
```

A head with yet another sample shape overrides `_dataset` (and, for multi-bar batches, `_dataloader` with its own sampler or collate function); `CrossSectionDataset` and `SymbolSequenceDataset` are the models to follow.

### Keep the training panel on the GPU

A torch head holds the whole collected panel (features, training target, masks and raw labels) as tensors on one device, and its datasets slice batches from it. On a GPU, slicing a bar from a panel already there takes a fraction of the time of copying it from CPU memory, so where the panel lives often decides how fast an epoch runs. `panel_device` chooses it when training starts and for every prediction:

- `"auto"` (the default) puts the panel on the GPU when it takes at most half of the free GPU memory, and otherwise keeps it in CPU memory. The choice is logged. Without CUDA, or with `num_workers > 0`, the panel stays in CPU memory.
- `"cuda"` forces the GPU. It raises `ValueError` before training with `num_workers > 0`, because a loader worker process cannot index a CUDA tensor, or when no CUDA device is available.
- `"cpu"` forces CPU memory. Use it when several runs share one GPU.

The default loader pins memory only for a CPU panel read by workers of a model on CUDA, since pinning without workers made loading slower in a measurement.

`panel_dtype="float16"` stores the features in half precision, halving the panel's largest part so a full-market panel fits on the GPU. Each batch is cast back to float32 before `_transform_feature`, so the network and the loss still run in float32. A feature too large for float16 (beyond ±65504) raises `ValueError` rather than becoming infinite; factors are normally z-scored long before that. Below, the minimal head from Train a torch model is trained again with its features stored in float16:

```python
>>> half = MinimalHead(replace(torch_config, hyperparameters={
...     "epochs": 20, "lr": 1e-2, "panel_dtype": "float16",
... })).collect()
>>> half_metrics = json.loads((half.train().parent / "metrics.json").read_text())
>>> round(half_metrics["test_rank_ic"], 3), round(torch_metrics["test_rank_ic"], 3)
(0.691, 0.69)
>>> gap = half.predict_panel(factor.ds)["ret"] - minimal.predict_panel(factor.ds)["ret"]
>>> f"{float(abs(gap).max()):.0e}"
'5e-05'
```

Measured once on the training server (RTX 5090 D with 32 GiB, 503 GB RAM) on 2026-09-29. The panel is CRSP market Alpha158, 3270 bars × 13015 symbols × 169 factors (2012–2024), which is 29 GB of float32 features, with a 5-bar forward return as the label. The head is a two-layer GRU (hidden 64) over `window_bars=8` with the default cross-section dataset. It trains 5 epochs on 2012–2019, with `val_size=0.2`, and is tested on 2020–2024; both runs use the same seed. The IC values compare the two runs with each other; they are not an evaluation of the model.

| | `panel_device="cuda"`, `panel_dtype="float16"` | `panel_device="cpu"`, `panel_dtype="float32"` |
|---|---|---|
| Epoch time (median of 5) | 10.8 s | 137.7 s |
| Peak GPU memory | 15.2 GiB | 1.4 GiB |
| Test IC / rank IC | 0.0265 / 0.0138 | 0.0260 / 0.0114 |

The two trained models' test predictions correlate at 0.986, both pooled and on average per bar; the difference is the training path, which float16 inputs change slightly. With the same weights, predicting the test segment from a float16 panel instead of a float32 one moves predictions by at most 7e-5 (their standard deviation is 0.21) and leaves the test IC and rank IC equal to nine decimals. The float32 panel does not fit in half of the GPU's free memory, so without float16 `"auto"` keeps this panel in CPU memory.

### Log to Weights & Biases

Each `train()` and each fold of `train_cv()` opens a W&B run named after the experiment inside a project named after the trial directory, with the full config attached. `XGBoostRegressor` logs the per-round training and validation curves and writes the final metrics and per-factor importance to the run summary. `XGBTDRegressor` logs the validation curve of every round (`val-rmse`, or `val-rmse/<label>` with several labels), the selected and trained round counts and the same importance charts, through a callback injected into pytabkit's inner `xgboost.train` call. `RealMLPRegressor` logs every epoch's mean training loss (`train-loss`) and validation error (`val-rmse`) at `step=epoch`, plus `best_val_rmse`, `epochs_trained` and the stopping epoch, through a Lightning callback injected into pytabkit's trainer (`quantlab.library_model.tabkit.active_callbacks`). `train_cv` opens an extra `<Class>_cv_summary` run whose summary is the manifest's `cv_mean` block. Torch heads log `train_loss` and `val_loss` every epoch and write the final metrics to the run summary. `WANDB_MODE=disabled` turns all of it off; `WANDB_MODE=offline` writes runs to a local `wandb/` directory that can be synced later with `wandb sync`. Without either setting, `wandb.init` needs a logged-in account.

## Extending

A new head subclasses `LibraryModel` or `TorchModel` and implements the methods listed in the table above; nothing else needs to change. The head is then usable with `train`, `train_cv`, `load`, `predict_panel` and the backtesters.

A `LibraryModel` head is fed rows, which the base builds. `_fit_model(train_rows, val_rows)` receives two `quantlab.library_model.data.Rows`, the second None when there is no validation segment or it has no usable row. Each carries `x [n, F]`, `y [n, L]` (the training target), `y_raw [n, L]` (the raw label) and `where`, the timestamp and symbol index of every row. Only cells with a valid training target become rows; NaN features stay, for the library's own missing-value handling. `_forward` maps `[n, F]` rows to `[n, L]` predictions, and at prediction time it sees every cell with a finite feature. `_fit_model` must leave the fitted object in `self.model`, and that object is what the checkpoint stores (via joblib). `_init_model` may return `None` when the real model is created during fitting. Three hooks are optional:

| Hook | Default |
|---|---|
| `_transform_feature(x)`: raw `[n, F]` rows to the library's input, same shape, never in place | infinities to NaN |
| `_transform_target(y, training)`: one bar's raw `[S_t, L]` labels (a float32 tensor, NaN where missing) to `(target, keep)`, computed once per bar before the fit, `training=True` on the training bars only; the same hook as a torch head's | the raw label |
| `_loss(target, pred)`: one bar's `[n, L]` rows to a number; its per-bar mean is `{split}_loss` | MSE |

```python
>>> from quantlab.base.model import LibraryModel
>>> class RidgeHead(LibraryModel):
...     """Closed-form ridge regression shared by every symbol."""
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         self.alpha = hyperparameters.get("alpha", 1.0)
...         return None  # the real model is built in _fit_model
...     def _fit_model(self, train_rows, val_rows):
...         x = np.nan_to_num(train_rows.x)  # rows keep NaN features; ridge needs numbers
...         x1 = np.c_[x, np.ones(len(x))]  # add an intercept column
...         penalty = self.alpha * np.eye(x1.shape[1])
...         penalty[-1, -1] = 0.0  # do not shrink the intercept
...         self.model = np.linalg.solve(x1.T @ x1 + penalty, x1.T @ train_rows.y)
...     def _forward(self, x):
...         return np.c_[np.nan_to_num(x), np.ones(len(x))] @ self.model
>>> ridge = RidgeHead(replace(config, hyperparameters={"alpha": 1.0})).collect()
>>> ridge_results = ridge.train_cv(train_periods=100)
>>> [round(r["test_rank_ic"], 3) for r in ridge_results]
[0.685, 0.667, 0.707, 0.672, 0.716]
>>> ridge.model.round(3).ravel().tolist()
[0.05, -0.02, -0.001]
```

Overriding `_transform_target` changes what the library fits and nothing else. Below, the ridge fits each bar's cross-sectional rank of the label, scaled to [-0.5, 0.5]. The metrics still score the raw label: the rank IC stays close, while the MSE against the raw return grows tenfold, because the predictions are now on the rank scale.

```python
>>> import torch
>>> class RankRidgeHead(RidgeHead):
...     def _transform_target(self, y, training):
...         ranks = torch.argsort(torch.argsort(y[:, 0])).float()  # this label has no NaN
...         return (ranks / (len(y) - 1) - 0.5)[:, None], None
>>> ranked = RankRidgeHead(replace(config, hyperparameters={"alpha": 1.0})).collect()
>>> ranked_metrics = json.loads((ranked.train().parent / "metrics.json").read_text())
>>> plain_metrics = json.loads((ridge.train().parent / "metrics.json").read_text())
>>> [(round(m["test_rank_ic"], 3), round(m["test_mse"], 3)) for m in (plain_metrics, ranked_metrics)]
[(0.716, 0.003), (0.69, 0.027)]
```

A `TorchModel` head is a window, a network and a loss, plus whichever optional hooks it overrides; `MinimalHead` under Train a torch model is a complete one, and `CorrHead` shows the optional hooks. The base class owns the training panel, the warm-up, the training target and its mask, the loaders' seeding, the epoch loop, evaluation, the placement of predictions through `where`, the metrics and the checkpoints.

## Notes

Errors below are quoted as raised, with paths shortened to `...`.

A head rejects anything but a `ModelConfig` as the first step of construction.

```text
TypeError: XGBoostRegressor requires a ModelConfig, got dict
```

A torch head whose `epochs` hyperparameter is not a positive integer fails when training starts.

```text
ValueError: MinimalHead: hyperparameters['epochs'] must be a positive integer, got 0
```

It also refuses a label among the factors and a factor among the labels. Wrap a factor in `Forward` to predict it.

```text
TypeError: XGBoostRegressor: factors[0] is the label LabelPanel, which reads bars after t; pass it in labels, not factors.
TypeError: XGBoostRegressor: labels[0] is Panel, which is not a label; wrap it in quantlab.label.forward.Forward to predict it.
```

`predict` and `predict_panel` need a trained or loaded model.

```text
ValueError: Model not initialized, please call load() or train() first
```

`load` reports a missing file, a file of the other variant, and a checkpoint whose recorded variables differ from the model's declared factors or labels, so a permuted or substituted input never reaches the model. Declare the same factors and labels in the same order as at training time.

```text
FileNotFoundError: checkpoints/missing.joblib not found
ValueError: Unsupported file type: '.pth'; XGBoostRegressor checkpoints use '.joblib' (...)
ValueError: XGBoostRegressor: checkpoint ... was trained on factor variables ['f_a', 'f_b'] (trained_on in its config.json), but this model declares ['f_z', 'f_b']; loading it would feed the model different or permuted inputs (...)
```

`predict_panel` needs every factor variable, and a torch head's `_forward` must return one row per sample of the batch (per symbol of the cross-section) and one column per label.

```text
ValueError: XGBoostRegressor.predict_panel: features are missing factor variable(s) ['f_b']
ValueError: TupleHead._forward must return a tensor shaped like the batch's mask plus the labels, [20, 1], got <class 'tuple'>
```

A torch head's evaluation dataset must predict every present cell of the bars asked for exactly once; here a custom dataset skips one symbol.

```text
ValueError: SkipHead: symbol 'S1' at bar 2024-02-05T00:00:00 was left unpredicted by the dataset CellDataset; every present cell must be predicted exactly once.
```

`train` needs all four window dates. Set them on the config, or through `train_cv`, which sets them per fold.

```text
ValueError: Training and testing start and end dates must be specified.
```

The purge must leave training bars. `train` raises when the train segment has no more than L bars before the purge (here 2 of them, with L = 2), and `train_cv` when `train_periods` is not above L (here a label with a lookahead of 10).

```text
ValueError: Empty training segment: purging the last 2 bars leaves 0 of 2 training timestamps for fitting.
ValueError: Fold 0: purging the last 10 bars leaves no training bar; raise train_periods.
```

Each fold tests on `train_periods // 5` bars, so `train_cv` refuses a `train_periods` below 5 before it trains anything.

```text
ValueError: XGBoostRegressor: train_cv(train_periods=4) needs at least 5 training bars, since each fold tests on train_periods // 5 bars.
```

`train_cv` overwrites the four `train_*` and `test_*` dates of the config with those of the last fold, so build a fresh config for a later `train()`. If `train_periods` leaves no room for a test segment, it logs `Skipping fold 0: test set exceeds data range` and returns an empty list (`[]`) without raising. Torch heads return no metrics from `train_cv`, so their fold dicts hold only dates and paths, `cv_mean` is empty and no summary run is opened.

`train()` returns only the checkpoint path; the metrics of the run are in `metrics.json` beside it. `train_cv` returns them directly for `LibraryModel` heads.

Checkpoints are pickles (`joblib` for `LibraryModel` heads, `torch.load` for `TorchModel` heads). Load only files you produced or trust.

Progress goes to stderr through `loguru` and `tqdm`. `logger.remove()` silences the log lines.

On macOS the `xgboost` wheel links Homebrew's OpenMP runtime while `torch` bundles its own. A process that uses both can crash or hang. Setting `OMP_NUM_THREADS=1` before the first import of either library avoids this, at the cost of single-threaded tree and torch code. Linux is not affected.

## See also

The factor guide (`docs/factor.md`) explains how factors and labels are produced, and the backtest guide (`docs/backtest.md`) shows how `predict_panel` output and a `cv_folds.json` manifest feed a backtest. The backend guide (`docs/backend.md`) covers the Zarr and xarray storage the panels use. API details are in the docstrings of `quantlab/base/model.py`, `quantlab/base/config.py` (`ModelConfig`, `ModelConfig`), `quantlab/torch_model/`, `quantlab/library_model/xgb.py`, `quantlab/library_model/backend.py` and `quantlab/utils/metrics.py`.
