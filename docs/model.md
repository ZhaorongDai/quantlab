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
>>> from quantlab.base.config import MLConfig
>>> from quantlab.ml_model.xgb import XGBoostRegressor
>>> config = MLConfig(
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

`train()` returns the absolute path of the checkpoint. Each call writes a new trial directory `checkpoints/XGBoostRegressor_trial_<timestamp>/XGBoostRegressor_total/`, holding the checkpoint, a `config.json` sidecar and `metrics.json`. The sidecar stores the full config plus a `trained_on` record: the feature names, label names and symbols the model saw. `metrics.json` holds the scores of the run (see Metrics below).

```python
>>> sorted(p.name for p in checkpoint.parent.iterdir())
['XGBoostRegressor_total.joblib', 'config.json', 'metrics.json']
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

`quantlab.utils.metrics` scores `[T, S]` panels. Only cells where both prediction and target are finite count. Besides MSE, RMSE, MAE and R2 it provides two cross-sectional measures. IC is the Pearson correlation between prediction and target across the symbols of one timestamp, averaged over time. RankIC does the same on the per-timestamp ranks, so it measures ordering and ignores scale. Every head computes all six, plus the fitting `loss`, on the raw values of the primary label (the first one) for the train, validation and test segments. They go to the W&B run summary as `train_*`, `val_*` and `test_*`, and `train()` writes the same dict to `metrics.json` beside `config.json`, with NaN and infinity as null. There are no `val_*` keys when the run has no validation segment (`val_size=0`). Torch heads report the same keys; their `loss` is the mean of the head's `_val_one_batch` over the segment, on the transformed target (see Train a torch model).

```python
>>> metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
>>> sorted(metrics)[:7]
['test_ic', 'test_loss', 'test_mae', 'test_mse', 'test_r2', 'test_rank_ic', 'test_rmse']
>>> {k: round(v, 3) for k, v in metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.707, 'val_rank_ic': 0.687, 'test_rank_ic': 0.679}
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
{'mse': 0.003, 'rmse': 0.051, 'mae': 0.04, 'r2': 0.491, 'ic': 0.698, 'rank_ic': 0.679}
```

### The class hierarchy

Every head derives from `BaseModel` through one of two variants. The variants differ in the training framework and in what a subclass must implement; `train`, `train_cv`, `load`, `predict` and `predict_panel` are written once in `BaseModel` and not overridden.

| Class | Framework | Config | Checkpoint | Methods a head implements |
|---|---|---|---|---|
| `DLModel` | torch, one cross-section of symbols per step | `DLConfig` | `.pth` | `_init_model`, `_init_optim`, `_train_one_batch`, `_val_one_batch`, `_test_one_batch`, plus the declarations `window_bars`, `target_transform`, `stopping` |
| `MLModel` | numpy, the library's own early stopping | `MLConfig` | `.joblib` | `_init_model`, `_preprocess`, `_fit_model`, `_forward` |

Shipped heads: `XGBoostRegressor`, `XGBTDRegressor` and `RealMLPRegressor`, all `MLModel` heads; no torch head ships yet. See the docstrings of `quantlab/base/model.py` and `quantlab/base/config.py` for the full config fields.

## Common tasks

### Stop training early

With `early_stopping=True`, training stops when the validation loss has not improved for `early_stopping_patience` boosting rounds and the best model is kept. These are `MLConfig` fields; a torch head declares its own stopping rule instead (see Train a torch model). For `XGBoostRegressor` the checkpoint is truncated to the best round. The metric is the RMSE on the validation segment. The booster itself is fit on a pooled concordance correlation loss (`1 - ccc`, see `ccc_objective` in `quantlab/ml_model/xgb.py`); giving `objective` in `hyperparameters` switches back to a built-in xgboost objective.

```python
>>> from dataclasses import replace
>>> stopping = replace(config, early_stopping=True, early_stopping_patience=5,
...                    hyperparameters={"num_boost_round": 500, "max_depth": 3})
>>> stopped = XGBoostRegressor(stopping).collect()
>>> _ = stopped.train()
>>> stopped.model.num_boosted_rounds(), stopped.model.best_iteration
(52, 51)
```

### Cross-validate over walk-forward folds

`train_cv(train_periods, parallel=False, njobs=-1)` slides a training window over the timestamps between `start_date` and `end_date`. Each fold trains on `train_periods` timestamps and tests on the `train_periods // 5` timestamps right after them; the next fold starts one test length later. Each fold is fitted like `train()` on its own dates, so its training window loses its last L bars before the test segment, and is split and purged into train and validation inside. Every fold gets its own checkpoint and its own W&B run, and the return value has one dict per fold with its dates (both ends inclusive), checkpoint path and `train_*`, `val_*` and `test_*` metrics. Its `train_end` is the last bar fitted, after the purge.

```python
>>> results = model.train_cv(train_periods=100)
>>> len(results)
5
>>> [(r["train_start"][:10], r["train_end"][:10], r["test_start"][:10], r["test_end"][:10]) for r in results]
[('2024-01-01', '2024-04-07', '2024-04-10', '2024-04-29'), ('2024-01-21', '2024-04-27', '2024-04-30', '2024-05-19'), ('2024-02-10', '2024-05-17', '2024-05-20', '2024-06-08'), ('2024-03-01', '2024-06-06', '2024-06-09', '2024-06-28'), ('2024-03-21', '2024-06-26', '2024-06-29', '2024-07-18')]
>>> [round(r["test_rank_ic"], 3) for r in results]
[0.691, 0.649, 0.695, 0.656, 0.697]
```

All folds share one trial directory. Besides one sub-directory per fold it contains `cv_folds.json`, a manifest with `format_version` (2), the fold list as returned, purged `train_end` included, and a `cv_mean` block: the mean over folds of every `train_*`, `val_*` and `test_*` metric as `cv_mean_<metric>`, plus `cv_n_folds`. Non-finite fold values are left out of a mean, and NaN and infinity are written as null. A backtester replays a cross-validation run from this file; it refuses a version 1 manifest, written before the `cv_mean` block, so rerun `train_cv` for an old project. Each fold's `config.json` records the dates the fold was configured with, before the purge, so its `train_end` lies L bars after the manifest's.

```python
>>> from pathlib import Path
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

A torch head (`DLModel`) trains on one bar per step: the *cross-section* of that bar, meaning the symbols with at least one finite feature there, each carrying its own last `window_bars` bars of features. The network maps `[S_t, N, F]` to `[S_t, L]`, where the number of symbols S_t changes from bar to bar, so it must not depend on the order or the number of symbols. A symbol that joins after training still gets a prediction. A symbol whose label is missing stays in the cross-section as input and adds nothing to the loss. Features are clipped to ±3 and NaN becomes 0 (a head can set `clip_features = False`); window rows before a symbol's first bar are zeros.

A head implements the same hooks as before the cross-section design, now fed one bar at a time. `_init_model(num_features, num_labels, hyperparameters)` builds the network. `_init_optim(model)` returns the optimizer, kept on `self.optim` (`config.lr` is there for it to read). `_train_one_batch(epoch, x, y)` takes one optimisation step and returns its loss; `_val_one_batch` returns the validation loss of one bar, and `_test_one_batch` is called on every test bar after each epoch. `x` is `[S_t, N, F]`; `y` is `[S_t, L]`, the bar's labels after the head's `target_transform`, with NaN where a label is missing, so the loss must leave those entries out (`masked_mse` in `quantlab.dl_model.training` does). The loss, the optimizer, gradient clipping and anything else about a step are the head's choice. The head also declares three things. `window_bars` is N. `target_transform` is `TargetTransform("rank")` (Qlib's `CSRankNorm`) or `TargetTransform("zscore")`, optionally with `drop_extreme`, the share of each tail removed from the bar's training cross-section; metrics still use the raw first label. `stopping` is `ValLossPatience(patience)` (stop after `patience` epochs without a lower validation loss, keep the best epoch's weights) or `TrainLossThreshold(threshold, max_epochs)` (stop once the epoch's training loss reaches the threshold, keep the last weights); both read the mean losses the step hooks return. `DLConfig` adds `epochs`, the cap on either rule, and `lr`. There is no batch size: bars are visited in shuffled order, one per step. An optional `_preprocess(x)` transforms every window at training and prediction time alike.

A model with `window_bars` N needs N - 1 bars of history before the first bar it predicts. `collect()`, and a backtest's feature request, ask each factor for that many extra bars, counted on the factor's own dataset calendar, and warn when the data does not reach that far back. The stand-in panels here have no dataset, so this head uses a one-bar window.

```python
>>> import torch.nn as nn
>>> from quantlab.base.config import DLConfig
>>> from quantlab.base.model import DLModel
>>> import torch
>>> from quantlab.dl_model.training import TargetTransform, ValLossPatience, masked_mse
>>> class LastBar(nn.Module):
...     """A linear map of each symbol's latest bar."""
...     def __init__(self, num_features, num_labels):
...         super().__init__()
...         self.linear = nn.Linear(num_features, num_labels)
...     def forward(self, x):              # x: [S_t, N, F]
...         return self.linear(x[:, -1])   # [S_t, L]
>>> class LinearHead(DLModel):
...     window_bars = 1
...     target_transform = TargetTransform("zscore")
...     stopping = ValLossPatience(patience=5)
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         return LastBar(num_features, num_labels)
...     def _init_optim(self, model):
...         return torch.optim.Adam(model.parameters(), lr=self.config.lr)
...     def _train_one_batch(self, epoch, x, y):
...         self.optim.zero_grad()
...         loss = masked_mse(self.model(x), y)
...         loss.backward()
...         self.optim.step()
...         return loss.detach()
...     def _val_one_batch(self, epoch, x, y):
...         return masked_mse(self.model(x), y)
...     def _test_one_batch(self, epoch, x, y):
...         return masked_mse(self.model(x), y)
>>> dl_config = DLConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     epochs=50, lr=1e-2,
... )
>>> linear = LinearHead(dl_config).collect()
>>> linear_checkpoint = linear.train()
>>> linear_checkpoint.name
'LinearHead_total.pth'
>>> dl_metrics = json.loads((linear_checkpoint.parent / "metrics.json").read_text())
>>> {k: round(v, 3) for k, v in dl_metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.699, 'val_rank_ic': 0.691, 'test_rank_ic': 0.69}
>>> one_more = factor.ds.isel(symbol=[0]).assign_coords(symbol=["S99"])
>>> wider = xr.concat([factor.ds, one_more], dim="symbol")
>>> linear.predict_panel(wider).symbol.size
21
```

### Log to Weights & Biases

Each `train()` and each fold of `train_cv()` opens a W&B run named after the experiment inside a project named after the trial directory, with the full config attached. `XGBoostRegressor` logs the per-round training and validation curves and writes the final metrics and per-factor importance to the run summary. `XGBTDRegressor` logs the validation curve of every round (`val-rmse`, or `val-rmse/<label>` with several labels), the selected and trained round counts and the same importance charts, through a callback injected into pytabkit's inner `xgboost.train` call. `RealMLPRegressor` logs every epoch's mean training loss (`train-loss`) and validation error (`val-rmse`) at `step=epoch`, plus `best_val_rmse`, `epochs_trained` and the stopping epoch, through a Lightning callback injected into pytabkit's trainer (`quantlab.ml_model.tabkit.active_callbacks`). `train_cv` opens an extra `<Class>_cv_summary` run whose summary is the manifest's `cv_mean` block. Torch heads log `train_loss` and `val_loss` every epoch and write the final metrics to the run summary. `WANDB_MODE=disabled` turns all of it off; `WANDB_MODE=offline` writes runs to a local `wandb/` directory that can be synced later with `wandb sync`. Without either setting, `wandb.init` needs a logged-in account.

## Extending

A new head subclasses `MLModel` or `DLModel` and implements the methods listed in the table above; nothing else needs to change. The head is then usable with `train`, `train_cv`, `load`, `predict_panel` and the backtesters.

An `MLModel` head gets `[T, S, F]` features and `[T, S, L]` labels as arrays. `_fit_model` must leave the fitted object in `self.model`, and that object is what the checkpoint stores (via joblib). `_preprocess` runs on every array, labels included, and must return a copy. `_init_model` may return `None` when the real model is created during fitting.

```python
>>> from quantlab.base.model import MLModel
>>> class RidgeHead(MLModel):
...     """Closed-form ridge regression shared by every symbol."""
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         self.alpha = hyperparameters.get("alpha", 1.0)
...         return None  # the real model is built in _fit_model
...     def _preprocess(self, data):
...         return np.array(data, dtype=np.float64, copy=True)  # a copy, never in place
...     def _fit_model(self, train_x, train_y, val_x, val_y):
...         x = np.nan_to_num(train_x.reshape(-1, train_x.shape[-1]))
...         y = train_y.reshape(-1, train_y.shape[-1])
...         keep = np.isfinite(y).all(axis=1)  # drop rows without a label
...         x1 = np.c_[x[keep], np.ones(keep.sum())]  # add an intercept column
...         penalty = self.alpha * np.eye(x1.shape[1])
...         penalty[-1, -1] = 0.0  # do not shrink the intercept
...         self.model = np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y[keep])
...     def _forward(self, x):
...         rows = np.nan_to_num(x.reshape(-1, x.shape[-1]))
...         out = np.c_[rows, np.ones(len(rows))] @ self.model
...         return out.reshape(x.shape[0], x.shape[1], -1)
>>> ridge = RidgeHead(replace(config, hyperparameters={"alpha": 1.0})).collect()
>>> ridge_results = ridge.train_cv(train_periods=100)
>>> [round(r["test_rank_ic"], 3) for r in ridge_results]
[0.685, 0.667, 0.707, 0.672, 0.716]
>>> ridge.model.round(3).ravel().tolist()
[0.05, -0.02, -0.001]
```

A `DLModel` head is its network, its optimizer, its three step hooks and its three declarations; `LinearHead` under Train a torch model is a complete one. The head decides what one step does: the loss, the optimizer, clipping, auxiliary outputs. The base class owns the windows, the bar order, the target transform, the stopping rule, the metrics and the checkpoints.

## Notes

Errors below are quoted as raised, with paths shortened to `...`.

A head rejects the wrong config class as the first step of construction.

```text
TypeError: XGBoostRegressor requires a MLConfig, got DLConfig
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

`predict_panel` needs every factor variable, and a torch network must return one row per symbol of the cross-section and one column per label.

```text
ValueError: XGBoostRegressor.predict_panel: features are missing factor variable(s) ['f_b']
ValueError: LinearHead: the network must map [S_t, N, F] to a [S_t, L] = [20, 1] tensor, got <class 'tuple'>
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

`train()` returns only the checkpoint path; the metrics of the run are in `metrics.json` beside it. `train_cv` returns them directly for `MLModel` heads.

Checkpoints are pickles (`joblib` for `MLModel` heads, `torch.load` for `DLModel` heads). Load only files you produced or trust.

Progress goes to stderr through `loguru` and `tqdm`. `logger.remove()` silences the log lines.

On macOS the `xgboost` wheel links Homebrew's OpenMP runtime while `torch` bundles its own. A process that uses both can crash or hang. Setting `OMP_NUM_THREADS=1` before the first import of either library avoids this, at the cost of single-threaded tree and torch code. Linux is not affected.

## See also

The factor guide (`docs/factor.md`) explains how factors and labels are produced, and the backtest guide (`docs/backtest.md`) shows how `predict_panel` output and a `cv_folds.json` manifest feed a backtest. The backend guide (`docs/backend.md`) covers the Zarr and xarray storage the panels use. API details are in the docstrings of `quantlab/base/model.py`, `quantlab/base/config.py` (`DLConfig`, `MLConfig`), `quantlab/dl_model/`, `quantlab/ml_model/xgb.py`, `quantlab/ml_model/backend.py` and `quantlab/utils/metrics.py`.
