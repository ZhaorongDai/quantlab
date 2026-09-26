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

A head is configured with a list of factor objects (the features) and a list of label objects (the targets). Both come from the factor layer (see the factor guide). `collect()` reads or computes them, merges them on `(timestamp, symbol)` and keeps the panel in memory. Internally a head works on arrays of shape `[num_times, num_symbols, num_features]` whose last axis follows `get_factor_names()` exactly, and produces `[num_times, num_symbols, num_labels]` predictions.

The sessions below use a small in-memory stand-in for the factor and label objects, so they need no data store. It implements the few methods the model layer calls. The label is a noisy linear function of two factors.

```python
>>> import numpy as np, xarray as xr
>>> from types import SimpleNamespace
>>> rng = np.random.default_rng(0)
>>> coords = {"timestamp": np.datetime64("2024-01-01") + np.arange(200),
...           "symbol": [f"S{i:02d}" for i in range(20)]}
>>> f_a, f_b = rng.standard_normal((2, 200, 20))
>>> ret = 0.05 * f_a - 0.02 * f_b + 0.05 * rng.standard_normal((200, 20))
>>> class Panel:
...     """Minimal stand-in for a factor or label object."""
...     def __init__(self, **variables):
...         data = {k: (("timestamp", "symbol"), v) for k, v in variables.items()}
...         self.ds = xr.Dataset(data, coords=coords)
...         self.config = SimpleNamespace(start_date=None, end_date=None)
...     def _reset_dataset_config(self): pass
...     def _get_factor_names(self): return list(self.ds.data_vars)
...     def read(self): return self
...     def get_features(self): return self.ds
...     def get_labels(self): return self.ds
...     def get_config(self): return {"factor_names": self._get_factor_names()}
>>> factor, label = Panel(f_a=f_a, f_b=f_b), Panel(ret=ret)
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

`train()` returns the absolute path of the checkpoint. Each call writes a new trial directory `checkpoints/XGBoostRegressor_trial_<timestamp>/XGBoostRegressor_total/`, holding the checkpoint and a `config.json` sidecar. The sidecar stores the full config plus a `trained_on` record: the feature names, label names and symbols the model saw.

```python
>>> sorted(p.name for p in checkpoint.parent.iterdir())
['XGBoostRegressor_total.joblib', 'config.json']
>>> import json
>>> record = json.loads((checkpoint.parent / "config.json").read_text())
>>> record["trained_on"]["factor_names"], record["trained_on"]["label_names"], len(record["trained_on"]["symbols"])
(['f_a', 'f_b'], ['ret'], 20)
```

### Predicting and loading

`predict_panel` takes a feature panel and returns a panel with one variable per label name. Positions where every feature is NaN get NaN predictions. `predict` is the array-level counterpart: `[T, S, F]` in, `[T, S, L]` out for `XGBoostRegressor`. Use `predict_panel` when in doubt, because the array contract of `predict` belongs to each head.

```python
>>> predictions = model.predict_panel(factor.get_features())
>>> predictions
<xarray.Dataset> Size: 34kB
Dimensions:    (timestamp: 200, symbol: 20)
Coordinates:
  * timestamp  (timestamp) datetime64[s] 2kB 2024-01-01 ... 2024-07-18
  * symbol     (symbol) <U3 240B 'S00' 'S01' 'S02' 'S03' ... 'S17' 'S18' 'S19'
Data variables:
    ret        (timestamp, symbol) float64 32kB -0.002638 -0.01672 ... -0.04101
>>> model.predict(np.zeros((5, 20, 2))).shape
(5, 20, 1)
```

`load()` restores a checkpoint into a model built from the same factors and labels. It first checks the file suffix, then compares the variable names recorded in `config.json` with the model's own.

```python
>>> restored = XGBoostRegressor(config).load(checkpoint)
>>> bool((restored.predict_panel(factor.get_features())["ret"] == predictions["ret"]).all())
True
```

### Metrics

`quantlab.utils.metrics` scores `[T, S]` panels. Only cells where both prediction and target are finite count. Besides MSE, RMSE, MAE and R2 it provides two cross-sectional measures. IC is the Pearson correlation between prediction and target across the symbols of one timestamp, averaged over time. RankIC does the same on the per-timestamp ranks, so it measures ordering and ignores scale. Tree models compute all six on the primary label (the first one) for the train, validation and test segments and write them to the W&B run summary as `train_*`, `val_*` and `test_*`.

```python
>>> from quantlab.utils.metrics import regression_panel_metrics
>>> test = slice("2024-06-01", "2024-07-18")
>>> scores = regression_panel_metrics(
...     predictions["ret"].sel(timestamp=test).values,
...     label.ds["ret"].sel(timestamp=test).values,
... )
>>> {name: round(value, 3) for name, value in scores.items()}
{'mse': 0.003, 'rmse': 0.05, 'mae': 0.04, 'r2': 0.501, 'ic': 0.706, 'rank_ic': 0.686}
```

### The class hierarchy

Every head derives from `BaseModel` through one of two variants. The variants differ in the training framework and in what a subclass must implement; `train`, `train_cv`, `load`, `predict` and `predict_panel` are written once in `BaseModel` and not overridden.

| Class | Framework | Config | Checkpoint | Methods a head implements |
|---|---|---|---|---|
| `DLModel` | torch, epoch loop over `DataLoader` batches | `DLConfig` | `.pth` | `_init_model`, `_init_optim`, `_preprocess`, `_train_one_batch`, `_val_one_batch`, `_test_one_batch` |
| `MLModel` | numpy, the library's own early stopping | `MLConfig` | `.joblib` | `_init_model`, `_preprocess`, `_fit_model`, `_forward` |

Shipped heads: `XGBoostRegressor` (`MLModel`), `MLPRegressor` (a two-hidden-layer perceptron over the flattened cross-section), `RNNRegressor` (a GRU or LSTM tower per label, at least two labels) and `RNNClassifier` (predicts the sign of the return and returns the up probability as a ranking score). See the docstrings of `quantlab/base/model.py` and `quantlab/base/config.py` for the full config fields.

## Common tasks

### Stop training early

With `early_stopping=True`, training stops when the validation loss has not improved for `early_stopping_patience` rounds (boosting rounds for `MLModel` heads, epochs for `DLModel` heads) and the best model is kept. For `XGBoostRegressor` the checkpoint is truncated to the best round. The metric is the RMSE on the validation segment. The booster itself is fit on a pooled concordance correlation loss (`1 - ccc`, see `ccc_objective` in `quantlab/ml_model/xgb.py`); giving `objective` in `hyperparameters` switches back to a built-in xgboost objective.

```python
>>> from dataclasses import replace
>>> stopping = replace(config, early_stopping=True, early_stopping_patience=5,
...                    hyperparameters={"num_boost_round": 500, "max_depth": 3})
>>> stopped = XGBoostRegressor(stopping).collect()
>>> _ = stopped.train()
>>> stopped.model.num_boosted_rounds(), stopped.model.best_iteration
(66, 65)
```

### Cross-validate over walk-forward folds

`train_cv(train_periods, gap_periods)` slides a training window over the timestamps between `start_date` and `end_date`. The test segment is `train_periods // 5` timestamps long and starts `gap_periods` timestamps after the training segment ends; each fold moves forward by one test length. Every fold gets its own checkpoint and its own W&B run, and the return value has one dict per fold with its dates (both ends inclusive), checkpoint path and `test_*` metrics.

```python
>>> results = model.train_cv(train_periods=100, gap_periods=2)
>>> len(results)
4
>>> [(str(r["train_start"])[:10], str(r["test_start"])[:10], str(r["test_end"])[:10]) for r in results]
[('2024-01-01', '2024-04-12', '2024-05-01'), ('2024-01-21', '2024-05-02', '2024-05-21'), ('2024-02-10', '2024-05-22', '2024-06-10'), ('2024-03-01', '2024-06-11', '2024-06-30')]
>>> [round(r["test_rank_ic"], 3) for r in results]
[0.674, 0.689, 0.696, 0.669]
```

All folds share one trial directory. Besides one sub-directory per fold it contains `cv_folds.json`, a manifest with `format_version` and the fold list. A backtester replays a cross-validation run from this file.

```python
>>> from pathlib import Path
>>> trial = Path(results[0]["checkpoint"]).parent.parent
>>> sorted(p.name for p in trial.iterdir())
['XGBoostRegressor_cv_fold_0', 'XGBoostRegressor_cv_fold_1', 'XGBoostRegressor_cv_fold_2', 'XGBoostRegressor_cv_fold_3', 'cv_folds.json']
>>> manifest = json.loads((trial / "cv_folds.json").read_text())
>>> manifest["format_version"], len(manifest["folds"])
(1, 4)
```

`parallel=True` trains the folds concurrently on threads (`njobs` sets the pool size). Each fold works on a deep copy of the model, so memory grows with the number of jobs. Tree libraries already use every core, so set `nthread` in `hyperparameters` to roughly `os.cpu_count() // njobs`.

### Train a torch model

`DLConfig` adds `epochs`, `batch_size`, `lr` and `num_workers` (the `DataLoader` worker count, 4 by default; use 0 for small in-memory data). The network sees every symbol of a bar at once, so it encodes symbol position. `predict_panel` therefore aligns the input to the symbols the model was trained on: a panel that lacks any of them is rejected, and extra symbols are dropped with a warning and receive no prediction.

```python
>>> from quantlab.base.config import DLConfig
>>> from quantlab.dl_model.mlp import MLPRegressor
>>> dl_config = DLConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     epochs=30, batch_size=32, num_workers=0,
...     early_stopping=True, early_stopping_patience=5,
...     hyperparameters={"hidden_size1": 16, "hidden_size2": 8},
... )
>>> mlp = MLPRegressor(dl_config).collect()
>>> mlp_checkpoint = mlp.train()
>>> mlp_checkpoint.name
'MLPRegressor_total.pth'
>>> one_more = factor.get_features().isel(symbol=[0]).assign_coords(symbol=["S99"])
>>> wider = xr.concat([factor.get_features(), one_more], dim="symbol")
>>> mlp.predict_panel(wider).symbol.size
20
```

### Log to Weights & Biases

Each `train()` and each fold of `train_cv()` opens a W&B run named after the experiment inside a project named after the trial directory, with the full config attached. `XGBoostRegressor` logs the per-round training and validation curves and writes the final metrics and per-factor importance to the run summary. `XGBTDRegressor` logs the validation curve of every round (`val-rmse`, or `val-rmse/<label>` with several labels), the selected and trained round counts and the same importance charts, through a callback injected into pytabkit's inner `xgboost.train` call. `RealMLPRegressor` logs every epoch's mean training loss (`train-loss`) and validation error (`val-rmse`) at `step=epoch`, plus `best_val_rmse`, `epochs_trained` and the stopping epoch, through a Lightning callback injected into pytabkit's trainer (`quantlab.ml_model.tabkit.active_callbacks`). `train_cv` opens an extra `<Class>_cv_summary` run with the mean of each `test_*` metric as `cv_mean_test_*`. Torch heads log their metrics every epoch. `WANDB_MODE=disabled` turns all of it off; `WANDB_MODE=offline` writes runs to a local `wandb/` directory that can be synced later with `wandb sync`. Without either setting, `wandb.init` needs a logged-in account.

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
>>> ridge_results = ridge.train_cv(train_periods=100, gap_periods=2)
>>> [round(r["test_rank_ic"], 3) for r in ridge_results]
[0.678, 0.691, 0.697, 0.677]
>>> ridge.model.round(3).ravel().tolist()
[0.05, -0.019, 0.001]
```

A `DLModel` head builds an `nn.Module` in `_init_model` from the panel shape, and the base class moves it to the device, runs the epoch loop and restores the best epoch when early stopping is on. `_train_one_batch` performs one optimizer step on a `[batch, num_symbols, num_features]` batch. `_val_one_batch` must return the validation loss as a scalar, because the base class averages it into the epoch loss that drives early stopping. `_test_one_batch` is called for the test segment each epoch, where a head usually logs metrics. `_preprocess` is shared by training and inference.

```python
>>> import torch, torch.nn as nn
>>> from quantlab.base.model import DLModel
>>> class LinearHead(DLModel):
...     """One nn.Linear applied to every symbol of a bar."""
...     def _init_model(self, num_symbols, num_features, num_labels, hyperparameters):
...         return nn.Linear(num_features, num_labels)
...     def _init_optim(self, model):
...         return torch.optim.Adam(model.parameters(), lr=self.config.lr)
...     def _preprocess(self, data):
...         return torch.nan_to_num(data, nan=0.0)
...     def _train_one_batch(self, epoch, x, y):
...         self.optim.zero_grad()
...         loss = nn.functional.mse_loss(self.model(x), y)
...         loss.backward()
...         self.optim.step()
...         return loss.detach()
...     def _val_one_batch(self, epoch, x, y):
...         return nn.functional.mse_loss(self.model(x), y)
...     def _test_one_batch(self, epoch, x, y):
...         return nn.functional.mse_loss(self.model(x), y)
>>> linear = LinearHead(replace(dl_config, epochs=300, lr=0.05, early_stopping_patience=20, hyperparameters={})).collect()
>>> linear_checkpoint = linear.train()
>>> [round(w, 3) for w in linear.model.weight[0].tolist()]
[0.05, -0.019]
>>> LinearHead.checkpoint_suffix, linear_checkpoint.suffix
('.pth', '.pth')
```

## Notes

Errors below are quoted as raised, with paths shortened to `...`.

A head rejects the wrong config class as the first step of construction.

```text
TypeError: XGBoostRegressor requires a MLConfig, got DLConfig
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

`predict_panel` needs every factor variable, and a torch head needs every training symbol.

```text
ValueError: XGBoostRegressor.predict_panel: features are missing factor variable(s) ['f_b']
ValueError: MLPRegressor.predict_panel: the feature panel lacks 5 of the 20 symbols this model was trained on: ['S15', 'S16', 'S17', 'S18', 'S19']. A DL head encodes symbol position, so it cannot predict without them (...)
```

`train` needs all four window dates. Set them on the config, or through `train_cv`, which sets them per fold.

```text
ValueError: Training and testing start and end dates must be specified.
```

`train_cv` overwrites the four `train_*` and `test_*` dates of the config with those of the last fold, so build a fresh config for a later `train()`. If `train_periods` leaves no room for a test segment, it logs `Skipping fold 0: test set exceeds data range` and returns an empty list (`[]`) without raising. Torch heads return no `test_*` metrics from `train_cv`, so their fold dicts hold only dates and paths and no summary run is opened.

`train()` returns only the checkpoint path. The test metrics of a single run are recorded in the W&B summary; `train_cv` returns them directly for `MLModel` heads.

Checkpoints are pickles (`joblib` for `MLModel` heads, `torch.load` for `DLModel` heads). Load only files you produced or trust.

Progress goes to stderr through `loguru` and `tqdm`. `logger.remove()` silences the log lines. On a machine without CUDA, `DataLoader` may print a `pin_memory` warning; it is harmless.

On macOS the `xgboost` wheel links Homebrew's OpenMP runtime while `torch` bundles its own. A process that uses both can crash or hang. Setting `OMP_NUM_THREADS=1` before the first import of either library avoids this, at the cost of single-threaded tree and torch code. Linux is not affected.

## See also

The factor guide (`docs/factor.md`) explains how factors and labels are produced, and the backtest guide (`docs/backtest.md`) shows how `predict_panel` output and a `cv_folds.json` manifest feed a backtest. The backend guide (`docs/backend.md`) covers the Zarr and xarray storage the panels use. API details are in the docstrings of `quantlab/base/model.py`, `quantlab/base/config.py` (`DLConfig`, `MLConfig`), `quantlab/dl_model/`, `quantlab/ml_model/xgb.py`, `quantlab/ml_model/backend.py` and `quantlab/utils/metrics.py`.
