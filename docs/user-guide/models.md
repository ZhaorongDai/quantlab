# Models

This page explains how quantlab trains return models on factor panels. It
covers the model hierarchy and the available heads, the `MLConfig` and
`DLConfig` configuration objects, training, prediction, evaluation with IC,
RankIC and R2, checkpoints, the purge of label lookahead at every split,
walk-forward cross-validation, and Weights & Biases logging. Read it after
[Factors and labels](factors.md). The backtester that consumes a model's
predictions is described in [Backtesting](backtesting.md).

The runnable script `examples/train_model.py` goes through every step on
this page with synthetic data, offline and on the CPU:

```bash
uv run python examples/train_model.py
```

## What a model does

A model in quantlab is configured with a list of factor objects, its
features, and a list of label objects, its targets. `collect()` asks every
factor and label for its panel and merges them into one `xarray.Dataset`
indexed by `(timestamp, symbol)`. Training cuts that panel into arrays of
shape `[num_times, num_symbols, num_features]` for the inputs and
`[num_times, num_symbols, num_labels]` for the targets, and fits a head on
them. Prediction goes the other way: a feature panel goes in and a panel with
one variable per label comes out. That output is the model's forecast of
future returns, or of the probability of an up move, for every symbol and
bar. The backtester ranks symbols by it.

A label is a `quantlab.label.forward.Forward`: a factor shifted forward so
that its value at bar t describes bars after t. `Return` and `BinaryReturn`
in `quantlab.label.fret` are `Forward` labels. Because a label reads the
future, the model checks the roles when it is built: every object in
`labels` must be a label and none in `factors` may be one. Either mistake
raises `TypeError`, naming the misplaced object.

## The model hierarchy

Every model derives from `quantlab.base.model.BaseModel`, which implements
the public entry points once for all heads: `collect`, `train`, `train_cv`,
`load`, `predict` and `predict_panel`. It also owns the checkpoint layout and
the cross-validation folds. Below it sit two variants, one per kind of
training library:

- `DLModel` is the PyTorch variant. It runs an epoch loop over `DataLoader`
  batches with optional early stopping, rolls back to the best epoch's weights
  and saves `.pth` checkpoints. A head receives whole bars, every symbol at
  once, so it may use a symbol's position. For that reason prediction is
  aligned to the symbols the model was trained on.
- `MLModel` is the NumPy variant for tree models and other libraries that
  run their own training loop. It calls the library once, lets the library do
  its own early stopping, computes metrics on the train, validation and test
  segments, and saves `.joblib` checkpoints. Each `(timestamp, symbol)` cell
  is predicted independently.

The head classes, the concrete models you instantiate, are:

| Class | Module | Variant | Library | Predicts |
|---|---|---|---|---|
| `XGBoostRegressor` | `quantlab.ml_model.xgb` | `MLModel` | xgboost | returns |
| `XGBTDRegressor` | `quantlab.ml_model.xgb_td` | `MLModel` | pytabkit (XGBoost with tuned defaults) | returns |
| `RealMLPRegressor` | `quantlab.ml_model.realmlp` | `MLModel` | pytabkit (RealMLP network) | returns |
| `MLPRegressor` | `quantlab.dl_model.mlp` | `DLModel` | torch | returns |
| `RNNRegressor` | `quantlab.dl_model.rnn` | `DLModel` | torch (GRU or LSTM) | returns |
| `RNNClassifier` | `quantlab.dl_model.rnn_classification` | `DLModel` | torch (GRU or LSTM) | probability of an up move |

`XGBoostRegressor` is the usual starting point. It is fast on the CPU,
handles missing feature values natively and records feature importance.
`XGBTDRegressor` and `RealMLPRegressor` use the tuned default settings from
Holzmüller et al., "Better by Default" (NeurIPS 2024), through the pytabkit
package. They replace missing feature values with 0. The torch heads flatten
or scan the symbol axis of each bar. `RNNRegressor` and `RNNClassifier` need
at least two labels: the first is the primary target and the others are
auxiliary horizons that help train it. `RNNClassifier` turns each return
label into up (1) or down (0) itself, so give it ordinary return labels.

Every head reads its architecture and library settings from
`config.hyperparameters`. The class docstrings list the keys and their
defaults. `XGBoostRegressor`, for example, merges your keys over
`XGBoostRegressor.DEFAULT_PARAMS` and accepts scikit-learn spellings such as
`learning_rate` and `n_estimators`.

## Configure a model

Heads of the `MLModel` variant take a `quantlab.base.config.MLConfig`, and
torch heads take a `DLConfig`. Passing the wrong one raises `TypeError`
before anything else happens. Both share these fields:

- `factors` and `labels`: lists of factor objects and of `Forward` labels.
- `model_save_dir`: the root directory checkpoints are written under.
- `factor_data_strategy` and `label_data_strategy`: `"cal"` computes each
  panel with `compute(start, end)` when you call `collect()`, `"read"` loads
  it from the factor's store with `read(start, end)`, so the store must have
  been written by `build()` over a range that covers the model's.
- `start_date` and `end_date`: the range of data to collect. It is passed to
  every factor and label per request; their configs are left unchanged, so
  one factor object can serve several models.
- `train_start`, `train_end`, `test_start`, `test_end`: the training and
  test windows, both ends inclusive.
- `val_size`: the share of the training window held out, at its end, for
  validation and early stopping. The default is 0.2.
- `early_stopping` and `early_stopping_patience`: stop when the validation
  loss has not improved for that many rounds (ML) or epochs (DL).
- `hyperparameters` and `random_seed`.

`DLConfig` adds the torch training settings `epochs`, `lr`, `batch_size` and
`num_workers`. `MLConfig` has no epochs, because the library decides how long
to train. See the docstrings of both classes for every field.

The validation segment is always the last part of the training window in time,
never a random sample. With daily returns, a random split would put days
next to each other into training and validation and overstate how well the
model generalises.

## Purging the label lookahead

A label's *lookahead* is how many bars past t its value at t reads:
`lookahead_bars()`, which is `delay + span` for a `Forward` label. The
one-day `Return` (`span=1`, `delay=1`) reads the opens of bars t+1 and t+2,
so its lookahead is 2. The label at t is known only once bar t + 2 has
closed, so a label fitted on the last bars of one segment would read bars
of the next.

The model therefore purges every split boundary. With L the largest
`lookahead_bars()` among its labels, each segment followed by another loses
its last L bars:

- `train()` cuts the training window into train and validation by position
  (`val_size`), then drops the last L bars of the train segment and the
  last L bars of the validation segment. The test segment keeps all its
  bars. With `val_size=0`, the train segment is purged against the test
  segment directly.
- `train_cv()` fits each fold the same way, so each fold's training window
  loses its last L bars before its test segment.

Fitting thus loses L bars at every boundary. L follows from the labels and
is not a parameter. The splitting is done by
`quantlab.utils.split.purge_segments`. When the purge leaves no training
bar, `train()` and `train_cv()` raise `ValueError`.

## Train a model

The example builds a two-feature KunQuant factor and a one-day forward-return
label (see [Factors and labels](factors.md)), then configures
`XGBoostRegressor` with bars 0 to 279 for training and bars 280 to 359 for
testing:

```python
from quantlab.base.config import MLConfig
from quantlab.ml_model.xgb import XGBoostRegressor

model = XGBoostRegressor(MLConfig(
    factors=[features],
    labels=[label],
    model_save_dir=str(root / "models"),
    factor_data_strategy="cal",
    label_data_strategy="cal",
    hyperparameters={"num_boost_round": 300, "max_depth": 3, "eta": 0.05, "nthread": 1},
    early_stopping=True,
    early_stopping_patience=20,
    val_size=0.2,
    start_date="2022-01-03", end_date="2023-05-19",
    train_start="2022-01-03", train_end="2023-01-27",
    test_start="2023-01-30", test_end="2023-05-19",
))
model.collect()
checkpoint = model.train()
```

`train()` fits the head on the training window, evaluates it and writes a
checkpoint. It returns the checkpoint's absolute path. With early stopping on,
`XGBoostRegressor` stops after 20 boosting rounds without improvement on the
validation segment and keeps only the trees up to the best round. Here 20 of
the 300 allowed trees were kept.

Beside the checkpoint, `train()` writes `config.json` and `metrics.json`.
`metrics.json` holds the scores of the first label on its raw values for each
segment: `train_*`, `val_*` and `test_*`, each of `loss`, `mse`, `rmse`,
`mae`, `r2`, `ic` and `rank_ic`. These are the values the W&B run summary
receives, with NaN and infinity written as null. A run with `val_size=0` has
no `val_*` keys. Torch heads do not write the file yet.

```python
import json

scores = json.loads((checkpoint.parent / "metrics.json").read_text())
scores["val_ic"], scores["test_ic"]
```

The label's lookahead is 2, so of the 280 training bars the first 224 form
the train segment and the last 56 the validation segment; after the purge
bars 0 to 221 (to 2022-11-08) are fitted, bars 224 to 277 (to 2023-01-25)
validate, and all 80 test bars are scored.

## Predict

There are two ways to predict. `predict_panel` is the one to use in practice.
It takes a panel containing every feature variable and returns a panel with
one variable per label on the same `(timestamp, symbol)` grid:

```python
panel = model.data_backend.get_xarray_dataset(["timestamp", "symbol"])
test = panel.sel(timestamp=slice("2023-01-30", "2023-05-19"))
pred = model.predict_panel(test[model.get_factor_names()])
```

Where every feature of a cell is missing, for example before a stock is
listed, the prediction is NaN rather than whatever the head would produce
from filled-in zeros. For torch heads `predict_panel` also aligns the symbol
axis to the training symbols. It raises `ValueError` when some are missing
and drops, with a warning, symbols the model never saw.

`predict` is the lower-level call on raw arrays. An `MLModel` takes a
`[T, S, F]` NumPy array and returns `[T, S, L]`. The torch heads take and
return tensors in their own layouts, described in each class docstring.
Either call raises `ValueError` if the model has been neither trained nor
loaded.

## Evaluate predictions

`quantlab.utils.metrics` scores a `[T, S]` prediction panel against the
realised label panel. Only cells where both are finite count.

- The IC (information coefficient) is the correlation between prediction and
  realised return across the symbols of one bar, averaged over all bars. It
  measures how well the model orders stocks on a given day, which is what a
  long-short or top-N strategy needs. An IC of a few hundredths is already
  useful on real daily data.
- The RankIC is the same correlation computed on ranks, a per-bar Spearman
  correlation. One extreme return cannot dominate it.
- R2 is the pooled coefficient of determination over all cells,
  `1 - SS_res / SS_tot`. It measures how close the predicted magnitudes are.
  Return forecasts usually have an R2 close to zero even when their IC is
  good, so judge a return model mainly by IC and RankIC.

`regression_panel_metrics(pred, target)` returns all of them (`mse`, `rmse`,
`mae`, `r2`, `ic`, `rank_ic`) in one dict:

```python
from quantlab.utils.metrics import regression_panel_metrics

m = regression_panel_metrics(pred["ret_1"].values, test["ret_1"].values)
```

`MLModel` heads compute the same metrics themselves during `train()` for the
`train`, `val` and `test` segments, under keys such as `test_ic` and
`val_rank_ic`. They are written to the Weights & Biases run summary and
returned per fold by `train_cv`. Torch heads log their own per-epoch metrics
and return none.

## Checkpoints and config.json

`train()` writes into a new trial directory under `model_save_dir`:

```text
models/
  XGBoostRegressor_trial_20260925_175317_715310/
    XGBoostRegressor_total/
      config.json
      XGBoostRegressor_total.joblib
```

The trial directory is named after the class and the time of the run, so
repeated runs never overwrite each other. `config.json` sits next to the
checkpoint. It holds the model's full configuration, including the nested
configurations of every factor, label and dataset. It also carries a
`trained_on` record with the feature names, label names and sorted training
symbols, and, for heads that merge your settings into library defaults, the
`resolved_hyperparameters` actually used.

To use a trained model later, rebuild it from `config.json` and load the
weights:

```python
import json
from quantlab.utils.module import load_model_from_config

saved = json.loads((checkpoint.parent / "config.json").read_text())
reloaded = load_model_from_config(saved).load(checkpoint)
```

`load()` first checks that the file suffix matches the head (`.joblib` or
`.pth`). It then checks that the feature and label names recorded in
`config.json` equal the model's own, name for name and in order. Neither
torch nor xgboost would notice permuted inputs by itself. A mismatch raises
`ValueError`. `.joblib` checkpoints are pickles, so load only files you
trust.

## Walk-forward cross-validation

A single train/test split gives one number from one period. Walk-forward
cross-validation repeats the split over time. It trains on a block of bars,
tests on the bars that follow, slides both forward and repeats. Every test
bar lies after every bar the model trained on, as it would in live trading.
Across folds you see how stable the model's quality is over time.

`train_cv(train_periods, parallel=False, njobs=-1)` lays the folds out over
the bars between `config.start_date` and `config.end_date`. Each fold's
training window is `train_periods` bars and its test segment the next
`train_periods // 5` bars, and the next fold starts that many bars later.
Each window is split into train and validation and purged as described
above, so the last bar fitted in each 200-bar window is its 198th:

```python
folds = model.train_cv(train_periods=200)
```

Each fold trains a fresh model with its own early stopping and writes its own
checkpoint directory, `XGBoostRegressor_cv_fold_{i}/`, inside one trial
directory. `train_cv` returns one dict per fold with the fold's dates,
experiment name, checkpoint path and, for `MLModel` heads, the `train_*`,
`val_*` and `test_*` metrics. The fold's `train_end` is the last bar fitted,
after the purge. The same list is written as `cv_folds.json` in the trial
directory, together with `"format_version": 2` and a `cv_mean` block: the
mean over folds of every metric, keyed `cv_mean_train_ic`,
`cv_mean_test_rank_ic` and so on, plus `cv_n_folds`. Folds whose value is
not finite are left out of a mean. Each fold's `config.json` holds the dates
the fold was configured with, before the purge, so its `train_end` lies L
bars later: 2022-10-07 against the manifest's 2022-10-05 for fold 0 of the
example. `BaseBacktester.run_cv()` reads `cv_folds.json` to backtest each
fold with its own checkpoint on its own test period (see
[Backtesting](backtesting.md)). A fold whose test period would run past the
end of the data is skipped with a warning. `run_cv` refuses a manifest of
format version 1, written before the `cv_mean` block existed; rerun
`train_cv` to replace it.

`train_cv` sets the model's `train_*` and `test_*` dates to each fold in
turn, so afterwards they hold the last fold's dates. `parallel=True` trains
the folds concurrently on copies of the model, using `njobs` threads. The
library also uses several threads per fold, so limit it (`nthread` for
`XGBoostRegressor`, `n_threads` for the pytabkit heads) to roughly the core
count divided by `njobs`.

## Weights & Biases logging

Every training run, including every CV fold, opens a Weights & Biases run with
the model's configuration. `XGBoostRegressor` logs per-round training and
validation curves, writes the final `train_*`, `val_*` and `test_*` metrics
and per-feature importance to the run summary, and adds an importance table
and bar chart. `train_cv` adds a `{class}_cv_summary` run whose summary is the manifest's
`cv_mean` block. The project name is the trial directory's name.

The model layer has no switch to skip W&B, so control it with the standard
environment variables before training starts:

```bash
WANDB_MODE=disabled uv run python my_training_script.py   # no logging at all
WANDB_MODE=offline  uv run python my_training_script.py   # log locally, sync later
```

With `WANDB_MODE=disabled` every W&B call becomes a no-op: nothing is sent,
no login is needed and no files are written. The test suite and both example
scripts use this mode, and `WANDB_SILENT=true` also silences W&B's console
messages. In a script you can set the variables at the top, before the model
layer is imported:

```python
import os
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
```

## Output of the example

This is the output of `uv run python examples/train_model.py` (a Zarr
warning printed on standard error is left out). The panel has a planted
one-day reversal, which the model finds, so the test IC is about 0.25 and
stable across folds. The line `metrics.json val` reads the validation scores
of the single run back from its `metrics.json`. Between each fold's `train_end` and `test_start` lie
the two purged bars.

```text
features: ['past_ret_1', 'ma_dev_5'] label: ['ret_1']
checkpoint: models/XGBoostRegressor_trial_20260928_084754_029062/XGBoostRegressor_total/XGBoostRegressor_total.joblib
trees kept by early stopping: 20
metrics.json val       IC=+0.285  RankIC=+0.259  R2=+0.084
prediction panel: {'timestamp': 80, 'symbol': 16} ['ret_1']
test window            IC=+0.263  RankIC=+0.248  R2=+0.068
trained_on symbols: 16 resolved eta: 0.05
reloaded model predicts the same values: True
fold 0: train 2022-01-03..2022-10-05  test 2022-10-10..2022-12-02  IC=+0.242  RankIC=+0.226
fold 1: train 2022-02-28..2022-11-30  test 2022-12-05..2023-01-27  IC=+0.277  RankIC=+0.262
fold 2: train 2022-04-25..2023-01-25  test 2023-01-30..2023-03-24  IC=+0.254  RankIC=+0.223
fold 3: train 2022-06-20..2023-03-22  test 2023-03-27..2023-05-19  IC=+0.264  RankIC=+0.237
cv_folds.json: format_version 2 with 4 folds
keys of one fold: ['checkpoint', 'experiment_name', 'fold', 'test_end', 'test_ic', 'test_loss', 'test_mae', 'test_mse', 'test_r2', 'test_rank_ic', 'test_rmse', 'test_start', 'train_end', 'train_ic', 'train_loss', 'train_mae', 'train_mse', 'train_r2', 'train_rank_ic', 'train_rmse', 'train_start', 'val_ic', 'val_loss', 'val_mae', 'val_mse', 'val_r2', 'val_rank_ic', 'val_rmse']
mean over folds: train IC 0.31 val IC 0.25 test IC 0.259
```

## Train a torch head

Torch heads are configured the same way with a `DLConfig`. This snippet
trains `MLPRegressor` on the same factor and label objects as above. It runs
on the CPU in a few seconds and uses the GPU automatically when CUDA is
available:

```python
from quantlab.base.config import DLConfig
from quantlab.dl_model.mlp import MLPRegressor

mlp = MLPRegressor(DLConfig(
    factors=[features], labels=[label],
    model_save_dir=str(root / "models"),
    factor_data_strategy="cal", label_data_strategy="cal",
    start_date="2022-01-03", end_date="2023-05-19",
    train_start="2022-01-03", train_end="2023-01-27",
    test_start="2023-01-30", test_end="2023-05-19",
    hyperparameters={"hidden_size1": 32, "hidden_size2": 16},
    epochs=20, batch_size=32, num_workers=0, lr=1e-3,
    early_stopping=True, early_stopping_patience=5,
))
print(mlp.collect().train().name)
```

```text
MLPRegressor_total.pth
```

A batch here is a set of bars, each with all its symbols. `num_workers=0`
loads batches in the main process, which is the simplest choice on small
data.

## Things to watch

- The model's feature list follows each factor's pinned
  `config.factor_names`, in the order given. `Alpha158Stock` pinned to three
  features trains on those three. A factor left unpinned contributes every
  name its class can produce (`_get_factor_names()`).
- On macOS, the xgboost and torch wheels ship different OpenMP runtimes that
  clash in one process. Because `quantlab.base.model` imports torch, any
  script that trains an `MLModel` head is affected. Set `OMP_NUM_THREADS=1`
  before anything imports torch or xgboost, as the example scripts do. Linux
  is not affected.
- XGBoost and pytabkit use every core by default. On a small panel or a busy
  machine that can make training much slower than with a single thread. Set
  `nthread` (xgboost) or `n_threads` (pytabkit) in `hyperparameters`.
- Each `train()` needs all four `train_*` and `test_*` dates. A `val_size`
  or a purge that leaves no training bars raises `ValueError`.

## See also

- [Factors and labels](factors.md) for building the inputs.
- [Backtesting](backtesting.md) for turning predictions into target weights
  and replaying a CV run.
- The docstrings of `quantlab.base.model.BaseModel`, `DLModel` and `MLModel`
  for every method, and of each head class for its hyperparameters.
- [Extending quantlab](../developer-guide/extending.md) for writing a new
  model head.
