---
task: 260907-fl6 (+ 260907-1du couplings)
reviewed: 2026-09-07T00:00:00Z
depth: standard (+ deep on the five named cross-file couplings)
files_reviewed: 15
files_reviewed_list:
  - base/backend.py
  - base/config.py
  - base/data.py
  - base/factor.py
  - base/model.py
  - base/pageledger.py
  - dataset/backend.py
  - dataset/spot.py
  - dl_model/mlp.py
  - dl_model/rnn.py
  - dl_model/rnn_classification.py
  - ml_model/backend.py
  - my_ops/preprocess.py
  - train_model.py
  - utils/nautilus.py
also_read_for_context:
  - dataset/cleaning.py
  - tests/test_model_layer.py
  - tests/test_dl_models.py
  - tests/test_cleaning.py
  - tests/test_backend_indexes.py
findings:
  critical: 2
  warning: 7
  info: 6
  total: 15
status: issues_found
---

# quick 260907-fl6: Code Review Report

**Reviewed:** 2026-09-07
**Depth:** standard, with deep cross-file tracing on the five named couplings
**Files reviewed:** 15 (diff `4d94070..HEAD`), plus `dataset/cleaning.py` and the
test files for couplings 4 and 5 (those landed in `260907-1du`, before `4d94070`)
**Status:** issues_found
**Suite state at review time:** `478 passed` on `~/.venv/bin/python -m pytest -q`
(reproduced locally, 27.5s)

## Summary

The five named couplings were traced individually and **four of the five hold up**:

- **Coupling 1 (`to_tensor` ↔ `train_model.py`) — column order: CORRECT.** Both
  sides now go through `BaseModel.to_tensor` and the last axis follows the
  caller's declared list. `grep` for `to_dataarray` / `from_numpy` outside
  `tests/` confirms no third hand-rolled conversion survives. The `.fillna(0)`
  in `train_model.py:73` is redundant (every head's `_preprocess` already runs
  `torch.nan_to_num`) but harmless. **However**, `to_tensor` also silently
  re-sorts the `symbol` and `timestamp` axes and returns a bare tensor —
  see WR-05.
- **Coupling 2 (`get_xarray_dataset(indexes)` ↔ callers): CORRECT.** Every call
  site in the repo was enumerated and checked. The market panel, the constituent
  panel (`base/constituent.py:288`) and the collected model panel all carry
  exactly `(timestamp, symbol)`, so no call site loses a variable or a dim. The
  `["timestamp"]` narrowing behaves as designed (verified empirically). The one
  site passing a different axis order, `base/data.py:290 _get_symbols(["symbol",
  "timestamp"])`, only reads `.symbol.values` so the added transpose is inert.
- **Coupling 3 (refit-optimizer cache ↔ model lifecycle): CORRECT.** `load()`,
  `_init_model_and_optim()` and the `.joblib` branch all rebind `self.model` to
  a fresh object, so identity keying invalidates. `.to(device)` returns the same
  module. `copy.deepcopy(self)` in `_train_fold_with_config` copies model and
  cached key through the same memo, so identity survives the copy. `self.optim =
  None` cannot resurrect it — different attribute. One residual gap: WR-06.
- **Coupling 4 (one-shot pending-panel handoff): CORRECT.** The two class
  attributes are only ever written through `self.`, so no cross-instance leak;
  `from_raw_data()` clears both before the guard; the guard requires panel
  identity AND a matching date window, and every intervening mutation
  (`read()`, `_filter()`, `from_raw_data_chunked()`) rebinds `data_backend.data`
  and so fails identity in the safe direction.
- **Coupling 5 (`validate_schema`'s structural mask ↔ `_clean()`): DEFECTIVE.**
  See BL-01. A genuinely all-null *extra* column warns on both paths, as
  claimed — but only while at least one required column has data. When the
  required columns are themselves all null, the mask swallows *every* warning.

Beyond the couplings, the strongest finding is BL-02: the newly-consolidated
`to_tensor` seam does no dtype normalization, and **every one of the 47 new tests
builds its synthetic panel with `.astype("float32")`**, so the suite is
structurally blind to the float64 panels that both real data paths produce. This
is the same shape as the escaped batch-1 regression the task write-up describes.

## Critical Issues

### BL-01: `validate_schema` goes completely silent when every required column is null

**File:** `dataset/cleaning.py:189-233` (mask built at 189-196, consumed at 209-233)
**Coupling:** #5

**Issue.** The structural mask is a logical AND over `isnull()` of every column in
`required_columns`. If the vendor returned the schema but no values — an empty
response written to a store, a mis-parsed CSV, a fully-failed backfill — every
required column is null everywhere, so the mask is `True` everywhere, and the
per-column loop's `column.isnull() & ~structural_mask` is `False` everywhere.
**Zero warnings are emitted, for any column.** The structural report at line 199
is `logger.info`, not `logger.warning`, so it does not surface either.

Reproduced (5 required columns all-NaN, plus a `vwap` with real data and one real
hole):

```
A) all-required-null warnings: []
```

The docstring at `dataset/cleaning.py:158-160` claims "a `vwap` the vendor
withheld entirely still warns" — in this configuration it does not, and neither
does anything else. `validate_schema` is the only null gate on the ingest path
(`clean_market_data` → `from_raw_data`), and it fails open in exactly the case
where the data is worst. `tests/test_cleaning.py:299-340` covers "sparse panel"
and "one withheld column" but never the all-required-null panel, so the gap is
green.

**Why it matters.** A totally empty ingest flows silently into Zarr, then into
factors, then into a model whose `_preprocess` turns every NaN into `0.0`. The
first visible symptom is a model that trains on zeros.

**Fix.** Treat a fully-structural panel as the anomaly it is, and never let the
mask suppress 100% of the grid:

```python
if structural_mask is not None:
    structural_cells = int(structural_mask.sum().item())
    total_cells = int(structural_mask.size)
    if total_cells > 0 and structural_cells == total_cells:
        raise ValueError(
            f"validate_schema: EVERY required column is null on every "
            f"(timestamp, symbol) cell ({total_cells} cells). This is not "
            f"the dense panel's structural sparsity -- it is an empty "
            f"ingest. Required columns: {list(required_columns)}."
        )
    if structural_cells > 0:
        logger.info(...)  # unchanged
```

(If raising is judged too strong for D-07's flag-don't-delete rule, at minimum
promote it to `logger.warning` **and** disable the mask for that call so the
per-column loop reports real counts.) Add a `tests/test_cleaning.py` case
mirroring `test_a_genuinely_withheld_column_still_warns` but with
`_SPARSE_REQUIRED` replaced by an all-NaN required block; it must be red against
today's code.

### BL-02: `to_tensor` does no dtype normalization — every real (float64) panel kills `train()`

**File:** `base/model.py:274-300` (`to_tensor`), reached from `base/model.py:410-419`
**Related:** `dl_model/rnn.py:339` `_preprocess`, `dl_model/rnn_classification.py`,
`dl_model/mlp.py:151`

**Issue.** `to_tensor` ends in `torch.from_numpy(...values)`, which preserves the
panel's numpy dtype. None of the three `_preprocess` implementations casts —
they all do `torch.nan_to_num(data, nan=0.0)` only. Every torch module in
`dl_model/` is created with default float32 parameters. Measured, on real
one-epoch runs with an otherwise-identical float64 panel:

```
RNNRegressor  float32: TRAIN OK
RNNRegressor  float64: ValueError: RNN input dtype (torch.float64) does not
                       match weight dtype (torch.float32).
RNNClassifier float32: TRAIN OK
RNNClassifier float64: ValueError: RNN input dtype (torch.float64) ...
```

float64 is not hypothetical — it is what both non-KunQuant data paths produce:

```
polars->xarray dtype: {'Close': 'float64'}     # FactorPolars / PlBackend
pandas->xarray dtype: {'close': 'float64'}     # StockDataset / Tiingo parquet
```

CLAUDE.md names Polars as a first-class second factor backend, so
`DLConfig(factors=[kunquant_factor, polars_factor])` — the exact
interchangeability contract Phase 03 D-03 exists to guarantee — cannot train.

**Why the suite cannot see it.** `tests/test_model_layer.py:106` and
`tests/test_dl_models.py:90` both build panels with `.astype("float32")`, and
`RecordingRegressor._preprocess` (`tests/test_model_layer.py:166`) adds a
`.float()` the shipped heads do not have. Reverting the whole dtype question
changes nothing about the 47 new tests — this is a base-class seam whose blast
radius reaches three concrete heads that no test exercises at this dtype.

**Fix.** Normalize at the one seam that now owns the conversion:

```python
def to_tensor(self, data: xr.Dataset, variables: list[str]) -> torch.Tensor:
    values = (
        data[variables]
        .to_dataarray()
        .sortby(["timestamp", "symbol"])
        .sel(variable=variables)
        .transpose("timestamp", "symbol", "variable")
        .values
    )
    # torch modules are float32; a float64 panel (the polars and pandas
    # paths both produce one) otherwise dies inside the first forward pass.
    if values.dtype == np.float64:
        values = values.astype(np.float32)
    return torch.from_numpy(values)
```

Then add a parametrized `@pytest.mark.parametrize("dtype", ["float32",
"float64"])` over `FakePanel` in `tests/test_dl_models.py` covering
`MLPRegressor.train()`, `RNNRegressor.train()` and `RNNClassifier.train()`.

## Warnings

### WR-01: `RNNRegressor._init_model` accepts `hyperparameters` and ignores it entirely

**File:** `dl_model/rnn.py:223-238`

**Issue.** The signature takes `hyperparameters: dict` and the body hardcodes
`hidden_sizes=[256, 128, 64]`, `dropout_rates=[0.1, 0.1, 0.1]`,
`hidden_sizes_linear=[32]`, `model_type="gru"`. `config.hyperparameters` is
silently discarded. This is the exact "declared, accepted, never referenced"
pattern this session went out of its way to delete three times over
(`_train_dl(backtest=...)`, `get_crypot_currency(name=...)`,
`XrBackend.get_xarray_dataset(indexes)`), and `MLPRegressor._init_model` was
changed **in this same diff** (`dl_model/mlp.py:64-83`) to honour it.

The gap is invisible for a specific reason: `tests/test_dl_models.py:154` feeds
`RNNRegressor` the MLP-shaped `{"hidden_size1": 16, "hidden_size2": 8}` and the
tests pass **because the argument is ignored**. Had it been honoured, they would
have raised. That is a test green for the wrong reason.

There are now three mutually incompatible contracts for the same abstract hook:
`MLPRegressor` uses `.get()` with defaults, `RNNClassifier` uses `[...]`
(`KeyError` on a missing key), `RNNRegressor` ignores it.

**Fix.** Mirror `RNNClassifier._init_model`'s reading of the dict, with
`.get(..., <today's hardcoded value>)` defaults so no existing config changes
shape:

```python
return ModelRCrypto(
    input_size=num_features,
    num_labels=num_labels,
    hidden_sizes=hyperparameters.get("hidden_sizes", [256, 128, 64]),
    dropout_rates=hyperparameters.get("dropout_rates", [0.1, 0.1, 0.1]),
    hidden_sizes_linear=hyperparameters.get("hidden_sizes_linear", [32]),
    dropout_rates_linear=hyperparameters.get("dropout_rates_linear", [0.1]),
    model_type=hyperparameters.get("model_type", "gru"),
)
```

and add a test asserting a non-default `hidden_sizes` reaches the built module.

### WR-02: early stopping tracks `best_loss` but the checkpoint saves the *worst* weights

**File:** `base/model.py:464-510` (loop), `base/model.py:514-519` (`_save_model`)

**Issue.** The rewritten block computes a per-epoch `epoch_val_loss` and keeps
`best_loss`, but nothing ever snapshots the corresponding `state_dict`.
`_save_model` runs once, after the loop, on whatever weights the last executed
epoch produced. When early stopping fires, that epoch is by construction the
`patience`-th consecutive epoch of *no improvement* — i.e. the mechanism's whole
purpose (keep the best model) is inverted into "keep the worst one we waited
for". `best_loss` is now a write-only variable.

This block was rewritten in this diff (the per-epoch aggregation and the
unconditional initialization), so it is fair game; leaving `best_loss` computed
but unused is dead state that reads like a feature.

**Fix.** Snapshot on improvement and restore before saving:

```python
best_state = None
...
if epoch_val_loss < best_loss:
    best_loss = epoch_val_loss
    counter = 0
    best_state = {
        k: v.detach().cpu().clone()
        for k, v in self.model.state_dict().items()
    }
...
# after the loop, before _save_model:
if self.config.early_stopping and best_state is not None:
    self.model.load_state_dict(best_state)
```

Test: a `_val_one_batch` returning a scripted descending-then-ascending sequence,
asserting the saved `state_dict` matches the minimum-loss epoch.

### WR-03: `MLPRegressor._val_one_batch` (new code) logs to W&B unguarded

**File:** `dl_model/mlp.py:149`

**Issue.** `BaseModel.__init__` sets `self._wandb_recorder = None`. The two RNN
heads guard every log with `if self._wandb_recorder:`
(`dl_model/rnn.py:220-221, 280-281, 334-335`). The `_val_one_batch` added in
this diff does not, so any path that reaches the epoch loop without
`_init_wandb()` dies with `AttributeError: 'NoneType' object has no attribute
'log'`. `_train_dl` is callable directly — `tests/test_model_layer.py:551` does
exactly that — and `train()` is only one of its entry points.

The new test `test_mlp_val_one_batch_returns_a_floatable_loss`
(`tests/test_dl_models.py:249`) papers over this by calling
`model._init_wandb(...)` explicitly, which is precisely the call the guard exists
to make optional.

**Fix.** Match the sibling heads in the three MLP log sites (`dl_model/mlp.py:62`,
`:113`, `:149`):

```python
if self._wandb_recorder:
    self._wandb_recorder.log(metrics, step=epoch)
```

### WR-04: `BaseDataset.time_interval` raises a bare `IndexError` on a one-timestamp panel

**File:** `base/data.py:114-141` (failing expression at `:136-140`)

**Issue.** `.diff(dim="timestamp")` on a length-1 axis yields an empty series;
`.mode()` on it is empty; `.values[0]` then raises. Reproduced:

```
time_interval -> IndexError: index 0 is out of bounds for axis 0 with size 0
```

This is the same `.values[0]`-on-an-empty-result shape as the `num_null` defect
this session fixed nine lines earlier in `base/model.py`, and this expression was
rewritten in this diff specifically to make the property reachable. It is called
from `dataset/spot.py:_xr_to_bars`, so a single-bar window (a partial ingest, a
one-day backfill) surfaces as an unattributable `IndexError` three frames deep.

**Fix.**

```python
intervals = timestamps.diff(dim="timestamp").to_series().mode()
if intervals.empty:
    raise ValueError(
        f"{self.class_name}.time_interval: need at least 2 timestamps to "
        f"infer a bar interval, the panel has "
        f"{timestamps.size}."
    )
return intervals.values[0]
```

### WR-05: `to_tensor` silently re-sorts both axes and returns a tensor with no axis metadata

**File:** `base/model.py:274-300`; consumer at `train_model.py:73-95`

**Issue.** `to_tensor` applies `.sortby(["timestamp", "symbol"])` and hands back a
bare `torch.Tensor`. The caller has no way to learn what row `i` or column `j`
corresponds to. Verified: a panel whose symbol axis is `["ZZZ", "AAA", "MMM"]`
comes back as `["AAA", "MMM", "ZZZ"]` with no signal.

Inside `_train_dl` this is safe (x and y are sorted identically and
`collect()` already sorted the panel). At the `train_model.py` inference site it
is not: `predicts` is indexed by `to_tensor`'s *sorted* timestamp axis of the
model's collected panel, while `timestamp` at `train_model.py:82` comes from a
**different object** — the dataset's own backend panel — and is never sorted or
intersected against it. `pd.Series(signals, index=timestamp)` at `:88` then
either raises on a length mismatch or, if the lengths happen to agree, pairs each
prediction with the wrong bar. The stated purpose of routing inference through
`to_tensor` was to end silent misalignment; the remaining misalignment is one
level up.

**Fix.** Either return the axes alongside the tensor, or (cheaper) align
explicitly at the call site:

```python
panel = data[factors].sortby(["timestamp", "symbol"])
tensor = model.to_tensor(panel, factors)
pred_index = pd.DatetimeIndex(panel["timestamp"].values)
...
price = price.sel(timestamp=pred_index)      # fails loudly on a mismatch
signals = pd.Series(pred_class.numpy().reshape(-1), index=pred_index)
```

and add an assertion in `to_tensor` (or a docstring line) that the returned axes
are the *sorted* ones.

### WR-06: `_get_refit_optim` reads a field only `DLConfig` has, and pins a superseded model in memory

**File:** `base/model.py:835-871` (`key` at `:855`, cache write at `:868`)

**Issue (a).** `BaseModel` is typed `config: DLConfig | MLConfig`, and `lr_refit`
was added only to `DLConfig` (`base/config.py:258-268`). `_get_refit_optim` lives
on the base class, so on an `MLConfig`-backed model it raises the same
`AttributeError: 'MLConfig' object has no attribute 'lr_refit'` that the fix
was written to remove — just moved one class up. `MLConfig` is scaffolding today
(`_auto_train` raises `NotImplementedError`), so this is a latent trap, not a
live break.

**Issue (b).** `self._refit_optim_cache` holds a strong reference to the *old*
`nn.Module` and its AdamW state. After `_init_model_and_optim()` or `load()`, the
superseded model stays resident until the next `_get_refit_optim()` call, which
directly undercuts the memory argument the diff makes three lines earlier for
`self.optim = None` (`base/model.py:521-528`).

**Fix.** Add `lr_refit: float = 0.0` to `MLConfig` as well (it costs nothing and
keeps the base-class method honest), and clear the cache eagerly where the model
is rebound:

```python
def _init_model_and_optim(self):
    self._refit_optim_cache = None      # drop the superseded model + its state
    ...
# and the same first line in load()
```

The identity keying stays as the correctness guarantee; the explicit clear is the
memory guarantee.

### WR-07: `MLPRegressor`'s inference contract contradicts its training contract

**File:** `dl_model/mlp.py:32-41` (train reshapes) vs `base/model.py:302-318`
(`_predict_nn` does not)

**Issue.** `_train_one_batch` / `_test_one_batch` / `_val_one_batch` all reshape
`[T, S, F] -> [T, S*F]` before calling `self.model`, because `MLP.forward` is a
plain `nn.Linear` stack. `_predict_nn` passes its input straight through. So
`model.predict(model.to_tensor(panel, factors))` — the documented inference path,
and literally what `train_model.py:73-74` does for the RNN head — raises a shape
error for `MLPRegressor`.

`tests/test_dl_models.py:212-234` **documents** this rather than fixing it
("Left as-is deliberately") and pre-flattens its input to make the test pass.
A test that encodes the inconsistency is not coverage of it.

**Fix.** Move the reshape into the head so all four entry points agree:

```python
def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
    data = torch.nan_to_num(data, nan=0.0)
    if data.dim() == 3:                      # [T, S, F] -> [T, S*F]
        data = data.reshape(data.shape[0], -1)
    return data
```

and drop the now-redundant reshapes from the three `_*_one_batch` methods.

## Info

### IN-01: `to_tensor` never uses `self`

**File:** `base/model.py:274-276`
It is a pure function of `(data, variables)`. Making it a `@staticmethod` makes
that contract explicit and lets callers use it without a live model.

### IN-02: `validate_schema` compares dim *order*, not the dim *set*

**File:** `dataset/cleaning.py:211-213`
`tuple(column.dims) == tuple(structural_mask.dims)` means a column stored as
`(symbol, timestamp)` silently drops to whole-column counting. Verified:

```
B) flipped-dim column warnings: ["... its dims ('symbol', 'timestamp') differ
   from the required-column grid, so the whole column is counted ..."]
```

The message is honest, so this is not a silent failure — but the correct handling
is a set comparison plus `column.transpose(*structural_mask.dims)`.

### IN-03: `Factor.save`'s wrapped error hardcodes `mode="a"` and matches on an upstream English string

**File:** `base/factor.py:150-171`
The guard is `"already exists with different dimension sizes" not in str(exc)` —
a zarr wording change silently reverts to the opaque original error (safe
direction, but the improvement evaporates without notice). The message then names
`save(mode="a")` unconditionally even though `mode` is a parameter. Use
`f'{self.class_name}.save(mode="{mode}")'`, and consider matching on the
exception type plus a looser token (`"dimension sizes"`).

### IN-04: `_init_model_and_optim` can leave `self.optim` as `None` after the `del` → `= None` change

**File:** `base/model.py:326-335` with `base/model.py:528`
`self.optim` is only assigned when `_init_optim` returns non-`None`. Before this
diff a second `train()` on such a head hit `AttributeError: 'BaseModel' object
has no attribute 'optim'`; now it hits `AttributeError: 'NoneType' object has no
attribute 'zero_grad'` inside the batch loop — later and less legible. Assign
unconditionally (`self.optim = optim`) or raise in `_init_model_and_optim` when a
DL head returns no optimizer.

### IN-05: `MLPRegressor._train_one_batch` logs post-update metrics from a second forward pass

**File:** `dl_model/mlp.py:49-51`
`pred = self.model(x)` runs *after* `self.optim.step()`, so `train_R²` /
`train_MSE` describe the updated weights while `train_loss` describes the
pre-update ones — the two logged numbers are not from the same model. Reuse
`outputs.detach()` instead; it is also one forward pass cheaper.

### IN-06: dangling `last_position()` references outside the code

**File:** `base/pageledger.py:303-315`; `example/pageledger.md:369`
The deletion is correct (zero call sites), and `describe()`'s docstring
explains it. But `example/pageledger.md:369` still names `last_position()` in a
runnable-looking snippet. Since `example/` is treated as source in this repo,
that line should be updated or removed.

## What was checked and found clean

Stated plainly rather than padded:

- No stale references to any renamed or deleted symbol anywhere in `*.py`,
  `*.md` or `*.ipynb` outside `.planning/` (`_*_one_epoch`, `_get_*_batch`,
  `get_crypot_currency`, `WindowedRobustStandardization`, `last_position`,
  `from dl_model.rnn import RNNClassifier`).
- No hardcoded credentials, `eval`, `exec`, shell interpolation, path traversal,
  unsafe deserialization or insecure randomness introduced by this diff.
  `train_model.py:61` reads its checkpoint path from
  `os.environ.get("QUANTLAB_CHECKPOINT_PATH", ...)`, consistent with CLAUDE.md.
- `XrBackend.get_xarray_dataset`'s `indexes=None` path still returns the held
  object by identity, which several dozen call sites depend on (verified).
- The `PlBackend.get_xarray_dataset(indexes=None)` rejection is correct: a
  `LazyFrame` genuinely has no dims to fall back on.
- `MlBackend`'s three `-> Self` returns now match `ModelBackend`'s ABC signatures
  including `**kwargs` (`base/backend.py:128-135`).
- `WindowedZScore`'s docstring correction changes no computed value; deleting the
  docstring's fillna claim rather than adding a fillna is the right call under
  the project's factor-normalization ruling.
- `my_ops/preprocess.py` and `base/pageledger.py` have no imports orphaned by the
  deletions.
- The `_do_vecbt` / `_vecbt` / `_train_dl(backtest=...)` "fail honestly" changes
  are correct and have no live callers, so the new `NotImplementedError`s cannot
  regress anything.

---

_Reviewed: 2026-09-07_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard + deep on the five named couplings_
