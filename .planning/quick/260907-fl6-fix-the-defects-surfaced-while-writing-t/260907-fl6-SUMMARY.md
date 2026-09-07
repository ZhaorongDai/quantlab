---
task: 260907-fl6
title: Fix the defects surfaced while writing the model-layer docs
status: complete
batch: 3 of 3 (all done)
scope: |
  batch 1 -- base/model.py (+ tests/test_model_layer.py, example/model.md, train_model.py)
  batch 2 -- dl_model/, dataset/backend.py, base/{backend,data,factor,config,model,pageledger}.py,
             my_ops/preprocess.py, ml_model/backend.py (+ 4 new test files, example/*, CLAUDE.md, README.md)
  batch 3 -- naming (base/model.py, dl_model/*), base/model.py:num_null, the vecbt skeleton
             (base/model.py + dl_model/rnn_classification.py), utils/nautilus.py + dataset/spot.py
             (+ tests/test_model_layer.py, tests/test_dl_models.py, tests/test_spot_dataset.py,
             tests/test_factor_hierarchy.py, example/{model,factor,README}.md, CLAUDE.md)
date: 2026-09-07
tests_before: 425 passed
tests_after: 478 passed
commits: 8 (batch 1) + 9 (batch 2) + 6 (batch 3) = 23
---

# 260907-fl6 — batch 1 of 2 (model layer)

**BATCH 1 OF 2.** Batch 1 covered `base/model.py` only. Batch 2 is written up
in the second half of this file, below the batch-1 section.

## Result

All six defects (A, B, C, D, L1, L2) were re-verified against the real code,
all six were genuine, all six are fixed, and each is locked by a test that was
**observed RED before its fix**.

Suite: **425 passed → 431 passed** (the 6 new tests; nothing else changed).

## The RED run (real output, before any fix)

`uv run pytest tests/test_model_layer.py -q --tb=line`:

```
E   UnboundLocalError: cannot access local variable 'early_stopping' where it is not associated with a value
/Users/daizhaorong/projects/quantlab/base/model.py:406: UnboundLocalError: cannot access local variable 'early_stopping' where it is not associated with a value
E   assert [0] == [0, 1, 2, 3]
/Users/daizhaorong/projects/quantlab/tests/test_model_layer.py:290: assert [0] == [0, 1, 2, 3]
E   AssertionError: x last axis is not in declared factor order ['zeta', 'alpha', 'mid']
    assert [2.0, 3.0, 1.0] == [1.0, 2.0, 3.0]
/Users/daizhaorong/projects/quantlab/tests/test_model_layer.py:328: AssertionError: x last axis is not in declared factor order ['zeta', 'alpha', 'mid']
E   AssertionError: predict() left the module in training mode
    assert True is False
/Users/daizhaorong/projects/quantlab/tests/test_model_layer.py:366: AssertionError: predict() left the module in training mode
E   AssertionError: assert (80 + 19) == 100
/Users/daizhaorong/projects/quantlab/tests/test_model_layer.py:398: AssertionError: assert (80 + 19) == 100
E   AttributeError: 'RecordingRegressor' object has no attribute 'model'
/Users/daizhaorong/projects/quantlab/tests/test_model_layer.py:424: AttributeError: 'RecordingRegressor' object has no attribute 'model'

6 failed, 5 warnings in 3.58s
```

That run is commit `1e70566`, committed red on purpose so the failure is in
history rather than only in this summary.

## Per defect

### A — `early_stopping=False` raised `UnboundLocalError`  (`dd395dc`)

Confirmed exactly as described. `best_loss` / `early_stopping` / `patience` /
`counter` were bound inside `if self.config.early_stopping:`; `if
early_stopping: break` at the end of the epoch loop read the name
unconditionally, so the crash landed at the end of epoch 0.

Fix: initialise all four unconditionally before the loop.

Test: `test_early_stopping_disabled_runs_all_epochs` — trains 3 epochs with
`early_stopping=False` and asserts all three ran.

### B — patience counted validation BATCHES  (`9b22f02`)

Confirmed. `counter += 1` was inside `for x_batch, y_batch in val_loader:`.

Fix: the validation loop now only accumulates a **sample-weighted** loss sum;
the comparison against `best_loss` runs once per epoch after the loop.
`val_sample_count > 0` guards an empty validation loader from reading as a
perfect 0.0 improvement.

Test: `test_early_stopping_patience_counts_epochs_not_batches` — 4 validation
batches per epoch, `patience=3`, constant validation loss. Batch-counting stops
after 1 epoch (`[0]`); epoch-counting stops after 4 (`[0,1,2,3]`). The test
also asserts `val_batches_in_first_epoch > 1` so it cannot silently degrade
into a one-batch-per-epoch test that could not tell the two apart.

Behavioural note for callers: `_val_one_epoch`'s return value is now passed
through `float()`. It must be a 0-d tensor or a number — which was always the
declared contract, but was previously only ever compared with `<`.

### C — last axis was alphabetical  (`905de76`)

Confirmed: `.sortby(["timestamp", "symbol", "variable"])` sorted by variable
name. Measured `['alpha','mid','zeta']` where the caller declared
`['zeta','alpha','mid']`.

Fix: extracted the conversion into **`BaseModel.to_tensor(data, variables)`**,
which sorts only `timestamp`/`symbol` (cross-panel alignment still matters) and
then pins the last axis with `.sel(variable=variables)`.

Per the user's locked decision: **no backward compatibility** — no version
stamp, no migration path, no compat branch. Old checkpoints are invalidated.

Test: `test_tensor_variable_axis_follows_declared_order`. Both name lists are
deliberately non-alphabetical — factors `zeta, alpha, mid` and labels
`ret_30, ret_60, ret_120` — so alphabetical output cannot pass by accident. The
panels are filled with a distinct constant per variable, so the value in column
*i* identifies which variable landed there regardless of row order or
`shuffle=True`.

### D — `predict()` ran in training mode  (`830c573`)

Confirmed: no `model.eval()`, no `torch.no_grad()`; `model.training == True`
after `load()` and output `requires_grad == True`.

Fix: `_predict_nn` calls `self.model.eval()` and wraps the forward in
`torch.no_grad()`. The module is deliberately **left** in eval mode; `_train_dl`
calls `self.model.train()` at the top of every epoch, so resuming training is
unaffected.

Test: `test_predict_runs_in_eval_mode_without_grad` — dropout 0.9, asserts
`training is False`, `requires_grad is False`, and that two calls on the same
input are bit-identical.

### L1 — validation split dropped a row  (`9d8c291`)

Confirmed: measured 80 train + 19 val out of 100 training timestamps. Slice
changed to `train_split:`.

Test: `test_val_split_keeps_every_training_row`.

### L2 — `del self.model` after training  (`6adc356`)

Confirmed: `AttributeError: 'RecordingRegressor' object has no attribute
'model'` after `train()`.

Fix: keep the model, drop only the optimizer (`self.optim = None`). Adam's
moment buffers are ~2x the parameter count and are the memory the `del` was
actually buying back. Verified every read of `self.optim` in the repo is inside
a `_train_one_epoch` (`base/model.py:311`, `dl_model/mlp.py`, `dl_model/rnn.py`,
`dl_model/rnn_classification.py`) — training-only — and
`_init_model_and_optim()` rebuilds it at the start of the next run or CV fold.

Test: `test_model_is_usable_immediately_after_train`.

## Deviation: `train_model.py` was edited (1 hunk)

Scope said `base/model.py` plus the new tests. One edit outside that was
**required by defect C** and is reported here rather than done silently.

`train_model.py`'s inference path hand-copied the training conversion,
including the bad `sortby([..., "variable"])`. Fixing training alone would have
left inference feeding columns in alphabetical order into a model trained on
declared order — a silent, unwarned mis-order strictly worse than the original
bug, since before the fix at least both sides agreed. The 10-line block now
calls `model.to_tensor(data[factors].fillna(0), factors)` — the same method the
training path uses, so the two cannot drift again.

`test.py` was **not** touched, staged or committed. Every commit staged files
explicitly by path.

## `example/model.md` sections changed

The doc documented A, B, C, D, L1, L2 as live pitfalls, so leaving it alone
would have made it wrong. Each fixed entry was **rewritten, not deleted** —
numbering in 「常见坑」 is preserved and each now reads "曾经…（已于 2026-09-07
修复）… 现在…", with a pointer to the test that locks it.

Sections edited, in the same commit as the corresponding fix:

| Section | Commit | What changed |
|---|---|---|
| 「常见坑」#1 (`early_stopping=False` 会直接崩) | `dd395dc` | now records the crash as history; `early_stopping=False` is a normal config |
| 「常见坑」#2 (早停计数器按 batch) | `9b22f02` | now epoch-level; drops the "patience / 每epoch验证batch数 折算" workaround |
| 「张量形状」 code block | `905de76` | shows `to_tensor`'s implementation, `sortby` without `variable` |
| 「简单用法」 inference excerpt | `905de76` | shows the `to_tensor` call; the "契约缺口/可提炼点" framing is now history |
| 「已知的不完整之处」#8 (推理张量没被封装) | `905de76` | now封装完成; notes `predict_from_xarray()` still does not exist |
| 「常见坑」#3 (字母序) | `905de76` | history + the locked no-backcompat decision + the regression test |
| 「常见坑」#10 (`predict()` 不切 eval) | `830c573` | now automatic; documents the "stays in eval" side effect |
| 「常见坑」#6 (验证集丢一行) | `9d8c291` | fixed; adds that this is still **not** a purge/embargo — adjacent bars, forward-looking labels, leakage remains |
| 「常见坑」#8 (训练完模型就没了) | `6adc356` | model kept, optimizer dropped |
| `BaseModel` walkthrough table (`_train_dl`, `predict` rows) | `78d1f79` | described pre-fix behaviour |
| 「验证集是从训练段尾部按时间切的」 code excerpt | `78d1f79` | showed `train_split + 1:` |
| `_init_optim` contract | `78d1f79` | "`del self.optim` → 返回 None 会 AttributeError" no longer true |
| `_val_one_epoch` contract | `78d1f79` | return value is now averaged per epoch and must be `float()`-able |

## Commits (8, oldest first)

```
1e70566 test(quick-260907-fl6): first model-layer tests, red against six known defects
dd395dc fix(quick-260907-fl6): early_stopping=False no longer raises UnboundLocalError
9b22f02 fix(quick-260907-fl6): early-stopping patience counts epochs, not val batches
905de76 fix(quick-260907-fl6): tensor last axis follows the caller's declared order
830c573 fix(quick-260907-fl6): predict() runs in eval mode under no_grad
9d8c291 fix(quick-260907-fl6): validation split no longer drops a row
6adc356 fix(quick-260907-fl6): keep the trained model so predict() works after train()
78d1f79 docs(quick-260907-fl6): sweep example/model.md for claims the fixes invalidated
```

`1e70566` is red by construction (the RED commit). Every commit from `dd395dc`
onwards leaves the previously-fixed tests green, and `78d1f79` is green at 431.

## New test file

`tests/test_model_layer.py` — 428 lines, the first tests the model layer has
ever had. Synthetic panels, CPU, no zarr / network / credentials / GPU.
`FakePanel` implements only the six methods `collect()` actually calls, so the
whole KunQuant + zarr stack stays out. `wandb.init` is unconditional in
`_init_wandb` and `DLConfig` has no opt-out, so an autouse fixture sets the
documented `WANDB_MODE=disabled` bypass.

## Known stubs / carried forward

None introduced. Still open in `base/model.py`, **untouched and deliberately
out of scope** (all still documented as open in `example/model.md`
「已知的不完整之处」):

- `_auto_train` raises `NotImplementedError` for `MLConfig` (#1)
- `_do_vecbt` computes `price` and returns nothing; nothing calls it (#2)
- `_train_dl(backtest=...)` is never read (#3)
- `_vecbt` is a dead `NotImplementedError` (#4)
- no `predict_from_xarray()` — callers still write the 3-line window/select/
  fillna dance themselves (#8, partially addressed by `to_tensor`)
- `pin_memory=True` hardcoded (「常见坑」#9)
- config setter mutates the caller's factor/label objects in place (#11)
- `test_periods = train_periods // 5` in `train_cv` is not configurable (#7)

---

# 260907-fl6 — batch 2 of 2 (dl_model / dataset / factor / my_ops / ml_model)

Sequential on the main working tree (auto-degraded per #1941: HEAD was ahead of
origin/HEAD, so a harness worktree would have forked from a stale base).
`test.py` is the user's own uncommitted scratch edit — never read into a commit,
never staged. Every commit staged files explicitly by path; `git add .` / `-A`
were not used.

## Result

Suite: **431 passed → 468 passed**. Nothing skipped, xfailed or weakened —
`grep -rn "pytest.mark.skip\|pytest.mark.xfail\|pytest.skip("` over `tests/`
returns nothing, and the final run is a bare `468 passed`.

37 new tests in four new files:

| File | Tests | Locks |
|---|---|---|
| `tests/test_dl_models.py` | 14 | F, H, the refit optimizer, the Rule-1 deviation |
| `tests/test_backend_indexes.py` | 10 | E and `BaseDataset.time_interval` |
| `tests/test_factor_save_mode.py` | 5 | J |
| `tests/test_ml_backend.py` | 8 | the `MlBackend` `Self` fix |

Every one of them was **observed RED against the pre-fix code**, by copying the
fixed sources aside, `git checkout --`-ing them back to HEAD, running, and
restoring. The real output of each RED run is quoted per defect below.

## Commits (9, oldest first)

```
204380a fix(quick-260907-fl6): MLPRegressor is instantiable and actually trains
4f7436f fix(quick-260907-fl6): add DLConfig.lr_refit so update() is reachable
cc3c054 fix(quick-260907-fl6): RNNRegressor._val_one_epoch returns its loss
fca9f7c fix(quick-260907-fl6): get_xarray_dataset(indexes) actually honours indexes
309ae2f fix(quick-260907-fl6): update() reuses its refit optimizer instead of rebuilding it
8a353e4 fix(quick-260907-fl6): Factor.save(mode="a") failure names mode="w" as the fix
6fb37e3 docs(quick-260907-fl6): WindowedZScore's docstring no longer claims a fillna it never did
f8de7db refactor(quick-260907-fl6): delete three never-executed entrances
a04c9f7 fix(quick-260907-fl6): MlBackend's methods return Self so chaining works
b4a3894 docs(quick-260907-fl6): test_dl_models header lists all four locked defects
```

Each commit is green on its own: the three `dl_model/` commits were built by
materialising progressive versions of the shared test file and doc file, so no
commit in this batch contains a red test.

---

## F — `MLPRegressor` could not be instantiated  (`204380a`)

Confirmed, and worse than described in that the three faults are **stacked** —
each one hides the next, which is why the class had never been run:

```
E   AssertionError: assert frozenset({'_val_one_epoch'}) == frozenset()
E   TypeError: Can't instantiate abstract class MLPRegressor without an
    implementation for abstract method '_val_one_epoch'
```

Fixing only `_val_one_epoch` then gives:

```
E   TypeError: MLPRegressor._init_model() got an unexpected keyword argument
    'hyperparameters'
```

and fixing that too:

```
E   AttributeError: 'Tensor' object has no attribute 'fillna'. Did you mean: 'fill_'?
```

All three were demonstrated independently by reverting each fix in turn, so the
test's claim that "each sub-defect fails at a different point" is measured, not
asserted.

**Fixed, not deleted**, per the recorded decision.

- `_val_one_epoch` added, returning `val_loss.detach()`. It honours the contract
  batch 1 tightened: the epoch loop now runs `float(val_loss)` on every
  validation batch, unconditionally.
- `_init_model` takes `hyperparameters` and reads `hidden_size1`/`hidden_size2`
  from it, defaulting to the previously hardcoded 512/256 — so no existing
  configuration changes model shape.
- `_preprocess` is `torch.nan_to_num(data, nan=0.0)`, matching both RNN heads.
  Its annotation said `xr.Dataset`; the base contract is `(Tensor) -> Tensor`.
- The now-unused `import xarray as xr` was dropped.

**Evidence is a real training run**, not an import:
`test_mlp_regressor_trains_two_epochs_and_predicts` runs `collect()` → two CPU
epochs of train/val/test → checkpoint → `predict()`, and asserts `fc1`'s weights
actually moved. A head that ran the loop without ever stepping the optimizer
would satisfy every other line in that test.

### One gap deliberately left open

`MLPRegressor.predict()` requires an **already-flattened**
`[num_times, num_symbols * num_features]` input. The reshape lives in
`_train_one_epoch`/`_test_one_epoch`, while `BaseModel._predict_nn` hands the
tensor straight to the module and `MLP.forward` is a plain `nn.Linear` stack.
Closing it means either editing `base/model.py` (batch 1's file) or changing
what the public `MLP` module accepts — both outside this scope. The test asserts
the flattened contract so it is pinned rather than rediscovered at a call site,
and `example/model.md` #5 records it.

## H — `update()` read `config.lr_refit`, a field `DLConfig` lacked  (`4f7436f`)

Confirmed:

```
E   AttributeError: 'DLConfig' object has no attribute 'lr_refit'
    /Users/daizhaorong/projects/quantlab/dl_model/rnn.py:348
E   AttributeError: 'DLConfig' object has no attribute 'lr_refit'
    /Users/daizhaorong/projects/quantlab/dl_model/rnn_classification.py:468
```

**Decision: add the field, do not drop the read.** The reason is in the code the
original author wrote — `update()` opens with `if self.config.lr_refit <= 0.0:
return`, i.e. it was designed around a config-supplied switch whose zero value
means "off". Dropping the read would force choosing a learning rate for the
fine-tuning step, and both docstrings state it must be **smaller** than the
training `lr`; that is a modelling decision with no basis anywhere in the
repository. `lr_refit: float = 0.0` makes `update()` a no-op unless a caller
opts in, so nothing that exists today changes behaviour (nothing constructs
`DLConfig` with it, and nothing calls `update()` at all).

Both halves are locked, because "field added but nobody reads it" would
otherwise pass: with the default config `update()` must not touch a single
parameter, and with `lr_refit > 0` it must actually step the optimizer.

## The refit optimizer — `update()` rebuilt it every call  (`309ae2f`)

Added mid-task by the user. Every `update()` did
`optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr_refit)`
on each call, so Adam's first- and second-moment estimates were zeroed every
step — online training silently degrading to SGD with an odd warmup, nothing
raised. RED:

```
E   AssertionError: the refit optimizer carries no per-parameter state at all
    (RNNRegressor and RNNClassifier both)
```

`BaseModel._get_refit_optim()` builds it once and caches it on the instance,
keyed on **`(self.model` object identity`, lr_refit)`**. Keying on the model
OBJECT is the invalidation mechanism: the optimizer holds references to the
parameter tensors, so after `load()` or another `_init_model()` a cached one
would go on stepping tensors that are no longer the model's — worse than the
bug being fixed. Because the key is identity rather than a dirty flag, no call
site has to remember to invalidate anything, so `load()` and
`_init_model_and_optim()` are untouched. A changed `lr_refit` also rebuilds.

**The tests assert on STATE, not identity**, as required: after two `update()`
calls every parameter's `state[p]["step"]` must be 2 and some `exp_avg` buffer
must be non-zero. `id(opt1) == id(opt2)` would have passed with the state wiped.
Two more tests cover invalidation (the rebuilt optimizer must have empty state
AND point at a disjoint set of parameter tensors) and the `lr_refit` change.

`self.optim = None` after `_train_dl` is untouched and asserted by
`test_train_dl_still_drops_the_training_optimizer` — the refit optimizer did not
resurrect it.

Out of scope by explicit instruction and NOT done: `cal_stream()` is still not
wired to `update()`, there is no online-training loop, `lr_refit` still defaults
to 0.0.

## E — `XrBackend.get_xarray_dataset` ignored `indexes`  (`fca9f7c`)

Confirmed. Body was `return self.data`; `PlBackend` on the same ABC method DID
use it. RED (7 of 10 failed):

```
E   AssertionError: assert ('symbol', 'timestamp') == ('timestamp', 'symbol')
E   AssertionError: assert {'timestamp': 4, 'symbol': 2} == {'timestamp': 4}
E   AssertionError: assert 'depth' not in Data variables: ...
E   Failed: DID NOT RAISE ValueError
E   TypeError: PlBackend.get_xarray_dataset() missing 1 required positional
    argument: 'indexes'
E   TypeError: numpy boolean subtract, the `-` operator, is not supported ...
    (x2 — both time_interval tests)
```

### Semantics chosen

**`indexes` names the dimensions the returned dataset is indexed by, in order**
— the meaning `PlBackend` already gave it (`set_index(indexes)` then
`Dataset.from_dataframe`). Concretely, `XrBackend`:

1. validates every name is a dimension, raising `ValueError` that lists the
   dimensions that ARE present;
2. drops every data variable laid out on a dimension outside `indexes`;
3. drops the dimensions left unused, together with their coordinates;
4. transposes the survivors onto `indexes`.

`indexes=None` means "no shape request, as-is" and keeps returning the backend's
own object (not a copy) — dozens of call sites depend on that identity.

### Why every existing call site keeps working

I checked all of them before choosing. The repository passes exactly two things:
`["timestamp", "symbol"]` (`base/data.py` ×4, `base/model.py` ×6,
`example`/`tests`) or nothing at all (`base/factor.py`, `dataset/masking.py`,
`train_model.py`, most tests). On a canonical panel every data variable is laid
out on exactly those two dims, so steps 2 and 3 drop nothing and only the
transpose remains — which is precisely the CLAUDE.md invariant that was
previously carried by `from_raw_data()`'s densification rather than checked at
the boundary. `test_two_dim_request_pins_the_axis_order_and_keeps_every_variable`
asserts that non-regression directly (on a deliberately `(symbol, timestamp)`-
ordered panel, so the transpose is observable), and
`test_variables_on_an_unrequested_dimension_are_dropped` uses a third axis to
prove the dropping rule is not vacuous — otherwise the non-regression test would
be passing for the wrong reason.

`PlBackend.get_xarray_dataset` gained the ABC's `None` default and rejects it
with a named error: a LazyFrame has no dimensions to fall back on.

`base/backend.py`'s ABC method gained the Chinese docstring `example/dataset.md`
already claimed existed, so that claim is now true.

### `BaseDataset.time_interval`

Both causes fixed: `["timestamp"]` now returns a time-axis-only dataset (bool
`anomaly_flag` no longer in the way), and the property takes the `timestamp`
COORDINATE — a `DataArray`, which does have `.to_series()` — before diffing.
`.mode()` is kept and tested: a daily panel with a two-day hole still reports
1 day. Its only caller remains `dataset/spot.py:_xr_to_bars`.

`get_xarray_dataset` also does not mutate `self.data`, which is asserted:
`filter_by_date`/`filter_by_symbol` on the same interface DO narrow in place,
and an implementation written by analogy with them would reintroduce RV-01.

## J — `Factor.save(mode="a")`'s failure was incomprehensible  (`8a353e4`)

Confirmed on the real path (two genuine `save()` calls, 29 then 31 timestamps):

```
E   AssertionError: variable 'alpha' already exists with different dimension
    sizes: {'timestamp': 29, 'symbol': 2} != {'timestamp': 31, 'symbol': 2}.
    to_zarr() only supports changing dimension sizes when explicitly appending,
    but append_dim=None. ...
```

**The default is unchanged**, per the recorded decision. The wrapper catches
only `ValueError`s carrying zarr's `"already exists with different dimension
sizes"` wording and re-raises one that names `mode="w"`, names the store, says
what zarr's `"a"` actually means, and points at `XrBackend.append()`. The
original exception is preserved on `__cause__`.

Two tests exist specifically to stop this fix from becoming a new bug:
`test_unrelated_value_errors_are_not_swallowed` (a catch-all relabelling every
failure as "use mode=w" would be worse than the original) and
`test_the_default_is_still_a`, which pins the DECISION so a future edit has to
argue with it rather than drift past it.

## M — `WindowedZScore`'s docstring claimed a fillna it never did  (`6fb37e3`)

Fixed the **docstring**, not the code, per instruction — and the direction is
the whole point. Both `Alpha101SpotKline` and `Alpha158SpotKline` wrap every
single `Output(...)` in `WindowedZScore`, so adding a `fillna(0)` would change
every factor value this op has ever produced, invalidating every persisted
factor store and every model trained on one. Nothing can be relying on the
documented behaviour, because that behaviour never existed for a day.

The replacement also says what the NaNs a caller DOES see are: the first
`window - 1` rows are a rolling op's normal warm-up, not missing-value handling.

## Deletions  (`f8de7db`)

Verified zero-call-site immediately before deleting. The greps:

**G — `dl_model/rnn.py:RNNClassifier`.** Before deletion, every reference to the
name in `*.py`/`*.ipynb`:

```
train_model.py:13:from dl_model.rnn_classification import RNNClassifier
train_model.py:57:model = RNNClassifier(mc)
train_model.py:63:  ".../RNNClassifier_trial_20250909_183651/..."      (a path string)
tests/test_model_layer.py:16,308                                      (prose)
tests/test_dl_models.py:6,38,140,153,293,401   -> rnn_classification
dl_model/rnn.py:301                                                   (prose)
dl_model/rnn_classification.py:164:class RNNClassifier(BaseModel)      <- the live one
dl_model/rnn.py:400:class RNNClassifier(BaseModel)                     <- the stale copy
base/model.py:259                                                     (prose)
```

and everything importing from that module:

```
cal.py:5:from dl_model.rnn import RNNRegressor
tests/test_dl_models.py:37:from dl_model.rnn import RNNRegressor
```

Zero references resolve to the stale copy. `ModelRBaseCrypto`/`ModelRCrypto`
stay — `RNNRegressor` uses them. Six sklearn classification metrics imported
only by the deleted copy went with it. After deletion, `dl_model/rnn.py` exposes
`ModelRBaseCrypto, ModelRCrypto, RNNRegressor` and nothing else.

**K1 — `my_ops/preprocess.py:WindowedRobustStandardization`.** Before:
`my_ops/preprocess.py:28:class WindowedRobustStandardization(...)` — the
definition and nothing else. After: `grep -rn` over `*.py`/`*.ipynb` returns
nothing.

**K2 — `base/pageledger.py:PageLedger.last_position()`.** Before:

```
base/pageledger.py:254:    def last_position(self) -> tuple[...]
base/pageledger.py:323:        token-free fallback -- see `last_position`.   (the dangling doc ref)
```

The `last_symbol`/`last_timestamp` FIELDS stay — they are the raw material for
the token-free fallback D-03 asks for; only the unused accessor went.
`_append_page`'s docstring cross-reference is rewritten to describe the fallback
directly and record where the reader went, rather than pointing at nothing.
After deletion, the only `last_position` hit in the repo is that explanatory
prose.

`example/pageledger.md`'s runnable demo called `last_position()` twice and
printed its output. Rewritten to read `pages[-1]` directly, then **re-run end to
end** (`exit=0`) and its recorded output updated to match the new label.

### `MlBackend` / `ModelBackend`: KEPT, and fixed  (`a04c9f7`)

Originally on the deletion list. The user confirmed mid-task that
`BaseModel.predict()`'s `np.ndarray` branch is deliberate — it exists for
non-torch `MLConfig` models such as xgboost, and `MlBackend` is their joblib
persistence. That is scaffolding for an intended path, not an abandoned
entrance. **Nothing had been deleted at that point, so no restore was needed**;
`ModelBackend` was never orphaned because `MlBackend` never went.

Its real defect was fixed instead. `ModelBackend` declares
`read`/`write`/`to_internal` as `-> Self` (and `**kwargs` on the first two), but
all three implementations returned `None` implicitly. RED, 6 of 8:

```
E   AttributeError: 'NoneType' object has no attribute 'write'
E   AssertionError: assert None is MlBackend()      (x3)
E   AssertionError: MlBackend.read drops the **kwargs ModelBackend declares
```

**No change to `ModelBackend` was needed** — the base already declared exactly
this contract; the implementations were the ones out of step, so there was no
conflict to stop and report. `tests/test_ml_backend.py` covers the full chain
round-trip, each method's return individually (so a regression names the one
that broke), parent-directory creation, and a real `write(..., compress=3)`
passthrough so `**kwargs` cannot become decoration.

## Deviations from the brief

**1. Rule 1 — `RNNRegressor._val_one_epoch` returned `None`  (`cc3c054`).** Not
on the list; found while writing the first tests `dl_model/` has ever had, and
live rather than theoretical. `BaseModel` declares `-> torch.Tensor`; this
implementation logged metrics and returned nothing. Batch 1 turned that from a
false annotation into a hard failure — the epoch loop now runs
`val_loss_sum += float(val_loss) * batch_samples` **unconditionally**, not under
`if self.config.early_stopping` — so `RNNRegressor.train()` raised
`TypeError: float() argument must be a string or a real number, not 'NoneType'`
on epoch 0 for every configuration. Treated as a Rule-1 auto-fix rather than a
design decision because the base contract, batch 1's tightened requirement and
the in-repo precedent (`rnn_classification.py`'s sibling already returned
`val_loss.detach()`) all name the same value.

**2. `base/model.py` was touched — batch 1's file.** One method added
(`_get_refit_optim`), zero existing lines modified, and only because the
coordinator asked for a shared helper over duplicating the caching logic into
each `update()`. `tests/test_model_layer.py` was not touched at all.

**3. Dead imports removed alongside the deletions.** Six sklearn classification
metrics in `dl_model/rnn.py` and `import xarray as xr` in `dl_model/mlp.py`,
each unused only as a consequence of a change in the same commit.

## Docs updated (a doc still describing a fixed bug is a wrong doc)

Batch 1's convention followed throughout — pitfalls **rewritten as**
「曾经…（已于 2026-09-07 修复）…现在…」 with the locking test named, so numbering
stays stable.

| Doc | Section | Commit |
|---|---|---|
| `example/model.md` | 「子类的五方法契约」intro; 「已知的不完整之处」#5 | `204380a` |
| `example/model.md` | 「已知的不完整之处」#7 (incl. the add-vs-drop reasoning) | `4f7436f` |
| `example/model.md` | `_val_one_epoch` contract section | `cc3c054` |
| `example/model.md` | #7 extended with the refit-optimizer fix | `309ae2f` |
| `example/model.md` | 「已知的不完整之处」#6 (the deleted `RNNClassifier`) | `f8de7db` |
| `example/backend.md` | contract summary; new-backend checklist #4; 「常见坑」#3 | `fca9f7c` |
| `example/backend.md` | `MlBackend` intro; checklist #3; 「常见坑」#5 | `a04c9f7` |
| `example/dataset.md` | 「核心契约」; the `time_interval` aside; 「常见坑」#6 | `fca9f7c` |
| `example/factor.md` | 「已知的坑」#4 (`save(mode="a")`) | `8a353e4` |
| `example/factor.md` | 「已知的坑」#11 (`WindowedZScore` docstring) | `6fb37e3` |
| `example/factor.md` | the `WindowedRobustStandardization` aside | `f8de7db` |
| `example/pageledger.md` | 「last position」section, the runnable demo, its output, the closing note | `f8de7db` |
| `example/README.md` | defect table: the `dataset/backend.py` row | `fca9f7c` |
| `example/README.md` | defect table: the `base/factor.py` row | `8a353e4` |
| `example/README.md` | defect table: the 死代码 row (incl. the `MlBackend` correction) | `f8de7db` |
| `CLAUDE.md` | 「Key Abstractions」 `get_xarray_dataset` bullet | `fca9f7c` |
| `CLAUDE.md` | component table: `my_ops/preprocess.py` row | `f8de7db` |
| `README.md` | `my_ops/` bullet | `f8de7db` |
| `tests/test_dl_models.py` | module docstring, after the refit section was appended | `b4a3894` |

## Known stubs / carried forward

None introduced. Deliberately still open and untouched:

- `MLPRegressor.predict()` wants a pre-flattened matrix (see F above).
- `base/model.py`: `_auto_train` raises `NotImplementedError` for `MLConfig`;
  `_do_vecbt` computes `price` and returns nothing; `_train_dl(backtest=...)`
  is never read; `_vecbt` is a dead `NotImplementedError`; no
  `predict_from_xarray()`; `pin_memory=True` hardcoded; the config setter
  mutates the caller's factor/label objects in place; `test_periods =
  train_periods // 5` in `train_cv` is not configurable.
- `Factor.save()` is still not wired to `XrBackend.append()`, so there is still
  no true incremental factor append — only a comprehensible error saying so.
- `MlBackend` still has zero call sites; `MLConfig` training remains unbuilt.
- `update()` is still not called by anything: `cal_stream()` is not wired to it
  and there is no online-training loop. Out of scope by explicit instruction.


---

# 260907-fl6 — batch 3 of 3 (naming, `num_null`, the vecbt skeleton, a typo)

Sequential on the main working tree (auto-degraded per #1941). `test.py` is the
user's own uncommitted scratch edit — never read into a commit, never staged;
`git log --name-only` over all six batch-3 commits contains zero `test.py`
entries, and `test.py` references none of the renamed identifiers, so the
renames could not have broken it. Every commit staged files explicitly by path.

## Result

Suite: **468 passed → 478 passed**. Nothing skipped, xfailed or weakened —
`grep -rn "pytest.mark.skip\|pytest.mark.xfail\|pytest.skip("` over `tests/`
returns nothing, and the final run is a bare `478 passed`.

10 new tests, all in existing files:

| File | New tests | Locks |
|---|---|---|
| `tests/test_model_layer.py` | 6 | `num_null` (2), the vecbt skeleton (4) |
| `tests/test_dl_models.py` | 1 | `RNNClassifier._vecbt` |
| `tests/test_spot_dataset.py` | 3 | the `get_crypto_currency` rename + dropped param |

Every behavioural one was **observed RED before its fix** (the two renames are
pure and have no RED to observe; the full suite was re-run green after each).

## Commits (6, oldest first)

```
ad8fe3b refactor(quick-260907-fl6): rename _*_one_epoch to _*_one_batch
36ded47 refactor(quick-260907-fl6): rename the collectors to _collect_all_*
13412d1 fix(quick-260907-fl6): num_null returns an int instead of raising IndexError
bd9c95c fix(quick-260907-fl6): the vecbt skeleton fails honestly instead of silently
f80e9fb fix(quick-260907-fl6): get_crypot_currency -> get_crypto_currency
87c5bac docs(quick-260907-fl6): sync the defect tables and test headers with batch 3
```

The full suite was run green (468) after each of the two renames before moving
on, as instructed — a missed site would have surfaced immediately rather than
three tasks later.

---

## Task 1 — `_*_one_epoch` → `_*_one_batch`  (`ad8fe3b`)

Confirmed per-batch: all three call sites sit inside
`for x_batch, y_batch in <loader>:` and each call performs a complete optimizer
step. Renamed across the three abstract definitions and all call/comment sites
in `base/model.py`, all overrides in `dl_model/{mlp,rnn,rnn_classification}.py`,
`tests/test_model_layer.py`, `tests/test_dl_models.py`, the prose in
`example/model.md`, and the 5-method contract line in `CLAUDE.md:107`.

Test function names that embedded the old method name went with it
(`test_mlp_val_one_epoch_returns_a_floatable_loss` →
`..._val_one_batch_...`, and the RNN equivalent), and `example/model.md`'s
pointers at those tests were updated to match — otherwise the doc would name
tests that no longer exist.

**Verified pure.** `git diff -U0` over the code files, filtered to drop every
line containing `one_epoch`/`one_batch`, returns *only* the added docstrings.
Nothing but identifiers and surrounding prose moved. Suite green at 468 before
committing.

### The docstrings added to the base class

Per instruction, the three abstract methods now state the per-batch contract
explicitly and record **why** the old name was not merely cosmetic — batch 1's
early-stopping defect (B) happened because the counter was written next to
`_val_one_epoch` and the name read as "per epoch". `_train_one_batch` also
points single-step / online training at where it actually lives:

> **想做单步 / 在线训练的不要来改这里。** 这个钩子属于 `_train_dl` 的批量训练
> 循环。在线学习的入口是各模型头的 `update()`，它用 `_get_refit_optim()` 拿一个
> **跨调用复用**的微调优化器（复用是必要的：每步新建会把 AdamW 的动量清零）。

That is the intent the user stated for these methods, and batch 2's persistent
refit optimizer is what now serves it.

`epoch` stays as the first parameter — it genuinely is the epoch index, passed
through so implementations can log to the right W&B step. The docstring says so,
because "the name said epoch and meant batch" is exactly the confusion being
retired.

`.planning/codebase/{ARCHITECTURE,STRUCTURE}.md` also name the old methods but
were left alone: the orchestrator owns `.planning/`.

## Task 2 — the collectors  (`36ded47`)

Confirmed: `_get_labels_batch()` / `_get_features_batch()` build no mini-batch.
They loop over **every** entry in `config.labels` / `config.factors`, dispatch
`cal()` or `read()`, and `xr.combine_by_coords` the lot. In a module where
`batch` already means the `DataLoader` mini-batch, one word meant two opposite
things.

Renamed to `_collect_all_labels` / `_collect_all_features` across the two
definitions and two call sites in `base/model.py`, plus the comment references
in `tests/test_factor_hierarchy.py` (the `# (2) base/model.py:...` markers at
~453, ~496 and ~626 — they point at the replicated call surface, so they were
updated rather than deleted) and the prose in `example/model.md:24` and
`example/factor.md:82`.

Both definitions gained a short docstring recording the reason, so the next
person to reach for the word `batch` here has to read why it is not available.

## Task 3 — `num_null` was broken  (`13412d1`)

Confirmed exactly as described, and the RED run is the reported error verbatim:

```
E   IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
/Users/daizhaorong/projects/quantlab/base/model.py:100: IndexError: ...
```

`.isnull().sum()` gives a Dataset of 0-d sums, `.to_dataarray()` stacks them on
a `variable` dim, and the second `.sum()` collapses that too — so `.values` is
already 0-dimensional and `[0]` could never work. **Every** read raised. The
property is annotated `-> int` and `example/model.md` recommends it as the
pre-training missing-value check, so it was documented, advertised and unusable.

Fix: take the 0-d array itself and cast explicitly — `int(...sum().item())`.
The explicit `int()` is not decoration: `.item()` yields a numpy scalar, and
returning that would leave `-> int` still approximately-true rather than true.

Two tests, both RED first:

- `test_num_null_counts_missing_cells_and_returns_an_int` — a new `HolePanel`
  punches a **different** number of NaNs into the factor panel (7) and the label
  panel (4), so a fix that reads only one of them, or that stops at the
  per-variable `.sum()` (a Dataset, not a scalar), cannot pass. Asserts the
  count *and* `isinstance(n, int)`.
- `test_num_null_is_zero_on_a_dense_panel` — the counterpart. Without it, an
  implementation returning some constant could pass the counting test by
  accident on one geometry.

`_make_config` gained additive `factors=` / `labels=` overrides (defaults
unchanged) so `HolePanel` could be injected without a second helper.

## Task 4 — the vecbt skeleton is now honest  (`bd9c95c`)

All four pieces confirmed. **KEPT, not deleted**, and the rationale is recorded
in the code: the `MLConfig`/xgboost path is confirmed intended, so a backtest
hook that will eventually serve both torch and non-torch models is in the right
place on the base class — it simply has no content yet, and **Phase 6** owns
end-to-end backtesting.

The RED run showed all three failure modes at once, including the silent one:

```
E   Failed: DID NOT RAISE NotImplementedError
----------------------------- Captured stderr call -----------------------------
RecordingRegressor_train: 100%|██████████| 2/2 [00:00<00:00, 117.33it/s]
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'read'   (_do_vecbt reached its dead body)
E   AssertionError: Regex pattern did not match. Expected regex: 'Phase 6'
      Actual message: ''                                                     (the bare stub)
```

That tqdm bar is the defect: `backtest=True` trained two full epochs and
returned normally.

### What changed, per piece

- **`_do_vecbt`** — now raises `NotImplementedError` naming Phase 6. Its two
  existing argument checks are deliberately kept **in front** of the raise: an
  unconfigured `backtest_data` is a mistake the caller can fix today, and they
  should hear about that one first. Test asserts both orderings.
- **`_train_dl(backtest=...)`** — **chosen: reject a truthy value**, rather than
  honour it. Honouring it would mean calling `_do_vecbt()`, which raises anyway,
  so the only real question was *where*. The guard is at the **top of
  `_train_dl`, before any work**: this parameter is passed once per run and the
  training behind it takes hours, so discovering the gap afterwards is barely
  better than not discovering it. `test_train_dl_rejects_a_truthy_backtest_flag`
  asserts `model.train_epochs == []`, which pins the fail-fast rather than just
  the raise. `test_train_dl_still_trains_when_backtest_is_falsy` exists so the
  new guard cannot become a landmine on the default path `_auto_train` uses.
- **`BaseModel._vecbt`** — stub **not deleted**, per instruction. Its message
  was the empty string; it now names Phase 6 and says which half of the job it
  is (`_do_vecbt` organises the data, `_vecbt` runs the backtest).
- **`RNNClassifier._vecbt`** — raises instead of returning `None`.

### A real discovery: the four preserved Series lines do not run

Writing the `RNNClassifier._vecbt` test surfaced something not in the brief. The
plan was to keep the four lines executable and raise after them. That failed:

```
pandas.errors.IndexingError: Unalignable boolean Series provided as indexer
(index of the boolean Series and of the indexed object do not match)
```

`long_exits = long_entries[short_entries == 1]` indexes one complementary subset
of `signals` with the other's boolean mask; their indices are disjoint by
construction. Measured across five signal series — `[1,0,1]`, `[0,1]`, `[1,1]`,
`[1,0,0,1,1]` all raise; only `[0,0]` (no long entries at all) survives. So the
line cannot execute on any real signal.

Leaving them ahead of the raise would have swapped "Phase 6 hasn't built this"
for a baffling pandas error — **not more honest, just a different lie**. So the
four lines were **preserved verbatim inside the docstring** instead of as dead
executable code. That satisfies "do not delete the skeleton": the author's
entry/exit convention (0 = short, 1 = long) is recorded in full, where the next
reader sees it, together with the measurement showing which line is broken and a
note that `short_exits` (a long mask computing a short exit) is suspect too and
must be re-derived rather than copied.

## Task 5 — `get_crypot_currency` → `get_crypto_currency`  (`f80e9fb`)

Confirmed: "crypot" for "crypto", sitting **directly above** a correctly spelled
`get_crypto_currency_pair` in the same file — two names side by side that look
like two different concepts. Imported by `dataset/spot.py:30` and called twice
(`base_currency` / `quote_currency`). Renamed at the definition, the import and
both call sites; `grep -rn crypot` over the repo now returns only the
explanatory sentence in the new docstring.

### The `name` parameter: **DROPPED**, not wired up

Reasons, in order of weight:

1. `Currency.from_str(code, strict=False)` — the entire body — **has no `name`
   argument**. Verified against the installed signature, not from memory.
2. Honouring it would mean switching to the `Currency(code, precision, iso4217,
   name, currency_type)` constructor, which forces a `precision` and a
   `currency_type` decision per coin. There is no basis for either anywhere in
   the repository, and it would change behaviour at both existing call sites.
3. Neither call site passes it.
4. An accepted-and-then-ignored parameter is the same lie as
   `_train_dl(backtest=...)`, fixed one commit earlier. Fixing one and keeping
   the other would be incoherent.

The now-unused `from typing import Optional` went with it.

Three tests: the correct spelling exists **and the misspelling does not survive
as an alias**; the signature is exactly `["symbol"]` (this pins the decision, so
a future edit has to argue with it rather than quietly re-add an ignored
parameter); and a real round-trip returning `Currency` for `BTC` / `USDT`, which
is the whole contract both call sites rely on.

## Deviation: `example/README.md`'s defect table was rewritten

One edit beyond the five tasks, reported rather than done silently.

`example/README.md`'s 「写文档时发现的缺陷」 table still listed the **batch 1 and
batch 2** defects as live problems — and its `base/model.py` row described the
early-stopping-counts-batches bug by naming the very methods task 1 renamed. A
doc that names `_val_one_epoch` as a current pitfall is wrong twice over after
this batch.

Those two rows were struck through and marked 已修复 (the convention batch 2
established), and four rows were added for batch 3: `num_null`, the vecbt
skeleton, the misleading names, and the `crypot` typo. Nothing was deleted from
the table.

## Docs updated

| Doc | Section | Commit |
|---|---|---|
| `example/model.md` | `_train_one_batch` contract + the new 改名 rationale block | `ad8fe3b` |
| `example/model.md` | `_val_one_batch` / `_test_one_batch` headings and prose | `ad8fe3b` |
| `example/model.md` | the 「跑一次真实训练」 output note (「per-batch 的直接证据」) | `ad8fe3b` |
| `CLAUDE.md` | 「Pattern Overview」 5-method contract line | `ad8fe3b` |
| `example/model.md` | 「吃：两份 xarray 面板」 collector names | `36ded47` |
| `example/factor.md` | line 82, the model-layer call-surface sentence | `36ded47` |
| `example/model.md` | `_preprocess` section — the `num_null` recommendation now says what it returns, plus the fixed-history note | `13412d1` |
| `example/model.md` | 「已知的不完整之处」 #2/#3/#4 merged into one 回测骨架 entry with a before/after table and the IndexingError measurement | `bd9c95c` |
| `example/README.md` | 缺陷 table: 6 rows (2 struck through, 4 added) | `87c5bac` |
| `tests/test_model_layer.py` | module docstring: batch-3 additions | `87c5bac` |
| `tests/test_dl_models.py` | module docstring: the `_vecbt` entry | `87c5bac` |

## Known stubs / carried forward

None introduced. The vecbt skeleton is **still a skeleton** — that was the
decision, not an oversight; it is now loud instead of silent. Still open and
untouched:

- `_auto_train` raises `NotImplementedError` for `MLConfig`; `MLConfig` training
  remains unbuilt and `MlBackend` still has zero call sites.
- End-to-end backtesting (Phase 6): `_do_vecbt`, `_vecbt` and
  `RNNClassifier._vecbt` all raise; `_train_dl(backtest=True)` is rejected.
  Nothing calls any of them. The real backtest is still the hand-written
  vectorbt block in `train_model.py`.
- `RNNClassifier._vecbt`'s `short_exits` line is suspect (a long mask computing
  a short exit) and `long_exits` is provably broken. Both recorded in the
  docstring for Phase 6 to re-derive; neither fixed, since the surrounding
  function has no implementation to be correct against.
- `MLPRegressor.predict()` still wants a pre-flattened matrix.
- No `predict_from_xarray()`; `pin_memory=True` hardcoded; the config setter
  mutates the caller's factor/label objects in place; `test_periods =
  train_periods // 5` in `train_cv` not configurable.
- `Factor.save()` still not wired to `XrBackend.append()`.
- `update()` is still called by nothing: `cal_stream()` is not wired to it and
  there is no online-training loop. Task 1's docstrings now at least point a
  reader at it, which is the extent of what was in scope.
- `.planning/codebase/{ARCHITECTURE,STRUCTURE}.md` still name the pre-rename
  methods. Left deliberately: the orchestrator owns `.planning/`.

---

# 260907-fl6 — batch 4 (code-review remediation: BL-01, BL-02, WR-01)

**Input:** `.planning/quick/260907-fl6-.../REVIEW.md` (2 Critical, 7 Warning,
6 Info). Scope for this batch was fixed at **BL-01, BL-02 and WR-01 only**.
The other five Warnings and six Infos were deliberately NOT touched — see
「Out of scope, still open」 below.

**Suite: 478 → 491 passed** (`~/.venv/bin/python -m pytest -q`, 27.9s).
Nothing skipped, nothing xfailed, nothing weakened. +13 tests.

**Commits (3, one per finding):**

| Commit | Finding |
|---|---|
| `b579ee1` | BL-01 — `validate_schema` goes silent when every required column is null |
| `417b363` | BL-02 — `to_tensor` does no dtype normalization, so a float64 panel cannot train |
| `fad4d46` | WR-01 — `RNNRegressor._init_model` accepts `hyperparameters` and ignores it |

Every fix was **observed RED before it was written**. `test.py` (the user's own
scratch edit) was never staged or touched.

## BL-01 — `dataset/cleaning.py:validate_schema`

**Observed RED.** With `open/high/low/close/volume` all-null and a `vwap`
carrying one real value plus nulls, the only output was a single INFO line:

```
INFO | validate_schema: 4/4 (100.0%) (timestamp, symbol) cell(s) hold no bar at
all — null in every required column. That is the dense panel's cartesian
product (D-06) ...
```

Zero WARNING records, for any of the six columns — reproducing the reviewer's
`[]` exactly. The mask is a logical AND over `isnull()` of every required
column, so when the required columns are themselves all-null the mask is True
everywhere, `~mask` is False everywhere, and every per-column warning is
suppressed.

### The decision: `logger.error` + disable the mask, do NOT raise

**What "loud" was set to, and why.** The empty-ingest case is reported at
`logger.error` — one level ABOVE the per-column warnings it used to swallow.
That asymmetry is the point: it does not mean "one column looks odd", it means
the ingest produced nothing usable at all, and it should outrank the warnings
whose absence was the symptom. The mask is then disabled for that panel, so the
per-column loop reports real whole-column counts instead of the zeros the mask
forces. Both halves are needed — the ERROR alone would still leave six columns
silently reporting nothing.

**Why not `raise`, which the reviewer suggested first.** `validate_schema`
raises today only on a SCHEMA violation (a missing column). An all-null panel
is a data-CONTENT problem, and content problems are governed by D-07
(flag-don't-delete) — the function's own docstring commits to not raising on
nulls. More concretely: `clean_market_data()` runs inside `from_raw_data()` on
every single ingest and **no caller anywhere catches anything**, so raising
would turn a diagnostic into an unrecoverable abort. A chunked backfill whose
window happens to contain no traded bar for any symbol (a holiday stretch, a
universe of not-yet-listed tickers) would die on that window rather than log
it and carry on. Escalate-and-continue keeps the operator informed without
handing a data-quality check the power to kill an ingest.

**Locked by 4 tests** in `tests/test_cleaning.py`:
`test_an_all_null_required_panel_is_reported_at_error_level`,
`::..._stops_suppressing_per_column_warnings`,
`::..._still_does_not_raise`, and
`test_the_sparse_panel_is_unaffected_by_the_all_null_escalation` — the last one
is the "existing behaviour unchanged" pin the constraint asked for: the D-06
sparse panel must still report at INFO, still emit no ERROR, and still keep
`trade_count` silent. A new `_captured_records()` helper keeps the loguru LEVEL
alongside the message, because "it was logged" is not the assertion that
matters here; "it was logged louder" is.

The docstring's false claim (a fully-withheld `vwap` still warns) is now
corrected rather than deleted: it holds while at least one required column has
data, and the degenerate case is documented directly beneath it.

## BL-02 — `base/model.py:to_tensor`

**Observed RED, real error, on a SHIPPED head:**

```
ValueError: RNN input dtype (torch.float64) does not match weight dtype
(torch.float32). Convert input: input.to(torch.float32), or convert model:
model.to(torch.float64)
../../.venv/lib/python3.13/site-packages/torch/nn/modules/rnn.py:319: ValueError
```

Both `RNNRegressor` and `RNNClassifier` failed. Not a stub — the classes that
ship, running `collect()` → `_init_model_and_optim()` → `train()`.

### Killing the test blindness first

The constraint was right that the existing 47 tests could not see this. The
panel had to produce float64 **the way the real pipeline produces it**, so
`FakePanel` gained `via_pandas=True`, which builds through
`DataFrame.set_index(["timestamp", "symbol"]).to_xarray()` — literally
`dataset/*._raw_data_to_xr()` — and casts nothing. `test_the_real_pipeline_
panel_is_float64_not_float32` asserts the premise rather than assuming it; if
`via_pandas` ever stopped yielding float64 the two training tests would go
green for the wrong reason again, which is exactly how this escaped twice.

### The fix, and the two stated decisions

Normalized at `to_tensor` — the one seam where a panel becomes a tensor — and
not in `_preprocess`. There are three `_preprocess` implementations; the fourth
head would forget. `test_to_tensor_downcasts_a_float64_panel` would stay red
under the per-head alternative, which is what makes that a pinned decision
rather than a preference.

1. **`torch.get_default_dtype()`, not a hardcoded `float32`.** Load-bearing,
   not cosmetic: someone who runs `torch.set_default_dtype(torch.float64)` gets
   float64 modules, and a hardcoded downcast would break them in the mirror
   image of BL-02. Pinned by `test_to_tensor_follows_torchs_default_dtype_not_
   a_hardcoded_float32`, which flips the global default (restored in `finally`)
   and asserts a float32 panel comes back float64. That test goes red the
   moment anyone writes `astype(np.float32)`.
2. **float64 → float32 loses precision, and that is the right trade here.**
   torch modules are float32 by default and market-data factors do not need
   float64 mantissas. Stated in the docstring in those words so it stays a
   decision. The cast is scoped to **floating** dtypes only: an int/bool panel
   (a constituent-membership mask, a categorical code) passes through
   untouched, because silently floating it would blur meaning rather than
   precision. `test_to_tensor_leaves_non_floating_panels_alone` makes widening
   that scope a decision rather than a drift.

**Reversion pins** (the constraint's "would fail if someone reverted"):
`test_to_tensor_downcasts_a_float64_panel` (dtype in, dtype out — fails on any
revert), the `get_default_dtype` test (fails on a hardcoded float32), and
`test_a_shipped_head_trains_on_a_float64_panel[RNNRegressor|RNNClassifier]`
(fails with the real `ValueError` above).

**One layering constraint hit en route, worth recording.** The first draft of
the docstring named `FactorPolars`, `PlBackend` and `StockDataset` to explain
where float64 comes from — and two existing purity guards caught it:
`test_core_layer_purity_no_market_specific_logic` and
`test_base_model_does_not_dispatch_on_concrete_factor_types`. `base/model.py`
must not name a concrete dataset subclass or factor backend even in prose. The
docstring was rewritten to describe the paths generically. The guards worked
exactly as designed; noted here because a future editor will hit the same wall.

## WR-01 — `dl_model/rnn.py:RNNRegressor._init_model`

**Observed RED:** with `hidden_sizes=[6, 5], model_type="lstm"` in the config,
the built module still had `[256, 128, 64]` GRU layers. `RNNClassifier`
(parametrized in alongside) passed, isolating the defect to the regressor.

**Decision: honour it, do not drop it.** The base class calls it by keyword
(`hyperparameters=self.config.hyperparameters`), the abstract signature
declares it, and both sibling heads read it — dropping the parameter would mean
changing the abstract contract and breaking `_init_model_and_optim()`'s call.
"Stop accepting it" was only available to the other three lies this session
removed because those had no other reader. Every value now reads
`.get(..., <the old hardcoded literal>)`, so no config that exists today —
`train_model.py` included — builds a different model.

**Making the dishonest test honest.** `tests/test_dl_models.py` fed
`RNNRegressor` the MLP-shaped `{"hidden_size1": 16, "hidden_size2": 8}` and
passed *because* the argument was ignored; honouring it would have left the
test green via defaults, i.e. still green for the wrong reason. So `_hp_for()`
now returns an RNN-shaped dict for **both** RNN heads (`_RNN_CLASSIFIER_HP` →
`_RNN_HP`) and all six `RNNRegressor` call sites pass it explicitly. Two new
tests assert on the **built module** — `test_rnn_head_hyperparameters_reach_
the_built_module` (parametrized over both RNN heads) and
`test_rnn_regressor_defaults_preserve_the_previously_hardcoded_shape`. Asserting
on the module rather than the config is the whole point: the only observable
difference between "honoured" and "accepted and discarded" is whether the number
shows up in a layer.

**Deliberately NOT done:** `RNNClassifier._init_model` still reads its dict with
`[...]` (KeyError on a missing key) rather than `.get()`. It has no
previously-hardcoded values to use as defaults, so inventing some would be
deciding the user's network shape for them. The review's "three mutually
incompatible contracts" observation is now two, and closing the last gap is a
separate decision. Recorded in the code and in `example/model.md` #12.

## Docs updated (same commits as their fixes)

Following the session convention — rewrite as 「曾经…（已于 2026-09-07 修复）…
现在…」 with the locking test name, appended as a new number so existing
numbering stays stable.

| Doc | Section | Commit |
|---|---|---|
| `example/dataset.md` | 「常见坑」 **#10** (new) — the mask swallowing every warning, the ERROR escalation, and the why-not-raise reasoning | `b579ee1` |
| `example/dataset.md` | 核心契约 #2 — the structural/anomaly distinction now points at #10 for its degenerate case | `b579ee1` |
| `example/dataset.md` | `read()` walkthrough step 7 — `_clean()`'s logging levels | `b579ee1` |
| `example/model.md` | 「常见坑」 **#4** — rewritten from "dtype 基类不管，请在 `_preprocess` 里 `.float()`" to the fixed behaviour, with the real `ValueError` and the two decisions | `417b363` |
| `example/model.md` | the `TinyRegressor` example's `_preprocess` — dropped the now-redundant `.float()` and says why it was there | `417b363` |
| `example/factor.md` | 「常见坑」 **#13** (new) — why the `rel_volume_10` output says `float64`, and why the fix belongs in the model layer, NOT in the Polars graph | `417b363` |
| `example/model.md` | 「常见坑」 **#12** (new) — the accepted-and-ignored `hyperparameters`, incl. the surviving `.get()` vs `[...]` split | `fad4d46` |

## Out of scope, still open

Not fixed, per the constraint. All still stand as written in `REVIEW.md`:

- **WR-02** — early stopping tracks `best_loss` but `_save_model` writes
  whatever the LAST epoch produced, which under early stopping is by
  construction the `patience`-th non-improving epoch. `best_loss` is
  write-only. **This is the most serious of the five left**; it silently
  inverts the mechanism's purpose.
- **WR-03** — `MLPRegressor._val_one_batch` logs to W&B unguarded
  (`AttributeError` on any path that skips `_init_wandb`).
- **WR-04** — `BaseDataset.time_interval` raises a bare `IndexError` on a
  one-timestamp panel.
- **WR-05** — `to_tensor` re-sorts both axes and returns a bare tensor;
  `train_model.py:88` pairs predictions with an unsorted timestamp axis from a
  *different* object. Touched the same method for BL-02 and deliberately did
  not widen scope — the misalignment is at the call site, not in `to_tensor`.
- **WR-06** — `_get_refit_optim` reads `lr_refit`, a `DLConfig`-only field,
  from the base class, and pins a superseded model in memory.
- **WR-07** — `MLPRegressor`'s inference contract contradicts its training
  contract (train reshapes `[T,S,F]→[T,S*F]`, `_predict_nn` does not).
- **IN-01..IN-06** — all untouched.

Nothing dangerous was found beyond what `REVIEW.md` already records.

---

## Follow-up: WR-02 closed (2026-09-07, commit `9993e10`)

**Scope:** WR-02 only. WR-03..WR-07 and IN-01..IN-06 stay open exactly as
listed above — this section supersedes only the WR-02 bullet.

### What was wrong

`best_loss` was write-only in the sense that matters: it gated the patience
counter and nothing ever snapshotted the `state_dict` that produced it.
`_save_model` runs *after* the epoch loop, so the checkpoint held whatever
the last executed epoch left in memory. When early stopping fires, that epoch
is by construction the `patience`-th consecutive epoch of *no* improvement.
Early stopping has two jobs — keep the best, stop wasting time. This did the
second and inverted the first: the optimum it spent its whole budget finding
was the one thing it threw away.

Confirmed by grep before touching anything: `best_loss` appeared at
`base/model.py:490` (init), `:523` (comparison), `:524` (update), and nowhere
else in the repo.

### The fix (`base/model.py`, 27 lines incl. comments)

1. `best_state: dict[str, torch.Tensor] | None = None`, initialised
   unconditionally alongside the other four loop locals (same reason they are:
   the post-loop read is unconditional).
2. On `epoch_val_loss < best_loss`, snapshot
   `{k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}`.
3. Immediately before `_save_model`:
   `if self.config.early_stopping and best_state is not None: self.model.load_state_dict(best_state)`.

**Snapshot cost, stated rather than hidden.** The snapshot is a second copy of
the parameters (not the optimizer state — `self.optim` is dropped after the
loop anyway). It is kept on **CPU** via `.detach().cpu().clone()` rather than
`copy.deepcopy(state_dict())`: `load_state_dict` copies in place, so CPU
tensors load back into a CUDA module without complaint, and a GPU run does not
pay double VRAM for the fix. The price is one host-memory copy of the
parameters, documented in an inline comment at the initialisation site.

**Both exit paths, deliberately.** The restore is placed after the loop, not
inside the `break`, so it covers the early-stopping break *and* the loop
running out of `epochs`. The natural-exit case is restored too, and the
reasoning is written into the code: `early_stopping=True` expresses the intent
"select the checkpoint by validation loss"; whether the loop hit `patience` or
hit the `epochs` ceiling is an accident of the schedule. One config persisting
the best epoch on one exit and the last epoch on the other is incoherent.
The final epoch may or may not be the best one — when it is, the restore is a
no-op; when it is not, restoring is the behaviour the flag asked for.

**The boundary, stated in the code.** `if self.config.early_stopping and ...`.
A run with early stopping OFF never maintains `best_loss` and has expressed no
intent to select by validation loss; silently changing which epoch it persists
would be a behaviour change nobody asked for. It keeps saving the last epoch,
and there is a test that fails if that ever changes.

### Verification — RED first, on the real class

The three tests drive `RecordingRegressor`'s shipped subclass through the real
`model.train()` → `_auto_train` → `_train_dl` → `_save_model` path. No stub
stands in for anything under test.

Two properties make the assertion discriminating, both guarded in-test:

- validation loss follows a **descending-then-ascending script**
  (`3.0, 1.0, 2.0, 2.0, 2.0`), so best (epoch 1) and last (epoch 4) are
  different epochs. The test asserts `best_epoch != last_epoch` explicitly —
  without that, "saved the best" and "saved the last" would be the same claim.
- `_train_one_batch` overwrites every parameter with `float(epoch)` instead of
  taking a gradient step, so the persisted weights **name** their epoch. Real
  SGD would leave the candidates numerically close and reduce the assertion to
  a tolerance argument.
- the assertion loads the `.pth` back off disk (`_saved_weight_value`) rather
  than reading `model.model`. WR-02 is a defect about which weights reach the
  file; an in-memory assertion cannot see it.

A test asserting only "training stopped early" or "a checkpoint exists" passes
with the bug present, so neither was written.

Observed RED against the pre-fix code:

```
FAILED tests/test_model_layer.py::test_early_stopping_saves_the_best_epoch_not_the_waited_out_one
E   AssertionError: checkpoint holds epoch 4's weights; expected the best
E   epoch 1 (the last executed epoch was 4)
E   assert 4.0 == 1.0
E    +  where 1.0 = float(1)

FAILED tests/test_model_layer.py::test_early_stopping_saves_the_best_epoch_when_epochs_run_out
E   AssertionError: assert 2.0 == 1.0
E    +  where 2.0 = _saved_weight_value(PosixPath('.../test_early_stopping_saves_the_1'))

2 failed, 1 passed, 12 deselected
```

The one that passed RED is `test_early_stopping_off_still_saves_the_last_epoch`
— correctly so: it locks the scope boundary, which the pre-fix code already
satisfied by accident. It is there to fail if the fix ever leaks past the
`early_stopping` guard.

All three green after the fix.

### Suite

`491 passed` before → **`494 passed`** after (`~/.venv/bin/python -m pytest -q`,
27.1s). No regressions; the three new tests are the whole delta.

### Docs

`example/model.md` gains 常见坑 **#13** in this session's
「曾经…（已于 2026-09-07 修复）… 现在…」 form, with the epoch/loss trace that
shows why the last epoch is the worst waited-for one, the three sub-points
(CPU snapshot cost, both exit paths, the `early_stopping=False` boundary) and
all three locking test names. Two existing cross-references were updated rather
than left stale: the `_train_dl` pipeline row (line 113) now names the snapshot
and restore steps, and the `_val_one_batch` contract note (line ~196) now says
its return value also decides *which epoch's weights are kept*, not just when
to stop. A pointer was added at the end of 常见坑 #2 (the other early-stopping
entry) rather than renumbering #3..#12 — six cross-references elsewhere in
`example/` address those by number.

### Files

- `base/model.py` — the fix
- `tests/test_model_layer.py` — `ScriptedValLossRegressor`,
  `_saved_weight_value`, three tests (+160 lines)
- `example/model.md` — 常见坑 #13 + three updated cross-references

One commit: `9993e10`. `test.py` (the user's own scratch edit) was neither
touched nor staged.
