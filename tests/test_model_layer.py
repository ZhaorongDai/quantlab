"""First automated tests for the model layer (`base/model.py:BaseModel`).

Before quick task 260907-fl6 there were ZERO tests for `base/model.py`. Every
test in this file was written RED against the pre-fix code and locks one of the
six defects written up in `example/model.md` ("常见坑" / "已知的不完整之处"):

- A  `early_stopping=False` raised `UnboundLocalError` -- `early_stopping`,
     `best_loss`, `patience` and `counter` were initialised INSIDE
     `if self.config.early_stopping:` but `if early_stopping: break` at the end
     of the epoch loop read the name unconditionally.
- B  the early-stopping counter incremented once per validation BATCH, so
     `early_stopping_patience` silently meant "N consecutive bad batches" and
     could fire inside a single epoch.
- C  `.sortby(["timestamp", "symbol", "variable"])` ordered the tensor's last
     axis ALPHABETICALLY, so `y[:, :, 0]` -- the primary target for
     `RNNClassifier` -- was `ret_120`, not the `ret_30` `train_model.py` lists
     first.
- D  `predict()` ran the module in training mode and built a graph: no
     `model.eval()`, no `torch.no_grad()`, so inference with dropout was
     non-deterministic.
- L1 the validation split dropped the row at index `train_split`
     (`train_x_t_all[train_split + 1:]`).
- L2 `_train_dl` ended with `del self.model`, so `predict()` right after
     `train()` was impossible.

Batch 3 added, for defects surfaced by the same doc pass:

- `num_null` ended in `.values[0]` on a 0-d array, so every read raised
     `IndexError` -- a property that is annotated `-> int` and recommended by
     `example/model.md` as the pre-training missing-value check.
- the vecbt skeleton (`_do_vecbt`, `_vecbt`, `_train_dl(backtest=...)`)
     did nothing, silently. It is KEPT -- Phase 6 owns end-to-end backtesting
     and the hook must eventually serve the `MLConfig`/xgboost path too -- but
     it now says so instead of returning `None`.

Batch 3 also renamed the three per-batch hooks from `_*_one_epoch` to
`_*_one_batch` (defect B above is what that name cost) and the two collectors
from `_get_*_batch` to `_collect_all_*`.

Everything here is synthetic, CPU-only and offline: no zarr store, no
credentials, no network, no GPU. `collect()` only ever calls six methods on a
factor/label object, so `FakePanel` below stands in for the whole
KunQuant + zarr stack. `wandb.init` is unconditional in `_init_wandb`, so the
autouse `_offline_wandb` fixture sets the documented `WANDB_MODE=disabled`
bypass.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr
from types import SimpleNamespace

from quantlab.base.config import DLConfig
from quantlab.base.model import BaseModel

# --------------------------------------------------------------------------
# Synthetic panel geometry
# --------------------------------------------------------------------------

N_TIMES = 130
N_SYMBOLS = 3
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)

#: 100 training timestamps, 30 test timestamps.
TRAIN_START = np.datetime_as_string(TIMES[0], unit="D")
TRAIN_END = np.datetime_as_string(TIMES[99], unit="D")
TEST_START = np.datetime_as_string(TIMES[100], unit="D")
TEST_END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")
N_TRAIN_TIMES = 100


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    """`_init_wandb` calls `wandb.init` unconditionally and `DLConfig` has no
    opt-out flag, so the only bypass is the environment variable documented in
    `example/model.md`."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


class FakePanel:
    """A stand-in for a factor/label object.

    Implements only what the model layer actually calls: the `config`
    attribute, `_reset_dataset_config`, `_get_factor_names`, `cal`,
    `get_features` / `get_labels` and `get_config`.

    `values` maps a variable name to the CONSTANT that variable is filled
    with. A constant panel is what makes the defect-C assertion possible: the
    value in column *i* of the tensor identifies which variable landed there,
    independently of row order or `shuffle=True`.
    """

    def __init__(self, values: dict[str, float]):
        self.values = dict(values)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    np.full((N_TIMES, N_SYMBOLS), const, dtype="float32"),
                )
                for name, const in self.values.items()
            },
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )
        self.config = SimpleNamespace(start_date=None, end_date=None)

    def _reset_dataset_config(self):
        pass

    def _get_factor_names(self):
        return list(self.values)

    def cal(self):
        return self

    def read(self):
        return self

    def get_features(self):
        return self._ds

    def get_labels(self):
        return self._ds

    def get_config(self):
        return {"name": "FakePanel", "factor_names": list(self.values)}


class RecordingRegressor(BaseModel):
    """Minimal concrete `BaseModel` that records everything the epoch loop
    hands it, so tests can assert on the loop's behaviour rather than on loss
    values."""

    def __init__(self, config: DLConfig):
        super().__init__(config)
        self.criterion = nn.MSELoss()
        self.train_epochs: list[int] = []
        self.val_epochs: list[int] = []
        self.test_epochs: list[int] = []
        self.train_rows = 0
        self.val_rows = 0
        self.seen_x: list[list[float]] = []
        self.seen_y: list[list[float]] = []

    def _init_model(
        self, num_symbols, num_features, num_labels, hyperparameters
    ):
        hidden = hyperparameters.get("hidden", 8)
        dropout = hyperparameters.get("dropout", 0.0)
        return nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def _init_optim(self, model):
        return torch.optim.SGD(model.parameters(), lr=self.config.lr)

    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(data, nan=0.0).float()

    def _train_one_batch(self, epoch, x, y):
        self.train_epochs.append(epoch)
        self.train_rows += int(x.shape[0])
        self.seen_x.append([float(v) for v in x[0, 0]])
        self.seen_y.append([float(v) for v in y[0, 0]])
        self.optim.zero_grad()
        loss = self.criterion(self.model(x), y)
        loss.backward()
        self.optim.step()
        return loss

    def _val_one_batch(self, epoch, x, y):
        self.val_epochs.append(epoch)
        self.val_rows += int(x.shape[0])
        return self.criterion(self.model(x), y)

    def _test_one_batch(self, epoch, x, y):
        self.test_epochs.append(epoch)
        return self.criterion(self.model(x), y)


class FlatValLossRegressor(RecordingRegressor):
    """Validation loss is a CONSTANT, so it never improves after the first
    observation. Used by the defect-B test: how many epochs run before early
    stopping fires is then a pure function of what the patience counter counts.
    """

    def _val_one_batch(self, epoch, x, y):
        self.val_epochs.append(epoch)
        self.val_rows += int(x.shape[0])
        return torch.tensor(1.0)


class HolePanel(FakePanel):
    """`FakePanel` with a KNOWN number of NaNs punched into its first variable.

    `FakePanel` fills every cell with a constant, so a panel built from it has
    exactly zero missing values -- which cannot tell a working `num_null` apart
    from one that always answers 0.
    """

    def __init__(self, values: dict[str, float], n_holes: int):
        super().__init__(values)
        first = next(iter(values))
        arr = self._ds[first].values.copy().reshape(-1)
        assert n_holes <= arr.size
        arr[:n_holes] = np.nan
        self._ds[first] = (
            ("timestamp", "symbol"),
            arr.reshape(N_TIMES, N_SYMBOLS),
        )
        self.n_holes = n_holes


def _make_config(
    tmp_path,
    *,
    factor_values: dict[str, float],
    label_values: dict[str, float],
    epochs: int = 2,
    batch_size: int = 64,
    early_stopping: bool = True,
    early_stopping_patience: int = 5,
    hyperparameters: dict | None = None,
    factors: list | None = None,
    labels: list | None = None,
) -> DLConfig:
    return DLConfig(
        factors=factors if factors is not None else [FakePanel(factor_values)],
        labels=labels if labels is not None else [FakePanel(label_values)],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=TRAIN_START,
        end_date=TEST_END,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=0,
        lr=1e-2,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        hyperparameters=hyperparameters or {"hidden": 8},
    )


# --------------------------------------------------------------------------
# A. early_stopping=False must not raise
# --------------------------------------------------------------------------


def test_early_stopping_disabled_runs_all_epochs(tmp_path):
    """Defect A: `early_stopping=False` crashed with

        UnboundLocalError: cannot access local variable 'early_stopping'
        where it is not associated with a value

    because the four early-stopping locals were only bound inside
    `if self.config.early_stopping:`. An ordinary config with early stopping
    off could not train at all.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
        epochs=3,
        early_stopping=False,
    )
    model = RecordingRegressor(cfg)
    model.collect()

    model.train()

    assert sorted(set(model.train_epochs)) == [0, 1, 2]


# --------------------------------------------------------------------------
# B. patience counts epochs, not validation batches
# --------------------------------------------------------------------------


def test_early_stopping_patience_counts_epochs_not_batches(tmp_path):
    """Defect B: `counter += 1` lived inside the validation BATCH loop.

    The config below produces MORE THAN ONE validation batch per epoch, which
    is what makes the two behaviours distinguishable at all:

    - counting batches: epoch 0 alone burns the whole patience budget
      (batch 0 sets `best_loss`, batches 1-3 each bump the counter), so
      training stops after 1 epoch;
    - counting epochs: epoch 0 sets `best_loss`, epochs 1-3 each bump the
      counter, so training stops after 4 epochs.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
        epochs=10,
        batch_size=5,
        early_stopping=True,
        early_stopping_patience=3,
    )
    model = FlatValLossRegressor(cfg)
    model.collect()

    model.train()

    epochs_run = sorted(set(model.val_epochs))
    val_batches_in_first_epoch = model.val_epochs.count(epochs_run[0])
    # Guard: a single validation batch per epoch could not tell the two
    # behaviours apart, so this test would pass for the wrong reason.
    assert val_batches_in_first_epoch > 1, (
        "test is not discriminating: needs >1 validation batch per epoch, "
        f"got {val_batches_in_first_epoch}"
    )
    assert epochs_run == [0, 1, 2, 3]


# --------------------------------------------------------------------------
# C. the tensor's last axis follows the caller's declared order
# --------------------------------------------------------------------------


def test_tensor_variable_axis_follows_declared_order(tmp_path):
    """Defect C: `.sortby([..., "variable"])` ordered the last axis by NAME.

    Both name lists below are deliberately NON-alphabetical, so a tensor built
    in alphabetical order cannot pass by accident:

    - factors declared `zeta, alpha, mid`  -> alphabetical is `alpha, mid, zeta`
    - labels  declared `ret_30, ret_60, ret_120`
      -> alphabetical is `ret_120, ret_30, ret_60`

    The label case is the one with real consequences: `RNNClassifier` treats
    `y[:, :, 0]` as the primary target, and `train_model.py` declares
    `ret_30` first.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"zeta": 1.0, "alpha": 2.0, "mid": 3.0},
        label_values={"ret_30": 30.0, "ret_60": 60.0, "ret_120": 120.0},
        epochs=1,
    )
    model = RecordingRegressor(cfg)
    model.collect()

    assert model.get_factor_names() == ["zeta", "alpha", "mid"]
    assert model.get_label_names() == ["ret_30", "ret_60", "ret_120"]

    model.train()

    assert model.seen_x, "no training batch was observed"
    for row in model.seen_x:
        assert row == [1.0, 2.0, 3.0], (
            "x last axis is not in declared factor order "
            "['zeta', 'alpha', 'mid']"
        )
    for row in model.seen_y:
        assert row == [30.0, 60.0, 120.0], (
            "y last axis is not in declared label order "
            "['ret_30', 'ret_60', 'ret_120']"
        )


# --------------------------------------------------------------------------
# D. predict() runs in eval mode, under no_grad
# --------------------------------------------------------------------------


def test_predict_runs_in_eval_mode_without_grad(tmp_path):
    """Defect D: `_predict_nn` never called `model.eval()` and was not wrapped
    in `torch.no_grad()`. A freshly built `nn.Module` defaults to training
    mode, so inference ran with dropout ACTIVE (0.9 here, 0.5 in
    `train_model.py`) and returned a different answer every call, while also
    retaining the whole autograd graph.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0, "f2": 3.0},
        label_values={"y0": 0.5},
        hyperparameters={"hidden": 16, "dropout": 0.9},
    )
    model = RecordingRegressor(cfg)
    model.collect()
    model._init_model_and_optim()
    model.model.train()  # the state `load()` leaves the module in
    assert model.model.training is True

    x = torch.randn(4, model.num_symbols, model.num_factors)
    first = model.predict(x)

    assert model.model.training is False, (
        "predict() left the module in training mode"
    )
    assert first.requires_grad is False, (
        "predict() built an autograd graph"
    )

    second = model.predict(x)
    assert torch.equal(first, second), (
        "predict() is non-deterministic -- dropout is still active"
    )


# --------------------------------------------------------------------------
# L1. the validation split keeps every row
# --------------------------------------------------------------------------


def test_val_split_keeps_every_training_row(tmp_path):
    """Defect L1: `val_x_t = train_x_t_all[train_split + 1:]` silently dropped
    the timestamp at index `train_split` -- it was in neither split."""
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0},
        label_values={"y0": 0.5},
        epochs=1,
    )
    model = RecordingRegressor(cfg)
    model.collect()

    model.train()

    assert model.train_rows + model.val_rows == N_TRAIN_TIMES


# --------------------------------------------------------------------------
# L2. the trained model survives train()
# --------------------------------------------------------------------------


def test_model_is_usable_immediately_after_train(tmp_path):
    """Defect L2: `_train_dl` ended with `del self.model`, so `predict()` right
    after `train()` raised

        ValueError: Model not initialized, please call load() or train() first

    even though `train()` had just been called."""
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
        epochs=1,
    )
    model = RecordingRegressor(cfg)
    model.collect()

    model.train()

    assert model.model is not None
    out = model.predict(
        torch.randn(2, model.num_symbols, model.num_factors)
    )
    assert out.shape == (2, model.num_symbols, model.num_labels)


# --------------------------------------------------------------------------
# num_null: every read raised
# --------------------------------------------------------------------------


def test_num_null_counts_missing_cells_and_returns_an_int(tmp_path):
    """`BaseModel.num_null` ended in `.values[0]`, but the `.sum()` before it
    produces a 0-d array, so EVERY read raised

        IndexError: too many indices for array: array is 0-dimensional,
        but 1 were indexed

    The property is annotated `-> int` and `example/model.md` recommends it as
    the pre-training missing-value check, so it was documented, advertised and
    unusable.

    The two panels punch a different number of holes so a fix that reads only
    one of them, or that stops at the per-variable `.sum()` (a Dataset, not a
    scalar), cannot pass.
    """
    factor_holes, label_holes = 7, 4
    cfg = _make_config(
        tmp_path,
        factor_values={},
        label_values={},
        factors=[HolePanel({"f0": 1.0, "f1": 2.0}, n_holes=factor_holes)],
        labels=[HolePanel({"y0": 0.5}, n_holes=label_holes)],
    )
    model = RecordingRegressor(cfg)
    model.collect()

    n = model.num_null

    assert n == factor_holes + label_holes
    assert isinstance(n, int), f"num_null is annotated -> int, got {type(n)}"


def test_num_null_is_zero_on_a_dense_panel(tmp_path):
    """The counterpart to the test above: a panel with no holes must report 0
    rather than raise. Without this, a `num_null` that returned a constant
    `len(...)` of something would still pass the counting test by accident on
    one specific geometry.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
    )
    model = RecordingRegressor(cfg)
    model.collect()

    assert model.num_null == 0


# --------------------------------------------------------------------------
# The vecbt skeleton: honest, not silent
# --------------------------------------------------------------------------
#
# `_do_vecbt`, `_vecbt`, `RNNClassifier._vecbt` and `_train_dl(backtest=...)`
# are four half-built pieces that never connect to each other. They are KEPT
# (Phase 6 owns end-to-end backtesting, and the hook has to serve the intended
# `MLConfig`/xgboost path as well as torch), but a caller must never be able to
# ask for a backtest and get silence.


def test_train_dl_rejects_a_truthy_backtest_flag(tmp_path):
    """`_train_dl(backtest=...)` was declared and then never referenced -- a
    caller asking for a backtest trained a model and got no backtest, with no
    warning anywhere.

    The rejection has to happen BEFORE training, which is what
    `model.train_epochs == []` pins: on a real run this parameter is passed
    once and the training it precedes takes hours, so discovering the gap
    afterwards is barely better than not discovering it.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
        epochs=2,
    )
    model = RecordingRegressor(cfg)
    model.collect()

    with pytest.raises(NotImplementedError, match="Phase 6"):
        model._train_dl(
            project_name="p",
            experiment_name="e",
            model_name="m.pth",
            backtest=True,
        )

    assert model.train_epochs == [], (
        "backtest=True must be rejected before any training runs"
    )


def test_train_dl_still_trains_when_backtest_is_falsy(tmp_path):
    """The guard above must not turn the default path into a landmine:
    `backtest=False` (the default, and what `_auto_train` passes) still trains
    normally."""
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
        epochs=2,
    )
    model = RecordingRegressor(cfg)
    model.collect()

    model.train()

    assert sorted(set(model.train_epochs)) == [0, 1]


def test_do_vecbt_says_it_is_unbuilt_instead_of_returning_none(tmp_path):
    """`_do_vecbt` read the backtest dataset, computed a local `price` and then
    the function ended -- no return, no use, and nothing in the repository ever
    called it. A caller who found it got `None` back and no indication that
    nothing had happened.

    Its two existing argument checks are deliberately kept in front of the
    raise: a caller who has not configured `backtest_data` at all should still
    hear about that first, since that is a mistake they can fix today.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
    )
    model = RecordingRegressor(cfg)

    with pytest.raises(ValueError, match="Backtest dataset must be specified"):
        model._do_vecbt()

    model.config.backtest_data = SimpleNamespace(
        config=SimpleNamespace(symbols=["S0"])
    )
    with pytest.raises(NotImplementedError, match="Phase 6"):
        model._do_vecbt()


def test_vecbt_stub_is_still_a_stub_and_names_phase_6(tmp_path):
    """`BaseModel._vecbt` stays -- it is where a shared torch/non-torch
    backtest hook belongs -- but its bare `raise NotImplementedError` said
    nothing about who owns it or when."""
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
    )
    model = RecordingRegressor(cfg)

    with pytest.raises(NotImplementedError, match="Phase 6"):
        model._vecbt(prices=None, signals=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# WR-02: the checkpoint must hold the BEST epoch's weights, not the last
# --------------------------------------------------------------------------


class ScriptedValLossRegressor(RecordingRegressor):
    """A regressor whose validation loss follows a SCRIPT and whose weights
    identify the epoch that produced them.

    Two properties make the WR-02 assertion possible at all:

    - `_val_one_batch` returns `val_loss_script[epoch]`, a constant within the
      epoch, so the epoch-level weighted mean is exactly that number. The
      script can therefore be made to descend and then ascend, which is the
      only shape in which "best" and "last" are different epochs.
    - `_train_one_batch` overwrites EVERY parameter with `float(epoch)`
      instead of taking a gradient step. A `state_dict` read back from disk
      then names, unambiguously, which epoch's weights were persisted. Real
      SGD would leave the two candidate epochs numerically close and the
      assertion would degrade into a tolerance argument.
    """

    val_loss_script: list[float] = []

    def _train_one_batch(self, epoch, x, y):
        self.train_epochs.append(epoch)
        self.train_rows += int(x.shape[0])
        with torch.no_grad():
            for p in self.model.parameters():  # type: ignore[union-attr]
                p.fill_(float(epoch))
        return torch.tensor(0.0)

    def _val_one_batch(self, epoch, x, y):
        self.val_epochs.append(epoch)
        self.val_rows += int(x.shape[0])
        return torch.tensor(self.val_loss_script[epoch])

    def _test_one_batch(self, epoch, x, y):
        self.test_epochs.append(epoch)
        return torch.tensor(0.0)


def _saved_weight_value(tmp_path) -> float:
    """Load the checkpoint `_save_model` actually wrote and return the single
    constant every parameter holds.

    Asserting on the FILE rather than on `model.model` in memory is the whole
    point: `_save_model` is the last thing `_train_dl` does, and WR-02 is a
    defect about which weights reach the disk.
    """
    checkpoints = sorted((tmp_path / "ckpt").rglob("*.pth"))
    assert len(checkpoints) == 1, f"expected one checkpoint, got {checkpoints}"
    state = torch.load(checkpoints[0], weights_only=True)
    values = {float(v.flatten()[0]) for v in state.values()}
    assert len(values) == 1, f"parameters disagree on their value: {values}"
    return values.pop()


def test_early_stopping_saves_the_best_epoch_not_the_waited_out_one(tmp_path):
    """WR-02: `best_loss` gated the patience counter and nothing else.

    Nothing ever snapshotted the weights that produced it, and `_save_model`
    runs AFTER the epoch loop -- so the checkpoint held whatever the last
    executed epoch left in memory. When early stopping fires, that epoch is by
    construction the `patience`-th consecutive epoch of NO improvement, i.e.
    the mechanism threw away the optimum it had just spent its budget finding.

    Script below: epoch 0 loss 3.0, epoch 1 loss 1.0 (the best), then 2.0 for
    epochs 2-4. With `patience=3` the counter fills on epochs 2, 3, 4 and the
    loop breaks at epoch 4.

    Against the pre-fix code this asserted 1.0 and got 4.0.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
        epochs=10,
        early_stopping=True,
        early_stopping_patience=3,
    )
    model = ScriptedValLossRegressor(cfg)
    model.val_loss_script = [3.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    model.collect()

    model.train()

    epochs_run = sorted(set(model.val_epochs))
    # Guards against passing for the wrong reason: early stopping must
    # actually have fired, and the best epoch must NOT be the last one --
    # otherwise "saved the best" and "saved the last" are the same assertion.
    assert epochs_run == [0, 1, 2, 3, 4], epochs_run
    best_epoch, last_epoch = 1, 4
    assert best_epoch != last_epoch

    saved = _saved_weight_value(tmp_path)
    assert saved == float(best_epoch), (
        f"checkpoint holds epoch {saved:.0f}'s weights; expected the best "
        f"epoch {best_epoch} (the last executed epoch was {last_epoch})"
    )


def test_early_stopping_saves_the_best_epoch_when_epochs_run_out(tmp_path):
    """The same guarantee on the OTHER exit path: the loop finishing its
    `epochs` budget without the patience counter ever filling.

    `early_stopping=True` is a request to select the checkpoint by validation
    loss; which of the two exits the loop happened to take is an accident of
    the schedule, and it would be incoherent for the same config to persist
    the best epoch when it breaks and the last epoch when it does not.

    Script: 3.0, 1.0, 2.0 over exactly 3 epochs with `patience=5`, so the
    counter reaches 1 and the loop simply runs out. Best is epoch 1, last is
    epoch 2.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
        epochs=3,
        early_stopping=True,
        early_stopping_patience=5,
    )
    model = ScriptedValLossRegressor(cfg)
    model.val_loss_script = [3.0, 1.0, 2.0]
    model.collect()

    model.train()

    assert sorted(set(model.val_epochs)) == [0, 1, 2]
    assert _saved_weight_value(tmp_path) == 1.0


def test_early_stopping_off_still_saves_the_last_epoch(tmp_path):
    """The scope boundary, locked.

    `best_loss` is only maintained under `if self.config.early_stopping:`, and
    a run with early stopping OFF has expressed no intent to select a
    checkpoint by validation loss. Restoring a "best" epoch for such a run
    would be a silent behaviour change nobody asked for, so it must keep
    persisting the last epoch -- here epoch 2, even though epoch 1 scored
    better on the very same script the two tests above select by.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
        epochs=3,
        early_stopping=False,
    )
    model = ScriptedValLossRegressor(cfg)
    model.val_loss_script = [3.0, 1.0, 2.0]
    model.collect()

    model.train()

    assert sorted(set(model.val_epochs)) == [0, 1, 2]
    assert _saved_weight_value(tmp_path) == 2.0
