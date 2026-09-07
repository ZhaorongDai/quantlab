"""Tests for the concrete `dl_model/` heads (quick task 260907-fl6, batch 2).

Before this file `dl_model/` had ZERO tests. `tests/test_model_layer.py`
(batch 1) exercises `base/model.py` through a purpose-built `RecordingRegressor`
stand-in, which means the SHIPPED heads -- `MLPRegressor`, `RNNRegressor`,
`RNNClassifier` -- were never instantiated by anything in the suite. Both defects
below were invisible for exactly that reason.

What is locked here:

- F  `MLPRegressor` could not be instantiated at all
     (`__abstractmethods__ == {'_val_one_epoch'}`), and even past that its
     `_init_model` was missing the `hyperparameters` keyword the base passes
     and its `_preprocess` called `.fillna()` on a `torch.Tensor`.
- H  every `update()` in `dl_model/` read `self.config.lr_refit`, a field
     `DLConfig` did not define -- `AttributeError` on the first line of the
     online-learning path.

Everything is synthetic, CPU-only and offline: no zarr store, no credentials,
no network, no GPU. `collect()` only ever calls six methods on a factor/label
object, so `FakePanel` stands in for the whole KunQuant + zarr stack.
`wandb.init` is unconditional in `_init_wandb`, so the autouse fixture sets the
documented `WANDB_MODE=disabled` bypass.
"""

import numpy as np
import pytest
import torch
import xarray as xr
from types import SimpleNamespace

from base.config import DLConfig
from dl_model.mlp import MLPRegressor
from dl_model.rnn import RNNRegressor
from dl_model.rnn_classification import RNNClassifier

# --------------------------------------------------------------------------
# Synthetic panel geometry -- deliberately tiny so a real training run is fast
# --------------------------------------------------------------------------

N_TIMES = 60
N_SYMBOLS = 2
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)

TRAIN_START = np.datetime_as_string(TIMES[0], unit="D")
TRAIN_END = np.datetime_as_string(TIMES[39], unit="D")
TEST_START = np.datetime_as_string(TIMES[40], unit="D")
TEST_END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    """`_init_wandb` calls `wandb.init` unconditionally and `DLConfig` has no
    opt-out flag, so the only bypass is the environment variable documented in
    `example/model.md`."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


class FakePanel:
    """A stand-in for a factor/label object.

    Implements only what `BaseModel.collect()` actually calls: `config`,
    `_reset_dataset_config`, `_get_factor_names`, `cal`/`read`,
    `get_features` / `get_labels` and `get_config`.

    Values are pseudo-random rather than constant on purpose: `MLPRegressor`
    logs `r2_score`, which is degenerate (and warns) on a constant target.
    """

    def __init__(self, names: list[str], seed: int):
        rng = np.random.default_rng(seed)
        self.names = list(names)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    rng.standard_normal((N_TIMES, N_SYMBOLS)).astype(
                        "float32"
                    ),
                )
                for name in self.names
            },
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )
        self.config = SimpleNamespace(start_date=None, end_date=None)

    def _reset_dataset_config(self):
        pass

    def _get_factor_names(self):
        return list(self.names)

    def cal(self):
        return self

    def read(self):
        return self

    def get_features(self):
        return self._ds

    def get_labels(self):
        return self._ds

    def get_config(self):
        return {"name": "FakePanel", "factor_names": list(self.names)}


def _make_config(tmp_path, *, epochs: int = 2, **overrides) -> DLConfig:
    params = dict(
        factors=[FakePanel(["f_a", "f_b", "f_c"], seed=1)],
        labels=[FakePanel(["ret_30", "ret_60"], seed=2)],
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
        batch_size=8,
        num_workers=0,
        lr=1e-3,
        early_stopping=False,
        hyperparameters={"hidden_size1": 16, "hidden_size2": 8},
    )
    params.update(overrides)
    return DLConfig(**params)


#: `RNNClassifier._init_model` indexes its `hyperparameters` dict directly
#: (`hyperparameters["hidden_sizes"]` etc.), so it needs a differently-shaped
#: dict from the MLP's. Kept tiny so the CPU runs stay fast.
_RNN_CLASSIFIER_HP = {
    "hidden_sizes": [8, 8],
    "dropout_rates": [0.0, 0.0],
    "hidden_sizes_linear": [8],
    "dropout_rates_linear": [0.0],
    "model_type": "gru",
}


def _hp_for(cls) -> dict:
    if cls is RNNClassifier:
        return {"hyperparameters": _RNN_CLASSIFIER_HP}
    return {}


# --------------------------------------------------------------------------
# F. MLPRegressor must be instantiable AND trainable
# --------------------------------------------------------------------------


def test_mlp_regressor_has_no_unimplemented_abstract_methods(tmp_path):
    """Defect F, first symptom: the class could not even be constructed.

        TypeError: Can't instantiate abstract class MLPRegressor with
        abstract method _val_one_epoch

    Asserting on `__abstractmethods__` rather than only on the constructor
    keeps the failure message pointing at the missing method name.
    """
    assert MLPRegressor.__abstractmethods__ == frozenset()
    model = MLPRegressor(_make_config(tmp_path))
    assert isinstance(model, MLPRegressor)


def test_mlp_regressor_trains_two_epochs_and_predicts(tmp_path):
    """Defect F, the part "it imports" could never have caught.

    This runs the REAL training loop end to end on CPU: `collect()` ->
    `_init_model_and_optim()` -> two epochs of train/val/test -> checkpoint ->
    `predict()`. Each of the three sub-defects fails it at a different point:

    - missing `_val_one_epoch`   -> TypeError at construction
    - `_init_model` without
      `hyperparameters`          -> TypeError inside `_init_model_and_optim()`
    - `_preprocess` on a Tensor  -> AttributeError: 'Tensor' object has no
                                    attribute 'fillna'

    The weight-change assertion is what makes this a training test rather than
    a smoke test: a head that runs the loop without ever stepping the optimizer
    would satisfy every other line here.
    """
    config = _make_config(tmp_path, epochs=2)
    model = MLPRegressor(config)
    model.collect()

    before = None
    model._init_model_and_optim()
    before = model.model.fc1.weight.detach().clone()  # type: ignore

    model.train()

    assert model.model is not None, "train() must leave the model in place"
    after = model.model.fc1.weight.detach()  # type: ignore
    assert not torch.allclose(before, after), (
        "fc1 weights are unchanged after two epochs -- the optimizer never "
        "stepped, so this is not evidence the model trained"
    )

    # Inference through the public path, i.e. through `_preprocess`.
    #
    # The input is FLATTENED to `[num_times, num_symbols * num_features]` on
    # purpose. `MLPRegressor` does that reshape inside `_train_one_epoch` /
    # `_test_one_epoch`, but `BaseModel._predict_nn` hands the tensor straight
    # to the module, and `MLP.forward` is a plain `nn.Linear` stack with no
    # reshape of its own. So the head's inference contract really is "pass the
    # flat matrix"; asserting it here is what keeps that from being discovered
    # again at a call site. (Left as-is deliberately: closing the gap means
    # either editing `base/model.py` -- batch 1's file -- or changing what the
    # public `MLP` module accepts.)
    x = torch.from_numpy(
        np.random.default_rng(0)
        .standard_normal((5, N_SYMBOLS * 3))
        .astype("float32")
    )
    out = model.predict(x)
    assert out.shape == (5, N_SYMBOLS * 2)
    assert torch.isfinite(out).all()


def test_mlp_preprocess_takes_a_tensor_and_scrubs_nan(tmp_path):
    """Defect F, third symptom, in isolation.

    `BaseModel._preprocess`'s contract is `(Tensor) -> Tensor` -- both call
    sites (`_train_dl`'s four tensors, `_predict_nn`'s inference input) pass
    the output of `to_tensor()`. The old body was `data.fillna(0.0)`.
    """
    model = MLPRegressor(_make_config(tmp_path))
    noisy = torch.tensor([[1.0, float("nan")], [float("nan"), 4.0]])
    cleaned = model._preprocess(noisy)
    assert isinstance(cleaned, torch.Tensor)
    assert torch.equal(cleaned, torch.tensor([[1.0, 0.0], [0.0, 4.0]]))


def test_mlp_val_one_epoch_returns_a_floatable_loss(tmp_path):
    """Defect F meets batch 1's tightened contract.

    `base/model.py`'s epoch loop now does
    `val_loss_sum += float(val_loss) * batch_samples` for every validation
    batch, so a `_val_one_epoch` that returns `None` raises `TypeError` on
    epoch 0 whether or not early stopping is on.
    """
    model = MLPRegressor(_make_config(tmp_path))
    model.collect()
    model._init_model_and_optim()
    model._init_wandb("quantlab-test", "mlp-val")

    x = torch.zeros((4, N_SYMBOLS, 3))
    y = torch.zeros((4, N_SYMBOLS, 2))
    loss = model._val_one_epoch(0, x, y)
    assert float(loss) >= 0.0
    assert not loss.requires_grad


# --------------------------------------------------------------------------
# H. DLConfig.lr_refit
# --------------------------------------------------------------------------


def test_dlconfig_defines_lr_refit_defaulting_to_disabled():
    """Defect H: both `update()` implementations open with

        if self.config.lr_refit <= 0.0:
            return

    but `DLConfig` had no such field, so the online-learning path died with
    `AttributeError: 'DLConfig' object has no attribute 'lr_refit'` on its
    very first line. The default is 0.0 -- the value the guard the original
    author wrote already treats as "online updating is off".
    """
    config = DLConfig(
        factors=[],
        labels=[],
        model_save_dir="/tmp/does-not-matter",
        factor_data_strategy="cal",
        label_data_strategy="cal",
    )
    assert config.lr_refit == 0.0
    assert "lr_refit" in config.to_dict()


@pytest.mark.parametrize("cls", [RNNRegressor, RNNClassifier])
def test_update_is_a_noop_under_the_default_config(tmp_path, cls):
    """`update()` must be REACHABLE, and by default must do nothing.

    Both surviving `update()` implementations -- `dl_model/rnn.py`'s and
    `dl_model/rnn_classification.py`'s -- are covered, because both read
    `config.lr_refit` on their first line.
    """
    model = cls(_make_config(tmp_path, **_hp_for(cls)))
    model.collect()
    model._init_model_and_optim()
    before = [p.detach().clone() for p in model.model.parameters()]  # type: ignore

    x = torch.zeros((4, N_SYMBOLS, 3))
    y = torch.zeros((4, N_SYMBOLS, 2))
    model.update(x, y)  # must not raise AttributeError on config.lr_refit

    after = list(model.model.parameters())  # type: ignore
    assert all(torch.equal(b, a) for b, a in zip(before, after)), (
        "lr_refit defaults to 0.0, so update() must return before touching "
        "any parameter"
    )


def test_update_steps_the_model_when_lr_refit_is_positive(tmp_path):
    """The other half of H: the field is a real knob, not a placeholder.

    Without this, `lr_refit = 0.0` could be satisfied by a field nothing ever
    reads.
    """
    model = RNNRegressor(_make_config(tmp_path, lr_refit=1e-2))
    model.collect()
    model._init_model_and_optim()
    before = [p.detach().clone() for p in model.model.parameters()]  # type: ignore

    rng = np.random.default_rng(3)
    x = torch.from_numpy(
        rng.standard_normal((4, N_SYMBOLS, 3)).astype("float32")
    )
    y = torch.from_numpy(
        rng.standard_normal((4, N_SYMBOLS, 2)).astype("float32")
    )
    model.update(x, y)

    after = list(model.model.parameters())  # type: ignore
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "lr_refit > 0 must actually take an optimizer step"
    )
