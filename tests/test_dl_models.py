"""Tests for the concrete `dl_model/` heads (quick task 260907-fl6, batch 2).

Before this file `dl_model/` had ZERO tests. `tests/test_model_layer.py`
(batch 1) exercises `base/model.py` through a purpose-built `RecordingRegressor`
stand-in, which means the SHIPPED heads -- `MLPRegressor`, `RNNRegressor`,
`RNNClassifier` -- were never instantiated by anything in the suite. Every
defect below was invisible for exactly that reason.

What is locked here:

- F  `MLPRegressor` could not be instantiated at all
     (`__abstractmethods__ == {'_val_one_batch'}`), and even past that its
     `_init_model` was missing the `hyperparameters` keyword the base passes
     and its `_preprocess` called `.fillna()` on a `torch.Tensor`.
- H  every `update()` in `dl_model/` read `self.config.lr_refit`, a field
     `DLConfig` did not define -- `AttributeError` on the first line of the
     online-learning path.
- The refit optimizer: `update()` built a fresh `AdamW` on every call, so
     Adam's moment estimates were zeroed every step and online training
     silently degraded to SGD with an odd warmup.
- Rule-1 deviation: `RNNRegressor._val_one_batch` returned `None`, which
     batch 1's per-epoch loss accumulation (`float(val_loss)`) turned into a
     hard `TypeError` on epoch 0.
- (batch 3) `RNNClassifier._vecbt` computed four pandas Series and then the
     file ended -- no return, no vectorbt call, no error. The method is kept,
     but it now raises instead of handing back `None`.

Everything is synthetic, CPU-only and offline: no zarr store, no credentials,
no network, no GPU. `collect()` only ever calls six methods on a factor/label
object, so `FakePanel` stands in for the whole KunQuant + zarr stack.
`wandb.init` is unconditional in `_init_wandb`, so the autouse fixture sets the
documented `WANDB_MODE=disabled` bypass.
"""

import numpy as np
import pandas as pd
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

    `via_pandas=True` builds the panel the way the REAL ingest builds one --
    long-format `DataFrame.set_index(["timestamp", "symbol"]).to_xarray()`,
    exactly `dataset/*._raw_data_to_xr()` -- and does NOT cast. numpy hands
    back `float64` and nothing downgrades it, which is what both non-KunQuant
    data paths actually produce (`FactorPolars`/`PlBackend`, and
    `StockDataset`'s Tiingo parquet). The default `False` keeps the
    `.astype("float32")` every pre-existing test in this file relies on.
    """

    def __init__(self, names: list[str], seed: int, via_pandas: bool = False):
        rng = np.random.default_rng(seed)
        self.names = list(names)
        if via_pandas:
            self._ds = self._panel_via_pandas(rng)
        else:
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

    def _panel_via_pandas(self, rng) -> xr.Dataset:
        frame = pd.DataFrame(
            {
                "timestamp": np.repeat(TIMES, N_SYMBOLS),
                "symbol": np.tile(SYMBOLS, N_TIMES),
                **{
                    name: rng.standard_normal(N_TIMES * N_SYMBOLS)
                    for name in self.names
                },
            }
        )
        return frame.set_index(["timestamp", "symbol"]).to_xarray()

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


def _make_config(
    tmp_path, *, epochs: int = 2, via_pandas: bool = False, **overrides
) -> DLConfig:
    params = dict(
        factors=[
            FakePanel(["f_a", "f_b", "f_c"], seed=1, via_pandas=via_pandas)
        ],
        labels=[
            FakePanel(["ret_30", "ret_60"], seed=2, via_pandas=via_pandas)
        ],
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


#: Both RNN heads build a `ModelRCrypto`, which needs a differently-shaped
#: hyperparameter dict from the MLP's. Kept tiny so the CPU runs stay fast.
#:
#: `RNNRegressor` used to be handed the MLP-shaped
#: `{"hidden_size1": 16, "hidden_size2": 8}` and passed anyway -- **because
#: its `_init_model` ignored the argument entirely** (WR-01). Every RNN test
#: now passes an RNN-shaped dict explicitly, so a green run means the head
#: was configured, not that the configuration was discarded.
_RNN_HP = {
    "hidden_sizes": [8, 8],
    "dropout_rates": [0.0, 0.0],
    "hidden_sizes_linear": [8],
    "dropout_rates_linear": [0.0],
    "model_type": "gru",
}


def _hp_for(cls) -> dict:
    if cls in (RNNClassifier, RNNRegressor):
        return {"hyperparameters": _RNN_HP}
    return {}


# --------------------------------------------------------------------------
# F. MLPRegressor must be instantiable AND trainable
# --------------------------------------------------------------------------


def test_mlp_regressor_has_no_unimplemented_abstract_methods(tmp_path):
    """Defect F, first symptom: the class could not even be constructed.

        TypeError: Can't instantiate abstract class MLPRegressor with
        abstract method _val_one_batch

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

    - missing `_val_one_batch`   -> TypeError at construction
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
    # purpose. `MLPRegressor` does that reshape inside `_train_one_batch` /
    # `_test_one_batch`, but `BaseModel._predict_nn` hands the tensor straight
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


def test_mlp_val_one_batch_returns_a_floatable_loss(tmp_path):
    """Defect F meets batch 1's tightened contract.

    `base/model.py`'s epoch loop now does
    `val_loss_sum += float(val_loss) * batch_samples` for every validation
    batch, so a `_val_one_batch` that returns `None` raises `TypeError` on
    epoch 0 whether or not early stopping is on.
    """
    model = MLPRegressor(_make_config(tmp_path))
    model.collect()
    model._init_model_and_optim()
    model._init_wandb("quantlab-test", "mlp-val")

    x = torch.zeros((4, N_SYMBOLS, 3))
    y = torch.zeros((4, N_SYMBOLS, 2))
    loss = model._val_one_batch(0, x, y)
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
    model = RNNRegressor(
        _make_config(tmp_path, lr_refit=1e-2, **_hp_for(RNNRegressor))
    )
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


# --------------------------------------------------------------------------
# Rule-1 deviation: RNNRegressor._val_one_batch returned None
# --------------------------------------------------------------------------


def test_rnn_regressor_val_one_batch_returns_a_floatable_loss(tmp_path):
    """`_val_one_batch` is declared `-> torch.Tensor` on `BaseModel`, and
    `RNNRegressor`'s implementation returned nothing at all.

    Batch 1 made that fatal rather than merely wrong: the epoch loop now runs
    `float(val_loss)` for EVERY validation batch, unconditionally, so
    `RNNRegressor.train()` raised

        TypeError: float() argument must be a string or a real number,
        not 'NoneType'

    on epoch 0. `dl_model/rnn_classification.py` already returned
    `val_loss.detach()`; this aligns the sibling.
    """
    model = RNNRegressor(_make_config(tmp_path, **_hp_for(RNNRegressor)))
    model.collect()
    model._init_model_and_optim()
    model._init_wandb("quantlab-test", "rnn-val")

    rng = np.random.default_rng(4)
    x = torch.from_numpy(
        rng.standard_normal((4, N_SYMBOLS, 3)).astype("float32")
    )
    y = torch.from_numpy(
        rng.standard_normal((4, N_SYMBOLS, 2)).astype("float32")
    )
    # Called inside `no_grad` because that is how `_train_dl` calls it: the
    # whole validation block runs under `torch.no_grad()`.
    with torch.no_grad():
        loss = model._val_one_batch(0, x, y)
    assert loss is not None, "_val_one_batch returned None"
    assert float(loss) >= 0.0


# --------------------------------------------------------------------------
# The refit optimizer must PERSIST across update() calls
# --------------------------------------------------------------------------


def _refit_ready(cls, tmp_path, lr_refit: float = 1e-2):
    model = cls(_make_config(tmp_path, lr_refit=lr_refit, **_hp_for(cls)))
    model.collect()
    model._init_model_and_optim()
    rng = np.random.default_rng(7)
    x = torch.from_numpy(
        rng.standard_normal((4, N_SYMBOLS, 3)).astype("float32")
    )
    y = torch.from_numpy(
        rng.standard_normal((4, N_SYMBOLS, 2)).astype("float32")
    )
    return model, x, y


@pytest.mark.parametrize("cls", [RNNRegressor, RNNClassifier])
def test_refit_optimizer_state_accumulates_across_update_calls(tmp_path, cls):
    """Every `update()` used to build a brand-new AdamW:

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.config.lr_refit)

    Adam's first- and second-moment estimates live on the optimizer instance,
    so a fresh one per call zeroed them every single step -- online training
    silently degraded to SGD with an odd warmup, with nothing raised.

    This asserts on the STATE, not on object identity. `id(opt1) == id(opt2)`
    would pass even if the state were being wiped; `step == 2` after two calls
    cannot. The EMA buffer check is the second half: a `step` counter could in
    principle be incremented by something that still discards the moments.
    """
    model, x, y = _refit_ready(cls, tmp_path)

    model.update(x, y)
    model.update(x, y)

    optim = model._get_refit_optim()
    states = [s for s in optim.state.values() if "step" in s]
    assert states, "the refit optimizer carries no per-parameter state at all"
    assert all(int(s["step"]) == 2 for s in states), (
        "expected step == 2 for every parameter after two update() calls; "
        f"got {sorted({int(s['step']) for s in states})} -- the moment "
        "estimates are being reset between calls"
    )
    assert any(
        float(s["exp_avg"].abs().sum()) > 0.0 for s in states
    ), "every exp_avg buffer is zero, so no momentum was accumulated"


def test_refit_optimizer_is_rebuilt_when_the_model_is_replaced(tmp_path):
    """The invalidation half.

    The optimizer holds references to the parameter TENSORS. After `load()` or
    another `_init_model()`, `self.model` is a different `nn.Module` and those
    tensors are no longer the model's -- a cached optimizer would go on
    stepping detached tensors, which is worse than rebuilding every call.

    Keyed on `self.model` object identity, so no call site has to remember to
    invalidate anything.
    """
    model, x, y = _refit_ready(RNNRegressor, tmp_path)
    model.update(x, y)
    first = model._get_refit_optim()

    model._init_model_and_optim()  # rebinds self.model
    second = model._get_refit_optim()

    assert second is not first, "stale optimizer survived a model swap"
    assert not second.state, "a rebuilt optimizer must start with no state"
    first_params = {id(p) for g in first.param_groups for p in g["params"]}
    second_params = {id(p) for g in second.param_groups for p in g["params"]}
    assert first_params.isdisjoint(second_params), (
        "the rebuilt optimizer still points at the old model's tensors"
    )


def test_refit_optimizer_is_rebuilt_when_lr_refit_changes(tmp_path):
    """`lr_refit` is a config field a caller can change between calls; a
    cached optimizer must not silently keep the old learning rate."""
    model, x, y = _refit_ready(RNNRegressor, tmp_path, lr_refit=1e-2)
    first = model._get_refit_optim()
    assert first.param_groups[0]["lr"] == pytest.approx(1e-2)

    model.config.lr_refit = 5e-3
    second = model._get_refit_optim()
    assert second is not first
    assert second.param_groups[0]["lr"] == pytest.approx(5e-3)


def test_train_dl_still_drops_the_training_optimizer(tmp_path):
    """Batch 1 set `self.optim = None` after training on purpose (Adam's
    moment buffers are ~2x the parameter count). The refit optimizer is a
    separate lifecycle and must not resurrect it."""
    model = MLPRegressor(_make_config(tmp_path, epochs=1))
    model.collect()
    model.train()
    assert model.optim is None
    assert model.model is not None


# --------------------------------------------------------------------------
# The vecbt skeleton must be honest
# --------------------------------------------------------------------------


def test_rnn_classifier_vecbt_raises_instead_of_returning_none(tmp_path):
    """`RNNClassifier._vecbt` computed four pandas Series and then the file
    ended -- no return, no vectorbt call, no error. Four lines that look like
    work and behave like `pass`.

    The method is KEPT (Phase 6 owns end-to-end backtesting, and the entry/exit
    convention encoded in those four lines is the starting point), but a caller
    now hears that it is unbuilt instead of receiving `None`.
    """
    cfg = _make_config(tmp_path, **_hp_for(RNNClassifier))
    model = RNNClassifier(cfg)

    prices = pd.Series([1.0, 2.0, 3.0])
    signals = pd.Series([1, 0, 1])

    with pytest.raises(NotImplementedError, match="Phase 6"):
        model._vecbt(prices=prices, signals=signals)


# --------------------------------------------------------------------------
# BL-02 (reviewed 2026-09-07): to_tensor must normalize the panel's dtype
# --------------------------------------------------------------------------


def test_the_real_pipeline_panel_is_float64_not_float32():
    """The premise, asserted rather than assumed.

    Every other test in this file builds its panel with `.astype("float32")`,
    and `tests/test_model_layer.py`'s `RecordingRegressor._preprocess` ends in
    `.float()` -- so the suite only ever fed float32 into a stub that would
    have coped anyway. That structural blindness is why BL-02 escaped. This
    pins that `via_pandas=True` really does reproduce the float64 the real
    ingest produces; without it the two training tests below would be green
    for the wrong reason.
    """
    panel = FakePanel(["f_a"], seed=1, via_pandas=True).get_features()
    assert panel["f_a"].dtype == np.float64


@pytest.mark.parametrize("cls", [RNNRegressor, RNNClassifier])
def test_a_shipped_head_trains_on_a_float64_panel(tmp_path, cls):
    """BL-02: a Polars-backed factor could not train at all.

    `to_tensor` ends in `torch.from_numpy(...values)`, which preserves the
    panel's numpy dtype, and NONE of the three shipped `_preprocess`
    implementations casts -- they all do `torch.nan_to_num` only. Every torch
    module in `dl_model/` is built with default float32 parameters, so the
    float64 panel that both non-KunQuant data paths produce died inside the
    first forward pass:

        ValueError: RNN input dtype (torch.float64) does not match weight
        dtype (torch.float32).

    CLAUDE.md names Polars as a first-class second factor backend, so
    `DLConfig(factors=[kunquant_factor, polars_factor])` -- the exact
    interchangeability contract Phase 03 D-03 exists to guarantee -- could not
    be trained.

    A SHIPPED head is instantiated on purpose: a stand-in with a `.float()` in
    its `_preprocess` proves nothing about the classes that ship.
    """
    model = cls(_make_config(tmp_path, epochs=1, via_pandas=True, **_hp_for(cls)))
    model.collect()
    model._init_model_and_optim()
    before = [p.detach().clone() for p in model.model.parameters()]  # type: ignore

    model.train()

    after = list(model.model.parameters())  # type: ignore
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "the head ran an epoch on a float64 panel without stepping the "
        "optimizer -- this is not evidence it trained"
    )


def test_to_tensor_downcasts_a_float64_panel(tmp_path):
    """The pin for the fix itself, at the seam that owns the conversion.

    Reddening mutation: delete the cast in `BaseModel.to_tensor` -- this
    returns `torch.float64`. Putting the cast in the three `_preprocess`
    implementations instead would leave this red, which is the point: the
    fourth head would forget.
    """
    model = RNNRegressor(
        _make_config(tmp_path, via_pandas=True, **_hp_for(RNNRegressor))
    )
    panel = FakePanel(["f_a", "f_b"], seed=1, via_pandas=True).get_features()
    assert panel["f_a"].dtype == np.float64

    tensor = model.to_tensor(panel, ["f_a", "f_b"])

    assert tensor.dtype == torch.get_default_dtype() == torch.float32


def test_to_tensor_follows_torchs_default_dtype_not_a_hardcoded_float32(
    tmp_path,
):
    """`torch.get_default_dtype()`, not `np.float32`.

    A user who calls `torch.set_default_dtype(torch.float64)` gets float64
    modules, and a hardcoded downcast to float32 would then break training in
    the mirror image of BL-02. This is what makes the choice of
    `get_default_dtype()` load-bearing rather than cosmetic.

    Reddening mutation: hardcode `values.astype(np.float32)` -- the float32
    panel below stays float32 and this goes red.
    """
    model = RNNRegressor(_make_config(tmp_path, **_hp_for(RNNRegressor)))
    panel = FakePanel(["f_a"], seed=1).get_features()  # float32
    assert panel["f_a"].dtype == np.float32

    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        tensor = model.to_tensor(panel, ["f_a"])
    finally:
        torch.set_default_dtype(original)

    assert tensor.dtype == torch.float64


def test_to_tensor_leaves_non_floating_panels_alone(tmp_path):
    """The cast is scoped to FLOATING dtypes on purpose.

    An integer or boolean panel (a constituent-membership mask, a categorical
    code) carries meaning that a silent float conversion would blur, and
    nothing in `dl_model/` feeds one to a module today. Widening the cast to
    "everything numeric" is a separate decision; this test is what makes it a
    decision rather than a drift.
    """
    model = RNNRegressor(_make_config(tmp_path, **_hp_for(RNNRegressor)))
    panel = xr.Dataset(
        {"flag": (("timestamp", "symbol"), np.ones((4, N_SYMBOLS), dtype="int64"))},
        coords={"timestamp": TIMES[:4], "symbol": SYMBOLS},
    )

    tensor = model.to_tensor(panel, ["flag"])

    assert tensor.dtype == torch.int64


# --------------------------------------------------------------------------
# WR-01 (reviewed 2026-09-07): RNNRegressor._init_model ignored its argument
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [RNNRegressor, RNNClassifier])
def test_rnn_head_hyperparameters_reach_the_built_module(tmp_path, cls):
    """WR-01: `RNNRegressor._init_model` took `hyperparameters: dict` and then
    hardcoded every value -- `hidden_sizes=[256, 128, 64]`,
    `dropout_rates=[0.1, 0.1, 0.1]`, `hidden_sizes_linear=[32]`,
    `model_type="gru"`. `config.hyperparameters` was silently discarded.

    This is the same "declared, accepted, never referenced" lie this session
    deleted three times over (`_train_dl(backtest=...)`,
    `get_crypot_currency(name=...)`, `XrBackend.get_xarray_dataset(indexes)`),
    and `MLPRegressor._init_model` was changed in the same batch to honour it.

    It was invisible because the tests fed `RNNRegressor` the MLP-shaped
    `{"hidden_size1": 16, "hidden_size2": 8}` and passed BECAUSE the argument
    was ignored -- had it been honoured they would have raised. Green for the
    wrong reason.

    Asserting on the BUILT MODULE rather than on the config is the point: the
    only thing that distinguishes "honoured" from "accepted and discarded" is
    whether the number shows up in a layer.

    `RNNClassifier` is parametrized in alongside it to pin that the two
    siblings now agree.
    """
    hp = {**_RNN_HP, "hidden_sizes": [6, 5], "model_type": "lstm"}
    model = cls(_make_config(tmp_path, hyperparameters=hp))
    model.collect()
    model._init_model_and_optim()

    layers = model.model.base_models[0].recurrent_layers  # type: ignore
    assert [layer.hidden_size for layer in layers] == [6, 5], (
        "hidden_sizes never reached the module -- _init_model discarded its "
        "hyperparameters argument"
    )
    assert all(isinstance(layer, torch.nn.LSTM) for layer in layers), (
        "model_type never reached the module"
    )


def test_rnn_regressor_defaults_preserve_the_previously_hardcoded_shape(
    tmp_path,
):
    """WR-01's compatibility half: honouring the dict must not change the
    model any existing config builds.

    Every value is read with `.get(..., <the old hardcoded literal>)`, so a
    config that passes no RNN keys -- which is every config that existed
    before this change, `train_model.py` included -- still gets the exact
    256/128/64 GRU stack it used to get.

    `RNNRegressor` is asserted rather than `RNNClassifier` on purpose: the
    classifier reads its dict with `[...]` and raises `KeyError` on a missing
    key, so it has no defaults to preserve. Aligning those two idioms is a
    separate decision and is NOT made here.
    """
    model = RNNRegressor(_make_config(tmp_path, hyperparameters={}))
    model.collect()
    model._init_model_and_optim()

    layers = model.model.base_models[0].recurrent_layers  # type: ignore
    assert [layer.hidden_size for layer in layers] == [256, 128, 64]
    assert all(isinstance(layer, torch.nn.GRU) for layer in layers)
