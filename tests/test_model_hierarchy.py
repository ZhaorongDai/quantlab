"""Structural contract of the three-layer model hierarchy (quick task 260914-lno).

`quantlab/base/model.py` is split into:

- `BaseModel` -- framework-agnostic lifecycle; the ONLY home of the public
  `train` / `train_cv` / `load` / `predict`;
- `DLModel` -- the torch variant, five tensor hooks;
- `MLModel` -- the numpy variant, four hooks, native early stopping.

What is locked here, and what turns it red:

- each layer's `__abstractmethods__` is an exact set, so a hook silently
  gaining a default (or a new abstract hook appearing) is caught;
- no class between a shipped head and `BaseModel` redefines a public method --
  one implementation of the public interface, not one per variant;
- each variant declares its config class and checkpoint suffix, and a
  mismatched config raises `TypeError` before any factor is touched;
- the retired names stay retired (`hasattr`, not "no longer raises": a bypassed
  guard still answers `hasattr`), and `MLModel` references no `deepcopy`;
- `load_model_from_config` builds the config class the model class declares,
  so an `MLConfig` dict can no longer become a `DLConfig`;
- `DLModel.predict` accepts a float64 ndarray.

Everything is synthetic, CPU-only and offline.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

import quantlab.utils.module as module_utils
from quantlab.base.config import DLConfig, MLConfig
from quantlab.base.model import BaseModel, DLModel, MLModel
from quantlab.dl_model.mlp import MLPRegressor
from quantlab.dl_model.rnn import RNNRegressor
from quantlab.dl_model.rnn_classification import RNNClassifier

N_TIMES = 40
N_SYMBOLS = 3
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)
START = np.datetime_as_string(TIMES[0], unit="D")
END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")

PUBLIC_METHODS = frozenset({"train", "train_cv", "load", "predict"})


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


class FakePanel:
    """A stand-in for a factor/label object: only what `collect()` calls."""

    def __init__(self, names, seed=0):
        rng = np.random.default_rng(seed)
        self.names = list(names)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    rng.standard_normal((N_TIMES, N_SYMBOLS)).astype("float32"),
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


class StubMLHead(MLModel):
    """The smallest concrete `MLModel`."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


class LinearDLHead(DLModel):
    """The smallest concrete `DLModel`: one `nn.Linear` on the last axis."""

    def _init_model(self, num_symbols, num_features, num_labels, hyperparameters):
        return nn.Linear(num_features, num_labels)

    def _init_optim(self, model):
        return torch.optim.SGD(model.parameters(), lr=1e-3)

    def _preprocess(self, data):
        return torch.nan_to_num(data, nan=0.0)

    def _train_one_batch(self, epoch, x, y):
        return torch.tensor(0.0)

    def _val_one_batch(self, epoch, x, y):
        return torch.tensor(0.0)

    def _test_one_batch(self, epoch, x, y):
        return torch.tensor(0.0)


def _kwargs(tmp_path, factors=None, labels=None):
    return dict(
        factors=factors if factors is not None else [FakePanel(["f_a", "f_b"], seed=1)],
        labels=labels if labels is not None else [FakePanel(["ret"], seed=2)],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
    )


# --------------------------------------------------------------------------
# Abstract-method contracts
# --------------------------------------------------------------------------


def test_base_model_abstract_methods_are_exactly_the_variant_seams():
    assert BaseModel.__abstractmethods__ == frozenset(
        {"config_cls", "checkpoint_suffix", "_fit", "_predict", "_write_checkpoint", "_read_checkpoint"}
    )


def test_dl_model_abstract_methods_are_the_five_tensor_hooks():
    assert DLModel.__abstractmethods__ == frozenset(
        {"_init_model", "_train_one_batch", "_val_one_batch", "_test_one_batch", "_preprocess"}
    )


def test_ml_model_abstract_methods_are_the_four_numpy_hooks():
    assert MLModel.__abstractmethods__ == frozenset(
        {"_init_model", "_preprocess", "_fit_model", "_forward"}
    )


def test_public_methods_live_on_base_model():
    assert PUBLIC_METHODS <= set(BaseModel.__dict__)


@pytest.mark.parametrize(
    "cls",
    [DLModel, MLModel, MLPRegressor, RNNRegressor, RNNClassifier, StubMLHead],
    ids=lambda c: c.__name__,
)
def test_no_class_above_base_model_redefines_a_public_method(cls):
    """One implementation of `train` / `train_cv` / `load` / `predict`. A
    variant or head that overrides one of them forks the public contract --
    exactly what the split exists to prevent."""
    mro = cls.__mro__
    above = mro[: mro.index(BaseModel)]
    offenders = {
        c.__name__: sorted(PUBLIC_METHODS & set(c.__dict__)) for c in above
    }
    assert all(not names for names in offenders.values()), offenders


def test_variants_declare_config_class_and_checkpoint_suffix():
    assert DLModel.config_cls is DLConfig
    assert DLModel.checkpoint_suffix == ".pth"
    assert MLModel.config_cls is MLConfig
    assert MLModel.checkpoint_suffix == ".joblib"


# --------------------------------------------------------------------------
# Config type guard
# --------------------------------------------------------------------------


def test_dl_head_rejects_an_ml_config_before_touching_factors(tmp_path):
    """The type check is the setter's FIRST statement: the rejected panel's
    dates must still be unset. Turns red if the check moves below the date
    injection or disappears."""
    panel = FakePanel(["f_a"])
    with pytest.raises(TypeError, match="MLPRegressor requires a DLConfig, got MLConfig"):
        MLPRegressor(MLConfig(**_kwargs(tmp_path, factors=[panel])))
    assert panel.config.start_date is None


def test_ml_head_rejects_a_dl_config_before_touching_factors(tmp_path):
    panel = FakePanel(["f_a"])
    with pytest.raises(TypeError, match="StubMLHead requires a MLConfig, got DLConfig"):
        StubMLHead(DLConfig(**_kwargs(tmp_path, factors=[panel])))
    assert panel.config.start_date is None


# --------------------------------------------------------------------------
# Deletion locks
# --------------------------------------------------------------------------


def test_retired_names_stay_retired():
    """Renamed without aliases: two live names for one method is the
    ambiguity a later reader resolves wrongly."""
    assert not hasattr(BaseModel, "_auto_train")
    assert not hasattr(DLModel, "_train_dl")
    assert not hasattr(DLModel, "_predict_nn")
    assert not hasattr(MLModel, "_snapshot_model")
    assert not hasattr(MLModel, "_train_one_epoch")


def test_ml_config_has_no_epochs_and_tree_friendly_early_stopping_defaults():
    """ML heads count patience in boosting rounds; an `epochs` field would
    invite an outer loop around the library's own training."""
    assert "epochs" not in MLConfig.__dataclass_fields__
    cfg = MLConfig(
        factors=[], labels=[], model_save_dir="x", factor_data_strategy="cal", label_data_strategy="cal"
    )
    assert cfg.early_stopping is False
    assert cfg.early_stopping_patience == 5


def test_ml_model_never_references_deepcopy():
    """Rollback to the best round is the library's job (slicing trees), never
    a deep copy of the model. Turns red on any `deepcopy` name or attribute in
    the class body; docstrings may still explain why."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(MLModel)))
    hits = [
        node
        for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id == "deepcopy")
        or (isinstance(node, ast.Attribute) and node.attr == "deepcopy")
    ]
    assert hits == []


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def _patch_factor_loader(monkeypatch):
    monkeypatch.setattr(
        module_utils,
        "load_factor_from_config",
        lambda cfg: FakePanel(cfg["factor_names"]),
    )


def test_loader_builds_a_dl_config_for_a_dl_head(tmp_path, monkeypatch):
    """Real dotted path, real class lookup; only factor reconstruction is
    faked."""
    _patch_factor_loader(monkeypatch)
    saved = MLPRegressor(DLConfig(**_kwargs(tmp_path))).get_config()
    assert saved["name"] == "quantlab.dl_model.mlp.MLPRegressor"

    model = module_utils.load_model_from_config(saved)

    assert isinstance(model, MLPRegressor)
    assert type(model.config) is DLConfig


def test_loader_builds_an_ml_config_for_an_ml_head(tmp_path, monkeypatch):
    """Before 260914-lno the loader hardcoded `DLConfig(**config)`, so an ML
    checkpoint's config silently came back as a DLConfig."""
    _patch_factor_loader(monkeypatch)
    saved = StubMLHead(MLConfig(**_kwargs(tmp_path))).get_config()
    monkeypatch.setattr(module_utils, "get_cls_from_path", lambda path: StubMLHead)

    model = module_utils.load_model_from_config(saved)

    assert isinstance(model, StubMLHead)
    assert type(model.config) is MLConfig


# --------------------------------------------------------------------------
# DLModel inference input types
# --------------------------------------------------------------------------


def test_dl_predict_accepts_a_float64_ndarray(tmp_path):
    """An ndarray goes through the same dtype normalisation as `to_tensor`, so
    it predicts exactly what the equal-valued float32 tensor predicts."""
    model = LinearDLHead(DLConfig(**_kwargs(tmp_path)))
    model.collect()
    model._init_model_and_optim()
    x64 = np.random.default_rng(0).standard_normal((5, N_SYMBOLS, 2))

    from_array = model.predict(x64)
    from_tensor = model.predict(torch.from_numpy(x64.astype(np.float32)))

    assert from_array.dtype == torch.get_default_dtype()
    assert torch.equal(from_array, from_tensor)


def test_dl_predict_rejects_other_types(tmp_path):
    model = LinearDLHead(DLConfig(**_kwargs(tmp_path)))
    model.collect()
    model._init_model_and_optim()
    with pytest.raises(TypeError):
        model.predict([[1.0]])
