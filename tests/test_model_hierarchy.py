"""Structural contract of the three-layer model hierarchy (quick task 260914-lno).

`quantlab/base/model.py` is split into:

- `BaseModel` -- framework-agnostic lifecycle; the ONLY home of the public
  `train` / `train_cv` / `load` / `predict`;
- `DLModel` -- the torch variant, step hooks plus two declarations;
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

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

import quantlab.utils.module as module_utils
from quantlab.base.config import DLConfig, MLConfig
from quantlab.base.model import BaseModel, DLModel, MLModel
from quantlab.ml_model.xgb import XGBoostRegressor
from tests.dl_heads import OneBarHead
from tests.label_stubs import StubLabel

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

    def _get_factor_names(self):
        return list(self.names)

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def read(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

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


def _kwargs(tmp_path, factors=None, labels=None):
    return dict(
        factors=factors if factors is not None else [FakePanel(["f_a", "f_b"], seed=1)],
        labels=[StubLabel(label) for label in labels]
        if labels is not None
        else [StubLabel(FakePanel(["ret"], seed=2))],
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


def test_dl_model_abstract_methods_are_the_step_hooks_and_two_declarations():
    assert DLModel.__abstractmethods__ == frozenset(
        {"_init_model", "_train_one_batch", "_val_one_batch", "_test_one_batch",
         "window_bars", "target_transform"}
    )


def test_ml_model_abstract_methods_are_the_four_numpy_hooks():
    assert MLModel.__abstractmethods__ == frozenset(
        {"_init_model", "_preprocess", "_fit_model", "_forward"}
    )


def test_public_methods_live_on_base_model():
    assert PUBLIC_METHODS <= set(BaseModel.__dict__)


@pytest.mark.parametrize(
    "cls",
    [DLModel, MLModel, OneBarHead, XGBoostRegressor, StubMLHead],
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


def test_dl_head_rejects_an_ml_config(tmp_path):
    with pytest.raises(TypeError, match="OneBarHead requires a DLConfig, got MLConfig"):
        OneBarHead(MLConfig(**_kwargs(tmp_path)))


def test_ml_head_rejects_a_dl_config(tmp_path):
    with pytest.raises(TypeError, match="StubMLHead requires a MLConfig, got DLConfig"):
        StubMLHead(DLConfig(**_kwargs(tmp_path)))


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


def test_the_fixed_symbol_dl_machinery_and_config_fields_are_deleted():
    """Issue #39: the step hooks stay, but each step is one bar's
    cross-section; the refit optimizer, the training-symbol alignment and
    the batch/early-stopping config fields went with the fixed-symbol heads."""
    for name in (
        "_preprocess_stream", "_get_refit_optim", "to_tensor",
        "_align_prediction_symbols", "stopping",
    ):
        assert not hasattr(DLModel, name), name
    for field in ("batch_size", "num_workers", "early_stopping",
                  "early_stopping_patience", "lr_refit"):
        assert field not in DLConfig.__dataclass_fields__, field


def test_stale_backtest_hooks_are_deleted():
    """D-37 (phase 03.7): the in-model backtest hooks are deleted, because
    backtesting now lives only in `quantlab/backtest/`.

    Four things were removed and must stay gone: the config setter's reset
    method for a backtest dataset, `DLModel._fit`'s `backtest` parameter with
    its NotImplementedError branch, and the backtest data slot on both config
    classes. Locked with `hasattr` / `inspect.signature`, not "no longer
    raises": a bypassed guard still answers `hasattr`. Turns red if any of
    them returns, or if `DLModel._fit` grows any parameter beyond the abstract
    `_fit(self, project_name, experiment_name, model_name)`.
    """
    assert not hasattr(BaseModel, "_reset_backtest_dataset_config")

    fit_params = list(inspect.signature(DLModel._fit).parameters)
    assert "backtest" not in fit_params
    assert fit_params == ["self", "project_name", "experiment_name", "model_name"]

    for config_cls in (DLConfig, MLConfig):
        assert not hasattr(config_cls, "backtest_data"), config_cls.__name__
        assert "backtest_data" not in config_cls.__dataclass_fields__, config_cls.__name__


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
    """Rebuild each saved panel as a `FakePanel`; the `_kwargs` label (the
    only panel named `ret`) comes back wrapped as a label, since a model
    rejects a label without `lookahead_bars()`."""

    def load(cfg):
        panel = FakePanel(cfg["factor_names"])
        return StubLabel(panel) if cfg["factor_names"] == ["ret"] else panel

    monkeypatch.setattr(module_utils, "load_factor_from_config", load)


def test_loader_builds_a_dl_config_for_a_dl_head(tmp_path, monkeypatch):
    """Real dotted path, real class lookup; only factor reconstruction is
    faked."""
    _patch_factor_loader(monkeypatch)
    saved = OneBarHead(DLConfig(**_kwargs(tmp_path))).get_config()
    assert saved["name"] == "tests.dl_heads.OneBarHead"

    model = module_utils.load_model_from_config(saved)

    assert isinstance(model, OneBarHead)
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


def test_loader_builds_an_ml_config_for_the_shipped_xgboost_head(tmp_path, monkeypatch):
    """Same as above through the REAL dotted path of the shipped ML head, so
    the class lookup itself is not faked."""
    _patch_factor_loader(monkeypatch)
    saved = XGBoostRegressor(MLConfig(**_kwargs(tmp_path))).get_config()
    assert saved["name"] == "quantlab.ml_model.xgb.XGBoostRegressor"

    model = module_utils.load_model_from_config(saved)

    assert isinstance(model, XGBoostRegressor)
    assert type(model.config) is MLConfig


def test_loader_drops_the_resolved_hyperparameters_record(tmp_path, monkeypatch):
    """`MLModel.get_config` may add a top-level `resolved_hyperparameters`
    record; it is not an `MLConfig` field, so the loader must drop it before
    `cls.config_cls(**config)` or every such checkpoint fails to reload."""
    _patch_factor_loader(monkeypatch)
    saved = XGBoostRegressor(MLConfig(**_kwargs(tmp_path))).get_config()
    saved["resolved_hyperparameters"] = {"eta": 0.3, "num_boost_round": 4}

    model = module_utils.load_model_from_config(saved)

    assert type(model.config) is MLConfig
    assert model.config.hyperparameters == {}


def test_loader_still_rejects_other_unknown_keys(tmp_path, monkeypatch):
    """Only that one record key is dropped; anything else unknown stays loud."""
    _patch_factor_loader(monkeypatch)
    saved = XGBoostRegressor(MLConfig(**_kwargs(tmp_path))).get_config()
    saved["not_a_config_field"] = 1

    with pytest.raises(TypeError, match="not_a_config_field"):
        module_utils.load_model_from_config(saved)


def test_dl_config_json_has_no_resolved_hyperparameters_key(tmp_path):
    """The record is an MLModel feature; DL checkpoints' config.json is
    unchanged."""
    cfg = DLConfig(
        **_kwargs(tmp_path),
        train_start=START,
        train_end=np.datetime_as_string(TIMES[29], unit="D"),
        test_start=np.datetime_as_string(TIMES[30], unit="D"),
        test_end=END,
        epochs=1,
    )
    model = OneBarHead(cfg)
    model.collect()
    model.train()

    written = sorted((tmp_path / "ckpt").rglob("config.json"))
    assert len(written) == 1
    import json

    assert "resolved_hyperparameters" not in json.loads(written[0].read_text())
    assert "resolved_hyperparameters" not in model.get_config()


# --------------------------------------------------------------------------
# DLModel inference input types
# --------------------------------------------------------------------------


def _untrained(tmp_path):
    model = OneBarHead(DLConfig(**_kwargs(tmp_path)))
    model.model = model._init_model(num_features=2, num_labels=1, hyperparameters={})
    return model


def test_dl_predict_accepts_an_ndarray_or_a_tensor(tmp_path):
    """Both go through the same windows, so equal values predict equally."""
    model = _untrained(tmp_path)
    x64 = np.random.default_rng(0).standard_normal((5, N_SYMBOLS, 2))

    from_array = model.predict(x64)
    from_tensor = model.predict(torch.from_numpy(x64.astype(np.float32)))

    assert tuple(from_array.shape) == (5, N_SYMBOLS, 1)
    assert torch.equal(from_array, from_tensor)


def test_dl_predict_rejects_other_types(tmp_path):
    model = _untrained(tmp_path)
    with pytest.raises(TypeError):
        model.predict([[1.0]])
