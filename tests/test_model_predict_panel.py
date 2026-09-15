"""Contract of `BaseModel.predict_panel` for every shipped head (phase 03.7, D-29 / D-33).

The backtester reads model output ONLY through `predict_panel`, so a head this
method cannot serve is a head that cannot be backtested. What is locked here,
and what turns it red:

- one output variable per label, in DECLARED label order (labels are declared
  reverse-alphabetically, so an alphabetical sort anywhere on the path would
  swap them);
- output coords are the sorted `(timestamp, symbol)` axes of the feature panel
  and every value sits at the coordinate whose features produced it (the input
  panel is given with both axes reversed);
- a row whose features are ALL NaN predicts NaN in every label; a partially-NaN
  row is NOT masked (03.7-RESEARCH.md Pitfall 7);
- a missing factor variable, an uninitialized model and an unadapted
  tuple-returning DL head each fail with an error that names the problem;
- the generic DL path, XGBoostRegressor and MLPRegressor return exactly what
  their own inference path returns.

MLP finding (recorded for the user): D-33 assumed MLP keeps the generic
`[T, S, L]` path. It cannot. `MLPRegressor._init_model` builds
`nn.Linear(num_symbols * num_features, ...)` and its training loop flattens
each bar to `[T, S*F]`, so its module consumes the flat matrix and the generic
path fails inside the first Linear layer. `MLPRegressor` therefore carries its
own `_predict_panel_array` adapter; its public `predict()` is unchanged.

Everything is synthetic, CPU-only and offline. Test-local stand-ins are copied
in the style of `tests/test_model_hierarchy.py` rather than imported from it.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

from quantlab.base.config import DLConfig, MLConfig
from quantlab.base.model import DLModel, MLModel
from quantlab.dl_model.mlp import MLPRegressor
from quantlab.dl_model.rnn import RNNRegressor
from quantlab.dl_model.rnn_classification import RNNClassifier
from quantlab.ml_model.xgb import XGBoostRegressor

N_TIMES = 30
SYMBOLS = ["S0", "S1", "S2"]
N_SYMBOLS = len(SYMBOLS)
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)
START = np.datetime_as_string(TIMES[0], unit="D")
END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")
TRAIN_END = np.datetime_as_string(TIMES[19], unit="D")
TEST_START = np.datetime_as_string(TIMES[20], unit="D")

FACTORS = ["f_a", "f_b"]
#: Reverse alphabetical on purpose: any alphabetical sort on the variable axis
#: would put `ret_30` first and swap the two outputs.
LABELS = ["ret_60", "ret_30"]


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


class ChannelMLHead(MLModel):
    """An `MLModel` whose label channels are distinguishable by construction.

    Channel i is `(i + 1) * f_a + 10 * i`, so channel 0 is exactly the first
    feature and channel 1 is `2 * f_a + 10`. `_preprocess` zero-fills NaN like
    the real heads do, which is what makes an all-NaN row predict a FINITE
    value unless `predict_panel` masks it.
    """

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _preprocess(self, data):
        return np.nan_to_num(np.array(data, dtype=np.float64, copy=True))

    def _fit_model(self, train_x, train_y, val_x, val_y):
        pass

    def _forward(self, x):
        return np.stack(
            [(i + 1) * x[..., 0] + 10.0 * i for i in range(self.model["num_labels"])],
            axis=-1,
        )


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


class _TupleLinear(nn.Module):
    def __init__(self, num_features, num_labels):
        super().__init__()
        self.fc = nn.Linear(num_features, num_labels)

    def forward(self, x):
        out = self.fc(x)
        return out, out


class TupleHeadWithoutAdapter(LinearDLHead):
    """A DL head whose module returns a tuple and that has no adapter."""

    def _init_model(self, num_symbols, num_features, num_labels, hyperparameters):
        return _TupleLinear(num_features, num_labels)


def _config_kwargs(tmp_path, *, labels=LABELS, seed=1):
    return dict(
        factors=[FakePanel(FACTORS, seed=seed)],
        labels=[FakePanel(labels, seed=seed + 1)],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
    )


def _ml_stub(tmp_path) -> ChannelMLHead:
    model = ChannelMLHead(MLConfig(**_config_kwargs(tmp_path)))
    model.model = model._init_model(
        num_features=len(FACTORS), num_labels=len(LABELS), hyperparameters={}
    )
    return model


def _features(model) -> xr.Dataset:
    return model.config.factors[0].get_features()


def _stack(features: xr.Dataset, names) -> np.ndarray:
    """`[T, S, F]` built independently of `BaseModel.to_array`."""
    ordered = features.sortby(["timestamp", "symbol"])
    return np.stack(
        [ordered[name].transpose("timestamp", "symbol").values for name in names],
        axis=-1,
    ).astype(np.float64)


# --------------------------------------------------------------------------
# Generic contract (ML stub)
# --------------------------------------------------------------------------


def test_ml_head_returns_one_variable_per_label_in_declared_order(tmp_path):
    """Locks declared label order (T-03.7-15).

    Labels are declared `["ret_60", "ret_30"]`. Goes red if the variable axis
    is sorted by name anywhere between `to_array` and the output Dataset:
    `ret_60` would then carry channel 1 (`2 * f_a + 10`) instead of channel 0.
    """
    model = _ml_stub(tmp_path)
    features = _features(model)

    pred = model.predict_panel(features)

    assert list(pred.data_vars) == LABELS
    f_a = _stack(features, ["f_a"])[..., 0]
    np.testing.assert_allclose(pred["ret_60"].values, f_a, atol=1e-12)
    np.testing.assert_allclose(pred["ret_30"].values, 2.0 * f_a + 10.0, atol=1e-6)
    assert pred["ret_60"].dims == ("timestamp", "symbol")


def test_output_coords_follow_the_sorted_feature_panel(tmp_path):
    """Locks coord/value alignment (T-03.7-15).

    The feature panel is handed over with BOTH axes reversed. The output must
    carry the sorted axes, and each value must sit at the symbol and timestamp
    whose features produced it. Goes red if coords are taken from the unsorted
    input while values come from the sorted array (values would land on the
    mirrored symbol).
    """
    model = _ml_stub(tmp_path)
    reversed_features = _features(model).isel(
        timestamp=slice(None, None, -1), symbol=slice(None, None, -1)
    )
    assert list(reversed_features.symbol.values) == SYMBOLS[::-1]

    pred = model.predict_panel(reversed_features)

    assert list(pred.symbol.values) == SYMBOLS
    assert (np.diff(pred.timestamp.values) > np.timedelta64(0)).all()
    for symbol in SYMBOLS:
        for t in (TIMES[0], TIMES[7], TIMES[-1]):
            expected = float(reversed_features["f_a"].sel(timestamp=t, symbol=symbol))
            got = float(pred["ret_60"].sel(timestamp=t, symbol=symbol))
            assert got == pytest.approx(expected, abs=1e-6), (symbol, t)


def test_all_nan_feature_rows_predict_nan_and_partial_rows_do_not(tmp_path):
    """Locks the Pitfall 7 mask and its scope (T-03.7-16).

    The stub zero-fills NaN inputs, exactly like the real heads, so without the
    mask an unlisted symbol (every feature NaN) gets a finite score and the
    backtester can select it. Goes red if the mask is removed (the all-NaN row
    becomes finite) or widened to "any feature NaN" (the partial row becomes
    NaN, which would empty the universe for wide factor sets).
    """
    model = _ml_stub(tmp_path)
    features = _features(model).copy(deep=True)
    features["f_a"].loc[dict(timestamp=TIMES[3], symbol="S1")] = np.nan
    features["f_b"].loc[dict(timestamp=TIMES[3], symbol="S1")] = np.nan
    features["f_a"].loc[dict(timestamp=TIMES[5], symbol="S2")] = np.nan

    pred = model.predict_panel(features)

    for label in LABELS:
        assert np.isnan(float(pred[label].sel(timestamp=TIMES[3], symbol="S1"))), label
        assert np.isfinite(float(pred[label].sel(timestamp=TIMES[5], symbol="S2"))), label
    stacked = pred.to_dataarray().values
    assert int(np.isnan(stacked).sum()) == len(LABELS), (
        "only the one all-NaN (t, s) row may be NaN"
    )


def test_missing_factor_variable_raises_naming_it(tmp_path):
    """Locks the missing-factor guard.

    Goes red if a dropped factor surfaces as a bare `KeyError` from xarray, or
    as a silently shorter feature axis, instead of a `ValueError` naming it.
    """
    model = _ml_stub(tmp_path)
    features = _features(model).drop_vars("f_b")

    with pytest.raises(ValueError, match="f_b"):
        model.predict_panel(features)


def test_predict_panel_before_train_or_load_raises(tmp_path):
    """Locks that `predict_panel` goes through the public `predict` guard.

    Goes red if `predict_panel` calls `_predict` or `_forward` directly and
    reaches a `None` model (an `AttributeError`/`TypeError` from inside the
    head instead of the documented message).
    """
    model = ChannelMLHead(MLConfig(**_config_kwargs(tmp_path)))
    assert model.model is None

    with pytest.raises(ValueError, match="Model not initialized"):
        model.predict_panel(_features(model))


# --------------------------------------------------------------------------
# DL heads
# --------------------------------------------------------------------------


def test_generic_dl_head_tensor_output(tmp_path):
    """Locks the DLModel default: a `[T, S, L]` tensor becomes float64 values.

    The expected values are the module's own output on the zero-filled input,
    computed here without `to_array`. Goes red if the tensor is not moved to
    numpy, if the dtype is not float64, or if the channel layout is changed.
    """
    model = LinearDLHead(DLConfig(**_config_kwargs(tmp_path)))
    model.collect()
    model._init_model_and_optim()
    features = _features(model)

    pred = model.predict_panel(features)

    x = torch.from_numpy(np.nan_to_num(_stack(features, FACTORS))).float()
    with torch.no_grad():
        expected = model.model(x).numpy()  # type: ignore[misc]
    got = pred.to_dataarray().transpose("timestamp", "symbol", "variable")
    assert got.dtype == np.float64
    np.testing.assert_allclose(got.values, expected, atol=1e-6)


def test_mlp_regressor_predict_panel_reshapes_the_flat_contract(tmp_path):
    """Locks the MLPRegressor adapter (corrects D-33's premise for MLP).

    The MLP module's inference input is the flat `[T, S*F]` matrix
    (`tests/test_dl_models.py::test_mlp_regressor_trains_two_epochs_and_predicts`),
    and it returns `[T, S*L]`. `predict_panel` must equal
    `predict(flat x).reshape(T, S, L)`. Goes red without the adapter: the
    generic DL path hands `[T, S, F]` to `nn.Linear(S*F, ...)`, which raises a
    shape error. Also red if the adapter flattens in a different order than
    `_train_one_batch` does.
    """
    model = MLPRegressor(
        DLConfig(
            **_config_kwargs(tmp_path),
            hyperparameters={"hidden_size1": 16, "hidden_size2": 8},
        )
    )
    model.collect()
    model._init_model_and_optim()
    features = _features(model)

    pred = model.predict_panel(features)

    x = _stack(features, FACTORS)
    flat = torch.from_numpy(x.reshape(N_TIMES, N_SYMBOLS * len(FACTORS))).float()
    expected = (
        model.predict(flat).detach().cpu().numpy().reshape(N_TIMES, N_SYMBOLS, len(LABELS))
    )
    got = pred.to_dataarray().transpose("timestamp", "symbol", "variable").values
    assert got.shape == (N_TIMES, N_SYMBOLS, len(LABELS))
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_tuple_returning_head_without_adapter_raises_naming_it(tmp_path):
    """Locks D-33's rejection arm (T-03.7-17).

    A head whose module returns a tuple and that does not override
    `_predict_panel_array` must fail with a `TypeError` naming the head. Goes
    red if the generic path silently picks one tuple element, or fails with an
    anonymous error that does not say which head needs an adapter.
    """
    model = TupleHeadWithoutAdapter(DLConfig(**_config_kwargs(tmp_path)))
    model.collect()
    model._init_model_and_optim()

    with pytest.raises(TypeError, match="TupleHeadWithoutAdapter"):
        model.predict_panel(_features(model))


# --------------------------------------------------------------------------
# XGBoostRegressor
# --------------------------------------------------------------------------


def test_xgboost_regressor_predict_panel_matches_predict(tmp_path):
    """Locks that the shipped ML head is served unchanged by the ML default.

    A few-round booster is really trained, then `predict_panel` must equal
    `predict(to_array(...))` to 1e-12. Goes red if the ML default reorders,
    rescales or re-preprocesses the booster's output.
    """
    config = MLConfig(
        **_config_kwargs(tmp_path, labels=["ret_a"]),
        train_start=START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=END,
        early_stopping=False,
        hyperparameters={"num_boost_round": 5, "nthread": 1},
    )
    model = XGBoostRegressor(config)
    model.collect()
    model.train()
    features = _features(model)

    pred = model.predict_panel(features)

    expected = np.asarray(
        model.predict(model.to_array(features, FACTORS)), dtype=np.float64
    )
    got = pred.to_dataarray().transpose("timestamp", "symbol", "variable").values
    assert got.shape == (N_TIMES, N_SYMBOLS, 1)
    np.testing.assert_allclose(got, expected, atol=1e-12)


# --------------------------------------------------------------------------
# RNN heads: forward returns (primary_pred_final, all_direct_preds) -- D-33
# --------------------------------------------------------------------------

#: Tiny `ModelRCrypto` shape; dropout 0 so eval/train mode cannot move values.
_RNN_HP = {
    "hidden_sizes": [8, 8],
    "dropout_rates": [0.0, 0.0],
    "hidden_sizes_linear": [8],
    "dropout_rates_linear": [0.0],
    "model_type": "gru",
}


def _module_outputs(model, features):
    """The head's raw module outputs, computed here, not through `predict`."""
    x = model._preprocess(
        model.to_tensor(features.sortby(["timestamp", "symbol"]), FACTORS)
    )
    model.model.eval()  # type: ignore[union-attr]
    with torch.no_grad():
        primary, direct = model.model(x)  # type: ignore[misc]
    return primary.numpy(), direct.numpy()


def test_rnn_regressor_labels_are_the_direct_prediction_channels(tmp_path):
    """Locks D-33 for RNNRegressor: label i is channel i of `all_direct_preds`.

    Label 0 is `base_models[0]`'s direct prediction, NOT the aux-combined
    `primary_pred_final`; labels 1.. are the auxiliary direct predictions. The
    last assertion proves the two candidates differ on this seed, so the test
    can tell them apart. Goes red if the adapter returns `primary_pred_final`
    for label 0, if the channels are reordered, or if the head has no adapter
    at all (the generic path raises `TypeError` on the tuple).
    """
    labels = ["ret_60", "ret_30", "ret_10"]
    model = RNNRegressor(
        DLConfig(
            **_config_kwargs(tmp_path, labels=labels),
            hyperparameters=_RNN_HP,
            random_seed=7,
        )
    )
    model.collect()
    model._init_model_and_optim()
    features = _features(model)

    pred = model.predict_panel(features)

    primary, direct = _module_outputs(model, features)
    got = pred.to_dataarray().transpose("timestamp", "symbol", "variable").values
    assert list(pred.data_vars) == labels
    assert got.shape == (N_TIMES, N_SYMBOLS, len(labels))
    np.testing.assert_allclose(got, direct, atol=1e-6)
    assert not np.allclose(got[..., 0], primary[..., 0], atol=1e-4), (
        "label 0 matches primary_pred_final everywhere, so this seed cannot "
        "distinguish the direct prediction from the aux-combined one"
    )


def test_rnn_classifier_labels_are_per_label_up_probabilities(tmp_path):
    """Locks D-33 for RNNClassifier: label i is P(up) from direct channels [2i, 2i+1].

    Each label variable holds a class-1 softmax probability in [0, 1], not a
    return. The channel pairing mirrors `_train_one_batch`'s
    `i * 2 : (i + 1) * 2` slice. Goes red if the adapter takes class 0, pairs
    channels differently, reads `primary_pred_final` for label 0, returns raw
    logits, or is missing (the generic path raises `TypeError` on the tuple).
    """
    model = RNNClassifier(
        DLConfig(
            **_config_kwargs(tmp_path),
            hyperparameters=_RNN_HP,
            random_seed=7,
        )
    )
    model.collect()
    model._init_model_and_optim()
    features = _features(model)

    pred = model.predict_panel(features)

    primary, direct = _module_outputs(model, features)
    expected = np.stack(
        [
            torch.softmax(torch.from_numpy(direct[..., 2 * i : 2 * i + 2]), dim=-1)[..., 1].numpy()
            for i in range(len(LABELS))
        ],
        axis=-1,
    )
    primary_up = torch.softmax(torch.from_numpy(primary), dim=-1)[..., 1].numpy()
    got = pred.to_dataarray().transpose("timestamp", "symbol", "variable").values
    assert list(pred.data_vars) == LABELS
    assert got.shape == (N_TIMES, N_SYMBOLS, len(LABELS))
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert ((got >= 0.0) & (got <= 1.0)).all()
    assert not np.allclose(got[..., 0], primary_up, atol=1e-4), (
        "label 0 equals P(up) of primary_pred_final everywhere, so this seed "
        "cannot distinguish the direct prediction from the aux-combined one"
    )
