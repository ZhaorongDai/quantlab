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

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr
from loguru import logger

import quantlab.utils.module as module_utils
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

#: The int64 PERMNO arm (03.11-04). `7000` is FOUR digits on purpose: numeric
#: and lexicographic order coincide on the whole five-digit historical PERMNO
#: universe and fork only here (`"10107" < "7000"`), so a panel without a
#: four-digit member cannot tell a numeric sort from a lexicographic one.
#: Deliberately given unsorted, so the sort is doing work.
PERMNOS = [10107, 7000, 14593]
#: What `sort_symbol_axis` must produce -- NOT `sorted(map(str, PERMNOS))`,
#: which is `['10107', '14593', '7000']`.
SORTED_PERMNOS = [7000, 10107, 14593]


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


class FakePanel:
    """A stand-in for a factor/label object: only what `collect()` calls.

    `symbols` is a parameter rather than the module constant so the PERMNO
    arm (03.11-04) can build an int64 symbol axis through exactly the same
    path the ticker arm uses. Everything else about the panel is identical,
    which is what makes the two arms comparable.
    """

    def __init__(self, names, seed=0, symbols=SYMBOLS):
        rng = np.random.default_rng(seed)
        self.names = list(names)
        self.symbols = list(symbols)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    rng.standard_normal(
                        (N_TIMES, len(self.symbols))
                    ).astype("float32"),
                )
                for name in self.names
            },
            coords={"timestamp": TIMES, "symbol": self.symbols},
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


def _config_kwargs(tmp_path, *, labels=LABELS, seed=1, symbols=SYMBOLS):
    return dict(
        factors=[FakePanel(FACTORS, seed=seed, symbols=symbols)],
        labels=[FakePanel(labels, seed=seed + 1, symbols=symbols)],
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


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    yield messages
    logger.remove(handler_id)


def _dl_train_kwargs(tmp_path, *, symbols=SYMBOLS) -> dict:
    return dict(
        **_config_kwargs(tmp_path, symbols=symbols),
        train_start=START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=END,
        epochs=1,
        batch_size=16,
        num_workers=0,
    )


def _trained_dl_checkpoint(tmp_path, *, symbols=SYMBOLS) -> tuple[LinearDLHead, Path]:
    trained = LinearDLHead(
        DLConfig(**_dl_train_kwargs(tmp_path, symbols=symbols))
    )
    trained.collect()
    trained.train()
    checkpoints = sorted((tmp_path / "ckpt").rglob("*.pth"))
    assert len(checkpoints) == 1, checkpoints
    return trained, checkpoints[0]


def test_dl_checkpoint_records_its_symbols_and_predict_panel_aligns_onto_them(
    tmp_path, warning_messages
):
    """Code review WR-02: a DL head predicts only on the symbols it was trained on.

    DL heads encode symbol POSITION: the MLP flattens `[S*F]` and the RNN
    heads recur across the symbol axis, so a panel with another symbol set
    shifts every prediction. The checkpoint's config.json now records the
    training symbols. A fresh instance loads them without collecting a panel
    (the old `_read_checkpoint` sized the net from an empty data backend and
    raised), and `predict_panel` reorders the input onto the training symbols
    and drops, with a warning, the symbol the model never saw. The old code
    predicted the extra symbol `S9` and passed the reversed four-symbol layout
    to the module, so this test goes red.
    """
    trained, checkpoint = _trained_dl_checkpoint(tmp_path)
    sidecar = json.loads((checkpoint.parent / "config.json").read_text())
    assert sidecar["trained_on"]["symbols"] == SYMBOLS

    fresh = LinearDLHead(DLConfig(**_dl_train_kwargs(tmp_path)))
    fresh.load(checkpoint)
    base = _features(trained)
    wider = xr.concat(
        [base, base.isel(symbol=[0]).assign_coords(symbol=["S9"])], dim="symbol"
    ).isel(symbol=slice(None, None, -1))

    pred = fresh.predict_panel(wider)

    assert pred.symbol.values.tolist() == SYMBOLS
    xr.testing.assert_allclose(pred, fresh.predict_panel(base))
    assert any("S9" in m and "WR-02" in m for m in warning_messages), warning_messages


@pytest.mark.parametrize(
    "rename",
    [{"drop": "S2"}, {"S2": "S7"}],
    ids=["dropped-symbol", "same-count-other-member"],
)
def test_dl_predict_panel_refuses_a_panel_missing_a_training_symbol(tmp_path, rename):
    """Code review WR-02: a DL head cannot predict without a training symbol's inputs.

    The second case keeps the symbol COUNT and swaps one member. That is the
    silent MLP failure the review describes: `nn.Linear(S*F)` only checks the
    count. The error must name the missing symbol. The old code accepted both
    panels, so both cases go red.
    """
    _, checkpoint = _trained_dl_checkpoint(tmp_path)
    fresh = LinearDLHead(DLConfig(**_dl_train_kwargs(tmp_path)))
    fresh.load(checkpoint)
    features = _features(fresh)
    if "drop" in rename:
        features = features.drop_sel(symbol=rename["drop"])
    else:
        features = features.assign_coords(
            symbol=[rename.get(s, s) for s in features.symbol.values.tolist()]
        )

    with pytest.raises(ValueError, match="S2"):
        fresh.predict_panel(features)


# --------------------------------------------------------------------------
# 03.11-04: the checkpoint symbol contract on an int64 PERMNO axis
#
# The `trained_on.symbols` record has THREE ends -- the write in
# `_save_model`, the read in `_read_trained_symbols`, and the alignment in
# `DLModel._align_prediction_symbols` -- and every one of them used to run the
# labels through an unconditional `str()`. On a ticker axis that is the
# identity, so the three ends agreed by accident. On an int64 PERMNO axis they
# produce the failure 03.11-RESEARCH B measured: the membership check at the
# top of `_align_prediction_symbols` PASSES (both sides are stringified), and
# the very last line raises `KeyError: "not all values found in index
# 'symbol'"` -- an error about a missing INDEX ENTRY for what is really a
# dtype mismatch, which sends the operator looking for a symbol that is right
# there in the panel.
#
# The ticker arm below is a CONTROL: it is the same assertion on the same code
# path with string labels, and it is what proves the Tiingo/Alpaca path was
# not disturbed by the fix (D-02).
# --------------------------------------------------------------------------


def _permno_head(tmp_path) -> LinearDLHead:
    return LinearDLHead(
        DLConfig(**_dl_train_kwargs(tmp_path, symbols=PERMNOS))
    )


def test_int64_trained_symbols_align(tmp_path, warning_messages):
    """An int64 panel + an int64-recorded checkpoint align without raising.

    The mirror image of
    `test_dl_checkpoint_records_its_symbols_and_predict_panel_aligns_onto_them`
    with PERMNOs instead of tickers: a fourth symbol the model never saw is
    dropped with a warning, the rest come back in the training layout, and the
    prediction equals the one from the panel that never carried the extra.

    RED before the three-end fix: `.sel(symbol=sorted(trained))` is handed
    `['10107', '14593', '7000']` against an int64 coordinate and raises
    `KeyError: "not all values found in index 'symbol'"` -- AFTER the
    membership check above it reported everything present.
    """
    trained, checkpoint = _trained_dl_checkpoint(tmp_path, symbols=PERMNOS)
    fresh = _permno_head(tmp_path)
    fresh.load(checkpoint)
    base = _features(trained)
    wider = xr.concat(
        [base, base.isel(symbol=[0]).assign_coords(symbol=[99999])],
        dim="symbol",
    ).isel(symbol=slice(None, None, -1))

    pred = fresh.predict_panel(wider)

    assert pred.symbol.values.tolist() == SORTED_PERMNOS
    assert pred.symbol.dtype.kind == "i"
    xr.testing.assert_allclose(pred, fresh.predict_panel(base))
    assert any(
        "99999" in m and "WR-02" in m for m in warning_messages
    ), warning_messages


def test_string_trained_symbols_on_int_panel_names_the_dtype(tmp_path):
    """A str-recorded checkpoint against an int64 panel is REFUSED by name.

    This is the 03.11-RESEARCH B shape, reproduced by hand-editing the record
    the way a pre-migration checkpoint carries it. What must come back is a
    `ValueError` that says the two sides disagree on TYPE and names both --
    not the `KeyError` about a missing index entry, which describes a
    different defect than the one present. `pytest.raises(ValueError)` is
    itself half the assertion: a `KeyError` does not satisfy it.
    """
    _, checkpoint = _trained_dl_checkpoint(tmp_path, symbols=PERMNOS)
    sidecar = checkpoint.parent / "config.json"
    saved = json.loads(sidecar.read_text())
    saved["trained_on"]["symbols"] = [str(permno) for permno in SORTED_PERMNOS]
    sidecar.write_text(json.dumps(saved, indent=4))

    fresh = _permno_head(tmp_path)
    fresh.load(checkpoint)

    with pytest.raises(ValueError) as excinfo:
        fresh.predict_panel(_features(fresh))

    message = str(excinfo.value)
    assert "refusing to align" in message, message
    # Both sides named: the checkpoint's element type and the panel's dtype.
    assert "str" in message, message
    assert "int64" in message, message
    # And a way out, so the operator is not left holding a diagnosis only.
    assert "retrain" in message or "rewrite" in message, message


def test_ticker_trained_symbols_still_align_unchanged(tmp_path, warning_messages):
    """CONTROL ARM (D-02): the string axis behaves exactly as it did.

    Same code path, same assertions as `test_int64_trained_symbols_align`,
    with tickers. If the three-end fix reached the Tiingo/Alpaca path at all,
    this is where it shows: the record is still JSON strings, the alignment
    still returns the lexicographic ticker layout, and the extra symbol is
    still dropped with a warning.
    """
    trained, checkpoint = _trained_dl_checkpoint(tmp_path)
    recorded = json.loads((checkpoint.parent / "config.json").read_text())
    assert recorded["trained_on"]["symbols"] == SYMBOLS
    assert all(type(s) is str for s in recorded["trained_on"]["symbols"])

    fresh = LinearDLHead(DLConfig(**_dl_train_kwargs(tmp_path)))
    fresh.load(checkpoint)
    assert fresh._trained_symbols == SYMBOLS
    base = _features(trained)
    wider = xr.concat(
        [base, base.isel(symbol=[0]).assign_coords(symbol=["S9"])], dim="symbol"
    ).isel(symbol=slice(None, None, -1))

    pred = fresh.predict_panel(wider)

    assert pred.symbol.values.tolist() == SYMBOLS
    xr.testing.assert_allclose(pred, fresh.predict_panel(base))
    assert any(
        "S9" in m and "WR-02" in m for m in warning_messages
    ), warning_messages


def _ticker_lookup(tmp_path, *, sidecar: str):
    """A `CrspTickerLookup` over a hand-written sidecar in one of three states.

    Hand-written rather than produced by a conversion: this file is about what
    the MODEL does with a labeller, and `tests/test_crsp_ticker_sidecar.py`
    already owns whether a conversion writes the file correctly.

    `sidecar` is a three-state string rather than the `write: bool` this helper
    started with, because "not written" now has two meanings that behave
    differently inside the lookup:

    - `"valid"`   -- a well-formed sidecar; 99999 spells GHOST
    - `"absent"`  -- no file at all; the `payload` property's `FileNotFoundError`
    - `"corrupt"` -- a file that PARSES and whose spans lack `start`/`end`

    `"corrupt"` is deliberately not `"{not json"`: that is a parse failure,
    guarded by the `payload` property since 03.11-09 and already covered in
    `tests/test_crsp_ticker_sidecar.py`. It could never have reached the span
    indexing that G-03.11-3 is about.
    """
    from quantlab.dataset.crsp_tickers import CrspTickerLookup

    spans = {
        "valid": [{"ticker": "GHOST", "start": "1990-01-01", "end": "2025-12-31"}],
        "corrupt": [{"ticker": "GHOST"}],
    }
    path = tmp_path / "prices.zarr.crsp_tickers.json"
    if sidecar != "absent":
        path.write_text(
            json.dumps(
                {
                    "generated_from": "stksecurityinfohist",
                    "vintage_product_end": "2025-12-31",
                    "intervals": {"99999": spans[sidecar]},
                }
            ),
            encoding="utf-8",
        )
    return CrspTickerLookup(path)


def test_a_symbol_labeller_spells_the_dropped_permnos(tmp_path, warning_messages):
    """03.11-09: the WR-02 warning says GHOST, not 99999.

    The model layer knows no vendor and no store path -- the backtester hands
    it a `(symbols, day) -> list[str]` callable and nothing else
    (`base/backtest.py:_align_and_predict`). The labeller touches the MESSAGE
    only: the panel is still selected by the int64 identity, which is why the
    returned axis below is unchanged.
    """
    trained, checkpoint = _trained_dl_checkpoint(tmp_path, symbols=PERMNOS)
    fresh = _permno_head(tmp_path)
    fresh.load(checkpoint)
    fresh.symbol_labeller = _ticker_lookup(tmp_path, sidecar="valid").label
    base = _features(trained)
    wider = xr.concat(
        [base, base.isel(symbol=[0]).assign_coords(symbol=[99999])],
        dim="symbol",
    ).isel(symbol=slice(None, None, -1))

    pred = fresh.predict_panel(wider)

    assert pred.symbol.values.tolist() == SORTED_PERMNOS
    assert any(
        "GHOST" in m and "WR-02" in m for m in warning_messages
    ), warning_messages


def test_a_missing_ticker_sidecar_leaves_the_warning_working(
    tmp_path, warning_messages
):
    """T-03.11-30: a labeller pointed at a file that does not exist must not
    turn a warning into a crash.

    Same run, sidecar absent. The warning falls back to the digits, which is
    exactly what it printed before 03.11-09.
    """
    trained, checkpoint = _trained_dl_checkpoint(tmp_path, symbols=PERMNOS)
    fresh = _permno_head(tmp_path)
    fresh.load(checkpoint)
    fresh.symbol_labeller = _ticker_lookup(tmp_path, sidecar="absent").label
    base = _features(trained)
    wider = xr.concat(
        [base, base.isel(symbol=[0]).assign_coords(symbol=[99999])],
        dim="symbol",
    ).isel(symbol=slice(None, None, -1))

    pred = fresh.predict_panel(wider)

    assert pred.symbol.values.tolist() == SORTED_PERMNOS
    assert any(
        "99999" in m and "WR-02" in m for m in warning_messages
    ), warning_messages


def test_a_corrupt_ticker_sidecar_leaves_the_warning_working(
    tmp_path, warning_messages
):
    """G-03.11-3: a sidecar that PARSES and is shaped wrong must not crash a
    prediction that was going to succeed.

    The worst call site in the repo is the `_spell` in `predict_panel`'s
    `extra` branch: it is a BARE call, inside no `try`, and it sits on the
    HAPPY PATH -- a panel carrying symbols the model never trained on is merely
    dropped with a `logger.warning` and the prediction completes normally. Until
    03.11-12 a half-written audit file -- a file whose only job is to put
    letters in a log line -- turned that successful `predict_panel` into an
    `AttributeError`/`KeyError`.

    The missing-sidecar twin above only exercised the `payload` property's
    guard. This one gets past it: the JSON parses, `intervals` is a real dict,
    and the damage only surfaces when a span is indexed for `start`.

    This test does NOT touch `quantlab/base/model.py`, and that is the point.
    The contract belongs to the lookup, where it is written down; wrapping each
    of the six display points in its own `try` would be the same guard copied
    six times, with six chances to forget the seventh.
    """
    trained, checkpoint = _trained_dl_checkpoint(tmp_path, symbols=PERMNOS)
    fresh = _permno_head(tmp_path)
    fresh.load(checkpoint)
    fresh.symbol_labeller = _ticker_lookup(tmp_path, sidecar="corrupt").label
    base = _features(trained)
    wider = xr.concat(
        [base, base.isel(symbol=[0]).assign_coords(symbol=[99999])],
        dim="symbol",
    ).isel(symbol=slice(None, None, -1))

    pred = fresh.predict_panel(wider)

    assert pred.symbol.values.tolist() == SORTED_PERMNOS
    assert any(
        "99999" in m and "WR-02" in m for m in warning_messages
    ), warning_messages
    assert not any("GHOST" in m for m in warning_messages), warning_messages


def test_int64_checkpoint_records_json_integers(tmp_path):
    """`trained_on.symbols` is a JSON INTEGER array on an int64 panel.

    The write end. A quoted `"7000"` in the sidecar is not a cosmetic
    difference: it is what the read end hands to `.sel()`, and it is the
    reason the old alignment raised. Asserted on the JSON TEXT as well as on
    the parsed value, because `json.loads` would happily give `['7000']` back
    and the parsed-value assertion alone cannot see the quotes.
    """
    _, checkpoint = _trained_dl_checkpoint(tmp_path, symbols=PERMNOS)
    raw = (checkpoint.parent / "config.json").read_text()
    recorded = json.loads(raw)["trained_on"]["symbols"]

    assert recorded == SORTED_PERMNOS
    assert all(type(value) is int for value in recorded)
    assert '"7000"' not in raw, raw

    fresh = _permno_head(tmp_path)
    fresh.load(checkpoint)
    assert fresh._trained_symbols == SORTED_PERMNOS
    assert all(type(value) is int for value in fresh._trained_symbols)


def test_four_digit_permno_aligns_in_numeric_order(tmp_path):
    """The aligned axis is `[7000, 10107, ...]`, not `['10107', ..., '7000']`.

    `_align_prediction_symbols`'s last line is the third bare `sorted()` the
    03.11-02 numeric-order contract had to absorb. A five-digit-only universe
    cannot tell the two orders apart, so the divergence is asserted to be LIVE
    in this fixture rather than assumed -- if `PERMNOS` ever loses its
    four-digit member this test says so instead of silently passing.
    """
    _, checkpoint = _trained_dl_checkpoint(tmp_path, symbols=PERMNOS)
    fresh = _permno_head(tmp_path)
    fresh.load(checkpoint)

    aligned = fresh._align_prediction_symbols(
        _features(fresh)[FACTORS].sortby(["timestamp", "symbol"])
    )

    assert aligned.symbol.values.tolist() == SORTED_PERMNOS
    lexicographic = sorted(PERMNOS, key=str)
    assert lexicographic != SORTED_PERMNOS, (
        "the fixture lost its four-digit PERMNO, so this test can no longer "
        "distinguish numeric order from lexicographic order"
    )


# --------------------------------------------------------------------------
# G-03.7-8: the symbol-sorted layout is the contract, whatever the record order
# --------------------------------------------------------------------------

#: Small MLP widths. `MLPRegressor` flattens `[T, S*F]`, so its output for a
#: symbol depends on that symbol's POSITION: the head that can see a mislabel.
_MLP_HP = {"hidden_size1": 16, "hidden_size2": 8}


def _mlp_config(tmp_path) -> DLConfig:
    return DLConfig(**_dl_train_kwargs(tmp_path), hyperparameters=dict(_MLP_HP))


def _mlp_sorted_layout_prediction(model, features, symbols) -> np.ndarray:
    """The network run on the symbol-sorted layout of `symbols`, without `to_array`.

    `[T, len(symbols), L]`, in the sorted symbol order. That is the layout
    `DLModel._fit` trains on (`to_tensor -> to_array` sorts the symbol axis).
    """
    ordered = sorted(symbols)
    x = _stack(features.sel(symbol=ordered), FACTORS)
    flat = torch.from_numpy(
        x.reshape(N_TIMES, len(ordered) * len(FACTORS))
    ).float()
    return (
        model.predict(flat)
        .detach()
        .cpu()
        .numpy()
        .reshape(N_TIMES, len(ordered), len(LABELS))
    )


def _assert_each_coord_holds_its_own_prediction(pred, expected, symbols):
    """Each `(symbol, label)` column of `pred` equals that symbol's ground truth."""
    for i, symbol in enumerate(sorted(symbols)):
        for j, label in enumerate(LABELS):
            np.testing.assert_allclose(
                pred[label].sel(symbol=symbol).values,
                expected[:, i, j],
                atol=1e-6,
                err_msg=f"coordinate {symbol}/{label} holds another symbol's prediction",
            )


@pytest.mark.parametrize(
    "record",
    [["S1", "S0", "S2"], ["S2", "S1", "S0"], ["S2", "S0", "S1"]],
    ids=["one-swap", "reversed", "rotated"],
)
def test_dl_predict_panel_is_correct_for_an_unsorted_training_record(
    tmp_path, record
):
    """G-03.7-8: an unsorted `trained_on.symbols` must not move any prediction.

    The network was trained on the symbol-sorted layout, so the record decides
    only WHICH symbols the head knows, never their order. The record is
    rewritten here to an unsorted order, as a hand edit or another producer
    would. The head is `MLPRegressor`, which is position-sensitive.
    Each coordinate must carry the network's output for that symbol on the
    sorted layout. The old code selected the panel in record order and took
    coords from it while `to_array` re-sorted the values, so every position
    where the record differs from sorted order held another symbol's
    prediction.
    """
    trained = MLPRegressor(_mlp_config(tmp_path))
    trained.collect()
    checkpoint = trained.train()
    sidecar = checkpoint.parent / "config.json"
    saved = json.loads(sidecar.read_text())
    saved["trained_on"]["symbols"] = record
    sidecar.write_text(json.dumps(saved, indent=4))

    fresh = MLPRegressor(_mlp_config(tmp_path))
    fresh.load(checkpoint)
    features = _features(fresh)

    pred = fresh.predict_panel(features)

    expected = _mlp_sorted_layout_prediction(fresh, features, SYMBOLS)
    _assert_each_coord_holds_its_own_prediction(pred, expected, SYMBOLS)
    assert pred.symbol.values.tolist() == SYMBOLS


class UnsortedHookLinearDLHead(LinearDLHead):
    """A DL head whose symbol hook hands back an unsorted panel."""

    def _align_prediction_symbols(self, feats):
        return feats.isel(symbol=[2, 0, 1])


def test_a_hook_returning_an_unsorted_panel_cannot_mislabel_coords(tmp_path):
    """G-03.7-8: predict_panel's coords are the panel `to_array` lays out.

    `_align_prediction_symbols` is a hook, so a future override can return
    any symbol order. `predict_panel` must re-sort after the hook and read its
    coords from that re-sorted panel, so coords and values cannot disagree.
    The old code read coords from the hook's panel (S2, S0, S1) while
    `to_array` laid the values out sorted.
    """
    model = UnsortedHookLinearDLHead(DLConfig(**_config_kwargs(tmp_path)))
    model.collect()
    model._init_model_and_optim()
    plain = LinearDLHead(DLConfig(**_config_kwargs(tmp_path)))
    plain.collect()
    plain._init_model_and_optim()
    plain.model.load_state_dict(model.model.state_dict())  # type: ignore[union-attr]
    features = _features(model)

    pred = model.predict_panel(features)

    assert pred.symbol.values.tolist() == SYMBOLS
    xr.testing.assert_allclose(pred, plain.predict_panel(features))


def test_checkpoint_config_json_with_the_training_record_rebuilds_the_model(tmp_path):
    """Code review WR-02: the new `trained_on` record does not break the config loader.

    `load_model_from_config` refuses unknown keys, so it must drop the record
    (as it drops `resolved_hyperparameters`) and rebuild a model whose config
    equals the one that trained. This goes red without the record, and red
    if the loader chokes on it.
    """
    from tests.backtest_fixtures import make_model, train_checkpoint, write_price_store

    dataset_config = write_price_store(tmp_path / "store", n_bars=40)
    dates = dict(
        start_date="2024-01-01",
        end_date="2024-02-23",
        train_start="2024-01-01",
        train_end="2024-01-31",
        test_start="2024-02-01",
        test_end="2024-02-23",
    )
    model = make_model(tmp_path / "train", dataset_config, **dates)
    checkpoint = train_checkpoint(model)
    saved = json.loads((checkpoint.parent / "config.json").read_text())
    assert saved["trained_on"]["symbols"] == sorted(model.symbols)

    rebuilt = module_utils.load_model_from_config(saved)

    assert json.loads(json.dumps(rebuilt.get_config())) == json.loads(
        json.dumps(model.get_config())
    )


def _mlp_on_a_backend_filled_without_collect(tmp_path, symbols) -> MLPRegressor:
    """An MLPRegressor whose data backend holds the panel in `symbols` order.

    `collect()` sorts the symbol axis. This fills the backend directly, as a
    caller that calls `data_backend.to_internal` and then `train()` would, so the
    backend order is whatever `symbols` says.
    """
    model = MLPRegressor(_mlp_config(tmp_path))
    panel = xr.merge(
        [model.config.factors[0].get_features(), model.config.labels[0].get_labels()]
    )
    model.data_backend.to_internal(panel.sel(symbol=list(symbols)))
    assert model.symbols == list(symbols)
    return model


def test_train_on_an_unsorted_backend_records_sorted_symbols_and_predicts_correctly(
    tmp_path,
):
    """G-03.7-8: the training record describes the layout the network trained on.

    `DLModel._fit` trains through `to_array`, which sorts the symbol axis, so
    a backend in S2, S0, S1 order still trains on S0, S1, S2. The record in
    config.json and `_trained_symbols` must say S0, S1, S2. The same
    instance's `predict_panel` must put each MLP prediction on its own
    coordinate. The old code recorded the backend order (S2, S0, S1).
    """
    model = _mlp_on_a_backend_filled_without_collect(tmp_path, ["S2", "S0", "S1"])

    checkpoint = model.train()

    sidecar = json.loads((checkpoint.parent / "config.json").read_text())
    assert sidecar["trained_on"]["symbols"] == SYMBOLS
    assert model._trained_symbols == SYMBOLS

    features = _features(model)
    pred = model.predict_panel(features)

    assert pred.symbol.values.tolist() == SYMBOLS
    expected = _mlp_sorted_layout_prediction(model, features, SYMBOLS)
    _assert_each_coord_holds_its_own_prediction(pred, expected, SYMBOLS)


def test_train_cv_on_an_unsorted_backend_records_sorted_symbols_in_every_fold(
    tmp_path,
):
    """G-03.7-8: every train_cv fold checkpoint records the sorted training layout.

    30 timestamps with `train_periods=20` give 2 folds. Each fold goes through
    `_save_model`, so each sidecar must record S0, S1, S2 even though the
    backend holds S2, S0, S1. The old code recorded the backend order in
    every fold.
    """
    model = _mlp_on_a_backend_filled_without_collect(tmp_path, ["S2", "S0", "S1"])

    results = model.train_cv(train_periods=20)

    assert len(results) == 2, results
    for result in results:
        sidecar = json.loads(
            (Path(result["checkpoint"]).parent / "config.json").read_text()
        )
        assert sidecar["trained_on"]["symbols"] == SYMBOLS, result["fold"]


def test_single_symbol_training_record_predicts_only_that_symbol(
    tmp_path, warning_messages
):
    """G-03.7-8 boundary: a one-symbol record through the drop-extras path.

    A one-element record is always sorted, so this passes on arrival. It is kept
    as a boundary lock: a network trained on S1 alone must predict only S1 on the
    full three-symbol panel. The warning must name the dropped S0 and S2, and
    the value must be the network's output on the S1-only flat layout.
    """
    trained = _mlp_on_a_backend_filled_without_collect(tmp_path, ["S1"])
    checkpoint = trained.train()

    fresh = MLPRegressor(_mlp_config(tmp_path))
    fresh.load(checkpoint)
    features = _features(fresh)

    pred = fresh.predict_panel(features)

    assert pred.symbol.values.tolist() == ["S1"]
    assert any(
        "S0" in m and "S2" in m and "WR-02" in m for m in warning_messages
    ), warning_messages
    expected = _mlp_sorted_layout_prediction(fresh, features, ["S1"])
    _assert_each_coord_holds_its_own_prediction(pred, expected, ["S1"])


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
