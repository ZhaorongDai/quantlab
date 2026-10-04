"""`BaseModel.load()` checks the checkpoint's variables against the model (phase 03.7, G-03.7-9).

Every trained unit's `run.json` holds a training record `trained_on`:
the factor names, label names and symbols, produced by the same
`get_factor_names()` / `get_label_names()` calls whose order built the training
arrays. Before G-03.7-9 nothing read the factor and label names back.
Neither head validates variable identity by itself: the XGBoost Booster is
nameless and `inplace_predict` on numpy checks only the column count, and a
torch `state_dict` checks only tensor shapes. So a model whose factor or label
list was reordered loaded and predicted silently on permuted inputs, or
labelled its outputs with the wrong variables.

What is locked here, and what turns it red:

- a fresh XGBoostRegressor or torch head whose factor list or label list is
  reordered refuses the checkpoint with a ValueError naming both lists and the
  path; for torch the refusal comes before `_read_checkpoint`, so a different
  factor COUNT gets the named error instead of torch's "size mismatch";
- an identical fresh model loads and predicts exactly like the trained one;
- the check is keyed on `trained_on`, never on the factor config field
  `factors[].factor_names`, which a user can set in another order than the
  names training actually used;
- a checkpoint without a readable `run.json` (written before #122, or
  corrupt) is refused before anything is loaded.

Everything is synthetic, CPU-only and offline. The stand-ins are local copies
in the style of `tests/test_xgb_model.py` and `tests/test_model_predict_panel.py`,
not imports. This module imports torch and xgboost in one process;
`tests/conftest.py` sets `OMP_NUM_THREADS=1` on macOS for exactly that.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from loguru import logger

from quantlab.base.config import ModelConfig
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.runs.trained_run import TrainedRun
from tests.torch_heads import OneBarHead
from tests.label_stubs import StubLabel

N_TIMES = 40
SYMBOLS = ["S0", "S1", "S2", "S3"]
N_SYMBOLS = len(SYMBOLS)
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype("timedelta64[D]")


def _day(i: int) -> str:
    return np.datetime_as_string(TIMES[i], unit="D")


START, TRAIN_END, TEST_START, END = _day(0), _day(29), _day(30), _day(N_TIMES - 1)

FACTORS = ["f_signal", "f_second", "f_noise"]
LABELS = ["ret_a", "ret_b"]

_rng = np.random.default_rng(20260915)
_SHAPE = (N_TIMES, N_SYMBOLS)
#: One fixed array per variable name, so a panel that lists the same names in
#: another order carries exactly the same data per name.
ARRAYS = {name: _rng.standard_normal(_SHAPE) for name in FACTORS}
ARRAYS["ret_a"] = 0.1 * ARRAYS["f_signal"] + 0.05 * _rng.standard_normal(_SHAPE)
ARRAYS["ret_b"] = -0.1 * ARRAYS["f_second"] + 0.05 * _rng.standard_normal(_SHAPE)

LIBRARY_HYPER = {"num_boost_round": 5, "nthread": 1}
MODEL_WARNING_TAG = "this model's declared"


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


class NamedPanel:
    """A stand-in for a factor/label object over the fixed `ARRAYS`.

    `names` is what `_get_factor_names()` derives (and what training uses).
    `config_names` is what `get_config()` reports as the config field
    `factor_names`; it defaults to `names`.
    """

    def __init__(self, names, config_names=None):
        self.names = list(names)
        self._config_names = list(names) if config_names is None else list(config_names)
        self._ds = xr.Dataset(
            {
                name: (("timestamp", "symbol"), ARRAYS[name].astype("float32"))
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
        return {"name": "NamedPanel", "factor_names": list(self._config_names)}


def _common_kwargs(root: Path, factors: NamedPanel, labels: NamedPanel) -> dict:
    return dict(
        factors=[factors],
        labels=[StubLabel(labels)],
        model_save_dir=str(root),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
        train_start=START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=END,
    )


def _library_model(root: Path, factor_names=FACTORS, label_names=LABELS, **panel_kwargs):
    return XGBoostRegressor(
        ModelConfig(
            **_common_kwargs(
                root, NamedPanel(factor_names, **panel_kwargs), NamedPanel(label_names)
            ),
            hyperparameters=dict(LIBRARY_HYPER),
        )
    )


def _torch_model(root: Path, factor_names=FACTORS, label_names=LABELS) -> OneBarHead:
    return OneBarHead(
        ModelConfig(
            **_common_kwargs(root, NamedPanel(factor_names), NamedPanel(label_names)),
            hyperparameters={"epochs": 1},
        )
    )


def _train(model, suffix: str) -> Path:
    model.collect()
    model.train()
    found = sorted(Path(model.config.model_save_dir).rglob(f"*{suffix}"))
    assert len(found) == 1, found
    return found[0]


def _run_json(checkpoint: Path) -> Path:
    return checkpoint.parent / "run.json"


def _model_warnings(messages: list[str]) -> list[str]:
    return [m for m in messages if MODEL_WARNING_TAG in m]


REORDER_CASES = [
    pytest.param(dict(factor_names=["f_noise", "f_second", "f_signal"]), "factor", id="reordered-factors"),
    pytest.param(dict(label_names=["ret_b", "ret_a"]), "label", id="reordered-labels"),
]


# --------------------------------------------------------------------------
# Task 1: the model-level check keyed on trained_on
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("fresh_kwargs", "kind"), REORDER_CASES)
def test_library_load_refuses_reordered_factors_and_labels(tmp_path, fresh_kwargs, kind):
    """An XGBoostRegressor checkpoint trained on [f_signal, f_second, f_noise]
    -> [ret_a, ret_b] refuses a fresh model that lists the same factors (or
    labels) in another order. Before G-03.7-9 `load()` never read the names and
    this loaded silently, so both ids go red on "DID NOT RAISE"."""
    checkpoint = _train(_library_model(tmp_path / "train"), ".joblib")
    fresh = _library_model(tmp_path / "fresh", **fresh_kwargs)

    with pytest.raises(ValueError, match="was trained on") as excinfo:
        fresh.load(checkpoint)

    message = str(excinfo.value)
    recorded = FACTORS if kind == "factor" else LABELS
    declared = fresh.get_factor_names() if kind == "factor" else fresh.get_label_names()
    assert str(checkpoint) in message
    assert str(recorded) in message
    assert str(declared) in message
    assert fresh.model is None


@pytest.mark.parametrize(
    ("fresh_kwargs", "kind"),
    [
        *REORDER_CASES,
        pytest.param(dict(factor_names=["f_signal", "f_second"]), "factor", id="fewer-factors"),
    ],
)
def test_torch_load_refuses_reordered_variables_before_reading_the_checkpoint(
    tmp_path, monkeypatch, fresh_kwargs, kind
):
    """A torch head's refusal comes before `_read_checkpoint` builds the network.

    Reordered variables keep every weight shape, so before G-03.7-9 they
    loaded silently. A different factor COUNT failed inside torch with an
    unnamed "size mismatch" RuntimeError, which `pytest.raises(ValueError)`
    does not catch. All three ids go red."""
    checkpoint = _train(_torch_model(tmp_path / "train"), ".pth")
    fresh = _torch_model(tmp_path / "fresh", **fresh_kwargs)
    reads: list[Path] = []
    read_checkpoint = fresh._read_checkpoint
    monkeypatch.setattr(
        fresh, "_read_checkpoint", lambda p: reads.append(p) or read_checkpoint(p)
    )

    with pytest.raises(ValueError, match="was trained on") as excinfo:
        fresh.load(checkpoint)

    assert str(checkpoint) in str(excinfo.value)
    assert f"{kind} variables" in str(excinfo.value)
    assert reads == []


def test_identical_model_loads_and_predicts_identically(tmp_path, warning_messages):
    trained = _library_model(tmp_path / "train")
    checkpoint = _train(trained, ".joblib")
    fresh = _library_model(tmp_path / "fresh")

    fresh.load(checkpoint)

    features = fresh.config.factors[0]._ds
    expected = trained.predict_panel(features)
    actual = fresh.predict_panel(features)
    assert list(actual.data_vars) == LABELS
    for name in LABELS:
        np.testing.assert_allclose(actual[name].values, expected[name].values, rtol=0, atol=1e-12)
    assert _model_warnings(warning_messages) == [], warning_messages


def test_trained_on_is_authoritative_over_the_factor_config_field(tmp_path, warning_messages):
    """The factor's config field `factor_names` says [f_noise, f_signal, f_second];
    the names training really used (and `trained_on` records) are
    [f_signal, f_second, f_noise]. The old backtester-side check compared the
    config field and refused this model's own checkpoint (a measured false
    positive). The model-level check must load it without error or warning."""
    config_names = ["f_noise", "f_signal", "f_second"]
    checkpoint = _train(_library_model(tmp_path / "train", config_names=config_names), ".joblib")
    run = TrainedRun.open(checkpoint)
    assert run.config["factors"][0]["factor_names"] == config_names
    assert run.trained_on["factor_names"] == FACTORS

    fresh = _library_model(tmp_path / "fresh", config_names=config_names)
    fresh.load(checkpoint)

    assert fresh.model is not None
    assert _model_warnings(warning_messages) == [], warning_messages


# --------------------------------------------------------------------------
# Task 2: a checkpoint without a readable run.json
# --------------------------------------------------------------------------


def test_a_checkpoint_without_run_json_is_refused_before_loading(tmp_path):
    """A checkpoint trained before run.json existed has nothing to check its
    variables against; it is refused with a message to retrain, not loaded
    unchecked."""
    checkpoint = _train(_library_model(tmp_path / "train"), ".joblib")
    _run_json(checkpoint).unlink()
    fresh = _library_model(tmp_path / "fresh")

    with pytest.raises(ValueError, match="no run.json.*retrain"):
        fresh.check_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="no run.json.*retrain"):
        fresh.load(checkpoint)
    assert fresh.model is None


def test_a_run_json_that_is_not_an_object_is_refused(tmp_path):
    """Corruption is not absence: valid JSON that is not a run record is
    refused, naming the file, and nothing is loaded."""
    checkpoint = _train(_library_model(tmp_path / "train"), ".joblib")
    _run_json(checkpoint).write_text(json.dumps(["not", "a", "run"]))
    fresh = _library_model(tmp_path / "fresh")

    with pytest.raises(ValueError, match="format_version None") as excinfo:
        fresh.load(checkpoint)

    assert str(_run_json(checkpoint)) in str(excinfo.value)
    assert fresh.model is None
