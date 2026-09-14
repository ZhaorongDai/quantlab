"""Orchestration tests for `quantlab/base/model.py:MLModel` (quick task 260914-lno).

`MLModel` is the non-torch variant of the model layer: no epoch loop, one
`_fit_model` call per fit, native early stopping left to the library. These
tests drive it through a purely numpy stub head so they lock the ORCHESTRATION
-- what `_fit` hands the hooks, what it writes to W&B and to disk, how `load`
and `predict` behave -- independently of any real library. The xgboost head is
covered in `tests/test_xgb_model.py`.

W&B assertions patch `_init_wandb` on the CLASS and collect every recorder the
model creates, because `train_cv(parallel=True)` deep-copies the instance and
an instance-level patch would attach recorders to the original, not the copy.

Everything is synthetic, CPU-only and offline.
"""

from pathlib import Path
from types import SimpleNamespace
import warnings

import joblib
import numpy as np
import pytest
import torch
import xarray as xr

from quantlab.base.config import MLConfig
from quantlab.base.model import BaseModel, MLModel

N_TIMES = 130
N_SYMBOLS = 4
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)
TRAIN_START = np.datetime_as_string(TIMES[0], unit="D")
TRAIN_END = np.datetime_as_string(TIMES[99], unit="D")
TEST_START = np.datetime_as_string(TIMES[100], unit="D")
TEST_END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")
N_TRAIN_TIMES = 100
N_TEST_TIMES = 30

METRIC_KEYS = ("loss", "mse", "rmse", "mae", "r2", "ic", "rank_ic")


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


class FakePanel:
    """A stand-in for a factor/label object: only what `collect()` calls.

    `values=None` fills each variable with pseudo-random numbers; a dict of
    constants fills each variable with its constant, so the value in column
    *i* identifies which variable landed there.
    """

    def __init__(self, names, seed=0, values=None):
        rng = np.random.default_rng(seed)
        self.names = list(names)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    (
                        np.full((N_TIMES, N_SYMBOLS), values[name], dtype="float32")
                        if values is not None
                        else rng.standard_normal((N_TIMES, N_SYMBOLS)).astype("float32")
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


class FakeRecorder:
    """Records what the model writes to a W&B run."""

    def __init__(self, name: str):
        self.name = name
        self.logs: list[tuple[dict, int | None]] = []
        self.summary: dict = {}
        self.finished = 0

    def log(self, data, step=None):
        self.logs.append((dict(data), step))

    def finish(self):
        self.finished += 1


@pytest.fixture
def recorders(monkeypatch) -> list[FakeRecorder]:
    created: list[FakeRecorder] = []

    def fake_init_wandb(self, project_name, experiment_name):
        recorder = FakeRecorder(experiment_name)
        created.append(recorder)
        self._wandb_recorder = recorder

    monkeypatch.setattr(BaseModel, "_init_wandb", fake_init_wandb)
    return created


class StubMLHead(MLModel):
    """A numpy head that records what `_fit` hands it.

    `_init_model` stores `num_labels` in the model dict, so `_forward` works on
    an instance that was loaded but never collected. `_forward` returns the
    first factor repeated over the label axis, which keeps IC finite on a
    random panel.
    """

    def __init__(self, config):
        super().__init__(config)
        self.init_model_calls = 0
        self.fit_calls: list[dict] = []

    def _init_model(self, num_features, num_labels, hyperparameters):
        self.init_model_calls += 1
        return {"num_labels": num_labels}

    def _preprocess(self, data):
        return np.nan_to_num(np.array(data, dtype=np.float64, copy=True), nan=0.0)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        self.fit_calls.append(
            {
                "train_x": train_x.shape,
                "train_y": train_y.shape,
                "val_x": None if val_x is None else val_x.shape,
                "val_y": None if val_y is None else val_y.shape,
                "train_x_first": train_x[0, 0].tolist(),
                "train_y_first": train_y[0, 0].tolist(),
            }
        )
        self.model = {**self.model, "offset": 0.25}

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1) + self.model["offset"]


def _config(tmp_path, *, val_size=0.2, factors=None, labels=None, save_dir="ckpt"):
    return MLConfig(
        factors=factors if factors is not None else [FakePanel(["f_a", "f_b"], seed=1)],
        labels=labels if labels is not None else [FakePanel(["ret_a"], seed=2)],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=TRAIN_START,
        end_date=TEST_END,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        val_size=val_size,
    )


def _trained(tmp_path, **kwargs) -> StubMLHead:
    model = StubMLHead(_config(tmp_path, **kwargs))
    model.collect()
    model.train()
    return model


def _checkpoint(tmp_path, save_dir="ckpt") -> Path:
    found = sorted((tmp_path / save_dir).rglob("*.joblib"))
    assert len(found) == 1, found
    return found[0]


# --------------------------------------------------------------------------
# _fit orchestration
# --------------------------------------------------------------------------


def test_fit_calls_fit_model_exactly_once(tmp_path, recorders):
    """No epoch loop: one `train()` is one `_fit_model` call. Turns red if an
    outer loop around the library's own training creeps back in."""
    model = _trained(tmp_path)
    assert len(model.fit_calls) == 1


def test_tail_validation_split_keeps_every_training_timestamp(tmp_path, recorders):
    """100 training timestamps with `val_size=0.2`: the first 80 train, the
    last 20 validate, and none is dropped. Turns red on a `train_split + 1`
    style off-by-one or a head/tail swap."""
    call = _trained(tmp_path).fit_calls[0]
    assert call["train_x"] == (80, N_SYMBOLS, 2)
    assert call["train_y"] == (80, N_SYMBOLS, 1)
    assert call["val_x"] == (20, N_SYMBOLS, 2)
    assert call["val_y"] == (20, N_SYMBOLS, 1)
    assert call["train_x"][0] + call["val_x"][0] == N_TRAIN_TIMES


def test_zero_val_size_passes_none_for_both_validation_arrays(tmp_path, recorders):
    """An empty validation segment is signalled with None, never with a
    zero-length array a library would choke on."""
    call = _trained(tmp_path, val_size=0.0).fit_calls[0]
    assert call["val_x"] is None and call["val_y"] is None
    assert call["train_x"][0] == N_TRAIN_TIMES


def test_full_val_size_raises_before_fit_model(tmp_path, recorders):
    """`val_size=1.0` leaves nothing to fit on; it must raise before the
    library is ever called rather than hand it an empty array."""
    model = StubMLHead(_config(tmp_path, val_size=1.0))
    model.collect()
    with pytest.raises(ValueError, match="Empty training segment"):
        model.train()
    assert model.fit_calls == []


def test_declared_factor_and_label_order_reaches_fit_model(tmp_path, recorders):
    """Both name lists are deliberately non-alphabetical; the constants
    identify which variable landed in which column. Turns red if the last axis
    is ever ordered by name again."""
    model = _trained(
        tmp_path,
        factors=[FakePanel(["zeta", "alpha", "mid"], values={"zeta": 1.0, "alpha": 2.0, "mid": 3.0})],
        labels=[
            FakePanel(
                ["ret_30", "ret_60", "ret_120"],
                values={"ret_30": 30.0, "ret_60": 60.0, "ret_120": 120.0},
            )
        ],
    )
    call = model.fit_calls[0]
    assert call["train_x_first"] == [1.0, 2.0, 3.0]
    assert call["train_y_first"] == [30.0, 60.0, 120.0]


def test_fit_returns_the_test_metrics(tmp_path, recorders):
    """`_fit` returns the prefixed test metrics -- the dict `train_cv` merges
    into each fold's result."""
    model = StubMLHead(_config(tmp_path))
    model.collect()
    model._init_wandb("p", "e")
    out = model._fit(project_name="p", experiment_name="e", model_name="e.joblib")
    assert set(out) == {f"test_{k}" for k in METRIC_KEYS}
    assert np.isfinite(out["test_loss"]) and np.isfinite(out["test_ic"])


# --------------------------------------------------------------------------
# W&B
# --------------------------------------------------------------------------


def test_summary_carries_all_split_metrics_and_run_finishes_once(tmp_path, recorders):
    """Exactly the 21 `{train,val,test}_{metric}` keys land in the run
    summary, and the run is finished exactly once."""
    _trained(tmp_path)
    assert len(recorders) == 1
    rec = recorders[0]
    assert set(rec.summary) == {
        f"{split}_{k}" for split in ("train", "val", "test") for k in METRIC_KEYS
    }
    assert rec.finished == 1


def test_no_val_metrics_without_a_validation_segment(tmp_path, recorders):
    _trained(tmp_path, val_size=0.0)
    keys = set(recorders[0].summary)
    assert not any(k.startswith("val_") for k in keys)
    assert {f"train_{k}" for k in METRIC_KEYS} | {f"test_{k}" for k in METRIC_KEYS} == keys


# --------------------------------------------------------------------------
# Persistence and inference
# --------------------------------------------------------------------------


def test_train_writes_one_joblib_and_config_json(tmp_path, recorders):
    """The ML path persists through `MlBackend` as `.joblib`; a `.pth` here
    would mean the torch persistence path ran."""
    model = _trained(tmp_path)
    files = {p.name for p in (tmp_path / "ckpt").rglob("*") if p.is_file()}
    assert files == {"StubMLHead_total.joblib", "config.json"}
    assert joblib.load(_checkpoint(tmp_path)) == model.model


def test_fresh_instance_loads_without_init_model_and_predicts_identically(tmp_path, recorders):
    """`MLModel.load` must not rebuild the model: the file IS the model, and a
    loaded-but-never-collected instance cannot know its feature count."""
    trained = _trained(tmp_path)
    fresh = StubMLHead(_config(tmp_path))

    fresh.load(_checkpoint(tmp_path))

    assert fresh.init_model_calls == 0
    x = np.random.default_rng(5).standard_normal((6, N_SYMBOLS, 2))
    assert np.array_equal(fresh.predict(x), trained.predict(x))


def test_load_rejects_a_pth_file_before_building_anything(tmp_path):
    model = StubMLHead(_config(tmp_path))
    wrong = tmp_path / "x.pth"
    wrong.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match=r"\.joblib"):
        model.load(wrong)
    assert model.init_model_calls == 0
    assert model.model is None


def test_predict_accepts_ndarray_and_tensor(tmp_path, recorders):
    model = _trained(tmp_path)
    x = np.random.default_rng(6).standard_normal((7, N_SYMBOLS, 2))
    out_np = model.predict(x)
    out_t = model.predict(torch.from_numpy(x))
    assert out_np.shape == (7, N_SYMBOLS, 1)
    assert np.array_equal(out_np, out_t)


def test_predict_rejects_other_types(tmp_path, recorders):
    model = _trained(tmp_path)
    with pytest.raises(TypeError):
        model.predict([[1.0, 2.0]])


def test_predict_before_train_or_load_raises(tmp_path):
    model = StubMLHead(_config(tmp_path))
    with pytest.raises(ValueError, match="Model not initialized"):
        model.predict(np.zeros((1, N_SYMBOLS, 2)))


# --------------------------------------------------------------------------
# Default loss
# --------------------------------------------------------------------------


def test_default_loss_excludes_rows_with_a_nan_label(tmp_path):
    """`(t, s)` rows where ANY label is missing are dropped entirely; the MSE
    is over every label of the remaining rows."""
    model = StubMLHead(_config(tmp_path))
    y = np.array([[[1.0, 2.0], [3.0, np.nan]], [[0.0, 0.0], [1.0, 1.0]]])
    pred = np.array([[[2.0, 2.0], [100.0, 100.0]], [[1.0, 1.0], [1.0, 3.0]]])
    # kept rows: (0,0) diffs 1,0; (1,0) diffs 1,1; (1,1) diffs 0,2 -> (1+0+1+1+0+4)/6
    assert model._loss(y, pred) == pytest.approx(7 / 6)


def test_default_loss_is_nan_without_valid_rows_and_does_not_warn(tmp_path):
    model = StubMLHead(_config(tmp_path))
    y = np.full((2, 2, 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isnan(model._loss(y, np.zeros_like(y)))
