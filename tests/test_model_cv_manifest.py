"""The walk-forward unit `BaseModel.train_cv` persists (#123; was `cv_folds.json`, D-30 / D-36).

`train_cv` writes `{model_save_dir}/{Class}_trial_{timestamp}/`, a
"walk_forward" trained unit: one "model" unit per fold in `fold_{i}/` and,
last, `run.json` holding `{"format_version": 1, "kind": "walk_forward",
"folds": [...], "cv_mean": {...}}`. It returns that unit as a `TrainedRun`.

Why the record exists: `run_cv` backtests every fold's out-of-sample segment,
and it can only replay the folds a training run actually used if that
geometry is on disk and equal to what training returned. Why it is versioned:
CV backtests of OLD training runs read it, so the on-disk shape is a persisted
format, and `TrainedRun.open` rejects a version it does not know.

What turns this file red:

- the record is missing, or its folds differ from the returned unit's, for a
  library or a torch head;
- a fold entry loses a key, or its fold unit has no checkpoint on disk;
- fold paths are not relative to the unit, so a relative `model_save_dir` or
  a copied trial directory breaks reading;
- an empty fold list writes no record (run_cv could then not tell "no folds"
  from "not a CV project");
- a non-finite metric is dumped as a bare `NaN` token, which strict JSON
  parsers reject (03.7-RESEARCH.md Pitfall 10).

The former test that the returned fold dicts carried no manifest keys is gone:
`train_cv` returns the unit itself, not a list of dicts.

Everything is synthetic, CPU-only and offline.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from quantlab.base.config import ModelConfig
from quantlab.model.library_model import LibraryModel
from quantlab.model.torch_model import TorchModel
from quantlab.utils.trained_run import TrainedRun
from tests.torch_heads import OneBarHead
from tests.label_stubs import StubLabel

N_TIMES = 40
N_SYMBOLS = 3
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)
START = np.datetime_as_string(TIMES[0], unit="D")
END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")

#: 40 timestamps, train 20, no gap -> test 4, 5 folds.
TRAIN_PERIODS = 20
N_FOLDS = 5

FOLD_ENTRY_KEYS = {
    "fold", "directory", "kind", "train_window", "fitted_train_window", "test_window", "metrics"
}


class FakePanel:
    """A stand-in for a factor/label object: only what `collect()` calls."""

    def __init__(self, names: list[str], seed: int):
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


class StubLibraryHead(LibraryModel):
    """A numpy head: predicts the first factor for every label."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


class NaNMetricLibraryHead(StubLibraryHead):
    """Every fold reports one non-finite metric, as a numpy scalar, beside a
    finite one -- the shape vectorbt-style and panel metrics really take."""

    def _compute_metrics(self, y, pred, split, timestamps):
        return {"nan_metric": np.float64("nan"), "finite_metric": np.float64(1.5)}


def _common(tmp_path: Path, save_dir: str) -> dict:
    return dict(
        factors=[FakePanel(["f_a", "f_b"], seed=1)],
        labels=[StubLabel(FakePanel(["ret"], seed=2))],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
    )


def _library(tmp_path: Path, save_dir: str, cls=StubLibraryHead) -> LibraryModel:
    model = cls(ModelConfig(**_common(tmp_path, save_dir)))
    model.collect()
    return model


def _torch(tmp_path: Path, save_dir: str) -> TorchModel:
    model = OneBarHead(ModelConfig(**_common(tmp_path, save_dir), hyperparameters={"epochs": 1}))
    model.collect()
    return model


def _project_dir(save_root: Path) -> Path:
    assert save_root.is_dir(), f"train_cv created no save root at {save_root}"
    projects = [p for p in save_root.iterdir() if p.is_dir()]
    assert len(projects) == 1, f"expected one CV project dir, got {projects}"
    return projects[0]


def _read_record(save_root: Path) -> dict:
    """Parse the walk-forward unit's run.json STRICTLY: a bare NaN/Infinity token raises."""
    path = _project_dir(save_root) / "run.json"
    assert path.is_file(), f"train_cv wrote no run.json at {path}"

    def _reject(token):
        raise ValueError(f"non-standard JSON token {token!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def _fold_tuple(fold: TrainedRun) -> tuple:
    return (fold.index, fold.train_window, fold.fitted_train_window, fold.test_window)


def test_sequential_ml_record_equals_returned_folds(tmp_path):
    """The record's folds are the returned unit's folds, and the wrapper
    carries `format_version` 1, the kind, the folds and `cv_mean`."""
    model = _library(tmp_path, "ckpt")

    cv = model.train_cv(train_periods=TRAIN_PERIODS)

    record = _read_record(tmp_path / "ckpt")
    assert set(record) == {"format_version", "kind", "folds", "cv_mean"}
    assert record["format_version"] == 1 and record["kind"] == "walk_forward"
    assert len(cv.folds) == N_FOLDS
    assert [
        (e["fold"], tuple(e["train_window"]), tuple(e["fitted_train_window"]),
         tuple(e["test_window"]))
        for e in record["folds"]
    ] == [_fold_tuple(f) for f in cv.folds]
    assert [e["metrics"] for e in record["folds"]] == [f.metrics for f in cv.folds]
    assert record["cv_mean"] == cv.cv_mean
    assert TrainedRun.open(cv.path) == cv


def test_torch_record_equals_returned_folds(tmp_path):
    """A torch fold unit carries its windows, its checkpoint and every split's
    metrics, like a library fold."""
    model = _torch(tmp_path, "ckpt")

    cv = model.train_cv(train_periods=TRAIN_PERIODS)

    record = _read_record(tmp_path / "ckpt")
    assert len(record["folds"]) == len(cv.folds) == N_FOLDS
    for entry, fold in zip(record["folds"], cv.folds):
        assert set(entry) == FOLD_ENTRY_KEYS
        assert {"train_mse", "val_mse", "test_mse", "test_rank_ic"} <= set(entry["metrics"])
        assert fold.checkpoint.suffix == ".pth" and fold.checkpoint.is_file()
    assert cv.cv_mean["cv_n_folds"] == N_FOLDS


def test_fold_entries_carry_their_keys_and_real_checkpoints(tmp_path):
    """Each fold entry: index, relative directory, kind, the three windows and
    the metrics. Every fold unit's checkpoint must exist, because run_cv
    deserializes exactly that path."""
    model = _library(tmp_path, "ckpt")

    cv = model.train_cv(train_periods=TRAIN_PERIODS)

    record = _read_record(tmp_path / "ckpt")
    assert [entry["fold"] for entry in record["folds"]] == list(range(N_FOLDS))
    for entry, fold in zip(record["folds"], cv.folds):
        assert set(entry) == FOLD_ENTRY_KEYS, sorted(entry)
        assert entry["directory"] == f"fold_{entry['fold']}"
        assert entry["kind"] == "model"
        assert any(key.startswith("test_") for key in entry["metrics"])
        assert fold.checkpoint.name == f"StubLibraryHead_cv_fold_{entry['fold']}.joblib"
        assert fold.checkpoint.is_file(), fold.checkpoint


def test_a_relative_save_dir_still_reads_from_another_directory(tmp_path, monkeypatch):
    """Code review WR-03: a run trained with a relative `model_save_dir` is
    read later from whatever directory `run_cv` runs in. Paths in the record
    are relative to the unit, so opening the unit by an absolute path from
    elsewhere finds every fold checkpoint inside it, never via the cwd."""
    monkeypatch.chdir(tmp_path)
    kwargs = _common(tmp_path, "unused")
    kwargs["model_save_dir"] = "ckpt"
    model = StubLibraryHead(ModelConfig(**kwargs))
    model.collect()

    cv = model.train_cv(train_periods=TRAIN_PERIODS)

    unit = (tmp_path / "ckpt" / cv.path.name).absolute()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    reopened = TrainedRun.open(unit)
    assert len(reopened.folds) == N_FOLDS
    for fold in reopened.folds:
        assert fold.checkpoint.is_absolute() and fold.checkpoint.is_file()
        assert fold.checkpoint.parent.parent == unit


def test_a_copied_trial_directory_opens_at_its_new_location(tmp_path):
    model = _library(tmp_path, "ckpt")
    cv = model.train_cv(train_periods=TRAIN_PERIODS)

    moved = tmp_path / "moved" / cv.path.name
    shutil.copytree(cv.path, moved)
    shutil.rmtree(cv.path)
    copied = TrainedRun.open(moved)

    assert [_fold_tuple(f) for f in copied.folds] == [_fold_tuple(f) for f in cv.folds]
    assert [f.metrics for f in copied.folds] == [f.metrics for f in cv.folds]
    for fold in copied.folds:
        assert fold.checkpoint.is_file() and fold.checkpoint.parent.parent == moved


def test_empty_fold_list_still_writes_a_record(tmp_path, monkeypatch):
    """With no folds the record is still written, with `folds: []`, so
    run_cv can say "this CV run produced no folds" instead of "not a CV
    project directory"."""
    monkeypatch.setattr("quantlab.base.model.walk_forward_folds", lambda timestamps, train_periods, **_: ())
    model = _library(tmp_path, "ckpt")

    cv = model.train_cv(train_periods=TRAIN_PERIODS)

    assert cv.folds == () and cv.cv_mean == {}
    record = _read_record(tmp_path / "ckpt")
    assert record == {"format_version": 1, "kind": "walk_forward", "folds": [], "cv_mean": {}}


def test_record_is_strict_json_with_null_for_non_finite_metrics(tmp_path):
    """A NaN numpy metric must reach the files as `null`: `json.dump`'s
    default writes a bare `NaN` token, which strict parsers reject. The fold
    units read back carry None."""
    model = _library(tmp_path, "ckpt", cls=NaNMetricLibraryHead)

    cv = model.train_cv(train_periods=TRAIN_PERIODS)

    record = _read_record(tmp_path / "ckpt")
    assert len(record["folds"]) == N_FOLDS
    for entry, fold in zip(record["folds"], cv.folds):
        assert entry["metrics"]["test_nan_metric"] is None
        assert entry["metrics"]["test_finite_metric"] == 1.5
        assert fold.metrics["test_nan_metric"] is None
    assert cv.cv_mean["cv_mean_test_finite_metric"] == 1.5
    # A metric undefined in every fold still has its mean, recorded as null.
    assert record["cv_mean"]["cv_mean_test_nan_metric"] is None
    assert cv.cv_mean["cv_mean_test_nan_metric"] is None
