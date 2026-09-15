"""The `cv_folds.json` manifest `BaseModel.train_cv` persists (phase 03.7, D-30 / D-36).

`train_cv` writes `{model_save_dir}/{project_name}/cv_folds.json` holding
`{"format_version": 1, "folds": [...]}`, where `folds` is the JSON form of
exactly the list `train_cv` returns.

Why the manifest exists: `run_cv` (plan 03.7-10) backtests every fold's
out-of-sample segment, and it can only replay the folds a training run actually
used if that geometry is on disk and equal to what training returned. Why it is
versioned (D-36): CV backtests of OLD training runs read it, so the on-disk
shape is a persisted format; `format_version` is the migration seam and
`run_cv` rejects a version it does not know.

What turns this file red:

- the manifest is missing, or its `folds` differ from the returned list, on the
  sequential branch, the parallel branch, or a DL head;
- a fold entry loses one of D-30's keys, or points at a checkpoint that is not
  on disk;
- manifest keys leak into the returned fold dicts (the return value is D-30's
  "unchanged" contract);
- an empty fold list writes no manifest (run_cv could then not tell "no folds"
  from "not a CV project");
- a non-finite metric is dumped as a bare `NaN` token, which strict JSON
  parsers reject (03.7-RESEARCH.md Pitfall 10).

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

from quantlab.base.config import DLConfig, MLConfig
from quantlab.base.model import BaseModel, DLModel, MLModel
from quantlab.utils.jsonable import to_jsonable

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

FOLD_KEYS = {"fold", "train_start", "train_end", "test_start", "test_end"}
D30_KEYS = FOLD_KEYS | {"experiment_name", "checkpoint"}


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    """`_init_wandb` calls `wandb.init` unconditionally; this is the
    documented bypass."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


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
    """A numpy head: predicts the first factor for every label."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


class NaNMetricMLHead(StubMLHead):
    """Every fold reports one non-finite metric, as a numpy scalar, beside a
    finite one -- the shape vectorbt-style and panel metrics really take."""

    def _compute_metrics(self, y, pred):
        return {"nan_metric": np.float64("nan"), "finite_metric": np.float64(1.5)}


class TinyDLHead(DLModel):
    """The smallest trainable torch head: one `nn.Linear` on the last axis."""

    def _init_model(self, num_symbols, num_features, num_labels, hyperparameters):
        return nn.Linear(num_features, num_labels)

    def _init_optim(self, model):
        return torch.optim.SGD(model.parameters(), lr=1e-3)

    def _preprocess(self, data):
        return torch.nan_to_num(data, nan=0.0)

    def _train_one_batch(self, epoch, x, y):
        self.optim.zero_grad()
        loss = nn.functional.mse_loss(self.model(x), y)
        loss.backward()
        self.optim.step()
        return loss

    def _val_one_batch(self, epoch, x, y):
        return nn.functional.mse_loss(self.model(x), y)

    def _test_one_batch(self, epoch, x, y):
        return nn.functional.mse_loss(self.model(x), y)


def _common(tmp_path: Path, save_dir: str) -> dict:
    return dict(
        factors=[FakePanel(["f_a", "f_b"], seed=1)],
        labels=[FakePanel(["ret"], seed=2)],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
    )


def _ml(tmp_path: Path, save_dir: str, cls=StubMLHead) -> MLModel:
    model = cls(MLConfig(**_common(tmp_path, save_dir)))
    model.collect()
    return model


def _dl(tmp_path: Path, save_dir: str) -> DLModel:
    model = TinyDLHead(
        DLConfig(
            **_common(tmp_path, save_dir),
            epochs=1,
            batch_size=64,
            num_workers=0,
        )
    )
    model.collect()
    return model


def _project_dir(save_root: Path) -> Path:
    assert save_root.is_dir(), f"train_cv created no save root at {save_root}"
    projects = [p for p in save_root.iterdir() if p.is_dir()]
    assert len(projects) == 1, f"expected one CV project dir, got {projects}"
    return projects[0]


def _read_manifest(save_root: Path) -> dict:
    """Parse the manifest STRICTLY: a bare NaN/Infinity token raises."""
    path = _project_dir(save_root) / "cv_folds.json"
    assert path.is_file(), f"train_cv wrote no manifest at {path}"

    def _reject(token):
        raise ValueError(f"non-standard JSON token {token!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def test_sequential_ml_manifest_equals_returned_folds(tmp_path):
    """The sequential branch: `folds` is the JSON form of the returned list,
    and the wrapper carries `format_version` 1 and nothing else."""
    model = _ml(tmp_path, "ckpt")

    results = model.train_cv(train_periods=TRAIN_PERIODS)

    manifest = _read_manifest(tmp_path / "ckpt")
    assert set(manifest) == {"format_version", "folds"}
    assert manifest["format_version"] == 1
    assert manifest["folds"] == to_jsonable(results)
    assert len(results) == N_FOLDS


def test_parallel_ml_manifest_equals_returned_folds(tmp_path):
    """The parallel branch writes the same manifest contract: the write must
    sit after BOTH branches, not inside one of them."""
    model = _ml(tmp_path, "ckpt")

    results = model.train_cv(train_periods=TRAIN_PERIODS, parallel=True, njobs=2)

    manifest = _read_manifest(tmp_path / "ckpt")
    assert manifest["format_version"] == 1
    assert manifest["folds"] == to_jsonable(results)
    assert len(results) == N_FOLDS


def test_dl_manifest_equals_returned_folds(tmp_path):
    """A DL head's `_fit` returns no metrics, so each manifest entry carries
    only the fold's dates, its run name and its checkpoint."""
    model = _dl(tmp_path, "ckpt")

    results = model.train_cv(train_periods=TRAIN_PERIODS)

    manifest = _read_manifest(tmp_path / "ckpt")
    assert manifest["format_version"] == 1
    assert manifest["folds"] == to_jsonable(results)
    assert len(manifest["folds"]) == N_FOLDS
    for entry in manifest["folds"]:
        assert set(entry) == D30_KEYS
        assert entry["checkpoint"].endswith(".pth")


def test_manifest_fold_entries_carry_the_d30_keys_and_real_checkpoints(tmp_path):
    """D-30's per-fold entry: fold, the four dates, experiment_name and
    checkpoint, plus the test metrics an ML head produces. Every checkpoint
    named must exist, because run_cv deserializes exactly that path."""
    model = _ml(tmp_path, "ckpt")

    model.train_cv(train_periods=TRAIN_PERIODS)

    manifest = _read_manifest(tmp_path / "ckpt")
    assert [entry["fold"] for entry in manifest["folds"]] == list(range(N_FOLDS))
    for entry in manifest["folds"]:
        assert D30_KEYS <= set(entry), sorted(entry)
        assert any(key.startswith("test_") and key not in FOLD_KEYS for key in entry)
        assert entry["experiment_name"] == f"StubMLHead_cv_fold_{entry['fold']}"
        assert Path(entry["checkpoint"]).is_file(), entry["checkpoint"]


def test_manifest_checkpoints_are_absolute_with_a_relative_save_dir(tmp_path, monkeypatch):
    """Code review WR-03: a relative `model_save_dir` still yields absolute checkpoints.

    The manifest is read later, by `run_cv`, from whatever directory that
    process runs in. A relative entry written from the training cwd resolves
    against the reader's cwd: from another directory it is missing, or worse, it
    names another run's checkpoint. Training runs from `tmp_path` with
    `model_save_dir="ckpt"`, and the entries are checked from a different
    directory. The old code wrote `ckpt/...` verbatim and goes red here.
    """
    monkeypatch.chdir(tmp_path)
    kwargs = _common(tmp_path, "unused")
    kwargs["model_save_dir"] = "ckpt"
    model = StubMLHead(MLConfig(**kwargs))
    model.collect()

    results = model.train_cv(train_periods=TRAIN_PERIODS)

    manifest = _read_manifest(tmp_path / "ckpt")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert len(manifest["folds"]) == len(results) == N_FOLDS
    for result, entry in zip(results, manifest["folds"]):
        assert Path(entry["checkpoint"]).is_absolute(), entry["checkpoint"]
        assert Path(entry["checkpoint"]).is_file(), entry["checkpoint"]
        assert entry["checkpoint"] == result["checkpoint"]


def test_return_value_carries_no_manifest_keys(tmp_path):
    """D-30: the returned value is unchanged by the manifest write. No fold
    dict gains the wrapper's keys, and the manifest holds exactly as many
    folds as were returned."""
    model = _ml(tmp_path, "ckpt")

    results = model.train_cv(train_periods=TRAIN_PERIODS)

    for result in results:
        assert "format_version" not in result
        assert "folds" not in result
    manifest = _read_manifest(tmp_path / "ckpt")
    assert len(manifest["folds"]) == len(results)


def test_empty_fold_list_still_writes_a_manifest(tmp_path, monkeypatch):
    """With no folds the manifest is still written, with `folds: []`, so
    run_cv can say "this CV run produced no folds" instead of "not a CV
    project directory"."""
    monkeypatch.setattr(
        BaseModel,
        "_cv_folds",
        staticmethod(lambda timestamps, train_periods, gap_periods: []),
    )
    model = _ml(tmp_path, "ckpt")

    results = model.train_cv(train_periods=TRAIN_PERIODS)

    assert results == []
    manifest = _read_manifest(tmp_path / "ckpt")
    assert manifest == {"format_version": 1, "folds": []}


def test_manifest_is_strict_json_with_null_for_non_finite_metrics(tmp_path):
    """A NaN numpy metric must reach the file as `null`: `json.dump`'s
    default writes a bare `NaN` token, which strict parsers reject. The
    returned list still carries the NaN -- only the file is converted."""
    model = _ml(tmp_path, "ckpt", cls=NaNMetricMLHead)

    results = model.train_cv(train_periods=TRAIN_PERIODS)

    manifest = _read_manifest(tmp_path / "ckpt")
    assert len(manifest["folds"]) == N_FOLDS
    for entry, result in zip(manifest["folds"], results):
        assert entry["test_nan_metric"] is None
        assert entry["test_finite_metric"] == 1.5
        assert np.isnan(result["test_nan_metric"])
