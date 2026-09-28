"""Training into a caller-given run directory (issue #57).

`BaseModel._train_into` trains, evaluates and saves one model into a run
directory the caller chooses, under W&B names the caller chooses, and creates
no `{class}_trial_{timestamp}` directory. `train()` and every `train_cv()`
fold go through it; an ensemble trains each member into `member_{k}/` with
it. It reseeds from `config.random_seed` first, so a model trained after its
siblings sees the same random state as one trained alone.

What turns this file red:

- the checkpoint, `config.json`, `metrics.json`, `ic_series.csv` or
  `test_predictions.zarr` is missing from the given directory, or something
  is written under `model_save_dir`;
- the W&B run is not opened under the given project and experiment names;
- `write_metrics=False` still writes `metrics.json`, or the returned metrics
  differ from the file;
- the random state left by earlier work changes what a fit produces.

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from quantlab.base.model import BaseModel
from tests.test_model_metrics_file import StubLibraryHead, _model
from tests.torch_heads import OneBarHead


class FakeRecorder:
    """Records the names a W&B run was opened under."""

    def __init__(self, project: str, name: str):
        self.project = project
        self.name = name
        self.summary: dict = {}

    def log(self, data, step=None):
        pass

    def finish(self):
        pass


@pytest.fixture
def recorders(monkeypatch) -> list[FakeRecorder]:
    created: list[FakeRecorder] = []

    def fake_init_wandb(self, project_name, experiment_name):
        recorder = FakeRecorder(project_name, experiment_name)
        created.append(recorder)
        self._wandb_recorder = recorder

    monkeypatch.setattr(BaseModel, "_init_wandb", fake_init_wandb)
    return created


@pytest.mark.parametrize("cls", [StubLibraryHead, OneBarHead], ids=["library", "torch"])
def test_train_into_writes_the_usual_files_into_the_given_directory(
    tmp_path, recorders, cls
):
    model = _model(tmp_path, cls=cls)
    run_dir = tmp_path / "ensemble" / "member_0"

    checkpoint, metrics = model._train_into(
        run_dir, project_name="my_project", experiment_name="Head_member_0"
    )

    assert checkpoint == (run_dir / f"Head_member_0{model.checkpoint_suffix}").absolute()
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(
        [
            checkpoint.name,
            "config.json",
            "ic_series.csv",
            "metrics.json",
            "test_predictions.zarr",
        ]
    )
    assert not (tmp_path / "ckpt").exists()
    ((project, name),) = [(r.project, r.name) for r in recorders]
    assert (project, name) == ("my_project", "Head_member_0")
    written = json.loads((run_dir / "metrics.json").read_text())
    assert set(written) == set(metrics)


def test_train_into_without_metrics_file_returns_the_metrics(tmp_path, recorders):
    model = _model(tmp_path)
    run_dir = tmp_path / "fold_0"

    checkpoint, metrics = model._train_into(
        run_dir, project_name="p", experiment_name="e", write_metrics=False
    )

    assert checkpoint.is_file()
    assert not (run_dir / "metrics.json").exists()
    assert (run_dir / "ic_series.csv").is_file()
    assert "test_ic" in metrics


class RandomWeightHead(StubLibraryHead):
    """Scales the first factor by a weight drawn from numpy's global generator."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels, "weight": float(np.random.rand())}


def test_train_into_reseeds_so_earlier_random_draws_do_not_leak(tmp_path, recorders):
    first = _model(tmp_path, cls=RandomWeightHead, random_seed=7)
    first._train_into(tmp_path / "a", project_name="p", experiment_name="a")

    np.random.rand(100)
    sibling = _model(tmp_path, cls=RandomWeightHead, random_seed=8)
    sibling._train_into(tmp_path / "b", project_name="p", experiment_name="b")
    second = _model(tmp_path, cls=RandomWeightHead, random_seed=7)
    np.random.rand(100)
    second._train_into(tmp_path / "c", project_name="p", experiment_name="c")

    assert first.model["weight"] == second.model["weight"]
    assert sibling.model["weight"] != first.model["weight"]


def test_train_still_creates_one_trial_directory(tmp_path, recorders):
    checkpoint = _model(tmp_path).train()

    (trial,) = list((tmp_path / "ckpt").iterdir())
    assert trial.name.startswith("StubLibraryHead_trial_")
    assert checkpoint.parent == trial / "StubLibraryHead_total"
    ((project, name),) = [(r.project, r.name) for r in recorders]
    assert (project, name) == (trial.name, "StubLibraryHead_total")
