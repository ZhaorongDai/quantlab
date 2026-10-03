"""Model training and CV track through the config's tracker (ticket #101, ADR 0015).

Every test puts a tracker in the model config and calls a public entry point
(``train``, ``train_cv``, ``load_model_from_config``); what it asserts is what
an outsider sees: the runs the tracker opened, how they ended and what was
written to ``config.json``.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from quantlab.base.config import ModelConfig
from quantlab.base.tracking import NullTracker
from quantlab.tracking.wandb import WandbTracker
from quantlab.utils.module import load_model_from_config
from tests.test_backtest_contracts import REPO_ROOT, _python_files
from tests.test_model_metrics_file import StubLibraryHead, _model
from tests.torch_heads import OneBarHead
from tests.tracking_fixtures import RecordingTracker


class FailingFitHead(StubLibraryHead):
    def _fit_model(self, train_rows, val_rows):
        self._run.log({"loss": 1.0}, step=0)
        raise RuntimeError("the library crashed")


def test_the_default_config_tracks_nowhere():
    assert ModelConfig.__dataclass_fields__["tracker"].default == NullTracker()


@pytest.mark.parametrize("cls", [StubLibraryHead, OneBarHead], ids=["library", "torch"])
def test_a_training_is_one_finished_run_in_the_class_project_grouped_by_trial(tmp_path, cls):
    tracker = RecordingTracker()

    checkpoint = _model(tmp_path, cls=cls, tracker=tracker).train()

    (run,) = tracker.runs
    trial = checkpoint.parent.name
    assert (run.project, run.group, run.name) == (cls.__name__, trial, f"{cls.__name__}_total")
    assert run.config["tracker"] == tracker.get_config()
    assert run.finished and not run.failed
    assert "test_ic" in run.summary


def test_the_trackers_own_project_replaces_the_class_name(tmp_path):
    tracker = RecordingTracker(project="momentum_research")

    _model(tmp_path, tracker=tracker).train()

    assert [run.project for run in tracker.runs] == ["momentum_research"]


def test_a_run_is_finished_as_failed_when_fitting_raises(tmp_path):
    tracker = RecordingTracker()
    model = _model(tmp_path, cls=FailingFitHead, tracker=tracker)

    with pytest.raises(RuntimeError, match="the library crashed"):
        model.train()

    (run,) = tracker.runs
    assert run.steps == [(0, {"loss": 1.0})]
    assert run.finished and run.failed


def test_every_run_of_a_train_cv_call_is_grouped_by_its_trial_directory(tmp_path):
    tracker = RecordingTracker()
    model = _model(tmp_path, tracker=tracker)

    results = model.train_cv(train_periods=20)

    (trial,) = [p.name for p in (tmp_path / "ckpt").iterdir()]
    assert {(run.project, run.group) for run in tracker.runs} == {("StubLibraryHead", trial)}
    assert [run.name for run in tracker.runs] == [
        r["experiment_name"] for r in results
    ] + ["StubLibraryHead_cv_summary"]
    summary = tracker.runs[-1].summary
    assert summary["cv_n_folds"] == len(results)
    assert summary["cv_mean_test_ic"] == pytest.approx(
        sum(r["test_ic"] for r in results) / len(results)
    )
    assert all(run.finished and not run.failed for run in tracker.runs)


def test_two_train_calls_are_two_groups_in_one_project(tmp_path):
    tracker = RecordingTracker()

    _model(tmp_path, tracker=tracker).train()
    _model(tmp_path, tracker=tracker).train()

    first, second = tracker.runs
    assert first.project == second.project == "StubLibraryHead"
    assert first.group != second.group


def test_the_tracker_round_trips_through_the_checkpoints_config_json(tmp_path):
    tracker = WandbTracker(project="research", entity="team", mode="disabled")

    checkpoint = _model(tmp_path, tracker=tracker).train()

    saved = json.loads((checkpoint.parent / "config.json").read_text())
    assert saved["tracker"] == {
        "project": "research",
        "entity": "team",
        "mode": "disabled",
        "name": "quantlab.tracking.wandb.WandbTracker",
    }
    saved["factors"] = []
    saved["labels"] = []
    saved["name"] = "tests.test_model_metrics_file.StubLibraryHead"
    assert load_model_from_config(saved).config.tracker == tracker


def test_training_with_the_default_config_imports_no_tracking_library(tmp_path):
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from tests.test_model_metrics_file import _model\n"
        f"_model(Path({str(tmp_path)!r})).train()\n"
        "print(sorted(name for name in ('wandb', 'mlflow') if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


def _guards_a_run(node: ast.AST) -> bool:
    """``x._run is None`` / ``is not None`` and the old ``_wandb_recorder`` checks."""
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr in ("_run", "_wandb_recorder")
        and any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops)
    )


def test_the_model_layer_never_checks_whether_a_run_is_open():
    files = [REPO_ROOT / "quantlab/base/model.py", *_python_files(REPO_ROOT / "quantlab/model")]
    # Positive control: the shipped heads are seen.
    assert any(path.name == "xgb.py" for path in files)
    offenders = sorted(
        f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
        for path in files
        for node in ast.walk(ast.parse(path.read_text()))
        if _guards_a_run(node)
    )
    assert offenders == []
