"""Walk-forward training, called directly with a stub trainable (issue #144).

`train_walk_forward` runs a walk-forward cross-validation of anything
satisfying `WalkForwardTrainable`; a model and an ensemble are its two
adapters. The stub here trains nothing: it records the folds and directories
it is asked to train into and writes a minimal model trained run, so fold and
directory layout run in milliseconds.

What turns this file red:

- folds are not trained in order into `fold_{i}/` of one
  `{class}_trial_{timestamp}/` under `model_save_dir`;
- the unit's `purge_bars` does not reach the fold layout;
- the hyperparameter check runs after a directory exists, or a refused
  setting or an empty date range leaves a directory behind;
- `cv_mean` is not the mean of the folds' finite metrics plus `cv_n_folds`,
  or the `{class}_cv_summary` run is not opened, in the trial's group,
  through the unit's tracker and project with its config;
- the walk-forward `run.json` misses the folds, the means or the unit's
  provenance;
- `BaseModel` or `BaseEnsemble` stops satisfying the protocol.

Every expected value is computed by hand in the comments. That a model and
an ensemble keep their own dates is the adapters' part, tested in
`test_model_cv.py` and `test_seed_ensemble_cv.py`.
"""

import numpy as np
import pytest

from quantlab.base.model import BaseModel
from quantlab.model.ensemble import BaseEnsemble
from quantlab.runs.trained_run import TrainedRun, write_model_run
from quantlab.utils.walk_forward_training import (
    WalkForwardTrainable,
    cv_mean_metrics,
    train_walk_forward,
)
from tests.tracking_fixtures import RecordingTracker

#: 20 daily bars; train_periods 10 and test_periods 5 give 2 folds, fold i
#: testing bars 10 + 5i .. 14 + 5i.
BARS = np.arange("2024-01-01", "2024-01-21", dtype="datetime64[D]")


class StubTrainable:
    """Trains nothing; records its calls and writes a model run per fold."""

    class_name = "Stub"
    tracking_project = "StubProject"

    def __init__(self, root, *, purge_bars=0, refuse=False):
        self.model_save_dir = root
        self.bars = BARS
        self.purge_bars = purge_bars
        self.tracker = RecordingTracker()
        self.refuse = refuse
        self.calls: list = []

    def walk_forward_bars(self):
        return self.bars

    def check_hyperparameters(self):
        self.calls.append(("check", self.model_save_dir.exists()))
        if self.refuse:
            raise ValueError("bad hyperparameter")

    def train_fold(self, fold, run_dir, group):
        self.calls.append(("fold", fold.index, run_dir, group))
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "stub.joblib"
        checkpoint.write_bytes(b"")
        write_model_run(
            run_dir,
            checkpoint=checkpoint,
            train_window=fold.train_window,
            fitted_train_window=fold.fitted_train_window,
            test_window=fold.test_window,
            trained_on={"factor_names": [], "label_names": [], "symbols": []},
            # Fold 0: test_ic 0.1; fold 1: 0.3 and an undefined val_ic.
            metrics={"test_ic": 0.1 + 0.2 * fold.index, "val_ic": None if fold.index else 0.5},
        )

    def get_config(self):
        return {"name": "Stub"}

    def provenance(self):
        return {"data_fingerprint": {"stub": {"digest": "abc"}}, "code": None}


def test_folds_train_in_order_into_fold_directories_of_one_trial(tmp_path):
    stub = StubTrainable(tmp_path / "models")

    run = train_walk_forward(stub, train_periods=10, test_periods=5)

    folds = [call for call in stub.calls if call[0] == "fold"]
    assert [call[1] for call in folds] == [0, 1]
    (trial,) = (tmp_path / "models").iterdir()
    assert trial.name.startswith("Stub_trial_")
    assert [call[2] for call in folds] == [trial / "fold_0", trial / "fold_1"]
    assert {call[3] for call in folds} == {trial.name}
    assert run.path == trial and [f.index for f in run.folds] == [0, 1]
    assert [f.test_window for f in run.folds] == [
        ("2024-01-11", "2024-01-15"),
        ("2024-01-16", "2024-01-20"),
    ]


def test_the_units_purge_length_shortens_every_fitted_window(tmp_path):
    stub = StubTrainable(tmp_path / "models", purge_bars=3)

    run = train_walk_forward(stub, train_periods=10, test_periods=5)

    # Fold 0 trains on bars 0-9 and fits 0-6; fold 1 on 5-14, fits 5-11.
    assert [f.fitted_train_window for f in run.folds] == [
        ("2024-01-01", "2024-01-07"),
        ("2024-01-06", "2024-01-12"),
    ]


def test_hyperparameters_are_checked_before_any_directory(tmp_path):
    stub = StubTrainable(tmp_path / "models", refuse=True)

    with pytest.raises(ValueError, match="bad hyperparameter"):
        train_walk_forward(stub, train_periods=10, test_periods=5)

    assert stub.calls == [("check", False)]
    assert not (tmp_path / "models").exists()


def test_refused_settings_name_the_unit_and_leave_nothing(tmp_path):
    stub = StubTrainable(tmp_path / "models")

    with pytest.raises(ValueError, match="Stub: train_cv: .*train_periods=4"):
        train_walk_forward(stub, train_periods=4)

    assert not (tmp_path / "models").exists()


def test_the_fold_means_go_to_the_summary_run_and_run_json(tmp_path):
    stub = StubTrainable(tmp_path / "models")

    run = train_walk_forward(stub, train_periods=10, test_periods=5)

    # test_ic: (0.1 + 0.3) / 2; val_ic: only fold 0's 0.5 is defined.
    expected = {"cv_mean_test_ic": 0.2, "cv_mean_val_ic": 0.5, "cv_n_folds": 2}
    assert run.cv_mean == pytest.approx(expected)
    reread = TrainedRun.open(run.path)
    assert reread.kind == "walk_forward"
    assert reread.cv_mean == pytest.approx(expected)
    assert reread.data_fingerprint == {"stub": {"digest": "abc"}}
    (summary,) = stub.tracker.runs
    assert (summary.project, summary.group, summary.name) == (
        "StubProject",
        run.path.name,
        "Stub_cv_summary",
    )
    assert summary.config == {"name": "Stub"}
    assert summary.summary == pytest.approx(expected)
    assert summary.finished and not summary.failed


def test_cv_mean_metrics_skips_undefined_values_and_other_keys():
    means = cv_mean_metrics(
        [
            {"test_ic": 0.1, "val_ic": None, "train_loss": float("nan"), "seed": 3},
            {"test_ic": 0.3, "val_ic": None, "train_loss": 2.0, "flag": True},
        ]
    )

    assert means["cv_mean_test_ic"] == pytest.approx(0.2)
    assert np.isnan(means["cv_mean_val_ic"])
    assert means["cv_mean_train_loss"] == 2.0
    assert means["cv_n_folds"] == 2
    assert set(means) == {
        "cv_mean_test_ic", "cv_mean_val_ic", "cv_mean_train_loss", "cv_n_folds"
    }


def test_no_metric_in_any_fold_gives_no_means():
    assert cv_mean_metrics([{}, {"seed": 1}]) == {}


def test_no_bar_in_the_date_range_names_the_unit_and_leaves_nothing(tmp_path):
    stub = StubTrainable(tmp_path / "models")
    stub.bars = BARS[:0]

    with pytest.raises(ValueError, match="Stub: train_cv: No data found"):
        train_walk_forward(stub, train_periods=10, test_periods=5)

    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("adapter", [BaseModel, BaseEnsemble])
def test_a_model_and_an_ensemble_are_the_two_adapters(adapter):
    members = [name for name in dir(WalkForwardTrainable) if not name.startswith("_")]
    assert all(hasattr(adapter, name) for name in members)
