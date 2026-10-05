"""A trained model describes itself in run.json, read through TrainedRun (#122).

``train()`` writes a trial directory that is one trained unit: the checkpoint,
``config.json`` (what rebuilding the model needs), the evaluation files and,
last, ``run.json``. ``TrainedRun.open`` reads it back from the directory, from
``run.json`` or from the checkpoint. What is locked here, through a real
``train()`` of a small library model:

- the three ways of opening agree, and give the windows, metrics, training
  record and files ``train()`` produced; the fitted training window is the
  configured one less the labels' purge;
- a copied unit opens at its new location;
- a model reports the fitted window after ``train()`` or ``load()`` as
  ``fitted_train_bounds``, and a loaded model's windows are its record's;
- an unknown version, a missing ``run.json``, and a file that is not the
  unit's checkpoint are refused.

Since #123 the same holds for the ``"ensemble"`` and ``"walk_forward"``
kinds, through a real ``train`` of a seed ensemble and a real ``train_cv`` of
a model and of a seed ensemble: every unit and every child (member, fold)
opens on its own with the windows and metrics training returned, a copied
trial directory opens at its new location, and no ``cv_folds.json``,
``ensemble.json`` or ``metrics.json`` is written.

Since #132 the module lives in the run layer (``quantlab.runs``), on the
run-directory mechanism: every kind opens through ``open_run`` as the same
``TrainedRun``, without loading the backtest, model, factor or label layers;
``run.json`` carries the shared header (``format_version``, ``kind``,
``written_at``), so a unit of the previous format version is refused; and a
library model's ``resolved_hyperparameters`` is recorded in the unit's
``run.json``, never in ``config.json``.

The unit's file names appear here only in the directory-listing lock of its layout,
in opening a unit from its ``run.json`` (one of the paths ``open`` takes) and in
rewriting or removing a record to check a refusal; results are read through
``TrainedRun``.
"""

import dataclasses
import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import xarray as xr

from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.runs.directory import FORMAT_VERSION, open_run
from quantlab.runs.trained_run import TrainedRun
from tests.backtest_fixtures import (
    FirstFeatureHead,
    SeededHead,
    make_model,
    write_price_store,
)

N_BARS = 40


def _day(ts) -> str:
    """The ``YYYY-MM-DD`` spelling of a bar."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@pytest.fixture
def setup(tmp_path):
    """A price store and the model dates: train on bars 0..29, test on 30..39."""
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    dates = dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[-1]),
        train_start=_day(bars[0]),
        train_end=_day(bars[29]),
        test_start=_day(bars[30]),
        test_end=_day(bars[-1]),
    )
    return tmp_path, dataset_config, bars, dates


def _trained(setup, name="train"):
    tmp_path, dataset_config, bars, dates = setup
    model = make_model(tmp_path / name, dataset_config, **dates)
    checkpoint = model.collect().train()
    return model, checkpoint


def test_a_trained_model_opens_from_its_directory_run_json_and_checkpoint(setup):
    _, _, bars, dates = setup
    model, checkpoint = _trained(setup)

    run = TrainedRun.open(checkpoint.parent)

    assert TrainedRun.open(checkpoint) == run
    assert TrainedRun.open(checkpoint.parent / "run.json") == run
    assert run.kind == "model"
    assert run.path == checkpoint.parent
    assert run.checkpoint == checkpoint
    assert run.train_window == (dates["train_start"], dates["train_end"])
    assert run.test_window == (dates["test_start"], dates["test_end"])
    # The label reads 2 bars ahead, so the fit stops 2 bars before train_end.
    lookahead = max(label.lookahead_bars() for label in model.labels)
    assert lookahead == 2
    assert pd.Timestamp(run.fitted_train_window[1]) == pd.Timestamp(bars[29 - lookahead])
    assert run.fitted_train_window[0] == dates["train_start"]
    assert {"test_ic", "train_ic"} <= set(run.metrics)
    assert run.trained_on["factor_names"] == [str(n) for n in model.get_factor_names()]
    assert run.trained_on["label_names"] == [str(n) for n in model.get_label_names()]
    assert run.ic_series is not None and run.ic_series.is_file()
    assert run.test_predictions is not None and run.test_predictions.is_dir()
    assert run.config == json.loads(json.dumps(model.get_config()))


def test_train_writes_the_trial_directory_as_the_unit(setup):
    model, checkpoint = _trained(setup)

    assert checkpoint.parent.parent.name == "models"
    assert checkpoint.parent.name.startswith(f"{model.class_name}_trial_")
    assert sorted(p.name for p in checkpoint.parent.iterdir()) == sorted(
        [checkpoint.name, "config.json", "ic_series.csv", "run.json", "test_predictions.zarr"]
    )
    assert "trained_on" not in TrainedRun.open(checkpoint).config


def test_a_copied_unit_opens_at_its_new_location(setup):
    tmp_path = setup[0]
    _, checkpoint = _trained(setup)
    original = TrainedRun.open(checkpoint)

    moved = tmp_path / "elsewhere"
    shutil.copytree(checkpoint.parent, moved)
    run = TrainedRun.open(moved)

    assert run.path == moved
    assert run.checkpoint == moved / checkpoint.name
    assert (run.train_window, run.fitted_train_window, run.test_window) == (
        original.train_window, original.fitted_train_window, original.test_window
    )
    assert run.metrics == original.metrics


def test_a_model_reports_the_fitted_window_after_train_and_after_load(setup):
    tmp_path, dataset_config, bars, dates = setup
    model, checkpoint = _trained(setup)
    run = TrainedRun.open(checkpoint)
    assert model.fitted_train_bounds == run.fitted_train_window

    # A model configured with other dates takes the checkpoint's on load.
    other = make_model(
        tmp_path / "other", dataset_config, **{**dates, "train_end": _day(bars[20])}
    )
    with pytest.raises(RuntimeError, match="train\\(\\) or load\\(\\)"):
        other.fitted_train_bounds
    other.load(checkpoint)

    assert other.fitted_train_bounds == run.fitted_train_window
    assert other.train_bounds == run.train_window
    assert other.test_bounds == run.test_window


def test_an_unknown_format_version_is_refused(setup):
    _, checkpoint = _trained(setup)
    path = checkpoint.parent / "run.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "format_version": 99}))

    with pytest.raises(ValueError, match="format_version 99.*retrain"):
        TrainedRun.open(checkpoint)


def test_a_directory_without_run_json_is_refused(setup):
    _, checkpoint = _trained(setup)
    (checkpoint.parent / "run.json").unlink()

    with pytest.raises(ValueError, match="no run.json.*retrain"):
        TrainedRun.open(checkpoint.parent)
    with pytest.raises(ValueError, match="no run.json.*retrain"):
        make_model_like = _trained(setup, name="again")[0]
        make_model_like.load(checkpoint)


def test_a_file_that_is_not_the_units_checkpoint_is_refused(setup):
    _, checkpoint = _trained(setup)

    with pytest.raises(ValueError, match="not the checkpoint"):
        TrainedRun.open(checkpoint.parent / "config.json")


def test_a_missing_path_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainedRun.open(tmp_path / "nowhere")


# ---------------------------------------------------------------- ensemble and walk-forward units

OLD_RECORDS = ("cv_folds.json", "ensemble.json", "metrics.json")


def _assert_opens_alone(child: TrainedRun) -> None:
    """A child unit read through its parent equals the unit opened on its own."""
    alone = TrainedRun.open(child.path)
    assert alone == dataclasses.replace(child, index=None, seed=None)
    if child.checkpoint is not None:
        assert TrainedRun.open(child.checkpoint) == alone


def _assert_no_old_records(root) -> None:
    for name in OLD_RECORDS:
        assert list(root.rglob(name)) == [], name


def _ensemble(setup, name):
    tmp_path, dataset_config, _, dates = setup
    model = make_model(tmp_path / name, dataset_config, head=SeededHead, **dates)
    return SeedEnsemble(model, [0, 1])


def test_a_seed_ensemble_unit_round_trips(setup):
    ensemble = _ensemble(setup, "ensemble")
    checkpoint = ensemble.collect().train()

    run = TrainedRun.open(checkpoint)

    assert TrainedRun.open(checkpoint.parent) == run
    assert run.kind == "ensemble" and run.checkpoint == checkpoint
    assert run.train_window == ensemble.train_bounds
    assert run.fitted_train_window == ensemble.fitted_train_bounds
    assert run.test_window == ensemble.test_bounds
    assert {"test_ic", "test_member_correlation"} <= set(run.metrics)
    assert run.ic_series.is_file() and run.test_predictions.is_dir()
    assert [member.seed for member in run.members] == [0, 1]
    for member, model in zip(run.members, ensemble.members):
        _assert_opens_alone(member)
        assert member.kind == "model"
        assert member.fitted_train_window == model.fitted_train_bounds
    _assert_no_old_records(setup[0] / "ensemble")


def test_a_model_walk_forward_unit_round_trips_and_survives_a_copy(setup):
    tmp_path, dataset_config, _, dates = setup
    model = make_model(tmp_path / "cv", dataset_config, **dates).collect()

    run = model.train_cv(train_periods=20)

    assert run == TrainedRun.open(run.path) == TrainedRun.open(run.path / "run.json")
    assert run.kind == "walk_forward" and run.checkpoint is None
    assert run.train_window is None and run.metrics == {}
    assert [fold.index for fold in run.folds] == list(range(len(run.folds))) and run.folds
    for fold in run.folds:
        _assert_opens_alone(fold)
        assert fold.kind == "model" and fold.path == run.path / f"fold_{fold.index}"
        assert "test_ic" in fold.metrics
    assert run.cv_mean["cv_n_folds"] == len(run.folds)
    _assert_no_old_records(tmp_path / "cv")

    moved = tmp_path / "copied" / run.path.name
    shutil.copytree(run.path, moved)
    copy = TrainedRun.open(moved)
    assert copy.path == moved
    assert [f.checkpoint for f in copy.folds] == [
        moved / f.checkpoint.relative_to(run.path) for f in run.folds
    ]
    assert [(f.train_window, f.fitted_train_window, f.test_window, f.metrics) for f in copy.folds] == [
        (f.train_window, f.fitted_train_window, f.test_window, f.metrics) for f in run.folds
    ]
    assert copy.cv_mean == run.cv_mean


def test_a_seed_ensemble_walk_forward_unit_round_trips(setup):
    ensemble = _ensemble(setup, "ensemble_cv").collect()

    run = ensemble.train_cv(train_periods=20)

    assert run == TrainedRun.open(run.path)
    assert run.folds and all(fold.kind == "ensemble" for fold in run.folds)
    for fold in run.folds:
        _assert_opens_alone(fold)
        assert [member.seed for member in fold.members] == [0, 1]
        for member in fold.members:
            _assert_opens_alone(member)
            assert member.test_window == fold.test_window
    _assert_no_old_records(setup[0] / "ensemble_cv")


def test_a_walk_forward_unit_with_a_fold_in_an_old_layout_is_refused(setup):
    tmp_path, dataset_config, _, dates = setup
    run = make_model(tmp_path / "old", dataset_config, **dates).collect().train_cv(
        train_periods=20
    )
    (run.path / "fold_1" / "run.json").unlink()

    with pytest.raises(ValueError, match="fold_1 has no run.json.*retrain"):
        TrainedRun.open(run.path)


def test_a_walk_forward_record_that_disagrees_with_a_fold_is_refused(setup):
    tmp_path, dataset_config, _, dates = setup
    run = make_model(tmp_path / "drift", dataset_config, **dates).collect().train_cv(
        train_periods=20
    )
    path = run.path / "fold_0" / "run.json"
    record = json.loads(path.read_text())
    record["test_window"][1] = record["test_window"][0]
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="fold 0 differently.*retrain"):
        TrainedRun.open(run.path)


# ---------------------------------------------------------------- the run layer (#132)


class _ResolvingHead(FirstFeatureHead):
    """A library head that reports the hyperparameters it resolved."""

    def _resolved_hyperparameters(self):
        return {"depth": 3, **self.config.hyperparameters}


def test_every_trained_kind_opens_through_open_run(setup):
    tmp_path, dataset_config, _, dates = setup
    _, checkpoint = _trained(setup)
    ensemble_checkpoint = _ensemble(setup, "open_ensemble").collect().train()
    walk = make_model(tmp_path / "open_cv", dataset_config, **dates).collect().train_cv(
        train_periods=20
    )

    for path in (checkpoint, checkpoint.parent, ensemble_checkpoint, walk.path):
        run = open_run(path)
        assert isinstance(run, TrainedRun)
        assert run == TrainedRun.open(path)
    assert [open_run(p).kind for p in (checkpoint, ensemble_checkpoint, walk.path)] == [
        "model", "ensemble", "walk_forward"
    ]


def test_the_header_records_the_kind_and_when_it_was_written(setup):
    _, checkpoint = _trained(setup)
    run = TrainedRun.open(checkpoint)

    assert pd.Timestamp(run.written_at).tzinfo is not None


def test_a_unit_of_the_previous_format_version_is_refused(setup):
    _, checkpoint = _trained(setup)
    path = checkpoint.parent / "run.json"
    record = json.loads(path.read_text())
    assert record["format_version"] == FORMAT_VERSION
    path.write_text(json.dumps({**record, "format_version": FORMAT_VERSION - 1}))

    with pytest.raises(ValueError, match=f"format_version {FORMAT_VERSION - 1}.*retrain"):
        open_run(checkpoint)
    with pytest.raises(ValueError, match=f"format_version {FORMAT_VERSION - 1}.*retrain"):
        TrainedRun.open(checkpoint)


def test_open_run_refuses_a_directory_without_run_json(tmp_path):
    (tmp_path / "unit").mkdir()

    with pytest.raises(ValueError, match="no run.json.*retrain"):
        open_run(tmp_path / "unit")


def test_open_run_refuses_an_unknown_kind(setup):
    _, checkpoint = _trained(setup)
    path = checkpoint.parent / "run.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "kind": "bogus"}))

    with pytest.raises(ValueError, match="kind 'bogus'"):
        open_run(checkpoint)


def test_opening_a_trained_run_loads_no_other_layer(setup):
    _, checkpoint = _trained(setup)
    code = (
        "import sys\n"
        "from quantlab.runs.directory import open_run\n"
        f"open_run({str(checkpoint)!r})\n"
        "layers = ('quantlab.backtest.base', 'quantlab.backtest', 'quantlab.model',\n"
        "          'quantlab.factor', 'quantlab.label', 'quantlab.runs.backtest_run')\n"
        "print(sorted(m for m in sys.modules if m.startswith(layers)))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_resolved_hyperparameters_are_recorded_in_the_trained_run(setup):
    tmp_path, dataset_config, _, dates = setup
    model = make_model(
        tmp_path / "resolved", dataset_config, head=_ResolvingHead,
        hyperparameters={"lr": 0.1}, **dates,
    )
    checkpoint = model.collect().train()

    run = TrainedRun.open(checkpoint)

    assert run.resolved_hyperparameters == {"depth": 3, "lr": 0.1}
    assert "resolved_hyperparameters" not in run.config
    assert "resolved_hyperparameters" not in model.get_config()
    # The config is the rebuild recipe alone, and rebuilds the model.
    assert type(model).from_config(run.config).get_config() == model.get_config()


def test_a_model_that_resolves_nothing_records_none(setup):
    _, checkpoint = _trained(setup)

    assert TrainedRun.open(checkpoint).resolved_hyperparameters is None


def test_a_config_json_with_a_record_key_is_refused_on_rebuild(setup):
    model, checkpoint = _trained(setup)
    config = {**TrainedRun.open(checkpoint).config, "resolved_hyperparameters": {"depth": 3}}

    with pytest.raises(ValueError, match="resolved_hyperparameters"):
        type(model).from_config(config)
