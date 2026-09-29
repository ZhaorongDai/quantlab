"""`SeedEnsemble` builds, trains, saves, checks and loads a seed ensemble.

What is locked here, and what turns it red:

- Construction refuses fewer than two seeds or a repeated seed; each member
  is the wrapped model's class on its config with `random_seed` replaced.
- `collect()` collects once: every member holds the same data backend object.
- Different seeds give different member predictions for a head that draws
  from the seeded generator, and `predict_window` is `average_predictions`
  of the members' predictions.
- `train()` writes `SeedEnsemble_trial_*/` holding `member_{k}/` (each with
  the usual checkpoint, config.json, metrics.json, ic_series.csv and
  test_predictions.zarr), an ensemble-level `config.json` with the shared
  dates and label configs, and `ensemble.json` listing every member's class,
  relative checkpoint and seed; it returns the path of `ensemble.json`.
- A member failing mid-training leaves no `ensemble.json`, and the member
  directories already written stay.
- `load(ensemble.json)` restores every member; `check_checkpoint` refuses a
  missing file, a malformed manifest, an unknown `format_version`, a member
  class, count or seed that does not match, and a missing member checkpoint.
- `get_config()` / `from_config()` round-trip the wrapped model's config and
  the seeds.
- Layout: `quantlab/ensemble_model/` has empty `__init__.py` files, no other
  quantlab module imports it, and it imports no backtest module.

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.ensemble_model.seed import SeedEnsemble
from quantlab.utils.ensemble import average_predictions
from quantlab.utils.jsonable import to_jsonable
from tests.test_backtest_contracts import (
    REPO_ROOT,
    _is_or_under,
    _python_files,
    _resolved_imports,
)
from tests.backtest_fixtures import (
    FirstFeatureHead,
    SeededHead,
    make_model,
    write_price_store,
)

N_BARS = 60
SEEDS = [0, 1, 2]
MEMBER_FILES = [
    "config.json",
    "ic_series.csv",
    "metrics.json",
    "test_predictions.zarr",
]


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    return dataset_config, bars


def _dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )


def _model(tmp_path, dataset_config, bars, *, head=SeededHead, name="models"):
    return make_model(tmp_path / name, dataset_config, head=head, **_dates(bars))


def _trained(tmp_path, seeds=SEEDS):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), list(seeds))
    manifest = ensemble.collect().train()
    return ensemble, manifest, dataset_config, bars


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seeds", [[], [0], [3, 3], [0, 1, 0]])
def test_fewer_than_two_or_repeated_seeds_raise(tmp_path, seeds):
    dataset_config, bars = _setup(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        SeedEnsemble(_model(tmp_path, dataset_config, bars), seeds)


def test_members_are_the_model_class_with_each_seed(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    model = _model(tmp_path, dataset_config, bars)

    ensemble = SeedEnsemble(model, [7, 3])

    assert [type(m) for m in ensemble.members] == [SeededHead, SeededHead]
    assert [m.config.random_seed for m in ensemble.members] == [7, 3]
    assert ensemble.seeds == (7, 3)
    assert all(m is not model for m in ensemble.members)
    assert ensemble.members[0].get_config() == {**model.get_config(), "random_seed": 7}
    assert ensemble.labels == model.labels
    assert ensemble.train_bounds == model.train_bounds
    assert ensemble.test_bounds == model.test_bounds
    assert ensemble.label_delays == model.label_delays


# --------------------------------------------------------------------------
# Shared panel and predictions
# --------------------------------------------------------------------------


def test_members_share_one_data_backend(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS)

    ensemble.collect()

    backend = ensemble.members[0].data_backend
    assert all(m.data_backend is backend for m in ensemble.members)
    assert backend.get_xarray_dataset().sizes["timestamp"] > 0


def test_seeds_give_different_members_and_the_prediction_is_their_average(tmp_path):
    ensemble, _, _, bars = _trained(tmp_path)
    start, end = _day(bars[30]), _day(bars[50])

    members = [m.predict_window(start, end) for m in ensemble.members]
    out = ensemble.predict_window(start, end)

    for a, b in zip(members, members[1:]):
        assert not np.allclose(a["fwd_ret_1"].values, b["fwd_ret_1"].values, equal_nan=True)
    xr.testing.assert_allclose(out, average_predictions(members))
    assert out.timestamp.values[0] == bars[30] and out.timestamp.values[-1] == bars[50]


# --------------------------------------------------------------------------
# train() on disk
# --------------------------------------------------------------------------


def test_train_writes_members_ensemble_config_and_manifest(tmp_path):
    ensemble, manifest, _, bars = _trained(tmp_path)
    directory = manifest.parent

    assert manifest.is_absolute() and manifest.name == "ensemble.json"
    assert directory.parent == Path(ensemble.members[0].config.model_save_dir).absolute()
    assert directory.name.startswith("SeedEnsemble_trial_")
    assert sorted(p.name for p in directory.iterdir()) == [
        "config.json",
        "ensemble.json",
        "member_0",
        "member_1",
        "member_2",
    ]

    saved = json.loads(manifest.read_text())
    assert saved["format_version"] == 1
    assert saved["members"] == [
        {
            "name": "tests.backtest_fixtures.SeededHead",
            "checkpoint": f"member_{k}/SeededHead_member_{k}.joblib",
            "seed": seed,
        }
        for k, seed in enumerate(SEEDS)
    ]
    for k, seed in enumerate(SEEDS):
        member_dir = directory / f"member_{k}"
        assert sorted(p.name for p in member_dir.iterdir()) == sorted(
            [f"SeededHead_member_{k}.joblib", *MEMBER_FILES]
        )
        member_config = json.loads((member_dir / "config.json").read_text())
        assert member_config["random_seed"] == seed

    shared = json.loads((directory / "config.json").read_text())
    dates = _dates(bars)
    for key in ("train_start", "train_end", "test_start", "test_end"):
        assert shared[key] == dates[key]
    assert shared["labels"] == json.loads(
        json.dumps(to_jsonable([label.get_config() for label in ensemble.labels]))
    )
    assert "factors" not in shared and "random_seed" not in shared


def test_two_trainings_get_two_directories(tmp_path):
    ensemble, first, _, _ = _trained(tmp_path)
    second = ensemble.train()
    assert first.parent != second.parent
    assert first.is_file() and second.is_file()


def test_a_member_failing_leaves_no_manifest(tmp_path, monkeypatch):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS)
    fit = SeededHead._fit_model

    def failing(self, train_rows, val_rows):
        if self.config.random_seed == 1:
            raise RuntimeError("member 1 broke")
        fit(self, train_rows, val_rows)

    monkeypatch.setattr(SeededHead, "_fit_model", failing)

    with pytest.raises(RuntimeError, match="member 1 broke"):
        ensemble.collect().train()

    root = Path(ensemble.members[0].config.model_save_dir)
    assert list(root.rglob("ensemble.json")) == []
    (directory,) = root.glob("SeedEnsemble_trial_*")
    assert (directory / "member_0" / "SeededHead_member_0.joblib").is_file()
    assert not (directory / "config.json").exists()


# --------------------------------------------------------------------------
# load() and check_checkpoint()
# --------------------------------------------------------------------------


def test_load_restores_every_member(tmp_path):
    trained, manifest, dataset_config, bars = _trained(tmp_path)
    fresh = SeedEnsemble(_model(tmp_path, dataset_config, bars, name="other"), SEEDS)

    assert fresh.load(manifest) is fresh
    assert fresh.load(str(manifest)) is fresh

    start, end = _day(bars[30]), _day(bars[50])
    xr.testing.assert_identical(
        fresh.predict_window(start, end), trained.predict_window(start, end)
    )


def test_check_checkpoint_accepts_the_manifest_it_wrote(tmp_path):
    ensemble, manifest, _, _ = _trained(tmp_path)
    assert ensemble.check_checkpoint(manifest) is None


def _rewrite(manifest: Path, edit) -> Path:
    saved = json.loads(manifest.read_text())
    edit(saved)
    manifest.write_text(json.dumps(saved))
    return manifest


def test_check_checkpoint_refuses_bad_manifests(tmp_path):
    ensemble, manifest, dataset_config, bars = _trained(tmp_path)
    original = manifest.read_text()

    with pytest.raises(FileNotFoundError):
        ensemble.check_checkpoint(manifest.parent / "missing.json")

    manifest.write_text("{not json")
    with pytest.raises(ValueError, match="ensemble.json"):
        ensemble.check_checkpoint(manifest)

    manifest.write_text(json.dumps([1, 2]))
    with pytest.raises(ValueError, match="ensemble.json"):
        ensemble.check_checkpoint(manifest)

    manifest.write_text(original)
    _rewrite(manifest, lambda saved: saved.update(format_version=2))
    with pytest.raises(ValueError, match="format_version"):
        ensemble.check_checkpoint(manifest)

    manifest.write_text(original)
    _rewrite(manifest, lambda saved: saved["members"][0].pop("checkpoint"))
    with pytest.raises(ValueError, match="checkpoint"):
        ensemble.check_checkpoint(manifest)

    manifest.write_text(original)
    _rewrite(manifest, lambda saved: saved["members"][2].update(seed=9))
    with pytest.raises(ValueError, match="seed"):
        ensemble.check_checkpoint(manifest)

    manifest.write_text(original)
    two = SeedEnsemble(_model(tmp_path, dataset_config, bars, name="two"), [0, 1])
    with pytest.raises(ValueError, match="3 members"):
        two.check_checkpoint(manifest)

    other_class = SeedEnsemble(
        _model(tmp_path, dataset_config, bars, head=FirstFeatureHead, name="ffh"), SEEDS
    )
    with pytest.raises(ValueError, match="SeededHead"):
        other_class.check_checkpoint(manifest)

    (manifest.parent / "member_1" / "SeededHead_member_1.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="member_1"):
        ensemble.check_checkpoint(manifest)
    with pytest.raises(FileNotFoundError, match="member_1"):
        ensemble.load(manifest)


# --------------------------------------------------------------------------
# get_config / from_config
# --------------------------------------------------------------------------


def test_config_round_trips_the_wrapped_model_and_seeds(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    model = _model(tmp_path, dataset_config, bars)
    ensemble = SeedEnsemble(model, [5, 1, 4])

    config = ensemble.get_config()

    assert config == {
        "name": "quantlab.ensemble_model.seed.SeedEnsemble",
        "seeds": [5, 1, 4],
        "model": model.get_config(),
    }
    saved = json.loads(json.dumps(to_jsonable(config)))
    rebuilt = SeedEnsemble.from_config(saved)
    assert isinstance(rebuilt, SeedEnsemble)
    assert to_jsonable(rebuilt.get_config()) == to_jsonable(config)
    assert [type(m) for m in rebuilt.members] == [SeededHead] * 3
    assert [m.config.random_seed for m in rebuilt.members] == [5, 1, 4]


# --------------------------------------------------------------------------
# Package layout and layering
# --------------------------------------------------------------------------


def test_ensemble_package_layout_and_layering():
    """The ensemble package sits above the model layer: its `__init__.py`
    files are empty, nothing else in quantlab imports it, and it imports no
    backtest module."""
    package = REPO_ROOT / "quantlab/ensemble_model"
    for init in (package / "__init__.py", package / "_support/__init__.py"):
        assert init.stat().st_size == 0, init

    # Positive control: the resolver sees the package's own imports.
    assert "quantlab.ensemble_model._support.base" in _resolved_imports(
        package / "seed.py"
    )

    offenders = {
        str(path.relative_to(REPO_ROOT)): sorted(
            name
            for name in _resolved_imports(path)
            if _is_or_under(name, "quantlab.ensemble_model")
        )
        for path in _python_files(REPO_ROOT / "quantlab")
        if package not in path.parents
    }
    assert {path: names for path, names in offenders.items() if names} == {}

    for path in _python_files(package):
        backtest = sorted(
            name
            for name in _resolved_imports(path)
            if _is_or_under(name, "quantlab.backtest")
            or _is_or_under(name, "quantlab.base.backtest")
        )
        assert backtest == [], (path, backtest)
