"""Ensemble-level evaluation files of the averaged prediction (issue #59).

After every member trains, the ensemble directory gets `metrics.json`,
`ic_series.csv` and `test_predictions.zarr` for the averaged prediction, in
the single-model layout, before `ensemble.json` is written.

What turns this file red:

- the ensemble `metrics.json` holds a key outside the IC family (`ic`,
  `rank_ic`, `icir`, `rank_icir` per split), or `val_*` keys without a
  validation segment, or misses them with one;
- a value differs from the same panel metrics computed independently on
  `average_predictions` of the members' predictions over the member's
  purged train / validation / test segments, against the raw first label;
- `ic_series.csv` is not the per-bar series of those metrics in the
  single-model layout (`split, timestamp, ic, rank_ic`, splits in order);
- `test_predictions.zarr` is not the average of the members' own
  `test_predictions.zarr`;
- a member's files differ from those of the same model trained alone;
- a failure while writing the ensemble files leaves an `ensemble.json`;
- `BaseEnsemble._train_into` does not return the manifest and the metrics,
  or writes `metrics.json` when told not to.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.ensemble_model._support import base as ensemble_base
from quantlab.ensemble_model.seed import SeedEnsemble
from quantlab.utils.ensemble import average_predictions
from quantlab.utils.metrics import regression_panel_metrics
from tests.backtest_fixtures import SeededHead, make_model, write_price_store

N_BARS = 60
SEEDS = [0, 1, 2]
IC_KEYS = ("ic", "rank_ic", "icir", "rank_icir")
ENSEMBLE_FILES = [
    "config.json",
    "ensemble.json",
    "ic_series.csv",
    "member_0",
    "member_1",
    "member_2",
    "metrics.json",
    "test_predictions.zarr",
]


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _model(tmp_path, *, val_size=0.0, seed=None, name="models"):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    model = make_model(
        tmp_path / name,
        dataset_config,
        head=SeededHead,
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )
    changes = {"val_size": val_size}
    if seed is not None:
        changes["random_seed"] = seed
    return SeededHead(dataclasses.replace(model.config, **changes))


def _trained(tmp_path, *, val_size=0.0):
    ensemble = SeedEnsemble(_model(tmp_path, val_size=val_size), SEEDS)
    manifest = ensemble.collect().train()
    return ensemble, manifest


def _expected(ensemble):
    """Metrics and per-bar series computed independently of the ensemble code."""
    first = ensemble.members[0]
    data = first.data_backend.get_xarray_dataset(["timestamp", "symbol"]).sortby(
        ["timestamp", "symbol"]
    )
    averaged = average_predictions([m.predict_panel(data) for m in ensemble.members])
    label = first.get_label_names()[0]
    metrics, series = {}, {}
    for split, part in zip(("train", "val", "test"), first._fit_segments(data)):
        stamps = part.timestamp.values
        if len(stamps) == 0:
            continue
        pred = averaged[label].sel(timestamp=stamps, symbol=data.symbol.values).values
        target = data[label].sel(timestamp=stamps).values
        values, per_bar = regression_panel_metrics(pred, target, return_series=True)
        for key in IC_KEYS:
            metrics[f"{split}_{key}"] = values[key]
        series[split] = (stamps, per_bar["ic"], per_bar["rank_ic"])
    return metrics, series


def _close(saved, expected):
    if expected is None or not np.isfinite(expected):
        return saved is None
    return saved == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "val_size, splits", [(0.0, ("train", "test")), (0.2, ("train", "val", "test"))]
)
def test_metrics_are_the_ic_family_of_the_averaged_prediction(tmp_path, val_size, splits):
    ensemble, manifest = _trained(tmp_path, val_size=val_size)

    saved = json.loads((manifest.parent / "metrics.json").read_text())
    expected, _ = _expected(ensemble)

    assert sorted(saved) == sorted(f"{s}_{k}" for s in splits for k in IC_KEYS)
    assert sorted(expected) == sorted(saved)
    for key, value in expected.items():
        assert _close(saved[key], value), key
    member = json.loads((manifest.parent / "member_0" / "metrics.json").read_text())
    assert saved["test_ic"] != member["test_ic"]


def test_train_lays_out_the_ensemble_files(tmp_path):
    _, manifest = _trained(tmp_path)
    assert sorted(p.name for p in manifest.parent.iterdir()) == ENSEMBLE_FILES


@pytest.mark.parametrize("val_size", [0.0, 0.2])
def test_ic_series_is_the_per_bar_series_in_the_single_model_layout(tmp_path, val_size):
    ensemble, manifest = _trained(tmp_path, val_size=val_size)

    frame = pd.read_csv(manifest.parent / "ic_series.csv", parse_dates=["timestamp"])
    _, series = _expected(ensemble)

    assert list(frame.columns) == ["split", "timestamp", "ic", "rank_ic"]
    rows = []
    for split, (stamps, ic, rank_ic) in series.items():
        keep = np.isfinite(ic) | np.isfinite(rank_ic)
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "timestamp": stamps[keep],
                    "ic": ic[keep],
                    "rank_ic": rank_ic[keep],
                }
            )
        )
    expected = pd.concat(rows, ignore_index=True)
    assert list(frame["split"].unique()) == list(series)
    pd.testing.assert_frame_equal(frame, expected, check_dtype=False, rtol=1e-12)


def test_test_predictions_are_the_average_of_the_members(tmp_path):
    ensemble, manifest = _trained(tmp_path)
    directory = manifest.parent

    saved = xr.open_zarr(directory / "test_predictions.zarr").load()
    members = [
        xr.open_zarr(directory / f"member_{k}" / "test_predictions.zarr").load()
        for k in range(len(SEEDS))
    ]

    assert list(saved.data_vars) == list(ensemble.members[0].get_label_names())
    xr.testing.assert_allclose(saved, average_predictions(members))
    assert saved.sizes["timestamp"] == members[0].sizes["timestamp"] > 0


def test_member_files_equal_a_model_trained_alone(tmp_path):
    _, manifest = _trained(tmp_path)
    alone = _model(tmp_path / "alone", seed=SEEDS[1])
    checkpoint = alone.collect().train()

    member_dir = manifest.parent / "member_1"
    for name in ("metrics.json", "ic_series.csv"):
        assert (member_dir / name).read_text() == (checkpoint.parent / name).read_text()
    xr.testing.assert_identical(
        xr.open_zarr(member_dir / "test_predictions.zarr").load(),
        xr.open_zarr(checkpoint.parent / "test_predictions.zarr").load(),
    )
    assert "test_loss" in json.loads((member_dir / "metrics.json").read_text())


def test_a_failure_while_writing_the_ensemble_files_leaves_no_manifest(
    tmp_path, monkeypatch
):
    ensemble = SeedEnsemble(_model(tmp_path), SEEDS)

    def failing(*args, **kwargs):
        raise RuntimeError("evaluation broke")

    monkeypatch.setattr(ensemble_base, "ic_panel_metrics", failing)

    with pytest.raises(RuntimeError, match="evaluation broke"):
        ensemble.collect().train()

    root = Path(ensemble.members[0].config.model_save_dir)
    assert list(root.rglob("ensemble.json")) == []
    (directory,) = root.glob("SeedEnsemble_trial_*")
    assert (directory / "member_2" / "SeededHead_member_2.joblib").is_file()


def test_train_into_returns_the_manifest_and_the_metrics(tmp_path):
    ensemble = SeedEnsemble(_model(tmp_path), SEEDS).collect()
    run_dir = tmp_path / "cv" / "fold_0"

    manifest, metrics = ensemble._train_into(run_dir, project_name="cv")

    assert manifest == (run_dir / "ensemble.json").absolute() and manifest.is_file()
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert sorted(saved) == sorted(metrics)
    for key, value in metrics.items():
        assert _close(saved[key], value), key

    quiet = tmp_path / "cv" / "fold_1"
    _, metrics = ensemble._train_into(quiet, project_name="cv", write_metrics=False)
    assert not (quiet / "metrics.json").exists()
    assert (quiet / "ic_series.csv").is_file() and (quiet / "ensemble.json").is_file()
    assert sorted(metrics) == sorted(f"{s}_{k}" for s in ("train", "test") for k in IC_KEYS)


def test_the_base_panel_predictions_match_the_seed_ensemble_override(tmp_path):
    ensemble, _ = _trained(tmp_path)

    base = ensemble_base.BaseEnsemble._member_panel_predictions(ensemble)
    shared = ensemble._member_panel_predictions()

    assert len(base) == len(shared) == len(SEEDS)
    for a, b in zip(base, shared):
        xr.testing.assert_identical(a, b)
