"""Metrics on disk: `metrics.json` from `train()` and the v2 `cv_folds.json` (issue #38).

`train()` writes the `train_*` / `val_*` / `test_*` metrics of the first label
to `metrics.json` beside the checkpoint's `config.json`, so a run's scores
survive without W&B. `train_cv()` keeps every split's metrics in each fold
entry of `cv_folds.json` and adds a top-level `cv_mean` block, the fold means
of every metric, which the `{cls}_cv_summary` W&B run also receives.

What turns this file red:

- `metrics.json` is missing, sits elsewhere, or differs from what the W&B
  summary received;
- a run without a validation segment writes `val_*` keys;
- a fold entry lacks a split, or `cv_mean` is not the fold mean of every
  metric, or differs from the summary run;
- a non-finite metric reaches a file as a bare `NaN` token.

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from quantlab.base.config import ModelConfig
from quantlab.base.model import BaseModel, LibraryModel
from quantlab.utils.jsonable import to_jsonable
from tests.torch_heads import OneBarHead
from tests.label_stubs import StubLabel

N_TIMES = 60
N_SYMBOLS = 4
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype("timedelta64[D]")
START = np.datetime_as_string(TIMES[0], unit="D")
END = np.datetime_as_string(TIMES[-1], unit="D")

METRIC_KEYS = ("loss", "mse", "rmse", "mae", "r2", "ic", "rank_ic")
SPLITS = ("train", "val", "test")


class FakeRecorder:
    """Records what the model writes to a W&B run."""

    def __init__(self, name: str):
        self.name = name
        self.summary: dict = {}
        self.finished = 0

    def log(self, data, step=None):
        pass

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

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


class NaNMetricLibraryHead(StubLibraryHead):
    def _compute_metrics(self, y, pred):
        return {"nan_metric": np.float64("nan"), "finite_metric": np.float64(1.5)}


def _model(tmp_path: Path, cls=StubLibraryHead, **overrides):
    kwargs = dict(
        factors=[FakePanel(["f_a", "f_b"], seed=1)],
        labels=[StubLabel(FakePanel(["ret"], seed=2))],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
        train_start=START,
        train_end=np.datetime_as_string(TIMES[39], unit="D"),
        test_start=np.datetime_as_string(TIMES[40], unit="D"),
        test_end=END,
    )
    kwargs.update(overrides)
    config_cls = ModelConfig if cls is OneBarHead else ModelConfig
    model = cls(config_cls(**kwargs))
    model.collect()
    return model


def _strict_json(path: Path):
    def _reject(token):
        raise ValueError(f"non-standard JSON token {token!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


@pytest.mark.parametrize("cls", [StubLibraryHead, OneBarHead], ids=["library", "torch"])
def test_train_writes_metrics_json_equal_to_the_wandb_summary(tmp_path, recorders, cls):
    checkpoint = _model(tmp_path, cls=cls).train()

    path = checkpoint.parent / "metrics.json"
    assert (checkpoint.parent / "config.json").is_file()
    metrics = _strict_json(path)
    assert set(metrics) == {f"{s}_{k}" for s in SPLITS for k in METRIC_KEYS}
    (run,) = recorders
    assert metrics == to_jsonable(run.summary)


@pytest.mark.parametrize("cls", [StubLibraryHead, OneBarHead], ids=["library", "torch"])
def test_metrics_json_has_no_val_keys_without_a_validation_segment(
    tmp_path, recorders, cls
):
    checkpoint = _model(tmp_path, cls=cls, val_size=0.0).train()

    metrics = _strict_json(checkpoint.parent / "metrics.json")
    assert set(metrics) == {f"{s}_{k}" for s in ("train", "test") for k in METRIC_KEYS}


def test_metrics_json_writes_non_finite_metrics_as_null(tmp_path, recorders):
    checkpoint = _model(tmp_path, cls=NaNMetricLibraryHead).train()

    metrics = _strict_json(checkpoint.parent / "metrics.json")
    for split in SPLITS:
        assert metrics[f"{split}_nan_metric"] is None
        assert metrics[f"{split}_finite_metric"] == 1.5


def test_cv_manifest_v2_holds_every_split_and_the_cv_mean_block(tmp_path, recorders):
    model = _model(tmp_path)

    results = model.train_cv(train_periods=20)

    (project,) = [p for p in (tmp_path / "ckpt").iterdir() if p.is_dir()]
    manifest = _strict_json(project / "cv_folds.json")
    assert manifest["format_version"] == 2
    assert manifest["folds"] == to_jsonable(results)
    for entry in manifest["folds"]:
        for split in SPLITS:
            for k in METRIC_KEYS:
                assert f"{split}_{k}" in entry, (entry["fold"], split, k)

    cv_mean = manifest["cv_mean"]
    assert cv_mean["cv_n_folds"] == len(results)
    for split in SPLITS:
        for k in METRIC_KEYS:
            key = f"{split}_{k}"
            values = [r[key] for r in results if np.isfinite(r[key])]
            assert cv_mean[f"cv_mean_{key}"] == pytest.approx(float(np.mean(values)))

    summary_run = recorders[-1]
    assert summary_run.name == "StubLibraryHead_cv_summary"
    assert cv_mean == to_jsonable(summary_run.summary)
