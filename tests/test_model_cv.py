"""Cross-validation tests for the model layer (quick task 260914-lno).

Before this file the repository had ZERO `train_cv` tests.

The first two tests are GOLDENS captured against the pre-refactor
`quantlab/base/model.py`, before `BaseModel` was split into
`BaseModel` / `DLModel` / `MLModel` and before the fold-boundary arithmetic
was pulled out of `train_cv`'s two copy-pasted branches into one generator.
They were run green on the untouched code and committed on their own, ahead of
any production change. Their assertions must not be edited afterwards: a
baseline taken AFTER an extraction can only detect later drift, never drift
the extraction itself introduced.

What the goldens pin, for a 130-timestamp panel and
`train_cv(train_periods=50, gap_periods=3)`:

- exactly 7 fold directories `{cls}_cv_fold_{i}` (i = 0..6), each holding
  exactly `{cls}_cv_fold_{i}.pth` and `config.json`;
- the dates each fold ACTUALLY trained on: training indices `i*10 .. i*10+49`,
  test indices `i*10+53 .. i*10+62`, rendered with `np.datetime_as_string`
  from the collected panel's own timestamp coordinate.

The dates are observed from inside training -- the stub head records
`self.config`'s four dates in `_init_optim` -- so a fold that computed the
right dates but trained on different ones still turns these tests red.

Everything is synthetic, CPU-only and offline.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

from quantlab.base.config import DLConfig, MLConfig
from quantlab.base.model import BaseModel, DLModel, MLModel

# --------------------------------------------------------------------------
# Synthetic panel geometry
# --------------------------------------------------------------------------

N_TIMES = 130
N_SYMBOLS = 3
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)
START = np.datetime_as_string(TIMES[0], unit="D")
END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")

#: The golden geometry: 130 timestamps, train 50, gap 3 -> test 10, 7 folds.
GOLDEN_TRAIN_PERIODS = 50
GOLDEN_GAP_PERIODS = 3
GOLDEN_N_FOLDS = 7

#: `(train_start, train_end, test_start, test_end)` for every fold that
#: reached `_init_optim`, in the order the folds trained. A list append is
#: atomic under the GIL, so the threading branch can share it.
DL_FOLD_DATES: list[tuple[str, str, str, str]] = []


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    """`_init_wandb` calls `wandb.init` unconditionally; this is the
    documented bypass."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture(autouse=True)
def _reset_recorded_dates():
    DL_FOLD_DATES.clear()
    yield
    DL_FOLD_DATES.clear()


class FakePanel:
    """A stand-in for a factor/label object: only what `collect()` calls.

    Values are pseudo-random so every fold sees a non-degenerate panel.
    """

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


class GoldenDLHead(DLModel):
    """A tiny torch head: one `nn.Linear` on the last axis, one epoch.

    `_init_optim` is where the dates are recorded: the training loop calls it
    once per fit, on the instance that is actually training (the deep copy,
    in the parallel branch), after the fold's dates were written to its
    config.
    """

    def _init_model(self, num_symbols, num_features, num_labels, hyperparameters):
        return nn.Linear(num_features, num_labels)

    def _init_optim(self, model):
        c = self.config
        DL_FOLD_DATES.append((c.train_start, c.train_end, c.test_start, c.test_end))
        return torch.optim.SGD(model.parameters(), lr=self.config.lr)

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


def _dl_config(tmp_path: Path, save_dir: str) -> DLConfig:
    return DLConfig(
        factors=[FakePanel(["f_a", "f_b"], seed=1)],
        labels=[FakePanel(["ret"], seed=2)],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
        epochs=1,
        batch_size=64,
        num_workers=0,
        lr=1e-3,
        early_stopping=False,
        hyperparameters={},
    )


def _golden_fold_dates(model) -> list[tuple[str, str, str, str]]:
    """The fold dates the pre-refactor formula gives, derived independently
    of any code under test."""
    ts = model.data_backend.get_xarray_dataset(["timestamp", "symbol"]).timestamp.values
    assert len(ts) == N_TIMES
    fmt = np.datetime_as_string
    return [
        (
            fmt(ts[i * 10]),
            fmt(ts[i * 10 + 49]),
            fmt(ts[i * 10 + 53]),
            fmt(ts[i * 10 + 62]),
        )
        for i in range(GOLDEN_N_FOLDS)
    ]


def _assert_golden_fold_dirs(root: Path, cls_name: str, suffix: str) -> None:
    projects = [p for p in root.iterdir() if p.is_dir()]
    assert len(projects) == 1, f"expected one CV project dir, got {projects}"
    fold_dirs = sorted(p.name for p in projects[0].iterdir())
    assert fold_dirs == sorted(
        f"{cls_name}_cv_fold_{i}" for i in range(GOLDEN_N_FOLDS)
    )
    for name in fold_dirs:
        contents = {p.name for p in (projects[0] / name).iterdir()}
        assert contents == {f"{name}{suffix}", "config.json"}, contents


def test_dl_train_cv_fold_geometry_golden_sequential(tmp_path):
    """Golden: the sequential branch's folds, checkpoints and trained dates.

    Turns red if the fold-boundary arithmetic changes in any way (test size,
    fold count, gap placement, off-by-one on an end index, skipped-fold
    rule), if a fold trains on dates other than the ones it computed, or if
    the checkpoint layout changes.
    """
    model = GoldenDLHead(_dl_config(tmp_path, "ckpt_seq"))
    model.collect()

    model.train_cv(
        train_periods=GOLDEN_TRAIN_PERIODS, gap_periods=GOLDEN_GAP_PERIODS
    )

    assert DL_FOLD_DATES == _golden_fold_dates(model)
    _assert_golden_fold_dirs(tmp_path / "ckpt_seq", "GoldenDLHead", ".pth")


def test_dl_train_cv_fold_geometry_golden_parallel(tmp_path):
    """Golden: `parallel=True` trains the same folds on the same dates.

    Order is not asserted -- threads finish in any order -- but the SET of
    trained date tuples and the checkpoint layout must equal the sequential
    golden. Turns red if the parallel branch's copy of the arithmetic drifts
    from the sequential one, or if a fold's deep copy trains on the original
    instance's dates.
    """
    model = GoldenDLHead(_dl_config(tmp_path, "ckpt_par"))
    model.collect()

    model.train_cv(
        train_periods=GOLDEN_TRAIN_PERIODS,
        gap_periods=GOLDEN_GAP_PERIODS,
        parallel=True,
        njobs=2,
    )

    assert sorted(DL_FOLD_DATES) == sorted(_golden_fold_dates(model))
    assert len(DL_FOLD_DATES) == GOLDEN_N_FOLDS
    _assert_golden_fold_dirs(tmp_path / "ckpt_par", "GoldenDLHead", ".pth")


# ==========================================================================
# Post-extraction tests (added after the goldens above were committed)
# ==========================================================================
#
# Everything below exercises the extracted pieces directly: the single fold
# generator `BaseModel._cv_folds`, the claim that BOTH `train_cv` branches
# consume it, the per-fold results `train_cv` now returns, and the
# `{cls}_cv_summary` W&B run holding the fold means.


ML_FOLD_DATES: list[tuple[str, str, str, str]] = []

METRIC_KEYS = ("loss", "mse", "rmse", "mae", "r2", "ic", "rank_ic")
FOLD_KEYS = {"fold", "train_start", "train_end", "test_start", "test_end"}


@pytest.fixture(autouse=True)
def _reset_ml_recorded_dates():
    ML_FOLD_DATES.clear()
    yield
    ML_FOLD_DATES.clear()


class FakeRecorder:
    """Records what the model writes to a W&B run."""

    def __init__(self, name: str):
        self.name = name
        self.logs: list = []
        self.summary: dict = {}
        self.finished = 0

    def log(self, data, step=None):
        self.logs.append((dict(data), step))

    def finish(self):
        self.finished += 1


@pytest.fixture
def recorders(monkeypatch) -> list[FakeRecorder]:
    """Patched on the CLASS: the parallel branch deep-copies the instance, and
    an instance-level patch would bind recorders to the original, not to the
    fold copy that trains."""
    created: list[FakeRecorder] = []

    def fake_init_wandb(self, project_name, experiment_name):
        recorder = FakeRecorder(experiment_name)
        created.append(recorder)
        self._wandb_recorder = recorder

    monkeypatch.setattr(BaseModel, "_init_wandb", fake_init_wandb)
    return created


class StubMLHead(MLModel):
    """A numpy head: predicts the first factor, records each fold's dates."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        c = self.config
        ML_FOLD_DATES.append((c.train_start, c.train_end, c.test_start, c.test_end))
        return {"num_labels": num_labels}

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def _ml_config(tmp_path: Path, save_dir: str) -> MLConfig:
    return MLConfig(
        factors=[FakePanel(["f_a", "f_b"], seed=1)],
        labels=[FakePanel(["ret"], seed=2)],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
    )


def _as_tuples(folds: list[dict]) -> list[tuple[str, str, str, str]]:
    return [
        (f["train_start"], f["train_end"], f["test_start"], f["test_end"])
        for f in folds
    ]


def _relative_checkpoints(results: list[dict], save_root: Path) -> set[str]:
    """Checkpoint paths relative to the project dir (the project name carries
    a timestamp, so two runs never share it)."""
    out = set()
    for r in results:
        rel = Path(r["checkpoint"]).relative_to(save_root)
        out.add(str(Path(*rel.parts[1:])))
    return out


# --------------------------------------------------------------------------
# _cv_folds geometry
# --------------------------------------------------------------------------


def _day_index(date: str) -> int:
    return int(
        (np.datetime64(date, "D") - TIMES[0]) / np.timedelta64(1, "D")
    )


def test_cv_folds_geometry():
    """The generator alone: 7 folds, exact keys, a 3-timestamp gap inside
    each fold, disjoint train/test, and back-to-back test windows. Turns red
    on any change to the extracted arithmetic, independently of the goldens'
    trained-date observation."""
    folds = BaseModel._cv_folds(TIMES, train_periods=50, gap_periods=3)

    assert len(folds) == GOLDEN_N_FOLDS
    assert [f["fold"] for f in folds] == list(range(GOLDEN_N_FOLDS))
    for f in folds:
        assert set(f) == FOLD_KEYS
        train_start, train_end = _day_index(f["train_start"]), _day_index(f["train_end"])
        test_start, test_end = _day_index(f["test_start"]), _day_index(f["test_end"])
        assert train_end - train_start + 1 == 50
        assert test_end - test_start + 1 == 10
        assert test_start - train_end - 1 == 3, "gap between train_end and test_start"
        assert train_end < test_start
    for prev, nxt in zip(folds, folds[1:]):
        assert _day_index(nxt["test_start"]) == _day_index(prev["test_end"]) + 1


def test_cv_folds_returns_empty_when_data_is_too_short():
    """60 timestamps cannot hold train 50 + gap 3 + test 10."""
    assert BaseModel._cv_folds(TIMES[:60], train_periods=50, gap_periods=3) == []


# --------------------------------------------------------------------------
# Both branches consume the one generator
# --------------------------------------------------------------------------


HANDMADE_FOLDS = [
    {
        "fold": 3,
        "train_start": np.datetime_as_string(TIMES[5], unit="D"),
        "train_end": np.datetime_as_string(TIMES[40], unit="D"),
        "test_start": np.datetime_as_string(TIMES[45], unit="D"),
        "test_end": np.datetime_as_string(TIMES[60], unit="D"),
    },
    {
        "fold": 5,
        "train_start": np.datetime_as_string(TIMES[20], unit="D"),
        "train_end": np.datetime_as_string(TIMES[70], unit="D"),
        "test_start": np.datetime_as_string(TIMES[71], unit="D"),
        "test_end": np.datetime_as_string(TIMES[90], unit="D"),
    },
]


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
def test_both_train_cv_branches_train_exactly_what_cv_folds_yields(tmp_path, monkeypatch, recorders, parallel):
    """Replace the generator with two handmade folds (numbered 3 and 5, with
    geometry the real formula never produces): each branch must train exactly
    those two, on exactly those dates. Turns red if either branch grows its
    own copy of the fold arithmetic again."""
    monkeypatch.setattr(
        BaseModel,
        "_cv_folds",
        staticmethod(lambda timestamps, train_periods, gap_periods: [dict(f) for f in HANDMADE_FOLDS]),
    )
    save_dir = "ckpt_par" if parallel else "ckpt_seq"
    model = StubMLHead(_ml_config(tmp_path, save_dir))
    model.collect()

    results = model.train_cv(train_periods=50, gap_periods=3, parallel=parallel, njobs=2)

    assert sorted(ML_FOLD_DATES) == sorted(_as_tuples(HANDMADE_FOLDS))
    assert sorted(r["fold"] for r in results) == [3, 5]
    projects = list((tmp_path / save_dir).iterdir())
    assert len(projects) == 1
    assert sorted(p.name for p in projects[0].iterdir()) == ["StubMLHead_cv_fold_3", "StubMLHead_cv_fold_5"]


# --------------------------------------------------------------------------
# CV results and the summary run (ML)
# --------------------------------------------------------------------------


def test_ml_train_cv_returns_per_fold_results_and_loadable_checkpoints(tmp_path, recorders):
    """`train_periods=50`, no gap, 130 timestamps -> 8 folds. Each result
    carries the fold's dates, its run name, an existing `.joblib` and the
    seven prefixed test metrics; every checkpoint loads into a fresh,
    never-collected instance that predicts `[T, S, L]`."""
    model = StubMLHead(_ml_config(tmp_path, "ckpt"))
    model.collect()
    expected = BaseModel._cv_folds(
        model.data_backend.get_xarray_dataset(["timestamp", "symbol"]).timestamp.values, 50, 0
    )

    results = model.train_cv(train_periods=50)

    assert len(results) == 8
    assert [{k: r[k] for k in FOLD_KEYS} for r in results] == expected
    for r in results:
        assert set(r) == FOLD_KEYS | {"experiment_name", "checkpoint"} | {f"test_{k}" for k in METRIC_KEYS}
        assert r["experiment_name"] == f"StubMLHead_cv_fold_{r['fold']}"
        ckpt = Path(r["checkpoint"])
        assert ckpt.suffix == ".joblib" and ckpt.is_file()
        fresh = StubMLHead(_ml_config(tmp_path, "unused")).load(ckpt)
        assert fresh.predict(np.zeros((4, N_SYMBOLS, 2))).shape == (4, N_SYMBOLS, 1)


def test_ml_train_cv_writes_fold_means_to_a_separate_summary_run(tmp_path, recorders):
    """8 fold runs plus ONE `{cls}_cv_summary` run, created last, whose
    summary holds `cv_mean_test_*` (finite-value means of the folds) and
    `cv_n_folds`, and which is finished exactly once. A separate run because
    each fold's `_fit` has already finished its own run by the time the means
    exist."""
    model = StubMLHead(_ml_config(tmp_path, "ckpt"))
    model.collect()

    results = model.train_cv(train_periods=50)

    names = [r.name for r in recorders]
    assert names[:-1] == [f"StubMLHead_cv_fold_{i}" for i in range(8)]
    assert names[-1] == "StubMLHead_cv_summary"
    summary_run = recorders[-1]
    assert summary_run.finished == 1
    assert summary_run.summary["cv_n_folds"] == 8
    assert set(summary_run.summary) == {f"cv_mean_test_{k}" for k in METRIC_KEYS} | {"cv_n_folds"}
    for k in METRIC_KEYS:
        values = [r[f"test_{k}"] for r in results if np.isfinite(r[f"test_{k}"])]
        assert values, k
        assert summary_run.summary[f"cv_mean_test_{k}"] == pytest.approx(float(np.mean(values)))


def test_ml_train_cv_parallel_matches_sequential(tmp_path, recorders):
    """Same folds, same checkpoint layout, same trained dates. Separate save
    dirs: the project name is only second-resolution."""
    seq = StubMLHead(_ml_config(tmp_path, "ckpt_seq"))
    seq.collect()
    seq_results = seq.train_cv(train_periods=50)
    seq_dates = sorted(ML_FOLD_DATES)
    ML_FOLD_DATES.clear()

    par = StubMLHead(_ml_config(tmp_path, "ckpt_par"))
    par.collect()
    par_results = par.train_cv(train_periods=50, parallel=True, njobs=2)

    assert sorted(ML_FOLD_DATES) == seq_dates
    assert sorted(_as_tuples(par_results)) == sorted(_as_tuples(seq_results))
    assert _relative_checkpoints(par_results, tmp_path / "ckpt_par") == _relative_checkpoints(
        seq_results, tmp_path / "ckpt_seq"
    )


def test_dl_train_cv_results_carry_dates_only_and_open_no_summary_run(tmp_path, recorders):
    """`DLModel._fit` returns no metrics, so a DL fold result is the fold's
    dates plus its run name and checkpoint, and no summary run is created --
    DL CV behaviour stays what the goldens pinned."""
    model = GoldenDLHead(_dl_config(tmp_path, "ckpt"))
    model.collect()

    results = model.train_cv(train_periods=GOLDEN_TRAIN_PERIODS, gap_periods=GOLDEN_GAP_PERIODS)

    assert len(results) == GOLDEN_N_FOLDS
    for r in results:
        assert set(r) == FOLD_KEYS | {"experiment_name", "checkpoint"}
        assert Path(r["checkpoint"]).suffix == ".pth" and Path(r["checkpoint"]).is_file()
    assert len(recorders) == GOLDEN_N_FOLDS
    assert all(not r.name.endswith("_cv_summary") for r in recorders)
