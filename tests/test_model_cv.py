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

from quantlab.base.config import DLConfig
from quantlab.base.model import BaseModel

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


class GoldenDLHead(BaseModel):
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
