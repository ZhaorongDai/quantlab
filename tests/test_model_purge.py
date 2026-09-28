"""A training run fits only on bars the purge leaves.

The label's value at bar i is i itself, so the rows a head receives in its
fitting hooks name the bars it was fitted on. The label claims a lookahead
of L bars, and every split boundary must drop the last L bars of the earlier
segment: train before test, train before validation, validation before test.

Everything is synthetic, CPU-only and offline.
"""

import json

import numpy as np
import pytest
import torch
import xarray as xr

from quantlab.base.config import DLConfig, MLConfig
from quantlab.base.model import MLModel
from quantlab.dl_model.training import cs_rank_norm
from tests.dl_heads import OneBarHead
from tests.label_stubs import StubLabel

N_TIMES = 20
SYMBOLS = ["S0", "S1"]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype("timedelta64[D]")


def date(i):
    return np.datetime_as_string(TIMES[i], unit="D")


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


class Panel:
    """A factor or label stand-in whose values are the bar index."""

    def __init__(self, name):
        self.name = name
        index = np.repeat(np.arange(N_TIMES, dtype="float32")[:, None], len(SYMBOLS), 1)
        self._ds = xr.Dataset(
            {name: (("timestamp", "symbol"), index)},
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )

    def _get_factor_names(self):
        return [self.name]

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    read = compute

    def get_config(self):
        return {"name": "Panel", "factor_names": [self.name]}


def bars(rows) -> list[int]:
    """The sorted bar indices a block of label rows came from."""
    return sorted({int(v) for v in np.asarray(rows)[..., 0].ravel()})


class RecordingMLHead(MLModel):
    """Records the bars ``_fit_model`` is handed."""

    fitted: dict = {}

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        RecordingMLHead.fitted = {
            "train": bars(train_y),
            "val": None if val_y is None else bars(val_y),
        }

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def common(tmp_path, lookahead, **dates):
    return dict(
        factors=[Panel("f")],
        labels=[StubLabel(Panel("y"), lookahead=lookahead)],
        model_save_dir=str(tmp_path / "models"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=date(0),
        end_date=date(N_TIMES - 1),
        **dates,
    )


FIXED = dict(
    train_start=date(0), train_end=date(11), test_start=date(12), test_end=date(19)
)


def test_ml_fixed_split_drops_the_last_lookahead_training_bars(tmp_path):
    model = RecordingMLHead(MLConfig(**common(tmp_path, 3, **FIXED), val_size=0.0))
    model.collect()
    model.train()
    assert RecordingMLHead.fitted == {"train": list(range(0, 9)), "val": None}


def test_ml_val_split_drops_the_last_lookahead_bars_before_validation_and_test(tmp_path):
    # Training window 0..11, val_size 0.25: train 0..8, validation 9..11.
    # L = 1 drops bar 8 (before validation) and bar 11 (before test).
    model = RecordingMLHead(MLConfig(**common(tmp_path, 1, **FIXED), val_size=0.25))
    model.collect()
    model.train()
    assert RecordingMLHead.fitted == {"train": list(range(0, 8)), "val": [9, 10]}


def test_zero_lookahead_fits_on_every_training_bar(tmp_path):
    model = RecordingMLHead(MLConfig(**common(tmp_path, 0, **FIXED), val_size=0.25))
    model.collect()
    model.train()
    assert RecordingMLHead.fitted == {"train": list(range(0, 9)), "val": [9, 10, 11]}


class RecordingDLHead(OneBarHead):
    """Records the bars its training and validation steps receive.

    Features are the bar index, fed unclipped, so the last row of a window
    names its bar; ranking the target gives every bar a finite target.
    """

    seen: dict = {}

    def _transform_feature(self, x):
        return torch.nan_to_num(x, nan=0.0)

    def _transform_target(self, y, training):
        return cs_rank_norm(y), None

    def _init_optim(self, model):
        RecordingDLHead.seen = {"train": set(), "val": set()}
        return super()._init_optim(model)

    def _train_one_batch(self, epoch, batch):
        RecordingDLHead.seen["train"].add(int(batch.x[0, -1, 0]))
        return super()._train_one_batch(epoch, batch)

    def _val_one_batch(self, epoch, batch):
        # The epoch's validation pass comes before its test pass; the later
        # calls score each split for the metrics.
        if not RecordingDLHead.seen.get("tested"):
            RecordingDLHead.seen["val"].add(int(batch.x[0, -1, 0]))
        return super()._val_one_batch(epoch, batch)

    def _test_one_batch(self, epoch, batch):
        RecordingDLHead.seen["tested"] = True


def test_dl_val_split_drops_the_last_lookahead_bars_before_validation_and_test(tmp_path):
    model = RecordingDLHead(
        DLConfig(**common(tmp_path, 1, **FIXED), val_size=0.25, epochs=1)
    )
    model.collect()
    model.train()
    assert sorted(RecordingDLHead.seen["train"]) == list(range(0, 8))
    assert sorted(RecordingDLHead.seen["val"]) == [9, 10]


class FoldRecordingMLHead(RecordingMLHead):
    """Keeps every fold's fitted bars, in fold order."""

    folds: list = []

    def _fit_model(self, train_x, train_y, val_x, val_y):
        FoldRecordingMLHead.folds.append(bars(train_y))


def cv_manifest(tmp_path) -> dict:
    (project,) = [p for p in (tmp_path / "models").iterdir() if p.is_dir()]
    return json.loads((project / "cv_folds.json").read_text())


def test_walk_forward_folds_purge_and_record_the_purged_training_end(tmp_path):
    # 20 bars, train_periods 10: test 2 bars, 5 folds. Fold i trains on
    # 2i..2i+9 and tests on 2i+10..2i+11; L = 2 leaves 2i..2i+7 to fit.
    FoldRecordingMLHead.folds = []
    model = FoldRecordingMLHead(MLConfig(**common(tmp_path, 2), val_size=0.0))
    model.collect()
    model.train_cv(train_periods=10)

    assert FoldRecordingMLHead.folds == [
        list(range(2 * i, 2 * i + 8)) for i in range(5)
    ]
    folds = cv_manifest(tmp_path)["folds"]
    keys = ("train_start", "train_end", "test_start", "test_end")
    assert [tuple(np.datetime64(f[k], "D") for k in keys) for f in folds] == [
        (TIMES[2 * i], TIMES[2 * i + 7], TIMES[2 * i + 10], TIMES[2 * i + 11])
        for i in range(5)
    ]
    assert all("gap_periods" not in f for f in folds)


def test_train_cv_takes_no_gap(tmp_path):
    model = FoldRecordingMLHead(MLConfig(**common(tmp_path, 2), val_size=0.0))
    model.collect()
    with pytest.raises(TypeError, match="gap_periods"):
        model.train_cv(train_periods=10, gap_periods=2)
