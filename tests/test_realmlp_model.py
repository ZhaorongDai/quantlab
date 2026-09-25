"""Tests for `quantlab/ml_model/realmlp.py:RealMLPRegressor`.

What is locked, and what turns it red:

- it LEARNS: on a panel whose label is driven by one factor, the test-set
  cross-sectional IC exceeds 0.5; on a control panel whose label is unrelated
  to every factor, |IC| stays under 0.2;
- pytabkit's early stopping: on a noise label with a small patience the
  reported `stop_epoch` is below `n_epochs` and reaches the W&B summary;
  with no usable validation segment training still succeeds and warns;
- hyperparameters pass straight through to the estimator, the config's dict
  is never mutated, and the resolved set (defaults, seed, early-stopping
  keys, user overrides) is written to `config.json` and the run config;
- `val_fraction` defaults to 0 so pytabkit never splits the training rows a
  second time;
- multi-label output, NaN labels, ±inf features (imputed to 0, since pytabkit
  refuses NaN), persistence through a fresh instance, the `MLModel` contract
  and sequential `train_cv`.

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import xarray as xr
from loguru import logger
from pytabkit import RealMLP_TD_Regressor

from quantlab.base.config import DLConfig, MLConfig
from quantlab.base.model import BaseModel, MLModel
from quantlab.ml_model.realmlp import RealMLPRegressor
from quantlab.utils.metrics import regression_panel_metrics

N_TIMES = 160
N_SYMBOLS = 30
SYMBOLS = [f"S{i:02d}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)


def _day(i: int) -> str:
    return np.datetime_as_string(TIMES[i], unit="D")


TRAIN_START, TRAIN_END = _day(0), _day(119)
TEST_START, TEST_END = _day(120), _day(N_TIMES - 1)

#: Small and single-threaded so every test fits in a few seconds on CPU.
FAST = {"n_epochs": 12, "n_threads": 1}


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


class ArrayPanel:
    """A stand-in for a factor/label object backed by explicit `[T, S]` arrays."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.names = list(arrays)
        self._ds = xr.Dataset(
            {name: (("timestamp", "symbol"), arr.astype("float32")) for name, arr in arrays.items()},
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
        return {"name": "ArrayPanel", "factor_names": list(self.names)}


def _panels(seed: int, *, signal: bool = True, second_label: bool = False):
    """Factors `f_signal`, `f_second`, `f_noise`; label `ret_a = 0.1*f_signal
    + 0.05*noise` (or pure noise when `signal=False`), and optionally
    `ret_b = -0.1*f_second + 0.05*noise`."""
    rng = np.random.default_rng(seed)
    shape = (N_TIMES, N_SYMBOLS)
    f_signal, f_second, f_noise = (rng.standard_normal(shape) for _ in range(3))
    factors = {"f_signal": f_signal, "f_second": f_second, "f_noise": f_noise}
    if signal:
        labels = {"ret_a": 0.1 * f_signal + 0.05 * rng.standard_normal(shape)}
    else:
        labels = {"ret_a": rng.standard_normal(shape)}
    if second_label:
        labels["ret_b"] = -0.1 * f_second + 0.05 * rng.standard_normal(shape)
    return ArrayPanel(factors), ArrayPanel(labels)


def _config(
    tmp_path: Path,
    factors,
    labels,
    *,
    save_dir: str = "ckpt",
    early_stopping: bool = False,
    patience: int = 5,
    hyperparameters: dict | None = None,
    val_size: float = 0.2,
) -> MLConfig:
    return MLConfig(
        factors=[factors],
        labels=[labels],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=TRAIN_START,
        end_date=TEST_END,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        early_stopping=early_stopping,
        early_stopping_patience=patience,
        hyperparameters=hyperparameters if hyperparameters is not None else dict(FAST),
        val_size=val_size,
    )


class FakeRunConfig(dict):
    """Mimics `wandb.Run.config.update(d, allow_val_change=...)`."""

    def update(self, data=(), allow_val_change=False, **kwargs):
        self.allow_val_change = allow_val_change
        super().update(data, **kwargs)


class FakeRecorder:
    def __init__(self, name: str):
        self.name = name
        self.logs: list[tuple[dict, int | None]] = []
        self.summary: dict = {}
        self.config = FakeRunConfig()
        self.finished = 0

    def log(self, data, step=None):
        self.logs.append((dict(data), step))

    def finish(self):
        self.finished += 1


@pytest.fixture
def recorders(monkeypatch) -> list[FakeRecorder]:
    """Patched on the class so deep-copied CV folds record too."""
    created: list[FakeRecorder] = []

    def fake_init_wandb(self, project_name, experiment_name):
        recorder = FakeRecorder(experiment_name)
        created.append(recorder)
        self._wandb_recorder = recorder

    monkeypatch.setattr(BaseModel, "_init_wandb", fake_init_wandb)
    return created


@pytest.fixture
def warnings_log():
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(handler_id)


def _test_arrays(model: RealMLPRegressor) -> tuple[np.ndarray, np.ndarray]:
    data = model.data_backend.get_xarray_dataset(["timestamp", "symbol"]).sel(
        timestamp=slice(TEST_START, TEST_END)
    )
    return (
        model.to_array(data, model.get_factor_names()),
        model.to_array(data, model.get_label_names()),
    )


def _only_checkpoint(root: Path) -> Path:
    found = sorted(root.rglob("*.joblib"))
    assert len(found) == 1, found
    return found[0]


def _train(tmp_path, factors, labels, **kwargs) -> RealMLPRegressor:
    model = RealMLPRegressor(_config(tmp_path, factors, labels, **kwargs))
    model.collect()
    model.train()
    return model


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------


def test_learns_a_factor_driven_label(tmp_path, recorders):
    """Label = 0.1*f_signal + 0.05*noise (true correlation ~0.89)."""
    factors, labels = _panels(seed=11)
    model = _train(tmp_path, factors, labels)
    test_x, test_y = _test_arrays(model)

    pred = model.predict(test_x)
    assert pred.shape == (N_TIMES - 120, N_SYMBOLS, 1)
    metrics = regression_panel_metrics(pred[..., 0], test_y[..., 0])
    assert metrics["ic"] > 0.5
    assert recorders[0].summary["test_ic"] == pytest.approx(metrics["ic"])


def test_unrelated_label_gives_no_ic(tmp_path, recorders):
    factors, labels = _panels(seed=12, signal=False)
    model = _train(tmp_path, factors, labels)
    test_x, test_y = _test_arrays(model)

    pred = model.predict(test_x)
    assert abs(regression_panel_metrics(pred[..., 0], test_y[..., 0])["ic"]) < 0.2


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------


def test_early_stopping_stops_before_n_epochs_and_records_the_epoch(tmp_path, recorders):
    """Noise label, 60 epochs, patience 3: pytabkit stops well short of 60 and
    the stopping epoch reaches the summary."""
    factors, labels = _panels(seed=13, signal=False)
    model = _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        patience=3,
        hyperparameters={**FAST, "n_epochs": 60},
    )

    stop = recorders[0].summary["stop_epoch"]
    assert isinstance(stop, int)
    assert 0 < stop < 60
    assert model.model.fit_params_["stop_epoch"] == {"rmse": stop}
    params = model._resolved_hyperparameters()
    assert params["use_early_stopping"] is True
    assert params["early_stopping_additive_patience"] == 3
    assert params["early_stopping_multiplicative_patience"] == 1.0


def test_early_stopping_off_passes_no_early_stopping_keys(tmp_path, recorders):
    factors, labels = _panels(seed=14, signal=False)
    model = _train(tmp_path, factors, labels)

    params = model._resolved_hyperparameters()
    assert "use_early_stopping" not in params
    assert model.model.get_params()["use_early_stopping"] is None


def test_early_stopping_without_a_validation_segment_trains_and_warns(
    tmp_path, recorders, warnings_log
):
    factors, labels = _panels(seed=15, signal=False)
    model = _train(tmp_path, factors, labels, early_stopping=True, val_size=0.0)

    assert any("early stopping skipped" in m for m in warnings_log)
    rec = recorders[0]
    assert "stop_epoch" not in rec.summary
    assert not any(key.startswith("val_") for key in rec.summary)
    test_x, _ = _test_arrays(model)
    assert np.isfinite(model.predict(test_x)).all()


def test_an_all_nan_validation_label_counts_as_no_validation_segment(tmp_path, recorders, warnings_log):
    factors, labels = _panels(seed=16, signal=False)
    labels._ds["ret_a"].values[96:120] = np.nan
    _train(tmp_path, factors, labels, early_stopping=True)

    assert any("no rows with finite labels" in m for m in warnings_log)
    assert any("early stopping skipped" in m for m in warnings_log)
    assert "stop_epoch" not in recorders[0].summary


# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------


def test_hyperparameters_pass_through_and_the_config_dict_is_not_mutated(tmp_path, recorders):
    hyper = {**FAST, "hidden_sizes": [64, 64], "lr": 0.05, "n_threads": 1}
    before = json.dumps(hyper, sort_keys=True)
    factors, labels = _panels(seed=17)
    model = _train(tmp_path, factors, labels, hyperparameters=hyper)

    fitted = model.model.get_params()
    assert fitted["hidden_sizes"] == [64, 64]
    assert fitted["lr"] == 0.05
    assert fitted["n_epochs"] == FAST["n_epochs"]
    assert fitted["n_threads"] == 1
    assert json.dumps(hyper, sort_keys=True) == before
    assert model.config.hyperparameters is hyper


def test_default_params_pin_val_fraction_to_zero(tmp_path, recorders):
    factors, labels = _panels(seed=18)
    model = _train(tmp_path, factors, labels)

    assert RealMLPRegressor.DEFAULT_PARAMS["val_fraction"] == 0.0
    fitted = model.model.get_params()
    assert fitted["val_fraction"] == 0.0
    assert fitted["device"] == "cpu"
    assert fitted["random_state"] == 42


def test_unknown_hyperparameter_raises_type_error(tmp_path, recorders):
    factors, labels = _panels(seed=19)
    model = RealMLPRegressor(
        _config(tmp_path, factors, labels, hyperparameters={**FAST, "bogus": 1})
    )
    model.collect()
    with pytest.raises(TypeError, match="bogus"):
        model.train()


def test_resolved_hyperparameters_are_written_to_config_json_and_wandb(tmp_path, recorders):
    hyper = {**FAST, "lr": 0.02}
    factors, labels = _panels(seed=20)
    _train(tmp_path, factors, labels, hyperparameters=dict(hyper), early_stopping=True, patience=4)

    saved = json.loads((_only_checkpoint(tmp_path / "ckpt").parent / "config.json").read_text())
    expected = {
        **RealMLPRegressor.DEFAULT_PARAMS,
        "random_state": 42,
        "use_early_stopping": True,
        "early_stopping_additive_patience": 4,
        "early_stopping_multiplicative_patience": 1.0,
        **hyper,
    }
    assert saved["resolved_hyperparameters"] == expected
    assert saved["hyperparameters"] == hyper

    rec = recorders[0]
    assert rec.config["resolved_hyperparameters"] == expected
    assert rec.config.allow_val_change is True


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------


def test_multi_label_predicts_every_label(tmp_path, recorders):
    factors, labels = _panels(seed=21, second_label=True)
    model = _train(tmp_path, factors, labels)
    test_x, test_y = _test_arrays(model)

    pred = model.predict(test_x)
    assert pred.shape == (N_TIMES - 120, N_SYMBOLS, 2)
    assert regression_panel_metrics(pred[..., 0], test_y[..., 0])["ic"] > 0.5
    assert regression_panel_metrics(pred[..., 1], test_y[..., 1])["ic"] > 0.5


def test_nan_labels_and_infinite_features(tmp_path, recorders):
    """10% NaN label cells and a sprinkling of ±inf feature cells: NaN-label
    rows are dropped, non-finite features are imputed to 0 (pytabkit refuses
    NaN), and test predictions are all finite."""
    rng = np.random.default_rng(22)
    factors, labels = _panels(seed=22)
    label_values = labels._ds["ret_a"].values
    label_values[rng.random(label_values.shape) < 0.1] = np.nan
    feature_values = factors._ds["f_noise"].values
    idx = rng.choice(feature_values.size, size=40, replace=False)
    feature_values.reshape(-1)[idx[:20]] = np.inf
    feature_values.reshape(-1)[idx[20:]] = -np.inf

    model = _train(tmp_path, factors, labels)
    test_x, _ = _test_arrays(model)
    assert np.isfinite(model.predict(test_x)).all()

    data = model.data_backend.get_xarray_dataset(["timestamp", "symbol"])
    x = model.to_array(data, model.get_factor_names())
    y = model.to_array(data, model.get_label_names())
    x_before = x.copy()
    x_pre = model._preprocess(x)
    assert np.array_equal(x, x_before, equal_nan=True), "_preprocess mutated its input"
    assert not np.isinf(x_pre).any()
    assert int(np.isnan(x_pre).sum()) - int(np.isnan(x).sum()) == int(np.isinf(x).sum()) == 40

    y_pre = model._preprocess(y)
    assert np.array_equal(np.isnan(y_pre), np.isnan(y)), "label NaN must survive _preprocess"
    x_rows, y_rows = RealMLPRegressor._to_rows(x_pre, y_pre)
    assert x_rows.shape[0] == y_rows.shape[0] == int(np.isfinite(y).all(axis=-1).sum())
    assert np.isfinite(x_rows).all()
    assert np.isfinite(y_rows).all()


def test_fresh_instance_loads_and_predicts_identically(tmp_path, recorders):
    factors, labels = _panels(seed=23)
    trained = _train(tmp_path, factors, labels)
    test_x, _ = _test_arrays(trained)
    fresh_factors, fresh_labels = _panels(seed=23)
    fresh = RealMLPRegressor(_config(tmp_path, fresh_factors, fresh_labels, save_dir="unused"))

    fresh.load(_only_checkpoint(tmp_path / "ckpt"))

    assert isinstance(joblib.load(_only_checkpoint(tmp_path / "ckpt")), RealMLP_TD_Regressor)
    assert np.array_equal(fresh.predict(test_x), trained.predict(test_x))


def test_same_seed_trains_the_same_model(tmp_path, recorders):
    first = _train(tmp_path, *_panels(seed=24), save_dir="a")
    second = _train(tmp_path, *_panels(seed=24), save_dir="b")
    test_x, _ = _test_arrays(first)
    assert np.array_equal(first.predict(test_x), second.predict(test_x))


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_rejects_a_dl_config(tmp_path):
    factors, labels = _panels(seed=25)
    kwargs = dict(
        factors=[factors],
        labels=[labels],
        model_save_dir=str(tmp_path),
        factor_data_strategy="cal",
        label_data_strategy="cal",
    )
    with pytest.raises(TypeError, match="RealMLPRegressor requires a MLConfig"):
        RealMLPRegressor(DLConfig(**kwargs))


def test_contract():
    assert RealMLPRegressor.__abstractmethods__ == frozenset()
    assert issubclass(RealMLPRegressor, MLModel)
    assert RealMLPRegressor.config_cls is MLConfig
    assert RealMLPRegressor.checkpoint_suffix == ".joblib"


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------


def test_train_cv_sequential(tmp_path, recorders):
    """160 timestamps, train 60 / gap 2 -> 8 folds. Every fold early-stops on
    its own validation segment, writes one `.joblib`, carries a finite
    `test_ic` and loads into a fresh instance; the summary run's
    `cv_mean_test_ic` is the fold mean and reflects the signal."""
    factors, labels = _panels(seed=31)
    model = RealMLPRegressor(
        _config(tmp_path, factors, labels, save_dir="ckpt_seq", early_stopping=True, patience=4)
    )
    model.collect()
    timestamps = model.data_backend.get_xarray_dataset(["timestamp", "symbol"]).timestamp.values

    results = model.train_cv(train_periods=60, gap_periods=2)

    assert len(results) == 8
    expected = RealMLPRegressor._cv_folds(timestamps, 60, 2)
    assert [
        {k: r[k] for k in ("fold", "train_start", "train_end", "test_start", "test_end")} for r in results
    ] == expected
    for r in results:
        ckpt = Path(r["checkpoint"])
        assert ckpt.suffix == ".joblib"
        assert {p.name for p in ckpt.parent.iterdir()} == {ckpt.name, "config.json"}
        assert isinstance(joblib.load(ckpt), RealMLP_TD_Regressor)
        assert np.isfinite(r["test_ic"])
        fresh = RealMLPRegressor(_config(tmp_path, *_panels(seed=31), save_dir="unused")).load(ckpt)
        assert fresh.predict(np.zeros((4, N_SYMBOLS, 3), dtype=np.float32)).shape == (4, N_SYMBOLS, 1)

    fold_runs = [rec for rec in recorders if rec.name != "RealMLPRegressor_cv_summary"]
    assert len(fold_runs) == 8
    assert all("stop_epoch" in rec.summary for rec in fold_runs)

    summary_run = recorders[-1]
    assert summary_run.name == "RealMLPRegressor_cv_summary"
    assert summary_run.summary["cv_n_folds"] == 8
    assert summary_run.summary["cv_mean_test_ic"] == pytest.approx(float(np.mean([r["test_ic"] for r in results])))
    assert summary_run.summary["cv_mean_test_ic"] > 0.3
