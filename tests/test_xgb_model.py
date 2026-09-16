"""Tests for `quantlab/ml_model/xgb.py:XGBoostRegressor` (quick task 260914-lno).

What is locked, and what turns it red:

- it LEARNS: on a panel whose label is driven by one factor, the test-set
  cross-sectional IC exceeds 0.5; on a control panel whose label is unrelated
  to every factor, |IC| stays under 0.2 (so the first number is not an
  artefact of the metric or the split);
- native early stopping: the Booster read back FROM DISK has exactly
  `best_iteration + 1` trees, fewer than `num_boost_round` on a noise label;
  with early stopping off it has exactly `num_boost_round`; with no usable
  validation segment it trains every round and warns;
- the per-round W&B callback sees the round that triggered the stop, which
  pins it BEFORE `EarlyStopping` in the callback list (measured on xgboost
  3.4.1: placed after, it misses that round);
- hyperparameters pass straight through (`nthread` included), except
  `num_boost_round`, and the config's dict is never mutated;
- multi-label output, NaN labels, ±inf features, persistence;
- sequential and parallel `train_cv`: same folds, native early stopping inside
  every fold, a correct `cv_mean_test_ic`, matching predictions, and `nthread`
  left exactly as the user set it.

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import xarray as xr
import xgboost as xgb
from loguru import logger

from quantlab.base.config import DLConfig, MLConfig
from quantlab.base.model import BaseModel, MLModel
from quantlab.ml_model.xgb import (
    XGBoostRegressor,
    ccc_loss_metric,
    pooled_ccc_loss,
)
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
N_TRAIN_TIMES = 120
#: `val_size=0.2` on 120 training timestamps: rows 96..119 validate.
VAL_START_IDX = 96


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
        hyperparameters=hyperparameters if hyperparameters is not None else {"num_boost_round": 20},
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


def _test_arrays(model: XGBoostRegressor) -> tuple[np.ndarray, np.ndarray]:
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


def _train(tmp_path, factors, labels, **kwargs) -> XGBoostRegressor:
    model = XGBoostRegressor(_config(tmp_path, factors, labels, **kwargs))
    model.collect()
    model.train()
    return model


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------


def test_learns_a_factor_driven_label(tmp_path, recorders):
    """Label = 0.1*f_signal + 0.05*noise (true correlation ~0.89). Turns red if
    features and labels are misaligned (wrong axis order, a shifted split,
    predictions reshaped in the wrong order)."""
    factors, labels = _panels(seed=11)
    model = _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        patience=20,
        hyperparameters={"num_boost_round": 300, "max_depth": 3, "eta": 0.1},
    )
    test_x, test_y = _test_arrays(model)

    pred = model.predict(test_x)

    assert pred.shape == (N_TIMES - N_TRAIN_TIMES, N_SYMBOLS, 1)
    assert regression_panel_metrics(pred[..., 0], test_y[..., 0])["ic"] > 0.5
    assert recorders[0].summary["test_ic"] > 0.5


def test_unrelated_label_gives_no_ic(tmp_path, recorders):
    """The control: same pipeline, label independent of every factor."""
    factors, labels = _panels(seed=12, signal=False)
    model = _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        patience=20,
        hyperparameters={"num_boost_round": 300, "max_depth": 3, "eta": 0.1},
    )
    test_x, test_y = _test_arrays(model)

    ic = regression_panel_metrics(model.predict(test_x)[..., 0], test_y[..., 0])["ic"]

    assert abs(ic) < 0.2


# --------------------------------------------------------------------------
# Native early stopping
# --------------------------------------------------------------------------


def test_early_stopping_saves_a_truncated_booster_and_logs_the_stopping_round(tmp_path, recorders):
    """Noise label, 500 rounds, patience 10: the Booster ON DISK stops well
    short of 500 and holds exactly `best_iteration + 1` trees.

    The per-round log runs from step 0 with no gap to step
    `best_iteration + patience` -- the round that triggered the stop (measured
    on xgboost 3.4.1 in this task's probe). A callback placed after
    `EarlyStopping` misses that last round, so this also pins callback order.
    """
    patience = 10
    factors, labels = _panels(seed=13, signal=False)
    _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        patience=patience,
        hyperparameters={"num_boost_round": 500},
    )

    booster = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    best = booster.best_iteration
    assert booster.num_boosted_rounds() < 500
    assert booster.num_boosted_rounds() == best + 1

    rec = recorders[0]
    assert rec.summary["best_iteration"] == best
    assert isinstance(rec.summary["best_score"], float)

    steps = [step for _, step in rec.logs]
    assert steps == list(range(len(steps)))
    assert steps[-1] == best + patience
    assert all({"train-rmse", "val-rmse"} <= set(row) for row, _ in rec.logs)


def test_early_stopping_off_trains_every_round(tmp_path, recorders):
    factors, labels = _panels(seed=14, signal=False)
    _train(tmp_path, factors, labels, early_stopping=False, hyperparameters={"num_boost_round": 25})

    booster = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    assert booster.num_boosted_rounds() == 25
    assert "best_iteration" not in recorders[0].summary
    assert len(recorders[0].logs) == 25


def test_early_stopping_without_a_validation_segment_trains_every_round_and_warns(
    tmp_path, recorders, warnings_log
):
    factors, labels = _panels(seed=15, signal=False)
    _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        val_size=0.0,
        hyperparameters={"num_boost_round": 15},
    )

    booster = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    assert booster.num_boosted_rounds() == 15
    assert any("early stopping skipped" in m for m in warnings_log)
    rec = recorders[0]
    assert not any(key.startswith("val-") for row, _ in rec.logs for key in row)
    assert not any(key.startswith("val_") for key in rec.summary)
    assert "best_iteration" not in rec.summary


def test_an_all_nan_validation_label_counts_as_no_validation_segment(tmp_path, recorders, warnings_log):
    """The validation segment has timestamps but no finite label: it must not
    be handed to xgboost as an empty DMatrix, and early stopping must not arm
    on it."""
    factors, labels = _panels(seed=16, signal=False)
    labels._ds["ret_a"][VAL_START_IDX:N_TRAIN_TIMES] = np.nan
    _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        hyperparameters={"num_boost_round": 15},
    )

    booster = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    assert booster.num_boosted_rounds() == 15
    assert any("no rows with finite labels" in m for m in warnings_log)


# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------


def test_hyperparameters_pass_through(tmp_path, recorders):
    """User keys reach xgboost verbatim -- `nthread` included -- while
    `num_boost_round` is consumed as the round budget and never becomes a
    param. The config's own dict is left untouched."""
    hyper = {"num_boost_round": 7, "max_depth": 2, "eval_metric": "mae", "nthread": 1}
    factors, labels = _panels(seed=17)
    model = _train(tmp_path, factors, labels, hyperparameters=dict(hyper))

    booster = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    saved = json.loads(booster.save_config())
    assert booster.num_boosted_rounds() == 7
    assert saved["learner"]["gradient_booster"]["tree_train_param"]["max_depth"] == "2"
    assert saved["learner"]["generic_param"]["nthread"] == "1"
    assert set(recorders[0].logs[0][0]) == {
        "train-mae",
        "val-mae",
        "train-ccc_loss",
        "val-ccc_loss",
    }
    assert "num_boost_round" not in model._params
    assert model.config.hyperparameters == hyper


def test_default_params(tmp_path, recorders):
    """Unset `seed` follows `config.random_seed`, unset `eval_metric` is
    RMSE, and unset `nthread` is left out entirely so xgboost picks its own
    default."""
    factors, labels = _panels(seed=18)
    model = _train(tmp_path, factors, labels, hyperparameters={"num_boost_round": 3})
    assert model._params["seed"] == model.config.random_seed
    assert model._params["eval_metric"] == "rmse"
    assert "nthread" not in model._params


def test_num_boost_round_must_be_positive(tmp_path):
    factors, labels = _panels(seed=19)
    model = XGBoostRegressor(_config(tmp_path, factors, labels, hyperparameters={"num_boost_round": 0}))
    with pytest.raises(ValueError, match="num_boost_round"):
        model._init_model(num_features=3, num_labels=1, hyperparameters={"num_boost_round": 0})


# --------------------------------------------------------------------------
# Multi-label, NaN / inf, persistence, config type
# --------------------------------------------------------------------------


def test_multi_label_predicts_every_label(tmp_path, recorders):
    """Second label = -0.1*f_second + noise: its own column must carry its
    own signal, so a head that only fits label 0 turns red."""
    factors, labels = _panels(seed=20, second_label=True)
    model = _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        patience=20,
        hyperparameters={"num_boost_round": 300, "max_depth": 3, "eta": 0.1},
    )
    test_x, test_y = _test_arrays(model)

    pred = model.predict(test_x)

    assert pred.shape == (N_TIMES - N_TRAIN_TIMES, N_SYMBOLS, 2)
    assert regression_panel_metrics(pred[..., 1], test_y[..., 1])["ic"] > 0.5


def test_nan_labels_and_infinite_features(tmp_path, recorders):
    """10% NaN label cells and a sprinkling of ±inf feature cells: training
    survives (xgboost itself raises on inf), NaN-label rows are dropped, and
    test predictions are all finite."""
    rng = np.random.default_rng(21)
    factors, labels = _panels(seed=21)
    label_values = labels._ds["ret_a"].values
    label_values[rng.random(label_values.shape) < 0.1] = np.nan
    feature_values = factors._ds["f_noise"].values
    idx = rng.choice(feature_values.size, size=40, replace=False)
    feature_values.reshape(-1)[idx[:20]] = np.inf
    feature_values.reshape(-1)[idx[20:]] = -np.inf

    model = _train(tmp_path, factors, labels, hyperparameters={"num_boost_round": 20})
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

    x_rows, y_rows = XGBoostRegressor._to_rows(x_pre, model._preprocess(y))
    assert x_rows.shape[0] == y_rows.shape[0] == int(np.isfinite(y).all(axis=-1).sum())
    assert np.isfinite(y_rows).all()


def test_fresh_instance_loads_and_predicts_identically(tmp_path, recorders):
    factors, labels = _panels(seed=22)
    trained = _train(tmp_path, factors, labels, hyperparameters={"num_boost_round": 20})
    test_x, _ = _test_arrays(trained)
    fresh_factors, fresh_labels = _panels(seed=22)
    fresh = XGBoostRegressor(_config(tmp_path, fresh_factors, fresh_labels, save_dir="unused"))

    fresh.load(_only_checkpoint(tmp_path / "ckpt"))

    assert np.array_equal(fresh.predict(test_x), trained.predict(test_x))


def test_rejects_a_dl_config(tmp_path):
    factors, labels = _panels(seed=23)
    kwargs = dict(
        factors=[factors],
        labels=[labels],
        model_save_dir=str(tmp_path),
        factor_data_strategy="cal",
        label_data_strategy="cal",
    )
    with pytest.raises(TypeError, match="XGBoostRegressor requires a MLConfig"):
        XGBoostRegressor(DLConfig(**kwargs))


def test_contract():
    assert XGBoostRegressor.__abstractmethods__ == frozenset()
    assert issubclass(XGBoostRegressor, MLModel)


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

CV_HYPER = {"num_boost_round": 60, "max_depth": 3, "nthread": 1}


def _cv_model(tmp_path, save_dir) -> XGBoostRegressor:
    factors, labels = _panels(seed=31)
    model = XGBoostRegressor(
        _config(
            tmp_path,
            factors,
            labels,
            save_dir=save_dir,
            early_stopping=True,
            patience=10,
            hyperparameters=dict(CV_HYPER),
        )
    )
    model.collect()
    return model


def _fold_key(result: dict) -> str:
    return result["experiment_name"]


def test_train_cv_sequential(tmp_path, recorders):
    """160 timestamps, train 60 / gap 2 -> 8 folds. Every fold runs native
    early stopping (its Booster holds `best_iteration + 1` trees), carries a
    finite `test_ic`, and loads into a fresh instance; the summary run's
    `cv_mean_test_ic` is the fold mean and reflects the signal."""
    model = _cv_model(tmp_path, "ckpt_seq")
    timestamps = model.data_backend.get_xarray_dataset(["timestamp", "symbol"]).timestamp.values

    results = model.train_cv(train_periods=60, gap_periods=2)

    assert len(results) == 8
    expected = XGBoostRegressor._cv_folds(timestamps, 60, 2)
    assert [
        {k: r[k] for k in ("fold", "train_start", "train_end", "test_start", "test_end")} for r in results
    ] == expected
    for r in results:
        ckpt = Path(r["checkpoint"])
        assert {p.name for p in ckpt.parent.iterdir()} == {ckpt.name, "config.json"}
        booster = joblib.load(ckpt)
        assert booster.num_boosted_rounds() <= 60
        assert booster.num_boosted_rounds() == booster.best_iteration + 1
        assert np.isfinite(r["test_ic"])
        fresh_factors, fresh_labels = _panels(seed=31)
        fresh = XGBoostRegressor(_config(tmp_path, fresh_factors, fresh_labels, save_dir="unused")).load(ckpt)
        assert fresh.predict(np.zeros((4, N_SYMBOLS, 3), dtype=np.float32)).shape == (4, N_SYMBOLS, 1)

    summary_run = recorders[-1]
    assert summary_run.name == "XGBoostRegressor_cv_summary"
    assert summary_run.summary["cv_n_folds"] == 8
    assert summary_run.summary["cv_mean_test_ic"] == pytest.approx(float(np.mean([r["test_ic"] for r in results])))
    assert summary_run.summary["cv_mean_test_ic"] > 0.3


def test_train_cv_parallel_matches_sequential(tmp_path, recorders):
    """`parallel=True, njobs=2` with `nthread=1`: the same fold checkpoints,
    predictions that agree fold by fold, and `nthread` still exactly 1 --
    the class never rewrites it."""
    seq = _cv_model(tmp_path, "ckpt_seq")
    seq_results = seq.train_cv(train_periods=60, gap_periods=2)
    par = _cv_model(tmp_path, "ckpt_par")
    par_results = par.train_cv(train_periods=60, gap_periods=2, parallel=True, njobs=2)

    def relative(results, root):
        return {str(Path(*Path(r["checkpoint"]).relative_to(root).parts[1:])) for r in results}

    assert relative(par_results, tmp_path / "ckpt_par") == relative(seq_results, tmp_path / "ckpt_seq")

    probe = np.random.default_rng(0).standard_normal((5, N_SYMBOLS, 3)).astype(np.float32)
    par_by_fold = {_fold_key(r): r for r in par_results}

    def loaded(checkpoint):
        fresh_factors, fresh_labels = _panels(seed=31)
        return XGBoostRegressor(_config(tmp_path, fresh_factors, fresh_labels, save_dir="unused")).load(checkpoint)

    for r in seq_results:
        seq_pred = loaded(r["checkpoint"]).predict(probe)
        par_pred = loaded(par_by_fold[_fold_key(r)]["checkpoint"]).predict(probe)
        assert np.allclose(seq_pred, par_pred, atol=1e-6)
    assert par.config.hyperparameters["nthread"] == 1


# --------------------------------------------------------------------------
# sklearn-style aliases (scope addition approved 2026-09-14)
# --------------------------------------------------------------------------
#
# Measured on xgboost 3.4.1 with a plain `{**DEFAULT_PARAMS, **user}` merge:
# `learning_rate` beside the default `eta` wins only by dict order,
# `n_estimators` is ignored with a "not used" warning, and `random_state`
# beside `seed` is ignored SILENTLY. Every assertion below reads the TRAINED
# booster, because a params dict that looks right proves nothing about what
# xgboost used.


def _booster_config(tmp_path) -> dict:
    return json.loads(joblib.load(_only_checkpoint(tmp_path / "ckpt")).save_config())["learner"]


def test_sklearn_aliases_reach_the_trained_booster(tmp_path, recorders):
    """Each alias overrides the matching default or config value: rounds from
    `n_estimators`, `eta` from `learning_rate` (default 0.05), `seed` from
    `random_state` (config.random_seed is 42), `nthread` from `n_jobs`, and
    the two regularisers. Turns red if aliases are merged after the defaults,
    passed through raw, or normalised on the wrong dict."""
    hyper = {
        "n_estimators": 9,
        "learning_rate": 0.3,
        "random_state": 7,
        "n_jobs": 1,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    }
    factors, labels = _panels(seed=41)
    model = _train(tmp_path, factors, labels, hyperparameters=dict(hyper))

    booster = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    learner = json.loads(booster.save_config())["learner"]
    tree = learner["gradient_booster"]["tree_train_param"]
    assert booster.num_boosted_rounds() == 9
    assert float(tree["eta"]) == pytest.approx(0.3)
    assert float(tree["alpha"]) == pytest.approx(0.5)
    assert float(tree["lambda"]) == pytest.approx(2.0)
    assert learner["generic_param"]["seed"] == "7"
    assert learner["generic_param"]["nthread"] == "1"
    assert model.config.hyperparameters == hyper, "config.hyperparameters was mutated"


@pytest.mark.parametrize(
    "alias, canonical, value",
    [
        ("n_estimators", "num_boost_round", 5),
        ("learning_rate", "eta", 0.1),
        ("random_state", "seed", 1),
        ("n_jobs", "nthread", 1),
        ("reg_alpha", "alpha", 0.1),
        ("reg_lambda", "lambda", 0.1),
    ],
)
def test_alias_and_canonical_key_together_raise(tmp_path, recorders, alias, canonical, value):
    """Both spellings of one parameter is ambiguous; picking one silently is
    exactly the dict-order accident this guards against."""
    hyper = {alias: value, canonical: value}
    factors, labels = _panels(seed=42)
    model = XGBoostRegressor(_config(tmp_path, factors, labels, hyperparameters=dict(hyper)))
    model.collect()

    with pytest.raises(ValueError, match=rf"'{alias}'.*'{canonical}'"):
        model.train()
    assert model.config.hyperparameters == hyper


# --------------------------------------------------------------------------
# Resolved hyperparameters are recorded (scope addition approved 2026-09-14)
# --------------------------------------------------------------------------


def test_resolved_hyperparameters_are_written_to_config_json_and_wandb(tmp_path, recorders):
    """The record holds what xgboost actually trained with -- every default,
    the user's overrides, alias-resolved keys and the round count -- so a run
    stays reproducible if `DEFAULT_PARAMS` changes later. The user-level
    `hyperparameters` stays exactly what the user passed."""
    hyper = {"learning_rate": 0.3, "max_depth": 2, "num_boost_round": 4}
    factors, labels = _panels(seed=43)
    _train(tmp_path, factors, labels, hyperparameters=dict(hyper))

    saved = json.loads((_only_checkpoint(tmp_path / "ckpt").parent / "config.json").read_text())
    resolved = saved["resolved_hyperparameters"]
    expected = {
        **XGBoostRegressor.DEFAULT_PARAMS,
        "seed": 42,
        "eta": 0.3,
        "max_depth": 2,
        "num_boost_round": 4,
    }
    assert resolved == expected
    assert "learning_rate" not in resolved
    assert saved["hyperparameters"] == hyper

    rec = recorders[0]
    assert rec.config["resolved_hyperparameters"] == expected
    assert rec.config.allow_val_change is True


# --------------------------------------------------------------------------
# Feature importance in the W&B summary (03.7-18, G-03.7-9 addition)
# --------------------------------------------------------------------------
#
# Importance comes from a NAMELESS Booster: `get_score` keys are `f{i}`, where
# `i` is the column index `_fit_model` built in `get_factor_names()` order.
# Variable order itself is checked by `BaseModel.load()` (03.7-17), not here.

IMPORTANCE_TYPES = ("weight", "gain", "total_gain")
IMPORTANCE_FACTORS = ["f_signal", "f_second", "f_const"]


def _importance_panels(seed: int):
    """Factors `f_signal`, `f_second` and an all-zero `f_const` that can never
    split; label `ret_a = 0.1*f_signal + 0.05*noise`."""
    rng = np.random.default_rng(seed)
    shape = (N_TIMES, N_SYMBOLS)
    f_signal = rng.standard_normal(shape)
    factors = {
        "f_signal": f_signal,
        "f_second": rng.standard_normal(shape),
        "f_const": np.zeros(shape),
    }
    labels = {"ret_a": 0.1 * f_signal + 0.05 * rng.standard_normal(shape)}
    return ArrayPanel(factors), ArrayPanel(labels)


def _importance_summary(recorder: FakeRecorder) -> dict:
    return {k: v for k, v in recorder.summary.items() if k.startswith("importance_")}


def test_importance_is_written_to_the_summary_for_every_factor(tmp_path, recorders):
    """Every declared factor gets `importance_{weight,gain,total_gain}/{name}`,
    zero-filled when it never split. Turns red if importance is missing, keyed
    by `f{i}` instead of the factor name, mapped to the wrong index (f_const
    would inherit f_signal's gain), or written through `recorder.log`."""
    factors, labels = _importance_panels(seed=51)
    model = _train(tmp_path, factors, labels, hyperparameters={"num_boost_round": 20, "max_depth": 3})
    assert model.get_factor_names() == IMPORTANCE_FACTORS

    importance = _importance_summary(recorders[0])

    assert set(importance) == {f"importance_{t}/{n}" for t in IMPORTANCE_TYPES for n in IMPORTANCE_FACTORS}
    assert all(type(v) is float for v in importance.values()), importance
    assert importance["importance_gain/f_const"] == 0.0
    assert importance["importance_weight/f_const"] == 0.0
    assert importance["importance_total_gain/f_const"] == 0.0
    assert importance["importance_gain/f_signal"] > 0.0
    assert importance["importance_total_gain/f_signal"] > importance["importance_total_gain/f_second"]
    assert not any(key.startswith("importance_") for row, _ in recorders[0].logs for key in row)


def test_importance_with_early_stopping_uses_the_saved_booster(tmp_path, recorders):
    """Noise label with native early stopping: importance describes the
    truncated Booster that lands on disk, and the per-round log still runs
    contiguously with the rmse keys in every row."""
    patience = 10
    factors, labels = _importance_panels(seed=52)
    labels._ds["ret_a"][:] = np.random.default_rng(52).standard_normal((N_TIMES, N_SYMBOLS))
    _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        patience=patience,
        hyperparameters={"num_boost_round": 300},
    )

    booster = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    rec = recorders[0]
    importance = _importance_summary(rec)
    assert set(importance) == {f"importance_{t}/{n}" for t in IMPORTANCE_TYPES for n in IMPORTANCE_FACTORS}
    for importance_type in IMPORTANCE_TYPES:
        on_disk = booster.get_score(importance_type=importance_type)
        for i, name in enumerate(IMPORTANCE_FACTORS):
            assert importance[f"importance_{importance_type}/{name}"] == pytest.approx(
                float(on_disk.get(f"f{i}", 0.0))
            )

    steps = [step for _, step in rec.logs]
    assert booster.num_boosted_rounds() == booster.best_iteration + 1 < 300
    assert steps == list(range(len(steps)))
    assert steps[-1] == booster.best_iteration + patience
    assert all({"train-rmse", "val-rmse"} <= set(row) for row, _ in rec.logs)


def test_training_without_a_recorder_writes_no_importance(tmp_path, monkeypatch):
    """W&B off (no recorder at all): training completes and writes its
    checkpoint; the importance block must not touch a missing recorder."""

    def no_recorder(self, project_name, experiment_name):
        self._wandb_recorder = None

    monkeypatch.setattr(BaseModel, "_init_wandb", no_recorder)
    factors, labels = _importance_panels(seed=53)

    model = _train(tmp_path, factors, labels, hyperparameters={"num_boost_round": 5})

    assert model._wandb_recorder is None
    assert _only_checkpoint(tmp_path / "ckpt").is_file()


@pytest.mark.parametrize("second_label", [False, True], ids=["single-label", "two-label"])
def test_gblinear_trains_and_writes_its_checkpoint_with_a_recorder(tmp_path, recorders, second_label):
    """REVIEW CR-01: `booster="gblinear"` is a valid configuration, but its
    Booster only has `weight` importance (`gain` raises XGBoostError) and, with
    a `[T, S, L>1]` target, `get_score("weight")` returns one list per factor.
    Importance runs inside `_fit_model`, before `_save_model`, so either case
    used to raise and lose the trained model: no `.joblib`, no `finish()`.
    Both ids go red on the pre-fix code."""
    factors, labels = _importance_panels(seed=55)
    if second_label:
        rng = np.random.default_rng(55)
        labels = ArrayPanel(
            {
                "ret_a": labels._ds["ret_a"].values,
                "ret_b": -0.1 * factors._ds["f_second"].values + 0.05 * rng.standard_normal((N_TIMES, N_SYMBOLS)),
            }
        )

    model = _train(tmp_path, factors, labels, hyperparameters={"booster": "gblinear", "num_boost_round": 5})

    checkpoint = _only_checkpoint(tmp_path / "ckpt")
    assert checkpoint.is_file()
    assert (checkpoint.parent / "config.json").is_file()
    rec = recorders[0]
    assert rec.finished == 1
    assert _importance_summary(rec) == {}
    test_x, _ = _test_arrays(model)
    assert model.predict(test_x).shape == (N_TIMES - N_TRAIN_TIMES, N_SYMBOLS, 2 if second_label else 1)


def test_unavailable_or_non_scalar_importance_is_skipped_not_fatal(tmp_path, recorders, warnings_log, monkeypatch):
    """REVIEW CR-01, beyond gblinear: importance is best-effort telemetry. An
    importance type the Booster cannot report (XGBoostError) or reports as a
    per-output list is skipped with a warning; the types that do work are
    still written, and the checkpoint is always saved."""
    real_get_score = xgb.Booster.get_score

    def flaky_get_score(self, fmap="", importance_type="weight"):
        if importance_type == "weight":
            raise xgb.core.XGBoostError("weight unavailable (test)")
        scores = real_get_score(self, fmap=fmap, importance_type=importance_type)
        if importance_type == "gain":
            return {key: [value, value] for key, value in scores.items()}
        return scores

    monkeypatch.setattr(xgb.Booster, "get_score", flaky_get_score)
    factors, labels = _importance_panels(seed=56)

    _train(tmp_path, factors, labels, hyperparameters={"num_boost_round": 10, "max_depth": 3})

    assert _only_checkpoint(tmp_path / "ckpt").is_file()
    importance = _importance_summary(recorders[0])
    assert set(importance) == {f"importance_total_gain/{n}" for n in IMPORTANCE_FACTORS}
    assert any("'weight'" in m for m in warnings_log), warnings_log
    assert any("'gain'" in m for m in warnings_log), warnings_log


def test_booster_stays_nameless_and_predictions_are_unchanged(tmp_path, recorders):
    """Scope lock (user decision 2026-09-15): no `feature_names` reach the
    DMatrix, so the saved Booster is nameless and a fresh instance loads it
    and predicts identically."""
    factors, labels = _importance_panels(seed=54)
    trained = _train(tmp_path, factors, labels, hyperparameters={"num_boost_round": 10})
    test_x, _ = _test_arrays(trained)

    booster = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    assert booster.feature_names is None
    fresh_factors, fresh_labels = _importance_panels(seed=54)
    fresh = XGBoostRegressor(_config(tmp_path, fresh_factors, fresh_labels, save_dir="unused"))
    fresh.load(_only_checkpoint(tmp_path / "ckpt"))
    assert np.array_equal(fresh.predict(test_x), trained.predict(test_x))


# --------------------------------------------------------------------------
# Pooled CCC loss as the early-stopping criterion (scope addition 2026-09-15)
# --------------------------------------------------------------------------
#
# `1 - ccc` on the validation set replaces RMSE as the DECISION metric, while
# rmse stays first in the metric list and therefore stays an observation curve.
# `EarlyStopping(metric_name=None)` resolves to the LAST metric, which is the
# custom one -- so the wiring is what these tests have to prove, not just the
# arithmetic.


def _reference_ccc_loss(y_true, y_pred):
    """The user's own implementation, pasted verbatim as an ORACLE (D-02).

    It lives here and ONLY here: production code is free to be shaped
    differently (D-04) as long as it returns this number on non-degenerate,
    all-finite input. If the shipped helper is ever rewritten, this is the
    fixed point it gets re-checked against.
    """
    mu_true = np.mean(y_true)
    mu_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_pred - mu_pred) * (y_true - mu_true))
    ccc = 2 * cov / (var_pred + var_true + (mu_pred - mu_true) ** 2)
    return 1 - ccc


@pytest.mark.parametrize(
    "seed, n", [(0, 3), (1, 50), (2, 500), (3, 10_000)], ids=["n3", "n50", "n500", "n10k"]
)
def test_pooled_ccc_loss_equals_the_reference(seed, n):
    """Several seeds and sizes: the shipped helper returns the reference's
    number, not merely a number that correlates with it."""
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n)
    y_pred = 0.4 * y_true + 0.6 * rng.standard_normal(n)

    assert pooled_ccc_loss(y_true, y_pred) == pytest.approx(
        _reference_ccc_loss(y_true, y_pred), rel=1e-12
    )


def test_pooled_ccc_loss_equals_the_reference_on_a_heavily_shrunk_prediction():
    """Prediction variance ~1e-6 of the label's -- the regime where CCC and
    RMSE disagree most, because RMSE rewards the shrinkage CCC punishes."""
    rng = np.random.default_rng(4)
    y_true = rng.standard_normal(400)
    y_pred = 0.001 * y_true

    assert pooled_ccc_loss(y_true, y_pred) == pytest.approx(
        _reference_ccc_loss(y_true, y_pred), rel=1e-12
    )


def test_pooled_ccc_loss_equals_the_reference_under_a_large_mean_offset():
    """A big location error feeds the `(mu_pred - mu_true)**2` term; an
    implementation that dropped it would still pass the two tests above."""
    rng = np.random.default_rng(5)
    y_true = rng.standard_normal(400)
    y_pred = y_true + 50.0

    assert pooled_ccc_loss(y_true, y_pred) == pytest.approx(
        _reference_ccc_loss(y_true, y_pred), rel=1e-12
    )


#: Pinned by a search over seeds 10..59 driving this exact harness (noise
#: label, 300 rounds, patience 10): on seed 17 the `val-ccc_loss` argmin is
#: round 65 while the `val-rmse` argmin is round 45 -- a 20-round gap, so the
#: divergence assertion below is not riding on a rounding accident. Many seeds
#: do NOT diverge (13 puts both argmins on round 1), which is why this constant
#: is a measured choice and not an arbitrary one.
CRITERION_SEED = 17
CRITERION_ROUNDS = 300
CRITERION_PATIENCE = 10


def test_early_stopping_selects_the_round_minimising_val_ccc_loss(tmp_path, recorders):
    """`best_iteration` is the argmin of the recorded `val-ccc_loss` series and
    `best_score` is that minimum -- so the custom metric really is what
    `EarlyStopping` watches (T-weq-01).

    The last assertion is what stops this test being vacuous. The seed is
    pinned precisely because on it the `val-rmse` argmin lands on a DIFFERENT
    round: without that arm, a change that silently reverted the criterion to
    RMSE would leave every other assertion here green.
    """
    factors, labels = _panels(seed=CRITERION_SEED, signal=False)
    _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        patience=CRITERION_PATIENCE,
        hyperparameters={"num_boost_round": CRITERION_ROUNDS},
    )

    booster = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    rec = recorders[0]
    ccc_curve = [row["val-ccc_loss"] for row, _ in rec.logs]
    rmse_curve = [row["val-rmse"] for row, _ in rec.logs]

    assert booster.best_iteration == int(np.argmin(ccc_curve))
    assert booster.best_score == pytest.approx(ccc_curve[booster.best_iteration])
    assert int(np.argmin(rmse_curve)) != booster.best_iteration


def test_every_round_records_both_ccc_curves(tmp_path, recorders):
    """`train-ccc_loss` / `val-ccc_loss` reach the W&B rows beside the rmse
    ones, on every round.

    This is the proof that `_WandbEvalCallback` needed ZERO changes: it walks
    `evals_log` generically, so a custom metric shows up by itself. If this
    ever fails, the callback is what to look at -- and any edit to it belongs
    in a summary with a reason, not a silent fix.
    """
    factors, labels = _panels(seed=61)
    _train(tmp_path, factors, labels, hyperparameters={"num_boost_round": 12})

    rec = recorders[0]
    assert len(rec.logs) == 12
    assert all({"train-ccc_loss", "val-ccc_loss"} <= set(row) for row, _ in rec.logs)
    assert all({"train-rmse", "val-rmse"} <= set(row) for row, _ in rec.logs)


def test_a_better_tracking_prediction_scores_a_smaller_loss():
    """Lower is better -- which is why `EarlyStopping` is left at its default
    `maximize=False` and `maximize=True` is never passed."""
    rng = np.random.default_rng(62)
    target = rng.standard_normal(300)
    tracks_well = target + 0.05 * rng.standard_normal(300)
    tracks_poorly = 0.1 * target + 2.0 * rng.standard_normal(300)

    assert pooled_ccc_loss(target, tracks_well) < pooled_ccc_loss(target, tracks_poorly)


def test_a_perfect_prediction_scores_about_zero():
    rng = np.random.default_rng(63)
    target = rng.standard_normal(300)

    assert pooled_ccc_loss(target, target) == pytest.approx(0.0, abs=1e-12)


#: Every degenerate input must land on 1.0 -- the WORST loss. NaN would make
#: each `EarlyStopping` comparison false (NaN compares false against
#: everything) and silently disable early stopping; 0.0 would record a fully
#: degenerate round as the best one. Both fail quietly, hence T-weq-03.
DEGENERATE_CASES = [
    ("constant-prediction", np.array([1.0, 2.0, 3.0, 4.0]), np.full(4, 0.5)),
    ("single-row", np.array([1.0]), np.array([2.0])),
    ("both-constant-and-equal", np.full(3, 3.0), np.full(3, 3.0)),
    ("all-nan-prediction", np.array([1.0, 2.0, 3.0]), np.full(3, np.nan)),
]


@pytest.mark.parametrize(
    "y_true, y_pred",
    [(case[1], case[2]) for case in DEGENERATE_CASES],
    ids=[case[0] for case in DEGENERATE_CASES],
)
def test_degenerate_input_returns_the_worst_loss_and_never_warns(y_true, y_pred):
    """Must not raise and must not emit a numpy RuntimeWarning.

    `np.errstate(..., "raise")` is what makes that second half real: a 0/0 that
    happens to produce NaN cannot slip through as a pass. `quantlab/utils/
    metrics.py` holds itself to the same "no RuntimeWarning on the empty case"
    bar.
    """
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        assert pooled_ccc_loss(y_true, y_pred) == 1.0


def test_the_adapter_scores_column_zero_of_a_two_label_dmatrix():
    """Pins the layout measured in Task 1: `get_label()` hands back
    `(n_rows, n_labels)` ROW-major, so column 0 is the primary label.

    The expected value is computed on `stored`, the labels as the DMatrix
    actually holds them: `xgb.DMatrix` stores labels as **float32**, so a
    float64 array handed in comes back out rounded (measured: the two scores
    differ in the 8th significant digit). The adapter scores what the DMatrix
    carries; only the label side is rounded, because `predt` never round-trips
    through xgboost.

    The last assertion is what makes the orientation load-bearing. Note that
    `reshape(-1, order="F")[:n]` is NOT a wrong reading -- it equals column 0 --
    so the misreading asserted against is the C-order one, which is what a
    reshape to `(n_labels, n_rows)` produces: the first `n` entries of the
    row-major flat vector, interleaving both labels.
    """
    rng = np.random.default_rng(64)
    n = 40
    labels = np.column_stack(
        [rng.standard_normal(n), rng.standard_normal(n) + 10.0]
    )
    predictions = np.column_stack(
        [rng.standard_normal(n), rng.standard_normal(n) - 5.0]
    )
    dmatrix = xgb.DMatrix(rng.standard_normal((n, 3)), label=labels)
    stored = labels.astype(np.float32).astype(np.float64)

    name, value = ccc_loss_metric(predictions, dmatrix)

    assert name == "ccc_loss"
    assert value == pytest.approx(
        _reference_ccc_loss(stored[:, 0], predictions[:, 0]), rel=1e-12
    )
    assert value != pytest.approx(
        _reference_ccc_loss(stored[:, 1], predictions[:, 1])
    )
    assert value != pytest.approx(
        _reference_ccc_loss(
            stored.reshape(-1)[:n], predictions.reshape(-1)[:n]
        )
    )


def test_the_adapter_refuses_a_prediction_of_the_wrong_size():
    """Element counts that disagree are a wiring bug, not something to
    broadcast: a broadcast score looks plausible and is computed from
    misaligned vectors."""
    dmatrix = xgb.DMatrix(np.zeros((4, 2)), label=np.arange(4.0))

    with pytest.raises(ValueError, match="4 label values but the prediction has 3"):
        ccc_loss_metric(np.zeros(3), dmatrix)
