"""`SeedEnsemble.train_cv`, the walk-forward CV of a seed ensemble, and its replay by `run_cv`.

What is locked here, and what turns it red:

- Sliding and expanding ensemble CV lay out, purge and train exactly the
  folds `BaseModel.train_cv` does on the wrapped model with the same data:
  the returned fold dates are equal, and every member of a fold trains on
  that fold's dates.
- Layout: `{model_save_dir}/SeedEnsemble_cv_{timestamp}/` holds
  `cv_folds.json` and one `fold_{i}/` per fold, each a complete ensemble
  directory (`ensemble.json`, `config.json`, `member_{k}/`, `ic_series.csv`,
  `test_predictions.zarr`) that `SeedEnsemble.load` restores. Like a single
  model's fold, a fold writes no ensemble-level `metrics.json`: its metrics
  live in `cv_folds.json`.
- `cv_folds.json` is format version 2: each fold record holds the purged
  fold dates, the absolute path of the fold's `ensemble.json` and the
  ensemble-level IC metrics (no error metric), and `cv_mean` averages them
  the way `BaseModel` does.
- W&B: one run per member per fold, `{MemberClass}_fold_{i}_member_{k}`, all
  in one project named after the CV directory, plus a
  `SeedEnsemble_cv_summary` run carrying the `cv_mean_*` values.
- `run_cv` with the ensemble as `config.model` replays the CV run end to end:
  each fold loads its own `ensemble.json`, its predictions are that fold
  ensemble's average, the checkpoint's recorded training dates agree with
  the manifest, and the stitched curve covers every test bar.
- `train_periods` below 5 raises, and member hyperparameters are checked
  before any directory is created.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig
from quantlab.base.model import BaseModel
from quantlab.model._support.ensemble import BaseEnsemble
from quantlab.model.seed_ensemble import SeedEnsemble
from tests.backtest_fixtures import (
    SeededHead,
    make_model,
    make_stock_dataset,
    write_price_store,
)

N_BARS = 80
TRAIN_PERIODS = 30
#: test_periods = 30 // 5 = 6, (80 - 30) // 6 = 8 folds testing bars 30..77.
TEST_PERIODS = 6
N_FOLDS = 8
FIRST_TEST_BAR = TRAIN_PERIODS
LAST_TEST_BAR = TRAIN_PERIODS + N_FOLDS * TEST_PERIODS - 1
#: A 2-bar forward label: lookahead 3, so every fold's training window is purged.
HORIZON = 2
SEEDS = [0, 1, 2]
DATE_KEYS = ("fold", "train_start", "train_end", "test_start", "test_end")
IC_METRICS = {"ic", "rank_ic", "icir", "rank_icir"}


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _model_dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[N_BARS - 1]),
        train_start=_day(bars[0]),
        train_end=_day(bars[TRAIN_PERIODS - 1]),
        test_start=_day(bars[TRAIN_PERIODS]),
        test_end=_day(bars[N_BARS - 1]),
    )


def _model(root, dataset_config, bars):
    return make_model(
        root, dataset_config, head=SeededHead, n_forward_periods=HORIZON,
        **_model_dates(bars),
    )


def _setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    return dataset_config, bars


def _cv_dir(ensemble) -> Path:
    (found,) = sorted(ensemble.model_save_dir.glob("SeedEnsemble_cv_*"))
    return found


def _dates(results) -> list[tuple]:
    return [tuple(r[key] for key in DATE_KEYS) for r in results]


@pytest.fixture(scope="module")
def cv_run(tmp_path_factory):
    """One sliding ensemble CV run, shared read-only by the module."""
    root = tmp_path_factory.mktemp("ensemble_cv")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WANDB_MODE", "disabled")
        mp.setenv("WANDB_SILENT", "true")
        dataset_config, bars = _setup(root)
        ensemble = SeedEnsemble(_model(root / "train", dataset_config, bars), SEEDS)
        results = ensemble.collect().train_cv(train_periods=TRAIN_PERIODS)
    cv_dir = _cv_dir(ensemble)
    return dict(
        root=root,
        dataset_config=dataset_config,
        bars=bars,
        ensemble=ensemble,
        results=results,
        cv_dir=cv_dir,
        manifest=json.loads((cv_dir / "cv_folds.json").read_text()),
    )


# --------------------------------------------------------------------------
# Fold geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("expanding", [False, True])
def test_fold_dates_equal_the_model_bases_train_cv(tmp_path, expanding, monkeypatch):
    dataset_config, bars = _setup(tmp_path)
    fitted: list[tuple] = []
    real_fit = SeededHead._fit

    def spy_fit(self, checkpoint):
        c = self.config
        fitted.append((c.random_seed, c.train_start, c.train_end, c.test_start, c.test_end))
        return real_fit(self, checkpoint)

    monkeypatch.setattr(SeededHead, "_fit", spy_fit)
    model = _model(tmp_path / "single", dataset_config, bars)
    single = model.collect().train_cv(TRAIN_PERIODS, expanding=expanding)
    single_fits = list(fitted)
    fitted.clear()

    ensemble = SeedEnsemble(_model(tmp_path / "ens", dataset_config, bars), SEEDS)
    results = ensemble.collect().train_cv(TRAIN_PERIODS, expanding=expanding)

    assert len(results) == N_FOLDS
    assert _dates(results) == _dates(single)
    # The purge moved every fold's train_end before the test segment.
    assert all(r["train_end"] < r["test_start"] for r in results)
    # Every member of every fold fitted the dates the single model fitted.
    expected = [
        (seed, *dates[1:]) for dates in single_fits for seed in SEEDS
    ]
    assert fitted == expected
    if expanding:
        assert {r["train_start"] for r in results} == {results[0]["train_start"]}


def test_train_cv_leaves_the_members_on_the_last_folds_dates(cv_run):
    """As `BaseModel.train_cv` leaves its config on the last fold's dates."""
    last = cv_run["manifest"]["folds"][-1]
    for member in cv_run["ensemble"].members:
        assert member.config.test_start == last["test_start"]
        assert member.config.test_end == last["test_end"]


# --------------------------------------------------------------------------
# Layout and the manifest
# --------------------------------------------------------------------------


def test_every_fold_is_a_complete_loadable_ensemble_directory(cv_run, tmp_path):
    cv_dir = cv_run["cv_dir"]
    assert sorted(p.name for p in cv_dir.iterdir()) == sorted(
        ["cv_folds.json"] + [f"fold_{i}" for i in range(N_FOLDS)]
    )
    for i in range(N_FOLDS):
        fold_dir = cv_dir / f"fold_{i}"
        assert sorted(p.name for p in fold_dir.iterdir()) == [
            "config.json",
            "ensemble.json",
            "ic_series.csv",
            "member_0",
            "member_1",
            "member_2",
            "test_predictions.zarr",
        ]
        for k in range(len(SEEDS)):
            assert (fold_dir / f"member_{k}" / f"SeededHead_fold_{i}_member_{k}.joblib").is_file()

    fresh = SeedEnsemble(
        _model(tmp_path, cv_run["dataset_config"], cv_run["bars"]), SEEDS
    )
    fresh.load(cv_dir / "fold_3" / "ensemble.json")
    assert all(member.model is not None for member in fresh.members)


def test_cv_folds_json_is_v2_with_ensemble_metrics_and_cv_mean(cv_run):
    manifest, results, cv_dir = cv_run["manifest"], cv_run["results"], cv_run["cv_dir"]

    assert manifest["format_version"] == BaseModel.CV_FOLDS_FORMAT_VERSION == 2
    assert [f["fold"] for f in manifest["folds"]] == list(range(N_FOLDS))
    for i, (entry, result) in enumerate(zip(manifest["folds"], results)):
        checkpoint = Path(entry["checkpoint"])
        assert checkpoint.is_absolute()
        assert checkpoint == cv_dir / f"fold_{i}" / "ensemble.json"
        assert result["checkpoint"] == entry["checkpoint"]
        metrics = set(entry) - set(DATE_KEYS) - {"checkpoint"}
        assert metrics == {f"{s}_{m}" for s in ("train", "test") for m in IC_METRICS}
        assert tuple(entry[k] for k in DATE_KEYS) == tuple(result[k] for k in DATE_KEYS)

    assert manifest["cv_mean"]["cv_n_folds"] == N_FOLDS
    expected = BaseModel._cv_mean_metrics(results)
    assert set(manifest["cv_mean"]) == set(expected)
    for key, value in expected.items():
        assert manifest["cv_mean"][key] == pytest.approx(value)


def test_fold_config_json_records_the_dates_before_the_purge(cv_run):
    """Like a single model's fold checkpoint, which `run_cv` purges itself."""
    folds = cv_run["manifest"]["folds"]
    bars = [_day(b) for b in cv_run["bars"]]
    for entry in folds:
        saved = json.loads(
            (cv_run["cv_dir"] / f"fold_{entry['fold']}" / "config.json").read_text()
        )
        assert saved["test_start"] == entry["test_start"]
        # The recorded train_end is the bar right before the test segment.
        assert bars.index(_day(saved["train_end"])) == bars.index(
            _day(entry["test_start"])
        ) - 1


# --------------------------------------------------------------------------
# W&B
# --------------------------------------------------------------------------


class FakeRecorder:
    def __init__(self, project: str, name: str):
        self.project = project
        self.name = name
        self.summary: dict = {}
        self.finished = 0

    def log(self, data, step=None):
        pass

    def finish(self):
        self.finished += 1


def test_wandb_runs_per_fold_member_and_one_summary(tmp_path, monkeypatch):
    created: list[FakeRecorder] = []

    def fake_init_wandb(self, project_name, experiment_name):
        self._wandb_recorder = FakeRecorder(project_name, experiment_name)
        created.append(self._wandb_recorder)

    monkeypatch.setattr(BaseModel, "_init_wandb", fake_init_wandb)
    monkeypatch.setattr(BaseEnsemble, "_init_wandb", fake_init_wandb)
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS)

    results = ensemble.collect().train_cv(TRAIN_PERIODS)

    project = _cv_dir(ensemble).name
    assert {r.project for r in created} == {project}
    assert [r.name for r in created] == [
        f"SeededHead_fold_{i}_member_{k}"
        for i in range(N_FOLDS)
        for k in range(len(SEEDS))
    ] + ["SeedEnsemble_cv_summary"]
    summary = created[-1]
    assert summary.finished == 1
    assert summary.summary["cv_n_folds"] == N_FOLDS
    for key, value in BaseModel._cv_mean_metrics(results).items():
        assert summary.summary[key] == pytest.approx(value, nan_ok=True)


def test_train_keeps_the_member_run_names(tmp_path, monkeypatch):
    names: list[str] = []

    def fake_init_wandb(self, project_name, experiment_name):
        names.append(experiment_name)
        self._wandb_recorder = FakeRecorder(project_name, experiment_name)

    monkeypatch.setattr(BaseModel, "_init_wandb", fake_init_wandb)
    dataset_config, bars = _setup(tmp_path)
    SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS).collect().train()

    assert names == [f"SeededHead_member_{k}" for k in range(len(SEEDS))]


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_train_periods_below_5_raises(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS).collect()

    with pytest.raises(ValueError, match="train_periods=4"):
        ensemble.train_cv(4)
    assert not ensemble.model_save_dir.exists()


def test_hyperparameters_are_checked_before_any_directory(tmp_path, monkeypatch):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS).collect()
    checked: list[int] = []

    def refuse(self):
        checked.append(self.config.random_seed)
        if self.config.random_seed == 2:
            raise ValueError("bad hyperparameter")

    monkeypatch.setattr(SeededHead, "_check_hyperparameters", refuse)
    with pytest.raises(ValueError, match="bad hyperparameter"):
        ensemble.train_cv(TRAIN_PERIODS)
    assert checked == SEEDS
    assert not ensemble.model_save_dir.exists()


def test_an_empty_date_range_raises(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS).collect()
    for member in ensemble.members:
        member.config = dataclasses.replace(
            member.config, start_date="2030-01-01", end_date="2030-12-31"
        )

    with pytest.raises(ValueError, match="No data found"):
        ensemble.train_cv(TRAIN_PERIODS)
    assert not ensemble.model_save_dir.exists()


# --------------------------------------------------------------------------
# run_cv replays the ensemble CV run
# --------------------------------------------------------------------------


def test_run_cv_replays_the_ensemble_cv_run(cv_run, tmp_path, monkeypatch):
    bars = cv_run["bars"]
    dataset_config = cv_run["dataset_config"]
    ensemble = SeedEnsemble(_model(tmp_path / "bt", dataset_config, bars), SEEDS)
    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=ensemble,
            model_mode="load",
            cv_project_dir=str(cv_run["cv_dir"]),
            start_date=_day(bars[FIRST_TEST_BAR]),
            end_date=_day(bars[LAST_TEST_BAR]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=2,
            direction="long_only",
            top_n=2,
            fees=0.0,
            slippage=0.0,
            init_cash=1_000_000.0,
        )
    )
    loaded: list[str] = []
    real_load = ensemble.load
    monkeypatch.setattr(ensemble, "load", lambda p: loaded.append(str(p)) or real_load(p))
    warnings: list[str] = []
    handler = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        result = backtester.run_cv()
    finally:
        logger.remove(handler)

    assert loaded == [f["checkpoint"] for f in cv_run["manifest"]["folds"]]
    assert not [w for w in warnings if "records training dates" in w]
    assert [r["fold"] for r in result.folds] == list(range(N_FOLDS))
    np.testing.assert_array_equal(
        result.weights.timestamp.values.astype("datetime64[ns]"),
        bars[FIRST_TEST_BAR : LAST_TEST_BAR + 1].astype("datetime64[ns]"),
    )
    assert result.metrics["stitched"]["in_sample_ranges"] == []

    # Fold 2 traded the average of fold 2's own members.
    fold = cv_run["manifest"]["folds"][2]
    replay = SeedEnsemble(_model(tmp_path / "replay", dataset_config, bars), SEEDS)
    replay.load(fold["checkpoint"])
    expected = replay.predict_window(fold["test_start"], fold["test_end"])
    xr.testing.assert_allclose(
        result.folds[2]["predictions"].transpose(*expected.dims), expected
    )
