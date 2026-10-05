"""A trained unit records the data its ``collect()`` read; a train-mode rebuild compares it.

``collect()`` opens a ``DataRecorder`` keyed by component path within the
model; ``train()`` and ``train_cv()`` write the record into the ``run.json``
of the unit that read, and ``TrainedRun.data_fingerprint`` exposes it. A
member or a fold holds none. A train-mode backtest rebuilt from its run
compares the retrained unit's record with the unit it used, and only warns;
load mode compares no training data.

Everything is synthetic, CPU-only and offline.
"""

import shutil

import pandas as pd
import pytest
import xarray as xr
import zarr
from loguru import logger

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.trained_run import TrainedRun
from tests.backtest_fixtures import (
    SeededHead,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
MODEL_KEYS = {"factors.0.dataset", "labels.0.factor.dataset"}


def _day(ts) -> str:
    """Return ``ts`` as an ISO day."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@pytest.fixture
def warnings_logged():
    """Collect the loguru warnings emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


def _store(tmp_path):
    """A 60-bar price store and its bars."""
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    return dataset_config, xr.open_zarr(dataset_config.zarr_file_path).timestamp.values


def _dates(bars) -> dict:
    """Train on bars 0..24, test on 25..29."""
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )


def _training_mismatches(messages: list[str]) -> list[str]:
    """The training-data warnings of a train-mode rebuild."""
    return [m for m in messages if "training: data fingerprint mismatch" in m]


def test_a_model_unit_records_its_training_reads(tmp_path):
    dataset_config, bars = _store(tmp_path)
    model = make_model(tmp_path / "m", dataset_config, **_dates(bars))

    unit = TrainedRun.open(train_checkpoint(model))

    assert set(unit.data_fingerprint) == MODEL_KEYS
    assert unit.data_fingerprint == model.training_record
    (factor,) = unit.data_fingerprint["factors.0.dataset"]
    assert pd.Timestamp(factor["request"]["end"]) == pd.Timestamp(bars[29])


def test_a_seed_ensemble_records_on_its_unit_and_its_members_record_none(tmp_path):
    dataset_config, bars = _store(tmp_path)
    model = make_model(tmp_path / "e", dataset_config, head=SeededHead, **_dates(bars))
    ensemble = SeedEnsemble(model, [0, 1, 2])

    unit = TrainedRun.open(ensemble.collect().train())

    assert set(unit.data_fingerprint) == {f"model.{key}" for key in MODEL_KEYS}
    # One read for every member: each request is recorded once.
    assert all(len(entries) == 1 for entries in unit.data_fingerprint.values())
    assert [member.data_fingerprint for member in unit.members] == [{}, {}, {}]


def test_a_walk_forward_unit_records_and_its_folds_record_none(tmp_path):
    dataset_config, bars = _store(tmp_path)
    dates = {**_dates(bars), "end_date": _day(bars[-1]), "test_end": _day(bars[-1])}
    model = make_model(tmp_path / "cv", dataset_config, **dates)

    unit = model.collect().train_cv(train_periods=30)

    assert set(TrainedRun.open(unit.path).data_fingerprint) == MODEL_KEYS
    assert all(fold.data_fingerprint == {} for fold in unit.folds)


def _train_mode_run(tmp_path, dataset_config, bars):
    """A train-mode backtest over bars 30..50, written to a run directory."""
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=make_model(tmp_path / "backtest", dataset_config, **_dates(bars)),
            model_mode="train",
            start_date=_day(bars[30]),
            end_date=_day(bars[50]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=5,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        )
    ).run()


def _unit_record(result) -> dict:
    """The training record of the unit a backtest result trained."""
    return BacktestRun.open(result.run_dir).trained_run().data_fingerprint


def test_a_train_mode_rebuild_is_silent_on_unchanged_training_data(tmp_path, warnings_logged):
    dataset_config, bars = _store(tmp_path)
    first = _train_mode_run(tmp_path, dataset_config, bars)

    rebuilt = BacktestRun.open(first.run_dir).rebuild_backtester()
    assert rebuilt.expected_training_fingerprint == _unit_record(first)
    warnings_logged.clear()
    rebuilt.run()

    assert [m for m in warnings_logged if "fingerprint mismatch" in m] == []


def test_a_train_mode_rebuild_warns_on_changed_training_data_and_completes(
    tmp_path, warnings_logged
):
    """A re-base at bar 10 lies in training only: the training record warns, the run's does not."""
    dataset_config, bars = _store(tmp_path)
    first = _train_mode_run(tmp_path, dataset_config, bars)
    group = zarr.open_group(dataset_config.zarr_file_path, mode="r+")
    group["adjClose"][10, 0] = float(group["adjClose"][10, 0]) * 1.25

    warnings_logged.clear()
    second = BacktestRun.open(first.run_dir).rebuild_backtester().run()

    assert second.run_dir.is_dir()
    training = _training_mismatches(warnings_logged)
    assert any("'factors.0.dataset'" in m and "digest differs" in m for m in training), warnings_logged
    assert any("'labels.0.factor.dataset'" in m for m in training), warnings_logged
    assert [m for m in warnings_logged if "fingerprint mismatch" in m and m not in training] == []


def test_a_load_mode_backtest_records_and_compares_no_training_data(tmp_path, warnings_logged):
    dataset_config, bars = _store(tmp_path)
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **_dates(bars)))
    first = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=make_model(tmp_path / "backtest", dataset_config, **_dates(bars)),
            model_mode="load",
            checkpoint=str(checkpoint),
            start_date=_day(bars[30]),
            end_date=_day(bars[50]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=5,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        )
    ).run()
    group = zarr.open_group(dataset_config.zarr_file_path, mode="r+")
    group["adjClose"][10, 0] = float(group["adjClose"][10, 0]) * 1.25

    rebuilt = BacktestRun.open(first.run_dir).rebuild_backtester()
    assert rebuilt.expected_training_fingerprint is None
    warnings_logged.clear()
    rebuilt.run()

    assert not any(".labels." in key for key in rebuilt.data_fingerprint)
    assert _training_mismatches(warnings_logged) == []


def test_a_train_mode_rebuild_without_its_trained_unit_still_runs(tmp_path, warnings_logged):
    dataset_config, bars = _store(tmp_path)
    first = _train_mode_run(tmp_path, dataset_config, bars)
    shutil.rmtree(BacktestRun.open(first.run_dir).trained_run().path)

    warnings_logged.clear()
    rebuilt = BacktestRun.open(first.run_dir).rebuild_backtester()

    assert rebuilt.expected_training_fingerprint is None
    assert any("cannot be opened" in m for m in warnings_logged)
    assert rebuilt.run().run_dir.is_dir()


def test_a_rebuild_that_replaces_the_model_mode_compares_no_training_data(tmp_path):
    dataset_config, bars = _store(tmp_path)
    first = _train_mode_run(tmp_path, dataset_config, bars)
    checkpoint = BacktestRun.open(first.run_dir).trained_run().checkpoint

    rebuilt = BacktestRun.open(first.run_dir).rebuild_backtester(
        model_mode="load", checkpoint=str(checkpoint)
    )

    assert rebuilt.expected_training_fingerprint is None
