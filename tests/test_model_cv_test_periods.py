"""Walk-forward CV with an explicit test length: ``train_cv(..., test_periods=k)``.

Without ``test_periods`` each fold tests on ``train_periods // 5`` bars, as
before. With it, each fold tests on ``k`` bars, the next fold starts ``k`` bars
later, and the fold count is ``(len - train_periods) // k``; a model and an
ensemble lay out the same folds, sliding or expanding. Everything is
synthetic, CPU-only and offline.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.model.predefined.model_ensemble import ModelEnsemble
from tests.backtest_fixtures import make_model, write_price_store
from tests.test_model_cv_expanding import (
    DATE_KEYS,
    FITTED,
    N_TIMES,
    TIMES,
    _model,
)

TRAIN_PERIODS = 10
TEST_PERIODS = 3
#: 20 bars: (20 - 10) // 3 = 3 folds, testing on bars 10-12, 13-15 and 16-18.
N_FOLDS = 3


@pytest.fixture(autouse=True)
def _reset_fitted():
    FITTED.clear()
    yield
    FITTED.clear()


def _days(record) -> tuple:
    return tuple(np.datetime64(record[k], "D") for k in DATE_KEYS)


@pytest.mark.parametrize("expanding", [False, True])
def test_each_fold_tests_on_test_periods_bars(tmp_path, expanding):
    model = _model(tmp_path, "model")
    model.collect()

    results = model.train_cv(
        train_periods=TRAIN_PERIODS, expanding=expanding, test_periods=TEST_PERIODS
    )

    first = 0
    expected = []
    for i in range(N_FOLDS):
        test = TRAIN_PERIODS + i * TEST_PERIODS
        start = first if expanding else i * TEST_PERIODS
        expected.append((TIMES[start], TIMES[test - 1], TIMES[test], TIMES[test + 2]))
    assert [_days(r) for r in results] == expected
    assert N_TIMES - (TRAIN_PERIODS + N_FOLDS * TEST_PERIODS) < TEST_PERIODS


def test_the_manifest_records_the_test_length_through_its_dates(tmp_path):
    model = _model(tmp_path, "model")
    model.collect()
    model.train_cv(train_periods=TRAIN_PERIODS, expanding=True, test_periods=TEST_PERIODS)

    (path,) = sorted((tmp_path / "model").rglob("cv_folds.json"))
    folds = json.loads(path.read_text())["folds"]
    assert [np.datetime64(f["test_start"], "D") for f in folds] == [
        TIMES[TRAIN_PERIODS + i * TEST_PERIODS] for i in range(N_FOLDS)
    ]


def test_without_test_periods_a_fold_tests_on_a_fifth_of_train_periods(tmp_path):
    model = _model(tmp_path, "model")
    model.collect()
    results = model.train_cv(train_periods=TRAIN_PERIODS)
    assert [np.datetime64(r["test_start"], "D") for r in results] == [
        TIMES[TRAIN_PERIODS + 2 * i] for i in range(5)
    ]


@pytest.mark.parametrize("value", [0, -1])
def test_a_non_positive_test_periods_raises_before_training(tmp_path, value):
    model = _model(tmp_path, "model")
    model.collect()
    with pytest.raises(ValueError, match="test_periods"):
        model.train_cv(train_periods=TRAIN_PERIODS, test_periods=value)
    assert FITTED == []
    assert not (tmp_path / "model").exists()


def test_a_small_train_periods_is_accepted_with_an_explicit_test_length(tmp_path):
    """The five-bar minimum only exists for the one-fifth default."""
    model = _model(tmp_path, "model")
    model.collect()
    results = model.train_cv(train_periods=4, test_periods=4)
    assert len(results) == (N_TIMES - 4) // 4


def test_an_ensemble_lays_out_the_folds_a_model_does(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=40)
    stamps = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values

    def member(name):
        return make_model(
            tmp_path / name,
            dataset_config,
            start_date=_day(stamps[0]),
            end_date=_day(stamps[35]),
            train_start=_day(stamps[0]),
            train_end=_day(stamps[19]),
            test_start=_day(stamps[20]),
            test_end=_day(stamps[35]),
        )

    model = member("model").collect()
    model_results = model.train_cv(train_periods=15, expanding=True, test_periods=4)
    ensemble = ModelEnsemble([member("a"), member("b")]).collect()
    ensemble_results = ensemble.train_cv(train_periods=15, expanding=True, test_periods=4)

    assert len(model_results) == (36 - 15) // 4
    assert [_days(r) for r in ensemble_results] == [_days(r) for r in model_results]
    with pytest.raises(ValueError, match="test_periods"):
        ensemble.train_cv(train_periods=15, test_periods=0)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")
