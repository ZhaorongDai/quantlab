"""Reads of a factor risk model's stores are recorded in the run's data fingerprint (#207, ADR 0021).

``RiskStore.read`` is the seam both readers go through: the covariance
estimator reads the estimate store's row at each bar, the backtest's factor
attribution reads both stores over its window. A read is recorded under the
risk model's component path and the store (``risk_model.regression``,
``risk_model.estimate``); one store's reads that differ only in their range
are one request, over the first to the last bar read, so reading the store a
bar at a time costs one entry and one hash. The stores' rows are not a
``(timestamp, symbol)`` panel: their ``factor`` axes are hashed too.

The risk model and backtest are the planted ones of
``tests.test_backtest_factor_attribution``; synthetic, CPU-only and offline.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from loguru import logger

import quantlab.runs.record as record
from quantlab.backtest.predefined.weights import WeightsVectorBt
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.record import DataRecorder, _dataset_fingerprint
from tests.test_backtest_factor_attribution import (
    BARS,
    _backtester,
    _plant,
    _risk_model,
    _weights,
)


@pytest.fixture
def warnings_logged():
    """Collect the loguru warnings emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


@pytest.fixture(scope="module")
def planted():
    return _plant()


@pytest.fixture(scope="module")
def model(planted, tmp_path_factory):
    return _risk_model(tmp_path_factory.mktemp("risk"), *planted)


@pytest.fixture(scope="module")
def run(planted, model, tmp_path_factory):
    output = tmp_path_factory.mktemp("runs")
    return _backtester(planted[0], model, output_dir=str(output)).run_weights(_weights())


def _day(ts) -> str:
    return pd.Timestamp(ts).isoformat()


def test_a_backtest_records_both_risk_stores_under_the_risk_model(run):
    fingerprint = BacktestRun.open(run.run_dir).data_fingerprint

    regression, = fingerprint["risk_model.regression"]
    estimate, = fingerprint["risk_model.estimate"]
    assert (regression["request"]["start"], regression["request"]["end"]) == (
        _day(BARS[0]), _day(BARS[-1])
    )
    # The last bar's forecast is never used: the estimate is read up to the bar before.
    assert (estimate["request"]["start"], estimate["request"]["end"]) == (
        _day(BARS[0]), _day(BARS[-2])
    )
    assert estimate["variables"] == ["factor_covariance", "specific_risk"]
    assert regression["variables"] == ["factor_return", "specific_return"]


def test_one_stores_reads_a_bar_at_a_time_are_one_request_hashed_once(model, monkeypatch):
    calls = []
    original = record._dataset_fingerprint
    monkeypatch.setattr(
        record, "_dataset_fingerprint",
        lambda ds, names, **kw: calls.append(ds.sizes["timestamp"]) or original(ds, names, **kw),
    )
    with DataRecorder(keys=[(model, "covariance.risk_model")]) as recorder:
        for bar in BARS[3:9]:
            model.estimate.read(bar, bar)
        model.estimate.read(BARS[5], BARS[7])

    entry, = recorder.records["covariance.risk_model.estimate"]
    assert (entry["request"]["start"], entry["request"]["end"]) == (_day(BARS[3]), _day(BARS[8]))
    assert entry["n_timestamps"] == 6
    assert calls == [6]


def test_a_store_read_outside_the_tree_falls_back_to_its_path(model):
    with DataRecorder() as recorder:
        model.regression.read(BARS[0], BARS[1])

    assert list(recorder.records) == [f"{type(model).__name__}:{model.config.regression_path}"]


def test_the_risk_stores_factor_axes_are_part_of_the_digest(planted):
    estimate = planted[3]
    relabelled = estimate.assign_coords(
        factor_i=[f"{name}_x" for name in estimate.factor_i.values],
        factor_j=[f"{name}_x" for name in estimate.factor_j.values],
    )
    names = ["factor_covariance", "specific_risk"]
    before = _dataset_fingerprint(estimate, names)
    after = _dataset_fingerprint(relabelled, names)

    assert before["digest"] != after["digest"]
    assert before["variable_digests"] == after["variable_digests"]


def test_a_rebuild_on_a_changed_estimate_store_warns(planted, tmp_path, warnings_logged):
    prices, exposures, regression, estimate = planted
    model = _risk_model(tmp_path / "risk", prices, exposures, regression, estimate)
    first = _backtester(prices, model, output_dir=str(tmp_path / "runs")).run_weights(_weights())

    # The estimate store is rebuilt from other forecasts, at the same path.
    restated = estimate.assign(specific_risk=estimate["specific_risk"] * 1.5)
    _risk_model(tmp_path / "risk", prices, exposures, regression, restated)
    BacktestRun.open(first.run_dir).rebuild_backtester(output_dir=None).run_weights(_weights())

    mismatches = [m for m in warnings_logged if "mismatch" in m]
    assert len(mismatches) == 1
    assert "'risk_model.estimate'" in mismatches[0]
    assert "specific_risk" in mismatches[0]


def test_extending_the_stores_past_the_window_changes_no_record(planted, tmp_path, warnings_logged):
    """The bar-interval check reads the whole regression store; that read is not the run's."""
    prices, exposures, regression, estimate = planted
    short = dict(
        regression=regression.isel(timestamp=slice(0, -3)),
        estimate=estimate.isel(timestamp=slice(0, -3)),
    )
    model = _risk_model(tmp_path / "risk", prices, exposures, **short)
    backtester = _backtester(prices, model, output_dir=str(tmp_path / "runs"))
    backtester = WeightsVectorBt(
        dataclasses.replace(backtester.config, end_date=str(BARS[-4].date()))
    )
    weights = _weights().isel(timestamp=slice(0, -3))
    first = backtester.run_weights(weights)

    _risk_model(tmp_path / "risk", prices, exposures, regression, estimate)  # now to BARS[-1]
    rebuilt = BacktestRun.open(first.run_dir).rebuild_backtester(output_dir=None)
    rebuilt.run_weights(weights)

    assert [m for m in warnings_logged if "mismatch" in m] == []
    assert np.array_equal(
        [e["digest"] for e in rebuilt.data_fingerprint["risk_model.regression"]],
        [e["digest"] for e in BacktestRun.open(first.run_dir).data_fingerprint["risk_model.regression"]],
    )
