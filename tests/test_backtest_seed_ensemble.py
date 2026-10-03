"""A `SeedEnsemble` backtests through `run()` and rebuilds like any predictor.

What is locked here, and what turns it red:

- `run()` in train mode trains every seed into one ensemble directory,
  records the unit's `run.json` as the trained checkpoint, and backtests the
  average of the members' predictions.
- `run()` in load mode with `checkpoint` = that `run.json` replays the
  ensemble with the training dates its members' records hold (a stale
  training window on the ensemble's own model is overridden by the recorded
  one).
- The data fingerprints of an ensemble run carry the same keys as a single
  model's: the members read identical inputs, which are hashed once.
- `load_backtester_from_config` rebuilds a load-mode and a train-mode
  ensemble backtest, and the re-run gives the same predictions, weights and
  equity curve.

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.utils import module as module_utils
from quantlab.utils.ensemble import average_predictions
from quantlab.utils.trained_run import TrainedRun
from quantlab.portfolio.predefined.top_n import TopNConstructor
from tests.backtest_fixtures import (
    SeededHead,
    make_model,
    make_stock_dataset,
    write_price_store,
)

N_BARS = 60
SEEDS = [0, 1, 2]
WINDOW = (30, 50)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    return dataset_config, bars


def _dates(bars, train_end_bar: int = 24) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[train_end_bar]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )


def _ensemble(root, dataset_config, bars, *, train_end_bar=24, head=SeededHead):
    model = make_model(
        root, dataset_config, head=head, **_dates(bars, train_end_bar)
    )
    return SeedEnsemble(model, SEEDS)


def _backtester(tmp_path, dataset_config, model, bars, *, name, checkpoint=None):
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=model,
            model_mode="train" if checkpoint is None else "load",
            checkpoint=None if checkpoint is None else str(checkpoint),
            start_date=_day(bars[WINDOW[0]]),
            end_date=_day(bars[WINDOW[1]]),
            output_dir=str(tmp_path / name / "runs"),
            rebalance_periods=5,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
            fees=0.0,
            slippage=0.0,
            init_cash=1_000_000.0,
        )
    )


def _saved_config(result) -> dict:
    return json.loads((result.run_dir / "config.json").read_text())


def _assert_same_result(a, b) -> None:
    xr.testing.assert_identical(a.predictions, b.predictions)
    xr.testing.assert_identical(a.weights, b.weights)
    np.testing.assert_array_equal(a.simulation.value.values, b.simulation.value.values)


def test_train_mode_trains_every_seed_and_backtests_the_average(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = _ensemble(tmp_path / "ens", dataset_config, bars)

    result = _backtester(tmp_path, dataset_config, ensemble, bars, name="train").run()

    checkpoint = Path(result.metrics["trained_checkpoint"])
    assert checkpoint.name == "run.json" and checkpoint.is_file()
    saved = TrainedRun.open(checkpoint)
    assert saved.kind == "ensemble"
    assert [m.seed for m in saved.members] == SEEDS
    assert all(m.checkpoint.is_file() for m in saved.members)
    assert _saved_config(result)["trained_checkpoint"] == str(checkpoint)

    start, end = _day(bars[WINDOW[0]]), _day(bars[WINDOW[1]])
    members = [m.predict_window(start, end) for m in ensemble.members]
    assert not np.allclose(
        members[0]["fwd_ret_1"].values, members[1]["fwd_ret_1"].values, equal_nan=True
    )
    expected = average_predictions(members)
    xr.testing.assert_allclose(
        result.predictions, expected.reindex_like(result.predictions)
    )
    assert np.isfinite(result.predictions["fwd_ret_1"].values).any()


def test_load_mode_replays_the_trained_unit_with_its_recorded_dates(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    trained = _ensemble(tmp_path / "trained", dataset_config, bars)
    checkpoint = trained.collect().train()
    # The ensemble given to the backtest carries a stale training window.
    stale = _ensemble(tmp_path / "stale", dataset_config, bars, train_end_bar=10)

    result = _backtester(
        tmp_path, dataset_config, stale, bars, name="load", checkpoint=checkpoint
    ).run()

    assert tuple(result.metrics["training_window"]) == (_day(bars[0]), _day(bars[24]))
    start, end = _day(bars[WINDOW[0]]), _day(bars[WINDOW[1]])
    xr.testing.assert_allclose(
        result.predictions,
        trained.predict_window(start, end).reindex_like(result.predictions),
    )


def test_ensemble_fingerprints_have_the_keys_of_a_single_model(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = _ensemble(tmp_path / "ens", dataset_config, bars)
    ensemble_checkpoint = ensemble.collect().train()
    single = make_model(tmp_path / "single", dataset_config, head=SeededHead, **_dates(bars))
    checkpoint = single.collect().train()

    ens_result = _backtester(
        tmp_path, dataset_config, _ensemble(tmp_path / "e2", dataset_config, bars),
        bars, name="ens", checkpoint=ensemble_checkpoint,
    ).run()
    single_result = _backtester(
        tmp_path, dataset_config,
        make_model(tmp_path / "s2", dataset_config, head=SeededHead, **_dates(bars)),
        bars, name="single", checkpoint=checkpoint,
    ).run()

    assert (
        _saved_config(ens_result)["data_fingerprint"]
        == _saved_config(single_result)["data_fingerprint"]
    )


def test_load_mode_rebuild_reproduces_the_run(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    checkpoint = _ensemble(tmp_path / "trained", dataset_config, bars).collect().train()
    first = _backtester(
        tmp_path, dataset_config, _ensemble(tmp_path / "ens", dataset_config, bars),
        bars, name="load", checkpoint=checkpoint,
    ).run()

    rebuilt = module_utils.load_backtester_from_config(_saved_config(first))

    assert type(rebuilt.config.model) is SeedEnsemble
    assert rebuilt.config.model.seeds == tuple(SEEDS)
    again = rebuilt.run()
    _assert_same_result(first, again)
    assert _saved_config(first)["data_fingerprint"] == _saved_config(again)["data_fingerprint"]


def test_train_mode_rebuild_retrains_and_reproduces_the_run(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    first = _backtester(
        tmp_path, dataset_config, _ensemble(tmp_path / "ens", dataset_config, bars),
        bars, name="train",
    ).run()

    rebuilt = module_utils.load_backtester_from_config(_saved_config(first))
    again = rebuilt.run()

    _assert_same_result(first, again)
    assert again.metrics["trained_checkpoint"] != first.metrics["trained_checkpoint"]
