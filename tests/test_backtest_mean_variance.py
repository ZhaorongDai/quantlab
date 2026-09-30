"""The vectorised backtester drives per-bar portfolio construction (#79).

What is locked here, and what turns it red:

- The constructor's `lookback_bars` extends the price warm-up: the first
  backtest bar's context already holds a full, finite window of one-bar
  valuation returns ending at that bar.
- The current weights a constructor is handed are the last rebalance's
  weights drifted by the valuation-price returns since, renormalised.
- A bar whose construction raises `PortfolioConstructionError` holds (an
  all-NaN row), is logged, and is listed in `metrics.json`.
- A `MeanVarianceOptimizer` backtest is fully invested on every rebalance
  bar, and its `config.json` (optimiser and risk model) rebuilds a
  backtester that re-runs identically.

Everything is synthetic, CPU-only and offline.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, LedoitWolfConfig, MeanVarianceConfig
from quantlab.base.portfolio import PortfolioConstructionError, PortfolioConstructor
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.utils.module import load_backtester_from_config
from tests.backtest_fixtures import make_model, make_stock_dataset, write_price_store

N_BARS = 90
LOOKBACK = 20
WINDOW = (40, 85)
REBALANCE = 5


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class RecorderConfig:
    lookback_bars: int = LOOKBACK
    fail_on: int | None = None


#: Contexts seen by every `Recorder`, in call order (rebuilt instances share it).
SEEN: list = []


class Recorder(PortfolioConstructor):
    """Holds AAA and BBB half and half, records its contexts, fails on one call."""

    config_cls = RecorderConfig

    @property
    def lookback_bars(self):
        return self.config.lookback_bars

    def construct(self, context):
        SEEN.append(context)
        if self.config.fail_on is not None and len(SEEN) - 1 == self.config.fail_on:
            raise PortfolioConstructionError("solver gave up")
        row = xr.zeros_like(context.current_weights)
        row.loc[{"symbol": ["AAA", "BBB"]}] = 0.5
        return row


def _setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS, seed=11)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    model = make_model(
        tmp_path / "model",
        dataset_config,
        start_date=_day(bars[0]),
        end_date=_day(bars[39]),
        train_start=_day(bars[0]),
        train_end=_day(bars[34]),
        test_start=_day(bars[35]),
        test_end=_day(bars[39]),
    )
    return dataset_config, model, bars


def _backtester(tmp_path, constructor, *, output_dir=None):
    dataset_config, model, bars = _setup(tmp_path)
    return (
        USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=make_stock_dataset(dataset_config),
                model=model,
                model_mode="train",
                start_date=_day(bars[WINDOW[0]]),
                end_date=_day(bars[WINDOW[1]]),
                output_dir=output_dir,
                rebalance_periods=REBALANCE,
                constructor=constructor,
                fees=0.0,
                slippage=0.0,
            )
        ),
        dataset_config,
        bars,
    )


@pytest.fixture(autouse=True)
def _clear_seen():
    SEEN.clear()


def test_the_first_backtest_bar_has_a_full_return_window(tmp_path):
    backtester, dataset_config, bars = _backtester(tmp_path, Recorder(RecorderConfig()))

    backtester.run()

    first = SEEN[0]
    assert first.timestamp == pd.Timestamp(bars[WINDOW[0]])
    assert first.returns.sizes["timestamp"] == LOOKBACK
    assert pd.Timestamp(first.returns.timestamp.values[-1]) == first.timestamp
    assert np.isfinite(first.returns.values).all()
    close = xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].transpose("timestamp", "symbol").values
    t = WINDOW[0]
    np.testing.assert_allclose(
        first.returns.values, close[t - LOOKBACK + 1 : t + 1] / close[t - LOOKBACK : t] - 1
    )


def test_the_current_weights_are_the_last_rebalance_weights_drifted(tmp_path):
    backtester, dataset_config, bars = _backtester(tmp_path, Recorder(RecorderConfig()))

    backtester.run()

    close = xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].transpose("timestamp", "symbol").values
    assert (SEEN[0].current_weights.values == 0.0).all()
    held = np.zeros(close.shape[1])
    held[:2] = 0.5
    for k in range(1, len(SEEN)):
        t0 = WINDOW[0] + (k - 1) * REBALANCE
        t1 = t0 + REBALANCE
        growth = close[t1] / close[t0]
        expected = held * growth / (1 + (held * (growth - 1)).sum())
        np.testing.assert_allclose(SEEN[k].current_weights.values, expected, rtol=1e-12)
        assert not np.allclose(expected, held)


def test_a_failing_bar_holds_is_logged_and_recorded(tmp_path):
    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        backtester, _, bars = _backtester(
            tmp_path, Recorder(RecorderConfig(fail_on=1)), output_dir=str(tmp_path / "runs")
        )
        result = backtester.run()
    finally:
        logger.remove(sink)

    failed_bar = bars[WINDOW[0] + REBALANCE]
    weights = result.weights["weight"].sel(timestamp=failed_bar).values
    assert np.isnan(weights).all()
    assert any("solver gave up" in m and pd.Timestamp(failed_bar).isoformat() in m for m in messages)
    saved = json.loads((result.run_dir / "metrics.json").read_text())
    assert saved["portfolio_construction"] == {
        "failed_bar_count": 1,
        "failed_bars": [pd.Timestamp(failed_bar).isoformat()],
    }
    # The bar after it drifts from the last rebalance that did trade.
    assert (SEEN[2].current_weights.values[:2] > 0).all()


def _optimizer():
    return MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="fwd_ret_1",
            risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=LOOKBACK)),
            ic=0.05,
            risk_aversion=5.0,
            turnover_penalty=0.001,
            weight_cap=0.4,
        )
    )


def test_a_mean_variance_backtest_is_fully_invested_and_rebuilds_identically(tmp_path):
    backtester, _, _ = _backtester(tmp_path, _optimizer(), output_dir=str(tmp_path / "runs"))

    original = backtester.run()

    weights = original.weights["weight"].values
    rebalance = np.isfinite(weights).all(axis=1)
    assert rebalance.sum() == len(range(0, WINDOW[1] - WINDOW[0], REBALANCE))
    np.testing.assert_allclose(weights[rebalance].sum(axis=1), 1.0, atol=1e-9)
    assert (weights[rebalance] >= 0).all() and (weights[rebalance] <= 0.4 + 1e-9).all()
    assert original.metrics["portfolio_construction"]["failed_bar_count"] == 0

    saved = json.loads((original.run_dir / "config.json").read_text())
    assert saved["constructor"]["name"] == "quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer"
    assert saved["constructor"]["risk_model"]["lookback_bars"] == LOOKBACK
    rebuilt = load_backtester_from_config(saved)
    assert rebuilt.config.constructor == backtester.config.constructor
    again = rebuilt.run()

    np.testing.assert_array_equal(again.weights["weight"].values, weights)
    np.testing.assert_array_equal(again.simulation.value.values, original.simulation.value.values)


def test_a_predictor_without_the_expected_return_label_is_refused_at_construction(tmp_path):
    optimizer = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="fwd_ret_5",
            risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=LOOKBACK)),
            ic=0.05,
            risk_aversion=5.0,
        )
    )
    with pytest.raises(ValueError, match="fwd_ret_5"):
        _backtester(tmp_path, optimizer)
