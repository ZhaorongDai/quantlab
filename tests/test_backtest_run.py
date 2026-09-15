"""End-to-end tracer for `BaseBacktester.run()` (phase 03.7, plan 01).

One test drives the whole slice: load a checkpoint, re-date the factors with a
bar-accurate warm-up, predict a panel, pick TopN target weights, simulate with
t+1-open fills in vectorbt, and persist the run directory.

What is locked, and what turns it red:

- the run directory holds exactly the D-24 artifacts: config.json,
  weights.zarr, equity.zarr, liquidations.json, metrics.json, report.html and
  fingerprint.json (their contents are locked in
  tests/test_backtest_persistence.py);
- predictions are one variable per label on (timestamp, symbol), cover every
  price symbol and exactly the window bars (D-29, D-06);
- weights follow D-03: rebalance rows every 5 bars from the window start, the
  last bar never rebalances, every other row is all-NaN, long_only books sum
  to 1 in two 0.5 slots (D-09, D-18);
- the chosen names at the first rebalance are the top two past returns
  recomputed here straight from the Zarr store -- reversed ranking or a
  raw-vs-adjusted column mix-up (D-04) changes them;
- the first order fills on window bar 1 at that bar's adjusted open (D-05);
- metrics.json is strict JSON with no benchmark row (D-08), and config.json
  names the backtester, model and price dataset by dotted path.

What this tracer does NOT lock (mutation-verified to stay green here, owned by
later plans):

- bar-accurate warm-up (D-15): with n=1, `Factor._reset_dataset_config`'s
  calendar-day buffer alone covers the lookback, so a zero-bar warm-up passes
  (plan 03.7-06);
- the predictions reindex onto the price symbol axis (D-06): the fixture's
  factor and price stores carry the same symbols (plan 03.7-06);
- NaN/inf -> null in `to_jsonable`: this seed's stats contain no non-finite
  value (plans 03.7-07 and 03.7-09).

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import CrossSectionBacktestConfig
from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from tests.backtest_fixtures import (
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
WINDOW_START_BAR = 30
WINDOW_END_BAR = 50
REBALANCE_PERIODS = 5
TOP_N = 2
INIT_CASH = 1_000_000.0


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _strict_json(path: Path) -> dict:
    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token!r} in {path}")

    return json.loads(path.read_text(), parse_constant=_reject)


def test_run_load_mode_end_to_end_long_only(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    raw = xr.open_zarr(dataset_config.zarr_file_path).load()
    bars = raw.timestamp.values
    model_dates = dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )

    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **model_dates)
    )

    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=make_model(tmp_path / "backtest", dataset_config, **model_dates),
            model_mode="load",
            checkpoint=str(checkpoint),
            start_date=_day(bars[WINDOW_START_BAR]),
            end_date=_day(bars[WINDOW_END_BAR]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=REBALANCE_PERIODS,
            direction="long_only",
            top_n=TOP_N,
            fees=0.0,
            slippage=0.0,
            init_cash=INIT_CASH,
        )
    )
    result = backtester.run()

    # --- run directory (D-24 subset) -------------------------------------
    assert sorted(p.name for p in result.run_dir.iterdir()) == [
        "config.json",
        "equity.zarr",
        "fingerprint.json",
        "liquidations.json",
        "metrics.json",
        "report.html",
        "weights.zarr",
    ]

    # --- predictions (D-29, D-06) -----------------------------------------
    window_bars = bars[WINDOW_START_BAR : WINDOW_END_BAR + 1]
    n_window = len(window_bars)
    assert n_window == 21
    predictions = result.predictions
    assert list(predictions.data_vars) == ["fwd_ret_1"]
    assert predictions["fwd_ret_1"].dims == ("timestamp", "symbol")
    assert predictions.symbol.values.tolist() == SYMBOLS
    np.testing.assert_array_equal(
        predictions.timestamp.values.astype("datetime64[ns]"),
        window_bars.astype("datetime64[ns]"),
    )

    # --- weights contract (D-03, D-09, D-18) ------------------------------
    weights = result.weights["weight"].values
    assert weights.shape == (n_window, len(SYMBOLS))
    rebalance_rows = [0, 5, 10, 15]
    for i in range(n_window):
        row = weights[i]
        if i in rebalance_rows:
            assert np.isfinite(row).all(), f"row {i} must be finite"
            assert abs(row.sum() - 1.0) <= 1e-9
            assert np.count_nonzero(row == 0.5) == 2
            assert np.count_nonzero(row) == 2
        else:
            assert np.isnan(row).all(), f"row {i} must be all-NaN"

    # --- first-rebalance selection recomputed from the raw store (D-15, D-04)
    adj_close = raw["adjClose"].transpose("timestamp", "symbol").values
    past_return = (
        adj_close[WINDOW_START_BAR] / adj_close[WINDOW_START_BAR - 1] - 1.0
    )
    expected_top = set(np.array(SYMBOLS)[np.argsort(-past_return)[:TOP_N]])
    chosen = set(np.array(SYMBOLS)[weights[0] > 0])
    assert chosen == expected_top

    # --- t+1 open fill (D-05) ----------------------------------------------
    orders = result.simulation.orders
    order_times = orders["timestamp"].values.astype("datetime64[ns]")
    first_time = order_times.min()
    assert first_time == window_bars[1].astype("datetime64[ns]")
    adj_open = raw["adjOpen"].transpose("timestamp", "symbol").values
    first = np.flatnonzero(order_times == first_time)
    assert len(first) == TOP_N
    for i in first:
        symbol = str(orders["symbol"].values[i])
        assert symbol in expected_top
        expected_price = adj_open[WINDOW_START_BAR + 1, SYMBOLS.index(symbol)]
        assert orders["price"].values[i] == pytest.approx(expected_price, rel=1e-12)

    # --- equity ------------------------------------------------------------
    value = result.simulation.value
    assert value.sizes["timestamp"] == n_window
    assert float(value.values[0]) == pytest.approx(INIT_CASH)
    persisted_equity = xr.open_zarr(result.run_dir / "equity.zarr")
    assert persisted_equity["value"].sizes["timestamp"] == n_window

    # --- metrics.json (D-08, Pitfall 10) ------------------------------------
    metrics = _strict_json(result.run_dir / "metrics.json")
    assert "Total Return [%]" in metrics["whole"]
    assert "Benchmark Return [%]" not in metrics["whole"]

    # --- config.json ---------------------------------------------------------
    config = _strict_json(result.run_dir / "config.json")
    assert (
        config["name"]
        == "quantlab.backtest.us_equity.USEquityCrossectionSelectStockVectorBt"
    )
    assert config["model"]["name"] == "tests.backtest_fixtures.FirstFeatureHead"
    assert config["price_dataset"]["name"].endswith("StockDataset")
