"""The vectorbt engine's sizing basis (#109, quantlab-trader ADR 0007, rung L1).

A target weight is turned into a share count against a price: by default
(``sizing_basis="fill"``) the fill price of the bar the order executes on,
which is vectorbt's own default; with ``sizing_basis="valuation"`` the
valuation price of the signal bar t (its close), with the portfolio valued at
those closes. The order still fills at t+1's fill price either way. What is
locked here, on a hand-computed two-symbol book through ``run_weights``:

- the fill basis sizes against t+1's open and the valuation basis against t's
  close, giving the literal share counts worked out below;
- the basis is recorded in the run's ``config.json`` and refused when unknown.

The default's bit-identity with earlier runs is locked by the TopN regression
anchor (``tests/test_backtest_topn_reference.py``), not here.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.weights import WeightsVectorBt
from quantlab.base.config import WeightsBacktestConfig
from quantlab.dataset.memory import FrameDataset

BARS = pd.bdate_range("2024-01-01", periods=5)

# AAA moves between the open and the close; BBB stays at 20 and is never held.
AAA_OPEN = [10.0, 10.0, 12.0, 16.0, 16.0]
AAA_CLOSE = [10.0, 10.0, 15.0, 20.0, 20.0]


def _backtester(**overrides) -> WeightsVectorBt:
    prices = FrameDataset(pd.DataFrame({
        "timestamp": np.repeat(BARS, 2),
        "symbol": ["AAA", "BBB"] * BARS.size,
        "open": [p for o in AAA_OPEN for p in (o, 20.0)],
        "close": [p for c in AAA_CLOSE for p in (c, 20.0)],
    }))
    config = WeightsBacktestConfig(
        price_dataset=prices, start_date="2024-01-01", end_date="2024-01-05",
        output_dir=None, rebalance_periods=1, fees=0.0, slippage=0.0, init_cash=1000.0,
        fill_price_column="open", valuation_price_column="close",
        trading_days_per_year=252, session_minutes_per_day=390,
    )
    return WeightsVectorBt(dataclasses.replace(config, **overrides))


def _weights() -> xr.DataArray:
    # Bar 0 buys AAA to half the book; bar 2 raises it to 84%.
    rows = [[0.5, 0.0], [np.nan, np.nan], [0.84, 0.0], [np.nan, np.nan], [np.nan, np.nan]]
    return xr.DataArray(
        rows, dims=("timestamp", "symbol"),
        coords={"timestamp": BARS, "symbol": ["AAA", "BBB"]},
    )


def _aaa_buys(result) -> list[float]:
    orders = result.simulation.orders
    return [
        float(size)
        for symbol, size in zip(orders["symbol"].values, orders["size"].values)
        if symbol == "AAA"
    ]


def test_fill_basis_sizes_against_the_next_open():
    # Bar 1: 0.5 * 1000 / open 10 = 50 shares. Bar 3: the book at open 16 is
    # 500 cash + 50 * 16 = 1300, so 0.84 * 1300 / 16 = 68.25 shares: buy 18.25.
    result = _backtester().run_weights(_weights())
    assert _aaa_buys(result) == pytest.approx([50.0, 18.25], rel=1e-12)


def test_valuation_basis_sizes_against_the_signal_close():
    # Bar 1: 0.5 * 1000 / close 10 = 50 shares, filled at open 10. Bar 3: the
    # book at bar 2's close 15 is 500 + 50 * 15 = 1250, so 0.84 * 1250 / 15
    # = 70 shares: buy 20, filled at open 16.
    result = _backtester(sizing_basis="valuation").run_weights(_weights())
    assert _aaa_buys(result) == pytest.approx([50.0, 20.0], rel=1e-12)
    prices = result.simulation.orders["price"].values
    assert prices.tolist() == [10.0, 16.0]


def test_the_basis_is_recorded_in_the_run_config():
    assert _backtester().get_config()["sizing_basis"] == "fill"
    assert _backtester(sizing_basis="valuation").get_config()["sizing_basis"] == "valuation"


def test_an_unknown_basis_is_refused():
    with pytest.raises(ValueError, match="sizing_basis"):
        _backtester(sizing_basis="open")
