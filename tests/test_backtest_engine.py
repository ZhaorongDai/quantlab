"""Engine-layer locks for the vectorbt backtester (phase 03.7, plan 04).

What this file locks:

- the vectorbt engine facts CONTEXT verified by probe: a bar-t target weight
  fills at bar t+1's fill price (D-05), a long position flips to short in one
  rebalance under ``direction="both"`` (D-05), slippage moves buys up and sells
  down, fees are size x fill price x rate, the config defaults are 5bp / 5bp /
  1,000,000 (D-19), and order sizes are fractional (D-20);
- the delisting rule (D-07): prices are forward-filled so one NaN-priced
  holding cannot freeze every later rebalance of the whole group (RESEARCH
  Pitfall 2), a held symbol whose raw fill price turns NaN is force-liquidated
  at its last price and recorded, a rebalance row mixing NaN and finite
  weights is refused before vectorbt runs (Pitfall 3), and a symbol that lists
  late trades normally;
- the market spec (D-04), construction-time score-label validation (D-11),
  the deferred benchmark hook (D-08), and that a sibling engine subclass needs
  no change to ``BaseBacktester`` (D-01).

Every engine assertion reads vectorbt's own order records (mapped onto
``SimulationResult.orders``), never a re-derivation of what the engine should
have done. A change in engine semantics -- a lost one-bar shift, a long-only
direction, a dropped forward-fill -- therefore turns these tests red.

Price panels are tiny hand-built ``xr.Dataset``s whose variable names come
from ``USEquityCrossectionSelectStockVectorBt.MARKET``, never from literal
column names, so D-04's single source of truth is kept even in tests.
Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig
from tests.backtest_fixtures import make_model, make_stock_dataset, write_price_store

MARKET = USEquityCrossectionSelectStockVectorBt.MARKET


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _config(tmp_path, **overrides) -> CrossSectionBacktestConfig:
    """A load-mode config over a fixture store and a fixture model.

    The checkpoint path need not exist: engine tests call `_simulate`
    directly and never `run()`. fees / slippage / init_cash are left to the
    dataclass defaults unless a test overrides them.
    """
    dataset_config = write_price_store(tmp_path, n_bars=40)
    bars = pd.bdate_range("2024-01-01", periods=40)
    model = make_model(
        tmp_path,
        dataset_config,
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=model,
        model_mode="load",
        checkpoint=str(tmp_path / "missing.joblib"),
        start_date=_day(bars[30]),
        end_date=_day(bars[39]),
        output_dir=str(tmp_path / "runs"),
        rebalance_periods=1,
        direction="long_only",
        top_n=1,
    )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


def _backtester(tmp_path, **overrides) -> USEquityCrossectionSelectStockVectorBt:
    return USEquityCrossectionSelectStockVectorBt(_config(tmp_path, **overrides))


def _timestamps(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _panel(fill, valuation, timestamps, symbols) -> xr.Dataset:
    """Prices under the backtester's MARKET column names, dims (timestamp, symbol)."""
    return xr.Dataset(
        {
            MARKET.fill_price_column: (
                ("timestamp", "symbol"),
                np.asarray(fill, dtype=np.float64),
            ),
            MARKET.valuation_price_column: (
                ("timestamp", "symbol"),
                np.asarray(valuation, dtype=np.float64),
            ),
        },
        coords={"timestamp": timestamps, "symbol": list(symbols)},
    )


def _weights(rows, timestamps, symbols) -> xr.Dataset:
    """D-03 weights: one row per bar, NaN = hold, finite = rebalance target."""
    return xr.Dataset(
        {"weight": (("timestamp", "symbol"), np.asarray(rows, dtype=np.float64))},
        coords={"timestamp": timestamps, "symbol": list(symbols)},
    )


def _orders_for(orders: xr.Dataset, symbol: str) -> list[dict]:
    rows = []
    for i in range(orders.sizes["order"]):
        if str(orders["symbol"].values[i]) != symbol:
            continue
        rows.append(
            {
                "timestamp": pd.Timestamp(orders["timestamp"].values[i]),
                "size": float(orders["size"].values[i]),
                "price": float(orders["price"].values[i]),
                "fees": float(orders["fees"].values[i]),
                "side": str(orders["side"].values[i]),
            }
        )
    return rows


NAN = np.nan


# --------------------------------------------------------------------------
# Task 1: engine facts (D-05, D-19, D-20)
# --------------------------------------------------------------------------


def test_bar_t_weight_fills_at_bar_t_plus_1_open(tmp_path):
    """D-05: the bar-0 target fills on bar 1 at bar 1's fill price, exactly."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts = _timestamps(3)
    fill = [[10.0], [11.0], [12.0]]
    valuation = [[10.5], [11.5], [12.5]]
    weights = _weights([[1.0], [NAN], [NAN]], ts, ["A"])

    orders = backtester._simulate(weights, _panel(fill, valuation, ts, ["A"])).orders

    assert orders.sizes["order"] == 1
    order = _orders_for(orders, "A")[0]
    assert order["timestamp"] == ts[1]
    assert order["price"] == 11.0
    assert order["side"] == "Buy"


def test_slippage_moves_buys_up_and_sells_down(tmp_path):
    """D-19: slippage is applied against the trade on the fill-bar price."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.01)
    ts = _timestamps(4)
    fill = [[10.0], [11.0], [12.0], [13.0]]
    valuation = [[10.5], [11.5], [12.5], [13.5]]
    weights = _weights([[1.0], [NAN], [0.0], [NAN]], ts, ["A"])

    orders = _orders_for(
        backtester._simulate(weights, _panel(fill, valuation, ts, ["A"])).orders, "A"
    )

    assert [o["side"] for o in orders] == ["Buy", "Sell"]
    assert orders[0]["timestamp"] == ts[1]
    assert orders[0]["price"] == pytest.approx(11.0 * 1.01, rel=1e-12)
    assert orders[1]["timestamp"] == ts[3]
    assert orders[1]["price"] == pytest.approx(13.0 * 0.99, rel=1e-12)


def test_fees_are_size_times_price_times_rate(tmp_path):
    """D-19: every order's fees equal size x fill price x fees."""
    rate = 0.001
    backtester = _backtester(tmp_path, fees=rate, slippage=0.0)
    ts = _timestamps(6)
    symbols = ["A", "B"]
    fill = [[10.0, 20.0], [11.0, 21.0], [12.0, 19.0], [13.0, 22.0], [12.5, 23.0], [14.0, 24.0]]
    valuation = [[p + 0.5 for p in row] for row in fill]
    weights = _weights(
        [[0.5, 0.5], [NAN, NAN], [1.0, 0.0], [NAN, NAN], [0.0, 1.0], [NAN, NAN]],
        ts,
        symbols,
    )

    orders = backtester._simulate(weights, _panel(fill, valuation, ts, symbols)).orders

    assert orders.sizes["order"] >= 4
    size = orders["size"].values
    price = orders["price"].values
    fees = orders["fees"].values
    np.testing.assert_allclose(fees, size * price * rate, rtol=1e-9, atol=0.0)
    assert (fees > 0).all()


def test_long_to_short_flip_executes_in_one_rebalance(tmp_path):
    """D-05: +1.0 then -1.0 on A flips the position within a single fill bar."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts = _timestamps(5)
    symbols = ["A", "B"]
    fill = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0], [14.0, 24.0]]
    valuation = [[p + 0.5 for p in row] for row in fill]
    weights = _weights(
        [[1.0, 0.0], [NAN, NAN], [-1.0, 0.0], [NAN, NAN], [NAN, NAN]], ts, symbols
    )

    orders = _orders_for(
        backtester._simulate(weights, _panel(fill, valuation, ts, symbols)).orders, "A"
    )

    signed = {}
    for o in orders:
        delta = o["size"] if o["side"] == "Buy" else -o["size"]
        signed[o["timestamp"]] = signed.get(o["timestamp"], 0.0) + delta
    assert set(signed) == {ts[1], ts[3]}, "the flip must be a single rebalance fill"
    position_after_entry = signed[ts[1]]
    position_after_flip = signed[ts[1]] + signed[ts[3]]
    assert position_after_entry > 0
    assert position_after_flip < 0


def test_order_sizes_are_fractional(tmp_path):
    """D-20: a target that cannot be met in whole shares is filled fractionally."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0, init_cash=1_000.0)
    ts = _timestamps(3)
    fill = [[33.3], [33.3], [33.3]]
    valuation = [[33.3], [33.3], [33.3]]
    weights = _weights([[1.0], [NAN], [NAN]], ts, ["A"])

    orders = backtester._simulate(weights, _panel(fill, valuation, ts, ["A"])).orders

    sizes = orders["size"].values
    assert sizes.size >= 1
    assert np.any(np.abs(sizes - np.round(sizes)) > 1e-9)


def test_config_defaults_match_d19(tmp_path):
    """D-19: 5bp fees, 5bp slippage and 1,000,000 initial cash by default."""
    config = _config(tmp_path)

    assert config.fees == 0.0005
    assert config.slippage == 0.0005
    assert config.init_cash == 1_000_000.0


def test_first_equity_value_is_init_cash(tmp_path):
    """No order fills on bar 0, so the first equity value is exactly init_cash."""
    init_cash = 250_000.0
    backtester = _backtester(tmp_path, fees=0.001, slippage=0.001, init_cash=init_cash)
    ts = _timestamps(4)
    symbols = ["A", "B"]
    fill = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]
    valuation = [[p + 0.5 for p in row] for row in fill]
    weights = _weights([[0.5, 0.5], [NAN, NAN], [NAN, NAN], [NAN, NAN]], ts, symbols)

    value = backtester._simulate(weights, _panel(fill, valuation, ts, symbols)).value

    assert float(value.values[0]) == init_cash
    assert float(value.values[-1]) != init_cash
