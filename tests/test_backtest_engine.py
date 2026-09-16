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
  no change to ``BaseBacktester`` (D-01);
- the position-level trade statistics reported beside the lot-level ones
  (quick 260915-udx): the ``positions`` sub-dict of the ``whole`` block is
  proved to be vectorbt's positions view by deriving its counts from the
  positions accessor directly, on a fixture proved to trim holdings without
  closing them. Without that divergence the lock would be vacuous -- a
  ``positions`` block that forgot to switch the trades type would be a
  byte-for-byte copy of the exit-trades block and would still pass.

Every engine assertion reads vectorbt's own order records (mapped onto
``SimulationResult.orders``), never a re-derivation of what the engine should
have done. A change in engine semantics -- a lost one-bar shift, a long-only
direction, a dropped forward-fill -- therefore turns these tests red.

Price panels are tiny hand-built ``xr.Dataset``s whose variable names come
from ``USEquityCrossectionSelectStockVectorBt.MARKET``, never from literal
column names, so D-04's single source of truth is kept even in tests.
Everything is synthetic, CPU-only and offline.
"""

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import quantlab.backtest.engine_vectorbt as engine_module
from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import rebalance_mask
from quantlab.backtest.us_equity import (
    US_EQUITY_MARKET,
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.base.config import CrossSectionBacktestConfig
from tests.backtest_fixtures import (
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

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


# --------------------------------------------------------------------------
# Task 2: delisting (D-07, RESEARCH Pitfalls 2 and 3)
# --------------------------------------------------------------------------

DELIST_BAR = 3


def _delisting_case():
    """A held from a bar-0 rebalance (fill bar 1); A's prices are NaN from bar 3.

    Rebalance signals: bar 0 -> A=1.0; bar 4 -> A=0.0, B=1.0; bar 8 -> B=0.0.
    """
    n = 10
    ts = _timestamps(n)
    symbols = ["A", "B"]
    fill = np.array([[10.0 + t, 20.0 + t] for t in range(n)])
    valuation = fill + 0.5
    fill[DELIST_BAR:, 0] = NAN
    valuation[DELIST_BAR:, 0] = NAN
    rows = np.full((n, 2), NAN)
    rows[0] = [1.0, 0.0]
    rows[4] = [0.0, 1.0]
    rows[8] = [0.0, 0.0]
    return ts, symbols, fill, valuation, _weights(rows, ts, symbols)


def test_held_symbol_that_delists_is_liquidated_at_its_last_price_and_recorded(tmp_path):
    """D-07: one record for A, filled on bar 5 at A's last finite fill price."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts, symbols, fill, valuation, weights = _delisting_case()
    last_price = float(fill[DELIST_BAR - 1, 0])

    simulation = backtester._simulate(weights, _panel(fill, valuation, ts, symbols))

    assert simulation.liquidations == [
        {
            "symbol": "A",
            "signal_timestamp": ts[4],
            "fill_timestamp": ts[5],
            "price": last_price,
        }
    ]
    record = simulation.liquidations[0]
    assert type(record["symbol"]) is str
    assert isinstance(record["signal_timestamp"], pd.Timestamp)
    assert isinstance(record["fill_timestamp"], pd.Timestamp)
    assert type(record["price"]) is float
    sells = [o for o in _orders_for(simulation.orders, "A") if o["side"] == "Sell"]
    assert [o["timestamp"] for o in sells] == [ts[5]]
    assert sells[0]["price"] == last_price


def test_delisting_does_not_freeze_other_symbols(tmp_path):
    """Pitfall 2: a NaN-priced holding must not silently freeze the whole group."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts, symbols, fill, valuation, weights = _delisting_case()

    orders = backtester._simulate(weights, _panel(fill, valuation, ts, symbols)).orders

    b_orders = _orders_for(orders, "B")
    assert any(o["side"] == "Buy" and o["timestamp"] == ts[5] for o in b_orders)
    assert any(o["side"] == "Sell" and o["timestamp"] == ts[9] for o in b_orders)


def test_symbol_listing_late_fills_normally_once_listed(tmp_path):
    """D-07 boundary: leading NaN on a never-held symbol is not a delisting."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    n = 8
    ts = _timestamps(n)
    symbols = ["A", "B", "C"]
    fill = np.array([[10.0 + t, 20.0 + t, 30.0 + t] for t in range(n)])
    valuation = fill + 0.5
    fill[:4, 2] = NAN
    valuation[:4, 2] = NAN
    rows = np.full((n, 3), NAN)
    rows[0] = [1.0, 0.0, 0.0]
    rows[5] = [0.0, 0.0, 1.0]

    simulation = backtester._simulate(
        _weights(rows, ts, symbols), _panel(fill, valuation, ts, symbols)
    )

    c_orders = _orders_for(simulation.orders, "C")
    assert [(o["side"], o["timestamp"]) for o in c_orders] == [("Buy", ts[6])]
    assert c_orders[0]["price"] == float(fill[6, 2])
    assert not [r for r in simulation.liquidations if r["symbol"] == "C"]


def test_rebalance_row_mixing_nan_and_finite_is_refused_before_simulating(
    tmp_path, monkeypatch
):
    """Pitfall 3: [1.0, NaN] would silently hold B and block A; refuse it loudly."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    calls = []

    def _spy(**kwargs):
        calls.append(kwargs)
        raise AssertionError("vectorbt from_orders was reached")

    # Replace the engine module's `vbt` name only, so the real vectorbt
    # Portfolio class stays untouched for every other test in the process.
    monkeypatch.setattr(
        engine_module,
        "vbt",
        types.SimpleNamespace(Portfolio=types.SimpleNamespace(from_orders=_spy)),
    )
    ts = _timestamps(4)
    symbols = ["A", "B"]
    fill = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]
    rows = [[NAN, NAN], [1.0, NAN], [NAN, NAN], [NAN, NAN]]

    with pytest.raises(ValueError, match=_day(ts[1])):
        backtester._simulate(_weights(rows, ts, symbols), _panel(fill, fill, ts, symbols))
    assert calls == []


RUN_BARS = 60
RUN_WINDOW_START = 30
RUN_WINDOW_END = 50


def _trained_run_config(tmp_path, dataset_config, **overrides) -> CrossSectionBacktestConfig:
    """Load-mode config over a checkpoint trained on bars 0-29, backtesting bars 30-50."""
    bars = pd.bdate_range("2024-01-01", periods=RUN_BARS)
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
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(tmp_path / "backtest", dataset_config, **model_dates),
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(bars[RUN_WINDOW_START]),
        end_date=_day(bars[RUN_WINDOW_END]),
        output_dir=str(tmp_path / "runs"),
        rebalance_periods=5,
        direction="long_only",
        top_n=2,
        fees=0.0,
        slippage=0.0,
    )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


def test_end_to_end_delisting_run_records_the_liquidation(tmp_path):
    """Through run(): the selected symbol delists mid-window and is recorded."""
    delist_bar = RUN_WINDOW_START + 3
    # The fixture store is seeded, so a probe store with the same seed tells
    # which symbol the model picks at the window's first rebalance. The model
    # scores by the fixture factor's past return on adjClose (its own input
    # column, not a price the engine reads); delisting from bar 33 does not
    # touch bars 29 and 30, which that pick depends on.
    probe = xr.open_zarr(
        write_price_store(tmp_path / "probe", n_bars=RUN_BARS).zarr_file_path
    ).load()
    close = probe["adjClose"].transpose("timestamp", "symbol").values
    past_return = close[RUN_WINDOW_START] / close[RUN_WINDOW_START - 1] - 1.0
    picked = SYMBOLS[int(np.argmax(past_return))]

    dataset_config = write_price_store(
        tmp_path / "store", n_bars=RUN_BARS, delist_at={picked: delist_bar}
    )
    result = USEquityCrossectionSelectStockVectorBt(
        _trained_run_config(tmp_path, dataset_config, top_n=1)
    ).run()

    bars = pd.bdate_range("2024-01-01", periods=RUN_BARS)
    liquidations = result.simulation.liquidations
    assert liquidations, "the delisted holding must be recorded"
    for record in liquidations:
        assert set(record) == {"symbol", "signal_timestamp", "fill_timestamp", "price"}
    first = liquidations[0]
    assert first["symbol"] == picked
    assert first["signal_timestamp"] == bars[RUN_WINDOW_START + 5]
    assert first["fill_timestamp"] == bars[RUN_WINDOW_START + 6]
    last_open = probe[MARKET.fill_price_column].sel(symbol=picked).values[delist_bar - 1]
    assert first["price"] == pytest.approx(float(last_open), rel=1e-12)


# --------------------------------------------------------------------------
# Task 3: market spec (D-04), construction check (D-11), benchmark deferral
# (D-08) and sibling extensibility (D-01)
# --------------------------------------------------------------------------


def test_us_equity_year_freq_daily_is_252_days_and_minute_is_252_sessions():
    """D-04 / Pitfall 6: 252 trading days and 390-minute sessions, not 365 days."""
    assert US_EQUITY_MARKET.year_freq(np.timedelta64(1, "D")) == pd.Timedelta(days=252)
    assert US_EQUITY_MARKET.year_freq(np.timedelta64(1, "m")) == pd.Timedelta(
        minutes=252 * 390
    )


def _bars_per_year(interval) -> float:
    interval = pd.Timedelta(interval)
    return US_EQUITY_MARKET.year_freq(interval) / interval


def test_us_equity_year_freq_multi_day_bars_are_calendar_spans():
    """CR-02: weekly and monthly bars annualize with about 52 and 12 bars a year.

    A bar longer than one day is a calendar span (a weekly bar is stamped
    once per calendar week, holidays or not), so bars per year is 365.25 days
    over the interval, capped at `trading_days_per_year`: one bar can never
    be shorter than one trading day. The old formula divided the 252
    TRADING-day count by the CALENDAR-day interval and gave 36 bars for a
    weekly series and 8.13 for a 31-day monthly one, understating weekly
    Sharpe and Sortino by about sqrt(52/36). Those values go red here.
    """
    assert _bars_per_year("1D") == pytest.approx(252.0, rel=1e-12)
    assert _bars_per_year("7D") == pytest.approx(365.25 / 7, rel=1e-12)
    assert _bars_per_year("30D") == pytest.approx(365.25 / 30, rel=1e-12)
    assert _bars_per_year("31D") == pytest.approx(365.25 / 31, rel=1e-12)
    assert _bars_per_year("91D") == pytest.approx(365.25 / 91, rel=1e-12)
    assert 52.0 <= _bars_per_year("7D") <= 52.2
    assert 11.7 <= _bars_per_year("31D") <= 12.2

    # Continuous and non-increasing from one day up: no interval between one
    # day and a quarter annualizes with more bars than a shorter interval.
    intervals = [pd.Timedelta(hours=hours) for hours in range(24, 24 * 92, 6)]
    counts = [_bars_per_year(interval) for interval in intervals]
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    assert max(counts) == pytest.approx(252.0, rel=1e-12)


def test_us_equity_market_uses_adjusted_columns():
    """D-04: fills and valuation use the split/dividend-adjusted Tiingo EOD columns."""
    from quantlab.enums.data import TiingoColumns

    eod_columns = TiingoColumns.EOD.split(",")
    assert USEquityCrossectionSelectStockVectorBt.MARKET is US_EQUITY_MARKET
    assert US_EQUITY_MARKET.fill_price_column == "adjOpen"
    assert US_EQUITY_MARKET.valuation_price_column == "adjClose"
    assert US_EQUITY_MARKET.fill_price_column in eod_columns
    assert US_EQUITY_MARKET.valuation_price_column in eod_columns


def test_unknown_score_label_fails_at_construction_before_training(tmp_path):
    """D-11: a mistyped score_label fails when the backtester is built, before training."""
    config = _config(tmp_path, score_label="no_such_label")

    with pytest.raises(ValueError, match="no_such_label"):
        USEquityCrossectionSelectStockVectorBt(config)

    assert config.model.model is None
    assert not list(Path(config.model.config.model_save_dir).rglob("*.joblib"))


def test_benchmark_dataset_is_refused_naming_d08(tmp_path):
    """D-08: benchmark comparison is deferred; a non-None slot is refused by name."""
    benchmark = make_stock_dataset(write_price_store(tmp_path / "benchmark", n_bars=40))

    with pytest.raises(NotImplementedError, match="D-08"):
        _backtester(tmp_path, benchmark_dataset=benchmark)


def test_engine_stats_carry_no_benchmark_metric_and_warn_nothing(tmp_path, recwarn):
    """D-08 / Pitfall 6: a real run reports no Benchmark row and no benchmark_rets warning."""
    config = _trained_run_config(
        tmp_path, write_price_store(tmp_path / "store", n_bars=RUN_BARS)
    )

    result = USEquityCrossectionSelectStockVectorBt(config).run()

    whole = result.metrics["whole"]
    assert "Total Return [%]" in whole
    assert [key for key in whole if "Benchmark" in key] == []
    assert "benchmark" not in result.metrics
    benchmark_warnings = [
        w
        for w in recwarn.list
        if issubclass(w.category, UserWarning) and "benchmark_rets" in str(w.message)
    ]
    assert benchmark_warnings == []


def test_simulate_benchmark_returns_none(tmp_path):
    """D-08: the engine's benchmark hook stays in place and returns None this phase."""
    backtester = _backtester(tmp_path)

    assert (
        backtester._simulate_benchmark(
            backtester.config.start_date, backtester.config.end_date
        )
        is None
    )


class EqualWeightEveryone(VectorBtBacktester):
    """Test-local D-01 sibling of the US-equity backtester.

    It declares only `config_cls`, `MARKET` and `_generate_signals`, and holds
    every symbol at 1/n on each rebalance bar. `BaseBacktester` is not edited
    to make it run.
    """

    config_cls = CrossSectionBacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions: xr.Dataset, prices: xr.Dataset) -> xr.Dataset:
        n_bars = prices.sizes["timestamp"]
        n_symbols = prices.sizes["symbol"]
        weights = np.full((n_bars, n_symbols), np.nan)
        weights[rebalance_mask(n_bars, self.config.rebalance_periods)] = 1.0 / n_symbols
        return xr.Dataset(
            {"weight": (("timestamp", "symbol"), weights)},
            coords={"timestamp": prices.timestamp.values, "symbol": prices.symbol.values},
        )


def test_a_sibling_engine_subclass_runs_without_touching_the_base(tmp_path):
    """D-01: a new engine-layer sibling needs config_cls, MARKET and _generate_signals only."""
    own_members = {
        name
        for name in vars(EqualWeightEveryone)
        if not (name.startswith("__") and name.endswith("__")) and name != "_abc_impl"
    }
    assert own_members == {"config_cls", "MARKET", "_generate_signals"}

    config = _trained_run_config(
        tmp_path, write_price_store(tmp_path / "store", n_bars=RUN_BARS)
    )
    result = EqualWeightEveryone(config).run()

    weights = result.weights["weight"].values
    n_bars, n_symbols = weights.shape
    mask = rebalance_mask(n_bars, config.rebalance_periods)
    assert mask.any()
    np.testing.assert_allclose(weights[mask], 1.0 / n_symbols, rtol=0.0, atol=1e-15)
    assert np.isnan(weights[~mask]).all()
    assert result.simulation.orders.sizes["order"] > 0
    assert result.run_dir.is_dir()


# --------------------------------------------------------------------------
# Task 4: position-level trade statistics beside the lot-level ones
# (quick 260915-udx)
# --------------------------------------------------------------------------

#: vectorbt's display names for the trade-derived metrics, written out here
#: rather than derived from the engine's own constant: a test that asked the
#: implementation what it should contain would agree with any answer.
TRADE_METRIC_NAMES = (
    "Total Trades",
    "Total Closed Trades",
    "Total Open Trades",
    "Open Trade PnL",
    "Win Rate [%]",
    "Best Trade [%]",
    "Worst Trade [%]",
    "Avg Winning Trade [%]",
    "Avg Losing Trade [%]",
    "Avg Winning Trade Duration",
    "Avg Losing Trade Duration",
    "Profit Factor",
    "Expectancy",
)


class RotateOneOutEqualWeight(VectorBtBacktester):
    """A test-local sibling whose rebalances trim holdings without closing them.

    On rebalance k every symbol is targeted at 1/(n-1) except symbol k % n,
    which is targeted at zero. Two consecutive rebalances therefore RESIZE the
    surviving holdings -- vectorbt's default exit-trades view books each resize
    as its own closed trade -- while a symbol stays ONE continuous position
    from entry until its turn to be dropped comes round.

    That divergence is the point. On a fixture where every holding is opened
    and closed in one go the two views coincide, and a `positions` block that
    forgot to switch the trades type would be a byte-for-byte copy of the
    exit-trades block and pass every assertion below. This class exists so the
    lock cannot be satisfied vacuously.
    """

    config_cls = CrossSectionBacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions: xr.Dataset, prices: xr.Dataset) -> xr.Dataset:
        n_bars = prices.sizes["timestamp"]
        n_symbols = prices.sizes["symbol"]
        weights = np.full((n_bars, n_symbols), np.nan)
        for k, bar in enumerate(
            np.flatnonzero(rebalance_mask(n_bars, self.config.rebalance_periods))
        ):
            row = np.full(n_symbols, 1.0 / (n_symbols - 1))
            row[k % n_symbols] = 0.0
            weights[bar] = row
        return xr.Dataset(
            {"weight": (("timestamp", "symbol"), weights)},
            coords={"timestamp": prices.timestamp.values, "symbol": prices.symbol.values},
        )


@pytest.fixture(scope="module")
def rotating_run(tmp_path_factory):
    """One real `run()` of the trimming fixture, shared by the read-only locks.

    Module-scoped, so the three assertions below read one simulation instead
    of paying for three. The autouse wandb fixture is function-scoped and
    cannot be requested here, so the environment is set the same way
    tests/test_backtest_persistence.py sets it for its shared runs.
    """
    root = tmp_path_factory.mktemp("rotating")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WANDB_MODE", "disabled")
        mp.setenv("WANDB_SILENT", "true")
        config = _trained_run_config(
            root, write_price_store(root / "store", n_bars=RUN_BARS)
        )
        return RotateOneOutEqualWeight(config).run()


def test_positions_block_carries_the_trade_metrics_each_with_a_lot_level_twin(
    rotating_run,
):
    """The `whole` block gains a positions sub-dict: the 13 trade metrics, no more."""
    whole = rotating_run.metrics["whole"]
    positions = whole["positions"]

    assert isinstance(positions, dict)
    assert set(positions) == set(TRADE_METRIC_NAMES)
    # Every position-level row has a lot-level twin at the top level, so the
    # two views are comparable row by row rather than being two metric sets.
    for key in TRADE_METRIC_NAMES:
        assert key in whole, key
    # Portfolio-level metrics are deliberately NOT recomputed: the trades type
    # does not affect them, so a second copy could only drift from the first.
    for key in ("Start", "End", "Period", "Total Return [%]", "Sharpe Ratio",
                "Max Drawdown [%]", "Total Fees Paid", "turnover"):
        assert key not in positions, key
    # The top level is untouched: nothing renamed, nothing removed, and D-08's
    # no-benchmark rule still holds.
    assert "Total Return [%]" in whole
    assert [key for key in whole if "Benchmark" in key] == []


def test_positions_block_is_the_positions_view_not_a_second_exit_trades_copy(
    rotating_run,
):
    """The counts are re-derived from the positions accessor, independently.

    This is what goes red if the implementation recomputed the exit-trades
    stats under a new key instead of switching the trades type -- the failure
    mode that per-call `trades_type` kwargs produce silently, because
    `stats()` and `get_trades()` ignore them.
    """
    positions = rotating_run.metrics["whole"]["positions"]
    records = rotating_run.simulation.native.positions.records_readable
    status = records["Status"].astype(str)

    assert int(positions["Total Trades"]) == len(records)
    assert int(positions["Total Closed Trades"]) == int((status == "Closed").sum())
    assert int(positions["Total Open Trades"]) == int((status == "Open").sum())


def test_the_fixture_really_diverges_lot_level_from_position_level(rotating_run):
    """Non-vacuity: on THIS run the two views genuinely disagree.

    Without this the test above could pass on a fixture where each holding is
    entered and exited once, which is precisely when an exit-trades copy is
    indistinguishable from the positions view.
    """
    whole = rotating_run.metrics["whole"]
    positions = whole["positions"]
    lots = rotating_run.simulation.native.trades.records_readable
    holdings = rotating_run.simulation.native.positions.records_readable

    assert len(lots) > len(holdings) > 0, "the fixture must trim without closing"
    assert int(whole["Total Closed Trades"]) > int(positions["Total Closed Trades"]) > 0
    # The headline defect: the lot-level win rate is inflated by partial trims.
    assert np.isfinite(whole["Win Rate [%]"])
    assert np.isfinite(positions["Win Rate [%]"])
    assert whole["Win Rate [%]"] != pytest.approx(positions["Win Rate [%]"])
