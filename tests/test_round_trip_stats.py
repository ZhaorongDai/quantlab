"""Round-trip trade statistics from fill records (#115).

``backtest_stats.round_trips`` turns fill records (timestamp, symbol, signed
quantity, price, fee) into position round trips, flat to flat per symbol,
and ``round_trip_stats`` summarizes them as the trade rows of a run's
``whole`` block. An executor without vectorbt (quantlab-trader) reports the
same rows from its own fills, adding the cash a position received while open
(dividends, distributions) to that round trip's PnL. What is locked here:

- **Equal to vectorbt.** On vectorbt's own fills, with no cash flows, every
  row equals what ``Portfolio.stats`` reports in the position trade view,
  on fixtures with shorts, reversals, partial closes, adds, fees and
  positions still open at the end, and on a quantlab run's ``whole`` block.
- **Gross exposure.** ``exposure_stats`` gives vectorbt's ``Max Gross
  Exposure [%]`` from the fills, the valuation prices and the cash balance.
- **Cash flows.** A flow is added to the PnL and the return of the round
  trip holding the symbol on its bar; a flow on a bar where the symbol is
  flat is refused.
"""

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt
import xarray as xr

from quantlab.runs.backtest_stats import exposure_stats, round_trip_stats, round_trips

#: The rows ``round_trip_stats`` reports, by vectorbt's metric name.
TRADE_METRICS = {
    "total_trades": "Total Trades",
    "total_closed_trades": "Total Closed Trades",
    "total_open_trades": "Total Open Trades",
    "open_trade_pnl": "Open Trade PnL",
    "win_rate": "Win Rate [%]",
    "best_trade": "Best Trade [%]",
    "worst_trade": "Worst Trade [%]",
    "avg_winning_trade": "Avg Winning Trade [%]",
    "avg_losing_trade": "Avg Losing Trade [%]",
    "avg_winning_trade_duration": "Avg Winning Trade Duration",
    "avg_losing_trade_duration": "Avg Losing Trade Duration",
    "profit_factor": "Profit Factor",
    "expectancy": "Expectancy",
}


def _portfolio(sizes: np.ndarray, close: np.ndarray, *, fees: float, freq: str):
    bars = pd.date_range("2024-01-01", periods=close.shape[0], freq=freq)
    symbols = [f"S{i}" for i in range(close.shape[1])]
    close_df = pd.DataFrame(close, index=bars, columns=symbols)
    return vbt.Portfolio.from_orders(
        close=close_df,
        price=close_df * 1.001,
        size=pd.DataFrame(sizes, index=bars, columns=symbols),
        direction="both",
        group_by=True,
        cash_sharing=True,
        fees=fees,
        init_cash=1e6,
        freq=freq,
    ), close_df


def _fills(portfolio) -> xr.Dataset:
    records = portfolio.orders.records_readable
    sign = np.where(records["Side"].astype(str) == "Buy", 1.0, -1.0)
    return xr.Dataset(
        {
            "timestamp": ("fill", pd.to_datetime(records["Timestamp"]).to_numpy()),
            "symbol": ("fill", records["Column"].astype(str).to_numpy()),
            "size": ("fill", sign * records["Size"].to_numpy(dtype=np.float64)),
            "price": ("fill", records["Price"].to_numpy(dtype=np.float64)),
            "fees": ("fill", records["Fees"].to_numpy(dtype=np.float64)),
        }
    )


def _vectorbt_rows(portfolio) -> dict:
    stats = portfolio.replace(trades_type="positions").stats(
        metrics=list(TRADE_METRICS), silence_warnings=True
    )
    return {TRADE_METRICS[key]: value for key, value in zip(TRADE_METRICS, stats.values)}


def _assert_rows_equal(got: dict, want: dict) -> None:
    assert list(got) == list(want)
    for key, value in want.items():
        if isinstance(value, float) and np.isnan(value):
            assert isinstance(got[key], float) and np.isnan(got[key]), key
        elif value is pd.NaT:
            assert got[key] is pd.NaT, key
        else:
            assert got[key] == value, (key, got[key], value)


def _random_fixture(seed: int, n_bars: int = 40, n_symbols: int = 5):
    rng = np.random.default_rng(seed)
    close = 20.0 * np.exp(np.cumsum(rng.normal(0, 0.03, (n_bars, n_symbols)), axis=0))
    sizes = np.where(
        rng.uniform(size=(n_bars, n_symbols)) < 0.35,
        np.round(rng.normal(0, 40, (n_bars, n_symbols))),
        np.nan,
    )
    sizes[sizes == 0] = np.nan
    return sizes, close


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("fees", [0.0, 0.001])
def test_random_long_short_fills_give_vectorbts_rows(seed, fees):
    sizes, close = _random_fixture(seed)
    portfolio, close_df = _portfolio(sizes, close, fees=fees, freq="1D")
    trips = round_trips(_fills(portfolio), xr.DataArray(close_df, dims=("timestamp", "symbol")))
    _assert_rows_equal(round_trip_stats(trips, bar_interval="1D"), _vectorbt_rows(portfolio))


def test_partial_closes_adds_reversals_and_open_positions_give_vectorbts_rows():
    nan = np.nan
    # S0: buy, add, trim, trim to flat; S1: short, reverse to long, still open;
    # S2: long, partial close, open at the end; S3: never traded.
    sizes = np.array(
        [
            [10, -5, 8, nan],
            [5, nan, nan, nan],
            [-6, 12, -3, nan],
            [nan, nan, nan, nan],
            [-9, nan, 4, nan],
            [nan, -2, nan, nan],
        ]
    )
    close = np.array(
        [
            [10.0, 20.0, 5.0, 7.0],
            [10.5, 19.0, 5.5, 7.1],
            [11.0, 18.0, 5.2, 7.2],
            [9.5, 18.5, 5.0, 7.3],
            [12.0, 21.0, 5.9, 7.4],
            [12.5, 22.0, 6.1, 7.5],
        ]
    )
    portfolio, close_df = _portfolio(sizes, close, fees=0.002, freq="1h")
    trips = round_trips(_fills(portfolio), xr.DataArray(close_df, dims=("timestamp", "symbol")))
    rows = round_trip_stats(trips, bar_interval="1h")
    _assert_rows_equal(rows, _vectorbt_rows(portfolio))
    assert rows["Total Open Trades"] == 2


def test_every_position_closed_gives_vectorbts_rows():
    nan = np.nan
    sizes = np.array([[5, -3], [nan, nan], [-5, 3], [2, nan], [-2, nan]])
    close = np.array([[10.0, 30.0], [9.0, 31.0], [11.0, 29.0], [12.0, 28.0], [10.0, 27.0]])
    portfolio, close_df = _portfolio(sizes, close, fees=0.001, freq="1D")
    trips = round_trips(_fills(portfolio), xr.DataArray(close_df, dims=("timestamp", "symbol")))
    rows = round_trip_stats(trips, bar_interval="1D")
    _assert_rows_equal(rows, _vectorbt_rows(portfolio))
    assert rows["Total Open Trades"] == 0


def test_no_fills_give_vectorbts_rows():
    sizes = np.full((5, 2), np.nan)
    close = np.linspace(10.0, 11.0, 10).reshape(5, 2)
    portfolio, close_df = _portfolio(sizes, close, fees=0.0, freq="1D")
    trips = round_trips(_fills(portfolio), xr.DataArray(close_df, dims=("timestamp", "symbol")))
    _assert_rows_equal(round_trip_stats(trips, bar_interval="1D"), _vectorbt_rows(portfolio))


def test_a_quantlab_runs_own_fills_give_its_whole_blocks_trade_rows(tmp_path):
    from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
    from quantlab.backtest.config import CrossSectionBacktestConfig
    from quantlab.portfolio.config import TopNConfig
    from quantlab.portfolio.predefined.top_n import TopNConstructor
    from tests.backtest_fixtures import make_stock_dataset, write_price_store

    dataset_config = write_price_store(tmp_path / "store", n_bars=60, delist_at={"CCC": 42})
    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            start_date="2024-02-12",
            end_date="2024-03-18",
            output_dir=None,
            rebalance_periods=3,
            constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=2)),
            fees=0.002,
        )
    )
    prices = xr.open_zarr(dataset_config.zarr_file_path).sel(
        timestamp=slice("2024-02-12", "2024-03-18")
    )
    bars, symbols = prices.timestamp.values, list(prices.symbol.values)
    rng = np.random.default_rng(3)
    rows = np.full((bars.size, len(symbols)), np.nan)
    for i in range(0, bars.size, 4):
        row = rng.normal(size=len(symbols))
        rows[i] = row / np.abs(row).sum()
    result = backtester.run_weights(
        xr.DataArray(rows, dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": symbols})
    )
    orders = result.simulation.orders
    fills = orders.assign(
        size=orders["size"] * xr.where(orders["side"] == "Buy", 1.0, -1.0)
    )
    close = prices[backtester.MARKET.valuation_price_column].load()
    trips = round_trips(fills, close)
    whole = result.metrics["whole"]
    want = {key: whole[key] for key in TRADE_METRICS.values()}
    _assert_rows_equal(round_trip_stats(trips, bar_interval=result.simulation.bar_interval), want)
    assert whole["Total Open Trades"] > 0
    cash = result.simulation.native.cash()
    exposure = exposure_stats(
        fills, close, xr.DataArray(cash.to_numpy(), dims="timestamp", coords={"timestamp": bars})
    )
    np.testing.assert_allclose(
        exposure["Max Gross Exposure [%]"], whole["Max Gross Exposure [%]"], rtol=1e-12
    )
    assert (trips["direction"] == "Short").any()


def _one_symbol(sizes, prices, flows=None):
    bars = pd.bdate_range("2024-01-01", periods=5)
    close = xr.DataArray(
        np.array([10.0, 11.0, 12.0, 13.0, 14.0])[:, None],
        dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": ["AAA"]},
    )
    at = [i for i, size in enumerate(sizes) if size]
    fills = xr.Dataset({
        "timestamp": ("fill", bars[at].values),
        "symbol": ("fill", ["AAA"] * len(at)),
        "size": ("fill", [float(sizes[i]) for i in at]),
        "price": ("fill", [prices[i] for i in at]),
        "fees": ("fill", [0.0] * len(at)),
    })
    cash_flows = None
    if flows is not None:
        cash_flows = xr.Dataset({
            "timestamp": ("flow", bars[[bar for bar, _ in flows]].values),
            "symbol": ("flow", ["AAA"] * len(flows)),
            "amount": ("flow", [amount for _, amount in flows]),
        })
    return round_trips(fills, close, cash_flows=cash_flows)


def test_a_dividend_received_while_open_counts_in_that_round_trips_pnl_and_return():
    # Long 10 at 10 on bar 0, flat at 12 on bar 2; short 5 at 13 on bar 3, open.
    trips = _one_symbol([10, 0, -10, -5, 0], [10.0, 0, 12.0, 13.0, 0], flows=[(1, 3.0), (2, 2.0), (4, -1.0)])
    assert trips["status"].values.tolist() == ["Closed", "Open"]
    assert trips["cash_flow"].values.tolist() == [5.0, -1.0]
    assert trips["pnl"].values.tolist() == [25.0, -6.0]
    np.testing.assert_allclose(trips["return"].values, [0.25, -6.0 / 65.0])


def test_a_cash_flow_on_a_flat_bar_is_refused():
    with pytest.raises(ValueError, match="flat"):
        _one_symbol([10, -10, 0, 0, 0], [10.0, 11.0, 0, 0, 0], flows=[(3, 1.0)])


def test_a_fill_off_the_close_axis_is_refused():
    close = xr.DataArray(
        [[10.0], [11.0]], dims=("timestamp", "symbol"),
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=2), "symbol": ["AAA"]},
    )
    fills = xr.Dataset({
        "timestamp": ("fill", pd.to_datetime(["2024-01-05"]).values),
        "symbol": ("fill", ["AAA"]), "size": ("fill", [1.0]),
        "price": ("fill", [10.0]), "fees": ("fill", [0.0]),
    })
    with pytest.raises(ValueError, match="not a bar of close"):
        round_trips(fills, close)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_target_percent_long_short_fills_give_vectorbts_max_gross_exposure(seed):
    rng = np.random.default_rng(seed)
    n_bars, n_symbols = 40, 5
    bars = pd.date_range("2024-01-01", periods=n_bars, freq="1D")
    symbols = [f"S{i}" for i in range(n_symbols)]
    close = pd.DataFrame(
        20.0 * np.exp(np.cumsum(rng.normal(0, 0.03, (n_bars, n_symbols)), axis=0)),
        index=bars, columns=symbols,
    )
    targets = np.full((n_bars, n_symbols), np.nan)
    for i in range(0, n_bars, 4):
        row = rng.normal(size=n_symbols)
        targets[i] = row / np.abs(row).sum() * 0.95
    portfolio = vbt.Portfolio.from_orders(
        close=close, size=pd.DataFrame(targets, index=bars, columns=symbols),
        size_type="targetpercent", direction="both", group_by=True, cash_sharing=True,
        call_seq="auto", fees=0.001, init_cash=1e6, freq="1D",
    )
    cash = xr.DataArray(portfolio.cash().to_numpy(), dims="timestamp", coords={"timestamp": bars})
    got = exposure_stats(_fills(portfolio), xr.DataArray(close, dims=("timestamp", "symbol")), cash)
    want = portfolio.stats(metrics=["max_gross_exposure"], silence_warnings=True)
    assert list(got) == ["Max Gross Exposure [%]"]
    np.testing.assert_allclose(got["Max Gross Exposure [%]"], want.iloc[0], rtol=1e-12)


def test_a_cash_flow_on_the_entry_bar_is_not_the_new_round_trips():
    # Bought at bar 2's open: a dividend going ex on bar 2 was earned by
    # whoever held into bar 2, not by this position.
    with pytest.raises(ValueError, match="flat"):
        _one_symbol([0, 0, 10, 0, 0], [0, 0, 12.0, 0, 0], flows=[(2, 1.0)])


def test_on_a_reversal_bar_the_cash_flow_goes_to_the_round_trip_that_ends():
    trips = _one_symbol([10, 0, -15, 0, 0], [10.0, 0, 12.0, 0, 0], flows=[(2, 4.0)])
    assert trips["direction"].values.tolist() == ["Long", "Short"]
    assert trips["cash_flow"].values.tolist() == [4.0, 0.0]


def test_exposure_of_a_symbol_off_the_close_axis_is_refused():
    close = xr.DataArray(
        [[10.0], [11.0]], dims=("timestamp", "symbol"),
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=2), "symbol": ["AAA"]},
    )
    fills = xr.Dataset({
        "timestamp": ("fill", close.timestamp.values[:1]),
        "symbol": ("fill", ["ZZZ"]), "size": ("fill", [1.0]), "price": ("fill", [10.0]),
    })
    cash = xr.DataArray([0.0, 0.0], dims="timestamp", coords={"timestamp": close.timestamp.values})
    with pytest.raises(ValueError, match="not a symbol of close"):
        exposure_stats(fills, close, cash)
