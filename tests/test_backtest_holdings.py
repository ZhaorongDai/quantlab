"""Holdings of a ``run()``: ``holdings.zarr`` and the report's Holdings tab (#225).

A *holding* is the fraction of the book, cash included, one symbol takes at a
bar's close, as the fills left it; the weight stays the target. What is
locked here, through real ``run()`` calls and the run directory they write:

- ``holdings.zarr`` holds ``holding`` on ``(timestamp, symbol)``: every bar
  of the window, the run's symbol axis, zero where nothing is held;
- holding times the book's value is the position's value, the shares the
  order records leave times the valuation price, and the holdings plus the
  cash the order records leave sum to one on every bar;
- on the bar a rebalance fills, the holdings valued at the fill prices are
  the targets (less costs), and after it they drift with prices;
- a short position is a negative holding, and a settled delisting is zero
  from its settlement bar;
- the report's Holdings tab embeds, per bar, the holdings of
  ``holdings.zarr``, the targets of the last rebalance before the bar and the
  cash, and its summary figures are the Performance tab's.

Nothing here recomputes a holding from vectorbt: the position values come
from the order records and the price store. Everything is synthetic,
CPU-only and offline.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from quantlab.utils.date_range import bar_label
from tests.backtest_fixtures import (
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
WINDOW_START = 30
WINDOW_END = 55
INIT_CASH = 1_000_000.0


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _run(root: Path, **overrides):
    """A load-mode ``run()`` over bars 30-55 of a six-symbol store."""
    bars = pd.bdate_range("2024-01-01", periods=N_BARS)
    delist_at = overrides.pop("delist_at", None)
    dataset_config = write_price_store(root / "store", n_bars=N_BARS, delist_at=delist_at)
    dates = dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )
    checkpoint = train_checkpoint(make_model(root / "train", dataset_config, **dates))
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / "backtest", dataset_config, **dates),
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(bars[WINDOW_START]),
        end_date=_day(bars[WINDOW_END]),
        output_dir=str(root / "runs"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        init_cash=INIT_CASH,
    )
    kwargs.update(overrides)
    result = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(**kwargs)).run()
    prices = xr.open_zarr(dataset_config.zarr_file_path).load()
    return result, prices


@pytest.fixture(scope="module")
def long_only(tmp_path_factory):
    """Long-only top two, no fees or slippage."""
    return _run(tmp_path_factory.mktemp("holdings_long_only"), fees=0.0, slippage=0.0)


@pytest.fixture(scope="module")
def long_short(tmp_path_factory):
    """Long-short top two with fees, CCC delisted on bar 42 while held."""
    return _run(
        tmp_path_factory.mktemp("holdings_long_short"),
        constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=2)),
        rebalance_periods=3,
        fees=0.002,
        delist_at={"CCC": 42},
    )


def _panel(dataset: xr.Dataset, name: str, bars, symbols) -> np.ndarray:
    return (
        dataset[name]
        .transpose("timestamp", "symbol")
        .sel(timestamp=bars, symbol=symbols)
        .values.astype(np.float64)
    )


def _book(result, prices: xr.Dataset):
    """Shares and cash after each bar from the order records alone, and the closes."""
    holdings = BacktestRun.open(result.run_dir).holdings()["holding"]
    bars = holdings.timestamp.values
    symbols = [str(s) for s in holdings.symbol.values]
    orders = result.simulation.orders
    shares = np.zeros((bars.size, len(symbols)))
    cash = np.zeros(bars.size)
    rows = np.searchsorted(bars, orders["timestamp"].values.astype("datetime64[ns]"))
    columns = [symbols.index(str(s)) for s in orders["symbol"].values]
    sign = np.where(orders["side"].values.astype(str) == "Buy", 1.0, -1.0)
    size = orders["size"].values
    np.add.at(shares, (rows, columns), sign * size)
    np.add.at(cash, rows, -sign * size * orders["price"].values - orders["fees"].values)
    close = pd.DataFrame(_panel(prices, "adjClose", bars, symbols)).ffill().to_numpy()
    return np.cumsum(shares, axis=0), INIT_CASH + np.cumsum(cash), close


def _open(result) -> BacktestRun:
    return BacktestRun.open(result.run_dir)


@pytest.mark.parametrize("scenario", ["long_only", "long_short"])
def test_holdings_cover_every_bar_of_the_window_on_the_run_symbol_axis(scenario, request):
    result, _ = request.getfixturevalue(scenario)
    run = _open(result)
    holdings = run.holdings()
    assert list(holdings.data_vars) == ["holding"]
    assert holdings["holding"].dims == ("timestamp", "symbol")
    np.testing.assert_array_equal(
        holdings.timestamp.values, run.equity().timestamp.values
    )
    assert holdings.symbol.values.tolist() == run.weights().symbol.values.tolist()
    assert np.isfinite(holdings["holding"].values).all()


@pytest.mark.parametrize("scenario", ["long_only", "long_short"])
def test_holding_times_value_is_the_position_value_and_holdings_plus_cash_are_one(
    scenario, request
):
    result, prices = request.getfixturevalue(scenario)
    run = _open(result)
    holding = run.holdings()["holding"].values
    value = run.equity()["value"].values
    shares, cash, close = _book(result, prices)
    position = np.where(shares != 0.0, shares * close, 0.0)
    np.testing.assert_allclose(holding * value[:, None], position, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(holding.sum(axis=1) + cash / value, 1.0, rtol=0, atol=1e-9)
    # Zero where nothing is held, exactly.
    assert (holding[np.abs(shares) < 1e-12] == 0.0).all()


def test_on_the_fill_bar_holdings_are_the_targets_and_then_drift_with_prices(long_only):
    result, prices = long_only
    run = _open(result)
    holdings = run.holdings()["holding"]
    bars = holdings.timestamp.values
    symbols = [str(s) for s in holdings.symbol.values]
    holding = holdings.values
    value = run.equity()["value"].values
    weights = run.weights()["weight"].values
    shares, cash, close = _book(result, prices)
    fill_open = _panel(prices, "adjOpen", bars, symbols)
    rebalances = np.flatnonzero(np.isfinite(weights).any(axis=1))
    assert rebalances.size >= 3
    for signal in rebalances:
        fill = signal + 1
        if fill >= bars.size:
            continue
        held_shares = holding[fill] * value[fill] / np.where(close[fill] > 0, close[fill], 1.0)
        at_open = np.where(held_shares != 0.0, held_shares * fill_open[fill], 0.0)
        book = cash[fill] + at_open.sum()
        np.testing.assert_allclose(at_open / book, weights[signal], atol=1e-9)
        # The next bar holds: the same shares, revalued at the new close.
        if fill + 1 < bars.size and not np.isfinite(weights[fill]).any():
            np.testing.assert_allclose(
                holding[fill + 1] * value[fill + 1],
                holding[fill] * value[fill] * np.where(
                    holding[fill] != 0.0, close[fill + 1] / close[fill], 0.0
                ),
                rtol=1e-9,
                atol=1e-6,
            )


def test_with_costs_the_fill_bar_holdings_miss_the_targets_by_the_costs(long_short):
    result, prices = long_short
    run = _open(result)
    holdings = run.holdings()["holding"]
    bars = holdings.timestamp.values
    symbols = [str(s) for s in holdings.symbol.values]
    holding = holdings.values
    weights = run.weights()["weight"].values
    value = run.equity()["value"].values
    _, cash, close = _book(result, prices)
    fill_open = _panel(prices, "adjOpen", bars, symbols)
    signal = int(np.flatnonzero(np.isfinite(weights).any(axis=1))[0])
    fill = signal + 1
    held_shares = holding[fill] * value[fill] / close[fill]
    at_open = np.where(held_shares != 0.0, held_shares * fill_open[fill], 0.0)
    gap = np.abs(at_open / (cash[fill] + at_open.sum()) - weights[signal])
    # Fees of 0.2% on a book of gross 1: each holding misses by a few bp.
    assert 1e-6 < gap.max() < 0.01


def test_short_positions_are_negative_holdings(long_short):
    result, _ = long_short
    run = _open(result)
    holding = run.holdings()["holding"].values
    weights = run.weights()["weight"].values
    signal = int(np.flatnonzero(np.isfinite(weights).any(axis=1))[0])
    shorts = weights[signal] < 0
    assert shorts.any()
    assert (holding[signal + 1][shorts] < 0).all()
    assert (holding[signal + 1][weights[signal] > 0] > 0).all()


def test_a_settled_delisting_is_zero_from_its_settlement_bar(long_short):
    result, _ = long_short
    run = _open(result)
    settlements = run.settlements()
    assert [record["axis_symbol"] for record in settlements] == ["CCC"]
    holdings = run.holdings()["holding"]
    settled = pd.Timestamp(settlements[0]["settlement_timestamp"])
    ccc = holdings.sel(symbol="CCC")
    assert float(ccc.sel(timestamp=settled - pd.offsets.BDay(1))) != 0.0
    assert (ccc.sel(timestamp=slice(settled, None)).values == 0.0).all()


# ---------------------------------------------------------------------------
# The report's Holdings tab
# ---------------------------------------------------------------------------

_DATA = re.compile(
    r'<script type="application/json" id="holdings-data">(.*?)</script>', re.S
)


def holdings_data(page: str) -> dict:
    """The JSON the Holdings tab embeds."""
    match = _DATA.search(page)
    assert match, "the page embeds no holdings data"
    return json.loads(match.group(1))


def _kpi(page: str, label: str) -> str:
    match = re.search(
        rf'<div class="kl">{re.escape(label)}</div><div class="kv[^"]*">([^<]*)</div>', page
    )
    assert match, label
    return match.group(1)


@pytest.mark.parametrize("scenario", ["long_only", "long_short"])
def test_the_holdings_tab_embeds_holdings_targets_and_cash_of_every_bar(scenario, request):
    result, _ = request.getfixturevalue(scenario)
    run = _open(result)
    page = run.report()
    assert '<button class="tab" data-tab="tab' in page and ">Holdings</button>" in page
    data = holdings_data(page)
    holdings = run.holdings()["holding"]
    bars = holdings.timestamp.values
    symbols = [str(s) for s in holdings.symbol.values]
    weights = run.weights()["weight"].transpose("timestamp", "symbol").values
    assert [day["d"] for day in data["days"]] == [bar_label(b) for b in bars]
    names = data["names"]
    targets = np.zeros(len(symbols))
    rebalance = None
    for i, day in enumerate(data["days"]):
        if i > 0 and np.isfinite(weights[i - 1]).any():
            targets = np.where(np.isfinite(weights[i - 1]), weights[i - 1], targets)
            rebalance = bar_label(bars[i - 1])
        assert day["r"] == rebalance
        row = holdings.values[i]
        shown = {names[k][2]: (t, h) for k, t, h in day["h"]}
        expected = {
            symbols[j]: (targets[j], row[j])
            for j in range(len(symbols))
            if row[j] != 0.0 or targets[j] != 0.0
        }
        assert shown.keys() == expected.keys()
        for symbol, (t, h) in shown.items():
            assert h == expected[symbol][1]
            assert t == expected[symbol][0]
        assert day["other"] == [0, 0, 0]
        assert day["cash"] == pytest.approx(1.0 - row.sum(), abs=1e-12)
    # No ticker sidecar beside the fixture store: each symbol is its own label.
    assert all(name[0] == name[2] and name[1] == "" for name in names)


@pytest.mark.parametrize("scenario", ["long_only", "long_short"])
def test_the_holdings_summary_figures_are_the_performance_tab_figures(scenario, request):
    result, _ = request.getfixturevalue(scenario)
    page = _open(result).report()
    summary = dict(holdings_data(page)["summary"])
    for label in ("Total return", "Annualised return", "Max drawdown"):
        assert summary[label] == _kpi(page, label)
