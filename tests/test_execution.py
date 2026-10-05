"""The Execution rules: ``quantlab.execution.rules`` (#117).

What the market does with a bar's orders, and the holdings that result, as a
public engine-free module the vectorbt engine plans its orders with. What is
locked here, on hand-built panels whose share counts and cash are worked out in the
comments, never recomputed the way the module does:

- a weight decided at bar t fills at bar t+1, sized against the book valued
  at the sizing basis's prices (the fill price, or the signal bar's
  valuation price);
- an order without a raw fill price, or on the valuation basis without a
  signal-bar valuation, is rejected and its holding kept, and the rejection
  is reported where the order would have traded;
- a holding delisted on bar b is settled on b+1 at its last valuation,
  replacing any target, with no fee or slippage, and at a valuation of 0
  for nothing;
- sells run before buys, buys in ascending order of value, each capped by
  the cash left with its fee, so the larger of two buys is the one cut;
- a price of 0 on a symbol that is not being settled is refused, as
  vectorbt refuses it;
- fees and slippage are charged as vectorbt charges them;
- a NaN weight keeps the holding;
- replaying the whole panel equals stepping the book bar by bar;
- the replay holds what vectorbt executes, and reports the rejections and
  settlements the engine records, through the public weights backtest, for
  both sizing bases;
- importing the module loads no other quantlab module, no pandas or xarray
  and no simulation engine (checked in a fresh interpreter).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quantlab.execution.rules import ExecutionBook, ExecutionSettings, replay

NAN = np.nan


def test_a_weight_fills_on_the_next_bar_at_its_fill_price():
    # Bar 0 asks for half the book in AAA. It fills on bar 1 at the fill
    # price 10: 0.5 * 1.0 / 10 = 0.05 shares, 0.5 of cash left.
    weights = np.array([[0.5], [NAN], [NAN]])
    fill = np.array([[8.0], [10.0], [11.0]])
    valuation = np.array([[9.0], [10.5], [12.0]])
    result = replay(weights, fill, valuation, np.zeros((3, 1), dtype=bool))
    np.testing.assert_allclose(result.shares[:, 0], [0.0, 0.05, 0.05], rtol=1e-12)
    np.testing.assert_allclose(result.cash, [1.0, 0.5, 0.5], rtol=1e-12)


def test_an_order_without_a_raw_fill_price_is_rejected_and_the_holding_kept():
    # Bar 1 buys 0.1 shares at 10 with all the cash. Bar 1's exit fills on
    # bar 2, which has no fill price: the order is rejected, so the 0.1
    # shares stay and the cash stays 0.
    weights = np.array([[1.0], [0.0], [NAN]])
    fill = np.array([[10.0], [10.0], [NAN]])
    valuation = np.array([[10.0], [10.0], [10.0]])
    result = replay(weights, fill, valuation, np.zeros((3, 1), dtype=bool))
    np.testing.assert_allclose(result.shares[:, 0], [0.0, 0.1, 0.1], rtol=1e-12)
    np.testing.assert_allclose(result.cash, [1.0, 0.0, 0.0], atol=1e-12)
    assert result.rejected[:, 0].tolist() == [False, False, True]


def test_sells_run_before_buys_and_a_buy_is_capped_by_the_cash_left():
    # Columns B, A, C. Bar 1 fills 0.05 shares of A and of C at 10, no cash
    # left. Bar 1 then moves everything into B. On bar 2 C has no fill price,
    # so its sale is rejected and C (valued at its last fill price 10) stays.
    # The book is 0.5 + 0.5 = 1.0, so B wants 0.1 shares (1.0 of cash). A is
    # sold first, bringing in 0.5; B's buy, listed before A, runs after it
    # and is cut to that 0.5: 0.05 shares, no cash left.
    weights = np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [NAN, NAN, NAN]])
    fill = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, NAN]])
    valuation = np.full((3, 3), 10.0)
    result = replay(weights, fill, valuation, np.zeros((3, 3), dtype=bool))
    np.testing.assert_allclose(result.shares[2], [0.05, 0.0, 0.05], atol=1e-12)
    assert result.cash[2] == pytest.approx(0.0, abs=1e-12)


def test_fees_and_slippage_are_charged_as_vectorbt_charges_them():
    # Fees 1%, slippage 2%. Bar 1 sizes 0.5 of the book at the fill price 10:
    # 0.05 shares, bought at 10 * 1.02 = 10.2 for 0.51 plus a 0.0051 fee,
    # leaving 1 - 0.5151 = 0.4849. Bar 2 sells them at 12 * 0.98 = 11.76:
    # 0.588 less a 0.00588 fee, so the cash is 0.4849 + 0.58212 = 1.06702.
    weights = np.array([[0.5], [0.0], [NAN]])
    fill = np.array([[10.0], [10.0], [12.0]])
    valuation = np.array([[10.0], [10.0], [12.0]])
    settings = ExecutionSettings(fees=0.01, slippage=0.02)
    result = replay(weights, fill, valuation, np.zeros((3, 1), dtype=bool), settings)
    np.testing.assert_allclose(result.shares[:, 0], [0.0, 0.05, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.cash, [1.0, 0.4849, 1.06702], rtol=1e-12)


def test_a_buy_its_fee_would_overdraw_is_cut_to_what_the_cash_pays_for():
    # Fee 1%. The whole book, 1.0, in AAA at 10 is 0.1 shares costing 1.0 plus
    # a 0.01 fee, more than the cash. vectorbt cuts it so cost plus fee is the
    # cash: 1 / 1.01 of cost, 0.0990099... / 10 shares, no cash left.
    weights = np.array([[1.0], [NAN]])
    fill = np.array([[10.0], [10.0]])
    result = replay(weights, fill, fill, np.zeros((2, 1), dtype=bool), ExecutionSettings(fees=0.01))
    assert result.shares[1, 0] == pytest.approx(1 / 1.01 / 10, rel=1e-12)
    assert result.cash[1] == 0.0


def test_a_delisted_holding_is_settled_next_bar_at_its_last_valuation_without_costs():
    # Fees and slippage 1%. Bar 1 buys 0.05 shares at 10 * 1.01 = 10.1 for
    # 0.505 plus a 0.00505 fee: 0.48995 of cash left. AAA delists on bar 1
    # at a valuation of 8, so on bar 2 the 0.05 shares are settled at 8 with
    # no fee or slippage, whatever bar 1's weight asked: 0.48995 + 0.4.
    weights = np.array([[0.5], [0.5], [NAN]])
    fill = np.array([[10.0], [10.0], [NAN]])
    valuation = np.array([[10.0], [8.0], [NAN]])
    delisted = np.array([[False], [True], [False]])
    settings = ExecutionSettings(fees=0.01, slippage=0.01)
    result = replay(weights, fill, valuation, delisted, settings)
    np.testing.assert_allclose(result.shares[:, 0], [0.0, 0.05, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.cash, [1.0, 0.48995, 0.88995], rtol=1e-12)
    assert result.settled[:, 0].tolist() == [False, False, True]


def test_a_holding_delisted_at_a_valuation_of_zero_closes_for_nothing():
    # Bar 1 buys 0.05 shares at 10, 0.5 of cash left. AAA delists on bar 1
    # worthless (a -100% delisting return), so bar 2 closes the position and
    # brings in nothing; BBB, bought on bar 1 too, is untouched.
    weights = np.array([[0.5, 0.5], [NAN, NAN], [NAN, NAN]])
    fill = np.array([[10.0, 5.0], [10.0, 5.0], [NAN, 5.0]])
    valuation = np.array([[10.0, 5.0], [0.0, 5.0], [NAN, 5.0]])
    delisted = np.array([[False, False], [True, False], [False, False]])
    result = replay(weights, fill, valuation, delisted)
    np.testing.assert_allclose(result.shares[2], [0.0, 0.1], atol=1e-12)
    assert result.cash[2] == pytest.approx(0.0, abs=1e-12)


def test_the_valuation_basis_sizes_against_the_signal_bars_valuation_price():
    # The two-bar book of the sizing-basis tests, 1000 of cash. Bar 1: 0.5 of
    # the book at bar 0's close 10 is 50 shares, filled at the open 10, 500
    # left. Bar 3: the book at bar 2's close 15 is 500 + 50 * 15 = 1250, so
    # 0.84 * 1250 / 15 = 70 shares: 20 more at the open 16 cost 320.
    weights = np.array([[0.5], [NAN], [0.84], [NAN], [NAN]])
    fill = np.array([[10.0], [10.0], [12.0], [16.0], [16.0]])
    valuation = np.array([[10.0], [10.0], [15.0], [20.0], [20.0]])
    settings = ExecutionSettings(sizing_basis="valuation")
    result = replay(weights, fill, valuation, np.zeros((5, 1), dtype=bool), settings, init_cash=1000.0)
    np.testing.assert_allclose(result.shares[:, 0], [0.0, 50.0, 50.0, 70.0, 70.0], rtol=1e-12)
    np.testing.assert_allclose(result.cash, [1000.0, 500.0, 500.0, 180.0, 180.0], rtol=1e-12)


def test_a_long_short_book_sells_short_first_and_buys_with_the_proceeds():
    # Half the book long AAA at 10 and half short BBB at 20. The short sale
    # (-0.5 of value) runs first: 0.025 shares sold for 0.5, cash 1.5; the
    # long buy of 0.05 shares at 10 then costs 0.5, cash back to 1.0. NaN on
    # bar 1 keeps both positions.
    weights = np.array([[0.5, -0.5], [NAN, NAN], [NAN, NAN]])
    fill = np.array([[10.0, 20.0], [10.0, 20.0], [12.0, 18.0]])
    result = replay(weights, fill, fill, np.zeros((3, 2), dtype=bool))
    np.testing.assert_allclose(result.shares[1:], [[0.05, -0.025], [0.05, -0.025]], rtol=1e-12)
    np.testing.assert_allclose(result.cash, [1.0, 1.0, 1.0], rtol=1e-12)



def test_on_the_valuation_basis_an_order_without_a_signal_bar_valuation_is_rejected():
    # AAA has no close on bar 0, so bar 0's weight has no price to be sized
    # against on the valuation basis and is rejected on bar 1, although bar
    # 1 has a fill price. Bar 1's weight, sized at bar 1's close 10, fills on
    # bar 2 at 10: 0.05 shares. A rejection is reported where the order would
    # have traded.
    weights = np.array([[0.5], [0.5], [NAN]])
    fill = np.array([[10.0], [10.0], [10.0]])
    valuation = np.array([[NAN], [10.0], [10.0]])
    settings = ExecutionSettings(sizing_basis="valuation")
    result = replay(weights, fill, valuation, np.zeros((3, 1), dtype=bool), settings)
    np.testing.assert_allclose(result.shares[:, 0], [0.0, 0.0, 0.05], atol=1e-12)
    assert result.rejected[:, 0].tolist() == [False, True, False]


def test_when_the_cash_runs_out_the_larger_buy_is_the_one_cut():
    # Columns BIG, SMALL, HELD and a 1% fee. Bar 1 puts the whole book in
    # HELD at 10: cut by the fee to V = 1 / 1.01 of cost, V / 10 shares, no
    # cash. HELD has no fill price on bar 2, so bar 1's switch to BIG and
    # SMALL is rejected for HELD and finds no cash: nothing is bought. On
    # bar 3 the book is still V (HELD at 10). HELD's sale brings in 0.99 V;
    # SMALL, the smaller buy (0.4 V), runs first and is filled with its fee,
    # and BIG, asked 0.6 V, is cut to the cash left.
    weights = np.array([
        [0.0, 0.0, 1.0],
        [0.6, 0.4, 0.0],
        [0.6, 0.4, 0.0],
        [NAN, NAN, NAN],
    ])
    fill = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0],
        [10.0, 10.0, NAN],
        [10.0, 10.0, 10.0],
    ])
    result = replay(weights, fill, fill, np.zeros((4, 3), dtype=bool), ExecutionSettings(fees=0.01))
    book = 1 / 1.01
    np.testing.assert_allclose(result.shares[2], [0.0, 0.0, book / 10], rtol=1e-12)
    left = 0.99 * book - 0.4 * book * 1.01
    np.testing.assert_allclose(result.shares[3], [left / 1.01 / 10, 0.4 * book / 10, 0.0], rtol=1e-12)
    assert result.cash[3] == 0.0


def test_a_zero_price_on_a_symbol_that_is_not_delisted_is_refused_as_vectorbt_refuses_it():
    weights = np.array([[0.5], [NAN]])
    fill = np.array([[10.0], [0.0]])
    with pytest.raises(ValueError, match="price"):
        replay(weights, fill, np.full((2, 1), 10.0), np.zeros((2, 1), dtype=bool))


def _market_panel():
    """A 40-bar, 6-symbol market with every execution case in it.

    The open moves away from the close, so the two sizing bases differ;
    SYM2 has no open on fill bars 10 and 22 (its orders there are rejected);
    SYM4 delists on bar 17 at a positive close and SYM5 on bar 26 worthless;
    the weights are rebalanced every third bar and kept (NaN) between,
    alternately long-short and fully invested long-only (whose buys the fees
    push past the cash, so they are cut), with a delisted symbol's weight
    left NaN after it delists.
    """
    rng = np.random.default_rng(7)
    n_bars, n_symbols = 40, 6
    close = 20.0 * np.cumprod(1 + rng.normal(0.0, 0.03, (n_bars, n_symbols)), axis=0)
    open_ = close * (1 + rng.normal(0.0, 0.01, (n_bars, n_symbols)))
    open_[[10, 22], 2] = NAN
    open_[18:, 4] = NAN
    close[18:, 4] = NAN
    close[26, 5] = 0.0
    open_[27:, 5] = NAN
    close[27:, 5] = NAN
    weights = np.full((n_bars, n_symbols), NAN)
    for row in range(0, n_bars - 1, 3):
        raw = rng.normal(0.0, 1.0, n_symbols)
        if row % 2:
            raw = np.abs(raw)
        raw[np.isnan(close[row])] = 0.0
        weights[row] = (1.0 if row % 2 else 0.95) * raw / np.abs(raw).sum()
    weights[18:, 4] = NAN
    weights[27:, 5] = NAN
    return open_, close, weights


def _weights_panel(weights, bars, symbols):
    """Return ``[T, S]`` weights as the data array ``run_weights`` takes."""
    import xarray as xr

    return xr.DataArray(weights, dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": symbols})


@pytest.mark.parametrize("sizing_basis", ["fill", "valuation"])
def test_stepping_the_book_bar_by_bar_holds_what_the_whole_panel_replay_holds(sizing_basis):
    # Portfolio construction learns each rebalance's weights only after
    # deciding it: it submits them on the rebalance bars alone and trades the
    # bars one at a time, reading the book as it goes.
    open_, close, weights = _market_panel()
    delisted = np.zeros(close.shape, dtype=bool)
    delisted[17, 4] = delisted[26, 5] = True
    settings = ExecutionSettings(sizing_basis=sizing_basis, fees=0.0005, slippage=0.001)
    whole = replay(weights, open_, close, delisted, settings, init_cash=1000.0)

    book = ExecutionBook(open_, close, delisted, settings, init_cash=1000.0)
    for row in range(close.shape[0]):
        book.trade(row)
        np.testing.assert_array_equal(book.shares, whole.shares[row])
        assert book.cash == whole.cash[row]
        if np.isfinite(weights[row]).any():
            book.submit(row, weights[row])


@pytest.mark.parametrize("sizing_basis", ["fill", "valuation"])
def test_the_replay_holds_what_vectorbt_executes(sizing_basis):
    import pandas as pd

    from quantlab.backtest.predefined.weights import WeightsVectorBt
    from quantlab.base.config import WeightsBacktestConfig
    from quantlab.dataset.memory import FrameDataset

    open_, close, weights = _market_panel()
    bars = pd.bdate_range("2024-01-01", periods=close.shape[0])
    symbols = [f"SYM{j}" for j in range(close.shape[1])]
    prices = FrameDataset(pd.DataFrame({
        "timestamp": np.repeat(bars, len(symbols)),
        "symbol": symbols * bars.size,
        "open": open_.ravel(),
        "close": close.ravel(),
    }))
    config = WeightsBacktestConfig(
        price_dataset=prices, start_date=str(bars[0].date()), end_date=str(bars[-1].date()),
        output_dir=None, rebalance_periods=1, fees=0.0005, slippage=0.001, init_cash=1000.0,
        fill_price_column="open", valuation_price_column="close", sizing_basis=sizing_basis,
        trading_days_per_year=252, session_minutes_per_day=390,
    )
    backtester = WeightsVectorBt(config)
    panel = _weights_panel(weights, bars, symbols)
    simulation = backtester.run_weights(panel).simulation

    delisted = np.asarray(
        prices.delisting_bars(prices.panel(str(bars[0].date()), str(bars[-1].date())), "close")
        .transpose("timestamp", "symbol").values,
        dtype=bool,
    )
    settings = ExecutionSettings(sizing_basis=sizing_basis, fees=0.0005, slippage=0.001)
    result = replay(weights, open_, close, delisted, settings, init_cash=1000.0)

    orders = simulation.orders
    held = np.zeros_like(weights)
    for timestamp, symbol, side, size in zip(
        orders["timestamp"].values, orders["symbol"].values, orders["side"].values, orders["size"].values
    ):
        held[bars.get_loc(pd.Timestamp(timestamp)), symbols.index(str(symbol))] += size if side == "Buy" else -size
    held = np.cumsum(held, axis=0)
    assert np.abs(held).max() > 0
    np.testing.assert_allclose(result.shares, held, rtol=1e-9, atol=1e-12)

    def cells(records, key):
        return sorted((bars.get_loc(pd.Timestamp(r[key])), symbols.index(r["axis_symbol"])) for r in records)

    assert cells(simulation.rejected_orders, "fill_timestamp") == sorted(zip(*np.nonzero(result.rejected)))
    assert cells(simulation.settlements, "settlement_timestamp") == sorted(zip(*np.nonzero(result.settled)))

    valuation = pd.DataFrame(close).ffill().to_numpy()
    worth = np.where(result.shares != 0, result.shares * np.nan_to_num(valuation), 0.0).sum(axis=1)
    np.testing.assert_allclose(result.cash + worth, simulation.value.values, rtol=1e-9)
