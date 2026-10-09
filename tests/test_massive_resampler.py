"""The Trade bar resampler (#238): SIP trades of one session onto right-closed bars.

``TradeBarResampler`` is pure polars. What is locked here, with hand-computed
values:

- a bar ``k`` covers ``(open + (k-1)d, open + kd]`` and is labelled at its
  end: a trade exactly on a boundary belongs to the bar that ends there, a
  trade exactly at the open belongs to no bar (it is before the session),
  one exactly at the close to the last bar;
- each trade counts towards high/low, open/close and volume separately, by
  the consolidated update rules of every one of its conditions (all must
  allow it); a trade without conditions counts for all three; a condition
  without update rules restricts nothing;
- a trade with a condition the table does not know, and a corrected or
  cancelled trade (``correction`` other than 0 and 12, the corrected print),
  is dropped and counted; a null ``correction`` is a regular trade;
- open and close are the first and last eligible trade by ``sip_timestamp``,
  ties broken by ``sequence_number``;
- a bar without an eligible trade has null prices, zero volume and zero
  ``n_trades``; a bar with volume-only trades has volume but null prices;
- (#239) ``dollar_volume`` is price times size, ``offexchange_volume`` the
  size of TRF trades and ``oddlot_volume`` of trades under 100 shares,
  fractional ones included, all over the volume-eligible trades;
- (#239) ``buy_volume`` and ``sell_volume`` by the tick rule over the
  volume-eligible trades: up-tick buy, down-tick sell, a zero tick takes the
  previous non-zero tick's side, carried across bars of a session but not
  across sessions; a session's first trade, and zero ticks before any
  non-zero one, count in neither.

Every value is SYNTHETIC; condition ids, names and rules are VERBATIM from
Massive's ``/v3/reference/conditions`` (pulled 2026-10-09).
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from tests.massive_fixtures import conditions_frame

DAY = date(2024, 1, 24)
OPEN = datetime(2024, 1, 24, 14, 30)
CLOSE = datetime(2024, 1, 24, 14, 33)


def _ns(text: str, day: date = DAY) -> int:
    """Nanoseconds since the epoch of a naive-UTC ``HH:MM:SS[.f]`` on ``day``."""
    moment = datetime.fromisoformat(f"{day.isoformat()}T{text}")
    return int((moment - datetime(1970, 1, 1)).total_seconds() * 1_000_000) * 1000


def _trades(rows: list[tuple], day: date = DAY) -> pl.DataFrame:
    """``(symbol, time, price, size, conditions, correction, sequence[, trf_id])`` rows of ``day``."""
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "date": [day] * len(rows),
            "conditions": [r[4] for r in rows],
            "correction": [r[5] for r in rows],
            "price": [r[2] for r in rows],
            "sequence_number": [r[6] for r in rows],
            "sip_timestamp": [_ns(r[1], day) for r in rows],
            "size": [float(r[3]) for r in rows],
            "trf_id": [r[7] if len(r) > 7 else 0 for r in rows],
        },
        schema={
            "symbol": pl.String,
            "date": pl.Date,
            "conditions": pl.String,
            "correction": pl.Int64,
            "price": pl.Float64,
            "sequence_number": pl.Int64,
            "sip_timestamp": pl.Int64,
            "size": pl.Float64,
            "trf_id": pl.Int64,
        },
    )


def _sessions() -> pl.DataFrame:
    return pl.DataFrame(
        {"date": [DAY], "open": [OPEN], "close": [CLOSE]},
        schema={"date": pl.Date, "open": pl.Datetime("ns"), "close": pl.Datetime("ns")},
    )


def _resample(rows, interval="1m"):
    from quantlab.dataset.massive.resample import TradeBarResampler

    return TradeBarResampler(interval).resample_with_stats(_trades(rows), conditions_frame(), _sessions())


def _bar(bars: pl.DataFrame, symbol: str, label: str) -> dict:
    (row,) = bars.filter(
        pl.col("symbol") == symbol, pl.col("timestamp") == datetime.fromisoformat(f"2024-01-24T{label}")
    ).to_dicts()
    return row


def test_bars_are_right_closed_and_labelled_at_their_end():
    bars, _ = _resample(
        [
            ("AAA", "14:30:00", 9.0, 100, None, 0, 1),  # at the open: before the session
            ("AAA", "14:30:30", 10.0, 100, None, 0, 2),
            ("AAA", "14:31:00", 11.0, 200, None, 0, 3),  # on the boundary: first bar
            ("AAA", "14:31:00.000001", 12.0, 300, None, 0, 4),
            ("AAA", "14:33:00", 13.0, 400, None, 0, 5),  # at the close: last bar
            ("AAA", "14:33:00.000001", 14.0, 500, None, 0, 6),  # after the close
        ]
    )
    assert bars["timestamp"].to_list() == [
        datetime(2024, 1, 24, 14, 31),
        datetime(2024, 1, 24, 14, 32),
        datetime(2024, 1, 24, 14, 33),
    ]
    first = _bar(bars, "AAA", "14:31")
    assert (first["open"], first["high"], first["low"], first["close"]) == (10.0, 11.0, 10.0, 11.0)
    assert (first["volume"], first["n_trades"]) == (300.0, 2)
    assert _bar(bars, "AAA", "14:32")["open"] == 12.0
    assert _bar(bars, "AAA", "14:33")["close"] == 13.0


def test_an_empty_bar_has_null_prices_and_zero_volume_and_count():
    bars, _ = _resample([("AAA", "14:30:10", 10.0, 100, None, 0, 1), ("AAA", "14:32:10", 11.0, 100, None, 0, 2)])
    empty = _bar(bars, "AAA", "14:32")
    assert [empty[k] for k in ("open", "high", "low", "close")] == [None] * 4
    assert (empty["volume"], empty["n_trades"]) == (0.0, 0)
    assert bars.height == 3


def test_each_condition_rule_counts_separately():
    bars, _ = _resample(
        [
            ("AAA", "14:30:10", 10.0, 100, "14,41", 0, 1),  # sweep, trade-thru exempt: counts for all
            ("AAA", "14:30:20", 50.0, 7, "37", 0, 2),  # odd lot: volume only
            ("AAA", "14:30:30", 1.0, 10, "32", 0, 3),  # sold out of sequence: high/low and volume
            ("AAA", "14:30:40", 70.0, 20, "38", 0, 4),  # corrected consolidated close: prices, no volume
            ("AAA", "14:30:50", 99.0, 30, "16", 0, 5),  # market center official open: nothing
            ("AAA", "14:30:55", 12.0, 40, "", 0, 6),  # no condition: counts for all
            ("AAA", "14:30:58", 11.0, 5, "14,37", 0, 7),  # one condition says volume only
        ]
    )
    bar = _bar(bars, "AAA", "14:31")
    assert (bar["open"], bar["close"]) == (10.0, 12.0)  # 70.0 (38) is open/close-eligible but not last
    assert (bar["high"], bar["low"]) == (70.0, 1.0)
    assert bar["volume"] == 100 + 7 + 10 + 40 + 5
    assert bar["n_trades"] == 5


def test_a_bar_of_volume_only_trades_has_volume_but_no_prices():
    bars, _ = _resample([("AAA", "14:31:30", 10.0, 50, "12,37", 0, 1)])
    bar = _bar(bars, "AAA", "14:32")
    assert [bar[k] for k in ("open", "high", "low", "close")] == [None] * 4
    assert (bar["volume"], bar["n_trades"]) == (50.0, 1)


def test_a_condition_without_update_rules_restricts_nothing():
    bars, _ = _resample([("AAA", "14:30:10", 10.0, 100, "60", 0, 1)])  # short sale restriction in effect
    bar = _bar(bars, "AAA", "14:31")
    assert (bar["open"], bar["volume"]) == (10.0, 100.0)


def test_corrected_and_unknown_condition_trades_are_dropped_and_counted():
    bars, stats = _resample(
        [
            ("AAA", "14:30:10", 10.0, 100, None, 0, 1),
            ("AAA", "14:30:20", 99.0, 100, None, 1, 2),  # corrected
            ("AAA", "14:30:30", 98.0, 100, None, 7, 3),  # cancelled
            ("AAA", "14:30:40", 97.0, 100, "999", 0, 4),  # unknown condition
            ("AAA", "16:00:00", 96.0, 100, None, 0, 5),  # outside the session
            ("AAA", "14:30:50", 9.5, 50, None, 12, 6),  # the corrected print of the trade at 99.0
            ("AAA", "14:30:55", 9.0, 10, None, None, 7),  # no correction field: a regular trade
        ]
    )
    bar = _bar(bars, "AAA", "14:31")
    assert (bar["high"], bar["low"], bar["volume"], bar["n_trades"]) == (10.0, 9.0, 160.0, 3)
    (row,) = stats.to_dicts()
    assert row == {
        "date": DAY,
        "symbol": "AAA",
        "trades_in": 7,
        "dropped_correction": 2,
        "dropped_unknown_condition": 1,
        "outside_session": 1,
        "volume_ineligible": 0,
    }


def test_open_and_close_follow_sip_time_then_sequence_number():
    bars, _ = _resample(
        [
            ("AAA", "14:30:20", 12.0, 1, None, 0, 9),
            ("AAA", "14:30:10", 10.0, 1, None, 0, 5),
            ("AAA", "14:30:10", 11.0, 1, None, 0, 4),  # same instant, earlier in sequence
            ("AAA", "14:30:20", 13.0, 1, None, 0, 8),
        ]
    )
    bar = _bar(bars, "AAA", "14:31")
    assert (bar["open"], bar["close"]) == (11.0, 12.0)


def test_symbols_are_resampled_apart():
    bars, stats = _resample([("AAA", "14:30:10", 10.0, 1, None, 0, 1), ("BBB", "14:32:10", 20.0, 2, None, 0, 2)])
    assert bars.group_by("symbol").len().sort("symbol")["len"].to_list() == [3, 3]
    assert _bar(bars, "BBB", "14:33")["open"] == 20.0
    assert _bar(bars, "BBB", "14:31")["volume"] == 0.0
    assert stats["symbol"].to_list() == ["AAA", "BBB"]


def test_coarser_bars_cut_from_the_open():
    bars, _ = _resample(
        [("AAA", "14:30:10", 10.0, 1, None, 0, 1), ("AAA", "14:31:00", 11.0, 2, None, 0, 2)], interval="30s"
    )
    assert bars.height == 6
    assert _bar(bars, "AAA", "14:30:30")["open"] == 10.0
    assert _bar(bars, "AAA", "14:31:00")["volume"] == 2.0


def test_an_unknown_bar_interval_is_refused():
    from quantlab.dataset.massive.resample import TradeBarResampler

    with pytest.raises(ValueError, match="bar interval"):
        TradeBarResampler("7m")


# -- #239: dollar, off-exchange, odd-lot volume and the tick rule ------------


def test_the_tick_rule_signs_volume_with_zero_tick_chains_carried_across_bars():
    bars, _ = _resample(
        [
            ("AAA", "14:30:10", 10.0, 100, None, 0, 1),  # first of the session: neither
            ("AAA", "14:30:20", 10.5, 200, None, 0, 2),  # up: buy
            ("AAA", "14:30:30", 10.5, 300, None, 0, 3),  # zero, after up: buy
            ("AAA", "14:30:40", 10.2, 400, None, 0, 4),  # down: sell
            ("AAA", "14:31:10", 10.2, 500, None, 0, 5),  # zero, carried into the next bar: sell
            ("AAA", "14:31:20", 10.2, 50, None, 0, 6),  # zero again: sell
            ("AAA", "14:31:30", 10.3, 60, None, 0, 7),  # up: buy
            ("AAA", "14:31:35", 99.0, 80, "16", 0, 8),  # counts for nothing: not a tick
            ("AAA", "14:31:40", 10.1, 7, "37", 0, 9),  # odd lot, volume only: down from 10.3, sell
            ("AAA", "14:31:50", 10.1, 10, None, 0, 10),  # zero: sell
        ]
    )
    first, second = _bar(bars, "AAA", "14:31"), _bar(bars, "AAA", "14:32")
    assert (first["buy_volume"], first["sell_volume"], first["volume"]) == (500.0, 400.0, 1000.0)
    assert (second["buy_volume"], second["sell_volume"], second["volume"]) == (60.0, 567.0, 627.0)
    assert (_bar(bars, "AAA", "14:33")["buy_volume"], _bar(bars, "AAA", "14:33")["sell_volume"]) == (0.0, 0.0)


def test_zero_ticks_before_any_price_change_count_in_neither_split():
    bars, _ = _resample(
        [
            ("AAA", "14:30:10", 10.0, 100, None, 0, 1),
            ("AAA", "14:30:20", 10.0, 200, None, 0, 2),  # zero, no non-zero tick yet
            ("AAA", "14:30:30", 9.9, 300, None, 0, 3),  # down
        ]
    )
    bar = _bar(bars, "AAA", "14:31")
    assert (bar["buy_volume"], bar["sell_volume"], bar["volume"]) == (0.0, 300.0, 600.0)


def test_the_tick_resets_across_sessions():
    from quantlab.dataset.massive.resample import TradeBarResampler

    next_day = date(2024, 1, 25)
    trades = pl.concat(
        [
            _trades([("AAA", "14:30:10", 10.0, 100, None, 0, 1), ("AAA", "14:30:20", 10.5, 100, None, 0, 2)]),
            # The next session's first trade is above yesterday's last, and its
            # second a zero tick: neither counts.
            _trades([("AAA", "14:30:10", 11.0, 300, None, 0, 1), ("AAA", "14:30:20", 11.0, 400, None, 0, 2)], next_day),
        ]
    )
    sessions = pl.DataFrame(
        {"date": [DAY, next_day], "open": [OPEN, datetime(2024, 1, 25, 14, 30)],
         "close": [CLOSE, datetime(2024, 1, 25, 14, 33)]},
        schema={"date": pl.Date, "open": pl.Datetime("ns"), "close": pl.Datetime("ns")},
    )
    bars, _ = TradeBarResampler("1m").resample_with_stats(trades, conditions_frame(), sessions)
    (row,) = bars.filter(pl.col("timestamp") == datetime(2024, 1, 25, 14, 31)).to_dicts()
    assert (row["buy_volume"], row["sell_volume"], row["volume"]) == (0.0, 0.0, 700.0)
    (row,) = bars.filter(pl.col("timestamp") == datetime(2024, 1, 24, 14, 31)).to_dicts()
    assert (row["buy_volume"], row["sell_volume"]) == (100.0, 0.0)


def test_dollar_offexchange_and_fractional_oddlot_volume():
    bars, _ = _resample(
        [
            ("AAA", "14:30:10", 10.0, 100, None, 0, 1),
            ("AAA", "14:30:20", 10.5, 0.5, None, 0, 2),  # fractional: an odd lot, not truncated
            ("AAA", "14:30:30", 11.0, 200, None, 0, 3, 201),  # reported through a TRF
            ("AAA", "14:30:40", 12.0, 30, "37", 0, 4, 202),  # odd lot through a TRF
            ("AAA", "14:30:50", 50.0, 40, "16", 0, 5, 201),  # counts for nothing
        ]
    )
    bar = _bar(bars, "AAA", "14:31")
    assert bar["volume"] == 330.5
    assert bar["dollar_volume"] == 10.0 * 100 + 10.5 * 0.5 + 11.0 * 200 + 12.0 * 30
    assert bar["offexchange_volume"] == 230.0
    assert bar["oddlot_volume"] == 30.5
    empty = _bar(bars, "AAA", "14:32")
    assert [empty[k] for k in ("dollar_volume", "offexchange_volume", "oddlot_volume", "buy_volume")] == [0.0] * 4


def test_trades_excluded_from_volume_are_counted():
    _, stats = _resample(
        [
            ("AAA", "14:30:10", 10.0, 100, None, 0, 1),
            ("AAA", "14:30:20", 70.0, 20, "38", 0, 2),  # prices only
            ("AAA", "14:30:30", 99.0, 30, "16", 0, 3),  # nothing
        ]
    )
    assert stats["volume_ineligible"].to_list() == [2]


def test_the_bar_columns():
    from quantlab.dataset.massive.resample import TRADE_BAR_VARIABLES

    bars, _ = _resample([("AAA", "14:30:10", 10.0, 100, None, 0, 1)])
    assert bars.columns == ["symbol", "date", "timestamp", *TRADE_BAR_VARIABLES]
    assert TRADE_BAR_VARIABLES == (
        "open", "high", "low", "close", "volume", "dollar_volume", "n_trades",
        "buy_volume", "sell_volume", "offexchange_volume", "oddlot_volume",
    )
    assert all(bars[name].dtype == pl.Float64 for name in TRADE_BAR_VARIABLES if name != "n_trades")
