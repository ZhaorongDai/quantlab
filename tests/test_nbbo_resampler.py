"""Hand-computed locks on `NbboResampler` semantics (phase 03.9, plan 05).

Every expected time-weighted value is written as an explicit duration-weighted
sum so a reader can re-derive it from the fixture rows. The fixtures are plain
polars frames; nothing here touches WRDS fakes or Zarr.

Both fixture days (2016-12-07, 2024-01-24) are EST days: 09:30 ET = 14:30Z.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import polars as pl
import pytest

from quantlab.base.config import NbboDatasetConfig
from quantlab.dataset.nbbo_resample import NbboFilterPolicy, NbboResampler

DAY = "2024-01-24"
_EST_OFFSET_HOURS = 5

_STATS_COUNTS = (
    "records_in",
    "dropped_nonpositive_price",
    "dropped_condition",
    "dropped_crossed",
    "dropped_locked",
    "one_sided_kept",
    "both_null_kept",
)


# ---------------------------------------------------------------- helpers ---


def _ts_ns(day: str, clock: str) -> int:
    """ET wall clock `HH:MM:SS[.fraction]` on an EST `day` -> naive-UTC ns."""
    hours, minutes, rest = clock.split(":")
    seconds, _, fraction = rest.partition(".")
    midnight = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    whole = (
        (int(hours) + _EST_OFFSET_HOURS) * 3600 + int(minutes) * 60 + int(seconds)
    )
    return (
        int(midnight.timestamp()) * 1_000_000_000
        + whole * 1_000_000_000
        + (int(fraction.ljust(9, "0")) if fraction else 0)
    )


def _q(
    clock: str,
    bid: float | None,
    bid_size: float | None,
    ask: float | None,
    ask_size: float | None,
    *,
    symbol: str = "AAPL",
    day: str = DAY,
    ordinal: int | None = None,
    qu_cond: str = "R",
) -> dict:
    return {
        "symbol": symbol,
        "date": date.fromisoformat(day),
        "ts_ns": _ts_ns(day, clock),
        "wrds_row_ord": ordinal,
        "best_bid": bid,
        "best_bidsizeshares": bid_size,
        "best_ask": ask,
        "best_asksizeshares": ask_size,
        "qu_cond": qu_cond,
    }


def _frame(quotes: list[dict]) -> pl.DataFrame:
    """Records frame; an unset ordinal is the row's position (arrival order)."""
    rows = [
        {**quote, "wrds_row_ord": index if quote["wrds_row_ord"] is None else quote["wrds_row_ord"]}
        for index, quote in enumerate(quotes)
    ]
    frame = pl.DataFrame(
        rows,
        schema={
            "symbol": pl.String,
            "date": pl.Date,
            "ts_ns": pl.Int64,
            "wrds_row_ord": pl.Int64,
            "best_bid": pl.Float64,
            "best_bidsizeshares": pl.Float64,
            "best_ask": pl.Float64,
            "best_asksizeshares": pl.Float64,
            "qu_cond": pl.String,
        },
    )
    return frame.with_columns(
        pl.col("ts_ns").cast(pl.Datetime("ns")).alias("timestamp")
    ).drop("ts_ns")


def _sessions(*days: str, close: str = "16:00:00") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date.fromisoformat(day) for day in days],
            "open": [_ts_ns(day, "09:30:00") for day in days],
            "close": [_ts_ns(day, close) for day in days],
        },
        schema={"date": pl.Date, "open": pl.Int64, "close": pl.Int64},
    ).with_columns(
        pl.col("open").cast(pl.Datetime("ns")),
        pl.col("close").cast(pl.Datetime("ns")),
    )


def _label(day: str, clock: str) -> datetime:
    """Bar label for ET `clock` on `day` as a naive-UTC `datetime`."""
    return datetime.fromtimestamp(
        _ts_ns(day, clock) / 1_000_000_000, tz=timezone.utc
    ).replace(tzinfo=None)


def _bar(panel: pl.DataFrame, clock: str, *, symbol: str = "AAPL", day: str = DAY) -> dict:
    rows = panel.filter(
        (pl.col("symbol") == symbol) & (pl.col("timestamp") == _label(day, clock))
    )
    assert rows.height == 1, f"expected one bar at {day} {clock} {symbol}, got {rows.height}"
    return rows.row(0, named=True)


def _stats_row(stats: pl.DataFrame, *, symbol: str = "AAPL", day: str = DAY) -> dict:
    rows = stats.filter(
        (pl.col("symbol") == symbol) & (pl.col("date") == date.fromisoformat(day))
    )
    assert rows.height == 1
    return rows.row(0, named=True)


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _run(quotes, policy=None, *, bar_interval="1m", sessions=None):
    resampler = NbboResampler(bar_interval, policy=policy)
    return resampler.resample_with_stats(
        _frame(quotes), sessions if sessions is not None else _sessions(DAY)
    )


#: The Task-1 seed: two-sided 100.00 x 100 / 100.02 x 100 before the open.
SEED = _q("09:29:00", 100.00, 100, 100.02, 100)


# ------------------------------------------- Task 1: filters and NULL sides ---


def test_null_ask_side_is_nan_and_tw_divides_by_each_variables_own_coverage() -> None:
    panel, stats = _run(
        [
            SEED,
            # Ask side NULL; the size WRDS sends alongside it must not become 0.
            _q("09:30:20", 100.01, 200, None, 0),
            _q("09:30:40", 100.00, 100, 100.04, 100),
        ]
    )
    bar = _bar(panel, "09:31:00")
    assert bar["n_updates"] == 2
    assert bar["bid"] == pytest.approx(100.00, abs=1e-9)
    assert bar["ask"] == pytest.approx(100.04, abs=1e-9)
    # Spread is defined for 20 s of seed + 20 s of the last record; the
    # ask-less 20 s in between is excluded, not counted as a zero spread.
    assert bar["tw_spread"] == pytest.approx((0.02 * 20 + 0.04 * 20) / 40, abs=1e-9)
    assert bar["tw_bid_size"] == pytest.approx(
        (100 * 20 + 200 * 20 + 100 * 20) / 60, abs=1e-9
    )
    assert bar["tw_ask_size"] == pytest.approx((100 * 20 + 100 * 20) / 40, abs=1e-9)
    row = _stats_row(stats)
    assert row["one_sided_kept"] == 1
    assert row["both_null_kept"] == 0
    assert row["records_in"] == 3


def test_null_side_as_last_record_leaves_that_side_and_derived_values_nan() -> None:
    panel, _ = _run([SEED, _q("09:30:30", 100.01, 300, None, None)])
    bar = _bar(panel, "09:31:00")
    for name in ("ask", "ask_size", "mid", "spread", "spread_bps", "imbalance"):
        assert _missing(bar[name]), name
    assert bar["bid"] == pytest.approx(100.01, abs=1e-9)
    assert bar["bid_size"] == pytest.approx(300, abs=1e-9)
    # The next, empty bar carries the one-sided state forward.
    carried = _bar(panel, "09:32:00")
    assert _missing(carried["ask"]) and carried["bid"] == pytest.approx(100.01)
    assert carried["n_updates"] == 0
    # tw_spread of the empty bar has zero coverage -> missing, not 0.
    assert _missing(carried["tw_spread"])


def test_both_sides_null_record_is_kept_as_a_state_with_every_price_nan() -> None:
    panel, stats = _run([SEED, _q("09:30:30", None, None, None, None)])
    bar = _bar(panel, "09:31:00")
    assert bar["n_updates"] == 1
    for name in (
        "bid", "ask", "bid_size", "ask_size", "mid", "spread", "spread_bps", "imbalance"
    ):
        assert _missing(bar[name]), name
    # Only the seed's 30 s carried a spread.
    assert bar["tw_spread"] == pytest.approx(0.02 * 30 / 30, abs=1e-9)
    assert _stats_row(stats)["both_null_kept"] == 1


def test_crossed_quote_is_dropped_by_default_and_leaves_previous_state() -> None:
    crossed = _q("09:30:30", 101.00, 100, 100.50, 100)
    baseline, _ = _run([SEED])
    panel, stats = _run([SEED, crossed])
    assert panel.equals(baseline)
    assert _stats_row(stats)["dropped_crossed"] == 1


def test_crossed_quote_is_kept_when_drop_crossed_is_off() -> None:
    crossed = _q("09:30:30", 101.00, 100, 100.50, 100)
    panel, stats = _run([SEED, crossed], NbboFilterPolicy(drop_crossed=False))
    bar = _bar(panel, "09:31:00")
    assert bar["n_updates"] == 1
    assert bar["spread"] == pytest.approx(-0.50, abs=1e-9)
    assert bar["tw_spread"] == pytest.approx((0.02 * 30 + -0.50 * 30) / 60, abs=1e-9)
    assert _stats_row(stats)["dropped_crossed"] == 0


def test_locked_quote_is_kept_by_default() -> None:
    panel, stats = _run([SEED, _q("09:30:30", 100.01, 100, 100.01, 100)])
    bar = _bar(panel, "09:31:00")
    assert bar["n_updates"] == 1
    assert bar["spread"] == pytest.approx(0.0, abs=1e-9)
    assert bar["tw_spread"] == pytest.approx((0.02 * 30 + 0.0 * 30) / 60, abs=1e-9)
    assert _stats_row(stats)["dropped_locked"] == 0


def test_locked_quote_is_dropped_when_drop_locked_is_on() -> None:
    locked = _q("09:30:30", 100.01, 100, 100.01, 100)
    baseline, _ = _run([SEED])
    panel, stats = _run([SEED, locked], NbboFilterPolicy(drop_locked=True))
    assert panel.equals(baseline)
    assert _stats_row(stats)["dropped_locked"] == 1


def test_nonpositive_price_is_dropped_and_previous_state_stands() -> None:
    baseline, _ = _run([SEED])
    panel, stats = _run([SEED, _q("09:30:30", 0.00, 100, 100.02, 100)])
    assert panel.equals(baseline)
    bar = _bar(panel, "09:31:00")
    assert bar["bid"] == pytest.approx(100.00, abs=1e-9)
    assert _stats_row(stats)["dropped_nonpositive_price"] == 1


def test_condition_allow_list_drops_other_conditions_only_when_set() -> None:
    opening = _q("09:30:30", 100.01, 100, 100.03, 100, qu_cond="O")
    kept, kept_stats = _run([SEED, opening])
    assert _bar(kept, "09:31:00")["n_updates"] == 1
    assert _stats_row(kept_stats)["dropped_condition"] == 0

    baseline, _ = _run([SEED])
    dropped, stats = _run([SEED, opening], NbboFilterPolicy(keep_qu_cond=("R",)))
    assert dropped.equals(baseline)
    assert _stats_row(stats)["dropped_condition"] == 1


def test_filter_counts_each_record_once_by_precedence() -> None:
    policy = NbboFilterPolicy(keep_qu_cond=("R",), drop_locked=True)
    _, stats = _run(
        [
            SEED,
            # non-positive AND off-list AND crossed -> nonpositive_price
            _q("09:30:10", 101.00, 100, -1.00, 100, qu_cond="O"),
            # off-list AND crossed -> condition
            _q("09:30:20", 101.00, 100, 100.50, 100, qu_cond="O"),
            # crossed only -> crossed
            _q("09:30:30", 101.00, 100, 100.50, 100),
            # locked only -> locked
            _q("09:30:40", 100.01, 100, 100.01, 100),
            # one-sided, kept
            _q("09:30:50", 100.01, 100, None, None),
        ],
        policy,
    )
    row = _stats_row(stats)
    assert {name: row[name] for name in _STATS_COUNTS} == {
        "records_in": 6,
        "dropped_nonpositive_price": 1,
        "dropped_condition": 1,
        "dropped_crossed": 1,
        "dropped_locked": 1,
        "one_sided_kept": 1,
        "both_null_kept": 0,
    }
    assert stats.columns == ["date", "symbol", *_STATS_COUNTS]
    for name in _STATS_COUNTS:
        assert stats.schema[name] == pl.Int64


def test_filter_policy_defaults_follow_d10_and_come_from_the_config() -> None:
    policy = NbboFilterPolicy()
    assert policy.drop_crossed is True
    assert policy.drop_locked is False
    assert policy.drop_nonpositive_price is True
    assert policy.keep_qu_cond is None

    config = NbboDatasetConfig(
        zarr_file_path="/tmp/nbbo-filter-policy-unused/panel.zarr",
        raw_data_dir_path="/tmp/nbbo-filter-policy-unused/raw",
        catalog_path="/tmp/nbbo-filter-policy-unused/catalog",
        drop_locked=True,
        keep_qu_cond=["R", "O"],  # a JSON round trip hands back a list
    )
    from_config = NbboFilterPolicy.from_config(config)
    assert from_config == NbboFilterPolicy(drop_locked=True, keep_qu_cond=("R", "O"))
    assert NbboResampler("1m").policy == NbboFilterPolicy()


def test_filter_policy_rejects_a_bare_string_condition_list() -> None:
    with pytest.raises(ValueError, match="keep_qu_cond"):
        NbboFilterPolicy(keep_qu_cond="R")


# ------------------------------- Task 2: total order, ties and ambiguity ---

DAY_2016 = "2016-12-07"


def _q16(clock, bid, bid_size, ask, ask_size, ordinal, **kwargs):
    return _q(clock, bid, bid_size, ask, ask_size, day=DAY_2016, ordinal=ordinal, **kwargs)


def _tie_fixture(sa_ordinal: int = 1) -> list[dict]:
    """2016-12-07 AAPL, microsecond timestamps, the D-19 tie cases."""
    return [
        _q16("09:29:00", 110.00, 100, 110.02, 100, 0),  # S0 seed
        # Bar 14:31Z: a three-way tie with three different states.
        _q16("09:30:10.000001", 110.01, 200, 110.03, 100, sa_ordinal),  # Sa
        _q16("09:30:10.000001", 110.02, 300, 110.03, 100, 2),  # Sb
        _q16("09:30:10.000001", 110.00, 100, 110.03, 100, 3),  # Sc
        _q16("09:30:40", 110.01, 100, 110.02, 200, 4),  # Sd
        # Bar 14:32Z: two IDENTICAL records exactly on the edge.
        _q16("09:32:00", 110.02, 100, 110.04, 100, 5),  # Se
        _q16("09:32:00", 110.02, 100, 110.04, 100, 6),  # Se
        # Bar 14:33Z: two different records exactly on the edge.
        _q16("09:33:00", 110.05, 100, 110.06, 100, 7),  # X
        _q16("09:33:00", 110.04, 100, 110.06, 100, 8),  # Y
        # Bar 14:34Z: empty.
    ]


def _run16(quotes, policy=None):
    return _run(quotes, policy, sessions=_sessions(DAY_2016))


def test_tie_collapse_keeps_the_last_record_by_ordinal_and_counts_ambiguity() -> None:
    panel, _ = _run16(_tie_fixture())
    bar = _bar(panel, "09:31:00", day=DAY_2016)
    assert bar["n_updates"] == 4
    assert bar["n_ambiguous_ties"] == 3
    # Snapshot = Sd.
    assert (bar["bid"], bar["bid_size"], bar["ask"], bar["ask_size"]) == pytest.approx(
        (110.01, 100, 110.02, 200), abs=1e-9
    )
    # S0 holds 10.000001 s, then only Sc (the last of the tie) holds until
    # 09:30:40; Sa and Sb have zero duration. Sd holds the last 20 s.
    assert bar["tw_spread"] == pytest.approx(
        (0.02 * 10.000001 + 0.03 * 29.999999 + 0.01 * 20) / 60, abs=1e-9
    )


def test_identical_tie_on_an_edge_counts_no_ambiguity_and_has_zero_duration() -> None:
    panel, _ = _run16(_tie_fixture())
    bar = _bar(panel, "09:32:00", day=DAY_2016)
    assert bar["n_updates"] == 2
    assert bar["n_ambiguous_ties"] == 0
    assert (bar["bid"], bar["ask"]) == pytest.approx((110.02, 110.04), abs=1e-9)
    # Se arrives exactly at the label, so Sd's spread fills the whole bar.
    assert bar["tw_spread"] == pytest.approx(0.01 * 60 / 60, abs=1e-9)


def test_different_tie_on_an_edge_snapshots_the_higher_ordinal() -> None:
    panel, _ = _run16(_tie_fixture())
    bar = _bar(panel, "09:33:00", day=DAY_2016)
    assert bar["n_updates"] == 2
    assert bar["n_ambiguous_ties"] == 2
    assert bar["bid"] == pytest.approx(110.04, abs=1e-9)  # Y, not X
    assert bar["tw_spread"] == pytest.approx(0.02 * 60 / 60, abs=1e-9)  # Se

    empty = _bar(panel, "09:34:00", day=DAY_2016)
    assert empty["n_updates"] == 0
    assert empty["n_ambiguous_ties"] == 0
    assert empty["bid"] == pytest.approx(110.04, abs=1e-9)  # Y carried
    assert empty["tw_spread"] == pytest.approx(0.02, abs=1e-9)


def test_ordinal_not_frame_position_decides_the_tie() -> None:
    original, _ = _run16(_tie_fixture())
    relabelled, _ = _run16(_tie_fixture(sa_ordinal=10))
    before = _bar(original, "09:31:00", day=DAY_2016)["tw_spread"]
    after = _bar(relabelled, "09:31:00", day=DAY_2016)["tw_spread"]
    # Sa (spread 0.02) is now the last of the tie.
    assert after == pytest.approx(
        (0.02 * 10.000001 + 0.02 * 29.999999 + 0.01 * 20) / 60, abs=1e-9
    )
    assert after != pytest.approx(before, abs=1e-9)


def test_permuting_input_rows_never_changes_the_panel() -> None:
    from polars.testing import assert_frame_equal

    quotes = _tie_fixture() + [
        _q16("09:29:30", 50.00, 10, 50.10, 10, 0, symbol="MSFT"),
        _q16("09:31:10", 50.01, 10, 50.10, 10, 1, symbol="MSFT"),
        _q16("09:31:10", 50.02, 20, 50.10, 10, 2, symbol="MSFT"),
    ]
    frame = _frame(quotes)
    resampler = NbboResampler("1m")
    sessions = _sessions(DAY_2016)
    expected, expected_stats = resampler.resample_with_stats(frame, sessions)
    for seed in range(20):
        shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
        panel, stats = resampler.resample_with_stats(shuffled, sessions)
        assert_frame_equal(panel, expected)
        assert_frame_equal(stats, expected_stats)


def test_tie_ambiguity_compares_null_sides_as_values_and_ignores_dropped_records() -> None:
    panel, _ = _run16(
        [
            _q16("09:29:00", 110.00, 100, 110.02, 100, 0),
            # Identical one-sided pair -> not ambiguous.
            _q16("09:30:20", 110.01, 100, None, None, 1),
            _q16("09:30:20", 110.01, 100, None, None, 2),
            # A one-sided and a two-sided record -> ambiguous (2).
            _q16("09:31:20", 110.01, 100, None, None, 3),
            _q16("09:31:20", 110.01, 100, 110.03, 100, 4),
            # A crossed record tied with a valid one: filtered before the
            # tie, so the survivor stands alone.
            _q16("09:32:20", 110.01, 100, 110.03, 100, 5),
            _q16("09:32:20", 111.00, 100, 110.03, 100, 6),
        ]
    )
    assert _bar(panel, "09:31:00", day=DAY_2016)["n_ambiguous_ties"] == 0
    assert _bar(panel, "09:32:00", day=DAY_2016)["n_ambiguous_ties"] == 2
    third = _bar(panel, "09:33:00", day=DAY_2016)
    assert third["n_ambiguous_ties"] == 0
    assert third["n_updates"] == 1
    assert third["bid"] == pytest.approx(110.01, abs=1e-9)


def test_distinct_nanosecond_timestamps_have_no_ambiguous_ties() -> None:
    panel, _ = _run(
        [
            SEED,
            _q("09:30:10.000000001", 100.01, 100, 100.03, 100),
            _q("09:30:10.000000002", 100.02, 100, 100.03, 100),
            _q("09:30:10.000000003", 100.00, 100, 100.03, 100),
            _q("09:31:00.000000001", 100.01, 100, 100.02, 100),
        ]
    )
    assert panel["n_ambiguous_ties"].to_list() == [0.0] * panel.height
    assert _bar(panel, "09:31:00")["bid"] == pytest.approx(100.00, abs=1e-9)
    assert _bar(panel, "09:31:00")["n_updates"] == 3


# ------------- Task 3: grid, right-closed labels, carry, variable contract ---


@pytest.mark.parametrize(
    ("bar_interval", "expected_bars"),
    [("1s", 23400), ("5s", 4680), ("1m", 390), ("5m", 78), ("30m", 13)],
)
def test_bar_interval_yields_a_regular_right_closed_grid(bar_interval, expected_bars) -> None:
    from quantlab.enums.data import BAR_INTERVAL_SECONDS

    panel, _ = _run([SEED], bar_interval=bar_interval)
    stamps = panel.filter(pl.col("symbol") == "AAPL")["timestamp"]
    assert stamps.len() == expected_bars
    step = BAR_INTERVAL_SECONDS[bar_interval]
    open_, close = _label(DAY, "09:30:00"), _label(DAY, "16:00:00")
    assert (stamps[0] - open_).total_seconds() == step  # first label = open + d
    assert stamps[-1] == close  # last label = close
    assert stamps.diff().drop_nulls().dt.total_seconds().unique().to_list() == [step]
    labels = NbboResampler(bar_interval).labels(_sessions(DAY))
    assert labels["timestamp"].to_list() == stamps.to_list()


def test_bar_interval_that_does_not_divide_the_session_raises() -> None:
    sessions = _sessions(DAY, close="15:59:59")
    with pytest.raises(ValueError, match=r"2024-01-24.*1m|1m.*2024-01-24"):
        NbboResampler("1m").resample(_frame([SEED]), sessions)
    with pytest.raises(ValueError, match=r"2024-01-24.*1m|1m.*2024-01-24"):
        NbboResampler("1m").labels(sessions)


def test_unknown_bar_interval_lists_the_accepted_tokens() -> None:
    with pytest.raises(ValueError, match=r"'7m'.*'1s'.*'1m'"):
        NbboResampler("7m")


def test_right_closed_label_edges_seed_and_after_close() -> None:
    panel, _ = _run(
        [
            SEED,
            # Exactly at the open: this is the seed, an update in no bar.
            _q("09:30:00", 100.01, 100, 100.02, 100),
            # Exactly on the 09:31 edge: bar 14:31Z, not 14:32Z.
            _q("09:31:00", 100.02, 100, 100.03, 100),
            # After the close: in no bar.
            _q("16:00:00.000001", 99.00, 100, 99.10, 100),
        ]
    )
    first = _bar(panel, "09:31:00")
    assert first["n_updates"] == 1
    assert first["bid"] == pytest.approx(100.02, abs=1e-9)
    # The seed at the open (spread 0.01) held for the whole bar; the edge
    # record has zero duration inside it.
    assert first["tw_spread"] == pytest.approx(0.01 * 60 / 60, abs=1e-9)
    assert _bar(panel, "09:32:00")["n_updates"] == 0
    assert panel["n_updates"].sum() == 1
    last = _bar(panel, "16:00:00")
    assert last["bid"] == pytest.approx(100.02, abs=1e-9)  # not 99.00


def test_carry_never_crosses_a_session_date() -> None:
    next_day = "2024-01-25"
    panel, _ = _run(
        [
            SEED,
            _q("15:59:00", 100.05, 100, 100.07, 100),
            # Next day: no pre-open record; the first arrives mid-bar.
            _q("10:00:30", 101.00, 100, 101.04, 300, day=next_day),
        ],
        sessions=_sessions(DAY, next_day),
    )
    day_two = panel.filter(pl.col("date") == date.fromisoformat(next_day))
    assert day_two.height == 390
    before = day_two.filter(pl.col("timestamp") <= _label(next_day, "10:00:00"))
    assert before.height == 30
    for name in (
        "bid", "ask", "bid_size", "ask_size", "mid", "spread",
        "tw_spread", "tw_bid_size", "tw_ask_size",
    ):
        assert before[name].is_null().all(), name
    assert before["n_updates"].to_list() == [0.0] * 30
    first = _bar(panel, "10:01:00", day=next_day)
    assert first["n_updates"] == 1
    assert first["bid"] == pytest.approx(101.00, abs=1e-9)
    # Covered only for the last 30 s of the bar: the TW is that state alone.
    assert first["tw_spread"] == pytest.approx(0.04 * 30 / 30, abs=1e-9)
    assert first["tw_ask_size"] == pytest.approx(300 * 30 / 30, abs=1e-9)
    # Day one's last state stood to its own close.
    assert _bar(panel, "16:00:00")["bid"] == pytest.approx(100.05, abs=1e-9)


def test_carry_rows_exist_for_a_symbol_whose_every_record_was_dropped() -> None:
    panel, stats = _run(
        [SEED, _q("09:45:00", 51.00, 100, 50.00, 100, symbol="MSFT")]
    )
    msft = panel.filter(pl.col("symbol") == "MSFT")
    assert msft.height == 390
    assert msft["bid"].is_null().all()
    assert msft["tw_spread"].is_null().all()
    assert msft["n_updates"].to_list() == [0.0] * 390
    assert _stats_row(stats, symbol="MSFT")["dropped_crossed"] == 1


def test_panel_variables_are_exactly_the_cleaning_contract_as_float64() -> None:
    from quantlab.dataset.cleaning import NBBO_PANEL_VARIABLES

    panel, _ = _run(
        [
            SEED,
            _q("09:30:30", 100.01, 300, 100.05, 100),
            _q("09:31:30", 100.02, 200, 100.03, 600, symbol="MSFT"),
        ]
    )
    assert panel.columns == ["symbol", "date", "timestamp", *NBBO_PANEL_VARIABLES]
    for name in NBBO_PANEL_VARIABLES:
        assert panel.schema[name] == pl.Float64, name

    bar = _bar(panel, "09:31:00")
    bid, ask, bid_size, ask_size = 100.01, 100.05, 300.0, 100.0
    mid = (bid + ask) / 2
    assert bar["mid"] == pytest.approx(mid, abs=1e-9)
    assert bar["spread"] == pytest.approx(ask - bid, abs=1e-9)
    assert bar["spread_bps"] == pytest.approx(1e4 * (ask - bid) / mid, abs=1e-9)
    assert bar["imbalance"] == pytest.approx(
        (bid_size - ask_size) / (bid_size + ask_size), abs=1e-9
    )
