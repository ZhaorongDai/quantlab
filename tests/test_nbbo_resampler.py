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
