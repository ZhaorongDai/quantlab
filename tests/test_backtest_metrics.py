"""In-sample detection and sliced metrics for the backtester (phase 03.7, plan 08).

What this file locks, and what turns each lock red:

- **D-17, the effective training window.** It is `[train_start, train_end +
  label horizon]`, with the horizon the maximum `n_forward_periods` across the
  model's labels, counted in BARS on the price calendar: the label on
  `train_end` reads the next n bars of prices, so those bars are in-sample too.
  Dropping the horizon, taking the minimum instead of the maximum, or adding
  calendar days instead of bars moves the window end, and the weekend test
  (a Friday `train_end` plus two bars is the following Tuesday) goes red.
- **D-17, overlap handling.** A backtest window that overlaps the training
  window logs one warning naming both ranges, still completes `run()`, and
  records the overlap as `in_sample_range`. A disjoint window records no
  in-sample range and does not warn. A label without `n_forward_periods` and a
  model without train dates each warn and produce an explicit value (horizon 0,
  `training_window: null`), never a silent guess.
- **D-34, the single-simulation rule.** vectorbt's `Portfolio` cannot be
  time-sliced (RESEARCH Pitfall 5), and re-simulating a slice would reset the
  capital and change the path. `Portfolio.from_orders` is therefore spied and
  must run exactly once per `run()`, even when the window is split. A slice's
  total return must equal the compounded whole-run returns over that slice, so
  a slice computed on anything else is red.
- **D-22 / D-34, the metric blocks.** `whole` is the full `pf.stats()` set plus
  turnover. `in_sample` / `out_of_sample` are returns statistics on the sliced
  returns, plus order, trade and turnover statistics filtered to the slice.
  In-sample and out-of-sample order counts and fees partition the whole run.
  Turnover is one-sided traded notional over the previous bar's portfolio
  value, so a full entry is 1 and a full book swap is 2.
- **Strict JSON.** A one-bar in-sample slice genuinely yields a NaN statistic,
  and `metrics.json` must still parse with a parser that rejects `NaN` and
  `Infinity` tokens, so writing slice metrics without `to_jsonable` is red.

Everything is synthetic, CPU-only and offline. Configs are built directly,
never through `quantlab/config/__init__.py` (D-32).
"""

import json
import math
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

import quantlab.backtest.engine_vectorbt as engine_module
from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.backtest import SimulationResult
from quantlab.base.config import CrossSectionBacktestConfig, PolarsFactorConfig
from tests.backtest_fixtures import (
    ForwardReturnLabel,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)
TRAIN_END_BAR = 24


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture
def warnings_sink():
    """Loguru WARNING-and-above messages emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _strict_json(path: Path) -> dict:
    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token!r} in {path}")

    return json.loads(path.read_text(), parse_constant=_reject)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _model_dates(train_end_bar: int = TRAIN_END_BAR) -> dict:
    return dict(
        start_date=_day(BARS[0]),
        end_date=_day(BARS[29]),
        train_start=_day(BARS[0]),
        train_end=_day(BARS[train_end_bar]),
        test_start=_day(BARS[train_end_bar + 1]),
        test_end=_day(BARS[29]),
    )


def _unit_backtester(
    tmp_path, *, n_forward_periods: int = 1, **model_dates
) -> USEquityCrossectionSelectStockVectorBt:
    """A backtester that is only constructed, never run: the checkpoint need not exist."""
    dataset_config = write_price_store(tmp_path / "store", n_bars=40)
    dates = _model_dates()
    dates.update(model_dates)
    model = make_model(
        tmp_path / "unit", dataset_config, n_forward_periods=n_forward_periods, **dates
    )
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=model,
            model_mode="load",
            checkpoint=str(tmp_path / "missing.joblib"),
            start_date=_day(BARS[30]),
            end_date=_day(BARS[39]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=5,
            direction="long_only",
            top_n=2,
        )
    )


def _run_backtester(
    tmp_path, *, window_start_bar: int, window_end_bar: int, **overrides
) -> USEquityCrossectionSelectStockVectorBt:
    """Load mode over a checkpoint trained on bars 0..TRAIN_END_BAR (horizon 1 bar)."""
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **_model_dates())
    )
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(tmp_path / "backtest", dataset_config, **_model_dates()),
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(BARS[window_start_bar]),
        end_date=_day(BARS[window_end_bar]),
        output_dir=str(tmp_path / "runs"),
        rebalance_periods=5,
        direction="long_only",
        top_n=2,
    )
    kwargs.update(overrides)
    return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(**kwargs))


# --------------------------------------------------------------------------
# D-17: the effective training window
# --------------------------------------------------------------------------


def test_label_horizon_is_the_max_n_forward_periods_across_labels(tmp_path):
    backtester = _unit_backtester(tmp_path, n_forward_periods=1)
    labels = backtester.config.model.config.labels
    labels.append(
        ForwardReturnLabel(
            PolarsFactorConfig(
                window=0,
                dataset=labels[0].config.dataset,
                kwargs={"n_forward_periods": 3},
            )
        )
    )
    assert [int(label.config.kwargs["n_forward_periods"]) for label in labels] == [1, 3]

    assert backtester._label_horizon_bars() == 3


def test_training_window_end_adds_the_horizon_in_bars_across_a_weekend(tmp_path):
    friday = "2024-01-05"
    assert pd.Timestamp(friday).day_name() == "Friday"
    backtester = _unit_backtester(
        tmp_path,
        n_forward_periods=2,
        start_date="2024-01-01",
        end_date="2024-01-26",
        train_start="2024-01-01",
        train_end=friday,
        test_start="2024-01-08",
        test_end="2024-01-26",
    )
    calendar = pd.bdate_range("2024-01-01", periods=20).values

    window = backtester._training_window(calendar, "2024-01-01", friday)

    # Friday + 2 bars = Monday, Tuesday. Two calendar days would give Sunday.
    assert tuple(window) == ("2024-01-01", "2024-01-09")


def test_label_without_n_forward_periods_warns_and_uses_zero(tmp_path, warnings_sink):
    backtester = _unit_backtester(tmp_path)
    backtester.config.model.config.labels[0].config.kwargs = {}

    assert backtester._label_horizon_bars() == 0
    assert any("ForwardReturnLabel" in message for message in warnings_sink), (
        warnings_sink
    )


# --------------------------------------------------------------------------
# D-17 on intraday bars (code review CR-01)
# --------------------------------------------------------------------------

#: Three 7-bar hourly sessions (10:00..16:00); no bar sits at midnight.
SESSION_DAYS = ("2024-01-01", "2024-01-02", "2024-01-03")
SESSION_HOURS = tuple(range(10, 17))


def _sessions(days, hours=SESSION_HOURS) -> np.ndarray:
    return pd.DatetimeIndex(
        [pd.Timestamp(f"{day} {hour:02d}:00") for day in days for hour in hours]
    ).values


def test_intraday_training_window_ends_horizon_bars_into_the_next_session(tmp_path):
    """CR-01: on intraday bars the label-horizon bars after `train_end` are in-sample.

    The model layer trains on `sel(timestamp=slice(train_start, train_end))`,
    and a date-only `train_end` selects that whole session (positive control
    below), so the last training label sits at 16:00 and reads the next two
    bars: 10:00 and 11:00 of the FOLLOWING session. Those two bars must be
    in-sample. The old code truncated `train_end` to midnight, landed on the
    previous session's last bar, added the horizon and then compared by day:
    the whole `train_end` session was in-sample and the two leaked bars on the
    next session were labelled out-of-sample. This test goes red on that.
    """
    backtester = _unit_backtester(tmp_path, n_forward_periods=2)
    calendar = _sessions(SESSION_DAYS)
    model_layer_slice = xr.DataArray(
        np.arange(calendar.size), dims="timestamp", coords={"timestamp": calendar}
    ).sel(timestamp=slice("2024-01-01", "2024-01-02"))
    assert pd.Timestamp(model_layer_slice.timestamp.values[-1]) == pd.Timestamp(
        "2024-01-02 16:00"
    )

    window = backtester._training_window(calendar, "2024-01-01", "2024-01-02")

    assert tuple(window) == ("2024-01-01T10:00:00", "2024-01-03T11:00:00")
    split = backtester._split_window(_sessions(SESSION_DAYS[2:]), window)
    assert tuple(split["in_sample_range"]) == (
        "2024-01-03T10:00:00",
        "2024-01-03T11:00:00",
    )
    assert [tuple(r) for r in split["out_of_sample_ranges"]] == [
        ("2024-01-03T12:00:00", "2024-01-03T16:00:00")
    ]


@pytest.mark.parametrize(
    "train_end",
    [
        "2024-01-02T13:00:00",
        "2024-01-02T13:00:00.000000000",
        pd.Timestamp("2024-01-02 13:00"),
        np.datetime64("2024-01-02T13:00"),
    ],
    ids=["iso", "fold-style-ns", "timestamp", "datetime64"],
)
def test_training_window_honours_a_time_of_day_train_end(tmp_path, train_end):
    """CR-01: a `train_end` carrying a time of day is not truncated to its date.

    `cv_folds.json` stores fold dates as nanosecond strings, which the model
    layer slices exactly. Training ends at 13:00, so with a 2-bar horizon the
    window ends at 15:00 on the same session. The old `_iso_date` truncation
    turned every spelling into the bare date and went red here.
    """
    backtester = _unit_backtester(tmp_path, n_forward_periods=2)

    window = backtester._training_window(
        _sessions(SESSION_DAYS), "2024-01-01", train_end
    )

    assert tuple(window) == ("2024-01-01T10:00:00", "2024-01-02T15:00:00")


def test_slice_statistics_compare_exact_bar_timestamps_not_days(tmp_path):
    """CR-01: slice masks, sliced returns and trade counts use exact bar times.

    A 24-hour hourly market has a bar AT midnight, whose range label is the
    bare date `2024-01-02`. The range 21:00 .. that midnight bar holds exactly
    four bars. Day-based comparison (the old `_in_ranges`, the old open-trade
    count and the old string `.loc` slice) stretches the end to the whole of
    2024-01-02 and goes red: 28 bars, a compounded return over 28 bars, a
    trade entered at 10:00 on 2024-01-02 counted as open at the range end, and
    a trade closed at 05:00 counted as closed inside the range.
    """
    backtester = _unit_backtester(tmp_path)
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    returns = pd.Series(np.linspace(0.001, 0.048, index.size), index=index)
    trades = xr.Dataset(
        {
            "symbol": ("trade", np.array(["AAA", "BBB"])),
            "entry_timestamp": (
                "trade",
                pd.to_datetime(["2024-01-01 22:00", "2024-01-02 10:00"]).values,
            ),
            "exit_timestamp": (
                "trade",
                pd.to_datetime(["2024-01-02 05:00", "2024-01-02 23:00"]).values,
            ),
            "pnl": ("trade", np.array([1.0, 2.0])),
            "return": ("trade", np.array([0.01, 0.02])),
            "status": ("trade", np.array(["Closed", "Open"])),
        }
    )
    simulation = SimulationResult(
        value=xr.DataArray(
            np.ones(index.size), dims="timestamp", coords={"timestamp": index.values}
        ),
        returns=xr.DataArray(
            returns.values, dims="timestamp", coords={"timestamp": index.values}
        ),
        orders=xr.Dataset(),
        liquidations=[],
        bar_interval=np.timedelta64(1, "h"),
        trades=trades,
        native=types.SimpleNamespace(returns=lambda: returns),
    )
    ranges = [("2024-01-01T21:00:00", "2024-01-02")]

    mask = backtester._in_ranges(index.values, ranges)
    assert list(index[mask]) == list(pd.date_range("2024-01-01 21:00", periods=4, freq="h"))

    stats = backtester._period_returns_stats(simulation, ranges)
    expected = (np.prod(1.0 + returns.iloc[21:25].values) - 1.0) * 100.0
    assert stats["Total Return [%]"] == pytest.approx(expected, rel=1e-12)

    records = backtester._period_record_stats(simulation, ranges)
    assert records["open_trade_count"] == 1
    assert records["closed_trade_count"] == 0


def test_whole_order_count_is_zero_for_an_order_less_simulation(tmp_path, monkeypatch):
    """`whole["order_count"]` must read `.sizes.get("order", 0)`, never `["order"]`.

    A simulation that never filled carries `orders=xr.Dataset()`, which has no
    `order` dimension at all, so a bare subscript raises KeyError. That raise
    would happen while the run directory is still the staging directory, and
    the failure handler deletes the ENTIRE staging directory -- so the cost of
    this one-character mistake is every artifact of a completed backtest, not
    just a missing metric (threat T-03.8-01-01, rated high).

    Why this test exists at all: the mutation `.sizes.get("order", 0)` ->
    `.sizes["order"]` was run against the whole of this file and ESCAPED, 17
    passed. The neighbouring order-less test above calls `_period_record_stats`
    and `_period_returns_stats` directly and never reaches `_compute_metrics`,
    so nothing covered this read. `_engine_stats` is stubbed here rather than
    driven through a real `Portfolio`: what needs proving is the order-record
    read, and a real vectorbt portfolio would only add a way for the test to
    fail for an unrelated reason.
    """
    backtester = _unit_backtester(tmp_path)
    monkeypatch.setattr(
        type(backtester), "_engine_stats", lambda self, simulation: {}
    )
    index = pd.date_range("2024-01-01", periods=3, freq="h")
    simulation = SimulationResult(
        value=xr.DataArray(
            np.ones(index.size), dims="timestamp", coords={"timestamp": index.values}
        ),
        returns=xr.DataArray(
            np.zeros(index.size), dims="timestamp", coords={"timestamp": index.values}
        ),
        orders=xr.Dataset(),
        liquidations=[],
        bar_interval=np.timedelta64(1, "h"),
        trades=xr.Dataset(),
        native=None,
    )
    # No in-sample and no out-of-sample range, so neither slice is computed and
    # the assertion below can only be about the whole-window read.
    split = {"in_sample_range": None, "out_of_sample_ranges": []}

    metrics = backtester._compute_metrics(simulation, None, split)

    whole = metrics["whole"]
    assert whole["order_count"] == 0
    assert isinstance(whole["order_count"], int) and not isinstance(
        whole["order_count"], bool
    )
    assert metrics["in_sample"] is None and metrics["out_of_sample"] is None


# --------------------------------------------------------------------------
# D-17: overlap handling through run()
# --------------------------------------------------------------------------


def test_overlapping_window_warns_naming_both_ranges_and_continues(
    tmp_path, warnings_sink
):
    # Training window: bar 0 .. bar 24 + 1-bar horizon = bar 25.
    # The window starts ON bar 25, so exactly one bar is in-sample; a one-bar
    # returns slice has NaN volatility, which strict JSON must still survive.
    window_start, window_end = TRAIN_END_BAR + 1, 45
    result = _run_backtester(
        tmp_path, window_start_bar=window_start, window_end_bar=window_end
    ).run()

    training = (_day(BARS[0]), _day(BARS[TRAIN_END_BAR + 1]))
    backtest = (_day(BARS[window_start]), _day(BARS[window_end]))
    overlap_warnings = [
        message
        for message in warnings_sink
        if all(date in message for date in (*training, *backtest))
    ]
    assert len(overlap_warnings) == 1, warnings_sink

    metrics = result.metrics
    assert tuple(metrics["training_window"]) == training
    assert tuple(metrics["in_sample_range"]) == (backtest[0], backtest[0])
    assert [tuple(r) for r in metrics["out_of_sample_ranges"]] == [
        (_day(BARS[window_start + 1]), backtest[1])
    ]

    persisted = _strict_json(result.run_dir / "metrics.json")
    assert persisted["in_sample_range"] == [backtest[0], backtest[0]]
    assert persisted["training_window"] == list(training)


def test_disjoint_window_does_not_warn_and_has_no_in_sample_range(
    tmp_path, warnings_sink
):
    window = (_day(BARS[30]), _day(BARS[50]))
    result = _run_backtester(tmp_path, window_start_bar=30, window_end_bar=50).run()

    metrics = result.metrics
    assert tuple(metrics["training_window"]) == (
        _day(BARS[0]),
        _day(BARS[TRAIN_END_BAR + 1]),
    )
    assert metrics["in_sample_range"] is None
    assert [tuple(r) for r in metrics["out_of_sample_ranges"]] == [window]
    assert not any("overlap" in message for message in warnings_sink), warnings_sink


def test_model_without_train_dates_warns_and_records_null(tmp_path, warnings_sink):
    backtester = _run_backtester(tmp_path, window_start_bar=20, window_end_bar=45)
    backtester.config.model.config.train_start = None
    # Code review WR-01: in load mode the checkpoint's own config.json records
    # the dates it really trained on and takes precedence over config.model.
    # The "no training dates at all" arm is therefore reached only when no such
    # record exists, so the record is removed here.
    (Path(backtester.config.checkpoint).parent / "config.json").unlink()

    result = backtester.run()

    metrics = result.metrics
    assert metrics["training_window"] is None
    assert metrics["in_sample_range"] is None
    assert [tuple(r) for r in metrics["out_of_sample_ranges"]] == [
        (_day(BARS[20]), _day(BARS[45]))
    ]
    assert any("train_start" in message for message in warnings_sink), warnings_sink
    assert _strict_json(result.run_dir / "metrics.json")["training_window"] is None


# --------------------------------------------------------------------------
# D-17 / D-34: one continuous simulation, sliced afterwards
# --------------------------------------------------------------------------

#: Training window bar 0 .. bar 25 (train_end bar 24 + 1-bar horizon); this
#: window puts bars 20..25 in-sample and 26..45 out-of-sample, each slice
#: holding at least one fill bar (fills land on window bars 1, 6, 11, ...).
OVERLAP_START, OVERLAP_END = 20, 45


def _slice_returns(returns: xr.DataArray, day_range) -> np.ndarray:
    start, end = day_range
    return returns.sel(timestamp=slice(start, end)).values


def test_from_orders_runs_exactly_once_per_run_even_with_an_overlap(
    tmp_path, monkeypatch
):
    backtester = _run_backtester(
        tmp_path, window_start_bar=OVERLAP_START, window_end_bar=OVERLAP_END
    )
    real_from_orders = engine_module.vbt.Portfolio.from_orders
    calls = []

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("size"))
        return real_from_orders(*args, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "vbt",
        types.SimpleNamespace(Portfolio=types.SimpleNamespace(from_orders=_spy)),
    )

    result = backtester.run()

    assert result.metrics["in_sample_range"] is not None
    assert result.metrics["in_sample"] is not None
    assert result.metrics["out_of_sample"] is not None
    assert len(calls) == 1


def test_slice_total_return_is_compounded_from_the_single_simulations_returns(
    tmp_path,
):
    result = _run_backtester(
        tmp_path, window_start_bar=OVERLAP_START, window_end_bar=OVERLAP_END
    ).run()
    metrics = result.metrics
    returns = result.simulation.returns

    in_sample = _slice_returns(returns, metrics["in_sample_range"])
    assert in_sample.size == TRAIN_END_BAR + 1 - OVERLAP_START + 1
    assert metrics["in_sample"]["Total Return [%]"] == pytest.approx(
        100.0 * (np.prod(1.0 + in_sample) - 1.0), abs=1e-9
    )

    (out_range,) = metrics["out_of_sample_ranges"]
    out_of_sample = _slice_returns(returns, out_range)
    assert in_sample.size + out_of_sample.size == returns.sizes["timestamp"]
    assert metrics["out_of_sample"]["Total Return [%]"] == pytest.approx(
        100.0 * (np.prod(1.0 + out_of_sample) - 1.0), abs=1e-9
    )
    # A slice statistic is not the whole-run statistic in disguise.
    assert metrics["out_of_sample"]["Total Return [%]"] != pytest.approx(
        metrics["whole"]["Total Return [%]"], abs=1e-9
    )


def test_slice_order_counts_partition_the_whole_run(tmp_path):
    result = _run_backtester(
        tmp_path, window_start_bar=OVERLAP_START, window_end_bar=OVERLAP_END
    ).run()
    metrics = result.metrics
    orders = result.simulation.orders
    inside, outside = metrics["in_sample"], metrics["out_of_sample"]

    assert inside["order_count"] > 0 and outside["order_count"] > 0
    assert inside["order_count"] + outside["order_count"] == orders.sizes["order"]

    # The whole-window counterpart (phase 03.8, CONTEXT item 3): the block that
    # gave up the lot-level trade set must still answer "how many fills
    # happened over the whole window". Both identities hold by construction,
    # because `_split_window` tiles the window into one in-sample range plus
    # 0-2 disjoint out-of-sample ranges covering every remaining bar. They are
    # asserted anyway: that construction is exactly what a future refactor of
    # the split could break silently.
    whole = metrics["whole"]
    assert whole["order_count"] == inside["order_count"] + outside["order_count"]
    assert whole["order_count"] == orders.sizes["order"]

    total_fees = float(orders["fees"].values.sum())
    assert total_fees > 0.0
    assert inside["fees_paid"] + outside["fees_paid"] == pytest.approx(
        total_fees, abs=1e-9
    )
    assert metrics["whole"]["Total Fees Paid"] == pytest.approx(total_fees, abs=1e-6)

    per_order_notional = np.abs(orders["size"].values) * orders["price"].values
    notional = float(per_order_notional.sum())
    assert inside["traded_notional"] + outside["traded_notional"] == pytest.approx(
        notional, rel=1e-12
    )

    # Trades: closed trades partition the run; the single out-of-sample piece
    # ends on the window's last bar, so its open count is the run's open count.
    #
    # These identities were RESTORED in phase 03.8 by unifying the trade view
    # (D-02) -- `SimulationResult.trades` and `whole` are both the position view
    # now -- and NOT by relaxing the assertions. If they ever go red again, the
    # source drifted back to two trade vocabularies under names that read
    # identically: fix the source, never the assertion.
    assert whole["Total Closed Trades"] > 0 and whole["Total Open Trades"] > 0
    assert (
        inside["closed_trade_count"] + outside["closed_trade_count"]
        == whole["Total Closed Trades"]
    )
    assert outside["open_trade_count"] == whole["Total Open Trades"]

    # Turnover, recomputed here from the records on random-walk prices: traded
    # notional on a fill bar over the equity value at the PREVIOUS bar (a
    # fill-bar denominator would differ, since prices move every bar).
    value = result.simulation.value
    value_ts = value.timestamp.values.astype("datetime64[ns]")
    order_ts = orders["timestamp"].values.astype("datetime64[ns]")
    expected = []
    for bar in np.unique(order_ts):
        i = int(np.searchsorted(value_ts, bar))
        prior = float(value.values[i - 1]) if i > 0 else backtester_init_cash(result)
        expected.append(float(per_order_notional[order_ts == bar].sum()) / prior)
    assert whole["turnover"]["sum"] == pytest.approx(sum(expected), rel=1e-12)
    assert inside["turnover"]["sum"] + outside["turnover"]["sum"] == pytest.approx(
        sum(expected), rel=1e-12
    )


def backtester_init_cash(result) -> float:
    """The run's init_cash, read back from its persisted config."""
    return float(_strict_json(result.run_dir / "config.json")["init_cash"])


def test_turnover_is_one_for_a_full_entry_and_two_for_a_full_swap(tmp_path):
    backtester = _unit_backtester(tmp_path)
    backtester.config.fees = 0.0
    backtester.config.slippage = 0.0
    market = backtester.MARKET
    timestamps = pd.bdate_range("2024-01-01", periods=6)
    symbols = ["A", "B"]
    constant = np.full((6, 2), 100.0)
    prices = xr.Dataset(
        {
            market.fill_price_column: (("timestamp", "symbol"), constant),
            market.valuation_price_column: (("timestamp", "symbol"), constant.copy()),
        },
        coords={"timestamp": timestamps, "symbol": symbols},
    )
    rows = np.full((6, 2), np.nan)
    rows[0] = [1.0, 0.0]  # full entry into A, fills on bar 1
    rows[2] = [0.0, 1.0]  # swap the whole book into B, fills on bar 3
    weights = xr.Dataset(
        {"weight": (("timestamp", "symbol"), rows)},
        coords={"timestamp": timestamps, "symbol": symbols},
    )

    turnover = backtester._turnover(backtester._simulate(weights, prices))

    assert turnover.dims == ("timestamp",)
    np.testing.assert_array_equal(
        turnover.timestamp.values.astype("datetime64[ns]"),
        timestamps[[1, 3]].values.astype("datetime64[ns]"),
    )
    assert turnover.values[0] == pytest.approx(1.0, abs=1e-9)
    assert turnover.values[1] == pytest.approx(2.0, abs=1e-9)


def test_metrics_blocks_have_the_d22_d34_keys(tmp_path):
    # One in-sample bar (bar 25): its returns slice genuinely has a NaN
    # volatility, so metrics.json must convert it to null to parse strictly.
    result = _run_backtester(
        tmp_path, window_start_bar=TRAIN_END_BAR + 1, window_end_bar=OVERLAP_END
    ).run()
    metrics = result.metrics
    turnover_keys = {"mean_per_rebalance", "sum", "annualized"}

    whole = metrics["whole"]
    for key in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]", "Total Fees Paid"):
        assert key in whole, key
    assert set(whole["turnover"]) == turnover_keys
    assert "benchmark" not in metrics

    slice_keys = (
        "Total Return [%]",
        "Sharpe Ratio",
        "order_count",
        "fees_paid",
        "traded_notional",
        "closed_trade_count",
        "open_trade_count",
    )
    blocks = [metrics["in_sample"], metrics["out_of_sample"]]
    assert all(block is not None for block in blocks)
    for block in blocks:
        for key in slice_keys:
            assert key in block, key
        assert set(block["turnover"]) == turnover_keys

    assert math.isnan(metrics["in_sample"]["Annualized Volatility [%]"])
    persisted = _strict_json(result.run_dir / "metrics.json")
    assert persisted["in_sample"]["Annualized Volatility [%]"] is None
    assert set(persisted["out_of_sample"]["turnover"]) == turnover_keys
