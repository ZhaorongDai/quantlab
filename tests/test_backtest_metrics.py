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

    total_fees = float(orders["fees"].values.sum())
    assert total_fees > 0.0
    assert inside["fees_paid"] + outside["fees_paid"] == pytest.approx(
        total_fees, abs=1e-9
    )
    assert metrics["whole"]["Total Fees Paid"] == pytest.approx(total_fees, abs=1e-6)

    notional = float(
        (np.abs(orders["size"].values) * orders["price"].values).sum()
    )
    assert inside["traded_notional"] + outside["traded_notional"] == pytest.approx(
        notional, rel=1e-12
    )


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
