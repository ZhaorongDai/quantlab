"""The vectorised backtester drives per-bar portfolio construction (#79).

What is locked here, and what turns it red:

- The constructor's `history_bars` sets the price warm-up (`history_bars - 1`
  bars before the window, a warning when the store holds fewer): the first
  backtest bar's context already holds a full, finite window of one-bar
  valuation returns ending at that bar.
- The current weights a constructor is handed are the holdings the
  simulation carries at that bar's close: the last traded weights filled at
  the next bar's open and marked to this bar's close, across a halt, on
  either sizing basis.
- A run reads the price dataset's delisting marks once, for the decision
  replay and the engine alike.
- A bar whose construction raises `PortfolioConstructionError` holds (an
  all-NaN row), is logged, and is listed in `metrics.json`.
- A `MeanVarianceOptimizer` backtest is fully invested on every rebalance
  bar, and its `config.json` (optimiser and risk model) rebuilds a
  backtester that re-runs identically.
- A long-short `MeanVarianceOptimizer` backtest (#80) is dollar-neutral with
  gross exposure at most one on every rebalance bar, and rebuilds from its
  `config.json`.
- A risk model declaring a `Factor` in `required_factors()` (#82, a Polars
  factor here) receives that factor's values at each rebalance bar, and
  only at it, in `context.factors`, warmed up like a model's features.

Everything is synthetic, CPU-only and offline.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig, LedoitWolfConfig, MeanVarianceConfig
from quantlab.base.portfolio import PortfolioConstructionError, PortfolioConstructor
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.utils.module import load_backtester_from_config
from quantlab.base.config import PolarsFactorConfig
from tests.backtest_fixtures import FirstFeatureHead, PastReturnFactor, make_model, make_stock_dataset, write_price_store

N_BARS = 90
LOOKBACK = 20
WINDOW = (40, 85)
REBALANCE = 5


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class RecorderConfig:
    lookback_bars: int = LOOKBACK
    fail_on: int | None = None


#: Contexts seen by every `Recorder`, in call order (rebuilt instances share it).
SEEN: list = []


class Recorder(PortfolioConstructor):
    """Holds AAA and BBB half and half around any locked position, records its
    contexts, fails on one call."""

    config_cls = RecorderConfig

    @property
    def lookback_bars(self):
        return self.config.lookback_bars

    def construct(self, context):
        SEEN.append(context)
        if self.config.fail_on is not None and len(SEEN) - 1 == self.config.fail_on:
            raise PortfolioConstructionError("solver gave up")
        locked = context.locked
        row = context.current_weights.where(locked, 0.0)
        free = [s for s in ("AAA", "BBB") if not bool(locked.sel(symbol=s))]
        row.loc[{"symbol": free}] = (1.0 - float(row.sum())) / len(free)
        return row


def _setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS, seed=11)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    model = make_model(
        tmp_path / "model",
        dataset_config,
        start_date=_day(bars[0]),
        end_date=_day(bars[39]),
        train_start=_day(bars[0]),
        train_end=_day(bars[34]),
        test_start=_day(bars[35]),
        test_end=_day(bars[39]),
    )
    return dataset_config, model, bars


def _backtester(tmp_path, constructor, *, output_dir=None, fees=0.0, slippage=0.0, sizing_basis="fill"):
    """``constructor`` may be a function of the price store's dataset config."""
    dataset_config, model, bars = _setup(tmp_path)
    if not isinstance(constructor, PortfolioConstructor):
        constructor = constructor(dataset_config)
    return (
        USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=make_stock_dataset(dataset_config),
                model=model,
                model_mode="train",
                start_date=_day(bars[WINDOW[0]]),
                end_date=_day(bars[WINDOW[1]]),
                output_dir=output_dir,
                rebalance_periods=REBALANCE,
                constructor=constructor,
                fees=fees,
                slippage=slippage,
                sizing_basis=sizing_basis,
            )
        ),
        dataset_config,
        bars,
    )


@pytest.fixture(autouse=True)
def _clear_seen():
    SEEN.clear()


def test_the_first_backtest_bar_has_a_full_return_window(tmp_path):
    backtester, dataset_config, bars = _backtester(tmp_path, Recorder(RecorderConfig()))

    backtester.run()

    first = SEEN[0]
    assert first.timestamp == pd.Timestamp(bars[WINDOW[0]])
    assert first.returns.sizes["timestamp"] == LOOKBACK
    assert pd.Timestamp(first.returns.timestamp.values[-1]) == first.timestamp
    assert np.isfinite(first.returns.values).all()
    close = xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].transpose("timestamp", "symbol").values
    t = WINDOW[0]
    np.testing.assert_allclose(
        first.returns.values, close[t - LOOKBACK + 1 : t + 1] / close[t - LOOKBACK : t] - 1
    )


class _LongHistoryRecorder(Recorder):
    """A recorder reading 100 raw prices per bar, more than the store holds before the window."""

    @property
    def history_bars(self):
        return 100


def test_the_warm_up_is_the_constructors_history_bars(tmp_path):
    backtester, _, bars = _backtester(tmp_path, _LongHistoryRecorder(RecorderConfig()))

    with pytest.warns(UserWarning, match=r"reads 99 bar\(s\) of prices before the window but the price dataset holds only 40; the first price windows are short by 59"):
        backtester.run()


def test_a_run_reads_the_delisting_marks_once(tmp_path, monkeypatch):
    """The marks the decision replay settles are the ones the engine settles."""
    backtester, _, _ = _backtester(tmp_path, Recorder(RecorderConfig()))
    dataset_cls = type(backtester.config.price_dataset)
    calls = []
    original = dataset_cls.delisting_bars

    def counting(self, prices, column):
        calls.append(prices.sizes["timestamp"])
        return original(self, prices, column)

    monkeypatch.setattr(dataset_cls, "delisting_bars", counting)
    backtester.run()

    assert calls == [WINDOW[1] - WINDOW[0] + 1]


def _halt(dataset_config, symbol, bars):
    """Blank every price of ``symbol`` on ``bars`` (a trading halt)."""
    store = xr.open_zarr(dataset_config.zarr_file_path).load()
    j = list(store.symbol.values).index(symbol)
    for name in store.data_vars:
        values = store[name].transpose("timestamp", "symbol").values.copy()
        values[bars, j] = np.nan
        store[name] = (("timestamp", "symbol"), values)
    store.to_zarr(dataset_config.zarr_file_path, mode="w")


def _engine_weights_at_close(result, close, bar) -> np.ndarray:
    """The simulation's holdings valued at ``bar``'s close, as weights of its portfolio value."""
    orders = result.simulation.orders
    symbols = list(result.weights.symbol.values)
    shares = np.zeros(len(symbols))
    for ts, symbol, size, side in zip(
        orders["timestamp"].values, orders["symbol"].values, orders["size"].values, orders["side"].values
    ):
        if ts <= bar:
            shares[symbols.index(str(symbol))] += size if side == "Buy" else -size
    close = close.sel(timestamp=bar).values
    value = float(result.simulation.value.sel(timestamp=bar))
    return shares * close / value


@pytest.mark.parametrize("sizing_basis", ["fill", "valuation"])
@pytest.mark.parametrize("fees, slippage", [(0.0, 0.0), (0.002, 0.001)])
def test_the_current_weights_are_the_holdings_the_simulation_carries(tmp_path, fees, slippage, sizing_basis):
    """The drifted weights handed to the rule at a rebalance bar equal the
    simulation's own holdings at that bar's close, the run's fees and
    slippage included, on either sizing basis (#119): filled at the next
    bar's open (no overnight move from the signal bar's close), carried
    across BBB's two-bar halt at its last price, then marked to the close
    once it trades again."""
    backtester, dataset_config, bars = _backtester(
        tmp_path, Recorder(RecorderConfig()), fees=fees, slippage=slippage, sizing_basis=sizing_basis
    )
    # A halt strictly inside the first holding period, after the fill bar.
    _halt(dataset_config, "BBB", [WINDOW[0] + 2, WINDOW[0] + 3])

    result = backtester.run()

    store = xr.open_zarr(dataset_config.zarr_file_path).load()
    close = store["adjClose"].ffill("timestamp").transpose("timestamp", "symbol")
    assert (SEEN[0].current_weights.values == 0.0).all()
    for k in range(1, len(SEEN)):
        bar = np.datetime64(SEEN[k].timestamp)
        np.testing.assert_allclose(
            SEEN[k].current_weights.values, _engine_weights_at_close(result, close, bar), rtol=1e-9, atol=1e-12
        )
    assert not np.allclose(SEEN[1].current_weights.values[:2], 0.5)


def test_a_failing_bar_holds_is_logged_and_recorded(tmp_path):
    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        backtester, _, bars = _backtester(
            tmp_path, Recorder(RecorderConfig(fail_on=1)), output_dir=str(tmp_path / "runs")
        )
        result = backtester.run()
    finally:
        logger.remove(sink)

    failed_bar = bars[WINDOW[0] + REBALANCE]
    weights = result.weights["weight"].sel(timestamp=failed_bar).values
    assert np.isnan(weights).all()
    assert any("solver gave up" in m and pd.Timestamp(failed_bar).isoformat() in m for m in messages)
    saved = json.loads((result.run_dir / "metrics.json").read_text())
    assert saved["portfolio_construction"] == {
        "failed_bar_count": 1,
        "failed_bars": [pd.Timestamp(failed_bar).isoformat()],
    }
    # The bar after it drifts from the last rebalance that did trade.
    assert (SEEN[2].current_weights.values[:2] > 0).all()


def _optimizer(**overrides):
    params = dict(
        expected_return_label="fwd_ret_1",
        risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=LOOKBACK)),
        ic=0.05,
        risk_aversion=5.0,
        turnover_penalty=0.001,
        weight_cap=0.4,
    )
    params.update(overrides)
    return MeanVarianceOptimizer(MeanVarianceConfig(**params))


def test_a_mean_variance_backtest_is_fully_invested_and_rebuilds_identically(tmp_path):
    backtester, _, _ = _backtester(tmp_path, _optimizer(), output_dir=str(tmp_path / "runs"))

    original = backtester.run()

    weights = original.weights["weight"].values
    rebalance = np.isfinite(weights).all(axis=1)
    assert rebalance.sum() == len(range(0, WINDOW[1] - WINDOW[0], REBALANCE))
    np.testing.assert_allclose(weights[rebalance].sum(axis=1), 1.0, atol=1e-9)
    assert (weights[rebalance] >= 0).all() and (weights[rebalance] <= 0.4 + 1e-9).all()
    assert original.metrics["portfolio_construction"]["failed_bar_count"] == 0

    saved = json.loads((original.run_dir / "config.json").read_text())
    assert saved["constructor"]["name"] == "quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer"
    assert saved["constructor"]["risk_model"]["lookback_bars"] == LOOKBACK
    rebuilt = load_backtester_from_config(saved)
    assert rebuilt.config.constructor == backtester.config.constructor
    again = rebuilt.run()

    np.testing.assert_array_equal(again.weights["weight"].values, weights)
    np.testing.assert_array_equal(again.simulation.value.values, original.simulation.value.values)


def test_a_predictor_without_the_expected_return_label_is_refused_at_construction(tmp_path):
    optimizer = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="fwd_ret_5",
            risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=LOOKBACK)),
            ic=0.05,
            risk_aversion=5.0,
        )
    )
    with pytest.raises(ValueError, match="fwd_ret_5"):
        _backtester(tmp_path, optimizer)


def test_a_long_short_mean_variance_backtest_is_dollar_neutral_and_rebuilds(tmp_path):
    optimizer = _optimizer(direction="long_short", risk_aversion=0.5, weight_cap=0.3, candidate_top_k=3)
    backtester, _, _ = _backtester(tmp_path, optimizer, output_dir=str(tmp_path / "runs"))

    original = backtester.run()

    weights = original.weights["weight"].values
    rebalance = np.isfinite(weights).all(axis=1)
    assert rebalance.sum() == len(range(0, WINDOW[1] - WINDOW[0], REBALANCE))
    np.testing.assert_allclose(weights[rebalance].sum(axis=1), 0.0, atol=1e-12)
    assert (np.abs(weights[rebalance]).sum(axis=1) <= 1 + 1e-12).all()
    assert (np.abs(weights[rebalance]) <= 0.3 + 1e-12).all()
    assert (weights[rebalance] < 0).any()
    assert original.metrics["portfolio_construction"]["failed_bar_count"] == 0

    saved = json.loads((original.run_dir / "config.json").read_text())
    assert saved["constructor"]["direction"] == "long_short"
    assert saved["constructor"]["candidate_top_k"] == 3
    rebuilt = load_backtester_from_config(saved)
    assert rebuilt.config.constructor == backtester.config.constructor
    again = rebuilt.run()

    np.testing.assert_array_equal(again.weights["weight"].values, weights)
    np.testing.assert_array_equal(again.simulation.value.values, original.simulation.value.values)


#: `context.factors` seen by every `_DeclaringRisk`, in call order.
FACTORS_SEEN: list = []


class _DeclaringRisk(LedoitWolfRiskModel):
    """Ledoit-Wolf that declares one factor and records what it is handed."""

    def __init__(self, config, factor):
        super().__init__(config)
        self._factor = factor

    def required_factors(self):
        return [self._factor]

    def estimate(self, context, volatility=None):
        FACTORS_SEEN.append((context.timestamp, context.factors))
        return super().estimate(context, volatility)


def test_a_declared_factor_reaches_the_risk_model_at_each_bar_only(tmp_path):
    FACTORS_SEEN.clear()

    def declaring(dataset_config):
        # A non-KunQuant factor whose first 3 bars are NaN without its warm-up.
        factor = PastReturnFactor(
            PolarsFactorConfig(warmup_bars=5, dataset=make_stock_dataset(dataset_config), kwargs={"n": 3})
        )
        return _optimizer(risk_model=_DeclaringRisk(LedoitWolfConfig(lookback_bars=LOOKBACK), factor))

    backtester, dataset_config, bars = _backtester(tmp_path, declaring)

    backtester.run()

    close = xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].transpose("timestamp", "symbol")
    symbols = close.symbol.values
    assert [ts for ts, _ in FACTORS_SEEN] == [
        pd.Timestamp(bars[t]) for t in range(WINDOW[0], WINDOW[1], REBALANCE)
    ]
    for ts, factors in FACTORS_SEEN:
        assert list(factors.data_vars) == ["past_ret_3"]
        assert factors["past_ret_3"].dims == ("symbol",)
        t = list(close.timestamp.values).index(np.datetime64(ts))
        expected = close.values[t] / close.values[t - 3] - 1.0
        got = factors["past_ret_3"].sel(symbol=symbols).values
        np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert np.isfinite(FACTORS_SEEN[0][1]["past_ret_3"].values).all()


class _RankedTargetHead(FirstFeatureHead):
    """Fitted on a transformed target, so it reports its label as standardized."""

    def _transform_target(self, y, training):
        return y, None


def test_raw_calibration_on_a_standardized_label_is_refused_when_the_backtest_is_built(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS, seed=11)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    model = make_model(
        tmp_path / "model",
        dataset_config,
        start_date=_day(bars[0]),
        end_date=_day(bars[39]),
        train_start=_day(bars[0]),
        train_end=_day(bars[34]),
        test_start=_day(bars[35]),
        test_end=_day(bars[39]),
        head=_RankedTargetHead,
    )
    assert model.label_scales == {"fwd_ret_1": "standardized"}
    optimizer = _optimizer(calibration="raw", ic=None)

    with pytest.raises(ValueError, match="'fwd_ret_1'.*standardized"):
        USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=make_stock_dataset(dataset_config),
                model=model,
                model_mode="train",
                start_date=_day(bars[WINDOW[0]]),
                end_date=_day(bars[WINDOW[1]]),
                output_dir=None,
                rebalance_periods=REBALANCE,
                constructor=optimizer,
            )
        )


def test_a_symbol_priced_at_the_bar_can_be_chosen_and_its_order_is_rejected(tmp_path):
    """BBB has a price on the first rebalance bar but none on the next: it is
    tradable at the bar (no look-ahead), so the rule buys it, and the engine
    rejects that order. The next rebalance starts from what was really held."""
    backtester, dataset_config, bars = _backtester(tmp_path, Recorder(RecorderConfig()))
    _halt(dataset_config, "BBB", [WINDOW[0] + 1])

    result = backtester.run()

    assert bool(SEEN[0].tradable.sel(symbol="BBB"))
    assert result.weights["weight"].sel(timestamp=bars[WINDOW[0]], symbol="BBB") == 0.5
    rejected = [(r["axis_symbol"], r["signal_timestamp"]) for r in result.simulation.rejected_orders]
    assert ("BBB", pd.Timestamp(bars[WINDOW[0]])) in rejected
    assert result.metrics["execution"]["rejected_order_count"] == len(result.simulation.rejected_orders)
    close = xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].load().ffill("timestamp").transpose("timestamp", "symbol")
    for context in SEEN[1:]:
        np.testing.assert_allclose(
            context.current_weights.values,
            _engine_weights_at_close(result, close, np.datetime64(context.timestamp)),
            rtol=1e-9,
            atol=1e-12,
        )
    assert SEEN[1].current_weights.sel(symbol="BBB") == 0.0


def test_a_holding_halted_at_a_rebalance_stays_locked(tmp_path):
    """BBB is held from the first rebalance and halted over the second one and
    its fill bar: it keeps its weight there and AAA gets the rest of the book."""
    backtester, dataset_config, bars = _backtester(tmp_path, Recorder(RecorderConfig()))
    second = WINDOW[0] + REBALANCE
    _halt(dataset_config, "BBB", [second, second + 1])

    result = backtester.run()

    locked = SEEN[1]
    assert bool(locked.locked.sel(symbol="BBB"))
    held = float(locked.current_weights.sel(symbol="BBB"))
    row = result.weights["weight"].sel(timestamp=bars[second])
    assert float(row.sel(symbol="BBB")) == held > 0
    assert float(row.sel(symbol="AAA")) == pytest.approx(1.0 - held, abs=1e-15)
    close = xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].load().ffill("timestamp").transpose("timestamp", "symbol")
    for context in SEEN[1:]:
        np.testing.assert_allclose(
            context.current_weights.values,
            _engine_weights_at_close(result, close, np.datetime64(context.timestamp)),
            rtol=1e-9,
            atol=1e-12,
        )


def test_a_mean_variance_backtest_keeps_a_halted_holding_locked_and_rebuilds(tmp_path):
    """The optimiser's largest first-rebalance holding halts over the second
    rebalance and its fill bar: the book keeps it at its drifted weight (the
    driver would refuse otherwise), nothing fails, and the run rebuilds."""
    probe, _, bars = _backtester(tmp_path / "probe", _optimizer())
    first = probe.run().weights["weight"].sel(timestamp=bars[WINDOW[0]])
    top = str(first.symbol.values[int(first.argmax())])

    backtester, dataset_config, bars = _backtester(tmp_path / "run", _optimizer(), output_dir=str(tmp_path / "runs"))
    second = WINDOW[0] + REBALANCE
    _halt(dataset_config, top, [second, second + 1])
    original = backtester.run()

    row = original.weights["weight"].sel(timestamp=bars[second])
    assert float(row.sel(symbol=top)) > 0
    assert float(row.sum()) == pytest.approx(1.0, abs=1e-9)
    assert original.metrics["portfolio_construction"]["failed_bar_count"] == 0
    saved = json.loads((original.run_dir / "config.json").read_text())
    again = load_backtester_from_config(saved).run()
    np.testing.assert_array_equal(again.weights["weight"].values, original.weights["weight"].values)


@dataclass(frozen=True)
class ReporterConfig:
    lookback_bars: int = 0


class Reporter(PortfolioConstructor):
    """Holds AAA and reports an event naming BBB on every bar."""

    config_cls = ReporterConfig

    def construct(self, context):
        row = xr.zeros_like(context.current_weights)
        row.loc[{"symbol": "AAA"}] = 1.0
        row.attrs["events"] = {"closed_without_risk": ["BBB"]}
        return row


def test_a_constructors_events_reach_metrics_json(tmp_path):
    backtester, _, bars = _backtester(tmp_path, Reporter(ReporterConfig()), output_dir=str(tmp_path / "runs"))

    result = backtester.run()

    block = json.loads((result.run_dir / "metrics.json").read_text())["portfolio_construction"]
    rebalances = [pd.Timestamp(bars[t]).isoformat() for t in range(WINDOW[0], WINDOW[1], REBALANCE)]
    assert block["closed_without_risk"] == {
        "count": len(rebalances),
        "bars": [{"bar": bar, "symbols": ["BBB"]} for bar in rebalances],
    }
