"""Model preparation and date alignment in `BaseBacktester.run()` (phase 03.7, plan 06).

This module locks D-06, D-13, D-14 and D-15, plus the read-cache trap that
sits underneath D-14.

- **D-15, bar-accurate warm-up.** The warm-up start is the largest factor
  `config.window`, counted in bars on the price dataset's own timestamp axis.
  `Factor._reset_dataset_config` subtracts `window` *calendar* days from the
  dataset start, and that is only an extra buffer. The warm-up test uses a
  5-bar lookback, so the buffer alone is not enough (5 calendar days before a
  Monday reach back only 3 bars). A zero-bar warm-up therefore goes red on the
  factor start date and on NaN first-bar predictions. Calendar-day
  arithmetic goes red too, because it lands on the preceding Wednesday rather
  than the Monday one week earlier. A short history clamps to the first bar
  and logs a warning that names the clamped start.
- **D-14, re-dating through the factor configs, and the read cache.**
  `XrBackend.read` returns early once the backend holds data, and
  `BaseDataset.read()` / `Factor.read()` then narrow that cached panel IN
  PLACE. A model trained first has already read its data narrowed to its own
  dates. Widening the factor dates for the backtest and reading again returns
  the same narrow panel, and nothing raises. RESEARCH Pitfall 1 measured it:
  a 10-bar store read from a later start gave 6 bars, still 6 after widening,
  and 10 only with `overwrite=True`. So the two strategy tests put the backtest
  window BEFORE the model's own start date. Without the dataset refresh
  ("cal"), or without the factor-store refresh ("read"), the window's first bar
  is absent from the features and its predictions are NaN.
- **D-06, the universe is every price symbol.** Predictions are reindexed onto
  the price dataset's symbol and timestamp axes before selection. A symbol the
  factor never saw scores NaN and gets 0.0 on every rebalance row. Predictions
  always come back symbol-sorted from `predict_panel`, so a price store written
  in a different symbol order has the same shape and a different order.
  Removing the reindex then either raises in the weights contract or selects
  by position. The alignment test checks both values and chosen names by
  symbol label.
- **D-13, model preparation.** "train" trains on the model's own
  train/test dates and never rewrites them. "load" refuses a missing file
  before any feature work. A DL head gets its feature panel collected before
  `load()`, because `DLModel._read_checkpoint` sizes the network from
  `num_symbols` on the model's data backend (RESEARCH Pitfall 11).
- **Fold-style dates.** `'2026-08-07T00:00:00.000000000'`, numpy datetimes
  and timestamps with a time all normalize to an ISO date through one helper
  (RESEARCH Pitfall 10).

Everything is synthetic, CPU-only and offline. Configs are constructed
directly, never through the factories in `quantlab/config/__init__.py` (D-32).
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import xarray as xr
from loguru import logger
from torch import nn

from quantlab.base.backtest import BaseBacktester
from quantlab.base.config import (
    CrossSectionBacktestConfig,
    DLConfig,
    MLConfig,
    PolarsFactorConfig,
)
from quantlab.base.model import DLModel
from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from tests.backtest_fixtures import (
    SYMBOLS,
    FirstFeatureHead,
    ForwardReturnLabel,
    PastReturnFactor,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
TOP_N = 2
REBALANCE_PERIODS = 5


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _bars(dataset_config) -> np.ndarray:
    return xr.open_zarr(dataset_config.zarr_file_path).timestamp.values


def _model_dates(bars, first: int, train_last: int, last: int) -> dict:
    return dict(
        start_date=_day(bars[first]),
        end_date=_day(bars[last]),
        train_start=_day(bars[first]),
        train_end=_day(bars[train_last]),
        test_start=_day(bars[train_last + 1]),
        test_end=_day(bars[last]),
    )


def _backtester(
    tmp_path: Path,
    price_config,
    model,
    bars,
    *,
    start_bar: int,
    end_bar: int,
    model_mode: str = "load",
    checkpoint=None,
) -> USEquityCrossectionSelectStockVectorBt:
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(price_config),
            model=model,
            model_mode=model_mode,
            checkpoint=None if checkpoint is None else str(checkpoint),
            start_date=_day(bars[start_bar]),
            end_date=_day(bars[end_bar]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=REBALANCE_PERIODS,
            direction="long_only",
            top_n=TOP_N,
            fees=0.0,
            slippage=0.0,
            init_cash=1_000_000.0,
        )
    )


def _loaded_model(tmp_path: Path, dataset_config, dates: dict, **model_kwargs):
    """A checkpoint trained by one model, and a fresh twin that will load it."""
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **dates, **model_kwargs)
    )
    model = make_model(tmp_path / "backtest", dataset_config, **dates, **model_kwargs)
    return model, checkpoint


def _adj_close(dataset_config) -> xr.DataArray:
    return (
        xr.open_zarr(dataset_config.zarr_file_path)["adjClose"]
        .load()
        .transpose("timestamp", "symbol")
    )


# --------------------------------------------------------------------------
# D-15: bar-accurate warm-up
# --------------------------------------------------------------------------


def test_warmup_counts_bars_on_the_price_calendar_not_calendar_days(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar = 30
    assert pd.Timestamp(bars[start_bar]).day_name() == "Monday"

    model, checkpoint = _loaded_model(
        tmp_path, dataset_config, _model_dates(bars, 0, 24, 29), n=5, window=5
    )
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=50, checkpoint=checkpoint,
    ).run()

    # 5 bars before Monday 2024-02-12 is Monday 2024-02-05 (7 calendar days).
    # Subtracting 5 calendar days would give Wednesday 2024-02-07.
    factor = model.config.factors[0]
    assert factor.config.start_date == _day(bars[start_bar - 5]) == "2024-02-05"

    # The 5-bar lookback of the first window bar is fully inside the warm-up.
    first = result.predictions["fwd_ret_1"].isel(timestamp=0)
    assert np.isfinite(first.values).all(), first.values


def test_warmup_clamps_to_first_bar_and_warns_on_short_history(
    tmp_path, warning_messages
):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)

    model, checkpoint = _loaded_model(
        tmp_path, dataset_config, _model_dates(bars, 0, 24, 29), window=5
    )
    _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=2, end_bar=20, checkpoint=checkpoint,
    ).run()

    first_day = _day(bars[0])
    assert model.config.factors[0].config.start_date == first_day

    shortfall = [m for m in warning_messages if "warm-up" in m]
    assert len(shortfall) == 1, warning_messages
    message = shortfall[0]
    assert "5 bars" in message
    assert "only 2" in message
    assert "short by 3" in message
    assert first_day in message


# --------------------------------------------------------------------------
# D-14: factor re-dating
# --------------------------------------------------------------------------


def _two_factor_model(root: Path, dataset_config, dates: dict) -> FirstFeatureHead:
    factors = [
        PastReturnFactor(
            PolarsFactorConfig(
                window=window,
                dataset=make_stock_dataset(dataset_config),
                kwargs={"n": n},
            )
        )
        for n, window in ((1, 3), (2, 7))
    ]
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            window=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": 1},
        )
    )
    return FirstFeatureHead(
        MLConfig(
            factors=factors,
            labels=[label],
            model_save_dir=str(root / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            val_size=0.0,
            **dates,
        )
    )


def test_factor_dates_are_pushed_and_predictions_cover_exactly_the_window(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar, end_bar = 30, 50

    model = _two_factor_model(tmp_path, dataset_config, _model_dates(bars, 0, 24, 29))
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=end_bar, model_mode="train",
    ).run()

    # The warm-up is the LARGEST window (7), applied to every factor.
    for factor in model.config.factors:
        assert factor.config.start_date == _day(bars[start_bar - 7])
        assert factor.config.end_date == _day(bars[end_bar])

    np.testing.assert_array_equal(
        result.predictions.timestamp.values.astype("datetime64[ns]"),
        bars[start_bar : end_bar + 1].astype("datetime64[ns]"),
    )
    # The model's own end date is bar 29; the last window bar is predicted
    # only because the end date was pushed into the factors.
    last = result.predictions["fwd_ret_1"].isel(timestamp=-1)
    assert np.isfinite(last.values).all(), last.values


def _strategy_model(tmp_path: Path, dataset_config, bars, strategy: str):
    """A model whose own dates (bars 35..59) start AFTER the backtest warm-up."""
    file_path = None
    if strategy == "read":
        file_path = str(tmp_path / "factors" / "past_ret.zarr")
        PastReturnFactor(
            PolarsFactorConfig(
                window=5,
                dataset=make_stock_dataset(dataset_config),
                file_path=file_path,
                kwargs={"n": 1},
            )
        ).cal().save(mode="w")

    factor = PastReturnFactor(
        PolarsFactorConfig(
            window=5,
            dataset=make_stock_dataset(dataset_config),
            file_path=file_path,
            kwargs={"n": 1},
        )
    )
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            window=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": 1},
        )
    )
    return FirstFeatureHead(
        MLConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(tmp_path / "models"),
            factor_data_strategy=strategy,
            label_data_strategy="cal",
            val_size=0.0,
            **_model_dates(bars, 35, 49, 59),
        )
    )


def _assert_first_window_bar_is_predicted_and_traded(result, bars, start_bar):
    first = result.predictions["fwd_ret_1"].sel(timestamp=bars[start_bar])
    assert np.isfinite(first.values).all(), first.values
    first_rebalance = result.weights["weight"].isel(timestamp=0).values
    assert np.count_nonzero(first_rebalance) == TOP_N, first_rebalance


def test_read_strategy_predicts_over_the_widened_range_after_training(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar = 25

    model = _strategy_model(tmp_path, dataset_config, bars, "read")
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=45, model_mode="train",
    ).run()

    _assert_first_window_bar_is_predicted_and_traded(result, bars, start_bar)


def test_cal_strategy_predicts_over_the_widened_range_after_training(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar = 25

    model = _strategy_model(tmp_path, dataset_config, bars, "cal")
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=45, model_mode="train",
    ).run()

    _assert_first_window_bar_is_predicted_and_traded(result, bars, start_bar)


# --------------------------------------------------------------------------
# D-06: the universe is every price symbol
# --------------------------------------------------------------------------


def test_symbol_absent_from_predictions_gets_zero_weight_on_rebalance_rows(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    absent = SYMBOLS[-1]
    dates = _model_dates(bars, 0, 24, 29)

    _, checkpoint = _loaded_model(tmp_path, dataset_config, dates)
    model = make_model(tmp_path / "backtest", dataset_config, **dates)
    narrowed = dataclasses.replace(dataset_config, symbols=tuple(SYMBOLS[:-1]))
    model.config.factors[0].config.dataset = make_stock_dataset(narrowed)

    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoint,
    ).run()

    assert result.weights.symbol.values.tolist() == SYMBOLS
    assert np.isnan(result.predictions["fwd_ret_1"].sel(symbol=absent).values).all()
    weight = result.weights["weight"]
    rebalance = np.isfinite(weight.values).all(axis=1)
    assert rebalance.sum() == 4
    absent_weight = weight.sel(symbol=absent).values
    assert (absent_weight[rebalance] == 0.0).all()
    assert np.isnan(absent_weight[~rebalance]).all()


def test_predictions_align_to_price_symbols_by_label_not_position(tmp_path):
    """Carried from plans 01 and 03: `CrossSectionTopNSelector.select` pairs
    scores with fill prices by shape only. `predict_panel` returns symbols
    sorted, so a price store written in reverse order has the same shape and
    a different order. Only the reindex onto `prices.symbol` keeps them paired.
    """
    reversed_symbols = list(reversed(SYMBOLS))
    dataset_config = write_price_store(tmp_path, symbols=reversed_symbols, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar = 30

    model, checkpoint = _loaded_model(
        tmp_path, dataset_config, _model_dates(bars, 0, 24, 29)
    )
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=50, checkpoint=checkpoint,
    ).run()

    assert result.predictions.symbol.values.tolist() == reversed_symbols
    assert result.weights.symbol.values.tolist() == reversed_symbols

    adj_close = _adj_close(dataset_config)
    past_return = (
        adj_close.isel(timestamp=start_bar) / adj_close.isel(timestamp=start_bar - 1)
        - 1.0
    )
    for symbol in reversed_symbols:
        assert float(
            result.predictions["fwd_ret_1"]
            .isel(timestamp=0)
            .sel(symbol=symbol)
        ) == pytest.approx(float(past_return.sel(symbol=symbol)), rel=1e-12)

    expected_top = set(past_return.to_series().nlargest(TOP_N).index)
    first_row = result.weights["weight"].isel(timestamp=0)
    chosen = {str(s) for s in first_row.symbol.values[first_row.values > 0]}
    assert chosen == expected_top


# --------------------------------------------------------------------------
# Pitfall 10: fold-style dates
# --------------------------------------------------------------------------


def test_iso_date_normalizes_fold_style_strings():
    fold_style = np.datetime_as_string(np.datetime64("2026-08-07", "ns"))
    assert fold_style == "2026-08-07T00:00:00.000000000"
    assert BaseBacktester._iso_date(fold_style) == "2026-08-07"
    assert BaseBacktester._iso_date(np.str_(fold_style)) == "2026-08-07"
    assert BaseBacktester._iso_date(np.datetime64("2026-08-07")) == "2026-08-07"
    assert BaseBacktester._iso_date(pd.Timestamp("2026-08-07 15:30")) == "2026-08-07"


# --------------------------------------------------------------------------
# D-13: model preparation
# --------------------------------------------------------------------------


class TinyLinearDLHead(DLModel):
    """The smallest trainable `DLModel`: one `nn.Linear` on the last axis."""

    def _init_model(self, num_symbols, num_features, num_labels, hyperparameters):
        return nn.Linear(num_features, num_labels)

    def _init_optim(self, model):
        return torch.optim.SGD(model.parameters(), lr=1e-2)

    def _preprocess(self, data):
        return torch.nan_to_num(data, nan=0.0)

    def _train_one_batch(self, epoch, x, y):
        self.optim.zero_grad()
        loss = F.mse_loss(self.model(x), y)
        loss.backward()
        self.optim.step()
        return loss.detach()

    def _val_one_batch(self, epoch, x, y):
        return F.mse_loss(self.model(x), y)

    def _test_one_batch(self, epoch, x, y):
        return F.mse_loss(self.model(x), y)


def _dl_model(root: Path, dataset_config, dates: dict) -> TinyLinearDLHead:
    factor = PastReturnFactor(
        PolarsFactorConfig(
            window=5, dataset=make_stock_dataset(dataset_config), kwargs={"n": 1}
        )
    )
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            window=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": 1},
        )
    )
    return TinyLinearDLHead(
        DLConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(root / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            epochs=1,
            batch_size=16,
            num_workers=0,
            val_size=0.0,
            **dates,
        )
    )


def test_train_mode_uses_the_models_own_dates_and_leaves_them_unchanged(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    dates = _model_dates(bars, 0, 24, 29)
    model = make_model(tmp_path, dataset_config, **dates)
    fields = ("train_start", "train_end", "test_start", "test_end")

    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, model_mode="train",
    ).run()

    assert {f: getattr(model.config, f) for f in fields} == {f: dates[f] for f in fields}
    checkpoints = sorted(Path(model.config.model_save_dir).rglob("*.joblib"))
    assert len(checkpoints) == 1, checkpoints
    first = result.predictions["fwd_ret_1"].isel(timestamp=0)
    assert np.isfinite(first.values).all(), first.values


def test_load_mode_with_missing_checkpoint_file_fails_before_predicting(
    tmp_path, monkeypatch
):
    """A DL head is used because it is the variant that does feature work
    before `load()`: the existence check must come first."""
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    model = _dl_model(tmp_path, dataset_config, _model_dates(bars, 0, 24, 29))
    missing = tmp_path / "never_trained" / "TinyLinearDLHead_total.pth"

    calls = {"collect": 0, "predict_panel": 0}
    collect, predict_panel = model._collect_all_features, model.predict_panel

    def spy_collect():
        calls["collect"] += 1
        return collect()

    def spy_predict_panel(features):
        calls["predict_panel"] += 1
        return predict_panel(features)

    monkeypatch.setattr(model, "_collect_all_features", spy_collect)
    monkeypatch.setattr(model, "predict_panel", spy_predict_panel)

    backtester = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, checkpoint=missing,
    )
    with pytest.raises(FileNotFoundError, match="never_trained"):
        backtester.run()
    assert calls == {"collect": 0, "predict_panel": 0}


def test_load_mode_without_checkpoint_is_refused_at_construction(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    model = make_model(tmp_path, dataset_config, **_model_dates(bars, 0, 24, 29))

    with pytest.raises(ValueError, match="checkpoint"):
        _backtester(
            tmp_path, dataset_config, model, bars,
            start_bar=30, end_bar=50, model_mode="load", checkpoint=None,
        )


@pytest.mark.parametrize(
    "backtest_model_kwargs",
    [{"n": 2}, {"n_forward_periods": 3}],
    ids=["different-factor", "different-label"],
)
def test_load_refuses_a_checkpoint_trained_on_other_variables(
    tmp_path, monkeypatch, backtest_model_kwargs
):
    """Code review WR-01: a checkpoint is checked against config.model's variables.

    The checkpoint is trained on `past_ret_1` -> `fwd_ret_1`. The backtest
    model declares the same feature COUNT but a different factor (or a
    different label). The deterministic head only reads feature 0 by
    position, exactly like xgboost only checks the feature count, so the old
    `run()` predicted on the wrong inputs without complaint and this test
    went red. The refusal must name both variable lists and come before any
    feature is computed.
    """
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    dates = _model_dates(bars, 0, 24, 29)
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **dates))
    model = make_model(
        tmp_path / "backtest", dataset_config, **dates, **backtest_model_kwargs
    )
    computed: list[int] = []
    collect = model._collect_all_features
    monkeypatch.setattr(
        model, "_collect_all_features", lambda: computed.append(1) or collect()
    )
    backtester = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoint,
    )

    with pytest.raises(ValueError, match="was trained on") as excinfo:
        backtester.run()

    message = str(excinfo.value)
    declared = model.get_factor_names() + model.get_label_names()
    assert any(name not in ("past_ret_1", "fwd_ret_1") and name in message for name in declared)
    assert str(checkpoint) in message
    assert computed == []


def test_load_uses_the_checkpoints_train_dates_over_stale_config_dates(
    tmp_path, warning_messages
):
    """Code review WR-01: D-17 classifies bars with the dates the checkpoint trained on.

    The checkpoint trained through bar 24 (1-bar horizon, so bar 25 is still
    in-sample). The backtest model carries a stale `train_end` of bar 10.
    Trusting config.model, the old code put the training window end at bar 11
    and reported bars 12..25, which the model had trained on, as
    out-of-sample. It goes red here on the window, the in-sample range and the
    missing warning that names both date pairs.
    """
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **_model_dates(bars, 0, 24, 29))
    )
    stale = make_model(
        tmp_path / "backtest", dataset_config, **_model_dates(bars, 0, 10, 29)
    )

    result = _backtester(
        tmp_path, dataset_config, stale, bars,
        start_bar=20, end_bar=45, checkpoint=checkpoint,
    ).run()

    assert tuple(result.metrics["training_window"]) == (_day(bars[0]), _day(bars[25]))
    assert tuple(result.metrics["in_sample_range"]) == (_day(bars[20]), _day(bars[25]))
    stale_warnings = [
        m
        for m in warning_messages
        if "using the checkpoint's dates" in m and _day(bars[24]) in m and _day(bars[10]) in m
    ]
    assert len(stale_warnings) == 1, warning_messages


def test_load_without_a_checkpoint_config_json_warns_and_trusts_config_model(
    tmp_path, warning_messages
):
    """Code review WR-01: a checkpoint with no config.json beside it cannot be checked.

    That is not necessarily an error (a hand-copied checkpoint), so the run
    continues with config.model as given, but it must say so. The old code
    never looked for the record and emitted no warning, so this goes red.
    """
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    dates = _model_dates(bars, 0, 24, 29)
    model, checkpoint = _loaded_model(tmp_path, dataset_config, dates)
    (checkpoint.parent / "config.json").unlink()

    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoint,
    ).run()

    assert tuple(result.metrics["training_window"]) == (_day(bars[0]), _day(bars[25]))
    unchecked = [m for m in warning_messages if "has no config.json" in m]
    assert len(unchecked) == 1, warning_messages
    assert str(checkpoint) in unchecked[0]


def _ns(bar) -> str:
    """The text shape `train_cv` writes: `np.datetime_as_string` of a ns bar."""
    return np.datetime_as_string(np.asarray(bar).astype("datetime64[ns]"))


def _ns_model_dates(bars, first: int, train_last: int, last: int) -> dict:
    dates = _model_dates(bars, first, train_last, last)
    dates.update(
        train_start=_ns(bars[first]),
        train_end=_ns(bars[train_last]),
        test_start=_ns(bars[train_last + 1]),
        test_end=_ns(bars[last]),
    )
    return dates


def _t00_model_dates(bars, first: int, train_last: int, last: int) -> dict:
    dates = _model_dates(bars, first, train_last, last)
    for key in ("train_start", "train_end", "test_start", "test_end"):
        dates[key] = dates[key] + "T00:00:00"
    return dates


@pytest.mark.parametrize(
    ("checkpoint_dates", "config_dates"),
    [(_ns_model_dates, _model_dates), (_model_dates, _t00_model_dates)],
    ids=["ns-checkpoint-vs-plain-config", "plain-checkpoint-vs-T00-config"],
)
def test_load_mode_same_train_dates_in_another_text_format_log_no_date_warning(
    tmp_path, warning_messages, checkpoint_dates, config_dates
):
    """UAT gap G-03.7-7: the same training bars written differently are the same window.

    The checkpoint's config.json records one text form of the dates (a train_cv
    nanosecond string, or a plain date) and config.model carries another form
    of the SAME instants. The old string comparison warned "using the
    checkpoint's dates" for both. On daily bars both forms select the same
    training bars, so nothing may warn, and the numbers stay as they were.
    """
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **checkpoint_dates(bars, 0, 24, 29))
    )
    model = make_model(
        tmp_path / "backtest", dataset_config, **config_dates(bars, 0, 24, 29)
    )

    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=20, end_bar=45, checkpoint=checkpoint,
    ).run()

    stale = [m for m in warning_messages if "using the checkpoint's dates" in m]
    assert stale == [], stale
    assert tuple(result.metrics["training_window"]) == (_day(bars[0]), _day(bars[25]))


def test_same_training_bars_resolves_endpoints_on_the_calendar():
    """G-03.7-7: two date pairs match iff they select the same bars (slice semantics).

    Bare Timestamp equality is not enough: on intraday bars "2024-02-09"
    includes the whole day while the nanosecond midnight stops at the previous
    session's last bar. Endpoints past the calendar are compared as instants,
    so a stale date that clips to the same last bar still counts as different.
    """
    same = BaseBacktester._same_training_bars

    sessions = [
        pd.date_range(f"{day} 09:30", f"{day} 16:00", freq="30min")
        for day in ("2024-02-07", "2024-02-08", "2024-02-09")
    ]
    intraday = np.concatenate([s.values for s in sessions]).astype("datetime64[ns]")
    assert not same(
        intraday,
        ("2024-02-07", "2024-02-09"),
        ("2024-02-07", "2024-02-09T00:00:00.000000000"),
    )

    daily = pd.bdate_range("2024-01-01", "2024-03-29").values.astype("datetime64[ns]")
    plain = ("2024-01-01", "2024-02-09")
    assert same(
        daily, plain, ("2024-01-01T00:00:00.000000000", "2024-02-09T00:00:00.000000000")
    )
    assert same(daily, plain, ("2024-01-01T00:00:00", "2024-02-09T00:00:00"))
    assert not same(daily, plain, ("2024-01-01", "2024-02-08"))

    # Both train_end endpoints lie after the calendar's last bar (2024-03-29).
    assert same(
        daily, ("2024-01-01", "2024-04-05"), ("2024-01-01", "2024-04-05T00:00:00.000000000")
    )
    assert not same(daily, ("2024-01-01", "2024-04-05"), ("2024-01-01", "2024-04-08"))

    assert same(daily, (None, "2024-02-09"), (None, "2024-02-09"))
    assert not same(daily, plain, (None, "2024-02-09"))


def test_dl_head_loads_after_its_feature_panel_is_collected(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    trainer = _dl_model(tmp_path, dataset_config, _model_dates(bars, 0, 24, 29))
    trainer.collect()
    trainer.train()
    checkpoints = sorted(Path(trainer.config.model_save_dir).rglob("*.pth"))
    assert len(checkpoints) == 1, checkpoints

    fresh = TinyLinearDLHead(trainer.config)
    assert fresh.model is None
    result = _backtester(
        tmp_path, dataset_config, fresh, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoints[0],
    ).run()

    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(fresh.model.state_dict()[name].cpu(), tensor.cpu()), name
    first = result.predictions["fwd_ret_1"].isel(timestamp=0)
    assert np.isfinite(first.values).all(), first.values


def _trained_dl_checkpoint(tmp_path: Path, dataset_config, bars) -> tuple[TinyLinearDLHead, Path]:
    trainer = _dl_model(tmp_path, dataset_config, _model_dates(bars, 0, 24, 29))
    trainer.collect()
    trainer.train()
    checkpoints = sorted(Path(trainer.config.model_save_dir).rglob("*.pth"))
    assert len(checkpoints) == 1, checkpoints
    return trainer, checkpoints[0]


def test_backtester_delegates_the_variable_check_to_the_model(tmp_path, monkeypatch):
    """G-03.7-9: one variable check, in the model layer, keyed on `trained_on`.

    The backtester used to carry its own comparison against the factor config
    field `factors[].factor_names`. It refused a model's own checkpoint when
    that field was ordered differently from the derived names, and accepted
    permuted inputs when the derived names had drifted. It must be gone, and
    `_load_model_checkpoint` must call `model._assert_trained_variables` before
    the DL-only feature collection, so a refusal still precedes any feature
    work. The old code has both attributes and never calls the model check
    before collection, so this goes red.
    """
    assert not hasattr(BaseBacktester, "_assert_checkpoint_variables")
    assert not hasattr(BaseBacktester, "_saved_variable_names")

    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    trainer, checkpoint = _trained_dl_checkpoint(tmp_path, dataset_config, bars)
    fresh = TinyLinearDLHead(trainer.config)
    events: list[str] = []
    check, collect = fresh._assert_trained_variables, fresh._collect_all_features
    monkeypatch.setattr(
        fresh, "_assert_trained_variables", lambda p: events.append("check") or check(p)
    )
    monkeypatch.setattr(
        fresh, "_collect_all_features", lambda: events.append("collect") or collect()
    )

    _backtester(
        tmp_path, dataset_config, fresh, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoint,
    ).run()

    assert "check" in events and "collect" in events, events
    assert events.index("check") < events.index("collect"), events


def test_dl_load_without_config_json_warns_once_per_concern(tmp_path, warning_messages):
    """A DL checkpoint with no config.json beside it: the backtester says once
    that the training dates cannot be checked ("has no config.json"), and the
    model says once that the variables cannot be checked (G-03.7-9), even
    though the model check runs both before collection and again inside
    `load()`. The run completes."""
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    trainer, checkpoint = _trained_dl_checkpoint(tmp_path, dataset_config, bars)
    (checkpoint.parent / "config.json").unlink()
    fresh = TinyLinearDLHead(trainer.config)

    result = _backtester(
        tmp_path, dataset_config, fresh, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoint,
    ).run()

    no_sidecar = [m for m in warning_messages if "has no config.json" in m]
    assert len(no_sidecar) == 1, warning_messages
    assert str(checkpoint) in no_sidecar[0]
    unchecked = [m for m in warning_messages if "trained on cannot be checked" in m]
    assert len(unchecked) == 1, warning_messages
    assert str(checkpoint) in unchecked[0]
    first = result.predictions["fwd_ret_1"].isel(timestamp=0)
    assert np.isfinite(first.values).all(), first.values
