"""A label is a factor shifted forward: ``Forward(ForwardConfig(factor, span, delay))``.

The label at bar t is the wrapped factor at bar t + delay + span. ``read`` and
``compute`` read that many bars past the requested end on the wrapped factor's
dataset calendar, so the last requested bars are filled wherever the dataset
has later bars, and NaN only past the calendar's end.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.base.config import (
    DatasetConfig,
    FactorConfig,
    ForwardConfig,
    PolarsFactorConfig,
)
from quantlab.base.factor import FactorKunQuant
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.momentum import Momentum
from quantlab.label.forward import Forward
from quantlab.utils.module import load_factor_from_config


class MaDeviation(FactorKunQuant):
    """Close over its 5-bar average, minus one."""

    def _get_factor_names(self):
        return ("ma_dev_5",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev_5")
        return Function(builder.ops)


def _kunquant(dataset_config: DatasetConfig, tmp_path: Path) -> MaDeviation:
    return MaDeviation(
        FactorConfig(
            warmup_bars=5,
            dataset=SpotKlineDataset(dataset_config),
            mode="batch",
            data_columns=("close",),
            file_path=str(tmp_path / "factors" / "ma_dev.zarr"),
            njobs=2,
        )
    )


def _momentum(dataset_config: DatasetConfig, tmp_path: Path) -> Momentum:
    return Momentum(
        PolarsFactorConfig(
            warmup_bars=5,
            dataset=SpotKlineDataset(dataset_config),
            file_path=str(tmp_path / "factors" / "momentum.zarr"),
            kwargs={"n": 5},
        )
    )


@pytest.fixture(params=["kunquant", "polars"])
def factor(request, spot_kline_zarr, tmp_path):
    """A 5-bar factor of either backend over 60 daily bars, 2024-01-01 to 2024-02-29."""
    build = _kunquant if request.param == "kunquant" else _momentum
    return build(spot_kline_zarr(periods=60), tmp_path)


def _values(panel, name):
    return panel[name].transpose("timestamp", "symbol").values


def _assert_same(got, expected):
    # KunQuant computes in float32, and a rolling sum started on another bar
    # differs in the last bits.
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_a_label_at_t_is_the_factor_at_t_plus_delay_plus_span(factor):
    label = Forward(ForwardConfig(factor=factor, span=3, delay=1))
    name = factor.get_factor_names()[0]

    got = label.compute("2024-01-10", "2024-01-20")
    # 2024-01-14 .. 2024-01-24 is 4 bars (delay + span) later on the daily calendar.
    expected = factor.compute("2024-01-14", "2024-01-24")

    assert got["timestamp"].values.tolist() == (
        factor.compute("2024-01-10", "2024-01-20")["timestamp"].values.tolist()
    )
    _assert_same(_values(got, name), _values(expected, name))


def test_the_last_bars_are_nan_only_past_the_calendar_end(factor):
    label = Forward(ForwardConfig(factor=factor, span=3, delay=1))
    name = factor.get_factor_names()[0]

    # The store ends on 2024-02-29; 4 bars after 2024-02-25 is that last bar.
    panel = label.compute("2024-02-20", "2024-02-29")
    values = _values(panel, name)

    assert panel["timestamp"].size == 10
    assert not np.isnan(values[:6]).any()
    assert np.isnan(values[6:]).all()


def test_a_request_ending_inside_the_data_has_no_nan_tail(factor):
    label = Forward(ForwardConfig(factor=factor, span=3, delay=1))
    name = factor.get_factor_names()[0]

    values = _values(label.compute("2024-01-10", "2024-01-20"), name)

    assert not np.isnan(values).any()


def test_read_shifts_the_wrapped_factor_store(factor):
    factor.build("2024-01-01", "2024-02-29")
    label = Forward(ForwardConfig(factor=factor, span=3, delay=1))
    name = factor.get_factor_names()[0]

    got = label.read("2024-01-10", "2024-01-20")
    expected = factor.read("2024-01-14", "2024-01-24")

    assert got["timestamp"].size == 11
    _assert_same(_values(got, name), _values(expected, name))


def test_read_refuses_a_store_that_stops_before_the_label_lookahead(factor):
    # The dataset has bars to 2024-02-29, but the store stops on 2024-01-20.
    factor.build("2024-01-01", "2024-01-20")
    label = Forward(ForwardConfig(factor=factor, span=3, delay=1))

    with pytest.raises(ValueError, match="extend"):
        label.read("2024-01-10", "2024-01-20")


def test_a_label_reports_its_lookahead_and_span(factor):
    label = Forward(ForwardConfig(factor=factor, span=5, delay=2))

    assert label.lookahead_bars() == 7
    assert label.span_bars() == 5


def test_delay_defaults_to_one_bar(factor):
    assert Forward(ForwardConfig(factor=factor, span=5)).lookahead_bars() == 6


@pytest.mark.parametrize(
    ("span", "delay", "match"),
    [(0, 1, "span must be at least 1"), (3, -1, "delay must be non-negative")],
)
def test_a_label_refuses_an_impossible_shift(factor, span, delay, match):
    with pytest.raises(ValueError, match=match):
        Forward(ForwardConfig(factor=factor, span=span, delay=delay))


def test_a_label_refuses_a_stream_mode_factor(spot_kline_zarr, tmp_path):
    batch = _kunquant(spot_kline_zarr(periods=60), tmp_path)
    stream = MaDeviation(dataclasses.replace(batch.config, mode="stream"))

    with pytest.raises(ValueError, match="stream"):
        Forward(ForwardConfig(factor=stream, span=3))


def test_a_label_refuses_a_resampled_factor(factor):
    resampled = factor.resample("1d", "last")

    with pytest.raises(ValueError, match="resampled"):
        Forward(ForwardConfig(factor=resampled, span=3))


def test_a_label_rebuilds_from_its_config(factor):
    label = Forward(ForwardConfig(factor=factor, span=3, delay=2))

    config = json.loads(json.dumps(label.get_config()))
    rebuilt = load_factor_from_config(config)

    assert config["name"] == "quantlab.label.forward.Forward"
    assert rebuilt == label
    xr.testing.assert_allclose(
        rebuilt.compute("2024-01-10", "2024-01-20"),
        label.compute("2024-01-10", "2024-01-20"),
    )


def test_a_label_keeps_the_factor_variable_names(factor):
    label = Forward(ForwardConfig(factor=factor, span=3))

    assert label.get_factor_names() == factor.get_factor_names()
    assert list(label.compute("2024-01-10", "2024-01-20").data_vars) == list(
        factor.get_factor_names()
    )


def test_build_covers_the_lookahead_past_end_so_read_answers_up_to_end(factor):
    label = Forward(ForwardConfig(factor=factor, span=3, delay=1))

    assert label.build("2024-01-01", "2024-01-20") is label
    assert label.store_range() == ("2024-01-01", "2024-01-24T00:00:00")
    name = factor.get_factor_names()[0]
    _assert_same(
        _values(label.read("2024-01-10", "2024-01-20"), name),
        _values(label.compute("2024-01-10", "2024-01-20"), name),
    )


def test_extend_moves_the_store_end_by_the_lookahead(factor):
    label = Forward(ForwardConfig(factor=factor, span=3, delay=1))
    label.build("2024-01-01", "2024-01-20")

    assert label.extend("2024-02-10") is label
    assert label.store_range() == ("2024-01-01", "2024-02-14T00:00:00")
