"""``ChunkedFactor``: a factor computed one chunk of time at a time.

When a range would not fit in memory, the wrapper splits it into chunks of
one calendar period each (``TimeChunkPlanner``), each warmed up on its own.
``build`` and ``extend`` write chunk by chunk through the wrapped factor's
own ``build`` and ``extend``, so the store is only ever the owner's;
``compute`` fills one in-memory panel chunk by chunk, or refuses when the
output alone would not fit. Joined in time order, the chunks equal the
whole range computed at once: NaN in the same cells, values within
floating-point tolerance.

The prices here are 91 daily bars over January to March 2024, so a
``"month"`` granularity gives three chunks.
"""

import json

import numpy as np
import pytest
import xarray as xr
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

import quantlab.factor.predefined.chunked as chunked_module
from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import (
    ChunkedConfig, FactorConfig, NeutralizedConfig, PolarsFactorConfig, RosterConfig,
)
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.predefined.chunked import ChunkedFactor
from quantlab.factor.predefined.momentum import Momentum
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.factor.predefined.roster import RosterFactor

_T = 91
_TIMES = np.datetime64("2024-01-01") + np.arange(_T).astype("timedelta64[D]")
_START, _END = "2024-01-01", "2024-03-31"
_SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


class RollingMean(FactorKunQuant):
    """``ma_dev``: the close over its 5-bar mean, minus 1."""

    def _get_factor_names(self):
        return ("ma_dev",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("Close")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev")
        return Function(builder.ops)


def _prices() -> xr.Dataset:
    rng = np.random.default_rng(7)
    close = 50 + rng.normal(0, 1, size=(_T, len(_SYMBOLS))).cumsum(axis=0)
    close[:20, 2] = np.nan  # a late listing: NaN must land in the same cells
    return xr.Dataset(
        {"Close": (("timestamp", "symbol"), close)},
        coords={"timestamp": _TIMES, "symbol": _SYMBOLS},
    )


@pytest.fixture
def prices(tmp_path):
    return FrameDataset(_prices()).to_zarr(tmp_path / "prices.zarr")


def _kunquant(prices, store) -> RollingMean:
    return RollingMean(FactorConfig(
        warmup_bars=4, dataset=prices, mode="batch", data_columns=("Close",),
        file_path=str(store), njobs=2,
    ))


def _polars(prices, store) -> Momentum:
    return Momentum(PolarsFactorConfig(
        warmup_bars=3, dataset=prices, file_path=str(store), kwargs={"n": 3},
    ))


@pytest.fixture(params=[_kunquant, _polars], ids=["kunquant", "polars"])
def make(request):
    return request.param


def _stored(factor) -> xr.Dataset:
    return xr.open_zarr(factor.store_path).load()


# A rolling sum is kept as a running total from the first bar computed, so a
# chunk that starts later differs in the last bits of its dtype: about one
# float32 step of the inputs' scale for KunQuant, which then survives the
# cancellation in a ratio minus one.
_TOLERANCE = {
    np.dtype("float32"): {"rtol": 1e-5, "atol": 1e-6},
    np.dtype("float64"): {"rtol": 1e-10, "atol": 1e-12},
}


def _assert_same(chunked: xr.Dataset, whole: xr.Dataset) -> None:
    assert list(chunked.data_vars) == list(whole.data_vars)
    xr.testing.assert_equal(chunked["timestamp"], whole["timestamp"])
    xr.testing.assert_equal(chunked["symbol"], whole["symbol"])
    for name in whole.data_vars:
        got, want = chunked[name].values, whole[name].values
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_allclose(got, want, equal_nan=True, **_TOLERANCE[got.dtype])


def test_a_chunked_build_equals_the_whole_build(tmp_path, prices, make):
    whole = make(prices, tmp_path / "whole.zarr").build(_START, _END)
    inner = make(prices, tmp_path / "chunked.zarr")

    ChunkedFactor(ChunkedConfig(factor=inner, granularity="month")).build(_START, _END)

    _assert_same(_stored(inner), _stored(whole))
    assert inner.store_range() == (_START, _END)


def test_a_chunked_extend_equals_the_whole_build(tmp_path, prices, make):
    whole = make(prices, tmp_path / "whole.zarr").build(_START, _END)
    inner = make(prices, tmp_path / "chunked.zarr").build(_START, "2024-01-10")

    ChunkedFactor(ChunkedConfig(factor=inner, granularity="month")).extend(_END)

    _assert_same(_stored(inner), _stored(whole))
    assert inner.store_range() == (_START, _END)


def test_the_chunks_are_one_calendar_period_each(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    calls = []
    build, extend = inner.build, inner.extend
    monkeypatch.setattr(inner, "build", lambda s, e: calls.append(("build", s, e)) or build(s, e))
    monkeypatch.setattr(inner, "extend", lambda e: calls.append(("extend", e)) or extend(e))

    ChunkedFactor(ChunkedConfig(factor=inner, granularity="month")).build(_START, _END)

    assert [(c[0], str(c[-1])[:10]) for c in calls] == [
        ("build", "2024-01-31"), ("extend", "2024-02-29"), ("extend", _END),
    ]
    assert calls[0][1] == _START


def test_a_failed_build_resumes_with_extend(tmp_path, prices, monkeypatch):
    whole = _kunquant(prices, tmp_path / "whole.zarr").build(_START, _END)
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    wrapper = ChunkedFactor(ChunkedConfig(factor=inner, granularity="month"))
    extend = inner.extend

    def crash_in_march(end):
        if str(end) >= "2024-03":
            raise MemoryError("killed")
        return extend(end)

    monkeypatch.setattr(inner, "extend", crash_in_march)
    with pytest.raises(MemoryError):
        wrapper.build(_START, _END)
    assert inner.store_range()[1].startswith("2024-02-29")

    monkeypatch.setattr(inner, "extend", extend)
    wrapper.extend(_END)

    _assert_same(_stored(inner), _stored(whole))


def test_a_chunked_compute_equals_the_whole_compute(tmp_path, prices, make):
    inner = make(prices, tmp_path / "chunked.zarr")
    whole = inner.compute(_START, _END)

    chunked = ChunkedFactor(ChunkedConfig(factor=inner, granularity="month"))

    _assert_same(chunked.compute("2024-01-05", _END), whole.sel(timestamp=slice("2024-01-05", None)))


def test_compute_refuses_when_the_output_alone_does_not_fit(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    # The output is 91 bars x 5 symbols x 1 variable x 8 bytes = 3640 bytes.
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: 3000)

    with pytest.raises(MemoryError, match=r"build\(start, end\)"):
        ChunkedFactor(ChunkedConfig(factor=inner)).compute(_START, _END)


def test_the_coarsest_granularity_that_fits_is_chosen(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    peak, _ = inner.cell_bytes()
    # Room for a month and its warm-up, not for a quarter.
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: (31 + 4) * 5 * peak)

    wrapper = ChunkedFactor(ChunkedConfig(factor=inner))

    assert wrapper.plan(_START, _END)[0] == "month"


def test_a_range_that_fits_is_one_chunk(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    calls = []
    build = inner.build
    monkeypatch.setattr(inner, "build", lambda s, e: calls.append((s, e)) or build(s, e))
    monkeypatch.setattr(inner, "extend", lambda e: pytest.fail("one chunk needs no extend"))

    ChunkedFactor(ChunkedConfig(factor=inner)).build(_START, _END)

    assert calls == [(_START, _END)]


def test_too_little_memory_for_the_finest_granularity_is_refused(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: 1)

    with pytest.raises(MemoryError, match="hour"):
        ChunkedFactor(ChunkedConfig(factor=inner)).build(_START, _END)


def test_the_graph_is_compiled_once_for_every_chunk(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    compiles = []
    make = inner._make
    monkeypatch.setattr(inner, "_make", lambda: compiles.append(1) or make())

    ChunkedFactor(ChunkedConfig(factor=inner, granularity="month")).build(_START, _END)

    assert len(compiles) == 1
    assert inner._lib is None


def test_a_resampled_factor_is_refused(tmp_path, prices):
    daily = _kunquant(prices, tmp_path / "chunked.zarr").resample("1d", "last")

    with pytest.raises(ValueError, match="resample"):
        ChunkedFactor(ChunkedConfig(factor=daily))


def test_read_reads_the_wrapped_store(tmp_path, prices):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    wrapper = ChunkedFactor(ChunkedConfig(factor=inner, granularity="month")).build(_START, _END)

    xr.testing.assert_equal(wrapper.read("2024-02-01", _END), inner.read("2024-02-01", _END))
    assert wrapper.store_path == inner.store_path
    assert wrapper.store_range() == inner.store_range()
    assert wrapper.get_factor_names() == ("ma_dev",)


def test_the_wrapper_rebuilds_from_its_config(tmp_path, prices):
    wrapper = ChunkedFactor(ChunkedConfig(
        factor=_kunquant(prices, tmp_path / "chunked.zarr"), granularity="month",
    ))

    again = rebuild(json.loads(json.dumps(wrapper.get_config())))

    assert again == wrapper
    assert again.config.granularity == "month"


# --- the symbols a chunk is sized on (#251) --------------------------------


def _wide_exposures(tmp_path) -> FrameDataset:
    """Market cap and industry of the five priced symbols and ten more."""
    symbols = [*_SYMBOLS, *(f"X{i}" for i in range(10))]
    shape = (_T, len(symbols))
    return FrameDataset(xr.Dataset(
        {
            "marketcap": (("timestamp", "symbol"), np.full(shape, 1e9)),
            "industry": (("timestamp", "symbol"), np.tile(np.arange(len(symbols)) % 2, (_T, 1)).astype(float)),
        },
        coords={"timestamp": _TIMES, "symbol": symbols},
    )).to_zarr(tmp_path / "exposures.zarr")


def _neutral(tmp_path, prices) -> NeutralizedFactor:
    """The five-symbol ``RollingMean`` neutralized against fifteen symbols' exposures."""
    return NeutralizedFactor(NeutralizedConfig(
        factor=_kunquant(prices, tmp_path / "inner.zarr"), dataset=_wide_exposures(tmp_path),
        file_path=str(tmp_path / "neutral.zarr"), njobs=2,
    ))


def test_a_plain_factor_outputs_its_datasets_symbols(tmp_path, prices):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")

    assert inner.output_symbols() == _SYMBOLS
    assert ChunkedFactor(ChunkedConfig(factor=inner)).output_symbols() == _SYMBOLS


def test_a_neutralized_factor_outputs_its_inner_factors_symbols(tmp_path, prices):
    neutral = _neutral(tmp_path, prices)

    assert len(neutral.config.dataset.stored_symbols()) == 15
    assert neutral.output_symbols() == _SYMBOLS


def test_a_roster_factor_outputs_the_kept_roster(tmp_path, prices):
    roster = FrameDataset(xr.Dataset(
        {"Close": (("timestamp", "symbol"), np.ones((_T, 3)))},
        coords={"timestamp": _TIMES, "symbol": ["DDD", "BBB", "ZZZ"]},
    )).to_zarr(tmp_path / "roster.zarr")
    factor = RosterFactor(RosterConfig(factor=_kunquant(prices, tmp_path / "x.zarr"), roster=roster))

    assert factor.output_symbols() == ["BBB", "DDD"]


def test_a_neutralized_factor_is_sized_on_its_inner_factors_symbols(tmp_path, prices, monkeypatch):
    neutral = _neutral(tmp_path, prices)
    peak, _ = neutral.cell_bytes()
    # Room for the whole range on the five output symbols, not on all fifteen.
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: (_T + neutral.warmup_bars) * 5 * peak)

    wrapper = ChunkedFactor(ChunkedConfig(factor=neutral))

    assert wrapper.plan(_START, _END)[0] is None
    # The exposures' fifteen symbols, the old count, would have cut it.
    monkeypatch.setattr(wrapper, "output_symbols", neutral.config.dataset.stored_symbols)
    assert wrapper.plan(_START, _END)[0] is not None
