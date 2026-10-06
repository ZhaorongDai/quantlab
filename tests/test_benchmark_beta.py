"""``BenchmarkBeta``: each symbol's rolling OLS beta of one-bar returns on a single-symbol benchmark.

The reference is the slope of ``numpy.linalg.lstsq`` on an intercept and the
benchmark's returns over each window, written from the definition and
independent of the factor's running sums. Prices are in-memory panels.
"""

import numpy as np
import pytest
import xarray as xr

from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import BenchmarkBetaConfig
from quantlab.factor.predefined.benchmark_beta import BenchmarkBeta

_T = 40
_TIMES = np.datetime64("2024-01-01") + np.arange(_T).astype("timedelta64[D]")
LOOKBACK, MIN_BARS = 10, 6


def _day(i: int) -> str:
    return str(np.datetime_as_string(_TIMES[i], unit="D"))


def _panel(closes: dict[str, np.ndarray]) -> FrameDataset:
    symbols = list(closes)
    values = np.column_stack([closes[s] for s in symbols])
    return FrameDataset(xr.Dataset(
        {"adjClose": (("timestamp", "symbol"), values)},
        coords={"timestamp": _TIMES, "symbol": symbols},
    ))


def _closes(returns: np.ndarray) -> np.ndarray:
    return 100.0 * np.cumprod(1.0 + np.nan_to_num(returns))


rng = np.random.default_rng(208)
BENCH_RET = rng.normal(0.0, 0.01, _T)
BENCH_RET[0] = 0.0
NOISY_RET = 0.7 * BENCH_RET + rng.normal(0.0, 0.01, _T)
NOISY_RET[0] = 0.0


def _factor(stocks: FrameDataset, benchmark: FrameDataset, **kwargs) -> BenchmarkBeta:
    return BenchmarkBeta(BenchmarkBetaConfig(
        warmup_bars=LOOKBACK, dataset=stocks, benchmark=benchmark,
        lookback_bars=LOOKBACK, min_bars=MIN_BARS, **kwargs,
    ))


def _reference(stock_ret: np.ndarray, bench_ret: np.ndarray, t: int) -> float:
    """lstsq slope over the window of returns ending at bar t (bar 0 has no return)."""
    window = np.arange(max(1, t - LOOKBACK + 1), t + 1)
    y, x = stock_ret[window], bench_ret[window]
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < MIN_BARS:
        return np.nan
    design = np.column_stack([np.ones(ok.sum()), x[ok]])
    return float(np.linalg.lstsq(design, y[ok], rcond=None)[0][1])


def test_a_symbol_moving_twice_the_benchmark_has_a_beta_of_two_and_none_before_min_bars():
    benchmark = _panel({"VT": _closes(BENCH_RET)})
    stocks = _panel({"AAA": _closes(2.0 * BENCH_RET)})
    beta = _factor(stocks, benchmark).compute(_day(0), _day(_T - 1))["beta"].sel(symbol="AAA").values
    assert np.isnan(beta[:MIN_BARS]).all()
    np.testing.assert_allclose(beta[MIN_BARS:], 2.0, rtol=1e-9)


def test_the_beta_is_the_ols_slope_over_the_trailing_window():
    benchmark = _panel({"VT": _closes(BENCH_RET)})
    stocks = _panel({"AAA": _closes(NOISY_RET)})
    beta = _factor(stocks, benchmark).compute(_day(0), _day(_T - 1))["beta"].sel(symbol="AAA").values
    expected = [_reference(NOISY_RET, BENCH_RET, t) for t in range(_T)]
    np.testing.assert_allclose(beta, expected, rtol=1e-9, equal_nan=True)


def test_a_bar_without_a_price_drops_out_of_the_window_and_later_bars_never_count():
    """A missing stock price removes that bar's return and the next one's; a
    change after bar t leaves the beta at t unchanged."""
    closes = _closes(NOISY_RET)
    closes[20] = np.nan
    ret = closes[1:] / closes[:-1] - 1.0
    stock_ret = np.concatenate([[np.nan], ret])
    benchmark = _panel({"VT": _closes(BENCH_RET)})
    beta = _factor(_panel({"AAA": closes}), benchmark).compute(_day(0), _day(_T - 1))
    beta = beta["beta"].sel(symbol="AAA").values
    expected = [_reference(stock_ret, BENCH_RET, t) for t in range(_T)]
    np.testing.assert_allclose(beta, expected, rtol=1e-9, equal_nan=True)

    later = closes.copy()
    later[30:] *= 1.5
    moved = _factor(_panel({"AAA": later}), benchmark).compute(_day(0), _day(_T - 1))
    np.testing.assert_array_equal(moved["beta"].sel(symbol="AAA").values[:30], beta[:30])


def test_warm_up_is_read_before_start():
    benchmark = _panel({"VT": _closes(BENCH_RET)})
    stocks = _panel({"AAA": _closes(NOISY_RET)})
    whole = _factor(stocks, benchmark).compute(_day(0), _day(_T - 1))
    late = _factor(stocks, benchmark).compute(_day(25), _day(_T - 1))
    xr.testing.assert_allclose(late, whole.sel(timestamp=late.timestamp))


def test_a_benchmark_with_more_than_one_symbol_and_bad_windows_are_refused():
    stocks = _panel({"AAA": _closes(NOISY_RET)})
    two = _panel({"VT": _closes(BENCH_RET), "ACWI": _closes(BENCH_RET)})
    with pytest.raises(ValueError, match="exactly one symbol"):
        _factor(stocks, two).compute(_day(0), _day(_T - 1))
    one = _panel({"VT": _closes(BENCH_RET)})
    with pytest.raises(ValueError, match="min_bars"):
        BenchmarkBeta(BenchmarkBetaConfig(
            warmup_bars=LOOKBACK, dataset=stocks, benchmark=one, lookback_bars=5, min_bars=6,
        ))
    with pytest.raises(ValueError, match="warmup_bars"):
        BenchmarkBeta(BenchmarkBetaConfig(
            warmup_bars=5, dataset=stocks, benchmark=one, lookback_bars=LOOKBACK, min_bars=MIN_BARS,
        ))


def test_the_warm_up_defaults_to_the_default_window():
    one = _panel({"VT": _closes(BENCH_RET)})
    beta = BenchmarkBeta(BenchmarkBetaConfig(dataset=_panel({"AAA": _closes(NOISY_RET)}), benchmark=one))
    assert beta.warmup_bars == beta.config.lookback_bars == 252


def test_the_factor_rebuilds_from_its_config(tmp_path):
    stocks = _panel({"AAA": _closes(NOISY_RET)}).to_zarr(tmp_path / "stocks.zarr")
    benchmark = _panel({"VT": _closes(BENCH_RET)}).to_zarr(tmp_path / "vt.zarr")
    factor = _factor(stocks, benchmark)
    again = rebuild(factor.get_config())
    xr.testing.assert_identical(again.compute(_day(0), _day(_T - 1)), factor.compute(_day(0), _day(_T - 1)))
