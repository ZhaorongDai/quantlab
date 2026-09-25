"""Cross-sectional winsorize (缩尾) and trim (截尾) KunQuant operator tests.

Locks `quantlab/my_ops/preprocess.py:CrossSectionalWinsorize` and
`CrossSectionalTrim`. At every timestamp the `lower`/`upper` quantiles are
taken over the non-NaN symbols with linear interpolation; winsorize clips to
those bounds and trim sets values strictly outside them to NaN. The reference
is `np.nanquantile` in float64 on the same float32 panel; outputs must match
within 2e-4 with an identical NaN pattern, in both TS batch and STREAM layouts.

The graph deliberately holds two parameter sets of the same op class. KunQuant
names generated C++ functions by class name plus layout, so without the
per-`(lower, upper)` subclass both outputs would silently share one function
and one of them would be wrong.

Cost control mirrors `test_cross_sectional_zscore.py`: one TS module and one
STREAM module, compiled once each in module-scoped fixtures. The panel has 16
symbols because KunQuant needs the symbol count to align with its SIMD block
width, and only `start=0` batch runs are exercised.
"""

import inspect
import warnings

import numpy as np
import pytest
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, CompositiveOp, Input, Output
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.ops import WindowedAvg
from KunQuant.ops.MiscOp import GenericCrossSectionalOp
from KunQuant.runner import KunRunner as kr

from quantlab.my_ops.preprocess import (
    CrossSectionalTrim,
    CrossSectionalWinsorize,
    CrossSectionalZScore,
)

_N_TIMES = 40
_N_SYMBOLS = 16
_ALL_NAN_ROW = 7
_SINGLE_VALID_ROW = 9
_CONSTANT_ROW = 11

# name -> (kind, lower, upper, input transform)
_CASES = {
    "win_default": ("winsorize", 0.01, 0.99, "raw"),
    "win_10_90": ("winsorize", 0.10, 0.90, "raw"),
    "trim_default": ("trim", 0.01, 0.99, "raw"),
    "trim_10_90": ("trim", 0.10, 0.90, "raw"),
    "win_10_90_ma5": ("winsorize", 0.10, 0.90, "ma5"),
}
_OUTPUT_NAMES = (*_CASES, "z_of_win")


def _panel() -> np.ndarray:
    """Seeded float32 panel with fat tails, ~10% NaN and three degenerate rows."""
    rng = np.random.default_rng(1)
    close = (100 + rng.standard_t(2, (_N_TIMES, _N_SYMBOLS))).astype(np.float32)
    close[rng.random((_N_TIMES, _N_SYMBOLS)) < 0.1] = np.nan
    close[_ALL_NAN_ROW, :] = np.nan
    close[_SINGLE_VALID_ROW, 1:] = np.nan
    close[_CONSTANT_ROW, :] = 5.0
    return close


def _op(kind: str):
    return CrossSectionalWinsorize if kind == "winsorize" else CrossSectionalTrim


def _build_function() -> Function:
    """Every case as one output, plus winsorize feeding CrossSectionalZScore."""
    b = Builder()
    with b:
        close = Input("close")
        ma5 = WindowedAvg(close, 5)
        for name, (kind, lower, upper, src) in _CASES.items():
            x = close if src == "raw" else ma5
            Output(_op(kind)(x, lower, upper), name)
        Output(CrossSectionalZScore(CrossSectionalWinsorize(close, 0.1, 0.9)), "z_of_win")
    return Function(b.ops)


def _bound_rows(x: np.ndarray, kind: str, lower: float, upper: float) -> np.ndarray:
    """Numpy reference: per-row nanquantile bounds, then clip or trim."""
    out = np.full_like(x, np.nan)
    for t, row in enumerate(x):
        if np.isnan(row).all():
            continue
        lo, hi = np.nanquantile(row, [lower, upper])
        if kind == "winsorize":
            out[t] = np.clip(row, lo, hi)
        else:
            out[t] = np.where((row < lo) | (row > hi), np.nan, row)
    return out


def _reference(close: np.ndarray) -> dict[str, np.ndarray]:
    x = close.astype(np.float64)
    ma5 = np.full_like(x, np.nan)
    for t in range(4, _N_TIMES):
        ma5[t] = x[t - 4 : t + 1].mean(axis=0)  # NaN in the window -> NaN, as KunQuant
    ref = {}
    for name, (kind, lower, upper, src) in _CASES.items():
        ref[name] = _bound_rows(x if src == "raw" else ma5, kind, lower, upper)
    win = _bound_rows(x, "winsorize", 0.1, 0.9)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(win, axis=1, ddof=1, keepdims=True)
        z = (win - np.nanmean(win, axis=1, keepdims=True)) / sd
    z[~(sd[:, 0] > 0)] = np.nan
    ref["z_of_win"] = z
    return ref


def _assert_matches(got: np.ndarray, want: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    finite = ~np.isnan(want)
    np.testing.assert_allclose(got[finite], want[finite], atol=2e-4, rtol=0)


@pytest.fixture(scope="module")
def batch_outputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    close = _panel()
    lib = cfake.compileit(
        [
            (
                "CrossSectionalBoundsBatch",
                _build_function(),
                KunCompilerConfig(input_layout="TS", output_layout="TS"),
            )
        ],
        "test_cs_bounds_batch",
        cfake.CppCompilerConfig(),
    )
    modu = lib.getModule("CrossSectionalBoundsBatch")
    executor = kr.createMultiThreadExecutor(4)
    out = kr.runGraph(
        executor, modu, {"close": np.ascontiguousarray(close)}, 0, _N_TIMES
    )
    return close, {name: np.array(out[name], copy=True) for name in _OUTPUT_NAMES}


@pytest.fixture(scope="module")
def stream_outputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    close = _panel()
    lib = cfake.compileit(
        [
            (
                "CrossSectionalBoundsStream",
                _build_function(),
                KunCompilerConfig(input_layout="STREAM", output_layout="STREAM"),
            )
        ],
        "test_cs_bounds_stream",
        cfake.CppCompilerConfig(),
    )
    modu = lib.getModule("CrossSectionalBoundsStream")
    executor = kr.createMultiThreadExecutor(4)
    ctx = kr.StreamContext(executor, modu, _N_SYMBOLS)
    h_close = ctx.queryBufferHandle("close")
    handles = {name: ctx.queryBufferHandle(name) for name in _OUTPUT_NAMES}
    got = {
        name: np.empty((_N_TIMES, _N_SYMBOLS), dtype=np.float32)
        for name in _OUTPUT_NAMES
    }
    for t in range(_N_TIMES):
        ctx.pushData(h_close, np.ascontiguousarray(close[t]))
        ctx.run()
        for name, handle in handles.items():
            got[name][t] = ctx.getCurrentBuffer(handle)[:_N_SYMBOLS]
    del ctx, executor
    return close, got


@pytest.mark.parametrize("name", _OUTPUT_NAMES, ids=list(_OUTPUT_NAMES))
def test_batch_matches_numpy(batch_outputs, name):
    """TS batch output equals the numpy nanquantile reference for every case,
    including two parameter sets of one class in the same graph, an
    intermediate-node input, and winsorize feeding CrossSectionalZScore."""
    close, got = batch_outputs
    _assert_matches(got[name], _reference(close)[name])


@pytest.mark.parametrize("name", _OUTPUT_NAMES, ids=list(_OUTPUT_NAMES))
def test_stream_matches_numpy(stream_outputs, name):
    """STREAM layout driven bar by bar gives the same values and NaN pattern."""
    close, got = stream_outputs
    _assert_matches(got[name], _reference(close)[name])


def test_ops_actually_change_the_panel(batch_outputs):
    """Guard against a vacuous pass: 10/90 winsorize changes some values and
    trim introduces NaN where the input was valid."""
    close, got = batch_outputs
    valid = ~np.isnan(close)
    assert (got["win_10_90"][valid] != close[valid]).any()
    assert np.isnan(got["trim_10_90"][valid]).any()
    assert not np.isnan(got["win_10_90"][valid]).any()


def test_degenerate_rows(batch_outputs):
    """All-NaN stays all NaN; a single valid value and a constant row pass
    through unchanged for both ops."""
    close, got = batch_outputs
    for name in ("win_10_90", "trim_10_90"):
        out = got[name]
        assert np.isnan(out[_ALL_NAN_ROW]).all()
        np.testing.assert_array_equal(out[_SINGLE_VALID_ROW], close[_SINGLE_VALID_ROW])
        np.testing.assert_array_equal(out[_CONSTANT_ROW], close[_CONSTANT_ROW])


def test_one_subclass_per_quantile_pair():
    """Each (lower, upper) pair gets its own cached subclass whose name carries
    the quantiles, and instances still satisfy isinstance on the public class."""
    with Builder():
        x = Input("close")
        a = CrossSectionalWinsorize(x)
        b = CrossSectionalWinsorize(x, 0.01, 0.99)
        c = CrossSectionalWinsorize(x, 0.05, 0.95)
        d = CrossSectionalTrim(x)
    assert type(a) is type(b)
    assert type(a) is not type(c)
    assert type(a).__name__ == "CrossSectionalWinsorize_0p01_0p99"
    assert type(d).__name__ == "CrossSectionalTrim_0p01_0p99"
    assert isinstance(c, CrossSectionalWinsorize)
    assert not isinstance(d, CrossSectionalWinsorize)
    assert dict(c.attrs) == {"lower": 0.05, "upper": 0.95}


@pytest.mark.parametrize(
    "lower, upper", [(-0.1, 0.9), (0.1, 1.1), (0.9, 0.1), (0.5, 0.5)]
)
@pytest.mark.parametrize("cls", [CrossSectionalWinsorize, CrossSectionalTrim])
def test_invalid_quantiles_raise(cls, lower, upper):
    with Builder():
        with pytest.raises(ValueError, match="lower < upper"):
            cls(Input("close"), lower, upper)


@pytest.mark.parametrize("cls", [CrossSectionalWinsorize, CrossSectionalTrim])
def test_is_a_generic_cross_sectional_op_whose_body_ignores_attrs(cls):
    """Structural lock, as for CrossSectionalZScore: the C++ body must not read
    `self.attrs`, because KunQuant dedups generated functions by class name."""
    assert issubclass(cls, GenericCrossSectionalOp)
    assert not issubclass(cls, CompositiveOp)
    assert "attrs" not in inspect.getsource(cls.generate_body)
