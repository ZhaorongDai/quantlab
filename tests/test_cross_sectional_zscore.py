"""Cross-sectional Z-score KunQuant operator tests (quick task 260915-ocw).

Locks `quantlab/my_ops/preprocess.py:CrossSectionalZScore`, a
`GenericCrossSectionalOp` that normalizes each time point across symbols:
`(x - nanmean) / nanstd(ddof=1)`. The reference is pandas
`df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=1), axis=0)` computed
in float64; outputs must match within 2e-4 with an identical NaN pattern, in
both TS batch and STREAM layouts, whether the op consumes a graph Input or an
intermediate node, and when its output feeds a downstream time-series op.

Cost control: compiling a KunQuant module takes several seconds, so the whole
file compiles exactly two modules (one TS batch, one STREAM), each holding all
three outputs, in module-scoped fixtures.

The panel has 16 symbols because KunQuant requires the symbol count to align
with its SIMD block width (on this aarch64 machine 16 works and 13 does not).

Only `runGraph(..., start=0, ...)` is exercised. KunQuant 0.1.11's
`CrossSectionalDataHolder` computes `base_time` from `num_time` before
`num_time` is assigned, so start>0 gives wrong and non-deterministic output for
every `GenericCrossSectionalOp`. A non-deterministic failure cannot be asserted
reliably, so the bug is documented in the class docstring instead of tested.
`FactorKunQuant.cal`, the only batch caller in the repo, always passes start=0.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, CompositiveOp, CrossSectionalOp, Input, Output
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.ops import WindowedAvg
from KunQuant.ops.MiscOp import GenericCrossSectionalOp
from KunQuant.runner import KunRunner as kr

from quantlab.my_ops.preprocess import CrossSectionalZScore

_N_TIMES = 40
_N_SYMBOLS = 16
_ALL_NAN_ROW = 7
_SINGLE_VALID_ROW = 9
_CONSTANT_ROW = 11
_DEGENERATE_ROWS = (_ALL_NAN_ROW, _SINGLE_VALID_ROW, _CONSTANT_ROW)
_OUTPUT_NAMES = ("z_raw", "z_ma5", "ma3_of_z")


def _panel() -> np.ndarray:
    """Seeded float32 random-walk panel with ~10% NaN and three degenerate rows."""
    rng = np.random.default_rng(0)
    close = (100 + rng.standard_normal((_N_TIMES, _N_SYMBOLS)).cumsum(axis=0)).astype(
        np.float32
    )
    close[rng.random((_N_TIMES, _N_SYMBOLS)) < 0.1] = np.nan
    close[_ALL_NAN_ROW, :] = np.nan
    close[_SINGLE_VALID_ROW, 1:] = np.nan
    close[_CONSTANT_ROW, :] = 5.0
    return close


def _build_function() -> Function:
    """Graph with the op on an Input, on an intermediate node, and feeding a TS op."""
    b = Builder()
    with b:
        close = Input("close")
        Output(CrossSectionalZScore(close), "z_raw")
        Output(CrossSectionalZScore(WindowedAvg(close, 5)), "z_ma5")
        Output(WindowedAvg(CrossSectionalZScore(close), 3), "ma3_of_z")
    return Function(b.ops)


def _pandas_reference(close: np.ndarray) -> dict[str, np.ndarray]:
    """The same three outputs computed with pandas in float64 (ddof=1)."""
    df = pd.DataFrame(close.astype(np.float64))

    def cs(d: pd.DataFrame) -> pd.DataFrame:
        return d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1, ddof=1), axis=0)

    return {
        "z_raw": cs(df).values,
        "z_ma5": cs(df.rolling(5).mean()).values,
        "ma3_of_z": cs(df).rolling(3).mean().values,
    }


def _assert_matches(got: np.ndarray, want: np.ndarray) -> None:
    """Identical NaN mask, and finite cells equal within atol 2e-4."""
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    finite = ~np.isnan(want)
    np.testing.assert_allclose(got[finite], want[finite], atol=2e-4, rtol=0)


@pytest.fixture(scope="module")
def batch_outputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compile one TS module with all three outputs and run it with start=0."""
    close = _panel()
    lib = cfake.compileit(
        [
            (
                "CrossSectionalZScoreBatch",
                _build_function(),
                KunCompilerConfig(input_layout="TS", output_layout="TS"),
            )
        ],
        "test_cs_zscore_batch",
        cfake.CppCompilerConfig(),
    )
    modu = lib.getModule("CrossSectionalZScoreBatch")
    executor = kr.createMultiThreadExecutor(4)
    out = kr.runGraph(
        executor, modu, {"close": np.ascontiguousarray(close)}, 0, _N_TIMES
    )
    return close, {name: np.array(out[name], copy=True) for name in _OUTPUT_NAMES}


@pytest.fixture(scope="module")
def stream_outputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compile one STREAM module with all three outputs and drive it bar by bar.

    `getCurrentBuffer` returns a buffer that the next `run()` overwrites, so
    each bar is copied into a preallocated (T, S) array. The executor is kept
    referenced for as long as the stream context lives.
    """
    close = _panel()
    lib = cfake.compileit(
        [
            (
                "CrossSectionalZScoreStream",
                _build_function(),
                KunCompilerConfig(input_layout="STREAM", output_layout="STREAM"),
            )
        ],
        "test_cs_zscore_stream",
        cfake.CppCompilerConfig(),
    )
    modu = lib.getModule("CrossSectionalZScoreStream")
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


def test_batch_start0_matches_pandas_for_graph_input(batch_outputs):
    """With the op applied directly to a graph Input, batch start=0 output
    equals the pandas cross-sectional z-score (ddof=1)."""
    close, got = batch_outputs
    _assert_matches(got["z_raw"], _pandas_reference(close)["z_raw"])


def test_batch_start0_matches_pandas_for_intermediate_node_input(batch_outputs):
    """With the op applied to an intermediate node (`WindowedAvg(close, 5)`),
    batch output equals pandas, including the rolling warm-up NaN rows."""
    close, got = batch_outputs
    _assert_matches(got["z_ma5"], _pandas_reference(close)["z_ma5"])


def test_batch_output_feeds_time_series_op(batch_outputs):
    """The cross-sectional output composes with a downstream time-series op:
    `WindowedAvg(CrossSectionalZScore(close), 3)` equals pandas."""
    close, got = batch_outputs
    _assert_matches(got["ma3_of_z"], _pandas_reference(close)["ma3_of_z"])


@pytest.mark.parametrize("name", _OUTPUT_NAMES, ids=list(_OUTPUT_NAMES))
def test_stream_matches_pandas(stream_outputs, name):
    """STREAM layout driven bar by bar through `StreamContext` produces the
    same values and NaN pattern as pandas for all three graph shapes."""
    close, got = stream_outputs
    _assert_matches(got[name], _pandas_reference(close)[name])


def test_degenerate_cross_sections_are_all_nan(batch_outputs):
    """An all-NaN row, a single-valid-value row and a constant row each give a
    fully NaN output row; on every other row NaN inputs map to NaN outputs in
    the same positions and nowhere else."""
    close, got = batch_outputs
    z = got["z_raw"]
    normal_rows = [t for t in range(_N_TIMES) if t not in _DEGENERATE_ROWS]

    # Guard against a vacuous pass: some non-degenerate row must hold a NaN.
    assert np.isnan(close[normal_rows]).any()

    for row in _DEGENERATE_ROWS:
        assert np.isnan(z[row]).all(), f"row {row} should be entirely NaN"
    np.testing.assert_array_equal(np.isnan(z[normal_rows]), np.isnan(close[normal_rows]))


def test_valid_rows_have_zero_mean_unit_sample_std(batch_outputs):
    """Independent of pandas: every non-degenerate output row has nanmean
    about 0 and sample std (ddof=1) about 1, within 1e-4."""
    _, got = batch_outputs
    z = got["z_raw"].astype(np.float64)
    for t in range(_N_TIMES):
        if t in _DEGENERATE_ROWS:
            continue
        assert abs(np.nanmean(z[t])) < 1e-4, f"row {t} mean {np.nanmean(z[t])}"
        sd = np.nanstd(z[t], ddof=1)
        assert abs(sd - 1.0) < 1e-4, f"row {t} sample std {sd}"


def test_is_a_generic_cross_sectional_op_without_attrs():
    """Structural lock: the op is a `GenericCrossSectionalOp` (hence a
    `CrossSectionalOp`), not a `CompositiveOp`, and its C++ body never reads
    `self.attrs`. KunQuant's CodegenCpp dedups generated functions by class
    name plus layout only, so an attrs-dependent body would silently share
    code across parameter sets."""
    assert issubclass(CrossSectionalZScore, GenericCrossSectionalOp)
    assert issubclass(CrossSectionalZScore, CrossSectionalOp)
    assert not issubclass(CrossSectionalZScore, CompositiveOp)
    assert "attrs" not in inspect.getsource(CrossSectionalZScore.generate_body)
