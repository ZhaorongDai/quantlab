"""Cross-sectional Z-score KunQuant operator tests (quick task 260915-ocw).

Locks `quantlab/my_ops/preprocess.py:CrossSectionalZScore`, a
`GenericCrossSectionalOp` that normalizes each time point across symbols:
`(x - nanmean) / nanstd(ddof=1)`. The reference is pandas
`df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=1), axis=0)` computed
in float64; outputs must match within 2e-4 with an identical NaN pattern.

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

import numpy as np
import pandas as pd
import pytest
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.ops import WindowedAvg
from KunQuant.runner import KunRunner as kr

from quantlab.my_ops.preprocess import CrossSectionalZScore

_N_TIMES = 40
_N_SYMBOLS = 16
_ALL_NAN_ROW = 7
_SINGLE_VALID_ROW = 9
_CONSTANT_ROW = 11
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


def test_batch_start0_matches_pandas_for_graph_input(batch_outputs):
    """With the op applied directly to a graph Input, batch start=0 output
    equals the pandas cross-sectional z-score (ddof=1)."""
    close, got = batch_outputs
    _assert_matches(got["z_raw"], _pandas_reference(close)["z_raw"])
