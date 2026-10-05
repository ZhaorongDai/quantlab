"""The Barra operators of ``quantlab/factor/kunquant_ops.py`` against float64 numpy.

Each operator is compiled into one small graph and compared with an
independent numpy reference written from its definition: the exponentially
weighted window statistics (``EWSum``, ``EWMean``, ``EWVar``, ``EWCov``,
``EWBeta``, ``EWAlpha``), the cross-sectional ``CrossSectionalWeightedMean``,
``CrossSectionalTopN`` and ``CapWeightedStandardize``, and ``SigmaClip``.

The panel has scattered NaN and infinities, an all-NaN bar and an all-NaN
symbol, and 13
symbols, padded to 16 with all-NaN columns the way ``FactorKunQuant`` pads
on macOS; the references see the 13 real symbols only, so a padded column
that leaked into a cross-section would show. The graph is compiled once in
double (what ``BarraStyle`` uses) and once in float, which the other
factors use.
"""

import numpy as np
import pytest
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, Input, Output
from KunQuant.ops import BackRef
from KunQuant.runner import KunRunner as kr
from KunQuant.Stage import Function

from quantlab.factor.kunquant import shared_executor
from quantlab.factor.kunquant_ops import (
    CapWeightedStandardize,
    CrossSectionalTopN,
    CrossSectionalWeightedMean,
    EWAlpha,
    EWBeta,
    EWCov,
    EWMean,
    EWSum,
    EWVar,
    SigmaClip,
)

_T = 60
_S = 13
_PADDED = 16
_WINDOW = 20
_HALF_LIFE = 5.0
_TOP = 5
_NAN_BAR = 30
_NAN_SYMBOL = 4
_TIE_BAR = 41


def _panel() -> dict[str, np.ndarray]:
    """``y``, ``x``, ``w`` (positive weights) and ``u`` (a 0/1 universe) on ``[T, S]``."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0005, 0.01, size=(_T, _S))
    y = 0.3 + 1.2 * x + rng.normal(0.0, 0.01, size=(_T, _S))
    x[rng.random((_T, _S)) < 0.1] = np.nan
    y[rng.random((_T, _S)) < 0.1] = np.nan
    w = rng.lognormal(20.0, 1.5, size=(_T, _S))
    w[rng.random((_T, _S)) < 0.05] = np.nan
    w[_TIE_BAR, :4] = w[_TIE_BAR, 6]  # ties at the top-N boundary
    u = (rng.random((_T, _S)) < 0.7).astype(np.float64)
    big = rng.normal(0.0, 1.0, size=(_T, _S)) * 6.0  # a z-like series with tails
    y[22, 1], y[35, 6] = np.inf, -np.inf  # infinities count as missing
    w[25, 2] = np.inf
    for values in (x, y, w, big):
        values[_NAN_BAR, :] = np.nan
        values[:, _NAN_SYMBOL] = np.nan
    return {"x": x, "y": y, "w": w, "u": u, "z": big}


def _function() -> Function:
    b = Builder()
    with b:
        x, y, w, u, z = (Input(name) for name in ("x", "y", "w", "u", "z"))
        Output(EWSum(x, _WINDOW, _HALF_LIFE), "ew_sum")
        Output(EWMean(x, _WINDOW, _HALF_LIFE), "ew_mean")
        Output(EWVar(x, _WINDOW, _HALF_LIFE), "ew_var")
        Output(EWCov(y, x, _WINDOW, _HALF_LIFE), "ew_cov")
        Output(EWBeta(y, x, _WINDOW, _HALF_LIFE), "ew_beta")
        Output(EWAlpha(y, x, _WINDOW, _HALF_LIFE), "ew_alpha")
        Output(EWMean(BackRef(x, 1), 4, 2.0), "ew_mean_short")
        Output(CrossSectionalWeightedMean(y, w * u), "wmean")
        Output(CrossSectionalTopN(w, _TOP), "top")
        Output(CapWeightedStandardize(y, w, u), "standardized")
        Output(SigmaClip(z, 10.0, 3.0), "clipped")
    return Function(b.ops)


_OUTPUTS = (
    "ew_sum", "ew_mean", "ew_var", "ew_cov", "ew_beta", "ew_alpha", "ew_mean_short",
    "wmean", "top", "standardized", "clipped",
)


@pytest.fixture(scope="module", params=["double", "float"])
def run(request) -> tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``(dtype, inputs, outputs)``: the graph run over the padded panel, cut back to 13."""
    dtype = request.param
    module = f"BarraOps_{dtype}"
    lib = cfake.compileit(
        [(module, _function(), KunCompilerConfig(
            dtype=dtype, input_layout="TS", output_layout="TS",
            options={"no_fast_stat": True},
        ))],
        f"test_barra_ops_{dtype}",
        cfake.CppCompilerConfig(),
    )
    inputs = _panel()
    numpy_dtype = np.float64 if dtype == "double" else np.float32
    padded = {
        name: np.ascontiguousarray(
            np.pad(values, ((0, 0), (0, _PADDED - _S)), constant_values=np.nan),
            dtype=numpy_dtype,
        )
        for name, values in inputs.items()
    }
    out = kr.runGraph(shared_executor(2), lib.getModule(module), padded, 0, _T)
    if dtype == "float":
        # The reference reads what the float graph read.
        inputs = {k: v.astype(np.float32).astype(np.float64) for k, v in inputs.items()}
    return dtype, inputs, {k: np.array(out[k], dtype=np.float64)[:, :_S] for k in _OUTPUTS}


# --- references -----------------------------------------------------------


def _windows(values: np.ndarray, window: int):
    """Yield ``(t, rows)`` for every bar with a full window, oldest row first."""
    for t in range(window - 1, values.shape[0]):
        yield t, values[t - window + 1 : t + 1]


def _weights(window: int, half_life: float) -> np.ndarray:
    """Weights of the window's rows, oldest first: ``0.5 ** (age / half_life)``."""
    return 0.5 ** (np.arange(window)[::-1] / half_life)


def _ew_reference(x: np.ndarray, y: np.ndarray | None, window: int, half_life: float):
    """Every EW statistic of ``x`` (and of ``y`` on ``x``) as a dict of ``[T, S]`` arrays."""
    weights = _weights(window, half_life)[:, None]
    shape = x.shape
    out = {k: np.full(shape, np.nan) for k in ("sum", "mean", "var", "cov", "beta", "alpha")}
    for t, rows in _windows(x, window):
        valid = np.isfinite(rows)
        w = np.where(valid, weights, 0.0)
        out["sum"][t] = np.where(valid, rows * weights, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (w * np.where(valid, rows, 0.0)).sum(axis=0) / w.sum(axis=0)
            out["mean"][t] = mean
            out["var"][t] = (w * np.where(valid, rows - mean, 0.0) ** 2).sum(axis=0) / w.sum(axis=0)
        if y is None:
            continue
        ys = y[t - window + 1 : t + 1]
        joint = valid & np.isfinite(ys)
        wj = np.where(joint, weights, 0.0)
        xs0, ys0 = np.where(joint, rows, 0.0), np.where(joint, ys, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            total = wj.sum(axis=0)
            mx, my = (wj * xs0).sum(axis=0) / total, (wj * ys0).sum(axis=0) / total
            dx, dy = np.where(joint, rows - mx, 0.0), np.where(joint, ys - my, 0.0)
            cov = (wj * dx * dy).sum(axis=0) / total
            var = (wj * dx * dx).sum(axis=0) / total
            out["cov"][t] = cov
            out["beta"][t] = cov / var
            out["alpha"][t] = my - cov / var * mx
    return out


def _weighted_mean_reference(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(keep, w * v, 0.0).sum(axis=1) / np.where(keep, w, 0.0).sum(axis=1)
    return np.broadcast_to(mean[:, None], v.shape)


def _top_reference(v: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(v)
    for t in range(v.shape[0]):
        valid = [i for i in range(v.shape[1]) if np.isfinite(v[t, i])]
        ranked = sorted(valid, key=lambda i: (-v[t, i], i))
        out[t, ranked[:n]] = 1.0
    return out


def _standardize_reference(v: np.ndarray, w: np.ndarray, u: np.ndarray) -> np.ndarray:
    out = np.full_like(v, np.nan)
    for t in range(v.shape[0]):
        inside = (u[t] > 0) & np.isfinite(v[t])
        weighted = inside & np.isfinite(w[t]) & (w[t] > 0)
        if inside.sum() < 2 or not weighted.any():
            continue
        mean = np.average(v[t, weighted], weights=w[t, weighted])
        std = v[t, inside].std(ddof=1)
        out[t] = np.where(np.isfinite(v[t]), (v[t] - mean) / std, np.nan)
    return out


def _clip_reference(z: np.ndarray) -> np.ndarray:
    return np.where(np.abs(z) > 10.0, np.nan, np.clip(z, -3.0, 3.0))


def _assert_matches(got: np.ndarray, want: np.ndarray, dtype: str, *, rtol: float) -> None:
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    finite = np.isfinite(want)
    np.testing.assert_allclose(
        got[finite], want[finite], rtol=rtol if dtype == "double" else max(rtol, 2e-5),
        atol=1e-12 if dtype == "double" else 1e-5,
    )


# --- tests ----------------------------------------------------------------


@pytest.mark.parametrize(
    "output, key, rtol",
    [
        ("ew_sum", "sum", 1e-12),
        ("ew_mean", "mean", 1e-12),
        ("ew_var", "var", 1e-10),
        ("ew_cov", "cov", 1e-10),
        ("ew_beta", "beta", 1e-10),
        ("ew_alpha", "alpha", 1e-10),
    ],
)
def test_ew_statistics_match_numpy(run, output, key, rtol) -> None:
    dtype, inputs, outputs = run
    want = _ew_reference(inputs["x"], inputs["y"], _WINDOW, _HALF_LIFE)[key]
    _assert_matches(outputs[output], want, dtype, rtol=rtol)
    # NaN until the window fills, then a value for every symbol with data.
    assert np.isnan(outputs[output][: _WINDOW - 1]).all()
    assert np.isfinite(outputs[output][_WINDOW - 1 :, 0]).all()


def test_ew_statistics_skip_missing_values_instead_of_going_nan(run) -> None:
    _, inputs, outputs = run
    t = _WINDOW + 5
    rows = inputs["x"][t - _WINDOW + 1 : t + 1]
    gappy = [s for s in range(_S) if s != _NAN_SYMBOL and np.isnan(rows[:, s]).any()]
    assert gappy, "the fixture should leave a NaN inside some window"
    assert np.isfinite(outputs["ew_mean"][t, gappy]).all()
    # The all-NaN symbol has nothing to weigh: its sum is 0, its mean NaN.
    assert (outputs["ew_sum"][_WINDOW:, _NAN_SYMBOL] == 0).all()
    assert np.isnan(outputs["ew_mean"][:, _NAN_SYMBOL]).all()


def test_an_ew_input_built_from_a_graph_node_keeps_its_own_fill_bars(run) -> None:
    dtype, inputs, outputs = run
    lagged = np.vstack([np.full((1, _S), np.nan), inputs["x"][:-1]])
    want = _ew_reference(lagged, None, 4, 2.0)["mean"]
    # BackRef's first bar is NaN, the window of 4 fills at bar 3 regardless.
    _assert_matches(outputs["ew_mean_short"], want, dtype, rtol=1e-12)


def test_cross_sectional_weighted_mean_matches_numpy(run) -> None:
    dtype, inputs, outputs = run
    want = _weighted_mean_reference(inputs["y"], inputs["w"] * inputs["u"])
    _assert_matches(outputs["wmean"], want, dtype, rtol=1e-12)
    assert np.isnan(outputs["wmean"][_NAN_BAR]).all()


def test_top_n_marks_the_largest_values_and_breaks_ties_by_symbol_order(run) -> None:
    _, inputs, outputs = run
    want = _top_reference(inputs["w"], _TOP)
    np.testing.assert_array_equal(outputs["top"], want)
    assert (outputs["top"][_NAN_BAR] == 0).all()
    assert (outputs["top"][:, _NAN_SYMBOL] == 0).all()
    assert (outputs["top"].sum(axis=1)[np.arange(_T) != _NAN_BAR] == _TOP).all()


def test_top_n_compiles_one_class_per_n() -> None:
    with Builder():
        v = Input("v")
        a, b, c = CrossSectionalTopN(v, 3), CrossSectionalTopN(v, 3), CrossSectionalTopN(v, 4)
    assert type(a) is type(b) is not type(c)
    assert isinstance(c, CrossSectionalTopN) and type(c).__name__ == "CrossSectionalTopN_4"
    with pytest.raises(ValueError, match="n >= 1"):
        CrossSectionalTopN(v, 0)


def test_cap_weighted_standardize_matches_numpy_and_reaches_outside_the_universe(run) -> None:
    dtype, inputs, outputs = run
    got = outputs["standardized"]
    want = _standardize_reference(inputs["y"], inputs["w"], inputs["u"])
    _assert_matches(got, want, dtype, rtol=1e-10)
    outside = (inputs["u"] == 0) & np.isfinite(want)
    assert outside.any() and np.isfinite(got[outside]).all()


def test_cap_weighted_standardize_gives_zero_weighted_mean_and_unit_std_in_the_universe(run) -> None:
    dtype, inputs, outputs = run
    got, w, u = outputs["standardized"], inputs["w"], inputs["u"]
    tol = 1e-10 if dtype == "double" else 1e-5
    for t in range(_T):
        inside = (u[t] > 0) & np.isfinite(got[t])
        weighted = inside & np.isfinite(w[t]) & (w[t] > 0)
        if inside.sum() < 2:
            continue
        assert abs(np.average(got[t, weighted], weights=w[t, weighted])) < tol
        assert abs(got[t, inside].std(ddof=1) - 1.0) < tol


def test_sigma_clip_drops_data_errors_and_clips_at_three(run) -> None:
    dtype, inputs, outputs = run
    _assert_matches(outputs["clipped"], _clip_reference(inputs["z"]), dtype, rtol=1e-12)
    z = inputs["z"]
    assert ((np.abs(z) > 10) & np.isfinite(z)).any(), "the fixture should hold a data error"
    assert ((np.abs(z) > 3) & (np.abs(z) <= 10)).any(), "and a value to clip"


@pytest.mark.parametrize("window, half_life", [(0, 5.0), (10, 0.0), (10, -1.0)])
def test_ew_operators_refuse_an_unusable_window(window, half_life) -> None:
    with Builder():
        with pytest.raises(ValueError, match="EW window"):
            EWMean(Input("x"), window, half_life)


def test_sigma_clip_refuses_a_clip_beyond_the_data_error() -> None:
    with Builder():
        with pytest.raises(ValueError, match="clip <= data_error"):
            SigmaClip(Input("z"), 2.0, 3.0)
