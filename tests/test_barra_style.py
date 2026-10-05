"""``BarraStyle`` (Size and Beta) against a float64 numpy reference.

The input is a merge of two stores: adjusted closes with a risk-free rate,
and market caps. The reference follows USE4's definitions directly: the
estimation universe is the top N by the previous bar's cap, the market is
its cap-weighted return, BETA is a weighted least-squares slope solved with
``numpy.linalg.lstsq`` (not the closed form the operators use), and each
descriptor and style is standardized with a cap-weighted mean and an
equally weighted standard deviation. 21 symbols make macOS pad the axis.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import compute_all

from quantlab.core.component import rebuild
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters

_T = 100
_S = 21
_WINDOW = 40
_HALF_LIFE = 10.0
_MIN_OBS = 20
_ESTU = 12
_LISTED_LATE = 3  # first price and cap at bar 55
_LISTING_BAR = 55
_CAP_GAP = 7  # cap missing on a few bars
_TINY = 9  # a micro cap far below the universe: a data error
_CLIPPED = 2  # a small cap 4-6 standard deviations below: clipped
_KWARGS = {
    "beta_window": _WINDOW,
    "beta_half_life": _HALF_LIFE,
    "beta_min_observations": _MIN_OBS,
    "estimation_universe_size": _ESTU,
}


def _inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    market = rng.normal(0.0004, 0.012, size=_T)
    risk_free = np.abs(rng.normal(0.0001, 0.00003, size=_T))
    betas = rng.normal(1.0, 0.4, size=_S)
    returns = risk_free[:, None] + market[:, None] * betas + rng.normal(0, 0.015, (_T, _S))
    price = 20.0 * np.cumprod(1.0 + returns, axis=0)
    shares = rng.lognormal(18.0, 1.0, size=_S)
    shares[_TINY] = 1.0e3
    cap = price * shares
    for values in (price, cap):
        values[:_LISTING_BAR, _LISTED_LATE] = np.nan
    cap[[30, 31, 64], _CAP_GAP] = np.nan
    price[47, 2] = np.nan  # a missing bar inside the window
    return {
        "adjClose": price,
        "marketcap": cap,
        "risk_free": np.broadcast_to(risk_free[:, None], (_T, _S)).copy(),
    }


def _store(tmp_path: Path, name: str, variables: dict[str, np.ndarray]) -> StockDataset:
    timestamps = pd.bdate_range("2021-01-04", periods=_T)
    panel = xr.Dataset(
        {k: (("timestamp", "symbol"), v) for k, v in variables.items()},
        coords={"timestamp": timestamps, "symbol": [f"S{i:02d}" for i in range(_S)]},
    )
    store = tmp_path / f"{name}.zarr"
    panel.to_zarr(store, mode="w")
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(tmp_path / "raw"),
        market="us_equity",
        frequency="1d",
        start_date=str(timestamps[0].date()),
        end_date=str(timestamps[-1].date()),
    ))


def _config(tmp_path: Path, **overrides) -> FactorConfig:
    inputs = _inputs()
    prices = _store(tmp_path, "prices", {k: inputs[k] for k in ("adjClose", "risk_free")})
    caps = _store(tmp_path, "caps", {"marketcap": inputs["marketcap"]})
    values = {
        "warmup_bars": 0,
        "dataset": [prices, caps],
        "mode": "batch",
        "data_columns": ("adjClose", "marketcap", "risk_free"),
        "file_path": str(tmp_path / "barra.zarr"),
        "njobs": 2,
        "kwargs": dict(_KWARGS),
    }
    values.update(overrides)
    return FactorConfig(**values)


# --- reference ------------------------------------------------------------


def _lag(values: np.ndarray) -> np.ndarray:
    return np.vstack([np.full((1, values.shape[1]), np.nan), values[:-1]])


def _estimation_universe(cap_before: np.ndarray) -> np.ndarray:
    estu = np.zeros_like(cap_before, dtype=bool)
    for t in range(_T):
        valid = [i for i in range(_S) if np.isfinite(cap_before[t, i])]
        estu[t, sorted(valid, key=lambda i: (-cap_before[t, i], i))[:_ESTU]] = True
    return estu


def _wls_slope(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> float:
    root = np.sqrt(weights)
    design = np.column_stack([np.ones_like(x), x]) * root[:, None]
    coefficients, *_ = np.linalg.lstsq(design, y * root, rcond=None)
    return coefficients[1]


def _standardize(values: np.ndarray, cap_before: np.ndarray, estu: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan)
    for t in range(_T):
        inside = estu[t] & np.isfinite(values[t])
        if inside.sum() < 2:
            continue
        mean = np.average(values[t, inside], weights=cap_before[t, inside])
        out[t] = (values[t] - mean) / values[t, inside].std(ddof=1)
    return out


def _clip(z: np.ndarray) -> np.ndarray:
    return np.where(np.abs(z) > 10.0, np.nan, np.clip(z, -3.0, 3.0))


def _reference() -> dict[str, np.ndarray]:
    inputs = _inputs()
    price, cap = inputs["adjClose"], inputs["marketcap"]
    stock_return = price / _lag(price) - 1.0
    risk_free = _lag(inputs["risk_free"])
    cap_before = _lag(cap)
    estu = _estimation_universe(cap_before)
    market = np.full(_T, np.nan)
    for t in range(_T):
        use = estu[t] & np.isfinite(stock_return[t])
        if use.any():
            market[t] = np.average(stock_return[t, use], weights=cap_before[t, use])
    excess = stock_return - risk_free
    market_excess = market[:, None] - risk_free

    weights = 0.5 ** (np.arange(_WINDOW)[::-1] / _HALF_LIFE)
    beta = np.full((_T, _S), np.nan)
    for t in range(_WINDOW - 1, _T):
        rows = slice(t - _WINDOW + 1, t + 1)
        for s in range(_S):
            y, x = excess[rows, s], market_excess[rows, s]
            ok = np.isfinite(y) & np.isfinite(x)
            if ok.sum() >= _MIN_OBS:
                beta[t, s] = _wls_slope(y[ok], x[ok], weights[ok])

    desc_lncap = _clip(_standardize(np.log(cap), cap_before, estu))
    desc_beta = _clip(_standardize(beta, cap_before, estu))
    return {
        "desc_lncap": desc_lncap,
        "desc_beta": desc_beta,
        "style_size": _standardize(desc_lncap, cap_before, estu),
        "style_beta": _standardize(desc_beta, cap_before, estu),
        "estu": estu.astype(np.float64),
        "cap_before": cap_before,
    }


@pytest.fixture(scope="module")
def computed(tmp_path_factory) -> tuple[BarraStyle, xr.Dataset, dict[str, np.ndarray]]:
    tmp_path = tmp_path_factory.mktemp("barra")
    factor = BarraStyle(_config(tmp_path))
    return factor, compute_all(factor), _reference()


# --- tests ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["desc_lncap", "desc_beta", "style_size", "style_beta", "estu"]
)
def test_outputs_match_the_numpy_reference(computed, name) -> None:
    _, out, want = computed
    got = out[name].transpose("timestamp", "symbol").to_numpy()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want[name]))
    finite = np.isfinite(want[name])
    np.testing.assert_allclose(got[finite], want[name][finite], rtol=1e-9, atol=1e-9)


def test_the_fixture_exercises_clipping_late_listing_and_the_minimum_count(computed) -> None:
    _, _, want = computed
    assert (want["desc_lncap"][1:, _CLIPPED] == -3.0).all()
    # BETA waits for _MIN_OBS returns after listing, then exists.
    first = _LISTING_BAR + 1 + _MIN_OBS - 1
    assert np.isnan(want["desc_beta"][:first, _LISTED_LATE]).all()
    assert np.isfinite(want["desc_beta"][first:, _LISTED_LATE]).all()
    assert np.isfinite(want["desc_beta"][_WINDOW - 1 :, 0]).all()


@pytest.mark.parametrize("name", ["style_size", "style_beta"])
def test_styles_have_cap_weighted_mean_zero_and_unit_std_in_the_universe(computed, name) -> None:
    _, out, want = computed
    got = out[name].transpose("timestamp", "symbol").to_numpy()
    estu, cap_before = want["estu"] > 0, want["cap_before"]
    checked = 0
    for t in range(_T):
        inside = estu[t] & np.isfinite(got[t])
        if inside.sum() < 2:
            continue
        assert abs(np.average(got[t, inside], weights=cap_before[t, inside])) < 1e-10
        assert abs(got[t, inside].std(ddof=1) - 1.0) < 1e-10
        checked += 1
    assert checked >= _T - _WINDOW


def test_symbols_outside_the_universe_still_get_exposures(computed) -> None:
    _, out, want = computed
    outside = want["estu"] == 0
    for name in ("style_size", "style_beta"):
        got = out[name].transpose("timestamp", "symbol").to_numpy()
        exposed = outside & np.isfinite(got)
        assert exposed[_WINDOW:].sum() > 0
    # A clipped small cap outside the universe keeps an exposure.
    clipped = out["desc_lncap"].sel(symbol=f"S{_CLIPPED:02d}").values[1:]
    assert (clipped == -3.0).all()
    assert np.isfinite(out["style_size"].sel(symbol=f"S{_CLIPPED:02d}").values[1:]).all()


def test_a_descriptor_beyond_the_data_error_threshold_is_dropped(computed) -> None:
    _, out, _ = computed
    # The micro cap sits about 20 standard deviations below the universe.
    assert np.isnan(out["desc_lncap"].sel(symbol=f"S{_TINY:02d}").values).all()
    assert np.isnan(out["style_size"].sel(symbol=f"S{_TINY:02d}").values).all()


def test_stream_mode_is_refused(tmp_path) -> None:
    single = _store(tmp_path, "all", _inputs())
    with pytest.raises(ValueError, match="batch only"):
        BarraStyle(_config(tmp_path, mode="stream", dataset=single))
    with pytest.raises(ValueError, match="stream mode"):
        BarraStyle(_config(tmp_path, mode="stream"))
    with pytest.raises(ValueError, match="batch only"):
        BarraStyle(_config(tmp_path)).init_stream()


def test_the_factor_rebuilds_from_its_config_json(computed, tmp_path) -> None:
    factor, out, _ = computed
    saved = json.loads(json.dumps(factor.get_config()))
    rebuilt = rebuild(saved)
    assert type(rebuilt) is BarraStyle
    assert rebuilt.config == factor.config
    again = compute_all(rebuilt)
    xr.testing.assert_identical(again, out)


def test_data_columns_and_kwargs_are_checked(tmp_path) -> None:
    with pytest.raises(ValueError, match="data_columns must exactly match"):
        BarraStyle(_config(tmp_path, data_columns=("adjClose", "marketcap")))
    with pytest.raises(ValueError, match="unknown config.kwargs"):
        BarraStyle(_config(tmp_path, kwargs={"beta_windw": 10}))
    with pytest.raises(ValueError, match="beta_min_observations"):
        BarraStyle(_config(tmp_path, kwargs={"beta_min_observations": 300}))


def test_defaults_are_use4_where_published() -> None:
    params = BarraStyleParameters()
    assert (params.beta_window, params.beta_half_life, params.clip_sigma) == (252, 63.0, 3.0)
    assert params.estimation_universe_size == 3000
    assert params.warmup_bars == 253
