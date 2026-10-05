"""``BarraStyle`` (the price-based styles) against a float64 numpy reference.

The input is a merge of two stores: adjusted closes with a risk-free rate,
and market caps. The reference follows USE4's definitions directly: the
estimation universe is the top N by the previous bar's cap, the market is
its cap-weighted return, BETA and HSIGMA come from a weighted least-squares fit solved with
``numpy.linalg.lstsq`` (not the closed forms the operators use), as do the
orthogonalizations, and each descriptor and style is standardized with a
cap-weighted mean and an equally weighted standard deviation. Windows are
shortened through ``kwargs`` so 100 bars cover them. 21 symbols make macOS
pad the axis. KunQuant's ``Log`` is accurate to about 4e-10, so outputs
built on log returns are compared at 1e-7.
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
_MOM_WINDOW, _MOM_HALF_LIFE, _MOM_LAG, _MOM_MIN = 30, 10.0, 5, 15
_DASTD_WINDOW, _DASTD_HALF_LIFE = 30, 8.0
_CMRA_MONTHS, _CMRA_LENGTH = 3, 8
_VOL_MIN = 12
_CRASH = 5  # a symbol whose cumulative log return falls below -1: no CMRA
_SPLIT, _TWIN = 19, 20  # one company twice: a 2-for-1 split, and no split
_SPLIT_BAR = 50
_MONTH, _STOQ, _STOA, _LIQ_MIN = 5, 3, 6, 0.5
_DIV_WINDOW = 40
_KWARGS = {
    "beta_window": _WINDOW,
    "beta_half_life": _HALF_LIFE,
    "beta_min_observations": _MIN_OBS,
    "estimation_universe_size": _ESTU,
    "momentum_window": _MOM_WINDOW,
    "momentum_half_life": _MOM_HALF_LIFE,
    "momentum_lag": _MOM_LAG,
    "momentum_min_observations": _MOM_MIN,
    "dastd_window": _DASTD_WINDOW,
    "dastd_half_life": _DASTD_HALF_LIFE,
    "cmra_months": _CMRA_MONTHS,
    "cmra_month_length": _CMRA_LENGTH,
    "volatility_min_observations": _VOL_MIN,
    "liquidity_month_length": _MONTH,
    "stoq_months": _STOQ,
    "stoa_months": _STOA,
    "liquidity_min_fraction": _LIQ_MIN,
    "dividend_window": _DIV_WINDOW,
}
_RESVOL_WEIGHTS = (0.75, 0.15, 0.10)
_LIQUIDITY_WEIGHTS = (0.35, 0.35, 0.30)
_COLUMNS = ("adjClose", "marketcap", "risk_free", "close", "volume", "divCash", "splitFactor")


def _inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    market = rng.normal(0.0004, 0.012, size=_T)
    risk_free = np.abs(rng.normal(0.0001, 0.00003, size=_T))
    betas = rng.normal(1.0, 0.4, size=_S)
    returns = risk_free[:, None] + market[:, None] * betas + rng.normal(0, 0.015, (_T, _S))
    returns[60:70, _CRASH] = -0.12  # a crash: the cumulative log return goes below -1
    returns[:, _TWIN] = returns[:, _SPLIT]
    price = 20.0 * np.cumprod(1.0 + returns, axis=0)
    shares = rng.lognormal(18.0, 1.0, size=_S)
    shares[_TINY] = 1.0e3
    shares[_TWIN] = shares[_SPLIT]
    cap = price * shares
    # Volume as a daily turnover of the share count; a few days without one.
    volume = rng.lognormal(-5.0, 0.6, size=(_T, _S)) * shares
    volume[rng.random((_T, _S)) < 0.05] = np.nan
    volume[:, _TWIN] = volume[:, _SPLIT]
    # Every other company pays 0.4% of its price every 15 bars.
    dividend = np.zeros((_T, _S))
    dividend[7::15, ::2] = 0.004 * price[7::15, ::2]
    dividend[:, _TWIN] = dividend[:, _SPLIT] = 0.004 * price[:, _SPLIT] * (np.arange(_T) % 15 == 7)
    close = price.copy()
    split = np.ones((_T, _S))
    # Before its 2-for-1 split the company had half the shares at twice the price.
    split[_SPLIT_BAR, _SPLIT] = 2.0
    close[:_SPLIT_BAR, _SPLIT] *= 2.0
    volume[:_SPLIT_BAR, _SPLIT] /= 2.0
    dividend[:_SPLIT_BAR, _SPLIT] *= 2.0
    for values in (price, cap, close, volume):
        values[:_LISTING_BAR, _LISTED_LATE] = np.nan
    cap[[30, 31, 64], _CAP_GAP] = np.nan
    price[47, 2] = np.nan  # a missing bar inside the window
    return {
        "adjClose": price,
        "marketcap": cap,
        "risk_free": np.broadcast_to(risk_free[:, None], (_T, _S)).copy(),
        "close": close,
        "volume": volume,
        "divCash": dividend,
        "splitFactor": split,
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
    price_columns = ("adjClose", "risk_free", "close", "volume", "divCash", "splitFactor")
    prices = _store(tmp_path, "prices", {k: inputs[k] for k in price_columns})
    caps = _store(tmp_path, "caps", {"marketcap": inputs["marketcap"]})
    values = {
        "warmup_bars": 0,
        "dataset": [prices, caps],
        "mode": "batch",
        "data_columns": _COLUMNS,
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


def _windowed(values: np.ndarray, window: int):
    """Yield ``(t, rows)`` for every bar with ``window`` bars of history."""
    for t in range(window - 1, values.shape[0]):
        yield t, values[t - window + 1 : t + 1]


def _ew_weights(window: int, half_life: float) -> np.ndarray:
    return 0.5 ** (np.arange(window)[::-1] / half_life)


def _wls_fit(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """``(slope, residual std)`` of a weighted fit with an intercept, and the weight sum."""
    root = np.sqrt(weights)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
    residual = y - design @ coefficients
    return coefficients[1], np.sqrt((weights * residual**2).sum() / weights.sum()), weights.sum()


def _residual(y: np.ndarray, regressors: list[np.ndarray], w: np.ndarray, estu: np.ndarray) -> np.ndarray:
    """Per-bar weighted least-squares residual, fitted in the universe, a missing regressor at its mean."""
    out = np.full_like(y, np.nan)
    for t in range(y.shape[0]):
        fit = estu[t] & np.isfinite(y[t]) & np.isfinite(w[t]) & (w[t] > 0)
        for x in regressors:
            fit &= np.isfinite(x[t])
        if fit.sum() <= len(regressors):
            continue
        root = np.sqrt(w[t, fit])
        design = np.column_stack([np.ones(fit.sum())] + [x[t, fit] for x in regressors])
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], y[t, fit] * root, rcond=None)
        means = [np.average(x[t, fit], weights=w[t, fit]) for x in regressors]
        fitted = coefficients[0] + sum(
            c * np.where(np.isfinite(x[t]), x[t], m)
            for c, x, m in zip(coefficients[1:], regressors, means)
        )
        out[t] = y[t] - fitted
    return out


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

    log_excess = np.log1p(stock_return) - np.log1p(risk_free)

    weights = _ew_weights(_WINDOW, _HALF_LIFE)
    beta = np.full((_T, _S), np.nan)
    hsigma = np.full((_T, _S), np.nan)
    for t in range(_WINDOW - 1, _T):
        rows = slice(t - _WINDOW + 1, t + 1)
        for s in range(_S):
            y, x = excess[rows, s], market_excess[rows, s]
            ok = np.isfinite(y) & np.isfinite(x)
            if ok.sum() >= _MIN_OBS:
                beta[t, s] = _wls_slope(y[ok], x[ok], weights[ok])
                hsigma[t, s] = _wls_fit(y[ok], x[ok], weights[ok])[1]

    lagged = np.vstack([np.full((_MOM_LAG, _S), np.nan), log_excess[:-_MOM_LAG]])
    rstr = np.full((_T, _S), np.nan)
    momentum_weights = _ew_weights(_MOM_WINDOW, _MOM_HALF_LIFE)[:, None]
    for t, rows in _windowed(lagged, _MOM_WINDOW):
        ok = np.isfinite(rows)
        weighted = np.where(ok, rows * momentum_weights, 0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = weighted / np.where(ok, momentum_weights, 0).sum(axis=0)
        rstr[t] = np.where(ok.sum(axis=0) >= _MOM_MIN, normalized, np.nan)

    dastd = np.full((_T, _S), np.nan)
    dastd_weights = _ew_weights(_DASTD_WINDOW, _DASTD_HALF_LIFE)
    for t, rows in _windowed(excess, _DASTD_WINDOW):
        for s in range(_S):
            ok = np.isfinite(rows[:, s])
            if ok.sum() >= _VOL_MIN:
                mean = np.average(rows[ok, s], weights=dastd_weights[ok])
                dastd[t, s] = np.sqrt(np.average((rows[ok, s] - mean) ** 2, weights=dastd_weights[ok]))

    cmra = np.full((_T, _S), np.nan)
    for t, rows in _windowed(log_excess, _CMRA_MONTHS * _CMRA_LENGTH):
        filled = np.where(np.isfinite(rows), rows, 0.0)
        z = np.array([filled[-k * _CMRA_LENGTH :].sum(axis=0) for k in range(1, _CMRA_MONTHS + 1)])
        low, high = z.min(axis=0), z.max(axis=0)
        enough = np.isfinite(rows).sum(axis=0) >= _VOL_MIN
        with np.errstate(invalid="ignore", divide="ignore"):
            cmra[t] = np.where(enough & (low > -1.0), np.log1p(high) - np.log1p(low), np.nan)

    turnover = inputs["volume"] * inputs["close"] / cap

    def share_turnover(months):
        window = months * _MONTH
        out = np.full((_T, _S), np.nan)
        for t, rows in _windowed(turnover, window):
            ok = np.isfinite(rows)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(ok, rows, 0).sum(axis=0) / ok.sum(axis=0)
                out[t] = np.where(ok.sum(axis=0) >= max(1.0, _LIQ_MIN * window), np.log(_MONTH * mean), np.nan)
        return out

    basis = np.cumprod(inputs["splitFactor"], axis=0)
    yild = np.full((_T, _S), np.nan)
    for t, rows in _windowed(inputs["divCash"] * basis, _DIV_WINDOW):
        yild[t] = rows.sum(axis=0) / (basis[t] * inputs["close"][t])

    def descriptor(raw):
        return _clip(_standardize(raw, cap_before, estu))

    def standardize(value):
        return _standardize(value, cap_before, estu)

    desc = {
        "lncap": descriptor(np.log(cap)),
        "beta": descriptor(beta),
        "rstr": descriptor(rstr),
        "dastd": descriptor(dastd),
        "cmra": descriptor(cmra),
        "hsigma": descriptor(hsigma),
        "stom": descriptor(share_turnover(1)),
        "stoq": descriptor(share_turnover(_STOQ)),
        "stoa": descriptor(share_turnover(_STOA)),
        "yild": descriptor(yild),
    }
    style_size, style_beta = standardize(desc["lncap"]), standardize(desc["beta"])
    desc["nlsize"] = descriptor(style_size**3)
    desc["nlbeta"] = descriptor(style_beta**3)
    parts = [desc["dastd"], desc["cmra"], desc["hsigma"]]
    total = sum(np.where(np.isfinite(d), d, 0.0) * w for d, w in zip(parts, _RESVOL_WEIGHTS))
    present = sum(np.isfinite(d) * w for d, w in zip(parts, _RESVOL_WEIGHTS))
    with np.errstate(invalid="ignore", divide="ignore"):
        combined = standardize(np.where(present > 0, total / present, np.nan))
    liquidity_parts = [desc["stom"], desc["stoq"], desc["stoa"]]
    liquidity_total = sum(np.where(np.isfinite(d), d, 0.0) * w for d, w in zip(liquidity_parts, _LIQUIDITY_WEIGHTS))
    liquidity_present = sum(np.isfinite(d) * w for d, w in zip(liquidity_parts, _LIQUIDITY_WEIGHTS))
    with np.errstate(invalid="ignore", divide="ignore"):
        liquidity = np.where(liquidity_present > 0, liquidity_total / liquidity_present, np.nan)
    root_cap = np.sqrt(cap_before)
    out = {f"desc_{name}": value for name, value in desc.items()}
    out.update({
        "style_size": style_size,
        "style_beta": style_beta,
        "style_momentum": standardize(desc["rstr"]),
        "style_residual_volatility": standardize(
            _residual(combined, [style_beta, style_size], root_cap, estu)
        ),
        "style_nonlinear_size": standardize(_residual(desc["nlsize"], [style_size], root_cap, estu)),
        "style_nonlinear_beta": standardize(_residual(desc["nlbeta"], [style_beta], root_cap, estu)),
        "style_liquidity": standardize(liquidity),
        "style_dividend_yield": standardize(desc["yild"]),
        "estu": estu.astype(np.float64),
        "cap_before": cap_before,
        "dastd_raw": dastd,
        "hsigma_raw": hsigma,
        "cmra_raw": cmra,
    })
    return out


@pytest.fixture(scope="module")
def computed(tmp_path_factory) -> tuple[BarraStyle, xr.Dataset, dict[str, np.ndarray]]:
    tmp_path = tmp_path_factory.mktemp("barra")
    factor = BarraStyle(_config(tmp_path))
    return factor, compute_all(factor), _reference()


# --- tests ----------------------------------------------------------------


_EXACT = ("desc_lncap", "desc_beta", "desc_dastd", "desc_hsigma", "desc_nlsize", "desc_nlbeta",
          "desc_yild", "style_size", "style_beta", "style_nonlinear_size", "style_nonlinear_beta",
          "style_dividend_yield", "estu")
# KunQuant's Log is accurate to about 4e-10.
_ON_LOGS = ("desc_rstr", "desc_cmra", "desc_stom", "desc_stoq", "desc_stoa", "style_momentum",
            "style_residual_volatility", "style_liquidity")
_STYLES = ("style_size", "style_beta", "style_momentum", "style_residual_volatility",
           "style_nonlinear_size", "style_nonlinear_beta", "style_liquidity", "style_dividend_yield")


@pytest.mark.parametrize("name", _EXACT + _ON_LOGS)
def test_outputs_match_the_numpy_reference(computed, name) -> None:
    _, out, want = computed
    got = out[name].transpose("timestamp", "symbol").to_numpy()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want[name]))
    finite = np.isfinite(want[name])
    assert finite.sum() > _S * 10, "the fixture should give most bars a value"
    tolerance = 1e-9 if name in _EXACT else 1e-7
    np.testing.assert_allclose(got[finite], want[name][finite], rtol=tolerance, atol=tolerance)


_ORTHOGONALIZED = {
    "style_residual_volatility": ("style_beta", "style_size"),
    "style_nonlinear_size": ("style_size",),
    "style_nonlinear_beta": ("style_beta",),
}


@pytest.mark.parametrize("name", list(_ORTHOGONALIZED))
def test_orthogonalized_styles_have_no_weighted_correlation_with_their_regressors(computed, name) -> None:
    _, out, want = computed
    got = out[name].transpose("timestamp", "symbol").to_numpy()
    regressors = [out[r].transpose("timestamp", "symbol").to_numpy() for r in _ORTHOGONALIZED[name]]
    weights = np.sqrt(want["cap_before"])
    checked = 0
    for t in range(_T):
        fit = (want["estu"][t] > 0) & np.isfinite(got[t]) & np.isfinite(weights[t])
        for x in regressors:
            fit &= np.isfinite(x[t])
        if fit.sum() < 5:
            continue
        w = weights[t, fit]
        dy = got[t, fit] - np.average(got[t, fit], weights=w)
        for x in regressors:
            dx = x[t, fit] - np.average(x[t, fit], weights=w)
            correlation = np.average(dy * dx, weights=w) / np.sqrt(
                np.average(dy**2, weights=w) * np.average(dx**2, weights=w)
            )
            assert abs(correlation) < 1e-9
        checked += 1
    assert checked > _T // 2


def test_a_split_leaves_turnover_and_dividend_yield_continuous(computed) -> None:
    _, out, _ = computed
    # The split company and its unsplit twin are the same economics in two
    # share bases, so every descriptor and style agrees.
    for name in ("desc_stom", "desc_stoq", "desc_stoa", "desc_yild", "style_liquidity",
                 "style_dividend_yield"):
        split = out[name].sel(symbol=f"S{_SPLIT:02d}").values
        twin = out[name].sel(symbol=f"S{_TWIN:02d}").values
        np.testing.assert_allclose(split, twin, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert np.isfinite(split[_SPLIT_BAR + _DIV_WINDOW :]).all()


def test_a_symbol_missing_a_descriptor_still_gets_residual_volatility(computed) -> None:
    _, out, want = computed
    style = out["style_residual_volatility"].transpose("timestamp", "symbol").to_numpy()
    # Before BETA exists anywhere the orthogonalization has no sample.
    regressed = np.arange(_T)[:, None] >= _WINDOW
    no_hsigma = np.isnan(want["desc_hsigma"]) & np.isfinite(want["desc_dastd"]) & regressed
    no_cmra = np.isnan(want["cmra_raw"]) & np.isfinite(want["desc_hsigma"]) & regressed
    assert no_hsigma[:, _LISTED_LATE].any(), "a late listing has DASTD before HSIGMA"
    assert no_cmra[:, _CRASH].any(), "the crash leaves CMRA's log domain"
    assert np.isfinite(style[no_hsigma]).all() and np.isfinite(style[no_cmra]).all()


def test_the_fixture_exercises_clipping_late_listing_and_the_minimum_count(computed) -> None:
    _, _, want = computed
    assert (want["desc_lncap"][1:, _CLIPPED] == -3.0).all()
    # BETA waits for _MIN_OBS returns after listing, then exists.
    first = _LISTING_BAR + 1 + _MIN_OBS - 1
    assert np.isnan(want["desc_beta"][:first, _LISTED_LATE]).all()
    assert np.isfinite(want["desc_beta"][first:, _LISTED_LATE]).all()
    assert np.isfinite(want["desc_beta"][_WINDOW - 1 :, 0]).all()


@pytest.mark.parametrize("name", _STYLES)
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
    assert checked >= _T - _WINDOW - _MOM_LAG


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
        BarraStyle(_config(tmp_path, data_columns=("adjClose", "marketcap", "risk_free")))
    with pytest.raises(ValueError, match="unknown config.kwargs"):
        BarraStyle(_config(tmp_path, kwargs={"beta_windw": 10}))
    with pytest.raises(ValueError, match="beta_min_observations"):
        BarraStyle(_config(tmp_path, kwargs={"beta_min_observations": 300}))


def test_defaults_are_use4_where_published() -> None:
    params = BarraStyleParameters()
    assert (params.beta_window, params.beta_half_life, params.clip_sigma) == (252, 63.0, 3.0)
    assert params.estimation_universe_size == 3000
    assert (params.momentum_window, params.momentum_half_life, params.momentum_lag) == (504, 126.0, 21)
    assert (params.dastd_window, params.dastd_half_life) == (252, 42.0)
    assert (params.cmra_months, params.cmra_month_length) == (12, 21)
    assert (params.dastd_weight, params.cmra_weight, params.hsigma_weight) == (0.75, 0.15, 0.10)
    assert (params.liquidity_month_length, params.stoq_months, params.stoa_months) == (21, 3, 12)
    assert (params.stom_weight, params.stoq_weight, params.stoa_weight) == (0.35, 0.35, 0.30)
    assert params.dividend_window == 252
    # Our choices, where MSCI publishes none.
    assert params.orthogonalization_weighting == "sqrt_cap"
    assert (params.momentum_min_observations, params.volatility_min_observations) == (252, 63)
    assert (params.liquidity_min_fraction, params.data_error_sigma) == (0.5, 10.0)
    assert params.warmup_bars == 526


def test_equal_weighting_orthogonalizes_with_equal_weights(tmp_path) -> None:
    config = _config(tmp_path, kwargs={**_KWARGS, "orthogonalization_weighting": "equal"},
                     factor_names=("style_nonlinear_size", "style_size", "estu"))
    out = compute_all(BarraStyle(config))
    got = out["style_nonlinear_size"].transpose("timestamp", "symbol").to_numpy()
    size = out["style_size"].transpose("timestamp", "symbol").to_numpy()
    estu = out["estu"].transpose("timestamp", "symbol").to_numpy() > 0
    checked = 0
    for t in range(_T):
        fit = estu[t] & np.isfinite(got[t]) & np.isfinite(size[t])
        if fit.sum() < 5:
            continue
        dy, dx = got[t, fit] - got[t, fit].mean(), size[t, fit] - size[t, fit].mean()
        assert abs((dy * dx).mean()) < 1e-9 * np.sqrt((dy**2).mean() * (dx**2).mean())
        checked += 1
    assert checked > _T // 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"orthogonalization_weighting": "volume"}, "orthogonalization_weighting"),
        ({"momentum_min_observations": 0}, "momentum_min_observations"),
        ({"volatility_min_observations": 100}, "volatility_min_observations"),
        ({"cmra_weight": 0.0}, "positive"),
        ({"momentum_lag": -1}, "momentum_lag"),
        ({"liquidity_min_fraction": 0.0}, "liquidity_min_fraction"),
        ({"stoq_months": 13}, "stoq_months"),
        ({"stoa_weight": -1.0}, "positive"),
        ({"dividend_window": 0}, "dividend_window"),
    ],
)
def test_invalid_price_style_parameters_are_refused(tmp_path, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        BarraStyle(_config(tmp_path, kwargs={**_KWARGS, **kwargs}))
