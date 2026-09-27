"""The first-priority additions to the factor analysis.

- Pearson IC beside the rank IC.
- A Newey-West t-statistic of the mean IC, which overlapping multi-bar
  forward returns need.
- The rank autocorrelation at several lags, not only one.
- Annualized return, volatility, Sharpe ratio and maximum drawdown of the
  long-short portfolio.
- The IC decay across forward-return horizons when two or more frets are
  analyzed.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

from quantlab.analysis.factor_report import FactorAnalyzer, newey_west_t_stat

T, S = 120, 25


def _coords(periods: int = T, symbols: int = S) -> dict:
    return {
        "timestamp": pd.date_range("2024-01-01", periods=periods),
        "symbol": [f"S{i}" for i in range(symbols)],
    }


def _da(values: np.ndarray) -> xr.DataArray:
    periods, symbols = values.shape
    return xr.DataArray(values, coords=_coords(periods, symbols), dims=("timestamp", "symbol"))


@pytest.fixture
def signal_and_return():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(T, S))
    f = 0.3 * r + rng.normal(size=(T, S)) + np.cumsum(rng.normal(size=(T, S)) * 0.2, axis=0)
    f[rng.random((T, S)) < 0.1] = np.nan
    return f, r


# -- Pearson IC -------------------------------------------------------------------


def test_the_pearson_ic_is_the_per_period_linear_correlation(signal_and_return):
    f, r = signal_and_return

    pair = FactorAnalyzer(plot=False).analyze_pair(_da(f), _da(r), "f", "ret_1")

    expected = []
    for t in range(T):
        ok = np.isfinite(f[t]) & np.isfinite(r[t])
        expected.append(np.corrcoef(f[t, ok], r[t, ok])[0, 1])
    np.testing.assert_allclose(pair.pearson_ic.to_numpy(), expected, rtol=1e-10)
    assert pair.summary["pearson_ic_mean"] == pytest.approx(np.mean(expected))
    assert pair.summary["pearson_ic_t_stat"] == pytest.approx(
        np.mean(expected) / np.std(expected, ddof=1) * np.sqrt(T)
    )


# -- Newey-West ---------------------------------------------------------------------


def _reference_newey_west(x: np.ndarray, lags: int) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    variance = e @ e / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return x.mean() / np.sqrt(variance / n)


@pytest.mark.parametrize("lags", [0, 1, 4, 9])
def test_the_newey_west_t_stat_matches_the_bartlett_formula(lags):
    x = np.random.default_rng(2).normal(0.05, 1.0, size=200)
    x[[3, 50]] = np.nan

    t_stat, p_value = newey_west_t_stat(pd.Series(x), lags)

    assert t_stat == pytest.approx(_reference_newey_west(x, lags), rel=1e-12)
    assert p_value == pytest.approx(2 * stats.t.sf(abs(t_stat), 197), rel=1e-12)


def test_overlapping_returns_get_a_newey_west_t_stat_with_horizon_lags():
    # A persistent factor and overlapping 5-bar returns: neighbouring ICs share
    # four of their five daily shocks, so the IC series is autocorrelated.
    rng = np.random.default_rng(3)
    f = np.tile(rng.normal(size=S), (T, 1)) + 0.1 * rng.normal(size=(T, S))
    daily = 0.1 * np.tile(f[0], (T + 5, 1)) + rng.normal(size=(T + 5, S))
    r5 = sum(daily[k : k + T] for k in range(5))

    pair = FactorAnalyzer(plot=False).analyze_pair(_da(f), _da(r5), "f", "ret_5", horizon=5)
    s = pair.summary

    assert s["ic_nw_lags"] >= 4
    assert s["ic_nw_t_stat"] == pytest.approx(
        _reference_newey_west(pair.ic.to_numpy(), s["ic_nw_lags"])
    )
    assert abs(s["ic_nw_t_stat"]) < abs(s["ic_t_stat"])


# -- rank autocorrelation at several lags ------------------------------------------------


def test_the_rank_autocorrelation_is_reported_at_every_lag(signal_and_return):
    f, r = signal_and_return

    pair = FactorAnalyzer(plot=False, autocorrelation_lags=(1, 5, 10)).analyze_pair(
        _da(f), _da(r), "f", "ret_1"
    )

    assert list(pair.rank_autocorrelations.columns) == [1, 5, 10]
    for lag in (1, 5, 10):
        expected = [np.nan] * lag
        for t in range(lag, T):
            ok = np.isfinite(f[t]) & np.isfinite(f[t - lag])
            expected.append(stats.spearmanr(f[t, ok], f[t - lag, ok]).statistic)
        np.testing.assert_allclose(
            pair.rank_autocorrelations[lag].to_numpy(), expected, rtol=1e-10, equal_nan=True
        )
        assert pair.summary[f"rank_autocorrelation_lag{lag}"] == pytest.approx(np.nanmean(expected))
    pd.testing.assert_series_equal(
        pair.rank_autocorrelation, pair.rank_autocorrelations[1].rename("rank_autocorrelation")
    )


def test_a_frozen_ranking_is_fully_autocorrelated_at_every_lag():
    rng = np.random.default_rng(4)
    f = np.tile(np.arange(S, dtype=float), (T, 1))

    pair = FactorAnalyzer(plot=False).analyze_pair(
        _da(f), _da(rng.normal(size=(T, S))), "frozen", "ret_1"
    )

    for lag in (1, 5, 10, 20):
        assert pair.summary[f"rank_autocorrelation_lag{lag}"] == pytest.approx(1.0)


# -- long-short statistics ---------------------------------------------------------------


def _two_bucket_pair(spread: np.ndarray):
    """Factor = symbol index; the top half earns +spread/2, the bottom -spread/2."""
    periods = len(spread)
    f = np.tile(np.arange(S - 1, dtype=float), (periods, 1))
    r = np.where(f >= (S - 1) / 2, 0.5, -0.5) * spread[:, None]
    return FactorAnalyzer(quantiles=2, plot=False).analyze_pair(_da(f), _da(r), "f", "ret_1")


def test_a_steady_long_short_has_no_drawdown_and_compounds_to_its_annual_rate():
    pair = _two_bucket_pair(np.full(T, 0.001))
    s = pair.summary

    assert s["periods_per_year"] == pytest.approx(365.25, rel=1e-3)
    assert s["long_short_annual_return"] == pytest.approx(1.001 ** s["periods_per_year"] - 1)
    assert s["long_short_max_drawdown"] == pytest.approx(0.0)
    assert s["long_short_annual_volatility"] == pytest.approx(0.0, abs=1e-12)


def test_the_long_short_drawdown_is_the_worst_fall_from_a_peak():
    spread = np.r_[np.full(40, 0.01), np.full(20, -0.01), np.full(60, 0.01)]
    s = _two_bucket_pair(spread).summary

    assert s["long_short_max_drawdown"] == pytest.approx(0.99**20 - 1)
    rate = spread
    assert s["long_short_sharpe"] == pytest.approx(
        rate.mean() / rate.std(ddof=1) * np.sqrt(s["periods_per_year"])
    )


# -- IC decay across horizons --------------------------------------------------------


class _Fret:
    """A duck-typed fret with a horizon."""

    def __init__(self, name: str, horizon: int):
        self.name = name
        self.config = SimpleNamespace(kwargs={"n_forward_periods": horizon})

    def get_config(self):
        return {"name": self.name}


class _Factor:
    class_name = "Stub"

    def __init__(self, names):
        self.names = tuple(names)

    def get_factor_names(self):
        return self.names

    def get_config(self):
        return {"name": "stub"}


def test_two_or_more_frets_add_an_ic_decay_table_and_figure(tmp_path):
    rng = np.random.default_rng(5)
    daily = rng.normal(size=(T + 10, S))
    coords = _coords()
    features = xr.Dataset(
        {"a": (("timestamp", "symbol"), daily[:T] + rng.normal(size=(T, S))),
         "b": (("timestamp", "symbol"), rng.normal(size=(T, S)))},
        coords=coords,
    )
    horizons = (1, 5, 10)
    labels = [
        xr.Dataset({f"ret_{h}": (("timestamp", "symbol"), sum(daily[k : k + T] for k in range(h)))},
                   coords=coords)
        for h in horizons
    ]
    frets = [_Fret(f"ret_{h}", h) for h in horizons]

    analysis = FactorAnalyzer(quantiles=3, plot=False).run(
        _Factor(["a", "b"]), frets, features, labels, output_dir=tmp_path / "report"
    )

    decay = analysis.ic_decay_table()
    assert list(decay.columns) == [
        "factor", "fret", "horizon", "ic_mean", "ic_nw_t_stat", "ci_low", "ci_high",
    ]
    a = decay[decay["factor"] == "a"]
    assert list(a["horizon"]) == [1, 5, 10]
    assert a["ic_mean"].iloc[0] > a["ic_mean"].iloc[-1] > 0
    assert (a["ci_low"] < a["ic_mean"]).all() and (a["ic_mean"] < a["ci_high"]).all()
    out = tmp_path / "report"
    assert (out / "ic_decay.csv").is_file()
    assert not list(out.glob("*decay*.png"))  # the decay is a panel of each pair figure
    summary = json.loads((out / "summary.json").read_text())
    assert "ic_nw_t_stat" in summary["pairs"][0]

    held = FactorAnalyzer(quantiles=3).run(_Factor(["a"]), frets, features, labels)
    titles = [ax.get_title(loc="left") for ax in held.figures["a__ret_5"].axes]
    assert any(title.startswith("Mean IC by horizon") for title in titles)


def test_one_fret_writes_no_decay_files(tmp_path):
    rng = np.random.default_rng(6)
    features = xr.Dataset({"a": (("timestamp", "symbol"), rng.normal(size=(T, S)))}, coords=_coords())
    labels = xr.Dataset({"ret_1": (("timestamp", "symbol"), rng.normal(size=(T, S)))}, coords=_coords())

    analysis = FactorAnalyzer(quantiles=3, plot=False).run(
        _Factor(["a"]), [_Fret("ret_1", 1)], features, [labels], output_dir=tmp_path / "r"
    )

    assert len(analysis.ic_decay_table()) == 1
    assert not list((tmp_path / "r").glob("*decay*"))

    held = FactorAnalyzer(quantiles=3).run(_Factor(["a"]), [_Fret("ret_1", 1)], features, [labels])
    titles = [ax.get_title(loc="left") for ax in held.figures["a__ret_1"].axes]
    assert not any(title.startswith("Mean IC by horizon") for title in titles)


def test_a_long_short_that_loses_everything_stays_at_zero():
    spread = np.r_[np.full(10, 0.01), np.full(1, -1.5), np.full(20, 0.01)]
    s = _two_bucket_pair(spread).summary

    assert s["long_short_annual_return"] == pytest.approx(-1.0)
    assert s["long_short_max_drawdown"] == pytest.approx(-1.0)


def test_the_autocorrelation_panel_draws_every_lag(signal_and_return):
    f, r = signal_and_return
    analyzer = FactorAnalyzer(autocorrelation_lags=(1, 5, 10))
    pair = analyzer.analyze_pair(_da(f), _da(r), "f", "ret_1")

    from quantlab.analysis.factor_report import FactorReportFigure

    fig = FactorReportFigure().render(pair)
    panel = next(ax for ax in fig.axes if ax.get_title(loc="left").startswith("Rank autocorrelation"))
    labels = [line.get_label() for line in panel.get_lines() if not line.get_label().startswith("_")]
    assert [label.split()[1] for label in labels] == ["1", "5", "10"]
    assert not any(ax.get_title(loc="left") == "Mean by lag" for ax in fig.axes)
