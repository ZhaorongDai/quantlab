"""``MarketFeatures`` turns index/ETF series into market features on every symbol.

Each series gets 21 features (the bar return; for d in 5, 10, 20, 30, 60 the
mean and standard deviation of the return and of the dollar amount, the
amount ones divided by the bar's amount), the MASTER market inputs of
``docs/research/qlib-gats-master.md`` section 2.1. The features are
broadcast to every symbol of the factor's target dataset that has a bar, so
the panel merges into a model's panel like any factor's.

Everything is synthetic: a six-symbol business-day stock store is the
target, and single-symbol stores with known prices are the series. The
expected values are computed here with plain numpy loops.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import DatasetConfig, MarketFeatureConfig, PolarsFactorConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.predefined.market import MarketFeatures
from quantlab.utils.module import load_factor_from_config
from tests.backtest_fixtures import PastReturnFactor, write_price_store

WINDOWS = (5, 10, 20, 30, 60)
N_BARS = 120
DAYS = pd.bdate_range("2024-01-01", periods=N_BARS)


def _write_series(
    path: Path,
    symbol: str,
    *,
    seed: int,
    amount: bool = False,
    drop_bars: tuple[int, ...] = (),
) -> tuple[DatasetConfig, dict[str, np.ndarray]]:
    """Write a one-symbol store on ``DAYS`` and return its config and values."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, N_BARS)))
    volume = rng.uniform(1e6, 5e6, N_BARS)
    values = {
        "adjClose": close,
        "adjVolume": volume,
        "close": close * 3.0,
        "volume": volume / 3.0,
    }
    if amount:
        values["amount"] = rng.uniform(1e8, 9e8, N_BARS)
    keep = np.setdiff1d(np.arange(N_BARS), drop_bars)
    panel = xr.Dataset(
        {k: (["timestamp", "symbol"], v[keep, None]) for k, v in values.items()},
        coords={"timestamp": DAYS[keep], "symbol": [symbol]},
    )
    panel.to_zarr(path, mode="w")
    config = DatasetConfig(
        raw_data_dir_path=str(path.parent / "raw"),
        zarr_file_path=str(path),
        market="us_equity",
        frequency="1d",
    )
    return config, {k: v[keep] for k, v in values.items()}


def _expected(close, amount) -> dict[str, np.ndarray]:
    """The 21 features of one series by plain loops, on the series' own bars."""
    n = len(close)
    ret = np.full(n, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1.0
    out = {"ret": ret}
    for d in WINDOWS:
        for key in ("ret_mean", "ret_std", "amount_mean", "amount_std"):
            out[f"{key}_{d}"] = np.full(n, np.nan)
        for t in range(d - 1, n):
            r = ret[t - d + 1 : t + 1]
            a = amount[t - d + 1 : t + 1]
            if np.isfinite(r).all():
                out[f"ret_mean_{d}"][t] = r.mean()
                out[f"ret_std_{d}"][t] = r.std(ddof=1)
            out[f"amount_mean_{d}"][t] = a.mean() / amount[t]
            out[f"amount_std_{d}"][t] = a.std(ddof=1) / amount[t]
    return out


@pytest.fixture
def stores(tmp_path):
    """Target stock store (FFF lists at bar 70, EEE delists at bar 100) and two series."""
    target = write_price_store(
        tmp_path / "target",
        n_bars=N_BARS,
        list_at={"FFF": 70},
        delist_at={"EEE": 100},
    )
    spy, spy_values = _write_series(tmp_path / "spy.zarr", "84398", seed=1)
    qqq, qqq_values = _write_series(tmp_path / "qqq.zarr", "86755", seed=2)
    return {
        "target": target,
        "spy": (spy, spy_values),
        "qqq": (qqq, qqq_values),
        "tmp": tmp_path,
    }


def _factor(stores, **overrides) -> MarketFeatures:
    fields = dict(
        dataset=StockDataset(stores["target"]),
        series={
            "spy": StockDataset(stores["spy"][0]),
            "qqq": StockDataset(stores["qqq"][0]),
        },
        file_path=str(stores["tmp"] / "factors" / "market.zarr"),
    )
    fields.update(overrides)
    return MarketFeatures(MarketFeatureConfig(**fields))


# -- names and config ------------------------------------------------------------


def test_every_series_gets_21_features_named_after_it(stores):
    names = _factor(stores).get_factor_names()

    assert len(names) == 42
    assert names[:5] == (
        "spy_ret",
        "spy_ret_mean_5",
        "spy_ret_std_5",
        "spy_amount_mean_5",
        "spy_amount_std_5",
    )
    assert names[20] == "spy_amount_std_60"
    assert names[21] == "qqq_ret"
    assert "qqq_ret_mean_20" in names


def test_the_warm_up_defaults_to_the_longest_window(stores):
    assert _factor(stores).warmup_bars == 60


def test_a_config_without_series_is_refused(stores):
    with pytest.raises(ValueError, match="at least one series"):
        _factor(stores, series={})


def test_a_series_name_that_is_not_an_identifier_is_refused(stores):
    with pytest.raises(ValueError, match="identifier"):
        _factor(stores, series={"s&p": StockDataset(stores["spy"][0])})


def test_an_unknown_kwarg_is_refused(stores):
    with pytest.raises(ValueError, match="unknown"):
        _factor(stores, kwargs={"close": "adjClose"})


def test_an_unknown_factor_name_is_refused(stores):
    with pytest.raises(ValueError, match="spy_ret_mean_7"):
        _factor(stores, factor_names=("spy_ret", "spy_ret_mean_7"))


# -- values -----------------------------------------------------------------------


def test_features_match_hand_computed_values(stores):
    panel = _factor(stores).compute(DAYS[60], DAYS[-1])
    for name in ("spy", "qqq"):
        values = stores[name][1]
        expected = _expected(values["adjClose"], values["adjClose"] * values["adjVolume"])
        for key, series in expected.items():
            np.testing.assert_allclose(
                panel[f"{name}_{key}"].sel(symbol="AAA").values,
                series[60:],
                rtol=1e-5,
                err_msg=f"{name}_{key}",
            )


def test_an_amount_column_replaces_volume_times_close(stores, tmp_path):
    config, values = _write_series(tmp_path / "iwm.zarr", "IWM", seed=3, amount=True)
    factor = _factor(
        stores,
        series={"iwm": StockDataset(config)},
        kwargs={"amount_column": "amount"},
    )
    panel = factor.compute(DAYS[60], DAYS[-1])
    expected = _expected(values["adjClose"], values["amount"])

    np.testing.assert_allclose(
        panel["iwm_amount_std_30"].sel(symbol="AAA").values,
        expected["amount_std_30"][60:],
        rtol=1e-5,
    )


def test_the_price_and_volume_columns_are_configurable(stores):
    factor = _factor(
        stores,
        series={"spy": StockDataset(stores["spy"][0])},
        kwargs={"close_column": "close", "volume_column": "volume"},
    )
    panel = factor.compute(DAYS[60], DAYS[-1])
    values = stores["spy"][1]
    expected = _expected(values["close"], values["close"] * values["volume"])

    np.testing.assert_allclose(
        panel["spy_amount_mean_10"].sel(symbol="AAA").values,
        expected["amount_mean_10"][60:],
        rtol=1e-5,
    )


def test_a_series_missing_a_bar_rolls_over_its_own_bars(stores, tmp_path):
    config, values = _write_series(
        tmp_path / "gap.zarr", "GAP", seed=4, drop_bars=(90,)
    )
    factor = _factor(stores, series={"gap": StockDataset(config)})
    panel = factor.compute(DAYS[60], DAYS[-1])
    expected = _expected(values["adjClose"], values["adjClose"] * values["adjVolume"])
    own_bars = pd.DatetimeIndex(np.delete(DAYS.values, 90))

    got = panel["gap_ret_mean_20"].sel(symbol="AAA").to_series()
    assert np.isnan(got.loc[DAYS[90]])
    np.testing.assert_allclose(
        got.drop(DAYS[90]).values,
        pd.Series(expected["ret_mean_20"], index=own_bars).loc[DAYS[60]:].values,
        rtol=1e-5,
    )


def test_a_series_missing_a_warm_up_bar_still_fills_the_longest_window(stores, tmp_path):
    """The target has bar 30, the series does not: the 60-bar window at the
    first requested bar reaches one bar further back on the series' own
    calendar instead of starting short."""
    config, values = _write_series(
        tmp_path / "gap.zarr", "GAP", seed=5, drop_bars=(30,)
    )
    factor = _factor(stores, series={"gap": StockDataset(config)})
    panel = factor.compute(DAYS[65], DAYS[-1])
    expected = _expected(values["adjClose"], values["adjClose"] * values["adjVolume"])
    own_bars = pd.DatetimeIndex(np.delete(DAYS.values, 30))

    got = panel["gap_ret_mean_60"].sel(symbol="AAA", timestamp=DAYS[65]).item()
    want = pd.Series(expected["ret_mean_60"], index=own_bars).loc[DAYS[65]]
    assert np.isfinite(want)
    np.testing.assert_allclose(got, want, rtol=1e-5)


# -- broadcast ----------------------------------------------------------------------


def test_every_symbol_with_a_bar_carries_the_same_values(stores):
    panel = _factor(stores).compute(DAYS[60], DAYS[-1])
    target = StockDataset(stores["target"]).panel(DAYS[60], DAYS[-1])
    has_bar = target["close"].notnull()
    live_count = has_bar.sum("symbol")

    assert list(panel["symbol"].values) == list(target["symbol"].values)
    for name in panel.data_vars:
        live = panel[name].where(has_bar)
        spread = (live.max("symbol") - live.min("symbol")).fillna(0.0)
        assert float(abs(spread).max()) == 0.0, name
        # Defined on a bar means defined for every symbol with a bar there.
        count = live.notnull().sum("symbol")
        assert bool(((count == 0) | (count == live_count)).all()), name


def test_a_symbol_without_a_bar_carries_no_market_feature(stores):
    panel = _factor(stores).compute(DAYS[60], DAYS[-1])

    fff = panel["spy_ret_mean_5"].sel(symbol="FFF").to_series()
    eee = panel["spy_ret_mean_5"].sel(symbol="EEE").to_series()
    assert fff.loc[: DAYS[69]].isna().all()
    assert fff.loc[DAYS[70]:].notna().all()
    assert eee.loc[DAYS[100]:].isna().all()
    assert eee.loc[: DAYS[99]].notna().all()


def test_the_panel_merges_with_another_factor_on_the_target_axis(stores):
    target = StockDataset(stores["target"])
    past = PastReturnFactor(PolarsFactorConfig(
        warmup_bars=5, dataset=target, kwargs={"n": 5},
    ))
    market = _factor(stores, dataset=target).compute(DAYS[60], DAYS[-1])
    merged = xr.combine_by_coords([market, past.compute(DAYS[60], DAYS[-1])])

    assert dict(merged.sizes) == {"timestamp": N_BARS - 60, "symbol": 6}
    assert set(merged.data_vars) == set(market.data_vars) | {"past_ret_5"}


# -- date ranges and warm-up ------------------------------------------------------------


def test_compute_is_warm_on_the_first_requested_bar(stores):
    factor = _factor(stores)
    panel = factor.compute(DAYS[60], DAYS[70])

    assert panel.sizes["timestamp"] == 11
    assert pd.Timestamp(panel["timestamp"].values[0]) == DAYS[60]
    first = panel.isel(timestamp=0).sel(symbol="AAA")
    assert all(np.isfinite(float(first[name])) for name in panel.data_vars)


def test_a_short_warm_up_warns_and_leaves_the_long_windows_empty(stores):
    factor = _factor(stores)
    with pytest.warns(UserWarning, match="short by 50 bar"):
        panel = factor.compute(DAYS[10], DAYS[20])

    aaa = panel.sel(symbol="AAA")
    assert np.isfinite(aaa["spy_ret_mean_5"].values).all()
    assert np.isnan(aaa["spy_ret_mean_60"].values).all()


def test_a_pinned_subset_computes_only_those_features(stores):
    factor = _factor(stores, factor_names=("qqq_ret_std_20", "spy_ret"))
    panel = factor.compute(DAYS[60], DAYS[65])

    assert list(panel.data_vars) == ["qqq_ret_std_20", "spy_ret"]


def test_build_then_read_returns_the_computed_range(stores):
    factor = _factor(stores)
    factor.build(DAYS[60], DAYS[-1])
    stored = factor.read(DAYS[70], DAYS[80]).load()
    computed = factor.compute(DAYS[70], DAYS[80])

    xr.testing.assert_allclose(stored, computed)
    with pytest.raises(ValueError, match="does not contain"):
        factor.read(DAYS[50], DAYS[80])


def test_extend_appends_warm_bars(stores):
    factor = _factor(stores)
    factor.build(DAYS[60], DAYS[90]).extend(DAYS[-1])

    xr.testing.assert_allclose(
        factor.read(DAYS[60], DAYS[-1]).load(),
        factor.compute(DAYS[60], DAYS[-1]),
    )


def test_a_series_store_with_several_symbols_is_refused(stores):
    factor = _factor(stores, series={"stocks": StockDataset(stores["target"])})
    with pytest.raises(
        ValueError,
        match="series 'stocks' must hold one symbol, its StockDataset holds 6",
    ):
        factor.compute(DAYS[60], DAYS[-1])


# -- config round trip --------------------------------------------------------------------


def test_the_config_round_trips_through_the_class_path_loader(stores):
    factor = _factor(stores, kwargs={"amount_column": None})
    saved = json.loads(json.dumps(factor.get_config()))

    assert saved["name"] == "quantlab.factor.predefined.market.MarketFeatures"
    assert list(saved["series"]) == ["spy", "qqq"]
    assert saved["series"]["spy"]["zarr_file_path"] == stores["spy"][0].zarr_file_path

    rebuilt = load_factor_from_config(saved)
    assert rebuilt == factor
    assert isinstance(rebuilt.config.series["qqq"], StockDataset)
    xr.testing.assert_allclose(
        rebuilt.compute(DAYS[60], DAYS[70]), factor.compute(DAYS[60], DAYS[70])
    )


def test_a_copy_shares_no_series_dataset(stores):
    factor = _factor(stores)
    other = factor.copy()

    assert other == factor
    assert other.config.series["spy"] is not factor.config.series["spy"]
    assert other.config.dataset is not factor.config.dataset
