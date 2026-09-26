"""Tests for ``Factor.analyze`` and ``quantlab.analysis.factor_report``.

The panels are synthetic: the ``spot_kline_zarr`` store from ``conftest.py``
feeds ``Momentum`` and two test-local Polars factors. ``_ForwardReturn`` is a
label (its ``get_labels()`` is the next bar's close-to-close return) and
``_Oracle`` is a factor that is a strictly increasing function of that same
forward return, so it orders symbols exactly as the label does.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from quantlab.analysis.factor_report import FactorAnalyzer
from quantlab.backend import XrBackend
from quantlab.base.config import DatasetConfig, FactorConfig, PolarsFactorConfig
from quantlab.base.factor import FactorPolars
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.momentum import Momentum
from quantlab.utils.module import load_factor_from_config

EXPECTED_FILES = [
    "config.json",
    "ic.csv",
    "monthly_ic.csv",
    "quantile_returns.csv",
    "summary.csv",
    "summary.json",
    "turnover.csv",
]


def _forward_return() -> pl.Expr:
    """Next bar's close-to-close return per symbol (rows sorted by symbol, time)."""
    close = pl.col("Close")
    return close.shift(-1).over("symbol") / close - 1.0


class _ForwardReturn(FactorPolars):
    """One-bar forward return label, ``fwd_1``."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns(_forward_return().alias("fwd_1"))
            .select(["timestamp", "symbol", "fwd_1"])
        )

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        return data


class _Oracle(FactorPolars):
    """A factor that is the cube of the forward return: a perfect ranker."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((_forward_return() ** 3).alias("oracle"))
            .select(["timestamp", "symbol", "oracle"])
        )

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data


class _StaticLabel:
    """A duck-typed fret that hands back a fixed panel."""

    def __init__(self, panel: xr.Dataset):
        self.panel = panel

    def get_labels(self) -> xr.Dataset:
        return self.panel

    def get_config(self) -> dict:
        return {"name": "static"}


def _polars_config(dataset_config: DatasetConfig, tmp_path: Path, name: str, **kwargs):
    return PolarsFactorConfig(
        window=5,
        dataset=SpotKlineDataset(dataset_config),
        file_path=str(tmp_path / "factors" / f"{name}.zarr"),
        **kwargs,
    )


@pytest.fixture
def oracle_and_label(spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path):
    """A computed ``_Oracle`` factor and ``_ForwardReturn`` label, 20 symbols."""
    dataset_config = spot_kline_zarr(
        symbols=[f"S{i}USDT" for i in range(20)], periods=90
    )
    factor = _Oracle(_polars_config(dataset_config, tmp_path, "oracle")).cal()
    label = _ForwardReturn(_polars_config(dataset_config, tmp_path, "fwd")).cal()
    return factor, label


def test_a_perfect_predictor_has_ic_one_and_monotone_quantile_returns(oracle_and_label):
    factor, label = oracle_and_label

    result = factor.analyze(frets=[label])

    assert list(result.pairs) == ["oracle__fwd_1"]
    pair = result.pairs["oracle__fwd_1"]
    ic = pair.ic.dropna()
    # 90 bars; the last has no forward return.
    assert len(ic) == 89
    np.testing.assert_allclose(ic.to_numpy(), 1.0, atol=1e-12)
    assert pair.summary["ic_mean"] == pytest.approx(1.0)
    assert pair.summary["ic_positive_ratio"] == 1.0

    means = pair.mean_quantile_returns.to_numpy()
    assert len(means) == 5
    assert np.all(np.diff(means) > 0)
    # Per period too: every bucket beats the one below it.
    per_period = pair.quantile_returns.dropna().to_numpy()
    assert np.all(np.diff(per_period, axis=1) > 0)
    assert (pair.spread.dropna() > 0).all()
    assert pair.summary["cumulative_long_short"] > 0
    turnover = pair.turnover.dropna().to_numpy()
    assert ((turnover >= 0) & (turnover <= 1)).all()
    assert result.figures["oracle__fwd_1"] is not None


def test_frequency_mismatch_raises(spot_kline_zarr, tmp_path):
    factor = Momentum(_polars_config(spot_kline_zarr(), tmp_path, "mom", kwargs={"n": 5})).cal()
    daily = factor.get_features()["momentum_5"]
    weekly = daily.isel(timestamp=slice(None, None, 7)).rename("ret").to_dataset()

    with pytest.raises(ValueError, match=r"1 days.*7 days"):
        factor.analyze(frets=[_StaticLabel(weekly)])


def test_panels_are_inner_joined_before_computing(oracle_and_label):
    factor, label = oracle_and_label
    labels = label.get_labels()
    narrowed = labels.isel(timestamp=slice(10, 40), symbol=slice(0, 12))

    pair = factor.analyze(frets=[_StaticLabel(narrowed)]).pairs["oracle__fwd_1"]

    assert len(pair.ic) == 30
    assert pair.summary["n_symbols"] == 12
    assert pair.start == pd.Timestamp(labels["timestamp"].values[10])


def test_output_dir_writes_tables_figure_and_rebuildable_config(
    spot_kline_zarr, tmp_path
):
    dataset_config = spot_kline_zarr(symbols=[f"S{i}USDT" for i in range(10)])
    factor = Momentum(_polars_config(dataset_config, tmp_path, "mom", kwargs={"n": 5})).cal()
    label = _ForwardReturn(_polars_config(dataset_config, tmp_path, "fwd")).cal()
    out = tmp_path / "report" / "momentum"

    result = factor.analyze(frets=[label], output_dir=str(out))

    assert sorted(p.name for p in out.iterdir()) == sorted(
        EXPECTED_FILES + ["momentum_5__fwd_1.png"]
    )
    summary = json.loads((out / "summary.json").read_text())
    assert [row["factor"] for row in summary["pairs"]] == ["momentum_5"]
    assert summary["pairs"][0]["ic_mean"] == pytest.approx(
        result.pairs["momentum_5__fwd_1"].summary["ic_mean"]
    )
    assert (out / "momentum_5__fwd_1.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    ic = pd.read_csv(out / "ic.csv")
    assert list(ic.columns) == ["timestamp", "factor", "fret", "ic"]
    quantile_returns = pd.read_csv(out / "quantile_returns.csv")
    assert sorted(quantile_returns["quantile"].unique()) == [1, 2, 3, 4, 5]

    config = json.loads((out / "config.json").read_text())
    assert list(config) == ["factor", "frets"]
    rebuilt = load_factor_from_config(config["factor"])
    assert type(rebuilt) is Momentum
    assert list(rebuilt.get_factor_names()) == ["momentum_5"]
    assert config["frets"][0]["name"].endswith("_ForwardReturn")


def test_output_dir_none_writes_nothing(oracle_and_label, tmp_path):
    factor, label = oracle_and_label
    before = sorted(str(p) for p in tmp_path.rglob("*"))

    result = factor.analyze(frets=[label])

    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    figure = result.figures["oracle__fwd_1"]
    assert type(figure).__name__ == "Figure"
    assert "oracle" in figure.get_suptitle() and "fwd_1" in figure.get_suptitle()


def test_bad_arguments_raise(oracle_and_label):
    factor, label = oracle_and_label

    with pytest.raises(ValueError, match="at least one forward-return"):
        factor.analyze(frets=[])
    with pytest.raises(ValueError, match="not in _Oracle's panel"):
        factor.analyze(factor_names=["nope"], frets=[label])


def test_a_frozen_ranking_has_zero_turnover_and_unit_autocorrelation():
    timestamps = pd.date_range("2024-01-01", periods=6)
    symbols = [f"S{i}" for i in range(10)]
    coords = {"timestamp": timestamps, "symbol": symbols}
    factor = xr.DataArray(
        np.tile(np.arange(10.0), (6, 1)), coords=coords, dims=("timestamp", "symbol")
    )
    fret = xr.DataArray(
        np.random.default_rng(1).normal(size=(6, 10)),
        coords=coords,
        dims=("timestamp", "symbol"),
    )

    pair = FactorAnalyzer(quantiles=5).analyze_pair(factor, fret, "f", "r")

    assert pair.turnover.iloc[0].isna().all()
    assert (pair.turnover.iloc[1:] == 0.0).all().all()
    np.testing.assert_allclose(pair.rank_autocorrelation.iloc[1:], 1.0)
    # Bucket 1 holds the two lowest factor values, S0 and S1.
    expected = fret.isel(symbol=[0, 1]).mean("symbol").to_numpy()
    np.testing.assert_allclose(pair.quantile_returns[1].to_numpy(), expected)


def test_a_multi_bar_fret_compounds_the_per_bar_rate():
    timestamps = pd.date_range("2024-01-01", periods=3)
    coords = {"timestamp": timestamps, "symbol": ["A", "B"]}
    factor = xr.DataArray([[0.0, 1.0]] * 3, coords=coords, dims=("timestamp", "symbol"))
    fret = xr.DataArray([[0.0, 0.21]] * 3, coords=coords, dims=("timestamp", "symbol"))

    pair = FactorAnalyzer(quantiles=2).analyze_pair(factor, fret, "f", "r", horizon=2)

    # 21% over two bars is 10% per bar, compounded over three bars.
    assert pair.cumulative_quantile_returns[2].iloc[-1] == pytest.approx(1.1**3 - 1)


def test_importing_the_factor_layer_does_not_import_matplotlib():
    code = "import sys, quantlab.base.factor; print('matplotlib' in sys.modules)"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert output.strip() == "False"


def test_analyze_against_the_kunquant_return_label(tmp_path):
    """End to end with ``quantlab.label.fret.Return`` and its horizon."""
    from quantlab.dataset.stock import StockDataset
    from quantlab.label.fret import Return

    rng = np.random.default_rng(0)
    symbols = [f"T{i}" for i in range(8)]
    timestamps = pd.date_range("2024-01-01", periods=40)
    price = 50 + np.cumsum(rng.normal(size=(40, 8)), axis=0)
    store = tmp_path / "stock.zarr"
    XrBackend().to_internal(
        xr.Dataset(
            {"adjOpen": (["timestamp", "symbol"], price), "Close": (["timestamp", "symbol"], price)},
            coords={"timestamp": timestamps, "symbol": symbols},
        )
    ).write(str(store))

    def dataset():
        return StockDataset(
            DatasetConfig(
                raw_data_dir_path=str(tmp_path / "raw"),
                zarr_file_path=str(store),
                market="us_equity",
                frequency="1d",
            )
        )

    factor = Momentum(
        PolarsFactorConfig(
            window=5,
            dataset=dataset(),
            file_path=str(tmp_path / "mom.zarr"),
            kwargs={"n": 3},
        )
    ).cal()
    label = Return(
        FactorConfig(
            window=5,
            mode="batch",
            data_columns=["adjOpen"],
            kwargs={"n_forward_periods": 2},
            dataset=dataset(),
            file_path=str(tmp_path / "ret.zarr"),
            njobs=2,
        )
    ).cal()

    result = factor.analyze(frets=[label], quantiles=4)

    pair = result.pairs["momentum_3__ret_2"]
    assert pair.horizon == 2
    # Momentum needs 3 bars of history; the label loses its last 3 bars.
    assert pair.ic.notna().sum() == 40 - 3 - 3
    assert set(result.summary_table()["fret"]) == {"ret_2"}
