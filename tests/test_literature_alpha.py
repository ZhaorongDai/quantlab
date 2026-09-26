"""Contract, graph-pruning and numerical tests for ``LiteratureAlpha``."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from KunQuant.Op import Input

from quantlab.base.config import DatasetConfig, FactorConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.literature_alpha import (
    LiteratureAlpha,
    LiteratureAlphaParameters,
)


CORE_NAMES = LiteratureAlpha._CORE_FACTOR_NAMES


def _synthetic_panel(periods: int = 90, symbols: int = 8, seed: int = 9):
    """Build a complete PIT-shaped panel and its common FF3 series."""

    rng = np.random.default_rng(seed)
    timestamps = pd.bdate_range("2021-01-04", periods=periods)
    symbol_axis = np.array([f"S{i}" for i in range(symbols)])
    factors = rng.normal(0.0, [0.012, 0.007, 0.008], size=(periods, 3))
    risk_free = np.full(periods, 0.00005)
    betas = rng.normal([1.0, 0.0, 0.0], [0.15, 0.15, 0.15], size=(symbols, 3))
    noise = rng.normal(0.0002, 0.009, size=(periods, symbols))
    returns = risk_free[:, None] + factors @ betas.T + noise
    adjusted_close = 100.0 * np.cumprod(1.0 + returns, axis=0)
    raw_close = adjusted_close * np.linspace(0.85, 1.15, symbols)
    volume = rng.uniform(1.0e6, 5.0e6, size=(periods, symbols))

    def common(values):
        return np.broadcast_to(values[:, None], (periods, symbols)).copy()

    gross_profit = np.broadcast_to(
        np.linspace(30.0, 65.0, symbols), (periods, symbols)
    ).copy()
    total_assets = np.broadcast_to(
        np.linspace(150.0, 260.0, symbols), (periods, symbols)
    ).copy()
    prior_assets = total_assets / np.linspace(0.92, 1.18, symbols)
    consensus = np.broadcast_to(
        np.linspace(1.2, 2.4, symbols), (periods, symbols)
    ).copy()
    actual = consensus + np.linspace(-0.2, 0.25, symbols)
    scale_price = np.broadcast_to(
        np.linspace(60.0, 130.0, symbols), (periods, symbols)
    ).copy()

    panel = xr.Dataset(
        {
            "ret": (("timestamp", "symbol"), returns),
            "adjClose": (("timestamp", "symbol"), adjusted_close),
            "close": (("timestamp", "symbol"), raw_close),
            "volume": (("timestamp", "symbol"), volume),
            "risk_free": (("timestamp", "symbol"), common(risk_free)),
            "mkt_rf": (("timestamp", "symbol"), common(factors[:, 0])),
            "smb": (("timestamp", "symbol"), common(factors[:, 1])),
            "hml": (("timestamp", "symbol"), common(factors[:, 2])),
            "gross_profit": (("timestamp", "symbol"), gross_profit),
            "total_assets": (("timestamp", "symbol"), total_assets),
            "prior_year_total_assets": (
                ("timestamp", "symbol"),
                prior_assets,
            ),
            "eps_actual_event": (("timestamp", "symbol"), actual),
            "eps_consensus_event": (("timestamp", "symbol"), consensus),
            "eps_scale_price_event": (("timestamp", "symbol"), scale_price),
        },
        coords={"timestamp": timestamps, "symbol": symbol_axis},
    )
    return panel, factors, risk_free


def _dataset(tmp_path: Path, panel: xr.Dataset, name: str = "market") -> StockDataset:
    """Persist ``panel`` and return a stock dataset over it."""

    store = tmp_path / f"{name}.zarr"
    panel.to_zarr(store, mode="w")
    return StockDataset(
        DatasetConfig(
            zarr_file_path=str(store),
            raw_data_dir_path=str(tmp_path / "raw"),
            market="us_equity",
            frequency="1d",
            start_date=str(panel.timestamp.values[0])[:10],
            end_date=str(panel.timestamp.values[-1])[:10],
        )
    )


def _config(dataset: StockDataset, tmp_path: Path, **overrides) -> FactorConfig:
    """Build a complete daily ``LiteratureAlpha`` configuration."""

    params = LiteratureAlphaParameters(
        high_52week_window=60,
        short_reversal_window=21,
        max_return_window=21,
        idio_vol_window=21,
        amihud_window=21,
    )
    values = {
        "window": 0,
        "dataset": dataset,
        "start_date": dataset.config.start_date,
        "end_date": dataset.config.end_date,
        "mode": "batch",
        "data_columns": params.required_panel_columns(CORE_NAMES),
        "factor_names": CORE_NAMES,
        "file_path": str(tmp_path / "literature_alpha.zarr"),
        "njobs": 2,
        "kwargs": {
            "high_52week_window": 60,
            "short_reversal_window": 21,
            "max_return_window": 21,
            "idio_vol_window": 21,
            "amihud_window": 21,
        },
    }
    values.update(overrides)
    return FactorConfig(**values)


def test_default_names_cover_eight_raw_and_rank_pairs(tmp_path: Path) -> None:
    """The default public surface is exactly eight raw/rank pairs."""

    panel, _, _ = _synthetic_panel(periods=30)
    dataset = _dataset(tmp_path, panel)
    params = LiteratureAlphaParameters(
        high_52week_window=10,
        short_reversal_window=5,
        max_return_window=5,
        idio_vol_window=10,
        amihud_window=5,
    )
    factor = LiteratureAlpha(
        FactorConfig(
            window=0,
            dataset=dataset,
            mode="batch",
            data_columns=params.required_panel_columns(CORE_NAMES),
            factor_names=None,
            kwargs={
                "high_52week_window": 10,
                "short_reversal_window": 5,
                "max_return_window": 5,
                "idio_vol_window": 10,
                "amihud_window": 5,
            },
        )
    )
    assert factor.get_factor_names() == CORE_NAMES
    assert len(CORE_NAMES) == 16


def test_selected_output_prunes_inputs_and_accepts_column_alias(
    tmp_path: Path,
) -> None:
    """A one-factor request needs only its aliased reachable input."""

    panel, _, _ = _synthetic_panel(periods=30)
    panel = panel.rename({"adjClose": "split_price"})[["split_price"]]
    dataset = _dataset(tmp_path, panel)
    factor = LiteratureAlpha(
        FactorConfig(
            window=0,
            dataset=dataset,
            mode="batch",
            data_columns=("split_price",),
            factor_names=("high_52week_proximity_raw",),
            kwargs={
                "split_adjusted_close_column": "split_price",
                "high_52week_window": 10,
            },
        )
    )

    inputs = {
        op.attrs["name"]
        for op in factor._get_factor_func().ops
        if isinstance(op, Input)
    }
    assert inputs == {"split_price"}
    assert factor.get_factor_names() == ("high_52week_proximity_raw",)


def test_unknown_output_kwargs_and_wrong_columns_fail_early(tmp_path: Path) -> None:
    """Configuration errors are reported before compilation."""

    panel, _, _ = _synthetic_panel(periods=30)
    dataset = _dataset(tmp_path, panel)
    with pytest.raises(ValueError, match="unknown config.kwargs"):
        LiteratureAlpha(
            FactorConfig(
                window=0,
                dataset=dataset,
                mode="batch",
                data_columns=("adjClose",),
                factor_names=("high_52week_proximity_raw",),
                kwargs={"unknown": 1},
            )
        )
    with pytest.raises(ValueError, match="unknown factor_names"):
        LiteratureAlpha(
            FactorConfig(
                window=0,
                dataset=dataset,
                mode="batch",
                data_columns=(),
                factor_names=("not_a_factor",),
            )
        )
    with pytest.raises(ValueError, match="data_columns must exactly match"):
        LiteratureAlpha(
            FactorConfig(
                window=0,
                dataset=dataset,
                mode="batch",
                data_columns=("ret",),
                factor_names=("high_52week_proximity_raw",),
            )
        )


def test_all_eight_factors_match_direct_formulas(tmp_path: Path) -> None:
    """One compiled graph returns every formula and valid cross-sectional ranks."""

    panel, factors, risk_free = _synthetic_panel()
    dataset = _dataset(tmp_path, panel)
    output = LiteratureAlpha(_config(dataset, tmp_path)).cal().get_features()

    assert tuple(output.data_vars) == CORE_NAMES
    assert dict(output.sizes) == {"timestamp": 90, "symbol": 8}
    for name in CORE_NAMES:
        assert np.isfinite(output[name].to_numpy()).sum() > 0
    for name in (name for name in CORE_NAMES if name.endswith("_rank")):
        values = output[name].to_numpy()
        finite = values[np.isfinite(values)]
        assert np.all((finite >= 0.0) & (finite <= 1.0))

    i = 3
    ret = panel["ret"].isel(symbol=i).to_numpy()
    adjusted = panel["adjClose"].isel(symbol=i).to_numpy()
    close = panel["close"].isel(symbol=i).to_numpy()
    volume = panel["volume"].isel(symbol=i).to_numpy()
    last = output.isel(timestamp=-1, symbol=i)

    expected = {
        "high_52week_proximity_raw": adjusted[-1] / adjusted[-60:].max(),
        "short_reversal_raw": -(np.prod(1.0 + ret[-21:]) - 1.0),
        "low_max_raw": -ret[-21:].max(),
        "amihud_illiquidity_raw": np.log(
            1.0e6 * np.mean(np.abs(ret[-21:]) / (close[-21:] * volume[-21:]))
            + 1.0e-12
        ),
        "gross_profitability_raw": float(
            panel["gross_profit"].isel(timestamp=-1, symbol=i)
            / panel["total_assets"].isel(timestamp=-1, symbol=i)
        ),
        "conservative_asset_growth_raw": -float(
            panel["total_assets"].isel(timestamp=-1, symbol=i)
            / panel["prior_year_total_assets"].isel(timestamp=-1, symbol=i)
            - 1.0
        ),
        "standardized_unexpected_earnings_raw": float(
            (
                panel["eps_actual_event"].isel(timestamp=-1, symbol=i)
                - panel["eps_consensus_event"].isel(timestamp=-1, symbol=i)
            )
            / panel["eps_scale_price_event"].isel(timestamp=-1, symbol=i)
        ),
    }

    x = factors[-21:]
    y = ret[-21:] - risk_free[-21:]
    factor_cov = np.cov(x, rowvar=False, ddof=1)
    cross_cov = (
        (x - x.mean(axis=0)) * (y - y.mean())[:, None]
    ).sum(axis=0) / 20
    beta = np.linalg.solve(factor_cov + 1.0e-10 * np.eye(3), cross_cov)
    residual = (y - y.mean()) - (x - x.mean(axis=0)) @ beta
    expected["low_idiosyncratic_volatility_raw"] = -residual.std(ddof=1)

    for name, value in expected.items():
        # KunQuant evaluates transcendental operators in float32; its
        # exp(sum(log1p(r))) reversal therefore needs a slightly wider
        # relative tolerance than the algebraic ratios and rolling moments.
        assert float(last[name]) == pytest.approx(value, rel=2e-3, abs=2e-6)


def test_fama_french_csv_reproduces_panel_ivol(tmp_path: Path) -> None:
    """CSV-injected FF3 values produce the panel-input IVOL values."""

    panel, factors, risk_free = _synthetic_panel(periods=50)
    full = _dataset(tmp_path, panel, "full")
    ret_only = _dataset(tmp_path, panel[["ret"]], "ret_only")
    csv = tmp_path / "ff3.csv"
    pd.DataFrame(
        {
            "mkt_rf": factors[:, 0],
            "smb": factors[:, 1],
            "hml": factors[:, 2],
            "risk_free": risk_free,
        },
        index=pd.Index(panel.timestamp.values, name="date"),
    ).to_csv(csv)
    names = (
        "low_idiosyncratic_volatility_raw",
        "low_idiosyncratic_volatility_rank",
    )
    panel_factor = LiteratureAlpha(
        FactorConfig(
            window=0,
            dataset=full,
            mode="batch",
            data_columns=("ret", "risk_free", "mkt_rf", "smb", "hml"),
            factor_names=names,
            kwargs={"idio_vol_window": 21},
            njobs=2,
        )
    ).cal().get_features()
    csv_factor = LiteratureAlpha(
        FactorConfig(
            window=0,
            dataset=ret_only,
            mode="batch",
            data_columns=("ret",),
            factor_names=names,
            kwargs={"idio_vol_window": 21, "fama_french_csv": str(csv)},
            njobs=2,
        )
    ).cal().get_features()

    for name in names:
        np.testing.assert_allclose(
            csv_factor[name], panel_factor[name], rtol=1e-5, atol=1e-6,
            equal_nan=True,
        )


def test_csv_mode_refuses_stream(tmp_path: Path) -> None:
    """A static CSV cannot supply bar-by-bar stream inputs."""

    panel, factors, risk_free = _synthetic_panel(periods=30)
    dataset = _dataset(tmp_path, panel[["ret"]])
    csv = tmp_path / "ff3.csv"
    pd.DataFrame(
        {
            "mkt_rf": factors[:, 0],
            "smb": factors[:, 1],
            "hml": factors[:, 2],
            "risk_free": risk_free,
        },
        index=pd.Index(panel.timestamp.values, name="date"),
    ).to_csv(csv)
    factor = LiteratureAlpha(
        FactorConfig(
            window=0,
            dataset=dataset,
            mode="stream",
            data_columns=("ret",),
            factor_names=("low_idiosyncratic_volatility_raw",),
            kwargs={"fama_french_csv": str(csv)},
            njobs=2,
        )
    )
    with pytest.raises(ValueError, match="put Fama-French values on the stream"):
        factor.cal_stream(
            {"ret": np.zeros(8, dtype=np.float32)},
            0,
            list(panel.symbol.values),
        )
