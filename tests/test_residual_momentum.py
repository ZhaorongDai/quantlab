"""Contract and execution tests for the FF3 residual-momentum factor."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import DatasetConfig, FactorConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.residual_momentum import ResidualMomentumFF3


def _monthly_dataset(tmp_path: Path, *, periods: int = 60, symbols: int = 7):
    """Create a small monthly panel with common FF3 series broadcast by symbol."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2000-01-31", periods=periods, freq="ME")
    symbol_axis = np.array([f"S{i}" for i in range(symbols)])
    factors = rng.normal(0.0, 0.025, size=(periods, 3))
    risk_free = np.full(periods, 0.002)
    betas = rng.normal(0.8, 0.2, size=(symbols, 3))
    noise = rng.normal(0.0, 0.015, size=(periods, symbols))
    stock_return = risk_free[:, None] + factors @ betas.T + noise

    def broadcast(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values[:, None], (periods, symbols)).copy()

    panel = xr.Dataset(
        {
            "stock_return": (("timestamp", "symbol"), stock_return),
            "risk_free": (("timestamp", "symbol"), broadcast(risk_free)),
            "mkt_rf": (("timestamp", "symbol"), broadcast(factors[:, 0])),
            "smb": (("timestamp", "symbol"), broadcast(factors[:, 1])),
            "hml": (("timestamp", "symbol"), broadcast(factors[:, 2])),
        },
        coords={"timestamp": timestamps, "symbol": symbol_axis},
    )
    store = tmp_path / "monthly.zarr"
    panel.to_zarr(store, mode="w")
    config = DatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(tmp_path / "raw"),
        market="us_equity",
        frequency="1d",
        start_date=str(timestamps[0].date()),
        end_date=str(timestamps[-1].date()),
    )
    return StockDataset(config), timestamps


def _factor_config(dataset, tmp_path: Path, **overrides) -> FactorConfig:
    values = {
        "window": 0,
        "dataset": dataset,
        "start_date": dataset.config.start_date,
        "end_date": dataset.config.end_date,
        "mode": "batch",
        "data_columns": (
            "stock_return",
            "risk_free",
            "mkt_rf",
            "smb",
            "hml",
        ),
        "factor_names": ("resmom_raw", "resmom_rank"),
        "file_path": str(tmp_path / "resmom.zarr"),
        "njobs": 2,
    }
    values.update(overrides)
    return FactorConfig(**values)


def test_residual_momentum_names_and_crsp_return_alias(tmp_path: Path) -> None:
    dataset, _ = _monthly_dataset(tmp_path)
    config = _factor_config(
        dataset,
        tmp_path,
        data_columns=("ret", "rf", "market", "size", "value"),
        factor_names=None,
        kwargs={
            "return_column": "ret",
            "risk_free_column": "rf",
            "market_column": "market",
            "smb_column": "size",
            "hml_column": "value",
            "emit_diagnostics": False,
        },
    )

    factor = ResidualMomentumFF3(config)

    assert factor.get_factor_names() == ("resmom_raw", "resmom_rank")
    assert factor.config.dataset.config.start_date == "1997-01-31"


def test_residual_momentum_rejects_mismatched_data_columns(tmp_path: Path) -> None:
    dataset, _ = _monthly_dataset(tmp_path)
    config = _factor_config(
        dataset,
        tmp_path,
        data_columns=("stock_return", "risk_free", "mkt_rf", "smb"),
    )

    with pytest.raises(ValueError, match="data_columns must exactly match"):
        ResidualMomentumFF3(config)


def test_residual_momentum_batch_calculates_unaligned_symbol_count(
    tmp_path: Path,
) -> None:
    dataset, timestamps = _monthly_dataset(tmp_path, symbols=7)
    factor = ResidualMomentumFF3(_factor_config(dataset, tmp_path))

    result = factor.cal().get_features()

    assert dict(result.sizes) == {"timestamp": len(timestamps), "symbol": 7}
    assert tuple(result.data_vars) == ("resmom_raw", "resmom_rank")
    assert np.isfinite(result["resmom_raw"].to_numpy()).sum() > 0
    finite_ranks = result["resmom_rank"].to_numpy()
    finite_ranks = finite_ranks[np.isfinite(finite_ranks)]
    assert finite_ranks.size > 0
    assert np.all((finite_ranks >= 0.0) & (finite_ranks <= 1.0))
