"""Contract and execution tests for the FF3 residual-momentum factor.

Two fixtures: a monthly panel that carries the five inputs itself (the
classic 36 / 12 / 1 design), and a daily panel with only ``ret`` plus a
Fama-French CSV (the CRSP setup). The CSV path is proved against the panel
path: the same series fed both ways give the same numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.base.config import DatasetConfig, FactorConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.residual_momentum import (
    ResidualMomentumFF3,
    ResidualMomentumParameters,
    compound_onto_bars,
    read_fama_french,
)

MONTHLY = {"regression_window": 36, "formation_lookback": 12, "skip_recent": 1}
DAILY = {"regression_window": 60, "formation_lookback": 20, "skip_recent": 5}


def _series(periods: int, symbols: int, seed: int = 42):
    """Common factor series, a flat risk-free rate and factor-driven stock returns."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.025, size=(periods, 3))
    risk_free = np.full(periods, 0.002)
    betas = rng.normal(0.8, 0.2, size=(symbols, 3))
    noise = rng.normal(0.0, 0.015, size=(periods, symbols))
    stock_return = risk_free[:, None] + factors @ betas.T + noise
    return factors, risk_free, stock_return


def _dataset(tmp_path: Path, name: str, panel: xr.Dataset) -> StockDataset:
    store = tmp_path / f"{name}.zarr"
    panel.to_zarr(store, mode="w")
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(tmp_path / "raw"),
        market="us_equity",
        frequency="1d",
        start_date=str(timestamps[0].date()),
        end_date=str(timestamps[-1].date()),
    ))


def _broadcast_panel(timestamps, symbol_axis, stock_return, risk_free, factors, return_name):
    periods, symbols = stock_return.shape

    def broadcast(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values[:, None], (periods, symbols)).copy()

    return xr.Dataset(
        {
            return_name: (("timestamp", "symbol"), stock_return),
            "risk_free": (("timestamp", "symbol"), broadcast(risk_free)),
            "mkt_rf": (("timestamp", "symbol"), broadcast(factors[:, 0])),
            "smb": (("timestamp", "symbol"), broadcast(factors[:, 1])),
            "hml": (("timestamp", "symbol"), broadcast(factors[:, 2])),
        },
        coords={"timestamp": timestamps, "symbol": symbol_axis},
    )


def _monthly_dataset(tmp_path: Path, *, periods: int = 60, symbols: int = 7):
    """A monthly panel carrying the five inputs, broadcast by symbol."""
    timestamps = pd.date_range("2000-01-31", periods=periods, freq="ME")
    symbol_axis = np.array([f"S{i}" for i in range(symbols)])
    factors, risk_free, stock_return = _series(periods, symbols)
    panel = _broadcast_panel(
        timestamps, symbol_axis, stock_return, risk_free, factors, "stock_return"
    )
    return _dataset(tmp_path, "monthly", panel), timestamps


def _daily_fixture(tmp_path: Path, *, periods: int = 120, symbols: int = 5):
    """``(ret-only dataset, five-input dataset, csv path)`` over the same daily series."""
    timestamps = pd.bdate_range("2020-01-01", periods=periods)
    symbol_axis = np.array([f"S{i}" for i in range(symbols)])
    factors, risk_free, stock_return = _series(periods, symbols, seed=7)
    full = _broadcast_panel(
        timestamps, symbol_axis, stock_return, risk_free, factors, "ret"
    )
    ret_only = full[["ret"]]
    csv = tmp_path / "ff3_daily.csv"
    pd.DataFrame(
        {"mkt_rf": factors[:, 0], "smb": factors[:, 1], "hml": factors[:, 2],
         "risk_free": risk_free},
        index=pd.Index(timestamps, name="date"),
    ).to_csv(csv)
    return _dataset(tmp_path, "daily", ret_only), _dataset(tmp_path, "full", full), csv


def _factor_config(dataset, tmp_path: Path, **overrides) -> FactorConfig:
    values = {
        "window": 0,
        "dataset": dataset,
        "start_date": dataset.config.start_date,
        "end_date": dataset.config.end_date,
        "mode": "batch",
        "data_columns": ("stock_return", "risk_free", "mkt_rf", "smb", "hml"),
        "factor_names": ("resmom_raw", "resmom_rank"),
        "file_path": str(tmp_path / "resmom.zarr"),
        "njobs": 2,
        "kwargs": {"return_column": "stock_return", **MONTHLY},
    }
    values.update(overrides)
    return FactorConfig(**values)


# ---------------------------------------------------------------------------
# parameters and config contract
# ---------------------------------------------------------------------------


def test_defaults_are_the_daily_blitz_design_and_panel_columns_follow_the_csv():
    params = ResidualMomentumParameters()
    assert (params.regression_window, params.formation_lookback, params.skip_recent) == (
        756, 252, 21
    )
    assert params.input_columns == ("ret", "risk_free", "mkt_rf", "smb", "hml")
    assert params.panel_columns == params.input_columns
    assert ResidualMomentumParameters(fama_french_csv="ff3.csv").panel_columns == ("ret",)


def test_residual_momentum_names_and_column_aliases(tmp_path: Path) -> None:
    dataset, _ = _monthly_dataset(tmp_path)
    config = _factor_config(
        dataset,
        tmp_path,
        data_columns=("ret", "rf", "market", "size", "value"),
        factor_names=None,
        kwargs={
            **MONTHLY,
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
    # The warm-up follows the base rule (window calendar days), nothing else.
    assert factor.config.dataset.config.start_date == "2000-01-31"


def test_residual_momentum_rejects_mismatched_data_columns(tmp_path: Path) -> None:
    dataset, _ = _monthly_dataset(tmp_path)
    config = _factor_config(
        dataset,
        tmp_path,
        data_columns=("stock_return", "risk_free", "mkt_rf", "smb"),
    )

    with pytest.raises(ValueError, match="data_columns must exactly match"):
        ResidualMomentumFF3(config)


def test_csv_mode_wants_only_the_return_column(tmp_path: Path) -> None:
    ret_only, _, csv = _daily_fixture(tmp_path)
    kwargs = {"fama_french_csv": str(csv), **DAILY}

    with pytest.raises(ValueError, match=r"must exactly match \('ret',\)"):
        ResidualMomentumFF3(_factor_config(
            ret_only, tmp_path,
            data_columns=("ret", "risk_free", "mkt_rf", "smb", "hml"), kwargs=kwargs,
        ))
    factor = ResidualMomentumFF3(_factor_config(
        ret_only, tmp_path, data_columns=("ret",), kwargs=kwargs,
    ))
    assert factor.get_factor_names() == ("resmom_raw", "resmom_rank")


def test_csv_mode_refuses_stream(tmp_path: Path) -> None:
    ret_only, _, csv = _daily_fixture(tmp_path)
    factor = ResidualMomentumFF3(_factor_config(
        ret_only, tmp_path, data_columns=("ret",),
        kwargs={"fama_french_csv": str(csv), **DAILY},
    ))
    with pytest.raises(ValueError, match="stream mode reads every input from the panel"):
        factor.cal_stream({"ret": np.zeros(5, dtype=np.float32)}, 0, list("ABCDE"))


# ---------------------------------------------------------------------------
# the Fama-French CSV helpers
# ---------------------------------------------------------------------------


def test_read_fama_french_checks_columns_and_dates(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    pd.DataFrame({
        "date": ["2024-01-03", "2024-01-02"], "mkt_rf": [0.01, 0.02], "smb": [0, 0],
        "hml": [0, 0], "risk_free": [0.0001, 0.0001], "extra": [1, 2],
    }).to_csv(good, index=False)
    table = read_fama_french(good)
    assert list(table.columns) == ["mkt_rf", "smb", "hml", "risk_free"]
    assert table.index.is_monotonic_increasing and table.index.name == "date"

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-02"], "mkt_rf": [0.01]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing"):
        read_fama_french(bad)

    dup = tmp_path / "dup.csv"
    pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02"], "mkt_rf": [0.01, 0.02], "smb": [0, 0],
        "hml": [0, 0], "risk_free": [0, 0],
    }).to_csv(dup, index=False)
    with pytest.raises(ValueError, match="repeated dates"):
        read_fama_french(dup)


def test_compound_onto_bars_matches_daily_rows_and_compounds_coarser_bars() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    table = pd.DataFrame({"x": [0.01, 0.02, -0.01, 0.03]}, index=dates)

    same_days = compound_onto_bars(table, dates.values)
    np.testing.assert_allclose(same_days["x"].to_numpy(), table["x"].to_numpy())

    weekly = compound_onto_bars(table, pd.to_datetime(["2024-01-02", "2024-01-05"]).values)
    assert weekly["x"].iloc[0] == pytest.approx(0.01)
    assert weekly["x"].iloc[1] == pytest.approx(1.02 * 0.99 * 1.03 - 1.0)

    # A bar the table does not reach, and a bar before its first row, are NaN.
    outside = compound_onto_bars(
        table, pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-08"]).values
    )
    assert np.isnan(outside["x"].iloc[0]) and np.isnan(outside["x"].iloc[2])
    assert outside["x"].iloc[1] == pytest.approx(1.01 * 1.02 * 0.99 * 1.03 - 1.0)

    with pytest.raises(ValueError, match="strictly ascending"):
        compound_onto_bars(table, pd.to_datetime(["2024-01-03", "2024-01-02"]).values)


def test_csv_inputs_are_broadcast_float32_and_uncovered_bars_warn(tmp_path: Path) -> None:
    ret_only, _, csv = _daily_fixture(tmp_path, periods=30)
    factor = ResidualMomentumFF3(_factor_config(
        ret_only, tmp_path, data_columns=("ret",),
        kwargs={"fama_french_csv": str(csv), "regression_window": 10,
                "formation_lookback": 6, "skip_recent": 1},
    ))
    timestamps = pd.bdate_range("2020-01-01", periods=30).values
    # One bar past the CSV's last date.
    later = np.append(timestamps, timestamps[-1] + np.timedelta64(1, "D"))
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        inputs = factor._fama_french_inputs(later, 5)
    finally:
        logger.remove(sink)

    assert set(inputs) == {"mkt_rf", "smb", "hml", "risk_free"}
    for array in inputs.values():
        assert array.shape == (31, 5) and array.dtype == np.float32
        assert array.flags.c_contiguous
        assert np.isfinite(array[:30]).all() and np.isnan(array[30]).all()
        # Every symbol sees the same series.
        np.testing.assert_array_equal(array[:, 0], array[:, 4])
    assert any("1 of 31 bars have no row" in message for message in messages)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


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


def test_regression_diagnostics_match_least_squares_by_hand(tmp_path: Path) -> None:
    """The closed-form alpha and betas at the last bar are the OLS fit of that window."""
    dataset, timestamps = _monthly_dataset(tmp_path, periods=60, symbols=4)
    factor = ResidualMomentumFF3(_factor_config(
        dataset, tmp_path,
        factor_names=("resmom_raw", "alpha", "beta_mkt", "beta_smb", "beta_hml",
                      "residual_sum", "residual_volatility"),
    ))
    out = factor.cal().get_features()
    panel = dataset.read().get_xarray_dataset()

    window, lookback, skip = 36, 12, 1
    for symbol in panel["symbol"].values:
        y = (panel["stock_return"] - panel["risk_free"]).sel(symbol=symbol).to_numpy()
        x = np.column_stack([
            panel[name].sel(symbol=symbol).to_numpy() for name in ("mkt_rf", "smb", "hml")
        ])
        design = np.column_stack([np.ones(window), x[-window:]])
        alpha, *betas = np.linalg.lstsq(design, y[-window:], rcond=None)[0]

        last = out.sel(symbol=symbol).isel(timestamp=-1)
        assert float(last["alpha"]) == pytest.approx(alpha, abs=2e-4)
        for name, beta in zip(("beta_mkt", "beta_smb", "beta_hml"), betas):
            assert float(last[name]) == pytest.approx(beta, rel=2e-3, abs=2e-3)

        # Formation period: the 11 bars ending one bar before the signal. The
        # volatility is the sample standard deviation (KunQuant's WindowedVar
        # divides by n - 1).
        formation = slice(-lookback, -skip)
        residuals = y[formation] - alpha - x[formation] @ np.asarray(betas)
        volatility = residuals.std(ddof=1)
        assert float(last["residual_sum"]) == pytest.approx(residuals.sum(), rel=5e-3, abs=1e-4)
        assert float(last["residual_volatility"]) == pytest.approx(volatility, rel=5e-3, abs=1e-4)
        assert float(last["resmom_raw"]) == pytest.approx(
            residuals.sum() / volatility, rel=5e-3, abs=1e-3
        )


def test_csv_path_reproduces_the_panel_path(tmp_path: Path) -> None:
    """Feeding the series from the CSV gives the numbers of feeding them as panel variables."""
    ret_only, full, csv = _daily_fixture(tmp_path)

    from_csv = ResidualMomentumFF3(_factor_config(
        ret_only, tmp_path, data_columns=("ret",),
        file_path=str(tmp_path / "from_csv.zarr"),
        kwargs={"fama_french_csv": str(csv), **DAILY},
    )).cal().get_features()
    from_panel = ResidualMomentumFF3(_factor_config(
        full, tmp_path, data_columns=("ret", "risk_free", "mkt_rf", "smb", "hml"),
        file_path=str(tmp_path / "from_panel.zarr"),
        kwargs={"return_column": "ret", **DAILY},
    )).cal().get_features()

    assert dict(from_csv.sizes) == {"timestamp": 120, "symbol": 5}
    for name in ("resmom_raw", "resmom_rank"):
        a, b = from_csv[name].to_numpy(), from_panel[name].to_numpy()
        assert np.isfinite(a).sum() > 0
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True)
