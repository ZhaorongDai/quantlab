"""Shared synthetic building blocks for the phase 03.7 backtester tests.

Plain helpers and classes, not pytest fixtures, imported as
`tests.backtest_fixtures`. Everything is synthetic, CPU-only and offline.
Configs are constructed directly, never through the factories in
`quantlab/config/__init__.py` (D-32).
"""

import dataclasses
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.base.config import DatasetConfig, MLConfig, PolarsFactorConfig
from quantlab.base.factor import FactorPolars
from quantlab.base.model import MLModel
from quantlab.dataset.stock import StockDataset

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]

ADJUSTED_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
RAW_COLUMNS = ("open", "high", "low", "close", "volume")


def write_price_store(
    root: Path,
    *,
    symbols: list[str] = SYMBOLS,
    n_bars: int = 60,
    start: str = "2024-01-01",
    seed: int = 0,
    delist_at: dict[str, int] | None = None,
    list_at: dict[str, int] | None = None,
) -> DatasetConfig:
    """Write a Tiingo-shaped business-day Zarr store and return its config.

    Prices are a seeded strictly-positive random walk. `adjOpen` is the
    previous `adjClose` times an independent noise term, so an open-vs-close
    mix-up changes results. The raw lowercase group is the adjusted group
    scaled by 1.7, so a raw-vs-adjusted mix-up changes results too.
    `delist_at={symbol: bar}` sets every variable NaN from that bar on;
    `list_at={symbol: bar}` sets every variable NaN before that bar.

    The store is written BEFORE the config is built, as in
    `tests/conftest.py:stock_zarr` -- dataset construction reads it.
    """
    root = Path(root)
    symbols = list(symbols)
    timestamps = pd.bdate_range(start, periods=n_bars)
    rng = np.random.default_rng(seed)
    n_symbols = len(symbols)

    close = 50.0 * np.exp(
        np.cumsum(rng.normal(0.0, 0.02, size=(n_bars, n_symbols)), axis=0)
    )
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, 0.01, size=(n_bars, n_symbols)))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.uniform(1e5, 1e6, size=(n_bars, n_symbols))

    adjusted = {
        "adjOpen": open_,
        "adjHigh": high,
        "adjLow": low,
        "adjClose": close,
        "adjVolume": volume,
    }
    variables = dict(adjusted)
    for raw_name, adj_name in zip(RAW_COLUMNS, ADJUSTED_COLUMNS):
        variables[raw_name] = adjusted[adj_name] * 1.7

    for name in variables:
        variables[name] = np.array(variables[name], dtype=np.float64)
        for symbol, bar in (delist_at or {}).items():
            variables[name][bar:, symbols.index(symbol)] = np.nan
        for symbol, bar in (list_at or {}).items():
            variables[name][:bar, symbols.index(symbol)] = np.nan

    dataset = xr.Dataset(
        {name: (["timestamp", "symbol"], values) for name, values in variables.items()},
        coords={"timestamp": timestamps, "symbol": symbols},
    )

    stock_dir = root / "stock"
    stock_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = stock_dir / "stock.zarr"
    dataset.to_zarr(zarr_path, mode="w")

    return DatasetConfig(
        raw_data_dir_path=str(stock_dir / "raw"),
        zarr_file_path=str(zarr_path),
        market="us_equity",
        frequency="1d",
    )


class PastReturnFactor(FactorPolars):
    """`past_ret_{n}` = adjClose / adjClose n bars earlier (per symbol) - 1."""

    @property
    def n(self) -> int:
        return int((self.config.kwargs or {})["n"])

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        name = f"past_ret_{self.n}"
        close = pl.col("adjClose")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((close / close.shift(self.n).over("symbol") - 1.0).alias(name))
            .select(["timestamp", "symbol", name])
        )

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{type(self).__name__} is a feature, not a label")


class ForwardReturnLabel(FactorPolars):
    """`fwd_ret_{n}` = adjClose n bars later (per symbol) / adjClose - 1."""

    @property
    def n(self) -> int:
        return int((self.config.kwargs or {})["n_forward_periods"])

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        name = f"fwd_ret_{self.n}"
        close = pl.col("adjClose")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((close.shift(-self.n).over("symbol") / close - 1.0).alias(name))
            .select(["timestamp", "symbol", name])
        )

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        return data

    def _get_features(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{type(self).__name__} is a label, not a feature")


class FirstFeatureHead(MLModel):
    """Deterministic ML head: every label's prediction is feature 0."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def make_stock_dataset(dataset_config: DatasetConfig) -> StockDataset:
    """A fresh `StockDataset` over a COPY of the config, so no two datasets
    alias one mutable config object."""
    return StockDataset(dataclasses.replace(dataset_config))


def make_model(
    root: Path,
    dataset_config: DatasetConfig,
    *,
    n: int = 1,
    window: int = 5,
    n_forward_periods: int = 1,
    start_date: str,
    end_date: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> FirstFeatureHead:
    """A `FirstFeatureHead` over one `PastReturnFactor` and one `ForwardReturnLabel`."""
    factor = PastReturnFactor(
        PolarsFactorConfig(
            window=window,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n": n},
        )
    )
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            window=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": n_forward_periods},
        )
    )
    return FirstFeatureHead(
        MLConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(Path(root) / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            start_date=start_date,
            end_date=end_date,
            val_size=0.0,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
    )


def train_checkpoint(model: MLModel) -> Path:
    """`collect()` + `train()`, then return the single `*.joblib` written."""
    model.collect()
    model.train()
    found = sorted(Path(model.config.model_save_dir).rglob("*.joblib"))
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly one checkpoint under {model.config.model_save_dir}, "
            f"found {found}"
        )
    return found[0]
