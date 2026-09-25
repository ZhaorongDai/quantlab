"""Backtest a cross-sectional stock-selection strategy on synthetic data.

This example shows the backtest layer on its own, with every other layer kept
as small as possible:

1. write a synthetic daily price panel (one symbol is delisted part-way),
2. define a Polars momentum factor, a forward-return label and a tiny
   least-squares model head,
3. backtest a long-only top-N strategy with ``run()`` in train mode,
4. replay the trained checkpoint as a long/short strategy in load mode,
5. rebuild a run from its ``config.json`` and re-run it,
6. train a walk-forward cross-validation and backtest it with ``run_cv()``.

Nothing touches the network and no credentials are needed. Weights & Biases
logging is switched off through ``WANDB_MODE=disabled``. Everything is
written to a temporary directory that is removed at the end.

Run it from the repository root with::

    uv run python examples/backtest.py
"""

import os
import sys

# Every model training run calls wandb.init(); "disabled" makes it a no-op.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import dataclasses
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import (
    CrossSectionBacktestConfig,
    DatasetConfig,
    MLConfig,
    PolarsFactorConfig,
)
from quantlab.base.factor import FactorPolars
from quantlab.base.model import MLModel
from quantlab.dataset.stock import StockDataset
from quantlab.utils.module import load_backtester_from_config

# zarr warns on every write that consolidated metadata is not part of the
# Zarr v3 specification; it is harmless here.
warnings.filterwarnings("ignore", message="Consolidated metadata")

# Only errors from the library; the backtester logs a lot at INFO level.
logger.remove()
logger.add(sys.stderr, level="ERROR")

SYMBOLS = [f"S{i:02d}" for i in range(12)]
N_BARS = 300
DELISTED = "S08"  # held going into its delisting, so it is force-sold
DELIST_BAR = 250


# --- 1. Data ---------------------------------------------------------------


def write_synthetic_prices(root: Path) -> DatasetConfig:
    """Write a daily price panel to Zarr and return the dataset config.

    Each symbol has its own constant drift, so past returns carry some
    information about future returns and a momentum strategy has something
    to find. The layout matches what the Tiingo converter produces: the
    split- and dividend-adjusted columns ``adjOpen`` ... ``adjVolume`` and
    their unadjusted counterparts, each on ``(timestamp, symbol)``.
    """
    rng = np.random.default_rng(7)
    timestamps = pd.bdate_range("2023-01-02", periods=N_BARS)
    shape = (N_BARS, len(SYMBOLS))

    drift = np.linspace(-0.002, 0.003, len(SYMBOLS))
    close = 50.0 * np.exp(np.cumsum(drift + rng.normal(0.0, 0.01, shape), axis=0))
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, 0.005, shape))
    columns = {
        "adjOpen": open_,
        "adjHigh": np.maximum(open_, close) * 1.01,
        "adjLow": np.minimum(open_, close) * 0.99,
        "adjClose": close,
        "adjVolume": rng.uniform(1e5, 1e6, shape),
    }
    for adjusted in list(columns):
        columns[adjusted.removeprefix("adj").lower()] = columns[adjusted].copy()

    # Delisting: from DELIST_BAR on the symbol has no prices at all.
    for values in columns.values():
        values[DELIST_BAR:, SYMBOLS.index(DELISTED)] = np.nan

    panel = xr.Dataset(
        {name: (("timestamp", "symbol"), v) for name, v in columns.items()},
        coords={"timestamp": timestamps, "symbol": SYMBOLS},
    )
    zarr_path = root / "data" / "stock.zarr"
    panel.to_zarr(zarr_path, mode="w")
    return DatasetConfig(
        zarr_file_path=str(zarr_path),
        raw_data_dir_path=str(root / "downloads" / "tiingo"),
        catalog_path=str(root / "catalog"),
        market="us_equity",
        frequency="1d",
        vendor="tiingo",
    )


def fresh_dataset(config: DatasetConfig) -> StockDataset:
    """Return a dataset over a copy of ``config``.

    Factors and the backtester move a dataset's dates around, so every
    consumer gets its own dataset object.
    """
    return StockDataset(dataclasses.replace(config))


# --- 2. Factor, label and model head -----------------------------------------


class PastReturn(FactorPolars):
    """``past_ret_{n}``: the adjusted close over the close ``n`` bars earlier, minus 1."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.config.kwargs["n"]
        close = pl.col("adjClose")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((close / close.shift(n).over("symbol") - 1.0).alias(f"past_ret_{n}"))
            .select(["timestamp", "symbol", f"past_ret_{n}"])
        )

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data


class ForwardReturn(FactorPolars):
    """``fwd_ret_{n}``: the return from the next bar's open to the open n bars later.

    This matches how the backtester trades: a signal formed at bar t fills at
    the open of bar t + 1, so that is where the return the model learns to
    predict starts.
    """

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.config.kwargs["n_forward_periods"]
        open_ = pl.col("adjOpen")
        entry = open_.shift(-1).over("symbol")
        exit_ = open_.shift(-(n + 1)).over("symbol")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((exit_ / entry - 1.0).alias(f"fwd_ret_{n}"))
            .select(["timestamp", "symbol", f"fwd_ret_{n}"])
        )

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        return data


class LeastSquaresHead(MLModel):
    """Linear regression of every label on the features, fitted with numpy."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return None  # the coefficients are created in _fit_model

    def _preprocess(self, data):
        return np.array(data, dtype=np.float64, copy=True)

    def _fit_model(self, train_x, train_y, val_x, val_y):
        x = train_x.reshape(-1, train_x.shape[-1])
        y = train_y.reshape(-1, train_y.shape[-1])
        rows = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        design = np.column_stack([np.ones(rows.sum()), x[rows]])
        coef, *_ = np.linalg.lstsq(design, y[rows], rcond=None)
        self.model = {"coef": coef}

    def _forward(self, x):
        coef = self.model["coef"]
        return coef[0] + x @ coef[1:]


def make_model(root: Path, prices: DatasetConfig, **dates) -> LeastSquaresHead:
    """Build the model with one factor and one label from plain config objects."""
    factor = PastReturn(
        # `window` is the factor's look-back in bars; the backtester uses it
        # to start the factor computation early enough (the warm-up).
        PolarsFactorConfig(window=5, dataset=fresh_dataset(prices), kwargs={"n": 5})
    )
    label = ForwardReturn(
        # `n_forward_periods` tells the backtester how far each label looks
        # ahead, which extends the in-sample window past train_end.
        PolarsFactorConfig(
            window=0, dataset=fresh_dataset(prices), kwargs={"n_forward_periods": 5}
        )
    )
    return LeastSquaresHead(
        MLConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(root / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            val_size=0.0,
            **dates,
        )
    )


def show(title: str, result) -> None:
    """Print the headline numbers of one backtest run."""
    metrics = result.metrics
    whole = metrics.get("whole") or metrics["stitched"]["whole"]
    print(f"\n== {title}")
    print("run directory:", result.run_dir.name)
    for key in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"):
        print(f"  {key:<18} {whole[key]:8.3f}")
    print(f"  {'order_count':<18} {whole['order_count']:8d}")
    print(f"  {'turnover/rebal.':<18} {whole['turnover']['mean_per_rebalance']:8.3f}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prices = write_synthetic_prices(root)
        bars = fresh_dataset(prices).read().get_xarray_dataset().timestamp.values
        day = lambda i: pd.Timestamp(bars[i]).strftime("%Y-%m-%d")

        # --- 3. run() in train mode, long-only --------------------------------
        # The model trains on bars 0..179 and is evaluated on 180..199. The
        # backtest window deliberately starts at bar 175, inside the training
        # window, to show how the metrics are split.
        dates = dict(
            start_date=day(0), end_date=day(199),
            train_start=day(0), train_end=day(179),
            test_start=day(180), test_end=day(199),
        )
        long_only = USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=fresh_dataset(prices),
                model=make_model(root, prices, **dates),
                model_mode="train",
                start_date=day(175),
                end_date=day(N_BARS - 1),
                output_dir=str(root / "backtests"),
                rebalance_periods=5,
                direction="long_only",
                top_n=3,
                fees=0.0005,
                slippage=0.0005,
            )
        )
        result = long_only.run()
        show("long-only top 3, train mode", result)
        print("  training window    ", result.metrics["training_window"])
        print("  in-sample range    ", result.metrics["in_sample_range"])
        print("  out-of-sample      ", result.metrics["out_of_sample_ranges"])
        oos = result.metrics["out_of_sample"]
        print(f"  out-of-sample Sharpe {oos['Sharpe Ratio']:.3f}")
        print("files:", sorted(p.name for p in result.run_dir.iterdir()))

        weights = result.weights["weight"].to_pandas()
        print("first rebalance:", weights.iloc[0][weights.iloc[0] > 0].round(4).to_dict())
        for record in result.simulation.liquidations:
            print(
                "forced liquidation:", record["symbol"],
                "signal", record["signal_timestamp"].date(),
                "fill", record["fill_timestamp"].date(),
                f"at {record['price']:.2f}",
            )

        # --- 4. load mode, long/short ----------------------------------------
        checkpoint = result.metrics["trained_checkpoint"]
        long_short = USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=fresh_dataset(prices),
                model=make_model(root, prices, **dates),
                model_mode="load",
                checkpoint=checkpoint,
                start_date=day(200),
                end_date=day(N_BARS - 1),
                output_dir=str(root / "backtests"),
                rebalance_periods=5,
                direction="long_short",
                top_n=3,
            )
        )
        ls_result = long_short.run()
        show("long/short top 3 / bottom 3, load mode", ls_result)
        first = ls_result.weights["weight"].to_pandas().iloc[0]
        print("  gross exposure", round(float(first.abs().sum()), 6),
              "net exposure", round(float(first.sum()), 6))

        # --- 5. rebuild from config.json and re-run ---------------------------
        # The classes above live in this script (`__main__`), which the loader
        # can import only because it is the running script; in a project they
        # would live in an importable module.
        saved = json.loads((ls_result.run_dir / "config.json").read_text())
        rebuilt = load_backtester_from_config(saved)
        again = rebuilt.run()
        same = np.allclose(again.simulation.value.values, ls_result.simulation.value.values)
        print("\nrebuilt from config.json, identical equity curve:", same)

        # --- 6. walk-forward CV and run_cv() ---------------------------------
        # 100-bar training segments, 20-bar test segments, stepping 20 bars.
        cv_model = make_model(
            root, prices,
            start_date=day(0), end_date=day(N_BARS - 1),
            train_start=day(0), train_end=day(99),
            test_start=day(100), test_end=day(119),
        )
        cv_model.collect()
        folds = cv_model.train_cv(train_periods=100)
        cv_project_dir = Path(folds[0]["checkpoint"]).parent.parent
        cv_backtester = USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=fresh_dataset(prices),
                model=make_model(root, prices, **dates),
                model_mode="load",
                cv_project_dir=str(cv_project_dir),
                start_date=day(100),
                end_date=day(N_BARS - 1),
                output_dir=str(root / "backtests"),
                rebalance_periods=5,
                direction="long_only",
                top_n=3,
            )
        )
        cv_result = cv_backtester.run_cv()
        show(f"run_cv over {len(cv_result.folds)} folds (stitched)", cv_result)
        for fold in cv_result.metrics["folds"][:3]:
            ret = fold["metrics"]["whole"]["Total Return [%]"]
            print(f"  fold {fold['fold']}: {fold['test_start']}..{fold['test_end']} "
                  f"return {ret:6.2f}%")
        print("files:", sorted(p.name for p in cv_result.run_dir.iterdir()))


if __name__ == "__main__":
    main()
