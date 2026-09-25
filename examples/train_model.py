"""Train, evaluate, reload and cross-validate a return model on synthetic data.

This example concentrates on the model layer. It

1. writes a small synthetic US-equity price panel with a planted one-day
   reversal effect (a stock that rose today tends to fall tomorrow),
2. defines a small KunQuant factor (yesterday's return and the distance
   from the 5-day average) and computes it with a one-day forward-return
   label,
3. trains an ``XGBoostRegressor`` with early stopping and saves a checkpoint,
4. predicts the test window with ``predict_panel`` and scores the prediction
   with IC, RankIC and R2,
5. rebuilds the model from the checkpoint's ``config.json`` and checks that
   the reloaded model predicts exactly the same values,
6. runs a walk-forward cross-validation with ``train_cv`` and prints the
   per-fold scores and the ``cv_folds.json`` manifest.

Everything runs offline on the CPU in well under a minute. No credentials
are needed, Weights & Biases logging is switched off with
``WANDB_MODE=disabled``, and all files go to a temporary directory that is
removed at the end.

Run it from the repository root with::

    uv run python examples/train_model.py
"""

import os
import sys

# Set the environment before torch or xgboost is imported (quantlab's model
# layer imports both). "disabled" turns every wandb call into a no-op.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
# macOS only: torch and xgboost ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.base.config import DatasetConfig, FactorConfig, MLConfig
from quantlab.base.factor import FactorKunQuant
from quantlab.dataset.stock import StockDataset
from quantlab.label.fret import Return
from quantlab.ml_model.xgb import XGBoostRegressor
from quantlab.utils.metrics import regression_panel_metrics
from quantlab.utils.module import load_model_from_config

# Only warnings and errors from the library, so the printed results stand out.
logger.remove()
logger.add(sys.stderr, level="WARNING")

# KunQuant processes symbols in SIMD blocks of 8, so use a multiple of 8.
SYMBOLS = [f"S{i:02d}" for i in range(16)]
N_BARS = 360
ADJUSTED = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


def write_synthetic_prices(root: Path) -> DatasetConfig:
    """Write a daily price panel with a planted reversal and return its config.

    Daily log returns follow ``r[t] = -0.3 * r[t-1] + noise`` for every
    symbol, so yesterday's move carries information about today's. The
    store uses the layout the Tiingo converter produces: one variable per
    price field on the dimensions ``(timestamp, symbol)``.
    """
    rng = np.random.default_rng(7)
    shape = (N_BARS, len(SYMBOLS))
    noise = rng.normal(0.0, 0.02, shape)
    log_ret = np.zeros(shape)
    for t in range(1, N_BARS):
        log_ret[t] = -0.3 * log_ret[t - 1] + noise[t]

    close = 50.0 * np.exp(np.cumsum(log_ret, axis=0))
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, 0.002, shape))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.uniform(1e5, 1e6, shape)

    columns = dict(zip(ADJUSTED, (open_, high, low, close, volume)))
    # With no splits or dividends the raw fields equal the adjusted ones.
    for raw, adj in zip(("open", "high", "low", "close", "volume"), ADJUSTED):
        columns[raw] = columns[adj]

    panel = xr.Dataset(
        {name: (("timestamp", "symbol"), values) for name, values in columns.items()},
        coords={
            "timestamp": pd.bdate_range("2022-01-03", periods=N_BARS),
            "symbol": SYMBOLS,
        },
    )
    zarr_path = root / "data" / "stock.zarr"
    panel.to_zarr(zarr_path, mode="w")
    return DatasetConfig(
        zarr_file_path=str(zarr_path),
        raw_data_dir_path=str(root / "downloads"),  # unused: the store exists
        market="us_equity",
        frequency="1d",
    )


class ReversalFeatures(FactorKunQuant):
    """Two price features computed by a compiled KunQuant graph.

    ``past_ret_1`` is the one-day close-to-close return and ``ma_dev_5`` the
    close's distance from its 5-day moving average. Graph inputs are named
    after the dataset variables listed in ``data_columns``, here the
    split-adjusted close ``adjClose``.
    """

    def _get_factor_names(self):
        """Return the names of the two factors this class computes."""
        return ("past_ret_1", "ma_dev_5")

    def _get_factor_func(self):
        """Build the KunQuant graph for the one-day return and the moving-average distance."""
        builder = Builder()
        with builder:
            close = Input("adjClose")
            Output(op.SubConst(op.Div(close, op.BackRef(close, 1)), 1.0), "past_ret_1")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev_5")
        return Function(builder.ops)

    def _get_features(self, data):
        """Return the factor values unchanged; no post-processing is needed."""
        return data


def build_model(root: Path, dataset_config: DatasetConfig, dates: dict) -> XGBoostRegressor:
    """Build the features, the label and the model from plain config objects."""
    # Every factor and label gets its own dataset object, because each one
    # moves its dataset's start date back by `window` days for warm-up.
    features = ReversalFeatures(
        FactorConfig(
            window=10,  # calendar days of warm-up history for the 5-day mean
            dataset=StockDataset(DatasetConfig(**vars(dataset_config))),
            mode="batch",
            data_columns=("adjClose",),
            file_path=str(root / "factors" / "reversal.zarr"),
            njobs=4,
        )
    )
    label = Return(
        FactorConfig(
            window=0,
            dataset=StockDataset(DatasetConfig(**vars(dataset_config))),
            mode="batch",
            data_columns=("adjOpen",),
            kwargs={"n_forward_periods": 1},
            file_path=str(root / "labels" / "ret_1.zarr"),
            njobs=4,
        )
    )
    return XGBoostRegressor(
        MLConfig(
            factors=[features],
            labels=[label],
            model_save_dir=str(root / "models"),
            factor_data_strategy="cal",  # compute now; "read" loads the stores
            label_data_strategy="cal",
            # One xgboost thread is plenty for this tiny panel and keeps the
            # run fast on a busy machine.
            hyperparameters={"num_boost_round": 300, "max_depth": 3, "eta": 0.05, "nthread": 1},
            early_stopping=True,
            early_stopping_patience=20,
            val_size=0.2,
            **dates,
        )
    )


def show(title: str, metrics: dict) -> None:
    """Print IC, RankIC and R2 from a metrics dict on one line."""
    ic, rank_ic, r2 = (metrics[k] for k in ("ic", "rank_ic", "r2"))
    print(f"{title:<22} IC={ic:+.3f}  RankIC={rank_ic:+.3f}  R2={r2:+.3f}")


def main() -> None:
    """Run the training walkthrough in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_config = write_synthetic_prices(root)

        # Dates are bar positions turned into strings: bars 0-279 train
        # (the last 20% of them validate), bars 280-359 test.
        bars = pd.bdate_range("2022-01-03", periods=N_BARS)
        day = lambda i: bars[i].strftime("%Y-%m-%d")
        dates = dict(
            start_date=day(0),
            end_date=day(N_BARS - 1),
            train_start=day(0),
            train_end=day(279),
            test_start=day(280),
            test_end=day(N_BARS - 1),
        )

        # 1. Build, collect and train --------------------------------------
        model = build_model(root, dataset_config, dates)
        print("features:", model.get_factor_names(), "label:", model.get_label_names())
        model.collect()  # computes both panels and merges them
        checkpoint = model.train()
        print("checkpoint:", checkpoint.relative_to(root))
        print("trees kept by early stopping:", model.model.num_boosted_rounds())

        # 2. Predict the test window and score it --------------------------
        panel = model.data_backend.get_xarray_dataset(["timestamp", "symbol"])
        test = panel.sel(timestamp=slice(dates["test_start"], dates["test_end"]))
        pred = model.predict_panel(test[model.get_factor_names()])
        print("prediction panel:", dict(pred.sizes), list(pred.data_vars))
        show("test window", regression_panel_metrics(pred["ret_1"].values, test["ret_1"].values))

        # 3. Rebuild the model from config.json and reload the checkpoint --
        saved = json.loads((checkpoint.parent / "config.json").read_text())
        print("trained_on symbols:", len(saved["trained_on"]["symbols"]),
              "resolved eta:", saved["resolved_hyperparameters"]["eta"])
        reloaded = load_model_from_config(saved).load(checkpoint)
        again = reloaded.predict_panel(test[model.get_factor_names()])
        print("reloaded model predicts the same values:",
              bool(np.allclose(again["ret_1"].values, pred["ret_1"].values, equal_nan=True)))

        # 4. Walk-forward cross-validation ---------------------------------
        # Each fold trains on 200 bars, skips 2 bars (the label looks two
        # bars ahead, so this keeps test information out of training) and
        # tests on the next 200 // 5 = 40 bars. The window then slides by 40.
        folds = model.train_cv(train_periods=200, gap_periods=2)
        for fold in folds:
            print(f"fold {fold['fold']}: train {fold['train_start'][:10]}..{fold['train_end'][:10]}"
                  f"  test {fold['test_start'][:10]}..{fold['test_end'][:10]}"
                  f"  IC={fold['test_ic']:+.3f}  RankIC={fold['test_rank_ic']:+.3f}")
        print("mean test IC over folds:", round(float(np.mean([f["test_ic"] for f in folds])), 3))

        manifest_path = Path(folds[0]["checkpoint"]).parents[1] / "cv_folds.json"
        manifest = json.loads(manifest_path.read_text())
        print("cv_folds.json: format_version", manifest["format_version"],
              "with", len(manifest["folds"]), "folds")
        print("keys of one fold:", sorted(manifest["folds"][0]))


if __name__ == "__main__":
    main()
