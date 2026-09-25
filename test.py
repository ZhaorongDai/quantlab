"""Ad hoc example: Alpha101 factors, a forward-return label, XGBoost, backtest.

A cell-style (``# %%``) walkthrough of the whole pipeline on daily US-equity
data: compute Alpha101 factors, build a 5-day forward-return label, train an
``XGBoostRegressor`` and run a cross-sectional long/short top-N backtest.
Runs at import with hardcoded, machine-specific store paths; edit them before
running. Not part of the library.
"""

# %%
import os
import sys

# Mixing torch and xgboost in one process on macOS needs single-threaded
# OpenMP (see docs/model.md); it must be set before quantlab is imported.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import (
    CrossSectionBacktestConfig,
    DatasetConfig,
    FactorConfig,
    MLConfig,
)
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.label.fret import Return
from quantlab.ml_model.xgb import XGBoostRegressor

start_date = "2020-01-01"
end_date = "2026-01-01"
data_columns = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


def us_equity() -> StockDataset:
    """Return a fresh ``StockDataset`` for the daily US-equity store.

    Each consumer gets its own instance so that none of them rewrites another
    one's date window.

    Example:
        >>> ds = us_equity()
        >>> ds.config.market
        'us_equity'
    """
    return StockDataset(
        DatasetConfig(
            zarr_file_path="/home/zhrdai/projects/quantlab2/data/data/us_equity/1d/us_all.zarr",
            raw_data_dir_path="/home/zhrdai/projects/quantlab2/data/downloads",
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
            catalog_path="",
        )
    )


def alpha101(**kwargs) -> Alpha101Stock:
    """Return an ``Alpha101Stock`` factor over a fresh dataset.

    Args:
        **kwargs: Extra ``FactorConfig`` fields, typically ``start_date`` and
            ``end_date``.

    Example:
        >>> factor = alpha101(start_date="2020-01-01", end_date="2026-01-01")
        >>> factor.config.window
        252
    """
    return Alpha101Stock(
        FactorConfig(
            window=252,
            dataset=us_equity(),
            njobs=64,
            mode="batch",
            data_columns=data_columns,
            **kwargs,
        )
    )


factor = alpha101(start_date=start_date, end_date=end_date)
# %%


factor.cal()

# %%

factor.get_features()

# %%
# Pipeline: Alpha101 factors, 5-day forward-return label, XGBoost, then a
# cross-sectional top-N backtest.
# W&B: training opens one run, and use_wandb=True opens a second one for the
# backtest (run `wandb login` first or set WANDB_API_KEY).

label = Return(
    FactorConfig(
        window=5,
        dataset=us_equity(),
        njobs=64,
        mode="batch",
        data_columns=("adjClose",),
        kwargs={"n_forward_periods": 5},
    )
)

model = XGBoostRegressor(
    MLConfig(
        factors=[alpha101()],
        labels=[label],
        model_save_dir="./model_ckpt",
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=start_date,
        end_date=end_date,
        train_start="2020-01-01",
        train_end="2023-12-31",
        test_start="2024-01-10",
        test_end=end_date,
        early_stopping=True,
        early_stopping_patience=50,
        hyperparameters={"num_boost_round": 1000},
    )
)

backtester = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        price_dataset=us_equity(),
        model=model,
        model_mode="train",  # train on the model's own train/test dates, then backtest
        start_date="2024-01-10",
        end_date=end_date,
        output_dir="./backtests",
        rebalance_periods=5,  # rebalance every 5 bars
        direction="long_short",
        top_n=50,
        use_wandb=True,
    )
)
result = backtester.run()

# %%

metrics = result.metrics
print("run dir:", result.run_dir)
print("training_window:", metrics["training_window"])
print("Total Return [%]:", metrics["whole"]["Total Return [%]"])
print("Sharpe Ratio:", metrics["whole"]["Sharpe Ratio"])
print("Max Drawdown [%]:", metrics["whole"]["Max Drawdown [%]"])
print("out_of_sample:", metrics["out_of_sample"])
