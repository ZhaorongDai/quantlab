"""Ad hoc example: Alpha101 factors, a forward-return label, XGBoost, backtest.

A cell-style walkthrough of the whole pipeline on daily US-equity data. The
``# %%`` markers split it into cells that VS Code or Jupyter can run one at a
time. It computes Alpha101 factors, builds a 5-day forward-return label,
trains an ``XGBoostRegressor`` on them and runs a cross-sectional long/short
top-N backtest. A cross-sectional backtest ranks all symbols against each
other on each rebalance day, buys the top N and sells short the bottom N.
It is not part of the library and has no command-line options.

The store paths are hardcoded and machine-specific. Edit them in
``us_equity()`` before running. The backtest logs to Weights & Biases
(``use_wandb=True``), so run ``wandb login`` first or set
``WANDB_API_KEY``. Checkpoints go to ``./model_ckpt`` and backtest results
to ``./backtests``, relative to the current directory.

Usage::

    uv run python test.py
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

    Each consumer gets its own instance, because a factor or model narrows
    its dataset's date window in place and a shared instance would leak one
    consumer's window into another.

    Returns
    -------
    StockDataset
        A dataset over the hardcoded Tiingo daily store.

    Examples
    --------
    ::

        ds = us_equity()
        ds.read()
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

    Parameters
    ----------
    **kwargs
        Extra ``FactorConfig`` fields, typically ``start_date`` and
        ``end_date``.

    Returns
    -------
    Alpha101Stock
        The factor, configured with a 252-bar window and 64 worker threads.

    Examples
    --------
    ::

        factor = alpha101(start_date="2020-01-01", end_date="2026-01-01")
        factor.cal()
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
# Train XGBoost on Alpha101 factors against a 5-day forward return, then
# backtest the predictions. Training opens one W&B run and the backtest,
# with use_wandb=True, opens a second one.

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
