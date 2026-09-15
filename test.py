# %%
import os
import sys

# macOS 上同一进程混用 torch 与 xgboost 需要单线程 OpenMP（见 example/model.md），必须在 import quantlab 之前设置。
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
    """每个使用方一个独立的数据集对象，互不改写日期。"""
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
# 流水线：Alpha101 因子 -> 5 日远期收益标签 -> XGBoost -> 截面 TopN 回测
# W&B：模型训练会开一个 run，use_wandb=True 再为回测开一个 run（需要先 `wandb login` 或设置 WANDB_API_KEY）。

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
        model_mode="train",  # 用模型自己的 train/test 日期训练，再回测
        start_date="2024-01-10",
        end_date=end_date,
        output_dir="./backtests",
        rebalance_periods=5,  # 每 5 个 bar 调仓一次
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
