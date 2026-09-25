# %%
import polars as pl

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import (
    ConstituentDatasetConfig,
    CrossSectionBacktestConfig,
    DatasetConfig,
    FactorConfig,
    MLConfig,
)
from quantlab.dataset.constituent import Nasdaq100ConstituentDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.alpha158 import Alpha158Stock
from quantlab.factor.universe_filter import UniverseFilteredFactor
from quantlab.label.fret import Return
from quantlab.ml_model.xgb import XGBoostRegressor

data_columns = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")

start_date = "2015-01-01"
end_date = "2026-01-01"


def us_equity() -> StockDataset:
    """每个使用方一个独立的数据集对象，互不改写日期。"""
    return StockDataset(
        DatasetConfig(
            zarr_file_path="/home/zhrdai/projects/quantlab2/data/data/us_equity/1d/wrds_crsp_all_1d.zarr",
            raw_data_dir_path="/home/zhrdai/projects/quantlab2/data/downloads",
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
            catalog_path="",
            start_date=start_date,
            end_date=end_date,
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
            file_path="/home/zhrdai/projects/quantlab2/data/factors/alpha101.zarr",
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    )


def alpha158(**kwargs) -> Alpha158Stock:
    return Alpha158Stock(
        FactorConfig(
            window=252,
            dataset=us_equity(),
            njobs=64,
            mode="batch",
            data_columns=data_columns,
            file_path="/home/zhrdai/projects/quantlab2/data/factors/alpha158.zarr",
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    )


# %%

alpha101_factor = alpha101()
price = us_equity()

# %%

alpha101_factor.cal()


# %%

alpha = pl.from_pandas(
    alpha101_factor.get_features().to_dataframe().reset_index()
)
# %%

prices = price.read().get_lazyframe().collect()
# %%
prices = prices.sort(["symbol", "timestamp"])

for period in [1, 5, 10]:
    entry = pl.col("adjOpen").shift(-1).over("symbol")
    exit_ = pl.col("adjOpen").shift(-(period + 1)).over("symbol")

    prices = prices.with_columns(
        (exit_ / entry - 1).alias(f"fret_{period}")
    )
# %%


import matplotlib.pyplot as plt
import polars as pl
from alphainspect.plotting import create_describe1_sheet
from alphainspect.portfolio import create_portfolio_sheet
from alphainspect.reports import create_1x3_sheet, create_3x2_sheet
from alphainspect.turnover import create_turnover_sheet
from alphainspect.utils import (  # noqa
    _DATE_,
    with_factor_quantile,
    with_factor_top_k,
)

df = alpha.join(
    prices,
    on=["timestamp", "symbol"],
    how="left",
)
df = df.rename({"timestamp": "date", "symbol": "asset"})
factor = "alpha010"  # 考察因子
forward_returns = ["fret_1", "fret_5", "fret_10"]  # 同一因子，不同持有期对比

# %% 因子值分层
df = with_factor_quantile(
    df, factor, quantiles=9, by=[_DATE_], factor_quantile="_fq_1"
)
# df = with_factor_top_k(df, factor, top_k=20, by=[_DATE_], factor_quantile='_fq_1')

# %% 分组后因子值的描述性统计
create_describe1_sheet(df, [factor], factor_quantile="_fq_1")
# %% 对应收益的描述性统计
create_describe1_sheet(df, forward_returns, factor_quantile="_fq_1")

# %% IC统计
axvlines = (
    "2020-01-01",
    "2024-01-01",
)

# 有多个，挑一个显示
for fwd_ret_1 in forward_returns[1:2]:
    fig, ic_dict, hist_dict, cum, avg, std = create_1x3_sheet(
        df, factor, fwd_ret_1, factor_quantile="_fq_1", axvlines=axvlines
    )

# %% 画比较全的图
create_3x2_sheet(
    df, factor, fwd_ret_1, factor_quantile="_fq_1", axvlines=axvlines
)
# %% 绩效曲线图
create_portfolio_sheet(
    df, fwd_ret_1, factor_quantile="_fq_1", axvlines=axvlines
)
# %% 换手率
create_turnover_sheet(
    df,
    factor,
    periods=(1, 5, 10, 20),
    factor_quantile="_fq_1",
    axvlines=axvlines,
)
# %%
plt.show()

# %%
