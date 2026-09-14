# %%
from quantlab.base.config import DatasetConfig, FactorConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock

start_date = "2020-01-01"
end_date = "2026-01-01"

ds_config = DatasetConfig(
    zarr_file_path="/home/zhrdai/projects/quantlab2/data/data/us_equity/1d/us_all.zarr",
    raw_data_dir_path="/home/zhrdai/projects/quantlab2/data/downloads",
    market="us_equity",
    frequency="1d",
    vendor="tiingo",
    catalog_path="",
)
ds = StockDataset(ds_config)

factor_config = FactorConfig(
    start_date=start_date,
    end_date=end_date,
    window=252,
    dataset=ds,
    njobs=64,
    mode="batch",
    data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
)

factor = Alpha101Stock(factor_config)
# %%


factor.cal()

# %%

factor.get_features()
