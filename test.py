# %%
import os
from pathlib import Path

import numpy as np

from config import DatasetConfig, FactorConfig
from dataset.stock import StockDataset
from factor.alpha101 import Alpha101Stock
from utils import file

_data_root = Path(os.environ.get("QUANTLAB_DATA_DIR", Path(__file__).resolve().parent))

dscfg = DatasetConfig(
    raw_data_dir_path=str(_data_root / "scripts" / "downloads" / "nasdaq_data"),
    zarr_file_path=str(
        _data_root / "scripts" / "downloads" / "nasdaq_data" / "stock.zarr"
    ),
    catalog_path="none",
)


ds = StockDataset(dscfg)

facfg = FactorConfig(
    window=10,
    dataset=ds,
    mode="batch",
    data_columns=["open", "high", "low", "close", "volume"],
)
# %%

fc = Alpha101Stock(facfg)
# %%
fc.get_features()

