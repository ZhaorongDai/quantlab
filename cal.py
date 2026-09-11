
from quantlab.factor.alpha101 import Alpha101SpotKline
from quantlab.factor.alpha158 import Alpha158SpotKline
from quantlab.config import alpha101_config, alpha158_config
from quantlab.dl_model.mlp import MLPRegressor
from quantlab.dl_model.rnn import RNNRegressor

# from dl_model.transformer import Regressor
from dataclasses import dataclass, asdict
from quantlab.base.config import DLConfig, FactorConfig, DatasetConfig
from quantlab.factor.alpha101 import Alpha101SpotKline
from quantlab.factor.alpha158 import Alpha158SpotKline
from quantlab.config import alpha101_config, alpha158_config, spot_kline_config
from quantlab.dataset.stock import StockDataset
from quantlab.config import spot_kline_config, spot_label_config
from quantlab.label.spot import SpotReturn, SpotBinaryReturn
from quantlab.base.model import BaseModel
import numpy as np
import xarray as xr


ds_cfg = DatasetConfig(
    zarr_file_path='/Users/daizhaorong/projects/quantlab/data/data/us_equity/1d/us_all.zarr',
    # start_date='2016-01-01',
    # end_date='2024-01-01',
    raw_data_dir_path='',
    catalog_path='',
    frequency='1d',
)
ds = StockDataset(ds_cfg)
print(ds.read().symbols)

    # alpha101 = Alpha101SpotKline(alpha101_config())
    # alpha101.cal().save()

    # alpha158 = Alpha158SpotKline(alpha158_config())
    # alpha158.cal().save()

    # label1 = SpotReturn(spot_label_config("ret_1m_60", n_forward_periods=60))
    # label2 = SpotReturn(spot_label_config("ret_1m_30", n_forward_periods=30))
    # label3 = SpotReturn(spot_label_config("ret_1m_10", n_forward_periods=10))
    # label4 = SpotReturn(spot_label_config("ret_1m_20", n_forward_periods=20))
    # label5 = SpotReturn(spot_label_config("ret_1m_5", n_forward_periods=5))
    # label6 = SpotReturn(spot_label_config("ret_1m_100", n_forward_periods=100))
    # label1.cal().save()
    # label2.cal().save()
    # label3.cal().save()
    # label4.cal().save()
    # label5.cal().save()
    # label6.cal().save()
