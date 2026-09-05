from factor.alpha101 import Alpha101SpotKline
from factor.alpha158 import Alpha158SpotKline
from config import alpha101_config, alpha158_config
from dl_model.mlp import MLPRegressor
from dl_model.rnn import RNNRegressor

# from dl_model.transformer import Regressor
from dataclasses import dataclass, asdict
from base.config import DLConfig, FactorConfig
from factor.alpha101 import Alpha101SpotKline
from factor.alpha158 import Alpha158SpotKline
from config import alpha101_config, alpha158_config, spot_kline_config
from dataset.spot import SpotKlineDataset
from config import spot_kline_config, spot_label_config
from label.spot import SpotReturn, SpotBinaryReturn
from base.model import BaseModel
import numpy as np
import xarray as xr

if __name__ == "__main__":
    # ds = SpotKlineDataset(spot_kline_config())
    # ds.from_csv().save()

    alpha101 = Alpha101SpotKline(alpha101_config())
    alpha101.cal().save()

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
