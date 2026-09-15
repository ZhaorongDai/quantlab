from typing import NoReturn

import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha158
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig
from quantlab.base.factor import FactorKunQuant
from quantlab.my_ops.preprocess import WindowedZScore


class Alpha158SpotKline(FactorKunQuant):
    """Alpha158 factor set over crypto-spot kline (`SpotKlineDataset`) data.
    """

    def __init__(self, factor_config: FactorConfig):
        super().__init__(factor_config)

    def _get_factor_names(self) -> tuple[str, ...]:
        return tuple(self._factor_names_stream())

    def _get_func_names(self):
        close = Input("close")
        low = Input("low")
        high = Input("high")
        vopen = Input("open")
        amount = Input("amount")
        vol = Input("volume")
        all_data = Alpha158.AllData(
            low=low,
            high=high,
            close=close,
            open=vopen,
            amount=amount,
            volume=vol,
        )
        alpha158, names = all_data.build(
            {
                "kbar": {},  # 是否使用K线特征
                "price": {
                    "windows": [0, 1, 2, 3, 4],
                    "feature": [
                        ("OPEN", all_data.open),
                        ("HIGH", all_data.high),
                        ("LOW", all_data.low),
                        ("CLOSE", all_data.close),
                        ("VWAP", all_data.vwap),
                    ],
                },
                "volume": {
                    "windows": [0, 1, 2, 3, 4],
                },
                "rolling": {  # 是否使用滚动窗口特征
                    "windows": [5, 10, 20, 30, 60],  # 滚动窗口大小
                    "exclude": ["BETA", "RSQR", "RESI"],
                },
            }
        )
        return alpha158, names

    def _factor_names_stream(self):
        return self._get_func_names()[-1]

    def _get_func_stream(self) -> Function:
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("close")
            low = Input("low")
            high = Input("high")
            vopen = Input("open")
            amount = Input("amount")
            vol = Input("volume")
            alpha158, names = self._get_func_names()
            for v, k in zip(alpha158, names):
                if k in factor_names:
                    Output(WindowedZScore(v, self.config.window), k)
        return Function(builder.ops)

    def _get_factor_func(self):
        return self._get_func_stream()

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data


class Alpha158Stock(FactorKunQuant):
    """Alpha158 factor set over US-equity (Tiingo/`StockDataset`) data.

    Reads only the adjusted series `adjOpen`/`adjHigh`/`adjLow`/`adjClose`/
    `adjVolume` (list exactly these in `data_columns`). `vwap` is the adjusted
    typical price `(adjHigh + adjLow + adjClose) / 3`, not `amount / volume`:
    Tiingo supplies no dollar volume, and mixing a raw amount with adjusted
    volume would break at every split (REVIEW WR-02).
    """

    def __init__(self, factor_config: FactorConfig):
        super().__init__(factor_config)

    def _get_factor_names(self) -> tuple[str, ...]:
        return tuple(self._factor_names_stream())

    def _get_func_names(self):
        close = Input("adjClose")
        low = Input("adjLow")
        high = Input("adjHigh")
        vopen = Input("adjOpen")
        vol = Input("adjVolume")
        # REVIEW WR-02: no stock store carries `amount`, and a raw dollar
        # volume divided by the split-adjusted `adjVolume` would jump at every
        # split. Use the adjusted typical price instead, so VWAP* features sit
        # on the same adjusted scale as the rest of the graph.
        vwap = (high + low + close) / 3.0
        all_data = Alpha158.AllData(
            low=low,
            high=high,
            close=close,
            open=vopen,
            volume=vol,
            vwap=vwap,
        )
        # KunQuant's `AllData.__init__` only assigns `self.vwap` when it derives
        # it from `amount`; a passed `vwap=` is otherwise dropped.
        all_data.vwap = vwap
        alpha158, names = all_data.build(
            {
                "kbar": {},  # 是否使用K线特征
                "price": {
                    "windows": [0, 1, 2, 3, 4],
                    "feature": [
                        ("OPEN", all_data.open),
                        ("HIGH", all_data.high),
                        ("LOW", all_data.low),
                        ("CLOSE", all_data.close),
                        ("VWAP", all_data.vwap),
                    ],
                },
                "volume": {
                    "windows": [0, 1, 2, 3, 4],
                },
                "rolling": {  # 是否使用滚动窗口特征
                    "windows": [5, 10, 20, 30, 60],  # 滚动窗口大小
                    "exclude": ["BETA", "RSQR", "RESI"],
                },
            }
        )
        return alpha158, names

    def _factor_names_stream(self):
        return self._get_func_names()[-1]

    def _get_func_stream(self) -> Function:
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            alpha158, names = self._get_func_names()
            for v, k in zip(alpha158, names):
                if k in factor_names:
                    Output(v, k)
        return Function(builder.ops)

    def _get_factor_func(self):
        return self._get_func_stream()

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data
