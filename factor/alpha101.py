from typing import NoReturn

import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha101
from KunQuant.Stage import Function

from base.config import FactorConfig
from base.factor import FactorKunQuant
from my_ops.preprocess import WindowedZScore


class Alpha101SpotKline(FactorKunQuant):
    def __init__(self, factor_config: FactorConfig):
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("close")
            low = Input("low")
            high = Input("high")
            vopen = Input("open")
            amount = Input("amount")
            vol = Input("volume")
            all_data = Alpha101.AllData(
                low=low,
                high=high,
                close=close,
                open=vopen,
                amount=amount,
                volume=vol,
            )
            for alpha in Alpha101.all_alpha:
                if alpha.__name__ in factor_names:
                    Output(
                        WindowedZScore(alpha(all_data), self.config.window),
                        alpha.__name__,
                    )
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        factors = [alpha.__name__ for alpha in Alpha101.all_alpha]
        return tuple(factors)

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data


class Alpha101Stock(FactorKunQuant):
    def __init__(self, factor_config: FactorConfig):
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("close")
            low = Input("low")
            high = Input("high")
            vopen = Input("open")
            vol = Input("volume")
            all_data = Alpha101.AllData(
                low=low,
                high=high,
                close=close,
                open=vopen,
                volume=vol,
            )
            for alpha in Alpha101.all_alpha:
                if alpha.__name__ in factor_names:
                    Output(
                        alpha(all_data),
                        alpha.__name__,
                    )
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        factors = [alpha.__name__ for alpha in Alpha101.all_alpha]
        return tuple(factors)

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data
