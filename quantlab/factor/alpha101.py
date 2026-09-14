from typing import NoReturn

import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha101
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig
from quantlab.base.factor import FactorKunQuant
from quantlab.my_ops.preprocess import WindowedZScore


class Alpha101SpotKline(FactorKunQuant):
    """Alpha101 factor set over crypto-spot kline (`SpotKlineDataset`) data.

    **Normalization: `my_ops/preprocess.py:WindowedZScore` over
    `self.config.window`, applied around every `Output(...)`.**

    That op is a 时序 / time-series normalization: it standardizes each symbol
    against that symbol's OWN rolling window. Crypto spot in this project is
    traded with 时序 / time-series strategies, which is exactly the
    normalization they want.

    This is a strategy-type choice, not a market-dependent defect -- the
    US-equity siblings (`Alpha101Stock`, `Alpha158Stock`) deliberately omit it
    because they serve 截面 / cross-sectional strategies. See NORM-01 in
    `03-03-PLAN.md` and the locked decision D-09 in
    `.planning/phases/03-factor-computation-kunquant-polars/03-CONTEXT.md`;
    `tests/test_factor_kunquant.py:test_normalization_matrix_matches_recorded_strategy_types`
    is the automated lock on the four-class matrix.
    """

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
    """Alpha101 factor set over US-equity (Tiingo/`StockDataset`) data."""

    def __init__(self, factor_config: FactorConfig):
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("adjClose")
            low = Input("adjLow")
            high = Input("adjHigh")
            vopen = Input("adjOpen")
            vol = Input("adjVolume")
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
