from typing import NoReturn

import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha101
from KunQuant.Stage import Function

from base.config import FactorConfig
from base.factor import FactorKunQuant
from my_ops.preprocess import WindowedZScore


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
    """Alpha101 factor set over US-equity (Tiingo/`StockDataset`) data.

    **Normalization: none. This class emits raw, un-normalized factor values.**

    US equities in this project are traded with 截面 / cross-sectional
    strategies, which normalize ACROSS SYMBOLS at each timestamp -- not across
    each symbol's own rolling time window. `my_ops/preprocess.py:WindowedZScore`
    is the latter, a 时序 / time-series normalization, which is why its absence
    here is correct-by-design and NOT a defect to "fix" by adding one to align
    this class with `Alpha101SpotKline`. The downstream consumer applies its own
    cross-sectional normalization to these raw values.

    This is a strategy-type choice, not a market-dependent bug. See NORM-01 in
    `03-03-PLAN.md` and the locked decision D-09 in
    `.planning/phases/03-factor-computation-kunquant-polars/03-CONTEXT.md`;
    `tests/test_factor_kunquant.py:test_normalization_matrix_matches_recorded_strategy_types`
    is the automated lock. An actual cross-sectional Z-score op is deferred to
    the ARCH-01/ARCH-02 work in Phase 6 -- D-09 asks for raw output here, not
    for that op to be built now.

    The class's one genuine historical defect was the missing `amount` input
    (fixed in 03-03: `Alpha101.AllData` unconditionally derives `vwap` from it,
    so construction raised `RuntimeError: Bad inputs, given <class 'NoneType'>`).
    Its lack of a rolling z-score was never part of that defect.
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
