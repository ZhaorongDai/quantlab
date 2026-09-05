from typing import NoReturn

import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha158
from KunQuant.Stage import Function

from base.config import FactorConfig
from base.factor import FactorKunQuant
from my_ops.preprocess import WindowedZScore


class Alpha158SpotKline(FactorKunQuant):
    """Alpha158 factor set over crypto-spot kline (`SpotKlineDataset`) data.

    **Normalization: `my_ops/preprocess.py:WindowedZScore` over
    `self.config.window`, applied around every `Output(...)`.**

    That op is a 时序 / time-series normalization: it standardizes each symbol
    against that symbol's OWN rolling window. Crypto spot in this project is
    traded with 时序 / time-series strategies, which is exactly the
    normalization they want.

    This is a strategy-type choice, not a market-dependent defect -- the
    US-equity siblings (`Alpha158Stock`, `Alpha101Stock`) deliberately omit it
    because they serve 截面 / cross-sectional strategies. See NORM-01 in
    `03-03-PLAN.md` and the locked decision D-09 in
    `.planning/phases/03-factor-computation-kunquant-polars/03-CONTEXT.md`;
    `tests/test_factor_kunquant.py:test_normalization_matrix_matches_recorded_strategy_types`
    is the automated lock on the four-class matrix.
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

    **Normalization: none. This class emits raw, un-normalized factor values.**

    US equities in this project are traded with 截面 / cross-sectional
    strategies, which normalize ACROSS SYMBOLS at each timestamp -- not across
    each symbol's own rolling time window. `my_ops/preprocess.py:WindowedZScore`
    is the latter, a 时序 / time-series normalization, which is why its absence
    here is correct-by-design and NOT a defect to "fix" by adding one to align
    this class with `Alpha158SpotKline`. The downstream consumer applies its own
    cross-sectional normalization to these raw values.

    This is a strategy-type choice, not a market-dependent bug. See NORM-01 in
    `03-03-PLAN.md` and the locked decision D-09 in
    `.planning/phases/03-factor-computation-kunquant-polars/03-CONTEXT.md`;
    `tests/test_factor_kunquant.py:test_normalization_matrix_matches_recorded_strategy_types`
    is the automated lock. An actual cross-sectional Z-score op is deferred to
    the ARCH-01/ARCH-02 work in Phase 6 -- D-09 asks for raw output here, not
    for that op to be built now.

    Everything else mirrors `Alpha158SpotKline` verbatim per D-01 ("调用的步骤
    完全一致"): the same five methods, the same `Alpha158.AllData(...)` keyword
    wiring including `amount` (supplied for US equities by
    `StockDataset._to_kunquant()`'s D-02 `volume * close` proxy), the same
    `build({...})` category dict, the same double-`AllData` build (names
    outside the builder, ops inside it) and the same declared-but-unused
    `Input(...)` nodes in `_get_func_stream()`. Those are deliberately
    replicated rather than tidied: divergence in one class only would silently
    break D-01's identical-invocation guarantee. The normalization wrapper is
    the single deliberate divergence.
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
                    Output(v, k)
        return Function(builder.ops)

    def _get_factor_func(self):
        return self._get_func_stream()

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return data
