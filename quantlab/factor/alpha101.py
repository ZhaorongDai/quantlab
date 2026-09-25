"""Alpha101 factor sets computed with the KunQuant backend.

The formulaic alphas of KunQuant's ``Alpha101`` library are exposed as two
``FactorKunQuant`` subclasses that differ in which input columns they read
and whether the outputs are normalized: ``Alpha101SpotKline`` for crypto spot
klines and ``Alpha101Stock`` for adjusted US-equity bars.
"""

from typing import NoReturn

import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha101
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig
from quantlab.base.factor import FactorKunQuant
from quantlab.my_ops.preprocess import WindowedZScore


class Alpha101SpotKline(FactorKunQuant):
    """Alpha101 factors over crypto spot klines, z-scored along time.

    Reads the lowercase ``open``/``high``/``low``/``close``/``volume``/
    ``amount`` columns the spot kline dataset exposes to KunQuant. Every
    output is wrapped in ``WindowedZScore`` over ``config.window`` bars, a
    time-series normalization that standardizes each symbol against its own
    trailing window. That suits the time-series strategies spot data is
    traded with; the US-equity sibling ``Alpha101Stock`` deliberately emits
    raw values because it serves cross-sectional strategies. The two are not
    meant to be aligned.

    Examples
    --------
    >>> factor = Alpha101SpotKline(FactorConfig(
    ...     window=20, dataset=dataset, mode="batch",
    ...     data_columns=["open", "high", "low", "close", "volume", "amount"],
    ...     factor_names=["alpha001", "alpha002"], file_path="alpha101.zarr",
    ... ))
    >>> panel = factor.cal().get_features()
    """

    def __init__(self, factor_config: FactorConfig):
        """Create the factor from a KunQuant factor config."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the graph: one rolling z-scored ``Output`` per requested alpha."""
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
        """Return the names of every alpha in KunQuant's ``Alpha101`` library."""
        factors = [alpha.__name__ for alpha in Alpha101.all_alpha]
        return tuple(factors)

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        """Raise; this factor set produces features only."""
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the computed panel unchanged."""
        return data


class Alpha101Stock(FactorKunQuant):
    """Alpha101 factors over adjusted US-equity bars, emitted raw.

    Reads ``adjOpen``/``adjHigh``/``adjLow``/``adjClose``/``adjVolume``; list
    exactly these in ``data_columns``. The stock stores carry no dollar-volume
    column, so ``vwap`` is the adjusted typical price
    ``(adjHigh + adjLow + adjClose) / 3``, as in ``Alpha158Stock``. Outputs are not normalized: US equities
    are traded with cross-sectional strategies here, and a rolling
    time-series z-score would change how symbols compare on the same day, so
    normalization across symbols is left to the consumer.

    Examples
    --------
    >>> factor = Alpha101Stock(FactorConfig(
    ...     window=20, dataset=dataset, mode="batch",
    ...     data_columns=["adjOpen", "adjHigh", "adjLow", "adjClose",
    ...                   "adjVolume"],
    ...     file_path="alpha101_stock.zarr",
    ... ))
    >>> panel = factor.cal().get_features()
    """

    def __init__(self, factor_config: FactorConfig):
        """Create the factor from a KunQuant factor config."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the graph: one raw ``Output`` per requested alpha."""
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("adjClose")
            low = Input("adjLow")
            high = Input("adjHigh")
            vopen = Input("adjOpen")
            vol = Input("adjVolume")
            # KunQuant derives vwap from `amount` unless one is given, and the
            # stock stores carry no dollar volume; the adjusted typical price
            # keeps vwap on the adjusted scale, as in `Alpha158Stock`.
            vwap = (high + low + close) / 3.0
            all_data = Alpha101.AllData(
                low=low,
                high=high,
                close=close,
                open=vopen,
                volume=vol,
                vwap=vwap,
            )
            for alpha in Alpha101.all_alpha:
                if alpha.__name__ in factor_names:
                    Output(
                        alpha(all_data),
                        alpha.__name__,
                    )
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the names of every alpha in KunQuant's ``Alpha101`` library."""
        factors = [alpha.__name__ for alpha in Alpha101.all_alpha]
        return tuple(factors)

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        """Raise; this factor set produces features only."""
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the computed panel unchanged."""
        return data
