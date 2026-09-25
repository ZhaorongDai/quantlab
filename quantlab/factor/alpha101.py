"""Alpha101 factor sets computed with the KunQuant backend.

"101 Formulaic Alphas" (Kakushadze, 2016) is a public list of 101 short
trading-signal formulas built from open, high, low, close, volume and VWAP.
KunQuant, the library this project uses for most factor computation, ships
them as its ``Alpha101`` library. KunQuant compiles a factor formula, written
as a graph of operators, to native code and runs it over a whole
``(timestamp, symbol)`` panel at once.

Two ``FactorKunQuant`` subclasses expose the library. They differ in which
input columns they read and in whether the outputs are normalized:
``Alpha101SpotKline`` works on crypto spot klines (candlestick bars) and
z-scores every output along time, while ``Alpha101Stock`` works on adjusted
US-equity bars and returns raw values.
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

    Reads the lowercase ``open``, ``high``, ``low``, ``close``, ``volume``
    and ``amount`` (traded value in quote currency) columns of the spot kline
    dataset. Every output is wrapped in ``WindowedZScore`` over
    ``config.window`` bars, which standardizes each symbol against its own
    trailing window. That time-series normalization suits the strategies
    spot data is traded with here, which follow one asset over time. The
    US-equity sibling ``Alpha101Stock`` returns raw values instead, because
    it serves cross-sectional strategies that compare symbols on the same
    bar. The two classes are intentionally different.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. ``window`` sets the z-score window,
        ``data_columns`` lists the six input columns above, and
        ``factor_names`` selects which alphas to compute (all 101 when
        unset).

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
        """Initialize the factor; see the class docstring for parameters."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph with one z-scored output per requested alpha."""
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
        """Raise ``RuntimeError``: this factor set produces features, not labels."""
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the computed panel unchanged; no post-processing is needed."""
        return data


class Alpha101Stock(FactorKunQuant):
    """Alpha101 factors over adjusted US-equity bars, returned raw.

    Reads ``adjOpen``, ``adjHigh``, ``adjLow``, ``adjClose`` and
    ``adjVolume``, the split- and dividend-adjusted series. Outputs are not
    normalized. US equities are traded here with cross-sectional strategies,
    and a per-symbol rolling z-score would change how symbols compare on the
    same day, so normalizing across symbols is left to the consumer.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. ``data_columns`` lists the five
        adjusted columns above and ``factor_names`` selects which alphas to
        compute (all 101 when unset).

    Notes
    -----
    The stock stores carry no dollar-volume column, so the graph passes no
    ``amount`` input. KunQuant's ``Alpha101.AllData`` would otherwise compute
    VWAP as ``amount / volume`` (and raise "Bad inputs" without it), so the
    graph passes the adjusted typical price
    ``(adjHigh + adjLow + adjClose) / 3`` as ``vwap``, as ``Alpha158Stock``
    does.

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
        """Initialize the factor; see the class docstring for parameters."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph with one raw output per requested alpha."""
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
        """Raise ``RuntimeError``: this factor set produces features, not labels."""
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the computed panel unchanged; no post-processing is needed."""
        return data
