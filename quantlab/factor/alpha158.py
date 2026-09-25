"""Alpha158 factor sets computed with the KunQuant backend.

KunQuant's ``Alpha158`` feature library (k-bar shape features, lagged price
and volume ratios, and rolling statistics) is exposed as two
``FactorKunQuant`` subclasses: ``Alpha158SpotKline`` for crypto spot klines,
with rolling z-score normalization, and ``Alpha158Stock`` for adjusted
US-equity bars, emitted raw.
"""

from typing import NoReturn

import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha158
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig
from quantlab.base.factor import FactorKunQuant
from quantlab.my_ops.preprocess import WindowedZScore


class Alpha158SpotKline(FactorKunQuant):
    """Alpha158 factors over crypto spot klines, z-scored along time.

    Builds the Alpha158 feature set from the lowercase ``open``/``high``/
    ``low``/``close``/``volume``/``amount`` columns: k-bar shape features,
    price and volume ratios lagged 0 to 4 bars, and rolling features over
    5/10/20/30/60 bars (``BETA``, ``RSQR`` and ``RESI`` excluded). Every
    output is wrapped in ``WindowedZScore`` over ``config.window`` bars, a
    time-series normalization chosen because spot data is traded with
    time-series strategies; ``Alpha158Stock`` deliberately emits raw values
    for cross-sectional strategies.

    Pin ``factor_names`` to a few columns while experimenting: the full set
    is over a hundred columns and compile time grows with the graph.

    Examples
    --------
    >>> factor = Alpha158SpotKline(FactorConfig(
    ...     window=10, dataset=dataset, mode="batch",
    ...     data_columns=["open", "high", "low", "close", "volume", "amount"],
    ...     factor_names=["KMID", "STD5"], file_path="alpha158.zarr",
    ... ))
    >>> panel = factor.cal().get_features()
    """

    def __init__(self, factor_config: FactorConfig):
        """Create the factor from a KunQuant factor config."""
        super().__init__(factor_config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the names of every feature the Alpha158 build produces."""
        return tuple(self._factor_names_stream())

    def _get_func_names(self):
        """Build the Alpha158 op list and its names from fresh ``Input`` nodes.

        Returns
        -------
        tuple[list, list[str]]
            A ``(ops, names)`` pair in matching order.
        """
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
                "kbar": {},  # k-bar shape features
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
                "rolling": {  # rolling-window features
                    "windows": [5, 10, 20, 30, 60],  # window lengths in bars
                    "exclude": ["BETA", "RSQR", "RESI"],
                },
            }
        )
        return alpha158, names

    def _factor_names_stream(self):
        """Return the feature names from a fresh Alpha158 build."""
        return self._get_func_names()[-1]

    def _get_func_stream(self) -> Function:
        """Build the graph: one rolling z-scored ``Output`` per requested feature."""
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
        """Return the graph built by ``_get_func_stream``."""
        return self._get_func_stream()

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        """Raise; this factor set produces features only."""
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the computed panel unchanged."""
        return data


class Alpha158Stock(FactorKunQuant):
    """Alpha158 factors over adjusted US-equity bars, emitted raw.

    Reads only the adjusted series ``adjOpen``/``adjHigh``/``adjLow``/
    ``adjClose``/``adjVolume``; list exactly these in ``data_columns``. The
    ``VWAP`` features use the adjusted typical price
    ``(adjHigh + adjLow + adjClose) / 3`` rather than ``amount / volume``:
    the stock stores carry no dollar volume, and dividing a raw amount by a
    split-adjusted volume would jump at every split. Outputs are not
    normalized, for the same reason as ``Alpha101Stock``: cross-sectional
    normalization is left to the consumer.

    Examples
    --------
    >>> factor = Alpha158Stock(FactorConfig(
    ...     window=10, dataset=dataset, mode="batch",
    ...     data_columns=["adjOpen", "adjHigh", "adjLow", "adjClose",
    ...                   "adjVolume"],
    ...     factor_names=["KMID", "STD5"], file_path="alpha158_stock.zarr",
    ... ))
    >>> panel = factor.cal().get_features()
    """

    def __init__(self, factor_config: FactorConfig):
        """Create the factor from a KunQuant factor config."""
        super().__init__(factor_config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the names of every feature the Alpha158 build produces."""
        return tuple(self._factor_names_stream())

    def _get_func_names(self):
        """Build the Alpha158 op list and its names from adjusted inputs.

        Returns
        -------
        tuple[list, list[str]]
            A ``(ops, names)`` pair in matching order.
        """
        close = Input("adjClose")
        low = Input("adjLow")
        high = Input("adjHigh")
        vopen = Input("adjOpen")
        vol = Input("adjVolume")
        # No stock store carries a dollar-volume column, and a raw amount
        # divided by the split-adjusted volume would jump at every split. The
        # adjusted typical price keeps VWAP features on the adjusted scale.
        vwap = (high + low + close) / 3.0
        all_data = Alpha158.AllData(
            low=low,
            high=high,
            close=close,
            open=vopen,
            volume=vol,
            vwap=vwap,
        )
        # Assigned explicitly as a guard: KunQuant's `AllData.__init__` has
        # not always kept a `vwap=` passed alongside a missing `amount`.
        all_data.vwap = vwap
        alpha158, names = all_data.build(
            {
                "kbar": {},  # k-bar shape features
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
                "rolling": {  # rolling-window features
                    "windows": [5, 10, 20, 30, 60],  # window lengths in bars
                    "exclude": ["BETA", "RSQR", "RESI"],
                },
            }
        )
        return alpha158, names

    def _factor_names_stream(self):
        """Return the feature names from a fresh Alpha158 build."""
        return self._get_func_names()[-1]

    def _get_func_stream(self) -> Function:
        """Build the graph: one raw ``Output`` per requested feature."""
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            alpha158, names = self._get_func_names()
            for v, k in zip(alpha158, names):
                if k in factor_names:
                    Output(v, k)
        return Function(builder.ops)

    def _get_factor_func(self):
        """Return the graph built by ``_get_func_stream``."""
        return self._get_func_stream()

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        """Raise; this factor set produces features only."""
        raise RuntimeError(f"{__class__.__name__} does not support get_label()")

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the computed panel unchanged."""
        return data
