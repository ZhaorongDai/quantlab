"""A vectorbt backtester of given target weights, its market conventions in its config.

``WeightsVectorBt`` is the backtester ``quantlab.api.backtest`` runs: a caller's frame
names its own price columns and its own market, so the ``MarketSpec`` comes from the
``WeightsBacktestConfig`` rather than from a class constant. It has no signal rule of its
own and takes no model; its only entry point is ``run_weights()``. A run given an
``output_dir`` writes the usual run directory (``quantlab.runs.backtest_run``), whose
recipe names this class with the market fields in its config and whose copy of a price
or benchmark ``FrameDataset`` is named relative to the run directory. The directory is
therefore self-contained: ``BacktestRun.open(run_dir).rebuild_backtester()`` rebuilds
the backtester, and ``run_weights`` given the run's ``weights()`` replays the run.
"""

import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.base import MarketSpec
from quantlab.backtest.config import WeightsBacktestConfig


class WeightsVectorBt(VectorBtBacktester):
    """Backtest given target weights with vectorbt, on the market its config describes.

    ``MARKET`` is built from the config's ``fill_price_column``,
    ``valuation_price_column``, ``trading_days_per_year`` and
    ``session_minutes_per_day`` whenever a config is assigned. Everything else is
    ``VectorBtBacktester``'s: a weight formed at bar t fills at bar t+1's fill price,
    and the metrics cover the whole window.

    Parameters
    ----------
    config : WeightsBacktestConfig
        The price dataset, window, costs, optional benchmark and market conventions;
        ``model`` and ``model_mode`` must be ``None``.

    Raises
    ------
    TypeError
        If ``config`` is not a ``WeightsBacktestConfig``.
    ValueError
        If the config carries a model, a price column name is empty, the day or session
        length is below 1, or the recorded ``top_n`` or ``direction`` is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr
    >>> from quantlab.backtest.predefined.weights import WeightsVectorBt
    >>> from quantlab.backtest.config import WeightsBacktestConfig
    >>> from quantlab.dataset.memory import FrameDataset
    >>> bars = pd.bdate_range("2024-01-01", periods=5)
    >>> prices = FrameDataset(pd.DataFrame({
    ...     "timestamp": np.repeat(bars, 2),
    ...     "symbol": ["AAA", "BBB"] * 5,
    ...     "open": [10.0, 20.0, 11.0, 20.0, 12.0, 21.0, 12.0, 22.0, 13.0, 22.0],
    ...     "close": [10.5, 20.0, 11.5, 20.5, 12.0, 21.5, 12.5, 22.0, 13.0, 22.5],
    ... }))
    >>> backtester = WeightsVectorBt(WeightsBacktestConfig(
    ...     price_dataset=prices, start_date="2024-01-01", end_date="2024-01-05",
    ...     output_dir=None, rebalance_periods=1, fees=0.0, slippage=0.0,
    ...     fill_price_column="open", valuation_price_column="close",
    ...     trading_days_per_year=252, session_minutes_per_day=390,
    ... ))
    >>> backtester.MARKET.fill_price_column
    'open'
    >>> weights = xr.DataArray(
    ...     [[1.0, 0.0]] + [[np.nan, np.nan]] * 4, dims=("timestamp", "symbol"),
    ...     coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
    ... )
    >>> result = backtester.run_weights(weights)
    >>> result.simulation.value.values.round(2).tolist()
    [1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]

    Kept on disk, the run rebuilds from its directory and replays its weights:

    >>> import dataclasses, tempfile
    >>> from quantlab.runs.backtest_run import BacktestRun
    >>> kept = WeightsVectorBt(
    ...     dataclasses.replace(backtester.config, output_dir=tempfile.mkdtemp())
    ... ).run_weights(weights)
    >>> run = BacktestRun.open(kept.run_dir)
    >>> replay = run.rebuild_backtester().run_weights(run.weights())
    >>> replay.simulation.value.values.round(2).tolist()
    [1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]
    """

    config_cls = WeightsBacktestConfig

    def __init__(self, config: WeightsBacktestConfig):
        """Build ``MARKET`` from ``config``, then assign it; see the class docstring."""
        # The config setter refuses a backtester without a MARKET before it
        # validates anything, so the spec must exist before the first
        # assignment. A config of the wrong class is left to the setter's type check.
        if isinstance(config, WeightsBacktestConfig):
            self.MARKET = self._market_of(config)
        super().__init__(config)

    @staticmethod
    def _market_of(config: WeightsBacktestConfig) -> MarketSpec:
        """Return the market conventions ``config`` describes."""
        return MarketSpec(
            fill_price_column=config.fill_price_column,
            valuation_price_column=config.valuation_price_column,
            trading_days_per_year=config.trading_days_per_year,
            session_minutes_per_day=config.session_minutes_per_day,
        )

    def _validate_config(self) -> None:
        """Refuse a model and invalid market fields, then rebuild ``MARKET``.

        Raises
        ------
        ValueError
            See the class docstring.
        """
        config = self.config
        if config.model is not None:
            raise ValueError(
                f"{self.class_name} backtests given weights only, through "
                f"run_weights(); config.model must be None, got "
                f"{type(config.model).__name__}"
            )
        for name in ("fill_price_column", "valuation_price_column"):
            value = getattr(config, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{self.class_name}: {name} must be a non-empty column name, got "
                    f"{value!r}"
                )
        for name in ("trading_days_per_year", "session_minutes_per_day"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"{self.class_name}: {name} must be an integer >= 1, got {value!r}"
                )
        if config.direction not in (None, "long_only", "long_short"):
            raise ValueError(
                f"{self.class_name}: direction must be 'long_only', 'long_short' or "
                f"None, got {config.direction!r}"
            )
        top_n = config.top_n
        if top_n is not None and (
            isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1
        ):
            raise ValueError(
                f"{self.class_name}: top_n must be an integer >= 1 or None, got {top_n!r}"
            )
        self.MARKET = self._market_of(config)

    def _generate_signals(
        self, predictions: xr.Dataset, prices: xr.Dataset, delisted: xr.DataArray
    ) -> xr.Dataset:
        """Refuse: this backtester has no signal rule; weights come to ``run_weights``.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(
            f"{self.class_name} turns no predictions into weights; backtest given "
            f"weights with run_weights()"
        )
