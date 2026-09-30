"""The US-equity cross-sectional stock-selection backtester.

``USEquityCrossectionSelectStockVectorBt`` is the concrete backtester of the
pipeline for daily US equities. It composes the ``US_EQUITY_MARKET`` price
conventions, the portfolio construction rule of its config that turns
predictions into target weights, and the vectorbt simulation engine
inherited from ``VectorBtBacktester``. A saved run's ``config.json`` rebuilds it through
``quantlab.utils.module.load_backtester_from_config``.
"""

import warnings

import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import rebalance_mask
from quantlab.base.backtest import MarketSpec
from quantlab.base.config import CrossSectionBacktestConfig
from quantlab.base.data import InsufficientHistoryError
from quantlab.base.portfolio import PortfolioConstructor

#: Price conventions for US equities. Orders fill at the split- and
#: dividend-adjusted open and the portfolio is valued at the adjusted close;
#: the unadjusted price columns never enter the profit and loss. The column
#: names are spelled only here, never inside a method body.
US_EQUITY_MARKET = MarketSpec(
    fill_price_column="adjOpen",
    valuation_price_column="adjClose",
    trading_days_per_year=252,
    session_minutes_per_day=390,
)


class USEquityCrossectionSelectStockVectorBt(VectorBtBacktester):
    """Daily US-equity backtester: a portfolio construction rule simulated with vectorbt.

    On every rebalance bar the config's ``constructor`` turns the
    predictions of that bar into target weights, for example the top
    ``top_n`` symbols with ``TopNConstructor``, and the targets are held
    until the next rebalance bar. A symbol is tradable at a bar as the price
    dataset's ``tradable_bars`` says (by default, when it has a fill price
    at that bar), and a held symbol that is not tradable stays locked. The
    class only wires the pieces together: the market conventions are
    ``MARKET``, the rule is the config's ``constructor``, and every engine
    behaviour comes from ``VectorBtBacktester``.

    Parameters
    ----------
    config : CrossSectionBacktestConfig
        The backtest configuration: price dataset, model, date window and
        output directory, plus ``rebalance_periods`` and ``constructor``.

    Raises
    ------
    TypeError
        If ``constructor`` is not a ``PortfolioConstructor``.
    ValueError
        If the constructor refuses the model's labels.

    Examples
    --------
    >>> from quantlab.base.config import CrossSectionBacktestConfig, TopNConfig
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> backtester = USEquityCrossectionSelectStockVectorBt(
    ...     CrossSectionBacktestConfig(
    ...         price_dataset=prices,  # a StockDataset over a Zarr store
    ...         model=model,  # a model whose checkpoint is given below
    ...         model_mode="load",
    ...         checkpoint="models/first_feature/checkpoint.joblib",
    ...         start_date="2024-02-12",
    ...         end_date="2024-03-11",
    ...         output_dir="runs",
    ...         rebalance_periods=5,
    ...         constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
    ...     )
    ... )
    >>> result = backtester.run()
    >>> sorted(p.name for p in result.run_dir.iterdir())
    ['config.json', 'equity.zarr', 'fingerprint.json', 'metrics.json',
     'report.html', 'settlements.json', 'weights.zarr']
    >>> result.weights["weight"].dims
    ('timestamp', 'symbol')
    """

    config_cls = CrossSectionBacktestConfig
    MARKET = US_EQUITY_MARKET

    def _validate_config(self) -> None:
        """Check the constructor against the model's labels at construction time.

        A label the rule needs and the model does not predict is reported
        before any data is read or any model trained. A config without a
        model (for ``run_weights()``) has no labels to check.
        """
        config = self.config
        if not isinstance(config.constructor, PortfolioConstructor):
            raise TypeError(
                f"{self.class_name}: constructor must be a PortfolioConstructor, "
                f"got {type(config.constructor).__name__}"
            )
        if config.model is not None:
            config.constructor.bind(config.model)

    def _generate_signals(
        self, predictions: xr.Dataset, prices: xr.Dataset
    ) -> xr.Dataset:
        """Turn the model's predictions into target weights through the constructor.

        Tradability at each bar comes from the price dataset's
        ``tradable_bars`` on the raw, not forward-filled, window panel; the
        rule also skips symbols without a finite prediction of the label it
        reads. The rule is handed the raw fill and valuation prices from the
        constructor's ``lookback_bars`` bars before the window
        (``_price_history``), for its return window and to model the
        holdings the way the simulation trades them, the dataset's
        ``delisting_bars`` of the window (the same marks the simulation
        settles), and the panels of its ``required_factors()`` over the
        window (``_required_factor_panels``). Bars the rule failed on are
        kept for ``_signal_metrics``.
        """
        dataset = self.config.price_dataset
        tradable = dataset.tradable_bars(prices, self.MARKET.fill_price_column)
        mask = rebalance_mask(prices.sizes["timestamp"], self.config.rebalance_periods)
        history = self._price_history(prices)
        weights = self.config.constructor.construct_panel(
            predictions,
            tradable,
            mask,
            fill_price=history[self.MARKET.fill_price_column],
            valuation_price=history[self.MARKET.valuation_price_column],
            delisted=dataset.delisting_bars(prices, self.MARKET.valuation_price_column),
            factors=self._required_factor_panels(prices),
        )
        self._failed_bars = list(weights.attrs.pop("failed_bars", []))
        self._events = dict(weights.attrs.pop("events", {}))
        return weights

    def _required_factor_panels(self, prices: xr.Dataset) -> xr.Dataset | None:
        """Return the constructor's ``required_factors()`` over the window, or None.

        Each factor is computed from the window's first to its last bar
        with ``Factor.compute``, which reads the factor's own
        ``warmup_bars`` before the window like a model's features, and the
        panels are merged onto the window's symbols. ``None`` when the
        constructor declares no factor.
        """
        factors = self.config.constructor.required_factors()
        if not factors:
            return None
        first, last = prices.timestamp.values[0], prices.timestamp.values[-1]
        panels = [factor.compute(first, last) for factor in factors]
        return xr.merge(panels, join="outer").reindex(symbol=prices.symbol.values)

    def _price_history(self, prices: xr.Dataset) -> xr.Dataset:
        """Return the raw fill and valuation prices over the window and its warm-up.

        The warm-up is the constructor's ``lookback_bars`` bars before the
        window's first bar, counted on the price dataset's calendar, so the
        first bar already has a full window of one-bar returns (the first
        return needs the bar before it, which the warm-up supplies); when
        the dataset holds fewer, a warning names the shortfall and the first
        windows are short. The panel is on the window's symbols.
        """
        dataset = self.config.price_dataset
        columns = [self.MARKET.fill_price_column, self.MARKET.valuation_price_column]
        first, last = prices.timestamp.values[0], prices.timestamp.values[-1]
        lookback = self.config.constructor.lookback_bars
        try:
            start = dataset.bar_before(first, lookback)
        except InsufficientHistoryError as exc:
            warnings.warn(
                f"{self.class_name}: {type(self.config.constructor).__name__} reads "
                f"{lookback} bar(s) of returns before the window but the price "
                f"dataset holds only {exc.available}; the first return windows are "
                f"short by {lookback - exc.available} bar(s).",
                UserWarning,
                stacklevel=2,
            )
            start = dataset.bar_before(first, exc.available)
        return (
            dataset.panel(start, last)[columns]
            .reindex(symbol=prices.symbol.values)
            .transpose("timestamp", "symbol")
            .load()
        )

    def _signal_metrics(self) -> dict:
        """Report the bars the constructor could not decide, and its events.

        ``{"portfolio_construction": {"failed_bar_count": n, "failed_bars":
        [...], <event>: {"count": m, "bars": [{"bar", "symbols"}, ...]}}}``,
        the bars as ISO timestamps; an event such as the optimiser's
        ``closed_without_risk`` appears only when it happened, ``count``
        being the number of symbols over all its bars.
        """
        failed = list(getattr(self, "_failed_bars", []))
        block = {"failed_bar_count": len(failed), "failed_bars": failed}
        for name, records in getattr(self, "_events", {}).items():
            block[name] = {
                "count": sum(len(record["symbols"]) for record in records),
                "bars": list(records),
            }
        return {"portfolio_construction": block}
