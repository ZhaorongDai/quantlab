"""The US-equity cross-sectional stock-selection backtester.

``USEquityCrossectionSelectStockVectorBt`` is the concrete backtester of the
pipeline for daily US equities. It composes the ``US_EQUITY_MARKET`` price
conventions, the portfolio construction rule of its config that turns
predictions into target weights, and the vectorbt simulation engine
inherited from ``VectorBtBacktester``. A saved run's ``config.json`` rebuilds it through
``quantlab.utils.module.load_backtester_from_config``.
"""

import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import next_bar_eligible, rebalance_mask
from quantlab.base.backtest import MarketSpec
from quantlab.base.config import CrossSectionBacktestConfig
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
    until the next rebalance bar. A symbol is eligible when it has a fill
    price at the next bar. The class only wires the pieces together: the
    market conventions are ``MARKET``, the rule is the config's
    ``constructor``, and every engine behaviour comes from
    ``VectorBtBacktester``.

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
    ['config.json', 'equity.zarr', 'fingerprint.json', 'liquidations.json',
     'metrics.json', 'report.html', 'weights.zarr']
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
            label_names = [
                str(name)
                for label in config.model.labels
                for name in label.get_factor_names()
            ]
            config.constructor.check_predictor(
                label_names, dict(config.model.label_scales)
            )

    def _generate_signals(
        self, predictions: xr.Dataset, prices: xr.Dataset
    ) -> xr.Dataset:
        """Turn the model's predictions into target weights through the constructor.

        A symbol is eligible when its next-bar fill price, read from the
        raw, not forward-filled, price panel, is finite; the rule also skips
        symbols without a finite prediction of the label it reads.
        """
        eligible = next_bar_eligible(prices[self.MARKET.fill_price_column])
        mask = rebalance_mask(prices.sizes["timestamp"], self.config.rebalance_periods)
        return self.config.constructor.construct_panel(predictions, eligible, mask)
