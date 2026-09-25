"""The US-equity cross-sectional stock-selection backtester.

``USEquityCrossectionSelectStockVectorBt`` is the concrete backtester of the
pipeline for daily US equities. It composes the ``US_EQUITY_MARKET`` price
conventions, a ``CrossSectionTopNSelector`` that turns model scores into
target weights, and the vectorbt simulation engine inherited from
``VectorBtBacktester``. A saved run's ``config.json`` rebuilds it through
``quantlab.utils.module.load_backtester_from_config``.
"""

import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import (
    CrossSectionTopNSelector,
    rebalance_mask,
    resolve_score_label,
)
from quantlab.base.backtest import MarketSpec
from quantlab.base.config import CrossSectionBacktestConfig

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
    """Daily US-equity backtester: top-N stock selection simulated with vectorbt.

    On every rebalance bar the model's predictions rank every symbol in the
    price dataset, the selector picks the top ``top_n`` (and, for
    ``direction="long_short"``, shorts the bottom ``top_n``), and the
    equal-weight targets are held until the next rebalance bar. The class only
    wires the pieces together: the market conventions are ``MARKET``, the
    selection rule is a ``CrossSectionTopNSelector`` built from the config, and
    every engine behaviour comes from ``VectorBtBacktester``. It is configured
    with a ``CrossSectionBacktestConfig``.

    Example:
        >>> from quantlab.base.config import CrossSectionBacktestConfig
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
        ...         direction="long_only",
        ...         top_n=2,
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
        """Resolve the score label and build the selector at construction time.

        Both come from the config once, so a label the model does not declare
        or an invalid ``direction`` / ``top_n`` is reported before any data is
        read or any model trained.
        """
        config = self.config
        self._score_label = resolve_score_label(
            config.score_label, list(config.model.get_label_names())
        )
        self._selector = CrossSectionTopNSelector(
            direction=config.direction, top_n=config.top_n
        )

    def _generate_signals(
        self, predictions: xr.Dataset, prices: xr.Dataset
    ) -> xr.Dataset:
        """Turn the model's predictions into top-N target weights.

        The scores are the predictions of the resolved score label. The
        next-bar fill price comes from the raw, not forward-filled, price
        panel, so a symbol with no price on the next bar is ineligible on this
        one.
        """
        scores = predictions[self._score_label]
        next_fill = prices[self.MARKET.fill_price_column].shift(timestamp=-1)
        mask = rebalance_mask(prices.sizes["timestamp"], self.config.rebalance_periods)
        return self._selector.select(scores, next_fill, mask)
