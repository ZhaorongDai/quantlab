"""美股截面选股回测（03.7 D-01、D-04）。"""

import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.selection import (
    CrossSectionTopNSelector,
    rebalance_mask,
    resolve_score_label,
)
from quantlab.base.backtest import MarketSpec
from quantlab.base.config import CrossSectionBacktestConfig

#: 美股的市场规格。成交价与估值价都用 Tiingo EOD 的拆股、分红复权列；
#: 不复权的开盘/收盘价从不参与盈亏计算（D-04）。列名只在这里出现。
US_EQUITY_MARKET = MarketSpec(
    fill_price_column="adjOpen",
    valuation_price_column="adjClose",
    trading_days_per_year=252,
    session_minutes_per_day=390,
)


class USEquityCrossectionSelectStockVectorBt(VectorBtBacktester):
    """美股 + 截面 TopN 选股 + vectorbt 引擎的具名组合（D-01）。

    每个调仓 bar 按模型预测从价格数据集的全部标的里选 `top_n` 个，等权持有到
    下一个调仓 bar。本类只做组合：市场规格是 `MARKET`，选股是一个
    `CrossSectionTopNSelector`，引擎行为全部继承自 `VectorBtBacktester`。
    """

    config_cls = CrossSectionBacktestConfig
    MARKET = US_EQUITY_MARKET

    def _validate_config(self) -> None:
        """打分标签与选股组件在构造期就解析，错配置在任何训练之前报错。"""
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
        """打分 -> TopN 目标权重。

        下一 bar 成交价取自**未 ffill** 的原始价格，已退市的标的因此不可选（D-12）。
        """
        scores = predictions[self._score_label]
        next_fill = prices[self.MARKET.fill_price_column].shift(timestamp=-1)
        mask = rebalance_mask(prices.sizes["timestamp"], self.config.rebalance_periods)
        return self._selector.select(scores, next_fill, mask)
