"""vectorbt 引擎层。

文件名刻意叫 `engine_vectorbt.py`：叫 `vectorbt.py` 会在包内遮蔽顶层的
`vectorbt` 包（与 `quantlab/ml_model/xgb.py` 同一规则，03.7-RESEARCH.md
Pitfall 12）。整个回测层只有这个模块 import vectorbt。
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
import xarray as xr

from quantlab.base.backtest import BaseBacktester, SimulationResult


class VectorBtBacktester(BaseBacktester):
    """vectorbt 引擎（D-01 的继承那一层）；仍是抽象类。

    `config_cls`、`MARKET` 与 `_generate_signals` 由具体类提供。三条引擎语义：

    - **成交时点**：bar t 收盘形成的信号在 bar t+1 开盘成交（D-05），实现方式
      是把权重整体后移一个 bar 再交给 `Portfolio.from_orders`。
    - **目标百分比的基数**：按下单价（t+1 开盘价）给整组资产估值后计算，这是
      vectorbt `val_price` 的默认行为（03.7-RESEARCH.md Pitfall 4）。
    - **不计融券/空头融资成本**（D-21），所以空头一侧的收益是偏乐观的。

    价格喂给引擎之前两列都做 ffill（D-07）：持仓标的的价格一旦变成 NaN，
    vectorbt 会冻结整组之后的**所有**调仓且不报错（Pitfall 2）。
    """

    #: vectorbt `Portfolio.stats` 的指标名，去掉了 `benchmark_return`（D-08：
    #: 没有基准时报告不带基准指标，Pitfall 6）。
    STATS_METRICS = (
        "start",
        "end",
        "period",
        "start_value",
        "end_value",
        "total_return",
        "max_gross_exposure",
        "total_fees_paid",
        "max_dd",
        "max_dd_duration",
        "total_trades",
        "total_closed_trades",
        "total_open_trades",
        "open_trade_pnl",
        "win_rate",
        "best_trade",
        "worst_trade",
        "avg_winning_trade",
        "avg_losing_trade",
        "avg_winning_trade_duration",
        "avg_losing_trade_duration",
        "profit_factor",
        "expectancy",
        "sharpe_ratio",
        "calmar_ratio",
        "omega_ratio",
        "sortino_ratio",
    )

    def _simulate(self, weights: xr.Dataset, prices: xr.Dataset) -> SimulationResult:
        """权重 -> `Portfolio.from_orders`；pandas 只活在这个方法里。"""
        cfg = self.config
        market = self.MARKET

        timestamps = prices.timestamp.values
        if timestamps.size < 2:
            raise ValueError(
                f"{self.class_name}: at least two price bars are needed to "
                f"simulate, got {timestamps.size}"
            )
        bar_interval = (
            pd.Series(np.diff(timestamps)).mode().iloc[0].to_timedelta64()
        )

        w = weights["weight"].transpose("timestamp", "symbol").to_pandas()
        fill = (
            prices[market.fill_price_column]  # type: ignore[union-attr]
            .transpose("timestamp", "symbol")
            .to_pandas()
            .ffill()
        )
        valuation = (
            prices[market.valuation_price_column]  # type: ignore[union-attr]
            .transpose("timestamp", "symbol")
            .to_pandas()
            .ffill()
        )

        pf = vbt.Portfolio.from_orders(
            close=valuation,
            price=fill,
            size=w.shift(1),
            size_type="targetpercent",
            direction="both",
            group_by=True,
            cash_sharing=True,
            call_seq="auto",
            fees=cfg.fees,
            slippage=cfg.slippage,
            init_cash=cfg.init_cash,
            freq=pd.Timedelta(bar_interval),
        )

        value = pf.value()
        returns = pf.returns()
        value_da = xr.DataArray(
            np.asarray(value.to_numpy(), dtype=np.float64),
            dims=("timestamp",),
            coords={"timestamp": value.index.to_numpy()},
        )
        returns_da = xr.DataArray(
            np.asarray(returns.to_numpy(), dtype=np.float64),
            dims=("timestamp",),
            coords={"timestamp": returns.index.to_numpy()},
        )

        records = pf.orders.records_readable
        orders = xr.Dataset(
            {
                "timestamp": ("order", pd.to_datetime(records["Timestamp"]).to_numpy()),
                "symbol": ("order", records["Column"].astype(str).to_numpy()),
                "size": ("order", records["Size"].to_numpy(dtype=np.float64)),
                "price": ("order", records["Price"].to_numpy(dtype=np.float64)),
                "fees": ("order", records["Fees"].to_numpy(dtype=np.float64)),
                "side": ("order", records["Side"].astype(str).to_numpy()),
            }
        )

        return SimulationResult(
            value=value_da,
            returns=returns_da,
            orders=orders,
            liquidations=[],
            bar_interval=bar_interval,
            native=pf,
        )

    def _simulate_benchmark(
        self, start_date: str, end_date: str
    ) -> SimulationResult | None:
        """本阶段没有基准：非 None 的 `benchmark_dataset` 已在基类 setter 被拒（D-08）。"""
        return None

    def _engine_stats(self, simulation: SimulationResult) -> dict:
        """整段的 vectorbt 统计，年化口径取自市场规格。"""
        stats = simulation.native.stats(  # type: ignore[union-attr]
            metrics=list(self.STATS_METRICS),
            settings=dict(
                year_freq=self.MARKET.year_freq(simulation.bar_interval)  # type: ignore[union-attr]
            ),
            silence_warnings=True,
        )
        return stats.to_dict()
