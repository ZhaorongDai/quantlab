"""vectorbt 引擎层。

文件名刻意叫 `engine_vectorbt.py`：叫 `vectorbt.py` 会在包内遮蔽顶层的
`vectorbt` 包（与 `quantlab/ml_model/xgb.py` 同一规则，03.7-RESEARCH.md
Pitfall 12）。整个回测层只有这个模块 import vectorbt。
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
import xarray as xr
from loguru import logger

from quantlab.base.backtest import BaseBacktester, SimulationResult


class VectorBtBacktester(BaseBacktester):
    """vectorbt 引擎（D-01 的继承那一层）；仍是抽象类。

    `config_cls`、`MARKET` 与 `_generate_signals` 由具体类提供。三条引擎语义：

    - **成交时点**：bar t 收盘形成的信号在 bar t+1 开盘成交（D-05），实现方式
      是把权重整体后移一个 bar 再交给 `Portfolio.from_orders`。
    - **目标百分比的基数**：按下单价（t+1 开盘价）给整组资产估值后计算，这是
      vectorbt `val_price` 的默认行为（03.7-RESEARCH.md Pitfall 4）。
    - **不计融券/空头融资成本**（D-21），所以空头一侧的收益是偏乐观的。

    **退市规则（D-07）。** 价格喂给引擎之前，成交价与估值价两列都做 ffill。
    原因：持仓标的的价格一旦变成 NaN，vectorbt 会按最后价值继续持有它，并且
    **整组**之后的所有调仓都被悄悄跳过、不报错——冻结的是整组，不只是退市的
    那一只（Pitfall 2）。ffill 之后，退市标的在下一个调仓 bar 按最后价格被强制
    平仓，其余标的照常调仓。

    判定一次强制平仓：调仓行 t（且 t+1 仍在窗口内），t 收盘时该标的持仓非零
    （由订单记录累计得出，见 `_signed_order_sizes`），并且 t+1 的**原始**（未
    ffill）成交价是 NaN。开头价格为 NaN、此前从未持有的标的（晚上市）不算退市，
    上市后正常成交。每次强制平仓在 `SimulationResult.liquidations` 里记一条 dict：

    - `symbol`：标的名（str）；
    - `signal_timestamp`：发出平仓信号的调仓 bar（pd.Timestamp）；
    - `fill_timestamp`：成交 bar，即 t+1（pd.Timestamp）；
    - `price`：t+1 上 ffill 后的成交价，也就是该标的最后一个有限成交价（float）。

    调仓行混有 NaN 与有限值时，在交给 vectorbt 之前直接报错（Pitfall 3）：NaN
    在调仓行上的意思是「保持原仓位」，会占着资金悄悄挡住同一行的其余订单。
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
        # 基类的契约检查只在 run() 里跑；直接调 `_simulate` 的调用方会绕过它，
        # 所以引擎边界在任何 pandas 转换之前再断言一次（Pitfall 3）。
        self._refuse_mixed_weight_rows(weights)

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
        raw_fill = prices[market.fill_price_column].transpose(  # type: ignore[union-attr]
            "timestamp", "symbol"
        )
        fill = raw_fill.to_pandas().ffill()
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

        liquidations = self._forced_liquidations(
            weight_values=np.asarray(w.to_numpy(), dtype=np.float64),
            raw_fill=np.asarray(raw_fill.values, dtype=np.float64),
            filled_fill=np.asarray(fill.to_numpy(), dtype=np.float64),
            orders=orders,
            timestamps=timestamps,
            symbols=np.asarray(fill.columns),
        )

        return SimulationResult(
            value=value_da,
            returns=returns_da,
            orders=orders,
            liquidations=liquidations,
            bar_interval=bar_interval,
            native=pf,
        )

    def _refuse_mixed_weight_rows(self, weights: xr.Dataset) -> None:
        """每一行权重必须要么全 NaN（持有），要么全有限（调仓）。"""
        values = np.asarray(
            weights["weight"].transpose("timestamp", "symbol").values, dtype=np.float64
        )
        mixed = ~(np.isnan(values).all(axis=1) | np.isfinite(values).all(axis=1))
        if mixed.any():
            first = pd.Timestamp(weights.timestamp.values[int(np.argmax(mixed))])
            raise ValueError(
                f"{self.class_name}: weight row at {first} mixes NaN and finite "
                f"values; a rebalance row must be all-finite and a hold row "
                f"all-NaN (vectorbt reads NaN on a rebalance row as 'keep the "
                f"position' and silently blocks the rest of the rebalance, "
                f"03.7-RESEARCH.md Pitfall 3)"
            )

    @staticmethod
    def _signed_order_sizes(
        orders: xr.Dataset, timestamps: np.ndarray, symbols: np.ndarray
    ) -> xr.DataArray:
        """订单记录 -> 每个 bar 收盘后的累计带符号持仓，维度 `(timestamp, symbol)`。

        Buy 记 +size、Sell 记 -size，按 `timestamps` / `symbols` 轴累加。持仓只从
        订单记录推出，不读引擎的组合对象，因此与 vectorbt 如何分组无关。
        """
        ts = np.asarray(timestamps).astype("datetime64[ns]")
        syms = [str(s) for s in symbols]
        positions = np.zeros((ts.size, len(syms)), dtype=np.float64)

        if orders.sizes.get("order", 0) > 0:
            order_ts = orders["timestamp"].values.astype("datetime64[ns]")
            t_idx = np.searchsorted(ts, order_ts)
            if (t_idx >= ts.size).any() or not np.array_equal(
                ts[np.minimum(t_idx, ts.size - 1)], order_ts
            ):
                raise ValueError("an order timestamp is not on the price timestamp axis")
            column = {s: i for i, s in enumerate(syms)}
            s_idx = np.array([column[str(s)] for s in orders["symbol"].values])
            side = orders["side"].values.astype(str)
            unknown = sorted(set(side) - {"Buy", "Sell"})
            if unknown:
                raise ValueError(f"unknown order side(s) {unknown}")
            signed = np.where(side == "Buy", 1.0, -1.0) * orders["size"].values
            np.add.at(positions, (t_idx, s_idx), signed)

        return xr.DataArray(
            np.cumsum(positions, axis=0),
            dims=("timestamp", "symbol"),
            coords={"timestamp": ts, "symbol": syms},
        )

    def _forced_liquidations(
        self,
        weight_values: np.ndarray,
        raw_fill: np.ndarray,
        filled_fill: np.ndarray,
        orders: xr.Dataset,
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ) -> list[dict]:
        """D-07 的强制平仓记录；判定规则与记录字段见类文档。"""
        n_bars = timestamps.size
        held = self._signed_order_sizes(orders, timestamps, symbols).values
        sizes = np.abs(np.asarray(orders["size"].values, dtype=np.float64))
        # 买卖相抵后的浮点残差不算持仓：容差取最大单笔成交数量的 1e-9 倍。
        tolerance = 1e-9 * max(1.0, float(sizes.max(initial=0.0)))

        records = []
        for t in np.flatnonzero(np.isfinite(weight_values).all(axis=1)):
            if t + 1 >= n_bars:
                continue
            delisted = (np.abs(held[t]) > tolerance) & np.isnan(raw_fill[t + 1])
            for j in np.flatnonzero(delisted):
                record = {
                    "symbol": str(symbols[j]),
                    "signal_timestamp": pd.Timestamp(timestamps[t]),
                    "fill_timestamp": pd.Timestamp(timestamps[t + 1]),
                    "price": float(filled_fill[t + 1, j]),
                }
                logger.info(
                    f"{self.class_name}: forced liquidation of {record['symbol']} "
                    f"(D-07): signal {record['signal_timestamp']}, fill "
                    f"{record['fill_timestamp']} at last price {record['price']}"
                )
                records.append(record)
        return records

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
