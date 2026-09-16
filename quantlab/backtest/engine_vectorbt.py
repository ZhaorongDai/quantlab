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
from vectorbt.generic.enums import DrawdownStatus

from quantlab.base.backtest import BaseBacktester, SimulationResult

#: `Drawdowns.records` 里 `status` 的「已修复」取值。`.records` 给的是 int，只有
#: `.records_readable` 才是 Active / Recovered 字符串，所以判定是数值比较。写成
#: 常量而不是裸 1：vectorbt 哪天把枚举重新编号，这里跟着变，而不是悄悄把「已修复」
#: 和「仍在回撤」标反（T-v6i-04）。
#:
#: 放在模块层而不是做类属性：`_drawdown_span` 因此只用到 `self._bar_label` 这一个
#: 成员，可以脱离整个回测器单独测，测的就只是「选哪一条记录」本身。
DRAWDOWN_RECOVERED = int(DrawdownStatus.Recovered)


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

    **两套交易统计（quick 260915-udx）。** vectorbt 默认的 exit trades 口径把每
    一次减仓都记成一笔独立的已平仓交易：等权调仓下，赢家每被削一刀就多算一笔
    盈利交易，胜率与盈亏比因此偏高。所以 `_engine_stats` 报两套——顶层的交易
    指标是 **lot 级**（exit trades），嵌套的 `positions` 子字典是 **持仓级**（一个
    标的从建仓到清空算一笔）。两套都是对的，衡量的东西不同：lot 级看的是每次
    调仓动作的质量，持仓级才是「选股选得对不对」。所以并列报出，谁也不替换谁。
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

    #: 只有这些指标由交易口径决定，`positions` 那一套只重算它们
    #: （quick 260915-udx）。组合级指标——收益、回撤、夏普、卡玛、欧米伽、索提诺、
    #: 暴露、费用、起止时间与净值——跟按 lot 还是按持仓切交易无关，重算一遍只会
    #: 多出一份可能与顶层漂移的副本。
    TRADE_STATS_METRICS = (
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

        trade_records = pf.trades.records_readable
        if len(trade_records) == 0:
            trades = xr.Dataset()
        else:
            trades = xr.Dataset(
                {
                    "symbol": ("trade", trade_records["Column"].astype(str).to_numpy()),
                    "entry_timestamp": (
                        "trade",
                        pd.to_datetime(trade_records["Entry Timestamp"]).to_numpy(),
                    ),
                    "exit_timestamp": (
                        "trade",
                        pd.to_datetime(trade_records["Exit Timestamp"]).to_numpy(),
                    ),
                    "pnl": ("trade", trade_records["PnL"].to_numpy(dtype=np.float64)),
                    "return": (
                        "trade",
                        trade_records["Return"].to_numpy(dtype=np.float64),
                    ),
                    "status": ("trade", trade_records["Status"].astype(str).to_numpy()),
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
            trades=trades,
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
        """整段的 vectorbt 统计，年化口径取自市场规格。

        返回两套交易统计（quick 260915-udx，口径的含义见类文档）：顶层是 vectorbt
        默认的 exit trades 口径（lot 级），`positions` 子字典是持仓级口径，只含
        `TRADE_STATS_METRICS` 那批受交易口径影响的指标。`positions` 的键与顶层
        同名，可以逐行对照。

        **切换口径只能用 `Portfolio.replace`。** `trades_type` 是 `Portfolio`
        构造函数的参数，不是 `from_orders` 的；`stats()` 与 `get_trades()` 都不接
        受按次传入的交易口径，传了会被静默忽略，于是得到一份与顶层逐字节相同、
        看起来却没问题的假 `positions`。`replace` 是实例级的，也不去动 vectorbt
        那个进程级的全局设置映射——改它会波及同进程里的每一个组合对象。

        两次 `stats()` 各自新建一个 settings dict，不共用同一个对象，免得其中一次
        调用改掉另一次要读的东西。
        """
        year_freq = self.MARKET.year_freq(simulation.bar_interval)  # type: ignore[union-attr]
        portfolio = simulation.native
        stats = portfolio.stats(  # type: ignore[union-attr]
            metrics=list(self.STATS_METRICS),
            settings=dict(year_freq=year_freq),
            silence_warnings=True,
        )
        positions = portfolio.replace(trades_type="positions").stats(  # type: ignore[union-attr]
            metrics=list(self.TRADE_STATS_METRICS),
            settings=dict(year_freq=year_freq),
            silence_warnings=True,
        )
        whole = stats.to_dict()
        whole["positions"] = positions.to_dict()
        return whole

    def _drawdown_span(self, simulation: SimulationResult) -> dict | None:
        """**最深**的那一次回撤的起止（quick 260915-v6i），给报告画三角用。

        和 `_engine_stats` 是同一形状的钩子：读 `simulation.native`，返回纯 Python
        值，所以「native 只由产出它的引擎读」这条规则没有被破坏。基类的默认实现
        返回 None。

        **按深度选，不按时长选。** 最深的那一次回撤和持续最久的那一次经常不是
        同一条记录（本仓库实测的一段净值：深度 `[-36.4%, -5.2%, -5.9%]`，时长
        `[1, 5, 1]` bar——最深的那条只有 1 个 bar，最久的那条有 5 个），所以这里
        只看 `valley_val / peak_val - 1`，绝不去碰 `max_duration()`。指标表里的
        Max Drawdown Duration 量的是「最久」，和这里标出来的可以是两回事，说明
        文字里写明了这一点。

        `bars` 取 `end_idx - start_idx`，也就是 **bar 数**：vectorbt 自己的
        duration 量的就是它，而 `max_duration()` 是 bar 数乘 freq 之后的
        Timedelta。页面上一律按交易日（bar 数）写，不写日历天。

        **不包 try/except（D-6）。** 拦截条件是显式的几条：没有记录、没有有限的
        深度、下标落在时间轴外。vectorbt 真改了 drawdowns 的形状，`_engine_stats`
        会先炸，那一步远在写报告之前，所以报告不该是发现它的地方。

        `.iloc[i]` 取一行会把整行强转成 float64（一行里混着 int 与 float 两类
        列），而 float 没法给 DatetimeIndex 定位，所以每一列各自按列取成数组，
        再把选中的那个元素 `int(...)`。
        """
        records = simulation.native.drawdowns.records  # type: ignore[union-attr]
        if len(records) == 0:
            return None

        peak = records["peak_val"].to_numpy(dtype=np.float64)
        valley = records["valley_val"].to_numpy(dtype=np.float64)
        # peak <= 0 的记录算不出有意义的百分比深度；先把分母置 NaN，除出来就是
        # NaN，既不用 try 也不会触发除零警告。
        depth = valley / np.where(peak > 0.0, peak, np.nan) - 1.0
        if not np.isfinite(depth).any():
            return None

        row = int(np.nanargmin(depth))
        start = int(records["start_idx"].to_numpy()[row])
        end = int(records["end_idx"].to_numpy()[row])
        status = int(records["status"].to_numpy()[row])

        timestamps = simulation.value.timestamp.values
        if not (0 <= start < timestamps.size and 0 <= end < timestamps.size):
            return None

        return {
            "start": self._bar_label(timestamps[start]),
            "end": self._bar_label(timestamps[end]),
            "bars": end - start,
            "depth": float(depth[row]),
            "recovered": status == DRAWDOWN_RECOVERED,
        }

    def _report_notes(self) -> list[str]:
        """基类那条说明，再加两条：交易口径（quick 260915-udx）与最深回撤（260915-v6i）。

        报告和 metrics.json 里两套交易指标并排出现，名字又都是「胜率」「盈亏比」
        这种一看就懂的词，读的人默认会把它们当成选股胜率。这条说明就是拦住这个
        误读的：顶层那批是 lot 级，`positions` 前缀那批才是持仓级。

        第二条同理拦另一个误读：净值上的三角标的是**最深**的那一次回撤，而指标表
        里的 Max Drawdown Duration 是**最久**的那一次，两者常常不是同一段；顺带
        写明三角之间的长度按交易日（bar 数）算，不是日历天（D-2、D-4）。

        **文本里不能出现尖括号、和号、双引号和单引号。** 每条说明都要过一次
        HTML 转义，而 `tests/test_backtest_persistence.py` 断言每条说明在
        report.html 里**逐字**出现；上面那几个字符里随便哪一个，都会让这条说明
        在页面上被改写成 HTML 实体、于是不再逐字相同，测试转红。所以英文缩写和
        所有格一律写全。
        """
        return super()._report_notes() + [
            "The trade metrics at the top level are vectorbt exit trades, that "
            "is lot level: every partial trim of a holding counts as its own "
            "closed trade, which inflates the win rate. The rows whose names "
            "begin with positions are the position level view, one entry to "
            "flat round trip per symbol.",
            "The two triangles on the equity curve mark the DEEPEST drawdown: "
            "the up triangle is the bar it started and the down triangle the "
            "bar it ended. Its length is counted in trading days, that is in "
            "bars, never in calendar days. The metric named Max Drawdown "
            "Duration measures the LONGEST drawdown instead, which is often a "
            "different episode.",
        ]

    def _period_returns_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """同一次模拟的收益截到 `ranges` 后，用 vectorbt 收益访问器算统计（D-34）。

        `Portfolio` 不能按时间切片（Pitfall 5），所以只切 `pf.returns()`：每段
        按**精确的** bar 时间戳 `.loc[Timestamp(start):Timestamp(end)]`（含两端），
        多段按时间顺序拼接。端点是 `_bar_label` 写出的标签，日期即午夜。不用
        字符串切片：`.loc["2024-01-02"]` 作终点会包含那一整天，日内数据上一个
        午夜 bar 的标签会把当天其余 bar 也切进来（代码审查 CR-01）。年化口径与
        整段统计一致，取自市场规格。截出来是空序列时报错：切片区间来自回测窗口
        的 bar，空序列说明调用方传错了区间。
        """
        returns = simulation.native.returns()  # type: ignore[union-attr]
        pieces = [
            returns.loc[pd.Timestamp(str(start)) : pd.Timestamp(str(end))]
            for start, end in ranges
        ]
        sliced = pd.concat(pieces) if len(pieces) > 1 else pieces[0]
        if sliced.empty:
            raise ValueError(
                f"{self.class_name}: no simulated returns inside {ranges}"
            )
        stats = sliced.vbt.returns(
            freq=pd.Timedelta(simulation.bar_interval),
            year_freq=self.MARKET.year_freq(simulation.bar_interval),  # type: ignore[union-attr]
        ).stats(silence_warnings=True)
        return stats.to_dict()
