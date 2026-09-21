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

    - `symbol`：标的名，**人读**的那个拼写（str）。价格库旁边有
      `.crsp_tickers.json` 时，是该标的在**成交那一天**的 ticker（13407 在
      2022-06-08 是 FB、次日是 META）；没有那份 sidecar 的库（Tiingo / Alpaca，
      或 03.11-09 之前转换的 CRSP 库）上，就是面板轴自己的拼写，一个字不变；
    - `axis_symbol`：面板 symbol 轴上的标签本身（str），机器身份。两个都记，
      因为署名与索引是两个问题：按名字回查面板会在改名那天查空；
    - `signal_timestamp`：发出平仓信号的调仓 bar（pd.Timestamp）；
    - `fill_timestamp`：成交 bar，即 t+1（pd.Timestamp）；
    - `price`：t+1 上 ffill 后的成交价，也就是该标的最后一个有限成交价（float）。

    调仓行混有 NaN 与有限值时，在交给 vectorbt 之前直接报错（Pitfall 3）：NaN
    在调仓行上的意思是「保持原仓位」，会占着资金悄悄挡住同一行的其余订单。

    **交易统计只有一套口径：持仓级（phase 03.8，D-02）。** 一笔交易 = 一个标的
    从建仓到清空（vectorbt 的 `positions` 口径），中途减仓不单独算一笔。为什么
    不用 vectorbt 默认的 exit trades（lot 级）口径：它把每一次减仓都记成一笔
    独立的已平仓交易，而等权调仓下只有**赢家**才需要被削回目标权重，于是胜率
    被系统性抬高——本仓库两次独立的真实运行实测都偏高约 6.5 个百分点（lot 级
    57.27%，持仓级 50.74% / 50.81%），而亏损一侧几乎不受影响（-6.28% vs
    -6.32%），正是「只有赢家会被削」所预测的形状。所以顶层直接报持仓级，不再
    并列报 lot 级，也不再有嵌套的 `positions` 子字典。整段「到底成交了多少次」
    由基类写入的 `whole["order_count"]` 回答，它数的是订单记录条数。
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

        # 持仓级口径（D-02）：与 `_engine_stats` 报的那一套、以及基类切片块里的
        # closed / open 计数是同一个口径，三者因此能对上账。下面读的六个字段在
        # positions 记录里同名同义，Closed / Open 状态词也一样。
        trade_records = pf.positions.records_readable
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
            delisted = np.flatnonzero(
                (np.abs(held[t]) > tolerance) & np.isnan(raw_fill[t + 1])
            )
            if delisted.size == 0:
                continue
            # 强平记录是**人读**的产物（日志 + liquidations.json），而 PERMNO 轴
            # 上的 `str(symbols[j])` 是一串裸数字。按**成交那一天**查名字：退市
            # 当天的拼写才是这条记录该署的名，用今天的名字去署十年前的记录正是
            # 区间表存在的理由。一次查一整批，不是一行查一次。
            fill_day = pd.Timestamp(timestamps[t + 1]).date()
            named = self.ticker_lookup.label(
                [symbols[j] for j in delisted], fill_day
            )
            for j, name in zip(delisted, named):
                record = {
                    "symbol": name,
                    "axis_symbol": str(symbols[j]),
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

        **一次 `stats()` 调用，交易统计是持仓级口径（D-02，口径的理由见类文档）。**
        先 `replace(trades_type="positions")` 换掉交易口径，再算一次全套
        `STATS_METRICS`。一次就够，实测依据：换成持仓口径后，那 14 个真正组合级
        的指标（收益、回撤、夏普、卡玛、欧米伽、索提诺、暴露、费用、起止时间与
        净值）逐字节不变，只有受交易口径影响的那批变成持仓级——既然不再报 lot
        级那一套，第二次调用没有任何东西可买。

        **切换口径只能用 `Portfolio.replace`。** `trades_type` 是 `Portfolio`
        构造函数的参数，不是 `from_orders` 的；`stats()` 与 `get_trades()` 都不接
        受按次传入的交易口径，传了会被静默忽略，于是得到一份与 exit trades 逐字节
        相同、看起来却没问题的假结果。`replace` 是实例级的，也不去动 vectorbt
        那个进程级的全局设置映射——改它会波及同进程里的每一个组合对象。

        settings dict 现场新建，不共用同一个对象。
        """
        year_freq = self.MARKET.year_freq(simulation.bar_interval)  # type: ignore[union-attr]
        portfolio = simulation.native.replace(trades_type="positions")  # type: ignore[union-attr]
        stats = portfolio.stats(
            metrics=list(self.STATS_METRICS),
            settings=dict(year_freq=year_freq),
            silence_warnings=True,
        )
        return stats.to_dict()

    def _drawdown_span(self, simulation: SimulationResult) -> dict | None:
        """**最深**的那一次回撤：从**最低点**到修复（quick 260916-hro），给报告画三角用。

        和 `_engine_stats` 是同一形状的钩子：读 `simulation.native`，返回纯 Python
        值，所以「native 只由产出它的引擎读」这条规则没有被破坏。基类的默认实现
        返回 None。

        **按深度选，不按时长选。** 最深的那一次回撤和持续最久的那一次经常不是
        同一条记录（本仓库实测的一段净值：深度 `[-36.4%, -5.2%, -5.9%]`，时长
        `[1, 5, 1]` bar——最深的那条只有 1 个 bar，最久的那条有 5 个），所以这里
        只看 `valley_val / peak_val - 1`，绝不去碰 `max_duration()`。

        **标出来的这一段是「最低点 -> 修复」，不是「开始 -> 修复」。** 向上三角落在
        `valley_idx`（这一段里最深的那个 bar），向下三角仍落在 `end_idx`；
        `bars` 取 `end_idx - valley_idx`，单位是**交易日（bar 数）**，页面上一律
        这么写，不写日历天。

        **这个数不是 `Max Drawdown Duration`，而且通常比它小。** 两条理由彼此
        独立：一是那个指标量的是「最久」的那一次回撤，和这里选中的「最深」那次
        经常不是同一段；二是就算碰巧是同一段，那个指标从回撤**开始**算起，而这里
        从**最低点**算起。所以这两个数对不上是正常的。

        这一条**推翻了** quick 260915-v6i 自己的第三条决定：那次特意取 `start_idx`，
        为的就是让两个三角之间的距离正好等于 `Max Drawdown Duration`。直接问过
        之后用户选了最低点——他要看的是「从底部回到本金要多久」，而不是「这段回撤
        是从哪个 bar 开始的」。（那条决定属于 260915-v6i 自己的编号，与本次
        260916-hro 的 D-03 无关。）

        **不包 try/except（D-6）。** 拦截条件是显式的几条：没有记录、没有有限的
        深度、下标落在时间轴外（`valley_idx` 与 `end_idx` 都查）。vectorbt 真改了
        drawdowns 的形状，`_engine_stats` 会先炸，那一步远在写报告之前，所以报告
        不该是发现它的地方。

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
        # `valley` 上面已经被 valley_val 那一列占了名字，所以下标叫 valley_idx。
        # `start_idx` 不再读：payload 里没有它，而对一个从不解引用的下标做边界
        # 检查，只会让本来画得出来的一段变成 None。
        valley_idx = int(records["valley_idx"].to_numpy()[row])
        end = int(records["end_idx"].to_numpy()[row])
        status = int(records["status"].to_numpy()[row])

        timestamps = simulation.value.timestamp.values
        if not (0 <= valley_idx < timestamps.size and 0 <= end < timestamps.size):
            return None

        return {
            "valley": self._bar_label(timestamps[valley_idx]),
            "end": self._bar_label(timestamps[end]),
            "bars": end - valley_idx,
            "depth": float(depth[row]),
            "recovered": status == DRAWDOWN_RECOVERED,
        }

    def _report_notes(self) -> list[str]:
        """基类那条说明，再加两条：交易口径（D-02）与最深回撤（260915-v6i）。

        「胜率」「盈亏比」这种词一看就懂，读的人会默认它就是选股胜率，却不会想到
        「一笔交易」还有怎么切的问题。这条说明就是把口径写在页面上：交易指标是
        持仓级，中途减仓不单独算一笔；顺带点明 `order_count` 数的才是真实成交
        笔数。

        第二条同理拦另一个误读：净值上的三角标的是**最深**的那一次回撤，向上三角
        是它的**最低点**、向下三角是它修复的那个 bar，所以两者之间量的是「从底部
        回到本金要多久」。而指标表里的 Max Drawdown Duration 是**最久**的那一次、
        且从回撤开始算起，两个数对不上是正常的；顺带写明三角之间的长度按交易日
        （bar 数）算，不是日历天（D-2、D-4）。

        **文本里不能出现尖括号、和号、双引号和单引号。** 每条说明都要过一次
        HTML 转义，而 `tests/test_backtest_persistence.py` 断言每条说明在
        report.html 里**逐字**出现；上面那几个字符里随便哪一个，都会让这条说明
        在页面上被改写成 HTML 实体、于是不再逐字相同，测试转红。所以英文缩写和
        所有格一律写全。
        """
        return super()._report_notes() + [
            "The trade metrics are the position level view: one entry to flat "
            "round trip per symbol, so a partial trim of a holding is not "
            "counted as its own closed trade. Counting every trim as a closed "
            "trade is what vectorbt does by default, and it inflates the win "
            "rate. The row named order_count is the number of fills that "
            "actually happened over the window.",
            "The two triangles on the equity curve mark the DEEPEST drawdown: "
            "the up triangle is its deepest bar, that is its valley, and the "
            "down triangle is the bar it recovered. The distance between them "
            "is how long it took to get from the bottom back to even, counted "
            "in trading days, that is in bars, never in calendar days. It is "
            "not the metric named Max Drawdown Duration, which measures the "
            "LONGEST drawdown and counts from where that drawdown began, so "
            "the two numbers usually differ.",
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
