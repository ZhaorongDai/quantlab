"""vectorbt simulation engine for the backtest layer.

``VectorBtBacktester`` implements the engine hooks of ``BaseBacktester`` on
top of ``vectorbt.Portfolio.from_orders``: target-percent weights, a one-bar
delay between signal and fill, rejected orders, delisting settlements, and
the whole-window portfolio statistics a run directory records. It is the
only module in the package that imports vectorbt; concrete backtesters such
as ``quantlab.backtest.predefined.us_equity`` subclass it and supply the market
conventions and the signal rule. The module is named ``engine_vectorbt``
rather than ``vectorbt`` so it cannot shadow the library it imports.

A *panel* is an ``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``.
A *target-percent weight* is the fraction of portfolio value a symbol should
hold after the order fills; a negative weight is a short position.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
import xarray as xr
from loguru import logger
from vectorbt.generic.enums import DrawdownStatus

from quantlab.base.backtest import BaseBacktester, SimulationResult

#: The integer ``status`` of a recovered drawdown in vectorbt's raw
#: ``Drawdowns.records``. Read from vectorbt's enum rather than written as a
#: literal, so a renumbering upstream cannot swap "recovered" and "active".
DRAWDOWN_RECOVERED = int(DrawdownStatus.Recovered)


class VectorBtBacktester(BaseBacktester):
    """Abstract backtester that simulates target weights with vectorbt.

    Subclasses supply ``config_cls``, ``MARKET`` and ``_generate_signals``;
    everything from the weights onward is implemented here. The constructor
    takes one ``config`` argument, an instance of ``config_cls``; see
    ``BaseBacktester``. Three engine conventions matter to a reader of the
    results:

    - A weight row formed at bar ``t`` fills at bar ``t + 1`` at the market's
      fill price (the weights are shifted one bar before they reach
      ``Portfolio.from_orders``).
    - A target percentage is sized against the price ``config.sizing_basis``
      names. ``"fill"`` (the default, vectorbt's own) measures it against the
      portfolio valued at the fill price of the bar it executes on and
      divides by that price. ``"valuation"`` uses the valuation price of the
      signal bar t instead (vectorbt's ``val_price``): the portfolio valued
      at t's close, divided by t's close, as an order placed after the close
      must be sized. Either way the order fills at t + 1's fill price.
    - No borrow or short-financing cost is modelled, so short-side returns
      are optimistic.

    Execution at the fill bar (ADR 0014): both price columns are
    forward-filled before they reach vectorbt, which would otherwise keep a
    NaN-priced holding at its last value and silently skip every later
    rebalance of the whole group. On each fill bar the engine then does what
    a market would:

    - A NaN weight keeps the symbol's holding; a row may mix NaN and finite
      targets.
    - An order whose raw (not forward-filled) fill price is NaN is a
      *rejected order*: the holding is kept and the order expires. It is
      recorded in ``SimulationResult.rejected_orders`` when it would have
      traded (a non-zero target, or a symbol held before the bar).
    - A symbol the price dataset marks as delisted on bar ``b``
      (``MarketDataset.delisting_bars``) is settled on bar ``b + 1``: the
      holding becomes cash at its last valuation price, with no fee or
      slippage, whatever the weights ask for it there. Each settlement is
      recorded in ``SimulationResult.settlements``.

    Records name the symbol by ``symbol`` (the display name: the ticker the
    symbol traded under that day when the price store has a ticker lookup
    file, otherwise the axis label) and ``axis_symbol`` (the label on the
    panel's ``symbol`` axis). A symbol whose prices are NaN at the start of
    the window is not yet listed rather than delisted, and trades normally
    once it lists.

    Trade statistics use vectorbt's position view: one trade is one symbol's
    round trip from entry to flat, so trimming a holding back to its target
    weight is not a closed trade. vectorbt's default exit-trade view counts
    every trim, and because only winners get trimmed under equal weighting it
    inflates the win rate. The number of fills is reported separately as
    ``Total Orders`` in the metrics.

    Examples
    --------
    A concrete engine names its config class and market conventions and
    turns predictions into target weights::

        class MyBacktester(VectorBtBacktester):
            config_cls = CrossSectionBacktestConfig
            MARKET = MarketSpec(
                fill_price_column="open",
                valuation_price_column="close",
                trading_days_per_year=252,
                session_minutes_per_day=390,
            )

            def _generate_signals(self, predictions, prices):
                ...  # a Dataset with ``weight`` on (timestamp, symbol)

        result = MyBacktester(config).run()
        result.simulation.orders  # one record per fill
    """

    #: A weight formed at bar t fills at bar t + 1; labels must use this delay.
    fill_delay_bars = 1

    #: The ``Portfolio.stats`` metric names reported for the whole window.
    #: ``benchmark_return`` is left out: vectorbt would compare against the
    #: equal-weighted traded universe, not the configured benchmark, which the
    #: ``benchmark`` and ``relative`` metric blocks report instead.
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

    def _simulate(
        self, weights: xr.Dataset, prices: xr.Dataset, dataset=None
    ) -> SimulationResult:
        """Simulate ``weights`` on ``prices`` with ``Portfolio.from_orders``.

        This is the only method that builds pandas objects: the weight and
        price panels are converted, the weights are shifted one bar so a signal
        at bar ``t`` fills at bar ``t + 1``, and vectorbt's results are mapped
        back onto the engine-neutral ``SimulationResult``. The bar interval is
        the most common difference between consecutive timestamps.

        Raises
        ------
        ValueError
            If fewer than two price bars are given.
        """
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
        raw_fill = np.asarray(
            prices[market.fill_price_column]  # type: ignore[union-attr]
            .transpose("timestamp", "symbol")
            .values,
            dtype=np.float64,
        )
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
        dataset = cfg.price_dataset if dataset is None else dataset
        delisted = np.asarray(
            dataset.delisting_bars(
                prices, market.valuation_price_column  # type: ignore[union-attr]
            )
            .transpose("timestamp", "symbol")
            .values,
            dtype=bool,
        )
        fill_values = np.asarray(fill.to_numpy(), dtype=np.float64)
        valuation_values = np.asarray(valuation.to_numpy(), dtype=np.float64)
        if cfg.sizing_basis == "valuation":
            # Sized at bar t's valuation price: what vectorbt's -inf val_price
            # (the previous bar's `close`, below) resolves to.
            sizing = np.full_like(valuation_values, np.nan)
            sizing[1:] = valuation_values[:-1]
        else:
            sizing = fill_values
        plan = self._execution_plan(
            target=np.asarray(w.shift(1).to_numpy(), dtype=np.float64),
            raw_fill=raw_fill,
            fill=fill_values,
            valuation=valuation_values,
            delisted=delisted,
            sizing=sizing,
        )

        def frame(values):
            return pd.DataFrame(values, index=fill.index, columns=fill.columns)

        pf = vbt.Portfolio.from_orders(
            close=valuation,
            price=frame(plan["price"]),
            size=frame(plan["size"]),
            size_type="targetpercent",
            direction="both",
            group_by=True,
            cash_sharing=True,
            call_seq="auto",
            # The valuation basis sizes against t's close: vectorbt's -inf
            # valuation price is the previous bar's `close`, which here is the
            # forward-filled valuation price of the signal bar.
            val_price=-np.inf if cfg.sizing_basis == "valuation" else np.inf,
            fees=frame(np.where(plan["settle"], 0.0, cfg.fees)),
            slippage=frame(np.where(plan["settle"], 0.0, cfg.slippage)),
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

        # Position-level trades: the same view `_engine_stats` reports, so the
        # trade counts agree. The six fields read below have the same names and
        # meaning in the positions records as in the exit-trade records.
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

        symbols = np.asarray(fill.columns)
        held = self._signed_order_sizes(orders, timestamps, symbols).values
        cash = np.asarray(pf.cash().to_numpy(), dtype=np.float64).reshape(
            timestamps.size, -1
        ).sum(axis=1)
        settlements, rejected = self._execution_records(
            plan, held, orders, timestamps, symbols
        )
        deviation = (
            self._max_sizing_deviation(plan, held, cash, cfg.init_cash)
            if cfg.sizing_basis == "valuation"
            else self._max_target_deviation(plan, held, cash)
        )

        return SimulationResult(
            value=value_da,
            returns=returns_da,
            orders=orders,
            settlements=settlements,
            rejected_orders=rejected,
            max_target_deviation=deviation,
            bar_interval=bar_interval,
            trades=trades,
            native=pf,
        )

    @staticmethod
    def _execution_plan(
        target: np.ndarray,
        raw_fill: np.ndarray,
        fill: np.ndarray,
        valuation: np.ndarray,
        delisted: np.ndarray,
        sizing: np.ndarray,
    ) -> dict:
        """Return the orders the market accepts at each fill bar, as ``[T, S]`` arrays.

        ``target`` is the weight each bar fills to (the weights shifted one
        bar, NaN where a symbol keeps its holding). An order whose raw fill
        price is NaN, or whose ``sizing`` price (the price the target is
        sized against) is NaN, is rejected: its size becomes NaN, so the
        holding is kept. A symbol delisted on bar ``b`` is settled on bar
        ``b + 1``: a target of 0.0 priced at its last valuation, which
        replaces any target the weights gave it there. The plan holds
        ``size``, ``price``, ``sizing``, ``target``, ``rejected`` and
        ``settle``.
        """
        settle = np.zeros_like(delisted)
        settle[1:] = delisted[:-1]
        rejected = (
            np.isfinite(target) & (np.isnan(raw_fill) | np.isnan(sizing)) & ~settle
        )
        size = np.where(rejected, np.nan, target)
        size[settle] = 0.0
        price = np.where(settle, valuation, fill)
        return {
            "size": size,
            "price": price,
            "sizing": sizing,
            "target": target,
            "rejected": rejected,
            "settle": settle,
        }

    def _execution_records(
        self,
        plan: dict,
        held: np.ndarray,
        orders: xr.Dataset,
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ) -> tuple[list[dict], list[dict]]:
        """Return the delisting settlements and the rejected orders, in time order.

        A settlement is recorded where a holding was settled; a rejection
        where the rejected order would have traded, that is, where the
        target is not zero or the symbol was held before the fill bar.
        ``held`` is the signed position after each bar. The record fields
        are described on ``SimulationResult``.
        """
        sizes = np.abs(np.asarray(orders["size"].values, dtype=np.float64))
        # A position that nets out to floating-point residue is not a holding.
        tolerance = 1e-9 * max(1.0, float(sizes.max(initial=0.0)))
        held_before = np.zeros_like(held)
        held_before[1:] = held[:-1]
        was_held = np.abs(held_before) > tolerance

        settlements, rejected = [], []
        for b in range(1, timestamps.size):
            settled = np.flatnonzero(plan["settle"][b] & was_held[b])
            refused = np.flatnonzero(
                plan["rejected"][b] & ((plan["target"][b] != 0.0) | was_held[b])
            )
            if settled.size == 0 and refused.size == 0:
                continue
            # These records are read by people (the log and the run's JSON
            # files). On a PERMNO axis (CRSP's permanent numeric security id)
            # the label is a bare number, so look up the ticker as of the day.
            day = pd.Timestamp(timestamps[b]).date()
            for j, name in zip(settled, self._symbol_labels([symbols[j] for j in settled], day)):
                record = {
                    "symbol": name,
                    "axis_symbol": str(symbols[j]),
                    "delisting_timestamp": pd.Timestamp(timestamps[b - 1]),
                    "settlement_timestamp": pd.Timestamp(timestamps[b]),
                    "price": float(plan["price"][b, j]),
                }
                logger.info(
                    f"{self.class_name}: delisting settlement of {record['symbol']}: "
                    f"delisted {record['delisting_timestamp']}, settled "
                    f"{record['settlement_timestamp']} at its last valuation "
                    f"{record['price']}"
                )
                settlements.append(record)
            for j, name in zip(refused, self._symbol_labels([symbols[j] for j in refused], day)):
                record = {
                    "symbol": name,
                    "axis_symbol": str(symbols[j]),
                    "signal_timestamp": pd.Timestamp(timestamps[b - 1]),
                    "fill_timestamp": pd.Timestamp(timestamps[b]),
                }
                logger.info(
                    f"{self.class_name}: rejected order for {record['symbol']} "
                    f"(no fill price): signal {record['signal_timestamp']}, fill "
                    f"{record['fill_timestamp']}; the holding is kept"
                )
                rejected.append(record)
        return settlements, rejected

    @staticmethod
    def _max_target_deviation(plan: dict, held: np.ndarray, cash: np.ndarray) -> float | None:
        """Return the largest |target - held weight| right after a fill bar, or None.

        The held weight is the position times the bar's order price over
        the portfolio valued at those prices after the orders. Settled
        symbols are left out; rejected ones are not.
        """
        compared = np.isfinite(plan["target"]) & ~plan["settle"]
        bars = np.flatnonzero(compared.any(axis=1))
        if bars.size == 0:
            return None
        worth = held[bars] * np.nan_to_num(plan["price"][bars])
        weights = worth / (cash[bars] + worth.sum(axis=1))[:, None]
        gap = np.abs(np.where(compared[bars], plan["target"][bars] - weights, 0.0))
        return float(gap.max())

    @staticmethod
    def _max_sizing_deviation(
        plan: dict, held: np.ndarray, cash: np.ndarray, init_cash: float
    ) -> float | None:
        """Return the largest |target - held weight| of the valuation basis, or None.

        Measured as the valuation basis sizes: the position after the fill
        bar at the signal bar's valuation price, over the book valued at
        those prices before the bar's orders (the previous bar's cash and
        positions). Without a capped buy it is 0 up to rounding. Settled
        symbols are left out; rejected ones are not.
        """
        compared = np.isfinite(plan["target"]) & ~plan["settle"]
        bars = np.flatnonzero(compared.any(axis=1))
        if bars.size == 0:
            return None
        price = np.nan_to_num(plan["sizing"][bars])
        held_before = np.zeros_like(held)
        held_before[1:] = held[:-1]
        cash_before = np.concatenate(([float(init_cash)], cash[:-1]))[bars]
        book = cash_before + (held_before[bars] * price).sum(axis=1)
        weights = held[bars] * price / book[:, None]
        gap = np.abs(np.where(compared[bars], plan["target"][bars] - weights, 0.0))
        return float(gap.max())

    @staticmethod
    def _signed_order_sizes(
        orders: xr.Dataset, timestamps: np.ndarray, symbols: np.ndarray
    ) -> xr.DataArray:
        """Return the cumulative signed position after each bar, from the orders.

        A ``Buy`` adds its size and a ``Sell`` subtracts it, accumulated along
        ``timestamps`` for each of ``symbols``. Positions are derived from the
        order records alone, not from the portfolio object, so the result does
        not depend on how vectorbt groups columns.

        Raises
        ------
        ValueError
            If an order timestamp is not on the price axis, or an
            order side is neither ``Buy`` nor ``Sell``.
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

    def _simulate_benchmark(self, benchmark_prices: xr.Dataset) -> SimulationResult:
        """Buy the single benchmark symbol with all capital and hold it.

        The benchmark goes through ``_simulate`` like the strategy: one
        target weight of 1 on the first bar and hold rows after it, so it
        fills at the second bar's fill price (the same one-bar delay as the
        strategy's first rebalance) and pays the same fees and slippage from
        the same initial cash. Its value is therefore comparable bar for bar
        with the strategy's.
        """
        timestamps = benchmark_prices.timestamp.values
        rows = np.full((timestamps.size, benchmark_prices.sizes["symbol"]), np.nan)
        rows[0, :] = 1.0
        weights = xr.Dataset(
            {"weight": (("timestamp", "symbol"), rows)},
            coords={
                "timestamp": timestamps,
                "symbol": benchmark_prices.symbol.values,
            },
        )
        return self._simulate(
            weights, benchmark_prices, dataset=self.config.benchmark_dataset
        )

    def _engine_stats(self, simulation: SimulationResult) -> dict:
        """Return vectorbt's whole-window statistics as a plain dict.

        The portfolio is first switched to the position trade view with
        ``Portfolio.replace(trades_type="positions")``: ``stats()`` does not
        accept a trade view per call and silently ignores one, and ``replace``
        keeps the change on this instance instead of touching vectorbt's
        process-wide settings. Only the trade-derived metrics differ between
        the two views; the portfolio-level ones (returns, drawdown, ratios,
        exposure, fees, dates and values) are identical. The annualisation
        frequency comes from the market spec.
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
        """Return the deepest drawdown as a span from its valley to its recovery.

        The record is chosen by depth (``valley_val / peak_val - 1``), never by
        duration: the deepest drawdown and the longest one are often different
        records. The returned dict has ``valley`` and ``end`` (bar labels),
        ``bars`` (``end_idx - valley_idx``, a bar count rather than calendar
        days), ``depth`` (a negative float) and ``recovered``. Because it is
        measured from the valley rather than from the drawdown's start, and on
        the deepest episode rather than the longest, ``bars`` is usually
        smaller than vectorbt's ``Max Drawdown Duration`` and is not comparable
        with it.

        Returns ``None`` when there are no drawdown records, no record has a
        finite depth, or an index falls outside the value axis. Nothing here is
        wrapped in a broad ``except``: a real change in vectorbt's record
        layout should surface in ``_engine_stats``, which runs earlier, rather
        than be swallowed while writing the report.
        """
        records = simulation.native.drawdowns.records  # type: ignore[union-attr]
        if len(records) == 0:
            return None

        peak = records["peak_val"].to_numpy(dtype=np.float64)
        valley = records["valley_val"].to_numpy(dtype=np.float64)
        # A non-positive peak has no meaningful percentage depth. Dividing by
        # NaN gives NaN without a division warning.
        depth = valley / np.where(peak > 0.0, peak, np.nan) - 1.0
        if not np.isfinite(depth).any():
            return None

        row = int(np.nanargmin(depth))
        # Read each column as an array: `.iloc[row]` would upcast the mixed
        # int/float row to float64, and a float cannot index timestamps.
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
        """Return the base notes plus the trade-view and drawdown-marker notes.

        The two extra notes tell the reader of ``report.html`` that the trade
        metrics are position-level and that the triangles on the equity curve
        mark the deepest drawdown from its valley to its recovery, in bars. The
        note text avoids angle brackets, ampersands and quotes so it survives
        HTML escaping unchanged.
        """
        return super()._report_notes() + [
            "The trade metrics are the position level view: one entry to flat "
            "round trip per symbol, so a partial trim of a holding is not "
            "counted as its own closed trade. Counting every trim as a closed "
            "trade is what vectorbt does by default, and it inflates the win "
            "rate. The row named Total Orders is the number of fills that "
            "actually happened over the window.",
            "The two triangles on the equity curve mark the deepest drawdown: "
            "the up triangle is its deepest bar, that is its valley, and the "
            "down triangle is the bar it recovered. The distance between them "
            "is how long it took to get from the bottom back to even, counted "
            "in trading days, that is in bars, never in calendar days. It is "
            "not the metric named Max Drawdown Duration, which measures the "
            "longest drawdown and counts from where that drawdown began, so "
            "the two numbers usually differ.",
        ]
