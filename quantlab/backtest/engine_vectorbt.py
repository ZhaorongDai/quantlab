"""vectorbt simulation engine for the backtest layer.

``VectorBtBacktester`` implements the engine hooks of ``BaseBacktester`` on
top of ``vectorbt.Portfolio.from_orders``: target-percent weights, a one-bar
delay between signal and fill, forced liquidation of delisted holdings, and
the whole-window and sliced statistics a run directory records. It is the
only module in the package that imports vectorbt; concrete backtesters such
as ``quantlab.backtest.us_equity`` subclass it and supply the market
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
    - A target percentage is measured against the portfolio valued at the
      fill price of the bar it executes on, which is vectorbt's default.
    - No borrow or short-financing cost is modelled, so short-side returns
      are optimistic.

    Delisting: both price columns are forward-filled before they reach the
    engine, because vectorbt would otherwise keep a NaN-priced holding at its
    last value and silently skip every later rebalance of the whole group.
    With the fill in place, a symbol that is held after a rebalance bar ``t``
    and has no raw fill price on bar ``t + 1`` is sold there at its last known
    price while the other symbols rebalance normally. Each such forced
    liquidation is recorded in ``SimulationResult.liquidations`` as a dict
    with the keys ``symbol`` (the display name: the ticker the symbol traded
    under on the fill day when the price store has a ticker lookup file,
    otherwise the axis label), ``axis_symbol`` (the label on the panel's ``symbol`` axis),
    ``signal_timestamp``, ``fill_timestamp`` and ``price`` (the last known
    fill price). A symbol whose prices are NaN at the start of the window and
    that was never held is not yet listed rather than delisted, and trades
    normally once it lists.

    A weight row that mixes NaN and finite values is refused before the
    simulation runs: on a rebalance row NaN means "keep the position", which
    would hold cash the other orders need and silently block them.

    Trade statistics use vectorbt's position view: one trade is one symbol's
    round trip from entry to flat, so trimming a holding back to its target
    weight is not a closed trade. vectorbt's default exit-trade view counts
    every trim, and because only winners get trimmed under equal weighting it
    inflates the win rate. The number of fills is reported separately as
    ``order_count`` in the metrics.

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

    #: The ``Portfolio.stats`` metric names reported for the whole window.
    #: ``benchmark_return`` is left out because no benchmark is simulated.
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
        """Simulate ``weights`` on ``prices`` with ``Portfolio.from_orders``.

        This is the only method that builds pandas objects: the weight and
        price panels are converted, the weights are shifted one bar so a signal
        at bar ``t`` fills at bar ``t + 1``, and vectorbt's results are mapped
        back onto the engine-neutral ``SimulationResult``. The bar interval is
        the most common difference between consecutive timestamps.

        Raises
        ------
        ValueError
            If fewer than two price bars are given, or a weight row
            mixes NaN and finite values.
        """
        # The base class checks the weights in run(), but a caller of
        # _simulate would skip that check, so check again here.
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
        """Raise ``ValueError`` if a weight row mixes NaN and finite values.

        A hold row is all NaN and a rebalance row is all finite; nothing else
        is accepted.
        """
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
                f"position' and silently blocks the rest of the rebalance)"
            )

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

    def _forced_liquidations(
        self,
        weight_values: np.ndarray,
        raw_fill: np.ndarray,
        filled_fill: np.ndarray,
        orders: xr.Dataset,
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ) -> list[dict]:
        """Return one record per holding sold because its price disappeared.

        A holding is force-liquidated when a rebalance bar ``t`` has a
        following bar inside the window, the position after bar ``t`` is
        non-zero and the raw fill price at ``t + 1`` is NaN. The record fields
        are described on the class.
        """
        n_bars = timestamps.size
        held = self._signed_order_sizes(orders, timestamps, symbols).values
        sizes = np.abs(np.asarray(orders["size"].values, dtype=np.float64))
        # A position that nets out to floating-point residue is not a holding.
        # The tolerance is 1e-9 of the largest single fill size.
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
            # These records are read by people (the log and liquidations.json).
            # On a PERMNO axis (CRSP's permanent numeric security id) the label
            # is a bare number, so look up the ticker as of the fill day.
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
                    f"(no fill price on the next bar): signal {record['signal_timestamp']}, fill "
                    f"{record['fill_timestamp']} at last price {record['price']}"
                )
                records.append(record)
        return records

    def _simulate_benchmark(
        self, start_date: str, end_date: str
    ) -> SimulationResult | None:
        """Return ``None``: no benchmark is simulated; the config setter rejects one."""
        return None

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
            "rate. The row named order_count is the number of fills that "
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

    def _period_returns_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """Return vectorbt return statistics for the bars inside ``ranges``.

        A ``Portfolio`` cannot be sliced by time, so the per-bar returns of the
        same simulation are sliced instead: each range is selected by its exact
        bar timestamps (both ends inclusive) and the pieces are concatenated in
        order. String slicing is avoided because a date used as an end label
        would take the whole day, which on intraday data would pull in bars
        past the labelled one. The annualisation frequency matches the
        whole-window statistics.

        Raises
        ------
        ValueError
            If no simulated return falls inside ``ranges``.
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
