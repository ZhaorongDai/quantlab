"""Resample SIP trades onto Trade bars, by the SIP's rules per trade condition.

A *Trade bar* is quantlab's own bar built from every trade of the
consolidated tape (ADR 0030), not the vendor's minute aggregate.
``TradeBarResampler`` turns one or more sessions of trade records into one
row per ``(symbol, bar label)``. It is pure polars with no file access, as
``quantlab.dataset.nbbo.resample.NbboResampler`` is;
``quantlab.dataset.massive.trade_bars`` feeds it the raw files and puts its
output on the ``(timestamp, symbol)`` panel.

The rules:

- **Time is** ``sip_timestamp``, when the SIP published the trade, so a bar
  holds only trades the market could see by its end.
- **Bars are right-closed and labelled at their end.** With session open
  ``o`` and bar length ``d``, bar ``k`` covers ``(o + (k-1)d, o + kd]`` for
  ``k = 1..N``: a trade exactly on a boundary belongs to the bar ending
  there; one exactly at the open is before the session.
- **Each trade counts towards high/low, open/close and volume separately**,
  by the consolidated update rules of its conditions (the condition table,
  ``quantlab.dataset.massive.raw.read_conditions``): it counts towards one
  only if every one of its conditions allows it. A trade without conditions
  counts towards all three. ``open`` and ``close`` are the first and last
  open/close-eligible trade by ``sip_timestamp``, ties broken by
  ``sequence_number``; ``high`` and ``low`` the extremes of the
  high/low-eligible trades; ``volume`` and ``n_trades`` the size and count
  of the volume-eligible ones.
- **Corrected and cancelled trades** are dropped: every ``correction``
  but 0 (a regular trade) and 12 (the corrected print that replaces a
  trade later corrected, code 1), so a correction keeps the trade at its
  corrected values. A null ``correction`` is a regular trade. A trade with a
  condition the table does not know is dropped too; both are counted.
- **Volumes are over the volume-eligible trades**, as floats (recent years
  carry fractional shares): ``dollar_volume`` is price times size (VWAP is
  ``dollar_volume / volume``), ``offexchange_volume`` the size of trades
  reported through a TRF (``trf_id`` non-zero), ``oddlot_volume`` of trades
  under 100 shares.
- **The tick rule** signs each volume-eligible trade against the price of
  the volume-eligible trade before it in the same session: an up-tick is a
  buy, a down-tick a sell, and a zero tick takes the side of the last
  non-zero tick. The previous price carries across bars but not across
  sessions; a session's first trade, and zero ticks before its first
  non-zero tick, count in neither ``buy_volume`` nor ``sell_volume``, so
  their sum may be less than ``volume``.
- **A bar without an eligible trade** has null prices and zero volumes and
  ``n_trades``, never a carried price.

Examples
--------
>>> bars, stats = TradeBarResampler("1m").resample_with_stats(trades, conditions, sessions)
>>> bars.columns
['symbol', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'dollar_volume', 'n_trades', 'buy_volume', 'sell_volume', 'offexchange_volume', 'oddlot_volume']
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset.massive.raw import UPDATE_FLAGS
from quantlab.enums.data import BAR_INTERVAL_SECONDS

#: The bar variables, in order, after the ``symbol``, ``date`` and
#: ``timestamp`` keys.
TRADE_BAR_VARIABLES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "n_trades",
    "buy_volume",
    "sell_volume",
    "offexchange_volume",
    "oddlot_volume",
)

#: The bar variables that are sums over trades: zero, never null, on a bar
#: without an eligible trade.
TRADE_BAR_SUMS = tuple(name for name in TRADE_BAR_VARIABLES if name not in ("open", "high", "low", "close"))

#: The count columns of the per-``(date, symbol)`` stats, after the keys.
#: ``volume_ineligible`` counts the trades kept whose conditions exclude them
#: from volume (and so from every volume variable).
TRADE_STATS_COUNTS = (
    "trades_in",
    "dropped_correction",
    "dropped_unknown_condition",
    "outside_session",
    "volume_ineligible",
)

#: Trades under this many shares are odd lots.
ROUND_LOT = 100

#: The ``correction`` codes a kept trade carries: a regular trade, and the
#: corrected print of a trade that was later corrected.
KEPT_CORRECTIONS = (0, 12)

_NS_PER_SECOND = 1_000_000_000


class TradeBarResampler:
    """Resample trade records onto right-closed Trade bars; see the module docstring.

    Parameters
    ----------
    bar_interval : str
        The bar size, a key of ``BAR_INTERVAL_SECONDS``.

    Raises
    ------
    ValueError
        If ``bar_interval`` is unknown.

    Examples
    --------
    >>> TradeBarResampler("5m").interval_ns
    300000000000
    """

    def __init__(self, bar_interval: str) -> None:
        """Initialize; see the class docstring."""
        if bar_interval not in BAR_INTERVAL_SECONDS:
            raise ValueError(
                f"Unknown bar interval {bar_interval!r}; expected one of {list(BAR_INTERVAL_SECONDS)}."
            )
        self.bar_interval = bar_interval
        self.interval_ns = BAR_INTERVAL_SECONDS[bar_interval] * _NS_PER_SECOND

    def labels(self, sessions: pl.DataFrame) -> pl.DataFrame:
        """Return every bar label of ``sessions``.

        Parameters
        ----------
        sessions : pl.DataFrame
            Columns ``date``, ``open`` and ``close`` (naive UTC).

        Returns
        -------
        pl.DataFrame
            Columns ``date`` and ``timestamp`` (``Datetime("ns")``): for each
            session ``open + k * interval`` for ``k = 1..N``, the last at or
            before the close, sorted.

        Examples
        --------
        >>> TradeBarResampler("1m").labels(sessions)["timestamp"].head(1).to_list()
        [datetime.datetime(2024, 1, 24, 14, 31)]
        """
        interval = pl.duration(nanoseconds=self.interval_ns)
        return (
            sessions.select(
                "date",
                pl.datetime_ranges(
                    pl.col("open").cast(pl.Datetime("ns")) + interval,
                    pl.col("close").cast(pl.Datetime("ns")),
                    interval=f"{self.interval_ns}ns",
                    time_unit="ns",
                    closed="both",
                ).alias("timestamp"),
            )
            .explode("timestamp", empty_as_null=True)
            .drop_nulls("timestamp")
            .sort("date", "timestamp")
        )

    @staticmethod
    def _eligibility(trades: pl.DataFrame, conditions: pl.DataFrame) -> pl.DataFrame:
        """Return each distinct ``conditions`` text with its three flags and whether it is known.

        Distinct texts are few (a few hundred a day) against tens of
        millions of trades, so the rules are evaluated once per text.
        """
        texts = trades.select(pl.col("conditions").fill_null("")).unique()
        exploded = texts.with_columns(
            pl.col("conditions")
            .str.split(",")
            .list.eval(pl.element().str.strip_chars().filter(pl.element() != "").cast(pl.Int64))
            .alias("id")
        ).explode("id", empty_as_null=True)
        joined = exploded.join(
            conditions.select("id", *UPDATE_FLAGS, pl.lit(True).alias("_known")), on="id", how="left"
        )
        # A text without conditions explodes to one null id: known, all flags.
        no_condition = pl.col("id").is_null()
        return joined.group_by("conditions").agg(
            (no_condition | pl.col("_known").fill_null(False)).all().alias("_known"),
            *((no_condition | pl.col(flag).fill_null(False)).all().alias(flag) for flag in UPDATE_FLAGS),
        )

    def resample_with_stats(
        self, trades: pl.DataFrame, conditions: pl.DataFrame, sessions: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Resample trades onto bars and count what each rule dropped.

        Parameters
        ----------
        trades : pl.DataFrame
            Columns ``symbol``, ``date`` (the session date), ``conditions``
            (comma-separated ids, or null), ``correction``, ``price``,
            ``size``, ``sip_timestamp`` (Int64 nanoseconds since the epoch,
            UTC), ``sequence_number`` and ``trf_id`` (0 or null when not
            reported through a TRF).
        conditions : pl.DataFrame
            The condition table, as ``read_conditions`` returns it.
        sessions : pl.DataFrame
            Columns ``date``, ``open`` and ``close`` (naive UTC) of the
            sessions the trades belong to.

        Returns
        -------
        bars : pl.DataFrame
            Columns ``symbol``, ``date``, ``timestamp`` and
            ``TRADE_BAR_VARIABLES``: every bar of the session for each
            symbol with at least one record on that date, sorted by symbol
            and label. Prices are null where no trade was eligible; the
            volumes are Float64, ``n_trades`` Int64.
        stats : pl.DataFrame
            Columns ``date``, ``symbol`` and ``TRADE_STATS_COUNTS`` (Int64),
            one row per ``(date, symbol)`` with a record.

        Examples
        --------
        With ``trades`` read by ``quantlab.dataset.massive.raw.read_trades``
        (``ticker`` renamed ``symbol``, a ``date`` column added), the
        condition table and that day's XNYS session bounds::

            sessions = XnysSessionCalendar().session_bounds([date(2016, 11, 25)])
            bars, stats = TradeBarResampler("1m").resample_with_stats(trades, conditions, sessions)
        """
        bounds = sessions.select(
            "date",
            pl.col("open").cast(pl.Datetime("ns")).dt.epoch("ns").alias("_open"),
            pl.col("close").cast(pl.Datetime("ns")).dt.epoch("ns").alias("_close"),
        )
        # Lazy, so polars reads each column once and holds no intermediate
        # copy of a day's tens of millions of trades.
        records = (
            trades.lazy()
            .with_columns(pl.col("symbol").cast(pl.String), pl.col("conditions").fill_null(""))
            .join(self._eligibility(trades, conditions).lazy(), on="conditions", how="left")
            .join(bounds.lazy(), on="date", how="left")
        )
        corrected = ~pl.col("correction").fill_null(0).is_in(KEPT_CORRECTIONS)
        unknown = ~corrected & ~pl.col("_known")
        outside = (
            ~corrected
            & pl.col("_known")
            & ~((pl.col("sip_timestamp") > pl.col("_open")) & (pl.col("sip_timestamp") <= pl.col("_close"))).fill_null(
                False
            )
        )
        stats = (
            records.group_by("date", "symbol")
            .agg(
                pl.len().cast(pl.Int64).alias("trades_in"),
                corrected.sum().cast(pl.Int64).alias("dropped_correction"),
                unknown.sum().cast(pl.Int64).alias("dropped_unknown_condition"),
                outside.sum().cast(pl.Int64).alias("outside_session"),
                (~corrected & ~unknown & ~outside & ~pl.col("updates_volume")).sum().cast(pl.Int64).alias(
                    "volume_ineligible"
                ),
            )
            .sort("date", "symbol")
        )

        kept = records.filter(~corrected & ~unknown & ~outside)
        step = self.interval_ns
        kept = kept.with_columns(
            (
                pl.col("_open")
                + (pl.col("sip_timestamp") - pl.col("_open") + step - 1) // step * step
            )
            .cast(pl.Datetime("ns"))
            .alias("timestamp")
        ).sort("symbol", "sip_timestamp", "sequence_number")
        volume_ok = pl.col("updates_volume")
        session = ("symbol", "date")
        # The tick rule over the volume-eligible trades of each session: the
        # previous such trade's price, the sign of the change, and a zero
        # tick taking the last non-zero one's side.
        previous = pl.when(volume_ok).then(pl.col("price")).shift(1).forward_fill().over(session)
        tick = pl.when(volume_ok).then((pl.col("price") - previous).sign()).replace(0, None)
        kept = kept.with_columns(tick.forward_fill().over(session).alias("_side"))
        price_oc = pl.col("price").filter(pl.col("updates_open_close"))
        price_hl = pl.col("price").filter(pl.col("updates_high_low"))
        size = pl.col("size").cast(pl.Float64)

        def volume_where(condition: pl.Expr) -> pl.Expr:
            return size.filter(volume_ok & condition.fill_null(False)).sum()

        observed = kept.group_by("symbol", "date", "timestamp", maintain_order=True).agg(
            price_oc.first().alias("open"),
            price_hl.max().alias("high"),
            price_hl.min().alias("low"),
            price_oc.last().alias("close"),
            volume_where(pl.lit(True)).alias("volume"),
            (pl.col("price") * size).filter(volume_ok).sum().alias("dollar_volume"),
            volume_ok.sum().cast(pl.Int64).alias("n_trades"),
            volume_where(pl.col("_side") == 1).alias("buy_volume"),
            volume_where(pl.col("_side") == -1).alias("sell_volume"),
            volume_where(pl.col("trf_id").fill_null(0) != 0).alias("offexchange_volume"),
            volume_where(size < ROUND_LOT).alias("oddlot_volume"),
        )

        grid = records.select("symbol", "date").unique().join(self.labels(sessions).lazy(), on="date", how="inner")
        bars = (
            grid.join(observed, on=["symbol", "date", "timestamp"], how="left")
            .with_columns(pl.col(name).fill_null(0) for name in TRADE_BAR_SUMS)
            .select("symbol", "date", "timestamp", *TRADE_BAR_VARIABLES)
            .sort("symbol", "timestamp")
        )
        bars, stats = pl.collect_all([bars, stats])
        return bars, stats
