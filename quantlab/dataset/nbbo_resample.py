"""NBBO event records -> a right-closed bar panel (phase 03.9).

`NbboResampler` turns a stream of individually-timestamped NBBO records into
one row per `(symbol, bar label)`, per trading session. Pure polars; no I/O,
no quantlab imports beyond the bar-interval table.

Semantics (D-11/D-13/D-22):

- **Right-closed bars labelled at bar end.** With session open `o` and bar
  length `d`, the labels are `o + k*d` for `k = 1..N` and bar `k` is the state
  over `(o + (k-1)*d, o + k*d]`. A record exactly on an edge belongs to the bar
  that edge labels, so a label is the time the information was available.
- **Snapshot** = the last record at or before the label.
- **Time-weighted** spread and sizes over the part of the bar that had a
  quote, from cumulative duration integrals.
- **Seed.** The last record at or before the open is moved to the open: it IS
  the NBBO in force when the session starts (the raw tier keeps the full day,
  D-04). Records after the close are dropped.
- **Carry-forward is state semantics, deliberately NOT a cleaning fill.** The
  NBBO stays in force until it is replaced, so an empty bar reports the
  prevailing state with `n_updates = 0` -- an observed state, not fabricated
  data. This lives here and never in `dataset/cleaning.py`, whose no-fill rule
  stands.

This first version (plan 03.9-01) is the tracer's slice: tie collapse, the
ambiguous-tie count, the record filters and NULL-side coverage are plan 05's
(D-10/D-19/D-25); `n_ambiguous_ties` is emitted as 0.0 until then so the D-12
variable contract holds from day one.
"""

from __future__ import annotations

import polars as pl

from quantlab.enums.data import BAR_INTERVAL_SECONDS

#: The output variables, in order. Kept in step with
#: `quantlab.dataset.cleaning.NBBO_PANEL_VARIABLES` (asserted by the Dataset's
#: `_clean`).
PANEL_VARIABLES = (
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "mid",
    "spread",
    "spread_bps",
    "imbalance",
    "n_updates",
    "tw_spread",
    "tw_bid_size",
    "tw_ask_size",
    "n_ambiguous_ties",
)

_GROUP = ["symbol", "date"]


class NbboResampler:
    """Resample NBBO records onto a regular right-closed bar grid."""

    def __init__(self, bar_interval: str) -> None:
        if bar_interval not in BAR_INTERVAL_SECONDS:
            raise ValueError(
                f"NbboResampler: bar_interval {bar_interval!r} is not one of "
                f"{list(BAR_INTERVAL_SECONDS)}."
            )
        self.bar_interval = bar_interval
        self.seconds = BAR_INTERVAL_SECONDS[bar_interval]

    @property
    def _interval_ns(self) -> int:
        return self.seconds * 1_000_000_000

    def labels(self, sessions: pl.DataFrame) -> pl.DataFrame:
        """`(date, timestamp)` of every bar label of every session, ascending.

        `sessions` carries `date`, `open`, `close` (naive UTC).
        """
        return self._edges(sessions).filter(pl.col("edge") > pl.col("open")).select(
            "date", pl.col("edge").alias("timestamp")
        )

    def _edges(self, sessions: pl.DataFrame) -> pl.DataFrame:
        """Every grid edge `e_0 = open, e_1, ..., e_N = close` per session."""
        return (
            sessions.with_columns(
                pl.col("open").cast(pl.Datetime("ns")),
                pl.col("close").cast(pl.Datetime("ns")),
            )
            .with_columns(
                pl.datetime_ranges(
                    pl.col("open"),
                    pl.col("close"),
                    interval=f"{self.seconds}s",
                    closed="both",
                    time_unit="ns",
                ).alias("edge")
            )
            .explode("edge", empty_as_null=False)
            .sort(["date", "edge"])
        )

    def resample(
        self, records: pl.DataFrame, sessions: pl.DataFrame
    ) -> pl.DataFrame:
        """Records -> one row per `(symbol, date, timestamp)` bar label.

        `records` carries `symbol`, `date` (session date), `timestamp` (naive
        UTC), `wrds_row_ord`, `best_bid`, `best_bidsizeshares`, `best_ask`,
        `best_asksizeshares`. `sessions` carries `date`, `open`, `close`.

        Only `(symbol, date)` pairs that have at least one record appear in the
        output; the Dataset reindexes onto its dense grid.
        """
        sessions = sessions.with_columns(
            pl.col("open").cast(pl.Datetime("ns")),
            pl.col("close").cast(pl.Datetime("ns")),
        )
        records = (
            records.select(
                pl.col("symbol").cast(pl.String),
                pl.col("date").cast(pl.Date),
                pl.col("timestamp").cast(pl.Datetime("ns")),
                pl.col("wrds_row_ord").cast(pl.Int64),
                pl.col("best_bid").cast(pl.Float64).alias("bid"),
                pl.col("best_bidsizeshares").cast(pl.Float64).alias("bid_size"),
                pl.col("best_ask").cast(pl.Float64).alias("ask"),
                pl.col("best_asksizeshares").cast(pl.Float64).alias("ask_size"),
            )
            # Total order, STABLE: the arrival ordinal breaks timestamp ties
            # (D-19). Never rely on any upstream scan's order.
            .sort(
                ["symbol", "date", "timestamp", "wrds_row_ord"],
                maintain_order=True,
            )
            .join(sessions, on="date", how="inner")
            .filter(pl.col("timestamp") <= pl.col("close"))
        )

        # Seed: of the records at or before the open keep only the last, and
        # move it to the open.
        pre = pl.col("timestamp") <= pl.col("open")
        records = (
            records.with_columns(
                pre.alias("_pre"),
                pre.shift(-1).over(_GROUP).fill_null(False).alias("_next_pre"),
            )
            .filter(pl.col("_pre").not_() | pl.col("_next_pre").not_())
            .with_columns(
                pl.when(pl.col("_pre"))
                .then(pl.col("open"))
                .otherwise(pl.col("timestamp"))
                .alias("timestamp")
            )
            .drop("_pre", "_next_pre")
        )

        interval = self._interval_ns

        # n_updates: every in-session record (the seed is not an update inside
        # any bar) counted in the bar whose right-closed span contains it.
        offset_ns = (pl.col("timestamp") - pl.col("open")).dt.total_nanoseconds()
        bar_index = (offset_ns + (interval - 1)) // interval
        updates = (
            records.filter(pl.col("timestamp") > pl.col("open"))
            .with_columns(
                (
                    pl.col("open")
                    + pl.duration(nanoseconds=bar_index * interval)
                ).alias("edge")
            )
            .group_by(["symbol", "date", "edge"])
            .agg(pl.len().cast(pl.Float64).alias("n_updates"))
        )

        # Derived per-record state and cumulative duration integrals.
        records = records.with_columns(
            (pl.col("ask") - pl.col("bid")).alias("spread"),
        )
        next_ts = pl.col("timestamp").shift(-1).over(_GROUP)
        records = records.with_columns(
            (
                pl.coalesce(next_ts, pl.col("close")) - pl.col("timestamp")
            )
            .dt.total_nanoseconds()
            .cast(pl.Float64)
            .alias("_dur")
        )
        integrals = {
            "spread": "_C_spread",
            "bid_size": "_C_bid_size",
            "ask_size": "_C_ask_size",
        }
        records = records.with_columns(
            *[
                (pl.col(name) * pl.col("_dur"))
                .cum_sum()
                .shift(1, fill_value=0.0)
                .over(_GROUP)
                .alias(target)
                for name, target in integrals.items()
            ],
            pl.col("_dur").cum_sum().shift(1, fill_value=0.0).over(_GROUP).alias(
                "_C_t"
            ),
        )

        # Grid of every edge for every (symbol, date) that has records.
        pairs = records.select(_GROUP).unique()
        grid = pairs.join(
            self._edges(sessions).select("date", "open", "edge"),
            on="date",
            how="inner",
        )

        # As-of backward join. With `by=` groups polars cannot check
        # sortedness and silently mis-matches unsorted input (Pitfall 8), so
        # both sides are sorted on (by..., on) immediately before the join.
        right = records.rename({"timestamp": "_rec_ts"}).drop("open", "close")
        grid = grid.sort(["symbol", "date", "edge"])
        right = right.sort(["symbol", "date", "_rec_ts"], maintain_order=True)
        joined = grid.join_asof(
            right,
            left_on="edge",
            right_on="_rec_ts",
            by=_GROUP,
            strategy="backward",
            # Both sides were sorted on (by..., on) just above; polars cannot
            # verify that itself when `by` is given.
            check_sortedness=False,
        ).sort(["symbol", "date", "edge"])

        elapsed = (
            (pl.col("edge") - pl.col("_rec_ts")).dt.total_nanoseconds().cast(pl.Float64)
        )
        joined = joined.with_columns(
            *[
                (pl.col(target) + pl.col(name) * elapsed)
                .fill_null(0.0)
                .alias(f"_F_{name}")
                for name, target in integrals.items()
            ],
            (pl.col("_C_t") + elapsed).fill_null(0.0).alias("_G"),
        )

        def delta(column: str) -> pl.Expr:
            return pl.col(column) - pl.col(column).shift(1).over(_GROUP).fill_null(
                0.0
            )

        joined = joined.with_columns(
            (delta("_F_spread") / delta("_G")).alias("tw_spread"),
            (delta("_F_bid_size") / delta("_G")).alias("tw_bid_size"),
            (delta("_F_ask_size") / delta("_G")).alias("tw_ask_size"),
        ).filter(pl.col("edge") > pl.col("open"))

        mid = (pl.col("bid") + pl.col("ask")) / 2.0
        joined = joined.with_columns(
            mid.alias("mid"),
            (1e4 * pl.col("spread") / mid).alias("spread_bps"),
            (
                (pl.col("bid_size") - pl.col("ask_size"))
                / (pl.col("bid_size") + pl.col("ask_size"))
            ).alias("imbalance"),
        )

        joined = joined.join(
            updates, on=["symbol", "date", "edge"], how="left"
        ).with_columns(
            pl.col("n_updates").fill_null(0.0),
            pl.lit(0.0, dtype=pl.Float64).alias("n_ambiguous_ties"),
        )

        return joined.select(
            "symbol",
            "date",
            pl.col("edge").alias("timestamp"),
            *[pl.col(name).cast(pl.Float64) for name in PANEL_VARIABLES],
        ).sort(["date", "timestamp", "symbol"])
