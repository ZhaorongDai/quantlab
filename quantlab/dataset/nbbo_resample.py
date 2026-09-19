"""NBBO event records -> a right-closed bar panel (phase 03.9).

`NbboResampler` turns a stream of individually-timestamped NBBO records into
one row per `(symbol, bar label)`, per trading session. Pure polars; no I/O,
no quantlab imports beyond the bar-interval table.

Semantics (D-10/D-11/D-13/D-22/D-25):

- **Filter first** (D-10). `NbboFilterPolicy` drops records before anything
  else, so a dropped record never becomes the seed or a member of a tie, and
  the previous valid NBBO stays standing. Every dropped record is counted per
  `(date, symbol)` under exactly one reason (`resample_with_stats`).
- **A NULL side is "no quote on that side"** (D-25): that side's price and
  size are NaN, never 0. `spread`, `mid`, `spread_bps` and `imbalance` are NaN
  while a side is missing, and each time-weighted average divides by the time
  its own variable was defined -- a zero would drag every average toward zero
  and read as a real, absurdly tight quote.
- **Right-closed bars labelled at bar end.** With session open `o` and bar
  length `d`, the labels are `o + k*d` for `k = 1..N` and bar `k` is the state
  over `(o + (k-1)*d, o + k*d]`. A record exactly on an edge belongs to the bar
  that edge labels, so a label is the time the information was available.
- **Snapshot** = the last record at or before the label.
- **Time-weighted** spread and sizes from cumulative duration integrals.
- **Seed.** The last record at or before the open is moved to the open: it IS
  the NBBO in force when the session starts (the raw tier keeps the full day,
  D-04). Records after the close are dropped.
- **Carry-forward is state semantics, deliberately NOT a cleaning fill.** The
  NBBO stays in force until it is replaced, so an empty bar reports the
  prevailing state with `n_updates = 0` -- an observed state, not fabricated
  data. This lives here and never in `dataset/cleaning.py`, whose no-fill rule
  stands. State never crosses a session date (D-13).
"""

from __future__ import annotations

from dataclasses import dataclass

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

#: Columns of the per-(date, symbol) filter-stats frame, all Int64 after the
#: two keys.
FILTER_STATS_COUNTS = (
    "records_in",
    "dropped_nonpositive_price",
    "dropped_condition",
    "dropped_crossed",
    "dropped_locked",
    "one_sided_kept",
    "both_null_kept",
)

_GROUP = ["symbol", "date"]

#: Drop reasons in precedence order: a record matching several is counted
#: under the first only.
_REASONS = ("nonpositive_price", "condition", "crossed", "locked")

#: Time-weighted variables and the per-record column each integrates.
_TW_SOURCES = {
    "tw_spread": "spread",
    "tw_bid_size": "bid_size",
    "tw_ask_size": "ask_size",
}


@dataclass(frozen=True)
class NbboFilterPolicy:
    """The resampler's record filter (D-10). Defaults are the D-10 defaults.

    - `drop_nonpositive_price`: drop a record whose PRESENT bid or ask is
      `<= 0`. A NULL side is not a price and is never dropped by this rule.
    - `keep_qu_cond`: when set, drop a record whose `qu_cond` is not in the
      allow-list (a NULL `qu_cond` is not in any list). None keeps all.
    - `drop_crossed`: drop `bid > ask`, evaluated only when both sides are
      present.
    - `drop_locked`: drop `bid == ask` (both sides present). Off by default:
      locked quotes are 4.4% of live rows and are legitimate states.

    **Precedence.** A record is counted once, under the first matching reason
    in the order nonpositive_price > condition > crossed > locked, so the
    per-reason counts partition the dropped records.
    """

    drop_crossed: bool = True
    drop_locked: bool = False
    drop_nonpositive_price: bool = True
    keep_qu_cond: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.keep_qu_cond is None:
            return
        if isinstance(self.keep_qu_cond, str):
            raise ValueError(
                f"NbboFilterPolicy: keep_qu_cond must be a sequence of "
                f"condition codes, got the bare string {self.keep_qu_cond!r} "
                f"(write ({self.keep_qu_cond!r},))."
            )
        object.__setattr__(self, "keep_qu_cond", tuple(self.keep_qu_cond))

    @classmethod
    def from_config(cls, config) -> "NbboFilterPolicy":
        """Read the four filter fields off an `NbboDatasetConfig`."""
        return cls(
            drop_crossed=config.drop_crossed,
            drop_locked=config.drop_locked,
            drop_nonpositive_price=config.drop_nonpositive_price,
            keep_qu_cond=config.keep_qu_cond,
        )

    def reason(self) -> pl.Expr:
        """Per-record drop reason (String), null for a kept record.

        Expects `bid`/`ask` with a NULL side already null and a `qu_cond`
        column.
        """
        bid_present = pl.col("bid").is_not_null()
        ask_present = pl.col("ask").is_not_null()
        both = bid_present & ask_present
        rules: list[tuple[str, pl.Expr]] = []
        if self.drop_nonpositive_price:
            rules.append(
                (
                    "nonpositive_price",
                    (bid_present & (pl.col("bid") <= 0))
                    | (ask_present & (pl.col("ask") <= 0)),
                )
            )
        if self.keep_qu_cond is not None:
            rules.append(
                (
                    "condition",
                    pl.col("qu_cond").is_in(list(self.keep_qu_cond)).fill_null(False).not_(),
                )
            )
        if self.drop_crossed:
            rules.append(("crossed", both & (pl.col("bid") > pl.col("ask"))))
        if self.drop_locked:
            rules.append(("locked", both & (pl.col("bid") == pl.col("ask"))))

        expr: pl.Expr = pl.lit(None, dtype=pl.String)
        for name, condition in reversed(rules):
            expr = pl.when(condition.fill_null(False)).then(pl.lit(name)).otherwise(expr)
        return expr


class NbboResampler:
    """Resample NBBO records onto a regular right-closed bar grid."""

    def __init__(
        self, bar_interval: str, policy: NbboFilterPolicy | None = None
    ) -> None:
        if bar_interval not in BAR_INTERVAL_SECONDS:
            raise ValueError(
                f"NbboResampler: bar_interval {bar_interval!r} is not one of "
                f"{list(BAR_INTERVAL_SECONDS)}."
            )
        self.bar_interval = bar_interval
        self.seconds = BAR_INTERVAL_SECONDS[bar_interval]
        self.policy = policy if policy is not None else NbboFilterPolicy()

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
        """The panel element of `resample_with_stats`."""
        return self.resample_with_stats(records, sessions)[0]

    # ------------------------------------------------------------ pipeline --

    @staticmethod
    def _normalise(records: pl.DataFrame) -> pl.DataFrame:
        """Typed working columns; a NULL (or NaN) side nulls its size too."""
        qu_cond = (
            pl.col("qu_cond").cast(pl.String)
            if "qu_cond" in records.columns
            else pl.lit(None, dtype=pl.String).alias("qu_cond")
        )
        frame = records.select(
            pl.col("symbol").cast(pl.String),
            pl.col("date").cast(pl.Date),
            pl.col("timestamp").cast(pl.Datetime("ns")),
            pl.col("wrds_row_ord").cast(pl.Int64),
            pl.col("best_bid").cast(pl.Float64).fill_nan(None).alias("bid"),
            pl.col("best_bidsizeshares").cast(pl.Float64).alias("bid_size"),
            pl.col("best_ask").cast(pl.Float64).fill_nan(None).alias("ask"),
            pl.col("best_asksizeshares").cast(pl.Float64).alias("ask_size"),
            qu_cond,
        )
        # D-25: no quote on a side means no size on it either, whatever size
        # value WRDS sent alongside the NULL price.
        return frame.with_columns(
            pl.when(pl.col("bid").is_not_null()).then(pl.col("bid_size")).alias("bid_size"),
            pl.when(pl.col("ask").is_not_null()).then(pl.col("ask_size")).alias("ask_size"),
        )

    def _filter(self, records: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """(kept records, per-(date, symbol) stats)."""
        records = records.with_columns(self.policy.reason().alias("_reason"))
        kept = pl.col("_reason").is_null()
        n_present = pl.col("bid").is_not_null().cast(pl.Int64) + pl.col(
            "ask"
        ).is_not_null().cast(pl.Int64)
        stats = (
            records.group_by(["date", "symbol"])
            .agg(
                pl.len().alias("records_in"),
                *[
                    (pl.col("_reason") == reason).fill_null(False).sum().alias(f"dropped_{reason}")
                    for reason in _REASONS
                ],
                (kept & (n_present == 1)).sum().alias("one_sided_kept"),
                (kept & (n_present == 0)).sum().alias("both_null_kept"),
            )
            .select(
                "date",
                "symbol",
                *[pl.col(name).cast(pl.Int64) for name in FILTER_STATS_COUNTS],
            )
            .sort(["date", "symbol"])
        )
        return records.filter(kept).drop("_reason"), stats

    def resample_with_stats(
        self, records: pl.DataFrame, sessions: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Records -> (bar panel, filter stats).

        `records` carries `symbol`, `date` (session date), `timestamp` (naive
        UTC), `wrds_row_ord`, `best_bid`, `best_bidsizeshares`, `best_ask`,
        `best_asksizeshares` and optionally `qu_cond`. `sessions` carries
        `date`, `open`, `close`.

        The panel has one row per bar label for every `(symbol, date)` present
        in the INPUT records (a pair whose records were all filtered out still
        gets its rows, all NaN); the Dataset reindexes onto its dense grid.
        The stats frame has one row per input `(date, symbol)` with the
        `FILTER_STATS_COUNTS` columns, counted over every input record.
        """
        sessions = sessions.with_columns(
            pl.col("open").cast(pl.Datetime("ns")),
            pl.col("close").cast(pl.Datetime("ns")),
        )
        records = self._normalise(records)
        pairs = records.select(_GROUP).unique()
        records, stats = self._filter(records)

        records = (
            records
            # Total order, STABLE: the arrival ordinal breaks timestamp ties
            # (D-19). Never rely on any upstream scan's order.
            .sort(
                ["symbol", "date", "timestamp", "wrds_row_ord"],
                maintain_order=True,
            )
            .join(sessions, on="date", how="inner")
            .filter(pl.col("timestamp") <= pl.col("close"))
        )

        interval = self._interval_ns
        offset_ns = (pl.col("timestamp") - pl.col("open")).dt.total_nanoseconds()
        bar_index = (offset_ns + (interval - 1)) // interval
        edge_of_record = pl.col("open") + pl.duration(nanoseconds=bar_index * interval)

        # n_updates: every kept in-session record (the seed is not an update
        # inside any bar) counted in the bar whose right-closed span holds it.
        updates = (
            records.filter(pl.col("timestamp") > pl.col("open"))
            .with_columns(edge_of_record.alias("edge"))
            .group_by(["symbol", "date", "edge"])
            .agg(pl.len().cast(pl.Float64).alias("n_updates"))
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

        records = records.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
        next_ts = pl.col("timestamp").shift(-1).over(_GROUP)
        records = records.with_columns(
            (pl.coalesce(next_ts, pl.col("close")) - pl.col("timestamp"))
            .dt.total_nanoseconds()
            .cast(pl.Float64)
            .alias("_dur")
        )

        # Cumulative integrals, one value integral and one COVERAGE integral
        # per variable (D-25): the time the variable was defined.
        def exclusive_cum(expr: pl.Expr) -> pl.Expr:
            return expr.cum_sum().shift(1, fill_value=0.0).over(_GROUP)

        sources = sorted(set(_TW_SOURCES.values()))
        records = records.with_columns(
            *[
                exclusive_cum(pl.col(name).fill_null(0.0) * pl.col("_dur")).alias(f"_C_{name}")
                for name in sources
            ],
            *[
                exclusive_cum(
                    pl.col(name).is_not_null().cast(pl.Float64) * pl.col("_dur")
                ).alias(f"_Ct_{name}")
                for name in sources
            ],
        )

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
            check_sortedness=False,
        ).sort(["symbol", "date", "edge"])

        elapsed = (
            (pl.col("edge") - pl.col("_rec_ts")).dt.total_nanoseconds().cast(pl.Float64)
        )
        joined = joined.with_columns(
            *[
                (pl.col(f"_C_{name}") + pl.col(name).fill_null(0.0) * elapsed)
                .fill_null(0.0)
                .alias(f"_F_{name}")
                for name in sources
            ],
            *[
                (
                    pl.col(f"_Ct_{name}")
                    + pl.col(name).is_not_null().cast(pl.Float64) * elapsed
                )
                .fill_null(0.0)
                .alias(f"_G_{name}")
                for name in sources
            ],
        )

        def delta(column: str) -> pl.Expr:
            return pl.col(column) - pl.col(column).shift(1).over(_GROUP).fill_null(0.0)

        joined = joined.with_columns(
            *[
                pl.when(delta(f"_G_{source}") > 0)
                .then(delta(f"_F_{source}") / delta(f"_G_{source}"))
                .otherwise(None)
                .alias(target)
                for target, source in _TW_SOURCES.items()
            ]
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

        panel = joined.select(
            "symbol",
            "date",
            pl.col("edge").alias("timestamp"),
            *[pl.col(name).cast(pl.Float64) for name in PANEL_VARIABLES],
        ).sort(["date", "timestamp", "symbol"])
        return panel, stats
