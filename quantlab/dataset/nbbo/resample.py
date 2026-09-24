"""Resampling of NBBO quote records onto a right-closed bar grid.

``NbboResampler`` turns individually timestamped NBBO records into one row
per ``(symbol, bar label)`` for each trading session, after
``NbboFilterPolicy`` has dropped the records the caller does not want. It is
pure polars with no I/O; ``NbboPanelDataset`` in this package feeds it the
raw shards and densifies its output onto the ``(timestamp, symbol)`` panel.

Bars are right-closed and labelled at their end: with session open ``o`` and
bar length ``d``, the labels are ``o + k*d`` for ``k = 1..N`` and bar ``k``
covers ``(o + (k-1)*d, o + k*d]``, so a label is the time its information
was available. Snapshot variables are the last record at or before the
label; ``tw_*`` variables are time-weighted averages over the bar. A quote
stays in force until it is replaced, so a bar with no update reports the
prevailing state with ``n_updates = 0``; that state never crosses a session
date. A null side means no quote on that side: its price and size are NaN,
never 0, and every derived variable is NaN while a side is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quantlab.dataset._support.cleaning import NBBO_PANEL_VARIABLES
from quantlab.enums.data import BAR_INTERVAL_SECONDS

#: The output variables, in order. Shared with ``clean_nbbo_panel`` so the
#: resampler and the validator cannot drift apart.
PANEL_VARIABLES = NBBO_PANEL_VARIABLES

#: Columns of the per-(date, symbol) filter-stats frame after the two keys,
#: all Int64.
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
    """Record filter applied before resampling.

    A dropped record never becomes a seed or a member of a tie; the previous
    valid quote stays in force. ``drop_nonpositive_price`` drops a record
    whose present bid or ask is ``<= 0`` (a null side is not a price).
    ``keep_qu_cond``, when set, drops a record whose ``qu_cond`` is not in
    the allow-list (a null condition is in no list). ``drop_crossed`` drops
    ``bid > ask`` and ``drop_locked`` drops ``bid == ask``, both evaluated
    only when both sides are present; locked quotes are a legitimate state
    and are kept by default. A record is counted under the first matching
    reason in the order nonpositive price, condition, crossed, locked, so the
    per-reason counts partition the dropped records.

    Example:
        >>> NbboFilterPolicy().drop_locked
        False
        >>> NbboFilterPolicy(drop_locked=True, keep_qu_cond=["R", "C"]).keep_qu_cond
        ('R', 'C')
    """

    drop_crossed: bool = True
    drop_locked: bool = False
    drop_nonpositive_price: bool = True
    keep_qu_cond: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Normalise ``keep_qu_cond`` to a tuple and reject a bare string.

        Raises:
            ValueError: If ``keep_qu_cond`` is a string rather than a
                sequence of condition codes.
        """
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
        """Build a policy from the four filter fields of an ``NbboDatasetConfig``.

        Example:
            >>> config = NbboDatasetConfig(
            ...     raw_data_dir_path="downloads/us_equity/tick/wrds_taq/wrds",
            ...     catalog_path="data/us_equity/catalog",
            ...     zarr_file_path="data/us_equity/tick/wrds_nbbo_1m.zarr",
            ... )
            >>> NbboFilterPolicy.from_config(config).drop_crossed
            True
        """
        return cls(
            drop_crossed=config.drop_crossed,
            drop_locked=config.drop_locked,
            drop_nonpositive_price=config.drop_nonpositive_price,
            keep_qu_cond=config.keep_qu_cond,
        )

    def reason(self) -> pl.Expr:
        """Return an expression giving each record's drop reason, null if kept.

        Expects ``bid`` and ``ask`` columns with a missing side already null,
        and a ``qu_cond`` column.

        Example:
            >>> policy = NbboFilterPolicy(drop_locked=True)
            >>> frame = pl.DataFrame({
            ...     "bid": [1.0, 2.0, None, 0.0],
            ...     "ask": [1.0, 1.5, 2.0, 1.0],
            ...     "qu_cond": ["R", "R", "R", "R"],
            ... })
            >>> frame.with_columns(policy.reason().alias("reason"))
            shape: (4, 4)
            ┌──────┬─────┬─────────┬───────────────────┐
            │ bid  ┆ ask ┆ qu_cond ┆ reason            │
            │ ---  ┆ --- ┆ ---     ┆ ---               │
            │ f64  ┆ f64 ┆ str     ┆ str               │
            ╞══════╪═════╪═════════╪═══════════════════╡
            │ 1.0  ┆ 1.0 ┆ R       ┆ locked            │
            │ 2.0  ┆ 1.5 ┆ R       ┆ crossed           │
            │ null ┆ 2.0 ┆ R       ┆ null              │
            │ 0.0  ┆ 1.0 ┆ R       ┆ nonpositive_price │
            └──────┴─────┴─────────┴───────────────────┘
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
    """Resample NBBO records onto a regular right-closed bar grid.

    Records are sorted by the total key ``(symbol, date, timestamp,
    wrds_row_ord)`` with a stable sort, so the input row order never affects
    the output. ``wrds_row_ord`` is the arrival ordinal recorded when the
    records were fetched; it is the only tie-breaker available on tables
    without a nanosecond field. All records of one instant count as updates,
    but only the last by that key survives the collapse: it is the quote in
    force after that instant, and its predecessors have zero duration.

    Data from 2018 onwards is unique on ``(symbol, time_m, time_m_nano)``,
    so the key fully decides the order. Earlier tables carry microseconds
    only, and a few percent of rows share a microsecond with a differently
    valued record, where the arrival ordinal is the only evidence of order.
    ``n_ambiguous_ties`` counts, per bar, the records that share their
    timestamp with a differently valued record (identical duplicates count
    0), so a consumer can see which bars' snapshots depend on that order.

    Example:
        A three-minute session, one symbol, four records (the third is
        crossed and dropped by the default policy):

        >>> sessions = pl.DataFrame({
        ...     "date": [date(2024, 1, 24)],
        ...     "open": [datetime(2024, 1, 24, 14, 30)],
        ...     "close": [datetime(2024, 1, 24, 14, 33)],
        ... })
        >>> records = pl.DataFrame({
        ...     "symbol": ["AAPL"] * 4,
        ...     "date": [date(2024, 1, 24)] * 4,
        ...     "timestamp": [
        ...         datetime(2024, 1, 24, 14, 29, 50),
        ...         datetime(2024, 1, 24, 14, 30, 30),
        ...         datetime(2024, 1, 24, 14, 31, 0),
        ...         datetime(2024, 1, 24, 14, 32, 10),
        ...     ],
        ...     "wrds_row_ord": [1, 2, 3, 4],
        ...     "best_bid": [100.0, 100.1, 100.3, 100.2],
        ...     "best_bidsizeshares": [200.0, 300.0, 100.0, 100.0],
        ...     "best_ask": [100.2, 100.3, 100.2, 100.4],
        ...     "best_asksizeshares": [100.0, 100.0, 100.0, 300.0],
        ...     "qu_cond": ["R"] * 4,
        ... })
        >>> panel = NbboResampler("1m").resample(records, sessions)
        >>> panel.select("timestamp", "bid", "ask", "mid", "n_updates")
        shape: (3, 5)
        ┌─────────────────────┬───────┬───────┬───────┬───────────┐
        │ timestamp           ┆ bid   ┆ ask   ┆ mid   ┆ n_updates │
        │ ---                 ┆ ---   ┆ ---   ┆ ---   ┆ ---       │
        │ datetime[ns]        ┆ f64   ┆ f64   ┆ f64   ┆ f64       │
        ╞═════════════════════╪═══════╪═══════╪═══════╪═══════════╡
        │ 2024-01-24 14:31:00 ┆ 100.1 ┆ 100.3 ┆ 100.2 ┆ 1.0       │
        │ 2024-01-24 14:32:00 ┆ 100.1 ┆ 100.3 ┆ 100.2 ┆ 0.0       │
        │ 2024-01-24 14:33:00 ┆ 100.2 ┆ 100.4 ┆ 100.3 ┆ 1.0       │
        └─────────────────────┴───────┴───────┴───────┴───────────┘
    """

    def __init__(
        self, bar_interval: str, policy: NbboFilterPolicy | None = None
    ) -> None:
        """Create a resampler for one bar size and an optional filter policy.

        Args:
            bar_interval: A key of ``BAR_INTERVAL_SECONDS`` such as ``"1m"``.
            policy: The record filter; the default ``NbboFilterPolicy()``
                when ``None``.

        Raises:
            ValueError: If ``bar_interval`` is not a known bar size.
        """
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
        """Return the bar length in nanoseconds."""
        return self.seconds * 1_000_000_000

    def labels(self, sessions: pl.DataFrame) -> pl.DataFrame:
        """Return ``(date, timestamp)`` for every bar label of every session.

        Args:
            sessions: A frame with ``date``, ``open`` and ``close`` columns,
                the last two in naive UTC.

        Returns:
            One row per bar label, ascending; the open itself is not a label.

        Example:
            >>> NbboResampler("1m").labels(sessions)
            shape: (3, 2)
            ┌────────────┬─────────────────────┐
            │ date       ┆ timestamp           │
            │ ---        ┆ ---                 │
            │ date       ┆ datetime[ns]        │
            ╞════════════╪═════════════════════╡
            │ 2024-01-24 ┆ 2024-01-24 14:31:00 │
            │ 2024-01-24 ┆ 2024-01-24 14:32:00 │
            │ 2024-01-24 ┆ 2024-01-24 14:33:00 │
            └────────────┴─────────────────────┘
        """
        return self._edges(sessions).filter(pl.col("edge") > pl.col("open")).select(
            "date", pl.col("edge").alias("timestamp")
        )

    def _edges(self, sessions: pl.DataFrame) -> pl.DataFrame:
        """Return every grid edge, open through close inclusive, per session.

        Raises:
            ValueError: If a session's length is not a whole, positive number
                of bars. A ragged last bar would differ in length from every
                other bar, and truncating it would drop in-session quotes.
        """
        sessions = sessions.with_columns(
            pl.col("open").cast(pl.Datetime("ns")),
            pl.col("close").cast(pl.Datetime("ns")),
        )
        length_ns = (pl.col("close") - pl.col("open")).dt.total_nanoseconds()
        ragged = sessions.filter(
            (length_ns <= 0) | (length_ns % self._interval_ns != 0)
        )
        if ragged.height:
            first = ragged.row(0, named=True)
            raise ValueError(
                f"NbboResampler: the session of {first['date']} "
                f"({first['open']} .. {first['close']} UTC) is not a whole, "
                f"positive number of bar_interval {self.bar_interval!r} "
                f"({self.seconds}s) bars; {ragged.height} session(s) affected."
            )
        return (
            sessions.with_columns(
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
        """Return the bar panel only; see ``resample_with_stats``.

        Example:
            >>> NbboResampler("1m").resample(records, sessions).shape
            (3, 16)
        """
        return self.resample_with_stats(records, sessions)[0]

    # ------------------------------------------------------------ pipeline --

    @staticmethod
    def _normalise(records: pl.DataFrame) -> pl.DataFrame:
        """Return typed working columns; a null or NaN side nulls its size too.

        A missing ``qu_cond`` column is added as all-null so the filter can
        always refer to it.
        """
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
        # No quote on a side means no size on it either, whatever size value
        # arrived alongside the null price.
        return frame.with_columns(
            pl.when(pl.col("bid").is_not_null()).then(pl.col("bid_size")).alias("bid_size"),
            pl.when(pl.col("ask").is_not_null()).then(pl.col("ask_size")).alias("ask_size"),
        )

    def _filter(self, records: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return ``(kept records, per-(date, symbol) stats)``."""
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
        """Resample ``records`` and return the bar panel with the filter stats.

        Args:
            records: A frame with ``symbol``, ``date`` (session date),
                ``timestamp`` (naive UTC), ``wrds_row_ord``, ``best_bid``,
                ``best_bidsizeshares``, ``best_ask``, ``best_asksizeshares``
                and optionally ``qu_cond``.
            sessions: A frame with ``date``, ``open`` and ``close`` columns.

        Returns:
            A ``(panel, stats)`` pair. The panel has ``symbol``, ``date``,
            ``timestamp`` and the ``PANEL_VARIABLES`` as float64, one row per
            bar label for every ``(symbol, date)`` present in the input (a
            pair whose records were all filtered out still gets its rows, all
            NaN). The stats frame has one row per input ``(date, symbol)``
            with the ``FILTER_STATS_COUNTS`` columns.

        Example:
            >>> panel, stats = NbboResampler("1m").resample_with_stats(
            ...     records, sessions
            ... )
            >>> stats.select("symbol", "records_in", "dropped_crossed")
            shape: (1, 3)
            ┌────────┬────────────┬─────────────────┐
            │ symbol ┆ records_in ┆ dropped_crossed │
            │ ---    ┆ ---        ┆ ---             │
            │ str    ┆ i64        ┆ i64             │
            ╞════════╪════════════╪═════════════════╡
            │ AAPL   ┆ 4          ┆ 1               │
            └────────┴────────────┴─────────────────┘
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
            # Stable total order; the arrival ordinal breaks timestamp ties.
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

        # A record is ambiguous when its (symbol, date, timestamp) group holds
        # more than one distinct quote state. Null sides compare as values, so
        # two identical one-sided records are not ambiguous. Counted after
        # filtering and before the tie collapse.
        state_key = pl.concat_str(
            [
                pl.col(name).cast(pl.String).fill_null("<null>")
                for name in ("bid", "bid_size", "ask", "ask_size")
            ],
            separator="|",
        )
        tie = ["symbol", "date", "timestamp"]
        records = records.with_columns(
            (state_key.n_unique().over(tie) > 1).cast(pl.Float64).alias("_ambiguous")
        )

        # n_updates counts every kept in-session record (each tied message is
        # one update) in the bar whose right-closed span holds it. The seed
        # is not an update inside any bar.
        updates = (
            records.filter(pl.col("timestamp") > pl.col("open"))
            .with_columns(edge_of_record.alias("edge"))
            .group_by(["symbol", "date", "edge"])
            .agg(
                pl.len().cast(pl.Float64).alias("n_updates"),
                pl.col("_ambiguous").sum().alias("n_ambiguous_ties"),
            )
        )

        # Tie collapse: the last record of each instant by the total order is
        # the quote after that instant; its predecessors have zero duration
        # and would otherwise be candidate as-of matches.
        records = records.unique(
            subset=tie, keep="last", maintain_order=True
        ).drop("_ambiguous")

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

        # Cumulative integrals: one value integral and one coverage integral
        # (the time the variable was defined) per time-weighted source.
        def exclusive_cum(expr: pl.Expr) -> pl.Expr:
            """Return the running sum of ``expr`` up to, not including, each row.

            Example:
                Over per-record durations 1, 2, 3 the result is 0, 1, 3.
            """
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
        # sortedness and silently mis-matches unsorted input, so both sides
        # are sorted on (by..., on) immediately before the join.
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
            """Return the change of ``column`` since the previous bar of the group.

            Example:
                Over cumulative values 1, 3, 6 the result is 1, 2, 3.
            """
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
            pl.col("n_ambiguous_ties").fill_null(0.0),
        )

        panel = joined.select(
            "symbol",
            "date",
            pl.col("edge").alias("timestamp"),
            *[pl.col(name).cast(pl.Float64) for name in PANEL_VARIABLES],
        ).sort(["date", "timestamp", "symbol"])
        return panel, stats
