"""Resample NBBO quote records onto a regular bar grid.

The NBBO (National Best Bid and Offer) is the best bid and best ask across
all US exchanges. Each NBBO *record* is one update of that quote, with a
timestamp, a bid and ask price, and the size (shares) offered at each. A
*session* is one trading day, with its open and close time.

``NbboResampler`` turns these irregular records into one row per
``(symbol, bar label)`` for each session, after ``NbboFilterPolicy`` has
dropped the records the caller does not want. It is pure polars with no
file access; ``NbboPanelDataset`` in this package feeds it the raw files and
puts its output onto the ``(timestamp, symbol)`` panel.

Bars are *right-closed* and labelled at their end. With session open ``o``
and bar length ``d``, the labels are ``o + k*d`` for ``k = 1..N``, and bar
``k`` covers ``(o + (k-1)*d, o + k*d]``, so a label is the time at which the
bar's information was known. *Snapshot* variables (``bid``, ``ask``, ...)
come from the last record at or before the label; ``tw_*`` variables are
time-weighted averages over the bar. A quote stays in force until it is
replaced, so a bar with no update reports the quote still in force, with
``n_updates = 0``; a quote is never carried over from one session date to
the next. A null side means there is no quote on that side: its price and
size are NaN, never 0, and every variable derived from it is NaN while
that side is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quantlab.dataset._support.cleaning import NBBO_PANEL_VARIABLES
from quantlab.enums.data import BAR_INTERVAL_SECONDS

#: The output variables, in order. Shared with ``clean_nbbo_panel`` so the
#: resampler and the validator always agree.
PANEL_VARIABLES = NBBO_PANEL_VARIABLES

#: Count columns of the per-(date, symbol) filter-stats frame, after the
#: ``date`` and ``symbol`` keys; all Int64.
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

#: Drop reasons in priority order: a record matching several is counted
#: under the first one only.
_REASONS = ("nonpositive_price", "condition", "crossed", "locked")

#: Each time-weighted output variable and the record column it averages.
_TW_SOURCES = {
    "tw_spread": "spread",
    "tw_bid_size": "bid_size",
    "tw_ask_size": "ask_size",
}


@dataclass(frozen=True)
class NbboFilterPolicy:
    """Filter applied to NBBO records before resampling.

    A dropped record is ignored completely: the previous valid quote stays
    in force, and the record neither starts a session (as the *seed*, the
    last quote before the open) nor takes part in a tie between records with
    the same timestamp. A record matching several rules is counted under the
    first in the order nonpositive price, condition, crossed, locked, so
    each dropped record is counted exactly once.

    Parameters
    ----------
    drop_crossed : bool, default True
        Drop *crossed* quotes, where ``bid > ask``. Only checked when both
        sides are present.
    drop_locked : bool, default False
        Drop *locked* quotes, where ``bid == ask``. Only checked when both
        sides are present. Locked quotes do occur legitimately, so they are
        kept by default.
    drop_nonpositive_price : bool, default True
        Drop a record whose bid or ask, where present, is ``<= 0``. A null
        side is not a price and is not checked.
    keep_qu_cond : sequence of str, optional
        If set, keep only records whose quote condition code ``qu_cond`` is
        in this list; a null condition is in no list. Stored as a tuple.

    Examples
    --------
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
        """Convert ``keep_qu_cond`` to a tuple, refusing a bare string.

        Raises
        ------
        ValueError
            If ``keep_qu_cond`` is a string rather than a sequence of
            condition codes.
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

        Parameters
        ----------
        config : NbboDatasetConfig
            The dataset config holding ``drop_crossed``, ``drop_locked``,
            ``drop_nonpositive_price`` and ``keep_qu_cond``.

        Returns
        -------
        NbboFilterPolicy
            The policy.

        Examples
        --------
        >>> config = NbboDatasetConfig(
        ...     raw_data_dir_path="downloads/us_equity/tick/wrds_taq/wrds",
        ...     zarr_file_path="data/us_equity/tick/wrds_nbbo_1m.zarr",
        ...     reference_dir="downloads/_reference",
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
        """Return an expression giving each record's drop reason, or null if it is kept.

        The frame must have ``bid`` and ``ask`` columns, with a missing side
        already null, and a ``qu_cond`` column.

        Returns
        -------
        pl.Expr
            A string expression: ``"nonpositive_price"``, ``"condition"``,
            ``"crossed"``, ``"locked"`` or null.

        Examples
        --------
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

    Records are sorted by ``(symbol, date, timestamp, wrds_row_ord)`` with a
    stable sort, so the input row order never changes the output.
    ``wrds_row_ord`` is the position of the record in the order it was
    downloaded; it is the only tie-breaker for tables without a nanosecond
    field. All records at one instant count as updates, but only the last
    one in that order is kept as the quote: it is the quote in force after
    that instant, and the ones before it lasted zero time.

    Data from 2018 on is unique on ``(symbol, time_m, time_m_nano)``, so the
    sort key fully decides the order. Earlier tables only have microseconds,
    and a few percent of rows share a microsecond with a record holding
    different values, where download order is the only clue to the true
    order. ``n_ambiguous_ties`` counts, per bar, the records that share their
    timestamp with a record holding different values (identical duplicates
    count 0), so a user can see which bars depend on that order.

    Parameters
    ----------
    bar_interval : str
        A key of ``BAR_INTERVAL_SECONDS`` such as ``"1m"``.
    policy : NbboFilterPolicy, optional
        The record filter; ``NbboFilterPolicy()`` when ``None``.

    Attributes
    ----------
    bar_interval : str
        The bar size name.
    seconds : int
        The bar length in seconds.
    policy : NbboFilterPolicy
        The record filter in use.

    Raises
    ------
    ValueError
        If ``bar_interval`` is not a known bar size.

    Examples
    --------
    A three-minute session, one symbol, four records (the third is crossed
    and dropped by the default policy):

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
        """Initialize the resampler; see the class docstring for parameters."""
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

        Parameters
        ----------
        sessions : pl.DataFrame
            A frame with ``date``, ``open`` and ``close`` columns, the last
            two in naive UTC.

        Returns
        -------
        pl.DataFrame
            One row per bar label, in ascending order. The open itself is
            not a label.

        Examples
        --------
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
        """Return every bar boundary of each session, from open to close inclusive.

        Parameters
        ----------
        sessions : pl.DataFrame
            A frame with ``date``, ``open`` and ``close`` columns.

        Returns
        -------
        pl.DataFrame
            ``sessions`` with one row per boundary in an ``edge`` column.

        Raises
        ------
        ValueError
            If a session's length is not a whole, positive number of bars.
            A shorter last bar would differ in length from every other bar,
            and cutting it off would drop quotes from inside the session.
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
        """Return only the bar panel of ``resample_with_stats``.

        Parameters
        ----------
        records : pl.DataFrame
            NBBO records, as for ``resample_with_stats``.
        sessions : pl.DataFrame
            Session bounds, as for ``resample_with_stats``.

        Returns
        -------
        pl.DataFrame
            The bar panel.

        Examples
        --------
        >>> NbboResampler("1m").resample(records, sessions).shape
        (3, 16)
        """
        return self.resample_with_stats(records, sessions)[0]

    # ------------------------------------------------------------ pipeline --

    @staticmethod
    def _normalise(records: pl.DataFrame) -> pl.DataFrame:
        """Return the working columns with fixed types; a null or NaN price also nulls its size.

        The vendor columns are renamed to ``bid``, ``bid_size``, ``ask`` and
        ``ask_size``. A missing ``qu_cond`` column is added as all-null so the
        filter can always use it.
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
        # No quote on a side means no size on it either, whatever size came
        # with the null price.
        return frame.with_columns(
            pl.when(pl.col("bid").is_not_null()).then(pl.col("bid_size")).alias("bid_size"),
            pl.when(pl.col("ask").is_not_null()).then(pl.col("ask_size")).alias("ask_size"),
        )

    def _filter(self, records: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Apply the filter policy and return the kept records and the per-(date, symbol) drop counts."""
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

        Parameters
        ----------
        records : pl.DataFrame
            A frame with ``symbol``, ``date`` (session date), ``timestamp``
            (naive UTC), ``wrds_row_ord``, ``best_bid``,
            ``best_bidsizeshares``, ``best_ask``, ``best_asksizeshares`` and
            optionally ``qu_cond``.
        sessions : pl.DataFrame
            A frame with ``date``, ``open`` and ``close`` columns.

        Returns
        -------
        panel : pl.DataFrame
            Columns ``symbol``, ``date``, ``timestamp`` and the
            ``PANEL_VARIABLES`` as float64, with one row per bar label for
            every ``(symbol, date)`` in the input. A pair whose records were
            all filtered out still gets its rows, all NaN.
        stats : pl.DataFrame
            One row per input ``(date, symbol)`` with the
            ``FILTER_STATS_COUNTS`` columns.

        Examples
        --------
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
            # Stable sort; download order breaks timestamp ties.
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

        # A record is ambiguous when its (symbol, date, timestamp) group has
        # more than one distinct quote. Null sides compare as values, so two
        # identical one-sided records are not ambiguous. Counted after
        # filtering and before ties are collapsed.
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

        # n_updates counts every kept record inside the session (each tied
        # record is one update) in the bar whose span holds it. The seed is
        # not an update in any bar.
        updates = (
            records.filter(pl.col("timestamp") > pl.col("open"))
            .with_columns(edge_of_record.alias("edge"))
            .group_by(["symbol", "date", "edge"])
            .agg(
                pl.len().cast(pl.Float64).alias("n_updates"),
                pl.col("_ambiguous").sum().alias("n_ambiguous_ties"),
            )
        )

        # Collapse ties: the last record at each instant is the quote after
        # that instant. The earlier ones lasted zero time and must not be
        # matched by the as-of join below.
        records = records.unique(
            subset=tie, keep="last", maintain_order=True
        ).drop("_ambiguous")

        # Seed: of the records at or before the open, keep only the last and
        # move its timestamp to the open.
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

        # Running integrals for each time-weighted source: value times time,
        # and the time during which the value was defined.
        def exclusive_cum(expr: pl.Expr) -> pl.Expr:
            """Return the running sum of ``expr`` up to, not including, each row.

            Examples
            --------
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

        # Backward as-of join (match the last record at or before each
        # edge). With `by=` groups polars cannot check sort order and silently
        # mismatches unsorted input, so both sides are sorted right before.
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

            Examples
            --------
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
