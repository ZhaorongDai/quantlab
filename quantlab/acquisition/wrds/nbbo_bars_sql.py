"""The SQL statement that resamples TAQ NBBO records into bars on the WRDS server.

ADR 0027: over the slow link to WRDS, downloading every NBBO record and
resampling locally takes weeks for a decade of S&P 500 data, so the
resampling runs in SQL and only bars cross the network. ``NbboBarsQuery``
builds that statement for one day table and one batch of symbols. It
reproduces ``quantlab.dataset.nbbo.resample.NbboResampler`` step by step:

1. **Order.** ``row_number() OVER ()`` numbers the rows in the server's scan
   order before anything else touches them, the same assumption the tick
   path makes with ``wrds_row_ord``. Records are ordered by time (with
   ``time_m_nano`` from 2018 on), then by that number, which before 2018 is
   the only way to order records sharing a microsecond.
2. **Normalise.** A null or NaN price nulls its side and that side's size.
3. **Filter.** The rules of ``NbboFilterPolicy`` that are enabled; a dropped
   record is ignored completely. Records after the session close are
   dropped too.
4. **Bars.** Bar ``k`` covers ``(open + (k-1)d, open + kd]``; a record at or
   before the open falls in bar 0, the seed. The snapshot of bar ``k`` is
   the last kept record at or before its end, so records sharing an instant
   collapse to the last one and a bar with no update carries the quote in
   force. ``n_updates`` counts every kept record inside bar ``k``, tied ones
   included.
5. **Grid.** Every ``(ticker, k)`` for ``k = 1..N`` is returned for every
   ticker with a record that day, also when all its records were dropped,
   so the result has exactly ``tickers x N`` rows. ``page_rows``, the row
   count the server computed, is on every row, so the client can tell a
   truncated transfer from a complete one.

Times are compared as nanoseconds since midnight on TAQ's ``time_m`` clock
(New York); the client computes the session bounds from the XNYS calendar
and passes them in, so the server needs no calendar. The bar labels are
computed by the client from ``k``. ``mid``, ``spread``, ``spread_bps`` and
``imbalance`` are derived by the client from the snapshot. The
time-weighted variables and ``n_ambiguous_ties`` are returned as NULL for
now.

Every value enters the statement as a ``psycopg2.sql.Literal`` and every
name as a ``psycopg2.sql.Identifier``; nothing is pasted in as text. The
statement is always restricted by the ``(sym_root, sym_suffix)`` pairs
condition of ``WrdsSession.where_clause``, so it never scans a whole day
table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

from psycopg2 import sql

from quantlab.acquisition.wrds.taq import WrdsSession
from quantlab.dataset.nbbo.bars import (
    SERVER_BAR_VARIABLES,
    NbboBarsRequest,
    NbboBarsSession,
)

#: The CSV columns the statement returns, in order.
RESULT_COLUMNS = ("sym_root", "sym_suffix", "bar", *SERVER_BAR_VARIABLES, "page_rows")

#: The server variables not computed yet; returned as NULL of these types.
_NOT_YET_COMPUTED = {
    "tw_spread": "float8",
    "tw_bid_size": "float8",
    "tw_ask_size": "float8",
    "n_ambiguous_ties": "bigint",
}

_STATEMENT = """COPY (
WITH raw AS (
    SELECT sym_root,
           coalesce(sym_suffix, '') AS sym_suffix,
           row_number() OVER () AS ord,
           round(extract(epoch FROM time_m - time '00:00') * 1000000)::bigint
               * 1000 + {nano} AS t_ns,
           qu_cond, best_bid, best_bidsizeshares, best_ask, best_asksizeshares
    FROM {table}
    WHERE {where}
),
pairs AS (
    SELECT DISTINCT sym_root, sym_suffix FROM raw
),
typed AS (
    SELECT sym_root, sym_suffix, ord, t_ns, qu_cond,
           nullif(best_bid::float8, 'NaN') AS bid,
           best_bidsizeshares::float8 AS bid_size,
           nullif(best_ask::float8, 'NaN') AS ask,
           best_asksizeshares::float8 AS ask_size
    FROM raw
),
quotes AS (
    SELECT sym_root, sym_suffix, ord, t_ns, qu_cond, bid, ask,
           CASE WHEN bid IS NOT NULL THEN bid_size END AS bid_size,
           CASE WHEN ask IS NOT NULL THEN ask_size END AS ask_size
    FROM typed
),
kept AS (
    SELECT sym_root, sym_suffix, ord, t_ns, bid, bid_size, ask, ask_size,
           CASE WHEN t_ns <= {open_ns} THEN 0
                ELSE (t_ns - {open_ns} + {bar_ns} - 1) / {bar_ns} END AS bar
    FROM quotes
    WHERE t_ns <= {close_ns} AND NOT ({dropped})
),
last_in_bar AS (
    SELECT DISTINCT ON (sym_root, sym_suffix, bar)
           sym_root, sym_suffix, bar, bid, bid_size, ask, ask_size
    FROM kept
    ORDER BY sym_root, sym_suffix, bar, t_ns DESC, ord DESC
),
updates AS (
    SELECT sym_root, sym_suffix, bar, count(*) AS n_updates
    FROM kept
    WHERE bar >= 1
    GROUP BY sym_root, sym_suffix, bar
),
grid AS (
    SELECT pairs.sym_root, pairs.sym_suffix, k AS bar
    FROM pairs CROSS JOIN generate_series(0, {n_bars}) AS k
),
marked AS (
    SELECT grid.sym_root, grid.sym_suffix, grid.bar,
           l.bid, l.bid_size, l.ask, l.ask_size,
           count(l.bar) OVER (
               PARTITION BY grid.sym_root, grid.sym_suffix ORDER BY grid.bar
           ) AS held
    FROM grid
    LEFT JOIN last_in_bar AS l
        ON l.sym_root = grid.sym_root
       AND l.sym_suffix = grid.sym_suffix
       AND l.bar = grid.bar
),
snapshots AS (
    SELECT sym_root, sym_suffix, bar,
           first_value(bid) OVER quote AS bid,
           first_value(bid_size) OVER quote AS bid_size,
           first_value(ask) OVER quote AS ask,
           first_value(ask_size) OVER quote AS ask_size
    FROM marked
    WINDOW quote AS (PARTITION BY sym_root, sym_suffix, held ORDER BY bar)
)
SELECT s.sym_root, s.sym_suffix, s.bar,
       s.bid, s.bid_size, s.ask, s.ask_size,
       coalesce(u.n_updates, 0) AS n_updates,
       {not_yet_computed},
       count(*) OVER () AS page_rows
FROM snapshots AS s
LEFT JOIN updates AS u
    ON u.sym_root = s.sym_root
   AND u.sym_suffix = s.sym_suffix
   AND u.bar = s.bar
WHERE s.bar >= 1
) TO STDOUT WITH (FORMAT csv, HEADER true)"""


def _nanoseconds(clock: time) -> int:
    """Return a clock time as nanoseconds since midnight.

    Examples
    --------
    >>> _nanoseconds(time(9, 30))
    34200000000000
    """
    seconds = clock.hour * 3600 + clock.minute * 60 + clock.second
    return (seconds * 1_000_000 + clock.microsecond) * 1000


@dataclass(frozen=True)
class NbboBarsQuery:
    """The server-side resampling statement for one day table and one symbol batch.

    Parameters
    ----------
    day : datetime.date
        The trading day, which names the table.
    pairs : tuple of tuple of (str, str or None)
        The batch's ``(sym_root, sym_suffix)`` pairs.
    session : NbboBarsSession
        The day's session bounds, from ``NbboBarsRequest.session``.
    request : NbboBarsRequest
        The bar size and the quote filters.
    has_nano : bool
        Whether the day table has ``time_m_nano`` (from 2018-01-02).

    Raises
    ------
    ValueError
        If ``pairs`` is empty, or ``session`` is for another day.

    Examples
    --------
    >>> request = NbboBarsRequest("1m")
    >>> session = request.session(request.calendar(), date(2024, 1, 24))
    >>> query = NbboBarsQuery(
    ...     date(2024, 1, 24), (("AAPL", None),), session, request, has_nano=True
    ... )
    >>> statement = query.statement()  # a psycopg2.sql.Composed COPY statement
    """

    day: date
    pairs: tuple
    session: NbboBarsSession
    request: NbboBarsRequest
    has_nano: bool

    def __post_init__(self) -> None:
        """Freeze the pairs and check them against the session."""
        object.__setattr__(
            self, "pairs", tuple((str(root), suffix or None) for root, suffix in self.pairs)
        )
        if not self.pairs:
            raise ValueError(
                "NbboBarsQuery: no (sym_root, sym_suffix) pairs; refusing to "
                "build a statement over a whole complete_nbbo table."
            )
        if self.session.day != self.day:
            raise ValueError(
                f"NbboBarsQuery: the session is for {self.session.day}, the "
                f"table for {self.day}."
            )

    def _dropped(self) -> sql.Composable:
        """Return the condition true for a record the filter policy drops.

        Only the enabled rules appear. A null side is never a price, so each
        comparison with a null side counts as false.
        """
        policy = self.request.policy
        rules: list[sql.Composable] = []
        if policy.drop_nonpositive_price:
            rules.append(
                sql.SQL("coalesce(bid <= 0, false) OR coalesce(ask <= 0, false)")
            )
        if policy.keep_qu_cond is not None:
            rules.append(
                sql.SQL("NOT coalesce(qu_cond::text = ANY({codes}::text[]), false)").format(
                    codes=sql.Literal(list(policy.keep_qu_cond))
                )
            )
        if policy.drop_crossed:
            rules.append(sql.SQL("coalesce(bid > ask, false)"))
        if policy.drop_locked:
            rules.append(sql.SQL("coalesce(bid = ask, false)"))
        if not rules:
            return sql.SQL("false")
        return sql.SQL(" OR ").join(
            sql.SQL("({})").format(rule) for rule in rules
        )

    def statement(self) -> sql.Composed:
        """Return the ``COPY (...) TO STDOUT`` statement.

        Its CSV output has a header line and the ``RESULT_COLUMNS``.

        Returns
        -------
        psycopg2.sql.Composed
            The statement.

        Examples
        --------
        With ``query`` built as in the class example::

            query.statement()   # Composed([SQL("COPY (\\nWITH raw AS (...
        """
        bar_ns = self.request.bar_seconds * 1_000_000_000
        nano = (
            sql.SQL("coalesce(time_m_nano, 0)") if self.has_nano else sql.SQL("0")
        )
        not_yet_computed = sql.SQL(", ").join(
            sql.SQL("NULL::{kind} AS {name}").format(
                kind=sql.SQL(kind), name=sql.Identifier(name)
            )
            for name, kind in _NOT_YET_COMPUTED.items()
        )
        return sql.SQL(_STATEMENT).format(
            nano=nano,
            table=WrdsSession.table_identifier(self.day),
            where=WrdsSession.where_clause(self.pairs),
            open_ns=sql.Literal(_nanoseconds(self.session.open_clock)),
            close_ns=sql.Literal(_nanoseconds(self.session.close_clock)),
            bar_ns=sql.Literal(bar_ns),
            n_bars=sql.Literal(self.session.n_bars),
            dropped=self._dropped(),
            not_yet_computed=not_yet_computed,
        )
