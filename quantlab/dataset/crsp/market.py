"""The CRSP market roster: every listed security, not an index's members.

CRSP (the Center for Research in Security Prices) is a US stock database
sold through WRDS (Wharton Research Data Services). A PERMNO is CRSP's
permanent integer id for one security. A *roster* is the list of PERMNOs
to download, with the dates each one qualifies. A *spell* is one row of a
CRSP history table: an interval of dates over which a security's
descriptive fields stayed the same.

``CrspMembership`` answers "who was in this index"; ``CrspMarketRoster``
answers "which securities existed at all". Both have the same public
interface (``permno_intervals``, ``permnos_in_range``, ``report``) but read
different tables, so a caller can use either one the same way.

The roster is read from ``crsp_a_stock.stksecurityinfohist``, which the
downloaded reference tables already include for ticker lookups. Each row is
one spell of a security's history and carries the type columns a security
filter checks, so a market roster needs no extra download.

The type filter is applied to each spell, and a PERMNO joins the roster if
any of its spells qualifies. Which of a security's daily rows are kept is
decided later, by the dataset's own security filter on the daily table;
the roster only decides which securities are downloaded.

``permnos_in_range`` keeps every PERMNO whose qualifying interval overlaps
the window, not only those that cover it fully, so a security delisted
inside the window stays in the roster. Dropping it would cause
*survivorship bias* (history made to look better by forgetting the
companies that failed), and a market roster has far more such
securities than any index.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from loguru import logger

from quantlab.dataset.crsp import resolve_security_filter
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.utils.symbol_axis import sort_symbol_axis

# Shared with `CrspMembership` rather than copied: `_merge_intervals` decides
# when two closed intervals touch, and `_as_date` converts dates. A copy here
# would have to be kept in step by hand.
from quantlab.dataset.crsp.membership import _as_date, _merge_intervals

#: How many PERMNOs ``stksecurityinfohist`` holds in the version this
#: project downloads, before any type filter. It is the upper bound for a
#: market roster; a roster the size of an index is clearly wrong.
SECINFO_PERMNO_COUNT_HINT: int = 40_518


class CrspMarketRoster:
    """Every CRSP security and the dates it was listed, from one reference directory.

    The roster is *point-in-time*: it records when each security was listed,
    not only whether it is listed today. It holds no database connection and
    needs no login, because it only reads parquet files on disk.

    Parameters
    ----------
    reference : CrspReference
        The downloaded CRSP reference tables.

    Attributes
    ----------
    reference : CrspReference
        The reference tables given to the constructor.
    report : dict
        Counts from the most recent ``permno_intervals`` call: spells read,
        dropped by type, dropped or clipped at the product end, and PERMNOs
        before and after the type filter.

    Examples
    --------
    >>> from quantlab.dataset.crsp.market import CrspMarketRoster
    >>> from quantlab.dataset.crsp.reference import CrspReference
    >>> ref = CrspReference("data/downloads/us_equity/1d/wrds_crsp/_reference")
    >>> roster = CrspMarketRoster(ref)
    >>> roster.permnos_in_range("2010-01-01", "2010-12-31")
    ['10001', '10002']
    >>> roster.report["permnos_after_type_filter"]
    2
    """

    #: The table this roster is read from; every reference download has it.
    SOURCE_TABLE = "stksecurityinfohist"

    def __init__(self, reference: CrspReference) -> None:
        """Initialize the roster; see the class docstring for parameters."""
        self.reference = reference
        #: What the last ``permno_intervals`` call excluded, clipped or
        #: filtered.
        self.report: dict = {}
        self._reset_report()

    def _reset_report(self) -> None:
        """Reset every report key to its empty value."""
        self.report.clear()
        self.report.update(
            {
                "security_filter": {},
                "spells_read": 0,
                "spells_dropped_by_type": 0,
                "spells_dropped_after_product_end": 0,
                "spells_clipped_to_product_end": 0,
                "permnos_before_type_filter": 0,
                "permnos_after_type_filter": 0,
            }
        )

    # -- public ---------------------------------------------------------------

    def permno_intervals(
        self, *, security_filter: str | dict = "equity_common"
    ) -> pl.DataFrame:
        """Return the qualifying listing intervals as ``(permno, start_date, end_date)``.

        There is one row per continuous qualifying interval of one PERMNO,
        sorted, with both ends inclusive. ``end_date`` is never null and
        never later than ``CrspReference.product_end`` (the last date of the
        downloaded CRSP data). Adjacent qualifying spells are merged, so a
        common share that changed ticker four times is one interval. Spells
        that do not qualify are dropped before merging, so a security that
        was an ADR (a foreign share traded in the US) and later ordinary
        common stock comes back as the common-stock period only.

        Parameters
        ----------
        security_filter : str or dict, default "equity_common"
            A preset name (``"equity_common"``, ``"shrcd_10_11"``,
            ``"none"``) or a ``{column: allowed values}`` mapping, resolved
            by ``resolve_security_filter``.

        Returns
        -------
        pl.DataFrame
            Columns ``permno`` (Int64), ``start_date`` and ``end_date``
            (Date).

        Raises
        ------
        ValueError
            If ``security_filter`` is not a valid preset or
            mapping, or if a source row has a null PERMNO or start.

        Examples
        --------
        >>> roster.permno_intervals()
        shape: (2, 3)
        ┌────────┬────────────┬────────────┐
        │ permno ┆ start_date ┆ end_date   │
        │ ---    ┆ ---        ┆ ---        │
        │ i64    ┆ date       ┆ date       │
        ╞════════╪════════════╪════════════╡
        │ 10001  ┆ 2000-01-03 ┆ 2025-12-31 │
        │ 10002  ┆ 2000-01-03 ┆ 2010-06-30 │
        └────────┴────────────┴────────────┘
        """
        resolved = resolve_security_filter(
            security_filter, owner=type(self).__name__
        )
        self._reset_report()
        self.report["security_filter"] = {
            column: list(allowed) for column, allowed in resolved.items()
        }
        pieces = self._market_pieces(resolved)
        frame = self._frame(_merge_intervals(pieces))
        logger.debug(
            "{}: {} spell(s) -> {} PERMNO(s) after filter {}",
            type(self).__name__,
            self.report["spells_read"],
            self.report["permnos_after_type_filter"],
            self.report["security_filter"] or "{} (none)",
        )
        return frame

    def permnos_in_range(
        self,
        start_date,
        end_date,
        *,
        security_filter: str | dict = "equity_common",
    ) -> list[str]:
        """Return every PERMNO whose qualifying interval overlaps the window.

        The window is ``[start_date, end_date]``, both ends inclusive. The
        test is overlap (``start_date <= end and end_date >= start``), as in
        ``CrspMembership.permnos_in_range``, so a security delisted inside
        the window is still included.

        The order is numeric, as ``quantlab.utils.symbol_axis.sort_symbol_axis``
        defines it, not lexicographic: ``"7000"`` sorts before ``"14593"``.

        Parameters
        ----------
        start_date : date or str
            Window start, as a ``date`` or ISO string.
        end_date : date or str
            Window end, inclusive.
        security_filter : str or dict, default "equity_common"
            As for ``permno_intervals``.

        Returns
        -------
        list of str
            PERMNOs as strings, in numeric order.

        Raises
        ------
        ValueError
            If ``start_date`` is after ``end_date``.

        Examples
        --------
        >>> roster.permnos_in_range("2011-01-01", "2011-12-31")
        ['10001']
        >>> roster.permnos_in_range(
        ...     "2011-01-01", "2011-12-31", security_filter="none"
        ... )
        ['10001', '10004']
        """
        start = _as_date(start_date)
        end = _as_date(end_date)
        if start > end:
            raise ValueError(
                f"{type(self).__name__}: start_date {start} is after end_date "
                f"{end}; an inverted window overlaps nothing and would return "
                f"an empty roster indistinguishable from a real one."
            )
        intervals = self.permno_intervals(security_filter=security_filter)
        overlapping = intervals.filter(
            (pl.col("start_date") <= end) & (pl.col("end_date") >= start)
        )
        return [
            str(permno)
            for permno in sort_symbol_axis(set(overlapping["permno"].to_list()))
        ]

    # -- internals ------------------------------------------------------------

    def _market_pieces(
        self, resolved: dict[str, tuple[str, ...]]
    ) -> list[tuple[int, date, date]]:
        """Read ``stksecurityinfohist`` into qualifying ``(permno, start, end)`` spells.

        Every filtered column must match, and a null never matches, because
        "unknown type" is not "the type you asked for". An empty filter keeps
        every spell. Spells are filtered before ``_merge_intervals`` sees
        them. A security that was common stock, became an ADR, and became
        common again must come back as two intervals with a real gap;
        merging first would cover the ADR period.

        Spells that start after the product end are dropped, and open or
        later ends are clipped to it; ``report`` counts both.

        Nulls are rejected twice, by ``.fill_null(False)`` in the expression
        and by ``if not row["_keep"]``, which treats null as false. Either
        alone is enough; both are kept so each line is correct on its own.

        Parameters
        ----------
        resolved : dict of str to tuple of str
            The resolved security filter, ``{column: allowed values}``.

        Returns
        -------
        list of tuple
            Qualifying ``(permno, start, end)`` spells.

        Raises
        ------
        ValueError
            If a row has a null PERMNO or start date, or ends before it
            starts.
        """
        product_end = self.reference.product_end
        table = self.reference.table(self.SOURCE_TABLE)

        keep = pl.lit(True)
        for column, allowed in resolved.items():
            keep = keep & pl.col(column).is_in(list(allowed)).fill_null(False)

        rows = table.select(
            pl.col("permno").cast(pl.Int64),
            pl.col("secinfostartdt").cast(pl.Date).alias("start"),
            pl.col("secinfoenddt").cast(pl.Date).alias("end"),
            keep.alias("_keep"),
        ).to_dicts()

        self.report["spells_read"] = len(rows)
        seen_before: set[int] = set()
        seen_after: set[int] = set()

        pieces: list[tuple[int, date, date]] = []
        for row in rows:
            permno, start = row["permno"], row["start"]
            if permno is None or start is None:
                raise ValueError(
                    f"{type(self).__name__}: a {self.SOURCE_TABLE} row has a "
                    f"null permno or secinfostartdt ({row!r}). A security "
                    f"with no identity or no start cannot be placed on a "
                    f"calendar, and tolerating it would put a PERMNO into the "
                    f"roster whose span is unknown."
                )
            seen_before.add(permno)
            if not row["_keep"]:
                self.report["spells_dropped_by_type"] += 1
                continue
            if start > product_end:
                self.report["spells_dropped_after_product_end"] += 1
                continue
            end = row["end"]
            if end is None or end > product_end:
                self.report["spells_clipped_to_product_end"] += 1
                end = product_end
            if end < start:
                raise ValueError(
                    f"{type(self).__name__}: {self.SOURCE_TABLE} permno "
                    f"{permno} has secinfoenddt {end} before secinfostartdt "
                    f"{start}."
                )
            seen_after.add(permno)
            pieces.append((permno, start, end))

        self.report["permnos_before_type_filter"] = len(seen_before)
        self.report["permnos_after_type_filter"] = len(seen_after)
        return pieces

    @staticmethod
    def _frame(intervals: list[tuple[int, date, date]]) -> pl.DataFrame:
        """Build the ``(permno, start_date, end_date)`` frame, typed correctly even when empty."""
        return pl.DataFrame(
            {
                "permno": [permno for permno, _, _ in intervals],
                "start_date": [start for _, start, _ in intervals],
                "end_date": [end for _, _, end in intervals],
            },
            schema={
                "permno": pl.Int64,
                "start_date": pl.Date,
                "end_date": pl.Date,
            },
        )
