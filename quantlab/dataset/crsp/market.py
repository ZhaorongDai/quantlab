"""The whole-market CRSP roster: every security, not an index's members.

``CrspMembership`` answers "who was in this index"; ``CrspMarketRoster``
answers "which securities existed at all". They are siblings with the same
public shape (``permno_intervals()``, ``permnos_in_range()``, ``report``),
sourced from different tables, so a caller can hold either behind the same
two calls.

The roster is read from ``crsp_a_stock.stksecurityinfohist``, which the
reference tier already holds for ticker lookups. Each row of that table is
one interval of a security's history carrying the type columns a security
filter names, so a whole-market roster costs no extra query.

The type filter is applied per interval, and a PERMNO joins the roster when
any of its intervals qualifies. The per-day verdict (which of a security's
daily rows survive) is made later by the dataset's own security filter
against the daily table; the roster only decides who gets pulled.

Overlap, not containment: ``permnos_in_range`` keeps every PERMNO whose
qualifying interval overlaps the window, so a security delisted inside the
window stays in the roster. Dropping it would be survivorship bias, and on a
whole-market roster there are far more such securities than in any index.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from loguru import logger

from quantlab.dataset.crsp import resolve_security_filter
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.utils.symbol_axis import sort_symbol_axis

# Shared with `CrspMembership` rather than copied: `_merge_intervals` defines
# what "touching" means for closed intervals, and `_as_date` the date
# coercion. A second copy here would be a second thing to keep in step.
from quantlab.dataset.crsp.membership import _as_date, _merge_intervals

#: How many PERMNOs ``stksecurityinfohist`` carries in the vintage this
#: project pulls, before any type filter: the upper bound on a whole-market
#: roster. A roster the size of an index is visibly wrong against it.
SECINFO_PERMNO_COUNT_HINT: int = 40_518

#: The name this roster answers to on a CLI, beside ``CrspMembership.INDEXES``.
#: It is not added to that tuple, because a whole-market roster has no
#: membership panel behind it in the sense the two indexes do.
MARKET = "crsp_all"


class CrspMarketRoster:
    """Every CRSP security, point-in-time, over one reference directory.

    Holds no connection and needs no credential: ``reference`` is a
    ``CrspReference`` over parquet on disk. ``report`` describes the most
    recent ``permno_intervals()`` call.

    Example:
        >>> from quantlab.dataset.crsp.market import CrspMarketRoster
        >>> from quantlab.dataset.crsp.reference import CrspReference
        >>> ref = CrspReference("data/downloads/us_equity/1d/wrds_crsp/_reference")
        >>> roster = CrspMarketRoster(ref)
        >>> roster.permnos_in_range("2010-01-01", "2010-12-31")
        ['10001', '10002']
        >>> roster.report["permnos_after_type_filter"]
        2
    """

    #: The table this roster is read from. Always present in a reference tier.
    SOURCE_TABLE = "stksecurityinfohist"

    def __init__(self, reference: CrspReference) -> None:
        """Bind a reference directory and start with an empty report."""
        self.reference = reference
        #: What the last ``permno_intervals()`` call excluded, clipped or
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
        """Return qualifying spans as ``(permno, start_date, end_date)``.

        One row per continuous qualifying span of one PERMNO, sorted, both
        ends inclusive, ``end_date`` never null and never later than
        ``CrspReference.product_end``. Adjacent qualifying intervals are
        merged, so a common share that changed ticker four times is one span.
        Non-qualifying intervals are dropped before the merge, so a security
        that was an ADR and later ordinary common comes back as the ordinary
        stretch alone.

        Args:
            security_filter: A preset name (``"equity_common"``,
                ``"shrcd_10_11"``, ``"none"``) or a ``{column: allowed
                values}`` mapping, resolved by ``resolve_security_filter``.

        Raises:
            ValueError: If ``security_filter`` is not a valid preset or
                mapping, or if a source row has a null PERMNO or start.

        Example:
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
        """Return every PERMNO whose qualifying span overlaps the window.

        The window is ``[start_date, end_date]``, both ends inclusive. The
        test is overlap (``start_date <= end and end_date >= start``), the
        same one ``CrspMembership.permnos_in_range`` uses, so a security
        delisted inside the window is still listed.

        The order is numeric, as ``quantlab.utils.symbol_axis.sort_symbol_axis``
        defines it, not lexicographic: ``"7000"`` sorts before ``"14593"``.

        Args:
            start_date: Window start, as a ``date`` or ISO string.
            end_date: Window end, inclusive.
            security_filter: As for ``permno_intervals``.

        Returns:
            PERMNOs as strings, in numeric order.

        Raises:
            ValueError: If ``start_date`` is after ``end_date``.

        Example:
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
        """Read ``stksecurityinfohist`` into ``(permno, start, end)`` qualifying spells.

        Every filtered column must match and a null never matches, because
        "unknown type" is not "the type you asked for". An empty filter keeps
        every spell. Spells are filtered before ``_merge_intervals`` sees
        them: a security that was ordinary common, became an ADR, and became
        ordinary again must come back as two spans with a real hole, and
        merging first would bridge the ADR era.

        The null rejection appears twice, as ``.fill_null(False)`` in the
        expression and as the falsy ``if not row["_keep"]`` that reads it.
        Either alone suffices; both are kept so each reads correctly on its
        own.
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
        """Return the public frame shape, correctly typed even when empty."""
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
