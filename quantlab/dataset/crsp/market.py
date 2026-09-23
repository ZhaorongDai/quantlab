"""The WHOLE-MARKET CRSP roster: every security, not an index's members.

`CrspMembership` answers "who was in this index"; this module answers "what
securities existed at all". The two are siblings, not a base and a subclass,
because they are sourced from different tables and mean different things --
folding a market roster into a class whose every docstring says "membership"
would make `crsp_all` read as an index with 40,518 members.

**Where the roster comes from, and why it costs nothing extra.**
`crsp_a_stock.stksecurityinfohist` is already pulled unconditionally by
`CrspReferenceTables.STOCK_TABLES`, because symbology needs it (D-04). It
carries one interval per security-info change with `permno`, the interval's
`(secinfostartdt, secinfoenddt)` and every column
`SECURITY_FILTER_PRESETS` names. So a whole-market roster is a read of a
table this project already has on disk: no new query, no new schema
entitlement, no extra round trip.

**Spell-level verdict, not per-date.** The type predicate is evaluated on each
`stksecurityinfohist` interval, and a PERMNO joins the roster when ANY of its
intervals qualifies. The per-DAY verdict stays where it already lives --
`CrspStockDataset._apply_security_filter`, reading `dsf_v2`'s own per-day type
columns, which is what lets a security keep exactly the era in which it was
common stock (D-17). This split is the existing division of labour, stated by
`resolve_security_filter`'s own refusal text ("To restrict the ROSTER ..."):
the roster decides who gets PULLED, the conversion decides which of their rows
survive. Evaluating the day-level rule here as well would be a second place
that can disagree with the first.

**Overlap, not containment.** `permnos_in_range` keeps every PERMNO whose
interval OVERLAPS the window (`start_date <= end AND end_date >= start`) --
the same predicate `CrspMembership.permnos_in_range` and
`UniverseCatalog.get_symbols_in_range` use. A security that was delisted
inside the window stays in the roster; dropping it is precisely the
survivorship bias this layer exists to remove, and on a whole-market roster
there are far more of them than in any index.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from loguru import logger

from quantlab.dataset.crsp import resolve_security_filter
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.utils.symbol_axis import sort_symbol_axis

# Imported rather than re-written. `_merge_intervals` IS the merge contract
# (closed intervals, touching means `next.start <= previous.end + 1 day`), and
# `_as_date` IS the date-coercion contract; `CrspMembership.permnos_in_range`'s
# own docstring records what happens when one contract is re-derived in several
# places. A second copy here would be a second thing to keep in step. If a
# third caller appears, these two belong in `quantlab/dataset/_support/`, and
# that move is a rename rather than a rewrite.
from quantlab.dataset.crsp.membership import _as_date, _merge_intervals

#: How many PERMNOs `stksecurityinfohist` carries in the vintage this project
#: pulls -- the upper bound on any whole-market roster, BEFORE the type filter.
#: Recorded so a roster that comes back the same order of magnitude as an index
#: is visibly wrong. The number is the one already stated in
#: `quantlab/dataset/crsp/__init__.py`'s ticker-sidecar docstring.
SECINFO_PERMNO_COUNT_HINT: int = 40_518

#: The name this roster answers to on a CLI, beside `CrspMembership.INDEXES`.
#: Deliberately NOT added to `CrspMembership.INDEXES`: that tuple is what the
#: index-membership CLI offers, and a whole-market roster has no membership
#: panel behind it in the sense those two do.
MARKET = "crsp_all"


class CrspMarketRoster:
    """Every CRSP security, point-in-time, over one reference directory.

    Holds no connection and needs no credential: `reference` is a
    `CrspReference`, i.e. parquet on disk. Mirrors `CrspMembership`'s public
    shape on purpose -- `permno_intervals()` / `permnos_in_range()` / `report`
    -- so a caller can hold either behind the same two calls, and `report`
    describes the MOST RECENT `permno_intervals()` call.
    """

    #: The table this roster is read from. Always present in a reference tier.
    SOURCE_TABLE = "stksecurityinfohist"

    def __init__(self, reference: CrspReference) -> None:
        self.reference = reference
        #: What was excluded, clipped or filtered.
        self.report: dict = {}
        self._reset_report()

    def _reset_report(self) -> None:
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
        """`(permno Int64, start_date Date, end_date Date)`, sorted.

        One row per continuous QUALIFYING span of one PERMNO, both ends
        inclusive, `end_date` never null and never later than
        `CrspReference.product_end`.

        Adjacent qualifying intervals are merged, so an ordinary common share
        that changed ticker four times is ONE span, not four. Non-qualifying
        intervals are dropped BEFORE the merge, which is what makes a security
        that was an ADR and later ordinary common come back as the ordinary
        stretch alone rather than one span covering both.

        Unlike `CrspMembership.permno_intervals` this takes no `window`: that
        parameter exists there to scope the Nasdaq-100 unlinked-spell refusal,
        and this roster has no link table and therefore no such refusal.
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
        """Every PERMNO whose qualifying span OVERLAPS `[start_date, end_date]`.

        The overlap predicate is `start_date <= end AND end_date >= start` --
        the same one `CrspMembership.permnos_in_range` uses, and the reason a
        security delisted inside the window stays in the roster.

        The order is NUMERIC and is part of the contract; it comes from
        `quantlab.utils.symbol_axis.sort_symbol_axis`, the single source of
        that contract, rather than a bare `sorted()` over digit strings (which
        would put "14593" before "7000" and move the acquisition batch
        boundaries between two runs of the same command).
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
        """`stksecurityinfohist` -> `(permno, start, end)` for qualifying spells.

        The type predicate is the same shape
        `CrspStockDataset._apply_security_filter` builds: every listed column
        must match and a NULL never matches, because "unknown type" is not
        "the type you asked for". An empty filter (`security_filter="none"`)
        keeps every spell, which is the whole 40,518-PERMNO table.

        The NULL rejection is stated twice -- `.fill_null(False)` below, and
        the falsy `if not row["_keep"]` that reads it -- and a mutation run
        showed either alone suffices. The expression keeps its `.fill_null`
        so it means the right thing on its own, rather than being correct only
        because of how the loop below happens to read it; the loop keeps its
        falsy test because that is also what it does for a `_keep` that is
        legitimately False. Neither is dead code, but no test can tell them
        apart, and `tests/test_crsp_market_roster.py` says so rather than
        claiming a coverage it does not have.

        Spells are filtered BEFORE `_merge_intervals` sees them, never after.
        A security that was ordinary common, became an ADR, and became
        ordinary again must come back as two spans with a real hole; merging
        first would bridge the ADR era and assert the security was ordinary
        common throughout.
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
        """The public frame shape, typed even when there are no intervals."""
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
