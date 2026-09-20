"""The two CRSP-vendor point-in-time universes, keyed by PERMNO (D-05, D-14).

This vendor offers exactly two index histories, and neither of them is a
Wikipedia change log:

- **`crsp_sp500`** -- CRSP's OWN S&P 500 membership, `crsp_a_indexes.dsp500list_v2`,
  one spell per `(permno, mbrstartdt, mbrenddt)`. Coverage runs from
  1925-12-31, the start of index family 1100500 (D-05).
- **`comp_nasdaq100`** -- Compustat's Nasdaq-100 constituent history,
  `comp.idxcst_his` at `gvkeyx = '000208'`, whose `gvkey`/`iid` pairs become
  PERMNOs through the CRSP/Compustat Merged link table
  `crsp_a_ccm.ccmxpf_lnkhist` (D-14).

**Keyed by PERMNO, deliberately.** A universe answered in tickers has to be
re-answered every time a security is renamed, and the rename is exactly where
a membership panel and a price panel stop agreeing (the standing `no ticker
rename mapping between membership history and prices` todo). The PERMNO does
not change on a rename, so the roster the CLI pulls and the collision
tie-break plan 08 applies both work in an identifier that is stable across the
event. Plan 09 maps these intervals onto `CrspSymbology`'s symbol intervals
for the panel side, where period-correct tickers are what a mask needs.

**Intervals are CLOSED on both ends, and the end is ALWAYS explicit.** Closed
matches `IndexConstituentDataset`'s convention verbatim
(`quantlab/base/constituent.py`: "Membership intervals are CLOSED on both
ends"). Explicit is the load-bearing half: `_densify`'s open-interval branch
sets the panel's right edge to `max(observed, today)` -- WALL-CLOCK today --
so a single null end would extend a CRSP universe months past the CRSP price
coverage it is supposed to be bounded by. Every end here is therefore either
the source's own end or `CrspReference.product_end`, never null.

**`quantlab/acquisition/universe.py` is untouched, on purpose.** The CRSP
universes live beside the existing ones rather than inside `UniverseCatalog`:
that module's structural "imports no acquisition module" rule is enforced by
an AST scan, its categories are Tiingo/Wikipedia-shaped (tickers, exchange
filters), and this one is a dataset-layer reader over parquet. Adding a
category would have coupled two vocabularies for no gain.

**A LEAF module.** polars, loguru, stdlib and `crsp_reference` only -- no
psycopg2, no acquisition import. The universes must resolve on a machine with
no WRDS credential.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from loguru import logger

from quantlab.dataset.crsp_reference import CrspReference

_ONE_DAY = timedelta(days=1)

#: How many unlinked spells the refusal lists before summarising the rest.
#: A full listing of a systematically broken link table would bury the remedy
#: sentence that follows it.
_MAX_LISTED = 20


def _as_date(value) -> date:
    """`"2020-01-02"` or a `date` -> `date`. Anything else raises."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _merge_intervals(
    pieces: list[tuple[int, date, date]],
) -> list[tuple[int, date, date]]:
    """Merge overlapping or TOUCHING intervals, per PERMNO.

    Touching means `next.start <= previous.end + 1 day`: the intervals are
    closed, so 2005-12-31 and 2006-01-01 are one continuous membership with no
    day between them. Anything wider stays two intervals -- a security that
    left the index and rejoined was genuinely not a member in between, and
    bridging the hole would fabricate membership.
    """
    merged: list[tuple[int, date, date]] = []
    for permno, start, end in sorted(pieces):
        if merged and merged[-1][0] == permno and start <= merged[-1][2] + _ONE_DAY:
            previous = merged[-1]
            merged[-1] = (permno, previous[1], max(previous[2], end))
        else:
            merged.append((permno, start, end))
    return merged


def _uncovered(
    start: date, end: date, covered: list[tuple[date, date]]
) -> list[tuple[date, date]]:
    """The calendar-day ranges of `[start, end]` that `covered` does not cover.

    Plain interval subtraction on closed ranges. `covered` may overlap itself
    and need not be sorted.
    """
    gaps: list[tuple[date, date]] = []
    cursor = start
    for piece_start, piece_end in sorted(covered):
        if piece_start > cursor:
            gaps.append((cursor, min(piece_start - _ONE_DAY, end)))
        cursor = max(cursor, piece_end + _ONE_DAY)
        if cursor > end:
            break
    if cursor <= end:
        gaps.append((cursor, end))
    return [(gap_start, gap_end) for gap_start, gap_end in gaps if gap_start <= gap_end]


class CrspMembership:
    """Point-in-time index membership over one CRSP reference directory.

    Holds no connection: `reference` is a `CrspReference`, i.e. parquet on
    disk. Every public method reports what it excluded through `self.report`,
    which describes the MOST RECENT `permno_intervals()` call -- the six keys
    below are reset on entry, and any key a later layer adds (plan 09's
    `unlabelled_members`) survives.
    """

    #: CRSP's own S&P 500 membership (D-05).
    SP500 = "crsp_sp500"
    #: Compustat's Nasdaq-100 membership, linked through CCM (D-14).
    NASDAQ100 = "comp_nasdaq100"
    #: Every universe this vendor serves, in the order the CLI lists them.
    INDEXES = (SP500, NASDAQ100)

    #: The earliest date each universe can be answered for AT ALL. The S&P
    #: start is index family 1100500's own start; the Nasdaq-100 one is a
    #: CENSOR rather than a start -- 100 spells begin exactly 1995-01-01
    #: (live check `L8_3`), which is where Compustat's history begins, not
    #: where those memberships did. Even censored it is twelve years earlier
    #: than the Wikipedia Nasdaq-100 panel's 2007-02-01.
    PIT_COVERAGE_START = {SP500: "1925-12-31", NASDAQ100: "1995-01-01"}

    #: The S&P 500's index number in `dsp500list_v2`; the table carries other
    #: indexes and a row of any other `indno` is not this universe.
    SP500_INDNO = 1000500

    #: `comp.idx_index`'s gvkeyx for "Nasdaq 100".
    NDX_GVKEYX = "000208"

    #: The CCM link types that assert a real gvkey <-> PERMNO identity. `NR`
    #: ("no research") and `NU` rows carry a NULL `lpermno` and are not links.
    LINK_TYPES = ("LC", "LU", "LS")

    #: A membership stretch of at most this many CALENDAR days between two
    #: links of one `(gvkey, iid)` is a seam in the link table, not a hole in
    #: the universe: a long weekend plus a holiday is four days and no trading
    #: happens in it. Wider than that, the days are genuinely unlinked.
    LINK_GAP_TOLERANCE_DAYS = 4

    def __init__(self, reference: CrspReference) -> None:
        self.reference = reference
        #: What was excluded, clipped, tolerated or left unlinked.
        self.report: dict = {}
        self._reset_report()

    def _reset_report(self) -> None:
        self.report.update(
            {
                "other_indno_rows": 0,
                "dropped_after_product_end": 0,
                "clipped_to_product_end": 0,
                "left_censored_spells": 0,
                "tolerated_gaps": [],
                "unlinked": [],
            }
        )

    # -- public -------------------------------------------------------------

    def permno_intervals(
        self, index: str, *, allow_unlinked: bool = False
    ) -> pl.DataFrame:
        """`(permno Int64, start_date Date, end_date Date)`, sorted.

        One row per continuous membership of one PERMNO, both ends inclusive,
        `end_date` never null and never later than
        `CrspReference.product_end`. `allow_unlinked` only affects the
        Nasdaq-100 branch (see `_nasdaq100_pieces`).
        """
        if index not in self.INDEXES:
            raise ValueError(
                f"{type(self).__name__}: {index!r} is not a CRSP universe; "
                f"this vendor serves {self.INDEXES}. An unknown name cannot "
                f"be answered with an empty roster -- an empty roster is a "
                f"legitimate answer for a REAL index, so a typo would be "
                f"indistinguishable from 'nobody was a member'."
            )
        self._reset_report()
        if index == self.SP500:
            pieces = self._sp500_pieces()
        else:
            pieces = self._nasdaq100_pieces(allow_unlinked=allow_unlinked)
        return self._frame(_merge_intervals(pieces))

    def permnos_in_range(
        self,
        index: str,
        start_date,
        end_date,
        *,
        allow_unlinked: bool = False,
    ) -> list[str]:
        """Every PERMNO whose membership OVERLAPS `[start_date, end_date]`.

        This is the roster a full-window backfill pulls: the overlap predicate
        is `start_date <= end AND end_date >= start`, the same one
        `UniverseCatalog.get_symbols_in_range` uses, and it is what keeps
        every security that left the index INSIDE the window -- dropping them
        is precisely the survivorship bias this layer removes.

        **The order is part of the contract, and it is NUMERIC.** PERMNOs are
        integers rendered as strings, so `sorted()` on the text would put
        `"14593"` before `"7000"`. `quantlab/utils/cli.py:resolve_symbols`
        slices this list for `--limit`; an unstable or surprising order
        truncates to a different batch on every run, and the second run never
        meets the watermarks the first one wrote.
        """
        start = _as_date(start_date)
        end = _as_date(end_date)
        if start > end:
            raise ValueError(
                f"{type(self).__name__}: start_date {start} is after end_date "
                f"{end}; an inverted window overlaps nothing and would return "
                f"an empty roster indistinguishable from a real one."
            )
        intervals = self.permno_intervals(index, allow_unlinked=allow_unlinked)
        overlapping = intervals.filter(
            (pl.col("start_date") <= end) & (pl.col("end_date") >= start)
        )
        return [str(permno) for permno in sorted(set(overlapping["permno"].to_list()))]

    # -- S&P 500 (D-05) ------------------------------------------------------

    def _sp500_pieces(self) -> list[tuple[int, date, date]]:
        """`dsp500list_v2` -> `(permno, start, end)` with explicit ends.

        Open membership in this table is already the product end rather than a
        NULL (live check `L9_1`: 503 rows read `mbrenddt = 2025-12-31`, no
        NULLs), so the NULL branch below is a guard against a future vintage
        rather than a path today's data takes.
        """
        product_end = self.reference.product_end
        rows = (
            self.reference.table("dsp500list_v2")
            .select(
                pl.col("permno").cast(pl.Int64),
                pl.col("indno").cast(pl.Int64),
                pl.col("mbrstartdt").cast(pl.Date).alias("start"),
                pl.col("mbrenddt").cast(pl.Date).alias("end"),
            )
            .to_dicts()
        )

        pieces: list[tuple[int, date, date]] = []
        for row in rows:
            if row["indno"] != self.SP500_INDNO:
                self.report["other_indno_rows"] += 1
                continue
            permno, start = row["permno"], row["start"]
            if permno is None or start is None:
                raise ValueError(
                    f"{type(self).__name__}: a dsp500list_v2 row has a null "
                    f"permno or mbrstartdt ({row!r}). A membership with no "
                    f"security or no start cannot be placed on a calendar, "
                    f"and tolerating it would put an all-False column into "
                    f"the panel that reads as 'never a member'."
                )
            if start > product_end:
                self.report["dropped_after_product_end"] += 1
                continue
            end = row["end"]
            if end is None or end > product_end:
                self.report["clipped_to_product_end"] += 1
                end = product_end
            if end < start:
                raise ValueError(
                    f"{type(self).__name__}: dsp500list_v2 permno {permno} has "
                    f"mbrenddt {end} before mbrstartdt {start}."
                )
            pieces.append((permno, start, end))
        return pieces

    # -- Nasdaq-100 (D-14) ---------------------------------------------------

    def _nasdaq100_pieces(
        self, *, allow_unlinked: bool
    ) -> list[tuple[int, date, date]]:
        """Compustat spells x CCM links -> PERMNO pieces. Task 2 of plan 07."""
        raise NotImplementedError(
            f"{type(self).__name__}: the {self.NASDAQ100} branch is implemented "
            f"in plan 03.10-07 Task 2."
        )

    # -- shared --------------------------------------------------------------

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
