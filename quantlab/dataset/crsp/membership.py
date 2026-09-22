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

**Keyed by PERMNO, and that is now the ONLY key.** A universe answered in
tickers has to be re-answered every time a security is renamed, and the rename
is exactly where a membership panel and a price panel stop agreeing. The
PERMNO does not change on a rename, so the roster the CLI pulls and the
intervals a constituent panel densifies are both in an identifier that is
stable across the event.

This module used to carry a second, ticker-keyed answer -- `symbol_intervals()`
-- which mapped these PERMNO intervals onto the ticker intervals
`quantlab/dataset/crsp/symbology.py` derives, because the price panel's columns
were tickers at the time. Since
03.11-03 the price panel's `symbol` IS the int64 PERMNO, so that branch became
a way to build a universe that intersects a price panel to NOTHING, and an
empty universe reads downstream as "no positions" rather than as an error. It
was deleted in 03.11-05 rather than deprecated:
`quantlab/dataset/constituent.py` renames the `permno` column to `symbol` and
hands the frame to `_densify` unchanged.

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
psycopg2, no acquisition import, and since 03.11-05 no `crsp_symbology` either
(it was imported for the deleted ticker branch alone). That sibling is itself a
leaf, so the universes still resolve on a machine with no WRDS credential and
no database driver.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from loguru import logger

from quantlab.dataset.crsp.reference import CrspReference
from quantlab.utils.symbol_axis import sort_symbol_axis

_ONE_DAY = timedelta(days=1)

#: What an interval list is keyed by: a PERMNO, and nothing else. It was
#: `int | str` while `symbol_intervals()` merged the SAME way by ticker, so the
#: boundary arithmetic below could be written once for both; on the PERMNO axis
#: there is only one kind of key, and a union type would advertise a choice
#: that no longer exists.
_Key = int

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
    pieces: list[tuple[_Key, date, date]],
) -> list[tuple[_Key, date, date]]:
    """Merge overlapping or TOUCHING intervals, per key (a PERMNO or a symbol).

    Touching means `next.start <= previous.end + 1 day`: the intervals are
    closed, so 2005-12-31 and 2006-01-01 are one continuous membership with no
    day between them. Anything wider stays two intervals -- a security that
    left the index and rejoined was genuinely not a member in between, and
    bridging the hole would fabricate membership.
    """
    merged: list[tuple[_Key, date, date]] = []
    for key, start, end in sorted(pieces):
        if merged and merged[-1][0] == key and start <= merged[-1][2] + _ONE_DAY:
            previous = merged[-1]
            merged[-1] = (key, previous[1], max(previous[2], end))
        else:
            merged.append((key, start, end))
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
    below are reset on entry.
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

        **The order is part of the contract, and it is NUMERIC** -- the
        argument for why MOVED to
        `quantlab/utils/symbol_axis.py:sort_symbol_axis` in 03.11-02, which is
        now the single source of that contract and which this method calls.
        It used to be stated here, and only here, while eight other call sites
        each spelled their own bare `sorted()`; a contract stated in one place
        and re-derived in eight is eight things that can drift apart.
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
        return [
            str(permno)
            for permno in sort_symbol_axis(set(overlapping["permno"].to_list()))
        ]

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
        """Compustat spells x CCM links -> `(permno, start, end)` pieces.

        **The join is `gvkey` AND `iid = liid`, never `linkprim`.** Both
        Alphabet classes are Nasdaq-100 members under ONE gvkey (160329, iid
        01 -> GOOGL 90319 and iid 03 -> GOOG 14542, live check `L8_2`/`L7_4`).
        The conventional `linkprim IN ('P','C')` filter keeps the primary
        issue only, which would silently drop one of the two -- a member of
        the index simply missing from the universe, and invisible downstream
        because a missing security looks like a data gap.

        **Every membership day must end up with a PERMNO.** After
        intersecting the clipped spell with each matched link, the days no
        piece covers are computed explicitly. Short stretches
        (`LINK_GAP_TOLERANCE_DAYS`) between two links are recorded as
        tolerated gaps; anything longer is UNLINKED and raises by default.
        Refusing rather than dropping is the whole point: a dropped spell is
        survivorship bias written into the universe (T-03.10-23), and
        `allow_unlinked=True` moves the fact into `report["unlinked"]` rather
        than making it disappear.

        A spell with NO matching link at all is always unlinked, however short
        it is: the tolerance bridges a seam BETWEEN two links, and with no
        link there is nothing to bridge.
        """
        product_end = self.reference.product_end
        spells = (
            self.reference.table("idxcst_his")
            .filter(pl.col("gvkeyx") == self.NDX_GVKEYX)
            .select(
                pl.col("gvkey").cast(pl.String),
                pl.col("iid").cast(pl.String),
                pl.col("from").cast(pl.Date).alias("start"),
                pl.col("thru").cast(pl.Date).alias("thru"),
            )
            .sort(["gvkey", "iid", "start"])
        )
        censor = _as_date(self.PIT_COVERAGE_START[self.NASDAQ100])
        self.report["left_censored_spells"] = int(
            spells.filter(pl.col("start") == censor).height
        )

        links_by_key = self._ccm_links_by_key()

        pieces: list[tuple[int, date, date]] = []
        unlinked: list[dict] = []
        for spell in spells.to_dicts():
            gvkey, iid, start = spell["gvkey"], spell["iid"], spell["start"]
            if gvkey is None or iid is None or start is None:
                raise ValueError(
                    f"{type(self).__name__}: an idxcst_his row has a null "
                    f"gvkey, iid or from ({spell!r}); it cannot be linked to "
                    f"a security or placed on a calendar."
                )
            if start > product_end:
                # Out of CRSP price coverage entirely -- not an unlinkable
                # spell, so it must not reach the refusal below.
                self.report["dropped_after_product_end"] += 1
                continue
            end = spell["thru"]
            if end is None or end > product_end:
                self.report["clipped_to_product_end"] += 1
                end = product_end
            if end < start:
                raise ValueError(
                    f"{type(self).__name__}: idxcst_his gvkey {gvkey} iid "
                    f"{iid} has thru {end} before from {start}."
                )

            matched: list[tuple[int, date, date]] = []
            for link in links_by_key.get((gvkey, iid), []):
                link_end = link["linkenddt"] or product_end
                link_end = min(link_end, product_end)
                piece_start = max(start, link["linkdt"])
                piece_end = min(end, link_end)
                if piece_start <= piece_end:
                    matched.append((link["permno"], piece_start, piece_end))

            self._assert_unambiguous(gvkey, iid, matched)
            pieces.extend(matched)

            gaps = _uncovered(
                start, end, [(piece[1], piece[2]) for piece in matched]
            )
            uncovered: list[tuple[date, date]] = []
            for gap_start, gap_end in gaps:
                days = (gap_end - gap_start).days + 1
                if matched and days <= self.LINK_GAP_TOLERANCE_DAYS:
                    self.report["tolerated_gaps"].append(
                        {
                            "gvkey": gvkey,
                            "iid": iid,
                            "start": str(gap_start),
                            "end": str(gap_end),
                            "days": days,
                        }
                    )
                else:
                    uncovered.append((gap_start, gap_end))
            if uncovered:
                unlinked.append(
                    {
                        "gvkey": gvkey,
                        "iid": iid,
                        "from": str(start),
                        "thru": str(end),
                        "uncovered": [
                            [str(gap_start), str(gap_end)]
                            for gap_start, gap_end in uncovered
                        ],
                    }
                )

        if unlinked and not allow_unlinked:
            raise ValueError(self._unlinked_message(unlinked))
        if unlinked:
            self.report["unlinked"] = unlinked
            logger.warning(
                f"{type(self).__name__}: {len(unlinked)} Nasdaq-100 membership "
                f"spell(s) have days no CRSP/Compustat link covers; those days "
                f"are absent from the universe. allow_unlinked=True was passed, "
                f"so they are listed in report['unlinked'] instead of raising."
            )
        return pieces

    def _ccm_links_by_key(self) -> dict[tuple[str, str], list[dict]]:
        """`(gvkey, liid) -> links`, keeping only real, dated identity links.

        `lpermno` is `double precision` on the server (live check `L7_2`), so
        it arrives as a float and a PERMNO is an integer. A non-integral value
        would silently become a DIFFERENT, real security under a plain cast,
        so it raises instead.

        A link with a null `linkdt` is not used: it has no start, so no
        interval can be intersected with it. It is not "dropped silently" --
        the days it would have covered simply stay uncovered and reach the
        unlinked refusal, which is the loud path.
        """
        links = self.reference.table("ccmxpf_lnkhist").filter(
            pl.col("linktype").is_in(self.LINK_TYPES)
            & pl.col("lpermno").is_not_null()
            & pl.col("linkdt").is_not_null()
        )
        non_integral = links.filter(
            pl.col("lpermno").cast(pl.Float64)
            != pl.col("lpermno").cast(pl.Float64).floor()
        )
        if non_integral.height:
            offenders = non_integral["lpermno"].to_list()[:_MAX_LISTED]
            raise ValueError(
                f"{type(self).__name__}: {non_integral.height} ccmxpf_lnkhist "
                f"row(s) carry a non-integral lpermno "
                f"({', '.join(str(value) for value in offenders)}). A PERMNO is "
                f"an integer; rounding one would name a DIFFERENT, real "
                f"security, so the link table is refused instead."
            )

        by_key: dict[tuple[str, str], list[dict]] = {}
        rows = links.select(
            pl.col("gvkey").cast(pl.String),
            pl.col("liid").cast(pl.String),
            pl.col("lpermno").cast(pl.Float64).cast(pl.Int64).alias("permno"),
            pl.col("linkdt").cast(pl.Date),
            pl.col("linkenddt").cast(pl.Date),
        ).to_dicts()
        for row in rows:
            by_key.setdefault((row["gvkey"], row["liid"]), []).append(row)
        return by_key

    def _assert_unambiguous(
        self, gvkey: str, iid: str, matched: list[tuple[int, date, date]]
    ) -> None:
        """One `(gvkey, iid)` is ONE security; two PERMNOs on one day is not.

        Left unchecked, both PERMNOs would be emitted as members and the
        universe would hold a security the index never contained
        (T-03.10-25).
        """
        for index, (permno, start, end) in enumerate(matched):
            for other_permno, other_start, other_end in matched[index + 1 :]:
                if other_permno == permno:
                    continue
                overlap_start = max(start, other_start)
                overlap_end = min(end, other_end)
                if overlap_start <= overlap_end:
                    raise ValueError(
                        f"{type(self).__name__}: gvkey {gvkey} iid {iid} links "
                        f"to TWO PERMNOs over the same dates -- {permno} and "
                        f"{other_permno} both cover {overlap_start}.."
                        f"{overlap_end}. One Compustat issue is one security, "
                        f"so this is a link-table conflict, not a choice this "
                        f"module may make on your behalf."
                    )

    def _unlinked_message(self, unlinked: list[dict]) -> str:
        """The refusal: what is unlinked, how much of it, and the remedy."""
        listed = []
        for record in unlinked[:_MAX_LISTED]:
            ranges = ", ".join(
                f"{gap_start}..{gap_end}" for gap_start, gap_end in record["uncovered"]
            )
            listed.append(
                f"  gvkey={record['gvkey']} iid={record['iid']} "
                f"from={record['from']} thru={record['thru']} uncovered=[{ranges}]"
            )
        more = (
            f"\n  ... and {len(unlinked) - _MAX_LISTED} more"
            if len(unlinked) > _MAX_LISTED
            else ""
        )
        return (
            f"{type(self).__name__}: {len(unlinked)} Nasdaq-100 membership "
            f"spell(s) have days that no CRSP/Compustat link covers, so those "
            f"membership days have no PERMNO:\n"
            + "\n".join(listed)
            + more
            + f"\nDropping them would remove real index members from the "
            f"universe -- survivorship bias that reads downstream as a data "
            f"gap rather than an error. Pass allow_unlinked=True (the CLI's "
            f"--allow-unlinked-ndx) to proceed with the linked days and read "
            f"the rest from report['unlinked']."
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
