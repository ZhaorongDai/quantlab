"""Point-in-time index membership from the CRSP reference tier, keyed by PERMNO.

``CrspMembership`` answers "which securities were in this index on these
dates" from parquet on disk, with no WRDS connection. It serves the two
index histories this vendor offers:

- ``crsp_sp500``: CRSP's own S&P 500 membership from
  ``crsp_a_indexes.dsp500list_v2``, one spell per
  ``(permno, mbrstartdt, mbrenddt)``, with coverage from 1925-12-31.
- ``comp_nasdaq100``: Compustat's Nasdaq-100 constituent history from
  ``comp.idxcst_his`` (``gvkeyx = '000208'``), whose ``gvkey``/``iid`` pairs
  are mapped to PERMNOs through the CRSP/Compustat link table
  ``crsp_a_ccm.ccmxpf_lnkhist``.

Memberships are returned as intervals keyed by PERMNO, because the PERMNO
survives a ticker change and a ticker does not. Intervals are closed on both
ends and every end is explicit: an open membership is clipped to
``CrspReference.product_end`` rather than left null, so a constituent panel
built from them cannot extend past the CRSP price coverage.

This module imports polars, loguru, the standard library and
``quantlab.dataset.crsp.reference`` only, so it works on a machine with no
database driver.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from loguru import logger

from quantlab.dataset.crsp.reference import CrspReference
from quantlab.utils.symbol_axis import sort_symbol_axis

_ONE_DAY = timedelta(days=1)

#: The key an interval list is grouped by: a PERMNO.
_Key = int

#: How many unlinked spells a refusal lists before summarising the rest, so
#: the remedy sentence at the end is not buried.
_MAX_LISTED = 20


def _as_date(value) -> date:
    """Coerce an ISO date string or a ``date`` to a ``date``."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _merge_intervals(
    pieces: list[tuple[_Key, date, date]],
) -> list[tuple[_Key, date, date]]:
    """Merge overlapping or touching closed intervals, per key.

    Touching means ``next.start <= previous.end + 1 day``, so 2005-12-31 and
    2006-01-01 form one continuous interval. Anything wider stays two
    intervals: a security that left and rejoined was genuinely absent in
    between, and bridging the hole would fabricate membership.
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
    """Return the day ranges of ``[start, end]`` that ``covered`` leaves out.

    Plain interval subtraction on closed ranges. ``covered`` may overlap
    itself and need not be sorted.
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

    Holds no connection: ``reference`` is a ``CrspReference`` over parquet on
    disk. ``report`` describes what the most recent ``permno_intervals()``
    call excluded, clipped, tolerated or left unlinked; its keys are reset on
    every call.

    Example:
        >>> from quantlab.dataset.crsp.membership import CrspMembership
        >>> from quantlab.dataset.crsp.reference import CrspReference
        >>> ref = CrspReference("data/downloads/us_equity/1d/wrds_crsp/_reference")
        >>> membership = CrspMembership(ref)
        >>> membership.permnos_in_range("comp_nasdaq100", "2010-01-01", "2020-12-31")
        ['14542', '90319']
        >>> membership.report["clipped_to_product_end"]
        2
    """

    #: CRSP's own S&P 500 membership.
    SP500 = "crsp_sp500"
    #: Compustat's Nasdaq-100 membership, linked to PERMNOs through CCM.
    NASDAQ100 = "comp_nasdaq100"
    #: Every universe this vendor serves, in the order a CLI lists them.
    INDEXES = (SP500, NASDAQ100)

    #: The earliest date each universe can be answered for. The S&P start is
    #: the index family's own start; the Nasdaq-100 date is a censor, not a
    #: start: Compustat's history begins 1995-01-01, so spells that begin on
    #: that day may have begun earlier in reality.
    PIT_COVERAGE_START = {SP500: "1925-12-31", NASDAQ100: "1995-01-01"}

    #: The S&P 500's index number in ``dsp500list_v2``; the table carries
    #: other indexes, and a row with any other ``indno`` is not this universe.
    SP500_INDNO = 1000500

    #: ``comp.idx_index``'s gvkeyx for "Nasdaq 100".
    NDX_GVKEYX = "000208"

    #: The CCM link types that assert a real gvkey-to-PERMNO identity. ``NR``
    #: ("no research") and ``NU`` rows carry a null ``lpermno`` and are not
    #: links.
    LINK_TYPES = ("LC", "LU", "LS")

    #: A membership stretch of at most this many calendar days between two
    #: links of one ``(gvkey, iid)`` is treated as a seam in the link table
    #: rather than a hole in the universe (a long weekend plus a holiday is
    #: four days with no trading). Wider gaps are genuinely unlinked.
    LINK_GAP_TOLERANCE_DAYS = 4

    def __init__(self, reference: CrspReference) -> None:
        """Bind a reference directory and start with an empty report."""
        self.reference = reference
        #: What the last ``permno_intervals()`` call excluded, clipped,
        #: tolerated or left unlinked.
        self.report: dict = {}
        self._reset_report()

    def _reset_report(self) -> None:
        """Reset every report key to its empty value."""
        self.report.update(
            {
                "other_indno_rows": 0,
                "dropped_after_product_end": 0,
                "clipped_to_product_end": 0,
                "left_censored_spells": 0,
                "tolerated_gaps": [],
                "unlinked": [],
                "unlinked_blocking": [],
            }
        )

    # -- public -------------------------------------------------------------

    def permno_intervals(
        self,
        index: str,
        *,
        allow_unlinked: bool = False,
        window: tuple[date, date] | tuple[str, str] | None = None,
    ) -> pl.DataFrame:
        """Return membership intervals as ``(permno, start_date, end_date)``.

        One row per continuous membership of one PERMNO, sorted, both ends
        inclusive, ``end_date`` never null and never later than
        ``CrspReference.product_end``. Adjacent spells are merged.

        ``allow_unlinked`` and ``window`` affect only the Nasdaq-100 branch.
        By default a Nasdaq-100 spell with membership days that no CCM link
        covers raises, because dropping those days would silently remove a
        real member; ``allow_unlinked=True`` records them in
        ``report["unlinked"]`` instead. ``window`` narrows that refusal to
        uncovered days the window could actually lose a member to; it never
        filters the rows returned.

        Args:
            index: ``"crsp_sp500"`` or ``"comp_nasdaq100"``.
            allow_unlinked: Record unlinked Nasdaq-100 days instead of raising.
            window: ``(start, end)`` as dates or ISO strings; scopes the
                unlinked refusal only.

        Raises:
            ValueError: If ``index`` is unknown, if ``window`` is inverted, or
                if Nasdaq-100 membership days inside the window have no
                PERMNO and ``allow_unlinked`` is false.

        Example:
            >>> membership.permno_intervals("crsp_sp500")
            shape: (1, 3)
            ┌────────┬────────────┬────────────┐
            │ permno ┆ start_date ┆ end_date   │
            │ ---    ┆ ---        ┆ ---        │
            │ i64    ┆ date       ┆ date       │
            ╞════════╪════════════╪════════════╡
            │ 14593  ┆ 1982-11-18 ┆ 2025-12-31 │
            └────────┴────────────┴────────────┘
        """
        if index not in self.INDEXES:
            raise ValueError(
                f"{type(self).__name__}: {index!r} is not a CRSP universe; "
                f"this vendor serves {self.INDEXES}. An unknown name cannot "
                f"be answered with an empty roster -- an empty roster is a "
                f"legitimate answer for a REAL index, so a typo would be "
                f"indistinguishable from 'nobody was a member'."
            )
        if window is not None:
            window = (_as_date(window[0]), _as_date(window[1]))
            if window[0] > window[1]:
                raise ValueError(
                    f"{type(self).__name__}: the requested window is inverted "
                    f"-- {window[0]} is after {window[1]}. An inverted window "
                    f"overlaps nothing, so it would suppress EVERY unlinked "
                    f"refusal and hand back a roster indistinguishable from a "
                    f"complete one."
                )
        self._reset_report()
        if index == self.SP500:
            pieces = self._sp500_pieces()
        else:
            pieces = self._nasdaq100_pieces(
                allow_unlinked=allow_unlinked, window=window
            )
        return self._frame(_merge_intervals(pieces))

    def permnos_in_range(
        self,
        index: str,
        start_date,
        end_date,
        *,
        allow_unlinked: bool = False,
    ) -> list[str]:
        """Return every PERMNO whose membership overlaps ``[start_date, end_date]``.

        This is the roster a full-window backfill pulls. The test is overlap,
        not containment (``start_date <= end and end_date >= start``), so a
        security that left the index inside the window is still listed;
        dropping it would be survivorship bias.

        The order is NUMERIC, as ``quantlab.utils.symbol_axis.sort_symbol_axis``
        defines it, not lexicographic: ``"7000"`` sorts before ``"14593"``.

        The window is passed down to ``permno_intervals``, so a Nasdaq-100
        roster is refused only for unlinked days inside this window.

        Args:
            index: ``"crsp_sp500"`` or ``"comp_nasdaq100"``.
            start_date: Window start, as a ``date`` or ISO string.
            end_date: Window end, inclusive.
            allow_unlinked: Record unlinked Nasdaq-100 days instead of raising.

        Returns:
            PERMNOs as strings, in numeric order.

        Raises:
            ValueError: If ``start_date`` is after ``end_date``, or for the
                reasons ``permno_intervals`` raises.

        Example:
            >>> membership.permnos_in_range("crsp_sp500", "2020-01-01", "2020-12-31")
            ['14593']
            >>> membership.permnos_in_range("crsp_sp500", "1970-01-01", "1980-12-31")
            []
        """
        start = _as_date(start_date)
        end = _as_date(end_date)
        if start > end:
            raise ValueError(
                f"{type(self).__name__}: start_date {start} is after end_date "
                f"{end}; an inverted window overlaps nothing and would return "
                f"an empty roster indistinguishable from a real one."
            )
        intervals = self.permno_intervals(
            index, allow_unlinked=allow_unlinked, window=(start, end)
        )
        overlapping = intervals.filter(
            (pl.col("start_date") <= end) & (pl.col("end_date") >= start)
        )
        return [
            str(permno)
            for permno in sort_symbol_axis(set(overlapping["permno"].to_list()))
        ]

    # -- S&P 500 -------------------------------------------------------------

    def _sp500_pieces(self) -> list[tuple[int, date, date]]:
        """Read ``dsp500list_v2`` into ``(permno, start, end)`` with explicit ends.

        Open membership in this table is already recorded as the product end
        rather than a null, so the null branch below guards against a future
        vintage rather than describing today's data.
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

    # -- Nasdaq-100 ----------------------------------------------------------

    def _nasdaq100_pieces(
        self,
        *,
        allow_unlinked: bool,
        window: tuple[date, date] | None = None,
    ) -> list[tuple[int, date, date]]:
        """Join Compustat spells to CCM links into ``(permno, start, end)`` pieces.

        The join is on ``gvkey`` and ``iid = liid``, never on ``linkprim``.
        Both Alphabet classes are Nasdaq-100 members under one gvkey, and the
        usual ``linkprim IN ('P', 'C')`` filter would silently drop one of
        them.

        Every membership day must end up with a PERMNO. After intersecting
        each clipped spell with its matching links, the days no link covers
        are computed explicitly. Gaps of at most ``LINK_GAP_TOLERANCE_DAYS``
        between two links are recorded as tolerated; anything longer, and any
        spell with no link at all, is unlinked. Unlinked days inside ``window``
        (or anywhere, when ``window`` is ``None``) raise unless
        ``allow_unlinked`` is set; unlinked days outside the window are logged
        and recorded but never raise. Every unlinked spell lands in
        ``report["unlinked"]``, and the subset that would refuse in
        ``report["unlinked_blocking"]``.
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
        blocking: list[dict] = []
        for spell in spells.to_dicts():
            gvkey, iid, start = spell["gvkey"], spell["iid"], spell["start"]
            if gvkey is None or iid is None or start is None:
                raise ValueError(
                    f"{type(self).__name__}: an idxcst_his row has a null "
                    f"gvkey, iid or from ({spell!r}); it cannot be linked to "
                    f"a security or placed on a calendar."
                )
            if start > product_end:
                # Entirely outside CRSP price coverage, so not an unlinkable
                # spell; it must not reach the refusal below.
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
                # The overlap test runs on the date tuples here, before the
                # report entry below turns them into strings.
                blocks = window is None or any(
                    gap_start <= window[1] and gap_end >= window[0]
                    for gap_start, gap_end in uncovered
                )
                entry = {
                    "gvkey": gvkey,
                    "iid": iid,
                    "from": str(start),
                    "thru": str(end),
                    "uncovered": [
                        [str(gap_start), str(gap_end)]
                        for gap_start, gap_end in uncovered
                    ],
                }
                unlinked.append(entry)
                # The same dict object goes into both lists, so the two
                # report keys cannot drift apart.
                if blocks:
                    blocking.append(entry)

        if blocking and not allow_unlinked:
            raise ValueError(self._unlinked_message(blocking, window=window))
        if unlinked:
            self.report["unlinked"] = unlinked
            self.report["unlinked_blocking"] = blocking
            if blocking:
                # Phrased without a verb that has to agree with the count.
                in_window = (
                    f" Uncovered days INSIDE the requested window "
                    f"{window[0]}..{window[1]}: {len(blocking)} of the "
                    f"{len(unlinked)}."
                    if window is not None
                    else ""
                )
                logger.warning(
                    f"{type(self).__name__}: {len(unlinked)} Nasdaq-100 "
                    f"membership spell(s) have days no CRSP/Compustat link "
                    f"covers; those days are absent from the universe. "
                    f"allow_unlinked=True was passed, so they are listed in "
                    f"report['unlinked'] instead of raising." + in_window
                )
            else:
                # `blocking` can only be empty when a window was given, so
                # this branch also fires on runs that did not pass
                # allow_unlinked: out-of-window gaps skip the refusal, never
                # the log.
                logger.warning(
                    f"{type(self).__name__}: {len(unlinked)} Nasdaq-100 "
                    f"membership spell(s) have days no CRSP/Compustat link "
                    f"covers, but NONE of those days fall inside the requested "
                    f"window {window[0]}..{window[1]}, so the universe over "
                    f"that window is complete. They are recorded in "
                    f"report['unlinked'] for inspection."
                )
        return pieces

    def _ccm_links_by_key(self) -> dict[tuple[str, str], list[dict]]:
        """Group real, dated CCM identity links by ``(gvkey, liid)``.

        ``lpermno`` arrives as a float because the server column is double
        precision. A non-integral value would become a different, real
        security under a plain cast, so the table is refused instead. A link
        with a null ``linkdt`` is skipped: it has no start to intersect with,
        and the days it would have covered reach the unlinked refusal.
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
        """Refuse when one ``(gvkey, iid)`` links to two PERMNOs on the same day.

        One Compustat issue is one security; left unchecked, both PERMNOs
        would be emitted as members.
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

    def _unlinked_message(
        self,
        unlinked: list[dict],
        *,
        window: tuple[date, date] | None = None,
    ) -> str:
        """Build the unlinked refusal: what blocks, how much, and the remedy.

        ``unlinked`` is the blocking subset only; with a ``window`` the two
        differ, and listing spells that did not refuse would misstate the
        reason for refusing.
        """
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
        scope = (
            f"\nThe refusal is scoped to the requested window "
            f"{window[0]}..{window[1]}: only spells with uncovered days inside "
            f"it are listed above, and only they refuse. Spells whose gaps "
            f"fall entirely outside it are recorded in report['unlinked'] "
            f"instead."
            if window is not None
            else ""
        )
        return (
            f"{type(self).__name__}: {len(unlinked)} Nasdaq-100 membership "
            f"spell(s) have days that no CRSP/Compustat link covers, so those "
            f"membership days have no PERMNO:\n"
            + "\n".join(listed)
            + more
            + scope
            + f"\nDropping them would remove real index members from the "
            f"universe -- survivorship bias that reads downstream as a data "
            f"gap rather than an error. Pass allow_unlinked=True (the CLI's "
            f"--allow-unlinked-ndx) to proceed with the linked days and read "
            f"the rest from report['unlinked']."
        )

    # -- shared --------------------------------------------------------------

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
