"""Point-in-time index membership from the CRSP reference tables, keyed by PERMNO.

CRSP (the Center for Research in Security Prices) is a US stock database
sold through WRDS (Wharton Research Data Services). A PERMNO is CRSP's
permanent integer id for one security. Compustat is S&P's fundamentals
database, which identifies a security by ``(gvkey, iid)`` (company key and
issue id). *Point-in-time* membership records who was in an index on each
past day, which is what a backtest needs to avoid *survivorship bias*
(testing only on companies that are still around today). A *spell* is one
row of a membership table: one security's membership over one date range.

``CrspMembership`` answers "which securities were in this index on these
dates" from parquet files on disk, with no WRDS connection. It serves the
two index histories this vendor offers:

- ``crsp_sp500``: CRSP's own S&P 500 membership from
  ``crsp_a_indexes.dsp500list_v2``, one spell per
  ``(permno, mbrstartdt, mbrenddt)``, with coverage from 1925-12-31.
- ``comp_nasdaq100``: Compustat's Nasdaq-100 constituent history from
  ``comp.idxcst_his`` (``gvkeyx = '000208'``), whose ``(gvkey, iid)`` pairs
  are mapped to PERMNOs through the CRSP/Compustat Merged (CCM) link table
  ``crsp_a_ccm.ccmxpf_lnkhist``.

Memberships are returned as intervals keyed by PERMNO, because a PERMNO
stays the same through a ticker change and a ticker does not. Intervals
include both ends and every end is explicit: an open membership is cut at
``CrspReference.product_end`` (the last date of the downloaded CRSP data)
rather than left null, so a membership panel built from them cannot extend
past the CRSP price data.

This module imports only polars, loguru, the standard library and
``quantlab.dataset.crsp.reference``, so it works on a machine with no
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

#: How many unlinked spells an error message lists before summarizing the
#: rest, so the fix at the end of the message stays visible.
_MAX_LISTED = 20


def _as_date(value) -> date:
    """Convert an ISO date string or a ``date`` to a ``date``."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _merge_intervals(
    pieces: list[tuple[_Key, date, date]],
) -> list[tuple[_Key, date, date]]:
    """Merge overlapping or touching closed intervals, per key.

    Two intervals touch when ``next.start <= previous.end + 1 day``, so
    intervals ending 2005-12-31 and starting 2006-01-01 become one. Anything
    further apart stays two intervals: a security that left and rejoined
    was really absent in between, and joining the two would invent
    membership.

    Parameters
    ----------
    pieces : list of tuple
        ``(key, start, end)`` intervals, both ends inclusive, in any order.

    Returns
    -------
    list of tuple
        The merged ``(key, start, end)`` intervals, sorted.
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
    """Return the date ranges inside ``[start, end]`` that ``covered`` does not cover.

    All ranges include both ends. ``covered`` may overlap itself and need not
    be sorted.

    Parameters
    ----------
    start, end : date
        The range to check.
    covered : list of tuple of date
        ``(start, end)`` ranges that are covered.

    Returns
    -------
    list of tuple of date
        The uncovered ``(start, end)`` ranges, in order.
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
    """Point-in-time index membership from one CRSP reference directory.

    It holds no database connection; it only reads parquet files on disk.

    Parameters
    ----------
    reference : CrspReference
        The downloaded CRSP reference tables.

    Attributes
    ----------
    reference : CrspReference
        The reference tables given to the constructor.
    report : dict
        What the most recent ``permno_intervals`` call excluded, clipped,
        tolerated or could not link. Its keys are reset on every call.

    Examples
    --------
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
    #: Every index this class serves, in the order the command line lists them.
    INDEXES = (SP500, NASDAQ100)

    #: The earliest date each index has data for. The S&P date is where the
    #: CRSP index history starts. The Nasdaq-100 date is where Compustat's
    #: history is cut off, not where the index began: spells that start on
    #: 1995-01-01 may really have started earlier.
    PIT_COVERAGE_START = {SP500: "1925-12-31", NASDAQ100: "1995-01-01"}

    #: The S&P 500's index number in ``dsp500list_v2``. The table also holds
    #: other indexes; a row with any other ``indno`` is ignored.
    SP500_INDNO = 1000500

    #: The Compustat index id (``gvkeyx`` in ``comp.idx_index``) of the Nasdaq 100.
    NDX_GVKEYX = "000208"

    #: The CCM link types that state a real gvkey-to-PERMNO match. ``NR``
    #: ("no research") and ``NU`` rows have a null ``lpermno`` and are not
    #: links.
    LINK_TYPES = ("LC", "LU", "LS")

    #: A gap of at most this many calendar days between two links of one
    #: ``(gvkey, iid)`` is treated as a joint in the link table rather than a
    #: hole in the membership (a long weekend plus a holiday is four days
    #: without trading). Longer gaps count as unlinked.
    LINK_GAP_TOLERANCE_DAYS = 4

    def __init__(self, reference: CrspReference) -> None:
        """Initialize with an empty report; see the class docstring for parameters."""
        self.reference = reference
        #: What the last ``permno_intervals`` call excluded, clipped,
        #: tolerated or could not link.
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

        There is one row per continuous membership of one PERMNO, sorted,
        with both ends inclusive. ``end_date`` is never null and never later
        than ``CrspReference.product_end``. Adjacent spells are merged.

        ``allow_unlinked`` and ``window`` only affect the Nasdaq-100. By
        default, a Nasdaq-100 spell with membership days that no CCM link
        covers raises an error, because dropping those days would silently
        remove a real member. ``allow_unlinked=True`` records them in
        ``report["unlinked"]`` instead. ``window`` limits the error to
        uncovered days inside the window, where a member would actually be
        lost; it never filters the rows returned.

        Parameters
        ----------
        index : str
            ``"crsp_sp500"`` or ``"comp_nasdaq100"``.
        allow_unlinked : bool, default False
            Record unlinked Nasdaq-100 days instead of raising.
        window : tuple of date or str, optional
            ``(start, end)`` as dates or ISO strings. Only limits which
            unlinked days raise.

        Returns
        -------
        pl.DataFrame
            Columns ``permno`` (Int64), ``start_date`` and ``end_date``
            (Date).

        Raises
        ------
        ValueError
            If ``index`` is unknown, if ``window`` ends before it starts,
            if Nasdaq-100 membership days inside the window have no PERMNO
            and ``allow_unlinked`` is false, or if a source table holds
            invalid or conflicting rows.

        Examples
        --------
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
                f"be answered with an empty roster: an empty roster is a "
                f"valid answer for a real index, so a typo would look the "
                f"same as 'nobody was a member'."
            )
        if window is not None:
            window = (_as_date(window[0]), _as_date(window[1]))
            if window[0] > window[1]:
                raise ValueError(
                    f"{type(self).__name__}: the requested window is inverted: "
                    f"{window[0]} is after {window[1]}. An inverted window "
                    f"overlaps nothing, so it would silence every unlinked-day "
                    f"error and return a roster that looks complete."
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

        This is the list of securities (the *roster*) a download over the
        whole window fetches. The test is overlap, not full coverage
        (``start_date <= end and end_date >= start``), so a security that
        left the index during the window is still included; dropping it
        would cause survivorship bias.

        The order is numeric, as ``quantlab.utils.symbol_axis.sort_symbol_axis``
        defines it, not alphabetical: ``"7000"`` sorts before ``"14593"``.

        The window is passed on to ``permno_intervals``, so a Nasdaq-100
        roster only fails for unlinked days inside this window.

        Parameters
        ----------
        index : str
            ``"crsp_sp500"`` or ``"comp_nasdaq100"``.
        start_date : date or str
            Window start, as a ``date`` or ISO string.
        end_date : date or str
            Window end, inclusive.
        allow_unlinked : bool, default False
            Record unlinked Nasdaq-100 days instead of raising.

        Returns
        -------
        list of str
            PERMNOs as strings, in numeric order.

        Raises
        ------
        ValueError
            If ``start_date`` is after ``end_date``, or for the
            reasons ``permno_intervals`` raises.

        Examples
        --------
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
        """Read ``dsp500list_v2`` into ``(permno, start, end)`` spells with explicit ends.

        Rows for other indexes are skipped, spells starting after the product
        end are dropped, and later or missing ends are clipped to it; the
        ``report`` counts each case. Today's table already records an open
        membership with the product end rather than a null, so the null
        branch only guards against a future data version.

        Returns
        -------
        list of tuple
            ``(permno, start, end)`` spells.

        Raises
        ------
        ValueError
            If a row has a null PERMNO or start date, or ends before it
            starts.
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
        """Join Compustat membership spells to CCM links, giving ``(permno, start, end)`` pieces.

        The join is on ``gvkey`` and ``iid = liid``, never on ``linkprim``
        (the "primary link" flag). Both Alphabet share classes are Nasdaq-100
        members under one gvkey, and the common ``linkprim IN ('P', 'C')``
        filter would silently drop one of them.

        Every membership day must end up with a PERMNO. Each spell, clipped
        to the product end, is intersected with its matching links, and the
        days no link covers are then computed. Gaps of at most
        ``LINK_GAP_TOLERANCE_DAYS`` between two links are recorded as
        tolerated. Longer gaps, and spells with no link at all, are
        unlinked. Unlinked days inside ``window`` (or anywhere, when
        ``window`` is ``None``) raise unless ``allow_unlinked`` is set;
        unlinked days outside the window are logged and recorded but never
        raise. Every unlinked spell goes into ``report["unlinked"]``, and the
        ones that would raise also go into ``report["unlinked_blocking"]``.

        Parameters
        ----------
        allow_unlinked : bool
            Record unlinked days instead of raising.
        window : tuple of date, optional
            Only unlinked days inside this window raise.

        Returns
        -------
        list of tuple
            ``(permno, start, end)`` pieces, not yet merged.

        Raises
        ------
        ValueError
            If a row is invalid, if one ``(gvkey, iid)`` links to two PERMNOs
            at once, or if blocking unlinked days exist and
            ``allow_unlinked`` is false.
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
                # Entirely after the CRSP price data, so it is not an
                # unlinked spell and must not trigger the error below.
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
                # Test overlap on the dates here, before the report entry
                # below turns them into strings.
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
                # The same dict goes into both lists, so the two report keys
                # always agree.
                if blocks:
                    blocking.append(entry)

        if blocking and not allow_unlinked:
            raise ValueError(self._unlinked_message(blocking, window=window))
        if unlinked:
            self.report["unlinked"] = unlinked
            self.report["unlinked_blocking"] = blocking
            if blocking:
                # Worded so no verb has to agree with the count.
                in_window = (
                    f" Uncovered days inside the requested window "
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
                # this also runs without allow_unlinked: gaps outside the
                # window skip the error but are still logged.
                logger.warning(
                    f"{type(self).__name__}: {len(unlinked)} Nasdaq-100 "
                    f"membership spell(s) have days no CRSP/Compustat link "
                    f"covers, but none of those days fall inside the requested "
                    f"window {window[0]}..{window[1]}, so the universe over "
                    f"that window is complete. They are recorded in "
                    f"report['unlinked'] for inspection."
                )
        return pieces

    def _ccm_links_by_key(self) -> dict[tuple[str, str], list[dict]]:
        """Group the real, dated CCM links by ``(gvkey, liid)``.

        ``lpermno`` arrives as a float because the server column is a double.
        A non-integer value would turn into a different, real security under
        a plain cast, so the table is refused instead. A link with a null
        ``linkdt`` (link start date) is skipped: it has no start to
        intersect with, so the days it would cover count as unlinked.

        Returns
        -------
        dict
            ``{(gvkey, liid): [link, ...]}``, each link a dict with
            ``permno``, ``linkdt`` and ``linkenddt``.

        Raises
        ------
        ValueError
            If any link has a non-integer ``lpermno``.
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
                f"an integer; rounding one would name a different, real "
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
        """Raise if one ``(gvkey, iid)`` links to two PERMNOs on the same day.

        One Compustat issue is one security; without this check, both
        PERMNOs would be reported as members.

        Parameters
        ----------
        gvkey, iid : str
            The Compustat security, for the error message.
        matched : list of tuple
            Its ``(permno, start, end)`` link pieces.

        Raises
        ------
        ValueError
            If two different PERMNOs overlap in time.
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
                        f"to two PERMNOs over the same dates: {permno} and "
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
        """Build the unlinked-days error message: what fails, how much, and the fix.

        ``unlinked`` holds only the spells that cause the error. With a
        ``window`` this is a subset of all unlinked spells, and listing the
        others would misstate why the call failed.

        Parameters
        ----------
        unlinked : list of dict
            The blocking unlinked spells from ``report``.
        window : tuple of date, optional
            The requested window, mentioned in the message if given.

        Returns
        -------
        str
            The error message.
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
            f"\nThis error only concerns the requested window "
            f"{window[0]}..{window[1]}: only spells with uncovered days inside "
            f"it are listed above, and only they cause it. Spells whose gaps "
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
            f"universe: survivorship bias that later looks like a data gap "
            f"rather than an error. Pass allow_unlinked=True to proceed with "
            f"the linked days and read the rest from report['unlinked']."
        )

    # -- shared --------------------------------------------------------------

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
