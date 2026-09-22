"""The two CRSP-vendor point-in-time universes, as PERMNO intervals.

`CrspMembership` answers "which securities were in this index on these dates"
from the reference tier alone -- no WRDS connection, no ticker resolution. The
two universes it serves are the two this vendor actually offers:

- `crsp_sp500` -- CRSP's OWN S&P 500 membership (`dsp500list_v2`), keyed by
  PERMNO (D-05);
- `comp_nasdaq100` -- Compustat's Nasdaq-100 constituent history
  (`idxcst_his`, `gvkeyx` 000208) linked to PERMNOs through the CRSP/Compustat
  Merged link table (D-14).

Every assertion below is either VERBATIM from a named live-check key or
carries a `# SYNTHETIC` comment on the spot, which is the phase's fixture
provenance rule.

**Every quantlab import lives inside a test (or inside `_membership`).** That
is not style: this module is written before
`quantlab/dataset/crsp/membership.py` exists, and a module-scope import would
turn the RED run into a COLLECTION error -- zero tests discovered, which
proves nothing about the behaviour (TDD gate #3770).
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Row builders -- the `{column: text-or-NULL}` shape a COPY would carry, the
# same one `tests/crsp_fixtures.py` builds its live rows with. Restated here
# (three lines) rather than imported, because `crsp_fixtures._row` is private
# and this plan must not edit that module.
# ---------------------------------------------------------------------------

_DSP500_COLUMNS = ("permno", "indno", "mbrstartdt", "mbrenddt", "mbrflg", "indfam")
_IDXCST_COLUMNS = ("gvkey", "iid", "gvkeyx", "from", "thru")
_CCM_COLUMNS = (
    "gvkey", "linkprim", "liid", "linktype", "lpermno", "lpermco",
    "linkdt", "linkenddt",
)

#: The live S&P 500 index number (`03.10-LIVE-CHECK.json` key
#: `C4_sample_crsp_a_indexes.dsp500list_v2`).
_SP500_INDNO = "1000500"

#: The live Nasdaq-100 index gvkeyx (`03.10-LIVE-CHECK-2.json` key `L8_2`).
_NDX_GVKEYX = "000208"

#: The CRSP product end every fixture in this module is written against
#: (`03.10-LIVE-CHECK-2.json` key `L9_1`: open S&P membership reads
#: `mbrenddt = 2025-12-31`, 503 rows, no NULLs).
_PRODUCT_END = "2025-12-31"


def _text_row(columns, values) -> dict[str, str | None]:
    """A live-check row: every value text, the token `"None"` meaning SQL NULL."""
    return {
        name: (None if value in (None, "None") else str(value))
        for name, value in zip(columns, values)
    }


def _sp500(permno, start, end, indno=_SP500_INDNO) -> dict[str, str | None]:
    """One `dsp500list_v2` membership spell."""
    return _text_row(
        _DSP500_COLUMNS, [permno, indno, start, end, "NORM", "1100500"]
    )


def _spell(gvkey, iid, start, thru, gvkeyx=_NDX_GVKEYX) -> dict[str, str | None]:
    """One `comp.idxcst_his` index-membership spell."""
    return _text_row(_IDXCST_COLUMNS, [gvkey, iid, gvkeyx, start, thru])


def _link(gvkey, liid, permno, start, end, linktype="LC") -> dict[str, str | None]:
    """One `crsp_a_ccm.ccmxpf_lnkhist` row.

    `lpermno` is rendered as a FLOAT because the live column type is `double
    precision` (`03.10-LIVE-CHECK-2.json` key `L7_2`).
    """
    return _text_row(
        _CCM_COLUMNS,
        [gvkey, "P", liid, linktype, permno, "45000.0", start, end],
    )


def _membership(tmp_path, rows_by_table, product_end=_PRODUCT_END):
    """A `CrspMembership` over a freshly written, offline reference tier."""
    from quantlab.dataset.crsp.membership import CrspMembership
    from quantlab.dataset.crsp.reference import CrspReference
    from tests.crsp_fixtures import write_reference_tables

    directory = write_reference_tables(
        tmp_path / "_reference", rows_by_table, product_end=product_end
    )
    return CrspMembership(CrspReference(directory))


def _sp500_tier(tmp_path, rows, product_end=_PRODUCT_END):
    return _membership(
        tmp_path, {"crsp_a_indexes.dsp500list_v2": rows}, product_end=product_end
    )


def _ndx_tier(tmp_path, spells, links, product_end=_PRODUCT_END):
    return _membership(
        tmp_path,
        {
            "comp.idxcst_his": spells,
            "crsp_a_ccm.ccmxpf_lnkhist": links,
        },
        product_end=product_end,
    )


# ---------------------------------------------------------------------------
# D-05 -- CRSP's own S&P 500 membership
# ---------------------------------------------------------------------------


def test_sp500_verbatim_aapl_row_becomes_a_closed_permno_interval(tmp_path):
    """AAPL's real open membership -> `(14593, 1982-11-18, 2025-12-31)`.

    VERBATIM `03.10-LIVE-CHECK.json` key
    `C4_sample_crsp_a_indexes.dsp500list_v2`. The interval is CLOSED on both
    ends, matching `IndexConstituentDataset`'s convention exactly, and the end
    is the EXPLICIT product end rather than a null.
    """
    from tests.crsp_fixtures import DSP500_ROWS

    membership = _sp500_tier(tmp_path, list(DSP500_ROWS))
    intervals = membership.permno_intervals(membership.SP500)

    assert intervals.to_dicts() == [
        {
            "permno": 14593,
            "start_date": date(1982, 11, 18),
            "end_date": date(2025, 12, 31),
        }
    ]
    assert intervals.schema["permno"] == pl.Int64
    assert intervals.schema["start_date"] == pl.Date
    assert intervals.schema["end_date"] == pl.Date


def test_sp500_keeps_two_disjoint_spells_and_merges_touching_ones(tmp_path):
    """A PERMNO that left and rejoined keeps BOTH intervals.

    Collapsing them into one would fabricate membership across the years the
    security was NOT in the index -- the survivorship bias in reverse. Only
    intervals that overlap or TOUCH (end, end+1 day) are merged.
    """
    rows = [
        _sp500("70000", "1990-01-02", "1995-06-30"),  # SYNTHETIC: left in 1995
        _sp500("70000", "2001-03-01", "2025-12-31"),  # SYNTHETIC: rejoined
        _sp500("70001", "2000-01-03", "2005-12-31"),  # SYNTHETIC
        _sp500("70001", "2006-01-01", "2010-06-30"),  # SYNTHETIC: touching
        _sp500("70001", "2009-01-01", "2012-12-31"),  # SYNTHETIC: overlapping
    ]
    membership = _sp500_tier(tmp_path, rows)
    intervals = membership.permno_intervals(membership.SP500)

    assert intervals.to_dicts() == [
        {
            "permno": 70000,
            "start_date": date(1990, 1, 2),
            "end_date": date(1995, 6, 30),
        },
        {
            "permno": 70000,
            "start_date": date(2001, 3, 1),
            "end_date": date(2025, 12, 31),
        },
        {
            "permno": 70001,
            "start_date": date(2000, 1, 3),
            "end_date": date(2012, 12, 31),
        },
    ]


def test_sp500_null_end_and_late_end_are_clipped_to_the_product_end(tmp_path):
    """A NULL or post-coverage end becomes the product end, and is counted.

    An open end would reach `IndexConstituentDataset._densify`, whose
    open-interval branch extends the panel's right edge to WALL-CLOCK TODAY --
    months past the CRSP price coverage the panel is supposed to be bounded
    by.
    """
    rows = [
        _sp500("70002", "2010-01-04", None),  # SYNTHETIC: NULL mbrenddt
        _sp500("70003", "2015-01-02", "2030-01-01"),  # SYNTHETIC: past coverage
    ]
    membership = _sp500_tier(tmp_path, rows)
    intervals = membership.permno_intervals(membership.SP500)

    assert intervals["end_date"].to_list() == [
        date(2025, 12, 31),
        date(2025, 12, 31),
    ]
    assert membership.report["clipped_to_product_end"] == 2


def test_sp500_spell_starting_after_the_product_end_is_dropped_and_counted(tmp_path):
    """A membership that begins after CRSP coverage ends is dropped, loudly.

    Keeping it would emit an interval whose start is later than its end; the
    count in the report is what keeps the drop from being silent.
    """
    from tests.crsp_fixtures import DSP500_ROWS

    rows = [
        *DSP500_ROWS,
        _sp500("70004", "2026-01-05", None),  # SYNTHETIC: after the product end
    ]
    membership = _sp500_tier(tmp_path, rows)
    intervals = membership.permno_intervals(membership.SP500)

    assert intervals["permno"].to_list() == [14593]
    assert membership.report["dropped_after_product_end"] == 1


def test_sp500_rows_of_another_indno_are_excluded_and_counted(tmp_path):
    """Only `indno == 1000500` is the S&P 500; other index numbers are counted."""
    from tests.crsp_fixtures import DSP500_ROWS

    rows = [
        *DSP500_ROWS,
        # SYNTHETIC: a different index number in the same table.
        _sp500("70005", "2000-01-03", "2005-12-31", indno="1000501"),
    ]
    membership = _sp500_tier(tmp_path, rows)
    intervals = membership.permno_intervals(membership.SP500)

    assert intervals["permno"].to_list() == [14593]
    assert membership.report["other_indno_rows"] == 1


def test_roster_in_range_returns_sorted_permno_strings_overlapping_the_window(
    tmp_path,
):
    """`permnos_in_range` is the roster the CLI pulls: overlap, numeric order.

    The order is part of the contract (`UniverseCatalog.get_symbols_in_range`
    argues it at length): `--limit` slices this list, so an unstable order
    truncates to a different batch on every run and no run ever meets the
    previous run's watermarks.
    """
    from tests.crsp_fixtures import DSP500_ROWS

    rows = [
        *DSP500_ROWS,  # 14593, 1982-11-18..2025-12-31
        _sp500("7000", "1998-01-02", "2005-12-31"),  # SYNTHETIC: in the window
        _sp500("90319", "1999-01-04", "2025-12-31"),  # SYNTHETIC: in the window
        _sp500("70000", "1990-01-02", "1995-06-30"),  # SYNTHETIC: ended before
    ]
    membership = _sp500_tier(tmp_path, rows)

    roster = membership.permnos_in_range(
        membership.SP500, "2000-01-01", "2000-12-31"
    )

    # NUMERIC order, not lexicographic: "14593" < "7000" as text.
    assert roster == ["7000", "14593", "90319"]


def test_unknown_index_name_raises_listing_the_supported_indexes(tmp_path):
    """A typo'd universe raises and names the two real ones.

    `[]` is a legitimate answer for a real index, so a typo must never be
    answerable -- it would hand back an empty roster indistinguishable from
    "nobody was a member".
    """
    membership = _sp500_tier(tmp_path, [])

    with pytest.raises(ValueError) as excinfo:
        membership.permno_intervals("sp500")

    message = str(excinfo.value)
    assert "sp500" in message
    assert "crsp_sp500" in message
    assert "comp_nasdaq100" in message


def test_every_sp500_end_date_is_explicit_and_within_the_product_end(tmp_path):
    """The invariant the whole module exists to hold (T-03.10-24)."""
    from tests.crsp_fixtures import DSP500_ROWS

    rows = [
        *DSP500_ROWS,
        _sp500("70002", "2010-01-04", None),  # SYNTHETIC
        _sp500("70003", "2015-01-02", "2030-01-01"),  # SYNTHETIC
    ]
    membership = _sp500_tier(tmp_path, rows)
    intervals = membership.permno_intervals(membership.SP500)

    assert intervals["end_date"].null_count() == 0
    assert intervals["end_date"].max() <= membership.reference.product_end
    assert (
        intervals.filter(pl.col("start_date") > pl.col("end_date")).height == 0
    )


# ---------------------------------------------------------------------------
# D-14 -- Nasdaq-100 through Compustat + the CCM link table
# ---------------------------------------------------------------------------


def test_nasdaq100_verbatim_alphabet_spells_link_to_both_permnos(tmp_path):
    """Alphabet's two share classes are two members under ONE gvkey.

    VERBATIM `03.10-LIVE-CHECK-2.json` keys `L8_2` (the spells) and `L7_4`
    (the links). The join is on `iid = liid`, never on `linkprim`: filtering
    `linkprim` would keep one Alphabet class and silently lose the other. The
    two `NR` links contribute nothing -- their `lpermno` is NULL and their
    linktype is not in `LINK_TYPES`.
    """
    from tests.crsp_fixtures import CCM_ROWS, IDXCST_ROWS

    membership = _ndx_tier(tmp_path, list(IDXCST_ROWS), list(CCM_ROWS))
    intervals = membership.permno_intervals(membership.NASDAQ100)

    assert intervals.to_dicts() == [
        {  # GOOG, liid 03
            "permno": 14542,
            "start_date": date(2014, 4, 3),
            "end_date": date(2025, 12, 31),
        },
        {  # GOOGL, liid 01
            "permno": 90319,
            "start_date": date(2005, 12, 21),
            "end_date": date(2025, 12, 31),
        },
    ]
    assert membership.report["unlinked"] == []


def test_nasdaq100_spell_covered_by_an_lu_link_keeps_its_own_end(tmp_path):
    """A closed spell inside a wider `LU` link keeps the SPELL's dates.

    The link bounds the mapping, not the membership: intersecting is what
    keeps a link that outlives a membership from extending it.
    """
    spells = [_spell("100001", "01", "2003-01-02", "2010-06-30")]  # SYNTHETIC
    links = [
        # SYNTHETIC: an LU link that starts before and ends after the spell.
        _link("100001", "01", "81001.0", "2002-01-01", "2012-12-31", linktype="LU"),
    ]
    membership = _ndx_tier(tmp_path, spells, links)
    intervals = membership.permno_intervals(membership.NASDAQ100)

    assert intervals.to_dicts() == [
        {
            "permno": 81001,
            "start_date": date(2003, 1, 2),
            "end_date": date(2010, 6, 30),
        }
    ]
    assert membership.report["unlinked"] == []


def test_nasdaq100_spell_after_the_product_end_is_dropped_and_an_open_one_clipped(
    tmp_path,
):
    """The two coverage edges at once (D-14: 10 spells start after 2025-12-31).

    A spell that starts after CRSP's price coverage is dropped and counted; an
    open spell whose link is also open ends at the product end and is counted.
    Neither may raise as "unlinked": a dropped spell is out of coverage, not
    unlinkable.
    """
    spells = [
        _spell("100002", "01", "2026-01-05", None),  # SYNTHETIC: after coverage
        _spell("100003", "01", "2019-05-02", None),  # SYNTHETIC: open spell
    ]
    links = [
        _link("100002", "01", "81002.0", "2010-01-01", None),  # SYNTHETIC
        _link("100003", "01", "81003.0", "2010-01-01", None),  # SYNTHETIC
    ]
    membership = _ndx_tier(tmp_path, spells, links)
    intervals = membership.permno_intervals(membership.NASDAQ100)

    assert intervals.to_dicts() == [
        {
            "permno": 81003,
            "start_date": date(2019, 5, 2),
            "end_date": date(2025, 12, 31),
        }
    ]
    assert membership.report["dropped_after_product_end"] == 1
    assert membership.report["clipped_to_product_end"] == 1
    assert membership.report["unlinked"] == []


def test_nasdaq100_spell_with_no_link_raises_naming_the_spell_and_the_remedy(
    tmp_path,
):
    """A membership day with no PERMNO is REFUSED, not dropped (T-03.10-23).

    Silently dropping it removes a real member from the universe -- exactly
    the survivorship bias this layer exists to prevent, and invisible
    downstream because the missing security looks like a data gap.
    """
    spells = [_spell("100004", "01", "2008-02-01", "2012-12-31")]  # SYNTHETIC
    links = [
        # SYNTHETIC: a link for a DIFFERENT gvkey -- nothing covers 100004.
        _link("999999", "01", "81999.0", "2000-01-01", None),
    ]
    membership = _ndx_tier(tmp_path, spells, links)

    with pytest.raises(ValueError) as excinfo:
        membership.permno_intervals(membership.NASDAQ100)

    message = str(excinfo.value)
    assert "100004" in message
    assert "2008-02-01" in message
    assert "2012-12-31" in message
    assert "allow_unlinked" in message


def test_nasdaq100_unlinked_tail_is_reported_when_allow_unlinked(tmp_path):
    """With the opt-in, the linked part is returned and the gap is RECORDED.

    The opt-in changes where the fact is written, never whether it exists.
    """
    spells = [_spell("100005", "01", "2015-01-02", None)]  # SYNTHETIC: open
    links = [
        # SYNTHETIC: the link ends long before the membership does.
        _link("100005", "01", "81005.0", "2010-01-01", "2020-06-30"),
    ]
    membership = _ndx_tier(tmp_path, spells, links)

    intervals = membership.permno_intervals(
        membership.NASDAQ100, allow_unlinked=True
    )

    assert intervals.to_dicts() == [
        {
            "permno": 81005,
            "start_date": date(2015, 1, 2),
            "end_date": date(2020, 6, 30),
        }
    ]
    assert membership.report["unlinked"] == [
        {
            "gvkey": "100005",
            "iid": "01",
            "from": "2015-01-02",
            "thru": "2025-12-31",
            "uncovered": [["2020-07-01", "2025-12-31"]],
        }
    ]


def test_nasdaq100_link_gap_within_tolerance_is_tolerated_and_reported(tmp_path):
    """A few calendar days between two links of the SAME PERMNO is not a hole.

    Link A ends on a Friday and link B starts the following Tuesday: three
    calendar days, none of them a trading day. Tolerated, and still listed --
    the tolerance suppresses the refusal, never the record.
    """
    spells = [_spell("100006", "01", "2012-01-02", "2012-12-31")]  # SYNTHETIC
    links = [
        # SYNTHETIC: ends Friday 2012-06-01.
        _link("100006", "01", "81006.0", "2010-01-01", "2012-06-01"),
        # SYNTHETIC: resumes Tuesday 2012-06-05, same PERMNO.
        _link("100006", "01", "81006.0", "2012-06-05", "2013-12-31"),
    ]
    membership = _ndx_tier(tmp_path, spells, links)
    intervals = membership.permno_intervals(membership.NASDAQ100)

    assert membership.report["tolerated_gaps"] == [
        {
            "gvkey": "100006",
            "iid": "01",
            "start": "2012-06-02",
            "end": "2012-06-04",
            "days": 3,
        }
    ]
    assert membership.report["unlinked"] == []
    # The gap is wider than a touch, so the two pieces stay two intervals.
    assert intervals.to_dicts() == [
        {
            "permno": 81006,
            "start_date": date(2012, 1, 2),
            "end_date": date(2012, 6, 1),
        },
        {
            "permno": 81006,
            "start_date": date(2012, 6, 5),
            "end_date": date(2012, 12, 31),
        },
    ]


def test_nasdaq100_ambiguous_links_for_one_iid_raise_naming_both_permnos(tmp_path):
    """One `(gvkey, iid)` cannot be two securities on one day (T-03.10-25)."""
    spells = [_spell("100007", "01", "2012-01-03", "2012-06-29")]  # SYNTHETIC
    links = [
        _link("100007", "01", "81007.0", "2011-01-01", "2012-03-30"),  # SYNTHETIC
        _link("100007", "01", "81008.0", "2012-01-03", "2012-12-31"),  # SYNTHETIC
    ]
    membership = _ndx_tier(tmp_path, spells, links)

    with pytest.raises(ValueError) as excinfo:
        membership.permno_intervals(membership.NASDAQ100)

    message = str(excinfo.value)
    assert "81007" in message
    assert "81008" in message
    assert "2012-01-03" in message
    assert "2012-03-30" in message


def test_nasdaq100_left_censored_spells_are_counted_and_coverage_starts_in_1995(
    tmp_path,
):
    """1995-01-01 is a CENSOR date, not 100 simultaneous index additions.

    `03.10-LIVE-CHECK-2.json` key `L8_3`: 100 of the Nasdaq-100 spells start
    that exact day, which is where Compustat's history begins rather than
    where those memberships did.
    """
    spells = [
        _spell("100008", "01", "1995-01-01", "1999-12-31"),  # SYNTHETIC
        _spell("100009", "01", "1995-01-01", None),  # SYNTHETIC
        _spell("100010", "01", "2005-06-01", None),  # SYNTHETIC
    ]
    links = [
        _link("100008", "01", "81008.0", "1990-01-01", None),  # SYNTHETIC
        _link("100009", "01", "81009.0", "1990-01-01", None),  # SYNTHETIC
        _link("100010", "01", "81010.0", "1990-01-01", None),  # SYNTHETIC
    ]
    membership = _ndx_tier(tmp_path, spells, links)
    membership.permno_intervals(membership.NASDAQ100)

    assert membership.report["left_censored_spells"] == 2
    assert membership.PIT_COVERAGE_START[membership.NASDAQ100] == "1995-01-01"
    assert membership.PIT_COVERAGE_START[membership.SP500] == "1925-12-31"


def test_ccm_float_lpermno_becomes_int64_and_a_non_integral_one_raises(tmp_path):
    """`lpermno` is `double precision` on the server; a PERMNO is an integer.

    Casting blindly would turn a corrupt `81009.5` into `81009` or `81010` --
    a real, different security -- so a non-integral value raises instead.
    """
    from tests.crsp_fixtures import CCM_ROWS, IDXCST_ROWS

    membership = _ndx_tier(tmp_path, list(IDXCST_ROWS), list(CCM_ROWS))
    intervals = membership.permno_intervals(membership.NASDAQ100)
    assert intervals.schema["permno"] == pl.Int64
    assert 90319 in intervals["permno"].to_list()

    spells = [_spell("100011", "01", "2012-01-03", "2012-12-31")]  # SYNTHETIC
    links = [_link("100011", "01", "81009.5", "2010-01-01", None)]  # SYNTHETIC
    broken = _ndx_tier(tmp_path / "broken", spells, links)

    with pytest.raises(ValueError) as excinfo:
        broken.permno_intervals(broken.NASDAQ100)

    assert "81009.5" in str(excinfo.value)


def test_nasdaq100_roster_in_range_resolves_the_alphabet_permnos(tmp_path):
    """The roster the CLI pulls for `--universe comp_nasdaq100`."""
    from tests.crsp_fixtures import CCM_ROWS, IDXCST_ROWS

    membership = _ndx_tier(tmp_path, list(IDXCST_ROWS), list(CCM_ROWS))

    assert membership.permnos_in_range(
        membership.NASDAQ100, "2015-01-01", "2015-12-31"
    ) == ["14542", "90319"]
    # GOOG (14542) joined 2014-04-03, so a 2010 window holds only GOOGL.
    assert membership.permnos_in_range(
        membership.NASDAQ100, "2010-01-01", "2010-12-31"
    ) == ["90319"]


# ---------------------------------------------------------------------------
# 03.11-05 -- the ticker branch is GONE, the PERMNO branch is intact
# ---------------------------------------------------------------------------


def test_the_ticker_branch_no_longer_exists_on_this_class():
    """`symbol_intervals` / `_symbol_frame` are DELETED, not deprecated.

    The whole point of the PERMNO migration is that the membership layer has
    exactly ONE identity to answer in. Leaving the ticker branch beside the
    PERMNO one would keep a second, silently-wrong way to build a universe:
    a ticker-keyed mask applied to a PERMNO-keyed price panel intersects to
    nothing, and an empty universe reads downstream as "no positions" rather
    than as the misconfiguration it is.

    `CrspSymbology` was imported for that branch alone, so the import going
    with it is the assertion that no second use crept in (S5: deleting code
    means deleting what points at it).
    """
    # Aliased: `membership` is the name every other test in this file gives a
    # `CrspMembership` INSTANCE, and an unaliased `import membership` here
    # would read as one.
    from quantlab.dataset.crsp import membership as membership_module
    from quantlab.dataset.crsp.membership import CrspMembership

    assert not hasattr(CrspMembership, "symbol_intervals")
    assert not hasattr(CrspMembership, "_symbol_frame")
    assert not hasattr(membership_module, "CrspSymbology")
    # The key type narrows with the branch: on the PERMNO axis there is only
    # one kind of key, so `int | str` would advertise a choice that is gone.
    assert membership_module._Key is int


def test_the_permno_branch_and_its_numeric_order_contract_survive_intact():
    """Everything the deletion must NOT take with it.

    `permnos_in_range`'s docstring is the DEPENDENCY of this whole migration
    -- it is where "the order is part of the contract, and it is NUMERIC" was
    first argued, and `quantlab/utils/symbol_axis.py` still cites it as the
    provenance of `sort_symbol_axis`. Deleting the sibling method is not a
    licence to touch it.
    """
    from quantlab.dataset.crsp.membership import CrspMembership

    for name in (
        "permno_intervals",
        "permnos_in_range",
        "_sp500_pieces",
        "_nasdaq100_pieces",
        "_ccm_links_by_key",
        "_frame",
    ):
        assert hasattr(CrspMembership, name), name

    assert "NUMERIC" in CrspMembership.permnos_in_range.__doc__


def test_permno_intervals_are_ordered_numerically_not_lexicographically(tmp_path):
    """A four-digit PERMNO is where numeric and text order fork.

    Historical PERMNOs happen to be five digits (~10000-93436), so the two
    orders COINCIDE on today's universe and a lexicographic regression would
    be invisible -- until one four-digit PERMNO appears, and then `7000`
    sorts after `14593`. PERMNO 7000 is a real never-ticker security
    (`tests/crsp_fixtures.py:SECINFO_ROWS`), which is exactly why it is the
    fixture that makes this assertion load-bearing.
    """
    rows = [
        _sp500("14593", "1990-01-02", "2025-12-31"),  # SYNTHETIC
        _sp500("7000", "1990-01-02", "2025-12-31"),  # SYNTHETIC
        _sp500("93436", "1990-01-02", "2025-12-31"),  # SYNTHETIC
    ]
    membership = _sp500_tier(tmp_path, rows)

    permnos = membership.permno_intervals(membership.SP500)["permno"].to_list()

    assert permnos == [7000, 14593, 93436]
    assert permnos != sorted(permnos, key=str)
