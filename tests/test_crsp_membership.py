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
`quantlab/dataset/crsp_membership.py` exists, and a module-scope import would
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
    from quantlab.dataset.crsp_membership import CrspMembership
    from quantlab.dataset.crsp_reference import CrspReference
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
