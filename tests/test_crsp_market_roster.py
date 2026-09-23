"""`CrspMarketRoster` -- the whole-market CRSP roster.

The four things worth locking, in the order they can break:

1. The TYPE predicate actually excludes. A roster test whose fixture holds
   only ordinary common shares passes under `equity_common`, under
   `shrcd_10_11` and under `none` alike, and therefore tests nothing. Every
   fixture here carries at least one security each preset judges differently.
2. The OVERLAP predicate, not containment. A security delisted inside the
   window must stay in the roster -- that is the survivorship bias this layer
   removes, and on a whole-market roster it is the majority of the history.
3. Adjacent qualifying spells MERGE and non-qualifying ones do not bridge.
   `stksecurityinfohist` splits a security on every info change, so an
   unmerged roster reports one PERMNO as many.
4. The spell-level verdict is NOT the per-date verdict. This module decides
   who gets PULLED; `CrspStockDataset._apply_security_filter` decides which of
   their rows survive. A test that asserts day-level behaviour here would be
   asserting it in the wrong layer.
"""

from __future__ import annotations

import polars as pl
import pytest

from quantlab.dataset.crsp.market import CrspMarketRoster
from quantlab.dataset.crsp.reference import CrspReference

from crsp_fixtures import secinfo_row, write_reference_tables

PRODUCT_END = "2025-12-31"


def _reference(tmp_path, rows) -> CrspReference:
    """A reference tier holding exactly `rows` in `stksecurityinfohist`."""
    directory = write_reference_tables(
        tmp_path / "reference",
        rows_by_table={"crsp_a_stock.stksecurityinfohist": list(rows)},
        product_end=PRODUCT_END,
    )
    return CrspReference(directory)


# -- the fixture the type tests share ----------------------------------------
#
# Five securities, deliberately spanning the three presets' disagreements:
#
#   10001 ordinary US common  NS/EQTY/COM/CORP/Y  -- every preset keeps it
#   10002 a REIT              SB/EQTY/COM/CORP/Y  -- equity_common keeps,
#                                                   shrcd_10_11 drops (sharetype)
#   10003 non-US incorporated NS/EQTY/COM/CORP/N  -- equity_common keeps,
#                                                   shrcd_10_11 drops (usincflg)
#   10004 an ADR              AD/EQTY/COM/CORP/Y  -- both drop, `none` keeps
#   10005 an ETF              NS/ETF/NA /CORP/Y   -- both drop, `none` keeps
#
# 10004 is the row that makes `equity_common`'s sharetype allow-list load
# bearing: an ADR satisfies securitytype=EQTY AND securitysubtype=COM, so
# without the sharetype clause it would pass.
MIXED_ROWS = [
    secinfo_row(10001, "2000-01-03", PRODUCT_END, "AAA"),
    secinfo_row(10002, "2000-01-03", PRODUCT_END, "BBB", sharetype="SB"),
    secinfo_row(10003, "2000-01-03", PRODUCT_END, "CCC", usincflg="N"),
    secinfo_row(10004, "2000-01-03", PRODUCT_END, "DDD", sharetype="AD"),
    secinfo_row(
        10005, "2000-01-03", PRODUCT_END, "EEE",
        securitytype="ETF", securitysubtype="NA",
    ),
]


def test_equity_common_keeps_reits_and_foreign_and_drops_adrs_and_etfs(tmp_path):
    """The default preset, on a fixture where all three presets disagree."""
    roster = CrspMarketRoster(_reference(tmp_path, MIXED_ROWS))

    assert roster.permnos_in_range("2020-01-01", "2020-12-31") == [
        "10001",
        "10002",
        "10003",
    ]


def test_shrcd_10_11_is_narrower_than_equity_common_on_the_same_fixture(tmp_path):
    """The narrow preset drops the REIT and the non-US issuer.

    Asserted against the SAME rows as the test above, so the two results
    differ only by the predicate -- which is the only way to show the preset
    argument is actually read rather than ignored in favour of a default.
    """
    roster = CrspMarketRoster(_reference(tmp_path, MIXED_ROWS))

    assert roster.permnos_in_range(
        "2020-01-01", "2020-12-31", security_filter="shrcd_10_11"
    ) == ["10001"]


def test_filter_none_keeps_every_security_including_the_etf(tmp_path):
    """`none` is the whole table -- the ADR and the ETF come back too."""
    roster = CrspMarketRoster(_reference(tmp_path, MIXED_ROWS))

    assert roster.permnos_in_range(
        "2020-01-01", "2020-12-31", security_filter="none"
    ) == ["10001", "10002", "10003", "10004", "10005"]


def test_a_security_delisted_inside_the_window_stays_in_the_roster(tmp_path):
    """Overlap, not containment -- this is the survivorship-bias removal.

    10001 dies mid-window and 10002 is born mid-window; a containment
    predicate would return NEITHER, which is exactly the roster that makes a
    backtest look better than the market was.
    """
    roster = CrspMarketRoster(
        _reference(
            tmp_path,
            [
                secinfo_row(10001, "1999-01-04", "2020-06-30", "DEAD"),
                secinfo_row(10002, "2020-07-01", PRODUCT_END, "BORN"),
            ],
        )
    )

    assert roster.permnos_in_range("2020-01-01", "2020-12-31") == [
        "10001",
        "10002",
    ]


def test_a_security_whose_whole_life_precedes_the_window_is_excluded(tmp_path):
    """The other side of overlap: no shared day means not in the roster.

    Without this the previous test would also pass on an implementation that
    ignored the window entirely and returned the whole table.
    """
    roster = CrspMarketRoster(
        _reference(
            tmp_path,
            [
                secinfo_row(10001, "1990-01-02", "1995-12-29", "OLD"),
                secinfo_row(10002, "2020-01-02", PRODUCT_END, "NOW"),
            ],
        )
    )

    assert roster.permnos_in_range("2020-01-01", "2020-12-31") == ["10002"]


def test_adjacent_qualifying_spells_merge_into_one_interval(tmp_path):
    """Four ticker changes are one security, not four.

    `stksecurityinfohist` splits a security on every info change, so the
    unmerged frame would report this PERMNO four times and any caller counting
    rows would report four securities.
    """
    roster = CrspMarketRoster(
        _reference(
            tmp_path,
            [
                secinfo_row(13407, "2012-05-18", "2015-06-30", "FB"),
                secinfo_row(13407, "2015-07-01", "2019-12-31", "FB"),
                secinfo_row(13407, "2020-01-01", "2022-06-08", "FB"),
                secinfo_row(13407, "2022-06-09", PRODUCT_END, "META"),
            ],
        )
    )

    intervals = roster.permno_intervals()

    assert intervals.height == 1
    row = intervals.row(0, named=True)
    assert row["permno"] == 13407
    assert str(row["start_date"]) == "2012-05-18"
    assert str(row["end_date"]) == PRODUCT_END


def test_a_non_qualifying_era_does_not_bridge_two_qualifying_ones(tmp_path):
    """A security that was ordinary, became an ADR, then ordinary again.

    The middle era is dropped BEFORE the merge, so the result is two spans
    with a real hole -- not one span that silently asserts the security was
    ordinary common throughout. Merging first and filtering second would
    produce the single span, which is why the order is part of the contract.
    """
    roster = CrspMarketRoster(
        _reference(
            tmp_path,
            [
                secinfo_row(10001, "2000-01-03", "2004-12-31", "AAA"),
                secinfo_row(10001, "2005-01-03", "2009-12-31", "AAA",
                            sharetype="AD"),
                secinfo_row(10001, "2010-01-04", PRODUCT_END, "AAA"),
            ],
        )
    )

    intervals = roster.permno_intervals().sort("start_date")

    assert intervals.height == 2
    assert str(intervals.row(0, named=True)["end_date"]) == "2004-12-31"
    assert str(intervals.row(1, named=True)["start_date"]) == "2010-01-04"
    # And the window that sees ONLY the ADR era gets an empty roster.
    assert roster.permnos_in_range("2006-01-01", "2006-12-31") == []


def test_a_null_type_column_never_matches(tmp_path):
    """"Unknown type" is not "the type you asked for".

    A NULL must FAIL the predicate rather than pass it, or every security
    whose type columns have gone blank is readmitted.

    This pins the OUTCOME, not one mechanism. Null-rejection is delivered
    twice over in `_market_pieces` -- `.fill_null(False)` on the polars
    expression, and the falsy `if not row["_keep"]` in the loop that reads it
    -- and a mutation run confirmed either alone suffices: deleting the
    `.fill_null(False)` leaves this file entirely green. That redundancy is
    deliberate (the expression states its own meaning rather than relying on a
    reader knowing `not None` is True), but it is NOT what this test proves,
    and claiming otherwise would be a coverage assertion no run supports.
    """
    roster = CrspMarketRoster(
        _reference(
            tmp_path,
            [
                secinfo_row(10001, "2000-01-03", PRODUCT_END, "AAA"),
                secinfo_row(10002, "2000-01-03", PRODUCT_END, "BBB",
                            securitysubtype=None),
            ],
        )
    )

    assert roster.permnos_in_range("2020-01-01", "2020-12-31") == ["10001"]


def test_an_open_ended_spell_is_clipped_to_the_product_end(tmp_path):
    """`end_date` is never null and never past the vintage's product end.

    Both arms of the clip share one counter, so both are exercised here:
    10001 has a NULL `secinfoenddt` and 10002 an end BEYOND the vintage. The
    NULL arm is a guard against a future vintage rather than a path today's
    data takes -- `CrspMembership._sp500_pieces` records the same about its
    own NULL branch -- but a guard with no test is a guard nobody knows works.

    The NULL is expressed through the `secinfoenddt=` override rather than the
    positional `end`: `secinfo_row` renders its positional arguments with
    `str()`, so `None` there becomes the literal "None" and the date cast
    fails inside the fixture writer, not in the code under test.
    """
    roster = CrspMarketRoster(
        _reference(
            tmp_path,
            [
                secinfo_row(10001, "2000-01-03", PRODUCT_END, "AAA",
                            secinfoenddt=None, securityenddt=None),
                secinfo_row(10002, "2000-01-03", "2099-12-31", "BBB",
                            securityenddt="2099-12-31"),
            ],
        )
    )

    intervals = roster.permno_intervals().sort("permno")

    assert str(intervals.row(0, named=True)["end_date"]) == PRODUCT_END
    assert str(intervals.row(1, named=True)["end_date"]) == PRODUCT_END
    assert roster.report["spells_clipped_to_product_end"] == 2


def test_the_report_accounts_for_every_spell_read(tmp_path):
    """The report is the only way an operator sees what the filter removed.

    A roster that silently shrank is indistinguishable from a market that did.
    """
    roster = CrspMarketRoster(_reference(tmp_path, MIXED_ROWS))

    roster.permnos_in_range("2020-01-01", "2020-12-31")

    assert roster.report["spells_read"] == 5
    assert roster.report["spells_dropped_by_type"] == 2  # the ADR and the ETF
    assert roster.report["permnos_before_type_filter"] == 5
    assert roster.report["permnos_after_type_filter"] == 3
    assert roster.report["security_filter"] == {
        "securitytype": ["EQTY"],
        "securitysubtype": ["COM"],
        "sharetype": ["NS", "SB", "CE"],
    }


def test_the_report_is_reset_between_calls(tmp_path):
    """Two calls in a row must not accumulate -- the report describes the LAST.

    `CrspMembership.report` carries the same contract; a roster whose counters
    doubled on the second call would read as twice the market.
    """
    roster = CrspMarketRoster(_reference(tmp_path, MIXED_ROWS))

    roster.permno_intervals()
    roster.permno_intervals()

    assert roster.report["spells_read"] == 5
    assert roster.report["spells_dropped_by_type"] == 2


def test_an_inverted_window_is_refused_rather_than_answered_empty(tmp_path):
    """An inverted window overlaps nothing, so it would look like a real
    empty roster -- which is a legitimate answer for a real window."""
    roster = CrspMarketRoster(_reference(tmp_path, MIXED_ROWS))

    with pytest.raises(ValueError, match="is after"):
        roster.permnos_in_range("2020-12-31", "2020-01-01")


def test_an_unknown_preset_is_refused_and_names_the_legal_ones(tmp_path):
    """Falling back to the default on a typo would be a silently different
    roster, and a silently different roster is indistinguishable from a right
    one. The refusal comes from the shared `resolve_security_filter`, so this
    also locks that the roster does not re-implement preset resolution."""
    roster = CrspMarketRoster(_reference(tmp_path, MIXED_ROWS))

    with pytest.raises(ValueError, match="equity_common"):
        roster.permnos_in_range(
            "2020-01-01", "2020-12-31", security_filter="equity_commmon"
        )


def test_the_roster_order_is_numeric_not_lexicographic(tmp_path):
    """PERMNOs are integers rendered as strings; a text sort would put
    "14593" before "7000" and move the acquisition batch boundaries between
    two runs of the same command."""
    roster = CrspMarketRoster(
        _reference(
            tmp_path,
            [
                secinfo_row(14593, "2000-01-03", PRODUCT_END, "AAPL"),
                secinfo_row(7000, "2000-01-03", PRODUCT_END, "SEVEN"),
                secinfo_row(93436, "2000-01-03", PRODUCT_END, "TSLA"),
            ],
        )
    )

    assert roster.permnos_in_range("2020-01-01", "2020-12-31") == [
        "7000",
        "14593",
        "93436",
    ]


def test_the_frame_is_typed_even_when_it_is_empty(tmp_path):
    """An empty roster must still carry the schema, or the first caller to
    `.filter()` on it meets a column-not-found error instead of no rows."""
    roster = CrspMarketRoster(
        _reference(
            tmp_path,
            [secinfo_row(10001, "2000-01-03", PRODUCT_END, "AAA",
                         sharetype="AD")],
        )
    )

    intervals = roster.permno_intervals()

    assert intervals.height == 0
    assert intervals.schema == {
        "permno": pl.Int64,
        "start_date": pl.Date,
        "end_date": pl.Date,
    }
