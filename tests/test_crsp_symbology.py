"""PERMNO -> period-correct ticker: the CRSP symbology (D-04, D-10, D-15, D-19).

Every naming case below is a LIVE CRSP row. The provenance rule of
`tests/crsp_fixtures.py` holds here too: a value transcribed from a live check
names the JSON key it came from, and a value invented for the test carries a
`# SYNTHETIC` comment on the spot.

The live sources:

- `03.10-LIVE-CHECK.json` key `C5_ticker_hist_crsp_a_stock.stksecurityinfohist`
  -- FB -> META (13407), BRK (83443), GOOG -> GOOGL (90319), AAPL (14593);
- `03.10-LIVE-CHECK-2.json` key `L3_3` (Lehman's NULL-ticker delisting
  interval), key `L5_1` (the ticker OVERLAPS between two PERMNOs: LYB, WIN),
  key `L6_1` (the three BF lines);
- `03.10-LIVE-CHECK-NDX-QQQ.json` key `C1_qqq_names` (QQQ -> QQQQ -> QQQ).

Why the overlap cases matter more than the rename cases: a rename moves a
PERMNO from one column to another, which is visible. Two PERMNOs landing in
ONE column is invisible -- the panel still has a `BF` series, it just holds two
different companies' prices. That is the failure D-04 exists to prevent, and it
is what the collision pass and `resolve_collisions` are tested for here.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quantlab.dataset.crsp_reference import REFERENCE_TABLES_BY_NAME
from quantlab.dataset.crsp_symbology import CrspSymbology
from quantlab.enums.data import TRADEABLE_TICKER_PATTERN
from tests.crsp_fixtures import LEHMAN_2008_ROWS, SECINFO_ROWS, dsf_row, secinfo_row

_SECINFO_SPEC = REFERENCE_TABLES_BY_NAME["stksecurityinfohist"]
_DSF_COLUMNS = ("permno", "dlycaldt", "dlydelflg", "ticker")


# ---------------------------------------------------------------------------
# Extra `stksecurityinfohist` intervals
# ---------------------------------------------------------------------------

#: Alphabet's class-C issue. SYNTHETIC as a `stksecurityinfohist` row: the
#: PERMNO 14542 and the 2014-04-03 start are VERBATIM `03.10-LIVE-CHECK-2.json`
#: key `L7_4` (the CCM link `lpermno 14542.0` from 2014-04-03), but that live
#: query read the LINK table, not the security history -- restating the spell
#: as an interval here is the invention. `ticker='GOOG'` with
#: `tradingsymbol='GOOG'` is what makes it the interesting case: a PERMNO that
#: CARRIES a class (C) and must nevertheless stay plain `GOOG`, because it
#: overlaps nothing.
GOOG_C_ROWS = [
    secinfo_row(14542, "2014-04-03", "2025-12-31", "GOOG", "GOOG", "C"),
]

#: Berkshire's class-A issue. SYNTHETIC: PERMNO 17778 and the interval edges
#: are invented. The SHAPE is verbatim 83443's own C5 rows -- `ticker='BRK'`
#: with a NULL `tradingsymbol` before 2002-01-02 and `BRK?` after -- which is
#: the point: before 2002 NEITHER issue can be spelled from its own row, and
#: only the overlap says they are two securities.
BRK_A_ROWS = [
    secinfo_row(17778, "1996-05-09", "2002-01-01", "BRK", None, "A"),
    secinfo_row(17778, "2002-01-02", "2025-12-31", "BRK", "BRKA", "A"),
]

#: LyondellBasell's two classes. The PERMNOs (12345, 12346), the shared ticker
#: `LYB`, the classes (A, B), the tradingsymbols (LYB, LYBB) and the overlap
#: window 2010-10-14..2010-12-06 are all VERBATIM `03.10-LIVE-CHECK-2.json`
#: key `L5_1` (that query projected `greatest(start)`/`least(end)`, i.e. the
#: overlap, which is exactly what is restated here).
LYB_ROWS = [
    secinfo_row(12345, "2010-10-14", "2010-12-06", "LYB", "LYB", "A"),
    secinfo_row(12346, "2010-10-14", "2010-12-06", "LYB", "LYBB", "B"),
]

#: The WIN overlap, VERBATIM `L5_1`: PERMNO 24803 with NO class and PERMNO
#: 59475 with class B share the ticker `WIN` over 1969-03-17..1981-04-16, and
#: BOTH have a NULL tradingsymbol. Nothing on either row can separate them --
#: only the overlap plus the class can.
WIN_ROWS = [
    secinfo_row(24803, "1969-03-17", "1981-04-16", "WIN", None, None),
    secinfo_row(59475, "1969-03-17", "1981-04-16", "WIN", None, "B"),
]

#: Every interval the naming tests run against.
ALL_SECINFO_ROWS = (
    list(SECINFO_ROWS) + GOOG_C_ROWS + BRK_A_ROWS + LYB_ROWS + WIN_ROWS
)

#: The QQQ override (D-15), VERBATIM the `symbol_overrides` value RESEARCH
#: § D-15/D-16 prescribes for the benchmark store.
QQQ_OVERRIDES = {"86755": "QQQ"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _secinfo_frame(rows) -> pl.DataFrame:
    """`rows` as the typed frame `CrspReference.table(...)` would return.

    Built through the spec's own `cast`, so the dtypes here are the production
    dtypes -- a test that passed on String dates would prove nothing about the
    Date-typed frame the reader hands over.
    """
    frame = pl.DataFrame(
        [{name: row.get(name) for name in _SECINFO_SPEC.columns} for row in rows],
        schema={name: pl.String for name in _SECINFO_SPEC.columns},
    )
    return _SECINFO_SPEC.cast(frame)


def _symbology(rows=None, overrides=None) -> CrspSymbology:
    return CrspSymbology(_secinfo_frame(ALL_SECINFO_ROWS if rows is None else rows),
                         overrides)


def _symbol_on(intervals: pl.DataFrame, permno: int, day: str) -> str | None:
    """The one symbol `permno` carries on `day`, or `None` if no interval
    covers it. Raises when two intervals of one PERMNO cover one day, which
    would itself be a symbology bug."""
    as_of = date.fromisoformat(day)
    hit = intervals.filter(
        (pl.col("permno") == permno)
        & (pl.col("start_date") <= as_of)
        & (pl.col("end_date") >= as_of)
    )
    if hit.height > 1:
        raise AssertionError(
            f"{permno} has {hit.height} intervals covering {day}: {hit.to_dicts()}"
        )
    return None if hit.is_empty() else hit["symbol"][0]


def _daily_frame(records) -> pl.DataFrame:
    """Daily rows as `label_rows` receives them: `permno` Int64, `timestamp`
    Datetime, `dlydelflg` String -- the shape `CrspStockDataset._scan_raw`
    produces (`quantlab/dataset/crsp.py`)."""
    return pl.DataFrame(
        [{name: record.get(name) for name in _DSF_COLUMNS} for record in records],
        schema={name: pl.String for name in _DSF_COLUMNS},
    ).select(
        pl.col("permno").cast(pl.Int64),
        pl.col("dlycaldt").str.to_date(strict=False).cast(pl.Datetime("us"))
        .alias("timestamp"),
        pl.col("dlydelflg"),
    )


# ---------------------------------------------------------------------------
# Task 1 -- the intervals
# ---------------------------------------------------------------------------


def test_intervals_are_sorted_typed_and_one_row_per_input_interval():
    intervals = _symbology().symbol_intervals()

    assert intervals.columns == ["permno", "symbol", "start_date", "end_date"]
    assert intervals.schema["permno"] == pl.Int64
    assert intervals.schema["symbol"] == pl.String
    assert intervals.schema["start_date"] == pl.Date
    assert intervals.schema["end_date"] == pl.Date
    assert intervals.height == len(ALL_SECINFO_ROWS)
    assert intervals.equals(intervals.sort(["permno", "start_date"]))


def test_rename_13407_is_fb_through_2022_06_08_and_meta_after():
    """VERBATIM C5: one PERMNO, two names, a hard boundary between them."""
    intervals = _symbology().symbol_intervals()

    assert _symbol_on(intervals, 13407, "2012-05-18") == "FB"
    assert _symbol_on(intervals, 13407, "2022-06-08") == "FB"
    assert _symbol_on(intervals, 13407, "2022-06-09") == "META"
    assert _symbol_on(intervals, 13407, "2025-12-31") == "META"

    fb = intervals.filter((pl.col("permno") == 13407) & (pl.col("symbol") == "FB"))
    assert fb["end_date"].max() == date(2022, 6, 8)


def test_share_class_a_alone_never_suffixes_goog_or_googl():
    """90319 carries class A throughout and is still plain GOOG/GOOGL, and the
    class-C issue 14542 is plain GOOG because it overlaps nobody."""
    intervals = _symbology().symbol_intervals()

    assert _symbol_on(intervals, 90319, "2014-04-02") == "GOOG"
    assert _symbol_on(intervals, 90319, "2014-04-03") == "GOOGL"
    assert _symbol_on(intervals, 14542, "2014-04-03") == "GOOG"
    # The two GOOG spells must not overlap, or the panel would hold two
    # companies in one column on at least one day.
    assert _symbol_on(intervals, 14542, "2014-04-02") is None


def test_brk_classes_are_suffixed_before_and_after_2002():
    """Before 2002-01-02 BOTH Berkshire issues have a NULL `tradingsymbol`
    (VERBATIM C5 for 83443), so the class suffix can only come from the
    overlap. After it, the tradingsymbol rule reaches the same answer."""
    intervals = _symbology().symbol_intervals()

    assert _symbol_on(intervals, 83443, "1996-05-09") == "BRK.B"
    assert _symbol_on(intervals, 17778, "1996-05-09") == "BRK.A"
    assert _symbol_on(intervals, 83443, "2002-01-02") == "BRK.B"
    assert _symbol_on(intervals, 17778, "2002-01-02") == "BRK.A"
    assert _symbol_on(intervals, 83443, "2025-12-31") == "BRK.B"


def test_bf_three_lines_are_bf_a_bf_b_and_bf_on_both_sides_of_2002():
    """VERBATIM L6_1: 29938 (A), 29946 (B) and 88279 (no class) all ticker
    `BF`. The unclassed line keeps the bare ticker on both sides."""
    intervals = _symbology().symbol_intervals()

    for day in ("2001-06-01", "2002-01-02", "2004-07-27"):
        assert _symbol_on(intervals, 29938, day) == "BF.A", day
        assert _symbol_on(intervals, 29946, day) == "BF.B", day
    for day in ("2001-06-01", "2002-01-02", "2004-06-10"):
        assert _symbol_on(intervals, 88279, day) == "BF", day


def test_lyb_uses_the_tradingsymbol_rule_for_the_b_line_only():
    """VERBATIM L5_1: `tradingsymbol='LYBB'` spells the class, `'LYB'` does
    not -- so the A line stays `LYB` even though it carries a class."""
    intervals = _symbology().symbol_intervals()

    assert _symbol_on(intervals, 12345, "2010-10-14") == "LYB"
    assert _symbol_on(intervals, 12346, "2010-10-14") == "LYB.B"


def test_win_overlap_suffixes_only_the_permno_that_carries_a_class():
    """VERBATIM L5_1: 24803 has NO class, 59475 has class B, both NULL
    tradingsymbols. The classed one moves; the unclassed one keeps `WIN`,
    because inventing a suffix for it would rename a security CRSP never
    renamed."""
    intervals = _symbology().symbol_intervals()

    assert _symbol_on(intervals, 24803, "1969-03-17") == "WIN"
    assert _symbol_on(intervals, 59475, "1969-03-17") == "WIN.B"


def test_a_null_ticker_interval_carries_the_previous_symbol():
    """VERBATIM L3_3: Lehman's 2008-09-18 interval has NO ticker at all, and
    it is the interval the delisting return falls in (D-10/D-19)."""
    intervals = _symbology().symbol_intervals()

    assert _symbol_on(intervals, 80599, "2008-09-17") == "LEH"
    assert _symbol_on(intervals, 80599, "2008-09-18") == "LEH"


def test_a_first_interval_with_a_null_ticker_has_no_symbol():
    """There is nothing to carry from, and inventing a label would be a guess
    about identity -- so the symbol is null and the rows become reported
    drops rather than a wrong column."""
    rows = [
        # SYNTHETIC: a PERMNO whose history OPENS with an unnamed interval.
        secinfo_row(99999, "2009-01-02", "2009-01-05", None, None, None),
        secinfo_row(99999, "2009-01-06", "2009-12-31", "ZZZ", "ZZZ", None),
    ]
    intervals = _symbology(rows).symbol_intervals()

    assert _symbol_on(intervals, 99999, "2009-01-02") is None
    assert _symbol_on(intervals, 99999, "2009-01-06") == "ZZZ"


def test_override_pins_qqq_across_the_qqqq_span():
    """D-15: PERMNO 86755 is `QQQ` for its whole history, including the
    2004-12-01..2011-03-22 spell when CRSP called it QQQQ
    (VERBATIM NDX-QQQ `C1_qqq_names`)."""
    intervals = _symbology(overrides=QQQ_OVERRIDES).symbol_intervals()

    for day in ("1999-03-10", "2004-12-01", "2010-06-01", "2011-03-22",
                "2025-12-31"):
        assert _symbol_on(intervals, 86755, day) == "QQQ", day
    assert "QQQQ" not in set(intervals["symbol"].drop_nulls())


def test_every_symbol_matches_the_tradeable_ticker_pattern():
    """The symbol becomes a Zarr coordinate label, so it must be spellable in
    the same alphabet every other quantlab symbol uses. Anything that is not
    is REPORTED rather than silently accepted (T-03.10-18)."""
    symbology = _symbology(overrides=QQQ_OVERRIDES)
    intervals = symbology.symbol_intervals()

    bad = [
        symbol
        for symbol in intervals["symbol"].drop_nulls().unique()
        if not TRADEABLE_TICKER_PATTERN.match(symbol)
    ]
    assert bad == []
    assert symbology.report["nonconforming_symbols"] == []


def test_class_suffixed_report_lists_every_interval_the_collision_pass_moved():
    """T-03.10-17: a symbol this module CHANGED must leave a trace, so the
    plan-08 sidecar can show which column a security actually landed in."""
    symbology = _symbology()
    symbology.symbol_intervals()

    suffixed = symbology.report["class_suffixed"]
    moved = {(entry["permno"], entry["symbol"]) for entry in suffixed}
    assert (83443, "BRK.B") in moved
    assert (17778, "BRK.A") in moved
    assert (59475, "WIN.B") in moved
    assert (29938, "BF.A") in moved
    assert (29946, "BF.B") in moved
    # The unclassed lines were never moved, so they are not in the report.
    assert 24803 not in {entry["permno"] for entry in suffixed}
    assert 88279 not in {entry["permno"] for entry in suffixed}
    # Every entry carries its own window, as ISO text (the report is written
    # to a JSON sidecar).
    entry = next(e for e in suffixed if e["permno"] == 59475)
    assert entry["start_date"] == "1969-03-17"
    assert entry["end_date"] == "1981-04-16"


def test_a_symbol_that_already_spells_its_class_is_not_suffixed_twice():
    """Two intervals that ALREADY read `BRK.B` (the tradingsymbol rule) and
    overlap must not become `BRK.B.B` -- the suffix is a spelling, not a
    counter."""
    rows = [
        # SYNTHETIC: two PERMNOs whose tradingsymbols both spell class B on
        # one ticker. Contrived, and the exact shape the guard exists for.
        secinfo_row(70001, "2005-01-03", "2006-12-29", "ABC", "ABCB", "B"),
        secinfo_row(70002, "2005-01-03", "2006-12-29", "ABC", "ABCB", "B"),
    ]
    intervals = _symbology(rows).symbol_intervals()

    assert set(intervals["symbol"]) == {"ABC.B"}
    # The two issues STILL collide -- that is left for `resolve_collisions`,
    # which refuses per (date, symbol) cell rather than inventing a spelling.
    assert set(intervals["permno"]) == {70001, 70002}
