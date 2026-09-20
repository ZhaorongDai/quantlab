"""The two CRSP-vendor universes as SYMBOL-level membership panels (D-04, D-05, D-14).

`tests/test_crsp_membership.py` owns the PERMNO-level half: which securities
were in an index, on which dates, from the reference tier alone. This module
owns the step that makes those intervals usable beside a price panel --
answering the same question in the price panel's OWN tickers -- and the two
ordinary `IndexConstituentDataset` subclasses built on top of it.

**Why the agreement test is the load-bearing one.** A universe mask is applied
to a price panel by SYMBOL. If the mask says `FB` on a day the price panel
calls that column `META`, the mask selects nothing and the run reports an
empty universe rather than an error. The only structural defence is that both
sides derive their symbols from ONE rule, `CrspSymbology.symbol_intervals()`,
and `test_membership_symbols_agree_with_the_crsp_price_panel` is what proves
they still do -- end to end, through a real pull and a real conversion, not by
inspecting the call graph.

**Provenance rule, continued from plan 02.** Every row below is either
VERBATIM from a named live-check key (through `tests/crsp_fixtures.py`) or
carries a `# SYNTHETIC` comment on the spot.

**Every quantlab import lives inside a test or inside a helper.** That is not
style: this module is written before `CrspMembership.symbol_intervals` and the
two constituent classes exist, and a module-scope import would turn the RED run
into a COLLECTION error -- zero tests discovered, which proves nothing about
the behaviour (TDD gate #3770).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from loguru import logger

# ---------------------------------------------------------------------------
# Row builders. The `{column: text-or-NULL}` shape a COPY would carry, the same
# one `tests/crsp_fixtures.py` builds its live rows with. Restated here (three
# lines) rather than imported, because `crsp_fixtures._row` is private and this
# plan must not edit that module.
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

#: The CRSP product end every fixture here is written against
#: (`03.10-LIVE-CHECK-2.json` key `L9_1`).
_PRODUCT_END = "2025-12-31"

#: The two coverage starts, restated from `CrspMembership.PIT_COVERAGE_START`
#: so a silent change to that constant fails here.
_SP500_COVERAGE_START = "1925-12-31"
_NDX_COVERAGE_START = "1995-01-01"


def _text_row(columns, values) -> dict[str, str | None]:
    """A live-check row: every value text, the token `"None"` meaning SQL NULL."""
    return {
        name: (None if value in (None, "None") else str(value))
        for name, value in zip(columns, values)
    }


def _sp500_spell(permno, start, end) -> dict[str, str | None]:
    """One `dsp500list_v2` membership spell."""
    return _text_row(
        _DSP500_COLUMNS, [permno, _SP500_INDNO, start, end, "NORM", "1100500"]
    )


def _ndx_spell(gvkey, iid, start, thru) -> dict[str, str | None]:
    """One `comp.idxcst_his` Nasdaq-100 membership spell."""
    return _text_row(_IDXCST_COLUMNS, [gvkey, iid, _NDX_GVKEYX, start, thru])


def _link(gvkey, liid, permno, start, end, linktype="LC") -> dict[str, str | None]:
    """One `crsp_a_ccm.ccmxpf_lnkhist` row.

    `lpermno` is rendered as a FLOAT because the live column type is `double
    precision` (`03.10-LIVE-CHECK-2.json` key `L7_2`).
    """
    return _text_row(
        _CCM_COLUMNS,
        [gvkey, "P", liid, linktype, permno, "45000.0", start, end],
    )


# ---------------------------------------------------------------------------
# Reference tiers
# ---------------------------------------------------------------------------

#: SYNTHETIC S&P 500 membership. The live `dsp500list_v2` sample
#: (`C4_sample_crsp_a_indexes.dsp500list_v2`) carries AAPL only, and AAPL never
#: renamed, so it cannot exercise D-04's rename or share-class arms. These two
#: spells put the FB->META rename (13407) and the BRK.B class suffix (83443)
#: inside a membership window; the SECURITIES and their symbol histories are
#: verbatim live rows, only the membership DATES here are invented.
_SP500_MEMBERSHIP_ROWS = [
    _sp500_spell("13407", "2013-12-23", _PRODUCT_END),  # SYNTHETIC
    _sp500_spell("83443", "2010-02-16", _PRODUCT_END),  # SYNTHETIC
]


def _secinfo_rows() -> list[dict[str, str | None]]:
    """The live `stksecurityinfohist` rows plus Alphabet's class C line.

    `tests/crsp_fixtures.py:SECINFO_ROWS` carries GOOG->GOOGL for PERMNO 90319
    (verbatim `C5`) but has no row for 14542, the class C issue that took the
    `GOOG` ticker over when 90319 became `GOOGL`. Without it the Nasdaq-100
    universe's second Alphabet member has no symbol at all.
    """
    from tests.crsp_fixtures import SECINFO_ROWS, secinfo_row

    return list(SECINFO_ROWS) + [
        # SYNTHETIC: PERMNO 14542 is verbatim (`L7_4`, the CCM link) and so is
        # 2014-04-03 (`L8_2`, the spell start); the secinfo INTERVAL carrying
        # ticker GOOG / shareclass C is invented, because the live check never
        # projected `stksecurityinfohist` for that PERMNO.
        secinfo_row(14542, "2014-04-03", "2025-12-31", "GOOG", "GOOG", "C"),
    ]


def _reference(tmp_path, rows_by_table, product_end=_PRODUCT_END) -> Path:
    """Write an offline reference tier holding exactly `rows_by_table`."""
    from tests.crsp_fixtures import write_reference_tables

    return write_reference_tables(
        Path(tmp_path) / "_reference", rows_by_table, product_end=product_end
    )


def _sp500_reference(tmp_path, spells=None, secinfo=None, product_end=_PRODUCT_END):
    return _reference(
        tmp_path,
        {
            "crsp_a_stock.stksecurityinfohist": (
                _secinfo_rows() if secinfo is None else secinfo
            ),
            "crsp_a_indexes.dsp500list_v2": (
                list(_SP500_MEMBERSHIP_ROWS) if spells is None else spells
            ),
        },
        product_end=product_end,
    )


def _ndx_reference(tmp_path, spells=None, links=None, secinfo=None):
    from tests.crsp_fixtures import CCM_ROWS, IDXCST_ROWS

    return _reference(
        tmp_path,
        {
            "crsp_a_stock.stksecurityinfohist": (
                _secinfo_rows() if secinfo is None else secinfo
            ),
            "comp.idxcst_his": list(IDXCST_ROWS) if spells is None else spells,
            "crsp_a_ccm.ccmxpf_lnkhist": list(CCM_ROWS) if links is None else links,
        },
    )


def _membership(reference_dir):
    from quantlab.dataset.crsp_membership import CrspMembership
    from quantlab.dataset.crsp_reference import CrspReference

    return CrspMembership(CrspReference(reference_dir))


def _rows(frame) -> list[tuple[str, str, str]]:
    """A symbol-interval frame as `(symbol, start ISO, end ISO)` tuples."""
    return [
        (str(record["symbol"]), str(record["start_date"]), str(record["end_date"]))
        for record in frame.to_dicts()
    ]


# ---------------------------------------------------------------------------
# D-04 / D-05 -- membership expressed in the price panel's own tickers
# ---------------------------------------------------------------------------


def test_symbol_intervals_rename_13407_is_fb_then_meta(tmp_path):
    """The FB -> META rename splits ONE PERMNO membership into TWO symbol rows.

    PERMNO 13407 never changed; its ticker did, on 2022-06-09 (verbatim `C5`).
    A membership panel keyed on the OLD ticker would hold nothing after that
    date while the price panel happily carried a `META` column -- the silent
    empty-mask failure this whole plan exists to prevent.

    The boundary is asserted on BOTH sides of the seam, because an off-by-one
    here is exactly what a closed-interval convention gets wrong.
    """
    membership = _membership(
        _sp500_reference(
            tmp_path, spells=[_sp500_spell("13407", "2013-12-23", _PRODUCT_END)]
        )
    )

    intervals = membership.symbol_intervals(membership.SP500)

    assert _rows(intervals) == [
        ("FB", "2013-12-23", "2022-06-08"),
        ("META", "2022-06-09", "2025-12-31"),
    ]


def test_symbol_intervals_share_class_83443_keeps_the_brk_b_suffix(tmp_path):
    """Berkshire's B line is `BRK.B` in the panel, so it is `BRK.B` in the mask.

    PERMNO 83443's live `stksecurityinfohist` rows (verbatim `C5`) spell
    `BRK` + `BRKB` + class `B`, which D-04 rule 4 renders `BRK.B`. A mask
    naming the bare `BRK` would select the A line -- a DIFFERENT security -- or
    nothing at all.
    """
    membership = _membership(
        _sp500_reference(
            tmp_path, spells=[_sp500_spell("83443", "2010-02-16", _PRODUCT_END)]
        )
    )

    intervals = membership.symbol_intervals(membership.SP500)

    assert _rows(intervals) == [("BRK.B", "2010-02-16", "2025-12-31")]


def test_symbol_intervals_googl_and_goog_survive_the_permno_handover(tmp_path):
    """A ticker held by two PERMNOs in turn is ONE continuous membership.

    Alphabet's Nasdaq-100 spells (verbatim `L8_2`) link to 90319 (iid 01) and
    14542 (iid 03) (verbatim `L7_4`). 90319 carried `GOOG` until 2014-04-02 and
    `GOOGL` after; 14542 took `GOOG` over on 2014-04-03. On the SYMBOL axis
    those two PERMNO halves touch, so `GOOG` is a single uninterrupted
    membership -- the PERMNO switch is a seam the PRICE panel marks (D-18), not
    a hole in the universe.
    """
    membership = _membership(_ndx_reference(tmp_path))

    intervals = membership.symbol_intervals(membership.NASDAQ100)

    assert _rows(intervals) == [
        ("GOOG", "2005-12-21", "2025-12-31"),
        ("GOOGL", "2014-04-03", "2025-12-31"),
    ]


def test_symbol_intervals_unlabelled_membership_is_dropped_and_reported(tmp_path):
    """Membership days with no symbol are dropped LOUDLY, never silently.

    PERMNO 13407's symbol history starts 2012-05-18; a membership spell opened
    in 2000 therefore has twelve years no ticker covers. Those days cannot go
    into a symbol-keyed panel at all -- there is no column for them -- but
    dropping them without a trace would read downstream as "not a member",
    which is a different and unfalsifiable claim.
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        membership = _membership(
            _sp500_reference(
                tmp_path,
                spells=[_sp500_spell("13407", "2000-01-03", _PRODUCT_END)],
            )
        )
        intervals = membership.symbol_intervals(membership.SP500)
    finally:
        logger.remove(sink_id)

    # The labelled part survives untouched; only the uncovered head is gone.
    assert _rows(intervals) == [
        ("FB", "2012-05-18", "2022-06-08"),
        ("META", "2022-06-09", "2025-12-31"),
    ]
    assert membership.report["unlabelled_members"] == [
        {"permno": 13407, "start": "2000-01-03", "end": "2012-05-17"}
    ]
    assert any("unlabelled" in message for message in messages), messages


def test_symbol_intervals_every_end_is_explicit_and_within_the_product_end(tmp_path):
    """No null end, no end past the CRSP product end, in EITHER universe.

    `IndexConstituentDataset._densify` extends an interval with a null
    `end_date` to WALL-CLOCK today. A single null here would therefore push a
    CRSP universe months past the CRSP price coverage that bounds it, and the
    resulting mask would be True over a region where no price exists at all
    (T-03.10-30).
    """
    product_end = date.fromisoformat(_PRODUCT_END)

    sp500 = _membership(_sp500_reference(tmp_path / "sp500"))
    ndx = _membership(_ndx_reference(tmp_path / "ndx"))

    for membership, index in ((sp500, sp500.SP500), (ndx, ndx.NASDAQ100)):
        intervals = membership.symbol_intervals(index)
        assert intervals.height > 0
        ends = intervals["end_date"].to_list()
        assert all(end is not None for end in ends), (index, ends)
        assert max(ends) <= product_end, (index, max(ends))
        starts = intervals["start_date"].to_list()
        assert all(
            start <= end for start, end in zip(starts, ends)
        ), (index, list(zip(starts, ends)))


# ---------------------------------------------------------------------------
# T-03.10-31 -- the mask's symbols ARE the price panel's symbols
# ---------------------------------------------------------------------------

#: PERMNO 13407 around the rename: two trading days as FB, two as META.
_FB_DAYS = ("2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10")
#: PERMNO 83443, two 2020 trading days, well inside its membership.
_BRK_DAYS = ("2020-01-02", "2020-01-03")


def test_membership_symbols_agree_with_the_crsp_price_panel(
    mock_crsp_session, tmp_path
):
    """The whole point, end to end: one reference tier, one symbology rule.

    A real pull into the raw tier, a real conversion into a Zarr panel, and the
    membership intervals resolved from the SAME reference directory. The two
    sides are compared as sets of `(date, symbol)` pairs per PERMNO, so a rename
    handled on one side only, a class suffix on one side only, or a symbol the
    mask names that the panel never produced all fail here.

    Restricting the comparison to the days each PERMNO actually has a price row
    is the honest predicate: the membership intervals are CALENDAR-day ranges
    while the panel's axis is trading days, so they cannot be compared over
    days the panel has no opinion about.
    """
    import numpy as np
    import xarray as xr

    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds_crsp import WrdsCrspDailyAcquisition
    from quantlab.base.config import CrspDatasetConfig
    from tests.crsp_fixtures import (
        FakeCrspSession,
        dsf_row,
        run_crsp_pull,
        write_reference_tables,
    )

    # SYNTHETIC daily rows: flat prices and zero returns, because this test is
    # about the SYMBOL axis and nothing else. The live arithmetic has its own
    # test (`tests/test_crsp_tracer.py`).
    FakeCrspSession.daily_rows = [
        dsf_row(
            13407,
            day,
            dlyprc="100.000000",
            dlyret="0.000000",
            ticker="FB" if day <= "2022-06-08" else "META",
        )
        for day in _FB_DAYS
    ] + [
        dsf_row(83443, day, dlyprc="200.000000", dlyret="0.000000", ticker="BRK")
        for day in _BRK_DAYS
    ]

    cfg, result = run_crsp_pull(
        tmp_path, ["13407", "83443"], start_date="2020-01-01", end_date="2022-06-30"
    )
    assert result.failures == {}, result.failures

    reference_dir = WrdsCrspDailyAcquisition.reference_dir_for(cfg)
    write_reference_tables(
        reference_dir,
        {
            "crsp_a_stock.stksecurityinfohist": _secinfo_rows(),
            "crsp_a_indexes.dsp500list_v2": list(_SP500_MEMBERSHIP_ROWS),
        },
    )

    dataset_config = CrspDatasetConfig(
        zarr_file_path=str(tmp_path / "crsp.zarr"),
        raw_data_dir_path=cfg.raw_data_dir_path,
        catalog_path=str(tmp_path / "catalog"),
        reference_dir=str(reference_dir),
        start_date="2020-01-01",
        end_date="2022-06-30",
    )
    registry.convert(
        WRDS_SOURCE, dataset_config, data_type="crsp_daily", granularity="year"
    )
    panel = xr.open_zarr(dataset_config.zarr_file_path)

    membership = _membership(reference_dir)
    intervals = membership.symbol_intervals(membership.SP500)

    panel_symbols = {str(value) for value in panel["symbol"].values}
    # Neither side may name a symbol the other does not have. This is the
    # assertion a rename handled on one side only fails first.
    assert {symbol for symbol, _, _ in _rows(intervals)} == panel_symbols

    days = [str(value)[:10] for value in panel["timestamp"].values]
    permnos = panel["permno"].values
    closes = panel["close"].values
    symbols = [str(value) for value in panel["symbol"].values]

    for permno, expected_days in ((13407, _FB_DAYS), (83443, _BRK_DAYS)):
        priced = {
            (day, symbol)
            for row, day in enumerate(days)
            for column, symbol in enumerate(symbols)
            if not np.isnan(closes[row][column])
            and int(permnos[row][column]) == permno
        }
        assert {day for day, _ in priced} == set(expected_days), (permno, priced)

        owned = {symbol for _, symbol in priced}
        member = {
            (day, symbol)
            for day in expected_days
            for symbol, start, end in _rows(intervals)
            if symbol in owned and start <= day <= end
        }
        assert priced == member, (permno, priced ^ member)
