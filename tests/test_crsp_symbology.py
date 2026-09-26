"""PERMNO -> ticker intervals: what is LEFT of the CRSP symbology (D-04, D-18).

This file used to hold 21 tests, and 16 of them tested machinery that 03.11-07
deleted: the share-class collision pass, `resolve_collisions`, the delisting
SYMBOL carry, the `symbol_overrides` pin, the nonconforming-symbol report. None
of that is missing coverage -- the PERMNO axis (D-01) made those mechanisms
unreachable, not untested. Two securities can no longer land in one column, so
there is no tie to break and no suffix to invent.

What survives is the one thing a PERMNO-keyed panel still cannot say for
itself: which HUMAN-READABLE name a security wore on a given date. The two
tests below are that contract -- the interval table's schema and a rename -- and
they are the schema the plan-09 ticker sidecar is built on.

Every naming case is a LIVE CRSP row. The provenance rule of
`tests/crsp_fixtures.py` holds here too: a value transcribed from a live check
names the JSON key it came from, and a value invented for the test carries a
`# SYNTHETIC` comment on the spot. The live source used here is
`03.10-LIVE-CHECK.json` key `C5_ticker_hist_crsp_a_stock.stksecurityinfohist`
-- FB -> META (13407).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from quantlab.dataset.crsp.reference import REFERENCE_TABLES_BY_NAME
from quantlab.dataset.crsp.symbology import CrspSymbology
from tests.crsp_fixtures import SECINFO_ROWS

_SECINFO_SPEC = REFERENCE_TABLES_BY_NAME["stksecurityinfohist"]

#: Every interval the naming tests run against.
ALL_SECINFO_ROWS = list(SECINFO_ROWS)


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


def _symbology(rows=None) -> CrspSymbology:
    return CrspSymbology(
        _secinfo_frame(ALL_SECINFO_ROWS if rows is None else rows)
    )


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


# ---------------------------------------------------------------------------
# The interval table -- the ticker sidecar's schema
# ---------------------------------------------------------------------------


def test_intervals_are_sorted_typed_and_one_row_per_input_interval():
    """The four columns, their dtypes and the sort order plan 09's sidecar
    reads. One row IN, one row OUT: nothing merges intervals any more."""
    intervals = _symbology().symbol_intervals()

    assert intervals.columns == ["permno", "symbol", "start_date", "end_date"]
    assert intervals.schema["permno"] == pl.Int64
    assert intervals.schema["symbol"] == pl.String
    assert intervals.schema["start_date"] == pl.Date
    assert intervals.schema["end_date"] == pl.Date
    assert intervals.height == len(ALL_SECINFO_ROWS)
    assert intervals.equals(intervals.sort(["permno", "start_date"]))


def test_rename_13407_is_fb_through_2022_06_08_and_meta_after():
    """VERBATIM C5: one PERMNO, two names, a hard boundary between them.

    On the PERMNO axis this is no longer about which COLUMN the rows land in
    -- 13407 is one column on both sides of 2022-06-09. It is about which name
    a reader of the sidecar is told the security wore that day, and the
    boundary has to be exact for that answer to be worth anything.
    """
    intervals = _symbology().symbol_intervals()

    assert _symbol_on(intervals, 13407, "2012-05-18") == "FB"
    assert _symbol_on(intervals, 13407, "2022-06-08") == "FB"
    assert _symbol_on(intervals, 13407, "2022-06-09") == "META"
    assert _symbol_on(intervals, 13407, "2025-12-31") == "META"

    fb = intervals.filter((pl.col("permno") == 13407) & (pl.col("symbol") == "FB"))
    assert fb["end_date"].max() == date(2022, 6, 8)


# ---------------------------------------------------------------------------
# The reverse mapping and the sidecar payload, shared with the NBBO panel.
# ---------------------------------------------------------------------------


def test_resolve_maps_each_date_ticker_pair_to_the_permno_that_used_it():
    """FB and META are one PERMNO; BRK.B is spelled in dot notation; an
    unknown ticker is left out rather than raising."""
    pairs = pl.DataFrame(
        {
            "date": [date(2022, 6, 8), date(2022, 6, 9), date(2024, 1, 24), date(2024, 1, 24)],
            "symbol": ["FB", "META", "BRK.B", "ZZZZ"],
        }
    )
    resolved = _symbology().resolve(pairs)
    assert resolved.schema == {"date": pl.Date, "symbol": pl.String, "permno": pl.Int64}
    assert resolved.to_dicts() == [
        {"date": date(2022, 6, 8), "symbol": "FB", "permno": 13407},
        {"date": date(2022, 6, 9), "symbol": "META", "permno": 13407},
        {"date": date(2024, 1, 24), "symbol": "BRK.B", "permno": 83443},
    ]


def test_resolve_leaves_out_a_ticker_outside_every_interval_of_its_permno():
    """META before 2022-06-09 names nothing in the fixture table, and FB after
    2022-06-08 names nothing either: a ticker is only a name for its dates."""
    pairs = pl.DataFrame(
        {"date": [date(2022, 6, 8), date(2022, 6, 9)], "symbol": ["META", "FB"]}
    )
    assert _symbology().resolve(pairs).is_empty()


def test_resolve_refuses_a_pair_two_permnos_claim():
    """Two PERMNOs naming the same ticker on the same day is a reference-table
    fault, not a choice to make silently."""
    import pytest
    from tests.crsp_fixtures import secinfo_row

    rows = list(SECINFO_ROWS) + [
        secinfo_row(99999, "2024-01-01", "2024-12-31", "AAPL", "AAPL", None),
    ]
    pairs = pl.DataFrame({"date": [date(2024, 1, 24)], "symbol": ["AAPL"]})
    with pytest.raises(ValueError, match="more than one PERMNO"):
        _symbology(rows).resolve(pairs)


def test_sidecar_payload_holds_only_the_requested_permnos_named_intervals():
    payload = _symbology().sidecar_payload([83443, 80599, 7000], date(2025, 12, 31))
    assert payload["generated_from"] == "stksecurityinfohist"
    assert payload["vintage_product_end"] == "2025-12-31"
    assert payload["intervals"]["83443"] == [
        {"ticker": "BRK", "start": "1996-05-09", "end": "2002-01-01"},
        {"ticker": "BRK.B", "start": "2002-01-02", "end": "2025-12-31"},
    ]
    # Lehman's delisting-day row carries LEH forward, so every interval is named.
    assert [span["ticker"] for span in payload["intervals"]["80599"]] == ["LEH"] * 7
    # 7000 never had a ticker: no entry rather than a nameless span.
    assert "7000" not in payload["intervals"]
    assert "14593" not in payload["intervals"]
    assert CrspSymbology.empty_sidecar_payload("2025-12-31T00:00:00") == {
        "generated_from": "stksecurityinfohist",
        "vintage_product_end": "2025-12-31",
        "intervals": {},
    }
