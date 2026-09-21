"""Tests for dataset/masking.py:UniverseMask (260906-13w Task 3, D-06/D-07).

Aligning an index `is_member` panel against a market panel is an intersection
in both directions, but the two directions are NOT symmetric and the asymmetry
is the whole point of this component:

- an index member the market data does not cover is a COVERAGE GAP. Dropping
  it silently reintroduces survivorship bias through exactly the
  hardest-to-obtain names -- long-delisted tickers -- so it is reported by
  count AND by name, never truncated, never sampled (D-06). That report is
  also the only mechanism by which D-07's assumption ("Tiingo serves delisted
  history") is falsifiable, which is why an EMPTY report is a result rather
  than a no-op.
- a market symbol that is not an index member is simply out of universe. It is
  dropped unreported, because it is not a gap in anything.

Timestamps are a third case: the membership panel is on a CALENDAR-day axis
(see base/constituent.py's class docstring) while market data is on trading
days, so the weekend and holiday rows an inner join drops are EXPECTED. They
produce no report of any kind (D-06).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.dataset.masking import UniverseMask

#: Five trading days (Mon..Fri) for the market panel.
_MARKET_DAYS = pd.to_datetime(
    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
)
#: Seven CALENDAR days for the membership panel -- the two extra are the
#: weekend rows the inner join must drop silently.
_CALENDAR_DAYS = pd.date_range("2024-01-01", "2024-01-07", freq="D")


def _membership(include_gone: bool = True) -> xr.Dataset:
    """`AAA` always a member; `BBB` a member except on the 4th; `GONE` a
    member for two days inside the window but absent from market data;
    `NEVER` carried on the axis but never True (the survivorship-bias column
    the panel deliberately keeps).
    """
    symbols = ["AAA", "BBB", "GONE", "NEVER"] if include_gone else ["AAA", "BBB", "NEVER"]
    values = np.zeros((len(_CALENDAR_DAYS), len(symbols)), dtype=bool)
    index = {symbol: i for i, symbol in enumerate(symbols)}
    values[:, index["AAA"]] = True
    values[:, index["BBB"]] = True
    values[_CALENDAR_DAYS == pd.Timestamp("2024-01-04"), index["BBB"]] = False
    if include_gone:
        window = (_CALENDAR_DAYS >= pd.Timestamp("2024-01-02")) & (
            _CALENDAR_DAYS <= pd.Timestamp("2024-01-03")
        )
        values[window, index["GONE"]] = True
    return xr.Dataset(
        {"is_member": (["timestamp", "symbol"], values)},
        coords={"timestamp": _CALENDAR_DAYS, "symbol": symbols},
    )


def _market() -> xr.Dataset:
    """`EXTRA` is a real traded symbol that is not an index member -- out of
    universe, not a coverage gap, so it must be dropped WITHOUT a report.
    """
    symbols = ["AAA", "BBB", "EXTRA"]
    close = np.arange(
        len(_MARKET_DAYS) * len(symbols), dtype=float
    ).reshape(len(_MARKET_DAYS), len(symbols)) + 100.0
    return xr.Dataset(
        {"close": (["timestamp", "symbol"], close)},
        coords={"timestamp": _MARKET_DAYS, "symbol": symbols},
    )


def _captured(level: str = "WARNING"):
    """Attach a temporary in-memory loguru sink (loguru does not propagate to
    stdlib `logging`, so `caplog` sees nothing).
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level=level, format="{message}")
    return messages, sink_id


def test_a_member_absent_from_market_data_is_reported_by_count_and_by_name() -> None:
    mask = UniverseMask(_market(), _membership())

    messages, sink_id = _captured()
    try:
        report = mask.report()
    finally:
        logger.remove(sink_id)

    assert report["missing_count"] == 1
    assert report["missing_symbols"] == ["GONE"]
    assert messages, "a coverage gap must be announced, not merely returned"
    joined = "\n".join(messages)
    assert "GONE" in joined
    assert "1" in joined


def test_the_missing_symbol_list_is_never_truncated() -> None:
    """A truncated list is worse than none: it looks like a complete answer.
    Twenty missing members must produce twenty names.
    """
    symbols = [f"DEAD{i:02d}" for i in range(20)]
    values = np.ones((len(_CALENDAR_DAYS), len(symbols)), dtype=bool)
    membership = xr.Dataset(
        {"is_member": (["timestamp", "symbol"], values)},
        coords={"timestamp": _CALENDAR_DAYS, "symbol": symbols},
    )
    market = xr.Dataset(
        {"close": (["timestamp", "symbol"], np.ones((len(_MARKET_DAYS), 1)))},
        coords={"timestamp": _MARKET_DAYS, "symbol": ["DEAD00"]},
    )

    messages, sink_id = _captured()
    try:
        report = UniverseMask(market, membership).report()
    finally:
        logger.remove(sink_id)

    assert report["missing_count"] == 19
    assert report["missing_symbols"] == sorted(symbols[1:])
    joined = "\n".join(messages)
    for symbol in symbols[1:]:
        assert symbol in joined
    assert "..." not in joined


def test_a_column_that_is_never_a_member_is_not_a_coverage_gap() -> None:
    """`NEVER` sits on the membership axis for survivorship-bias reasons and
    is False at every timestamp. It is absent from market data too, but it is
    not a missing MEMBER and must not be reported.
    """
    report = UniverseMask(_market(), _membership()).report()

    assert "NEVER" not in report["missing_symbols"]


def test_an_empty_report_costs_nothing_and_emits_no_warning() -> None:
    """The empty report is the falsification of D-07's assumption -- it has to
    be reachable, and it has to be quiet.
    """
    mask = UniverseMask(_market(), _membership(include_gone=False))

    warnings, warning_sink = _captured("WARNING")
    infos, info_sink = _captured("INFO")
    try:
        report = mask.report()
    finally:
        logger.remove(warning_sink)
        logger.remove(info_sink)

    assert report["missing_count"] == 0
    assert report["missing_symbols"] == []
    assert warnings == []
    assert any("0" in message for message in infos), infos


def test_timestamps_are_inner_joined_and_deliberately_not_reported() -> None:
    """The membership panel is on a calendar-day axis and market data is on
    trading days, so dropped weekend rows are expected, not a defect. Nothing
    about them belongs in the report (D-06).
    """
    mask = UniverseMask(_market(), _membership())

    assert list(mask.timestamps) == list(_MARKET_DAYS)

    messages, sink_id = _captured()
    try:
        report = mask.report()
    finally:
        logger.remove(sink_id)

    assert set(report) == {
        "in_window_members",
        "missing_count",
        "missing_symbols",
        "missing_labels",
    }
    joined = "\n".join(messages)
    assert "2024-01-06" not in joined
    assert "2024-01-07" not in joined


def test_a_market_symbol_that_is_not_a_member_is_dropped_unreported() -> None:
    mask = UniverseMask(_market(), _membership())

    messages, sink_id = _captured()
    try:
        result = mask.apply()
    finally:
        logger.remove(sink_id)

    assert mask.symbols == ["AAA", "BBB"]
    assert result["symbol"].values.tolist() == ["AAA", "BBB"]
    assert "EXTRA" not in "\n".join(messages)


def test_apply_keeps_member_cells_and_nans_non_member_cells() -> None:
    market = _market()
    result = UniverseMask(market, _membership()).apply()

    assert result["symbol"].values.tolist() == ["AAA", "BBB"]
    assert list(pd.DatetimeIndex(result["timestamp"].values)) == list(_MARKET_DAYS)

    expected = market["close"].sel(symbol=["AAA", "BBB"], timestamp=_MARKET_DAYS)
    actual = result["close"]

    # AAA is a member on every day -- values untouched.
    np.testing.assert_allclose(
        actual.sel(symbol="AAA").values, expected.sel(symbol="AAA").values
    )
    # BBB is a member on every day EXCEPT the 4th, which is NaN and nothing
    # else -- the surrounding days keep their original values.
    bbb = actual.sel(symbol="BBB")
    assert np.isnan(bbb.sel(timestamp="2024-01-04").item())
    for day in ("2024-01-02", "2024-01-03", "2024-01-05"):
        assert bbb.sel(timestamp=day).item() == expected.sel(
            symbol="BBB", timestamp=day
        ).item()


def test_apply_reports_before_it_masks() -> None:
    """`apply()` must not be a quiet way to skip the report."""
    messages, sink_id = _captured()
    try:
        UniverseMask(_market(), _membership()).apply()
    finally:
        logger.remove(sink_id)

    assert any("GONE" in message for message in messages), messages


def test_a_membership_panel_without_is_member_is_rejected() -> None:
    not_a_panel = xr.Dataset(
        {"close": (["timestamp", "symbol"], np.ones((len(_CALENDAR_DAYS), 1)))},
        coords={"timestamp": _CALENDAR_DAYS, "symbol": ["AAA"]},
    )

    with pytest.raises(ValueError, match="is_member"):
        UniverseMask(_market(), not_a_panel)


def test_a_disjoint_timestamp_axis_is_refused_rather_than_returning_empty() -> None:
    """An empty overlap is never a useful answer -- it is a misconfiguration
    (two panels from different eras), and returning an empty panel would let
    it flow silently into a backtest.
    """
    membership = _membership()
    membership = membership.assign_coords(
        timestamp=pd.date_range("2030-01-01", periods=len(_CALENDAR_DAYS), freq="D")
    )

    with pytest.raises(ValueError, match="overlap"):
        UniverseMask(_market(), membership).apply()


# ---------------------------------------------------------------------------
# 03.11-05 -- the same claims on an int64 PERMNO axis
# ---------------------------------------------------------------------------
#
# Since 03.11-03 a CRSP price panel's `symbol` is the int64 PERMNO, and since
# 03.11-05 so is a CRSP membership panel's. `UniverseMask` used to `str()`
# both sides in three separate places, which broke in two different ways at
# once: `apply()` raised `KeyError` selecting digit strings out of an integer
# index, and `missing_members` differenced a set of STRINGS against a set of
# INTS, so every in-window member read as missing -- the whole universe
# reported as a survivorship-bias hole (T-03.11-16). The three had to move
# together; repairing `symbols` alone would have left the second failure
# intact and SILENT.

#: 14593 is priced but is not an index member -- out of universe, not a gap.
#: 7000 and 10107 are the pair on which numeric and lexicographic order fork.
_PERMNO_MARKET = [7000, 10107, 14593]
#: 93436 is an in-window member the market panel does not carry (the gap);
#: 88801 sits on the axis for survivorship-bias reasons and is never True.
_PERMNO_MEMBERS = [7000, 10107, 93436, 88801]


def _permno_membership() -> xr.Dataset:
    values = np.zeros((len(_CALENDAR_DAYS), len(_PERMNO_MEMBERS)), dtype=bool)
    index = {permno: i for i, permno in enumerate(_PERMNO_MEMBERS)}
    values[:, index[7000]] = True
    values[:, index[10107]] = True
    window = (_CALENDAR_DAYS >= pd.Timestamp("2024-01-02")) & (
        _CALENDAR_DAYS <= pd.Timestamp("2024-01-03")
    )
    values[window, index[93436]] = True
    return xr.Dataset(
        {"is_member": (["timestamp", "symbol"], values)},
        coords={
            "timestamp": _CALENDAR_DAYS,
            "symbol": np.asarray(_PERMNO_MEMBERS, dtype="int64"),
        },
    )


def _permno_market() -> xr.Dataset:
    close = np.arange(
        len(_MARKET_DAYS) * len(_PERMNO_MARKET), dtype=float
    ).reshape(len(_MARKET_DAYS), len(_PERMNO_MARKET)) + 100.0
    return xr.Dataset(
        {"close": (["timestamp", "symbol"], close)},
        coords={
            "timestamp": _MARKET_DAYS,
            "symbol": np.asarray(_PERMNO_MARKET, dtype="int64"),
        },
    )


def test_missing_members_is_not_the_whole_universe() -> None:
    """The failure T-03.11-16 names, asserted by its SHAPE rather than by a
    KeyError.

    With `str()` on one side of the set difference and integers on the other,
    `missing_members` is the ENTIRE in-window membership -- every name a
    survivorship-bias hole, a report that is loud, complete and completely
    wrong. Exactly ONE member here is genuinely absent from the market panel,
    and the third assertion spells out what the regression produced, so a
    reader can see how this test fails.
    """
    mask = UniverseMask(_permno_market(), _permno_membership())

    assert mask.missing_members == [93436]
    assert mask.in_window_members == [7000, 10107, 93436]
    assert mask.missing_members != mask.in_window_members


def test_the_intersected_symbol_axis_is_int64_and_numerically_ordered() -> None:
    """`symbols` takes the shape of its neighbour `timestamps`.

    Both are now plain index intersections that convert nothing. The order is
    numeric, and 7000 vs 10107 is where numeric and lexicographic fork -- on
    five-digit PERMNOs alone a regression to `sorted(str(...))` is invisible.
    """
    mask = UniverseMask(_permno_market(), _permno_membership())

    assert mask.symbols == [7000, 10107]
    assert mask.symbols != sorted(mask.symbols, key=str)
    assert all(isinstance(symbol, int) for symbol in mask.symbols), mask.symbols


def test_apply_masks_an_int64_panel_without_a_keyerror() -> None:
    """`.sel(symbol=[...])` against an integer index, end to end.

    The pre-03.11-05 `symbols` handed `['7000', '10107']` to `.sel`, which
    raised `KeyError: "not all values found in index 'symbol'"` (measured
    2026-09-20). The surviving cells are asserted too, so a version that
    "worked" by quietly returning an all-NaN panel would not pass.
    """
    result = UniverseMask(_permno_market(), _permno_membership()).apply()

    assert result["symbol"].values.tolist() == [7000, 10107]
    assert result["symbol"].dtype.kind == "i", result["symbol"].dtype
    assert not np.isnan(result["close"].sel(symbol=7000).values).any()


def test_the_complete_missing_list_survives_the_int64_axis() -> None:
    """`report()`'s never-truncated, never-sampled promise is not weakened.

    PERMNOs read worse than tickers, and truncating the list is the obvious
    way to make the log tidier. It is also the one thing that docstring
    forbids: a truncated list looks like a complete answer. 03.11-09 restored
    the readable labels -- see the test below -- which is the right way round:
    name them all, do not print fewer.
    """
    permnos = list(range(80000, 80020))
    values = np.ones((len(_CALENDAR_DAYS), len(permnos)), dtype=bool)
    membership = xr.Dataset(
        {"is_member": (["timestamp", "symbol"], values)},
        coords={
            "timestamp": _CALENDAR_DAYS,
            "symbol": np.asarray(permnos, dtype="int64"),
        },
    )
    market = xr.Dataset(
        {"close": (["timestamp", "symbol"], np.ones((len(_MARKET_DAYS), 1)))},
        coords={
            "timestamp": _MARKET_DAYS,
            "symbol": np.asarray([80000], dtype="int64"),
        },
    )

    messages, sink_id = _captured()
    try:
        report = UniverseMask(market, membership).report()
    finally:
        logger.remove(sink_id)

    assert report["missing_count"] == 19
    assert report["missing_symbols"] == permnos[1:]
    # No lookup was attached (this mask was built from two bare panels), so
    # the display half falls back to the digits -- same length, same order,
    # nothing dropped. That fallback is the CONTRACT, not a degradation: the
    # coverage report must not become breakable by a missing audit file
    # (T-03.11-30, 03.11-09).
    assert report["missing_labels"] == [str(permno) for permno in permnos[1:]]
    joined = "\n".join(messages)
    for permno in permnos[1:]:
        assert str(permno) in joined
    assert "..." not in joined


def test_a_ticker_sidecar_spells_the_missing_permnos_without_shortening_them(
    tmp_path,
) -> None:
    """03.11-09: the digits get NAMES, and the list stays complete.

    `missing_symbols` keeps the int64 identities -- a caller that wants to
    `.sel()` them needs the number, and a name is only true as of a day.
    `missing_labels` is the same list, same order, spelled for a human, with an
    un-named PERMNO keeping its digits rather than vanishing.
    """
    import json

    from quantlab.dataset.crsp_tickers import CrspTickerLookup

    sidecar = tmp_path / "crsp.zarr.crsp_tickers.json"
    sidecar.write_text(
        json.dumps(
            {
                "generated_from": "stksecurityinfohist",
                "vintage_product_end": "2025-12-31",
                "intervals": {
                    "80001": [
                        {"ticker": "OLD", "start": "1990-01-01",
                         "end": "2020-12-31"},
                        {"ticker": "NEW", "start": "2021-01-01",
                         "end": "2025-12-31"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    permnos = [80000, 80001, 80002]
    membership = xr.Dataset(
        {
            "is_member": (
                ["timestamp", "symbol"],
                np.ones((len(_CALENDAR_DAYS), len(permnos)), dtype=bool),
            )
        },
        coords={
            "timestamp": _CALENDAR_DAYS,
            "symbol": np.asarray(permnos, dtype="int64"),
        },
    )
    market = xr.Dataset(
        {"close": (["timestamp", "symbol"], np.ones((len(_MARKET_DAYS), 1)))},
        coords={
            "timestamp": _MARKET_DAYS,
            "symbol": np.asarray([80000], dtype="int64"),
        },
    )

    report = UniverseMask(
        market, membership, ticker_lookup=CrspTickerLookup(sidecar)
    ).report()

    assert report["missing_symbols"] == [80001, 80002]
    # The window is in 2024, so 80001 is NEW -- not OLD, the name it wore
    # until 2020. 80002 is not in the sidecar and keeps its digits.
    assert report["missing_labels"] == ["NEW", "80002"]
    assert len(report["missing_labels"]) == len(report["missing_symbols"])


def test_a_missing_ticker_sidecar_leaves_the_report_working(tmp_path) -> None:
    """T-03.11-30: a display layer must not be able to break the report.

    The mask is handed a lookup pointing at a file that does not exist, which
    is what every non-CRSP store produces. `report()` answers with the digits
    instead of raising.
    """
    from quantlab.dataset.crsp_tickers import CrspTickerLookup

    permnos = [80000, 80001]
    membership = xr.Dataset(
        {
            "is_member": (
                ["timestamp", "symbol"],
                np.ones((len(_CALENDAR_DAYS), len(permnos)), dtype=bool),
            )
        },
        coords={
            "timestamp": _CALENDAR_DAYS,
            "symbol": np.asarray(permnos, dtype="int64"),
        },
    )
    market = xr.Dataset(
        {"close": (["timestamp", "symbol"], np.ones((len(_MARKET_DAYS), 1)))},
        coords={
            "timestamp": _MARKET_DAYS,
            "symbol": np.asarray([80000], dtype="int64"),
        },
    )

    report = UniverseMask(
        market,
        membership,
        ticker_lookup=CrspTickerLookup(tmp_path / "absent.crsp_tickers.json"),
    ).report()

    assert report["missing_symbols"] == [80001]
    assert report["missing_labels"] == ["80001"]


def test_a_permno_panel_against_a_ticker_universe_is_still_refused() -> None:
    """A GENUINE axis mismatch must keep raising, not quietly align.

    This is a pre-existing good behaviour, and removing the `str()` calls is
    exactly the change that could have weakened it: a caller who "fixed" the
    mismatch by coercing one side onto the other's dtype would mask the right
    panel against the wrong universe, silently. The two panels genuinely
    disagree about what a security IS, and an empty overlap is never a useful
    answer -- it flows into a backtest as "no positions".
    """
    with pytest.raises(ValueError, match="overlap"):
        UniverseMask(_permno_market(), _membership()).apply()
