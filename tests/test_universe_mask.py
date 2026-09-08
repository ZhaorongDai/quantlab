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

    assert set(report) == {"in_window_members", "missing_count", "missing_symbols"}
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
