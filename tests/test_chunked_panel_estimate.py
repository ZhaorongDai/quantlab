"""`estimate_chunked_panel` answers; `assert_chunked_panel_fits` refuses (D-10).

Today's `assert_chunked_panel_fits` raises INSIDE its per-chunk loop, so a caller
whose second window overflows never learns what the fifth window costs -- the
chunk list is truncated at the first refusal. That is exactly the case SC-4's
"ask what peak it predicts, without starting the conversion" exists for.

This module pins the split:

- `estimate_chunked_panel` completes the loop, returns EVERY window, and marks
  each one with `fits` and a per-window `remedy` (DATA-08).
- `assert_chunked_panel_fits` becomes the thin raising wrapper whose message is
  byte-identical to today's at the default `bars_per_day`, remedy sentence
  included (SC-5).
- Both carry a keyword-only `bars_per_day: int = 1` whose default is the
  ARITHMETIC IDENTITY, which is what makes the widening backward compatible
  (D-10 as amended 2026-09-11) -- asserted here rather than promised in a
  docstring.

Every catalog below is `quantlab.utils.cli._explicit_symbol_catalog`, which
overrides exactly one method (`estimate_dense_panel`) and touches no reference
table, so these tests stay offline and credential-free.

**The category must be a REGISTERED one.** `_explicit_symbol_catalog` does NOT
override `_validate_category`, and the chunked pair validates on its first line
(the whole-window pair never does, which is why `tests/test_volume_guard.py` can
pass a placeholder). The explicit catalog ignores the name anyway -- its
override returns the synthetic roster regardless -- so `us_all` is passed purely
to satisfy the validator.
"""

import datetime

import pytest

from quantlab.acquisition.universe import UniverseCatalog
from quantlab.base.chunking import TimeChunkPlanner
from quantlab.utils.cli import _explicit_symbol_catalog

#: The remedy sentence SC-5 preserves. Asserted as a SUBSTRING of both the
#: refusal and the per-chunk `remedy`, because "names a concrete narrowing that
#: would fit" is the property, not "raises".
REMEDY = "finer --chunk (year -> quarter -> month)"

#: A single-day window plans to exactly ONE chunk of exactly ONE trading day
#: (`round(1 * 252 / 365.25) == 1`), which is what lets the boundary tests below
#: drive `dense_bytes` with the symbol count alone at `num_variables=1,
#: bytes_per_value=1`. Sizing the budget test on one free variable is what makes
#: "exactly the budget" and "one byte above it" expressible at all.
_ONE_DAY = ("2024-06-03", "2024-06-03")


def _longhand_dense_bytes(
    symbols: int,
    window_start: str,
    window_end: str,
    num_variables: int,
    bytes_per_value: int,
    bars_per_day: int = 1,
) -> int:
    """The per-chunk arithmetic, written out in the test rather than read off
    the function under test.

    Asserting a function against its own output proves nothing. This repeats
    the 252/365.25 approximation and Python's `round()` (half-to-even) longhand,
    so a re-derivation that changed the tie-breaking or dropped the `max(..., 1)`
    floor would be caught.
    """
    window_days = (
        datetime.date.fromisoformat(window_end)
        - datetime.date.fromisoformat(window_start)
    ).days + 1
    trading_days = max(
        round(
            window_days
            * UniverseCatalog.TRADING_DAYS_PER_YEAR
            / UniverseCatalog.CALENDAR_DAYS_PER_YEAR
        ),
        1,
    )
    return symbols * trading_days * bars_per_day * num_variables * bytes_per_value


# ---------------------------------------------------------------------------
# The loop completes


def test_a_window_under_budget_marks_every_chunk_as_fitting():
    """The admitted case: every chunk carries `fits` True and no remedy, and
    the wrapper returns the same dict rather than raising."""
    catalog = _explicit_symbol_catalog(500)
    report = catalog.estimate_chunked_panel("us_all", "2024-01-01", "2025-12-31")

    assert len(report["chunks"]) == 2
    assert all(chunk["fits"] for chunk in report["chunks"])
    assert all(chunk["remedy"] is None for chunk in report["chunks"])
    assert report["max_chunk_bytes"] <= UniverseCatalog.MAX_DENSE_PANEL_BYTES

    wrapped = catalog.assert_chunked_panel_fits("us_all", "2024-01-01", "2025-12-31")
    assert wrapped == report


def test_the_estimator_completes_the_loop_when_the_second_of_five_overflows():
    """THE defect this split exists to fix.

    `2020-12-31..2024-12-31` at year granularity plans five windows, the first
    of which is a single day. At 20,000,000 pinned symbols with one byte per
    cell, that one-day window fits and all four full years do not -- so the
    SECOND chunk is the first refusal. Today's implementation returns ONE chunk
    and raises; a caller asking "what does the whole conversion cost" is told
    about 20 MB and nothing else.

    Asserted on the LENGTH as well as the verdicts, because "returns one" is
    exactly today's behaviour and would otherwise pass silently.
    """
    catalog = _explicit_symbol_catalog(20_000_000)
    report = catalog.estimate_chunked_panel(
        "us_all",
        "2020-12-31",
        "2024-12-31",
        num_variables=1,
        bytes_per_value=1,
    )

    assert len(report["chunks"]) == 5, [c["start"] for c in report["chunks"]]
    assert report["chunks"][0]["fits"] is True
    assert report["chunks"][0]["remedy"] is None
    assert [c["fits"] for c in report["chunks"]] == [True, False, False, False, False]
    assert REMEDY in report["chunks"][1]["remedy"]
    # The largest window is still identified across the WHOLE list, not across
    # the truncated prefix today's loop would have built.
    assert report["max_chunk"]["start"] == "2024-01-01"

    with pytest.raises(ValueError, match="Refusing to densify"):
        catalog.assert_chunked_panel_fits(
            "us_all",
            "2020-12-31",
            "2024-12-31",
            num_variables=1,
            bytes_per_value=1,
        )


# ---------------------------------------------------------------------------
# The budget boundary


def test_the_chunk_verdict_flips_exactly_at_the_budget():
    """A chunk sized to exactly `MAX_DENSE_PANEL_BYTES` fits; one byte above it
    does not.

    Both the `fits` FLAG and the wrapper's raise/no-raise are asserted for each
    side, so a flag that drifts from the wrapper's own decision is caught -- the
    whole point of the split is that the two agree.
    """
    budget = UniverseCatalog.MAX_DENSE_PANEL_BYTES

    at_budget = _explicit_symbol_catalog(budget)
    report = at_budget.estimate_chunked_panel(
        "us_all", *_ONE_DAY, num_variables=1, bytes_per_value=1
    )
    assert report["chunks"][0]["dense_bytes"] == budget
    assert report["chunks"][0]["fits"] is True
    assert report["chunks"][0]["remedy"] is None
    assert (
        at_budget.assert_chunked_panel_fits(
            "us_all", *_ONE_DAY, num_variables=1, bytes_per_value=1
        )
        == report
    )

    over = _explicit_symbol_catalog(budget + 1)
    over_report = over.estimate_chunked_panel(
        "us_all", *_ONE_DAY, num_variables=1, bytes_per_value=1
    )
    assert over_report["chunks"][0]["dense_bytes"] == budget + 1
    assert over_report["chunks"][0]["fits"] is False
    assert REMEDY in over_report["chunks"][0]["remedy"]
    with pytest.raises(ValueError, match="Refusing to densify"):
        over.assert_chunked_panel_fits(
            "us_all", *_ONE_DAY, num_variables=1, bytes_per_value=1
        )


# ---------------------------------------------------------------------------
# The arithmetic is MOVED, never re-derived


def test_each_chunks_dense_bytes_matches_the_longhand_arithmetic():
    """Precision. The per-chunk arithmetic moved out of the loop unchanged --
    including the `max(round(...), 1)` floor and Python's half-to-even
    `round()`.

    Note no reachable window produces an exact `.5` tie: `window_days * 252 /
    365.25` never lands on a half within the 4,000-day range `plan_calendar`
    can be driven over (the closest, 1,640 days, is 1131.4990). The tie-breaking
    is therefore inherited BY CONSTRUCTION -- `_longhand_dense_bytes` calls the
    same `round()` -- rather than exercised by a hand-picked window.
    """
    catalog = _explicit_symbol_catalog(1_234)
    report = catalog.estimate_chunked_panel(
        "us_all",
        "2020-12-31",
        "2024-12-31",
        num_variables=7,
        bytes_per_value=8,
    )

    for chunk in report["chunks"]:
        assert chunk["dense_bytes"] == _longhand_dense_bytes(
            1_234, chunk["start"], chunk["end"], 7, 8
        ), chunk
        assert chunk["symbols"] == 1_234
        assert chunk["dense_cells"] * 7 * 8 == chunk["dense_bytes"]


def test_adjacent_chunk_windows_touch_without_overlapping():
    """No calendar day is sized twice. Asserted as ISO strings, which compare
    lexicographically exactly as dates in this table always do."""
    catalog = _explicit_symbol_catalog(10)
    report = catalog.estimate_chunked_panel(
        "us_all", "2022-11-15", "2024-02-10", granularity="month"
    )

    assert len(report["chunks"]) > 2
    for earlier, later in zip(report["chunks"], report["chunks"][1:]):
        assert earlier["end"] < later["start"], (earlier, later)


# ---------------------------------------------------------------------------
# Degenerate and tied windows


def test_a_window_that_plans_no_chunks_reports_nothing_rather_than_raising(
    monkeypatch,
):
    """The empty report is a VALUE, not an exception.

    `plan_calendar` cannot be driven to zero windows from valid input -- it
    raises when `end_date` precedes `start_date` and otherwise always yields at
    least one day -- so the empty planner result is injected rather than
    reached. The branch still has to exist: `max()` over an empty sequence
    raises, and `print_chunk_report` indexes `max_chunk`.
    """
    monkeypatch.setattr(
        TimeChunkPlanner, "plan_calendar", lambda self, start, end: []
    )
    catalog = _explicit_symbol_catalog(500)
    report = catalog.estimate_chunked_panel("us_all", "2024-01-01", "2024-12-31")

    assert report["chunks"] == []
    assert report["max_chunk"] is None
    assert report["max_chunk_bytes"] == 0
    assert (
        catalog.assert_chunked_panel_fits("us_all", "2024-01-01", "2024-12-31")
        == report
    )


def test_tied_chunks_resolve_to_the_earliest_window():
    """January and March 2023 are both 31 calendar days, so both size to 21
    trading days and tie exactly on `dense_bytes`. `max_chunk` must be the
    EARLIER of the two, and `chunks` must stay in ascending window order -- a
    specified, stable tie-break rather than whatever `max()` happened to
    return."""
    catalog = _explicit_symbol_catalog(500)
    report = catalog.estimate_chunked_panel(
        "us_all", "2023-01-01", "2023-03-31", granularity="month"
    )

    january, february, march = report["chunks"]
    assert [january["start"], february["start"], march["start"]] == [
        "2023-01-01",
        "2023-02-01",
        "2023-03-01",
    ]
    assert january["dense_bytes"] == march["dense_bytes"] > february["dense_bytes"]
    assert report["max_chunk"]["start"] == "2023-01-01"
    assert report["max_chunk_bytes"] == january["dense_bytes"]


def test_the_refusal_still_names_the_finer_chunk_remedy():
    """SC-5 is PRESERVED across the split, not rebuilt. The refusal keeps every
    factor of today's arithmetic and the remedy sentence character for
    character."""
    catalog = _explicit_symbol_catalog(UniverseCatalog.MAX_DENSE_PANEL_BYTES + 1)

    with pytest.raises(ValueError) as excinfo:
        catalog.assert_chunked_panel_fits(
            "us_all", *_ONE_DAY, num_variables=1, bytes_per_value=1
        )

    message = str(excinfo.value)
    assert "Refusing to densify us_all in year chunks" in message
    assert f"the window {_ONE_DAY[0]}..{_ONE_DAY[1]} alone is" in message
    assert "pinned symbol(s) x 1 trading days x 1 variables" in message
    assert REMEDY in message
    # At the default, the message does NOT grow a rows-per-day factor: the
    # daily refusal every existing caller reads is unchanged byte for byte.
    assert "row(s)/day" not in message


# ---------------------------------------------------------------------------
# D-10 as amended 2026-09-11 -- the timestamp axis, not the trading day


def test_an_intraday_window_sizes_390x_the_same_daily_window():
    """T-03.5-22's mitigation, asserted on ARITHMETIC rather than on a
    parameter existing.

    At `1m` a session is 390 rows, so a minute window is 390x the dense grid of
    the same window at `1d`. Sized on the trading-day count instead, the chunked
    guard would admit a conversion three orders of magnitude over the budget
    while reporting a number that looks fine. Compared against the DEFAULT
    call's own numbers, never against a constant typed into this test -- a
    hand-copied expected value passes when both sides are wrong.
    """
    catalog = _explicit_symbol_catalog(500)
    window = ("2024-01-01", "2025-12-31")

    daily = catalog.estimate_chunked_panel("us_all", *window)
    minute = catalog.estimate_chunked_panel("us_all", *window, bars_per_day=390)

    assert minute["bars_per_day"] == 390
    assert daily["bars_per_day"] == 1
    assert (
        minute["advisory"]["dense_bytes"] == daily["advisory"]["dense_bytes"] * 390
    )
    assert [c["dense_bytes"] * 390 for c in daily["chunks"]] == [
        c["dense_bytes"] for c in minute["chunks"]
    ]
    assert [c["start"] for c in daily["chunks"]] == [
        c["start"] for c in minute["chunks"]
    ]


def test_the_default_bars_per_day_is_the_arithmetic_identity():
    """The claim that makes the widening backward compatible, and the one claim
    that would be easy to state in a docstring and never check.

    A timestamp axis of one row per trading day is exactly what every existing
    caller computes today, so the defaulted call and the explicit
    `bars_per_day=1` call must return EQUAL dicts -- which is why
    `tests/test_volume_guard.py` needs no edit at all.
    """
    catalog = _explicit_symbol_catalog(500)
    window = ("2024-01-01", "2025-12-31")

    assert catalog.estimate_chunked_panel(
        "us_all", *window
    ) == catalog.estimate_chunked_panel("us_all", *window, bars_per_day=1)
    assert catalog.assert_chunked_panel_fits(
        "us_all", *window
    ) == catalog.assert_chunked_panel_fits("us_all", *window, bars_per_day=1)


def test_the_wrapper_forwards_bars_per_day_to_the_refusal():
    """A parameter that reached the ESTIMATE but not the REFUSAL is the failure
    mode this catches: the guard would compute the right number and then decline
    to act on it."""
    catalog = _explicit_symbol_catalog(500)
    window = ("2024-01-01", "2025-12-31")

    assert catalog.assert_chunked_panel_fits("us_all", *window)["chunks"]

    with pytest.raises(ValueError) as excinfo:
        catalog.assert_chunked_panel_fits("us_all", *window, bars_per_day=390)

    message = str(excinfo.value)
    assert "x 390 row(s)/day" in message
    assert REMEDY in message


def test_bars_per_day_is_keyword_only_on_both_chunked_guards():
    """Keyword-only is what makes it impossible for an existing POSITIONAL call
    to bind the new parameter by accident -- the mechanism behind the
    backward-compatibility claim, not a style choice."""
    import inspect

    for method in (
        UniverseCatalog.estimate_chunked_panel,
        UniverseCatalog.assert_chunked_panel_fits,
    ):
        parameter = inspect.signature(method).parameters["bars_per_day"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, method
        assert parameter.default == 1, method

    with pytest.raises(ValueError, match="bars_per_day must be >= 1"):
        _explicit_symbol_catalog(10).estimate_chunked_panel(
            "us_all", *_ONE_DAY, bars_per_day=0
        )
