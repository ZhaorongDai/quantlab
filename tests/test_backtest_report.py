"""Leaf tests for `quantlab/utils/backtest_report.py` (quick task 260915-sxx).

`write_backtest_report` is a leaf: it takes arrays and already-formatted
strings and writes one self-contained HTML page. These tests call it directly
with small synthetic inputs -- no dataset, no model, no backtest run -- so the
page composition is covered in milliseconds instead of behind a two-minute
end-to-end run. The run-level locks (the page really carries the persisted
run's dates and numbers) live in tests/test_backtest_persistence.py.

T-sxx-01: every value the caller interpolates into the page is escaped. The
title and the summary values come from a run directory name, from operator
notes and from a metrics mapping, so a page that pasted them raw would turn
whatever produced them into markup.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.utils.backtest_report import write_backtest_report

N_BARS = 12
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)


def _value(n: int = N_BARS) -> xr.DataArray:
    """A rising-then-falling portfolio value, so drawdown is really negative."""
    path = np.concatenate(
        [np.linspace(1.0, 1.4, n // 2), np.linspace(1.4, 1.1, n - n // 2)]
    )
    return xr.DataArray(
        1_000_000.0 * path,
        dims=("timestamp",),
        coords={"timestamp": BARS[:n]},
    )


def _returns(n: int = N_BARS) -> xr.DataArray:
    """Per-bar returns of `_value`, all inside one calendar month."""
    values = _value(n).values
    returns = np.concatenate([[np.nan], values[1:] / values[:-1] - 1.0])
    return xr.DataArray(
        returns, dims=("timestamp",), coords={"timestamp": BARS[:n]}
    )


def _write(tmp_path, **kwargs) -> str:
    """Write a report with the mandatory arguments defaulted, return the page."""
    path = tmp_path / "report.html"
    arguments = dict(
        in_sample_range=None,
        notes=["a note"],
        title="Backtester_20240101_000000_000000",
    )
    arguments.update(kwargs)
    write_backtest_report(_value(), path, **arguments)
    return path.read_text(encoding="utf-8")


def test_the_original_five_arguments_still_write_a_page(tmp_path):
    """The widened signature keeps the old call form legal."""
    path = tmp_path / "report.html"
    write_backtest_report(
        _value(),
        path,
        in_sample_range=("2024-01-01", "2024-01-04"),
        notes=["short-side returns are optimistic"],
        title="a run",
    )

    html = path.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert '"name":"equity"' in html and '"name":"drawdown"' in html
    assert '"type":"rect"' in html, "the in-sample band must still be shaded"
    assert "short-side returns are optimistic" in html


def test_a_page_without_a_summary_is_still_valid(tmp_path):
    html = _write(tmp_path, summary=None)
    assert "Dates and setup" not in html
    assert '"name":"equity"' in html


def test_the_summary_labels_and_values_appear_on_the_page(tmp_path):
    summary = {
        "Backtest window": "2024-01-01 .. 2024-01-16 (12 bars)",
        "Bar interval": "1 days 00:00:00",
        "Training window": "2023-06-01 .. 2023-12-29",
        "In-sample range": "—",
    }
    html = _write(tmp_path, summary=summary)

    for label, text in summary.items():
        assert f"<th>{label}</th>" in html, label
        assert f"<td>{text}</td>" in html, text
    # The block keeps the caller's order: the page reads top-down as the
    # caller composed it, not in dict-sorted order.
    positions = [html.index(f"<th>{label}</th>") for label in summary]
    assert positions == sorted(positions)


def test_the_page_states_the_notes_as_text(tmp_path):
    notes = ["first note", "second note"]
    html = _write(tmp_path, notes=notes)
    for note in notes:
        assert f"<li>{note}</li>" in html


def test_angle_brackets_in_the_title_are_escaped(tmp_path):
    """T-sxx-01: a run directory name cannot inject markup.

    The title reaches the page only through the escaped `<h1>`; it is
    deliberately not handed to plotly's layout title, so no unescaped copy of
    it reaches the embedded JSON either. The raw text must be absent from the
    WHOLE page, not merely from the heading.
    """
    html = _write(tmp_path, title="run <img src=x onerror=alert(1)>")

    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<h1>run &lt;img src=x onerror=alert(1)&gt;</h1>" in html


def test_angle_brackets_in_a_summary_value_are_escaped(tmp_path):
    """T-sxx-01: a metrics value or a label cannot inject markup either."""
    html = _write(
        tmp_path,
        summary={"<b>label</b>": "<script>alert(1)</script>"},
    )

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<b>label</b>" not in html
    assert "&lt;b&gt;label&lt;/b&gt;" in html


def test_angle_brackets_in_a_note_are_escaped(tmp_path):
    html = _write(tmp_path, notes=["<i>borrow cost</i> is not modelled"])

    assert "<li>&lt;i&gt;borrow cost&lt;/i&gt; is not modelled</li>" in html


def test_the_report_is_one_self_contained_file_loading_plotly_from_the_cdn(tmp_path):
    """D-23: one file per run, whose only external reference is the CDN."""
    path = tmp_path / "report.html"
    write_backtest_report(
        _value(), path, in_sample_range=None, notes=[], title="a run"
    )

    assert [p.name for p in tmp_path.iterdir()] == ["report.html"]
    assert "cdn.plot.ly" in path.read_text(encoding="utf-8")


def test_the_module_is_a_leaf():
    """No `quantlab.*` import may appear in the report module."""
    import pathlib

    import quantlab.utils.backtest_report as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    offenders = [
        line
        for line in source.splitlines()
        if line.startswith(("from quantlab", "import quantlab"))
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# The metric table is generic: no metric name drives it
# ---------------------------------------------------------------------------
#
# These are the behavioural proof of the rule the module docstring states. The
# project's metric set is being replaced with vectorbt's own, and the report is
# written inside a run's staging directory -- an exception here deletes the
# whole run, not just the report. So "the table survives the metric set
# changing under it" is asserted by rendering mappings the code has never seen,
# not by reading the source for metric names.


def _cells(html: str) -> dict[str, list[str]]:
    """Every metric row as `name -> [cell, ...]`, parsed back off the page."""
    rows = re.findall(r"<tr><th>([^<]+)</th>((?:<td>[^<]*</td>)+)</tr>", html)
    return {name: re.findall(r"<td>([^<]*)</td>", cells) for name, cells in rows}


def test_an_unknown_metric_key_renders_and_raises_nothing(tmp_path):
    html = _write(
        tmp_path,
        metrics={
            "whole": {"a_metric_nobody_has_written_yet": 1.5},
            "in_sample": None,
            "out_of_sample": {"a_metric_nobody_has_written_yet": 2.5},
        },
    )

    cells = _cells(html)
    assert cells["a_metric_nobody_has_written_yet"] == ["1.5", "—", "2.5"]


def test_an_unseen_nested_sub_dict_is_flattened_to_dotted_paths(tmp_path):
    html = _write(
        tmp_path,
        metrics={
            "whole": {"group": {"leaf": 3.0, "deeper": {"leaf": 4.0}}},
            "in_sample": {},
            "out_of_sample": {},
        },
    )

    cells = _cells(html)
    assert cells["group.leaf"][0] == "3"
    assert cells["group.deeper.leaf"][0] == "4"


def test_a_mapping_with_every_shipped_key_deleted_still_renders(tmp_path):
    """The day the metric set is replaced wholesale, the page must still write."""
    html = _write(tmp_path, metrics={"whole": {}, "in_sample": {}, "out_of_sample": {}})

    assert html.startswith("<!DOCTYPE html>")
    assert '"name":"equity"' in html


def test_a_key_present_in_no_block_produces_no_row(tmp_path):
    html = _write(tmp_path, metrics={"whole": {"kept": 1.0}, "in_sample": {}})

    cells = _cells(html)
    assert "kept" in cells
    assert not [name for name in cells if name == "dropped"]


def test_a_null_block_is_a_column_of_dashes_rather_than_being_dropped(tmp_path):
    """A block that exists and is empty must be visible as such."""
    html = _write(
        tmp_path,
        metrics={"whole": {"x": 1.0, "y": 2.0}, "in_sample": None, "out_of_sample": {}},
    )

    assert "<th>in_sample</th>" in html
    cells = _cells(html)
    assert [row[1] for row in cells.values()] == ["—", "—"]


def test_unrenderable_values_become_a_dash_never_nan_or_none(tmp_path):
    html = _write(
        tmp_path,
        metrics={
            "whole": {
                "missing": None,
                "not_a_number": float("nan"),
                "infinite": float("inf"),
                "blank": "   ",
            }
        },
    )

    cells = _cells(html)
    for name in ("missing", "not_a_number", "infinite", "blank"):
        assert cells[name][0] == "—", name
    assert "nan" not in html.lower().split("<h2>metrics</h2>")[1].split("</table>")[0]


def test_bools_ints_and_strings_render_as_themselves(tmp_path):
    html = _write(
        tmp_path,
        metrics={"whole": {"flag": True, "count": 7, "label": "Closed"}},
    )

    cells = _cells(html)
    assert cells["flag"][0] == "true"
    assert cells["count"][0] == "7"
    assert cells["label"][0] == "Closed"


def test_metric_names_and_values_are_escaped(tmp_path):
    """T-sxx-01 reaches the table too: metric names are data, not markup."""
    html = _write(tmp_path, metrics={"whole": {"<b>name</b>": "<i>value</i>"}})

    assert "<b>name</b>" not in html and "<i>value</i>" not in html
    assert "&lt;b&gt;name&lt;/b&gt;" in html and "&lt;i&gt;value&lt;/i&gt;" in html


def test_no_metrics_mapping_means_no_metrics_table(tmp_path):
    assert "<h2>Metrics</h2>" not in _write(tmp_path, metrics=None)


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def _traces(html: str) -> dict[str, dict]:
    """The trace list plotly embeds, by name -- the run-level locks' parser."""
    start = html.index("[", html.index("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    return {trace["name"]: trace for trace in traces}


def test_every_trace_carries_a_name(tmp_path):
    """A trace without `name` raises KeyError in the persisted-report locks."""
    html = _write(
        tmp_path,
        returns=_returns(),
        drawdown_span=_span(),
    )

    start = html.index("[", html.index("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    assert traces and all("name" in trace for trace in traces)


def test_the_equity_y_stays_the_raw_value_and_the_multiple_is_customdata(tmp_path):
    """The page must not disagree with `equity.zarr`."""
    value = _value()
    html = _write(tmp_path, init_cash=1_000_000.0)

    equity = _traces(html)["equity"]
    np.testing.assert_allclose(equity["y"], value.values, rtol=1e-12)
    np.testing.assert_allclose(
        equity["customdata"], value.values / 1_000_000.0, rtol=1e-12
    )
    assert "customdata" in equity["hovertemplate"]


def test_equity_renders_without_init_cash(tmp_path):
    """`init_cash=None` must not turn into a division or a mislabelled axis."""
    html = _write(tmp_path, init_cash=None)
    equity = _traces(html)["equity"]

    np.testing.assert_allclose(equity["y"], _value().values, rtol=1e-12)
    assert "customdata" not in equity["hovertemplate"]


def test_the_log_linear_toggle_defaults_to_linear(tmp_path):
    html = _write(tmp_path)

    assert '"yaxis.type":"linear"' in html
    assert '"yaxis.type":"log"' in html
    # Default: the figure's own axis is not log.
    assert '"yaxis":{"type":"log"' not in html


def test_monthly_returns_are_grouped_by_calendar_month(tmp_path):
    """Compounded per calendar month, via `to_period`, not a resample alias."""
    index = pd.to_datetime(
        ["2024-01-10", "2024-01-20", "2024-02-05", "2024-03-01", "2024-03-20"]
    )
    returns = xr.DataArray(
        np.array([0.1, 0.1, -0.5, 0.2, 0.2]),
        dims=("timestamp",),
        coords={"timestamp": index},
    )
    html = _write(tmp_path, returns=returns)

    monthly = _traces(html)["monthly_return"]
    assert len(monthly["y"]) == 3
    np.testing.assert_allclose(
        monthly["y"], [1.1 * 1.1 - 1.0, -0.5, 1.2 * 1.2 - 1.0], atol=1e-12
    )


def test_a_one_month_run_is_a_single_bar(tmp_path):
    """A run whose whole P&L lands in one month is legible, not an error."""
    html = _write(tmp_path, returns=_returns())
    assert len(_traces(html)["monthly_return"]["y"]) == 1


def test_no_returns_means_no_monthly_trace(tmp_path):
    assert "monthly_return" not in _traces(_write(tmp_path, returns=None))


# Quick 260916-hro deleted three tests here -- the liquidation markers, the
# off-axis liquidation and the no-liquidation case -- together with
# `test_the_span_markers_are_not_the_liquidation_colour` below. All four
# exercised the `liquidations` parameter, which no longer exists, so they can
# no longer be WRITTEN rather than merely being redundant. What replaces them
# is the exact-set lock below, which goes red if a marker trace reappears.


def test_the_figure_draws_exactly_these_five_traces(tmp_path):
    """The whole trace set, pinned by name (quick 260916-hro).

    An exact set rather than a bare `not in`: it catches a liquidation trace
    coming back AND any other trace arriving unnoticed. The two
    `deepest_drawdown_*` traces are here because a span is passed, and
    `monthly_return` because returns are.
    """
    traces = _traces(_write(tmp_path, returns=_returns(), drawdown_span=_span()))

    assert set(traces) == {
        "equity",
        "drawdown",
        "monthly_return",
        "deepest_drawdown_valley",
        "deepest_drawdown_end",
    }


# ---------------------------------------------------------------------------
# The deepest drawdown's span (quick task 260915-v6i)
# ---------------------------------------------------------------------------
#
# The report module is told WHICH episode to mark; it never selects one. So
# these tests cover the drawing and the wording -- that the two endpoints land
# on the equity curve with the right symbols, that the span is stated in
# TRADING DAYS rather than as a calendar duration, that a drawdown which never
# recovered is not described as having ended, and that an endpoint the equity
# axis does not carry is dropped instead of raising. Choosing the deepest
# record is the engine's job and is locked in tests/test_backtest_engine.py.
#
# Quick 260916-hro moved the up triangle from the bar the drawdown STARTED to
# its VALLEY, so the payload key is `valley` and the trace is
# `deepest_drawdown_valley`. Which bar is the valley is the engine's decision
# and is proved there; here the payload is simply taken at its word.


def _span(**overrides) -> dict:
    """The span payload the engine hands the report, with defaults."""
    span = {
        "valley": BARS[3].strftime("%Y-%m-%d"),
        "end": BARS[8].strftime("%Y-%m-%d"),
        "bars": 5,
        "depth": -0.2,
        "recovered": True,
    }
    span.update(overrides)
    return span


def test_the_span_draws_one_triangle_at_each_end_on_the_equity_curve(tmp_path):
    """Up triangle at the VALLEY bar, down triangle at the recovery bar."""
    traces = _traces(_write(tmp_path, drawdown_span=_span()))

    valley = traces["deepest_drawdown_valley"]
    end = traces["deepest_drawdown_end"]
    assert valley["marker"]["symbol"] == "triangle-up"
    assert end["marker"]["symbol"] == "triangle-down"
    assert valley["mode"] == "markers" and end["mode"] == "markers"
    # One point each, sitting exactly on the plotted equity values.
    assert len(valley["y"]) == 1 and len(end["y"]) == 1
    np.testing.assert_allclose(valley["y"], [_value().values[3]], rtol=1e-12)
    np.testing.assert_allclose(end["y"], [_value().values[8]], rtol=1e-12)
    # Row 1 is the equity row; the drawdown row is x2.
    assert valley["xaxis"] == "x" and end["xaxis"] == "x"
    # Folded in from `test_the_span_markers_are_not_the_liquidation_colour`,
    # deleted in quick 260916-hro: its other half compared against the
    # liquidation marker, which no longer exists on the page. The surviving
    # half is that the two ends share ONE colour, because they are the two
    # ends of a single measurement and are told apart by shape.
    assert valley["marker"]["color"] == end["marker"]["color"]


def test_the_end_marker_states_the_span_in_trading_days_not_calendar_days(tmp_path):
    """D-2: the number beside the span is a BAR COUNT, worded as such.

    The span runs from a Thursday to the following Thursday -- 5 trading days
    but 7 calendar days -- so a page that rendered a timedelta would say
    something different from what vectorbt's own duration measures. The
    `days` count assertion is what catches a `Timedelta` leaking in: its repr
    (`7 days 00:00:00`) carries a `days` that is not part of `trading days`.
    """
    traces = _traces(_write(tmp_path, drawdown_span=_span(bars=5)))
    hover = traces["deepest_drawdown_end"]["hovertemplate"]

    assert "5" in hover
    assert "trading days" in hover
    assert hover.count("days") == hover.count("trading days"), hover
    assert "7" not in hover, "the calendar span must not appear anywhere"
    # The depth is stated as a percentage, so the marker is self-describing.
    assert "-20.00%" in hover
    # T-hro-03: since 260916-hro the count runs from the VALLEY, so it is not
    # that metric for two independent reasons (a possibly different episode,
    # and a different starting bar). The hover must not claim otherwise.
    assert "Max Drawdown Duration" not in hover


def test_a_never_recovered_span_says_so_and_never_claims_it_ended(tmp_path):
    """A drawdown still open at the last bar must not be described as recovered."""
    traces = _traces(_write(tmp_path, drawdown_span=_span(recovered=False)))
    hover = traces["deepest_drawdown_end"]["hovertemplate"]

    assert "not recovered" in hover
    assert "recovers" not in hover
    # Still a bar count, still in trading days.
    assert "trading days" in hover
    assert hover.count("days") == hover.count("trading days"), hover


def test_a_recovered_span_says_it_recovered(tmp_path):
    """The other branch: the marked episode really did end at that bar."""
    hover = _traces(_write(tmp_path, drawdown_span=_span(recovered=True)))[
        "deepest_drawdown_end"
    ]["hovertemplate"]

    assert "not recovered" not in hover
    assert "recovers" in hover


def test_no_span_means_no_marker_traces(tmp_path):
    """A run with no drawdown record renders exactly today's page."""
    for traces in (
        _traces(_write(tmp_path, drawdown_span=None)),
        _traces(_write(tmp_path)),  # the argument omitted entirely
    ):
        assert "deepest_drawdown_valley" not in traces
        assert "deepest_drawdown_end" not in traces


@pytest.mark.parametrize(
    ("span", "kept"),
    [
        ({"valley": "1999-01-01"}, "deepest_drawdown_end"),
        ({"end": "1999-01-01"}, "deepest_drawdown_valley"),
    ],
)
def test_an_endpoint_off_the_equity_axis_drops_that_marker_only(tmp_path, span, kept):
    """T-v6i-02: the report is the last step of a run that already succeeded.

    It is written inside the staging directory, so an exception here deletes
    the ENTIRE run, not just the report. An endpoint the equity axis does not
    carry therefore drops its own marker and leaves the other one standing,
    rather than raising.
    """
    traces = _traces(_write(tmp_path, drawdown_span=_span(**span)))

    assert kept in traces
    dropped = {"deepest_drawdown_valley", "deepest_drawdown_end"} - {kept}
    assert dropped.isdisjoint(traces)


def test_a_span_missing_its_keys_renders_the_page_instead_of_raising(tmp_path):
    """Every key is read with `.get`, so a malformed payload is not fatal."""
    html = _write(tmp_path, drawdown_span={"bars": 3})

    assert html.startswith("<!DOCTYPE html>")
    traces = _traces(html)
    assert "equity" in traces
    assert "deepest_drawdown_valley" not in traces
    assert "deepest_drawdown_end" not in traces


@pytest.mark.parametrize("n_bars", [1, 2])
def test_a_very_short_window_still_renders(tmp_path, n_bars):
    """A one- or two-bar window is legitimate, not an error."""
    path = tmp_path / "report.html"
    write_backtest_report(
        _value(n_bars),
        path,
        in_sample_range=None,
        notes=[],
        title="short",
        summary={"Backtest window": f"2024-01-01 .. 2024-01-02 ({n_bars} bars)"},
    )
    assert '"name":"equity"' in path.read_text(encoding="utf-8")
