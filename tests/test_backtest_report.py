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
    # The fourth cell is the out_of_sample - in_sample delta: in_sample is None,
    # so there is nothing to subtract and the delta is a dash.
    assert cells["a_metric_nobody_has_written_yet"] == ["1.5", "—", "2.5", "—"]


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
    # The full row, not one indexed column: whole, in_sample, out_of_sample and
    # the delta. Indexing column 1 alone would stay green whatever the column
    # count became, so it could not lock the table's arity.
    assert cells == {"x": ["1", "—", "—", "—"], "y": ["2", "—", "—", "—"]}


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
# The delta column: out_of_sample - in_sample, decided by type alone (03.8 D-01)
# ---------------------------------------------------------------------------
#
# Every operand below is the RAW object the report receives at render time --
# pd.Timedelta, pd.Timestamp, pd.NaT, np.float64 -- never the JSON shape
# `to_jsonable` later writes to metrics.json. A predicate written against the
# JSON shape (strings for durations, None for nan) would pass tests fed only
# plain floats and still fail on a real run.


def _delta_of(tmp_path, in_sample, out_of_sample) -> str:
    """The delta cell of one row whose two slice values are the given objects."""
    html = _write(
        tmp_path,
        metrics={
            "whole": {"m": 0.0},
            "in_sample": {"m": in_sample},
            "out_of_sample": {"m": out_of_sample},
        },
    )
    row = _cells(html)["m"]
    assert len(row) == 4, f"expected whole, in_sample, out_of_sample and delta: {row}"
    return row[3]


def _headers(html: str) -> list[str]:
    """The metric table's header cells, in order."""
    head = re.search(r"<thead><tr>(.*?)</tr></thead>", html).group(1)
    return re.findall(r"<th>([^<]*)</th>", head)


def test_the_delta_header_states_the_arithmetic(tmp_path):
    """The header names the operation, not a judgement such as `degradation`."""
    html = _write(
        tmp_path,
        metrics={"whole": {"m": 1.0}, "in_sample": {"m": 1.0}, "out_of_sample": {"m": 2.0}},
    )

    assert _headers(html) == [
        "metric",
        "whole",
        "in_sample",
        "out_of_sample",
        "out_of_sample - in_sample",
    ]


def test_the_delta_of_two_finite_floats_is_their_difference(tmp_path):
    assert _delta_of(tmp_path, 1.0, 3.5) == "2.5"


def test_the_delta_of_two_numpy_floats_is_their_difference(tmp_path):
    """numpy scalars are admitted by numbers.Real without the leaf importing numpy."""
    assert _delta_of(tmp_path, np.float64(1.25), np.float64(0.25)) == "-1"


def test_the_delta_of_two_ints_is_their_integer_difference(tmp_path):
    assert _delta_of(tmp_path, 7, 12) == "5"


def test_the_delta_of_two_numpy_ints_is_their_integer_difference(tmp_path):
    """np.int64 is not an int subclass; a (int, float) check would dash it."""
    assert _delta_of(tmp_path, np.int64(7), np.int64(12)) == "5"


def test_the_delta_of_a_bool_pair_is_a_dash_never_one(tmp_path):
    """bool is an int subclass: True - False == 1 would render as a real number."""
    assert _delta_of(tmp_path, False, True) == "—"
    assert _delta_of(tmp_path, np.bool_(False), np.bool_(True)) == "—"


@pytest.mark.parametrize(
    "in_sample, out_of_sample",
    [
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (float("inf"), 1.0),
        (1.0, float("-inf")),
        (np.float64("nan"), np.float64(1.0)),
        (np.float64(1.0), np.float64("nan")),
        (np.float32("nan"), np.float32(1.0)),
        (np.float32(1.0), np.float32("inf")),
    ],
)
def test_the_delta_with_a_non_finite_operand_is_a_dash(tmp_path, in_sample, out_of_sample):
    """nan - 1 is nan and inf - 1 is inf: neither is a difference a reader can use.

    The np.float32 cases are the ones that make the predicate's own finiteness
    check load-bearing. A Python or np.float64 nan result is a `float`, so
    `_cell` would dash it anyway; np.float32 is a `numbers.Real` but not a
    `float`, so without the check its nan result would reach `_cell`'s `str()`
    fallback and print the token `nan` as though it were a number.
    """
    assert _delta_of(tmp_path, in_sample, out_of_sample) == "—"


def test_the_delta_of_a_timedelta_pair_is_a_dash(tmp_path):
    """Durations are excluded by type, never by metric name."""
    assert _delta_of(tmp_path, pd.Timedelta(days=3), pd.Timedelta(days=10)) == "—"


def test_the_delta_of_a_timestamp_pair_is_a_dash(tmp_path):
    """Timestamp - Timestamp is a Timedelta: not a difference of two metric values."""
    assert (
        _delta_of(tmp_path, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01"))
        == "—"
    )


def test_the_delta_with_a_nat_operand_is_a_dash(tmp_path):
    assert _delta_of(tmp_path, pd.NaT, pd.NaT) == "—"
    assert _delta_of(tmp_path, pd.NaT, pd.Timestamp("2024-06-01")) == "—"


def test_the_delta_of_a_string_pair_is_a_dash_and_raises_nothing(tmp_path):
    """str - str raises TypeError; inside the staging directory that costs the run."""
    assert _delta_of(tmp_path, "Closed", "Open") == "—"


def test_the_delta_of_mixed_types_is_a_dash(tmp_path):
    assert _delta_of(tmp_path, 1.0, pd.Timedelta(days=1)) == "—"
    assert _delta_of(tmp_path, None, 3.0) == "—"


def test_the_delta_of_a_key_only_in_whole_is_a_dash_and_raises_nothing(tmp_path):
    """Both slice operands absent: None - None must never be attempted."""
    html = _write(
        tmp_path,
        metrics={
            "whole": {"only_whole": 4.0, "m": 1.0},
            "in_sample": {"m": 1.0},
            "out_of_sample": {"m": 2.0},
        },
    )

    cells = _cells(html)
    assert cells["only_whole"] == ["4", "—", "—", "—"]
    assert cells["m"] == ["1", "1", "2", "1"]


def test_a_null_in_sample_block_keeps_the_delta_column_all_dashes(tmp_path):
    """A None block is still present: the column shows, and every delta is a dash."""
    html = _write(
        tmp_path,
        metrics={
            "whole": {"a": 1.0, "b": 2.0},
            "in_sample": None,
            "out_of_sample": {"a": 5.0, "b": 6.0},
        },
    )

    assert html.startswith("<!DOCTYPE html>")
    assert "out_of_sample - in_sample" in _headers(html)
    assert [row[3] for row in _cells(html).values()] == ["—", "—"]


def test_a_mapping_with_only_whole_grows_no_delta_column(tmp_path):
    html = _write(tmp_path, metrics={"whole": {"m": 1.0}})

    assert _headers(html) == ["metric", "whole"]
    assert _cells(html)["m"] == ["1"]
    assert "out_of_sample - in_sample" not in html


def test_a_mapping_without_out_of_sample_grows_no_delta_column(tmp_path):
    html = _write(tmp_path, metrics={"whole": {"m": 1.0}, "in_sample": {"m": 2.0}})

    assert _headers(html) == ["metric", "whole", "in_sample"]
    assert "out_of_sample - in_sample" not in html


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def _traces(html: str) -> dict[str, dict]:
    """The trace list plotly embeds, by name -- the run-level locks' parser."""
    start = html.index("[", html.index("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    return {trace["name"]: trace for trace in traces}


def _layout(html: str) -> dict:
    """The layout object plotly embeds: the JSON value after the trace array."""
    start = html.index("[", html.index("Plotly.newPlot("))
    _, end = json.JSONDecoder().raw_decode(html, start)
    layout, _ = json.JSONDecoder().raw_decode(html, html.index("{", end))
    return layout


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


# ---------------------------------------------------------------------------
# The year-by-month heatmap: a SECOND plotly div (03.8 D-04)
# ---------------------------------------------------------------------------
#
# The heatmap is its own figure rendered as a second div, never a fourth
# subplot row (RESEARCH Pitfall 6). `_traces` and `_layout` above anchor on the
# FIRST `Plotly.newPlot(` by design -- that is what keeps the exactly-five-
# traces and pixel-budget locks meaningful -- so the heatmap is invisible to
# them. These tests reach it through `_second_figure_traces`.

HEATMAP = "monthly_return_heatmap"
MONTH_LABELS = [f"{month:02d}" for month in range(1, 13)]


def _second_figure(html: str) -> tuple[list[dict], dict]:
    """Trace list and layout of the SECOND `Plotly.newPlot(` on the page."""
    first = html.index("Plotly.newPlot(")
    second = html.index("Plotly.newPlot(", first + 1)
    start = html.index("[", second)
    traces, end = json.JSONDecoder().raw_decode(html, start)
    layout, _ = json.JSONDecoder().raw_decode(html, html.index("{", end))
    return traces, layout


def _second_figure_traces(html: str) -> dict[str, dict]:
    """Traces of the SECOND `Plotly.newPlot(` on the page, by name.

    The existing parsers read the FIRST one by design -- that is what keeps
    the three-row figure's exact-trace-set and pixel-budget locks meaningful.
    The heatmap lives in its own div after it, so without this parser it
    would ship with no lock at all.
    """
    traces, _ = _second_figure(html)
    return {trace["name"]: trace for trace in traces}


def _dated_returns(dates: list[str], values: list[float]) -> xr.DataArray:
    """Returns on the given calendar dates, in `_returns`' shape."""
    return xr.DataArray(
        np.array(values, dtype=float),
        dims=("timestamp",),
        coords={"timestamp": pd.DatetimeIndex(pd.to_datetime(dates))},
    )


def _multi_year_returns() -> xr.DataArray:
    """One return on the 15th of every month, November 2023 to February 2025.

    Starts late in its first year and ends early in its last, so both partial
    rows are exercised. Values are distinct per month so a cell that lands in
    the wrong column cannot match by accident.
    """
    months = pd.period_range("2023-11", "2025-02", freq="M")
    dates = [(period.to_timestamp() + pd.Timedelta(days=14)).strftime("%Y-%m-%d") for period in months]
    values = [0.01 * (i + 1) for i in range(len(months))]
    return _dated_returns(dates, values)


def _non_null_cells(heatmap: dict) -> list[tuple[str, str, float]]:
    """Every non-null cell as `(year label, month label, value)`."""
    return [
        (heatmap["y"][row], heatmap["x"][col], value)
        for row, cells in enumerate(heatmap["z"])
        for col, value in enumerate(cells)
        if value is not None
    ]


def test_the_heatmap_puts_a_march_return_in_the_march_column(tmp_path):
    """RESEARCH Pitfall 7: the month off-by-one is otherwise invisible.

    ONE return in ONE known month, and the assertion is on the x LABEL of the
    single non-null cell. Using `period.month` instead of `period.month - 1`
    would put it at `04` with nothing else differing -- the grid's shape, row
    count and non-null count all survive that bug, so asserting shape would
    stay green.
    """
    html = _write(tmp_path, returns=_dated_returns(["2024-03-12"], [0.25]))
    heatmap = _second_figure_traces(html)[HEATMAP]

    assert heatmap["x"] == MONTH_LABELS
    cells = _non_null_cells(heatmap)
    assert len(cells) == 1
    year, month, value = cells[0]
    assert (year, month) == ("2024", "03")
    assert value == pytest.approx(0.25)


def test_the_heatmap_puts_the_same_month_of_two_years_in_two_rows(tmp_path):
    """Year alignment: March 2023 and March 2024 land in their own rows."""
    html = _write(
        tmp_path, returns=_dated_returns(["2023-03-10", "2024-03-10"], [0.1, -0.2])
    )
    heatmap = _second_figure_traces(html)[HEATMAP]

    assert heatmap["y"] == ["2023", "2024"]
    assert sorted(_non_null_cells(heatmap)) == [
        ("2023", "03", pytest.approx(0.1)),
        ("2024", "03", pytest.approx(-0.2)),
    ]


def test_the_heatmap_leaves_the_months_before_a_november_start_null(tmp_path):
    """Uncovered months are JSON null, never a fabricated 0.0.

    A 0.0 cell would read as a flat month the run actually traded.
    """
    html = _write(tmp_path, returns=_multi_year_returns())
    heatmap = _second_figure_traces(html)[HEATMAP]

    assert heatmap["y"] == ["2023", "2024", "2025"]
    first_year = heatmap["z"][0]
    assert len(first_year) == 12
    assert first_year[:10] == [None] * 10
    assert first_year[10:] == pytest.approx([0.01, 0.02])


def test_the_heatmap_leaves_the_months_after_a_february_end_null(tmp_path):
    """The partial LAST year: ten trailing nulls, again never zeros."""
    html = _write(tmp_path, returns=_multi_year_returns())
    heatmap = _second_figure_traces(html)[HEATMAP]

    last_year = heatmap["z"][2]
    assert len(last_year) == 12
    assert last_year[2:] == [None] * 10
    assert last_year[:2] == pytest.approx([0.15, 0.16])
    # The full middle year is covered end to end.
    assert None not in heatmap["z"][1]


@pytest.mark.parametrize(
    "returns",
    [
        None,
        _dated_returns(["2024-01-02", "2024-02-02"], [float("nan"), float("nan")]),
        _dated_returns([], []),
    ],
    ids=["none", "all_nan", "empty"],
)
def test_no_heatmap_div_without_returns_and_the_page_is_still_written(tmp_path, returns):
    """T-03.8-03-01: the report is written inside the run's staging directory.

    An exception here would delete the whole run, so nothing-to-draw is an
    early return: no second `Plotly.newPlot(`, no stray caption, and the page
    is still a complete document.
    """
    html = _write(tmp_path, returns=returns)

    assert html.startswith("<!DOCTYPE html>")
    assert html.count("Plotly.newPlot(") == 1
    assert "Monthly returns by year" not in html
    assert '"name":"equity"' in html


def test_the_heatmap_cells_equal_the_monthly_bars(tmp_path):
    """Both panels come from one helper, so the same months carry the same numbers."""
    html = _write(tmp_path, returns=_multi_year_returns())
    bars = _traces(html)["monthly_return"]
    heatmap = _second_figure_traces(html)[HEATMAP]

    from_bars = {
        (label[:4], label[5:7]): value for label, value in zip(bars["x"], bars["y"])
    }
    from_heatmap = {(year, month): value for year, month, value in _non_null_cells(heatmap)}
    assert from_heatmap.keys() == from_bars.keys()
    for key, value in from_bars.items():
        assert from_heatmap[key] == pytest.approx(value, abs=1e-12), key


def test_the_heatmap_trace_carries_a_name(tmp_path):
    """Every trace on the page is parsed by name; a nameless one raises KeyError."""
    traces, _ = _second_figure(_write(tmp_path, returns=_multi_year_returns()))

    assert len(traces) == 1
    assert traces[0].get("name") == HEATMAP
    assert traces[0]["type"] == "heatmap"


def test_the_heatmap_puts_the_earliest_year_on_top(tmp_path):
    """Reading order: the y axis is reversed, and both axes are categorical."""
    _, layout = _second_figure(_write(tmp_path, returns=_multi_year_returns()))

    assert layout["yaxis"]["autorange"] == "reversed"
    assert layout["yaxis"]["type"] == "category"
    assert layout["xaxis"]["type"] == "category"


def test_the_heatmap_height_grows_with_the_number_of_years(tmp_path):
    """A six-year run must not be squashed into the height of a one-year run."""
    one_year = _second_figure(
        _write(tmp_path, returns=_dated_returns(["2024-03-12"], [0.25]))
    )[1]["height"]
    six_years = _second_figure(
        _write(
            tmp_path,
            returns=_dated_returns(
                [f"{year}-03-12" for year in range(2019, 2025)], [0.1] * 6
            ),
        )
    )[1]["height"]

    assert six_years > one_year


def test_the_heatmap_div_sits_between_the_figure_and_the_metrics_table(tmp_path):
    """Picture, picture, numbers, notes -- and it loads no second plotly.js."""
    html = _write(
        tmp_path, returns=_multi_year_returns(), metrics={"whole": {"m": 1.0}}
    )

    first = html.index("Plotly.newPlot(")
    second = html.index("Plotly.newPlot(", first + 1)
    assert first < html.index("Monthly returns by year") < second
    assert second < html.index("<h2>Metrics</h2>") < html.index("<h2>Notes</h2>")
    assert html.count("cdn.plot.ly") == 1


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


def test_the_layout_gives_every_axis_title_room_to_render(tmp_path):
    """D-03: an explicit height, short titles, and a per-row pixel budget.

    The overlap this locks was VERTICAL. A y-axis title is rotated 90
    degrees, so its rendered length is measured against its own axis height.
    Without an explicit `height` the div falls back to plotly's 450px
    default, which leaves rows 2 and 3 about 44px tall -- shorter than the
    titles they carry, so all three collided.

    The height alone is not asserted, because a later `row_heights` or margin
    change could re-create the collision at any height. The per-row budget is
    what actually encodes the rule, and it goes red at the old 450px default.
    """
    html = _write(tmp_path, returns=_returns(), drawdown_span=_span())
    layout = _layout(html)

    assert layout.get("height") is not None, "an inherited 450px default is the bug"
    titles = (("yaxis", "value"), ("yaxis2", "drawdown"), ("yaxis3", "monthly return"))
    for axis, title in titles:
        assert layout[axis]["title"]["text"] == title, axis

    # The plotting area is the figure height less its margins. plotly's own
    # defaults are t=100 / b=80; only `b` is overridden here, so the top
    # default is what the figure really uses.
    plot_area = layout["height"] - layout["margin"].get("t", 100) - layout["margin"]["b"]

    # About 6.5px per character at the default font size, the title being
    # rotated onto the vertical axis. Derived from the measured figures in
    # the task's F-5: roughly 33 / 52 / 91px of text against 328 / 140 / 140px
    # of row -- rather than a bare pixel constant with no way to re-derive it.
    for axis, title in titles:
        domain = layout[axis]["domain"]
        row_px = (domain[1] - domain[0]) * plot_area
        assert row_px >= len(title) * 6.5, (axis, row_px, title)


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
