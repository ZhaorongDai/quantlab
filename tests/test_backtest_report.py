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
