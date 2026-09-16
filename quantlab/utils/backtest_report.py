"""Interactive HTML report for one backtest run (03.7 D-23, D-21, D-08).

One self-contained page built around a plotly div: an escaped `<h1>` of the run
name, a dates-and-setup block stating the window in words, the plotly figure,
and the notes. The figure keeps two panels on a shared time axis -- the equity
curve on top and the drawdown curve below. When the backtest window overlaps
the model's effective training window, the in-sample range is shaded grey across
both panels. The in-sample range is the intersection of two intervals, so it is
always one contiguous band (the out-of-sample part may be two pieces, but it is
the unshaded remainder).

The module composes its own HTML document rather than calling
`fig.write_html`, so every value interpolated into the page passes through
`html.escape` first: a run directory name or a note carrying angle brackets
renders as text, never as live markup. The page title is rendered ONLY in the
escaped `<h1>`; it is deliberately not handed to plotly's layout title, so no
unescaped copy of it reaches the embedded JSON payload.

The page loads plotly.js from the CDN (`include_plotlyjs="cdn"`). That keeps
each run directory at a few kilobytes instead of several megabytes per report;
the cost is that viewing the page needs network access. The page contains only
local backtest numbers, so the browser's CDN fetch reveals nothing about them.

No benchmark trace is drawn: benchmark comparison is excluded this phase (D-08).

A LEAF module: stdlib, pandas, xarray and plotly only, zero project-internal
imports.
"""

import html
from pathlib import Path

import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

__all__ = ["write_backtest_report"]

#: Rendered in place of a value the run does not have. An em dash rather than a
#: hyphen so it cannot be misread as the minus sign of a negative number.
DASH = "—"

_STYLE = """
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
         margin: 24px; color: #1a1a1a; }
  h1 { font-size: 20px; margin: 0 0 16px 0; }
  h2 { font-size: 15px; margin: 24px 0 8px 0; color: #444; }
  table.summary { border-collapse: collapse; font-size: 13px; }
  table.summary th { text-align: left; padding: 3px 16px 3px 0;
                     font-weight: 600; color: #444; white-space: nowrap; }
  table.summary td { padding: 3px 0; font-variant-numeric: tabular-nums; }
  ul.notes { font-size: 13px; color: #444; padding-left: 20px; }
"""


def write_backtest_report(
    value: xr.DataArray,
    path: str | Path,
    *,
    in_sample_range: tuple[str, str] | None,
    notes: list[str],
    title: str,
    summary: dict[str, str] | None = None,
    metrics: dict | None = None,
    returns: xr.DataArray | None = None,
    liquidations: list[dict] | None = None,
    init_cash: float | None = None,
) -> None:
    """Write the report for `value` to `path`.

    - `value`: portfolio value on the `timestamp` dimension;
    - `in_sample_range`: bar-label pair (first, last in-sample bar), shaded
      when given, or None for a fully out-of-sample window. A midnight bar is
      labelled by its ISO date and any other bar by its full ISO timestamp;
      plotly reads both;
    - `notes`: lines printed below the plot (e.g. what the simulation does not
      model);
    - `title`: page title, typically the run directory name;
    - `summary`: ordered mapping of display label to already-formatted display
      string, rendered as the dates-and-setup block. This module formats
      nothing and computes nothing: the caller decides both the labels and the
      text, so the page can state the same strings the run's `metrics.json`
      carries.

    `metrics`, `returns`, `liquidations` and `init_cash` are accepted and
    reserved for the metric table and the expanded chart set; they are not read
    yet. Every new parameter defaults to None so the original five-argument call
    form stays legal.

    Drawdown is `value / running max - 1`, so it is 0 at a new high and
    negative below it.
    """
    equity = value.to_pandas()
    drawdown = equity / equity.cummax() - 1.0

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, name="equity", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, name="drawdown", mode="lines"),
        row=2,
        col=1,
    )
    if in_sample_range is not None:
        fig.add_vrect(
            x0=in_sample_range[0],
            x1=in_sample_range[1],
            row="all",
            col=1,
            fillcolor="grey",
            opacity=0.2,
            line_width=0,
        )
    if notes:
        fig.add_annotation(
            text="<br>".join(notes),
            xref="paper",
            yref="paper",
            x=0.0,
            y=-0.12,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
        )
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="drawdown", tickformat=".1%", row=2, col=1)
    fig.update_layout(margin={"b": 120})

    div = fig.to_html(full_html=False, include_plotlyjs="cdn")
    Path(path).write_text(
        _document(title, summary, div, notes), encoding="utf-8"
    )


def _escape(value: object) -> str:
    """`str(value)` with every HTML-significant character escaped (T-sxx-01)."""
    return html.escape(str(value))


def _summary_section(summary: dict[str, str] | None) -> str:
    """The dates-and-setup block; empty string when the caller passed nothing."""
    if not summary:
        return ""
    rows = "\n".join(
        f"      <tr><th>{_escape(label)}</th><td>{_escape(text)}</td></tr>"
        for label, text in summary.items()
    )
    return (
        "  <h2>Dates and setup</h2>\n"
        '  <table class="summary">\n'
        f"{rows}\n"
        "  </table>\n"
    )


def _notes_section(notes: list[str] | None) -> str:
    """The notes list; empty string when there are none."""
    if not notes:
        return ""
    items = "\n".join(f"    <li>{_escape(note)}</li>" for note in notes)
    return (
        "  <h2>Notes</h2>\n"
        '  <ul class="notes">\n'
        f"{items}\n"
        "  </ul>\n"
    )


def _document(
    title: str, summary: dict[str, str] | None, div: str, notes: list[str] | None
) -> str:
    """One self-contained HTML document around the plotly `div`.

    `div` is plotly's own fragment and is inserted verbatim -- plotly owns its
    escaping. Everything else on the page comes from the run and is escaped.
    """
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{_escape(title)}</title>\n"
        f"  <style>{_STYLE}  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>{_escape(title)}</h1>\n"
        f"{_summary_section(summary)}"
        f"{div}\n"
        f"{_notes_section(notes)}"
        "</body>\n"
        "</html>\n"
    )
