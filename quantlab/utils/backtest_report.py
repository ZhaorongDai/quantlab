"""Interactive HTML report for one backtest run (03.7 D-23, D-21, D-08).

One self-contained page built around a plotly div:

1. an escaped `<h1>` of the run name;
2. a dates-and-setup block stating the window, the bar count and interval, the
   training window(s), the in-sample range(s) and the out-of-sample ranges in
   words, so the reader never has to open `metrics.json` to learn what the
   picture covers;
3. a metric table with one column per metrics block (whole / in-sample /
   out-of-sample);
4. the figure: three rows on a shared time axis -- equity (with forced
   liquidation markers) on top, drawdown below it, per-calendar-month returns
   at the bottom -- plus a log/linear toggle for the equity axis;
5. the notes.

When the backtest window overlaps the model's effective training window, the
in-sample range is shaded grey across the panels. The in-sample range is the
intersection of two intervals, so it is always one contiguous band (the
out-of-sample part may be two pieces, but it is the unshaded remainder).

**The metric table names no metric.** Its rows are derived by walking whatever
mapping the caller passes, at render time, flattening nested dicts to dotted
paths. This is load-bearing rather than stylistic: the project is moving its
metrics to vectorbt's own set, so the keys this table renders are going to be
replaced wholesale. A report that listed metric names in code would raise the
day that lands -- and because the report is written inside the staging
directory of a run, that exception would delete the ENTIRE run directory, not
just the report. Deriving the rows from the data, reading split values with
`.get`, and rendering an unknown value as a dash is what keeps a future metric
change a cosmetic event instead of a data-loss event.

The module composes its own HTML document rather than calling
`fig.write_html`, so every value interpolated into the page passes through
`html.escape` first: a run directory name, a note or a metric name carrying
angle brackets renders as text, never as live markup. The page title is
rendered ONLY in the escaped `<h1>`; it is deliberately not handed to plotly's
layout title, so no unescaped copy of it reaches the embedded JSON payload.

The equity trace keeps the RAW persisted portfolio value on `y`, identical to
`equity.zarr`'s `value`; the multiple of initial capital rides along as
`customdata` and is surfaced by the hover template. Normalising `y` itself was
rejected: it would break the page's correspondence with the persisted equity
and would silently mislabel the axis whenever `init_cash` is unknown.

The page loads plotly.js from the CDN (`include_plotlyjs="cdn"`). That keeps
each run directory at a few kilobytes instead of several megabytes per report;
the cost is that viewing the page needs network access. The page contains only
local backtest numbers, so the browser's CDN fetch reveals nothing about them.

No benchmark trace is drawn: benchmark comparison is excluded this phase (D-08).

A LEAF module: stdlib, pandas, xarray and plotly only, zero project-internal
imports.
"""

import html
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

__all__ = ["write_backtest_report"]

#: Rendered in place of a value the run does not have. An em dash rather than a
#: hyphen so it cannot be misread as the minus sign of a negative number.
DASH = "—"

#: The metrics blocks the table shows, in column order. These are SPLIT keys
#: (which slice of the window), never metric names: the rows inside each block
#: are whatever that block turns out to carry.
_BLOCKS = ("whole", "in_sample", "out_of_sample")

_STYLE = """
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
         margin: 24px; color: #1a1a1a; }
  h1 { font-size: 20px; margin: 0 0 16px 0; }
  h2 { font-size: 15px; margin: 24px 0 8px 0; color: #444; }
  table.summary { border-collapse: collapse; font-size: 13px; }
  table.summary th { text-align: left; padding: 3px 16px 3px 0;
                     font-weight: 600; color: #444; white-space: nowrap; }
  table.summary td { padding: 3px 0; font-variant-numeric: tabular-nums; }
  table.metrics { border-collapse: collapse; font-size: 13px; }
  table.metrics th, table.metrics td { padding: 3px 18px 3px 0;
                                       border-bottom: 1px solid #eee;
                                       white-space: nowrap; }
  table.metrics thead th { text-align: left; color: #444; }
  table.metrics tbody th { text-align: left; font-weight: 400; color: #444; }
  table.metrics td { text-align: right; font-variant-numeric: tabular-nums; }
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

    - `value`: portfolio value on the `timestamp` dimension. Drawn raw, so the
      page and `equity.zarr` carry the same numbers;
    - `path`: the `report.html` to write;
    - `in_sample_range`: bar-label pair (first, last in-sample bar), shaded
      when given, or None for a fully out-of-sample window. A midnight bar is
      labelled by its ISO date and any other bar by its full ISO timestamp;
      plotly reads both;
    - `notes`: lines printed below the plot (e.g. what the simulation does not
      model);
    - `title`: page title, typically the run directory name;
    - `summary`: ordered mapping of display label to already-formatted display
      string, rendered as the dates-and-setup block. This module formats
      nothing there and computes nothing: the caller decides both the labels
      and the text, so the page can state the same strings the run's
      `metrics.json` carries;
    - `metrics`: the metrics mapping for ONE curve -- for a CV run that is the
      stitched block, not the whole file. Its `whole` / `in_sample` /
      `out_of_sample` entries become the table's columns and may be None or
      absent; their contents are walked, never assumed. `trained_checkpoint`
      is shown when present;
    - `returns`: per-bar portfolio returns on `timestamp`, compounded per
      calendar month for the bottom panel;
    - `liquidations`: forced-liquidation records (`symbol`, `fill_timestamp`),
      drawn as markers on the equity row. Records whose timestamp is not on the
      equity axis are dropped rather than raising;
    - `init_cash`: starting capital, used only to express equity as a multiple
      in the hover text.

    Every argument after `title` defaults to None, so the original
    five-argument call form stays legal.

    Drawdown is `value / running max - 1`, so it is 0 at a new high and
    negative below it.
    """
    equity = value.to_pandas()
    drawdown = equity / equity.cummax() - 1.0

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.54, 0.23, 0.23],
        vertical_spacing=0.04,
    )
    _add_equity(fig, equity, init_cash)
    _add_liquidations(fig, equity, liquidations)
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, name="drawdown", mode="lines"),
        row=2,
        col=1,
    )
    _add_monthly_returns(fig, returns)

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
            y=-0.16,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
        )
    fig.update_yaxes(title_text="value (x initial capital on hover)", row=1, col=1)
    fig.update_yaxes(title_text="drawdown", tickformat=".1%", row=2, col=1)
    fig.update_yaxes(title_text="monthly return", tickformat=".1%", row=3, col=1)
    fig.update_layout(
        margin={"b": 140},
        showlegend=False,
        updatemenus=[_axis_toggle()],
    )

    div = fig.to_html(full_html=False, include_plotlyjs="cdn")
    Path(path).write_text(
        _document(title, summary, metrics, div, notes), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def _add_equity(fig, equity: pd.Series, init_cash: float | None) -> None:
    """The equity trace: raw value on `y`, the multiple as `customdata`.

    `y` stays the persisted portfolio value so the page cannot disagree with
    `equity.zarr`. The multiple of initial capital is what makes a run that
    compounded by orders of magnitude legible, so it rides along in
    `customdata` and is shown on hover instead of replacing `y`.
    """
    if init_cash:
        multiple = equity.values / float(init_cash)
        hover = "%{x}<br>value %{y:,.2f}<br>%{customdata:,.4f}x initial<extra></extra>"
    else:
        multiple = [None] * len(equity)
        hover = "%{x}<br>value %{y:,.2f}<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="equity",
            mode="lines",
            customdata=multiple,
            hovertemplate=hover,
        ),
        row=1,
        col=1,
    )


def _add_liquidations(fig, equity: pd.Series, liquidations) -> None:
    """Markers on the equity row at each forced liquidation's fill bar.

    A record whose timestamp is not on the equity axis is dropped: the report
    is the last step of a run that already succeeded, so it must not be the
    thing that fails. The trace is omitted entirely when nothing was
    liquidated, so a clean run's page carries no empty legend entry.
    """
    if not liquidations:
        return
    xs, texts = [], []
    for record in liquidations:
        stamp = pd.Timestamp(str(record.get("fill_timestamp")))
        if stamp not in equity.index:
            continue
        xs.append(stamp)
        texts.append(str(record.get("symbol", "")))
    if not xs:
        return
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[equity.loc[stamp] for stamp in xs],
            name="liquidation",
            mode="markers",
            marker={"symbol": "x", "size": 9, "color": "#c0392b"},
            text=texts,
            hovertemplate="%{x}<br>forced liquidation: %{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _add_monthly_returns(fig, returns: xr.DataArray | None) -> None:
    """Per-calendar-month compounded return of `returns`, as bars.

    Grouped with `index.to_period("M")` rather than a resample alias: the
    monthly alias was renamed (`M` -> `ME`) across pandas versions while
    `to_period` reads the same in both. A short window legitimately produces
    one or two bars -- that is the point, since it shows at a glance that a
    run's whole P&L landed in a single month.
    """
    if returns is None:
        return
    series = returns.to_pandas().dropna()
    if series.empty:
        return
    index = pd.DatetimeIndex(series.index)
    monthly = (1.0 + series).groupby(index.to_period("M")).prod() - 1.0
    fig.add_trace(
        go.Bar(
            x=[period.to_timestamp() for period in monthly.index],
            y=monthly.values,
            name="monthly_return",
            hovertemplate="%{x|%Y-%m}<br>%{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
    )


def _axis_toggle() -> dict:
    """Linear/log buttons for the equity axis.

    Linear is the default: log is one click away, while a run that lost
    everything has a non-positive value that renders an empty log panel. The
    log button is what turns a curve that compounded by orders of magnitude
    from a flat line with a final spike into a readable slope.
    """
    return {
        "type": "buttons",
        "direction": "right",
        "x": 0.0,
        "y": 1.12,
        "xanchor": "left",
        "yanchor": "top",
        "showactive": True,
        "buttons": [
            {"label": "linear", "method": "relayout", "args": [{"yaxis.type": "linear"}]},
            {"label": "log", "method": "relayout", "args": [{"yaxis.type": "log"}]},
        ],
    }


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def _escape(value: object) -> str:
    """`str(value)` with every HTML-significant character escaped (T-sxx-01)."""
    return html.escape(str(value))


def _flatten(value: dict, prefix: str = "") -> dict:
    """A block flattened to `dotted.path -> leaf`, walking nested dicts.

    Generic by construction: a sub-dict the code has never seen becomes rows
    under its own name, and a block that loses one keeps rendering. Nothing
    here knows any metric name.
    """
    flat: dict = {}
    for key, item in value.items():
        path = f"{prefix}{key}"
        if isinstance(item, dict):
            flat.update(_flatten(item, f"{path}."))
        else:
            flat[path] = item
    return flat


def _cell(value: object) -> str:
    """One metric value as display text; anything unrenderable becomes a dash.

    NaN and infinity become a dash rather than the tokens `nan` / `inf`, which
    a reader would take for a real number. Floats are rendered with `g` so a
    ratio and a figure in the hundreds of millions are both legible without a
    per-metric rule.
    """
    if value is None:
        return DASH
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return DASH if not math.isfinite(value) else f"{value:.6g}"
    text = str(value)
    return text if text.strip() else DASH


def _metrics_section(metrics: dict | None) -> str:
    """The metric table: one column per block, rows derived from the data.

    A key missing from one block is a dash in that column; a key present in no
    block produces no row; a block that is None becomes a full column of
    dashes rather than being dropped, so the reader can see it exists and is
    empty. It is an HTML table, not a plotly `go.Table`, because every trace on
    the page must carry a `name` for the persisted-report locks to parse it.
    """
    if not metrics:
        return ""
    present = [name for name in _BLOCKS if name in metrics]
    if not present:
        return ""

    columns = {
        name: _flatten(metrics[name]) if isinstance(metrics.get(name), dict) else {}
        for name in present
    }
    rows: list[str] = []
    seen: list[str] = []
    for column in columns.values():
        for key in column:
            if key not in seen:
                seen.append(key)
    for key in seen:
        cells = "".join(
            f"<td>{_escape(_cell(columns[name].get(key)))}</td>" for name in present
        )
        rows.append(f"      <tr><th>{_escape(key)}</th>{cells}</tr>")
    if not rows:
        return ""

    headers = "".join(f"<th>{_escape(name)}</th>" for name in present)
    return (
        "  <h2>Metrics</h2>\n"
        '  <table class="metrics">\n'
        f"    <thead><tr><th>metric</th>{headers}</tr></thead>\n"
        "    <tbody>\n"
        + "\n".join(rows)
        + "\n    </tbody>\n"
        "  </table>\n"
    )


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
    title: str,
    summary: dict[str, str] | None,
    metrics: dict | None,
    div: str,
    notes: list[str] | None,
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
        f"{_metrics_section(metrics)}"
        f"{_notes_section(notes)}"
        "</body>\n"
        "</html>\n"
    )
