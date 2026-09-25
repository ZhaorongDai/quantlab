"""HTML report writer for one backtest run.

``write_backtest_report`` renders a single self-contained page for a run: a
title, a dates-and-setup table, a plotly figure with the equity curve, the
drawdown and the compounded monthly returns on a shared time axis, a
year-by-month heatmap of those monthly returns, a metric table with one column
per window slice, and the notes. The in-sample range is shaded across the
figure and the deepest drawdown is marked by a pair of triangles on the
equity curve.

The module knows no metric name: the table's rows are derived from whatever
mapping it is given, and any value it cannot render becomes a dash. That is
deliberate, because the report is written inside a run's staging directory,
where an exception would discard the whole run rather than just the page. For
the same reason every string that comes from the run passes through
``html.escape`` before it reaches the page, and plotly.js is loaded from its
CDN so a run directory stays a few kilobytes (viewing the page needs network
access). This module imports only the standard library, pandas, xarray and
plotly.
"""

import html
import math
import numbers
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

__all__ = ["write_backtest_report"]

#: Shown in place of a value the run does not have. An em dash rather than a
#: hyphen so it cannot be read as the minus sign of a negative number.
DASH = "—"

#: The window slices the metric table shows, in column order. These are slice
#: keys, not metric names: the rows inside each column are whatever that slice
#: turns out to carry.
_BLOCKS = ("whole", "in_sample", "out_of_sample")

#: Header of the difference column. It names the arithmetic rather than a
#: judgement: the table cannot know whether larger is better for a metric it
#: has never seen, so it states the operation and lets the reader decide.
DELTA_HEADER = "out_of_sample - in_sample"

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
    init_cash: float | None = None,
    drawdown_span: dict | None = None,
) -> None:
    """Write the HTML report for one backtest run to ``path``.

    Drawdown is ``value / running max - 1``: zero at a new high and negative
    below it. Everything after ``title`` is optional; a section whose input
    is missing is simply left off the page.

    Args:
        value: Portfolio value on the ``timestamp`` dimension. It is drawn as
            is, so the page and the persisted equity carry the same numbers.
        path: The ``report.html`` to write.
        in_sample_range: ``(first, last)`` bar labels of the in-sample part of
            the window, shaded grey, or ``None`` for a fully out-of-sample
            window. A midnight bar is labelled by its ISO date and any other
            bar by its full ISO timestamp; plotly reads both.
        notes: Lines printed below the plot, such as what the simulation does
            not model.
        title: Page title, typically the run directory name.
        summary: Ordered mapping of display label to already-formatted text,
            rendered as the dates-and-setup table. Nothing is computed or
            formatted here, so the page can state the same strings the run's
            ``metrics.json`` carries.
        metrics: The metrics mapping of one curve, whose ``whole``,
            ``in_sample`` and ``out_of_sample`` entries become the table's
            columns. An entry may be absent or ``None``; the rows are derived
            from the keys present, with nested dicts flattened to dotted
            paths. When both slice columns exist a fourth column shows
            ``out_of_sample - in_sample`` wherever both values are finite
            numbers.
        returns: Per-bar portfolio returns on ``timestamp``, compounded per
            calendar month for the bar panel and the heatmap.
        init_cash: Starting capital, used only to show equity as a multiple of
            it in the hover text.
        drawdown_span: The deepest drawdown as a dict with ``valley`` and
            ``end`` (bar labels), ``bars`` (bars from the valley to the end),
            ``depth`` (a negative float) and ``recovered`` (bool). It is drawn
            as an up triangle at the valley and a down triangle at the
            recovery bar, so the pair spans bottom-back-to-even rather than
            the whole episode. The caller chooses the episode; an endpoint the
            equity axis does not carry drops that marker rather than raising.

    Example:
        >>> import pandas as pd, xarray as xr
        >>> ts = pd.bdate_range("2024-01-01", periods=5)
        >>> value = xr.DataArray([100.0, 104.0, 98.0, 103.0, 110.0],
        ...                      dims=("timestamp",), coords={"timestamp": ts})
        >>> write_backtest_report(
        ...     value,
        ...     "report.html",
        ...     in_sample_range=("2024-01-01", "2024-01-02"),
        ...     notes=["No borrow cost is modelled."],
        ...     title="demo_run",
        ...     metrics={"whole": {"Total Return [%]": 10.0},
        ...              "in_sample": {"Total Return [%]": 4.0},
        ...              "out_of_sample": {"Total Return [%]": 6.0}},
        ...     init_cash=100.0,
        ... )
        >>> "<h1>demo_run</h1>" in open("report.html").read()
        True
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
    _add_drawdown_span(fig, equity, drawdown_span)
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
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="drawdown", tickformat=".1%", row=2, col=1)
    fig.update_yaxes(title_text="monthly return", tickformat=".1%", row=3, col=1)
    # The explicit height matters: a y-axis title is rotated, so its length is
    # measured against the row height. At plotly's 450px default the margins
    # leave rows 2 and 3 about 44px each, shorter than the titles "drawdown"
    # and "monthly return", and the three titles collide. At 900px the rows
    # are roughly 328, 140 and 140px and every title fits.
    fig.update_layout(
        height=900,
        margin={"b": 140},
        showlegend=False,
        updatemenus=[_axis_toggle()],
    )

    div = fig.to_html(full_html=False, include_plotlyjs="cdn")
    heatmap = _monthly_heatmap_div(returns)
    Path(path).write_text(
        _document(title, summary, metrics, div, notes, heatmap), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def _add_equity(fig, equity: pd.Series, init_cash: float | None) -> None:
    """Add the equity trace, with the multiple of ``init_cash`` in the hover text.

    ``y`` stays the persisted portfolio value so the page cannot disagree with
    the stored equity; the multiple of initial capital, which makes a run that
    compounded by orders of magnitude legible, rides along as ``customdata``.
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


#: Colour of both drawdown triangles. They are the two ends of one measurement
#: (the valley and the recovery of a single episode) and are told apart by
#: shape, up versus down, not by colour.
SPAN_COLOUR = "#8e44ad"


def _add_drawdown_span(fig, equity: pd.Series, span) -> None:
    """Add the two triangles marking the deepest drawdown to the equity row.

    The caller has chosen and measured the episode; this only draws it. The
    up triangle sits on the valley and the down triangle on the recovery bar,
    so the distance between them is the time from the bottom back to even.
    ``bars`` is a bar count (trading days on a daily panel) and the hover text
    says so; no calendar duration is shown, since the time axis spans more
    calendar days than the span lasts bars. Every key is read with ``.get``
    and an endpoint missing from the equity axis drops only its own marker,
    because an exception here would discard the whole run directory. The two
    markers are separate named traces: only the end marker can say that the
    drawdown never recovered.
    """
    if not span:
        return

    bars = span.get("bars")
    depth = span.get("depth")
    length = "an unknown number of" if bars is None else str(bars)
    depth_text = "" if depth is None else f"<br>depth {float(depth):.2%}"

    if span.get("recovered"):
        tail = (
            "deepest drawdown recovers here"
            f"<br>{length} trading days (bars) from its deepest point"
        )
    else:
        tail = (
            "deepest drawdown had not recovered by the last bar"
            f"<br>{length} trading days (bars) since its deepest point"
        )

    endpoints = (
        (
            "valley",
            "deepest_drawdown_valley",
            "triangle-up",
            "deepest drawdown bottoms here",
        ),
        ("end", "deepest_drawdown_end", "triangle-down", tail),
    )
    for key, name, marker_symbol, text in endpoints:
        label = span.get(key)
        if label is None:
            continue
        stamp = pd.Timestamp(str(label))
        if stamp not in equity.index:
            continue
        suffix = depth_text if key == "end" else ""
        fig.add_trace(
            go.Scatter(
                x=[stamp],
                y=[equity.loc[stamp]],
                name=name,
                mode="markers",
                marker={"symbol": marker_symbol, "size": 11, "color": SPAN_COLOUR},
                hovertemplate=f"%{{x}}<br>{text}{suffix}<extra></extra>",
            ),
            row=1,
            col=1,
        )


def _monthly_series(returns: xr.DataArray | None) -> pd.Series | None:
    """Return the compounded return of each calendar month, indexed by period.

    Both the monthly bar panel and the heatmap read this one series, so they
    cannot drift apart. Months are formed with ``to_period("M")`` rather than
    a resample alias, whose spelling changed across pandas versions. NaN bars
    are dropped first: ``prod`` skips NaN, so a month of only NaN bars would
    otherwise compound to a fabricated ``0.0`` instead of being absent.
    Returns ``None`` when there is nothing to compute.
    """
    if returns is None:
        return None
    series = returns.to_pandas().dropna()
    if series.empty:
        return None
    index = pd.DatetimeIndex(series.index)
    return (1.0 + series).groupby(index.to_period("M")).prod() - 1.0


def _add_monthly_returns(fig, returns: xr.DataArray | None) -> None:
    """Add the per-calendar-month compounded returns as a bar row.

    A short window legitimately yields one or two bars; that shows at a glance
    that a run's whole profit landed in a single month. Each bar is coloured
    by its sign with the same constants as the heatmap, so a month reads the
    same colour in both panels.
    """
    monthly = _monthly_series(returns)
    if monthly is None or monthly.empty:
        return
    fig.add_trace(
        go.Bar(
            x=[period.to_timestamp() for period in monthly.index],
            y=monthly.values,
            marker={"color": [_sign_colour(value) for value in monthly.values]},
            name="monthly_return",
            hovertemplate="%{x|%Y-%m}<br>%{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
    )


#: The colours of a monthly return's sign, shared by the monthly bars and the
#: heatmap so the two views of the same numbers cannot disagree. Green is a
#: gain and red a loss (the Western convention). The midpoint is a neutral
#: grey, which the heatmap pins to zero with ``zmid=0.0``. The two poles have
#: matched luminance and stay separable under red-green colour-vision
#: deficiency; sign is also carried without colour, by bar direction and by
#: the hover percentage.
GAIN_COLOUR = "#1b8a5a"
LOSS_COLOUR = "#e03b30"
NEUTRAL_COLOUR = "#f0efec"

#: The heatmap's diverging colorscale: the most negative value is red, zero
#: (through ``zmid=0.0``) is grey, the most positive value is green.
RETURN_COLOURSCALE = (
    (0.0, LOSS_COLOUR),
    (0.5, NEUTRAL_COLOUR),
    (1.0, GAIN_COLOUR),
)


def _sign_colour(value: float) -> str:
    """Return the colour for one monthly return: gain, loss or neutral at zero.

    Exactly zero takes the neutral grey so that a month spent wholly in cash
    matches the heatmap's zero cell. A NaN fails both comparisons and falls
    through to neutral rather than raising.
    """
    if value > 0:
        return GAIN_COLOUR
    if value < 0:
        return LOSS_COLOUR
    return NEUTRAL_COLOUR


#: The heatmap's month columns, as two-digit strings so they sort and read the
#: same way and plotly treats them as categories.
MONTH_LABELS = [f"{month:02d}" for month in range(1, 13)]

#: Caption above the heatmap. Escaped on the way onto the page like every
#: other non-plotly string.
HEATMAP_CAPTION = "Monthly returns by year"


def _monthly_grid(monthly: pd.Series) -> tuple[list[int], list[list[float | None]]]:
    """Return ``(years, z)``: one row of twelve cells per year in ``monthly``.

    A month the run did not cover stays ``None``, which serialises to JSON
    null and renders as an empty cell rather than a fabricated ``0.0``.
    ``period.month`` is 1-based, hence the ``- 1`` on the column index.
    """
    years = sorted({period.year for period in monthly.index})
    row_of = {year: row for row, year in enumerate(years)}
    grid: list[list[float | None]] = [[None] * 12 for _ in years]
    for period, value in monthly.items():
        grid[row_of[period.year]][period.month - 1] = float(value)
    return years, grid


def _monthly_heatmap_div(returns: xr.DataArray | None) -> str:
    """Return the year-by-month returns heatmap as its own HTML fragment.

    Returns the empty string when there is nothing to draw. The numbers come
    from ``_monthly_series``, the same helper the bar row uses. It is a
    separate figure rather than a fourth subplot row because ``shared_xaxes``
    is figure-wide: a fourth row would bind the three datetime axes to this
    chart's categorical month axis. ``include_plotlyjs=False`` reuses the
    library the main div already loads.
    """
    monthly = _monthly_series(returns)
    if monthly is None or monthly.empty:
        return ""
    years, z = _monthly_grid(monthly)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=MONTH_LABELS,
            y=[str(year) for year in years],
            name="monthly_return_heatmap",
            colorscale=RETURN_COLOURSCALE,
            zmid=0.0,
            colorbar={"tickformat": ".1%"},
            hoverongaps=False,
            hovertemplate="%{y}-%{x}<br>%{z:.2%}<extra></extra>",
        )
    )
    # About 36px per year row plus room for the axis labels and margins, with
    # a floor so a one-year run is still a legible strip.
    fig.update_layout(
        height=max(220, 120 + 36 * len(years)),
        margin={"t": 20, "b": 40},
    )
    fig.update_xaxes(type="category", title_text="month")
    fig.update_yaxes(type="category", autorange="reversed", title_text="year")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _axis_toggle() -> dict:
    """Return the linear/log button group for the equity axis.

    Linear is the default because a run that lost everything has a
    non-positive value that renders an empty log panel. Log is one click away
    and turns a curve that compounded by orders of magnitude from a flat line
    with a final spike into a readable slope.
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
    """Return ``str(value)`` with every HTML-significant character escaped."""
    return html.escape(str(value))


def _flatten(value: dict, prefix: str = "") -> dict:
    """Flatten nested dicts into ``{"dotted.path": leaf}``.

    A sub-dict becomes rows under its own name, so the table needs no
    knowledge of any metric name.
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
    """Render one metric value as text; anything unrenderable becomes a dash.

    NaN and infinity become a dash rather than the tokens ``nan`` / ``inf``,
    which read like numbers. Floats use the ``g`` format so a ratio and a
    figure in the hundreds of millions are both legible without a per-metric
    rule.
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


def _delta(later: object, earlier: object) -> object:
    """Return ``later - earlier`` when both are finite real numbers, else ``None``.

    The check is on type alone, never on a metric name. ``bool`` is rejected
    first because it is an ``int`` subclass and ``True - False`` would render
    as a plausible number. ``numbers.Real`` admits numpy scalars without
    importing numpy while refusing timestamps, timedeltas, strings and
    ``None``, each of which either raises on subtraction or yields something
    that is not a difference of two metric values. Returning ``None`` lets
    ``_cell`` render the dash.
    """
    for value in (later, earlier):
        if isinstance(value, bool):
            return None
        if not isinstance(value, numbers.Real):
            return None
        if not math.isfinite(value):
            return None
    return later - earlier


def _metrics_section(metrics: dict | None) -> str:
    """Render the metric table: one column per slice, rows derived from the data.

    A key missing from one slice is a dash in that column; a key present in no
    slice makes no row; a slice that is ``None`` is a full column of dashes so
    the reader sees it exists and is empty. Returns the empty string when there
    is nothing to show. It is an HTML table rather than a plotly table so that
    every trace on the page keeps a ``name``.
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
    # Gate on membership, not truthiness: a None slice is still present (as an
    # empty column) and must give dashed deltas rather than drop the column.
    show_delta = "in_sample" in present and "out_of_sample" in present
    for key in seen:
        cells = "".join(
            f"<td>{_escape(_cell(columns[name].get(key)))}</td>" for name in present
        )
        if show_delta:
            delta = _delta(
                columns.get("out_of_sample", {}).get(key),
                columns.get("in_sample", {}).get(key),
            )
            cells += f"<td>{_escape(_cell(delta))}</td>"
        rows.append(f"      <tr><th>{_escape(key)}</th>{cells}</tr>")
    if not rows:
        return ""

    headers = "".join(f"<th>{_escape(name)}</th>" for name in present)
    if show_delta:
        headers += f"<th>{_escape(DELTA_HEADER)}</th>"
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
    """Render the dates-and-setup table, or the empty string when there is none."""
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
    """Render the notes list, or the empty string when there are none."""
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
    heatmap: str = "",
) -> str:
    """Assemble the full HTML document around the plotly fragments.

    ``div`` and ``heatmap`` are plotly's own fragments and are inserted
    verbatim; everything else comes from the run and is escaped. The heatmap
    sits between the main figure and the metric table, and an empty
    ``heatmap`` inserts nothing, not even its caption.
    """
    heatmap_section = (
        f"  <h2>{_escape(HEATMAP_CAPTION)}</h2>\n{heatmap}\n" if heatmap else ""
    )
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
        f"{heatmap_section}"
        f"{_metrics_section(metrics)}"
        f"{_notes_section(notes)}"
        "</body>\n"
        "</html>\n"
    )
