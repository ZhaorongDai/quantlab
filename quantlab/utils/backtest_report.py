"""Interactive HTML report for one backtest run (03.7 D-23, D-21, D-08).

One self-contained page built around a plotly div:

1. an escaped `<h1>` of the run name;
2. a dates-and-setup block stating the window, the bar count and interval, the
   training window(s), the in-sample range(s) and the out-of-sample ranges in
   words, so the reader never has to open `metrics.json` to learn what the
   picture covers;
3. a metric table with one column per metrics block (whole / in-sample /
   out-of-sample), plus an `out_of_sample - in_sample` column whenever both
   slice blocks are present (03.8 D-01). Which rows get a number there is
   decided by the values' types alone -- finite real non-booleans -- so it,
   too, names no metric. The column is report-only (D-05);
4. the figure: three rows on a shared time axis -- equity (with a pair of
   triangles marking the deepest drawdown's valley and the bar it recovered)
   on top, drawdown below it, per-calendar-month returns at the bottom
   (green bars for a gain, red for a loss, matching the heatmap below) --
   plus a log/linear toggle for the equity axis. Forced liquidations are NOT
   drawn: quick 260916-hro removed those markers from the chart, while the
   records themselves still persist to the run's `liquidations.json`;
5. a year-by-month heatmap of the same compounded monthly returns (03.8
   D-04), rendered as a SECOND plotly div rather than a fourth subplot row
   (see `_monthly_heatmap_div`), and omitted when there are no returns. It is
   red for a loss, green for a gain and grey at zero (G-03.8-1);
6. the notes.

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

The row-1 axis title is therefore just `value`. It used to carry a
parenthetical announcing the hover text, but that merely described the
`customdata` the trace already shows, so no information was lost when quick
260916-hro dropped it -- and the rotated title got back the vertical room it
needs.

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
import numbers
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

#: Header of the fourth metric-table column (03.8 D-01). It states the
#: ARITHMETIC rather than passing judgement: the table cannot know whether a
#: larger number is better for a metric it has never seen, so it states the
#: operation and lets the reader judge. That is what makes differencing ratio
#: metrics defensible.
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
    - `init_cash`: starting capital, used only to express equity as a multiple
      in the hover text;
    - `drawdown_span`: a mapping describing the DEEPEST drawdown, with keys
      `valley` and `end` (bar labels), `bars` (the number of bars from the
      valley to the end), `depth` (a negative float) and `recovered` (bool).
      It is drawn as an up triangle at the VALLEY -- the deepest bar of that
      drawdown -- and a down triangle at the bar it recovered, so the pair
      spans bottom-back-to-even rather than the whole episode. The caller
      selects the episode and measures it; this module draws the one it is
      given and never picks one. An endpoint the equity axis does not carry,
      or a key that is absent, drops that marker rather than raising.

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
    # `height` is load-bearing, not decoration (quick 260916-hro): a y-axis
    # title is rotated 90 degrees, so its rendered length is measured against
    # the axis HEIGHT, not the width. With no explicit height the div falls
    # back to plotly's 450px default; the top and bottom margins take 240 of
    # that, and `row_heights` then leaves rows 2 and 3 at roughly 44px each --
    # shorter than `drawdown` and `monthly return` render, which is what made
    # the three titles collide. At 900 the plotting area is
    # 900 - 100 - 140 = 660px, so the rows are about 328 / 140 / 140px and
    # every title fits. A left margin would not have helped: the collision is
    # between vertically stacked titles, not between a title and its ticks.
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


#: The deepest drawdown's two triangles. Both ends SHARE one colour because
#: they are the two ends of a single measurement -- the valley and the
#: recovery of one episode -- and are told apart by shape, up versus down.
#:
#: The value is unchanged from when it was picked to differ from the
#: forced-liquidation red: those markers were removed from the chart in quick
#: 260916-hro (the records still persist to the run's `liquidations.json`), so
#: that contrast no longer exists on the page. Kept as it was rather than
#: re-picked, to avoid visual churn nobody asked for.
SPAN_COLOUR = "#8e44ad"


def _add_drawdown_span(fig, equity: pd.Series, span) -> None:
    """Triangles on the equity row at the deepest drawdown's valley and end.

    The caller has already chosen the episode and measured it; this draws the
    one it is given and states its numbers in the hover text.

    The pair spans VALLEY to recovery, not start to recovery: the up triangle
    sits on the deepest bar of that drawdown and the down triangle on the bar
    it recovered, so the distance between them is how long it took to get from
    the bottom back to even.

    `bars` is a BAR COUNT -- trading days on a daily panel. The text therefore
    says trading days, and no calendar duration is rendered here: the time
    axis spans more calendar days than the span lasts bars, so a reader who
    measured the axis against a timedelta would be misled. That count is NOT
    the metric named Max Drawdown Duration, which measures the LONGEST
    drawdown and counts from where that drawdown began.

    Every key is read with `.get`, and an endpoint the equity axis does not
    carry drops only its own marker rather than raising: the report is the
    last step of a run that already succeeded and is written inside that run's
    staging directory, so an exception here would delete the ENTIRE run rather
    than merely losing the markers.

    Two traces rather than one, each with its own `name`: the ends say
    different things -- only the end marker can report that the drawdown never
    recovered -- and the persisted-report locks parse traces by name.
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
    """Per-calendar-month compounded return of `returns`, indexed by period.

    The single place the monthly compounding rule lives: the bar row and the
    year-by-month heatmap both consume it, so the two panels of the same
    numbers cannot drift apart.

    Grouped by converting the index to monthly periods (`to_period` with the
    month frequency) rather than with a resample alias: the monthly alias was
    renamed (`M` -> `ME`) across pandas versions while `to_period` reads the
    same in both.

    The NaN drop is load-bearing: `prod()` over `1 + NaN` skips the NaN, so a
    month holding only NaN bars would compound to a flat `0.0` -- a fabricated
    month indistinguishable at read time from a real one. Dropping first keeps
    such a month absent instead.

    Returns None when there is nothing to compute (no returns, or none left
    after the drop).
    """
    if returns is None:
        return None
    series = returns.to_pandas().dropna()
    if series.empty:
        return None
    index = pd.DatetimeIndex(series.index)
    return (1.0 + series).groupby(index.to_period("M")).prod() - 1.0


def _add_monthly_returns(fig, returns: xr.DataArray | None) -> None:
    """Per-calendar-month compounded return of `returns`, as bars.

    The numbers come from `_monthly_series`. A short window legitimately
    produces one or two bars -- that is the point, since it shows at a glance
    that a run's whole P&L landed in a single month.

    Each bar is coloured by its sign through `_sign_colour`, from the same
    constants the heatmap's colorscale is built from (G-03.8-1), so a month
    reads the same colour in both panels.
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
#: year-by-month heatmap (G-03.8-1).
#:
#: Green means up and red means down. The user chose this Western convention
#: on 2026-09-19; it is deliberately NOT the East-Asian red-up convention.
#:
#: Both panels read these constants, so the two views of the same numbers
#: cannot disagree about colour.
#:
#: The midpoint is a neutral grey, not a hue: a diverging scale's centre must
#: read as "nothing happened", and the heatmap's `zmid=0.0` pins zero to it.
#:
#: The two poles have matched luminance (about 0.19 each), so neither sign
#: looks heavier. The pair passes the dataviz palette validator, including
#: red-green colour-vision separation (deutan dE 9.4, above the 8 target).
#: That matters because red/green is the classic colour-blind confusion pair.
#: Sign is also carried without colour: bars point up or down, and every
#: heatmap cell states its percentage on hover.
GAIN_COLOUR = "#1b8a5a"
LOSS_COLOUR = "#e03b30"
NEUTRAL_COLOUR = "#f0efec"

#: The heatmap's diverging colorscale, built from the three constants above:
#: the most negative value is red, zero (via `zmid=0.0`) is grey, the most
#: positive value is green.
RETURN_COLOURSCALE = (
    (0.0, LOSS_COLOUR),
    (0.5, NEUTRAL_COLOUR),
    (1.0, GAIN_COLOUR),
)


def _sign_colour(value: float) -> str:
    """The colour of one monthly return's sign: gain, loss, or neutral.

    Above 0 is `GAIN_COLOUR`, below 0 is `LOSS_COLOUR`, and exactly 0 is
    `NEUTRAL_COLOUR`. The zero rule follows the heatmap: a month that
    compounds to exactly 0.0 (one spent wholly in cash, say) lands on the
    heatmap's grey midpoint under `zmid=0`, so its bar takes the same grey.
    The bar has zero height, so the grey is invisible there, but pinning the
    rule keeps the two panels identical by construction.

    A NaN fails both comparisons and falls through to neutral rather than
    raising. `_monthly_series` already drops NaN, so none should arrive, but
    this module must never raise: the report is written inside the run's
    staging directory, where an exception deletes the ENTIRE run.
    """
    if value > 0:
        return GAIN_COLOUR
    if value < 0:
        return LOSS_COLOUR
    return NEUTRAL_COLOUR


#: The heatmap's month columns, in calendar order. Two-digit strings so they
#: sort and read the same way, and so plotly treats them as categories.
MONTH_LABELS = [f"{month:02d}" for month in range(1, 13)]

#: Caption above the heatmap div. A plain string, yet still escaped on the way
#: onto the page like every other non-plotly string (T-03.8-03-04).
HEATMAP_CAPTION = "Monthly returns by year"


def _monthly_grid(monthly: pd.Series) -> tuple[list[int], list[list[float | None]]]:
    """`monthly` (indexed by month periods) -> `(years, z)`, one row per year.

    `years` is the sorted set of years the series covers; `z` holds one row of
    12 cells per year, initialised to None, so a month the run did not cover
    stays None -- it serializes to JSON null and renders as an empty cell,
    never as a fabricated 0.0.

    `period.month` is 1-BASED, so the column index is `period.month - 1`. That
    `- 1` is the whole correctness question here, and it needs a value-level
    lock: an off-by-one still renders every cell, still looks like a heatmap
    and raises nothing (except on December) -- the numbers are simply in the
    wrong month. Rows are allocated from the series' own years, so no write
    can land outside the grid.
    """
    years = sorted({period.year for period in monthly.index})
    row_of = {year: row for row, year in enumerate(years)}
    grid: list[list[float | None]] = [[None] * 12 for _ in years]
    for period, value in monthly.items():
        grid[row_of[period.year]][period.month - 1] = float(value)
    return years, grid


def _monthly_heatmap_div(returns: xr.DataArray | None) -> str:
    """A year-by-month heatmap of compounded monthly returns, as its own div.

    Returns an HTML fragment, or the empty string when there is nothing to
    draw. Nothing-to-draw is an early return, never an exception, mirroring
    the bar row: the report is written inside the run's staging directory,
    where an exception deletes the ENTIRE run.

    The numbers come from `_monthly_series`, the same helper the bar row
    uses, so the two panels cannot disagree. The bars answer "when did the
    P&L land" on the shared time axis; the grid answers "which months of
    which years were good" (03.8 D-04 keeps both).

    The palette is `RETURN_COLOURSCALE`: red for a loss month, green for a
    gain month, and a neutral grey at zero, which `zmid=0.0` pins to the
    scale's midpoint whatever the run's range.

    **It is a SEPARATE figure, never a fourth subplot row.** The main figure
    is three subplot rows with `shared_xaxes=True`; a fourth row makes plotly
    set `matches='x4'` on the three datetime x axes, binding the equity,
    drawdown and monthly-bar axes to this chart's CATEGORICAL month axis --
    `shared_xaxes` is figure-wide with no per-row opt-out. It would also force
    re-deriving the main figure's height and per-row pixel budget. Keep it
    here.

    `include_plotlyjs=False`: the main div already loads plotly.js from the
    CDN, so this adds a few kilobytes rather than a second library copy.
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


def _delta(later: object, earlier: object) -> object:
    """`later - earlier` when both are finite real non-booleans, else None.

    Dispatches on TYPE and never on metric name: a rule keyed to a name would
    be dead the day the metric set is replaced, and this table renders
    whatever mapping arrives. The operands are the RAW objects the report
    receives (live pandas / numpy scalars), not the JSON shape metrics.json
    is later written in.

    `bool` is tested first because it is an int subclass: `True - False == 1`
    would render as a plausible number. `numbers.Real` rather than
    `(int, float)` admits numpy scalars (`np.int64` is not an int) without
    this leaf importing numpy, while still refusing `pd.Timedelta`,
    `pd.Timestamp`, `pd.NaT`, strings and None -- every one of which either
    raises on subtraction or yields something that is not a difference of two
    metric values. `np.bool_` is not a `numbers.Real`, so it needs no case.

    It must never raise: it runs inside the run's staging directory, where an
    exception deletes the entire run. Returning None lets `_cell` render the
    dash; this function never formats one itself.
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
    # Gate on membership, not truthiness: a None block is still present (as an
    # empty column), and must give dashed deltas rather than drop the column.
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
    heatmap: str = "",
) -> str:
    """One self-contained HTML document around the plotly `div`.

    `div` and `heatmap` are plotly's own fragments and are inserted verbatim
    -- plotly owns their escaping. Everything else on the page comes from the
    run and is escaped. The heatmap goes between the main figure and the
    metric table (picture, picture, numbers, notes); an empty `heatmap`
    inserts nothing, not even its caption.
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
