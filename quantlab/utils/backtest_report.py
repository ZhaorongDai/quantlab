"""HTML report writer for one backtest run.

``write_backtest_report`` renders a single self-contained page for a run. A
row of headline cards sits on top; below it, the metric tables are on the
left and the charts on the right, in tabs. The tables compare the strategy
with the benchmark by group (returns, risk, risk-adjusted), list the excess
over the benchmark and the trading statistics, and, when the run has an
in-sample part, set the in-sample and out-of-sample slices side by side.
*In-sample* means the bars the model was trained on; *out-of-sample* means
the bars it never saw, which are the honest test, so the cards and the main
tables show the out-of-sample slice whenever the run has both. The tabs are
Performance (equity, drawdown, monthly returns and their year-by-month
heatmap), Excess (cumulative excess return and excess drawdown), Rolling
(one-year rolling statistics) and Portfolio (turnover, holdings and exposure
per rebalance). ``backtest_report_figure`` returns the Performance chart
alone, as a plotly figure.

The module holds a catalogue of the metrics it knows, with each one's
group, label, unit and definition, but it renders every metric it is given:
a key missing from the catalogue appears in an "Other" table with a unit
guessed from its name, and a value it cannot render becomes a dash. That is
deliberate, because the report is written inside a run's staging directory,
where an exception would discard the whole run rather than just the page.
For the same reason every string that comes from the run passes through
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

__all__ = ["backtest_report_figure", "write_backtest_report"]

#: Shown in place of a value the run does not have. An em dash rather than a
#: hyphen so it cannot be read as the minus sign of a negative number.
DASH = "—"

_STYLE = """
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
         margin: 24px; color: #1a1a1a; }
  h1 { font-size: 20px; margin: 0 0 12px 0; word-break: break-all; }
  h2 { font-size: 15px; margin: 20px 0 8px 0; color: #444; }
  .kpis { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 8px 0; }
  .kpi { border: 1px solid #e5e5e5; border-radius: 8px; padding: 8px 12px; min-width: 120px; }
  .kl { font-size: 11px; color: #777; text-transform: uppercase; letter-spacing: .04em; }
  .kv { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .kv.pos { color: #1b8a5a; } .kv.neg { color: #e03b30; }
  .ks { font-size: 11px; color: #777; }
  .layout { display: flex; flex-wrap: wrap; gap: 28px; align-items: flex-start; }
  .tables { flex: 0 1 440px; min-width: 320px; max-width: 100%; overflow-x: auto; }
  .charts { flex: 1 1 720px; min-width: 0; }
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid #e5e5e5; margin-top: 20px; }
  .tab { border: 0; background: none; padding: 6px 12px; cursor: pointer; color: #555;
         border-bottom: 2px solid transparent; font-size: 13px; }
  .tab.on { color: #1a1a1a; border-bottom-color: #1f77b4; }
  .pane { display: none; } .pane.on { display: block; }
  table.summary { border-collapse: collapse; font-size: 13px; }
  table.summary th { text-align: left; padding: 3px 16px 3px 0;
                     font-weight: 600; color: #444; white-space: nowrap; }
  table.summary td { padding: 3px 0; font-variant-numeric: tabular-nums;
                     overflow-wrap: anywhere; }
  table.metrics { border-collapse: collapse; font-size: 13px; }
  table.metrics th, table.metrics td { padding: 3px 16px 3px 0;
                                       border-bottom: 1px solid #eee;
                                       white-space: nowrap; }
  table.metrics thead th { text-align: right; color: #444; }
  table.metrics thead th:first-child { text-align: left; }
  table.metrics tbody th { text-align: left; font-weight: 400; color: #333;
                           cursor: help; text-decoration: underline dotted #bbb; }
  table.metrics td { text-align: right; font-variant-numeric: tabular-nums; }
  table.metrics td.better { font-weight: 600; color: #146c43; background: #e8f5ee; }
  table.metrics tr.group td { text-align: left; font-size: 11px; color: #777;
                              text-transform: uppercase; letter-spacing: .05em;
                              padding-top: 10px; border-bottom: 1px solid #ccc; }
  ul.notes { font-size: 13px; color: #444; padding-left: 20px; }
  .timeline { font-size: 12px; color: #333; }
  .timeline .caption { color: #555; margin: 0 0 6px; }
  .timeline svg text { font-size: 10px; fill: #555; }
  .timeline .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: #555; margin-top: 4px; }
  .timeline .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                  vertical-align: -1px; margin-right: 4px; }
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
    benchmark_value: xr.DataArray | None = None,
    benchmark_returns: xr.DataArray | None = None,
    benchmark_name: str = "benchmark",
    weights: xr.DataArray | None = None,
    turnover: xr.DataArray | None = None,
    bars_per_year: float | None = None,
    windows: dict | None = None,
) -> None:
    """Write the HTML report for one backtest run to ``path``.

    The page opens with a row of headline numbers, then the tables on the
    left (a timeline of the run's windows, the setup, strategy against the
    benchmark by group, the excess over it,
    trading, and in-sample against out-of-sample when the run has an
    in-sample part) and the charts on the right in tabs: Performance (NAV,
    drawdown, monthly returns and the monthly heatmap), Excess (cumulative
    excess return with a log / arithmetic toggle, and the excess drawdown),
    Rolling (one-year panels) and Portfolio (turnover, holdings and
    exposure per rebalance). The headline numbers and the main tables are
    the out-of-sample slice when the run has an in-sample part, else the
    whole window. Drawdown is ``value / running max - 1``, negative below a
    peak, in every table, card and chart. Everything after ``title`` is
    optional; a section, card or tab whose input is missing is left off the
    page, and a metric the page does not know is still shown, in an "Other"
    table.

    Parameters
    ----------
    value : xr.DataArray
        Portfolio value on the ``timestamp`` dimension. It is drawn as
        is, so the page and the persisted equity carry the same numbers.
    path : str | Path
        The ``report.html`` to write.
    in_sample_range : tuple[str, str] | None
        ``(first, last)`` bar labels of the in-sample part of
        the window, shaded grey, or ``None`` for a fully out-of-sample
        window. A midnight bar is labelled by its ISO date and any other
        bar by its full ISO timestamp; plotly reads both.
    notes : list[str]
        Lines listed in the page's Notes section, such as what the
        simulation does not model.
    title : str
        Page title, typically the run directory name.
    summary : dict[str, str] | None
        Ordered mapping of display label to already-formatted text,
        rendered as the dates-and-setup table. Nothing is computed or
        formatted here, so the page can state the same strings the run's
        ``metrics.json`` carries.
    metrics : dict | None
        The metrics mapping of one curve: the slices ``whole``,
        ``in_sample`` and ``out_of_sample`` (each may be absent or
        ``None``), with a benchmark ``benchmark`` and ``relative`` holding
        the same slices, and ``execution`` and ``portfolio_construction``
        for the trading table.
    returns : xr.DataArray | None
        Per-bar portfolio returns on ``timestamp``, compounded per
        calendar month for the bar panel and the heatmap.
    init_cash : float | None
        Starting capital, used only to show equity as a multiple of
        it in the hover text.
    drawdown_span : dict | None
        The deepest drawdown as a dict with ``valley`` and
        ``end`` (bar labels), ``bars`` (bars from the valley to the end),
        ``depth`` (a negative float) and ``recovered`` (bool). It is drawn
        as an up triangle at the valley and a down triangle at the
        recovery bar, so the pair spans bottom-back-to-even rather than
        the whole episode. The caller chooses the episode; an endpoint the
        equity axis does not carry drops that marker rather than raising.
    benchmark_value : xr.DataArray | None
        The benchmark's portfolio value on the same ``timestamp`` axis,
        started from the same capital. When given, the page draws it with
        the portfolio's NAV and adds the excess-return and excess-drawdown
        rows; ``metrics["relative"]`` and ``metrics["benchmark"]`` become
        their own tables.
    benchmark_returns : xr.DataArray | None
        The benchmark's per-bar returns, drawn beside the portfolio's
        monthly returns.
    benchmark_name : str
        Display name of the benchmark in the legend, the headings and the
        hover text.
    weights : xr.DataArray | None
        Target weights on ``(timestamp, symbol)``, an all-NaN row on a bar
        that holds; the Portfolio tab draws the holdings and exposure of
        every other row, a NaN cell counted at that symbol's last target.
    turnover : xr.DataArray | None
        Turnover per fill bar on ``timestamp``: buys plus sells over the
        previous bar's value, so replacing the whole book is about 2. Drawn
        as bars on the Portfolio tab.
    bars_per_year : float | None
        Bars in a year, the window of the Rolling tab; 252 when not given.
    windows : dict | None
        The run's windows, drawn as a timeline at the top of the left
        column: ``backtest`` (first and last bar label), ``bars`` (bar
        count), ``in_sample`` and ``out_of_sample`` (lists of label pairs)
        and ``folds``, one dict per trained model with ``label`` (row
        name), ``training`` (its effective training window), ``traded``
        (the bars it traded) and ``in_sample`` (the traded bars inside its
        training window), each pair or ``None``. A pair that does not parse
        is left out of the drawing.

    Examples
    --------
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
    # The page lists the notes in its own section; inside the figure they
    # would be cut off at the plot's width.
    fig = backtest_report_figure(
        value,
        in_sample_range=in_sample_range,
        notes=None,
        returns=returns,
        init_cash=init_cash,
        drawdown_span=drawdown_span,
        benchmark_value=benchmark_value,
        benchmark_returns=benchmark_returns,
        benchmark_name=benchmark_name,
    )
    equity = value.to_pandas()
    reference = _aligned_benchmark(benchmark_value, equity)
    name = str(benchmark_name) if reference is not None else None
    tabs = [("Performance", fig.to_html(full_html=False, include_plotlyjs="cdn",
                                        config={"responsive": True})
             + _heatmap_block(_monthly_heatmap_div(returns)))]
    extra = [
        ("Excess", _excess_figure(equity, reference, name) if name else None),
        ("Rolling", _rolling_figure(equity, reference, bars_per_year, name)),
        ("Portfolio", _portfolio_figure(weights, turnover)),
    ]
    for label, extra_fig in extra:
        if extra_fig is not None:
            _shade(extra_fig, in_sample_range)
            tabs.append((label, _figure_div(extra_fig)))
    Path(path).write_text(
        _document(title, summary, metrics, tabs, notes, benchmark_name=name, windows=windows),
        encoding="utf-8",
    )


def backtest_report_figure(
    value: xr.DataArray,
    *,
    in_sample_range: tuple[str, str] | None = None,
    notes: list[str] | None = None,
    returns: xr.DataArray | None = None,
    init_cash: float | None = None,
    drawdown_span: dict | None = None,
    benchmark_value: xr.DataArray | None = None,
    benchmark_returns: xr.DataArray | None = None,
    benchmark_name: str = "benchmark",
) -> go.Figure:
    """Return the plotly figure of a backtest report, the chart ``report.html`` embeds.

    Equity, drawdown and compounded monthly returns on a shared time axis, the
    in-sample range shaded, the deepest drawdown marked and the notes below; with a
    benchmark, its NAV, drawdown and monthly returns beside the portfolio's. The
    excess over the benchmark is not drawn here: ``report.html`` draws it on its
    own tab, apart from the drawdowns, because the drawdown of the relative NAV
    is a different quantity from either NAV's drawdown. The parameters mean what
    they mean for ``write_backtest_report``, which draws its Performance tab with
    this function.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure, not yet shown or written.

    Examples
    --------
    >>> import pandas as pd, xarray as xr
    >>> from quantlab.utils.backtest_report import backtest_report_figure
    >>> ts = pd.bdate_range("2024-01-01", periods=5)
    >>> value = xr.DataArray([100.0, 104.0, 98.0, 103.0, 110.0],
    ...                      dims=("timestamp",), coords={"timestamp": ts})
    >>> figure = backtest_report_figure(value, init_cash=100.0)
    >>> [trace.name for trace in figure.data]
    ['equity', 'drawdown']
    """
    equity = value.to_pandas()
    drawdown = equity / equity.cummax() - 1.0
    reference = _aligned_benchmark(benchmark_value, equity)

    rows = {"equity": 1, "drawdown": 2, "monthly": 3}
    fig = make_subplots(
        rows=len(rows),
        cols=1,
        shared_xaxes=True,
        row_heights=[0.54, 0.23, 0.23],
        vertical_spacing=0.04,
    )
    _add_equity(fig, equity, init_cash)
    _add_drawdown_span(fig, equity, drawdown_span)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="drawdown",
            mode="lines",
            line={"color": PORTFOLIO_COLOUR},
        ),
        row=rows["drawdown"],
        col=1,
    )
    if reference is not None:
        _add_benchmark(fig, equity, reference, benchmark_name, init_cash, rows)
    _add_monthly_returns(
        fig,
        returns,
        row=rows["monthly"],
        benchmark_returns=benchmark_returns if reference is not None else None,
        benchmark_name=benchmark_name,
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
            y=-0.16,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
        )
    fig.update_yaxes(title_text="value", row=rows["equity"], col=1)
    fig.update_yaxes(
        title_text="drawdown", tickformat=".1%", row=rows["drawdown"], col=1
    )
    fig.update_yaxes(
        title_text="monthly return", tickformat=".1%", row=rows["monthly"], col=1
    )
    if reference is not None:
        # Only the curves that need telling apart carry a legend entry.
        for trace in fig.data:
            trace.showlegend = trace.name in _LEGEND_TRACES
    # Rotated y-axis titles must fit in their row. At plotly's default 450px
    # rows 2 and 3 are about 44px tall and the titles collide; at 900px they
    # are about 140px and every title fits.
    fig.update_layout(
        height=900,
        margin={"b": 140},
        showlegend=reference is not None,
        # Below the chart: in the page's half-width column a legend on top
        # wraps onto two rows and runs into the linear/log buttons.
        legend={"orientation": "h", "x": 0.0, "xanchor": "left", "y": -0.06, "yanchor": "top"},
        barmode="group",
        updatemenus=[_axis_toggle()],
    )
    return fig


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


#: Line colours of the two curves compared on every row: the portfolio in a
#: saturated blue, the benchmark in a neutral grey so the eye lands on the
#: portfolio first. The excess rows use their own accent.
PORTFOLIO_COLOUR = "#1f77b4"
BENCHMARK_COLOUR = "#8a8f98"
EXCESS_COLOUR = "#d9822b"

#: Trace names shown in the legend when a benchmark is drawn; everything
#: else (markers, the drawdown rows) is identified by its row title.
_LEGEND_TRACES = ("equity", "benchmark_equity", "monthly_return", "benchmark_monthly_return")


def _aligned_benchmark(
    benchmark_value: xr.DataArray | None, equity: pd.Series
) -> pd.Series | None:
    """Return the benchmark value on the equity index, or ``None`` to draw none.

    The backtester values both on the same bars; a benchmark that does not
    cover the equity index is left off the page rather than raising, because
    an exception here would discard the whole staged run directory.
    """
    if benchmark_value is None:
        return None
    series = benchmark_value.to_pandas()
    if not isinstance(series, pd.Series) or series.empty:
        return None
    series = series.reindex(equity.index)
    if not series.notna().all():
        return None
    return series


def _excess_curves(equity: pd.Series, reference: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return ``(excess return, excess drawdown)`` of the portfolio over the benchmark.

    The relative NAV is portfolio value over benchmark value. Both start from
    the same capital, so it is 1 before the first bar and its final value
    minus 1 is the ``Excess Return [%]`` metric of ``relative.whole`` (as a
    fraction here, a percent there); its drawdown is measured from a running
    peak that starts at 1, like ``Excess Max Drawdown [%]``.
    """
    relative = equity / reference
    excess = relative - 1.0
    peak = relative.cummax().clip(lower=1.0)
    return excess, relative / peak - 1.0


def _add_benchmark(
    fig,
    equity: pd.Series,
    reference: pd.Series,
    name: str,
    init_cash: float | None,
    rows: dict,
) -> None:
    """Add the benchmark NAV and its drawdown beside the portfolio's."""
    label = html.escape(str(name))
    if init_cash:
        multiple = reference.values / float(init_cash)
        hover = (
            f"%{{x}}<br>{label} value %{{y:,.2f}}"
            "<br>%{customdata:,.4f}x initial<extra></extra>"
        )
    else:
        multiple = [None] * len(reference)
        hover = f"%{{x}}<br>{label} value %{{y:,.2f}}<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=reference.index,
            y=reference.values,
            name="benchmark_equity",
            mode="lines",
            line={"color": BENCHMARK_COLOUR, "dash": "dash"},
            customdata=multiple,
            hovertemplate=hover,
        ),
        row=rows["equity"],
        col=1,
    )

    benchmark_drawdown = reference / reference.cummax() - 1.0
    fig.add_trace(
        go.Scatter(
            x=benchmark_drawdown.index,
            y=benchmark_drawdown.values,
            name="benchmark_drawdown",
            mode="lines",
            line={"color": BENCHMARK_COLOUR, "dash": "dash"},
            hovertemplate=f"%{{x}}<br>{label} drawdown %{{y:.2%}}<extra></extra>",
        ),
        row=rows["drawdown"],
        col=1,
    )


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
            line={"color": PORTFOLIO_COLOUR},
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


def _add_monthly_returns(
    fig,
    returns: xr.DataArray | None,
    *,
    row: int = 3,
    benchmark_returns: xr.DataArray | None = None,
    benchmark_name: str = "benchmark",
) -> None:
    """Add the per-calendar-month compounded returns as a bar row.

    A short window legitimately yields one or two bars; that shows at a glance
    that a run's whole profit landed in a single month. Each bar is coloured
    by its sign with the same constants as the heatmap, so a month reads the
    same colour in both panels. With ``benchmark_returns`` the benchmark's
    months stand beside the portfolio's in grey.
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
        row=row,
        col=1,
    )
    reference = _monthly_series(benchmark_returns)
    if reference is None or reference.empty:
        return
    label = html.escape(str(benchmark_name))
    fig.add_trace(
        go.Bar(
            x=[period.to_timestamp() for period in reference.index],
            y=reference.values,
            marker={"color": BENCHMARK_COLOUR},
            name="benchmark_monthly_return",
            hovertemplate=f"%{{x|%Y-%m}}<br>{label} %{{y:.2%}}<extra></extra>",
        ),
        row=row,
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


#: How a metric is displayed: ``(group, label, unit, definition)``. Only the
#: rows' presentation lives here; the values are whatever the run computed. A
#: key the catalogue does not know is still shown, in the "Other" group, with
#: a unit guessed from its name and type (see ``_guess_unit``).
_STRATEGY_METRICS: dict[str, tuple[str, str, str, str]] = {
    "Start Value": ("Returns", "Start value", "money", "Portfolio value at the first bar."),
    "End Value": ("Returns", "End value", "money", "Portfolio value at the last bar."),
    "Total Return [%]": ("Returns", "Total return", "pct", "Compounded return over the window."),
    "Annualized Return [%]": ("Returns", "Annualised return", "pct", "Total return compounded to one year."),
    "Annualized Volatility [%]": ("Risk", "Annualised volatility", "pct",
                                  "Standard deviation of the per-bar returns, annualised."),
    "Max Drawdown [%]": ("Risk", "Max drawdown", "neg_pct",
                         "Deepest fall of the value from its running peak; negative."),
    "Max Drawdown Duration": ("Risk", "Longest drawdown", "days",
                              "Longest time spent below a previous peak, in calendar days."),
    "Value at Risk": ("Risk", "Value at risk (95%)", "frac_pct", "5th percentile of the per-bar returns."),
    "Skew": ("Risk", "Skew", "ratio", "Skewness of the per-bar returns."),
    "Kurtosis": ("Risk", "Kurtosis", "ratio", "Excess kurtosis of the per-bar returns."),
    "Sharpe Ratio": ("Risk-adjusted", "Sharpe ratio", "ratio",
                     "Annualised mean over annualised volatility of the per-bar returns (no risk-free rate)."),
    "Sortino Ratio": ("Risk-adjusted", "Sortino ratio", "ratio",
                      "Like the Sharpe ratio, with the downside deviation in the denominator."),
    "Calmar Ratio": ("Risk-adjusted", "Calmar ratio", "ratio", "Annualised return over the max drawdown."),
    "Omega Ratio": ("Risk-adjusted", "Omega ratio", "ratio",
                    "Probability-weighted gains over probability-weighted losses of the per-bar returns."),
    "Tail Ratio": ("Risk-adjusted", "Tail ratio", "ratio",
                   "95th percentile of the per-bar returns over the absolute 5th percentile."),
    "Common Sense Ratio": ("Risk-adjusted", "Common sense ratio", "ratio", "Profit factor times the tail ratio."),
    "Rebalance Win Rate [%]": ("Win rates", "Rebalances with a gain", "pct1",
                               "Share of holding periods (fill bar to the bar before the next fill) with a "
                               "positive compounded return."),
    "Monthly Win Rate [%]": ("Win rates", "Months with a gain", "pct1",
                             "Share of calendar months with a positive compounded return."),
}

#: Which side of a strategy-vs-benchmark row is better: +1 when the larger
#: value is, -1 when the smaller is. A metric absent here (skew, kurtosis,
#: the start value) has no better side and marks nothing. Drawdowns and the
#: value at risk are negative, so the larger (closer to zero) is better.
_BETTER = {
    "End Value": 1, "Total Return [%]": 1, "Annualized Return [%]": 1,
    "Annualized Volatility [%]": -1, "Max Drawdown [%]": 1, "Max Drawdown Duration": -1,
    "Value at Risk": 1, "Sharpe Ratio": 1, "Sortino Ratio": 1, "Calmar Ratio": 1,
    "Omega Ratio": 1, "Tail Ratio": 1, "Common Sense Ratio": 1,
}

#: Strategy rows the "Relative to" table replaces when a benchmark ran.
_WITHOUT_BENCHMARK_ONLY = frozenset({"Rebalance Win Rate [%]", "Monthly Win Rate [%]"})

#: The strategy's trading rows, from the same slice as the strategy metrics.
_TRADING_METRICS: dict[str, tuple[str, str, str]] = {
    "Annualized Turnover [%]": ("Annualised turnover", "pct0",
                                "Buys plus sells per year, as a share of the portfolio value."),
    "Turnover per Rebalance [%]": ("Turnover per rebalance", "pct0",
                                   "Mean of buys plus sells per fill bar over the previous bar's value: "
                                   "buying a full book from cash is 100%, replacing the whole book about 200%."),
    "Total Turnover [%]": ("Total turnover", "pct0", "Buys plus sells over the window, as a share of the portfolio value."),
    "Traded Notional": ("Traded notional", "money", "Value of every fill over the window."),
    "Total Fees Paid": ("Fees paid", "money", "Fees and slippage the simulation charged."),
    "Max Gross Exposure [%]": ("Max gross exposure", "pct0", "Largest gross exposure held."),
    "Total Orders": ("Orders filled", "int", "Fills that happened over the window."),
    "Total Trades": ("Round trips", "int", "Entry-to-flat round trips per symbol, open ones included."),
    "Total Closed Trades": ("Round trips closed", "int", "Round trips that ended flat."),
    "Total Open Trades": ("Round trips open", "int", "Round trips still open at the last bar."),
    "Open Trade PnL": ("Open round-trip P&L", "money", "Unrealised profit of the open round trips."),
    "Win Rate [%]": ("Round-trip win rate", "pct0", "Share of closed round trips with a profit."),
    "Best Trade [%]": ("Best round trip", "pct", "Return of the best closed round trip."),
    "Worst Trade [%]": ("Worst round trip", "pct", "Return of the worst closed round trip."),
    "Avg Winning Trade [%]": ("Avg winning round trip", "pct", "Mean return of the winning round trips."),
    "Avg Losing Trade [%]": ("Avg losing round trip", "pct", "Mean return of the losing round trips."),
    "Avg Winning Trade Duration": ("Avg winning round-trip duration", "days1",
                                   "Mean holding time of the winning round trips."),
    "Avg Losing Trade Duration": ("Avg losing round-trip duration", "days1",
                                  "Mean holding time of the losing round trips."),
    "Profit Factor": ("Profit factor", "ratio", "Gross profit over gross loss of the closed round trips."),
    "Expectancy": ("Expectancy", "money", "Mean profit per closed round trip."),
}

#: The relative rows, from ``metrics["relative"]``.
_RELATIVE_METRICS: dict[str, tuple[str, str, str]] = {
    "Excess Return [%]": ("Excess return (geometric)", "spct",
                          "Strategy value over benchmark value, minus 1, at the last bar."),
    "Annualized Excess Return [%]": ("Annualised excess return", "spct",
                                     "The geometric excess compounded to one year."),
    "Total Return Difference [%]": ("Total return difference (arithmetic)", "spct",
                                    "Strategy total return minus benchmark total return; differs from "
                                    "the geometric excess by compounding."),
    "Excess Max Drawdown [%]": ("Excess max drawdown", "pct",
                                "Deepest fall of the relative value (strategy over benchmark) from its "
                                "running peak, which starts at 1; negative."),
    "Tracking Error [%]": ("Tracking error", "pct", "Annualised standard deviation of the per-bar excess r - b."),
    "Information Ratio": ("Information ratio", "ratio", "Annualised mean of r - b over the tracking error."),
    "Beta": ("Beta", "ratio", "Slope of the strategy's per-bar returns on the benchmark's."),
    "Correlation": ("Correlation", "ratio", "Correlation of the per-bar returns with the benchmark's."),
    "CAPM Alpha [%]": ("CAPM alpha", "spct", "Annualised intercept of that regression: return beta does not explain."),
    "Win Rate vs Benchmark [%]": ("Bars beating the benchmark", "pct0", "Share of bars with r > b."),
    "Rebalance Win Rate vs Benchmark [%]": ("Rebalances beating the benchmark", "pct1",
                                            "Share of holding periods (fill bar to the bar before the next "
                                            "fill) whose compounded return beats the benchmark's."),
    "Monthly Win Rate vs Benchmark [%]": ("Months beating the benchmark", "pct1",
                                          "Share of calendar months whose compounded return beats the "
                                          "benchmark's."),
}

#: Keys every run carries that the page shows elsewhere (the dates-and-setup
#: table) or that are bookkeeping, so they make no "Other" row.
_HIDDEN_METRICS = frozenset({"Start", "End", "Period", "Bars"})

#: Relative keys that repeat the "Total return" row of the comparison table.
_RELATIVE_SHOWN_ELSEWHERE = frozenset({"Strategy Total Return [%]", "Benchmark Total Return [%]"})

#: The key rows of the in-sample vs out-of-sample table.
_SPLIT_METRICS = ("Total Return [%]", "Annualized Return [%]", "Annualized Volatility [%]",
                  "Sharpe Ratio", "Max Drawdown [%]")
_SPLIT_RELATIVE = ("Excess Return [%]", "Excess Max Drawdown [%]", "Information Ratio", "Beta")


def _number(value: object) -> float | None:
    """``value`` as a finite float, or ``None`` for anything that is not one.

    ``bool`` is rejected because it is an ``int`` subclass and would render as
    a plausible number; ``numbers.Real`` admits numpy scalars without
    importing numpy.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _days(value: object) -> float | None:
    """A duration (a ``pd.Timedelta`` or its string) in days, or ``None``."""
    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(delta) else delta / pd.Timedelta(days=1)


def _guess_unit(key: str, value: object) -> str:
    """The display unit of a metric the catalogues do not know."""
    if key.endswith("[%]"):
        return "pct"
    if value is pd.NaT or isinstance(value, pd.Timedelta):
        return "days1"
    if isinstance(value, bool):
        return "text"
    if isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, numbers.Real):
        return "num"
    if isinstance(value, str) and _days(value) is not None and "day" in value:
        return "days1"
    return "text"


def _format(value: object, unit: str) -> str:
    """Render one metric value in its unit; anything unrenderable becomes a dash.

    NaN and infinity become a dash rather than ``nan`` / ``inf``, which read
    like numbers. ``neg_pct`` shows a positive magnitude as a negative percent,
    so every drawdown on the page carries the same sign.
    """
    if value is None:
        return DASH
    if unit in ("days", "days1"):
        days = _days(value)
        if days is None:
            return DASH
        return f"{days:,.0f} d" if unit == "days" else f"{days:,.1f} d"
    if unit == "text":
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value)
        return text if text.strip() else DASH
    number = _number(value)
    if number is None:
        return DASH if not isinstance(value, str) or not value.strip() else str(value)
    if unit == "pct":
        return f"{number:,.2f}%"
    if unit == "spct":
        return f"{number:+,.2f}%" if number else "0.00%"
    if unit == "neg_pct":
        return f"{-abs(number):,.2f}%"
    if unit == "pct0":
        return f"{number:,.0f}%"
    if unit == "pct1":
        return f"{number:,.1f}%"
    if unit == "frac_pct":
        return f"{number * 100:,.2f}%"
    if unit == "ratio":
        return f"{number:,.2f}"
    if unit == "money":
        return f"{number:,.0f}"
    if unit == "int":
        return f"{int(round(number)):,}"
    # "num": a number of unknown meaning, legible at any magnitude.
    return f"{number:,.0f}" if abs(number) >= 1000 else f"{number:.4g}"


def _difference(later: object, earlier: object, unit: str) -> str:
    """``later - earlier`` in the unit's terms: percentage points for percents."""
    if unit in ("days", "days1"):
        a, b = _days(later), _days(earlier)
        return DASH if a is None or b is None else f"{a - b:+,.0f} d"
    a, b = _number(later), _number(earlier)
    if a is None or b is None:
        return DASH
    if unit == "neg_pct":
        a, b = -abs(a), -abs(b)
    if unit in ("pct", "spct", "neg_pct", "pct0", "pct1"):
        return f"{a - b:+,.2f} pp"
    if unit == "frac_pct":
        return f"{(a - b) * 100:+,.2f} pp"
    if unit == "money":
        return f"{a - b:+,.0f}"
    if unit == "int":
        return f"{int(round(a - b)):+,}"
    return f"{a - b:+,.2f}"


def _slice(block: object, name: str, *, fallback: bool) -> dict:
    """``block[name]`` as a flat dict; ``fallback`` fills a key missing from
    ``whole`` from ``out_of_sample``, for a run whose whole window is its
    out-of-sample part (vectorbt reports some period statistics per slice)."""
    if not isinstance(block, dict):
        return {}
    out = _flatten(block[name]) if isinstance(block.get(name), dict) else {}
    if fallback and isinstance(block.get("out_of_sample"), dict):
        for key, value in _flatten(block["out_of_sample"]).items():
            if out.get(key) is None:
                out[key] = value
    return out


def _has_in_sample(metrics: dict) -> bool:
    """Whether the run's metrics carry an in-sample slice with any value."""
    return isinstance(metrics.get("in_sample"), dict) and bool(metrics["in_sample"])


def _headline(block: object, metrics: dict) -> dict:
    """The slice the cards and the main tables show, from ``block``.

    With an in-sample part the out-of-sample slice, the bars the model never
    saw; otherwise the whole window, which is then all out-of-sample, gaps
    filled from ``out_of_sample``.
    """
    if _has_in_sample(metrics):
        return _slice(block, "out_of_sample", fallback=False)
    return _slice(block, "whole", fallback=True)


def _suffix(metrics: dict) -> str:
    """`` (out-of-sample)`` on the headings when the headline is not the whole window."""
    return " (out-of-sample)" if _has_in_sample(metrics) else ""


def _row(label: str, definition: str, cells: list[str], better: list[bool] | None = None) -> str:
    """One table row: the label with its definition on hover, then the cells;
    a cell whose ``better`` flag is set is marked as the better side."""
    flags = better or [False] * len(cells)
    tds = "".join(
        ('<td class="better">' if flag else "<td>") + f"{_escape(cell)}</td>"
        for cell, flag in zip(cells, flags + [False] * (len(cells) - len(flags)))
    )
    return f'<tr><th title="{_escape(definition)}">{_escape(label)}</th>{tds}</tr>'


def _table(heading: str, header: list[str], rows: list[str]) -> str:
    """A captioned metric table, or the empty string when it has no row."""
    if not any(not row.startswith('<tr class="group">') for row in rows):
        return ""
    head = "".join(f"<th>{_escape(cell)}</th>" for cell in header)
    return (
        f"  <h2>{_escape(heading)}</h2>\n"
        '  <table class="metrics">\n'
        f"    <thead><tr>{head}</tr></thead>\n"
        "    <tbody>\n      " + "\n      ".join(rows) + "\n    </tbody>\n"
        "  </table>\n"
    )


def _group_row(name: str, width: int) -> str:
    """A full-width row naming the group of the rows below it."""
    return f'<tr class="group"><td colspan="{width}">{_escape(name)}</td></tr>'


def _comparison_section(metrics: dict, benchmark_name: str | None) -> str:
    """Strategy (and benchmark and difference) by group, then an "Other" table.

    The columns are the headline slice (``_headline``) of the strategy and
    of ``metrics["benchmark"]``, and their difference.
    """
    strategy = _headline(metrics, metrics)
    benchmark = _headline(metrics.get("benchmark"), metrics) if benchmark_name else {}
    width = 4 if benchmark_name else 2
    rows, group = [], None
    for key, (name, label, unit, definition) in _STRATEGY_METRICS.items():
        if strategy.get(key) is None and benchmark.get(key) is None:
            continue
        if benchmark_name and key in _WITHOUT_BENCHMARK_ONLY:
            continue
        if name != group:
            rows.append(_group_row(name, width))
            group = name
        cells = [_format(strategy.get(key), unit)]
        better = None
        if benchmark_name:
            cells += [
                _format(benchmark.get(key), unit),
                _difference(strategy.get(key), benchmark.get(key), unit),
            ]
            better = _better_side(key, strategy.get(key), benchmark.get(key))
        rows.append(_row(label, definition, cells, better))
    header = ["", "Strategy"] + ([benchmark_name, "Difference"] if benchmark_name else [])
    heading = (f"Strategy vs {benchmark_name}" if benchmark_name else "Strategy") + _suffix(metrics)
    out = _table(heading, header, rows)

    return out + _other_section(metrics, benchmark_name, strategy, benchmark)


def _other_section(metrics: dict, benchmark_name: str | None, strategy: dict, benchmark: dict) -> str:
    """Every headline metric no catalogue knows, with a unit guessed from it.

    The strategy's keys are listed as they are; a benchmark or relative key
    is prefixed with where it came from, so the three sources share one
    table.
    """
    known = set(_STRATEGY_METRICS) | set(_TRADING_METRICS) | _HIDDEN_METRICS
    sources = [("", strategy, known)]
    if benchmark_name:
        sources += [
            (f"{benchmark_name}: ", benchmark, known),
            ("relative: ", _headline(metrics.get("relative"), metrics), set(_RELATIVE_METRICS) | _HIDDEN_METRICS | _RELATIVE_SHOWN_ELSEWHERE),
        ]
    rows = [
        _row(f"{prefix}{key}", "Reported by the run; the report has no description of it.",
             [_format(value, _guess_unit(key, value))])
        for prefix, block, catalogue in sources
        for key, value in block.items()
        if key not in catalogue
    ]
    return _table("Other", ["", "Value"], rows)


def _better_side(key: str, strategy: object, benchmark: object) -> list[bool] | None:
    """``[strategy is better, benchmark is better]``, or ``None`` when the row
    has no direction, a value is missing, or the two are equal."""
    direction = _BETTER.get(key)
    if direction is None:
        return None
    if key == "Max Drawdown Duration":
        a, b = _days(strategy), _days(benchmark)
    else:
        a, b = _number(strategy), _number(benchmark)
        if key == "Max Drawdown [%]" and a is not None and b is not None:
            a, b = -abs(a), -abs(b)
    if a is None or b is None or a == b:
        return None
    strategy_better = (a > b) == (direction > 0)
    return [strategy_better, not strategy_better]


def _relative_section(metrics: dict, benchmark_name: str | None) -> str:
    """The excess over the benchmark, from ``metrics["relative"]``."""
    if not benchmark_name:
        return ""
    relative = _headline(metrics.get("relative"), metrics)
    rows = [
        _row(label, definition, [_format(relative.get(key), unit)])
        for key, (label, unit, definition) in _RELATIVE_METRICS.items()
        if relative.get(key) is not None
    ]
    return _table(f"Relative to {benchmark_name}{_suffix(metrics)}", ["", "Value"], rows)


def _trading_section(metrics: dict) -> str:
    """Turnover, costs and round trips, then execution and construction counts."""
    strategy = _headline(metrics, metrics)
    rows = [
        _row(label, definition, [_format(strategy.get(key), unit)])
        for key, (label, unit, definition) in _TRADING_METRICS.items()
        if strategy.get(key) is not None
    ]
    execution = metrics.get("execution")
    if isinstance(execution, dict) and execution.get("rejected_order_count") is not None:
        rows.append(_row("Orders rejected", "Orders without a fill price at the next bar; the holding was kept.",
                         [_format(execution["rejected_order_count"], "int")]))
    construction = metrics.get("portfolio_construction")
    if isinstance(construction, dict):
        if construction.get("failed_bar_count") is not None:
            rows.append(_row("Rebalances held after a failure",
                             "Rebalance bars the portfolio constructor could not decide; the backtest held the position.",
                             [_format(construction["failed_bar_count"], "int")]))
        for event, record in construction.items():
            if isinstance(record, dict) and "count" in record:
                bars = record.get("bars")
                on = f" on {len(bars):,} bar(s)" if isinstance(bars, list) else ""
                rows.append(_row(f"Constructor event: {event}",
                                 "Symbols the portfolio constructor reported this event for, summed over "
                                 "the rebalance bars it happened on.",
                                 [_format(record["count"], "int") + on]))
    return _table(f"Trading{_suffix(metrics)}", ["", "Value"], rows)


def _split_section(metrics: dict, benchmark_name: str | None) -> str:
    """Key metrics per slice, only when the run has an in-sample part.

    The columns are the in-sample slice, the out-of-sample slice, their
    difference (out-of-sample minus in-sample, in percentage points for a
    percent) and the whole window.
    """
    if not _has_in_sample(metrics):
        return ""
    slices = ("in_sample", "out_of_sample", "whole")
    rows = []
    for block, keys, catalogue in (
        (metrics, _SPLIT_METRICS, {k: v[1:] for k, v in _STRATEGY_METRICS.items()}),
        (metrics.get("relative") if benchmark_name else None, _SPLIT_RELATIVE, _RELATIVE_METRICS),
    ):
        columns = [_slice(block, name, fallback=False) for name in slices]
        for key in keys:
            label, unit, definition = catalogue[key]
            if all(column.get(key) is None for column in columns):
                continue
            first, second, whole = (column.get(key) for column in columns)
            rows.append(_row(label, definition, [
                _format(first, unit), _format(second, unit), _difference(second, first, unit), _format(whole, unit),
            ]))
    return _table("In-sample vs out-of-sample", ["", "In-sample", "Out-of-sample", "Difference", "Whole"], rows)


def _card(label: str, value: str, note: str, sign: float | None = None) -> str:
    """One KPI card; ``sign`` colours the value green above zero, red below."""
    cls = "" if sign is None or sign == 0 else (" pos" if sign > 0 else " neg")
    return (
        f'<div class="kpi"><div class="kl">{_escape(label)}</div>'
        f'<div class="kv{cls}">{_escape(value)}</div><div class="ks">{_escape(note)}</div></div>'
    )


def _kpi_section(metrics: dict | None, benchmark_name: str | None) -> str:
    """The row of headline numbers above the tables; empty without metrics."""
    if not isinstance(metrics, dict) or not isinstance(metrics.get("whole"), dict):
        return ""
    s = _headline(metrics, metrics)
    turnover = _card("Turnover / year", _format(s.get("Annualized Turnover [%]"), "pct0"),
                     f"fees {_format(s.get('Total Fees Paid'), 'money')}")
    if benchmark_name:
        b = _headline(metrics.get("benchmark"), metrics)
        r = _headline(metrics.get("relative"), metrics)
        cards = [
            _card("Total return", _format(s.get("Total Return [%]"), "pct"),
                  f"{benchmark_name} {_format(b.get('Total Return [%]'), 'pct')}"),
            _card("Excess return", _format(r.get("Excess Return [%]"), "spct"),
                  f"annualised {_format(r.get('Annualized Excess Return [%]'), 'spct')}",
                  _number(r.get("Excess Return [%]"))),
            _card("Information ratio", _format(r.get("Information Ratio"), "ratio"),
                  f"tracking error {_format(r.get('Tracking Error [%]'), 'pct')}"),
            _card("Win rate", _format(r.get("Rebalance Win Rate vs Benchmark [%]"), "pct1"),
                  f"monthly {_format(r.get('Monthly Win Rate vs Benchmark [%]'), 'pct1')}"),
            _card("Sharpe ratio", _format(s.get("Sharpe Ratio"), "ratio"),
                  f"{benchmark_name} {_format(b.get('Sharpe Ratio'), 'ratio')}"),
            _card("Max drawdown", _format(s.get("Max Drawdown [%]"), "neg_pct"),
                  f"{benchmark_name} {_format(b.get('Max Drawdown [%]'), 'neg_pct')}"),
            _card("Beta", _format(r.get("Beta"), "ratio"), f"correlation {_format(r.get('Correlation'), 'ratio')}"),
            turnover,
        ]
    else:
        cards = [
            # The out-of-sample slice ends on the last bar too, but only the
            # whole window reports the value there.
            _card("Total return", _format(s.get("Total Return [%]"), "pct"),
                  f"end value {_format(_slice(metrics, 'whole', fallback=True).get('End Value'), 'money')}"),
            _card("Annualised return", _format(s.get("Annualized Return [%]"), "pct"), ""),
            _card("Win rate", _format(s.get("Rebalance Win Rate [%]"), "pct1"),
                  f"monthly {_format(s.get('Monthly Win Rate [%]'), 'pct1')}"),
            _card("Sharpe ratio", _format(s.get("Sharpe Ratio"), "ratio"),
                  f"Sortino {_format(s.get('Sortino Ratio'), 'ratio')}"),
            _card("Max drawdown", _format(s.get("Max Drawdown [%]"), "neg_pct"),
                  f"longest {_format(s.get('Max Drawdown Duration'), 'days')}"),
            _card("Volatility", _format(s.get("Annualized Volatility [%]"), "pct"), "annualised"),
            turnover,
        ]
    return '  <div class="kpis">' + "".join(cards) + "</div>\n"


# ---------------------------------------------------------------------------
# excess, rolling and portfolio figures
# ---------------------------------------------------------------------------


def _excess_figure(equity: pd.Series, reference: pd.Series, benchmark_name: str) -> go.Figure:
    """Cumulative excess return (log / arithmetic toggle) above the excess drawdown.

    Log is ``cumsum(log((1 + r) / (1 + b)))``, whose ``exp`` minus 1 is the
    geometric excess of the tables; arithmetic is ``cumsum(r - b)``, whose
    slope is the mean excess per bar, the way a cumulative IC is read. The
    per-bar returns are the equity's and the benchmark's own changes, so the
    curve follows the drawn values exactly.
    """
    r = equity.pct_change().fillna(0.0)
    b = reference.pct_change().fillna(0.0)
    # A ratio at or below zero (a value that reached zero or went negative)
    # has no log; it becomes NaN rather than raising inside the staged run.
    ratio = (1.0 + r) / (1.0 + b)
    log_cum = ratio.where(ratio > 0).map(math.log).cumsum()
    arith_cum = (r - b).cumsum()
    _, excess_drawdown = _excess_curves(equity, reference)
    label = html.escape(str(benchmark_name))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(
        x=log_cum.index, y=log_cum.values, name="cumulative_excess_log", mode="lines",
        line={"color": EXCESS_COLOUR},
        hovertemplate=f"%{{x}}<br>cumulative log excess vs {label} %{{y:.2%}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=arith_cum.index, y=arith_cum.values, name="cumulative_excess_arithmetic", mode="lines",
        line={"color": PORTFOLIO_COLOUR}, visible=False,
        hovertemplate=f"%{{x}}<br>cumulative excess vs {label} %{{y:.2%}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=excess_drawdown.index, y=excess_drawdown.values, name="excess_drawdown", mode="lines",
        fill="tozeroy", line={"color": LOSS_COLOUR},
        hovertemplate=f"%{{x}}<br>excess drawdown vs {label} %{{y:.2%}}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0.0, line={"color": BENCHMARK_COLOUR, "width": 1}, row=1, col=1)
    fig.update_yaxes(title_text=f"cumulative excess vs {label}", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text=f"excess drawdown vs {label}", tickformat=".0%", row=2, col=1)
    fig.update_layout(
        height=620, showlegend=False,
        updatemenus=[{
            "type": "buttons", "direction": "right", "x": 1.0, "y": 1.1, "xanchor": "right",
            "yanchor": "top", "showactive": True,
            "buttons": [
                {"label": "Log", "method": "update", "args": [{"visible": [True, False, True]}]},
                {"label": "Arithmetic", "method": "update", "args": [{"visible": [False, True, True]}]},
            ],
        }],
    )
    return fig


def _rolling_figure(equity: pd.Series, reference: pd.Series | None, bars_per_year: float | None,
                    benchmark_name: str) -> go.Figure | None:
    """Rolling one-year panels: excess return, IR and beta against the benchmark,
    or return, volatility and Sharpe ratio without one. ``None`` when the
    window is shorter than a year."""
    window = int(round(bars_per_year or TRADING_BARS_PER_YEAR))
    if window < 2 or len(equity) <= window:
        return None
    r = equity.pct_change()
    if reference is not None:
        b = reference.pct_change()
        excess = r - b
        relative = equity / reference
        panels = [
            ("rolling_excess_return", f"1y excess vs {benchmark_name}", relative / relative.shift(window) - 1.0, ".0%", 0.0),
            ("rolling_information_ratio", "1y information ratio",
             excess.rolling(window).mean() / excess.rolling(window).std() * math.sqrt(window), ".2f", 0.0),
            ("rolling_beta", "1y beta", r.rolling(window).cov(b) / b.rolling(window).var(), ".2f", 1.0),
        ]
    else:
        panels = [
            ("rolling_return", "1y return", equity / equity.shift(window) - 1.0, ".0%", 0.0),
            ("rolling_volatility", "1y volatility", r.rolling(window).std() * math.sqrt(window), ".0%", None),
            ("rolling_sharpe", "1y Sharpe ratio",
             r.rolling(window).mean() / r.rolling(window).std() * math.sqrt(window), ".2f", 0.0),
        ]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for row, (name, title, series, fmt, reference_line) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=name, mode="lines", line={"color": PORTFOLIO_COLOUR},
            hovertemplate=f"%{{x}}<br>{html.escape(title)} %{{y:{fmt}}}<extra></extra>",
        ), row=row, col=1)
        if reference_line is not None:
            fig.add_hline(y=reference_line, line={"color": BENCHMARK_COLOUR, "width": 1}, row=row, col=1)
        fig.update_yaxes(title_text=html.escape(title), tickformat=fmt, row=row, col=1)
    fig.update_layout(height=720, showlegend=False)
    return fig


#: The smallest absolute weight counted as a holding: an optimiser leaves
#: residues of a few basis points or less on symbols it does not want.
HOLDING_THRESHOLD = 1e-3


def _portfolio_figure(weights: xr.DataArray | None, turnover: xr.DataArray | None) -> go.Figure | None:
    """Per rebalance: turnover, holdings and exposure.

    Turnover is the series given (buys plus sells over the previous bar's
    value), drawn as a filled step line with its trailing one-year mean,
    because a bar one day wide vanishes on a multi-year axis. Holdings count
    the target weights of at least ``HOLDING_THRESHOLD`` in absolute value
    (a symbol a row leaves NaN, to keep it, counts at its last target),
    beside the effective number ``(sum |w|)^2 / sum w^2``, which residues do
    not move. Exposure is the gross (and net, when anything is short)
    exposure on an axis that always spans 0 to 100%, so a fully invested
    book's round-off is not magnified into noise.
    """
    if weights is None and turnover is None:
        return None
    rows: list[tuple[str, str, list, dict]] = []
    if turnover is not None and turnover.size:
        series = turnover.to_pandas()
        mean = series.rolling("365D").mean()
        rows.append(("turnover (buys + sells)", "per fill bar · line: trailing one-year mean", [
            go.Scatter(x=series.index, y=series.values, name="turnover", mode="lines",
                       line={"color": EXCESS_COLOUR, "width": 1, "shape": "hv"}, fill="tozeroy",
                       hovertemplate="%{x}<br>turnover %{y:.1%}<extra></extra>"),
            go.Scatter(x=mean.index, y=mean.values, name="turnover_1y_mean", mode="lines",
                       line={"color": PORTFOLIO_COLOUR, "width": 2},
                       hovertemplate="%{x}<br>trailing 1y mean %{y:.1%}<extra></extra>"),
        ], {"tickformat": ".0%", "rangemode": "tozero"}))
    if weights is not None and weights.ndim == 2:
        frame = weights.transpose("timestamp", ...).to_pandas()
        # A NaN cell keeps that symbol's holding, so a rebalance row counts
        # it at its last target; a row with no target at all is no rebalance.
        rebalances = frame.notna().any(axis=1)
        frame = frame.ffill().fillna(0.0)[rebalances]
        if not frame.empty:
            absolute = frame.abs()
            holdings = (absolute >= HOLDING_THRESHOLD).sum(axis=1).astype(float)
            gross = absolute.sum(axis=1)
            squares = (frame**2).sum(axis=1)
            effective = (gross**2 / squares).where(squares > 0, 0.0)
            top = float(max(holdings.max(), effective.max(), 1.0))
            rows.append(("holdings", f"solid: weights of at least {HOLDING_THRESHOLD:.1%} · dotted: effective number", [
                go.Scatter(x=holdings.index, y=holdings.values, name="holdings", mode="lines",
                           line={"color": PORTFOLIO_COLOUR, "shape": "hv"},
                           hovertemplate="%{x}<br>holdings %{y:.0f}<extra></extra>"),
                go.Scatter(x=effective.index, y=effective.values, name="effective_holdings", mode="lines",
                           line={"color": BENCHMARK_COLOUR, "shape": "hv", "dash": "dot"},
                           hovertemplate="%{x}<br>effective holdings %{y:.1f}<extra></extra>"),
            ], {"tickformat": ",.0f", "rangemode": "tozero",
                "dtick": max(1, math.ceil(top / 5))}))
            exposure = [
                go.Scatter(x=gross.index, y=gross.values, name="gross_exposure", mode="lines",
                           line={"color": GAIN_COLOUR, "shape": "hv"},
                           hovertemplate="%{x}<br>gross exposure %{y:.1%}<extra></extra>"),
            ]
            net = frame.sum(axis=1)
            if (frame < 0).any().any():
                exposure.append(go.Scatter(
                    x=net.index, y=net.values, name="net_exposure", mode="lines",
                    line={"color": BENCHMARK_COLOUR, "shape": "hv", "dash": "dash"},
                    hovertemplate="%{x}<br>net exposure %{y:.1%}<extra></extra>"))
            low = min(0.0, float(net.min()))
            high = max(1.0, float(gross.max()))
            pad = 0.05 * (high - low)
            key = "gross (solid) and net (dashed)" if len(exposure) > 1 else "gross"
            rows.append(("exposure", key, exposure, {"tickformat": ".0%", "range": [low - pad, high + pad]}))
    if not rows:
        return None
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.07)
    for row, (title, key, traces, axis) in enumerate(rows, start=1):
        for trace in traces:
            fig.add_trace(trace, row=row, col=1)
        fig.update_yaxes(title_text=title, row=row, col=1, **axis)
        suffix = "" if row == 1 else str(row)
        fig.add_annotation(text=html.escape(key), xref=f"x{suffix} domain", yref=f"y{suffix} domain",
                           x=0.0, y=1.0, xanchor="left", yanchor="bottom", showarrow=False,
                           font={"size": 11, "color": "#666"})
    fig.update_layout(height=240 * len(rows) + 80, showlegend=False)
    return fig


def _shade(fig: go.Figure, in_sample_range) -> None:
    """Shade the in-sample range on every row, as the Performance figure does."""
    if in_sample_range is not None:
        fig.add_vrect(x0=in_sample_range[0], x1=in_sample_range[1], row="all", col=1,
                      fillcolor="grey", opacity=0.2, line_width=0)


#: Bars in a year when the caller does not say (US equity trading days).
TRADING_BARS_PER_YEAR = 252


def _figure_div(fig: go.Figure) -> str:
    """A figure as an HTML fragment reusing the plotly.js the page loads."""
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

#: The timeline's bar kinds: ``(fold key, colour, legend text)``. A fold's
#: ``training`` window, the bars it ``traded`` out-of-sample and its
#: ``in_sample`` bars (traded inside its own training window).
_TIMELINE_KINDS = (
    ("training", "#c6dbef", "training"),
    ("traded", "#2b8a3e", "traded, out-of-sample"),
    ("in_sample", "#e03b30", "traded, in-sample"),
)
_TIMELINE_COLOUR = {key: colour for key, colour, _ in _TIMELINE_KINDS}

#: Timeline geometry in pixels: the drawing width, the row-label gutter and
#: the height the fold rows share before a row stops shrinking.
_TIMELINE_WIDTH = 420
_TIMELINE_GUTTER = 56
_TIMELINE_ROWS_HEIGHT = 260


def _span(pair: object) -> tuple[pd.Timestamp, pd.Timestamp, str] | None:
    """Parse a pair of bar labels for the timeline.

    Parameters
    ----------
    pair : object
        A ``[first, last]`` pair of bar labels, as ``metrics.json`` carries
        them.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp, str] or None
        The two bars as naive UTC timestamps, so labels with and without a
        UTC offset share one axis, and the pair as ``"first .. last"``
        text; ``None`` for anything that is not two parseable, ordered
        labels.
    """
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    try:
        start, end = pd.Timestamp(str(pair[0])), pd.Timestamp(str(pair[1]))
    except (TypeError, ValueError):
        return None
    if pd.isna(start) or pd.isna(end):
        return None
    start, end = (t.tz_convert(None) if t.tzinfo is not None else t for t in (start, end))
    if end < start:
        return None
    return start, end, f"{pair[0]} .. {pair[1]}"


def _ticks(low: pd.Timestamp, high: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
    """Choose the timeline's axis ticks.

    Parameters
    ----------
    low, high : pd.Timestamp
        The first and last instant on the axis, naive.

    Returns
    -------
    list[tuple[pd.Timestamp, str]]
        At most seven month starts inside the axis, each with its label:
        the year when the step is whole years, else ``YYYY-MM``.
    """
    months = (high.year - low.year) * 12 + high.month - low.month + 1
    step = next((m for m in (1, 3, 6, 12, 24, 36, 60, 120) if months / m <= 7), 240)
    first = pd.Timestamp(year=low.year, month=1, day=1)
    ticks = []
    for date in pd.date_range(first, high, freq=pd.DateOffset(months=step)):
        if date > low:
            ticks.append((date, str(date.year) if step % 12 == 0 else date.strftime("%Y-%m")))
    return ticks


def _timeline_section(windows: dict | None) -> str:
    """Draw the run's windows as an inline SVG timeline.

    With several folds (a walk-forward CV run) the top row is the backtest
    window, its out-of-sample bars green and the bars inside a training
    window red, and each fold has a row below it with its training window
    in light blue, the bars it traded in green and its in-sample bars red,
    so sliding and expanding folds read as a staircase of equal or growing
    bars. A model backtest has a single row, its training window beside
    the backtest window, and a run without a model a single row of traded
    bars. Every bar carries its dates as a tooltip; rows shrink as folds
    are added, and only every fifth fold is named once they get thin. The
    legend lists only the kinds drawn.

    Parameters
    ----------
    windows : dict or None
        The ``windows`` input of ``write_backtest_report``.

    Returns
    -------
    str
        The section's HTML, or the empty string without a parseable
        backtest window.
    """
    if not isinstance(windows, dict):
        return ""
    backtest = _span(windows.get("backtest"))
    if backtest is None:
        return ""
    folds = [fold for fold in windows.get("folds") or [] if isinstance(fold, dict)]
    spans = [backtest] + [
        span for fold in folds for span in (_span(fold.get("training")), _span(fold.get("traded"))) if span
    ]
    low, high = min(s[0] for s in spans), max(s[1] for s in spans)
    width = max((high - low).total_seconds(), 1.0)
    drawn: set[str] = set()

    def x(when: pd.Timestamp) -> float:
        """The horizontal position of ``when`` on the drawing."""
        return _TIMELINE_GUTTER + (when - low).total_seconds() / width * (_TIMELINE_WIDTH - _TIMELINE_GUTTER - 4)

    def bar(span: tuple, y: float, height: float, kind: str, tip: str) -> str:
        """One bar of ``kind`` over ``span`` with its dates as the tooltip."""
        drawn.add(kind)
        start, end, text = span
        return (f'<rect x="{x(start):.1f}" y="{y:.1f}" width="{max(1.0, x(end) - x(start)):.1f}" '
                f'height="{height:.1f}" fill="{_TIMELINE_COLOUR[kind]}"><title>{_escape(tip)} '
                f"{_escape(text)}</title></rect>")

    # The backtest row: out-of-sample runs and in-sample runs, or, for a run
    # without a split, the whole window as traded bars.
    out_of_sample = [s for s in map(_span, windows.get("out_of_sample") or []) if s]
    in_sample = [s for s in map(_span, windows.get("in_sample") or []) if s]
    split = bool(folds or out_of_sample)
    backtest_bars = [("traded", span, "out-of-sample") for span in out_of_sample] or [
        ("traded", backtest, "traded" if not split else "out-of-sample")
    ]
    backtest_bars += [("in_sample", span, "in-sample") for span in in_sample]

    # One fold: its training window joins the backtest row, whose traded
    # bars are the fold's, so the two rows would repeat each other.
    single = len(folds) <= 1
    rows = [(str(folds[0].get("label", "")) if folds else "backtest",
             ([("training", _span(folds[0].get("training")), "training")] if folds else []) + backtest_bars)]
    if not single:
        rows = [("backtest", backtest_bars)] + [
            (str(fold.get("label", "")),
             [(key, _span(fold.get(key)), tip) for key, tip in
              (("training", "training"), ("traded", "traded"), ("in_sample", "in-sample"))])
            for fold in folds
        ]

    row = 14.0 if single else max(4.0, min(12.0, _TIMELINE_ROWS_HEIGHT / len(folds)))
    gap = 2.0 if row > 6 else 1.0
    height = len(rows) * (row + gap) + 16
    parts = []
    for when, text in _ticks(low, high):
        parts.append(f'<line x1="{x(when):.1f}" x2="{x(when):.1f}" y1="0" y2="{height - 14:.1f}" stroke="#eee"/>'
                     f'<text x="{x(when):.1f}" y="{height - 3:.1f}" text-anchor="middle">{_escape(text)}</text>')
    for i, (name, bars) in enumerate(rows):
        y = i * (row + gap)
        # Fold rows are numbered from the second row; name every fifth when thin.
        if row >= 9 or i == 0 or (i - 1) % 5 == 0:
            parts.append(f'<text x="0" y="{y + row - 1:.1f}">{_escape(name)}</text>')
        prefix = "" if i == 0 and name == "backtest" else f"{name} "
        parts += [bar(span, y, row, kind, f"{prefix}{tip}") for kind, span, tip in bars if span]

    caption = f"Backtest {backtest[2]}"
    if _number(windows.get("bars")) is not None:
        caption += f" ({int(windows['bars']):,} bars)"
    trained = [s for s in (_span(fold.get("training")) for fold in folds) if s]
    if not single:
        kind = "expanding" if len(trained) > 1 and len({s[0] for s in trained}) == 1 else "sliding"
        caption += f"; {len(folds)} folds, {kind} training window"
    legend = "".join(
        f'<span><span class="sw" style="background:{colour}"></span>'
        f'{"traded" if key == "traded" and not split else text}</span>'
        for key, colour, text in _TIMELINE_KINDS
        if key in drawn
    )
    return (
        "  <h2>Windows</h2>\n"
        f'  <div class="timeline"><div class="caption">{_escape(caption)}</div>\n'
        f'  <svg viewBox="0 0 {_TIMELINE_WIDTH} {height:.0f}" width="100%" style="max-width:{_TIMELINE_WIDTH}px">'
        + "".join(parts) + "</svg>\n"
        f'  <div class="legend">{legend}</div></div>\n'
    )


def _summary_section(summary: dict[str, str] | None) -> str:
    """Render the setup table, or the empty string when there is none."""
    if not summary:
        return ""
    rows = "\n".join(
        f"      <tr><th>{_escape(label)}</th><td>{_escape(text)}</td></tr>"
        for label, text in summary.items()
    )
    return (
        "  <h2>Setup</h2>\n"
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


def _heatmap_block(heatmap: str) -> str:
    """The heatmap under its caption, or nothing when there is no heatmap."""
    return f"\n  <h2>{_escape(HEATMAP_CAPTION)}</h2>\n{heatmap}\n" if heatmap else ""


def _document(
    title: str,
    summary: dict[str, str] | None,
    metrics: dict | None,
    tabs: list[tuple[str, str]],
    notes: list[str] | None,
    *,
    benchmark_name: str | None = None,
    windows: dict | None = None,
) -> str:
    """Assemble the page: headline cards, the tables beside the chart tabs, the notes.

    ``tabs`` are ``(label, html)`` pairs whose html is plotly's own fragments,
    inserted verbatim; everything else comes from the run and is escaped. The
    first tab is shown; a tab switch asks plotly to resize the figures it
    reveals, which were laid out while hidden.
    """
    tables = ""
    if isinstance(metrics, dict):
        tables = (
            _comparison_section(metrics, benchmark_name)
            + _relative_section(metrics, benchmark_name)
            + _trading_section(metrics)
            + _split_section(metrics, benchmark_name)
        )
    buttons = "".join(
        f'<button class="tab{" on" if i == 0 else ""}" data-tab="tab{i}">{_escape(label)}</button>'
        for i, (label, _) in enumerate(tabs)
    )
    panes = "".join(
        f'\n  <div class="pane{" on" if i == 0 else ""}" id="tab{i}">\n{body}\n  </div>'
        for i, (_, body) in enumerate(tabs)
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
        f"{_kpi_section(metrics, benchmark_name)}"
        '  <div class="layout">\n'
        f'  <div class="tables">\n{_timeline_section(windows)}{_summary_section(summary)}{tables}  </div>\n'
        f'  <div class="charts">\n  <div class="tabs">{buttons}</div>{panes}\n  </div>\n'
        "  </div>\n"
        f"{_notes_section(notes)}"
        f"  <script>{_TAB_SCRIPT}</script>\n"
        "</body>\n"
        "</html>\n"
    )


#: Switches the chart tabs and resizes the figures a switch reveals.
_TAB_SCRIPT = """
document.querySelectorAll('.tab').forEach(function (tab) {
  tab.addEventListener('click', function () {
    document.querySelectorAll('.tab').forEach(function (t) { t.classList.toggle('on', t === tab); });
    document.querySelectorAll('.pane').forEach(function (pane) {
      var on = pane.id === tab.dataset.tab;
      pane.classList.toggle('on', on);
      if (on && window.Plotly) {
        pane.querySelectorAll('.plotly-graph-div').forEach(function (div) { Plotly.Plots.resize(div); });
      }
    });
  });
});
"""
