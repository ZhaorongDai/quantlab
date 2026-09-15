"""Interactive HTML report for one backtest run (03.7 D-23, D-21, D-08).

One plotly page with two panels on a shared time axis: the equity curve on top
and the drawdown curve below. When the backtest window overlaps the model's
effective training window, the in-sample range is shaded grey across both
panels. The in-sample range is the intersection of two intervals, so it is
always one contiguous band (the out-of-sample part may be two pieces, but it is
the unshaded remainder). Notes such as the short-side disclosure are printed
below the plot.

The page loads plotly.js from the CDN (`include_plotlyjs="cdn"`). That keeps
each run directory at a few kilobytes instead of several megabytes per report;
the cost is that viewing the page needs network access. The page contains only
local backtest numbers, so the browser's CDN fetch reveals nothing about them.

No benchmark trace is drawn: benchmark comparison is excluded this phase (D-08).

A LEAF module: pandas, xarray and plotly only, zero project-internal imports.
"""

from pathlib import Path

import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

__all__ = ["write_backtest_report"]


def write_backtest_report(
    value: xr.DataArray,
    path: str | Path,
    *,
    in_sample_range: tuple[str, str] | None,
    notes: list[str],
    title: str,
) -> None:
    """Write the equity and drawdown report for `value` to `path`.

    - `value`: portfolio value on the `timestamp` dimension;
    - `in_sample_range`: bar-label pair (first, last in-sample bar), shaded
      when given, or None for a fully out-of-sample window. A midnight bar is
      labelled by its ISO date and any other bar by its full ISO timestamp;
      plotly reads both;
    - `notes`: lines printed below the plot (e.g. what the simulation does not
      model);
    - `title`: page title, typically the run directory name.

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
    fig.update_layout(title=title, margin={"b": 120})
    fig.write_html(str(path), include_plotlyjs="cdn")
