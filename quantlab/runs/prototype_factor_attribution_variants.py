"""PROTOTYPE, throwaway: three ways to draw the Factor attribution section, on the B (Dashboard) page.

Question: how should factor attribution be drawn so that it is clear and
readable? The page is variant B of ``prototype_report_variants.py``
(dashboard, sidebar); only its Factor attribution section changes, opened on
load. ``index.html?variant=F1|F2|F3``.

- F1  Story: one column read top to bottom, each chart answering one question
      (KPI tiles; waterfall of the return; cumulative contribution lines with
      direct labels; contribution per calendar year; styles' exposure next to
      their contribution; style exposure heatmap over time; top/bottom
      industries as diverging bars; forecast risk stacked per month against
      realized volatility; return share against risk share per part).
- F2  Return | Risk: two aligned columns, the return chart on the left and
      its risk counterpart on the right, row by row (parts, over time,
      styles, industries).
- F3  Factor table: a Barra-style table first, one row per factor grouped
      (exposure and contribution as in-cell bars, ex-ante and ex-post risk,
      an exposure sparkline), sortable, groups foldable; two small charts
      above it.

Palette (dataviz reference, validated: all checks pass, contrast WARN on
aqua/yellow/magenta, relieved by direct labels and the table):
blue, orange, aqua, yellow, magenta, green, violet in that fixed order for
the parts; blue/red with a grey midpoint for signs.

Run: ``uv run python quantlab/runs/prototype_factor_attribution_variants.py ARGS.pkl OUT_DIR``
"""

import html
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantlab.runs import backtest_report as br
from quantlab.runs import prototype_report_variants as pv

BARS_PER_YEAR = 252
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"
#: The parts, in the fixed categorical order.
PARTS = [
    ("country", "Country", "#2a78d6"),
    ("industry", "Industry", "#eb6834"),
    ("style", "Style", "#1baf7a"),
    ("specific", "Specific", "#eda100"),
    ("uncovered", "Uncovered", "#e87ba4"),
    ("risk_free", "Risk-free", "#008300"),
    ("trading", "Trading", "#4a3aa7"),
]
GROUP_COLOUR = {key: colour for key, _, colour in PARTS}
VARIANTS = {"F1": "Story", "F2": "Return | Risk", "F3": "Factor table"}

_CSS = """
<style>
  .fa { display:flex; flex-direction:column; gap:16px; }
  .fa .row { display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; align-items:start; }
  .fa .card { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:14px 16px; min-width:0; }
  .fa .q { font-size:13px; font-weight:600; color:#111827; margin:0 0 2px; }
  .fa .a { font-size:12px; color:#6b7280; margin:0 0 6px; }
  .fa .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
  .fa .tile { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; }
  .fa .tile .l { font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.05em; }
  .fa .tile .v { font-size:22px; font-weight:700; margin-top:2px; }
  .fa .tile .s { font-size:12px; color:#6b7280; }
  .fa .colhead { font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }
  .fa table.ft { border-collapse:collapse; width:100%; font-size:12.5px; }
  .fa table.ft th, .fa table.ft td { padding:4px 8px; border-bottom:1px solid #f1f2f4; white-space:nowrap; }
  .fa table.ft thead th { background:#fff; color:#6b7280; font-weight:500; font-size:11px;
                          text-transform:uppercase; letter-spacing:.04em; text-align:right; cursor:pointer; user-select:none; }
  .fa table.ft thead th:first-child { text-align:left; }
  .fa table.ft td { text-align:right; font-variant-numeric:tabular-nums; }
  .fa table.ft td.name { text-align:left; }
  .fa table.ft tr.grp td { background:#f9fafb; font-weight:600; text-align:left; cursor:pointer; color:#111827; }
  .fa table.ft tr.grp td span { color:#6b7280; font-weight:400; margin-left:6px; }
  .fa .bar { position:relative; width:120px; height:14px; display:inline-block; vertical-align:middle; }
  .fa .bar i { position:absolute; top:2px; height:10px; border-radius:2px; }
  .fa .bar b { position:absolute; left:50%; top:0; bottom:0; width:1px; background:#c3c2b7; }
  .fa .num { display:inline-block; width:64px; }
  .fa .note { font-size:12px; color:#6b7280; }
</style>
"""

_TABLE_JS = """
<script>
document.querySelectorAll('table.ft').forEach(function (t) {
  t.querySelectorAll('tr.grp').forEach(function (g) {
    g.addEventListener('click', function () {
      var body = g.parentNode; var open = body.dataset.open !== '0'; body.dataset.open = open ? '0' : '1';
      body.querySelectorAll('tr.f').forEach(function (r) { r.style.display = open ? 'none' : ''; });
    });
  });
  t.querySelectorAll('thead th[data-k]').forEach(function (h, i) {
    h.addEventListener('click', function () {
      var k = h.dataset.k, dir = h.dataset.dir === 'desc' ? 'asc' : 'desc'; h.dataset.dir = dir;
      t.querySelectorAll('tbody').forEach(function (body) {
        var rows = Array.from(body.querySelectorAll('tr.f'));
        rows.sort(function (a, b) { var x = +a.dataset[k], y = +b.dataset[k]; return dir === 'desc' ? y - x : x - y; });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    });
  });
});
</script>
"""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


# PROTOTYPE only: the report layer may not import the dataset layer; production
# would need the industry names in factor_attribution.zarr.
from quantlab.dataset._support.ff48 import FF48_INDUSTRIES  # noqa: E402

_INDUSTRY = {f"industry_{i.code}": i.name for i in FF48_INDUSTRIES}


def _label(factor: str) -> str:
    if factor in _INDUSTRY:
        return _INDUSTRY[factor]
    return re.sub(r"^(style|industry)_", "", factor).replace("_", " ")


def _dodge(values: list[float], gap: float) -> list[float]:
    """Spread label positions at least ``gap`` apart, keeping their order."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out.tolist()


def _headline(block: dict, metrics: dict) -> tuple[str, dict]:
    if br._has_in_sample(metrics):
        return "out-of-sample", block["out_of_sample"]
    return "whole window", block["whole"]


def _parts(attribution) -> dict:
    """Per-bar log contribution of each part, as pandas Series on timestamp."""
    group = attribution["group"].values
    factor_log = attribution["factor_log_contribution"].values
    out = {}
    for key, _, _ in PARTS[:3]:
        out[key] = factor_log[:, group == key].sum(axis=1)
    for key, _, _ in PARTS[3:]:
        out[key] = attribution["log_contribution"].sel(term=key).values
    index = pd.DatetimeIndex(attribution["timestamp"].values)
    return {k: pd.Series(v, index=index) for k, v in out.items()}


def _fig(height: int) -> go.Figure:
    fig = go.Figure()
    _style(fig, height)
    return fig


def _style(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=12, color=INK2),
        hoverlabel=dict(bgcolor="#fff", bordercolor=GRID, font=dict(color=INK)),
        legend=dict(orientation="h", x=0, y=1.0, xanchor="left", yanchor="bottom", font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))


def _padded(values, pad=1.45) -> list[float]:
    """An x range holding the bars and their outside labels on both sides."""
    lo = min(min(values), 0.0)
    hi = max(max(values), 0.0)
    span = (hi - lo) or 1.0
    return [lo - span * (pad - 1) if lo < 0 else -span * 0.05, hi + span * (pad - 1) if hi > 0 else span * 0.05]


def _div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})


def _card(question: str, answer: str, body: str) -> str:
    return f'<div class="card"><p class="q">{html.escape(question)}</p><p class="a">{html.escape(answer)}</p>{body}</div>'


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def tiles(seg: dict, name: str) -> str:
    g = seg["annualized_log_return"]
    ea, ep, cov = seg["ex_ante_risk"], seg["ex_post_risk"], seg["coverage"]
    vol = ea["volatility"]
    share = (vol["factor"] ** 2 / vol["total"] ** 2) if vol["total"] else None

    def tile(label, value, sub):
        return f'<div class="tile"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'

    pct = lambda v: br._format(None if v is None else 100 * v, "spct")
    return '<div class="tiles">' + "".join([
        tile("Log growth / yr", pct(g["total"]), name),
        tile("From factors", pct(g["factor"]), f"specific {pct(g['specific'])}"),
        tile("Risk-free + trading", pct(g["risk_free"] + g["trading"]), f"uncovered {pct(g['uncovered'])}"),
        tile("Forecast vol", br._format(100 * vol["total"], "pct"), f"{share:.0%} of variance from factors" if share else ""),
        tile("Realized vol", br._format(None if ep["volatility"] is None else 100 * ep["volatility"], "pct"), "ex post"),
        tile("Coverage", br._format(100 * cov["mean_covered_weight"], "pct1"), f"min {cov['min_covered_weight']:.1%}"),
    ]) + "</div>"


def waterfall(seg: dict, height=330) -> go.Figure:
    g = seg["group_annualized_log_return"]
    a = seg["annualized_log_return"]
    values = [g.get(k, 0.0) for k, _, _ in PARTS[:3]] + [a[k] for k, _, _ in PARTS[3:]]
    labels = [label for _, label, _ in PARTS]
    fig = _fig(height)
    fig.add_trace(go.Waterfall(
        x=labels + ["Total"], y=values + [a["total"]], measure=["relative"] * len(values) + ["total"],
        text=[f"{v:+.1%}" for v in values + [a["total"]]], textposition="outside", textfont=dict(color=INK2),
        increasing=dict(marker=dict(color=POS)), decreasing=dict(marker=dict(color=NEG)),
        totals=dict(marker=dict(color=INK2)), connector=dict(line=dict(color=AXIS, width=1)),
        hovertemplate="%{x}<br>%{y:+.2%} a year<extra></extra>", name="",
    ))
    running = np.cumsum(values)
    top = max(float(running.max()), a["total"], 0.0)
    bottom = min(float(running.min()), a["total"], 0.0)
    pad = (top - bottom) * 0.15
    fig.update_yaxes(tickformat=".0%", zeroline=True, range=[bottom - pad, top + pad])
    fig.update_layout(showlegend=False)
    return fig


def cumulative(attribution, height=380) -> go.Figure:
    parts = _parts(attribution)
    fig = _fig(height)
    last = parts["country"].index[-1]
    labels = []
    for key, label, colour in PARTS:
        curve = parts[key].cumsum()
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name=label, mode="lines",
                                 line=dict(color=colour, width=2),
                                 hovertemplate=f"{label} %{{y:+.1%}}<extra></extra>"))
        labels.append((curve.values[-1], f"{label} {curve.values[-1]:+.0%}", INK2, colour))
    total = sum(parts.values()).cumsum()
    fig.add_trace(go.Scatter(x=total.index, y=total.values, name="Total (log NAV)", mode="lines",
                             line=dict(color=INK, width=3), hovertemplate="Total %{y:+.1%}<extra></extra>"))
    labels.append((total.values[-1], f"<b>Total {total.values[-1]:+.0%}</b>", INK, INK))
    lo = min(min(float(np.min(v.cumsum())) for v in parts.values()), float(total.min()))
    hi = max(max(float(np.max(v.cumsum())) for v in parts.values()), float(total.max()))
    placed = _dodge([y for y, *_ in labels], (hi - lo) * 0.055)
    for (y, text, ink, colour), at in zip(labels, placed):
        fig.add_annotation(x=last, y=at, text=f'<span style="color:{colour}">■</span> {text}', showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=11, color=ink))
    fig.update_layout(hovermode="x unified", margin=dict(l=10, r=130, t=10, b=10), showlegend=False)
    fig.update_xaxes(range=[parts["country"].index[0], last])
    fig.update_yaxes(tickformat=".0%")
    return fig


def yearly(attribution, height=330) -> go.Figure:
    parts = _parts(attribution)
    frame = pd.DataFrame(parts).groupby(lambda t: t.year).sum()
    fig = _fig(height)
    for key, label, colour in PARTS:
        fig.add_trace(go.Bar(x=frame.index.astype(str), y=frame[key], name=label, marker=dict(color=colour,
                             line=dict(color="#fff", width=1)), hovertemplate=f"%{{x}} {label} %{{y:+.1%}}<extra></extra>"))
    total = frame.sum(axis=1)
    fig.add_trace(go.Scatter(x=total.index.astype(str), y=total.values, name="Total", mode="markers+text",
                             marker=dict(symbol="diamond", size=11, color=INK, line=dict(color="#fff", width=2)),
                             text=[f"{v:+.0%}" for v in total.values], textposition="top center",
                             textfont=dict(color=INK, size=11), hovertemplate="%{x} total %{y:+.1%}<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.35)
    fig.update_yaxes(tickformat=".0%")
    return fig


def styles(seg: dict, attribution, height=380) -> go.Figure:
    names = [str(f) for f, g in zip(attribution["factor"].values, attribution["group"].values) if g == "style"]
    growth = seg["factor_annualized_log_return"]
    exposure = seg["style_mean_exposure"]
    names.sort(key=lambda n: growth[n])
    labels = [_label(n) for n in names]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Mean net exposure", "Contribution, log growth / yr"))
    _style(fig, height)
    ex = [exposure[n] for n in names]
    gr = [growth[n] for n in names]
    fig.add_trace(go.Bar(y=labels, x=ex, orientation="h", marker=dict(color=[POS if v >= 0 else NEG for v in ex]),
                         text=[f"{v:+.2f}" for v in ex], textposition="outside", textfont=dict(color=INK2, size=11),
                         cliponaxis=False, hovertemplate="%{y}: exposure %{x:.2f}<extra></extra>", name=""), row=1, col=1)
    fig.add_trace(go.Bar(y=labels, x=gr, orientation="h", marker=dict(color=[POS if v >= 0 else NEG for v in gr]),
                         text=[f"{v:+.1%}" for v in gr], textposition="outside", textfont=dict(color=INK2, size=11),
                         cliponaxis=False, hovertemplate="%{y}: %{x:+.2%} a year<extra></extra>", name=""), row=1, col=2)
    fig.update_xaxes(range=_padded(ex), row=1, col=1)
    fig.update_xaxes(tickformat=".0%", range=_padded(gr), row=1, col=2)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10), bargap=0.3)
    fig.update_annotations(font=dict(size=11, color=MUTED))
    return fig


def style_heatmap(attribution, height=360, seg: dict | None = None) -> go.Figure:
    group = attribution["group"].values
    names = [str(f) for f, g in zip(attribution["factor"].values, group) if g == "style"]
    if seg is not None:
        names.sort(key=lambda n: -seg["factor_annualized_log_return"][n])
    exposure = attribution["exposure"].sel(factor=names).to_pandas()
    weekly = exposure[attribution["gross_weight"].to_pandas() > 0].resample("W-FRI").mean()
    lim = float(np.nanpercentile(np.abs(weekly.values), 98)) or 1.0
    fig = _fig(height)
    fig.add_trace(go.Heatmap(
        z=weekly.T.values, x=weekly.index, y=[_label(n) for n in names], zmid=0, zmin=-lim, zmax=lim,
        colorscale=[[0, NEG], [0.5, MID], [1, POS]], xgap=0, ygap=2,
        colorbar=dict(title=dict(text="exposure", font=dict(size=11, color=MUTED)), thickness=10, len=0.9),
        hovertemplate="%{y} · week of %{x|%Y-%m-%d}<br>exposure %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", autorange="reversed")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def industries(seg: dict, attribution, n=10, height=440, key="factor_annualized_log_return",
               fmt="%{x:+.2%} a year") -> go.Figure:
    names = [str(f) for f, g in zip(attribution["factor"].values, attribution["group"].values) if g == "industry"]
    if key == "risk":
        values = {k: seg["ex_ante_risk"]["factor_contribution"][k] for k in names}
    else:
        values = {k: seg[key][k] for k in names}
    exposure = attribution["exposure"].sel(factor=names).where(attribution["gross_weight"] > 0).mean("timestamp")
    exposure = dict(zip(names, exposure.values))
    ranked = sorted(names, key=lambda k: -values[k])
    shown = ranked[:n] + ranked[-n:] if len(ranked) > 2 * n else ranked
    fig = _fig(height)
    xs = [values[k] for k in shown]
    fig.add_trace(go.Bar(
        y=[_label(k) for k in shown], x=xs, orientation="h",
        marker=dict(color=[POS if v >= 0 else NEG for v in xs]),
        text=[f"{v:+.2%}" for v in xs], textposition="outside", cliponaxis=False,
        textfont=dict(size=10, color=INK2),
        customdata=[exposure[k] for k in shown],
        hovertemplate="%{y}<br>" + fmt + "<br>mean exposure %{customdata:.1%}<extra></extra>", name="",
    ))
    fig.update_yaxes(tickvals=[_label(k) for k in shown],
                     ticktext=[f"{_label(k)}  <span style='color:{MUTED}'>{exposure[k]:.0%}</span>" for k in shown])
    fig.update_xaxes(tickformat=".2%" if key == "risk" else ".1%", nticks=6, range=_padded(xs, 1.3))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False, bargap=0.25)
    return fig


def risk_over_time(attribution, height=360) -> go.Figure:
    scale = np.sqrt(BARS_PER_YEAR)
    group = attribution["group"].values
    risk = attribution["factor_risk_contribution"].to_pandas()
    frame = pd.DataFrame({key: risk.loc[:, group == key].sum(axis=1, min_count=1) * scale for key, _, _ in PARTS[:3]})
    frame["specific"] = attribution["specific_risk_contribution"].to_pandas() * scale
    monthly = frame.resample("ME").mean()
    returns = attribution["return"].to_pandas()
    realized = returns.rolling(63).std() * scale
    fig = _fig(height)
    for key, label, colour in PARTS[:4]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[key], name=f"{label} (forecast)",
                             marker=dict(color=colour, line=dict(color="#fff", width=0.5)),
                             hovertemplate=f"%{{x|%Y-%m}} {label} %{{y:.1%}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=realized.index, y=realized.values, name="Realized (63-bar)", mode="lines",
                             line=dict(color=INK, width=2, dash="dot"),
                             hovertemplate="%{x|%Y-%m-%d} realized %{y:.1%}<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.1)
    fig.update_yaxes(tickformat=".0%", title=dict(text="annualized volatility", font=dict(size=11, color=MUTED)))
    return fig


def return_vs_risk(seg: dict, height=330) -> go.Figure:
    g, a = seg["group_annualized_log_return"], seg["annualized_log_return"]
    post = seg["ex_post_risk"]
    labels, ret, risk = [], [], []
    for key, label, _ in PARTS:
        labels.append(label)
        ret.append(g[key] if key in g else a[key])
        risk.append(post["group_contribution"].get(key) if key in g else post["term_contribution"][key])
    fig = _fig(height)
    risk = [r or 0.0 for r in risk]
    fig.add_trace(go.Bar(y=labels, x=ret, orientation="h", name="Return, log growth / yr",
                         marker=dict(color=POS), text=[f"{v:+.1%}" for v in ret], textposition="outside",
                         cliponaxis=False, textfont=dict(size=10, color=INK2),
                         hovertemplate="%{y} return %{x:+.2%}<extra></extra>"))
    fig.add_trace(go.Bar(y=labels, x=risk, orientation="h", name="Realized risk contribution",
                         marker=dict(color="#86b6ef"), text=[f"{v:+.1%}" for v in risk], textposition="outside",
                         cliponaxis=False, textfont=dict(size=10, color=INK2),
                         hovertemplate="%{y} risk %{x:.2%}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", range=_padded(ret + risk, 1.15))
    return fig


def risk_split(block: dict, height=260) -> go.Figure:
    """Forecast volatility split, x-sigma-rho, per segment: stacked horizontal bars."""
    fig = _fig(height)
    segs = [(name, s) for name, s in (("Whole", block.get("whole")), ("In-sample", block.get("in_sample")),
                                      ("Out-of-sample", block.get("out_of_sample"))) if s]
    for key, label, colour in PARTS[:4]:
        xs = [(s["ex_ante_risk"]["group_contribution"].get(key) if key != "specific"
               else s["ex_ante_risk"]["contribution"]["specific"]) for _, s in segs]
        fig.add_trace(go.Bar(y=[n for n, _ in segs], x=xs, orientation="h", name=label,
                             marker=dict(color=colour, line=dict(color="#fff", width=2)),
                             text=[f"{label} {x:.1%}" if x and x > 0.25 * max(s["ex_ante_risk"]["volatility"]["total"] for _, s in segs) else ""
                                   for x in xs], textposition="inside", insidetextanchor="middle", textangle=0,
                             textfont=dict(size=11, color="#fff"),
                             hovertemplate=f"%{{y}} {label} %{{x:.2%}}<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.35)
    fig.update_xaxes(tickformat=".0%", title=dict(text="contribution to forecast volatility (sums to it)",
                                                  font=dict(size=11, color=MUTED)))
    fig.update_yaxes(autorange="reversed")
    return fig


def style_risk(seg: dict, attribution, height=380) -> go.Figure:
    names = [str(f) for f, g in zip(attribution["factor"].values, attribution["group"].values) if g == "style"]
    risk = seg["ex_ante_risk"]["factor_contribution"]
    growth = seg["factor_annualized_log_return"]
    names.sort(key=lambda n: growth[n])
    post = seg["ex_post_risk"]["factor_contribution"]
    labels = [_label(n) for n in names]
    xs = [risk[n] for n in names]
    ps = [post[n] or 0.0 for n in names]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Forecast (x-sigma-rho)", "Realized, cov(c, r) / sigma(r)"))
    _style(fig, height)
    for col, values, label in ((1, xs, "forecast"), (2, ps, "realized")):
        fig.add_trace(go.Bar(y=labels, x=values, orientation="h",
                             marker=dict(color=[POS if v >= 0 else NEG for v in values]),
                             text=[f"{v:+.2%}" for v in values], textposition="outside", cliponaxis=False,
                             textfont=dict(size=11, color=INK2),
                             hovertemplate=f"%{{y}}: %{{x:+.2%}} of {label} vol<extra></extra>", name=""), row=1, col=col)
        fig.update_xaxes(tickformat=".1%", range=_padded(values), row=1, col=col)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10), bargap=0.3)
    fig.update_annotations(font=dict(size=11, color=MUTED))
    return fig


def risk_waterfall(seg: dict, height=330) -> go.Figure:
    """Forecast volatility built from its x-sigma-rho parts, beside the realized volatility."""
    ante = seg["ex_ante_risk"]
    values = [ante["group_contribution"].get(k, 0.0) for k, _, _ in PARTS[:3]] + [ante["contribution"]["specific"]]
    labels = [label for _, label, _ in PARTS[:4]]
    total = ante["volatility"]["total"]
    realized = seg["ex_post_risk"]["volatility"]
    fig = _fig(height)
    fig.add_trace(go.Waterfall(
        x=labels + ["Forecast"], y=values + [total], measure=["relative"] * len(values) + ["total"],
        text=[f"{v:+.1%}" for v in values] + [f"{total:.1%}"], textposition="outside", textfont=dict(color=INK2),
        increasing=dict(marker=dict(color="#86b6ef")), decreasing=dict(marker=dict(color=NEG)),
        totals=dict(marker=dict(color=INK2)), connector=dict(line=dict(color=AXIS, width=1)),
        hovertemplate="%{x}<br>%{y:.2%} of annualized volatility<extra></extra>", name="",
    ))
    if realized is not None:
        fig.add_trace(go.Bar(x=["Realized"], y=[realized], marker=dict(color=INK), text=[f"{realized:.1%}"],
                             textposition="outside", textfont=dict(color=INK2),
                             hovertemplate="realized %{y:.2%}<extra></extra>", name=""))
    running = np.cumsum(values)
    top = max(float(running.max()), total, realized or 0.0)
    bottom = min(float(running.min()), 0.0)
    fig.update_yaxes(tickformat=".0%", range=[bottom - (top - bottom) * 0.1, top * 1.15])
    fig.update_layout(showlegend=False)
    return fig


def coverage(attribution, height=140) -> go.Figure:
    series = attribution["covered_weight"].to_pandas()
    fig = _fig(height)
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(color=POS, width=1.5),
                             fill="tozeroy", fillcolor="rgba(42,120,214,.12)", name="",
                             hovertemplate="%{x|%Y-%m-%d} covered %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[max(0, float(np.nanmin(series.values)) - 0.05), 1.01])
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# the factor table (F3)
# ---------------------------------------------------------------------------


def _bar_cell(value, lim, fmt):
    if value is None or not np.isfinite(value):
        return "<td>—</td>"
    width = min(abs(value) / lim, 1) * 50 if lim else 0
    left = 50 if value >= 0 else 50 - width
    colour = POS if value >= 0 else NEG
    return (f'<td><span class="num">{fmt(value)}</span><span class="bar"><b></b>'
            f'<i style="left:{left:.1f}%;width:{width:.1f}%;background:{colour}"></i></span></td>')


def _spark(values: np.ndarray, colour: str) -> str:
    v = values[np.isfinite(values)]
    if v.size < 2:
        return ""
    lo, hi = min(v.min(), 0), max(v.max(), 0)
    span = (hi - lo) or 1
    xs = np.linspace(0, 100, v.size)
    pts = " ".join(f"{x:.1f},{20 - (y - lo) / span * 18 - 1:.1f}" for x, y in zip(xs, v))
    zero = 20 - (0 - lo) / span * 18 - 1
    return (f'<svg width="100" height="20" viewBox="0 0 100 20"><line x1="0" x2="100" y1="{zero:.1f}" y2="{zero:.1f}" '
            f'stroke="{AXIS}" stroke-width="0.6"/><polyline fill="none" stroke="{colour}" stroke-width="1.2" points="{pts}"/></svg>')


def factor_table(seg: dict, attribution) -> str:
    factors = [str(f) for f in attribution["factor"].values]
    group = dict(zip(factors, (str(g) for g in attribution["group"].values)))
    holding = attribution["gross_weight"].values > 0
    exposure = attribution["exposure"].to_pandas()[holding]
    weekly = exposure.resample("W-FRI").mean()
    mean_exposure = exposure.mean()
    growth = seg["factor_annualized_log_return"]
    ante = seg["ex_ante_risk"]["factor_contribution"]
    post = seg["ex_post_risk"]["factor_contribution"]
    pct = lambda v: f"{v:+.2%}"
    rows = []
    for key, label, colour in PARTS[:3]:
        members = [f for f in factors if group[f] == key]
        if not members:
            continue
        # Bars scaled within the group, so 48 small industries stay readable.
        lim_e = float(np.abs(mean_exposure[members].values).max()) or 1
        lim_g = max(abs(growth[f]) for f in members) or 1
        lim_r = max(abs(ante[f]) for f in members if ante[f] is not None) or 1
        total_g = sum(growth[f] for f in members)
        total_r = sum(ante[f] for f in members)
        body = [f'<tr class="grp"><td colspan="6">{label}<span>{len(members)} factor(s) · '
                f'{total_g:+.2%} a year · {total_r:.2%} of forecast vol — click to fold</span></td></tr>']
        for f in sorted(members, key=lambda x: -growth[x]):
            body.append(
                f'<tr class="f" data-e="{mean_exposure[f]:.6f}" data-g="{growth[f]:.6f}" data-r="{ante[f]:.6f}" '
                f'data-p="{(post[f] or 0):.6f}"><td class="name">{html.escape(_label(f))}</td>'
                + _bar_cell(mean_exposure[f], lim_e, lambda v: f"{v:+.2f}")
                + _bar_cell(growth[f], lim_g, pct)
                + _bar_cell(ante[f], lim_r, lambda v: f"{v:.2%}")
                + f"<td>{'—' if post[f] is None else f'{post[f]:+.2%}'}</td>"
                + f"<td>{_spark(weekly[f].values, colour)}</td></tr>"
            )
        rows.append(f'<tbody data-open="1">{"".join(body)}</tbody>')
    head = ('<thead><tr><th>Factor</th><th data-k="e">Mean exposure ↕</th><th data-k="g">Contribution / yr ↕</th>'
            '<th data-k="r">Forecast risk ↕</th><th data-k="p">Realized risk ↕</th><th>Exposure over time</th></tr></thead>')
    return f'<div style="overflow-x:auto"><table class="ft">{head}{"".join(rows)}</table></div>{_TABLE_JS}'


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------


def f1(block, attribution, metrics) -> str:
    name, seg = _headline(block, metrics)
    return f"""<div class="fa">{_CSS}{tiles(seg, name)}
{_card("Where did the return come from?", f"Annualized log growth by part, {name}; the bars add up to the total.", _div(waterfall(seg)))}
{_card("When did it come?", "Cumulative log contribution of each part; the black line is log NAV.", _div(cumulative(attribution)))}
{_card("Which years?", "Log contribution per calendar year, stacked; the diamond is the year's total.", _div(yearly(attribution)))}
<div class="row">
{_card("Which styles did the book carry, and did they pay?", "Mean net exposure and annualized contribution per style, sorted by contribution.", _div(styles(seg, attribution)))}
{_card("How did the style bets move?", "Weekly mean net exposure per style; blue long, red short.", _div(style_heatmap(attribution)))}
</div>
{_card("Which industries?", "The 10 best and 10 worst industries by annualized contribution; beside each name, the book's mean net weight in it.", _div(industries(seg, attribution)))}
<div class="row">
{_card("What risk was taken?", "Monthly mean of the forecast volatility, split by x-sigma-rho (the bars add up to the forecast); dotted: realized 63-bar volatility.", _div(risk_over_time(attribution)))}
{_card("Was the risk paid?", "Each part's return against its contribution to realized volatility.", _div(return_vs_risk(seg)))}
</div>
{_card("How much of the book is covered?", f"Covered share of the gross held weight; mean {seg['coverage']['mean_covered_weight']:.1%}.", _div(coverage(attribution)))}
</div>"""


def f2(block, attribution, metrics) -> str:
    name, seg = _headline(block, metrics)
    pair = lambda left, right: f'<div class="row">{left}{right}</div>'
    return f"""<div class="fa">{_CSS}{tiles(seg, name)}
<div class="row"><div class="colhead">Return</div><div class="colhead">Risk</div></div>
{pair(_card("Where did the return come from?", f"Annualized log growth by part, {name}; the bars add up to the total.", _div(waterfall(seg))),
      _card("Where did the risk come from?", "Forecast volatility built from each part's x-sigma-rho contribution, against the realized volatility.", _div(risk_waterfall(seg))))}
{pair(_card("Return over time", "Cumulative log contribution of each part; black: log NAV.", _div(cumulative(attribution))),
      _card("Risk over time", "Forecast volatility by part, monthly mean (bars add up to the forecast); dotted: realized 63-bar volatility.", _div(risk_over_time(attribution, height=380))))}
{pair(_card("Styles: exposure and return", "Mean net exposure and annualized contribution, sorted by contribution.", _div(styles(seg, attribution))),
      _card("Styles: risk", "Each style's contribution to forecast and to realized volatility, same order.", _div(style_risk(seg, attribution))))}
{pair(_card("Industries: return", "Best and worst 10 by annualized contribution; grey: the book's mean net weight.", _div(industries(seg, attribution))),
      _card("Industries: risk", "Largest and smallest 10 contributions to forecast volatility; grey: mean net weight.", _div(industries(seg, attribution, key="risk", fmt="%{x:.2%} of forecast vol"))))}
{_card("How did the style bets move?", "Weekly mean net exposure per style; blue long, red short. Exposure drives both columns.", _div(style_heatmap(attribution, seg=seg)))}
{_card("Was the risk paid?", "Each part's annualized return against its contribution to realized volatility.", _div(return_vs_risk(seg)))}
</div>"""


def f3(block, attribution, metrics) -> str:
    name, seg = _headline(block, metrics)
    return f"""<div class="fa">{_CSS}{tiles(seg, name)}
<div class="row">
{_card("Return by part", f"Annualized log growth, {name}.", _div(waterfall(seg, height=280)))}
{_card("Return over time", "Cumulative log contribution; black: log NAV.", _div(cumulative(attribution, height=280)))}
</div>
{_card("Every factor", "Mean net exposure, annualized contribution, contribution to forecast volatility (x-sigma-rho) and to realized volatility, "
       "and the weekly exposure. Click a column to sort within groups, a group row to fold it.", factor_table(seg, attribution))}
</div>"""


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

_OPEN_FA = """<script>
window.addEventListener('load', function () {
  document.querySelectorAll('nav button').forEach(function (b) { if (b.textContent === 'Factor attribution') b.click(); });
});
</script>"""


def main(args_path: str, out_dir: str) -> None:
    kwargs = pickle.loads(Path(args_path).read_bytes())
    value = kwargs.pop("value")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = (br._document, br._factor_attribution_tables, br._factor_attribution_figure, br._figure_div)
    figure_div = br._figure_div

    def b_document(*a, **k):
        page = pv._b_document(*a, **k)
        page = re.sub(r'<div class="card"><!--FA-->(.*?)<!--/FA--></div>', r"\1", page, flags=re.S)
        return page.replace("</body>", _OPEN_FA + "</body>")

    for key, build in {"F1": f1, "F2": f2, "F3": f3}.items():
        br._document = b_document
        br._factor_attribution_tables = lambda block, attribution, metrics, build=build: (
            "<!--FA-->" + build(block, attribution, metrics) + "<!--/FA-->")
        br._factor_attribution_figure = lambda *a, **k: None
        br._figure_div = lambda fig: "" if fig is None else figure_div(fig)
        try:
            br.write_backtest_report(value, out / f"report_{key}.html", **kwargs)
        finally:
            br._document, br._factor_attribution_tables, br._factor_attribution_figure, br._figure_div = saved
    pv.VARIANTS.update(VARIANTS)
    index = pv._index(list(VARIANTS))
    (out / "index.html").write_text(index, encoding="utf-8")
    print(out / "index.html")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
