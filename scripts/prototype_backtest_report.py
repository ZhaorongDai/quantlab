"""PROTOTYPE, throwaway: three layouts of a reworked backtest report, switchable with ?variant=A|B|C.

Question: what should the backtest report look like once the metric tables are
tidied (no empty/duplicate columns, unit-aware formatting, groups, strategy |
benchmark | difference side by side, one drawdown sign, metric definitions) and
it gains a cumulative excess-return chart (log / arithmetic toggle), rolling
one-year excess return / information ratio / beta, and the portfolio structure
per rebalance (turnover, holdings, gross exposure)?

Not library code: no tests, no error handling. Run it on a finished run directory:

    uv run python scripts/prototype_backtest_report.py <run_dir> <out.html>
"""

import html as _html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
metrics = json.loads((run_dir / "metrics.json").read_text())
config = json.loads((run_dir / "config.json").read_text())
eq = xr.open_zarr(run_dir / "equity.zarr").load()
W = xr.open_zarr(run_dir / "weights.zarr")["weight"].transpose("timestamp", "symbol").load()
ts = pd.DatetimeIndex(eq.timestamp.values)
r = pd.Series(eq["returns"].values, ts).fillna(0.0)
b = pd.Series(eq["benchmark_returns"].values, ts).fillna(0.0)
value = pd.Series(eq["value"].values, ts)
bvalue = pd.Series(eq["benchmark_value"].values, ts)
bench = "QQQ"
YEAR = 252

# --------------------------------------------------------------------------- data


def pick(block, key):
    """A metric from the whole window, else from out_of_sample (period stats live there)."""
    for name in (block, "out_of_sample"):
        d = metrics.get(name) or {}
        if d.get(key) is not None:
            return d[key]
    return None


def days(text):
    return None if text is None else pd.Timedelta(text).days


strategy = metrics["whole"]
benchmark = metrics.get("benchmark", {}).get("whole", {})
relative = metrics["relative"]["whole"]

#: (group, label, strategy value, benchmark value, unit, definition)
ROWS = [
    ("Returns", "Total return", pick("whole", "Total Return [%]"), benchmark.get("Total Return [%]"), "pct",
     "Compounded return over the window."),
    ("Returns", "Annualised return", pick("whole", "Annualized Return [%]"), benchmark.get("Annualized Return [%]"), "pct",
     "Total return compounded to one year."),
    ("Risk", "Annualised volatility", pick("whole", "Annualized Volatility [%]"), benchmark.get("Annualized Volatility [%]"), "pct",
     "Standard deviation of daily returns, times sqrt(252)."),
    ("Risk", "Max drawdown", -pick("whole", "Max Drawdown [%]"), -benchmark["Max Drawdown [%]"], "pct",
     "Deepest fall of the NAV from its running peak (negative)."),
    ("Risk", "Longest drawdown", days(pick("whole", "Max Drawdown Duration")), days(benchmark.get("Max Drawdown Duration")), "days",
     "Longest time spent below a previous peak, in calendar days."),
    ("Risk", "Daily VaR (95%)", pick("whole", "Value at Risk"), benchmark.get("Value at Risk"), "frac_pct",
     "5th percentile of daily returns."),
    ("Risk-adjusted", "Sharpe ratio", pick("whole", "Sharpe Ratio"), benchmark.get("Sharpe Ratio"), "ratio",
     "Annualised mean over annualised volatility of daily returns (no risk-free rate)."),
    ("Risk-adjusted", "Sortino ratio", pick("whole", "Sortino Ratio"), benchmark.get("Sortino Ratio"), "ratio",
     "Like Sharpe, with downside deviation in the denominator."),
    ("Risk-adjusted", "Calmar ratio", pick("whole", "Calmar Ratio"), benchmark.get("Calmar Ratio"), "ratio",
     "Annualised return over max drawdown."),
]
RELATIVE = [
    ("Excess return (geometric)", relative["Excess Return [%]"], "pct",
     "Strategy NAV / benchmark NAV - 1 at the end: what the strategy earned on top of holding the benchmark."),
    ("Annualised excess return", relative["Annualized Excess Return [%]"], "pct",
     "The geometric excess compounded to one year."),
    ("Total return difference (arithmetic)", relative["Total Return Difference [%]"], "pct",
     "Strategy total return minus benchmark total return. Differs from the geometric excess by compounding."),
    ("Excess max drawdown", relative["Excess Max Drawdown [%]"], "pct",
     "Deepest fall of the relative NAV (strategy / benchmark) from its running peak, which starts at 1."),
    ("Tracking error", relative["Tracking Error [%]"], "pct",
     "Annualised standard deviation of daily excess returns r - b."),
    ("Information ratio", relative["Information Ratio"], "ratio",
     "Annualised mean of r - b over the tracking error."),
    ("Beta", relative["Beta"], "ratio", "Slope of the strategy's daily returns on the benchmark's."),
    ("Correlation", relative["Correlation"], "ratio", "Correlation of daily returns with the benchmark."),
    ("CAPM alpha", relative["CAPM Alpha [%]"], "pct",
     "Annualised intercept of that regression: return not explained by beta."),
    ("Days beating the benchmark", relative["Win Rate vs Benchmark [%]"], "pct0",
     "Share of bars with r > b."),
]
TRADING = [
    ("Annualised turnover", pick("whole", "Annualized Turnover [%]"), "pct0",
     "One-way traded value per year, as a share of the portfolio."),
    ("Turnover per rebalance", pick("whole", "Turnover per Rebalance [%]"), "pct0",
     "Mean one-way traded value per rebalance."),
    ("Fees paid", pick("whole", "Total Fees Paid"), "money", "Fees and slippage charged by the simulation."),
    ("Orders filled", pick("whole", "Total Orders"), "int", "Fills that happened over the window."),
    ("Orders rejected", metrics["execution"]["rejected_order_count"], "int",
     "Orders without a fill price at the next bar; the holding was kept."),
    ("Round trips closed", pick("whole", "Total Closed Trades"), "int", "Entry-to-flat round trips per symbol."),
    ("Round-trip win rate", pick("whole", "Win Rate [%]"), "pct0", "Share of closed round trips with a profit."),
    ("Profit factor", pick("whole", "Profit Factor"), "ratio", "Gross profit over gross loss of closed round trips."),
    ("Rebalances held after a failure", metrics["portfolio_construction"]["failed_bar_count"], "int",
     "Bars the constructor could not decide; the backtest held the position."),
]


def fmt(v, unit, signed=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    s = "+" if signed and v > 0 else ""
    if unit == "pct":
        return f"{s}{v:,.2f}%"
    if unit == "pct0":
        return f"{s}{v:,.0f}%"
    if unit == "frac_pct":
        return f"{s}{v * 100:,.2f}%"
    if unit == "ratio":
        return f"{s}{v:,.2f}"
    if unit == "days":
        return f"{s}{v:,.0f} d"
    if unit == "money":
        return f"{s}{v:,.0f}"
    if unit == "int":
        return f"{s}{int(v):,}"
    return str(v)


def diff(a, c, unit):
    if a is None or c is None:
        return "—"
    d = a - c
    if unit == "pct":
        return fmt(d, "ratio", signed=True) + " pp"
    if unit == "frac_pct":
        return fmt(d * 100, "ratio", signed=True) + " pp"
    return fmt(d, unit, signed=True)


def comparison_table(groups=True, compact=False):
    head = f"<tr><th></th><th>Strategy</th><th>{bench}</th><th>Difference</th></tr>"
    rows, last = [], None
    for g, label, sv, bv, unit, desc in ROWS:
        if groups and g != last:
            rows.append(f'<tr class="group"><td colspan="4">{g}</td></tr>')
            last = g
        rows.append(f'<tr><th title="{_html.escape(desc)}">{label}<span class="q">?</span></th><td>{fmt(sv, unit)}</td>'
                    f"<td>{fmt(bv, unit)}</td><td class='d'>{diff(sv, bv, unit)}</td></tr>")
    return f'<table class="m"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def list_table(items, title=None):
    rows = "".join(
        f'<tr><th title="{_html.escape(desc)}">{label}<span class="q">?</span></th><td>{fmt(v, unit, signed=unit == "pct" and "excess" in label.lower() or label == "CAPM alpha")}</td></tr>'
        for label, v, unit, desc in items)
    cap = f"<caption>{title}</caption>" if title else ""
    return f'<table class="m">{cap}<tbody>{rows}</tbody></table>'


def kpi(label, v, sub, good=None):
    cls = "" if good is None else (" pos" if good else " neg")
    return f'<div class="kpi"><div class="kl">{label}</div><div class="kv{cls}">{v}</div><div class="ks">{sub}</div></div>'


def kpi_strip():
    return '<div class="kpis">' + "".join([
        kpi("Total return", fmt(pick("whole", "Total Return [%]"), "pct"), f"{bench} {fmt(benchmark['Total Return [%]'], 'pct')}"),
        kpi("Excess return", fmt(relative["Excess Return [%]"], "pct", True), f"annualised {fmt(relative['Annualized Excess Return [%]'], 'pct', True)}",
            relative["Excess Return [%]"] > 0),
        kpi("Information ratio", fmt(relative["Information Ratio"], "ratio"), f"tracking error {fmt(relative['Tracking Error [%]'], 'pct')}"),
        kpi("Sharpe", fmt(pick("whole", "Sharpe Ratio"), "ratio"), f"{bench} {fmt(benchmark['Sharpe Ratio'], 'ratio')}"),
        kpi("Max drawdown", fmt(-pick("whole", "Max Drawdown [%]"), "pct"), f"{bench} {fmt(-benchmark['Max Drawdown [%]'], 'pct')}"),
        kpi("Beta", fmt(relative["Beta"], "ratio"), f"correlation {fmt(relative['Correlation'], 'ratio')}"),
        kpi("Turnover / year", fmt(pick("whole", "Annualized Turnover [%]"), "pct0"), f"fees {fmt(pick('whole', 'Total Fees Paid'), 'money')}"),
    ]) + "</div>"


# ------------------------------------------------------------------------ figures
PLOTLY = dict(template="plotly_white", margin=dict(l=60, r=20, t=40, b=30), hovermode="x unified")
excess = r - b
log_cum = np.log((1 + r) / (1 + b)).cumsum()
arith_cum = excess.cumsum()


def fig_cum_excess(height=320):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=log_cum, name="log: Σ log((1+r)/(1+b))", line=dict(color="#2b6cb0")))
    fig.add_trace(go.Scatter(x=ts, y=arith_cum, name="arithmetic: Σ (r − b)", line=dict(color="#c05621"), visible=False))
    fig.add_hline(y=0, line=dict(color="#999", width=1))
    fig.update_layout(**PLOTLY, height=height, title=f"Cumulative excess return vs {bench}", yaxis_tickformat=".0%",
                      showlegend=False,
                      updatemenus=[dict(type="buttons", direction="right", x=1, y=1.18, xanchor="right", buttons=[
                          dict(label="Log", method="update", args=[{"visible": [True, False]}]),
                          dict(label="Arithmetic", method="update", args=[{"visible": [False, True]}]),
                      ])])
    return fig


def rolling():
    rel = (1 + r) / (1 + b)
    roll_excess = rel.rolling(YEAR).apply(np.prod, raw=True) - 1
    roll_ir = excess.rolling(YEAR).mean() / excess.rolling(YEAR).std() * np.sqrt(YEAR)
    roll_beta = r.rolling(YEAR).cov(b) / b.rolling(YEAR).var()
    return roll_excess, roll_ir, roll_beta


def fig_rolling(height=420, rows=True):
    ex, ir, beta = rolling()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Rolling 1y excess return", "Rolling 1y information ratio", "Rolling 1y beta"))
    fig.add_trace(go.Scatter(x=ts, y=ex, name="1y excess", line=dict(color="#2b6cb0")), 1, 1)
    fig.add_trace(go.Scatter(x=ts, y=ir, name="1y IR", line=dict(color="#6b46c1")), 2, 1)
    fig.add_trace(go.Scatter(x=ts, y=beta, name="1y beta", line=dict(color="#2f855a")), 3, 1)
    for row, ref in ((1, 0), (2, 0), (3, 1)):
        fig.add_hline(y=ref, line=dict(color="#999", width=1), row=row, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_layout(**PLOTLY, height=height, showlegend=False)
    return fig


def structure():
    Wv = W.values
    reb = np.isfinite(Wv).all(axis=1)
    rows = np.flatnonzero(reb)
    t_reb = ts[rows]
    Wr = np.nan_to_num(Wv[rows])
    prev = np.vstack([np.zeros(Wr.shape[1]), Wr[:-1]])
    turnover = 0.5 * np.abs(Wr - prev).sum(axis=1)  # one-way, against the previous target
    holdings = (np.abs(Wr) > 1e-9).sum(axis=1)
    gross = np.abs(Wr).sum(axis=1)
    return t_reb, turnover, holdings, gross


def fig_structure(height=380):
    t, turnover, holdings, gross = structure()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("One-way turnover per rebalance (vs previous target)", "Holdings", "Gross exposure"))
    fig.add_trace(go.Bar(x=t, y=turnover, name="turnover", marker_color="#c05621"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=holdings, name="holdings", mode="lines", line=dict(color="#2b6cb0", shape="hv")), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=gross, name="gross", mode="lines", line=dict(color="#2f855a", shape="hv")), 3, 1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    fig.update_layout(**PLOTLY, height=height, showlegend=False)
    return fig


def fig_nav(height=300):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=value / value.iloc[0], name="Strategy", line=dict(color="#1a202c")))
    fig.add_trace(go.Scatter(x=ts, y=bvalue / bvalue.iloc[0], name=bench, line=dict(color="#a0aec0")))
    fig.update_layout(**PLOTLY, height=height, title="NAV (x initial)", legend=dict(orientation="h", y=1.12, x=0))
    return fig


def fig_drawdowns(height=260):
    rel = value / bvalue
    dd = value / value.cummax() - 1
    bdd = bvalue / bvalue.cummax() - 1
    xdd = rel / np.maximum(rel.cummax(), rel.iloc[0]) - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=dd, name="Strategy", line=dict(color="#1a202c")))
    fig.add_trace(go.Scatter(x=ts, y=bdd, name=bench, line=dict(color="#a0aec0")))
    fig.add_trace(go.Scatter(x=ts, y=xdd, name=f"Excess vs {bench}", line=dict(color="#c53030", dash="dot")))
    fig.update_layout(**PLOTLY, height=height, title="Drawdowns", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=1.15, x=0))
    return fig


def div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


setup = (f"{ts[0].date()} .. {ts[-1].date()} ({len(ts)} bars) · out-of-sample only · rebalance every "
         f"{config.get('rebalance_periods')} bars · {config['constructor']['name'].rsplit('.', 1)[-1]}"
         f"(top_n={config['constructor'].get('top_n')}) · fees {config.get('fees')} · slippage {config.get('slippage')} · benchmark {bench}")

# ----------------------------------------------------------------------- variants
# A: tear sheet. KPI strip, then every chart full width, tables at the bottom.
A = f"""
<h1>E5 TopN 10 · Nasdaq-100 <span class="sub">variant A: tear sheet</span></h1>
<p class="setup">{setup}</p>
{kpi_strip()}
{div(fig_nav())}
{div(fig_cum_excess())}
{div(fig_drawdowns())}
{div(fig_rolling())}
{div(fig_structure())}
<h2>Strategy vs {bench}</h2>{comparison_table()}
<div class="cols"><div><h2>Relative to {bench}</h2>{list_table(RELATIVE)}</div>
<div><h2>Trading</h2>{list_table(TRADING)}</div></div>
<p class="foot">Hover a metric name for its definition. Drawdowns are negative everywhere. "pp" = percentage points.</p>
"""

# B: comparison first. One consolidated table on the left, charts in tabs on the right.
B = f"""
<h1>E5 TopN 10 · Nasdaq-100 <span class="sub">variant B: comparison first</span></h1>
<p class="setup">{setup}</p>
<div class="split">
  <div class="left">
    <h2>Strategy vs {bench}</h2>{comparison_table()}
    <h2>Relative</h2>{list_table(RELATIVE)}
    <h2>Trading</h2>{list_table(TRADING)}
  </div>
  <div class="right">
    <div class="tabs">
      <button class="tab on" data-tab="t1">Performance</button><button class="tab" data-tab="t2">Excess</button>
      <button class="tab" data-tab="t3">Rolling</button><button class="tab" data-tab="t4">Portfolio</button>
    </div>
    <div class="pane on" id="t1">{div(fig_nav(360))}{div(fig_drawdowns(300))}</div>
    <div class="pane" id="t2">{div(fig_cum_excess(420))}</div>
    <div class="pane" id="t3">{div(fig_rolling(620))}</div>
    <div class="pane" id="t4">{div(fig_structure(620))}</div>
  </div>
</div>
"""

# C: question-driven. Each section answers one question with its chart next to its numbers.
C = f"""
<h1>E5 TopN 10 · Nasdaq-100 <span class="sub">variant C: by question</span></h1>
<p class="setup">{setup}</p>
<section><h2>1. Did it beat {bench}?</h2>
  <div class="qa"><div class="chart">{div(fig_cum_excess(340))}</div>
  <div class="nums">{list_table(RELATIVE[:4] + RELATIVE[5:6])}</div></div></section>
<section><h2>2. Was the edge steady?</h2>
  <div class="qa"><div class="chart">{div(fig_rolling(460))}</div>
  <div class="nums">{list_table(RELATIVE[4:5] + RELATIVE[6:])}</div></div></section>
<section><h2>3. How much risk did it take?</h2>
  <div class="qa"><div class="chart">{div(fig_nav(260))}{div(fig_drawdowns(240))}</div>
  <div class="nums">{comparison_table(groups=True)}</div></div></section>
<section><h2>4. How does it trade?</h2>
  <div class="qa"><div class="chart">{div(fig_structure(420))}</div>
  <div class="nums">{list_table(TRADING)}</div></div></section>
"""

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px 32px 80px;color:#1a202c}
h1{font-size:20px;margin:0 0 4px} .sub{font-size:12px;color:#718096;font-weight:400;margin-left:8px}
h2{font-size:15px;margin:22px 0 8px;color:#2d3748} .setup{font-size:12px;color:#4a5568;margin:0 0 14px}
.kpis{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 14px}
.kpi{border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;min-width:120px}
.kl{font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.04em}
.kv{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums} .kv.pos{color:#2f855a} .kv.neg{color:#c53030}
.ks{font-size:11px;color:#718096}
table.m{border-collapse:collapse;font-size:13px;margin-bottom:6px}
table.m caption{text-align:left;font-weight:600;padding:4px 0}
table.m th,table.m td{padding:4px 14px 4px 0;border-bottom:1px solid #edf2f7;white-space:nowrap}
table.m thead th{color:#4a5568;text-align:right} table.m thead th:first-child{text-align:left}
table.m tbody th{text-align:left;font-weight:400;color:#2d3748;cursor:help}
table.m td{text-align:right;font-variant-numeric:tabular-nums} td.d{color:#4a5568}
tr.group td{text-align:left;font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.05em;padding-top:10px;border-bottom:1px solid #cbd5e0}
.q{display:inline-block;margin-left:4px;font-size:10px;color:#a0aec0}
.cols{display:flex;gap:40px;flex-wrap:wrap} .foot{font-size:12px;color:#718096}
.split{display:flex;gap:28px;align-items:flex-start} .left{flex:0 0 auto} .right{flex:1 1 auto;min-width:0}
.tabs{display:flex;gap:4px;border-bottom:1px solid #e2e8f0;margin-top:22px}
.tab{border:0;background:none;padding:6px 12px;cursor:pointer;color:#4a5568;border-bottom:2px solid transparent}
.tab.on{color:#1a202c;border-bottom-color:#2b6cb0} .pane{display:none} .pane.on{display:block}
section{border-top:1px solid #e2e8f0;margin-top:12px} .qa{display:flex;gap:24px;align-items:flex-start}
.qa .chart{flex:1 1 auto;min-width:0} .qa .nums{flex:0 0 auto;padding-top:28px}
.variant{display:none} .variant.on{display:block}
#proto-bar{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#1a202c;color:#fff;border-radius:999px;
 padding:6px 10px;display:flex;gap:10px;align-items:center;box-shadow:0 4px 14px rgba(0,0,0,.3);font-size:13px;z-index:10}
#proto-bar button{background:#2d3748;color:#fff;border:0;border-radius:999px;width:28px;height:28px;cursor:pointer}
"""

JS = """
const names = {A: "tear sheet", B: "comparison first", C: "by question"}, keys = Object.keys(names);
function show(k){
  document.querySelectorAll('.variant').forEach(v => v.classList.toggle('on', v.id === 'v' + k));
  document.getElementById('proto-label').textContent = 'PROTOTYPE ' + k + ' (' + names[k] + ')';
  const u = new URL(location); u.searchParams.set('variant', k); history.replaceState(null, '', u);
  window.dispatchEvent(new Event('resize'));
}
function step(d){ const k = new URL(location).searchParams.get('variant') || 'A'; show(keys[(keys.indexOf(k) + d + keys.length) % keys.length]); }
document.addEventListener('keydown', e => { if (!e.target.closest('input,textarea,[contenteditable]')) { if (e.key === 'ArrowLeft') step(-1); if (e.key === 'ArrowRight') step(1); } });
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x === t));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.id === t.dataset.tab));
  window.dispatchEvent(new Event('resize'));
});
show(new URL(location).searchParams.get('variant') || 'A');
"""

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>PROTOTYPE backtest report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>{CSS}</style></head><body>
<div class="variant" id="vA">{A}</div><div class="variant" id="vB">{B}</div><div class="variant" id="vC">{C}</div>
<div id="proto-bar"><button onclick="step(-1)">←</button><span id="proto-label"></span><button onclick="step(1)">→</button></div>
<script>{JS}</script></body></html>"""
out.write_text(html, encoding="utf-8")
print(out)
