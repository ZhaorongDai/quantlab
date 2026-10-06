"""PROTOTYPE, throwaway: four layouts of the backtest report page, switchable via ``?variant=``.

Question: what should ``report.html`` look like to be prettier and clearer?
Four variants of the same page, from the same real inputs, on one route:
``index.html?variant=A|B|C|D`` (a full-page frame per variant; the floating
bar at the bottom and the arrow keys cycle).

- A  Current: the page as quantlab writes it today (the baseline).
- B  Dashboard: sticky header, grouped KPI hero, left sidebar of sections
     replacing the tabs, each section a grid of cards holding its own charts
     AND the tables that explain them.
- C  Document: no tabs; one long column read top to bottom, numbered
     sections with a sticky table of contents, each chart followed by its
     tables; prints well.
- D  Tear sheet (dark): a strategy-vs-benchmark scoreboard first, then every
     chart stacked full width, the tables folded into accordions.

Run: ``uv run python quantlab/runs/prototype_report_variants.py ARGS.pkl OUT_DIR``
where ``ARGS.pkl`` holds the keyword arguments of one ``write_backtest_report``
call (``value`` included). Nothing here is production code.
"""

import pickle
import re
import sys
from pathlib import Path

from quantlab.runs import backtest_report as br

VARIANTS = {"A": "Current", "B": "Dashboard", "C": "Document", "D": "Tear sheet (dark)"}

_FONT = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">'

#: Restyles every plotly figure once loaded: white plot area, light grid, the page font.
_PLOTLY_CLEAN = """
function restyle(opts) {
  if (!window.Plotly) return;
  document.querySelectorAll('.plotly-graph-div').forEach(function (div) {
    var axes = {};
    Object.keys(div.layout || {}).forEach(function (k) {
      if (/^[xy]axis\\d*$/.test(k)) {
        axes[k + '.gridcolor'] = opts.grid; axes[k + '.zerolinecolor'] = opts.zero;
        axes[k + '.linecolor'] = opts.grid; axes[k + '.tickfont.color'] = opts.tick;
        axes[k + '.title.font.color'] = opts.tick;
      }
    });
    (div.layout.updatemenus || []).forEach(function (_, i) {
      axes['updatemenus[' + i + '].bgcolor'] = opts.plot; axes['updatemenus[' + i + '].bordercolor'] = opts.zero;
      axes['updatemenus[' + i + '].font.color'] = opts.text; axes['updatemenus[' + i + '].activecolor'] = opts.zero;
    });
    Plotly.relayout(div, Object.assign({
      paper_bgcolor: opts.paper, plot_bgcolor: opts.plot,
      'font.family': 'Inter, -apple-system, Segoe UI, sans-serif', 'font.color': opts.text,
      'legend.font.color': opts.text,
    }, axes));
  });
}
"""

#: Forwards the arrow keys to the switcher in the parent frame.
_KEYS = """
document.addEventListener('keydown', function (e) {
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') parent.postMessage({key: e.key}, '*');
});
"""


def _resize_on_show() -> str:
    return """
function resizeIn(el) {
  if (window.Plotly) el.querySelectorAll('.plotly-graph-div').forEach(function (d) { Plotly.Plots.resize(d); });
}
"""


def _keep(table_html: str) -> str:
    """Wrap every heading with its table so a grid never separates them."""
    return re.sub(r"(<h2>.*?</table>)", r'<div class="t">\1</div>', table_html, flags=re.S)


def _cards(*bodies: str) -> str:
    """One card per non-empty body."""
    return "".join(f'<div class="card">{_keep(b)}</div>' for b in bodies if b.strip())


def _tab(tabs, label):
    """The html of a tab, or the empty string."""
    return next((body for name, body in tabs if name == label), "")


# ---------------------------------------------------------------------------
# B: dashboard
# ---------------------------------------------------------------------------

_B_STYLE = """
  :root { --bg:#f4f5f7; --card:#fff; --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb;
          --pos:#059669; --neg:#dc2626; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,-apple-system,Segoe UI,sans-serif; font-size:13px; }
  header { position:sticky; top:0; z-index:5; background:#0f172a; color:#fff; padding:12px 24px;
           display:flex; align-items:baseline; gap:16px; }
  header h1 { font-size:15px; font-weight:600; margin:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  header .sub { color:#94a3b8; font-size:12px; }
  .shell { display:flex; min-height:calc(100vh - 46px); }
  nav { width:210px; flex:none; padding:20px 12px; position:sticky; top:46px; height:calc(100vh - 46px);
        border-right:1px solid var(--line); background:#fff; }
  nav button { display:block; width:100%; text-align:left; border:0; background:none; padding:9px 12px;
               border-radius:8px; font:inherit; color:#374151; cursor:pointer; margin-bottom:2px; }
  nav button:hover { background:#f3f4f6; }
  nav button.on { background:#eff6ff; color:var(--accent); font-weight:600; }
  main { flex:1; min-width:0; padding:20px 24px 80px; }
  .hero { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:20px; }
  .kpi { background:var(--card); border-radius:12px; padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,.06); border:1px solid var(--line); }
  .kl { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
  .kv { font-size:24px; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums; }
  .kv.pos { color:var(--pos); } .kv.neg { color:var(--neg); }
  .ks { font-size:12px; color:var(--muted); margin-top:2px; }
  section { display:none; } section.on { display:block; }
  .grid { display:grid; grid-template-columns:minmax(0,1fr) 380px; gap:16px; align-items:start; }
  .grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:16px; align-items:start; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px;
          box-shadow:0 1px 2px rgba(0,0,0,.05); min-width:0; overflow-x:auto; }
  .card h2, h2 { font-size:13px; font-weight:600; color:var(--ink); margin:0 0 10px; }
  .card h2 ~ h2 { margin-top:18px; }
  table.metrics, table.summary { border-collapse:collapse; width:100%; font-size:12.5px; }
  table.metrics th, table.metrics td, table.summary th, table.summary td { padding:5px 8px 5px 0; border-bottom:1px solid #f1f2f4; white-space:nowrap; }
  table.metrics thead th { text-align:right; color:var(--muted); font-weight:500; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  table.metrics thead th:first-child { text-align:left; }
  table.metrics tbody th { text-align:left; font-weight:400; color:#374151; cursor:help; }
  table.metrics td { text-align:right; font-variant-numeric:tabular-nums; }
  table.metrics td.better { color:var(--pos); font-weight:600; }
  table.metrics tr.group td { text-align:left; font-size:10.5px; color:var(--muted); text-transform:uppercase;
                              letter-spacing:.06em; padding-top:12px; border-bottom:1px solid var(--line); }
  table.summary th { text-align:left; color:var(--muted); font-weight:500; }
  table.summary td { white-space:normal; overflow-wrap:anywhere; }
  table.summary td .scroll { max-height:4.8em; overflow-y:auto; }
  ul.notes { color:#4b5563; padding-left:18px; line-height:1.55; }
  .timeline { font-size:12px; } .timeline .caption { color:var(--muted); margin:0 0 6px; }
  .timeline svg text { font-size:10px; fill:#6b7280; }
  .timeline .legend { display:flex; gap:10px; font-size:11px; color:var(--muted); margin-top:4px; flex-wrap:wrap; }
  .timeline .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
  @media (max-width: 900px) { nav { display:none; } .grid { grid-template-columns:1fr; } }
"""


def _b_document(title, summary, metrics, tabs, notes, *, benchmark_name=None, windows=None, extra_tables=None):
    m = metrics if isinstance(metrics, dict) else {}
    kpis = br._kpi_section(metrics, benchmark_name).replace('class="kpis"', 'class="hero"')
    window = (windows or {}).get("backtest") or ["", ""]
    sections = [
        ("Overview", f"""
<div class="grid2">{_cards(br._timeline_section(windows), br._split_section(m, benchmark_name) if m else '')}</div>
<div class="grid" style="margin-top:16px">
  <div class="card">{_tab(tabs, 'Performance')}</div>
  <div class="card">{br._comparison_section(m, benchmark_name) if m else ''}</div>
</div>"""),
        ("Excess", f'<div class="grid"><div class="card">{_tab(tabs, "Excess")}</div>'
                   f'<div class="card">{br._relative_section(m, benchmark_name) if m else ""}</div></div>'),
        ("Rolling", f'<div class="card">{_tab(tabs, "Rolling")}</div>'),
        ("Portfolio", f'<div class="grid"><div class="card">{_tab(tabs, "Portfolio")}</div>'
                      f'<div class="card">{br._trading_section(m) if m else ""}{br._extra_section(extra_tables)}</div></div>'),
        ("Attribution", f'<div class="card">{_tab(tabs, "Attribution")}</div>'),
        ("Factor attribution", f'<div class="card">{_tab(tabs, "Factor attribution")}</div>'),
        ("Setup & notes", f'<div class="grid2"><div class="card">{br._summary_section(summary)}</div>'
                          f'<div class="card">{br._notes_section(notes)}</div></div>'),
    ]
    sections = [(name, body) for name, body in sections if "plotly-graph-div" in body or name in ("Overview", "Setup & notes")]
    nav = "".join(f'<button class="{"on" if i == 0 else ""}" data-s="s{i}">{name}</button>' for i, (name, _) in enumerate(sections))
    body = "".join(f'<section id="s{i}" class="{"on" if i == 0 else ""}">{html}</section>' for i, (_, html) in enumerate(sections))
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{br._escape(title)}</title>{_FONT}
<style>{_B_STYLE}</style></head><body>
<header><h1>{br._escape(title)}</h1><span class="sub">{br._escape(window[0])} → {br._escape(window[1])}
{(' · vs ' + br._escape(benchmark_name)) if benchmark_name else ''}</span></header>
<div class="shell"><nav>{nav}</nav><main>{kpis}{body}</main></div>
<script>{_resize_on_show()}{_PLOTLY_CLEAN}{_KEYS}
document.querySelectorAll('nav button').forEach(function (b) {{
  b.addEventListener('click', function () {{
    document.querySelectorAll('nav button').forEach(function (x) {{ x.classList.toggle('on', x === b); }});
    document.querySelectorAll('main section').forEach(function (s) {{
      var on = s.id === b.dataset.s; s.classList.toggle('on', on); if (on) resizeIn(s);
    }});
    window.scrollTo(0, 0);
  }});
}});
window.addEventListener('load', function () {{ restyle({{paper:'#fff', plot:'#fff', grid:'#eef0f3', zero:'#d1d5db', tick:'#6b7280', text:'#111827'}}); }});
</script></body></html>"""


# ---------------------------------------------------------------------------
# C: long-read document
# ---------------------------------------------------------------------------

_C_STYLE = """
  * { box-sizing: border-box; }
  body { margin:0; color:#1f2328; background:#fff; font-family:Inter,-apple-system,Segoe UI,sans-serif; font-size:14px; line-height:1.5; }
  .page { display:grid; grid-template-columns:220px minmax(0,1100px); gap:40px; max-width:1400px; margin:0 auto; padding:32px 24px 100px; }
  aside { position:sticky; top:24px; align-self:start; font-size:13px; }
  aside .t { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#8b949e; margin-bottom:8px; }
  aside a { display:block; color:#57606a; text-decoration:none; padding:4px 0 4px 10px; border-left:2px solid #eaeef2; }
  aside a.on { color:#0969da; border-left-color:#0969da; font-weight:600; }
  h1 { font-size:26px; font-weight:700; margin:0 0 4px; letter-spacing:-.01em; overflow-wrap:anywhere; }
  .lede { color:#57606a; margin:0 0 20px; }
  .strip { display:flex; flex-wrap:wrap; border-top:2px solid #1f2328; border-bottom:1px solid #d0d7de; margin-bottom:28px; }
  .strip .kpi { flex:1 1 120px; padding:12px 14px 12px 0; }
  .kl { font-size:11px; color:#57606a; text-transform:uppercase; letter-spacing:.05em; }
  .kv { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }
  .kv.pos { color:#1a7f37; } .kv.neg { color:#cf222e; }
  .ks { font-size:12px; color:#57606a; }
  section { padding-top:8px; margin-bottom:44px; border-top:1px solid #eaeef2; }
  section > h2.sec { font-size:20px; margin:20px 0 4px; letter-spacing:-.01em; }
  section > h2.sec span { color:#8b949e; font-weight:500; margin-right:8px; }
  section > p.why { color:#57606a; margin:0 0 14px; max-width:760px; }
  .tables { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:8px 32px; margin-top:12px; }
  h2 { font-size:14px; font-weight:600; margin:16px 0 6px; }
  table.metrics, table.summary { border-collapse:collapse; width:100%; font-size:13px; }
  table.metrics th, table.metrics td, table.summary th, table.summary td { padding:4px 10px 4px 0; border-bottom:1px solid #eaeef2; white-space:nowrap; }
  table.metrics thead th { text-align:right; color:#57606a; font-weight:500; }
  table.metrics thead th:first-child { text-align:left; }
  table.metrics tbody th { text-align:left; font-weight:400; cursor:help; }
  table.metrics td { text-align:right; font-variant-numeric:tabular-nums; }
  table.metrics td.better { color:#1a7f37; font-weight:600; }
  table.metrics tr.group td { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.06em; padding-top:12px; }
  table.summary th { text-align:left; color:#57606a; font-weight:500; }
  table.summary td { white-space:normal; overflow-wrap:anywhere; }
  table.summary td .scroll { max-height:4.8em; overflow-y:auto; }
  ul.notes { color:#57606a; padding-left:18px; }
  .timeline svg text { font-size:10px; fill:#57606a; } .timeline .caption { color:#57606a; }
  .timeline .legend { display:flex; gap:10px; font-size:11px; color:#57606a; flex-wrap:wrap; }
  .timeline .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
  @media (max-width: 900px) { .page { grid-template-columns:1fr; } aside { display:none; } }
  @media print { aside { display:none; } .page { grid-template-columns:1fr; } section { break-inside:avoid-page; } }
"""


def _c_document(title, summary, metrics, tabs, notes, *, benchmark_name=None, windows=None, extra_tables=None):
    m = metrics if isinstance(metrics, dict) else {}
    window = (windows or {}).get("backtest") or ["", ""]
    against = f" against {benchmark_name}" if benchmark_name else ""
    parts = [
        ("Performance", "How the account grew, how deep it fell and how each month went" + against + ".",
         _tab(tabs, "Performance"),
         (br._comparison_section(m, benchmark_name) + br._split_section(m, benchmark_name)) if m else ""),
        ("Excess return", "What the strategy earned over the benchmark, and its worst run below it.",
         _tab(tabs, "Excess"), br._relative_section(m, benchmark_name) if m else ""),
        ("Rolling one-year", "Whether the edge was steady or came in bursts.", _tab(tabs, "Rolling"), ""),
        ("Portfolio and trading", "How much was held, how often it turned over and what trading cost.",
         _tab(tabs, "Portfolio"), (br._trading_section(m) if m else "") + br._extra_section(extra_tables)),
        ("Attribution", "Universe, selection and costs, and the score groups.", _tab(tabs, "Attribution"), ""),
        ("Factor attribution", "Which risks the return came from, and which risks the book carried.",
         _tab(tabs, "Factor attribution"), ""),
        ("Setup and windows", "What was run, on which bars, trained on which window.",
         br._timeline_section(windows), br._summary_section(summary) + br._notes_section(notes)),
    ]
    parts = [p for p in parts if p[2] or p[3]]
    toc = "".join(f'<a href="#c{i}">{i + 1}. {name}</a>' for i, (name, *_ ) in enumerate(parts))
    body = "".join(
        f'<section id="c{i}"><h2 class="sec"><span>{i + 1}</span>{name}</h2><p class="why">{why}</p>'
        f'{chart}<div class="tables">{_keep(tables)}</div></section>'
        for i, (name, why, chart, tables) in enumerate(parts)
    )
    kpis = br._kpi_section(metrics, benchmark_name).replace('class="kpis"', 'class="strip"')
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{br._escape(title)}</title>{_FONT}
<style>{_C_STYLE}</style></head><body><div class="page">
<aside><div class="t">Contents</div>{toc}</aside>
<article><h1>Backtest report</h1>
<p class="lede">{br._escape(title)} · {br._escape(window[0])} to {br._escape(window[1])}{br._escape(against)}</p>
{kpis}{body}</article></div>
<script>{_PLOTLY_CLEAN}{_KEYS}
var links = Array.from(document.querySelectorAll('aside a'));
var observer = new IntersectionObserver(function (entries) {{
  entries.forEach(function (e) {{ if (e.isIntersecting) links.forEach(function (a) {{ a.classList.toggle('on', a.hash === '#' + e.target.id); }}); }});
}}, {{rootMargin: '-20% 0px -70% 0px'}});
document.querySelectorAll('section').forEach(function (s) {{ observer.observe(s); }});
window.addEventListener('load', function () {{ restyle({{paper:'#fff', plot:'#fff', grid:'#eaeef2', zero:'#d0d7de', tick:'#57606a', text:'#1f2328'}}); }});
</script></body></html>"""


# ---------------------------------------------------------------------------
# D: tear sheet, dark, scoreboard first
# ---------------------------------------------------------------------------

_D_STYLE = """
  * { box-sizing: border-box; }
  body { margin:0; background:#0b0f17; color:#e5e7eb; font-family:Inter,-apple-system,Segoe UI,sans-serif; font-size:13px; }
  .wrap { max-width:1500px; margin:0 auto; padding:24px 24px 100px; }
  .top { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; margin-bottom:18px; }
  h1 { font-size:14px; color:#9ca3af; font-weight:500; margin:0; overflow-wrap:anywhere; }
  .big { font-size:32px; font-weight:700; color:#fff; letter-spacing:-.02em; }
  .board { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:1px; background:#1f2937;
           border:1px solid #1f2937; border-radius:14px; overflow:hidden; margin-bottom:22px; }
  .cell { background:#111827; padding:14px 16px; }
  .cell .n { font-size:11px; color:#9ca3af; text-transform:uppercase; letter-spacing:.06em; }
  .cell .pair { display:flex; align-items:baseline; gap:10px; margin-top:6px; font-variant-numeric:tabular-nums; }
  .cell .s { font-size:22px; font-weight:700; color:#fff; }
  .cell .b { font-size:13px; color:#9ca3af; }
  .cell .bar { height:4px; border-radius:2px; background:#1f2937; margin-top:10px; position:relative; }
  .cell .bar i { position:absolute; top:0; height:4px; border-radius:2px; }
  .cell .d { font-size:12px; margin-top:6px; } .up { color:#34d399; } .down { color:#f87171; }
  .chart { background:#111827; border:1px solid #1f2937; border-radius:14px; padding:12px 14px; margin-bottom:16px; }
  .chart > .lbl { font-size:12px; font-weight:600; color:#9ca3af; text-transform:uppercase; letter-spacing:.08em; margin:2px 0 8px; }
  details { background:#111827; border:1px solid #1f2937; border-radius:12px; margin-bottom:10px; }
  summary { cursor:pointer; padding:12px 16px; font-weight:600; color:#e5e7eb; list-style:none; }
  summary::before { content:'▸ '; color:#6b7280; } details[open] summary::before { content:'▾ '; }
  details > div { padding:0 16px 14px; overflow-x:auto; }
  h2 { font-size:12px; color:#9ca3af; text-transform:uppercase; letter-spacing:.06em; margin:14px 0 6px; }
  table.metrics, table.summary { border-collapse:collapse; font-size:12.5px; width:100%; }
  table.metrics th, table.metrics td, table.summary th, table.summary td { padding:4px 12px 4px 0; border-bottom:1px solid #1f2937; white-space:nowrap; }
  table.metrics thead th { text-align:right; color:#9ca3af; font-weight:500; }
  table.metrics thead th:first-child { text-align:left; }
  table.metrics tbody th { text-align:left; font-weight:400; color:#d1d5db; }
  table.metrics td { text-align:right; font-variant-numeric:tabular-nums; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  table.metrics td.better { color:#34d399; }
  table.metrics tr.group td { text-align:left; font-size:10.5px; color:#6b7280; text-transform:uppercase; letter-spacing:.06em; padding-top:12px; }
  table.summary th { text-align:left; color:#9ca3af; font-weight:500; } table.summary td { white-space:normal; }
  table.summary td .scroll { max-height:4.8em; overflow-y:auto; }
  ul.notes { color:#9ca3af; padding-left:18px; }
  .timeline svg text { font-size:10px; fill:#9ca3af; } .timeline .caption, .timeline .legend { color:#9ca3af; }
  .timeline .legend { display:flex; gap:10px; font-size:11px; flex-wrap:wrap; }
  .timeline .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
  .cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:10px; }
"""


def _scoreboard(metrics, benchmark_name):
    s = br._headline(metrics, metrics)
    b = br._headline(metrics.get("benchmark"), metrics) if benchmark_name else {}
    rows = [
        ("Total return", "Total Return [%]", "pct", True),
        ("Annualised return", "Annualized Return [%]", "pct", True),
        ("Volatility", "Annualized Volatility [%]", "pct", False),
        ("Sharpe ratio", "Sharpe Ratio", "ratio", True),
        ("Max drawdown", "Max Drawdown [%]", "neg_pct", True),
        ("Calmar ratio", "Calmar Ratio", "ratio", True),
    ]
    cells = []
    for label, key, unit, higher in rows:
        sv, bv = br._number(s.get(key)), br._number(b.get(key))
        if unit == "neg_pct":
            sv, bv = (None if sv is None else -abs(sv)), (None if bv is None else -abs(bv))
        bar, delta = "", ""
        if sv is not None and bv is not None:
            hi = max(abs(sv), abs(bv)) or 1
            bar = (f'<div class="bar"><i style="left:0;width:{50 * abs(sv) / hi:.0f}%;background:#60a5fa"></i>'
                   f'<i style="left:50%;width:{50 * abs(bv) / hi:.0f}%;background:#6b7280"></i></div>')
            good = (sv - bv > 0) == higher
            delta = f'<div class="d {"up" if good else "down"}">{"▲" if sv > bv else "▼"} {br._difference(sv, bv, "pct" if "pct" in unit else unit)}</div>'
        cells.append(
            f'<div class="cell"><div class="n">{label}</div><div class="pair"><span class="s">{br._format(sv, unit)}</span>'
            + (f'<span class="b">{br._escape(benchmark_name)} {br._format(bv, unit)}</span>' if benchmark_name else "")
            + f"</div>{bar}{delta}</div>"
        )
    return '<div class="board">' + "".join(cells) + "</div>"


def _d_document(title, summary, metrics, tabs, notes, *, benchmark_name=None, windows=None, extra_tables=None):
    m = metrics if isinstance(metrics, dict) else {}
    s = br._headline(m, m) if m else {}
    window = (windows or {}).get("backtest") or ["", ""]
    charts = "".join(f'<div class="chart"><div class="lbl">{br._escape(label)}</div>{body}</div>' for label, body in tabs)
    folds = [
        ("Strategy vs benchmark", br._comparison_section(m, benchmark_name) if m else ""),
        ("Excess over the benchmark", br._relative_section(m, benchmark_name) if m else ""),
        ("In-sample vs out-of-sample", br._split_section(m, benchmark_name) if m else ""),
        ("Trading", (br._trading_section(m) if m else "") + br._extra_section(extra_tables)),
        ("Setup and windows", br._timeline_section(windows) + br._summary_section(summary)),
        ("Notes", br._notes_section(notes)),
    ]
    details = "".join(f"<details><summary>{name}</summary><div>{_keep(html)}</div></details>" for name, html in folds if html)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{br._escape(title)}</title>{_FONT}
<style>{_D_STYLE}</style></head><body><div class="wrap">
<div class="top"><div><h1>{br._escape(title)}</h1><div class="big">{br._format(s.get('Total Return [%]'), 'pct')}
<span style="font-size:14px;color:#9ca3af;font-weight:500"> total return{br._suffix(m) if m else ''}, {br._escape(window[0])} → {br._escape(window[1])}</span></div></div></div>
{_scoreboard(m, benchmark_name) if m else ''}
<div class="cols"><div>{charts}</div></div>
{details}
</div><script>{_PLOTLY_CLEAN}{_KEYS}
document.querySelectorAll('details').forEach(function (d) {{ d.addEventListener('toggle', function () {{
  if (window.Plotly) d.querySelectorAll('.plotly-graph-div').forEach(function (x) {{ Plotly.Plots.resize(x); }}); }}); }});
window.addEventListener('load', function () {{ restyle({{paper:'#111827', plot:'#111827', grid:'#1f2937', zero:'#374151', tick:'#9ca3af', text:'#e5e7eb'}}); }});
</script></body></html>"""


# ---------------------------------------------------------------------------
# switcher route
# ---------------------------------------------------------------------------

def _index(keys) -> str:
    names = {k: VARIANTS[k] for k in keys}
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Report layout prototype</title>
<style>
  html,body {{ margin:0; height:100%; }} iframe {{ border:0; width:100%; height:100%; display:block; }}
  #bar {{ position:fixed; bottom:18px; left:50%; transform:translateX(-50%); z-index:99; display:flex; align-items:center;
         gap:6px; background:#ff3d71; color:#fff; padding:6px 8px; border-radius:999px; font:600 13px -apple-system,sans-serif;
         box-shadow:0 6px 24px rgba(0,0,0,.35); }}
  #bar button {{ border:0; background:rgba(255,255,255,.2); color:#fff; width:30px; height:30px; border-radius:50%;
                font-size:15px; cursor:pointer; }}
  #bar span {{ padding:0 10px; white-space:nowrap; }}
</style></head><body>
<iframe id="f"></iframe>
<div id="bar"><button id="p">←</button><span id="l"></span><button id="n">→</button></div>
<script>
var names = {names!r}; var keys = Object.keys(names);
function show(k) {{
  var u = new URL(location); u.searchParams.set('variant', k); history.replaceState(null, '', u);
  document.getElementById('f').src = 'report_' + k + '.html';
  document.getElementById('l').textContent = 'PROTOTYPE · ' + k + ' (' + names[k] + ')';
}}
function step(d) {{
  var k = new URL(location).searchParams.get('variant') || keys[0];
  show(keys[(keys.indexOf(k) + d + keys.length) % keys.length]);
}}
document.getElementById('p').onclick = function () {{ step(-1); }};
document.getElementById('n').onclick = function () {{ step(1); }};
function onKey(key) {{ if (key === 'ArrowLeft') step(-1); if (key === 'ArrowRight') step(1); }}
document.addEventListener('keydown', function (e) {{ onKey(e.key); }});
window.addEventListener('message', function (e) {{ if (e.data && e.data.key) onKey(e.data.key); }});
var start = new URL(location).searchParams.get('variant'); show(keys.indexOf(start) >= 0 ? start : keys[0]);
</script></body></html>"""


def main(args_path: str, out_dir: str) -> None:
    kwargs = pickle.loads(Path(args_path).read_bytes())
    value = kwargs.pop("value")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    original = br._document
    documents = {"A": original, "B": _b_document, "C": _c_document, "D": _d_document}
    for key, document in documents.items():
        if key == "A":
            def document(*a, **k):  # the page as it is, plus the key forwarding
                return original(*a, **k).replace("</body>", f"<script>{_KEYS}</script></body>")
        br._document = document
        try:
            br.write_backtest_report(value, out / f"report_{key}.html", **kwargs)
        finally:
            br._document = original
    (out / "index.html").write_text(_index(documents), encoding="utf-8")
    print(out / "index.html")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
