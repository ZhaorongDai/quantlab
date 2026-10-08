"""PROTOTYPE, wipe me: inject three Holdings-analysis variants into the R223L5C5_231 report.

Three variants of a "holdings analysis" block at the top of the existing
Holdings tab, switchable via ?variant=A|B|C and a floating bottom bar.

Contribution of a symbol on bar i = its holding at the close of bar i-1
(share of NAV) x its valuation-price return over bar i. The NAV return less
the summed contributions is the residual (trading at the open, costs).
"""

import json
import re
import sys
from pathlib import Path

here = Path(sys.argv[1])
page = (here / "report.html").read_text(encoding="utf-8")
data = json.loads((here / "contrib.json").read_text())

hd = json.loads(re.search(r'id="holdings-data">(.*?)</script>', page, re.S).group(1))
ticker = {}
for t, company, sym in hd["names"]:
    ticker[sym] = [t, company]  # last span wins: the latest ticker
data["names"] = {s: ticker.get(s, [s, ""]) for s in data["symbols"]}

BLOCK = r"""
<style>
  #proto-ha { border: 2px dashed #f59e0b; border-radius: 10px; padding: 14px 16px; margin-bottom: 18px; }
  #proto-ha h3 { margin: 0 0 4px; font-size: 15px; }
  #proto-ha .sub { color: #6b7280; font-size: 12px; margin-bottom: 12px; }
  #proto-ha .row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; align-items: center; }
  #proto-ha .t { border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 12px; min-width: 150px; }
  #proto-ha .t .k { color: #6b7280; font-size: 11.5px; } #proto-ha .t .v { font-size: 18px; font-weight: 600; }
  #proto-ha .t .n { color: #6b7280; font-size: 11px; }
  #proto-ha .pos { color: #059669; } #proto-ha .neg { color: #dc2626; }
  #proto-ha select, #proto-ha button.seg { font: inherit; border: 1px solid #e5e7eb; background: #fff;
    border-radius: 6px; padding: 4px 8px; cursor: pointer; }
  #proto-ha button.seg.on { border-color: #2563eb; color: #2563eb; }
  #proto-ha table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 12.5px; }
  #proto-ha th, #proto-ha td { padding: 4px 8px; text-align: right; border-bottom: 1px solid #f3f4f6; }
  #proto-ha th:first-child, #proto-ha td:first-child { text-align: left; }
  #proto-ha .cols { display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start; }
  #proto-bar { position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%); z-index: 9999;
    background: #111827; color: #fff; border-radius: 999px; padding: 6px 10px; display: flex; gap: 10px;
    align-items: center; box-shadow: 0 6px 20px rgba(0,0,0,.3); font: 13px system-ui, sans-serif; }
  #proto-bar button { background: #374151; color: #fff; border: 0; border-radius: 999px; width: 28px; height: 28px;
    cursor: pointer; font-size: 15px; }
</style>
<div id="proto-ha"></div>
<div id="proto-bar"><button id="proto-prev">&larr;</button><span id="proto-label"></span><button id="proto-next">&rarr;</button></div>
<script type="application/json" id="proto-ha-data">__DATA__</script>
<script>
(function () {
  var P = JSON.parse(document.getElementById('proto-ha-data').textContent);
  var root = document.getElementById('proto-ha');
  var T = P.dates.length;
  var pct = function (x, d) { return (x * 100).toFixed(d === undefined ? 1 : d) + '%'; };
  var spct = function (x, d) { return (x > 0 ? '+' : '') + pct(x, d); };
  var cls = function (x) { return x > 0 ? 'pos' : x < 0 ? 'neg' : ''; };
  var sum = function (a) { return a.reduce(function (s, x) { return s + x; }, 0); };
  var navSum = sum(P.nav);
  function tile(k, v, n, c) {
    return '<div class="t"><div class="k">' + k + '</div><div class="v ' + (c || '') + '">' + v + '</div>' +
      (n ? '<div class="n">' + n + '</div>' : '') + '</div>';
  }
  function head(title, sub) { return '<h3>' + title + '</h3><div class="sub">' + sub + '</div>'; }
  // Top share of names by weight -> how many deciles.
  function split(i, deciles) {
    var top = 0, rest = 0;
    for (var k = 0; k < 10; k++) { if (k < deciles) top += P.dec[i][k]; else rest += P.dec[i][k]; }
    return [top, rest];
  }

  // A: by weight, over time.
  function A() {
    var cut = +(new URLSearchParams(location.search).get('cut') || 1);
    var top = [], rest = [], resid = [], nav = [], a = 0, b = 0, c = 0, n = 0;
    for (var i = 0; i < T; i++) {
      var s = split(i, cut);
      a += s[0]; b += s[1]; c += P.nav[i] - s[0] - s[1]; n += P.nav[i];
      top.push(a); rest.push(b); resid.push(c); nav.push(n);
    }
    var segs = [1, 2, 5].map(function (d) {
      return '<button class="seg' + (d === cut ? ' on' : '') + '" data-cut="' + d + '">Top ' + d * 10 + '%</button>';
    }).join(' ');
    root.innerHTML = head('Holdings analysis &middot; A: largest positions vs the rest, over time',
      'Each day the held names are ranked by weight at the previous close; the top share of names (by count) vs the rest. ' +
      'Contribution = weight &times; the name\'s return that day, summed over days (share of NAV). ' +
      'Residual = NAV return less all contributions: trading at the open and costs.') +
      '<div class="row">' + segs + '</div><div class="row">' +
      tile('Top ' + cut * 10 + '% of names', spct(a), pct(a / n, 0) + ' of the summed return', cls(a)) +
      tile('Remaining ' + (100 - cut * 10) + '%', spct(b), pct(b / n, 0) + ' of the summed return', cls(b)) +
      tile('Trading &amp; costs (residual)', spct(c, 2), '', cls(c)) +
      tile('NAV, summed daily returns', spct(n), 'not compounded') + '</div><div id="proto-a-chart"></div>';
    root.querySelectorAll('button[data-cut]').forEach(function (btn) {
      btn.onclick = function () { var u = new URL(location); u.searchParams.set('cut', btn.dataset.cut); history.replaceState(null, '', u); A(); };
    });
    Plotly.newPlot('proto-a-chart', [
      { x: P.dates, y: top, name: 'Top ' + cut * 10 + '%', line: { color: '#2563eb' } },
      { x: P.dates, y: rest, name: 'Remaining ' + (100 - cut * 10) + '%', line: { color: '#93c5fd' } },
      { x: P.dates, y: resid, name: 'Trading & costs', line: { color: '#9ca3af', dash: 'dot' } },
      { x: P.dates, y: nav, name: 'NAV', line: { color: '#111827', width: 1 } },
    ], { height: 300, margin: { l: 50, r: 10, t: 10, b: 30 }, yaxis: { tickformat: '.0%' },
         legend: { orientation: 'h', y: 1.12 }, hovermode: 'x unified' }, { displayModeBar: false, responsive: true });
  }

  // B: by name, Pareto.
  function B() {
    var year = new URLSearchParams(location.search).get('year') || 'all';
    var vals = year === 'all' ? P.sym_total : P.sym_year[year];
    var rows = P.symbols.map(function (s, j) { return { s: s, v: vals[j], d: P.held_days[j], w: P.avg_w[j] }; })
      .filter(function (r) { return r.v !== 0; });
    rows.sort(function (x, y) { return y.v - x.v; });
    var total = sum(rows.map(function (r) { return r.v; }));
    var k = Math.max(1, Math.round(rows.length * 0.1));
    var top = sum(rows.slice(0, k).map(function (r) { return r.v; }));
    var losers = rows.filter(function (r) { return r.v < 0; });
    var cum = [], acc = 0;
    rows.forEach(function (r, i) { acc += r.v; cum.push(acc); });
    var opts = ['all'].concat(P.years.map(String)).map(function (y) {
      return '<option' + (y === year ? ' selected' : '') + ' value="' + y + '">' + (y === 'all' ? 'Whole run' : y) + '</option>';
    }).join('');
    function table(list) {
      return '<table><thead><tr><th>Ticker</th><th>Company</th><th>Days held</th><th>Avg weight</th><th>Contribution</th></tr></thead><tbody>' +
        list.map(function (r) {
          var n = P.names[r.s];
          return '<tr><td>' + n[0] + '</td><td style="text-align:left">' + n[1] + '</td><td>' + r.d + '</td><td>' +
            pct(r.w, 2) + '</td><td class="' + cls(r.v) + '">' + spct(r.v, 2) + '</td></tr>';
        }).join('') + '</tbody></table>';
    }
    root.innerHTML = head('Holdings analysis &middot; B: which names made the return',
      'Every name ever held, ranked by its summed contribution (weight &times; return, summed over the days held). ' +
      'The curve adds them up from the best name to the worst. Days held and average weight are over the whole run.') +
      '<div class="row"><select id="proto-b-year">' + opts + '</select></div><div class="row">' +
      tile('Top 10% of names (' + k + ')', spct(top), pct(top / total, 0) + ' of all names\' contribution', cls(top)) +
      tile('Remaining 90% (' + (rows.length - k) + ')', spct(total - top), pct((total - top) / total, 0) + ' of it', cls(total - top)) +
      tile('Names that lost', String(losers.length), spct(sum(losers.map(function (r) { return r.v; }))) + ' together', 'neg') +
      tile('Names held', String(rows.length)) + '</div>' +
      '<div class="cols"><div id="proto-b-chart" style="flex:1 1 380px;min-width:300px"></div><div>' +
      '<div class="sub">Best 10</div>' + table(rows.slice(0, 10)) + '<div class="sub" style="margin-top:10px">Worst 5</div>' +
      table(rows.slice(-5).reverse()) + '</div></div>';
    document.getElementById('proto-b-year').onchange = function (e) {
      var u = new URL(location); u.searchParams.set('year', e.target.value); history.replaceState(null, '', u); B();
    };
    Plotly.newPlot('proto-b-chart', [
      { x: rows.map(function (r, i) { return (i + 1) / rows.length; }), y: cum, name: 'cumulative', line: { color: '#2563eb' },
        customdata: rows.map(function (r) { return P.names[r.s][0]; }),
        hovertemplate: '%{x:.0%} of names, through %{customdata}<br>%{y:+.1%}<extra></extra>' },
    ], { height: 320, margin: { l: 50, r: 10, t: 10, b: 40 }, xaxis: { tickformat: '.0%', title: 'share of names, best first' },
         yaxis: { tickformat: '.0%', title: 'cumulative contribution' },
         shapes: [{ type: 'line', x0: 0.1, x1: 0.1, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dot', color: '#f59e0b' } }] },
      { displayModeBar: false, responsive: true });
  }

  // C: decile x year table.
  function C() {
    var byYear = {};
    P.years.forEach(function (y) { byYear[y] = { d: [0,0,0,0,0,0,0,0,0,0], nav: 0, n: 0, days: 0 }; });
    for (var i = 0; i < T; i++) {
      var y = +P.dates[i].slice(0, 4), e = byYear[y];
      for (var k = 0; k < 10; k++) e.d[k] += P.dec[i][k];
      e.nav += P.nav[i]; e.n += P.cnt[i]; e.days++;
    }
    var all = { d: [0,0,0,0,0,0,0,0,0,0], nav: 0, n: 0, days: 0 };
    P.years.forEach(function (y) { var e = byYear[y]; for (var k = 0; k < 10; k++) all.d[k] += e.d[k]; all.nav += e.nav; all.n += e.n; all.days += e.days; });
    var max = Math.max.apply(null, P.years.map(function (y) { return Math.max.apply(null, byYear[y].d.map(Math.abs)); }));
    function cell(x) {
      var a = Math.min(1, Math.abs(x) / max), bg = x >= 0 ? 'rgba(37,99,235,' + (a * 0.55).toFixed(2) + ')' : 'rgba(220,38,38,' + (a * 0.55).toFixed(2) + ')';
      return '<td style="background:' + bg + '">' + spct(x) + '</td>';
    }
    function line(label, e, bold) {
      var s = sum(e.d), res = e.nav - s;
      return '<tr' + (bold ? ' style="font-weight:600"' : '') + '><td>' + label + '</td><td>' + (e.n / Math.max(1, e.days)).toFixed(0) + '</td>' +
        e.d.map(cell).join('') + '<td>' + spct(s) + '</td><td>' + spct(res, 2) + '</td><td>' + spct(e.nav) + '</td></tr>';
    }
    root.innerHTML = head('Holdings analysis &middot; C: contribution by weight decile and year',
      'Each day the held names are split into ten equal-count groups by weight at the previous close (D1 = the largest 10%). ' +
      'Cells sum each group\'s contribution over the year (share of NAV, not compounded). Residual = trading at the open and costs.') +
      '<div style="overflow-x:auto"><table><thead><tr><th>Year</th><th>Names/day</th>' +
      [1,2,3,4,5,6,7,8,9,10].map(function (k) { return '<th>D' + k + '</th>'; }).join('') +
      '<th>Held</th><th>Residual</th><th>NAV</th></tr></thead><tbody>' +
      P.years.map(function (y) { return line(String(y), byYear[y]); }).join('') + line('Whole run', all, true) +
      '</tbody></table></div><div id="proto-c-chart"></div>';
    var s = sum(all.d);
    Plotly.newPlot('proto-c-chart', [{ type: 'bar', x: [1,2,3,4,5,6,7,8,9,10].map(function (k) { return 'D' + k; }),
      y: all.d.map(function (x) { return x / s; }), marker: { color: '#2563eb' },
      hovertemplate: '%{x}: %{y:.0%} of the held names\' contribution<extra></extra>' }],
      { height: 220, margin: { l: 50, r: 10, t: 20, b: 30 }, yaxis: { tickformat: '.0%' },
        title: { text: 'Whole run: share of the held names\' contribution by decile', font: { size: 12 } } },
      { displayModeBar: false, responsive: true });
  }

  var V = { A: ['A', 'Largest positions over time', A], B: ['B', 'Which names made it', B], C: ['C', 'Decile x year', C] };
  var keys = Object.keys(V);
  function show() {
    var v = new URLSearchParams(location.search).get('variant') || 'A';
    if (!V[v]) v = 'A';
    document.getElementById('proto-label').textContent = 'PROTOTYPE ' + v + ' (' + V[v][1] + ')';
    V[v][2]();
  }
  function go(step) {
    var v = new URLSearchParams(location.search).get('variant') || 'A';
    var u = new URL(location);
    u.searchParams.set('variant', keys[(keys.indexOf(v) + step + keys.length) % keys.length]);
    history.replaceState(null, '', u); show();
  }
  document.getElementById('proto-prev').onclick = function () { go(-1); };
  document.getElementById('proto-next').onclick = function () { go(1); };
  document.addEventListener('keydown', function (e) {
    var t = e.target;
    if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable) return;
    if (e.shiftKey && e.key === 'ArrowLeft') { go(-1); e.stopImmediatePropagation(); }
    if (e.shiftKey && e.key === 'ArrowRight') { go(1); e.stopImmediatePropagation(); }
  }, true);
  function start() { if (window.Plotly) show(); else setTimeout(start, 100); }
  start();
  // Open the Holdings tab directly.
  var tab = document.querySelector('[data-tab="tab4"]');
  if (tab) tab.click();
})();
</script>
"""

payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
anchor = '<div class="tiles hd-tiles">'
assert page.count(anchor) == 1
page = page.replace(anchor, BLOCK.replace("__DATA__", payload) + anchor)
out = here / "report_holdings_analysis_PROTOTYPE.html"
out.write_text(page, encoding="utf-8")
print(out)
