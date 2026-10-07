"""PROTOTYPE, throw away: alternative layouts of the report's Holdings tab.

Question: what should the Holdings tab of ``report.html`` look like?
Four variants of the tab on the real report page, switchable with
``?variant=A|B|C|D|E`` or the floating bar at the bottom (Shift + arrow keys too;
plain arrows still step days):

  A  Day table (what ships today)
  B  Timeline heatmap: symbols x days, click a day for its book
  C  Rebalance ledger: one rebalance at a time, entries / exits / drift
  D  Long | short book: two bar columns, exposure strip as scrubber
  E  C + A: rebalance list driving a day table over that holding period

Every variant reads the JSON the shipped tab already embeds
(``#holdings-data``), so nothing in the library changes.

Usage::

    python scripts/prototype_holdings_tab.py <run_dir>/report.html out.html
"""

import sys
from pathlib import Path

INJECT = r"""
<style>
  #proto-bar { position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%); z-index: 9999;
    display: flex; align-items: center; gap: 10px; background: #111827; color: #fff; border-radius: 999px;
    padding: 8px 14px; box-shadow: 0 6px 24px rgba(0,0,0,.35); font: 13px system-ui, sans-serif; }
  #proto-bar button { background: #374151; color: #fff; border: 0; border-radius: 999px; width: 28px; height: 28px;
    cursor: pointer; font-size: 15px; }
  #proto-bar .lbl { min-width: 230px; text-align: center; }
  #proto-bar .tag { color: #fbbf24; font-weight: 600; margin-right: 6px; }
  .pv { font-size: 13px; }
  .pv .muted { color: #6b7280; }
  .pv .pos { color: #2563eb; } .pv .neg { color: #dc2626; }
  .pv table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  .pv th { text-align: left; font-weight: 500; color: #6b7280; font-size: 11px; text-transform: uppercase;
    letter-spacing: .04em; padding: 4px 8px; border-bottom: 1px solid #e5e7eb; }
  .pv td { padding: 4px 8px; border-bottom: 1px solid #f3f4f6; }
  .pv td.n { text-align: right; }
  /* B */
  .pvb-wrap { position: relative; overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px; }
  .pvb-tip { position: fixed; pointer-events: none; background: #111827; color: #fff; padding: 4px 8px;
    border-radius: 6px; font-size: 12px; display: none; z-index: 50; }
  .pvb-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 14px; }
  /* C */
  .pvc { display: grid; grid-template-columns: 240px 1fr; gap: 18px; }
  .pvc-list { max-height: 640px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 8px; }
  .pvc-item { padding: 8px 12px; border-bottom: 1px solid #f3f4f6; cursor: pointer; }
  .pvc-item:hover { background: #f9fafb; } .pvc-item.on { background: #eff6ff; }
  .pvc-item b { display: block; } .pvc-chip { font-size: 11px; padding: 0 5px; border-radius: 4px; margin-right: 4px; }
  .pvc-chip.in { background: #ecfdf5; color: #059669; } .pvc-chip.out { background: #fef2f2; color: #dc2626; }
  .pvc h4 { margin: 14px 0 6px; font-size: 13px; }
  /* E */
  .pve { display: grid; grid-template-columns: 220px 1fr; gap: 18px; }
  .pve-days { display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }
  .pve-days button { font: inherit; font-size: 12px; border: 1px solid #e5e7eb; background: #fff; border-radius: 6px;
    padding: 3px 8px; cursor: pointer; } .pve-days button.on { background: #2563eb; color: #fff; border-color: #2563eb; }
  .pve-ctl { display: flex; gap: 8px; align-items: center; margin: 6px 0 10px; }
  .pve-ctl input, .pve-ctl button { font: inherit; border: 1px solid #e5e7eb; border-radius: 6px; padding: 4px 10px; background: #fff; }
  .pve th[data-k] { cursor: pointer; }
  .pve tr.out td { color: #9ca3af; }
  /* D */
  .pvd-strip { width: 100%; height: 90px; cursor: crosshair; border: 1px solid #e5e7eb; border-radius: 8px; }
  .pvd-gauge { display: flex; gap: 28px; margin: 12px 0; }
  .pvd-gauge div b { font-size: 20px; display: block; }
  .pvd-books { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
  .pvd-row { display: grid; grid-template-columns: 70px 1fr 64px; align-items: center; gap: 8px; padding: 3px 0; }
  .pvd-track { position: relative; height: 14px; background: #f3f4f6; border-radius: 3px; }
  .pvd-fill { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 3px; }
  .pvd-tgt { position: absolute; top: -2px; bottom: -2px; width: 2px; background: #111827; }
</style>
<div id="proto-bar"><button id="pb-prev">&#8249;</button><span class="lbl" id="pb-lbl"></span>
<button id="pb-next">&#8250;</button></div>
<div class="pvb-tip" id="pvb-tip"></div>
<script>
(function () {
  var VARIANTS = [['A', 'Day table (shipped)'], ['B', 'Timeline heatmap'], ['C', 'Rebalance ledger'],
                  ['D', 'Long | short book'], ['E', 'Ledger + day table (C + A)']];
  var data = JSON.parse(document.getElementById('holdings-data').textContent);
  var days = data.days, names = data.names;
  var pane = document.getElementById('tab2');
  var card = pane.querySelector('.card');
  var shipped = Array.prototype.slice.call(card.children, 1);
  var host = document.createElement('div'); host.className = 'pv'; card.appendChild(host);
  var params = new URLSearchParams(location.search);
  var cur = params.get('variant') || 'A';
  function pct(x, d) { return (x * 100).toFixed(d === undefined ? 2 : d) + '%'; }
  function tk(i) { return names[i][0]; }
  function el(tag, html, cls) { var e = document.createElement(tag); if (html !== undefined) e.innerHTML = html;
    if (cls) e.className = cls; return e; }
  // symbol -> per-day holding/target
  var symIdx = {}, syms = [];
  days.forEach(function (d, i) { d.h.forEach(function (r) { var s = names[r[0]][2];
    if (!(s in symIdx)) { symIdx[s] = syms.length; syms.push({s: s, first: i, name: r[0]}); } }); });
  var H = syms.map(function () { return new Float64Array(days.length); });
  var T = syms.map(function () { return new Float64Array(days.length); });
  days.forEach(function (d, i) { d.h.forEach(function (r) { var k = symIdx[names[r[0]][2]];
    H[k][i] = r[2]; T[k][i] = r[1]; }); });
  var rebs = []; days.forEach(function (d, i) { if (i === 0 || d.r !== days[i - 1].r) if (d.r) rebs.push(i); });
  function gross(i) { return days[i].h.reduce(function (s, r) { return s + Math.abs(r[2]); }, 0); }
  function net(i) { return days[i].h.reduce(function (s, r) { return s + r[2]; }, 0); }

  function bookColumns(i, target) {
    var longs = days[i].h.filter(function (r) { return r[2] > 0; });
    var shorts = days[i].h.filter(function (r) { return r[2] < 0; }).sort(function (a, b) { return a[2] - b[2]; });
    var wrap = el('div', undefined, 'pvb-cols');
    [['Long', longs, 'pos'], ['Short', shorts, 'neg']].forEach(function (g) {
      var t = '<table><tr><th>' + g[0] + ' (' + g[1].length + ')</th><th>Company</th><th style="text-align:right">Target</th>' +
        '<th style="text-align:right">Holding</th></tr>';
      g[1].forEach(function (r) { t += '<tr><td>' + tk(r[0]) + ' <span class="muted">' + names[r[0]][2] + '</span></td><td class="muted">' +
        (names[r[0]][1] || '—') + '</td><td class="n">' + pct(r[1]) + '</td><td class="n ' + g[2] + '">' + pct(r[2]) + '</td></tr>'; });
      wrap.appendChild(el('div', t + '</table>'));
    });
    target.appendChild(wrap);
  }

  // ---- B: timeline heatmap ----
  function variantB() {
    var order = syms.map(function (_, k) { return k; }).sort(function (a, b) { return syms[a].first - syms[b].first; });
    var cw = Math.max(3, Math.floor(1100 / days.length)), rh = 14, left = 70, top = 18;
    host.appendChild(el('div', '<span class="muted">Each row a symbol, each column a day; blue long, red short, ' +
      'deeper = larger holding. Ticks on top are rebalances. Click a column for that day&rsquo;s book.</span>'));
    var wrap = el('div', undefined, 'pvb-wrap'); host.appendChild(wrap);
    var c = document.createElement('canvas'); var W = left + cw * days.length + 10, Hh = top + rh * order.length + 4;
    var dpr = window.devicePixelRatio || 1; c.width = W * dpr; c.height = Hh * dpr; c.style.width = W + 'px'; c.style.height = Hh + 'px';
    wrap.appendChild(c); var g = c.getContext('2d'); g.scale(dpr, dpr);
    var max = 0; H.forEach(function (row) { row.forEach(function (v) { max = Math.max(max, Math.abs(v)); }); });
    var sel = days.length - 1, detail = el('div'); host.appendChild(detail);
    function draw() {
      g.clearRect(0, 0, W, Hh); g.font = '11px system-ui'; g.fillStyle = '#6b7280';
      rebs.forEach(function (i) { g.fillRect(left + i * cw, 4, 1, 10); });
      order.forEach(function (k, row) {
        var y = top + row * rh; g.fillStyle = '#374151'; g.fillText(tk(syms[k].name), 4, y + 11);
        for (var i = 0; i < days.length; i++) { var v = H[k][i]; if (!v) continue;
          var a = Math.min(1, Math.abs(v) / max) * 0.9 + 0.1;
          g.fillStyle = v > 0 ? 'rgba(37,99,235,' + a + ')' : 'rgba(220,38,38,' + a + ')';
          g.fillRect(left + i * cw, y + 1, cw, rh - 2); }
      });
      g.strokeStyle = '#111827'; g.lineWidth = 1.5; g.strokeRect(left + sel * cw - 0.5, top - 2, cw + 1, rh * order.length + 4);
      detail.innerHTML = '<h4 style="margin:14px 0 4px">' + days[sel].d + ' <span class="muted">targets from ' +
        (days[sel].r || '—') + ' · gross ' + pct(gross(sel), 1) + ' · net ' + pct(net(sel), 1) + ' · cash ' +
        pct(days[sel].cash) + '</span></h4>';
      bookColumns(sel, detail);
    }
    var tip = document.getElementById('pvb-tip');
    function at(ev) { var r = c.getBoundingClientRect(); var i = Math.floor((ev.clientX - r.left - left) / cw);
      var row = Math.floor((ev.clientY - r.top - top) / rh); return [i, row]; }
    c.addEventListener('mousemove', function (ev) { var p = at(ev);
      if (p[0] < 0 || p[0] >= days.length || p[1] < 0 || p[1] >= order.length) { tip.style.display = 'none'; return; }
      var k = order[p[1]]; tip.style.display = 'block'; tip.style.left = ev.clientX + 12 + 'px'; tip.style.top = ev.clientY + 12 + 'px';
      tip.textContent = tk(syms[k].name) + '  ' + days[p[0]].d + '  holding ' + pct(H[k][p[0]]) + '  target ' + pct(T[k][p[0]]); });
    c.addEventListener('mouseleave', function () { tip.style.display = 'none'; });
    c.addEventListener('click', function (ev) { var p = at(ev); if (p[0] >= 0 && p[0] < days.length) { sel = p[0]; draw(); } });
    draw();
  }

  // ---- C: rebalance ledger ----
  function variantC() {
    var box = el('div', undefined, 'pvc'); host.appendChild(box);
    var list = el('div', undefined, 'pvc-list'), body = el('div'); box.appendChild(list); box.appendChild(body);
    function setOf(i) { var o = {}; days[i].h.forEach(function (r) { if (r[1] !== 0) o[names[r[0]][2]] = r; }); return o; }
    var info = rebs.map(function (start, n) {
      var end = n + 1 < rebs.length ? rebs[n + 1] - 1 : days.length - 1;
      var now = setOf(start), before = n ? setOf(rebs[n - 1]) : {};
      var inn = Object.keys(now).filter(function (s) { return !before[s]; });
      var out = Object.keys(before).filter(function (s) { return !now[s]; });
      return {start: start, end: end, now: now, before: before, inn: inn, out: out};
    });
    var sel = info.length - 1;
    function render() {
      list.innerHTML = '';
      info.slice().reverse().forEach(function (x) { var n = info.indexOf(x);
        var it = el('div', '<b>' + days[x.start].r + '</b><span class="muted">' + Object.keys(x.now).length + ' names · ' +
          (x.end - x.start + 1) + ' days</span><br>' + (x.inn.length ? '<span class="pvc-chip in">+' + x.inn.length + '</span>' : '') +
          (x.out.length ? '<span class="pvc-chip out">−' + x.out.length + '</span>' : ''), 'pvc-item' + (n === sel ? ' on' : ''));
        it.onclick = function () { sel = n; render(); }; list.appendChild(it); });
      var x = info[sel];
      var h = '<div class="muted">Targets from ' + days[x.start].r + ' · held ' + days[x.start].d + ' … ' + days[x.end].d +
        ' · gross at fill ' + pct(gross(x.start), 1) + ' → ' + pct(gross(x.end), 1) + ' · cash ' + pct(days[x.end].cash) + '</div>';
      h += '<h4>Book over the period</h4><table><tr><th>Ticker</th><th></th><th style="text-align:right">Target</th>' +
        '<th style="text-align:right">At fill</th><th style="text-align:right">At end</th><th style="text-align:right">Drift</th><th>Path</th></tr>';
      var keys = Object.keys(x.now).sort(function (a, b) { return x.now[b][1] - x.now[a][1]; });
      keys.forEach(function (s) { var k = symIdx[s], r = x.now[s], a = H[k][x.start], b = H[k][x.end];
        h += '<tr><td>' + tk(r[0]) + '</td><td>' + (x.inn.indexOf(s) >= 0 ? '<span class="pvc-chip in">new</span>' : '') +
          '</td><td class="n">' + pct(r[1]) + '</td><td class="n">' + pct(a) + '</td><td class="n">' + pct(b) +
          '</td><td class="n ' + (b - a >= 0 ? 'pos' : 'neg') + '">' + (b - a >= 0 ? '+' : '') + pct(b - a) + '</td><td>' +
          spark(H[k], x.start, x.end) + '</td></tr>'; });
      h += '</table>';
      if (x.out.length) { h += '<h4>Exited</h4><table><tr><th>Ticker</th><th style="text-align:right">Last target</th>' +
        '<th style="text-align:right">Held after fill</th></tr>';
        x.out.forEach(function (s) { var k = symIdx[s]; h += '<tr><td>' + tk(x.before[s][0]) + '</td><td class="n">' + pct(x.before[s][1]) +
          '</td><td class="n">' + pct(H[k][x.start]) + '</td></tr>'; }); h += '</table>'; }
      body.innerHTML = h;
    }
    function spark(row, a, b) { var w = 90, hgt = 18, vals = Array.prototype.slice.call(row, a, b + 1);
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals); if (hi === lo) { hi += 1e-9; }
      var pts = vals.map(function (v, i) { return (i / Math.max(1, vals.length - 1) * w).toFixed(1) + ',' + (hgt - (v - lo) / (hi - lo) * hgt).toFixed(1); });
      return '<svg width="' + w + '" height="' + hgt + '"><polyline fill="none" stroke="' + (vals[0] < 0 ? '#dc2626' : '#2563eb') +
        '" stroke-width="1.5" points="' + pts.join(' ') + '"/></svg>'; }
    render();
  }

  // ---- D: long | short book with exposure strip ----
  function variantD() {
    host.appendChild(el('div', '<span class="muted">Gross (grey) and net (black) exposure over the run; drag or click to ' +
      'pick a day, ←/→ step. Bars are holdings, the black tick the target.</span>'));
    var c = document.createElement('canvas'); c.className = 'pvd-strip'; host.appendChild(c);
    var gauge = el('div', undefined, 'pvd-gauge'), books = el('div', undefined, 'pvd-books');
    host.appendChild(gauge); host.appendChild(books);
    var sel = days.length - 1, G = days.map(function (_, i) { return gross(i); }), N = days.map(function (_, i) { return net(i); });
    function draw() {
      var r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; c.width = r.width * dpr; c.height = r.height * dpr;
      var g = c.getContext('2d'); g.scale(dpr, dpr); var W = r.width, Hh = r.height;
      var hi = Math.max.apply(null, G.concat([1])), lo = Math.min.apply(null, N.concat([0]));
      function y(v) { return 8 + (hi - v) / (hi - lo) * (Hh - 16); } function x(i) { return i / (days.length - 1) * W; }
      g.fillStyle = '#e5e7eb'; g.beginPath(); g.moveTo(0, y(0)); G.forEach(function (v, i) { g.lineTo(x(i), y(v)); });
      g.lineTo(W, y(0)); g.fill();
      g.strokeStyle = '#111827'; g.beginPath(); N.forEach(function (v, i) { i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v)); }); g.stroke();
      g.fillStyle = '#9ca3af'; rebs.forEach(function (i) { g.fillRect(x(i), Hh - 4, 1, 4); });
      g.fillStyle = '#2563eb'; g.fillRect(x(sel) - 1, 0, 2, Hh);
      var d = days[sel];
      gauge.innerHTML = '<div><span class="muted">Date</span><b>' + d.d + '</b></div><div><span class="muted">Targets from</span><b>' +
        (d.r || '—') + '</b></div><div><span class="muted">Gross</span><b>' + pct(G[sel], 1) + '</b></div><div><span class="muted">Net</span><b>' +
        pct(N[sel], 1) + '</b></div><div><span class="muted">Cash</span><b>' + pct(d.cash, 1) + '</b></div>';
      books.innerHTML = '';
      var m = Math.max.apply(null, d.h.map(function (r) { return Math.max(Math.abs(r[1]), Math.abs(r[2])); }).concat([1e-9]));
      [['Long', function (r) { return r[2] > 0; }, '#2563eb'], ['Short', function (r) { return r[2] < 0; }, '#dc2626']].forEach(function (s) {
        var rows = d.h.filter(s[1]).sort(function (a, b) { return Math.abs(b[2]) - Math.abs(a[2]); });
        var col = el('div', '<div class="muted" style="margin-bottom:6px">' + s[0] + ' · ' + rows.length + ' names · ' +
          pct(rows.reduce(function (t, r) { return t + r[2]; }, 0), 1) + '</div>');
        rows.forEach(function (r) { var row = el('div', undefined, 'pvd-row');
          row.innerHTML = '<span title="' + (names[r[0]][1] || names[r[0]][2]) + '">' + tk(r[0]) + '</span><div class="pvd-track"><div class="pvd-fill" style="width:' +
            (Math.abs(r[2]) / m * 100) + '%;background:' + s[2] + '"></div><div class="pvd-tgt" style="left:' + (Math.abs(r[1]) / m * 100) +
            '%"></div></div><span style="text-align:right">' + pct(r[2]) + '</span>'; col.appendChild(row); });
        books.appendChild(col); });
    }
    function pick(ev) { var r = c.getBoundingClientRect(); sel = Math.max(0, Math.min(days.length - 1, Math.round((ev.clientX - r.left) / r.width * (days.length - 1)))); draw(); }
    var down = false; c.addEventListener('mousedown', function (e) { down = true; pick(e); });
    window.addEventListener('mouseup', function () { down = false; }); c.addEventListener('mousemove', function (e) { if (down) pick(e); });
    host.dvStep = function (k) { sel = Math.max(0, Math.min(days.length - 1, sel + k)); draw(); };
    window.addEventListener('resize', draw); draw();
  }

  // ---- E: rebalance list (C) driving a day table (A) over that holding period ----
  function variantE() {
    var box = el('div', undefined, 'pve'); host.appendChild(box);
    var list = el('div', undefined, 'pvc-list'), body = el('div'); box.appendChild(list); box.appendChild(body);
    function setOf(i) { var o = {}; days[i].h.forEach(function (r) { if (r[1] !== 0) o[names[r[0]][2]] = r; }); return o; }
    var info = rebs.map(function (start, n) {
      var end = n + 1 < rebs.length ? rebs[n + 1] - 1 : days.length - 1;
      var now = setOf(start), before = n ? setOf(rebs[n - 1]) : {};
      return {start: start, end: end, now: now, before: before,
              inn: Object.keys(now).filter(function (s) { return !before[s]; }),
              out: Object.keys(before).filter(function (s) { return !now[s]; })};
    });
    var day = days.length - 1, sortK = 'h', sortDir = -1, filter = '';
    function periodOf(i) { for (var n = info.length - 1; n >= 0; n--) if (info[n].start <= i) return n; return -1; }
    function spark(row, a, b, mark) { var w = 90, hgt = 18, vals = Array.prototype.slice.call(row, a, b + 1);
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals); if (hi === lo) hi += 1e-9;
      function px(i) { return i / Math.max(1, vals.length - 1) * w; } function py(v) { return hgt - 2 - (v - lo) / (hi - lo) * (hgt - 4); }
      var pts = vals.map(function (v, i) { return px(i).toFixed(1) + ',' + py(v).toFixed(1); });
      var m = mark - a; return '<svg width="' + w + '" height="' + hgt + '"><polyline fill="none" stroke="' + (vals[0] < 0 ? '#dc2626' : '#2563eb') +
        '" stroke-width="1.5" points="' + pts.join(' ') + '"/><circle cx="' + px(m).toFixed(1) + '" cy="' + py(vals[m]).toFixed(1) +
        '" r="2.5" fill="#111827"/></svg>'; }
    function rows(p) {
      var x = info[p], out = [];
      days[day].h.forEach(function (r) { var s = names[r[0]][2], k = symIdx[s];
        out.push({ticker: tk(r[0]), company: names[r[0]][1] || '', symbol: s, t: r[1], fill: H[k][x.start], h: r[2],
                  drift: r[2] - H[k][x.start], isNew: x.inn.indexOf(s) >= 0, k: k}); });
      return out;
    }
    function render() {
      var p = periodOf(day);
      list.innerHTML = '';
      info.slice().reverse().forEach(function (x) { var n = info.indexOf(x);
        var it = el('div', '<b>' + days[x.start].r + '</b><span class="muted">' + Object.keys(x.now).length + ' names · ' +
          (x.end - x.start + 1) + ' days</span><br>' + (x.inn.length ? '<span class="pvc-chip in">+' + x.inn.length + '</span>' : '') +
          (x.out.length ? '<span class="pvc-chip out">−' + x.out.length + '</span>' : ''), 'pvc-item' + (n === p ? ' on' : ''));
        it.onclick = function () { day = x.start; render(); }; list.appendChild(it); });
      var sel = list.querySelector('.on'); if (sel && sel.scrollIntoViewIfNeeded) sel.scrollIntoViewIfNeeded();
      body.innerHTML = '';
      if (p < 0) { body.appendChild(el('div', '<span class="muted">' + days[day].d + ': before the first rebalance, nothing held.</span>')); return; }
      var x = info[p], d = days[day];
      body.appendChild(el('div', '<span class="muted">Targets from <b>' + d.r + '</b> · held ' + days[x.start].d + ' … ' + days[x.end].d +
        ' · <b>' + d.d + '</b> · gross ' + pct(gross(day), 1) + ' · net ' + pct(net(day), 1) + ' · cash ' + pct(d.cash) +
        ' · other (dust) ' + pct(d.other[2], 3) + '</span>'));
      var strip = el('div', undefined, 'pve-days');
      for (var i = x.start; i <= x.end; i++) (function (i) { var b = el('button', days[i].d.slice(5), i === day ? 'on' : '');
        b.onclick = function () { day = i; render(); }; strip.appendChild(b); })(i);
      body.appendChild(strip);
      var ctl = el('div', undefined, 'pve-ctl');
      ctl.innerHTML = '<button id="pve-prev">‹ Day</button><button id="pve-next">Day ›</button>' +
        '<input id="pve-filter" placeholder="Filter ticker or company" value="' + filter + '"><button id="pve-csv">Download day as CSV</button>';
      body.appendChild(ctl);
      var rs = rows(p).filter(function (r) { var q = filter.toLowerCase();
        return !q || r.ticker.toLowerCase().indexOf(q) >= 0 || r.company.toLowerCase().indexOf(q) >= 0; });
      rs.sort(function (a, b) { var u = a[sortK], v = b[sortK]; return (u < v ? -1 : u > v ? 1 : 0) * sortDir; });
      var cols = [['ticker', 'Ticker'], ['company', 'Company'], ['symbol', 'Symbol'], ['t', 'Target'], ['fill', 'At fill'],
                  ['h', 'Holding'], ['drift', 'Drift since fill'], [null, 'Path in period']];
      var t = '<table><tr>' + cols.map(function (c) { return '<th' + (c[0] ? ' data-k="' + c[0] + '"' : '') +
        (c[0] && 'tfillhdrift'.indexOf(c[0]) >= 0 && c[0].length ? ' style="text-align:right"' : '') + '>' + c[1] +
        (c[0] === sortK ? (sortDir < 0 ? ' ↓' : ' ↑') : '') + '</th>'; }).join('') + '</tr>';
      rs.forEach(function (r) { t += '<tr><td>' + r.ticker + (r.isNew ? ' <span class="pvc-chip in">new</span>' : '') + '</td><td class="muted">' +
        (r.company || '—') + '</td><td class="muted">' + r.symbol + '</td><td class="n">' + pct(r.t) + '</td><td class="n">' + pct(r.fill) +
        '</td><td class="n ' + (r.h < 0 ? 'neg' : 'pos') + '">' + pct(r.h) + '</td><td class="n ' + (r.drift < 0 ? 'neg' : 'pos') + '">' +
        (r.drift >= 0 ? '+' : '') + pct(r.drift) + '</td><td>' + spark(H[r.k], x.start, x.end, day) + '</td></tr>'; });
      t += '<tr><td colspan="5" class="muted">Cash</td><td class="n muted">' + pct(d.cash) + '</td><td></td><td></td></tr>';
      x.out.forEach(function (s) { var k = symIdx[s]; t += '<tr class="out"><td>' + tk(x.before[s][0]) + ' <span class="pvc-chip out">exited</span></td><td></td><td>' +
        s + '</td><td class="n">' + pct(x.before[s][1]) + ' → 0</td><td class="n">' + pct(H[k][x.start]) + '</td><td class="n">' + pct(H[k][day]) + '</td><td></td><td></td></tr>'; });
      body.appendChild(el('div', t + '</table>'));
      body.querySelectorAll('th[data-k]').forEach(function (th) { th.onclick = function () { var k = th.getAttribute('data-k');
        if (k === sortK) sortDir = -sortDir; else { sortK = k; sortDir = -1; } render(); }; });
      document.getElementById('pve-prev').onclick = function () { host.dvStep(-1); };
      document.getElementById('pve-next').onclick = function () { host.dvStep(1); };
      var fi = document.getElementById('pve-filter');
      fi.oninput = function () { filter = fi.value; render(); var f2 = document.getElementById('pve-filter'); f2.focus();
        f2.setSelectionRange(f2.value.length, f2.value.length); };
      document.getElementById('pve-csv').onclick = function () {
        var lines = ['ticker,company,symbol,target,at_fill,holding,drift'].concat(rs.map(function (r) {
          return [r.ticker, '"' + r.company + '"', r.symbol, r.t, r.fill, r.h, r.drift].join(','); }));
        var a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([lines.join('\n')], {type: 'text/csv'}));
        a.download = 'holdings_' + d.d + '.csv'; a.click(); };
    }
    host.dvStep = function (k) { day = Math.max(0, Math.min(days.length - 1, day + k)); render(); };
    render();
  }

  function show(v) {
    cur = v; params.set('variant', v); history.replaceState(null, '', '?' + params.toString());
    host.innerHTML = ''; host.dvStep = null;
    shipped.forEach(function (n) { n.style.display = v === 'A' ? '' : 'none'; });
    if (v === 'B') variantB(); if (v === 'C') variantC(); if (v === 'D') variantD(); if (v === 'E') variantE();
    var k = VARIANTS.filter(function (x) { return x[0] === v; })[0];
    document.getElementById('pb-lbl').innerHTML = '<span class="tag">PROTOTYPE</span>' + k[0] + ' (' + k[1] + ')';
  }
  function cycle(d) { var i = VARIANTS.map(function (x) { return x[0]; }).indexOf(cur);
    show(VARIANTS[(i + d + VARIANTS.length) % VARIANTS.length][0]); }
  document.getElementById('pb-prev').onclick = function () { cycle(-1); };
  document.getElementById('pb-next').onclick = function () { cycle(1); };
  document.addEventListener('keydown', function (e) {
    var t = e.target; if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if (!e.shiftKey) { if ((cur === 'D' || cur === 'E') && host.dvStep && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      host.dvStep(e.key === 'ArrowLeft' ? -1 : 1); e.stopImmediatePropagation(); } return; }
    if (e.key === 'ArrowLeft') cycle(-1); if (e.key === 'ArrowRight') cycle(1);
  }, true);
  var btn = document.querySelector('[data-tab="tab2"]'); if (btn) btn.click();
  show(VARIANTS.some(function (x) { return x[0] === cur; }) ? cur : 'A');
})();
</script>
"""


def main(source: str, target: str) -> None:
    page = Path(source).read_text(encoding="utf-8")
    if 'id="holdings-data"' not in page:
        raise SystemExit("this report has no Holdings tab")
    Path(target).write_text(page.replace("</body>", INJECT + "</body>"), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main(*sys.argv[1:3])
