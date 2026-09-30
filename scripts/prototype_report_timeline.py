"""PROTOTYPE, throwaway: the report's time windows as a timeline, three variants x four cases.

Question: the "Dates and setup" table of report.html lists the backtest window,
the training window(s), the in-sample range(s) and the out-of-sample ranges as
text; with a walk-forward CV run (tens of folds) that text is unreadable. What
should a timeline of those windows look like so it reads for a single training
window, sliding CV and expanding CV alike?

It rewrites a real report page: the four window rows are removed from the
"Dates and setup" table and three variants are injected, switchable with
``?variant=A|B|C``; the windows come from four synthetic cases switchable with
``?case=single|overlap|sliding|expanding`` (fold geometry from
``BaseModel._cv_folds``, in-sample slivers from the label lookahead). The bar
at the bottom (and the arrow keys) cycles the variants.

Not library code: no tests, no error handling.

    uv run python scripts/prototype_report_timeline.py <report.html> <out.html>
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
page = src.read_text(encoding="utf-8")

# --------------------------------------------------------------------- cases

BARS = pd.bdate_range("2012-01-03", "2024-12-31")
LOOKAHEAD = 6  # label horizon 5 + the t+1 fill: bars after train_end the label still sees


def label(i):
    return str(BARS[min(i, len(BARS) - 1)].date())


def cv(train_periods, expanding):
    """Folds as BaseModel._cv_folds computes them, plus the stitched split."""
    test = train_periods // 5
    folds = []
    for i in range(max(1, (len(BARS) - train_periods) // test)):
        te0 = i * test + train_periods
        if te0 + test > len(BARS):
            continue
        tr0 = 0 if expanding else i * test
        folds.append({
            "fold": i,
            "train": [label(tr0), label(te0 - 1)],
            "effective": [label(tr0), label(te0 - 1 + LOOKAHEAD)],
            "lookahead": [label(te0 - 1), label(te0 - 1 + LOOKAHEAD)],
            "test": [label(te0), label(te0 + test - 1)],
            "in_sample": [label(te0), label(te0 + LOOKAHEAD - 1)],
        })
    return {
        "kind": "expanding" if expanding else "sliding",
        "summary": f"{len(folds)} folds, {'expanding' if expanding else 'sliding'} training window "
                   f"({'from ' + label(0) if expanding else f'{train_periods} bars'}), "
                   f"{test}-bar test segments",
        "backtest": [folds[0]["test"][0], folds[-1]["test"][1]],
        "folds": folds,
    }


def single(backtest_start):
    tr = [BARS.get_loc(pd.Timestamp("2012-01-03")), BARS.searchsorted(pd.Timestamp("2019-12-31"))]
    b0 = BARS.searchsorted(pd.Timestamp(backtest_start))
    # The last LOOKAHEAD bars are purged inside the training window, so the
    # effective window ends at train_end.
    eff_end = tr[1]
    ins = [label(b0), label(eff_end)] if b0 <= eff_end else None
    return {
        "kind": "single",
        "summary": "one training window" + (", backtest overlaps it" if ins else ""),
        "backtest": [label(b0), label(len(BARS) - 1)],
        "folds": [{
            "fold": None,
            "train": [label(tr[0]), label(tr[1])],
            "effective": [label(tr[0]), label(eff_end)],
            "lookahead": [label(tr[1] - LOOKAHEAD + 1), label(tr[1])],
            "test": [label(max(b0, eff_end + 1)), label(len(BARS) - 1)],
            "in_sample": ins,
        }],
    }


CASES = {
    "single": single("2020-01-02"),
    "overlap": single("2018-01-02"),
    "sliding": cv(504, expanding=False),
    "expanding": cv(504, expanding=True),
}

# --------------------------------------------------------------------- host page

# The four window rows go; the timeline replaces them.
for name in ("Backtest window", "Training window", "Training windows", "In-sample range",
             "In-sample ranges", "Out-of-sample ranges"):
    page = re.sub(rf"\s*<tr><th>{re.escape(name)}</th><td>.*?</td></tr>", "", page, flags=re.S)

STYLE = """
  .proto-bar { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); z-index: 99;
               background: #111; color: #fff; border-radius: 999px; padding: 6px 10px;
               display: flex; gap: 8px; align-items: center; font: 13px -apple-system, sans-serif;
               box-shadow: 0 4px 16px rgba(0,0,0,.35); }
  .proto-bar button { background: #333; color: #fff; border: 0; border-radius: 999px;
                      padding: 4px 10px; cursor: pointer; }
  .proto-bar button.on { background: #1f77b4; }
  .proto-bar .sep { width: 1px; height: 18px; background: #555; }
  .tl { font-size: 12px; color: #333; margin-bottom: 4px; }
  .tl .sum { color: #555; margin: 2px 0 6px; }
  .tl svg text { font: 10px -apple-system, sans-serif; fill: #555; }
  .tl .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: #555; margin-top: 4px; }
  .tl .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: -1px; margin-right: 4px; }
  .tl-c { max-height: 300px; overflow-y: auto; border: 1px solid #eee; }
  .tl-c table { border-collapse: collapse; font-size: 12px; width: 100%; }
  .tl-c th, .tl-c td { padding: 2px 8px 2px 0; border-bottom: 1px solid #f2f2f2; white-space: nowrap;
                       font-variant-numeric: tabular-nums; text-align: left; }
  .tl-c thead th { position: sticky; top: 0; background: #fff; color: #444; }
  .tl-c tr:hover td { background: #f6f9fc; }
  .tl details { font-size: 12px; margin-top: 6px; } .tl details li { font-variant-numeric: tabular-nums; }
"""

SCRIPT = r"""
const CASES = __CASES__;
const P = new URLSearchParams(location.search);
const VARIANTS = {A: "Gantt, one row per fold (left column)", B: "Full-width strip + training coverage", C: "Fold table with inline bars (left column)"};
let variant = P.get("variant") || "A", kase = P.get("case") || "sliding";
const C = {train: "#c6dbef", eff: "#6baed6", test: "#2b8a3e", testAlt: "#69b37a", ins: "#e03b30", bt: "#999"};
const t = s => Date.parse(s);
function scale(lo, hi, x0, x1) { return d => x0 + (t(d) - lo) / (hi - lo) * (x1 - x0); }
function bounds(c) {
  const all = c.folds.flatMap(f => [f.train[0], f.test[1]]).concat(c.backtest);
  return [Math.min(...all.map(t)), Math.max(...all.map(t))];
}
function yearTicks(lo, hi, x, y, h) {
  let s = "";
  for (let yr = new Date(lo).getUTCFullYear() + 1; yr <= new Date(hi).getUTCFullYear(); yr++) {
    const d = `${yr}-01-01`, px = x(d);
    s += `<line x1="${px}" x2="${px}" y1="0" y2="${h}" stroke="#eee"/>` +
         `<text x="${px}" y="${y}" text-anchor="middle">${yr % 2 ? "" : yr}</text>`;
  }
  return s;
}
const rect = (x, a, b, y, h, fill, tip, extra = "") =>
  `<rect x="${x(a)}" y="${y}" width="${Math.max(1, x(b) - x(a))}" height="${h}" fill="${fill}" ${extra}><title>${tip}</title></rect>`;
const legend = items => `<div class="legend">${items.map(([c, l]) => `<span><span class="sw" style="background:${c}"></span>${l}</span>`).join("")}</div>`;
const LEG = [[C.train, "training"], [C.eff, "label lookahead"], [C.test, "test (out-of-sample)"], [C.ins, "in-sample bars traded"]];

// A: Gantt. One row per fold, the backtest window as a row on top.
function variantA(c) {
  const W = 430, L = c.kind === "single" ? 60 : 44, n = c.folds.length;
  const rh = c.kind === "single" ? 16 : Math.max(4, Math.min(12, 260 / n)), gap = rh > 6 ? 2 : 1;
  const [lo, hi] = bounds(c), x = scale(lo, hi, L, W - 4);
  const top = 20, H = top + (n + 1) * (rh + gap) + 18;
  let s = yearTicks(lo, hi, x, H - 4, H - 14);
  s += `<text x="0" y="${top - 6 + rh}">backtest</text>`;
  s += rect(x, c.backtest[0], c.backtest[1], top - 6, rh, C.bt, `backtest ${c.backtest[0]} .. ${c.backtest[1]}`, 'opacity=".5"');
  c.folds.forEach((f, i) => {
    const y = top + (i + 1) * (rh + gap) - 6;
    const lab = f.fold === null ? "model" : (rh >= 9 || i % 5 === 0 ? `fold ${f.fold}` : "");
    if (lab) s += `<text x="0" y="${y + rh - 1}">${lab}</text>`;
    s += rect(x, f.train[0], f.train[1], y, rh, C.train, `fold ${f.fold ?? ""} training ${f.train[0]} .. ${f.train[1]}`);
    s += rect(x, f.lookahead[0], f.lookahead[1], y, rh, C.eff, `label lookahead ${f.lookahead[0]} .. ${f.lookahead[1]}`);
    s += rect(x, f.test[0], f.test[1], y, rh, C.test, `test ${f.test[0]} .. ${f.test[1]}`);
    if (f.in_sample) {
      s += rect(x, f.in_sample[0], f.in_sample[1], y, rh, C.ins, `in-sample ${f.in_sample[0]} .. ${f.in_sample[1]}`);
      s += rect(x, f.in_sample[0], f.in_sample[1], top - 6, rh, C.ins, `in-sample ${f.in_sample[0]} .. ${f.in_sample[1]}`);
    }
  });
  return `<h2>Windows</h2><div class="tl"><div class="sum">${c.summary}; backtest ${c.backtest[0]} .. ${c.backtest[1]}</div>
    <svg width="${W}" height="${H}">${s}</svg>${legend(LEG)}</div>`;
}

// B: one full-width strip of the traded window (test segments alternate shade,
// in-sample slivers red) above a training-coverage strip (how many folds
// trained on each date), then the full list folded away.
function variantB(c) {
  const W = document.querySelector(".kpis").getBoundingClientRect().width || 1300, L = 80;
  const [lo, hi] = bounds(c), x = scale(lo, hi, L, W - 8);
  let s = yearTicks(lo, hi, x, 78, 64);
  s += `<text x="0" y="17">traded</text><text x="0" y="47">trained on</text>`;
  s += rect(x, c.backtest[0], c.backtest[1], 4, 18, "#eee", `backtest ${c.backtest[0]} .. ${c.backtest[1]}`);
  c.folds.forEach((f, i) => {
    s += rect(x, f.test[0], f.test[1], 4, 18, i % 2 ? C.testAlt : C.test,
              `${f.fold === null ? "test" : "fold " + f.fold} ${f.test[0]} .. ${f.test[1]}`);
    const w = x(f.test[1]) - x(f.test[0]);
    if (f.fold !== null && w > 18) s += `<text x="${(x(f.test[0]) + x(f.test[1])) / 2}" y="16" text-anchor="middle" style="fill:#fff">${f.fold}</text>`;
    if (f.in_sample) s += rect(x, f.in_sample[0], f.in_sample[1], 2, 22, C.ins, `in-sample ${f.in_sample[0]} .. ${f.in_sample[1]}`);
  });
  // coverage: count the folds whose effective window holds each month
  const months = [];
  for (let d = new Date(lo); d <= hi; d.setUTCMonth(d.getUTCMonth() + 1)) months.push(new Date(d));
  const counts = months.map(m => c.folds.filter(f => t(f.effective[0]) <= +m && +m <= t(f.effective[1])).length);
  const max = Math.max(1, ...counts);
  months.forEach((m, i) => {
    if (!counts[i]) return;
    const a = m.toISOString().slice(0, 10), b = (months[i + 1] || new Date(hi)).toISOString().slice(0, 10);
    s += rect(x, a, b, 34, 18, C.eff, `${a.slice(0, 7)}: in ${counts[i]} training window(s)`, `opacity="${0.15 + 0.85 * counts[i] / max}"`);
  });
  const list = c.folds.map(f => `<li>${f.fold === null ? "" : "fold " + f.fold + ": "}train ${f.train[0]} .. ${f.train[1]}, test ${f.test[0]} .. ${f.test[1]}${f.in_sample ? `, in-sample ${f.in_sample[0]} .. ${f.in_sample[1]}` : ""}</li>`).join("");
  return `<h2>Windows</h2><div class="tl"><div class="sum">${c.summary}; backtest ${c.backtest[0]} .. ${c.backtest[1]}; darker = more folds trained on that month</div>
    <svg width="${W}" height="84">${s}</svg>${legend([[C.test, "traded test segment"], [C.ins, "in-sample bars traded"], [C.eff, "trained on (shade = folds)"]])}
    <details><summary>All ${c.folds.length} window(s) as text</summary><ul>${list}</ul></details></div>`;
}

// C: the numbers stay the primary thing; each fold row carries a mini bar on
// the shared axis, the table scrolls past ~12 rows.
function variantC(c) {
  const [lo, hi] = bounds(c), BW = 130, x = scale(lo, hi, 1, BW - 1);
  const bar = f => `<svg width="${BW}" height="10"><rect x="0" y="4" width="${BW}" height="2" fill="#eee"/>` +
    rect(x, f.train[0], f.effective[1], 1, 8, C.eff, `training ${f.train[0]} .. ${f.effective[1]}`, 'opacity=".55"') +
    rect(x, f.test[0], f.test[1], 1, 8, C.test, `test ${f.test[0]} .. ${f.test[1]}`) +
    (f.in_sample ? rect(x, f.in_sample[0], f.in_sample[1], 0, 10, C.ins, "in-sample") : "") + "</svg>";
  const short = d => d.slice(2).replaceAll("-", "");
  const rows = c.folds.map(f => `<tr><td>${f.fold ?? "—"}</td><td>${f.train[0]} → ${f.train[1]}</td><td>${f.test[0]} → ${f.test[1]}</td><td>${bar(f)}</td></tr>`).join("");
  return `<h2>Windows</h2><div class="tl"><div class="sum">${c.summary}<br>backtest ${c.backtest[0]} .. ${c.backtest[1]}${c.folds.some(f => f.in_sample) ? `, in-sample bars: ${c.folds.filter(f => f.in_sample).length} range(s), red` : ""}</div>
    <div class="tl-c"><table><thead><tr><th>fold</th><th>train</th><th>test</th><th>${new Date(lo).getUTCFullYear()} – ${new Date(hi).getUTCFullYear()}</th></tr></thead><tbody>${rows}</tbody></table></div>${legend([[C.eff, "training (+ lookahead)"], [C.test, "test"], [C.ins, "in-sample"]])}</div>`;
}

function render() {
  const c = CASES[kase];
  document.getElementById("tl-left").innerHTML = variant === "B" ? "" : (variant === "A" ? variantA(c) : variantC(c));
  document.getElementById("tl-wide").innerHTML = variant === "B" ? variantB(c) : "";
  const bar = document.getElementById("proto-bar");
  bar.innerHTML = `<button data-v="-1">◀</button><b>${variant}</b> ${VARIANTS[variant]}<button data-v="1">▶</button><span class="sep"></span>` +
    Object.keys(CASES).map(k => `<button data-c="${k}" class="${k === kase ? "on" : ""}">${k}</button>`).join("");
  history.replaceState(null, "", `?variant=${variant}&case=${kase}`);
}
function step(d) { const k = Object.keys(VARIANTS); variant = k[(k.indexOf(variant) + d + k.length) % k.length]; render(); }
document.addEventListener("click", e => {
  const b = e.target.closest("#proto-bar button"); if (!b) return;
  if (b.dataset.v) step(+b.dataset.v); else { kase = b.dataset.c; render(); }
});
document.addEventListener("keydown", e => {
  if (e.target.closest("input,textarea,[contenteditable]")) return;
  if (e.key === "ArrowLeft") step(-1); if (e.key === "ArrowRight") step(1);
});
render();
"""

page = page.replace("</style>", STYLE + "</style>", 1)
page = page.replace('<div class="tables">', '<div class="tables"><div id="tl-left"></div>', 1)
kpis_end = page.index("</div>\n", page.index('<div class="kpis">')) + len("</div>\n")
page = page[:kpis_end] + '  <div id="tl-wide"></div>\n' + page[kpis_end:]
page = page.replace("</body>", '<div class="proto-bar" id="proto-bar"></div>\n<script>'
                    + SCRIPT.replace("__CASES__", json.dumps(CASES)) + "</script>\n</body>", 1)
out.write_text(page, encoding="utf-8")
print(out)
