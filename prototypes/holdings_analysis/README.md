# PROTOTYPE: Holdings-tab contribution analysis (throwaway)

Question: what should a "holdings analysis" in the report's Holdings tab look like
(how much return the top 10% of positions made against the rest)?

- `contrib.py` (run on the server against a run directory) wrote per-bar contributions
  (previous-close holding x valuation-price return) by weight decile and per symbol.
- `build_proto.py` injected three variants into the R223L5C5_231 report
  (experiment 2026-10-07-h1-daily-mvo), switchable with `?variant=A|B|C`:
  A largest positions vs the rest over time; B which names made the return (Pareto);
  C contribution by weight decile and year.

Verdict (user, 2026-10-09): A with more cut-off buttons, plus C. Folded into
`quantlab/runs/backtest_report.py` on main (`CONTRIBUTION_BANDS`, `holding_contributions`).
On R223L5C5_231: top 10% of names by weight made +92.6% of +185% summed; by name,
64 of 640 names made all of it.
