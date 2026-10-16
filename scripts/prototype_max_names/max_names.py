"""PROTOTYPE (#326): a name-count cap on the paper-trading optimiser, exact MIQP against sparse projected gradient.

    QUANTLAB_DATA_DIR=/data/quantlab numactl --physcpubind=64-79 --membind=1 \
        python max_names.py OUT_DIR [--bars 60] [--names 20 50] [--threads 16] [--time-limit 60]
        [--loose-time-limit 30] [--top-k 200]

Questions (spec #325): how long does MOSEK take per bar to solve the exact
name-count program on us3000, and how close does a sparse projected gradient
(the approximate backtest's planned method) come to it?

Setup: the paper-trading rule of us3000_tdM_mvo (ActiveRiskOptimizer against
the cap-weighted us3000, USE4 factor risk, lambda 20, ic 0.0122, turnover
penalty 0.002, cap 5%, style_beta / style_size in +-0.1, long-only), with
``candidate_top_k=200`` as the spec asks (the live rule takes every member),
on the recorded predictions of the approximate-backtest acceptance's exact
MVO run (xgb_td_s0 + master_s0, membership-masked). The last BARS decision
bars of that run. Each bar's context is built by ``DecisionInputs.context``
with the holdings the previous bar's exact (bounded) solution left, no drift
and no execution (the first bar starts from cash); a bar the exact solve
cannot decide keeps the holdings.

Per bar and N:

- ``convex``: the rule's own uncapped solve (CLARABEL), its name count
  (weights > 1e-6, and >= min_trade 1e-3) as reference;
- ``convex_nb``: the same without the style bounds;
- ``exact``: the rule's program plus one boolean per candidate,
  ``w <= cap * z``, ``sum(z) + locked <= N``, MOSEK (default relative gap
  1e-4, TIME_LIMIT), its wall and solver seconds and status; skipped when
  the bounded convex program is already infeasible (with
  ``candidate_top_k=200`` the 200 best-predicted lean small, and size
  within +-0.1 cannot hold from cash);
- ``exact_nb``: the same without the style bounds: the reference for the
  approximate method, since the approximate backtest refuses exposure
  bounds (#303) and so never solves the bounded program;
- ``exact_nb_gap1``: ``exact_nb`` stopped at a 1% relative gap
  (LOOSE_TIME_LIMIT); the better of the two is ``best``, the next bar's
  holdings (the chain runs on the unbounded program, since the bounded one
  is infeasible at top 200);
- ``approx``: projected gradient (no momentum), step 1 / (lambda L) with L
  the largest eigenvalue of the pool's covariance, warm-started from the
  holdings, the prox the L1 pull toward the holdings within the capped
  simplex onto the N entries whose zeroing costs most; L the curvature on
  the budget's tangent space (the library kernels' bound); compared with
  ``exact_nb`` and ``best``: objective gap relative to |objective|, the gap
  over the exact book's expected return, and the share of the exact names
  it holds.

The objective is the rule's: mu.w - lambda/2 (x - b)' Sigma (x - b) - kappa
|w - w_current|_1, x the candidates then the locked positions; evaluated in
numpy for every solution after the cleanup (excluded names exact zero, the
rest projected onto the capped simplex).

Writes OUT_DIR/bars_N<N>.csv (one row per bar) and OUT_DIR/summary.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

ACCEPTANCE = Path(os.environ.get(
    "ACCEPTANCE_SCRIPTS",
    Path.home() / "projects" / "quantlab2" / "experiments" / "2026-10-15-approximate-backtest-acceptance" / "scripts"))
sys.path.insert(0, str(ACCEPTANCE))

import setup  # noqa: E402  (also puts the us3000 IC experiment's scripts on the path)
import mvo_backtest  # noqa: E402
from quantlab.portfolio.decision_inputs import DecisionInputs  # noqa: E402
from quantlab.portfolio.base import PortfolioConstructionError  # noqa: E402
from quantlab.portfolio.predefined.mean_variance import _project_capped_simplex  # noqa: E402
from quantlab.runs.backtest_run import BacktestRun  # noqa: E402

#: The paper-trading rule (results ten), with the spec's candidate_top_k=200 and the style bounds.
ACTIVE = {"te": None, "top_k": 200, "bounds": True, "index": "us3000", "raw": False,
          "turnover_penalty": None, "risk_aversion": 20.0}
IC = 0.0122
SMALL = 1e-6


# -- the bar's program ----------------------------------------------------------------------------


class Bar:
    """One bar's problem in numpy: the pool's covariance, the linear term and the objective."""

    def __init__(self, rule, inputs):
        config = rule.config
        self.inputs = inputs
        self.lam, self.kappa, self.cap = config.risk_aversion, config.turnover_penalty, config.weight_cap
        self.n = n = len(inputs.symbols)
        self.total = 1.0 - float(inputs.locked_weights.sum())
        self.current = np.asarray(inputs.current_weights, dtype=np.float64)
        exposures, factor_cov, specific = inputs.estimate.factor_form()
        rows = len(specific)
        # x = P w + r: the candidates, then the covered locked weights, then the reference's others at 0, less b.
        offset = np.zeros(rows)
        locked = inputs.risk_locked_weights
        offset[n:n + len(locked)] = locked
        if len(inputs.reference_weights):
            offset -= inputs.reference_weights
        self.offset = offset
        values, vectors = np.linalg.eigh(factor_cov)
        self.root = exposures @ (vectors * np.sqrt(np.clip(values, 0.0, None)))  # [rows, k], U U' = B F B'
        self.specific = np.clip(specific, 0.0, None)
        u = self.root[:n]
        self.sigma = u @ u.T + np.diag(self.specific[:n])
        sigma_r = self.root @ (self.root.T @ offset) + self.specific * offset
        self.q = np.asarray(inputs.expected_return, dtype=np.float64) - self.lam * sigma_r[:n]
        self.constant = -self.lam / 2 * float(offset @ sigma_r)
        # The step's curvature on the budget's tangent space {sum(dw) = 0}, as the library's kernels bound it.
        centred = self.sigma - self.sigma.mean(axis=0) - self.sigma.mean(axis=1)[:, None] + self.sigma.mean()
        self.lipschitz = float(np.linalg.eigvalsh(centred)[-1]) if n else 0.0

    def objective(self, w: np.ndarray) -> float:
        """mu.w - lambda/2 (x - b)' Sigma (x - b) - kappa |w - w_current|_1."""
        return float(self.q @ w - self.lam / 2 * w @ self.sigma @ w + self.constant
                     - self.kappa * np.abs(w - self.current).sum())

    def locked_names(self) -> int:
        return int((self.inputs.locked_weights != 0).sum())


def clean(values: np.ndarray, keep: np.ndarray, cap: float, total: float) -> np.ndarray:
    """Excluded names exact zero, the kept ones projected onto the capped simplex."""
    w = np.zeros(len(values))
    if keep.any():
        w[keep] = _project_capped_simplex(values[keep], cap, total)
    return w


def exact(rule, bar: Bar, names: int | None, bounds: bool, solver: str, options: dict) -> dict:
    """The rule's program with a name-count cap (``names``) or without (convex), on ``solver``."""
    inputs, n = bar.inputs, bar.n
    if bar.total <= 1e-12:
        return {"status": "locked_full", "weights": np.zeros(n)}
    slots = None if names is None else names - bar.locked_names()
    if slots is not None and slots * bar.cap < bar.total - 1e-12:
        return {"status": "infeasible_slots", "weights": None}
    w = cp.Variable(n)
    x = (cp.hstack([w, np.zeros(len(bar.offset) - n)]) if len(bar.offset) > n else w) + bar.offset
    factor = cp.Variable(bar.root.shape[1])
    risk = cp.sum_squares(factor) + cp.sum(cp.multiply(bar.specific, cp.square(x)))
    objective = inputs.expected_return @ w - bar.lam / 2 * risk
    if bar.kappa:
        objective = objective - bar.kappa * cp.norm1(w - bar.current)
    constraints = [factor == bar.root.T @ x, cp.sum(w) == bar.total, w >= 0]
    z = None
    if slots is None:
        constraints.append(w <= bar.cap)
    else:
        z = cp.Variable(n, boolean=True)
        constraints += [w <= bar.cap * z, cp.sum(z) <= slots]
    if bounds:
        for (lower, upper), row, locked in zip(inputs.exposure_bounds, inputs.exposures, inputs.locked_exposures):
            constraints += [row @ w + locked >= lower, row @ w + locked <= upper]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    started = time.perf_counter()
    try:
        problem.solve(solver=solver, **options)
    except cp.error.SolverError as exc:
        return {"status": f"solver_error: {exc}"[:200], "weights": None,
                "seconds": time.perf_counter() - started}
    seconds = time.perf_counter() - started
    stats = problem.solver_stats
    out = {"status": problem.status, "seconds": seconds,
           "solve_seconds": None if stats is None else stats.solve_time,
           "nodes": None if stats is None or stats.extra_stats is None else _nodes(stats)}
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
        out["weights"] = None
        return out
    values = np.asarray(w.value, dtype=np.float64)
    keep = np.ones(n, dtype=bool) if z is None else np.asarray(z.value) > 0.5
    out["weights"] = clean(values, keep, bar.cap, bar.total)
    return out


def _nodes(stats):
    try:
        import mosek
        return int(stats.extra_stats.getintinf(mosek.iinfitem.mio_num_branch))
    except Exception:  # noqa: BLE001
        return None


def approximate(bar: Bar, names: int, max_iterations: int = 5000, tolerance: float = 1e-10) -> dict:
    """Projected gradient on the sparse set: no momentum, step 1 / (lambda L), warm-started from the holdings."""
    n, cap, total = bar.n, bar.cap, bar.total
    slots = names - bar.locked_names()
    if total <= 1e-12:
        return {"weights": np.zeros(n), "iterations": 0}
    if slots * cap < total - 1e-12:
        return {"weights": None, "iterations": 0}
    step = 1.0 / (bar.lam * bar.lipschitz)
    shrink = step * bar.kappa
    c = bar.current
    w = c.copy()
    started = time.perf_counter()
    for iteration in range(1, max_iterations + 1):
        v = w + step * (bar.q - bar.lam * (bar.sigma @ w))
        nxt = _sparse_prox(v, c, shrink, cap, total, slots)
        moved = np.abs(nxt - w).max()
        w = nxt
        if moved < tolerance:
            break
    return {"weights": w, "iterations": iteration, "seconds": time.perf_counter() - started}


def _pulled(u: np.ndarray, c: np.ndarray, shrink: float, cap: float) -> np.ndarray:
    """argmin_w 1/2 (w - u)^2 + shrink |w - c| over [0, cap], per entry."""
    d = u - c
    return np.clip(c + np.sign(d) * np.maximum(np.abs(d) - shrink, 0.0), 0.0, cap)


def _shift(v, c, shrink, cap, total, keep) -> float:
    """The tau with sum(_pulled(v - tau)) = total over ``keep``, by bisection."""
    low, high = v[keep].min() - cap - shrink - 1.0, v[keep].max() + shrink + 1.0
    for _ in range(200):
        tau = (low + high) / 2
        if _pulled(v[keep] - tau, c[keep], shrink, cap).sum() > total:
            low = tau
        else:
            high = tau
        if high - low < 1e-15:
            break
    return high


def _sparse_prox(v, c, shrink, cap, total, slots) -> np.ndarray:
    """The prox onto the capped simplex with the L1 pull, kept on the ``slots`` entries whose zeroing costs most."""
    every = np.ones(len(v), dtype=bool)
    tau = _shift(v, c, shrink, cap, total, every)
    u = v - tau
    full = _pulled(u, c, shrink, cap)
    cost = (0.5 * u ** 2 + shrink * np.abs(c)) - (0.5 * (full - u) ** 2 + shrink * np.abs(full - c))
    keep = np.zeros(len(v), dtype=bool)
    keep[np.argsort(-cost, kind="stable")[:slots]] = True
    tau = _shift(v, c, shrink, cap, total, keep)
    w = np.zeros(len(v))
    w[keep] = _pulled(v[keep] - tau, c[keep], shrink, cap)
    return w


# -- the run ----------------------------------------------------------------------------------------


def names_of(w: np.ndarray | None, threshold: float = SMALL) -> int | None:
    return None if w is None else int((w > threshold).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--bars", type=int, default=60)
    parser.add_argument("--names", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--loose-time-limit", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=200, help="candidate_top_k; 0 for every member (the live rule)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mosek = {"mosek_params": {"MSK_IPAR_NUM_THREADS": args.threads,
                              "MSK_DPAR_OPTIMIZER_MAX_TIME": args.time_limit}}
    # MOSEK's default relative gap is 1e-4; this one stops at 1%.
    loose_gap = {"mosek_params": {**mosek["mosek_params"], "MSK_DPAR_MIO_TOL_REL_GAP": 1e-2,
                                  "MSK_DPAR_OPTIMIZER_MAX_TIME": args.loose_time_limit}}

    run_dir = json.loads((setup.RESULTS / "exact_mvo.json").read_text())["run_dir"]
    panel = BacktestRun.open(run_dir).predictions()
    rule = mvo_backtest.optimiser(setup.recipe.risk_model(), IC, {**ACTIVE, "top_k": args.top_k or None})
    rule.bind(panel.labels)
    recorded = DecisionInputs.from_run(run_dir)
    inputs = DecisionInputs(
        recorded.dataset, rule, fill_column=recorded.fill_column, valuation_column=recorded.valuation_column,
        rebalance_periods=1, anchor=recorded.anchor,
    )
    predictions = panel.predictions
    # The last decision bar of a run is its second to last (the last has no next bar to fill on).
    bars = pd.DatetimeIndex(predictions.timestamp.values)[-args.bars - 1:-1]
    logger.info(f"{len(bars)} bars {bars[0].date()}..{bars[-1].date()}, rule {rule.config.exposure_bounds}, "
                f"top_k {rule.config.candidate_top_k}")
    summary = {"run_dir": run_dir, "bars": [str(bars[0].date()), str(bars[-1].date())],
               "rule": {"risk_aversion": rule.config.risk_aversion, "ic": rule.config.ic,
                        "turnover_penalty": rule.config.turnover_penalty, "weight_cap": rule.config.weight_cap,
                        "candidate_top_k": rule.config.candidate_top_k,
                        "exposure_bounds": dict(rule.config.exposure_bounds)},
               "mosek": mosek["mosek_params"], "results": {}}
    with inputs.preloaded(bars[0], bars[-1]):
        for names in args.names:
            held = xr.DataArray(np.zeros(0), dims="symbol", coords={"symbol": predictions.symbol.values[:0]})
            rows = []
            for t in bars:
                context = inputs.context(t, predictions.sel(timestamp=t), held)
                try:
                    problem_inputs = rule.problem_inputs(context)
                except PortfolioConstructionError as exc:
                    rows.append({"bar": t, "error": str(exc)[:200]})
                    continue
                bar = Bar(rule, problem_inputs)
                convex = exact(rule, bar, None, True, "CLARABEL", {})
                convex_nb = exact(rule, bar, None, False, "CLARABEL", {})
                capped = (exact(rule, bar, names, True, "MOSEK", mosek) if convex["weights"] is not None
                          else {"status": "skipped: convex infeasible", "weights": None})
                unbounded = exact(rule, bar, names, False, "MOSEK", mosek)
                loose = exact(rule, bar, names, False, "MOSEK", loose_gap)
                approx = approximate(bar, names)
                row = {
                    "bar": t, "candidates": bar.n, "locked": bar.locked_names(),
                    "held_before": int((held.values != 0).sum()),
                    "convex_status": convex["status"], "convex_names": names_of(convex["weights"]),
                    "convex_names_min_trade": names_of(convex["weights"], 1e-3),
                    "convex_objective": None if convex["weights"] is None else bar.objective(convex["weights"]),
                    "convex_nb_status": convex_nb["status"], "convex_nb_names": names_of(convex_nb["weights"]),
                    "convex_nb_names_min_trade": names_of(convex_nb["weights"], 1e-3),
                    "convex_nb_objective": None if convex_nb["weights"] is None
                    else bar.objective(convex_nb["weights"]),
                }
                for key, out in (("exact", capped), ("exact_nb", unbounded), ("exact_nb_gap1", loose)):
                    row.update({
                        f"{key}_status": out["status"], f"{key}_seconds": out.get("seconds"),
                        f"{key}_solve_seconds": out.get("solve_seconds"), f"{key}_nodes": out.get("nodes"),
                        f"{key}_names": names_of(out["weights"]),
                        f"{key}_objective": None if out["weights"] is None else bar.objective(out["weights"]),
                    })
                row.update({"approx_iterations": approx["iterations"], "approx_seconds": approx.get("seconds"),
                            "approx_names": names_of(approx["weights"])})
                if approx["weights"] is not None:
                    row["approx_objective"] = bar.objective(approx["weights"])
                if approx["weights"] is not None and unbounded["weights"] is not None:
                    reference = unbounded["weights"] > SMALL
                    row["overlap"] = float((reference & (approx["weights"] > SMALL)).sum() / max(reference.sum(), 1))
                    f_exact, f_approx = row["exact_nb_objective"], row["approx_objective"]
                    row["gap"] = (f_exact - f_approx) / abs(f_exact)
                    row["gap_over_alpha"] = (f_exact - f_approx) / max(
                        float(problem_inputs.expected_return @ unbounded["weights"]), 1e-12)
                rows.append(row)
                logger.info({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
                # The best of the two exact solves as the reference and the next bar's holdings.
                best = max((out for out in (unbounded, loose) if out["weights"] is not None),
                           key=lambda out: bar.objective(out["weights"]), default=None)
                if best is not None and approx["weights"] is not None:
                    reference = best["weights"] > SMALL
                    row["overlap_best"] = float(
                        (reference & (approx["weights"] > SMALL)).sum() / max(reference.sum(), 1))
                    row["gap_best"] = (bar.objective(best["weights"]) - row["approx_objective"]) / abs(
                        bar.objective(best["weights"]))
                if best is not None:
                    weights = np.concatenate([best["weights"], problem_inputs.locked_weights])
                    symbols = np.concatenate([problem_inputs.symbols, problem_inputs.locked_symbols])
                    keep = weights != 0
                    held = xr.DataArray(weights[keep], dims="symbol", coords={"symbol": symbols[keep]})
            frame = pd.DataFrame(rows)
            frame.to_csv(args.out / f"bars_N{names}.csv", index=False)
            summary["results"][str(names)] = summarise(frame)
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
            logger.info(f"N={names}: {summary['results'][str(names)]}")


def summarise(frame: pd.DataFrame) -> dict:
    out = {"bars": int(len(frame))}
    for key in ("exact", "exact_nb", "exact_nb_gap1"):
        seconds = frame[f"{key}_seconds"].dropna()
        if seconds.empty:
            out[key] = {"statuses": frame[f"{key}_status"].value_counts().to_dict()}
            continue
        out[key] = {
            "seconds_median": float(seconds.median()), "seconds_max": float(seconds.max()),
            "solve_seconds_median": float(frame[f"{key}_solve_seconds"].dropna().median()),
            "statuses": frame[f"{key}_status"].value_counts().to_dict(),
            "names_median": float(frame[f"{key}_names"].dropna().median()),
        }
    out["convex_nb"] = {"statuses": frame["convex_nb_status"].value_counts().to_dict(),
                        "names_median": float(frame["convex_nb_names"].dropna().median()),
                        "names_min_trade_median": float(frame["convex_nb_names_min_trade"].dropna().median())}
    out["convex"] = {"statuses": frame["convex_status"].value_counts().to_dict(),
                     "names_median": float(frame["convex_names"].dropna().median()),
                     "names_min_trade_median": float(frame["convex_names_min_trade"].dropna().median())}
    if "overlap" in frame:
        out["approx"] = {
            "overlap_mean": float(frame["overlap"].mean()), "overlap_min": float(frame["overlap"].min()),
            "gap_median": float(frame["gap"].median()), "gap_max": float(frame["gap"].max()),
            "gap_over_alpha_median": float(frame["gap_over_alpha"].median()),
            "overlap_best_mean": float(frame["overlap_best"].mean()),
            "gap_best_median": float(frame["gap_best"].median()), "gap_best_max": float(frame["gap_best"].max()),
            "iterations_median": float(frame["approx_iterations"].median()),
            "iterations_max": int(frame["approx_iterations"].max()),
            "seconds_median": float(frame["approx_seconds"].median()),
        }
    return out


if __name__ == "__main__":
    main()
