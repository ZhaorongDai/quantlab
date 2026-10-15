"""PROTOTYPE, throwaway: the vectorized backtest against ``run()`` on one synthetic store.

usage: PYTHONPATH=. python scripts/prototype_vectorized_backtest/compare.py {topn,mvo} S T R [penalty]

Prints, after every stage, the timings and how far the vectorized result is
from the exact one: (1) the simulator alone, fed the exact run's weights;
(2) the vectorized constructor's weights against the exact ones; (3) end to end.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from tests.backtest_fixtures import make_model, make_stock_dataset, write_price_store

sys.path.insert(0, str(Path(__file__).parent))
import vectorized as vec  # noqa: E402

kind, S, T, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
PENALTY = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
FEES, LOOKBACK, IC, LAMBDA, CAP = 0.0005, 40, 0.05, 5.0, 0.05
TOP = max(5, S // 10)

day = lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d")
tmp = Path(tempfile.mkdtemp())
dc = write_price_store(tmp, symbols=[f"S{i:04d}" for i in range(S)], n_bars=T, seed=1)
store = xr.open_zarr(dc.zarr_file_path).load()
bars = store.timestamp.values
model = make_model(tmp / "model", dc, start_date=day(bars[0]), end_date=day(bars[59]),
                   train_start=day(bars[0]), train_end=day(bars[49]),
                   test_start=day(bars[50]), test_end=day(bars[59]))
if kind == "topn":
    cons = TopNConstructor(TopNConfig(direction="long_only", top_n=TOP))
else:
    cons = MeanVarianceOptimizer(MeanVarianceConfig(
        expected_return_label="fwd_ret_1",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
        ic=IC, risk_aversion=LAMBDA, turnover_penalty=PENALTY, weight_cap=CAP))
bt = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
    price_dataset=make_stock_dataset(dc), model=model, model_mode="train",
    start_date=day(bars[60]), end_date=day(bars[T - 3]), output_dir=str(tmp / "out"),
    rebalance_periods=R, constructor=cons, fees=FEES, slippage=0.0))

t0 = time.perf_counter()
bt.run()
t_exact = time.perf_counter() - t0
run_dir = next((tmp / "out").iterdir())
pred = xr.open_zarr(run_dir / "predictions.zarr").load()
wx = xr.open_zarr(run_dir / "weights.zarr").load()
eq = xr.open_zarr(run_dir / "equity.zarr").load()

rows = pd.Index(bars).get_indexer(eq.timestamp.values)  # the window's rows in the store
open_full = store["adjOpen"].transpose("timestamp", "symbol").values
close_full = store["adjClose"].transpose("timestamp", "symbol").values
open_, close = open_full[rows], close_full[rows]
exact_eq = eq["value"].values / eq["value"].values[0]
scores_full = pred["fwd_ret_1"].reindex(timestamp=bars, symbol=store.symbol.values).transpose("timestamp", "symbol").values


def stats(equity, turnover, n_fills):
    r = np.diff(equity) / equity[:-1]
    return (f"total {equity[-1] - 1:+.4%}  vol {r.std() * np.sqrt(252):.2%}  "
            f"sharpe {r.mean() / r.std() * np.sqrt(252):.3f}  turnover/fill {turnover.sum() / n_fills:.3f}")


def compare(name, equity):
    gap = np.abs(equity / exact_eq - 1)
    print(f"[{name}] max |equity/exact - 1| {gap.max():.2e}  final {equity[-1] / exact_eq[-1] - 1:+.2e}")


print(f"== {kind} S={S} window={len(rows)} R={R} penalty={PENALTY}")
print(f"[exact] run() {t_exact:.2f}s")
decided = np.flatnonzero(wx["decided"].values)
W_exact = np.nan_to_num(wx["weight"].values[decided])
exact_turn = np.abs(np.diff(np.vstack([np.zeros(S), W_exact]), axis=0)).sum(axis=1)
print(f"[exact] {stats(exact_eq, exact_turn, len(decided))}  (turnover vs previous target, approx)")

# (1) the simulator alone, on the exact weights
t0 = time.perf_counter()
sim_eq, sim_turn = vec.simulate(W_exact, decided, open_, close, FEES)
print(f"[sim on exact weights] {time.perf_counter() - t0:.3f}s  {stats(sim_eq, sim_turn, len(decided))}")
compare("sim on exact weights", sim_eq)

# (2) the constructor
decisions = np.arange(0, len(rows) - 1, R)
print(f"[schedule] vectorized decisions {len(decisions)}, exact decided {len(decided)}, same rows {np.array_equal(decisions, decided)}")
t0 = time.perf_counter()
if kind == "topn":
    W = vec.top_n(scores_full[rows][decisions], np.isfinite(close[decisions]), TOP)
else:
    W = vec.mean_variance(scores_full[rows[decisions]], close_full, rows[decisions],
                          lookback=LOOKBACK, ic=IC, risk_aversion=LAMBDA, weight_cap=CAP)
t_cons = time.perf_counter() - t0
same = np.array_equal(decisions, decided)
if same:
    l1 = np.abs(W - W_exact).sum(axis=1)
    print(f"[weights] {t_cons:.3f}s  L1 to exact: mean {l1.mean():.4f} max {l1.max():.4f} (2 = disjoint)")

# (3) end to end
t0 = time.perf_counter()
v_eq, v_turn = vec.simulate(W, decisions, open_, close, FEES)
t_sim = time.perf_counter() - t0
print(f"[vectorized] {t_cons + t_sim:.3f}s  {stats(v_eq, v_turn, len(decisions))}")
compare("vectorized", v_eq)
if kind == "mvo":
    for k in (0.5, 0.2):
        Ws = vec.smooth(W, k)
        s_eq, s_turn = vec.simulate(Ws, decisions, open_, close, FEES)
        print(f"[vectorized + smooth k={k}] {stats(s_eq, s_turn, len(decisions))}")
        compare(f"vectorized + smooth k={k}", s_eq)

if kind == "mvo" and PENALTY > 0:
    for sweeps, warm in ((20, 10),):
        t0 = time.perf_counter()
        Wt = vec.mean_variance_turnover(scores_full[rows[decisions]], close_full, open_full, rows[decisions],
                                        lookback=LOOKBACK, ic=IC, risk_aversion=LAMBDA, weight_cap=CAP,
                                        penalty=PENALTY, sweeps=sweeps, warm_iters=warm, log=lambda *_: None)
        t_cons = time.perf_counter() - t0
        t_eq, t_turn = vec.simulate(Wt, decisions, open_, close, FEES)
        l1 = np.abs(Wt - W_exact).sum(axis=1) if same else np.array([np.nan])
        print(f"[vectorized + penalty, {sweeps} sweeps x {warm} warm iters] {t_cons:.2f}s  L1 to exact mean {l1.mean():.4f} max {l1.max():.4f}  {stats(t_eq, t_turn, len(decisions))}")
        compare(f"vectorized + penalty, {sweeps} sweeps", t_eq)
