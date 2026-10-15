"""PROTOTYPE, throwaway: a backtest with no loop over bars.

Answers one question: how fast, and how far from ``BaseBacktester.run()``,
is a backtest whose TopN, mean-variance and simulation are array operations
over the whole ``[T, S]`` panel? Not library code; nothing imports it.

Semantics kept from the exact engine: a target decided at bar t's close
fills at bar t + 1's open (``adjOpen``), the book is marked at ``adjClose``
(forward-filled, so a delisted holding sits at its last valuation), costs
are ``cost * |traded weight|`` at each fill, and between fills the holdings
drift with prices. Dropped: rejected orders, locked positions, settlement
into cash, the cash cap, ``min_trade``, and for mean-variance the turnover
penalty (each bar is solved on its own) and the QP solver (batched
projected gradient instead).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along axis 0."""
    rows = np.where(np.isfinite(a), np.arange(len(a))[:, None], 0)
    np.maximum.accumulate(rows, axis=0, out=rows)
    return a[rows, np.arange(a.shape[1])]


def simulate(targets: np.ndarray, decisions: np.ndarray, open_: np.ndarray, close: np.ndarray, cost: float):
    """Equity (1 at the first bar) of holding ``targets[k]`` from the open after ``decisions[k]``.

    targets : [K, S] weights decided at the close of row ``decisions[k]``.
    Returns (equity [T], turnover [K]); no loop over bars or fills.
    """
    T, S = close.shape
    c = ffill(close)
    prev_c = np.vstack([np.full((1, S), np.nan), c[:-1]])
    o = np.where(np.isfinite(open_), open_, prev_c)
    keep = decisions + 1 < T
    fills = decisions[keep] + 1
    W = np.nan_to_num(targets[keep])
    K = len(fills)
    cash = 1.0 - W.sum(axis=1)

    o_fill = o[fills]                                     # [K, S]
    # Growth of each fill's book to the next fill's open.
    growth = np.nan_to_num(o_fill[1:] / o_fill[:-1], nan=1.0)  # [K-1, S]
    rel_open = cash[:-1] + (W[:-1] * growth).sum(axis=1)  # [K-1]
    drifted = W[:-1] * growth / rel_open[:, None]
    turnover = np.empty(K)
    turnover[0] = np.abs(W[0]).sum()
    turnover[1:] = np.abs(W[1:] - drifted).sum(axis=1)
    step = np.ones(K)
    step[1:] = rel_open
    A = np.cumprod(step * (1.0 - cost * turnover))       # value at each fill's open, after costs

    seg = np.searchsorted(fills, np.arange(T), side="right") - 1  # last fill at or before t
    live = seg >= 0
    s = seg[live]
    ratio = np.nan_to_num(c[live] / o_fill[s], nan=1.0)
    equity = np.ones(T)
    equity[live] = A[s] * (cash[s] + (W[s] * ratio).sum(axis=1))
    return equity, turnover


def top_n(scores: np.ndarray, eligible: np.ndarray, n: int) -> np.ndarray:
    """Equal-weight long-only top ``n`` per row among ``eligible`` finite scores."""
    s = np.where(eligible & np.isfinite(scores), scores, -np.inf)
    k = min(n, s.shape[1])
    picks = np.argpartition(-s, k - 1, axis=1)[:, :k]
    chosen = np.take_along_axis(s, picks, axis=1) > -np.inf
    W = np.zeros_like(s)
    np.put_along_axis(W, picks, chosen.astype(float), axis=1)
    count = W.sum(axis=1, keepdims=True)
    return np.divide(W, count, out=np.zeros_like(W), where=count > 0)


def project_capped_simplex(v: np.ndarray, cap: np.ndarray, total: float = 1.0, iters: int = 60) -> np.ndarray:
    """Row-wise Euclidean projection onto {sum = total, 0 <= w <= cap}, by batched bisection."""
    low = (v - cap).min(axis=1, keepdims=True)
    high = v.max(axis=1, keepdims=True)
    for _ in range(iters):
        tau = (low + high) / 2
        over = np.clip(v - tau, 0.0, cap).sum(axis=1, keepdims=True) > total
        low = np.where(over, tau, low)
        high = np.where(over, high, tau)
    return np.clip(v - high, 0.0, cap)


def ledoit_wolf_lowrank(close: np.ndarray, rows: np.ndarray, lookback: int):
    """Batched Ledoit-Wolf over the ``lookback`` one-bar returns ending at each of ``rows``.

    Returns (a [B], U [B, S, L], covered [B, S]) with Sigma = a I + U U' on
    the covered symbols (Woodbury form; the [S, S] matrix is never built).
    Matches ``sklearn.covariance.ledoit_wolf`` on the covered columns.
    """
    c = ffill(close)
    r = np.vstack([np.full((1, c.shape[1]), np.nan), c[1:] / c[:-1] - 1.0])
    win = sliding_window_view(r, lookback, axis=0)[rows - lookback + 1]  # [B, S, L]
    covered = np.isfinite(win).all(axis=2) & (np.nanstd(win, axis=2) > 0)
    X = np.where(covered[..., None], win - np.nanmean(win, axis=2, keepdims=True), 0.0)
    X = np.transpose(X, (0, 2, 1))                                        # [B, L, S]
    n = lookback
    p = covered.sum(axis=1).astype(float)
    X2 = X**2
    trace = X2.sum(axis=(1, 2)) / n
    mu = trace / p
    G = X @ np.transpose(X, (0, 2, 1))                                    # [B, L, L]
    delta_ = (G**2).sum(axis=(1, 2)) / n**2
    beta_ = (X2.sum(axis=2) ** 2).sum(axis=1)  # sum(X2' X2) = sum_k (sum_i x_ki^2)^2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * trace + p * mu**2) / p
    beta = np.minimum(beta, delta)
    shrink = np.where(delta > 0, beta / np.where(delta > 0, delta, 1.0), 0.0)
    a = shrink * mu
    U = np.transpose(X, (0, 2, 1)) * np.sqrt((1.0 - shrink) / n)[:, None, None]
    return a, U, covered


def mean_variance(scores, close, rows, *, lookback, ic, risk_aversion, weight_cap, span=1.0,
                  iters=200, chunk=64):
    """Long-only Grinold mean-variance per decision row, no turnover term, batched over rows.

    max mu'w - lambda/2 w' Sigma w, sum w = 1, 0 <= w <= cap; mu = ic * sigma * z.
    Solved by accelerated projected gradient (FISTA) on all rows at once; the
    only loop over rows is the memory chunking.
    """
    out = np.zeros((len(rows), close.shape[1]))
    for lo in range(0, len(rows), chunk):
        part = rows[lo:lo + chunk]
        a, U, covered = ledoit_wolf_lowrank(close, part, lookback)
        a, U = a * span, U * np.sqrt(span)
        pred = scores[lo:lo + chunk]
        cand = covered & np.isfinite(pred)
        m = np.where(cand, pred, np.nan)
        z = (m - np.nanmean(m, axis=1, keepdims=True)) / np.nanstd(m, axis=1, ddof=1, keepdims=True)
        sigma = np.sqrt(a[:, None] + (U**2).sum(axis=2))
        mu = np.where(cand, ic * sigma * np.nan_to_num(z), 0.0)
        cap = np.where(cand, weight_cap, 0.0)
        UtU = np.transpose(U, (0, 2, 1)) @ U
        lmax = a + np.linalg.eigvalsh(UtU)[:, -1]
        eta = (1.0 / (risk_aversion * lmax))[:, None]

        def sigma_w(w):
            return a[:, None] * w + np.einsum("bsl,bl->bs", U, np.einsum("bsl,bs->bl", U, w))

        w = project_capped_simplex(cap, cap)
        y, t = w, 1.0
        for _ in range(iters):
            w_next = project_capped_simplex(y + eta * (mu - risk_aversion * sigma_w(y)), cap)
            t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
            y = w_next + ((t - 1) / t_next) * (w_next - w)
            w, t = w_next, t_next
        out[lo:lo + chunk] = w
    return out


def smooth(targets: np.ndarray, k: float) -> np.ndarray:
    """Partial adjustment w_t = w_{t-1} + k (w*_t - w_{t-1}), an EWMA along time (lfilter)."""
    from scipy.signal import lfilter

    # Starts at targets[0] (y[-1] = targets[0]), so every row stays a convex combination.
    return lfilter([k], [1.0, -(1.0 - k)], targets, axis=0, zi=(1.0 - k) * targets[:1])[0]


def project_capped_simplex_l1(v, cap, anchor, theta, total=1.0, iters=60):
    """Row-wise prox of theta*|w - anchor| + indicator{sum = total, 0 <= w <= cap}, by batched bisection.

    For a shift tau each coordinate's minimiser is the soft-threshold of
    ``v - tau`` toward ``anchor`` clipped to [0, cap]; the sum falls with tau.
    """
    def at(tau):
        u = v - tau
        d = u - anchor
        return np.clip(np.where(np.abs(d) <= theta, anchor, u - theta * np.sign(d)), 0.0, cap)

    low = (v - cap - theta).min(axis=1, keepdims=True) - 1e-12
    high = (v + theta).max(axis=1, keepdims=True) + 1e-12
    for _ in range(iters):
        tau = (low + high) / 2
        over = at(tau).sum(axis=1, keepdims=True) > total
        low = np.where(over, tau, low)
        high = np.where(over, high, tau)
    return at(high)


def mean_variance_turnover(scores, close, open_, rows, *, lookback, ic, risk_aversion, weight_cap,
                           penalty, span=1.0, iters=200, warm_iters=200, sweeps=10, tol=1e-6, log=print):
    """Long-only Grinold mean-variance with the L1 turnover penalty, parallel in time.

    max mu'w - lambda/2 w'Sigma w - penalty |w - h|, h the previous decision's
    weights drifted to this bar's close. The decisions are coupled only
    through h, so every sweep solves ALL bars at once with h taken from the
    previous sweep (block Jacobi over time); sweep j is exact on the first j
    bars, and stops early once no row moves by more than ``tol`` (L1).
    ``close``/``open_`` are the full panels ``rows`` index into.
    """
    a, U, covered = ledoit_wolf_lowrank(close, rows, lookback)
    a, U = a * span, U * np.sqrt(span)
    cand = covered & np.isfinite(scores)
    m = np.where(cand, scores, np.nan)
    z = (m - np.nanmean(m, axis=1, keepdims=True)) / np.nanstd(m, axis=1, ddof=1, keepdims=True)
    sigma = np.sqrt(a[:, None] + (U**2).sum(axis=2))
    mu = np.where(cand, ic * sigma * np.nan_to_num(z), 0.0)
    cap = np.where(cand, weight_cap, 0.0)
    lmax = a + np.linalg.eigvalsh(np.transpose(U, (0, 2, 1)) @ U)[:, -1]
    eta = (1.0 / (risk_aversion * lmax))[:, None]
    theta = eta * penalty

    c = ffill(close)
    # Growth from the previous decision's fill (its next bar's open) to this decision's close.
    prev_fill = np.concatenate([[rows[0]], rows[:-1] + 1])
    growth = np.nan_to_num(c[rows] / np.where(np.isfinite(open_[prev_fill]), open_[prev_fill], c[prev_fill]), nan=1.0)

    def sigma_w(w):
        return a[:, None] * w + np.einsum("bsl,bl->bs", U, np.einsum("bsl,bs->bl", U, w))

    W = np.zeros_like(mu)
    for sweep in range(sweeps):
        prev = np.vstack([np.zeros((1, W.shape[1])), W[:-1]]) * growth
        held = prev.sum(axis=1, keepdims=True)
        h = prev / (1.0 - np.vstack([np.zeros((1, W.shape[1])), W[:-1]]).sum(axis=1, keepdims=True) + held)
        w = W.copy() if sweep else project_capped_simplex(cap, cap)
        y, t = w, 1.0
        for _ in range(iters if sweep == 0 else warm_iters):
            w_next = project_capped_simplex_l1(y + eta * (mu - risk_aversion * sigma_w(y)), cap, h, theta)
            t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
            y = w_next + ((t - 1) / t_next) * (w_next - w)
            w, t = w_next, t_next
        moved = np.abs(w - W).sum(axis=1)
        W = w
        log(f"  sweep {sweep + 1}: max row move {moved.max():.2e}, rows moved > tol {(moved > tol).sum()}")
        if moved.max() <= tol:
            break
    return W
