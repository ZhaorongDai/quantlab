"""回归面板指标：误差类（MSE/RMSE/MAE/R²）与截面相关类（IC/RankIC）。

给收益模型用：预测值和目标值都是 `[T, S]` 的面板（时间 x 标的）。

公共约定（每个函数都遵守）：

- 输入先转成 float64；两个输入形状不同时抛 `ValueError`。
- 只统计「联合掩码」位置——预测和目标**两边都有限**的格子。一边是 NaN/inf 的
  格子不参与任何求和，也不参与排名。
- 没有可统计的位置时返回 NaN，并且**不发 RuntimeWarning**：所有「空集求均值」
  都用显式计数判断，不依赖 `np.nanmean` 在全 NaN 上的告警行为。

截面 IC 是逐时间戳的 Pearson 相关再对时间求均值；RankIC 是先按联合掩码置 NaN、
再逐行排名（平均秩处理平局）、然后对秩求 IC。两者都是向量化实现，函数体里没有
Python 行级循环（由 `tests/test_metrics.py` 的 AST 锁住）——量化面板动辄上万个
时间戳，逐行 `scipy.stats.pearsonr` 慢两到三个数量级。

首个调用点：`quantlab/base/model.py:MLModel._compute_metrics`（260914-lno）。
"""

import numpy as np
from scipy.stats import rankdata


def _joint(pred, target) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """转 float64、校验形状，返回 `(pred, target, 联合有限掩码)`。"""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {p.shape} vs {t.shape}"
        )
    return p, t, np.isfinite(p) & np.isfinite(t)


def mse(pred, target) -> float:
    """联合掩码上的均方误差；没有有效位置时为 NaN。"""
    p, t, mask = _joint(pred, target)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    diff = p[mask] - t[mask]
    return float(np.dot(diff, diff) / n)


def rmse(pred, target) -> float:
    """`sqrt(mse)`；没有有效位置时为 NaN。"""
    value = mse(pred, target)
    return float(np.sqrt(value)) if np.isfinite(value) else float("nan")


def mae(pred, target) -> float:
    """联合掩码上的平均绝对误差；没有有效位置时为 NaN。"""
    p, t, mask = _joint(pred, target)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.abs(p[mask] - t[mask]).sum() / n)


def r2(pred, target) -> float:
    """联合掩码上的决定系数 `1 - SS_res / SS_tot`。

    有效数 <2，或目标在有效位置上是常数（`SS_tot == 0`）时为 NaN——此时 R² 没有
    定义，返回任何数字都是编造。
    """
    p, t, mask = _joint(pred, target)
    n = int(mask.sum())
    if n < 2:
        return float("nan")
    tv = t[mask]
    ss_tot = float(np.sum((tv - tv.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((tv - p[mask]) ** 2))
    return 1.0 - ss_res / ss_tot


def cross_sectional_ic(pred, target) -> float:
    """截面 IC：逐时间戳（逐行）Pearson 相关，再对时间求均值。

    输入必须是 2-D `[T, S]`。以下行被跳过、不参与均值：

    - 联合有效标的数 <2；
    - 预测或目标在有效位置上是常数。常数用「掩码后 max 等于 min」**精确**判定，
      不用浮点方差阈值——阈值会把真实的小方差截面误判成常数。

    全部行都被跳过时返回 NaN。
    """
    p, t, mask = _joint(pred, target)
    if p.ndim != 2:
        raise ValueError(f"cross_sectional_ic expects a 2-D [T, S] panel, got shape {p.shape}")

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        n = mask.sum(axis=1)
        safe_n = np.maximum(n, 1)
        p_mean = np.where(mask, p, 0.0).sum(axis=1) / safe_n
        t_mean = np.where(mask, t, 0.0).sum(axis=1) / safe_n
        p_dev = np.where(mask, p - p_mean[:, None], 0.0)
        t_dev = np.where(mask, t - t_mean[:, None], 0.0)
        cov = (p_dev * t_dev).sum(axis=1)
        var_p = (p_dev * p_dev).sum(axis=1)
        var_t = (t_dev * t_dev).sum(axis=1)

        p_varies = np.max(np.where(mask, p, -np.inf), axis=1, initial=-np.inf) > np.min(
            np.where(mask, p, np.inf), axis=1, initial=np.inf
        )
        t_varies = np.max(np.where(mask, t, -np.inf), axis=1, initial=-np.inf) > np.min(
            np.where(mask, t, np.inf), axis=1, initial=np.inf
        )
        valid = (n >= 2) & p_varies & t_varies & (var_p > 0) & (var_t > 0)

        n_valid = int(valid.sum())
        if n_valid == 0:
            return float("nan")
        per_row = cov[valid] / np.sqrt(var_p[valid] * var_t[valid])
        return float(per_row.sum() / n_valid)


def cross_sectional_rank_ic(pred, target) -> float:
    """截面 RankIC：先联合掩码、再逐行排名、然后对秩求 `cross_sectional_ic`。

    顺序很重要：必须先把联合掩码之外的格子在**两个**数组上都置为 NaN 再排名。
    反过来做，一个只在目标侧缺失的标的仍会占用预测侧的一个秩位，把其余标的的秩
    整体推移，得到的就不再是有效标的之间的秩相关。平局取平均秩。
    """
    p, t, mask = _joint(pred, target)
    if p.ndim != 2:
        raise ValueError(
            f"cross_sectional_rank_ic expects a 2-D [T, S] panel, got shape {p.shape}"
        )
    p_ranks = rankdata(np.where(mask, p, np.nan), axis=1, nan_policy="omit")
    t_ranks = rankdata(np.where(mask, t, np.nan), axis=1, nan_policy="omit")
    return cross_sectional_ic(p_ranks, t_ranks)


def regression_panel_metrics(pred, target) -> dict[str, float]:
    """一次算齐 `[T, S]` 面板的六个指标：`mse, rmse, mae, r2, ic, rank_ic`。"""
    return {
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "r2": r2(pred, target),
        "ic": cross_sectional_ic(pred, target),
        "rank_ic": cross_sectional_rank_ic(pred, target),
    }
