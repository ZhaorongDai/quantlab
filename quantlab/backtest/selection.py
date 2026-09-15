"""截面选股组件：调仓日程、打分标签解析与 TopN 等权目标权重（03.7 D-09..D-12、D-18）。

这些是纯函数/小组件，不依赖任何回测引擎，回测器以组合的方式使用它们（D-01）。
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


def rebalance_mask(n_bars: int, rebalance_periods: int) -> np.ndarray:
    """调仓 bar 的布尔掩码。

    锚点是窗口第一个 bar，之后每 `rebalance_periods` 个 bar 调仓一次（D-18）。
    最后一个 bar 强制不调仓：它的信号在窗口内没有 t+1 成交 bar
    （03.7-RESEARCH.md Pitfall 13）。
    """
    if rebalance_periods < 1:
        raise ValueError(
            f"rebalance_periods must be >= 1, got {rebalance_periods}"
        )
    mask = np.zeros(n_bars, dtype=bool)
    mask[::rebalance_periods] = True
    if n_bars > 0:
        mask[-1] = False
    return mask


def resolve_score_label(score_label: str | None, label_names: list[str]) -> str:
    """打分用的标签名：None 取第一个标签（D-11），未知名字报错。"""
    if not label_names:
        raise ValueError("the model declares no labels to score by")
    if score_label is None:
        return label_names[0]
    if score_label not in label_names:
        raise ValueError(
            f"score_label {score_label!r} is not one of the model's labels "
            f"{list(label_names)}"
        )
    return score_label


@dataclass(frozen=True)
class CrossSectionTopNSelector:
    """截面 TopN 等权选股（D-09、D-10、D-12）。

    - `long_only`：分数最高的 k 个标的各 1/k，合计 100%；
    - `long_short`：最高 k 个各 +0.5/k，最低 k 个各 -0.5/k，两本书不共享标的，
      毛敞口 100%、净敞口 0。
    """

    direction: Literal["long_only", "long_short"]
    top_n: int

    def __post_init__(self):
        if self.direction not in ("long_only", "long_short"):
            raise ValueError(
                f"direction must be 'long_only' or 'long_short', got "
                f"{self.direction!r}"
            )
        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")

    @staticmethod
    def _align_to_scores(
        scores: xr.DataArray, next_fill_price: xr.DataArray
    ) -> xr.DataArray:
        """把下一 bar 成交价按**坐标标签**对齐到分数上（代码审查 WR-09）。

        以前只比较形状，然后各取 `.values` 按位置配对：形状相同、但标的或时间
        顺序不同的两块面板，会把每个分数配上**另一个**标的的成交价可得性，而
        权重却挂在分数的坐标上。回测器内部靠 `reindex` 保证了轴一致，但直接
        调用本组件的人（Phase 5 的优化器旁路、将来的兄弟回测器）不受保护。

        - 任一轴有重复标签：ValueError；
        - 两个轴的标签集合不同（缺或多）：ValueError，写明哪个轴、缺了哪些、
          多了哪些。缺成交价不能悄悄当成「不可选」：错位的时间轴（比如移错了
          方向的 shift）会整行变成不可选，看起来只是没选到股票；
        - 集合相同、顺序不同：按分数的顺序重排成交价。
        """
        for dim in ("timestamp", "symbol"):
            wanted = pd.Index(scores[dim].values)
            got = pd.Index(next_fill_price[dim].values)
            if wanted.has_duplicates or got.has_duplicates:
                raise ValueError(
                    f"{dim} labels must be unique in both scores and "
                    f"next_fill_price"
                )
            if wanted.equals(got):
                continue
            missing = wanted.difference(got, sort=False)
            extra = got.difference(wanted, sort=False)
            if len(missing) or len(extra):
                raise ValueError(
                    f"next_fill_price {dim} labels differ from the scores': "
                    f"missing {[str(v) for v in missing[:10]]}, extra "
                    f"{[str(v) for v in extra[:10]]}; fill prices must be given "
                    f"on exactly the scores' labels (WR-09)"
                )
        return next_fill_price.sel(
            timestamp=scores.timestamp.values, symbol=scores.symbol.values
        )

    def select(
        self,
        scores: xr.DataArray,
        next_fill_price: xr.DataArray,
        rebalance: np.ndarray,
    ) -> xr.Dataset:
        """分数 + 下一 bar 成交价 + 调仓掩码 -> 满足 D-03 契约的目标权重。

        规则（均由 tests/test_backtest_selection.py 锁定）：

        - 非调仓 bar 整行 NaN（持有）。调仓 bar 先整行写 0.0：NaN 在调仓 bar 上
          的意思是「保持原仓位」，会悄悄挡住整次调仓（03.7-RESEARCH.md Pitfall 3）。
        - 可选 = 分数有限且下一 bar 成交价有限（D-12）。NaN 与 inf 分数都不可选；
          t+1 没有成交价（已退市）的标的也不可选。
        - 可选标的按分数从高到低排序。排序是稳定的，平分时按标的轴顺序决定，
          同一面板永远得到同一组权重（D-25）。
        - `long_only`：k = min(top_n, 可选数)，排名前 k 个各 1/k（D-09）。
        - `long_short`：k = min(top_n, 可选数 // 2)，排名前 k 个各 +0.5/k、
          末尾 k 个各 -0.5/k。可选数不足 2·top_n 时两本书依然不共享标的，
          毛敞口 1、净敞口 0（D-09，03.7-RESEARCH.md A6）。
        - k < top_n 时记一条 warning，写明该 bar 的时间戳（D-12）；k == 0 时
          该行全 0.0，即清仓，而不是 NaN。
        """
        scores = scores.transpose("timestamp", "symbol")
        next_fill_price = self._align_to_scores(
            scores, next_fill_price.transpose("timestamp", "symbol")
        )
        score_values = np.asarray(scores.values, dtype=np.float64)
        fill_values = np.asarray(next_fill_price.values, dtype=np.float64)
        rebalance = np.asarray(rebalance, dtype=bool)
        if rebalance.shape != (score_values.shape[0],):
            raise ValueError(
                f"rebalance mask shape {rebalance.shape} does not match "
                f"{score_values.shape[0]} timestamps"
            )
        if fill_values.shape != score_values.shape:
            raise ValueError(
                f"next_fill_price shape {fill_values.shape} does not match "
                f"scores shape {score_values.shape}"
            )

        weights = np.full(score_values.shape, np.nan, dtype=np.float64)
        timestamps = scores.timestamp.values
        for t in np.flatnonzero(rebalance):
            row = np.zeros(score_values.shape[1], dtype=np.float64)
            eligible = np.isfinite(score_values[t]) & np.isfinite(fill_values[t])
            idx = np.flatnonzero(eligible)
            order = idx[np.argsort(-score_values[t, idx], kind="stable")]

            if self.direction == "long_only":
                k = min(self.top_n, order.size)
            else:
                k = min(self.top_n, order.size // 2)

            if k < self.top_n:
                logger.warning(
                    f"{pd.Timestamp(timestamps[t])}: only {k} eligible "
                    f"symbol(s) per book for top_n={self.top_n}"
                )
            if k > 0:
                if self.direction == "long_only":
                    row[order[:k]] = 1.0 / k
                else:
                    row[order[:k]] = 0.5 / k
                    row[order[-k:]] = -0.5 / k
            weights[t] = row

        return xr.Dataset(
            {"weight": (("timestamp", "symbol"), weights)},
            coords={"timestamp": timestamps, "symbol": scores.symbol.values},
        )
