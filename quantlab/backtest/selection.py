"""Cross-sectional selection building blocks for the backtest layer.

This module holds the pieces a cross-sectional backtester composes to turn a
panel of model scores into target weights: ``rebalance_mask`` decides which
bars rebalance, ``resolve_score_label`` picks the model label to rank by, and
``CrossSectionTopNSelector`` builds equal-weight top-N (or top-N and bottom-N)
target weights on those bars. Nothing here depends on a simulation engine, so
the weights can be handed to any backtester that accepts a ``weight`` panel on
``(timestamp, symbol)``.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


def rebalance_mask(n_bars: int, rebalance_periods: int) -> np.ndarray:
    """Return a boolean mask marking the bars on which the portfolio rebalances.

    The first bar rebalances and so does every ``rebalance_periods``-th bar
    after it. The last bar never rebalances: a signal formed there has no
    following bar inside the window to fill on.

    Args:
        n_bars: Number of bars in the backtest window.
        rebalance_periods: Rebalance every this many bars.

    Returns:
        A boolean array of length ``n_bars``.

    Raises:
        ValueError: If ``rebalance_periods`` is smaller than 1.

    Example:
        >>> rebalance_mask(7, 3)
        array([ True, False, False,  True, False, False, False])
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
    """Return the model label whose predictions rank the symbols.

    ``None`` selects the model's first label.

    Args:
        score_label: The requested label name, or ``None``.
        label_names: The label names the model declares, in order.

    Returns:
        The label name to score by.

    Raises:
        ValueError: If the model declares no labels, or ``score_label`` is not
            one of them.

    Example:
        >>> resolve_score_label(None, ["fwd_ret_1", "fwd_ret_5"])
        'fwd_ret_1'
        >>> resolve_score_label("fwd_ret_5", ["fwd_ret_1", "fwd_ret_5"])
        'fwd_ret_5'
    """
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
    """Equal-weight top-N selection on each rebalance bar.

    With ``direction="long_only"`` each of the ``k`` highest-scoring symbols
    gets a weight of ``1/k``, for a gross exposure of 100%. With
    ``direction="long_short"`` the top ``k`` symbols get ``+0.5/k`` each and
    the bottom ``k`` get ``-0.5/k`` each; the two books never share a symbol,
    so the gross exposure is 100% and the net exposure is zero. ``k`` is
    ``top_n``, reduced when fewer symbols are eligible.

    Attributes:
        direction: ``"long_only"`` or ``"long_short"``.
        top_n: Number of symbols selected per book on each rebalance bar.

    Raises:
        ValueError: If ``direction`` is not one of the two literals or
            ``top_n`` is smaller than 1.

    Example:
        >>> selector = CrossSectionTopNSelector(direction="long_only", top_n=2)
        >>> selector
        CrossSectionTopNSelector(direction='long_only', top_n=2)
    """

    direction: Literal["long_only", "long_short"]
    top_n: int

    def __post_init__(self):
        """Validate ``direction`` and ``top_n`` at construction time."""
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
        """Reorder ``next_fill_price`` onto the coordinate labels of ``scores``.

        Alignment is by label rather than by position: two panels of the same
        shape but a different symbol or timestamp order would otherwise pair
        each score with another symbol's fill price. A missing or extra label
        is refused instead of being treated as "not eligible", because a
        misaligned time axis would silently make every row unselectable.

        Raises:
            ValueError: If either axis has duplicate labels, or the two label
                sets differ on either axis.
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
        """Build target weights from scores, next-bar fill prices and a mask.

        A bar that does not rebalance gets an all-NaN row, meaning "hold the
        current position". On a rebalance bar every symbol gets a finite
        weight, with unselected symbols at exactly ``0.0``: a NaN there would
        be read by the engine as "keep the position" and would silently block
        the rest of the rebalance. A symbol is eligible when its score and its
        next-bar fill price are both finite, so a symbol with no price on the
        next bar (delisted) is never picked. Eligible symbols are ranked with a
        stable sort, so ties resolve by symbol order and the same panel always
        yields the same weights. When fewer than ``top_n`` symbols are eligible
        the book is split among those that are and a warning names the bar;
        with no eligible symbol at all the row is all ``0.0``, that is, flat.

        Args:
            scores: Model scores on ``(timestamp, symbol)``; higher is better.
            next_fill_price: The price each symbol would fill at on the next
                bar, on the same labels as ``scores`` (in any order).
            rebalance: Boolean mask with one entry per timestamp, ``True`` on
                rebalance bars.

        Returns:
            A dataset with one ``weight`` variable on ``(timestamp, symbol)``.

        Raises:
            ValueError: If ``rebalance`` does not have one entry per timestamp
                or the two panels' labels do not match.

        Example:
            >>> ts = pd.bdate_range("2024-01-01", periods=3)
            >>> scores = xr.DataArray(
            ...     [[0.3, 0.1, np.nan, 0.2],
            ...      [0.0, 0.5, 0.4, 0.1],
            ...      [0.9, 0.8, 0.7, 0.6]],
            ...     dims=("timestamp", "symbol"),
            ...     coords={"timestamp": ts, "symbol": ["AAA", "BBB", "CCC", "DDD"]},
            ... )
            >>> next_fill = xr.full_like(scores, 100.0)
            >>> selector = CrossSectionTopNSelector(direction="long_only", top_n=2)
            >>> weights = selector.select(scores, next_fill, rebalance_mask(3, 2))
            >>> weights["weight"].values
            array([[0.5, 0. , 0. , 0.5],
                   [nan, nan, nan, nan],
                   [nan, nan, nan, nan]])
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
