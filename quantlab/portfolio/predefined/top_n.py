"""Equal-weight top-n selection, the simplest portfolio construction rule.

``TopNConstructor`` ranks the symbols of a bar by one label's prediction and
holds the top ``top_n`` with equal weights, or with ``direction="long_short"``
also shorts the bottom ``top_n``, around the locked positions it must keep.
"""

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.base.config import TopNConfig
from quantlab.base.portfolio import PortfolioConstructor, PortfolioContext


class TopNConstructor(PortfolioConstructor):
    """Equal-weight top-n selection on each rebalance bar.

    A locked position (held, not tradable at the bar) keeps its current
    weight and is never picked. The picks come from the other symbols that
    are tradable with a finite score, ranked with a stable sort, so ties
    resolve by symbol order. With ``direction="long_only"`` each of the
    ``k`` highest-scoring picks gets ``(1 - L) / k``, ``L`` the sum of the
    locked weights; with nothing locked that is ``1/k``, for a gross
    exposure of 100%. With ``direction="long_short"`` the top ``k`` get
    ``(0.5 - L_long) / k`` each and the bottom ``k`` get ``-(0.5 - L_short)
    / k`` each, ``L_long`` and ``L_short`` the locked long and short
    exposure; the two books never share a symbol. A side with no budget
    left adds nothing. ``k`` is ``top_n``, reduced when fewer symbols can
    be picked, and a warning names the bar. With no pick the row is the
    locked positions and 0.0 elsewhere.

    Parameters
    ----------
    config : TopNConfig
        ``direction``, ``top_n`` and ``score_label``, the label ranking the
        symbols (``None`` for the predictor's first label).

    Raises
    ------
    ValueError
        If ``direction`` is not ``"long_only"`` or ``"long_short"``, or
        ``top_n`` is smaller than 1.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.base.config import TopNConfig
    >>> from quantlab.base.portfolio import PortfolioContext
    >>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
    >>> symbols = ["AAA", "BBB", "CCC", "DDD"]
    >>> context = PortfolioContext(
    ...     timestamp=pd.Timestamp("2024-01-02"),
    ...     predictions=xr.Dataset({"ret": ("symbol", [0.3, 0.1, np.nan, 0.2])},
    ...                            coords={"symbol": symbols}),
    ...     tradable=xr.DataArray([True, True, True, True], dims="symbol",
    ...                           coords={"symbol": symbols}),
    ...     current_weights=xr.DataArray(np.zeros(4), dims="symbol",
    ...                                  coords={"symbol": symbols}),
    ... )
    >>> rule.construct(context).values
    array([0.5, 0. , 0. , 0.5])
    """

    config_cls = TopNConfig

    def __init__(self, config: TopNConfig):
        """Initialize the rule; see the class docstring for parameters."""
        super().__init__(config)
        if config.direction not in ("long_only", "long_short"):
            raise ValueError(
                f"direction must be 'long_only' or 'long_short', got "
                f"{config.direction!r}"
            )
        if config.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {config.top_n}")

    def bind(self, predictor) -> None:
        """Refuse a predictor without labels, or without ``score_label``.

        Any scale ranks, so ``label_scales`` is not read.

        Raises
        ------
        ValueError
            If ``labels`` is empty, or ``score_label`` is set and not one of
            them; the message lists the labels.

        Examples
        --------
        With ``model`` a model predicting ``ret_5`` and ``ret_1``:

        >>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2, score_label="ret_20"))
        >>> rule.bind(model)
        Traceback (most recent call last):
        ValueError: score_label 'ret_20' is not one of the predictor's labels ['ret_5', 'ret_1']
        """
        labels = self._label_names(predictor)
        if not labels:
            raise ValueError("the predictor declares no labels to score by")
        label = self.config.score_label
        if label is not None and label not in labels:
            raise ValueError(
                f"score_label {label!r} is not one of the predictor's labels "
                f"{list(labels)}"
            )

    def _score_label(self, predictions: xr.Dataset) -> str:
        """``score_label``, or the first prediction variable when it is None."""
        if self.config.score_label is not None:
            return self.config.score_label
        return str(next(iter(predictions.data_vars)))

    def _book(
        self, scores: np.ndarray, tradable: np.ndarray, current: np.ndarray, timestamp
    ) -> np.ndarray:
        """Return one rebalance bar's weights from its scores, tradability and holdings."""
        locked = ~tradable & (current != 0)
        row = np.where(locked, current, 0.0)
        idx = np.flatnonzero(tradable & ~locked & np.isfinite(scores))
        order = idx[np.argsort(-scores[idx], kind="stable")]
        top_n = self.config.top_n
        long_only = self.config.direction == "long_only"
        k = min(top_n, order.size) if long_only else min(top_n, order.size // 2)
        if k < top_n:
            logger.warning(
                f"{pd.Timestamp(timestamp)}: only {k} symbol(s) to pick per book "
                f"for top_n={top_n}"
            )
        if k > 0:
            if long_only:
                budget = 1.0 - row.sum()
                if budget > 0:
                    row[order[:k]] = budget / k
            else:
                long_budget = 0.5 - row[row > 0].sum()
                short_budget = 0.5 + row[row < 0].sum()
                if long_budget > 0:
                    row[order[:k]] = long_budget / k
                if short_budget > 0:
                    row[order[-k:]] = -short_budget / k
        return row

    def construct(self, context: PortfolioContext) -> xr.DataArray:
        """Return the equal-weight top-n book of the context's bar.

        See the class docstring for the rule and an example.

        Parameters
        ----------
        context : PortfolioContext
            The bar's predictions, tradability and current weights.

        Returns
        -------
        xr.DataArray
            One finite weight per symbol of ``context.symbols``.

        Examples
        --------
        >>> rule = TopNConstructor(TopNConfig(direction="long_short", top_n=1))
        >>> rule.construct(context).values
        array([ 0.5, -0.5,  0. ,  0. ])
        """
        symbols = context.symbols
        scores = context.predictions[self._score_label(context.predictions)]
        row = self._book(
            np.asarray(scores.sel(symbol=symbols).values, dtype=np.float64),
            np.asarray(context.tradable.values, dtype=bool),
            np.asarray(context.current_weights.sel(symbol=symbols).values, dtype=np.float64),
            context.timestamp,
        )
        return xr.DataArray(row, dims="symbol", coords={"symbol": symbols})
