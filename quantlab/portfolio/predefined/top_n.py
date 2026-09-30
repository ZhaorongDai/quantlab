"""Equal-weight top-n selection, the simplest portfolio construction rule.

``TopNConstructor`` ranks the symbols of a bar by one label's prediction and
holds the top ``top_n`` with equal weights, or with ``direction="long_short"``
also shorts the bottom ``top_n``. Its per-bar ``construct`` and its
vectorised ``construct_panel`` share one book-building function, so the two
agree bit for bit.
"""

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.base.config import TopNConfig
from quantlab.base.portfolio import PortfolioConstructor, PortfolioContext


class TopNConstructor(PortfolioConstructor):
    """Equal-weight top-n selection on each rebalance bar.

    With ``direction="long_only"`` each of the ``k`` highest-scoring symbols
    gets a weight of ``1/k``, for a gross exposure of 100%. With
    ``direction="long_short"`` the top ``k`` symbols get ``+0.5/k`` each and
    the bottom ``k`` get ``-0.5/k`` each; the two books never share a
    symbol, so the gross exposure is 100% and the net exposure is zero.
    ``k`` is ``top_n``, reduced when fewer symbols are eligible, and a
    warning names the bar. A symbol is eligible when the context marks it
    so and its score is finite. Eligible symbols are ranked with a stable
    sort, so ties resolve by symbol order. With no eligible symbol the row
    is all 0.0, that is, flat. The current weights are not read.

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
    ...     eligible=xr.DataArray([True, True, True, True], dims="symbol",
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

    def _book(self, scores: np.ndarray, eligible: np.ndarray, timestamp) -> np.ndarray:
        """Return one rebalance bar's weights from its scores and eligibility."""
        row = np.zeros(scores.shape[0], dtype=np.float64)
        idx = np.flatnonzero(eligible & np.isfinite(scores))
        order = idx[np.argsort(-scores[idx], kind="stable")]
        top_n = self.config.top_n
        long_only = self.config.direction == "long_only"
        k = min(top_n, order.size) if long_only else min(top_n, order.size // 2)
        if k < top_n:
            logger.warning(
                f"{pd.Timestamp(timestamp)}: only {k} eligible symbol(s) per book "
                f"for top_n={top_n}"
            )
        if k > 0:
            if long_only:
                row[order[:k]] = 1.0 / k
            else:
                row[order[:k]] = 0.5 / k
                row[order[-k:]] = -0.5 / k
        return row

    def construct(self, context: PortfolioContext) -> xr.DataArray:
        """Return the equal-weight top-n book of the context's bar.

        See the class docstring for the rule and an example.

        Parameters
        ----------
        context : PortfolioContext
            The bar's predictions and eligibility.

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
            np.asarray(context.eligible.values, dtype=bool),
            context.timestamp,
        )
        return xr.DataArray(row, dims="symbol", coords={"symbol": symbols})

    def construct_panel(
        self,
        predictions: xr.Dataset,
        eligible: xr.DataArray,
        rebalance: np.ndarray,
        *,
        fill_price: xr.DataArray | None = None,
        valuation_price: xr.DataArray | None = None,
        factors: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Build top-n target weights for every bar of a panel, vectorised.

        Returns exactly what ``PortfolioConstructor.construct_panel``'s
        per-bar loop returns, without building a context per bar. Top-n
        reads neither the current weights nor the prices, and never fails
        a bar.

        Parameters
        ----------
        predictions : xr.Dataset
            One variable per label on ``(timestamp, symbol)``.
        eligible : xr.DataArray
            Booleans on the same labels as ``predictions`` (in any order).
        rebalance : np.ndarray
            One boolean per timestamp, True on rebalance bars.
        fill_price, valuation_price : xr.DataArray, optional
            Prices, checked like the loop checks them and not read.
        factors : xr.Dataset, optional
            Factor panels, checked like the loop checks them and not read;
            top-n declares no ``required_factors()``.

        Returns
        -------
        xr.Dataset
            One ``weight`` variable on ``(timestamp, symbol)``; all NaN on a
            bar that does not rebalance. ``attrs["failed_bars"]`` is empty.

        Raises
        ------
        ValueError
            If ``rebalance`` does not have one entry per timestamp or the
            eligibility panel is on other labels.

        Examples
        --------
        >>> ts = pd.bdate_range("2024-01-01", periods=3)
        >>> scores = xr.Dataset(
        ...     {"ret": (("timestamp", "symbol"), [[0.3, 0.1, np.nan, 0.2],
        ...                                        [0.0, 0.5, 0.4, 0.1],
        ...                                        [0.9, 0.8, 0.7, 0.6]])},
        ...     coords={"timestamp": ts, "symbol": ["AAA", "BBB", "CCC", "DDD"]},
        ... )
        >>> eligible = xr.ones_like(scores["ret"], dtype=bool)
        >>> rule.construct_panel(scores, eligible, np.array([True, False, False]))["weight"].values
        array([[0.5, 0. , 0. , 0.5],
               [nan, nan, nan, nan],
               [nan, nan, nan, nan]])
        """
        predictions = predictions.transpose("timestamp", "symbol")
        eligible_values = self._check_eligible(eligible, predictions)
        rebalance = self._check_rebalance(rebalance, predictions)
        self._check_prices(fill_price, valuation_price, predictions)
        self._check_factors(factors, predictions)
        scores = np.asarray(
            predictions[self._score_label(predictions)].values, dtype=np.float64
        )
        timestamps = predictions.timestamp.values
        weights = np.full(scores.shape, np.nan, dtype=np.float64)
        for t in np.flatnonzero(rebalance):
            weights[t] = self._book(scores[t], eligible_values[t], timestamps[t])
        out = xr.Dataset(
            {"weight": (("timestamp", "symbol"), weights)},
            coords={"timestamp": timestamps, "symbol": predictions.symbol.values},
        )
        out.attrs["failed_bars"] = []
        return out
