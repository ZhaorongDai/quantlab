"""Markowitz mean-variance weights with a turnover penalty, solved with cvxpy.

``MeanVarianceOptimizer`` maximises, on every rebalance bar,

    w @ mu - risk_aversion / 2 * w @ Sigma @ w - turnover_penalty * |w - w_current|_1

over long-only, fully invested weights under a per-symbol cap. The expected
return ``mu`` is calibrated from a label's prediction (Grinold: ``ic *
sigma * z``), the covariance ``Sigma`` comes from a risk model, and both are
on the span of the expected-return label. This is the only quantlab module
that imports cvxpy.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
import xarray as xr

from quantlab.base.config import MeanVarianceConfig
from quantlab.base.portfolio import (
    PortfolioConstructionError,
    PortfolioConstructor,
    PortfolioContext,
)

_SOLVED = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


@dataclass(frozen=True)
class MeanVarianceInputs:
    """The problem one bar poses: the candidate symbols and their inputs.

    Attributes
    ----------
    symbols : np.ndarray
        The candidates: eligible, with a finite expected-return prediction
        and covered by the risk model.
    expected_return : np.ndarray
        ``mu`` per candidate, on the expected-return label's span.
    covariance : np.ndarray
        ``Sigma`` over the candidates, on the same span.
    current_weights : np.ndarray
        The weights currently held on the candidates.

    Examples
    --------
    >>> inputs = optimizer.problem_inputs(context)
    >>> inputs.symbols.tolist(), inputs.covariance.shape
    (['AAA', 'BBB', 'CCC', 'DDD'], (4, 4))
    """

    symbols: np.ndarray
    expected_return: np.ndarray
    covariance: np.ndarray
    current_weights: np.ndarray


def _zscore(values: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score with ``ddof=1``; all 0.0 when it is undefined."""
    if values.size < 2:
        return np.zeros_like(values)
    std = values.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def _project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    """Return the nearest point to ``values`` with ``sum = 1`` and ``0 <= w <= cap``.

    The Euclidean projection is ``clip(values - tau, 0, cap)`` for the shift
    ``tau`` that makes it sum to one, found by bisection; it removes a
    solver's round-off without moving an exact solution.
    """
    low, high = values.min() - cap, values.max()
    for _ in range(200):
        tau = (low + high) / 2
        if np.clip(values - tau, 0.0, cap).sum() > 1:
            low = tau
        else:
            high = tau
    projected = np.clip(values - (low + high) / 2, 0.0, cap)
    return projected


class MeanVarianceOptimizer(PortfolioConstructor):
    """Long-only mean-variance weights with a turnover penalty.

    On each rebalance bar the candidates are the symbols the context marks
    eligible, with a finite prediction of ``expected_return_label`` and
    covered by the risk model (enough return history); every other symbol
    gets 0.0. Over the candidates the optimiser solves

        maximise    w @ mu - risk_aversion / 2 * w @ Sigma @ w
                    - turnover_penalty * |w - w_current|_1
        subject to  sum(w) = 1,  0 <= w <= weight_cap

    where ``Sigma`` is the risk model's one-bar covariance times the span
    ``n`` of ``expected_return_label`` (variance is linear in time), and
    ``mu = ic * sigma * z`` (Grinold): ``sigma`` the square root of
    ``Sigma``'s diagonal and ``z`` the candidates' cross-sectional z-score
    (``ddof=1``) of the prediction. So the prediction only ranks, and any
    model's output can feed the optimiser. ``w_current`` is the context's
    current weights. The solver's solution is projected onto the feasible
    set (the nearest weights summing to one within ``[0, weight_cap]``),
    removing its round-off.

    A bar without candidates, or with fewer than ``1 / weight_cap``, is
    infeasible, and a bar the solver fails on or leaves unsolved raises
    ``PortfolioConstructionError``: the backtest holds the current position
    there and records the bar.

    ``lookback_bars`` is the risk model's. ``bind`` reads the span from the
    predictor's label, so a backtest binds the optimiser when it is built.

    Parameters
    ----------
    config : MeanVarianceConfig
        ``expected_return_label``, ``risk_model``, ``ic``,
        ``risk_aversion``, ``turnover_penalty``, ``weight_cap`` and
        ``direction`` (only ``"long_only"``).

    Raises
    ------
    ValueError
        If ``direction`` is not ``"long_only"``, ``weight_cap`` is not in
        ``(0, 1]``, or ``risk_aversion`` or ``turnover_penalty`` is
        negative.

    Examples
    --------
    ``model`` predicts the 5-bar ``ret_5``; ``context`` is a bar of four
    symbols ``AAA``..``DDD``, all eligible and none held, with 60 bars of
    one-bar returns of volatility 1%, 1.5%, 2% and 2.5% and ``ret_5``
    predictions 0.8, -0.1, -0.3 and 0.2. With a risk aversion of 5 on so small
    an expected return the book leans toward the low-volatility symbols:

    >>> from quantlab.base.config import LedoitWolfConfig, MeanVarianceConfig
    >>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
    >>> optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
    ...     expected_return_label="ret_5",
    ...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
    ...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
    ... ))
    >>> optimizer.lookback_bars
    60
    >>> optimizer.bind(model)
    >>> weights = optimizer.construct(context)
    >>> float(weights.sum().round(6)), bool((weights >= 0).all()), bool((weights <= 0.4).all())
    (1.0, True, True)
    """

    config_cls = MeanVarianceConfig

    def __init__(self, config: MeanVarianceConfig):
        """Initialize the optimiser; see the class docstring for parameters."""
        super().__init__(config)
        if config.direction != "long_only":
            raise ValueError(
                f"MeanVarianceOptimizer supports direction='long_only' only, got "
                f"{config.direction!r}"
            )
        if not 0 < config.weight_cap <= 1:
            raise ValueError(f"weight_cap must be in (0, 1], got {config.weight_cap}")
        for name in ("risk_aversion", "turnover_penalty"):
            if getattr(config, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(config, name)}")
        self._span: int | None = None

    @property
    def lookback_bars(self) -> int:
        """The risk model's ``lookback_bars``.

        Examples
        --------
        >>> optimizer.lookback_bars
        60
        """
        return self.config.risk_model.lookback_bars

    @property
    def span(self) -> int | None:
        """The expected-return label's span in bars, once ``bind`` has read it.

        Examples
        --------
        >>> optimizer.span
        5
        """
        return self._span

    def bind(self, predictor) -> None:
        """Check the predictor predicts ``expected_return_label`` and read its span.

        Parameters
        ----------
        predictor : Predictor
            The backtest's predictor.

        Raises
        ------
        ValueError
            If the predictor does not predict ``expected_return_label``, or
            the label has no ``span_bars()`` (it is not a ``Forward`` label).

        Examples
        --------
        >>> optimizer.bind(model)
        >>> optimizer.span
        5
        """
        name = self.config.expected_return_label
        labels = self._label_names(predictor)
        if name not in labels:
            raise ValueError(
                f"expected_return_label {name!r} is not one of the predictor's "
                f"labels {labels}"
            )
        label = next(
            label for label in predictor.labels if name in label.get_factor_names()
        )
        span_bars = getattr(label, "span_bars", None)
        if span_bars is None:
            raise ValueError(
                f"expected_return_label {name!r} is a {type(label).__name__}, which "
                f"has no span_bars(); the expected return needs a Forward label"
            )
        self._span = int(span_bars())

    def problem_inputs(self, context: PortfolioContext) -> MeanVarianceInputs:
        """Return the candidates of the context's bar and their ``mu``, ``Sigma`` and weights.

        See the class docstring for how each is computed.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context.

        Returns
        -------
        MeanVarianceInputs
            The candidates in the context's symbol order.

        Raises
        ------
        RuntimeError
            If the optimiser has not been bound to a predictor.

        Examples
        --------
        >>> inputs = optimizer.problem_inputs(context)
        >>> inputs.expected_return.round(4)
        array([ 0.0017, -0.0008, -0.0022,  0.0003])
        """
        if self._span is None:
            raise RuntimeError(
                "MeanVarianceOptimizer is not bound to a predictor; call bind() "
                "first (a backtest does when it is built)"
            )
        symbols = context.symbols
        prediction = np.asarray(
            context.predictions[self.config.expected_return_label]
            .sel(symbol=symbols)
            .values,
            dtype=np.float64,
        )
        estimate = self.config.risk_model.estimate(context).scaled(self._span)
        position = pd.Index(estimate.symbols).get_indexer(symbols)
        candidate = (
            np.asarray(context.eligible.values, dtype=bool)
            & np.isfinite(prediction)
            & (position >= 0)
        )
        index = np.flatnonzero(candidate)
        rows = position[index]
        covariance = estimate.covariance[np.ix_(rows, rows)]
        sigma = np.sqrt(np.diag(covariance))
        expected = self.config.ic * sigma * _zscore(prediction[index])
        current = np.asarray(
            context.current_weights.sel(symbol=symbols).values, dtype=np.float64
        )[index]
        return MeanVarianceInputs(
            symbols=symbols[index],
            expected_return=expected,
            covariance=covariance,
            current_weights=current,
        )

    def construct(self, context: PortfolioContext) -> xr.DataArray:
        """Solve the bar's mean-variance problem and return its weights.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context.

        Returns
        -------
        xr.DataArray
            One finite weight per symbol of ``context.symbols``, summing to
            one.

        Raises
        ------
        PortfolioConstructionError
            If the bar is infeasible or the solver fails or leaves it
            unsolved.

        Examples
        --------
        >>> optimizer.construct(context).values.round(3)
        array([0.4  , 0.4  , 0.009, 0.191])
        """
        config = self.config
        inputs = self.problem_inputs(context)
        n = len(inputs.symbols)
        if n == 0 or n * config.weight_cap < 1 - 1e-12:
            raise PortfolioConstructionError(
                f"infeasible: {n} candidate symbol(s) cannot hold a fully invested "
                f"book under weight_cap={config.weight_cap}"
            )
        w = cp.Variable(n)
        objective = (
            inputs.expected_return @ w
            - config.risk_aversion / 2 * cp.quad_form(w, cp.psd_wrap(inputs.covariance))
            - config.turnover_penalty * cp.norm1(w - inputs.current_weights)
        )
        problem = cp.Problem(
            cp.Maximize(objective), [cp.sum(w) == 1, w >= 0, w <= config.weight_cap]
        )
        try:
            problem.solve()
        except cp.error.SolverError as exc:
            raise PortfolioConstructionError(f"the solver failed: {exc}") from exc
        if problem.status not in _SOLVED or w.value is None:
            raise PortfolioConstructionError(f"no solution: status {problem.status!r}")
        solution = _project_capped_simplex(
            np.asarray(w.value, dtype=np.float64), config.weight_cap
        )

        row = pd.Series(0.0, index=context.symbols)
        row[inputs.symbols] = solution
        return xr.DataArray(row.values, dims="symbol", coords={"symbol": context.symbols})
