"""Markowitz mean-variance weights with a turnover penalty, solved with cvxpy.

``MeanVarianceOptimizer`` maximises, on every rebalance bar,

    w @ mu - risk_aversion / 2 * w @ Sigma @ w - turnover_penalty * |w - w_current|_1

over long-only, fully invested weights, or dollar-neutral long-short
weights of gross exposure at most one, under a per-symbol cap. The expected
return ``mu`` is calibrated from a label's prediction (Grinold: ``ic *
sigma * z``) or is the prediction itself (``raw``), the covariance ``Sigma``
comes from a risk model, and both are on the span of the expected-return
label. This is the only quantlab module that imports cvxpy.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import pandas as pd
import xarray as xr

from quantlab.base.config import MeanVarianceConfig
from quantlab.base.portfolio import (
    CovarianceEstimate,
    FactorCovarianceEstimate,
    PortfolioConstructionError,
    PortfolioConstructor,
    PortfolioContext,
)
from quantlab.utils.ensemble import _cross_sectional_zscore

if TYPE_CHECKING:
    from quantlab.base.factor import Factor

_SOLVED = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


@dataclass(frozen=True)
class MeanVarianceInputs:
    """The problem one bar poses: the candidate symbols and their inputs.

    Attributes
    ----------
    symbols : np.ndarray
        The candidates: tradable, with a finite expected-return prediction
        and covered by the risk model; with ``candidate_top_k``, only the
        pool.
    expected_return : np.ndarray
        ``mu`` per candidate, on the expected-return label's span.
    estimate : CovarianceEstimate or FactorCovarianceEstimate
        The risk model's estimate over the candidates, on the same span.
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
    estimate: CovarianceEstimate | FactorCovarianceEstimate
    current_weights: np.ndarray

    @property
    def covariance(self) -> np.ndarray:
        """``Sigma`` over the candidates as a dense matrix.

        Examples
        --------
        >>> inputs.covariance.shape
        (4, 4)
        """
        return self.estimate.covariance


def _zscore(values: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score with ``ddof=1``; all 0.0 when it is undefined.

    ``_cross_sectional_zscore``, which also treats fewer than two values or
    a constant cross-section (a one-ulp residue in the standard deviation
    included) as undefined.
    """
    return np.nan_to_num(_cross_sectional_zscore(values), nan=0.0)


def _project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    """Return the nearest point to ``values`` with ``sum = 1`` and ``0 <= w <= cap``.

    The Euclidean projection is ``clip(values - tau, 0, cap)`` for the shift
    ``tau`` that makes it sum to one; it removes a solver's round-off
    without moving an exact solution. The sum falls as ``tau`` grows, so
    ``tau`` is bisected between ``low`` (every weight at the cap, a sum of
    ``n * cap >= 1``) and ``high`` (every weight at zero) until no float
    lies between the two. The ``high`` end is returned: its sum is at most
    one, so the gross exposure never exceeds one.
    """
    low, high = values.min() - cap, values.max()
    while True:
        tau = (low + high) / 2
        if not low < tau < high:
            break
        if np.clip(values - tau, 0.0, cap).sum() > 1:
            low = tau
        else:
            high = tau
    return np.clip(values - high, 0.0, cap)


def _clean_long_short(values: np.ndarray, cap: float) -> np.ndarray:
    """Return ``values`` made exactly dollar-neutral, capped and of gross at most one.

    It removes a solver's round-off: weights are clipped to the cap, the
    heavier side is scaled down to the lighter one, and the book is scaled
    down if its gross exposure still exceeds one. Scaling keeps the sign
    and the cap of every weight.
    """
    w = np.clip(values, -cap, cap)
    long, short = w[w > 0].sum(), -w[w < 0].sum()
    if long > short:
        w = np.where(w > 0, w * (short / long), w)
    elif short > long:
        w = np.where(w < 0, w * (long / short), w)
    gross = np.abs(w).sum()
    return w / gross if gross > 1 else w


def _risk_term(w: cp.Variable, estimate) -> cp.Expression:
    """``w' Sigma w``: low-rank when the estimate has a factor form, else dense.

    With ``Sigma = B F B' + diag(D)`` the term is ``|R B' w|^2 + sum(D
    w^2)``, ``R' R = F`` from ``F``'s eigen-decomposition (negative
    eigenvalues from round-off clipped to zero), so the dense ``[n, n]``
    matrix is never built.
    """
    form = estimate.factor_form()
    if form is None:
        return cp.quad_form(w, cp.psd_wrap(estimate.covariance))
    exposures, factor_covariance, specific = form
    eigenvalues, eigenvectors = np.linalg.eigh(factor_covariance)
    root = np.sqrt(np.clip(eigenvalues, 0.0, None))[:, None] * eigenvectors.T
    return cp.sum_squares((root @ exposures.T) @ w) + cp.sum(
        cp.multiply(np.clip(specific, 0.0, None), cp.square(w))
    )


class MeanVarianceOptimizer(PortfolioConstructor):
    """Mean-variance weights with a turnover penalty, long-only or long-short.

    On each rebalance bar the candidates are the symbols the context marks
    tradable, with a finite prediction of ``expected_return_label`` and
    covered by the risk model (enough return history); every other symbol
    gets 0.0. Over the candidates the optimiser solves

        maximise    w @ mu - risk_aversion / 2 * w @ Sigma @ w
                    - turnover_penalty * |w - w_current|_1

    subject to, with ``direction="long_only"``,

        sum(w) = 1,  0 <= w <= weight_cap

    and with ``direction="long_short"``,

        sum(w) = 0,  |w|_1 <= 1,  |w| <= weight_cap.

    The long-short gross exposure is a ceiling, not an equality: when the
    expected returns do not pay for the risk and the turnover, part of the
    book stays uninvested, down to no position at all.

    ``Sigma`` is the risk model's one-bar covariance times the span ``n`` of
    ``expected_return_label`` (variance is linear in time). When the risk
    model's estimate has a factor form (``B F B' + diag(D)``), the risk term
    is built as ``|F^(1/2) B' w|^2 + w' diag(D) w`` and the dense matrix is
    never formed. ``mu`` depends on ``calibration``: ``"grinold"`` (the
    default) gives ``mu = ic * sigma * z``, ``sigma`` the square root of
    ``Sigma``'s diagonal and ``z`` the candidates' cross-sectional z-score
    (``ddof=1``) of the prediction, so the prediction only ranks and any
    model's output can feed the optimiser; ``"raw"`` takes the prediction
    itself as ``mu``, which ``bind`` allows only when the predictor reports
    the label's scale as ``"raw"``. ``w_current`` is the context's current
    weights.

    With ``candidate_top_k`` set, only a pool is optimised: the
    ``candidate_top_k`` candidates with the largest ``mu`` (largest ``|mu|``
    long-short) plus every candidate currently held, so a held symbol that
    fell out of the top can still be closed at its turnover cost. The
    z-score is taken over every candidate before the pool is cut.

    The solver's solution is cleaned of its round-off: projected onto the
    feasible set long-only (the nearest weights summing to one within
    ``[0, weight_cap]``), clipped and rescaled to dollar neutrality and a
    gross exposure of at most one long-short.

    A bar without candidates, or long-only with fewer than ``1 /
    weight_cap``, is infeasible, and a bar the solver fails on or leaves
    unsolved raises ``PortfolioConstructionError``: the backtest holds the
    current position there and records the bar. So does a bar with a locked
    position (held, not tradable), which the optimiser does not yet price.

    ``lookback_bars`` and ``required_factors()`` are the risk model's.
    ``bind`` reads the span from the predictor's label, so a backtest binds
    the optimiser when it is built.

    Parameters
    ----------
    config : MeanVarianceConfig
        ``expected_return_label``, ``risk_model``, ``risk_aversion``,
        ``calibration``, ``ic``, ``turnover_penalty``, ``weight_cap``,
        ``direction`` and ``candidate_top_k``.

    Raises
    ------
    ValueError
        If ``direction`` or ``calibration`` is unknown, ``ic`` is missing
        for ``"grinold"``, ``weight_cap`` is not in ``(0, 1]``,
        ``risk_aversion`` or ``turnover_penalty`` is negative, or
        ``candidate_top_k`` is not a positive integer.

    Examples
    --------
    ``model`` predicts the 5-bar ``ret_5``; ``context`` is a bar of four
    symbols ``AAA``..``DDD``, all tradable and none held, with 60 bars of
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

    Long-short on the same bar the book is dollar-neutral, and its gross
    exposure (0.946) stays under the ceiling of one:

    >>> long_short = MeanVarianceOptimizer(MeanVarianceConfig(
    ...     expected_return_label="ret_5",
    ...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60)),
    ...     ic=0.05, risk_aversion=5.0, weight_cap=0.4, direction="long_short",
    ... ))
    >>> long_short.bind(model)
    >>> weights = long_short.construct(context)
    >>> weights.values.round(3)
    array([ 0.4  , -0.212, -0.261,  0.073])
    """

    config_cls = MeanVarianceConfig

    def __init__(self, config: MeanVarianceConfig):
        """Initialize the optimiser; see the class docstring for parameters."""
        super().__init__(config)
        if config.direction not in ("long_only", "long_short"):
            raise ValueError(
                f"direction must be 'long_only' or 'long_short', got {config.direction!r}"
            )
        if config.calibration not in ("grinold", "raw"):
            raise ValueError(
                f"calibration must be 'grinold' or 'raw', got {config.calibration!r}"
            )
        if config.calibration == "grinold" and config.ic is None:
            raise ValueError("calibration='grinold' needs an ic")
        if not 0 < config.weight_cap <= 1:
            raise ValueError(f"weight_cap must be in (0, 1], got {config.weight_cap}")
        for name in ("risk_aversion", "turnover_penalty"):
            if getattr(config, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(config, name)}")
        top_k = config.candidate_top_k
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1
        ):
            raise ValueError(
                f"candidate_top_k must be a positive integer or None, got {top_k!r}"
            )
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

    def required_factors(self) -> list["Factor"]:
        """The risk model's ``required_factors()``.

        Examples
        --------
        >>> optimizer.required_factors()
        []
        """
        return self.config.risk_model.required_factors()

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
            If the predictor does not predict ``expected_return_label``, the
            label has no ``span_bars()`` (it is not a ``Forward`` label), or
            ``calibration="raw"`` and the predictor does not report the
            label's scale as ``"raw"``.

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
        if self.config.calibration == "raw":
            scale = dict(predictor.label_scales).get(name)
            if scale != "raw":
                raise ValueError(
                    f"calibration='raw' reads the prediction of {name!r} as a return, "
                    f"but the predictor reports its scale as {scale!r}, not 'raw' "
                    f"(a model fitted on a transformed target, or a label an ensemble "
                    f"averages); use calibration='grinold'"
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
        config = self.config
        symbols = context.symbols
        prediction = np.asarray(
            context.predictions[config.expected_return_label].sel(symbol=symbols).values,
            dtype=np.float64,
        )
        estimate = config.risk_model.estimate(context).scaled(self._span)
        position = pd.Index(estimate.symbols).get_indexer(symbols)
        if bool(context.locked.any()):
            raise PortfolioConstructionError(
                f"held symbols that are not tradable at the bar "
                f"{context.symbols[context.locked.values].tolist()[:5]}; the optimiser "
                f"does not yet price locked positions"
            )
        candidate = (
            np.asarray(context.tradable.values, dtype=bool)
            & np.isfinite(prediction)
            & (position >= 0)
        )
        index = np.flatnonzero(candidate)
        current = np.asarray(
            context.current_weights.sel(symbol=symbols).values, dtype=np.float64
        )[index]
        if config.calibration == "raw":
            expected = prediction[index]
        else:
            sigma = np.sqrt(estimate.subset(position[index]).variance)
            expected = config.ic * sigma * _zscore(prediction[index])

        if config.candidate_top_k is not None and config.candidate_top_k < len(index):
            strength = expected if config.direction == "long_only" else np.abs(expected)
            # Stable, so ties keep the context's symbol order.
            top = np.argsort(-strength, kind="stable")[: config.candidate_top_k]
            pool = np.zeros(len(index), dtype=bool)
            pool[top] = True
            pool |= current != 0
            index, expected, current = index[pool], expected[pool], current[pool]
        return MeanVarianceInputs(
            symbols=symbols[index],
            expected_return=expected,
            estimate=estimate.subset(position[index]),
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
            One finite weight per symbol of ``context.symbols``: summing to
            one long-only; summing to zero with gross exposure at most one
            long-short.

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
        long_only = config.direction == "long_only"
        inputs = self.problem_inputs(context)
        n = len(inputs.symbols)
        if n == 0:
            raise PortfolioConstructionError("infeasible: no candidate symbol")
        if long_only and n * config.weight_cap < 1 - 1e-12:
            raise PortfolioConstructionError(
                f"infeasible: {n} candidate symbol(s) cannot hold a fully invested "
                f"book under weight_cap={config.weight_cap}"
            )
        w = cp.Variable(n)
        objective = (
            inputs.expected_return @ w
            - config.risk_aversion / 2 * _risk_term(w, inputs.estimate)
            - config.turnover_penalty * cp.norm1(w - inputs.current_weights)
        )
        if long_only:
            constraints = [cp.sum(w) == 1, w >= 0, w <= config.weight_cap]
        else:
            constraints = [cp.sum(w) == 0, cp.norm1(w) <= 1, cp.abs(w) <= config.weight_cap]
        problem = cp.Problem(cp.Maximize(objective), constraints)
        try:
            problem.solve()
        except (cp.error.SolverError, ValueError, ArithmeticError) as exc:
            # cvxpy raises ValueError for non-finite problem data.
            raise PortfolioConstructionError(f"the solver failed: {exc}") from exc
        if problem.status not in _SOLVED or w.value is None:
            raise PortfolioConstructionError(f"no solution: status {problem.status!r}")
        values = np.asarray(w.value, dtype=np.float64)
        solution = (
            _project_capped_simplex(values, config.weight_cap)
            if long_only
            else _clean_long_short(values, config.weight_cap)
        )

        row = pd.Series(0.0, index=context.symbols)
        row[inputs.symbols] = solution
        return xr.DataArray(row.values, dims="symbol", coords={"symbol": context.symbols})
