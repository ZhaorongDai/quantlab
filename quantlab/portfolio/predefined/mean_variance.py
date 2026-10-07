"""Markowitz mean-variance weights with a turnover penalty, solved with cvxpy.

``MeanVarianceOptimizer`` maximises, on every rebalance bar,

    w @ mu - risk_aversion / 2 * w @ Sigma @ w - turnover_penalty * |w - w_current|_1

over long-only, fully invested weights, or dollar-neutral long-short
weights of gross exposure at most one, under a per-symbol cap. The expected
return ``mu`` is calibrated from a label's prediction (Grinold: ``ic *
sigma * z``) or is the prediction itself (``raw``), the covariance ``Sigma``
comes from a covariance estimator, its volatilities optionally from a volatility
label's prediction, and all are on the span of the expected-return label.
This is the only quantlab module that imports cvxpy.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import pandas as pd
import xarray as xr

from quantlab.portfolio.config import MeanVarianceConfig
from quantlab.portfolio.base import (
    CovarianceEstimate,
    FactorCovarianceEstimate,
    PortfolioConstructionError,
    PortfolioConstructor,
    PortfolioContext,
)
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.utils.cross_section import cross_sectional_zscore

if TYPE_CHECKING:
    from quantlab.factor.base import Factor
    from quantlab.risk.base import FactorRiskModel

_SOLVED = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


@dataclass(frozen=True)
class MeanVarianceInputs:
    """The problem one bar poses: the candidate symbols, the locked ones and their inputs.

    Attributes
    ----------
    symbols : np.ndarray
        The candidates the optimiser sets: tradable, not locked, covered by
        the covariance estimator, and either with a finite expected-return prediction
        or held; with ``candidate_top_k``, only the pool.
    expected_return : np.ndarray
        ``mu`` per candidate, on the expected-return label's span; 0.0 for a
        held candidate without a prediction.
    estimate : CovarianceEstimate or FactorCovarianceEstimate
        The covariance estimator's estimate, on the same span, over the candidates
        followed by the locked symbols it covers.
    current_weights : np.ndarray
        The weights currently held on the candidates.
    locked_symbols : np.ndarray
        The locked positions: held and not tradable at the bar.
    locked_weights : np.ndarray
        Their current weights, which the solution keeps.
    risk_locked_weights : np.ndarray
        The locked weights the estimate covers, in its order after the
        candidates.
    closed_without_risk : np.ndarray
        Held, tradable symbols the covariance estimator does not cover, closed.
    exposures : np.ndarray
        ``[k, candidates]``: each bounded exposure per candidate;
        ``[0, candidates]`` without bounds.
    locked_exposures : np.ndarray
        ``[k]``: each bounded exposure of the locked positions, ``sum w * x``.
    exposure_bounds : np.ndarray
        ``[k, 2]``: each bounded exposure's ``(lower, upper)``, in the rows'
        order.
    closed_without_exposure : np.ndarray
        Held, tradable symbols lacking a bounded exposure, closed.

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
    locked_symbols: np.ndarray = field(default_factory=lambda: np.array([]))
    locked_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    risk_locked_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    closed_without_risk: np.ndarray = field(default_factory=lambda: np.array([]))
    exposures: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    locked_exposures: np.ndarray = field(default_factory=lambda: np.zeros(0))
    exposure_bounds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    closed_without_exposure: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def covariance(self) -> np.ndarray:
        """``Sigma`` as a dense matrix, over the candidates then the covered locked symbols.

        Examples
        --------
        >>> inputs.covariance.shape
        (4, 4)
        """
        return self.estimate.covariance


def _exposure(context: PortfolioContext, name: str) -> xr.DataArray:
    """Return a bounded exposure at the bar: a declared factor's output, else a risk model's exposure."""
    if context.factors is not None and name in context.factors:
        return context.factors[name]
    if context.risk_exposures is not None and name in context.risk_exposures:
        return context.risk_exposures[name]
    raise ValueError(
        f"the context holds no exposure {name!r} in its factors or risk exposures"
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score with ``ddof=1``; all 0.0 when it is undefined.

    ``cross_sectional_zscore``, which also treats fewer than two values or
    a constant cross-section (a one-ulp residue in the standard deviation
    included) as undefined.
    """
    return np.nan_to_num(cross_sectional_zscore(values), nan=0.0)


def _project_capped_simplex(values: np.ndarray, cap: float, total: float = 1.0) -> np.ndarray:
    """Return the nearest point to ``values`` with ``sum = total`` and ``0 <= w <= cap``.

    The Euclidean projection is ``clip(values - tau, 0, cap)`` for the shift
    ``tau`` that makes it sum to ``total``; it removes a solver's round-off
    without moving an exact solution. The sum falls as ``tau`` grows, so
    ``tau`` is bisected between ``low`` (every weight at the cap, a sum of
    ``n * cap >= total``) and ``high`` (every weight at zero) until no float
    lies between the two. The ``high`` end is returned: its sum is at most
    ``total``, so the gross exposure never exceeds it.
    """
    low, high = values.min() - cap, values.max()
    while True:
        tau = (low + high) / 2
        if not low < tau < high:
            break
        if np.clip(values - tau, 0.0, cap).sum() > total:
            low = tau
        else:
            high = tau
    return np.clip(values - high, 0.0, cap)


def _clean_long_short(
    values: np.ndarray, cap: float, net: float = 0.0, gross: float = 1.0
) -> np.ndarray:
    """Return ``values`` capped, with ``sum = net`` and gross exposure at most ``gross``.

    It removes a solver's round-off: weights are clipped to the cap, the
    side that makes the sum miss ``net`` is scaled down, and when the gross
    exposure still exceeds ``gross`` both sides shrink by the same amount,
    which keeps the sum; if a side is too small for that, the whole book is
    scaled to the ceiling. Shrinking keeps the sign and the cap of every
    weight.
    """
    w = np.clip(values, -cap, cap)
    long, short = w[w > 0].sum(), -w[w < 0].sum()
    if long - short > net and short + net >= 0 and long > 0:
        w = np.where(w > 0, w * ((short + net) / long), w)
    elif long - short < net and long - net >= 0 and short > 0:
        w = np.where(w < 0, w * ((long - net) / short), w)
    long, short = w[w > 0].sum(), -w[w < 0].sum()
    cut = (long + short - gross) / 2
    if cut > 0 and long >= cut and short >= cut:
        w = np.where(w > 0, w * ((long - cut) / long), w * ((short - cut) / short))
    total = np.abs(w).sum()
    if total > gross:
        # A near one-sided book: the ceiling wins over the last ulp of the sum.
        w = w * (gross / total)
    return w


def _risk_term(w: cp.Expression, estimate) -> cp.Expression:
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
    tradable, covered by the covariance estimator (enough recent return history), and
    either with a finite prediction of ``expected_return_label`` or held; a
    held candidate without a prediction has an expected return of 0.0, so
    its turnover cost decides whether it is closed. A locked position (held,
    not tradable) keeps its current weight; a held, tradable symbol the risk
    model does not cover is closed, and the row reports it in
    ``attrs["events"]["closed_without_risk"]``; every other symbol gets 0.0.
    Over the candidates the optimiser solves

        maximise    w @ mu - risk_aversion / 2 * w @ Sigma @ w
                    - turnover_penalty * |w - w_current|_1

    subject to, with ``direction="long_only"``,

        sum(w) = 1,  0 <= w <= weight_cap

    and with ``direction="long_short"``,

        sum(w) = 0,  |w|_1 <= 1,  |w| <= weight_cap.

    ``w`` in the risk term also holds the locked weights, so their
    covariance with the candidates counts, while the turnover and the cap
    apply to the candidates only; the budget is what the locked positions
    leave: the candidates sum to ``1 - L`` long-only (nothing when ``L >=
    1``) and to ``-L`` long-short with ``|w|_1 <= 1 - |L|_1``, ``L`` the
    locked weights. A locked position the covariance estimator does not cover adds no
    variance. The long-short gross exposure is a ceiling, not an
    equality: when the expected returns do not pay for the risk and the
    turnover, part of the book stays uninvested, down to no position at all.

    ``Sigma`` is the covariance estimator's one-bar covariance times the span ``n`` of
    ``expected_return_label`` (variance is linear in time). When the risk
    model's estimate has a factor form (``B F B' + diag(D)``), the risk term
    is built as ``|F^(1/2) B' w|^2 + w' diag(D) w`` and the dense matrix is
    never formed. ``mu`` depends on ``calibration``: ``"grinold"`` (the
    default) gives ``mu = ic * sigma * z``, ``sigma`` the square root of
    ``Sigma``'s diagonal and ``z`` the candidates' cross-sectional z-score
    (``ddof=1``) of the prediction over the predicted candidates, so the
    prediction only ranks and any
    model's output can feed the optimiser; ``"raw"`` takes the prediction
    itself as ``mu``, which ``bind`` allows only when the label's spec
    reports its scale as ``"raw"``. ``w_current`` is the context's current
    weights.

    With ``volatility_label`` set (a ``Volatility`` label, say), the
    volatilities come from a model: the label's prediction at the bar,
    divided by ``sqrt(n)``, is handed to the covariance estimator as the one-bar
    volatilities, so ``Sigma`` is the predicted volatilities around the risk
    model's historical correlations, its diagonal the squared predictions,
    and the Grinold ``sigma`` is the prediction itself. A symbol without a
    finite positive volatility prediction has no risk estimate, like one
    without enough history. ``bind`` requires the label to have the span of
    ``expected_return_label`` and a ``"raw"`` scale.

    With ``candidate_top_k`` set, only a pool is optimised: the
    ``candidate_top_k`` candidates with the largest ``mu`` (largest ``|mu|``
    long-short) plus every candidate currently held, so a held symbol that
    fell out of the top can still be closed at its turnover cost. The
    z-score is taken over every candidate before the pool is cut.

    The solver's solution is cleaned of its round-off: projected onto the
    feasible set long-only (the nearest weights summing to one within
    ``[0, weight_cap]``), clipped and rescaled to dollar neutrality and a
    gross exposure of at most one long-short.

    A bar whose candidates cannot hold their budget under ``weight_cap``, a
    long-short bar whose locked positions exceed a gross exposure of one,
    a bar the solver fails on or leaves unsolved (bounds the candidates
    cannot reach among them), and a bar with a locked position lacking a
    bounded exposure raise ``PortfolioConstructionError``: the backtest holds
    the current position there and records the bar.

    ``lookback_bars``, ``history_bars`` and ``required_risk_model()`` are the
    covariance estimator's; ``required_factors()`` is the
    ``exposure_factors``, whose outputs must not repeat a name.
    ``exposure_bounds`` may bound any of their outputs or any of the risk
    model's ``exposure_names`` (read from ``context.risk_exposures``); a
    bounded name must come from one of the two, not both.
    ``bind`` reads the span from the label's ``LabelSpec``, so a backtest
    binds the optimiser when it is built.

    Parameters
    ----------
    config : MeanVarianceConfig
        ``expected_return_label``, ``covariance``, ``risk_aversion``,
        ``calibration``, ``ic``, ``turnover_penalty``, ``weight_cap``,
        ``direction``, ``candidate_top_k`` and ``volatility_label``.

    Raises
    ------
    ValueError
        If ``direction`` or ``calibration`` is unknown, ``ic`` is missing
        for ``"grinold"``, ``weight_cap`` is not in ``(0, 1]``,
        ``risk_aversion`` or ``turnover_penalty`` is negative, or
        ``candidate_top_k`` is not a positive integer.

    Examples
    --------
    ``specs`` holds the spec of a 5-bar ``ret_5`` label
    (``LabelSpec("ret_5", "raw", 1, 5)``); ``context`` is a bar of four
    symbols ``AAA``..``DDD``, all tradable and none held, with 60 bars of
    one-bar returns of volatility 1%, 1.5%, 2% and 2.5% and ``ret_5``
    predictions 0.8, -0.1, -0.3 and 0.2. With a risk aversion of 5 on so small
    an expected return the book leans toward the low-volatility symbols:

    >>> from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig
    >>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
    >>> optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
    ...     expected_return_label="ret_5",
    ...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
    ...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
    ... ))
    >>> optimizer.lookback_bars
    60
    >>> optimizer.bind(specs)
    >>> weights = optimizer.construct(context)
    >>> float(weights.sum().round(6)), bool((weights >= 0).all()), bool((weights <= 0.4).all())
    (1.0, True, True)

    Long-short on the same bar the book is dollar-neutral, and its gross
    exposure (0.946) stays under the ceiling of one:

    >>> long_short = MeanVarianceOptimizer(MeanVarianceConfig(
    ...     expected_return_label="ret_5",
    ...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
    ...     ic=0.05, risk_aversion=5.0, weight_cap=0.4, direction="long_short",
    ... ))
    >>> long_short.bind(specs)
    >>> weights = long_short.construct(context)
    >>> weights.values.round(3)
    array([ 0.4  , -0.212, -0.261,  0.073])

    With ``DDD`` held at 0.3 and halted, it is a locked position: it keeps
    its weight and the candidates share the remaining 0.7:

    >>> import dataclasses
    >>> halted = dataclasses.replace(
    ...     context,
    ...     tradable=context.tradable.copy(data=[True, True, True, False]),
    ...     current_weights=context.current_weights.copy(data=[0.0, 0.0, 0.0, 0.3]),
    ... )
    >>> optimizer.construct(halted).values.round(3)
    array([0.4, 0.3, 0. , 0.3])
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
        declared = [
            name for factor in config.exposure_factors for name in factor.get_factor_names()
        ]
        repeated = sorted({name for name in declared if declared.count(name) > 1})
        if repeated:
            raise ValueError(
                f"the declared factors produce {repeated} more than once; "
                f"context.factors holds one variable per name"
            )
        risk_model = self.required_risk_model() if config.exposure_bounds else None
        risk_names = set() if risk_model is None else set(risk_model.exposure_names)
        for name, (lower, upper) in config.exposure_bounds.items():
            if name in declared and name in risk_names:
                raise ValueError(
                    f"exposure_bounds names {name!r}, which is both an output of the "
                    f"declared factors and an exposure of the risk model; name it once"
                )
            if name not in declared and name not in risk_names:
                raise ValueError(
                    f"exposure_bounds names {name!r}, which is not an output of the "
                    f"declared factors ({sorted(declared)}) or an exposure of the risk "
                    f"model ({sorted(risk_names)})"
                )
            if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
                raise ValueError(
                    f"exposure_bounds[{name!r}] must be finite with its lower bound at most "
                    f"its upper one, got ({lower}, {upper})"
                )
        self._span: int | None = None

    @property
    def lookback_bars(self) -> int:
        """The covariance estimator's ``lookback_bars``.

        Examples
        --------
        >>> optimizer.lookback_bars
        60
        """
        return self.config.covariance.lookback_bars

    @property
    def history_bars(self) -> int:
        """The covariance estimator's ``history_bars``.

        Examples
        --------
        >>> optimizer.history_bars  # 60 + 1 + 5
        66
        """
        return self.config.covariance.history_bars

    def required_factors(self) -> list["Factor"]:
        """The ``exposure_factors``.

        Examples
        --------
        >>> optimizer.required_factors()
        []
        """
        return list(self.config.exposure_factors)

    def required_risk_model(self) -> "FactorRiskModel | None":
        """The covariance estimator's ``required_risk_model()``.

        Examples
        --------
        >>> optimizer.required_risk_model() is None  # Ledoit-Wolf
        True
        """
        return self.config.covariance.required_risk_model()

    @property
    def span(self) -> int | None:
        """The expected-return label's span in bars, once ``bind`` has read it.

        Examples
        --------
        >>> optimizer.span
        5
        """
        return self._span

    def bind(self, labels: Sequence[LabelSpec]) -> None:
        """Check the specs hold the optimiser's labels and read the span.

        Parameters
        ----------
        labels : Sequence[LabelSpec]
            The specs of the predicted labels.

        Raises
        ------
        ValueError
            If ``expected_return_label`` or ``volatility_label`` is not one
            of the specs, either has no span (it is not a ``Forward``
            label), their spans differ, ``volatility_label``'s scale is not
            ``"raw"``, or ``calibration="raw"`` and
            ``expected_return_label``'s scale is not ``"raw"``.

        Examples
        --------
        >>> from quantlab.runs.prediction_panel import LabelSpec
        >>> optimizer.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
        >>> optimizer.span
        5
        """
        config = self.config
        specs = {spec.name: spec for spec in labels}
        name = config.expected_return_label
        span = self._label_span(specs, "expected_return_label", name)
        scale = specs[name].scale
        if config.calibration == "raw" and scale != "raw":
            raise ValueError(
                f"calibration='raw' reads the prediction of {name!r} as a return, "
                f"but its label spec reports its scale as {scale!r}, not 'raw' "
                f"(a model fitted on a transformed target, or a label an ensemble "
                f"averages); use calibration='grinold'"
            )
        volatility = config.volatility_label
        if volatility is not None:
            volatility_span = self._label_span(specs, "volatility_label", volatility)
            if volatility_span != span:
                raise ValueError(
                    f"the span of volatility_label {volatility!r} is {volatility_span} "
                    f"bars, but the span of expected_return_label {name!r} is {span}; "
                    f"the two must match"
                )
            volatility_scale = specs[volatility].scale
            if volatility_scale != "raw":
                raise ValueError(
                    f"volatility_label {volatility!r} is read as a volatility, but the "
                    f"label spec reports its scale as {volatility_scale!r}, not "
                    f"'raw' (a model fitted on a transformed target, or a label an "
                    f"ensemble averages)"
                )
        self._span = span

    @staticmethod
    def _label_span(specs: dict[str, LabelSpec], config_field: str, name: str) -> int:
        """Return the span of the spec ``name``, which ``config_field`` names."""
        if name not in specs:
            raise ValueError(
                f"{config_field} {name!r} is not one of the predicted labels {list(specs)}"
            )
        span = specs[name].span
        if span is None:
            raise ValueError(
                f"{config_field} {name!r} has no span; it needs a Forward label"
            )
        return int(span)

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
            If the optimiser has not been bound to label specs.

        Examples
        --------
        >>> inputs = optimizer.problem_inputs(context)
        >>> inputs.expected_return.round(4)
        array([ 0.0017, -0.0008, -0.0022,  0.0003])
        """
        if self._span is None:
            raise RuntimeError(
                "MeanVarianceOptimizer is not bound to label specs; call bind() "
                "first (a backtest does when it is built)"
            )
        config = self.config
        symbols = context.symbols
        prediction = np.asarray(
            context.predictions[config.expected_return_label].sel(symbol=symbols).values,
            dtype=np.float64,
        )
        current_all = np.asarray(
            context.current_weights.sel(symbol=symbols).values, dtype=np.float64
        )
        tradable = np.asarray(context.tradable.values, dtype=bool)
        locked = np.asarray(context.locked.values, dtype=bool)
        held = current_all != 0
        volatility = None
        if config.volatility_label is not None:
            # The label is on the span; the covariance estimator takes one-bar volatilities.
            volatility = context.predictions[config.volatility_label].sel(
                symbol=symbols
            ) / np.sqrt(self._span)
        estimate = config.covariance.estimate(context, volatility).scaled(self._span)
        position = pd.Index(estimate.symbols).get_indexer(symbols)
        covered = position >= 0
        bounded = list(config.exposure_bounds)
        exposure_all = (
            np.stack([
                np.asarray(_exposure(context, name).sel(symbol=symbols).values, dtype=np.float64)
                for name in bounded
            ])
            if bounded
            else np.zeros((0, len(symbols)))
        )
        exposed = np.isfinite(exposure_all).all(axis=0)
        if (locked & ~exposed).any():
            raise PortfolioConstructionError(
                f"the locked positions {symbols[locked & ~exposed].tolist()[:5]} lack a "
                f"bounded exposure ({bounded}), so the book's exposure is unknown"
            )
        free = tradable & ~locked & covered & exposed & (np.isfinite(prediction) | held)
        index = np.flatnonzero(free)
        current = current_all[index]
        predicted = np.isfinite(prediction[index])
        expected = np.zeros(len(index))
        if config.calibration == "raw":
            expected[predicted] = prediction[index][predicted]
        elif predicted.any():
            sigma = np.sqrt(estimate.subset(position[index]).variance)
            expected[predicted] = (
                config.ic * sigma[predicted] * _zscore(prediction[index][predicted])
            )

        if config.candidate_top_k is not None and config.candidate_top_k < len(index):
            strength = expected if config.direction == "long_only" else np.abs(expected)
            # Stable, so ties keep the context's symbol order.
            top = np.argsort(-strength, kind="stable")[: config.candidate_top_k]
            pool = np.zeros(len(index), dtype=bool)
            pool[top] = True
            pool |= current != 0
            index, expected, current = index[pool], expected[pool], current[pool]
        locked_index = np.flatnonzero(locked)
        risk_locked = locked_index[covered[locked_index]]
        return MeanVarianceInputs(
            symbols=symbols[index],
            expected_return=expected,
            estimate=estimate.subset(np.concatenate([position[index], position[risk_locked]])),
            current_weights=current,
            locked_symbols=symbols[locked_index],
            locked_weights=current_all[locked_index],
            risk_locked_weights=current_all[risk_locked],
            closed_without_risk=symbols[tradable & ~locked & held & ~covered],
            exposures=exposure_all[:, index],
            locked_exposures=exposure_all[:, locked_index] @ current_all[locked_index],
            exposure_bounds=np.array(
                [config.exposure_bounds[name] for name in bounded], dtype=np.float64
            ).reshape(len(bounded), 2),
            closed_without_exposure=symbols[tradable & ~locked & held & covered & ~exposed],
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
            One finite weight per symbol of ``context.symbols``, locked
            positions at their current weights: summing to one long-only
            (to the locked weights' sum when that is already one or more);
            summing to zero with gross exposure at most one long-short. When
            a held symbol was closed for want of a risk estimate,
            ``attrs["events"]["closed_without_risk"]`` lists it.

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
        locked_sum = float(inputs.locked_weights.sum())
        if long_only:
            total = 1.0 - locked_sum
            if total <= 1e-12:
                solution = np.zeros(n)
            elif n == 0 or n * config.weight_cap < total - 1e-12:
                raise PortfolioConstructionError(
                    f"infeasible: {n} candidate symbol(s) cannot hold {total:.4g} of the "
                    f"book under weight_cap={config.weight_cap}"
                )
            else:
                solution = self._solve(inputs, total)
        else:
            gross = 1.0 - float(np.abs(inputs.locked_weights).sum())
            if gross < -1e-12:
                raise PortfolioConstructionError(
                    f"the locked positions {inputs.locked_symbols.tolist()[:5]} have a "
                    f"gross exposure of {1.0 - gross:.4g}, above one"
                )
            reach = min(max(gross, 0.0), n * config.weight_cap)
            if n == 0 or abs(locked_sum) > reach + 1e-12:
                raise PortfolioConstructionError(
                    f"infeasible: {n} candidate symbol(s) cannot offset a locked net "
                    f"exposure of {locked_sum:.4g} within a gross of {max(gross, 0.0):.4g} "
                    f"and weight_cap={config.weight_cap}"
                )
            solution = self._solve(inputs, -locked_sum, max(gross, 0.0))

        row = pd.Series(0.0, index=context.symbols)
        row[inputs.symbols] = solution
        row[inputs.locked_symbols] = inputs.locked_weights
        weights = xr.DataArray(row.values, dims="symbol", coords={"symbol": context.symbols})
        events = {}
        if len(inputs.closed_without_risk):
            events["closed_without_risk"] = [str(s) for s in inputs.closed_without_risk]
        if len(inputs.closed_without_exposure):
            events["closed_without_exposure"] = [str(s) for s in inputs.closed_without_exposure]
        if events:
            weights.attrs["events"] = events
        return weights

    def _solve(self, inputs: MeanVarianceInputs, total: float, gross: float = 1.0) -> np.ndarray:
        """Solve for the candidates' weights: summing to ``total``, within ``gross`` long-short."""
        config = self.config
        n = len(inputs.symbols)
        w = cp.Variable(n)
        held = (
            cp.hstack([w, inputs.risk_locked_weights])
            if len(inputs.risk_locked_weights)
            else w
        )
        objective = inputs.expected_return @ w - config.risk_aversion / 2 * _risk_term(
            held, inputs.estimate
        )
        if config.turnover_penalty:
            # Added only for a non-zero penalty: even multiplied by 0 the term
            # changes the solver's path, and with it the weights' last digits.
            objective = objective - config.turnover_penalty * cp.norm1(
                w - inputs.current_weights
            )
        if config.direction == "long_only":
            constraints = [cp.sum(w) == total, w >= 0, w <= config.weight_cap]
        else:
            constraints = [
                cp.sum(w) == total,
                cp.norm1(w) <= gross,
                cp.abs(w) <= config.weight_cap,
            ]
        for (lower, upper), row, locked in zip(
            inputs.exposure_bounds, inputs.exposures, inputs.locked_exposures
        ):
            exposure = row @ w + locked
            constraints += [exposure >= lower, exposure <= upper]
        problem = cp.Problem(cp.Maximize(objective), constraints)
        try:
            problem.solve()
        except (cp.error.SolverError, ValueError, ArithmeticError) as exc:
            # cvxpy raises ValueError for non-finite problem data.
            raise PortfolioConstructionError(f"the solver failed: {exc}") from exc
        if problem.status not in _SOLVED or w.value is None:
            raise PortfolioConstructionError(f"no solution: status {problem.status!r}")
        values = np.asarray(w.value, dtype=np.float64)
        if config.direction == "long_only":
            return _project_capped_simplex(values, config.weight_cap, total)
        return _clean_long_short(values, config.weight_cap, total, gross)
