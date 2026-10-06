"""The configs of the portfolio layer.

A portfolio construction rule or risk model is constructed from one of these and
exposes it as ``self.config``. They are frozen; a rule's risk model is declared with
``component()`` and written as its own config.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig

if TYPE_CHECKING:
    from quantlab.portfolio.base import RiskModel
    from quantlab.risk.base import FactorRiskModel


@dataclass(frozen=True)
class TopNConfig(FrozenConfig):
    """Config of ``TopNConstructor``: equal-weight top-n selection.

    Examples
    --------
    >>> cfg = TopNConfig(direction="long_short", top_n=20)
    >>> cfg.score_label is None
    True
    """

    #: ``"long_only"`` holds the top ``top_n`` names; ``"long_short"`` also
    #: shorts the bottom ``top_n``, with the two books disjoint.
    direction: Literal["long_only", "long_short"]
    #: Number of names selected per book at every rebalance.
    top_n: int
    #: The label whose prediction ranks the symbols. ``None`` selects the
    #: predictor's first label.
    score_label: str | None = None


@dataclass(frozen=True)
class LedoitWolfConfig(FrozenConfig):
    """Config of ``LedoitWolfRiskModel``: a shrunk sample covariance of trailing returns.

    Examples
    --------
    >>> cfg = LedoitWolfConfig(lookback_bars=252)
    >>> cfg.lookback_bars, cfg.max_stale_bars
    (252, 5)
    """

    #: Bars of trailing one-bar returns the covariance is estimated from;
    #: at least 2. A symbol needs a finite return on every one of them.
    lookback_bars: int
    #: Largest staleness (bars since the symbol's last real price) a symbol
    #: may have at the bar and still be covered; a halt no longer than this
    #: stays in the estimate, flat returns and then its gap.
    max_stale_bars: int = 5


@dataclass(frozen=True)
class FactorRiskReaderConfig(FrozenConfig):
    """Config of ``FactorRiskReader``: the factor risk model whose stores it reads.

    Examples
    --------
    With ``use4`` a ``Use4RiskModel`` whose estimate store is built:

    >>> cfg = FactorRiskReaderConfig(risk_model=use4)
    >>> cfg.risk_model is use4
    True
    """

    #: The factor risk model; its estimate store must cover the backtest.
    risk_model: "FactorRiskModel" = component()


@dataclass(frozen=True, kw_only=True)
class MeanVarianceConfig(FrozenConfig):
    """Config of ``MeanVarianceOptimizer``: Markowitz weights with a turnover penalty.

    The optimiser maximises ``w @ mu - risk_aversion / 2 * w @ Sigma @ w -
    turnover_penalty * |w - w_current|_1``, with ``mu`` and ``Sigma`` on the
    span of ``expected_return_label``.

    Examples
    --------
    >>> cfg = MeanVarianceConfig(
    ...     expected_return_label="ret_5",
    ...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=252)),
    ...     ic=0.05, risk_aversion=10.0, weight_cap=0.05,
    ... )
    >>> cfg.direction, cfg.calibration, cfg.turnover_penalty, cfg.candidate_top_k
    ('long_only', 'grinold', 0.0, None)
    >>> cfg.volatility_label is None
    True
    """

    #: The label whose prediction gives the expected return; its span sets
    #: the horizon of the expected return and the covariance.
    expected_return_label: str
    #: The risk model estimating the covariance of one-bar returns.
    risk_model: "RiskModel" = component()
    #: Risk aversion ``lambda`` of the variance penalty.
    risk_aversion: float
    #: How the prediction becomes the expected return ``mu``.
    #: ``"grinold"``: ``mu = ic * sigma * z``, ``z`` the prediction's
    #: cross-sectional z-score, so any score will do. ``"raw"``: ``mu`` is
    #: the prediction itself, which must be in the label's own units (the
    #: predictor reports the label as ``"raw"`` in ``label_scales``).
    calibration: Literal["grinold", "raw"] = "grinold"
    #: Information coefficient of the Grinold calibration, for example a CV
    #: run's mean IC; required by ``"grinold"``, unused by ``"raw"``.
    ic: float | None = None
    #: Penalty ``kappa`` per unit of one-way turnover against the current
    #: weights; 0 trades freely.
    turnover_penalty: float = 0.0
    #: Largest absolute weight of one symbol.
    weight_cap: float = 1.0
    #: ``"long_only"``: fully invested, non-negative weights (``sum(w) =
    #: 1``). ``"long_short"``: dollar-neutral (``sum(w) = 0``) with gross
    #: exposure ``|w|_1 <= 1``, a ceiling, not an equality: the optimiser
    #: may leave part of the book uninvested.
    direction: Literal["long_only", "long_short"] = "long_only"
    #: Optimise only over the ``candidate_top_k`` symbols with the largest
    #: expected return (largest absolute one for ``"long_short"``) plus
    #: every symbol currently held; the rest get 0.0. ``None`` optimises
    #: over every tradable symbol.
    candidate_top_k: int | None = None
    #: The label whose prediction gives each symbol's volatility over the
    #: span, such as ``Volatility``; it must have the expected-return
    #: label's span and a ``"raw"`` scale. The covariance is then these
    #: volatilities around the risk model's correlations, and the Grinold
    #: ``sigma`` is the prediction. ``None`` keeps the risk model's own
    #: (historical) volatilities.
    volatility_label: str | None = None
