"""The root classes of portfolio construction: the rule from one bar's scores to weights.

A *portfolio construction* rule turns the scores of one bar, and the weights
currently held, into the weights to hold after that bar (ADR 0012). Every
rule derives from ``PortfolioConstructor``, whose one abstract method,
``construct(context)``, is handed a ``PortfolioContext`` holding only what is
known at that bar. ``decide`` runs ``construct``, checks the row and holds
a bar the rule cannot solve. A rule only decides: its contexts are
assembled by ``quantlab.portfolio.decision_inputs.DecisionInputs``, for the
vectorised backtest (a whole panel, the holdings replayed between
rebalances) and for an event-driven executor (one bar, with the holdings it
really has).

The output follows the weights contract (D-03): on a rebalance bar every
symbol gets a finite weight, an unselected one exactly 0.0, with gross
exposure at most one; an all-NaN row means "hold the current position". A
*locked position*, a symbol held but not tradable at the bar, keeps its
current weight, and a symbol neither tradable nor held gets 0.0 (ADR 0014);
the loop checks both. A rule holds no state between bars. A bar the rule
cannot solve raises ``PortfolioConstructionError``, and the loop holds it
instead.

A *risk model* (``RiskModel``) estimates the covariance of one-bar returns
at a bar, as a ``CovarianceEstimate``; a rule that prices risk, such as a
mean-variance optimiser, holds one. The interface is shaped for a factor
risk model, which none of the shipped ones is: its estimate is a
``FactorCovarianceEstimate`` whose ``factor_form()`` lets an optimiser build
a low-rank risk term, and it declares the ``Factor`` panels it reads (its
exposures) through ``required_factors()``, which the backtest reads and
slices into each bar's context.

Shipped rules and risk models live in ``quantlab/portfolio/predefined``; this
module imports no solver.
"""

import dataclasses
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.core.component import Component
from quantlab.runs.prediction_panel import LabelSpec

if TYPE_CHECKING:
    from quantlab.base.factor import Factor

class PortfolioConstructionError(RuntimeError):
    """A rule could not decide a bar: the optimisation failed, was infeasible or had no solution.

    ``PortfolioConstructor.decide`` holds such a bar (all-NaN weights,
    the message as the decision's ``failure``) and logs a warning;
    ``DecisionInputs.weights`` lists the bar in the result's
    ``attrs["failed_bars"]``.

    Examples
    --------
    >>> raise PortfolioConstructionError("infeasible: 3 symbols under a 0.2 cap")
    Traceback (most recent call last):
    quantlab.base.portfolio.PortfolioConstructionError: infeasible: 3 symbols under a 0.2 cap
    """


@dataclass(frozen=True)
class PortfolioContext:
    """Everything a portfolio construction rule may read at one bar.

    Nothing in it reaches past the bar it describes, so a rule handed a
    context cannot look ahead.

    Attributes
    ----------
    timestamp : pd.Timestamp
        The bar the rule decides on; the weights it returns fill at the next
        bar.
    predictions : xr.Dataset
        Every label's prediction at the bar: one variable per label name, in
        the predictor's label order, on ``symbol``.
    tradable : xr.DataArray
        Booleans on ``symbol``: whether the symbol can be traded at the bar,
        judged from nothing later than the bar (the price dataset's
        ``tradable_bars``: by default, a real fill price at the bar). A rule
        treats a symbol without a finite prediction of the label it reads as
        not selectable too.
    current_weights : xr.DataArray
        The weights currently held, on ``symbol``, valued at this bar's
        valuation price: what the orders of the earlier rebalances left,
        rejected orders and delisting settlements included; 0.0 where
        nothing is held, all 0.0 before the first rebalance.
    returns : xr.DataArray or None
        The trailing window of one-bar returns ending at the bar, on
        ``(timestamp, symbol)``, of the rule's ``lookback_bars`` length (no
        bars for a rule that needs none), read from the rule's last
        ``history_bars`` raw valuation prices up to and including the bar.
        Each return is computed from the last valuation price known at its
        bar within those, so a halt shows as zero returns and then the whole
        gap on the bar the symbol trades again; NaN before a symbol's first
        price in them and on that price's bar. ``None`` in a
        context built by hand for a rule that reads none.
    staleness : xr.DataArray or None
        Bars since each symbol's last real valuation price within the
        rule's last ``history_bars`` bars, on ``symbol``: 0 when it has one
        at the bar, NaN when it has none in those bars. ``None`` without
        prices.
    factors : xr.Dataset or None
        The values at the bar of the ``Factor`` panels the rule declares in
        ``required_factors()`` (a factor risk model's exposures, for
        example): one variable per factor name, on ``symbol``, NaN where a
        symbol has none. ``None`` when the rule declares none.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> symbols = ["AAA", "BBB", "CCC"]
    >>> context = PortfolioContext(
    ...     timestamp=pd.Timestamp("2024-01-02"),
    ...     predictions=xr.Dataset({"ret": ("symbol", [0.3, 0.1, 0.2])}, coords={"symbol": symbols}),
    ...     tradable=xr.DataArray([True, True, False], dims="symbol", coords={"symbol": symbols}),
    ...     current_weights=xr.DataArray([0.0, 0.4, 0.6], dims="symbol", coords={"symbol": symbols}),
    ... )
    >>> context.symbols.tolist()
    ['AAA', 'BBB', 'CCC']
    >>> context.locked.values
    array([False, False,  True])
    """

    timestamp: pd.Timestamp
    predictions: xr.Dataset
    tradable: xr.DataArray
    current_weights: xr.DataArray
    returns: xr.DataArray | None = None
    factors: xr.Dataset | None = None
    staleness: xr.DataArray | None = None

    @property
    def symbols(self) -> np.ndarray:
        """The symbols of the bar, the axis the returned weights must be on.

        Examples
        --------
        >>> context.symbols.tolist()
        ['AAA', 'BBB', 'CCC']
        """
        return self.tradable.symbol.values

    @property
    def locked(self) -> xr.DataArray:
        """Booleans on ``symbol``: held and not tradable, so kept at the current weight.

        Examples
        --------
        >>> context.locked.values
        array([False, False,  True])
        """
        held = self.current_weights.sel(symbol=self.symbols) != 0
        return held & ~self.tradable.astype(bool)


@dataclass(frozen=True)
class Decision:
    """What a rule decided at one bar: ``PortfolioConstructor.decide``'s result.

    Attributes
    ----------
    weights : xr.DataArray
        The weights to hold after the bar, on ``symbol`` in the order of the
        context's symbols: finite on a traded bar, all NaN to hold the
        current position.
    failure : str or None
        The message of the ``PortfolioConstructionError`` that made the bar
        a hold, or ``None`` when the rule decided it.
    events : dict
        The events the rule's row reported in ``attrs["events"]``, as the
        rule gave them (``{name: [symbol, ...]}`` or ``{name: count}``);
        empty after a failure.

    Examples
    --------
    >>> decision = Decision(
    ...     weights=xr.DataArray([0.5, 0.5, 0.0], dims="symbol", coords={"symbol": ["AAA", "BBB", "CCC"]}),
    ... )
    >>> decision.failure is None, decision.events
    (True, {})
    """

    weights: xr.DataArray
    failure: str | None = None
    events: dict[str, Any] = dataclasses.field(default_factory=dict)


class _Configured(Component):
    """A component built from one frozen config dataclass: a rule or a risk model.

    It is serialised and rebuilt by the component rule
    (``quantlab.core.component``): a field holding another component (a
    rule's risk model) is declared with ``component()`` on the config
    dataclass, and a free-form parameter dict is kept as data.
    """

    #: The dataclass of the component's parameters.
    config_cls: type

    def __init__(self, config):
        """Initialize the component from an instance of ``config_cls``."""
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{type(self).__name__} takes a {self.config_cls.__name__}, got "
                f"{type(config).__name__}"
            )
        self._config = config

    @property
    def config(self):
        """The component's parameters, as given.

        Examples
        --------
        >>> rule.config
        TopNConfig(direction='long_only', top_n=2, score_label=None)
        """
        return self._config

    def __repr__(self) -> str:
        """Return ``ClassName(field=value, ...)`` of the config's fields."""
        fields = ", ".join(
            f"{f.name}={getattr(self._config, f.name)!r}"
            for f in dataclasses.fields(self._config)
        )
        return f"{type(self).__name__}({fields})"

    def __eq__(self, other) -> bool:
        """Equal when of the same class with equal configs."""
        return type(other) is type(self) and other.config == self.config

    def __hash__(self) -> int:
        """Hash of the class and the config."""
        return hash((type(self), self.config))


@dataclass(frozen=True)
class CovarianceEstimate:
    """A risk model's covariance of returns at one bar, over the symbols it covers.

    Attributes
    ----------
    symbols : np.ndarray
        The symbols the estimate covers, the order of ``covariance``'s
        rows and columns; a symbol with too little history is left out.
    covariance : np.ndarray
        The dense ``[n, n]`` covariance matrix.

    Examples
    --------
    >>> estimate = CovarianceEstimate(
    ...     symbols=np.array(["AAA", "BBB"]),
    ...     covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
    ... )
    >>> estimate.variance
    array([0.04, 0.09])
    >>> estimate.factor_form() is None
    True
    """

    symbols: np.ndarray
    covariance: np.ndarray

    @property
    def variance(self) -> np.ndarray:
        """Each covered symbol's variance, the covariance's diagonal.

        Examples
        --------
        >>> estimate.variance
        array([0.04, 0.09])
        """
        return np.diag(self.covariance).copy()

    def scaled(self, factor: float) -> Self:
        """Return the estimate with every entry multiplied by ``factor``.

        Variance is linear in time, so a one-bar covariance times ``n`` is
        the covariance of ``n``-bar returns.

        Examples
        --------
        >>> estimate.scaled(5).variance
        array([0.2 , 0.45])
        """
        return dataclasses.replace(self, covariance=self.covariance * factor)

    def subset(self, rows: np.ndarray) -> Self:
        """Return the estimate over the symbols at positions ``rows``, in that order.

        Examples
        --------
        >>> estimate.subset(np.array([1])).symbols.tolist()
        ['BBB']
        """
        rows = np.asarray(rows, dtype=np.intp)
        return dataclasses.replace(
            self,
            symbols=self.symbols[rows],
            covariance=self.covariance[np.ix_(rows, rows)],
        )

    def factor_form(self):
        """Return the factor form of the covariance, or ``None`` when it has none.

        A dense estimate such as Ledoit-Wolf has none. A
        ``FactorCovarianceEstimate`` returns ``(exposures,
        factor_covariance, specific_variance)``, from which an optimiser
        builds a low-rank risk term instead of the dense one.

        Examples
        --------
        >>> estimate.factor_form() is None
        True
        """
        return None


@dataclass(frozen=True)
class FactorCovarianceEstimate:
    """A covariance of returns in factor form: ``B F B' + diag(D)``.

    The estimate a factor risk model returns (reserved: no shipped risk
    model returns one). With ``n`` symbols and ``k`` factors, ``B`` holds
    each symbol's exposures, ``F`` the factor returns' covariance and ``D``
    each symbol's specific (idiosyncratic) variance. ``factor_form()``
    returns the three, and an optimiser that finds them prices risk as
    ``|F^(1/2) B' w|^2 + w' diag(D) w``, which costs ``O(n k)`` rather than
    the ``O(n^2)`` of the dense matrix. It is used wherever a
    ``CovarianceEstimate`` is: ``covariance`` builds the dense matrix on
    demand.

    Attributes
    ----------
    symbols : np.ndarray
        The ``n`` symbols the estimate covers, the order of ``exposures``'
        rows and ``specific_variance``.
    exposures : np.ndarray
        ``B``, ``[n, k]``.
    factor_covariance : np.ndarray
        ``F``, ``[k, k]``, symmetric positive semi-definite.
    specific_variance : np.ndarray
        ``D``, ``[n]``, non-negative.

    Examples
    --------
    >>> estimate = FactorCovarianceEstimate(
    ...     symbols=np.array(["AAA", "BBB"]),
    ...     exposures=np.array([[1.0], [0.5]]),
    ...     factor_covariance=np.array([[0.04]]),
    ...     specific_variance=np.array([0.01, 0.02]),
    ... )
    >>> estimate.covariance
    array([[0.05, 0.02],
           [0.02, 0.03]])
    >>> estimate.variance
    array([0.05, 0.03])
    >>> [part.shape for part in estimate.factor_form()]
    [(2, 1), (1, 1), (2,)]
    """

    symbols: np.ndarray
    exposures: np.ndarray
    factor_covariance: np.ndarray
    specific_variance: np.ndarray

    @property
    def covariance(self) -> np.ndarray:
        """The dense ``[n, n]`` covariance ``B F B' + diag(D)``.

        Examples
        --------
        >>> estimate.covariance.shape
        (2, 2)
        """
        b = self.exposures
        return b @ self.factor_covariance @ b.T + np.diag(self.specific_variance)

    @property
    def variance(self) -> np.ndarray:
        """Each symbol's variance, ``diag(B F B') + D``, without the dense matrix.

        Examples
        --------
        >>> estimate.variance
        array([0.05, 0.03])
        """
        b = self.exposures
        return np.einsum("ij,jk,ik->i", b, self.factor_covariance, b) + self.specific_variance

    def scaled(self, factor: float) -> Self:
        """Return the estimate with ``F`` and ``D`` multiplied by ``factor``.

        Examples
        --------
        >>> estimate.scaled(5).variance
        array([0.25, 0.15])
        """
        return dataclasses.replace(
            self,
            factor_covariance=self.factor_covariance * factor,
            specific_variance=self.specific_variance * factor,
        )

    def subset(self, rows: np.ndarray) -> Self:
        """Return the estimate over the symbols at positions ``rows``, in that order.

        Examples
        --------
        >>> estimate.subset(np.array([1])).variance
        array([0.03])
        """
        rows = np.asarray(rows, dtype=np.intp)
        return dataclasses.replace(
            self,
            symbols=self.symbols[rows],
            exposures=self.exposures[rows],
            specific_variance=self.specific_variance[rows],
        )

    def factor_form(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(exposures, factor_covariance, specific_variance)``.

        Examples
        --------
        >>> exposures, factor_covariance, specific = estimate.factor_form()
        >>> specific
        array([0.01, 0.02])
        """
        return self.exposures, self.factor_covariance, self.specific_variance


class RiskModel(_Configured, ABC):
    """Base class of every risk model: the covariance of returns at one bar.

    "Risk model" is the industry's name (a Barra-style risk model), not a
    model in this project's sense: a risk model is not trained, has no
    checkpoint and is not a ``BaseModel``. It is an estimator run afresh at
    every bar from what the context holds, such as the trailing return
    window. A forecast that feeds it, such as predicted volatility, comes
    from a model in ``quantlab/model`` through the predictor; the risk model
    only combines it with what it estimates.

    Subclass it, set ``config_cls`` to a dataclass of the model's
    parameters (with a ``lookback_bars`` field when it reads a return
    window) and implement ``estimate``. The estimate is of one-bar returns;
    a rule scales it to its own horizon.

    The interface is reserved for a factor risk model, which is not
    implemented yet. Such a model declares the ``Factor`` panels it reads,
    its exposures for example, in ``required_factors()``; the backtest reads
    them over its window, each warmed up like a model's features, and puts
    their values at the bar in ``context.factors``. It returns a
    ``FactorCovarianceEstimate``, whose ``factor_form()`` makes the
    mean-variance optimiser build a low-rank risk term.

    Parameters
    ----------
    config : dataclass instance
        An instance of ``config_cls``.

    Examples
    --------
    >>> from quantlab.base.config import LedoitWolfConfig
    >>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
    >>> risk = LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60))
    >>> isinstance(risk, RiskModel), risk.lookback_bars
    (True, 60)
    """

    @property
    def lookback_bars(self) -> int:
        """Bars of one-bar returns ``estimate`` reads, ending at the bar.

        ``config.lookback_bars`` when the config has one, else 0.

        Examples
        --------
        >>> risk.lookback_bars
        60
        """
        return int(getattr(self._config, "lookback_bars", 0))

    @property
    def history_bars(self) -> int:
        """Raw valuation prices up to and including a bar that the model reads there.

        ``lookback_bars + 1`` by default: the last price before the window
        seeds its first return. A model whose coverage reads staleness
        reaches further back (see ``LedoitWolfRiskModel``).

        Examples
        --------
        >>> risk.history_bars
        66
        """
        return self.lookback_bars + 1

    def required_factors(self) -> list["Factor"]:
        """The ``Factor`` panels ``estimate`` reads from ``context.factors``.

        Any ``Factor`` qualifies (KunQuant, Polars, or a plain one such as
        a one-hot industry exposure). The backtest computes each over its
        window, the factor's own ``warmup_bars`` before it included, and
        hands ``estimate`` their values at the bar. Empty by default.

        Examples
        --------
        >>> risk.required_factors()
        []
        """
        return []

    @abstractmethod
    def estimate(
        self, context: PortfolioContext, volatility: xr.DataArray | None = None
    ) -> CovarianceEstimate:
        """Estimate the covariance of one-bar returns at the context's bar.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context; its ``returns`` window is the history.
        volatility : xr.DataArray, optional
            Per-symbol one-bar volatilities on ``symbol`` to use in place of
            the historical ones.

        Returns
        -------
        CovarianceEstimate or FactorCovarianceEstimate
            The covariance over the symbols with enough history; in factor
            form when the model has one.

        Examples
        --------
        >>> risk.estimate(context).symbols.tolist()
        ['AAA', 'BBB', 'CCC']
        """


class PortfolioConstructor(_Configured, ABC):
    """Base class of every rule that turns one bar's scores into target weights.

    Subclass it, set ``config_cls`` to a dataclass of the rule's parameters
    and implement ``construct``; override ``bind`` to check the label specs
    and read what the rule needs from them, ``lookback_bars`` when the rule
    reads a return window, ``history_bars`` when it reads more raw prices
    than ``lookback_bars + 1``, and ``required_factors`` when it reads
    factor panels. ``decide`` makes the one-bar decision on a
    context ``DecisionInputs`` assembles.
    ``get_config`` and ``from_config`` serialise the rule as its config's
    fields plus the class's import path under ``"name"``, which a
    backtest's ``config.json`` records.

    Parameters
    ----------
    config : dataclass instance
        An instance of ``config_cls``.

    Raises
    ------
    TypeError
        If ``config`` is not a ``config_cls`` instance.

    Examples
    --------
    >>> from quantlab.base.config import TopNConfig
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
    >>> isinstance(rule, PortfolioConstructor), rule.config.top_n
    (True, 2)
    """

    @property
    def lookback_bars(self) -> int:
        """Bars of one-bar returns each context's ``returns`` window holds.

        It sets the default ``history_bars``, the prices a context is built
        from. 0 by default.

        Examples
        --------
        >>> rule.lookback_bars
        0
        """
        return 0

    @property
    def history_bars(self) -> int:
        """Raw valuation prices up to and including a bar that each context there is built from.

        A context's ``returns`` and ``staleness`` read only these: the
        prices are forward-filled within this window alone, and a symbol
        without a real price in it has NaN staleness, so a decision does not
        depend on the first bar of the caller's history, however far back it
        goes. ``DecisionInputs`` reads the ``history_bars - 1`` bars before
        a panel as its warm-up.
        ``lookback_bars + 1`` by default (the window's first return needs
        the price before it).

        Examples
        --------
        >>> rule.history_bars
        1
        """
        return self.lookback_bars + 1

    def required_factors(self) -> list["Factor"]:
        """The ``Factor`` panels whose values at each bar the rule reads from ``context.factors``.

        The backtest computes each over its window, warm-up included, and
        slices it per bar. Empty by default; a rule holding a risk model
        declares the risk model's.

        Examples
        --------
        >>> rule.required_factors()
        []
        """
        return []

    def bind(self, labels: Sequence[LabelSpec]) -> None:
        """Check the label specs and read from them what the rule needs.

        Called once when a backtest is constructed, before any data is read
        or model trained, with the specs of the predictor's labels; and by
        ``DecisionInputs.from_run`` with the specs a run's prediction panel
        records. A rule reading a label's span resolves it here. The specs
        are the only metadata a rule may read about a prediction. The
        default accepts any specs.

        Parameters
        ----------
        labels : Sequence[LabelSpec]
            One spec per prediction variable, in the predictor's order.

        Raises
        ------
        ValueError
            If the rule cannot use the labels.

        Examples
        --------
        >>> rule.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)]) is None
        True
        """

    @abstractmethod
    def construct(self, context: PortfolioContext) -> xr.DataArray:
        """Return the weights to hold after the context's bar.

        Parameters
        ----------
        context : PortfolioContext
            What is known at the bar.

        Returns
        -------
        xr.DataArray
            One weight per symbol of ``context.symbols``, on ``symbol``:
            finite, 0.0 where nothing is held, gross exposure at most one;
            or all NaN to hold the current position.

        Raises
        ------
        PortfolioConstructionError
            If the bar cannot be decided (the loop then holds it).

        Examples
        --------
        >>> rule.construct(context).values  # CCC is locked at 0.6
        array([0.2, 0.2, 0.6])
        """

    def decide(self, context: PortfolioContext) -> Decision:
        """Decide one bar: run ``construct``, check its row and hold a bar it cannot solve.

        The one-bar decision, shared by ``DecisionInputs.weights`` and
        event-driven executors. A ``PortfolioConstructionError`` from
        ``construct`` becomes a hold (all-NaN weights) carrying the error's
        message, with a warning. A row that breaks the weights contract is a
        bug in the rule, not a hold, and raises. ``decide`` is not meant to
        be overridden: a rule implements ``construct``.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context, from ``DecisionInputs`` or built by hand.

        Returns
        -------
        Decision
            The weights on ``context.symbols`` (all NaN to hold), the failure
            message or ``None``, and the row's events.

        Raises
        ------
        ValueError
            If ``construct`` returns weights on other symbols than the
            context's, a row mixing NaN and finite values, a changed locked
            position (held, not tradable) or weight on a symbol neither
            tradable nor held.

        Examples
        --------
        >>> decision = rule.decide(context)
        >>> decision.weights.values, decision.failure, decision.events
        (array([0.2, 0.2, 0.6]), None, {})
        """
        symbols = context.symbols
        try:
            decided = self.construct(context)
        except PortfolioConstructionError as exc:
            logger.warning(
                f"{type(self).__name__}: holding the current position at "
                f"{pd.Timestamp(context.timestamp).isoformat()}: {exc}"
            )
            return Decision(
                weights=xr.DataArray(np.full(len(symbols), np.nan), dims="symbol", coords={"symbol": symbols}),
                failure=str(exc),
            )
        row = self._checked_row(decided, context)
        return Decision(
            weights=xr.DataArray(row, dims="symbol", coords={"symbol": symbols}),
            events=dict(decided.attrs.get("events", {})),
        )

    def _checked_row(self, weights: xr.DataArray, context: PortfolioContext) -> np.ndarray:
        """Return one bar's weights on the context's symbols, refusing a row that breaks the contract.

        Refused: another axis, a row mixing NaN and finite values, and a
        rebalance row that changes a locked position or gives weight to a
        symbol neither tradable nor held.
        """
        symbols, timestamp = context.symbols, context.timestamp
        got = pd.Index(weights.symbol.values)
        if not (len(got) == len(symbols) and got.sort_values().equals(pd.Index(symbols).sort_values())):
            raise ValueError(
                f"{type(self).__name__}.construct returned weights on other symbols "
                f"than the context's at {pd.Timestamp(timestamp)}"
            )
        row = np.asarray(weights.sel(symbol=symbols).values, dtype=np.float64)
        finite = np.isfinite(row)
        if finite.any() and not finite.all():
            raise ValueError(
                f"{type(self).__name__}.construct returned a row mixing finite "
                f"weights and NaN at {pd.Timestamp(timestamp)}; return every symbol "
                f"finite (0.0 when not held) or all NaN to hold"
            )
        if not finite.any():
            return row
        current = np.asarray(context.current_weights.sel(symbol=symbols).values, dtype=np.float64)
        tradable = np.asarray(context.tradable.values, dtype=bool)
        locked = np.asarray(context.locked.values, dtype=bool)
        moved = locked & (np.abs(row - current) > 1e-12)
        stray = ~tradable & ~locked & (row != 0)
        for bad, what in ((moved, "changed the locked position"), (stray, "gave weight to the untradable, unheld symbol")):
            if bad.any():
                shown = [str(v) for v in symbols[bad][:5]]
                raise ValueError(
                    f"{type(self).__name__}.construct {what}(s) {shown} at "
                    f"{pd.Timestamp(timestamp)}; a symbol not tradable at the bar "
                    f"keeps its current weight (0.0 when not held)"
                )
        return row
