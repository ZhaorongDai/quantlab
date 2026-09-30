"""The root classes of portfolio construction: the rule from one bar's scores to weights.

A *portfolio construction* rule turns the scores of one bar, and the weights
currently held, into the weights to hold after that bar (ADR 0012). Every
rule derives from ``PortfolioConstructor``, whose one abstract method,
``construct(context)``, is handed a ``PortfolioContext`` holding only what is
known at that bar. The vectorised backtest calls ``construct_panel``, which
by default loops ``construct`` over the rebalance bars; a future event-driven
backtest calls ``construct`` from its bar handler. A rule that can be
vectorised may override ``construct_panel`` for speed, but the override must
equal the loop exactly.

The output follows the weights contract (D-03): on a rebalance bar every
symbol gets a finite weight, an unselected or ineligible one exactly 0.0,
with gross exposure at most one; an all-NaN row means "hold the current
position". A rule holds no state between bars. A bar the rule cannot solve
raises ``PortfolioConstructionError``, and the loop holds it instead.

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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

if TYPE_CHECKING:
    from quantlab.base.factor import Factor

_DIMS = ("timestamp", "symbol")


class PortfolioConstructionError(RuntimeError):
    """A rule could not decide a bar: the optimisation failed, was infeasible or had no solution.

    ``PortfolioConstructor.construct_panel`` holds such a bar (an all-NaN
    row), logs a warning and lists the bar in the result's
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
    eligible : xr.DataArray
        Booleans on ``symbol``: whether the driver lets the symbol be held
        after the bar. The vectorised backtest marks a symbol eligible when
        it has a fill price at the next bar. A rule treats a symbol without
        a finite prediction of the label it reads as ineligible too.
    current_weights : xr.DataArray
        The weights currently held, on ``symbol``: the last traded weights,
        filled at the next bar's fill price and drifted to this bar's
        valuation price; 0.0 where nothing is held, all 0.0 before the first
        rebalance.
    returns : xr.DataArray or None
        The trailing window of one-bar returns ending at the bar, on
        ``(timestamp, symbol)``, of the rule's ``lookback_bars`` length (no
        bars for a rule that needs none); NaN where a symbol has no return.
        ``None`` in a context built by hand for a rule that reads none.
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
    ...     eligible=xr.DataArray([True, True, False], dims="symbol", coords={"symbol": symbols}),
    ...     current_weights=xr.DataArray(np.zeros(3), dims="symbol", coords={"symbol": symbols}),
    ... )
    >>> context.symbols.tolist()
    ['AAA', 'BBB', 'CCC']
    """

    timestamp: pd.Timestamp
    predictions: xr.Dataset
    eligible: xr.DataArray
    current_weights: xr.DataArray
    returns: xr.DataArray | None = None
    factors: xr.Dataset | None = None

    @property
    def symbols(self) -> np.ndarray:
        """The symbols of the bar, the axis the returned weights must be on.

        Examples
        --------
        >>> context.symbols.tolist()
        ['AAA', 'BBB', 'CCC']
        """
        return self.eligible.symbol.values


def _align_eligible(eligible: xr.DataArray, predictions: xr.Dataset) -> np.ndarray:
    """Return ``eligible`` reordered onto the labels of ``predictions``, as booleans.

    Alignment is by label rather than by position: a panel of the same
    shape but another symbol or timestamp order would otherwise pair each
    prediction with another symbol's eligibility. A missing or extra label
    is refused instead of being treated as "not eligible", because a
    misaligned time axis would silently make every row unselectable.

    Raises
    ------
    ValueError
        If either axis has duplicate labels, or the two label sets differ
        on either axis.
    """
    eligible = eligible.transpose(*_DIMS)
    for dim in _DIMS:
        wanted = pd.Index(predictions[dim].values)
        got = pd.Index(eligible[dim].values)
        if wanted.has_duplicates or got.has_duplicates:
            raise ValueError(
                f"{dim} labels must be unique in both the predictions and the "
                f"eligibility panel"
            )
        if wanted.equals(got):
            continue
        missing = wanted.difference(got, sort=False)
        extra = got.difference(wanted, sort=False)
        if len(missing) or len(extra):
            raise ValueError(
                f"the eligibility panel's {dim} labels differ from the "
                f"predictions': missing {[str(v) for v in missing[:10]]}, extra "
                f"{[str(v) for v in extra[:10]]}; eligibility must be given on "
                f"exactly the predictions' labels"
            )
    aligned = eligible.sel(
        timestamp=predictions.timestamp.values, symbol=predictions.symbol.values
    )
    return np.asarray(aligned.values, dtype=bool)


@dataclass(frozen=True)
class _PriceHistory:
    """The prices ``construct_panel`` reads, on the prediction symbols.

    ``returns`` holds the raw one-bar valuation returns; ``fill`` and
    ``valuation`` the forward-filled prices as ``[T, S]`` arrays; and
    ``positions`` each prediction bar's row in them.
    """

    returns: xr.DataArray
    fill: np.ndarray
    valuation: np.ndarray
    positions: np.ndarray


class _Configured:
    """A component built from one frozen config dataclass: a rule or a risk model.

    ``get_config`` returns the config's fields plus the class's import path
    under ``"name"``, a field holding another such component as that
    component's own config; ``from_config`` rebuilds both.
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

    @property
    def import_path(self) -> str:
        """The class as a dotted import path, the ``name`` of its config.

        Examples
        --------
        >>> rule.import_path
        'quantlab.portfolio.predefined.top_n.TopNConstructor'
        """
        return f"{type(self).__module__}.{type(self).__qualname__}"

    def get_config(self) -> dict[str, Any]:
        """Return the config's fields plus the class's import path under ``"name"``.

        A field holding another component (a rule's risk model) is written
        as that component's own ``get_config()``.

        Examples
        --------
        >>> rule.get_config()
        {'direction': 'long_only', 'top_n': 2, 'score_label': None, 'name': 'quantlab.portfolio.predefined.top_n.TopNConstructor'}
        """
        out = {}
        for f in dataclasses.fields(self._config):
            value = getattr(self._config, f.name)
            out[f.name] = (
                value.get_config() if isinstance(value, _Configured) else value
            )
        return {**out, "name": self.import_path}

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Rebuild the component from the dict ``get_config()`` returned.

        A nested dict carrying a ``"name"`` is rebuilt by ``from_config`` of
        the class it names.

        Parameters
        ----------
        config : dict
            The dict ``get_config()`` returned, for example read back from a
            backtest run's ``config.json``; the top-level ``"name"`` is
            ignored here.

        Examples
        --------
        >>> TopNConstructor.from_config(rule.get_config()) == rule
        True
        """
        from quantlab.utils.module import get_cls_from_path

        params = {}
        for key, value in config.items():
            if key == "name":
                continue
            if isinstance(value, dict) and "name" in value:
                value = get_cls_from_path(value["name"]).from_config(value)
            params[key] = value
        return cls(cls.config_cls(**params))


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
    and implement ``construct``; override ``bind`` to check the predictor
    and read what the rule needs from it, ``lookback_bars`` when the rule
    reads a return window, and ``construct_panel`` to vectorise the rule.
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

        The backtester adds it to its bar-counted warm-up, so the first
        backtest bar already has a full window. 0 by default.

        Examples
        --------
        >>> rule.lookback_bars
        0
        """
        return 0

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

    def bind(self, predictor) -> None:
        """Check the predictor and read from it what the rule needs.

        Called once when a backtest is constructed, before any data is read
        or model trained; a rule reading a label's span resolves it here.
        The default accepts any predictor.

        Parameters
        ----------
        predictor : Predictor
            The backtest's predictor: its ``labels`` (label objects) and
            ``label_scales``.

        Raises
        ------
        ValueError
            If the rule cannot use the predictor.

        Examples
        --------
        >>> rule.bind(model) is None
        True
        """

    @staticmethod
    def _label_names(predictor) -> list[str]:
        """The predictor's label variable names, in its order."""
        return [
            str(name) for label in predictor.labels for name in label.get_factor_names()
        ]

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
        >>> rule.construct(context).values
        array([0.5, 0.5, 0. ])
        """

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
        """Build target weights for every bar of a panel.

        The default loops ``construct`` over the rebalance bars in time
        order, handing each the context of its own bar only: its
        predictions, its eligibility, the ``lookback_bars`` one-bar returns
        of ``valuation_price`` ending at it, the values at it of the
        ``factors`` the rule declares, and the weights currently held.
        Those are the weights of the last bar that traded, filled at the
        next bar's fill price and held to this bar's valuation price: each
        symbol grows by ``g = valuation[t] / fill[t0 + 1]``, both prices
        forward-filled as the simulation values them (a delisted holding
        keeps its last price until it is sold), and the weights are
        renormalised to the portfolio's value, ``w * g / (1 + sum(w * (g -
        1)))``. They are all 0.0 before the first rebalance, and not drifted
        without prices. A bar that returns all NaN holds. A bar whose
        ``construct`` raises ``PortfolioConstructionError`` holds too, with
        a warning, and is listed in the result's ``attrs["failed_bars"]``. A
        bar that does not rebalance gets an all-NaN row, meaning "hold". A
        rule may override this for speed; the override must return exactly
        what the loop does.

        Parameters
        ----------
        predictions : xr.Dataset
            One variable per label on ``(timestamp, symbol)``.
        eligible : xr.DataArray
            Booleans on the same labels as ``predictions`` (in any order).
        rebalance : np.ndarray
            One boolean per timestamp, True on rebalance bars.
        fill_price, valuation_price : xr.DataArray, optional
            Raw (not forward-filled) prices on ``(timestamp, symbol)``: every
            timestamp of ``predictions`` and, before them, the warm-up the
            return window needs. Given together; required when
            ``lookback_bars`` is positive.
        factors : xr.Dataset, optional
            The panels of the rule's ``required_factors()``, one variable
            per factor name on ``(timestamp, symbol)``, covering every
            prediction timestamp; required when the rule declares any.

        Returns
        -------
        xr.Dataset
            One ``weight`` variable on ``(timestamp, symbol)``, on the
            predictions' labels, with ``attrs["failed_bars"]`` the ISO
            timestamps of the bars held after a failure.

        Raises
        ------
        ValueError
            If ``rebalance`` does not have one entry per timestamp, the
            eligibility panel is on other labels, only one price is given, a
            price is missing or lacks a prediction timestamp, the factor
            panels are missing or lack a prediction timestamp, or
            ``construct`` returns weights on other symbols or a row mixing
            NaN and finite values.

        Examples
        --------
        >>> ts = pd.bdate_range("2024-01-01", periods=3)
        >>> scores = xr.Dataset(
        ...     {"ret": (("timestamp", "symbol"), [[0.3, 0.1, 0.2], [0.0, 0.5, 0.4], [0.9, 0.8, 0.7]])},
        ...     coords={"timestamp": ts, "symbol": ["AAA", "BBB", "CCC"]},
        ... )
        >>> eligible = xr.ones_like(scores["ret"], dtype=bool)
        >>> weights = PortfolioConstructor.construct_panel(rule, scores, eligible, np.array([True, False, True]))
        >>> weights["weight"].values
        array([[0.5, 0. , 0.5],
               [nan, nan, nan],
               [0.5, 0.5, 0. ]])
        >>> weights.attrs["failed_bars"]
        []
        """
        predictions = predictions.transpose(*_DIMS)
        eligible_values = self._check_eligible(eligible, predictions)
        rebalance = self._check_rebalance(rebalance, predictions)
        timestamps = predictions.timestamp.values
        symbols = predictions.symbol.values
        history = self._check_prices(fill_price, valuation_price, predictions)
        factors = self._check_factors(factors, predictions)
        lookback = self.lookback_bars

        weights = np.full((len(timestamps), len(symbols)), np.nan)
        traded = np.zeros(len(symbols))  # the last traded row
        traded_at = None  # its position in the price history
        failed = []
        for t in np.flatnonzero(rebalance):
            if history is None:
                current = traded
                window = xr.DataArray(
                    np.empty((0, len(symbols))),
                    dims=_DIMS,
                    coords={"timestamp": timestamps[:0], "symbol": symbols},
                )
            else:
                position = int(history.positions[t])
                current = (
                    traded
                    if traded_at is None
                    else self._drift(
                        traded, history.valuation[position] / history.fill[traded_at + 1]
                    )
                )
                window = history.returns.isel(
                    timestamp=slice(max(0, position - lookback + 1), position + 1)
                    if lookback
                    else slice(position + 1, position + 1)
                )
            context = PortfolioContext(
                timestamp=pd.Timestamp(timestamps[t]),
                predictions=predictions.isel(timestamp=t, drop=True),
                eligible=xr.DataArray(
                    eligible_values[t].copy(), dims="symbol", coords={"symbol": symbols}
                ),
                current_weights=xr.DataArray(
                    np.array(current, dtype=np.float64),
                    dims="symbol",
                    coords={"symbol": symbols},
                ),
                returns=window,
                factors=None if factors is None else factors.isel(timestamp=t, drop=True),
            )
            try:
                row = self._checked_row(self.construct(context), symbols, timestamps[t])
            except PortfolioConstructionError as exc:
                label = pd.Timestamp(timestamps[t]).isoformat()
                logger.warning(
                    f"{type(self).__name__}: holding the current position at "
                    f"{label}: {exc}"
                )
                failed.append(label)
                continue
            if np.isnan(row).all():
                continue
            weights[t] = row
            traded = row
            traded_at = None if history is None else int(history.positions[t])
        out = xr.Dataset(
            {"weight": (_DIMS, weights)},
            coords={"timestamp": timestamps, "symbol": symbols},
        )
        out.attrs["failed_bars"] = failed
        return out

    @staticmethod
    def _drift(weights: np.ndarray, growth: np.ndarray) -> np.ndarray:
        """Grow each weight by ``growth`` and renormalise to the portfolio's value.

        A symbol without a growth (never priced) is held flat.
        """
        growth = np.where(np.isfinite(growth), growth, 1.0)
        value = 1.0 + float(np.sum(weights * (growth - 1.0)))
        return weights * growth / value

    @staticmethod
    def _check_eligible(eligible: xr.DataArray, predictions: xr.Dataset) -> np.ndarray:
        """Return ``eligible`` as booleans on the predictions' labels (``_align_eligible``)."""
        return _align_eligible(eligible, predictions)

    @staticmethod
    def _check_rebalance(rebalance, predictions: xr.Dataset) -> np.ndarray:
        """Return ``rebalance`` as booleans, refusing a mask of the wrong length."""
        rebalance = np.asarray(rebalance, dtype=bool)
        n_bars = predictions.sizes["timestamp"]
        if rebalance.shape != (n_bars,):
            raise ValueError(
                f"rebalance mask shape {rebalance.shape} does not match "
                f"{n_bars} timestamps"
            )
        return rebalance

    def _check_prices(self, fill_price, valuation_price, predictions: xr.Dataset):
        """Return the price history the loop reads, or None without prices.

        The history holds the raw one-bar valuation returns (the context's
        window), the forward-filled fill and valuation prices (the drift),
        and each prediction bar's position in them.
        """
        if fill_price is None and valuation_price is None:
            if self.lookback_bars:
                raise ValueError(
                    f"{type(self).__name__} reads {self.lookback_bars} bars of "
                    f"returns; pass fill_price= and valuation_price= to construct_panel"
                )
            return None
        if fill_price is None or valuation_price is None:
            raise ValueError("pass fill_price and valuation_price together")
        symbols = predictions.symbol.values
        valuation_price = valuation_price.transpose(*_DIMS).reindex(symbol=symbols)
        fill_price = fill_price.transpose(*_DIMS).reindex(
            timestamp=valuation_price.timestamp.values, symbol=symbols
        )
        positions = pd.Index(valuation_price.timestamp.values).get_indexer(
            predictions.timestamp.values
        )
        if (positions < 0).any():
            raise ValueError(
                "the prices must cover every prediction timestamp; missing "
                f"{[str(v) for v in predictions.timestamp.values[positions < 0][:5]]}"
            )
        return _PriceHistory(
            returns=valuation_price / valuation_price.shift(timestamp=1) - 1.0,
            fill=fill_price.ffill("timestamp").values.astype(np.float64),
            valuation=valuation_price.ffill("timestamp").values.astype(np.float64),
            positions=positions,
        )

    def _check_factors(self, factors: xr.Dataset | None, predictions: xr.Dataset) -> xr.Dataset | None:
        """Return the declared factor panels on the predictions' labels, or None.

        A symbol the panels lack gets NaN. Refused: no panels when the rule
        declares factors, and panels lacking a prediction timestamp.
        """
        if factors is None:
            if self.required_factors():
                raise ValueError(
                    f"{type(self).__name__} declares required_factors(); pass "
                    f"their panels as factors= to construct_panel"
                )
            return None
        factors = factors.transpose(*_DIMS)
        missing = pd.Index(predictions.timestamp.values).difference(
            pd.Index(factors.timestamp.values)
        )
        if len(missing):
            raise ValueError(
                "the factor panels must cover every prediction timestamp; missing "
                f"{[str(v) for v in missing[:5]]}"
            )
        return factors.reindex(
            timestamp=predictions.timestamp.values, symbol=predictions.symbol.values
        ).load()

    def _checked_row(self, weights: xr.DataArray, symbols: np.ndarray, timestamp) -> np.ndarray:
        """Return one bar's weights on ``symbols``, refusing another axis or a mixed row."""
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
        return row
