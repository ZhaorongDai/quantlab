"""The root classes of portfolio construction: the rule from one bar's scores to weights.

A *portfolio construction* rule turns the scores of one bar, and the weights
currently held, into the weights to hold after that bar (ADR 0012). Every
rule derives from ``PortfolioConstructor``, whose one abstract method,
``construct(context)``, is handed a ``PortfolioContext`` holding only what is
known at that bar. One bar is decided by two public methods:
``build_context`` assembles the bar's context from its predictions,
tradability, current holdings and valuation prices, and ``decide`` runs
``construct``, checks the row and holds a bar the rule cannot solve. The
vectorised backtest calls ``construct_panel``, which loops ``decide`` over
the rebalance bars and models the holdings between them; an event-driven
executor (quantlab-trader) calls ``build_context`` and ``decide`` from its
bar handler with the holdings it really has.

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
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.backend import XrBackend

if TYPE_CHECKING:
    from quantlab.base.factor import Factor

_DIMS = ("timestamp", "symbol")


@dataclass(frozen=True)
class LabelSpec:
    """What a portfolio construction rule may know about one prediction variable.

    A rule is bound to the specs of the labels it is handed predictions of
    (``PortfolioConstructor.bind``), never to the model that predicted them,
    so a rule can be rebuilt from a run directory without its model
    (``quantlab.portfolio.prediction_panel.load_constructor``). The
    backtester derives the specs of its predictor with
    ``quantlab.base.backtest.label_specs``.

    Parameters
    ----------
    name : str
        The label's variable name, the prediction variable it scores.
    scale : str
        The prediction's scale: ``"raw"`` in the label's own units,
        ``"standardized"`` when it only ranks the cross-section.
    delay : int
        Bars between the bar a signal forms on and the first bar the label
        counts.
    span : int or None
        Bars the label accumulates over, or ``None`` for a label that is
        not a ``Forward`` label.

    Examples
    --------
    >>> spec = LabelSpec(name="ret_5", scale="raw", delay=1, span=5)
    >>> spec.span, dataclasses.asdict(spec)["scale"]
    (5, 'raw')
    """

    name: str
    scale: str
    delay: int
    span: int | None


@dataclass(frozen=True, eq=False)
class PredictionPanel:
    """A run's predictions together with the specs of the labels they predict.

    Every backtest run with a model writes the predictions its rule read
    into ``predictions.zarr`` (``FILE_NAME``) in its run directory: one
    variable per label on ``(timestamp, symbol)``, the store's ``attrs``
    holding ``format_version`` (``FORMAT_VERSION``) and ``labels``, a JSON
    list of the specs' fields. An executor rebuilds the run's rule from it
    with ``quantlab.portfolio.prediction_panel.load_constructor``.

    Parameters
    ----------
    predictions : xr.Dataset
        One variable per label, each on ``(timestamp, symbol)``, named
        exactly as ``labels`` name them; NaN where a symbol has no
        prediction.
    labels : Sequence[LabelSpec]
        The label specs, in the order of the prediction variables; stored
        as a tuple.

    Raises
    ------
    ValueError
        If the label names repeat, the variables are not exactly the label
        names, or a variable is not on ``(timestamp, symbol)``.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.base.portfolio import LabelSpec
    >>> predictions = xr.Dataset(
    ...     {"ret_5": (("timestamp", "symbol"), np.array([[0.1, -0.2], [0.3, np.nan]]))},
    ...     coords={"timestamp": pd.date_range("2024-01-02", periods=2),
    ...             "symbol": np.array(["AAA", "BBB"], dtype=object)},
    ... )
    >>> panel = PredictionPanel(predictions, [LabelSpec("ret_5", "raw", 1, 5)])
    >>> panel.labels
    (LabelSpec(name='ret_5', scale='raw', delay=1, span=5),)
    """

    #: The version of the ``predictions.zarr`` layout ``write`` writes and ``read`` reads.
    FORMAT_VERSION: ClassVar[int] = 1

    #: The file name of the prediction panel inside a run directory.
    FILE_NAME: ClassVar[str] = "predictions.zarr"

    predictions: xr.Dataset
    labels: tuple[LabelSpec, ...]

    def __post_init__(self) -> None:
        """Check the variables against the labels and order them by the labels."""
        labels = tuple(self.labels)
        names = [spec.name for spec in labels]
        if len(set(names)) != len(names):
            raise ValueError(f"PredictionPanel: label names repeat: {names}")
        variables = [str(name) for name in self.predictions.data_vars]
        if sorted(variables) != sorted(names):
            raise ValueError(
                f"PredictionPanel: the prediction variables {variables} are not "
                f"exactly the labels {names}"
            )
        for name in names:
            dims = self.predictions[name].dims
            if dims != _DIMS:
                raise ValueError(
                    f"PredictionPanel: variable {name!r} is on {dims}, not {_DIMS}"
                )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "predictions", self.predictions[names])

    def write(self, path: str | PathLike) -> Path:
        """Write the panel to the Zarr store ``path``, replacing any store there.

        The variables are written as they are; the store's ``attrs`` are
        exactly ``format_version`` and ``labels`` (the specs' fields as a
        JSON list).

        Parameters
        ----------
        path : str or os.PathLike
            Directory of the store, ``<run_dir>/predictions.zarr`` in a run.

        Returns
        -------
        Path
            ``path``.

        Examples
        --------
        >>> import tempfile
        >>> path = panel.write(Path(tempfile.mkdtemp()) / "predictions.zarr")
        >>> PredictionPanel.read(path).labels == panel.labels
        True
        """
        path = Path(path)
        data = self.predictions.copy()
        data.attrs = {
            "format_version": self.FORMAT_VERSION,
            "labels": json.dumps([dataclasses.asdict(spec) for spec in self.labels]),
        }
        for name in data.data_vars:
            data[name].attrs = {}
        XrBackend().to_internal(data).write(str(path))
        return path

    @classmethod
    def read(cls, path: str | PathLike) -> Self:
        """Read a panel ``write`` stored, into memory.

        Parameters
        ----------
        path : str or os.PathLike
            Directory of the store.

        Returns
        -------
        PredictionPanel
            The predictions, without the store's ``attrs``, and the specs.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the store is not a prediction panel of ``FORMAT_VERSION``.

        Examples
        --------
        >>> PredictionPanel.read(path).predictions["ret_5"].dims
        ('timestamp', 'symbol')
        """
        data = XrBackend().read(path).data
        labels = cls._labels_of(data, path)
        predictions = data.load()
        predictions.attrs = {}
        return cls(predictions, labels)

    @classmethod
    def read_labels(cls, path: str | PathLike) -> tuple[LabelSpec, ...]:
        """Read only the label specs of the panel stored at ``path``.

        The store is opened lazily and no prediction is loaded.

        Parameters
        ----------
        path : str or os.PathLike
            Directory of the store.

        Returns
        -------
        tuple[LabelSpec, ...]
            The specs, in the order of the prediction variables.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the store is not a prediction panel of ``FORMAT_VERSION``.

        Examples
        --------
        >>> PredictionPanel.read_labels(path)
        (LabelSpec(name='ret_5', scale='raw', delay=1, span=5),)
        """
        return cls._labels_of(XrBackend().read(path).data, path)

    @classmethod
    def _labels_of(cls, data: xr.Dataset, path) -> tuple[LabelSpec, ...]:
        """Return the label specs of the opened store ``data`` read from ``path``."""
        version = data.attrs.get("format_version")
        if version != cls.FORMAT_VERSION:
            raise ValueError(
                f"{path} is not a prediction panel of format_version "
                f"{cls.FORMAT_VERSION} (found {version!r})"
            )
        return tuple(
            LabelSpec(**fields) for fields in json.loads(data.attrs["labels"])
        )


class PortfolioConstructionError(RuntimeError):
    """A rule could not decide a bar: the optimisation failed, was infeasible or had no solution.

    ``PortfolioConstructor.decide`` holds such a bar (all-NaN weights,
    the message as the decision's ``failure``) and logs a warning;
    ``construct_panel`` lists the bar in the result's
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
        bars for a rule that needs none). Each return is computed from the
        last valuation price known at its bar, so a halt shows as zero
        returns and then the whole gap on the bar the symbol trades again;
        NaN before a symbol's first price and on its first bar. ``None`` in a
        context built by hand for a rule that reads none.
    staleness : xr.DataArray or None
        Bars since each symbol's last real valuation price, on ``symbol``:
        0 when it has one at the bar, NaN before its first. ``None`` without
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


def _align_mask(mask: xr.DataArray, predictions: xr.Dataset) -> np.ndarray:
    """Return a boolean panel reordered onto the labels of ``predictions``.

    Alignment is by label rather than by position: a panel of the same
    shape but another symbol or timestamp order would otherwise pair each
    prediction with another symbol's tradability. A missing or extra label
    is refused instead of being treated as "not tradable", because a
    misaligned time axis would silently make every row unselectable.

    Raises
    ------
    ValueError
        If either axis has duplicate labels, or the two label sets differ
        on either axis.
    """
    mask = mask.transpose(*_DIMS)
    for dim in _DIMS:
        wanted = pd.Index(predictions[dim].values)
        got = pd.Index(mask[dim].values)
        if wanted.has_duplicates or got.has_duplicates:
            raise ValueError(
                f"{dim} labels must be unique in both the predictions and the "
                f"tradability panel"
            )
        if wanted.equals(got):
            continue
        missing = wanted.difference(got, sort=False)
        extra = got.difference(wanted, sort=False)
        if len(missing) or len(extra):
            raise ValueError(
                f"the tradability panel's {dim} labels differ from the "
                f"predictions': missing {[str(v) for v in missing[:10]]}, extra "
                f"{[str(v) for v in extra[:10]]}; tradability must be given on "
                f"exactly the predictions' labels"
            )
    aligned = mask.sel(
        timestamp=predictions.timestamp.values, symbol=predictions.symbol.values
    )
    return np.asarray(aligned.values, dtype=bool)


def _valuation_history(valuation_price: xr.DataArray) -> tuple[xr.DataArray, np.ndarray, xr.DataArray]:
    """Return the one-bar returns, staleness and forward-filled prices of raw valuation prices.

    The one formula behind every context's ``returns`` and ``staleness``,
    whether ``construct_panel`` or ``build_context`` builds it. Each row
    reads only the rows up to it, so the rows of a history that ends at a
    bar equal those of a longer history over the same start.
    """
    filled = valuation_price.ffill("timestamp")
    priced = np.isfinite(np.asarray(valuation_price.values, dtype=np.float64))
    rows = np.arange(priced.shape[0])[:, None]
    last = np.maximum.accumulate(np.where(priced, rows, -1), axis=0)
    returns = filled / filled.shift(timestamp=1) - 1.0
    staleness = np.where(last >= 0, rows - last, np.nan).astype(np.float64)
    return returns, staleness, filled


def _empty_window(symbols: np.ndarray) -> xr.DataArray:
    """Return a return window of no bars on ``symbols``: the window of a context built without prices."""
    return xr.DataArray(
        np.empty((0, len(symbols))),
        dims=_DIMS,
        coords={"timestamp": np.array([], dtype="datetime64[ns]"), "symbol": symbols},
    )


def _on_symbol(bar, what: str) -> xr.DataArray | xr.Dataset:
    """Return one bar's values on ``symbol`` alone, refusing a ``timestamp`` axis or duplicate symbols.

    A scalar ``timestamp`` coordinate (what ``.sel(timestamp=t)`` leaves)
    is dropped.
    """
    if "timestamp" in bar.dims:
        raise ValueError(f"{what} must be one bar's values on symbol, not a panel over timestamp")
    if tuple(bar.dims) != ("symbol",):
        raise ValueError(f"{what} must be on the symbol dimension only; got dims {tuple(bar.dims)}")
    if pd.Index(bar.symbol.values).has_duplicates:
        raise ValueError(f"{what} has duplicate symbols")
    return bar.drop_vars("timestamp", errors="ignore")


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


@dataclass(frozen=True)
class _PriceHistory:
    """The prices ``construct_panel`` reads, on the prediction symbols.

    ``returns`` holds the one-bar returns of the forward-filled valuation
    price and ``staleness`` the bars since each symbol's last real valuation
    price; ``raw_fill`` the fill prices as given, ``fill`` and ``valuation``
    the forward-filled prices, and ``delisted`` the delisting marks, as
    ``[T, S]`` arrays; and ``positions`` each prediction bar's row in them.
    """

    returns: xr.DataArray
    staleness: np.ndarray
    raw_fill: np.ndarray
    fill: np.ndarray
    valuation: np.ndarray
    delisted: np.ndarray
    positions: np.ndarray


class _Book:
    """The holdings the driver models between rebalances: shares and cash.

    It replays what the simulation does with each rebalance's targets, so
    the current weights handed to a rule are the ones really held, up to
    fees and slippage. The portfolio starts as 1.0 of cash. On each bar the
    orders are the queued targets, if any fill there, and the settlements of
    holdings delisted on the bar before, which close at that bar's
    predecessor's valuation whatever the targets say. An order without a raw
    fill price is rejected and leaves its holding alone. The book is valued
    at the order prices, then sells run first and buys in ascending order of
    value, each capped by the cash left.
    """

    def __init__(self, history: _PriceHistory):
        """Start flat on the given price history."""
        self.history = history
        self.shares = np.zeros(history.fill.shape[1])
        self.cash = 1.0
        self.applied_to = -1  # the last price row whose orders are applied
        self.queued: tuple[int, np.ndarray] | None = None  # (fill row, targets)

    def weights_at(self, row: int) -> np.ndarray:
        """Apply every order up to ``row`` and return the weights valued there."""
        for bar in range(self.applied_to + 1, row + 1):
            targets = None
            if self.queued is not None and self.queued[0] == bar:
                targets = self.queued[1]
                self.queued = None
            self._trade(bar, targets)
        self.applied_to = max(self.applied_to, row)
        worth = self.shares * np.nan_to_num(self.history.valuation[row])
        return worth / (self.cash + worth.sum())

    def queue(self, row: int, targets: np.ndarray) -> None:
        """Queue ``targets``, decided at ``row``, to fill on ``row + 1``."""
        if row + 1 < self.history.fill.shape[0]:
            self.queued = (row + 1, targets)

    def _trade(self, bar: int, targets: np.ndarray | None) -> None:
        """Run ``bar``'s orders: the targets that fill there and the settlements."""
        n = self.shares.size
        settle = (
            self.history.delisted[bar - 1] & (self.shares != 0)
            if bar > 0
            else np.zeros(n, dtype=bool)
        )
        if targets is None and not settle.any():
            return
        wanted = np.full(n, np.nan) if targets is None else np.array(targets, dtype=np.float64)
        price = np.where(settle, self.history.valuation[bar - 1] if bar > 0 else np.nan, self.history.fill[bar])
        priced = np.isfinite(price) & (price > 0)
        at = np.where(priced, price, 0.0)
        value = self.cash + float(np.sum(self.shares * at))
        accepted = priced & (settle | (np.isfinite(wanted) & np.isfinite(self.history.raw_fill[bar])))
        wanted[settle] = 0.0
        delta = np.zeros(n)
        delta[accepted] = wanted[accepted] * value / price[accepted] - self.shares[accepted]
        trade_value = delta * at
        for j in np.flatnonzero(trade_value < 0):
            self.shares[j] += delta[j]
            self.cash -= trade_value[j]
        for j in sorted(np.flatnonzero(trade_value > 0), key=lambda j: trade_value[j]):
            spend = min(trade_value[j], max(self.cash, 0.0))
            self.shares[j] += spend / price[j]
            self.cash -= spend


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
    and implement ``construct``; override ``bind`` to check the label specs
    and read what the rule needs from them, ``lookback_bars`` when the rule
    reads a return window, and ``required_factors`` when it reads factor
    panels. ``build_context`` and ``decide`` make the one-bar decision,
    and ``construct_panel`` loops them over a panel.
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

    def bind(self, labels: Sequence[LabelSpec]) -> None:
        """Check the label specs and read from them what the rule needs.

        Called once when a backtest is constructed, before any data is read
        or model trained, with the specs of the predictor's labels; and by
        ``load_constructor`` with the specs a run's prediction panel
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

    def build_context(
        self,
        timestamp,
        predictions: xr.Dataset,
        tradable: xr.DataArray,
        current_weights: xr.DataArray,
        *,
        valuation_price: xr.DataArray | None = None,
        factors: xr.Dataset | None = None,
    ) -> PortfolioContext:
        """Build the context of one bar from what is known at it.

        The public half of the one-bar decision: ``construct_panel`` and an
        event-driven executor such as quantlab-trader hand ``decide`` the
        same context for the same bar. The ``returns`` window and
        ``staleness`` come from ``valuation_price`` by the formula the panel
        loop uses: the one-bar returns of the forward-filled prices, the
        last ``lookback_bars`` of them ending at the bar, and the bars since
        each symbol's last real price, counted within the history given (NaN
        when it holds none). Given the valuation prices the panel loop read,
        up to the bar, the context equals the one the loop built there.

        Parameters
        ----------
        timestamp : pd.Timestamp or datetime-like
            The bar decided on.
        predictions : xr.Dataset
            Every label's prediction at the bar, one variable per label on
            ``symbol``; its symbols, in their order, are the context's. A
            scalar ``timestamp`` coordinate is dropped.
        tradable : xr.DataArray
            Booleans on ``symbol``: exactly the predictions' symbols, in any
            order.
        current_weights : xr.DataArray
            The weights held, valued at the bar's valuation price, on
            ``symbol``: finite, 0.0 where nothing is held. A symbol it lacks
            is not held.
        valuation_price : xr.DataArray, optional
            Raw (not forward-filled) valuation prices on ``(timestamp,
            symbol)`` whose last timestamp is ``timestamp``, from far enough
            back to seed the forward fill; required when ``lookback_bars``
            is positive. Without it the window has no bars and
            ``staleness`` is ``None``.
        factors : xr.Dataset, optional
            The values at the bar of the rule's ``required_factors()``, one
            variable per factor name on ``symbol``; NaN where a symbol has
            none. Required when the rule declares any.

        Returns
        -------
        PortfolioContext

        Raises
        ------
        ValueError
            If an input is not on ``symbol`` alone or repeats a symbol, the
            tradability is on other symbols than the predictions, a current
            weight is not finite or sits on a symbol without a prediction,
            ``tradable`` is not boolean, ``valuation_price`` is missing while
            the rule reads returns, repeats a symbol or a timestamp, is not
            in time order or does not end at ``timestamp``, or ``factors``
            is missing or lacks a name while the rule declares factors.

        Examples
        --------
        >>> symbols = ["AAA", "BBB", "CCC"]
        >>> prices = xr.DataArray(
        ...     [[10.0, 20.0, np.nan], [11.0, np.nan, 30.0], [12.0, 22.0, 33.0]],
        ...     dims=("timestamp", "symbol"),
        ...     coords={"timestamp": pd.bdate_range("2024-01-01", periods=3), "symbol": symbols},
        ... )
        >>> bar = rule.build_context(
        ...     pd.Timestamp("2024-01-03"),
        ...     xr.Dataset({"ret": ("symbol", [0.3, 0.1, 0.2])}, coords={"symbol": symbols}),
        ...     xr.DataArray([True, True, True], dims="symbol", coords={"symbol": symbols}),
        ...     xr.DataArray([0.0, 0.5, 0.0], dims="symbol", coords={"symbol": ["CCC", "BBB", "AAA"]}),
        ...     valuation_price=prices,
        ... )
        >>> bar.current_weights.values, bar.staleness.values
        (array([0. , 0.5, 0. ]), array([0., 0., 0.]))
        >>> bar.returns.sizes["timestamp"]  # rule.lookback_bars is 0
        0
        """
        timestamp = pd.Timestamp(timestamp)
        predictions = _on_symbol(predictions, "predictions")
        symbols = predictions.symbol.values
        tradable = _on_symbol(tradable, "tradable")
        if not pd.Index(tradable.symbol.values).sort_values().equals(pd.Index(symbols).sort_values()):
            raise ValueError("tradable must be given on exactly the predictions' symbols")
        if tradable.dtype != bool:
            raise ValueError(f"tradable must be booleans; got dtype {tradable.dtype}")
        current_weights = _on_symbol(current_weights, "current_weights")
        given = np.asarray(current_weights.values, dtype=np.float64)
        if not np.isfinite(given).all():
            raise ValueError("current_weights must be finite (0.0 where nothing is held)")
        outside = ~pd.Index(current_weights.symbol.values).isin(symbols)
        if (given[outside] != 0).any():
            shown = [str(v) for v in current_weights.symbol.values[outside][:5]]
            raise ValueError(f"current_weights holds symbols without a prediction: {shown}")
        current = np.asarray(
            current_weights.reindex(symbol=symbols, fill_value=0.0).values, dtype=np.float64
        )

        lookback = self.lookback_bars
        if valuation_price is None:
            if lookback:
                raise ValueError(
                    f"{type(self).__name__} reads {lookback} bars of returns; pass "
                    f"valuation_price= to build_context"
                )
            window, staleness = _empty_window(symbols), None
        else:
            valuation_price = valuation_price.transpose(*_DIMS)
            bars = pd.Index(valuation_price.timestamp.values)
            if not bars.is_monotonic_increasing or bars.has_duplicates:
                raise ValueError("valuation_price timestamps must be unique and increasing")
            if pd.Index(valuation_price.symbol.values).has_duplicates:
                raise ValueError("valuation_price has duplicate symbols")
            if not len(bars) or pd.Timestamp(bars[-1]) != timestamp:
                raise ValueError(
                    f"valuation_price must end at the bar {timestamp}; pass the "
                    f"prices up to and including it"
                )
            returns, stale, _ = _valuation_history(valuation_price.reindex(symbol=symbols))
            n = returns.sizes["timestamp"]
            window = returns.isel(timestamp=slice(max(0, n - lookback), n) if lookback else slice(n, n))
            staleness = xr.DataArray(stale[-1].copy(), dims="symbol", coords={"symbol": symbols})

        if factors is None:
            if self.required_factors():
                raise ValueError(
                    f"{type(self).__name__} declares required_factors(); pass their "
                    f"values at the bar as factors= to build_context"
                )
        else:
            factors = _on_symbol(factors, "factors")
            self._check_factor_names(factors)
            factors = factors.reindex(symbol=symbols).load()

        return PortfolioContext(
            timestamp=timestamp,
            predictions=predictions,
            tradable=xr.DataArray(
                np.asarray(tradable.sel(symbol=symbols).values, dtype=bool),
                dims="symbol",
                coords={"symbol": symbols},
            ),
            current_weights=xr.DataArray(current, dims="symbol", coords={"symbol": symbols}),
            returns=window,
            factors=factors,
            staleness=staleness,
        )

    def decide(self, context: PortfolioContext) -> Decision:
        """Decide one bar: run ``construct``, check its row and hold a bar it cannot solve.

        The other half of the one-bar decision, shared by ``construct_panel``
        and event-driven executors. A ``PortfolioConstructionError`` from
        ``construct`` becomes a hold (all-NaN weights) carrying the error's
        message, with a warning. A row that breaks the weights contract is a
        bug in the rule, not a hold, and raises. ``decide`` is not meant to
        be overridden: a rule implements ``construct``.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context, from ``build_context`` or built by hand.

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

    def construct_panel(
        self,
        predictions: xr.Dataset,
        tradable: xr.DataArray,
        rebalance: np.ndarray,
        *,
        fill_price: xr.DataArray | None = None,
        valuation_price: xr.DataArray | None = None,
        delisted: xr.DataArray | None = None,
        factors: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Build target weights for every bar of a panel.

        Loops ``decide`` over the rebalance bars in time order, handing
        each the context ``build_context`` would build for its own bar from
        the valuation prices up to it (the returns and staleness are
        computed once for the whole panel by the same formula, since each
        bar's rows read nothing after it): its predictions, its
        tradability, the ``lookback_bars`` one-bar returns of
        ``valuation_price`` ending at it, the values at it of the
        ``factors`` the rule declares, and the weights currently held.
        With prices, those are the holdings the earlier targets left,
        modelled the way the simulation trades them: filled at the next
        bar's forward-filled fill price, sells before buys and each buy
        capped by the cash left, an order without a raw fill price rejected
        (the holding kept), a holding marked in ``delisted`` closed at its
        last valuation on the next bar, and the book valued at this bar's
        forward-filled valuation price. Fees and slippage are not modelled.
        Without prices the current weights are the last traded row. They
        are all 0.0 before the first rebalance.

        A bar that returns all NaN holds. A bar whose ``construct`` raises
        ``PortfolioConstructionError`` holds too, with a warning, and is
        listed in the result's ``attrs["failed_bars"]``. Events a returned
        row reports in ``attrs["events"]`` are gathered by name into the
        result's ``attrs["events"]``, one record per bar: an event that
        names its symbols (``{name: [symbol, ...]}``, such as the
        optimiser's ``closed_without_risk``) gives ``{"bar", "symbols"}``,
        one that only counts them (``{name: n}``, such as the top-n rule's
        ``tie_at_cutoff``) gives ``{"bar", "count"}``. A bar that does not
        rebalance gets an all-NaN row, meaning "hold". A returned row must
        keep every locked position (held, not tradable) at its current
        weight and give 0.0 to a symbol neither tradable nor held.

        Parameters
        ----------
        predictions : xr.Dataset
            One variable per label on ``(timestamp, symbol)``.
        tradable : xr.DataArray
            Booleans on the same labels as ``predictions`` (in any order).
        rebalance : np.ndarray
            One boolean per timestamp, True on rebalance bars.
        fill_price, valuation_price : xr.DataArray, optional
            Raw (not forward-filled) prices on ``(timestamp, symbol)``: every
            timestamp of ``predictions`` and, before them, the warm-up the
            return window needs. Given together; required when
            ``lookback_bars`` is positive.
        delisted : xr.DataArray, optional
            Booleans marking each delisted symbol's last priced bar (the
            price dataset's ``delisting_bars``), on the prices' labels or a
            part of them; read only with prices.
        factors : xr.Dataset, optional
            The panels of the rule's ``required_factors()``, one variable
            per factor name on ``(timestamp, symbol)``, covering every
            prediction timestamp; required when the rule declares any.

        Returns
        -------
        xr.Dataset
            One ``weight`` variable on ``(timestamp, symbol)``, on the
            predictions' labels, with ``attrs["failed_bars"]`` the ISO
            timestamps of the bars held after a failure and
            ``attrs["events"]`` the rows' events by name.

        Raises
        ------
        ValueError
            If ``rebalance`` does not have one entry per timestamp, the
            tradability panel is on other labels, only one price is given, a
            price is missing or lacks a prediction timestamp, the factor
            panels are missing or lack a prediction timestamp or a declared
            factor name, or ``construct`` returns weights on other symbols, a row mixing
            NaN and finite values, a changed locked position or weight on a
            symbol neither tradable nor held.

        Examples
        --------
        >>> ts = pd.bdate_range("2024-01-01", periods=3)
        >>> scores = xr.Dataset(
        ...     {"ret": (("timestamp", "symbol"), [[0.3, 0.1, 0.2], [0.0, 0.5, 0.4], [0.9, 0.8, 0.7]])},
        ...     coords={"timestamp": ts, "symbol": ["AAA", "BBB", "CCC"]},
        ... )
        >>> tradable = xr.ones_like(scores["ret"], dtype=bool)
        >>> weights = rule.construct_panel(scores, tradable, np.array([True, False, True]))
        >>> weights["weight"].values
        array([[0.5, 0. , 0.5],
               [nan, nan, nan],
               [0.5, 0.5, 0. ]])
        >>> weights.attrs["failed_bars"], weights.attrs["events"]
        ([], {})
        """
        predictions = predictions.transpose(*_DIMS)
        tradable_values = _align_mask(tradable, predictions)
        rebalance = self._check_rebalance(rebalance, predictions)
        timestamps = predictions.timestamp.values
        symbols = predictions.symbol.values
        history = self._check_prices(fill_price, valuation_price, delisted, predictions)
        factors = self._check_factors(factors, predictions)
        lookback = self.lookback_bars
        book = None if history is None else _Book(history)

        weights = np.full((len(timestamps), len(symbols)), np.nan)
        traded = np.zeros(len(symbols))  # the last traded row, without prices
        failed = []
        events: dict[str, list[dict]] = {}
        for t in np.flatnonzero(rebalance):
            if history is None:
                current = traded
                window = _empty_window(symbols)
            else:
                position = int(history.positions[t])
                current = book.weights_at(position)
                window = history.returns.isel(
                    timestamp=slice(max(0, position - lookback + 1), position + 1)
                    if lookback
                    else slice(position + 1, position + 1)
                )
            context = PortfolioContext(
                timestamp=pd.Timestamp(timestamps[t]),
                predictions=predictions.isel(timestamp=t, drop=True),
                tradable=xr.DataArray(
                    tradable_values[t].copy(), dims="symbol", coords={"symbol": symbols}
                ),
                current_weights=xr.DataArray(
                    np.array(current, dtype=np.float64),
                    dims="symbol",
                    coords={"symbol": symbols},
                ),
                returns=window,
                factors=None if factors is None else factors.isel(timestamp=t, drop=True),
                staleness=None
                if history is None
                else xr.DataArray(
                    history.staleness[position].copy(),
                    dims="symbol",
                    coords={"symbol": symbols},
                ),
            )
            decision = self.decide(context)
            label = pd.Timestamp(timestamps[t]).isoformat()
            if decision.failure is not None:
                failed.append(label)
                continue
            row = decision.weights.values
            for name, value in decision.events.items():
                record = {"bar": label}
                if isinstance(value, (int, np.integer)):
                    record["count"] = int(value)
                else:
                    record["symbols"] = list(value)
                events.setdefault(name, []).append(record)
            if np.isnan(row).all():
                continue
            weights[t] = row
            traded = row
            if book is not None:
                book.queue(position, row)
        out = xr.Dataset(
            {"weight": (_DIMS, weights)},
            coords={"timestamp": timestamps, "symbol": symbols},
        )
        out.attrs["failed_bars"] = failed
        out.attrs["events"] = events
        return out

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

    def _check_prices(self, fill_price, valuation_price, delisted, predictions: xr.Dataset):
        """Return the price history the loop reads, or None without prices.

        The history holds the one-bar returns of the forward-filled valuation
        price and the staleness (the context's window and staleness), the
        raw and forward-filled fill prices, the forward-filled
        valuation prices and the delisting marks (the modelled holdings),
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
        if delisted is None:
            marks = np.zeros(fill_price.shape, dtype=bool)
        else:
            marks = np.asarray(
                delisted.transpose(*_DIMS)
                .reindex(
                    timestamp=valuation_price.timestamp.values,
                    symbol=symbols,
                    fill_value=False,
                )
                .values,
                dtype=bool,
            )
        returns, staleness, valuation_filled = _valuation_history(valuation_price)
        return _PriceHistory(
            returns=returns,
            staleness=staleness,
            raw_fill=fill_price.values.astype(np.float64),
            fill=fill_price.ffill("timestamp").values.astype(np.float64),
            valuation=valuation_filled.values.astype(np.float64),
            delisted=marks,
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
        self._check_factor_names(factors)
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

    def _check_factor_names(self, factors: xr.Dataset) -> None:
        """Refuse factor values that lack a name of the rule's ``required_factors()``."""
        declared = [name for factor in self.required_factors() for name in factor.get_factor_names()]
        missing = [name for name in declared if name not in factors.data_vars]
        if missing:
            raise ValueError(
                f"{type(self).__name__} declares factors {missing} that the given "
                f"factor values lack"
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
