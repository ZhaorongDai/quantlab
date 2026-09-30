"""The root class of portfolio construction: the rule from one bar's scores to weights.

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
position". A rule holds no state between bars.

Shipped rules live in ``quantlab/portfolio/predefined``; this module imports
no solver.
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd
import xarray as xr

_DIMS = ("timestamp", "symbol")


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
        The weights currently held, on ``symbol``; 0.0 where nothing is
        held and all 0.0 before the first rebalance.

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


class PortfolioConstructor(ABC):
    """Base class of every rule that turns one bar's scores into target weights.

    Subclass it, set ``config_cls`` to a dataclass of the rule's parameters
    and implement ``construct``; override ``check_predictor`` to refuse a
    predictor whose labels the rule cannot use, and ``construct_panel`` to
    vectorise the rule. ``get_config`` and ``from_config`` serialise the
    rule as its config's fields plus the class's import path under
    ``"name"``, which a backtest's ``config.json`` records.

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

    #: The dataclass of the rule's parameters.
    config_cls: type

    def __init__(self, config):
        """Initialize the rule; see the class docstring for parameters."""
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{type(self).__name__} takes a {self.config_cls.__name__}, got "
                f"{type(config).__name__}"
            )
        self._config = config

    @property
    def config(self):
        """The rule's parameters, as given.

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
        """The rule's class as a dotted import path, the ``name`` of its config.

        Examples
        --------
        >>> rule.import_path
        'quantlab.portfolio.predefined.top_n.TopNConstructor'
        """
        return f"{type(self).__module__}.{type(self).__qualname__}"

    def get_config(self) -> dict[str, Any]:
        """Return the config's fields plus the class's import path under ``"name"``.

        Examples
        --------
        >>> rule.get_config()
        {'direction': 'long_only', 'top_n': 2, 'score_label': None, 'name': 'quantlab.portfolio.predefined.top_n.TopNConstructor'}
        """
        return {**dataclasses.asdict(self._config), "name": self.import_path}

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Rebuild the rule from the dict ``get_config()`` returned.

        Parameters
        ----------
        config : dict
            The dict ``get_config()`` returned, for example read back from a
            backtest run's ``config.json``; ``"name"`` is ignored here.

        Examples
        --------
        >>> TopNConstructor.from_config(rule.get_config()) == rule
        True
        """
        params = {key: value for key, value in config.items() if key != "name"}
        return cls(cls.config_cls(**params))

    def check_predictor(self, labels: list[str], label_scales: dict[str, str]) -> None:
        """Refuse a predictor whose labels the rule cannot use.

        Called when a backtest is constructed, before any data is read or
        model trained. The default accepts any predictor.

        Parameters
        ----------
        labels : list[str]
            The predictor's label names, in its order.
        label_scales : dict[str, str]
            Each label name's scale, ``"raw"`` or ``"standardized"``.

        Raises
        ------
        ValueError
            If the rule cannot use the predictor.

        Examples
        --------
        >>> rule.check_predictor(["fwd_ret_1"], {"fwd_ret_1": "raw"}) is None
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
    ) -> xr.Dataset:
        """Build target weights for every bar of a panel.

        The default loops ``construct`` over the rebalance bars in time
        order, handing each the context of its own bar only: its
        predictions, its eligibility and the weights currently held, which
        are the last rebalance's weights (all 0.0 before the first; a bar
        that returns all NaN holds and keeps them). A bar that does not
        rebalance gets an all-NaN row, meaning "hold". A rule may override
        this for speed; the override must return exactly what the loop does.

        Parameters
        ----------
        predictions : xr.Dataset
            One variable per label on ``(timestamp, symbol)``.
        eligible : xr.DataArray
            Booleans on the same labels as ``predictions`` (in any order).
        rebalance : np.ndarray
            One boolean per timestamp, True on rebalance bars.

        Returns
        -------
        xr.Dataset
            One ``weight`` variable on ``(timestamp, symbol)``, on the
            predictions' labels.

        Raises
        ------
        ValueError
            If ``rebalance`` does not have one entry per timestamp, the
            eligibility panel is on other labels, or ``construct`` returns
            weights on other symbols or a row mixing NaN and finite values.

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
        """
        predictions = predictions.transpose(*_DIMS)
        eligible_values = self._check_eligible(eligible, predictions)
        rebalance = self._check_rebalance(rebalance, predictions)
        timestamps = predictions.timestamp.values
        symbols = predictions.symbol.values
        weights = np.full((len(timestamps), len(symbols)), np.nan)
        current = np.zeros(len(symbols))
        for t in np.flatnonzero(rebalance):
            context = PortfolioContext(
                timestamp=pd.Timestamp(timestamps[t]),
                predictions=predictions.isel(timestamp=t, drop=True),
                eligible=xr.DataArray(
                    eligible_values[t].copy(), dims="symbol", coords={"symbol": symbols}
                ),
                current_weights=xr.DataArray(
                    current.copy(), dims="symbol", coords={"symbol": symbols}
                ),
            )
            row = self._checked_row(self.construct(context), symbols, timestamps[t])
            if np.isnan(row).all():
                continue
            weights[t] = row
            current = row
        return xr.Dataset(
            {"weight": (_DIMS, weights)},
            coords={"timestamp": timestamps, "symbol": symbols},
        )

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
