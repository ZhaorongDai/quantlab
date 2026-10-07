"""The root class of the risk layer: ``FactorRiskModel``, the contract of a factor risk model.

A *factor risk model* explains each bar's stock returns by a few factors and
forecasts their risk from that (ADR 0024). It is estimated ahead of any
backtest, over the whole history, as stores of one row per bar, and each row
uses nothing later than its bar. The portfolio layer reads the stores; it
does not estimate anything itself.

A store is a ``RiskStore``. It follows the factor lifecycle: ``compute(start,
end)`` returns the rows of a range, ``build`` writes them, ``extend`` appends
the bars after the recorded range and ``read`` returns a range of the store,
which must lie inside the range recorded beside it. Its rows are not a
``(timestamp, symbol)`` panel (factor returns are on ``(timestamp, factor)``),
so the store reads and writes its Zarr store itself. Every row depends only
on a bounded window of bars ending at its own, so a store built in one go and
one built and then extended are identical.

``FactorRiskModel`` fixes what every factor risk model produces, whatever its
method: two stores of one-bar quantities (``REGRESSION_VARIABLES`` and
``ESTIMATE_VARIABLES``), the factor axis (``factor_names``) and each symbol's
exposures to it at a bar (``exposure_matrix``). The portfolio side
(``FactorRiskStoreEstimator``) and the bias statistics (``quantlab.risk.bias``)
read only that. How the factor returns and the forecasts are estimated is a
subclass's; shipped models live in ``quantlab.risk.predefined`` (USE4 on the
``BarraStyle`` exposures).
"""

import dataclasses
import datetime
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.core.component import Component
from quantlab.dataset.merged import MergedDataset
from quantlab.risk.config import FactorRiskConfig
from quantlab.runs.record import record_read
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.date_range import as_label, check_range, last_moment, range_text
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: A date a store is asked for.
Date = str | datetime.date | pd.Timestamp


def covered_factors(covariance: np.ndarray) -> np.ndarray:
    """Return the factors of a ``[K, K]`` covariance whose block is all finite.

    A factor without a variance is dropped first; then, while a pair has no
    covariance (too few common bars), the factor missing the most pairs (a
    greedy choice, the first such factor on a tie).

    Examples
    --------
    >>> covariance = np.array([[1.0, np.nan, 0.1], [np.nan, 2.0, 0.2], [0.1, 0.2, np.nan]])
    >>> covered_factors(covariance)
    array([False,  True, False])
    """
    kept = np.isfinite(np.diag(covariance))
    while True:
        missing = ~np.isfinite(covariance) & kept[:, None] & kept[None, :]
        if not missing.any():
            return kept
        kept[np.argmax(missing.sum(axis=1))] = False


class RiskStore:
    """One store of a factor risk model: rows per bar, written and read by date range.

    Built by the risk model that owns it, from a function computing the rows
    of a date range (its warm-up included). A row may carry a ``symbol``
    axis; it is written sorted (numerically for integer symbols), and only
    symbols with a value somewhere in the rows are kept, so the axis does not
    depend on how a range was split between ``build`` and ``extend``.

    A ``read`` inside an open ``quantlab.runs.record.DataRecorder`` is
    recorded under the model's key and the store's ``part``
    (``risk_model.estimate``); one store's reads are merged into one request
    over the first to the last bar read, so a reader taking a row per bar
    costs one record.

    Parameters
    ----------
    model : FactorRiskModel
        The risk model owning the store: the source of its recorded reads.
    part : str
        The store's name in the model, ``"regression"`` or ``"estimate"``.
    path : str or None
        The Zarr store; the date range is recorded in
        ``<path>.range.json``.
    compute : callable
        ``compute(start, end)`` returning the rows of the range as an
        ``xr.Dataset`` on ``timestamp`` (and its own other axes).
    warmup_bars : int
        Bars before the requested start ``compute`` reads.

    Examples
    --------
    >>> store = model.regression
    >>> store.build("2024-01-01", "2024-03-31").store_range()
    ('2024-01-01', '2024-03-31')
    >>> store.extend("2024-04-30").store_range()
    ('2024-01-01', '2024-04-30')
    >>> dict(store.read("2024-04-01", "2024-04-30").sizes)["timestamp"]
    22
    """

    #: Appended to ``path`` to name the JSON file recording the store's range.
    RANGE_SUFFIX = ".range.json"

    def __init__(
        self,
        model: "FactorRiskModel",
        part: str,
        path: str | None,
        compute: Callable[[object, object], xr.Dataset],
        warmup_bars: int,
    ):
        """Initialize the store; see the class docstring for parameters."""
        self.model = model
        self.part = part
        #: The ``Class.store`` named in log and error messages.
        self.owner = f"{model.class_name}.{part}"
        self.path = path
        self._compute = compute
        self.warmup_bars = warmup_bars

    def __repr__(self) -> str:
        """Return the owner and the path."""
        return f"RiskStore({self.owner!r}, path={self.path!r})"

    def compute(self, start: Date, end: Date) -> xr.Dataset:
        """Compute the rows from ``start`` to ``end``, both inclusive.

        The compute function reads ``warmup_bars`` bars before ``start`` and
        returns only the requested bars. Nothing is written.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range. A date-only ``end`` includes every bar of that day.

        Returns
        -------
        xr.Dataset
            The rows, in memory, symbols sorted and the empty ones dropped.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.

        Examples
        --------
        >>> rows = model.regression.compute("2024-02-01", "2024-02-29")
        >>> rows["factor_return"].dims
        ('timestamp', 'factor')
        """
        check_range(start, end, f"{self.owner}.compute()")
        rows = self._compute(start, end)
        if "symbol" not in rows.dims:
            return rows
        with_symbol = [name for name in rows.data_vars if "symbol" in rows[name].dims]
        present = xr.concat(
            [rows[name].notnull().any([d for d in rows[name].dims if d != "symbol"])
             for name in with_symbol],
            dim="variable",
        ).any("variable")
        kept = [s for s, keep in zip(rows["symbol"].values.tolist(), present.values) if keep]
        return rows.sel(symbol=sort_symbol_axis(kept))

    def build(self, start: Date, end: Date) -> Self:
        """Compute ``start`` to ``end`` and write it as the store.

        The store at ``path`` is replaced and the range recorded beside it.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range the store covers.

        Returns
        -------
        RiskStore
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the store has no ``path``.

        Examples
        --------
        >>> model.regression.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        path = self._require_path("build")
        rows = self.compute(start, end)
        with Timer(f"{self.owner}: build"):
            self._range_file(path).unlink(missing_ok=True)
            XrBackend().to_internal(rows).write(path, mode="w")
            self._record_range(path, start, end)
        return self

    def extend(self, end: Date) -> Self:
        """Append the bars after the recorded range, up to ``end``.

        The bars are computed with ``compute``, so they are warmed from the
        inputs' history; a symbol axis widens to fit.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range.

        Returns
        -------
        RiskStore
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the store has no ``path`` or no recorded range, or the
            recorded range already reaches ``end``.

        Examples
        --------
        >>> model.regression.extend("2024-04-30").store_range()
        ('2024-01-01', '2024-04-30')
        """
        path = self._require_path("extend")
        recorded = self.store_range()
        if recorded is None:
            raise ValueError(
                f"{self.owner}.extend(): the store at {path} has no recorded range; "
                f"write it with build(start, end) first."
            )
        recorded_start, recorded_end = recorded
        if last_moment(end) <= last_moment(recorded_end):
            raise ValueError(
                f"{self.owner}.extend(): the store at {path} already covers "
                f"{recorded_start} to {recorded_end}; extend() appends only bars after "
                f"{recorded_end}, got end {end!r}."
            )
        rows = self.compute(last_moment(recorded_end) + pd.Timedelta(1, "ns"), end)
        with Timer(f"{self.owner}: extend"):
            if rows.sizes["timestamp"]:
                XrBackend().to_internal(rows).widen_and_append(path)
            self._record_range(path, recorded_start, end)
        return self

    def read(self, start: Date, end: Date) -> xr.Dataset:
        """Return the store from ``start`` to ``end``, both inclusive.

        The store is opened lazily. The range must lie inside the one
        recorded beside it (``store_range``). The read is recorded by an open
        ``DataRecorder`` (see the class docstring).

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range. A date-only ``end`` includes every bar of that day.

        Returns
        -------
        xr.Dataset
            The rows of the range.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, the store has no recorded range,
            or the recorded range does not contain the requested one.

        Examples
        --------
        >>> model.regression.read("2024-03-20", "2024-04-10")
        Traceback (most recent call last):
        ValueError: Use4RiskModel.regression.read(): the store at ... covers 2024-01-01 to 2024-03-31, ...
        """
        check_range(start, end, f"{self.owner}.read()")
        path = self._require_path("read")
        recorded = self.store_range()
        if recorded is None:
            raise ValueError(
                f"{self.owner}.read(): the store at {path} has no recorded range, so it "
                f"cannot answer a date-range request; write it with build(start, end)."
            )
        recorded_start, recorded_end = recorded
        if pd.Timestamp(start) < pd.Timestamp(recorded_start) or last_moment(
            end
        ) > last_moment(recorded_end):
            raise ValueError(
                f"{self.owner}.read(): the store at {path} covers {recorded_start} to "
                f"{recorded_end}, which does not contain {start} to {end}. Extend it "
                f"with extend(end) or rebuild it with build(start, end)."
            )
        rows = xr.open_zarr(path).sel(timestamp=slice(as_label(start), as_label(end)))
        record_read(self.model, rows, part=self.part, store=path, reread_range=self.read)
        return rows

    def store_range(self) -> tuple[str, str] | None:
        """Return the ``(start, end)`` the store was built for, or ``None``.

        Examples
        --------
        >>> model.regression.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        if self.path is None or not self._range_file(self.path).is_file():
            return None
        recorded = json.loads(self._range_file(self.path).read_text())
        return recorded["start"], recorded["end"]

    def _require_path(self, method: str) -> str:
        """Return ``path``, or raise if the store has none."""
        if self.path is None:
            raise ValueError(f"{self.owner}.{method}(): the store has no path in the config.")
        return self.path

    def _range_file(self, path: str) -> Path:
        """Return the file recording the range of the store at ``path``."""
        return Path(f"{path}{self.RANGE_SUFFIX}")

    def _record_range(self, path: str, start, end) -> None:
        """Record ``start`` and ``end`` as the store's range."""
        write_json_atomically(
            self._range_file(path), {"start": range_text(start), "end": range_text(end)}
        )


#: The variables every regression store holds, with their dimensions: the
#: realized factor returns and specific returns of each bar.
REGRESSION_VARIABLES = {
    "factor_return": ("timestamp", "factor"),
    "specific_return": ("timestamp", "symbol"),
}

#: The variables every estimate store holds, with their dimensions: the
#: forecasts, made at each bar, of the next bar's factor covariance and
#: specific volatilities.
ESTIMATE_VARIABLES = {
    "factor_covariance": ("timestamp", "factor_i", "factor_j"),
    "specific_risk": ("timestamp", "symbol"),
}


#: The groups a factor can belong to (``FactorRiskModel.factor_groups``), in
#: the order the factor attribution reports them.
FACTOR_GROUPS = ("country", "industry", "style")


class FactorRiskModel(Component, ABC):
    """A factor risk model: the stores estimated ahead of a backtest.

    The interface every factor risk model presents, whatever its method:

    - ``factor_names``: the factor axis, and ``factor_groups``: each
      factor's group (country, industry or style; style by default);
    - ``exposure_names`` and ``exposure_matrix(exposures)``: the outputs of
      the exposures factor (``config.exposures``) it reads, and each
      symbol's exposures to ``factor_names`` at a bar, with whether it has
      them all;
    - the regression store (``regression``), one row per bar ``t``:
      ``factor_return`` on ``(timestamp, factor)``, the factor returns over
      the bar ending at ``t``, and ``specific_return`` on ``(timestamp,
      symbol)``, each symbol's return over that bar less its factor part,
      the exposures being those of the bar before;
    - the estimate store (``estimate``), one row per bar ``t``:
      ``factor_covariance`` on ``(timestamp, factor_i, factor_j)`` and
      ``specific_risk`` on ``(timestamp, symbol)``, forecasts made at ``t``
      of the next bar's factor covariance and specific volatilities;
    - ``prices`` and ``exposures``: its inputs over a date range.

    Every quantity is of one-bar returns, a row uses nothing later than its
    bar, and a store may add its own diagnostics. The ``factor`` axes are
    ``factor_names``. A missing value is NaN: a factor without a return on
    a bar, a forecast without enough history.

    A subclass implements the method: ``factor_names``,
    ``exposure_names``, ``exposure_matrix``, ``_compute_regression`` and
    ``_compute_estimate`` (the rows of a date range, its warm-up read
    before it), and the warm-ups ``regression_warmup_bars`` and
    ``estimate_warmup_bars``; it sets ``config_cls`` to its config, a
    ``FactorRiskConfig``, and checks its parameters in ``_validate``
    (calling this class's). The rows it computes are checked against the
    stores' variables before they are returned or written.

    Parameters
    ----------
    config : FactorRiskConfig
        The exposures factor, the price dataset and the method's
        parameters; an instance of the class's ``config_cls``.

    Raises
    ------
    TypeError
        If ``config`` is not an instance of ``config_cls``.
    ValueError
        If a parameter is invalid.

    Examples
    --------
    With ``model`` a ``Use4RiskModel``:

    >>> isinstance(model, FactorRiskModel)
    True
    >>> model.regression.build("2012-01-01", "2024-12-31")
    >>> model.regression.read("2020-01-01", "2020-12-31")["factor_return"].dims
    ('timestamp', 'factor')
    >>> model.estimate.build("2018-01-01", "2024-12-31")
    >>> model.estimate.read("2024-12-31", "2024-12-31")["factor_covariance"].dims
    ('timestamp', 'factor_i', 'factor_j')
    """

    #: The config class ``from_config`` rebuilds this model with.
    config_cls = FactorRiskConfig

    def __init__(self, config: FactorRiskConfig):
        """Initialize the model; see the class docstring for parameters."""
        self.config = config

    def __repr__(self) -> str:
        """Return the class name and its config."""
        return f"{type(self).__name__}(config={self.config})"

    def __eq__(self, other: object) -> bool:
        """Equal when of the same class with equal configs."""
        if type(other) is not type(self):
            return NotImplemented
        return self.config == other.config

    @property
    def class_name(self) -> str:
        """Bare class name, used in log and error messages.

        Examples
        --------
        >>> model.class_name
        'Use4RiskModel'
        """
        return type(self).__name__

    @property
    def config(self) -> FactorRiskConfig:
        """The model's normalised config: ``name`` filled, datasets merged.

        Examples
        --------
        >>> model.config is config
        False
        >>> model.config.name
        'quantlab.risk.predefined.use4.Use4RiskModel'
        """
        return self._config

    @config.setter
    def config(self, config: FactorRiskConfig):
        """Install a normalised copy of ``config``.

        ``name`` is set to this class's import path, a list or tuple in
        ``dataset`` becomes a ``MergedDataset`` of it, and the parameters are
        checked. The config passed in is never edited.
        """
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} takes a {self.config_cls.__name__}, got "
                f"{type(config).__name__}"
            )
        dataset = config.dataset
        if isinstance(dataset, (list, tuple)):
            dataset = MergedDataset(dataset)
        config = dataclasses.replace(config, dataset=dataset, name=self.import_path)
        self._validate(config)
        self._config = config

    def _validate(self, config: FactorRiskConfig) -> None:
        """Raise ``ValueError`` for a parameter the model cannot use.

        Checks the fields every factor risk model has; a subclass checks its
        own after calling this.
        """
        if config.exposure_data_strategy not in ("read", "cal"):
            raise ValueError(
                f"{self.class_name}: exposure_data_strategy must be 'read' or 'cal', got "
                f"{config.exposure_data_strategy!r}."
            )

    # ------------------------------------------------------------------
    # The method: implemented by a subclass
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def factor_names(self) -> tuple[str, ...]:
        """The ``factor`` axis of both stores.

        Examples
        --------
        >>> model.factor_names[:2]
        ('country', 'industry_1')
        """

    @property
    @abstractmethod
    def exposure_names(self) -> tuple[str, ...]:
        """The outputs of the exposures factor the model reads (the estimation universe aside).

        Examples
        --------
        >>> model.exposure_names[-1]
        'industry'
        """

    @abstractmethod
    def exposure_matrix(self, exposures: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Return each symbol's exposures to ``factor_names`` and whether it has them all.

        Parameters
        ----------
        exposures : xr.Dataset
            The exposures factor's values at one bar, on ``symbol``.

        Returns
        -------
        matrix : np.ndarray
            ``[n_symbols, n_factors]``.
        covered : np.ndarray
            Booleans: the symbol has an exposure to every factor it needs;
            its row is not meaningful otherwise.

        Examples
        --------
        >>> matrix, covered = model.exposure_matrix(style.compute(day, day).isel(timestamp=0))
        >>> matrix.shape[1] == len(model.factor_names)
        True
        """

    def factor_groups(self) -> dict[str, str]:
        """Return each factor's group, one of ``FACTOR_GROUPS``, keyed by factor name.

        The factor attribution of a backtest (``quantlab.risk.attribution``)
        sums its factors' contributions and risk by group. By default every
        factor is a ``"style"``; a model with a country (market) factor or
        industry factors says so by overriding this method.

        Returns
        -------
        dict
            ``{factor: group}`` in ``factor_names`` order.

        Examples
        --------
        >>> set(model.factor_groups().values()) <= set(FACTOR_GROUPS)
        True
        >>> model.factor_groups()["country"]  # a Use4RiskModel
        'country'
        """
        return {name: "style" for name in self.factor_names}

    def factor_labels(self) -> dict[str, str]:
        """Return each factor's display name, keyed by factor name.

        The backtest report names factors by these labels (stored as the
        ``label`` coordinate of a backtest's factor attribution). By default
        a factor is labelled by its name; a model with coded factor names
        (``industry_34``) overrides this method.

        Returns
        -------
        dict
            ``{factor: label}`` in ``factor_names`` order.

        Examples
        --------
        >>> list(model.factor_labels()) == list(model.factor_names)
        True
        >>> model.factor_labels()["industry_34"]  # a Use4RiskModel
        'Business Services'
        """
        return {name: name for name in self.factor_names}

    @property
    @abstractmethod
    def regression_warmup_bars(self) -> int:
        """Bars before a range ``_compute_regression`` reads.

        Examples
        --------
        >>> model.regression_warmup_bars
        1
        """

    @property
    @abstractmethod
    def estimate_warmup_bars(self) -> int:
        """Regression-store bars before a range ``_compute_estimate`` reads.

        Examples
        --------
        >>> model.estimate_warmup_bars  # Use4RiskModel's default
        1637
        """

    @abstractmethod
    def _compute_regression(self, start, end) -> xr.Dataset:
        """Return the regression rows from ``start`` to ``end`` (``REGRESSION_VARIABLES``)."""

    @abstractmethod
    def _compute_estimate(self, start, end) -> xr.Dataset:
        """Return the estimate rows from ``start`` to ``end`` (``ESTIMATE_VARIABLES``)."""

    # ------------------------------------------------------------------
    # The stores
    # ------------------------------------------------------------------

    @property
    def regression(self) -> RiskStore:
        """The regression store: realized factor and specific returns (see the class docstring).

        Examples
        --------
        >>> model.regression.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        return RiskStore(
            self,
            "regression",
            self.config.regression_path,
            lambda start, end: self._checked(
                self._compute_regression(start, end), REGRESSION_VARIABLES, "regression"
            ),
            warmup_bars=self.regression_warmup_bars,
        )

    @property
    def estimate(self) -> RiskStore:
        """The estimate store: forecast factor covariance and specific risk (see the class docstring).

        Examples
        --------
        >>> model.estimate.build("2018-01-01", "2024-12-31")
        >>> rows = model.estimate.read("2024-12-31", "2024-12-31")
        >>> rows["factor_covariance"].dims
        ('timestamp', 'factor_i', 'factor_j')
        """
        return RiskStore(
            self,
            "estimate",
            self.config.estimate_path,
            lambda start, end: self._checked(
                self._compute_estimate(start, end), ESTIMATE_VARIABLES, "estimate"
            ),
            warmup_bars=self.estimate_warmup_bars,
        )

    def _checked(self, rows: xr.Dataset, variables: dict, store: str) -> xr.Dataset:
        """Return ``rows``, or raise ``TypeError`` when they break the store's contract."""
        names = list(self.factor_names)
        for name, dims in variables.items():
            if name not in rows.data_vars or rows[name].dims != dims:
                raise TypeError(
                    f"{self.class_name}: its {store} rows must hold {name!r} on {dims}; "
                    f"got {rows[name].dims if name in rows.data_vars else 'nothing'}."
                )
            for dim in dims:
                if dim.startswith("factor") and rows[dim].values.tolist() != names:
                    raise TypeError(
                        f"{self.class_name}: the {dim!r} axis of its {store} rows must be "
                        f"factor_names."
                    )
        return rows

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def prices(self, start: Date, end: Date) -> xr.Dataset:
        """Return price, market cap and risk-free rate on the bars with a price.

        With ``risk_free_symbol`` the rate is that symbol's, forward-filled
        and broadcast across the others, and the symbol is dropped.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range, both inclusive.

        Returns
        -------
        xr.Dataset
            ``price_column``, ``market_cap_column`` and ``risk_free_column``
            on ``(timestamp, symbol)``, only the bars where some symbol has a
            price.

        Examples
        --------
        >>> model.prices("2024-01-02", "2024-01-31")[model.config.market_cap_column].dims
        ('timestamp', 'symbol')
        """
        config = self.config
        dataset = config.dataset
        columns = [config.price_column, config.market_cap_column, config.risk_free_column]
        panel = dataset.to_shared_names(
            dataset.panel(start, end, variables=dataset.own_names(columns))
        )
        if config.risk_free_symbol is not None:
            symbol = config.risk_free_symbol
            if symbol not in panel["symbol"].values.tolist():
                raise ValueError(
                    f"{self.class_name}: risk_free_symbol {symbol!r} is not a symbol of "
                    f"the dataset."
                )
            rate = panel[config.risk_free_column].sel(symbol=symbol, drop=True).ffill(
                "timestamp"
            )
            panel = panel.drop_sel(symbol=[symbol])
            panel = panel.assign(
                {config.risk_free_column: rate.broadcast_like(panel[config.price_column])}
            )
        priced = panel[config.price_column].notnull().any("symbol")
        return panel.sel(timestamp=priced)

    def exposures(self, start: Date, end: Date) -> xr.Dataset:
        """Return the exposure variables from ``start`` to ``end``, read or computed.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range, both inclusive.

        Returns
        -------
        xr.Dataset
            The exposures factor's ``exposure_names`` and, when the config
            names one, its estimation-universe flag, read from its store or
            computed per ``exposure_data_strategy``.

        Examples
        --------
        >>> list(model.exposures("2024-01-02", "2024-01-31").data_vars)[-2:]
        ['industry', 'estu']
        """
        config = self.config
        factor = config.exposures
        if config.exposure_data_strategy == "read":
            panel = factor.read(start, end)
        else:
            panel = factor.compute(start, end)
        estu = [config.estu_name] if config.estu_name is not None else []
        return panel[[*self.exposure_names, *estu]]
