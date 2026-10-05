"""The root class of the risk layer: ``FactorRiskModel``, a factor risk model's stores.

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

``FactorRiskModel`` is any factor risk model: its config names the exposures
(a factor's outputs: continuous style exposures, optionally an industry code
and an estimation-universe flag) and whether there is a country factor. Its
regression store holds, for every bar ``t``, the cross-sectional regression
of the excess returns of ``t`` on the exposures of ``t-1``:

- the excess return of a symbol is its adjusted close of ``t`` over that of
  ``t-1``, less one, less the risk-free rate of ``t-1`` (the rate of a day is
  published the next business day; ``BarraStyle`` lags it the same way);
- exposures: 1 on the country factor, 1 on the symbol's industry and 0 on
  every other industry, and its style exposures;
- weighted least squares over the estimation universe of ``t-1`` (every
  symbol without one), weighted by ``config.weighting`` of the market cap of
  ``t-1``; with both a country factor and industries, subject to the
  cap-weighted industry factor returns summing to 0, which removes their
  collinearity and makes the country factor the cap-weighted market (USE4
  Methodology Notes eq. 3.3);
- an industry with fewer than ``min_industry_members`` fitted members is left
  out of the bar: it has no factor return that bar, its members are not in
  the fit, and their specific returns carry no industry term (our choice);
- a fitted return more than ``return_outlier_sigma`` robust standard
  deviations (1.4826 times the median absolute deviation) from the
  cross-sectional median is trimmed to that bound for the fit only. A robust
  bound does not move when the outlier grows, so a vendor price error cannot
  move a factor return (our choice);
- the specific return of every symbol with all exposures and a return, fitted
  or not, is its untrimmed excess return less the fitted factor part.

Shipped models with their own factor sets live in ``quantlab.risk.predefined``
(USE4 on the ``BarraStyle`` exposures).
"""

import dataclasses
import datetime
import json
import warnings
from pathlib import Path
from typing import Callable, Self

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.core.component import Component
from quantlab.dataset.base import InsufficientHistoryError
from quantlab.dataset.merged import MergedDataset
from quantlab.risk.config import REGRESSION_WEIGHTINGS, FactorRiskConfig
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.date_range import as_label, check_range, last_moment, range_text
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: A date a store is asked for.
Date = str | datetime.date | pd.Timestamp


class RiskStore:
    """One store of a factor risk model: rows per bar, written and read by date range.

    Built by the risk model that owns it, from a function computing the rows
    of a date range (its warm-up included). A row may carry a ``symbol``
    axis; it is written sorted (numerically for integer symbols), and only
    symbols with a value somewhere in the rows are kept, so the axis does not
    depend on how a range was split between ``build`` and ``extend``.

    Parameters
    ----------
    owner : str
        The ``Class.store`` named in log and error messages.
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
    21
    """

    #: Appended to ``path`` to name the JSON file recording the store's range.
    RANGE_SUFFIX = ".range.json"

    def __init__(
        self,
        owner: str,
        path: str | None,
        compute: Callable[[object, object], xr.Dataset],
        warmup_bars: int,
    ):
        """Initialize the store; see the class docstring for parameters."""
        self.owner = owner
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
        recorded beside it (``store_range``).

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
        return xr.open_zarr(path).sel(timestamp=slice(as_label(start), as_label(end)))

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


#: Scales a median absolute deviation to the standard deviation of a normal.
MAD_TO_SIGMA = 1.4826


def _excess_returns(price: np.ndarray, risk_free: np.ndarray) -> np.ndarray:
    """Return ``[T, S]`` excess returns; row 0 is NaN (no previous bar)."""
    excess = np.full(price.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        excess[1:] = price[1:] / price[:-1] - 1.0 - risk_free[:-1]
    return excess


class FactorRiskModel(Component):
    """A factor risk model: the stores estimated ahead of a backtest.

    See the module docstring for the regression. Any exposures fit: the
    config names them. A subclass in ``quantlab.risk.predefined`` fixes a
    factor set through its config's defaults (``Use4RiskModel``).

    The regression store (``regression``) holds:

    - ``factor_return`` on ``(timestamp, factor)``: the factor returns, NaN
      for an industry left out of the bar and on a bar without a regression;
    - ``specific_return`` on ``(timestamp, symbol)``;
    - ``r_squared`` on ``timestamp``: the weighted R-squared of the fit,
      ``1 - sum(w e^2) / sum(w y^2)`` over the trimmed returns ``y`` (our
      choice: uncentred, as the country factor plays the intercept);
    - ``estu_count`` on ``timestamp``: the symbols in the fit;
    - ``industry_members`` on ``(timestamp, industry)``: estimation-universe
      symbols of each industry with exposures and a return, before thin
      industries are left out;
    - ``industry_excluded`` on ``(timestamp, industry)``: whether the
      industry was left out of the bar.

    Its warm-up is one bar: row ``t`` reads the exposures and the price of
    the previous priced bar. A bar has no regression (all NaN, a count of 0)
    when it has no more fitted symbols than factors to fit.

    Parameters
    ----------
    config : FactorRiskConfig
        The exposures factor, the price dataset and the regression's
        parameters; an instance of the class's ``config_cls``.

    Raises
    ------
    TypeError
        If ``config`` is not an instance of ``config_cls``.
    ValueError
        If a parameter is invalid.

    Examples
    --------
    With ``signals`` a factor outputting ``value`` and ``quality`` and
    ``prices`` a dataset with ``adjClose``, ``marketcap`` and ``risk_free``:

    >>> model = FactorRiskModel(FactorRiskConfig(
    ...     exposures=signals, dataset=prices, exposure_data_strategy="cal",
    ...     style_names=("value", "quality"),
    ...     regression_path="risk/two_style_regression.zarr",
    ... ))
    >>> model.factor_names
    ('country', 'value', 'quality')
    >>> model.regression.build("2012-01-01", "2024-12-31")
    >>> model.regression.read("2020-01-01", "2020-12-31")["factor_return"].dims
    ('timestamp', 'factor')
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
        'FactorRiskModel'
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
        'quantlab.risk.base.FactorRiskModel'
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
        """Raise ``ValueError`` for a parameter the regression cannot use."""
        owner = self.class_name
        if config.exposure_data_strategy not in ("read", "cal"):
            raise ValueError(
                f"{owner}: exposure_data_strategy must be 'read' or 'cal', got "
                f"{config.exposure_data_strategy!r}."
            )
        if len(set(config.style_names)) != len(config.style_names):
            raise ValueError(f"{owner}: style_names must be distinct.")
        if config.industry_name is None and config.industries:
            raise ValueError(f"{owner}: industries need an industry_name.")
        if config.industry_name is not None and not config.industries:
            raise ValueError(f"{owner}: industry_name needs the industries.")
        if len(set(config.industries)) != len(config.industries):
            raise ValueError(f"{owner}: industries must be distinct.")
        if not (config.country or config.industries or config.style_names):
            raise ValueError(f"{owner}: the model has no factor.")
        if config.weighting not in REGRESSION_WEIGHTINGS:
            raise ValueError(
                f"{owner}: weighting must be one of {REGRESSION_WEIGHTINGS}, got "
                f"{config.weighting!r}."
            )
        if config.min_industry_members < 1:
            raise ValueError(f"{owner}: min_industry_members must be at least 1.")
        if not config.return_outlier_sigma > 0:
            raise ValueError(f"{owner}: return_outlier_sigma must be positive.")

    @property
    def factor_names(self) -> tuple[str, ...]:
        """The ``factor`` axis: ``country``, ``industry_<code>`` per industry, styles.

        Examples
        --------
        >>> model.factor_names
        ('country', 'value', 'quality')
        """
        config = self.config
        country = ("country",) if config.country else ()
        industries = tuple(f"industry_{code}" for code in config.industries)
        return (*country, *industries, *config.style_names)

    @property
    def regression(self) -> RiskStore:
        """The regression store (see the class docstring).

        Examples
        --------
        >>> model.regression.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        return RiskStore(
            f"{self.class_name}.regression",
            self.config.regression_path,
            self._compute_regression,
            warmup_bars=1,
        )

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _previous_priced_bar(self, first: pd.Timestamp) -> pd.Timestamp | None:
        """Return the last bar before ``first`` with a price, or ``None``.

        Counts back one bar on the dataset's calendar, further when the bars
        found carry no price (a merged rate series' own days).
        """
        dataset = self.config.dataset
        back = 1
        while True:
            try:
                candidate, exhausted = dataset.bar_before(first, back), False
            except InsufficientHistoryError as exc:
                if exc.available == 0:
                    return None
                candidate, exhausted = dataset.bar_before(first, exc.available), True
            prices = self._prices(candidate, first - pd.Timedelta(1, "ns"))
            if prices.sizes["timestamp"]:
                return pd.Timestamp(prices["timestamp"].values[-1])
            if exhausted:
                return None
            back *= 2

    def _prices(self, start, end) -> xr.Dataset:
        """Return price, market cap and risk-free rate on the bars with a price.

        With ``risk_free_symbol`` the rate is that symbol's, forward-filled
        and broadcast across the others, and the symbol is dropped.
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

    def _exposures(self, start, end) -> xr.Dataset:
        """Return the exposure variables from ``start`` to ``end``, read or computed."""
        config = self.config
        factor = config.exposures
        if config.exposure_data_strategy == "read":
            panel = factor.read(start, end)
        else:
            panel = factor.compute(start, end)
        names = [config.industry_name, config.estu_name]
        return panel[[*config.style_names, *(name for name in names if name is not None)]]

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def _compute_regression(self, start, end) -> xr.Dataset:
        """Return the regression rows of the priced bars from ``start`` to ``end``."""
        config = self.config
        first, _ = check_range(start, end, f"{self.class_name}.regression.compute()")
        previous = self._previous_priced_bar(first)
        if previous is None:
            warnings.warn(
                f"{self.class_name}.regression.compute(): no priced bar before "
                f"{start!r}; the first bar has no regression.",
                UserWarning,
                stacklevel=3,
            )
        read_from = first if previous is None else previous
        prices = self._prices(read_from, end)
        exposures = self._exposures(read_from, end)
        symbols = sort_symbol_axis(
            set(prices["symbol"].values.tolist()) & set(exposures["symbol"].values.tolist())
        )
        bars = prices["timestamp"].values
        prices = prices.sel(symbol=symbols)
        exposures = exposures.reindex(timestamp=bars, symbol=symbols)

        def values(panel, name):
            return panel[name].transpose("timestamp", "symbol").values.astype(np.float64)

        shape = (len(bars), len(symbols))
        excess = _excess_returns(
            values(prices, config.price_column), values(prices, config.risk_free_column)
        )
        cap = values(prices, config.market_cap_column)
        if config.style_names:
            styles = np.stack([values(exposures, n) for n in config.style_names], axis=-1)
        else:
            styles = np.zeros((*shape, 0))
        if config.industry_name is None:
            industry = np.full(shape, -1, dtype=np.int64)
        else:
            code_index = {float(code): j for j, code in enumerate(config.industries)}
            codes = values(exposures, config.industry_name)
            industry = np.vectorize(lambda c: code_index.get(c, -1), otypes=[np.int64])(codes)
        if config.estu_name is None:
            estu = np.ones(shape, dtype=bool)
        else:
            estu = values(exposures, config.estu_name) == 1.0

        rows = np.flatnonzero(bars >= first.to_datetime64())
        n_industries = len(config.industries)
        factor_return = np.full((len(rows), len(self.factor_names)), np.nan)
        specific = np.full((len(rows), len(symbols)), np.nan)
        r_squared = np.full(len(rows), np.nan)
        estu_count = np.zeros(len(rows), dtype=np.int64)
        members = np.zeros((len(rows), n_industries), dtype=np.int64)
        excluded = np.ones((len(rows), n_industries), dtype=bool)
        with Timer(f"{self.class_name}: regression"):
            for row, i in enumerate(rows):
                if i == 0:
                    continue
                fit = self._regress_bar(
                    excess[i], styles[i - 1], industry[i - 1], estu[i - 1], cap[i - 1]
                )
                factor_return[row], specific[row] = fit["factor_return"], fit["specific"]
                r_squared[row], estu_count[row] = fit["r_squared"], fit["count"]
                members[row], excluded[row] = fit["members"], fit["excluded"]
        return xr.Dataset(
            {
                "factor_return": (("timestamp", "factor"), factor_return),
                "specific_return": (("timestamp", "symbol"), specific),
                "r_squared": (("timestamp",), r_squared),
                "estu_count": (("timestamp",), estu_count),
                "industry_members": (("timestamp", "industry"), members),
                "industry_excluded": (("timestamp", "industry"), excluded),
            },
            coords={
                "timestamp": bars[rows],
                "factor": list(self.factor_names),
                "symbol": symbols,
                "industry": list(config.industries),
            },
        )

    def _regress_bar(
        self,
        excess: np.ndarray,
        styles: np.ndarray,
        industry: np.ndarray,
        estu: np.ndarray,
        cap: np.ndarray,
    ) -> dict:
        """Fit one bar: excess returns of ``t`` on the exposures of ``t-1``.

        ``industry`` holds each symbol's position in ``config.industries``, -1
        for none, and ``estu`` whether it is in the estimation universe.
        Returns the factor returns over every factor, the specific returns
        over every symbol and the diagnostics.
        """
        config = self.config
        n_industries = len(config.industries)
        has_industry = config.industry_name is not None
        covered = np.isfinite(styles).all(axis=1) & ((industry >= 0) | (not has_industry))
        candidates = covered & estu & np.isfinite(excess) & np.isfinite(cap) & (cap > 0)
        slot = np.where(industry >= 0, industry, 0)
        members = np.bincount(industry[candidates & (industry >= 0)], minlength=n_industries)
        kept = members >= config.min_industry_members
        fit = candidates & (kept[slot] if has_industry else True)
        kept_index = np.flatnonzero(kept)
        result = {
            "factor_return": np.full(len(self.factor_names), np.nan),
            "specific": np.full(len(excess), np.nan),
            "r_squared": np.nan,
            "count": 0,
            "members": members,
            "excluded": ~kept,
        }

        # Industry dummies of the kept industries. With a country factor the
        # last one is expressed through the others, f_last = -sum_j (c_j /
        # c_last) f_j, so the cap-weighted industry factor returns sum to 0.
        dummies = (industry[fit][:, None] == kept_index[None, :]).astype(np.float64)
        ratio = np.zeros(0)
        restricted = dummies
        if config.country and len(kept_index):
            industry_cap = cap[fit] @ dummies
            ratio = industry_cap[:-1] / industry_cap[-1]
            restricted = dummies[:, :-1] - dummies[:, -1:] * ratio[None, :]
        country = [np.ones(fit.sum())] if config.country else []
        design = np.column_stack([*country, restricted, styles[fit]])
        if fit.sum() <= design.shape[1]:
            return result

        y = excess[fit]
        median = np.median(y)
        bound = config.return_outlier_sigma * MAD_TO_SIGMA * np.median(np.abs(y - median))
        # More than half the returns equal (stale prices) leaves no spread to
        # measure outliers by; nothing is trimmed then.
        if bound > 0:
            y = np.clip(y, median - bound, median + bound)
        weight = {
            "sqrt_cap": np.sqrt(cap[fit]),
            "cap": cap[fit],
            "equal": np.ones(fit.sum()),
        }[config.weighting]
        root = np.sqrt(weight)
        solution = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]

        n_country, n_free = len(country), restricted.shape[1]
        industry_returns = np.full(n_industries, np.nan)
        if len(kept_index):
            free = solution[n_country : n_country + n_free]
            industry_returns[kept_index] = (
                np.append(free, -ratio @ free) if config.country else free
            )
        style_returns = solution[n_country + n_free :]
        country_return = solution[:n_country]
        factor_return = np.concatenate([country_return, industry_returns, style_returns])

        residual = y - design @ solution
        with np.errstate(divide="ignore", invalid="ignore"):
            r_squared = 1.0 - (weight * residual**2).sum() / (weight * y**2).sum()

        # Specific returns of every covered symbol with a return, from the
        # untrimmed return; a left-out industry contributes nothing.
        fitted = styles @ style_returns + (country_return.sum() if config.country else 0.0)
        if has_industry:
            fitted = fitted + np.nan_to_num(industry_returns)[slot]
        specific = np.where(covered & np.isfinite(excess), excess - fitted, np.nan)
        result.update(
            factor_return=factor_return, specific=specific, r_squared=r_squared,
            count=int(fit.sum()),
        )
        return result
