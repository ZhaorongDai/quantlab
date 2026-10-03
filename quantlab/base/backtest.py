"""Backtester base class and the result types every backtest run produces.

This module sits at the end of the pipeline: a trained return model and a
price dataset go in, and a run directory holding target weights, an equity
curve, metrics and an HTML report comes out. ``BaseBacktester`` owns the
three public entry points, ``run()`` (backtest one model), ``run_cv()``
(replay every fold of a cross-validation run as one continuous curve) and
``run_weights()`` (backtest a precomputed target-weight panel, no model),
and every step between them that does not depend on the simulation engine, plus
``report_figure()``, the report chart of a result held in memory. Engine
layers such as ``VectorBtBacktester`` implement the simulation hooks, and
concrete classes add a ``MarketSpec`` and a signal generator.

A few terms are used throughout. A *panel* is an ``xarray.Dataset`` indexed
by ``timestamp`` and ``symbol``, and a *bar* is one timestamp of it. *Target
weights* are the fraction of portfolio value each symbol should hold after
a rebalance. The *warm-up* is the stretch of bars before the backtest window
that factors need to fill their rolling windows. *In-sample* bars are bars
the model was trained on (including the bars its labels looked ahead into),
and *out-of-sample* bars are bars it never saw; results are reported for
both separately. A *fold* is one train/test split of a walk-forward
cross-validation run (``train_cv``), and the *stitched* curve simulates the
test segments of all folds back to back. A *fingerprint* is a hash of the
data a run read, stored so that a later rebuild of the run can tell whether
the data has changed.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Protocol, Self, get_protocol_members

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.base.data import MarketDataset
from quantlab.base.model import BaseModel
from quantlab.base.portfolio import LabelSpec, PredictionPanel
from quantlab.base.tracking import TrackingRun
from quantlab.backend import XrBackend
# Importing this submodule also runs the `crsp` package `__init__` (the CRSP
# converter and polars), which adds about a second of import time.
from quantlab.dataset.crsp.tickers import CrspTickerLookup
from quantlab.enums.constant import Date
from quantlab.utils import backtest_stats
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.backtest_report import (
    backtest_report_figure,
    report_chart_inputs,
    report_portfolio_inputs,
    report_summary,
    report_windows,
    write_backtest_report,
)
from quantlab.utils.fingerprint import dataset_fingerprint
from quantlab.utils.jsonable import to_jsonable
from quantlab.utils.split import in_sample_window, split_ranges
from quantlab.utils.timer import Timer

from .config import BacktestConfig, FactorConfig, ForwardConfig

#: The config fields holding datasets a run directory records, each asked
#: through ``persist_with_run`` what a rebuild needs written beside the run.
RUN_DATASET_FIELDS = ("price_dataset", "benchmark_dataset")

#: Fields of a data fingerprint that are compared against the expected run;
#: any difference logs a warning.
FINGERPRINT_COMPARED_FIELDS = ("digest", "start", "end", "n_timestamps", "n_symbols")

#: Tail of every fingerprint warning emitted after a failed run, in place of
#: the usual "continuing". Log readers and tests find partial comparisons by
#: the substring ``"comparison is PARTIAL"``, so keep it when rewording.
FINGERPRINT_PARTIAL_NOTE = (
    "this comparison is PARTIAL: the run failed before it finished reading, so "
    "a differing digest/start/end/n_timestamps may reflect the interrupted read "
    "(under run_cv, a single fold's window) rather than a data change; the "
    "original error follows"
)

class Predictor(Protocol):
    """What the backtester needs from a model: the whole contract between the two.

    ``BacktestConfig.model`` is any object with these members; ``BaseModel``
    has them without inheriting this class, and so can an ensemble that
    composes several models. The backtester reads no model config and calls
    no other model method, so a predictor built from models with different
    configs is backtested unchanged. A config whose ``model`` lacks a member
    is refused at construction.

    Attributes
    ----------
    labels : list
        The label objects, whose ``lookahead_bars()`` extends the effective
        training window of the in-sample split and whose variable names are
        the prediction variables.
    train_bounds, test_bounds : tuple
        The ``(start, end)`` training and test windows: as configured, or,
        after ``load``, as the checkpoint records them.
    fitted_train_bounds : tuple
        The ``(start, end)`` training window actually fitted, after the
        purge; known after ``train`` or ``load``. The in-sample split starts
        from it.
    label_delays : tuple[int, ...]
        Each label's ``delay`` in bars, in the order of ``labels``; each must
        equal the engine's ``fill_delay_bars``.
    label_scales : dict[str, str]
        Each label name's prediction scale: ``"raw"`` when the prediction is
        in the label's own units, ``"standardized"`` when it only ranks the
        cross-section (a model fitted on a transformed target, a label an
        ensemble averages). A rule that needs return units checks it.

    Methods
    -------
    predict_window(start, end)
        The predictions for every bar from ``start`` to ``end``, one variable
        per label on ``(timestamp, symbol)``; the predictor requests its own
        features, warm-up included.
    fingerprint_inputs(start, end), training_fingerprint_inputs()
        The data ``predict_window`` and ``collect`` read, as
        ``(key, factor or label, strategy, first, last)`` entries. The
        backtester hashes ``factor.read(first, last)`` for strategy
        ``"read"`` and the dataset inputs of ``factor.compute(first, last)``
        otherwise, and records the result under ``key``.
    collect(), train()
        Train-mode preparation; ``train`` returns the checkpoint it wrote.
    load(path), check_checkpoint(path)
        Load-mode preparation; ``check_checkpoint`` validates a checkpoint
        without loading it and runs first.
    get_config(), from_config(config)
        A JSON-ready dict naming the class in ``"name"``, and the class
        method that rebuilds the predictor from it.

    Examples
    --------
    >>> from typing import get_protocol_members
    >>> sorted(get_protocol_members(Predictor))[:4]
    ['check_checkpoint', 'collect', 'fingerprint_inputs', 'from_config']
    >>> from quantlab.base.model import BaseModel
    >>> all(hasattr(BaseModel, name) for name in get_protocol_members(Predictor))
    True
    """

    @property
    def labels(self) -> list: ...

    @property
    def train_bounds(self) -> tuple: ...

    @property
    def test_bounds(self) -> tuple: ...

    @property
    def fitted_train_bounds(self) -> tuple: ...

    @property
    def label_delays(self) -> tuple[int, ...]: ...

    @property
    def label_scales(self) -> dict[str, str]: ...

    def predict_window(self, start, end) -> xr.Dataset: ...

    def fingerprint_inputs(self, start, end) -> list[tuple]: ...

    def training_fingerprint_inputs(self) -> list[tuple]: ...

    def collect(self): ...

    def train(self) -> Path: ...

    def load(self, path): ...

    def check_checkpoint(self, path): ...

    def get_config(self) -> dict: ...

    @classmethod
    def from_config(cls, config: dict) -> Self: ...


def label_specs(predictor: Predictor) -> tuple[LabelSpec, ...]:
    """Return the ``LabelSpec`` of every prediction variable of ``predictor``.

    One spec per variable name of each label, in the order of ``labels``:
    the name, its scale from ``label_scales``, the label's delay from
    ``label_delays`` and its ``span_bars()``, or ``None`` for a label
    without one (it is not a ``Forward`` label). These are what a portfolio
    construction rule is bound to and what a run's ``predictions.zarr``
    records.

    Parameters
    ----------
    predictor : Predictor
        The model or ensemble whose labels are described.

    Returns
    -------
    tuple[LabelSpec, ...]
        The specs, in the order of the prediction variables.

    Raises
    ------
    ValueError
        If ``label_scales`` has no entry for a label variable.

    Examples
    --------
    With ``model`` a model predicting the 5-bar forward return ``ret_5``:

    >>> label_specs(model)
    (LabelSpec(name='ret_5', scale='raw', delay=1, span=5),)
    """
    scales = dict(predictor.label_scales)
    specs = []
    for label, delay in zip(predictor.labels, predictor.label_delays, strict=True):
        span_bars = getattr(label, "span_bars", None)
        span = None if span_bars is None else int(span_bars())
        for name in label.get_factor_names():
            if name not in scales:
                raise ValueError(f"the predictor reports no label_scales entry for {name!r}")
            specs.append(
                LabelSpec(name=str(name), scale=scales[name], delay=int(delay), span=span)
            )
    return tuple(specs)


@dataclass(frozen=True)
class MarketSpec:
    """Backtest conventions of one market: price columns and annualization.

    Column names live only on a market's spec instance; backtester methods
    read them from ``self.MARKET`` and never spell them out, so supporting a
    new market means writing a new spec rather than changing the base class.

    Parameters
    ----------
    fill_price_column : str
        Price variable that orders execute at, for example the open.
    valuation_price_column : str
        Price variable the portfolio is marked to at the end of each bar,
        for example the close.
    trading_days_per_year : int
        Trading days in a year, used to annualize daily statistics.
    session_minutes_per_day : int
        Length of one trading session in minutes, used to annualize
        intraday statistics.

    Examples
    --------
    >>> spec = MarketSpec(
    ...     fill_price_column="open",
    ...     valuation_price_column="close",
    ...     trading_days_per_year=252,
    ...     session_minutes_per_day=390,
    ... )
    >>> spec.year_freq("1D")
    Timedelta('252 days 00:00:00')
    """

    fill_price_column: str
    valuation_price_column: str
    trading_days_per_year: int
    session_minutes_per_day: int

    def year_freq(self, bar_interval) -> pd.Timedelta:
        """Return one year in vectorbt's convention for bars of ``bar_interval``.

        ``quantlab.utils.backtest_stats.year_freq`` with this market's
        trading days and session minutes; its docstring gives the rule.

        Parameters
        ----------
        bar_interval
            Anything ``pd.Timedelta`` accepts, such as
            ``"1D"``, ``"5min"`` or a ``numpy.timedelta64``.

        Returns
        -------
        pd.Timedelta
            The year length as a ``pd.Timedelta``.

        Raises
        ------
        ValueError
            If ``bar_interval`` is not positive.

        Examples
        --------
        >>> spec.year_freq("1min") / pd.Timedelta("1min")
        98280.0
        >>> spec.year_freq("1D") / pd.Timedelta("1D")
        252.0
        >>> round(spec.year_freq("7D") / pd.Timedelta("7D"), 2)
        52.18
        """
        return backtest_stats.year_freq(
            bar_interval, self.trading_days_per_year, self.session_minutes_per_day
        )


@dataclass
class SimulationResult:
    """Output of one engine simulation in engine-independent form.

    ``value`` and ``returns`` are the portfolio value and per-bar returns on
    the ``timestamp`` dimension. ``orders`` is a dataset on an ``order``
    dimension with the variables ``timestamp``, ``symbol``, ``size``,
    ``price``, ``fees`` and ``side``. ``trades`` is a dataset on a ``trade``
    dimension with ``symbol``, ``entry_timestamp``, ``exit_timestamp``,
    ``pnl``, ``return`` and ``status`` (``"Open"`` or ``"Closed"``), empty
    when nothing traded. ``settlements`` records delisted holdings turned
    into cash at their last valuation (``symbol``, ``axis_symbol``,
    ``delisting_timestamp``, ``settlement_timestamp``, ``price``).
    ``rejected_orders`` records orders that found no fill price, or no
    price to size them against (``symbol``, ``axis_symbol``,
    ``signal_timestamp``, ``fill_timestamp``); the holding was kept.
    ``max_target_deviation`` is the largest absolute gap between a target
    weight and the weight held right after its fill bar, rejections, cash
    and fees included, with the weight valued at the prices the order was
    sized against (the fill prices, or the signal bar's valuation prices
    for ``sizing_basis="valuation"``); ``None`` when nothing rebalanced. ``native`` is the engine's own result object and is read
    only by the engine that produced it.

    Examples
    --------
    >>> sim = result.simulation  # from ``BaseBacktester.run()``
    >>> sim.value.dims, int(sim.value.values[0])
    (('timestamp',), 1000000)
    >>> list(sim.orders.data_vars)
    ['timestamp', 'symbol', 'size', 'price', 'fees', 'side']
    >>> sim.bar_interval
    np.timedelta64(86400000000000,'ns')
    """

    value: xr.DataArray
    returns: xr.DataArray
    orders: xr.Dataset
    settlements: list[dict]
    bar_interval: np.timedelta64
    trades: xr.Dataset | None = None
    native: object | None = None
    rejected_orders: list[dict] = field(default_factory=list)
    max_target_deviation: float | None = None


@dataclass
class BacktestResult:
    """Return value of ``BaseBacktester.run()`` and ``BaseBacktester.run_weights()``.

    ``run_dir`` is the directory this run wrote its artifacts to, or ``None``
    when ``config.output_dir`` is ``None`` and nothing was written.
    ``predictions`` and ``weights`` are panels on ``(timestamp, symbol)``
    covering exactly the backtest window; ``predictions`` is ``None`` for a
    ``run_weights()`` run, which has no model. ``metrics`` is the same
    mapping written to ``metrics.json``. ``benchmark`` is the buy-and-hold
    simulation of ``config.benchmark_dataset`` on the same bars, or ``None``
    when no benchmark is configured.

    Examples
    --------
    >>> result = backtester.run()
    >>> sorted(p.name for p in result.run_dir.iterdir())
    ['config.json', 'equity.zarr', 'fingerprint.json', 'metrics.json',
     'predictions.zarr', 'report.html', 'settlements.json', 'weights.zarr']
    >>> sorted(result.metrics)
    ['execution', 'in_sample', 'in_sample_range', 'notes', 'out_of_sample',
     'out_of_sample_ranges', 'portfolio_construction', 'training_window', 'whole']
    """

    run_dir: Path | None
    predictions: xr.Dataset | None
    weights: xr.Dataset
    simulation: SimulationResult
    metrics: dict = field(default_factory=dict)
    benchmark: SimulationResult | None = None


@dataclass
class _BacktestWindow:
    """One backtested window before persistence.

    Shared by ``run()``, each ``run_cv()`` fold and ``run_weights()``, which
    has no ``predictions`` and no ``split``.
    """

    predictions: xr.Dataset | None
    prices: xr.Dataset
    weights: xr.Dataset
    simulation: SimulationResult
    split: dict | None
    metrics: dict
    benchmark: SimulationResult | None = None


@dataclass
class CVBacktestResult:
    """Return value of ``BaseBacktester.run_cv()``.

    ``run_dir`` is ``None`` when ``config.output_dir`` is ``None``.
    ``folds`` holds one record per replayed fold: the manifest fields
    (``fold``, the four dates, ``checkpoint``) plus that fold's own
    ``predictions``, ``weights``, ``simulation`` and ``metrics`` from an
    independent per-fold simulation. ``weights`` are built in one pass over
    the concatenated fold predictions, as one account whose holdings carry
    across fold boundaries, and ``simulation`` is the single continuous
    simulation of them. ``metrics`` mirrors ``metrics.json`` with the keys ``stitched``,
    ``folds`` and ``notes``. ``benchmark`` is the buy-and-hold benchmark
    simulated over the stitched span, or ``None`` without a benchmark; each
    fold record also carries its own ``benchmark``.

    Examples
    --------
    >>> cv = backtester.run_cv()
    >>> len(cv.folds), sorted(cv.metrics)
    (8, ['folds', 'notes', 'stitched'])
    >>> sorted(cv.metrics["stitched"])
    ['in_sample', 'in_sample_ranges', 'out_of_sample', 'out_of_sample_ranges',
     'training_windows', 'whole']
    """

    run_dir: Path | None
    folds: list[dict]
    weights: xr.Dataset
    simulation: SimulationResult
    metrics: dict = field(default_factory=dict)
    benchmark: SimulationResult | None = None

class BaseBacktester(ABC):
    """Abstract base of every backtester: the public entry points and shared steps.

    The engine varies by inheritance and the market and selection logic by
    composition. The hierarchy is ``BaseBacktester`` (this class), then an
    engine layer such as ``VectorBtBacktester`` that implements
    ``_simulate``, ``_simulate_benchmark`` and ``_engine_stats`` and sets
    ``fill_delay_bars``, then a named concrete class that composes a
    ``MarketSpec`` (the ``MARKET`` class attribute) and a signal generator
    (``_generate_signals``) onto that engine. Concrete classes also set
    ``config_cls``, the config class the ``config`` setter accepts.

    The public entry points ``run()``, ``run_cv()`` and ``run_weights()``
    are defined here and never overridden. ``run()`` and ``run_cv()`` run the
    same fixed sequence of steps and call the hooks above along the way:
    prepare the model, request the factor panels for the window and predict,
    generate signals, simulate, compute metrics, then write the run
    directory. ``run_weights()`` starts from given target weights, so it
    skips the model and ``_generate_signals`` and needs no model on the
    config.

    Parameters
    ----------
    config : BacktestConfig
        The backtest configuration, an instance of ``config_cls``. It is
        validated and normalized on assignment; see the ``config`` setter.

    Attributes
    ----------
    MARKET : MarketSpec or None
        The market conventions. ``None`` on abstract classes; a concrete
        class must set it.
    fill_delay_bars : int
        Bars between the bar a weight forms on and the bar it fills on, set
        by the engine layer. ``run()`` and ``run_cv()`` refuse a model whose
        label ``delay`` differs from it, since the label would then measure
        a return the engine never trades.
    expected_fingerprint : dict or None
        Fingerprints of a previous run of the same config. When set, each
        run compares the data it reads against them and warns on a
        difference. It is filled in when a run is rebuilt from its saved
        ``config.json``.

    Examples
    --------
    A concrete class over the vectorbt engine needs only three members::

        class EqualWeightBacktester(VectorBtBacktester):
            config_cls = BacktestConfig
            MARKET = MarketSpec("open", "close", 252, 390)

            def _generate_signals(self, predictions, prices):
                ...  # return a ``weight`` panel on (timestamp, symbol)

    >>> backtester = EqualWeightBacktester(config)
    >>> result = backtester.run()
    """

    MARKET: MarketSpec | None = None
    fill_delay_bars: ClassVar[int]

    def __init__(self, config: BacktestConfig):
        """Initialize the backtester; see the class docstring for parameters."""
        # Created before the config is assigned, because the setter and the
        # validation hook may read them.
        self.expected_fingerprint: dict | None = None
        self._fingerprints: dict = {}
        # Absolute path of the checkpoint a train-mode run() produced; None in
        # load mode and before any run.
        self._trained_checkpoint: str | None = None
        # Built on first use by the ticker_lookup property.
        self._ticker_lookup: "CrspTickerLookup | None" = None
        # The benchmark's symbol-axis label, set by `_load_benchmark_prices`.
        self._benchmark_axis_symbol: str | None = None
        self.config = config

    @property
    def ticker_lookup(self) -> CrspTickerLookup | None:
        """Lookup that turns symbol ids into readable ticker names.

        It reads the ``.crsp_tickers.json`` file stored next to the price
        store. For CRSP data (the Center for Research in Security Prices),
        symbols are PERMNOs, permanent numeric security ids, and this file
        records which ticker each PERMNO traded under on each date. The
        backtester is the only layer that knows where the price store is, so
        it owns the lookup: the engine uses it for settlement and
        rejected-order records and
        the model for its lists of missing or extra symbols. The lookup is
        built on first access and reset whenever a new config is assigned.
        When no such file exists, ``label()`` returns each symbol unchanged,
        so panels from other vendors are unaffected. The price dataset names
        the store to look beside (``ticker_store()``); a dataset for which no
        sidecar can apply (a ``FrameDataset``, even one read back from a run
        directory) names none: the property is ``None`` and
        ``_symbol_labels`` shows its symbols as they are.

        Examples
        --------
        >>> from datetime import date
        >>> backtester.ticker_lookup.label(["AAA", "BBB"], date(2024, 3, 1))
        ['AAA', 'BBB']
        """
        path = self.config.price_dataset.ticker_store()
        if path is None:
            return None
        if self._ticker_lookup is None:
            self._ticker_lookup = CrspTickerLookup.beside_store(path)
        return self._ticker_lookup

    def _symbol_labels(self, symbols, day) -> list[str]:
        """Return readable labels of price-dataset ``symbols`` as of ``day``.

        Through ``ticker_lookup`` when the price dataset has a store, and the
        symbols themselves (as ``str``) for a dataset held in memory.
        """
        lookup = self.ticker_lookup
        if lookup is None:
            return [str(symbol) for symbol in symbols]
        return lookup.label(symbols, day)

    @property
    @abstractmethod
    def config_cls(self) -> type:
        """The config class this backtester accepts.

        Concrete classes satisfy it with a plain class attribute; the
        ``config`` setter checks ``isinstance(config, config_cls)`` first.

        Examples
        --------
        >>> class MyBacktester(VectorBtBacktester):
        ...     config_cls = BacktestConfig
        ...     MARKET = MarketSpec("open", "close", 252, 390)
        ...     def _generate_signals(self, predictions, prices): ...
        """

    @property
    def config(self) -> BacktestConfig:
        """The validated config this backtester was built with.

        Examples
        --------
        >>> backtester.config.start_date, backtester.config.rebalance_periods
        ('2024-02-12', 5)
        """
        return self._config

    @config.setter
    def config(self, config: BacktestConfig):
        """Validate ``config``, normalize its paths and store it.

        The type check is the first statement, before any other validation,
        so a wrong config class fails with a message naming the expected
        class. ``model`` and ``model_mode`` are both set or both ``None`` (a
        config for ``run_weights()`` only, which ``run()`` and ``run_cv()``
        refuse when called); a half-set pair is refused here.
        ``model_mode="load"`` needs at least one of ``checkpoint`` (used by
        ``run()``) and ``cv_project_dir`` (used by ``run_cv()``);
        whichever entry point is called later rejects the missing one. The
        ``checkpoint``, ``cv_project_dir`` and ``output_dir`` fields are
        rewritten as absolute paths so a saved ``config.json`` rebuilds the
        same run from any working directory. ``config.name`` is set to this
        class's import path and ``_validate_config`` runs last.

        Raises
        ------
        TypeError
            If ``config`` is not a ``config_cls``, ``MARKET`` is
            unset, ``config.model`` lacks a ``Predictor`` member, or
            ``config.benchmark_dataset`` is neither ``None`` nor a
            ``MarketDataset``.
        ValueError
            If ``model_mode`` (other than ``None``), only one of ``model``
            and ``model_mode`` being set, the load-mode paths,
            ``rebalance_periods``, ``fees``, ``slippage``, ``init_cash``,
            ``sizing_basis`` or the date order are invalid.

        Examples
        --------
        >>> backtester.config = config
        >>> backtester.config.name
        mypkg.backtest.MyBacktester
        >>> MyBacktester(object())
        Traceback (most recent call last):
        TypeError: MyBacktester requires a BacktestConfig, got object
        """
        # Keep the type check first, so a wrong config class gets a clear error.
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} requires a {self.config_cls.__name__}, "
                f"got {type(config).__name__}"
            )
        if self.MARKET is None:
            raise TypeError(
                f"{self.class_name} declares no MARKET spec; a concrete "
                f"backtester must set the MARKET class attribute"
            )
        # Checked on the class first, so a property is not evaluated here. A
        # config without a model serves run_weights() only; run() and
        # run_cv() refuse it when called.
        missing = sorted(
            name
            for name in get_protocol_members(Predictor)
            if config.model is not None
            and not (hasattr(type(config.model), name) or hasattr(config.model, name))
        )
        if missing:
            raise TypeError(
                f"{self.class_name}: config.model must implement the Predictor "
                f"protocol (quantlab.base.backtest.Predictor), but "
                f"{type(config.model).__name__} lacks {missing}"
            )

        if config.model_mode not in ("train", "load", None):
            raise ValueError(
                f"{self.class_name}: model_mode must be 'train', 'load' or None, "
                f"got {config.model_mode!r}"
            )
        # Both set (run(), run_cv()) or both None (run_weights() only); a
        # half-set pair is a mistake better caught now than at run time.
        if (config.model is None) != (config.model_mode is None):
            model_name = "None" if config.model is None else type(config.model).__name__
            raise ValueError(
                f"{self.class_name}: model and model_mode must be both set or "
                f"both None, got model={model_name} and "
                f"model_mode={config.model_mode!r}"
            )
        # run() needs the checkpoint and run_cv() needs cv_project_dir; each
        # entry point rejects its own missing field when called.
        if (
            config.model_mode == "load"
            and config.checkpoint is None
            and config.cv_project_dir is None
        ):
            raise ValueError(
                f"{self.class_name}: model_mode='load' requires a checkpoint path "
                f"(for run()) or a cv_project_dir (for run_cv())"
            )
        if config.rebalance_periods < 1:
            raise ValueError(
                f"{self.class_name}: rebalance_periods must be >= 1, got "
                f"{config.rebalance_periods}"
            )
        if config.fees < 0 or config.slippage < 0:
            raise ValueError(
                f"{self.class_name}: fees and slippage must be >= 0, got "
                f"fees={config.fees}, slippage={config.slippage}"
            )
        if config.init_cash <= 0:
            raise ValueError(
                f"{self.class_name}: init_cash must be > 0, got {config.init_cash}"
            )
        if config.sizing_basis not in ("fill", "valuation"):
            raise ValueError(
                f"{self.class_name}: sizing_basis must be 'fill' or 'valuation', "
                f"got {config.sizing_basis!r}"
            )
        if pd.Timestamp(config.start_date) > pd.Timestamp(config.end_date):
            raise ValueError(
                f"{self.class_name}: start_date {config.start_date} is after "
                f"end_date {config.end_date}"
            )
        # Only the type is checked here: reading the store to count its
        # symbols would make construction do I/O. `_load_benchmark_prices`
        # refuses a panel that is not exactly one symbol when a run reads it.
        if config.benchmark_dataset is not None and not isinstance(
            config.benchmark_dataset, MarketDataset
        ):
            raise TypeError(
                f"{self.class_name}: config.benchmark_dataset must be a "
                f"MarketDataset holding a single symbol, got "
                f"{type(config.benchmark_dataset).__name__}"
            )

        # Store paths as absolute, because config.json may rebuild the run
        # from another working directory.
        for name in ("checkpoint", "cv_project_dir", "output_dir"):
            value = getattr(config, name)
            if value is not None:
                setattr(config, name, str(Path(value).absolute()))

        self._config = config
        self._config.name = self.import_path
        # The cached lookup describes the previous config's price store.
        self._ticker_lookup = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Run extra construction-time checks of a concrete class; a no-op here."""

    @property
    def class_name(self) -> str:
        """The class's short name, used as the prefix of every message it logs.

        Examples
        --------
        >>> backtester.class_name
        MyBacktester
        """
        return self.__class__.__name__

    @property
    def import_path(self) -> str:
        """The dotted import path recorded as ``config.name``.

        Examples
        --------
        >>> backtester.import_path
        mypkg.backtest.MyBacktester
        """
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def get_config(self) -> dict:
        """Return the scalar config fields plus the nested dataset and model configs.

        The whole config is never passed through ``asdict``: the live objects
        are replaced by their own ``get_config()`` output. After a run the
        mapping also carries ``data_fingerprint``, one fingerprint per dataset
        the run read, and after a train-mode run ``trained_checkpoint``, so a
        saved ``config.json`` can rebuild and replay the same run. Every
        mapping carries ``market``, the ``fill_price_column`` and
        ``valuation_price_column`` of ``MARKET``, so a tool reading a run
        directory (an executor such as quantlab-trader) learns the price
        columns without importing the backtester class. ``market``,
        ``data_fingerprint`` and ``trained_checkpoint`` are records, not
        config fields: ``load_backtester_from_config`` drops ``market`` and
        the rebuilt class supplies its own ``MARKET``. A run
        directory's ``config.json`` differs in one respect: a price or
        benchmark ``FrameDataset`` is recorded reading the copy of its panel
        under ``inputs/``, named relative to the run directory.

        Examples
        --------
        >>> cfg = backtester.get_config()
        >>> cfg["name"], cfg["model_mode"], cfg["rebalance_periods"]
        ('mypkg.backtest.MyBacktester', 'load', 5)
        >>> sorted(cfg["data_fingerprint"])  # present once run() has read
        ['factor[0]:PastReturnFactor', 'price_dataset']
        >>> cfg["market"]
        {'fill_price_column': 'open', 'valuation_price_column': 'close'}

        A config without a model (for ``run_weights()``) records ``None``:

        >>> import dataclasses
        >>> no_model = dataclasses.replace(
        ...     backtester.config, model=None, model_mode=None, checkpoint=None
        ... )
        >>> type(backtester)(no_model).get_config()["model"] is None
        True
        """
        cfg = self.config.to_dict()
        cfg["price_dataset"] = self.config.price_dataset.get_config()
        model = self.config.model
        cfg["model"] = None if model is None else model.get_config()
        cfg["benchmark_dataset"] = (
            None
            if self.config.benchmark_dataset is None
            else self.config.benchmark_dataset.get_config()
        )
        cfg["tracker"] = self.config.tracker.get_config()
        constructor = getattr(self.config, "constructor", None)
        if constructor is not None:
            cfg["constructor"] = constructor.get_config()
        # A record of the class's MARKET, so a reader of a run directory
        # learns the price columns without importing the backtester class.
        cfg["market"] = {
            "fill_price_column": self.MARKET.fill_price_column,  # type: ignore[union-attr]
            "valuation_price_column": self.MARKET.valuation_price_column,  # type: ignore[union-attr]
        }
        if self._fingerprints:
            cfg["data_fingerprint"] = dict(self._fingerprints)
        # After a train-mode run, record the checkpoint it produced so load
        # mode can replay exactly this model.
        if self._trained_checkpoint is not None:
            cfg["trained_checkpoint"] = self._trained_checkpoint
        return cfg

    @staticmethod
    def _iso_date(value) -> str:
        """Normalize any date-like value to an ISO ``YYYY-MM-DD`` string.

        Every date this module writes into a dataset or factor config goes
        through here: the config setters only normalize dates when a whole
        config is assigned, and downstream date comparisons are string
        comparisons. ``str(value)`` comes first because ``pd.Timestamp``
        rejects ``numpy.str_``, which is what fold manifests hold.
        """
        return pd.Timestamp(str(value)).strftime("%Y-%m-%d")

    @staticmethod
    def _bar_label(value) -> str:
        """Return the persisted label of a bar timestamp.

        A bar at midnight is written as an ISO date, any other bar as a full
        ISO timestamp, so daily labels stay dates while intraday range
        endpoints keep their time of day. Labels are read back by
        ``backtest_stats.label_ns`` and compared as exact timestamps, never
        by day. ``backtest_stats.bar_label``, the public form.
        """
        return backtest_stats.bar_label(value)

    @staticmethod
    def _slice_bound(value):
        """Return a training-window endpoint in the form the model layer slices with.

        The model trains on ``data.sel(timestamp=slice(train_start,
        train_end))``, and pandas interprets a string endpoint at its own
        resolution: ``"2024-05-17"`` includes the whole day while
        ``"2024-05-17T13:00"`` stops at 13:00. Strings are therefore passed
        through unchanged (as plain ``str``) and other values become
        ``pd.Timestamp``, so ``_window_split`` selects the same bars the
        model trained on.
        """
        if isinstance(value, str):
            return str(value)
        return pd.Timestamp(value)

    def run(self) -> BacktestResult:
        """Backtest one model over the configured window and write a run directory.

        Subclasses do not override this method. In train mode the
        model is trained on its own dates first; in load mode the checkpoint
        is restored and the training dates recorded beside it define the
        in-sample split (a warning is logged if they select different bars
        than ``config.model``'s dates). Each factor is computed (or read)
        for the window, warming itself up by its own ``warmup_bars``; no
        dataset, factor or label config is changed. The model predicts the
        window, the concrete class turns the predictions into target weights, the engine simulates them, metrics
        are computed for the whole window and for the in-sample and
        out-of-sample parts, and everything is written to a new directory
        under ``config.output_dir``. Data fingerprints are compared against
        ``expected_fingerprint`` when one is set, also on the failure path.

        Returns
        -------
        BacktestResult
            A ``BacktestResult`` with the run directory, the predictions and
            weights on the window bars, the simulation and the metrics.

        Raises
        ------
        ValueError
            If ``config.model`` is ``None``, ``model_mode="load"`` comes
            without ``config.checkpoint``, a label's ``delay`` differs from
            ``fill_delay_bars``, or the window has no price bars.

        Examples
        --------
        >>> backtester = MyBacktester(
        ...     BacktestConfig(
        ...         price_dataset=prices,
        ...         model=model,
        ...         model_mode="load",
        ...         checkpoint="models/head.joblib",
        ...         start_date="2024-02-12",
        ...         end_date="2024-03-11",
        ...         output_dir="runs",
        ...         rebalance_periods=5,
        ...     )
        ... )
        >>> result = backtester.run()
        >>> result.weights["weight"].dims, result.simulation.value.sizes
        (('timestamp', 'symbol'), Frozen({'timestamp': 21}))
        >>> result.metrics["out_of_sample_ranges"]
        [('2024-02-12', '2024-03-11')]
        """
        self._require_model("run()")
        if self.config.model_mode == "load" and self.config.checkpoint is None:
            raise ValueError(
                f"{self.class_name}: run() with model_mode='load' requires "
                f"config.checkpoint; cv_project_dir is read only by run_cv()"
            )
        self._check_label_delays()

        def _model_window(start_date: str, end_date: str) -> _BacktestWindow:
            """Prepare the model, then predict and backtest the window."""
            # `_prepare_model` runs inside `_run_window`'s failure guard on
            # purpose: in train mode it records the training-data
            # fingerprints before `model.train()`, which can fail for the
            # same data reasons. In load mode the model takes the training
            # dates its checkpoint records.
            configured = self._prepare_model()
            calendar = self._price_calendar(end_date)
            # The comparison with config.model's dates is bar-based, so it
            # needs the calendar.
            if self.config.model_mode == "load":
                self._warn_if_config_model_dates_differ(calendar, configured)
            return self._backtest_window(
                start_date,
                end_date,
                calendar,
                *self.config.model.fitted_train_bounds,
            )

        return self._run_window(_model_window)

    def run_cv(self) -> CVBacktestResult:
        """Replay a ``train_cv`` run fold by fold and simulate the stitched weights.

        Subclasses do not override this method. It reads the
        ``cv_folds.json`` manifest under ``config.cv_project_dir``, keeps the
        folds whose test segment lies inside the backtest window, and checks
        on the price calendar that those test segments are contiguous and
        non-overlapping before any model is loaded (a stitched curve with a
        gap or an overlap corresponds to no real trading path). Each fold is
        then backtested on its own test segment with its own checkpoint, and
        its in-sample split uses that fold's training dates. A label looks a
        few bars ahead (its *lookahead*), so the training labels of a fold
        already saw the bars after ``train_end``. ``train_cv`` purges those
        bars from every training window and records the purged ``train_end``,
        so a test bar counts as in-sample only if a label reads further than
        the purge removed. The loaded checkpoint's ``fitted_train_bounds``
        is compared with the manifest's purged window.

        The fold predictions are then concatenated and turned into weights
        in one pass of ``_generate_signals`` over the prices from the first
        ``test_start`` to the last ``test_end``, so the holdings a rule is
        handed, locked positions included, carry across fold boundaries and
        the rebalance schedule runs on from the first bar; the weights are
        simulated once, with capital carried across as well, and the
        ``stitched`` metrics record the pass's ``portfolio_construction``.
        Per-fold metrics still come from the independent per-fold backtests. Fingerprints cover the
        whole stitched window. The manifest's fold dates are authoritative;
        a checkpoint whose recorded training dates select different bars
        only logs a warning.

        Returns
        -------
        CVBacktestResult
            A ``CVBacktestResult`` with the run directory, the per-fold
            records, and the stitched weights, simulation and metrics.

        Raises
        ------
        ValueError
            If ``config.model`` is ``None``, ``config.cv_project_dir`` is
            unset, ``model_mode`` is not ``"load"``, a label's ``delay`` differs from
            ``fill_delay_bars``, the manifest is malformed, no fold falls
            inside the window, or the fold test segments are not
            contiguous.
        FileNotFoundError
            If the manifest or a fold checkpoint is missing.

        Examples
        --------
        >>> backtester = MyBacktester(
        ...     BacktestConfig(
        ...         price_dataset=prices,
        ...         model=model,
        ...         model_mode="load",
        ...         cv_project_dir="models/head_trial_20260925",
        ...         start_date="2024-02-12",
        ...         end_date="2024-04-17",
        ...         output_dir="runs",
        ...         rebalance_periods=2,
        ...     )
        ... )
        >>> cv = backtester.run_cv()
        >>> len(cv.folds), cv.weights.sizes
        (8, Frozen({'timestamp': 48, 'symbol': 6}))
        >>> cv.metrics["stitched"]["in_sample_ranges"]
        []
        """
        self._require_model("run_cv()")
        if self.config.cv_project_dir is None:
            raise ValueError(
                f"{self.class_name}: run_cv() requires config.cv_project_dir, the "
                f"train_cv project directory holding "
                f"{BaseModel.CV_FOLDS_FILENAME}"
            )
        if self.config.model_mode != "load":
            raise ValueError(
                f"{self.class_name}: run_cv() replays the checkpoints of an "
                f"existing train_cv run and requires model_mode='load', got "
                f"{self.config.model_mode!r}"
            )
        with self._tracking_run() as run:
            result = self._replay_cv()
            self._track(run, result.run_dir, result.metrics["stitched"])
        return result

    def _replay_cv(self) -> CVBacktestResult:
        """Backtest every fold, simulate the stitched weights and persist; see ``run_cv``."""
        self._check_label_delays()
        self._fingerprints = {}
        # run_cv only loads; never carry a checkpoint trained by an earlier run().
        self._trained_checkpoint = None

        folds = self._select_folds(self._read_cv_folds())
        calendar = self._price_calendar(folds[-1]["test_end"])
        self._assert_contiguous_folds(folds, calendar)

        records: list[dict] = []
        for fold in folds:
            # Resolved under the project directory, never the working
            # directory; the resolved path is what the records persist.
            fold["checkpoint"] = self._resolve_fold_checkpoint(fold["checkpoint"])
            self._load_model_checkpoint(fold["checkpoint"])
            # The manifest's dates are authoritative. The fitted window the
            # checkpoint records is only cross-checked against them, by the
            # bars they select on the calendar rather than by text.
            recorded = self.config.model.fitted_train_bounds
            manifest_bounds = fold["_train_bounds"]
            if not self._same_training_bars(calendar, recorded, manifest_bounds):
                logger.warning(
                    f"{self.class_name}: fold {fold['fold']} checkpoint "
                    f"{fold['checkpoint']} records training dates "
                    f"{recorded[0]}..{recorded[1]}, but the manifest says "
                    f"{manifest_bounds[0]}..{manifest_bounds[1]}; using the "
                    f"manifest's dates"
                )
            # Only the fold window records fingerprints, so only its failure
            # triggers the partial comparison; checkpoint errors above do not.
            try:
                window = self._backtest_window(
                    fold["test_start"],
                    fold["test_end"],
                    calendar,
                    *fold["_train_bounds"],
                )
            except Exception:
                self._compare_fingerprints_on_failure()
                raise
            records.append(
                {
                    **{key: fold[key] for key in self._CV_RECORD_KEYS},
                    "predictions": window.predictions,
                    "weights": window.weights,
                    "simulation": window.simulation,
                    "benchmark": window.benchmark,
                    "metrics": window.metrics,
                }
            )

        # One account over the whole span: the folds' predictions are
        # concatenated and turned into weights in one pass, so the current
        # weights carry across fold boundaries, and simulated once, so the
        # capital does too.
        first_start = folds[0]["test_start"]
        last_end = folds[-1]["test_end"]
        stitched_predictions = xr.concat(
            [record["predictions"] for record in records], dim="timestamp"
        )

        # The per-fold loop left only the last fold's fingerprints. Take the
        # factor fingerprints over the whole stitched window (with the first
        # fold's warm-up) and the price fingerprint from the stitched prices,
        # then compare.
        self._fingerprints = {}
        try:
            self._record_fingerprint_entries(
                self.config.model.fingerprint_inputs(first_start, last_end)
            )
            stitched_prices = self._load_prices(first_start, last_end)
            stitched_benchmark_prices = self._load_benchmark_prices(
                first_start, last_end, stitched_prices.timestamp.values
            )
        except Exception:
            self._compare_fingerprints_on_failure()
            raise
        self._compare_fingerprints()

        if not np.array_equal(
            stitched_predictions.timestamp.values.astype("datetime64[ns]"),
            stitched_prices.timestamp.values.astype("datetime64[ns]"),
        ):
            raise ValueError(
                f"{self.class_name}: the concatenated fold predictions do not cover "
                f"exactly the price bars {first_start}..{last_end}"
            )
        stitched_predictions = stitched_predictions.reindex(
            symbol=stitched_prices.symbol.values
        )
        stitched_weights = self._generate_signals(stitched_predictions, stitched_prices)
        self._assert_weights_contract(stitched_weights, stitched_prices)
        stitched_simulation = self._simulate(stitched_weights, stitched_prices)
        stitched_benchmark = (
            None
            if stitched_benchmark_prices is None
            else self._simulate_benchmark(stitched_benchmark_prices)
        )
        stitched_metrics = self._compute_metrics(
            stitched_simulation,
            stitched_benchmark,
            self._stitched_split(stitched_prices.timestamp.values, records),
        )
        stitched_metrics.update(self._signal_metrics())

        notes = self._report_notes() + [
            f"run_cv: the stitched curve is one continuous simulation over folds "
            f"{[fold['fold'] for fold in folds]} ({first_start}..{last_end}), "
            f"capital and holdings carried across fold boundaries; per-fold "
            f"metrics come from "
            f"separate per-fold simulations. Each fold's first label-lookahead "
            f"bars are in-sample (metrics stitched.in_sample_ranges) and are not "
            f"shaded in this report."
        ]
        metrics = {
            "stitched": stitched_metrics,
            "folds": [
                {
                    **{key: record[key] for key in self._CV_RECORD_KEYS},
                    "metrics": record["metrics"],
                }
                for record in records
            ],
            "notes": notes,
        }
        run_dir = self._persist_cv(
            records,
            stitched_weights,
            stitched_simulation,
            metrics,
            benchmark=stitched_benchmark,
            predictions=stitched_predictions,
        )

        return CVBacktestResult(
            run_dir=run_dir,
            folds=records,
            weights=stitched_weights,
            simulation=stitched_simulation,
            metrics=metrics,
            benchmark=stitched_benchmark,
        )

    def run_weights(self, weights: xr.Dataset | xr.DataArray) -> BacktestResult:
        """Backtest a precomputed target-weight panel over the configured window.

        Subclasses do not override this method. No model is involved:
        ``config.model`` and ``config.model_mode`` are ignored and may both
        be ``None``, and ``_generate_signals`` is not called. The fill and
        valuation prices of ``config.price_dataset`` are read for the window
        ``config.start_date``..``config.end_date``, the weights are checked
        against the target-weight contract on exactly those bars, and the
        engine simulates them (a weight formed at bar t fills at bar t+1's
        fill price). The benchmark, when configured, is simulated and
        compared as in ``run()``. There is no training window, so the metrics
        cover the whole window only: ``whole`` (plus ``benchmark`` and
        ``relative``, each with a ``whole`` block, when a benchmark ran) and
        ``notes``, with no in-sample or out-of-sample block. The run
        directory is written like ``run()``'s when ``config.output_dir`` is
        set; the data fingerprints cover the prices and the benchmark.

        Parameters
        ----------
        weights : xarray.Dataset or xarray.DataArray
            Target weights on ``(timestamp, symbol)``, in either axis order.
            A dataset must carry a ``weight`` variable; a data array is used
            whatever its name. A run directory's ``weights.zarr``, read with
            ``XrBackend().read(path).data``, replays that run. The
            timestamps must be exactly the price bars
            of the window and the symbols exactly the price dataset's
            symbols, in any order (they are aligned to the price axes). A
            NaN keeps the symbol's holding and a finite value is its target;
            the targets of a row have a gross exposure, the sum of their
            absolute values, of at most 1.

        Returns
        -------
        BacktestResult
            The run directory (``None`` without ``output_dir``), the weights
            aligned to the price axes, the simulation, the metrics and the
            benchmark simulation; ``predictions`` is ``None``.

        Raises
        ------
        ValueError
            If the window has no price bars, the weights are not on
            ``(timestamp, symbol)``, their bars or symbols differ from the
            prices' (naming the first missing or extra ones), or a row breaks
            the contract (naming the offending bar).

        Examples
        --------
        With ``prices`` a price dataset and ``weights`` a weight panel on its
        30 bars from 2024-02-12 to 2024-03-22 (here the weights of an
        earlier ``run()``):

        >>> backtester = USEquityCrossectionSelectStockVectorBt(
        ...     CrossSectionBacktestConfig(
        ...         price_dataset=prices,
        ...         start_date="2024-02-12",
        ...         end_date="2024-03-22",
        ...         output_dir=None,
        ...         rebalance_periods=5,
        ...         constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        ...     )
        ... )
        >>> result = backtester.run_weights(weights)
        >>> result.run_dir is None, result.predictions is None
        (True, True)
        >>> sorted(result.metrics), result.simulation.value.sizes
        (['execution', 'notes', 'whole'], Frozen({'timestamp': 30}))
        """
        return self._run_window(
            lambda start_date, end_date: self._weights_window(
                weights, start_date, end_date
            ),
            notes=(
                "run_weights: the target weights were given, not predicted by a "
                "model, so there is no training window and the metrics cover "
                "the whole window only.",
            ),
        )

    def report_figure(self, result: BacktestResult):
        """Return the chart of ``report.html`` for ``result`` as a plotly figure.

        The same figure a run directory's report embeds (equity, drawdown and
        monthly returns, the in-sample range shaded, the deepest drawdown
        marked, the benchmark rows when a benchmark ran), built from the
        result in memory, so a run kept with ``output_dir=None`` can be
        looked at too. ``result`` must come from ``run()`` or
        ``run_weights()`` of a backtester with this config, which is checked
        without reading any data: its bars lie inside the configured window,
        its curve starts at ``init_cash``, and it has a benchmark curve exactly
        when a benchmark is configured. A ``run_cv()`` result is refused: its
        stitched curve is drawn in the ``report.html`` of its run directory.

        Parameters
        ----------
        result : BacktestResult
            The result to draw.

        Returns
        -------
        plotly.graph_objects.Figure
            The figure, not yet shown or written.

        Raises
        ------
        TypeError
            If ``result`` is a ``CVBacktestResult`` or not a ``BacktestResult``.
        ValueError
            If ``result`` does not match this config (see above), naming what
            differs.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from quantlab.backtest.predefined.weights import WeightsVectorBt
        >>> from quantlab.base.config import WeightsBacktestConfig
        >>> from quantlab.dataset.memory import FrameDataset
        >>> bars = pd.bdate_range("2024-01-01", periods=5)
        >>> prices = FrameDataset(pd.DataFrame({
        ...     "timestamp": np.repeat(bars, 2), "symbol": ["AAA", "BBB"] * 5,
        ...     "open": np.linspace(10.0, 14.0, 10), "close": np.linspace(10.5, 14.5, 10),
        ... }))
        >>> backtester = WeightsVectorBt(WeightsBacktestConfig(
        ...     price_dataset=prices, start_date="2024-01-01", end_date="2024-01-05",
        ...     output_dir=None, rebalance_periods=1,
        ...     fill_price_column="open", valuation_price_column="close",
        ...     trading_days_per_year=252, session_minutes_per_day=390,
        ... ))
        >>> weights = xr.DataArray(
        ...     [[0.5, 0.5]] + [[np.nan, np.nan]] * 4, dims=("timestamp", "symbol"),
        ...     coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
        ... )
        >>> result = backtester.run_weights(weights)
        >>> figure = backtester.report_figure(result)
        >>> [trace.name for trace in figure.data][:2]
        ['equity', 'drawdown']
        """
        if isinstance(result, CVBacktestResult):
            raise TypeError(
                f"{self.class_name}.report_figure() draws one run() or "
                f"run_weights() result, got a run_cv() result; its stitched curve "
                f"is drawn in the report.html of its run directory"
            )
        if not isinstance(result, BacktestResult):
            raise TypeError(
                f"{self.class_name}.report_figure() takes the BacktestResult of "
                f"run() or run_weights(), got {type(result).__name__}"
            )
        self._check_result_matches_config(result)
        return backtest_report_figure(
            result.simulation.value,
            **self._report_chart_inputs(
                result.simulation, result.metrics, result.benchmark
            ),
        )

    def _check_result_matches_config(self, result: BacktestResult) -> None:
        """Refuse a result another config produced; reads no data.

        Raises
        ------
        ValueError
            If the result's bars leave the configured window, its curve does
            not start at ``init_cash``, or it has a benchmark curve without a
            configured benchmark or the other way round.
        """
        timestamps = result.simulation.value.timestamp.values
        start = backtest_stats.label_ns(self._iso_date(self.config.start_date))
        end = backtest_stats.label_ns(self._iso_date(self.config.end_date)) + np.timedelta64(1, "D")
        if timestamps.size and (timestamps[0] < start or timestamps[-1] >= end):
            raise ValueError(
                f"{self.class_name}: the result covers {self._bar_label(timestamps[0])} "
                f"..{self._bar_label(timestamps[-1])}, outside this config's window "
                f"{self._iso_date(self.config.start_date)}.."
                f"{self._iso_date(self.config.end_date)}; draw a result with the "
                f"backtester that produced it"
            )
        first_value = float(result.simulation.value.values[0]) if timestamps.size else None
        if first_value is not None and not np.isclose(first_value, self.config.init_cash):
            raise ValueError(
                f"{self.class_name}: the result starts at {first_value}, not this "
                f"config's init_cash {self.config.init_cash}"
            )
        configured = self.config.benchmark_dataset is not None
        if configured != (result.benchmark is not None):
            raise ValueError(
                f"{self.class_name}: the result has "
                f"{'a' if result.benchmark is not None else 'no'} benchmark curve but "
                f"this config has {'a' if configured else 'no'} benchmark_dataset"
            )

    def _run_window(
        self, backtest_window, notes: tuple[str, ...] = ()
    ) -> BacktestResult:
        """Run one backtest window and persist it; shared by ``run()`` and ``run_weights()``.

        ``backtest_window(start_date, end_date)`` computes the window from
        the config's ISO dates without persisting anything. Per-run state is
        reset first, so a second run on the same object starts clean. The
        data fingerprints are compared after the window, and also on its
        failure path, since fingerprints may already have been recorded and
        differ when it raises. ``notes`` are appended to the default report
        notes. The window runs inside its tracking run (see
        ``_tracking_run``); the run directory is written (unless
        ``output_dir`` is ``None``) and tracked (see ``_track``).
        """
        start_date = self._iso_date(self.config.start_date)
        end_date = self._iso_date(self.config.end_date)
        self._fingerprints = {}
        self._trained_checkpoint = None
        with self._tracking_run() as run:
            try:
                window = backtest_window(start_date, end_date)
            except Exception:
                self._compare_fingerprints_on_failure()
                raise
            # Outside the try (and not in a finally) so a clean run compares once.
            self._compare_fingerprints()

            metrics = window.metrics
            if self._trained_checkpoint is not None:
                metrics["trained_checkpoint"] = self._trained_checkpoint
            metrics["notes"] = self._report_notes() + list(notes)
            run_dir = self._report_and_persist(
                window.weights,
                window.simulation,
                metrics,
                benchmark=window.benchmark,
                predictions=window.predictions,
            )
            self._track(run, run_dir, metrics)

        return BacktestResult(
            run_dir=run_dir,
            predictions=window.predictions,
            weights=window.weights,
            simulation=window.simulation,
            metrics=metrics,
            benchmark=window.benchmark,
        )

    def _weights_window(
        self, weights: xr.Dataset | xr.DataArray, start_date: str, end_date: str
    ) -> _BacktestWindow:
        """Backtest given ``weights`` over the window, with no model and no split.

        Raises
        ------
        ValueError
            If the window has no price bars or the weights break the
            contract (see ``run_weights``).
        """
        prices = self._load_prices(start_date, end_date)
        if prices.sizes.get("timestamp", 0) == 0:
            raise ValueError(
                f"{self.class_name}: no price bars between {start_date} and "
                f"{end_date}"
            )
        benchmark_prices = self._load_benchmark_prices(
            start_date, end_date, prices.timestamp.values
        )
        weights = self._align_weights(weights, prices)
        self._assert_weights_contract(weights, prices)
        with Timer(f"{self.class_name}: simulate"):
            simulation = self._simulate(weights, prices)
        benchmark = (
            None
            if benchmark_prices is None
            else self._simulate_benchmark(benchmark_prices)
        )
        return _BacktestWindow(
            predictions=None,
            prices=prices,
            weights=weights,
            simulation=simulation,
            split=None,
            metrics=self._compute_metrics(simulation, benchmark, None),
            benchmark=benchmark,
        )

    def _require_model(self, entry: str) -> None:
        """Refuse to run ``entry`` on a config without a model.

        The config setter already guarantees ``model`` and ``model_mode`` are
        both set or both ``None``, so checking ``model`` covers both.

        Raises
        ------
        ValueError
            If ``config.model`` is ``None``.
        """
        if self.config.model is None:
            raise ValueError(
                f"{self.class_name}: {entry} requires config.model, but it is "
                f"None; set config.model and config.model_mode, or backtest "
                f"precomputed weights with run_weights()"
            )

    def _align_weights(
        self, weights: xr.Dataset | xr.DataArray, prices: xr.Dataset
    ) -> xr.Dataset:
        """Return ``weights`` as a ``weight`` panel on exactly the price axes.

        A data array becomes the ``weight`` variable whatever its name. The
        panel is transposed to ``(timestamp, symbol)`` and reordered onto the
        price bars and symbols. The two sets must match exactly, since a
        missing bar or symbol has no defined weight; the row contract itself
        is left to ``_assert_weights_contract``.

        Raises
        ------
        ValueError
            If there is no ``weight`` variable, the dimensions are not
            ``timestamp`` and ``symbol``, or the bars or symbols differ from
            the prices'; the message names the first few differences.
        """
        if isinstance(weights, xr.DataArray):
            weights = weights.rename("weight").to_dataset()
        if "weight" not in weights.data_vars:
            raise ValueError(
                f"{self.class_name}: weights must carry a 'weight' variable, got "
                f"{list(weights.data_vars)}"
            )
        weight = weights["weight"]
        if set(weight.dims) != {"timestamp", "symbol"}:
            raise ValueError(
                f"{self.class_name}: weight dims must be timestamp and symbol, "
                f"got {weight.dims}"
            )
        bars = prices.timestamp.values.astype("datetime64[ns]")
        symbols = prices.symbol.values.astype(str)
        weight = weight.transpose("timestamp", "symbol").assign_coords(
            timestamp=weight.timestamp.values.astype("datetime64[ns]"),
            symbol=weight.symbol.values.astype(str),
        )
        self._assert_same_axis("bars", weight.timestamp.values, bars)
        self._assert_same_axis("symbols", weight.symbol.values, symbols)
        weight = weight.reindex(timestamp=bars, symbol=symbols)
        return weight.to_dataset().assign_coords(
            timestamp=prices.timestamp.values, symbol=prices.symbol.values
        )

    def _assert_same_axis(
        self, axis: str, given: np.ndarray, wanted: np.ndarray
    ) -> None:
        """Raise unless ``given`` holds exactly the labels of ``wanted``, once each.

        ``axis`` names the axis in the message (``"bars"`` or
        ``"symbols"``); bar labels are written with ``_bar_label``.
        """

        def _show(values: np.ndarray) -> list[str]:
            """Format the first five labels of ``values`` for the message."""
            if axis == "bars":
                return [self._bar_label(value) for value in values[:5]]
            return [str(value) for value in values[:5]]

        missing = np.setdiff1d(wanted, given)
        extra = np.setdiff1d(given, wanted)
        duplicated = given.size - np.unique(given).size
        if missing.size or extra.size or duplicated:
            raise ValueError(
                f"{self.class_name}: the weight {axis} must be exactly the price "
                f"{axis} of the backtest window: {missing.size} missing "
                f"{_show(missing)}, {extra.size} extra {_show(extra)}, "
                f"{duplicated} duplicated"
            )

    #: Manifest fields shared by each fold record and the per-fold entries of
    #: metrics.json.
    _CV_RECORD_KEYS = (
        "fold",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "checkpoint",
    )

    def _read_cv_folds(self) -> list[dict]:
        """Read and validate the fold manifest under ``cv_project_dir``.

        The manifest is a persisted format, so an absent or unsupported
        ``format_version`` is refused rather than guessed at. Each fold must
        carry every ``_CV_RECORD_KEYS`` field and a test segment that does
        not end before it starts. The four dates are normalized with
        ``_iso_date`` on a copy of each entry, and the raw training endpoints
        are kept under ``_train_bounds`` for ``_window_split``, which
        slices the model layer's way and needs them at full resolution.

        Returns
        -------
        list[dict]
            The folds sorted by ``fold``.

        Raises
        ------
        FileNotFoundError
            If the manifest does not exist.
        ValueError
            If the format version is unsupported, ``folds`` is
            not a non-empty list, or a fold entry is invalid.
        """
        path = Path(self.config.cv_project_dir) / BaseModel.CV_FOLDS_FILENAME  # type: ignore[arg-type]
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.class_name}: CV fold manifest {path} does not exist; "
                f"cv_project_dir must be the project directory a train_cv run "
                f"wrote"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        supported = BaseModel.CV_FOLDS_FORMAT_VERSION
        if not isinstance(payload, dict) or "format_version" not in payload:
            raise ValueError(
                f"{self.class_name}: {path} has no format_version (supported: "
                f"{supported}); it is not a cv_folds manifest this reader "
                f"understands"
            )
        version = payload["format_version"]
        if isinstance(version, bool) or version != supported:
            raise ValueError(
                f"{self.class_name}: {path} format_version {version!r} is not "
                f"supported (supported: {supported}); older manifests are not "
                f"migrated, so rerun train_cv to write a current one"
            )
        raw_folds = payload.get("folds")
        if not isinstance(raw_folds, list):
            raise ValueError(
                f"{self.class_name}: {path} 'folds' must be a list, got "
                f"{type(raw_folds).__name__}"
            )
        if not raw_folds:
            raise ValueError(
                f"{self.class_name}: {path} lists no folds; the train_cv run "
                f"produced no fold to backtest"
            )

        folds = []
        for entry in raw_folds:
            missing = [
                key
                for key in self._CV_RECORD_KEYS
                if not isinstance(entry, dict) or key not in entry
            ]
            if missing:
                raise ValueError(
                    f"{self.class_name}: {path} fold entry "
                    f"{entry.get('fold') if isinstance(entry, dict) else entry!r} "
                    f"is missing {missing}"
                )
            fold = dict(entry)
            # Keep the training endpoints as written (nanosecond strings) for
            # `_window_split`: truncating them to dates would make the
            # whole train_end day count as training on intraday data.
            fold["_train_bounds"] = (entry["train_start"], entry["train_end"])
            for key in ("train_start", "train_end", "test_start", "test_end"):
                fold[key] = self._iso_date(fold[key])
            if fold["test_start"] > fold["test_end"]:
                raise ValueError(
                    f"{self.class_name}: {path} fold {fold['fold']} test segment "
                    f"starts {fold['test_start']} after it ends {fold['test_end']}"
                )
            folds.append(fold)
        return sorted(folds, key=lambda fold: fold["fold"])

    def _resolve_fold_checkpoint(self, recorded) -> str:
        """Resolve a manifest checkpoint entry to the file to load.

        ``train_cv`` lays checkpoints out as
        ``{cv_project_dir}/{experiment}/{model}``, so the entry's last two
        path components are first looked up under ``cv_project_dir``; this
        survives moving the project directory and relative manifest entries
        alike. Failing that, an absolute entry that exists is accepted.
        Relative entries are never resolved against the working directory,
        which could silently load a same-named checkpoint of another run.

        Raises
        ------
        FileNotFoundError
            If neither candidate exists; the message
            names both.
        """
        project_dir = Path(self.config.cv_project_dir)  # type: ignore[arg-type]
        recorded_path = Path(str(recorded))
        in_project = project_dir / recorded_path.parent.name / recorded_path.name
        if in_project.is_file():
            return str(in_project)
        if recorded_path.is_absolute() and recorded_path.is_file():
            return str(recorded_path)
        raise FileNotFoundError(
            f"{self.class_name}: fold checkpoint {recorded!r} was found neither "
            f"inside cv_project_dir as {in_project} nor as an existing absolute "
            f"path; relative manifest entries are never resolved against the "
            f"working directory"
        )

    def _select_folds(self, folds: list[dict]) -> list[dict]:
        """Keep the folds whose whole test segment lies inside the backtest window.

        ISO date strings compare in time order. An info line lists the
        selection when some folds are dropped.

        Raises
        ------
        ValueError
            If no fold remains; the message gives the window and
            the span of the manifest's test segments.
        """
        start = self._iso_date(self.config.start_date)
        end = self._iso_date(self.config.end_date)
        selected = [
            fold
            for fold in folds
            if fold["test_start"] >= start and fold["test_end"] <= end
        ]
        if not selected:
            raise ValueError(
                f"{self.class_name}: no fold's test segment lies within the "
                f"backtest window {start}..{end}; the manifest's test segments "
                f"span {folds[0]['test_start']}..{folds[-1]['test_end']}"
            )
        if len(selected) < len(folds):
            logger.info(
                f"{self.class_name}: run_cv backtests folds "
                f"{[fold['fold'] for fold in selected]} of "
                f"{[fold['fold'] for fold in folds]} (window {start}..{end})"
            )
        return selected

    def _assert_contiguous_folds(self, folds: list[dict], calendar: np.ndarray) -> None:
        """Check on the price calendar that the fold test segments abut exactly.

        A fold's first bar is the first calendar bar on or after
        ``test_start`` and its last bar the last one on ``test_end``'s day.
        Each fold must start exactly one bar after the previous one ends: a
        larger index means bars that belong to no fold and would be skipped
        by the stitched curve, a smaller one means bars traded by two
        models.

        Raises
        ------
        ValueError
            On a gap, an overlap, or a fold with no price bars;
            the message names the folds and dates involved.
        """
        cal = np.asarray(calendar).astype("datetime64[ns]")
        one_day = np.timedelta64(1, "D")

        def _span(fold: dict) -> tuple[int, int]:
            """Return the (first, last) calendar indices of ``fold``'s test segment."""
            day_start = np.datetime64(fold["test_start"], "D").astype("datetime64[ns]")
            day_after_end = (np.datetime64(fold["test_end"], "D") + one_day).astype(
                "datetime64[ns]"
            )
            first = int(np.searchsorted(cal, day_start, side="left"))
            last = int(np.searchsorted(cal, day_after_end, side="left")) - 1
            if first > last:
                raise ValueError(
                    f"{self.class_name}: fold {fold['fold']} test segment "
                    f"{fold['test_start']}..{fold['test_end']} has no price bars "
                    f"on the price calendar"
                )
            return first, last

        previous = folds[0]
        _, previous_last = _span(previous)
        for fold in folds[1:]:
            first, last = _span(fold)
            expected = previous_last + 1
            if first > expected:
                raise ValueError(
                    f"{self.class_name}: fold test segments are not contiguous: "
                    f"gap between fold {previous['fold']} ending "
                    f"{previous['test_end']} and fold {fold['fold']} starting "
                    f"{fold['test_start']}; {first - expected} price bar(s) in "
                    f"between belong to no fold, so a stitched out-of-sample "
                    f"curve would silently skip them"
                )
            if first < expected:
                raise ValueError(
                    f"{self.class_name}: fold test segments overlap: fold "
                    f"{fold['fold']} starts {fold['test_start']}, on or before "
                    f"fold {previous['fold']} ends {previous['test_end']}; "
                    f"{expected - first} price bar(s) would be traded by two "
                    f"models"
                )
            previous, previous_last = fold, last

    def _backtest_window(
        self,
        start_date: str,
        end_date: str,
        calendar: np.ndarray,
        train_start,
        train_end,
    ) -> _BacktestWindow:
        """Run every step of one backtest window without persisting anything.

        Shared by ``run()`` and by each fold of ``run_cv()``: request the
        factor panels for the window and predict, load the prices, reindex
        the predictions onto the price axes (symbols without a prediction
        become NaN and are never selected), split the window against the
        fitted training window ``[train_start, train_end]`` plus the labels'
        lookahead, generate and check the weights, simulate, simulate
        the benchmark (when one is configured, on the same bars) and compute
        the metrics. The model must already be prepared.

        Raises
        ------
        ValueError
            If the window contains no price bars.
        """
        predictions = self._predict_window(start_date, end_date)

        prices = self._load_prices(start_date, end_date)
        if prices.sizes.get("timestamp", 0) == 0:
            raise ValueError(
                f"{self.class_name}: no price bars between {start_date} and "
                f"{end_date}"
            )
        benchmark_prices = self._load_benchmark_prices(
            start_date, end_date, prices.timestamp.values
        )

        # Spread the predictions over every price symbol; a missing symbol is
        # NaN and therefore never selectable.
        predictions = predictions.reindex(
            timestamp=prices.timestamp.values, symbol=prices.symbol.values
        )

        split = self._window_split(
            prices.timestamp.values, calendar, train_start, train_end
        )

        weights = self._generate_signals(predictions, prices)
        self._assert_weights_contract(weights, prices)

        with Timer(f"{self.class_name}: simulate"):
            simulation = self._simulate(weights, prices)
        benchmark = (
            None
            if benchmark_prices is None
            else self._simulate_benchmark(benchmark_prices)
        )
        metrics = self._compute_metrics(simulation, benchmark, split)
        metrics.update(self._signal_metrics())
        return _BacktestWindow(
            predictions=predictions,
            prices=prices,
            weights=weights,
            simulation=simulation,
            split=split,
            metrics=metrics,
            benchmark=benchmark,
        )

    def _prepare_model(self) -> tuple:
        """Train or load the model and return its configured training window.

        The window is ``config.model``'s ``train_bounds`` before this call.
        In train mode the model is collected and trained on the dates in its
        own config; the backtest window never overwrites them, because it
        only decides the prediction span and the in-sample split. In load
        mode the checkpoint is restored, and the model takes the dates it
        records, since those are the dates the checkpoint was really trained
        on. Either way the model's ``fitted_train_bounds`` is then known.
        """
        model = self.config.model
        configured = model.train_bounds
        if self.config.model_mode == "load":
            self._load_model_checkpoint(self.config.checkpoint)
            return configured
        model.collect()
        # Fingerprint the training data right after collect() and before
        # train().
        self._record_fingerprint_entries(model.training_fingerprint_inputs())
        # The checkpoint train() wrote is recorded in config.json and metrics.
        self._trained_checkpoint = str(model.train())
        return configured

    def _warn_if_config_model_dates_differ(self, calendar, configured: tuple) -> None:
        """Warn when the checkpoint's training dates and ``config.model``'s disagree.

        Used by ``run()`` in load mode only, after the load. ``configured``
        is ``config.model``'s training window before the load, which may be
        stale or hand-written; the loaded model's ``train_bounds`` are the
        checkpoint's recorded dates, and the split uses them regardless. The
        warning names both pairs and the checkpoint path. Two pairs count as
        equal when they select the same bars on ``calendar``, so a different
        spelling of the same window does not warn.
        """
        recorded = self.config.model.train_bounds
        if self._same_training_bars(calendar, recorded, configured):
            return
        logger.warning(
            f"{self.class_name}: checkpoint {self.config.checkpoint} was trained on "
            f"{recorded[0]}..{recorded[1]} (its run.json), but config.model "
            f"says train_start={configured[0]!r}, train_end={configured[1]!r}; "
            f"using the checkpoint's dates for the effective training window "
            f"(the split between in-sample and out-of-sample bars)"
        )

    def _check_label_delays(self) -> None:
        """Refuse a label whose ``delay`` differs from the engine's ``fill_delay_bars``.

        Raises
        ------
        ValueError
            Naming the first such label, its delay and the fill delay.
        """
        model = self.config.model
        for i, (label, delay) in enumerate(zip(model.labels, model.label_delays)):
            if delay != self.fill_delay_bars:
                raise ValueError(
                    f"{self.class_name}: labels[{i}] {label.class_name} "
                    f"{label.get_factor_names()} has delay={delay}, "
                    f"but the engine fills a weight fill_delay_bars="
                    f"{self.fill_delay_bars} bar(s) after the bar it forms on; "
                    f"the model would learn a return the backtest never trades"
                )

    def _lookahead_bars(self) -> int:
        """Return L, the largest ``lookahead_bars()`` of the model's labels."""
        return max(
            (label.lookahead_bars() for label in self.config.model.labels),
            default=0,
        )

    @classmethod
    def _same_training_bars(cls, calendar, a: tuple, b: tuple) -> bool:
        """Return whether two ``(train_start, train_end)`` pairs select the same bars.

        Every training-date consistency check goes through here instead of
        comparing text: one instant can be spelled ``"2024-01-01"``,
        ``"2024-01-01T00:00:00"`` or as a nanosecond string, and plain
        ``pd.Timestamp`` equality is not enough either, because the model
        slices string endpoints at their own resolution (on intraday data
        ``"2024-02-09"`` includes the whole day while a midnight nanosecond
        string stops at the previous bar). The pairs are therefore run
        through the same pandas ``slice_indexer`` as ``_window_split``.

        Identical pairs are equal; a ``None`` endpoint in a non-identical
        pair makes them different; on an empty calendar the endpoints are
        compared as timestamps; otherwise the selected ``(start, stop)``
        positions must match and any endpoint lying outside the calendar
        must equal its counterpart as a timestamp, so two different
        ``train_end`` values past the calendar's last bar do not collapse
        into the same stop. Unparseable dates raise from pandas.
        """
        if tuple(a) == tuple(b):
            return True
        if any(value is None for value in (*a, *b)):
            return False
        a_ts = tuple(pd.Timestamp(str(value)) for value in a)
        b_ts = tuple(pd.Timestamp(str(value)) for value in b)
        bars = np.sort(np.asarray(calendar).astype("datetime64[ns]"))
        if bars.size == 0:
            return a_ts == b_ts

        index = pd.DatetimeIndex(bars)
        a_slice = index.slice_indexer(cls._slice_bound(a[0]), cls._slice_bound(a[1]))
        b_slice = index.slice_indexer(cls._slice_bound(b[0]), cls._slice_bound(b[1]))
        if (a_slice.start, a_slice.stop) != (b_slice.start, b_slice.stop):
            return False

        first, last = index[0], index[-1]
        for x, y in zip(a_ts, b_ts):
            outside = not (first <= x <= last) or not (first <= y <= last)
            if outside and x != y:
                return False
        return True

    def _load_model_checkpoint(self, checkpoint) -> None:
        """Load ``checkpoint`` into ``config.model``.

        Shared by ``run()`` and each fold of ``run_cv()``. The file's
        existence and the factor/label variable check
        (``model.check_checkpoint``) both run before any feature is
        computed, so a wrong path or a mismatched model fails cheaply.

        Raises
        ------
        FileNotFoundError
            If ``checkpoint`` does not exist.
        """
        model = self.config.model
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.class_name}: checkpoint {path} does not exist"
            )
        model.check_checkpoint(path)
        model.load(path)

    def _price_calendar(self, end_date: str) -> np.ndarray:
        """Return the price dataset's sorted bar timestamps up to ``end_date``.

        Used wherever the backtester counts bars on the strategy's calendar:
        the effective training window and the fold contiguity checks. Only
        the timestamps of a date-range request are read; the dataset's
        config is not touched.
        """
        panel = self.config.price_dataset.panel(Date.START_DATE, end_date)
        return np.sort(panel.timestamp.values)

    def _predict_window(self, start_date: str, end_date: str) -> xr.Dataset:
        """Fingerprint the model's inputs, then predict the window.

        The model requests its own features for ``start_date`` to
        ``end_date`` (``Predictor.predict_window``), warm-up included; no
        config is changed.
        """
        with Timer(f"{self.class_name}: predict_window"):
            model = self.config.model
            self._record_fingerprint_entries(
                model.fingerprint_inputs(start_date, end_date)
            )
            return model.predict_window(start_date, end_date)

    def _load_prices(self, start_date: str, end_date: str) -> xr.Dataset:
        """Return the fill and valuation price columns over the window.

        The columns come from a date-range request, which leaves the
        dataset untouched, so the price dataset may be the same object as a
        factor's dataset. The price fingerprint is recorded here.

        Raises
        ------
        ValueError
            If either price column is missing from the store.
        """
        dataset = self.config.price_dataset
        ds = dataset.panel(start_date, end_date)

        fill = self.MARKET.fill_price_column  # type: ignore[union-attr]
        valuation = self.MARKET.valuation_price_column  # type: ignore[union-attr]
        for column in (fill, valuation):
            if column not in ds.data_vars:
                raise ValueError(
                    f"{self.class_name}: price column {column!r} not found in "
                    f"{self._where(dataset)}"
                )
        prices = ds[[fill, valuation]].load()
        self._record_price_fingerprint(prices)
        return prices

    @staticmethod
    def _where(dataset: MarketDataset) -> str:
        """Name where ``dataset`` reads from, for messages: its store, or memory."""
        path = dataset.config.zarr_file_path
        if path is None:
            return f"the {type(dataset).__name__} held in memory"
        return str(path)

    def _record_price_fingerprint(self, prices: xr.Dataset) -> None:
        """Record the fingerprint of the two price columns under ``price_dataset``."""
        columns = [
            self.MARKET.fill_price_column,  # type: ignore[union-attr]
            self.MARKET.valuation_price_column,  # type: ignore[union-attr]
        ]
        self._fingerprints["price_dataset"] = dataset_fingerprint(prices, columns)

    def _load_benchmark_prices(
        self, start_date: str, end_date: str, timestamps: np.ndarray
    ) -> xr.Dataset | None:
        """Return the benchmark's price columns on the strategy's bars, or ``None``.

        ``config.benchmark_dataset`` is a market dataset like the price
        dataset, a ``(timestamp, symbol)`` panel, but it must hold exactly
        one symbol (an index ETF such as QQQ, in a store of its own). Its
        fill and valuation columns (the ``MARKET`` names, the same the
        strategy trades on) are read over the window and reindexed onto
        ``timestamps``, the strategy's price bars, so both curves are
        valued on the same bars. A benchmark bar the strategy calendar does
        not have is dropped; a strategy bar the benchmark lacks carries the
        benchmark's previous price forward and is counted in one warning.
        The fingerprint is recorded under ``benchmark_dataset``, over the
        data as read rather than as aligned.

        Raises
        ------
        ValueError
            If a price column is missing, the panel does not hold exactly
            one symbol, or the benchmark has no price on some bar of the
            window even after carrying prices forward (it starts after the
            window starts).
        """
        dataset = self.config.benchmark_dataset
        if dataset is None:
            return None
        ds = dataset.panel(start_date, end_date)

        fill = self.MARKET.fill_price_column  # type: ignore[union-attr]
        valuation = self.MARKET.valuation_price_column  # type: ignore[union-attr]
        for column in (fill, valuation):
            if column not in ds.data_vars:
                raise ValueError(
                    f"{self.class_name}: benchmark price column {column!r} not "
                    f"found in {self._where(dataset)}"
                )
        symbols = [str(symbol) for symbol in ds.symbol.values]
        if len(symbols) != 1:
            raise ValueError(
                f"{self.class_name}: the benchmark dataset must hold exactly one "
                f"symbol, got {len(symbols)} in "
                f"{self._where(dataset)}: {symbols[:10]}"
            )
        self._benchmark_axis_symbol = symbols[0]
        read = ds[[fill, valuation]].load()
        self._fingerprints["benchmark_dataset"] = dataset_fingerprint(
            read, [fill, valuation]
        )

        bars = np.asarray(timestamps).astype("datetime64[ns]")
        read = read.assign_coords(
            timestamp=read.timestamp.values.astype("datetime64[ns]")
        )
        aligned = read.reindex(timestamp=bars)
        gaps = int(
            (
                ~np.isfinite(aligned[fill].values)
                | ~np.isfinite(aligned[valuation].values)
            )
            .any(axis=1)
            .sum()
        )
        aligned = aligned.ffill("timestamp")
        unpriced = (
            ~np.isfinite(aligned[fill].values) | ~np.isfinite(aligned[valuation].values)
        ).any(axis=1)
        if unpriced.any():
            first = pd.Timestamp(bars[int(np.argmax(unpriced))])
            raise ValueError(
                f"{self.class_name}: benchmark {symbols[0]} has no price on or "
                f"before bar {first} of the window {start_date}..{end_date}; the "
                f"benchmark must cover the whole backtest window"
            )
        if gaps:
            logger.warning(
                f"{self.class_name}: benchmark {symbols[0]} has no price on "
                f"{gaps} of the {bars.size} strategy bars in "
                f"{start_date}..{end_date}; its previous price is carried "
                f"forward on those bars"
            )
        return aligned

    @staticmethod
    def _dataset_variables_fingerprint(factor, ds: xr.Dataset) -> dict:
        """Fingerprint the data a factor (or label) consumes from its dataset.

        ``ds`` is the dataset panel to fingerprint. A KunQuant factor
        (``FactorConfig``) reads ``data_columns``; a Polars factor consumes
        the whole frame, so every data variable is covered.
        """
        if isinstance(factor.config, ForwardConfig):
            return BaseBacktester._dataset_variables_fingerprint(factor.config.factor, ds)
        if isinstance(factor.config, FactorConfig):
            variables = list(factor.config.data_columns)
        else:
            variables = list(ds.data_vars)
        return dataset_fingerprint(ds, variables)

    @staticmethod
    def _store_fingerprint(ds: xr.Dataset) -> dict:
        """Fingerprint a panel read from a factor or label store, all variables."""
        return dataset_fingerprint(ds, list(ds.data_vars))

    def _record_fingerprint_entries(self, entries) -> None:
        """Hash the data a model reports it reads and record it by key.

        ``entries`` come from ``Predictor.fingerprint_inputs`` (the ``run()``
        window, each ``run_cv()`` fold and the stitched window) or
        ``Predictor.training_fingerprint_inputs`` (train mode, after
        ``collect()`` and before ``train()``, since the window fingerprints
        do not cover the training span). Each is
        ``(key, item, strategy, first, last)``: under ``"read"`` the panel
        ``item.read(first, last)`` is fingerprinted over all its variables,
        otherwise the variables ``item`` consumes from the dataset panel
        ``item.compute(first, last)`` reads, its warm-up bars included.
        """
        for key, item, strategy, first, last in entries:
            if strategy == "read":
                self._fingerprints[key] = self._store_fingerprint(item.read(first, last))
            else:
                self._fingerprints[key] = self._dataset_variables_fingerprint(
                    item, self._compute_inputs(item, first, last)
                )

    @classmethod
    def _compute_inputs(cls, item, start, end) -> xr.Dataset:
        """Return the dataset panel ``item.compute(start, end)`` reads.

        The range comes from the factor itself, so the fingerprint covers
        exactly the warm-up and resample padding ``compute`` reads. A label
        (``Forward``) computes its factor up to ``lookahead_bars()`` bars
        after ``end``, so its factor's inputs are fingerprinted over that
        later range.
        """
        if isinstance(item.config, ForwardConfig):
            return cls._compute_inputs(item.config.factor, start, item._later_end(end))
        return item.config.dataset.panel(*item._input_range(start, end, warn=False))

    def _compare_fingerprints(self, *, partial: bool = False) -> None:
        """Compare this run's fingerprints against ``expected_fingerprint``.

        Does nothing when ``expected_fingerprint`` is ``None``. A key present
        on one side only, or a key whose ``FINGERPRINT_COMPARED_FIELDS``
        differ, logs one warning naming the key and the fields. Datasets get
        appended to and adjusted prices get restated, so a rebuilt run must
        notice changed data, but changed data can still be backtested, so
        this never raises.

        With ``partial=True`` (the failure path) each warning ends with
        ``FINGERPRINT_PARTIAL_NOTE`` instead of ``"continuing"``, because
        the original error follows and an interrupted read may explain a
        difference, and keys expected but not yet read are skipped, since
        "not read yet" is not "not read".

        Parameters
        ----------
        partial : bool
            Whether this is the failure-path comparison.
        """
        expected = self.expected_fingerprint
        if expected is None:
            return
        tail = FINGERPRINT_PARTIAL_NOTE if partial else "continuing"
        actual = to_jsonable(self._fingerprints)
        for key in sorted(set(expected) | set(actual)):  # type: ignore[arg-type]
            if key not in actual:
                # On the failure path this only means "not read yet".
                if partial:
                    continue
                logger.warning(
                    f"{self.class_name}: data fingerprint mismatch for {key!r}: "
                    f"present in expected_fingerprint but not read by this run; "
                    f"{tail}"
                )
                continue
            if key not in expected:
                logger.warning(
                    f"{self.class_name}: data fingerprint mismatch for {key!r}: "
                    f"read by this run but absent from expected_fingerprint; "
                    f"{tail}"
                )
                continue
            wanted, got = expected[key], actual[key]
            differing = [
                name
                for name in FINGERPRINT_COMPARED_FIELDS
                if wanted.get(name) != got.get(name)
            ]
            if differing:
                details = "; ".join(
                    f"{name}: expected {wanted.get(name)!r}, got {got.get(name)!r}"
                    for name in differing
                )
                logger.warning(
                    f"{self.class_name}: data fingerprint mismatch for {key!r} "
                    f"(differing fields: {', '.join(differing)}): {details}. The "
                    f"data changed since the expected run; {tail}"
                )

    def _compare_fingerprints_on_failure(self) -> None:
        """Run a partial fingerprint comparison on the failure path, never raising.

        ``run()`` and ``run_cv()`` compare fingerprints only after a window
        completes, but the factor fingerprints are recorded before
        prediction. When the window fails part-way the data may already
        have changed and the operator would only see the downstream error,
        so the comparison is repeated here with ``partial=True``.

        Swallowing exceptions is deliberate and must stay: this diagnostic
        is extra information and must never replace the original error. A
        failure of the diagnostic itself logs a warning that does not
        contain "data fingerprint mismatch", and even that logging is
        guarded so a broken log sink cannot become the raised error.
        """
        try:
            self._compare_fingerprints(partial=True)
        except BaseException as error:  # noqa: BLE001 - never replace the error
            try:
                logger.warning(
                    f"{self.class_name}: the failure-path data diagnostic itself "
                    f"raised {type(error).__name__}: {error!r}; it is skipped and "
                    f"the original error follows"
                )
            except BaseException:  # noqa: BLE001 - a broken log sink must not raise
                pass

    def _assert_weights_contract(
        self, weights: xr.Dataset, prices: xr.Dataset
    ) -> None:
        """Check the target-weight contract of ``weights`` against ``prices``.

        ``weights`` must carry a ``weight`` variable on ``("timestamp",
        "symbol")`` with exactly the price axes. A NaN keeps the symbol's
        holding and a finite value is its target; a row may mix the two
        (all NaN holds the whole book). The gross exposure of a row's
        targets, the sum of their absolute values, is at most 1.

        Raises
        ------
        ValueError
            On the first violated rule, naming the offending bar.
        """
        if "weight" not in weights.data_vars:
            raise ValueError(
                f"{self.class_name}: weights must carry a 'weight' variable, got "
                f"{list(weights.data_vars)}"
            )
        weight = weights["weight"]
        if weight.dims != ("timestamp", "symbol"):
            raise ValueError(
                f"{self.class_name}: weight dims must be ('timestamp', 'symbol'), "
                f"got {weight.dims}"
            )
        if not np.array_equal(weight.timestamp.values, prices.timestamp.values):
            raise ValueError(
                f"{self.class_name}: weight timestamps do not match the price "
                f"timestamps"
            )
        if not np.array_equal(weight.symbol.values, prices.symbol.values):
            raise ValueError(
                f"{self.class_name}: weight symbols do not match the price symbols"
            )

        values = weight.values
        timestamps = weight.timestamp.values
        gross = np.abs(np.nan_to_num(values)).sum(axis=1)
        over = gross > 1 + 1e-9
        if over.any():
            idx = int(np.argmax(over))
            raise ValueError(
                f"{self.class_name}: weight row at {self._bar_label(timestamps[idx])} "
                f"has gross exposure {gross[idx]} > 1"
            )

    @abstractmethod
    def _generate_signals(
        self, predictions: xr.Dataset, prices: xr.Dataset
    ) -> xr.Dataset:
        """Turn predictions and prices into target weights satisfying the contract.

        Both inputs share the price axes. The result must pass
        ``_assert_weights_contract``: a ``weight`` variable on
        ``(timestamp, symbol)``, NaN where a symbol keeps its holding, with
        the gross exposure of each row's targets at most 1.
        """

    def _signal_metrics(self) -> dict:
        """Return metrics about the last ``_generate_signals`` call, for ``metrics.json``.

        Called right after the window's metrics are computed; the keys are
        added to them. The default adds nothing; a backtester whose signal
        generation can hold a bar after a failure reports it here.
        """
        return {}

    @abstractmethod
    def _simulate(
        self, weights: xr.Dataset, prices: xr.Dataset, dataset=None
    ) -> SimulationResult:
        """Simulate the portfolio; a signal at bar t fills at bar t+1's fill price.

        ``dataset`` is the market dataset ``prices`` came from, which marks
        its delistings; ``config.price_dataset`` when omitted.
        """

    @abstractmethod
    def _simulate_benchmark(self, benchmark_prices: xr.Dataset) -> SimulationResult:
        """Simulate buying and holding the single benchmark symbol.

        ``benchmark_prices`` is ``_load_benchmark_prices``'s output: the fill
        and valuation columns of one symbol on the strategy's own bars, all
        finite. The benchmark must follow the strategy's execution
        conventions (fill delay, fees, slippage, initial cash), so its curve
        is comparable bar for bar with the strategy's. Called only when a
        benchmark is configured.
        """

    @abstractmethod
    def _engine_stats(self, simulation: SimulationResult) -> dict:
        """Return the engine's whole-window statistics keyed by metric name."""

    def _return_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """Return the return statistics of ``simulation`` cut to ``ranges``.

        A portfolio object cannot be sliced in time and re-simulating a
        sub-period would reset capital and change the path, so the
        statistics come from the return series of the same simulation, cut
        to ``ranges`` (inclusive pairs of bar labels) and concatenated in time
        order: ``backtest_stats.return_stats`` annualized by ``MARKET``.

        Raises
        ------
        ValueError
            If no simulated return falls inside ``ranges``.
        """
        return backtest_stats.return_stats(
            simulation.returns,
            bar_interval=simulation.bar_interval,
            year_freq=self.MARKET.year_freq(simulation.bar_interval),  # type: ignore[union-attr]
            ranges=ranges,
        )

    def _window_split(
        self, window_timestamps: np.ndarray, calendar: np.ndarray, train_start, train_end
    ) -> dict:
        """Split the window bars into in-sample and out-of-sample ranges.

        ``train_start`` / ``train_end`` are the fitted training window, after
        the model's purge. The *effective training window* is every bar the
        model has seen, ``quantlab.utils.split.in_sample_window`` of the
        fitted window and the labels' lookahead L, counted in calendar bars.
        Returns three keys that are merged into the top level of the metrics,
        all as ``_bar_label`` endpoints: ``training_window`` (the effective
        training window or ``None``), ``in_sample_range`` (first and last
        window bar inside it, or ``None``) and ``out_of_sample_ranges`` (the
        0, 1 or 2 runs of window bars outside it, from
        ``quantlab.utils.split.split_ranges``). A non-empty overlap logs a
        warning naming both windows, and the backtest goes on with the two
        parts reported separately. A ``None`` training date logs a warning
        and every bar counts as out-of-sample.
        """
        if train_start is None or train_end is None:
            logger.warning(
                f"{self.class_name}: model config has train_start={train_start!r}, "
                f"train_end={train_end!r}; the effective training window is "
                f"unknown, so metrics record training_window as null and every "
                f"backtest bar as out-of-sample"
            )
            training_window = None
        else:
            training_window = in_sample_window(
                np.sort(np.asarray(calendar).astype("datetime64[ns]")),
                self._slice_bound(train_start),
                self._slice_bound(train_end),
                self._lookahead_bars(),
            )
        timestamps = np.asarray(window_timestamps).astype("datetime64[ns]")
        in_sample, out_of_sample = split_ranges(timestamps, [training_window])
        split = {
            "training_window": self._label_pair(training_window),
            "in_sample_range": self._label_pair(in_sample[0]) if in_sample else None,
            "out_of_sample_ranges": [self._label_pair(r) for r in out_of_sample],
        }
        if in_sample:
            logger.warning(
                f"{self.class_name}: backtest window {self._bar_label(timestamps[0])}.."
                f"{self._bar_label(timestamps[-1])} overlaps the model's effective "
                f"training window {split['training_window'][0]}.."
                f"{split['training_window'][1]} (train_start..train_end + label "
                f"lookahead); bars {split['in_sample_range'][0]}.."
                f"{split['in_sample_range'][1]} are in-sample. Continuing: in-sample "
                f"and out-of-sample results are reported separately"
            )
        return split

    @classmethod
    def _label_pair(cls, pair: tuple | None) -> tuple[str, str] | None:
        """Return a ``(first, last)`` pair of bars as ``_bar_label`` endpoints."""
        if pair is None:
            return None
        return cls._bar_label(pair[0]), cls._bar_label(pair[1])

    def _turnover(self, simulation: SimulationResult) -> xr.DataArray:
        """Return the turnover of every fill bar of ``simulation``.

        ``backtest_stats.turnover`` of its orders and value, from
        ``config.init_cash``.
        """
        return backtest_stats.turnover(
            simulation.orders, simulation.value, self.config.init_cash
        )

    def _turnover_stats(self, turnover: xr.DataArray, bar_interval) -> dict:
        """Return ``backtest_stats.turnover_stats`` for this market and rebalance step."""
        return backtest_stats.turnover_stats(
            turnover,
            bar_interval=bar_interval,
            year_freq=self.MARKET.year_freq(bar_interval),  # type: ignore[union-attr]
            rebalance_periods=self.config.rebalance_periods,
        )

    def _period_record_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """Return order, trade and turnover statistics restricted to ``ranges``.

        ``Total Orders``, ``Total Fees Paid`` and ``Traded Notional`` count
        the orders filled inside the ranges; ``Total Closed Trades`` the
        trades with status ``"Closed"`` whose exit falls inside them;
        ``Total Open Trades`` the trades still open at each range's end
        (entered on or before it, and not yet exited or exited after it);
        the three turnover rows are the ``_turnover_stats`` of the fill
        bars inside the ranges. Several ranges never overlap, so the counts
        add up across them. The names are vectorbt's whole-window names, so
        a slice column and the whole column share a row.

        The trade counts use the same position-level definition as the
        whole-window statistics (one entry-to-flat round trip per symbol),
        so the per-range closed trades sum to the whole window's total.
        """
        orders = simulation.orders
        if orders.sizes.get("order", 0) > 0:
            in_range = backtest_stats.in_ranges(orders["timestamp"].values, ranges)
            sizes = np.abs(orders["size"].values.astype(np.float64))[in_range]
            prices = orders["price"].values.astype(np.float64)[in_range]
            fees = orders["fees"].values.astype(np.float64)[in_range]
            order_count = int(in_range.sum())
            fees_paid = float(fees.sum())
            traded_notional = float((sizes * prices).sum())
        else:
            order_count, fees_paid, traded_notional = 0, 0.0, 0.0

        trades = simulation.trades
        closed_trade_count = open_trade_count = 0
        if trades is not None and trades.sizes.get("trade", 0) > 0:
            status = trades["status"].values.astype(str)
            closed = status == "Closed"
            closed_trade_count = int(
                (closed & backtest_stats.in_ranges(trades["exit_timestamp"].values, ranges)).sum()
            )
            # Exact timestamps, not days: on intraday data a trade entered
            # later on the range's last day is not "open at the range end".
            entry_ts = trades["entry_timestamp"].values.astype("datetime64[ns]")
            exit_ts = trades["exit_timestamp"].values.astype("datetime64[ns]")
            for _, end in ranges:
                end_ts = backtest_stats.label_ns(end)
                open_at_end = (entry_ts <= end_ts) & (~closed | (exit_ts > end_ts))
                open_trade_count += int(open_at_end.sum())

        turnover = self._turnover(simulation)
        turnover = turnover.isel(
            timestamp=backtest_stats.in_ranges(turnover.timestamp.values, ranges)
        )
        return {
            "Total Orders": order_count,
            "Total Fees Paid": fees_paid,
            "Traded Notional": traded_notional,
            "Total Closed Trades": closed_trade_count,
            "Total Open Trades": open_trade_count,
            **self._turnover_stats(turnover, simulation.bar_interval),
        }

    def _compute_metrics(
        self,
        simulation: SimulationResult,
        benchmark: SimulationResult | None,
        split: dict | None,
    ) -> dict:
        """Compute the whole, in-sample and out-of-sample metric blocks.

        All three come from the same continuous simulation; nothing here
        simulates a second time. ``whole`` is the engine's whole-window
        statistics plus the three turnover rows and ``Total Orders``.
        ``in_sample`` and ``out_of_sample`` merge ``_return_stats``
        and ``_period_record_stats`` over their ranges and are ``None`` when
        there is no such range. ``benchmark`` and ``relative`` appear only
        when a benchmark was simulated: ``benchmark`` holds the benchmark's
        ``symbol`` (display name) and ``axis_symbol`` plus the same three
        slices of return statistics (``_return_stats`` over the whole
        window and over each slice's ranges), and ``relative`` holds the
        three slices of ``_relative_stats``, the strategy measured against
        the benchmark. ``execution`` holds the simulation's
        ``rejected_order_count``, ``rejected_orders`` and
        ``max_target_deviation``. Every key of ``split`` is copied to the top level;
        the in-sample ranges come from ``split["in_sample_ranges"]`` when
        present (the stitched curve) and from the single
        ``split["in_sample_range"]`` otherwise. ``split=None`` (a
        ``run_weights()`` run, which has no training window) computes the
        ``whole`` blocks only: no ``in_sample`` or ``out_of_sample`` key
        appears at any level and no split key is added.
        """
        whole = self._engine_stats(simulation)
        whole.update(
            self._turnover_stats(self._turnover(simulation), simulation.bar_interval)
        )
        # The number of fills: position-level trade counts do not say how
        # often we traded. A run with no fills has no `order` dimension, so
        # use `.sizes.get` rather than a subscript that would raise.
        whole["Total Orders"] = int(simulation.orders.sizes.get("order", 0))
        timestamps = simulation.value.timestamp.values
        whole_range = [
            (self._bar_label(timestamps[0]), self._bar_label(timestamps[-1]))
        ]
        whole.update(self._win_rates(simulation, whole_range))
        metrics: dict = {
            "whole": whole,
            "execution": {
                "rejected_order_count": len(simulation.rejected_orders),
                "rejected_orders": list(simulation.rejected_orders),
                "max_target_deviation": simulation.max_target_deviation,
            },
        }

        # The slices other than `whole`, by name; none without a split.
        slices: dict[str, list[tuple[str, str]]] = {}
        if split is not None:
            if "in_sample_ranges" in split:
                slices["in_sample"] = list(split["in_sample_ranges"])
            else:
                in_sample_range = split["in_sample_range"]
                slices["in_sample"] = [in_sample_range] if in_sample_range else []
            slices["out_of_sample"] = list(split["out_of_sample_ranges"])

        for name, ranges in slices.items():
            metrics[name] = (
                {
                    **self._return_stats(simulation, ranges),
                    **self._period_record_stats(simulation, ranges),
                    **self._win_rates(simulation, ranges),
                }
                if ranges
                else None
            )

        if benchmark is not None:
            axis_symbol = self._benchmark_axis_symbol or ""
            metrics["benchmark"] = {
                "symbol": self._benchmark_display_name(axis_symbol, timestamps[-1]),
                "axis_symbol": axis_symbol,
                "whole": self._return_stats(benchmark, whole_range),
                **{
                    name: self._return_stats(benchmark, ranges) if ranges else None
                    for name, ranges in slices.items()
                },
            }
            metrics["relative"] = {
                "whole": {
                    **self._relative_stats(simulation, benchmark, whole_range),
                    **self._win_rates(simulation, whole_range, benchmark),
                },
                **{
                    name: (
                        {
                            **self._relative_stats(simulation, benchmark, ranges),
                            **self._win_rates(simulation, ranges, benchmark),
                        }
                        if ranges
                        else None
                    )
                    for name, ranges in slices.items()
                },
            }
        metrics.update(split or {})
        return metrics

    @staticmethod
    def _win_rates(
        simulation: SimulationResult,
        ranges: list[tuple[str, str]],
        benchmark: SimulationResult | None = None,
    ) -> dict:
        """Return ``backtest_stats.win_rates`` of ``simulation``, its fills marking the periods.

        Against ``benchmark`` when given, otherwise against zero.
        """
        orders = simulation.orders
        fills = (
            orders["timestamp"].values
            if orders.sizes.get("order", 0)
            else np.array([], dtype="datetime64[ns]")
        )
        return backtest_stats.win_rates(
            simulation.returns,
            fills,
            ranges=ranges,
            benchmark_returns=None if benchmark is None else benchmark.returns,
        )

    def _benchmark_display_name(self, axis_symbol: str, as_of) -> str:
        """Return the benchmark's readable name, its ticker when the store has one.

        A CRSP benchmark store is keyed by PERMNO, a bare number, so the
        ticker sidecar beside the benchmark's own store (not the price
        store, as ``ticker_store()`` names it) names it as of the window's last
        bar. Without a sidecar, or when the dataset names no store to look
        beside (a ``FrameDataset``), the axis label is returned unchanged.
        """
        dataset = self.config.benchmark_dataset
        store = None if dataset is None else dataset.ticker_store()
        if store is None or not axis_symbol:
            return axis_symbol
        lookup = CrspTickerLookup.beside_store(store)
        return str(lookup.label([axis_symbol], pd.Timestamp(as_of).date())[0])

    def _relative_stats(
        self,
        simulation: SimulationResult,
        benchmark: SimulationResult,
        ranges: list[tuple[str, str]],
    ) -> dict:
        """Return ``backtest_stats.relative_stats`` of the strategy against the benchmark.

        Engine-independent: it reads only the two per-bar return series,
        annualized by ``MARKET``.

        Raises
        ------
        ValueError
            If the two return series are not on the same bars.
        """
        return backtest_stats.relative_stats(
            simulation.returns,
            benchmark.returns,
            bar_interval=simulation.bar_interval,
            year_freq=self.MARKET.year_freq(simulation.bar_interval),  # type: ignore[union-attr]
            ranges=ranges,
        )

    def _report_notes(self) -> list[str]:
        """Return the notes attached to the report and to ``metrics.json``.

        The default note says that no borrow or short-financing cost is
        modelled. An engine that models borrow costs overrides this.
        """
        return [
            "No borrow or short-financing cost is modelled, so short-side "
            "returns are optimistic."
        ]

    #: Metric blocks a tracking run's summary receives, when present.
    _TRACKED_BLOCKS = ("whole", "in_sample", "out_of_sample", "benchmark", "relative")

    @contextmanager
    def _tracking_run(self) -> Iterator[TrackingRun]:
        """Open the tracking run of one backtest through ``config.tracker``.

        The run name, ``{ClassName}_{timestamp}``, is fixed here and is also
        the name of the run directory written later, apart from the model's
        training runs. The project is ``{ClassName}_backtest`` unless the
        tracker sets its own, and the run config is ``get_config()``. The
        run is opened before the backtest, so one that raises is finished as
        failed.
        """
        self._run_name = self._run_dir_name()
        with self.config.tracker.start_run(
            project=f"{self.class_name}_backtest",
            group=None,
            name=self._run_name,
            config=self.get_config(),
        ) as run:
            yield run

    def _track(self, run: TrackingRun, run_dir: Path | None, metrics: dict) -> None:
        """Write one finished backtest to its tracking run.

        The run config gains what the backtest resolved (the data
        fingerprints), the summary holds the ``whole``, ``in_sample`` and
        ``out_of_sample`` blocks as ``whole/<metric>`` and so on (plus
        ``benchmark`` and ``relative`` when a benchmark ran), and
        ``report.html`` is attached. A run kept in memory (``run_dir`` is
        ``None``) has no report to attach.
        """
        run.update_config(self.get_config())
        run.summarize(
            {
                block: metrics[block]
                for block in self._TRACKED_BLOCKS
                if metrics.get(block) is not None
            }
        )
        if run_dir is not None:
            run.log_file(run_dir / "report.html")

    def _run_dir_name(self) -> str:
        """Return ``{ClassName}_{timestamp}``, unique down to the microsecond."""
        return f"{self.class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _drawdown_span(self, simulation: SimulationResult) -> dict | None:
        """Return the deepest drawdown from its valley to recovery, or ``None``.

        The base class does not read ``simulation.native``, so it returns
        ``None`` and the report shows no drawdown markers. An engine that
        can find the deepest drawdown in its own result overrides this and
        returns ``{"valley", "end", "bars", "depth", "recovered"}``:
        ``valley`` is the deepest bar and ``end`` the bar it recovered on
        (both ``_bar_label`` strings), ``bars`` the number of bars between
        them (not calendar days, and not the maximum drawdown duration),
        ``depth`` a negative fraction and ``recovered`` whether the
        recovery happened before the last bar. Deliberately not abstract: it
        is a report detail no engine is forced to implement.
        """
        return None

    def _report_summary(
        self,
        simulation: SimulationResult,
        block: dict,
        *,
        drawdown_span: dict | None = None,
    ) -> dict:
        """Return the "Setup" lines of ``report.html``: ``report_summary`` of this run.

        ``block`` is the metric level carrying the split keys (``run()``
        passes the metrics, ``run_cv()`` ``metrics["stitched"]``); the
        config mapping is ``get_config()`` and the benchmark is named with
        where it was read from.
        """
        dataset = self.config.benchmark_dataset
        return report_summary(
            self.get_config(),
            block,
            bar_interval=simulation.bar_interval,
            drawdown_span=drawdown_span,
            benchmark_source=None if dataset is None else self._where(dataset),
        )

    def _report_windows(
        self,
        simulation: SimulationResult,
        block: dict,
        records: list[dict] | None = None,
    ) -> dict:
        """Return the timeline of ``report.html``: ``report_windows`` of this run.

        ``records`` are the folds of a ``run_cv()`` run, each with its
        ``fold`` number, its own ``simulation`` (the bars it traded) and
        ``metrics`` (its ``training_window`` and ``in_sample_range``).
        """
        folds = None
        if records is not None:
            folds = [
                {
                    "fold": record["fold"],
                    "training_window": record["metrics"].get("training_window"),
                    "traded": self._label_pair(
                        record["simulation"].value.timestamp.values[[0, -1]]
                    ),
                    "in_sample_range": record["metrics"].get("in_sample_range"),
                }
                for record in records
            ]
        return report_windows(simulation.value.timestamp.values, block, folds)

    def _report_and_persist(
        self,
        weights: xr.Dataset,
        simulation: SimulationResult,
        metrics: dict,
        *,
        benchmark: SimulationResult | None = None,
        predictions: xr.Dataset | None = None,
    ) -> Path | None:
        """Write a new run directory with every artifact of ``run()`` or ``run_weights()``.

        Nothing is written, and ``None`` is returned, when
        ``config.output_dir`` is ``None``.

        The directory holds ``config.json``, ``weights.zarr``,
        ``equity.zarr`` (``value`` and ``returns``, plus ``benchmark_value``
        and ``benchmark_returns`` when a benchmark ran), ``settlements.json``,
        ``metrics.json``, ``report.html`` and ``fingerprint.json``, plus
        ``predictions.zarr`` when ``predictions`` is given (a run with a
        model; see ``_write_predictions``) and ``inputs/`` when a dataset is
        held in memory (see ``_run_dir_config``). Each
        JSON file goes through ``to_jsonable`` (NaN and infinities become
        null, timestamps become ISO strings) and is written atomically.

        ``report.html`` is self-contained: headline numbers, a timeline
        of the backtest and training windows from ``_report_windows``, the
        setup lines from ``_report_summary``, the metric tables (the
        out-of-sample slice when the run has an in-sample part, with an
        in-sample vs out-of-sample table) and the chart tabs: Performance
        (equity, drawdown and monthly returns with the in-sample range
        shaded and the deepest drawdown marked, the benchmark beside the
        portfolio), Excess (with a benchmark), Rolling and Portfolio
        (turnover, holdings and exposure per rebalance), followed by the
        notes. A metric the report does not know is still shown, so a
        change in the metric set cannot make the report raise and discard
        the staged run.

        Returns
        -------
        Path or None
            The final run directory, or ``None`` without ``output_dir``.
        """
        # Everything is written to a staging directory that is renamed into
        # place only after the last artifact succeeds.
        def _write(run_dir: Path, name: str) -> None:
            """Write every artifact of this run into ``run_dir`` titled ``name``."""
            write_json_atomically(
                run_dir / "config.json",
                to_jsonable(self._run_dir_config(run_dir)),
                indent=2,
            )
            self._write_weights_and_equity(run_dir, weights, simulation, benchmark)
            if predictions is not None:
                self._write_predictions(run_dir, predictions)
            write_json_atomically(
                run_dir / "settlements.json",
                to_jsonable(simulation.settlements),
                indent=2,
            )
            write_json_atomically(
                run_dir / "metrics.json", to_jsonable(metrics), indent=2
            )
            chart = self._report_chart_inputs(simulation, metrics, benchmark)
            write_backtest_report(
                simulation.value,
                run_dir / "report.html",
                title=name,
                summary=self._report_summary(
                    simulation, metrics, drawdown_span=chart["drawdown_span"]
                ),
                windows=self._report_windows(simulation, metrics),
                metrics=metrics,
                **chart,
                **self._report_portfolio_inputs(weights, simulation),
            )
            write_json_atomically(
                run_dir / "fingerprint.json", to_jsonable(self._fingerprints), indent=2
            )

        return self._persist_run_dir(_write)

    def _run_dir_config(self, run_dir: Path) -> dict:
        """Return the ``config.json`` of ``run_dir``, writing what its datasets need.

        ``get_config()``, except that each dataset of ``RUN_DATASET_FIELDS``
        is asked through ``persist_with_run(run_dir, field)`` what a rebuild
        needs: a dataset read from a project store writes nothing and is
        recorded as it is; a ``FrameDataset`` writes its panel to
        ``inputs/<field>.zarr`` and is recorded reading it, relative to the
        run directory, which ``load_backtester_from_config(config,
        run_dir=...)`` resolves again.
        """
        config = self.get_config()
        for name in RUN_DATASET_FIELDS:
            dataset = getattr(self.config, name)
            if dataset is None:
                continue
            recorded = dataset.persist_with_run(run_dir, name)
            if recorded is not None:
                config[name] = recorded
        return config

    def _persist_run_dir(self, write) -> Path | None:
        """Create ``output_dir/{ClassName}_{timestamp}/`` and fill it through ``write``.

        With ``config.output_dir`` set to ``None`` the run stays in memory:
        ``write`` is never called, nothing is created, and ``None`` is
        returned.

        ``write(directory, name)`` writes every artifact into ``directory``;
        ``name`` is the final directory name, used as the report title. The
        artifacts go into a hidden sibling ``.{name}.partial`` first and the
        directory is renamed into place only when everything succeeded
        (a rename within one filesystem is atomic). On any exception,
        including ``KeyboardInterrupt``, the staging directory is removed
        and the error re-raised, so ``output_dir`` only ever contains
        complete run directories that a loader can safely rebuild from.

        Raises
        ------
        RuntimeError
            If the final directory already exists; it is never
            overwritten.
        """
        import shutil

        if self.config.output_dir is None:
            return None
        final = Path(self.config.output_dir) / self._run_name
        if final.exists():
            raise RuntimeError(f"{final} already exists")
        staging = final.parent / f".{final.name}.partial"
        staging.mkdir(parents=True)
        try:
            write(staging, final.name)
            if final.exists():
                raise RuntimeError(f"{final} already exists")
            staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return final

    def _report_chart_inputs(
        self,
        simulation: SimulationResult,
        metrics: dict,
        benchmark: SimulationResult | None,
        *,
        block: dict | None = None,
    ) -> dict:
        """Return the chart keyword arguments shared by ``report.html`` and ``report_figure``.

        ``report_chart_inputs`` of this run: ``metrics`` carries the
        ``notes`` and, unless ``block`` is given (``run_cv()`` passes
        ``metrics["stitched"]``), the split keys and the benchmark record;
        ``simulation.value`` is passed positionally by the callers.
        """
        return report_chart_inputs(
            metrics if block is None else block,
            metrics["notes"],
            returns=simulation.returns,
            init_cash=self.config.init_cash,
            drawdown_span=self._drawdown_span(simulation),
            benchmark_value=None if benchmark is None else benchmark.value,
            benchmark_returns=None if benchmark is None else benchmark.returns,
        )

    def _report_portfolio_inputs(
        self, weights: xr.Dataset, simulation: SimulationResult
    ) -> dict:
        """Return the Portfolio and Rolling tab inputs: ``report_portfolio_inputs`` of this run.

        ``weights`` are the run's target weights, with a ``weight`` variable
        on ``(timestamp, symbol)``.
        """
        return report_portfolio_inputs(
            weights["weight"],
            simulation.orders,
            simulation.value,
            init_cash=self.config.init_cash,
            bar_interval=simulation.bar_interval,
            trading_days_per_year=self.MARKET.trading_days_per_year,  # type: ignore[union-attr]
            session_minutes_per_day=self.MARKET.session_minutes_per_day,  # type: ignore[union-attr]
        )

    @staticmethod
    def _write_weights_and_equity(
        directory: Path,
        weights: xr.Dataset,
        simulation: SimulationResult,
        benchmark: SimulationResult | None = None,
    ) -> None:
        """Write ``weights.zarr`` and ``equity.zarr`` into ``directory``.

        With a benchmark, ``equity.zarr`` also carries ``benchmark_value`` and
        ``benchmark_returns`` on the same ``timestamp`` axis.
        """
        XrBackend().to_internal(weights).write(str(directory / "weights.zarr"))
        equity = {"value": simulation.value, "returns": simulation.returns}
        if benchmark is not None:
            equity["benchmark_value"] = benchmark.value
            equity["benchmark_returns"] = benchmark.returns
        XrBackend().to_internal(xr.Dataset(equity)).write(
            str(directory / "equity.zarr")
        )

    def _write_predictions(self, directory: Path, predictions: xr.Dataset) -> None:
        """Write the predictions the rule read as ``predictions.zarr`` into ``directory``.

        ``predictions`` are on the price axes (as handed to
        ``_generate_signals``) and are stored with ``label_specs`` of the
        model as a ``PredictionPanel``.
        """
        PredictionPanel(predictions, label_specs(self.config.model)).write(
            directory / PredictionPanel.FILE_NAME
        )

    def _stitched_split(
        self, timestamps: np.ndarray, records: list[dict]
    ) -> dict:
        """Build the in-sample/out-of-sample split of the stitched curve from the folds.

        The stitched curve is out-of-sample by construction, since each fold
        trades only its own test segment, except for the first bars of each
        fold that overlap that fold's effective training window (the label
        lookahead). The in-sample part is therefore a list: ``training_windows``
        holds every fold's effective training window in fold order,
        ``in_sample_ranges`` every fold's non-empty ``in_sample_range`` in
        fold order, and ``out_of_sample_ranges`` the contiguous runs of
        ``timestamps`` outside all of them, from
        ``quantlab.utils.split.split_ranges``. No singular ``in_sample_range``
        is produced, because several ranges do not fit one pair.
        """
        in_sample_ranges = [
            record["metrics"]["in_sample_range"]
            for record in records
            if record["metrics"]["in_sample_range"] is not None
        ]
        _, out_of_sample = split_ranges(timestamps, in_sample_ranges)
        pieces = [self._label_pair(r) for r in out_of_sample]
        return {
            "training_windows": [
                record["metrics"]["training_window"] for record in records
            ],
            "in_sample_ranges": in_sample_ranges,
            "out_of_sample_ranges": pieces,
        }

    def _persist_cv(
        self,
        records: list[dict],
        weights: xr.Dataset,
        simulation: SimulationResult,
        metrics: dict,
        *,
        benchmark: SimulationResult | None = None,
        predictions: xr.Dataset,
    ) -> Path | None:
        """Write a new run directory with every artifact of a ``run_cv()``.

        The top level describes the stitched curve with the same files as a
        ``run()`` directory: ``config.json``, ``weights.zarr``,
        ``equity.zarr``, ``metrics.json`` (``stitched``, ``folds``,
        ``notes``), ``settlements.json`` (``stitched`` plus per-fold
        ``folds``), ``fingerprint.json`` (the stitched window),
        ``predictions.zarr`` (the concatenated fold predictions the stitched
        pass read, see ``_write_predictions``) and ``report.html``. The report receives ``metrics["stitched"]``, shades
        no in-sample range (the several in-sample ranges are listed in the
        summary lines and the notes) and marks the deepest drawdown of the
        stitched simulation. Each fold's own simulation is written under
        ``folds/fold_{i}/`` as ``weights.zarr`` and ``equity.zarr``, where
        ``i`` is the manifest's fold number.

        Returns
        -------
        Path or None
            The final run directory, or ``None`` without ``output_dir``.
        """
        # As in run(): write to a staging directory, rename when complete.
        def _write(run_dir: Path, name: str) -> None:
            """Write every artifact of this CV run into ``run_dir`` titled ``name``."""
            write_json_atomically(
                run_dir / "config.json",
                to_jsonable(self._run_dir_config(run_dir)),
                indent=2,
            )
            self._write_weights_and_equity(run_dir, weights, simulation, benchmark)
            self._write_predictions(run_dir, predictions)
            for record in records:
                fold_dir = run_dir / "folds" / f"fold_{record['fold']}"
                fold_dir.mkdir(parents=True)
                self._write_weights_and_equity(
                    fold_dir,
                    record["weights"],
                    record["simulation"],
                    record.get("benchmark"),
                )
            write_json_atomically(
                run_dir / "settlements.json",
                to_jsonable(
                    {
                        "stitched": simulation.settlements,
                        "folds": [
                            {
                                "fold": record["fold"],
                                "settlements": record["simulation"].settlements,
                            }
                            for record in records
                        ],
                    }
                ),
                indent=2,
            )
            write_json_atomically(
                run_dir / "metrics.json", to_jsonable(metrics), indent=2
            )
            chart = self._report_chart_inputs(
                simulation, metrics, benchmark, block=metrics["stitched"]
            )
            write_backtest_report(
                simulation.value,
                run_dir / "report.html",
                title=name,
                summary=self._report_summary(
                    simulation, metrics["stitched"], drawdown_span=chart["drawdown_span"]
                ),
                windows=self._report_windows(simulation, metrics["stitched"], records),
                metrics=metrics["stitched"],
                **chart,
                **self._report_portfolio_inputs(weights, simulation),
            )
            write_json_atomically(
                run_dir / "fingerprint.json", to_jsonable(self._fingerprints), indent=2
            )

        return self._persist_run_dir(_write)
