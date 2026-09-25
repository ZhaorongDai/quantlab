"""Backtester base class and the result types every backtest run produces.

This module sits at the end of the pipeline: a trained return model and a
price dataset go in, and a run directory holding target weights, an equity
curve, metrics and an HTML report comes out. ``BaseBacktester`` owns the two
public entry points, ``run()`` (backtest one model) and ``run_cv()`` (replay
every fold of a cross-validation run as one continuous curve), and every
step between them that does not depend on the simulation engine. Engine
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import xarray as xr
from loguru import logger

from quantlab.base.data import MarketDataset
from quantlab.base.model import BaseModel, DLModel
from quantlab.backend import XrBackend
# Importing this submodule also runs the `crsp` package `__init__` (the CRSP
# converter and polars), which adds about a second of import time.
from quantlab.dataset.crsp.tickers import CrspTickerLookup
from quantlab.enums.constant import Date
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.backtest_report import DASH, write_backtest_report
from quantlab.utils.fingerprint import dataset_fingerprint
from quantlab.utils.jsonable import to_jsonable
from quantlab.utils.timer import Timer

from .config import BacktestConfig, FactorConfig

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

#: Mean length of a calendar year in days. Bars longer than one day span
#: calendar time (weekly, monthly), so ``MarketSpec.year_freq`` annualizes
#: them against this number.
CALENDAR_DAYS_PER_YEAR = 365.25


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

        ``year_freq / bar_interval`` is the number of bars per year. Intraday
        bars use trading days times session minutes divided by the bar's
        minutes (one-minute bars: 252 x 390). A bar of exactly one day is one
        trading day, giving ``trading_days_per_year``. A longer bar spans
        calendar time (a weekly bar is one calendar week whatever the
        holidays), so bars per year is ``CALENDAR_DAYS_PER_YEAR`` divided by
        the bar's days, capped at ``trading_days_per_year``. The function is
        continuous at one day and non-increasing in the interval.

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
        interval = pd.Timedelta(bar_interval)
        if interval <= pd.Timedelta(0):
            raise ValueError(f"bar_interval must be positive, got {interval}")
        one_day = pd.Timedelta(days=1)
        if interval >= one_day:
            bars_per_year = min(
                float(self.trading_days_per_year),
                CALENDAR_DAYS_PER_YEAR * (one_day / interval),
            )
        else:
            minutes = interval / pd.Timedelta(minutes=1)
            bars_per_year = (
                self.trading_days_per_year * self.session_minutes_per_day / minutes
            )
        return interval * bars_per_year


@dataclass
class SimulationResult:
    """Output of one engine simulation in engine-independent form.

    ``value`` and ``returns`` are the portfolio value and per-bar returns on
    the ``timestamp`` dimension. ``orders`` is a dataset on an ``order``
    dimension with the variables ``timestamp``, ``symbol``, ``size``,
    ``price``, ``fees`` and ``side``. ``trades`` is a dataset on a ``trade``
    dimension with ``symbol``, ``entry_timestamp``, ``exit_timestamp``,
    ``pnl``, ``return`` and ``status`` (``"Open"`` or ``"Closed"``), empty
    when nothing traded. ``liquidations`` records forced exits of delisted
    holdings. ``native`` is the engine's own result object and is read only
    by the engine that produced it.

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
    liquidations: list[dict]
    bar_interval: np.timedelta64
    trades: xr.Dataset | None = None
    native: object | None = None


@dataclass
class BacktestResult:
    """Return value of ``BaseBacktester.run()``.

    ``run_dir`` is the directory this run wrote its artifacts to.
    ``predictions`` and ``weights`` are panels on ``(timestamp, symbol)``
    covering exactly the backtest window; ``metrics`` is the same mapping
    written to ``metrics.json``. ``benchmark`` is the buy-and-hold simulation
    of ``config.benchmark_dataset`` on the same bars, or ``None`` when no
    benchmark is configured.

    Examples
    --------
    >>> result = backtester.run()
    >>> sorted(p.name for p in result.run_dir.iterdir())
    ['config.json', 'equity.zarr', 'fingerprint.json', 'liquidations.json',
     'metrics.json', 'report.html', 'weights.zarr']
    >>> sorted(result.metrics)
    ['in_sample', 'in_sample_range', 'notes', 'out_of_sample',
     'out_of_sample_ranges', 'training_window', 'whole']
    """

    run_dir: Path
    predictions: xr.Dataset
    weights: xr.Dataset
    simulation: SimulationResult
    metrics: dict = field(default_factory=dict)
    benchmark: SimulationResult | None = None


@dataclass
class _BacktestWindow:
    """One backtested window before persistence, shared by ``run()`` and each fold."""

    predictions: xr.Dataset
    prices: xr.Dataset
    weights: xr.Dataset
    simulation: SimulationResult
    split: dict
    metrics: dict
    benchmark: SimulationResult | None = None


@dataclass
class CVBacktestResult:
    """Return value of ``BaseBacktester.run_cv()``.

    ``folds`` holds one record per replayed fold: the manifest fields
    (``fold``, the four dates, ``checkpoint``) plus that fold's own
    ``predictions``, ``weights``, ``simulation`` and ``metrics`` from an
    independent per-fold simulation. ``weights`` and ``simulation`` are the
    concatenated fold weights and the single continuous simulation over
    them. ``metrics`` mirrors ``metrics.json`` with the keys ``stitched``,
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

    run_dir: Path
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
    ``_simulate``, ``_simulate_benchmark``, ``_engine_stats`` and
    ``_period_returns_stats``, then a named concrete class that composes a
    ``MarketSpec`` (the ``MARKET`` class attribute) and a signal generator
    (``_generate_signals``) onto that engine. Concrete classes also set
    ``config_cls``, the config class the ``config`` setter accepts.

    The public entry points ``run()`` and ``run_cv()`` are defined here and
    never overridden. Both run the same fixed sequence of steps and call the
    hooks above along the way: prepare the model, re-date the factors and
    predict, generate signals, simulate, compute metrics, then write the run
    directory.

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
    def ticker_lookup(self) -> CrspTickerLookup:
        """Lookup that turns symbol ids into readable ticker names.

        It reads the ``.crsp_tickers.json`` file stored next to the price
        store. For CRSP data (the Center for Research in Security Prices),
        symbols are PERMNOs, permanent numeric security ids, and this file
        records which ticker each PERMNO traded under on each date. The
        backtester is the only layer that knows where the price store is, so
        it owns the lookup: the engine uses it for liquidation records and
        the model for its lists of missing or extra symbols. The lookup is
        built on first access and reset whenever a new config is assigned.
        When no such file exists, ``label()`` returns each symbol unchanged,
        so panels from other vendors are unaffected.

        Examples
        --------
        >>> from datetime import date
        >>> backtester.ticker_lookup.label(["AAA", "BBB"], date(2024, 3, 1))
        ['AAA', 'BBB']
        """
        if self._ticker_lookup is None:
            self._ticker_lookup = CrspTickerLookup.beside_store(
                self.config.price_dataset.config.zarr_file_path
            )
        return self._ticker_lookup

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
        class. ``model_mode="load"`` needs at least one of ``checkpoint``
        (used by ``run()``) and ``cv_project_dir`` (used by ``run_cv()``);
        whichever entry point is called later rejects the missing one. The
        ``checkpoint``, ``cv_project_dir`` and ``output_dir`` fields are
        rewritten as absolute paths so a saved ``config.json`` rebuilds the
        same run from any working directory. ``config.name`` is set to this
        class's import path and ``_validate_config`` runs last.

        Raises
        ------
        TypeError
            If ``config`` is not a ``config_cls``, ``MARKET`` is
            unset, ``config.model`` is not a ``BaseModel``, or
            ``config.benchmark_dataset`` is neither ``None`` nor a
            ``MarketDataset``.
        ValueError
            If ``model_mode``, the load-mode paths,
            ``rebalance_periods``, ``fees``, ``slippage``, ``init_cash``
            or the date order are invalid.

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
        if not isinstance(config.model, BaseModel):
            raise TypeError(
                f"{self.class_name}: config.model must be a BaseModel, got "
                f"{type(config.model).__name__}"
            )

        if config.model_mode not in ("train", "load"):
            raise ValueError(
                f"{self.class_name}: model_mode must be 'train' or 'load', got "
                f"{config.model_mode!r}"
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
        saved ``config.json`` can rebuild and replay the same run.

        Examples
        --------
        >>> cfg = backtester.get_config()
        >>> cfg["name"], cfg["model_mode"], cfg["rebalance_periods"]
        ('mypkg.backtest.MyBacktester', 'load', 5)
        >>> sorted(cfg["data_fingerprint"])  # present once run() has read
        ['factor[0]:PastReturnFactor', 'price_dataset']
        """
        cfg = self.config.to_dict()
        cfg["price_dataset"] = self.config.price_dataset.get_config()
        cfg["model"] = self.config.model.get_config()
        cfg["benchmark_dataset"] = (
            None
            if self.config.benchmark_dataset is None
            else self.config.benchmark_dataset.get_config()
        )
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
        ``_label_ns`` and compared as exact timestamps, never by day.
        """
        ts = pd.Timestamp(str(value)) if isinstance(value, str) else pd.Timestamp(value)
        if ts == ts.normalize():
            return ts.strftime("%Y-%m-%d")
        return ts.isoformat()

    @staticmethod
    def _label_ns(label) -> np.datetime64:
        """Convert a ``_bar_label`` string back to an exact ``datetime64[ns]``."""
        return np.datetime64(pd.Timestamp(str(label)).to_datetime64(), "ns")

    @staticmethod
    def _slice_bound(value):
        """Return a training-window endpoint in the form the model layer slices with.

        The model trains on ``data.sel(timestamp=slice(train_start,
        train_end))``, and pandas interprets a string endpoint at its own
        resolution: ``"2024-05-17"`` includes the whole day while
        ``"2024-05-17T13:00"`` stops at 13:00. Strings are therefore passed
        through unchanged (as plain ``str``) and other values become
        ``pd.Timestamp``, so ``_training_window`` selects the same bars the
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
        than ``config.model``'s dates). The factors are re-dated to cover the
        warm-up, the model predicts the window, the concrete class turns the
        predictions into target weights, the engine simulates them, metrics
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
            If ``model_mode="load"`` without ``config.checkpoint``,
            or the window has no price bars.

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
        if self.config.model_mode == "load" and self.config.checkpoint is None:
            raise ValueError(
                f"{self.class_name}: run() with model_mode='load' requires "
                f"config.checkpoint; cv_project_dir is read only by run_cv()"
            )
        start_date = self._iso_date(self.config.start_date)
        end_date = self._iso_date(self.config.end_date)
        # Per-run state: a second run() on the same object starts clean.
        self._fingerprints = {}
        self._trained_checkpoint = None

        # Any exception in this block would skip the fingerprint comparison
        # below although fingerprints have already been recorded and may
        # already differ. `_prepare_model` is inside on purpose: in train
        # mode it records the training-data fingerprints before
        # `model.train()`, which can fail for the same data reasons.
        try:
            # In load mode the training dates come from the checkpoint's own
            # config.json.
            train_bounds = self._prepare_model()
            calendar = self._price_calendar(end_date)
            # The comparison with config.model's dates is bar-based, so it
            # needs the calendar.
            if self.config.model_mode == "load":
                self._warn_if_config_model_dates_differ(calendar, train_bounds)
            window = self._backtest_window(
                start_date, end_date, calendar, *train_bounds
            )
        except Exception:
            self._compare_fingerprints_on_failure()
            raise
        # Outside the try (and not in a finally) so a clean run compares once.
        self._compare_fingerprints()

        metrics = window.metrics
        if self._trained_checkpoint is not None:
            metrics["trained_checkpoint"] = self._trained_checkpoint
        metrics["notes"] = self._report_notes()
        run_dir = self._report_and_persist(
            window.predictions,
            window.weights,
            window.simulation,
            metrics,
            benchmark=window.benchmark,
        )
        # wandb is off by default; nothing leaves the machine unless enabled.
        if self.config.use_wandb:
            self._log_to_wandb(run_dir, metrics)

        return BacktestResult(
            run_dir=run_dir,
            predictions=window.predictions,
            weights=window.weights,
            simulation=window.simulation,
            metrics=metrics,
            benchmark=window.benchmark,
        )

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
        few bars ahead (its *label horizon*), so the training labels of a fold
        already saw the first few bars after ``train_end``. With no gap
        between a fold's training and test segments, those first bars of
        every test segment therefore count as in-sample.

        The per-fold weights are concatenated and simulated once over the
        prices from the first ``test_start`` to the last ``test_end``, with
        capital carried across fold boundaries; per-fold metrics still come
        from the independent per-fold simulations. Fingerprints cover the
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
            If ``config.cv_project_dir`` is unset, ``model_mode``
            is not ``"load"``, the manifest is malformed, no fold falls
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
        >>> cv.metrics["stitched"]["in_sample_ranges"][:2]
        [('2024-02-12', '2024-02-13'), ('2024-02-20', '2024-02-21')]
        """
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
            saved = self._load_model_checkpoint(fold["checkpoint"])
            # The manifest's dates are authoritative. The checkpoint's own
            # recorded dates are only cross-checked against them, by the bars
            # they select on the calendar rather than by text.
            recorded = self._recorded_train_bounds(saved)
            manifest_bounds = fold["_train_bounds"]
            if recorded is not None and not self._same_training_bars(
                calendar, recorded, manifest_bounds
            ):
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

        # Concatenate the fold weights and simulate the whole span once, with
        # capital carried across fold boundaries.
        first_start = folds[0]["test_start"]
        last_end = folds[-1]["test_end"]
        stitched_weights = xr.concat(
            [record["weights"] for record in records], dim="timestamp"
        )

        # The per-fold loop left only the last fold's fingerprints. Re-date
        # the factors to the whole stitched window (with the first fold's
        # warm-up), re-read, take the price fingerprint from the stitched
        # prices, then compare.
        self._fingerprints = {}
        try:
            self._redate_factors(first_start, last_end, calendar)
            stitched_prices = self._load_prices(first_start, last_end)
            stitched_benchmark_prices = self._load_benchmark_prices(
                first_start, last_end, stitched_prices.timestamp.values
            )
        except Exception:
            self._compare_fingerprints_on_failure()
            raise
        self._compare_fingerprints()

        if not np.array_equal(
            stitched_weights.timestamp.values.astype("datetime64[ns]"),
            stitched_prices.timestamp.values.astype("datetime64[ns]"),
        ):
            raise ValueError(
                f"{self.class_name}: the concatenated fold weights do not cover "
                f"exactly the price bars {first_start}..{last_end}"
            )
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

        notes = self._report_notes() + [
            f"run_cv: the stitched curve is one continuous simulation over folds "
            f"{[fold['fold'] for fold in folds]} ({first_start}..{last_end}), "
            f"capital carried across fold boundaries; per-fold metrics come from "
            f"separate per-fold simulations. Each fold's first label-horizon "
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
        )
        # wandb is off by default; when enabled it logs the stitched metrics.
        if self.config.use_wandb:
            self._log_to_wandb(run_dir, stitched_metrics)

        return CVBacktestResult(
            run_dir=run_dir,
            folds=records,
            weights=stitched_weights,
            simulation=stitched_simulation,
            metrics=metrics,
            benchmark=stitched_benchmark,
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
        are kept under ``_train_bounds`` for ``_training_window``, which
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
                f"supported (supported: {supported})"
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
            # `_training_window`: truncating them to dates would make the
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

        Shared by ``run()`` and by each fold of ``run_cv()``: align the factor
        dates and predict, load the prices, reindex the predictions onto the
        price axes (symbols without a prediction become NaN and are never
        selected), split the window against ``[train_start, train_end +
        label horizon]``, generate and check the weights, simulate, simulate
        the benchmark (when one is configured, on the same bars) and compute
        the metrics. The model must already be prepared.

        Raises
        ------
        ValueError
            If the window contains no price bars.
        """
        predictions = self._align_and_predict(start_date, end_date, calendar)

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

        split = self._split_window(
            prices.timestamp.values,
            self._training_window(calendar, train_start, train_end),
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
        """Train or load the model and return its ``(train_start, train_end)``.

        In train mode the model is collected and trained on the dates in its
        own config; the backtest window never overwrites them, because it
        only decides the prediction span and the in-sample split. In load
        mode the checkpoint is restored and the dates recorded in the
        ``config.json`` beside it are returned when present, since those are
        the dates the checkpoint was really trained on; otherwise
        ``config.model``'s dates are returned.
        """
        model = self.config.model
        if self.config.model_mode == "load":
            saved = self._load_model_checkpoint(self.config.checkpoint)
            recorded = self._recorded_train_bounds(saved)
            if recorded is not None:
                return recorded
            return model.config.train_start, model.config.train_end
        model.collect()
        # Fingerprint the training data right after collect() and before
        # train().
        self._record_training_fingerprints()
        # The checkpoint train() wrote is recorded in config.json and metrics.
        self._trained_checkpoint = str(model.train())
        return model.config.train_start, model.config.train_end

    def _warn_if_config_model_dates_differ(self, calendar, train_bounds: tuple) -> None:
        """Warn when the checkpoint's training dates and ``config.model``'s disagree.

        Used by ``run()`` in load mode only. ``config.model``'s dates may be
        stale or hand-written; the split uses ``train_bounds`` (the
        checkpoint's recorded dates) regardless, and this warning names both
        pairs and the checkpoint path. Two pairs count as equal when they
        select the same bars on ``calendar``, so a different spelling of the
        same window does not warn.
        """
        model = self.config.model
        configured = (model.config.train_start, model.config.train_end)
        if self._same_training_bars(calendar, train_bounds, configured):
            return
        logger.warning(
            f"{self.class_name}: checkpoint {self.config.checkpoint} was trained on "
            f"{train_bounds[0]}..{train_bounds[1]} (its config.json), but config.model "
            f"says train_start={configured[0]!r}, train_end={configured[1]!r}; "
            f"using the checkpoint's dates for the effective training window "
            f"(the split between in-sample and out-of-sample bars)"
        )

    @staticmethod
    def _recorded_train_bounds(saved: dict | None) -> tuple | None:
        """Return the ``(train_start, train_end)`` recorded beside a checkpoint.

        Returns ``None`` when ``saved`` is not a mapping or either date is
        missing. Pure read: no warning and no comparison, which the callers
        do themselves.
        """
        if not isinstance(saved, dict):
            return None
        if saved.get("train_start") is None or saved.get("train_end") is None:
            return None
        return saved["train_start"], saved["train_end"]

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
        through the same pandas ``slice_indexer`` as ``_training_window``.

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

    def _load_model_checkpoint(self, checkpoint) -> dict | None:
        """Load ``checkpoint`` into ``config.model`` and return its ``config.json``.

        Shared by ``run()`` and each fold of ``run_cv()``. The file's
        existence and the factor/label variable check
        (``model._assert_trained_variables``) both run before any feature is
        computed, so a wrong path or a mismatched model fails cheaply. For a
        ``DLModel`` the feature panel is placed in the model's data backend
        first, because rebuilding the network reads ``num_symbols`` from it;
        an ``MLModel`` checkpoint is the whole model and skips this.

        Returns
        -------
        dict | None
            The mapping read from the ``config.json`` beside the checkpoint,
            or ``None`` when there is none.

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
        saved = self._read_checkpoint_config(path)
        model._assert_trained_variables(path)
        if isinstance(model, DLModel):
            model.data_backend.to_internal(model._collect_all_features())
        model.load(path)
        return saved

    def _read_checkpoint_config(self, path: Path) -> dict | None:
        """Read the ``config.json`` beside a checkpoint, or warn and return ``None``.

        Without it the training dates cannot be checked and ``config.model``
        is used as given, which is legitimate for a hand-copied checkpoint,
        so this only warns. Variable checks are the model layer's job.

        Raises
        ------
        ValueError
            If the file exists but is not a JSON object.
        """
        sidecar = path.parent / "config.json"
        if not sidecar.is_file():
            logger.warning(
                f"{self.class_name}: checkpoint {path} has no config.json beside "
                f"it, so its training dates cannot be checked against "
                f"config.model; continuing with config.model as given"
            )
            return None
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError(
                f"{self.class_name}: {sidecar} is not a model config object"
            )
        return saved

    def _price_calendar(self, end_date: str) -> np.ndarray:
        """Return the price dataset's sorted bar timestamps up to ``end_date``.

        Used wherever the backtester counts bars. The dates are written
        straight into the dataset config as ISO strings, and
        ``read(overwrite=True)`` bypasses the backend's read cache, which
        would otherwise return an earlier, narrower read.
        """
        dataset = self.config.price_dataset
        dataset.config.start_date = Date.START_DATE
        dataset.config.end_date = end_date
        dataset.read(overwrite=True)
        return np.sort(dataset.get_xarray_dataset().timestamp.values)

    def _warmup_start(self, calendar: np.ndarray, start_date: str) -> str:
        """Return the bar-counted warm-up start before ``start_date``.

        Counted in calendar bars, not calendar days; the calendar-day buffer
        a factor subtracts on its own is only extra slack. When the calendar
        is too short the start is clamped to its first bar and a warning
        says by how many bars.
        """
        factors = self.config.model.config.factors
        window = max((int(f.config.window) for f in factors), default=0)
        idx = int(
            np.searchsorted(
                calendar, np.datetime64(pd.Timestamp(start_date)), side="left"
            )
        )
        warmup = self._iso_date(calendar[max(idx - window, 0)])
        if idx - window < 0:
            logger.warning(
                f"{self.class_name}: warm-up needs {window} bars before "
                f"{start_date} but the price calendar has only {idx}; short by "
                f"{window - idx} bar(s), clamping the warm-up start to the "
                f"first bar {warmup}"
            )
        return warmup

    def _refresh_factor_reads(self, factor) -> None:
        """Force ``factor``'s underlying data to be re-read after its dates changed.

        The Zarr backend returns its cached data once it holds any, and both
        the dataset and the factor narrow that cache in place when they
        filter. After the model has collected on its own dates, widening the
        factor dates to warm-up plus window without this refresh would
        silently return the old, narrower panel: a shorter warm-up or a
        first rebalance bar with no predictions. The dataset is always
        re-read; the factor store is re-read as well under the ``"read"``
        strategy, whose features come from that store.
        """
        factor.config.dataset.read(overwrite=True)
        if self.config.model.config.factor_data_strategy == "read":
            factor.read(overwrite=True)

    def _redate_factors(
        self, start_date: str, end_date: str, calendar: np.ndarray
    ) -> None:
        """Re-date every factor to warm-up plus window, re-read and fingerprint it.

        Used for the ``run()`` window, each ``run_cv()`` fold and the
        stitched window. The fingerprints are recorded right after the
        re-read, when each factor's dataset holds exactly warm-up plus
        window.
        """
        warmup = self._warmup_start(calendar, start_date)
        for factor in self.config.model.config.factors:
            factor.config.start_date = warmup
            factor.config.end_date = end_date
            factor._reset_dataset_config()
            self._refresh_factor_reads(factor)
        self._record_factor_fingerprints()

    def _align_and_predict(
        self, start_date: str, end_date: str, calendar: np.ndarray
    ) -> xr.Dataset:
        """Re-date the factors, compute the features, predict, and cut to the window."""
        with Timer(f"{self.class_name}: align_and_predict"):
            model = self.config.model
            # The model's missing/extra symbol lists are bare identifiers.
            # It cannot reach the price store, so the labelling callable is
            # handed over here; the model knows no vendor, only a
            # `(symbols, day) -> list[str]` callable.
            model.symbol_labeller = self.ticker_lookup.label
            self._redate_factors(start_date, end_date, calendar)

            features = model._collect_all_features()
            return model.predict_panel(features).sel(
                timestamp=slice(start_date, end_date)
            )

    def _load_prices(self, start_date: str, end_date: str) -> xr.Dataset:
        """Return the fill and valuation price columns over the window.

        The result is a deep copy, because the price dataset object may be
        shared with a factor whose dates change later. The price
        fingerprint is recorded here.

        Raises
        ------
        ValueError
            If either price column is missing from the store.
        """
        dataset = self.config.price_dataset
        dataset.config.start_date = start_date
        dataset.config.end_date = end_date
        ds = dataset.read(overwrite=True).get_xarray_dataset()

        fill = self.MARKET.fill_price_column  # type: ignore[union-attr]
        valuation = self.MARKET.valuation_price_column  # type: ignore[union-attr]
        for column in (fill, valuation):
            if column not in ds.data_vars:
                raise ValueError(
                    f"{self.class_name}: price column {column!r} not found in "
                    f"{dataset.config.zarr_file_path}"
                )
        prices = ds[[fill, valuation]].load().copy(deep=True)
        self._record_price_fingerprint(prices)
        return prices

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
        dataset.config.start_date = start_date
        dataset.config.end_date = end_date
        ds = dataset.read(overwrite=True).get_xarray_dataset()

        fill = self.MARKET.fill_price_column  # type: ignore[union-attr]
        valuation = self.MARKET.valuation_price_column  # type: ignore[union-attr]
        for column in (fill, valuation):
            if column not in ds.data_vars:
                raise ValueError(
                    f"{self.class_name}: benchmark price column {column!r} not "
                    f"found in {dataset.config.zarr_file_path}"
                )
        symbols = [str(symbol) for symbol in ds.symbol.values]
        if len(symbols) != 1:
            raise ValueError(
                f"{self.class_name}: the benchmark dataset must hold exactly one "
                f"symbol, got {len(symbols)} in "
                f"{dataset.config.zarr_file_path}: {symbols[:10]}"
            )
        self._benchmark_axis_symbol = symbols[0]
        read = ds[[fill, valuation]].load().copy(deep=True)
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
    def _dataset_variables_fingerprint(factor) -> dict:
        """Fingerprint the data a factor (or label) consumes from its dataset.

        A KunQuant factor (``FactorConfig``) reads ``data_columns``; a
        Polars factor consumes the whole frame, so every data variable is
        covered.
        """
        ds = factor.config.dataset.get_xarray_dataset()
        if isinstance(factor.config, FactorConfig):
            variables = list(factor.config.data_columns)
        else:
            variables = list(ds.data_vars)
        return dataset_fingerprint(ds, variables)

    @staticmethod
    def _store_fingerprint(ds: xr.Dataset) -> dict:
        """Fingerprint a panel read from a factor or label store, all variables."""
        return dataset_fingerprint(ds, list(ds.data_vars))

    def _record_factor_fingerprints(self) -> None:
        """Record one fingerprint per factor of the model.

        Keys are ``factor[{i}]:{ClassName}`` over the variables the factor
        consumes, for the range its dataset currently holds (the caller
        guarantees this includes the warm-up). Under the ``"read"``
        strategy a second key ``factor_store[{i}]:{ClassName}`` covers the
        panel returned by ``factor.get_features()``, because that store, not
        the dataset, is what the predictions are built from.
        """
        strategy = self.config.model.config.factor_data_strategy
        for i, factor in enumerate(self.config.model.config.factors):
            name = type(factor).__name__
            self._fingerprints[f"factor[{i}]:{name}"] = (
                self._dataset_variables_fingerprint(factor)
            )
            if strategy == "read":
                self._fingerprints[f"factor_store[{i}]:{name}"] = (
                    self._store_fingerprint(factor.get_features())
                )

    def _record_training_fingerprints(self) -> None:
        """Record the fingerprints of the data a train-mode run trains on.

        Called after ``collect()`` and before ``train()``. The window
        fingerprints do not cover the training span, so without these a
        rebuilt run could train a different model unnoticed. One key per
        factor and label over the range ``collect()`` just read:
        ``train_factor[{i}]:{ClassName}`` / ``train_label[{i}]:{ClassName}``
        over the consumed dataset columns under the ``"cal"`` strategy, or
        ``train_factor_store[{i}]:{ClassName}`` /
        ``train_label_store[{i}]:{ClassName}`` over the store panels under
        the ``"read"`` strategy, where the stores are the data actually used.
        """
        model_config = self.config.model.config
        for prefix, items, strategy, getter in (
            ("train_factor", model_config.factors, model_config.factor_data_strategy, "get_features"),
            ("train_label", model_config.labels, model_config.label_data_strategy, "get_labels"),
        ):
            for i, item in enumerate(items):
                name = type(item).__name__
                if strategy == "read":
                    self._fingerprints[f"{prefix}_store[{i}]:{name}"] = (
                        self._store_fingerprint(getattr(item, getter)())
                    )
                else:
                    self._fingerprints[f"{prefix}[{i}]:{name}"] = (
                        self._dataset_variables_fingerprint(item)
                    )

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
        "symbol")`` with exactly the price axes. Every row is either all-NaN
        (hold) or all-finite (rebalance), and a rebalance row's gross
        exposure, the sum of absolute weights, is at most 1.

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
        all_nan = np.isnan(values).all(axis=1)
        all_finite = np.isfinite(values).all(axis=1)
        mixed = ~(all_nan | all_finite)
        if mixed.any():
            first = timestamps[int(np.argmax(mixed))]
            raise ValueError(
                f"{self.class_name}: weight row at {first} mixes NaN and finite "
                f"values; a row must be all-NaN (hold) or all-finite (rebalance)"
            )
        gross = np.where(all_finite, np.abs(np.nan_to_num(values)).sum(axis=1), 0.0)
        over = gross > 1 + 1e-9
        if over.any():
            idx = int(np.argmax(over))
            raise ValueError(
                f"{self.class_name}: weight row at {timestamps[idx]} has gross "
                f"exposure {gross[idx]} > 1"
            )

    @abstractmethod
    def _generate_signals(
        self, predictions: xr.Dataset, prices: xr.Dataset
    ) -> xr.Dataset:
        """Turn predictions and prices into target weights satisfying the contract.

        Both inputs share the price axes. The result must pass
        ``_assert_weights_contract``: a ``weight`` variable on
        ``(timestamp, symbol)`` whose rows are all-NaN on hold bars and
        all-finite with gross exposure at most 1 on rebalance bars.
        """

    @abstractmethod
    def _simulate(self, weights: xr.Dataset, prices: xr.Dataset) -> SimulationResult:
        """Simulate the portfolio; a signal at bar t fills at bar t+1's fill price."""

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

    @abstractmethod
    def _period_returns_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """Return return-based statistics restricted to the bars inside ``ranges``.

        A portfolio object cannot be sliced in time and re-simulating a
        sub-period would reset capital and change the path, so the
        statistics are computed from the return series of the same
        simulation, cut to ``ranges`` (inclusive pairs of bar labels) and
        concatenated in time order when there are several.
        """

    def _label_horizon_bars(self) -> int:
        """Return the largest ``n_forward_periods`` of the model's labels, in bars.

        The label of the ``train_end`` bar reads the next n bars of prices,
        so those bars are in-sample too. A label with no
        ``n_forward_periods`` in ``config.kwargs`` contributes 0 and logs a
        warning naming its class rather than guessing a value.
        """
        horizon = 0
        for label in self.config.model.config.labels:
            kwargs = label.config.kwargs
            if kwargs is None or "n_forward_periods" not in kwargs:
                logger.warning(
                    f"{self.class_name}: label {type(label).__name__} has no "
                    f"n_forward_periods in config.kwargs; it contributes a "
                    f"0-bar horizon to the effective training window"
                )
                continue
            horizon = max(horizon, int(kwargs["n_forward_periods"]))
        return horizon

    def _training_window(
        self, calendar: np.ndarray, train_start, train_end
    ) -> tuple[str, str] | None:
        """Return the effective training window as a pair of bar labels.

        The *effective training window* is every bar the trained model has
        seen: its training bars plus the label horizon after ``train_end``.

        The window is ``[train_start, train_end + label horizon]`` counted
        in calendar bars, not calendar days (a Friday ``train_end`` plus two
        bars is the next Tuesday). The trained bars are the ones the model
        layer's ``data.sel(timestamp=slice(train_start, train_end))``
        selects, found with the same pandas ``slice_indexer`` on the
        unchanged endpoints; the end is then advanced by the horizon and
        clamped to the last calendar bar. Both labels come from
        ``_bar_label``: dates for daily bars, full timestamps intraday.

        Returns ``None``, with a warning, when either date is ``None``; the
        metrics then record a null training window and every bar counts as
        out-of-sample.
        """
        if train_start is None or train_end is None:
            logger.warning(
                f"{self.class_name}: model config has train_start={train_start!r}, "
                f"train_end={train_end!r}; the effective training window is "
                f"unknown, so metrics record training_window as null and every "
                f"backtest bar as out-of-sample"
            )
            return None

        calendar = np.sort(np.asarray(calendar).astype("datetime64[ns]"))
        last = calendar.size - 1
        trained = pd.DatetimeIndex(calendar).slice_indexer(
            self._slice_bound(train_start), self._slice_bound(train_end)
        )
        start_idx = int(trained.start)
        end_idx = int(trained.stop) - 1 + self._label_horizon_bars()
        window_start = (
            self._bar_label(calendar[start_idx])
            if start_idx <= last
            else self._bar_label(train_start)
        )
        window_end = (
            self._bar_label(calendar[min(end_idx, last)])
            if end_idx >= 0
            else self._bar_label(train_end)
        )
        return window_start, window_end

    def _split_window(
        self, window_timestamps: np.ndarray, training_window: tuple[str, str] | None
    ) -> dict:
        """Split the window bars into in-sample and out-of-sample ranges.

        Returns three keys that are merged into the top level of the
        metrics: ``training_window`` (the input pair or ``None``),
        ``in_sample_range`` (first and last bar of the overlap between the
        window and the training window, or ``None``) and
        ``out_of_sample_ranges`` (the 0, 1 or 2 contiguous runs of bars
        outside the overlap). Bars are compared as exact timestamps. Both
        windows are intervals, so the overlap is one contiguous run; when it
        is non-empty a warning names both windows and the backtest goes on
        with the two parts reported separately.
        """
        timestamps = np.asarray(window_timestamps).astype("datetime64[ns]")
        split = {
            "training_window": training_window,
            "in_sample_range": None,
            "out_of_sample_ranges": [],
        }
        if timestamps.size == 0:
            return split

        if training_window is None:
            in_sample = np.zeros(timestamps.size, dtype=bool)
        else:
            first = self._label_ns(training_window[0])
            last = self._label_ns(training_window[1])
            in_sample = (timestamps >= first) & (timestamps <= last)

        pieces = []
        if in_sample.any():
            idx = np.flatnonzero(in_sample)
            lo, hi = int(idx[0]), int(idx[-1])
            split["in_sample_range"] = (
                self._bar_label(timestamps[lo]),
                self._bar_label(timestamps[hi]),
            )
            if lo > 0:
                pieces.append((0, lo - 1))
            if hi < timestamps.size - 1:
                pieces.append((hi + 1, timestamps.size - 1))
            window = (self._bar_label(timestamps[0]), self._bar_label(timestamps[-1]))
            logger.warning(
                f"{self.class_name}: backtest window {window[0]}..{window[1]} "
                f"overlaps the model's effective training window "
                f"{training_window[0]}..{training_window[1]} (train_start.."  # type: ignore[index]
                f"train_end + label horizon); bars "
                f"{split['in_sample_range'][0]}..{split['in_sample_range'][1]} "
                f"are in-sample. Continuing: in-sample and out-of-sample results "
                f"are reported separately"
            )
        else:
            pieces.append((0, timestamps.size - 1))

        split["out_of_sample_ranges"] = [
            (self._bar_label(timestamps[a]), self._bar_label(timestamps[b]))
            for a, b in pieces
        ]
        return split

    @classmethod
    def _in_ranges(cls, timestamps: np.ndarray, ranges: list[tuple[str, str]]) -> np.ndarray:
        """Return a mask of the ``timestamps`` inside any of the inclusive label pairs.

        Labels are ``_bar_label`` endpoints and are compared as exact
        timestamps (a date is midnight), never by day.
        """
        ts = np.asarray(timestamps).astype("datetime64[ns]")
        mask = np.zeros(ts.size, dtype=bool)
        for start, end in ranges:
            mask |= (ts >= cls._label_ns(start)) & (ts <= cls._label_ns(end))
        return mask

    def _turnover(self, simulation: SimulationResult) -> xr.DataArray:
        """Return the turnover of every bar that had fills, on a ``timestamp`` axis.

        Turnover is the one-sided traded notional of the bar (the sum of
        ``|size| x price`` over its orders) divided by the portfolio value
        of the previous bar, or ``config.init_cash`` for the first bar of
        the window. One-sided means a full buy-in from cash is about 1 and
        replacing the whole book (sell then buy) about 2. Using the value
        before the fills keeps the bar's own profit or loss out of the
        ratio. An empty array is returned when there are no orders.

        Raises
        ------
        ValueError
            If an order timestamp is not on the equity axis.
        """
        orders = simulation.orders
        if orders.sizes.get("order", 0) == 0:
            return xr.DataArray(
                np.array([], dtype=np.float64),
                dims=("timestamp",),
                coords={"timestamp": np.array([], dtype="datetime64[ns]")},
            )

        order_ts = orders["timestamp"].values.astype("datetime64[ns]")
        notional = np.abs(orders["size"].values.astype(np.float64)) * orders[
            "price"
        ].values.astype(np.float64)
        fill_bars, inverse = np.unique(order_ts, return_inverse=True)
        traded = np.zeros(fill_bars.size, dtype=np.float64)
        np.add.at(traded, inverse, notional)

        value_ts = simulation.value.timestamp.values.astype("datetime64[ns]")
        idx = np.searchsorted(value_ts, fill_bars)
        if (idx >= value_ts.size).any() or not np.array_equal(
            value_ts[np.minimum(idx, value_ts.size - 1)], fill_bars
        ):
            raise ValueError(
                f"{self.class_name}: an order timestamp is not on the equity "
                f"timestamp axis"
            )
        values = np.asarray(simulation.value.values, dtype=np.float64)
        previous = np.where(
            idx > 0, values[np.maximum(idx - 1, 0)], float(self.config.init_cash)
        )
        return xr.DataArray(
            traded / previous, dims=("timestamp",), coords={"timestamp": fill_bars}
        )

    def _turnover_summary(self, turnover: xr.DataArray, bar_interval) -> dict:
        """Summarize turnover as mean per rebalance, total and annualized.

        Annualized is the mean times bars per year (``MARKET.year_freq``
        divided by ``bar_interval``) divided by ``rebalance_periods``. With
        no fills the mean and the annualized value are NaN (written as
        null) and the sum is 0.
        """
        values = np.asarray(turnover.values, dtype=np.float64)
        interval = pd.Timedelta(bar_interval)
        bars_per_year = self.MARKET.year_freq(interval) / interval  # type: ignore[union-attr]
        mean = float(values.mean()) if values.size else float("nan")
        return {
            "mean_per_rebalance": mean,
            "sum": float(values.sum()),
            "annualized": mean * bars_per_year / self.config.rebalance_periods,
        }

    def _period_record_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """Return order, trade and turnover statistics restricted to ``ranges``.

        ``order_count``, ``fees_paid`` and ``traded_notional`` count the
        orders filled inside the ranges; ``closed_trade_count`` the trades
        with status ``"Closed"`` whose exit falls inside them;
        ``open_trade_count`` the trades still open at each range's end
        (entered on or before it, and not yet exited or exited after it);
        ``turnover`` is the ``_turnover_summary`` of the fill bars inside
        the ranges. Several ranges never overlap, so the counts add up
        across them.

        The trade counts use the same position-level definition as the
        whole-window statistics (one entry-to-flat round trip per symbol),
        so the per-range closed trades sum to the whole window's total.
        """
        orders = simulation.orders
        if orders.sizes.get("order", 0) > 0:
            in_range = self._in_ranges(orders["timestamp"].values, ranges)
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
                (closed & self._in_ranges(trades["exit_timestamp"].values, ranges)).sum()
            )
            # Exact timestamps, not days: on intraday data a trade entered
            # later on the range's last day is not "open at the range end".
            entry_ts = trades["entry_timestamp"].values.astype("datetime64[ns]")
            exit_ts = trades["exit_timestamp"].values.astype("datetime64[ns]")
            for _, end in ranges:
                end_ts = self._label_ns(end)
                open_at_end = (entry_ts <= end_ts) & (~closed | (exit_ts > end_ts))
                open_trade_count += int(open_at_end.sum())

        turnover = self._turnover(simulation)
        turnover = turnover.isel(
            timestamp=self._in_ranges(turnover.timestamp.values, ranges)
        )
        return {
            "order_count": order_count,
            "fees_paid": fees_paid,
            "traded_notional": traded_notional,
            "closed_trade_count": closed_trade_count,
            "open_trade_count": open_trade_count,
            "turnover": self._turnover_summary(turnover, simulation.bar_interval),
        }

    def _compute_metrics(
        self,
        simulation: SimulationResult,
        benchmark: SimulationResult | None,
        split: dict,
    ) -> dict:
        """Compute the whole, in-sample and out-of-sample metric blocks.

        All three come from the same continuous simulation; nothing here
        simulates a second time. ``whole`` is the engine's whole-window
        statistics plus the ``turnover`` summary and ``order_count``.
        ``in_sample`` and ``out_of_sample`` merge ``_period_returns_stats``
        and ``_period_record_stats`` over their ranges and are ``None`` when
        there is no such range. ``benchmark`` and ``relative`` appear only
        when a benchmark was simulated: ``benchmark`` holds the benchmark's
        ``symbol`` (display name) and ``axis_symbol`` plus the same three
        slices of return statistics (``_period_returns_stats`` over the whole
        window and over each slice's ranges), and ``relative`` holds the
        three slices of ``_relative_stats``, the strategy measured against
        the benchmark. Every key of ``split`` is copied to the top level;
        the in-sample ranges come from ``split["in_sample_ranges"]`` when
        present (the stitched curve) and from the single
        ``split["in_sample_range"]`` otherwise.
        """
        whole = self._engine_stats(simulation)
        whole["turnover"] = self._turnover_summary(
            self._turnover(simulation), simulation.bar_interval
        )
        # The number of fills: position-level trade counts do not say how
        # often we traded. A run with no fills has no `order` dimension, so
        # use `.sizes.get` rather than a subscript that would raise.
        whole["order_count"] = int(simulation.orders.sizes.get("order", 0))
        metrics: dict = {"whole": whole}

        def _slice(ranges: list[tuple[str, str]]) -> dict | None:
            """Return the merged period statistics over ``ranges``, or ``None``."""
            if not ranges:
                return None
            return {
                **self._period_returns_stats(simulation, ranges),
                **self._period_record_stats(simulation, ranges),
            }

        if "in_sample_ranges" in split:
            in_sample_ranges = list(split["in_sample_ranges"])
        else:
            in_sample_range = split["in_sample_range"]
            in_sample_ranges = [in_sample_range] if in_sample_range else []
        metrics["in_sample"] = _slice(in_sample_ranges)
        metrics["out_of_sample"] = _slice(list(split["out_of_sample_ranges"]))

        if benchmark is not None:
            timestamps = simulation.value.timestamp.values
            whole_range = [
                (self._bar_label(timestamps[0]), self._bar_label(timestamps[-1]))
            ]
            axis_symbol = self._benchmark_axis_symbol or ""
            metrics["benchmark"] = {
                "symbol": self._benchmark_display_name(axis_symbol, timestamps[-1]),
                "axis_symbol": axis_symbol,
                "whole": self._period_returns_stats(benchmark, whole_range),
                "in_sample": (
                    self._period_returns_stats(benchmark, in_sample_ranges)
                    if in_sample_ranges
                    else None
                ),
                "out_of_sample": (
                    self._period_returns_stats(
                        benchmark, list(split["out_of_sample_ranges"])
                    )
                    if split["out_of_sample_ranges"]
                    else None
                ),
            }
            metrics["relative"] = {
                "whole": self._relative_stats(simulation, benchmark, whole_range),
                "in_sample": (
                    self._relative_stats(simulation, benchmark, in_sample_ranges)
                    if in_sample_ranges
                    else None
                ),
                "out_of_sample": (
                    self._relative_stats(
                        simulation, benchmark, list(split["out_of_sample_ranges"])
                    )
                    if split["out_of_sample_ranges"]
                    else None
                ),
            }
        for key, value in split.items():
            metrics[key] = value
        return metrics

    def _benchmark_display_name(self, axis_symbol: str, as_of) -> str:
        """Return the benchmark's readable name, its ticker when the store has one.

        A CRSP benchmark store is keyed by PERMNO, a bare number, so the
        ticker sidecar beside the benchmark's own store (not the price
        store) names it as of the window's last bar. Without a sidecar the
        axis label is returned unchanged.
        """
        dataset = self.config.benchmark_dataset
        if dataset is None or not axis_symbol:
            return axis_symbol
        lookup = CrspTickerLookup.beside_store(dataset.config.zarr_file_path)
        return str(lookup.label([axis_symbol], pd.Timestamp(as_of).date())[0])

    def _relative_stats(
        self,
        simulation: SimulationResult,
        benchmark: SimulationResult,
        ranges: list[tuple[str, str]],
    ) -> dict:
        """Return the strategy's statistics relative to the benchmark over ``ranges``.

        Engine-independent: it reads only the two per-bar return series,
        which share the strategy's bars, cut to ``ranges`` (inclusive bar
        labels) and concatenated in time order. The *relative NAV* compounds
        ``(1 + r) / (1 + b)`` bar by bar (``r`` the strategy's return, ``b``
        the benchmark's); over the whole window it equals the strategy's
        value divided by the benchmark's, which is the excess-return curve
        the report draws. Returned keys, fractions rather than percents:

        - ``strategy_total_return`` / ``benchmark_total_return``: compounded
          returns of each series;
        - ``excess_return``: relative NAV at the end minus 1, the geometric
          excess (alpha in the everyday sense);
        - ``excess_return_annualized``: the same compounded to one year;
        - ``total_return_difference``: the arithmetic difference of the two
          total returns;
        - ``excess_max_drawdown``: the deepest fall of the relative NAV from
          its running peak (starting at 1), a negative fraction or 0;
        - ``tracking_error``: annualized standard deviation of ``r - b``;
        - ``information_ratio``: annualized mean of ``r - b`` over the
          tracking error;
        - ``beta`` and ``correlation`` of ``r`` on ``b``, and ``capm_alpha``,
          the annualized regression intercept ``mean(r) - beta * mean(b)``;
        - ``win_rate_vs_benchmark``: share of bars with ``r > b``;
        - ``bars``: number of bars used.

        A statistic that is undefined (fewer than two bars, a flat
        benchmark, zero tracking error) is NaN, which persists as null.

        Raises
        ------
        ValueError
            If the two return series are not on the same bars.
        """
        strategy = simulation.returns
        reference = benchmark.returns
        ts = strategy.timestamp.values.astype("datetime64[ns]")
        if not np.array_equal(ts, reference.timestamp.values.astype("datetime64[ns]")):
            raise ValueError(
                f"{self.class_name}: the benchmark returns are not on the "
                f"strategy's bars"
            )
        mask = self._in_ranges(ts, ranges)
        r = np.asarray(strategy.values, dtype=np.float64)[mask]
        b = np.asarray(reference.values, dtype=np.float64)[mask]
        finite = np.isfinite(r) & np.isfinite(b)
        r, b = r[finite], b[finite]
        n = int(r.size)
        interval = pd.Timedelta(simulation.bar_interval)
        bars_per_year = float(self.MARKET.year_freq(interval) / interval)  # type: ignore[union-attr]
        nan = float("nan")

        stats = {
            "strategy_total_return": nan,
            "benchmark_total_return": nan,
            "excess_return": nan,
            "excess_return_annualized": nan,
            "total_return_difference": nan,
            "excess_max_drawdown": nan,
            "tracking_error": nan,
            "information_ratio": nan,
            "beta": nan,
            "correlation": nan,
            "capm_alpha": nan,
            "win_rate_vs_benchmark": nan,
            "bars": n,
        }
        if n == 0:
            return stats

        strategy_total = float(np.prod(1.0 + r) - 1.0)
        benchmark_total = float(np.prod(1.0 + b) - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.cumprod((1.0 + r) / (1.0 + b))
            peak = np.maximum.accumulate(np.concatenate(([1.0], relative)))[1:]
            relative_drawdown = relative / peak - 1.0
        final = float(relative[-1])
        stats.update(
            strategy_total_return=strategy_total,
            benchmark_total_return=benchmark_total,
            excess_return=final - 1.0,
            excess_return_annualized=(
                float(final ** (bars_per_year / n) - 1.0) if final > 0 else nan
            ),
            total_return_difference=strategy_total - benchmark_total,
            excess_max_drawdown=float(min(np.nanmin(relative_drawdown), 0.0))
            if np.isfinite(relative_drawdown).any()
            else nan,
            win_rate_vs_benchmark=float(np.mean(r > b)),
        )
        if n < 2:
            return stats

        active = r - b
        tracking = float(np.std(active, ddof=1) * np.sqrt(bars_per_year))
        variance = float(np.var(b, ddof=1))
        stats["tracking_error"] = tracking
        if tracking > 0:
            stats["information_ratio"] = float(
                np.mean(active) * bars_per_year / tracking
            )
        if variance > 0:
            beta = float(np.cov(r, b, ddof=1)[0, 1] / variance)
            stats["beta"] = beta
            stats["capm_alpha"] = float(
                (np.mean(r) - beta * np.mean(b)) * bars_per_year
            )
            if np.std(r) > 0:
                stats["correlation"] = float(np.corrcoef(r, b)[0, 1])
        return stats

    def _report_notes(self) -> list[str]:
        """Return the notes attached to the report and to ``metrics.json``.

        The default note says that no borrow or short-financing cost is
        modelled. An engine that models borrow costs overrides this.
        """
        return [
            "No borrow or short-financing cost is modelled, so short-side "
            "returns are optimistic."
        ]

    @classmethod
    def _flatten_numeric(cls, prefix: str, value, out: dict) -> None:
        """Flatten the finite numeric leaves of a nested dict into ``out``.

        Booleans, NaN, infinities, timestamps and strings are skipped; the
        wandb summary only takes comparable numbers.
        """
        if isinstance(value, dict):
            for key, item in value.items():
                cls._flatten_numeric(f"{prefix}/{key}", item, out)
            return
        if isinstance(value, (bool, np.bool_)):
            return
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value) if isinstance(value, (float, np.floating)) else int(value)
            if np.isfinite(number):
                out[prefix] = number

    def _log_to_wandb(self, run_dir: Path, metrics: dict) -> None:
        """Log the metrics and report to a separate wandb run.

        Called only when ``use_wandb`` is set. The project is
        ``{ClassName}_backtest`` and the run is named after the run
        directory, apart from the model's training runs. The run config is
        ``get_config()`` (fingerprints included), the summary holds the
        finite numeric leaves of the ``whole``, ``in_sample`` and
        ``out_of_sample`` blocks as ``whole/<metric>`` and so on (plus the
        ``benchmark`` and ``relative`` blocks when a benchmark ran), and
        ``report`` carries the HTML report.
        """
        run = wandb.init(
            project=f"{self.class_name}_backtest",
            name=run_dir.name,
            config=to_jsonable(self.get_config()),
        )
        summary: dict = {}
        for block in ("whole", "in_sample", "out_of_sample", "benchmark", "relative"):
            if metrics.get(block) is not None:
                self._flatten_numeric(block, metrics[block], summary)
        run.summary.update(summary)
        run.log({"report": wandb.Html((run_dir / "report.html").read_text())})
        run.finish()

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
        """Return the "dates and settings" lines at the top of ``report.html``.

        Pure presentation: it reads ``block`` and ``self.config``, computes
        no statistics, and returns an ordered mapping of label to
        formatted text that the report module escapes and renders.
        ``block`` is the metric level carrying the split keys:
        ``run()`` passes the metrics themselves, ``run_cv()`` passes
        ``metrics["stitched"]``. Both spellings of the split are handled,
        the singular ``training_window`` / ``in_sample_range`` of a run and
        the plural ``training_windows`` / ``in_sample_ranges`` of the
        stitched curve.

        The window's first and last bar are taken from the range endpoints
        already in ``block``, so the page shows the same strings as
        ``metrics.json``; only when there is no range at all are they
        formatted from the timestamps. Every key is read with ``.get()`` and
        a missing value renders as a dash, never as ``None``, so a renamed
        split key degrades the page instead of raising inside the staged
        run directory. ``drawdown_span`` adds one line describing the
        deepest drawdown in trading days (bars), matching the markers on
        the equity chart.
        """

        def _pair(value) -> str | None:
            """Format a label pair as ``a .. b``, or ``None`` when empty."""
            return f"{value[0]} .. {value[1]}" if value else None

        def _pairs(values) -> str | None:
            """Format several label pairs joined by ``; ``, or ``None`` when empty."""
            rendered = [text for text in map(_pair, values or []) if text]
            return "; ".join(rendered) if rendered else None

        def _text(value) -> str:
            """Render ``value`` as text, with a dash for ``None``."""
            return DASH if value is None else str(value)

        ranges = [block.get("in_sample_range")]
        ranges += list(block.get("in_sample_ranges") or [])
        ranges += list(block.get("out_of_sample_ranges") or [])
        endpoints = [label for item in ranges if item for label in (item[0], item[1])]
        timestamps = simulation.value.timestamp.values
        if endpoints:
            first = min(endpoints, key=self._label_ns)
            last = max(endpoints, key=self._label_ns)
        else:
            first = self._bar_label(timestamps[0])
            last = self._bar_label(timestamps[-1])

        summary = {
            "Backtest window": f"{first} .. {last} ({timestamps.size} bars)",
            "Bar interval": str(pd.Timedelta(simulation.bar_interval)),
        }
        if "training_windows" in block:
            summary["Training windows"] = _text(_pairs(block.get("training_windows")))
            summary["In-sample ranges"] = _text(_pairs(block.get("in_sample_ranges")))
        else:
            summary["Training window"] = _text(_pair(block.get("training_window")))
            summary["In-sample range"] = _text(_pair(block.get("in_sample_range")))
        summary["Out-of-sample ranges"] = _text(
            _pairs(block.get("out_of_sample_ranges"))
        )
        benchmark = block.get("benchmark")
        if isinstance(benchmark, dict):
            dataset = self.config.benchmark_dataset
            where = "" if dataset is None else f" ({dataset.config.zarr_file_path})"
            summary["Benchmark"] = f"{_text(benchmark.get('symbol'))}{where}, buy and hold"
            whole = (block.get("relative") or {}).get("whole") or {}
            for label, key in (
                ("Excess return vs benchmark", "excess_return"),
                ("Excess max drawdown vs benchmark", "excess_max_drawdown"),
            ):
                value = whole.get(key)
                summary[label] = (
                    format(float(value), ".2%")
                    if isinstance(value, (int, float)) and np.isfinite(value)
                    else DASH
                )
        if drawdown_span:
            bars = drawdown_span.get("bars")
            depth = drawdown_span.get("depth")
            summary["Deepest drawdown (valley to recovery)"] = (
                f"{_text(drawdown_span.get('valley'))} .. "
                f"{_text(drawdown_span.get('end'))}, "
                f"{DASH if bars is None else f'{bars} trading days'}, "
                f"depth {DASH if depth is None else format(float(depth), '.2%')}, "
                f"{'recovered' if drawdown_span.get('recovered') else 'not recovered by the last bar'}"
            )
        summary["Model mode"] = _text(self.config.model_mode)
        summary["Rebalance every"] = f"{self.config.rebalance_periods} bars"
        # Selection fields exist only on cross-sectional configs; a
        # time-series backtester's report simply lacks these two lines.
        summary["Top N"] = _text(getattr(self.config, "top_n", None))
        summary["Direction"] = _text(getattr(self.config, "direction", None))
        summary["Initial cash"] = f"{float(self.config.init_cash):,.2f}"
        summary["Fees"] = _text(self.config.fees)
        if block.get("trained_checkpoint") is not None:
            summary["Trained checkpoint"] = _text(block["trained_checkpoint"])
        return summary

    def _report_and_persist(
        self,
        predictions: xr.Dataset,
        weights: xr.Dataset,
        simulation: SimulationResult,
        metrics: dict,
        *,
        benchmark: SimulationResult | None = None,
    ) -> Path:
        """Write a new run directory with every artifact of a ``run()``.

        The directory holds ``config.json``, ``weights.zarr``,
        ``equity.zarr`` (``value`` and ``returns``, plus ``benchmark_value``
        and ``benchmark_returns`` when a benchmark ran), ``liquidations.json``,
        ``metrics.json``, ``report.html`` and ``fingerprint.json``. Each
        JSON file goes through ``to_jsonable`` (NaN and infinities become
        null, timestamps become ISO strings) and is written atomically.

        ``report.html`` is self-contained: the summary lines from
        ``_report_summary``, a metric table with the ``whole``,
        ``in_sample`` and ``out_of_sample`` columns, and equity, drawdown and
        monthly-return charts on a shared time axis with the in-sample range
        shaded and the deepest drawdown marked (with a benchmark, its NAV is
        drawn beside the portfolio's and the excess-return and
        excess-drawdown rows are added), followed by the notes. The
        report module derives the table from whatever keys ``metrics``
        holds; nothing is selected or computed here, so a change in the
        metric set cannot make the report raise and discard the staged run.

        Returns
        -------
        Path
            The final run directory.
        """
        # Everything is written to a staging directory that is renamed into
        # place only after the last artifact succeeds.
        def _write(run_dir: Path, name: str) -> None:
            """Write every artifact of this run into ``run_dir`` titled ``name``."""
            write_json_atomically(
                run_dir / "config.json", to_jsonable(self.get_config()), indent=2
            )
            self._write_weights_and_equity(run_dir, weights, simulation, benchmark)
            write_json_atomically(
                run_dir / "liquidations.json",
                to_jsonable(simulation.liquidations),
                indent=2,
            )
            write_json_atomically(
                run_dir / "metrics.json", to_jsonable(metrics), indent=2
            )
            drawdown_span = self._drawdown_span(simulation)
            write_backtest_report(
                simulation.value,
                run_dir / "report.html",
                in_sample_range=metrics.get("in_sample_range"),
                notes=self._report_notes(),
                title=name,
                summary=self._report_summary(
                    simulation, metrics, drawdown_span=drawdown_span
                ),
                metrics=metrics,
                returns=simulation.returns,
                init_cash=self.config.init_cash,
                drawdown_span=drawdown_span,
                **self._benchmark_report_inputs(benchmark, metrics),
            )
            write_json_atomically(
                run_dir / "fingerprint.json", to_jsonable(self._fingerprints), indent=2
            )

        return self._persist_run_dir(_write)

    def _persist_run_dir(self, write) -> Path:
        """Create ``output_dir/{ClassName}_{timestamp}/`` and fill it through ``write``.

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

        final = Path(self.config.output_dir) / self._run_dir_name()
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

    @staticmethod
    def _benchmark_report_inputs(
        benchmark: SimulationResult | None, block: dict
    ) -> dict:
        """Return the benchmark keyword arguments of ``write_backtest_report``.

        Empty without a benchmark, so the report is drawn exactly as before.
        """
        if benchmark is None:
            return {}
        info = block.get("benchmark") or {}
        return {
            "benchmark_value": benchmark.value,
            "benchmark_returns": benchmark.returns,
            "benchmark_name": info.get("symbol") or "benchmark",
        }

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

    def _stitched_split(
        self, timestamps: np.ndarray, records: list[dict]
    ) -> dict:
        """Build the in-sample/out-of-sample split of the stitched curve from the folds.

        The stitched curve is out-of-sample by construction, since each fold
        trades only its own test segment, except for the first bars of each
        fold that overlap that fold's effective training window (the label
        horizon). The in-sample part is therefore a list: ``training_windows``
        holds every fold's effective training window in fold order,
        ``in_sample_ranges`` every fold's non-empty ``in_sample_range`` in
        fold order, and ``out_of_sample_ranges`` the contiguous runs of
        ``timestamps`` outside all of them. No singular ``in_sample_range``
        is produced, because several ranges do not fit one pair.
        """
        in_sample_ranges = [
            record["metrics"]["in_sample_range"]
            for record in records
            if record["metrics"]["in_sample_range"] is not None
        ]
        ts = np.asarray(timestamps).astype("datetime64[ns]")
        out_mask = ~self._in_ranges(ts, in_sample_ranges)
        pieces = []
        idx = np.flatnonzero(out_mask)
        if idx.size:
            breaks = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate(([idx[0]], idx[breaks + 1]))
            ends = np.concatenate((idx[breaks], [idx[-1]]))
            pieces = [
                (self._bar_label(ts[a]), self._bar_label(ts[b]))
                for a, b in zip(starts, ends)
            ]
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
    ) -> Path:
        """Write a new run directory with every artifact of a ``run_cv()``.

        The top level describes the stitched curve with the same files as a
        ``run()`` directory: ``config.json``, ``weights.zarr``,
        ``equity.zarr``, ``metrics.json`` (``stitched``, ``folds``,
        ``notes``), ``liquidations.json`` (``stitched`` plus per-fold
        ``folds``), ``fingerprint.json`` (the stitched window) and
        ``report.html``. The report receives ``metrics["stitched"]``, shades
        no in-sample range (the several in-sample ranges are listed in the
        summary lines and the notes) and marks the deepest drawdown of the
        stitched simulation. Each fold's own simulation is written under
        ``folds/fold_{i}/`` as ``weights.zarr`` and ``equity.zarr``, where
        ``i`` is the manifest's fold number.

        Returns
        -------
        Path
            The final run directory.
        """
        # As in run(): write to a staging directory, rename when complete.
        def _write(run_dir: Path, name: str) -> None:
            """Write every artifact of this CV run into ``run_dir`` titled ``name``."""
            write_json_atomically(
                run_dir / "config.json", to_jsonable(self.get_config()), indent=2
            )
            self._write_weights_and_equity(run_dir, weights, simulation, benchmark)
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
                run_dir / "liquidations.json",
                to_jsonable(
                    {
                        "stitched": simulation.liquidations,
                        "folds": [
                            {
                                "fold": record["fold"],
                                "liquidations": record["simulation"].liquidations,
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
            drawdown_span = self._drawdown_span(simulation)
            write_backtest_report(
                simulation.value,
                run_dir / "report.html",
                in_sample_range=None,
                notes=metrics["notes"],
                title=name,
                summary=self._report_summary(
                    simulation, metrics["stitched"], drawdown_span=drawdown_span
                ),
                metrics=metrics["stitched"],
                returns=simulation.returns,
                init_cash=self.config.init_cash,
                drawdown_span=drawdown_span,
                **self._benchmark_report_inputs(benchmark, metrics["stitched"]),
            )
            write_json_atomically(
                run_dir / "fingerprint.json", to_jsonable(self._fingerprints), indent=2
            )

        return self._persist_run_dir(_write)
