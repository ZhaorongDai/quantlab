"""Model layer: the training lifecycle shared by every model head.

A *head* is one concrete predictive model, for example a torch network or an XGBoost
regressor. It reads *features* (factor values) and learns *labels* (the
targets to predict, typically forward returns). Both arrive as *panels*: an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, with one data
variable per feature or label.

The module defines a three-level class hierarchy. ``BaseModel`` holds
everything that does not depend on the training framework: config
validation, requesting the factor and label panels over the model's date
range and collecting them into one dataset, the public ``train`` /
``train_cv`` / ``load`` / ``predict`` / ``predict_panel`` methods, the
checkpoint directory layout with its ``config.json`` sidecar file, and the
fold boundaries of rolling cross-validation. ``TorchModel`` is the PyTorch
variant: one cross-section of symbols per training step, each with its own
window of past bars, and ``.pth`` checkpoints. ``LibraryModel`` is the numpy
variant for tree models and other libraries that train themselves, with
``.joblib`` checkpoints. Both take one ``ModelConfig``; the reserved keys of
its ``hyperparameters`` are listed in ``RESERVED_HYPERPARAMETERS``. Concrete heads live in ``quantlab/torch_model`` and
``quantlab/library_model``.
"""

import copy
import dataclasses
import json
import os
import random
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
import wandb
import wandb.sdk
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from quantlab.backend import XrBackend
from quantlab.base.data import InsufficientHistoryError
from quantlab.enums.constant import Date
from quantlab.library_model.backend import MlBackend
from quantlab.library_model.data import Rows
from quantlab.torch_model.data import Batch, CrossSectionDataset, TrainingPanel
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.jsonable import to_jsonable
from quantlab.utils.metrics import regression_panel_metrics
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.split import purge_segments
from quantlab.utils.timer import Timer

from .config import ModelConfig

#: Keys of ``ModelConfig.hyperparameters`` a ``TorchModel`` reads itself:
#: ``epochs`` (the epoch cap, default 100), ``lr`` (learning rate of the
#: default ``_init_optim``, default ``1e-3``), and ``batch_size``,
#: ``num_workers``, ``panel_device`` and ``panel_dtype`` (reserved for the
#: torch data loader and training panel).
TORCH_RESERVED_HYPERPARAMETERS: frozenset[str] = frozenset(
    {"epochs", "lr", "batch_size", "num_workers", "panel_device", "panel_dtype"}
)

#: Keys of ``ModelConfig.hyperparameters`` a ``LibraryModel`` reads itself:
#: ``early_stopping`` (default False) and ``early_stopping_patience``
#: (default 5), the shipped library heads' native early stopping.
LIBRARY_RESERVED_HYPERPARAMETERS: frozenset[str] = frozenset(
    {"early_stopping", "early_stopping_patience"}
)

#: Every reserved key. ``_init_model`` receives them along with the head's
#: own keys, so a head never splats the whole dict into a network or a
#: library constructor; see ``BaseModel.head_hyperparameters``.
RESERVED_HYPERPARAMETERS: frozenset[str] = (
    TORCH_RESERVED_HYPERPARAMETERS | LIBRARY_RESERVED_HYPERPARAMETERS
)


class BaseModel(ABC):
    """Framework-agnostic base class of every model head.

    A head is configured with a list of factor objects (its features) and
    a list of label objects (its targets). ``collect()`` pulls both into one
    panel indexed by ``(timestamp, symbol)``; ``train()`` fits the head on the
    ``train_*`` dates of its config and writes a checkpoint; ``load()`` restores
    one; ``predict()`` and ``predict_panel()`` run inference. Those public entry
    points are implemented once, here, and are not overridden by any head.

    The training framework is the job of the two variants. ``TorchModel`` is the
    torch variant and ``LibraryModel`` the numpy variant. Both accept the one
    ``ModelConfig``, named by the class attribute ``config_cls``: the
    ``config`` setter checks it first, and the config loader reads it from
    the class before creating an instance. Each variant declares
    ``checkpoint_suffix``, the checkpoint file suffix. ``train`` and
    ``train_cv`` use it to name files, and ``load()`` uses it to reject a
    file of the wrong kind before building any model.

    Checkpoints are written under ``config.model_save_dir`` as
    ``{class}_trial_{timestamp}/{experiment}/{experiment}{suffix}``, with a
    ``config.json`` sidecar file next to the checkpoint. The sidecar holds
    the full config and a record of what the head was trained on.

    Parameters
    ----------
    config : ModelConfig
        The model configuration.

    Attributes
    ----------
    model : object or None
        The fitted model (an ``nn.Module`` or a library model object), or
        None before ``train()`` or ``load()``.
    data_backend : XrBackend
        Holds the panel built by ``collect()``.

    Raises
    ------
    TypeError
        If ``config`` is not an instance of ``config_cls``.

    Examples
    --------
    Given a head ``MyHead`` (a subclass of ``LibraryModel`` or ``TorchModel``), a
    factor object exposing variables ``f_a`` and ``f_b``, and a label
    object exposing ``ret``::

        >>> model = MyHead(ModelConfig(
        ...     factors=[factor], labels=[label],
        ...     model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> checkpoint = model.collect().train()
        >>> checkpoint.name
        'MyHead_total.joblib'
    """

    def __init__(self, config: ModelConfig):
        """Initialize the model; see the class docstring for parameters."""
        self.config = config
        self._set_random_seed(self.config.random_seed)

        self.model = None
        # The backtester checks the checkpoint before collecting data and
        # `load()` checks it again; remembering warnings logs each only once.
        self._emitted_load_warnings: set[str] = set()

        self.data_backend = XrBackend()
        self._wandb_recorder: wandb.sdk.wandb_run.Run = None  # type: ignore
        # Per-split (timestamps, ic, rank_ic) series of the current fit, filled
        # by `_compute_metrics` and written by `_write_evaluation_files`.
        self._ic_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    #: The config class every model accepts; the config loader reads it from
    #: the class to rebuild a model from ``config.json``.
    config_cls = ModelConfig

    #: The ``hyperparameters`` keys this variant reads itself, which
    #: ``head_hyperparameters`` leaves out.
    reserved_hyperparameters: frozenset[str] = frozenset()

    def head_hyperparameters(self, hyperparameters: dict) -> dict:
        """Return ``hyperparameters`` without the keys this variant reads itself.

        A head that forwards its hyperparameters to a network or a library
        constructor passes them through this first. Only the variant's own
        ``reserved_hyperparameters`` are dropped: a library head keeps
        ``lr``, which a torch head reserves, because the library may take it.
        The dict given is not modified.

        Examples
        --------
        >>> xgb_head.head_hyperparameters({"early_stopping": True, "lr": 0.1})
        {'lr': 0.1}
        >>> torch_head.head_hyperparameters({"epochs": 5, "hidden": 8})
        {'hidden': 8}
        """
        return {
            key: value
            for key, value in hyperparameters.items()
            if key not in self.reserved_hyperparameters
        }

    @property
    @abstractmethod
    def checkpoint_suffix(self) -> str:
        """The checkpoint file suffix, including the leading dot.

        Concrete variants satisfy it with a plain class attribute.

        Examples
        --------
        >>> LibraryModel.checkpoint_suffix
        '.joblib'
        """

    @staticmethod
    def _set_random_seed(seed: int):
        """Seed the framework-agnostic generators: ``random`` and numpy.

        ``TorchModel._set_random_seed`` adds the torch seeds on top of this.
        """
        random.seed(seed)
        np.random.seed(seed)

    def __repr__(self) -> str:
        """Return ``ClassName(config=...)``."""
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> ModelConfig:
        """The model's configuration object.

        Examples
        --------
        >>> model.config.model_save_dir
        checkpoints
        """
        return self._config

    @config.setter
    def config(self, config: ModelConfig):
        """Install a normalised copy of ``config`` after checking its class.

        Missing ``start_date`` / ``end_date`` fall back to the project-wide
        defaults, and ``name`` is set to the model's import path, on the
        model's own config; the config passed in is never edited. The
        factors and labels are left untouched: ``collect()`` passes the
        model's date range to each of them per request, so one factor object
        can serve several models with different ranges.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of ``config_cls``.

        Examples
        --------
        >>> model.config = ModelConfig(factors=[factor], labels=[label],
        ...                         model_save_dir="checkpoints",
        ...                         factor_data_strategy="read",
        ...                         label_data_strategy="read",
        ...                         start_date="2024-01-01")
        >>> model.config.end_date
        '2100-01-01'
        """
        self._config = self._normalize_config(config)

    def _normalize_config(self, config: ModelConfig) -> ModelConfig:
        """Return ``config`` checked against ``config_cls`` with its defaults filled in.

        A model variant that validates or completes its config overrides
        this, calls ``super()._normalize_config(config)`` first and returns a
        new config built with ``dataclasses.replace``. It reads the config
        it is given, never ``self.config``.

        Parameters
        ----------
        config : ModelConfig
            The config to normalise. It is not modified.

        Returns
        -------
        ModelConfig
            A new config with ``name``, ``start_date`` and ``end_date`` set.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of ``config_cls``, a factor is
            a label, or a label is not one (see ``_check_roles``).

        Examples
        --------
        >>> model._normalize_config(config).name
        'quantlab.library_model.xgb.XGBoostRegressor'
        """
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} requires a {self.config_cls.__name__}, "
                f"got {type(config).__name__}"
            )
        self._check_roles(config)
        return dataclasses.replace(
            config,
            name=self.import_path,
            start_date=(
                Date.START_DATE if config.start_date is None else config.start_date
            ),
            end_date=Date.END_DATE if config.end_date is None else config.end_date,
        )

    def _check_roles(self, config: ModelConfig) -> None:
        """Refuse a label among the factors or a factor among the labels.

        A label is anything with ``lookahead_bars()``, such as
        ``quantlab.label.forward.Forward``: it reads bars after t, so it must
        never be a feature, and a factor without it would be an unshifted
        target.

        Raises
        ------
        TypeError
            Naming the misplaced object and its position.
        """
        for i, factor in enumerate(config.factors):
            if callable(getattr(factor, "lookahead_bars", None)):
                raise TypeError(
                    f"{self.class_name}: factors[{i}] is the label "
                    f"{type(factor).__name__}, which reads bars after t; "
                    f"pass it in labels, not factors."
                )
        for i, label in enumerate(config.labels):
            if not callable(getattr(label, "lookahead_bars", None)):
                raise TypeError(
                    f"{self.class_name}: labels[{i}] is {type(label).__name__}, "
                    f"which is not a label; wrap it in "
                    f"quantlab.label.forward.Forward to predict it."
                )

    @property
    def num_times(self) -> int:
        """Number of timestamps in the collected panel.

        Examples
        --------
        >>> model.num_times
        40
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).timestamp.size

    @property
    def class_name(self) -> str:
        """The head's class name.

        Examples
        --------
        >>> model.class_name
        MyHead
        """
        return self.__class__.__name__

    @property
    def num_symbols(self) -> int:
        """Number of symbols in the collected panel.

        Examples
        --------
        >>> model.num_symbols
        3
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.size

    @property
    def symbols(self) -> list[str]:
        """Symbols of the collected panel, as a plain list.

        Examples
        --------
        >>> model.symbols
        ['S0', 'S1', 'S2']
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.values.tolist()

    @property
    def num_null(self) -> int:
        """Total number of NaN cells across every variable of the collected panel.

        Examples
        --------
        >>> model.num_null
        0
        """
        return int(
            self.data_backend.get_xarray_dataset(["timestamp", "symbol"])
            .isnull()
            .sum()
            .to_dataarray()
            .sum()
            .item()
        )

    @property
    def import_path(self) -> str:
        """Dotted ``module.QualName`` path of the head's class.

        Examples
        --------
        >>> model.import_path
        __main__.MyHead
        """
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def num_factors(self) -> int:
        """Number of feature variables across all configured factors.

        Examples
        --------
        >>> model.num_factors
        2
        """
        return len(self.get_factor_names())

    @property
    def num_labels(self) -> int:
        """Number of label variables across all configured labels.

        Examples
        --------
        >>> model.num_labels
        1
        """
        return len(self.get_label_names())

    def _request_panel(self, obj, strategy: str, start, end) -> xr.Dataset:
        """Return ``obj``'s panel from ``start`` to ``end`` by ``strategy``.

        ``"read"`` maps onto ``obj.read(start, end)`` and ``"cal"`` onto
        ``obj.compute(start, end)``.

        Raises
        ------
        ValueError
            If the strategy is neither ``"cal"`` nor ``"read"``.
        """
        match strategy:
            case "cal":
                return obj.compute(start, end)
            case "read":
                return obj.read(start, end)
            case _:
                raise ValueError(f"data strategy {strategy!r} is not supported")

    def _collect_panels(self, objs, strategy: str, start, end) -> xr.Dataset:
        """Request each object's panel from ``start`` to ``end`` and merge them.

        ``start`` and ``end`` default to the model's ``start_date`` and
        ``end_date``. Each panel comes from ``_request_panel``.
        """
        start = self.config.start_date if start is None else start
        end = self.config.end_date if end is None else end
        panels = [self._request_panel(obj, strategy, start, end) for obj in objs]
        return xr.combine_by_coords(panels)  # type: ignore

    def _collect_all_labels(self, start=None, end=None) -> xr.Dataset:
        """Gather every label in ``config.labels`` into one sorted panel.

        Each label's panel from ``start`` to ``end`` (by default the model's
        ``start_date`` and ``end_date``) is read from its store or computed,
        according to ``config.label_data_strategy``.

        Raises
        ------
        ValueError
            If the strategy is neither ``"cal"`` nor ``"read"``.
        """
        data = self._collect_panels(
            self.config.labels, self.config.label_data_strategy, start, end
        )
        return data.sortby(["timestamp", "symbol"])

    @property
    def warmup_bars(self) -> int:
        """Bars of features the model needs before the first bar it predicts.

        0 here: a row model predicts each bar from that bar alone.
        ``TorchModel`` returns ``window_bars - 1``.

        Examples
        --------
        >>> model.warmup_bars
        0
        """
        return 0

    def _collect_all_features(self, start=None, end=None) -> xr.Dataset:
        """Gather every factor in ``config.factors`` into one panel.

        Each factor's panel up to ``end`` (by default the model's
        ``end_date``) is read from its store or computed, according to
        ``config.factor_data_strategy``. It starts ``warmup_bars`` bars
        before ``start`` (by default the model's ``start_date``), counted on
        the factor's own dataset calendar (see ``_feature_start``), so the
        panel may begin before ``start``.

        Raises
        ------
        ValueError
            If the strategy is neither ``"cal"`` nor ``"read"``.
        """
        start = self.config.start_date if start is None else start
        end = self.config.end_date if end is None else end
        strategy = self.config.factor_data_strategy
        panels = [
            self._request_panel(
                factor, strategy, self._feature_start(factor, strategy, start), end
            )
            for factor in self.config.factors
        ]
        return xr.combine_by_coords(panels)  # type: ignore

    def _feature_start(self, factor, strategy: str, start, *, warn: bool = True):
        """Return ``start`` moved ``warmup_bars`` bars back on ``factor``'s dataset.

        With fewer bars before ``start`` in the dataset, or, under
        ``"read"``, in the factor's store, the request starts at the earliest
        one there is and, when ``warn`` is true, a ``UserWarning`` states the
        shortfall; the first windows then hold zero rows for the missing
        history. The backtester fingerprints the same range with
        ``warn=False``.
        """
        bars = self.warmup_bars
        if bars == 0:
            return start
        dataset = factor.config.dataset
        try:
            warm = dataset.bar_before(start, bars)
        except InsufficientHistoryError as exc:
            warm = dataset.bar_before(start, exc.available)
            if warn:
                self._warn_short_warmup(
                    factor, start, bars, f"its dataset holds only {exc.available}"
                )
        if strategy == "read":
            stored = factor.store_range()
            if stored is not None and warm < pd.Timestamp(stored[0]):
                warm = pd.Timestamp(stored[0])
                if warn:
                    self._warn_short_warmup(
                        factor, start, bars, f"its store starts at {stored[0]}"
                    )
        return warm

    def _warn_short_warmup(self, factor, start, bars: int, why: str) -> None:
        """Warn that fewer than ``bars`` feature bars exist before ``start``."""
        warnings.warn(
            f"{self.class_name}: the model needs {bars} warm-up bar(s) of "
            f"{type(factor).__name__} before {start!r}, but {why}; the first "
            f"windows hold zero rows for the missing bars.",
            UserWarning,
            stacklevel=5,
        )

    def collect(
        self,
    ) -> Self:
        """Load features and labels into the model's data backend.

        Each factor and label is asked for its panel from ``start_date`` to
        ``end_date``: ``read(start, end)`` from its store under the
        ``"read"`` strategy, ``compute(start, end)`` from its inputs under
        ``"cal"``. Factor panels start ``warmup_bars`` bars earlier (see
        ``_collect_all_features``); labels do not, so they are NaN on those
        bars. The factor and label panels are merged on the union of their
        ``(timestamp, symbol)`` coordinates, sorted on both axes, and stored
        in ``self.data_backend``. Call this before ``train()`` or ``train_cv()``.

        Returns
        -------
        Self
            The model itself, for chaining.

        Examples
        --------
        >>> model.collect().num_times
        40
        """
        feature = self._collect_all_features()
        label = self._collect_all_labels()
        with Timer(f"{self.class_name}: collect merge"):
            # Outer join: the features may start the model's warm-up earlier.
            d = xr.combine_by_coords([feature, label], join="outer")
            d = d.sortby(["timestamp", "symbol"])
        self.data_backend.to_internal(d)  # type: ignore
        return self

    @staticmethod
    def _variable_names(obj) -> tuple[str, ...]:
        """Return the variables a factor or label object actually provides.

        This is ``obj.get_factor_names()``, i.e. ``config.factor_names``: a
        factor pinned to a subset computes only that subset, and ``Factor``
        fills an unset ``config.factor_names`` from ``_get_factor_names()``.
        Objects without ``get_factor_names`` (lightweight stand-ins) fall back
        to ``_get_factor_names()``.

        Examples
        --------
        >>> alpha158.config.factor_names       # pinned to two features
        ('KMID', 'STD5')
        >>> BaseModel._variable_names(alpha158)
        ('KMID', 'STD5')
        """
        getter = getattr(obj, "get_factor_names", None)
        if callable(getter):
            return tuple(getter())
        return tuple(obj._get_factor_names())

    def get_factor_names(self):
        """Return the feature variable names, in factor order then variable order.

        Each factor contributes its pinned ``config.factor_names`` when set,
        else every name its class can produce (see ``_variable_names``).

        Examples
        --------
        >>> model.get_factor_names()
        ['f_a', 'f_b']
        """
        return list(
            chain.from_iterable(
                [self._variable_names(factor) for factor in self.config.factors]
            )
        )

    def get_label_names(self):
        """Return the label variable names, in label order then variable order.

        Examples
        --------
        >>> model.get_label_names()
        ['ret']
        """
        return list(
            chain.from_iterable(
                [self._variable_names(label) for label in self.config.labels]
            )
        )

    def get_config(self) -> dict:
        """Return the config as a JSON-ready dict with nested factor and label configs.

        The ``factors`` and ``labels`` entries are replaced by each object's own
        ``get_config()`` so the dict can be written to ``config.json`` and
        used to rebuild the model later.

        Examples
        --------
        >>> cfg = model.get_config()
        >>> sorted(cfg)[:3]
        ['end_date', 'factor_data_strategy', 'factors']
        """
        cfg = self.config.to_dict()
        cfg["factors"] = [factor.get_config() for factor in self.config.factors]  # type: ignore
        cfg["labels"] = [label.get_config() for label in self.config.labels]  # type: ignore
        return cfg  # type: ignore

    def _get_config_with_extra_kv(self, extra_kv: dict) -> dict:
        """Return ``get_config()`` updated with ``extra_kv``."""
        cfg = self.get_config()
        cfg.update(extra_kv)
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Rebuild a model, with its factors and labels, from a ``get_config()`` dict.

        The factors and labels are rebuilt from their own config dicts and
        the model is constructed with ``cls.config_cls``. Two keys a
        checkpoint's ``config.json`` carries as training records rather than
        config fields, ``resolved_hyperparameters`` and ``trained_on``, are
        dropped first; any other unknown key raises ``TypeError`` from the
        config class. The caller's dict is never modified.

        Parameters
        ----------
        config : dict
            The dict ``get_config()`` returned, or the ``config.json`` written
            beside a checkpoint.

        Returns
        -------
        Self
            An untrained model; call ``load`` to restore a checkpoint.

        Examples
        --------
        >>> rebuilt = FirstFeatureHead.from_config(model.get_config())
        >>> rebuilt.get_config() == model.get_config()
        True
        """
        # Imported here: the loaders import model classes by dotted path.
        from quantlab.utils.module import load_factor_from_config

        config = copy.deepcopy(config)
        # `resolved_hyperparameters` (what the library actually trained with)
        # and `trained_on` (factor/label names and training symbols) are
        # records, not config fields. Drop only those so any other unknown key
        # still fails.
        config.pop("resolved_hyperparameters", None)
        config.pop(cls.TRAINED_ON_KEY, None)
        config["factors"] = [load_factor_from_config(f) for f in config["factors"]]
        config["labels"] = [load_factor_from_config(f) for f in config["labels"]]
        return cls(cls.config_cls(**config))

    @property
    def labels(self) -> list:
        """The label objects the model is trained on, in config order.

        Examples
        --------
        >>> [label.get_factor_names() for label in model.labels]
        [('fwd_ret_1',)]
        """
        return list(self.config.labels)

    @property
    def train_bounds(self) -> tuple:
        """The configured training window, ``(train_start, train_end)``, before the purge.

        Examples
        --------
        >>> model.train_bounds
        ('2024-01-01', '2024-02-02')
        """
        return self.config.train_start, self.config.train_end

    @property
    def test_bounds(self) -> tuple:
        """The configured test window, ``(test_start, test_end)``.

        Examples
        --------
        >>> model.test_bounds
        ('2024-02-05', '2024-02-09')
        """
        return self.config.test_start, self.config.test_end

    @property
    def label_delays(self) -> tuple[int, ...]:
        """Each label's ``delay`` in bars, in the order of ``labels``.

        Examples
        --------
        >>> model.label_delays
        (1,)
        """
        return tuple(label.config.delay for label in self.config.labels)

    def predict_window(self, start, end) -> xr.Dataset:
        """Predict every bar from ``start`` to ``end`` from freshly requested features.

        Each factor is asked for its panel from ``warmup_bars`` bars before
        ``start`` (counted on its own dataset calendar) to ``end``, by
        ``config.factor_data_strategy``; the panel goes through
        ``predict_panel`` and the result is cut to ``start``..``end``. No
        config is changed. The model must be trained or loaded.

        Parameters
        ----------
        start, end : str
            First and last bar to predict, inclusive.

        Returns
        -------
        xr.Dataset
            One variable per label name on ``(timestamp, symbol)``.

        Examples
        --------
        >>> out = model.predict_window("2024-02-12", "2024-03-11")
        >>> list(out.data_vars), out.sizes["timestamp"]
        (['fwd_ret_1'], 21)
        """
        features = self._collect_all_features(start, end)
        return self.predict_panel(features).sel(timestamp=slice(start, end))

    def fingerprint_inputs(self, start, end) -> list[tuple]:
        """Return the data ``predict_window(start, end)`` reads, for fingerprinting.

        One entry ``(key, factor, strategy, first, last)`` per factor,
        ``strategy`` ``"cal"``, under the key ``factor[{i}]:{ClassName}``:
        the dataset inputs ``factor.compute(first, last)`` reads. Under the
        ``"read"`` factor strategy a second entry
        ``factor_store[{i}]:{ClassName}``, strategy ``"read"``, covers the
        store panel ``factor.read(first, last)``, which is what the
        predictions are built from. ``first`` is ``start`` moved back by the
        model's warm-up, the range ``predict_window`` requests.

        Parameters
        ----------
        start, end : str
            The window passed to ``predict_window``.

        Returns
        -------
        list[tuple]
            ``(key, factor, strategy, first, last)`` entries in factor order.

        Examples
        --------
        >>> [(key, strategy, first, last)
        ...  for key, _, strategy, first, last
        ...  in model.fingerprint_inputs("2024-02-12", "2024-03-11")]
        [('factor[0]:PastReturnFactor', 'cal', '2024-02-12', '2024-03-11')]
        """
        strategy = self.config.factor_data_strategy
        entries: list[tuple] = []
        for i, factor in enumerate(self.config.factors):
            name = type(factor).__name__
            first = self._feature_start(factor, strategy, start, warn=False)
            entries.append((f"factor[{i}]:{name}", factor, "cal", first, end))
            if strategy == "read":
                entries.append((f"factor_store[{i}]:{name}", factor, "read", first, end))
        return entries

    def training_fingerprint_inputs(self) -> list[tuple]:
        """Return the data ``collect()`` reads, for fingerprinting.

        One entry ``(key, item, strategy, first, last)`` per factor and per
        label over ``start_date`` to ``end_date``, a factor's ``first``
        moved back by the model's warm-up as ``collect()`` requests it. Under
        the ``"cal"`` strategy the keys are ``train_factor[{i}]:{ClassName}``
        and ``train_label[{i}]:{ClassName}`` (the dataset inputs ``compute``
        reads); under ``"read"`` they are ``train_factor_store[{i}]:...`` and
        ``train_label_store[{i}]:...`` (the store panels ``read`` returns).

        Returns
        -------
        list[tuple]
            ``(key, item, strategy, first, last)`` entries, factors first.

        Examples
        --------
        >>> entries = model.training_fingerprint_inputs()
        >>> [key for key, *_ in entries]
        ['train_factor[0]:PastReturnFactor', 'train_label[0]:ForwardReturnLabel']
        >>> entries[1][2:]
        ('cal', '2024-01-01', '2024-02-09')
        """
        config = self.config
        end = config.end_date
        entries: list[tuple] = []
        for prefix, items, strategy in (
            ("train_factor", config.factors, config.factor_data_strategy),
            ("train_label", config.labels, config.label_data_strategy),
        ):
            for i, item in enumerate(items):
                name = type(item).__name__
                start = config.start_date
                if prefix == "train_factor":
                    start = self._feature_start(item, strategy, start, warn=False)
                kind = f"{prefix}_store" if strategy == "read" else prefix
                entries.append((f"{kind}[{i}]:{name}", item, strategy, start, end))
        return entries

    def to_array(self, data: xr.Dataset, variables: list[str]) -> np.ndarray:
        """Convert a panel to a ``[num_times, num_symbols, len(variables)]`` array.

        Both axes are sorted, and the last axis follows the order of
        ``variables`` exactly (not alphabetical order), so ``x[..., i]`` is
        always ``variables[i]``.

        Parameters
        ----------
        data : xr.Dataset
            A dataset indexed by ``(timestamp, symbol)``.
        variables : list[str]
            The data variables to stack, in the wanted order.

        Examples
        --------
        >>> panel = model.data_backend.get_xarray_dataset(["timestamp", "symbol"])
        >>> x = model.to_array(panel, ["f_b", "f_a"])
        >>> x.shape
        (40, 3, 2)
        >>> np.allclose(x[..., 0], panel["f_b"].values)
        True
        """
        return (
            data[variables]
            .to_dataarray()
            .sortby(["timestamp", "symbol"])
            .sel(variable=variables)
            .transpose("timestamp", "symbol", "variable")
            .values
        )

    #: Key of the training record inside the checkpoint's ``config.json``. It is
    #: a record, not a config field; the config loader drops it when rebuilding.
    TRAINED_ON_KEY = "trained_on"

    @staticmethod
    def _jsonable_symbol(symbol):
        """Return ``symbol`` as a value ``json.dump`` accepts, keeping its kind.

        Integers (including numpy integers) become ``int``; everything else
        becomes ``str``. Dispatching rather than stringifying everything keeps
        an integer symbol axis recorded as JSON integers, so the record can be
        used with ``.sel()`` on that axis after it is read back.
        """
        if isinstance(symbol, bool):
            # bool is a subclass of int, but a boolean symbol axis is an
            # upstream defect, not an integer axis.
            return str(symbol)
        if isinstance(symbol, (int, np.integer)):
            return int(symbol)
        return str(symbol)

    def _save_model(self, p: Path):
        """Create the checkpoint directory, write ``config.json``, then the checkpoint.

        ``config.json`` holds ``get_config()`` plus a ``trained_on`` record with
        the feature names, label names and the sorted training symbols. The
        symbols are recorded in sorted order because ``to_array`` sorts the
        symbol axis.

        Raises
        ------
        ValueError
            If no model has been built.
        RuntimeError
            If the checkpoint directory already exists.
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not initialized")

        if p.parent.exists():
            raise RuntimeError(f"{p.parent} already exists")
        else:
            p.parent.mkdir(parents=True)

        symbols = [
            self._jsonable_symbol(symbol)
            for symbol in sort_symbol_axis(self.symbols)
        ]
        record = {
            "factor_names": [str(name) for name in self.get_factor_names()],
            "label_names": [str(name) for name in self.get_label_names()],
            "symbols": symbols,
        }
        with open(p.parent / Path("config.json"), "w") as f:
            json.dump({**self.get_config(), self.TRAINED_ON_KEY: record}, f, indent=4)

        self._write_checkpoint(p)

    def load(self, p: Path | str) -> Self:
        """Restore the model from a checkpoint file.

        The file suffix is checked first, so a ``.pth`` file is never handed to
        joblib and a ``.joblib`` file never to ``torch.load``. The feature and
        label variables recorded in the sidecar ``config.json`` must then match
        this model's declared variables, name for name and in order; neither
        torch nor a tree library would notice a permuted or substituted input
        by itself. Finally the checkpoint is loaded.

        Parameters
        ----------
        p : Path | str
            Path to a checkpoint file ending in ``checkpoint_suffix``.

        Returns
        -------
        Self
            The model itself, for chaining.

        Raises
        ------
        FileNotFoundError
            If ``p`` does not exist.
        ValueError
            If the suffix is wrong or the recorded variables differ
            from the model's declared variables.

        Examples
        --------
        >>> model.load(checkpoint) is model
        True
        """
        if isinstance(p, str):
            p = Path(p)

        if not p.exists():
            raise FileNotFoundError(f"{p} not found")

        if p.suffix != self.checkpoint_suffix:
            raise ValueError(
                f"Unsupported file type: {p.suffix!r}; {self.class_name} "
                f"checkpoints use {self.checkpoint_suffix!r} ({p})"
            )

        self.check_checkpoint(p)
        self._read_checkpoint(p)
        return self

    def _read_checkpoint_sidecar(self, p: Path) -> dict | None:
        """Parse the ``config.json`` next to checkpoint ``p``; None if absent.

        Raises
        ------
        ValueError
            If the file exists but is not a JSON object. A corrupt
            sidecar must not be treated as a missing one, since that would
            silently skip every check that depends on it.
        """
        sidecar = p.parent / "config.json"
        if not sidecar.is_file():
            return None
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError(
                f"{self.class_name}: {sidecar} is not a model config object"
            )
        return saved

    def check_checkpoint(self, p: Path | str) -> None:
        """Check a checkpoint's recorded variables against the model's own.

        The feature and label names (and their order) the checkpoint was
        trained on are taken from ``trained_on`` in its ``config.json``. If
        that record is missing, the older ``factors[]`` / ``labels[]``
        ``factor_names`` config fields are used instead; because a user can
        order those freely, they are compared as sets, and a difference in
        order only logs a warning. When neither is available the checkpoint
        is loaded as given, with a warning.

        Only ``config.json`` is read, so the check can run before any data is
        collected; ``load`` runs it too. Each distinct warning is logged once
        per model instance.

        Parameters
        ----------
        p : Path | str
            Path to a checkpoint file; its ``config.json`` sidecar is read.

        Raises
        ------
        ValueError
            If the recorded and declared variables differ, naming
            the checkpoint and both variable lists.

        Examples
        --------
        >>> model.check_checkpoint(checkpoint)  # trained on the same variables
        >>> other.check_checkpoint(checkpoint)
        Traceback (most recent call last):
        ValueError: FirstFeatureHead: checkpoint .../FirstFeatureHead_total.joblib
        was trained on factor variables ['past_ret_1'] (trained_on in its
        config.json), but this model declares ['past_ret_2']; loading it would
        feed the model different or permuted inputs
        """
        p = Path(p)
        saved = self._read_checkpoint_sidecar(p)
        record = saved.get(self.TRAINED_ON_KEY) if saved is not None else None
        legacy: list[tuple[str, list[str]]] = []
        unrecorded: list[str] = []
        for kind, key, entries_key, declared in (
            ("factor", "factor_names", "factors", self.get_factor_names()),
            ("label", "label_names", "labels", self.get_label_names()),
        ):
            current = [str(name) for name in declared]
            recorded = record.get(key) if isinstance(record, dict) else None
            if isinstance(recorded, list):
                recorded = [str(name) for name in recorded]
                source = "trained_on"
            else:
                recorded = (
                    self._legacy_variable_names(saved.get(entries_key))
                    if saved is not None
                    else None
                )
                if recorded is None:
                    unrecorded.append(kind)
                    continue
                legacy.append((kind, recorded))
                source = f"legacy {entries_key}[].factor_names"
            if recorded == current:
                continue
            if source != "trained_on" and sorted(recorded) == sorted(current):
                # The legacy config field cannot certify the training order,
                # so a pure reordering is a warning rather than a rejection.
                self._warn_load_record_once(
                    f"{self.class_name}: checkpoint {p}: the legacy "
                    f"{entries_key}[].factor_names record lists {kind} variables "
                    f"{recorded}, which differ only in order from this model's "
                    f"declared {current}; that config field cannot certify the "
                    f"training order, so the {kind} order is unchecked and the "
                    f"checkpoint is loaded as given"
                )
                continue
            consequence = (
                "feed the model different or permuted inputs"
                if kind == "factor"
                else "label the model's outputs with the wrong variables"
            )
            raise ValueError(
                f"{self.class_name}: checkpoint {p} was trained on {kind} "
                f"variables {recorded} ({source} in its config.json), but this "
                f"model declares {current}; loading it would {consequence}"
            )

        if legacy:
            kinds = " and ".join(kind for kind, _ in legacy)
            names = "; ".join(f"{kind} {names}" for kind, names in legacy)
            self._warn_load_record_once(
                f"{self.class_name}: checkpoint {p} has no trained_on record for "
                f"its {kinds} variables, so this model's declared variables were "
                f"checked against the legacy factors[]/labels[] factor_names "
                f"config field ({names}), a weaker record than trained_on: the "
                f"checkpoint was trained before that record existed"
            )
        if unrecorded:
            kinds = " and ".join(unrecorded)
            where = (
                "no config.json lies beside it"
                if saved is None
                else "its config.json records neither trained_on nor legacy "
                "factor_names for them"
            )
            self._warn_load_record_once(
                f"{self.class_name}: checkpoint {p}: the {kinds} variables it was "
                f"trained on cannot be checked against this model's declared "
                f"variables because {where}; loading it as given"
            )

    @staticmethod
    def _legacy_variable_names(entries) -> list[str] | None:
        """Flatten ``factor_names`` of ``factors[]`` / ``labels[]`` entries in order.

        Returns None when ``entries`` is not a list or any entry lacks
        ``factor_names``.
        """
        if not isinstance(entries, list):
            return None
        names: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("factor_names") is None:
                return None
            names.extend(str(name) for name in entry["factor_names"])
        return names

    def _warn_load_record_once(self, message: str) -> None:
        """Log ``message`` as a warning unless this instance already logged it."""
        if message in self._emitted_load_warnings:
            return
        self._emitted_load_warnings.add(message)
        logger.warning(message)

    def predict(
        self, data: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Return ``[T, S, L]`` predictions for a ``[T, S, F]`` input.

        The variant's ``_predict`` decides the accepted and returned types: the
        torch variant returns a tensor, the numpy variant an array.

        Raises
        ------
        ValueError
            If neither ``train()`` nor ``load()`` has been called.

        Examples
        --------
        >>> x = np.random.default_rng(0).standard_normal((5, 3, 2))
        >>> model.predict(x).shape
        (5, 3, 1)
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "Model not initialized, please call load() or train() first"
            )
        return self._predict(data)

    def predict_panel(self, features: xr.Dataset) -> xr.Dataset:
        """Predict from a feature panel and return a panel of the same shape.

        The result has one variable per label name, on ``("timestamp",
        "symbol")``, with coordinates taken from the sorted panel that was
        actually fed to ``to_array``, so values and coordinates cannot drift
        apart. Positions where every feature is NaN get NaN for every label:
        a head that fills NaN with zero would otherwise produce finite
        predictions for symbols that do not exist yet. Any symbol may be
        present, including ones the model never saw in training.

        The variant hook ``_predict_panel_array`` performs the numeric
        prediction. A torch head predicts bar t from the bars before it in
        ``features``, so pass ``warmup_bars`` bars before the first bar you
        need.

        Parameters
        ----------
        features : xr.Dataset
            A dataset containing every variable named by
            ``get_factor_names()``.

        Raises
        ------
        ValueError
            If a factor variable is missing, or if the prediction
            does not have shape ``[num_times, num_symbols, num_labels]``.

        Examples
        --------
        >>> feats = panel[["f_a", "f_b"]].isel(timestamp=slice(0, 5))
        >>> out = model.predict_panel(feats)
        >>> dict(out.sizes), list(out.data_vars)
        ({'timestamp': 5, 'symbol': 3}, ['ret'])
        """
        factors = self.get_factor_names()
        labels = self.get_label_names()
        missing = [name for name in factors if name not in features.data_vars]
        if missing:
            raise ValueError(
                f"{self.class_name}.predict_panel: features are missing factor "
                f"variable(s) {missing}"
            )

        if self.model is None:
            raise ValueError(
                "Model not initialized, please call load() or train() first"
            )
        feats = features[factors].sortby(["timestamp", "symbol"])
        x = self.to_array(feats, factors)
        y = np.asarray(
            self._predict_panel_array(x, feats.timestamp.values, feats.symbol.values),
            dtype=np.float64,
        )
        expected = (x.shape[0], x.shape[1], len(labels))
        if y.shape != expected:
            raise ValueError(
                f"{self.class_name}.predict_panel: expected prediction shape "
                f"{expected} [num_times, num_symbols, num_labels], got {y.shape}"
            )

        y = y.copy()
        y[np.isnan(x).all(axis=-1)] = np.nan
        return xr.Dataset(
            {
                name: (("timestamp", "symbol"), y[..., i])
                for i, name in enumerate(labels)
            },
            coords={
                "timestamp": feats.timestamp.values,
                "symbol": feats.symbol.values,
            },
        )

    def _predict_panel_array(
        self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Variant hook of ``predict_panel``: ``[T, S, F]`` array in, ``[T, S, L]`` out.

        ``timestamps`` and ``symbols`` are the coordinates of ``x``, for
        error messages. A plain method rather than an abstract one, so that the set of
        abstract methods of each variant stays unchanged.
        """
        raise NotImplementedError(
            f"{self.class_name} does not implement _predict_panel_array"
        )

    def _new_project_name(self) -> str:
        """Return a fresh ``{class}_trial_{%Y%m%d_%H%M%S_%f}`` directory name.

        The name is guaranteed not to exist under ``model_save_dir`` yet: if
        it does, ``_1``, ``_2``, ... are appended. Both ``train`` and
        ``train_cv`` go through here.
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = f"{self.class_name}_trial_{stamp}"
        root = Path(self.config.model_save_dir)
        name, suffix = base, 1
        while (root / name).exists():
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def train(self) -> Path:
        """Train once on the config's ``train_*`` / ``test_*`` dates and save.

        A new trial directory is created under ``model_save_dir``, and
        ``_train_into`` trains into its ``{class}_total`` subdirectory under a
        wandb run of that name in a project named after the trial directory.
        The metrics ``_fit`` returns are written to
        ``metrics.json`` beside the checkpoint's ``config.json``, with NaN and
        inf as null; a variant that returns no metrics writes no file.
        Beside it go the per-bar IC series (``ic_series.csv``) and the
        test-segment predictions (``test_predictions.zarr``), see
        ``_write_evaluation_files``.
        Returning the path lets a caller record exactly which model was
        trained and reload it later instead of retraining.

        Returns
        -------
        Path
            Absolute path of the checkpoint file, named
            ``{class}_total{checkpoint_suffix}``.

        Examples
        --------
        >>> checkpoint = model.train()
        >>> checkpoint.name, checkpoint.parent.name
        ('MyHead_total.joblib', 'MyHead_total')
        >>> sorted(json.loads((checkpoint.parent / "metrics.json").read_text()))[:4]
        ['test_ic', 'test_icir', 'test_loss', 'test_mae']
        >>> sorted(p.name for p in checkpoint.parent.iterdir())
        ['MyHead_total.joblib', 'config.json', 'ic_series.csv', 'metrics.json', 'test_predictions.zarr']
        """
        self._check_hyperparameters()
        project_name = self._new_project_name()
        experiment_name = f"{self.class_name}_total"
        checkpoint, _ = self._train_into(
            Path(self.config.model_save_dir) / project_name / experiment_name,
            project_name=project_name,
            experiment_name=experiment_name,
        )
        return checkpoint

    def _train_into(
        self,
        run_dir: Path | str,
        project_name: str,
        experiment_name: str,
        *,
        write_metrics: bool = True,
    ) -> tuple[Path, dict | None]:
        """Train, evaluate and save once into a caller-given run directory.

        The random generators are reseeded from ``config.random_seed`` first,
        so models trained one after another in one process do not share
        random state. A wandb run named ``experiment_name`` is opened in the
        wandb project ``project_name``, and the variant's ``_fit`` trains,
        evaluates and writes the checkpoint ``{experiment_name}{checkpoint_suffix}``
        and its ``config.json`` into ``run_dir``. When ``_fit`` returns
        metrics, ``metrics.json`` (only with ``write_metrics``, NaN and inf as
        null), ``ic_series.csv`` and ``test_predictions.zarr`` are written
        beside it, see ``_write_evaluation_files``. No trial directory is
        created: ``train`` and every ``train_cv`` fold pass the directory
        they lay out themselves.

        Parameters
        ----------
        run_dir : Path or str
            Directory that receives the checkpoint and the evaluation files.
            It must not exist yet; it is created with its parents.
        project_name : str
            wandb project of the run.
        experiment_name : str
            wandb run name, also the checkpoint file's stem.
        write_metrics : bool, default True
            Write ``metrics.json``; ``train_cv`` folds keep their metrics in
            ``cv_folds.json`` instead.

        Returns
        -------
        tuple[Path, dict or None]
            The absolute checkpoint path and the metrics ``_fit`` returned.

        Raises
        ------
        RuntimeError
            If ``run_dir`` already exists (see ``_save_model``).
        """
        self._set_random_seed(self.config.random_seed)
        checkpoint = (
            Path(run_dir) / f"{experiment_name}{self.checkpoint_suffix}"
        ).absolute()
        self._init_wandb(
            project_name=project_name,
            experiment_name=experiment_name,
        )
        self._ic_series = {}
        metrics = self._fit(checkpoint)
        if metrics is not None:
            if write_metrics:
                write_json_atomically(
                    checkpoint.parent / self.METRICS_FILENAME,
                    to_jsonable(metrics),
                    indent=2,
                )
            self._write_evaluation_files(checkpoint.parent)
        return checkpoint, metrics

    #: Name of the per-bar IC series file written beside ``metrics.json``.
    IC_SERIES_FILENAME = "ic_series.csv"
    #: Name of the zarr store holding the test-segment prediction panel.
    TEST_PREDICTIONS_FILENAME = "test_predictions.zarr"

    def _write_evaluation_files(self, run_dir: Path) -> None:
        """Write the per-bar IC series and the test-segment predictions of a fit.

        Called by ``_train_into`` right after ``_fit`` returned metrics, with
        the run's checkpoint directory.

        ``ic_series.csv`` has the columns ``split, timestamp, ic, rank_ic``:
        one row per bar of each evaluated split (``train``, ``val``,
        ``test``, in that order, each in time order), holding the per-bar IC
        and rank IC of the first label on raw values that ``_compute_metrics``
        averaged into ``{split}_ic`` / ``{split}_rank_ic`` and turned into
        ``{split}_icir`` / ``{split}_rank_icir``. A bar the IC skips (fewer
        than two symbols with both a finite prediction and a finite label, or
        a constant cross-section) has no row; a cell is empty only when one
        of the two values is defined and the other is not.

        ``test_predictions.zarr`` is ``predict_panel`` on the test segment:
        one variable per label on ``(timestamp, symbol)``, over the test bars
        and every collected symbol. The panel is predicted from the
        collected features with ``warmup_bars`` bars before the first test
        bar. No store is written when the test segment has no bars.
        """
        rows = []
        for split in ("train", "val", "test"):
            if split not in self._ic_series:
                continue
            stamps, ic, rank_ic = self._ic_series[split]
            keep = np.isfinite(ic) | np.isfinite(rank_ic)
            rows.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "timestamp": stamps[keep],
                        "ic": ic[keep],
                        "rank_ic": rank_ic[keep],
                    }
                )
            )
        frame = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(columns=["split", "timestamp", "ic", "rank_ic"])
        )
        path = run_dir / self.IC_SERIES_FILENAME
        staging = path.with_name(path.name + ".tmp")
        frame.to_csv(staging, index=False)
        os.replace(staging, path)

        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"]).sortby(
            ["timestamp", "symbol"]
        )
        test_stamps = self._fit_segments(data)[2].timestamp.values
        if len(test_stamps) == 0:
            return
        stamps = data.timestamp.values
        first, last = np.searchsorted(stamps, [test_stamps[0], test_stamps[-1]])
        features = data.isel(
            timestamp=slice(max(0, int(first) - self.warmup_bars), int(last) + 1)
        )
        predictions = self.predict_panel(features).sel(timestamp=test_stamps)
        predictions.to_zarr(run_dir / self.TEST_PREDICTIONS_FILENAME, mode="w")

    def _check_hyperparameters(self) -> None:
        """Validate the reserved hyperparameters this variant reads.

        Called first by ``train`` and ``train_cv``, before any W&B run or
        checkpoint directory is opened. The default checks nothing.

        Raises
        ------
        ValueError
            If a reserved hyperparameter is invalid.
        """

    def _purge_bars(self) -> int:
        """L, the largest ``lookahead_bars()`` among the model's labels."""
        return max(
            (label.lookahead_bars() for label in self.config.labels), default=0
        )

    def _fit_segments(
        self, data: xr.Dataset
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """Split the collected panel into purged train, validation and test panels.

        The training window ``[train_start, train_end]`` is cut by position:
        its first ``1 - val_size`` share of bars trains, the rest validates.
        The train, validation and test segments then go through
        ``purge_segments`` with L = ``_purge_bars()``, so each segment
        followed by another loses its last L bars and no fitted label reads
        a bar of the next segment.

        Returns
        -------
        tuple of xr.Dataset
            The train, validation and test panels. The validation panel has
            no timestamps when ``val_size`` is 0 or the purge empties it.

        Raises
        ------
        ValueError
            If no training bar is left, by ``val_size`` or by the purge.
        """
        config = self.config
        timestamps = np.sort(data.timestamp.values)
        (window,) = purge_segments(
            timestamps, [(config.train_start, config.train_end)], 0
        )
        split = int(len(window) * (1 - config.val_size))
        if split == 0:
            raise ValueError(
                f"Empty training segment: val_size={config.val_size} "
                f"leaves 0 of {len(window)} training timestamps for fitting."
            )
        segments = [(window[0], window[split - 1])]
        if split < len(window):
            segments.append((window[split], window[-1]))
        segments.append((config.test_start, config.test_end))

        lookahead = self._purge_bars()
        parts = purge_segments(timestamps, segments, lookahead)
        if len(parts[0]) == 0:
            raise ValueError(
                f"Empty training segment: purging the last {lookahead} bars "
                f"leaves 0 of {split} training timestamps for fitting."
            )
        train, test = parts[0], parts[-1]
        val = parts[1] if len(parts) == 3 else timestamps[:0]
        return (
            data.sel(timestamp=train),
            data.sel(timestamp=val),
            data.sel(timestamp=test),
        )

    @staticmethod
    def _cv_folds(timestamps, train_periods: int) -> list[dict]:
        """Compute the fold boundaries of a rolling walk-forward cross-validation.

        This is the only implementation of the fold arithmetic; both the
        sequential and the parallel branch of ``train_cv`` use it. With
        ``test_periods = train_periods // 5``, fold ``i`` has the training
        window ``[i * test_periods, i * test_periods + train_periods)`` and
        tests on the next ``test_periods`` positions. These are the windows
        before the purge: ``_fit`` drops the last L bars of the training
        window. The number of folds is
        ``max(1, (len(timestamps) - train_periods) // test_periods)``;
        a fold whose test segment runs past the end is logged and skipped, so
        the result can be empty.

        Returns
        -------
        list[dict]
            One dict per fold with keys ``fold``, ``train_start``,
            ``train_end``, ``test_start`` and ``test_end``. Dates are
            ``np.datetime_as_string`` values and both ends are inclusive.
        """
        total_periods = len(timestamps)
        test_periods = train_periods // 5  # Test set is 20% of training set
        n_splits = max(1, (total_periods - train_periods) // test_periods)

        folds: list[dict] = []
        for i in range(n_splits):
            train_start_idx = i * test_periods
            train_end_idx = train_start_idx + train_periods
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_periods

            if test_end_idx > total_periods:
                logger.warning(
                    f"Skipping fold {i}: test set exceeds data range"
                )
                continue

            folds.append(
                {
                    "fold": i,
                    "train_start": np.datetime_as_string(
                        timestamps[train_start_idx]
                    ),
                    "train_end": np.datetime_as_string(
                        timestamps[train_end_idx - 1]
                    ),
                    "test_start": np.datetime_as_string(
                        timestamps[test_start_idx]
                    ),
                    "test_end": np.datetime_as_string(
                        timestamps[test_end_idx - 1]
                    ),
                }
            )
        return folds

    @staticmethod
    def _purged_fold(timestamps, fold: dict, lookahead: int) -> dict:
        """The fold as fitted: its ``train_end`` moved to the last bar the purge keeps.

        Raises
        ------
        ValueError
            If the purge leaves the fold no training bar.
        """
        usable, _ = purge_segments(
            timestamps,
            [
                (fold["train_start"], fold["train_end"]),
                (fold["test_start"], fold["test_end"]),
            ],
            lookahead,
        )
        if len(usable) == 0:
            raise ValueError(
                f"Fold {fold['fold']}: purging the last {lookahead} bars "
                f"leaves no training bar; raise train_periods."
            )
        return {**fold, "train_end": np.datetime_as_string(usable[-1])}

    def _train_one_fold(self, fold: dict, record: dict, project_name: str) -> dict:
        """Train one fold on this instance and return its result dict.

        ``fold`` holds the dates before the purge, which ``_fit`` purges
        itself; ``record`` holds the purged dates actually fitted. The result
        is ``record`` plus ``experiment_name``, ``checkpoint`` (the absolute
        path of the fold's checkpoint, since the manifest may be read from
        another working directory) and whatever ``train_*`` / ``val_*`` /
        ``test_*`` metrics ``_fit`` returned. When there are metrics, the
        fold's checkpoint directory also gets ``ic_series.csv`` and
        ``test_predictions.zarr`` (see ``_write_evaluation_files``).
        """
        self.config = dataclasses.replace(
            self.config,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
        )

        experiment_name = f"{self.class_name}_cv_fold_{fold['fold']}"
        checkpoint, metrics = self._train_into(
            Path(self.config.model_save_dir) / project_name / experiment_name,
            project_name=project_name,
            experiment_name=experiment_name,
            write_metrics=False,
        )
        return {
            **record,
            "experiment_name": experiment_name,
            "checkpoint": str(checkpoint),
            **(metrics or {}),
        }

    def _train_fold_with_config(
        self, fold: dict, record: dict, project_name: str
    ) -> dict:
        """Train one fold on a deep copy of this instance (parallel branch).

        Each fold gets its own copy so folds share no config dates, model or
        wandb run; the price is one copy of the panel per job.
        """
        return copy.deepcopy(self)._train_one_fold(fold, record, project_name)

    #: Name of the metrics file ``train`` writes beside the checkpoint.
    METRICS_FILENAME = "metrics.json"
    #: Name of the fold manifest ``train_cv`` writes into the trial directory.
    CV_FOLDS_FILENAME = "cv_folds.json"
    #: Format version written into the manifest. Readers reject versions they
    #: do not know, so bump this whenever the manifest structure changes.
    #: Version 2 added the ``train_*`` / ``val_*`` fold metrics and ``cv_mean``.
    CV_FOLDS_FORMAT_VERSION = 2

    #: Keys every fold dict carries. The four dates start with ``train_`` or
    #: ``test_`` but are not metrics, and are excluded from CV means.
    _CV_FOLD_KEYS = frozenset(
        {"fold", "train_start", "train_end", "test_start", "test_end"}
    )

    #: Prefixes of the metric keys ``_fit`` returns, one per split.
    _METRIC_PREFIXES = ("train_", "val_", "test_")

    @staticmethod
    def _cv_mean_metrics(results: list[dict]) -> dict:
        """Average every ``train_*`` / ``val_*`` / ``test_*`` metric over folds.

        Each mean is keyed ``cv_mean_{key}``. Only finite numeric values
        count; a metric with no finite value in any fold averages to NaN.
        ``cv_n_folds`` is added. Returns an empty dict when no fold carries a
        metric, in which case ``train_cv`` opens no summary run.
        """
        keys: list[str] = []
        for result in results:
            for key, value in result.items():
                if (
                    key.startswith(BaseModel._METRIC_PREFIXES)
                    and key not in BaseModel._CV_FOLD_KEYS
                    and isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, bool)
                    and key not in keys
                ):
                    keys.append(key)
        if not keys:
            return {}

        means: dict = {}
        for key in keys:
            finite = [
                float(r[key])
                for r in results
                if key in r and np.isfinite(float(r[key]))
            ]
            means[f"cv_mean_{key}"] = (
                sum(finite) / len(finite) if finite else float("nan")
            )
        means["cv_n_folds"] = len(results)
        return means

    def train_cv(
        self,
        train_periods: int,
        parallel: bool = False,
        njobs: int = -1,
    ) -> list[dict]:
        """Run a rolling walk-forward cross-validation and return per-fold results.

        Walk-forward cross-validation trains on a window of past data and
        tests on the period right after it, then slides both forward, so a
        test period never precedes its training data. Folds are laid out by
        ``_cv_folds`` over the timestamps between ``config.start_date`` and
        ``config.end_date``. Each fold's training window loses its last L
        bars, L being the largest ``lookahead_bars()`` among the labels, so
        no fitted label reads a test-period bar. Every fold trains on
        its own dates, gets its own wandb run and its own checkpoint directory
        ``{class}_cv_fold_{i}/`` inside one trial directory. The fold means of
        every ``train_*`` / ``val_*`` / ``test_*`` metric, keyed
        ``cv_mean_{key}``, plus ``cv_n_folds`` are written to the summary of a
        separate ``{class}_cv_summary`` run.

        Before returning, the manifest ``cv_folds.json`` is written atomically
        into the trial directory as ``{"format_version": 2, "folds": [...],
        "cv_mean": {...}}``, where ``folds`` is the JSON form of the returned
        list and ``cv_mean`` the summary run's means (NaN and inf become
        null; ``cv_mean`` is empty when no fold has metrics). Its
        ``train_end`` is the last bar the purge keeps. Backtesters replay a
        CV run from that file.

        Parameters
        ----------
        train_periods : int
            Number of timestamps in each training segment. The test
            segment is one fifth of it.
        parallel : bool, default False
            Train the folds concurrently, each on a deep copy of this
            model, using a thread pool.
        njobs : int, default -1
            Number of threads for the parallel branch; ``-1`` uses all
            cores.

        Returns
        -------
        list[dict]
            One dict per fold: the purged fold boundaries, ``experiment_name``, the
            absolute ``checkpoint`` path and the fold's ``train_*``, ``val_*``
            (when the fold has a validation segment) and ``test_*`` metrics.

        Raises
        ------
        ValueError
            If ``train_periods`` is below 5 (the test segment would be
            empty), no timestamps fall inside the config's date range, or
            the purge leaves a fold no training bar, or a reserved
            hyperparameter is invalid (see ``_check_hyperparameters``).

        Examples
        --------
        >>> results = model.train_cv(train_periods=20)
        >>> len(results)
        4
        >>> results[0]["fold"], results[0]["checkpoint"].endswith("fold_0.joblib")
        (0, True)
        """
        self._check_hyperparameters()
        if train_periods < 5:
            raise ValueError(
                f"{self.class_name}: train_cv(train_periods={train_periods}) needs "
                f"at least 5 training bars, since each fold tests on "
                f"train_periods // 5 bars."
            )
        start_date = self.config.start_date
        end_date = self.config.end_date

        project_name = self._new_project_name()
        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"])

        data_in_range = data.sel(timestamp=slice(start_date, end_date))
        timestamps = data_in_range.timestamp.values

        if len(timestamps) == 0:
            raise ValueError(
                f"No data found between {start_date} and {end_date}"
            )

        logger.info(
            f"Starting CV from {start_date} to {end_date} with {train_periods} training periods"
        )

        folds = self._cv_folds(timestamps, train_periods)
        lookahead = self._purge_bars()
        records = [
            self._purged_fold(timestamps, fold, lookahead) for fold in folds
        ]

        logger.info(f"Total {len(folds)} folds will be created")
        for fold in records:
            logger.info(
                f"Fold {fold['fold']}: Train [{fold['train_start']} to {fold['train_end']}], Test [{fold['test_start']} to {fold['test_end']}]"
            )

        if parallel:
            logger.info(f"Starting parallel training of {len(folds)} folds")
            results = list(
                Parallel(n_jobs=njobs, backend="threading")(
                    delayed(self._train_fold_with_config)(
                        fold, record, project_name
                    )
                    for fold, record in zip(folds, records)
                )
            )
        else:
            results = [
                self._train_one_fold(fold, record, project_name)
                for fold, record in zip(folds, records)
            ]

        means = self._cv_mean_metrics(results)
        if means:
            self._init_wandb(
                project_name=project_name,
                experiment_name=f"{self.class_name}_cv_summary",
            )
            if self._wandb_recorder is not None:
                self._wandb_recorder.summary.update(means)
                self._wandb_recorder.finish()

        # The manifest is a converted copy; `results` is returned unchanged.
        write_json_atomically(
            Path(self.config.model_save_dir)
            / project_name
            / self.CV_FOLDS_FILENAME,
            {
                "format_version": self.CV_FOLDS_FORMAT_VERSION,
                "folds": to_jsonable(results),
                "cv_mean": to_jsonable(means),
            },
            indent=2,
        )

        return results

    def _assert_shape_match_y(self, data: np.ndarray | torch.Tensor):
        """Raise ``ValueError`` unless ``data`` is ``[T, num_symbols, num_labels]``."""
        num_symbols, num_labels = (
            self.num_symbols,
            self.num_labels,
        )
        if num_symbols != data.shape[1] or num_labels != data.shape[2]:
            raise ValueError(
                f"Train y shape mismatch: [num_times, {num_symbols}, {num_labels}] vs {data.shape}"
            )

    def _assert_shape_match_x(self, data: np.ndarray | torch.Tensor):
        """Raise ``ValueError`` unless ``data`` is ``[T, num_symbols, num_factors]``."""
        num_symbols, num_features = (
            self.num_symbols,
            self.num_factors,
        )
        if num_symbols != data.shape[1] or num_features != data.shape[2]:
            raise ValueError(
                f"Train x shape mismatch: [num_times, {num_symbols}, {num_features}] vs {data.shape}"
            )

    def _init_wandb(self, project_name: str, experiment_name: str):
        """Open a wandb run for this experiment with ``get_config()`` as its config."""
        self._wandb_recorder = wandb.init(
            project=project_name, name=experiment_name, config=self.get_config()
        )

    @abstractmethod
    def _fit(self, checkpoint: Path) -> dict | None:
        """Train, evaluate and save once, then finish the current wandb run.

        The checkpoint is saved to ``checkpoint``, with its ``config.json``
        beside it. Returns the metrics of every evaluated
        split as one dict keyed ``{split}_{metric}`` with split ``train``,
        ``val`` (only when there is a validation segment) and ``test``
        (``train`` writes it to ``metrics.json``, ``train_cv`` averages it),
        or None when the variant produces no metrics.
        """

    def _compute_metrics(
        self, y: np.ndarray, pred: np.ndarray, split: str, timestamps
    ) -> dict:
        """Return ``regression_panel_metrics`` for the first label on raw values.

        ``y`` and ``pred`` are ``[T, S, L]``; only label index 0 is scored.
        ``timestamps`` are the ``T`` bars of ``y`` in order. The per-bar IC
        and rank IC series behind ``ic`` / ``icir`` and ``rank_ic`` /
        ``rank_icir`` are kept under ``split`` for ``_write_evaluation_files``,
        so the file and ``metrics.json`` come from the same predictions.
        """
        metrics, series = regression_panel_metrics(
            pred[..., 0], y[..., 0], return_series=True
        )
        self._ic_series[split] = (
            np.asarray(timestamps),
            series["ic"],
            series["rank_ic"],
        )
        return metrics

    def _transform_target(self, y: torch.Tensor, training: bool):
        """Turn one bar's raw ``[S_t, L]`` labels into ``(target, keep)``.

        Shared by both variants: ``y`` is a float32 CPU tensor for a torch
        head and a library head alike, so one transform (a rank, a z-score,
        dropping the extremes) serves both.

        Called once per bar per fit, before training: ``training`` is True on
        the training bars and False on the validation and test bars. ``y``
        holds the labels of the bar's present symbols, NaN where missing.
        ``keep`` is None or ``[S_t]`` booleans; a symbol it drops leaves the
        loss but stays in the input as context. ``target`` has one row per
        symbol, or one per kept symbol. A symbol whose target is not finite
        in every label is masked out. The default returns ``(y, None)``.
        """
        return y, None

    def _training_panel(self, data: xr.Dataset) -> TrainingPanel:
        """Return the collected panel as a ``TrainingPanel`` with no target yet."""
        with Timer(f"{self.class_name}: to_array"):
            return TrainingPanel.from_arrays(
                self.to_array(data, self.get_factor_names()),
                timestamps=data.timestamp.values,
                symbols=data.symbol.values,
                y_raw=self.to_array(data, self.get_label_names()),
            )

    def _fill_target(self, panel: TrainingPanel, bars, training: bool) -> None:
        """Compute the training target of ``bars`` once, through ``_transform_target``.

        Each bar's raw labels of its present symbols go through the hook
        once; the result and its validity are written into ``panel.target``
        and ``panel.mask``. ``keep`` only clears ``mask``: a dropped symbol
        stays in the feature panel as context.

        Raises
        ------
        ValueError
            If the hook returns a ``keep`` or a target of the wrong shape.
        """
        num_labels = panel.y_raw.shape[-1]
        for t in bars:
            symbols = torch.nonzero(panel.present[t]).flatten()
            n = len(symbols)
            if not n:
                continue
            target, keep = self._transform_target(panel.y_raw[t, symbols], training)
            target = torch.as_tensor(target, dtype=torch.float32).cpu()
            if keep is None:
                keep = torch.ones(n, dtype=torch.bool)
            else:
                keep = torch.as_tensor(keep, dtype=torch.bool).cpu()
                if tuple(keep.shape) != (n,):
                    raise ValueError(
                        f"{self.class_name}._transform_target: keep must have {n} "
                        f"entries at bar {panel.timestamps[t]}, got {tuple(keep.shape)}"
                    )
                if target.shape[0] == int(keep.sum()) != n:
                    full = torch.full((n, num_labels), float("nan"))
                    full[keep] = target
                    target = full
            if tuple(target.shape) != (n, num_labels):
                raise ValueError(
                    f"{self.class_name}._transform_target: expected a target of shape "
                    f"{(n, num_labels)} at bar {panel.timestamps[t]}, "
                    f"got {tuple(target.shape)}"
                )
            valid = keep & torch.isfinite(target).all(dim=-1)
            panel.target[t, symbols] = torch.where(
                valid[:, None], target, torch.zeros_like(target)
            )
            panel.mask[t, symbols] = valid

    @abstractmethod
    def _predict(self, data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Variant implementation of ``predict``; ``self.model`` is guaranteed set."""

    @abstractmethod
    def _write_checkpoint(self, path: Path) -> None:
        """Write ``self.model`` to ``path``; the directory and sidecar already exist."""

    @abstractmethod
    def _read_checkpoint(self, path: Path) -> None:
        """Restore ``self.model`` from ``path``; the suffix is already checked."""


class TorchModel(BaseModel):
    """Torch variant: standard PyTorch datasets and loaders, every learning choice a hook.

    The base class assembles the collected panel as a ``TrainingPanel`` of
    torch tensors and computes the training target once per fit. The head's
    ``_dataset`` hook turns it into a PyTorch ``Dataset`` and ``_dataloader``
    batches it. The default is one cross-section per step (ADR 0006): one
    bar's present symbols, each with its last ``window_bars`` bars, so the
    network sees ``[S_t, N, F]``; S_t changes from bar to bar, so it must not
    depend on the order or the number of symbols, and a symbol the model
    never saw in training still gets a prediction. Every sample carries its
    ``where`` (timestamp and symbol index), through which the base puts
    predictions back into the panel for any sample shape.

    The base class owns every data contract: the warm-up, the training
    target and its mask, moving batches to the device, the epoch loop,
    evaluation under ``no_grad`` in eval mode, the ``train_*`` / ``val_*`` /
    ``test_*`` metrics on the raw first label, ``.pth`` checkpoints and
    prediction. A head writes three things:

    ``window_bars``
        N, the bars in each symbol's window.
    ``_init_model(num_features, num_labels, hyperparameters)``
        The ``nn.Module``; several networks go in an ``nn.ModuleDict``.
    ``_loss(output, batch)``
        The loss of one ``Batch`` from the network's raw output; count only
        ``batch.mask`` samples.

    and may override any of these, each of which has a working default:

    ``_dataset(panel, bars, training)``
        The PyTorch ``Dataset`` over ``bars``. Default:
        ``CrossSectionDataset``, one item per bar; ``SymbolSequenceDataset``
        gives Qlib-style ``(bar, symbol)`` samples batched to ``[B, N, F]``.
    ``_dataloader(dataset, training)``
        The ``DataLoader``. Default: ``batch_size`` and ``num_workers`` from
        the hyperparameters (``None`` and 0, one item per step), shuffled
        only in training with a generator seeded from ``random_seed``, the
        last batch never dropped.
    ``_transform_feature(x)``
        A batch's raw ``x``, NaN where a value or a bar is missing, to the
        network's input. Default: clip to ±3, NaN to 0.
    ``_transform_target(y, training)``
        One bar's raw ``[S_t, L]`` labels to ``(target, keep)``, computed
        once per fit; ``keep`` (or None) removes symbols from the loss only.
        Default: ``(y, None)``.
    ``_init_optim(model)``
        Default: Adam at ``hyperparameters["lr"]`` (``1e-3``), kept on
        ``self.optim``; a head may return anything its own
        ``_train_one_batch`` understands, such as a dict of optimizers.
    ``_train_one_batch(epoch, batch)``
        One optimisation step; returns the loss. Default: forward,
        ``_loss``, backward, gradient values clipped to ``grad_clip_value``
        (3.0; None disables), step.
    ``_val_one_batch(epoch, batch)``
        The evaluation loss of one batch. Default: ``_loss``.
    ``_test_one_batch(epoch, batch)``
        Called on every test batch after each epoch. Default: nothing.
    ``_forward(x)``
        The prediction, ``batch.mask.shape + (L,)``, from a transformed
        ``x``, used for metrics and ``predict_panel``. Default:
        ``self.model(x)``; override it when the network returns more than the
        prediction.
    ``_on_fit_start()``, ``_should_stop(epoch, train_loss, val_loss)``, ``_on_fit_end()``
        When to stop and which weights to keep. Default: run ``epochs``
        epochs, keep the last weights.

    ``epochs``, the epoch cap, is read from the hyperparameters (default
    100) and must be a positive integer; see ``RESERVED_HYPERPARAMETERS``
    for the other keys the base reads.

    The model's warm-up is N - 1 bars: every feature request, in training
    and in a backtest, starts that many bars earlier on each factor's
    dataset calendar, so the first requested bar has a full window.

    Examples
    --------
    A minimal head: a linear map of each symbol's latest bar::

        >>> class LastBarHead(TorchModel):
        ...     window_bars = 5
        ...     def _init_model(self, num_features, num_labels, hyperparameters):
        ...         return LastBarLinear(num_features, num_labels)
        ...     def _loss(self, output, batch):
        ...         return masked_mse(output, batch.y, batch.mask)
        >>> head = LastBarHead(ModelConfig(
        ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     hyperparameters={"epochs": 2},
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> head.collect().train().suffix
        '.pth'

    where ``LastBarLinear`` is ``nn.Linear(num_features, num_labels)``
    applied to ``x[:, -1]``.
    """

    checkpoint_suffix = ".pth"
    reserved_hyperparameters = TORCH_RESERVED_HYPERPARAMETERS

    #: Accepted ``hyperparameters["panel_device"]`` values.
    PANEL_DEVICES: tuple[str, ...] = ("auto", "cuda", "cpu")
    #: Accepted ``hyperparameters["panel_dtype"]`` values and the feature dtype each stores.
    PANEL_DTYPES: dict[str, torch.dtype] = {"float32": torch.float32, "float16": torch.float16}
    #: Largest share of free GPU memory ``panel_device="auto"`` gives the panel.
    PANEL_GPU_BUDGET: float = 0.5

    #: Device the current training or prediction panel was placed on, set by
    #: ``_place_panel``; None before the first placement.
    _panel_device: str | None = None

    #: Gradient value clip of the default ``_train_one_batch``; None disables.
    grad_clip_value: float | None = 3.0

    @property
    @abstractmethod
    def window_bars(self) -> int:
        """N, the number of bars in each symbol's input window.

        Examples
        --------
        >>> head.window_bars
        5
        """

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ) -> torch.nn.Module:
        """Build the network for ``[S_t, N, F]`` windows and ``num_labels`` labels.

        The base class moves it to ``device``. ``hyperparameters`` is the
        whole ``config.hyperparameters``, reserved keys such as ``epochs``
        and ``lr`` included: read the keys the network needs by name, or
        pass it through ``self.head_hyperparameters``, never splat it into the
        network as is.
        """

    @abstractmethod
    def _loss(self, output, batch: Batch) -> torch.Tensor:
        """Return the scalar loss of one batch.

        ``output`` is whatever the network returned for ``batch.x``; samples
        where ``batch.mask`` is False must not count.
        """

    def _dataset(self, panel: TrainingPanel, bars, training: bool) -> Dataset:
        """Return the PyTorch ``Dataset`` of ``bars``; default ``CrossSectionDataset``.

        ``panel`` is the whole collected panel, so windows may reach before
        the first of ``bars``. With ``training=True`` the dataset feeds the
        training steps and may keep only samples with a valid target; with
        ``training=False`` it feeds evaluation and prediction and must cover
        every present cell of ``bars`` exactly once. Every item is a
        ``Batch`` whose ``where`` places its samples in the panel.

        Examples
        --------
        >>> type(head._dataset(panel, bars=[3, 4], training=False)).__name__
        'CrossSectionDataset'
        """
        return CrossSectionDataset(panel, bars, self.window_bars, training)

    def _dataloader(self, dataset: Dataset, training: bool) -> DataLoader:
        """Return the ``DataLoader`` over ``dataset``.

        The default reads ``batch_size`` (default None: each item is one
        step, as the cross-section dataset needs) and ``num_workers``
        (default 0) from the hyperparameters, shuffles only when training
        with a generator seeded from ``config.random_seed``, and never drops
        the last batch. Memory is pinned only when the panel sits in CPU
        memory, workers load it and the model runs on CUDA.

        Examples
        --------
        >>> loader = head._dataloader(dataset, training=False)
        >>> loader.batch_size is None, loader.drop_last
        (True, False)
        """
        hyperparameters = self.config.hyperparameters
        # A random sampler refuses an empty dataset; a fit without a single
        # valid target trains on nothing.
        empty = hasattr(dataset, "__len__") and len(dataset) == 0  # type: ignore[arg-type]
        workers = self._num_workers
        return DataLoader(
            dataset,
            batch_size=hyperparameters.get("batch_size"),
            shuffle=training and not empty,
            num_workers=workers,
            generator=torch.Generator().manual_seed(self.config.random_seed),
            drop_last=False,
            pin_memory=bool(workers) and self._panel_device == "cpu" and self.device == "cuda",
        )

    def _transform_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Turn a batch's raw ``x`` into the network's input.

        ``x`` is float32 and holds NaN where a value is missing and on the
        window rows before a symbol's first bar. The result must have the
        same shape and be finite. The default clips to ±3 and replaces NaN
        with 0. Applied in training and prediction alike.
        """
        return torch.nan_to_num(x.clamp(-3.0, 3.0), nan=0.0)

    def _init_optim(self, model: torch.nn.Module):
        """Return the optimizer, kept on ``self.optim``.

        The default is Adam at ``hyperparameters["lr"]``, ``1e-3`` when unset.
        """
        lr = self.config.hyperparameters.get("lr", 1e-3)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def _train_one_batch(self, epoch: int, batch: Batch) -> torch.Tensor:
        """Run one optimisation step on one batch and return its loss.

        The base class has already called ``model.train()``. The mean of the
        returned losses is the epoch's ``train_loss``. The default runs the
        network, ``_loss``, ``backward``, clips gradient values to
        ``grad_clip_value`` and steps ``self.optim``.
        """
        self.optim.zero_grad()  # type: ignore[union-attr]
        loss = self._loss(self.model(batch.x), batch)  # type: ignore[misc]
        loss.backward()
        if self.grad_clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.grad_clip_value  # type: ignore[union-attr]
            )
        self.optim.step()  # type: ignore[union-attr]
        return loss.detach()

    def _val_one_batch(self, epoch: int, batch: Batch) -> torch.Tensor:
        """Return the evaluation loss of one batch; default ``_loss``.

        Called in eval mode under ``no_grad``, on the batches with at least
        one valid target. The mean over the validation batches is the
        epoch's ``val_loss``; the mean over each split's batches is its
        ``{split}_loss`` metric. With the default dataset a batch is one bar,
        so every bar weighs the same.
        """
        return self._loss(self.model(batch.x), batch)  # type: ignore[misc]

    def _test_one_batch(self, epoch: int, batch: Batch) -> None:
        """Evaluate one test batch after each epoch; the default does nothing."""

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the prediction for a transformed ``x``; default ``model(x)``.

        Its shape is the batch's ``mask`` shape plus ``L``: ``[S_t, L]`` for
        one cross-section.
        """
        return self.model(x)  # type: ignore[misc]

    def _on_fit_start(self) -> None:
        """Prepare per-fit state; called once the network and optimizer exist.

        The default does nothing. Stopping state belongs here, so every fit
        and every cross-validation fold starts fresh.
        """

    def _should_stop(
        self, epoch: int, train_loss: float, val_loss: float | None
    ) -> bool:
        """Return True to stop after this epoch; the default never stops early.

        ``val_loss`` is None without a validation segment. Training never
        runs past ``epochs``.
        """
        return False

    def _on_fit_end(self) -> None:
        """Choose the weights to keep, after the last epoch; the default keeps the last."""

    def _check_hyperparameters(self) -> None:
        """Refuse an invalid ``epochs``, ``panel_device`` or ``panel_dtype``."""
        self.epochs
        self._panel_settings()

    @property
    def _num_workers(self) -> int:
        """``hyperparameters["num_workers"]``, 0 when unset."""
        return int(self.config.hyperparameters.get("num_workers", 0) or 0)

    def _panel_settings(self) -> tuple[str, torch.dtype]:
        """Return the validated ``(panel_device, feature dtype)`` of the hyperparameters.

        Raises
        ------
        ValueError
            If ``panel_device`` or ``panel_dtype`` is not an accepted value,
            or ``panel_device="cuda"`` is asked for with ``num_workers > 0``
            (a worker process cannot index a CUDA tensor) or without a CUDA
            device.
        """
        hyperparameters = self.config.hyperparameters
        device = hyperparameters.get("panel_device", "auto")
        dtype = hyperparameters.get("panel_dtype", "float32")
        if not isinstance(device, str) or device not in self.PANEL_DEVICES:
            raise ValueError(
                f"{self.class_name}: hyperparameters['panel_device'] must be one of "
                f"{list(self.PANEL_DEVICES)}, got {device!r}"
            )
        if not isinstance(dtype, str) or dtype not in self.PANEL_DTYPES:
            raise ValueError(
                f"{self.class_name}: hyperparameters['panel_dtype'] must be one of "
                f"{list(self.PANEL_DTYPES)}, got {dtype!r}"
            )
        if device == "cuda" and self._num_workers:
            raise ValueError(
                f"{self.class_name}: hyperparameters['panel_device']='cuda' cannot be "
                f"combined with num_workers={self._num_workers}, because a loader "
                f"worker process cannot index a CUDA tensor; use num_workers=0, or "
                f"panel_device='cpu' or 'auto'"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"{self.class_name}: hyperparameters['panel_device']='cuda' but no "
                f"CUDA device is available"
            )
        return device, self.PANEL_DTYPES[dtype]

    def _panel_bytes(self, panel: TrainingPanel) -> int:
        """Bytes ``panel`` takes once its features are stored at ``panel_dtype``."""
        _, dtype = self._panel_settings()
        feature_bytes = panel.x.numel() * torch.empty((), dtype=dtype).element_size()
        return feature_bytes + sum(
            tensor.numel() * tensor.element_size()
            for tensor in (panel.target, panel.y_raw, panel.mask, panel.present)
        )

    def _resolve_panel_device(self, nbytes: int) -> str:
        """Return the device, ``"cuda"`` or ``"cpu"``, for a panel of ``nbytes`` bytes.

        ``panel_device="cpu"`` and ``"cuda"`` are obeyed. ``"auto"`` picks
        the GPU when CUDA is available, no loader workers are asked for, and
        the panel takes at most ``PANEL_GPU_BUDGET`` of the free GPU
        memory; it logs the choice.

        Raises
        ------
        ValueError
            As ``_panel_settings``.

        Examples
        --------
        >>> cpu_head.config.hyperparameters["panel_device"]
        'cpu'
        >>> cpu_head._resolve_panel_device(10_000)
        'cpu'
        """
        setting, _ = self._panel_settings()
        if setting != "auto":
            return setting
        if not torch.cuda.is_available():
            return "cpu"
        if self._num_workers:
            logger.info(
                f"{self.class_name}: training panel in CPU memory, because "
                f"num_workers={self._num_workers} loader workers read it"
            )
            return "cpu"
        free, _ = torch.cuda.mem_get_info()
        choice = "cuda" if nbytes <= self.PANEL_GPU_BUDGET * free else "cpu"
        logger.info(
            f"{self.class_name}: training panel of {nbytes / 2**30:.2f} GiB "
            f"{'on the GPU' if choice == 'cuda' else 'in CPU memory'} "
            f"({free / 2**30:.2f} GiB of GPU memory free, budget "
            f"{self.PANEL_GPU_BUDGET:.0%})"
        )
        return choice

    def _place_panel(self, panel: TrainingPanel) -> TrainingPanel:
        """Return ``panel`` on the resolved device with features at ``panel_dtype``.

        Records the device on ``_panel_device``. Called after the training
        target is filled, and on every prediction panel.

        Raises
        ------
        ValueError
            As ``_panel_settings``, or if a feature is too large for
            ``panel_dtype`` (see ``TrainingPanel.to``).
        """
        _, dtype = self._panel_settings()
        device = self._resolve_panel_device(self._panel_bytes(panel))
        try:
            placed = panel.to(device, dtype)
        except ValueError as exc:
            raise ValueError(f"{self.class_name}: {exc}") from None
        self._panel_device = device
        return placed

    @property
    def epochs(self) -> int:
        """The epoch cap, ``hyperparameters["epochs"]``, 100 when unset.

        Raises
        ------
        ValueError
            If the value is not a positive integer (a bool is refused too).

        Examples
        --------
        >>> head.epochs
        100
        """
        epochs = self.config.hyperparameters.get("epochs", 100)
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError(
                f"{self.class_name}: hyperparameters['epochs'] must be a positive "
                f"integer, got {epochs!r}"
            )
        return int(epochs)

    @property
    def warmup_bars(self) -> int:
        """``window_bars - 1``: bars requested before the start of every feature panel.

        Examples
        --------
        >>> head.warmup_bars
        4
        """
        return int(self.window_bars) - 1

    @staticmethod
    def _set_random_seed(seed: int):
        """Seed ``random``, numpy and torch (CPU and CUDA); make cuDNN deterministic."""
        BaseModel._set_random_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    @property
    def device(self) -> str:
        """``"cuda"`` when a CUDA device is available, else ``"cpu"``.

        Examples
        --------
        >>> head.device
        'cpu'
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _prepare(self, batch: Batch) -> Batch:
        """Move ``batch`` to ``device`` and run ``_transform_feature`` on its ``x``.

        Raises
        ------
        ValueError
            If the transform changes the shape or leaves a non-finite value.
        """
        batch = Batch(*batch).to(self.device)
        raw = batch.x.float()
        x = self._transform_feature(raw)
        if tuple(x.shape) != tuple(raw.shape) or not bool(torch.isfinite(x).all()):
            raise ValueError(
                f"{self.class_name}._transform_feature must return a finite "
                f"tensor of shape {tuple(raw.shape)}"
            )
        return batch._replace(x=x)

    def _loader(self, panel: TrainingPanel, bars, training: bool) -> DataLoader:
        """Return the head's loader over the head's dataset of ``bars``."""
        return self._dataloader(self._dataset(panel, bars, training), training)

    @staticmethod
    def _mean_loss(losses) -> float:
        """Mean of the ``float`` of each loss; NaN when there is none."""
        values = [float(loss) for loss in losses]
        return float(np.mean(values)) if values else float("nan")

    def _train_epoch(self, epoch: int, loader: DataLoader) -> float:
        """Call ``_train_one_batch`` on every training batch; their mean loss."""
        self.model.train()  # type: ignore[union-attr]
        return self._mean_loss(
            self._train_one_batch(epoch, self._prepare(batch)) for batch in loader
        )

    def _eval_loss(self, epoch: int, loader: DataLoader) -> float:
        """Per-bar ``_val_one_batch`` averaged over bars; NaN without a valid target.

        Runs in eval mode without gradients. A batch holding several bars is
        split by bar along its first dimension, and a bar spread over several
        batches averages its pieces weighted by their valid samples, so every
        bar weighs the same whatever its number of symbols or batches.

        Raises
        ------
        ValueError
            If a batch mixes bars but cannot be split along its first
            dimension (see ``_split_by_bar``).
        """
        pieces: dict[int, list[tuple[float, int]]] = {}
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            for raw in loader:
                for bar, piece in self._split_by_bar(self._prepare(raw)):
                    valid = int(piece.mask.sum())
                    loss = float(self._val_one_batch(epoch, piece))
                    pieces.setdefault(bar, []).append((loss, valid))
        return self._mean_loss(
            sum(loss * n for loss, n in parts) / sum(n for _, n in parts)
            for parts in pieces.values()
        )

    def _split_by_bar(self, batch: Batch) -> list[tuple[int, Batch]]:
        """Return ``(bar, piece)`` for each bar with a valid sample in ``batch``.

        A batch of one bar, such as a cross-section, is returned whole.

        Raises
        ------
        ValueError
            If the batch mixes bars but its ``mask`` is not one-dimensional or
            its ``x`` does not start with the sample dimension.
        """
        times = batch.where[0]
        bars = torch.unique(times[batch.mask]).tolist()
        if len(bars) <= 1:
            return [(int(bar), batch) for bar in bars]
        if batch.mask.ndim != 1 or batch.x.shape[0] != batch.mask.shape[0]:
            raise ValueError(
                f"{self.class_name}: a batch mixing bars needs a one-dimensional "
                f"mask and an x starting with the sample dimension to be scored "
                f"per bar; got mask {tuple(batch.mask.shape)} and x "
                f"{tuple(batch.x.shape)}"
            )
        split = []
        for bar in bars:
            rows = times == bar
            split.append((int(bar), Batch(
                x=batch.x[rows], y=batch.y[rows], mask=batch.mask[rows],
                y_raw=batch.y_raw[rows], where=(times[rows], batch.where[1][rows]),
            )))
        return split

    def _test_epoch(self, epoch: int, loader: DataLoader) -> None:
        """Call ``_test_one_batch`` on every test batch, in eval mode without gradients."""
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            for batch in loader:
                self._test_one_batch(epoch, self._prepare(batch))

    def _predict_panel_bars(self, panel: TrainingPanel, bars) -> np.ndarray:
        """Return ``[T, S, L]`` predictions for ``bars``, NaN everywhere else.

        The ``training=False`` dataset and loader are run under ``no_grad``
        in eval mode, and each ``_forward`` output is scattered into the
        panel through the batch's ``where``.

        Raises
        ------
        ValueError
            If ``_forward`` returns something other than a tensor of shape
            ``mask.shape + (L,)``, or a present cell of ``bars`` is left
            unpredicted or predicted twice, or a cell outside them is
            predicted; the error names the bar.
        """
        num_times, num_symbols = panel.present.shape
        out = torch.full((num_times, num_symbols, self.num_labels), float("nan"))
        count = torch.zeros((num_times, num_symbols), dtype=torch.int64)
        loader = self._loader(panel, bars, training=False)
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            for raw in loader:
                batch = self._prepare(raw)
                pred = self._forward(batch.x)
                expected = tuple(batch.mask.shape) + (self.num_labels,)
                if not isinstance(pred, torch.Tensor) or tuple(pred.shape) != expected:
                    got = tuple(pred.shape) if isinstance(pred, torch.Tensor) else type(pred)
                    raise ValueError(
                        f"{self.class_name}._forward must return a tensor shaped like "
                        f"the batch's mask plus the labels, {list(expected)}, got {got}"
                    )
                t_idx, s_idx = (index.cpu() for index in batch.where)
                out[t_idx, s_idx] = pred.detach().float().cpu()
                count.index_put_((t_idx, s_idx), torch.ones_like(t_idx), accumulate=True)
        requested = torch.zeros(num_times, dtype=torch.bool)
        requested[torch.as_tensor(np.asarray(bars, dtype=np.int64))] = True
        expected_cells = panel.present.cpu() & requested[:, None]
        problems = (
            (count > 1, "predicted twice"),
            (expected_cells & (count == 0), "left unpredicted"),
            (~expected_cells & (count > 0),
             "predicted outside the present cells of the bars asked for"),
        )
        for problem, what in problems:
            if bool(problem.any()):
                t, s = (int(i) for i in torch.nonzero(problem)[0])
                stamp = panel.timestamps[t]
                if isinstance(stamp, np.datetime64):
                    stamp = pd.Timestamp(stamp).isoformat()
                raise ValueError(
                    f"{self.class_name}: symbol {str(panel.symbols[s])!r} at bar "
                    f"{stamp} was {what} by the dataset "
                    f"{type(loader.dataset).__name__}; every "
                    f"present cell must be predicted exactly once."
                )
        return out.numpy()

    def _evaluate(self, epoch: int, split: str, panel: TrainingPanel, bars) -> dict[str, float]:
        """Return one split's metrics and write them to the wandb summary.

        ``{split}_loss`` is the mean ``_val_one_batch`` over the split's
        batches, one per bar with the default dataset; the other keys are
        ``_compute_metrics`` on the raw labels.
        """
        pred = self._predict_panel_bars(panel, bars)
        metrics = {
            f"{split}_loss": self._eval_loss(epoch, self._loader(panel, bars, training=False))
        }
        y_raw = panel.y_raw.cpu().numpy()
        for key, value in self._compute_metrics(
            y_raw[bars], pred[bars], split, panel.timestamps[bars]
        ).items():
            metrics[f"{split}_{key}"] = value
        if self._wandb_recorder is not None:
            self._wandb_recorder.summary.update(metrics)
        return metrics

    def _fit(self, checkpoint: Path) -> dict:
        """Build the training panel and target, train until ``_should_stop``, evaluate, save.

        The panel is split by ``_fit_segments``, but every window reads the
        whole collected panel, so the first validation and test bars (and
        the first training bar, through the warm-up) have full windows. The
        training target is computed once, before the first epoch:
        ``_transform_target`` sees ``training=True`` on the training bars
        and ``training=False`` on the validation and test bars. Each epoch
        runs ``_train_one_batch`` on the shuffled training loader, then
        ``_val_one_batch`` on the validation loader when there is a
        validation segment, then ``_test_one_batch`` on the test loader.
        The per-epoch ``train_loss`` / ``val_loss`` are logged to wandb and
        passed to ``_should_stop``; ``_on_fit_start`` runs before the first
        epoch and ``_on_fit_end`` after the last, and the loop never runs
        past ``epochs``. Torch is reseeded with ``config.random_seed`` first,
        so a fit is reproducible on CPU.

        Returns
        -------
        dict
            ``{split}_loss`` and the ``_compute_metrics`` keys for ``train``,
            ``val`` (only with a validation segment) and ``test`` (only when
            it has bars).

        Raises
        ------
        ValueError
            If any of the four ``train_*`` / ``test_*`` dates is unset, or
            ``val_size`` or the purge leaves no timestamps to fit on, or
            ``epochs`` is not a positive integer.
        """
        config = self.config
        epochs = self.epochs
        if not all(
            (config.train_start, config.train_end, config.test_start, config.test_end)
        ):
            raise ValueError(
                "Training and testing start and end dates must be specified."
            )
        self._set_random_seed(config.random_seed)

        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"]).sortby(
            ["timestamp", "symbol"]
        )
        segments = self._fit_segments(data)
        stamps = data.timestamp.values
        train_bars, val_bars, test_bars = [
            np.searchsorted(stamps, part.timestamp.values) for part in segments
        ]

        panel = self._training_panel(data)
        self._fill_target(panel, train_bars, training=True)
        self._fill_target(panel, val_bars, training=False)
        self._fill_target(panel, test_bars, training=False)
        panel = self._place_panel(panel)

        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=config.hyperparameters,
        ).to(self.device)
        self.optim = self._init_optim(self.model)  # type: ignore[arg-type]
        self._on_fit_start()
        train_loader = self._loader(panel, train_bars, training=True)
        val_loader = self._loader(panel, val_bars, training=False) if len(val_bars) else None
        test_loader = self._loader(panel, test_bars, training=False) if len(test_bars) else None

        epoch = 0
        for epoch in tqdm(range(epochs), desc=f"{self.class_name}_train"):
            train_loss = self._train_epoch(epoch, train_loader)
            val_loss = self._eval_loss(epoch, val_loader) if val_loader is not None else None
            if test_loader is not None:
                self._test_epoch(epoch, test_loader)
            if self._wandb_recorder is not None:
                logged = {"train_loss": train_loss}
                if val_loss is not None:
                    logged["val_loss"] = val_loss
                self._wandb_recorder.log(logged, step=epoch)
            if self._should_stop(epoch, train_loss, val_loss):
                logger.info(f"{self.class_name}: stopping after epoch {epoch}")
                break
        self._on_fit_end()

        with Timer(f"{self.class_name}: evaluate"):
            metrics = self._evaluate(epoch, "train", panel, train_bars)
            for split, bars in (("val", val_bars), ("test", test_bars)):
                if len(bars):
                    metrics.update(self._evaluate(epoch, split, panel, bars))

        self._save_model(checkpoint)
        if self._wandb_recorder is not None:
            self._wandb_recorder.finish()
        self.optim = None
        return metrics

    def _predict(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Return ``[T, S, L]`` predictions for a ``[T, S, F]`` input.

        Bar ``t`` is predicted from the windows ending at ``t`` inside
        ``data`` itself, so the first N - 1 bars have NaN rows for the
        missing history, which ``_transform_feature`` handles. The input
        becomes a ``TrainingPanel`` without labels, fed through the head's
        ``training=False`` dataset and loader. A cell outside its bar's
        cross-section is NaN.

        Raises
        ------
        TypeError
            If ``data`` is neither a tensor nor an array.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")
        num_times, num_symbols = data.shape[:2]
        return torch.from_numpy(
            self._predict_array(data, np.arange(num_times), np.arange(num_symbols))
        )

    def _predict_array(
        self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Return ``[T, S, L]`` predictions of every bar of ``x`` on its coordinates."""
        panel = self._place_panel(
            TrainingPanel.from_arrays(x, timestamps=timestamps, symbols=symbols)
        )
        return self._predict_panel_bars(panel, np.arange(x.shape[0]))

    def _predict_panel_array(
        self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Return the ``[T, S, L]`` predictions of ``x`` on its own coordinates."""
        return self._predict_array(x, timestamps, symbols)

    def _write_checkpoint(self, path: Path) -> None:
        """Save the network's ``state_dict`` to ``path`` with ``torch.save``."""
        torch.save(self.model.state_dict(), path)  # type: ignore[union-attr]

    def _read_checkpoint(self, path: Path) -> None:
        """Rebuild the network and load the ``state_dict`` stored at ``path``.

        The network depends only on the feature and label counts, so no data
        needs to be collected first.
        """
        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.config.hyperparameters,
        ).to(self.device)
        self.model.load_state_dict(  # type: ignore[union-attr]
            torch.load(path, map_location=self.device)
        )


class LibraryModel(BaseModel):
    """Numpy variant for tree models and other non-torch libraries.

    There is no epoch loop and no copy-based rollback. Training, early
    stopping and the choice of the best model are left to the library's own
    mechanism inside ``_fit_model``: boosting libraries decide early stopping
    per round with cached validation scores, and rolling back to the best
    round is a matter of keeping the first ``k`` trees, both of which an
    outer epoch loop would only make coarser and slower.

    The base builds the rows. The training target is computed once per fit,
    one bar at a time, by ``_transform_target`` (``training=True`` on the
    training bars only), exactly as for a torch head. Only cells with a valid
    training target become rows; NaN features are kept, so the library's own
    missing-value handling decides what they mean. Each row carries its cell
    in ``where`` (see ``Rows``).

    A head implements three hooks: ``_init_model``, ``_fit_model`` and
    ``_forward``. ``_transform_feature`` (inf to NaN), ``_transform_target``
    (the raw label), ``_loss`` (MSE), ``_resolved_hyperparameters`` and the
    inherited ``_compute_metrics`` have defaults that may be overridden.
    ``{split}_loss`` is ``_loss`` on the training target per bar, averaged
    over bars; the other metrics score the raw first label. Checkpoints are
    ``.joblib`` files written through ``MlBackend``; they are pickles, so only
    load files you trust.

    Examples
    --------
    A minimal head that predicts the first feature for every label::

        >>> class FirstFeatureHead(LibraryModel):
        ...     def _init_model(self, num_features, num_labels, hyperparameters):
        ...         return {"num_labels": num_labels}
        ...     def _fit_model(self, train_rows, val_rows):
        ...         pass
        ...     def _forward(self, x):
        ...         return np.repeat(x[:, :1], self.model["num_labels"], axis=-1)
        >>> head = FirstFeatureHead(ModelConfig(
        ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> head.collect().train().suffix
        '.joblib'
    """

    checkpoint_suffix = ".joblib"
    reserved_hyperparameters = LIBRARY_RESERVED_HYPERPARAMETERS

    @property
    def early_stopping(self) -> bool:
        """``hyperparameters["early_stopping"]``, False when unset.

        Whether the head turns on its library's native early stopping.

        Examples
        --------
        >>> head.early_stopping
        False
        """
        return bool(self.config.hyperparameters.get("early_stopping", False))

    @property
    def early_stopping_patience(self) -> int:
        """``hyperparameters["early_stopping_patience"]``, 5 when unset.

        Rounds (or the library's own unit) without improvement before the
        library's early stopping triggers.

        Examples
        --------
        >>> head.early_stopping_patience
        5
        """
        return int(self.config.hyperparameters.get("early_stopping_patience", 5))

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """Prepare the model for the given shape; the result becomes ``self.model``.

        Tree libraries often build the real model only inside ``_fit_model``,
        in which case this may just resolve the hyperparameters and return
        None. ``load()`` does not call it: the checkpoint holds the whole
        model. ``hyperparameters`` holds the reserved keys too; pass it
        through ``self.head_hyperparameters`` before handing it to a library.
        """

    @abstractmethod
    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Fit the model on the training rows.

        ``val_rows`` is None when there is no validation segment
        (``val_size == 0``) or it has no cell with a valid training target.
        Early stopping and rollback to the best model are the hook's job,
        using the library's native mechanism and honouring
        ``early_stopping`` and ``early_stopping_patience``. On return
        ``self.model`` must be the model to save.
        """

    @abstractmethod
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``[n, L]`` predictions for ``[n, F]`` rows from ``_transform_feature``."""

    def _transform_feature(self, x: np.ndarray) -> np.ndarray:
        """Turn raw ``[n, F]`` float32 feature rows into the library's input.

        The result must keep the shape. The default returns a copy with
        infinities replaced by NaN, which tree libraries read as missing; a
        library that refuses NaN overrides this to impute. Applied to the
        training and validation rows and at prediction alike; never modify
        ``x`` in place.
        """
        return np.where(np.isinf(x), np.float32(np.nan), x)

    def _resolved_hyperparameters(self) -> dict | None:
        """Return the hyperparameters actually in effect, or None to record nothing.

        A head that merges user overrides into library defaults in
        ``_init_model`` overrides this to expose the merged result. When not
        None, ``_fit`` adds it to the wandb run config and ``get_config`` adds
        it to ``config.json`` as ``resolved_hyperparameters``, so a run stays
        reproducible after defaults change. It is a record, not an input:
        ``config.hyperparameters`` is left as the user wrote it, and the
        config loader drops the key when rebuilding.
        """
        return None

    def get_config(self) -> dict:
        """Return ``BaseModel.get_config()`` plus any resolved hyperparameters.

        Examples
        --------
        >>> "resolved_hyperparameters" in head.get_config()
        False
        """
        cfg = super().get_config()
        resolved = self._resolved_hyperparameters()
        if resolved is not None:
            cfg["resolved_hyperparameters"] = dict(resolved)
        return cfg

    def _loss(self, target: np.ndarray, pred: np.ndarray) -> float:
        """Return the MSE over every label of one bar's ``[n, L]`` rows.

        ``target`` is the bar's training target, finite everywhere. Returns
        NaN when there are no rows.
        """
        target = np.asarray(target, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if target.size == 0:
            return float("nan")
        diff = pred - target
        return float(np.sum(diff * diff) / diff.size)

    def _features(self, x: np.ndarray) -> np.ndarray:
        """Return ``_transform_feature(x)`` after checking it kept the shape."""
        out = np.asarray(self._transform_feature(x))
        if out.shape != x.shape:
            raise ValueError(
                f"{self.class_name}._transform_feature must keep the shape "
                f"{x.shape}, got {out.shape}"
            )
        return out

    def _forward_rows(self, x: np.ndarray) -> np.ndarray:
        """Run ``_forward`` on raw ``[n, F]`` rows and check it returns ``[n, L]``."""
        if len(x) == 0:
            return np.empty((0, self.num_labels), dtype=np.float32)
        pred = np.asarray(self._forward(self._features(x)))
        if pred.shape != (len(x), self.num_labels):
            raise ValueError(
                f"{self.class_name}._forward must return [n, L] = "
                f"{[len(x), self.num_labels]} predictions, got {list(pred.shape)}"
            )
        return pred

    def _rows(self, panel: TrainingPanel, bars) -> Rows:
        """Return the rows of ``bars``: every cell with a valid training target."""
        cells = panel.mask & self._bar_mask(panel, bars)[:, None]
        t, s = torch.nonzero(cells, as_tuple=True)
        return Rows(
            x=self._features(panel.x[t, s].numpy()),
            y=panel.target[t, s].numpy(),
            y_raw=panel.y_raw[t, s].numpy(),
            where=(t.numpy(), s.numpy()),
        )

    @staticmethod
    def _bar_mask(panel: TrainingPanel, bars) -> torch.Tensor:
        """Return ``[T]`` booleans, True on ``bars``."""
        chosen = torch.zeros(panel.present.shape[0], dtype=torch.bool)
        chosen[torch.as_tensor(np.asarray(bars, dtype=np.int64))] = True
        return chosen

    def _evaluate(self, split: str, panel: TrainingPanel, bars) -> dict[str, float]:
        """Evaluate one split and write the prefixed metrics to the wandb summary.

        Every present cell of ``bars`` is predicted. ``{split}_loss`` is
        ``_loss`` on each bar's training target, averaged over the bars
        that have one, so every bar weighs the same; the other keys are
        ``_compute_metrics`` on the raw labels. They go to the run summary
        (final values, no step), so they do not interfere with per-round
        ``log(step=...)`` curves.
        """
        num_times, num_symbols = panel.present.shape
        pred = np.full((num_times, num_symbols, self.num_labels), np.nan, dtype=np.float64)
        cells = panel.present & self._bar_mask(panel, bars)[:, None]
        t, s = torch.nonzero(cells, as_tuple=True)
        pred[t.numpy(), s.numpy()] = self._forward_rows(panel.x[t, s].numpy())

        target = panel.target.numpy()
        mask = panel.mask.numpy()
        losses = [
            self._loss(target[bar, mask[bar]], pred[bar, mask[bar]])
            for bar in bars
            if mask[bar].any()
        ]
        metrics = {f"{split}_loss": float(np.mean(losses)) if losses else float("nan")}
        y_raw = panel.y_raw.numpy()
        for key, value in self._compute_metrics(
            y_raw[bars], pred[bars], split, panel.timestamps[bars]
        ).items():
            metrics[f"{split}_{key}"] = value
        if self._wandb_recorder is not None:
            self._wandb_recorder.summary.update(metrics)
        return metrics

    def _fit(self, checkpoint: Path) -> dict:
        """Build the rows, fit once with ``_fit_model``, evaluate, save and finish the run.

        The validation segment is the trailing ``val_size`` share of the
        training window, and the purge of ``_fit_segments`` drops the last L
        bars before validation and before test, as in the torch variant.
        The training target is computed once, before the fit:
        ``_transform_target`` sees ``training=True`` on the training bars and
        ``training=False`` on the validation and test bars. Empty splits skip
        evaluation: no validation segment means no ``val_*`` metrics, and an
        empty test segment no ``test_*`` metrics.

        Returns
        -------
        dict
            The ``train_*``, ``val_*`` and ``test_*`` metrics, with the
            values ``_evaluate`` wrote to the wandb summary.

        Raises
        ------
        ValueError
            If any of the four ``train_*`` / ``test_*`` dates is
            unset, ``val_size`` or the purge leaves no timestamps to fit
            on, or no training cell has a valid training target.
        """
        config = self.config
        if not all(
            (config.train_start, config.train_end, config.test_start, config.test_end)
        ):
            raise ValueError(
                "Training and testing start and end dates must be specified."
            )

        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"]).sortby(
            ["timestamp", "symbol"]
        )
        segments = self._fit_segments(data)
        stamps = data.timestamp.values
        train_bars, val_bars, test_bars = [
            np.searchsorted(stamps, part.timestamp.values) for part in segments
        ]
        panel = self._training_panel(data)
        self._fill_target(panel, train_bars, training=True)
        self._fill_target(panel, val_bars, training=False)
        self._fill_target(panel, test_bars, training=False)

        train_rows = self._rows(panel, train_bars)
        if len(train_rows.x) == 0:
            raise ValueError(
                f"{self.class_name}: the training segment has no cell with a "
                f"valid training target."
            )
        val_rows = self._rows(panel, val_bars) if len(val_bars) else None
        if val_rows is not None and len(val_rows.x) == 0:
            logger.warning(
                f"{self.class_name}: the validation segment has no cell with a "
                f"valid training target; training without a validation set."
            )
            val_rows = None

        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=config.hyperparameters,
        )
        resolved = self._resolved_hyperparameters()
        if resolved is not None and self._wandb_recorder is not None:
            # The run was opened in `_init_wandb`, before the hyperparameters
            # were resolved; record them now.
            self._wandb_recorder.config.update(
                {"resolved_hyperparameters": dict(resolved)},
                allow_val_change=True,
            )

        with Timer(f"{self.class_name}: fit_model"):
            self._fit_model(train_rows, val_rows)

        with Timer(f"{self.class_name}: evaluate"):
            metrics = self._evaluate("train", panel, train_bars)
            for split, bars in (("val", val_bars), ("test", test_bars)):
                if len(bars):
                    metrics.update(self._evaluate(split, panel, bars))

        self._save_model(checkpoint)
        if self._wandb_recorder is not None:
            self._wandb_recorder.finish()
        return metrics

    def _predict(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        """Return ``[T, S, L]`` predictions for a ``[T, S, F]`` input.

        Every cell with a finite feature is a row for ``_forward``; the
        others are NaN. Tensors are converted to numpy. A floating input
        keeps its dtype, and the predictions are float64.

        Raises
        ------
        TypeError
            If ``data`` is neither a tensor nor an array.
        ValueError
            If ``_transform_feature`` changes the shape of the rows or
            ``_forward`` does not return ``[n, L]``.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")
        x = data if np.issubdtype(data.dtype, np.floating) else data.astype(np.float64)
        present = np.isfinite(x).any(axis=-1)
        out = np.full(x.shape[:-1] + (self.num_labels,), np.nan, dtype=np.float64)
        out[present] = self._forward_rows(x[present])
        return out

    def _predict_panel_array(
        self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Return ``predict(x)`` as an array."""
        return np.asarray(self.predict(x))

    def _write_checkpoint(self, path: Path) -> None:
        """Serialize ``self.model`` to ``path`` through ``MlBackend``."""
        MlBackend().to_internal(self.model).write(str(path))

    def _read_checkpoint(self, path: Path) -> None:
        """Load the whole model from ``path`` through ``MlBackend``.

        ``_init_model`` is not called: the file holds the complete model, and
        rebuilding an empty one first would require collecting data to know
        the feature count.
        """
        self.model = MlBackend().read(str(path)).get_model()
